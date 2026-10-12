from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from statistics import median, quantiles

import pandas as pd

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer

from common import recorder
from common.run import new_run, finish_run

# RTX 3090 DRAM peak, GA102 whitepaper Appendix A. Stand-in until peaks.py exists. VERIFY.
MEM_BW_PEAK = 936e9
# A SOL sample stamped t_server=T covers roughly the ~0.75-1 s BEFORE T (measured in Phase 0).
SOL_LAG_S = 1.0


def _exit_on_signal(signum, frame):
    raise SystemExit(128 + signum)   # unwinds the stack → finally runs → recorder.stop()

class CustomTextStreamer(BaseStreamer):
    def __init__(self, tokenizer:AutoTokenizer):
        self.tokenizer = tokenizer

    def put(self, value:torch.Tensor):
        if len(value.shape) > 1:
            value = value[0]

        text = self.tokenizer.decode(value, skip_special_tokens=True)
        print(text, end="", flush=True)

    def end(self):
        print("\n[Text] End of stream.")

class TokenTimeTrackerStreamer(BaseStreamer):
    def __init__(self):

        self.steps = []
        self.last_time = None

    def put(self, value:torch.Tensor):

        current_time = time.perf_counter()

        if self.last_time is not None:
            elapsed = current_time - self.last_time
            self.steps.append(elapsed)
            print(f"  [Time] Step {len(self.steps)} took {elapsed:.4f} seconds")

        self.last_time = current_time

    def end(self):
        """Called when generation finishes."""
        print(f"\n[Time] Generation complete over {len(self.steps)} steps.")
        if self.steps:
            avg_time = sum(self.steps) / len(self.steps)
            print(f"[Time] Average step latency: {avg_time:.4f} seconds")
        self.last_time = None  # Reset for subsequent calls

class CompositeStreamer(BaseStreamer):
    def __init__(self, streamers):
        self.streamers = streamers

    def put(self, value):
        for streamer in self.streamers:
            # Clone or detach tensor if streamers modify inplace (safety measure)
            streamer.put(value)

    def end(self):
        for streamer in self.streamers:
            streamer.end()

class TokenClock(BaseStreamer):
    """Silent per-step wall clock for measured runs (the printing streamers above are for debugging).

    generate() calls put() once with the prompt, then once per generated token. HF checks its
    stopping criteria on the GPU every step, which syncs the CPU, so these times track the GPU.
    time.time() (not perf_counter) so they line up with utlz.csv's t_server.
    """
    def __init__(self):
        self.times: list[float] = []

    def put(self, value):
        self.times.append(time.time())

    def end(self):
        pass

class HFNativeInference:
    def __init__(self, model_name:str, device:torch.device="cuda", mode:str="eager"):

        self.model = None
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.mode = mode

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Llama has no pad token. Decoder-only models must be LEFT-padded:
            # right padding puts pad tokens between the prompt and the generated text.
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

            self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                ).to(self.device).eval()

            pad_id = (
                self.model.generation_config.pad_token_id
                or self.model.generation_config.eos_token_id
            )

            if pad_id is None:
                raise ValueError(
                    f"Model '{model_name}' has neither a pad token ID "
                    "nor an EOS token ID configured."
                )

            self.pad_id = pad_id[0] if isinstance(pad_id, list) else pad_id

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize model '{model_name}' "
                f"on device '{self.device}'."
            ) from e

        # Static mode does NOT torch.compile model.forward here. Wrapping the whole forward in a
        # CUDA graph made the lazily-allocated static KV cache a graph output, which the next
        # replay overwrote ("accessing tensor output of CUDAGraphs that has been overwritten").
        # generate() with cache_implementation="static" compiles the decode step itself, with
        # the cache allocated outside the graph and prefill left eager.

    def _init_streamer(self):
        writer = CustomTextStreamer(self.tokenizer)
        tracker = TokenTimeTrackerStreamer()

        pipeline_streamer = CompositeStreamer([tracker, writer])
        return pipeline_streamer

    def _init_run(self, run_config:dict, args=None):

        phase = run_config.get("phase", "hf_native_inference")
        exp = run_config.get("exp", "hf-eager")

        run = new_run(phase, exp, args=None if args is None else vars(args))
        rec = recorder.start(run.path("utlz.csv"), client_id=run.run_id)

        return run, rec

    def run_batch(self, batch_text:list[str], osl: int | None = None):

        if not (hasattr(self, "isl") and hasattr(self, "osl")):
            raise AttributeError(f"Object does not has Input Sequence Length and Output Sequence Length Configured")

        inputs = self.tokenizer(
            batch_text, return_tensors="pt", padding=True, truncation=True, max_length=self.isl,
        ).to(self.device)

        osl = osl or self.osl
        clock = TokenClock()

        torch.cuda.synchronize()
        t_start = time.time()        

        with torch.inference_mode(True):
            gen_kwargs = {}
            if self.mode == "static":
                gen_kwargs = {
                    # preallocated [batch, isl+osl] KV → fixed shapes → transformers compiles the
                    # decode step (CUDA graphs). Verify it did: warmup 0 should take minutes.
                    "cache_implementation": "static",
                }
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=osl,
                min_new_tokens=osl,        # suppress EOS → identical work every batch
                do_sample=False,           # greedy; same decoding in every phase
                repetition_penalty=1.0,    # Qwen's generation_config sets 1.05; neutralize it
                pad_token_id=self.pad_id,
                streamer=clock,
                **gen_kwargs
            )

        torch.cuda.synchronize()
        t_end = time.time()

        batch_size = len(batch_text)
        prompt_width = inputs["input_ids"].shape[1]
        generated_tokens = outputs.shape[1] - prompt_width

        # clock.times[0] = prompt pushed (before prefill); [1] = first generated token
        t_first = clock.times[1]
        itl = [b - a for a, b in zip(clock.times[1:], clock.times[2:])]
        decode_s = t_end - t_first

        return {
            "batch_size": batch_size,
            "prompt_width": prompt_width,
            "generated_tokens_per_sequence": generated_tokens,
            "t_start": t_start,
            "t_first_token": t_first,
            "t_end": t_end,
            "batch_latency_s": t_end - t_start,
            "ttft_s": t_first - t_start,
            "decode_s": decode_s,
            "itl_median_ms": 1e3 * median(itl) if itl else None,
            "itl_p99_ms": 1e3 * quantiles(itl, n=100)[98] if len(itl) >= 2 else None,
            "sequences_per_second": batch_size / (t_end - t_start),
            "generated_tokens_per_second": generated_tokens * batch_size / (t_end - t_start),
            # decode only: tokens after the first, over the time after the first token
            "decode_tokens_per_second": (generated_tokens - 1) * batch_size / decode_s,
        }

    def warmup(self, batch_text):
        # Not measured. Eager: one short call absorbs CUDA/cuBLAS init, allocator, clock ramp-up.
        # Static: the KV cache is sized isl+osl, so warmup MUST use the real osl (and the real batch
        # size), or the first measured batch recompiles. Call 0 compiles (minutes), call 1 captures
        # the CUDA graphs; after that every batch should replay with no recompiles.
        calls = 2 if self.mode == "static" else 1
        osl = self.osl if self.mode == "static" else 16
        for i in range(calls):
            t0 = time.time()
            self.run_batch(batch_text, osl=osl)
            print(f"warmup {i} ({self.mode}, osl={osl}): {time.time() - t0:.1f}s", flush=True)

    def bytes_per_decode_step(self, batch_size: int) -> float:
        """Hand estimate of DRAM bytes one decode step must read: weights + KV cache."""
        cfg = self.model.config
        weight_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        emb = self.model.get_input_embeddings().weight
        if not getattr(cfg, "tie_word_embeddings", False):
            # untied: a step reads only `batch_size` rows of the input embedding table
            weight_bytes -= emb.numel() * emb.element_size()

        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        kv_per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * emb.element_size()
        avg_ctx = self.isl + self.osl / 2   # KV cache grows linearly during decode
        return weight_bytes + batch_size * avg_ctx * kv_per_token

    def summarize(self, utlz_csv: Path, rows: list[dict]) -> pd.DataFrame:
        """Per batch size: measured throughput, hand-computed expectation, utlz SOL over decode windows."""
        u = pd.read_csv(utlz_csv)
        b = pd.DataFrame(rows)
        out = []
        for batch_size, calls in b.groupby("batch_size"):
            # utlz samples inside any batch's decode window, trimming the SOL lag at each start
            mask = pd.Series(False, index=u.index)
            for _, c in calls.iterrows():
                mask |= (u.t_server >= c.t_first_token + SOL_LAG_S) & (u.t_server <= c.t_end)
            window = u[mask]
            sol = window[window.sol_valid == 1]

            tok_s = calls.decode_tokens_per_second.median()
            step_s = batch_size / tok_s
            step_bytes = self.bytes_per_decode_step(batch_size)
            out.append({
                "batch_size": batch_size,
                "batches": len(calls),
                "decode_tok_s": tok_s,
                "step_ms": 1e3 * step_s,
                "ceiling_step_ms": 1e3 * step_bytes / MEM_BW_PEAK,   # if DRAM were 100% busy
                "expected_mem_pct": 100 * step_bytes / step_s / MEM_BW_PEAK,
                "memory_sol": sol.memory_sol.mean(),
                "compute_sol": sol.compute_sol.mean(),
                "sm_active": sol.sm_active.mean(),
                "nvml_util": window.nvml_util.mean(),
                "ttft_ms": 1e3 * calls.ttft_s.median(),
                "itl_ms": calls.itl_median_ms.median(),
                "sol_samples": len(sol),
            })
        return pd.DataFrame(out).round(1)

    def infer(self, prompt_fp:str, run_config:dict=None,
            default_isl:int=128,
            default_osl:int=512,
            default_batch:int = 96,
            args=None):

        file_path = Path(prompt_fp)
        if not file_path.exists():
            raise FileNotFoundError(f'Source File: {prompt_fp} not found!')

        with open(file_path, mode="r", encoding="utf-8") as f:
            content = [line.strip() for line in f if line.strip()]   # no trailing "\n"

        self.isl = args.isl if args is not None else default_isl
        self.osl = args.osl if args is not None else default_osl
        self.batch_size = args.batch_size if args is not None else default_batch
        max_batches = getattr(args, "max_batches", None)

        warmed_up = False
        start_idx = 0
        rows : list[dict] = []
        batch_idx = 0
        run, rec = None, None

        try:
            run, rec = self._init_run(run_config, args)

            while start_idx < len(content):
                if max_batches is not None and batch_idx >= max_batches:
                    break                

                end_idx = min(start_idx + self.batch_size, len(content))
                if self.mode == "static" and end_idx - start_idx < self.batch_size:
                    break

                batch_text = content[start_idx:end_idx]
                if not warmed_up:
                    self.warmup(batch_text)
                    warmed_up = True

                batch_stats = self.run_batch(batch_text)
                batch_stats["batch_idx"] = batch_idx
                rows.append(batch_stats)

                print(f"batch {batch_idx}: {batch_stats['decode_tokens_per_second']:.0f} decode tok/s "
                      f"ITL {batch_stats['itl_median_ms']:.1f} ms  TTFT {1e3 * batch_stats['ttft_s']:.0f} ms "
                      f"({batch_stats['batch_latency_s']:.1f}s)", flush=True)

                start_idx = end_idx
                batch_idx += 1

        except Exception as e:
            raise RuntimeError(
                f"Inference failed at batch {batch_idx}, "
                f"prompt index {start_idx}, "
                f"batch size {self.batch_size}"
            ) from e

        finally:
            failed = sys.exc_info()[0] is not None
            if rec is not None:
                recorder.stop(rec)   # first: lets the process (and the ssh session) exit
            if run is not None:
                summary = None
                if rows:
                    pd.DataFrame(rows).to_csv(run.path("bench.csv"), index=False)
                    if run.path("utlz.csv").exists():
                        summary = self.summarize(run.path("utlz.csv"), rows)
                        summary.to_csv(run.path("summary.csv"), index=False)
                        print(summary.to_string(index=False), flush=True)
                finish_run(
                    run,
                    failed=failed,
                    batches_completed=len(rows),
                    model=self.model.config._name_or_path,
                    summary=None if summary is None else summary.to_dict(orient="records"),
                )

def main():

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _exit_on_signal)

    ap = argparse.ArgumentParser(description="HF Native LLM Decode")

    ap.add_argument("--prompts_path", default="configs/workloads/prompts.txt", type=str)
    # 128 < every prompt's token length, so truncation makes every row exactly 128 (no padding).
    ap.add_argument("--isl", default=128, type=int)
    ap.add_argument("--osl", default=512, type=int)
    ap.add_argument("--batch_size", default=64, type=int)
    ap.add_argument("--max_batches", default=None, type=int, help="stop early (debugging)")
    ap.add_argument("--mode", choices=["eager", "static"], default="eager",
                    help="eager: dynamic KV cache. static: static KV cache + torch.compile (CUDA graphs)")
    ap.add_argument("--exp", default=None, help="run-dir name (default: hf-eager / hf-static-compile)")

    ap.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")

    args = ap.parse_args()

    run_config = {
        "phase": f"hf_native_inference-{args.model_name}",
        "exp": args.exp or {"eager": "hf-eager", "static": "hf-static-compile"}[args.mode],
    }

    native_inference = HFNativeInference(model_name=args.model_name, mode=args.mode)
    native_inference.infer(
        prompt_fp=args.prompts_path,
        default_isl=args.isl,
        default_osl=args.osl,
        default_batch=args.batch_size,
        run_config=run_config,
        args=args
    )

if __name__ == "__main__":
    main()
