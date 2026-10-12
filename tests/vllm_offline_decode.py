"""Phase 2 / E0–E5: vLLM offline (LLM class) on the Phase 1 workload and harness.

  /workspace/venv-vllm/bin/python -m tests.vllm_offline_decode --max_num_seqs 64
  ... --enforce_eager            # E2
  ... --max_num_seqs 8 --batch_size 8   # E3 sweep point
"""
from __future__ import annotations

import argparse
import signal
import time

import torch   # version only. Do NOT touch torch.cuda in this process (see gotchas).
import vllm
from vllm import LLM, SamplingParams

from common.bench import Bench
from common.workload import build


def _exit_on_signal(signum, frame):
    raise SystemExit(128 + signum)   # unwinds → finally → recorder.stop()


class VLLMOfflineBench(Bench):
    phase = "vllm"

    def load(self) -> None:
        a = self.args
        self.llm = LLM(
            model=a.model_name,
            dtype="bfloat16",
            seed=0,
            gpu_memory_utilization=a.gpu_mem_util,
            max_model_len=a.isl + a.osl,   # TODO(you): predict the "max concurrency" log line first
            max_num_seqs=a.max_num_seqs,
            max_num_batched_tokens=a.max_num_batched_tokens,   # None → vLLM default
            enable_prefix_caching=a.prefix_caching,
            enforce_eager=a.enforce_eager,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.params = SamplingParams(
            temperature=0.0,      # greedy, as in Phase 1
            max_tokens=a.osl,
            ignore_eos=True,      # same effect as Phase 1's min_new_tokens=osl
        )

    @staticmethod
    def _prompts(batch: list[list[int]]) -> list[dict]:
        return [{"prompt_token_ids": ids} for ids in batch]

    def warmup(self, batch: list[list[int]]) -> None:
        # TODO(you): vLLM captures CUDA graphs at startup. Is a warmup still needed?
        #   Decide from data: time batch 0 vs batch 1 with and without this call.
        short = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)
        self.llm.generate(self._prompts(batch), short, use_tqdm=False)

    def run_batch(self, batch: list[list[int]]) -> dict:
        n, osl = len(batch), self.args.osl
        t_start = time.time()
        outs = self.llm.generate(self._prompts(batch), self.params, use_tqdm=False)
        t_end = time.time()   # generate() blocks until every request is done: no sync needed

        # TODO(you): sanity checks. Every len(o.prompt_token_ids) == isl and every
        #   len(o.outputs[0].token_ids) == osl. Fail loudly, otherwise tok/s is a lie.

        # TODO(you): TTFT / ITL / decode-only tok/s. generate() only gives you end-to-end.
        #   Pick one and write down why:
        #   (a) o.metrics, if your version fills it in (often None in V1; check),
        #   (b) drive the engine yourself: llm.llm_engine.add_request(...) then a
        #       while-loop over llm.llm_engine.step(), timestamping each step.
        #       That gives one timestamp per engine step, the closest match to Phase 1's TokenClock,
        #   (c) leave latency to the online client and report throughput only here.
        t_first_token = ...

        return {
            "batch_size": n,
            "t_start": t_start,
            "t_first_token": t_first_token,
            "t_end": t_end,
            "batch_latency_s": t_end - t_start,
            "generated_tokens_per_second": n * osl / (t_end - t_start),
            # TODO(you): ttft_s, decode_s, decode_tokens_per_second, itl_median_ms, itl_p99_ms
        }

    def engine_meta(self) -> dict:
        a = self.args
        return {
            "engine": "vllm-offline",
            "vllm_version": vllm.__version__,
            "torch_version": torch.__version__,
            "model": a.model_name,
            "gpu_memory_utilization": a.gpu_mem_util,
            "max_model_len": a.isl + a.osl,
            "max_num_seqs": a.max_num_seqs,
            "max_num_batched_tokens": a.max_num_batched_tokens,
            "enable_prefix_caching": a.prefix_caching,
            "enforce_eager": a.enforce_eager,
            # TODO(you): KV capacity (num GPU blocks, tokens), block size, attention backend,
            #   CUDA graph capture sizes. Find where these live under self.llm.llm_engine
            #   (vllm_config / cache_config), or parse them out of the startup log.
        }


def main():
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _exit_on_signal)

    ap = argparse.ArgumentParser(description="vLLM offline decode (Phase 2)")
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--prompts_path", default="configs/workloads/prompts.txt")
    ap.add_argument("--isl", type=int, default=128)
    ap.add_argument("--osl", type=int, default=512)
    # batch_size = prompts per generate() call (what's SUBMITTED).
    # max_num_seqs = what the scheduler RUNS at once. Equal for E1; they diverge in E4/E5.
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_num_seqs", type=int, default=64)
    ap.add_argument("--max_num_batched_tokens", type=int, default=None)
    ap.add_argument("--gpu_mem_util", type=float, default=0.90)
    ap.add_argument("--prefix_caching", action="store_true")
    ap.add_argument("--enforce_eager", action="store_true")
    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument("--exp", default=None)
    args = ap.parse_args()

    exp = args.exp or f"offline-seqs{args.max_num_seqs}" + ("-eager" if args.enforce_eager else "")

    bench = VLLMOfflineBench(args)
    bench.load()
    wl = build(args.prompts_path, bench.tokenizer, args.isl, args.osl)
    bench.main_loop(wl, exp)


if __name__ == "__main__":   # REQUIRED: vLLM spawns the EngineCore process
    main()
