# utlz-lab

Measuring how well LLM inference actually uses the GPU, from a single GEMM up to a served model.
Every experiment runs under [Utilyze](https://github.com/systalyze/utilyze/tree/main) (`utlz`),
which reports speed-of-light (SOL) compute and memory utilization from GPU hardware counters,
next to the usual throughput and latency numbers.

Hardware so far: one RTX 3090 (24 GB) on a Vast.ai VM, driven from a Mac with
[runway](https://github.com/shiv-vignesh/runway), my tool for remote GPU work: push code, set up
the venv, run tasks (`runway-descriptors/*.toml`), follow logs, pull results. Box setup: [ONBOARDING.md](ONBOARDING.md).

> **[runway](https://github.com/shiv-vignesh/runway)**: mirrors a local project folder to a remote machine over rsync, sets up an environment there, and runs commands on it, from a terminal UI or a plain CLI.

## What's covered

| Phase | Test | Code | What it measures |
|---|---|---|---|
| P0 | Large NxN GEMM | `tests/matmul.py` | 8192×8192 FP16 matmul for 30 s between idle windows: sanity check that utlz counters read real load |
| P1 | HF native inference | `tests/hf_native_llm_decode.py` | `transformers.generate()` decode, eager vs static KV cache + `torch.compile`, Qwen2.5-7B and Llama-3.1-8B, ISL 128 / OSL 512, batch 64 |
| P2 | vLLM deployed server | `server/`, `client/` | Own OpenAI-compatible server on vLLM's `AsyncLLM` (box) + streaming load client (Mac); client vs server vs network TTFT, ITL, goodput, SOL per load point |

Every run writes the same run dir: `meta.json` (args, GPU snapshot, versions), `pip_freeze.txt`,
`utlz.csv` (SOL sampled every 250 ms), and `bench.csv` / `summary.csv` or `server_requests.csv`.

## Results

**P0, GEMM** (`results/p0/matmul-test/20261007-054613`)

| Phase | TFLOPS | Compute SOL % | Memory SOL % | SM active % | NVML util % |
|---|---|---|---|---|---|
| idle | – | 0.0 | 0.0 | 0.0 | 0.0 |
| load (8192² FP16) | 69.7 | 49.7 | 37.7 | 99.8 | 98.3 |

**P1, HF native decode**, batch 64. Full table, column definitions and analysis:
[docs/phase1-hf-results.md](docs/phase1-hf-results.md)

| Model | Mode | Decode tok/s | Step ms | Ceiling step ms | Memory SOL % | Compute SOL % |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | Eager | 894 | 71.6 | 16.6 | 59.1 | 24.1 |
| Qwen2.5-7B | Static + compile | 817 | 78.4 | 16.6 | 70.7 | 16.9 |
| Llama-3.1-8B | Eager | 660 | 97.0 | 19.5 | 57.7 | 23.7 |
| Llama-3.1-8B | Static + compile | CUDA OOM (static KV 5 GiB + allocator fragmentation) | | | | |

HF decode runs 4–5× slower than the DRAM-bandwidth floor ("ceiling step"), and NVML says the GPU is
~95% "utilized" throughout. That gap is what P2 is about.

**P2, vLLM server**: harness built and tested locally against a fake engine; GPU results pending.
Plan and experiments: [docs/phase2-vllm-primer.md](docs/phase2-vllm-primer.md) §2.

## Layout

```
tests/        P0 matmul, P1 HF decode
server/       P2 box side: AsyncLLM + OpenAI routes, timing, lab admin (/admin/runs)
client/       P2 Mac side: streaming requests, open/closed-loop load, report
common/       run dirs (run.py), utlz WebSocket recorder (recorder.py), workload (prompts -> exact-ISL ids)
configs/      workloads/prompts.txt (1200 prompts, varied openings so prefix caching can't help)
scripts/      counter check, GPU snapshot, clock locking
runway-descriptors/   one .toml per runnable task
docs/         results and design notes
```

## Running

Box setup, utlz server and tunnel: [ONBOARDING.md](ONBOARDING.md). `rw` is an alias for
[runway](https://github.com/shiv-vignesh/runway) pointed at this repo.

```bash
rw task p0-matmul                 # P0
rw task "HF Native Inference"     # P1 eager (see runway-descriptors/ for static)
rw task p2-server                 # P2 server on the box (needs /workspace/venv-vllm)

# on the Mac, through ssh -L 8000:127.0.0.1:8000
python -m client chat "hello"
python -m client sweep --rates 1 2 4 8 inf --users 16 64 -n 200
rw pull
python -m client report results/vllm-client/<run_id> --server-dir results/vllm-serve/<exp>/<ts>
```

## Docs

- [docs/phase1-hf-results.md](docs/phase1-hf-results.md): P1 results, Qwen vs Llama, the Llama OOM
- [docs/phase2-vllm-primer.md](docs/phase2-vllm-primer.md): what vLLM solves and how, V1 architecture, experiments E0–E9
- [docs/phase2-server-client-design.md](docs/phase2-server-client-design.md): server/client design, what lives where, mapped to vLLM's examples
- [docs/vllm-code-reading-learning-plan.md](docs/vllm-code-reading-learning-plan.md): how to read vLLM's codebase
