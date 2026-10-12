# Phase 1 — HF transformers: eager vs static + compile

Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct, BF16, RTX 3090, greedy, ISL 128, OSL 512, batch 64,
18 full batches each. torch 2.14.1+cu130, driver 580.95.05.

| Model | Mode | Run | tok/s (end-to-end) | Batch latency (s) | Decode tok/s | Step ms | Ceiling step ms | Expected mem % | Memory SOL % | Compute SOL % | SM active % | NVML util % | TTFT ms | ITL ms | ITL p99 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | Eager | `hf_native_inference/hf-eager/20261011-002600` | 856.6 | 38.3 | 894.3 | 71.6 | 16.6 | 23.2 | 59.1 | 24.1 | 83.0 | 94.5 | 1685 | 69.9 | 97.6 |
| Qwen2.5-7B | Static + compile | `hf_native_inference/hf-static-compile/20261010-195834` | 783.6 | 41.8 | 816.6 | 78.4 | 16.6 | 21.2 | 70.7 | 16.9 | 89.9 | 96.9 | 1769 | 78.3 | 79.5 |
| Llama-3.1-8B | Eager | `hf_native_inference-meta-llama/Llama-3.1-8B-Instruct/hf-eager/20261011-010915` | 637.5 | 51.4 | 660.0 | 97.0 | 19.5 | 20.1 | 57.7 | 23.7 | 80.6 | 95.7 | 1846 | 96.8 | 131.3 |
| Llama-3.1-8B | Static + compile | `hf_native_inference-meta-llama/Llama-3.1-8B-Instruct/hf-static-compile/20261011-015202` | **failed: CUDA OOM in batch 0** (see below) | | | | | | | | | | | | |

Columns:

| Column | Meaning |
|---|---|
| tok/s (end-to-end) | Tokens per second, **summed over all sequences** in the batch: (batch size × OSL generated tokens) ÷ full `generate()` time (prefill + decode). Per sequence: ÷ batch size (here 64 → ≈ 13 tok/s) |
| Batch latency (s) | Wall time of one `generate()` call for the whole batch |
| Decode tok/s | Same aggregate as above, but decode only: tokens after the first ÷ time from first token to end |
| Step ms | Time for one decode step, i.e. one new token for every sequence = batch size ÷ decode tok/s |
| Ceiling step ms | Fastest possible step if DRAM ran at peak: (weight bytes + KV bytes) ÷ peak bandwidth |
| Expected mem % | Bandwidth the step *must* use: (weight + KV bytes per step) ÷ step time ÷ peak |
| Memory SOL % | utlz: max of DRAM throughput and L1 load/store traffic, as % of hardware peak |
| Compute SOL % | utlz: max over SM pipes (tensor, FMA, ALU, LSU, issue), as % of each pipe's peak |
| SM active % | utlz: % of time SMs had at least one warp resident |
| NVML util % | `nvidia-smi` GPU-Util: % of time *any* kernel was running (says nothing about how busy) |
| TTFT ms | Time to first token: `generate()` start → first new token (≈ prefill) |
| ITL ms / ITL p99 ms | Inter-token latency: median / 99th-percentile gap between consecutive generated tokens |

Notes:
- Each mode's `summary.csv` (batch-64 row); end-to-end tok/s, batch latency and ITL p99 are medians over `bench.csv`.
- utlz columns: averaged over each batch's decode window (first token → end, first 1 s trimmed).
- Ceiling/expected use weights + KV bytes at 936 GB/s (unverified peak). Each eager run's last batch had 48 prompts
  and is excluded here (it is a separate row in its `summary.csv`).
- Static runs drop the short last batch (fixed shapes), hence 18 batches.

## Qwen vs Llama (eager)

| | Qwen2.5-7B | Llama-3.1-8B | Llama / Qwen |
|---|---|---|---|
| Layers / KV heads / head dim | 28 / 4 / 128 | 32 / 8 / 128 | |
| KV bytes per token (BF16) | 56 KiB | 128 KiB | 2.3× |
| Ceiling step ms | 16.6 | 19.5 | 1.17× |
| Measured step ms | 71.6 | 97.0 | 1.35× |
| Step ÷ ceiling | 4.3× | 5.0× | |

Llama's step got slower by more than its extra bytes explain (1.35× vs 1.17×), so the extra cost is
mostly not DRAM traffic. Open question for P2: how much of it is per-layer overhead (32 vs 28 layers of
kernel launches) vs bigger GEMMs (hidden 4096 vs 3584)? A trace with and without CUDA graphs answers it.

## Llama static + compile: CUDA OOM

Job log: `logs/runway-jobs/runway-hf-20261010-215148.log`. Died ~114 s in, during batch 0 (warmup), 0 batches done:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 128.00 MiB. GPU 0 has a total capacity of
23.56 GiB of which 82.00 MiB is free. ... Of the allocated memory 20.97 GiB is allocated by PyTorch, with
744.00 MiB allocated in private pools (e.g., CUDA Graphs), and 2.17 GiB is reserved by PyTorch but unallocated.
```

Why Llama runs out and Qwen doesn't. The static cache is preallocated at `batch × (ISL + OSL)` = 64 × 640 = 40,960 tokens:

| | Weights | Static KV (40,960 tok) | Sum |
|---|---|---|---|
| Qwen2.5-7B | ~14.2 GiB | 40,960 × 56 KiB ≈ 2.2 GiB | ~16.4 GiB |
| Llama-3.1-8B | ~15.0 GiB | 40,960 × 128 KiB = 5.0 GiB | ~20.0 GiB |

Llama's ~20 GiB of weights + KV plus the CUDA graph pools (0.74 GiB) and activations reach the 20.97 GiB
allocated. The **2.17 GiB "reserved but unallocated"** is the caching allocator's own fragmentation, and it
pushed the run over 23.56 GiB. That is the same problem vLLM avoids by sizing one KV pool up front.

Not re-run. Options for a Llama static row: batch 48, or `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
(reduces the fragmented 2.17 GiB; may or may not be enough).
