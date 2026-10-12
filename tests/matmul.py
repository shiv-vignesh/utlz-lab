from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import pandas as pd
import torch

from common import recorder
from common.run import new_run, finish_run

DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bfp16": torch.bfloat16
}

SOL_LAG_S = 0.5
BENCH_COLUMNS = ["phase", "t_start", "t_end", "iters", "gpu_s", "tflops"]

def idle(seconds:float) -> dict:
    t_start = time.time()
    time.sleep(seconds)
    return {
        "t_start": t_start,
        "t_end": time.time()
    }

def gemm_load(n:int, dtype: torch.dtype, seconds: float, batch: int = 10) -> dict:
    a = torch.rand(n, n, device="cuda", dtype=dtype)
    b = torch.rand(n, n, device="cuda", dtype=dtype)
    c = torch.rand(n, n, device="cuda", dtype=dtype)

    for _ in range(5):
        torch.matmul(a, b, out=c)
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    iters = 0

    t_start = time.time()
    start_ev.record()

    while time.time() - t_start < seconds:
        for _ in range(batch):
            torch.matmul(a, b, out=c)
        torch.cuda.synchronize()
        iters += batch

    end_ev.record()
    end_ev.synchronize()
    t_end = time.time()

    gpu_s = start_ev.elapsed_time(end_ev) / 1e3  # elapsed_time is in ms
    tflops = 2 * n**3 * iters / gpu_s / 1e12      # 2N^3 FLOPs per square matmul
    return {"t_start": t_start, "t_end": t_end, "iters": iters, "gpu_s": gpu_s, "tflops": tflops}

def write_bench(path: Path, phases: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BENCH_COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(phases)


def summarize(utlz_csv: Path, phases: list[dict]) -> pd.DataFrame:
    u = pd.read_csv(utlz_csv)
    rows = []
    for p in phases:
        window = u[(u.t_server >= p["t_start"] + SOL_LAG_S) & (u.t_server <= p["t_end"])]
        sol = window[window.sol_valid == 1]
        rows.append({
            "phase": p["phase"],
            "sol_samples": len(sol),
            "compute_sol": sol.compute_sol.mean(),
            "memory_sol": sol.memory_sol.mean(),
            "sm_active": sol.sm_active.mean(),
            "nvml_util": window.nvml_util.mean(),
            "tflops": p.get("tflops"),
        })
    return pd.DataFrame(rows).round(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="One GEMM under utlz, end to end")
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--dtype", choices=list(DTYPES), default="fp16")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--idle", type=float, default=5)
    args = ap.parse_args()

    run = new_run("p0", "matmul-test", vars(args))
    rec = recorder.start(run.path("utlz.csv"), client_id=run.run_id)
    phases: list[dict] = []
    try:
        phases.append({"phase": "idle_before", **idle(args.idle)})
        phases.append({"phase": "load", **gemm_load(args.n, DTYPES[args.dtype], args.seconds)})
        phases.append({"phase": "idle_after", **idle(args.idle)})
    finally:
        recorder.stop(rec)
        write_bench(run.path("bench.csv"), phases)
        summary = summarize(run.path("utlz.csv"), phases)
        print(summary.to_string(index=False))
        finish_run(run, phases=phases, summary=summary.to_dict(orient="records"))


if __name__ == "__main__":
    main()