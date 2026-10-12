"""
python -m client chat "hello"
python -m client sweep --rates 1 2 4 inf -n 200 [--users 8 16] [--no-lab]
python -m client report results/vllm-client/<run_id> [--server-dir results/vllm-serve/<exp>/<ts>]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from client.config import ClientConfig
from client.load import closed_loop, open_loop, wait_idle
from client.report import report, summarize
from client.request import chat_once, one_request
from client.transport import admin_client, openai_client
from common.workload import build


def jsonable(args) -> dict:
    # JSON has no inf: --rates inf -> "inf"
    fix = lambda v: str(v) if isinstance(v, float) and math.isinf(v) else v
    return {k: [fix(x) for x in v] if isinstance(v, list) else fix(v) for k, v in vars(args).items()}


def load_workload(cfg: ClientConfig, args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer)
    return build(args.prompts_path, tok, args.isl, args.osl)


async def sweep(cfg: ClientConfig, args) -> None:
    wl = load_workload(cfg, args)
    client, admin = openai_client(cfg), admin_client(cfg)

    for i in range(args.warmup):   # before the lab run, so it isn't recorded
        await one_request(client, cfg.model, f"warmup-{i}", "warmup", wl.token_ids[i], 16)

    if args.no_lab:
        run_id = time.strftime("local-%Y%m%d-%H%M%S")
    else:
        r = await admin.post("/admin/runs", json={"exp": args.exp, "client": jsonable(args) | {"model": cfg.model}})
        r.raise_for_status()
        run_id = r.json()["run_id"]
    out_dir = cfg.out_base / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    print(f"[client] run {run_id} -> {out_dir}", flush=True)

    rows: list[dict] = []
    stop_info = None
    try:
        offset = 0
        plans = [("rate", r) for r in args.rates] + [("users", u) for u in args.users or []]
        for kind, x in plans:
            await wait_idle(admin)
            t0 = time.time()
            if kind == "rate":
                traces = await open_loop(client, cfg.model, wl.token_ids, x, args.n, args.osl, run_id, offset)
            else:
                traces = await closed_loop(client, cfg.model, wl.token_ids, int(x), args.n, args.osl, run_id, offset)
            offset += args.n   # fresh prompts per point
            rows += [t.row() for t in traces]
            n_ok = sum(t.error is None for t in traces)
            print(f"[client] {kind}={x:g}: {n_ok}/{len(traces)} ok in {time.time() - t0:.1f}s", flush=True)
    finally:
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / "client_requests.csv", index=False)
        if not args.no_lab:
            await wait_idle(admin)
            stop_info = (await admin.post(f"/admin/runs/{run_id}/stop")).json()
        (out_dir / "client_meta.json").write_text(json.dumps(
            {"run_id": run_id, "args": jsonable(args), "config": {k: str(v) for k, v in vars(cfg).items()},
             "server_stop": stop_info}, indent=2))
        await client.close()
        await admin.aclose()

    if rows:
        print(summarize(pd.DataFrame(rows)).to_string(index=False))
    if stop_info:
        print(f"[client] server wrote {stop_info['requests']} rows to {stop_info['dir']} (rw pull, then report)")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m client")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--model", default=None, help="served model name")
    ap.add_argument("--tokenizer", default=None, help="HF id for exact-ISL token ids")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("chat")
    c.add_argument("message")
    c.add_argument("--max-tokens", type=int, default=256)

    s = sub.add_parser("sweep")
    s.add_argument("--rates", type=float, nargs="*", default=[1.0], help="Poisson req/s; 'inf' = all at once")
    s.add_argument("--users", type=int, nargs="*", default=None, help="closed-loop concurrency points")
    s.add_argument("-n", type=int, default=200, help="requests per point")
    s.add_argument("--isl", type=int, default=128)
    s.add_argument("--osl", type=int, default=512)
    s.add_argument("--prompts-path", default="configs/workloads/prompts.txt")
    s.add_argument("--warmup", type=int, default=4)
    s.add_argument("--exp", default="mac-sweep")
    s.add_argument("--no-lab", action="store_true", help="server without --enable-lab-admin")

    r = sub.add_parser("report")
    r.add_argument("client_dir", type=Path)
    r.add_argument("--server-dir", type=Path, default=None)
    r.add_argument("--slo-ttft-s", type=float, default=1.0)
    r.add_argument("--slo-itl-ms", type=float, default=50.0)

    args = ap.parse_args()
    cfg = ClientConfig()
    overrides = {k: getattr(args, k) for k in ("base_url", "api_key", "model", "tokenizer") if getattr(args, k)}
    cfg = replace(cfg, **overrides)

    if args.cmd == "chat":
        async def run():
            client = openai_client(cfg)
            try:
                await chat_once(client, cfg.model, args.message, args.max_tokens)
            finally:
                await client.close()
        asyncio.run(run())
    elif args.cmd == "sweep":
        asyncio.run(sweep(cfg, args))
    else:
        report(args.client_dir, args.server_dir, slo_ttft_s=args.slo_ttft_s, slo_itl_ms=args.slo_itl_ms)


if __name__ == "__main__":
    main()
