"""Per-load-point summary from client_requests.csv, optionally joined with the pulled server run dir."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SOL_LAG_S = 1.0   # same trim as Phase 1


def _p(s: pd.Series, q: float) -> float:
    s = s.dropna()
    return float(s.quantile(q)) if len(s) else float("nan")


def summarize(client: pd.DataFrame, slo_ttft_s: float = 1.0, slo_itl_ms: float = 50.0) -> pd.DataFrame:
    out = []
    for point, g in client.groupby("point", sort=False):
        ok = g[g.ok]
        span = (ok.t_send + ok.client_e2e_s).max() - g.t_send.min()
        good = ok[(ok.client_ttft_s <= slo_ttft_s) & (ok.get("itl_p99_ms", ok.itl_median_ms) <= slo_itl_ms)]
        out.append({
            "point": point,
            "requests": len(g),
            "ok": len(ok),
            "out_tok_s": ok.completion_tokens.sum() / span if span > 0 else float("nan"),
            "req_s": len(ok) / span if span > 0 else float("nan"),
            "goodput_req_s": len(good) / span if span > 0 else float("nan"),
            "ttft_p50_ms": 1e3 * _p(ok.client_ttft_s, 0.5),
            "ttft_p99_ms": 1e3 * _p(ok.client_ttft_s, 0.99),
            "srv_ttft_p50_ms": 1e3 * _p(ok.get("srv_ttft_s", pd.Series(dtype=float)), 0.5),
            "net_ttft_p50_ms": 1e3 * _p(ok.get("network_ttft_s", pd.Series(dtype=float)), 0.5),
            "queue_p50_ms": 1e3 * _p(ok.get("srv_engine_queue_s", pd.Series(dtype=float)), 0.5),
            "itl_p50_ms": _p(ok.itl_median_ms, 0.5),
            "itl_p99_ms": _p(ok.get("itl_p99_ms", pd.Series(dtype=float)), 0.99),
        })
    return pd.DataFrame(out).round(1)


def utlz_by_point(client: pd.DataFrame, server_dir: Path) -> pd.DataFrame:
    """SOL per load point over the box-clock window of that point's requests."""
    server = pd.read_csv(server_dir / "server_requests.csv")
    utlz = pd.read_csv(server_dir / "utlz.csv")
    joined = server.merge(client[["request_id", "point"]], on="request_id", how="inner")
    out = []
    for point, g in joined.groupby("point", sort=False):
        lo, hi = g.t_arrival.min() + SOL_LAG_S, g.t_last.max()
        w = utlz[(utlz.t_server >= lo) & (utlz.t_server <= hi)]
        sol = w[w.sol_valid == 1]
        out.append({
            "point": point,
            "matched": len(g),
            "isl_seen": int(g.n_prompt.median()),
            "memory_sol": sol.memory_sol.mean(),
            "compute_sol": sol.compute_sol.mean(),
            "sm_active": sol.sm_active.mean(),
            "sol_samples": len(sol),
        })
    return pd.DataFrame(out).round(1)


def report(client_dir: Path, server_dir: Path | None = None, **slo) -> None:
    client = pd.read_csv(client_dir / "client_requests.csv")
    print(summarize(client, **slo).to_string(index=False))
    if server_dir is not None:
        print()
        print(utlz_by_point(client, server_dir).to_string(index=False))
