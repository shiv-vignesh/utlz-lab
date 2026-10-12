from __future__ import annotations

import sys, time, signal, re, argparse, csv, json
import subprocess
import os
import asyncio
import websockets

import urllib.request as request

from datetime import datetime
from pathlib import Path
from urllib.parse import quote


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = os.environ.get("UTLZ_URL", "ws://127.0.0.1:8079/live")

COLUMNS = [
    "t_wall", "t_server", "device",
    "sol_valid", "compute_sol", "memory_sol", "sm_active",
    "nvml_util", "pcie_tx_gbs", "pcie_rx_gbs",
]

def log(msg: str) -> None:
    print(f"[recorder] {msg}", file=sys.stderr, flush=True)

_FRACTION = re.compile(r"\.(\d+)")


def parse_go_time(s: str) -> float:

    s = s.replace("Z", "+00:00")
    s = _FRACTION.sub(lambda m: "." + m.group(1)[:6].ljust(6, "0"), s, count=1)
    return datetime.fromisoformat(s).timestamp()


def valid_or_blank(block: dict, key: str, scale: float = 1.0):
    return block[key] * scale if block.get("Valid") else ""


def rows_from_event(event: dict, t_wall: float) -> list[dict]:
    if event.get("type") != "metrics" or not event.get("snapshot"):
        return []
    snap = event["snapshot"]
    t_server = parse_go_time(snap["Timestamp"])

    rows = []
    for gpu in snap.get("GPUs") or []:
        sol = gpu.get("SOL") or {}
        dcgm = gpu.get("DCGMUtilization") or {}
        nvml = gpu.get("NVMLUtilization") or {}
        bw = gpu.get("Bandwidth") or {}
        rows.append({
            "t_wall": f"{t_wall:.3f}",
            "t_server": f"{t_server:.3f}",
            "device": gpu.get("DeviceID"),
            "sol_valid": int(bool(sol.get("Valid"))),
            "compute_sol": valid_or_blank(sol, "ComputePct"),
            "memory_sol": valid_or_blank(sol, "MemoryPct"),
            "sm_active": valid_or_blank(dcgm, "SMActivePct"),
            "nvml_util": valid_or_blank(nvml, "UtilPct"),
            "pcie_tx_gbs": valid_or_blank(bw, "PCIeTxBps", 1e-9),
            "pcie_rx_gbs": valid_or_blank(bw, "PCIeRxBps", 1e-9),
        })
    return rows

async def record(url:str, out:Path, client_id:str, duration:float):
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    deadline = time.time() + duration if duration > 0 else None

    full_url = f"{url}?client_id={quote(client_id)}"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    n_rows = 0

    with open(out, "a", newline="", buffering=1) as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()

        while not stop.is_set():
            try:
                async with websockets.connect(full_url, max_size=None, open_timeout=5) as ws:
                    log(f"connected to {full_url}")

                    while not stop.is_set():
                        if deadline and time.time() >= deadline:
                            stop.set()
                            break
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue  # wake up to re-check stop / deadline
                        for row in rows_from_event(json.loads(raw), time.time()):
                            writer.writerow(row)
                            n_rows += 1

            except (OSError, asyncio.TimeoutError, websockets.exceptions.WebSocketException) as e:
                if stop.is_set():
                    break
                log(f"connection problem ({e!r}); retrying in 1 s")
                try:
                    await asyncio.wait_for(stop.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass

    log(f"stopped; {n_rows} rows in {out}")

def server_healthy(url: str = DEFAULT_URL) -> bool:
    # --- helpers for experiment scripts ------------------------------------------

    health = url.replace("ws://", "http://", 1).rsplit("/live", 1)[0] + "/healthz"
    try:
        with request.urlopen(health, timeout=2) as r:
            return r.status == 200
    except OSError:  # URLError is an OSError
        return False

def start(out: Path, client_id: str, url: str = DEFAULT_URL, wait_s: float = 10.0) -> subprocess.Popen:
    """Launch the recorder as a subprocess; return once it has written its first row."""
    if not server_healthy(url):
        raise RuntimeError(f"utlz server not reachable at {url}. Start it: runway-tui -C . task utlz-server")

    proc = subprocess.Popen(
        [sys.executable, "-m", "common.recorder", "--out", str(out), "--client_id", client_id, "--url", url],
        cwd=PROJECT_ROOT,
    )
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"recorder exited early with code {proc.returncode}")
        if out.exists() and sum(1 for _ in open(out)) > 1:  # header + ≥1 row
            return proc
        time.sleep(0.2)
    stop(proc)
    raise RuntimeError(f"recorder wrote no rows within {wait_s}s")


def stop(proc: subprocess.Popen, timeout: float = 5.0) -> None:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

def main():
    ap = argparse.ArgumentParser(description="Record utlz live metrics to csv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--client_id", default=os.environ.get("UTLZ_CLIENT_ID", "utlz-lab-recorder"))
    ap.add_argument("--duration", type=float, default=0, help="seconds; 0 = until SIGINT/SIGTERM")

    args = ap.parse_args()
    asyncio.run(record(args.url, args.out, args.client_id, args.duration))

if __name__ == "__main__":
    main()