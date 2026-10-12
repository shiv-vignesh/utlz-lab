#!/usr/bin/env bash
# Smoke test: can Utilyze read real SOL counters on this GPU?
#
#   1. start `utlz -mode server` (headless) unless one is already on $UTLZ_PORT
#   2. connect to its WebSocket, record ~IDLE_S s of idle samples
#   3. run a big FP16 matmul for LOAD_S s, keep recording
#   4. PASS if SOL samples are valid and compute SOL under load is clearly
#      above idle. Exit 0 = pass, 1 = fail.
#
# Run on the box, inside the venv (needs torch + websockets):
#   runway-tui -C . run bash scripts/check_counters.sh
#
# What failure looks like (from reading utilyze v0.1.3 source):
#   - Counters unreadable -> the server EXITS at startup with
#     "error: sampler could not initialize ... device N could not initialize: <reason>".
#     utlz never prints ERR_NVGPUCTRPERM itself. That name comes from NVIDIA's
#     docs/ncu; the <reason> text is whatever Nsight Perf returned.
#   - Server up but no counter data -> metrics events arrive with SOL.Valid=false
#     (NVML utilization can still be valid, so "utlz shows something" isn't enough).
set -euo pipefail

UTLZ_PORT="${UTLZ_PORT:-8079}"
# Label this script sends as ?client_id= on the WebSocket. utlz requires a
# non-empty value (400 otherwise) and only uses it to track connected clients.
# Our own name: utlz doesn't read UTLZ_CLIENT_ID itself.
UTLZ_CLIENT_ID="${UTLZ_CLIENT_ID:-check-counters}"
IDLE_S="${IDLE_S:-8}"
LOAD_S="${LOAD_S:-30}"
MIN_LOAD_COMPUTE_SOL="${MIN_LOAD_COMPUTE_SOL:-20}"   # % ; FP16 8k GEMM should be far above this
OUT_BASE="${OUT_BASE:-/workspace/utlz-out}"
LOG_DIR="$OUT_BASE/logs/check_counters/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"

command -v utlz >/dev/null || { echo "ERROR: utlz not on PATH (run setup first)"; exit 1; }
python -c "import torch, websockets" 2>/dev/null || { echo "ERROR: run inside the venv (needs torch + websockets)"; exit 1; }

echo "utlz:    $(utlz -version)"
echo "client:  $UTLZ_CLIENT_ID"
echo "logs:    $LOG_DIR"

# ---------------------------------------------------------------------------
# 1. utlz server
# ---------------------------------------------------------------------------
healthy() { curl -sf "http://127.0.0.1:$UTLZ_PORT/healthz" >/dev/null; }

STARTED_SERVER=0
SERVER_PID=""
cleanup() {
    if [ "$STARTED_SERVER" -eq 1 ] && [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if healthy; then
    echo "server:  reusing utlz already listening on :$UTLZ_PORT"
else
    echo "server:  starting utlz -mode server on :$UTLZ_PORT"
    # UTLZ_DISABLE_PROFILING_WARNING: skip the CAP_SYS_ADMIN nag (we're root).
    UTLZ_DISABLE_PROFILING_WARNING=1 \
        utlz -mode server -port "$UTLZ_PORT" \
             -log "$LOG_DIR/utlz.log" -log-level DEBUG \
             >"$LOG_DIR/utlz.stdout" 2>"$LOG_DIR/utlz.stderr" &
    SERVER_PID=$!
    STARTED_SERVER=1

    for _ in $(seq 1 30); do
        healthy && break
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo
            echo "FAIL: utlz server exited during startup. stderr:"
            sed 's/^/    /' "$LOG_DIR/utlz.stderr"
            exit 1
        fi
        sleep 0.5
    done
    healthy || { echo "FAIL: utlz server not healthy after 15 s"; sed 's/^/    /' "$LOG_DIR/utlz.stderr"; exit 1; }
fi

# ---------------------------------------------------------------------------
# 2-4. record idle + load, then judge
# ---------------------------------------------------------------------------
export UTLZ_PORT UTLZ_CLIENT_ID IDLE_S LOAD_S MIN_LOAD_COMPUTE_SOL LOG_DIR
python - <<'PY'
import asyncio, json, os, subprocess, sys, time
from urllib.parse import quote

import websockets

PORT = os.environ["UTLZ_PORT"]
CLIENT_ID = os.environ["UTLZ_CLIENT_ID"]
IDLE_S = float(os.environ["IDLE_S"])
LOAD_S = float(os.environ["LOAD_S"])
MIN_SOL = float(os.environ["MIN_LOAD_COMPUTE_SOL"])
LOG_DIR = os.environ["LOG_DIR"]
URL = f"ws://127.0.0.1:{PORT}/live?client_id={quote(CLIENT_ID)}"

# Load: square FP16 GEMM, reports its own achieved TFLOP/s as a sanity number.
LOAD = r"""
import sys, time, torch
dur = float(sys.argv[1]); n = 8192
a = torch.randn(n, n, device="cuda", dtype=torch.float16)
b = torch.randn(n, n, device="cuda", dtype=torch.float16)
for _ in range(3): a @ b
torch.cuda.synchronize()
t0 = time.time(); iters = 0
while time.time() - t0 < dur:
    for _ in range(10): c = a @ b
    torch.cuda.synchronize(); iters += 10
el = time.time() - t0
print(f"load: {iters} x {n}^3 fp16 GEMM in {el:.1f}s = {2*n**3*iters/el/1e12:.1f} TFLOP/s", flush=True)
"""

samples = []  # (phase, t_recv, gpu dict)

async def record():
    async with websockets.connect(URL, max_size=None) as ws:
        phase = "idle"
        t_start = time.time()
        load = None
        while True:
            now = time.time()
            if phase == "idle" and now - t_start >= IDLE_S:
                phase = "load"
                load = subprocess.Popen([sys.executable, "-c", LOAD, str(LOAD_S)])
                print(f"load started (pid {load.pid})", flush=True)
            if phase == "load" and load.poll() is not None:
                break
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            ev = json.loads(raw)
            with open(os.path.join(LOG_DIR, "events.jsonl"), "a") as f:
                f.write(raw if isinstance(raw, str) else raw.decode())
                f.write("\n")
            if ev.get("type") != "metrics" or not ev.get("snapshot"):
                continue
            for g in ev["snapshot"].get("GPUs") or []:
                samples.append((phase, time.time(), g))
        return load.returncode

rc = asyncio.run(record())
if rc != 0:
    print(f"FAIL: load process exited with {rc}")
    sys.exit(1)

def summarize(phase):
    rows = [g for p, _, g in samples if p == phase]
    sol = [g["SOL"] for g in rows if g.get("SOL", {}).get("Valid")]
    nvml = [g["NVMLUtilization"]["UtilPct"] for g in rows if g.get("NVMLUtilization", {}).get("Valid")]
    sm = [g["DCGMUtilization"]["SMActivePct"] for g in rows if g.get("DCGMUtilization", {}).get("Valid")]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    return {
        "events": len(rows),
        "sol_valid": len(sol),
        "compute_sol": mean([s["ComputePct"] for s in sol]),
        "memory_sol": mean([s["MemoryPct"] for s in sol]),
        "sm_active": mean(sm),
        "nvml_util": mean(nvml),
    }

idle, load = summarize("idle"), summarize("load")
print()
print(f"{'phase':<6} {'events':>6} {'SOLvalid':>8} {'computeSOL%':>11} {'memorySOL%':>10} {'SMactive%':>9} {'nvml%':>6}")
for name, s in (("idle", idle), ("load", load)):
    print(f"{name:<6} {s['events']:>6} {s['sol_valid']:>8} {s['compute_sol']:>11.1f} "
          f"{s['memory_sol']:>10.1f} {s['sm_active']:>9.1f} {s['nvml_util']:>6.1f}")
with open(os.path.join(LOG_DIR, "summary.json"), "w") as f:
    json.dump({"idle": idle, "load": load}, f, indent=2)

print()
if load["events"] == 0:
    print("FAIL: no metrics events received during load")
    sys.exit(1)
if load["sol_valid"] == 0:
    print("FAIL: utlz is running but SOL was never valid (counters not readable?). "
          "NVML utilization alone doesn't count. See utlz.log.")
    sys.exit(1)
if not load["compute_sol"] >= MIN_SOL:
    print(f"FAIL: compute SOL under FP16 GEMM load is {load['compute_sol']:.1f}% (< {MIN_SOL}%)")
    sys.exit(1)
if idle["sol_valid"] and not load["compute_sol"] - idle["compute_sol"] >= 10:
    print(f"FAIL: compute SOL barely moves between idle ({idle['compute_sol']:.1f}%) "
          f"and load ({load['compute_sol']:.1f}%); the numbers aren't tracking the GPU")
    sys.exit(1)
print(f"PASS: SOL counters readable; compute SOL {load['compute_sol']:.1f}% under load "
      f"vs {idle['compute_sol']:.1f}% idle")
PY
