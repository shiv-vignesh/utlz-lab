#!/usr/bin/env bash
# Lock / reset / inspect GPU clocks so "% of peak" has a stable denominator.
#
#   bash scripts/lock_clocks.sh status              # current clocks, power, throttle reasons
#   bash scripts/lock_clocks.sh list                # supported SM/memory clock pairs
#   bash scripts/lock_clocks.sh lock <SM_MHZ>       # pin SM (graphics) clock: -lgc SM,SM
#   bash scripts/lock_clocks.sh lock <SM_MHZ> <MEM_MHZ>   # also try -lmc (may be unsupported)
#   bash scripts/lock_clocks.sh reset               # -rgc (+ -rmc)
#   bash scripts/lock_clocks.sh probe [SECONDS]     # FP16 GEMM load, print clock/power/reasons each second
#
# Picking SM_MHZ: there's deliberately no default. A clock the card can't hold
# under a heavy FP16 GEMM is useless: the power limit pulls it down anyway and
# your "locked" denominator is wrong. Procedure:
#   1. `list` to see what's supported
#   2. `lock <candidate>` then `probe 30`
#   3. if clocks.sm sags below the lock or sw_power_cap / thermal shows Active,
#      pick a lower clock and repeat
#   4. record the chosen clock in notes/phase0-findings.md
#
# UNVERIFIED on this setup (check and note the result):
#   - whether -lgc works on a GeForce card inside a Vast VM (needs root + driver support)
#   - whether -lmc (memory clock lock) is supported on GA102 GeForce
#   - locks do NOT survive a reboot / driver reload; persistence mode keeps
#     the driver loaded between processes so the lock isn't lost when idle
set -euo pipefail

cmd="${1:-status}"

status() {
    nvidia-smi --query-gpu=index,pstate,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,power.draw,power.limit,temperature.gpu,utilization.gpu \
               --format=csv
    echo
    echo "Throttle reasons (Active = holding the clock down):"
    nvidia-smi -q -d PERFORMANCE | sed -n '/Clocks Event Reasons\|Clocks Throttle Reasons/,/^$/p' | sed 's/^/  /'
}

case "$cmd" in
    status)
        status
        ;;

    list)
        # Long output: memory clock, then every graphics clock supported at it.
        nvidia-smi -q -d SUPPORTED_CLOCKS
        ;;

    lock)
        sm="${2:-}"
        mem="${3:-}"
        [ -n "$sm" ] || { echo "usage: $0 lock <SM_MHZ> [MEM_MHZ]   (see '$0 list')" >&2; exit 2; }

        nvidia-smi -pm 1 || echo "WARN: could not enable persistence mode" >&2
        nvidia-smi -lgc "$sm,$sm"
        if [ -n "$mem" ]; then
            nvidia-smi -lmc "$mem,$mem" || echo "WARN: -lmc failed (memory clock locking may be unsupported on this GPU)" >&2
        fi
        echo
        echo "Locked. Idle clocks may still read low; check under load with: $0 probe"
        status
        ;;

    reset)
        nvidia-smi -rgc
        nvidia-smi -rmc 2>/dev/null || true   # harmless if -lmc was never set / unsupported
        echo "Clocks reset to default (persistence mode left as is)."
        status
        ;;

    probe)
        secs="${2:-30}"
        python -c "import torch" 2>/dev/null || { echo "ERROR: probe needs torch; run inside the venv" >&2; exit 1; }
        echo "FP16 8192^3 GEMM for ${secs}s; sampling every 1s. Watch clocks.sm vs your lock, and the reasons."
        python - "$secs" <<'PY' &
import sys, time, torch
dur = float(sys.argv[1]); n = 8192
a = torch.randn(n, n, device="cuda", dtype=torch.float16)
b = torch.randn(n, n, device="cuda", dtype=torch.float16)
t0 = time.time()
while time.time() - t0 < dur:
    for _ in range(10): a @ b
    torch.cuda.synchronize()
PY
        load_pid=$!
        trap 'kill $load_pid 2>/dev/null || true' EXIT
        # -lms 1000: one line per second until we kill it
        nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_slowdown,clocks_event_reasons.sw_thermal_slowdown \
                   --format=csv -lms 1000 &
        smi_pid=$!
        wait "$load_pid" || true
        kill "$smi_pid" 2>/dev/null || true
        wait "$smi_pid" 2>/dev/null || true
        ;;

    *)
        echo "usage: $0 {status|list|lock <SM_MHZ> [MEM_MHZ]|reset|probe [SECONDS]}" >&2
        exit 2
        ;;
esac
