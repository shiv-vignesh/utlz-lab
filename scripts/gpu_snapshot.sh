#!/usr/bin/env bash
# Snapshot GPU state (clocks, power, throttle reasons, driver, CUDA) as JSON.
#
#   bash scripts/gpu_snapshot.sh              # JSON to stdout
#   bash scripts/gpu_snapshot.sh out.json     # JSON to a file
#
# Meant to be embedded into each run's meta.json (common/run.py), taken once
# before and once after the measured phase.
#
# Caveat: a snapshot of an IDLE GPU shows idle clocks (P8, a few hundred MHz).
# For scaling peaks you want the clock *under load*. Take that from a snapshot
# during the run, or from nvidia-smi sampling alongside the benchmark.
#
# Two different "CUDA versions" are recorded, and they mean different things:
#   driver_cuda_version: the newest CUDA the driver supports (nvidia-smi header)
#   torch_cuda_build:    the CUDA runtime torch was compiled against
set -euo pipefail

OUT="${1:-}"

command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi not found" >&2; exit 1; }

# Uses whatever python3 is on PATH. Inside the venv it also records torch info;
# outside, those fields are null.
json="$(python3 - <<'PY'
import json, platform, re, shutil, socket, subprocess, time

def smi(*args):
    r = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None

# Queried one field at a time on purpose: if a field name doesn't exist on this
# driver/GPU, nvidia-smi rejects the WHOLE query. This way it just becomes null.
# (Run `nvidia-smi --help-query-gpu` for the full list.)
FIELDS = [
    "index", "name", "uuid", "pci.bus_id", "vbios_version",
    "pstate", "persistence_mode", "compute_mode",
    "clocks.sm", "clocks.mem", "clocks.gr",
    "clocks.max.sm", "clocks.max.mem",
    "clocks.applications.graphics", "clocks.applications.memory",
    "power.draw", "power.limit", "enforced.power.limit", "power.max_limit",
    "temperature.gpu", "fan.speed",
    "utilization.gpu", "utilization.memory",
    "memory.total", "memory.used",
    "pcie.link.gen.current", "pcie.link.width.current",
    # Why the clock isn't at max. Bitmask + the individual reasons that matter here.
    "clocks_event_reasons.active",
    "clocks_event_reasons.gpu_idle",
    "clocks_event_reasons.sw_power_cap",
    "clocks_event_reasons.hw_slowdown",
    "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.hw_power_brake_slowdown",
    "clocks_event_reasons.applications_clocks_setting",
]

gpus = None
for field in FIELDS:
    out = smi(f"--query-gpu={field}", "--format=csv,noheader,nounits")
    values = out.splitlines() if out is not None else None
    if gpus is None:
        gpus = [{} for _ in (values or [None])]
    for i, g in enumerate(gpus):
        v = values[i].strip() if values and i < len(values) else None
        g[field] = None if v in (None, "[N/A]", "N/A", "[Not Supported]") else v

header = smi() or ""
m = re.search(r"CUDA Version:\s*([\d.]+)", header)

snap = {
    "t_wall": time.time(),
    "t_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "hostname": socket.gethostname(),
    "kernel": platform.release(),
    "driver_version": smi("--query-gpu=driver_version", "--format=csv,noheader").splitlines()[0],
    "driver_cuda_version": m.group(1) if m else None,
    "torch_version": None,
    "torch_cuda_build": None,
    "utlz_version": None,
    "gpus": gpus,
}

try:
    import torch
    snap["torch_version"] = torch.__version__
    snap["torch_cuda_build"] = torch.version.cuda
except ImportError:
    pass

if shutil.which("utlz"):
    r = subprocess.run(["utlz", "-version"], capture_output=True, text=True)
    snap["utlz_version"] = r.stdout.strip() or None

print(json.dumps(snap, indent=2))
PY
)"

if [ -n "$OUT" ]; then
    mkdir -p "$(dirname "$OUT")"
    printf '%s\n' "$json" >"$OUT"
    echo "wrote $OUT" >&2
else
    printf '%s\n' "$json"
fi
