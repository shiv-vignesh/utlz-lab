#!/usr/bin/env bash
# One-time (and re-runnable) box setup for utlz-lab.
#
# runway runs this as part of `runway setup`, AFTER it has created the venv:
#     cd <remote.dir> && . <venv>/bin/activate && bash setup/setup.sh
# So: cwd is the project root, the venv is already active, and `pip` is the venv's pip.
#
# Idempotent: every step checks before acting, so running it twice is harmless.
#
# Env overrides:
#   UTLZ_VERSION=v0.x.y   pin a Utilyze release (default: latest)
#   UTLZ_FORCE=1          reinstall Utilyze even if it is already present
#   REQS=path.txt         requirements file to install (default: requirements/hf.txt,
#                         which includes base.txt)
set -euo pipefail

REQS="${REQS:-requirements/hf.txt}"
UTLZ_BIN_DIR="/usr/local/bin"

log()  { printf '\n==> %s\n' "$*"; }
warn() { printf 'WARN: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0. Preconditions
# ---------------------------------------------------------------------------
log "Preconditions"

[ "$(id -u)" -eq 0 ] || warn "not running as root; apt steps and utlz profiling will likely fail"

[ -n "${VIRTUAL_ENV:-}" ] || die "no venv active. Run this through 'runway setup', or source the venv first."
echo "venv: $VIRTUAL_ENV"

command -v nvidia-smi >/dev/null || die "nvidia-smi not found. Is the NVIDIA driver installed on this VM?"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# ---------------------------------------------------------------------------
# 1. System packages
#    tmux: runway runs `background = true` descriptors under tmux.
#    curl: the Utilyze installer needs it.
#    NOTE: python3-venv can't be installed here. runway creates the venv
#    *before* this script runs, so if `python3 -m venv` fails on a fresh
#    Ubuntu 22.04 box, run `apt-get install -y python3-venv` by hand once.
# ---------------------------------------------------------------------------
log "System packages"

APT_PKGS=(tmux curl git)
missing=()
for p in "${APT_PKGS[@]}"; do
    dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
done
if [ "${#missing[@]}" -gt 0 ]; then
    echo "installing: ${missing[*]}"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq "${missing[@]}"
else
    echo "already present: ${APT_PKGS[*]}"
fi

# ---------------------------------------------------------------------------
# 2. Python deps
#    pip is already idempotent: it skips requirements that are satisfied.
# ---------------------------------------------------------------------------
log "Python deps from $REQS"

[ -f "$REQS" ] || die "$REQS not found."
python -m pip install --upgrade --quiet pip
python -m pip install --quiet -r "$REQS"

python - <<'PY'
import torch
print(f"torch {torch.__version__}  CUDA build {torch.version.cuda}  available={torch.cuda.is_available()}")
try:
    import transformers
    print(f"transformers {transformers.__version__}")
except ImportError:
    print("transformers not installed (REQS without hf.txt)")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}  cc={torch.cuda.get_device_capability(0)}")
else:
    raise SystemExit("torch can't see the GPU. Check the driver vs. the torch CUDA build.")
PY

# ---------------------------------------------------------------------------
# 3. Utilyze
#    The official installer (https://systalyze.com/utilyze/install.sh) asks
#    y/N questions on /dev/tty. That doesn't work over runway's non-interactive
#    ssh, so drive it with its own env vars instead:
#      UTLZ_INSTALL_DIR          put the binary straight in /usr/local/bin (on PATH)
#      UTLZ_INSTALL_WITHOUT_ROOT skip the system-wide-symlink / setcap prompts
#                                (we're root, so CAP_SYS_ADMIN isn't needed)
#      UTLZ_INSTALL_WITHOUT_PATH skip the "edit your shell profile" prompt
#    These variable names come from reading install.sh on 2026-10-05; re-check
#    them if the installer changes.
# ---------------------------------------------------------------------------
log "Utilyze"

if command -v utlz >/dev/null && [ -z "${UTLZ_FORCE:-}" ]; then
    echo "already installed: $(command -v utlz) ($(utlz -version 2>/dev/null || echo 'version unknown'))"
    echo "set UTLZ_FORCE=1 to reinstall"
else
    curl -sSfL https://systalyze.com/utilyze/install.sh \
        | UTLZ_INSTALL_DIR="$UTLZ_BIN_DIR" \
          UTLZ_INSTALL_WITHOUT_ROOT=1 \
          UTLZ_INSTALL_WITHOUT_PATH=1 \
          UTLZ_VERSION="${UTLZ_VERSION:-latest}" \
          sh
fi
command -v utlz >/dev/null || die "utlz not on PATH after install"

# ---------------------------------------------------------------------------
# 4. CUPTI for utlz
#    utlz needs NVIDIA CUPTI 12+. If it can't find one, it ASKS on stdin
#    whether to download it into ~/.cache/utlz. Runway's tasks and tmux jobs
#    are non-interactive, so the question goes unanswered and utlz exits 1
#    ("failed to load CUPTI: missing and did not download"). Two of the first
#    three Vast boxes lacked CUPTI.
#    Fix: start a throwaway server once with "y" piped in, on port 18079 so it
#    never clashes with a real server on 8079. `timeout` ends it; rc 124 is
#    the expected outcome. If CUPTI is already on the system, utlz never asks.
# ---------------------------------------------------------------------------
log "CUPTI for utlz"

UTLZ_CACHE="$HOME/.cache/utlz"
if compgen -G "$UTLZ_CACHE/cupti-*" >/dev/null; then
    echo "already cached: $(ls -d "$UTLZ_CACHE"/cupti-*)"
else
    probe_out="$(echo y | UTLZ_DISABLE_PROFILING_WARNING=1 UTLZ_DISABLE_METRICS=1 \
        timeout 20 utlz -mode server -port 18079 2>&1 || true)"
    if grep -qE "failed to load CUPTI|did not download" <<<"$probe_out"; then
        printf '%s\n' "$probe_out" | sed 's/^/    /'
        die "utlz could not get CUPTI; utlz-server will not start on this box"
    fi
    if compgen -G "$UTLZ_CACHE/cupti-*" >/dev/null; then
        echo "downloaded: $(ls -d "$UTLZ_CACHE"/cupti-*)"
    else
        echo "no download needed (utlz found a system CUPTI)"
    fi
fi

# ---------------------------------------------------------------------------
# 5. Counter-access sanity (informational only, doesn't fail)
#    The driver param RmProfilingAdminOnly=1 means only root / CAP_SYS_ADMIN
#    can read perf counters. We're root, so that's fine either way. The real
#    test is scripts/check_counters.sh (SOL valid and rising under load).
# ---------------------------------------------------------------------------
log "Counter access"

if [ -r /proc/driver/nvidia/params ]; then
    grep -E '^RmProfilingAdminOnly' /proc/driver/nvidia/params || echo "RmProfilingAdminOnly not listed"
else
    warn "/proc/driver/nvidia/params not readable; can't check profiling restriction"
fi

log "Setup done. Next: run check-counters to confirm utlz can read SOL on this GPU."
