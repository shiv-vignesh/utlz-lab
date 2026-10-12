#!/usr/bin/env bash
# Build the Phase 2 venv. Run: rw run "bash setup/setup_vllm.sh"
set -euo pipefail
VENV=/workspace/venv-vllm

[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -U pip
"$VENV/bin/pip" install -r requirements/vllm.txt

"$VENV/bin/python" - <<'EOF'
import torch, vllm
print("vllm", vllm.__version__, "| torch", torch.__version__, "| cuda", torch.version.cuda)
assert torch.cuda.is_available(), "torch can't see the GPU (driver vs wheel CUDA?)"
EOF

# Checkpoint 1: Phase 1's venv must be untouched
/workspace/venv/bin/python -c "import torch; print('phase1 torch', torch.__version__)"
