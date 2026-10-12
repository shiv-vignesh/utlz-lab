from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = Path(os.environ.get("OUT_BASE", "/workspace/utlz-out"))

@dataclass
class Run:
    run_id: str
    _dir: Path
    meta:dict = field(default_factory=dict)

    def path(self, name:str) -> Path:
        return self._dir / name

    def save_meta(self) -> None:

        tmp = self.path("meta.json.tmp")
        tmp.write_text(json.dumps(self.meta, indent=2, default=str))
        tmp.replace(self.path("meta.json"))

def gpu_snapshot() -> dict | None:
    script = PROJECT_ROOT / "scripts" / "gpu_snapshot.sh"
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[run] gpu_snapshot failed: {r.stderr.strip()}", file=sys.stderr)
        return

    return json.loads(r.stdout)

def new_run(phase:str, exp:str, args: dict | None = None) -> Run:

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{phase}-{exp}-{ts}"

    run_dir = OUT_BASE / "results" / phase / exp / ts
    run_dir.mkdir(parents=True, exist_ok=False)

    run = Run(run_id=f"{phase}-{exp}-{ts}", _dir=run_dir)
    run.meta = {
        "run_id": run_id,
        "phase": phase,
        "exp": exp,
        "argv": sys.argv,
        "args": args or {},
        "t_start": time.time(),
        "gpu_before": gpu_snapshot(),        
    }
    run.save_meta()

    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    run.path("pip_freeze.txt").write_text(freeze.stdout)

    print(f"[run] {run_id} -> {run_dir}", file=sys.stderr)
    return run

def finish_run(run: Run, **extra) -> None:
    run.meta["t_end"] = time.time()
    run.meta["gpu_after"] = gpu_snapshot()
    run.meta.update(extra)
    run.save_meta()