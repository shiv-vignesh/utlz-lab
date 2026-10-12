# Onboarding: bring up a fresh GPU box

From "just rented a Vast instance" to "live Utilyze metrics on the Mac".
Run the steps in order; don't move on until the **expect** line holds.

```
Mac                                   Box (Vast VM, root)
rw run/task/pull ──── ssh ──────────► /workspace/utlz-lab   code (push mirrors with --delete)
                                      /workspace/venv       venv
                                      /workspace/utlz-out   results + logs (pulled back)
utlz-tunnel ── ssh -L 8079 ─────────► utlz -mode server :8079   (ONE per GPU)
utlz --connect localhost:8079
```

---

## A. One-time Mac setup

Add to `~/.zshrc`, then `source ~/.zshrc`:

```bash
# runway shortcut
alias rw='/Users/shiv/Developer/tools/test-runner/runway-tui -C /Users/shiv/Developer/systalyze-ramp-up/utlz-lab'

# tunnel to whatever box runway.toml points at (kills any tunnel to an old box first)
utlz-tunnel() {
  local cfg=~/Developer/systalyze-ramp-up/utlz-lab/runway.toml
  local host=$(awk -F'"' '/^host/{print $2; exit}' "$cfg")
  local port=$(awk -F' *= *' '/^port/{print $2; exit}' "$cfg")
  pkill -f "8079:127.0.0.1:8079" 2>/dev/null
  ssh -f -N -p "$port" -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
      -L 8079:127.0.0.1:8079 "$host" && echo "tunnel → $host:$port"
}
```

```bash
curl -sSfL https://systalyze.com/utilyze/install.sh | sh        # utlz client (no root needed)
python3 -m venv .venv && .venv/bin/pip install pandas numpy matplotlib pyyaml   # local analysis
```

**Expect:** `rw tasks` lists `utlz-server`, `check-counters`, `p0-matmul`.

---

## B. Every new box

| # | Do | Expect |
|---|---|---|
| 1 | Rent: **Ubuntu 22.04 VM template** (not Docker; counters need host access), RTX 3090, ≥60 GB disk | SSH host + port from Vast |
| 2 | Edit `runway.toml` → `[remote]` `host` and `port` only | `rw status` reachable |
| 3 | `rw run "nvidia-smi --query-gpu=name,driver_version --format=csv"` | RTX 3090; note the driver version |
| 4 | `rw setup` (installs base + transformers, utlz, and CUPTI) | `available=True`, a `transformers` version, `==> CUPTI for utlz` shows `downloaded:` / `already cached:`, ends `setup: venv ready` |
| 5 | `rw task utlz-server` then `rw jobs` | `running … utlz-server` |
| 6 | `rw task check-counters` | `reusing utlz already listening`, load compute SOL ≈ 45–50%, `PASS` |
| 7 | `rw run "bash scripts/gpu_snapshot.sh"` | note `clocks.max.sm`, `power.limit` (boards differ) |
| 8 | Mac: `utlz-tunnel` | `tunnel → root@<ip>:<port>` |
| 9 | Mac: `curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8079/healthz` | `200` |
| 10 | Mac: `utlz --connect localhost:8079` | GPU 0 live |
| 11 | `rw task p0-matmul` then `rw pull` | `results/p0/…/` has `meta.json`, `utlz.csv`, `bench.csv`; load SOL ≈ 50% |

Notes:
- **Step 3:** driver **< 580** can't run the default CUDA 13 torch. Before step 4,
  put `--extra-index-url https://download.pytorch.org/whl/cu128` above `torch`
  in `requirements/base.txt`.
- **Step 4, CUPTI:** utlz needs CUPTI 12+ and *asks* before downloading it.
  Runway is non-interactive, so `setup.sh` answers "y" once via a throwaway
  server on port 18079. If setup dies at this step, utlz-server can't start on
  that box.
- **Step 5:** `✓ done` only means *launched*. Always confirm with `rw jobs`.
  Only one server per GPU: check `rw jobs` before starting another.
- **Steps 8–10** run on the **Mac**. A `200` from the box only proves the server
  is up, not the tunnel.

---

## C. Phase 1 extras

```bash
rw run "df -h /workspace"                      # ~20 GB free for the model
ssh -p <port> root@<ip> /workspace/venv/bin/hf auth login   # by hand; never put tokens in descriptors
rw task p1-download
```

---

## D. Troubleshooting

| Symptom | Fix |
|---|---|
| `CUPTI 12+ was not found` / `failed to load CUPTI` | Re-run `rw setup` (its CUPTI step answers utlz's download question) |
| `rw jobs`: `exited … utlz-server (rc=1)` | SSH in, run `UTLZ_DISABLE_PROFILING_WARNING=1 timeout 15 utlz -mode server -port 8079; echo rc=$?`, read the `error:` line (`rc=124` = healthy) |
| Mac: `failed to WebSocket dial … localhost:8079` | No tunnel or a tunnel to an old box → `utlz-tunnel` |
| `pip … (from versions: none)` | Box can't reach PyPI or pip points at a bad mirror: check `pip config list`, `/etc/pip.conf`, `curl https://pypi.org/simple/numpy/` |
| `torch can't see the GPU` in setup | Driver too old for torch's CUDA build → see step 3 note |
| `runway: error: unrecognized arguments` | Quote the remote command: `rw run "python -m … --flag x"` |
| `curl` prints nothing | Needs `-w '%{http_code}\n'` (underscore, not hyphen) |
| `check-counters` FAIL, SOL never valid | Server up but counters unreadable: read `utlz.log` in the check's log dir |
| Attainable SOL never appears | Expected on the 3090 (needs vLLM + supported model on H100/A100) |

---

## E. Before destroying a box

```bash
rw pull                          # anything not pulled is lost
rw jobs                          # rw stop <utlz-server session>
pkill -f "8079:127.0.0.1:8079"   # close the Mac tunnel
```
