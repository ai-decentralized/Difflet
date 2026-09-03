#!/usr/bin/env bash
# Per-cell driver for scripts/verify_cli.py with resume.
#   REPO=/path/to/Difflet bash run_cells.sh "flux wan" "tp4 tp2cp2ring"
# One verify_cli invocation per (model, config); any cell with a terminal outcome
# (PASS/SKIP/XFAIL/FAIL) in ANY /tmp/logs/verify_matrix_*/results.json is skipped,
# so a swept run redoes at most the in-flight cell. Quarantine a run dir (rename it
# out of the verify_matrix_* glob) to force a cell to rerun. flock prevents two
# drivers from fighting for the NeuronCores. Writes /tmp/logs/RESUME_DONE.marker.
set -u
exec 9>/tmp/logs/run_cells.lock
flock -n 9 || { echo "[driver] another driver holds the lock, exiting"; exit 0; }
REPO=${REPO:-$PWD}
MODELS=$1; CONFIGS=$2
cd "$REPO" && source .venv/bin/activate
mkdir -p /tmp/logs

cell_done() {
  python3 - "$1" "$2" <<'PY'
import glob, json, sys
key = f"{sys.argv[1]}/{sys.argv[2]}"
for p in glob.glob("/tmp/logs/verify_matrix_*/results.json"):
    try: d = json.load(open(p))
    except Exception: continue
    v = d.get("cells", {}).get(key)
    if v and v.get("outcome") in ("PASS", "SKIP", "XFAIL", "FAIL"):
        sys.exit(0)
sys.exit(1)
PY
}

overall=0
for model in $MODELS; do
  for config in $CONFIGS; do
    if cell_done "$model" "$config"; then echo "[driver] $model/$config already done, skipping"; continue; fi
    echo "[driver] running $model/$config"
    python scripts/verify_cli.py --models "$model" --configs "$config" || { overall=1; echo "[driver] $model/$config exited non-zero"; }
  done
done
echo "RESUME_DONE overall_rc=$overall" | tee /tmp/logs/RESUME_DONE.marker
exit $overall
