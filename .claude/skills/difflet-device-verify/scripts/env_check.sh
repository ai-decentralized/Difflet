#!/usr/bin/env bash
# Phase 0: host readiness for a Difflet device campaign. Run from the repo root.
# Reports; never changes anything.
REPO=${REPO:-$PWD}
echo "== repo: $REPO ($(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null))"
echo "== NeuronCores"
if command -v neuron-ls >/dev/null; then
  neuron-ls -j 2>/dev/null | python3 -c 'import json,sys
for d in json.load(sys.stdin):
    print(f"  {d[\"instance_type\"]}: {d[\"nc_count\"]} cores, LNC={d[\"logical_neuroncore_config\"]}, "
          f"{d[\"memory_size\"]/2**30:.0f} GiB, busy processes: {len(d.get(\"neuron_processes\", []))}")'
else
  echo "  neuron-ls not found — not a Neuron host"
fi
echo "== venv"
if [ -x "$REPO/.venv/bin/difflet" ]; then
  echo "  $REPO/.venv OK ($("$REPO/.venv/bin/python" --version 2>&1))"
else
  echo "  MISSING: run ./scripts/setup_env.sh (Python 3.12 + lockfile keeps compile caches portable)"
fi
echo "== Hugging Face token: $([ -f ~/.cache/huggingface/token ] && echo present || echo MISSING — FLUX.1-dev is gated)"
echo "== disk"
df -h / | tail -1 | awk '{print "  free " $4 " of " $2 " (" $5 " used)"}'
du -sh ~/.cache/huggingface ~/.cache/difflet 2>/dev/null | sed 's/^/  /'
echo "== raw run dirs: $(ls -d /tmp/logs/verify_matrix_* 2>/dev/null | wc -l)"
