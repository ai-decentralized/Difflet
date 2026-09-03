#!/usr/bin/env bash
# Copy one model's cells from every /tmp run dir into the repo evidence tree,
# preserving run-dir provenance, then prune intermediate tensors.
#   REPO=... CAMPAIGN=verification-2026-08-29 bash curate_cells.sh <model>
set -euo pipefail
MODEL=$1
REPO=${REPO:-$PWD}
DEST="$REPO/artifacts/${CAMPAIGN:?set CAMPAIGN}/phase2-parallelism"
for run in /tmp/logs/verify_matrix_*; do
  [ -d "$run/$MODEL" ] || continue
  name="run-$(basename "$run" | sed 's/verify_matrix_//')"
  mkdir -p "$DEST/$name"
  cp -r "$run/$MODEL" "$DEST/$name/"
  [ -f "$run/results.json" ] && cp "$run/results.json" "$DEST/$name/"
  [ -f "$run/main.log" ] && cp "$run/main.log" "$DEST/$name/"
done
find "$DEST" -path "*/work/*" -name "*.pt" -delete
du -sh "$DEST"
