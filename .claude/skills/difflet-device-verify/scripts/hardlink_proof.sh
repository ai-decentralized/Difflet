#!/usr/bin/env bash
# Weight-sharing proof: hardlink counts on shard0 of every tp4 store entry.
# Run before and after a bucketed compile; the count rises when a new artifact
# attaches to the same bytes (same inode, zero duplicate storage).
#   bash hardlink_proof.sh [label]
echo "--- ${1:-hardlink counts} $(date -u +%FT%TZ) ---"
stat -c '%h links  %n' ~/.cache/difflet/_shared_weights/*__tp4__*/shard0.safetensors 2>/dev/null \
  | sed 's|/home/[^ ]*/_shared_weights/||'
