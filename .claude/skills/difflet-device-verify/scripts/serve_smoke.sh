#!/usr/bin/env bash
# Multi-shape serving smoke for one model.
#   REPO=... bash serve_smoke.sh <model-id> <kind:image|video> "<shape,shape>" <off-set-shape> [tp]
# image shapes HxW, video shapes HxWxF. Starts difflet serve --shapes, waits for
# /ready (cold startup builds a serving generation — allow hours), requests every
# shape, sends one off-set request (expect 400 profile_mismatch), stops the server.
set -uo pipefail
MODEL=$1; KIND=$2; SHAPES=$3; OFFSET=$4; TP=${5:-4}
REPO=${REPO:-$PWD}; PORT=${PORT:-8091}
OUT=${OUT:-/tmp/logs/serve_smoke}; mkdir -p "$OUT"
cd "$REPO" && source .venv/bin/activate
pkill -f "difflet.cli.main serve" 2>/dev/null || true; sleep 5
python -m difflet.cli.main serve --model-id "$MODEL" --tp-degree "$TP" --shapes "$SHAPES" --port "$PORT" \
  > "$OUT/serve.log" 2>&1 &
t=0; until curl -sf "http://127.0.0.1:$PORT/ready" >/dev/null 2>&1; do sleep 20; t=$((t+20)); [ $t -ge ${READY_TIMEOUT:-10800} ] && { echo "READY_TIMEOUT"; pkill -f "difflet.cli.main serve"; exit 1; }; done
echo "ready after ${t}s"
req() { # shape outfile
  local h=${1%%x*} rest=${1#*x} w f; w=${rest%%x*}; f=${rest#*x}
  if [ "$KIND" = image ]; then
    curl -s -m 900 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"a small red sailboat on a calm blue lake\"}],\"extra_body\":{\"height\":$h,\"width\":$w,\"steps\":20,\"seed\":42}}" \
      -o "$2" -w "%{http_code}"
  else
    curl -s -m 1800 -X POST "http://127.0.0.1:$PORT/v1/videos/sync" -F model="$MODEL" \
      -F "prompt=a cat walking through a garden" -F height=$h -F width=$w -F num_frames=$f \
      -F num_inference_steps=20 -F seed=42 -o "$2" -w "%{http_code}"
  fi
}
for s in ${SHAPES//,/ }; do echo "shape $s -> HTTP $(req "$s" "$OUT/out_$s.bin")"; done
echo "off-set $OFFSET -> HTTP $(req "$OFFSET" "$OUT/offset.json") (expect 400): $(head -c 200 "$OUT/offset.json")"
pkill -f "difflet.cli.main serve" 2>/dev/null || true
echo "SERVE_SMOKE DONE" | tee "$OUT/DONE"
