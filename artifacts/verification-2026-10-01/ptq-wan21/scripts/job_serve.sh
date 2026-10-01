#!/usr/bin/env bash
# Phase 4: difflet serve --quant fp8 smoke (same shape/steps/seed as the CLI A/B), compare with the CLI fp8 output;
# then a bf16 serve start to confirm the pre-existing bf16 artifact still resolves.
set -uo pipefail
cd /home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan
source .venv/bin/activate
export PYTHONPATH=$PWD
EVID=$PWD/artifacts/verification-2026-10-01/ptq-wan21/serve
AB=$PWD/artifacts/verification-2026-10-01/ptq-wan21/ab
ABF=$PWD/artifacts/verification-2026-10-01/ptq-wan21/ab-fixed
mkdir -p "$EVID"
MODEL=Wan-AI/Wan2.1-T2V-14B-Diffusers
PROMPT="a cinematic shot of a red fox running through a snowy forest"

serve_once() {  # <tag> <extra flags...>
  local tag=$1; shift
  local port=8091
  echo "=== serve $tag start $(date -u +%FT%TZ)"
  setsid env NEURON_RT_NUM_CORES=4 python -m difflet.cli.main serve --model-id $MODEL \
      --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 --tp-degree 4 --height 480 --width 832 --num-frames 9 \
      --port $port "$@" > "$EVID/serve_$tag.log" 2>&1 &
  local spid=$!
  echo "server pid $spid"
  local t0=$(date +%s)
  local code=000
  while true; do
    code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/ready || true)
    [ "$code" = "200" ] && break
    if ! kill -0 $spid 2>/dev/null; then echo "server $tag died before ready"; tail -30 "$EVID/serve_$tag.log"; return 1; fi
    if [ $(( $(date +%s) - t0 )) -gt 14400 ]; then echo "server $tag not ready after 4h"; kill $spid; return 1; fi
    sleep 10
  done
  echo "ready after $(( $(date +%s) - t0 )) s"
  curl -s http://127.0.0.1:$port/ready; echo
  for i in 0 1; do
    local r0=$(date +%s.%N)
    local http=$(curl -s -o "$EVID/serve_${tag}_run$i.mp4" -w '%{http_code}' -X POST http://127.0.0.1:$port/v1/videos/sync \
      -F model=$MODEL -F "prompt=$PROMPT" -F height=480 -F width=832 -F num_frames=9 \
      -F num_inference_steps=20 -F guidance_scale=1.0 -F seed=42)
    local r1=$(date +%s.%N)
    echo "request $i: http=$http wall=$(python -c "print(round($r1-$r0,3))") s size=$(stat -c %s "$EVID/serve_${tag}_run$i.mp4")"
  done
  # off-profile shape must be rejected
  local bad=$(curl -s -o "$EVID/serve_${tag}_badshape.json" -w '%{http_code}' -X POST http://127.0.0.1:$port/v1/videos/sync \
      -F model=$MODEL -F "prompt=$PROMPT" -F height=320 -F width=576 -F num_frames=9 -F num_inference_steps=2 -F seed=42)
  echo "off-profile shape: http=$bad $(head -c 300 "$EVID/serve_${tag}_badshape.json")"
  kill $spid; sleep 5; kill -9 $spid 2>/dev/null; pkill -f 'difflet.cli.main serve' 2>/dev/null
  sleep 5
  echo "=== serve $tag end $(date -u +%FT%TZ)"
  return 0
}

serve_once fp8 --host-vae --quant fp8 --quant-granularity tensor --quant-act dynamic
echo "SERVE_FP8_RC=$?"
if [ -f "$ABF/fp8_run1_hostvae.mp4" ]; then
  python scripts/ptq_compare_outputs.py --reference "$ABF/fp8_run1_hostvae.mp4" --test "$EVID/serve_fp8_run0.mp4" --out "$EVID/compare_serve_fp8_vs_cli_fp8.json" 2>&1 | grep -vE 'Warning|warn' | tail -5
  python scripts/ptq_compare_outputs.py --reference "$EVID/serve_fp8_run0.mp4" --test "$EVID/serve_fp8_run1.mp4" --out "$EVID/compare_serve_fp8_run1_vs_run0.json" 2>&1 | grep -vE 'Warning|warn' | tail -5
  python scripts/ptq_compare_outputs.py --reference "$AB/bf16_run1_hostvae.mp4" --test "$EVID/serve_fp8_run0.mp4" --out "$EVID/compare_serve_fp8_vs_cli_bf16.json" 2>&1 | grep -vE 'Warning|warn' | tail -5
fi
serve_once bf16 --host-vae
echo "SERVE_BF16_RC=$?"
if [ -f "$AB/bf16_run1_hostvae.mp4" ]; then
  python scripts/ptq_compare_outputs.py --reference "$AB/bf16_run1_hostvae.mp4" --test "$EVID/serve_bf16_run0.mp4" --out "$EVID/compare_serve_bf16_vs_cli_bf16.json" 2>&1 | grep -vE 'Warning|warn' | tail -5
fi
grep -hE 'latency|request .* (took|completed)|generation .*s\b|\[serve\]' "$EVID"/serve_fp8.log "$EVID"/serve_bf16.log 2>/dev/null | tail -20
echo "SERVE_DONE"
