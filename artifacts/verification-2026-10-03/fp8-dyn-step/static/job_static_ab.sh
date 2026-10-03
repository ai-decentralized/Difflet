#!/usr/bin/env bash
# Static activation scales on the device: build the fp8-tensor-static checkpoint from the calibration
# JSON, compile + generate the Wan 2.1 fp8 arm (host VAE, 1 run), compare quality against the bf16
# reference and report the DiT step vs the dynamic law (607.2 ms) and bf16 (573 ms).
set -uo pipefail
T=/home/ubuntu/.claude/jobs/b5f130d0/tmp
W=/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan
cd "$W"
source .venv/bin/activate
export PYTHONPATH=$PWD
E=$W/artifacts/verification-2026-10-03/fp8-dyn-step/static
CAL=$E/act_calibration_wan21.json
[ -f "$CAL" ] || { echo "no calibration json"; exit 3; }
mkdir -p "$E"
bash scripts/gate_idle.sh | tee "$E/gate.txt" | tail -1 | grep -q IDLE || { echo "host busy"; exit 2; }
START=$(date -u +%s)
echo "=== static Wan 2.1 fp8 arm (quantize + compile + generate) $(date -u +%FT%TZ)"
python scripts/ptq_fp8_ab.py --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 \
  --tp-degree 4 --height 480 --width 832 --num-frames 9 --steps 20 --guidance-scale 1.0 --seed 42 \
  --runs 1 --host-vae --only fp8 --quant-calibration "$CAL" --out-dir "$E/ab" 2>&1 | grep -vE 'Warning|warn|import_nki' | tee "$E/ab.log" | tail -12
echo "AB_STATIC_RC=${PIPESTATUS[0]}"
grep -h 'dit-step ms\|dit-step-seconds\|Finished traced model weight initialization' "$E"/ab/logs/generate_*.log 2>/dev/null | sed 's|^|[static] |' | cut -c1-160
REF=$W/artifacts/verification-2026-10-01/ptq-wan21/ab/bf16_run1_hostvae.mp4
DYN=$W/artifacts/verification-2026-10-03/fp8-dyn-step/lean_round1/ab/fp8_run0.mp4
if [ -f "$E/ab/fp8_run0.mp4" ]; then
  echo "=== quality static vs bf16"
  python scripts/ptq_compare_outputs.py --reference "$REF" --test "$E/ab/fp8_run0.mp4" --out "$E/compare_static_vs_bf16.json" 2>&1 | grep -vE 'Warning|warn' | tail -6
  if [ -f "$DYN" ]; then
    echo "=== quality static vs dynamic-fp8"
    python scripts/ptq_compare_outputs.py --reference "$DYN" --test "$E/ab/fp8_run0.mp4" --out "$E/compare_static_vs_dyn.json" 2>&1 | grep -vE 'Warning|warn' | tail -6
  fi
fi
C=/var/tmp/neuron-compile-cache/neuronxcc-2.26.6360.0+6f180f47
mods=$(find "$C" -maxdepth 1 -name 'MODULE_*' -newermt "@$START" -printf '%f\n')
echo "new 2.26 modules: $mods"
[ -n "$mods" ] && for m in $mods; do python "$T/hlo_ops.py" "$C" "$m" 2>/dev/null > "$E/hlo_ops_$m.txt"; grep -c . "$E/hlo_ops_$m.txt"; done
echo "STATIC_DONE"
