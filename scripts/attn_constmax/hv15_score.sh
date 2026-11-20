#!/usr/bin/env bash
# HunyuanVideo-1.5 lane quality: each arm's second (warm) request against arm A's, and D against C
# (const-max on top of MXFP8), with scripts/compare_difflet_runs.py (clip PSNR mean and worst
# frame, SSIM, LPIPS; final latents cosine / rel L2). Also A against the evidence's bf16 clip
# (docs/verification/2026-11-07-mxfp8-video-models.md 9b `bf16_pre`, d3ef0f1c, 96 encode threads).
# No CPU fp32 reference exists at 480x848x121. Prints one line per pair.
#
# usage: hv15_score.sh <cpus>   env: ROOT (default /mnt/nvme/difflet-exp/constmax-models/hunyuan_video_15)
set -u
CPUS=$1
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-/mnt/nvme/difflet-exp/constmax-models/hunyuan_video_15}
PY=${PY:-$HOME/venvs/native-fp8/bin/python}
EV=/home/ubuntu/fp8-bench/hv15/bf16_pre
S=$ROOT/scores
mkdir -p "$S"
lat() { find "$ROOT/$1/work" -name 'latents_req0001.pt' | head -1; }
cmp() {  # cmp <name> <ref latents> <ref mp4> <cand latents> <cand mp4>
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 taskset -c "$CPUS" "$PY" "$HERE/../compare_difflet_runs.py" \
    --ref-latents "$2" --ref-image "$3" --cand-latents "$4" --cand-image "$5" --out "$S/$1.json" \
    > "$S/$1.log" 2>&1 || echo "FAILED $1 (see $S/$1.log)"
}
for a in B C D; do
  [ -f "$ROOT/$a/warm_2.mp4" ] && cmp "${a}_vs_A" "$(lat A)" "$ROOT/A/warm_2.mp4" "$(lat $a)" "$ROOT/$a/warm_2.mp4"
done
[ -f "$ROOT/D/warm_2.mp4" ] && cmp D_vs_C "$(lat C)" "$ROOT/C/warm_2.mp4" "$(lat D)" "$ROOT/D/warm_2.mp4"
cmp A_vs_evidence_bf16 "$EV/work_a/latents.pt" "$EV/gena.mp4" "$(lat A)" "$ROOT/A/warm_2.mp4"
"$PY" - "$S" <<'EOF'
import json, sys
from pathlib import Path
for f in sorted(Path(sys.argv[1]).glob("*_vs_*.json")):
    d = json.loads(f.read_text()); v = d.get("video") or {}; l = d.get("latents") or {}
    print(f"{f.stem}: psnr {v.get('psnr'):.2f} (worst {v.get('psnr_worst'):.2f}) ssim {v.get('ssim'):.4f} "
          f"lpips {v.get('lpips'):.4f} latent cos {l.get('cosine')} rel_l2 {l.get('rel_l2')}")
EOF
