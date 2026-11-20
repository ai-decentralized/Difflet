#!/usr/bin/env bash
# Experiment E1 quality: each arm's second (warm) request against the CPU fp32 reference, and each
# const-max arm against the bf16 arm (scripts/compare_difflet_runs.py: PSNR (frame mean and worst)
# / SSIM / LPIPS on the output, rel L2 / cosine on the final packed latents). One line per pair.
#
# usage: score.sh <model> <cpus> <arm>...   (the first arm is bf16, the baseline; <model> is the
#   directory under ROOT: flux and qwen_image score png, the video models mp4)
# env: ROOT (default /mnt/nvme/difflet-exp/constmax), REF (default /home/ubuntu/fp8-bench/fp32_ref/<model>),
#      SCORES (default ROOT/<model>/scores)
set -u
MODEL=$1 CPUS=$2; shift 2
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
ROOT=${ROOT:-/mnt/nvme/difflet-exp/constmax}
F=$ROOT/$MODEL S=${SCORES:-$ROOT/$MODEL/scores} R=${REF:-/home/ubuntu/fp8-bench/fp32_ref/$MODEL}
PY=${PY:-$HOME/venvs/native-fp8/bin/python}
EXT=mp4; case $MODEL in flux|qwen_image) EXT=png ;; esac
mkdir -p "$S"
out() { echo "$F/$1/warm_2.$EXT"; }
lat() { echo "$F/$1/work/worker_0/latents_req0001.pt"; }
cmp() {  # cmp <name> <ref latents> <ref output> <cand latents> <cand output>
  local lat_args=()
  [ -f "$2" ] && [ -f "$4" ] && lat_args=(--ref-latents "$2" --cand-latents "$4")
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 taskset -c "$CPUS" "$PY" "$REPO/scripts/compare_difflet_runs.py" \
    "${lat_args[@]}" --ref-image "$3" --cand-image "$5" --out "$S/$1.json" > "$S/$1.log" 2>&1 \
    || echo "FAILED $1 (see $S/$1.log)"
}
REF_OUT=$R/out.$EXT
for a in "$@"; do cmp "${a}_vs_fp32" "$R/latents.pt" "$REF_OUT" "$(lat "$a")" "$(out "$a")"; done
base=$1; shift
for a in "$@"; do cmp "${a}_vs_$base" "$(lat "$base")" "$(out "$base")" "$(lat "$a")" "$(out "$a")"; done
"$PY" - "$S" <<'EOF'
import json, sys
from pathlib import Path
for f in sorted(Path(sys.argv[1]).glob("*_vs_*.json")):
    d = json.loads(f.read_text()); i = d.get("image") or d.get("video") or {}
    l = d.get("latents") or {}
    l = l.get("video", l)  # LTX-2: video and audio latents; the video's
    worst = f" (worst {i['psnr_worst']:.2f})" if i.get("psnr_worst") is not None else ""
    print(f"{f.stem}: psnr {i.get('psnr'):.2f}{worst} ssim {i.get('ssim'):.4f} lpips {i.get('lpips'):.4f} "
          f"latent cos {l.get('cosine')} rel_l2 {l.get('rel_l2')}")
EOF
