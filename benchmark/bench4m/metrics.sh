#!/bin/bash
# Quality metrics for the Qwen-Image native text-mask investigation (2026-10-28; not in repo).
# Usage: metrics.sh LABEL REF_DIR_OR_FILE_LATENTS CAND_LATENTS [REF_FRAMES CAND_FRAMES]
# Host paths are settings (2026-11-03, moved into the repo): BENCH4M_DIR is the data root, the
# native venv's python scores, and quality_metrics.py sits next to this file.
B=${BENCH4M_DIR:?set BENCH4M_DIR (see benchmark/bench4m/README.md)}
PY=${BENCH4M_NATIVE_VENV:?set BENCH4M_NATIVE_VENV}/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p $B/nf_qwen_text/metrics
cd $B
label=$1; rl=$2; cl=$3; rf=$4; cf=$5
args=(--latents final "$rl" "$cl" --label "$label" --out "nf_qwen_text/metrics/$label.json")
[ -n "$rf" ] && args+=(--ref-frames "$rf" --cand-frames "$cf")
"$PY" "$HERE/quality_metrics.py" "${args[@]}" > /dev/null
/usr/bin/python3 - "nf_qwen_text/metrics/$label.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
lat = d["latents"]["final"]
fr = d.get("frames") or {}
def g(k):
    v = fr.get(k)
    return None if v is None else (round(v, 3) if isinstance(v, float) else v)
print(sys.argv[1].split("/")[-1], "cos", round(lat["cosine"], 4), "rel", round(lat["rel_l2"], 4),
      {k: g(k) for k in ("psnr_mean", "ssim_mean")} if fr else "")
EOF
