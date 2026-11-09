#!/usr/bin/env bash
# Before/after quantization bench (docs/verification/2026-11-07-mxfp8-video-models.md, sections 11
# and 12): per model key, the bf16 arm then the MXFP8 arm, one after the other on one lane (one
# device's cores and one CPU set). Per arm:
#   1. compile into a fresh, empty cache dir (NEFF/HLO caches, the bf16 copy of an fp32 hub DiT and
#      the quantized directory all rebuilt);
#   2. e2e cold, twice: `generate` in a new process each time (report.py takes the lower: the host
#      is shared, and one-off contention made a single cold run unreliable in section 11's run 1);
#   3. e2e warm: one process, two identical requests (--requests); the second is the warm latency.
# Every step runs under /usr/bin/time -v and taskset, from the checkout this script is in, every
# log line timestamped (stamp.py), so stages without a timer of their own (quantization) can be
# timed afterwards.
#
# usage: bench.sh --cores <NEURON_RT_VISIBLE_CORES list> --cpus <taskset list> <key>...
#   keys: section 11's flux qwen_image ltx_2 wan21 wan22 hunyuan_video hunyuan_video_15 (their
#   phase G shapes and settings) and the official settings wan21_official wan22_official
#   hv15_official (docs/plans/2026-11-08-mxfp8-official-settings-rerun.md, D2).
# env: BENCH_ROOT (results: <key>/<arm>/{compile,cold_1,cold_2,warm}.log, outputs, cache_size.txt;
#      <key>/lane.txt; progress.txt), CACHE_ROOT (the per-arm compile caches, removed after each
#      arm), HF_HUB_CACHE (the hub snapshots), PY (the native-fp8 venv's python). Keep all three
#      directories on the local NVMe RAID: on the 125 MB/s EBS root compiles wait on I/O.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
BENCH_ROOT=${BENCH_ROOT:-/mnt/nvme/difflet-bench/results}
CACHE_ROOT=${CACHE_ROOT:-/mnt/nvme/difflet-bench/cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/mnt/nvme/difflet-bench/hub}
PY=${PY:-$HOME/venvs/native-fp8/bin/python}
# torch_neuronx spawns neuronx-cc by name: without the SDK's bin, every NEFF cache miss fails
# with "posix_spawn failed".
export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:$PATH
export HF_HUB_OFFLINE=1
unset DIFFLET_QUANT_TARGETS

CORES= CPUS=
while [ $# -gt 0 ]; do
  case $1 in
    --cores) CORES=$2; shift 2 ;;
    --cpus) CPUS=$2; shift 2 ;;
    -*) echo "bench.sh: unknown option $1" >&2; exit 2 ;;
    *) break ;;
  esac
done
if [ -z "$CORES" ] || [ -z "$CPUS" ] || [ $# -eq 0 ]; then
  echo "usage: bench.sh --cores <list> --cpus <list> <key>..." >&2
  exit 2
fi
export NEURON_RT_VISIBLE_CORES=$CORES

# The model cards' negative prompts (D2), byte for byte: Wan 2.1's (README.md:176 of 38ec498c) and
# Wan 2.2's (README.md:185 of 5be7df96, the same bytes as both Wan repos' sample_neg_prompt).
WAN21_NEG='Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards'
WAN22_NEG='色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
# HunyuanVideo-1.5 has no MXFP8 default (plan Task F8): its MXFP8 arms measure the former default,
# G1 + G3, `;`-joined in table order, so they resolve the same quantized directory as before.
HV15_MX_TARGETS='transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v);transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)'

args_for() {  # args_for <key>: M (model, shape, layout), P (sampling, no prompt), PROMPT, STEPS,
              # NEG (the request lines' negative prompt), QT (the MXFP8 arm's targets), EXT, KEEP
  NEG= QT= KEEP=() EXT=mp4 STEPS=50 PROMPT="a cat walking through a garden"
  case $1 in
    flux) M=(--model-id black-forest-labs/FLUX.1-dev --height 1024 --width 1024 --tp-degree 4)
      P=(--seed 42); PROMPT="a cat sitting on a bench"; EXT=png ;;
    qwen_image) M=(--model-id Qwen/Qwen-Image --height 1024 --width 1024 --tp-degree 4)
      P=(--seed 42 --guidance-scale 7.5); PROMPT="a cat sitting on a bench"; EXT=png ;;
    ltx_2) M=(--model-id Lightricks/LTX-2 --height 512 --width 768 --num-frames 121 --tp-degree 4)
      P=(--seed 42) ;;
    wan21) M=(--model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --height 480 --width 832 --num-frames 9
              --tp-degree 4)
      P=(--seed 42) ;;
    wan22) M=(--model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers --dual-expert --height 480 --width 832
              --num-frames 9 --tp-degree 4)
      P=(--seed 42) ;;
    hunyuan_video) M=(--model-id hunyuanvideo-community/HunyuanVideo
                      --revision e8c2aaa66fe3742a32c11a6766aecbf07c56e773
                      --height 320 --width 512 --num-frames 61 --tp-degree 4)
      P=(--seed 42 --guidance-scale 6.0)
      PROMPT="a cinematic shot of a red fox running through a snowy forest" ;;
    hunyuan_video_15) M=(--model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v
                         --num-frames 9)
      P=(--seed 42); QT=$HV15_MX_TARGETS ;;
    # The official settings (D2). CFG runs as two sequential forwards on the tp4 group (D3); the
    # first cold run keeps its work dir, so the final latents stay for the CPU fp32 scores (Task 8).
    wan21_official) M=(--model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --height 480 --width 832
                       --num-frames 81 --tp-degree 4)
      P=(--seed 42 --guidance-scale 5.0); NEG=$WAN21_NEG; KEEP=(--keep-work-dir) ;;
    wan22_official) M=(--model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers --dual-expert --height 480
                       --width 832 --num-frames 81 --tp-degree 4)
      P=(--seed 42 --guidance-scale 4.0 --guidance-scale-2 3.0); NEG=$WAN22_NEG; STEPS=40
      KEEP=(--keep-work-dir) ;;
    hv15_official) M=(--model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v
                      --height 480 --width 848 --num-frames 121)
      P=(--seed 42); QT=$HV15_MX_TARGETS; KEEP=(--keep-work-dir) ;;
    *) echo "bench.sh: unknown key $1" >&2; exit 2 ;;
  esac
  P=(--steps "$STEPS" "${P[@]}")
  if [ -n "$NEG" ]; then P+=(--negative-prompt "$NEG"); fi
}

timed() {  # timed <log> <cmd...>: /usr/bin/time -v under the lane's taskset, from the checkout,
           # lines timestamped; appends "exit <rc>" and returns the command's exit status
  local log=$1; shift
  ( cd "$REPO" && env PYTHONPATH="$REPO" /usr/bin/time -v taskset -c "$CPUS" "$@" 2>&1 \
      | "$PY" "$HERE/stamp.py"; exit "${PIPESTATUS[0]}" ) > "$log" 2>&1
  local rc=$?
  echo "exit $rc" >> "$log"
  return $rc
}

mark() { echo "$(date '+%F %T') lane $CORES $*" >> "$BENCH_ROOT/progress.txt"; }

warm_hub() {  # read the model's hub files through, so every arm starts page-cached whatever
              # evicted them in between (other jobs' model loads do, even with RAM free)
  local repo=${M[1]}
  find -L "$HF_HUB_CACHE/models--${repo//\//--}/snapshots" -type f -exec cat {} + > /dev/null
}

request_line() {  # request_line <output>: one --requests line, JSON-encoded by python
  "$PY" -c 'import json, sys
line = {"prompt": sys.argv[1], "seed": 42, "steps": int(sys.argv[2]), "output": sys.argv[3]}
if sys.argv[4]:
    line["negative_prompt"] = sys.argv[4]
print(json.dumps(line, ensure_ascii=False))' "$PROMPT" "$STEPS" "$1" "$NEG"
}

for key in "$@"; do args_for "$key"; done  # every key known before anything is written
mkdir -p "$BENCH_ROOT" "$CACHE_ROOT"
for key in "$@"; do
  args_for "$key"
  mkdir -p "$BENCH_ROOT/$key"
  printf 'cores %s\ncpus %s\n' "$CORES" "$CPUS" > "$BENCH_ROOT/$key/lane.txt"
  for arm in bf16 mx; do
    W=$BENCH_ROOT/$key/$arm
    mkdir -p "$W"
    warm_hub
    C=$CACHE_ROOT/$key-$arm
    rm -rf "$C"; mkdir -p "$C"
    QF=() QENV=()
    if [ $arm = mx ]; then
      QF=(--quant mxfp8)
      if [ -n "$QT" ]; then QENV=(DIFFLET_QUANT_TARGETS="$QT"); fi  # its three commands only
    fi
    mark "$key $arm compile"
    if timed "$W/compile.log" env "${QENV[@]}" "$PY" -m difflet.cli.main compile --backend neuron \
        "${M[@]}" "${QF[@]}" --cache-dir "$C"; then
      for i in 1 2; do
        mark "$key $arm cold_$i"
        keep=(); if [ $i = 1 ]; then keep=("${KEEP[@]}"); fi
        timed "$W/cold_$i.log" env "${QENV[@]}" "$PY" -m difflet.cli.main generate --backend neuron \
          "${M[@]}" "${QF[@]}" --cache-dir "$C" "${P[@]}" --prompt "$PROMPT" \
          --output "$W/cold_$i.$EXT" --work-dir "$W/work_cold_$i" "${keep[@]}"
      done
      : > "$W/requests.jsonl"
      for i in 1 2; do request_line "$W/warm_$i.$EXT" >> "$W/requests.jsonl"; done
      mark "$key $arm warm"
      timed "$W/warm.log" env "${QENV[@]}" "$PY" -m difflet.cli.main generate --backend neuron \
        "${M[@]}" "${QF[@]}" --cache-dir "$C" "${P[@]}" --requests "$W/requests.jsonl" \
        --work-dir "$W/work_warm"
    else
      mark "$key $arm COMPILE FAILED (see $W/compile.log)"
    fi
    du -sh "$C" > "$W/cache_size.txt" 2>&1
    du -sh "$C"/* >> "$W/cache_size.txt" 2>&1
    rm -rf "$C"  # per arm: fresh caches for every lane must fit the free disk
    mark "$key $arm done"
  done
done
