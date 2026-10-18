#!/usr/bin/env bash
# run_gates.sh: one model's GPU gate sequence ON the pod, mirroring the F3 / F5 / F1 / F1b runbooks:
# smoke -> DiT parity -> parity gate (pytest) -> e2e vs diffusers -> e2e gate (pytest).
# Usage:
#   cd ~/Difflet && bash .claude/skills/difflet-cuda-verify/scripts/run_gates.sh \
#       <qwen|flux|wan> <snapshot|auto> [--out ~/logs/<name>] [--dry-run]
#   detached (survives the ssh session):
#     setsid nohup bash .claude/skills/difflet-cuda-verify/scripts/run_gates.sh wan auto \
#       --out ~/logs/f9-wan > ~/logs/f9-wan.out 2>&1 < /dev/null &
# <snapshot>  $HF_HOME/hub/models--<org>--<name>/snapshots/<rev>/; HF_HOME is derived from it.
#             `auto` takes the first snapshot under $HF_HOME (default /data/hf when it exists,
#             else $HOME/hf). Download first (this runs with HF_HUB_OFFLINE=1), e.g.
#             HF_XET_HIGH_PERFORMANCE=1 hf download Wan-AI/Wan2.2-T2V-A14B-Diffusers
# --out       logs, result JSONs, images, references and caches (default ~/logs/<model>-<ts>);
#             keep logs under ~/logs, not /tmp (the gate scripts still create /tmp/logs).
# --dry-run   print the env and each step's command line, run nothing.
# Env per step: DIFFLET_BACKEND=cuda PATH=$REPO/.venv/bin:$PATH HF_HOME=... HF_HUB_OFFLINE=1
# PYTHONPATH=$REPO (CUDA_VISIBLE_DEVICES passes through; F1b pinned it to 0). Each step logs to
# <out>/<step>.log and appends "<step> exit: <rc> in <s>s" to <out>/gates.txt; a failed step does
# not stop the rest, and the exit code is 1 when any step failed. An early exit (no snapshot, no
# transformer/, no .venv) still appends "GATES DONE <model> overall_rc=2 (<reason>)" to gates.txt,
# so a poll on "GATES DONE" ends.
# qwen (F3): cuda_smoke.sh 50 steps; cuda_parity_gpu.py; cuda_e2e_vs_diffusers.py.
# flux (F5): cuda_smoke_flux.sh 28 steps; cuda_parity_flux_gpu.py; cuda_e2e_flux_vs_diffusers.py.
# wan (F1, F1b): cuda_smoke_wan.sh 40 steps x 9 frames; cuda_parity_wan_gpu.py --expert both (the
#   sdpa block and the 999/750 rounding block, references + compile cache under <out>);
#   cuda_e2e_wan_vs_diffusers.py --timestep-dtypes float32,model (float32 gated, the bf16 cast
#   recorded as the A/B).
set -u
case "${1:-}" in -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \{0,1\}//p;}' "$0"; exit 0 ;; esac
usage() { echo "usage: run_gates.sh <qwen|flux|wan> <snapshot|auto> [--out DIR] [--dry-run]" >&2; exit 2; }
MODEL=${1:-}; SNAP=${2:-}
case "$MODEL" in
  qwen) REPO_DIR=models--Qwen--Qwen-Image ;;
  flux) REPO_DIR=models--black-forest-labs--FLUX.1-dev ;;
  wan) REPO_DIR=models--Wan-AI--Wan2.2-T2V-A14B-Diffusers ;;
  *) usage ;;
esac
[ -n "$SNAP" ] || usage
shift 2
OUT=""; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --out) [ $# -ge 2 ] || usage; OUT=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) usage ;;
  esac
done
REPO=${REPO:-$PWD}
L=${OUT:-$HOME/logs/$MODEL-$(date +%Y%m%d-%H%M%S)}
# bail REASON: an early exit that still ends the poll loop (a real run writes the DONE marker)
bail() {
  echo "run_gates.sh: $1" >&2
  if [ "$DRY" = 0 ]; then
    mkdir -p "$L" && echo "GATES DONE $MODEL overall_rc=2 ($1)" >> "$L/gates.txt"
  fi
  exit 2
}

if [ "$SNAP" = auto ]; then
  if [ -z "${HF_HOME:-}" ]; then
    if [ -d /data/hf ]; then HF_HOME=/data/hf; else HF_HOME=$HOME/hf; fi
  fi
  SNAP=$(ls -d "$HF_HOME/hub/$REPO_DIR/snapshots/"*/ 2>/dev/null | head -1)
  [ -n "$SNAP" ] || bail "no $REPO_DIR snapshot under $HF_HOME/hub"
else
  case "${SNAP%/}" in
    */hub/models--*/snapshots/*)  # HF_HOME is four levels up from the snapshot
      HF_HOME=$(dirname "$(dirname "$(dirname "$(dirname "${SNAP%/}")")")") ;;
    *) HF_HOME=${HF_HOME:-$HOME/hf} ;;
  esac
  case "$SNAP" in */"$REPO_DIR"/*) ;; *) echo "run_gates.sh: warning: $SNAP is not a $REPO_DIR snapshot" >&2 ;; esac
fi
SNAP=${SNAP%/}/

export DIFFLET_BACKEND=cuda PATH="$REPO/.venv/bin:$PATH" HF_HOME HF_HUB_OFFLINE=1 PYTHONPATH="$REPO"
echo "model: $MODEL  snapshot: $SNAP  out: $L"
echo "env: DIFFLET_BACKEND=cuda PATH=$REPO/.venv/bin:\$PATH HF_HOME=$HF_HOME HF_HUB_OFFLINE=1 PYTHONPATH=$REPO"

overall=0
# shell-quote one word for the --dry-run line, only when it needs it
quote() {
  case "$1" in
    *[!A-Za-z0-9_./:=,@%+{}-]*) local sq="'"; printf "'%s'" "${1//$sq/$sq\\$sq$sq}" ;;
    *) printf '%s' "$1" ;;
  esac
}
# step NAME [VAR=value ...] command args...: one gate, its output in $L/NAME.log
step() {
  local name=$1 word; shift
  if [ "$DRY" = 1 ]; then
    printf '[step %s]' "$name"
    for word in "$@" ">" "$L/$name.log"; do printf ' '; [ "$word" = ">" ] && printf '>' || quote "$word"; done
    printf '\n'
    return 0
  fi
  SECONDS=0
  env "$@" > "$L/$name.log" 2>&1
  local rc=$?
  echo "$name exit: $rc in ${SECONDS}s" | tee -a "$L/gates.txt"
  [ "$rc" -eq 0 ] || overall=1
}

if [ "$DRY" = 0 ]; then
  [ -d "$SNAP/transformer" ] || bail "$SNAP has no transformer/"
  [ -x "$REPO/.venv/bin/python" ] || bail "no $REPO/.venv (run pod_setup.sh)"
  mkdir -p "$L" /tmp/logs
  cd "$REPO" || bail "cannot cd to $REPO"
  {
    echo "== $MODEL gates $(date -u +%FT%TZ) =="
    echo "snapshot: $SNAP"
    git log --oneline -1
    nvidia-smi --query-gpu=index,name,driver_version,memory.used --format=csv,noheader
    df -h "$HF_HOME" "$HOME" | tail -n +2
  } 2>&1 | tee -a "$L/gates.txt"
fi

GATE=(DIFFLET_RUN_CUDA_NUMERICAL=1)
PYTEST=(python -m pytest -v -rA -p no:cacheprovider)
case "$MODEL" in
  qwen)
    step smoke scripts/cuda_smoke.sh --prompt "a photorealistic cat sitting in a sunlit garden" \
      --output "$L/qwen.png" --steps 50 --seed 0
    step parity python tests/numerical/cuda_parity_gpu.py --model-path "$SNAP" \
      --out "$L/cuda_parity_qwen_image.json" --reference "$L/cuda_parity_reference.pt"
    step parity-gate "${GATE[@]}" DIFFLET_CUDA_PARITY_RESULT="$L/cuda_parity_qwen_image.json" \
      "${PYTEST[@]}" tests/numerical/test_cuda_vs_diffusers.py
    step e2e python scripts/cuda_e2e_vs_diffusers.py --model-path "$SNAP" \
      --out "$L/cuda_e2e_qwen_image.json" --images-dir "$L"
    step e2e-gate "${GATE[@]}" DIFFLET_CUDA_E2E_RESULT="$L/cuda_e2e_qwen_image.json" \
      "${PYTEST[@]}" tests/numerical/test_cuda_e2e_vs_diffusers.py
    ;;
  flux)
    step smoke scripts/cuda_smoke_flux.sh --prompt "a photorealistic cat sitting in a sunlit garden" \
      --output "$L/flux.png" --steps 28 --seed 0
    step parity python tests/numerical/cuda_parity_flux_gpu.py --model-path "$SNAP" \
      --out "$L/cuda_parity_flux.json" --reference "$L/cuda_parity_flux_reference.pt"
    step parity-gate "${GATE[@]}" DIFFLET_CUDA_FLUX_PARITY_RESULT="$L/cuda_parity_flux.json" \
      "${PYTEST[@]}" tests/numerical/test_cuda_flux_vs_diffusers.py
    step e2e python scripts/cuda_e2e_flux_vs_diffusers.py --model-path "$SNAP" \
      --out "$L/cuda_e2e_flux.json" --images-dir "$L"
    step e2e-gate "${GATE[@]}" DIFFLET_CUDA_FLUX_E2E_RESULT="$L/cuda_e2e_flux.json" \
      "${PYTEST[@]}" tests/numerical/test_cuda_flux_e2e_vs_diffusers.py
    ;;
  wan)
    step smoke CUDA_SMOKE_LOGDIR="$L" scripts/cuda_smoke_wan.sh \
      --prompt "a cat walking through a sunlit garden, photorealistic" --output "$L/wan.mp4" \
      --steps 40 --num-frames 9 --seed 0
    step parity python tests/numerical/cuda_parity_wan_gpu.py --model-path "$SNAP" --expert both \
      --out "$L/cuda_parity_wan.json" --reference "$L/cuda_parity_wan_reference_{expert}.pt" \
      --cache-dir "$L/cuda_parity_wan_cache"
    step parity-gate "${GATE[@]}" DIFFLET_CUDA_WAN_PARITY_RESULT="$L/cuda_parity_wan.json" \
      "${PYTEST[@]}" tests/numerical/test_cuda_wan_vs_diffusers.py
    step e2e python scripts/cuda_e2e_wan_vs_diffusers.py --model-path "$SNAP" \
      --timestep-dtypes float32,model --steps 40 --num-frames 9 --out "$L/cuda_e2e_wan.json" \
      --images-dir "$L" --compile-cache-dir "$L/cuda_e2e_wan_cache"
    step e2e-gate "${GATE[@]}" DIFFLET_CUDA_WAN_E2E_RESULT="$L/cuda_e2e_wan.json" \
      "${PYTEST[@]}" tests/numerical/test_cuda_wan_e2e_vs_diffusers.py
    ;;
esac
[ "$DRY" = 1 ] && exit 0
nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>&1 | tee -a "$L/gates.txt"
echo "GATES DONE $MODEL overall_rc=$overall" | tee -a "$L/gates.txt"
exit $overall
