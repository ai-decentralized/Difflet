#!/usr/bin/env bash
# Re-run the step-caching evaluation at each model's official default step count,
# with every component that has a Trainium implementation running on device.
#
# Paper context: the caching table was measured at 20 steps for every video
# model. The official defaults are HunyuanVideo 50, Wan 2.1 T2V-14B 50, LTX-2 40
# (LTX-2.0, stage 1). Resolutions stay exactly as benchmark/models.py has them --
# the official resolutions do not fit one trn2.3xlarge.
#
# COMPONENT PLACEMENT -- read before trusting any number this produces:
#
#   Wan 2.1       Fully on device, at the largest frame count its VAE decoder
#                 compiles at. The VAE binds, not the transformer: tracing
#                 flattens the per-latent-frame loop in
#                 difflet/models/wan/vae/modeling_vae.py:487, so the graph grows
#                 with frames against neuronx-cc's 10,000,000-instruction ceiling
#                 (NCC_EVRF007). Measured on this host, 2026-09-23, at 480x832:
#                 81 frames (21 latent) 39,093,968; 33 (9) 15,719,922;
#                 25 (7) 11,823,804; 21 (6) 11,164,146. The transformer compiles
#                 at 81 — its attention is tiled, so it does not unroll.
#                 DIFFLET_WAN_FRAMES sets the count; see
#                 scripts/jobs/wan_full_device.sh.
#   HunyuanVideo  Fully on device. Staged clip -> llama -> generate; the generate
#                 stage builds the app with enable_vae_decoder=True, so the
#                 transformer and the Neuron VAE decoder are device components
#                 and both text encoders are their own device stages.
#   LTX-2         NOT fully on device, and no flag changes that. Its Trainium
#                 backend holds only transformer.py and teacache_probe_fused.py
#                 -- there is no Neuron text encoder and no Neuron VAE for LTX-2
#                 in this tree -- and difflet/cli/orchestrators/ltx_2.py:141
#                 hardcodes enable_host_pipeline/enable_decode_components=True.
#                 Its text encode and VAE decode run on the host CPU. Putting
#                 them on device is a code change, not a configuration change.
#
# All three adaptive probes ARE on device as of 0f9ef0f: that commit added the
# stateful Wan and LTX-2 device probes and gave every probe its own weight store
# (shared_weights_layout = "teacache-prefix-v1", loading prefix weights only), so
# the Hunyuan probe no longer duplicates backbone residency. The paper's
# host-CPU-shadow footnote for Wan and LTX-2, and its claim that HunyuanVideo
# cannot run the adaptive mode, both predate that commit and need re-measuring.
#
# Step count is a generate-time argument and is not part of the NEFF shape key
# (difflet/cli/main.py: --steps lives in _add_generate_flags, shapes in
# _add_shape_flags), so this reuses existing compiled artifacts. Only the
# calibrated-adaptive mode needs new calibration, because num_steps is part of
# the calibration contract (difflet/pipeline/teacache.py sync_probe_free_num_steps
# is a no-op for adaptive controllers).
#
#   bash scripts/rerun_caching_official_steps.sh              # all three models
#   bash scripts/rerun_caching_official_steps.sh hunyuan      # one model
#   DIFFLET_RERUN_COMPILE=1 bash scripts/rerun_caching_official_steps.sh
#   DIFFLET_RERUN_REPEATS=3 bash scripts/rerun_caching_official_steps.sh
#
# Outputs land in cclogs/caching-official-steps/<model>/<mode>/ as the generated
# video plus the run log; scripts/psnr_compare.py turns them into the quality
# column.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Recent Neuron DLAMIs no longer ship /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference,
# so prefer the repo venv that scripts/setup_env.sh builds from the lock file.
if [[ -n "${DIFFLET_NEURON_VENV:-}" ]]; then
  NEURON_VENV="${DIFFLET_NEURON_VENV}"
elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
  NEURON_VENV="${ROOT}/.venv"
elif [[ -x /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python ]]; then
  NEURON_VENV=/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference
else
  echo "error: no Neuron venv found. Run ./scripts/setup_env.sh first," >&2
  echo "       or point DIFFLET_NEURON_VENV at an existing one." >&2
  exit 1
fi
export PATH="${NEURON_VENV}/bin:${PATH}"
export PYTHONPATH="${ROOT}"
export DIFFLET_BACKEND=trainium
export NEURON_RT_VIRTUAL_CORE_SIZE=2
export NEURON_RT_NUM_CORES=4
cd "${ROOT}"

OUT_ROOT="${DIFFLET_RERUN_OUT:-${ROOT}/cclogs/caching-official-steps}"
CALIB_DIR="${DIFFLET_RERUN_CALIB_DIR:-${ROOT}/cclogs/m9-teacache}"
REPEATS="${DIFFLET_RERUN_REPEATS:-3}"
SEED="${DIFFLET_RERUN_SEED:-42}"
PROMPT="${DIFFLET_RERUN_PROMPT:-a cinematic shot of a red fox running through a snowy forest}"
ONLINE_ALPHA="${DIFFLET_RERUN_ALPHA:-0.6}"
DO_COMPILE="${DIFFLET_RERUN_COMPILE:-0}"

# The implementation keeps this many full steps at each end in every mode.
WARMUP=5
COOLDOWN=5

# model | model-id | shape flags | steps | output extension | extra generate flags
#
# Steps and Wan's frame count are each model's official default; the resolutions
# stay as benchmark/models.py has them, because the official ones do not fit one
# trn2.3xlarge. Wan's 9-frame entry there is a fast smoke shape (3 latent frames,
# ~4.7k attention tokens); 81 frames is 21 latent frames and ~33k tokens, which is
# the regime a caching result should be claimed in.
# Wan's frame count is the largest whose VAE decoder compiles on device; see
# scripts/jobs/wan_full_device.sh for the measured ceiling and why it binds.
# Wan's official default. 9 was the old smoke shape, kept only by the VAE's
# single-shot compile ceiling; the chunked decoder lifts that, so 81 is the
# number a caching result should be claimed at.
WAN_FRAMES="${DIFFLET_WAN_FRAMES:-81}"
# The VAE decoder has no device build at 81 frames: its traced graph exceeds
# neuronx-cc's 5,000,000-instruction ceiling, and the alternatives are closed on
# this stack (torch_xla eager compiles per operator; torch.compile's openxla
# backend fails on silu; torch_xla.compile hits NCC_INLA001 inside the compiler).
# DIFFLET_WAN_HOST_VAE=1 decodes on the host instead. That leaves the caching
# columns intact -- the cache acts inside the denoise loop and the decode runs
# once outside it -- and costs only the end-to-end column, which must say so.
WAN_VAE_FLAG=""
[[ -n "${DIFFLET_WAN_HOST_VAE:-}" ]] && WAN_VAE_FLAG="--host-vae"
read -r -d '' MODELS <<EOF
hunyuan|hunyuanvideo-community/HunyuanVideo|--height 320 --width 512 --num-frames 61|50|mp4|--guidance-scale 6.0
wan|Wan-AI/Wan2.1-T2V-14B-Diffusers|--height 480 --width 832 --num-frames ${WAN_FRAMES}|50|mp4|--guidance-scale 1.0 ${WAN_VAE_FLAG}
ltx2|Lightricks/LTX-2|--height 480 --width 704 --num-frames 49|40|mp4|--guidance-scale 1.0
EOF

steps_for() {
  while IFS='|' read -r name _ _ steps _ _; do
    [[ "${name}" == "$1" ]] && { echo "${steps}"; return; }
  done <<<"${MODELS}"
}

# Cadence 2's skip budget, which the adaptive threshold is matched to so the
# modes differ in WHICH steps they cache, not how many.
cadence2_target_speedup() {
  local steps="$1"
  python3 -c "
steps, warm, cool = ${steps}, ${WARMUP}, ${COOLDOWN}
window = steps - warm - cool
skipped = window // 2
print(f'{steps / (steps - skipped):.3f}')
"
}

# "off" is the reference row, not a caching mode: PSNR is measured against it and
# it is the denominator of every loop speedup, so it always runs.
# cadence2 and online-delta are the two probe-free modes (runtime flags, no extra
# graph); adaptive needs a fitted calibration and is opt-in via DIFFLET_RERUN_MODES.
modes_for() { echo "${DIFFLET_RERUN_MODES:-off cadence2 online adaptive}"; }

mode_flags() {
  local model="$1" mode="$2" steps="$3"
  case "${mode}" in
    off)      echo "" ;;
    cadence2) echo "--teacache-cadence 2" ;;
    online)   echo "--teacache-online-delta ${ONLINE_ALPHA}" ;;
    adaptive)
      local calib="${CALIB_DIR}/teacache_calib_${model}_${steps}steps.json"
      if [[ ! -f "${calib}" ]]; then
        echo "MISSING_CALIB:${calib}"
      else
        echo "--teacache-speedup $(cadence2_target_speedup "${steps}") --teacache-calibration ${calib}"
      fi
      ;;
  esac
}

# Compile every component this model can put on device. For Wan and HunyuanVideo
# this walks their stages; LTX-2 compiles its transformer only, by construction.
compile_model() {
  local name="$1" model_id="$2" shape="$3"
  echo "===== [rerun] compiling ${name} (all device components) ====="
  # shellcheck disable=SC2086
  difflet compile --model-id "${model_id}" --tp-degree 4 ${shape} 2>&1 \
    | tee "${OUT_ROOT}/${name}/compile.log"
  echo "[rerun] ${name} compile exit=${PIPESTATUS[0]}"
}

# Say so when a run used a host component, rather than letting a host-pipeline
# number be reported as a Trainium one.
check_placement() {
  local name="$1" log="$2"
  if grep -qiE "host pipeline|host CPU|shadow" "${log}"; then
    echo "[rerun] NOTE ${name}: this run touched a host component -- see ${log}" >&2
    if [[ "${name}" == "ltx2" ]]; then
      echo "[rerun]   expected for LTX-2 (host text encode + host VAE decode)" >&2
    else
      echo "[rerun]   NOT expected for ${name}: investigate before using the number" >&2
    fi
  fi
}

run_one() {
  local name="$1" model_id="$2" shape="$3" steps="$4" ext="$5" extra="$6" mode="$7" rep="$8"
  local flags; flags="$(mode_flags "${name}" "${mode}" "${steps}")"
  if [[ "${flags}" == MISSING_CALIB:* ]]; then
    echo "[rerun] SKIP ${name}/${mode}: no calibration at ${flags#MISSING_CALIB:}" >&2
    echo "[rerun]   fit one first -- see the calibration note at the end of this script" >&2
    return 0
  fi

  local dir="${OUT_ROOT}/${name}/${mode}"
  mkdir -p "${dir}"
  local out="${dir}/rep${rep}.${ext}"
  local log="${dir}/rep${rep}.log"

  echo "===== [rerun] ${name} / ${mode} / rep ${rep} (${steps} steps) =====" | tee "${log}"
  local start; start=$(date +%s.%N)
  # The loop column counts the steps the cache skipped, so it cannot come from a
  # DiT-forward benchmark (executed calls only) nor from subtracting two
  # load-dominated end-to-end runs. difflet/pipeline/step_timing.py times the
  # scheduler loop itself; it is inert unless DIFFLET_STEP_TIMING is set.
  export DIFFLET_STEP_TIMING=1
  export DIFFLET_STEP_TIMING_OUT="${dir}/rep${rep}.steptiming.json"
  # shellcheck disable=SC2086
  difflet generate \
    --model-id "${model_id}" \
    --tp-degree 4 \
    ${shape} \
    --steps "${steps}" \
    --seed "${SEED}" \
    --prompt "${PROMPT}" \
    ${extra} \
    ${flags} \
    --output "${out}" 2>&1 | tee -a "${log}"
  local rc=${PIPESTATUS[0]}
  local end; end=$(date +%s.%N)

  check_placement "${name}" "${log}"

  python3 - "$log" "$start" "$end" "$rc" "$mode" "$steps" >"${dir}/rep${rep}.json" <<'PY'
import json, sys
log, start, end, rc, mode, steps = sys.argv[1:7]
print(json.dumps({
    "log": log, "mode": mode, "steps": int(steps),
    "wall_s": round(float(end) - float(start), 3), "exit_code": int(rc),
}))
PY
  echo "[rerun] ${name}/${mode}/rep${rep} exit=${rc}"
}

WANT="${1:-all}"
while IFS='|' read -r name model_id shape steps ext extra; do
  [[ -z "${name}" ]] && continue
  [[ "${WANT}" != "all" && "${WANT}" != "${name}" ]] && continue
  mkdir -p "${OUT_ROOT}/${name}"
  if [[ "${DO_COMPILE}" == "1" ]]; then
    compile_model "${name}" "${model_id}" "${shape}"
  fi
  for mode in $(modes_for "${name}"); do
    for rep in $(seq 1 "${REPEATS}"); do
      run_one "${name}" "${model_id}" "${shape}" "${steps}" "${ext}" "${extra}" "${mode}" "${rep}"
    done
  done
done <<<"${MODELS}"

cat <<'NOTE'

===== next steps =====

1. Confirm every component really loaded on device, per model:

     difflet cache ls --json | python3 -m json.tool | less

   Wan must show text_encoder, transformer and vae_decoder; HunyuanVideo must
   show clip, llama, transformer and vae_decoder. LTX-2 will show transformer
   (plus teacache_probe) only -- that is the known gap, not a compile failure.

2. Quality (PSNR/SSIM), against each model's own caching-off output at the SAME
   step count -- the 20-step references are not valid here:

     python scripts/psnr_compare.py --sweep cclogs/caching-official-steps/wan

3. Pure DiT cost per call (caching-independent, measure once per model). The
   e2e wall above is load-dominated and noisy -- benchmark/step_latency.py's
   header documents an LTX-2 pair that measured 293 s and 628 s for the same
   config -- so never derive a per-step number by subtracting two e2e runs:

     python -m benchmark.step_latency --model hunyuan_video
     python -m benchmark.step_latency --model wan_2_1
     python -m benchmark.step_latency --model ltx_2

4. Calibration for the adaptive rows. No calibration JSON exists in this tree
   yet, so every adaptive row will SKIP until one is fitted, per model, AT THE
   NEW STEP COUNT (num_steps is part of the calibration contract):

     python scripts/calibrate_teacache.py --pairs-json <recorded pairs> \
       --out cclogs/m9-teacache/teacache_calib_wan_50steps.json

   Record the pairs on prompts OTHER than the benchmark prompt, as the paper's
   method paragraph states.

5. The device probes are new (0f9ef0f). Smoke-test them before a long sweep:

     python tests/manual/check_wan_ltx2_teacache_probe.py --model wan --tp-degree 1
     python tests/manual/check_wan_ltx2_teacache_probe.py --model ltx_2 --tp-degree 2

Expected loop speedup at cadence 2, with the 5 warm-up and 5 cool-down full
steps the implementation keeps: 50 steps skips 20 -> 1.67x; 40 steps skips 15
-> 1.60x. The old 20-step rows could only reach 1.33x.
NOTE
