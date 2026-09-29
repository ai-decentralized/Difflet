# FP8 PTQ of the Wan DiT — on-device verification plan

**Target:** Wan 2.1 14B (`Wan-AI/Wan2.1-T2V-14B-Diffusers`) on a trn2 host, tp4.
**Design:** `docs/superpowers/specs/2026-09-29-ptq-fp8-linear-design.md`.
**Evidence doc to write:** `docs/verification/<date>-ptq-fp8-wan-evidence.md`
(`git add -f`; `docs/` is gitignored but tracked by precedent). Follow the
`difflet-device-verify` skill: supervise long jobs, checkpoint after every phase,
one bug = one commit, quote diagnostics verbatim.

Everything below is a hypothesis until the device says otherwise; the plan is
ordered so the cheap checks (minutes) run before the expensive ones (hours).

## What is being verified

| id | claim | where it is decided |
|---|---|---|
| A1 | fp8 `weight` + `scale` (`[1]` / `[out, 1]`) load into NxD's quantized parallel linears | Phase 0 |
| A2 | NxD `DYNAMIC` activation quantization = per-tensor absmax (device-fp8 ≈ CPU-fp8) | Phase 0, 2 |
| A3 | neuronx-cc runs the fp8 layers as tensor-engine FP8 (per-step drops) | Phase 3 |
| A4 | fp8 tensors survive sharding / presharded safetensors / shared store | Phase 0, 1 |
| A5 | the `--experimental-unsafe-fp8e4m3fn-as-fp8e4m3` flag reaches the compiler | Phase 0 (compiler_args in the report) |
| Q  | FP8 output quality vs bf16: latent cosine/MSE/SNR, PSNR/SSIM/LPIPS | Phase 2 |
| P  | compile time, e2e cold / warm, DiT per-step, bf16 vs fp8 | Phase 3 |
| S  | `difflet serve --quant fp8` serves the same result | Phase 4 |

## Phase 0 — host and tiny probe (≈ 15 min)

```bash
cd <repo> && git checkout quantization
scripts/env_check.sh                       # cores idle, venv, HF token, disk
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate   # or <repo>/.venv
PYTHONPATH=$PWD pytest tests/unit/quant tests/unit/cli/test_cli_quant.py \
    tests/unit/serving/test_serve_quant.py tests/unit/test_ptq_scripts.py -q
PYTHONPATH=$PWD python scripts/ptq_fp8_device_probe.py --work-dir /tmp/ptq_probe --force-clean
```

The unit run includes `test_wan_application_resolves_and_ensures_quantized_checkpoints`,
which is skipped on hosts without `neuronx_distributed` and must pass here.

The probe writes `/tmp/ptq_probe/ptq_probe_report.json`. Read, in order:

1. `arms.fp8.compiled` / `arms.fp8.loaded` — A1, A4, A5. A load-time shape error
   on a `.scale` parameter means NxD wants a different scale layout: fix it in
   `difflet/quant/checkpoint.py::quantize_state_dict` (one reshape) and rerun.
   `arms.fp8.compiler_args` must contain `--experimental-unsafe-fp8e4m3fn-as-fp8e4m3`.
2. `checks.device_fp8_vs_cpu_fp8.cosine` ≥ 0.999 — A2. If device-fp8 is instead
   closer to `cpu_bf16` than to `cpu_fp8`, the layers compiled as dequant-to-bf16
   (weight-only), or NxD's activation scaling differs; note it, it decides how to
   read Phase 3.
3. `checks.device_bf16_vs_cpu_bf16.cosine` — the control; it bounds what
   "accumulation-order noise" means on this toolchain.
4. `cpu_fp8_vs_cpu_bf16` — the algorithm's own error on the tiny model.

Record the JSON and the two compiler command lines in the evidence doc. Do not
continue to Phase 1 with a red probe: everything downstream would be attributing
a plumbing bug to quantization error.

Disk: the probe is tiny. Wan 14B needs: bf16 weights (present), one fp8 checkpoint
copy (~14 GB), one more compile-cache entry per fp8 configuration (presharded
shards at half the bf16 size).

## Phase 1 — quantize + compile, timed (hours: the fp8 compile is a full DiT compile)

```bash
PYTHONPATH=$PWD python scripts/ptq_fp8_ab.py \
    --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --tp-degree 4 \
    --height 480 --width 832 --num-frames 9 --steps 20 --guidance-scale 1.0 --seed 42 \
    --runs 2 --drop-caches --out-dir artifacts/verification-<date>/ptq-wan21
```

Run it under `scripts/supervise.sh` in a Monitor. It performs, logging each
subprocess under `<out-dir>/logs/`:

- `difflet quantize --quant fp8 --quant-granularity tensor` (CPU; the fp8
  checkpoint lands under `~/.cache/difflet/quantized/Wan-AI--Wan2.1-…/transformer/fp8-tensor-<hash>/`,
  manifest `difflet_quant.json` with the layer count and byte sizes);
- `difflet compile` for the bf16 arm (cache hit if already compiled: the log
  says "already compiled … skipping") and for the fp8 arm (new artifact — the
  transformer stage identity carries the `quant` block, the VAE stage is shared);
- two `difflet generate` per arm with `--keep-work-dir` (latents kept), page
  cache dropped before run 0 of each arm when `sudo -n` works;
- the comparisons of Phase 2 and the table of Phase 3.

Compile-time evidence: `arms.fp8.compile_seconds` vs `arms.bf16.compile_seconds`
(only meaningful when neither was a cache hit; `--skip-compile` reruns later
phases on warm caches). The neuronx-cc per-component breakdown comes from
`python -m benchmark.parse_compile <out-dir>/logs/compile_fp8.log`.

Fingerprints to record (A4): `ls -la` of the fp8 artifact's `weights/` (shard
sizes ≈ half of the bf16 shards), the `_shared_weights` store entry named with
`qf8e4m3` (`difflet/backends/trainium/core/shared_weights.py::store_label`), and
the stage `manifest.json` showing `"quant": {...}`.

## Phase 2 — accuracy (Q, A2 at scale)

**2a. Per-linear matmul error (CPU, real weights).** Runs on any CPU box with
the snapshot (28 GB RAM for the full model; `--max-blocks` limits it):

```bash
PYTHONPATH=$PWD python scripts/ptq_linear_error_sweep.py \
    --model-dir ~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-14B-Diffusers/snapshots/<sha> \
    --height 480 --width 832 --num-frames 9 --timestep 500 --m-slice 1024 \
    --out artifacts/verification-<date>/ptq-wan21/linear_error_t500.json
```

Repeat at `--timestep 900` and `--timestep 100` (the activation range moves
with the noise level). Report per scheme: min/mean cosine, max rel-L2, min
SNR, and the worst cells. Expect `fp8-*-wo` (weight-only) to sit between the
bf16 floor and `fp8-*-dyn`; cells whose cosine falls far below the rest
(cross-attention `to_k`/`to_v` on text tokens are the usual suspects) are the
candidates for keeping in bf16 if the output metrics disappoint.

**2b. Latent- and pixel-level error (device).** From the A/B summary,
`compare.fp8_vs_bf16_run0` and `_run1`:

- latents (DiT output before the VAE): cosine, MSE, rel-L2, SNR dB;
- decoded video: PSNR, SSIM, LPIPS (`pip install lpips` in the venv first,
  otherwise the field is `null`);
- `compare.bf16_run1_vs_run0_control` must be identical (PSNR inf): if it is
  not, the run is not deterministic and every other number needs more runs.

There is no pass threshold to inherit: FastVideo publishes no PTQ-vs-bf16
number for FP8 linear PTQ (its only figure is MS-SSIM 0.907 for INT8 weight-only
PTQ of the 1.3B model vs its own FP16 output, and 0.933 after QAD). Report the
measured values; a PSNR above ~30 dB / SSIM above ~0.9 vs bf16 with the same
seed is the regime where the difference is not visible in the video. Then
inspect both mp4s by eye and say so in the doc.

**2c. Scheme sweep (optional, one extra compile each):** `--quant-act none`
(weight-only) and `--quant-granularity channel` via `ptq_fp8_ab.py --only fp8
--quant-act none --out-dir …/ptq-wan21-wo` etc. Each is a separate artifact.

## Phase 3 — performance (P, A3)

From the same A/B summary, per arm and run: `e2e_wall_seconds` (staged CLI wall:
two process loads + text encode + denoise + VAE), `weights_load_total_seconds`,
`compute_and_overhead_seconds`, and `dit_step_ms` (real-loop DiT wall per step,
step 0 excluded, from the `[wan] dit-step ms:` line in each generate log).
Run 0 with `--drop-caches` is the true cold e2e; run 1 is warm.

Then the benchmark harness for the canonical report files:

```bash
python -m benchmark.bench --model wan_2_1_fp8 --skip-download --skip-compile --iters 1
python -m benchmark.cold_warm_e2e --model wan_2_1_fp8
```

(`wan_2_1_fp8` is `wan_2_1` plus the fp8 knobs; results land in
`benchmark/trn2/wan_2_1_t2v_14b_diffusers_fp8_tensor_dyn.{json,md}` next to the
bf16 report, with a `quant` record in the JSON.)

Reading A3: the DiT per-step mean of the fp8 arm against bf16. Wan 14B at tp4
spends most of its step in the linears, so a tensor-engine FP8 path should show
a clear per-step drop; weight-load time drops with the halved shards
regardless. If per-step does not drop while Phase 0 showed device-fp8 ≈ CPU-fp8,
the compiler is dequantizing to bf16 before the matmul — record it as the
finding, and the follow-up is an NKI FP8 matmul kernel behind the same
`QuantSpec` (not built in this delivery by design).

## Phase 4 — serving (S)

```bash
NEURON_RT_NUM_CORES=4 difflet serve --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --tp-degree 4 --height 480 --width 832 --num-frames 9 --quant fp8 --host-vae --port 8091
# wait for /ready (startup builds the immutable generation; cold = full compile)
curl -s -X POST http://127.0.0.1:8091/v1/videos/sync \
    -F model=Wan-AI/Wan2.1-T2V-14B-Diffusers -F prompt="a cinematic shot of a red fox running through a snowy forest" \
    -F height=480 -F width=832 -F num_frames=9 -F num_inference_steps=20 -F seed=42 -o serve_fp8.mp4
PYTHONPATH=$PWD python scripts/ptq_compare_outputs.py --reference <ab-out-dir>/fp8_run1.mp4 --test serve_fp8.mp4
```

The served fp8 result must match the CLI fp8 result (same artifact identity:
`quant` is in the generation compile spec; the cache path is not). Then the
request latency the server logs is the number comparable to FastVideo's
"1.8 s" class of claim (warm per-request: text encode + denoise + decode),
not the staged CLI wall time. Also start once without `--quant` to confirm the
bf16 profile still resolves the pre-existing artifact (identity unchanged).

## Phase 5 — FastVideo comparison protocol

Like-for-like is PTQ vs PTQ on the same architecture: FastVideo's
`FP8` config (`granularity="tensor"`, dynamic activations) on Wan 2.1 at the
same resolution / steps / seed, each quantized run compared against its own
bf16 output (PSNR / SSIM / LPIPS / MS-SSIM), plus DiT denoise time and warm
per-request latency, each on its own hardware. Report QAD rows separately and
label them: they are 3-step DMD students of the 1.3B model with NVFP4 and a
tiny VAE, and are not a PTQ result.

## Bug protocol and triage

Before calling anything a Difflet bug: disk (`[Errno 28]`), a swept task, a
stale artifact from a pre-fix compile, a page-cache anomaly. Then, per
`difflet-device-verify`: root cause with the diagnostic, unit test that pins it,
one commit per bug, re-verify the cell. Likely first-contact issues and where
they live:

- scale shape / name mismatch at load → `difflet/quant/checkpoint.py` (A1);
- `ActivationQuantizationType(None)` or q-config key errors →
  `difflet/backends/trainium/core/quant.py::build_q_config` (mirrors NxDI);
- compiler rejects fp8 dot / flag missing → `NeuronWanBackboneApplication.get_compiler_args`
  and the duplicated `--internal-hlo2tensorizer-options` (A5);
- shared store served bf16 shards to the fp8 app → `shared_weights._key_inputs`
  (must contain `quantized_checkpoint`);
- per-step unchanged with correct numerics → A3, NKI follow-up (not a bug).

## Deliverables

1. Evidence doc with the Phase 0 JSON, the Phase 1 fingerprints, the Phase 2
   tables (linear sweep per timestep; latent + pixel metrics per run pair), the
   Phase 3 table (compile s, e2e cold/warm s, weight load s, DiT step ms; bf16
   vs fp8), the Phase 4 serve check, and the assumption table with PASS / FAIL
   per id.
2. Curated evidence under `artifacts/verification-<date>/ptq-wan21/` (logs,
   summary JSON/MD, the mp4 pairs; no `.pt` latents, no weights).
3. Benchmark report files for `wan_2_1_fp8`.
4. A README feature-matrix row for "FP8 PTQ (Wan)" with the measured deltas —
   only after the assumptions pass.
