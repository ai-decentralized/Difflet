# FP8 PTQ of the Wan 2.1 DiT — on-device evidence

Campaign: 2026-10-01, host `trn2.3xlarge` (`i-0202fe506b1d498c5`, 1 Neuron device,
4 NeuronCores, 96 GB HBM, LNC=2, 12 vCPU, 124 GB RAM, 1.5 TB disk), branch `quantization`.
Plan this executes: `docs/verification/2026-09-29-ptq-fp8-wan-plan.md`.
Design: `docs/superpowers/specs/2026-09-29-ptq-fp8-linear-design.md`.

Curated evidence (logs, JSON summaries, rendered videos; no `.pt` latents, no weights,
no NEFFs) lives under `artifacts/verification-2026-10-01/ptq-wan21/`. Each phase ends
with a "How to inspect manually" recipe. Every number below is transcribed from a file
in that directory or from a command whose output is quoted; inferences are labelled.

Statuses: PASS / FAIL / NOT MEASURED. Assumption ids (A1–A5, Q, P, S) are the plan's.

## Campaign result (final)

_filled at the end_

## Host and toolchain (Phase 0a)

Fresh host: no prebuilt `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`; the venv was
built from `requirements-neuron.lock` with `scripts/setup_env.sh` (Python 3.12.3) plus
`lpips` and `scikit-image` for the pixel metrics.

| component | version |
|---|---|
| Python | 3.12.3 |
| torch / torch-neuronx / torch-xla | 2.9.1 / 2.9.0.2.15.32035+de43f57c / 2.9.0 |
| neuronx-cc | 2.26.6360.0+6f180f47 |
| neuronx-distributed / -inference | 0.19.28492+435aae2b / 0.10.18399+ed62453e |
| libneuronxla | 2.2.17544.0+fb9962bf |
| diffusers / transformers | 0.38.0 / 4.57.6 |
| lpips | 0.1.4 |

Model: `Wan-AI/Wan2.1-T2V-14B-Diffusers` @ `38ec498cb3208fb688890f8cc7e94ede2cbd7f68`
(HF `main` at campaign time; the revision `benchmark/models.py` pins for `wan_2_1` and
`wan_2_1_fp8`). Snapshot on disk: transformer 12 shards (54 GB, stored fp32),
text encoder 22 GB, VAE 485 MB.

Host isolation: a second Claude session on this host ran five model downloads
(FLUX.1-dev, Wan 2.2, HunyuanVideo, Qwen-Image, LTX-2; ~375 GB) during Phase 0a.
Timed phases (device probe, A/B, benchmark, serving) were held until those downloads
finished and an idle gate (no NeuronCore holder, no foreign CPU-heavy or Python
processes) passed; the gate output is quoted at the start of each timed phase.

## Phase plan

| phase | what | runner | status |
|---|---|---|---|
| 0a | host prep (venv, weights) | `scripts/setup_env.sh`, `hf download` | PASS |
| 0b | quant unit tests + tiny device probe (A1 A2 A4 A5) | `pytest`, `scripts/ptq_fp8_device_probe.py` | PASS after 4 fixes |
| 1 | quantize + compile, timed (A4, A5) | `scripts/ptq_fp8_ab.py` | _pending_ |
| 2a | CPU per-linear fp8 error sweep, t = 500 / 900 / 100 | `scripts/ptq_linear_error_sweep.py` | _pending_ |
| 2b | device latent + pixel error vs bf16 (Q, A2) | `scripts/ptq_fp8_ab.py` | _pending_ |
| 3 | performance: compile, e2e cold/warm, weight load, DiT per-step (P, A3) | `scripts/ptq_fp8_ab.py`, `benchmark.bench`, `benchmark.cold_warm_e2e` | _pending_ |
| 4 | `difflet serve --quant fp8` (S) | `difflet serve`, `scripts/ptq_compare_outputs.py` | _pending_ |

## Phase 0b — unit tests and tiny device probe

**Unit tests (CPU, this venv):** `pytest tests/unit/quant tests/unit/cli/test_cli_quant.py
tests/unit/serving/test_serve_quant.py tests/unit/test_ptq_scripts.py` → **58 passed, 0 skipped,
1 warning, 21.8 s**. No skips means the `neuronx_distributed`-gated
`test_wan_application_resolves_and_ensures_quantized_checkpoints` ran here.
Log: `artifacts/verification-2026-10-01/ptq-wan21/phase0/pytest_quant.log`.

**Tiny device probe** (`scripts/ptq_fp8_device_probe.py`, random 2-block Wan transformer,
64×64 latent, 512 text tokens, tp1, one NeuronCore). Four runs were needed; each failure
is a real finding, kept verbatim under `phase0/`:

| attempt | arms | outcome | evidence |
|---|---|---|---|
| 1 | bf16, fp8-dyn | bf16: compiled + loaded, forward rejected (`Input shape [[1, 16, 1, 8, 8], [1], [1, 16, 64]] not found in input_shape_map: [[[1, 16, 1, 8, 8], [1], [1, 512, 64]]]`) — probe traced text at 16 tokens, backbone traces 512. fp8: `AssertionError: Unsupported activation quantization type: DYNAMIC` from `NeuronConfig` (bug 1). | `probe_attempt1_failed.log`, `ptq_probe_report_attempt1_failed.json` |
| 2 | bf16, fp8-dyn | bf16 **PASS**: compile 16.8 s, load 6.6 s, forward 0.63 ms (n=10); device-bf16 vs CPU-bf16 cosine 0.9999911, SNR 47.5 dB (the accumulation-noise control). fp8: trace failed inside NxD `QuantizedColumnParallel.forward` — `RuntimeError: Check failed: input_sizes.size() <= output_sizes.size() (4 vs. 3)` (bug 2). | `probe_attempt2_fp8_trace_failed.log`, `ptq_probe_report_attempt2_fp8_trace_failed.json` |
| 2b | fp8 weight-only (`--quant-act none`) | compiled (16.3 s) + loaded (5.6 s); `compiler_args` carries `--experimental-unsafe-fp8e4m3fn-as-fp8e4m3` (A5 ✅); sharded shard has fp8 weights + float32 `.scale` (A1, A4 ✅ for tp1); **output all NaN** (bug 3). | `probe_weight_only_attempt1_nan.log`, `ptq_probe_report_weight_only_attempt1_nan.json` |
| 3 | bf16, fp8-dyn; then fp8 weight-only | bf16 PASS again (compile 15.9 s, load 7.3 s, 0.46 ms, 0 non-finite). fp8-dyn: attention linears (128→128) traced, first FFN linear (128→256) failed `Shapes are not compatible for broadcasting: f32[1,16,256] vs. f32[1,16,128]` — on XLA `amax()` with no dims traced as a no-op, so the "per-tensor" scale was input-shaped (bug 2, second half). **fp8 weight-only PASS** with the 240 range: compile 17.4 s, load 6.4 s, forward 0.41 ms, 0 non-finite; device-fp8 vs CPU-fp8 cosine **0.9999798**, SNR 43.9 dB; device-fp8 vs CPU-bf16 cosine 0.9999737, SNR 42.8 dB; CPU-fp8 vs CPU-bf16 (the algorithm's own error) cosine 0.9999784, SNR 43.6 dB. | `probe_attempt3_amax_noop.log`, `ptq_probe_report_attempt3_amax_noop.json`, `probe_weight_only.log`, `ptq_probe_report_weight_only.json` |
| 4 | bf16, fp8-dyn | fp8-dyn traced and compiled; weight sharding failed `expected shape torch.Size([128, 1]) for blocks.0.attn1.to_q.scale but found torch.Size([1])` — the vendored NeuronConfig rewrote `quantization_type` to `per_channel_symmetric` because `is_mlp_quantized()` is true for any activation quantization (bug 4). | `probe_attempt4_scale_shape.log`, `ptq_probe_report_attempt4_scale_shape.json` |
| 5 | bf16, fp8-dyn | **PASS** (`passed: true`). See the table below. | `probe.log`, `ptq_probe_report.json` |

Bug 1 — **`"DYNAMIC"` is the NxD enum member name; the values are lowercase.**
`difflet/backends/trainium/core/quant.py` emitted `activation_quantization_type="DYNAMIC"`;
`NeuronConfig` validates by enum membership (`MyEnumMeta.__contains__` constructs the enum),
so the fp8 arm never reached compile. The unit test had a stand-in enum with uppercase
values. Fix `dece4f4`: emit `"dynamic"`; the stand-in mirrors NxD; a new test runs the kwargs
through NxD's validator wherever `neuronx_distributed` is installed.

Bug 2 — **NxD's DYNAMIC activation path cannot trace 3-D DiT activations.**
`QuantizedColumnParallel.forward` calls `quantize_fp8_per_channel(input, channel_axis=1)` (a
per-token `[1, S, 1]` scale) and dequantizes with `scale_dequantize`, which does
`scale.unsqueeze(len(scale.shape)-1)` — written for 2-D weight scales — producing a 4-D scale
against the 3-D output. NxDI only uses DYNAMIC via its NKI quantized-MLP kernels, so the
generic forward is unexercised upstream. Design assumption **A2 ("NxD DYNAMIC = per-tensor
absmax") is false**. Fix `6fe599d`: Difflet maps the NxD float layers onto subclasses of the
NxD quantized layers whose DYNAMIC branch uses one per-tensor absmax scale (the CPU
reference's law), the fp8×fp8 matmul through NxD's `_forward_impl`, and a rank-agnostic
dequantize; weight-only keeps NxD's forward.

Bug 3 — **the 448 absmax range is NaN on Trainium.**
The weight-only arm's device output was all NaN with a healthy checkpoint (fp8 weights, no
NaN, scales loaded as `.scale`). Every fp8 weight's absmax was exactly 448 (torch's
`float8_e4m3fn` max, FastVideo's GPU law). Trainium's native fp8 e4m3 tops out at **240**
(`neuronx_distributed.quantization.quantization_config.DtypeBound.F8E4M3_MAX = 240.0`, what
NxD's own quantizers clamp to) and the `e4m3fn-as-e4m3` compiler flag reinterprets the bit
patterns, so encodings above 240 decode as inf/NaN. Count on the device shard
(`tp0_sharded_checkpoint.safetensors`): **175,744 of 393,216 fp8 elements (44.7 %) above 240**.
Fix `959f2cc`: `FP8_MAX = 240.0` for weights and activations, the range in the checkpoint
identity (hash, manifest) so a 448-range copy is never reused.

Bug 4 — **NeuronConfig forces per-channel weights under any activation quantization.**
`NeuronConfig.__init__` (vendored NxDI) sets `quantization_type = "per_channel_symmetric"`
whenever `is_mlp_quantized()` — `quantized_mlp_kernel_enabled or activation_quantization_type`
— is truthy: NxDI's DYNAMIC scheme is its per-token/per-channel quantized-MLP kernel. The NxD
layers were therefore built with `[out, 1]` scales against Difflet's per-tensor `[1]`
checkpoint. In the same round, `amax()` with no dims traced as a no-op on XLA (attempt 3),
so the activation absmax now reduces over explicit dims. Fix `cfc1adf`: the override
applies only with the quantized MLP kernel; a test constructs `NeuronConfig` from the spec
kwargs and asserts the granularity survives.

Probe script fixes (`5fe2a49`): trace text at `--text-seq-len` (default 512), save each arm's
device output and its non-finite count.

Reading the weight-only numbers: device-fp8 differs from the CPU-fp8 reference by 43.9 dB SNR,
about 3.6 dB below the bf16 control (47.5 dB). Inference (not proven from the logs): NxD's
weight-only forward casts the fp8 weight to bf16, runs a bf16 matmul and scales the output
afterwards, i.e. one more bf16 rounding per linear than the bf16 arm, while the CPU reference
is fp32 throughout. The difference is far below the quantization error itself and the arm
passes the probe's 0.999 cosine gate.

**Attempt 5 (final) — tensor-granularity weights, dynamic per-tensor activations, tp1:**

| arm | compile s | load s | forward ms (n=10, mean / median) | non-finite |
|---|---:|---:|---:|---:|
| bf16 | 8.0 | 5.8 | 0.57 / 0.55 | 0 |
| fp8-dyn | 23.7 | 0.09 ¹ | 0.93 / 0.92 | 0 |

| check | cosine | rel-L2 | SNR dB |
|---|---:|---:|---:|
| device-bf16 vs CPU-bf16 (control) | 0.9999911 | 0.00424 | 47.45 |
| CPU-fp8 vs CPU-bf16 (algorithm's own error) | 0.9999686 | 0.00793 | 42.02 |
| **device-fp8 vs CPU-fp8 (A2)** | **0.9999827** | 0.00591 | **44.57** |
| device-fp8 vs CPU-bf16 | 0.9999718 | 0.00753 | 42.47 |
| device-fp8 vs device-bf16 | 0.9999729 | 0.00737 | 42.66 |

¹ the fp8 arm loads second in the same process (runtime already initialised); not a
weight-load measurement. The forward times at this 2-block, 64-token shape are sanity
signals only (the plan's A3 is decided by the 14B per-step numbers); at this size the
quantize / dequantize elementwise ops outweigh the matmuls, so fp8 is slower here.

fp8 `compiler_args`: `--model-type=transformer -O1 --tensorizer-options='--enable-ccop-compute-overlap' --auto-cast=none --internal-hlo2tensorizer-options='--experimental-unsafe-fp8e4m3fn-as-fp8e4m3 --verify-hlo=true'` (A5 ✅).
Presharded shards: `compiled_fp8/weights/tp0_sharded_checkpoint.safetensors` 749,976 B vs
`compiled_bf16/...` 1,135,864 B (fp8 weights + float32 `.scale`, A1/A4 ✅ at tp1).

Reading: device-fp8 tracks the CPU-fp8 reference (44.6 dB) more closely than either tracks
bf16 (42.0–42.7 dB), and the residual sits below the bf16 control (47.5 dB) by 2.9 dB —
consistent with Difflet's per-tensor dynamic law running on the device, plus
accumulation-order / bf16-rounding noise. **Phase 0 verdict: A1, A2 (via Difflet's own
activation path, not NxD's), A4 (tp1), A5 PASS; continue to Phase 1.**

**How to inspect manually:** `python -m json.tool phase0/ptq_probe_report.json` — `arms.*.compiled/loaded`,
`arms.fp8.compiler_args`, `checks.device_fp8_vs_cpu_fp8.cosine`; the failed attempts' reports
carry `arms.fp8.error` and `traceback` verbatim.

## Phase 1 — quantize + compile

Driver: `scripts/ptq_fp8_ab.py --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498…
--tp-degree 4 --height 480 --width 832 --num-frames 9 --steps 20 --guidance-scale 1.0 --seed 42
--runs 2 --drop-caches --out-dir artifacts/verification-2026-10-01/ptq-wan21/ab`, launched
after the idle gate (`gate_before_ab.txt`: no NeuronCore holder, no foreign processes, load
0.33) with nothing else on the host. Subprocess logs under `ab/logs/`.

**Quantize** (`ab/logs/quantize.log`): `difflet quantize --quant fp8 --quant-granularity tensor`
→ `~/.cache/difflet/quantized/Wan-AI--Wan2.1-T2V-14B-Diffusers/transformer/fp8-tensor-1d3afee3/`
(2 safetensors shards + index, `config.json`, `difflet_quant.json`): **400 linears quantized,
57,153,966,336 → 15,001,212,736 bytes, 30.8 s wall** (CPU; the source snapshot is fp32, so the
"before" is the fp32 size — the bf16 equivalent is 28.6 GB). Manifest `checkpoint_identity`
= `{format: fp8_e4m3, fp8_max: 240.0, weight_granularity: tensor, targets: [...]}`.

**Compile** (`ab/logs/compile_{bf16,fp8}.log`, breakdown via `benchmark.parse_compile`; neither
arm was a cache hit for the transformer — fresh host):

| stage | bf16 build s (priority HLO s) | fp8 build s (priority HLO s) |
|---|---:|---:|
| text_encoder_t5 | 39.4 (7.2) | 39.7 (5.2) |
| transformer | 297.4 (82.9); presharding 116.3 s, 4 shards, 28,602.8 MB | 395.2 (161.0); presharding 138.9 s, 4 shards, **15,211.0 MB** |
| vae_decoder | 5918.2 (5903.8) | — (shared stage, reused) |
| **wall** | **6255.0** | **434.9** |

The fp8 transformer graph takes ~2× the compiler time of the bf16 one (161 s vs 83 s for the
priority HLO); the presharded fp8 shards are 53 % of the bf16 bytes (A4 at tp4 ✅; the
remaining non-fp8 tensors — patch embed, adaLN, norms, `proj_out`, biases, scales — stay
bf16/float32). The VAE decoder dominates the bf16 wall (98 min) and is shared with the fp8
arm, so the fp8 arm's compile cost is the transformer stage only.

**A4 fingerprints** (`ab/fingerprints.txt`, from `ls -la` / `stat` on `~/.cache/difflet`):

- compiled transformer artifacts: bf16 `wan2_1_t2v_14b_diffusers_transformer/c3e3b1be10ccc035/`
  (manifest `cache_inputs` has **no** `quant` key — bf16 identity unchanged), fp8
  `…/edb936fbb4e3825e/` (manifest `cache_inputs.quant = {activation: dynamic, format:
  fp8_e4m3, targets: [...], weight_granularity: tensor}`);
- `transformer/weights/tp{0..3}_sharded_checkpoint.safetensors`: bf16 4 × 7,498,045,044 B,
  fp8 4 × 3,987,469,508 B; every shard has link count 2 (hardlinked from the store);
- `_shared_weights/Wan-AI--Wan2.1-T2V-14B-Diffusers__transformer__38ec498c__bfloat16__qf8e4m3__tp4__c7ca57f373d92cea/`
  (the `qf8e4m3` label from `shared_weights.store_label`) next to the bf16 entry
  `…__transformer__38ec498c__bfloat16__tp4__c06caa3c7836f86e/`, the text-encoder entry
  (4 × 2,840,786,528 B, link count 3: shared by both arms' artifacts) and the VAE entry.

## Phase 2a — per-linear matmul error (CPU, real weights)

`scripts/ptq_linear_error_sweep.py` on the 14B snapshot (all 40 blocks, 400 target linears),
480×832×9 latent, 512 text tokens, random Gaussian latent / text inputs (no `--bundle`),
1024 tokens kept per captured activation; one bf16 forward per timestep captures every
target linear's real input, then each linear is recomputed under every scheme and compared
with the fp32 exact product. **All numbers are at the 240 range** (the first t=500 pass ran
before fix `959f2cc`; it is kept as `linear_error_t500_fp8max448_superseded.json` and was
rerun). Files: `linear_sweep/linear_error_t{900,500,100}.json`, `sweep_t*.log`. The sweep ran
on the CPU while nothing else used the host except, for part of t=500/t=900, the tiny device
probe; its `forward_seconds` (130–146 s) is not a timing claim.

| t | scheme | min cos | mean cos | max rel-L2 | min SNR dB | mean SNR dB | worst cell |
|---:|---|---:|---:|---:|---:|---:|---|
| 900 | bf16 (floor) | 0.999999 | 0.999999 | 0.0017 | 55.41 | 55.60 | blocks.14.attn2.to_out.0 |
| 900 | fp8-tensor-dyn | 0.999069 | 0.999480 | 0.0431 | 27.30 | 30.09 | blocks.19.attn1.to_out.0 |
| 900 | fp8-tensor-wo | 0.999540 | 0.999738 | 0.0303 | 30.36 | 33.08 | blocks.19.attn1.to_out.0 |
| 900 | fp8-channel-dyn | 0.999069 | 0.999484 | 0.0432 | 27.29 | 30.12 | blocks.16.attn1.to_out.0 |
| 900 | fp8-channel-wo | 0.999557 | 0.999742 | 0.0298 | 30.53 | 33.14 | blocks.16.attn1.to_out.0 |
| 500 | bf16 (floor) | 0.999999 | 0.999999 | 0.0017 | 55.30 | 55.60 | blocks.14.attn2.to_out.0 |
| 500 | fp8-tensor-dyn | 0.999083 | 0.999485 | 0.0428 | 27.37 | 30.15 | blocks.19.attn1.to_out.0 |
| 500 | fp8-tensor-wo | 0.999539 | 0.999741 | 0.0304 | 30.35 | 33.14 | blocks.16.attn1.to_out.0 |
| 500 | fp8-channel-dyn | 0.999105 | 0.999490 | 0.0423 | 27.47 | 30.19 | blocks.17.attn1.to_out.0 |
| 500 | fp8-channel-wo | 0.999553 | 0.999745 | 0.0299 | 30.49 | 33.20 | blocks.16.attn1.to_out.0 |
| 100 | bf16 (floor) | 0.999999 | 0.999999 | 0.0017 | 55.32 | 55.60 | blocks.14.attn2.to_out.0 |
| 100 | fp8-tensor-dyn | 0.999107 | 0.999493 | 0.0423 | 27.48 | 30.21 | blocks.16.attn1.to_out.0 |
| 100 | fp8-tensor-wo | 0.999539 | 0.999742 | 0.0304 | 30.35 | 33.17 | blocks.16.attn1.to_out.0 |
| 100 | fp8-channel-dyn | 0.999111 | 0.999497 | 0.0422 | 27.50 | 30.25 | blocks.17.attn1.to_out.0 |
| 100 | fp8-channel-wo | 0.999554 | 0.999747 | 0.0299 | 30.50 | 33.24 | blocks.16.attn1.to_out.0 |

Mean SNR of the production scheme (fp8-tensor-dyn) by linear type, t=500: attn1.to_q 30.3 ·
to_k 31.0 · to_v 29.5 · to_out 28.7 · attn2.to_q 30.0 · to_k 29.9 · to_v 29.5 · to_out 31.1 ·
ffn.net_in 30.0 · ffn.net_out 31.4 dB. The five worst cells at every timestep are
`blocks.{14..19}.attn1.to_out.0` (self-attention output projections, cosine ≥ 0.99907).

Reading: (1) the per-linear error is timestep-independent to the second decimal (the random
inputs only change the adaLN modulation), (2) weight-only sits ~3 dB above dynamic — the
activation quantization costs as much as the weights do, (3) per-channel weight scales buy
nothing over per-tensor on these weights (≤ 0.1 dB), so the default `tensor` granularity is
the right choice, (4) the cross-attention `to_k`/`to_v` the plan flagged are *not* the weak
cells; the self-attention output projections of blocks 14–19 are, at ~27.4 dB. Caveat: the
inputs are Gaussian, not denoising-trajectory activations (`--bundle`), so this measures the
weights' quantization behaviour under typical-magnitude inputs, not outlier channels of real
activations; the device latent / pixel metrics (Phase 2b) are the end-to-end answer.

**How to inspect manually:** `python -c "import json; d=json.load(open('…/linear_error_t500.json')); print(d['summary'])"`;
each `rows[i]` carries per-scheme `cosine / mse / max_abs / rel_l2 / snr_db` for one linear.

## Phase 2b — latent and pixel error on device

Same prompt ("a cinematic shot of a red fox running through a snowy forest"), seed 42,
480×832×9, 20 steps, guidance 1.0, tp4, Neuron VAE; two runs per arm. From
`ab/ab_summary.{json,md}` (`difflet.quant.metrics`: PSNR / SSIM on the decoded mp4 frames,
LPIPS-alex, latent metrics on the DiT output `latents.pt` before the VAE):

| pair | PSNR dB | SSIM | LPIPS | latent cosine | latent MSE | latent SNR dB |
|---|---:|---:|---:|---:|---:|---:|
| fp8 vs bf16, run 0 | 36.50 | 0.9269 | 0.0789 | 0.997457 | 2.240e-03 | 22.93 |
| fp8 vs bf16, run 1 | 36.50 | 0.9269 | 0.0789 | 0.997457 | 2.240e-03 | 22.93 |
| bf16 run 1 vs run 0 (control) | ∞ | 1.0000 | 0.0000 | 1.000000 | 0 | ∞ |
| fp8 run 1 vs run 0 (control) | ∞ | 1.0000 | 0.0000 | 1.000000 | 0 | ∞ |

Both arms are bit-deterministic run to run (the controls are exact), so the fp8-vs-bf16 numbers
are the quantization effect alone. PSNR 36.5 dB / SSIM 0.927 vs the bf16 render is inside the
plan's "not visible in the video" regime (> 30 dB / > 0.9). Latent SNR 22.9 dB over 20
denoising steps, against ~30 dB per linear (Phase 2a): the per-step errors compound but stay
well-conditioned (cosine 0.9975). Videos: `ab/bf16_run{0,1}.mp4`, `ab/fp8_run{0,1}.mp4`;
frame grid `ab/frames_bf16_vs_fp8.png` (bf16 / fp8 / 8×|diff| at frames 0, 2, 4, 6, 8; mean
|diff| 2.85 / 255, max 136).

**Visual verdict.** Frame 0 of both arms is a clean render of the prompt (red fox, snowy
trunks); the fp8 frame is indistinguishable from the bf16 frame at a glance and the |diff|
image is edge-aligned noise with no structure of its own. **Frames 2–8 are a washed-out
brown-grey texture in *both* arms** — the bf16 baseline itself degrades after the first frame
at this configuration (20 steps, guidance 1.0, 9 frames, Neuron VAE), reproduced
independently by decoding `bf16_run1.mp4` with ffmpeg (frames 0 / 4 / 8). Per latent frame
(`work_*/latents.pt`, 3 latent frames) the fp8 arm tracks bf16 equally well on all three
(cosine 0.997575 / 0.997259 / 0.997576; SNR 23.1 / 22.6 / 23.2 dB), and the bf16 latents
themselves have a rising std over latent frames (0.496 / 0.682 / 0.771), so the degradation
is already present in the DiT output or introduced by the decoder equally for both arms —
it is not a quantization finding, but it limits what "not visible" means here to frame 0.
**Host-VAE decode of the same latents settles it:** `work_{bf16,fp8}_run1/latents.pt` decoded
on the host with the diffusers `AutoencoderKLWan` (fp32, 73 s per clip;
`ab/{bf16,fp8}_run1_hostvae.mp4`, frames `ab/frames_bf16_hostvae_0_4_8.png`) give a proper
clip — the fox runs through the forest in every frame, 0 / 4 / 8 alike — while the Neuron
VAE decode of the *same* tensor (`ab/frames_bf16_neuronvae_0_4_8.png`) is sharp only at
frame 0. **The frame-2…8 degradation is the compiled Neuron VAE decoder at 480×832×9, not
the DiT and not quantization** — a pre-existing bf16 pipeline issue outside this campaign's
scope (`--host-vae` is the documented workaround; recorded as a follow-up, not fixed here).
On the host-decoded clips (`ab/compare_hostvae_fp8_vs_bf16.json`,
`ab/frames_hostvae_bf16_vs_fp8.png`), where every frame carries content, **fp8 vs bf16 is
PSNR 32.70 dB, SSIM 0.9515, LPIPS 0.0381**, pixel cosine 0.99825 — and the two clips are the
same clip to the eye in all nine frames; the 8×-amplified difference image sits on the fox's
fur edges and eye highlights. (These are the Phase-0-era fp8 layers; the pre-VAE latents of
the fixed arms are compared in the re-measured table below.)

## Phase 3 — performance

From `ab/ab_summary.json` and the `Neuron: Finished traced model weight initialization` lines
of `ab/logs/generate_*.log` (transformer stage; the text encoder is 89 s cold / 8.5 s warm and
the VAE 32 s in every run, identical across arms). Run 0 of each arm follows a page-cache drop
(`sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`, `page_cache_dropped: true`); run 1 is
warm. e2e is the staged CLI wall (two process loads + text encode + denoise + VAE decode).

| arm | compile wall s | e2e cold s | e2e warm s | transformer weight load cold s (file read / device init) | warm s | DiT step ms, mean / median (n=19) |
|---|---:|---:|---:|---:|---:|---:|
| bf16 | 6711.9 | 412.7 | 87.1 | 236.0 (2.5 / 233.5) | 9.9 | **573.2 / 573.0** (run 1: 573.7 / 573.6) |
| fp8-tensor-dyn | 590.1 (+32.2 quantize) | 317.4 | 94.7 | 132.0 (2.8 / 129.2) | 11.7 | **822.1 / 821.9** (run 1: 822.0 / 821.7) |

- **A3 FAIL — the fp8 DiT step is 1.43× slower than bf16** (822 vs 573 ms), identically on the
  cold and the warm run. The compiler did receive fp8 matmuls: the fp8 transformer HLO in the
  compile cache (`/var/tmp/neuron-compile-cache/neuronxcc-2.26.6360.0+6f180f47/MODULE_a1cd99ffd9d0c5735511+04929ab2/model.hlo_module.pb`,
  parsed with `torch_neuronx.pyhlo`) has **400 `dot` ops with `F8E4M3FN × F8E4M3FN` operands**
  (result dtype `F8E4M3FN`, immediately converted to F32), 400 `F32 → F8E4M3FN` converts (the
  dynamic activation quantization) and 400 `F8E4M3FN → F32` converts, plus 5 bf16 dots; the bf16
  HLO has 406 `BF16 × BF16` dots and none of those converts. Its `compile_flags.json` carries
  `--experimental-unsafe-fp8e4m3fn-as-fp8e4m3` (twice, by design). So neuronx-cc 2.26 lowers an
  fp8 dot more slowly than a bf16 one on trn2 and the 800 extra elementwise converts on
  [1, 4680, 5120]-class activations add on top; the per-linear numerics (Phase 0, 2b) show the
  math is right, the speed is not. Follow-up per the plan: an NKI FP8 matmul kernel behind the
  same `QuantSpec` (not in this delivery); the weight-only arm below separates the dot cost
  from the quantize/dequantize cost.
- **Weight load (cold) −44 %**: 132 s vs 236 s for the transformer stage, tracking the shard
  bytes (15.2 GB vs 28.6 GB). Warm loads are within 2 s of each other (page cache hot; the fp8
  arm is 1.8 s slower — inference: the per-layer `scale` tensors and fp8 → device transfers).
- **e2e cold −23 %** (317 vs 413 s) because the cold weight read dominates; **e2e warm +9 %**
  (94.7 vs 87.1 s) because the 19 slower DiT steps (+4.7 s) and the slower warm load outweigh
  nothing else.
- Compile: the fp8 arm compiles in 590 s wall because only the transformer stage is new
  (text encoder and VAE reused); the bf16 arm's 6712 s is dominated by the VAE decoder
  (Phase 1 table). Note the harness's `weights_load_total_seconds` field is 0 for every run:
  `benchmark.parse_generate` does not match this toolchain's `load_weights` lines — a harness
  gap, recorded here rather than fixed mid-campaign; the numbers above are read from the logs.

**Weight-only arm** (`--quant-act none`, same fp8 checkpoint, own transformer compile;
`ab-wo/ab_summary.{json,md}`, `ab-wo/logs/`; gate `gate_before_ab_wo.txt`):

| arm | compile wall s | e2e cold s | e2e warm s | transformer load cold s | DiT step ms, mean / median (n=19) |
|---|---:|---:|---:|---:|---:|
| fp8-tensor-wo | 444.6 (priority HLO 151.5) | 312.3 | 92.4 | 131.7 | **732.8 / 732.7** (run 1: 733.0 / 732.9) |

So the weight-only path is already 1.28× slower than bf16 and the dynamic activation
quantization adds another 90 ms on top. Its HLO (`ab/hlo_dots_fp8_wo.txt`, module
`MODULE_764c515c759bcef5a61e+fd859317`) explains the first part: **316 of its 400 quantized
linears run as `F32 × F32 → F32` dots** (84 as bf16), with 316 `F8E4M3FN → F32` and 84
`F8E4M3FN → BF16` weight converts — NxD dequantizes the fp8 weight to the *input's* dtype, and
those inputs are fp32. In the bf16 arm every one of the 400 linears is a `BF16 × BF16` dot.

**Bug 5 — the quantized layers are typed float32, which leaks fp32 into the whole block.**
NxD's `from_float` builds the quantized layer with `dtype=mod.dtype`, the *construction-time*
dtype of the float layer; Difflet builds the Wan model and then casts it (`model.to(bf16)` in
`WanBackbone.get_model_instance`), so `mod.dtype` is still float32. The quantized layers'
`bias` (and `dequantized_dtype`) therefore come out float32 — visible in the presharded
checkpoint of Phase 0 (`blocks.0.attn1.to_k.bias torch.float32` next to bf16 norms) — and
every biased quantized linear (`to_q/k/v`, `ffn.net_in`) promotes its output to fp32: the
attention core, the GELU and the next linear's input run in fp32 until the residual add casts
back. The dynamic arm's HLO shows the same leak from the other side (816 `BF16 → F32`
converts vs 1683 in the bf16 arm: fewer casts because the tensors already *are* fp32). Fix
`42ae642`: Difflet's `from_float` overrides type the layer from the live `mod.weight.dtype`
(bias cast, `dtype`/`dequantized_dtype` set); pinned by a unit test. The fp8 arms are
re-measured below with the fix.

### Re-measured fp8 arms after bug 5 (final numbers)

Same commands with the fixed layers (`42ae642`) and the schema-keyed artifacts (`c3357c6`):
new transformer artifacts / store entries (`…transformer/9c7576a9fdbb564b`, store
`…qf8e4m3__tp4__62aade90b95a3378`), `--only fp8 --skip-quantize`, two runs each with the
page-cache drop before run 0; the bf16 arm is unchanged (its keys and NEFF are untouched, and
its numbers above stand). Dirs `ab-fixed/` (dynamic) and `ab-wo-fixed/` (weight-only); gate
`gate_before_ab_fixed.txt`.

| arm | compile wall s (transformer priority HLO s) | e2e cold s | e2e warm s | transformer load cold / warm s | DiT step ms, mean / median (n=19), run 0 · run 1 |
|---|---:|---:|---:|---:|---:|
| bf16 (unchanged) | 6711.9 (82.9) | 412.7 | 87.1 | 236.0 / 9.9 | **573.2** / 573.0 · 573.7 / 573.6 |
| fp8-tensor-dyn, fixed | 505.0 (99.9) | 307.5 | **85.2** | 128.0 / 8.2 | **656.7** / 656.6 · 656.9 / 656.8 |
| fp8-tensor-wo, fixed | 354.9 (79.3) | 305.5 | **85.2** | 127.9 / 8.0 | **563.0** / 562.9 · 563.0 / 562.7 |
| fp8-tensor-dyn, before fix 5 | 590.1 (161.0) | 317.4 | 94.7 | 132.0 / 11.7 | 822.1 / 821.9 · 822.0 / 821.7 |
| fp8-tensor-wo, before fix 5 | 444.6 (151.5) | 312.3 | 92.4 | 131.7 / — | 732.8 / 732.7 · 733.0 / 732.9 |

Fixed-arm HLO (`ab/hlo_dots_fp8_dyn_fixed.txt`, `MODULE_f1d9baa9d8bbf1aadd37+a0010e47`): still
400 `F8E4M3FN × F8E4M3FN` dots and the 400 + 400 quantize/dequantize converts, but 9,463 bf16
instructions instead of 977 — the block runs in bf16 again, and the priority-HLO compile
drops from 161 s to 100 s.

Fixed weight-only HLO (`ab/hlo_dots_fp8_wo_fixed.txt`, `MODULE_32ad40a41217be5fec9e+77b143b0`):
406 `BF16 × BF16` dots — the same dot set as the bf16 arm — plus 400 `F8E4M3FN → BF16` weight
casts, no fp32 dots; its priority-HLO compile (79 s) matches bf16's (83 s).

**A3 verdict (final).** With fp8 × fp8 dots (per-tensor dynamic activations) the DiT step is
**1.15× bf16** (656.7 vs 573.2 ms; 1.43× before fix 5): the fp8 dot lowering plus the 400
activation-quantize and 400 dequantize passes over `[1, 4680, 5120]`-class tensors cost more
than the bf16 matmul — neuronx-cc 2.26 gives no tensor-engine FP8 win on trn2 for this graph.
**Weight-only fp8 runs the step at 563.0 ms, 1.8 % *faster* than bf16**, i.e. per-step parity
with half the transformer bytes: the fp8 → bf16 weight cast is cheaper than reading twice the
bytes from HBM. Both fp8 modes give **−46 % cold transformer load (128 vs 236 s), −25 % cold e2e
(305–308 vs 413 s) and warm e2e at parity or better (85.2 vs 87.1 s)**. Recommendation for
users today: `--quant fp8 --quant-act none`; the dynamic path is the hook for an NKI FP8
matmul kernel (the follow-up that could turn the fp8 dot into a per-step win, out of scope by
design).

**Quality of the fixed arms vs bf16** (run 1 pairs, Neuron VAE decode;
`ab-fixed/compare_fp8_vs_bf16_run1.json`, `ab-wo-fixed/compare_fp8_vs_bf16_run1.json`; both
arms remain bit-deterministic run to run):

| pair | PSNR dB | SSIM | LPIPS | latent cosine | latent MSE | latent SNR dB |
|---|---:|---:|---:|---:|---:|---:|
| fp8-dyn (fixed) vs bf16 | 32.97 | 0.8876 | 0.1285 | 0.975817 | 2.104e-02 | 13.20 |
| fp8-wo (fixed) vs bf16 | 32.61 | 0.8812 | 0.1374 | 0.973650 | 2.293e-02 | 12.83 |
| fp8-dyn (before fix 5) vs bf16 | 36.50 | 0.9269 | 0.0789 | 0.997457 | 2.240e-03 | 22.93 |

Pairwise latent similarity of every run-1 arm (`latent_matrix_run1.txt`, cosine / SNR):

| | bf16 | dyn-pre | wo-pre | dyn-fixed | wo-fixed |
|---|---|---|---|---|---|
| bf16 | — | 0.99746 / 22.9 | 0.97486 / 13.0 | 0.97582 / 13.2 | 0.97365 / 12.8 |
| dyn-pre | | — | 0.97770 / 13.6 | 0.97840 / 13.7 | 0.97597 / 13.2 |
| wo-pre | | | — | 0.99731 / 22.7 | 0.99850 / 25.0 |
| dyn-fixed | | | | — | 0.99822 / 24.5 |

Reading: the three arms that quantize only the weights or run the fixed layers (wo-pre,
dyn-fixed, wo-fixed) agree with each other at 22–25 dB — three independent compiles, two
different forwards (NxD's weight-only and Difflet's dynamic) — and all sit ~13 dB from the
bf16 run. **That 13 dB (PSNR ~32.6–33 dB, SSIM ~0.88, LPIPS ~0.13 after the Neuron VAE) is
the measured fp8 weight-quantization effect over 20 steps at this shape**; dynamic activation
quantization adds little on top of it (dyn-fixed vs wo-fixed 24.5 dB). The pre-fix dynamic
arm is the outlier: it matched bf16 at 22.9 dB while differing from the three others by
13.6 dB — its fp32-typed block (fix 5) ran the attention core and FFN intermediates in fp32,
a different numerics regime from every other arm. Why that regime lands *closer* to the bf16
run than the bf16-regime fp8 arms do is not explained by this campaign's data; the honest
reading is that the bf16 run is itself ~13 dB from the fp32 truth over 20 steps, so that the
distance between two arms measures their numerics regimes as much as their quantization
error. A CPU reference is the way to attribute it (full-forward numerics below; a 20-step CPU
fp32 / fp8 loop at this shape is the follow-up). Per-step and per-linear, the device fp8 math
is verified against the CPU reference (Phase 0, fixed-layer probe below).

### Phase 3b — benchmark harness report files

`python -m benchmark.bench --model <slug> --skip-download --skip-compile --iters 1` then
`python -m benchmark.cold_warm_e2e --model <slug>` (page cache dropped before the cold run,
warm run immediately after), on the fixed artifacts, nothing else on the host (`gate_before_bench.txt`).
Logs under `bench/`; report files in `benchmark/trn2/` (committed):

| slug | report | e2e cold s | e2e warm s (n=1) | DiT step s (real loop, n=19) |
|---|---|---:|---:|---:|
| `wan_2_1` (bf16, re-measured on this host) | `wan_2_1.{json,md}` | 413.9 | 85.7 | 0.5548 ¹ |
| `wan_2_1_fp8` (dynamic) | `wan_2_1_fp8.{json,md}` | 304.3 | 84.2 | 0.6568 |
| `wan_2_1_fp8_wo` (weight-only) | `wan_2_1_fp8_wo.{json,md}` | 302.9 | 82.6 | 0.5627 |

¹ `cold_warm_e2e` only re-measures e2e; the bf16 step field is the prior host's value carried
over by the harness (the note in the JSON says so). The same-host bf16 real-loop step from
the A/B is 573.2 ms, which is the number to compare the fp8 steps against.
The harness numbers reproduce the A/B within 1–3 s on e2e and within 1 ms on the step.

_Serving: see Phase 4._

## Phase 4 — serving

_pending_

## Assumption table

| id | claim | status | evidence |
|---|---|---|---|
| A1 | fp8 weight + scale load into NxD quantized parallel linears | PASS (tp1) | phase0: fp8 shard with `.scale` loads, attempts 2b/3/5 |
| A2 | NxD DYNAMIC activation quant = per-tensor absmax (device-fp8 ≈ CPU-fp8) | PASS — with Difflet's activation path; NxD's own DYNAMIC path is unusable (bug 2) | phase0 attempt 5: device-fp8 vs CPU-fp8 cosine 0.9999827 |
| A3 | neuronx-cc runs the fp8 layers as tensor-engine FP8 (per-step drops) | _pending_ | |
| A4 | fp8 tensors survive sharding / presharded safetensors / shared store | PASS (tp1); tp4 pending Phase 1 | phase0 shard sizes 749,976 B vs 1,135,864 B |
| A5 | `--experimental-unsafe-fp8e4m3fn-as-fp8e4m3` reaches the compiler | PASS | phase0 `arms.fp8.compiler_args` |
| Q | FP8 output quality vs bf16 | _pending_ | |
| P | compile / e2e / per-step, bf16 vs fp8 | _pending_ | |
| S | `difflet serve --quant fp8` serves the same result | _pending_ | |

## Bug ledger

| # | SHA | symptom (device) | fix |
|---|---|---|---|
| 1 | `dece4f4` | `AssertionError: Unsupported activation quantization type: DYNAMIC` at NeuronConfig | emit NxD's enum value `"dynamic"`; test against the real validator |
| 2 | `6fe599d` | fp8 trace: `Check failed: input_sizes.size() <= output_sizes.size() (4 vs. 3)` in NxD's DYNAMIC forward | Difflet-owned per-tensor dynamic activation quantization in subclassed NxD layers |
| 3 | `959f2cc` | weight-only fp8 output all NaN; 44.7 % of fp8 weights above 240 | absmax law saturates at Trainium's e4m3 max 240; range in checkpoint identity |
| 4 | `cfc1adf` | fp8-dyn load: `expected shape torch.Size([128, 1]) for ...to_q.scale but found torch.Size([1])`; XLA `amax()` no-op | NeuronConfig keeps the spec granularity unless the quantized MLP kernel is on; explicit-dims absmax |
| 5 | `42ae642` | fp8 arms: 316/400 weight-only dots `F32 × F32`; biased quantized linears promote the block to fp32 (DiT step 733 / 822 ms vs 573) | type the quantized layers from the live weight dtype (bias bf16) |
| 6 | `c3357c6` | after bug 5, `difflet compile` would reuse the old NEFF and the store would relink the old fp32-bias shards | `QUANT_LAYER_SCHEMA` in the fp8 stage cache key and the store key (fp8 only, additive) |
| – | `5fe2a49` | probe: text traced at 16 tokens vs backbone's 512; no device output kept | probe script |
| – | `f21d79b` | `benchmark.bench` / `cold_warm_e2e`: `FileNotFoundError: /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/difflet` (harness rot) | adapter uses its own interpreter's `difflet`; A/B runner `--force-compile` |
| – | not fixed | **Neuron VAE decoder at 480×832×9 degrades frames 2–8 in bf16** (host VAE of the same latents is clean) | follow-up for the Wan VAE stage; `--host-vae` works |
