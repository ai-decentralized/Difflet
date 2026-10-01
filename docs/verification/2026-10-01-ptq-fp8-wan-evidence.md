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

_pending_

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

_pending_

## Phase 3 — performance

_pending_

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
| – | `5fe2a49` | probe: text traced at 16 tokens vs backbone's 512; no device output kept | probe script |
