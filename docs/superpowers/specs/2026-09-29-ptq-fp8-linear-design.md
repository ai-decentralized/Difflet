# PTQ FP8 for DiT linear layers — design

**Date:** 2026-09-29
**Status:** implemented on CPU; device verification pending (see
`docs/verification/2026-09-29-ptq-fp8-wan-plan.md`). Written without a live
review round: every decision below is flagged for the maintainer to overturn.

## Goal

Post-training FP8 quantization of the DiT backbone's linear layers, matching the
PTQ half of FastVideo's QAD work, so Difflet can report on Trainium2:

- accuracy: bf16-vs-fp8 matmul error (MSE + cosine) per linear, latent-level
  error, and PSNR / SSIM / LPIPS of the generated output against the bf16 run;
- speed: compile time, e2e cold / warm, and DiT per-step time, bf16 vs fp8;
- through both the CLI (`difflet compile/generate/run`) and `difflet serve`.

Comparison model: **Wan 2.1 14B** (`Wan-AI/Wan2.1-T2V-14B-Diffusers`). FastVideo's
QAD checkpoints are all Wan-2.1-family DiTs (same block structure as 14B), and it
is the only Difflet model with any FastVideo QAD artifact. Wan 2.2 A14B shares the
code path (two transformers, two quantized checkpoints).

## What FastVideo does (research, 2026-09-29)

FastVideo's runtime PTQ (`fastvideo/layers/quantization/fp8_config.py`) is
deliberately simple: `float8_e4m3fn`, absmax scales (`amax / 448`), **no
calibration set**, weights quantized once at load, activations quantized
dynamically per call. Granularity `tensor` (default) = per-tensor weight scale +
per-tensor dynamic activation scale; `channel` = per-output-row weight scale +
per-token dynamic activation scale. Only attention q/k/v/out and FFN in/out are
quantized; patch embedding, time/text embedders, adaLN modulation
(`scale_shift_table`), norms and `proj_out` stay bf16. The 1.8 s headline is a
QAD-trained 3-step student with NVFP4, not PTQ; PTQ alone is what we mirror.

## Decisions

**D1 — Quantization scheme = FastVideo FP8Config parity.** `fp8_e4m3` weights,
absmax scale, `weight_granularity ∈ {tensor, channel}` (default `tensor`),
`activation ∈ {dynamic, none}` (default `dynamic`, per-tensor absmax at run
time; `none` = weight-only). No calibration data. Targets (Difflet names):
`to_q, to_k, to_v, to_out.0, ffn.net_in, ffn.net_out`. Static (calibrated)
activation scales are out of scope until the dynamic reduction shows up in a
profile.

**D2 — Device execution reuses NxD's quantized parallel linears; no NKI change.**
The vendored NxDI plumbing already carries `NeuronConfig.quantized`,
`quantization_type`, `quantization_dtype="f8e4m3"`,
`activation_quantization_type`, `modules_to_not_convert`, and the compiler flag
`--experimental-unsafe-fp8e4m3fn-as-fp8e4m3`. Difflet's own model instances
(`BaseModelInstance(module_cls=_create_model)`) never called the conversion, so the
change is one hook in the backbone's `_create_model`: after building the bf16
float model, call `neuronx_distributed.quantization.quantize.convert` with the
q-config NxDI builds (`difflet/backends/trainium/core/quant.py` mirrors
`DecoderModelInstance.load_module`). The FP8 GEMM is then whatever neuronx-cc
lowers for NxD's `QuantizedColumnParallel` / `QuantizedRowParallel` — AWS's
production FP8 path on trn2 for LLMs. The repository's NKI MX kernels are
Trainium3 `nc_matmul_mx` microscaling and are not involved.

Answer to "framework-only?": yes. The only way an NKI kernel enters is if the
device probe (plan Phase 0) shows the NxD path compiling but not running FP8 on
the tensor engine (no per-step speedup); that fallback is documented, not built.

**D3 — Quantized checkpoint is produced offline on CPU by Difflet.**
`difflet quantize` (also run automatically by `difflet compile --quant fp8`)
rewrites the HF `transformer/` safetensors: for each target linear `weight` →
`float8_e4m3fn` and a new `weight_scale` (`float32`, shape `[1]` per-tensor or
`[out, 1]` per-channel); everything else copied unchanged. Output:
`<cache_dir>/quantized/<model-slug>/<subfolder>/<fp8-<granularity>>-<hash8>/`
with `model.safetensors`, `config.json`, `difflet_quant.json`. The vendored loader
(`NeuronApplicationBase.get_state_dict`) already renames `.weight_scale` → `.scale`
and `checkpoint_loader_fn` already skips casting fp8 tensors and scales. The
activation mode does not change the checkpoint, so one checkpoint serves both
`dynamic` and `none`.

**D4 — Model scope: Wan only, mechanism generic.** `difflet/quant` is model-agnostic
(module-name suffix targets; NxD `convert` targets all parallel linears, which for
Wan is exactly the target set). CLI and serving reject `--quant` for other models
with a "not wired" message; wiring another model is: pass `quant` kwargs to its
backbone config and add the `_create_model` hook.

**D5 — Cache identity.** The `quant` spec dict (`format`, `weight_granularity`,
`activation`, `targets`) enters the staged transformer stage cache inputs and the
pipeline/serving `application_kwargs` hash, additive-only (absent → key unchanged,
every existing cache stays valid). `quant_checkpoint_dir` is a machine path and is
runtime-only (excluded from the hash). The shared weight store key gains
`quantized_checkpoint` (realpath) when quantized, so fp8 shards never collide with
bf16 shards of the same source.

**D6 — CPU backend emulation.** `difflet.quant.fake_linear.FakeQuantLinear`
implements the same math (fp8 RTNE via `torch.float8_e4m3fn`, absmax scales,
dynamic per-tensor activations) on plain `nn.Linear` subclasses — the CPU backend's
`ColumnParallelLinear` / `RowParallelLinear`. It is exact for the algorithm; only
fp32 accumulation order differs from the device, so CPU metrics are the reference
the device numbers are checked against.

**D7 — Metrics and timing.**
- per-linear: MSE, cosine, max-abs, mean-abs, relative L2, SNR(dB) of the fp8
  matmul vs the bf16 matmul on captured activations (`scripts/ptq_linear_error_sweep.py`,
  CPU, real weights);
- latent: MSE / cosine / SNR of the DiT output latents (`latents.pt` from
  `--keep-work-dir`) vs the bf16 run, same seed;
- output: PSNR, SSIM (11×11 Gaussian σ=1.5, per frame, mean), LPIPS (`alex`, as
  FastVideo) vs the bf16 output (`scripts/ptq_compare_outputs.py`);
- time: compile seconds (benchmark adapter wall + neuronx-cc breakdown), e2e cold
  (page cache dropped) and warm (`benchmark/cold_warm_e2e.py`), and DiT per-step as
  inter-call deltas of the real loop with step 0 excluded, emitted by the Wan
  pipeline as `[wan] dit-step-seconds: [...]` and parsed by the benchmark adapter.

FastVideo's "5-second video in 1.8 s" is warm per-request latency (text encode +
3-step CFG-off DiT + tiny-VAE decode; excludes load, compile, mp4 encode). Its
Difflet analog is the resident-serving request latency or e2e-warm minus weight
load, not the staged `difflet generate` wall time (which reloads weights in two
subprocesses).

## Interfaces

CLI (compile / generate / run / serve): `--quant fp8`,
`--quant-granularity {tensor,channel}` (default tensor),
`--quant-act {dynamic,none}` (default dynamic). New `difflet quantize --model-id
… [--quant fp8 --quant-granularity …] [--cache-dir …] [--force]`.

Python: `NeuronWanApplication(..., quant={"format": "fp8_e4m3",
"weight_granularity": "tensor", "activation": "dynamic"}, quant_checkpoint_dir=…)`;
`difflet.quant.QuantSpec`, `quantize_checkpoint_dir`, `quantize_module_`,
`tensor_error_metrics`, `psnr`, `ssim`, `lpips_distance`.

Serving: `ServeOptions.quant / quant_granularity / quant_act` →
`ServingProfile.quant: QuantSpec | None` → Wan adapter application kwargs and the
`generation` compile identity.

## Files

- new `difflet/quant/{__init__,spec,fp8,checkpoint,fake_linear,metrics}.py`
- new `difflet/backends/trainium/core/quant.py`
- `difflet/backends/trainium/wan/backbone.py` (hook + compiler flag)
- `difflet/backends/trainium/core/shared_weights.py` (key)
- `difflet/models/wan/application.py` (config kwargs, checkpoint resolution)
- `difflet/models/wan/pipeline.py` (per-step timing lines)
- `difflet/pipeline/compile_cache.py` (`quant_checkpoint_dir` runtime-only)
- `difflet/cli/{main,stage}.py`, `difflet/cli/orchestrators/wan.py`,
  `difflet/cli/dp/router.py`, new `difflet/cli/quantize.py`
- `difflet/serving/{options,types,model_registry}.py`, `difflet/cli/serve.py`,
  `difflet/serving/models/wan.py`
- `benchmark/models.py`, `benchmark/adapters/trainium.py`
- scripts: `ptq_linear_error_sweep.py`, `ptq_compare_outputs.py`,
  `ptq_fp8_device_probe.py`, `ptq_fp8_ab.py`
- tests: `tests/unit/quant/`, `tests/unit/cli/test_cli_quant.py`,
  `tests/unit/serving/test_serve_quant.py`, benchmark/pipeline additions

## Assumptions the device probe must confirm

- A1 NxD per-channel `scale` shape is `[out, 1]` (per-tensor `[1]`) and the key
  is `<layer>.scale` — matches llm-compressor checkpoints NxDI documents as
  loadable. A shape mismatch surfaces at the tiny-model probe as a load error;
  the fix is a reshape in `convert_backbone_state_dict`.
- A2 `activation_quantization_type="DYNAMIC"` is per-tensor absmax at run time.
  The CPU emulation assumes that; the probe reports CPU-vs-device cosine.
- A3 neuronx-cc 2.26 lowers the NxD fp8 layers to tensor-engine FP8 on trn2 (the
  `--experimental-unsafe-fp8e4m3fn-as-fp8e4m3` flag exists for it). If per-step
  time does not drop, the layers compiled as dequant-to-bf16; that is the case
  where an NKI FP8 matmul kernel would be needed.
- A4 `ModelBuilder.shard_checkpoint` handles `float8_e4m3fn` tensors (torch
  slicing of fp8 works; safetensors 0.8 serializes F8_E4M3).
- A5 The duplicate `--internal-hlo2tensorizer-options` (Wan's explicit compiler
  args + ModelWrapper's append) resolves to a command line that still carries the
  fp8 flag; Wan's own args now include it so either precedence works.

## Out of scope

Static activation scales, per-token activation scales on device (NxD API not
confirmed), FLUX / Qwen / Hunyuan / LTX wiring, NVFP4, attention quantization,
README feature-matrix row (added after device PASS).
