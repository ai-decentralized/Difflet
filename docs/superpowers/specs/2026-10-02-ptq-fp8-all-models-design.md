# FP8 PTQ for every Difflet backbone — design

**Date:** 2026-10-02
**Status:** approved design (chat review 2026-10-02); implementation plan to follow.
**Builds on:** `2026-09-29-ptq-fp8-linear-design.md` (the Wan path) and the trn2 evidence
in `docs/verification/2026-10-01-ptq-fp8-wan-evidence.md`.

## Goal

Extend FP8 post-training quantization of the DiT linears from Wan to every model Difflet
runs on Trainium, and verify each one on the device the same way Wan was verified, so the
support matrix can say per model whether fp8 is better than bf16 and by how much:

- models: **FLUX.1-dev, Qwen-Image, HunyuanVideo 1.0, LTX-2 (single-transformer mode)** get
  the wiring; **Wan 2.2** is verified on the existing Wan wiring (its CLI path runs the
  high-noise `transformer` only);
- modes: weight-only (`--quant-act none`) and dynamic per-tensor activations
  (`--quant-act dynamic`), per-tensor weights;
- metrics per model and mode, bf16 vs fp8 on the same host: DiT per-step, cold / warm e2e,
  cold weight load, compile time, and PSNR / SSIM / LPIPS of the generated image or video
  against the bf16 output;
- explicitly out of scope: HunyuanVideo 1.5 (compile/generate are a scaffold), the
  segmented LTX-2 / HunyuanVideo 1.5 runtimes (plain `nn.Linear`, loaders bypass the
  quantized checkpoint), per-channel weights (measured no better than per-tensor on Wan), an
  NKI FP8 matmul kernel, and the neuronx-cc 2.27 experiment (parked).

## What the Wan campaign established (inputs to this design)

- The device math is right: per step, device-fp8 reproduces the CPU fp8 reference
  (44–46 dB); per linear the fp8 matmul error is ~30 dB SNR.
- Weight-only fp8 is at bf16 per-step parity with half the transformer bytes and −46 % cold
  load; dynamic activations cost +15 % per step on neuronx-cc 2.26 (fp8 × fp8 dots lower
  slower than bf16 ones). Expect the same shape of result elsewhere; the point of the
  campaign is to measure it, not assume it.
- Six Difflet bugs were found on device (enum value, NxD DYNAMIC path, 240 range,
  granularity override, fp32-typed layers, schema in cache keys). All are in the shared
  core, so the other models inherit the fixes; the per-model work is wiring plus the gaps
  below.

## Why a straight copy of the Wan wiring is not enough (code survey 2026-10-02)

1. **The convert step is unscoped.** `quantize_traced_model_` swaps every NxD
   `ColumnParallelLinear` / `RowParallelLinear` in the model. On Wan that set equals the
   quantization targets (`time_proj`, `proj_out` are plain `nn.Linear`). FLUX, Qwen-Image and
   HunyuanVideo also build embedders, adaLN/modulation linears, `norm_out.linear`,
   timestep MLPs and the root `proj_out` as parallel linears; swapping them would create
   quantized layers with no fp8 weights in the checkpoint. NxD's `modules_to_not_convert` is
   a leaky substring exclusion; its `include=` (fnmatch on the full module name) is the
   correct scoping primitive.
2. **`DEFAULT_TARGETS` is Wan-specific** (`ffn.net_in`, `ffn.net_out`, `ffn.net.0.proj`,
   `ffn.net.2`). Other models use `ff.*`, `ff_context.*`, `img_mlp.*`, `txt_mlp.*`,
   `audio_ff.*`, `add_{q,k,v}_proj`, `to_add_out`, `proj_mlp` and a fused single-block
   `proj_out`; a bare `proj_out` suffix would also match the root projection, and
   HunyuanVideo's token refiner has its own `to_q/to_k/to_v/to_out.0`.
3. **Fused weights split at load.** FLUX and HunyuanVideo split each single block's
   `proj_out` into `proj_out_attn` / `proj_out_mlp` along the input dimension in their
   state-dict converters; the fp8 weight has to be split the same way and its scale
   carried to both halves.
4. **Compiler flag, probes, gates.** Only Wan's backbone adds the fp8 hlo2tensorizer flag,
   rejects the adaptive TeaCache probe under quant, and threads `quant` through its CLI
   orchestrator, serving module, cache inputs and benchmark slugs.

## Design

### 1. Core (model-agnostic)

**Per-model targets.** New `difflet/quant/targets.py`:

```python
TARGETS_BY_MODEL: dict[str, tuple[str, ...]] = {
    "wan": DEFAULT_TARGETS,                              # unchanged
    "flux": ("transformer_blocks.*.attn.to_q", ... "single_transformer_blocks.*.proj_out"),
    "qwen_image": (...), "hunyuan_video": (...), "ltx_2": (...),
}
def targets_for(model_type: str) -> tuple[str, ...]   # KeyError -> ValueError with the list
```

`QuantSpec.for_model(model_type, **fields)` builds a spec with that model's targets;
`QuantSpec.from_args` gains a `model_type` argument so the CLI, serving and benchmark build
specs with the right set. `QuantSpec.matches` accepts two pattern kinds: a dotted suffix
(today's behaviour, `to_q` matches `blocks.3.attn1.to_q`) and a glob containing `*`
(`fnmatch` against the full dotted module name, so
`single_transformer_blocks.*.proj_out` matches the block projections and not the root
`proj_out`). The targets stay part of `checkpoint_identity`, so every model's fp8
checkpoint hashes differently and a Wan checkpoint can never be mistaken for a FLUX one.

Target sets (module names in the Difflet modeling, after TP replacement; the offline
quantizer sees the HF names, which coincide except where noted):

| model | attention | FFN | notes |
|---|---|---|---|
| FLUX | double blocks `attn.{to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out}`; single blocks `attn.{to_q,to_k,to_v}`, `proj_mlp`, `proj_out` (HF) → `proj_out_attn` + `proj_out_mlp` (device) | `ff.net.0.proj`, `ff.net.2`, `ff_context.net.0.proj`, `ff_context.net.2` | root `proj_out`, `x_embedder`, `context_embedder`, `norm*.linear`, `time_text_embed.*` stay bf16 |
| Qwen-Image | `attn.{to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out}` | `img_mlp.net.0.proj`, `img_mlp.net.2`, `txt_mlp.net.0.proj`, `txt_mlp.net.2` | `img_in`, `txt_in`, `img_mod.1`, `txt_mod.1`, `norm_out.linear`, timestep MLP stay bf16; device names carry the `transformer.` prefix the converter adds |
| HunyuanVideo 1.0 | as FLUX, anchored to `transformer_blocks.*` / `single_transformer_blocks.*` so the token refiner (`context_embedder.*`) is excluded | `ff.*`, `ff_context.*`, `proj_mlp`, single-block `proj_out` split | refiner, `x_embedder`, root `proj_out`, modulation stay bf16 |
| LTX-2 | `{attn1,attn2,audio_attn1,audio_attn2,audio_to_video_attn,video_to_audio_attn}.{to_q,to_k,to_v,to_out.0}` | `ff.net.0.proj`, `ff.net.2`, `audio_ff.net.0.proj`, `audio_ff.net.2` | only targets are parallel linears (like Wan); `transformer.` prefix on device |

Text-stream linears of double-stream blocks (`add_*_proj`, `to_add_out`, `ff_context`,
`txt_mlp`) are quantized: they are the same FastVideo layer set, and they run dense.

**Scoped convert.** `quantize_traced_model_(model, neuron_config, spec)` passes
`include=[pattern(t) for t in spec.targets]` to NxD `convert()` — `*.<suffix>` plus the
bare suffix for suffix targets, the glob itself for glob targets — and no longer uses
`modules_to_not_convert` (NxD forbids both). Wan's behaviour is unchanged (asserted by a
test on a tiny Wan model: the same 20 layers swap). The backbone obtains the spec from its
config (`neuron_config.quant_targets`, a new additive field carried next to the existing
quant fields; absent for bf16 so no existing identity changes).

**Checkpoint converters.** A helper `split_fused_proj_out(state_dict, prefix, out_attn,
out_mlp, cols)` in `difflet/quant/checkpoint.py` splits `<prefix>.weight` (fp8 or bf16)
along dim 1 and, when `<prefix>.scale` exists, copies it to both halves (valid for `[1]`
and `[out, 1]` scales because the split is along the input dim). FLUX and HunyuanVideo
converters use it; Qwen and LTX-2 converters prefix `.scale` keys exactly like `.weight`.

### 2. Per-backbone wiring (the Wan pattern, applied four times)

For each of FLUX, Qwen-Image, HunyuanVideo 1.0, LTX-2:

- config builder: `quant` + `quant_checkpoint_dir` kwargs → `neuron_config_kwargs(spec, dir)`
  merged into `NeuronConfig`, plus `quant_targets`;
- application: `__init__` reads `quant` / `quant_cache_dir`; `_quant_checkpoint_dir`,
  `ensure_quantized_checkpoints` (shared helper lifted from the Wan app into
  `difflet/quant/application_mixin.py` so the five apps share one implementation);
  `compile()` ensures the checkpoint first; adaptive TeaCache probe + quant → `ValueError`
  (probe configs are deep-copied before the quant fields are dropped, so a probe never
  inherits `quantized=True`);
- backbone `_create_model`: build → `.to(bf16)` → `quantize_traced_model_`;
  `get_compiler_args`: prepend `fp8_hlo2tensorizer_options(neuron_config)`;
- CLI orchestrator: pass `quant=spec.to_dict()`, `quant_cache_dir`, and add `quant` +
  `quant_layer_schema` to the transformer stage's cache inputs (Qwen/HunyuanVideo stage
  inputs; FLUX/LTX-2 `CacheSpec.application_kwargs`, which already hashes `quant`, gains
  `quant_layer_schema`); `QUANT_MODEL_TYPES = {"wan","flux","qwen_image","hunyuan_video","ltx_2"}`;
  HunyuanVideo 1.5 and segmented modes raise a fail-fast `ValueError` ("FP8 PTQ is not
  wired for …; run the single-transformer mode or drop --quant");
- serving: options gate follows `QUANT_MODEL_TYPES`; each serving module adds `quant` and
  `quant_cache_dir` to the application kwargs exactly as `serving/models/wan.py` does;
- benchmark: `<slug>_fp8` (dynamic) and `<slug>_fp8_wo` entries mirroring the bf16 entry.

### 3. Verification (per model, on trn2)

Reuse the benchmark harness and the Wan comparison tooling rather than generalising the
Wan A/B runner:

1. gate (host idle, nothing else on the device), `difflet quantize` for the model (CPU);
2. `python -m benchmark.bench --model <slug>[_fp8|_fp8_wo] --skip-download --iters 1` for
   bf16, fp8-wo, fp8-dyn (the bf16 compile is a cold compile on this host), then
   `benchmark.cold_warm_e2e` for each — per-step, cold / warm e2e, load, compile;
3. `scripts/ptq_compare_outputs.py` on the harness outputs: fp8-wo vs bf16, fp8-dyn vs
   bf16 (PSNR / SSIM / LPIPS; frame grids for video), plus a second bf16 run as the
   determinism control where the harness allows;
4. A4 fingerprint per model: presharded fp8 shard bytes vs bf16 and the `qf8e4m3` store
   entry;
5. serving smoke (`difflet serve --quant fp8 …`, one request, off-profile shape) last and
   best-effort, image models first.

Order on the device: FLUX → Qwen-Image → LTX-2 → HunyuanVideo → Wan 2.2, each model's
device run starting as soon as its wiring lands while the next model is wired. Evidence
goes to `artifacts/verification-2026-10-02/ptq-all/<model>/` and
`docs/verification/2026-10-02-ptq-fp8-all-models-evidence.md` (checkpointed and pushed per
model); the public result is a per-model row in the README feature table (PASS / LIMIT /
BLOCKED with the measured deltas).

### 4. Error handling

- Unsupported runtime or model type with `--quant`: `ValueError` naming the model and the
  supported list, before any compile.
- Target set with no matching linear in the built model (a renamed layer): the convert
  helper raises with the unmatched patterns, so a silent no-op cannot pass as fp8.
- Quantized checkpoint missing at generate/serve: existing `FileNotFoundError` with the
  `difflet quantize` command (unchanged).

### 5. Testing

TDD per model, mirroring the Wan test set:

- `tests/unit/quant/test_targets.py`: every model's target set resolves, globs vs
  suffixes match as specified (root `proj_out` excluded, refiner excluded), identities
  differ per model;
- `tests/unit/quant/test_checkpoint_<model>.py`: quantizer on a tiny HF-named state dict →
  converter → Difflet names: fp8 weights where expected, scales carried (incl. the
  `proj_out` split), non-targets untouched, no leftover HF keys;
- `tests/unit/quant/test_fake_linear.py`: per model, exactly N target linears swap on the
  tiny CPU model, embedders/modulation/proj_out untouched, quantized forward cosine > 0.95;
- `tests/unit/quant/test_trainium_quant.py`: scoped convert passes `include` built from the
  spec; Wan still swaps the same layer set;
- `tests/unit/cli/test_cli_quant.py` and `tests/unit/serving/test_serve_quant.py`: per
  model, stage identity unchanged for bf16 and extended by exactly `quant` +
  `quant_layer_schema` for fp8; quant kwargs reach the application; serving profile
  carries the spec; the "rejects FLUX" test flips to "accepts FLUX, rejects
  hunyuan_video_15";
- `tests/unit/test_benchmark_quant.py`: all fp8 slugs mirror their bf16 partner.

### 6. Deliverables

1. Code: core targets/scoping, four backbone wirings, converters, CLI/serving/benchmark
   entries, tests — one commit per model plus one for the core, bugs found on device as
   their own commits.
2. Evidence doc + curated artifacts per model, pushed per model.
3. README feature table rows for FP8 PTQ per model with measured deltas.
