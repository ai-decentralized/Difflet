# MXFP8 on the video DiTs (LTX-2, Wan, HunyuanVideo, HunyuanVideo-1.5) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `--quant mxfp8` (the branch `feat/fp8-native-trn3` recipe: nkilib `matmul_mxfp8` DiT
linears, Trainium3, compile mode) on every native-neuron model except MiniMax-H3, with device
evidence per model; this plan covers the shared layer (phase A) and the first model, LTX-2
(phase B). Wan, FLUX/HV1 `proj_out`, HV1 and HV1.5 get their own plans once LTX-2 is on the
device (roadmap at the end).

**Architecture:** the quantization mechanism stays as it is on the branch (spec → offline
quantized checkpoint directory → FP8 twin layers swapped on the meta module → `_after_load`
packs). What changes: the per-model wiring that Qwen-Image and FLUX each hand-rolled (transformer
application, pipeline application, orchestrator, HBM) moves into the base classes and two
helpers, so a new model declares its target table and three attributes. Phase B wires LTX-2
through those helpers and takes it to the device.

**Tech Stack:** PyTorch-native neuron backend (`difflet/backends/neuron`), nkilib
`matmul_mxfp8`, safetensors, pytest on the CPU (`compute="dequant"` toys), trn3.48xlarge
(LNC=2 is the host default, 16 devices) for the device phase.

**Spec:** this conversation's plan (2026-11-07) and its review; the branch's evidence doc
`docs/verification/2026-10-31-qwen-image-fp8-native-trn3.md` for the recipe and its limits.

## Global Constraints

- Branch `feat/mxfp8-video-models`, based on `feat/fp8-native-trn3` (`03b43ba`); every commit is
  pushed to `origin` right after it lands; the status page is updated per completed task.
- MXFP8 is Trainium3 + compile mode only (decision D43, `require_fp8_supported`); the eager
  top level of every model (embedders, modulation, `proj_out`) stays bf16.
- No q/k/v fusion on any new model: their attention processors call `to_q/to_k/to_v` directly
  (`prepare_fp8_kernels_(fuse=False)`); only Qwen-Image's processor reads `fused_qkv`.
- Every quantized weight's K (per rank for a row-parallel layer) is a multiple of **128**
  (`difflet/quant/mx._tile_sizes`), not 32: a spec that cannot satisfy that at the launch's tp
  is refused on the host, naming the layer, K and tp.
- The CLI's default command lines stay byte-identical: quant flags are forwarded only when
  given, cache keys gain fields only with `--quant` (`quant_cache_fields`).
- Device runs: 50 steps, seed 42, LNC=2 (the host default), two generates per arm (determinism);
  the quick quality signal is mxfp8-vs-device-bf16; the final PSNR/SSIM reference is diffusers
  on the CPU in fp32 at the same config, run once at the end of the whole campaign.
- Qwen-Image and FLUX must stay bit-identical through the refactor: the Qwen mxfp8 generate
  (tp4, 1024x1024, 50 steps, seed 42) reproduces its md5 before phase B starts.

## Review Focus

1. A `TorchNeuronApplicationBase` subclass without `quant_model` given `quant=` must refuse at
   construction (not silently run bf16) — test in Task A3.
2. A spec whose targets match no linear, or match a linear whose K is not a multiple of 128,
   must fail on the driver before torchrun, with the layer name — tests in Task A2 and B4.
3. `--quant` with `DIFFLET_LTX2_TP_REPLICATE_ATTN`/`_MLP` set would load a per-rank K layout
   the writer did not produce: refused in the application — test in Task B2.
4. A DP worker or a stage launch that drops `--quant` runs the bf16 DiT against the fp8 stage
   key: the forwarding tests in Task B4 cover LTX-2's stage and DP command lines.
5. The HBM check with `--quant` must count the quantized directory's bytes as stored, for the
   DiT alone and beside the device text encoder — tests in Task A4 and B3.

---

# Phase A — the shared layer

### Task A1: K multiple of 128 is checked where the layer is built

**Files:**
- Modify: `difflet/backends/neuron/ops_impl/fp8_linear.py:66-69` (`_init_tensors`)
- Modify: `difflet/quant/mx.py:36` (export `LAYOUT_K = 128` beside `BLOCK = 32`)
- Test: `tests/unit/backends/test_neuron_fp8_linear.py` (`test_unknown_compute_and_odd_k_are_refused`)

**Interfaces:**
- Produces: `difflet.quant.mx.LAYOUT_K: int = 128`; `_init_tensors` raises
  `ValueError("mxfp8 needs in_features a multiple of 128 (the MX layout), got {in_local}")`.

- [ ] **Step 1: Extend the failing test**

```python
def test_unknown_compute_and_odd_k_are_refused():
    ...  # existing body
    with pytest.raises(ValueError, match="multiple of 128"):
        Q.FP8Linear(64, 8, compute="dequant")        # a multiple of 32 but not of 128
```

- [ ] **Step 2: Run it** — `pytest tests/unit/backends/test_neuron_fp8_linear.py -k odd_k -v`; expected FAIL (64 is accepted today).
- [ ] **Step 3: Implement** — `mx.LAYOUT_K = 128`; `_init_tensors` checks `in_local % LAYOUT_K`.
- [ ] **Step 4: Run** — the same command; expected PASS. Then `pytest tests/unit/quant tests/unit/backends/test_neuron_fp8_linear.py -q`; expected all green.
- [ ] **Step 5: Commit** — `fix(quant): the MX layout needs K in multiples of 128, checked where the layer is built`; push.

### Task A2: `quantize_module_` matches by checkpoint key and names the first bad layer

**Files:**
- Modify: `difflet/backends/neuron/ops_impl/fp8_linear.py:280-301` (`quantize_module_`)
- Test: `tests/unit/backends/test_neuron_fp8_linear.py`

**Interfaces:**
- Produces: `quantize_module_(module, spec, *, key_fn: Callable[[str], str] | None = None) -> list[str]`
  — `key_fn` maps a module name to its checkpoint key before `spec.matches`; the returned names
  are still module names. A twin whose construction raises `ValueError` is re-raised as
  `ValueError(f"{name}: {exc} (tp too high for this layer at this spec?)")`.

- [ ] **Step 1: Write the failing tests**

```python
def test_quantize_module_matches_through_a_key_function():
    module = torch.nn.Module()
    module.net_in = torch.nn.Linear(128, 16)
    spec = QuantSpec(targets=(r"net\.0\.proj",), compute="dequant")
    rename = lambda name: name.replace("net_in", "net.0.proj")
    assert Q.quantize_module_(module, spec, key_fn=rename) == ["net_in"]
    assert isinstance(module.net_in, Q.FP8Linear)

def test_quantize_module_names_the_layer_whose_k_is_wrong():
    module = torch.nn.Module()
    module.proj = torch.nn.Linear(64, 16)
    with pytest.raises(ValueError, match=r"proj: mxfp8 needs in_features a multiple of 128"):
        Q.quantize_module_(module, QuantSpec(targets=("proj",), compute="dequant"))
```

- [ ] **Step 2: Run** — `pytest tests/unit/backends/test_neuron_fp8_linear.py -k "key_function or k_is_wrong" -v`; expected FAIL (`key_fn` unknown; bare message).
- [ ] **Step 3: Implement** in `quantize_module_` (apply `key_fn` only for matching; wrap the twin construction).
- [ ] **Step 4: Run** — same selection, PASS; then the whole file.
- [ ] **Step 5: Commit** — `feat(fp8): quantize_module_ matches through a checkpoint-key function and names a layer whose K is wrong`; push.

### Task A3: the transformer application base owns the quant wiring

**Files:**
- Modify: `difflet/backends/neuron/core/application_base.py:125-172` (`__init__`), `:221-271` (`load`), `:203-206` (`_after_load`)
- Modify: `difflet/backends/neuron/qwen_image/transformer.py:164-236` (drop its quant code; set the attributes)
- Modify: `difflet/backends/neuron/flux/transformer.py:49-98` (same)
- Test: `tests/unit/backends/test_neuron_fp8_linear.py` (new base-class tests), `tests/unit/backends/test_neuron_qwen_fp8.py`, `tests/unit/backends/test_neuron_flux_fp8.py` (unchanged, must pass)

**Interfaces:**
- Produces, on `TorchNeuronApplicationBase`:
  - class attributes `quant_model: str | None = None` (the `difflet.quant.targets` key; `None`
    means the model has no FP8 support), `quant_label: str = ""` (the name in refusals, e.g.
    `"LTX-2"`; defaults to `quant_model`), `quant_fuse_qkv: bool = False`;
  - `__init__(..., quant: QuantSpec | dict | None = None, **kwargs)`: stores `self.quant`
    (`QuantSpec.from_dict` for a dict); with a spec and `quant_model is None` raises
    `NotImplementedError(f"{type(self).__name__} has no MXFP8 target layers")`; with a spec
    calls `require_fp8_supported(self.quant_label or self.quant_model, self.device.type, self.exec_mode)`;
    `self.quantized_linears: list[str] = []`;
  - `load`: after `build_on_meta(self.build_module)`, with a spec:
    `self.quantized_linears = quantize_module_(module, self.quant, key_fn=self.checkpoint_key)`;
  - `_after_load(self, module)`: default body `if self.quant is not None: prepare_fp8_kernels_(module, fuse=self.quant_fuse_qkv)`.
- Qwen-Image's application: `quant_model = "qwen_image"`, `quant_label = "Qwen-Image"`,
  `quant_fuse_qkv = True`; no `quant` parameter, no `build_module`/`_after_load` quant code.
  FLUX's: `quant_model = "flux"`, `quant_label = "FLUX"`.

- [ ] **Step 1: Write the failing tests** (in `test_neuron_fp8_linear.py`, CPU, tp=1, using `tests/unit/backends/_neuron_fp8_workers.py`'s `FP8Toy` or a two-linear toy built on meta)

```python
class _ToyApp(TorchNeuronApplicationBase):
    quant_model = "toy"; block_attrs = ()
    def build_module(self): return _TwoLinears(128, 16)
    def checkpoint_dir(self): return str(self._dir)
    def get_example_inputs(self): return (torch.randn(2, 128),)

def test_base_refuses_quant_without_a_target_table():
    class NoQuant(_ToyApp): quant_model = None
    with pytest.raises(NotImplementedError, match="no MXFP8 target layers"):
        NoQuant(model_path="/x", device="cpu", exec_mode="compile", quant=_SPEC)

def test_base_quantizes_after_build_and_prepares_after_load(tmp_path):
    app = _ToyApp(..., quant=_SPEC)   # _SPEC targets "fc1", compute="dequant"
    app.load()
    assert app.quantized_linears == ["fc1"]
    assert app.module.fc1.weight is None and app.module.fc1.weight_bf16 is not None
```

- [ ] **Step 2: Run** — `pytest tests/unit/backends/test_neuron_fp8_linear.py -k base -v`; expected FAIL.
- [ ] **Step 3: Implement** the base-class changes; delete the duplicated code from the Qwen and FLUX applications; keep their docstrings' D43 note.
- [ ] **Step 4: Run** — `pytest tests/unit/backends/test_neuron_fp8_linear.py tests/unit/backends/test_neuron_qwen_fp8.py tests/unit/backends/test_neuron_flux_fp8.py tests/unit/backends/test_neuron_qwen_image_l1.py -q`; expected all PASS (the Qwen/FLUX fp8 tests unchanged).
- [ ] **Step 5: Commit** — `refactor(neuron): the application base owns the MXFP8 wiring (quant_model, quant_fuse_qkv, _after_load)`; push.

### Task A4: one resolver for the pipeline side and one HBM helper for quantized directories

**Files:**
- Create: `difflet/backends/neuron/core/quant.py`
- Modify: `difflet/backends/neuron/hbm.py` (after `safetensors_bytes`)
- Modify: `difflet/backends/neuron/qwen_image/hbm.py:57-75` (delegate), `difflet/backends/neuron/ltx_2/hbm.py:46-55` and `:65-82` (add `dit_dir`), `difflet/backends/neuron/hunyuan_video/hbm.py:29-38` (add `dit_dir`)
- Modify: `difflet/models/qwen_image/neuron_application.py:567-593`, `difflet/models/flux/neuron_application.py:169-196` (both `_resolve_quant` → the helper)
- Test: `tests/unit/backends/test_neuron_fp8_linear.py` (resolver), `tests/unit/backends/test_neuron_hbm.py` (quantized bytes)

**Interfaces:**
- Produces `difflet.backends.neuron.core.quant.resolve_quant(model: str, quant, quant_checkpoint, transformer_path: str, *, tp_degree: int) -> tuple[QuantSpec | None, str]`
  — the body of today's two `_resolve_quant` methods (`"mxfp8"` → `default_spec(model, tp_degree=tp_degree)`;
  dict/`QuantSpec` as given; `quant_checkpoint` checked with `check_quantized_dir`, else
  `prepare_quantized_checkpoint(..., create=False)`; `quant is None and quant_checkpoint` →
  `ValueError("quant_checkpoint needs quant ('mxfp8' or a QuantSpec)")`).
- Produces `difflet.backends.neuron.hbm.quantized_dir_bytes(dit_dir, *, tp_degree) -> int`
  (`safetensors_bytes(dit_dir) // tp`) and `quantized_dir_fits(dit_dir, *, tp_degree) -> tuple[bool, int, int]`,
  plus `Component.stored_in(directory: str | Path) -> Component` returning a component whose
  `numel`/bytes come from that directory's safetensors as stored (a new field
  `stored_dir: str | None = None`; `components_bytes` uses `safetensors_bytes(stored_dir)` for it).
- Each model's `dit_fits(model_dir, *, tp_degree, dit_dtype_bytes=..., dit_dir=None)`: with
  `dit_dir`, returns `quantized_dir_fits(dit_dir, tp_degree=tp_degree)`. LTX-2's
  `device_text_encoder_components(..., dit_dir=None)` returns `DIT.stored_in(dit_dir)` for the DiT
  when given; `device_text_encoder_fits(..., dit_dir=None)` passes it.

- [ ] **Step 1: Write the failing tests**

```python
def test_resolve_quant_default_spec_and_checkpoint(tiny_quantized_toy, tmp_path):
    spec, path = resolve_quant("flux", "mxfp8", None, transformer_dir, tp_degree=2)
    assert spec == default_spec("flux", tp_degree=2) and path == str(quantized_dir)
    with pytest.raises(ValueError, match="quant_checkpoint needs quant"):
        resolve_quant("flux", None, "/q", transformer_dir, tp_degree=1)

def test_quantized_dir_bytes_count_fp8_as_stored(tiny_quantized_toy):
    # fp8 weight + uint8 scale (K/32) + bf16 rest, over tp
    assert quantized_dir_bytes(quantized_dir, tp_degree=2) == safetensors_bytes(quantized_dir) // 2

def test_component_stored_in_a_directory(tiny_quantized_toy):
    comp = Component("transformer", 2).stored_in(quantized_dir)
    assert components_bytes(model_dir, (comp,), tp_degree=1) == safetensors_bytes(quantized_dir)
```

- [ ] **Step 2: Run** — `pytest tests/unit/backends/test_neuron_fp8_linear.py -k resolve tests/unit/backends/test_neuron_hbm.py -k "quantized or stored_in" -v`; expected FAIL (imports).
- [ ] **Step 3: Implement**; replace the two `_resolve_quant` bodies with one-line calls to the helper (Qwen passes `"qwen_image"`, FLUX `"flux"`).
- [ ] **Step 4: Run** — `pytest tests/unit/backends tests/unit/quant -q -x`; expected PASS.
- [ ] **Step 5: Commit** — `refactor(neuron): one quant resolver for the pipeline applications; HBM estimates read a quantized directory as stored`; push.

### Task A5: one driver-side preparer, with the launch's tp

**Files:**
- Modify: `difflet/cli/quant.py` (`quant_spec(args, *, tp_degree=None)`, new `prepare_quant`)
- Modify: `difflet/cli/orchestrators/qwen_image.py:388-416` and `difflet/cli/orchestrators/flux.py:142-166` (`_prepare_quant` → the helper)
- Test: `tests/unit/cli/test_quant_cli.py`

**Interfaces:**
- Produces `quant_spec(args, *, tp_degree: int | None = None) -> QuantSpec | None`: `tp_degree`
  overrides `args.tp_degree or 4` (a driver whose default tp is not 4 passes its own).
- Produces `prepare_quant(args, component_dirs: dict[str, str], *, create: bool, tag: str, tp_degree: int | None = None) -> dict[str, str] | None`
  (torch imported lazily): `None` when `--quant` is off; else sets
  `DIFFLET_NKI_TRACE_CACHE=1` (setdefault) when `spec.compute == "nki"`, and for every
  `name -> source_dir` returns `name -> str(prepare_quantized_checkpoint(source_dir, spec, create=create))`,
  printing `[{tag}] MXFP8 {name} ({spec.compute}): {path}`; a `ValueError` becomes
  `print("Error: --quant: ...", file=sys.stderr); raise SystemExit(1)`. A missing source
  directory (not downloaded) returns `None` and prints nothing.

- [ ] **Step 1: Write the failing tests**

```python
def test_quant_spec_takes_the_drivers_tp(monkeypatch):
    args = _parse("generate", "--model-id", QWEN, "--backend", "neuron", "--quant", "mxfp8")
    assert quant_spec(args).tp_degree == 4 and quant_spec(args, tp_degree=2).tp_degree == 2

def test_prepare_quant_writes_on_create_and_refuses_when_missing(tmp_path, monkeypatch, tiny_dir):
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path))
    args = _parse("compile", "--model-id", QWEN, "--backend", "neuron", "--quant", "mxfp8")
    with pytest.raises(SystemExit): prepare_quant(args, {"transformer": tiny_dir}, create=False, tag="t")
    out = prepare_quant(args, {"transformer": tiny_dir}, create=True, tag="t")
    assert (Path(out["transformer"]) / "difflet_quant.json").is_file()
    assert os.environ["DIFFLET_NKI_TRACE_CACHE"] == "1"
```

- [ ] **Step 2: Run** — `pytest tests/unit/cli/test_quant_cli.py -k "drivers_tp or prepare_quant" -v`; FAIL.
- [ ] **Step 3: Implement**; the Qwen and FLUX orchestrators call `prepare_quant` (Qwen keeps `self._quantized_dit = out["transformer"]`).
- [ ] **Step 4: Run** — `pytest tests/unit/cli/test_quant_cli.py tests/unit/cli/test_orchestrator_flux_neuron.py tests/unit/cli/test_orchestrator_qwen_image*.py -q`; PASS.
- [ ] **Step 5: Commit** — `refactor(cli): one --quant preparer for the drivers; quant_spec takes the launch's tp`; push.

### Task A6: the full CPU gate, then Qwen bit-identity on the device

- [ ] **Step 1:** `pytest tests/unit -q -x -p no:cacheprovider` (the CPU gate the CI runs; `scripts/neuron-skips.txt` as before); expected: the same pass count as `03b43ba` plus the new tests.
- [ ] **Step 2 (device):** `difflet generate --model-id Qwen/Qwen-Image --backend neuron --tp-degree 4 --quant mxfp8 --steps 50 --seed 42 --height 1024 --width 1024 --prompt "<the evidence doc's cat prompt>" --output /home/ubuntu/fp8-bench/refactor_check/qwen_mx.png`; expected: md5 equal to the branch's recorded mxfp8 image for that prompt/seed (`/home/ubuntu/fp8-bench/qwen_mx_tp4_ref2/` per the status memory) and step median within 239 ± 3 ms.
- [ ] **Step 3:** record both in the status page; commit nothing (verification only).

# Phase B — LTX-2

### Task B1: the LTX-2 target table

**Files:**
- Modify: `difflet/quant/targets.py` (new constants, registries)
- Modify: `difflet/cli/quant.py:21-24` (`QUANT_MODELS["Lightricks/LTX-2"] = "ltx_2"`)
- Test: `tests/unit/quant/test_targets_ltx2.py` (new)

**Interfaces:**
- Produces `LTX_2_TARGETS`, `LTX_2_KEPT_BF16`, `LTX_2_ROW_PARALLEL`, and `"ltx_2"` in
  `TARGETS_BY_MODEL` / `ROW_PARALLEL_BY_MODEL`. With `_BLOCK = r"transformer_blocks\.\d+"`:
  - targets (the linears whose input is the video stream or the text stream):
    `rf"{_BLOCK}\.attn1\.(to_q|to_k|to_v)"`, `rf"{_BLOCK}\.attn2\.(to_q|to_k|to_v)"`,
    `rf"{_BLOCK}\.audio_to_video_attn\.to_q"`, `rf"{_BLOCK}\.video_to_audio_attn\.(to_k|to_v)"`,
    `rf"{_BLOCK}\.ff\.net\.0\.proj"`, `rf"{_BLOCK}\.ff\.net\.2"`;
  - kept bf16 (documented, not matched): every `.to_out.0`, the audio-input linears
    (`audio_attn1`, `audio_attn2`, `audio_ff`, `audio_to_video_attn.to_k/to_v`,
    `video_to_audio_attn.to_q`; M = 126 tokens at the default shape — a device variant via
    `DIFFLET_QUANT_TARGETS` decides them), `proj_in`, `audio_proj_in`, `caption_projection.*`,
    `audio_caption_projection.*`, `time_embed.*`, `audio_time_embed.*`, `av_cross_attn_*`,
    `proj_out`, `audio_proj_out`;
  - row-parallel: `rf"{_BLOCK}\.(attn1|attn2|audio_attn1|audio_attn2|audio_to_video_attn|video_to_audio_attn)\.to_out\.0"`,
    `rf"{_BLOCK}\.ff\.net\.2"`, `rf"{_BLOCK}\.audio_ff\.net\.2"`.
- The module docstring gets an LTX-2 paragraph with the dims (video 4096, audio 2048, ff 16384 / 8192).

- [ ] **Step 1: Write the failing test** — builds `diffusers.LTX2VideoTransformer3DModel(**TINY_FP8_LTX2)` (Task B2's toy; put the dict in `tests/unit/backends/_neuron_ltx2_toy.py` now) and asserts:

```python
def test_default_targets_cover_the_video_and_text_input_linears_only():
    spec = default_spec("ltx_2")
    names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and spec.matches(n)]
    per_block = {n.split(".", 2)[2] for n in names}
    assert per_block == {"attn1.to_q", "attn1.to_k", "attn1.to_v", "attn2.to_q", "attn2.to_k", "attn2.to_v",
                         "audio_to_video_attn.to_q", "video_to_audio_attn.to_k", "video_to_audio_attn.to_v",
                         "ff.net.0.proj", "ff.net.2"}
    assert len(names) == 11 * TINY_FP8_LTX2["num_layers"]
    assert all(spec.is_row_parallel(n) == n.endswith(("to_out.0", "ff.net.2")) for n, _ in model.named_modules())
    assert not any(spec.matches(n) for n in ("proj_out", "caption_projection.linear_1", "transformer_blocks.0.audio_ff.net.2"))
```

- [ ] **Step 2: Run** — `pytest tests/unit/quant/test_targets_ltx2.py -v`; FAIL (`ltx_2` unknown).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — PASS; also `pytest tests/unit/cli/test_quant_cli.py -q`.
- [ ] **Step 5: Commit** — `feat(quant): the LTX-2 MXFP8 target table (video- and text-input linears; audio stream and output projections bf16)`; push.

### Task B2: the LTX-2 transformer application

**Files:**
- Modify: `difflet/backends/neuron/ltx_2/transformer.py:352-392` (attributes; the replicate-flag refusal)
- Modify: `tests/unit/backends/_neuron_ltx2_toy.py` (`TINY_FP8_LTX2`, `write_tiny_ltx2_model(root, *, config=None)`)
- Test: `tests/unit/backends/test_neuron_ltx2_fp8.py` (new; the FLUX fp8 test's shape)

**Interfaces:**
- `NeuronLtx2TransformerApplication`: `quant_model = "ltx_2"`, `quant_label = "LTX-2"`; in
  `__init__`, after `super().__init__`, with `self.quant is not None` and either
  `DIFFLET_LTX2_TP_REPLICATE_ATTN` or `DIFFLET_LTX2_TP_REPLICATE_MLP` truthy
  (`modeling_ltx_2._env_flag`): `ValueError("--quant mxfp8 quantizes the row-parallel layers per rank K range; unset DIFFLET_LTX2_TP_REPLICATE_ATTN / _MLP")`.
- `TINY_FP8_LTX2 = {**TINY_LTX2_TRANSFORMER, "attention_head_dim": 32, "audio_attention_head_dim": 32, "cross_attention_dim": 128, "audio_cross_attention_dim": 128}`
  (4 heads x 32 = 128 inner for both streams; every quantized K is 128 or 512).
- `TINY_TARGETS_LTX2 = 11 * num_layers`.

- [ ] **Step 1: Write the failing tests** (CPU, tp=1, `compute="dequant"`, the `tiny` fixture as in `test_neuron_flux_fp8.py:45-66` with `write_tiny_ltx2_model(root, config=TINY_FP8_LTX2)`; the weight-only reference built from `diffusers_reference` with the dequantized weights)

```python
def test_the_quantized_directory_holds_the_default_targets_only(tiny): ...  # manifest["tensors"]["quantized"] == TINY_TARGETS_LTX2
def test_application_quantizes_the_targets_and_matches_the_weight_only_reference(tiny):
    app = NeuronLtx2TransformerApplication(model_path=tiny.quantized, config=tiny_ltx2_config(...), dtype=torch.float32, exec_mode="compile", device="cpu", quant=tiny.spec)
    app.load()
    assert len(app.quantized_linears) == TINY_TARGETS_LTX2 and app.load_report == {"missing": [], "unexpected": []}
    block = app.module.transformer.transformer_blocks[0]
    assert isinstance(block.attn1.to_q, FP8Linear) and isinstance(block.audio_attn1.to_q, torch.nn.Linear)
    assert isinstance(block.attn1.to_out[0], torch.nn.Linear)
    video, audio = app(*tiny_ltx2_inputs())
    assert torch.allclose(video, tiny.reference_video, atol=2e-2, rtol=2e-2)
def test_replicate_flags_are_refused_with_quant(tiny, monkeypatch):
    monkeypatch.setenv("DIFFLET_LTX2_TP_REPLICATE_MLP", "1")
    with pytest.raises(ValueError, match="REPLICATE"): ...
def test_eager_is_refused(tiny): ...  # ValueError match "D43"
```

- [ ] **Step 2: Run** — `pytest tests/unit/backends/test_neuron_ltx2_fp8.py -v`; FAIL (`quant` refused: no target table).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — PASS; then `pytest tests/unit/backends/test_neuron_ltx2_l1.py -q` (unchanged).
- [ ] **Step 5: Commit** — `feat(ltx-2): --quant mxfp8 on the DiT application (base-class wiring; replicate flags refused)`; push.

### Task B3: the LTX-2 pipeline application

**Files:**
- Modify: `difflet/models/ltx_2/neuron_application.py:77-98` (`_KWARG_DEFAULTS` + `"quant": None, "quant_checkpoint": None`), `:404-423` (resolve, pass `quant=`), `:473-486` (`_check_device_text_encoder_fits` with the quantized dir)
- Test: `tests/unit/backends/test_neuron_ltx2_fp8.py`

**Interfaces:**
- In `__init__`: `self.quant, dit_path = resolve_quant("ltx_2", options["quant"], options["quant_checkpoint"], transformer_path, tp_degree=int(parallel.tp_degree))`;
  the transformer gets `model_path=dit_path, quant=self.quant`; `self.quantized_dit: str | None`
  (the directory, or None). `_check_device_text_encoder_fits` passes `dit_dir=self.quantized_dit`
  to `device_text_encoder_components`.

- [ ] **Step 1: Write the failing test** (as `test_neuron_flux_fp8.py::test_pipeline_application_resolves_the_quantized_directory`): construct `TorchNeuronLtx2Application` on the CPU with `enable_text_encoder=False, enable_decode_components=False, quant="mxfp8"`, `DIFFLET_COMPILE_CACHE` at the toy's cache root; assert `app.transformer.model_path == str(tiny.quantized)` and `app.quant == default_spec("ltx_2", tp_degree=1)`; and `quant_checkpoint` without `quant` raises.
- [ ] **Step 2: Run** — FAIL (`TypeError`: unknown kwarg).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — `pytest tests/unit/backends/test_neuron_ltx2_fp8.py tests/unit/backends/test_neuron_ltx2_l2.py -q`; PASS.
- [ ] **Step 5: Commit** — `feat(ltx-2): the pipeline application resolves the quantized DiT directory and counts its HBM bytes as stored`; push.

### Task B4: the LTX-2 orchestrator

**Files:**
- Modify: `difflet/cli/orchestrators/ltx_2.py:122-137` (`_torchrun_application_kwargs` + `quant`), `:419-426` (`_check_dit_fits` with `dit_dir`), `:455-459` (`_prepare_torchrun` calls `prepare_quant` before `_check_dit_fits`; `create = not generating`), `:631-658` (`_stage_cache_inputs` + `**quant_cache_fields(quant_spec(args))`), `:665-707` (`_shared_cli_args` + `quant_cli_args(a)`)
- Test: `tests/unit/cli/test_quant_cli.py` (an LTX-2 twin of `test_qwen_orchestrator_forwards_and_keys_quant` and of `test_application_kwargs_carry_the_spec`), `tests/unit/cli/test_ltx2_neuron_cli.py` (DP worker forwarding: `worker_cli_args` already carries `--quant`; assert it for the LTX-2 args namespace)

- [ ] **Step 1: Write the failing tests** (copy the Qwen ones with `LTX2 = "Lightricks/LTX-2"`; the key test asserts the five added fields and `quant_tp_degree == 2` when `--tp-degree 2`).
- [ ] **Step 2: Run** — `pytest tests/unit/cli/test_quant_cli.py -k ltx2 -v`; FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — `pytest tests/unit/cli -q -x`; PASS.
- [ ] **Step 5: Commit** — `feat(ltx-2): --quant mxfp8 through the driver — quantized directory on compile, stage flags, cache key, HBM`; push.

### Task B5: the comparison tool reads videos

**Files:**
- Modify: `scripts/compare_difflet_runs.py`
- Test: `tests/unit/scripts/test_compare_difflet_runs.py` (new)

**Interfaces:**
- `compare(a: Path, b: Path) -> dict`: an `.mp4` (or a directory of frame PNGs) pair is read
  frame by frame (`imageio`/`torchvision.io.read_video`, whichever the venv has — the script
  says which in its header); the metrics are per-frame PSNR/SSIM (and LPIPS when available)
  averaged over frames, plus `frames`, `min_psnr_frame`; images keep today's output. A frame
  count mismatch is an error naming both counts.

- [ ] **Step 1: Write the failing test** — two synthetic 4-frame clips written with the same reader library; `compare(a, a)["psnr"] == inf`-safe (capped as today), `compare(a, b)["frames"] == 4`, mismatch raises.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** — `scripts: compare_difflet_runs reads videos (per-frame PSNR/SSIM/LPIPS averaged)`; push.

### Task B6: LTX-2 on the device (trn3, tp4, 512x768x121, 50 steps, seed 42, LNC=2)

No code unless a step fails (then one bug, one commit, as `difflet-device-verify` prescribes).
Work dir `/home/ubuntu/fp8-bench/ltx2_tp4/`; each step's command, step-time median and md5s go
into `docs/verification/2026-11-07-mxfp8-video-models.md` (new; one section per model).

- [ ] **Step 1: kernel shape probe (one core):** a `nki` script calling `fp8_kernels.mx_linear` at the per-rank shapes `(6144 x 4096 -> 1024)`, `(6144 x 4096 -> 4096)`, `(6144 x 16384/4 -> 4096)` (row), `(1024 x 4096 -> 1024)`, `(126 x 2048 -> 512)` (the audio variant), against `mx.dequantize_mx` on the CPU; expected rel error < 2e-3 at every shape, no kernel assert. Record the per-shape time only as a sanity figure.
- [ ] **Step 2: download** `Lightricks/LTX-2` (38 GB transformer + the rest); `df` before and after.
- [ ] **Step 3: bf16 bring-up:** `difflet compile ... --backend neuron --tp-degree 4` then two `difflet generate` with the fixed config; expected: identical md5s, a step median (the first trn3 number for LTX-2), the bf16-vs-bf16 noise floor from a second seed pair is NOT needed here (determinism is bit-exact); profile one block if the budget allows and note the quantizable share.
- [ ] **Step 4: mxfp8:** `difflet compile ... --quant mxfp8` (writes the quantized directory; record its size and the HBM estimate line), two generates; expected: identical md5s; `compare_difflet_runs.py` vs the bf16 clip; the step ratio.
- [ ] **Step 5: audio variant:** `DIFFLET_QUANT_TARGETS="<default>;<audio patterns>"` one compile + generate; record the ratio and PSNR; keep whichever set is faster at equal quality as the default (a code change to `targets.py` + commit if it is the audio set).
- [ ] **Step 6: CP2 (gather_kv) and DP2** at the same config: `--cp-degree 2` one generate (md5 recorded, PSNR vs the tp4 clip); a DP2 launch to prove the worker forwards `--quant` (its log line `[ltx_2] MXFP8 transformer (nki)` on both workers).
- [ ] **Step 7: evidence and docs:** the evidence doc section, README's feature note gains LTX-2, `docs/cli-staged-commands.md` the LTX-2 quant command; `git commit -m "docs: LTX-2 in MXFP8 on Trainium3 — evidence, README, staged commands"`; push; status page updated.

## Roadmap after phase B (each gets its own plan file when reached)

- **C. Wan 2.1 → 2.2-A14B:** quantize from the bf16 copy (`weights_bf16.ensure_bf16_copy`) with
  the copy's manifest sha256 added to `source_fingerprint`; targets in checkpoint-key spelling
  (`ffn.net.0.proj` / `net.2`) through `quantize_module_(key_fn=checkpoint_key)`; one quantized
  directory per expert, same spec for both; `resolve_component_dir` replaced by the quantized
  directory at both sites; tp8 refused for `net_out` (K=1728); both experts' HBM summed; device at
  480x832x9 and 480x832x81.
- **D. Single-block `proj_out` for FLUX and HV1:** the writer stores `proj_out_attn.weight` /
  `proj_out_mlp.weight` (column ranges of `proj_out.weight`, each quantized per rank K range) and
  the loader skips the slice for a quantized directory; measured on FLUX first (0.99x today).
- **E. HunyuanVideo:** FLUX's pattern; `quant` an explicit kwarg (the `**overrides` collision);
  `_KWARG_DEFAULTS`; default 320x512x61 and 720p x 129 frames.
- **F. HunyuanVideo-1.5:** memory-only; `quant_spec(args, tp_degree=layout.tp)`; quantize from the
  bf16 copy; no stage cache-key dict (NEFF cache keys on the HLO — asserted in a test).
- **G. CPU fp32 arbiter:** `scripts/<model>_fp8_cpu_reference.py` per model on diffusers fp32 at
  the short shape, 50 steps, seed 42; bf16-vs-fp32 and mxfp8-vs-fp32 per model; the relative
  quality gate is fixed from Qwen's pair first.
