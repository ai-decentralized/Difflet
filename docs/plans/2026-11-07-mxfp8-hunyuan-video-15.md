# MXFP8 on HunyuanVideo-1.5 (480p t2v) — Implementation Plan (phase F)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `--quant mxfp8` for `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v` on the
native neuron backend, through the shared layer of phase A, with device evidence at the default
480x848x121 shape and layout (tp2 x CFG2) that measures memory (per-rank DiT bytes resident,
peak HBM, headroom) and step time honestly, plus CP2. Memory-only by decision: no speed work
beyond the bisection rule of F6.

**Architecture:** the LTX-2 / Wan mechanism: a target table, the base-class attributes on the
transformer application, `resolve_quant` in the pipeline, `prepare_quant` in the driver.
HunyuanVideo-1.5's own points: the module is diffusers' class (module names are the checkpoint
keys, `checkpoint_key` is the identity); the hub DiT is fp32, so the quantized checkpoint is
written from the bf16 copy `compile` already makes; the driver's default layout is tp2 x CFG2
but `_layout` gives tp1 whenever `--cfg-parallel` or `--cp-degree` is passed without
`--tp-degree`, so the spec's tp comes from `_layout`, never a constant; the native path has no
stage cache key (NEFFs are keyed by the HLO, the weight store by content and layout), which
F2's graph-and-store test (`test_a_quantized_and_a_bf16_dit_share_no_block_graph_and_no_store_entry`)
and F4's stage-spec test (`test_transformer_stage_builds_the_pipeline_with_the_spec`) prove
instead of adding one. The rotary at HV1.5's diffusers call site is phase D's
(`docs/plans/2026-11-07-mxfp8-proj-out-rope.md`: D3's `apply_rotary_emb_fp32`, D4's
`_joint_qkv`); this plan only depends on it and measures it on the device (F6).

**Tech Stack:** as phase B. HunyuanVideo-1.5 facts (diffusers 0.38 `transformer_hunyuan_video15.py`,
`difflet/backends/neuron/hunyuan_video15/transformer.py`, counted on meta 2026-11-07):
54 dual-stream `transformer_blocks` (no single-stream blocks, so no FLUX/HV1 `proj_out` split),
2048 wide (16 heads x 128), MLP 8192; per block `norm1.linear` / `norm1_context.linear`
(2048 -> 12288, modulation), `attn.to_q/k/v` and `attn.add_q/k/v_proj` (2048 -> 2048, column,
`gather_output=False`), `attn.to_out.0` / `attn.to_add_out` (row), `ff` / `ff_context`
`net.0.proj` (column) and `net.2` (row, K 8192/tp: a multiple of 128 at every tp the 16 heads
allow, so no tp is refused; F1 pins it on a meta-built model: 540 targets, every per-rank K a
multiple of 128 at tp 1, 2, 4, 8 and 16, probed 2026-11-07). 8.327B parameters (16.65 GB bf16): targets 4.983B (60%),
modulation 2.718B (33%), the token refiner and embedders in the eager top level. Rows per rank
at 480x848x121 and tp2 x CFG2: 49,290 image (31 x 30 x 53) and 1,985 text (729 + 256 + 1000),
both padded by B7 (to 49,296 and 1,992); under CP2 24,645 image rows (to 24,648). The image
q/k pass `norm_q`/`norm_k`, the interleaved rotary (diffusers' form until D4) and the
key-bounded attention kernel: the shape of Wan's self-attention q/k/v, which cost Wan 5x.
The largest activation M in the evidence doc is 6,144 (section 1); Wan 2.1 at 81 frames ran
21 x 30 x 52 = 32,760 rows per rank (its section is not written yet). HV1.5's 49,290 is beyond
both, so F6 probes its per-rank shapes on one core before any compile (Step 0b).
Memory arithmetic (probed on the CPU toy, exact): an `nki` target holds N·K fp8 bytes, the
stored scales N·K/32 and the kernel's scale layout (`mx.kernel_layout`: uint8 `[K/4, N]`)
N·K/4, so 1.28 bytes per weight against bf16's 2 (resident = bf16 − Σ N·K·23/32); the pre-load
estimate (`weight_bytes`) omits the N·K/4. HV1.5: quantized directory 11.83 GB, resident
13.07 GB (78% of bf16), per rank at tp2 about 7.75 -> 6.09 GiB. `--quant-compute dequant`
keeps a bf16 copy of each target beside its stored scales: resident = bf16 + Σ N·K/32.

**Spec:** the roadmap entry F of `docs/plans/2026-11-07-mxfp8-video-models.md`, phase C's plan
`docs/plans/2026-11-07-mxfp8-wan.md` (the pattern), phase D's plan (the rotary, D3/D4), the
device facts in `docs/verification/2026-11-07-mxfp8-video-models.md`, and this phase's
decisions: memory-only; quantize from the bf16 copy; no q/k/v fusion; the device default set
chosen by bisection when the first run is slower than bf16.

## Decisions that settle the open questions (2026-11-07, the same for phases D, E and F)

- **Quality standard.** The PSNR/SSIM standard is diffusers on the CPU in fp32, run once for
  every model at the end (phase G; the user's decision). Device comparisons against device
  bf16 are a screen, with one floor-relative gate for every phase: PSNR >= min(30, floor - 4),
  SSIM >= min(0.97, floor - 0.10), LPIPS <= max(0.05, 2 x floor), where the floor is two bf16
  runs at different TP degrees at the same shape. A set that fails the screen is not made the
  default; a shipped set that fails it at another shape is recorded as pending phase G (Wan 2.1
  at 81 frames: 28.8 dB against a 37.2 dB floor).
- **Default-set rule.** Phase E's leave-one-out rule (E5 step 5) for every phase: the default is
  always a measured set; "faster" means below bf16 by more than J = max(the run-to-run gap,
  1% of the bf16 step); among sets within J the one quantizing more layers wins; no union of
  separately measured groups. Screening arms run one generate; the chosen set runs two.
- **Disk.** Arm directories stay until the decision; the losers are deleted afterwards by
  literal path. FLUX's quantized-directory digest change under D2 is accepted.
- **HunyuanVideo-1.5's rotary change (D4)** is measured in phase F, not in D.
- **Cross-plan references.** E6 imports the split-column law from `difflet/quant/fake_linear.py`
  (`split_linear_columns`); F6 pins the pre-D4 tree by the recorded SHA of D4's parent, not by a
  commit-message grep.

## Global Constraints

- Everything in the phase-A/B plan's Global Constraints (branch, push per commit, status page
  per task, D43, no fusion, K % 128, byte-identical default command lines, 50 steps / seed 42 /
  LNC=2; the CPU fp32 reference is phase G, not here).
- F6 needs phase D's D3 and D4 merged (the bf16 baseline is measured with the new rotary);
  F1–F5 do not depend on D's code.
- Order: phase D, then phase E's E1, then F1. D1/D2 and E1 edit the same parts of
  `difflet/quant/targets.py` and `difflet/cli/quant.py` that F1 does, and their plans were
  written against the same HEAD (917d087). So F1's edits there are anchored by symbol, not by
  line: the module docstring's paragraph before "Patterns are regexes", the constants after the
  last model table (`WAN_ROW_PARALLEL` today, E1's HunyuanVideo tables once E1 has landed), the
  `TARGETS_BY_MODEL` / `ROW_PARALLEL_BY_MODEL` dicts, `__all__`, `QUANT_MODELS`, and the
  `--quant` help string. Line numbers elsewhere in this plan are against 917d087. Where D or E
  has moved a line, use the named symbol.
- `difflet.cli.orchestrators.hunyuan_video_15` stays free of torch and of every
  `difflet.backends.neuron*` module at import (checked 2026-11-07: neither is in `sys.modules`
  after importing it). The AoT path's native-free contract depends on this. Its guard,
  `test_aot_stages_do_not_import_native`, is `importorskip`'d on `neuronx_distributed`, which
  the native-fp8 venv lacks, so it never runs here. Every import F4 and F5 add of `torch`,
  `torch_neuronx`, `difflet.backends.neuron.core.weights_bf16.resolve_component_dir`,
  `difflet.backends.neuron.core.application_base.rank_weight_bytes` and
  `difflet.backends.neuron.hbm` (`GIB`, `per_core_hbm_bytes`) goes inside the function that
  uses it, as in Wan's `_prepare_quant` (`difflet/cli/orchestrators/wan.py:421-423`). The
  module-top imports may add only `difflet.cli.quant` and `difflet.cli.orchestrators.torchrun`
  names (both torch-free). F4 adds a test for this that is never skipped.
- `DIFFLET_QUANT_TARGETS` replaces the default set in `quant_spec` (`difflet/cli/quant.py:111`).
  Every test that compares `quant_spec(...)` with `default_spec(...)`, or counts targets through
  the driver, runs with it cleared. That is an autouse fixture in `tests/unit/quant/test_targets_hv15.py`
  and in `tests/unit/cli/test_orchestrator_hv15_neuron.py`. The pipeline's
  `resolve_quant` builds `default_spec` without reading it, so F3's tests are unaffected. On
  the device it is a per-command prefix, never exported.
- Targets name only the compiled `transformer_blocks`; the eager top level (`context_embedder.*`
  incl. the token refiner, `context_embedder_2.*`, `image_embedder.*`, `time_embed.*`,
  `norm_out.linear`, `proj_out`) and the per-block modulation `norm1(_context).linear` stay bf16.
- The quantized checkpoint's source is `resolve_component_dir(<root>/transformer, torch.bfloat16)`:
  the bf16 copy `compile` makes, else the fp32 directory (then a `generate` finds no quantized
  directory and is refused naming `difflet compile`; nothing is written outside `compile`).
- The spec's tp is `_layout(args)[0]` (`prepare_quant(..., default_tp_degree=tp)`); the stage
  line already carries `--tp-degree <tp>`, so both sides build the same spec.
- No new stage cache key on the native path, and the AoT path is untouched (`validate_quant`
  refuses `--quant` off `--backend neuron`, so the flags never reach `_run_aot`).
- No driver-side HBM check: the bf16 DiT (16.65 GB) fits one trn3 core at tp1, so a header
  estimate refuses no layout; each rank's `_check_hbm_budget` already counts fp8 as stored.
- Unit tests never reach `ensure_bf16_copy` on the real snapshot (the `tests/unit/conftest.py`
  guard): F6 downloads it into the hub cache, so from F4 on every HV1.5 driver test resolves a
  toy or no snapshot.
- Unit tests run as `cd /home/ubuntu/Difflet/.claude/worktrees/mxfp8-video-models && DIFFLET_BACKEND=cpu PYTHONPATH=. /home/ubuntu/venvs/native-fp8/bin/python -m pytest <paths> -q -p no:cacheprovider`
  (written `pytest <paths>` below).

## Review Focus

1. `--cfg-parallel` or `--cp-degree 2` without `--tp-degree` runs at tp1 (`_layout`): the
   driver must quantize for tp1 and the stage rebuild the same spec; a constant
   `default_tp_degree=2` would write a tp2 directory every rank then refuses — F4's
   `test_quant_reaches_the_transformer_stage_at_the_layout_tp`.
2. A `--quant` generate on a host where compile never ran (no bf16 copy, no quantized
   directory) must be refused on the driver, exit 2, naming `difflet compile`, before the encode
   stage starts and without writing anything — F4's
   `test_quant_generate_before_compile_is_refused_before_any_launch`.
3. The memory the decision rests on must be the resident bytes after `_after_load` (the
   kernel's `[K/4, N]` scale layout included), not the pre-load estimate. F5's
   `test_dit_memory_line_names_resident_and_estimated_bytes` pins it (the line reads the module,
   not `weight_bytes`); `test_resident_bytes_count_the_kernel_layout` pins the arithmetic.
4. `--quant` with `--sp` (the image MLP's row twin must keep `reduce_output=False,
   skip_bias_add=True`) and with CFG2 (each group runs its branch) must match the dequantized
   reference. Each rank's row-parallel weights must be its own K range quantized as such, not a
   slice of a whole-K quantization. Output tolerances cannot see that on a toy (measured
   2026-11-07, see F1), so the check compares the weights exactly — F2's
   `test_quantized_layouts_match_the_dequantized_reference`.
5. With no stage key on the native path, a stage line or a rank that drops the spec would run
   the bf16 DiT under the driver's MXFP8 line. The transformer stage must build its pipeline
   with the spec the driver prepared — F4's `test_transformer_stage_builds_the_pipeline_with_the_spec`
   (with F2's graph-and-store test for what the HLO keys).

---

### Task F1: the HunyuanVideo-1.5 target table and its FP8 toy

Lands after phase E's E1 (Global Constraints: order). Every edit below in `targets.py` and
`cli/quant.py` is anchored by symbol.

**Files:**
- Modify: `difflet/quant/targets.py`:
  - the module docstring: a HunyuanVideo-1.5 paragraph as the last model paragraph, right
    before "Patterns are regexes" (after E1's HunyuanVideo paragraph);
  - the three constants after the last model table (after E1's HunyuanVideo tables and helper;
    `WAN_ROW_PARALLEL` if E1 has not landed);
  - `TARGETS_BY_MODEL` / `ROW_PARALLEL_BY_MODEL`: a `"hunyuan_video_15"` entry each;
  - `__all__`: the three names, in sorted order.
- Modify: `difflet/cli/quant.py`:
  - `QUANT_MODELS`: the HV1.5 id;
  - the `--quant` help (`add_quant_flags`): the model list's tail
    `"... Wan 2.2-A14B (both experts) and HunyuanVideo."` (E1's) becomes
    `"... Wan 2.2-A14B (both experts), HunyuanVideo and HunyuanVideo-1.5 (a memory saving)."`
    (if E1 has not landed: `"... Wan 2.2-A14B (both experts) and HunyuanVideo-1.5 (a memory saving)."`).
    The bf16 clause before it stays as D2 (and E's late task) left it. HV1.5 has no
    single-stream blocks, so that clause needs no HV1.5 mention.
- Modify: `difflet/cli/orchestrators/hunyuan_video_15.py` `_unsupported` (`:63-64` at 917d087): the temporary `--quant` refusal
- Create: `tests/unit/backends/_neuron_hv15_toy.py`
- Test: `tests/unit/quant/test_targets_hv15.py` (new), `tests/unit/cli/test_orchestrator_hv15_neuron.py`
  (`_args` at `:16-24` gains `quant=None, quant_compute=None`; an autouse fixture
  `_no_quant_targets_override(monkeypatch)` that runs `monkeypatch.delenv("DIFFLET_QUANT_TARGETS", raising=False)`; one refusal test)

**Interfaces:**
- Produces, with `_BLOCK = r"transformer_blocks\.\d+"`:
  `HUNYUAN_VIDEO_15_TARGETS = (rf"{_BLOCK}\.attn\.(to_q|to_k|to_v|add_q_proj|add_k_proj|add_v_proj)", rf"{_BLOCK}\.(ff|ff_context)\.net\.0\.proj", rf"{_BLOCK}\.(ff|ff_context)\.net\.2")`;
  `HUNYUAN_VIDEO_15_KEPT_BF16 = (rf"{_BLOCK}\.attn\.(to_out\.0|to_add_out)", rf"{_BLOCK}\.norm1(_context)?\.linear", r"context_embedder\..*", r"context_embedder_2\..*", r"image_embedder\..*", r"time_embed\..*", r"norm_out\.linear", r"proj_out")`;
  `HUNYUAN_VIDEO_15_ROW_PARALLEL = (rf"{_BLOCK}\.attn\.(to_out\.0|to_add_out)", rf"{_BLOCK}\.(ff|ff_context)\.net\.2")`;
  `TARGETS_BY_MODEL["hunyuan_video_15"]`, `ROW_PARALLEL_BY_MODEL["hunyuan_video_15"]`.
- `QUANT_MODELS["hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"] = "hunyuan_video_15"`.
- `_neuron_hv15_toy.py`: `TINY_FP8_HV15 = dict(in_channels=9, out_channels=4, num_attention_heads=4, attention_head_dim=32, num_layers=2, num_refiner_layers=1, mlp_ratio=4.0, patch_size=1, patch_size_t=1, qk_norm="rms_norm", text_embed_dim=32, text_embed_2_dim=24, image_embed_dim=16, rope_axes_dim=(8, 12, 12), rope_theta=256.0, target_size=640, task_type="t2v", use_meanflow=False)`
  (inner 128, MLP 512: every target K is 128 or 512, `net.2` per rank 512/tp);
  `TINY_SHAPE_HV15 = dict(height=32, width=48, num_frames=5, text_seq_len=16, text_2_seq_len=8, image_tokens=9)`;
  `TINY_TARGETS_PER_BLOCK = {"attn.to_q", "attn.to_k", "attn.to_v", "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2"}`;
  `tiny_hv15_inputs(*, seed: int = 1, valid_text: int = 6) -> dict[str, torch.Tensor]` (`test_neuron_hv15_dit.py:_inputs`, parameterised);
  `ROW_K_SPREAD = 2.0 ** -14`;
  `HV15_ROW_TARGETS = ("transformer_blocks.0.ff.net.2", "transformer_blocks.1.ff_context.net.2")` (the two the tests read back);
  `write_tiny_hv15_snapshot(root, *, config: dict = TINY_FP8_HV15) -> Path`. It writes, as
  `test_neuron_hv15_pipeline.py:36-53` does but without the VAE: an fp32 `transformer/`
  (`torch.manual_seed(0)`, the model, then the K spread below, then `save_pretrained`),
  `scheduler/`, `guider/guider_config.json` and `model_index.json`.
  The K spread: under `torch.no_grad()`, every linear named
  `transformer_blocks.<i>.ff.net.2` or `transformer_blocks.<i>.ff_context.net.2` (the
  row-parallel targets) gets `weight[:, : in_features // 2] *= ROW_K_SPREAD`. Why it is needed:
  on a kaiming-uniform weight every 32-value block lands on the same E8M0 exponent, and MX
  rounding of a normal E4M3 value does not depend on its power-of-two scale. So whole-K and
  per-rank quantization agree bit for bit (`torch.equal`, probed 2026-11-07), and a writer or
  loader that quantized a row-parallel target whole would go unseen. With the spread they
  differ in 24,533 of the 65,536 values of `transformer_blocks.0.ff.net.2` (24,777 for block 1's
  `ff_context.net.2`, the same count at tp2 and tp4). They differ only in the values that are
  subnormal in one layout, 2.4e-7 at most, so the two references differ by 6e-7 at the output.
  That is below any useful forward tolerance, which is why F2 compares the weights.
  `hv15_reference(transformer_dir, inputs: dict, *, spec: QuantSpec | None = None) -> torch.Tensor`
  (diffusers' fp32 forward with the empty glyph stream and zero image embeddings; with `spec`,
  every `nn.Linear` whose name `spec.matches` gets its weight replaced by
  `mx.fake_quantize_mx_sharded(w, spec.k_shards(name))`: what dequant mode computes, a
  row-parallel target quantized per rank K range, as D2's `_fake_quant_reference` and E2's
  fixture do).
- Until F4 lands, `_unsupported(args)` also returns `"--quant"` when `args.quant` is set (the
  B1/C1 precedent: no silent-bf16 window on the pushed branch).

- [x] **Step 1: Write the failing tests** (`model` = `HunyuanVideo15Transformer3DModel(**TINY_FP8_HV15)`)

```python
def test_default_targets_cover_the_dual_stream_projections_and_mlps_only(model):
    spec = default_spec("hunyuan_video_15")
    names = [n for n in _linears(model) if spec.matches(n)]
    assert {n.split(".", 2)[2] for n in names} == TINY_TARGETS_PER_BLOCK
    assert len(names) == len(TINY_TARGETS_PER_BLOCK) * TINY_FP8_HV15["num_layers"]  # 20
    assert all(spec.is_row_parallel(n) == (n.startswith("transformer_blocks.") and n.endswith(
        ("attn.to_out.0", "attn.to_add_out", "ff.net.2", "ff_context.net.2"))) for n in _linears(model))
    assert not any(spec.matches(n) for n in (
        "context_embedder.token_refiner.refiner_blocks.0.attn.to_q",
        "context_embedder.token_refiner.refiner_blocks.0.ff.net.0.proj",
        "transformer_blocks.0.norm1.linear", "transformer_blocks.0.attn.to_add_out", "proj_out"))

def test_every_linear_is_a_target_or_documented_bf16(model):
    target, kept = default_spec("hunyuan_video_15"), default_spec("hunyuan_video_15", targets=HUNYUAN_VIDEO_15_KEPT_BF16)
    assert all(target.matches(n) != kept.matches(n) for n in _linears(model))

def test_the_fp8_toy_has_the_mx_geometry(model, tmp_path):
    spec = default_spec("hunyuan_video_15")
    for n in _linears(model):
        if spec.matches(n):
            k = model.get_submodule(n).in_features
            assert all((k // tp if spec.is_row_parallel(n) else k) % LAYOUT_K == 0 for tp in (1, 2, 4)), n
    root = write_tiny_hv15_snapshot(tmp_path)
    assert json.loads((root / "transformer" / "config.json").read_text())["attention_head_dim"] == 32
    assert weights_bf16._has_fp32(root / "transformer")  # the copy path is exercised

def test_the_toy_row_parallel_targets_quantize_differently_per_rank(tmp_path):
    # The spread makes a whole-K row-parallel quantization visible (Review Focus 4, F2).
    root = write_tiny_hv15_snapshot(tmp_path)
    weights = safetensors.torch.load_file(next((root / "transformer").glob("*.safetensors")))
    for name in HV15_ROW_TARGETS:
        w = weights[f"{name}.weight"]
        assert (mx.fake_quantize_mx(w) != mx.fake_quantize_mx_sharded(w, 2)).float().mean() > 0.3, name  # 0.374 / 0.378

def test_every_default_target_has_an_mx_layout_at_every_tp():
    # diffusers' defaults with task_type="t2v" are the 480p_t2v geometry (54 blocks of 16 x 128,
    # MLP ratio 4; F6 Step 1 checks the downloaded config.json against them).
    from accelerate import init_empty_weights
    from diffusers import HunyuanVideo15Transformer3DModel

    with init_empty_weights():
        real = HunyuanVideo15Transformer3DModel(task_type="t2v")
    for tp in (1, 2, 4, 8, 16):
        spec = default_spec("hunyuan_video_15", tp_degree=tp)
        targets = [(n, real.get_submodule(n)) for n in _linears(real) if spec.matches(n)]
        assert len(targets) == 54 * len(TINY_TARGETS_PER_BLOCK), tp  # 540, probed 2026-11-07
        assert all((m.in_features // spec.k_shards(n)) % LAYOUT_K == 0 for n, m in targets), tp

def test_the_registries_and_the_cli_id_name_the_table():
    assert TARGETS_BY_MODEL["hunyuan_video_15"] is HUNYUAN_VIDEO_15_TARGETS
    assert ROW_PARALLEL_BY_MODEL["hunyuan_video_15"] is HUNYUAN_VIDEO_15_ROW_PARALLEL
    assert QUANT_MODELS[HV15] == "hunyuan_video_15"
    args = argparse.Namespace(quant="mxfp8", quant_compute=None, model_id=HV15, tp_degree=None)
    assert quant_spec(args, default_tp_degree=2) == default_spec("hunyuan_video_15", compute="nki", tp_degree=2)
```

  (`_linears` and the imports as in `tests/unit/quant/test_targets_wan.py`, plus `safetensors.torch`
  and `from difflet.quant import mx`; `HV15` the CLI id; the module's autouse fixture
  `_no_quant_targets_override(monkeypatch)` runs `monkeypatch.delenv("DIFFLET_QUANT_TARGETS", raising=False)`,
  as E1's registry test does.)
  And in `test_orchestrator_hv15_neuron.py`: `test_quant_is_refused_until_the_driver_carries_it(launches, tmp_path, capsys)`:
  `HunyuanVideo15Orchestrator(_args(work_dir=..., quant="mxfp8")).generate()` raises `SystemExit`
  with code 2, stderr names `--quant` and ends `nothing was launched.\n`, `launches == []`.
- [x] **Step 2: Run** — `pytest tests/unit/quant/test_targets_hv15.py -v` and `pytest tests/unit/cli/test_orchestrator_hv15_neuron.py -k until -v`; FAIL (ImportError: `HUNYUAN_VIDEO_15_TARGETS`; the generate launches).
- [x] **Step 3: Implement** (table + docstring paragraph with the facts above, the CLI id and help, the toy module, the temporary refusal).
- [x] **Step 4: Run** — PASS; then `pytest tests/unit/quant tests/unit/cli/test_quant_cli.py tests/unit/cli/test_orchestrator_hv15_neuron.py -q`.
- [x] **Step 5: Commit** — `feat(quant): the HunyuanVideo-1.5 MXFP8 target table (dual-stream q/k/v, text q/k/v, both MLPs; output projections, modulation and the eager top level bf16)`; push.

**Done** (`ee13e44`).

### Task F2: the HunyuanVideo-1.5 transformer application

**Files:**
- Modify: `difflet/backends/neuron/hunyuan_video15/transformer.py`, the class body of
  `NeuronHunyuanVideo15TransformerApplication` (`:53-56` at 917d087; D4 edits `_joint_qkv`
  below it): `quant_model = "hunyuan_video_15"`, `quant_label = "HunyuanVideo-1.5"`; the class
  docstring, and a "Quantization." paragraph in the module docstring (targets, identity
  `checkpoint_key`, no fusion, the top level bf16)
- Modify: `tests/unit/backends/_neuron_workers.py` (after `hv15_sp_worker`, `:1564-1574`: `hv15_fp8_worker`)
- Test: `tests/unit/backends/test_neuron_hv15_fp8.py` (new)

**Interfaces:**
- Consumes F1's toy (`HV15_ROW_TARGETS`, the spread row-parallel weights). `quant=` already reaches the base through `**kwargs` (`:58-63`).
- `hv15_fp8_worker(rank, world_size, model_dir: str, shape: dict, inputs: dict, spec: dict, tp: int, cfg: bool, sp: bool) -> dict`:
  the application at `DiffletParallelConfig(tp_degree=tp, cfg_parallel_enabled=cfg, sp_enabled=sp)`,
  fp32, `exec_mode="compile"`, CPU, `quant=spec`. It asserts a clean `load_report`, that
  `transformer_blocks[0].ff.net[2]` is an `FP8RowParallelLinear` and, with `sp`, that
  `reduce_output is False and skip_bias_add is True`. It returns
  `{"out": <output>.tolist(), "tp_rank": get_tp_rank(), "row_weights": {n: app.module.get_submodule(n).weight_bf16.float().tolist() for n in HV15_ROW_TARGETS}}`.
  `get_tp_rank` comes from `difflet.backends.neuron.ops_impl.collectives`. `weight_bf16` is
  the rank's dequantized `[N, K/tp]` slice, which `_after_load`'s `prepare_fp8_kernels_` fills
  in dequant mode; E4M3 values are exact in bf16.

- [x] **Step 1: Write the failing tests** — module fixture `tiny`: `write_tiny_hv15_snapshot(root)`,
  `spec = default_spec("hunyuan_video_15", compute="dequant", tp_degree=1)`, the quantized
  directory (`prepare_quantized_checkpoint(..., create=True, cache_root=root / "cache")`) and
  `hv15_reference(transformer_dir, tiny_hv15_inputs(), spec=spec)`. `_app(tiny, **over)` builds
  the application on the quantized directory, fp32, compile mode, CPU, `**TINY_SHAPE_HV15`.
  `N_TARGETS = len(TINY_TARGETS_PER_BLOCK) * TINY_FP8_HV15["num_layers"]` (20; F6 Step 7
  changes only `TINY_TARGETS_PER_BLOCK` if the default set changes).
  - `test_the_quantized_directory_holds_every_target`: `manifest["tensors"]["quantized"] == N_TARGETS`,
    `{n.split(".", 2)[2] for n in manifest["quantized_linears"]} == TINY_TARGETS_PER_BLOCK`.
  - `test_application_quantizes_the_dual_stream_projections`: after `load()`,
    `len(app.quantized_linears) == N_TARGETS`, a clean `load_report`, and
    `{n.split(".", 2)[2] for n, m in app.module.named_modules() if isinstance(m, _FP8LinearBase)} == TINY_TARGETS_PER_BLOCK`
    with every one of them an `FP8Linear` (tp1: diffusers' plain linears). `block.attn.to_out[0]`,
    `block.attn.to_add_out`, `block.norm1.linear`, `block.norm1_context.linear` and `app.module.proj_out`
    are plain `nn.Linear`. There is no `_FP8LinearBase` in `app.module.context_embedder.modules()`.
    `torch.testing.assert_close(app(**tiny_hv15_inputs())[0], tiny.reference, atol=1e-4, rtol=1e-4)`.
  - `test_eager_is_refused`: `_app(tiny, quant=tiny.spec.to_dict()).quant == tiny.spec`; `exec_mode="eager"` -> `ValueError` match `"D43"`.
  - `test_a_spec_for_another_tp_is_refused`: `quant=default_spec("hunyuan_video_15", compute="dequant", tp_degree=2)` at tp1 -> `ValueError` match `r"the HunyuanVideo-1.5 FP8 DiT spec .*tp_degree=2 .*tp_degree=1"`.
  - `test_quantized_layouts_match_the_dequantized_reference` (Review Focus 4), parametrised
    `(tp, cfg, sp, world) in [(2, True, False, 4), (2, False, True, 2)]`, ids `tp2-cfg2`, `tp2-sp`:
    a tp2 dequant directory built in the test (`spec2 = default_spec("hunyuan_video_15", compute="dequant", tp_degree=2)`);
    `run_ranks(hv15_fp8_worker, ...)`. With CFG the inputs are stacked
    `[tiny_hv15_inputs(seed=2, valid_text=3), tiny_hv15_inputs()]` and compared against `torch.cat`
    of both references (`hv15_reference(..., spec=spec2)`); every rank's `out` must `assert_close`
    (1e-4) to it. The weight check carries the layout: with `w` the fp32 source weight
    (`safetensors.torch.load_file` of the toy's `transformer/`) and `k = w.shape[1] // 2`, every
    rank and every `n` in `HV15_ROW_TARGETS` must satisfy
    `torch.equal(torch.tensor(r["row_weights"][n]), mx.fake_quantize_mx_sharded(w, 2)[:, r["tp_rank"] * k:(r["tp_rank"] + 1) * k])`,
    and `not torch.equal(mx.fake_quantize_mx(w), mx.fake_quantize_mx_sharded(w, 2))` holds
    (F1's spread). A writer or loader that quantized `net.2` whole would fail the first assertion
    on rank 0 and rank 1, while the outputs would still pass. On the unspread toy the outputs
    matched to 2.4e-7 at both layouts (scratch copy, 2026-11-07). The spread moves the two
    references 6e-7 apart, far inside 1e-4.
  - `test_a_quantized_and_a_bf16_dit_share_no_block_graph_and_no_store_entry` (the native path
    has no stage key; the NEFF cache keys on the HLO): a module-level
    `torch.library.custom_op("difflet_hv15_test::mx_linear_stub", mutates_args=())` returning
    `zeros(x.shape[0], data.shape[1], bf16)` with a fake; `monkeypatch.setattr(fp8_kernels, "_MX_OP", stub)`;
    `DIFFLET_NEURON_WEIGHT_STORE=<tmp>/store`, `DIFFLET_NEURON_WEIGHT_STORE_ALL=1`; the bf16 app on
    `ensure_bf16_copy(transformer_dir, cache)` and the quant app on the directory built from that
    copy with `default_spec("hunyuan_video_15", tp_degree=1)` (nki), both bf16, compile mode,
    `compile_backend` a recording `aot_eager`, `torch._dynamo.reset()` before each `load()`:

```python
    stub_calls = lambda graphs: [sum("mx_linear_stub" in str(n.target) for n in g.graph.nodes) for g in graphs]
    assert stub_calls(quant_graphs) and all(c == len(TINY_TARGETS_PER_BLOCK) for c in stub_calls(quant_graphs))  # one kernel call per target
    assert stub_calls(bf16_graphs) and not any(stub_calls(bf16_graphs))
    assert bf16_app.weight_store.status == quant_app.weight_store.status == "written"
    assert bf16_app.weight_store.directory != quant_app.weight_store.directory
```

- [x] **Step 2: Run** — `pytest tests/unit/backends/test_neuron_hv15_fp8.py -v`; FAIL (`NotImplementedError: ... has no MXFP8 target layers`).
- [x] **Step 3: Implement** (two attributes, the docstrings, the worker).
- [x] **Step 4: Run** — PASS; `pytest tests/unit/backends/test_neuron_hv15_dit.py tests/unit/backends/test_neuron_fp8_linear.py -q` unchanged.
- [x] **Step 5: Commit** — `feat(hv15): --quant mxfp8 on the DiT application (base-class wiring; tp2 x CFG2 and SP match the dequantized reference; a quantized and a bf16 DiT share no block graph and no weight-store entry)`; push.

**Done** (`38a16dc`, review fix `7151334`).

### Task F3: the pipeline resolves the quantized directory from the bf16 copy

**Files:**
- Modify: `difflet/backends/neuron/hunyuan_video15/pipeline.py:34-40` (import `resolve_quant`), `:63-79` (`__init__`; the class docstring names the two options)
- Test: `tests/unit/backends/test_neuron_hv15_fp8.py`

**Interfaces:**
- `TorchNeuronHunyuanVideo15Pipeline.__init__(..., quant=None, quant_checkpoint=None)`:
  `tp = int(getattr(parallel, "tp_degree", 1) or 1)`;
  `self.quant, dit_path = resolve_quant("hunyuan_video_15", quant, quant_checkpoint, resolve_component_dir(os.path.join(self.model_path, "transformer"), dtype), tp_degree=tp)`;
  `self.quantized_dit: str | None = dit_path if self.quant is not None else None`; the DiT
  application gets `model_path=dit_path, quant=self.quant`. `quant` is `"mxfp8"`, a `QuantSpec`
  or its dict (the stage passes the dict, F4).

- [x] **Step 1: Write the failing tests** — fixture `snap`: `write_tiny_hv15_snapshot(root)`, `cache = root / "cache"`,
  `copy = ensure_bf16_copy(root / "transformer", cache)`, quantized directories built from `copy`
  for `default_spec("hunyuan_video_15", tp_degree=1)` and `tp_degree=2`; each test sets
  `DIFFLET_COMPILE_CACHE=cache`; `_pipe(**over)` = the pipeline with `dtype=torch.bfloat16,
  exec_mode="compile", device="cpu", text_seq_len=16, text_2_seq_len=8, image_tokens=9`, 32x48x5 (not loaded).
  - `test_pipeline_resolves_the_quantized_directory_from_the_bf16_copy`: `quant="mxfp8"` ->
    `pipe.dit.model_path == pipe.quantized_dit == str(<tp1 dir>)`, `pipe.quant == pipe.dit.quant == default_spec("hunyuan_video_15", tp_degree=1)`.
  - `test_the_pipeline_spec_takes_the_launch_tp`: `parallel=DiffletParallelConfig(tp_degree=2, cfg_parallel_enabled=True)`
    -> `pipe.quant.tp_degree == 2` and the tp2 directory.
  - `test_bf16_loads_the_copy_and_holds_no_spec`: `quant=None` -> `pipe.dit.model_path == str(copy)`, `pipe.quant is None`, `pipe.quantized_dit is None`.
  - `test_without_the_copy_the_quantized_directory_is_missing`: a second toy root without a copy
    -> `ValueError` match ``"run `difflet compile`"``; `quant_checkpoint="/q"` with `quant=None`
    -> `ValueError` match `"quant_checkpoint needs quant"`.
- [x] **Step 2: Run** — FAIL (`TypeError`: unexpected keyword argument `quant`).
- [x] **Step 3: Implement.**
- [x] **Step 4: Run** — `pytest tests/unit/backends/test_neuron_hv15_fp8.py tests/unit/backends/test_neuron_hv15_pipeline.py -q`; PASS.
- [x] **Step 5: Commit** — `feat(hv15): the pipeline resolves the quantized DiT directory from the bf16 copy at the launch's tp`; push.

**Done** (`df1e593`).

### Task F4: the HunyuanVideo-1.5 orchestrator

**Files:**
- Modify: `difflet/cli/orchestrators/hunyuan_video_15.py`: module docstring `:1-35` (a `--quant` paragraph:
  where the directory comes from, the tp, no stage key and why); module-top imports `:45-52` gain
  only torch-free names (`_local_snapshot`, `refuse` from `difflet.cli.orchestrators.torchrun`;
  `prepare_quant`, `quant_cli_args`, `quant_spec` from `difflet.cli.quant`); `_unsupported` (`:63-64`, F1's refusal removed);
  `:91-132` (`_run`: `previous` read before `_prepare_quant`, which runs after the bf16 copy and
  `_layout`, before the work dir and any launch); `:134-159` (`_stage_cli_args`: `parts += quant_cli_args(a)` last);
  `:222-229` (`_stage_transformer`: the pipeline gets the spec)
- Test: `tests/unit/cli/test_orchestrator_hv15_neuron.py`:
  - the `launches` fixture at `:31-46` also stubs the snapshot as not local, and records-then-deletes
    `DIFFLET_NKI_TRACE_CACHE` with `setenv(..., "")` + `delenv`;
  - F1's refusal test is replaced;
  - the two inline `FakePipeline` classes (`test_transformer_stage_saves_latents` `:113-125`,
    `test_batch_transformer_replays_the_claims_with_each_seed` `:190-201`) become one module helper,
    `_fake_pipeline(seen: dict, latents=lambda generator: torch.full((1, 4, 2, 2, 3), 7.0)) -> type`.
    Its class records `seen["init"] = kw`, `seen["skip_warmup"]`, `seen["steps"]` and appends
    `generator.initial_seed()` to `seen.setdefault("seeds", [])`. It holds
    `self.dit = SimpleNamespace(phase_seconds={})` and `self.guidance_scale = 6.0`, and returns
    `latents(generator)`. The batch test passes
    `latents=lambda g: torch.full((1, 4, 2, 2, 3), float(g.initial_seed()))` and asserts
    `seen["seeds"] == [5, 9]`. Every transformer-stage test, including the new one below, uses
    it, so F5 extends one fake.

**Interfaces:**
- `HunyuanVideo15Orchestrator._quantized_dit: str | None = None` (class attribute).
- `_prepare_quant(self, *, create: bool, tp: int) -> None`: no-op without `--quant`; `root = _local_snapshot(lambda: self._root(self.args))`,
  `None` -> `refuse(f"--quant: the {_HF_MODEL_ID} weights are not local; run `difflet download --model-id {_HF_MODEL_ID}` first")`;
  then, inside the function (Global Constraints: the module stays torch-free; Wan's `_prepare_quant`
  at `wan.py:417-423` does the same), `import torch` and
  `from difflet.backends.neuron.core.weights_bf16 import resolve_component_dir`;
  `sources = {"transformer": resolve_component_dir(os.path.join(root, "transformer"), torch.bfloat16)}`;
  `self._quantized_dit = (prepare_quant(self.args, sources, create=create, tag="hv15", default_tp_degree=tp, fail=refuse) or {}).get("transformer")`.
- `_run` calls `self._prepare_quant(create=mode == "compile", tp=tp)`.
- `_stage_transformer`: `spec = quant_spec(args, default_tp_degree=int(args.tp_degree or 1))`;
  `TorchNeuronHunyuanVideo15Pipeline(..., quant=None if spec is None else spec.to_dict())`.
- Test helpers: `_no_local_snapshot(monkeypatch)` (patches `difflet.pipeline.path_resolver.resolve_model_path`
  to raise `FileNotFoundError`), `_toy_snapshot(monkeypatch, tmp_path) -> Path` (`write_tiny_hv15_snapshot`,
  `resolve_model_path` returning it, `DIFFLET_COMPILE_CACHE=tmp_path / "cache"`).

- [x] **Step 1: Write the failing tests**
  - `test_default_layout_is_tp2_x_cfg2` gains the byte-identity check: `launches[1][2] == ["--model-id", hv15._HF_MODEL_ID, "--stage-mode", "generate", "--work-dir", str(tmp_path / "w"), "--tp-degree", "2", "--cp-degree", "1", "--height", "480", "--width", "848", "--num-frames", "121", "--steps", "50", "--seed", "42", "--cfg-parallel", "--prompt", "a cat", "--output", "/tmp/o.mp4"]`.
  - `test_quant_reaches_the_transformer_stage_at_the_layout_tp` (Review Focus 1, 5), parametrised
    `(over, tp, cores) in [({}, 2, 4), ({"cfg_parallel": True}, 1, 2), ({"cp_degree": 2}, 1, 2), ({"tp_degree": 4}, 4, 4)]`:
    on the toy, `compile()` then `generate()` with `quant="mxfp8"`; the transformer launch has `cores`
    cores and `--quant mxfp8`, `--tp-degree str(tp)`; stdout has `[hv15] MXFP8 transformer (nki): {orch._quantized_dit}`;
    `read_manifest(orch._quantized_dit)["spec"]["tp_degree"] == tp`;
    `check_quantized_dir(orch._quantized_dit, valid_copy(root / "transformer"), quant_spec(args, default_tp_degree=tp))` passes (built from the copy);
    the stage parser's namespace for that line gives `quant_spec(stage_args) == quant_spec(args, default_tp_degree=tp)`.
  - `test_quant_generate_before_compile_is_refused_before_any_launch` (Review Focus 2): toy, no copy,
    `generate()` -> `SystemExit` code 2; stderr starts `"Error: --quant: transformer: "`, contains
    ``"run `difflet compile` with the same --quant flags first"``, ends `"; nothing was launched.\n"`;
    `launches == []`; `not (tmp_path / "cache" / "quantized").exists()`.
  - `test_quant_without_local_weights_is_refused_naming_download`: `_no_local_snapshot` -> exit 2,
    stderr `== f"Error: --quant: the {HF} weights are not local; run `difflet download --model-id {HF}` first; nothing was launched.\n"`.
  - `test_the_native_path_keeps_no_args_keyed_artifact` (a regression pin, not a proof: the
    native path never calls these helpers, so after F4 it passes whether or not NEFFs key on the
    HLO. That proof is F2's graph-and-store test. The pin fails if a later change adds an
    args-keyed stage artifact to the native path): `difflet.cli.orchestrators.base.hashed_stage_dir`,
    `write_stage_manifest` and `has_valid_stage_manifest` patched to `pytest.fail`; a `--quant` compile and
    generate on the toy complete.
  - `test_a_quant_launch_restores_the_trace_cache_opt_in`: after a `--quant` generate,
    `launches.env_seen == ["1", "1", "1"]` and `"DIFFLET_NKI_TRACE_CACHE" not in os.environ`.
  - `test_transformer_stage_builds_the_pipeline_with_the_spec` (Review Focus 5): `_fake_pipeline(seen)`
    with `quant="mxfp8", model_id=hv15._HF_MODEL_ID`, tp2, CFG -> `seen["init"]["quant"] == default_spec("hunyuan_video_15", tp_degree=2).to_dict()`;
    without `--quant` -> `seen["init"]["quant"] is None`.
  - `test_the_driver_module_imports_no_torch_and_no_neuron_backend` (a regression pin, never
    skipped; it passes at 917d087 and must still pass after F4 and F5):
    ```python
    code = ("import sys\n"
            "import difflet.cli.orchestrators.hunyuan_video_15\n"
            "bad = sorted(m for m in sys.modules if m == 'torch' or m.startswith('difflet.backends.neuron'))\n"
            "assert not bad, bad\n")
    subprocess.run([sys.executable, "-c", code], check=True, env=dict(os.environ, DIFFLET_BACKEND="cpu"))
    ```
- [x] **Step 2: Run** — `pytest tests/unit/cli/test_orchestrator_hv15_neuron.py -v`.
  - FAIL: every test that runs a `--quant` generate or compile. F1's refusal exits 2 with its own
    message. That includes `test_the_native_path_keeps_no_args_keyed_artifact` and the
    trace-cache test, which fail now only through that refusal.
  - FAIL: `test_transformer_stage_builds_the_pipeline_with_the_spec`, with a `KeyError` on
    `"quant"` (the stage passes no spec).
  - PASS already: the byte-identity check and the import test. Both are pins.
- [x] **Step 3: Implement.**
- [x] **Step 4: Run** — `pytest tests/unit/cli -q`; PASS (`test_cli_hunyuan_video_15_orchestrator.py`'s AoT tests unchanged).
- [x] **Step 5: Commit** — `feat(hv15): --quant mxfp8 through the driver — quantized directory from the bf16 copy at the layout's tp, stage flags, the spec on the ranks; no stage key on the native path (NEFFs key on the HLO)`; push.

**Done** (`cc8838d`, review fix `b086588`).

### Task F5: the transformer stage reports the DiT's memory

**Files:**
- Modify: `difflet/cli/orchestrators/hunyuan_video_15.py:230-233` (rank 0 prints `_dit_memory_line` after the load line), `:252-257` (rank 0 prints `_peak_hbm_line` after each request's timing line when it is not None); two module functions beside `_aot_write_output`, each importing what it uses inside its body (Global Constraints: `rank_weight_bytes` from `difflet.backends.neuron.core.application_base`, `GIB` / `per_core_hbm_bytes` from `difflet.backends.neuron.hbm`, `torch_neuronx`), so the module stays torch-free at import
- Test: `tests/unit/cli/test_orchestrator_hv15_neuron.py` (F4's `_fake_pipeline` helper's dit gains `module=torch.nn.Linear(2, 2), dtype=torch.bfloat16, weight_bytes=8, quant=None, device=torch.device("cpu")`; every transformer-stage test uses that helper), `tests/unit/backends/test_neuron_hv15_fp8.py`

**Interfaces:**
- `_dit_memory_line(dit) -> str`:
  `f"[hv15] DiT weights per rank: {rank_weight_bytes(dit.module, dit.dtype) / GIB:.2f} GiB resident ({dit.weight_bytes / GIB:.2f} GiB estimated at load), {label}"`,
  `label = "bf16"` or `f"MXFP8 {dit.quant.compute}"` (`rank_weight_bytes` from `core/application_base.py`,
  on the loaded module: every parameter and buffer, the kernel layout's `weight_packed` / `weight_scales_hw` included).
- `_peak_hbm_line(dit) -> str | None`: None unless `dit.device.type == "neuron"`; else, with
  `peak = torch_neuronx.max_memory_allocated()` and `hbm, _ = per_core_hbm_bytes()`:
  `f"[hv15] peak HBM (rank 0, allocator): {peak / GIB:.2f} GiB of {hbm / GIB:.1f} GiB per core, {(hbm - peak) / GIB:.2f} GiB free"`.

- [x] **Step 1: Write the failing tests**
  - `test_resident_bytes_count_the_kernel_layout` (`test_neuron_hv15_fp8.py`; a regression pin on
    F2's application and the existing `rank_weight_bytes` / `prepare_kernel_`, so it passes
    before this task's change. It pins the arithmetic the memory line reports and the Tech
    Stack's 1.28 bytes per weight): the bf16
    app, the nki app and the dequant app on F2's toy (bf16 dtype, `load(skip_warmup=True)`), `nk` = Σ N·K
    of the targets in diffusers' model:
    `rank_weight_bytes(bf16.module, bf16.dtype) - rank_weight_bytes(nki.module, nki.dtype) == nk * 23 // 32`,
    `rank_weight_bytes(nki.module, nki.dtype) - nki.weight_bytes == nk // 4`,
    `rank_weight_bytes(dq.module, dq.dtype) - rank_weight_bytes(bf16.module, bf16.dtype) == nk // 32` (probed exact).
  - `test_dit_memory_line_names_resident_and_estimated_bytes` (Review Focus 3: the line reads the
    loaded module, not `weight_bytes`): a dit with `module = torch.nn.Linear(32768, 32768, bias=False, device="meta")`,
    `dtype=torch.bfloat16, weight_bytes=GIB, quant=None` -> `"[hv15] DiT weights per rank: 2.00 GiB resident (1.00 GiB estimated at load), bf16"`;
    with `quant=default_spec("hunyuan_video_15", tp_degree=2)` the line ends `", MXFP8 nki"`.
  - `test_peak_hbm_line_only_on_the_device`: `device=torch.device("cpu")` -> None; `SimpleNamespace(type="neuron")`
    with `sys.modules["torch_neuronx"]` (`monkeypatch.setitem`) a namespace whose `max_memory_allocated` returns
    `20 * GIB` and `difflet.backends.neuron.hbm.per_core_hbm_bytes` returning `(36 * GIB, "test")` ->
    `"[hv15] peak HBM (rank 0, allocator): 20.00 GiB of 36.0 GiB per core, 16.00 GiB free"`.
    The function imports `per_core_hbm_bytes` inside its body, so the patched attribute is the
    one it calls.
  - `test_transformer_stage_saves_latents` additionally finds `"[hv15] DiT weights per rank: "` in stdout and no `"peak HBM"`.
- [x] **Step 2: Run** — `pytest tests/unit/cli/test_orchestrator_hv15_neuron.py tests/unit/backends/test_neuron_hv15_fp8.py -v`.
  - FAIL: the two line tests (ImportError: `_dit_memory_line`) and the stdout check in
    `test_transformer_stage_saves_latents`.
  - PASS already: `test_resident_bytes_count_the_kernel_layout`, a pin.
- [x] **Step 3: Implement.**
- [x] **Step 4: Run** — `pytest tests/unit/cli tests/unit/backends/test_neuron_hv15_fp8.py -q`; PASS. This
  includes F4's `test_the_driver_module_imports_no_torch_and_no_neuron_backend`, since F5 adds
  module functions to the driver.
- [x] **Step 5: Commit** — `feat(hv15): the transformer stage reports the DiT's resident bytes per rank (the kernel's scale layout included) and the peak HBM`; push.

**Done** (`f61e833`, review fix `b35ec41`).

### Task F6: HunyuanVideo-1.5 on the device (trn3, tp2 x CFG2, 480x848x121, 50 steps, seed 42)

**Done 2026-11-08** (evidence section 9; raw files `/home/ubuntu/fp8-bench/hv15/`). `mx` (G1-G4)
failed S (21551 ms, 2.10x; 2.74x at 9 frames); the bisection at 9 frames found every group alone
at speed and every set with both the image and the text q/k/v 2.7x, so `S` = G1 + G3 (`u3`,
0.917x). `FINAL` = `mx_final`: 10314 / 10313 ms against 10241 in bf16 (1.007x, passes S),
bit-identical, 6.97 against 7.80 GiB of DiT weights per rank but the same peak HBM (17.66 against
17.56 GiB), and 12.76 dB / SSIM 0.190 / LPIPS 0.510 from bf16 against the floor's 15.90 / 0.312 /
0.363: it fails Q on SSIM (0.190 against 0.212), so Step 7(e) stopped and reported. **Decision
(user, 2026-11-08, option A):** G1 + G3 is the default despite the nominal SSIM miss, quality
pending phase G; committed as `9f6aedc` (Step 7(f)). Issue
[#145](https://github.com/ai-decentralized/Difflet/issues/145) tracks the problem. Per step:
- Step 0: `$D4` found and `apply_rotary_emb_fp32` called at lines 236 and 245 of the transformer
  (`step0_d4.txt`); `phase_watch.sh` written. The pytest gate's output is not in the run
  directory.
- Step 0b: every shape passes (rel L2 2.325-2.358e-3, cosine 0.999997), every ratio below 1.0
  (0.40 to 0.87). Deviation: Dynamo's recompile limit stopped the probe after eight shapes; the
  ninth (CP2 image MLP down) ran alone and passed (`probe_9th.log`).
- Step 1: the snapshot is `286be7ce`; this step's `df`, config assertion and pytest are not in
  the run directory.
- Steps 2-3: `bf16_pre` (`d1af3e1`) 9924 / 9924 ms and `bf16` (`41d0685`) 10238 / 10235 ms, each
  bit-identical. Deviations: both ran unsampled (`run_bf16_arms.sh`), and `41d0685` predates F5,
  so the bf16 memory lines and the `compare` against `bf16_pre` come from `bf16f`, one sampled
  generate of bf16 from `73f1722` whose mp4 and latents equal `bf16`'s. D4 fails phase D's keep
  rule (1.031x `bf16_pre`; 15.44 dB from it against the floor's 15.90) and was not reverted
  before Step 4: recorded in evidence 9b, open.
- Step 4: floor 15.90 dB / 0.312 / 0.363; `gate.json`: 11.90 dB / 0.212 / 0.727.
- Step 5: `bf16_9` 557 / 559 ms, `mx_9` 1528 / 1519 ms (2.74x), both bit-identical;
  `mxfp8-3495b42101abbf8d`, 12 GB, 339 GB free after it. The bf16 copy's size is not recorded.
- Step 6: `mx` 21551 ms (2.10x, fails S), no `NCC_INKI016`, 6.14 GiB resident (5.56 estimated)
  against 7.80; to Step 7.
- Step 7: (a) 9 frames; (b) `g1`-`g4` 546 / 554 / 528 / 556 ms, all pass S; (c) `U` = G1-G4, which
  is `mx_9`'s set, so `mx_9` stands for `u1`; (d) `u2` (no G4) 1514 ms fails, `u3` (no G2) 511 ms
  passes; (e) `mx_final` as above; (f) `9f6aedc`, CPU gate at `c7c2346` 2103 passed, 111 skipped,
  0 failed; device check from a snapshot of `c7c2346` with no env: `compile --quant mxfp8` exit
  0, its MXFP8 line names `mxfp8-afcbee695501d1d2` (`mx_final`'s); (g) the `g1`-`g4` and `u2`
  directories deleted, `mx_9`'s kept (disk not tight, 282 GB free).
- Step 8: `bf16_cp2` 5687 / 5682 ms, `mx_cp2` 10198 / 9539 ms (1.79x / 1.68x), both
  bit-identical, the same 6.97 GiB per rank as `FINAL`. Deviations: `mx_cp2` ran before
  `9f6aedc` with `DIFFLET_QUANT_TARGETS` equal to the default's spec (the same directory), not
  with no env; its two medians differ by 7% on a busy host; the three `compare`s were not run.
- Step 9: evidence section 9, README and `docs/cli-staged-commands.md`. Not reported:
  neuron-monitor's per-core maximum and the G1 + G3 directory's size on disk. The status page is
  updated outside this plan's commits.

No code unless a step fails (one bug, one commit, the `difflet-device-verify` protocol). The
one exception is Step 7(f)'s default-set commit. Every command runs with this preamble. Each
arm's logs, mp4s, work dirs and resource samples go under `$W/<arm>/`.

```bash
export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:$PATH
PY=/home/ubuntu/venvs/native-fp8/bin/python; W=/home/ubuntu/fp8-bench/hv15; mkdir -p $W
HEAD_SRC=/home/ubuntu/Difflet/.claude/worktrees/mxfp8-video-models
D4=$(git -C $HEAD_SRC log --format=%H -1 --grep="HunyuanVideo-1.5's attention processors rotate")  # Step 0 checks it
C=(--model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v --backend neuron
   --cache-dir /home/ubuntu/difflet-cache-video)
G=("${C[@]}" --steps 50 --seed 42 --prompt "a cat walking through a garden")
# C is for compile: its subparser has no generate flags (difflet/cli/main.py:780-789: --steps,
#   --seed, --prompt exit 2), and its stage lines carry --steps 50 --seed 42 by default
#   (_stage_cli_args). G is for generate.
# compile:  PYTHONPATH=<tree> $PY -m difflet.cli.main compile "${C[@]}" <layout> [--quant mxfp8] > $W/<arm>/compile.log 2>&1
# generate: PYTHONPATH=<tree> $PY -m difflet.cli.main generate "${G[@]}" <layout> [--quant mxfp8] --output $W/<arm>/gen<x>.mp4 \
#   --work-dir $W/<arm>/work_<x> --keep-work-dir > $W/<arm>/gen<x>.log 2>&1   (x = a, b)
# A bisection arm puts DIFFLET_QUANT_TARGETS='<patterns, ;-joined>' as a prefix on both its
#   compile line and its generate line. Never export it; `unset DIFFLET_QUANT_TARGETS` before
#   any pytest.
# step median: the "[hv15] request 0: denoise ..., median N ms (first M ms)" line.
# memory: the "[hv15] DiT weights per rank" and "[hv15] peak HBM" lines (F5).
# sampled a-run (every 121-frame arm): the generate runs in the background, labelled by stage:
#   PYTHONPATH=<tree> $PY -m difflet.cli.main generate ... > $W/<arm>/gena.log 2>&1 & GEN=$!
#   bash $W/phase_watch.sh $W/<arm> $W/<arm>/gena.log & WATCH=$!
#   $PY $HEAD_SRC/scripts/sample_serving_resources.py --pid $GEN --interval 15 \
#     --output $W/<arm>/resources.jsonl --phase-file $W/<arm>/phase &
#   wait $GEN; kill $WATCH 2>/dev/null
# compare:  PYTHONPATH=$HEAD_SRC $PY $HEAD_SRC/scripts/compare_difflet_runs.py --ref-latents $W/A/work_a/latents.pt \
#   --ref-image $W/A/gena.mp4 --cand-latents $W/B/work_a/latents.pt --cand-image $W/B/gena.mp4 --out $W/B/vs_A.json
#   (mp4s carry the codec floor; the stage's .pt output is the raw (B, C, F, H, W) decode in
#   [-1, 1], which the tool refuses; the latents are lossless)
```

`$W/phase_watch.sh` (Step 0 writes it) labels each sample by the stage that is running. The
runner prints nothing at a stage boundary, so the script keys on the stages' own lines:

```bash
#!/bin/bash
# $1 = the arm's directory, $2 = the generate's log
echo encode > "$1/phase"
until grep -q '\[hv15\] encode request 0:' "$2" 2>/dev/null; do sleep 2; done; echo transformer > "$1/phase"
until grep -q '\[hv15\] request 0: denoise' "$2" 2>/dev/null; do sleep 2; done; echo vae > "$1/phase"
```

The `transformer` label spans the DiT stage's start-up, NEFF and weight load and the 50 steps.
The one-core VAE decode (also on core 0) is labelled `vae`. The memory table's neuron-monitor
number is, for each core of the transformer launch, the maximum `memory_used` over the samples
labelled `transformer`.

**Gates.** Every ratio is a step median against the bf16 arm at the same shape and layout.
- Speed (S): a ratio at or below 1.01x.
- Quality (Q): the set's `compare` against `bf16` at 121 frames, judged against the floor. The
  floor is Step 4's `compare` of `bf16_tp4` against `bf16`; `fp`, `fs` and `fl` are its clip
  `psnr`, `ssim` and `lpips`. A set passes Q when:
  - `psnr ≥ min(30, fp − 4)`;
  - `ssim ≥ min(0.97, fs − 0.10)`;
  - `lpips ≤ max(0.05, 2 · fl)`.

  When the floor is high, these are D5's gates (PSNR ≥ 30 dB, SSIM ≥ 0.97, LPIPS ≤ 0.05). D5's
  gates cannot be used unchanged at 121 frames: LTX-2's bf16 floor at 121 frames was
  25.9 dB / 0.909 / 0.040 (evidence section 3a), below D5's PSNR and SSIM gates, so they would
  refuse bf16 against itself. The margins are where LTX-2's shipped default sat against its
  floor (22.5 dB / 0.830 / 0.078: 3.4 dB, 0.079 and 1.95x).
- A set that fails Q is never made the default by this task. Stop and report the numbers:
  phase G's CPU fp32 reference decides whether the loss is fp8's, and the trade is the user's.

**Generates.** Every arm gets two generates (determinism: identical mp4 and latents md5s).
There are two exceptions. The bisection arms (`g1`..`g4`, `u1`..) get one, because they are
timing screens; the set they choose is run twice as `mx_final`. `mx` gets one when Step 7 runs
and replaces its set.

| arm | tree | layout flags | quant / env | generates |
|---|---|---|---|---|
| `bf16_pre` | `$W/src-pre-d4` | (default: tp2 x CFG2, 4 cores) | bf16, old rotary | 2 |
| `bf16` | `$HEAD_SRC` | (default) | bf16 | 2 |
| `bf16_tp4` | `$HEAD_SRC` | `--tp-degree 4 --cfg-parallel` (8 cores) | bf16 (noise floor) | 2 |
| `bf16_9`, `mx_9` | `$HEAD_SRC` | (default) `--num-frames 9` | bf16 / `--quant mxfp8` | 2 |
| `mx` | `$HEAD_SRC` | (default) | `--quant mxfp8` | 1, then 2 unless Step 7 replaces its set |
| `g1`..`g4`, `u1`.. | `$HEAD_SRC` | (default), at Step 7(a)'s frame count | `--quant mxfp8`, `DIFFLET_QUANT_TARGETS` | 1 |
| `mx_final` | `$HEAD_SRC` | (default) | `--quant mxfp8`, `DIFFLET_QUANT_TARGETS` = Step 7's set | 2 |
| `bf16_cp2`, `mx_cp2` | `$HEAD_SRC` | `--tp-degree 2 --cfg-parallel --cp-degree 2` (8 cores, Ulysses) | bf16 / `--quant mxfp8` (the final default) | 2 |

`FINAL` names the arm of the final default set: `mx`, or `mx_final` when Step 7 changes the set.

- [x] (partial: the pytest gate's output is not recorded, see Done) **Step 0: CPU gate.**
  - `unset DIFFLET_QUANT_TARGETS`, then run `pytest tests/unit/quant tests/unit/backends/test_neuron_hv15_dit.py tests/unit/backends/test_neuron_hv15_pipeline.py tests/unit/backends/test_neuron_hv15_fp8.py tests/unit/cli -q`. All must PASS.
  - `$D4` is non-empty (phase D's D4 landed), and `grep -n apply_rotary_emb_fp32 $HEAD_SRC/difflet/backends/neuron/hunyuan_video15/transformer.py`
    finds the call.
  - Write `$W/phase_watch.sh`.
- [x] **Step 0b: one-core kernel probe at HV1.5's per-rank shapes**, before any download or
  model compile. The largest M probed so far is 6,144 (evidence section 1); HV1.5 runs 49,290.
  1. Copy `/home/ubuntu/fp8-bench/ltx2_probe/probe_mx_shapes.py` to `$W/probe/probe_mx_shapes_hv15.py`
     (a bench script outside the repo), with `OUT = Path("/home/ubuntu/fp8-bench/hv15/probe")`.
  2. Replace its `SHAPES` with HV1.5's tp2 per-rank shapes, `(label, M, K, N)`:

     | label | M | K | N |
     |---|---|---|---|
     | `attn.to_q/k/v (image, col)` | 49290 | 2048 | 1024 |
     | `ff.net.0.proj (image, col)` | 49290 | 2048 | 4096 |
     | `ff.net.2 (image, row K=8192/2)` | 49290 | 4096 | 2048 |
     | `attn.add_q/k/v_proj (text, col)` | 1985 | 2048 | 1024 |
     | `ff_context.net.0.proj (text, col)` | 1985 | 2048 | 4096 |
     | `ff_context.net.2 (text, row)` | 1985 | 4096 | 2048 |
     | `CP2 attn.to_q/k/v (image/2)` | 24645 | 2048 | 1024 |
     | `CP2 ff.net.0.proj (image/2)` | 24645 | 2048 | 4096 |
     | `CP2 ff.net.2 (image/2)` | 24645 | 4096 | 2048 |
  3. Add a bf16 arm per shape: the same `Layer` with an `nn.Linear(k, n, bias=True, dtype=torch.bfloat16)`
     holding `mx.dequantize_mx(q, s)` and the bias, compiled and timed the same way. The result
     gains `bf16_median_ms` and `ratio` (mx over bf16).
  4. In the timing loops of both arms, sync on `out[0, 0].item()` instead of `out.cpu()`. A
     49,290 x 4096 bf16 output is 404 MB, and its host copy would swamp both arms. The accuracy
     check still copies the first call's whole output.

  Run `PROBE_WT=$HEAD_SRC NEURON_RT_NUM_CORES=1 $PY $W/probe/probe_mx_shapes_hv15.py > $W/probe/probe.log 2>&1`.
  Gate: every shape passes the probe's own criterion (rel L2 < 3e-3, cosine > 0.9999, output
  `[M, N]`, no kernel assert). A failure stops F6 before any model compile (one bug, one commit).
  Record each shape's two medians and ratio. A ratio above 1.0 predicts the group Step 7 may
  drop: image rows are G1/G3, text rows G2/G4.
- [x] (partial: the `df`, the config assertion and the pytest are not recorded, see Done) **Step 1: download.**
  - `df -h /home/ubuntu`, then `PYTHONPATH=$HEAD_SRC $PY -m difflet.cli.main download --model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v`, then `df` again.
  - Check the premise of F1's real-geometry test. With `S` the path that
    `$PY -c "from difflet.pipeline.path_resolver import resolve_model_path as r; print(r('hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v', local_files_only=True))"`
    prints, `$PY -c "import json,sys; c=json.load(open(sys.argv[1])); assert (c['num_layers'], c['num_attention_heads'], c['attention_head_dim'], c['mlp_ratio']) == (54, 16, 128, 4.0), c" $S/transformer/config.json`
    must pass.
  - `unset DIFFLET_QUANT_TARGETS; pytest tests/unit/cli -q`. The real snapshot is now in the hub
    cache, so this must stay green with no conftest-guard failure.
- [x] (partial: the generates ran unsampled, see Done) **Step 2: `bf16_pre`.**
  - Here, after Step 0 has checked `$D4`, extract the tree before D4 (the old rotary):
    `mkdir -p $W/src-pre-d4 && git -C $HEAD_SRC archive "$D4^" | tar -x -C $W/src-pre-d4`.
  - Compile. It writes the 16.7 GB bf16 copy, shared by every later arm, and the first trace of
    the key-bounded attention kernel goes to the NKI trace cache.
  - Two generates (sampled a-run): md5s, and the bf16 step with diffusers' rotary, so the
    evidence can state what D4 did to HunyuanVideo-1.5. This tree predates F, so its logs have
    no F5 memory lines; its memory comes from the resource samples.
- [x] (partial: unsampled; the memory lines and the `compare` come from `bf16f`, see Done) **Step 3: `bf16`.** Compile, two generates (sampled a-run). Expected:
  - identical mp4 and latents md5s;
  - the step median, the baseline every 121-frame ratio uses, against `bf16_pre`'s;
  - `compare` vs `bf16_pre` (D4 is bit-identical on the CPU; on the device, record what it is);
  - the memory lines.
- [x] **Step 4: `bf16_tp4` (noise floor).** Compile and two generates (sampled a-run). Run
  `compare` against `bf16`: that is the floor. Write `fp`, `fs`, `fl` and Q's three thresholds
  to `$W/gate.json`.
- [x] (partial: the bf16 copy's size is not recorded, see Done) **Step 5: `bf16_9` and `mx_9`** at `--num-frames 9` (4,770 image rows): the cheap signal,
  before any 121-frame MXFP8 compile.
  - Compile and two generates each.
  - `mx_9`'s compile writes the quantized directory. The spec carries no shape, so `mx` and
    `mx_cp2` reuse it. Record the `[hv15] MXFP8 transformer (nki): <dir>` line, `du -sh <dir>`
    against the copy, and `df`.
  - Record the md5s, the ratio `mx_9` / `bf16_9`, and `compare` of `mx_9` vs `bf16_9`. The
    comparison is recorded but not gated: there is no 9-frame floor.
- [x] **Step 6: `mx` at 121 frames.**
  - Compile. Its MXFP8 line names `mx_9`'s directory.
  - Generate a (sampled). Expected: no `NCC_INKI016` (49,290 and 1,985 rows padded) and about
    6.1 against 7.8 GiB resident per rank. Record the median and first-step ratios and
    `compare` vs `bf16`, and apply Q.
  - If the median ratio passes S: generate b (md5s identical), and `FINAL = mx`. If it then fails
    Q, stop and report.
  - If the median ratio fails S, go to Step 7.
- [x] **Step 7: bisection** (only when `mx` failed S). The groups:
  - G1 `transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v)` (image q/k/v: norm, rotary, attention);
  - G2 `transformer_blocks\.\d+\.attn\.(add_q_proj|add_k_proj|add_v_proj)`;
  - G3 `transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)`;
  - G4 `transformer_blocks\.\d+\.ff_context\.net\.(0\.proj|2)`.

  (a) The shape: 9 frames if `mx_9` failed S against `bf16_9`, else 121. The reference is
  `bf16_9` or `bf16`.
  (b) Arms `g1`..`g4`: `DIFFLET_QUANT_TARGETS='<Gi>'`, compile and one generate each at that
  shape. Record each ratio.
  (c) `U` is the union of the groups that pass S, in G order. If no group passes, run `mx`'s
  b-generate, then stop and report: the memory-for-time trade is the user's.
  (d) If `U` holds one group, `S` is that group. If it holds two or more, measure `U` itself as arm `u1` at the same shape
  (`DIFFLET_QUANT_TARGETS` = its patterns `;`-joined in G order): compile and one generate. The
  slowdowns need not add up (Wan's 5x was a compiler scheduling effect), so while the last `u`
  arm fails S, drop the remaining group with the highest `g` ratio and measure the rest
  (`u2`, `u3`, ...). A single group has already passed S as its `g` arm, so this loop ends. Its
  last set is `S`.
  (e) `mx_final`: `S` at 121 frames under the same env prefix. Compile and two generates
  (sampled a-run). Record the md5s, the ratio and `compare` vs `bf16`. Apply S and Q; if either
  fails, stop and report.
  (f) Make `S` the default:
  - `HUNYUAN_VIDEO_15_TARGETS` becomes `S`'s group patterns in G order, written
    `rf"{_BLOCK}\..."` so that each renders exactly the string the env carried. The quantized
    directory is keyed by `targets_sha256`, the ordered pattern strings (`difflet/quant/spec.py:134-135`),
    so the new default resolves `mx_final`'s directory.
  - The dropped groups go to `HUNYUAN_VIDEO_15_KEPT_BF16`, with the measured reason in the
    table comment and the docstring paragraph. `TINY_TARGETS_PER_BLOCK` follows; F1's and F2's
    counts derive from it.
  - If G3 or G4 is dropped, `HV15_ROW_TARGETS` loses the dropped stream's `net.2`, and
    `hv15_fp8_worker`'s `FP8RowParallelLinear` / SP-twin assertion moves to the remaining
    stream's `net.2`. With both dropped, no row-parallel target is left: the weight check and
    the twin assertion go, and the commit message says so.
  - `unset DIFFLET_QUANT_TARGETS`, then run Step 0's gate.
  - Commit `feat(quant): HunyuanVideo-1.5's default MXFP8 set is <groups> — <dropped> cost the tp2 x CFG2 step <x> (<a> vs <b> ms), without it <c> ms against <d> in bf16 (<ratio>), <psnr> dB from bf16 (floor <fp> dB)`;
    push. `FINAL = mx_final`.

  (g) Delete the quantized directories of the `g` and `u` arms (each compile log's MXFP8 line
  names one), never `FINAL`'s: `mx_cp2` reads it. The old default's directory (`mx_9`'s) goes
  only if the disk is tight. Run `df` before and after.
- [x] (partial: `mx_cp2` ran with `DIFFLET_QUANT_TARGETS` equal to the default's spec and the three `compare`s were not run, see Done) **Step 8: `bf16_cp2` and `mx_cp2`**, with the final default and no env. Compile and two
  generates each (sampled a-runs). Expected:
  - `mx_cp2`'s MXFP8 line names `FINAL`'s directory (the same tp2 spec);
  - no `NCC_INKI016` (24,645 image rows per rank, padded to 24,648);
  - the same per-rank DiT bytes as `FINAL` (CP replicates the weights);
  - md5s, and the ratio `mx_cp2` / `bf16_cp2`;
  - `compare` of `mx_cp2` vs `bf16`, vs `FINAL` and vs `bf16_cp2` (recorded; CP2 moves the
    output by about the reduction-order floor, evidence 3c).
- [x] (partial: neuron-monitor's per-core maximum and the G1 + G3 directory's size are not reported, see Done) **Step 9: evidence.** Add a HunyuanVideo-1.5 section to `docs/verification/2026-11-07-mxfp8-video-models.md`,
  numbered the next free section. Wan's will be section 4 (`difflet/quant/targets.py:166` cites
  it); D's and E's follow when they land. The section holds:
  - the kernel probe table (Step 0b);
  - the memory table for `bf16` vs `FINAL` at tp2 x CFG2 and `bf16_cp2` vs `mx_cp2`: the bf16
    copy and the quantized directory on disk; per rank resident and estimated; peak HBM,
    per-core HBM and headroom; neuron-monitor's per-core maximum over the `transformer` samples;
  - the step table: `bf16_pre`, `bf16`, `bf16_tp4`, `bf16_9`, `mx_9`, `mx`, the bisection arms
    if they ran, `FINAL` and CP2, with medians and ratios;
  - quality vs `bf16` beside the floor and Q's thresholds;
  - determinism (the md5 pairs).

  Then README's FP8 note gains HunyuanVideo-1.5 (memory: the per-rank numbers, the step ratio
  as measured), and `docs/cli-staged-commands.md` gains the native `--backend neuron --quant mxfp8`
  command for it and its name in the `--quant` model list. Commit
  `docs: HunyuanVideo-1.5 in MXFP8 on Trainium3 — per-rank DiT <a> -> <b> GiB resident at tp2 x CFG2, peak HBM, step <r>x, <psnr> dB from bf16 (noise floor <fp> dB), CP2`;
  push; update the status page.

## After phase F

G (the CPU fp32 arbiter for Qwen-Image, FLUX, LTX-2, Wan 2.1/2.2, HV1, HV1.5 at the short
shapes, 50 steps, seed 42), as the phase-A/B plan's roadmap says.

### Task F7: revert D4 (phase D's keep rule, measured 2026-11-08)

Phase D's rule ("After phase D" in `docs/plans/2026-11-07-mxfp8-proj-out-rope.md`) keeps D4 (`71f3986` + `3e9e6c6`:
HunyuanVideo-1.5's `_joint_qkv` rotates q/k with the neuron `apply_rotary_emb_fp32` instead of diffusers'
`apply_rotary_emb(..., sequence_dim=1)`) only if bf16 is at most 1.01x `bf16_pre` and bit-identical or above the floor.
At 121 frames bf16 is 10,236 vs 9,924 ms (1.031x), not bit-identical, 15.44 dB < the 15.90 floor: the rule fails.
The MXFP8 default does not need D4 either (`fp8-bench/hv15/nod4`, a scratch of `8514606` with D4 undone):
121 frames 10,032 ms without D4 vs 10,314 with it (2.7% faster without); 9 frames 518 vs 511 ms, and bf16 at 9 frames
is bit-identical either way (`9ab6e81c`). So D4 goes.

**Files:** `difflet/backends/neuron/hunyuan_video15/transformer.py` (`_joint_qkv` back to
`from diffusers.models.embeddings import apply_rotary_emb` and `apply_rotary_emb(q|k, image_rotary_emb, sequence_dim=1)`,
its docstring and `_JointUlyssesProcessor`'s docstring as before D4), `difflet/backends/neuron/ops_impl/embeddings.py`
(`apply_rotary_emb_fp32`'s docstring no longer names HunyuanVideo-1.5 as a caller; the op itself stays for FLUX),
`tests/unit/backends/test_neuron_hv15_dit.py` (D4's spy on the neuron op goes; keep a test that `_joint_qkv` calls
diffusers' `apply_rotary_emb` with `sequence_dim=1` for q and k and stays bit-identical to diffusers' processor).
The exact code after the revert must equal the scratch's `_joint_qkv` (`/home/ubuntu/fp8-bench/hv15/nod4/src`), so the
`nod4` device runs are the final tree's.

- [x] **Step 1:** update the tests (fail against D4), **Step 2:** revert, **Step 3:** run
  `pytest tests/unit/backends/test_neuron_hv15_dit.py tests/unit/backends/test_neuron_hv15_fp8.py tests/unit/backends/test_neuron_hv15_pipeline.py tests/unit/backends/test_neuron_rotary.py tests/unit/backends/test_neuron_flux_fp8.py tests/unit/quant tests/unit/cli -q`: pass.
- [x] **Step 4: Commit** `revert(hv15): D4 — HunyuanVideo-1.5 rotates q/k with diffusers' apply_rotary_emb again; the neuron op made bf16 1.031x and the MXFP8 default 1.028x slower at 480x848x121 (10,236 vs 9,924 and 10,314 vs 10,032 ms), phase D's keep rule`.
- [x] **Step 5 (device, not the implementer's):** a second 9-frame MXFP8 generate from the final tree: bit-identical to `nod4/mx_gena.mp4` (`7a9ddae7`). Done on host A: `hv15/f7/` from `0f75d11` reproduced `7a9ddae7` and latents `aaadc89f` (evidence 9b, `f7/md5.txt`).

### Task F8: HunyuanVideo-1.5 has no MXFP8 default (user decision, 2026-11-09)

**Decision (user, 2026-11-09):** HunyuanVideo-1.5 does not quantize by default. This supersedes
F6 Step 7's option A (G1 + G3 as the default). Grounds (evidence sections 9 and 10): at 480x848x121
the G1 + G3 set gains neither speed (10,032 vs 9,924 ms, 1.011x) nor peak HBM (the same peak; only
the resident DiT drops, 7.80 -> 6.97 GiB per rank), and against diffusers fp32 on the CPU it is
7.47 dB further than bf16 (the image MLP alone 4.69 dB further), the largest drop of the seven
models. Issue [#145](https://github.com/ai-decentralized/Difflet/issues/145) is closed by this task.

Intended behaviour: `difflet compile|generate hunyuan-video-1.5 --quant mxfp8` with no
`DIFFLET_QUANT_TARGETS` is refused before anything is written or launched, through the driver's
`refuse` (exit 2), naming the reason (no default set; quality, evidence section 10) and the override.
With `DIFFLET_QUANT_TARGETS` set, the path works as today: the table and the whole F1-F5 wiring
stay, so the measured groups remain reachable for experiments. bf16 runs are unchanged.

**Files:** `difflet/quant/targets.py` (HunyuanVideo-1.5 has no default set; the G1-G4 patterns stay
as a named override table; the comment above them records this decision and supersedes "option A"),
`difflet/cli/orchestrators/hunyuan_video_15.py` (the refusal, before the download/compile checks),
`tests/unit/cli/test_orchestrator_hv15_neuron.py` and `tests/unit/quant/test_targets_hv15.py`
(the refusal on `compile` and `generate`, the message names `DIFFLET_QUANT_TARGETS`; with the env set
the existing tests still pass), `README.md` (the "user's decision, pending" sentence becomes the
decision; HunyuanVideo-1.5 leaves the `--quant` support list, with the override noted),
`docs/cli-staged-commands.md` (the staged `--quant` command for HunyuanVideo-1.5 goes or carries the
override), `docs/verification/2026-11-07-mxfp8-video-models.md` (10e and 11: the decision recorded;
the numbers stay as measured).

- [x] **Step 1:** tests for the refusal (fail against the current tree).
- [x] **Step 2:** targets and driver change; **Step 3:** docs as above.
- [x] **Step 4:** run `pytest tests/unit/quant tests/unit/cli tests/unit/backends/test_neuron_hv15_fp8.py -q`: pass.
- [x] **Step 5: Commit** `feat(hv15): no MXFP8 default — --quant mxfp8 without DIFFLET_QUANT_TARGETS is refused (fp32 reference: 7.5 dB further than bf16; 1.011x step and the same peak HBM at 121 frames); user decision 2026-11-09, issue #145`.
- [x] **Step 6:** close issue #145 with the decision and a link to evidence 10e.
