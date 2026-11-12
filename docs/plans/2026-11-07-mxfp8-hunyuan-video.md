# MXFP8 on HunyuanVideo 1.0 — Implementation Plan (phase E)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `--quant mxfp8` for HunyuanVideo 1.0 (`hunyuanvideo-community/HunyuanVideo`, a bf16 hub
checkpoint) on the native neuron backend, through phase A's shared layer. Device evidence at the
default 320x512x61 and at 720x1280x129, with the default layer set picked by device bisection.

**Architecture:** this is the FLUX mechanism. HV1's dual- and single-stream DiT keeps diffusers'
module names, which are also the checkpoint keys, apart from `proj_out_attn.bias`. The pieces are
a target table, the base-class attributes on the transformer application, `resolve_quant` in the
pipeline application and `prepare_quant` in the driver. Four things are HV1's own:
1. `NeuronHunyuanVideoTransformerApplication.__init__` sends `**overrides` to
   `NeuronHunyuanVideoConfig`. If `quant=` were passed today it would end up in the config and
   fail with a TypeError: from the dataclass when no config is given, and as "takes either config
   or config overrides" when one is. So `quant` becomes a named keyword that the class passes on
   to the base.
2. The single-stream `proj_out` is one checkpoint tensor that the neuron module reads as two
   column ranges (`proj_out_attn` / `proj_out_mlp`). Today's writer quantizes a key as a whole.
   So in Tasks E1–E5 `proj_out` stays bf16, and a spec that names it is refused by name. Once
   phase D's split writer lands (its plan, `docs/plans/2026-11-07-mxfp8-proj-out-rope.md`, D1 +
   D2), Task E6 adds the two halves to the table and removes the refusal. The table therefore
   depends on D in exactly one task, and E runs in either order relative to D.
3. The hub checkpoint is already bf16. The quantized directory is written straight from
   `<snapshot>/transformer`, with no bf16 copy.
4. The driver gets the stage's per-step timing (`timing=True`), which the device task reads.

**Tech Stack:** as phase B. HV1 facts, surveyed 2026-11-07 from
`difflet/models/hunyuan_video/modeling_hunyuan_video.py` and
`difflet/backends/neuron/hunyuan_video/transformer.py`:
- **Size and layout:** D=3072 (24 heads x 128), MLP 12288. 20 dual-stream blocks and 40
  single-stream blocks. 12,821,012,544 parameters, 25.6 GB in bf16.
- **Dual-stream blocks:** `attn.to_q/k/v` (image stream) and `attn.add_q/k/v_proj` (text stream)
  are Column layers with `gather_output=False`. `attn.to_out.0` and `attn.to_add_out` are Row
  layers. `ff` / `ff_context`: `net.0.proj` is Column and `net.2` is Row. Under SP the image
  `ff.net.2` and `to_out.0` use `reduce_output=False`, corrected by `_sp_unbias`.
- **Single-stream blocks:** `attn.to_q/k/v` run on the joined image+text sequence. `proj_mlp` is
  Column. `proj_out_attn` / `proj_out_mlp` are Row layers that read column ranges of the
  checkpoint's `proj_out` (`checkpoint_slices`).
- **Modulation:** `norm1.linear`, `norm1_context.linear` and `norm.linear` are Column layers with
  `gather_output=True` (`shard_modulation_linears`).
- **Attention:** it calls each projection directly, so there is no fusion. q/k go through RMSNorm,
  then `difflet.ops.apply_rotary_emb`, which is the backend pair-swap op since `fcb0dca`.
- **Token counts:** 10,240 image + 256 text tokens at 320x512x61, and 118,800 + 256 at
  720x1280x129.
- **Targets:** the default table holds 360 linears, 6.79 B of the 12.82 B parameters (53%). The
  quantized directory should be about 19.1 GB, 74% of bf16.

**Spec:** roadmap entry E of `docs/plans/2026-11-07-mxfp8-video-models.md`. Also the device facts
in `docs/verification/2026-11-07-mxfp8-video-models.md`: the LTX-2 sections 1-3e, and the Wan
section that C6 step 7 writes (`difflet/quant/targets.py:166` already cites it as section 4; on
2026-11-07 the doc ends at 3e). The decisions of this conversation:
- no fusion;
- the default set is decided by device bisection (E5 step 5's rule, which is D5's);
- `proj_out` waits for phase D;
- HV1.5 is phase F, memory-only.

Task E6 relies on the D1/D2 interfaces of the phase-D plan,
`docs/plans/2026-11-07-mxfp8-proj-out-rope.md`, which is being drafted alongside this one.

The 720p history is in `docs/verification/hunyuan-video-neuron-evidence.md` ("720x1280x129 at
TP4: BLOCKED"). On trn2.3xlarge, neuronx-cc compiling the block graph was OOM-killed at 109.8 GiB
of the host's 124.8 GiB. No forward ever ran. This host has 2 TB of RAM.

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

- **Inherited:** everything in the phase-A/B plan's Global Constraints applies:
  - branch `feat/mxfp8-video-models`; push every commit; update the status page per task;
  - D43;
  - no fusion;
  - K % 128;
  - default command lines stay byte-identical;
  - 50 steps / seed 42 / LNC=2;
  - the CPU fp32 reference is phase G and is not planned here.
- **Unit tests:** run with
  `cd /home/ubuntu/Difflet/.claude/worktrees/mxfp8-video-models && DIFFLET_BACKEND=cpu PYTHONPATH=. /home/ubuntu/venvs/native-fp8/bin/python -m pytest <paths> -q -p no:cacheprovider`
  (written `pytest <paths>` below). No test runs the CLI's compile against a real snapshot.
  HV1's DiT runs only in gloo workers (`hv1_dit_worker`), following the convention of
  `tests/unit/backends/test_neuron_hv1_dit.py`, so the HV1 port is never imported in-process.
  HV1 is quantized straight from the hub snapshot, which becomes local at E5 step 1. The
  conftest guard of `33504a7` wraps only `ensure_bf16_copy`, so Task E4 extends it to
  `write_quantized_dir` before any driver test can reach the writer.
  `tests/unit/cli/test_orchestrator_hunyuan_neuron.py` stays torch-free at import: toys are
  imported inside fixtures, as `test_quant_cli.py`'s `tiny_dir` does.
- **Pattern spelling:** HV1 target patterns use diffusers' module names, which are the checkpoint
  keys. HV1's `checkpoint_key` only rewrites `proj_out_attn.bias`.
- **Quantization source:** the quantized directory's source is `<snapshot>/transformer`, which is
  bf16 on the hub. There is no `ensure_bf16_copy` and no `resolve_component_dir`, so the source
  fingerprint is the HF blobs' and carries no bf16 manifest.
- **`proj_out` and phase D:** until Task E6, the single blocks' `proj_out` is not in the table.
  A spec that matches any `single_transformer_blocks.N.proj_out` is refused by name, on the driver
  before the writer runs and in the application. The check is
  `difflet.quant.targets.fused_proj_out_targets`.
- **Order with phase D:**
  - E1–E5 need nothing from phase D.
  - E6 needs D1 and D2 landed. Check for `split_linears` in `difflet/quant/spec.py` and for
    `SPLITS_BY_MODEL` / `SINGLE_BLOCK_SPLIT_LINEARS` in `difflet/quant/targets.py`.
  - If D1 and D2 have landed by the time E4 is done, run E6 before E5, so the device campaign
    measures the final table once.
  - Otherwise run E6, then E6's device step, after D lands.
  - E6's halves are the first HV1 targets with a K remainder: `proj_out_attn` at tp4 has a
    per-rank K of 768 = 512 + 256 (`mx._tile_sizes`). No device run has checked a remainder of
    256, and Wan's 384 was caught only by quality, at cosine 0.92 (`c813a47`). So the first
    device run of the halves is preceded by E6 step 6a's one-core kernel probe, whether or not
    D5 has run FLUX's halves, which have the same 768.
  - D1 adds the same `TINY_FP8_HV1` and `write_tiny_hv1(root, *, config=None)` as E1. Whichever
    lands second keeps the first one's definition, which is identical. `TINY_FP8_HV1_TP2` (E1)
    is E's own.
- **`quant` keyword:** `quant` is a named keyword of
  `NeuronHunyuanVideoTransformerApplication.__init__` and is passed to
  `TorchNeuronApplicationBase`. It never travels in `**overrides`.
- **No tp refusal:** every default target has a per-rank K that is a multiple of 128 at tp 1, 2, 4
  and 8. Column targets have K 3072. The row-parallel MLP-downs have K 12288/tp, which is 6144,
  3072 and 1536.
- **Choosing the default set:** this is D5's leave-one-out rule (E5 step 5), with a quality gate
  scaled to the video noise floor. Arm Vi is the default minus group Gi. Two pair arms always
  run: V15 (both q/k/v groups that feed RMSNorm, the rotary op and the attention kernel, which
  was Wan's 5x case) and V24 (both 256-row text groups, LTX-2's small-stream case). They run even
  when the full set is faster, because a slow group can hide behind the MLP gains. V1..V6 run
  when the full set is not faster than bf16. No set is ever built as a union of groups that were
  only measured apart: custom-op interactions decided both FLUX's 1.04x and Wan's 5x. The
  default is a measured arm. At 720p attention dominates, so expect about 1.0x there; that
  shape is measured, not bisected.
- **Two generates per arm (determinism):** every arm runs gen50a and gen50b with equal md5s,
  except two:
  - `bf16_tp2_L`, the 720p noise floor: one generate, because a tp2 generate at L takes about
    1.5 h. Its role is the distance to bf16 tp4; determinism at that shape is `bf16_L`'s.
  - DP2: one batch, whose check is md5 equality with the single launch.
- **Comparisons:** compare runs with `scripts/compare_difflet_runs.py` on the `.mp4` and the
  latents. Call it through the native-fp8 python with absolute paths
  (`PYTHONPATH=$SRC $PY $SRC/scripts/compare_difflet_runs.py`, as D5 does), because the
  neuronx venv's `python` comes first on PATH. HV1's `.pt` fallback is `(1, 3, F, H, W)` in
  [-1, 1], and the script refuses that layout. Use revision
  `e8c2aaa66fe3742a32c11a6766aecbf07c56e773`, the one the HV1 toy and benchmark pin.
- **The device host:** there is one trn3 host, shared with C6/C7 (Wan) and D5 (FLUX). E5 starts
  only when:
  - C6 and C7 are finished, and with them the Wan evidence section (C6 step 7);
  - D5 is finished, or not started and held until E5 ends;
  - `pgrep -fa 'difflet.cli.main|torchrun'` prints nothing.

  E6's device step has the same precondition.
- **Text shared with phases D and F:**
  - `--quant` help (`difflet/cli/quant.py:41-46`). D2 owns the bf16 clause, E1/F1 own the model
    list and E6 owns one name in D2's clause. Each task edits only its own part, so the result
    does not depend on the order. With D2, E1, E6 and F1 all landed, the help ends:
    "... the output projections, modulation, embedders and the output layer stay bf16 (except
    the single-stream output projection of FLUX and HunyuanVideo, quantized as its attention
    and MLP halves; LTX-2 keeps its audio stream bf16). Qwen-Image, FLUX.1-dev, LTX-2, Wan 2.1,
    Wan 2.2-A14B (both experts), HunyuanVideo and HunyuanVideo-1.5 (a memory saving)."
  - Evidence sections: D5, E5/E6 and F each add one numbered section to
    `docs/verification/2026-11-07-mxfp8-video-models.md`. Each takes the next free number at
    the time it is written, after Wan's 4.

## Review Focus

1. **`quant=` must reach the base class.** With or without `config=`, it must not leak into
   `**overrides`. Today both calls raise a TypeError that names the config, not quant. Pinned in
   Task E2 by `test_quant_is_the_base_class_keyword_not_a_config_override`.
2. **A spec that names the fused single-block `proj_out` is refused early.** Expect this from a
   hand-written `DIFFLET_QUANT_TARGETS` variant before phase D; E5's arms are subsets of the table and never name it. Without the guard, the writer spends
   minutes writing the whole quantized directory, about 17 GB, with the fused tensor quantized
   whole. Then every rank refuses with a dead-pattern message from `quantize_module_`: the module
   has `proj_out_attn` / `proj_out_mlp`, not `proj_out`. That message names neither the cause nor
   phase D. With the guard it is
   refused on the driver before anything is written, and at the application. Pinned in Task E1
   (helper), Task E2 (application) and Task E4 (driver: exit 2, nothing under the cache). After
   Task E6, D's writer refuses the fused name as a dead target on the driver; E6's tests pin that.
3. **`--sp --quant` must give the `--quant` output.** Under SP, the dual blocks' `ff.net.2` and
   `to_out.0` run with `reduce_output=False`, and `_sp_unbias` subtracts `(tp-1) x bias`. The FP8
   row twin must return the same unreduced partial plus the full bias. Pinned in Task E2 by the
   tp2-sp gloo case against the tp2 weight-only reference.
4. **`generate` before `compile` with `--quant` is refused cleanly.** With no quantized directory
   yet, the driver must exit 2 before torchrun, name `difflet compile`, and write nothing. Pinned
   in Task E4 by `test_generate_before_compile_is_refused_naming_difflet_compile`.
5. **The flags and the spec must survive every launch path.** That means the stage line, a DP
   worker's line and the generate stage's cache key. A stage that drops `--quant` runs the bf16
   DiT against the fp8 key. The vae stage key and the bf16 lines must stay as they are. Pinned in
   Task E4 (`test_the_stage_and_dp_worker_lines_carry_quant_only_when_given`) and in
   `tests/unit/cli/test_quant_cli.py`'s HV1 twins.

---

### Task E1: the HunyuanVideo target table, its FP8 toy and the phase-D guard

**Files:**
- Modify: `difflet/quant/targets.py`:
  - `:1-54`: a HunyuanVideo paragraph before "Patterns are regexes";
  - after `:194`: the new constants and the helper;
  - `:196-207`: the registries;
  - `:232-250`: `__all__`.
- Modify: `difflet/cli/quant.py`:
  - `:26-32`: `QUANT_MODELS["hunyuanvideo-community/HunyuanVideo"] = "hunyuan_video"`;
  - `:41-46`: the help's model list only (Global Constraints, "Text shared with phases D and
    F"). Without F1 it ends "... Wan 2.1, Wan 2.2-A14B (both experts) and HunyuanVideo."; after
    F1 it ends "... Wan 2.1, Wan 2.2-A14B (both experts), HunyuanVideo and HunyuanVideo-1.5 (a
    memory saving)."
- Modify: `tests/unit/backends/_neuron_hv1_toy.py:30-44`: add `TINY_FP8_HV1` and
  `write_tiny_hv1(root, *, config=None)`. Skip these two if phase D1 already added them; its
  definition is identical. Add `TINY_FP8_HV1_TP2` in all cases.
- Modify: `difflet/cli/orchestrators/hunyuan_video.py:210-214`. `_torchrun_unsupported_options`
  appends `"--quant"` when `args.quant` is set, until Task E4 lands. This follows the B1/C1
  precedent: the pushed branch never has a window where `--quant` silently runs bf16.
- Modify: `tests/unit/cli/test_quant_cli.py:58-59`. The "not available for" probe moves from the
  HV1 id to `"MiniMaxAI/MiniMax-H3"`, which never gets a table.
- Test: `tests/unit/quant/test_targets_hunyuan_video.py` (new), plus one temporary row in
  `tests/unit/cli/test_orchestrator_hunyuan_neuron.py`.

**Interfaces:**
- Produces, with `_BLOCK` / `_SINGLE` as defined in `targets.py:60,93`. One pattern per bisection
  group, in this order (Task E5 uses the indices):
  ```python
  HUNYUAN_VIDEO_TARGETS = (
      rf"{_BLOCK}\.attn\.(to_q|to_k|to_v)",                   # G1 dual image q/k/v (RMSNorm + rotary)
      rf"{_BLOCK}\.attn\.(add_q_proj|add_k_proj|add_v_proj)",  # G2 dual text q/k/v (256 rows)
      rf"{_BLOCK}\.ff\.net\.(0\.proj|2)",                      # G3 dual image MLP
      rf"{_BLOCK}\.ff_context\.net\.(0\.proj|2)",              # G4 dual text MLP (256 rows)
      rf"{_SINGLE}\.attn\.(to_q|to_k|to_v)",                   # G5 single joint q/k/v
      rf"{_SINGLE}\.proj_mlp",                                 # G6 single joint MLP-up
  )
  ```
- `HUNYUAN_VIDEO_KEPT_BF16`:
  - `rf"{_BLOCK}\.attn\.(to_out\.0|to_add_out)"`
  - `rf"{_BLOCK}\.norm1(_context)?\.linear"`
  - `rf"{_SINGLE}\.norm\.linear"`
  - `rf"{_SINGLE}\.proj_out"`, commented "phase D: the writer must store its two column ranges"
  - `rf"{_SINGLE}\.proj_out_(attn|mlp)"`, the neuron module's names for it
  - `r"context_embedder\..*"`
  - `r"time_text_embed\..*"`
  - `r"norm_out\.linear"`
  - `r"proj_out"`
- `HUNYUAN_VIDEO_ROW_PARALLEL`:
  - `rf"{_BLOCK}\.attn\.(to_out\.0|to_add_out)"`, commented "not default targets; kept for
    overrides"
  - `rf"{_BLOCK}\.(ff|ff_context)\.net\.2"`
- `"hunyuan_video"` joins `TARGETS_BY_MODEL` and `ROW_PARALLEL_BY_MODEL`.
- The tuple is HV1's own even though it starts out equal to `FLUX_TARGETS`: bisection may drop
  groups for HV1 alone.
- `FUSED_PROJ_OUT_REASON: str = "is read as two column ranges (proj_out_attn / proj_out_mlp) by
  the neuron module, which the MXFP8 checkpoint writer cannot store yet (roadmap phase D); drop it
  from the targets"`.
- `fused_proj_out_targets(spec: QuantSpec, num_single_layers: int) -> list[str]` returns the
  names `f"single_transformer_blocks.{i}.proj_out"`, for `i` in `range(num_single_layers)`, that
  `spec.matches`, in block order. Its docstring says the same applies to FLUX (same names, same
  split) and that only HV1 calls it in this phase.
- `TINY_FP8_HV1 = {**TINY_HV1, "attention_head_dim": 32, "rope_axes_dim": (8, 12, 12)}`. This
  gives inner width 128 and MLP 512, so every target's K is 128 and `ff*.net.2`'s per-rank K is
  512/tp. diffusers builds this configuration (checked). It serves the name tests and D1.
- `TINY_FP8_HV1_TP2 = {**TINY_HV1, "attention_head_dim": 64, "rope_axes_dim": (16, 24, 24)}`
  gives inner width 256 and MLP 1024. diffusers builds it (checked 2026-11-07: 5.9 M
  parameters, `proj_out` `[256, 1280]`). It is the toy of every quantized E2/E3/E4 fixture, so
  that E6 can add the halves without changing a fixture: with `TINY_FP8_HV1`, the per-rank K of
  `proj_out_attn` at tp2 would be 64, which the writer refuses. Its comment says so. FLUX's
  `TINY_FP8_TP2` (D2) is the precedent. Per-rank K at tp2: column targets 256, `ff*.net.2` 512,
  `proj_out_attn` 128, `proj_out_mlp` 512.
- `write_tiny_hv1(root, *, config: dict | None = None) -> str` saves
  `HunyuanVideoTransformer3DModel(**(config or TINY_HV1))`. The default output is byte-identical.

- [x] **Step 1: Write the failing tests.** In `test_targets_hunyuan_video.py`, the module fixture
  is `model = diffusers.HunyuanVideoTransformer3DModel(**TINY_FP8_HV1)` and
  `_linears(model)` lists the `nn.Linear` names.

```python
PER_DUAL = {"attn.to_q", "attn.to_k", "attn.to_v", "attn.add_q_proj", "attn.add_k_proj",
            "attn.add_v_proj", "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2"}
PER_SINGLE = {"attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp"}

def test_default_targets_cover_the_block_qkv_and_mlps_only(model):
    spec = default_spec("hunyuan_video"); names = [n for n in _linears(model) if spec.matches(n)]
    assert {n.split(".", 2)[2] for n in names if n.startswith("transformer_blocks.")} == PER_DUAL
    assert {n.split(".", 2)[2] for n in names if n.startswith("single_")} == PER_SINGLE
    assert len(names) == 10 * TINY_FP8_HV1["num_layers"] + 4 * TINY_FP8_HV1["num_single_layers"]
    assert not any(n.startswith("context_embedder.") for n in names)   # the refiner's own ff/attn
    assert {n for n in _linears(model) if spec.is_row_parallel(n)} == {
        n for n in _linears(model) if n.startswith("transformer_blocks.")
        and n.endswith(("attn.to_out.0", "attn.to_add_out", "ff.net.2", "ff_context.net.2"))}

def test_targets_and_kept_bf16_cover_every_linear_once(model):
    for n in _linears(model):
        hits = [any(re.fullmatch(p, n) for p in t) for t in (HUNYUAN_VIDEO_TARGETS, HUNYUAN_VIDEO_KEPT_BF16)]
        assert hits.count(True) == 1, n

def test_every_default_target_has_an_mx_layout_at_tp_1_2_4_8():
    with accelerate.init_empty_weights():
        real = HunyuanVideoTransformer3DModel()   # diffusers' defaults are HunyuanVideo's own
    for tp in (1, 2, 4, 8):
        spec = default_spec("hunyuan_video", tp_degree=tp)
        targets = [(n, m) for n, m in real.named_modules()
                   if isinstance(m, torch.nn.Linear) and spec.matches(n)]
        assert len(targets) == 360
        assert all((m.in_features // spec.k_shards(n)) % LAYOUT_K == 0 for n, m in targets), tp

def test_fused_proj_out_targets_name_the_single_blocks_proj_out():
    assert fused_proj_out_targets(default_spec("hunyuan_video"), 40) == []
    every = QuantSpec(targets=(r"single_transformer_blocks\.\d+\.proj_out",))
    assert fused_proj_out_targets(every, 2) == ["single_transformer_blocks.0.proj_out",
                                                "single_transformer_blocks.1.proj_out"]
    one = QuantSpec(targets=(r"single_transformer_blocks\.1\.proj_out",))
    assert fused_proj_out_targets(one, 2) == ["single_transformer_blocks.1.proj_out"]
    # module spelling: no checkpoint key, so the writer's dead-pattern refusal owns it
    split = QuantSpec(targets=(r"single_transformer_blocks\.\d+\.proj_out_attn",))
    assert fused_proj_out_targets(split, 2) == []

def test_the_cli_id_resolves_to_the_table_at_the_launchs_tp(monkeypatch):
    monkeypatch.delenv("DIFFLET_QUANT_TARGETS", raising=False)
    assert QUANT_MODELS[HV1] == "hunyuan_video"
    assert TARGETS_BY_MODEL["hunyuan_video"] is HUNYUAN_VIDEO_TARGETS
    assert ROW_PARALLEL_BY_MODEL["hunyuan_video"] is HUNYUAN_VIDEO_ROW_PARALLEL
    args = argparse.Namespace(model_id=HV1, quant="mxfp8", quant_compute=None, tp_degree=None)
    assert quant_spec(args) == default_spec("hunyuan_video", compute="nki", tp_degree=4)
    assert quant_spec(argparse.Namespace(**{**vars(args), "tp_degree": 2})).tp_degree == 2

def test_the_fp8_toy_has_the_mx_geometry(tmp_path):
    fp8 = json.loads((Path(write_tiny_hv1(tmp_path / "a", config=TINY_FP8_HV1)) / "config.json").read_text())
    assert (fp8["attention_head_dim"], fp8["rope_axes_dim"]) == (32, [8, 12, 12])
    tp2 = json.loads((Path(write_tiny_hv1(tmp_path / "c", config=TINY_FP8_HV1_TP2)) / "config.json").read_text())
    assert (tp2["attention_head_dim"], tp2["rope_axes_dim"]) == (64, [16, 24, 24])
    inner = tp2["num_attention_heads"] * tp2["attention_head_dim"]          # 256: proj_out_attn K
    assert all(k % (2 * LAYOUT_K) == 0 for k in (inner, 4 * inner))         # per-rank K at tp2
    assert json.loads((Path(write_tiny_hv1(tmp_path / "b")) / "config.json").read_text())["attention_head_dim"] == 16
```

  Also add, in `tests/unit/cli/test_orchestrator_hunyuan_neuron.py`,
  `test_quant_is_refused_until_the_driver_carries_it(launches, capsys, mode)`. It is parametrized
  over `generate` and `compile`, calls `_args(quant="mxfp8")`, and expects `SystemExit` with code
  2, `stages == []` and `"does not support --quant"` in stderr.
- [x] **Step 2: Run** `pytest tests/unit/quant/test_targets_hunyuan_video.py -v`. Expected: FAIL
  on import (`HUNYUAN_VIDEO_TARGETS` unknown).
- [x] **Step 3: Implement** the constants, the helper, the docstring paragraph, the registry
  entries, `QUANT_MODELS`, the help text, the toy, the temporary driver refusal and the moved
  probe. The docstring paragraph covers:
  - the dims and the stream each target sees;
  - why `proj_out` waits for phase D;
  - the row-parallel K per rank at tp 2/4/8.
- [x] **Step 4: Run** `pytest tests/unit/quant tests/unit/cli/test_quant_cli.py tests/unit/cli/test_orchestrator_hunyuan_neuron.py tests/unit/backends/test_neuron_hv1_dit.py -q`.
  Expected: PASS. The HV1 toy's default output is unchanged.
- [x] **Step 5: Commit** `feat(quant): the HunyuanVideo MXFP8 target table (dual-stream q/k/v, add_* and both MLPs; single-stream q/k/v and proj_mlp; output projections, modulation, token refiner and the fused single-block proj_out bf16 until phase D)`, then push.

**Done** (`20abd63`, review fix `74104aa`).

### Task E2: the HunyuanVideo transformer application

**Files:**
- Modify: `difflet/backends/neuron/hunyuan_video/transformer.py`:
  - `:1-41`: a docstring paragraph on MXFP8;
  - `:43-62`: imports (`QuantSpec`, `FUSED_PROJ_OUT_REASON`, `fused_proj_out_targets`);
  - `:105-169`: the class.
- Modify: `tests/unit/backends/_neuron_workers.py:1577-1609` (`hv1_dit_worker`).
- Test: `tests/unit/backends/test_neuron_hv1_fp8.py` (new).

**Interfaces:**
- Consumes from Task E1: `TINY_FP8_HV1_TP2`, `write_tiny_hv1(root, *, config=)`,
  `default_spec("hunyuan_video", ...)`, `fused_proj_out_targets`, `FUSED_PROJ_OUT_REASON`.
- `NeuronHunyuanVideoTransformerApplication` class attributes: `quant_model = "hunyuan_video"`
  and `quant_label = "HunyuanVideo"`. `quant_fuse_qkv` stays `False`.
- `__init__(self, *, model_path, config=None, parallel=None, dtype=torch.bfloat16, exec_mode=None, device="neuron", quant: QuantSpec | dict | None = None, **overrides)`
  passes `quant=quant` to `super().__init__`. Once `config.validate()` has run:
  `fused = fused_proj_out_targets(self.quant, config.num_single_layers)` when `self.quant` is
  set, and a non-empty result raises
  `ValueError(f"HunyuanVideo: {fused[0]} {FUSED_PROJ_OUT_REASON}")`.
- `hv1_dit_worker` gets two new arguments,
  `(rank, world_size, model_dir, parallel, exec_mode, valid_text=16, overrides=None, quant: dict | None = None, quantized_dir: str | None = None)`.
  - The config is still read from `model_dir` (the source `transformer`).
  - The application gets `model_path=quantized_dir or model_dir` and `quant=quant`.
  - The result gains `"quantized_linears": list(app.quantized_linears)` and
    `"fp8": {name: type(m).__name__ for name, m in app.module.named_modules() if isinstance(m, _FP8LinearBase)}`.
  - Existing positional callers are unchanged.

- [x] **Step 1: Write the failing tests.** The module fixture `tiny` does this:
  - writes `write_tiny_hv1(root, config=TINY_FP8_HV1_TP2)`. This is the tp2 geometry, so the
    same fixture still works once E6 makes the halves row-parallel targets;
  - builds, for tp 1 and 2, `specs[tp] = default_spec("hunyuan_video", compute="dequant", tp_degree=tp)`;
  - writes `quantized[tp] = prepare_quantized_checkpoint(transformer_dir, specs[tp], create=True, cache_root=root / "cache")`;
  - computes `reference[tp]`: diffusers' model with every target weight replaced by
    `mx.fake_quantize_mx_sharded(w, specs[tp].k_shards(name))`, run on `tiny_inputs(9)`. That is
    what dequant mode computes; a row-parallel target is quantized per rank K range.

  `_run(tiny, parallel, world, tp)` is
  `run_ranks(hv1_dit_worker, tiny.transformer_dir, parallel, "compile", 9, None, tiny.specs[tp].to_dict(), str(tiny.quantized[tp]), world_size=world)`.

```python
TINY_TARGETS_HV1 = 10 * TINY_FP8_HV1_TP2["num_layers"] + 4 * TINY_FP8_HV1_TP2["num_single_layers"]  # 18

def test_the_quantized_directory_holds_the_default_targets_only(tiny):
    for tp in (1, 2):
        manifest = read_manifest(tiny.quantized[tp]); names = set(manifest["quantized_linears"])
        assert manifest["tensors"]["quantized"] == TINY_TARGETS_HV1 == len(names)
        assert not any(n.endswith(("to_out.0", "to_add_out", "proj_out", ".linear")) for n in names)

def test_quant_is_the_base_class_keyword_not_a_config_override(tiny):          # Review Focus 1
    spec, common = tiny.specs[1], dict(model_path=tiny.quantized[1], dtype=torch.float32,
                                       exec_mode="compile", device="cpu")
    config = NeuronHunyuanVideoConfig.from_pretrained(tiny.transformer_dir, **TINY_SHAPE)
    assert NeuronHunyuanVideoTransformerApplication(config=config, quant=spec, **common).quant == spec
    app = NeuronHunyuanVideoTransformerApplication(quant=spec.to_dict(), **TINY_SHAPE, **common)
    assert app.quant == spec and app.config.num_frames == TINY_SHAPE["num_frames"]
    assert not hasattr(app.config, "quant")

def test_application_quantizes_the_targets_and_matches_the_weight_only_reference(tiny):
    [r] = _run(tiny, {"tp_degree": 1}, 1, 1)
    assert r["report"] == {"missing": [], "unexpected": []} and r["unwarmed_shapes"] == []
    assert sorted(r["fp8"]) == sorted(r["quantized_linears"]) and len(r["fp8"]) == TINY_TARGETS_HV1
    assert r["fp8"]["transformer_blocks.0.attn.add_v_proj"] == "FP8ColumnParallelLinear"
    assert r["fp8"]["transformer_blocks.0.ff_context.net.2"] == "FP8RowParallelLinear"
    assert r["fp8"]["single_transformer_blocks.1.proj_mlp"] == "FP8ColumnParallelLinear"
    assert not any(n.endswith(("to_out.0", "to_add_out", "proj_out_attn", "proj_out_mlp", ".linear"))
                   or n.startswith("context_embedder.") for n in r["fp8"])
    torch.testing.assert_close(torch.tensor(r["out"]), tiny.reference[1], atol=1e-4, rtol=1e-4)

@pytest.mark.parametrize(("parallel", "world", "tp"), [
    ({"tp_degree": 2}, 2, 2),
    ({"tp_degree": 2, "sp_enabled": True}, 2, 2),                              # Review Focus 3
    ({"tp_degree": 1, "cp_degree": 2, "cp_mode": "gather_kv"}, 2, 1),
], ids=["tp2", "tp2-sp", "cp2-gather_kv"])
def test_parallel_layouts_run_the_fp8_twins_as_tp_does(tiny, parallel, world, tp):
    for r in _run(tiny, parallel, world, tp):
        assert r["report"] == {"missing": [], "unexpected": []}
        assert len(r["quantized_linears"]) == TINY_TARGETS_HV1
        torch.testing.assert_close(torch.tensor(r["out"]), tiny.reference[tp], atol=1e-4, rtol=1e-4)

def test_eager_and_a_spec_for_another_tp_are_refused(tiny): ...
    # exec_mode="eager" -> ValueError match "D43"; quant=tiny.specs[2] at tp1 ->
    # ValueError match r"the HunyuanVideo FP8 DiT spec .*tp_degree=2 .*tp_degree=1"

def test_a_target_on_the_fused_single_block_proj_out_is_refused_until_phase_d(tiny):  # RF 2
    spec = QuantSpec(targets=(*tiny.specs[1].targets, r"single_transformer_blocks\.1\.proj_out"),
                     compute="dequant", row_parallel=tiny.specs[1].row_parallel)
    with pytest.raises(ValueError, match=r"single_transformer_blocks\.1\.proj_out is read as two "
                                         r"column ranges \(proj_out_attn / proj_out_mlp\).*phase D"):
        NeuronHunyuanVideoTransformerApplication(model_path=tiny.transformer_dir, quant=spec,
            dtype=torch.float32, exec_mode="compile", device="cpu", **TINY_SHAPE)
```

- [x] **Step 2: Run** `pytest tests/unit/backends/test_neuron_hv1_fp8.py -v`. Expected: FAIL.
  `quant` reaches the config, so construction raises TypeError (with `config=`: "either config or
  config overrides ... ['quant']"), and the worker rejects the extra arguments.
- [x] **Step 3: Implement** the keyword, the two attributes, the `proj_out` refusal, the
  docstrings and the worker change.
- [x] **Step 4: Run** `pytest tests/unit/backends/test_neuron_hv1_fp8.py tests/unit/backends/test_neuron_hv1_dit.py -q`.
  Expected: PASS, with the bf16 DiT tests unchanged.
- [x] **Step 5: Commit** `feat(hunyuan-video): --quant mxfp8 on the DiT application (quant a keyword of its own, not a config override; a target on the fused single-block proj_out refused until phase D)`, then push.

**Done** (`29345ab`).

### Task E3: the HunyuanVideo pipeline application

**Files:**
- Modify: `difflet/models/hunyuan_video/neuron_application.py`:
  - `:1-52`: a paragraph modeled on LTX-2's `quant` paragraph;
  - `:56-75`: imports `resolve_quant`;
  - `:81-102`: `_KWARG_DEFAULTS` adds `"quant": None, "quant_checkpoint": None`, with LTX-2's
    comment;
  - `:298-311`: resolve the spec and pass it on.
- Test: `tests/unit/backends/test_neuron_hv1_fp8.py`.

**Interfaces:**
- Once the config is built, the constructor runs
  `self.quant, dit_path = resolve_quant("hunyuan_video", options["quant"], options["quant_checkpoint"], transformer_path, tp_degree=int(parallel.tp_degree))`.
- The transformer is then built with `model_path=dit_path, ..., quant=self.quant`.
- The pipeline application has no device-component HBM check (text encoder on the host only), so
  it needs no `quantized_dit` attribute. The base's meta-module check counts fp8 at 1 byte.

- [x] **Step 1: Write the failing tests.** They construct the application in-process; nothing
  imports the HV1 port. `tiny.root` holds only `transformer/` plus the cache, and construction
  reads nothing else, so the tokenizers are not needed.
  `_pipeline_app(tiny, **kw)` is
  `TorchNeuronHunyuanVideoApplication(model_path=str(tiny.root), parallel=kw.pop("parallel", DiffletParallelConfig()), dtype=torch.float32, shape=dict(height=64, width=64, num_frames=17), text_seq_len=16, device="cpu", exec_mode="compile", **kw)`.

```python
def test_pipeline_application_resolves_the_quantized_directory(tiny, monkeypatch):
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tiny.root / "cache"))
    app = _pipeline_app(tiny, quant="mxfp8")         # nki compute; dequant shares the directory
    assert app.quant == default_spec("hunyuan_video", tp_degree=1)
    assert app.transformer.quant is app.quant and app.transformer.model_path == str(tiny.quantized[1])
    tp2 = _pipeline_app(tiny, quant="mxfp8", parallel=DiffletParallelConfig(tp_degree=2))
    assert tp2.quant.tp_degree == 2 and tp2.transformer.model_path == str(tiny.quantized[2])
    plain = _pipeline_app(tiny)
    assert plain.quant is None and plain.transformer.quant is None
    assert plain.transformer.model_path == str(tiny.transformer_dir)
    explicit = _pipeline_app(tiny, quant=tiny.specs[1].to_dict(), quant_checkpoint=str(tiny.quantized[1]))
    assert explicit.transformer.model_path == str(tiny.quantized[1])

def test_pipeline_application_option_refusals(tiny, monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="quant_checkpoint needs quant"):
        _pipeline_app(tiny, quant_checkpoint=str(tiny.quantized[1]))
    with pytest.raises(ValueError, match="quant must be"):
        _pipeline_app(tiny, quant="fp8")
    with pytest.raises(ValueError, match="built with another quantization spec"):
        _pipeline_app(tiny, quant=tiny.specs[1], quant_checkpoint=str(tiny.quantized[2]))
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path / "empty"))
    with pytest.raises(ValueError, match="difflet compile"):
        _pipeline_app(tiny, quant="mxfp8")
```

- [x] **Step 2: Run** `pytest tests/unit/backends/test_neuron_hv1_fp8.py -k pipeline -v`.
  Expected: FAIL with `TypeError: ... unexpected keyword arguments: quant`.
- [x] **Step 3: Implement.**
- [x] **Step 4: Run** `pytest tests/unit/backends/test_neuron_hv1_fp8.py tests/unit/backends/test_neuron_hv1_app.py -q`.
  Expected: PASS. The app tests skip without the HV1 tokenizers, as before.
- [x] **Step 5: Commit** `feat(hunyuan-video): the pipeline application resolves the quantized DiT directory from the bf16 hub checkpoint`, then push.

**Done** (`ee618f0`).

### Task E4: the HunyuanVideo orchestrator

**Files:**
- Modify: `difflet/cli/orchestrators/hunyuan_video.py`:
  - `:1-32`: one docstring sentence on `--quant`, as LTX-2's;
  - `:34-65`: imports: `functools`, `json`, `_local_snapshot`, `refuse`,
    `prepare_quant, quant_cli_args, quant_spec`, `quant_cache_fields`,
    `FUSED_PROJ_OUT_REASON, fused_proj_out_targets`;
  - `:191-207`: `_torchrun_application_kwargs`;
  - `:210-214`: drop Task E1's `--quant` row;
  - `:417-430`: `_check_dit_fits` and `_prepare_torchrun`, plus the new methods;
  - `:432-470`: the two callers;
  - `:777-788`: `timing=True`;
  - `:876-910`: the generate key;
  - `:928-980`: `_shared_cli_args`.
- Modify: `tests/unit/conftest.py`:
  - `:72-76`: the hub-cache resolution becomes a module function `_hub_cache() -> Path`, which
    the bf16 guard calls;
  - after `:85`: a second autouse fixture, `_no_quantized_dir_of_a_real_snapshot`.
- Test: `tests/unit/quant/test_checkpoint.py` (the guard's test, after `:318`),
  `tests/unit/cli/test_orchestrator_hunyuan_neuron.py` (drop Task E1's temporary test, add the
  tests below), and `tests/unit/cli/test_quant_cli.py` (HV1 twins of the Wan tests at
  `:371-430`, with `HV1 = "hunyuanvideo-community/HunyuanVideo"`).

**Interfaces:**
- `_torchrun_application_kwargs(args)` adds `kwargs["quant"] = quant_spec(args).to_dict()` when
  the spec is not None. It never adds `quant_checkpoint`.
- New class attribute `_quantized_dit: str | None = None`.
- `_prepare_quant(self, *, create: bool) -> None`. It does nothing without `--quant`. Otherwise:
  1. `model_dir = _local_snapshot(self._model_dir)`. If it is None, call
     ``refuse(f"--quant: the {_HF_MODEL_ID} weights are not local; run `difflet download --model-id {_HF_MODEL_ID}` first")``.
  2. Read `num_single_layers` from `<model_dir>/transformer/config.json`, or use 40 when the file
     is absent.
  3. Compute `fused_proj_out_targets(quant_spec(self.args), n)`. If it is non-empty, call
     `refuse(f"--quant: {fused[0]} {FUSED_PROJ_OUT_REASON}")`. This happens before the writer
     runs.
  4. Guard the result as LTX-2's driver does (`difflet/cli/orchestrators/ltx_2.py:459-461`).
     `prepare_quant` returns None, and prints no line, when `<model_dir>/transformer` is
     missing. The ranks then report the partial download.
     ```python
     prepared = prepare_quant(self.args, {"transformer": str(Path(model_dir) / "transformer")},
                              create=create, tag="hunyuan_video", fail=refuse)
     if prepared is not None:
         self._quantized_dit = prepared["transformer"]
     ```
- `_fits_as_stored(self, fits)` is LTX-2's (`functools.partial(fits, dit_dir=...)`).
- `_check_dit_fits` calls
  `refuse_dit_overflow("HunyuanVideo", self._model_dir, self._fits_as_stored(dit_fits), self.args.tp_degree or 4, stored="bf16" if self._quantized_dit is None else "MXFP8")`.
- `_prepare_torchrun(self, *, generating: bool)` runs, in order:
  `_reject_unsupported_torchrun_options()`, then `root_compile_cache_at_cache_dir(...)`, then
  `_prepare_quant(create=not generating)`, then `_check_dit_fits()`. `_compile_torchrun` passes
  `generating=False` and `_generate_torchrun` passes `generating=True`.
- `run_generate_stage(..., timing=True, ...)` in `_stage_generate_neuron`. HV1's orchestrator
  calls `scheduler.step` once per step, so this prints the
  `[hunyuan_video] request: ... step median` line.
- `_torchrun_stage_cache_inputs("generate", args)` gains `**quant_cache_fields(quant_spec(args))`
  inside `torchrun_stage_keys({...})`. The vae key and the AoT `_stage_cache_inputs` are
  unchanged.
- `_shared_cli_args` appends `parts += quant_cli_args(a)` after the TeaCache flags, only when the
  flags were given.

- [x] **Step 1: the conftest guard (its own commit, first).**
  - **Test.** Add to `tests/unit/quant/test_checkpoint.py`. It goes through the driver path:
    `prepare_quantized_checkpoint` -> `ensure_quantized_dir`, which fingerprints the source
    from headers and then calls the module-global `write_quantized_dir`. A real snapshot's
    files exist, so the fingerprint passes and the writer is reached. The test therefore needs
    a fake hub that holds safetensors:
    ```python
    def test_unit_tests_never_quantize_a_real_snapshot(tiny, tmp_path, monkeypatch):
        from huggingface_hub import constants
        monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
        src = (tmp_path / "hub" / "models--hunyuanvideo-community--HunyuanVideo" / "snapshots"
               / ("0" * 40) / "transformer")
        shutil.copytree(tiny["transformer_dir"], src)
        with pytest.raises(AssertionError, match="quantized directory of a real snapshot"):
            prepare_quantized_checkpoint(src, tiny["spec"], create=True, cache_root=tmp_path / "cache")
        assert not (tmp_path / "cache" / "quantized").exists()
    ```
  - **Run** `pytest tests/unit/quant/test_checkpoint.py -k real_snapshot -v`. Expected: FAIL
    with "DID NOT RAISE": the unguarded writer quantizes the toy.
  - **Implement.**
    - `_hub_cache() -> Path` reads `huggingface_hub.constants.HF_HUB_CACHE` when called (falling
      back to `~/.cache/huggingface/hub` on ImportError) and returns it expanded and resolved.
      The bf16 guard calls it once at fixture time, as it does today.
    - The new fixture calls `_hub_cache()` per call, so the test above can point it at a fake
      hub:
    ```python
    @pytest.fixture(autouse=True)
    def _no_quantized_dir_of_a_real_snapshot(monkeypatch):
        """``write_quantized_dir`` (every driver's compile with --quant) is refused for a source
        under the Hugging Face hub cache, as ``ensure_bf16_copy`` is: HunyuanVideo, FLUX, LTX-2
        and Qwen-Image are quantized straight from the hub snapshot (about 19 GB for
        HunyuanVideo), so a driver test whose resolver patch missed would write one per call."""
        from pathlib import Path

        from difflet.quant import checkpoint

        real = checkpoint.write_quantized_dir

        def guarded(model_dir, out_dir, spec, **kwargs):
            if Path(model_dir).resolve().is_relative_to(_hub_cache()):
                raise AssertionError(
                    f"a unit test asked for a quantized directory of a real snapshot: {model_dir}")
            return real(model_dir, out_dir, spec, **kwargs)

        monkeypatch.setattr(checkpoint, "write_quantized_dir", guarded)
    ```
    A test module that imports `write_quantized_dir` by name, as `test_checkpoint.py:19-31`
    does, keeps the unguarded function for its own toy calls. The drivers reach the writer only
    through `ensure_quantized_dir`, which looks up the module global.
  - **Run** `pytest tests/unit/quant tests/unit/cli tests/unit/backends/test_neuron_qwen_fp8.py tests/unit/backends/test_neuron_ltx2_fp8.py -q`.
    Expected: PASS. Every existing writer test writes from a toy under tmp_path.
  - **Commit** `test: unit tests never write a quantized directory from a real snapshot (HunyuanVideo is quantized straight from the hub snapshot; the conftest guard now covers write_quantized_dir as it covers ensure_bf16_copy)`, then push.
- [x] **Step 2: Write the failing driver tests.** In `test_orchestrator_hunyuan_neuron.py`:
  - A module fixture `tiny_fp8_snapshot(tmp_path_factory)`. It calls
    `pytest.importorskip("diffusers")`, imports
    `from tests.unit.backends._neuron_hv1_toy import TINY_FP8_HV1_TP2, write_tiny_hv1` inside
    the fixture, so the module stays torch-free at import, as `test_quant_cli.py:118-127`'s
    `tiny_dir` does. It writes `write_tiny_hv1(root, config=TINY_FP8_HV1_TP2)` under
    `tmp_path_factory.mktemp("tiny_hv1_quant")` and returns `str(root)`.
  - Every test that runs `prepare_quant` calls `_unset(monkeypatch, "DIFFLET_NKI_TRACE_CACHE")`
    and sets `DIFFLET_COMPILE_CACHE` to a fresh `tmp_path / "cache"`.
  - `_args(quant="mxfp8", quant_compute=None)` stays at `tp_degree=2`. The toy's per-rank
    `ff.net.2` K is then 512, and after E6 `proj_out_attn`'s is 128.
  - Where a test points `resolve_model_path` at the toy, it patches
    `difflet.pipeline.path_resolver.resolve_model_path` after `launches`.

  The tests:
  - **`test_prepare_quant_runs_after_the_cache_root_and_before_the_fit_check(launches, monkeypatch, mode)`**,
    over compile and generate. Recorders replace `hv_mod.root_compile_cache_at_cache_dir`,
    `_prepare_quant` and `_check_dit_fits`. Expect the order
    `["root", ("quant", mode == "compile"), "fits"]` and one launch.
  - **`test_the_stage_and_dp_worker_lines_carry_quant_only_when_given(launches, monkeypatch, tmp_path)`**
    (Review Focus 5). With `_prepare_quant` stubbed:
    - generate with `--quant mxfp8 --quant-compute dequant`, and the stage's `cli_args` contain
      both pairs;
    - `difflet.cli.dp.router.worker_cli_args(args)` contains them too;
    - a bf16 `_args()` gives no `--quant` in either line.
  - **`test_generate_before_compile_is_refused_naming_difflet_compile(launches, monkeypatch, tmp_path, capsys, tiny_fp8_snapshot)`**
    (Review Focus 4). Expect:
    - `SystemExit` with code 2;
    - `stages == []`;
    - stderr starts with `"Error: --quant: transformer: "`, contains `` `difflet compile` ``, and
      ends with `"; nothing was launched.\n"`;
    - `not (tmp_path / "cache" / "quantized").exists()`.
  - **`test_compile_writes_the_directory_and_generate_resolves_it(...)`**. Compile does this:
    - prints `[hunyuan_video] MXFP8 transformer (nki): <dir>`;
    - `<dir>/difflet_quant.json` exists under the cache;
    - `os.environ["DIFFLET_NKI_TRACE_CACHE"] == "1"`;
    - one generate launch whose `cli_args` contain `"--quant", "mxfp8"`.

    A following generate resolves the same `_quantized_dit` and launches one generate stage.
  - **`test_a_target_on_the_fused_proj_out_is_refused_before_the_writer(...)`** (Review Focus 2).
    Set `DIFFLET_QUANT_TARGETS=single_transformer_blocks\.\d+\.proj_out`, then compile. Expect:
    - exit 2;
    - stderr equal to
      `f"Error: --quant: single_transformer_blocks.0.proj_out {FUSED_PROJ_OUT_REASON}; nothing was launched.\n"`;
    - no `MXFP8` line on stdout;
    - `stages == []`;
    - no `quantized` directory under the cache.
  - **`test_quant_without_local_weights_is_refused_naming_the_download(launches, monkeypatch, capsys)`**.
    `resolve_model_path` raises `OSError`. Expect exit 2 and stderr containing
    `"difflet download --model-id hunyuanvideo-community/HunyuanVideo"`.
  - **`test_the_dit_fit_check_counts_the_quantized_directory_as_stored(monkeypatch, tmp_path, capsys)`**.
    Use `_header_only_snapshot(tmp_path / "snap", _HV_DIT_PARAMS)`,
    `DIFFLET_NEURON_HBM_GIB=24` and `_args(tp_degree=1, quant="mxfp8")`. `_quantized_dit` is set
    to a header-only directory (a `model.safetensors` header `F8_E4M3`,
    `shape=[n]`, `data_offsets=[0, n]`):
    - 17 GiB passes `_check_dit_fits()`;
    - 23 GiB exits 2 with `"GiB of MXFP8 weights per core at tp=1"` in stderr.
  - **`test_generate_stage_times_the_steps_and_hands_the_spec_to_the_application(neuron_stage, monkeypatch, tmp_path)`**.
    `hv_mod.run_generate_stage` is replaced by a recorder. Expect `seen["timing"] is True`,
    `seen["application_kwargs"]["quant"] == Q.quant_spec(args).to_dict()` with `--quant`, and no
    `"quant"` key without it.

  In `test_quant_cli.py`:
  - **`test_hunyuan_video_orchestrator_forwards_and_keys_quant(monkeypatch)`**. This is the Wan
    twin. It calls `orch._stage_cache_inputs("generate", args)` with `--tp-degree 2`. Expect:
    - the added keys are exactly
      `["quant", "quant_compute", "quant_schema", "quant_targets_sha256", "quant_tp_degree"]`;
    - `quant_tp_degree == 2`;
    - the bf16 key is the rest;
    - `mx_dir.parent == plain_dir.parent`;
    - the stage parser consumes the line with `extra == []` and rebuilds the same spec;
    - `_stage_cache_inputs("vae", ...)` has no `quant*` key;
    - dequant compute is keyed apart.
  - **`test_hunyuan_video_application_kwargs_carry_the_spec()`**. Expect
    `kwargs["quant"]["format"] == "mxfp8"`, compute `"nki"`, tp 4, no `quant_checkpoint`, and the
    remaining kwargs equal to the bf16 kwargs.
- [x] **Step 3: Run** `pytest tests/unit/cli/test_orchestrator_hunyuan_neuron.py tests/unit/cli/test_quant_cli.py -v`
  with no `-k`. A keyword filter would miss Review Focus 2's and 4's pins and
  `test_generate_stage_times_...`, since neither the names nor the module name contain "quant".
  Expected: every new test FAILs. Task E1's refusal still blocks `--quant`, and there is no
  `_prepare_quant` and no `timing`. The existing tests pass.
- [x] **Step 4: Implement**, and remove Task E1's `--quant` row and its test.
- [x] **Step 5: Run** `pytest tests/unit/cli tests/unit/quant/test_checkpoint.py -q`. Expected:
  PASS. The existing HV1 driver tests (stage plan, the tp1 refusal, the AoT path) are unchanged.
- [x] **Step 6: Commit** `feat(hunyuan-video): --quant mxfp8 through the driver — quantized directory on compile, stage and DP flags, cache key, HBM as stored, the step median, the fused proj_out refused before the writer`, then push.

**Done** (Step 1's conftest guard `1cffb29`, the driver `7f0eecf`).

### Task E5: HunyuanVideo on the device (trn3, tp4, 50 steps, seed 42, LNC=2)

**Done 2026-11-08** (evidence section 7; E6 ran first, so the arms measured the final table, G7
included). Default: V15 (no G1, G5), committed as `fc38fb6`: 634.9 against 660.0 ms (0.96x) at
320x512x61, 31.09 dB / 0.938 / 0.050 from bf16, the only arm that clears the screen; 0.97x at
544x960x129. Deviations: step 1's download ran earlier; `Q` is the Decisions block's
floor-relative screen (30 dB / 0.857 / 0.062), not the margin form in step 5's text; the step-3
floor ran one generate (Decisions block: screening arms run one), as did step 8's bf16 SP context;
V1-V6 ran although `mx` was faster than bf16; step 5's profile used the Wan phase H capture recipe
(1-step dump, 4-worker replay), as step 5 allows; at 720x1280x129 both layouts ran out of HBM at
load and the compile's peak RSS was not recorded with `time -v` (one `ps` sample), so the
realistic shape is 544x960x129 at tp4, with no floor of its own (`bf16_tp2_L` does not fit: HBM
OOM on the rerun, evidence 7f; step 3's 320x512x61 floor is quoted, and the long-shape screen is
unresolved, pending phase G);
step 8 ran from `f61e833` with V15's override (the same spec and directory as `fc38fb6`). Open:
MXFP8 under SP is 1.92x bf16 SP (the rotary's element-wise DMAs, user decision pending; README and
`docs/cli-staged-commands.md` say not to combine `--quant mxfp8` with `--sp`); the DP
router ignores an inherited `NEURON_RT_VISIBLE_CORES` (pre-existing on `main`). The status page
is updated outside this plan's commits.

No code unless a step fails; then one bug, one commit, as `difflet-device-verify` prescribes.
Step 5's default-set commit is the exception.

**Preconditions:**
- E4 has landed, and E6 too when the order rule put it first. In that case E6 step 6a's
  K-remainder probe has passed before step 4.
- The host is free and the Wan section is in the evidence doc (Global Constraints, "The device
  host").
- `df -h /home/ubuntu` shows at least 300 GB free (655 GB on 2026-11-07). The snapshot is
  about 40-45 GB, each quantized directory 17-21 GB, and step 5 keeps up to ten of them (`mx`
  and nine arms) until its decision. The rest is room for the NEFF caches.

**Run conventions:**
- Each arm `<a>` produces `$W/<a>_compile.log`, `$W/<a>_gen50{a,b}.{log,mp4}` and
  `$W/work_<a>_gen50{a,b}/latents.pt` (`--work-dir ... --keep-work-dir`).
- Every log ends with an `exit <code>` line (`run` below), and a nonzero code stops the step.
  No `| tee`, which hides the exit status.
- Record per arm:
  - each generate's step median, from the `[hunyuan_video] request: ... step median` line;
  - the rank line: `compiled_blocks=60 unwarmed_shapes=[] fallbacks: load=[] generate=[]`;
  - md5 of each mp4 and each latents file (gen50a == gen50b);
  - for mxfp8, the `[hunyuan_video] MXFP8 transformer (nki): <dir>` line and `du -sh <dir>`.
    Expect about 19.1 GB for the full set and 19.6-21.1 GB with one or two groups in bf16;
  - `$W/<a>_vs_bf16.json`, against `bf16_gen50a`.
- **Jitter `J`:** the larger of two values:
  - the largest |gen50a - gen50b| step-median difference among the arms being compared;
  - 1% of bf16's median, so a sub-millisecond drift between two identical runs never decides.

  An arm's step is the mean of its two medians. "A is faster than B" means
  step(A) < step(B) - J.
- **Quality gate `Q`:** measured against `bf16_gen50a`, on the clip's mean PSNR, SSIM and LPIPS
  in the compare JSON. `F` is step 3's bf16 noise floor.
  - PSNR >= min(30 dB, F_psnr - 4.1 dB);
  - SSIM >= min(0.97, F_ssim - 0.017);
  - LPIPS <= max(0.05, F_lpips + 0.017).

  These are D5's image gates (Qwen 6d, FLUX section 7). A gate is loosened to the floor minus a
  margin only where the floor lies within that margin of it, that is, where a model with Wan's
  accepted quantization error on top of this floor would fail it. The margin is that of Wan's
  committed default against its own floor (C6 at 9 frames: 37.02 -> 32.96 dB, 0.9603 ->
  0.9439, 0.0201 -> 0.0362; see `fp8-bench/wan21_tp4/bf16_tp2_vs_tp4_f9.json` and
  `mx_ffnx_vs_bf16_f9.json`), rounded up. Video cannot meet the image gates alone: Wan's own
  bf16 floor has SSIM 0.960, and LTX-2's floor is 25.9 dB / 0.909 (section 3a). Wan's
  committed default clears `Q` against its own floor (30 dB / 0.9433 / 0.05).

```bash
export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:$PATH
PY=/home/ubuntu/venvs/native-fp8/bin/python
SRC=/home/ubuntu/Difflet/.claude/worktrees/mxfp8-video-models
W=/home/ubuntu/fp8-bench/hv1_tp4; mkdir -p $W
REV=e8c2aaa66fe3742a32c11a6766aecbf07c56e773
SNAP=/home/ubuntu/.cache/huggingface/hub/models--hunyuanvideo-community--HunyuanVideo/snapshots/$REV
D=(env PYTHONPATH=$SRC $PY -m difflet.cli.main)
M=(--backend neuron --model-id hunyuanvideo-community/HunyuanVideo --revision $REV
   --cache-dir /home/ubuntu/difflet-cache-video)
S=(--height 320 --width 512 --num-frames 61); L=(--height 720 --width 1280 --num-frames 129)
G=(--steps 50 --guidance-scale 6.0 --seed 42)
P=(--prompt "a cinematic shot of a red fox running through a snowy forest")
run() { local log=$1; shift; "$@" > "$log" 2>&1; local rc=$?; echo "exit $rc" >> "$log"; return $rc; }
compare() {  # compare <ref arm> <cand arm>: gen50a against gen50a -> $W/<cand>_vs_<ref>.json
  PYTHONPATH=$SRC $PY $SRC/scripts/compare_difflet_runs.py \
    --ref-latents $W/work_$1_gen50a/latents.pt --ref-image $W/$1_gen50a.mp4 \
    --cand-latents $W/work_$2_gen50a/latents.pt --cand-image $W/$2_gen50a.mp4 \
    --out $W/$2_vs_$1.json; }
# One arm <a> at shape <SH> (S or L) with flags <f...>; an env prefix (E...) goes on all three:
#   run $W/<a>_compile.log E... "${D[@]}" compile "${M[@]}" "${SH[@]}" --tp-degree 4 <f...>
#   for x in a b; do run $W/<a>_gen50$x.log E... "${D[@]}" generate "${M[@]}" "${SH[@]}" \
#     --tp-degree 4 <f...> "${G[@]}" "${P[@]}" --output $W/<a>_gen50$x.mp4 \
#     --work-dir $W/work_<a>_gen50$x --keep-work-dir; done
#   md5sum $W/<a>_gen50{a,b}.mp4 $W/work_<a>_gen50{a,b}/latents.pt
#   grep -ho "step median [0-9.]* ms (n=[0-9]*" $W/<a>_gen50{a,b}.log; compare bf16 <a>
```

- [x] **Step 1: Download, online.** The snapshot is not on this host; checked 2026-11-07: there
  is no `models--hunyuanvideo-community--HunyuanVideo` in the hub cache. `difflet download`
  goes through `resolve_model_path(local_files_only=False)` -> `snapshot_download`, which
  honours `HF_HUB_OFFLINE`. So this step runs with the variable unset, and the export comes
  after it:
  ```bash
  df -h /home/ubuntu
  run $W/download.log env -u HF_HUB_OFFLINE "${D[@]}" download \
    --model-id hunyuanvideo-community/HunyuanVideo --revision $REV
  df -h /home/ubuntu; ls $SNAP
  du -cshL $SNAP/transformer/*.safetensors | tail -1
  export HF_HUB_OFFLINE=1   # every later step
  ```
  Expected:
  - `ls` shows `model_index.json scheduler text_encoder text_encoder_2 tokenizer tokenizer_2 transformer vae`;
  - the DiT totals 25.6 GB (12,821,012,544 bf16 parameters; `du` prints about 24G);
  - the rest (Llama-3 8B, CLIP-L and the VAE) is about 15-20 GB.
- [x] **Step 2: bf16 at S, tp4** (arm `bf16`, no flags). Compile, then gen50a and gen50b.
  Expected: the md5s match, and the step median becomes this branch's bf16 reference. The
  reference is new because `fcb0dca`'s rotary op is in HV1's attention and this host is trn3.
  The trn2 native 808 ms and the trn3 AoT 650 ms are context, not the reference.
- [x] **Step 3: bf16 noise floor at S** (arm `bf16_tp2`: the arm template with `--tp-degree 2`
  in place of 4). Compile, then gen50a and gen50b (md5s match). Then `compare bf16 bf16_tp2`.
  `bf16_tp2_vs_bf16.json` is `F`, the floor of `Q`.
- [x] **Step 4: mxfp8 at S, tp4** (arm `mx`, flags `--quant mxfp8`). Compile, recording the
  MXFP8 line and `du`; then gen50a and gen50b. Expected: the md5s match. Record the ratio to
  step 2 and `mx_vs_bf16.json`.
- [x] **Step 5: the default set at S, by leave-one-out.** This is D5's rule, with the `J` and
  `Q` defined above.
  - **Groups:** `Gi` is the i-th pattern of `HUNYUAN_VIDEO_TARGETS` as it stands: G1..G6 (Task
    E1), plus G7 when E6 ran first.
  - **Arms:** an arm sets `T` to the default's patterns minus its groups, `;`-joined in table
    order. It runs the arm template with `--quant mxfp8` and the prefix
    `env DIFFLET_QUANT_TARGETS="$T"` on the compile and on both generates. The spec's targets
    key the directory and the stage cache, and `mx`'s directory exists after step 4, so a
    generate that lost the prefix would quietly run `mx`. Check that each generate log's
    MXFP8 line names the directory its arm's compile printed. With table order kept, a
    committed table equal to an arm has that arm's spec digest and reuses its directory and
    NEFFs.
    - `V15` (no G1, no G5): the q/k/v groups whose output feeds RMSNorm, the rotary op and the
      attention kernel (Wan's 5x case). It always runs.
    - `V24` (no G2, no G4): the 256-row text groups (LTX-2 kept its 126-row audio stream in
      bf16 for want of gain). It always runs.
    - `V7` (no G7), when E6 ran first: the new K-remainder group, and E6 step 6's arm. It always
      runs.
    - `V1`..`V6`: these run when `mx` is not faster than `bf16`.
  - **Per arm:** `df -h /home/ubuntu` first. Record as the run conventions say. Keep every arm's
    directory until the decision.
  - **Rule:**
    1. The candidates are `mx` and every arm run that clears `Q`.
    2. If any candidate is faster than `bf16`, the default is the fastest of those. Sets within
       `J` of it count as tied; among tied sets, the one with more quantized linears wins (it
       saves more HBM).
    3. Otherwise, if any candidate exists, the default is the fastest candidate (same tie
       rule). The evidence and the README then document HV1 as memory-only: a memory saving,
       with the step not below bf16's by more than `J`.
    4. Otherwise (no set clears `Q`), choose by 2-3 over every measured set. Record each failed
       gate and its distance from the floor. Mark the quality "below the device gate, pending
       phase G" in the evidence and the README, as LTX-2's 22.5 dB was.

    The default is always a measured arm. No set is ever built as a union of groups measured
    separately, and every arm keeps at least four patterns, so no spec is ever empty (QuantSpec
    refuses an empty target list).
  - **Profile**, only when the default is not `mx`: take 3-step captures of `bf16`, `mx` and the
    default arm. Record each one's median device time for the dual-stream and the
    single-stream block NEFF; the commit message quotes them. The recipe uses neuron-explorer
    2.33 (`/opt/aws/neuron/bin/neuron-explorer`; `inspect -o` and
    `view -d/--output-format/--output-file` as its `--help` lists them, and the json format of
    the Qwen doc's 6e). If the Wan section (C6 step 7) records a different capture recipe, use
    that one and say so in the evidence.
    ```bash
    X=/opt/aws/neuron/bin/neuron-explorer
    # $W/prof_<arm>.sh (chmod +x): the arm's generate line with its env prefix and flags, but
    #   --steps 3 --guidance-scale 6.0 --seed 42 "${P[@]}" --output $W/prof_<arm>/out.mp4
    run $W/prof_<arm>.log $X inspect -o $W/prof_<arm> $W/prof_<arm>.sh
    ses=$(ls -d $W/prof_<arm>/i-*_pid_*/*/ | head -1)   # one rank's session: ntrace.pb, trace_info.pb (as wan21_tp4/inspect_mx)
    run $W/prof_<arm>_view.log $X view -d $ses --output-format json \
      --output-file $W/prof_<arm>/system_profile.json
    $PY /home/ubuntu/fp8-bench/wan21_tp4/analyze_profile.py $W/prof_<arm>/system_profile.json \
      > $W/prof_<arm>_neffs.txt
    ```
    The TP ranks run the same block NEFFs, so any one rank's session serves. Check that the
    table in `prof_<arm>_neffs.txt` lists the two block NEFFs (dual- and single-stream) with
    the most executions. If it lists no `nc_exec_running` executions, or `view` wrote nothing
    to that path, stop and ask for the Wan capture's recipe rather than guess one.
  - **Commit**, only when the default is not `mx`:
    - `difflet/quant/targets.py`: the dropped groups move to `HUNYUAN_VIDEO_KEPT_BF16`, with the
      measurement in a comment, as Wan's has;
    - the per-block sets and the count (360 -> n) in `tests/unit/quant/test_targets_hunyuan_video.py`;
    - `TINY_TARGETS_HV1` and the name assertions in `tests/unit/backends/test_neuron_hv1_fp8.py`.

    Message: `feat(quant): HunyuanVideo's default MXFP8 set is <groups> — <dropped groups> cost the tp4 step (<a> vs <b> ms device time per <dual|single> block with and without them), without them the step is <c> ms against <d> in bf16 (<ratio>x), <psnr> dB from bf16 (floor <F> dB)`.
    Push, then run `pytest tests/unit/quant tests/unit/backends/test_neuron_hv1_fp8.py tests/unit/cli -q`.
    When the default is `mx`, nothing is committed; record "default kept: mx, <ratio>x".
  - **Clean up:** delete every arm's quantized directory except the default's (the paths are on
    their MXFP8 lines), then run `df -h /home/ubuntu`. From here on, `<dflt>` names the default's
    arm (`mx` or a `V`), and the default needs no env prefix.
- [x] (partial: the compile's peak RSS was not recorded with `time -v`, see Done) **Step 6: bf16 at L, tp4** (arm `bf16_L`).
  - **Disk and memory:** run `df -h /home/ubuntu` first. Then wrap the compile in a memory
    sampler and `/usr/bin/time -v`:
    ```bash
    (while sleep 30; do date +%T; free -g | sed -n 2p; done) > $W/mem_L.log 2>&1 & MEMPID=$!
    run $W/bf16_L_compile.log /usr/bin/time -v "${D[@]}" compile "${M[@]}" "${L[@]}" --tp-degree 4
    kill $MEMPID
    ```
  - **Run:** gen50a, then gen50b. Estimate about 45-60 s per step, so 40-50 min of steps per
    generate (scaled from Wan's 5.0 s at 32,760 tokens to 119,056 joint tokens and 60 blocks).
    Supervise with the log, not by polling.
  - **Decode, recorded apart from the step median:** use the request line's time after the last
    step (the in-process decode on rank 0's core, the default placement), or the vae launch's
    own time under `--device-vae`. At L the decode covers about 12x S's pixels (16.0 s at S),
    so expect a few minutes.
  - **If an HBM OOM kills the DiT forward:** rerun this arm and step 7 at
    `--tp-degree 4 --cp-degree 2 --cp-mode gather_kv`.
  - **If the rank-0 decode or its compile-time warm-up decode fails** (the traceback is in the
    decoder application, after the DiT's warm-up or after the last step): rerun the arm with
    `--device-vae`. That is the one-core vae launch, which never shares a core with the DiT.
    Step 7 then uses it too.
  - **If the compile fails** (host OOM, BIR emission, or more than 3 h): record the log, the
    exit code, the peak RSS (`time -v`'s "Maximum resident set size"), `dmesg | tail` and the
    attempt number, as the trn2 table does. Then try `--height 544 --width 960 --num-frames 129`,
    then `--height 720 --width 1280 --num-frames 61`. The first shape and layout that runs is
    "the realistic shape" for steps 6-7.
- [x] **Step 7: the default at the realistic shape** (arm `mx_L`, `--quant mxfp8`, plus any flag
  step 6 needed). Compile, then gen50a and gen50b. Record the ratio and `compare bf16_L mx_L`.
  - **Floor at this shape:** arm `bf16_tp2_L` (`--tp-degree 2`), if tp2 fits. It is the one
    single-generate exception (Global Constraints), and gives `compare bf16_L bf16_tp2_L`. If
    tp2 does not fit, record why and quote step 3's floor.
  - **Expected:** about 1.0x. At 118,800 image tokens attention dominates the block, as at Wan's
    81 frames (0.98x). Record it as measured.
  - **Done 2026-11-08** (evidence 7f). `mx_L` ran at 544x960x129: 0.97x, 26.46 dB / 0.907 /
    0.128 from bf16. The floor sub-step: tp2 does not fit. `bf16_tp2_L`'s first attempt (cores
    0-1, 05:31) died in the 06:14 full-disk window ("No space left on device" from the compiler;
    its ranks lingered on core 0 until the wrapper exited); the rerun on device 8 (cores 32-33)
    ran out of HBM at 07:19 ("Failed to allocate DEVICE memory (415236096 bytes)") and was
    stopped cleanly by the OOM watcher. As this step says, step 3's floor is quoted (36.64 dB /
    0.957 / 0.031 at 320x512x61); a floor from another shape cannot resolve the long-shape
    screen, so it stays pending phase G (CPU fp32 reference, running).
- [x] **Step 8: layouts at S with the default set** (`--quant mxfp8`).
  - **CP2:** arm `mx_cp2` with `--tp-degree 4 --cp-degree 2 --cp-mode gather_kv`: 8 cores, 5,120
    image and 5,376 joint rows per rank. Compile, then gen50a and gen50b (md5s match). Record
    the step median, `compare bf16 mx_cp2` and `compare <dflt> mx_cp2`.
  - **SP:** arm `mx_sp` with `--tp-degree 4 --sp`. Compile, then gen50a and gen50b. Matching
    md5s separate the reduction-order floor from run-to-run nondeterminism. Run
    `compare <dflt> mx_sp` and `compare bf16 mx_sp`. This is Review Focus 3 on the device:
    against `<dflt>`, expect at most step 3's floor.
  - **DP2:** the request lines carry seed, steps and guidance. `"${P[@]}"` is left out, because
    `--prompt` and `--requests` are mutually exclusive (`difflet/cli/main.py:1163-1165`).
    `"${G[@]}"` is passed, and the router forwards `--steps` and `--guidance-scale`
    (`difflet/cli/dp/router.py:103-104`).
    ```bash
    cat > $W/dp2_requests.jsonl <<EOF
    {"prompt": "a cinematic shot of a red fox running through a snowy forest", "seed": 42, "steps": 50, "guidance_scale": 6.0, "output": "$W/dp2_fox.mp4"}
    {"prompt": "a red bicycle leaning against a brick wall, soft afternoon light", "seed": 42, "steps": 50, "guidance_scale": 6.0, "output": "$W/dp2_bicycle.mp4"}
    EOF
    run $W/mx_dp2.log "${D[@]}" generate "${M[@]}" "${S[@]}" --tp-degree 4 --quant mxfp8 \
      --dp 2 --requests $W/dp2_requests.jsonl "${G[@]}"
    ```
    Expected:
    - both workers print the `[hunyuan_video] MXFP8 transformer (nki)` line;
    - each worker's step median is within `J` of `<dflt>`'s;
    - `md5(dp2_fox.mp4) == md5(<dflt>_gen50a.mp4)`.
- [x] **Step 9: evidence and docs.**
  - **Evidence doc:** a HunyuanVideo section in
    `docs/verification/2026-11-07-mxfp8-video-models.md`, numbered as the next free number at
    the time of writing (Global Constraints). It holds:
    - steps 1-8, with `J`, `F` and `Q`'s resolved thresholds;
    - the arm table of step 5: arms, patterns, both medians, md5s, ratio, PSNR/SSIM/LPIPS against
      bf16, directory size, and whether the arm clears `Q`; then the rule's outcome and branch;
    - the profile table, if step 5 took one;
    - a summary table like LTX-2's 3e: layout | DiT step | vs bf16 clip PSNR/SSIM/LPIPS | latents
      rel-L2/cosine;
    - the 720p compile record, with peak RSS and the decode time;
    - the quantized size against bf16;
    - the state of the single-block `proj_out`: bf16 until Task E6, or quantized as its halves
      when E6 ran before E5.
  - **README:** add HunyuanVideo to the FP8 note: step ratio (or "memory-only"), quality,
    quantized size, and the `proj_out` caveat as FLUX's.
  - **`docs/cli-staged-commands.md`:** an HV1 paragraph after LTX-2's, and the refusal
    sentence's model list made current (Wan and HunyuanVideo added).
  - **Commit** `docs: HunyuanVideo in MXFP8 on Trainium3 — evidence (tp4 at 320x512x61 and <realistic shape>, noise floor, default-set arms, CP2, SP, DP2), README feature note, staged command`.
    Then push and update the status page.

### Task E6: the single-stream `proj_out` halves join HV1's table (needs phase D's D1 + D2)

**Precondition:** D1 and D2 have landed (see Global Constraints, "Order with phase D"). The names
below are D's plan's names (`docs/plans/2026-11-07-mxfp8-proj-out-rope.md`, D1/D2 Interfaces);
use them as they landed.

**Files:**
- Modify: `difflet/quant/targets.py`:
  - `HUNYUAN_VIDEO_TARGETS` and `HUNYUAN_VIDEO_ROW_PARALLEL` gain G7
    `rf"{_SINGLE}\.proj_out_(attn|mlp)"`;
  - `HUNYUAN_VIDEO_KEPT_BF16` loses `{_SINGLE}\.proj_out_(attn|mlp)`; the fused
    `{_SINGLE}\.proj_out` stays, as D2 did for FLUX;
  - `SPLITS_BY_MODEL["hunyuan_video"] = SINGLE_BLOCK_SPLIT_LINEARS`;
  - `fused_proj_out_targets` and `FUSED_PROJ_OUT_REASON` are deleted with their `__all__`
    entries, since nothing else calls them;
  - the HV1 docstring paragraph is rewritten: the halves are quantized per rank K range of each
    half.
- Modify: `difflet/backends/neuron/hunyuan_video/transformer.py`: drop the guard. The docstring
  says the quantized directory holds the halves, read by name, because the base drops the slices
  its spec splits.
- Modify: `difflet/cli/orchestrators/hunyuan_video.py`: `_prepare_quant` drops the config read
  and the guard.
- Modify: `difflet/cli/quant.py:41-46`: in D2's bf16 clause only, "except FLUX's single-stream
  output projection" becomes "except the single-stream output projection of FLUX and
  HunyuanVideo". The full merged text is in Global Constraints, "Text shared with phases D and
  F".
- Modify: `tests/unit/backends/_neuron_workers.py` (`hv1_dit_worker`, as E2 left it). The result
  gains two keys for `m = app.module.single_transformer_blocks[0].proj_out_attn`:
  - `"attn_twin": [type(m).__name__, m.skip_bias_add, m.reduce_output]` (D2's
    `flux_dit_fp8_worker` key);
  - `"attn_weight": m.weight.detach().float().tolist()`.
- Test: `tests/unit/quant/test_targets_hunyuan_video.py`,
  `tests/unit/backends/test_neuron_hv1_fp8.py`,
  `tests/unit/cli/test_orchestrator_hunyuan_neuron.py`.

**Interfaces:**
- Consumes D1's `QuantSpec.split_linears` / `is_split` and D1's splitting writer (the halves are
  `<p>_attn` = columns `[0, N)` and `<p>_mlp` = `[N, K)`, and `<p>.bias` is copied). Also D2's
  `SPLITS_BY_MODEL` / `splits_for` in `default_spec`, and the base dropping every slice its spec
  splits.
- Real geometry: per-rank K is 3072/tp for `proj_out_attn` and 12288/tp for `proj_out_mlp`.
  These are multiples of 128 at tp 1, 2, 4 and 8 (384 and 1536 at tp8). At tp4,
  `proj_out_attn`'s 768 lays out as 512 + one partial tile of 256 (`mx._tile_sizes`). It is the
  first HV1 target with a K remainder, so step 6a probes it on one core before any device arm.
- The toy: E2's and E4's fixtures already use `TINY_FP8_HV1_TP2` (Task E1). Its halves' per-rank
  K at tp2 is 128 and 512, so no fixture changes here.
- Size: G7 is 1,887,436,800 parameters (40 x 3072 x 15360). The full set with G7 is about
  17.2 GB, against 19.1 GB without it.

- [x] **Step 1: Update the tests.**
  - **`test_targets_hunyuan_video.py`:**
    - the default spec matches `single_transformer_blocks.0.proj_out_attn` and `.39.proj_out_mlp`
      by name and not `single_transformer_blocks.0.proj_out`;
    - `spec.is_split("single_transformer_blocks.0.proj_out")` holds;
    - `default_spec("hunyuan_video").split_linears == SINGLE_BLOCK_SPLIT_LINEARS`;
    - `spec.k_shards("single_transformer_blocks.0.proj_out_attn") == spec.k_shards("single_transformer_blocks.0.proj_out_mlp") == 4`
      at tp4;
    - the real-geometry test also checks the two halves' K / tp % 128 at tp 1, 2, 4 and 8;
    - the fused-helper test goes;
    - the coverage test keeps diffusers' fused `proj_out` matched once, by the kept table.
  - **`test_neuron_hv1_fp8.py`:**
    - `TINY_TARGETS_HV1 = 10 * num_layers + 6 * num_single_layers` (22);
    - the reference splits each fused `proj_out` weight at `inner` and fake-quantizes each half
      with `spec.k_shards(f"{name}_attn" / f"{name}_mlp")`, which is D2's `_fake_quant_reference`
      law;
    - the tp1 test asserts `r["fp8"]["single_transformer_blocks.0.proj_out_attn"] == r["fp8"]["single_transformer_blocks.1.proj_out_mlp"] == "FP8RowParallelLinear"`;
    - the tp2 and tp2-sp cases now cover the per-rank halves. Each rank asserts
      `r["attn_twin"] == ["FP8RowParallelLinear", True, False]` (skip_bias_add, no
      reduce_output; D's Review Focus 3), and its output against the reference;
    - `test_a_target_on_the_fused_single_block_proj_out_is_refused_until_phase_d` becomes
      `test_a_target_on_the_fused_proj_out_is_a_dead_pattern_for_the_split_writer`: with
      `default_spec(..., targets=(*default, r"single_transformer_blocks\.\d+\.proj_out"))`,
      `prepare_quantized_checkpoint(..., create=True)` raises ValueError "match no linear
      weight" and writes nothing;
    - new `test_a_target_set_without_the_halves_loads_them_in_bf16_by_name`, D's Review Focus 1
      for HV1 and the twin of D2's FLUX test. Before this task it fails at its first assertion
      (`split_linears == ()`), and the directory assertions would fail as well: with no
      `SPLITS_BY_MODEL` entry for HV1, the directory holds the fused `proj_out.weight` and the
      loader slices it. The old version of this test (load, plain row layer, output) passed
      without the change, so it pinned nothing.
      ```python
      def test_a_target_set_without_the_halves_loads_them_in_bf16_by_name(tiny, tmp_path):
          no_halves = tuple(p for p in HUNYUAN_VIDEO_TARGETS if "proj_out" not in p)
          spec = default_spec("hunyuan_video", compute="dequant", tp_degree=1, targets=no_halves)
          assert spec.split_linears == SINGLE_BLOCK_SPLIT_LINEARS
          quantized = prepare_quantized_checkpoint(tiny.transformer_dir, spec, create=True,
                                                   cache_root=tmp_path / "cache")
          stored, source = _tensors(quantized), _tensors(tiny.transformer_dir)
          inner = TINY_FP8_HV1_TP2["num_attention_heads"] * TINY_FP8_HV1_TP2["attention_head_dim"]
          for i in range(TINY_FP8_HV1_TP2["num_single_layers"]):
              p, w = f"single_transformer_blocks.{i}", source[f"single_transformer_blocks.{i}.proj_out.weight"]
              assert f"{p}.proj_out.weight" not in stored                      # nothing to slice
              assert torch.equal(stored[f"{p}.proj_out_attn.weight"], w[:, :inner])
              assert torch.equal(stored[f"{p}.proj_out_mlp.weight"], w[:, inner:])
              assert f"{p}.proj_out_attn.weight_scale" not in stored            # not quantized
          [r] = run_ranks(hv1_dit_worker, tiny.transformer_dir, {"tp_degree": 1}, "compile", 9,
                          None, spec.to_dict(), str(quantized), world_size=1)
          assert r["report"] == {"missing": [], "unexpected": []}   # a slice would miss proj_out.weight
          assert r["attn_twin"][0] == "RowParallelLinear"
          assert torch.equal(torch.tensor(r["attn_weight"]),
                             source["single_transformer_blocks.0.proj_out.weight"][:, :inner])
          torch.testing.assert_close(torch.tensor(r["out"]), _reference(tiny, spec), atol=1e-4, rtol=1e-4)
      ```
      Here `_tensors(dir)` reads every tensor of a directory's safetensors, and
      `_reference(tiny, spec)` is the fixture's reference law (D2's `_fake_quant_reference`) for
      that spec. The load uses no slice: the directory has no `proj_out.weight` to slice, and
      the report is clean.
  - **`test_orchestrator_hunyuan_neuron.py`:** the fused-`proj_out` driver test now expects exit 2
    with "match no linear weight" in stderr, no `MXFP8` line and nothing under the cache.
- [x] **Step 2: Run** `pytest tests/unit/quant/test_targets_hunyuan_video.py tests/unit/backends/test_neuron_hv1_fp8.py tests/unit/cli/test_orchestrator_hunyuan_neuron.py -v`.
  Expected: FAIL, because the halves are not targets yet.
- [x] **Step 3: Implement.**
- [x] **Step 4: Run** `pytest tests/unit/quant tests/unit/backends/test_neuron_hv1_fp8.py tests/unit/backends/test_neuron_hv1_dit.py tests/unit/cli -q`.
  Expected: PASS.
- [x] **Step 5: Commit** `feat(quant): HunyuanVideo's single-stream proj_out joins its MXFP8 table as the attention and MLP halves (phase D's split writer, each half per rank K range); the pre-D refusal goes`, then push.
  **Done** (`fd0d68f`).
- [x] **Step 6 (device).** Use E5's preconditions (the host is free) and E5's run conventions,
  `J` and `Q`. Step 6a always runs before the first device arm that quantizes the halves, in
  E5 too when E6 ran first. Steps 6b-6d run only when E5 ran before E6; otherwise E5's arms
  include V7 and measure the final table.
  **Done 2026-11-08** (evidence section 7d): 6a passed (`proj_out_attn` at K = 768 = 512 + 256:
  rel L2 2.35e-3, 1.50 against 1.48 ms in bf16; `proj_out_mlp` 0.66x, `proj_mlp` 0.69x). 6b-6d
  did not run: E6 (`fd0d68f`) ran before E5, so E5's arms included V7 (667.9 / 668.2 ms, outside
  the screen) and measured the final table, and E5's rule kept G7 in the default (`fc38fb6`).
  - **6a, kernel probe on one core**, for the K remainder.
    - Copy `/home/ubuntu/fp8-bench/wan21_tp4/probe_wan_shapes.py` to `$W/probe_hv1_shapes.py`
      with three changes: `OUT = "/home/ubuntu/fp8-bench/hv1_tp4"`, the output file
      `probe_hv1_shapes.json`, and
      ```python
      SHAPES = [
          ("hv1 proj_out_attn (row, K=3072/4 = 512 + 256)", 10496, 768, 3072),
          ("hv1 proj_out_mlp (row, K=12288/4)", 10496, 3072, 3072),
          ("hv1 proj_mlp (col, control)", 10496, 3072, 3072),
      ]
      ```
      M is the 10,240 + 256 joint rows of a single block at 320x512x61.
    - Run `run $W/probe_hv1_shapes.log env PROBE_WT=$SRC $PY $W/probe_hv1_shapes.py`.
    - Pass: rel L2 < 3e-3 and cosine > 0.9999 at every shape (the criterion of evidence-doc
      section 1).
    - If `proj_out_attn` fails, no arm quantizes it. Its layout is a bug, fixed as `c813a47`
      fixed Wan's 384: one bug, one commit, then the probe again.
    - If D5 has run, quote FLUX's result on its own 768-wide halves beside this one.
  - **6b, arms at S, in one session** so that `J` covers all three:
    - `bf16`: re-run its two generates (cached; no compile);
    - `mxE`: E5's default, its two generates re-run (its directory and NEFFs are cached);
    - `mx7`: the new default, E5's default plus G7. Compile it (about 17.2 GB when E5's default
      is G1-G6), then gen50a and gen50b.

    Record medians, md5s, `compare bf16 mx7` and `compare bf16 mxE`.
  - **6c, rule** (E5 step 5's, on these two sets): G7 stays when `mx7` clears `Q` and is not
    slower than `mxE` by more than `J`. A tie keeps G7, because the set quantizes more and is
    about 1.8 GB smaller. Otherwise G7 moves back to `HUNYUAN_VIDEO_KEPT_BF16`, with the
    measurement in its comment. That is a commit as in E5 step 5, profiled as there:
    `feat(quant): HunyuanVideo keeps the single-block proj_out halves bf16 — they cost the tp4 step <x> (<a> vs <b> ms per single block), <c> vs <d> ms`.
  - **6d, follow-ups:** whenever G7 is kept, the default has changed, so re-run E5 step 7's
    `mx_L` with it. Step 8's layout rows stay as measured, labelled "E5 default (no G7)". The
    halves run the same twin (`reduce_output=False`, `skip_bias_add=True`) at every layout, and
    `mx7` at tp4 already runs it. When G7 goes back, nothing re-runs.
    Update the evidence section (6a's probe table, 6b's arm table, the rule's outcome) and the
    README, then commit
    `docs: HunyuanVideo MXFP8 with the single-block proj_out halves — <ratio> at tp4 320x512x61, <kept|kept bf16>`.
    Push and update the status page.

## After phase E

F (HunyuanVideo-1.5, memory-only) and G (the CPU fp32 arbiter, including HV1 at 320x512x61, 50
steps, seed 42) follow, as the phase-A/B plan's roadmap says.
