# MXFP8 on the single-stream `proj_out` (FLUX, HunyuanVideo) and a neuron-friendly rotary at the diffusers-law call sites — Implementation Plan (phase D)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** make the single-stream blocks' fused `proj_out` of FLUX (and, ready for phase E,
HunyuanVideo) quantizable as its two row-parallel halves, give the FLUX and HunyuanVideo-1.5
rotary call sites the pair-swap-by-matmul law the neuron backend already uses for Wan, and let
the device decide FLUX's default set (tp4, 1024x1024, 50 steps, seed 42, against the history's
222.6 ms bf16 / 220 ms mxfp8 at 28 steps). The halves join the default only in D5, on measurement.

**Architecture:** the split is data in the spec. `QuantSpec.split_linears` names the fused
checkpoint linears the model runs as two column ranges. The writer stores each matched
`<p>.weight [N, K]` as `<p>_attn.weight` (columns `[0, N)`) and `<p>_mlp.weight` (`[N, K)`), and
quantizes each half on its own (per rank K range when row-parallel). The application base then
drops every `CheckpointSlice` whose source the spec splits, so the halves load by name. A spec
without splits writes the same bytes at the same digest. The modeling needs no change. The
FP8 twin of a `RowParallelLinear` keeps `reduce_output=False` / `skip_bias_add=True`
(`fp8_linear.fp8_twin:276-282`) and has `tensor_parallel_group`, so
`modeling_flux.py:720-728` and `modeling_hunyuan_video.py:768-773` run unchanged. FLUX's spec
splits from D2 on, but the halves are override targets (`FLUX_SPLIT_TARGETS`) until D5 Step 7
measures them. The CPU reference layer (`difflet/quant/fake_linear.py`) runs a split linear as
its two halves, so the phase-G arbiter quantizes what the device quantizes. For the rotary, one
new dispatched op, `difflet.ops.apply_rotary_emb_fp32`, carries diffusers' `use_real`
interleaved law in fp32. Every backend except neuron exports the reference code verbatim, so the
Trainium AoT FLUX graph is unchanged. The neuron backend computes the rotated operand as `x @ R`
(R a signed permutation, exact for bf16 x). FLUX reaches it through
`difflet.layers.embeddings.apply_rotary_emb`. HunyuanVideo-1.5's processors (a neuron-only
file) import it directly.

**Tech Stack:** as phases A–C. FLUX.1-dev has 19 dual-stream and 38 single-stream blocks, inner
3072 (24 heads x 128) and MLP 12288. The single block's `proj_out` is `[3072, 15360]`, split at
3072. Per-rank K of the halves is 768 (= 512 + a 256 remainder) and 3072 at tp4, 384 (one
partial tile, the schema-3 law of `c813a47`) and 1536 at tp8: all multiples of 128, but section 1
of the evidence doc probed K 4096 and 2048 only, so D5 Step 3 probes 768 and 384 on the kernel
before any arm quantizes a half. `proj_out_attn` holds the bias (`skip_bias_add=True`),
`proj_out_mlp` has none, and both use `reduce_output=False` with one merged all-reduce. At
1024x1024 there are 4096 image and 512 text tokens (M = 4608 in the single blocks). The neuron
FLUX rotary table is bf16 `[4608, 128, 2]` (`flux/transformer.py:89-100`). Its 8 call sites
(`modeling_flux.py:1263-1264, 1303-1307, 1323-1327, 1410-1411`) all go through
`difflet.layers.embeddings.apply_rotary_emb:40-79`, which is diffusers' `reshape(...,2).unbind(-1)`
/ `stack` / `flatten` form. So the neuron FLUX DiT never got the backend op. Its input is the
`norm_q` / `norm_k` RMSNorm output, not a custom-op output, so the change is measured in bf16
too (D5 Step 2). `difflet/models/flux/tp_sharding.py` is the TPU port's
(`backends/tpu/flux/transformer.py:57`) and is not touched. HunyuanVideo 1.0 has 20 dual and 40
single blocks, hidden 3072 and MLP 12288, with the same split (`modeling_hunyuan_video.py:712-728`).
It already uses `difflet.ops.apply_rotary_emb` (`:1550-1563`). HunyuanVideo-1.5's only diffusers
rotary call is `hunyuan_video15/transformer.py:204-216` (`_joint_qkv`, q/k `[B, S, H, d]`,
`sequence_dim=1`).

**Spec:** roadmap entry D of `docs/plans/2026-11-07-mxfp8-video-models.md`; section 7 of
`docs/verification/2026-10-31-qwen-image-fp8-native-trn3.md` (lines 522-566), the FLUX history:
0.99x, the single-stream set alone 1.04x because a bf16 `proj_out` sits between custom calls,
and "at one seed they rank nothing"; its 6h (lines 391-412), the quality gate as the user
redefined it (mean PSNR over 4 prompts x 3 seeds >= 30 dB, every image >= 27 dB); this
conversation's device facts: the interleaved rotary on a custom-op output, and the bisection
rule.

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

- Everything in the phase-A/B plan's Global Constraints: branch `feat/mxfp8-video-models`,
  every commit pushed, the status page updated per task, D43, no q/k/v fusion, K % 128 per rank,
  byte-identical default command lines, the CPU fp32 reference last (phase G, not here).
- **No `QUANT_SCHEMA` bump.** `split_linears` enters `to_dict()` and `weights_identity()` only
  when non-empty. Every Qwen-Image, LTX-2 and Wan directory keeps its digest, which the Qwen
  toy's literal `17bfb1e6d80f20ff` (`test_checkpoint.py:275`) pins. `cache_fields()` does not
  change: the split changes bytes, not the compiled graph. FLUX's default directory gets a new
  digest in D2 (its spec splits from then on); its bytes differ from the old one only by the two
  halves in place of the fused weight.
- **The split law:** `<p>.weight [N, K]` with `K > N` gives `<p>_attn` = columns `[0, N)` and
  `<p>_mlp` = `[N, K)`. `<p>.bias` is copied unchanged (`single_block_proj_out_key` maps
  `proj_out_attn.bias` to it). Each half is quantized when a target matches it, per rank K
  range when it is row-parallel. Otherwise it is stored as the column range of the source
  tensor in the source's dtype (bf16 for the hub FLUX / HV1 checkpoints, fp32 for the toys),
  as the writer copies every other non-target. The fused `<p>.weight` is not written. For a
  split linear the shard count of half `h` is `spec.k_shards(f"{p}_{h}")`, never
  `spec.k_shards(p)` (the fused name is not row-parallel: 1). This holds in the writer, the CPU
  reference layer and every test reference.
- **FLUX's default set moves only on measurement.** D2 adds the halves as override targets
  (`FLUX_SPLIT_TARGETS`, row-parallel). `FLUX_TARGETS` stays the pre-D set until D5 Step 7
  commits a measured one. This is phase C's lesson: Wan's self-attention q/k/v went into the
  default unmeasured, cost the tp4 block 5x, and needed `917d087` to come out. The halves' tests
  name their target sets literally, so a default change never rewrites them.
  `SPLITS_BY_MODEL["flux"]` stays in every outcome: unquantized halves read by name are the
  fused weight's columns, byte for byte.
- **The rotary law:** `out = (x.float() * cos + rot(x).float() * sin).to(x.dtype)` with
  `rot(x)[2i] = -x[2i+1]` and `rot(x)[2i+1] = x[2i]`, i.e. diffusers'
  `apply_rotary_emb(use_real=True, use_real_unbind_dim=-1)`. On the CPU, every replacement must
  be `torch.equal` to the function it replaces, for any table dtype and any table values (no
  pair-equality assumption). On the device, `x @ R` is exact for bf16 x, which FLUX and HV1.5
  feed. An fp32 x relies on neuronx-cc not auto-casting the matmul to bf16.
- HunyuanVideo 1.0 gets no quant wiring here (phase E). D proves only that the writer's halves
  are what its loader's slices read, and the base rule that E inherits.
- Unit tests: `T` = `cd /home/ubuntu/Difflet/.claude/worktrees/mxfp8-video-models && DIFFLET_BACKEND=cpu PYTHONPATH=. /home/ubuntu/venvs/native-fp8/bin/python -m pytest <paths> -q -p no:cacheprovider`.
  Toys only. A test never compiles a real snapshot (the conftest guard).
- Device (D5): trn3, LNC=2, tp4, 1024x1024, 50 steps, seed 42, "a cat sitting on a bench",
  two generates per arm. The bf16 noise floor is tp2 against tp4. Comparisons use
  `scripts/compare_difflet_runs.py` (images: FLUX decodes in process at tp4). The CLI runs as
  `PYTHONPATH=<checkout> /home/ubuntu/venvs/native-fp8/bin/python -m difflet.cli.main` with
  `/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin` on PATH and `--cache-dir /home/ubuntu/difflet-cache-video`.
  The default's confirmation (D5 Step 7) adds the 6h gate's 4 prompts
  (`scripts/qwen_image_fp8_cpu_reference.py:37-42`) x 3 seeds (42, 7, 1234). D5 starts only
  after phase C's device work is finished: C6 Step 7 and C7 Step 5 have landed their evidence
  sections, and no Wan run holds the device or its disk.

## Review Focus

1. A target set without the halves must still load. That covers FLUX's default until D5, every
   bisection arm without G5/G6, and any `DIFFLET_QUANT_TARGETS` override. The spec keeps
   `split_linears`, so the halves are stored unquantized in the source's dtype and read by
   name with no slice. Pinned in D2 by
   `test_a_target_set_without_the_halves_loads_them_unquantized_by_name` and by the default
   fixture's `test_the_quantized_directory_holds_the_default_targets_and_the_halves_by_name`.
2. A row-parallel half must be quantized per rank K range **of the half**, not of the fused
   weight: rank r reads `proj_out_attn` columns `[r·N/tp, (r+1)·N/tp)` with scales computed from
   exactly those columns. Its shard count is the half's (`k_shards(f"{p}_attn")`) in the writer
   and in the CPU reference alike. Pinned by D1 `test_a_split_linear_is_written_as_its_two_column_ranges`
   (bytes), D2 `test_a_split_linear_runs_as_its_two_column_ranges_each_quantized_on_its_own`
   (the reference layer) and D2 `test_tp2_quantized_halves_match_the_per_rank_reference` (end to
   end on gloo).
3. At tp > 1 the twin of `proj_out_attn` must keep `reduce_output=False` / `skip_bias_add=True`.
   Otherwise the merged all-reduce runs twice or the bias is added tp times. Pinned by D2's gloo
   test: the `attn_twin` assertion, and its output against the reference.
4. Qwen-Image, LTX-2 and Wan specs (no splits) must keep their spec dicts, weights identities and
   directory digests byte-identical, so their 24–38 GB directories are not rewritten on the
   device. Pinned by D1 `test_split_linears_stay_out_of_every_unsplit_spec` and the unchanged
   literal digest test.
5. The rotary must be exact for every input the call sites feed: bf16 or fp32 activations,
   FLUX's bf16 tables and fp32 tables, the `[B, H, S, D]` and `[B, S, H, D]` layouts, and tables
   that are not pair-equal. On the neuron backend the call sites must actually reach the new op,
   which no CPU numeric test can tell apart. Pinned by D3
   `test_fp32_rope_is_bit_identical_to_diffusers`, D3
   `test_flux_rotary_reaches_the_neuron_op_on_the_neuron_backend` and D4
   `test_the_processors_rotate_with_the_neuron_op`.

---

### Task D1: the spec names split linears; the writer stores their halves

**Files:**
- Modify: `difflet/quant/spec.py:46-60` (field), `:61-69` (`__post_init__` compiles the new patterns), `:77-101` (`is_split`, `unmatched_splits`), `:105-120` (`to_dict` / `from_dict`), `:137-146` (`weights_identity`)
- Modify: `difflet/quant/targets.py:93` (`SINGLE_BLOCK_SPLIT_LINEARS` after `_SINGLE`; `__all__`)
- Modify: `difflet/quant/checkpoint.py:10-27` (docstring: the split layout, halves in the source's dtype unless targeted), `:164-177` (`_quantizable_prefixes` → `_weight_shapes` + split-aware prefixes), `:180-198` (`quantize_tensors`), `:272-279` (dead-split refusal before the dead-target one)
- Modify: `difflet/backends/neuron/core/checkpoint.py:85-102` (docstring only: a directory whose spec splits the source holds the halves under these parameter names)
- Modify: `tests/unit/backends/_neuron_flux_toy.py:7-21` (`TINY_FP8` moved here from `test_neuron_flux_fp8.py:27-29`; `write_tiny_flux(root, *, config: dict | None = None)`), `tests/unit/backends/test_neuron_flux_fp8.py:27-29` (imports `TINY_FP8`)
- Modify: `tests/unit/backends/_neuron_hv1_toy.py:26-44` (`TINY_FP8_HV1`, `write_tiny_hv1(root, *, config: dict | None = None)`)
- Test: `tests/unit/quant/test_checkpoint_split.py` (new), `tests/unit/quant/test_spec.py`

**Interfaces:**
- Produces `QuantSpec.split_linears: tuple[str, ...] = ()`, placed after `row_parallel`. The
  patterns are full-matched like `targets`, with the prefix stripped. It also produces
  `is_split(name: str) -> bool` and `unmatched_splits(names: Iterable[str]) -> list[str]`.
  `to_dict()` writes `"split_linears": [...]` only when non-empty, and `from_dict` defaults it to
  `()`. `weights_identity()` adds `"quant_split_linears"` only when non-empty.
- Produces `difflet.quant.targets.SINGLE_BLOCK_SPLIT_LINEARS = (r"single_transformer_blocks\.\d+\.proj_out",)`
  (FLUX's and HunyuanVideo's single-stream `proj_out`, `Linear(dim + mlp_dim, dim)` over
  `cat([attn, mlp])`). No model's default spec carries it until D2.
- Produces `checkpoint._split_parts(prefix: str, shape) -> list[tuple[str, int, int]]`, which
  returns `[(f"{prefix}_attn", 0, N), (f"{prefix}_mlp", N, K - N)]`. When `K <= N` it raises
  `ValueError(f"{prefix}: a split linear needs K > N (its first N input columns are the attention half), got [{N}, {K}]")`.
- `write_quantized_dir` refuses dead splits first:
  `ValueError(f"split patterns match no linear weight in {model_dir}: {'; '.join(dead)}")`.
  It then refuses dead targets against the post-split prefixes, in which a split source counts
  as its two halves.
- `TINY_FP8_HV1 = {**TINY_HV1, "attention_head_dim": 32, "rope_axes_dim": (8, 12, 12)}` gives
  hidden 128 and MLP 512.

- [x] **Step 0: pin the pre-D tree.** Before D1's first commit, run
  `mkdir -p /home/ubuntu/fp8-bench/flux_d_tp4 && git -C /home/ubuntu/Difflet/.claude/worktrees/mxfp8-video-models rev-parse HEAD > /home/ubuntu/fp8-bench/flux_d_tp4/BASE`,
  and name the SHA on the status page. D5 archives exactly this commit. A later commit that
  repeats D1's wording (a `fix(d1): review findings` commit), an amend or a rebase cannot move it.
- [x] **Step 1: Write the failing tests.** The file `test_checkpoint_split.py` has a helper
  `_source(tmp_path, dtype=torch.bfloat16)` that writes, with `save_file`,
  `single_transformer_blocks.0.proj_out.weight` `[256, 1280]` (`W`), `.proj_out.bias` `[256]`
  (`b`) and `.proj_mlp.weight` `[1024, 256]` (`up`), all in `dtype`. `_tensors(dir)` reads every
  tensor.

```python
SPLIT = SINGLE_BLOCK_SPLIT_LINEARS
HALVES = (r"single_transformer_blocks\.\d+\.proj_out_(attn|mlp)",)
P = "single_transformer_blocks.0"

def test_a_split_linear_is_written_as_its_two_column_ranges(tmp_path):
    src, W, b, up = _source(tmp_path)
    spec = QuantSpec(targets=HALVES, row_parallel=HALVES, split_linears=SPLIT, tp_degree=2)
    manifest = write_quantized_dir(src, tmp_path / "q", spec); t = _tensors(tmp_path / "q")
    assert f"{P}.proj_out.weight" not in t
    assert torch.equal(t[f"{P}.proj_out.bias"], b) and torch.equal(t[f"{P}.proj_mlp.weight"], up)
    for half, cols in (("proj_out_attn", W[:, :256]), ("proj_out_mlp", W[:, 256:])):
        q, s = t[f"{P}.{half}.weight"], t[f"{P}.{half}.weight_scale"]
        assert q.dtype == torch.float8_e4m3fn and q.shape == cols.shape
        k = cols.shape[1] // 2
        for r in range(2):  # what rank r's FP8RowParallelLinear reads (shard dim 1, both tensors)
            eq, es = mx.quantize_mx(cols[:, r * k:(r + 1) * k])
            assert torch.equal(q[:, r * k:(r + 1) * k], eq)
            assert torch.equal(s[:, r * k // 32:(r + 1) * k // 32], es)
    assert manifest["quantized_linears"] == [f"{P}.proj_out_attn", f"{P}.proj_out_mlp"]
    # quantizing the fused weight per rank K range and slicing gives another block membership
    assert not torch.equal(t[f"{P}.proj_out_attn.weight_scale"][:, :4], mx.quantize_mx_sharded(W, 2)[1][:, :4])

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_a_split_linear_that_no_target_names_keeps_the_source_dtype(tmp_path, dtype):
    src, W, b, up = _source(tmp_path, dtype)
    spec = QuantSpec(targets=(r"single_transformer_blocks\.\d+\.proj_mlp",), split_linears=SPLIT)
    # t[f"{P}.proj_out_attn.weight"].dtype == t[f"{P}.proj_out_mlp.weight"].dtype == dtype;
    # torch.equal to W[:, :256] / W[:, 256:]; no proj_out_*.weight_scale;
    # manifest["quantized_linears"] == [f"{P}.proj_mlp"]; tensors == {"quantized": 1, "kept": 3, "total": 5}

def test_a_split_pattern_that_matches_no_linear_weight_is_refused_before_writing(tmp_path):
    # split_linears=(r"nothing\.proj_out",) -> ValueError ending ": nothing\.proj_out"
    # split_linears=(r"single_transformer_blocks\.\d+\.proj_mlp",) -> match "proj_mlp: a split linear needs K > N"
    # both: nothing under tmp_path but the source (no temporary sibling)

@pytest.mark.parametrize("model", ["flux", "hunyuan_video"])
def test_the_split_law_is_what_the_neuron_loader_slices(model, tmp_path):
    # write_tiny_flux(tmp_path, config=TINY_FP8) / write_tiny_hv1(tmp_path, config=TINY_FP8_HV1);
    # write_quantized_dir with targets=(proj_mlp,) + split_linears=SPLIT (halves unquantized);
    # for name, sl in single_block_proj_out_slices(num_single, heads*head_dim, 4*heads*head_dim).items():
    #     torch.equal(stored[name], _read_checkpoint_slice(src_map[sl.key], sl, shard=None, tp_size=1, tp_rank=0))
    # and every single block's proj_out.bias is stored, no proj_out.weight
```

  In `test_spec.py`:
  `test_split_linears_stay_out_of_every_unsplit_spec`. It asserts that `default_spec(m, tp_degree=4)`
  for `m in ("qwen_image", "ltx_2", "wan")` has `split_linears == ()`, no `"split_linears"` in
  `to_dict()` and no `"quant_split_linears"` in `weights_identity()`. For a spec with
  `split_linears=SPLIT` it asserts the round-trip through `to_dict`/`from_dict`, that
  `is_split("transformer.single_transformer_blocks.3.proj_out")` holds and
  `is_split("single_transformer_blocks.3.proj_out_attn")` does not, that its identity differs
  from `dataclasses.replace(spec, split_linears=())` while its `cache_fields()` are equal, that
  `unmatched_splits([...proj_out])` is `[]`, and that `split_linears=("(",)` raises `re.error`.
- [x] **Step 2: Run** — `T tests/unit/quant/test_checkpoint_split.py tests/unit/quant/test_spec.py`; FAIL (`split_linears` unknown).
- [x] **Step 3: Implement** as the Interfaces say. `quantize_tensors` yields, for a split
  source, each half as `(f"{part}.weight", q, True)` + `(f"{part}.weight_scale", s, True)` via
  `mx.quantize_mx_sharded(half, spec.k_shards(part))`, or as `(f"{part}.weight", half, False)`
  with `half` the source tensor's column range, uncast. The K refusal names the part.
- [x] **Step 4: Run** — PASS; then `T tests/unit/quant tests/unit/backends/test_neuron_flux_fp8.py tests/unit/backends/test_neuron_qwen_fp8.py tests/unit/backends/test_neuron_wan_fp8.py tests/unit/backends/test_neuron_ltx2_fp8.py`. The literal digest test passes unchanged.
- [x] **Step 5: Commit** — `feat(quant): a quantized checkpoint stores a split linear's column ranges as tensors of their own (QuantSpec.split_linears: the single-stream proj_out of FLUX and HunyuanVideo as proj_out_attn / proj_out_mlp, each quantized per rank K range or kept in the source's dtype); a spec without splits writes the same bytes at the same digest`; push.

### Task D2: FLUX's spec splits; the halves are override targets; the base and the CPU reference read them

**Files:**
- Modify: `difflet/quant/targets.py:86-119`. Rewrite the FLUX comment: the halves are stored
  as tensors of their own and can be quantized, and they are override targets until the device
  picks the default (D5). Add
  `FLUX_SPLIT_TARGETS: tuple[str, ...] = (rf"{_SINGLE}\.proj_out_(attn|mlp)",)` after
  `FLUX_TARGETS`, and `*FLUX_SPLIT_TARGETS` to `FLUX_ROW_PARALLEL` with the existing comment
  ("not default targets; kept for overrides"). `FLUX_TARGETS` and `FLUX_KEPT_BF16` do not change.
  In `:196-229` add `SPLITS_BY_MODEL: dict[str, tuple[str, ...]] = {"flux": SINGLE_BLOCK_SPLIT_LINEARS}`
  and `splits_for(model) -> tuple[str, ...]` (`.get(model, ())`); `default_spec` calls
  `overrides.setdefault("split_linears", splits_for(model))`. Update `__all__`.
- Modify: `difflet/quant/mx.py:46-53` (`_tile_sizes` docstring). "Qwen-Image, FLUX and LTX-2
  have no remainder" becomes: "Qwen-Image's and LTX-2's default targets have no remainder.
  FLUX's `proj_out_attn` has K 768 = 512 + 256 at tp4 (a remainder the two loaders agree on) and
  384 at tp8 (this law)."
- Modify: `difflet/quant/fake_linear.py:1-12` (docstring: a split linear runs as its halves),
  `:64-75` (`fake_quantize_model_`), `:78` (`__all__`). Add `split_linear_columns` and
  `SplitLinear` before `fake_quantize_model_`.
- Modify: `difflet/backends/neuron/core/application_base.py:310`. Under `self.quant`, keep only
  the slices with `not self.quant.is_split(s.key.removesuffix(".weight"))`. That runs before
  `extra_key`, so a fully split FLUX directory has no slice key in its weight-store key.
- Modify: `difflet/backends/neuron/flux/transformer.py:9-14, 52-57` (docstrings: the quantized directory holds the halves, read by name)
- Modify: `tests/unit/backends/_neuron_flux_toy.py` (`TINY_FP8_TP2 = {**TINY, "attention_head_dim": 64, "axes_dims_rope": (16, 24, 24)}`), `tests/unit/backends/_neuron_workers.py:1432` (`flux_dit_fp8_worker` after `flux_dit_worker`)
- Test: `tests/unit/quant/test_spec.py:144-171`, `tests/unit/quant/test_fake_linear.py:60-72`, `tests/unit/backends/test_neuron_flux_fp8.py`

**Interfaces:**
- Consumes D1's `split_linears`, `is_split`, `SINGLE_BLOCK_SPLIT_LINEARS` and the writer.
- Produces `difflet.quant.fake_linear.split_linear_columns(linear: nn.Linear) -> tuple[nn.Linear, nn.Linear]`.
  With `N = out_features` and `K = in_features`, it returns the attention half
  `nn.Linear(N, N)` (weight `W[:, :N]`, the bias) and the MLP half `nn.Linear(K - N, N, bias=False)`
  (weight `W[:, N:]`), cloned in the source's dtype and device. When `K <= N` it raises
  `ValueError(f"a split linear needs K > N (its first N input columns are the attention half), got [{N}, {K}]")`.
- Produces `SplitLinear(attn: nn.Module, mlp: nn.Module, split: int)` (an `nn.Module`):
  `forward(x) = attn(x[..., :split]) + mlp(x[..., split:])`.
- `fake_quantize_model_(model, spec)`: for each `nn.Linear` at `name` with `spec.is_split(name)`,
  half `h` in `("attn", "mlp")` becomes `FakeQuantMXLinear.from_linear(half, k_shards=spec.k_shards(f"{name}_{h}"))`
  when `spec.matches(f"{name}_{h}")`, and stays the plain half otherwise. The module becomes
  `SplitLinear(attn, mlp, split=N)`. Unsplit targets are replaced as before. The function
  returns the quantized names in module order (a split module contributes its quantized part
  names). It raises `ValueError(f"target patterns match no nn.Linear in the model: {'; '.join(dead)}")`
  with `dead = spec.unmatched_targets(<every nn.Linear name, a split one as its two part names>)`,
  before replacing anything. So a halves target on diffusers' fused FLUX model under a spec
  without splits is refused, not silently skipped. The existing
  `match="match no nn.Linear"` of `test_fake_linear.py:71-72` still holds.
- `flux_dit_fp8_worker(rank: int, world_size: int, quantized_dir: str, spec: dict) -> dict`.
  It builds `NeuronFluxConfig.from_pretrained(quantized_dir, tp_degree=world_size, **TINY_SHAPE)`
  and `NeuronFluxTransformerApplication(..., dtype=torch.float32, exec_mode="compile", device="cpu", quant=spec)`,
  and returns `{"out": list, "report": dict, "attn_twin": [type name, skip_bias_add, reduce_output]}`
  of `single_transformer_blocks[0].proj_out_attn`.
- In `test_neuron_flux_fp8.py`, `_fake_quant_reference(transformer_dir: str, spec: QuantSpec, config) -> torch.Tensor`
  is the **weight-only** reference (dequant mode multiplies unquantized activations by the
  dequantized weight). It loads diffusers' `FluxTransformer2DModel` and walks every `nn.Linear`
  `name`. For `spec.is_split(name)`, it takes `attn, mlp = split_linear_columns(module)`, and
  for each half `h` with `spec.matches(f"{name}_{h}")` it sets
  `half.weight.data = mx.fake_quantize_mx_sharded(half.weight.data, spec.k_shards(f"{name}_{h}"))`.
  The module is then replaced by `SplitLinear(attn, mlp, split=module.out_features)`. Otherwise,
  when `spec.matches(name)`, it sets `weight.data = mx.fake_quantize_mx_sharded(weight.data, spec.k_shards(name))`.
  The forward runs on `tiny_inputs(config)`.
- Test constants in `test_neuron_flux_fp8.py`:
  `PRE_D` = the five pre-D FLUX patterns written literally (`FLUX_TARGETS` as of `917d087`),
  `WITH_HALVES = PRE_D + FLUX_SPLIT_TARGETS`, and
  `TINY_TARGETS_HALVES = TINY["num_layers"] * 10 + TINY["num_single_layers"] * 6`.
  `TINY_TARGETS` (the default set's count, `* 4`) does not change.

- [x] **Step 1: Write the failing tests.**
  In `test_spec.py`, the two FLUX lists stay as they are (the default does not quantize the
  halves). Add `test_flux_splits_its_proj_out_and_the_halves_are_row_parallel_override_targets`:
  `spec = default_spec("flux", tp_degree=4)` has `spec.split_linears == SINGLE_BLOCK_SPLIT_LINEARS`,
  `spec.is_split("single_transformer_blocks.0.proj_out")` and
  `not spec.matches("single_transformer_blocks.0.proj_out_attn")`, and
  `spec.k_shards("single_transformer_blocks.0.proj_out_attn") == spec.k_shards("single_transformer_blocks.37.proj_out_mlp") == 4`.
  With `d = default_spec("flux", tp_degree=4, targets=FLUX_TARGETS + FLUX_SPLIT_TARGETS)`,
  `d.matches` holds for `single_transformer_blocks.0.proj_out_attn` and
  `single_transformer_blocks.37.proj_out_mlp` and not for `single_transformer_blocks.0.proj_out`.
  `default_spec("qwen_image").split_linears == ()`.

  In `test_fake_linear.py` (the toy has `single_transformer_blocks.0.proj_out = nn.Linear(768, 256)`,
  so the halves have K 256 and 512, which tp2 splits into multiples of 128):

```python
SPLIT = SINGLE_BLOCK_SPLIT_LINEARS
HALVES = (r"single_transformer_blocks\.\d+\.proj_out_(attn|mlp)",)
P = "single_transformer_blocks.0"

def test_a_split_linear_runs_as_its_two_column_ranges_each_quantized_on_its_own():
    model = _SplitToy(); fused = model.single_transformer_blocks[0].proj_out
    attn_ref, mlp_ref = split_linear_columns(fused)
    spec = QuantSpec(targets=HALVES, row_parallel=HALVES, split_linears=SPLIT, tp_degree=2)
    assert fake_quantize_model_(model, spec) == [f"{P}.proj_out_attn", f"{P}.proj_out_mlp"]
    m = model.single_transformer_blocks[0].proj_out
    assert isinstance(m, SplitLinear) and m.attn.k_shards == m.mlp.k_shards == 2  # the halves' names
    assert m.mlp.bias is None and torch.equal(m.attn.bias, fused.bias)
    x = torch.randn(3, 768)
    expect = (FakeQuantMXLinear.from_linear(attn_ref, k_shards=2)(x[:, :256])
              + FakeQuantMXLinear.from_linear(mlp_ref, k_shards=2)(x[:, 256:]))
    assert torch.equal(m(x), expect)
    # only the MLP half targeted: the attention half stays the plain column range
    model = _SplitToy(); spec = QuantSpec(targets=(r"single_transformer_blocks\.\d+\.proj_out_mlp",), split_linears=SPLIT)
    fake_quantize_model_(model, spec)
    assert type(model.single_transformer_blocks[0].proj_out.attn) is nn.Linear
    # a halves target on the fused model without the split, and a dead pattern beside live ones
    with pytest.raises(ValueError, match="match no nn.Linear"):
        fake_quantize_model_(_SplitToy(), QuantSpec(targets=HALVES))
    with pytest.raises(ValueError, match=r"match no nn.Linear in the model: nothing$"):
        fake_quantize_model_(_SplitToy(), QuantSpec(targets=(*HALVES, "nothing"), split_linears=SPLIT))
    with pytest.raises(ValueError, match="needs K > N"):
        split_linear_columns(nn.Linear(128, 256))
```

  In `test_neuron_flux_fp8.py`, the default fixture `tiny` keeps `default_spec("flux", compute="dequant", tp_degree=1)`
  and its reference becomes `_fake_quant_reference`. A second module fixture, `tiny_halves`,
  quantizes the same toy with `default_spec("flux", compute="dequant", tp_degree=1, targets=WITH_HALVES)`.
  The application test against `tiny` keeps its asserts
  (`type(single.proj_out_attn).__name__ == "RowParallelLinear"`, an empty `load_report`). The
  first test below replaces `test_the_quantized_directory_holds_the_default_targets_only`
  (`:68-70`), and the others are new:

```python
def test_the_quantized_directory_holds_the_default_targets_and_the_halves_by_name(tiny):
    # manifest["tensors"]["quantized"] == TINY_TARGETS; per single block i: proj_out_attn/_mlp
    # .weight present in fp32 (the toy's dtype), torch.equal to the fused W[:, :N] / W[:, N:],
    # no .weight_scale for them; proj_out.bias present; proj_out.weight absent

def test_the_quantized_directory_holds_the_halves_not_the_fused_weight(tiny_halves):
    # per single block i: proj_out_attn/_mlp .weight fp8 + .weight_scale present, proj_out.bias
    # present, proj_out.weight absent; manifest["tensors"]["quantized"] == TINY_TARGETS_HALVES

def test_application_quantizes_the_halves_and_matches_the_reference(tiny_halves):
    # single.proj_out_attn and single.proj_out_mlp are FP8RowParallelLinear;
    # single.proj_out_attn.skip_bias_add is True; len(app.quantized_linears) == TINY_TARGETS_HALVES;
    # load_report == {"missing": [], "unexpected": []}; output close (1e-4) to tiny_halves.reference

def test_a_target_set_without_the_halves_loads_them_unquantized_by_name(tiny, tmp_path):
    spec = default_spec("flux", compute="dequant", tp_degree=1, targets=PRE_D)
    assert spec.split_linears == SINGLE_BLOCK_SPLIT_LINEARS
    # prepare its directory (create=True) under tmp_path; load; load_report == {"missing": [], "unexpected": []};
    # type(single.proj_out_attn).__name__ == "RowParallelLinear";
    # torch.equal(single.proj_out_attn.weight, fused W[:, :N]); output close (1e-4) to
    # _fake_quant_reference(transformer_dir, spec, config)

@pytest.mark.multirank
def test_tp2_quantized_halves_match_the_per_rank_reference(tmp_path):
    # transformer_dir = write_tiny_flux(tmp_path, config=TINY_FP8_TP2)
    spec = default_spec("flux", compute="dequant", tp_degree=2, targets=WITH_HALVES)
    # quantized = prepare_quantized_checkpoint(transformer_dir, spec, create=True, cache_root=tmp_path / "cache")
    reference = _fake_quant_reference(transformer_dir, spec, NeuronFluxConfig.from_pretrained(transformer_dir, **TINY_SHAPE))
    results = run_ranks(flux_dit_fp8_worker, str(quantized), spec.to_dict(), world_size=2)
    for r in results:
        assert r["report"] == {"missing": [], "unexpected": []}
        assert r["attn_twin"] == ["FP8RowParallelLinear", True, False]
        torch.testing.assert_close(torch.tensor(r["out"]), reference, atol=1e-4, rtol=1e-4)
```

- [x] **Step 2: Run** — `T tests/unit/quant/test_spec.py tests/unit/quant/test_fake_linear.py tests/unit/backends/test_neuron_flux_fp8.py`; FAIL (`FLUX_SPLIT_TARGETS` / `split_linear_columns` unknown; then the loader asks for `proj_out.weight`).
- [x] **Step 3: Implement.**
- [x] **Step 4: Run** — PASS; then `T tests/unit/quant tests/unit/backends/test_neuron_flux_dit.py tests/unit/backends/test_neuron_flux_pipeline.py tests/unit/backends/test_neuron_fp8_linear.py tests/unit/cli`.
- [x] **Step 5: Commit** — `feat(flux): the single-stream blocks' proj_out can run in MXFP8 — FLUX's quantized directory stores it as its attention and MLP halves, read by name (the application base drops every slice its spec splits), each half per rank K range when targeted; FLUX_SPLIT_TARGETS are row-parallel override targets until the device picks the default (D5); the CPU reference layer runs a split linear as its halves`; push.

### Task D3: `difflet.ops.apply_rotary_emb_fp32`, and FLUX's rotary through it

**Files:**
- Modify: `difflet/backends/_torch_megatron/embeddings.py:13-24`. Add the reference
  `apply_rotary_emb_fp32`, whose body is diffusers' lines verbatim:
  `x.reshape(*x.shape[:-1], -1, 2).unbind(-1)`, `torch.stack([-x_imag, x_real], dim=-1).flatten(3)`,
  `(x.float() * cos + x_rotated.float() * sin).to(x.dtype)`. It takes a 4-D `x`.
- Modify: `difflet/backends/cpu/ops_impl/embeddings.py`, `trainium/ops_impl/embeddings.py`, `tpu/ops_impl/embeddings.py`, `cuda/ops_impl/embeddings.py` (re-export the reference, `__all__`; the `rms_norm_fp32` precedent)
- Modify: `difflet/backends/neuron/ops_impl/embeddings.py:19-43` (own `apply_rotary_emb_fp32`; a docstring paragraph that states the bf16 exactness and the fp32 auto-cast caveat below)
- Modify: `difflet/ops/__init__.py:18` (`"apply_rotary_emb_fp32": ("embeddings", "apply_rotary_emb_fp32")`), `difflet/ops/embeddings.py:4-5` (wrapper)
- Modify: `difflet/layers/embeddings.py:30-34` (import `apply_rotary_emb_fp32` from `difflet.ops`), `:66-79` (the `use_real_unbind_dim == -1` branch returns `apply_rotary_emb_fp32(x, cos, sin)` after the existing unbind / `[None, None]` / `.to(x.device)`; the `-2` and complex branches unchanged)
- Test: `tests/unit/backends/test_neuron_rotary.py`, `tests/unit/backends/test_neuron_ops_c2.py:69-75`

**Interfaces:**
- Produces `apply_rotary_emb_fp32(hidden_states, freqs_cos, freqs_sin) -> Tensor` on every
  backend. The tables are already broadcastable to `hidden_states`. The neuron body:

```python
idx = torch.arange(d, device=x.device)
sign = torch.where(idx % 2 == 0, -1.0, 1.0)
rot = ((idx[:, None] == (idx ^ 1)[None, :]).to(torch.float32) * sign[None, :]).to(x.dtype)
return (x.float() * freqs_cos + (x @ rot).float() * freqs_sin).to(x.dtype)
```

  `x @ rot` has one ±1 term per column. It is exact for bf16 x, which FLUX and HV1.5 feed on
  the device: the tensor engine multiplies bf16 by ±1, accumulates in fp32 and rounds a value
  that is already bf16. An fp32 x (a debug run) is exact only if neuronx-cc does not auto-cast
  the matmul to bf16. The neuron backend passes no `--auto-cast` flag, and the compiler's
  default casts fp32 matmuls, so the docstring says so instead of claiming exactness for fp32.

- [x] **Step 1: Write the failing tests** (`test_neuron_rotary.py`):

```python
from diffusers.models.embeddings import apply_rotary_emb as diffusers_rope
from difflet.backends._torch_megatron.embeddings import apply_rotary_emb_fp32 as reference_rope_fp32
from difflet.backends.neuron.ops_impl.embeddings import apply_rotary_emb_fp32 as neuron_rope_fp32

@pytest.mark.parametrize("op", [neuron_rope_fp32, reference_rope_fp32], ids=["neuron", "reference"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("table_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("sequence_dim", [1, 2])
@pytest.mark.parametrize("d", [16, 128])
def test_fp32_rope_is_bit_identical_to_diffusers(op, dtype, table_dtype, sequence_dim, d):
    # x [2, 3, 7, d] (sequence_dim 2) or [2, 7, 3, d] (1); cos, sin = torch.randn(7, d) in table_dtype
    # (not pair-equal); tables broadcast as diffusers does ([None, None] / [None, :, None])
    expect = diffusers_rope(x, (cos, sin), sequence_dim=sequence_dim)
    out = op(x, c, s)
    assert out.dtype == dtype and torch.equal(out, expect)

def test_every_other_backend_exports_the_reference_fp32_rope():
    for b in ("cpu", "cuda", "tpu", "trainium"):  # the Trainium AoT FLUX graph is the old one
        assert importlib.import_module(f"difflet.backends.{b}.ops_impl.embeddings").apply_rotary_emb_fp32 is reference_rope_fp32

def test_flux_rotary_reaches_the_neuron_op_on_the_neuron_backend():
    # subprocess, env DIFFLET_BACKEND=neuron (test_neuron_flux_split.py's pattern):
    # assert difflet.layers.embeddings.apply_rotary_emb_fp32 is neuron.apply_rotary_emb_fp32;
    # bf16 x [1, 3, 7, 16], bf16 cos/sin [7, 16]:
    # torch.equal(E.apply_rotary_emb(x, torch.stack([cos, sin], -1)), diffusers_rope(x, (cos, sin)))
```

  In `test_neuron_ops_c2.py`, add `load_backend_attr("embeddings", "apply_rotary_emb_fp32") is neuron_embeddings.apply_rotary_emb_fp32`.
- [x] **Step 2: Run** — `T tests/unit/backends/test_neuron_rotary.py tests/unit/backends/test_neuron_ops_c2.py`; FAIL (no `apply_rotary_emb_fp32`).
- [x] **Step 3: Implement.**
- [x] **Step 4: Run** — PASS. Then `T tests/unit/layers/test_embeddings.py tests/unit/ops tests/unit/backends/test_cuda_import_hygiene.py tests/unit/backends/test_torch_megatron.py tests/unit/backends/test_tpu_layers.py tests/unit/backends/test_neuron_flux_dit.py tests/unit/backends/test_neuron_flux_fp8.py tests/unit/backends/test_neuron_flux_split.py`.
  The gloo FLUX tests run under `DIFFLET_BACKEND=neuron`, i.e. on the new op, against diffusers.
- [x] **Step 5: Commit** — `perf(neuron): FLUX's rotary (diffusers' fp32 interleaved law) swaps each pair by a signed-permutation matmul on the neuron backend — bit-identical on the CPU; every other backend exports the reference, so the Trainium AoT graph is unchanged (kept on the device only if D5's bf16 gate passes)`; push.

### Task D4: HunyuanVideo-1.5's processors rotate with the neuron op

**Files:**
- Modify: `difflet/backends/neuron/hunyuan_video15/transformer.py:204-216`. `_joint_qkv` imports
  `apply_rotary_emb_fp32` from `difflet.backends.neuron.ops_impl.embeddings` instead of
  diffusers' `apply_rotary_emb`. It applies
  `cos, sin = (t[None, :, None, :].to(q.device) for t in image_rotary_emb)` and rotates q and k
  with them (`sequence_dim=1`'s broadcast). Update the docstring.
- Test: `tests/unit/backends/test_neuron_hv15_dit.py`

- [x] **Step 1: Write the failing test:**

```python
def test_the_processors_rotate_with_the_neuron_op(tiny, monkeypatch):
    reference = _reference(tiny)  # diffusers' own forward, before the patch
    import diffusers.models.embeddings as E
    monkeypatch.setattr(E, "apply_rotary_emb", lambda *a, **k: pytest.fail("diffusers' rotary on the neuron path"))
    app = NeuronHunyuanVideo15TransformerApplication(model_path=tiny, dtype=torch.float32,
                                                     exec_mode="eager", device="cpu", **SHAPE)
    app.load()
    torch.testing.assert_close(app(**_inputs())[0], reference, atol=1e-4, rtol=1e-4)
```

- [x] **Step 2: Run** — `T tests/unit/backends/test_neuron_hv15_dit.py -k neuron_op`; FAIL (the call-time import hits the patch).
- [x] **Step 3: Implement.**
- [x] **Step 4: Run** — `T tests/unit/backends/test_neuron_hv15_dit.py tests/unit/backends/test_neuron_hv15_pipeline.py`; PASS (every layout against diffusers unchanged).
- [x] **Step 5: Commit** — `perf(neuron): HunyuanVideo-1.5's attention processors rotate q/k with the neuron rotary op instead of diffusers' stride-2 unbind/stack — bit-identical on the CPU (measured on the device in phase F6 against the pre-D4 tree, reverted there if slower)`; push.

  D4 lands without a device run on purpose. HunyuanVideo-1.5 is not on this host until F6
  Step 1 downloads it (with a 16.7 GB bf16 copy), and F6 Steps 2–3 already run the two arms
  this needs (`bf16_pre` on the tree before D4, `bf16` on HEAD). The accept-or-revert rule is
  under "After phase D".

### Task D5: FLUX.1-dev on the device (tp4, 1024x1024, 50 steps, seed 42)

**Done 2026-11-07** (evidence section 6). Deviation from Step 6: the single-image screen fails every set at 50 steps, bf16's own tp2 floor included (29.4 dB), so the bisection would have ranked noise; the 12-image confirmation ran for B1, B0, bf16 and the floor instead, and B1 (0.82x, no more drift than B0) became the default (`3d7dd6c`).

No code unless a step fails, in which case it is one bug, one commit (`difflet-device-verify`).
The exception is the default-set commit of Step 7.

```bash
export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:$PATH
PY=/home/ubuntu/venvs/native-fp8/bin/python; W=/home/ubuntu/fp8-bench/flux_d_tp4
HEAD_SRC=/home/ubuntu/Difflet/.claude/worktrees/mxfp8-video-models
BASE=$(cat $W/BASE)                                   # D1 Step 0: the pre-D commit, pinned
git -C $HEAD_SRC cat-file -e "$BASE^{commit}" && mkdir -p $W/src-base
git -C $HEAD_SRC archive "$BASE" | tar -x -C $W/src-base          # the pre-D tree
ARGS="--model-id black-forest-labs/FLUX.1-dev --backend neuron --height 1024 --width 1024 --cache-dir /home/ubuntu/difflet-cache-video"
# compile:  PYTHONPATH=<tree> $PY -m difflet.cli.main compile $ARGS --tp-degree <tp> [--quant mxfp8] > $W/<arm>_compile.log 2>&1
# generate (x2, a and b): ... generate $ARGS --tp-degree <tp> [--quant mxfp8] --steps 50 --seed 42 \
#   --prompt "a cat sitting on a bench" --output $W/<arm>_gen<x>.png --work-dir $W/work_<arm>_gen<x> > $W/<arm>_gen<x>.log 2>&1
# per run: grep -o "step median [0-9.]* ms (n=[0-9]*" ; md5sum <png>
# compare:  PYTHONPATH=$HEAD_SRC $PY $HEAD_SRC/scripts/compare_difflet_runs.py --ref-image $W/A1_gena.png --cand-image $W/<arm>_gena.png --out $W/<arm>_vs_A1.json
```

**Every arm, every time:**
- *Disk:* before every compile that writes a quantized directory (M0, B0, B1, each Vi), run
  `df -h /home/ubuntu` and require at least 30 GB available. A FLUX mxfp8 directory is about
  16–18 GB: about 6.5–8.2B of 11.9B parameters in fp8 plus block scales, the rest bf16. If the
  free space is short, delete finished arms' directories (M0's first, then any Vi's), never the
  hub snapshot.
- *Compile log:* the compile exits 0. `grep -c "matched no parameter\|Traceback\|NCC_"` is 0.
  A quant arm prints exactly one `[flux] MXFP8 transformer (nki): <path>` line. Record the
  path and `du -sh` of it.
- *Determinism:* md5(gena) == md5(genb). If not, stop: record both, re-run genb once. A repeat
  is a determinism bug (one bug, one commit), and no later arm runs until it is fixed.
- *Tables:* every comparison row carries F2's PSNR / SSIM / LPIPS beside it (the bf16 noise
  floor, Step 2).

| arm | tree | tp | flags / env |
|---|---|---|---|
| A0 | `$W/src-base` | 4 | bf16 (old rotary) |
| M0 | `$W/src-base` | 4 | `--quant mxfp8` (pre-D set, old rotary, fused `proj_out` sliced): the re-measure of 220 ms |
| A1 | `$HEAD_SRC` | 4 | bf16 (new rotary) |
| F2 | `$HEAD_SRC` | 2 | bf16 (noise floor against A1) |
| B0 | `$HEAD_SRC` | 4 | `--quant mxfp8` (the default after D2: pre-D set, halves unquantized by name, new rotary) |
| B1 | `$HEAD_SRC` | 4 | `--quant mxfp8`, `DIFFLET_QUANT_TARGETS="$D_SET"` (the candidate default) |
| V1..V6 | `$HEAD_SRC` | 4 | `--quant mxfp8`, `DIFFLET_QUANT_TARGETS=<Vi>` (Step 6 only) |
| gate | `$HEAD_SRC` | 4 | bf16 and the chosen set, 4 prompts x 3 seeds, one generate each (Step 7 only) |

`PRE_D='transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v|add_q_proj|add_k_proj|add_v_proj);transformer_blocks\.\d+\.(ff|ff_context)\.net\.0\.proj;transformer_blocks\.\d+\.(ff|ff_context)\.net\.2;single_transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v);single_transformer_blocks\.\d+\.proj_mlp'`
(`FLUX_TARGETS`, pattern for pattern and in order).
`D_SET="$PRE_D;single_transformer_blocks\.\d+\.proj_out_(attn|mlp)"`
(`FLUX_TARGETS + FLUX_SPLIT_TARGETS`, so a D-set default has B1's targets and reuses its
directory and NEFFs).
The groups for Step 6 are:
G1 `transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v|add_q_proj|add_k_proj|add_v_proj)`;
G2 `transformer_blocks\.\d+\.(ff|ff_context)\.net\.0\.proj;transformer_blocks\.\d+\.(ff|ff_context)\.net\.2`;
G3 `single_transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v)`;
G4 `single_transformer_blocks\.\d+\.proj_mlp`;
G5 `single_transformer_blocks\.\d+\.proj_out_attn`;
G6 `single_transformer_blocks\.\d+\.proj_out_mlp`.
For i = 1..4, Vi is `D_SET` with Gi's patterns removed and the others kept in order.
V5 is `$PRE_D;single_transformer_blocks\.\d+\.proj_out_mlp` and V6 is
`$PRE_D;single_transformer_blocks\.\d+\.proj_out_attn`.

- [x] **Step 0: prerequisites.** Phase C's C6 Step 7 and C7 Step 5 have landed their sections
  in `docs/verification/2026-11-07-mxfp8-video-models.md`, and no Wan run is on the device.
  FLUX's section is the next free number: 6 if Wan 2.1 and 2.2 took 4 and 5. `$W/BASE` exists
  and `cat-file` succeeds. The CPU gate passes:
  `T tests/unit/quant tests/unit/backends/test_neuron_flux_fp8.py tests/unit/backends/test_neuron_flux_dit.py tests/unit/backends/test_neuron_rotary.py tests/unit/cli`.
  Run `df -h /home/ubuntu` and compare it with what C7 left (two Wan 2.2 bf16 copies, 56 GB,
  and its quantized directories).
- [x] **Step 1: pre-D baselines.** Compile and two generates each for A0 and M0. Record the
  md5s, the step medians, and M0/A0 against the history's 222.6 / 220 ms at 28 steps. After
  M0's generates, delete its quantized directory. It has the pre-D digest, which no later arm
  reads.
- [x] **Step 2: A1 and F2 (the rotary in bf16), and D3's gate.** Compile and two generates for
  A1 (tp4), then for F2 (tp2). Compare F2 against A1 (the floor) and A1 against A0.
  **Keep D3 only if** the mean of A1's two step medians is ≤ 1.01 × A0's, **and** either
  md5(A1) == md5(A0) or PSNR(A1 vs A0) ≥ PSNR(F2 vs A1). Otherwise stop and make one bug, one
  commit: revert D3's `difflet/layers/embeddings.py` reroute and its
  `test_flux_rotary_reaches_the_neuron_op_on_the_neuron_backend`. The op itself stays (D4 uses
  it). Then re-run A1. The evidence section names the decision and its numbers; B0 and B1
  then run on the rotary that was kept.
- [x] **Step 3: the kernel at FLUX's per-rank half shapes (one core).** Copy
  `/home/ubuntu/fp8-bench/ltx2_probe/probe_mx_shapes.py` to `$W/probe_mx_shapes_flux.py`. In
  the copy, set `OUT = Path("/home/ubuntu/fp8-bench/flux_d_tp4")` and `SHAPES` to
  `("proj_out_attn tp4 (row, K=3072/4 = 512+256)", 4608, 768, 3072)`,
  `("proj_out_mlp tp4 (row, K=12288/4)", 4608, 3072, 3072)` and
  `("proj_out_attn tp8 (row, K=3072/8 = one 384 tile)", 4608, 384, 3072)`. With no other run
  on the device, run
  `PROBE_WT=$HEAD_SRC PROBE_OUT=probe_mx_shapes_flux.json $PY $W/probe_mx_shapes_flux.py > $W/probe_mx_shapes_flux.log 2>&1`.
  The probe quantizes the per-rank K range with `mx.quantize_mx`, which is what
  `quantize_mx_sharded` gives each rank. Pass: every row is `"pass": true` (rel L2 < 3e-3,
  cosine > 0.9999, shape `[4608, 3072]`: the section 1 criterion). On a failure, stop and make
  one bug, one commit (that remainder's layout in `mx._tile_sizes`, or a writer refusal naming
  the layer). No arm that quantizes a half (B1, V1–V4, V5, V6) runs until the probe passes, so
  a layout error never reaches an end-to-end metric where it would read as quantization error.
- [x] **Step 4: B0 (the post-D2 default: rotary and the by-name halves on mxfp8).** Compile
  with `--quant mxfp8` and no env. Record the MXFP8 path, `du -sh` against the bf16 source
  (`du -shL <snapshot>/transformer`) and the `[weight store]` line's GB per rank. Run two
  generates. Record the step median, the ratio against A1, the comparison against A1, and B0
  against M0 (the rotary's effect on mxfp8, step and md5 / PSNR).
- [x] **Step 5: B1 (the D set).** Compile with `DIFFLET_QUANT_TARGETS="$D_SET"`, record what
  Step 4 records, and run two generates. Record the md5s, the step median, the ratio against A1
  and B0, and the comparison against A1.
- [x] **Step 6: bisection.** Run it when B1's step ≥ A1's, or B1's step > B0's, **or B1 misses
  any single-image gate against A1** (PSNR ≥ 30 dB, SSIM ≥ 0.97, LPIPS ≤ 0.05: the Qwen 6d /
  FLUX section 7 gates). When it does not run, B1 leads Step 7's screening only because it
  cleared all three gates in Step 5 and is faster than A1 and B0. For i = 1..6, compile Vi with
  `DIFFLET_QUANT_TARGETS=<Vi>` and run two generates. Record the step and the comparison
  against A1. After its generates, delete Vi's quantized directory (the path on its compile
  log's MXFP8 line).
- [x] **Step 7: the default, screened on one image and confirmed on twelve.**
  *Screening.* The candidates are the arms among B0, B1 and V1..V6 (those that ran) that clear
  all three single-image gates against A1. Rank them by the mean of their two step medians.
  Two candidates within 1% of each other are tied, and a tie goes to the smaller quantized
  directory. PSNR / SSIM / LPIPS only screen and never rank: section 7 found that single-seed
  numbers swing 6–10 dB between sets in no monotonic order, and the bisection arms differ by
  less than that. If no candidate clears the gates, the default stays B0's set (`FLUX_TARGETS`
  unchanged, the pre-D set), FLUX is memory-only in the docs, and every miss is recorded.
  *Confirmation (the 6h gate).* The top-ranked candidate X runs 4 prompts
  (`scripts/qwen_image_fp8_cpu_reference.py:37-42`) x 3 seeds (42, 7, 1234). That is one
  generate each for bf16 (A1's tree and flags) and for X (X's env), with
  `--output $W/gate/p<j>_s<seed>_{bf16,X}.png`. Each X image is compared against the bf16 image
  of the same prompt and seed. The cat / seed-42 cell is A1_gena against X_gena. Gate: the mean
  PSNR over the 12 images is ≥ 30 dB and every image is ≥ 27 dB. If X misses, drop it and
  confirm the next candidate (its bf16 cells reuse X's). If none passes, the default stays B0's
  set and the docs say FLUX is memory-only. If B0 itself was confirmed and missed, its 6h table
  goes into the docs as the recipe's quality caveat, with no code change. The gate runs on
  whatever set ends up as the default, B0 included.
  *Speed.* With X confirmed: if X's step is below A1's by more than 1%, the docs call FLUX's
  mxfp8 a speed-up; otherwise FLUX is memory-only.
  *Commit, when X is not B0.* `FLUX_TARGETS` becomes X's `DIFFLET_QUANT_TARGETS` list, pattern
  for pattern and in order, so its targets are the arm's. `FLUX_KEPT_BF16` loses the patterns X
  quantizes and gains any pre-D group X dropped. Also update the FLUX comment in `targets.py`,
  the two FLUX lists in `test_spec.py`, and `TINY_TARGETS` and its comment in
  `test_neuron_flux_fp8.py`. When X quantizes a half, the `--quant` help's bf16 clause
  (`difflet/cli/quant.py:41-46`) becomes "the output projections, modulation, embedders and
  the output layer stay bf16 (except FLUX's single-stream output projection, quantized as its
  attention and MLP halves; LTX-2: its audio stream too)". Do not touch the halves' tests
  (`PRE_D`, `WITH_HALVES`, `TINY_TARGETS_HALVES`, the gloo test, `test_fake_linear.py`): they
  name their sets literally. `SPLITS_BY_MODEL["flux"]` stays. Run
  `T tests/unit/quant tests/unit/backends/test_neuron_flux_fp8.py tests/unit/cli`. Then
  `compile $ARGS --tp-degree 4 --quant mxfp8` with no env must print X's MXFP8 path and reuse
  its NEFFs, which shows the default is what was measured. Commit
  `feat(quant): FLUX's default MXFP8 set is <set> — <cause> (<step> vs <A1> ms at tp4 1024x1024 50 steps; 6h gate mean <x> dB, min <y> dB over 4 prompts x 3 seeds)`; push.
- [x] **Step 8: evidence.** Add section `<n>` (Step 0's number) to
  `docs/verification/2026-11-07-mxfp8-video-models.md`. It holds: `BASE`'s SHA; the arm table
  (steps of both runs, md5s, ratios, PSNR / SSIM / LPIPS against A1 with F2 beside each, the
  directory sizes, the per-rank weight-store GB); D3's gate and its decision; the Step 3 probe
  table; the bisection table when it ran; the 6h table; and the conclusion. Update its status
  line. Close section 7 of `docs/verification/2026-10-31-qwen-image-fp8-native-trn3.md` with
  one line pointing to the new section. Rewrite the FLUX sentence in README's FP8 note and the
  FLUX paragraph of `docs/cli-staged-commands.md:209-214` to the measured outcome. Commit
  `docs: FLUX.1-dev in MXFP8 on Trainium3 revisited — single-block proj_out quantizable, neuron rotary; <ratio> at tp4 1024x1024 50 steps, <PSNR> dB from bf16 (noise floor <F2> dB), 6h gate <mean>/<min> dB`;
  push; update the status page.

## After phase D

- **E (HunyuanVideo)** adds its target table with `single_transformer_blocks\.\d+\.proj_out_(attn|mlp)`
  in its targets and row-parallel lists, and `SPLITS_BY_MODEL["hunyuan_video"] = SINGLE_BLOCK_SPLIT_LINEARS`.
  The base already drops its slices, and `TINY_FP8_HV1` is its toy. E6's reference reuses D2's
  `difflet.quant.fake_linear.split_linear_columns` / `SplitLinear`, so it does not re-derive
  the split law. Its rotary is already the backend op, and `quant` becomes an explicit kwarg
  (the `**overrides` collision). E's halves enter its default only by its own bisection rule,
  as FLUX's do here.
- **F (HunyuanVideo-1.5, memory-only)** measures D4 on the device: F6 Step 2 (`bf16_pre`, the
  tree before D4) against F6 Step 3 (`bf16`, HEAD). **Keep D4 only if** `bf16`'s step median is
  ≤ 1.01 × `bf16_pre`'s, **and** either `compare_difflet_runs.py` reports the latents
  `bit_identical` or the clip PSNR (`bf16` vs `bf16_pre`) is ≥ the noise floor's (`bf16_tp4` vs
  `bf16`). Otherwise revert
  D4 before F6 Step 4 (one bug, one commit), and record the decision in F's evidence section.
  F6 pins its pre-D4 tree by a commit grep. It should pin a SHA recorded when D4 lands, as D1
  Step 0 does for `BASE`.
- **G (the CPU fp32 arbiter)** depends on D2's split-aware `fake_quantize_model_`. On
  diffusers' fused FLUX / HunyuanVideo models it runs `proj_out` as its two halves, each
  fake-quantized with its own row-parallel shard count when the spec targets it. A target
  pattern that matches no linear there is refused, never skipped.
