# CUDA Backend PR 3 — Phases 2–4: Qwen-Image at tp=1 on one GPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Tasks 1–7 run on a CPU host (this Mac); Tasks 8–11 need the RTX Pro 6000 box. Do Tasks 1–7 while the box provisions.

**Goal:** `DIFFLET_BACKEND=cuda` generates a prompt-faithful Qwen-Image picture on one RTX PRO 6000 at `tp=1` through `DiffletPipeline`, with the DiT running through `difflet.ops` bound to the new CUDA backend, and passes the diffusers-relative parity gate. Release 1 of the spec.

**Architecture:** The CUDA backend copies the TPU backend's shape with the TPU-only parts removed: a `parallel_mesh` + `collectives` pair that is the identity at world size 1, Megatron layers lifted into a shared `difflet/backends/_torch_megatron/` package that both TPU and CUDA instantiate against their own collectives, attention on `F.scaled_dot_product_attention`, an application base whose `compile()` writes a manifest and whose `load()` builds the module eagerly on the device. Qwen-Image's DiT (`_QwenImageTransformerTraceModule`, already backend-neutral) runs through the CUDA ops; the text encoder and VAE are diffusers/transformers modules placed on the same GPU in the same process (D8); the existing `QwenImageOrchestrator` drives denoising and decoding.

**Tech Stack:** torch (CUDA 12.8 wheel on the box; CPU wheel on the Mac), diffusers 0.38.0 (`AutoencoderKLQwenImage`, `QwenImagePipeline.encode_prompt`, `FlowMatchEulerDiscreteScheduler`), transformers (`Qwen2_5_VLForConditionalGeneration`, `Qwen2Tokenizer`), accelerate (`init_empty_weights`), safetensors, pytest.

**Spec:** `docs/plans/2026-10-16-cuda-backend-support.md` — Decisions D1–D19, Global Constraints, Phases 0, 2, 3, 4. Conflicts resolve against the Decisions table. Facts below were read from the repo at HEAD `10a15be` (file:line references are to that commit).

## Global Constraints

- Nothing under `difflet/backends/cuda/` or `difflet/backends/_torch_megatron/` imports `neuronx_distributed`, `torch_neuronx`, `torch_xla`, `nkilib`, `difflet.backends.trainium`, or `difflet.backends.tpu` (`tests/unit/backends/test_cuda_import_hygiene.py` scans; Task 1 adds `_torch_megatron` to `SCANNED`).
- Every name in `difflet/ops/__init__.py:_EXPORTS` (40: linear 3, norm 3, attention 6, embeddings 1, mx 4, platform 2, collectives 21) resolves on the CUDA backend by the end of Task 4; the `xfail(strict=True)` markers in the hygiene test are removed as each module lands.
- World size 1 creates no process group; every collective is the identity, as the CPU backend. Multi-process paths raise `NotImplementedError("... follow-up F2")`.
- Reflective contract (`difflet_pipeline.py:281–318`): `compile(path, debug=...)` (debug passed only if in the signature), `has_compiled_artifacts(path) -> bool`, `load(path, start_rank_id=None, local_ranks_size=None, skip_warmup=False)` (only the kwargs present in the signature are passed), `__call__`. Pin with `inspect.signature` as `tests/unit/backends/test_tpu_application.py:32` does.
- TPU behaviour is unchanged by the lift: run `DIFFLET_BACKEND=cpu .venv/bin/python -m pytest tests/unit/backends/test_tpu_layers.py tests/unit/backends/test_tpu_application.py tests/unit/backends/test_tpu_checkpoint.py tests/unit/backends/test_tpu_collectives.py -q` before Task 1 and after; the pass/skip counts must be identical.
- Attention contract (from the TPU and CPU implementations): `scale=None` means **1.0**, not SDPA's default; `causal`, `attention_mask` (bool or additive), `bound_min`/`bound_max` (both or neither; mutually exclusive with a mask); `tp_q/tp_k/tp_out` accepted and ignored.
- bf16 only; MX ops raise (D9). SDPA only (D7). Eager (D4).
- Test command for every task: `.claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh`. GPU tasks additionally log to `/tmp/logs/<what>-<ts>.log` on the box and write numbers to the spec's Status log.
- Black, line length 100. Tests mirror the source tree. One task, one commit (a few commits is fine; BASE is from `task-start`). The branch stays local; no push, no PR, unless the owner says so.

## Review Focus

1. `RowParallelLinear` on TPU silently swallows a `reduce_output=` kwarg that `modeling_qwen_image.py:156–165` passes; the lifted layer must keep accepting it (compat) but a reviewer should know the kwarg does nothing. (Task 1 test `test_row_parallel_accepts_and_ignores_reduce_output`.)
2. `attention(scale=None)` must mean 1.0; a CUDA implementation that forwards `scale=None` to SDPA silently applies `1/sqrt(d)` and every Qwen attention block is wrong by a constant. (Task 4 test `test_attention_scale_none_means_one_not_sdpa_default`.)
3. A bool mask and a causal flag together must AND the mask with the causal triangle, and an additive mask must use `finfo.min`, not `-inf` (NaN rows for fully masked queries). (Task 4 tests `test_attention_bool_mask_and_causal_combine`, `test_attention_additive_mask_uses_finfo_min`.)
4. `load()` on CUDA must put *every* parameter and buffer on the device; a buffer left on CPU (RoPE tables are buffers) fails only at forward time with a device-mismatch error deep in diffusers. (Task 5 test `test_load_moves_all_params_and_buffers_to_device`.)
5. `has_compiled_artifacts()` must be False for a manifest written by a different mesh or dtype, or a cached CUDA "artifact" silently serves a different configuration. (Task 5 test `test_has_compiled_artifacts_rejects_other_mesh_or_dtype`.)

---

### Task 1: Lift the Megatron layers, weights and checkpoint loader into `difflet/backends/_torch_megatron/`

**Files:**
- Create: `difflet/backends/_torch_megatron/__init__.py` (docstring: why shared, D10), `linear.py`, `norm.py`, `embeddings.py`, `weights.py`, `checkpoint.py`
- Modify: `difflet/backends/tpu/ops_impl/linear.py` (becomes a binding), `difflet/backends/tpu/ops_impl/norm.py`, `embeddings.py` (re-exports), `difflet/backends/tpu/core/weights.py`, `checkpoint.py` (re-exports)
- Modify: `tests/unit/backends/test_cuda_import_hygiene.py:29` — `SCANNED.append(Path("difflet/backends/_torch_megatron"))`
- Test: `tests/unit/backends/test_torch_megatron.py`

**Interfaces:**
- Produces: `difflet.backends._torch_megatron.linear.build_parallel_layers(collectives) -> types.SimpleNamespace` with attributes `ColumnParallelLinear`, `RowParallelLinear`, `ParallelEmbedding` — classes created per call, closed over `collectives`, which must provide `gather_tp_dim(tensor, *, dim)`, `scatter_tp_dim(tensor, *, dim)`, `reduce_tp(tensor)`, `get_tp_size() -> int`, `get_tp_rank() -> int`. Constructor signatures and `_difflet_shard` values exactly as `difflet/backends/tpu/ops_impl/linear.py:84–93, 112, 134–143, 161, 185–194, 211, 223` (bodies move verbatim; the only edit replaces the module-level collectives import with the closed-over object). `RowParallelLinear.__init__` keeps `**kwargs` and documents that `reduce_output` is accepted and ignored.
- Produces: `_torch_megatron.weights` = `shard_dim`, `narrow_to_rank`, `shard_state_dict`, `load_sharded_state_dict`, `materialize_meta_` (verbatim from `tpu/core/weights.py:23–156`); `_torch_megatron.checkpoint` = `INDEX_SUFFIX`, `build_weight_map`, `load_checkpoint_into` (verbatim from `tpu/core/checkpoint.py`, importing `shard_dim` from `_torch_megatron.weights`); `_torch_megatron.norm` = `LayerNorm`, `RMSNorm`, `CustomRMSNorm`; `_torch_megatron.embeddings` = `apply_rotary_emb` (verbatim from the TPU files, which are byte-identical to CPU's).
- TPU after the lift: `tpu/ops_impl/linear.py` is `from difflet.backends.tpu.ops_impl import collectives; _L = build_parallel_layers(collectives); ColumnParallelLinear = _L.ColumnParallelLinear; ...; __all__` unchanged. `tpu/core/weights.py` and `checkpoint.py` re-export with `# noqa: F401`.

- [x] **Step 1: Record the TPU baseline** — Run: `DIFFLET_BACKEND=cpu .venv/bin/python -m pytest tests/unit/backends/test_tpu_layers.py tests/unit/backends/test_tpu_application.py tests/unit/backends/test_tpu_checkpoint.py tests/unit/backends/test_tpu_collectives.py -q -p no:cacheprovider 2>&1 | tail -1` — Expected: a line of the form `N passed, M skipped`; write N and M into the ledger.
- [x] **Step 2: Write the failing tests** — `tests/unit/backends/test_torch_megatron.py`:

```python
from types import SimpleNamespace
import torch, torch.nn.functional as F, pytest
from difflet.backends._torch_megatron.linear import build_parallel_layers

def _identity_collectives(tp_size=1, tp_rank=0):
    return SimpleNamespace(
        gather_tp_dim=lambda t, *, dim: t, scatter_tp_dim=lambda t, *, dim: t,
        reduce_tp=lambda t: t, get_tp_size=lambda: tp_size, get_tp_rank=lambda: tp_rank)

def test_layers_are_distinct_classes_per_collectives_module():
    a = build_parallel_layers(_identity_collectives()); b = build_parallel_layers(_identity_collectives())
    assert a.ColumnParallelLinear is not b.ColumnParallelLinear

def test_column_parallel_matches_plain_linear_at_tp1():
    L = build_parallel_layers(_identity_collectives())
    layer = L.ColumnParallelLinear(8, 16, bias=True); x = torch.randn(4, 8)
    assert torch.allclose(layer(x), F.linear(x, layer.weight, layer.bias))
    assert layer._difflet_shard == {"weight": 0, "bias": 0}

def test_row_parallel_shard_metadata_and_bias_after_reduce():
    calls = []
    c = _identity_collectives(); c.reduce_tp = lambda t: calls.append("reduce") or t
    L = build_parallel_layers(c); layer = L.RowParallelLinear(16, 8, bias=True, input_is_parallel=True)
    layer(torch.randn(4, 16))
    assert calls == ["reduce"] and layer._difflet_shard == {"weight": 1, "bias": None}

def test_row_parallel_accepts_and_ignores_reduce_output():
    L = build_parallel_layers(_identity_collectives())
    L.RowParallelLinear(16, 8, reduce_output=False)  # compat with modeling_qwen_image.py:156-165

def test_embedding_shard_metadata_follows_the_mode():
    L = build_parallel_layers(_identity_collectives())
    assert L.ParallelEmbedding(10, 4)._difflet_shard == {"weight": 0}
    assert L.ParallelEmbedding(10, 4, shard_across_embedding=True)._difflet_shard == {"weight": 1}

def test_weights_and_checkpoint_import_without_tpu():
    import sys
    from difflet.backends._torch_megatron import weights, checkpoint  # noqa: F401
    assert not any(k.startswith("difflet.backends.tpu") for k in sys.modules)
```

- [x] **Step 3: Run** — Run: `DIFFLET_BACKEND=cpu .venv/bin/python -m pytest tests/unit/backends/test_torch_megatron.py -q` — Expected: `ModuleNotFoundError: difflet.backends._torch_megatron`
- [x] **Step 4: Create the package** per Interfaces (move bodies verbatim; `git mv` is not required since TPU files keep existing as bindings/re-exports). Rewire the five TPU files. Append `_torch_megatron` to `SCANNED` in the hygiene test.
- [x] **Step 5: Run** — the new test file → PASS; the Step 1 command → identical N and M; `tests/unit/backends/test_cuda_import_hygiene.py` → `test_cuda_backend_has_no_forbidden_imports` PASS; gate → `regressions=0 new_skips=0`.
- [x] **Step 6: Commit** — `git commit -m "backends: lift Megatron layers, weights and checkpoint loader into _torch_megatron; TPU binds them"`

---

### Task 2: CUDA parallel mesh and collectives at world size 1

**Files:**
- Create: `difflet/backends/cuda/ops_impl/__init__.py` (docstring), `parallel_mesh.py`, `collectives.py`
- Test: `tests/unit/backends/test_cuda_collectives.py`

**Interfaces:**
- Produces `cuda/ops_impl/parallel_mesh.py`: same public surface as `tpu/ops_impl/parallel_mesh.py:40–154` (`mesh_spec_from_config`, `init_parallel_mesh`, `get_mesh_spec`, `axis_replica_groups`, `get_tp_groups`, `get_cfg_group`, `get_cp_group`, `get_cp_mesh`, `get_tp_size`, `get_tp_rank`, `get_axis_rank`, `destroy_parallel_mesh`, `__all__`), bodies copied, with `_runtime_world_size()` = `int(os.environ.get("WORLD_SIZE", "1"))` and `_runtime_ordinal()` = `int(os.environ.get("RANK", "0"))` instead of torch_xla. Keep the `tests/unit/test_no_dp_parasites.py` rule: no dp accessor.
- Produces `cuda/ops_impl/collectives.py`: every one of the 21 collectives names in `_EXPORTS` plus the mesh re-exports the TPU module makes (`tpu/ops_impl/collectives.py:38–50`). At `get_tp_size() == 1` (and for the sequence-parallel family, when the relevant group is trivial) each is the identity / local slice, matching `cpu/ops_impl/collectives.py` line for line in semantics. When the mesh says tp>1 or world>1, `gather_tp_dim`, `reduce_tp`, `scatter_to_sequence_parallel_region`, `gather_from_sequence_parallel_region`, `reduce_scatter_to_sequence_parallel_region`, `gather_from_tensor_model_parallel_region_with_dim`, `reduce_from_tensor_model_parallel_region` raise `NotImplementedError("CUDA multi-process collectives are follow-up F2")`. `scatter_tp_dim` / `scatter_to_process_group_spmd` stay pure narrows as on TPU. `SPMDRank`, `_WorldGroup`, `get_world_group`, `get_cp_rank_spmd`, `get_cfg_rank_spmd` copied from TPU with `_global_rank()` = `RANK` env. `__all__` lists the 21 + mesh names.

- [x] **Step 1: Write the failing tests** (mirror `test_tpu_collectives.py`; same autouse `_clean_mesh`):

```python
def test_mesh_not_initialized_is_an_explicit_error(): ...  # pm.get_tp_size() raises RuntimeError
def test_tp_only_mesh_groups(): ... # init MeshSpec(tp=2) → get_tp_groups() == [[0,1]]
def test_every_axis_covers_every_replica_exactly_once[axis]: ...
def test_tp_degree_one_collectives_are_identity(): ... # all 21 names on a tensor return it unchanged / correct slice
def test_public_surface_matches_the_ops_contract():
    from difflet.ops import _EXPORTS
    expected = {n for n, (m, _) in _EXPORTS.items() if m == "collectives"}
    assert expected <= set(C.__all__)
def test_multi_rank_collectives_raise_f2(): ... # MeshSpec(tp=2) + WORLD_SIZE=2 → reduce_tp raises NotImplementedError match "F2"
def test_import_does_not_require_torch_distributed_init(): ... # importing the module creates no process group: torch.distributed.is_initialized() is False
```

- [x] **Step 2: Run** — Expected: `ModuleNotFoundError: difflet.backends.cuda.ops_impl`
- [x] **Step 3: Implement** per Interfaces. - [x] **Step 4: Run** file → PASS; hygiene test: remove `xfail` for `collectives` (the parametrized marker becomes per-module: keep a `_PENDING = {...}` set and apply `pytest.mark.xfail` only to names in it); gate → clean.
- [x] **Step 5: Commit** — `git commit -m "cuda: parallel mesh and collectives at world size 1 (multi-rank raises F2)"`

---

### Task 3: CUDA linear, norm, embeddings

**Files:**
- Create: `difflet/backends/cuda/ops_impl/linear.py`, `norm.py`, `embeddings.py`
- Test: `tests/unit/backends/test_cuda_layers.py`

**Interfaces:**
- `cuda/ops_impl/linear.py`: `from difflet.backends.cuda.ops_impl import collectives; _L = build_parallel_layers(collectives)`; exports the three classes and `__all__`. `norm.py` / `embeddings.py`: re-export from `_torch_megatron`.

- [x] **Step 1: Write the failing tests** (mirror `test_tpu_layers.py:40–96, 148–163`): `test_column_parallel_matches_plain_linear_at_tp1`, `test_row_parallel_shards_the_input_dim_and_replicates_bias` (via mesh tp=2 metadata only, no collective call), `test_indivisible_shapes_are_rejected_at_construction`, `test_layers_require_an_initialized_mesh`, `test_embedding_matches_plain_embedding_at_tp1`, `test_rmsnorm_computes_variance_in_fp32`, `test_rotary_matches_the_cpu_reference_implementation` (compare against `difflet.backends.cpu.ops_impl.embeddings.apply_rotary_emb` on random inputs, `torch.equal`).
- [x] **Step 2: Run** → `ModuleNotFoundError`. - [x] **Step 3: Implement.** - [x] **Step 4: Run** → PASS; remove `linear`, `norm`, `embeddings` from the hygiene `_PENDING` set; gate clean.
- [x] **Step 5: Commit** — `git commit -m "cuda: linear/norm/embeddings bound from _torch_megatron"`

---

### Task 4: CUDA attention on SDPA, platform, mx

**Files:**
- Create: `difflet/backends/cuda/ops_impl/attention.py`, `platform.py`, `mx.py`
- Test: `tests/unit/backends/test_cuda_attention.py`
- Modify: `tests/unit/backends/test_cuda_import_hygiene.py` — `_PENDING` becomes empty; the parametrized test now asserts every module imports.

**Interfaces:**
- `attention(q, k, v, *, scale=None, causal=False, attention_mask=None, bound_min=None, bound_max=None, tp_q=False, tp_k=False, tp_out=False, **kwargs)` — same contract as `tpu/ops_impl/attention.py:233–285` minus flash/chunking: `scale = 1.0 if scale is None`; bounds → mask via a copy of `_bounds_to_mask` (`tpu:81`); bool mask AND causal triangle; additive mask with `finfo.min` for causal; then `F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=(causal and mask is None), scale=scale)`. Inputs may be `[B, S, D]` or `[B, H, S, D]` exactly as the CPU/TPU versions accept them (check `cpu/ops_impl/attention.py:8–49` for the reshape rule and keep it). `cross_attention` forces `tp_*=False`. `ring_attention`, `joint_ring_attention`, `ulysses_attention`, `joint_ulysses_attention` raise `NotImplementedError("... follow-up F2/F6")`.
- `platform.py`: `class hardware(Enum)` with `CPU, TRN1, TRN2, TPU, CUDA`; `get_platform_target() -> hardware.CUDA`; `compute_capability() -> tuple[int, int] | None` (None without CUDA); `distributed_backend_name() -> "nccl"`; `sdpa_kernel_report(q, k, v) -> str` naming which SDPA backend `torch.nn.attention` would pick (used by Phase 0.4).
- `mx.py`: the four names raise `NotImplementedError("<name>: MX is not implemented on the CUDA backend (D9)")`, mirroring `tpu/ops_impl/mx.py`.

- [x] **Step 1: Write the failing tests**: `test_attention_matches_cpu_reference_unmasked` (vs `difflet.backends.cpu.ops_impl.attention.attention`, `allclose` atol 1e-5 in fp32), `test_attention_scale_none_means_one_not_sdpa_default`, `test_attention_causal_masks_the_future`, `test_attention_bool_mask_and_causal_combine`, `test_attention_additive_mask_uses_finfo_min` (a fully masked query row yields finite output, no NaN), `test_attention_bounds_require_both_ends`, `test_attention_bounds_and_mask_are_mutually_exclusive`, `test_context_parallel_modes_fail_loudly[name]`, `test_platform_target_is_cuda`, `test_mx_ops_refuse`, `test_every_name_qwen_image_imports_resolves_on_cuda[name]` over the 13 names `modeling_qwen_image.py:29–43` imports.
- [x] **Step 2: Run** → `ModuleNotFoundError`. - [x] **Step 3: Implement.** - [x] **Step 4: Run** → PASS; hygiene test fully green (7/7 import, 0 xfail); gate clean.
- [x] **Step 5: Commit** — `git commit -m "cuda: attention on SDPA, platform, mx raises — the full difflet.ops surface resolves"`

---

### Task 5: `CudaApplicationBase` — manifest-only compile, eager load

**Files:**
- Create: `difflet/backends/cuda/core/__init__.py`, `application_base.py`
- Test: `tests/unit/backends/test_cuda_application.py`

**Interfaces:**
- `class CudaApplicationBase(torch.nn.Module)`: `__init__(self, config=None, *, device: str | torch.device | None = None)` (`device` defaults to `"cuda"`; tests pass `"cpu"`). Hooks: `build_module() -> nn.Module` (raises), `get_example_inputs() -> tuple` (raises), `get_state_dict() -> dict | None` (None). `_prepare_module()`: `build_module()` → `materialize_meta_(module, device=self.device, dtype=self.dtype)` if any meta params → optional `load_sharded_state_dict` at tp (world size 1: identity) → `.to(self.device)` → `.eval()`; asserts afterwards that every parameter and buffer is on `self.device`. Contract: `compile(self, compiled_model_path, debug: bool = False) -> None` writes `<path>/cuda_manifest.json` = `{"format": "eager", "world_size": 1, "mesh": {dp, cfg, cp, tp}, "dtype": str, "torch": torch.__version__, "cuda": torch.version.cuda}` (tmp + `os.replace`); `has_compiled_artifacts(self, compiled_model_path) -> bool` = manifest exists, parses, and its `mesh` and `dtype` equal this app's; `load(self, compiled_model_path, start_rank_id=None, local_ranks_size=None, skip_warmup: bool = False) -> None` validates the manifest (FileNotFoundError / ValueError on mismatch), runs `_prepare_module()`, sets `is_loaded`, calls `warmup()` unless `skip_warmup` (one forward on `get_example_inputs()` under `no_grad`); `forward(*args, **kwargs)` raises `RuntimeError("model is not loaded; call load() first")` before load, else delegates to the module. `MANIFEST_FILE_NAME = "cuda_manifest.json"`.

- [x] **Step 1: Write the failing tests** (mirror `test_tpu_application.py:32–110`): `test_signatures_match_what_the_pipeline_reflects_on`, `test_forward_before_load_is_an_explicit_error`, `test_compile_writes_manifest_and_has_compiled_artifacts_sees_it(tmp_path)`, `test_has_compiled_artifacts_is_false_without_a_manifest`, `test_has_compiled_artifacts_rejects_other_mesh_or_dtype`, `test_load_moves_all_params_and_buffers_to_device` (a `_Block` with a registered buffer; `device="cpu"`; assert via `all(p.device.type == "cpu")` — on the Mac the point is the assertion exists; it is re-run on the box with `device="cuda"` in Task 8), `test_load_then_forward_runs_the_module` (tiny synthetic `_Block` of `_torch_megatron` layers, `compile → has → load → __call__` round trip with `device="cpu"`).
- [x] **Step 2: Run** → `ModuleNotFoundError`. - [x] **Step 3: Implement.** - [x] **Step 4: Run** → PASS; gate clean. - [x] **Step 5: Commit** — `git commit -m "cuda: application base — manifest-only compile, eager load on device"`

---

### Task 6: Neutral Qwen-Image DiT config

**Files:**
- Create: `difflet/models/qwen_image/dit_config.py` — `@dataclass QwenImageDiTConfig`, the body of `difflet/backends/tpu/qwen_image/config.py:25–114` moved verbatim (fields, properties, `validate`, `from_pretrained`); it is pure (json, dataclasses, pathlib, torch).
- Modify: `difflet/backends/tpu/qwen_image/config.py` → `TpuQwenImageConfig = QwenImageDiTConfig` re-export (keeps `tpu_application.py:39` and `tests/unit/models/qwen_image/test_qwen_entry.py` working).
- Modify: `difflet/models/qwen_image/pipeline.py:18` → `from difflet.models.qwen_image.contract import QwenImageDiTInputBundle` (the type already lives in `contract.py:22`; the stale import is what keeps the pipeline module in the Trainium tier of `scripts/test_imports.sh` — move it back to `NEUTRAL` in the same commit).
- Test: `tests/unit/models/qwen_image/test_dit_config.py`

- [x] **Step 1: Write the failing tests** (mirror `tests/unit/backends/test_tpu_wan_config.py` shape): `test_latent_geometry_follows_vae_scale_factor` (1024×1024 → latent 128, packed 64, image_seq_len 4096), `test_tp_must_divide_the_attention_heads`, `test_axes_dims_rope_must_sum_to_head_dim`, `test_pixel_dims_must_be_aligned`, `test_from_pretrained_reads_a_diffusers_config(tmp_path)` (write a minimal `transformer/config.json`), `test_tpu_alias_is_the_same_class`.
- [x] **Step 2: Run** → `ModuleNotFoundError`. - [x] **Step 3: Implement**; move `difflet.models.qwen_image.pipeline` to the NEUTRAL tier in `scripts/test_imports.sh` and delete its "Phase 4 moves that type" comment. - [x] **Step 4: Run** → PASS; `tests/unit/models/qwen_image/test_qwen_entry.py` unchanged result; `DIFFLET_BACKEND=cpu PYTHON_BIN=.venv/bin/python ./scripts/test_imports.sh` → `ok import difflet.models.qwen_image.pipeline`; gate clean.
- [x] **Step 5: Commit** — `git commit -m "qwen_image: backend-neutral DiT config; pipeline imports the bundle from contract"`

---

### Task 7: `CudaQwenImageApplication` — DiT through difflet.ops, encoder and VAE on the same GPU

**Files:**
- Create: `difflet/backends/cuda/qwen_image/__init__.py`, `transformer.py`
- Create: `difflet/models/qwen_image/cuda_application.py`
- Modify: `difflet/models/qwen_image/entry.py:19,33` — accept `"cuda"`; `elif backend == "cuda": from difflet.models.qwen_image.cuda_application import CudaQwenImageApplication; return CudaQwenImageApplication(...)`; error text becomes "trainium, tpu and cuda" (update `test_qwen_entry.py:27` match).
- Modify: `difflet/registry.py:390` — `backends=("trainium", "tpu", "cuda")`.
- Test: `tests/unit/models/qwen_image/test_cuda_application.py`; extend `test_qwen_entry.py`.

**Interfaces:**
- `cuda/qwen_image/transformer.py`: `class CudaQwenImageTransformerApplication(CudaApplicationBase)`: `__init__(self, *, model_path, config: QwenImageDiTConfig, device=None)`; `_init_runtime()` = `parallel_mesh.init_parallel_mesh(MeshSpec(tp=config.tp_degree))`; `build_module()` = `_QwenImageTransformerTraceModule(self.config)` under `accelerate.init_empty_weights(include_buffers=False)` (as `tpu/qwen_image/transformer.py:52–55`); `_prepare_module()` = mesh → build → `materialize_meta_(dtype=config.torch_dtype)` → `load_checkpoint_into(module, model_path, tp_size=1, tp_rank=0, prefix="transformer.", dtype=config.torch_dtype, strict=True)` → `.to(device)` → `requires_grad_(False).eval()`; `get_example_inputs()` as `tpu:118`; `forward(hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_mask, guidance)`.
- `models/qwen_image/cuda_application.py`: `class CudaQwenImageApplication(torch.nn.Module)` with `__init__(self, *, model_path: str, parallel, dtype, shape: dict, **kwargs)` reading `text_seq_len` (default 1024), `batch_size` (1), `device` ("cuda"), `load_text_encoder` (True), `load_vae` (True). It builds `self.config = QwenImageDiTConfig.from_pretrained(<model_path>/transformer, height, width, text_seq_len, batch_size, tp_degree=parallel.tp_degree, torch_dtype=dtype)` when `transformer/config.json` exists (else `transformer=None`, as TPU), and `self.transformer = CudaQwenImageTransformerApplication(...)`. Contract methods delegate to the transformer: `compile(path, debug=False)`, `has_compiled_artifacts(path)`, `load(path, start_rank_id=None, local_ranks_size=None, skip_warmup=False)` — after loading the DiT, `load()` also loads, on the same device and dtype, `Qwen2_5_VLForConditionalGeneration` + `Qwen2Tokenizer` (subfolders `text_encoder`, `tokenizer`) and `AutoencoderKLQwenImage` (`vae`) and a `FlowMatchEulerDiscreteScheduler` (`scheduler`), then builds `self.pipeline = QwenImageOrchestrator(model_path=..., transformer=self, vae=self.vae, dtype=..., height=..., width=..., text_seq_len=..., scheduler=...)` (`pipeline.py:31–47`). `encode_prompt(prompt: str | list[str]) -> tuple[Tensor, Tensor]` uses a `QwenImagePipeline(scheduler, vae=None, text_encoder, tokenizer, transformer=None)` exactly as `scripts/qwen_image_cache_dit_inputs.py:128–160` and pads to `text_seq_len` with `pad_or_truncate_prompt_embeds` (copy the helper into `difflet/models/qwen_image/prompt.py` so the script and the app share it). `__call__`: a `QwenImageDiTInputBundle` → `forward_dit`; five keyword tensors → bundle; `prompt=` (str) with `num_inference_steps=50`, `seed=None`, `output_type="pil"` → encode → `self.pipeline(encoder_hidden_states=..., encoder_hidden_states_mask=..., num_inference_steps=..., generator=..., output_type="pt")` → to PIL via `diffusers.image_processor.VaeImageProcessor.postprocess` or `numpy_to_pil` → returns `QwenImagePipelineOutput`-like object with `.images`. Positional args go to the transformer.
- `forward_dit(bundle)` and `dit_input_contract()` mirror `tpu_application.py:108–131` so the parity script can call the DiT alone.

- [x] **Step 1: Write the failing tests** (CPU, no weights): `test_entry_builds_the_cuda_application` (a `tmp_path` with no `transformer/config.json` → application with `transformer is None`, no downloads), `test_entry_rejects_cfg_parallel_on_cuda`, `test_entry_error_lists_all_three_backends`, `test_registry_lists_cuda_for_qwen_image`, `test_cuda_application_signatures_match_the_pipeline_contract` (`inspect.signature` on compile/load/has_compiled_artifacts), `test_pad_or_truncate_prompt_embeds_pads_and_masks` (for the moved helper).
- [x] **Step 2: Run** → `ImportError` / `NotImplementedError("trainium and tpu")`. - [x] **Step 3: Implement.** - [x] **Step 4: Run** → PASS; `test_qwen_entry.py` PASS; gate clean; remove `tests/unit/registry/...qwen...` entries from `neuron-skips.txt` that now pass under cpu because the registration includes it? (Only if the gate reports them as resolved; follow its output.)
- [x] **Step 5: Commit** — `git commit -m "qwen_image: CUDA application — DiT via difflet.ops, encoder and VAE on the same GPU"`

---

### Task 8: Phase 0 on the box — environment facts, torch wheel, lock

**Files:**
- Create: `requirements-cuda.lock`; modify `pyproject.toml` `cuda = [...]` (torch pin as installed); `scripts/cuda_env_check.sh` (prints `nvidia-smi`, driver, `torch.__version__`, `torch.version.cuda`, `get_device_capability`, free disk, HF cache path).
- Modify: `docs/plans/2026-10-16-cuda-backend-support.md` Status log (Phase 0 verdict), tick 0.1–0.6.

- [x] **Step 1** On the box: `nvidia-smi; python3 --version; df -h /; free -g` → record. - [x] **Step 2** `python3 -m venv .venv && .venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 torch && .venv/bin/pip install -e ".[test]"` → `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(), torch.cuda.get_device_name())"` → Expected: a 12.8+ CUDA build, capability `(12, 0)`, name containing `RTX PRO 6000`. If the image's preinstalled torch 2.9.0+cu128 already satisfies this, keep it and record that instead of reinstalling.
- [x] **Step 3** Run the CPU gate on the box (`PYTHON_BIN=.venv/bin/python`) → Expected: same green as the Mac (neuron-skips may differ by host; follow the gate's output and ledger any difference). Then `DIFFLET_BACKEND=cpu pytest tests/unit/backends/test_cuda_application.py -q` and the same with a `device="cuda"` parametrization enabled via env `DIFFLET_TEST_CUDA_DEVICE=1` → PASS on the GPU (Review Focus 4 proven on real hardware).
- [x] **Step 4** SDPA report: `python -c "from difflet.backends.cuda.ops_impl.platform import sdpa_kernel_report; ..."` for q/k/v of shape `[24, 4096+1024, 128]` bf16 → record which backend (flash / efficient / cudnn / math).
- [x] **Step 5** `pip freeze --exclude-editable > requirements-cuda.lock`; set the `cuda` extra. - [x] **Step 6** Status log entry "Phase 0 verdict" with every number; commit — `git commit -m "cuda: Phase 0 — box facts, torch cu128 pin, requirements-cuda.lock"`

---

### Task 9: First image — `scripts/cuda_smoke.py` on the box

**Files:**
- Create: `scripts/cuda_smoke.py` (argparse: `--model-id Qwen/Qwen-Image`, `--prompt`, `--steps 50`, `--seed 0`, `--height/--width 1024`, `--output /tmp/qwen.png`; calls `DiffletPipeline.from_pretrained(model_id, backend="cuda", parallel=DiffletParallelConfig(tp_degree=1), dtype="bf16", height, width)` then `pipe(prompt=..., num_inference_steps=..., seed=...)`, saves the PNG, prints wall time per phase and `torch.cuda.max_memory_allocated()`), `scripts/cuda_smoke.sh` (venv + `DIFFLET_BACKEND=cuda` + log to `/tmp/logs/cuda-smoke-<ts>.log`).

- [x] **Step 1** `difflet download --model-id Qwen/Qwen-Image` (or `huggingface-cli download`) → record size and time. - [x] **Step 2** Run the smoke with `--steps 4` first (fast fail), then `--steps 50`. Expected: a PNG whose content matches the prompt by eye (open it; say what is in it); no device-mismatch errors; peak memory under 96 GB. - [x] **Step 3** Status log: wall time (load, encode, denoise/step, decode), peak memory, image path, the prompt. - [x] **Step 4** Commit — `git commit -m "cuda: Qwen-Image smoke script; first image on one RTX PRO 6000 at tp=1"`

---

### Task 10: Parity against diffusers on the GPU

**Files:**
- Create: `tests/numerical/cuda_parity_gpu.py` (one script, three forwards on the GPU on identical seeded inputs: diffusers' `QwenImageTransformer2DModel` in fp32 → in bf16 → Difflet's DiT via `app.forward_dit(bundle)` in bf16; writes `cuda_parity_qwen_image.json` with cosines, the control's deviation and the ratios), `tests/numerical/test_cuda_vs_diffusers.py` (reads the JSON; the three relative gates with the constants from `tests/numerical/test_tpu_vs_diffusers.py`: ratio ≤ 1.15, inside the control's distance, loose absolute floor; `DIFFLET_RUN_CUDA_NUMERICAL=1` to enable).

- [x] **Step 1** Write `test_cuda_vs_diffusers.py` first; run it without the JSON → Expected: skipped with "no result file". - [x] **Step 2** Write and run the GPU script → JSON. - [x] **Step 3** `DIFFLET_RUN_CUDA_NUMERICAL=1 pytest tests/numerical/test_cuda_vs_diffusers.py -v` → Expected: PASS; record all three numbers and the residual in the Status log **even if the gate fails** — a failure is a finding, not something to tune away. - [x] **Step 4** Commit — `git commit -m "cuda: diffusers parity gate for Qwen-Image on the GPU"`

---

### Task 11: Status log, docs, hand-off

- [x] **Step 1** Status log: Phase 2, 3, 4 entries (counts, times, cosines), the Release 1 exit criterion checked, rulings, deferred items; tick 2.x/3.x/4.1–4.5; note what `has_compiled_artifacts` means on CUDA (manifest only).
- [x] **Step 2** `DEVELOPER.md`: a "CUDA (Release 1)" subsection — venv on the box, `scripts/cuda_smoke.sh`, the parity script, the gate.
- [x] **Step 3** Commit — `git commit -m "plan: Release 1 — Qwen-Image tp=1 on one RTX PRO 6000; status log and docs"`. Do not push. Report rulings and deferred minors.

---

## Execution ledger (2026-10-16, inline execution on the Mac + the Yottalabs pod)

Copied verbatim from the executing-plans ledger before the workspace was deleted; the ``Ruling:`` lines are the decisions taken on the owner's behalf, each with its cost if wrong.

- Pre-flight: Task 2 collectives must expose gather_tp_dim/scatter_tp_dim/reduce_tp/get_tp_size/get_tp_rank for Task 3 build_parallel_layers (Task 1 Produces) — consistent. Task 5 CudaApplicationBase consumes _torch_megatron.weights.materialize_meta_/load_sharded_state_dict (Task 1) — consistent. Task 7 consumes Task 5 base, Task 6 config, Task 4 attention names — consistent. Task 6 moves TpuQwenImageConfig; tpu_application.py:39 keeps importing TpuQwenImageConfig via alias — consistent.
- Ruling: execute inline (executing-plans) without a fresh handoff review — the owner chose inline for PR 2 and asked for Qwen-Image today — cost if wrong: a plan defect found mid-run instead of before
- Task 1: complete (commits 10a15be..e6c56d9, tests: .claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh → cpu-gate: imports=ok unit=[2154 passed, 172 skipped, 7 xfailed in 19.20s] known_red=0 fixed=0 regressions=0 neuron_skips=129 new_skips=0 log=/tmp/logs/cpu-gate-unit-20261016-160347.log)
- Task 2: complete (commits e6c56d9..36f606d, tests: .claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh → cpu-gate: imports=ok unit=[2169 passed, 172 skipped, 6 xfailed in 18.93s] known_red=0 fixed=0 regressions=0 neuron_skips=129 new_skips=0 log=/tmp/logs/cpu-gate-unit-20261016-160552.log)
- Task 3: Ruling: the rmsnorm test mirrors test_tpu_layers.py (fp32 variance, output in the promoted dtype) instead of the brief's bf16-output assertion — the shared RMSNorm multiplies by a float32 weight, which is the existing CPU/TPU behaviour and the oracle — cost if wrong: none, no production change
- Task 3: complete (commits 36f606d..848cc20, tests: .claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh → cpu-gate: imports=ok unit=[2182 passed, 172 skipped, 3 xfailed in 19.10s] known_red=0 fixed=0 regressions=0 neuron_skips=129 new_skips=0 log=/tmp/logs/cpu-gate-unit-20261016-160717.log)
- Task 4: complete (commits 848cc20..e120a69, tests: .claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh → cpu-gate: imports=ok unit=[2220 passed, 172 skipped in 19.07s] known_red=0 fixed=0 regressions=0 neuron_skips=129 new_skips=0 log=/tmp/logs/cpu-gate-unit-20261016-160912.log)
- Task 5: complete (commits e120a69..17eaba9, tests: .claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh → cpu-gate: imports=ok unit=[2230 passed, 172 skipped in 19.07s] known_red=0 fixed=0 regressions=0 neuron_skips=129 new_skips=0 log=/tmp/logs/cpu-gate-unit-20261016-161058.log)
- Task 6: complete (commits 17eaba9..46f626a, tests: .claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh → cpu-gate: imports=ok unit=[2237 passed, 172 skipped in 19.08s] known_red=0 fixed=0 regressions=0 neuron_skips=129 new_skips=0 log=/tmp/logs/cpu-gate-unit-20261016-161215.log)
- Task 6: note: tests/unit/models/qwen_image/test_qwen_entry.py is in the Neuron collect-ignore list (imports the Trainium application at module level); run explicitly it errors on this host — test_dit_config.py ran alone: see above
- Task 7: Ruling: CUDA entry tests live in tests/unit/models/qwen_image/test_cuda_application.py, not in test_qwen_entry.py — that file imports the Trainium application at module level and is Neuron-only (collect-ignored here); its error-string regex was updated to "trainium, tpu and cuda" but could not be run on this host — cost if wrong: one regex fix on a Neuron host
- Task 7: Ruling: added test_tiny_qwen_dit_runs_through_the_cuda_ops_on_cpu (subprocess, DIFFLET_BACKEND=cuda, device=cpu) beyond the brief — the only pre-GPU proof that modeling_qwen_image resolves and runs on the CUDA ops; it caught my own test's wrong checkpoint layout, not a code bug — cost if wrong: ~2 s per suite run
- Task 7: Ruling: __call__(prompt=...) returns PIL images via VaeImageProcessor.postprocess with vae_scale_factor*2 (the Qwen-Image pipeline's convention); the orchestrator's output_type stays "pt" internally — cost if wrong: a resize/normalization mismatch visible in the first smoke image
- Task 7: complete (commits 46f626a..1f92e21, tests: .claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh → cpu-gate: imports=ok unit=[2245 passed, 172 skipped in 21.24s] known_red=0 fixed=0 regressions=0 neuron_skips=129 new_skips=0 log=/tmp/logs/cpu-gate-unit-20261016-161706.log)
- Task 8: facts: pod difflet-cuda-qwen (ID 502004828165112280), container b2823aca7326, Ubuntu 22.04.5, kernel 6.8.0-90, 16 vCPU, 141 GB RAM, 713 GB disk (612 free), NVIDIA RTX PRO 6000 Blackwell Server Edition 97887 MiB, driver 580.126.09, CUDA 13.0 driver, capability (12,0). Image's own torch is a nightly (2.12.0.dev20260311+cu128 under /opt/conda) — not used.
- Task 8: Ruling: fresh uv venv (Python 3.11.14) with released torch==2.11.0+cu128, cudnn 91900, triton 3.6.0 instead of the image's nightly — a dev wheel cannot be reproduced from a lock — cost if wrong: none observed; bf16 matmul and all 53 CUDA tests pass on the GPU
- Task 8: facts: hf download Qwen/Qwen-Image → 54 GB in ~93 s (hf_transfer); snapshot 75e0b4be04f60ec59a75f475837eced720f823b6
- Task 8: finding: SDPA picks "flash" for (B,H,S,D) but "math" for the (B*H,S,D) layout the model code uses → fixed in cuda attention (lift 3-D to 4-D before SDPA; test_heads_folded_inputs_reach_sdpa_as_four_dimensional RED→GREEN)
- Task 8: finding: cpu-gate.sh used -rs so FAILED lines never reached the regression diff (box showed "2 failed … regressions=0") → fixed to -rfEs
- Task 8: minor (deferred): tests/unit/serving/test_video_storage.py::test_staging_validation_rejects_path_mismatch_replacement_and_symlink and ::test_delete_staging_refuses_replaced_inode fail on the Linux box only (DID NOT RAISE VideoArtifactPathViolation; overlayfs/inode semantics in the container) — pre-existing, not CUDA, pass on macOS
- Task 8: complete (commits 1f92e21..4d1d637, tests: .claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh → cpu-gate: imports=ok unit=[2248 passed, 172 skipped in 21.75s] known_red=0 fixed=0 regressions=0 neuron_skips=129 new_skips=0 log=/tmp/logs/cpu-gate-unit-20261016-184712.log)
- Task 9: Ruling: generate() computes timesteps with dynamic_shift_mu + linear sigmas and passes them in (the orchestrator's own _timesteps omits mu and the model's scheduler has use_dynamic_shifting=True → ValueError on the first run); guidance defaults to 4.0 as the Trainium CLI does — cost if wrong: timestep schedule differs from Trainium's; mitigated by copying the CLI formula and testing it (test_dynamic_shift_mu_matches_the_cli_formula)
- Task 9: Ruling: the 50-step PNG (1.3 MB) and timings JSON are committed under artifacts/cuda-release1/ following the repo's artifacts/verification-<date>/ convention; the 4-step preview is not — cost if wrong: 1.3 MB in history
- Task 9: facts: 4 steps 1.9 s (0.47 s/step); 50 steps 20.5 s (0.41 s/step); load 10.3 s; peak alloc 57.95 GB; image prompt-faithful by eye (cat, garden, sunlight)
- Task 9: complete (commits 4d1d637..f07fe25, tests: .claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh → cpu-gate: imports=ok unit=[2248 passed, 172 skipped in 21.37s] known_red=0 fixed=0 regressions=0 neuron_skips=129 new_skips=0 log=/tmp/logs/cpu-gate-unit-20261016-185148.log)
- Task 10: facts: parity ran in ~40 s on the pod; fp32 reference fit on the 96 GB card; gates: ratio 0.997 ≤ 1.15, same-dtype 0.190 ≤ 1.2, cosine 0.999991 ≥ 0.995 — 3 passed
- Task 10: minor (deferred): tests/unit/serving/test_video_service.py::test_queued_video_uses_queue_timeout_before_execution_timeout flaked once in the gate run (asyncio timing; machine busy with scp), passes alone — second serving-timing flake seen today
- Task 10: complete (commits f07fe25..c91cd5b, tests: .claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh → cpu-gate: imports=ok unit=[2248 passed, 172 skipped in 21.41s] known_red=0 fixed=0 regressions=0 neuron_skips=129 new_skips=0 log=/tmp/logs/cpu-gate-unit-20261016-185258.log)
- Task 11: complete (commits c91cd5b..5effb25, tests: .claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh → cpu-gate: imports=ok unit=[2248 passed, 172 skipped in 21.27s] known_red=0 fixed=0 regressions=0 neuron_skips=129 new_skips=0 log=/tmp/logs/cpu-gate-unit-20261016-185430.log)
