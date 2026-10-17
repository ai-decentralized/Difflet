# CUDA F3b — `difflet serve` for Qwen-Image on the CUDA backend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `DIFFLET_BACKEND=cuda difflet serve --model-id Qwen/Qwen-Image` answers `/v1/chat/completions` with a PNG generated on one GPU through the CUDA backend, with the model resident for the life of the server and every request handled exactly as `difflet run` on CUDA handles it.

**Architecture:** The serving stack is `resolve_serving_model → ArtifactPreparer.prepare_runtime → ResidentWorkerServingEngine`, and the engine owns exactly one resident worker process that runs a `StagePipelineEngine` over the model's stage adapter. The Trainium Qwen-Image adapter (`difflet/serving/orchestrators/qwen_image.py`) fills that worker with three Neuron stage applications and a NeuronCore allocation; on CUDA at `tp=1` the whole model is one `CudaQwenImageApplication` behind `DiffletPipeline`, so a second, small adapter module gives the worker **one opaque `pipeline` stage** that holds `DiffletPipeline.from_pretrained(backend="cuda", …)` resident and calls `app.generate()` per request — the same two calls `difflet/cli/orchestrators/qwen_image_cuda.py` makes. Selection is by backend name in `difflet/serving/model_registry.py`, the way `main._get_orchestrator` selects the CLI orchestrator: a CUDA metadata overlay (same checkpoint ids, one-stage pipeline definition, CUDA factories) replaces the Trainium metadata when `current_backend() == "cuda"`, and nothing changes on any other backend. The engine, the HTTP layer and the Trainium adapter are not modified.

**Tech Stack:** FastAPI/uvicorn serving stack already in the repo, `DiffletPipeline` + `CudaQwenImageApplication` (Release 1), PIL for PNG encoding, pytest on CPU with the pipeline faked.

**Spec:** `docs/plans/2026-10-16-cuda-backend-support.md` — follow-up **F3** ("`difflet serve` for Qwen-Image"), cut to **F3b** in the F3 status-log entry; decisions **D5** (one process per GPU), **D2** (the CLI/serving surface is the user-facing proof), Release 1 scope (`tp=cp=dp=1`, no CFG, bf16).

## Global Constraints

- Release 1 scope on CUDA: `tp=1`, `cp=1`, `dp=1`, no CFG-parallel, no TeaCache, one request shape; anything wider is refused up front naming the follow-up (`F2` second card / CFG-parallel / CP, `F4` TeaCache, `F6` TP / SP, `F3` bucketed shapes — the labels `difflet/cli/orchestrators/qwen_image_cuda.py:refuse_unsupported` uses).
- Request handling identical to the CLI path: `DiffletPipeline.from_pretrained(backend="cuda", parallel=tp1, dtype="bf16", height, width, compile_cache_dir, local_files_only=True, revision, force_compile)` then `app.generate(prompt, num_inference_steps=…, seed=…, output_type="pil")`; no `guidance_scale` kwarg (the application refuses it: `guidance_embeds=False`).
- Existing Trainium serving behaviour byte-for-byte unchanged: `difflet/serving/orchestrators/qwen_image.py`, `difflet/serving/engines/**`, `difflet/serving/openai/**` are not edited; `tests/unit/serving` is run before (baseline `/tmp/logs/f3b-serving-before.log`) and after, same pass set plus the new tests.
- `difflet/ops/__init__.py:_EXPORTS` frozen; nothing new under `difflet/backends/cuda/` imports Neuron/XLA (`test_cuda_import_hygiene` stays green).
- CPU gate green at the end: `.claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh` (known-red empty, `neuron-skips.txt` unchanged).
- No GPU in this session: every task is proven on CPU with `DiffletPipeline.from_pretrained` replaced by a recorder; the GPU run is a recorded pending step.
- Black, line length 100. Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. No push.

## Review Focus

1. `difflet serve --model-id Qwen/Qwen-Image` on CUDA **without** `--tp-degree`: the registry default is `tp_degree=4`, so the profile would be refused with F6 unless CUDA defaults it to 1 the way the CLI's `args.tp_degree or 1` does — Task 1 `test_cuda_defaults_tp_degree_to_one`.
2. A chat request carrying `extra_body.guidance_scale` other than the no-op value: must be refused (the value cannot change the image; the CLI refuses `--guidance-scale`), never silently accepted — Task 3 `test_guidance_scale_other_than_the_noop_value_is_refused`.
3. Startup with `--shapes` or `--teacache-speedup`: refused **before** the 54 GB snapshot resolution, naming F3 / F4 — Task 2 `test_unsupported_profiles_are_refused_before_the_download`.
4. The CUDA runtime plan handed to the worker: `_apply_worker_runtime_environment` demands one allocation with positive cores, enough `available_core_ids`, child distributed 1/1/0/0 — a plan that fails there is only visible on the pod — Task 2 `test_runtime_plan_is_accepted_by_the_resident_worker_environment`.
5. A request shape that is not the profile's (e.g. 768×768 on a 1024×1024 server): the resident application is built for one shape, so the validator must refuse with `profile_mismatch` before the worker sees it — Task 3 `test_request_shape_must_equal_the_profile_shape`.

---

### Task 1: CUDA serving metadata overlay, selected by backend name

**Files:**
- Modify: `difflet/common/registry/qwen_image.py` (add `cuda_serving_metadata()`)
- Modify: `difflet/serving/model_registry.py` (`resolve_serving_model`)
- Modify: `difflet/serving/types.py` (`StagePlacement`, `StageRuntimeSpec.__post_init__`)
- Test: `tests/unit/serving/test_model_registry_cuda.py`

**Interfaces:**
- Produces: `difflet.common.registry.qwen_image.cuda_serving_metadata() -> ServingModelMetadata` with `model_type="qwen_image"`, `checkpoint_ids=("Qwen/Qwen-Image",)`, `output_modality="image"`, `output_mime_type="image/png"`, `chat_content_type="image_url"`, `default_steps=50` (the CUDA CLI default, `qwen_image_cuda.DEFAULT_STEPS`), `default_guidance_scale=4.0` (the value the CUDA application fills into the ignored guidance tensor), a one-stage `PipelineDefinition(stages=(StageDefinition("pipeline", "opaque_pipeline", "pipeline", output_keys=("output",), final_output=True),))`, `artifact_preparer_factory="difflet.serving.orchestrators.qwen_image_cuda:CudaQwenImageServingArtifactPreparer"`, `orchestrator_factory="difflet.serving.orchestrators.qwen_image_cuda:CudaQwenImageServingStageAdapter"`, `request_validator_factory="difflet.serving.orchestrators.qwen_image_cuda:CudaQwenImageServingRequestValidator"`.
- Produces: `model_registry.resolve_serving_model(options)` — when `difflet.backends.registry.current_backend() == "cuda"`: looks the entry name up in `_CUDA_SERVING_OVERLAYS: dict[str, str]` (`{"qwen_image": "difflet.common.registry.qwen_image:cuda_serving_metadata"}`, loaded with the existing `load_factory`), raises `ValueError(f"{model_id} is not on the CUDA backend yet (plan follow-ups F1 Wan, F5 Flux); set DIFFLET_BACKEND=trainium for the staged Neuron path")` for any other entry, and builds the profile with `tp_degree=1` when `options.tp_degree is None`. On every other backend the function body is unchanged.
- Produces: `StagePlacement` gains `"cuda"`; `StageRuntimeSpec(placement="cuda")` follows the `"tpu"` rule (allocation and topology required, `artifact_id` must be `None`).

- [ ] **Step 1: Write the failing tests**

```python
"""`difflet serve` picks the CUDA Qwen-Image serving adapter by backend name (CUDA plan F3b)."""
from __future__ import annotations

import pytest

from difflet.common.registry import qwen_image as qwen_registry
from difflet.serving.model_registry import resolve_serving_model
from difflet.serving.options import ServeOptions
from difflet.serving.types import ParallelTopology, StageRuntimeSpec

_CUDA_MODULE = "difflet.serving.orchestrators.qwen_image_cuda"


def _resolve(monkeypatch, backend, **options):
    monkeypatch.setenv("DIFFLET_BACKEND", backend)
    return resolve_serving_model(ServeOptions(model_id="Qwen/Qwen-Image", **options))


def test_cuda_routes_qwen_to_the_cuda_factories(monkeypatch):
    resolved = _resolve(monkeypatch, "cuda")
    md = resolved.metadata
    assert md.artifact_preparer_factory == f"{_CUDA_MODULE}:CudaQwenImageServingArtifactPreparer"
    assert md.orchestrator_factory == f"{_CUDA_MODULE}:CudaQwenImageServingStageAdapter"
    assert md.request_validator_factory == f"{_CUDA_MODULE}:CudaQwenImageServingRequestValidator"
    assert [(s.stage_id, s.kind, s.role, s.final_output) for s in md.pipeline_definition.stages] == [
        ("pipeline", "opaque_pipeline", "pipeline", True)
    ]


def test_cuda_request_defaults_match_the_cuda_cli(monkeypatch):
    # The CLI defaults --steps to 50 on CUDA and the application fills guidance 4.0.
    md = _resolve(monkeypatch, "cuda").metadata
    assert md.default_steps == 50 and md.default_guidance_scale == 4.0
    assert md.checkpoint_ids == ("Qwen/Qwen-Image",) and md.output_mime_type == "image/png"


def test_cuda_defaults_tp_degree_to_one(monkeypatch):
    # Registry default is tp=4 (Trainium); the CUDA CLI uses `args.tp_degree or 1`.
    assert _resolve(monkeypatch, "cuda").profile.parallel.tp_degree == 1
    assert _resolve(monkeypatch, "cuda", tp_degree=1).profile.parallel.tp_degree == 1


def test_explicit_tp_degree_is_kept_for_the_preparer_to_refuse(monkeypatch):
    assert _resolve(monkeypatch, "cuda", tp_degree=2).profile.parallel.tp_degree == 2


@pytest.mark.parametrize("backend", ["cpu", "trainium"])
def test_other_backends_keep_the_trainium_metadata(monkeypatch, backend):
    resolved = _resolve(monkeypatch, backend)
    assert resolved.metadata == qwen_registry.serving_metadata()
    assert resolved.profile.parallel.tp_degree == 4


def test_other_models_on_cuda_name_their_follow_up(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    with pytest.raises(ValueError) as exc:
        resolve_serving_model(ServeOptions(model_id="black-forest-labs/FLUX.1-dev"))
    assert "F5" in str(exc.value) and "F1" in str(exc.value)


def test_cuda_placement_needs_allocation_and_topology_but_no_artifact():
    topology = ParallelTopology(1, 1, 1)
    StageRuntimeSpec("pipeline", "qwen-cuda", topology, None, placement="cuda")
    with pytest.raises(ValueError):
        StageRuntimeSpec("pipeline", None, None, None, placement="cuda")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `DIFFLET_BACKEND=cpu PYTHONPATH=$PWD /Users/hang/code/Difflet/.venv/bin/python -m pytest tests/unit/serving/test_model_registry_cuda.py -q -p no:cacheprovider`
Expected: FAIL — `test_cuda_routes_qwen_to_the_cuda_factories` on the Trainium factory string; the placement test with `ValueError: unsupported stage placement 'cuda'`.

- [ ] **Step 3: Implement**

`cuda_serving_metadata()` in `difflet/common/registry/qwen_image.py` next to `serving_metadata()` (values from Interfaces). In `model_registry.py`: `_CUDA_SERVING_OVERLAYS` module constant; in `resolve_serving_model` after the checkpoint check, `backend = current_backend()` (imported inside the function from `difflet.backends.registry`); `if backend == "cuda": metadata = _cuda_serving_metadata(entry.name, options.model_id)`; `tp_degree = options.tp_degree if options.tp_degree is not None or backend != "cuda" else 1` passed to `build_serving_profile`. In `types.py`: add `"cuda"` to the `StagePlacement` literal and `if self.placement in {"tpu", "cuda"}:` for the existing TPU branch, error message `f"{self.placement} stage {self.stage_id!r} requires allocation and topology"`.

- [ ] **Step 4: Run the tests, then the serving suite**

Run: the Step 2 command, then `DIFFLET_BACKEND=cpu PYTHONPATH=$PWD /Users/hang/code/Difflet/.venv/bin/python -m pytest tests/unit/serving -q -p no:cacheprovider`
Expected: new file PASS; serving suite same pass count as `/tmp/logs/f3b-serving-before.log` plus 8.

- [ ] **Step 5: Commit**

```bash
git add difflet/common/registry/qwen_image.py difflet/serving/model_registry.py difflet/serving/types.py tests/unit/serving/test_model_registry_cuda.py
git commit -m "serving(cuda): select a CUDA Qwen-Image serving overlay by backend name (F3b)"
```

---

### Task 2: Artifact preparer — refusals before the download, an eager runtime plan the worker accepts

**Files:**
- Create: `difflet/serving/orchestrators/qwen_image_cuda.py`
- Test: `tests/unit/serving/test_qwen_image_cuda_serving.py`

**Interfaces:**
- Consumes: `difflet.serving.orchestrators.base.resolve_hf_model_source(model_id, *, revision, download_policy)`; `difflet.serving.types.{ResolvedRuntimeBundle, RuntimePlan, RuntimeEnvironment, WorkerAllocationSpec, StageRuntimeSpec, ParallelTopology, DistributedProcessEnvironment, ArtifactSet, CompileArtifactIdentity}`; `difflet.serving.errors.feature_not_supported`.
- Produces: `refuse_unsupported_profile(profile: ServingProfile) -> None` — raises `feature_not_supported(f"{_TAG} {what} is not supported on the CUDA backend in Release 1 (plan follow-up {F})")` for: `profile.shapes` → `"--shapes (bucketed request shapes)"`, F3; `parallel.tp_degree > 1` → F6; `parallel.sp_enabled` → F6; `parallel.cp_degree > 1` → F2; `parallel.cfg_parallel_enabled` → F2; `profile.teacache_speedup is not None` → F4. `_TAG = "[qwen_image/cuda serve]"`.
- Produces: `class CudaQwenImageServingArtifactPreparer(model_id="Qwen/Qwen-Image", revision=None)` with `model_id`, `model_type="qwen_image"`, `prepare_runtime(profile, *, download_policy, compile_policy) -> ResolvedRuntimeBundle`: refuses first, then `resolve_hf_model_source`, then a bundle with `compile_specs=()`, `artifacts=ArtifactSet(())`, `pipeline_definition=cuda_serving_metadata().pipeline_definition`, `runtime_plan=cuda_runtime_plan(profile, source)`; `adapter_config={"compile_policy": compile_policy}` so the worker knows `FORCE`.
- Produces: `cuda_runtime_plan(profile, source) -> RuntimePlan(mode="resident", profile_identity=<sha256 digest of CompileArtifactIdentity.from_cache_inputs({"backend": "cuda", "model_type", "source_id": source.resolved_source_id, "height", "width", "tp_degree", "dtype": profile.dtype})>, environment=RuntimeEnvironment(available_core_ids=(0,), None, None, None, d, d), allocations=(WorkerAllocationSpec("qwen-cuda", 1, 1, 1),), stages=(StageRuntimeSpec("pipeline", "qwen-cuda", ParallelTopology(1, 1, 1), None, placement="cuda"),))` with `d = DistributedProcessEnvironment(1, 1, 0, 0)`. Core id `0` is the one GPU: the engine's "cores" are abstract device slots and it sets `NEURON_RT_VISIBLE_CORES=0` in a worker that never loads the Neuron runtime (ruling, recorded in the ledger).

- [ ] **Step 1: Write the failing tests**

```python
"""Qwen-Image serving on CUDA: preparer, validator and resident adapter, CPU-only (plan F3b).

DiffletPipeline.from_pretrained is replaced by a recorder; no weights, no GPU.
"""
from __future__ import annotations

import pytest

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.serving.engines.resident_worker import _apply_worker_runtime_environment
from difflet.serving.errors import DiffletServingError
from difflet.serving.options import CompilePolicy, DownloadPolicy
from difflet.serving.orchestrators import qwen_image_cuda as mod
from difflet.serving.types import ResolvedModelSource, ServingProfile

_SOURCE = ResolvedModelSource("hf_snapshot", "Qwen/Qwen-Image", None, "/snap/qwen", "a" * 40)


def _profile(**over) -> ServingProfile:
    d = dict(model_id="Qwen/Qwen-Image", model_type="qwen_image", height=1024, width=1024,
             num_frames=None, parallel=DiffletParallelConfig(tp_degree=1), cache_dir="/c")
    d.update(over)
    return ServingProfile(**d)


@pytest.fixture
def source(monkeypatch):
    calls = []
    monkeypatch.setattr(mod, "resolve_hf_model_source", lambda *a, **k: calls.append((a, k)) or _SOURCE)
    return calls


def _prepare(profile, source_calls, policy=CompilePolicy.AUTO):
    return mod.CudaQwenImageServingArtifactPreparer().prepare_runtime(
        profile, download_policy=DownloadPolicy.AUTO, compile_policy=policy)


def test_prepare_runtime_resolves_the_snapshot_and_builds_an_eager_bundle(source):
    bundle = _prepare(_profile(), source)
    assert source and source[0][0] == ("Qwen/Qwen-Image",)
    assert bundle.compile_specs == () and bundle.artifacts.bindings == ()
    assert [s.stage_id for s in bundle.pipeline_definition.stages] == ["pipeline"]
    stage = bundle.runtime_plan.stages[0]
    assert stage.placement == "cuda" and stage.artifact_id is None
    assert bundle.runtime_plan.profile_identity and bundle.adapter_config["compile_policy"] is CompilePolicy.AUTO


@pytest.mark.parametrize(
    "over,follow_up",
    [
        ({"shapes": ((1024, 1024, None), (768, 768, None))}, "F3"),
        ({"parallel": DiffletParallelConfig(tp_degree=2)}, "F6"),
        ({"parallel": DiffletParallelConfig(tp_degree=2, sp_enabled=True)}, "F6"),
        ({"parallel": DiffletParallelConfig(tp_degree=1, cp_degree=2)}, "F2"),
        ({"parallel": DiffletParallelConfig(tp_degree=1, cfg_parallel_enabled=True)}, "F2"),
        ({"teacache_speedup": 1.5}, "F4"),
    ],
)
def test_unsupported_profiles_are_refused_before_the_download(source, over, follow_up):
    with pytest.raises(DiffletServingError) as exc:
        _prepare(_profile(**over), source)
    assert exc.value.code == "feature_not_supported" and follow_up in exc.value.message
    assert source == []


def test_runtime_plan_is_accepted_by_the_resident_worker_environment(source, monkeypatch):
    for name in ("NEURON_RT_VISIBLE_CORES", "NEURON_RT_NUM_CORES", "WORLD_SIZE", "RANK"):
        monkeypatch.delenv(name, raising=False)
    bundle = _prepare(_profile(), source)
    _apply_worker_runtime_environment(bundle)  # raises ValueError on a plan the worker rejects


def test_profile_identity_tracks_shape_and_source(source):
    a = _prepare(_profile(), source).runtime_plan.profile_identity
    b = _prepare(_profile(height=768, width=768), source).runtime_plan.profile_identity
    assert a != b and a == _prepare(_profile(), source).runtime_plan.profile_identity
```

Check which `DiffletParallelConfig` combinations above are constructible (`sp_enabled` may require `tp_degree > 1`; `cfg_parallel_enabled` may need `cp`/`dp` rules) and adjust the parametrize rows to values the config accepts, keeping one row per refusal.

- [ ] **Step 2: Run to verify failure**

Run: `DIFFLET_BACKEND=cpu PYTHONPATH=$PWD /Users/hang/code/Difflet/.venv/bin/python -m pytest tests/unit/serving/test_qwen_image_cuda_serving.py -q -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: difflet.serving.orchestrators.qwen_image_cuda`.

- [ ] **Step 3: Implement the module header, `refuse_unsupported_profile`, `cuda_runtime_plan`, `CudaQwenImageServingArtifactPreparer`**

Module docstring states the design (one resident worker, one opaque stage, same two calls as the CUDA CLI, what is refused and why core id 0). Imports at module level: stdlib, `difflet.serving.*` only (no torch, no diffusers, no CLI import at module level — the preparer runs in the parent process).

- [ ] **Step 4: Run the tests to verify they pass**

Run: Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add difflet/serving/orchestrators/qwen_image_cuda.py tests/unit/serving/test_qwen_image_cuda_serving.py
git commit -m "serving(cuda): Qwen-Image artifact preparer — refusals before the download, eager runtime plan"
```

---

### Task 3: Request validator — the CLI's request contract at the HTTP boundary

**Files:**
- Modify: `difflet/serving/orchestrators/qwen_image_cuda.py`
- Test: `tests/unit/serving/test_qwen_image_cuda_serving.py` (append)

**Interfaces:**
- Consumes: `difflet.serving.errors.{feature_not_supported, profile_mismatch, prompt_too_long}`; `difflet.common.orchestrators.qwen_image.TEXT_SEQ_LEN` (1024, the DiT text length the CUDA application pads/truncates to).
- Produces: `NOOP_GUIDANCE_SCALE = 4.0`; `class CudaQwenImageServingRequestValidator(runtime: ResolvedRuntimeBundle)` with `model_id`, `model_type`, `runtime`, `preload()` (loads the tokenizer), `validate(request)`:
  1. `request.guidance_scale != NOOP_GUIDANCE_SCALE` → `feature_not_supported("guidance_scale has no effect on Qwen-Image (guidance_embeds=False); Release 1 generation on CUDA is single-pass without classifier-free guidance — omit it")`.
  2. `(request.height, request.width) != (profile.height, profile.width)` → `profile_mismatch(f"request shape {h}x{w} is not the resident CUDA profile shape {ph}x{pw}")`.
  3. Prompt tokens beyond the template exceed `TEXT_SEQ_LEN` → `prompt_too_long(f"Qwen prompt exceeds the DiT text length {TEXT_SEQ_LEN}")`; tokens counted as `len(tokenizer(_QWEN_TEMPLATE.format(prompt)).input_ids) - _QWEN_DROP_IDX`, with `_QWEN_TEMPLATE` and `_QWEN_DROP_IDX = 34` copied from the Trainium module (diffusers' `QwenImagePipeline` template and drop index).
  - `_tokenizer_for_runtime()` → `transformers.AutoTokenizer.from_pretrained(<pinned_model_path>/tokenizer)`, cached; tests monkeypatch it.

- [ ] **Step 1: Write the failing tests (append)**

```python
class _Tok:
    def __init__(self, n): self.n = n
    def __call__(self, text, **kw):
        import types
        return types.SimpleNamespace(input_ids=list(range(self.n)))


def _validator(bundle, n_tokens=40):
    v = mod.CudaQwenImageServingRequestValidator(bundle)
    v._tokenizer_for_runtime = lambda: _Tok(n_tokens)
    return v


def _request(**over):
    from difflet.serving.types import DiffletGenerateRequest
    d = dict(request_id="r1", model="Qwen/Qwen-Image", prompt="a cat", height=1024, width=1024,
             num_inference_steps=50, guidance_scale=mod.NOOP_GUIDANCE_SCALE, seed=7)
    d.update(over)
    return DiffletGenerateRequest(**d)


def test_default_request_is_accepted(source):
    _validator(_prepare(_profile(), source)).validate(_request())


def test_guidance_scale_other_than_the_noop_value_is_refused(source):
    with pytest.raises(DiffletServingError) as exc:
        _validator(_prepare(_profile(), source)).validate(_request(guidance_scale=1.0))
    assert exc.value.code == "feature_not_supported" and "guidance_embeds" in exc.value.message


def test_request_shape_must_equal_the_profile_shape(source):
    with pytest.raises(DiffletServingError) as exc:
        _validator(_prepare(_profile(), source)).validate(_request(height=768, width=768))
    assert exc.value.code == "profile_mismatch"


def test_prompt_longer_than_the_dit_text_length_is_refused(source):
    bundle = _prepare(_profile(), source)
    _validator(bundle, n_tokens=34 + 1024).validate(_request())
    with pytest.raises(DiffletServingError) as exc:
        _validator(bundle, n_tokens=34 + 1025).validate(_request())
    assert exc.value.code == "prompt_too_long"


def test_preload_loads_the_tokenizer_from_the_pinned_snapshot(source, monkeypatch):
    seen = {}
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        staticmethod(lambda p, **k: seen.update(path=str(p)) or _Tok(5)))
    mod.CudaQwenImageServingRequestValidator(_prepare(_profile(), source)).preload()
    assert seen["path"] == "/snap/qwen/tokenizer"
```

- [ ] **Step 2: Run to verify failure** — Step 2 command of Task 2. Expected: `AttributeError: … has no attribute 'CudaQwenImageServingRequestValidator'`.

- [ ] **Step 3: Implement the validator** (Interfaces above).

- [ ] **Step 4: Run to verify pass.** Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add difflet/serving/orchestrators/qwen_image_cuda.py tests/unit/serving/test_qwen_image_cuda_serving.py
git commit -m "serving(cuda): Qwen-Image request validator — the CUDA CLI's request contract at the HTTP boundary"
```

---

### Task 4: Resident stage adapter — the model lives in the worker, one `pipeline` stage, the CLI's two calls

**Files:**
- Modify: `difflet/cli/orchestrators/qwen_image_cuda.py` (extract `cuda_pipeline_kwargs`)
- Modify: `difflet/serving/orchestrators/qwen_image_cuda.py`
- Test: `tests/unit/serving/test_qwen_image_cuda_serving.py` (append); `tests/unit/cli/test_orchestrator_qwen_cuda.py` stays green

**Interfaces:**
- Produces (CLI module): `cuda_pipeline_kwargs(*, tp_degree: int | None, height: int | None, width: int | None, cache_dir: str | None, revision: str | None, force_compile: bool, load: bool) -> dict` — the dict `pipeline_kwargs(args, load=)` builds today; `pipeline_kwargs` becomes a one-line call to it.
- Consumes: `difflet.serving.engines.stage_pipeline.{ValidatedStageRunner, stage_result, require_exact_payload}`; `difflet.serving.types.{QwenInitialPayload, QwenFinalPayload, DiffletGenerateOutput, StageInvocation, StageExecutionResult}`; `difflet.pipeline.difflet_pipeline.DiffletPipeline.from_pretrained` (imported inside `create_loaded_runners`, so tests monkeypatch `difflet.pipeline.difflet_pipeline.DiffletPipeline.from_pretrained` exactly as `tests/unit/cli/test_orchestrator_qwen_cuda.py` does).
- Produces: `SMOKE_STEPS = 4`; `class CudaQwenImagePipelineRunner(adapter)` with `async execute(invocation: StageInvocation[QwenInitialPayload]) -> StageExecutionResult[QwenFinalPayload]` (calls `invocation.context.cancellation.throw_if_cancelled()` then `adapter.generate_png(invocation.request)`) and `async shutdown()` (drops `adapter.pipe`).
- Produces: `class CudaQwenImageServingStageAdapter(model_id="Qwen/Qwen-Image")` implementing `ServingStageAdapter`: `active_runtime`, `active_profile`, `pipe` (the `DiffletPipeline`), `create_loaded_runners(runtime)` → refuses via `refuse_unsupported_profile`, then `DiffletPipeline.from_pretrained(self.model_id, **cuda_pipeline_kwargs(tp_degree=profile.parallel.tp_degree, height=profile.height, width=profile.width, cache_dir=profile.cache_dir, revision=profile.revision, force_compile=runtime.adapter_config["compile_policy"] is CompilePolicy.FORCE, load=True))`, returns `OrderedDict(pipeline=ValidatedStageRunner(CudaQwenImagePipelineRunner(self), QwenInitialPayload, QwenFinalPayload))`; `generate_png(request) -> DiffletGenerateOutput`: `image = self.pipe.app.generate(request.prompt, num_inference_steps=request.num_inference_steps, seed=request.seed, output_type="pil")[0]`, PNG-encoded with PIL, `mime_type="image/png"`, `output_format="png"`; `initial_payload` → `QwenInitialPayload()`; `finalize` → `require_exact_payload(payload, QwenFinalPayload, boundary="Qwen CUDA final payload").output`; `smoke_request()` → `DiffletGenerateRequest("startup-smoke", model_id, "a small red square", profile.height, profile.width, SMOKE_STEPS, NOOP_GUIDANCE_SCALE, 0)`; `validate_smoke_output(output)` → non-empty data and `PIL.Image.open(BytesIO(data)).size == (profile.width, profile.height)` else `RuntimeError`; `reset_request_state` no-op; `shutdown()` drops `pipe`, `active_runtime`, `active_profile`.

- [ ] **Step 1: Write the failing tests (append)**

```python
import asyncio, io
from PIL import Image


class _App:
    def __init__(self, size=(1024, 1024)):
        self.calls, self.size = [], size
    def generate(self, prompt, **kw):
        self.calls.append((prompt, kw))
        return [Image.new("RGB", self.size, "red")]


class _Pipe:
    def __init__(self, size=(1024, 1024)):
        self.app = _App(size)


@pytest.fixture
def recorder(monkeypatch):
    calls = []
    def fake(model_id, **kw):
        calls.append((model_id, kw))
        return _Pipe(kw.get("_size", (kw["width"], kw["height"])))
    monkeypatch.setattr("difflet.pipeline.difflet_pipeline.DiffletPipeline.from_pretrained", staticmethod(fake))
    return calls


def _engine(bundle):
    from difflet.serving.engines.stage_pipeline import StagePipelineEngine
    adapter = mod.CudaQwenImageServingStageAdapter()
    return adapter, StagePipelineEngine(runtime=bundle, adapter=adapter)


def test_startup_loads_the_pipeline_exactly_as_the_cuda_cli_would(source, recorder):
    import argparse
    from difflet.cli.orchestrators import qwen_image_cuda as cli
    bundle = _prepare(_profile(cache_dir="/c", revision="r1"), source, CompilePolicy.FORCE)
    adapter, engine = _engine(bundle)
    asyncio.run(engine.start())
    model_id, kw = recorder[0]
    cli_kw = cli.pipeline_kwargs(argparse.Namespace(tp_degree=1, height=1024, width=1024,
                                 cache_dir="/c", revision="r1", force=True), load=True)
    assert model_id == "Qwen/Qwen-Image" and kw == cli_kw
    assert kw["backend"] == "cuda" and kw["load"] is True and kw["force_compile"] is True


def test_startup_smoke_runs_four_steps_at_the_profile_shape(source, recorder):
    adapter, engine = _engine(_prepare(_profile(), source))
    asyncio.run(engine.start())
    prompt, kw = adapter.pipe.app.calls[0]
    assert kw == {"num_inference_steps": mod.SMOKE_STEPS, "seed": 0, "output_type": "pil"}
    assert "guidance_scale" not in kw


def test_generate_passes_prompt_steps_and_seed_and_returns_png(source, recorder):
    from difflet.serving.types import WorkerRequestContext
    adapter, engine = _engine(_prepare(_profile(), source))
    asyncio.run(engine.start())
    req = _request(prompt="a cat", num_inference_steps=7, seed=3)
    out = asyncio.run(engine.generate(req, WorkerRequestContext.with_timeout("r1", 30.0)))
    prompt, kw = adapter.pipe.app.calls[-1]
    assert prompt == "a cat" and kw == {"num_inference_steps": 7, "seed": 3, "output_type": "pil"}
    assert out.mime_type == "image/png" and out.output_format == "png"
    assert Image.open(io.BytesIO(out.data)).size == (1024, 1024)


def test_smoke_output_of_the_wrong_size_fails_startup(source, monkeypatch):
    monkeypatch.setattr("difflet.pipeline.difflet_pipeline.DiffletPipeline.from_pretrained",
                        staticmethod(lambda m, **kw: _Pipe((8, 8))))
    adapter, engine = _engine(_prepare(_profile(), source))
    with pytest.raises(RuntimeError):
        asyncio.run(engine.start())
    assert adapter.pipe is None


def test_cancelled_request_never_reaches_generate(source, recorder):
    from difflet.serving.types import CancellationSignal, WorkerRequestContext
    adapter, engine = _engine(_prepare(_profile(), source))
    asyncio.run(engine.start())
    n = len(adapter.pipe.app.calls)
    signal = CancellationSignal(); signal.cancel()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(engine.generate(_request(), WorkerRequestContext.with_timeout("r", 30.0, cancellation=signal)))
    assert len(adapter.pipe.app.calls) == n


def test_unsupported_profile_is_refused_before_any_load(source, recorder):
    adapter, engine = _engine(_prepare(_profile(), source))
    adapter.active_profile = None
    bundle = _prepare(_profile(), source)
    object.__setattr__(bundle, "profile", _profile(parallel=DiffletParallelConfig(tp_degree=2)))
    with pytest.raises(DiffletServingError):
        asyncio.run(mod.CudaQwenImageServingStageAdapter().create_loaded_runners(bundle))
    assert recorder == []


def test_shutdown_releases_the_resident_pipeline(source, recorder):
    adapter, engine = _engine(_prepare(_profile(), source))
    asyncio.run(engine.start())
    asyncio.run(engine.shutdown())
    assert adapter.pipe is None and adapter.active_runtime is None
```

`ResolvedRuntimeBundle` is frozen: build the tp=2 bundle for `test_unsupported_profile_is_refused_before_any_load` by calling `cuda_runtime_plan` directly with a tp=2 profile (the preparer refuses it) and constructing `ResolvedRuntimeBundle(...)` by hand, instead of `object.__setattr__`.

- [ ] **Step 2: Run to verify failure.** Expected: `AttributeError … CudaQwenImageServingStageAdapter` / `cuda_pipeline_kwargs`.

- [ ] **Step 3: Implement** — `cuda_pipeline_kwargs` in the CLI module (keep `pipeline_kwargs`'s dict byte-identical; `__all__` gains the name); the runner and adapter per Interfaces. `create_loaded_runners` imports `DiffletPipeline` and `cuda_pipeline_kwargs` inside the method (worker-side imports; the parent process never imports torch for this adapter).

- [ ] **Step 4: Run** `tests/unit/serving/test_qwen_image_cuda_serving.py tests/unit/cli/test_orchestrator_qwen_cuda.py`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add difflet/cli/orchestrators/qwen_image_cuda.py difflet/serving/orchestrators/qwen_image_cuda.py tests/unit/serving/test_qwen_image_cuda_serving.py
git commit -m "serving(cuda): resident Qwen-Image stage adapter — one pipeline stage, the CUDA CLI's two calls"
```

---

### Task 5: `scripts/cuda_serve_smoke.py` — the serve column of the GPU evidence

**Files:**
- Create: `scripts/cuda_serve_smoke.py`
- Create: `scripts/cuda_serve_smoke.sh`
- Test: `tests/unit/scripts/test_cuda_serve_smoke.py`

**Interfaces:**
- Produces: `build_chat_body(prompt: str, *, steps: int, seed: int, height: int, width: int) -> dict` → `{"model": "Qwen/Qwen-Image", "messages": [{"role": "user", "content": prompt}], "extra_body": {"num_inference_steps": steps, "seed": seed, "height": height, "width": width}}`; `decode_image_data_url(url: str) -> bytes` (`data:image/png;base64,…` → bytes, `ValueError` otherwise); `image_url_from_response(body: dict) -> str` (first `image_url.url` in `choices[0].message.content`); `main(argv) -> int`: starts `python -m difflet.cli.main serve --model-id Qwen/Qwen-Image --port P [--cache-dir]` with `DIFFLET_BACKEND=cuda` as a subprocess (its own session), polls `GET /ready` until 200 or `--startup-timeout` (default 900 s), POSTs the body, saves the PNG to `--output`, writes `{"ready_s", "request_s", "image"}` next to it as `serve_timings.json`, terminates the server, exits 0 on success. The `.sh` wrapper mirrors `scripts/cuda_smoke.sh` (log under `/tmp/logs/cuda-serve-smoke-<ts>.log`).

- [ ] **Step 1: Write the failing tests**

```python
import base64
from scripts import cuda_serve_smoke as smoke  # tests/unit/scripts/__init__.py exists; add sys.path handling as tests/unit/scripts/test_cuda_e2e_metrics.py does

def test_chat_body_carries_steps_seed_and_shape():
    body = smoke.build_chat_body("a cat", steps=50, seed=0, height=1024, width=1024)
    assert body["messages"] == [{"role": "user", "content": "a cat"}]
    assert body["extra_body"] == {"num_inference_steps": 50, "seed": 0, "height": 1024, "width": 1024}
    assert "guidance_scale" not in body["extra_body"]

def test_image_data_url_round_trip():
    url = "data:image/png;base64," + base64.b64encode(b"\x89PNG...").decode()
    assert smoke.decode_image_data_url(url) == b"\x89PNG..."
    with pytest.raises(ValueError):
        smoke.decode_image_data_url("https://example.com/x.png")

def test_image_url_is_taken_from_the_first_choice():
    body = {"choices": [{"message": {"content": [{"type": "image_url", "image_url": {"url": "data:x"}}]}}]}
    assert smoke.image_url_from_response(body) == "data:x"
```

Confirm the response shape against `difflet/serving/openai/serving_chat.py` (`content` list with `image_url` entries) before fixing the third assertion.

- [ ] **Step 2: Run to verify failure.** Expected: import error.
- [ ] **Step 3: Implement** (stdlib `urllib.request`, `subprocess`, `json`, `base64`; no requests dependency).
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit**

```bash
git add scripts/cuda_serve_smoke.py scripts/cuda_serve_smoke.sh tests/unit/scripts/test_cuda_serve_smoke.py
git commit -m "scripts: cuda_serve_smoke — start difflet serve on CUDA, fetch one image, record timings"
```

---

### Task 6: Docs, status log, serving suite after, CPU gate

**Files:**
- Modify: `DEVELOPER.md` (CUDA section: serve quick start; "Not in Release 1" loses `difflet serve` (F3b))
- Modify: `docs/plans/2026-10-16-cuda-backend-support.md` (F3 bullet: `difflet serve` landed CPU-side; Status log entry "F3b")
- Modify: this plan (execution ledger)

- [ ] **Step 1: DEVELOPER.md** — after the `difflet run` quick start add `DIFFLET_BACKEND=cuda difflet serve --model-id Qwen/Qwen-Image --port 8091` and `scripts/cuda_serve_smoke.sh --output /tmp/qwen-serve.png`; one paragraph: one resident worker, one opaque stage, requests handled as the CLI handles them, what is refused (`guidance_scale`, other shapes, tp/cp/sp/CFG/TeaCache/shapes with their follow-ups).
- [ ] **Step 2: Serving suite after** — `DIFFLET_BACKEND=cpu PYTHONPATH=$PWD /Users/hang/code/Difflet/.venv/bin/python -m pytest tests/unit/serving -q -p no:cacheprovider -rfEs > /tmp/logs/f3b-serving-after.log`; diff the per-test outcome lines against the before log: identical apart from the new test files. Record counts in the status entry.
- [ ] **Step 3: CPU gate** — `PYTHON_BIN=/Users/hang/code/Difflet/.venv/bin/python .claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh`. Expected: `regressions=0 new_skips=0`, imports ok.
- [ ] **Step 4: Status log entry** (append, format from the skill): host Mac; ran: the two suites and the gate; what landed (Tasks 1–5, test names); rulings (resident worker kept — the engine's one worker process is the server's model process; core id 0 as the device slot; `default_steps=50` / `default_guidance_scale=4.0` on CUDA; guidance refused unless the no-op value; `CompilePolicy.NEVER` unreachable from the CLI, treated as AUTO (manifest only)); **next pod must run:** `scripts/cuda_serve_smoke.sh` (startup smoke + one 50-step request, record `ready_s` / `request_s` / peak memory), then a second request to prove the model stays resident (`request_s` without the load), then `curl` with `extra_body.guidance_scale=1.0` and with `height=768` to see the two 400s, then `difflet serve --tp-degree 2` to see the F6 refusal before the download.
- [ ] **Step 5: Commit**

```bash
git add DEVELOPER.md docs/plans/2026-10-16-cuda-backend-support.md docs/superpowers/plans/2026-10-16-cuda-f3b-serve.md
git commit -m "docs: F3b status — difflet serve on CUDA (CPU-proven, GPU run pending)"
```

---

## Execution ledger (folded from the SDD workspace, 2026-10-16)

```
# SDD ledger — plan: docs/superpowers/plans/2026-10-16-cuda-f3b-serve.md
Spec: docs/plans/2026-10-16-cuda-backend-support.md (F3/F3b, D5, D2, Release 1 scope) — read.
Pre-flight: Task 2/4 consume Task 1's cuda_serving_metadata().pipeline_definition (one opaque "pipeline" stage) and the "cuda" StagePlacement — consistent.
Pre-flight: Task 4 consumes Task 2's adapter_config["compile_policy"] and refuse_unsupported_profile, Task 3's NOOP_GUIDANCE_SCALE — all produced by name; clean.
Pre-flight: Task 4 consumes the CLI module's pipeline_kwargs(args, load=) (exists today) and extracts cuda_pipeline_kwargs from it — additive.
Pre-flight: serving baseline before any change: tests/unit/serving → 526 passed (/tmp/logs/f3b-serving-before.log).
Task 1: complete (commits 5508a06..547181e, tests: env DIFFLET_BACKEND=cpu /Users/hang/code/Difflet/.venv/bin/python -m pytest tests/unit/serving -q -p no:cacheprovider → 534 passed, 1 warning in 10.55s)
Task 2: Ruling: black collapsed one raise line in difflet/serving/types.py after the Task 1 commit (style only, Task 1's commit had it long) — folded into this commit; cost if wrong: none
Task 2: complete (commits 547181e..b240ce6, tests: env DIFFLET_BACKEND=cpu /Users/hang/code/Difflet/.venv/bin/python -m pytest tests/unit/serving -q -p no:cacheprovider → 543 passed, 1 warning in 10.66s)
Task 3: Ruling: the plan's tokenizer fake returned input_ids as a flat list; a real tokenizer with return_tensors='pt' returns a (1, n) batch, so the fake returns [[...]] and the validator counts input_ids[0] — cost if wrong: none (matches the Trainium validator's shape[1])
Task 3: complete (commits b240ce6..2653403, tests: env DIFFLET_BACKEND=cpu /Users/hang/code/Difflet/.venv/bin/python -m pytest tests/unit/serving -q -p no:cacheprovider → 548 passed, 1 warning in 12.48s)
Task 4: complete (commits 2653403..acf4a1f, tests: env DIFFLET_BACKEND=cpu /Users/hang/code/Difflet/.venv/bin/python -m pytest tests/unit/serving tests/unit/cli/test_orchestrator_qwen_cuda.py -q -p no:cacheprovider → 573 passed, 1 warning in 12.32s)
Task 5: complete (commits acf4a1f..289cac8, tests: env DIFFLET_BACKEND=cpu /Users/hang/code/Difflet/.venv/bin/python -m pytest tests/unit/scripts tests/unit/serving -q -p no:cacheprovider → 562 passed, 1 warning in 12.98s)
Task 6: cpu-gate (full): imports=ok unit=[2322 passed, 172 skipped] regressions=0 neuron_skips=129 new_skips=0; tests/unit/serving before 526 / after 556
Task 6: complete (commits 289cac8..bbf9472, tests: env DIFFLET_BACKEND=cpu /Users/hang/code/Difflet/.venv/bin/python -m pytest tests/unit/serving tests/unit/scripts tests/unit/cli/test_orchestrator_qwen_cuda.py -q -p no:cacheprovider → 579 passed, 1 warning in 12.49s)
Final review: fresh reviewer (Agent, opus) on review-5508a06..bbf9472.diff — 0 Critical, 3 Important, 7 Minor.
Final: fixed existing serving tests failing under DIFFLET_BACKEND=cuda (97 failed) — test_legacy_serving_tests_run_on_the_cpu_backend_whatever_the_host_has RED→GREEN via tests/unit/serving/conftest.py; tests/unit/serving under cuda 559 passed, under cpu 559 passed
Final: fixed worker re-resolving refs/main instead of the pinned commit — test_worker_loads_the_commit_the_parent_pinned RED→GREEN (revision=runtime.source.resolved_source_id; exactness test compares all kwargs but revision)
Final: fixed (re-graded Important: contaminates sibling tests) NEURON_RT_* leaking from test_runtime_plan_is_accepted_by_the_resident_worker_environment — test_worker_environment_does_not_leak_out_of_the_previous_test RED (against the committed file: 1 failed) → GREEN with the preserve_worker_environment fixture
Final: fixed (re-graded Important: the pod script would fail after a successful generation) cuda_serve_smoke cannot read S3 image URLs / send DIFFLET_API_KEY — test_image_bytes_come_from_a_data_url_or_an_http_fetch, test_request_headers_carry_the_api_key_when_the_server_requires_one RED→GREEN
Final: fixed (re-graded Important: misleads the pod operator) DEVELOPER.md and the status entry overstated "exactly as difflet run" and the F2/F4 naming — docs now state the three differences (steps cap, guidance accepted only at 4.0, long prompts refused) and that cp/CFG/TeaCache are refused by the shared profile builder's wording; runbook gains the concurrent-request step and the queue/request timeout behaviour
Final: Ruling: 50-step default vs the 30 s image queue timeout (reviewer #3) — kept: the Release 1 measurement is 20.5 s/50 steps, inside the default queue timeout for one waiting request; documented in DEVELOPER.md and added to the pod runbook rather than changing options.py defaults the plan does not own; cost if wrong: a second concurrent client sees 429 queue_timeout until --queue-timeout is raised
Final: Ruling: cp/CFG-parallel/TeaCache on the serve path are refused by build_serving_profile's wording before the preparer names F2/F4 (reviewer #4) — kept: the user is refused either way; the preparer's F2/F4 rows stay as defence in depth; docs corrected; cost if wrong: a less specific message on three flags
Final: Ruling: reviewer's declined-to-judge list (steps cap 1..50, engine restart on a clean cancel, NEURON_RT_VISIBLE_CORES=0 in the CUDA worker, CompilePolicy.NEVER as AUTO, parent importing torch via _replica_count, current_backend raising on a misspelled DIFFLET_BACKEND, commit trailer, GPU footprint, duplicated shape check) — all stand as out of plan, pre-existing, or ledgered rulings; cost if wrong: none ship-blocking
Final: minor (deferred): test_cancelled_request_never_reaches_generate passes without the runner's own throw_if_cancelled (the engine checks first)
Final: minor (deferred): the non-Qwen CUDA refusal names "F1 Wan, F5 Flux" for HunyuanVideo/LTX-2 too, which have no follow-up
Final: minor (deferred): the smoke script's ready_s includes any download the parent performs when the snapshot is not cached (documented in the script docstring instead)
```
