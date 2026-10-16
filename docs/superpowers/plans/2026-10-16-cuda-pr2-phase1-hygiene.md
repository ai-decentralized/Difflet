# CUDA Backend PR 2 — Phase 1 Import Hygiene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `DIFFLET_BACKEND=cpu .claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh` passes on a host with no Neuron SDK and no GPU, with `CPU_GATE_IMPORTS_REQUIRED=1` and an empty `known-red.txt`, so the GPU box (which also lacks the Neuron SDK) can import Difflet, and the CPU CI job turns green.

**Architecture:** Three kinds of change, kept separate so each can be reviewed on its own. (1) Production code: the backend-neutral `canonicalize_shapes` moves out of the Trainium package, the pipeline stops hard-coding Trainium for load ranks, and one utils module defers its Neuron import. (2) The CUDA backend gets its scaffolding and its own hygiene test, so later phases have a place to land. (3) Tests that genuinely need the Neuron SDK — because they import Trainium code or exercise trainium-only registrations — are skipped on hosts without it by one explicit mechanism in `tests/conftest.py`, instead of 167 markers. Nothing here changes behaviour on a Neuron host.

**Tech Stack:** Python ≥ 3.10, pytest, `importlib`, `ast` for the static import scan. No new dependencies beyond declaring `transformers`, which the code already imports.

**Spec:** `docs/plans/2026-10-16-cuda-backend-support.md` — Phase 1 (tasks 1.3–1.10), Global Constraints, and Decisions D1–D19. Conflicts resolve against the Decisions table.

## Global Constraints

- Nothing under `difflet/backends/cuda/` imports `neuronx_distributed`, `torch_neuronx`, `torch_xla`, `nkilib`, `difflet.backends.trainium`, or `difflet.backends.tpu` (spec Global Constraints; Task 2 enforces it).
- Every CPU-host command sets `DIFFLET_BACKEND=cpu`: auto-detect falls back to **trainium** on a host with no accelerator (`difflet/backends/registry.py:_auto_detect_backend`).
- A Trainium compile-cache key must stay byte-identical: only a non-default backend enters the key (`compile_cache.py` lines 93–97).
- Unimplemented paths raise `NotImplementedError` naming the follow-up (`"follow-up F2"` for multi-process).
- Black, line length 100. Tests mirror the source tree under `tests/unit/`. One task, one commit.
- Test command for every task: `.claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh`. It fails on any red test not in `scripts/known-red.txt` and reports known-red tests that turned green; **remove those lines from `known-red.txt` in the same commit.**
- The branch stays local. Do not push, do not open the PR.

## Review Focus

1. A CPU-only host with `DIFFLET_BACKEND` unset still auto-detects `trainium` and dies on import; the gate script and CI set the variable, and Task 1 pins the fallback order so a future "helpful" change to default-to-cpu is a visible decision. (Task 1 test `test_auto_detect_falls_back_to_trainium_without_accelerator`.)
2. The Neuron-gated skip must never fire on a host where the SDK **is** present: a genuine regression there must stay a failure. (Task 9 test `test_neuron_gate_inactive_when_sdk_present`.)
3. The skip must not swallow unrelated exceptions whose message happens to contain "trainium": only the three exact signatures convert. (Task 9 test `test_neuron_gate_ignores_other_errors`.)
4. Moving `canonicalize_shapes` must keep the old import path working for the on-device scripts and serving code that still use it. (Task 3 test `test_trainium_bucketing_reexports_same_object`.)
5. `CudaBackend.prepare_runtime` under `torchrun` with `WORLD_SIZE > 1` must refuse loudly rather than run every rank on GPU 0. (Task 1 test `test_prepare_runtime_multi_process_raises_f2`.)

---

### Task 1: CUDA runtime scaffolding

**Files:**
- Modify: `difflet/backends/cuda/runtime.py`
- Create: `tests/unit/backends/test_cuda_backend.py`

**Interfaces:**
- Produces: `CudaBackend.prepare_runtime(parallel) -> None` — raises `RuntimeError` mentioning `torch.cuda.is_available()` when CUDA is absent; raises `NotImplementedError("... follow-up F2")` when `WORLD_SIZE` env var is set and > 1; otherwise calls `torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))` and creates no process group. `capabilities` unchanged: `requires_aot=False, single_process_multi_core=False, supports_torchrun_mpmd=True`.

- [ ] **Step 1: Write the failing tests** — `tests/unit/backends/test_cuda_backend.py`, modelled on `test_tpu_backend.py` (same `_clear_backend_cache` fixture):

```python
def test_resolve_backend_name_cuda():
    assert registry.resolve_backend_name("cuda") == "cuda"

def test_cuda_capabilities_match_decision_d5():
    caps = registry.get_backend("cuda").capabilities
    assert (caps.requires_aot, caps.single_process_multi_core, caps.supports_torchrun_mpmd) == (False, False, True)

def test_prepare_runtime_without_cuda_raises_clear_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="torch.cuda.is_available"):
        registry.get_backend("cuda").prepare_runtime(parallel=object())

def test_prepare_runtime_multi_process_raises_f2(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(NotImplementedError, match="F2"):
        registry.get_backend("cuda").prepare_runtime(parallel=object())

def test_prepare_runtime_single_process_sets_local_rank_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setenv("LOCAL_RANK", "0")
    calls = []
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: calls.append(d))
    registry.get_backend("cuda").prepare_runtime(parallel=object())
    assert calls == [0]

def test_auto_detect_order_trainium_tpu_cuda_rocm(monkeypatch):
    # torch_neuronx present wins over a CUDA device (Neuron venvs can have both importable)
    present = {"torch_neuronx"}
    monkeypatch.setattr(registry.importlib.util, "find_spec", lambda n: object() if n in present else None)
    assert registry._auto_detect_backend() == "trainium"

def test_auto_detect_cuda_when_device_available(monkeypatch):
    monkeypatch.setattr(registry.importlib.util, "find_spec", lambda n: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert registry._auto_detect_backend() == "cuda"

def test_auto_detect_falls_back_to_trainium_without_accelerator(monkeypatch):
    monkeypatch.setattr(registry.importlib.util, "find_spec", lambda n: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert registry._auto_detect_backend() == "trainium"
```

- [ ] **Step 2: Run to verify failure**

Run: `DIFFLET_BACKEND=cpu .venv/bin/python -m pytest tests/unit/backends/test_cuda_backend.py -q`
Expected: the three `prepare_runtime` tests FAIL (stub raises `NotImplementedError` for all); the rest PASS.

- [ ] **Step 3: Implement `prepare_runtime` in `difflet/backends/cuda/runtime.py`** per the Interfaces block; keep the module docstring honest ("Release 1: single process, single device").

- [ ] **Step 4: Run the file, then the gate**

Run: `DIFFLET_BACKEND=cpu .venv/bin/python -m pytest tests/unit/backends/test_cuda_backend.py -q`
Expected: 8 passed
Run: `.claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh`
Expected: last line contains `regressions=0`

- [ ] **Step 5: Commit** — `git add difflet/backends/cuda/runtime.py tests/unit/backends/test_cuda_backend.py && git commit -m "cuda: runtime scaffolding — prepare_runtime for one process, one device"`

---

### Task 2: CUDA import-hygiene test

**Files:**
- Create: `tests/unit/backends/test_cuda_import_hygiene.py`

**Interfaces:**
- Produces: `FORBIDDEN_PREFIXES = ("neuronx_distributed", "torch_neuronx", "torch_xla", "nkilib", "difflet.backends.trainium", "difflet.backends.tpu")`; `scan_forbidden_imports(root: Path) -> list[tuple[Path, int, str]]` using `ast` (reuse the walk pattern from `scripts/test_imports.sh`). Later tasks and Phase 2 add `difflet/backends/_torch_megatron` to the scanned roots.

- [ ] **Step 1: Write the tests**

```python
SCANNED = [Path("difflet/backends/cuda")]  # Phase 2 appends difflet/backends/_torch_megatron

def test_cuda_backend_has_no_forbidden_imports():
    assert [v for root in SCANNED for v in scan_forbidden_imports(root)] == []

OPS_MODULES = ("attention", "collectives", "embeddings", "linear", "mx", "norm", "platform")

@pytest.mark.parametrize("module", OPS_MODULES)
@pytest.mark.xfail(strict=True, reason="lands in Phase 2 (PR 3)")
def test_cuda_ops_impl_module_imports(module):
    importlib.import_module(f"difflet.backends.cuda.ops_impl.{module}")
```

- [ ] **Step 2: Run**

Run: `DIFFLET_BACKEND=cpu .venv/bin/python -m pytest tests/unit/backends/test_cuda_import_hygiene.py -q`
Expected: 1 passed, 7 xfailed

- [ ] **Step 3: Gate, then commit** — `.claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh` → `regressions=0`; `git commit -m "cuda: import-hygiene test (forbidden-import scan, ops_impl modules xfail until Phase 2)"`

---

### Task 3: Move `canonicalize_shapes` to `difflet/pipeline/bucketing.py`

**Files:**
- Create: `difflet/pipeline/bucketing.py`
- Modify: `difflet/backends/trainium/core/bucketing.py` (keep everything else; import `CompileShape`, `_normalize_shape`, `_shape_sort_key`, `canonicalize_shapes` from the new module and re-export)
- Modify: the 17 import sites listed by `grep -rn "trainium.core.bucketing import canonicalize_shapes" difflet` — rewrite each to `from difflet.pipeline.bucketing import canonicalize_shapes`
- Test: `tests/unit/pipeline/test_bucketing.py`

**Interfaces:**
- Produces: `difflet.pipeline.bucketing.canonicalize_shapes(shapes: Iterable) -> tuple[CompileShape, ...]` and `CompileShape` — identical semantics (the function bodies move verbatim; they depend only on stdlib and `torch`).
- Consumes: nothing new.

- [ ] **Step 1: Write the failing tests**

```python
def test_canonicalize_shapes_importable_without_trainium(monkeypatch):
    for m in [k for k in sys.modules if k.startswith("difflet.backends.trainium")]:
        monkeypatch.delitem(sys.modules, m)
    from difflet.pipeline.bucketing import canonicalize_shapes
    assert canonicalize_shapes([(512, 512), (1024, 1024)]) == ((1024, 1024, None), (512, 512, None))
    assert canonicalize_shapes([(320, 512, 33), (320, 512, 61)]) == ((320, 512, 61), (320, 512, 33))
    assert not any(k.startswith("difflet.backends.trainium") for k in sys.modules)

def test_trainium_bucketing_reexports_same_object():
    from difflet.backends.trainium.core import bucketing as trn
    from difflet.pipeline import bucketing as neutral
    assert trn.canonicalize_shapes is neutral.canonicalize_shapes
```

(The re-export test will skip on this host via Task 9 if the Trainium module cannot import; keep it — it is the Neuron-host guard. The expected tuples are the ones `tests/unit/backends/test_bucketing.py` already asserts.)

- [ ] **Step 2: Run to verify failure** — Expected: `ModuleNotFoundError: difflet.pipeline.bucketing`
- [ ] **Step 3: Create the module, re-export from the Trainium module, rewrite the 17 import sites** (a `sed` over the grep output is fine; re-run the grep afterwards and expect only the re-export line in `backends/trainium/core/bucketing.py`).
- [ ] **Step 4: Run** — `tests/unit/pipeline/test_bucketing.py` → PASS; gate → `regressions=0`, and `known-red` entries that turned green (expect some under `tests/unit/pipeline/` and `models/`) removed from `known-red.txt`.
- [ ] **Step 5: Commit** — `git commit -m "pipeline: backend-neutral canonicalize_shapes; trainium re-exports it"`

---

### Task 4: Load-rank range from the selected backend

**Files:**
- Modify: `difflet/pipeline/difflet_pipeline.py:323-330` (`_resolve_load_rank_range`)
- Modify: `tests/unit/pipeline/test_pipeline.py:387-400`

**Interfaces:**
- Produces: `_resolve_load_rank_range(*, start_rank_id, local_ranks_size) -> tuple[int | None, int | None]` delegating to `get_backend().resolve_load_rank_range(...)` — the backend selected by `DIFFLET_BACKEND`, not `"trainium"`.

- [ ] **Step 1: Adjust the two existing tests** so they select Trainium explicitly (`monkeypatch.setenv("DIFFLET_BACKEND", "trainium")` + `registry._get_backend_by_name.cache_clear()`), keeping their assertions `(2, 1)` and `(0, 4)`; add:

```python
def test_cpu_backend_load_rank_range_is_passthrough(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu"); registry._get_backend_by_name.cache_clear()
    monkeypatch.setenv("WORLD_SIZE", "4"); monkeypatch.setenv("RANK", "2")
    assert _resolve_load_rank_range(start_rank_id=None, local_ranks_size=None) == (None, None)
```

- [ ] **Step 2: Run** — Expected: the new test FAILS with `(2, 1) != (None, None)`.
- [ ] **Step 3: Implement** the one-line delegation.
- [ ] **Step 4: Run file + gate** — Expected: new test PASS; the two Trainium tests PASS on a Neuron host and are Neuron-gated here after Task 9 (until then they stay in `known-red.txt` if red); `regressions=0`.
- [ ] **Step 5: Commit** — `git commit -m "pipeline: resolve load ranks via the selected backend, not trainium"`

---

### Task 5: `difflet/utils/random.py` defers its Neuron import

**Files:**
- Modify: `difflet/utils/random.py:9,19-20`
- Test: `tests/unit/utils/test_random.py` (exists; currently a collection error)

**Interfaces:**
- Produces: `set_seed(seed)` (whatever the public name is in the file) importable without `neuronx_distributed`; the `model_parallel_xla_manual_seed` branch runs only when `importlib.util.find_spec("neuronx_distributed")` is not None and `model_parallel_is_initialized()`.

**Ruling recorded here (spec task 1.6 named nine files):** `accuracy.py`, `benchmark.py`, `constants.py`, `debug_utils.py`, `testing.py`, `tensor_capture_utils.py`, `version_utils.py` have no importers outside `difflet/utils/` (they are NxDI LLM-era leftovers) and are left untouched; `hf_adapter.py`, `diffusers_adapter.py`, `kv_cache_reconstruct_utils.py` and `tensor_replacement/registry.py` import `difflet.backends.trainium.core.config` or attention utils at module level and are Trainium adapters by nature — their tests are Neuron-gated by Task 9 rather than the modules being split. Only `random.py` is reached by backend-neutral code (`utils/testing.py`, `tests/unit/utils/test_random.py`). Cost if wrong: a later phase that needs one of those adapters on CUDA will have to split it then.

- [ ] **Step 1: Run the existing test file to see the collection error** — `DIFFLET_BACKEND=cpu .venv/bin/python -m pytest tests/unit/utils/test_random.py -q` → Expected: `ModuleNotFoundError: neuronx_distributed`.
- [ ] **Step 2: Move the import inside the function behind a `find_spec` check.**
- [ ] **Step 3: Run** — Expected: the file collects; its tests PASS (or, if any assert Neuron seeding, they become Neuron-gated after Task 9). Gate → `regressions=0`; remove the file's ids from `known-red.txt`.
- [ ] **Step 4: Commit** — `git commit -m "utils: random.py imports neuronx_distributed lazily"`

---

### Task 6: `scripts/test_imports.sh` in two tiers

**Files:**
- Modify: `scripts/test_imports.sh` (the embedded Python: split `modules` into `NEUTRAL` and `TRAINIUM` lists; the scripts list is Trainium-only)

**Interfaces:**
- Produces: the script imports `NEUTRAL` always; imports `TRAINIUM` and the `scripts/*.py` probes only when `importlib.util.find_spec("neuronx_distributed")` is not None, printing `skip trainium tier (no neuronx_distributed)` otherwise; the forbidden-import AST guard runs always.
- `NEUTRAL` = `difflet`, `difflet.registry`, `difflet.pipeline.{difflet_pipeline,compile_cache,precision_schedule,parallel_config,bucketing}`, every `difflet.models.<m>.entry`, `difflet.models.{wan.modeling_wan, wan.umt5.modeling_umt5, wan.vae.modeling_vae, hunyuan_video.modeling_hunyuan_video, hunyuan_video.vae.modeling_vae, qwen_image.modeling_qwen, flux.pipeline, wan.pipeline, qwen_image.pipeline, hunyuan_video.pipeline, ltx_2.pipeline}`, `difflet.ops.mx`, `difflet.backends.cpu.ops_impl.mx`, `difflet.backends.cuda.runtime`, `difflet.backends.tpu.runtime`. Any of these that fails to import on this host with `DIFFLET_BACKEND=cpu` is either moved to `TRAINIUM` with a one-line reason in the script, or fixed if the reason is a stray Trainium import (ruling, ledgered).
- `TRAINIUM` = everything else from today's list.

- [ ] **Step 1: Run today's script** — `DIFFLET_BACKEND=cpu PYTHON_BIN=.venv/bin/python ./scripts/test_imports.sh` → Expected: fails at `difflet.models.flux.application`.
- [ ] **Step 2: Restructure the list** per the Interfaces block.
- [ ] **Step 3: Run** — Expected: every `ok import` line for the neutral tier, the skip line, `ok forbidden import guard`, exit 0.
- [ ] **Step 4: Flip the gate default** — in `.claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh` change `${CPU_GATE_IMPORTS_REQUIRED:-0}` to `:-1` and update its header comment. Run the gate → `imports=ok`, `regressions=0`.
- [ ] **Step 5: Commit** — `git commit -m "scripts: test_imports.sh runs the Trainium tier only when the SDK is present; gate requires imports"`

---

### Task 7: Declare `transformers`; document non-Neuron hosts; CI cleanup

**Files:**
- Modify: `pyproject.toml` (`dependencies`: add `"transformers"` with a comment; `[project.optional-dependencies]`: add `cuda = []` with a comment that Phase 0 fills it from the box — D17)
- Modify: `scripts/test_unit.sh`, `scripts/test_imports.sh` header comments: document `PYTHON_BIN` and `DIFFLET_BACKEND=cpu` for hosts without the Neuron venv
- Modify: `.github/workflows/unit-cpu.yml`: drop the explicit `transformers` from the install line and its comment

- [ ] **Step 1: Verify** — `.venv/bin/pip install -e ".[test]" -q && .venv/bin/python -c "import transformers"` → Expected: no error (already installed; the point is the declaration).
- [ ] **Step 2: Edit the four files.**
- [ ] **Step 3: Gate** → `regressions=0`. **Commit** — `git commit -m "deps: declare transformers; document non-Neuron hosts; CI installs from pyproject only"`

---

### Task 8: Compile-cache key knows CUDA

**Files:**
- Modify: `difflet/pipeline/compile_cache.py` (`_BACKEND_EXTRA_TOOLCHAIN_PACKAGES["cuda"] = ("triton",)`; `_backend_extra_toolchain_versions("cuda")` additionally sets `versions["cuda"] = torch.version.cuda` and `versions["cudnn"] = str(torch.backends.cudnn.version())` when CUDA is built in, else `None`)
- Test: `tests/unit/pipeline/test_compile_cache.py`

**Interfaces:**
- Produces: `_backend_extra_toolchain_versions("cuda") -> dict` with keys `triton`, `cuda`, `cudnn`. Trainium specs are untouched: `CacheSpec(... backend="trainium").cache_inputs()` has no `"backend"` key and no CUDA toolchain entries (existing lines 93–97 guarantee it; the test pins it).

- [ ] **Step 1: Write the failing tests** (reuse the file's `_spec` helper if there is one):

```python
def test_cuda_backend_enters_key_and_adds_toolchain():
    a = _spec(backend="cuda"); b = _spec(backend="trainium")
    assert a.cache_key() != b.cache_key()
    inputs = a.cache_inputs()
    assert inputs["backend"] == "cuda"
    assert {"triton", "cuda", "cudnn"} <= set(inputs["toolchain"])

def test_trainium_key_unchanged_by_cuda_support():
    inputs = _spec(backend="trainium").cache_inputs()
    assert "backend" not in inputs
    assert not {"triton", "cuda", "cudnn"} & set(inputs["toolchain"])
    assert _spec(backend="trainium").cache_key() == _spec(backend=None).cache_key()
```

- [ ] **Step 2: Run** — Expected: first test FAILS on the `{"triton","cuda","cudnn"}` subset assertion.
- [ ] **Step 3: Implement.** - [ ] **Step 4: Run file + gate** → PASS, `regressions=0`.
- [ ] **Step 5: Commit** — `git commit -m "compile_cache: cuda backend hashes triton, CUDA and cuDNN versions into the key"`

---

### Task 9: Neuron-gated skips in `tests/conftest.py`

**Files:**
- Modify: `tests/conftest.py`
- Test: `tests/unit/test_conftest_neuron_gate.py`

**Interfaces:**
- Produces, in `tests/conftest.py`: `NEURON_AVAILABLE: bool = importlib.util.find_spec("neuronx_distributed") is not None`; `NEURON_COLLECT_IGNORE: list[str]` — exactly the 27 test modules that fail collection today (the `ERROR tests/...` lines of the gate log), applied via `collect_ignore` only when `not NEURON_AVAILABLE`; `neuron_only_failure(excinfo) -> str | None` returning a skip reason when the exception is one of: `ModuleNotFoundError` whose `.name` starts with `neuronx_distributed`, `torch_neuronx`, `torch_xla`, `nkilib` or `neuronxcc`; `RuntimeError` whose message starts with `"Trainium runtime dependencies are unavailable"`; `ValueError` matching `r"does not support backend 'cpu'; supported backends: [^;]*trainium"`; else `None`. A `pytest_runtest_makereport` hookwrapper converts a failed `call` report to `skipped` with that reason, only when `not NEURON_AVAILABLE`.
- Each ignored module and each converted failure is reported in the pytest summary as a skip with reason `needs Neuron SDK: <signature>`, so the gate's summary line shows the skipped count rising as the failed count falls — nothing disappears silently.

- [ ] **Step 1: Write the tests** (import the helpers from `tests.conftest` or re-import the module by path):

```python
def test_neuron_gate_converts_known_signatures():
    assert neuron_only_failure(_exc(ModuleNotFoundError, name="neuronx_distributed.foo")) is not None
    assert neuron_only_failure(_exc(RuntimeError("Trainium runtime dependencies are unavailable."))) is not None
    assert neuron_only_failure(_exc(ValueError("model 'wan' does not support backend 'cpu'; supported backends: trainium, tpu"))) is not None

def test_neuron_gate_ignores_other_errors():
    assert neuron_only_failure(_exc(ValueError("trainium is mentioned but this is a real bug"))) is None
    assert neuron_only_failure(_exc(ModuleNotFoundError, name="requests")) is None
    assert neuron_only_failure(_exc(AssertionError("x"))) is None

def test_neuron_gate_inactive_when_sdk_present(pytester, monkeypatch):
    # with NEURON_AVAILABLE forced True, a test raising the RuntimeError signature stays FAILED
    ...
```

(`pytester` needs `pytest_plugins = ["pytester"]` in the test module or `-p pytester`; alternatively assert on the hook function's early return when `NEURON_AVAILABLE` is monkeypatched True.)

- [ ] **Step 2: Run** — Expected: `ImportError: cannot import name 'neuron_only_failure'`.
- [ ] **Step 3: Implement in `tests/conftest.py`.** Keep the existing docstring and extend it with why this exists (one paragraph: non-Neuron hosts, nothing hidden on Neuron hosts).
- [ ] **Step 4: Run the whole gate** — Expected: `unit=[... passed, 0 failed, N skipped ...]` with `fixed=<all remaining>`; empty `known-red.txt` (`: > .claude/skills/difflet-cuda-backend/scripts/known-red.txt`); `regressions=0`; exit 0.
- [ ] **Step 5: Commit** — `git commit -m "tests: skip Neuron-only tests on hosts without the SDK, by exact signature"`

---

### Task 10: Status log and CI check

**Files:**
- Modify: `docs/plans/2026-10-16-cuda-backend-support.md` (Status log, append only; tick the Phase 1 checkboxes 1.3–1.10)

- [ ] **Step 1: Run the gate one more time from a clean shell** — `.claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh` → Expected: `imports=ok`, `0 failed`, `known_red=0`, `regressions=0`, exit 0.
- [ ] **Step 2: Append the status-log entry** in the format the project skill gives: host (this Mac, torch 2.14.1), the before/after counts (before: 2088 passed / 131 failed / 36 errors / 43 skipped), every ruling made in this plan (Task 5's, any from Task 6), and `next: Phase 2 — Task 2.1 lift`.
- [ ] **Step 3: Track execution plans too** — `.gitignore` gains `!docs/superpowers/` + `docs/superpowers/*` + `!docs/superpowers/plans/` + `docs/superpowers/plans/*` + `!docs/superpowers/plans/*.md` (same pattern as `docs/plans/`; extends D18 to the execution plans, which today are only tracked by force-add). `git check-ignore docs/superpowers/plans/2026-10-16-cuda-pr2-phase1-hygiene.md` → Expected: exit 1 (not ignored). Add this plan file.
- [ ] **Step 4: Commit** — `git commit -m "plan: Phase 1 done — status log, checkboxes, track execution plans"`
- [ ] **Step 5: Do not push.** Report the rulings and deferred minors in the final message, per executing-plans.
