# DP Replication + Request Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Data parallelism for Difflet on Trainium — replica-per-process full-model replication with a request router in the CLI, per the approved spec `docs/superpowers/specs/2026-07-06-dp-replication-routing-design.md`.

**Architecture:** The CLI parent becomes a router: it writes a file-based request manifest, spawns k worker subprocesses (each pinned to a disjoint NeuronCore range via `NEURON_RT_VISIBLE_CORES`, each running today's dp=1 stage chain), and gathers per-request status files. The dp axis never enters a compiled graph — workers always build `DiffletParallelConfig(dp_degree=1)`, so replicas reuse the dp=1 compile cache and no dp collective can exist.

**Tech Stack:** Python 3.10, argparse CLI, subprocess/Popen, pytest (+pytest-cov), safetensors header parsing (stdlib json/struct only), torch only where already imported.

## Global Constraints

- **After every task: commit AND `git push`** (user requirement). Branch: `worktree-dp`, upstream already set.
- **≥ 90% line coverage** on all new CPU-testable modules (`difflet/cli/dp/*`, `difflet/cli/modes.py`) — spec requirement; repo bar is ~91%.
- **Every new stage-level flag must be added to BOTH `difflet/cli/main.py` parsers AND `difflet/cli/stage.py:_build_stage_parser` AND each orchestrator's `_shared_cli_args`** — the stage parser uses `parse_known_args` and *silently drops* unknown flags (this exact bug shipped before: commit `49b22c5` "stage parser was silently dropping --sp").
- Back-compat: `difflet generate --model-id … --prompt … --output …` with no `--dp/--mode/--requests` must behave identically to today (no router in the loop; failures propagate as before).
- TeaCache flags + batch/DP mode are mutually exclusive (validated, clear error).
- New code is black-formatted, line length 100 (`pyproject.toml`).
- Tests mirror the source tree under `tests/unit/` (repo convention).
- On-device tasks (15–16) require trn2 hardware; long runs must be detached (`setsid`) with output logged to files — the harness kills foreground background-Bash jobs. Exit-zero is NOT verification: check mode-specific compiled dirs, compile times, and output artifacts.

## File Structure

New:
- `difflet/cli/dp/__init__.py` — empty package marker
- `difflet/cli/dp/requests_io.py` — `RequestSpec`, JSONL parsing/validation, manifest read/write
- `difflet/cli/dp/claims.py` — atomic claim protocol + done/failed markers + summary
- `difflet/cli/dp/router.py` — core ranges, worker env/command, spawn, gather, `run_router`
- `difflet/cli/dp/stage_loop.py` — request iteration helpers used inside stages/orchestrators
- `difflet/cli/dp/hbm_check.py` — safetensors weight-bytes scan + 96 GB assert
- `difflet/cli/modes.py` — mode table (latency/throughput/mixed × model class)
- `scripts/verify_dp_correctness.py` — on-device dp=k vs dp=1 latent comparison
- Tests: `tests/unit/cli/dp/test_requests_io.py`, `test_claims.py`, `test_router.py`, `test_stage_loop.py`, `test_hbm_check.py`; `tests/unit/cli/test_modes.py`, `tests/unit/cli/test_main_dp.py`, `tests/unit/cli/test_dp_isolation.py`; `tests/unit/cli/orchestrators/test_batch_wiring.py`

Modified:
- `difflet/cli/main.py` — new flags, validation, router dispatch
- `difflet/cli/stage.py` — stage parser gains the new flags
- `difflet/cli/orchestrators/wan.py`, `hunyuan_video.py`, `qwen_image.py` — staged batch loops
- `difflet/cli/orchestrators/flux.py`, `ltx_2.py` — in-process batch loops
- `scripts/` verify_cli matrix — mode × model rows

## Shared protocol (used by Tasks 1–2, referenced everywhere)

A DP run's `requests_dir` contains, per request index i (zero-padded to 4):
- `req_000i.json` — the RequestSpec (written once by the router)
- `req_000i.claim` — created by the claiming worker; content = worker index as str
- `req_000i.done` — created by the final stage on success (empty)
- `req_000i.failed` — created on failure; content = traceback text

Schedules: `round_robin` pre-assigns `assigned_worker = i % dp` in the manifest; a worker claims only its own. `least_loaded` leaves `assigned_worker = None`; any worker may claim via atomic `O_CREAT|O_EXCL`.

---

### Task 1: RequestSpec + JSONL + manifest (`requests_io.py`)

**Files:**
- Create: `difflet/cli/dp/__init__.py` (empty), `difflet/cli/dp/requests_io.py`
- Test: `tests/unit/cli/dp/test_requests_io.py` (+ empty `tests/unit/cli/__init__.py`, `tests/unit/cli/dp/__init__.py` if the suite needs packages — mirror existing tests/unit convention: check `ls tests/unit/backends` for `__init__.py` presence and copy it)

**Interfaces (Produces):**
```python
@dataclass(frozen=True)
class RequestSpec:
    index: int
    prompt: str
    output: str
    seed: int = 42
    negative_prompt: str | None = None
    guidance_scale: float | None = None
    steps: int | None = None
    assigned_worker: int | None = None

def load_requests_jsonl(path: str | Path) -> list[RequestSpec]   # ValueError on bad JSON, missing prompt/output, duplicate outputs, empty file
def write_manifest(requests: list[RequestSpec], requests_dir: Path) -> None
def read_manifest(requests_dir: Path) -> list[RequestSpec]        # sorted by index
def request_path(requests_dir: Path, index: int) -> Path          # requests_dir / f"req_{index:04d}.json"
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/cli/dp/test_requests_io.py
import json
import pytest
from difflet.cli.dp.requests_io import (
    RequestSpec, load_requests_jsonl, read_manifest, request_path, write_manifest,
)


def _write_jsonl(tmp_path, lines):
    p = tmp_path / "requests.jsonl"
    p.write_text("\n".join(json.dumps(l) for l in lines) + "\n", encoding="utf-8")
    return p


def test_load_minimal_jsonl(tmp_path):
    p = _write_jsonl(tmp_path, [
        {"prompt": "a cat", "output": "a.png"},
        {"prompt": "a dog", "output": "b.png", "seed": 7, "steps": 12,
         "guidance_scale": 5.0, "negative_prompt": "blurry"},
    ])
    reqs = load_requests_jsonl(p)
    assert [r.index for r in reqs] == [0, 1]
    assert reqs[0].seed == 42 and reqs[0].steps is None
    assert reqs[1].seed == 7 and reqs[1].guidance_scale == 5.0
    assert reqs[1].negative_prompt == "blurry"


def test_load_rejects_duplicate_outputs(tmp_path):
    p = _write_jsonl(tmp_path, [
        {"prompt": "x", "output": "same.png"},
        {"prompt": "y", "output": "same.png"},
    ])
    with pytest.raises(ValueError, match="duplicate output"):
        load_requests_jsonl(p)


def test_load_rejects_missing_fields_and_empty(tmp_path):
    with pytest.raises(ValueError, match="line 1"):
        load_requests_jsonl(_write_jsonl(tmp_path, [{"prompt": "no output"}]))
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no requests"):
        load_requests_jsonl(empty)


def test_load_rejects_bad_json(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"prompt": "ok", "output": "a.png"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_requests_jsonl(p)


def test_manifest_roundtrip(tmp_path):
    reqs = [
        RequestSpec(index=0, prompt="a", output="a.png", assigned_worker=0),
        RequestSpec(index=1, prompt="b", output="b.png", assigned_worker=1),
    ]
    write_manifest(reqs, tmp_path)
    assert request_path(tmp_path, 1).name == "req_0001.json"
    back = read_manifest(tmp_path)
    assert back == reqs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/cli/dp/test_requests_io.py -v`
Expected: FAIL / collection error — `ModuleNotFoundError: difflet.cli.dp`

- [ ] **Step 3: Implement**

```python
# difflet/cli/dp/requests_io.py
"""Request manifest I/O for the DP router (spec 2026-07-06 §Router)."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

_OPTIONAL_FIELDS = ("negative_prompt", "guidance_scale", "steps")


@dataclass(frozen=True)
class RequestSpec:
    index: int
    prompt: str
    output: str
    seed: int = 42
    negative_prompt: str | None = None
    guidance_scale: float | None = None
    steps: int | None = None
    assigned_worker: int | None = None


def load_requests_jsonl(path: str | Path) -> list[RequestSpec]:
    requests: list[RequestSpec] = []
    text = Path(path).read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: line {lineno}: invalid JSON: {exc}") from exc
        for field in ("prompt", "output"):
            if not data.get(field):
                raise ValueError(f"{path}: line {lineno}: missing required field {field!r}")
        kwargs = {k: data[k] for k in _OPTIONAL_FIELDS if k in data}
        requests.append(
            RequestSpec(
                index=len(requests),
                prompt=str(data["prompt"]),
                output=str(data["output"]),
                seed=int(data.get("seed", 42)),
                **kwargs,
            )
        )
    if not requests:
        raise ValueError(f"{path}: no requests found")
    outputs = [r.output for r in requests]
    dupes = {o for o in outputs if outputs.count(o) > 1}
    if dupes:
        raise ValueError(f"{path}: duplicate output paths: {sorted(dupes)}")
    return requests


def request_path(requests_dir: Path, index: int) -> Path:
    return Path(requests_dir) / f"req_{index:04d}.json"


def write_manifest(requests: list[RequestSpec], requests_dir: Path) -> None:
    requests_dir = Path(requests_dir)
    requests_dir.mkdir(parents=True, exist_ok=True)
    for req in requests:
        request_path(requests_dir, req.index).write_text(
            json.dumps(dataclasses.asdict(req)), encoding="utf-8"
        )


def read_manifest(requests_dir: Path) -> list[RequestSpec]:
    specs = []
    for p in sorted(Path(requests_dir).glob("req_*.json")):
        specs.append(RequestSpec(**json.loads(p.read_text(encoding="utf-8"))))
    return specs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/cli/dp/test_requests_io.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit and push**

```bash
git add difflet/cli/dp/ tests/unit/cli/
git commit -m "feat(dp): request manifest I/O for the DP router"
git push
```

---

### Task 2: Claim protocol (`claims.py`)

**Files:**
- Create: `difflet/cli/dp/claims.py`
- Test: `tests/unit/cli/dp/test_claims.py`

**Interfaces:**
- Consumes: `RequestSpec`, `read_manifest`, `request_path` from Task 1.
- Produces:
```python
@dataclass
class RouterSummary:
    done: list[int]
    failed: dict[int, str]      # index -> error text
    unfinished: list[int]       # claimed or assigned but neither done nor failed

def claim_next(requests_dir: Path, worker_index: int, schedule: str) -> RequestSpec | None
def claimed_by(requests_dir: Path, worker_index: int) -> list[RequestSpec]   # this worker's claims, manifest order
def mark_done(requests_dir: Path, index: int) -> None
def mark_failed(requests_dir: Path, index: int, error: str) -> None
def is_failed(requests_dir: Path, index: int) -> bool
def summarize(requests_dir: Path) -> RouterSummary
```
`claim_next` semantics — `least_loaded`: first manifest request whose `.claim` can be created with `O_CREAT|O_EXCL` (content = worker index). `round_robin`: first request with `assigned_worker == worker_index` and no `.claim` yet (claim file still written, for accounting). Returns `None` when nothing is left.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/cli/dp/test_claims.py
from concurrent.futures import ThreadPoolExecutor

from difflet.cli.dp.claims import (
    claim_next, claimed_by, is_failed, mark_done, mark_failed, summarize,
)
from difflet.cli.dp.requests_io import RequestSpec, write_manifest


def _manifest(tmp_path, n, dp=None):
    reqs = [
        RequestSpec(index=i, prompt=f"p{i}", output=f"o{i}.png",
                    assigned_worker=(i % dp if dp else None))
        for i in range(n)
    ]
    write_manifest(reqs, tmp_path)
    return reqs


def test_round_robin_claims_only_own(tmp_path):
    _manifest(tmp_path, 5, dp=2)
    got = []
    while (req := claim_next(tmp_path, 0, "round_robin")) is not None:
        got.append(req.index)
    assert got == [0, 2, 4]
    assert claim_next(tmp_path, 0, "round_robin") is None
    assert [r.index for r in claimed_by(tmp_path, 0)] == [0, 2, 4]


def test_least_loaded_exhausts_queue(tmp_path):
    _manifest(tmp_path, 4)
    a = claim_next(tmp_path, 0, "least_loaded")
    b = claim_next(tmp_path, 1, "least_loaded")
    assert {a.index, b.index} == {0, 1}
    assert claim_next(tmp_path, 1, "least_loaded").index == 2
    assert claim_next(tmp_path, 0, "least_loaded").index == 3
    assert claim_next(tmp_path, 0, "least_loaded") is None


def test_least_loaded_claims_are_race_safe(tmp_path):
    _manifest(tmp_path, 64)
    def drain(w):
        got = []
        while (req := claim_next(tmp_path, w, "least_loaded")) is not None:
            got.append(req.index)
        return got
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(drain, range(8)))
    all_claims = [i for r in results for i in r]
    assert sorted(all_claims) == list(range(64))          # no dupes, no losses


def test_markers_and_summary(tmp_path):
    _manifest(tmp_path, 3)
    for _ in range(3):
        claim_next(tmp_path, 0, "least_loaded")
    mark_done(tmp_path, 0)
    mark_failed(tmp_path, 1, "boom\ntrace")
    assert is_failed(tmp_path, 1) and not is_failed(tmp_path, 0)
    s = summarize(tmp_path)
    assert s.done == [0]
    assert s.failed == {1: "boom\ntrace"}
    assert s.unfinished == [2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/cli/dp/test_claims.py -v`
Expected: FAIL — `ModuleNotFoundError: difflet.cli.dp.claims`

- [ ] **Step 3: Implement**

```python
# difflet/cli/dp/claims.py
"""Filesystem claim protocol: the requests dir IS the queue (spec §Router)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from difflet.cli.dp.requests_io import RequestSpec, read_manifest


@dataclass
class RouterSummary:
    done: list[int]
    failed: dict[int, str]
    unfinished: list[int]


def _marker(requests_dir: Path, index: int, kind: str) -> Path:
    return Path(requests_dir) / f"req_{index:04d}.{kind}"


def _try_claim(requests_dir: Path, index: int, worker_index: int) -> bool:
    try:
        fd = os.open(_marker(requests_dir, index, "claim"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as handle:
        handle.write(str(worker_index))
    return True


def claim_next(requests_dir: Path, worker_index: int, schedule: str) -> RequestSpec | None:
    if schedule not in ("round_robin", "least_loaded"):
        raise ValueError(f"unknown schedule {schedule!r}")
    for req in read_manifest(requests_dir):
        if schedule == "round_robin" and req.assigned_worker != worker_index:
            continue
        if _try_claim(requests_dir, req.index, worker_index):
            return req
    return None


def claimed_by(requests_dir: Path, worker_index: int) -> list[RequestSpec]:
    mine = []
    for req in read_manifest(requests_dir):
        claim = _marker(requests_dir, req.index, "claim")
        if claim.exists() and claim.read_text(encoding="utf-8") == str(worker_index):
            mine.append(req)
    return mine


def mark_done(requests_dir: Path, index: int) -> None:
    _marker(requests_dir, index, "done").touch()


def mark_failed(requests_dir: Path, index: int, error: str) -> None:
    _marker(requests_dir, index, "failed").write_text(error, encoding="utf-8")


def is_failed(requests_dir: Path, index: int) -> bool:
    return _marker(requests_dir, index, "failed").exists()


def summarize(requests_dir: Path) -> RouterSummary:
    done, failed, unfinished = [], {}, []
    for req in read_manifest(requests_dir):
        if _marker(requests_dir, req.index, "done").exists():
            done.append(req.index)
        elif is_failed(requests_dir, req.index):
            failed[req.index] = _marker(requests_dir, req.index, "failed").read_text(
                encoding="utf-8"
            )
        else:
            unfinished.append(req.index)
    return RouterSummary(done=done, failed=failed, unfinished=unfinished)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/cli/dp/test_claims.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit and push**

```bash
git add difflet/cli/dp/claims.py tests/unit/cli/dp/test_claims.py
git commit -m "feat(dp): atomic filesystem claim protocol for request routing"
git push
```

---

### Task 3: Mode table (`difflet/cli/modes.py`)

**Files:**
- Create: `difflet/cli/modes.py`
- Test: `tests/unit/cli/test_modes.py`

**Interfaces:**
- Produces:
```python
@dataclass(frozen=True)
class ModeConfig:
    dp: int
    cfg_parallel: bool
    cp_degree: int

MODEL_CLASS: dict[str, str]   # model_id -> "distilled" | "true_cfg" | "true_cfg_no_cp"
def resolve_mode(model_id: str, mode: str | None, args) -> ModeConfig | None
```
`resolve_mode` returns `None` when `mode is None`. Explicit CLI flags override mode fields: `args.dp is not None` overrides dp, `args.cp_degree is not None` overrides cp, `args.cfg_parallel is True` overrides cfg (a distilled model with explicit `--cfg-parallel` is still rejected later by `_validate_cfg_parallel` — unchanged). Rules: distilled ⇒ cfg False in every mode; `true_cfg_no_cp` (LTX-2) ⇒ cp forced 1.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/cli/test_modes.py
import argparse

import pytest

from difflet.cli.modes import MODEL_CLASS, ModeConfig, resolve_mode

WAN = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
FLUX = "black-forest-labs/FLUX.1-dev"
HYV = "hunyuanvideo-community/HunyuanVideo"
HYV15 = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
QWEN = "Qwen/Qwen-Image"
LTX = "Lightricks/LTX-2"


def _args(**kw):
    ns = argparse.Namespace(dp=None, cp_degree=None, cfg_parallel=False)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_all_six_models_classified():
    for mid in (WAN, FLUX, HYV, HYV15, QWEN, LTX):
        assert mid in MODEL_CLASS


@pytest.mark.parametrize("mid", [FLUX, HYV, HYV15, QWEN])
def test_distilled_never_gets_cfg2(mid):
    for mode in ("latency", "throughput", "mixed"):
        assert resolve_mode(mid, mode, _args()).cfg_parallel is False


def test_mode_table_matches_spec():
    assert resolve_mode(FLUX, "latency", _args()) == ModeConfig(1, False, 4)
    assert resolve_mode(FLUX, "throughput", _args()) == ModeConfig(4, False, 1)
    assert resolve_mode(FLUX, "mixed", _args()) == ModeConfig(2, False, 2)
    assert resolve_mode(WAN, "latency", _args()) == ModeConfig(1, True, 1)
    assert resolve_mode(WAN, "throughput", _args()) == ModeConfig(4, False, 1)
    assert resolve_mode(WAN, "mixed", _args()) == ModeConfig(2, True, 1)


def test_ltx2_cp_always_capped():
    for mode in ("latency", "throughput", "mixed"):
        cfg = resolve_mode(LTX, mode, _args())
        assert cfg.cp_degree == 1


def test_explicit_flags_override_mode():
    cfg = resolve_mode(FLUX, "throughput", _args(dp=2, cp_degree=2))
    assert cfg == ModeConfig(2, False, 2)


def test_no_mode_returns_none_and_unknown_raises():
    assert resolve_mode(FLUX, None, _args()) is None
    with pytest.raises(ValueError, match="unknown mode"):
        resolve_mode(FLUX, "warp", _args())
    with pytest.raises(ValueError, match="unknown model"):
        resolve_mode("nope/nope", "latency", _args())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/cli/test_modes.py -v`
Expected: FAIL — `ModuleNotFoundError: difflet.cli.modes`

- [ ] **Step 3: Implement**

```python
# difflet/cli/modes.py
"""Runtime-mode table: latency / throughput / mixed → per-model-class dp/cfg/cp.

Spec: docs/superpowers/specs/2026-07-06-dp-replication-routing-design.md.
Rules (encoded, not per-cell): distilled ⇒ cfg forced 1; LTX-2 ⇒ cp capped 1;
distilled mixed backfills the freed cfg lane with cp=2. When HunyuanVideo-1.5 /
Qwen-Image gain true CFG, only their MODEL_CLASS entry flips.
"""

from __future__ import annotations

from dataclasses import dataclass

MODES = ("latency", "throughput", "mixed")

MODEL_CLASS: dict[str, str] = {
    "black-forest-labs/FLUX.1-dev": "distilled",
    "hunyuanvideo-community/HunyuanVideo": "distilled",
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v": "distilled",
    "Qwen/Qwen-Image": "distilled",
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": "true_cfg",
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": "true_cfg",
    "Lightricks/LTX-2": "true_cfg_no_cp",
}


@dataclass(frozen=True)
class ModeConfig:
    dp: int
    cfg_parallel: bool
    cp_degree: int


_BASE = {
    ("latency", "distilled"): ModeConfig(dp=1, cfg_parallel=False, cp_degree=4),
    ("latency", "true_cfg"): ModeConfig(dp=1, cfg_parallel=True, cp_degree=1),
    ("throughput", "distilled"): ModeConfig(dp=4, cfg_parallel=False, cp_degree=1),
    ("throughput", "true_cfg"): ModeConfig(dp=4, cfg_parallel=False, cp_degree=1),
    ("mixed", "distilled"): ModeConfig(dp=2, cfg_parallel=False, cp_degree=2),
    ("mixed", "true_cfg"): ModeConfig(dp=2, cfg_parallel=True, cp_degree=1),
}


def resolve_mode(model_id: str, mode: str | None, args) -> ModeConfig | None:
    if mode is None:
        return None
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if model_id not in MODEL_CLASS:
        raise ValueError(f"unknown model {model_id!r}")
    klass = MODEL_CLASS[model_id]
    base = _BASE[(mode, "true_cfg" if klass == "true_cfg_no_cp" else klass)]
    cp = 1 if klass == "true_cfg_no_cp" else base.cp_degree
    return ModeConfig(
        dp=args.dp if getattr(args, "dp", None) is not None else base.dp,
        cfg_parallel=True if getattr(args, "cfg_parallel", False) else base.cfg_parallel,
        cp_degree=args.cp_degree if getattr(args, "cp_degree", None) is not None else cp,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/cli/test_modes.py -v`
Expected: all PASS (9 incl. parametrized)

- [ ] **Step 5: Commit and push**

```bash
git add difflet/cli/modes.py tests/unit/cli/test_modes.py
git commit -m "feat(cli): latency/throughput/mixed mode table with class rules"
git push
```

---

### Task 4: HBM fit assertion (`hbm_check.py`)

**Files:**
- Create: `difflet/cli/dp/hbm_check.py`
- Test: `tests/unit/cli/dp/test_hbm_check.py`

**Interfaces:**
- Produces:
```python
HBM_LIMIT_BYTES = 96_000_000_000
def component_weight_bytes(model_path: str | Path, dtype_bytes: int = 2) -> dict[str, int]
def assert_replica_fits(model_path, *, dtype_bytes: int = 2, limit_bytes: int = HBM_LIMIT_BYTES) -> None  # raises RuntimeError with breakdown
```
Parses safetensors headers only (format: 8-byte little-endian header length, then a JSON dict `{tensor_name: {"dtype": …, "shape": […], "data_offsets": […]}, "__metadata__": …}`). Element count = prod(shape); bytes = count × `dtype_bytes` (runtime dtype, bf16 default — NOT the on-disk dtype, since weights are cast at load). Component = first-level subdirectory of `model_path` containing `*.safetensors` (recursive glob).

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/cli/dp/test_hbm_check.py
import json
import struct

import pytest

from difflet.cli.dp.hbm_check import assert_replica_fits, component_weight_bytes


def _write_safetensors(path, tensors):
    """Minimal valid safetensors file: header only, zero-filled data."""
    header, offset = {}, 0
    for name, shape in tensors.items():
        n = 1
        for d in shape:
            n *= d
        header[name] = {"dtype": "BF16", "shape": list(shape),
                        "data_offsets": [offset, offset + 2 * n]}
        offset += 2 * n
    blob = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * offset)


def test_component_weight_bytes(tmp_path):
    _write_safetensors(tmp_path / "transformer" / "model.safetensors",
                       {"w1": (1024, 1024), "w2": (512,)})
    _write_safetensors(tmp_path / "vae" / "diffusion_pytorch_model.safetensors",
                       {"conv": (16, 16, 3, 3)})
    sizes = component_weight_bytes(tmp_path, dtype_bytes=2)
    assert sizes["transformer"] == 2 * (1024 * 1024 + 512)
    assert sizes["vae"] == 2 * (16 * 16 * 3 * 3)


def test_assert_replica_fits_passes_under_limit(tmp_path):
    _write_safetensors(tmp_path / "transformer" / "m.safetensors", {"w": (10, 10)})
    assert_replica_fits(tmp_path, limit_bytes=1_000_000)   # no raise


def test_assert_replica_fits_raises_with_breakdown(tmp_path):
    _write_safetensors(tmp_path / "transformer" / "m.safetensors", {"w": (1000, 1000)})
    with pytest.raises(RuntimeError) as exc:
        assert_replica_fits(tmp_path, limit_bytes=1_000_000)
    assert "transformer" in str(exc.value) and "96" not in str(exc.value).split("limit")[0]


def test_no_safetensors_raises(tmp_path):
    with pytest.raises(RuntimeError, match="no safetensors"):
        component_weight_bytes(tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/cli/dp/test_hbm_check.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# difflet/cli/dp/hbm_check.py
"""Replica HBM fit check: weights-only, header-only safetensors scan (spec §CLI)."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

HBM_LIMIT_BYTES = 96_000_000_000  # one Trainium2 chip


def _file_param_count(path: Path) -> int:
    with path.open("rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_len))
    count = 0
    for name, info in header.items():
        if name == "__metadata__":
            continue
        count += math.prod(info["shape"]) if info["shape"] else 1
    return count


def component_weight_bytes(model_path: str | Path, dtype_bytes: int = 2) -> dict[str, int]:
    model_path = Path(model_path)
    sizes: dict[str, int] = {}
    for st_file in sorted(model_path.rglob("*.safetensors")):
        component = st_file.relative_to(model_path).parts[0]
        sizes[component] = sizes.get(component, 0) + dtype_bytes * _file_param_count(st_file)
    if not sizes:
        raise RuntimeError(f"no safetensors found under {model_path}")
    return sizes


def assert_replica_fits(
    model_path: str | Path, *, dtype_bytes: int = 2, limit_bytes: int = HBM_LIMIT_BYTES
) -> None:
    sizes = component_weight_bytes(model_path, dtype_bytes=dtype_bytes)
    total = sum(sizes.values())
    if total > limit_bytes:
        breakdown = ", ".join(f"{k}={v / 1e9:.1f}GB" for k, v in sorted(sizes.items()))
        raise RuntimeError(
            f"replica weights {total / 1e9:.1f}GB exceed the per-replica HBM "
            f"limit {limit_bytes / 1e9:.0f}GB ({breakdown})"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/cli/dp/test_hbm_check.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit and push**

```bash
git add difflet/cli/dp/hbm_check.py tests/unit/cli/dp/test_hbm_check.py
git commit -m "feat(dp): replica HBM fit assertion from safetensors headers"
git push
```

---

### Task 5: Router core (`router.py`)

**Files:**
- Create: `difflet/cli/dp/router.py`
- Test: `tests/unit/cli/dp/test_router.py`

**Interfaces:**
- Consumes: Tasks 1–2 (`write_manifest`, `RequestSpec`, `summarize`, `RouterSummary`), Task 4 (`assert_replica_fits`).
- Produces:
```python
def replica_core_ranges(dp: int, replica_cores: int) -> list[str]              # ["0-3", "4-7"] (range "i" when replica_cores == 1)
def worker_env(base_env: Mapping[str, str], core_range: str, replica_cores: int) -> dict[str, str]
def worker_cli_args(args) -> list[str]                                          # passthrough flags, NEVER --dp / --mode / --prompt / --output / --requests
def run_router(args, requests: list[RequestSpec], *, replica_cores: int,
               worker_argv_prefix: list[str] | None = None) -> int
```
`worker_env` sets `NEURON_RT_VISIBLE_CORES=<range>` and `NEURON_RT_NUM_CORES=<replica_cores>` (plain assignment — these MUST override, not setdefault, because `run_stage` inside the worker uses setdefault and would otherwise see the parent's values). `run_router`: assign workers if `round_robin` → `write_manifest` → HBM check → spawn k `subprocess.Popen(worker_argv_prefix + ["generate"] + worker_cli_args(args) + ["--requests-dir", …, "--worker-index", str(w), "--dp-schedule", schedule, "--work-dir", str(work_dir / f"worker_{w}")], env=…)` → wait all → for `round_robin`, mark requests assigned to a non-zero-exit worker that are neither done nor failed as failed(`"worker crashed"`); claimed-but-unfinished likewise → print summary → return 0 iff `summarize().failed == {}` and `unfinished == []`. `worker_argv_prefix` defaults to `[sys.executable, "-m", "difflet.cli.main"]`; tests inject a stub prefix.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/cli/dp/test_router.py
import argparse
import os
import sys
import textwrap

import pytest

from difflet.cli.dp.claims import summarize
from difflet.cli.dp.requests_io import RequestSpec
from difflet.cli.dp.router import (
    replica_core_ranges, run_router, worker_cli_args, worker_env,
)


def test_replica_core_ranges():
    assert replica_core_ranges(4, 4) == ["0-3", "4-7", "8-11", "12-15"]
    assert replica_core_ranges(2, 1) == ["0", "1"]


def test_worker_env_overrides_parent_values():
    base = {"NEURON_RT_NUM_CORES": "16", "PATH": "/bin"}
    env = worker_env(base, "4-7", 4)
    assert env["NEURON_RT_VISIBLE_CORES"] == "4-7"
    assert env["NEURON_RT_NUM_CORES"] == "4"
    assert env["PATH"] == "/bin"


def _args(**kw):
    defaults = dict(
        model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers", tp_degree=4, cp_degree=None,
        cp_mode="gather_kv", cfg_parallel=False, sp_enabled=False, height=None,
        width=None, num_frames=None, steps=None, guidance_scale=None, seed=42,
        cache_dir=None, work_dir=None, keep_work_dir=False, dp=2,
        dp_schedule="round_robin", requests=None, revision=None, force=False,
        teacache_cadence=None, teacache_online_delta=None, teacache_speedup=None,
        teacache_calibration=None, prompt=None, output=None,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def test_worker_cli_args_never_leak_router_flags():
    argv = worker_cli_args(_args(cp_degree=2, cfg_parallel=True, height=480))
    text = " ".join(argv)
    assert "--dp " not in text + " " and "--mode" not in text
    assert "--prompt" not in text and "--requests " not in text + " "
    assert "--model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers" in text
    assert "--cp-degree 2" in text and "--cfg-parallel" in text and "--height 480" in text


# A stub "worker" used instead of difflet.cli.main: claims and completes requests.
_STUB = textwrap.dedent("""
    import argparse, sys
    from difflet.cli.dp import claims
    p = argparse.ArgumentParser()
    p.add_argument("command")
    p.add_argument("--requests-dir", required=True)
    p.add_argument("--worker-index", type=int, required=True)
    p.add_argument("--dp-schedule", required=True)
    p.add_argument("--fail-index", type=int, default=None)
    p.add_argument("--crash-worker", type=int, default=None)
    args, _ = p.parse_known_args()
    if args.crash_worker == args.worker_index:
        sys.exit(3)
    while (req := claims.claim_next(args.requests_dir, args.worker_index,
                                    args.dp_schedule)) is not None:
        if args.fail_index == req.index:
            claims.mark_failed(args.requests_dir, req.index, "stub failure")
        else:
            claims.mark_done(args.requests_dir, req.index)
""")


def _stub_prefix(tmp_path, extra=""):
    stub = tmp_path / "stub_worker.py"
    stub.write_text(_STUB + extra, encoding="utf-8")
    return [sys.executable, str(stub)]


def _requests(n):
    return [RequestSpec(index=i, prompt=f"p{i}", output=f"o{i}.png") for i in range(n)]


def test_run_router_all_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = _args(work_dir=str(tmp_path / "work"))
    rc = run_router(args, _requests(5), replica_cores=4,
                    worker_argv_prefix=_stub_prefix(tmp_path))
    assert rc == 0
    s = summarize(tmp_path / "work" / "requests")
    assert s.done == [0, 1, 2, 3, 4] and not s.failed and not s.unfinished


def test_run_router_reports_failed_request(tmp_path):
    args = _args(work_dir=str(tmp_path / "work"), dp_schedule="least_loaded")
    prefix = _stub_prefix(tmp_path)
    rc = run_router(args, _requests(4), replica_cores=4,
                    worker_argv_prefix=prefix + ["--fail-index", "2"])
    assert rc != 0
    s = summarize(tmp_path / "work" / "requests")
    assert 2 in s.failed and sorted(s.done) == [0, 1, 3]


def test_run_router_crashed_worker_round_robin(tmp_path):
    args = _args(work_dir=str(tmp_path / "work"), dp=2)
    prefix = _stub_prefix(tmp_path)
    rc = run_router(args, _requests(4), replica_cores=4,
                    worker_argv_prefix=prefix + ["--crash-worker", "1"])
    assert rc != 0
    s = summarize(tmp_path / "work" / "requests")
    assert sorted(s.done) == [0, 2]            # worker 0's assignments
    assert set(s.failed) == {1, 3}             # dead worker's assignments marked failed
```

Note: `worker_cli_args` receives extra stub args appended to the prefix — `run_router` must append its own flags AFTER the prefix, and the stub parses with `parse_known_args`, so `worker_cli_args(args)` output is ignored by the stub. The HBM check must be skippable when `model_path` is unavailable: guard it behind a `try: resolve_model_path(..., local_files_only=True) except OSError: skip-with-warning` so CPU tests (no weights) pass.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/cli/dp/test_router.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# difflet/cli/dp/router.py
"""DP router: scatter (manifest) → spawn pinned workers → gather (markers).

Spec: docs/superpowers/specs/2026-07-06-dp-replication-routing-design.md §Router.
Workers are full difflet CLI invocations with dp=1 semantics; NEURON_RT_* are
plain-assigned (run_stage's setdefault must see the worker's range, not the
parent's).
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path
from typing import Mapping

from difflet.cli.dp.claims import mark_failed, summarize
from difflet.cli.dp.requests_io import RequestSpec, write_manifest


def replica_core_ranges(dp: int, replica_cores: int) -> list[str]:
    ranges = []
    for w in range(dp):
        lo = w * replica_cores
        hi = lo + replica_cores - 1
        ranges.append(str(lo) if replica_cores == 1 else f"{lo}-{hi}")
    return ranges


def worker_env(base_env: Mapping[str, str], core_range: str, replica_cores: int) -> dict[str, str]:
    env = dict(base_env)
    env["NEURON_RT_VISIBLE_CORES"] = core_range
    env["NEURON_RT_NUM_CORES"] = str(replica_cores)
    return env


def worker_cli_args(args) -> list[str]:
    """Flags forwarded to worker CLI processes. NEVER --dp/--mode/--prompt/--output/--requests."""
    argv = ["--model-id", args.model_id]
    for flag, value in (
        ("--tp-degree", args.tp_degree),
        ("--cp-degree", args.cp_degree),
        ("--cp-mode", args.cp_mode),
        ("--height", args.height),
        ("--width", args.width),
        ("--num-frames", args.num_frames),
        ("--steps", args.steps),
        ("--guidance-scale", args.guidance_scale),
        ("--seed", args.seed),
        ("--cache-dir", args.cache_dir),
        ("--revision", getattr(args, "revision", None)),
    ):
        if value is not None:
            argv += [flag, str(value)]
    if getattr(args, "cfg_parallel", False):
        argv.append("--cfg-parallel")
    if getattr(args, "sp_enabled", False):
        argv.append("--sp")
    if getattr(args, "keep_work_dir", False):
        argv.append("--keep-work-dir")
    return argv


def _check_hbm(args) -> None:
    from difflet.cli.dp.hbm_check import assert_replica_fits
    from difflet.pipeline.path_resolver import resolve_model_path

    try:
        model_path = resolve_model_path(args.model_id, local_files_only=True)
    except OSError:
        print("[dp-router] weights not local; skipping HBM fit check", flush=True)
        return
    assert_replica_fits(model_path)


def run_router(
    args,
    requests: list[RequestSpec],
    *,
    replica_cores: int,
    worker_argv_prefix: list[str] | None = None,
) -> int:
    dp = int(args.dp or 1)
    schedule = args.dp_schedule
    work_dir = Path(args.work_dir or Path.home() / ".cache" / "difflet" / "work" / "dp")
    requests_dir = work_dir / "requests"

    if schedule == "round_robin":
        requests = [
            dataclasses.replace(req, assigned_worker=req.index % dp) for req in requests
        ]
    write_manifest(requests, requests_dir)
    _check_hbm(args)

    prefix = worker_argv_prefix or [sys.executable, "-m", "difflet.cli.main"]
    import os

    procs = []
    for w, core_range in enumerate(replica_core_ranges(dp, replica_cores)):
        argv = prefix + ["generate"] + worker_cli_args(args) + [
            "--requests-dir", str(requests_dir),
            "--worker-index", str(w),
            "--dp-schedule", schedule,
            "--work-dir", str(work_dir / f"worker_{w}"),
        ]
        env = worker_env(os.environ, core_range, replica_cores)
        print(f"[dp-router] worker {w}: cores {core_range}", flush=True)
        procs.append(subprocess.Popen(argv, env=env))

    exit_codes = [p.wait() for p in procs]

    summary = summarize(requests_dir)
    for req in requests:
        if req.index in summary.done or req.index in summary.failed:
            continue
        crashed = (
            schedule == "round_robin"
            and req.assigned_worker is not None
            and exit_codes[req.assigned_worker] != 0
        )
        claim = requests_dir / f"req_{req.index:04d}.claim"
        if crashed or claim.exists():
            mark_failed(requests_dir, req.index, "worker crashed before finishing request")

    summary = summarize(requests_dir)
    print(
        f"[dp-router] done={len(summary.done)} failed={len(summary.failed)} "
        f"unfinished={len(summary.unfinished)}",
        flush=True,
    )
    for index, error in sorted(summary.failed.items()):
        print(f"[dp-router] request {index} FAILED: {error.splitlines()[0]}", flush=True)
    return 0 if (not summary.failed and not summary.unfinished) else 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/cli/dp/test_router.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit and push**

```bash
git add difflet/cli/dp/router.py tests/unit/cli/dp/test_router.py
git commit -m "feat(dp): router core — core pinning, worker spawn, gather/accounting"
git push
```

---

### Task 6: Stage request-loop helpers (`stage_loop.py`)

**Files:**
- Create: `difflet/cli/dp/stage_loop.py`
- Test: `tests/unit/cli/dp/test_stage_loop.py`

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces:
```python
def batch_mode(args) -> bool                                        # getattr(args, "requests_dir", None) is not None
def claim_requests(args) -> Iterator[RequestSpec]                   # FIRST stage of a worker chain (or the only phase)
def claimed_requests(args) -> Iterator[RequestSpec]                 # later stages: replay this worker's claims, skipping failed
@contextmanager
def request_scope(args, request: RequestSpec, *, final: bool)       # marks .failed + continues in batch mode; re-raises in legacy; marks .done when final on success
def work_file(args, request: RequestSpec | None, name: str) -> Path # legacy: work_dir/name; batch: work_dir/{stem}_req{idx:04d}{suffix}
def effective(request: RequestSpec | None, args, field: str, default)  # per-request override -> args -> default
```
Legacy mode (`not batch_mode(args)`): `claim_requests` yields ONE `RequestSpec` built from `args.prompt/args.output/args.seed` with `steps`/`guidance_scale` taken from args (may be None → stage defaults apply via `effective`); `request_scope` lets exceptions propagate unchanged (today's behavior); `work_file` returns the un-suffixed path so on-disk names are byte-identical to today.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/cli/dp/test_stage_loop.py
import argparse
from pathlib import Path

import pytest

from difflet.cli.dp import stage_loop
from difflet.cli.dp.claims import claim_next, summarize
from difflet.cli.dp.requests_io import RequestSpec, write_manifest


def _batch_args(tmp_path, worker=0, schedule="least_loaded"):
    return argparse.Namespace(
        requests_dir=str(tmp_path / "requests"), worker_index=worker,
        dp_schedule=schedule, work_dir=str(tmp_path / "work"),
        prompt=None, output=None, seed=42, steps=None, guidance_scale=None,
    )


def _legacy_args(tmp_path):
    return argparse.Namespace(
        requests_dir=None, worker_index=None, dp_schedule=None,
        work_dir=str(tmp_path / "work"), prompt="a cat", output="cat.png",
        seed=7, steps=3, guidance_scale=None,
    )


def test_legacy_single_request_and_paths(tmp_path):
    args = _legacy_args(tmp_path)
    assert not stage_loop.batch_mode(args)
    reqs = list(stage_loop.claim_requests(args))
    assert len(reqs) == 1
    assert reqs[0].prompt == "a cat" and reqs[0].seed == 7 and reqs[0].steps == 3
    assert stage_loop.work_file(args, reqs[0], "latents.pt") == Path(args.work_dir) / "latents.pt"


def test_legacy_scope_propagates_exceptions(tmp_path):
    args = _legacy_args(tmp_path)
    req = next(iter(stage_loop.claim_requests(args)))
    with pytest.raises(RuntimeError, match="boom"):
        with stage_loop.request_scope(args, req, final=True):
            raise RuntimeError("boom")


def test_batch_claims_and_marks(tmp_path):
    write_manifest([RequestSpec(index=i, prompt=f"p{i}", output=f"o{i}.png")
                    for i in range(3)], tmp_path / "requests")
    args = _batch_args(tmp_path)
    seen = []
    for req in stage_loop.claim_requests(args):
        with stage_loop.request_scope(args, req, final=True):
            if req.index == 1:
                raise RuntimeError("bad request")
            seen.append(req.index)
    assert seen == [0, 2]
    s = summarize(tmp_path / "requests")
    assert s.done == [0, 2] and 1 in s.failed and "bad request" in s.failed[1]


def test_batch_nonfinal_scope_does_not_mark_done(tmp_path):
    write_manifest([RequestSpec(index=0, prompt="p", output="o.png")], tmp_path / "requests")
    args = _batch_args(tmp_path)
    for req in stage_loop.claim_requests(args):
        with stage_loop.request_scope(args, req, final=False):
            pass
    s = summarize(tmp_path / "requests")
    assert s.done == [] and s.unfinished == [0]


def test_claimed_requests_replays_claims_skipping_failed(tmp_path):
    write_manifest([RequestSpec(index=i, prompt=f"p{i}", output=f"o{i}.png")
                    for i in range(4)], tmp_path / "requests")
    args0, args1 = _batch_args(tmp_path, 0), _batch_args(tmp_path, 1)
    claim_next(args0.requests_dir, 0, "least_loaded")   # req 0 -> worker 0
    claim_next(args1.requests_dir, 1, "least_loaded")   # req 1 -> worker 1
    claim_next(args0.requests_dir, 0, "least_loaded")   # req 2 -> worker 0
    from difflet.cli.dp.claims import mark_failed
    mark_failed(args0.requests_dir, 2, "earlier stage failed")
    assert [r.index for r in stage_loop.claimed_requests(args0)] == [0]
    assert [r.index for r in stage_loop.claimed_requests(args1)] == [1]


def test_batch_work_file_is_request_scoped(tmp_path):
    args = _batch_args(tmp_path)
    req = RequestSpec(index=3, prompt="p", output="o.png")
    assert stage_loop.work_file(args, req, "latents.pt") == (
        Path(args.work_dir) / "latents_req0003.pt"
    )


def test_effective_override_chain(tmp_path):
    args = argparse.Namespace(steps=10)
    assert stage_loop.effective(RequestSpec(0, "p", "o", steps=5), args, "steps", 2) == 5
    assert stage_loop.effective(RequestSpec(0, "p", "o"), args, "steps", 2) == 10
    assert stage_loop.effective(None, argparse.Namespace(steps=None), "steps", 2) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/cli/dp/test_stage_loop.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# difflet/cli/dp/stage_loop.py
"""Request iteration for stages: legacy single-request or batch claim loop.

A worker's FIRST stage iterates claim_requests(args) (claims until the queue is
empty); every LATER stage iterates claimed_requests(args) (replays this
worker's claims, skipping ones an earlier stage failed). request_scope marks
.failed-and-continue in batch mode, re-raises in legacy mode, and marks .done
on success when final=True (the stage that writes the user-visible output).
"""

from __future__ import annotations

import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from difflet.cli.dp import claims
from difflet.cli.dp.requests_io import RequestSpec


def batch_mode(args) -> bool:
    return getattr(args, "requests_dir", None) is not None


def _legacy_request(args) -> RequestSpec:
    return RequestSpec(
        index=0,
        prompt=args.prompt,
        output=args.output,
        seed=int(getattr(args, "seed", 42)),
        guidance_scale=getattr(args, "guidance_scale", None),
        steps=getattr(args, "steps", None),
    )


def claim_requests(args) -> Iterator[RequestSpec]:
    if not batch_mode(args):
        yield _legacy_request(args)
        return
    while (
        req := claims.claim_next(args.requests_dir, int(args.worker_index), args.dp_schedule)
    ) is not None:
        yield req


def claimed_requests(args) -> Iterator[RequestSpec]:
    if not batch_mode(args):
        yield _legacy_request(args)
        return
    for req in claims.claimed_by(args.requests_dir, int(args.worker_index)):
        if not claims.is_failed(args.requests_dir, req.index):
            yield req


@contextmanager
def request_scope(args, request: RequestSpec, *, final: bool):
    if not batch_mode(args):
        yield
        return
    try:
        yield
    except Exception:
        claims.mark_failed(args.requests_dir, request.index, traceback.format_exc())
        print(f"[dp-worker] request {request.index} failed; continuing", flush=True)
    else:
        if final:
            claims.mark_done(args.requests_dir, request.index)


def work_file(args, request: RequestSpec | None, name: str) -> Path:
    base = Path(args.work_dir) / name
    if not batch_mode(args) or request is None:
        return base
    return base.with_name(f"{base.stem}_req{request.index:04d}{base.suffix}")


def effective(request: RequestSpec | None, args, field: str, default):
    if request is not None and getattr(request, field, None) is not None:
        return getattr(request, field)
    value = getattr(args, field, None)
    return value if value is not None else default
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/cli/dp/test_stage_loop.py -v`
Expected: 7 PASS

- [ ] **Step 5: Commit and push**

```bash
git add difflet/cli/dp/stage_loop.py tests/unit/cli/dp/test_stage_loop.py
git commit -m "feat(dp): stage request-loop helpers (claim loop, scopes, work files)"
git push
```

---

### Task 7: CLI wiring (`main.py` + `stage.py`)

**Files:**
- Modify: `difflet/cli/main.py` (`_add_parallel_flags`, `_add_generate_flags`, `main()`, new `_validate_dp`), `difflet/cli/stage.py:_build_stage_parser`
- Test: `tests/unit/cli/test_main_dp.py`

**Interfaces:**
- Consumes: `resolve_mode` (Task 3), `load_requests_jsonl`/`RequestSpec` (Task 1), `run_router` (Task 5).
- Produces: CLI contract —
  - `_add_parallel_flags` adds `--dp` (type=int, default=None), `--mode` (choices latency/throughput/mixed, default None), `--dp-schedule` (choices round_robin/least_loaded, default "round_robin"); **`--cp-degree` default changes `1` → `None`** (every consumer already reads `args.cp_degree or 1`, verified in wan/flux/ltx_2/hunyuan_video/qwen_image/stage parser — same effective value; needed so `resolve_mode` can distinguish explicit).
  - `_add_generate_flags`: `--prompt`/`--output` become `required=False`; add `--requests` (default None), and internal flags `--requests-dir`, `--worker-index` (type=int), both `default=None, help=argparse.SUPPRESS`.
  - `stage.py:_build_stage_parser` gains: `--requests-dir` (default None), `--worker-index` (type=int, default None), `--dp-schedule` (default "round_robin"), `--keep-work-dir` (store_true) — the silent-drop rule.
  - `main()` flow (after existing validations): resolve mode → overwrite `args.cfg_parallel/args.cp_degree/args.dp` from `ModeConfig` → `_validate_dp(args)` → if generate/run and `(args.requests or (args.dp or 1) > 1)` and `args.requests_dir is None`: run the router path; else fall through to orchestrator (unchanged).

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/cli/test_main_dp.py
import json

import pytest

import difflet.cli.main as cli_main


def _argv(extra, model="Wan-AI/Wan2.2-T2V-A14B-Diffusers"):
    return ["generate", "--model-id", model, *extra]


def test_parser_accepts_dp_flags_on_generate_and_compile():
    p = cli_main._build_parser()
    args = p.parse_args(_argv(["--dp", "4", "--mode", "throughput",
                               "--dp-schedule", "least_loaded",
                               "--requests", "r.jsonl"]))
    assert args.dp == 4 and args.mode == "throughput"
    args = p.parse_args(["compile", "--model-id", "black-forest-labs/FLUX.1-dev",
                         "--dp", "4", "--mode", "throughput"])
    assert args.dp == 4


def test_single_prompt_still_parses_without_batch_flags():
    p = cli_main._build_parser()
    args = p.parse_args(_argv(["--prompt", "cat", "--output", "c.mp4"]))
    assert args.dp is None and args.requests is None and args.requests_dir is None


def test_generate_requires_prompt_or_requests():
    p = cli_main._build_parser()
    args = p.parse_args(_argv([]))
    with pytest.raises(SystemExit):
        cli_main._validate_dp(args)


def test_teacache_rejected_in_batch_mode(tmp_path):
    p = cli_main._build_parser()
    req = tmp_path / "r.jsonl"
    req.write_text(json.dumps({"prompt": "x", "output": "x.png"}) + "\n")
    args = p.parse_args(_argv(["--requests", str(req), "--teacache-cadence", "2"]))
    with pytest.raises(SystemExit):
        cli_main._validate_dp(args)


def test_mode_resolution_applied_to_args(monkeypatch, tmp_path):
    calls = {}
    def fake_router(args, requests, *, replica_cores, worker_argv_prefix=None):
        calls["dp"] = args.dp
        calls["cfg"] = args.cfg_parallel
        calls["cp"] = args.cp_degree
        calls["replica_cores"] = replica_cores
        calls["n"] = len(requests)
        return 0
    monkeypatch.setattr("difflet.cli.dp.router.run_router", fake_router)
    req = tmp_path / "r.jsonl"
    req.write_text(json.dumps({"prompt": "x", "output": str(tmp_path / "x.mp4")}) + "\n")
    with pytest.raises(SystemExit) as exc:
        cli_main.main(_argv(["--mode", "mixed", "--requests", str(req),
                             "--tp-degree", "4"]))
    assert exc.value.code == 0
    assert calls == {"dp": 2, "cfg": True, "cp": 1, "replica_cores": 8, "n": 1}
    # Wan mixed: dp=2 cfg=2 cp=1 tp=4 -> replica_cores = 2*1*4 = 8


def test_stage_parser_accepts_new_flags():
    from difflet.cli.stage import _build_stage_parser
    args, _ = _build_stage_parser().parse_known_args([
        "--orchestrator", "Wan-AI/Wan2.2-T2V-A14B-Diffusers", "--stage", "transformer",
        "--requests-dir", "/tmp/reqs", "--worker-index", "1",
        "--dp-schedule", "least_loaded", "--keep-work-dir",
    ])
    assert args.requests_dir == "/tmp/reqs" and args.worker_index == 1
    assert args.dp_schedule == "least_loaded" and args.keep_work_dir
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/cli/test_main_dp.py -v`
Expected: FAIL — unrecognized arguments `--dp`, missing `_validate_dp`, stage parser lacks flags

- [ ] **Step 3: Implement**

In `difflet/cli/main.py`, append to `_add_parallel_flags` (and change `--cp-degree` default):

```python
    p.add_argument("--cp-degree", type=int, default=None,   # was default=1; all consumers use `or 1`
                   help="Context-parallel degree (default: 1)")
    p.add_argument("--dp", type=int, default=None,
                   help="Data-parallel replica count. The router spawns N workers, "
                        "each a full dp=1 model copy on its own core range; requests "
                        "are distributed across them (default: 1)")
    p.add_argument("--mode", choices=["latency", "throughput", "mixed"], default=None,
                   help="Runtime mode preset selecting dp/cfg/cp per model class "
                        "(explicit parallelism flags override individual fields)")
    p.add_argument("--dp-schedule", choices=["round_robin", "least_loaded"],
                   default="round_robin",
                   help="Request-to-replica schedule for --dp>1 (default: round_robin)")
    p.add_argument("--total-cores", type=int, default=None,
                   help="Total NeuronCores available for dp*cfg*cp*tp validation "
                        "(default: NEURON_RT_NUM_CORES when set, else unchecked)")
```

In `_add_generate_flags`: change `--prompt`/`--output` to `required=False`, add:

```python
    p.add_argument("--requests", default=None,
                   help="JSONL batch file: one request per line with prompt/output/"
                        "seed and optional negative_prompt/guidance_scale/steps")
    p.add_argument("--requests-dir", default=None, help=argparse.SUPPRESS)   # internal: worker mode
    p.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
```

New validation + dispatch in `main.py` (after `_validate_teacache`):

```python
def _replica_cores(args) -> int:
    from difflet.registry import resolve_model
    entry = resolve_model(args.model_id, model_type=_MODEL_TYPE[args.model_id])
    tp = args.tp_degree or entry.default_parallel.tp_degree
    cfg = 2 if getattr(args, "cfg_parallel", False) else 1
    return tp * (args.cp_degree or 1) * cfg


def _validate_dp(args) -> None:
    batch = getattr(args, "requests", None) is not None or (getattr(args, "dp", None) or 1) > 1
    worker = getattr(args, "requests_dir", None) is not None
    if args.command in ("generate", "run") and not worker:
        if not batch and not (args.prompt and args.output):
            print("Error: --prompt and --output are required (or use --requests FILE).",
                  file=sys.stderr)
            raise SystemExit(1)
        if batch and args.prompt and args.requests:
            print("Error: --prompt and --requests are mutually exclusive.", file=sys.stderr)
            raise SystemExit(1)
    if batch and any(
        getattr(args, name, None) is not None
        for name in ("teacache_cadence", "teacache_online_delta", "teacache_speedup")
    ):
        print("Error: TeaCache flags are not supported in batch/DP mode "
              "(per-request controller reset is a follow-up).", file=sys.stderr)
        raise SystemExit(1)
    if (getattr(args, "dp", None) or 1) > 1 and not worker:
        total = getattr(args, "total_cores", None)
        if total is None and os.environ.get("NEURON_RT_NUM_CORES"):
            total = int(os.environ["NEURON_RT_NUM_CORES"])
        needed = args.dp * _replica_cores(args)
        if total is not None and needed > total:
            print(f"Error: dp*cfg*cp*tp = {needed} cores exceeds available cores ({total}).",
                  file=sys.stderr)
            raise SystemExit(1)


def _dispatch_dp(args) -> None:
    """Route batch/DP generate runs through the router; exits the process."""
    from difflet.cli.dp.requests_io import RequestSpec, load_requests_jsonl
    from difflet.cli.dp import router

    if args.requests:
        requests = load_requests_jsonl(args.requests)
    else:
        requests = [RequestSpec(index=0, prompt=args.prompt, output=args.output,
                                seed=args.seed, guidance_scale=args.guidance_scale,
                                steps=args.steps)]
    dp = args.dp or 1
    if dp > 1 and len(requests) == 1:
        print("Warning: --dp > 1 with a single request leaves replicas idle.",
              file=sys.stderr)
    if args.command == "run":
        orch = _get_orchestrator(args)
        orch.download()
        orch.compile()
    raise SystemExit(router.run_router(args, requests, replica_cores=_replica_cores(args)))
```

And in `main()` (`args.command in ("compile", "generate", "run")` block), after the existing validators:

```python
        from difflet.cli.modes import resolve_mode
        mode_cfg = resolve_mode(args.model_id, getattr(args, "mode", None), args)
        if mode_cfg is not None:
            args.dp = mode_cfg.dp
            args.cfg_parallel = mode_cfg.cfg_parallel
            args.cp_degree = mode_cfg.cp_degree
            _validate_cfg_parallel(args)   # re-run with resolved flags
            _validate_sp(args)
    if args.command in ("generate", "run"):
        _validate_teacache(args)
        _validate_dp(args)
        if args.requests_dir is None and (
            args.requests is not None or (args.dp or 1) > 1
        ):
            _dispatch_dp(args)   # does not return
```

(`import os` at module top; the pre-existing `_validate_teacache` call site is replaced by this block — keep call order: cfg → sp → mode resolve → cfg/sp re-run → teacache → dp.)

In `difflet/cli/stage.py:_build_stage_parser`, add before `return p`:

```python
    p.add_argument("--requests-dir", default=None)
    p.add_argument("--worker-index", type=int, default=None)
    p.add_argument("--dp-schedule", default="round_robin")
    p.add_argument("--keep-work-dir", action="store_true")
```

- [ ] **Step 4: Run tests — new and full unit suite (regressions: required=False on --prompt)**

Run: `python -m pytest tests/unit/cli/ tests/unit/ -x -q`
Expected: all PASS (existing CLI tests must not depend on `--prompt` being required; if one does, update it to call `_validate_dp` semantics)

- [ ] **Step 5: Commit and push**

```bash
git add difflet/cli/main.py difflet/cli/stage.py tests/unit/cli/test_main_dp.py
git commit -m "feat(cli): --dp/--mode/--dp-schedule/--requests flags + router dispatch"
git push
```

---

### Task 8: Wan orchestrator batch wiring

**Files:**
- Modify: `difflet/cli/orchestrators/wan.py` (`generate`, `_stage_transformer`, `_stage_vae`, `_shared_cli_args`)
- Test: `tests/unit/cli/orchestrators/test_batch_wiring.py` (created here, extended by Tasks 9–12)

**Interfaces:**
- Consumes: `stage_loop.claim_requests/claimed_requests/request_scope/work_file/effective/batch_mode` (Task 6).
- Produces: the batch-wiring pattern all staged orchestrators follow — first device stage claims, last stage replays claims and is `final=True`; `_shared_cli_args` forwards `--requests-dir/--worker-index/--dp-schedule`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/cli/orchestrators/test_batch_wiring.py
"""CPU-testable batch wiring: _shared_cli_args must forward the DP worker flags
(stage parser silently drops what isn't forwarded — the --sp bug class)."""
import argparse


def _args(**kw):
    defaults = dict(
        model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers", tp_degree=None, cp_degree=None,
        cp_mode="gather_kv", cfg_parallel=False, sp_enabled=False, height=None,
        width=None, num_frames=None, steps=None, guidance_scale=None, seed=42,
        cache_dir=None, prompt=None, output=None, requests_dir="/tmp/reqs",
        worker_index=1, dp_schedule="least_loaded", keep_work_dir=False,
        teacache_cadence=None, teacache_online_delta=None, teacache_speedup=None,
        teacache_calibration=None, work_dir=None, revision=None, force=False,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def test_wan_shared_args_forward_dp_worker_flags():
    from difflet.cli.orchestrators.wan import WanOrchestrator
    parts = WanOrchestrator(_args())._shared_cli_args(stage_mode="generate")
    text = " ".join(parts)
    assert "--requests-dir /tmp/reqs" in text
    assert "--worker-index 1" in text
    assert "--dp-schedule least_loaded" in text


def test_wan_shared_args_legacy_unchanged():
    from difflet.cli.orchestrators.wan import WanOrchestrator
    parts = WanOrchestrator(_args(requests_dir=None, worker_index=None,
                                  prompt="cat", output="c.mp4"))._shared_cli_args(
        stage_mode="generate")
    text = " ".join(parts)
    assert "--requests-dir" not in text and "--worker-index" not in text
    assert "--prompt cat" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/cli/orchestrators/test_batch_wiring.py -v`
Expected: FAIL — `--requests-dir` not in shared args

- [ ] **Step 3: Implement**

`_shared_cli_args` (wan.py) — append before `return parts`:

```python
        if getattr(a, "requests_dir", None):
            parts += ["--requests-dir", str(a.requests_dir),
                      "--worker-index", str(a.worker_index),
                      "--dp-schedule", str(getattr(a, "dp_schedule", "round_robin"))]
```

`_stage_transformer` — replace the post-`app.load(...)` single-request block (from `latent_frames = ...` through `print(f"[wan] latents saved ...")`; the `app.load` call itself moves ABOVE the loop so weights load once) with:

```python
        app.load(str(compiled_dir), start_rank_id=0,
                 local_ranks_size=parallel.world_size, skip_warmup=True)
        work_dir = Path(args.work_dir)
        latent_frames = _latent_num_frames(args.num_frames or 9)
        for req in stage_loop.claim_requests(args):
            with stage_loop.request_scope(args, req, final=False):
                torch.manual_seed(req.seed)
                latents = torch.randn(
                    1, 16, latent_frames,
                    (args.height or 480) // 8, (args.width or 832) // 8,
                    dtype=torch.bfloat16,
                ) * 0.1
                out = app(
                    latents=latents,
                    prompt=req.prompt,
                    height=args.height or 480,
                    width=args.width or 832,
                    num_frames=args.num_frames or 9,
                    num_inference_steps=int(stage_loop.effective(req, args, "steps", 2)),
                    guidance_scale=float(
                        stage_loop.effective(req, args, "guidance_scale", 1.0)
                    ),
                    output_type="latent",
                )
                latents_out = out.latents if hasattr(out, "latents") else out[0]
                dest = stage_loop.work_file(args, req, "latents.pt")
                torch.save(latents_out.cpu(), dest)
                print(f"[wan] latents saved to {dest}")
```

with `from difflet.cli.dp import stage_loop` added to the stage-local imports. Legacy behavior is preserved exactly: one iteration, `req.seed == args.seed`, `req.prompt == args.prompt`, `work_file` returns `work_dir/latents.pt`, exceptions propagate.

`_stage_vae` — same transformation of its post-load block (load once, loop `stage_loop.claimed_requests(args)`, `final=True`):

```python
        app.load(str(compiled_dir), start_rank_id=0, local_ranks_size=1, skip_warmup=True)
        for req in stage_loop.claimed_requests(args):
            with stage_loop.request_scope(args, req, final=True):
                latents = torch.load(stage_loop.work_file(args, req, "latents.pt"))
                out = app(
                    latents=latents,
                    height=args.height or 480,
                    width=args.width or 832,
                    num_frames=args.num_frames or 9,
                    num_inference_steps=1,
                    output_type="pt",
                )
                frames = out.frames if hasattr(out, "frames") else out[0]
                out_path = Path(req.output)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if out_path.suffix == ".mp4" and _save_video(frames.cpu(), str(out_path)):
                    pass
                else:
                    pt_path = out_path.with_suffix(".pt")
                    torch.save(frames.cpu(), pt_path)
                    print(f"[wan] video tensor saved to {pt_path}")
```

(Note: legacy mode `req.output == args.output` — unchanged behavior. `args.output` may be None in batch mode; `req.output` is always set.)

`generate()` — the work_dir default must be worker-unique in batch mode; the router already passes `--work-dir work/worker_{w}`, so no change needed beyond NOT deleting the shared requests dir (it lives outside worker work_dir). No change to `generate()`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/unit/cli/orchestrators/test_batch_wiring.py tests/unit/ -q`
Expected: all PASS

- [ ] **Step 5: Commit and push**

```bash
git add difflet/cli/orchestrators/wan.py tests/unit/cli/orchestrators/test_batch_wiring.py
git commit -m "feat(dp): Wan orchestrator batch claim-loop wiring"
git push
```

---

### Task 9: HunyuanVideo orchestrator batch wiring

**Files:**
- Modify: `difflet/cli/orchestrators/hunyuan_video.py` (`_shared_cli_args`, `_stage_clip`, `_stage_llama`, `_stage_generate`)
- Test: extend `tests/unit/cli/orchestrators/test_batch_wiring.py`

**Interfaces:** Consumes Task 6 helpers. HYV chain: clip (FIRST stage → `claim_requests`) → llama (`claimed_requests`, final=False) → generate (`claimed_requests`, final=True; DiT+VAE fused). Per-request tensors: `clip_req0000.pt`, `llama_req0000.pt` via `work_file`.

- [ ] **Step 1: Write the failing test** — add to `test_batch_wiring.py`:

```python
def test_hunyuan_shared_args_forward_dp_worker_flags():
    from difflet.cli.orchestrators.hunyuan_video import HunyuanVideoOrchestrator
    parts = HunyuanVideoOrchestrator(_args(
        model_id="hunyuanvideo-community/HunyuanVideo"))._shared_cli_args(
        stage_mode="generate")
    text = " ".join(parts)
    assert "--requests-dir /tmp/reqs" in text and "--worker-index 1" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/cli/orchestrators/test_batch_wiring.py -v -k hunyuan`
Expected: FAIL

- [ ] **Step 3: Implement** — same `_shared_cli_args` append as Task 8. Stage loops (each stage keeps `app.load(...)` once before its loop, `from difflet.cli.dp import stage_loop` in stage-local imports):

`_stage_clip` post-load block becomes:

```python
        app.load(str(compiled_dir))
        tok = CLIPTokenizer.from_pretrained(str(Path(model_dir) / "tokenizer_2"))
        work_dir = Path(args.work_dir)
        for req in stage_loop.claim_requests(args):
            with stage_loop.request_scope(args, req, final=False):
                ids = tok(req.prompt, padding="max_length", max_length=77,
                          truncation=True, return_tensors="pt").input_ids.to(torch.int64)
                out = app(ids)
                pooled = out.pooler_output.to(torch.bfloat16).cpu().reshape(1, -1)
                dest = stage_loop.work_file(args, req, "clip.pt")
                torch.save({"pooled_projections": pooled}, dest)
                print(f"[clip] pooled_projections {tuple(pooled.shape)} -> {dest}")
```

`_stage_llama` post-load block: loop `stage_loop.claimed_requests(args)`, `final=False`, tokenize `_LLAMA_TEMPLATE.format(req.prompt)`, save to `stage_loop.work_file(args, req, "llama.pt")` (same tensor keys as today).

`_stage_generate` post-compile block: loop `stage_loop.claimed_requests(args)`, `final=True`; per request — `torch.manual_seed(req.seed)`, load `work_file(args, req, "llama.pt")` / `work_file(args, req, "clip.pt")`, `num_inference_steps=int(stage_loop.effective(req, args, "steps", 4))`, guidance from `stage_loop.effective(req, args, "guidance_scale", 6.0)`, output path `Path(req.output)` (same `_save_video`/`.pt` fallback body as today). `app.load` and the scheduler `sigmas`/`timesteps` computation stay OUTSIDE the loop (steps may vary per request → move `sigmas`/`timesteps` INSIDE the loop, computed from the per-request step count).

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/unit/cli/orchestrators/ tests/unit/ -q`
Expected: all PASS

- [ ] **Step 5: Commit and push**

```bash
git add difflet/cli/orchestrators/hunyuan_video.py tests/unit/cli/orchestrators/test_batch_wiring.py
git commit -m "feat(dp): HunyuanVideo orchestrator batch claim-loop wiring"
git push
```

---

### Task 10: Qwen-Image orchestrator batch wiring

**Files:**
- Modify: `difflet/cli/orchestrators/qwen_image.py` (`_shared_cli_args`, `_stage_text`, `_stage_generate`, `_stage_vae`)
- Test: extend `tests/unit/cli/orchestrators/test_batch_wiring.py`

**Interfaces:** Consumes Task 6. Chain: text (FIRST → `claim_requests`, final=False, saves `text_req0000.pt`) → generate (`claimed_requests`, final=False, seeds `torch.manual_seed(req.seed)` before `app.pipeline(...)`, saves `latents_req0000.pt`) → vae (`claimed_requests`, final=True, writes `Path(req.output)`).

- [ ] **Step 1: Write the failing test** — add to `test_batch_wiring.py`:

```python
def test_qwen_shared_args_forward_dp_worker_flags():
    from difflet.cli.orchestrators.qwen_image import QwenImageOrchestrator
    parts = QwenImageOrchestrator(_args(model_id="Qwen/Qwen-Image"))._shared_cli_args(
        stage_mode="generate")
    text = " ".join(parts)
    assert "--requests-dir /tmp/reqs" in text and "--worker-index 1" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/cli/orchestrators/test_batch_wiring.py -v -k qwen`
Expected: FAIL

- [ ] **Step 3: Implement** — `_shared_cli_args` append (as Task 8). `_stage_text` and `_stage_vae` follow the exact Task 8/9 loop shape (`claim_requests`/final=False for text, `claimed_requests`/final=True for vae, tensors via `stage_loop.work_file(args, req, "text.pt")` / `"latents.pt"`, final output `Path(req.output)`). `_stage_generate` is the seeding-critical one — its post-`app.load` block becomes:

```python
        app.load(str(compiled_dir), skip_warmup=True)
        sched = app.pipeline.scheduler
        sc = sched.config
        image_seq_len = (h // 16) * (w // 16)
        slope = (sc.max_shift - sc.base_shift) / (sc.max_image_seq_len - sc.base_image_seq_len)
        mu = image_seq_len * slope + (sc.base_shift - slope * sc.base_image_seq_len)
        for req in stage_loop.claimed_requests(args):
            with stage_loop.request_scope(args, req, final=False):
                text = torch.load(stage_loop.work_file(args, req, "text.pt"))
                num_steps = int(stage_loop.effective(req, args, "steps", 4))
                sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps).tolist()
                sched.set_timesteps(sigmas=sigmas, mu=mu, device="cpu")
                guidance = torch.full(
                    [1], float(stage_loop.effective(req, args, "guidance_scale", 4.0)),
                    dtype=torch.bfloat16,
                )
                torch.manual_seed(req.seed)
                out = app.pipeline(
                    encoder_hidden_states=text["encoder_hidden_states"],
                    encoder_hidden_states_mask=text["encoder_hidden_states_mask"],
                    guidance=guidance,
                    timesteps=sched.timesteps,
                    num_inference_steps=num_steps,
                    output_type="latent",
                )
                packed = out.latents.cpu()
                dest = stage_loop.work_file(args, req, "latents.pt")
                torch.save(packed, dest)
                print(f"[generate] packed latents {tuple(packed.shape)} -> {dest}")
```

(`torch.manual_seed(req.seed)` sits immediately before `app.pipeline(...)`, matching today's placement relative to latent sampling.)

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/unit/cli/orchestrators/ tests/unit/ -q`
Expected: all PASS

- [ ] **Step 5: Commit and push**

```bash
git add difflet/cli/orchestrators/qwen_image.py tests/unit/cli/orchestrators/test_batch_wiring.py
git commit -m "feat(dp): Qwen-Image orchestrator batch claim-loop wiring"
git push
```

---

### Task 11: Flux orchestrator batch wiring (in-process)

**Files:**
- Modify: `difflet/cli/orchestrators/flux.py:generate` (lines 91–103: the `pipe(...)`/save block)
- Test: extend `tests/unit/cli/orchestrators/test_batch_wiring.py`

**Interfaces:** Consumes Task 6. Flux is NOT staged — `generate()` loads `DiffletPipeline` in-process. Batch mode = single phase: `claim_requests` + `request_scope(final=True)` around each `pipe(...)` call. Flux workers read `args.requests_dir` directly off the main-parser namespace (no stage subprocess, no `_shared_cli_args`).

- [ ] **Step 1: Write the failing test** — add to `test_batch_wiring.py`:

```python
def test_flux_generate_loops_requests(monkeypatch, tmp_path):
    """Flux batch mode: pipeline loads once, pipe() called once per claimed request."""
    from difflet.cli.dp.requests_io import RequestSpec, write_manifest
    from difflet.cli.orchestrators import flux as flux_mod

    write_manifest(
        [RequestSpec(index=i, prompt=f"p{i}", output=str(tmp_path / f"o{i}.png"), seed=i)
         for i in range(3)],
        tmp_path / "requests",
    )
    calls = []

    class FakeImage:
        def save(self, path):
            calls.append(("save", path))

    class FakePipe:
        def __call__(self, **kw):
            calls.append(("pipe", kw["prompt"]))
            return type("Out", (), {"images": [FakeImage()]})()

    loads = []
    monkeypatch.setattr(
        flux_mod, "_load_pipeline", lambda self: (loads.append(1), FakePipe())[1],
        raising=False,
    )
    args = _args(model_id="black-forest-labs/FLUX.1-dev",
                 requests_dir=str(tmp_path / "requests"), worker_index=0,
                 dp_schedule="least_loaded")
    flux_mod.FluxOrchestrator(args).generate()
    assert loads == [1]                                  # loaded once
    assert [c for c in calls if c[0] == "pipe"] == [("pipe", "p0"), ("pipe", "p1"),
                                                    ("pipe", "p2")]
    from difflet.cli.dp.claims import summarize
    assert summarize(tmp_path / "requests").done == [0, 1, 2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/cli/orchestrators/test_batch_wiring.py -v -k flux`
Expected: FAIL — `_load_pipeline` doesn't exist / no loop

- [ ] **Step 3: Implement** — refactor `flux.py:generate`: extract everything from the `resolve_model_path` try-block through `DiffletPipeline.from_pretrained(...)` into `def _load_pipeline(self):` returning `pipe` (body unchanged, `self.args` for flags — cache-manifest validation included). `generate` becomes:

```python
    def generate(self) -> None:
        import torch

        from difflet.cli.dp import stage_loop

        pipe = self._load_pipeline()
        args = self.args
        for req in stage_loop.claim_requests(args):
            with stage_loop.request_scope(args, req, final=True):
                output = pipe(
                    prompt=req.prompt,
                    num_inference_steps=int(stage_loop.effective(req, args, "steps", 28)),
                    height=args.height or 1024,
                    width=args.width or 1024,
                    guidance_scale=float(
                        stage_loop.effective(req, args, "guidance_scale", 3.5)
                    ),
                    generator=torch.Generator().manual_seed(req.seed),
                )
                image = output.images[0]
                out = Path(req.output)
                out.parent.mkdir(parents=True, exist_ok=True)
                image.save(str(out))
                print(f"[difflet] image saved to {out}")
```

Legacy: one iteration with `req` built from `args` — identical to today's behavior including default steps 28 / guidance 3.5. Batch workers need `args.work_dir` optional — flux has no work_dir usage in generate; nothing else changes. (The monkeypatched test requires `_load_pipeline` to be resolved via the module-level name on the instance — define it as a normal method; the test patches `flux_mod.FluxOrchestrator._load_pipeline` equivalently via `setattr` on the module class attribute: patch target in the test is `flux_mod.FluxOrchestrator._load_pipeline` — adjust the monkeypatch line accordingly:
`monkeypatch.setattr(flux_mod.FluxOrchestrator, "_load_pipeline", lambda self: (loads.append(1), FakePipe())[1])`.)

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/unit/cli/orchestrators/ tests/unit/ -q`
Expected: all PASS

- [ ] **Step 5: Commit and push**

```bash
git add difflet/cli/orchestrators/flux.py tests/unit/cli/orchestrators/test_batch_wiring.py
git commit -m "feat(dp): Flux orchestrator in-process batch loop"
git push
```

---

### Task 12: LTX-2 orchestrator batch wiring (in-process)

**Files:**
- Modify: `difflet/cli/orchestrators/ltx_2.py:generate` (lines 102–113)
- Test: extend `tests/unit/cli/orchestrators/test_batch_wiring.py`

**Interfaces:** Same pattern as Task 11: extract `_load_pipeline(self)` (the `resolve_model_path` → `DiffletPipeline.from_pretrained(...)` block including the `enable_host_pipeline`/`enable_decode_components` application_kwargs), loop `claim_requests` with `request_scope(final=True)`. Per request: `num_inference_steps=int(stage_loop.effective(req, args, "steps", 40))`, `guidance_scale=float(stage_loop.effective(req, args, "guidance_scale", 3.5))`, `generator=torch.Generator().manual_seed(req.seed)`, save `frames.cpu()` to `Path(req.output).with_suffix(".pt")`. LTX-2's 3-guidance-forward structure lives inside `pipe(...)` (batch entries within the replica) — untouched by this task.

- [ ] **Step 1: Write the failing test** — add to `test_batch_wiring.py` (mirror of the flux test with `frames` output):

```python
def test_ltx2_generate_loops_requests(monkeypatch, tmp_path):
    import torch
    from difflet.cli.dp.requests_io import RequestSpec, write_manifest
    from difflet.cli.orchestrators import ltx_2 as ltx_mod

    write_manifest(
        [RequestSpec(index=i, prompt=f"p{i}", output=str(tmp_path / f"o{i}.mp4"), seed=i)
         for i in range(2)],
        tmp_path / "requests",
    )
    calls = []

    class FakePipe:
        def __call__(self, **kw):
            calls.append(kw["prompt"])
            return type("Out", (), {"frames": torch.zeros(1, 3, 2, 8, 8)})()

    monkeypatch.setattr(ltx_mod.LTX2Orchestrator, "_load_pipeline",
                        lambda self: FakePipe())
    args = _args(model_id="Lightricks/LTX-2",
                 requests_dir=str(tmp_path / "requests"), worker_index=0,
                 dp_schedule="least_loaded")
    ltx_mod.LTX2Orchestrator(args).generate()
    assert calls == ["p0", "p1"]
    assert (tmp_path / "o0.pt").exists() and (tmp_path / "o1.pt").exists()
    from difflet.cli.dp.claims import summarize
    assert summarize(tmp_path / "requests").done == [0, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/cli/orchestrators/test_batch_wiring.py -v -k ltx2`
Expected: FAIL

- [ ] **Step 3: Implement** — as described in Interfaces (exact transformation of lines 102–113, mirroring Task 11's code shape).

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/unit/cli/orchestrators/ tests/unit/ -q`
Expected: all PASS

- [ ] **Step 5: Commit and push**

```bash
git add difflet/cli/orchestrators/ltx_2.py tests/unit/cli/orchestrators/test_batch_wiring.py
git commit -m "feat(dp): LTX-2 orchestrator in-process batch loop"
git push
```

---

### Task 13: DP isolation assertions

**Files:**
- Create: `tests/unit/cli/test_dp_isolation.py`
- Modify (only if a check fails): none expected

**Interfaces:** Consumes `worker_cli_args` (Task 5), `DiffletParallelConfig` — pure assertions, no new product code.

- [ ] **Step 1: Write the tests (expected to pass immediately — they PIN the isolation invariants)**

```python
# tests/unit/cli/test_dp_isolation.py
"""DP isolation invariants (spec §Testing 2): the dp axis must never reach a
worker's compiled graph, and dp>1 must not perturb the compile-cache key."""
from pathlib import Path

import difflet
from difflet.pipeline.parallel_config import DiffletParallelConfig

REPO = Path(difflet.__file__).resolve().parent


def test_worker_cli_args_never_forward_dp_or_mode():
    import argparse
    from difflet.cli.dp.router import worker_cli_args
    args = argparse.Namespace(
        model_id="m", tp_degree=4, cp_degree=2, cp_mode="ring", cfg_parallel=True,
        sp_enabled=False, height=None, width=None, num_frames=None, steps=None,
        guidance_scale=None, seed=42, cache_dir=None, revision=None,
        keep_work_dir=False, dp=4, mode="throughput",
    )
    argv = worker_cli_args(args)
    assert "--dp" not in argv and "--mode" not in argv


def test_dp1_cache_key_identical_to_no_dp():
    with_dp = DiffletParallelConfig(tp_degree=4, dp_degree=1).to_cache_dict()
    assert "dp_degree" not in with_dp
    assert with_dp == DiffletParallelConfig(tp_degree=4).to_cache_dict()


def test_no_orchestrator_constructs_dp_parallel_config():
    """Workers must always build dp_degree=1 configs (the dataclass default)."""
    offenders = []
    for path in sorted((REPO / "cli").rglob("*.py")):
        if "dp_degree" in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(REPO)))
    assert offenders == [], f"CLI code must never set dp_degree: {offenders}"


def test_mesh_dp1_builds_no_dp_group():
    from difflet.pipeline.parallel_mesh import MeshSpec
    spec = MeshSpec(dp=1, cfg=2, cp=1, tp=4)
    # dp axis groups are all singletons at dp=1 — nothing to communicate over.
    assert all(len(g) == 1 for g in spec.axis_groups("dp"))
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/unit/cli/test_dp_isolation.py tests/unit/test_no_dp_parasites.py -v`
Expected: all PASS (if `test_no_orchestrator_constructs_dp_parallel_config` fails, a CLI file sets `dp_degree` — remove it; the router passes dp only via worker count)

- [ ] **Step 3: Commit and push**

```bash
git add tests/unit/cli/test_dp_isolation.py
git commit -m "test(dp): isolation invariants — no dp axis in worker graphs or cache keys"
git push
```

---

### Task 14: Coverage gate

**Files:**
- Possibly extend: any `tests/unit/cli/**` test file (gap filling)

- [ ] **Step 1: Measure**

Run:
```bash
python -m pytest tests/unit/cli/ \
  --cov=difflet.cli.dp --cov=difflet.cli.modes \
  --cov-report=term-missing -q
```
(If pytest-cov is missing: `pip install pytest-cov`.)
Expected: report per module.

- [ ] **Step 2: Fill gaps until every new module ≥ 90%**

Typical residue: `router.py:_check_hbm` weights-missing branch (test with monkeypatched `resolve_model_path` raising `OSError`), `requests_io` blank-line skip, `claims.claim_next` unknown-schedule `ValueError`, `stage_loop.work_file` with `request=None`. Write a test per uncovered branch; no product changes unless a branch is unreachable (then delete it).

- [ ] **Step 3: Run the FULL unit suite**

Run: `python -m pytest tests/unit/ -q`
Expected: all PASS

- [ ] **Step 4: Commit and push**

```bash
git add tests/unit/cli/
git commit -m "test(dp): close coverage gaps — new DP modules >= 90%"
git push
```

---

### Task 15: On-device DP correctness script (**requires trn2 hardware**)

**Files:**
- Create: `scripts/verify_dp_correctness.py`

**Interfaces:** Consumes the CLI only (subprocess). Compares latent/output `.pt` tensors between a dp=k batch run and a dp=1 batch run of the same requests file.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
# scripts/verify_dp_correctness.py
"""DP correctness: dp=k outputs must be bit-identical to dp=1 (same requests).

Usage (on trn2, weights + compile cache present — detach long runs):
  setsid python scripts/verify_dp_correctness.py \
      --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers --dp 2 --tp-degree 4 \
      --steps 2 > /tmp/dp_correctness.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

PROMPTS = ["a red fox in snow", "a sailboat at dusk", "a neon city street",
           "a bowl of ramen"]


def run_batch(model_id, dp, tp, steps, out_dir, extra):
    req_file = out_dir / "requests.jsonl"
    lines = [
        {"prompt": p, "output": str(out_dir / f"out_{i}.mp4"), "seed": 1000 + i}
        for i, p in enumerate(PROMPTS)
    ]
    req_file.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    cmd = [sys.executable, "-m", "difflet.cli.main", "generate",
           "--model-id", model_id, "--dp", str(dp), "--tp-degree", str(tp),
           "--steps", str(steps), "--requests", str(req_file),
           "--work-dir", str(out_dir / "work"), "--keep-work-dir", *extra]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return [Path(l["output"]).with_suffix(".pt") for l in lines]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--dp", type=int, default=2)
    ap.add_argument("--tp-degree", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("extra", nargs="*", default=[])
    args = ap.parse_args()

    base = Path(tempfile.mkdtemp(prefix="dp_correctness_"))
    serial = run_batch(args.model_id, 1, args.tp_degree, args.steps,
                       base / "serial", args.extra)
    parallel = run_batch(args.model_id, args.dp, args.tp_degree, args.steps,
                         base / "parallel", args.extra)

    failures = 0
    for s, p in zip(serial, parallel):
        a, b = torch.load(s), torch.load(p)
        if torch.equal(a, b):
            print(f"PASS bit-identical: {s.name}")
        else:
            diff = (a.float() - b.float()).abs().max().item()
            print(f"FAIL {s.name}: max_abs_diff={diff:.3e}")
            failures += 1
    print(f"[verify_dp_correctness] {len(serial) - failures}/{len(serial)} identical")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on device** (Wan first — cheapest staged model; then Flux/LTX-2/HYV/Qwen)

Run (detached): `setsid python scripts/verify_dp_correctness.py --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers --dp 2 --steps 2 > /tmp/dp_wan.log 2>&1 &` then monitor `tail -f /tmp/dp_wan.log`.
Expected: `4/4 identical`, exit 0. **Verify evidence, not exit codes**: the log must show `[dp-router] worker 0: cores 0-3` and `worker 1: cores 4-7`, and both runs must reuse the SAME compiled dir (compile once, load k times — no recompile lines in the dp=2 run).

- [ ] **Step 3: Commit and push**

```bash
git add scripts/verify_dp_correctness.py
git commit -m "test(dp): on-device dp=k vs dp=1 bit-identical correctness script"
git push
```

---

### Task 16: Mode × model smoke matrix (**requires trn2 hardware**)

**Files:**
- Modify: the existing verify_cli matrix under `scripts/` (see `scripts/` for the file added by commit `71cb6ba`, per `docs/superpowers/specs/2026-07-05-verify-cli-parallelism-matrix-design.md`) — add mode rows

**Interfaces:** Consumes the CLI's `--mode` flag. Rows: {FLUX.1-dev, HunyuanVideo, Qwen-Image, Wan2.2, LTX-2} × {latency, throughput, mixed} (HYV-1.5 excluded — scaffold). Each row asserts: exit 0, expected dp/cfg/cp echoed by the CLI (add a one-line `[difflet] parallel: dp=… cfg=… cp=… tp=…` print in `main()` when dispatching — tiny follow-through change, covered by a unit assert in `test_main_dp.py`), mode-specific compiled dir exists with fresh artifacts, all request outputs present.

- [ ] **Step 1: Add the parallel-echo line** in `main.py` where mode resolution lands (inside the `mode_cfg is not None` branch):

```python
            print(
                f"[difflet] parallel: dp={args.dp or 1} "
                f"cfg={2 if args.cfg_parallel else 1} cp={args.cp_degree or 1} "
                f"(mode={args.mode})",
                flush=True,
            )
```

with a unit test in `test_main_dp.py` asserting the line appears (capsys) for `--mode mixed` on Wan: `"parallel: dp=2 cfg=2 cp=1 (mode=mixed)"`.

- [ ] **Step 1b: Extend the matrix script** following its existing row format (read the script first; keep its conventions — the diff is additive mode rows only).
- [ ] **Step 2: Run the matrix detached on device**; verify per the repo rule: compiled-dir prefixes per mode, compile times, output files — NOT just exit zero. Throughput rows on models whose latency rows already compiled dp=1/cfg=1/cp=1 artifacts must show ZERO new compilation (cache-reuse proof).
- [ ] **Step 3: Commit and push**

```bash
git add scripts/ difflet/cli/main.py tests/unit/cli/test_main_dp.py
git commit -m "test(dp): mode x model verify_cli smoke rows + parallel echo"
git push
```

---

## Execution notes

- Tasks 1–14 are CPU-only and executable in this environment; Tasks 15–16 need trn2 hardware with model weights and will be long — detach with `setsid`, monitor log tails.
- Task order is dependency order; Tasks 8–12 are mutually independent (any order after 7).
- If any existing unit test breaks on the `--prompt required=False` change (Task 7), the fix belongs in `_validate_dp` coverage, not in reverting the parser.
