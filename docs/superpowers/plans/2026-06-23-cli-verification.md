# CLI Verification Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `scripts/verify_cli.py` — a script that runs `difflet download → compile → generate` for all 5 supported models, checks expected artifacts exist after each step, and writes a pass/fail/skip summary to a timestamped log file.

**Architecture:** Single Python script with four layers: (1) a static model config dict keyed by short name, (2) a `run_step()` function that shells out via `subprocess.run` and checks artifact globs, (3) a `_run_model()` function that sequences the three steps and propagates FAIL→SKIP, and (4) a `main()` that sets up the log file and drives the loop. A pure `format_summary()` function writes the final table.

**Tech Stack:** Python 3.10+ stdlib only — `argparse`, `subprocess`, `pathlib`, `datetime`, `dataclasses`, `enum`. No third-party deps. Tests use `pytest` + `unittest.mock`.

## Global Constraints

- New file only: `scripts/verify_cli.py`. No changes to any existing source file.
- 5 models: `flux`, `ltx_2`, `wan`, `hunyuan_video`, `qwen_image`. HunyuanVideo 1.5 excluded.
- Log file at `/tmp/logs/verify_cli_<YYYYMMDD_HHMMSS>.log`. Work dirs at `/tmp/logs/verify_<model_key>/`.
- On FAIL for any step: mark remaining steps for that model as SKIP; move to next model.
- Step statuses: `PASS`, `FAIL`, `SKIP`.
- Artifact glob check: at least one filesystem match required per glob pattern.
- Terminal output: log path at start, one progress line per step, log path at end.
- All `difflet` commands use exact flags from `docs/cli-staged-commands.md`.

---

## File Map

```
scripts/verify_cli.py          — new, all implementation (~210 lines)
tests/unit/test_verify_cli.py  — new, all unit tests
```

---

### Task 1: Data structures and model config

**Files:**
- Create: `scripts/verify_cli.py` (partial — enums, dataclass, MODEL_CONFIGS only)
- Create: `tests/unit/test_verify_cli.py` (partial — config tests only)

**Interfaces:**
- Produces:
  - `Status` — `str` enum with values `"PASS"`, `"FAIL"`, `"SKIP"`
  - `StepResult(status: Status, reason: str = "", cmd: list[str] = [])` — dataclass
  - `MODEL_CONFIGS: dict[str, dict]` — keys: `"flux"`, `"ltx_2"`, `"wan"`, `"hunyuan_video"`, `"qwen_image"`; each value has keys: `model_id`, `compile_args`, `generate_args`, `output_filename`, `staged`, `download_glob`, `compile_globs`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_verify_cli.py
from __future__ import annotations
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "scripts"))

from verify_cli import Status, StepResult, MODEL_CONFIGS, STEPS


def test_all_models_present():
    assert set(MODEL_CONFIGS.keys()) == {"flux", "ltx_2", "wan", "hunyuan_video", "qwen_image"}


def test_steps_order():
    assert STEPS == ["download", "compile", "generate"]


def test_each_model_has_required_keys():
    required = {"model_id", "compile_args", "generate_args", "output_filename",
                "staged", "download_glob", "compile_globs"}
    for key, cfg in MODEL_CONFIGS.items():
        missing = required - cfg.keys()
        assert not missing, f"{key} missing keys: {missing}"


def test_staged_flags():
    assert MODEL_CONFIGS["flux"]["staged"] is False
    assert MODEL_CONFIGS["ltx_2"]["staged"] is False
    assert MODEL_CONFIGS["wan"]["staged"] is True
    assert MODEL_CONFIGS["hunyuan_video"]["staged"] is True
    assert MODEL_CONFIGS["qwen_image"]["staged"] is True


def test_compile_glob_counts():
    assert len(MODEL_CONFIGS["flux"]["compile_globs"]) == 1
    assert len(MODEL_CONFIGS["ltx_2"]["compile_globs"]) == 1
    assert len(MODEL_CONFIGS["wan"]["compile_globs"]) == 2
    assert len(MODEL_CONFIGS["hunyuan_video"]["compile_globs"]) == 3
    assert len(MODEL_CONFIGS["qwen_image"]["compile_globs"]) == 3


def test_step_result_defaults():
    r = StepResult(status=Status.PASS)
    assert r.reason == ""
    assert r.cmd == []


def test_status_values():
    assert Status.PASS == "PASS"
    assert Status.FAIL == "FAIL"
    assert Status.SKIP == "SKIP"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/ubuntu/Difflet && python -m pytest tests/unit/test_verify_cli.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'verify_cli'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/verify_cli.py
from __future__ import annotations

import argparse
import datetime
import pathlib
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum

STEPS = ["download", "compile", "generate"]

_HF_CACHE = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
_DC = pathlib.Path.home() / ".cache" / "difflet"


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class StepResult:
    status: Status
    reason: str = ""
    cmd: list[str] = field(default_factory=list)


MODEL_CONFIGS: dict[str, dict] = {
    "flux": {
        "model_id": "black-forest-labs/FLUX.1-dev",
        "compile_args": ["--tp-degree", "2", "--cp-degree", "2",
                         "--height", "1024", "--width", "1024"],
        "generate_args": ["--tp-degree", "2", "--cp-degree", "2",
                          "--height", "1024", "--width", "1024",
                          "--prompt", "a cat sitting on a bench"],
        "output_filename": "flux.png",
        "staged": False,
        "download_glob": str(_HF_CACHE / "models--black-forest-labs--FLUX.1-dev" / "snapshots" / "*"),
        "compile_globs": [str(_DC / "flux" / "*")],
    },
    "ltx_2": {
        "model_id": "Lightricks/LTX-2",
        "compile_args": ["--tp-degree", "4",
                         "--height", "512", "--width", "768", "--num-frames", "121"],
        "generate_args": ["--tp-degree", "4",
                          "--height", "512", "--width", "768", "--num-frames", "121",
                          "--prompt", "a cat walking through a garden"],
        "output_filename": "ltx2.mp4",
        "staged": False,
        "download_glob": str(_HF_CACHE / "models--Lightricks--LTX-2" / "snapshots" / "*"),
        "compile_globs": [str(_DC / "ltx_2" / "*")],
    },
    "wan": {
        "model_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "compile_args": ["--tp-degree", "2", "--cp-degree", "2",
                         "--height", "480", "--width", "832", "--num-frames", "9"],
        "generate_args": ["--tp-degree", "2", "--cp-degree", "2",
                          "--height", "480", "--width", "832", "--num-frames", "9",
                          "--steps", "50", "--guidance-scale", "1.0", "--seed", "42",
                          "--prompt", "a cat walking through a garden"],
        "output_filename": "wan.mp4",
        "staged": True,
        "download_glob": str(_HF_CACHE / "models--Wan-AI--Wan2.2-T2V-A14B-Diffusers" / "snapshots" / "*"),
        "compile_globs": [
            str(_DC / "wan_transformer_tp2cp2_*"),
            str(_DC / "wan_vae_*"),
        ],
    },
    "hunyuan_video": {
        "model_id": "hunyuanvideo-community/HunyuanVideo",
        "compile_args": ["--tp-degree", "2", "--cp-degree", "2",
                         "--height", "320", "--width", "512", "--num-frames", "61"],
        "generate_args": ["--tp-degree", "2", "--cp-degree", "2",
                          "--height", "320", "--width", "512", "--num-frames", "61",
                          "--steps", "50", "--guidance-scale", "6.0", "--seed", "42",
                          "--prompt", "a cat sitting on a bench"],
        "output_filename": "hunyuan.mp4",
        "staged": True,
        "download_glob": str(_HF_CACHE / "models--hunyuanvideo-community--HunyuanVideo" / "snapshots" / "*"),
        "compile_globs": [
            str(_DC / "hunyuan_video_clip"),
            str(_DC / "hunyuan_video_llama_*"),
            str(_DC / "hunyuan_video_dit_*"),
        ],
    },
    "qwen_image": {
        "model_id": "Qwen/Qwen-Image",
        "compile_args": ["--tp-degree", "2", "--cp-degree", "2",
                         "--height", "1024", "--width", "1024"],
        "generate_args": ["--tp-degree", "2", "--cp-degree", "2",
                          "--height", "1024", "--width", "1024",
                          "--steps", "50", "--guidance-scale", "7.5", "--seed", "42",
                          "--prompt", "a cat sitting on a bench"],
        "output_filename": "qwen.png",
        "staged": True,
        "download_glob": str(_HF_CACHE / "models--Qwen--Qwen-Image" / "snapshots" / "*"),
        "compile_globs": [
            str(_DC / "qwen_image_enc_*"),
            str(_DC / "qwen_image_dit_*"),
            str(_DC / "qwen_image_vae_*"),
        ],
    },
}
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /home/ubuntu/Difflet && python -m pytest tests/unit/test_verify_cli.py -v 2>&1 | tail -20
```

Expected: all 8 tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add scripts/verify_cli.py tests/unit/test_verify_cli.py
git commit -m "feat(verify): data structures and model config"
```

---

### Task 2: `_check_glob()` and `run_step()`

**Files:**
- Modify: `scripts/verify_cli.py` — append after MODEL_CONFIGS
- Modify: `tests/unit/test_verify_cli.py` — append new tests

**Interfaces:**
- Consumes: `Status`, `StepResult` from Task 1
- Produces:
  - `_check_glob(glob_str: str) -> bool` — returns True if glob matches ≥1 path; handles no-wildcard paths with `.exists()`
  - `run_step(cmd: list[str], artifact_globs: list[str], log_fh, *, _check_artifact=None) -> StepResult` — runs subprocess, checks globs; `_check_artifact` overrides `_check_glob` in tests

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_verify_cli.py`:

```python
import io
from unittest.mock import MagicMock, patch

from verify_cli import _check_glob, run_step


def _log():
    return io.StringIO()


def test_check_glob_no_wildcard_exists(tmp_path):
    p = tmp_path / "somedir"
    p.mkdir()
    assert _check_glob(str(p)) is True


def test_check_glob_no_wildcard_missing(tmp_path):
    assert _check_glob(str(tmp_path / "missing")) is False


def test_check_glob_wildcard_matches(tmp_path):
    (tmp_path / "abc123").mkdir()
    assert _check_glob(str(tmp_path / "*")) is True


def test_check_glob_wildcard_no_match(tmp_path):
    assert _check_glob(str(tmp_path / "*")) is False


def test_run_step_pass_no_globs():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        result = run_step(["difflet", "download", "--model-id", "foo"], [], _log())
    assert result.status == Status.PASS
    assert result.cmd == ["difflet", "download", "--model-id", "foo"]


def test_run_step_fail_nonzero_exit():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=2)
        result = run_step(["difflet", "compile"], [], _log())
    assert result.status == Status.FAIL
    assert "exit code 2" in result.reason


def test_run_step_fail_missing_artifact():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        result = run_step(
            ["difflet", "compile"],
            ["~/.cache/difflet/flux/*"],
            _log(),
            _check_artifact=lambda g: False,
        )
    assert result.status == Status.FAIL
    assert "artifact not found" in result.reason
    assert "~/.cache/difflet/flux/*" in result.reason


def test_run_step_pass_with_artifacts():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        result = run_step(
            ["difflet", "compile"],
            ["~/.cache/difflet/flux/*", "~/.cache/difflet/flux/neff"],
            _log(),
            _check_artifact=lambda g: True,
        )
    assert result.status == Status.PASS


def test_run_step_writes_banner_to_log():
    log = _log()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        run_step(["difflet", "download", "--model-id", "foo"], [], log)
    content = log.getvalue()
    assert "difflet download --model-id foo" in content
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/ubuntu/Difflet && python -m pytest tests/unit/test_verify_cli.py -k "check_glob or run_step" -v 2>&1 | tail -20
```

Expected: `AttributeError` or `ImportError` — `_check_glob` and `run_step` not yet defined

- [ ] **Step 3: Write the implementation**

Append to `scripts/verify_cli.py` after MODEL_CONFIGS:

```python

def _check_glob(glob_str: str) -> bool:
    """Return True if glob_str matches at least one existing path."""
    p = pathlib.Path(glob_str).expanduser()
    if "*" not in glob_str:
        return p.exists()
    return bool(list(p.parent.glob(p.name)))


def run_step(
    cmd: list[str],
    artifact_globs: list[str],
    log_fh,
    *,
    _check_artifact=None,
) -> StepResult:
    checker = _check_artifact or _check_glob
    ts = datetime.datetime.now().isoformat()
    log_fh.write(f"\n{'='*60}\nCMD: {' '.join(cmd)}\nSTARTED: {ts}\n{'='*60}\n")
    log_fh.flush()

    proc = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    log_fh.flush()

    if proc.returncode != 0:
        return StepResult(status=Status.FAIL, reason=f"exit code {proc.returncode}", cmd=cmd)

    for glob_str in artifact_globs:
        if not checker(glob_str):
            return StepResult(
                status=Status.FAIL,
                reason=f"artifact not found: {glob_str}",
                cmd=cmd,
            )

    return StepResult(status=Status.PASS, cmd=cmd)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /home/ubuntu/Difflet && python -m pytest tests/unit/test_verify_cli.py -v 2>&1 | tail -25
```

Expected: all tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add scripts/verify_cli.py tests/unit/test_verify_cli.py
git commit -m "feat(verify): add _check_glob and run_step"
```

---

### Task 3: `_build_cmd()` and `_run_model()`

**Files:**
- Modify: `scripts/verify_cli.py` — append after `run_step`
- Modify: `tests/unit/test_verify_cli.py` — append new tests

**Interfaces:**
- Consumes: `run_step`, `StepResult`, `Status`, `MODEL_CONFIGS`, `STEPS` from Tasks 1–2
- Produces:
  - `_build_cmd(step: str, model_key: str, cfg: dict, work_dir: pathlib.Path) -> tuple[list[str], list[str]]` — returns `(cmd, artifact_globs)`
  - `_run_model(model_key: str, cfg: dict, log_fh, *, _check_artifact=None) -> dict[str, StepResult]` — sequences steps, propagates FAIL→SKIP

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_verify_cli.py`:

```python
from verify_cli import _build_cmd, _run_model


def test_build_cmd_download():
    work_dir = pathlib.Path("/tmp/test_work")
    cmd, globs = _build_cmd("download", "flux", MODEL_CONFIGS["flux"], work_dir)
    assert cmd == ["difflet", "download", "--model-id", "black-forest-labs/FLUX.1-dev"]
    assert len(globs) == 1
    assert "FLUX.1-dev" in globs[0]


def test_build_cmd_compile_flux():
    work_dir = pathlib.Path("/tmp/test_work")
    cmd, globs = _build_cmd("compile", "flux", MODEL_CONFIGS["flux"], work_dir)
    assert cmd[:3] == ["difflet", "compile", "--model-id"]
    assert "--tp-degree" in cmd
    assert "--cp-degree" in cmd
    assert globs == MODEL_CONFIGS["flux"]["compile_globs"]


def test_build_cmd_generate_non_staged():
    work_dir = pathlib.Path("/tmp/test_work")
    cmd, globs = _build_cmd("generate", "flux", MODEL_CONFIGS["flux"], work_dir)
    assert "--work-dir" not in cmd
    assert "--keep-work-dir" not in cmd
    assert "--output" in cmd
    output_idx = cmd.index("--output")
    assert cmd[output_idx + 1] == str(work_dir / "flux.png")
    assert globs == [str(work_dir / "flux.png")]


def test_build_cmd_generate_staged():
    work_dir = pathlib.Path("/tmp/test_work")
    cmd, globs = _build_cmd("generate", "wan", MODEL_CONFIGS["wan"], work_dir)
    assert "--work-dir" in cmd
    assert "--keep-work-dir" in cmd
    wd_idx = cmd.index("--work-dir")
    assert cmd[wd_idx + 1] == str(work_dir)
    assert globs == [str(work_dir / "wan.mp4")]


def test_run_model_all_pass():
    log = _log()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        results = _run_model("flux", MODEL_CONFIGS["flux"], log, _check_artifact=lambda g: True)
    assert results["download"].status == Status.PASS
    assert results["compile"].status == Status.PASS
    assert results["generate"].status == Status.PASS


def test_run_model_fail_propagates_to_skip():
    log = _log()

    def fake_run(cmd, **kwargs):
        rc = 1 if "compile" in cmd else 0
        return MagicMock(returncode=rc)

    with patch("subprocess.run", side_effect=fake_run):
        results = _run_model("flux", MODEL_CONFIGS["flux"], log, _check_artifact=lambda g: True)

    assert results["download"].status == Status.PASS
    assert results["compile"].status == Status.FAIL
    assert results["generate"].status == Status.SKIP


def test_run_model_download_fail_skips_all():
    log = _log()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)
        results = _run_model("flux", MODEL_CONFIGS["flux"], log, _check_artifact=lambda g: True)

    assert results["download"].status == Status.FAIL
    assert results["compile"].status == Status.SKIP
    assert results["generate"].status == Status.SKIP
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/ubuntu/Difflet && python -m pytest tests/unit/test_verify_cli.py -k "build_cmd or run_model" -v 2>&1 | tail -20
```

Expected: `ImportError` — `_build_cmd` and `_run_model` not defined

- [ ] **Step 3: Write the implementation**

Append to `scripts/verify_cli.py` after `run_step`:

```python

def _build_cmd(
    step: str,
    model_key: str,
    cfg: dict,
    work_dir: pathlib.Path,
) -> tuple[list[str], list[str]]:
    model_id = cfg["model_id"]
    base = ["difflet", step, "--model-id", model_id]

    if step == "download":
        return base, [cfg["download_glob"]]

    if step == "compile":
        return base + cfg["compile_args"], cfg["compile_globs"]

    # generate
    output_path = work_dir / cfg["output_filename"]
    cmd = base + cfg["generate_args"] + ["--output", str(output_path)]
    if cfg["staged"]:
        cmd += ["--work-dir", str(work_dir), "--keep-work-dir"]
    return cmd, [str(output_path)]


def _run_model(
    model_key: str,
    cfg: dict,
    log_fh,
    *,
    _check_artifact=None,
) -> dict[str, StepResult]:
    work_dir = pathlib.Path(f"/tmp/logs/verify_{model_key}")
    work_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, StepResult] = {}
    failed = False

    for step in STEPS:
        if failed:
            results[step] = StepResult(status=Status.SKIP)
            continue
        cmd, artifact_globs = _build_cmd(step, model_key, cfg, work_dir)
        result = run_step(cmd, artifact_globs, log_fh, _check_artifact=_check_artifact)
        results[step] = result
        if result.status == Status.FAIL:
            failed = True

    return results
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /home/ubuntu/Difflet && python -m pytest tests/unit/test_verify_cli.py -v 2>&1 | tail -30
```

Expected: all tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add scripts/verify_cli.py tests/unit/test_verify_cli.py
git commit -m "feat(verify): add _build_cmd and _run_model"
```

---

### Task 4: `format_summary()`

**Files:**
- Modify: `scripts/verify_cli.py` — append after `_run_model`
- Modify: `tests/unit/test_verify_cli.py` — append new tests

**Interfaces:**
- Consumes: `Status`, `StepResult`, `MODEL_CONFIGS`, `STEPS` from Tasks 1–3
- Produces:
  - `format_summary(results: dict[str, dict[str, StepResult]], active_configs: dict[str, dict], log_path: str) -> str` — returns the full summary block as a string; pure function

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_verify_cli.py`:

```python
from verify_cli import format_summary

_MINI_CFG = {
    "flux": {
        "download_glob": "~/.cache/hf/models--black-forest-labs--FLUX.1-dev/snapshots/*",
        "compile_globs": ["~/.cache/difflet/flux/*"],
        "output_filename": "flux.png",
    },
    "ltx_2": {
        "download_glob": "~/.cache/hf/models--Lightricks--LTX-2/snapshots/*",
        "compile_globs": ["~/.cache/difflet/ltx_2/*"],
        "output_filename": "ltx2.mp4",
    },
}

_ALL_PASS = {
    "flux": {
        "download": StepResult(Status.PASS, cmd=["difflet", "download"]),
        "compile": StepResult(Status.PASS, cmd=["difflet", "compile"]),
        "generate": StepResult(Status.PASS, cmd=["difflet", "generate"]),
    },
    "ltx_2": {
        "download": StepResult(Status.PASS, cmd=["difflet", "download"]),
        "compile": StepResult(Status.PASS, cmd=["difflet", "compile"]),
        "generate": StepResult(Status.PASS, cmd=["difflet", "generate"]),
    },
}

_WITH_FAILURE = {
    "flux": {
        "download": StepResult(Status.PASS, cmd=["difflet", "download", "--model-id", "flux"]),
        "compile": StepResult(Status.FAIL, reason="exit code 1", cmd=["difflet", "compile", "--model-id", "flux"]),
        "generate": StepResult(Status.SKIP, cmd=[]),
    },
    "ltx_2": {
        "download": StepResult(Status.PASS, cmd=["difflet", "download"]),
        "compile": StepResult(Status.PASS, cmd=["difflet", "compile"]),
        "generate": StepResult(Status.PASS, cmd=["difflet", "generate"]),
    },
}


def test_format_summary_all_pass_has_no_failed_section():
    text = format_summary(_ALL_PASS, _MINI_CFG, "/tmp/logs/test.log")
    assert "PASS" in text
    assert "FAILED COMMANDS" not in text


def test_format_summary_contains_summary_header():
    text = format_summary(_ALL_PASS, _MINI_CFG, "/tmp/logs/test.log")
    assert "SUMMARY" in text


def test_format_summary_lists_all_models():
    text = format_summary(_ALL_PASS, _MINI_CFG, "/tmp/logs/test.log")
    assert "flux" in text
    assert "ltx_2" in text


def test_format_summary_shows_failure_details():
    text = format_summary(_WITH_FAILURE, _MINI_CFG, "/tmp/logs/test.log")
    assert "FAILED COMMANDS" in text
    assert "[flux] compile" in text
    assert "exit code 1" in text
    assert "/tmp/logs/test.log" in text


def test_format_summary_shows_skip():
    text = format_summary(_WITH_FAILURE, _MINI_CFG, "/tmp/logs/test.log")
    assert "SKIP" in text


def test_format_summary_artifact_locations():
    text = format_summary(_ALL_PASS, _MINI_CFG, "/tmp/logs/test.log")
    assert "ARTIFACT LOCATIONS" in text
    assert "~/.cache/difflet/flux/*" in text
    assert "/tmp/logs/verify_flux/flux.png" in text
    assert "/tmp/logs/verify_ltx_2/ltx2.mp4" in text
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/ubuntu/Difflet && python -m pytest tests/unit/test_verify_cli.py -k "format_summary" -v 2>&1 | tail -20
```

Expected: `ImportError` — `format_summary` not defined

- [ ] **Step 3: Write the implementation**

Append to `scripts/verify_cli.py` after `_run_model`:

```python

def format_summary(
    results: dict[str, dict[str, StepResult]],
    active_configs: dict[str, dict],
    log_path: str,
) -> str:
    NAME_W, COL_W = 18, 10
    sep = "=" * 60

    lines: list[str] = [f"\n{sep}", "=== SUMMARY ===", ""]

    header = f"{'Model':<{NAME_W}}" + "".join(f"  {s:<{COL_W - 2}}" for s in STEPS)
    lines.append(header)
    lines.append("-" * NAME_W + ("  " + "-" * (COL_W - 2)) * len(STEPS))

    for model_key in active_configs:
        row = f"{model_key:<{NAME_W}}"
        for step in STEPS:
            r = results.get(model_key, {}).get(step)
            status = r.status.value if r else Status.SKIP.value
            row += f"  {status:<{COL_W - 2}}"
        lines.append(row)

    failures = [
        (model_key, step, r)
        for model_key in active_configs
        for step in STEPS
        if (r := results.get(model_key, {}).get(step)) and r.status == Status.FAIL
    ]

    if failures:
        lines += ["", "FAILED COMMANDS:"]
        for model_key, step, r in failures:
            lines += [
                f"  [{model_key}] {step}",
                f"    cmd:  {' '.join(r.cmd)}",
                f"    why:  {r.reason}",
                f"    log:  {log_path}",
                "",
            ]

    lines += ["", "ARTIFACT LOCATIONS:"]
    for model_key, cfg in active_configs.items():
        lines.append(f"  {model_key:<14} download   {cfg['download_glob']}")
        for g in cfg["compile_globs"]:
            lines.append(f"  {model_key:<14} compile    {g}")
        out = f"/tmp/logs/verify_{model_key}/{cfg['output_filename']}"
        lines.append(f"  {model_key:<14} generate   {out}")

    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /home/ubuntu/Difflet && python -m pytest tests/unit/test_verify_cli.py -v 2>&1 | tail -35
```

Expected: all tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add scripts/verify_cli.py tests/unit/test_verify_cli.py
git commit -m "feat(verify): add format_summary"
```

---

### Task 5: `main()` — arg parsing, log setup, model loop

**Files:**
- Modify: `scripts/verify_cli.py` — append `main()` and `if __name__ == "__main__"` guard
- Modify: `tests/unit/test_verify_cli.py` — append integration smoke test

**Interfaces:**
- Consumes: `MODEL_CONFIGS`, `STEPS`, `_run_model`, `format_summary` from Tasks 1–4
- Produces:
  - `main(argv: list[str] | None = None) -> None` — entry point; `argv=None` reads `sys.argv`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_verify_cli.py`:

```python
from verify_cli import main


def test_main_runs_all_models(tmp_path, monkeypatch):
    """main() should call _run_model once per model and write a log."""
    called_models = []

    def fake_run_model(model_key, cfg, log_fh, **kwargs):
        called_models.append(model_key)
        return {
            "download": StepResult(Status.PASS, cmd=[]),
            "compile": StepResult(Status.PASS, cmd=[]),
            "generate": StepResult(Status.PASS, cmd=[]),
        }

    monkeypatch.setattr("verify_cli._run_model", fake_run_model)
    monkeypatch.setattr("verify_cli.pathlib.Path.mkdir", lambda *a, **kw: None)

    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    import builtins
    real_open = builtins.open

    def fake_open(path, mode="r", **kw):
        if "verify_cli_" in str(path):
            return real_open(str(log_dir / "verify_cli_test.log"), mode, **kw)
        return real_open(path, mode, **kw)

    monkeypatch.setattr("builtins.open", fake_open)

    main([])  # no --models arg → defaults to all 5

    assert set(called_models) == {"flux", "ltx_2", "wan", "hunyuan_video", "qwen_image"}


def test_main_respects_models_flag(monkeypatch, tmp_path):
    """--models flux ltx_2 should only run those two models."""
    called_models = []

    def fake_run_model(model_key, cfg, log_fh, **kwargs):
        called_models.append(model_key)
        return {
            "download": StepResult(Status.PASS, cmd=[]),
            "compile": StepResult(Status.PASS, cmd=[]),
            "generate": StepResult(Status.PASS, cmd=[]),
        }

    monkeypatch.setattr("verify_cli._run_model", fake_run_model)
    monkeypatch.setattr("verify_cli.pathlib.Path.mkdir", lambda *a, **kw: None)

    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    import builtins
    real_open = builtins.open

    def fake_open(path, mode="r", **kw):
        if "verify_cli_" in str(path):
            return real_open(str(log_dir / "verify_cli_test.log"), mode, **kw)
        return real_open(path, mode, **kw)

    monkeypatch.setattr("builtins.open", fake_open)

    main(["--models", "flux", "ltx_2"])

    assert set(called_models) == {"flux", "ltx_2"}
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/ubuntu/Difflet && python -m pytest tests/unit/test_verify_cli.py -k "main" -v 2>&1 | tail -20
```

Expected: `ImportError` — `main` not defined

- [ ] **Step 3: Write the implementation**

Append to `scripts/verify_cli.py` after `format_summary`:

```python

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="verify_cli",
        description="Verify difflet CLI commands for all supported models.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=list(MODEL_CONFIGS.keys()),
        default=list(MODEL_CONFIGS.keys()),
        metavar="MODEL",
        help="Models to verify (default: all). Choices: " + ", ".join(MODEL_CONFIGS.keys()),
    )
    args = parser.parse_args(argv)

    pathlib.Path("/tmp/logs").mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"/tmp/logs/verify_cli_{ts}.log"

    print(f"Log: {log_path}")

    active_configs = {k: MODEL_CONFIGS[k] for k in args.models}
    all_results: dict[str, dict[str, StepResult]] = {}

    with open(log_path, "w") as log_fh:
        log_fh.write(f"verify_cli started {datetime.datetime.now().isoformat()}\n")
        log_fh.write(f"models: {args.models}\n")

        for model_key in args.models:
            cfg = MODEL_CONFIGS[model_key]
            log_fh.write(f"\n{'#'*60}\n# MODEL: {model_key}\n{'#'*60}\n")
            log_fh.flush()

            for step in STEPS:
                print(f"  [{model_key}] {step} ...")

            all_results[model_key] = _run_model(model_key, cfg, log_fh)

        summary = format_summary(all_results, active_configs, log_path)
        log_fh.write(summary)
        print(summary)

    print(f"\nDone. Log: {log_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all tests to confirm they pass**

```bash
cd /home/ubuntu/Difflet && python -m pytest tests/unit/test_verify_cli.py -v 2>&1 | tail -40
```

Expected: all tests `PASSED`. Note: `test_main_*` tests monkeypatch `_run_model` so no real `difflet` binary is needed.

- [ ] **Step 5: Smoke-test the script interface (no real hardware needed)**

```bash
cd /home/ubuntu/Difflet && python scripts/verify_cli.py --help
```

Expected output includes `--models` flag and list of valid model names.

```bash
python scripts/verify_cli.py --models flux 2>&1 | head -5
```

Expected: prints `Log: /tmp/logs/verify_cli_<timestamp>.log` and `[flux] download ...` before failing (no real `difflet` binary in this env — that's expected).

- [ ] **Step 6: Commit**

```bash
git add scripts/verify_cli.py tests/unit/test_verify_cli.py
git commit -m "feat(verify): add main entrypoint; verification script complete"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| 5 models (no HunyuanVideo 1.5) | Task 1 |
| `--models` subset flag | Task 5 |
| FAIL → SKIP propagation | Task 3 |
| Artifact glob check per step | Task 2 |
| Timestamped log at `/tmp/logs/verify_cli_*.log` | Task 5 |
| Terminal: log path + per-step progress | Task 5 |
| Summary table at end of log | Task 4 |
| FAILED COMMANDS section with cmd + reason + log | Task 4 |
| ARTIFACT LOCATIONS section | Task 4 |
| Work dirs at `/tmp/logs/verify_<model>/` | Task 3 |
| `--keep-work-dir` for staged models | Task 3 |
| Exact CLI flags from docs | Task 1 |

All requirements covered. No placeholders. Type signatures consistent across all tasks (`StepResult`, `Status`, `MODEL_CONFIGS` referenced by the same names throughout).
