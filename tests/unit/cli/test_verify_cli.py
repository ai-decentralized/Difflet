"""Unit tests for scripts/verify_cli.py — the parallelism-config matrix runner.

Pure logic only: no device, no subprocess, no network. Subprocess calls are
mocked; artifact existence is injected. Drift-guard tests import
difflet.cli.main to pin the script's local skip-rule sets to the CLI's
source-of-truth sets.
"""
from __future__ import annotations

import io
import time
import os
import pathlib
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent / "scripts"))

from verify_cli import (
    _DIFFLET_CMD,
    CP_UNSUPPORTED,
    DISTILLED,
    EXPECTED_FAIL_CELLS,
    MODELS,
    PARALLEL_CONFIGS,
    SP_SUPPORTED,
    ULYSSES_UNSUPPORTED,
    Status,
    StepResult,
    build_compile_cmd,
    build_download_cmd,
    build_generate_cmd,
    cell_outcome,
    compute_exit_code,
    format_summary,
    plan_cells,
    run_cell,
    run_step,
    skip_reason,
)

MODEL_KEYS = ["flux", "qwen_image", "ltx_2", "wan", "wan2_1",
              "hunyuan_video", "hunyuan_video_15"]
CONFIG_KEYS = ["tp4", "tp2cp2", "tp2cfg", "tp4sp", "dp2tp2", "tp2cp2ulysses"]


# ---------------------------------------------------------------- matrix shape

def test_all_seven_models_present():
    assert list(MODELS.keys()) == MODEL_KEYS


def test_all_six_configs_present():
    assert list(PARALLEL_CONFIGS.keys()) == CONFIG_KEYS


def test_every_config_uses_exactly_four_cores():
    for cfg in PARALLEL_CONFIGS.values():
        assert cfg.world_size == 4, cfg.key


def test_config_flags():
    assert PARALLEL_CONFIGS["tp4"].flags == ("--tp-degree", "4")
    assert PARALLEL_CONFIGS["tp2cp2"].flags == ("--tp-degree", "2", "--cp-degree", "2")
    assert PARALLEL_CONFIGS["tp2cfg"].flags == ("--tp-degree", "2", "--cfg-parallel")
    assert PARALLEL_CONFIGS["tp4sp"].flags == ("--tp-degree", "4", "--sp")
    assert PARALLEL_CONFIGS["dp2tp2"].flags == ("--tp-degree", "2", "--dp", "2")
    assert PARALLEL_CONFIGS["tp2cp2ulysses"].flags == (
        "--tp-degree", "2", "--cp-degree", "2", "--cp-mode", "ulysses")


def test_model_ids():
    assert MODELS["flux"].model_id == "black-forest-labs/FLUX.1-dev"
    assert MODELS["qwen_image"].model_id == "Qwen/Qwen-Image"
    assert MODELS["ltx_2"].model_id == "Lightricks/LTX-2"
    assert MODELS["wan"].model_id == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    assert MODELS["wan2_1"].model_id == "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    assert MODELS["hunyuan_video"].model_id == "hunyuanvideo-community/HunyuanVideo"
    assert MODELS["hunyuan_video_15"].model_id == (
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v")


def test_staged_flags():
    staged = {k for k, m in MODELS.items() if m.staged}
    assert staged == {"qwen_image", "wan", "wan2_1", "hunyuan_video"}


# ------------------------------------------------------------- skip-rule sets
#
# These were drift guards pinning hand-written literals here to hand-written
# literals in difflet.cli.main. Both sides now derive from each model's registry
# ModelCapabilities, so pinning them to each other would only assert that
# frozenset comprehension works. What is still worth asserting is that the
# derived sets match the support matrix documented in README.md -- that catches
# a capability edited in the registry without the docs following.

def test_distilled_set_matches_documented_matrix():
    assert DISTILLED == {"flux", "qwen_image", "hunyuan_video", "hunyuan_video_15"}


def test_sp_supported_set_matches_documented_matrix():
    assert SP_SUPPORTED == {"flux", "wan", "wan2_1", "hunyuan_video", "qwen_image"}


def test_all_model_ids_valid_in_cli():
    from difflet.cli.main import VALID_MODELS
    assert {m.model_id for m in MODELS.values()} <= VALID_MODELS


def test_cp_unsupported_models():
    # Rule lives in the model entries (ltx_2/entry.py, hunyuan_video/entry.py),
    # not in difflet.cli.main — pinned here as data.
    assert CP_UNSUPPORTED == {"ltx_2", "hunyuan_video_15"}


def test_ulysses_unsupported_models():
    # Rule lives in the model attention paths (modeling_hunyuan_video.py,
    # qwen_image/transformer.py raise on attention_mask under ulysses/ring),
    # not in difflet.cli.main — pinned here as data.
    assert ULYSSES_UNSUPPORTED == {"hunyuan_video"}


def test_expected_fail_cells():
    assert EXPECTED_FAIL_CELLS == {
        ("hunyuan_video_15", "tp4"),      # scaffold: NotImplementedError
        ("hunyuan_video_15", "dp2tp2"),   # same scaffold gap via the router
        ("hunyuan_video", "tp2cp2"),      # neuronx-cc 2.25 NCC_INLA001 internal error
        ("hunyuan_video", "dp2tp2"),      # VAE alloc failure: f121 exceeds 2-core replica
    }


# ---------------------------------------------------------------- skip rules

# The full support table from the design spec.
_EXPECTED_SKIPS = {
    ("flux", "tp2cfg"): "distilled",
    ("qwen_image", "tp2cfg"): "distilled",
    ("ltx_2", "tp2cp2"): "no-CP",
    ("ltx_2", "tp2cp2ulysses"): "no-CP",
    ("ltx_2", "tp4sp"): "no-SP",
    ("hunyuan_video", "tp2cfg"): "distilled",
    ("hunyuan_video", "tp2cp2ulysses"): "no-ulysses",
    ("hunyuan_video_15", "tp2cp2"): "no-CP",
    ("hunyuan_video_15", "tp2cp2ulysses"): "no-CP",
    ("hunyuan_video_15", "tp2cfg"): "distilled",
    ("hunyuan_video_15", "tp4sp"): "no-SP",
}


def test_skip_reason_full_support_table():
    for model_key in MODEL_KEYS:
        for config_key in CONFIG_KEYS:
            expected = _EXPECTED_SKIPS.get((model_key, config_key))
            assert skip_reason(model_key, config_key) == expected, (model_key, config_key)


def test_plan_cells_counts():
    cells = plan_cells(MODEL_KEYS, CONFIG_KEYS)
    assert len(cells) == 42
    skipped = [c for c in cells if c.skip_reason]
    runnable = [c for c in cells if not c.skip_reason]
    # tp2cp2ulysses is a CP config, so it adds the same two no-CP skips as
    # tp2cp2, plus hunyuan_video's no-ulysses skip (attention_mask); qwen's
    # tp4sp is runnable since the modeling_qwen SP fork landed (2026-09-02).
    assert len(skipped) == 11
    assert len(runnable) == 31
    xfail = {(c.model_key, c.config_key) for c in runnable if c.expected_fail}
    assert xfail == {("hunyuan_video_15", "tp4"), ("hunyuan_video_15", "dp2tp2"),
                     ("hunyuan_video", "tp2cp2"), ("hunyuan_video", "dp2tp2")}


def test_dp_config_never_skipped():
    for model_key in MODEL_KEYS:
        assert skip_reason(model_key, "dp2tp2") is None, model_key


def test_plan_cells_respects_subset_filters():
    cells = plan_cells(["wan"], ["tp4", "tp2cfg"])
    assert [(c.model_key, c.config_key) for c in cells] == [("wan", "tp4"), ("wan", "tp2cfg")]
    assert all(c.skip_reason is None for c in cells)


# ---------------------------------------------------------------- command building

def test_build_download_cmd():
    cmd = build_download_cmd(MODELS["flux"])
    assert cmd == _DIFFLET_CMD + ["download", "--model-id", "black-forest-labs/FLUX.1-dev"]


def test_difflet_cmd_is_module_invocation():
    # The editable install may point at a different checkout; the matrix must
    # test THIS checkout via `python -m` + cwd, not the console script.
    assert _DIFFLET_CMD[-2:] == ["-m", "difflet.cli.main"]


def test_download_globs_point_at_hf_hub_snapshots():
    for m in MODELS.values():
        assert "huggingface" in m.download_glob and "snapshots" in m.download_glob
        org, name = m.model_id.split("/")
        assert f"models--{org}--{name}" in m.download_glob


def test_build_compile_cmd_includes_config_and_shape_flags():
    cmd = build_compile_cmd(MODELS["wan"], PARALLEL_CONFIGS["tp2cp2"])
    assert cmd[: len(_DIFFLET_CMD) + 3] == _DIFFLET_CMD + [
        "compile", "--model-id", "Wan-AI/Wan2.2-T2V-A14B-Diffusers"]
    assert ("--tp-degree", "2") == (cmd[cmd.index("--tp-degree")], cmd[cmd.index("--tp-degree") + 1])
    assert ("--cp-degree", "2") == (cmd[cmd.index("--cp-degree")], cmd[cmd.index("--cp-degree") + 1])
    for flag, val in [("--height", "480"), ("--width", "832"), ("--num-frames", "9")]:
        assert val == cmd[cmd.index(flag) + 1]


def test_build_compile_cmd_cfg_parallel():
    cmd = build_compile_cmd(MODELS["ltx_2"], PARALLEL_CONFIGS["tp2cfg"])
    assert "--cfg-parallel" in cmd
    assert "--cp-degree" not in cmd


def test_build_compile_cmd_sp():
    cmd = build_compile_cmd(MODELS["flux"], PARALLEL_CONFIGS["tp4sp"])
    assert "--sp" in cmd


def test_wan2_1_gets_isolated_cache_dir():
    # Wan 2.2 and Wan 2.1 share staged compiled-dir names (wan.py
    # _stage_compiled_dir has no model version), so wan2_1 must be isolated.
    compile_cmd = build_compile_cmd(MODELS["wan2_1"], PARALLEL_CONFIGS["tp4"])
    gen_cmd, _ = build_generate_cmd(MODELS["wan2_1"], PARALLEL_CONFIGS["tp4"],
                                    pathlib.Path("/tmp/cell"))
    for cmd in (compile_cmd, gen_cmd):
        assert "--cache-dir" in cmd
        assert "wan2_1" in cmd[cmd.index("--cache-dir") + 1]
    assert "--cache-dir" not in build_compile_cmd(MODELS["wan"], PARALLEL_CONFIGS["tp4"])


def test_hunyuan_video_15_compile_cmd_minimal():
    cmd = build_compile_cmd(MODELS["hunyuan_video_15"], PARALLEL_CONFIGS["tp4"])
    assert cmd[: len(_DIFFLET_CMD) + 3] == _DIFFLET_CMD + [
        "compile", "--model-id",
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"]
    assert "--height" not in cmd  # scaffold fails before shape matters


def test_build_generate_cmd_non_staged():
    cell_dir = pathlib.Path("/tmp/cell")
    cmd, artifacts = build_generate_cmd(MODELS["flux"], PARALLEL_CONFIGS["tp4"], cell_dir)
    assert "--prompt" in cmd
    assert "--work-dir" not in cmd and "--keep-work-dir" not in cmd
    assert cmd[cmd.index("--output") + 1] == str(cell_dir / "flux.png")
    assert artifacts == [[cell_dir / "flux.png"]]


def test_build_generate_cmd_staged_gets_work_dir():
    cell_dir = pathlib.Path("/tmp/cell")
    cmd, artifacts = build_generate_cmd(MODELS["wan"], PARALLEL_CONFIGS["tp4"], cell_dir)
    assert cmd[cmd.index("--work-dir") + 1] == str(cell_dir / "work")
    assert "--keep-work-dir" in cmd
    # mp4 export can fall back to a .pt tensor; both prove inference ran
    assert artifacts == [[cell_dir / "wan.mp4", cell_dir / "wan.pt"]]


def test_build_generate_cmd_ltx_2_accepts_mp4_or_pt():
    # LTX2Orchestrator.generate now exports MP4 with a .pt fallback
    cell_dir = pathlib.Path("/tmp/cell")
    cmd, artifacts = build_generate_cmd(MODELS["ltx_2"], PARALLEL_CONFIGS["tp4"], cell_dir)
    assert cmd[cmd.index("--output") + 1] == str(cell_dir / "ltx2.mp4")
    assert artifacts == [[cell_dir / "ltx2.mp4", cell_dir / "ltx2.pt"]]


def test_build_generate_cmd_qwen_accepts_png_or_pt():
    cell_dir = pathlib.Path("/tmp/cell")
    _, artifacts = build_generate_cmd(MODELS["qwen_image"], PARALLEL_CONFIGS["tp4"], cell_dir)
    assert artifacts == [[cell_dir / "qwen.png", cell_dir / "qwen.pt"]]


def test_build_generate_cmd_dp_batches_requests(tmp_path):
    import json as _json

    cmd, artifacts = build_generate_cmd(MODELS["wan"], PARALLEL_CONFIGS["dp2tp2"], tmp_path)
    assert "--requests" in cmd and "--prompt" not in cmd and "--output" not in cmd
    assert "--dp" in cmd and cmd[cmd.index("--dp") + 1] == "2"
    assert cmd[cmd.index("--work-dir") + 1] == str(tmp_path / "work")
    lines = [_json.loads(l) for l in
             (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["seed"] == 42 and lines[1]["seed"] == 43
    assert lines[0]["output"] != lines[1]["output"]
    # one artifact group per request, each with an mp4->pt fallback
    assert artifacts == [
        [tmp_path / "wan_dp0.mp4", tmp_path / "wan_dp0.pt"],
        [tmp_path / "wan_dp1.mp4", tmp_path / "wan_dp1.pt"],
    ]



def _as_popen(fake_run):
    """Adapt a ``fake_run(cmd, **kw) -> MagicMock(returncode=N)`` stub to the
    ``subprocess.Popen`` seam ``run_step`` uses (``.wait()`` then ``.returncode``).
    Every run_cell test must go through this: an unpatched Popen launches the
    real ``difflet`` CLI on the host."""
    def fake_popen(cmd, **kwargs):
        proc = fake_run(cmd, **kwargs)
        proc.wait = MagicMock(return_value=proc.returncode)
        proc.pid = os.getpid()
        return proc
    return fake_popen


def test_run_cell_dp_requires_every_request_artifact(tmp_path):
    spec, cfg = MODELS["flux"], PARALLEL_CONFIGS["dp2tp2"]

    def fake_run(cmd, **kwargs):
        if "generate" in cmd:
            # only request 0 produced an output
            (tmp_path / "flux" / "dp2tp2" / "flux_dp0.png").touch()
        return MagicMock(returncode=0)

    with patch("subprocess.Popen", side_effect=_as_popen(fake_run)):
        result = run_cell(spec, cfg, tmp_path / "flux" / "dp2tp2", timeout=60)
    assert result["generate"].status == Status.FAIL
    assert "flux_dp1" in result["generate"].reason


# ---------------------------------------------------------------- run_step

def _log():
    return io.StringIO()


def _file_log(tmp_path):
    # run_step hands the log to Popen as stdout, so it must be a real file.
    return open(tmp_path / "step.log", "w")


def test_run_step_pass_records_duration(tmp_path):
    result = run_step([sys.executable, "-c", "pass"], _file_log(tmp_path), timeout=60)
    assert result.status == Status.PASS
    assert result.duration is not None and result.duration >= 0.0


def test_run_step_fail_nonzero_exit(tmp_path):
    result = run_step([sys.executable, "-c", "raise SystemExit(2)"], _file_log(tmp_path), timeout=60)
    assert result.status == Status.FAIL
    assert "exit code 2" in result.reason


def test_run_step_timeout_is_fail(tmp_path):
    result = run_step([sys.executable, "-c", "import time; time.sleep(30)"], _file_log(tmp_path), timeout=0.5)
    assert result.status == Status.FAIL
    assert "timeout" in result.reason


def test_run_step_timeout_kills_the_whole_process_tree(tmp_path):
    # Device evidence (ltx_2/dp2tp2, 2026-09-07): the driver's timeout killed
    # `difflet generate` (the DP router) but its two worker children survived
    # and held all four NeuronCores. A grandchild must not outlive a timeout.
    pidfile = tmp_path / "grandchild.pid"
    cmd = ["bash", "-c", f"sleep 300 & echo $! > {pidfile}; wait"]
    result = run_step(cmd, _file_log(tmp_path), timeout=1)
    assert result.status == Status.FAIL and "timeout" in result.reason
    pid = int(pidfile.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("grandchild sleep outlived the step timeout")


# ---------------------------------------------------------------- run_cell

def test_run_cell_pass(tmp_path):
    spec, cfg = MODELS["flux"], PARALLEL_CONFIGS["tp4"]

    def fake_run(cmd, **kwargs):
        if "generate" in cmd:
            out = cmd[cmd.index("--output") + 1]
            pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(out).touch()
        return MagicMock(returncode=0)

    with patch("subprocess.Popen", side_effect=_as_popen(fake_run)):
        result = run_cell(spec, cfg, tmp_path / "flux" / "tp4", timeout=60)
    assert result["compile"].status == Status.PASS
    assert result["generate"].status == Status.PASS
    assert result["generate"].duration is not None


def test_run_cell_compile_fail_skips_generate(tmp_path):
    spec, cfg = MODELS["flux"], PARALLEL_CONFIGS["tp4"]

    def fake_run(cmd, **kwargs):
        return MagicMock(returncode=1 if "compile" in cmd else 0)

    with patch("subprocess.Popen", side_effect=_as_popen(fake_run)):
        result = run_cell(spec, cfg, tmp_path / "flux" / "tp4", timeout=60)
    assert result["compile"].status == Status.FAIL
    assert result["generate"].status == Status.SKIP


def test_run_cell_generate_fail_when_artifact_missing(tmp_path):
    spec, cfg = MODELS["flux"], PARALLEL_CONFIGS["tp4"]
    with patch("subprocess.Popen", side_effect=_as_popen(lambda cmd, **kw: MagicMock(returncode=0))):
        # exit 0, no file written
        result = run_cell(spec, cfg, tmp_path / "flux" / "tp4", timeout=60)
    assert result["generate"].status == Status.FAIL
    assert "artifact" in result["generate"].reason


def test_run_cell_accepts_fallback_artifact(tmp_path):
    spec, cfg = MODELS["wan"], PARALLEL_CONFIGS["tp4"]

    def fake_run(cmd, **kwargs):
        if "generate" in cmd:
            out = pathlib.Path(cmd[cmd.index("--output") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.with_suffix(".pt").touch()  # mp4 export failed, .pt fallback
        return MagicMock(returncode=0)

    with patch("subprocess.Popen", side_effect=_as_popen(fake_run)):
        result = run_cell(spec, cfg, tmp_path / "wan" / "tp4", timeout=60)
    assert result["generate"].status == Status.PASS


# ---------------------------------------------------------------- outcomes / exit

def _step(status, duration=1.0):
    return StepResult(status=status, duration=duration)


def test_cell_outcome_pass():
    assert cell_outcome(_step(Status.PASS), _step(Status.PASS), expected_fail=False) == Status.PASS


def test_cell_outcome_fail():
    assert cell_outcome(_step(Status.FAIL), _step(Status.SKIP), expected_fail=False) == Status.FAIL


def test_cell_outcome_xfail():
    assert cell_outcome(_step(Status.FAIL), _step(Status.SKIP), expected_fail=True) == Status.XFAIL


def test_cell_outcome_xpass():
    assert cell_outcome(_step(Status.PASS), _step(Status.PASS), expected_fail=True) == Status.XPASS


def test_exit_code_zero_with_xfail_and_skips():
    assert compute_exit_code([Status.PASS, Status.XFAIL, Status.SKIP]) == 0


def test_exit_code_one_on_fail():
    assert compute_exit_code([Status.PASS, Status.FAIL]) == 1


def test_exit_code_one_on_xpass():
    assert compute_exit_code([Status.PASS, Status.XPASS]) == 1


# ---------------------------------------------------------------- summary

def _fake_results():
    """Minimal results structure for two models / two configs."""
    return {
        "downloads": {
            "flux": StepResult(status=Status.PASS, duration=120.0),
            "hunyuan_video_15": StepResult(status=Status.PASS, duration=60.0),
        },
        "cells": {
            ("flux", "tp4"): {
                "outcome": Status.PASS,
                "compile": StepResult(status=Status.PASS, duration=600.0),
                "generate": StepResult(status=Status.PASS, duration=41.2),
            },
            ("flux", "tp2cfg"): {"outcome": Status.SKIP, "reason": "distilled"},
            ("hunyuan_video_15", "tp4"): {
                "outcome": Status.XFAIL,
                "compile": StepResult(status=Status.FAIL, duration=2.0,
                                      reason="exit code 1",
                                      cmd=["difflet", "compile", "--model-id", "x"]),
                "generate": StepResult(status=Status.SKIP),
            },
        },
    }


def test_format_summary_matrix_cells():
    text = format_summary(_fake_results(), ["flux", "hunyuan_video_15"], ["tp4", "tp2cfg"],
                          "/tmp/run")
    assert "PASS" in text and "41.2" in text          # timed generate is the headline
    assert "SKIP" in text and "distilled" in text     # skip reason inline
    assert "XFAIL" in text                            # known gap visible


def test_format_summary_lists_unexpected_failures_only():
    results = _fake_results()
    text = format_summary(results, ["flux", "hunyuan_video_15"], ["tp4", "tp2cfg"], "/tmp/run")
    assert "FAILED" not in text  # XFAIL is expected: no failure section
    results["cells"][("flux", "tp4")]["outcome"] = Status.FAIL
    results["cells"][("flux", "tp4")]["generate"] = StepResult(
        status=Status.FAIL, reason="artifact not found", cmd=["difflet", "generate"])
    text = format_summary(results, ["flux", "hunyuan_video_15"], ["tp4", "tp2cfg"], "/tmp/run")
    assert "FAILED" in text and "artifact not found" in text


# ---------------------------------------------------------------- fidelity settings

def test_generate_cmds_carry_fidelity_flags():
    """Typical sample-grade settings so outputs are prompt-faithful and human-verifiable."""
    expected = {
        "flux": ("--steps", "28"),
        "qwen_image": ("--steps", "50"),
        "ltx_2": ("--steps", "40"),
        "wan": ("--steps", "40", "--guidance-scale", "4.0"),
        "wan2_1": ("--steps", "40", "--guidance-scale", "4.0"),
        "hunyuan_video": ("--steps", "50"),
    }
    for key, flags in expected.items():
        cmd, _ = build_generate_cmd(MODELS[key], PARALLEL_CONFIGS["tp4"],
                                    pathlib.Path("/tmp/cell"))
        assert cmd[cmd.index("--steps") + 1] == flags[flags.index("--steps") + 1], key
        if "--guidance-scale" in flags:
            assert cmd[cmd.index("--guidance-scale") + 1] == \
                flags[flags.index("--guidance-scale") + 1], key


def test_compile_cmds_have_no_generate_only_flags():
    for key in ("flux", "wan", "hunyuan_video"):
        cmd = build_compile_cmd(MODELS[key], PARALLEL_CONFIGS["tp4"])
        assert "--steps" not in cmd and "--guidance-scale" not in cmd, key


def test_video_models_use_5s_clip_shapes():
    """ltx_2/hunyuan keep 5 s clips; wan is pinned to 9 frames — device
    attention fidelity collapses on long sequences (~33k tokens at f81)."""
    expected = {"wan": "9", "wan2_1": "9", "ltx_2": "121", "hunyuan_video": "121"}
    for key, frames in expected.items():
        flags = MODELS[key].shape_flags
        assert flags[flags.index("--num-frames") + 1] == frames, key


def test_wan_models_all_device_at_f9():
    """At 9 frames the Neuron VAE compiles fine (and its dirs are warm), so
    wan cells stay all-device; --host-vae remains available for long clips."""
    for key in ("wan", "wan2_1"):
        assert "--host-vae" not in build_compile_cmd(MODELS[key], PARALLEL_CONFIGS["tp4"])
