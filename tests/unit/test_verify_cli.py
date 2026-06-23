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

    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    import builtins
    real_open = builtins.open

    def fake_open(path, mode="r", **kw):
        if "verify_cli_" in str(path):
            return real_open(str(log_dir / "verify_cli_test.log"), mode, **kw)
        return real_open(path, mode, **kw)

    monkeypatch.setattr("verify_cli._run_model", fake_run_model)
    monkeypatch.setattr("verify_cli.pathlib.Path.mkdir", lambda *a, **kw: None)
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

    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    import builtins
    real_open = builtins.open

    def fake_open(path, mode="r", **kw):
        if "verify_cli_" in str(path):
            return real_open(str(log_dir / "verify_cli_test.log"), mode, **kw)
        return real_open(path, mode, **kw)

    monkeypatch.setattr("verify_cli._run_model", fake_run_model)
    monkeypatch.setattr("verify_cli.pathlib.Path.mkdir", lambda *a, **kw: None)
    monkeypatch.setattr("builtins.open", fake_open)

    main(["--models", "flux", "ltx_2"])

    assert set(called_models) == {"flux", "ltx_2"}
