from __future__ import annotations

import argparse
import importlib
import os

import pytest

from difflet.cli.serve import (
    _build_serving_logging_config,
    _load_serving_environment,
    options_from_args,
)
from difflet.serving.options import CompilePolicy


@pytest.fixture(autouse=True)
def _clear_difflet_api_key(monkeypatch):
    monkeypatch.delenv("DIFFLET_API_KEY", raising=False)


def test_difflet_serve_routes_to_serving_command(monkeypatch):
    calls = []

    def fake_run(args):
        calls.append(args)

    cli_main = importlib.import_module("difflet.cli.main")
    monkeypatch.setattr("difflet.cli.serve.run", fake_run)
    cli_main.main(["serve", "--model-id", "black-forest-labs/FLUX.1-dev", "--port", "9000"])

    assert calls
    assert calls[0].command == "serve"
    assert calls[0].port == 9000


def test_serve_loads_dotenv_from_current_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DIFFLET_TEST_DOTENV", raising=False)
    (tmp_path / ".env").write_text("DIFFLET_TEST_DOTENV=loaded\n", encoding="utf-8")

    _load_serving_environment()

    assert os.environ["DIFFLET_TEST_DOTENV"] == "loaded"


def test_serve_dotenv_does_not_override_exported_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DIFFLET_TEST_DOTENV", "exported")
    (tmp_path / ".env").write_text("DIFFLET_TEST_DOTENV=file\n", encoding="utf-8")

    _load_serving_environment()

    assert os.environ["DIFFLET_TEST_DOTENV"] == "exported"


def test_serve_ignores_missing_dotenv(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    _load_serving_environment()


def test_serve_console_and_file_logs_include_local_timestamp():
    config = _build_serving_logging_config("Difflet")

    console = config["formatters"]["console"]
    file_formatter = config["formatters"]["file"]
    assert console["fmt"].startswith("%(asctime)s.%(msecs)03d ")
    assert console["datefmt"] == "%Y-%m-%d %H:%M:%S"
    assert file_formatter["format"].startswith("%(asctime)s.%(msecs)03d ")
    assert file_formatter["datefmt"] == "%Y-%m-%d %H:%M:%S"
    assert config["handlers"]["file"]["formatter"] == "file"


def test_serve_help_exposes_operational_tuning_but_hides_artifact_policy(capsys):
    cli_main = importlib.import_module("difflet.cli.main")

    with pytest.raises(SystemExit):
        cli_main.main(["serve", "--help"])

    out = capsys.readouterr().out
    assert "--max-running-requests" not in out
    assert "--worker-cancel-timeout" in out
    assert "--worker-restart-timeout" in out
    assert "--max-queued-requests" in out
    assert "--queue-timeout" in out
    assert "--request-timeout" in out
    assert "--api-key" in out
    assert "--artifact-store-timeout" in out
    assert "--download-policy" not in out
    assert "--compile-policy" not in out
    assert "--artifact-store " not in out
    assert "--artifact-ttl-seconds" not in out
    assert "--cfg-parallel" in out
    assert "--no-cfg-parallel" in out
    assert "--sp" in out
    assert "--no-sp" in out
    assert "--worker-heartbeat-interval" in out
    assert "--validation-workers" in out
    assert "--validation-max-waiting" in out
    assert "--validation-timeout" in out
    assert "--video-retention-seconds" in out
    assert "--video-max-jobs" in out
    assert "--video-sweep-interval" in out
    assert "--host-vae" in out
    assert "--clip-placement" in out
    assert "--vae-placement" not in out
    assert "--num-frames" in out
    assert "--teacache-cadence" in out
    assert "--teacache-online-delta" in out
    assert "--teacache-speedup" in out
    assert "--teacache-calibration" in out
    assert "Wan-AI/Wan2.2-T2V-A14B-Diffusers" not in out
    assert "hunyuanvideo-community/HunyuanVideo-1.5" not in out
    assert "Wan-AI/Wan2.1-T2V-14B-Diffusers" in out
    # Argparse may wrap this long choice between the organization and model.
    assert "community/HunyuanVideo" in out
    assert "Lightricks/LTX-2" in out
    assert "FLUX.1-dev" in out
    assert "Qwen/Qwen-Image" in out


def test_difflet_serve_operational_controls_have_documented_defaults(monkeypatch):
    calls = []
    monkeypatch.setattr("difflet.cli.serve.run", calls.append)
    cli_main = importlib.import_module("difflet.cli.main")

    cli_main.main(["serve", "--model-id", "black-forest-labs/FLUX.1-dev"])

    options = options_from_args(calls[0])
    assert options.max_queued_requests == 8
    assert options.queue_timeout is None
    assert options.effective_queue_timeout("image") == 30.0
    assert options.effective_queue_timeout("video") == 24 * 60 * 60
    assert options.request_timeout == 300.0
    assert options.artifact_store_timeout == 60.0
    assert options.worker_cancel_timeout == 10.0
    assert options.worker_restart_timeout == 900.0
    assert options.artifact_ttl_seconds == 3600
    assert options.validation_workers == 4
    assert options.validation_max_waiting == 32
    assert options.validation_timeout == 30.0
    assert options.video_retention_seconds == 25 * 60 * 60
    assert options.video_max_jobs == 4096
    assert options.video_sweep_interval_seconds == 300.0
    assert options.api_key is None


def test_difflet_serve_maps_optional_api_key_without_exposing_it_in_repr(monkeypatch):
    calls = []
    monkeypatch.setattr("difflet.cli.serve.run", calls.append)
    cli_main = importlib.import_module("difflet.cli.main")

    cli_main.main(
        [
            "serve",
            "--model-id",
            "black-forest-labs/FLUX.1-dev",
            "--api-key",
            "cli-secret",
        ]
    )

    options = options_from_args(calls[0])
    assert options.api_key == "cli-secret"
    assert "cli-secret" not in repr(options)


def test_difflet_serve_uses_environment_api_key_and_cli_takes_precedence(monkeypatch):
    calls = []
    monkeypatch.setenv("DIFFLET_API_KEY", "environment-secret")
    monkeypatch.setattr("difflet.cli.serve.run", calls.append)
    cli_main = importlib.import_module("difflet.cli.main")

    cli_main.main(["serve", "--model-id", "black-forest-labs/FLUX.1-dev"])
    cli_main.main(
        [
            "serve",
            "--model-id",
            "black-forest-labs/FLUX.1-dev",
            "--api-key",
            "cli-secret",
        ]
    )

    assert options_from_args(calls[0]).api_key == "environment-secret"
    assert options_from_args(calls[1]).api_key == "cli-secret"


@pytest.mark.parametrize("value", ["", "two words", " leading", "trailing "])
def test_difflet_serve_rejects_invalid_api_key(value):
    cli_main = importlib.import_module("difflet.cli.main")

    with pytest.raises(SystemExit):
        cli_main.main(
            [
                "serve",
                "--model-id",
                "black-forest-labs/FLUX.1-dev",
                "--api-key",
                value,
            ]
        )


def test_difflet_serve_maps_operational_controls(monkeypatch):
    calls = []
    monkeypatch.setattr("difflet.cli.serve.run", calls.append)
    cli_main = importlib.import_module("difflet.cli.main")

    cli_main.main(
        [
            "serve",
            "--model-id",
            "Qwen/Qwen-Image",
            "--max-queued-requests",
            "0",
            "--queue-timeout",
            "12",
            "--request-timeout",
            "600",
            "--artifact-store-timeout",
            "45",
            "--worker-cancel-timeout",
            "15",
            "--worker-restart-timeout",
            "1200",
            "--validation-workers",
            "2",
            "--validation-max-waiting",
            "7",
            "--validation-timeout",
            "9",
            "--video-retention-seconds",
            "3600",
            "--video-max-jobs",
            "100",
            "--video-sweep-interval",
            "60",
        ]
    )

    options = options_from_args(calls[0])
    assert options.max_queued_requests == 0
    assert options.queue_timeout == 12.0
    assert options.request_timeout == 600.0
    assert options.artifact_store_timeout == 45.0
    assert options.worker_cancel_timeout == 15.0
    assert options.worker_restart_timeout == 1200.0
    assert options.validation_workers == 2
    assert options.validation_max_waiting == 7
    assert options.validation_timeout == 9.0
    assert options.video_retention_seconds == 3600
    assert options.video_max_jobs == 100
    assert options.video_sweep_interval_seconds == 60.0


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--max-queued-requests", "-1"),
        ("--queue-timeout", "0"),
        ("--request-timeout", "nan"),
        ("--artifact-store-timeout", "inf"),
        ("--worker-cancel-timeout", "-1"),
        ("--worker-restart-timeout", "0"),
        ("--validation-workers", "0"),
        ("--validation-max-waiting", "-1"),
        ("--validation-timeout", "0"),
        ("--video-retention-seconds", "0"),
        ("--video-max-jobs", "0"),
        ("--video-sweep-interval", "nan"),
    ],
)
def test_difflet_serve_rejects_invalid_operational_controls(flag, value):
    cli_main = importlib.import_module("difflet.cli.main")

    with pytest.raises(SystemExit):
        cli_main.main(
            [
                "serve",
                "--model-id",
                "black-forest-labs/FLUX.1-dev",
                flag,
                value,
            ]
        )


def test_serve_options_preserves_num_frames_for_adapter_validation():
    args = argparse.Namespace(
        model_id="black-forest-labs/FLUX.1-dev",
        revision=None,
        host="0.0.0.0",
        port=8091,
        tp_degree=None,
        cp_degree=None,
        cp_mode=None,
        cfg_parallel=False,
        sp_enabled=False,
        height=None,
        width=None,
        num_frames=1,
        cache_dir=None,
        force=False,
    )

    options = options_from_args(args)

    assert options.num_frames == 1


def test_difflet_serve_accepts_flux_sp_and_adaptive_teacache(monkeypatch):
    calls = []

    def fake_run(args):
        calls.append(args)

    cli_main = importlib.import_module("difflet.cli.main")
    monkeypatch.setattr("difflet.cli.serve.run", fake_run)
    cli_main.main(
        [
            "serve",
            "--model-id",
            "black-forest-labs/FLUX.1-dev",
            "--sp",
            "--teacache-speedup",
            "1.5",
            "--teacache-calibration",
            "/tmp/flux-calibration.json",
        ]
    )

    assert calls[0].sp_enabled is True
    assert calls[0].teacache_speedup == 1.5
    assert calls[0].teacache_calibration == "/tmp/flux-calibration.json"


def test_difflet_serve_parses_compile_and_load_profile_flags(monkeypatch):
    calls = []

    monkeypatch.setattr("difflet.cli.serve.run", calls.append)
    cli_main = importlib.import_module("difflet.cli.main")
    cli_main.main(
        [
            "serve",
            "--model-id",
            "black-forest-labs/FLUX.1-dev",
            "--revision",
            "rev-1",
            "--tp-degree",
            "4",
            "--cp-degree",
            "2",
            "--cp-mode",
            "ring",
            "--height",
            "768",
            "--width",
            "1024",
            "--cache-dir",
            "/tmp/difflet-cache",
            "--force",
        ]
    )

    args = calls[0]
    assert (args.revision, args.tp_degree, args.cp_degree, args.cp_mode) == ("rev-1", 4, 2, "ring")
    assert (args.height, args.width, args.cache_dir, args.force) == (
        768,
        1024,
        "/tmp/difflet-cache",
        True,
    )


def test_difflet_serve_maps_hunyuan_clip_and_legacy_host_vae_flags(monkeypatch):
    calls = []
    monkeypatch.setattr("difflet.cli.serve.run", calls.append)
    cli_main = importlib.import_module("difflet.cli.main")

    cli_main.main(
        [
            "serve",
            "--model-id",
            "hunyuanvideo-community/HunyuanVideo",
            "--clip-placement",
            "neuron",
            "--host-vae",
        ]
    )

    options = options_from_args(calls[0])
    assert options.clip_placement == "neuron"
    assert options.host_vae is True


def test_serve_options_maps_force_to_compile_policy():
    args = argparse.Namespace(
        model_id="black-forest-labs/FLUX.1-dev",
        revision=None,
        host="0.0.0.0",
        port=8091,
        tp_degree=4,
        cp_degree=1,
        cp_mode="gather_kv",
        height=768,
        width=1024,
        num_frames=None,
        cache_dir="/tmp/difflet-cache",
        force=True,
    )

    options = options_from_args(args)

    assert options.compile_policy == CompilePolicy.FORCE
    assert (options.tp_degree, options.height, options.cache_dir) == (4, 768, "/tmp/difflet-cache")


@pytest.mark.parametrize(
    "flag,value,field,expected",
    [
        ("--teacache-cadence", "2", "teacache_cadence", 2),
        ("--teacache-online-delta", "0.6", "teacache_online_delta", 0.6),
    ],
)
def test_difflet_serve_forwards_probe_free_teacache_modes_to_options(
    monkeypatch,
    flag,
    value,
    field,
    expected,
):
    """The CLI only parses; model-type support is decided by build_serving_profile."""
    calls = []
    monkeypatch.setattr("difflet.cli.serve.run", calls.append)
    cli_main = importlib.import_module("difflet.cli.main")

    cli_main.main(
        [
            "serve",
            "--model-id",
            "Qwen/Qwen-Image",
            flag,
            value,
        ]
    )

    assert len(calls) == 1
    assert getattr(calls[0], field) == expected


def test_difflet_serve_preserves_calibration_without_speedup_for_adapter(monkeypatch):
    calls = []
    monkeypatch.setattr("difflet.cli.serve.run", calls.append)
    cli_main = importlib.import_module("difflet.cli.main")

    cli_main.main(
        [
            "serve",
            "--model-id",
            "Qwen/Qwen-Image",
            "--teacache-calibration",
            "/tmp/qwen-calibration.json",
        ]
    )

    assert calls[0].teacache_calibration == "/tmp/qwen-calibration.json"


def test_difflet_serve_preserves_qwen_sp_for_adapter(monkeypatch):
    calls = []
    cli_main = importlib.import_module("difflet.cli.main")
    monkeypatch.setattr("difflet.cli.serve.run", calls.append)

    cli_main.main(["serve", "--model-id", "Qwen/Qwen-Image", "--sp"])

    assert calls[0].sp_enabled is True


def test_serve_boolean_overrides_are_tristate(monkeypatch):
    calls = []
    cli_main = importlib.import_module("difflet.cli.main")
    monkeypatch.setattr("difflet.cli.serve.run", calls.append)

    cli_main.main(["serve", "--model-id", "black-forest-labs/FLUX.1-dev"])
    cli_main.main(
        [
            "serve",
            "--model-id",
            "black-forest-labs/FLUX.1-dev",
            "--no-cfg-parallel",
            "--no-sp",
        ]
    )

    assert (calls[0].cfg_parallel, calls[0].sp_enabled) == (None, None)
    assert (calls[1].cfg_parallel, calls[1].sp_enabled) == (False, False)


@pytest.mark.parametrize("interval", ["0", "4.99", "120.01", "nan", "inf", "-inf"])
def test_serve_rejects_invalid_heartbeat_before_run(monkeypatch, capsys, interval):
    calls = []
    cli_main = importlib.import_module("difflet.cli.main")
    monkeypatch.setattr("difflet.cli.serve.run", calls.append)

    with pytest.raises(SystemExit):
        cli_main.main(
            [
                "serve",
                "--model-id",
                "black-forest-labs/FLUX.1-dev",
                f"--worker-heartbeat-interval={interval}",
            ]
        )

    assert calls == []
    assert "between 5 and 120 seconds inclusive" in capsys.readouterr().err


@pytest.mark.parametrize("interval", ["5", "120"])
def test_serve_accepts_heartbeat_boundaries(monkeypatch, interval):
    calls = []
    cli_main = importlib.import_module("difflet.cli.main")
    monkeypatch.setattr("difflet.cli.serve.run", calls.append)

    cli_main.main(
        [
            "serve",
            "--model-id",
            "black-forest-labs/FLUX.1-dev",
            f"--worker-heartbeat-interval={interval}",
        ]
    )

    assert calls[0].worker_heartbeat_interval == float(interval)


def test_serve_options_maps_shapes_csv_for_multi_shape_serving():
    args = argparse.Namespace(
        model_id="black-forest-labs/FLUX.1-dev",
        revision=None,
        host="0.0.0.0",
        port=8091,
        tp_degree=None,
        cp_degree=None,
        cp_mode=None,
        cfg_parallel=False,
        sp_enabled=False,
        height=None,
        width=None,
        num_frames=None,
        shapes="1024x1024,512x512",
        cache_dir=None,
        force=False,
    )

    options = options_from_args(args)

    assert options.shapes == "1024x1024,512x512"
