"""C10: torchrun launch path and eager/compile selection for MPMD backends.

CPU only. The launch itself is exercised for real on CPU (torchrun + gloo, cpu
pipeline backend); the neuron-device run of the same path is
tests/manual/check_neuron_launch_c10.py.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

# As in test_neuron_application_c8.py: importorskip keeps torch_neuronx's import-time nki.jit
# DeprecationWarning (torch autoloads torch_neuronx) out of the report when this module is
# collected first. The CLI modules under test stay torch-free; a subprocess test checks that.
torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[3]
# fp32 TP-N vs the TP1 reference differs by reduction order only.
FP32_TOL = {"atol": 1e-4, "rtol": 1e-4}
WAN_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"  # registry backends: trainium, tpu


def _isolate_env(monkeypatch, *names):
    """Unset ``names`` now and restore them after the test, even when the code under
    test assigns os.environ itself (delenv of an absent name records nothing to undo)."""
    for name in names:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def _pythonpath() -> str:
    return os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))


def _cli_main():
    # difflet/cli/__init__.py re-exports the function ``main``, so
    # ``import difflet.cli.main as m`` would bind the function, not the module.
    return importlib.import_module("difflet.cli.main")


# ------------------------------------------------------------------ registry


def test_registry_available_backends_sorted():
    from difflet.backends.registry import available_backends

    assert available_backends() == ("cpu", "cuda", "neuron", "rocm", "tpu", "trainium")


# ------------------------------------------------------------ DiffletPipeline


class _RecordingApp:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.compile_calls: list[str] = []
        self.load_calls: list[dict] = []

    def compile(self, compiled_model_path, debug=False):
        self.compile_calls.append(compiled_model_path)
        Path(compiled_model_path).mkdir(parents=True, exist_ok=True)

    def has_compiled_artifacts(self, compiled_model_path):
        return True

    def load(
        self, compiled_model_path, start_rank_id=None, local_ranks_size=None, skip_warmup=False
    ):
        self.load_calls.append({"path": compiled_model_path, "skip_warmup": skip_warmup})


def _dummy_pipeline(tmp_path, backend, **kwargs):
    from difflet.pipeline.difflet_pipeline import DiffletPipeline
    from difflet.pipeline.parallel_config import DiffletParallelConfig
    from difflet.registry import register_model

    @register_model(
        name="c10_launch_dummy",
        application_factory=lambda **app_kwargs: _RecordingApp(**app_kwargs),
        backends=("trainium", "cpu", "neuron"),
        default_shape={"height": 8, "width": 8, "num_frames": None},
    )
    class _Registration:
        pass

    model_dir = tmp_path / "c10-dummy-model"
    model_dir.mkdir(exist_ok=True)
    return DiffletPipeline.from_pretrained(
        str(model_dir),
        model_type="c10_launch_dummy",
        parallel=DiffletParallelConfig(tp_degree=1),
        dtype="bf16",
        compile_cache_dir=str(tmp_path / "cache"),
        backend=backend,
        **kwargs,
    )


@pytest.mark.parametrize("force_compile", [False, True])
def test_pipeline_non_aot_backend_loads_without_compile_or_manifest(tmp_path, force_compile):
    from difflet.pipeline.compile_cache import cache_path

    pipe = _dummy_pipeline(tmp_path, "cpu", force_compile=force_compile)

    assert pipe.app.compile_calls == []
    assert len(pipe.app.load_calls) == 1
    assert not pipe.compiled_path.exists()
    # The cache location is computed exactly as before; only the AoT step is skipped.
    assert pipe.compiled_path == cache_path(str(tmp_path / "cache"), pipe.cache_spec)


def test_pipeline_non_aot_compile_is_a_noop(tmp_path):
    pipe = _dummy_pipeline(tmp_path, "cpu", load=False)

    pipe.compile(force=True, debug=True)

    assert pipe.app.compile_calls == []
    assert pipe.app.load_calls == []
    assert not pipe.compiled_path.exists()


def test_pipeline_aot_backend_still_compiles_then_loads(tmp_path):
    pipe = _dummy_pipeline(tmp_path, "trainium")

    assert pipe.app.compile_calls == [str(pipe.compiled_path)]
    assert (pipe.compiled_path / "manifest.json").exists()
    assert len(pipe.app.load_calls) == 1
    # Trainium keys carry no backend entry (compile_cache.py:93-97): byte-identical.
    assert "backend" not in pipe.cache_spec.cache_inputs()


def test_pipeline_neuron_single_rank_skips_compile(tmp_path, monkeypatch):
    from difflet.backends.neuron import runtime

    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    # prepare_runtime -> configure_compile_cache (C9) writes these; keep them in
    # tmp_path and restore them afterwards.
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path / "difflet-cache"))
    monkeypatch.setenv("TORCH_NEURONX_NEFF_CACHE_DIR", str(tmp_path / "neff"))
    monkeypatch.setenv("TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR", str(tmp_path / "neff_local"))
    monkeypatch.setenv("NKI_ENABLE_TRACE_CACHE", "0")
    monkeypatch.setattr(runtime, "_runtime_prepared", False)  # restored after the call sets it

    pipe = _dummy_pipeline(tmp_path, "neuron")

    assert pipe.backend.name == "neuron"
    assert pipe.app.compile_calls == []
    assert len(pipe.app.load_calls) == 1
    assert pipe.cache_spec.cache_inputs()["backend"] == "neuron"


def test_pipeline_requires_aot_defaults_to_true():
    from difflet.pipeline.difflet_pipeline import _requires_aot

    assert _requires_aot(SimpleNamespace()) is True
    no_aot = SimpleNamespace(capabilities=SimpleNamespace(requires_aot=False))
    assert _requires_aot(no_aot) is False


@pytest.mark.parametrize("name", ["cpu", "cuda", "neuron", "rocm", "tpu", "trainium"])
def test_pipeline_requires_aot_follows_backend_capabilities(name):
    """Only trainium and tpu build an AoT artifact; the branch also covers cpu, cuda, rocm."""
    from difflet.backends import get_backend
    from difflet.pipeline.difflet_pipeline import _requires_aot

    assert _requires_aot(get_backend(name)) is (name in ("trainium", "tpu"))


# ------------------------------------------------------------------- prewarm


@pytest.fixture
def prewarm_threads(monkeypatch):
    """Swap prewarm's thread for a recorder whose target never runs, so nothing can
    touch the device even before the gate exists."""
    import difflet.cli.prewarm as prewarm

    started: list[str] = []

    class _Recorder:
        def __init__(self, *, target, name, daemon):
            self.name = name

        def start(self):
            started.append(self.name)

    monkeypatch.setattr(prewarm, "threading", SimpleNamespace(Thread=_Recorder))
    monkeypatch.delenv("DIFFLET_DISABLE_PREWARM", raising=False)
    return started


def test_prewarm_skipped_for_non_aot_backend(monkeypatch, prewarm_threads):
    from difflet.cli.prewarm import prewarm_neuron_runtime

    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")

    assert prewarm_neuron_runtime(4) is None
    assert prewarm_threads == []


def test_prewarm_still_starts_for_aot_backend(monkeypatch, prewarm_threads):
    from difflet.cli.prewarm import prewarm_neuron_runtime

    monkeypatch.setenv("DIFFLET_BACKEND", "trainium")

    assert prewarm_neuron_runtime(4) is not None
    assert prewarm_threads == ["neuron-prewarm"]


def test_prewarm_lookup_error_keeps_old_behaviour(monkeypatch, prewarm_threads):
    from difflet.cli.prewarm import prewarm_neuron_runtime

    monkeypatch.setenv("DIFFLET_BACKEND", "no-such-backend")

    assert prewarm_neuron_runtime(1) is not None
    assert prewarm_threads == ["neuron-prewarm"]


@pytest.mark.parametrize("detected, starts", [("neuron", False), ("trainium", True)])
def test_prewarm_follows_auto_detected_backend(monkeypatch, prewarm_threads, detected, starts):
    """Without DIFFLET_BACKEND the gate follows auto-detection, which picks neuron on a
    TorchNeuron host: the flux / ltx_2 orchestrators call prewarm in-process too."""
    from difflet.backends import registry
    from difflet.cli.prewarm import prewarm_neuron_runtime

    monkeypatch.delenv("DIFFLET_BACKEND", raising=False)
    monkeypatch.setattr(registry, "_auto_detect_backend", lambda: detected)

    assert (prewarm_neuron_runtime(4) is not None) is starts
    assert prewarm_threads == (["neuron-prewarm"] if starts else [])


# --------------------------------------------------------------------- stage


def test_stage_parser_backend_flags_default_to_none():
    import difflet.cli.stage as stage

    args, extra = stage._build_stage_parser().parse_known_args(
        ["--orchestrator", "X", "--stage", "toy"]
    )
    assert (args.backend, args.exec_mode, extra) == (None, None, [])


@pytest.mark.parametrize("flag, value", [("--backend", "gpu"), ("--exec-mode", "jit")])
def test_stage_parser_rejects_unknown_choices(flag, value):
    import difflet.cli.stage as stage

    with pytest.raises(SystemExit):
        stage._build_stage_parser().parse_known_args(
            ["--orchestrator", "X", "--stage", "toy", flag, value]
        )


def test_stage_exec_mode_choices_track_neuron_compile():
    from difflet.backends.neuron.compile import EXEC_MODES
    from difflet.cli.stage import EXEC_MODE_CHOICES

    assert EXEC_MODE_CHOICES == EXEC_MODES


def test_stage_main_exports_selection_before_loading(monkeypatch, prewarm_threads):
    import difflet.cli.stage as stage

    _isolate_env(monkeypatch, "DIFFLET_BACKEND", "DIFFLET_EXEC_MODE")
    seen = {}

    class _Orch:
        def __init__(self, args):
            pass

        def _run_stage_internal(self, stage_name, args):
            seen["stage"] = stage_name
            seen["exec_mode"] = os.environ.get("DIFFLET_EXEC_MODE")

    def _load(name):
        seen["backend_at_load"] = os.environ.get("DIFFLET_BACKEND")
        return _Orch

    monkeypatch.setattr(stage, "_load_orchestrator_class", _load)

    rc = stage.main([
        "--orchestrator", "pkg.mod:Orch", "--stage", "toy",
        "--backend", "neuron", "--exec-mode", "eager",
    ])

    assert rc == 0
    assert seen == {"backend_at_load": "neuron", "stage": "toy", "exec_mode": "eager"}
    # A generate-mode stage on a non-AoT backend must not prewarm (stage.py:84-92).
    assert prewarm_threads == []


def test_stage_loads_module_colon_class():
    import difflet.cli.stage as stage
    from difflet.cli.orchestrators.wan import WanOrchestrator

    loaded = stage._load_orchestrator_class("difflet.cli.orchestrators.wan:WanOrchestrator")
    assert loaded is WanOrchestrator


@pytest.mark.parametrize("name", [":Orch", "difflet.cli.orchestrators.wan:"])
def test_stage_rejects_malformed_reference(name):
    import difflet.cli.stage as stage

    with pytest.raises(SystemExit, match="invalid orchestrator reference"):
        stage._load_orchestrator_class(name)


# -------------------------------------------------------------------- runner


@pytest.fixture
def captured_run(monkeypatch):
    """Record the launch instead of spawning it: ``api`` is "run" for the single-process
    ``subprocess.run(cmd, env=env, check=True)``, "popen" for the torchrun branch."""
    captured: dict = {}

    def fake_run(cmd, env, check):
        captured.update(api="run", cmd=cmd, env=env, check=check)

    class FakePopen:
        def __init__(self, cmd, env):
            captured.update(api="popen", cmd=cmd, env=env)

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("subprocess.Popen", FakePopen)
    return captured


def test_runner_command_single_process_is_legacy():
    from difflet.cli.runner import build_stage_command

    assert build_stage_command("Org/Model", "transformer", ["--tp-degree", "4"]) == [
        sys.executable, "-m", "difflet.cli.stage",
        "--orchestrator", "Org/Model", "--stage", "transformer", "--tp-degree", "4",
    ]


def test_runner_command_torchrun_launches_stage_module():
    from difflet.cli.runner import build_stage_command

    assert build_stage_command("pkg.mod:Orch", "toy", ["--exec-mode", "eager"], nproc=4) == [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "4",
        "-m", "difflet.cli.stage", "--orchestrator", "pkg.mod:Orch", "--stage", "toy",
        "--exec-mode", "eager",
    ]


def test_runner_command_rejects_empty_launch():
    from difflet.cli.runner import build_stage_command

    with pytest.raises(ValueError, match="nproc"):
        build_stage_command("o", "s", [], nproc=0)


def test_runner_explicit_neuron_backend_uses_torchrun(monkeypatch, captured_run):
    from difflet.cli.runner import build_stage_command, run_stage

    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    for name, value in (
        ("WORLD_SIZE", "8"), ("RANK", "3"), ("LOCAL_RANK", "3"), ("LOCAL_WORLD_SIZE", "8"),
    ):
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("NEURON_RT_NUM_CORES", raising=False)
    monkeypatch.delenv("NEURON_RT_VIRTUAL_CORE_SIZE", raising=False)

    run_stage(
        "pkg.mod:Orch", "toy", num_cores=4, virtual_core_size=2,
        cli_args=["--exec-mode", "compile"],
    )

    assert captured_run["api"] == "popen"  # _run_torchrun, not subprocess.run
    assert captured_run["cmd"] == build_stage_command(
        "pkg.mod:Orch", "toy", ["--exec-mode", "compile"], nproc=4
    )
    env = captured_run["env"]
    for name in (
        "WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
        "NEURON_RT_NUM_CORES", "NEURON_RT_VIRTUAL_CORE_SIZE",
    ):
        assert name not in env, name
    assert env["DIFFLET_BACKEND"] == "neuron"


def test_runner_mpmd_strict_environment_pins_visible_cores(monkeypatch, captured_run):
    from difflet.cli.runner import run_stage

    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "4-9")
    monkeypatch.setenv("NEURON_RT_NUM_CORES", "8")

    run_stage(
        "pkg.mod:Orch", "toy", num_cores=4, virtual_core_size=None, cli_args=[],
        strict_environment=True,
    )

    env = captured_run["env"]
    assert env["NEURON_RT_VISIBLE_CORES"] == "4,5,6,7"
    assert "NEURON_RT_NUM_CORES" not in env
    assert captured_run["cmd"][1:3] == ["-m", "torch.distributed.run"]
    assert captured_run["api"] == "popen"


def test_runner_single_process_without_explicit_backend(monkeypatch, captured_run):
    # Auto-detection already resolves to neuron on this host (registry.py:55-61); the
    # torchrun launch is opt-in through DIFFLET_BACKEND / --backend only.
    from difflet.cli.runner import run_stage

    monkeypatch.delenv("DIFFLET_BACKEND", raising=False)

    run_stage("Org/Model", "transformer", num_cores=4, virtual_core_size=None, cli_args=[])

    assert captured_run["cmd"][:3] == [sys.executable, "-m", "difflet.cli.stage"]
    assert (captured_run["api"], captured_run["check"]) == ("run", True)


@pytest.mark.parametrize("backend", ["trainium", "tpu", "cpu"])
def test_runner_single_process_for_aot_or_non_mpmd_backends(monkeypatch, captured_run, backend):
    from difflet.cli.runner import run_stage

    monkeypatch.setenv("DIFFLET_BACKEND", backend)
    monkeypatch.setenv("WORLD_SIZE", "8")

    run_stage("Org/Model", "transformer", num_cores=4, virtual_core_size=None, cli_args=[])

    assert captured_run["cmd"][:3] == [sys.executable, "-m", "difflet.cli.stage"]
    assert captured_run["env"]["WORLD_SIZE"] == "8"  # legacy non-strict path: untouched
    # still subprocess.run(cmd, env=env, check=True): Ctrl-C there is unchanged
    assert (captured_run["api"], captured_run["check"]) == ("run", True)


def test_runner_rejects_unknown_backend(monkeypatch, captured_run):
    from difflet.cli.runner import run_stage

    monkeypatch.setenv("DIFFLET_BACKEND", "no-such-backend")

    with pytest.raises(ValueError, match="unknown Difflet backend"):
        run_stage("Org/Model", "transformer", num_cores=4, virtual_core_size=None, cli_args=[])
    assert captured_run == {}


class _ScriptedTorchrun:
    """subprocess.Popen stand-in for the torchrun branch. ``waits`` scripts each wait() call
    in turn: an exception (class or instance) is raised, anything else is the return code.
    ``calls`` records wait(timeout), kill, terminate and send_signal in order."""

    def __init__(self, waits):
        self.waits = list(waits)
        self.calls: list[tuple] = []
        self.pid = 4242

    def __call__(self, cmd, env):  # used as the Popen class: "constructs" itself
        self.cmd = cmd
        return self

    def wait(self, timeout=None):
        self.calls.append(("wait", timeout))
        outcome = self.waits.pop(0)
        if isinstance(outcome, BaseException) or (
            isinstance(outcome, type) and issubclass(outcome, BaseException)
        ):
            raise outcome
        return outcome

    def kill(self):
        self.calls.append(("kill",))

    def terminate(self):
        self.calls.append(("terminate",))

    def send_signal(self, sig):
        self.calls.append(("send_signal", sig))


def _run_scripted_torchrun(monkeypatch, waits, *, shutdown_timeout=None):
    """run_stage's torchrun branch against _ScriptedTorchrun; returns (fake, raised)."""
    from difflet.cli.runner import run_stage

    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    if shutdown_timeout is None:
        monkeypatch.delenv("TORCH_ELASTIC_SHUTDOWN_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("TORCH_ELASTIC_SHUTDOWN_TIMEOUT", shutdown_timeout)
    fake = _ScriptedTorchrun(waits)
    used_run = []
    monkeypatch.setattr(subprocess, "Popen", fake)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: used_run.append(a))
    raised = None
    try:
        run_stage("pkg.mod:Orch", "toy", num_cores=2, virtual_core_size=None, cli_args=[])
    except BaseException as exc:  # noqa: BLE001 - the scripted KeyboardInterrupt included
        raised = exc
    assert used_run == [], "the torchrun branch went through subprocess.run"
    assert fake.waits == [], f"scripted wait() outcomes left unused: {fake.waits}"
    return fake, raised


def _timeouts(fake) -> list:
    return [call[1] for call in fake.calls if call[0] == "wait"]


def test_runner_torchrun_ctrl_c_waits_for_torchrun_and_reraises(monkeypatch):
    """Ctrl-C reached torchrun too (same process group): wait for it to stop its ranks, never
    kill or signal it, then re-raise. subprocess.run would SIGKILL it after 0.25 s."""
    fake, raised = _run_scripted_torchrun(monkeypatch, [KeyboardInterrupt, 0])
    assert isinstance(raised, KeyboardInterrupt)
    assert [call[0] for call in fake.calls] == ["wait", "wait"]
    first, grace = _timeouts(fake)
    assert first is None
    assert grace == pytest.approx(60.0, abs=1.0)  # torchrun's default 30 s + 30 s margin


def test_runner_torchrun_repeated_ctrl_c_keeps_waiting(monkeypatch):
    """Another Ctrl-C while waiting (torchrun gets that one too) does not end the wait early:
    the wait has one deadline, counted from the first interrupt."""
    fake, raised = _run_scripted_torchrun(monkeypatch, [KeyboardInterrupt, KeyboardInterrupt, 0])
    assert isinstance(raised, KeyboardInterrupt)
    assert [call[0] for call in fake.calls] == ["wait", "wait", "wait"]
    _, grace, rest = _timeouts(fake)
    assert grace == pytest.approx(60.0, abs=1.0)
    assert 0 <= rest <= grace


def test_runner_torchrun_is_killed_only_after_its_shutdown_timeout(monkeypatch, capsys):
    """Last resort: torchrun still running TORCH_ELASTIC_SHUTDOWN_TIMEOUT + 30 s after the
    interrupt is killed and reaped, with a warning, and KeyboardInterrupt is still re-raised."""
    fake, raised = _run_scripted_torchrun(
        monkeypatch,
        [KeyboardInterrupt, subprocess.TimeoutExpired("torchrun", 35), 0],
        shutdown_timeout="5",
    )
    assert isinstance(raised, KeyboardInterrupt)
    assert fake.calls[2:] == [("kill",), ("wait", None)]
    assert _timeouts(fake)[1] == pytest.approx(35.0, abs=1.0)
    err = capsys.readouterr().err
    assert "torchrun (pid 4242) did not exit 35s after being interrupted; killing it" in err


def test_runner_torchrun_other_errors_ask_torchrun_to_stop_first(monkeypatch):
    """An exception torchrun did not also receive (no terminal SIGINT): send it SIGTERM, which
    it forwards to its ranks like SIGINT, then wait for it the same way and re-raise."""
    fake, raised = _run_scripted_torchrun(
        monkeypatch, [subprocess.TimeoutExpired("torchrun", 1), 0], shutdown_timeout="junk"
    )
    assert isinstance(raised, subprocess.TimeoutExpired)
    assert [call[0] for call in fake.calls] == ["wait", "terminate", "wait"]
    assert _timeouts(fake)[1] == pytest.approx(60.0, abs=1.0)  # unparsable value: default 30 s


def test_runner_torchrun_nonzero_exit_raises_called_process_error(monkeypatch):
    """check=True semantics of the subprocess.run call it replaces."""
    from difflet.cli.runner import build_stage_command

    fake, raised = _run_scripted_torchrun(monkeypatch, [3])
    assert isinstance(raised, subprocess.CalledProcessError)
    assert raised.returncode == 3
    assert raised.cmd == build_stage_command("pkg.mod:Orch", "toy", [], nproc=2)
    assert fake.calls == [("wait", None)]


# ---------------------------------------------------------------------- main


def _parse_main(argv):
    from difflet.cli.main import _build_parser

    return _build_parser().parse_args(argv)


def _allow_backend(monkeypatch, model_type: str, backend: str) -> None:
    """Add ``backend`` to a builtin registry entry for this test (no builtin model lists
    neuron in phase 1)."""
    from difflet import registry

    registry._ensure_builtin_models_registered()
    entry = registry._REGISTRY[model_type]
    monkeypatch.setitem(
        registry._REGISTRY, model_type,
        dataclasses.replace(entry, backends=(*entry.backends, backend)),
    )


@pytest.mark.parametrize("command", ["compile", "generate", "run"])
def test_main_commands_accept_backend_flags(command):
    args = _parse_main([
        command, "--model-id", WAN_ID,
        "--backend", "neuron", "--exec-mode", "eager", "--mode", "latency",
    ])
    # --exec-mode is its own flag; --mode stays the latency/throughput preset.
    assert (args.backend, args.exec_mode, args.mode) == ("neuron", "eager", "latency")


def test_main_backend_flags_default_to_none():
    args = _parse_main(["generate", "--model-id", "x"])
    assert (args.backend, args.exec_mode) == (None, None)


def test_main_apply_backend_flags_exports_environment(monkeypatch):
    from difflet.cli.main import _apply_backend_flags

    _isolate_env(monkeypatch, "DIFFLET_BACKEND", "DIFFLET_EXEC_MODE")

    _apply_backend_flags(SimpleNamespace(backend="neuron", exec_mode="compile"))

    assert os.environ["DIFFLET_BACKEND"] == "neuron"
    assert os.environ["DIFFLET_EXEC_MODE"] == "compile"


def test_main_apply_backend_flags_noop_without_flags(monkeypatch):
    from difflet.cli.main import _apply_backend_flags

    _isolate_env(monkeypatch, "DIFFLET_BACKEND", "DIFFLET_EXEC_MODE")

    _apply_backend_flags(SimpleNamespace(backend=None, exec_mode=None))

    assert "DIFFLET_BACKEND" not in os.environ
    assert "DIFFLET_EXEC_MODE" not in os.environ


def test_main_apply_backend_flags_rejects_exec_mode_for_aot(monkeypatch, capsys):
    from difflet.cli.main import _apply_backend_flags

    _isolate_env(monkeypatch, "DIFFLET_BACKEND", "DIFFLET_EXEC_MODE")

    with pytest.raises(SystemExit) as exc:
        _apply_backend_flags(SimpleNamespace(backend="trainium", exec_mode="eager"))

    assert exc.value.code == 2
    assert "--exec-mode" in capsys.readouterr().err
    assert "DIFFLET_BACKEND" not in os.environ
    assert "DIFFLET_EXEC_MODE" not in os.environ


def test_main_exports_selection_before_orchestrator(monkeypatch, tmp_path):
    cli_main = _cli_main()

    _isolate_env(monkeypatch, "DIFFLET_BACKEND", "DIFFLET_EXEC_MODE")
    monkeypatch.delenv("NEURON_RT_NUM_CORES", raising=False)
    _allow_backend(monkeypatch, "wan", "neuron")
    seen = {}

    class _Orch:
        def generate(self):
            seen["backend"] = os.environ.get("DIFFLET_BACKEND")
            seen["exec_mode"] = os.environ.get("DIFFLET_EXEC_MODE")

    monkeypatch.setattr(cli_main, "_get_orchestrator", lambda args: _Orch())

    cli_main.main([
        "generate", "--model-id", WAN_ID,
        "--backend", "neuron", "--exec-mode", "eager",
        "--prompt", "p", "--output", str(tmp_path / "c10.mp4"),
    ])

    assert seen == {"backend": "neuron", "exec_mode": "eager"}


@pytest.mark.parametrize("command", ["compile", "generate", "run"])
def test_main_rejects_backend_the_model_does_not_support(monkeypatch, capsys, tmp_path, command):
    """A model whose registry entry does not list --backend fails fast with exit code 2,
    before any orchestrator (and so any stage launch) is built."""
    cli_main = _cli_main()

    _isolate_env(monkeypatch, "DIFFLET_BACKEND", "DIFFLET_EXEC_MODE")
    built = []
    monkeypatch.setattr(cli_main, "_get_orchestrator", lambda args: built.append(args))
    argv = [command, "--model-id", WAN_ID, "--backend", "neuron"]
    if command != "compile":
        argv += ["--prompt", "p", "--output", str(tmp_path / "c10.mp4")]

    with pytest.raises(SystemExit) as exc:
        cli_main.main(argv)

    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "model 'wan' does not support backend 'neuron'" in err
    assert "supported backends: trainium, tpu" in err
    assert built == []
    assert "DIFFLET_BACKEND" not in os.environ


HUNYUAN_ID = "hunyuanvideo-community/HunyuanVideo"  # registry backends: trainium


def _main_argv(command: str, model_id: str, tmp_path) -> list[str]:
    argv = [command, "--model-id", model_id]
    if command != "compile":
        argv += ["--prompt", "p", "--output", str(tmp_path / "c10.mp4")]
    return argv


def _recording_orchestrator(monkeypatch, cli_main) -> list:
    built: list = []

    class _Orch:
        def __init__(self, args):
            built.append(args.command)

        def compile(self):
            pass

        def generate(self):
            pass

        def run(self):
            pass

    monkeypatch.setattr(cli_main, "_get_orchestrator", _Orch)
    return built


@pytest.mark.parametrize("command", ["compile", "generate", "run"])
def test_main_rejects_ambient_non_aot_backend_the_model_does_not_support(
    monkeypatch, capsys, tmp_path, command
):
    """DIFFLET_BACKEND=neuron in the shell, no --backend: the stages inherit it and would each
    become N torchrun ranks running the model's Trainium application. Same exit-2 check as
    --backend, naming the variable; nothing is built."""
    cli_main = _cli_main()
    _isolate_env(monkeypatch, "DIFFLET_BACKEND", "DIFFLET_EXEC_MODE")
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    built = _recording_orchestrator(monkeypatch, cli_main)

    with pytest.raises(SystemExit) as exc:
        cli_main.main(_main_argv(command, WAN_ID, tmp_path))

    assert exc.value.code == 2
    assert capsys.readouterr().err.strip() == (
        "Error: model 'wan' does not support backend 'neuron'; supported backends: "
        "trainium, tpu (DIFFLET_BACKEND=neuron)."
    )
    assert built == []


def test_main_ambient_non_aot_backend_the_model_lists_goes_ahead(monkeypatch, tmp_path):
    cli_main = _cli_main()
    _isolate_env(monkeypatch, "DIFFLET_BACKEND", "DIFFLET_EXEC_MODE")
    monkeypatch.delenv("NEURON_RT_NUM_CORES", raising=False)
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    _allow_backend(monkeypatch, "wan", "neuron")
    built = _recording_orchestrator(monkeypatch, cli_main)

    cli_main.main(_main_argv("generate", WAN_ID, tmp_path))

    assert built == ["generate"]
    assert os.environ["DIFFLET_BACKEND"] == "neuron"


@pytest.mark.parametrize(
    "backend, model_id",
    [("trainium", WAN_ID), ("tpu", WAN_ID), ("tpu", HUNYUAN_ID), ("Trainium ", HUNYUAN_ID)],
)
@pytest.mark.parametrize("command", ["compile", "generate"])
def test_main_ambient_aot_backend_keeps_its_old_path(monkeypatch, tmp_path, backend, model_id,
                                                     command):
    """Trainium/TPU (requires_aot) selected through DIFFLET_BACKEND: no CLI model check, as
    before, even for a model whose registry entry does not list the backend (HunyuanVideo
    lists trainium only); the orchestrator is built and the variable left as it was."""
    cli_main = _cli_main()
    _isolate_env(monkeypatch, "DIFFLET_BACKEND", "DIFFLET_EXEC_MODE")
    monkeypatch.delenv("NEURON_RT_NUM_CORES", raising=False)
    monkeypatch.setenv("DIFFLET_BACKEND", backend)
    built = _recording_orchestrator(monkeypatch, cli_main)

    cli_main.main(_main_argv(command, model_id, tmp_path))

    assert built == [command]
    assert os.environ["DIFFLET_BACKEND"] == backend
    assert "DIFFLET_EXEC_MODE" not in os.environ


@pytest.mark.parametrize("exec_mode", [None, "eager"])
def test_main_unknown_ambient_backend_exits_2(monkeypatch, capsys, tmp_path, exec_mode):
    """An unknown DIFFLET_BACKEND is a usage error (exit 2, the registry's message), not a
    traceback, with or without --exec-mode."""
    cli_main = _cli_main()
    _isolate_env(monkeypatch, "DIFFLET_BACKEND", "DIFFLET_EXEC_MODE")
    monkeypatch.setenv("DIFFLET_BACKEND", "no-such-backend")
    built = _recording_orchestrator(monkeypatch, cli_main)
    argv = _main_argv("generate", WAN_ID, tmp_path)
    if exec_mode is not None:
        argv += ["--exec-mode", exec_mode]

    with pytest.raises(SystemExit) as exc:
        cli_main.main(argv)

    assert exc.value.code == 2
    assert capsys.readouterr().err.strip() == (
        "Error: unknown Difflet backend 'no-such-backend'; known backends: "
        "cpu, cuda, neuron, rocm, tpu, trainium (DIFFLET_BACKEND)."
    )
    assert built == []
    assert "DIFFLET_EXEC_MODE" not in os.environ


def test_main_apply_backend_flags_unknown_ambient_backend_with_exec_mode_exits_2(
    monkeypatch, capsys
):
    """The --exec-mode lookup itself (no model id in the namespace): exit 2, no traceback."""
    from difflet.cli.main import _apply_backend_flags

    _isolate_env(monkeypatch, "DIFFLET_BACKEND", "DIFFLET_EXEC_MODE")
    monkeypatch.setenv("DIFFLET_BACKEND", "no-such-backend")

    with pytest.raises(SystemExit) as exc:
        _apply_backend_flags(SimpleNamespace(backend=None, exec_mode="eager"))

    assert exc.value.code == 2
    assert "unknown Difflet backend 'no-such-backend'" in capsys.readouterr().err
    assert "DIFFLET_EXEC_MODE" not in os.environ


def test_main_cli_modules_stay_torch_free():
    code = (
        "import importlib, sys\n"
        "m = importlib.import_module('difflet.cli.main')\n"
        "s = importlib.import_module('difflet.cli.stage')\n"
        "import difflet.cli.runner, difflet.cli.prewarm\n"
        "m._build_parser(); s._build_stage_parser()\n"
        "bad = [n for n in ('torch', 'torch_neuronx') if n in sys.modules]\n"
        "assert not bad, bad\n"
    )
    env = {**os.environ, "PYTHONPATH": _pythonpath()}
    subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=120)


def test_main_stage_launch_loads_stage_once():
    """`python -m difflet.cli.stage` imports the difflet.cli package (and so main.py)
    first; a module-level `import difflet.cli.stage` in main.py would make runpy load
    the stage module twice (RuntimeWarning)."""
    env = {**os.environ, "PYTHONPATH": _pythonpath()}
    proc = subprocess.run(
        [sys.executable, "-m", "difflet.cli.stage", "--orchestrator", "bogus", "--stage", "x"],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 1
    assert "unknown orchestrator" in proc.stderr
    assert "RuntimeWarning" not in proc.stderr


# ------------------------------------------------------------- toy launch


def test_toy_registered_by_model_type_only():
    from difflet.registry import resolve_model
    from tests.unit.backends._neuron_toy import (
        TOY_MODEL_TYPE,
        create_toy_application,
        register_toy_model,
    )

    register_toy_model()
    register_toy_model()  # idempotent

    entry = resolve_model("ignored", model_type=TOY_MODEL_TYPE)
    assert entry.application_factory is create_toy_application
    assert entry.backends == ("neuron", "cpu")
    assert (entry.hf_paths, entry.detector) == ((), None)


def test_toy_orchestrator_loads_by_module_reference():
    import difflet.cli.stage as stage
    from tests.unit.backends._neuron_toy import TOY_ORCHESTRATOR, ToyOrchestrator

    assert TOY_ORCHESTRATOR == "tests.unit.backends._neuron_toy:ToyOrchestrator"
    assert stage._load_orchestrator_class(TOY_ORCHESTRATOR) is ToyOrchestrator


def test_toy_orchestrator_rejects_unknown_stage(tmp_path):
    from tests.unit.backends._neuron_toy import ToyOrchestrator

    args = SimpleNamespace(work_dir=str(tmp_path), exec_mode=None)
    with pytest.raises(ValueError, match="unknown toy stage"):
        ToyOrchestrator(args)._run_stage_internal("vae", args)


def test_toy_pipeline_refuses_a_runtime_already_up(tmp_path, monkeypatch):
    """The stage path checks, right before from_pretrained, that nothing (a prewarm, a
    device tensor) started the Neuron runtime before prepare_runtime binds the core."""
    from difflet.backends.neuron import runtime
    from difflet.pipeline import difflet_pipeline
    from tests.unit.backends._neuron_toy import run_toy_pipeline

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(runtime, "_neuron_runtime_initialized", lambda: True)
    monkeypatch.setattr(
        difflet_pipeline.DiffletPipeline, "from_pretrained",
        classmethod(lambda cls, *a, **kw: pytest.fail("from_pretrained must not run")),
    )

    with pytest.raises(RuntimeError, match="already initialised before DiffletPipeline"):
        run_toy_pipeline(exec_mode="eager", work_dir=tmp_path, device="cpu")


@pytest.mark.parametrize("exec_mode", ["eager", "compile"])
def test_toy_pipeline_gloo_matches_tp1_reference(tmp_path, monkeypatch, exec_mode):
    """4 gloo ranks through DiffletPipeline (non-AoT branch) into the toy lifecycle; compile
    mode runs its forward too (C8's compile_backend is aot_eager on CPU)."""
    from tests.unit.backends._neuron_gloo import run_ranks
    from tests.unit.backends._neuron_toy import TOY_N_BLOCKS, prepare_toy_work_dir
    from tests.unit.backends._neuron_workers import c10_pipeline_worker

    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    prepare_toy_work_dir(tmp_path)

    results = run_ranks(c10_pipeline_worker, exec_mode, str(tmp_path), world_size=4)

    assert [r["rank"] for r in results] == [0, 1, 2, 3]
    assert {(r["exec_mode"], r["backend"], r["world_size"]) for r in results} == {
        (exec_mode, "cpu", 4)
    }
    blocks = [f"blocks.{i}" for i in range(TOY_N_BLOCKS)] if exec_mode == "compile" else []
    assert all(r["forward_ran"] and r["compiled_blocks"] == blocks for r in results)
    assert not any(r["manifest_written"] or r["runtime_initialized_before_load"] for r in results)
    reference = torch.load(tmp_path / "reference.pt")["output"]
    torch.testing.assert_close(torch.load(tmp_path / results[0]["output"]), reference, **FP32_TOL)
    assert json.loads((tmp_path / f"result-{exec_mode}.json").read_text())["world_size"] == 4


_REAL_POPEN = subprocess.Popen


def _bounded_popen(limit: float):
    """subprocess.Popen for run_stage's torchrun branch, with a deadline: the first wait()
    without a timeout gives up after ``limit`` seconds (TimeoutExpired); the branch then sends
    torchrun SIGTERM (torchrun stops its ranks, which run in sessions of their own) and waits
    for it before re-raising. Later waits are left alone."""

    class BoundedPopen(_REAL_POPEN):
        _bounded = False

        def wait(self, timeout=None):
            if timeout is None and not self._bounded:
                self._bounded = True
                timeout = limit
            return super().wait(timeout=timeout)

    return BoundedPopen


def _toy_torchrun_env(monkeypatch, *, limit: float = 300):
    from tests.unit.backends._neuron_toy import TOY_DEVICE_ENV, TOY_FAIL_ENV

    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    monkeypatch.setenv("DIFFLET_DISABLE_PREWARM", "1")
    monkeypatch.setenv(TOY_DEVICE_ENV, "cpu")
    monkeypatch.setenv("PYTHONPATH", _pythonpath())
    monkeypatch.delenv(TOY_FAIL_ENV, raising=False)
    monkeypatch.delenv("DIFFLET_EXEC_MODE", raising=False)
    monkeypatch.setattr(subprocess, "Popen", _bounded_popen(limit))


@pytest.mark.parametrize("exec_mode", ["eager", "compile"])
def test_toy_run_stage_torchrun_cpu_end_to_end(tmp_path, monkeypatch, exec_mode):
    """The real launch: run_stage -> torchrun (2 ranks) -> difflet.cli.stage --exec-mode
    -> ToyOrchestrator -> DiffletPipeline (non-AoT) -> toy lifecycle, on gloo/cpu."""
    from difflet.cli.runner import run_stage
    from tests.unit.backends._neuron_toy import (
        TOY_N_BLOCKS,
        TOY_ORCHESTRATOR,
        TOY_STAGE,
        prepare_toy_work_dir,
    )

    _toy_torchrun_env(monkeypatch)
    prepare_toy_work_dir(tmp_path)

    run_stage(
        TOY_ORCHESTRATOR, TOY_STAGE, num_cores=2, virtual_core_size=None,
        cli_args=["--exec-mode", exec_mode, "--work-dir", str(tmp_path)],
    )

    result = json.loads((tmp_path / f"result-{exec_mode}.json").read_text())
    assert (result["exec_mode"], result["world_size"], result["backend"]) == (exec_mode, 2, "cpu")
    blocks = [f"blocks.{i}" for i in range(TOY_N_BLOCKS)] if exec_mode == "compile" else []
    assert result["compiled_blocks"] == blocks
    assert result["forward_ran"] and not result["runtime_initialized_before_load"]
    assert not result["manifest_written"]
    reference = torch.load(tmp_path / "reference.pt")["output"]
    torch.testing.assert_close(torch.load(tmp_path / result["output"]), reference, **FP32_TOL)


@pytest.mark.parametrize("step", ["build_module", "forward"])
def test_toy_run_stage_torchrun_cpu_rank_failure_stops_the_launch(
    tmp_path, monkeypatch, capfd, step
):
    """One rank raising inside load() makes its stage process exit non-zero, and torchrun
    stops the others and exits non-zero. ``build_module`` fails inside a status-synced phase
    (every rank raises); ``forward`` fails on rank 1 between block 0's two all-reduces, so
    rank 0 sits in a collective until torchrun stops it (C8's no-hang design)."""
    from difflet.cli.runner import run_stage
    from tests.unit.backends._neuron_toy import (
        TOY_FAIL_ENV,
        TOY_ORCHESTRATOR,
        TOY_STAGE,
        prepare_toy_work_dir,
    )

    _toy_torchrun_env(monkeypatch, limit=120)
    monkeypatch.setenv(TOY_FAIL_ENV, f"{step}:1")
    prepare_toy_work_dir(tmp_path)

    start = time.monotonic()
    with pytest.raises(subprocess.CalledProcessError) as exc:
        run_stage(
            TOY_ORCHESTRATOR, TOY_STAGE, num_cores=2, virtual_core_size=None,
            cli_args=["--exec-mode", "eager", "--work-dir", str(tmp_path)],
        )
    elapsed = time.monotonic() - start

    assert exc.value.returncode != 0
    assert elapsed < 60, f"torchrun took {elapsed:.0f}s to stop after one rank failed"
    out = capfd.readouterr()
    log = out.out + out.err
    assert f"injected failure: {step} on rank 1" in log
    if step == "build_module":
        assert "RankFailureError" in log  # rank 0 stopped at the same phase
    assert not (tmp_path / "result-eager.json").exists()


# The driver of the Ctrl-C test, a process of its own: run_stage's torchrun branch, as the CLI
# runs it. argv: orchestrator, stage, nproc, work dir. SIGINT gets Python's default handler even
# if the test runner ignores SIGINT (a background job), so Ctrl-C raises KeyboardInterrupt.
_CTRL_C_DRIVER = """
import signal, sys, time
signal.signal(signal.SIGINT, signal.default_int_handler)
from difflet.cli.runner import run_stage
start = time.monotonic()
try:
    run_stage(sys.argv[1], sys.argv[2], num_cores=int(sys.argv[3]), virtual_core_size=None,
              cli_args=["--work-dir", sys.argv[4]])
except KeyboardInterrupt:
    print(f"driver: KeyboardInterrupt after {time.monotonic() - start:.1f}s", flush=True)
    raise
print("driver: run_stage returned", flush=True)
"""


def _running(pid: int) -> bool:
    """True while ``pid`` runs; a zombie (exited, not yet reaped) counts as gone."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] not in ("Z", "X")


@pytest.mark.parametrize("presses", [1, 2])
def test_toy_run_stage_torchrun_ctrl_c_leaves_no_rank_behind(tmp_path, monkeypatch, presses):
    """Ctrl-C at a terminal sends SIGINT to the foreground process group: the driver
    (run_stage) and torchrun, but not the ranks, which torchrun starts in sessions of their
    own. These ranks ignore SIGINT and SIGTERM, as ranks blocked in a native Neuron compile or
    collective do, so only torchrun's SIGKILL after its shutdown timeout (3 s here) stops them.
    run_stage must wait for that instead of killing torchrun, then re-raise KeyboardInterrupt:
    once the driver has exited, torchrun and every rank are gone. A second Ctrl-C a second
    later (presses=2) must not cut that wait short."""
    from tests.unit.backends._neuron_toy import TOY_HANG_STAGE, TOY_ORCHESTRATOR

    _toy_torchrun_env(monkeypatch)
    monkeypatch.setattr(subprocess, "Popen", _REAL_POPEN)  # the driver runs run_stage itself
    nproc = 2
    env = {**os.environ, "TORCH_ELASTIC_SHUTDOWN_TIMEOUT": "3"}
    log_path = tmp_path / "driver.log"
    pid_files = [tmp_path / f"hang-rank{rank}.pid" for rank in range(nproc)]
    ranks: list[int] = []
    with log_path.open("w") as log:
        driver = subprocess.Popen(
            [sys.executable, "-c", _CTRL_C_DRIVER, TOY_ORCHESTRATOR, TOY_HANG_STAGE,
             str(nproc), str(tmp_path)],
            env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,  # a process group of its own, like a terminal's job
        )
    try:
        deadline = time.monotonic() + 120
        while not all(path.exists() for path in pid_files):
            assert driver.poll() is None, f"the driver exited early:\n{log_path.read_text()}"
            assert time.monotonic() < deadline, "the hang ranks never started"
            time.sleep(0.2)
        pids = [tuple(map(int, path.read_text().split())) for path in pid_files]
        ranks = [rank for rank, _ in pids]
        parents = {parent for _, parent in pids}
        assert len(parents) == 1, pids  # every rank is a child of the one torchrun
        torchrun = parents.pop()
        assert _running(torchrun) and all(_running(rank) for rank in ranks)

        for press in range(presses):
            if press:
                time.sleep(1.0)
            os.killpg(driver.pid, signal.SIGINT)  # what the terminal does on Ctrl-C
        returncode = driver.wait(timeout=120)
        left = [pid for pid in (torchrun, *ranks) if _running(pid)]  # right as it exited
    finally:
        if driver.poll() is None:
            os.killpg(driver.pid, signal.SIGKILL)
            driver.wait()
        for pid in ranks:  # never leave a deaf rank behind, whatever happened above
            if _running(pid):
                os.kill(pid, signal.SIGKILL)
    output = log_path.read_text()
    print(output)  # shown with -rP: torchrun's shutdown log and the driver's line
    assert left == [], f"still running after the driver exited: {left}\n{output}"
    assert returncode == -signal.SIGINT, output  # KeyboardInterrupt reached the top
    assert "driver: KeyboardInterrupt after" in output
    assert "killing it" not in output  # torchrun stopped its ranks itself; no last resort
