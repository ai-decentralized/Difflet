from __future__ import annotations

import asyncio
import json
import sys
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from difflet.common.orchestrators import qwen_image
from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.pipeline.teacache import TeaCacheCalibration
from difflet.serving.errors import DiffletServingError
from difflet.serving.types import (
    ArtifactPublishTarget,
    DiffletGenerateRequest,
    ResolvedModelSource,
    ServingProfile,
)
from difflet.serving.orchestrators.qwen_image import (
    QwenImageServingRequestValidator,
    QwenImageServingStageAdapter,
    _build_tpu_teacache,
    _probe_free_teacache,
    _runtime_plan,
    _packed_latent_grid,
    _tpu_denoise_loop,
)


@pytest.mark.parametrize("guidance", [-1.0, 20.0001, 1e308])
def test_qwen_request_validator_rejects_guidance_before_tokenization(guidance):
    validator = object.__new__(QwenImageServingRequestValidator)
    validator._tokenizer = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("invalid guidance must be rejected before tokenization")
    )
    request = DiffletGenerateRequest(
        "request",
        "Qwen/Qwen-Image",
        "prompt",
        1024,
        1024,
        4,
        guidance,
        0,
    )

    with pytest.raises(DiffletServingError) as exc:
        validator.validate(request)

    assert exc.value.code == "invalid_extra_body"
    assert "0 <= value <= 20" in exc.value.message


def test_qwen_runtime_plan_preserves_inherited_core_visibility(monkeypatch):
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "4-7")
    profile = ServingProfile(
        model_id="Qwen/Qwen-Image",
        model_type="qwen_image",
        height=1024,
        width=1024,
        num_frames=None,
        parallel=DiffletParallelConfig(tp_degree=4),
    )
    pipeline = SimpleNamespace(
        stages=tuple(SimpleNamespace(stage_id=stage) for stage in ("text", "generate", "vae"))
    )
    specs = tuple(
        SimpleNamespace(artifact_id=stage, identity=SimpleNamespace(digest=stage))
        for stage in ("text", "generate", "vae")
    )

    plan = _runtime_plan(profile, pipeline, specs)

    assert plan.environment.available_core_ids == (4, 5, 6, 7)


def _profile(
    tmp_path: Path,
    *,
    teacache_speedup: float | None = None,
    teacache_calibration: str | None = None,
) -> ServingProfile:
    return ServingProfile(
        model_id="Qwen/Qwen-Image",
        model_type="qwen_image",
        height=1024,
        width=1024,
        num_frames=None,
        parallel=DiffletParallelConfig(tp_degree=4, cp_degree=1),
        cache_dir=str(tmp_path),
        teacache_speedup=teacache_speedup,
        teacache_calibration=teacache_calibration,
    )


def _source(tmp_path: Path) -> ResolvedModelSource:
    model_path = tmp_path / "snapshots" / ("a" * 40)
    model_path.mkdir(parents=True, exist_ok=True)
    return ResolvedModelSource(
        source_kind="hf_snapshot",
        model_id="Qwen/Qwen-Image",
        requested_revision="main",
        pinned_model_path=str(model_path),
        resolved_source_id="a" * 40,
    )


def _write_component(path: Path, stage: str, *, probe: bool = False) -> None:
    component = path / "transformer" if stage == "generate" else path
    component.mkdir(parents=True, exist_ok=True)
    (component / "model.pt").write_bytes(b"torchscript")
    (component / "neuron_config.json").write_text("{}", encoding="utf-8")
    if probe:
        probe_path = path / "teacache_probe"
        probe_path.mkdir(parents=True)
        (probe_path / "model.pt").write_bytes(b"probe")
        (probe_path / "neuron_config.json").write_text("{}", encoding="utf-8")


def test_qwen_compile_plan_is_path_free_and_pins_source(tmp_path):
    profile = _profile(tmp_path)
    specs = qwen_image.build_compile_plan(_source(tmp_path), profile)

    assert [(spec.artifact_id, spec.component_id) for spec in specs] == [
        ("text", "text"),
        ("generate", "generate"),
        ("vae", "vae"),
    ]
    identities = {
        spec.component_id: json.loads(spec.identity.canonical_cache_inputs_json) for spec in specs
    }
    assert all(value["resolved_source_id"] == "a" * 40 for value in identities.values())
    assert identities["vae"]["tp_degree"] == 4
    assert identities["vae"]["cp_degree"] == 1
    assert identities["text"]["stage_inputs"]["enc_seq"] == qwen_image.ENC_SEQ
    assert identities["generate"]["stage_inputs"]["text_seq_len"] == qwen_image.TEXT_SEQ_LEN
    assert identities["generate"]["virtual_core_size"] == qwen_image.VIRTUAL_CORE_SIZE
    assert "toolchain" in identities["generate"]


def test_qwen_compile_identity_changes_with_stage_contract_and_toolchain(monkeypatch, tmp_path):
    profile = _profile(tmp_path)
    source = _source(tmp_path)
    baseline = {
        spec.component_id: spec.identity for spec in qwen_image.build_compile_plan(source, profile)
    }

    monkeypatch.setattr(qwen_image, "ENC_SEQ", qwen_image.ENC_SEQ + 1)
    changed_sequence = {
        spec.component_id: spec.identity for spec in qwen_image.build_compile_plan(source, profile)
    }
    assert changed_sequence["text"] != baseline["text"]
    assert changed_sequence["generate"] == baseline["generate"]

    monkeypatch.setattr(qwen_image, "toolchain_versions", lambda: {"python": "changed"})
    changed_toolchain = qwen_image.build_compile_plan(source, profile)
    assert all(spec.identity != baseline[spec.component_id] for spec in changed_toolchain)


def test_qwen_compile_forwards_pinned_source_and_manager_target(monkeypatch, tmp_path):
    profile = _profile(tmp_path)
    source = _source(tmp_path)
    spec = qwen_image.build_compile_plan(source, profile)[0]
    staging = tmp_path / "staging"
    staging.mkdir()
    target = ArtifactPublishTarget("text", spec.identity, tmp_path, staging)
    calls = []

    def fake_run_stage(
        orchestrator,
        stage,
        *,
        num_cores,
        virtual_core_size,
        cli_args,
        strict_environment,
    ):
        calls.append(
            (
                orchestrator,
                stage,
                num_cores,
                virtual_core_size,
                cli_args,
                strict_environment,
            )
        )
        _write_component(staging, stage)

    monkeypatch.setattr("difflet.cli.runner.run_stage", fake_run_stage)

    qwen_image.compile_serving_artifact(source, profile, spec, target)

    args = calls[0][4]
    assert args[args.index("--model-path") + 1] == source.pinned_model_path
    assert args[args.index("--revision") + 1] == source.resolved_source_id
    assert args[args.index("--compiled-dir") + 1] == str(staging)
    assert calls[0][5] is True
    qwen_image.validate_compiled_artifact(spec, staging)


def test_qwen_vae_compile_uses_resident_world_size(monkeypatch, tmp_path):
    profile = _profile(tmp_path)
    source = _source(tmp_path)
    spec = qwen_image.build_compile_plan(source, profile)[2]
    staging = tmp_path / "staging"
    staging.mkdir()
    target = ArtifactPublishTarget("vae", spec.identity, tmp_path, staging)
    calls = []

    def fake_run_stage(
        orchestrator,
        stage,
        *,
        num_cores,
        virtual_core_size,
        cli_args,
        strict_environment,
    ):
        assert strict_environment is True
        calls.append((num_cores, cli_args))
        _write_component(staging, stage)

    monkeypatch.setattr("difflet.cli.runner.run_stage", fake_run_stage)

    qwen_image.compile_serving_artifact(source, profile, spec, target)

    num_cores, args = calls[0]
    assert num_cores == 4
    assert args[args.index("--vae-tp-degree") + 1] == "4"


def test_qwen_compile_materializes_frozen_calibration_in_staging(monkeypatch, tmp_path):
    calibration = TeaCacheCalibration(
        model="qwen_image",
        shape_label="1024x1024",
        num_steps=50,
        poly_coef=(0.0, 1.0),
        threshold=0.1,
        target_speedup=1.5,
    )
    profile = replace(
        _profile(tmp_path, teacache_speedup=1.5),
        teacache_calibration_data=calibration,
    )
    source = _source(tmp_path)
    spec = qwen_image.build_compile_plan(source, profile)[1]
    staging = tmp_path / "staging"
    staging.mkdir()
    target = ArtifactPublishTarget("generate", spec.identity, tmp_path, staging)
    calls = []

    def fake_run_stage(
        orchestrator,
        stage,
        *,
        num_cores,
        virtual_core_size,
        cli_args,
        strict_environment,
    ):
        assert strict_environment is True
        calls.append(cli_args)
        _write_component(staging, stage, probe=True)

    monkeypatch.setattr("difflet.cli.runner.run_stage", fake_run_stage)

    qwen_image.compile_serving_artifact(source, profile, spec, target)

    args = calls[0]
    calibration_path = Path(args[args.index("--teacache-calibration") + 1])
    assert calibration_path == staging / "teacache_calibration.json"
    assert json.loads(calibration_path.read_text(encoding="utf-8"))["model"] == "qwen_image"


def test_qwen_adaptive_generate_requires_probe(monkeypatch, tmp_path):
    profile = _profile(
        tmp_path,
        teacache_speedup=1.5,
        teacache_calibration="/tmp/calibration.json",
    )
    source = _source(tmp_path)
    spec = qwen_image.build_compile_plan(source, profile)[1]
    staging = tmp_path / "staging"
    staging.mkdir()
    target = ArtifactPublishTarget("generate", spec.identity, tmp_path, staging)

    def incomplete_compile(*args, **kwargs):
        _write_component(staging, "generate")

    monkeypatch.setattr("difflet.cli.runner.run_stage", incomplete_compile)

    with pytest.raises(ValueError, match="TeaCache probe"):
        qwen_image.compile_serving_artifact(source, profile, spec, target)


def test_qwen_validation_rejects_tampered_marker(monkeypatch, tmp_path):
    profile = _profile(tmp_path)
    source = _source(tmp_path)
    spec = qwen_image.build_compile_plan(source, profile)[0]
    staging = tmp_path / "staging"
    staging.mkdir()
    target = ArtifactPublishTarget("text", spec.identity, tmp_path, staging)

    def fake_compile(*args, **kwargs):
        _write_component(staging, "text")

    monkeypatch.setattr("difflet.cli.runner.run_stage", fake_compile)
    qwen_image.compile_serving_artifact(source, profile, spec, target)
    marker = staging / qwen_image.SERVING_ARTIFACT_MARKER
    marker.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        qwen_image.validate_compiled_artifact(spec, staging)


def test_staged_cli_compiled_path_helper_is_unchanged(tmp_path):
    assert (
        qwen_image.stage_compiled_dir_from_values(
            "generate",
            cache_dir=str(tmp_path),
            tp_degree=2,
            cp_degree=2,
            height=768,
            width=1024,
        )
        == tmp_path / "qwen_image_dit_tp2cp2_h768w1024"
    )


def test_qwen_serving_derives_rectangular_packed_latent_grid_from_request_shape():
    assert _packed_latent_grid(512, 1024, seq=2048) == (32, 64)


def test_qwen_serving_rejects_packed_latents_that_do_not_match_request_shape():
    with pytest.raises(ValueError, match="does not match the request shape"):
        _packed_latent_grid(1024, 1024, seq=2048)


@pytest.mark.parametrize("steps,expected", [(50, True), (4, False)])
def test_qwen_serving_selects_teacache_per_request_steps(monkeypatch, tmp_path, steps, expected):
    calibration = TeaCacheCalibration(
        model="qwen_image",
        shape_label="1024x1024",
        num_steps=50,
        poly_coef=(0.0, 1.0),
        threshold=0.1,
    )
    profile = replace(
        _profile(tmp_path, teacache_speedup=1.5),
        teacache_calibration_data=calibration,
    )
    captured = {}

    class _Scheduler:
        config = SimpleNamespace(
            max_shift=1.0,
            base_shift=0.0,
            max_image_seq_len=4096,
            base_image_seq_len=1,
        )
        timesteps = []

        def set_timesteps(self, *, sigmas, mu, device):
            self.timesteps = list(sigmas)

    class _Latents:
        def cpu(self):
            return self

    class _Pipeline:
        scheduler = _Scheduler()

        def __call__(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(latents=_Latents())

    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            bfloat16="bf16",
            full=lambda *args, **kwargs: "guidance",
            manual_seed=lambda seed: None,
            randn=lambda *args, **kwargs: "request-shape-latents",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "numpy",
        types.SimpleNamespace(
            linspace=lambda start, stop, count: SimpleNamespace(tolist=lambda: [start] * count)
        ),
    )
    orchestrator = QwenImageServingStageAdapter()
    orchestrator.active_profile = profile
    orchestrator.denoise_app = SimpleNamespace(pipeline=_Pipeline())
    request = DiffletGenerateRequest(
        "request",
        profile.model_id,
        "prompt",
        profile.height,
        profile.width,
        steps,
        1.0,
        0,
    )

    orchestrator._denoise(
        {"encoder_hidden_states": "hidden", "encoder_hidden_states_mask": "mask"},
        request,
    )

    assert captured["teacache_enabled"] is expected


@pytest.mark.parametrize(
    "probe_free",
    [{"teacache_cadence": 2}, {"teacache_online_delta": 0.6}],
)
def test_qwen_serving_leaves_probe_free_controller_to_the_pipeline(monkeypatch, tmp_path, probe_free):
    """Probe-free modes have no calibration step count to match, so the adapter
    passes ``teacache_enabled=None`` and the pipeline's own controller decides.
    ``False`` here (what the adaptive gate emits on a step mismatch) would
    silently disable the cadence the operator asked for."""
    profile = replace(_profile(tmp_path), **probe_free)
    captured = {}

    class _Scheduler:
        config = SimpleNamespace(
            max_shift=1.0, base_shift=0.0, max_image_seq_len=4096, base_image_seq_len=1
        )
        timesteps = []

        def set_timesteps(self, *, sigmas, mu, device):
            self.timesteps = list(sigmas)

    class _Pipeline:
        scheduler = _Scheduler()

        def __call__(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(latents=SimpleNamespace(cpu=lambda: None))

    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            bfloat16="bf16",
            full=lambda *args, **kwargs: "guidance",
            manual_seed=lambda seed: None,
            randn=lambda *args, **kwargs: "latents",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "numpy",
        types.SimpleNamespace(
            linspace=lambda start, stop, count: SimpleNamespace(tolist=lambda: [start] * count)
        ),
    )
    adapter = QwenImageServingStageAdapter()
    adapter.active_profile = profile
    adapter.denoise_app = SimpleNamespace(pipeline=_Pipeline())
    request = DiffletGenerateRequest(
        "request", profile.model_id, "prompt", profile.height, profile.width, 20, 1.0, 0
    )

    adapter._denoise({"encoder_hidden_states": "h", "encoder_hidden_states_mask": "m"}, request)

    assert captured["teacache_enabled"] is None


def test_probe_free_teacache_tolerates_profiles_without_the_fields():
    """benchmark/adapters/tpu.py drives the adapter with a SimpleNamespace
    profile; missing fields must read as 'off', not as an AttributeError."""
    assert _probe_free_teacache(SimpleNamespace(height=1024, width=1024)) == (None, None)
    assert _probe_free_teacache(SimpleNamespace(teacache_cadence=2)) == (2, None)
    assert _probe_free_teacache(SimpleNamespace(teacache_online_delta=0.6)) == (None, 0.6)


def test_build_tpu_teacache_is_off_unless_a_probe_free_mode_is_set(tmp_path):
    assert _build_tpu_teacache(_profile(tmp_path)) is None

    controller = _build_tpu_teacache(replace(_profile(tmp_path), teacache_cadence=2))
    cal = controller.calibration
    assert (cal.model, cal.shape_label, cal.cadence, cal.online_delta_alpha) == (
        "qwen_image",
        "1024x1024",
        2,
        0.0,
    )
    # No probe NEFF exists on TPU; the controller must not ask for a signal.
    assert controller.needs_signal() is False


# --- TPU device-resident denoise loop with the probe-free controller -----------


class _CountingDiT:
    """Stands in for the sharded DiT: records calls, returns a scripted velocity."""

    def __init__(self, velocities):
        self.velocities = velocities
        self.calls: list[int] = []

    def __call__(self, hidden_states, timestep, states, mask, guidance):
        index = int(timestep.item())
        self.calls.append(index)
        return self.velocities[index].to(hidden_states.dtype)


def _loop_inputs(steps: int):
    torch = pytest.importorskip("torch")
    latents = torch.zeros(1, 4, 3, dtype=torch.float32)
    # The loop indexes these per step; the "timestep" doubles as the step id so
    # the fake DiT can look up its scripted output.
    timesteps = [torch.tensor([float(i)]) for i in range(steps)]
    deltas = [torch.tensor([-1.0 / steps]) for _ in range(steps)]
    states = torch.zeros(1, 2, 3)
    return torch, latents, timesteps, deltas, states


def _reference(torch, latents, deltas, velocities, skipped: set[int]):
    """The Euler loop with TeaCache's residual extrapolation applied by hand.

    A skipped step uses ``prev + residual`` where ``residual`` is what the last
    FULL step recorded (``v_full - prev_at_that_time``); the skipped prediction
    then becomes ``prev`` while the residual is carried unchanged. The DiT ran
    in bf16, so the recorded velocities are bf16-quantized before the fp32
    arithmetic, exactly as the loop's ``.to(torch.float32)`` sees them.
    """
    quantized = [v.to(torch.bfloat16).to(torch.float32) for v in velocities]
    prev = residual = None
    for i, delta in enumerate(deltas):
        if i in skipped:
            v = prev + residual
        else:
            v = quantized[i]
            if prev is not None:
                residual = v - prev
        prev = v
        latents = latents + delta * v
    return latents


def test_tpu_denoise_loop_without_a_controller_is_the_plain_euler_loop():
    torch, latents, timesteps, deltas, states = _loop_inputs(6)
    velocities = [torch.full((1, 4, 3), float(i + 1)) for i in range(6)]
    dit = _CountingDiT(velocities)
    marks = []

    out = _tpu_denoise_loop(
        dit, latents, timesteps, deltas, states, controller=None, mark_step=lambda: marks.append(1)
    )

    assert dit.calls == list(range(6))
    assert len(marks) == 6  # one graph cut per step, skipped or not
    torch.testing.assert_close(out, _reference(torch, latents, deltas, velocities, set()))


def test_tpu_denoise_loop_fixed_cadence_skips_the_dit_and_extrapolates():
    """cadence=2 over 20 steps with the controller's default warmup/cooldown of
    5: steps 5..14 are the window and every other one is skipped — 5 skips, 15
    DiT calls — and a skipped step's velocity is prev + cached residual."""
    from difflet.pipeline.teacache import build_probe_free_controller

    torch, latents, timesteps, deltas, states = _loop_inputs(20)
    # A varying trajectory, so extrapolation is distinguishable from reuse.
    velocities = [torch.full((1, 4, 3), float(i * i)) for i in range(20)]
    dit = _CountingDiT(velocities)
    controller = build_probe_free_controller(
        model="qwen_image", shape_label="1024x1024", cadence=2
    )
    marks = []

    out = _tpu_denoise_loop(
        dit, latents, timesteps, deltas, states, controller=controller,
        mark_step=lambda: marks.append(1),
    )

    skipped = {6, 8, 10, 12, 14}
    assert dit.calls == [i for i in range(20) if i not in skipped]
    assert len(marks) == 20
    assert controller.stats()["skipped_steps"] == 5
    assert controller.stats()["full_steps"] == 15
    # num_steps was 0 at build time; the loop synced it so the cooldown held.
    assert controller.calibration.num_steps == 20
    torch.testing.assert_close(out, _reference(torch, latents, deltas, velocities, skipped))


def test_tpu_denoise_loop_resets_the_controller_between_requests():
    from difflet.pipeline.teacache import build_probe_free_controller

    torch, latents, timesteps, deltas, states = _loop_inputs(20)
    velocities = [torch.full((1, 4, 3), 1.0) for _ in range(20)]
    controller = build_probe_free_controller(
        model="qwen_image", shape_label="1024x1024", cadence=2
    )
    first = _CountingDiT(velocities)
    _tpu_denoise_loop(first, latents, timesteps, deltas, states, controller=controller,
                      mark_step=lambda: None)
    second = _CountingDiT(velocities)
    _tpu_denoise_loop(second, latents, timesteps, deltas, states, controller=controller,
                      mark_step=lambda: None)

    # A second request sees a fresh cache: same 15 calls, not 14 or an
    # extrapolation from the previous request's residual.
    assert second.calls == first.calls
    assert controller.stats() == {
        "full_steps": 15, "skipped_steps": 5, "probe_calls": 0,
        "last_delta_estimate": None, "cache_initialized": True,
    }


def test_tpu_denoise_loop_online_delta_skips_only_flat_steps():
    """online-delta: a step is skipped iff the previous full step's rel-L1 change
    fell below alpha x the first post-warmup delta, never two in a row."""
    from difflet.pipeline.teacache import build_probe_free_controller

    torch, latents, timesteps, deltas, states = _loop_inputs(20)
    # Steps 0..9 move a lot (delta ~ 1 relative); steps 10..19 barely move, so
    # the trajectory turns flat exactly where skipping should begin.
    values = [float(2 ** i) for i in range(10)] + [512.0 + 0.001 * i for i in range(10)]
    velocities = [torch.full((1, 4, 3), v) for v in values]
    dit = _CountingDiT(velocities)
    controller = build_probe_free_controller(
        model="qwen_image", shape_label="1024x1024", online_delta_alpha=0.5
    )

    _tpu_denoise_loop(dit, latents, timesteps, deltas, states, controller=controller,
                      mark_step=lambda: None)

    skipped = sorted(set(range(20)) - set(dit.calls))
    assert skipped, "the flat tail must produce skips"
    assert min(skipped) >= 11  # never inside warmup, never before the trajectory flattens
    assert max(skipped) < 15  # never inside the 5-step cooldown
    assert all(b - a >= 2 for a, b in zip(skipped, skipped[1:]))  # no two in a row


# --- Multi-shape (--shapes) serving: one worker serves the compiled shape set ---

_QWEN_SHAPES = ((512, 512, None), (1024, 1024, None))


def _multi_shape_profile(tmp_path: Path) -> ServingProfile:
    return replace(_profile(tmp_path), shapes=_QWEN_SHAPES)


def test_qwen_compile_identities_cover_the_shape_set(tmp_path):
    source = _source(tmp_path)
    single = {
        spec.component_id: spec.identity
        for spec in qwen_image.build_compile_plan(source, _profile(tmp_path))
    }
    multi = {
        spec.component_id: spec.identity
        for spec in qwen_image.build_compile_plan(source, _multi_shape_profile(tmp_path))
    }

    # The text encoder is shape-invariant: one artifact serves every set.
    assert multi["text"] == single["text"]
    assert multi["generate"] != single["generate"]
    assert multi["vae"] != single["vae"]

    inputs = json.loads(multi["generate"].canonical_cache_inputs_json)
    assert inputs["compile_contract_version"] == 3
    assert inputs["shapes"] == [[1024, 1024, None], [512, 512, None]]
    assert "height" not in inputs
    text_inputs = json.loads(multi["text"].canonical_cache_inputs_json)
    assert "shapes" not in text_inputs and "height" not in text_inputs

    reordered = {
        spec.component_id: spec.identity
        for spec in qwen_image.build_compile_plan(
            source,
            replace(
                _profile(tmp_path),
                shapes=((1024, 1024, None), (512, 512, None), (512, 512, None)),
            ),
        )
    }
    assert reordered == multi

    # K=1 as an explicit one-member set matches the legacy single-shape form.
    explicit_single = {
        spec.component_id: spec.identity
        for spec in qwen_image.build_compile_plan(
            source, replace(_profile(tmp_path), shapes=((1024, 1024, None),))
        )
    }
    assert explicit_single == single


def test_qwen_compile_forwards_shape_set_to_the_cli_stage(monkeypatch, tmp_path):
    profile = _multi_shape_profile(tmp_path)
    source = _source(tmp_path)
    spec = qwen_image.build_compile_plan(source, profile)[1]
    staging = tmp_path / "staging"
    staging.mkdir()
    target = ArtifactPublishTarget("generate", spec.identity, tmp_path, staging)
    calls = []

    def fake_run_stage(orchestrator, stage, *, cli_args, **kwargs):
        calls.append(cli_args)
        _write_component(staging, stage)

    monkeypatch.setattr("difflet.cli.runner.run_stage", fake_run_stage)

    qwen_image.compile_serving_artifact(source, profile, spec, target)

    args = calls[0]
    assert args[args.index("--shapes") + 1] == "1024x1024,512x512"


def test_qwen_namespace_from_profile_omits_shapes_for_single_shape(tmp_path):
    namespace = qwen_image.namespace_from_profile(_profile(tmp_path), stage_mode="compile")
    assert namespace.shapes is None


def test_qwen_request_validator_enforces_shape_set_membership(tmp_path):
    validator = object.__new__(QwenImageServingRequestValidator)
    validator.runtime = SimpleNamespace(profile=_multi_shape_profile(tmp_path))
    validator._tokenizer = lambda *args, **kwargs: SimpleNamespace(
        input_ids=SimpleNamespace(shape=(1, 10))
    )

    def _request(height, width):
        return DiffletGenerateRequest(
            "request",
            "Qwen/Qwen-Image",
            "prompt",
            height,
            width,
            4,
            1.0,
            0,
        )

    validator.validate(_request(1024, 1024))
    validator.validate(_request(512, 512))

    with pytest.raises(DiffletServingError) as exc:
        validator.validate(_request(768, 768))
    assert exc.value.code == "profile_mismatch"
    assert "768x768" in exc.value.message
    assert "1024x1024" in exc.value.message and "512x512" in exc.value.message
