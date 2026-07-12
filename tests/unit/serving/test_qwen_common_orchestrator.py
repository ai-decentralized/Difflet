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
from difflet.serving.types import (
    ArtifactPublishTarget,
    DiffletGenerateOutput,
    DiffletGenerateRequest,
    PipelineDefinition,
    QwenGenerateStageOutputs,
    QwenTextStageOutputs,
    QwenVaeStageOutputs,
    ResolvedModelSource,
    ServingProfile,
    StageDefinition,
    WorkerRequestContext,
)
from difflet.serving.orchestrators.qwen_image import (
    QwenImageServingOrchestrator,
    _packed_latent_grid,
)


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


def test_qwen_serving_derives_rectangular_packed_latent_grid_from_profile(tmp_path):
    profile = replace(_profile(tmp_path), height=512, width=1024)

    assert _packed_latent_grid(profile, seq=2048) == (32, 64)


def test_qwen_serving_rejects_packed_latents_that_do_not_match_profile(tmp_path):
    with pytest.raises(ValueError, match="does not match the serving profile"):
        _packed_latent_grid(_profile(tmp_path), seq=2048)


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
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "numpy",
        types.SimpleNamespace(
            linspace=lambda start, stop, count: SimpleNamespace(tolist=lambda: [start] * count)
        ),
    )
    orchestrator = QwenImageServingOrchestrator()
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


def test_qwen_generate_traverses_frozen_pipeline_and_returns_final_output():
    orchestrator = QwenImageServingOrchestrator()
    orchestrator.active_runtime = SimpleNamespace(
        pipeline_definition=PipelineDefinition(
            model_type="qwen_image",
            stages=(
                StageDefinition("text", "extracted", "prompt_encoder", runner_factory="x"),
                StageDefinition("generate", "extracted", "denoiser", runner_factory="x"),
                StageDefinition(
                    "vae", "extracted", "decoder", final_output=True, runner_factory="x"
                ),
            ),
        )
    )
    calls = []

    class _Runner:
        def __init__(self, stage):
            self.stage = stage

        def execute(self, value, request, context):
            calls.append(self.stage)
            if self.stage == "text":
                return QwenTextStageOutputs("hidden", "mask")
            if self.stage == "generate":
                return QwenGenerateStageOutputs("latents")
            return QwenVaeStageOutputs(DiffletGenerateOutput(b"png", "image/png"))

    orchestrator.runners = {stage: _Runner(stage) for stage in ("text", "generate", "vae")}
    request = DiffletGenerateRequest("id", "Qwen/Qwen-Image", "prompt", 64, 64, 1, 1.0, 0)

    output = asyncio.run(
        orchestrator.generate(request, WorkerRequestContext.with_timeout("id", 1.0))
    )

    assert calls == ["text", "generate", "vae"]
    assert output.data == b"png"
