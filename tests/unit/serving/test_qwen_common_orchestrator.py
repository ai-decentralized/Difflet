from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest

from difflet.common.orchestrators import qwen_image
from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.serving.orchestrators.qwen_image import (
    QwenImageServingArtifactPreparer,
    QwenImageServingOrchestrator,
)
from difflet.serving.options import CompilePolicy
from difflet.serving.types import ServingProfile


def _profile(cache_dir: Path) -> ServingProfile:
    return ServingProfile(
        model_id="Qwen/Qwen-Image",
        model_type="qwen_image",
        height=1024,
        width=1024,
        num_frames=None,
        parallel=DiffletParallelConfig(tp_degree=4, cp_degree=1),
        cache_dir=str(cache_dir),
    )


def test_qwen_artifact_check_rejects_missing_and_empty_dirs(tmp_path):
    profile = _profile(tmp_path)
    for spec in qwen_image.compile_plan(profile):
        Path(spec.artifact_path).mkdir(parents=True)

    missing = qwen_image.missing_artifacts(profile)

    assert {spec.stage_id for spec in missing} == {"text", "generate", "vae"}


def test_qwen_serving_compile_plan_uses_full_tp_vae_artifact(tmp_path):
    profile = _profile(tmp_path)

    by_stage = {
        spec.stage_id: Path(spec.artifact_path)
        for spec in qwen_image.compile_plan(profile)
    }

    assert by_stage["text"].name == "qwen_image_enc_tp4cp1_seq256"
    assert by_stage["generate"].name == "qwen_image_dit_tp4cp1_h1024w1024"
    assert by_stage["vae"].name == "qwen_image_vae_tp4_h1024w1024"


def test_qwen_serving_stage_topology_uses_full_world_size_for_decoder(tmp_path):
    stages = QwenImageServingArtifactPreparer().stage_specs(_profile(tmp_path))

    assert [(stage.stage_id, stage.num_cores) for stage in stages] == [
        ("prompt_encoder", 4),
        ("denoiser", 4),
        ("decoder", 4),
    ]


def test_qwen_resident_vae_load_uses_full_world_size(monkeypatch, tmp_path):
    profile = _profile(tmp_path)
    neuron_configs = []
    vae_configs = []
    created = []

    class FakeVaeApplication:
        def __init__(self, **kwargs):
            created.append(self)
            self.kwargs = kwargs
            self.loaded = None

        def load(self, path):
            self.loaded = path

    config_module = types.ModuleType("difflet.backends.trainium.core.config")
    config_module.NeuronConfig = lambda **kwargs: neuron_configs.append(kwargs) or kwargs
    vae_module = types.ModuleType("difflet.backends.trainium.wan.vae")
    vae_module.NeuronWanVAEDecoderApplication = FakeVaeApplication
    vae_module.WanVAEDecoderInferenceConfig = (
        lambda **kwargs: vae_configs.append(kwargs) or kwargs
    )
    adapter_module = types.ModuleType("difflet.utils.diffusers_adapter")
    adapter_module.load_diffusers_config = lambda path: {"path": path}
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(sys.modules, "difflet.backends.trainium.core.config", config_module)
    monkeypatch.setitem(sys.modules, "difflet.backends.trainium.wan.vae", vae_module)
    monkeypatch.setitem(sys.modules, "difflet.utils.diffusers_adapter", adapter_module)
    orchestrator = QwenImageServingOrchestrator()
    orchestrator.model_dir = "/tmp/qwen-model"

    orchestrator._load_vae_stage(profile)

    assert neuron_configs == [
        {"tp_degree": 4, "world_size": 4, "torch_dtype": "bf16"}
    ]
    assert vae_configs[0]["neuron_config"] == neuron_configs[0]
    assert created[0].loaded == str(
        qwen_image.serving_stage_compiled_dir("vae", profile)
    )


def test_qwen_artifact_check_rejects_manifest_only_dirs(tmp_path):
    profile = _profile(tmp_path)
    for spec in qwen_image.compile_plan(profile):
        artifact_path = Path(spec.artifact_path)
        artifact_path.mkdir(parents=True)
        (artifact_path / "manifest.json").write_text("{}", encoding="utf-8")

    missing = qwen_image.missing_artifacts(profile)

    assert {spec.stage_id for spec in missing} == {"text", "generate", "vae"}


def test_qwen_artifact_check_rejects_neff_without_serving_marker(tmp_path):
    profile = _profile(tmp_path)
    for spec in qwen_image.compile_plan(profile):
        artifact_path = Path(spec.artifact_path)
        artifact_path.mkdir(parents=True)
        (artifact_path / "graph.neff").write_bytes(b"neff")

    missing = qwen_image.missing_artifacts(profile)

    assert {spec.stage_id for spec in missing} == {"text", "generate", "vae"}


def test_qwen_artifact_check_accepts_neff_with_matching_serving_marker(tmp_path):
    profile = _profile(tmp_path)
    for spec in qwen_image.compile_plan(profile):
        artifact_path = Path(spec.artifact_path)
        artifact_path.mkdir(parents=True)
        (artifact_path / "graph.neff").write_bytes(b"neff")

    qwen_image.write_serving_markers(profile)

    assert qwen_image.missing_artifacts(profile) == []


def test_qwen_artifact_check_accepts_nxd_model_with_matching_serving_marker(tmp_path):
    profile = _profile(tmp_path)
    for spec in qwen_image.compile_plan(profile):
        artifact_path = Path(spec.artifact_path)
        component_path = (
            artifact_path / "transformer" if spec.stage_id == "generate" else artifact_path
        )
        component_path.mkdir(parents=True)
        (component_path / "model.pt").write_bytes(b"torchscript")
        (component_path / "neuron_config.json").write_text("{}", encoding="utf-8")

    qwen_image.write_serving_markers(profile)

    assert qwen_image.missing_artifacts(profile) == []


def test_qwen_artifact_check_rejects_nxd_model_without_neuron_config(tmp_path):
    profile = _profile(tmp_path)
    for spec in qwen_image.compile_plan(profile):
        artifact_path = Path(spec.artifact_path)
        artifact_path.mkdir(parents=True)
        (artifact_path / "model.pt").write_bytes(b"torchscript")

    qwen_image.write_serving_markers(profile)

    assert {spec.stage_id for spec in qwen_image.missing_artifacts(profile)} == {
        "text",
        "generate",
        "vae",
    }


def test_qwen_artifact_check_rejects_mismatched_serving_marker(tmp_path):
    profile = _profile(tmp_path)
    for spec in qwen_image.compile_plan(profile):
        artifact_path = Path(spec.artifact_path)
        artifact_path.mkdir(parents=True)
        (artifact_path / "graph.neff").write_bytes(b"neff")
    qwen_image.write_serving_markers(profile)
    other = ServingProfile(
        model_id=profile.model_id,
        model_type=profile.model_type,
        height=512,
        width=512,
        num_frames=None,
        parallel=profile.parallel,
        cache_dir=profile.cache_dir,
    )

    missing = qwen_image.missing_artifacts(other)

    assert {spec.stage_id for spec in missing} == {"text", "generate", "vae"}


def test_qwen_ensure_artifacts_never_fails_for_missing_artifacts(tmp_path):
    profile = _profile(tmp_path)

    with pytest.raises(RuntimeError, match="missing Qwen compiled artifacts"):
        qwen_image.ensure_artifacts(profile, CompilePolicy.NEVER)


def test_qwen_ensure_artifacts_compiles_serving_vae_with_full_world_size(
    monkeypatch,
    tmp_path,
):
    profile = _profile(tmp_path)
    calls = []

    def fake_run_stage(orchestrator, stage, *, num_cores, virtual_core_size, cli_args):
        calls.append((orchestrator, stage, num_cores, virtual_core_size, cli_args))
        spec = next(spec for spec in qwen_image.compile_plan(profile) if spec.stage_id == stage)
        component = Path(spec.artifact_path)
        if stage == "generate":
            component /= "transformer"
        component.mkdir(parents=True, exist_ok=True)
        (component / "model.pt").write_bytes(b"torchscript")
        (component / "neuron_config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr("difflet.cli.runner.run_stage", fake_run_stage)

    qwen_image.ensure_artifacts(profile, CompilePolicy.AUTO)

    assert [call[1] for call in calls] == ["text", "generate", "vae"]
    assert all(call[2] == 4 for call in calls)
    vae_args = calls[-1][4]
    assert vae_args[vae_args.index("--vae-tp-degree") + 1] == "4"
    assert vae_args[vae_args.index("--compiled-dir") + 1].endswith(
        "qwen_image_vae_tp4_h1024w1024"
    )
