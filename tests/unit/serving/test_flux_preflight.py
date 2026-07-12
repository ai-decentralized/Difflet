from __future__ import annotations

import asyncio
import io
import os
import sys
import types
from types import SimpleNamespace
from pathlib import Path
from dataclasses import replace

import pytest

from difflet.serving.types import (
    ArtifactPublishTarget,
    DiffletGenerateRequest,
    ResolvedModelSource,
    ServingProfile,
    WorkerRequestContext,
)
from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.pipeline.teacache import TeaCacheCalibration
from difflet.common.orchestrators import flux as flux_common
from difflet.serving.orchestrators.flux import FluxServingOrchestrator


def test_flux_build_pipeline_forwards_adaptive_teacache(monkeypatch):
    captured = {}
    monkeypatch.setattr(flux_common, "_torch_bfloat16", lambda: "bfloat16")

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(
        "difflet.pipeline.difflet_pipeline.DiffletPipeline",
        FakePipeline,
    )
    profile = ServingProfile(
        model_id="black-forest-labs/FLUX.1-dev",
        model_type="flux",
        height=1024,
        width=1024,
        num_frames=None,
        parallel=DiffletParallelConfig(tp_degree=4, sp_enabled=True),
        teacache_speedup=1.5,
        teacache_calibration="/tmp/flux-calibration.json",
    )

    flux_common.build_pipeline(profile.model_id, profile, load=False)

    assert captured["parallel"].sp_enabled is True
    assert captured["teacache_speedup"] == 1.5
    assert captured["teacache_calibration_path"] == "/tmp/flux-calibration.json"


def test_flux_compile_uses_pinned_source_and_manager_target(monkeypatch, tmp_path):
    source_path = tmp_path / "snapshots" / ("a" * 40)
    source_path.mkdir(parents=True)
    source = ResolvedModelSource(
        source_kind="hf_snapshot",
        model_id="black-forest-labs/FLUX.1-dev",
        requested_revision="main",
        pinned_model_path=str(source_path),
        resolved_source_id="a" * 40,
    )
    profile = ServingProfile(
        model_id=source.model_id,
        model_type="flux",
        height=1024,
        width=1024,
        num_frames=None,
        parallel=DiffletParallelConfig(tp_degree=4),
    )
    monkeypatch.setattr(flux_common, "_torch_bfloat16", lambda: "bfloat16")
    spec = flux_common.build_compile_plan(source, profile)[0]
    staging = tmp_path / "staging"
    target = ArtifactPublishTarget("pipeline", spec.identity, tmp_path, staging)
    captured = {}
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "7")
    monkeypatch.setenv("NEURON_RT_NUM_CORES", "8")
    monkeypatch.setenv("NEURON_RT_VIRTUAL_CORE_SIZE", "2")
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "1")
    monkeypatch.setenv("WORLD_SIZE", "8")

    def fake_build_pipeline(*args, **kwargs):
        captured.update(kwargs)
        captured["environment"] = {
            name: os.environ.get(name)
            for name in (
                "NEURON_RT_VISIBLE_CORES",
                "NEURON_RT_NUM_CORES",
                "NEURON_RT_VIRTUAL_CORE_SIZE",
                "NEURON_LOGICAL_NC_CONFIG",
                "WORLD_SIZE",
                "LOCAL_WORLD_SIZE",
                "RANK",
                "LOCAL_RANK",
            )
        }
        return object()

    monkeypatch.setattr(flux_common, "build_pipeline", fake_build_pipeline)
    monkeypatch.setattr(flux_common, "pipeline_artifacts_ready", lambda pipe: True)

    flux_common.compile_serving_artifact(source, profile, spec, target)

    assert captured["model_path_override"] == source.pinned_model_path
    assert captured["resolved_source_id"] == source.resolved_source_id
    assert captured["compiled_path_override"] == str(staging)
    assert captured["force_compile"] is True
    assert captured["environment"] == {
        "NEURON_RT_VISIBLE_CORES": "0,1,2,3",
        "NEURON_RT_NUM_CORES": "4",
        "NEURON_RT_VIRTUAL_CORE_SIZE": None,
        "NEURON_LOGICAL_NC_CONFIG": None,
        "WORLD_SIZE": "1",
        "LOCAL_WORLD_SIZE": "1",
        "RANK": "0",
        "LOCAL_RANK": "0",
    }
    assert os.environ["NEURON_RT_VISIBLE_CORES"] == "7"
    assert os.environ["NEURON_RT_NUM_CORES"] == "8"
    assert os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] == "2"
    assert os.environ["NEURON_LOGICAL_NC_CONFIG"] == "1"
    assert os.environ["WORLD_SIZE"] == "8"


def test_flux_compile_identity_uses_probe_mode_not_runtime_speedup(monkeypatch, tmp_path):
    source_path = tmp_path / "snapshots" / ("a" * 40)
    source_path.mkdir(parents=True)
    source = ResolvedModelSource(
        "hf_snapshot",
        "black-forest-labs/FLUX.1-dev",
        "main",
        str(source_path),
        "a" * 40,
    )
    profile = ServingProfile(
        model_id=source.model_id,
        model_type="flux",
        height=1024,
        width=1024,
        num_frames=None,
        parallel=DiffletParallelConfig(tp_degree=4),
    )
    monkeypatch.setattr(flux_common, "_torch_bfloat16", lambda: "bfloat16")

    baseline = flux_common.build_compile_plan(source, profile)[0].identity
    adaptive_a = flux_common.build_compile_plan(source, replace(profile, teacache_speedup=1.3))[
        0
    ].identity
    adaptive_b = flux_common.build_compile_plan(source, replace(profile, teacache_speedup=1.8))[
        0
    ].identity

    assert adaptive_a == adaptive_b
    assert adaptive_a != baseline


@pytest.mark.parametrize("steps,expected", [(28, True), (4, False)])
def test_flux_serving_selects_teacache_per_request_steps(monkeypatch, steps, expected):
    class _Generator:
        def manual_seed(self, seed):
            return self

    class _Image:
        def save(self, buffer: io.BytesIO, *, format: str):
            buffer.write(b"png")

    captured = {}

    def pipe(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(images=[_Image()])

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(Generator=_Generator))
    profile = ServingProfile(
        model_id="black-forest-labs/FLUX.1-dev",
        model_type="flux",
        height=1024,
        width=1024,
        num_frames=None,
        parallel=DiffletParallelConfig(tp_degree=4),
        teacache_speedup=1.5,
        teacache_calibration_data=TeaCacheCalibration(
            model="flux",
            shape_label="1024x1024",
            num_steps=28,
            poly_coef=(0.0, 1.0),
            threshold=0.1,
        ),
    )
    orchestrator = FluxServingOrchestrator()
    orchestrator.active_profile = profile
    orchestrator.pipe = pipe
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

    asyncio.run(orchestrator.generate(request, WorkerRequestContext.with_timeout("request", 1.0)))

    assert captured["teacache_enabled"] is expected


@pytest.mark.parametrize("adaptive", [False, True])
def test_flux_smoke_runs_real_inference_and_uses_profile_steps(monkeypatch, adaptive):
    from PIL import Image

    class _Generator:
        def manual_seed(self, seed):
            return self

    captured = {}
    profile = ServingProfile(
        model_id="black-forest-labs/FLUX.1-dev",
        model_type="flux",
        height=64,
        width=96,
        num_frames=None,
        parallel=DiffletParallelConfig(tp_degree=4),
        teacache_speedup=1.5 if adaptive else None,
        teacache_calibration_data=(
            TeaCacheCalibration(
                model="flux",
                shape_label="64x96",
                num_steps=28,
                poly_coef=(0.0, 1.0),
                threshold=0.1,
            )
            if adaptive
            else None
        ),
    )

    def pipe(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(images=[Image.new("RGB", (profile.width, profile.height))])

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(Generator=_Generator))
    orchestrator = FluxServingOrchestrator()
    orchestrator.active_profile = profile
    orchestrator.pipe = pipe

    orchestrator.smoke()

    assert captured["num_inference_steps"] == (28 if adaptive else 4)
    assert captured["teacache_enabled"] is adaptive
    assert captured["height"] == profile.height
    assert captured["width"] == profile.width
