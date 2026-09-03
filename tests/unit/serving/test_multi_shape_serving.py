"""Task 3: one resident worker serving a bucketed shape set (HunyuanVideo)."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import difflet.serving.models._common as video_common
from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.serving.errors import DiffletServingError
from difflet.serving.models import hunyuan_video
from difflet.serving.types import (
    ArtifactBinding,
    ArtifactSet,
    DiffletGenerateRequest,
    ResolvedModelSource,
    ResolvedRuntimeBundle,
    ServingProfile,
    VideoGenerateOptions,
)

SHAPE_A = (64, 96, 9)  # largest (priority)
SHAPE_B = (64, 96, 5)
OUT_OF_SET = (64, 96, 13)


def _profile(tmp_path: Path, *, shapes=(SHAPE_B, SHAPE_A)) -> ServingProfile:
    return ServingProfile(
        model_id="hunyuanvideo-community/HunyuanVideo",
        model_type="hunyuan_video",
        height=SHAPE_A[0],
        width=SHAPE_A[1],
        num_frames=SHAPE_A[2],
        parallel=DiffletParallelConfig(tp_degree=4),
        cache_dir=str(tmp_path / "cache"),
        dtype="bfloat16",
        output_modality="video",
        output_mime_type="video/mp4",
        output_fps=24,
        host_vae=True,
        clip_placement="host",
        shapes=shapes,
    )


def _source(tmp_path: Path) -> ResolvedModelSource:
    return ResolvedModelSource(
        source_kind="hf_snapshot",
        model_id="hunyuanvideo-community/HunyuanVideo",
        requested_revision=None,
        pinned_model_path=str(tmp_path / "snapshots" / ("a" * 40)),
        resolved_source_id="a" * 40,
    )


def _runtime(tmp_path, monkeypatch, profile) -> ResolvedRuntimeBundle:
    source = _source(tmp_path)
    monkeypatch.setattr(
        hunyuan_video, "toolchain_versions", lambda: {"python": "3.10", "neuronx-cc": "test"}
    )
    monkeypatch.setattr(
        video_common,
        "resolve_available_neuron_core_ids",
        lambda *, required_num_cores: tuple(range(8, 8 + required_num_cores)),
    )
    specs = hunyuan_video._compile_specs(source, profile)
    pipeline = hunyuan_video._pipeline_definition(profile)
    plan = hunyuan_video._runtime_plan(profile, specs)
    bindings = tuple(
        ArtifactBinding(
            artifact_id=spec.artifact_id,
            path=tmp_path / "published" / spec.artifact_id,
            manifest_path=(
                tmp_path / "published" / spec.artifact_id / "difflet_generation_manifest.json"
            ),
            identity=spec.identity,
            generation_id=f"g-{spec.artifact_id}",
            content_digest="2" * 64,
        )
        for spec in specs
    )
    return ResolvedRuntimeBundle(
        profile=profile,
        source=source,
        pipeline_definition=pipeline,
        runtime_plan=plan,
        compile_specs=specs,
        artifacts=ArtifactSet(bindings),
    )


def _request(profile: ServingProfile, shape) -> DiffletGenerateRequest:
    return DiffletGenerateRequest(
        request_id="request-1",
        model=profile.model_id,
        prompt="a paper boat on water",
        height=shape[0],
        width=shape[1],
        num_inference_steps=4,
        guidance_scale=6.0,
        seed=42,
        output_format="mp4",
        video=VideoGenerateOptions(num_frames=shape[2], fps=int(profile.output_fps)),
    )


class TestProfileShapeSet:
    def test_canonical_shapes_largest_first(self, tmp_path):
        profile = _profile(tmp_path)
        assert profile.canonical_shapes() == (SHAPE_A, SHAPE_B)
        assert profile.shape_set() == frozenset({SHAPE_A, SHAPE_B})
        assert profile.shape_dicts()[0] == {"height": 64, "width": 96, "num_frames": 9}

    def test_single_shape_profile_has_one_entry_set(self, tmp_path):
        profile = _profile(tmp_path, shapes=None)
        assert profile.canonical_shapes() == ((64, 96, 9),)

    def test_validate_profile_requires_largest_pinned(self, tmp_path):
        bad = replace(_profile(tmp_path), num_frames=SHAPE_B[2])
        with pytest.raises(ValueError, match="largest shape"):
            hunyuan_video._validate_profile(bad)

    def test_validate_profile_checks_every_shape(self, tmp_path):
        bad = _profile(tmp_path, shapes=((64, 96, 9), (64, 96, 6)))  # 6 != 4n+1
        with pytest.raises(ValueError, match="4n\\+1"):
            hunyuan_video._validate_profile(bad)


class TestValidatorMembership:
    def _validator(self, tmp_path, monkeypatch):
        runtime = _runtime(tmp_path, monkeypatch, _profile(tmp_path))
        validator = hunyuan_video.HunyuanVideoServingRequestValidator(runtime)
        validator._tokenizer = lambda *a, **kw: SimpleNamespace(
            input_ids=SimpleNamespace(shape=(1, 10))
        )
        return runtime, validator

    def test_accepts_every_shape_in_the_set(self, tmp_path, monkeypatch):
        runtime, validator = self._validator(tmp_path, monkeypatch)
        validator.validate(_request(runtime.profile, SHAPE_A))
        validator.validate(_request(runtime.profile, SHAPE_B))

    def test_rejects_out_of_set_shape_with_allowed_list(self, tmp_path, monkeypatch):
        runtime, validator = self._validator(tmp_path, monkeypatch)
        with pytest.raises(DiffletServingError) as exc:
            validator.validate(_request(runtime.profile, OUT_OF_SET))
        assert exc.value.code == "profile_mismatch"
        assert "64x96x9" in str(exc.value) and "64x96x5" in str(exc.value)


class TestIdentity:
    def test_denoiser_identity_covers_the_set(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            hunyuan_video, "toolchain_versions", lambda: {"python": "3.10"}
        )
        source = _source(tmp_path)
        multi = hunyuan_video._compile_specs(source, _profile(tmp_path))
        single = hunyuan_video._compile_specs(source, _profile(tmp_path, shapes=None))
        reordered = hunyuan_video._compile_specs(
            source, _profile(tmp_path, shapes=(SHAPE_A, SHAPE_B, SHAPE_B))
        )

        def denoiser(specs):
            return next(s for s in specs if s.component_id == "denoiser").identity

        assert denoiser(multi) != denoiser(single)
        assert denoiser(multi) == denoiser(reordered)
        # llama identity is shape-independent: same artifact serves both sets
        def llama(specs):
            return next(s for s in specs if s.component_id == "llama").identity

        assert llama(multi) == llama(single)


def _build_profile_with_shapes(
    *,
    model_id: str,
    model_type: str,
    output_modality: str,
    shapes: str,
    teacache_speedup: float | None = None,
):
    from difflet.serving.options import build_serving_profile

    is_video = output_modality == "video"
    return build_serving_profile(
        model_id=model_id,
        model_type=model_type,
        entry=SimpleNamespace(
            resolve_shape=lambda **kw: {
                "height": kw.get("height"),
                "width": kw.get("width"),
                "num_frames": kw.get("num_frames"),
            },
            default_parallel=DiffletParallelConfig(tp_degree=4),
        ),
        output_modality=output_modality,
        output_mime_type="video/mp4" if is_video else "image/png",
        default_fps=24 if is_video else None,
        default_host_vae=False,
        revision=None,
        cache_dir=None,
        tp_degree=None,
        cp_degree=None,
        cp_mode=None,
        cfg_parallel=None,
        sp_enabled=None,
        height=None,
        width=None,
        num_frames=None,
        shapes=shapes,
        host_vae=False,
        clip_placement=None,
        teacache_cadence=None,
        teacache_online_delta=None,
        teacache_speedup=teacache_speedup,
        teacache_calibration=None,
    )


class TestServeOptionsPlumbing:
    @pytest.mark.parametrize(
        "model_id,model_type,output_modality,shapes,expected_pin,expected_shapes",
        [
            (
                "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                "wan",
                "video",
                "480x832x5,480x832x9",
                (480, 832, 9),
                ((480, 832, 9), (480, 832, 5)),
            ),
            (
                "black-forest-labs/FLUX.1-dev",
                "flux",
                "image",
                "512x512,1024x1024",
                (1024, 1024, None),
                ((1024, 1024, None), (512, 512, None)),
            ),
            (
                "Qwen/Qwen-Image",
                "qwen_image",
                "image",
                "1024x1024,512x512",
                (1024, 1024, None),
                ((1024, 1024, None), (512, 512, None)),
            ),
        ],
    )
    def test_build_serving_profile_accepts_shapes_for_supported_models(
        self, model_id, model_type, output_modality, shapes, expected_pin, expected_shapes
    ):
        profile = _build_profile_with_shapes(
            model_id=model_id,
            model_type=model_type,
            output_modality=output_modality,
            shapes=shapes,
        )
        assert (profile.height, profile.width, profile.num_frames) == expected_pin
        assert profile.shapes == expected_shapes

    @pytest.mark.parametrize(
        "model_id,model_type,output_modality,shapes,match",
        [
            (
                "Lightricks/LTX-2",
                "ltx_2",
                "video",
                "480x832x9,480x832x5",
                "supported for",
            ),
            (
                "black-forest-labs/FLUX.1-dev",
                "flux",
                "image",
                "1024x1024x9,512x512x9",
                "image serving takes HxW",
            ),
            (
                "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                "wan",
                "video",
                "480x832,480x416",
                "video serving takes HxWxF",
            ),
        ],
    )
    def test_build_serving_profile_rejects_unsupported_shape_requests(
        self, model_id, model_type, output_modality, shapes, match
    ):
        with pytest.raises(DiffletServingError, match=match):
            _build_profile_with_shapes(
                model_id=model_id,
                model_type=model_type,
                output_modality=output_modality,
                shapes=shapes,
            )

    def test_build_serving_profile_rejects_shapes_with_teacache(self):
        with pytest.raises(DiffletServingError, match="TeaCache"):
            _build_profile_with_shapes(
                model_id="black-forest-labs/FLUX.1-dev",
                model_type="flux",
                output_modality="image",
                shapes="1024x1024,512x512",
                teacache_speedup=1.5,
            )
