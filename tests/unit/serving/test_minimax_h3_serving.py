"""MiniMax-H3 resident serving: metadata, profile rules, runtime plan, request validation, the
audio+video MP4, and the native-neuron pieces of the resident worker and the video service."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.serving.errors import DiffletServingError
from difflet.serving.models import minimax_h3
from difflet.serving.types import (
    ArtifactSet,
    DiffletGenerateRequest,
    FileOutputTarget,
    ResolvedModelSource,
    ResolvedRuntimeBundle,
    ServingProfile,
    StageRuntimeSpec,
    VideoGenerateOptions,
)

_ULYSSES2 = DiffletParallelConfig(tp_degree=4, cp_degree=2, cp_mode="ulysses")


def _profile(tmp_path: Path, **overrides) -> ServingProfile:
    values = dict(
        model_id="MiniMaxAI/MiniMax-H3",
        model_type="minimax_h3",
        height=768,
        width=1344,
        num_frames=175,
        parallel=_ULYSSES2,
        cache_dir=str(tmp_path / "cache"),
        revision=None,
        dtype="bfloat16",
        output_modality="video",
        output_mime_type="video/mp4",
        output_fps=24,
        host_vae=False,
    )
    values.update(overrides)
    return ServingProfile(**values)


def _runtime(tmp_path: Path, profile: ServingProfile | None = None) -> ResolvedRuntimeBundle:
    profile = profile or _profile(tmp_path)
    pipeline = minimax_h3._pipeline_definition()
    return ResolvedRuntimeBundle(
        profile=profile,
        source=ResolvedModelSource(
            source_kind="hf_snapshot",
            model_id=profile.model_id,
            requested_revision=None,
            pinned_model_path=str(tmp_path / "snapshot"),
            resolved_source_id="a" * 40,
        ),
        pipeline_definition=pipeline,
        runtime_plan=minimax_h3.build_runtime_plan(profile, pipeline),
        compile_specs=(),
        artifacts=ArtifactSet(()),
    )


@pytest.fixture(autouse=True)
def _neuron_cores(monkeypatch):
    # build_runtime_plan resolves the inherited core list; give it eight.
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "40-47")
    monkeypatch.delenv(minimax_h3.TEXT_BUCKET_ENV, raising=False)
    monkeypatch.delenv(minimax_h3.KEYFRAMES_ENV, raising=False)


def _request(**overrides) -> DiffletGenerateRequest:
    video = overrides.pop("video", None) or VideoGenerateOptions(
        num_frames=175, fps=24, output_target=FileOutputTarget(staging_path="/tmp/x.part.mp4"))
    values = dict(request_id="r1", model="MiniMaxAI/MiniMax-H3", prompt="a red fox",
                  height=768, width=1344, num_inference_steps=30, guidance_scale=1.0, seed=42,
                  output_format="mp4", video=video)
    values.update(overrides)
    return DiffletGenerateRequest(**values)


class _Tokenizer:
    """One token per word, like the validator's use of the real tokenizer."""

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": text.split()}


def _validator(tmp_path, **env):
    runtime = _runtime(tmp_path)
    validator = minimax_h3.MiniMaxH3ServingRequestValidator(runtime)
    validator._tokenizer = _Tokenizer()
    return validator


# --------------------------------------------------------------------------- metadata


def test_metadata_is_one_opaque_stage_with_an_audio_track():
    from difflet.common.registry.minimax_h3 import serving_metadata
    from difflet.serving.model_registry import _SERVING_METADATA, _ensure_builtin_serving_models_registered

    meta = serving_metadata()
    assert meta.model_type == "minimax_h3" and meta.checkpoint_ids == ("MiniMaxAI/MiniMax-H3",)
    assert meta.output_modality == "video" and meta.default_fps == 24
    assert meta.output_has_audio and not meta.default_host_vae
    assert [s.stage_id for s in meta.pipeline_definition.stages] == ["pipeline"]
    _ensure_builtin_serving_models_registered()
    assert _SERVING_METADATA["minimax_h3"].output_has_audio


def test_other_video_models_stay_silent():
    from difflet.common.registry import hunyuan_video, ltx_2, wan

    for module in (wan, hunyuan_video, ltx_2):
        assert module.serving_metadata().output_has_audio is False


# --------------------------------------------------------------------------- profile


@pytest.mark.parametrize("parallel", [
    DiffletParallelConfig(tp_degree=4),
    _ULYSSES2,
    DiffletParallelConfig(tp_degree=4, cp_degree=4, cp_mode="gather_kv"),
])
def test_profile_accepts_tp_and_context_parallel(tmp_path, parallel):
    minimax_h3._validate_profile(_profile(tmp_path, parallel=parallel))


@pytest.mark.parametrize("overrides, message", [
    (dict(num_frames=176), "17n\\+5"),
    (dict(num_frames=362), "17n\\+5"),
    (dict(height=760), "multiple of 32"),
    (dict(output_fps=30), "24 fps"),
    (dict(host_vae=True), "on the device"),
    (dict(dtype="float32"), "bfloat16"),
    (dict(parallel=DiffletParallelConfig(tp_degree=4, cp_degree=4, cp_mode="ulysses")),
     "divisible by tp\\*cp"),
    (dict(parallel=DiffletParallelConfig(tp_degree=4, cp_degree=2, cp_mode="ring")),
     "ulysses' or 'gather_kv"),
])
def test_profile_rejections(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        minimax_h3._validate_profile(_profile(tmp_path, **overrides))


def test_runtime_plan_is_one_native_neuron_allocation_of_tp_times_cp_cores(tmp_path):
    plan = _runtime(tmp_path).runtime_plan
    (allocation,) = plan.allocations
    assert allocation.world_size == allocation.effective_num_cores == 8
    assert plan.environment.available_core_ids == tuple(range(40, 48))
    (stage,) = plan.stages
    assert stage.placement == "neuron_native" and stage.artifact_id is None
    assert (stage.topology.tp_degree, stage.topology.cp_degree) == (4, 2)


def test_preparer_requires_the_native_neuron_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(minimax_h3, "_backend_name", lambda: "trainium")
    preparer = minimax_h3.MiniMaxH3ServingArtifactPreparer()
    with pytest.raises(ValueError, match="DIFFLET_BACKEND=neuron"):
        preparer.prepare_runtime(_profile(tmp_path), download_policy="never",
                                 compile_policy="never")


def test_text_bucket_env(monkeypatch):
    assert minimax_h3.text_bucket() == 512
    monkeypatch.setenv(minimax_h3.TEXT_BUCKET_ENV, "1024")
    assert minimax_h3.text_bucket() == 1024
    monkeypatch.setenv(minimax_h3.TEXT_BUCKET_ENV, "300")
    with pytest.raises(ValueError):
        minimax_h3.text_bucket()


# --------------------------------------------------------------------------- requests


def test_validator_accepts_a_matching_request(tmp_path):
    _validator(tmp_path).validate(_request())


@pytest.mark.parametrize("overrides, code", [
    (dict(height=512), "profile_mismatch"),
    (dict(model="other/model"), "profile_mismatch"),
    (dict(num_inference_steps=1), "invalid"),
    (dict(seed=-1), "invalid"),
    (dict(guidance_scale=float("nan")), "invalid"),
    (dict(video=VideoGenerateOptions(num_frames=124, fps=24)), "profile_mismatch"),
    (dict(video=VideoGenerateOptions(num_frames=175, fps=24, negative_prompt="blurry")),
     "invalid"),
    (dict(prompt=""), "invalid"),
])
def test_validator_rejections(tmp_path, overrides, code):
    with pytest.raises(DiffletServingError) as info:
        _validator(tmp_path).validate(_request(**overrides))
    assert code in info.value.code


def test_validator_refuses_a_prompt_longer_than_the_bucket(tmp_path):
    validator = _validator(tmp_path)
    validator.validate(_request(prompt="w " * 512))
    with pytest.raises(DiffletServingError) as info:
        validator.validate(_request(prompt="w " * 513))
    assert info.value.code == "prompt_too_long"


# --------------------------------------------------------------------------- media


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs an ffmpeg binary")
def test_write_av_mp4_muxes_a_stereo_track_into_the_parent_target(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("av")
    from difflet.serving.video_media import VideoMediaValidationError, validate_mp4

    frames, height, width, rate = 24, 64, 96, 32000
    video = torch.rand(1, 3, frames, height, width)
    audio = torch.rand(1, 2, rate) * 2 - 1
    target = tmp_path / "out.part.mp4"
    target.touch()
    inode = os.stat(target).st_ino
    media = minimax_h3.write_av_mp4(video, audio, rate, target, fps=24)
    assert (media.width, media.height, media.num_frames) == (width, height, frames)
    assert os.stat(target).st_ino == inode  # written in place, the parent's file
    with pytest.raises(VideoMediaValidationError, match="audio stream"):
        validate_mp4(target, expected_width=width, expected_height=height,
                     expected_num_frames=frames, expected_fps=24)


def test_service_media_check_allows_audio_only_when_told(monkeypatch, tmp_path):
    from difflet.serving import video_service

    seen = []
    monkeypatch.setattr("difflet.serving.video_media.validate_mp4",
                        lambda path, **kw: seen.append(kw["require_silent"]))
    request = _request()
    video_service._validate_media(tmp_path / "x.mp4", request)
    video_service._validate_media(tmp_path / "x.mp4", request, allow_audio=True)
    assert seen == [True, False]


# --------------------------------------------------------------------------- worker


def test_neuron_native_placement_needs_allocation_and_topology_but_no_artifact():
    from difflet.serving.types import ParallelTopology

    topology = ParallelTopology(tp_degree=4, cp_degree=2, world_size=8)
    StageRuntimeSpec("pipeline", "a", topology, None, placement="neuron_native")
    with pytest.raises(ValueError, match="requires allocation and topology"):
        StageRuntimeSpec("pipeline", None, topology, None, placement="neuron_native")


def _apply_env(monkeypatch, tmp_path, *, native: bool, rank: int, count: int):
    from difflet.serving.engines import resident_worker

    # setenv first: monkeypatch records an undo only for a variable it changes, and the code
    # under test exports these names, which must not leak into later tests of the worker.
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "NEURON_RT_NUM_CORES",
                 "NEURON_RT_VISIBLE_CORES", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setattr(resident_worker, "_on_native_neuron", lambda: native)
    monkeypatch.setattr(resident_worker, "_on_tpu", lambda: False)
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "40-47")
    runtime = _runtime(tmp_path)
    if count > 1 and native:
        resident_worker._initialize_replica_runtime(rank, count)
    resident_worker._apply_worker_runtime_environment(runtime)
    return {k: os.environ.get(k) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE",
                                           "LOCAL_WORLD_SIZE", "NEURON_RT_VISIBLE_CORES",
                                           "NEURON_RT_NUM_CORES")}


def test_native_neuron_replicas_get_their_torchrun_identity(monkeypatch, tmp_path):
    env = _apply_env(monkeypatch, tmp_path, native=True, rank=5, count=8)
    assert env == {"RANK": "5", "LOCAL_RANK": "5", "WORLD_SIZE": "8", "LOCAL_WORLD_SIZE": "8",
                   "NEURON_RT_VISIBLE_CORES": ",".join(str(c) for c in range(40, 48)),
                   "NEURON_RT_NUM_CORES": None}


def test_other_backends_keep_the_single_process_environment(monkeypatch, tmp_path):
    env = _apply_env(monkeypatch, tmp_path, native=False, rank=0, count=1)
    assert (env["RANK"], env["WORLD_SIZE"], env["NEURON_RT_NUM_CORES"]) == ("0", "1", "8")


def test_rendezvous_takes_a_fresh_port_per_start_unless_the_operator_set_one(monkeypatch):
    from difflet.serving.engines import resident_worker

    for name in ("MASTER_ADDR", "MASTER_PORT"):  # setenv first, so the exports are undone
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setattr(resident_worker, "_OPERATOR_MASTER_PORT", None)
    resident_worker._export_native_neuron_rendezvous()
    first = os.environ["MASTER_PORT"]
    assert os.environ["MASTER_ADDR"] == "127.0.0.1" and int(first) > 0
    monkeypatch.setattr(resident_worker, "_OPERATOR_MASTER_PORT", "29999")
    resident_worker._export_native_neuron_rendezvous()
    assert os.environ["MASTER_PORT"] == "29999"


# --------------------------------------------------------------------------- resident


def test_text_encoder_mode(monkeypatch):
    from difflet.backends.neuron.minimax_h3 import resident

    monkeypatch.delenv(resident.TEXT_ENCODER_ENV, raising=False)
    assert resident.text_encoder_mode() == "host"
    monkeypatch.setenv(resident.TEXT_ENCODER_ENV, "AUTO")
    assert resident.text_encoder_mode() == "auto"
    monkeypatch.setenv(resident.TEXT_ENCODER_ENV, "gpu")
    with pytest.raises(ValueError):
        resident.text_encoder_mode()


def test_auto_falls_back_to_the_host_once_the_device_runs_out_of_hbm(monkeypatch):
    from difflet.backends.neuron.minimax_h3 import resident

    monkeypatch.setenv(resident.TEXT_ENCODER_ENV, "auto")
    res = resident.NeuronMiniMaxH3Resident(
        model_dir="/m", parallel=_ULYSSES2, height=768, width=1344, num_frames=175,
        text_bucket=512, cache_dir=None, log=lambda m: None)
    calls = []
    monkeypatch.setattr(res, "_encode_on_device", lambda p: calls.append("device") or None)
    monkeypatch.setattr(res, "_encode_on_host", lambda p: calls.append("host") or {"ok": p})
    assert res.encode("a") == {"ok": "a"} and res.encode("b") == {"ok": "b"}
    assert calls == ["device", "host", "host"] and res.text_encoder_mode == "host"
    assert resident._is_hbm_exhaustion(RuntimeError(
        "NRT Execution error occurred on Neuron for operation=neuron::alloc::lazy"))
    assert not resident._is_hbm_exhaustion(ValueError("alloc"))


def test_cancel_check_steps_env(monkeypatch):
    from difflet.backends.neuron.minimax_h3 import resident

    monkeypatch.delenv(resident.CANCEL_CHECK_STEPS_ENV, raising=False)
    assert resident.cancel_check_steps() == 5
    monkeypatch.setenv(resident.CANCEL_CHECK_STEPS_ENV, "2")
    assert resident.cancel_check_steps() == 2
    monkeypatch.setenv(resident.CANCEL_CHECK_STEPS_ENV, "0")
    with pytest.raises(ValueError):
        resident.cancel_check_steps()


@pytest.mark.parametrize("every, stop_at_poll, expected_steps", [
    (5, 0, 0),     # cancelled before the denoise: no step runs
    (5, 2, 10),    # polls: encode, step 5, step 10 -> stops after step 10
    (1, 3, 3),     # every step: encode, 1, 2, 3 -> stops after step 3
])
def test_generate_stops_every_rank_at_an_agreed_check(monkeypatch, every, stop_at_poll,
                                                       expected_steps):
    from difflet.backends.neuron.minimax_h3 import resident
    from difflet.models.minimax_h3 import pipeline

    monkeypatch.setenv(resident.CANCEL_CHECK_STEPS_ENV, str(every))
    res = resident.NeuronMiniMaxH3Resident(
        model_dir="/m", parallel=_ULYSSES2, height=768, width=1344, num_frames=175,
        text_bucket=512, cache_dir=None, log=lambda m: None)
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(res, "encode", lambda p: {
        "encoder_hidden_states": torch.zeros(1, 512, 8), "num_text_tokens": 3,
        "attention_mask": torch.ones(1, 512, dtype=torch.bool)})
    monkeypatch.setattr(res, "modulation_table", lambda steps: None)
    monkeypatch.setattr(pipeline, "load_minimax_h3_schedulers", lambda d: (None, None))
    ran = []

    def fake_denoise(*a, num_inference_steps, step_callback, **kw):
        for index in range(num_inference_steps):
            ran.append(index)
            step_callback(index)
        raise AssertionError("not cancelled")

    monkeypatch.setattr(pipeline, "denoise_minimax_h3_t2va", fake_denoise)
    polls = []

    def should_stop():
        polls.append(1)
        return len(polls) > stop_at_poll

    with pytest.raises(resident.ResidentRequestCancelled):
        res.generate("p", steps=30, seed=0, scratch_dir="/tmp/x", should_stop=should_stop)
    assert len(ran) == expected_steps


def test_serve_cp_modes_open_ulysses_for_minimax_h3_only():
    from difflet.registry import resolve_model
    from difflet.serving.options import build_serving_profile, serving_cp_modes

    assert serving_cp_modes("minimax_h3") == {"gather_kv", "ulysses"}
    for model_type in ("flux", "wan", "hunyuan_video", "qwen_image", "ltx_2"):
        assert serving_cp_modes(model_type) == {"gather_kv", "ring"}
    common = dict(output_modality="image", output_mime_type="image/png", default_fps=None,
                  default_host_vae=False, revision=None, cache_dir=None, tp_degree=None,
                  cp_degree=2, cfg_parallel=None, sp_enabled=None, height=None, width=None,
                  num_frames=None, host_vae=False, clip_placement=None, teacache_cadence=None,
                  teacache_online_delta=None, teacache_speedup=None, teacache_calibration=None)
    flux = resolve_model("black-forest-labs/FLUX.1-dev")
    with pytest.raises(DiffletServingError, match="not served for flux"):
        build_serving_profile(entry=flux, model_id="black-forest-labs/FLUX.1-dev", model_type="flux",
                              cp_mode="ulysses", **common)


def test_request_should_stop_reads_the_real_cancellation_signal_and_deadline():
    from difflet.serving.types import CancellationSignal, WorkerRequestContext

    context = WorkerRequestContext.with_timeout("r1", 60.0, cancellation=CancellationSignal())
    assert minimax_h3.request_should_stop(context) is False
    context.cancellation.cancel()
    assert minimax_h3.request_should_stop(context) is True
    expired = WorkerRequestContext.with_timeout("r2", 0.0, cancellation=CancellationSignal())
    import time

    time.sleep(0.01)
    assert minimax_h3.request_should_stop(expired) is True


# --------------------------------------------------------------------------- FL2VA


def _keyframe(anchor):
    import io

    from PIL import Image

    from difflet.serving.types import KeyframeInput

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buffer, format="PNG")
    return KeyframeInput(anchor=anchor, png=buffer.getvalue())


def _fl2va_request(*anchors, prompt="a red fox"):
    return _request(prompt=prompt, video=VideoGenerateOptions(
        num_frames=175, fps=24, output_target=FileOutputTarget(staging_path="/tmp/x.part.mp4"),
        keyframes=tuple(_keyframe(a) for a in anchors)))


def test_keyframe_env_switches_to_the_fl2va_buckets(monkeypatch):
    assert minimax_h3.num_keyframes() == 0
    monkeypatch.setenv(minimax_h3.KEYFRAMES_ENV, "2")
    assert minimax_h3.text_bucket() == 3072
    monkeypatch.setenv(minimax_h3.TEXT_BUCKET_ENV, "512")
    with pytest.raises(ValueError, match="2048"):
        minimax_h3.text_bucket()
    monkeypatch.setenv(minimax_h3.KEYFRAMES_ENV, "3")
    with pytest.raises(ValueError, match="0, 1 or 2"):
        minimax_h3.num_keyframes()


def test_metadata_takes_keyframes_for_minimax_h3_only():
    from difflet.common.registry.minimax_h3 import serving_metadata

    assert serving_metadata().input_keyframes is True
    from difflet.common.registry.base import ServingModelMetadata

    assert ServingModelMetadata.__dataclass_fields__["input_keyframes"].default is False


def test_validator_counts_the_keyframe_blocks_and_requires_the_server_s_count(tmp_path,
                                                                            monkeypatch):
    with pytest.raises(DiffletServingError, match="text only"):
        _validator(tmp_path).validate(_fl2va_request("first"))
    monkeypatch.setenv(minimax_h3.KEYFRAMES_ENV, "2")
    validator = _validator(tmp_path)
    validator.validate(_fl2va_request("first", "last"))
    with pytest.raises(DiffletServingError, match="takes 2 keyframe"):
        validator.validate(_fl2va_request("first"))
    # 768x1344: 24 * 42 image tokens + 2 + a 2-word label per keyframe = 2024 rows.
    assert validator.keyframe_rows(2, 768, 1344) == 2024
    validator.validate(_fl2va_request("first", "last", prompt="w " * (3072 - 2024)))
    with pytest.raises(DiffletServingError) as info:
        validator.validate(_fl2va_request("first", "last", prompt="w " * (3072 - 2024 + 1)))
    assert info.value.code == "prompt_too_long"


def test_runner_hands_the_keyframes_to_the_resident_by_anchor():
    images = minimax_h3.keyframe_images(_fl2va_request("first", "last"))
    assert set(images) == {"first", "last"} and images["first"].mode == "RGB"
    assert minimax_h3.keyframe_images(_request()) == {}


def test_fl2va_smoke_brings_the_server_s_keyframes(tmp_path):
    frames = minimax_h3.smoke_keyframes(2, _profile(tmp_path))
    assert [f.anchor for f in frames] == ["first", "last"]
    assert minimax_h3.smoke_keyframes(0, _profile(tmp_path)) == ()
