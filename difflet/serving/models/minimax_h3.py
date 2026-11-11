"""MiniMax-H3 resident serving on the native neuron backend (``DIFFLET_BACKEND=neuron``).

One opaque stage runs a request end to end on every rank of the worker's mesh
(``NeuronMiniMaxH3Resident``): the conditioner loaded and released per request, the resident
DiT (TP, or TP x Ulysses/gather_kv CP across chips) and the TP-sharded video decoder. The
primary replica decodes the audio on the host and writes the MP4 with its stereo track; the
other replicas run the collectives and return an empty descriptor (the engine keeps the
primary's reply).

The profile is static: one canvas, one frame count and one text bucket (a prompt longer than
the bucket is refused before it reaches the worker). The NEFFs live in the backend's compile
cache under ``--cache-dir``; there is no separate compile artifact, so the first start at a new
profile compiles the blocks (tens of minutes at 768p) and later starts reuse them.
"""

from __future__ import annotations

import asyncio
import math
import os
import shutil
import subprocess
import tempfile
import time
import wave
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from difflet.registry import resolve_model
from difflet.serving.engines.stage_pipeline import (
    ErasedStageRunner,
    ValidatedStageRunner,
    require_exact_payload,
    stage_result,
)
from difflet.serving.errors import invalid_extra_body, profile_mismatch, prompt_too_long
from difflet.serving.models._common import StartupSmokeTarget, require_video_target
from difflet.serving.options import CompilePolicy, DownloadPolicy
from difflet.serving.orchestrators.base import resolve_hf_model_source
from difflet.serving.types import (
    ArtifactSet,
    DiffletGenerateRequest,
    DistributedProcessEnvironment,
    FileBackedGenerateOutput,
    GenerateOutput,
    ParallelTopology,
    ResolvedRuntimeBundle,
    RuntimeEnvironment,
    RuntimePlan,
    ServingProfile,
    StageExecutionResult,
    StageInvocation,
    StagePayload,
    StageRuntimeSpec,
    VideoGenerateOptions,
    WorkerAllocationSpec,
)

_HF_MODEL_ID = "MiniMaxAI/MiniMax-H3"
_MODEL_TYPE = "minimax_h3"
_STAGE_ID = "pipeline"
_FPS = 24
_NUM_HEADS = 56
_CANVAS_MULTIPLE = 32
_MIN_FRAMES, _MAX_FRAMES = 120, 360
_FRAMES_PER_CHUNK, _LATENTS_PER_CHUNK = 17, 5
_MAX_INFERENCE_STEPS = 200
_MAX_SEED = 2**63 - 1
_TEXT_BUCKETS = (128, 256, 512, 1024)
#: The serving profile's one text bucket (prompts up to this many tokens), set at startup.
TEXT_BUCKET_ENV = "DIFFLET_H3_TEXT_BUCKET"
_DEFAULT_TEXT_BUCKET = 512
_UNSUPPORTED_VIDEO_FIELDS = ("guidance_scale_2", "boundary_ratio", "flow_shift", "true_cfg_scale",
                             "negative_prompt")


def text_bucket() -> int:
    raw = os.environ.get(TEXT_BUCKET_ENV, "").strip()
    bucket = int(raw) if raw else _DEFAULT_TEXT_BUCKET
    if bucket not in _TEXT_BUCKETS:
        raise ValueError(f"{TEXT_BUCKET_ENV} must be one of {_TEXT_BUCKETS}, got {bucket}")
    return bucket


def _backend_name() -> str | None:
    try:
        from difflet.backends.registry import current_backend
    except Exception:  # noqa: BLE001 - registry is optional on old installs
        return None
    return current_backend()


def _is_primary_replica() -> bool:
    return int(os.environ.get("DIFFLET_REPLICA_RANK", "0")) == 0


@dataclass(frozen=True, slots=True)
class MiniMaxH3InitialPayload(StagePayload):
    """Nominal input of the one opaque MiniMax-H3 serving stage."""


@dataclass(frozen=True, slots=True)
class MiniMaxH3FinalPayload(StagePayload):
    """The MP4 (video + stereo audio) the stage wrote to the parent's target."""

    output: FileBackedGenerateOutput


# --------------------------------------------------------------------------- profile


def _validate_profile(profile: ServingProfile, *, expected_model_id: str | None = None) -> None:
    if profile.model_type != _MODEL_TYPE:
        raise ValueError("MiniMax-H3 serving profile must use model_type='minimax_h3'")
    if profile.model_id != (expected_model_id or _HF_MODEL_ID):
        raise ValueError("MiniMax-H3 serving profile model ID does not match the adapter")
    if profile.output_modality != "video" or profile.output_mime_type != "video/mp4":
        raise ValueError("MiniMax-H3 serving profile must produce video/mp4")
    if profile.dtype.lower().removeprefix("torch.") not in {"bf16", "bfloat16"}:
        raise ValueError("MiniMax-H3 serving requires bfloat16")
    for value, name in ((profile.height, "height"), (profile.width, "width")):
        if type(value) is not int or value <= 0 or value % _CANVAS_MULTIPLE:
            raise ValueError(f"MiniMax-H3 serving {name} must be a positive multiple of "
                             f"{_CANVAS_MULTIPLE}")
    frames = profile.num_frames
    if (type(frames) is not int or frames % _FRAMES_PER_CHUNK != _LATENTS_PER_CHUNK
            or not _MIN_FRAMES <= frames <= _MAX_FRAMES):
        raise ValueError("MiniMax-H3 serving num_frames must be 17n+5 between 120 and 360 "
                         "(5 to 15 s at 24 fps)")
    if profile.output_fps != _FPS:
        raise ValueError(f"MiniMax-H3 serving requires {_FPS} fps")
    if profile.host_vae:
        raise ValueError("MiniMax-H3 serving decodes the video on the device (no --host-vae)")
    parallel = profile.parallel
    tp, cp = int(parallel.tp_degree), int(parallel.cp_degree)
    if parallel.cfg_parallel_enabled or parallel.sp_enabled or int(parallel.dp_degree) != 1:
        raise ValueError("MiniMax-H3 serving runs without CFG parallelism, SP or DP")
    if tp not in (2, 4, 8) or profile.world_size != tp * cp:
        raise ValueError("MiniMax-H3 serving requires TP 2/4/8 and world_size = tp * cp")
    if cp > 1:
        mode = str(parallel.cp_mode)
        if mode not in ("ulysses", "gather_kv"):
            raise ValueError("MiniMax-H3 serving runs CP as cp_mode 'ulysses' or 'gather_kv'")
        if mode == "ulysses" and _NUM_HEADS % (tp * cp):
            raise ValueError(f"MiniMax-H3 Ulysses needs {_NUM_HEADS} heads divisible by tp*cp")
    if profile.teacache_speedup is not None or profile.teacache_cadence is not None or (
            profile.teacache_online_delta is not None):
        raise ValueError("MiniMax-H3 resident serving does not support TeaCache")


def _require_neuron_backend() -> None:
    if _backend_name() != "neuron":
        raise ValueError("MiniMax-H3 serving runs on the native neuron backend "
                         "(DIFFLET_BACKEND=neuron)")


def build_runtime_plan(profile: ServingProfile, pipeline_definition: Any) -> RuntimePlan:
    """One resident worker allocation of ``tp * cp`` cores: the worker spawns one process per
    rank (the neuron backend is torchrun-MPMD), each bound to its own core."""
    from difflet.common.neuron_cores import resolve_available_neuron_core_ids

    _validate_profile(profile)
    stages = pipeline_definition.stages
    if len(stages) != 1 or stages[0].stage_id != _STAGE_ID:
        raise ValueError("MiniMax-H3 serving requires exactly one opaque 'pipeline' stage")
    world = int(profile.world_size)
    distributed = DistributedProcessEnvironment(1, 1, 0, 0)
    environment = RuntimeEnvironment(
        available_core_ids=resolve_available_neuron_core_ids(required_num_cores=world),
        num_cores_override=None,
        virtual_core_size_override=None,
        logical_nc_config_override=None,
        inherited_distributed=distributed,
        child_distributed=distributed,
    )
    allocation = WorkerAllocationSpec(
        allocation_id="minimax-h3-resident",
        requested_num_cores=world,
        effective_num_cores=world,
        world_size=world,
    )
    parallel = profile.parallel
    return RuntimePlan(
        mode="resident",
        profile_identity="",
        environment=environment,
        allocations=(allocation,),
        stages=(
            StageRuntimeSpec(
                stage_id=_STAGE_ID,
                allocation_id=allocation.allocation_id,
                topology=ParallelTopology(tp_degree=int(parallel.tp_degree),
                                          cp_degree=int(parallel.cp_degree), world_size=world),
                artifact_id=None,
                placement="neuron_native",
            ),
        ),
    )


def _pipeline_definition():
    from difflet.common.registry.minimax_h3 import serving_metadata

    return serving_metadata().pipeline_definition


class MiniMaxH3ServingArtifactPreparer:
    """Resolve the commit-pinned snapshot; the NEFFs live in the backend's compile cache."""

    model_id = _HF_MODEL_ID
    model_type = _MODEL_TYPE

    def __init__(self, model_id: str = _HF_MODEL_ID, revision: str | None = None) -> None:
        self.model_id = model_id
        self.revision = revision

    def prepare_runtime(self, profile: ServingProfile, *, download_policy: DownloadPolicy,
                        compile_policy: CompilePolicy) -> ResolvedRuntimeBundle:
        _validate_profile(profile, expected_model_id=self.model_id)
        _require_neuron_backend()
        if profile.revision != self.revision:
            raise ValueError("MiniMax-H3 preparer revision must match the serving profile")
        text_bucket()  # a bad bucket fails here, before the workers start
        entry = resolve_model(self.model_id, model_type=_MODEL_TYPE)
        print(f"[difflet serve] resolving MiniMax-H3 weights for {self.model_id}")
        source = resolve_hf_model_source(self.model_id, revision=self.revision,
                                         download_policy=download_policy,
                                         allow_patterns=entry.download_patterns)
        pipeline_definition = _pipeline_definition()
        return ResolvedRuntimeBundle(
            profile=profile,
            source=source,
            pipeline_definition=pipeline_definition,
            runtime_plan=build_runtime_plan(profile, pipeline_definition),
            compile_specs=(),
            artifacts=ArtifactSet(()),
        )


# --------------------------------------------------------------------------- requests


class MiniMaxH3ServingRequestValidator:
    """Refuse what the loaded profile cannot run, before the request reaches the worker."""

    model_id = _HF_MODEL_ID
    model_type = _MODEL_TYPE

    def __init__(self, runtime: ResolvedRuntimeBundle) -> None:
        _validate_profile(runtime.profile)
        self.runtime = runtime
        self.model_id = runtime.profile.model_id
        self.bucket = text_bucket()
        self._tokenizer = None

    def preload(self) -> None:
        self._tokenizer_for_runtime()

    def validate(self, request: DiffletGenerateRequest) -> None:
        profile = self.runtime.profile
        if request.model != profile.model_id:
            raise profile_mismatch("request model does not match the loaded MiniMax-H3 profile")
        if (request.height, request.width) != (profile.height, profile.width):
            raise profile_mismatch("request width and height do not match the loaded "
                                   "MiniMax-H3 profile")
        if request.output_format != "mp4":
            raise invalid_extra_body("MiniMax-H3 serving only supports MP4 output")
        if type(request.num_inference_steps) is not int or not (
                2 <= request.num_inference_steps <= _MAX_INFERENCE_STEPS):
            raise invalid_extra_body("MiniMax-H3 num_inference_steps must be an integer "
                                     "between 2 and 200")
        if type(request.guidance_scale) not in (int, float) or not math.isfinite(
                float(request.guidance_scale)):
            raise invalid_extra_body("MiniMax-H3 guidance_scale must be a finite number "
                                     "(it is guidance-distilled and ignores it)")
        if type(request.seed) is not int or not 0 <= request.seed <= _MAX_SEED:
            raise invalid_extra_body(f"MiniMax-H3 seed must be an integer between 0 and "
                                     f"{_MAX_SEED}")
        video = request.video
        if video is None:
            raise invalid_extra_body("MiniMax-H3 serving requires video request options")
        if video.num_frames != profile.num_frames:
            raise profile_mismatch("request num_frames does not match the loaded MiniMax-H3 "
                                   "profile")
        if video.fps != profile.output_fps:
            raise profile_mismatch("request fps does not match the loaded MiniMax-H3 profile")
        for name in _UNSUPPORTED_VIDEO_FIELDS:
            if getattr(video, name, None) is not None:
                raise invalid_extra_body(f"{name} is not supported by MiniMax-H3 serving")
        count = self.prompt_tokens(request.prompt)
        if count < 1:
            raise invalid_extra_body("MiniMax-H3 prompt must produce at least one token")
        if count > self.bucket:
            raise prompt_too_long(f"MiniMax-H3 prompt has {count} tokens; this server's text "
                                  f"bucket is {self.bucket}")

    def prompt_tokens(self, prompt: str) -> int:
        # The conditioner's tokenization: the raw prompt, no special tokens.
        ids = self._tokenizer_for_runtime()(prompt, add_special_tokens=False)["input_ids"]
        return len(ids)

    def _tokenizer_for_runtime(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(Path(self.runtime.source.pinned_model_path) / "tokenizer"))
        return self._tokenizer


# --------------------------------------------------------------------------- media


def write_av_mp4(video, audio, sample_rate: int, path: str | Path, *, fps: int):
    """Encode ``video`` ``(1, 3, F, H, W)`` in ``[0, 1]`` (H.264) and mux ``audio``
    ``(1, C, S)`` in ``[-1, 1]`` (AAC) into ``path``; returns the MP4's validated metadata."""
    import numpy as np

    from difflet.serving.video_media import encode_tensor_to_mp4, validate_mp4

    destination = Path(path)
    with tempfile.TemporaryDirectory(prefix="difflet-h3-av-") as temp:
        silent = Path(temp) / "video.mp4"
        wav_path = Path(temp) / "audio.wav"
        encode_tensor_to_mp4(video, silent, fps=fps, layout="BCTHW", value_range="zero_to_one")
        pcm = np.rint(audio[0].transpose(0, 1).float().clamp(-1, 1).cpu().numpy()
                      * 32767.0).astype("<i2")
        with wave.open(str(wav_path), "wb") as handle:
            handle.setnchannels(int(pcm.shape[1]))
            handle.setsampwidth(2)
            handle.setframerate(int(sample_rate))
            handle.writeframes(pcm.tobytes())
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("MiniMax-H3 serving needs an ffmpeg binary on PATH to mux audio")
        # Into the parent's pre-created target, in place (the file must stay the same inode).
        muxed = Path(temp) / "av.mp4"
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(silent),
                        "-i", str(wav_path), "-c:v", "copy", "-c:a", "aac", "-shortest",
                        "-movflags", "+faststart", str(muxed)], check=True)
        with open(muxed, "rb") as source, open(destination, "r+b") as target:
            target.truncate(0)
            shutil.copyfileobj(source, target)
    return validate_mp4(destination, expected_width=int(video.shape[-1]),
                        expected_height=int(video.shape[-2]),
                        expected_num_frames=int(video.shape[2]), expected_fps=fps,
                        require_silent=False)


def _empty_output(request: DiffletGenerateRequest) -> FileBackedGenerateOutput:
    video = request.video
    target = video.output_target
    return FileBackedGenerateOutput(
        path="", mime_type=target.mime_type, output_format=target.output_format, size_bytes=0,
        width=request.width, height=request.height, num_frames=video.num_frames,
        fps=float(video.fps), duration_s=0.0)


# --------------------------------------------------------------------------- stage


class MiniMaxH3PipelineRunner:
    """One request end to end on every rank; the primary replica writes the MP4."""

    def __init__(self, resident, profile: ServingProfile) -> None:
        _validate_profile(profile)
        self.resident = resident
        self.profile = profile

    async def execute(
        self, invocation: StageInvocation[MiniMaxH3InitialPayload],
    ) -> StageExecutionResult[MiniMaxH3FinalPayload]:
        started = time.monotonic()
        request = invocation.request
        context = invocation.context
        context.cancellation.throw_if_cancelled()
        primary = _is_primary_replica()
        target = require_video_target(request) if primary else request.video.output_target
        with tempfile.TemporaryDirectory(prefix="difflet-h3-tiles-") as scratch:
            # Every rank sees the same scratch name: the tile files cross ranks through it.
            shared = Path(tempfile.gettempdir()) / f"difflet-h3-tiles-{request.request_id}"

            def should_stop() -> bool:
                return (context.cancellation.cancelled()
                        or time.monotonic() > float(context.deadline_monotonic))

            from difflet.backends.neuron.minimax_h3.resident import ResidentRequestCancelled

            try:
                result = self.resident.generate(
                    request.prompt, steps=request.num_inference_steps, seed=request.seed,
                    scratch_dir=shared, should_stop=should_stop)
            except ResidentRequestCancelled as exc:
                # Every rank stopped at the same step; report it the way the engine expects.
                raise asyncio.CancelledError(str(exc)) from exc
            del scratch
        context.cancellation.throw_if_cancelled()
        if not primary:
            return stage_result(MiniMaxH3FinalPayload(_empty_output(request)),
                                started_monotonic=started)
        shutil.rmtree(Path(tempfile.gettempdir()) / f"difflet-h3-tiles-{request.request_id}",
                      ignore_errors=True)
        media = write_av_mp4(result.video, result.audio, result.sample_rate,
                             target.staging_path, fps=int(request.video.fps))
        expected = (self.profile.width, self.profile.height, self.profile.num_frames,
                    float(self.profile.output_fps))
        actual = (media.width, media.height, media.num_frames, float(media.fps))
        if actual != expected:
            raise RuntimeError(f"MiniMax-H3 media does not match the serving profile: expected "
                               f"{expected!r}, got {actual!r}")
        return stage_result(
            MiniMaxH3FinalPayload(FileBackedGenerateOutput(
                path=target.staging_path, mime_type=target.mime_type,
                output_format=target.output_format, size_bytes=media.size_bytes,
                width=media.width, height=media.height, num_frames=media.num_frames,
                fps=media.fps, duration_s=media.duration_s)),
            started_monotonic=started,
        )

    async def shutdown(self) -> None:
        if self.resident is not None:
            self.resident.shutdown()
        self.resident = None


class MiniMaxH3ServingStageAdapter:
    """Bridge the resident MiniMax-H3 model into the generic resident stage engine."""

    model_id = _HF_MODEL_ID
    model_type = _MODEL_TYPE

    def __init__(self, model_id: str = _HF_MODEL_ID) -> None:
        self.model_id = model_id
        self.active_profile: ServingProfile | None = None
        self._smoke: StartupSmokeTarget | None = None

    async def create_loaded_runners(
        self, runtime: ResolvedRuntimeBundle,
    ) -> OrderedDict[str, ErasedStageRunner]:
        from difflet.backends import get_backend
        from difflet.backends.neuron.minimax_h3.resident import NeuronMiniMaxH3Resident

        profile = runtime.profile
        _validate_profile(profile, expected_model_id=self.model_id)
        _require_neuron_backend()
        self.active_profile = profile
        if profile.cache_dir and not os.environ.get("DIFFLET_COMPILE_CACHE"):
            # The NEFF/HLO caches live under --cache-dir, as for `difflet generate`.
            os.environ["DIFFLET_COMPILE_CACHE"] = str(Path(profile.cache_dir).expanduser()
                                                      .resolve())
        get_backend("neuron").prepare_runtime(profile.parallel)
        resident = NeuronMiniMaxH3Resident(
            model_dir=runtime.source.pinned_model_path, parallel=profile.parallel,
            height=profile.height, width=profile.width, num_frames=profile.num_frames,
            text_bucket=text_bucket(), cache_dir=profile.cache_dir,
            log=lambda message: print(message, flush=True))
        print(f"[difflet serve] loading MiniMax-H3 worker profile {profile}", flush=True)
        resident.load()
        runner = ValidatedStageRunner(MiniMaxH3PipelineRunner(resident, profile),
                                      MiniMaxH3InitialPayload, MiniMaxH3FinalPayload)
        print("[difflet serve] MiniMax-H3 worker loaded", flush=True)
        return OrderedDict(((_STAGE_ID, runner),))

    def initial_payload(self, request: DiffletGenerateRequest) -> MiniMaxH3InitialPayload:
        return MiniMaxH3InitialPayload()

    def finalize(self, payload: StagePayload) -> FileBackedGenerateOutput:
        return require_exact_payload(payload, MiniMaxH3FinalPayload,
                                     boundary="MiniMax-H3 final payload").output

    def smoke_request(self) -> DiffletGenerateRequest:
        if self.active_profile is None:
            raise RuntimeError("MiniMax-H3 serving profile is not loaded")
        self._cleanup_smoke()
        profile = self.active_profile
        self._smoke = StartupSmokeTarget()
        print("[difflet serve] running MiniMax-H3 generation smoke (2 steps)", flush=True)
        return DiffletGenerateRequest(
            request_id="startup-smoke", model=profile.model_id,
            prompt="a small red ball rolls slowly across a wooden table",
            height=profile.height, width=profile.width, num_inference_steps=2,
            guidance_scale=1.0, seed=0, output_format="mp4",
            video=VideoGenerateOptions(num_frames=int(profile.num_frames),
                                       fps=int(profile.output_fps),
                                       output_target=self._smoke.file_target()),
        )

    def validate_smoke_output(self, output: GenerateOutput) -> None:
        from difflet.serving.video_media import validate_mp4

        profile, smoke = self.active_profile, self._smoke
        try:
            if not _is_primary_replica():
                return
            if profile is None or smoke is None:
                raise RuntimeError("MiniMax-H3 smoke target is not active")
            if not isinstance(output, FileBackedGenerateOutput) or output.path != str(smoke.path):
                raise RuntimeError("MiniMax-H3 smoke did not write its temporary target")
            validate_mp4(smoke.path, expected_width=profile.width, expected_height=profile.height,
                         expected_num_frames=profile.num_frames, expected_fps=profile.output_fps,
                         require_silent=False)
            print("[difflet serve] MiniMax-H3 generation smoke passed", flush=True)
        finally:
            self._cleanup_smoke()

    def reset_request_state(self, outcome: str) -> None:
        if outcome == "error":
            self._cleanup_smoke()

    async def shutdown(self) -> None:
        self._cleanup_smoke()
        self.active_profile = None

    def _cleanup_smoke(self) -> None:
        smoke, self._smoke = self._smoke, None
        if smoke is not None:
            smoke.cleanup()


__all__ = [
    "MiniMaxH3ServingArtifactPreparer",
    "MiniMaxH3ServingRequestValidator",
    "MiniMaxH3ServingStageAdapter",
    "TEXT_BUCKET_ENV",
    "build_runtime_plan",
    "write_av_mp4",
]
