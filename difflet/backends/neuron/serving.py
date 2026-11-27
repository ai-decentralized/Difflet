"""``difflet serve`` for FLUX.1-dev, Qwen-Image, Wan 2.1, HunyuanVideo and LTX-2 on the native
``neuron`` backend (``DIFFLET_BACKEND=neuron``, torch-neuronx).

Each model stays resident as one ``DiffletPipeline`` per rank: the resident worker spawns one
process per rank of ``tp * cp`` (``supports_torchrun_mpmd``) with RANK/WORLD_SIZE set, every rank
builds and loads the same pipeline that ``difflet generate`` builds (``run_generate_stage``,
``difflet/cli/orchestrators/torchrun.py``), every rank runs every request, and the primary
replica alone writes the PNG or MP4 (the others return an empty output the worker drops). The
VAE decode runs in the same process — the CLI's separate vae launch is a one-shot process with
its own tp1 mesh, which cannot be built beside the tp4 DiT (``parallel_mesh`` is set once per
process). MiniMax-H3 keeps its own adapter (``difflet.serving.models.minimax_h3``).

The request contract comes from each model's ``ModelSpec`` (``difflet/models/<model>/spec.py``)
through the GPU flow's validator. Importing this module must not import torch: the serving
parent process resolves the preparer and the validator itself.
"""

from __future__ import annotations

import dataclasses
import io
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from difflet.backends.gpu.serving import GpuServingRequestValidator, gpu_serving_metadata
from difflet.common.registry.base import ServingModelMetadata
from difflet.models.spec import ModelSpec, gpu_model_spec
from difflet.serving.engines.stage_pipeline import (
    ErasedStageRunner,
    ValidatedStageRunner,
    require_exact_payload,
    stage_result,
)
from difflet.serving.errors import feature_not_supported, profile_mismatch
from difflet.serving.options import CompilePolicy, DownloadPolicy
from difflet.serving.orchestrators.base import resolve_hf_model_source
from difflet.serving.types import (
    ArtifactSet,
    CompileArtifactIdentity,
    DiffletGenerateOutput,
    DiffletGenerateRequest,
    DistributedProcessEnvironment,
    FileBackedGenerateOutput,
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

STAGE_ID = "pipeline"
_MODULE = "difflet.backends.neuron.serving"
#: Read once, at import, by ``ops_impl/attention.py`` and the models' const-max tables
#: (``--attention const-max`` on the CLI exports the same variable to its stages).
_CONSTMAX_ENV = "DIFFLET_EXP_ATTN_CONSTMAX"
#: Semicolon-separated regexes replacing a model's default MXFP8 target layers
#: (``difflet/cli/quant.py:QUANT_TARGETS_ENV``, an experiment knob, honored as the CLI does).
_QUANT_TARGETS_ENV = "DIFFLET_QUANT_TARGETS"

#: The checkpoints served natively, by registry entry name. Wan is 2.1 only: the native Wan
#: app at serving runs one transformer (``enable_transformer_2=False``), so a Wan 2.2 request
#: would silently lose its low-noise expert.
NATIVE_SERVED_CHECKPOINTS: dict[str, tuple[str, ...]] = {
    "flux": ("black-forest-labs/FLUX.1-dev",),
    "qwen_image": ("Qwen/Qwen-Image",),
    "wan": ("Wan-AI/Wan2.1-T2V-14B-Diffusers",),
    "hunyuan_video": ("hunyuanvideo-community/HunyuanVideo",),
    "ltx_2": ("Lightricks/LTX-2",),
}

#: Profile options the native resident path does not run: (profile attribute, flag). The
#: probe-free TeaCache modes (--teacache-cadence, --teacache-online-delta) are served
#: (``native_teacache_kwargs``); the calibrated-adaptive one needs a probe graph the resident
#: apps do not build.
_REFUSED_PROFILE_OPTIONS = (
    ("shapes", "--shapes (bucketed request shapes)"),
    ("teacache_speedup", "--teacache-speedup (adaptive TeaCache; use --teacache-cadence)"),
    ("host_vae", "--host-vae (the decoder placement is fixed per model)"),
    ("torch_compile", "--torch-compile (CUDA only)"),
)


@dataclass(frozen=True, slots=True)
class NeuronInitialPayload(StagePayload):
    pass


@dataclass(frozen=True, slots=True)
class NeuronImagePayload(StagePayload):
    output: DiffletGenerateOutput


@dataclass(frozen=True, slots=True)
class NeuronVideoPayload(StagePayload):
    output: FileBackedGenerateOutput


# --------------------------------------------------------------------- metadata


def neuron_serving_metadata(model_type: str) -> ServingModelMetadata:
    """The GPU flow's metadata for ``model_type`` (same request contract, one opaque stage),
    with this module's factories and the natively served checkpoints."""
    return dataclasses.replace(
        gpu_serving_metadata(_spec(model_type)),
        checkpoint_ids=NATIVE_SERVED_CHECKPOINTS[model_type],
        artifact_preparer_factory=f"{_MODULE}:NeuronServingArtifactPreparer",
        orchestrator_factory=f"{_MODULE}:NeuronServingStageAdapter",
        request_validator_factory=f"{_MODULE}:NeuronServingRequestValidator",
    )


def _spec(model_type: str) -> ModelSpec:
    spec = gpu_model_spec(model_type)
    if spec is None or model_type not in NATIVE_SERVED_CHECKPOINTS:
        raise feature_not_supported(f"{model_type!r} is not served on the native neuron backend")
    return spec


def _spec_for_model_id(model_id: str) -> ModelSpec:
    from difflet.registry import resolve_model

    return _spec(resolve_model(model_id).name)


def _is_primary_replica() -> bool:
    return int(os.environ.get("DIFFLET_REPLICA_RANK", "0")) == 0


def _require_neuron_backend() -> None:
    from difflet.backends.registry import current_backend

    if current_backend() != "neuron":
        raise ValueError("native neuron serving needs DIFFLET_BACKEND=neuron")


def refuse_unsupported_profile(profile: ServingProfile, spec: ModelSpec) -> None:
    for attribute, label in _REFUSED_PROFILE_OPTIONS:
        value = getattr(profile, attribute, None)
        if value not in (None, False, (), []):
            raise feature_not_supported(
                f"[{spec.model_type}/neuron serve] {label} is not supported by native serving"
            )
    # HunyuanVideo's profile defaults CLIP to "host", which is where the native app runs it.
    if profile.clip_placement not in (None, "host"):
        raise feature_not_supported(
            f"[{spec.model_type}/neuron serve] --clip-placement {profile.clip_placement}: the "
            "native app encodes CLIP on the host")


# --------------------------------------------------------------------- per model


def native_application_kwargs(model_type: str, spec: ModelSpec) -> dict[str, Any]:
    """The application kwargs of the CLI's default native generate, with the decoder loaded in
    the same process (``difflet/cli/orchestrators/<model>.py``, ``_torchrun_application_kwargs``).
    ``text_seq_len`` is the spec's prompt budget, so the validator and the DiT agree."""
    if model_type == "flux":
        # Every rank holds a VAE copy; rank 0 alone decodes (TorchNeuronFluxApplication).
        return {"decode_vae": True}
    common = {
        "text_seq_len": spec.prompt_budgets[0].max_tokens,
        "batch_size": 1,
        "exec_mode": None,
        "encoder_threads": 12,
    }
    if model_type == "qwen_image":
        # shortcut: the decoder runs on every rank (enable_vae_decoder); its HBM beside the tp4
        # DiT is unmeasured (fits trn3's ~36 GB/core on paper); decode on rank 0 only if it OOMs.
        return {**common, "enable_text_encoder": True, "enable_transformer": True,
                "enable_vae_decoder": True}
    if model_type == "wan":
        # shortcut: as Qwen-Image — every rank decodes; 81 frames decode in fp32 (~12 GiB).
        return {**common, "enable_text_encoder": True, "enable_transformer": True,
                "enable_transformer_2": False, "enable_vae_decoder": True,
                "text_encoder_device": "neuron"}
    if model_type == "hunyuan_video":
        # The CLI default: Llama/CLIP on rank 0's host, the VAE decodes on rank 0's core.
        return {**common, "text_encoder_device": "host", "enable_vae_decoder": True,
                "vae_dtype": "bfloat16", "vae_exec_mode": "compile"}
    if model_type == "ltx_2":
        # Gemma on the device (the CLI's placement at tp4); the video/audio VAEs and the
        # vocoder on rank 0's host (enable_decode_components), as the CLI's host decode.
        return {**common, "enable_text_encoder": True, "enable_transformer": True,
                "enable_decode_components": True, "text_encoder_device": "neuron"}
    raise feature_not_supported(f"{model_type!r} is not served on the native neuron backend")


def native_teacache_kwargs(profile: ServingProfile) -> dict[str, Any]:
    """The probe-free TeaCache mode the profile asks for, under the application keys the CLI's
    native generate passes (``_torchrun_application_kwargs``): ``teacache_cadence`` or
    ``teacache_online_delta_alpha``; empty when TeaCache is off. Each pipeline resets its
    controller at the start of every denoise call, so one resident app serves many requests
    with the same skip schedule. Option resolution has already refused a model or backend
    without the controller (``difflet.serving.options._validate_probe_free_teacache``)."""
    kwargs: dict[str, Any] = {}
    if profile.teacache_cadence is not None:
        kwargs["teacache_cadence"] = int(profile.teacache_cadence)
    if profile.teacache_online_delta is not None:
        kwargs["teacache_online_delta_alpha"] = float(profile.teacache_online_delta)
    return kwargs


def native_quant_spec(profile: ServingProfile):
    """The profile's MXFP8 spec, or None for bf16: the recipe `difflet compile --quant mxfp8`
    uses (``difflet/cli/quant.py:quant_spec``), at the profile's tp, because a row-parallel
    weight is quantized per rank K range (tp2 x CFG2 and tp4 are different directories), and
    with ``DIFFLET_QUANT_TARGETS`` when set. The serving parent (which writes the directory)
    and every rank (which reads it) build it with this one function."""
    if profile.quant is None:
        return None
    from difflet.quant.spec import DEFAULT_COMPUTE
    from difflet.quant.targets import default_spec

    overrides = {}
    targets = os.environ.get(_QUANT_TARGETS_ENV)
    if targets:
        overrides["targets"] = tuple(t for t in targets.split(";") if t)
    return default_spec(profile.model_type, compute=DEFAULT_COMPUTE,
                        tp_degree=int(profile.parallel.tp_degree), **overrides)


def native_quant_kwargs(profile: ServingProfile) -> dict[str, Any]:
    """``{"quant": spec dict}`` for the resident app (every native app's ``quant`` option,
    resolved by ``backends/neuron/core/quant.py:resolve_quant`` from the compile cache, read
    only), or empty for bf16."""
    spec = native_quant_spec(profile)
    return {} if spec is None else {"quant": spec.to_dict()}


def _quant_source_dir(model_type: str, model_dir: str) -> str:
    """The DiT directory the rank's app quantizes from, so the parent writes the directory
    ``resolve_quant`` will look up: ``<snapshot>/transformer``; for Wan the directory its app
    loads in bf16 (``resolve_component_dir``: the bf16 copy `difflet compile` made when there is
    one, else the hub's fp32 directory), as ``wan/neuron_application.py`` resolves it."""
    transformer = os.path.join(model_dir, "transformer")
    if model_type != "wan":
        return transformer
    import torch

    from difflet.backends.neuron.core.weights_bf16 import resolve_component_dir

    return resolve_component_dir(transformer, torch.bfloat16)


def prepare_native_quant(profile: ServingProfile, model_dir: str) -> str | None:
    """Write the quantized DiT under the compile cache in the serving parent, before any rank
    starts (the ranks only read it): a valid directory from an earlier start or a
    `difflet compile --quant mxfp8` at the same tp is reused, a missing or stale one written
    (``ensure_quantized_dir``). Returns its path, None for bf16."""
    spec = native_quant_spec(profile)
    if spec is None:
        return None
    from difflet.quant.checkpoint import prepare_quantized_checkpoint

    source = _quant_source_dir(profile.model_type, model_dir)
    start = time.monotonic()
    try:
        quantized = prepare_quantized_checkpoint(source, spec, create=True)
    except (ValueError, FileNotFoundError) as exc:
        raise feature_not_supported(f"--quant mxfp8: transformer: {exc}") from exc
    print(f"[difflet serve] MXFP8 transformer ({spec.compute}, tp{spec.tp_degree}): {quantized} "
          f"({time.monotonic() - start:.1f} s)", flush=True)
    return str(quantized)


def native_pipeline_kwargs(model_type: str, profile: ServingProfile, *, revision: str | None
                           ) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model_type": model_type,
        "height": int(profile.height),
        "width": int(profile.width),
        "revision": revision,
    }
    if profile.num_frames is not None:
        kwargs["num_frames"] = int(profile.num_frames)
    return kwargs


def native_request_kwargs(model_type: str, request: DiffletGenerateRequest,
                          profile: ServingProfile, *, primary: bool) -> dict[str, Any]:
    """``pipe(**kwargs)`` for one request, without the generator (the adapter adds a
    ``torch.Generator`` seeded identically on every rank). Each application takes a different
    set (``difflet/models/<model>/neuron_application.py``); an unknown kwarg is a TypeError."""
    video = request.video
    kwargs: dict[str, Any] = {
        "prompt": request.prompt,
        "num_inference_steps": int(request.num_inference_steps),
    }
    if model_type == "flux":
        kwargs.update(height=int(profile.height), width=int(profile.width),
                      guidance_scale=float(request.guidance_scale), output_type="pil")
    elif model_type == "qwen_image":
        kwargs["output_type"] = "pt"  # no guidance: the DiT has guidance_embeds=False
    elif model_type == "wan":
        kwargs.update(height=int(profile.height), width=int(profile.width),
                      num_frames=int(profile.num_frames),
                      guidance_scale=float(request.guidance_scale), output_type="pt")
        if video is not None and video.negative_prompt is not None:
            kwargs["negative_prompt"] = video.negative_prompt
    elif model_type == "hunyuan_video":
        kwargs.update(guidance_scale=float(request.guidance_scale), output_type="pt")
    elif model_type == "ltx_2":
        # Only rank 0 holds the decoders; "pt" raises on the others.
        kwargs.update(guidance_scale=float(request.guidance_scale),
                      output_type="pt" if primary else "latent")
        if video is not None and video.negative_prompt is not None:
            kwargs["negative_prompt"] = video.negative_prompt
    else:
        raise feature_not_supported(f"{model_type!r} is not served on the native neuron backend")
    return kwargs


def encode_output(model_type: str, out: Any, request: DiffletGenerateRequest):
    """The primary replica's pipeline output as the served PNG or MP4."""
    from difflet.serving.models._common import encode_video_tensor

    if model_type == "flux":
        return _png(out.images[0])
    if model_type == "qwen_image":
        return _png(_qwen_pil(out.images))
    frames = out.frames.detach().to(device="cpu").float()
    if model_type == "ltx_2":  # (B, F, C, H, W) in [0, 1]; the audio track is not served
        return encode_video_tensor(frames.clamp(0.0, 1.0), request, layout="BFCHW",
                                   value_range="zero_to_one")
    # Wan, HunyuanVideo: (B, C, F, H, W) in [-1, 1]
    return encode_video_tensor(frames.clamp(-1.0, 1.0), request, layout="BCTHW",
                               value_range="minus_one_to_one")


def _qwen_pil(images):
    """``[1, 3, H, W]`` in [-1, 1] -> PIL, the CLI's mapping (``qwen_image._write_image``)."""
    import torch
    from PIL import Image

    x = images.detach().to(device="cpu").float()[0]
    x = ((x * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)
    return Image.fromarray(x.permute(1, 2, 0).numpy())


def _png(image) -> DiffletGenerateOutput:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return DiffletGenerateOutput(data=buf.getvalue(), mime_type="image/png", output_format="png")


def _empty_output(spec: ModelSpec, request: DiffletGenerateRequest):
    """A non-primary replica's output: the worker drops it (only rank 0 replies)."""
    if spec.output != "video":
        return DiffletGenerateOutput(data=b"", mime_type="image/png", output_format="png")
    video = request.video
    return FileBackedGenerateOutput(
        path="", mime_type="video/mp4", output_format="mp4", size_bytes=0,
        width=request.width, height=request.height, num_frames=video.num_frames,
        fps=float(video.fps), duration_s=0.0)


# --------------------------------------------------------------------- runtime plan


def native_runtime_plan(profile: ServingProfile, spec: ModelSpec, source_id: str) -> RuntimePlan:
    """One resident allocation of ``tp * cp`` cores; the worker spawns one process per rank,
    each bound to its own core of ``NEURON_RT_VISIBLE_CORES`` (as MiniMax-H3's plan)."""
    from difflet.common.neuron_cores import resolve_available_neuron_core_ids

    world = int(profile.world_size)
    parallel = profile.parallel
    inputs = {
        "backend": "neuron",
        "model_type": profile.model_type,
        "source_id": source_id,
        "height": int(profile.height),
        "width": int(profile.width),
        "tp_degree": int(parallel.tp_degree),
        "cp_degree": int(parallel.cp_degree),
    }
    if profile.num_frames is not None:
        inputs["num_frames"] = int(profile.num_frames)
    # Additive only: a default layout keeps the identity it had before these axes were served.
    if parallel.cfg_parallel_enabled:
        inputs["cfg_parallel"] = True
    if parallel.sp_enabled:
        inputs["sp"] = True
    if parallel.cp_mode != "gather_kv":
        inputs["cp_mode"] = str(parallel.cp_mode)
    if profile.attention is not None:  # additive: the default kernel keeps the old identity
        inputs["attention"] = str(profile.attention)
    if profile.quant is not None:  # additive: a bf16 profile keeps its identity
        inputs["quant"] = str(profile.quant)
        targets = os.environ.get(_QUANT_TARGETS_ENV)
        if targets:
            inputs["quant_targets"] = targets
    distributed = DistributedProcessEnvironment(1, 1, 0, 0)
    allocation = WorkerAllocationSpec(
        allocation_id=f"{spec.short_name}-neuron-resident",
        requested_num_cores=world,
        effective_num_cores=world,
        world_size=world,
    )
    return RuntimePlan(
        mode="resident",
        profile_identity=CompileArtifactIdentity.from_cache_inputs(inputs).digest,
        environment=RuntimeEnvironment(
            available_core_ids=resolve_available_neuron_core_ids(required_num_cores=world),
            num_cores_override=None,
            virtual_core_size_override=None,
            logical_nc_config_override=None,
            inherited_distributed=distributed,
            child_distributed=distributed,
        ),
        allocations=(allocation,),
        stages=(
            StageRuntimeSpec(
                stage_id=STAGE_ID,
                allocation_id=allocation.allocation_id,
                topology=ParallelTopology(tp_degree=int(parallel.tp_degree),
                                          cp_degree=int(parallel.cp_degree), world_size=world),
                artifact_id=None,
                placement="neuron_native",
            ),
        ),
    )


# --------------------------------------------------------------------- preparer


class NeuronServingArtifactPreparer:
    """Resolve the commit-pinned snapshot; nothing is compiled ahead of time — the ranks
    compile or load their NEFFs at load, from the cache under ``--cache-dir``."""

    def __init__(self, model_id: str, revision: str | None = None) -> None:
        self.model_id = model_id
        self.revision = revision
        self.spec = _spec_for_model_id(model_id)
        self.model_type = self.spec.model_type

    def prepare_runtime(self, profile: ServingProfile, *, download_policy: DownloadPolicy,
                        compile_policy: CompilePolicy) -> ResolvedRuntimeBundle:
        del compile_policy  # the NEFF cache is content-keyed; there is nothing to force
        _require_neuron_backend()
        refuse_unsupported_profile(profile, self.spec)
        # Parent-side, before the engine spawns the ranks: they inherit it at start().
        export_attention_environment(profile)
        from difflet.registry import resolve_model

        entry = resolve_model(self.model_id)
        print(f"[difflet serve] resolving {self.spec.display_name} weights for {self.model_id}")
        source = resolve_hf_model_source(self.model_id, revision=self.revision,
                                         download_policy=download_policy,
                                         allow_patterns=entry.download_patterns)
        if profile.quant is not None:
            # The compile root the ranks will use (they inherit it), then the quantized DiT
            # under it, once, here: no rank ever writes one (resolve_quant only reads).
            _export_rank_environment(profile)
            prepare_native_quant(profile, source.pinned_model_path)
        print("[difflet serve] neuron backend: resident pipeline per rank, NEFFs at load")
        return ResolvedRuntimeBundle(
            profile=profile,
            source=source,
            pipeline_definition=neuron_serving_metadata(self.model_type).pipeline_definition,
            runtime_plan=native_runtime_plan(profile, self.spec, source.resolved_source_id),
            compile_specs=(),
            artifacts=ArtifactSet(()),
        )


# --------------------------------------------------------------------- validator


class NeuronServingRequestValidator(GpuServingRequestValidator):
    """The GPU flow's spec-driven contract, plus what the native resident apps do not take."""

    def validate(self, request: DiffletGenerateRequest) -> None:
        super().validate(request)
        video = request.video
        if self.model_type == "wan" and video is not None and video.guidance_scale_2 is not None:
            raise feature_not_supported(
                "guidance_scale_2 is not taken by Wan 2.1: it has one transformer")
        if (self.model_type == "ltx_2" and self.runtime.profile.parallel.cfg_parallel_enabled
                and float(request.guidance_scale) <= 1.0):
            # The LTX-2 app refuses a call without CFG under CFG parallel (its DiT takes only
            # the batch-2 pair); refuse the request here rather than fail it on the ranks. The
            # audio guidance the app also checks defaults to guidance_scale, which serving
            # does not override. Wan runs batch 2 either way, so guidance 1 stays valid there.
            raise profile_mismatch(
                "this LTX-2 server runs --cfg-parallel, which needs classifier-free guidance: "
                "guidance_scale must be > 1")


# --------------------------------------------------------------------- adapter


class NeuronPipelineRunner:
    """The one stage, on every rank: prompt in, PNG or MP4 out on the primary replica."""

    def __init__(self, adapter: "NeuronServingStageAdapter") -> None:
        self.adapter = adapter

    async def execute(self, invocation: StageInvocation[NeuronInitialPayload]
                      ) -> StageExecutionResult:
        started = time.monotonic()
        invocation.context.cancellation.throw_if_cancelled()
        output = self.adapter.generate(invocation.request)
        invocation.context.cancellation.throw_if_cancelled()
        return stage_result(self.adapter.final_payload(output), started_monotonic=started)

    async def shutdown(self) -> None:
        self.adapter.pipe = None


class NeuronServingStageAdapter:
    """``ServingStageAdapter`` for the five models on the native neuron backend."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.spec = _spec_for_model_id(model_id)
        self.model_type = self.spec.model_type
        self.active_profile: ServingProfile | None = None
        self.pipe: Any = None
        self._smoke = None

    @property
    def _final_payload(self) -> type:
        return NeuronVideoPayload if self.spec.output == "video" else NeuronImagePayload

    def final_payload(self, output) -> StagePayload:
        return self._final_payload(output=output)

    async def create_loaded_runners(self, runtime: ResolvedRuntimeBundle
                                    ) -> OrderedDict[str, ErasedStageRunner]:
        profile = runtime.profile
        refuse_unsupported_profile(profile, self.spec)
        _require_neuron_backend()
        _export_rank_environment(profile)
        # Worker-side import: the parent process never loads torch for this adapter.
        from difflet.pipeline.difflet_pipeline import DiffletPipeline

        self.active_profile = profile
        print(f"[difflet serve] loading {self.spec.display_name} on the neuron backend "
              f"(tp{profile.parallel.tp_degree} cp{profile.parallel.cp_degree})", flush=True)
        # from_pretrained prepares the runtime (core binding, NEFF cache, process group) on
        # this rank; load() is collective, so every rank must call it.
        self.pipe = DiffletPipeline.from_pretrained(
            self.model_id,
            local_files_only=True,
            parallel=profile.parallel,
            compile_cache_dir=profile.cache_dir,
            backend="neuron",
            application_kwargs={**native_application_kwargs(self.model_type, self.spec),
                                **native_teacache_kwargs(profile),
                                **native_quant_kwargs(profile)},
            load=False,
            # The commit the parent pinned, so a worker restart loads the snapshot the bundle
            # and the validator's tokenizer describe.
            **native_pipeline_kwargs(self.model_type, profile,
                                     revision=runtime.source.resolved_source_id),
        )
        self.pipe.load(skip_warmup=False)
        check_attention_kernel(profile)
        print(f"[difflet serve] {self.spec.display_name} pipeline resident", flush=True)
        runner = ValidatedStageRunner(NeuronPipelineRunner(self), NeuronInitialPayload,
                                      self._final_payload)
        return OrderedDict(((STAGE_ID, runner),))

    def generate(self, request: DiffletGenerateRequest):
        """One request on this rank. Every rank calls the pipeline with the same kwargs and an
        identically seeded generator (FLUX draws its noise on every rank)."""
        if self.pipe is None or self.active_profile is None:
            raise RuntimeError(f"{self.spec.display_name} pipeline is not loaded")
        import torch

        primary = _is_primary_replica()
        kwargs = native_request_kwargs(self.model_type, request, self.active_profile,
                                       primary=primary)
        kwargs["generator"] = torch.Generator().manual_seed(int(request.seed))
        out = self.pipe(**kwargs)
        if not primary:
            return _empty_output(self.spec, request)
        return encode_output(self.model_type, out, request)

    def initial_payload(self, request: DiffletGenerateRequest) -> NeuronInitialPayload:
        return NeuronInitialPayload()

    def finalize(self, payload: StagePayload):
        return require_exact_payload(
            payload, self._final_payload, boundary=f"{self.spec.display_name} final payload"
        ).output

    def smoke_request(self) -> DiffletGenerateRequest:
        """``spec.smoke_steps`` steps at the resident shape: proves load plus one generate."""
        spec, profile = self.spec, self.active_profile
        if self.pipe is None or profile is None:
            raise RuntimeError(f"{spec.display_name} load did not leave a resident pipeline")
        guidance = (spec.default_guidance_scale if spec.default_guidance_scale is not None
                    else spec.serving_noop_guidance_scale)
        request = dict(request_id="startup-smoke", model=self.model_id, prompt=spec.smoke_prompt,
                       height=profile.height, width=profile.width,
                       num_inference_steps=spec.smoke_steps, guidance_scale=guidance, seed=0)
        if spec.output != "video":
            return DiffletGenerateRequest(**request)
        from difflet.serving.models._common import StartupSmokeTarget

        self._cleanup_smoke()
        self._smoke = StartupSmokeTarget()
        return DiffletGenerateRequest(
            **request, output_format="mp4",
            video=VideoGenerateOptions(num_frames=int(profile.num_frames),
                                       fps=int(profile.output_fps),
                                       output_target=self._smoke.file_target()))

    def validate_smoke_output(self, output) -> None:
        spec, profile = self.spec, self.active_profile
        try:
            if not _is_primary_replica():
                return
            if spec.output == "video":
                from difflet.serving.models._common import validate_smoke_output

                if profile is None or self._smoke is None:
                    raise RuntimeError(f"{spec.display_name} smoke ran without a target")
                validate_smoke_output(output, profile=profile, expected_path=self._smoke.path)
            else:
                from PIL import Image

                if not output.data:
                    raise RuntimeError(f"{spec.display_name} smoke produced empty output")
                size = Image.open(io.BytesIO(output.data)).size
                if profile is None or size != (profile.width, profile.height):
                    raise RuntimeError(
                        f"{spec.display_name} smoke image is {size}, not the profile shape")
            print(f"[difflet serve] {spec.display_name} generation smoke passed", flush=True)
        finally:
            self._cleanup_smoke()

    def reset_request_state(self, outcome: str) -> None:
        if outcome == "error":
            self._cleanup_smoke()

    async def shutdown(self) -> None:
        self._cleanup_smoke()
        self.pipe = None
        self.active_profile = None

    def _cleanup_smoke(self) -> None:
        smoke, self._smoke = self._smoke, None
        if smoke is not None:
            smoke.cleanup()


def export_attention_environment(profile: ServingProfile) -> None:
    """Set ``DIFFLET_EXP_ATTN_CONSTMAX`` for ``--attention const-max`` in the serving parent,
    before the ranks are spawned: the attention module reads it once, at import, so a rank
    cannot pick it up later. A variable already exported without the flag is refused: the
    ranks would run const-max under a profile (and identity) that says cte."""
    if profile.attention == "const-max":
        os.environ[_CONSTMAX_ENV] = "1"
    elif os.environ.get(_CONSTMAX_ENV) == "1":
        raise feature_not_supported(
            f"{_CONSTMAX_ENV}=1 is set but the profile's attention is cte: pass "
            f"--attention const-max (the profile records it) or unset {_CONSTMAX_ENV}")


def check_attention_kernel(profile: ServingProfile) -> None:
    """After a rank has loaded, refuse a const-max mismatch between the profile and the
    attention module, which read the variable at import: a rank that imported it before the
    variable was set would otherwise run cte silently under a const-max profile."""
    from difflet.backends.neuron.ops_impl import attention

    want = profile.attention == "const-max"
    if bool(attention._CONSTMAX) != want:
        raise RuntimeError(
            f"the neuron attention module runs {'const-max' if attention._CONSTMAX else 'cte'} "
            f"but the serving profile asks for {'const-max' if want else 'cte'}: "
            f"{_CONSTMAX_ENV} must be set before the rank imports it")


def _export_rank_environment(profile: ServingProfile) -> None:
    """What the CLI launcher exports to its ranks, set before the runtime is prepared: the
    compile cache under ``--cache-dir`` (as MiniMax-H3's adapter) and the launcher's 3 h
    process-group timeout (a cold load holds peers in a collective for many minutes). An
    operator-exported value wins."""
    from difflet.backends.neuron.runtime import PG_TIMEOUT_ENV, TORCHRUN_PG_TIMEOUT_S

    if profile.cache_dir and not os.environ.get("DIFFLET_COMPILE_CACHE"):
        os.environ["DIFFLET_COMPILE_CACHE"] = str(Path(profile.cache_dir).expanduser().resolve())
    if not os.environ.get(PG_TIMEOUT_ENV, "").strip():
        os.environ[PG_TIMEOUT_ENV] = str(TORCHRUN_PG_TIMEOUT_S)


__all__ = [
    "NATIVE_SERVED_CHECKPOINTS",
    "NeuronServingArtifactPreparer",
    "NeuronServingRequestValidator",
    "NeuronServingStageAdapter",
    "encode_output",
    "native_application_kwargs",
    "native_pipeline_kwargs",
    "native_request_kwargs",
    "native_runtime_plan",
    "neuron_serving_metadata",
]
