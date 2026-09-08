"""Qwen-Image shared-worker serving adapter."""

from __future__ import annotations

import io
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from difflet.common.orchestrators import qwen_image as qwen_common
from difflet.serving.artifact_manager import ArtifactPublishTarget, ImmutableArtifactManager
from difflet.serving.engines.stage_pipeline import (
    ErasedStageRunner,
    ValidatedStageRunner,
    require_exact_payload,
    stage_result,
)
from difflet.serving.errors import profile_mismatch, prompt_too_long
from difflet.serving.options import CompilePolicy, DownloadPolicy
from difflet.serving.orchestrators.base import (
    request_uses_teacache,
    resolve_available_neuron_core_ids,
    resolve_hf_model_source,
    validate_guidance_scale,
)
from difflet.serving.types import (
    ArtifactSet,
    DistributedProcessEnvironment,
    DiffletGenerateOutput,
    DiffletGenerateRequest,
    ParallelTopology,
    QwenFinalPayload,
    QwenInitialPayload,
    QwenLatentPayload,
    QwenTextPayload,
    ResolvedRuntimeBundle,
    RuntimeEnvironment,
    RuntimePlan,
    ServingProfile,
    StageRuntimeSpec,
    WorkerAllocationSpec,
    StageExecutionResult,
    StageInvocation,
    StagePayload,
)

_HF_MODEL_ID = "Qwen/Qwen-Image"
_MODEL_TYPE = "qwen_image"
_ENC_SEQ = qwen_common.ENC_SEQ
_TEXT_SEQ_LEN = qwen_common.TEXT_SEQ_LEN
_MAX_GUIDANCE_SCALE = 20.0
_QWEN_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
_QWEN_DROP_IDX = 34

logger = logging.getLogger(__name__)


class QwenTextStageRunner:
    def __init__(self, adapter: "QwenImageServingStageAdapter") -> None:
        self.adapter = adapter

    async def execute(
        self,
        invocation: StageInvocation[QwenInitialPayload],
    ) -> StageExecutionResult[QwenTextPayload]:
        started = time.monotonic()
        values = self.adapter._encode_prompt(invocation.request.prompt)
        return stage_result(
            QwenTextPayload(
                encoder_hidden_states=values["encoder_hidden_states"],
                encoder_hidden_states_mask=values["encoder_hidden_states_mask"],
            ),
            started_monotonic=started,
        )

    async def shutdown(self) -> None:
        self.adapter.text_app = None
        self.adapter.tokenizer = None


class QwenGenerateStageRunner:
    def __init__(self, adapter: "QwenImageServingStageAdapter") -> None:
        self.adapter = adapter

    async def execute(
        self,
        invocation: StageInvocation[QwenTextPayload],
    ) -> StageExecutionResult[QwenLatentPayload]:
        started = time.monotonic()
        inputs = invocation.input
        packed_latents = self.adapter._denoise(
            {
                "encoder_hidden_states": inputs.encoder_hidden_states,
                "encoder_hidden_states_mask": inputs.encoder_hidden_states_mask,
            },
            invocation.request,
        )
        return stage_result(
            QwenLatentPayload(packed_latents=packed_latents),
            started_monotonic=started,
        )

    async def shutdown(self) -> None:
        self.adapter.denoise_app = None


class QwenVaeStageRunner:
    def __init__(self, adapter: "QwenImageServingStageAdapter") -> None:
        self.adapter = adapter

    async def execute(
        self,
        invocation: StageInvocation[QwenLatentPayload],
    ) -> StageExecutionResult[QwenFinalPayload]:
        started = time.monotonic()
        return stage_result(
            QwenFinalPayload(
                output=DiffletGenerateOutput(
                    data=self.adapter._decode(
                        invocation.input.packed_latents, invocation.request
                    ),
                    mime_type="image/png",
                    output_format="png",
                )
            ),
            started_monotonic=started,
        )

    async def shutdown(self) -> None:
        self.adapter.vae_app = None
        self.adapter.vae_config = None


class QwenImageServingArtifactPreparer:
    model_id = _HF_MODEL_ID
    model_type = _MODEL_TYPE

    def __init__(self, model_id: str = _HF_MODEL_ID, revision: str | None = None) -> None:
        self.model_id = model_id
        self.revision = revision

    def prepare_runtime(
        self,
        profile: ServingProfile,
        *,
        download_policy: DownloadPolicy,
        compile_policy: CompilePolicy,
    ) -> ResolvedRuntimeBundle:
        print(f"[difflet serve] resolving Qwen-Image weights for {self.model_id}")
        source = resolve_hf_model_source(
            self.model_id,
            revision=self.revision,
            download_policy=download_policy,
        )
        if _backend_is_tpu():
            # No compile phase: the TPU stages run eagerly, so there are no
            # artifacts to build, publish or validate. An empty spec/binding
            # pair keeps ResolvedRuntimeBundle's "every required spec has a
            # binding" invariant true rather than special-casing it.
            print("[difflet serve] tpu backend: eager stages, no compile artifacts")
            pipeline = _pipeline_definition()
            return ResolvedRuntimeBundle(
                profile=profile,
                source=source,
                pipeline_definition=pipeline,
                runtime_plan=_runtime_plan(profile, pipeline, ()),
                compile_specs=(),
                artifacts=ArtifactSet(()),
            )
        specs = qwen_common.build_compile_plan(source, profile)
        manager = ImmutableArtifactManager(profile.cache_dir or Path.home() / ".cache" / "difflet")

        def _prepare_binding(spec):
            def _compile_artifact(target: ArtifactPublishTarget) -> None:
                qwen_common.compile_serving_artifact(source, profile, spec, target)

            def _validate_payload(path: Path) -> None:
                qwen_common.validate_compiled_artifact(spec, path)

            return manager.prepare(
                model_type=self.model_type,
                artifact_id=spec.artifact_id,
                identity=spec.identity,
                policy=compile_policy,
                compile_artifact=_compile_artifact,
                validate_payload=_validate_payload,
            )

        bindings = tuple(_prepare_binding(spec) for spec in specs)
        pipeline = _pipeline_definition()
        runtime_plan = _runtime_plan(profile, pipeline, specs)
        return ResolvedRuntimeBundle(
            profile=profile,
            source=source,
            pipeline_definition=pipeline,
            runtime_plan=runtime_plan,
            compile_specs=specs,
            artifacts=ArtifactSet(bindings),
        )


class QwenImageServingRequestValidator:
    model_id = _HF_MODEL_ID
    model_type = _MODEL_TYPE

    def __init__(self, runtime: ResolvedRuntimeBundle) -> None:
        self.model_id = runtime.profile.model_id
        self.runtime = runtime
        self._tokenizer = None

    def preload(self) -> None:
        self._tokenizer_for_runtime()

    def validate(self, request: DiffletGenerateRequest) -> None:
        validate_guidance_scale(request, maximum=_MAX_GUIDANCE_SCALE)
        profile = self.runtime.profile
        # Strict membership in the compiled bucket set: the NxD router only
        # accepts exactly-compiled shapes, so anything else is rejected here
        # (with the allowed set) instead of surfacing a runtime ValueError.
        if (request.height, request.width, None) not in profile.shape_set():
            allowed = [f"{h}x{w}" for h, w, _ in profile.canonical_shapes()]
            raise profile_mismatch(
                f"request shape {request.height}x{request.width} is not in the "
                f"Qwen-Image serving profile's compiled shape set {allowed}"
            )
        encoded = self._tokenizer_for_runtime()(
            _QWEN_TEMPLATE.format(request.prompt),
            padding=False,
            truncation=False,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if int(encoded.input_ids.shape[1]) > _ENC_SEQ:
            raise prompt_too_long(f"Qwen prompt exceeds encoder bucket {_ENC_SEQ}")

    def _tokenizer_for_runtime(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            model_dir = self.runtime.source.pinned_model_path
            self._tokenizer = AutoTokenizer.from_pretrained(str(Path(model_dir) / "tokenizer"))
        return self._tokenizer


#: Encode outcomes broadcast alongside the embeddings so every rank fails the
#: same way. A raise on the encoding rank alone would hang the others on the
#: collective.
_ENCODE_OK = 0
_ENCODE_TOO_LONG = 1
_ENCODE_FAILED = 2


def _backend_is_tpu() -> bool:
    from difflet.backends import current_backend

    try:
        return current_backend() == "tpu"
    except Exception:  # noqa: BLE001 - never fail load on backend lookup
        return False


class QwenImageServingStageAdapter:
    model_id = _HF_MODEL_ID
    model_type = _MODEL_TYPE

    def __init__(self, model_id: str = _HF_MODEL_ID) -> None:
        self.model_id = model_id
        self.active_runtime: ResolvedRuntimeBundle | None = None
        self.active_profile: ServingProfile | None = None
        self.model_dir: str | None = None
        self.text_app: Any = None
        self.tokenizer = None
        self.denoise_app: Any = None
        self.vae_app: Any = None
        self.vae_config: Any = None
        self._runner_ownership_transferred = False
        self._tpu: bool = False
        self._tpu_module: Any = None
        # Probe-free TeaCache controller for the TPU denoise loop (None = off).
        # The Trainium path keeps its controller inside the Qwen pipeline.
        self._tpu_teacache: Any = None
        self._tpu_teacache_last_stats: dict[str, Any] | None = None

    async def create_loaded_runners(
        self,
        runtime: ResolvedRuntimeBundle,
    ) -> OrderedDict[str, ErasedStageRunner]:
        profile = runtime.profile
        if profile.parallel.cp_degree != 1:
            raise RuntimeError("Qwen-Image P0 shared-worker serving requires cp_degree=1")
        _validate_profile_shapes(profile)
        self.active_runtime = runtime
        self.active_profile = profile
        self.model_dir = runtime.source.pinned_model_path
        self._tpu = _backend_is_tpu()
        # The TPU path runs the graph eagerly, so there are no compiled
        # artifacts to validate. Direction A (torch.export -> StableHLO) is
        # implemented but has never been exercised on the real 60-layer model,
        # so serving does not depend on it yet.
        bindings = () if self._tpu else runtime.artifacts.bindings
        manager = ImmutableArtifactManager(profile.cache_dir or Path.home() / ".cache" / "difflet")
        for binding in bindings:
            spec = runtime.require_compile_spec(binding.artifact_id)

            def _validate_runtime_payload(path: Path) -> None:
                qwen_common.validate_compiled_artifact(spec, path)

            manager.validate_binding(
                binding,
                validate_payload=_validate_runtime_payload,
            )
        print("[difflet serve] loading Qwen prompt_encoder stage")
        self._load_text_stage(profile)
        print("[difflet serve] loading Qwen denoiser stage")
        self._load_denoiser_stage(profile)
        print("[difflet serve] loading Qwen decoder stage")
        self._load_vae_stage(profile)
        print("[difflet serve] Qwen shared-worker co-load completed")
        runners: OrderedDict[str, ErasedStageRunner] = OrderedDict(
            (
                (
                    "text",
                    ValidatedStageRunner(
                        QwenTextStageRunner(self), QwenInitialPayload, QwenTextPayload
                    ),
                ),
                (
                    "generate",
                    ValidatedStageRunner(
                        QwenGenerateStageRunner(self), QwenTextPayload, QwenLatentPayload
                    ),
                ),
                (
                    "vae",
                    ValidatedStageRunner(
                        QwenVaeStageRunner(self), QwenLatentPayload, QwenFinalPayload
                    ),
                ),
            )
        )
        self._runner_ownership_transferred = True
        return runners

    def initial_payload(self, request: DiffletGenerateRequest) -> QwenInitialPayload:
        return QwenInitialPayload()

    def finalize(self, payload: StagePayload) -> DiffletGenerateOutput:
        return require_exact_payload(
            payload,
            QwenFinalPayload,
            boundary="Qwen final payload",
        ).output

    def smoke_request(self) -> DiffletGenerateRequest:
        # On TPU the text encoder is deliberately resident on ONE rank, which
        # broadcasts the embeddings (see _load_text_stage_tpu), so requiring it
        # everywhere would fail startup on three replicas out of four. The
        # denoiser and VAE are still per-rank and are still required.
        needs_text_stage = not getattr(self, "_tpu", False) or getattr(
            self, "_tpu_is_encoder", True
        )
        if needs_text_stage and not (self.text_app and self.tokenizer):
            raise RuntimeError("Qwen shared-worker load did not initialize the text stage")
        if not (self.denoise_app and self.vae_app):
            raise RuntimeError("Qwen shared-worker load did not initialize all stages")
        if self.active_profile is None:
            raise RuntimeError("Qwen serving profile is not loaded")
        profile = self.active_profile
        return DiffletGenerateRequest(
            request_id="startup-smoke",
            model=self.model_id,
            prompt="a small red square",
            height=profile.height,
            width=profile.width,
            num_inference_steps=(
                profile.teacache_calibration_data.num_steps
                if profile.teacache_calibration_data is not None
                else 4
            ),
            guidance_scale=1.0,
            seed=0,
        )

    def validate_smoke_output(self, output: DiffletGenerateOutput) -> None:
        if not output.data:
            raise RuntimeError("Qwen shared-worker smoke produced empty output")
        print("[difflet serve] Qwen shared-worker generation smoke passed")

    def reset_request_state(self, outcome: str) -> None:
        return None

    async def shutdown(self) -> None:
        if not self._runner_ownership_transferred:
            self.text_app = None
            self.tokenizer = None
            self.denoise_app = None
            self.vae_app = None
        self.vae_config = None
        self.active_profile = None
        self.active_runtime = None
        self.model_dir = None
        self._runner_ownership_transferred = False

    def _load_text_stage(self, profile: ServingProfile) -> None:
        if self._tpu:
            return self._load_text_stage_tpu(profile)
        import torch
        from neuronx_distributed_inference.models.config import NeuronConfig, TensorCaptureConfig
        from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_text import (
            NeuronQwen2VLTextForCausalLM,
        )
        from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
        from transformers import AutoConfig, AutoTokenizer

        assert self.model_dir is not None
        enc_path = str(Path(self.model_dir) / "text_encoder")
        text_cfg = AutoConfig.from_pretrained(enc_path).text_config
        if getattr(text_cfg, "pad_token_id", None) is None:
            text_cfg.pad_token_id = 0
        neuron_config = NeuronConfig(
            tp_degree=profile.parallel.tp_degree,
            batch_size=1,
            seq_len=_ENC_SEQ,
            torch_dtype=torch.bfloat16,
            on_device_sampling_config={},
            tensor_capture_config=TensorCaptureConfig(modules_to_capture=["norm"]),
        )
        config = NeuronQwen2VLTextForCausalLM.get_config_cls()(
            neuron_config,
            load_config=load_pretrained_config(hf_config=text_cfg),
        )
        self.text_app = NeuronQwen2VLTextForCausalLM(enc_path, config)
        assert self.active_runtime is not None
        self.text_app.load(str(self.active_runtime.artifacts.require("text").path))
        self.tokenizer = AutoTokenizer.from_pretrained(str(Path(self.model_dir) / "tokenizer"))

    def _load_denoiser_stage(self, profile: ServingProfile) -> None:
        if self._tpu:
            return self._load_denoiser_stage_tpu(profile)
        import torch
        from difflet.models.qwen_image.application import NeuronQwenImageApplication

        assert self.model_dir is not None
        teacache_cadence, teacache_online_delta_alpha = _probe_free_teacache(profile)
        self.denoise_app = NeuronQwenImageApplication(
            model_path=self.model_dir,
            parallel=profile.parallel,
            dtype=torch.bfloat16,
            shape=profile.shape_dict(),
            shapes=profile.canonical_shapes() if profile.shapes else None,
            text_seq_len=_TEXT_SEQ_LEN,
            enable_transformer=True,
            teacache_fused=profile.teacache_speedup is not None,
            teacache_speedup=profile.teacache_speedup,
            teacache_calibration=profile.teacache_calibration_data,
            teacache_calibration_path=profile.teacache_calibration,
            # Probe-free modes: host-side controller inside QwenImagePipeline;
            # no probe NEFF, no calibration, no change to the compiled graph.
            teacache_cadence=teacache_cadence,
            teacache_online_delta_alpha=teacache_online_delta_alpha,
        )
        assert self.active_runtime is not None
        # Warm every compiled bucket at startup for multi-shape profiles so the
        # first request at ANY profile shape sees steady-state latency;
        # single-shape profiles keep relying on the startup smoke.
        self.denoise_app.load(
            str(self.active_runtime.artifacts.require("generate").path),
            skip_warmup=not profile.shapes,
        )

    def _load_vae_stage(self, profile: ServingProfile) -> None:
        if self._tpu:
            return self._load_vae_stage_tpu(profile)
        import torch
        from difflet.backends.trainium.core.config import NeuronConfig
        from difflet.backends.trainium.wan.vae import (
            NeuronWanVAEDecoderApplication,
            WanVAEDecoderInferenceConfig,
        )
        from difflet.utils.diffusers_adapter import load_diffusers_config

        assert self.model_dir is not None
        vae_path = str(Path(self.model_dir) / "vae")
        self.vae_config = WanVAEDecoderInferenceConfig(
            neuron_config=NeuronConfig(
                tp_degree=profile.world_size,
                world_size=profile.world_size,
                torch_dtype=torch.bfloat16,
            ),
            load_config=load_diffusers_config(vae_path),
            height=profile.height,
            width=profile.width,
            num_frames=1,
            # Image bucket set -> single-frame video shapes for the reused Wan
            # VAE decoder (same mapping as the CLI vae stage).
            compile_shapes=(
                tuple((h, w, 1) for h, w, _ in profile.canonical_shapes())
                if profile.shapes
                else None
            ),
        )
        self.vae_app = NeuronWanVAEDecoderApplication(model_path=vae_path, config=self.vae_config)
        assert self.active_runtime is not None
        self.vae_app.load(str(self.active_runtime.artifacts.require("vae").path))

    # ------------------------------------------------------------------ TPU
    #
    # The TPU stages mirror the Trainium ones in contract, not implementation.
    # Every Neuron piece has no TPU counterpart: the text encoder is an NxDI
    # model, the denoiser exposes its own pipeline object, and the VAE is a
    # Neuron application. Each is replaced by the stock HuggingFace/diffusers
    # equivalent, with difflet's sharded DiT substituted into the middle.

    def _load_text_stage_tpu(self, profile: ServingProfile) -> None:
        import torch
        import torch_xla.runtime as xr
        from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

        assert self.model_dir is not None
        enc_path = str(Path(self.model_dir) / "text_encoder")
        # ONE replica holds the encoder and broadcasts the embeddings, rather
        # than four computing the same tensor from the same prompt. That is
        # what makes fp32 affordable: one copy is ~28 GiB, four were what OOM'd
        # a 188 GiB host and forced bf16 here. fp32 is worth having -- measured
        # on this VM (an AMD EPYC with no AVX512-BF16, so bf16 matmul is
        # emulated) a 512-token encode is 3.6 s in fp32 against 16.1 s in bf16.
        #
        # It must be XLA *ordinal* 0, not worker index 0: the two are a
        # scrambled mapping (measured: worker 0 -> ordinal 2, 1 -> 0, 2 -> 3,
        # 3 -> 1), and encoding on the wrong one silently broadcasts a zero
        # placeholder to every rank -- it does not fail, it generates a
        # different image.
        self._tpu_is_encoder = int(xr.global_ordinal()) == 0
        if not self._tpu_is_encoder:
            self.text_app = None
            self.tokenizer = None
            return
        self.text_app = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            enc_path, torch_dtype=torch.float32
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(Path(self.model_dir) / "tokenizer")
        )

    def _load_denoiser_stage_tpu(self, profile: ServingProfile) -> None:
        import torch
        import torch_xla
        import torch_xla.core.xla_model as xm

        from difflet.models.qwen_image.entry import create_qwen_image_application

        assert self.model_dir is not None
        app = create_qwen_image_application(
            model_path=self.model_dir,
            parallel=profile.parallel,
            dtype=torch.bfloat16,
            shape=profile.shape_dict(),
            backend="tpu",
            text_seq_len=_TEXT_SEQ_LEN,
        )
        module = app.transformer._prepare_module().to(torch_xla.device())
        xm.mark_step()
        self.denoise_app = app
        self._tpu_module = module
        self._tpu_teacache = _build_tpu_teacache(profile)

    def _load_vae_stage_tpu(self, profile: ServingProfile) -> None:
        import torch
        import torch_xla
        from diffusers import AutoencoderKLQwenImage

        assert self.model_dir is not None
        vae_path = str(Path(self.model_dir) / "vae")
        vae = AutoencoderKLQwenImage.from_pretrained(
            vae_path, torch_dtype=torch.bfloat16
        ).eval()
        # Kept on the HOST between requests and moved to device only for the
        # decode. The stages are sequential so they never need to co-reside —
        # and the headroom does not allow it: after the DiT shard ~6 GiB is
        # free, and the DiT forward's transient footprint at 1024x1024/tp=4
        # consumes very nearly all of it. Parking the VAE's 0.24 GiB on device
        # was enough to turn a working generate into "Attempting to allocate
        # 15.00M. There are 11.25M free."
        self.vae_app = vae
        self.vae_config = vae.config

    def _encode_prompt_tpu(self, prompt: str) -> dict[str, object]:
        """Encode on ordinal 0, then broadcast to every replica.

        Every replica needs the identical tensor, so three of the four encodes
        were pure waste. The broadcast payload is small next to a second of
        host matmul.

        Every replica MUST reach this collective, in the same order, or the
        request deadlocks -- which holds here because the resident-worker
        engine drives all replicas through the same request in lockstep.
        """
        import torch
        import torch_xla
        import torch_xla.core.xla_model as xm

        device = torch_xla.device()
        width = self._tpu_joint_dim()
        states = torch.zeros(1, _TEXT_SEQ_LEN, width, dtype=torch.bfloat16)
        seq, status = 0, _ENCODE_OK

        if getattr(self, "_tpu_is_encoder", True):
            # The encoding rank must NOT raise before the collective below.
            # Every other rank is already committed to reaching it, so an
            # early raise here hangs them until the engine's cancel timeout --
            # and `prompt_too_long` makes that reachable from user input. So
            # failures are converted to a status code, broadcast with the
            # embeddings, and re-raised identically on every rank.
            try:
                if self.text_app is None or self.tokenizer is None:
                    raise RuntimeError("Qwen prompt encoder is not loaded")
                text = _QWEN_TEMPLATE.format(prompt)
                encoded = self.tokenizer(
                    text, padding=False, truncation=False, return_tensors="pt"
                )
                if int(encoded.input_ids.shape[1]) > _ENC_SEQ:
                    status = _ENCODE_TOO_LONG
                else:
                    with torch.no_grad():
                        out = self.text_app(
                            input_ids=encoded.input_ids,
                            attention_mask=encoded.attention_mask,
                            output_hidden_states=True,
                        )
                    # Same slice as the Trainium path: drop the template
                    # prefix, keep the valid tokens, then pad to the compiled
                    # text length.
                    hidden = out.hidden_states[-1]
                    valid = int(encoded.attention_mask.sum())
                    used = hidden[:, _QWEN_DROP_IDX:valid]
                    seq = int(used.shape[1])
                    states[:, :seq] = used.to(torch.bfloat16)
            except Exception:  # noqa: BLE001 - re-raised below on every rank
                logger.exception("qwen.tpu_encode_failed")
                status = _ENCODE_FAILED
                seq = 0

        meta = torch.tensor([seq, status], dtype=torch.int32)
        payload = [states.to(device), meta.to(device)]
        xm.collective_broadcast(payload, root_ordinal=0)
        xm.mark_step()
        states = payload[0].cpu()
        seq, status = (int(v) for v in payload[1].cpu().tolist())

        if status == _ENCODE_TOO_LONG:
            raise prompt_too_long(f"Qwen prompt exceeds encoder bucket {_ENC_SEQ}")
        if status != _ENCODE_OK:
            raise RuntimeError("Qwen prompt encoding failed on the encoder rank")

        # The mask is derived rather than broadcast: bool is an awkward
        # collective dtype and the valid length is one int.
        mask = torch.zeros(1, _TEXT_SEQ_LEN, dtype=torch.bool)
        mask[:, :seq] = True
        return {"encoder_hidden_states": states, "encoder_hidden_states_mask": mask}

    def _tpu_joint_dim(self) -> int:
        """Width of the DiT's text input, known on every rank from the config."""
        if self.denoise_app is not None:
            return int(self.denoise_app.config.joint_attention_dim)
        import json

        assert self.model_dir is not None
        body = json.loads(
            (Path(self.model_dir) / "transformer" / "config.json").read_text()
        )
        return int(body["joint_attention_dim"])

    def _denoise_tpu(self, text: dict[str, object], request: DiffletGenerateRequest):
        import numpy as np
        import torch
        import torch_xla
        import torch_xla.core.xla_model as xm
        from diffusers import FlowMatchEulerDiscreteScheduler

        if self.denoise_app is None or self.active_profile is None:
            raise RuntimeError("Qwen denoiser is not loaded")
        profile = self.active_profile
        device = torch_xla.device()
        config = self.denoise_app.config

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            str(Path(self.model_dir) / "scheduler")
        )
        sc = scheduler.config
        image_seq_len = (profile.height // 16) * (profile.width // 16)
        slope = (sc.max_shift - sc.base_shift) / (sc.max_image_seq_len - sc.base_image_seq_len)
        mu = image_seq_len * slope + (sc.base_shift - slope * sc.base_image_seq_len)
        steps = int(request.num_inference_steps)
        scheduler.set_timesteps(
            sigmas=np.linspace(1.0, 1.0 / steps, steps).tolist(), mu=mu, device="cpu"
        )

        generator = torch.Generator("cpu").manual_seed(int(request.seed))
        latents = torch.randn(
            1, image_seq_len, int(config.in_channels),
            generator=generator, dtype=torch.float32,
        ).to(device)
        states = text["encoder_hidden_states"].to(device, torch.bfloat16)

        # Latents stay on device for the whole loop. Round-tripping them to the
        # host each step cost ~0.7 s on top of a 0.85 s forward, and - the
        # reason this is not merely an optimization - the extra device buffers
        # pushed a 1024x1024/tp=4 forward past the ~6 GiB of HBM left after the
        # weight shard.
        #
        # Staying on device means doing the update here instead of calling
        # scheduler.step, which wants host tensors. For flow matching the Euler
        # update is exactly x + (sigma_next - sigma) * v; the sigmas still come
        # from the scheduler, so the noise schedule remains its own.
        # Every per-step scalar is precomputed as a DEVICE tensor. A Python
        # float gets constant-folded into the graph, so each step becomes a
        # different graph and XLA recompiles all of them: measured 2.45 s/step
        # against 0.25 s of actual device work (~10% TensorCore utilization,
        # confirmed with tpu-info). As tensors, one graph serves the whole loop.
        sigmas = scheduler.sigmas.to(torch.float32)
        steps = len(scheduler.timesteps)
        deltas = [
            (sigmas[i + 1] - sigmas[i]).reshape(1).to(device) for i in range(steps)
        ]
        timesteps = [
            (step / 1000).to(torch.bfloat16).reshape(1).to(device)
            for step in scheduler.timesteps
        ]
        with torch.no_grad():
            latents = _tpu_denoise_loop(
                self._tpu_module,
                latents,
                timesteps,
                deltas,
                states,
                controller=self._tpu_teacache,
                mark_step=xm.mark_step,
            )
        if self._tpu_teacache is not None:
            self._tpu_teacache_last_stats = self._tpu_teacache.stats()
            logger.info("qwen.tpu_teacache stats=%s", self._tpu_teacache_last_stats)
        return latents.cpu()

    def _decode_tpu(self, packed, request) -> bytes:
        import torch
        import torch_xla
        import torch_xla.core.xla_model as xm

        if self.vae_app is None or self.active_profile is None:
            raise RuntimeError("Qwen VAE decoder is not loaded")
        device = torch_xla.device()
        b, seq, _ = packed.shape
        hh, ww = _packed_latent_grid(request.height, request.width, seq)
        z = packed.float().view(b, hh, ww, 16, 2, 2)
        z = z.permute(0, 3, 1, 4, 2, 5).reshape(b, 16, hh * 2, ww * 2).unsqueeze(2)
        mean = torch.tensor(self.vae_config.latents_mean).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.vae_config.latents_std).view(1, -1, 1, 1, 1)
        # Cast after the arithmetic: bf16 * fp32 promotes back to fp32, which
        # the bf16 VAE then rejects.
        z = ((z * std) + mean).to(torch.bfloat16).to(device)
        self.vae_app.to(device)
        try:
            with torch.no_grad():
                img = self.vae_app.decode(z, return_dict=False)[0]
            xm.mark_step()
            img = img.float().cpu()[:, :, 0]
        finally:
            # Hand the HBM back before the next request's denoise loop.
            self.vae_app.to("cpu")
        return _tensor_to_png_bytes((img[0] * 0.5 + 0.5).clamp(0, 1))

    def _encode_prompt(self, prompt: str) -> dict[str, object]:
        import torch

        if self._tpu:
            return self._encode_prompt_tpu(prompt)
        if self.text_app is None or self.tokenizer is None:
            raise RuntimeError("Qwen prompt encoder is not loaded")
        encoded = self.tokenizer(
            _QWEN_TEMPLATE.format(prompt),
            padding=False,
            truncation=False,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if int(encoded.input_ids.shape[1]) > _ENC_SEQ:
            raise prompt_too_long(f"Qwen prompt exceeds encoder bucket {_ENC_SEQ}")
        ti = self.tokenizer(
            _QWEN_TEMPLATE.format(prompt),
            max_length=_ENC_SEQ,
            padding="max_length",
            truncation=False,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids = ti.input_ids.to(torch.int32)
        attn = ti.attention_mask.to(torch.int32)
        out = self.text_app(
            input_ids=input_ids,
            attention_mask=attn,
            position_ids=torch.arange(_ENC_SEQ, dtype=torch.int32).unsqueeze(0),
            sampling_params=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        )
        hs = out.captured_tensors[0].float()
        valid = int(attn.sum())
        dev = hs[:, _QWEN_DROP_IDX:valid]
        seq = dev.shape[1]
        ehs = torch.zeros(1, _TEXT_SEQ_LEN, dev.shape[-1], dtype=torch.bfloat16)
        ehs[:, :seq] = dev.to(torch.bfloat16)
        mask = torch.zeros(1, _TEXT_SEQ_LEN, dtype=torch.bool)
        mask[:, :seq] = True
        return {"encoder_hidden_states": ehs, "encoder_hidden_states_mask": mask}

    def _denoise(self, text: dict[str, object], request: DiffletGenerateRequest):
        import numpy as np
        import torch

        if self._tpu:
            return self._denoise_tpu(text, request)
        if self.denoise_app is None or self.active_profile is None:
            raise RuntimeError("Qwen denoiser is not loaded")
        profile = self.active_profile
        guidance = torch.full([1], float(request.guidance_scale), dtype=torch.bfloat16)
        sched = self.denoise_app.pipeline.scheduler
        sc = sched.config
        # The resident pipeline is pinned to the profile's largest shape, so
        # the request shape drives the schedule and latents explicitly; the
        # runtime router picks the matching compiled bucket by signature.
        image_seq_len = (request.height // 16) * (request.width // 16)
        slope = (sc.max_shift - sc.base_shift) / (sc.max_image_seq_len - sc.base_image_seq_len)
        mu = image_seq_len * slope + (sc.base_shift - slope * sc.base_image_seq_len)
        num_steps = request.num_inference_steps
        use_teacache = request_uses_teacache(profile, num_steps)
        if profile.teacache_speedup is not None and not use_teacache:
            logger.info(
                "Qwen request uses baseline inference fallback_reason=step_mismatch "
                "request_steps=%s calibration_steps=%s",
                num_steps,
                getattr(profile.teacache_calibration_data, "num_steps", None),
            )
        # Adaptive TeaCache is pinned to the calibration's step count, so it is
        # switched per request. The probe-free modes have no such contract: the
        # pipeline's controller follows the request's steps itself, and passing
        # None leaves it in charge (False would silently disable it).
        teacache_enabled: bool | None = (
            use_teacache if profile.teacache_speedup is not None else None
        )
        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps).tolist()
        sched.set_timesteps(sigmas=sigmas, mu=mu, device="cpu")
        torch.manual_seed(request.seed)
        latents = torch.randn(
            (1, 1, 16, request.height // 8, request.width // 8),
            dtype=torch.bfloat16,
        )
        out = self.denoise_app.pipeline(
            latents=latents,
            encoder_hidden_states=text["encoder_hidden_states"],
            encoder_hidden_states_mask=text["encoder_hidden_states_mask"],
            guidance=guidance,
            timesteps=sched.timesteps,
            num_inference_steps=num_steps,
            teacache_enabled=teacache_enabled,
            output_type="latent",
        )
        return out.latents.cpu()

    def _decode(self, packed, request: DiffletGenerateRequest) -> bytes:
        import torch

        if self._tpu:
            return self._decode_tpu(packed, request)
        if self.vae_app is None or self.vae_config is None:
            raise RuntimeError("Qwen VAE decoder is not loaded")
        if self.active_profile is None:
            raise RuntimeError("Qwen serving profile is not loaded")
        b, seq, _ = packed.shape
        hh, ww = _packed_latent_grid(request.height, request.width, seq)
        z = packed.float().view(b, hh, ww, 16, 2, 2)
        z = z.permute(0, 3, 1, 4, 2, 5).reshape(b, 16, hh * 2, ww * 2)
        z = z.unsqueeze(2)
        mean = torch.tensor(self.vae_config.latents_mean).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.vae_config.latents_std).view(1, -1, 1, 1, 1)
        z = (
            (z * std + mean).to(torch.bfloat16)
            if len(self.vae_config.latents_mean)
            else z.to(torch.bfloat16)
        )
        img = self.vae_app(z)
        img = (img[0] if isinstance(img, (tuple, list)) else img).float().cpu()
        img = img[:, :, 0]
        return _tensor_to_png_bytes((img[0] * 0.5 + 0.5).clamp(0, 1))


def _validate_profile_shapes(profile: ServingProfile) -> None:
    for height, width, _frames in profile.canonical_shapes():
        for value, name in ((height, "height"), (width, "width")):
            if type(value) is not int or value <= 0 or value % 16:
                raise ValueError(
                    f"Qwen-Image serving {name} must be a positive integer divisible by 16"
                )
    if profile.shapes:
        largest = profile.canonical_shapes()[0]
        if (profile.height, profile.width, None) != largest:
            raise ValueError(
                "Qwen-Image serving profile height/width must equal the largest "
                "shape of the compiled shape set"
            )


def _probe_free_teacache(profile: Any) -> tuple[int | None, float | None]:
    """``(cadence, online_delta_alpha)`` from a profile, None when off.

    ``getattr`` rather than attribute access on purpose: the benchmark harness
    drives this adapter with a ``SimpleNamespace`` profile that predates these
    fields, and an ``AttributeError`` there would read as a TPU regression.
    """
    cadence = getattr(profile, "teacache_cadence", None)
    alpha = getattr(profile, "teacache_online_delta", None)
    return (
        int(cadence) if cadence is not None else None,
        float(alpha) if alpha is not None else None,
    )


def _build_tpu_teacache(profile: Any):
    """Probe-free TeaCache controller for the TPU loop, or None when off.

    Only the probe-free modes exist on TPU. Adaptive TeaCache needs the block-0
    modulated-input signal, which on Trainium comes from a fused probe NEFF
    that has no TPU counterpart; ``build_serving_profile`` never lets an
    adaptive profile reach the TPU adapter with a cadence/online-delta set, so
    this is not a silent downgrade of an adaptive request.
    """
    cadence, alpha = _probe_free_teacache(profile)
    if cadence is None and alpha is None:
        return None
    from difflet.pipeline.teacache import build_probe_free_controller

    return build_probe_free_controller(
        model=_MODEL_TYPE,
        shape_label=f"{int(profile.height)}x{int(profile.width)}",
        cadence=cadence,
        online_delta_alpha=alpha,
    )


def _tpu_denoise_loop(module, latents, timesteps, deltas, states, *, controller, mark_step):
    """Device-resident flow-matching Euler loop with optional TeaCache skipping.

    Everything the loop touches is a device tensor (see ``_denoise_tpu`` on why
    a Python scalar here recompiles every step), and that includes the
    controller's state: ``record_full_step`` keeps ``noise_pred - prev`` lazy on
    the chip and ``skip_noise_pred`` is one lazy add, so a skipped step costs an
    elementwise op instead of a 20B forward. XLA sees three graph shapes over
    the whole loop — step 0 (no residual yet), a full step, a skipped step —
    and caches each after its first compile.

    Two modes, both probe-free (no per-model signal, no calibration file):

    * fixed cadence — the decision is index-based and forces no device sync,
      so the loop keeps the tracing/execution overlap the natural basis in
      ``benchmark/v5e`` measures.
    * online-delta — ``record_full_step`` reads one scalar back
      (``float(...)``), which is a device sync on every FULL step. On this
      loop that is the same cost as the harness's synced basis; the skipped
      steps still pay nothing.

    The velocity is recorded in fp32, the dtype the latent update already casts
    to, so the cached residual is not quantized to bf16 on the way through.
    """
    import torch

    steps = len(timesteps)
    if controller is not None:
        from difflet.pipeline.teacache import sync_probe_free_num_steps

        # The controller is built at load time, before any request's step
        # count is known; sync so the cooldown protects the schedule's real tail.
        sync_probe_free_num_steps(controller, steps)
        controller.reset()
    for index in range(steps):
        if controller is not None and controller.should_skip(index, None):
            velocity = controller.skip_noise_pred()
        else:
            velocity = module(
                latents.to(torch.bfloat16), timesteps[index], states, None, None
            ).to(torch.float32)
            if controller is not None:
                controller.record_full_step(velocity)
        latents = latents + deltas[index] * velocity
        mark_step()
    return latents


def _packed_latent_grid(height: int, width: int, seq: int) -> tuple[int, int]:
    grid_height = int(height) // 16
    grid_width = int(width) // 16
    if int(seq) != grid_height * grid_width:
        raise ValueError(
            "Qwen packed latent sequence does not match the request shape: "
            f"seq={seq}, expected={grid_height * grid_width} for "
            f"height={height}, width={width}."
        )
    return grid_height, grid_width


def _tensor_to_png_bytes(tensor) -> bytes:
    from torchvision.transforms.functional import to_pil_image

    image = to_pil_image(tensor)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _pipeline_definition():
    from difflet.common.registry.qwen_image import serving_metadata

    return serving_metadata().pipeline_definition


def _runtime_plan(profile: ServingProfile, pipeline, specs) -> RuntimePlan:
    world_size = profile.world_size
    distributed = DistributedProcessEnvironment(1, 1, 0, 0)
    environment = RuntimeEnvironment(
        available_core_ids=resolve_available_neuron_core_ids(required_num_cores=world_size),
        num_cores_override=None,
        virtual_core_size_override=None,
        logical_nc_config_override=None,
        inherited_distributed=distributed,
        child_distributed=distributed,
    )
    allocation = WorkerAllocationSpec(
        allocation_id="qwen-resident",
        requested_num_cores=world_size,
        effective_num_cores=world_size,
        world_size=world_size,
        requested_virtual_core_size=qwen_common.VIRTUAL_CORE_SIZE,
        effective_virtual_core_size=qwen_common.VIRTUAL_CORE_SIZE,
    )
    by_id = {spec.artifact_id: spec for spec in specs}
    stages = tuple(
        StageRuntimeSpec(
            stage_id=stage.stage_id,
            allocation_id=allocation.allocation_id,
            topology=ParallelTopology(
                tp_degree=world_size if stage.stage_id == "vae" else profile.parallel.tp_degree,
                cp_degree=1 if stage.stage_id == "vae" else profile.parallel.cp_degree,
                world_size=world_size,
            ),
            # None when there is no compile plan (the tpu backend runs eagerly).
            artifact_id=(
                by_id[stage.stage_id].artifact_id if stage.stage_id in by_id else None
            ),
            placement="tpu" if not specs else "neuron",
        )
        for stage in pipeline.stages
    )
    profile_identity = "-".join(spec.identity.digest for spec in specs)
    return RuntimePlan(
        mode="resident",
        profile_identity=profile_identity,
        environment=environment,
        allocations=(allocation,),
        stages=stages,
    )
