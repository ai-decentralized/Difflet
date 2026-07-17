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
from difflet.serving.errors import prompt_too_long
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
                    data=self.adapter._decode(invocation.input.packed_latents),
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

    async def create_loaded_runners(
        self,
        runtime: ResolvedRuntimeBundle,
    ) -> OrderedDict[str, ErasedStageRunner]:
        profile = runtime.profile
        if profile.parallel.cp_degree != 1:
            raise RuntimeError("Qwen-Image P0 shared-worker serving requires cp_degree=1")
        self.active_runtime = runtime
        self.active_profile = profile
        self.model_dir = runtime.source.pinned_model_path
        manager = ImmutableArtifactManager(profile.cache_dir or Path.home() / ".cache" / "difflet")
        for binding in runtime.artifacts.bindings:
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
        if not (self.text_app and self.tokenizer and self.denoise_app and self.vae_app):
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
        import torch
        from difflet.models.qwen_image.application import NeuronQwenImageApplication

        assert self.model_dir is not None
        self.denoise_app = NeuronQwenImageApplication(
            model_path=self.model_dir,
            parallel=profile.parallel,
            dtype=torch.bfloat16,
            shape=profile.shape_dict(),
            text_seq_len=_TEXT_SEQ_LEN,
            enable_transformer=True,
            teacache_fused=profile.teacache_speedup is not None,
            teacache_speedup=profile.teacache_speedup,
            teacache_calibration=profile.teacache_calibration_data,
            teacache_calibration_path=profile.teacache_calibration,
        )
        assert self.active_runtime is not None
        self.denoise_app.load(
            str(self.active_runtime.artifacts.require("generate").path), skip_warmup=True
        )

    def _load_vae_stage(self, profile: ServingProfile) -> None:
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
        )
        self.vae_app = NeuronWanVAEDecoderApplication(model_path=vae_path, config=self.vae_config)
        assert self.active_runtime is not None
        self.vae_app.load(str(self.active_runtime.artifacts.require("vae").path))

    def _encode_prompt(self, prompt: str) -> dict[str, object]:
        import torch

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

        if self.denoise_app is None or self.active_profile is None:
            raise RuntimeError("Qwen denoiser is not loaded")
        profile = self.active_profile
        guidance = torch.full([1], float(request.guidance_scale), dtype=torch.bfloat16)
        sched = self.denoise_app.pipeline.scheduler
        sc = sched.config
        image_seq_len = (profile.height // 16) * (profile.width // 16)
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
        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps).tolist()
        sched.set_timesteps(sigmas=sigmas, mu=mu, device="cpu")
        torch.manual_seed(request.seed)
        out = self.denoise_app.pipeline(
            encoder_hidden_states=text["encoder_hidden_states"],
            encoder_hidden_states_mask=text["encoder_hidden_states_mask"],
            guidance=guidance,
            timesteps=sched.timesteps,
            num_inference_steps=num_steps,
            teacache_enabled=use_teacache,
            output_type="latent",
        )
        return out.latents.cpu()

    def _decode(self, packed) -> bytes:
        import torch

        if self.vae_app is None or self.vae_config is None:
            raise RuntimeError("Qwen VAE decoder is not loaded")
        if self.active_profile is None:
            raise RuntimeError("Qwen serving profile is not loaded")
        b, seq, _ = packed.shape
        hh, ww = _packed_latent_grid(self.active_profile, seq)
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


def _packed_latent_grid(profile: ServingProfile, seq: int) -> tuple[int, int]:
    height = int(profile.height) // 16
    width = int(profile.width) // 16
    if int(seq) != height * width:
        raise ValueError(
            "Qwen packed latent sequence does not match the serving profile: "
            f"seq={seq}, expected={height * width} for "
            f"height={profile.height}, width={profile.width}."
        )
    return height, width


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
            artifact_id=by_id[stage.stage_id].artifact_id,
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
