"""Qwen-Image shared-worker serving adapter."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

from difflet.common.orchestrators import qwen_image as qwen_common
from difflet.serving.options import CompilePolicy, DownloadPolicy
from difflet.serving.errors import prompt_too_long
from difflet.serving.types import (
    DiffletCompileSpec,
    DiffletGenerateOutput,
    DiffletGenerateRequest,
    DiffletStageSpec,
    ServingProfile,
    WorkerRequestContext,
)

_HF_MODEL_ID = "Qwen/Qwen-Image"
_MODEL_TYPE = "qwen_image"
_ENC_SEQ = qwen_common.ENC_SEQ
_TEXT_SEQ_LEN = qwen_common.TEXT_SEQ_LEN
_QWEN_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
_QWEN_DROP_IDX = 34


class QwenImageServingArtifactPreparer:
    model_id = _HF_MODEL_ID
    model_type = _MODEL_TYPE

    def __init__(self, model_id: str = _HF_MODEL_ID, revision: str | None = None) -> None:
        self.model_id = model_id
        self.revision = revision

    def resolve_model_path(self, *, download_policy: DownloadPolicy) -> Path:
        from difflet.pipeline.path_resolver import resolve_model_path

        print(f"[difflet serve] resolving Qwen-Image weights for {self.model_id}")
        path = resolve_model_path(
            self.model_id,
            revision=self.revision,
            local_files_only=download_policy == DownloadPolicy.NEVER,
        )
        print(f"[difflet serve] Qwen-Image weights ready at {path}")
        return Path(path)

    def stage_specs(self, profile: ServingProfile) -> tuple[DiffletStageSpec, ...]:
        full_cores = profile.parallel.world_size
        return (
            DiffletStageSpec("prompt_encoder", "prompt_encoder", full_cores, ("text",)),
            DiffletStageSpec("denoiser", "denoiser", full_cores, ("latents",)),
            DiffletStageSpec(
                "decoder", "decoder", full_cores, ("image",), final_output=True
            ),
        )

    def compile_plan(self, profile: ServingProfile) -> tuple[DiffletCompileSpec, ...]:
        return qwen_common.compile_plan(profile)

    def ensure_artifacts(self, profile: ServingProfile, policy: CompilePolicy) -> None:
        qwen_common.ensure_artifacts(profile, policy)
        print("[difflet serve] Qwen-Image AOT artifacts ready")


class QwenImageServingRequestValidator:
    model_id = _HF_MODEL_ID
    model_type = _MODEL_TYPE

    def __init__(self, model_id: str = _HF_MODEL_ID, revision: str | None = None) -> None:
        self.model_id = model_id
        self.revision = revision
        self._tokenizer = None

    def validate(self, request: DiffletGenerateRequest, profile: ServingProfile) -> None:
        encoded = self._tokenizer_for_profile(profile)(
            _QWEN_TEMPLATE.format(request.prompt),
            padding=False,
            truncation=False,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if int(encoded.input_ids.shape[1]) > _ENC_SEQ:
            raise prompt_too_long(f"Qwen prompt exceeds encoder bucket {_ENC_SEQ}")

    def _tokenizer_for_profile(self, profile: ServingProfile):
        if self._tokenizer is None:
            from difflet.pipeline.path_resolver import resolve_model_path
            from transformers import AutoTokenizer

            model_dir = resolve_model_path(
                self.model_id,
                revision=profile.revision,
                local_files_only=True,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(str(Path(model_dir) / "tokenizer"))
        return self._tokenizer


class QwenImageServingOrchestrator:
    model_id = _HF_MODEL_ID
    model_type = _MODEL_TYPE

    def __init__(self, model_id: str = _HF_MODEL_ID) -> None:
        self.model_id = model_id
        self.active_profile: ServingProfile | None = None
        self.model_dir: str | None = None
        self.text_app = None
        self.tokenizer = None
        self.denoise_app = None
        self.vae_app = None
        self.vae_config = None

    def load(self, profile: ServingProfile) -> None:
        if profile.parallel.cp_degree != 1:
            raise RuntimeError("Qwen-Image P0 shared-worker serving requires cp_degree=1")
        self.active_profile = profile
        self.model_dir = self._resolve_model_dir()
        print("[difflet serve] loading Qwen prompt_encoder stage")
        self._load_text_stage(profile)
        print("[difflet serve] loading Qwen denoiser stage")
        self._load_denoiser_stage(profile)
        print("[difflet serve] loading Qwen decoder stage")
        self._load_vae_stage(profile)
        print("[difflet serve] Qwen shared-worker co-load completed")

    def smoke(self) -> None:
        if not (self.text_app and self.tokenizer and self.denoise_app and self.vae_app):
            raise RuntimeError("Qwen shared-worker load did not initialize all stages")
        assert self.active_profile is not None
        print("[difflet serve] running Qwen shared-worker generation smoke")
        profile = self.active_profile
        request = DiffletGenerateRequest(
            request_id="startup-smoke",
            model=self.model_id,
            prompt="a small red square",
            height=profile.height,
            width=profile.width,
            num_inference_steps=4,
            guidance_scale=1.0,
            seed=0,
        )
        context = WorkerRequestContext.with_timeout("startup-smoke", 300.0)
        output = asyncio.run(self.generate(request, context))
        if not output.data:
            raise RuntimeError("Qwen shared-worker smoke produced empty output")
        print("[difflet serve] Qwen shared-worker generation smoke passed")

    async def generate(
        self,
        request: DiffletGenerateRequest,
        context: WorkerRequestContext,
    ) -> DiffletGenerateOutput:
        context.cancellation.throw_if_cancelled()
        text = self._encode_prompt(request.prompt)
        context.cancellation.throw_if_cancelled()
        latents = self._denoise(text, request)
        context.cancellation.throw_if_cancelled()
        image_bytes = self._decode(latents)
        return DiffletGenerateOutput(data=image_bytes, mime_type="image/png", output_format="png")

    def shutdown(self) -> None:
        self.text_app = None
        self.tokenizer = None
        self.denoise_app = None
        self.vae_app = None
        self.vae_config = None
        self.active_profile = None

    def _resolve_model_dir(self) -> str:
        from difflet.pipeline.path_resolver import resolve_model_path

        assert self.active_profile is not None
        return resolve_model_path(
            self.model_id,
            revision=self.active_profile.revision,
            local_files_only=True,
        )

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
        self.text_app.load(str(qwen_common.stage_compiled_dir("text", profile)))
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
        )
        self.denoise_app.load(
            str(qwen_common.serving_stage_compiled_dir("generate", profile)),
            skip_warmup=True,
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
        self.vae_app.load(str(qwen_common.serving_stage_compiled_dir("vae", profile)))

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
        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps).tolist()
        sched.set_timesteps(sigmas=sigmas, mu=mu, device="cpu")
        torch.manual_seed(request.seed)
        out = self.denoise_app.pipeline(
            encoder_hidden_states=text["encoder_hidden_states"],
            encoder_hidden_states_mask=text["encoder_hidden_states_mask"],
            guidance=guidance,
            timesteps=sched.timesteps,
            num_inference_steps=num_steps,
            output_type="latent",
        )
        return out.latents.cpu()

    def _decode(self, packed) -> bytes:
        import torch

        if self.vae_app is None or self.vae_config is None:
            raise RuntimeError("Qwen VAE decoder is not loaded")
        b, seq, _ = packed.shape
        hh = ww = int(seq**0.5)
        z = packed.float().view(b, hh, ww, 16, 2, 2)
        z = z.permute(0, 3, 1, 4, 2, 5).reshape(b, 16, hh * 2, ww * 2)
        z = z.unsqueeze(2)
        mean = torch.tensor(self.vae_config.latents_mean).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.vae_config.latents_std).view(1, -1, 1, 1, 1)
        z = (z * std + mean).to(torch.bfloat16) if len(self.vae_config.latents_mean) else z.to(torch.bfloat16)
        img = self.vae_app(z)
        img = (img[0] if isinstance(img, (tuple, list)) else img).float().cpu()
        img = img[:, :, 0]
        return _tensor_to_png_bytes((img[0] * 0.5 + 0.5).clamp(0, 1))


def _tensor_to_png_bytes(tensor) -> bytes:
    from torchvision.transforms.functional import to_pil_image

    image = to_pil_image(tensor)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
