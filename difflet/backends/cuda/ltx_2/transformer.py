"""LTX-2 audiovisual transformer on the generic single-device GPU lifecycle."""

from __future__ import annotations

from contextlib import contextmanager

import torch

from difflet.backends._torch_megatron.checkpoint import load_checkpoint_into
from difflet.backends.cuda.ltx_2.dit_config import Ltx2DiTConfig
from difflet.backends.gpu.application import GpuComponent
from difflet.pipeline.parallel_mesh import MeshSpec


class CudaLtx2TransformerApplication(GpuComponent):
    """Strict diffusers checkpoint loading and the ten-tensor Difflet DiT interface."""

    # Fused bf16 modulation and gated residuals must retain eager intermediate
    # rounding. Otherwise small block differences accumulate during denoising.
    compile_options = {
        "emulate_precision_casts": True,
        # Expanded-bias GEMM autotuning can select different BF16 rounding from
        # eager addmm. Retain eager's GEMM selection while compiling the block.
        "max_autotune": False,
        "max_autotune_gemm": False,
    }

    def __init__(
        self,
        *,
        model_path: str,
        config: Ltx2DiTConfig,
        device=None,
        torch_compile: bool = False,
        generation_profile: str = "diffusers",
    ):
        # The application validated the profile and dtype; the config validated itself.
        self.generation_profile = generation_profile
        super().__init__(
            config=MeshSpec(), device=device, dtype=config.torch_dtype, torch_compile=torch_compile
        )
        self.model_path = model_path
        self.dit_config = config
        self.load_report = None

    def build_module(self) -> torch.nn.Module:
        from accelerate import init_empty_weights

        from .modeling import CudaLtx2TransformerModule

        self._init_runtime()
        with init_empty_weights(include_buffers=False):
            return CudaLtx2TransformerModule(self.dit_config)

    def load_weights(self, module: torch.nn.Module) -> None:
        self.load_report = load_checkpoint_into(
            module,
            self.model_path,
            tp_size=1,
            tp_rank=0,
            prefix="transformer.",
            dtype=self.dtype,
            strict=True,
        )
        if self.load_report["unexpected"]:
            raise ValueError(
                f"Unexpected LTX-2 checkpoint parameters: {self.load_report['unexpected'][:5]}"
            )

    def apply_load_time_swaps(self, module: torch.nn.Module) -> None:
        if self.generation_profile == "ltx_official_bf16":
            from .official_compatibility import (
                configure_official_forward,
                configure_official_transformer,
            )

            configure_official_transformer(module.transformer)
            configure_official_forward(module.transformer)
        if self.torch_compile:
            from .compile_policy import preserve_native_rms_norm

            preserve_native_rms_norm(module)

    def repeated_blocks(self, module: torch.nn.Module) -> list[torch.nn.ModuleList]:
        return [module.transformer.transformer_blocks]

    def get_example_inputs(self) -> tuple[torch.Tensor, ...]:
        from difflet.models.ltx_2.coords import make_audio_coords, make_video_coords

        cfg = self.dit_config
        video_coords = make_video_coords(
            batch_size=1,
            num_frames=cfg.latent_num_frames,
            height=cfg.latent_height,
            width=cfg.latent_width,
            patch_size=cfg.patch_size,
            patch_size_t=cfg.patch_size_t,
            scale_factors=cfg.vae_scale_factors,
            causal_offset=cfg.causal_offset,
            fps=cfg.frame_rate,
        )
        audio_coords = make_audio_coords(
            batch_size=1,
            audio_num_frames=cfg.audio_num_frames,
            patch_size_t=cfg.audio_patch_size_t,
            scale_factor=cfg.audio_scale_factor,
            causal_offset=cfg.causal_offset,
            sampling_rate=cfg.audio_sampling_rate,
            hop_length=cfg.audio_hop_length,
        )
        return (
            torch.zeros(1, cfg.video_seq_len, cfg.in_channels, dtype=self.dtype),
            torch.zeros(1, cfg.audio_seq_len, cfg.audio_in_channels, dtype=self.dtype),
            torch.zeros(1, cfg.text_seq_len, cfg.video_text_dim, dtype=self.dtype),
            torch.zeros(1, cfg.text_seq_len, cfg.audio_text_dim, dtype=self.dtype),
            torch.full((1,), 500.125, dtype=torch.float32),
            torch.full((1,), 0.500125, dtype=torch.float32),
            torch.ones(1, cfg.text_seq_len, dtype=torch.bool),
            torch.ones(1, cfg.text_seq_len, dtype=torch.bool),
            video_coords,
            audio_coords,
        )


class Ltx2DiTAdapter(torch.nn.Module):
    """Expose the complete stock diffusers call on the loaded Difflet component.

    The fixed ten-tensor API remains on the component. The pipeline-facing API
    calls its loaded inner transformer directly to preserve independent modality
    timesteps, sigmas, coordinates, masks and all upstream guidance arguments.
    """

    def __init__(self, app: CudaLtx2TransformerApplication):
        super().__init__()
        if not app.is_loaded or app.module is None:
            raise RuntimeError("LTX-2 transformer is not loaded; call load() first")
        self.app = app

    @property
    def config(self):
        return self.app.module.transformer.config

    @property
    def dtype(self) -> torch.dtype:
        return self.app.module.transformer.dtype

    @property
    def device(self) -> torch.device:
        return self.app.module.transformer.device

    @property
    def rope(self):
        return self.app.module.transformer.rope

    @property
    def audio_rope(self):
        return self.app.module.transformer.audio_rope

    @contextmanager
    def cache_context(self, name: str):
        with self.app.module.transformer.cache_context(name):
            yield

    def forward(self, *args, **kwargs):
        return self.app.module.transformer(*args, **kwargs)


__all__ = ["CudaLtx2TransformerApplication", "Ltx2DiTAdapter"]
