"""Difflet HunyuanVideo transformer through the shared GPU component lifecycle."""

from __future__ import annotations

import torch

from difflet.backends._torch_megatron.checkpoint import build_weight_map
from difflet.backends.gpu.application import GpuComponent
from difflet.backends.cuda.hunyuan_video.checkpoint import split_single_block_projections
from difflet.backends.cuda.hunyuan_video.dit_config import HunyuanVideoDiTConfig
from difflet.pipeline.parallel_mesh import MeshSpec


class CudaHunyuanVideoTransformerApplication(GpuComponent):
    def __init__(
        self,
        *,
        model_path: str,
        config: HunyuanVideoDiTConfig,
        device=None,
        torch_compile=False,
    ):
        super().__init__(
            config=MeshSpec(tp=1),
            device=device,
            dtype=config.torch_dtype,
            torch_compile=torch_compile,
        )
        self.model_path = model_path
        self.dit_config = config

    def build_module(self):
        from accelerate import init_empty_weights

        from difflet.backends.cuda.hunyuan_video.modeling import (
            CudaHunyuanVideoTransformer3DModel,
        )

        self._init_runtime()
        with init_empty_weights(include_buffers=False):
            return CudaHunyuanVideoTransformer3DModel(self.dit_config).to(dtype=self.dtype)

    def load_weights(self, module):
        from safetensors.torch import load_file

        state = {}
        for file in sorted(set(build_weight_map(self.model_path).values())):
            state.update(load_file(file))
        cfg = self.dit_config
        converted = split_single_block_projections(
            state, num_single_layers=cfg.num_single_layers, inner_dim=cfg.inner_dim, clone=False
        )
        module.load_state_dict(converted, strict=True)

    def repeated_blocks(self, module):
        return [module.transformer_blocks, module.single_transformer_blocks]

    def _prepare_module(self):
        # Keep the shared GPU lifecycle unchanged; only Hunyuan requests these
        # BF16 cast-preserving compiler options.
        if self.module is None:
            module = self.build_module()
            if any(p.is_meta for p in module.parameters()) or any(
                b.is_meta for b in module.buffers()
            ):
                from difflet.backends._torch_megatron.weights import materialize_meta_

                materialize_meta_(module, device=self.device, dtype=self.dtype)
            self.load_weights(module)
            module.to(self.device).requires_grad_(False).eval()
            self._assert_on_device(module)
            self.apply_load_time_swaps(module)
            if self.torch_compile:
                from difflet.backends.gpu.application import COMPILE_MODE

                options = {
                    **torch._inductor.list_mode_options(COMPILE_MODE),
                    "emulate_precision_casts": True,
                    "max_autotune_gemm_backends": "ATEN",
                }
                for blocks in self.repeated_blocks(module):
                    for i, block in enumerate(blocks):
                        blocks[i] = torch.compile(
                            block, dynamic=self.compile_dynamic, fullgraph=False, options=options
                        )
            self.module = module
        return self.module

    def apply_load_time_swaps(self, module):
        from difflet.backends.cuda.hunyuan_video.numerics import (
            fuse_single_projection,
            restore_reference_row_bias,
        )

        self.numerics_swaps = {
            "projections": sum(
                fuse_single_projection(block, tp_degree=self.config.tp)
                for block in module.single_transformer_blocks
            ),
            "biased_rows": restore_reference_row_bias(module, tp_degree=self.config.tp),
        }
        if self.torch_compile:
            from difflet.backends.cuda.hunyuan_video.numerics import (
                preserve_reference_compiled_ops,
            )

            self.reference_compile_ops = preserve_reference_compiled_ops(
                self.repeated_blocks(module), tp_degree=self.config.tp
            )

    def get_example_inputs(self):
        cfg = self.dit_config
        mask = torch.arange(cfg.text_seq_len)[None] < min(37, cfg.text_seq_len)
        return (
            torch.zeros(
                1,
                cfg.in_channels,
                cfg.latent_frames,
                cfg.latent_height,
                cfg.latent_width,
                dtype=self.dtype,
            ),
            torch.zeros(1, dtype=torch.float32),
            torch.zeros(1, cfg.text_seq_len, cfg.text_embed_dim, dtype=self.dtype),
            mask,
            torch.zeros(1, cfg.pooled_projection_dim, dtype=self.dtype),
            torch.tensor([6.0], dtype=self.dtype) * 1000,
        )
