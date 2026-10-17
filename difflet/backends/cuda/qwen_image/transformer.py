"""Qwen-Image DiT as a CUDA component (CUDA plan Phase 4).

The module is the backend-neutral ``_QwenImageTransformerTraceModule``, whose
``difflet.ops`` calls bind to the CUDA backend at import time. It is built on the
meta device, materialized in the requested dtype, and filled from the diffusers
``transformer/`` safetensors through the shared checkpoint loader (tp=1 in
Release 1: every rank's slice is the whole tensor). Mirrors
``difflet/backends/tpu/qwen_image/transformer.py`` minus XLA.
"""

from __future__ import annotations

import os

import torch

from difflet.backends._torch_megatron.checkpoint import load_checkpoint_into
from difflet.backends._torch_megatron.weights import materialize_meta_
from difflet.backends.cuda.core.application_base import CudaApplicationBase
from difflet.backends.cuda.ops_impl import parallel_mesh
from difflet.models.qwen_image.dit_config import QwenImageDiTConfig
from difflet.pipeline.parallel_mesh import MeshSpec

CHECKPOINT_PREFIX = "transformer."


class CudaQwenImageTransformerApplication(CudaApplicationBase):
    def __init__(
        self,
        *,
        model_path: str,
        config: QwenImageDiTConfig,
        device=None,
        torch_compile: bool = False,
    ):
        super().__init__(
            config=MeshSpec(tp=int(config.tp_degree)), device=device, dtype=config.torch_dtype
        )
        self.model_path = model_path
        self.dit_config = config
        # D4: eager is the premise; ``torch.compile`` is a measured, opt-in option (F4 task 3).
        # Each transformer block is compiled on its own (one graph reused 60x; the prompt
        # trim's host sync stays outside the graph), so ``self.module`` is still the trace
        # module and the parity scripts call it unchanged.
        self.torch_compile = bool(torch_compile)

    def _init_runtime(self) -> None:
        parallel_mesh.init_parallel_mesh(self.config)

    def build_module(self) -> torch.nn.Module:
        from accelerate import init_empty_weights

        from difflet.models.qwen_image.modeling_qwen_image import _QwenImageTransformerTraceModule

        self._init_runtime()
        # CUDA runs eagerly, so the prompt padding mask can reach the attention
        # (Trainium/TPU keep dropping it: the mask path does not trace under XLA), and
        # the padding is sliced off before the DiT (F4): same output, flash SDPA back.
        with init_empty_weights(include_buffers=False):
            module = _QwenImageTransformerTraceModule(
                self.dit_config, apply_text_mask=True, trim_text_padding=True
            )
        return module

    def _prepare_module(self) -> torch.nn.Module:
        if self.module is None:
            module = self.build_module()
            materialize_meta_(module, device=self.device, dtype=self.dit_config.torch_dtype)
            spec = self._mesh()
            load_checkpoint_into(
                module,
                self.model_path,
                tp_size=spec.tp,
                tp_rank=parallel_mesh.get_tp_rank(),
                prefix=CHECKPOINT_PREFIX,
                dtype=self.dit_config.torch_dtype,
                strict=True,
            )
            module.to(self.device)
            module.requires_grad_(False)
            module.eval()
            self._assert_on_device(module)
            if self.torch_compile:
                blocks = module.transformer.transformer_blocks
                for i, block in enumerate(blocks):
                    blocks[i] = torch.compile(
                        block, mode="max-autotune-no-cudagraphs", dynamic=False, fullgraph=False
                    )
            self.module = module
        return self.module

    def get_example_inputs(self) -> tuple:
        cfg = self.dit_config
        batch = int(cfg.batch_size)
        dtype = cfg.torch_dtype
        return (
            torch.zeros(batch, int(cfg.image_seq_len), int(cfg.in_channels), dtype=dtype),
            torch.zeros(batch, dtype=dtype),
            torch.zeros(batch, int(cfg.text_seq_len), int(cfg.joint_attention_dim), dtype=dtype),
            None,
            None,
        )

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_hidden_states_mask=None,
        guidance=None,
    ):
        if self.module is None or not self.is_loaded:
            raise RuntimeError("model is not loaded; call load() first")
        return self.module(
            hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_mask, guidance
        )


def transformer_dir(model_path: str) -> str:
    return os.path.join(model_path, "transformer")


__all__ = ["CHECKPOINT_PREFIX", "CudaQwenImageTransformerApplication", "transformer_dir"]
