"""Qwen-Image DiT as a GPU component.

The module is the backend-neutral ``_QwenImageTransformerTraceModule``, whose
``difflet.ops`` calls bind to the selected backend at import time. It is built on the
meta device, materialized in the requested dtype, and filled from the diffusers
``transformer/`` safetensors through the shared checkpoint loader (at tp=1 every rank's
slice is the whole tensor).
"""

from __future__ import annotations

import os

import torch

from difflet.backends._torch_megatron.checkpoint import load_checkpoint_into
from difflet.backends.cuda.ops_impl import parallel_mesh
from difflet.backends.gpu.application import GpuComponent
from difflet.models.qwen_image.dit_config import QwenImageDiTConfig
from difflet.pipeline.parallel_mesh import MeshSpec

CHECKPOINT_PREFIX = "transformer."
# The warm-up prompt length: trimmed like a real prompt (so warm-up compiles the shape the
# run will use, not the 1024-token padded one), and a length nothing else in the graph has
# (dynamo's duck sizing ties equal sizes together at trace time; 16 would collide with
# out_channels, 1024 with the text table).
WARMUP_VALID_TEXT_TOKENS = 37


class CudaQwenImageTransformerApplication(GpuComponent):
    # dynamic=True: one graph for every prompt length. dynamic=False specialised per trimmed
    # prompt length and hit dynamo's recompile limit; dynamic=None's automatic-dynamic graph
    # failed its own guard on the next length and compiled a third graph (torch 2.11 / 2.14,
    # see test_compiled_blocks_do_not_recompile_per_prompt_length).
    compile_dynamic = True
    # dynamo's guard on a None block argument errors once ("'NoneType' object has no attribute
    # 'size'") and forces a second compile; the second warm-up takes it at load.
    warmup_passes_when_compiled = 2

    def __init__(
        self,
        *,
        model_path: str,
        config: QwenImageDiTConfig,
        device=None,
        torch_compile: bool = False,
    ):
        super().__init__(
            config=MeshSpec(tp=int(config.tp_degree)),
            device=device,
            dtype=config.torch_dtype,
            torch_compile=torch_compile,
        )
        self.model_path = model_path
        self.dit_config = config

    def build_module(self) -> torch.nn.Module:
        from accelerate import init_empty_weights

        from difflet.models.qwen_image.modeling_qwen_image import _QwenImageTransformerTraceModule

        self._init_runtime()
        # Eager GPU execution lets the prompt padding mask reach the attention
        # (Trainium/TPU keep dropping it: the mask path does not trace under XLA), and
        # the padding is sliced off before the DiT: same output, flash SDPA back.
        with init_empty_weights(include_buffers=False):
            module = _QwenImageTransformerTraceModule(
                self.dit_config, apply_text_mask=True, trim_text_padding=True
            )
        return module

    def load_weights(self, module: torch.nn.Module) -> None:
        load_checkpoint_into(
            module,
            self.model_path,
            tp_size=self._mesh().tp,
            tp_rank=parallel_mesh.get_tp_rank(),
            prefix=CHECKPOINT_PREFIX,
            dtype=self.dit_config.torch_dtype,
            strict=True,
        )

    def repeated_blocks(self, module: torch.nn.Module) -> list[torch.nn.ModuleList]:
        # Each block is compiled on its own: one graph reused 60x, and the prompt trim's host
        # sync stays outside the graph, so ``self.module`` is still the trace module.
        return [module.transformer.transformer_blocks]

    def get_example_inputs(self) -> tuple:
        cfg = self.dit_config
        batch = int(cfg.batch_size)
        dtype = cfg.torch_dtype
        mask = torch.zeros(batch, int(cfg.text_seq_len), dtype=torch.bool)
        mask[:, : min(WARMUP_VALID_TEXT_TOKENS, int(cfg.text_seq_len))] = True
        return (
            torch.zeros(batch, int(cfg.image_seq_len), int(cfg.in_channels), dtype=dtype),
            torch.zeros(batch, dtype=dtype),
            torch.zeros(batch, int(cfg.text_seq_len), int(cfg.joint_attention_dim), dtype=dtype),
            mask,
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


__all__ = [
    "CHECKPOINT_PREFIX",
    "WARMUP_VALID_TEXT_TOKENS",
    "CudaQwenImageTransformerApplication",
    "transformer_dir",
]
