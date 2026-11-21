"""MiniMax-H3's joint audio/video transformer on the generic GPU lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from difflet.backends._torch_megatron.checkpoint import load_checkpoint_into
from difflet.backends.gpu.application import GpuComponent
from difflet.pipeline.parallel_mesh import MeshSpec


class CudaMiniMaxH3TransformerApplication(GpuComponent):
    """CUDA's transformer module (``modeling``) with the released checkpoint's precision."""

    # One graph per block for every sequence length (the token refiner and the main blocks),
    # in Inductor's default mode without autotuning, and rounding to BF16 where eager does:
    # otherwise the fused blocks keep FP32 intermediates and the 50-step trajectory drifts.
    compile_dynamic = True
    compile_options = {
        "emulate_precision_casts": True,
        "max_autotune": False,
        "coordinate_descent_tuning": False,
    }

    def __init__(self, *, model_path: str, device=None, torch_compile: bool = False):
        super().__init__(
            config=MeshSpec(), device=device, dtype=torch.bfloat16, torch_compile=torch_compile
        )
        self.model_path = model_path
        self._model_config = json.loads((Path(model_path) / "config.json").read_text())

    def build_module(self):
        from accelerate import init_empty_weights

        from difflet.backends.cuda.minimax_h3.modeling import CudaMiniMaxH3Transformer

        self._init_runtime()
        # Keep the computed RoPE frequencies real and FP32. The shared lifecycle
        # materializes only meta parameters, directly in BF16.
        with init_empty_weights(include_buffers=False):
            return CudaMiniMaxH3Transformer.from_config(self._model_config)

    def load_weights(self, module):
        # The checkpoint stores the patch projections, timestep embedding and output heads in
        # FP32. Restore their storage before streaming the weights in: converting BF16
        # values after loading would already have lost the precision.
        for name in ("proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out"):
            module.get_submodule(name).float()
        load_checkpoint_into(module, self.model_path, tp_size=1, tp_rank=0, strict=True)

    def repeated_blocks(self, module):
        return [module.token_refiner.refiner_blocks, module.transformer_blocks]


__all__ = ["CudaMiniMaxH3TransformerApplication"]
