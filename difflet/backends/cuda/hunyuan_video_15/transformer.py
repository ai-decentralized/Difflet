"""HunyuanVideo 1.5 transformer loading through the shared GPU lifecycle."""

from __future__ import annotations

import torch

from difflet.backends._torch_megatron.checkpoint import build_weight_map
from difflet.backends.gpu.application import GpuComponent
from difflet.pipeline.parallel_mesh import MeshSpec


class CudaHunyuanVideo15TransformerApplication(GpuComponent):
    # Timing-based reduction choices can change BF16 trajectories between processes; bind
    # stable ones per compiled block, without changing Inductor's global policy.
    compile_options = {"deterministic": True}

    def __init__(self, *, model_path, config, device=None, torch_compile=False):
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
        from difflet.backends.cuda.hunyuan_video_15.modeling import (
            HunyuanVideo15Transformer3DModel,
        )

        self._init_runtime()
        with init_empty_weights(include_buffers=False):
            return HunyuanVideo15Transformer3DModel(self.dit_config).to(dtype=self.dtype)

    def load_weights(self, module):
        from safetensors.torch import load_file

        state = {}
        for file in sorted(set(build_weight_map(self.model_path).values())):
            state.update(load_file(file))
        module.load_state_dict(state, strict=True)

    def apply_load_time_swaps(self, module):
        # On the device and before compile: one QKV GEMM per stream for the joint attention
        # path (weights become views; memory and keys unchanged). Eager only: the joint path
        # trims the context to its valid tokens, so static-shape compiled blocks would
        # recompile per prompt length; they keep the prefix path, one graph for every length.
        from difflet.backends.cuda.hunyuan_video_15.modeling import joint_attention_supported

        attn = module.transformer_blocks[0].attn
        if not self.torch_compile and joint_attention_supported(
            attn.to_q.weight, attn.norm_q.weight.shape[0]
        ):
            module.pack_attention_projections()

    def repeated_blocks(self, module):
        return [module.transformer_blocks]

    def get_example_inputs(self):
        c = self.dit_config
        return (
            torch.zeros(
                1, c.in_channels, c.latent_frames, c.latent_height, c.latent_width, dtype=self.dtype
            ),
            torch.full((1,), 500.0, dtype=self.dtype),
            torch.zeros(1, c.text_seq_len, c.text_embed_dim, dtype=self.dtype),
            torch.arange(c.text_seq_len)[None] < min(3, c.text_seq_len),
            torch.zeros(1, c.text_2_seq_len, c.text_embed_2_dim, dtype=self.dtype),
            torch.zeros(1, c.text_2_seq_len, dtype=torch.bool),
            torch.zeros(1, c.image_tokens, c.image_embed_dim, dtype=self.dtype),
        )
