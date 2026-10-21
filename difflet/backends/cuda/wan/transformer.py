"""One Wan DiT expert as a CUDA component.

The module is the backend-neutral ``WanTransformer3DModel``
(``difflet/models/wan/modeling_wan.py``), whose ``difflet.ops`` calls bind to the
CUDA backend at import time. It is built on the meta device, materialized in the
requested dtype, and filled from one expert's diffusers safetensors directory
(``transformer/`` or ``transformer_2/``) through the shared checkpoint loader with
the FFN rename (``checkpoint_key``) — tp=1, so every tensor loads whole. Mirrors
``difflet/backends/tpu/wan/transformer.py`` minus XLA.

The expert is the unit, as on TPU: Wan 2.2 A14B ships two independent experts and
the caller decides how many are resident.
"""

from __future__ import annotations

import torch

from difflet.backends._torch_megatron.checkpoint import load_checkpoint_into
from difflet.backends._torch_megatron.weights import materialize_meta_
from difflet.backends.cuda.core.application_base import CudaApplicationBase
from difflet.backends.cuda.ops_impl import parallel_mesh
from difflet.models.wan.dit_config import WanDiTConfig, checkpoint_key
from difflet.pipeline.parallel_mesh import MeshSpec


def _refuse_parallel_modes(config: WanDiTConfig) -> None:
    """One expert on one device; anything else is refused."""
    if int(config.tp_degree) != 1:
        raise NotImplementedError(
            f"Wan on CUDA runs tp=1 (got tp_degree={config.tp_degree}); "
            "tensor parallelism (tp > 1) is not supported on the CUDA backend yet"
        )
    if config.context_parallel_enabled or config.cfg_parallel_enabled:
        raise NotImplementedError(
            "Wan on CUDA runs one process: context and CFG parallel are "
            "not supported on the CUDA backend yet"
        )
    if config.sp_enabled:
        raise NotImplementedError(
            "sequence parallelism is not supported on the CUDA backend yet (Wan runs at tp=1)"
        )


class CudaWanTransformerApplication(CudaApplicationBase):
    """One Wan DiT expert on one device."""

    def __init__(self, *, model_path: str, config: WanDiTConfig, device=None):
        _refuse_parallel_modes(config)
        super().__init__(config=MeshSpec(tp=1), device=device, dtype=config.torch_dtype)
        # This expert's diffusers directory (``transformer/`` or ``transformer_2/``).
        self.model_path = model_path
        self.dit_config = config

    def _init_runtime(self) -> None:
        parallel_mesh.init_parallel_mesh(self.config)

    def build_module(self) -> torch.nn.Module:
        from accelerate import init_empty_weights

        from difflet.models.wan.modeling_wan import WanTransformer3DModel

        # Before the build: WanAttention sizes its head shard from the tp size in its
        # own __init__.
        self._init_runtime()
        # include_buffers=False: the 3D RoPE frequency grid is computed for real in
        # __init__, and that arithmetic fails on meta buffers.
        with init_empty_weights(include_buffers=False):
            return WanTransformer3DModel(self.dit_config, dtype=self.dit_config.torch_dtype)

    def _prepare_module(self) -> torch.nn.Module:
        if self.module is None:
            cfg = self.dit_config
            module = self.build_module()
            materialize_meta_(module, device=self.device, dtype=cfg.torch_dtype)
            load_checkpoint_into(
                module,
                self.model_path,
                tp_size=1,
                tp_rank=0,
                dtype=cfg.torch_dtype,
                # difflet's FFN is net_in/net_out where diffusers has net.0.proj/net.2.
                rename=checkpoint_key,
                # strict: a parameter with no checkpoint entry would keep the
                # uninitialized storage materialize_meta_ gave it.
                strict=True,
            )
            module.to(self.device)
            module.requires_grad_(False)
            module.eval()
            self._assert_on_device(module)
            self.module = module
        return self.module

    def get_example_inputs(self) -> tuple:
        """Unpatchified 5D latents, a per-sample timestep and the text embeddings."""
        cfg = self.dit_config
        dtype = cfg.torch_dtype
        batch = int(cfg.batch_size)
        return (
            torch.zeros(
                batch,
                int(cfg.in_channels),
                cfg.latent_frames,
                cfg.latent_height,
                cfg.latent_width,
                dtype=dtype,
            ),
            torch.zeros(batch, dtype=dtype),
            torch.zeros(batch, int(cfg.text_seq_len), int(cfg.text_dim), dtype=dtype),
        )

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        if self.module is None or not self.is_loaded:
            raise RuntimeError("model is not loaded; call load() first")
        return self.module(hidden_states, timestep, encoder_hidden_states)


__all__ = ["CudaWanTransformerApplication"]
