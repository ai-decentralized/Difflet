"""One Wan DiT expert as a GPU component.

The module is the backend-neutral ``WanTransformer3DModel``
(``difflet/models/wan/modeling_wan.py``), whose ``difflet.ops`` calls bind to the
selected backend at import time. It is built on the meta device, materialized in the
requested dtype, and filled from one expert's diffusers safetensors directory
(``transformer/`` or ``transformer_2/``) through the shared checkpoint loader with
the FFN rename (``checkpoint_key``); at tp=1 every tensor loads whole.

The expert is the unit, as on TPU: Wan 2.2 A14B ships two independent experts and
the application decides how many are resident.
"""

from __future__ import annotations

import torch

from difflet.backends._torch_megatron.checkpoint import load_checkpoint_into
from difflet.backends.gpu.application import GpuComponent
from difflet.models.wan.dit_config import WanDiTConfig, checkpoint_key
from difflet.pipeline.parallel_mesh import MeshSpec


class CudaWanTransformerApplication(GpuComponent):
    """One Wan DiT expert on one device.

    Under CFG-parallel two processes each hold the whole expert on a cfg=2 mesh:
    the modeling scatters the [uncond, cond] batch of 2 by cfg rank and all-gathers
    the outputs, so every rank returns the full batch. ``config.batch_size`` is then 2,
    which the warm-up's example inputs follow.
    """

    def __init__(
        self, *, model_path: str, config: WanDiTConfig, device=None, torch_compile: bool = False
    ):
        mesh = MeshSpec(cfg=2 if config.cfg_parallel_enabled else 1, tp=1)
        super().__init__(
            config=mesh, device=device, dtype=config.torch_dtype, torch_compile=torch_compile
        )
        # This expert's diffusers directory (``transformer/`` or ``transformer_2/``).
        self.model_path = model_path
        self.dit_config = config

    def _refuse_cfg_without_its_world(self) -> None:
        """A cfg=2 expert holds one CFG branch per process: loading it needs the world-2
        process group (the CLI re-execs itself under torchrun for this). Refuse before the
        weights are read rather than fail in the warm-up gather. Compile needs no group:
        the CLI parent writes the manifest before torchrun starts the ranks."""
        need = int(self.config.world_size)
        if need == 1:
            return
        dist = torch.distributed
        have = dist.get_world_size() if dist.is_available() and dist.is_initialized() else None
        if have != need:
            raise RuntimeError(
                f"Wan CFG-parallel runs one CFG branch per process on {need} GPUs, "
                f"but this process has {'no' if have is None else f'a world-{have}'} "
                "torch.distributed process group: run under "
                f"`torchrun --nproc_per_node={need}` (the difflet CLI does this for "
                "--cfg-parallel)"
            )

    def load(
        self,
        compiled_model_path,
        start_rank_id=None,
        local_ranks_size=None,
        skip_warmup: bool = False,
    ) -> None:
        self._refuse_cfg_without_its_world()
        super().load(
            compiled_model_path,
            start_rank_id=start_rank_id,
            local_ranks_size=local_ranks_size,
            skip_warmup=skip_warmup,
        )

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

    def load_weights(self, module: torch.nn.Module) -> None:
        load_checkpoint_into(
            module,
            self.model_path,
            tp_size=1,
            tp_rank=0,
            dtype=self.dit_config.torch_dtype,
            # difflet's FFN is net_in/net_out where diffusers has net.0.proj/net.2.
            rename=checkpoint_key,
            # strict: a parameter with no checkpoint entry would keep the
            # uninitialized storage materialize_meta_ gave it.
            strict=True,
        )

    def repeated_blocks(self, module: torch.nn.Module) -> list[torch.nn.ModuleList]:
        # One static shape per process, so the compile is static (dynamic=False).
        return [module.blocks]

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
