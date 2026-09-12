"""TPU application for the FLUX.1-dev DiT (diffusers' module, TP-sharded per rank)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import torch

from difflet.backends.tpu.core.application_base import TpuApplicationBase
from difflet.backends.tpu.core.checkpoint import CheckpointSlice, load_checkpoint_into
from difflet.backends.tpu.core.weights import materialize_meta_
from difflet.backends.tpu.ops_impl import parallel_mesh
from difflet.pipeline.parallel_mesh import MeshSpec

logger = logging.getLogger(__name__)

_SINGLE_PROJ_OUT = re.compile(r"^(single_transformer_blocks\.\d+\.)proj_out_(attn|mlp)\.(weight|bias)$")


def make_checkpoint_key(inner_dim: int):
    """Module parameter name -> checkpoint key or window.

    The single blocks' fused ``proj_out`` [dim, dim + 4*dim] feeds the two
    row-parallel halves: attn takes input columns ``[:inner_dim]``, mlp the rest;
    the bias belongs to the attn half (skip_bias_add).
    """

    def checkpoint_key(name: str):
        match = _SINGLE_PROJ_OUT.match(name)
        if match is None:
            return name
        prefix, stream, kind = match.groups()
        key = f"{prefix}proj_out.{kind}"
        if kind == "bias":
            return key
        if stream == "attn":
            return CheckpointSlice(key, dim=1, start=0, stop=int(inner_dim))
        return CheckpointSlice(key, dim=1, start=int(inner_dim), stop=None)

    return checkpoint_key


class TpuFluxTransformerApplication(TpuApplicationBase):
    def __init__(self, *, model_path, config):
        super().__init__(config=config)
        self.model_path = Path(model_path)

    def _init_runtime(self) -> None:
        from difflet.backends.tpu.ops_impl.platform import configure_matmul_precision

        configure_matmul_precision()
        parallel_mesh.init_parallel_mesh(MeshSpec(tp=int(getattr(self.config, "tp_degree", 1) or 1)))

    def build_module(self) -> torch.nn.Module:
        from difflet.models.flux.tp_sharding import build_flux_transformer, shard_flux_transformer

        try:
            from accelerate import init_empty_weights
        except ImportError:  # pragma: no cover
            init_empty_weights = None
        tp = int(getattr(self.config, "tp_degree", 1) or 1)

        def _build():
            transformer = build_flux_transformer(self.config)
            shard_flux_transformer(transformer, tp)
            return transformer

        if init_empty_weights is None:
            return _build()
        with init_empty_weights(include_buffers=False):
            return _build()

    def _prepare_module(self) -> torch.nn.Module:
        if self.module is not None:
            return self.module
        try:
            parallel_mesh.get_mesh_spec()
        except RuntimeError:
            self._init_runtime()
        module = self.build_module()
        spec = self._mesh()
        dtype = getattr(self.config, "torch_dtype", None)
        materialize_meta_(module, dtype=dtype)
        load_checkpoint_into(
            module, self.model_path, tp_size=spec.tp, tp_rank=parallel_mesh.get_tp_rank(),
            dtype=dtype, rename=make_checkpoint_key(self.config.inner_dim), strict=True,
        )
        module.requires_grad_(False)
        self.module = module.eval()
        return self.module

    def get_example_inputs(self) -> tuple:
        cfg = self.config
        dtype = getattr(cfg, "torch_dtype", torch.bfloat16)
        b = int(cfg.batch_size)
        return (
            torch.zeros(b, cfg.image_seq_len, int(cfg.in_channels), dtype=dtype),
            torch.zeros(b, int(cfg.text_seq_len), int(cfg.joint_attention_dim), dtype=dtype),
            torch.zeros(b, int(cfg.pooled_projection_dim), dtype=dtype),
            torch.zeros(b, dtype=dtype),
            torch.zeros(cfg.image_seq_len, 3, dtype=torch.float32),
            torch.zeros(int(cfg.text_seq_len), 3, dtype=torch.float32),
            torch.zeros(b, dtype=dtype),
        )

    def forward(self, hidden_states, encoder_hidden_states, pooled_projections, timestep,
                img_ids, txt_ids, guidance):
        if self.module is None:
            self._init_runtime()
            self._prepare_module()
        return self.module(
            hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections, timestep=timestep, img_ids=img_ids,
            txt_ids=txt_ids, guidance=guidance, return_dict=False,
        )[0]


__all__ = ["TpuFluxTransformerApplication", "make_checkpoint_key"]
