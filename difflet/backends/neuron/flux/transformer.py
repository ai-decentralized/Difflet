"""Neuron application for the Flux DiT.

Wraps the backend-neutral ``NeuronFluxTransformer2DModel`` (``difflet/models/flux/
modeling_flux.py``) with ``TorchNeuronApplicationBase``'s lifecycle: the module is built on ``meta``
after the TP mesh exists, each rank loads its own slices of the diffusers checkpoint, and in
compile mode the 19 dual-stream ``transformer_blocks`` and the 38 single-stream
``single_transformer_blocks`` are compiled in place into one shared graph per block type.

Checkpoint mapping: diffusers fuses each single-stream block's attention and MLP output
projections into one ``proj_out`` (input columns ``[0, inner)`` attention, ``[inner, 5 inner)``
MLP); the model splits them so each is row-parallel on its own. ``proj_out_attn.weight`` and
``proj_out_mlp.weight`` are therefore column ranges of ``proj_out.weight`` (read per TP shard),
and ``proj_out_attn.bias`` is ``proj_out.bias`` (``proj_out_mlp`` has none).

``forward`` takes the pipeline's call (``img_ids``/``txt_ids``, guidance as a number or tensor) like
the Trainium wrapper: the rotary table is ``FluxPosEmbed`` over ``[txt_ids; img_ids]``, computed on
the host and cached per shape.
"""

from __future__ import annotations

import re

import torch

from difflet.backends.neuron.core.application_base import TorchNeuronApplicationBase
from difflet.backends.neuron.core.checkpoint import CheckpointSlice
from difflet.backends.neuron.flux.config import NeuronFluxConfig

_MODULE_TO_CHECKPOINT: list[tuple[str, str]] = [
    (r"\.proj_out_attn\.bias$", ".proj_out.bias"),
]


def checkpoint_key(name: str) -> str:
    """Diffusers checkpoint key holding the module parameter ``name``."""
    for pattern, replacement in _MODULE_TO_CHECKPOINT:
        name = re.sub(pattern, replacement, name)
    return name


def flux_position_ids(config: NeuronFluxConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """``(txt_ids, img_ids)`` for the config's shape, as diffusers' FluxPipeline builds them:
    text ids all zero, image ids (0, row, col) over the packed latent grid."""
    txt = torch.zeros(int(config.text_seq_len), 3)
    rows = torch.arange(config.packed_height)[:, None].expand(-1, config.packed_width)
    cols = torch.arange(config.packed_width)[None, :].expand(config.packed_height, -1)
    img = torch.stack([torch.zeros_like(rows), rows, cols], dim=-1).reshape(-1, 3).float()
    return txt, img


class NeuronFluxTransformerApplication(TorchNeuronApplicationBase):
    """The Flux DiT, sharded across the TP group, at the config's static shape."""

    block_attrs = ("transformer_blocks", "single_transformer_blocks")

    def __init__(self, *, model_path, config: NeuronFluxConfig, parallel=None,
                 dtype=torch.bfloat16, exec_mode: str | None = None, device="neuron",
                 **kwargs) -> None:
        super().__init__(model_path=model_path, parallel=parallel, dtype=dtype,
                         exec_mode=exec_mode, device=device, **kwargs)
        config.torch_dtype = self.dtype
        config.validate()
        if int(config.tp_degree) != int(self.parallel.tp_degree):
            raise ValueError(f"NeuronFluxConfig has tp_degree={config.tp_degree} but the parallel "
                             f"config has tp_degree={self.parallel.tp_degree}")
        self.config = config
        self._rotary: dict[tuple, torch.Tensor] = {}

    def build_module(self) -> torch.nn.Module:
        from difflet.models.flux.modeling_flux import NeuronFluxTransformer2DModel

        return NeuronFluxTransformer2DModel(self.config)

    def checkpoint_key(self, name: str) -> str:
        return checkpoint_key(name)

    def checkpoint_slices(self) -> dict[str, CheckpointSlice]:
        inner, mlp = self.config.inner_dim, self.config.mlp_hidden_dim
        slices = {}
        for i in range(int(self.config.num_single_layers)):
            source = f"single_transformer_blocks.{i}.proj_out.weight"
            slices[f"single_transformer_blocks.{i}.proj_out_attn.weight"] = CheckpointSlice(source, 1, 0, inner)
            slices[f"single_transformer_blocks.{i}.proj_out_mlp.weight"] = CheckpointSlice(source, 1, inner, mlp)
        return slices

    def rotary_emb(self, txt_ids: torch.Tensor, img_ids: torch.Tensor) -> torch.Tensor:
        """``[S_txt + S_img, head_dim, 2]`` (cos, sin) for these ids, cached per id shape."""
        txt_ids = txt_ids[0] if txt_ids.ndim == 3 else txt_ids
        img_ids = img_ids[0] if img_ids.ndim == 3 else img_ids
        key = (tuple(txt_ids.shape), tuple(img_ids.shape))
        if key not in self._rotary:
            from difflet.layers.embeddings import FluxPosEmbed

            pos_embed = FluxPosEmbed(theta=10000, axes_dim=tuple(self.config.axes_dims_rope))
            ids = torch.cat((txt_ids.float().cpu(), img_ids.float().cpu()), dim=0)
            self._rotary[key] = torch.stack(pos_embed(ids), dim=2).to(self.device, self.dtype)
        return self._rotary[key]

    def get_example_inputs(self) -> tuple[torch.Tensor, ...]:
        """The model's positional inputs at the static shape:
        ``(hidden_states, encoder_hidden_states, pooled_projections, timestep, guidance, rotary)``."""
        cfg, dt = self.config, self.dtype
        batch = int(cfg.batch_size)
        txt, img = flux_position_ids(cfg)
        from difflet.layers.embeddings import FluxPosEmbed

        rotary = torch.stack(FluxPosEmbed(theta=10000, axes_dim=tuple(cfg.axes_dims_rope))(
            torch.cat((txt, img), dim=0)), dim=2).to(dt)
        guidance = torch.ones(batch, dtype=dt) if cfg.guidance_embeds else torch.zeros(0, dtype=dt)
        return (
            torch.zeros(batch, cfg.image_seq_len, int(cfg.in_channels), dtype=dt),
            torch.zeros(batch, int(cfg.text_seq_len), int(cfg.joint_attention_dim), dtype=dt),
            torch.zeros(batch, int(cfg.pooled_projection_dim), dtype=dt),
            torch.ones(batch, dtype=dt),
            guidance,
            rotary,
        )

    @torch.no_grad()
    def teacache_mod_input(self, hidden_states, timestep, pooled_projections, guidance):
        """Block 0's modulated image input, as the Trainium TeaCache probe computes it: the model's
        forward up to ``transformer_blocks[0].norm1`` (timestep and guidance rescaled by 1000 as
        the forward does). Inputs already on this rank's device, in the model's dtype."""
        model = self.module
        hs = model.x_embedder(hidden_states)
        timestep = timestep * 1000
        if self.config.guidance_embeds:
            temb = model.time_text_embed(timestep, guidance * 1000, pooled_projections)
        else:
            temb = model.time_text_embed(timestep, pooled_projections)
        modulated, *_ = model.transformer_blocks[0].norm1(hs, emb=temb, hlomarker=True)
        return modulated

    def forward(self, hidden_states, encoder_hidden_states=None, pooled_projections=None,
                timestep=None, img_ids=None, txt_ids=None, guidance=None,
                joint_attention_kwargs=None, return_dict=False):
        """The pipeline's call, as the Trainium wrapper takes it; returns the model's output."""
        dev, dt = self.device, self.dtype
        batch = hidden_states.shape[0]
        if guidance is None:
            guidance = torch.zeros(0, dtype=dt)
        elif not isinstance(guidance, torch.Tensor):
            guidance = torch.full((batch,), float(guidance), dtype=dt)
        rotary = self.rotary_emb(txt_ids, img_ids)
        args = tuple(t.to(dev, dt) for t in (hidden_states, encoder_hidden_states,
                                             pooled_projections, timestep, guidance))
        return super().forward(*args, rotary)


__all__ = ["NeuronFluxTransformerApplication", "checkpoint_key", "flux_position_ids"]
