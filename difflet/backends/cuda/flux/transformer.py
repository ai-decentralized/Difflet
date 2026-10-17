"""Flux DiT as a CUDA component (CUDA plan F5, task 3).

The module is the backend-neutral ``NeuronFluxTransformer2DModel``, whose
``difflet.ops`` calls bind to the CUDA backend at import time. It is built on the
meta device, materialized in the requested dtype, and filled from the diffusers
``transformer/`` safetensors through ``convert_flux_hf_state_dict`` (tp=1 in
Release 1: the full tensors load as they are). Mirrors
``difflet/backends/cuda/qwen_image/transformer.py``.

``FluxDiTAdapter`` gives the application the call signature ``FluxPipeline`` uses for
its transformer (ids instead of a rope tensor, optional guidance, ``return_dict``),
doing on the host what ``trainium_backbone.ModelWrapperFluxBackbone.forward`` does.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from difflet.backends._torch_megatron.checkpoint import build_weight_map
from difflet.backends._torch_megatron.weights import materialize_meta_
from difflet.backends.cuda.core.application_base import CudaApplicationBase
from difflet.backends.cuda.ops_impl import parallel_mesh
from difflet.layers.embeddings import FluxPosEmbed
from difflet.models.flux.dit_config import FluxDiTConfig
from difflet.pipeline.parallel_mesh import MeshSpec

# FluxPipeline's T5 prompt length (max_sequence_length) — the warm-up shape.
TEXT_SEQ_LEN = 512
ROPE_THETA = 10000


class CudaFluxTransformerApplication(CudaApplicationBase):
    def __init__(self, *, model_path: str, config: FluxDiTConfig, device=None):
        super().__init__(
            config=MeshSpec(tp=int(config.tp_degree)), device=device, dtype=config.torch_dtype
        )
        # The diffusers ``transformer/`` directory (sharded index or a single shard).
        self.model_path = model_path
        self.dit_config = config

    def _init_runtime(self) -> None:
        parallel_mesh.init_parallel_mesh(self.config)

    def build_module(self) -> torch.nn.Module:
        from accelerate import init_empty_weights

        from difflet.models.flux.modeling_flux import NeuronFluxTransformer2DModel

        self._init_runtime()
        with init_empty_weights(include_buffers=False):
            module = NeuronFluxTransformer2DModel(self.dit_config)
        return module

    def _prepare_module(self) -> torch.nn.Module:
        if self.module is None:
            from safetensors.torch import load_file

            from difflet.models.flux.modeling_flux import convert_flux_hf_state_dict

            cfg = self.dit_config
            module = self.build_module()
            materialize_meta_(module, device=self.device, dtype=cfg.torch_dtype)
            state: dict[str, torch.Tensor] = {}
            for path in sorted(set(build_weight_map(self.model_path).values())):
                state.update(load_file(path))
            converted = convert_flux_hf_state_dict(
                state, num_single_layers=int(cfg.num_single_layers), inner_dim=cfg.inner_dim
            )
            del state  # the converter cloned every tensor; don't hold the model twice
            module.load_state_dict(converted, strict=True)
            del converted
            module.to(self.device)
            module.requires_grad_(False)
            module.eval()
            self._assert_on_device(module)
            self.module = module
        return self.module

    def get_example_inputs(self) -> tuple:
        cfg = self.dit_config
        dtype = cfg.torch_dtype
        image_seq_len = int(cfg.image_seq_len)
        guidance = (
            torch.tensor([3.5], dtype=dtype)
            if cfg.guidance_embeds
            else torch.tensor([], dtype=dtype)
        )
        return (
            torch.zeros(1, image_seq_len, int(cfg.in_channels), dtype=dtype),
            torch.zeros(1, TEXT_SEQ_LEN, int(cfg.joint_attention_dim), dtype=dtype),
            torch.zeros(1, int(cfg.pooled_projection_dim), dtype=dtype),
            torch.zeros(1, dtype=dtype),
            guidance,
            torch.zeros(TEXT_SEQ_LEN + image_seq_len, int(cfg.attention_head_dim), 2, dtype=dtype),
        )

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        timestep,
        guidance,
        image_rotary_emb,
    ):
        if self.module is None or not self.is_loaded:
            raise RuntimeError("model is not loaded; call load() first")
        return self.module(
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            guidance,
            image_rotary_emb,
        )


class FluxDiTAdapter(torch.nn.Module):
    """``FluxPipeline``'s transformer interface over a loaded ``CudaFluxTransformerApplication``.

    The pipeline reads ``config.in_channels``, ``config.guidance_embeds`` and ``dtype``,
    enters ``cache_context(name)``, and calls the transformer with ``txt_ids``/``img_ids``;
    the Difflet DiT takes the stacked ``(S, head_dim, 2)`` rope instead, computed here
    with ``FluxPosEmbed`` and cast to the model dtype (as the Trainium wrapper does).
    """

    def __init__(self, app: CudaFluxTransformerApplication):
        super().__init__()
        self.app = app
        self.config = app.dit_config
        self.dtype = app.dit_config.torch_dtype
        self.pos_embed = FluxPosEmbed(theta=ROPE_THETA, axes_dim=self.config.axes_dims_rope)
        # (key, txt_ids, img_ids, rope): the denoising loop passes the same ids every step.
        self._rope_cache = None

    @property
    def device(self) -> torch.device:
        """``DiffusionPipeline.device`` reads ``.device`` off its first module, which is this
        adapter when the pipeline holds no text encoder."""
        return self.app.device

    @contextmanager
    def cache_context(self, name: str):
        """FluxPipeline enters this around each forward; nothing is cached across it here."""
        del name
        yield

    def rotary_embedding(self, txt_ids: torch.Tensor, img_ids: torch.Tensor) -> torch.Tensor:
        """The DiT's ``image_rotary_emb`` for these ids, cached per ids shape."""
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        key = (tuple(txt_ids.shape), tuple(img_ids.shape))
        cached = self._rope_cache
        # Shapes alone are not enough: a 2x8 and an 8x2 latent grid have equal-shaped
        # img_ids with different positions, so the ids themselves must match too.
        if (
            cached is not None
            and cached[0] == key
            and torch.equal(cached[1], txt_ids)
            and torch.equal(cached[2], img_ids)
        ):
            return cached[3]
        ids = torch.cat((txt_ids, img_ids), dim=0)
        rope = torch.stack(self.pos_embed(ids), dim=2).to(device=self.app.device, dtype=self.dtype)
        self._rope_cache = (key, txt_ids.clone(), img_ids.clone(), rope)
        return rope

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        pooled_projections=None,
        timestep=None,
        img_ids=None,
        txt_ids=None,
        guidance=None,
        joint_attention_kwargs=None,
        return_dict=True,
        **ignored,
    ):
        del joint_attention_kwargs, ignored
        device = self.app.device
        if guidance is None:
            guidance = torch.tensor([], dtype=self.dtype, device=device)
        elif not isinstance(guidance, torch.Tensor):
            guidance = torch.tensor([guidance], dtype=self.dtype, device=device)
        else:
            guidance = guidance.to(self.dtype)
        out = self.app(
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep.to(self.dtype),
            guidance,
            self.rotary_embedding(txt_ids, img_ids),
        )
        if not return_dict:
            return (out,)
        return Transformer2DModelOutput(sample=out)


__all__ = ["CudaFluxTransformerApplication", "FluxDiTAdapter", "TEXT_SEQ_LEN"]
