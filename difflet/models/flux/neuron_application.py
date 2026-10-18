"""Flux on the neuron backend: our DiT, T5 and VAE applications under ``NeuronFluxPipeline``.

For FLUX.1-dev (guidance-distilled, no true CFG, no TeaCache) ``NeuronFluxPipeline`` runs diffusers'
own ``FluxPipeline.__call__``, so the denoising loop, scheduler, latent packing and guidance are
diffusers'. The components it calls are adapters onto the neuron applications:

- ``transformer``: the DiT application (TP across the ranks), output returned to the host;
- ``text_encoder_2``: the T5 application (TP across the ranks), embeddings returned to the host;
- ``text_encoder``: transformers' ``CLIPTextModel`` on the host (one pooled vector per prompt; the
  whole 0.25 GB model on every rank, deterministic, so no broadcast);
- ``vae``: the VAE application on rank 0 only; the other ranks return latents.

The adapters are plain objects, not ``nn.Module``s holding device tensors: diffusers infers the
pipeline's execution device from its modules, and the loop and scheduler belong on the host (each
application moves its own inputs).

Factory signature as ``TorchNeuronWanApplication``'s: ``model_path`` is the diffusers snapshot root,
``shape`` the request's ``height``/``width``.
"""

from __future__ import annotations

import contextlib
import os
from types import SimpleNamespace
from typing import Any

import torch

_KWARG_DEFAULTS: dict[str, Any] = {
    "device": "neuron",
    "exec_mode": None,
    "text_seq_len": 512,
    "t5_dtype": torch.float32,
}


class _Transformer:
    """The DiT application as ``FluxPipeline.transformer``."""

    def __init__(self, app) -> None:
        self.app = app
        self.config = app.config
        self.dtype = app.dtype

    @contextlib.contextmanager
    def image_rotary_emb_cache_context(self):
        yield  # the application caches the rotary table per shape

    @contextlib.contextmanager
    def cache_context(self, name):
        yield  # diffusers' cache hooks (FasterCache etc.) are not used here

    def __call__(self, **kwargs):
        out = self.app(**kwargs)
        out = out[0] if isinstance(out, (tuple, list)) else out
        return (out.to("cpu"),)


class _T5:
    """The T5 application as ``FluxPipeline.text_encoder_2``: ``(ids) -> (embeds,)`` on the host."""

    def __init__(self, app, dtype) -> None:
        self.app = app
        self.dtype = dtype

    def __call__(self, input_ids, output_hidden_states=False):
        return (self.app.encode(input_ids).to("cpu", self.dtype),)


class _Vae:
    """The VAE application as ``FluxPipeline.vae``: ``config`` and ``decode`` on the host."""

    def __init__(self, app) -> None:
        self.app = app
        cfg = app.vae_config
        self.config = SimpleNamespace(
            scaling_factor=cfg.get("scaling_factor", 0.3611),
            shift_factor=cfg.get("shift_factor", 0.1159),
            block_out_channels=list(cfg.get("block_out_channels", [128, 256, 512, 512])),
            latent_channels=int(cfg.get("latent_channels", 16)),
        )
        self.dtype = app.dtype

    def decode(self, latents, return_dict=False):
        return (self.app.decode(latents).to("cpu"),)


def _rank() -> int:
    import torch.distributed as dist

    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


class TorchNeuronFluxApplication:
    """Host CLIP + the DiT and T5 on every rank's core + the VAE on rank 0, under
    ``NeuronFluxPipeline``. Keyword options and defaults are ``_KWARG_DEFAULTS``."""

    def __init__(self, *, model_path, parallel, dtype: Any = torch.bfloat16,
                 shape: dict[str, int], **kwargs: Any) -> None:
        unknown = sorted(set(kwargs) - set(_KWARG_DEFAULTS))
        if unknown:
            raise TypeError(f"{type(self).__name__} got unexpected keyword arguments: {', '.join(unknown)}")
        self.options = {**_KWARG_DEFAULTS, **kwargs}
        self.model_path = str(model_path)
        self.parallel = parallel
        self.dtype = dtype
        self.height, self.width = int(shape["height"]), int(shape["width"])
        self.pipeline = None

    def _path(self, name: str) -> str:
        return os.path.join(self.model_path, name)

    def load(self) -> None:
        from diffusers import FlowMatchEulerDiscreteScheduler

        from difflet.backends.neuron.flux.config import NeuronFluxConfig
        from difflet.backends.neuron.flux.text_encoder import NeuronFluxT5Application
        from difflet.backends.neuron.flux.transformer import NeuronFluxTransformerApplication
        from difflet.backends.neuron.flux.vae import NeuronFluxVaeDecoderApplication
        from difflet.models.flux.pipeline import NeuronFluxPipeline

        opts, parallel = self.options, self.parallel
        common = dict(parallel=parallel, exec_mode=opts["exec_mode"], device=opts["device"])
        rank0 = _rank() == 0

        vae = None
        if rank0:
            vae = NeuronFluxVaeDecoderApplication(model_path=self._path("vae"), height=self.height,
                                                  width=self.width, device=opts["device"])
            vae.load()
        vae_scale = vae.scale_factor if vae is not None else 8
        config = NeuronFluxConfig.from_pretrained(
            self._path("transformer"), height=self.height, width=self.width,
            text_seq_len=int(opts["text_seq_len"]), vae_scale_factor=vae_scale,
            tp_degree=int(parallel.tp_degree))
        dit = NeuronFluxTransformerApplication(model_path=self._path("transformer"), config=config,
                                               dtype=self.dtype, **common)
        dit.load()

        t5 = None
        if os.path.isdir(self._path("text_encoder_2")):
            t5 = NeuronFluxT5Application(model_path=self._path("text_encoder_2"),
                                         text_seq_len=int(opts["text_seq_len"]),
                                         dtype=opts["t5_dtype"], **common)
            t5.load()
        clip = tokenizer = tokenizer_2 = None
        if os.path.isdir(self._path("text_encoder")):
            from transformers import CLIPTextModel

            clip = CLIPTextModel.from_pretrained(self._path("text_encoder")).eval()
        if os.path.isdir(self._path("tokenizer")):
            from transformers import CLIPTokenizer

            tokenizer = CLIPTokenizer.from_pretrained(self._path("tokenizer"))
        if os.path.isdir(self._path("tokenizer_2")):
            from transformers import T5TokenizerFast

            tokenizer_2 = T5TokenizerFast.from_pretrained(self._path("tokenizer_2"))
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self._path("scheduler"))

        self.dit, self.t5, self.vae = dit, t5, vae
        self.pipeline = NeuronFluxPipeline(
            scheduler=scheduler, vae=_Vae(vae) if vae is not None else None, text_encoder=clip,
            tokenizer=tokenizer, text_encoder_2=_T5(t5, self.dtype) if t5 is not None else None,
            tokenizer_2=tokenizer_2, transformer=_Transformer(dit))
        if vae is None:
            self.pipeline.vae_scale_factor = vae_scale  # diffusers' default without a VAE is 8

    @torch.no_grad()
    def __call__(self, **kwargs):
        """``NeuronFluxPipeline``'s call; ranks other than 0 return latents (only rank 0 decodes)."""
        if self.pipeline is None:
            raise RuntimeError("call load() first")
        kwargs.setdefault("height", self.height)
        kwargs.setdefault("width", self.width)
        kwargs.setdefault("max_sequence_length", int(self.options["text_seq_len"]))
        if _rank() != 0:
            kwargs["output_type"] = "latent"
        return self.pipeline(**kwargs)


__all__ = ["TorchNeuronFluxApplication"]
