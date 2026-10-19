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
import time
from types import SimpleNamespace
from typing import Any

import torch

_KWARG_DEFAULTS: dict[str, Any] = {
    "device": "neuron",
    "exec_mode": None,
    "text_seq_len": 512,
    # bf16 passes the parity gate on the device (0.87x the bf16 control vs fp32) and halves T5's
    # memory per core; fp32 remains an option.
    "t5_dtype": torch.bfloat16,
    # False: no VAE is loaded and every call returns packed latents (the CLI decodes them in a
    # separate stage, as for Wan, so the decoder never shares a core with the DiT and T5 shards).
    "decode_vae": True,
    # TeaCache, as the Trainium application takes it: the adaptive modes (teacache_fused,
    # teacache_speedup with a calibration) use the probe; the probe-free ones (teacache_cadence,
    # teacache_online_delta_alpha) decide on the host alone.
    "teacache_fused": False,
    "teacache_speedup": None,
    "teacache_calibration": None,
    "teacache_calibration_path": None,
    "teacache_cadence": None,
    "teacache_online_delta_alpha": None,
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
        # In-process decode (decode_vae=True): the denoising loop leaves the DiT's per-step
        # activation blocks cached by the device allocator; return them first (at TP2 the
        # 1024x1024 decode otherwise crawls through OOM recovery).
        empty_cache = getattr(getattr(torch, "neuron", None), "empty_cache", None)
        if callable(empty_cache) and str(self.app.device).startswith("neuron"):
            empty_cache()
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
        #: Wall seconds per component of the last ``load`` (each application's own phases are on
        #: ``dit``, ``t5`` and ``vae``).
        self.phase_seconds: dict[str, float] = {}

    def _path(self, name: str) -> str:
        return os.path.join(self.model_path, name)

    def load(self, compiled_model_path=None, skip_warmup: bool = False) -> None:
        """Build and load every component (``compiled_model_path`` is an AoT notion: unused).
        ``skip_warmup`` skips each application's warm-up forward."""
        from diffusers import FlowMatchEulerDiscreteScheduler

        from difflet.backends.neuron.flux.config import NeuronFluxConfig
        from difflet.backends.neuron.flux.text_encoder import NeuronFluxT5Application
        from difflet.backends.neuron.flux.transformer import NeuronFluxTransformerApplication
        from difflet.backends.neuron.flux.vae import NeuronFluxVaeDecoderApplication
        from difflet.models.flux.pipeline import NeuronFluxPipeline

        opts, parallel = self.options, self.parallel
        common = dict(parallel=parallel, exec_mode=opts["exec_mode"], device=opts["device"])
        phases = self.phase_seconds = {}
        mark = time.perf_counter()

        def lap(name):
            nonlocal mark
            now = time.perf_counter()
            phases[name] = round(now - mark, 2)
            mark = now

        # Every rank loads the VAE (the load phases run in step across the process group, as for
        # Wan's device VAE); it is not sharded, so each holds a full copy and only rank 0 decodes.
        vae = None
        if opts["decode_vae"]:
            vae = NeuronFluxVaeDecoderApplication(model_path=self._path("vae"), height=self.height,
                                                  width=self.width, parallel=parallel,
                                                  device=opts["device"])
            vae.load(skip_warmup=skip_warmup)
            vae_scale = vae.scale_factor
        else:
            import json

            with open(self._path("vae/config.json"), encoding="utf-8") as f:
                vae_scale = 2 ** (len(json.load(f)["block_out_channels"]) - 1)
        lap("vae")
        config = NeuronFluxConfig.from_pretrained(
            self._path("transformer"), height=self.height, width=self.width,
            text_seq_len=int(opts["text_seq_len"]), vae_scale_factor=vae_scale,
            tp_degree=int(parallel.tp_degree),
            cp_degree=int(getattr(parallel, "cp_degree", 1) or 1),
            context_parallel_enabled=int(getattr(parallel, "cp_degree", 1) or 1) > 1,
            cp_mode=getattr(parallel, "cp_mode", "gather_kv") or "gather_kv",
            sp_enabled=bool(getattr(parallel, "sp_enabled", False)))
        dit = NeuronFluxTransformerApplication(model_path=self._path("transformer"), config=config,
                                               dtype=self.dtype, **common)
        dit.load(skip_warmup=skip_warmup)
        lap("dit")

        t5 = None
        if os.path.isdir(self._path("text_encoder_2")):
            t5 = NeuronFluxT5Application(model_path=self._path("text_encoder_2"),
                                         text_seq_len=int(opts["text_seq_len"]),
                                         dtype=opts["t5_dtype"], **common)
            t5.load(skip_warmup=skip_warmup)
        lap("t5")
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
        lap("clip+tokenizers")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self._path("scheduler"))

        self.dit, self.t5, self.vae = dit, t5, vae
        self.pipeline = NeuronFluxPipeline(
            scheduler=scheduler, vae=_Vae(vae) if vae is not None else None, text_encoder=clip,
            tokenizer=tokenizer, text_encoder_2=_T5(t5, self.dtype) if t5 is not None else None,
            tokenizer_2=tokenizer_2, transformer=_Transformer(dit))
        self.pipeline.vae_scale_factor = vae_scale  # diffusers' default without a VAE is 8
        self._configure_teacache()
        lap("scheduler+pipeline")

    def _configure_teacache(self) -> None:
        """Mount the probe and the controller on the pipeline as the Trainium application does
        (same calibration checks, same probe-free controller); a no-op with TeaCache off."""
        opts, pipe = self.options, self.pipeline
        pipe.teacache_probe = pipe.teacache_controller = pipe.teacache_speedup = None
        adaptive = bool(opts["teacache_fused"]) or opts["teacache_speedup"] is not None
        probe_free = opts["teacache_cadence"] is not None or opts["teacache_online_delta_alpha"] is not None
        if adaptive and probe_free:
            raise ValueError("teacache_cadence/teacache_online_delta_alpha are mutually exclusive "
                             "with the adaptive probe modes (teacache_fused/teacache_speedup).")
        from difflet.pipeline.teacache import (
            TeaCacheCalibration,
            TeaCacheController,
            load_teacache_calibration_or_raise,
        )

        shape_label = f"{self.height}x{self.width}"
        if adaptive:
            from difflet.backends.neuron.flux.teacache import NeuronFluxTeaCacheProbe

            pipe.teacache_probe = NeuronFluxTeaCacheProbe(self.dit)
            speedup = opts["teacache_speedup"]
            if speedup is not None:
                calibration, path = opts["teacache_calibration"], opts["teacache_calibration_path"]
                if calibration is not None and path is not None:
                    raise ValueError("teacache_calibration and teacache_calibration_path are "
                                     "mutually exclusive")
                if calibration is None:
                    calibration = load_teacache_calibration_or_raise(path, model="flux",
                                                                     shape_label=shape_label)
                elif calibration.model != "flux" or calibration.shape_label != shape_label:
                    raise ValueError("TeaCache calibration does not match Flux model/profile")
                if (calibration.target_speedup is not None
                        and float(speedup) > float(calibration.target_speedup) + 1e-6):
                    raise ValueError(
                        "Flux TeaCache calibration target speedup is lower than requested: "
                        f"requested {speedup}, calibration has {calibration.target_speedup}.")
                pipe.teacache_controller = TeaCacheController(calibration)
                pipe.teacache_speedup = float(speedup)
        elif probe_free:
            pipe.teacache_controller = TeaCacheController(TeaCacheCalibration(
                model="flux", shape_label=shape_label, num_steps=0, poly_coef=(0.0,), threshold=0.0,
                cadence=int(opts["teacache_cadence"] or 0),
                online_delta_alpha=float(opts["teacache_online_delta_alpha"] or 0.0)))

    @torch.no_grad()
    def __call__(self, **kwargs):
        """``NeuronFluxPipeline``'s call; ranks other than 0 return latents (only rank 0 decodes)."""
        if self.pipeline is None:
            raise RuntimeError("call load() first")
        kwargs.setdefault("height", self.height)
        kwargs.setdefault("width", self.width)
        kwargs.setdefault("max_sequence_length", int(self.options["text_seq_len"]))
        if _rank() != 0 or self.vae is None:
            kwargs["output_type"] = "latent"
        return self.pipeline(**kwargs)


__all__ = ["TorchNeuronFluxApplication"]
