"""LTX-2 application for the neuron backend (TorchNeuron, one process per NeuronCore).

``TorchNeuronLtx2Application`` keeps the outward contract of the Trainium
``NeuronLTX2Application`` (``difflet/models/ltx_2/application.py``): the registry factory
signature, ``compile``/``has_compiled_artifacts``/``load``, ``.pipeline``,
``dit_input_contract()``, ``forward_dit(bundle)``, and a ``__call__`` that sends an
``LTX2DiTInputBundle`` to the DiT and every other call to the ``LTX2Orchestrator``. Every rank of
the torchrun launch builds one and holds:

* ``transformer``: ``NeuronLtx2TransformerApplication``, the audio-video DiT on this rank's core
  (TP shards; under CFG parallelism one branch of the pair per TP group);
* ``text_encoder``: ``_HostLtx2TextEncoder``, on rank 0's host only: diffusers' own
  ``LTX2Pipeline`` text path (the Gemma-3 12B encoder, every hidden state stacked, and the two
  ``LTX2TextConnectors`` that turn them into the video and the audio text streams), in
  ``text_encoder_dtype`` (bf16 by default, about 25 GB of host memory once instead of once per
  rank). Its output is the Trainium path's exactly: the same ``LTX2Orchestrator.
  prepare_conditioning`` call on the same diffusers modules. Rank 0 broadcasts the two streams
  (the masks, bool, travel as int32);
* ``pipeline``: the backend-neutral ``LTX2Orchestrator`` (the two FlowMatch Euler schedulers,
  CFG, the velocity/x0 conversions, on the host in the application dtype, as on Trainium), which
  every rank runs on identical inputs; its DiT calls go through ``_OnDeviceLtx2Transformer``.

The decode (video VAE, audio VAE, vocoder) is not part of this application by default: the CLI
decodes after the generate launch has exited (``difflet/cli/orchestrators/ltx_2.py``), so the
decoder never shares a core or a host process with the DiT. ``enable_decode_components=True``
loads them on rank 0's host in the application dtype (the Trainium CLI's placement) for
``output_type="pt"``.

Guidance: CFG when ``guidance_scale`` > 1 (the orchestrator's, diffusers' formula on x0): two
DiT forwards per step, or with CFG parallelism (``cfg_parallel_enabled``) one batch-2 call
whose branches run on the two CFG groups. STG and modality guidance need extra transformer
kwargs no Difflet LTX-2 DiT takes (``supports_ltx_2_extra_kwargs`` is False, as on Trainium),
so the orchestrator refuses them.

Phase 1 limits, each rejected in ``__init__``: one static shape, batch 1, no TeaCache, no DP
within one application.

Timing hooks (for the benchmark): ``step_hook`` is called with no arguments once per denoise
iteration, after the audio scheduler step that ends it; ``transformer_adapter.forward_hook`` is
called after every DiT forward with that call's wall seconds.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import Any

import torch

from difflet.backends.neuron.core.application_base import _normalize_dtype
from difflet.backends.neuron.core.checkpoint import _resolve_device
from difflet.backends.neuron.core.distributed import broadcast_tensor, rank0_call, world_info
from difflet.backends.neuron.ltx_2.config import LTX_2_TEXT_SEQ_LEN, NeuronLtx2Config
from difflet.backends.neuron.ltx_2.transformer import NeuronLtx2TransformerApplication
from difflet.backends.neuron.runtime import check_parallel_supported
from difflet.models.ltx_2.contract import (
    LTX_2_DEFAULT_HEIGHT,
    LTX_2_DEFAULT_NUM_FRAMES,
    LTX_2_DEFAULT_WIDTH,
    LTX2DiTInputBundle,
)

logger = logging.getLogger(__name__)

#: The keyword arguments the application accepts, with their defaults. The Trainium CLI's
#: generate kwargs (``enable_host_pipeline``, ``enable_decode_components``) are accepted under
#: the same names; anything else is a TypeError rather than silently ignored.
_KWARG_DEFAULTS: dict[str, Any] = {
    "text_seq_len": LTX_2_TEXT_SEQ_LEN,
    "batch_size": 1,
    "shapes": None,
    "frame_rate": 24.0,
    "enable_text_encoder": True,
    "enable_host_pipeline": None,  # the Trainium name for enable_text_encoder
    "enable_transformer": True,
    "enable_decode_components": False,
    "num_layers": None,
    "exec_mode": None,
    "device": "neuron",
    "encoder_threads": 12,
    "text_encoder_dtype": torch.bfloat16,
    "teacache_fused": False,
    "teacache_speedup": None,
    "teacache_calibration": None,
    "teacache_calibration_path": None,
    "teacache_cadence": None,
    "teacache_online_delta_alpha": None,
}
_TEACACHE_KWARGS = (
    "teacache_speedup",
    "teacache_calibration",
    "teacache_calibration_path",
    "teacache_cadence",
    "teacache_online_delta_alpha",
)
#: The orchestrator's default step count when the call names none.
_DEFAULT_STEPS = 4
#: The host encoder's discarded warm-up encode at load (``_HostLtx2TextEncoder._load``).
_WARMUP_PROMPT = "a photo"
_LATENT_CHANNELS = 128
_AUDIO_LATENT_CHANNELS = 8


def _host_ltx2_pipeline(model_path: str, *, dtype: torch.dtype, text: bool, decode: bool):
    """diffusers' ``LTX2Pipeline`` on the host CPU without its transformer, with the text path
    (tokenizer, Gemma-3, connectors) and/or the decode path (video VAE, audio VAE, vocoder);
    the components left out are not read from disk."""
    from difflet.models.ltx_2.pipeline import disable_ltx_2_xla_lazy_import

    disable_ltx_2_xla_lazy_import()
    from diffusers import LTX2Pipeline

    kwargs: dict[str, Any] = {"torch_dtype": dtype, "transformer": None}
    if not text:
        kwargs.update({"text_encoder": None, "tokenizer": None, "connectors": None})
    if not decode:
        kwargs.update({"vae": None, "audio_vae": None, "vocoder": None})
    pipe = LTX2Pipeline.from_pretrained(model_path, **kwargs)
    pipe.to("cpu")
    pipe.set_progress_bar_config(disable=True)
    return pipe


class _HostLtx2TextEncoder:
    """The LTX-2 text path on rank 0's host; every rank receives its output by broadcast.

    ``__call__(prompt, negative_prompt, guidance_scale)`` returns what
    ``LTX2Orchestrator.prepare_conditioning`` returns on Trainium: ``(video_text [B, L, 3840],
    audio_text [B, L, 3840], video_mask [B, L] bool, audio_mask [B, L] bool)``, with B = 2
    (``[negative, positive]``) when ``guidance_scale`` > 1, else 1, and L = ``seq_len``. Rank 0
    computes it with the same call on the same diffusers modules (``_host_ltx2_pipeline``), so
    the bits are the Trainium host's for the same torch build. The encode runs with
    ``threads`` intra-op threads and restores the previous count.

    ``load()`` reads the encoder on rank 0 inside its own status-synced phase
    (``"text encoder load"``) and spends one encode there (see ``_load``).
    """

    def __init__(self, model_path, *, seq_len: int, dtype: torch.dtype, device, threads: int,
                 frame_rate: float, shape: dict[str, int]):
        self.model_path = str(model_path)
        self.seq_len = int(seq_len)
        self.dtype = dtype
        self.device = torch.device(device)
        self.threads = int(threads)
        self.frame_rate = float(frame_rate)
        self.shape = dict(shape)
        self.orchestrator = None  # rank 0 only, after load()

    def load(self) -> None:
        loaded = rank0_call(self._load, device=self.device, what="text encoder load")
        if loaded is not None:
            self.orchestrator = loaded

    def _load(self):
        from difflet.models.ltx_2.pipeline import LTX2Orchestrator

        pipe = _host_ltx2_pipeline(self.model_path, dtype=self.dtype, text=True, decode=False)
        self.orchestrator = LTX2Orchestrator(
            model_path=self.model_path, host_pipeline=pipe, dtype=self.dtype,
            height=self.shape["height"], width=self.shape["width"],
            num_frames=self.shape["num_frames"], text_seq_len=self.seq_len,
            frame_rate=self.frame_rate,
        )
        # As for the Qwen-Image host encoder (6d38344): a fresh process's first bf16 encode is
        # spent and discarded, so every request is encoded by a warmed-up process.
        self._encode(_WARMUP_PROMPT, None, 1.0)
        return self.orchestrator

    def __call__(self, prompt: str, negative_prompt: str | None,
                 guidance_scale: float) -> tuple[torch.Tensor, ...]:
        encoded = rank0_call(self._encode, prompt, negative_prompt, guidance_scale,
                             device=self.device, what="prompt encode")
        out = []
        for i in range(4):
            tensor = None if encoded is None else encoded[i]
            if tensor is not None and tensor.dtype == torch.bool:
                tensor = tensor.to(torch.int32)
            out.append(broadcast_tensor(tensor, device=self.device).cpu())
        video, audio, video_mask, audio_mask = out
        return video, audio, video_mask.to(torch.bool), audio_mask.to(torch.bool)

    def _encode(self, prompt, negative_prompt, guidance_scale):
        if self.orchestrator is None:
            raise RuntimeError("the text encoder is not loaded on rank 0; call load() first")
        previous = torch.get_num_threads()
        torch.set_num_threads(self.threads)
        try:
            with torch.no_grad():
                return tuple(t.contiguous() for t in self.orchestrator.prepare_conditioning(
                    prompt=prompt, negative_prompt=negative_prompt,
                    guidance_scale=float(guidance_scale), max_sequence_length=self.seq_len,
                ))
        finally:
            torch.set_num_threads(previous)


class _OnDeviceLtx2Transformer:
    """The orchestrator's DiT callable: an ``LTX2DiTInputBundle`` in, the host ``(video,
    audio)`` velocity pair out.

    The five model tensors are cast to ``dtype`` and made contiguous on the host before they move
    to ``device`` (a strided host view would be restrided through a CPU round trip on the device,
    decision D16); the RoPE coordinates stay on the host as fp32 (the module computes the rotary
    tables there, see ``difflet/backends/neuron/ltx_2/transformer.py``); the masks and sigma
    are not sent (the DiT attends the text unmasked; sigma is the timestep). ``.dtype``,
    ``.config`` and ``cfg_parallel_enabled`` are what the orchestrator reads.
    """

    supports_ltx_2_extra_kwargs = False
    teacache_probe_fused = False

    def __init__(self, module: Callable[..., Any], config: NeuronLtx2Config, dtype, device):
        self.module = module
        self.config = config
        self.dtype = dtype
        self.device = torch.device(device)
        self.cfg_parallel_enabled = bool(config.cfg_parallel_enabled)
        #: Called after every forward with its wall seconds; see the module docstring.
        self.forward_hook: Callable[[float], None] | None = None
        self.calls = 0

    def __call__(self, bundle: LTX2DiTInputBundle, **kwargs: Any):
        if kwargs:
            raise NotImplementedError(
                f"the LTX-2 DiT takes no extra kwargs on the neuron backend: {sorted(kwargs)} "
                "(STG and modality guidance are not supported)"
            )
        if not isinstance(bundle, LTX2DiTInputBundle):
            raise TypeError(f"expected an LTX2DiTInputBundle, got {type(bundle).__name__}")
        _check_dit_shapes(self.config, bundle)
        start = time.perf_counter()
        args = tuple(
            tensor.to(self.dtype).contiguous().to(self.device)
            for tensor in (bundle.hidden_states, bundle.audio_hidden_states,
                           bundle.encoder_hidden_states, bundle.audio_encoder_hidden_states,
                           bundle.timestep)
        )
        coords = tuple(t.to(torch.float32).contiguous()
                       for t in (bundle.video_coords, bundle.audio_coords))
        video, audio = self.module(*args, *coords)
        if self.device.type == "neuron":
            torch.neuron.synchronize()
        video, audio = video.cpu(), audio.cpu()
        self.calls += 1
        if self.forward_hook is not None:
            self.forward_hook(time.perf_counter() - start)
        return video, audio


def _check_dit_shapes(config: NeuronLtx2Config, bundle: LTX2DiTInputBundle) -> None:
    """The one static shape: the warm-up compiled the blocks for it, so another shape would
    compile again (or, for the text, attend the wrong rows)."""
    batch = config.dit_batch
    expected = {
        "hidden_states": (batch, config.video_seq_len, int(config.in_channels)),
        "audio_hidden_states": (batch, config.audio_seq_len, int(config.audio_in_channels)),
        "encoder_hidden_states": (batch, int(config.text_seq_len), config.video_text_dim),
        "audio_encoder_hidden_states": (batch, int(config.text_seq_len), config.audio_text_dim),
        "timestep": (batch,),
        "video_coords": (batch, 3, config.video_seq_len, 2),
        "audio_coords": (batch, 1, config.audio_seq_len, 2),
    }
    for name, shape in expected.items():
        actual = tuple(getattr(bundle, name).shape)
        if actual != shape:
            raise ValueError(
                f"LTX-2 DiT input {name!r} has shape {actual}, expected {shape} (the one static "
                f"shape, {config.height}x{config.width}x{config.num_frames}, batch {batch})"
            )


def _resolve_shape(shape: dict[str, int | None], shapes) -> dict[str, int]:
    resolved = {
        "height": int(shape.get("height") or LTX_2_DEFAULT_HEIGHT),
        "width": int(shape.get("width") or LTX_2_DEFAULT_WIDTH),
        "num_frames": int(shape.get("num_frames") or LTX_2_DEFAULT_NUM_FRAMES),
    }
    if shapes:
        shapes = [tuple(entry) for entry in shapes]
        if len(shapes) > 1:
            raise NotImplementedError(
                "the neuron backend compiles one static shape per process (phase 1); "
                f"got shapes={shapes}"
            )
        bucket = tuple(int(v) for v in shapes[0])
        if bucket != (resolved["height"], resolved["width"], resolved["num_frames"]):
            raise ValueError(f"the request shape {resolved} is not the one compiled shape {bucket}")
    return resolved


def _reject_unsupported(options: dict[str, Any]) -> None:
    teacache = [name for name in _TEACACHE_KWARGS if options[name] is not None]
    if options["teacache_fused"]:
        teacache.append("teacache_fused")
    if teacache:
        raise NotImplementedError(
            f"TeaCache is not supported on the neuron backend in phase 1 ({', '.join(teacache)})"
        )
    if int(options["batch_size"]) != 1:
        raise NotImplementedError(
            "the neuron LTX-2 application runs batch_size=1 only (phase 1), got "
            f"batch_size={options['batch_size']}"
        )


class TorchNeuronLtx2Application(torch.nn.Module):
    """Host text path on rank 0 + the DiT on each rank's core + ``LTX2Orchestrator``, on every
    rank.

    Factory signature (``difflet/registry.py`` ``create_application``): ``model_path`` is the
    diffusers snapshot root, ``shape`` the request's ``height``/``width``/``num_frames``. Keyword
    arguments and their defaults are ``_KWARG_DEFAULTS``; ``device`` is ``"neuron"`` except in
    CPU tests (``"cpu"`` on gloo).
    """

    def __init__(self, *, model_path, parallel, dtype: Any, shape: dict[str, int | None],
                 **kwargs: Any) -> None:
        super().__init__()
        unknown = sorted(set(kwargs) - set(_KWARG_DEFAULTS))
        if unknown:
            raise TypeError(
                f"{type(self).__name__} got unexpected keyword arguments: {', '.join(unknown)}"
            )
        options = {**_KWARG_DEFAULTS, **kwargs}
        _reject_unsupported(options)
        check_parallel_supported(parallel)
        self.model_path = str(model_path)
        self.parallel = parallel
        self.dtype = _normalize_dtype(dtype)
        self.device = _resolve_device(options["device"])
        self.shape = _resolve_shape(shape, options["shapes"])
        self.text_seq_len = int(options["text_seq_len"])
        self.frame_rate = float(options["frame_rate"])
        self.batch_size = 1
        transformer_path = os.path.join(self.model_path, "transformer")
        overrides: dict[str, Any] = dict(
            height=self.shape["height"], width=self.shape["width"],
            num_frames=self.shape["num_frames"], frame_rate=self.frame_rate,
            text_seq_len=self.text_seq_len, batch_size=1,
            tp_degree=int(parallel.tp_degree),
            cfg_parallel_enabled=bool(getattr(parallel, "cfg_parallel_enabled", False)),
            torch_dtype=self.dtype,
        )
        if options["num_layers"] is not None:
            overrides["num_layers"] = int(options["num_layers"])
        self.config = NeuronLtx2Config.from_pretrained(transformer_path, **overrides)
        self.cfg_parallel_enabled = bool(self.config.cfg_parallel_enabled)
        self.transformer: NeuronLtx2TransformerApplication | None = None
        if options["enable_transformer"]:
            self.transformer = NeuronLtx2TransformerApplication(
                model_path=transformer_path, config=self.config, parallel=parallel,
                dtype=self.dtype, exec_mode=options["exec_mode"], device=self.device,
            )
        enable_text = options["enable_text_encoder"]
        if options["enable_host_pipeline"] is not None:
            enable_text = bool(options["enable_host_pipeline"])
        self.text_encoder: _HostLtx2TextEncoder | None = None
        if enable_text:
            self.text_encoder = _HostLtx2TextEncoder(
                self.model_path, seq_len=self.text_seq_len,
                dtype=_normalize_dtype(options["text_encoder_dtype"]), device=self.device,
                threads=int(options["encoder_threads"]), frame_rate=self.frame_rate,
                shape=self.shape,
            )
        self.enable_decode_components = bool(options["enable_decode_components"])
        self.decoder = None  # rank 0's host LTX2Pipeline decode components, after load()
        self.transformer_adapter: _OnDeviceLtx2Transformer | None = None
        self.pipeline = None
        #: Called with no arguments once per denoise iteration; see the module docstring.
        self.step_hook: Callable[[], None] | None = None
        self._text_encoder_load_seconds: float | None = None

    # --------------------------------------------------- DiffletPipeline API

    def compile(self, compiled_model_path, debug: bool = False) -> None:
        """No-op: the neuron backend compiles nothing ahead of time (see ``load``)."""
        del compiled_model_path, debug

    def has_compiled_artifacts(self, compiled_model_path) -> bool:
        del compiled_model_path
        return True

    def load(self, compiled_model_path=None, start_rank_id=None, local_ranks_size=None,
             skip_warmup: bool = False) -> None:
        """DiT (runtime init, shards, per-block compile, warm-up), then the text path on rank 0,
        then (if enabled) the decode components on rank 0, then the orchestrator. Every rank must
        call it; the steps are status-synced phases."""
        del compiled_model_path, start_rank_id, local_ranks_size
        if self.pipeline is not None:
            return
        from difflet.models.ltx_2.pipeline import LTX2Orchestrator

        if self.transformer is not None:
            self.transformer.load(skip_warmup=skip_warmup)
            self.transformer_adapter = _OnDeviceLtx2Transformer(
                self.transformer, self.config, self.dtype, self.device
            )
        if self.text_encoder is not None:
            start = time.perf_counter()
            self.text_encoder.load()
            self._text_encoder_load_seconds = time.perf_counter() - start
        if self.enable_decode_components:
            self.decoder = rank0_call(
                _host_ltx2_pipeline, self.model_path, dtype=self.dtype, text=False, decode=True,
                device=self.device, what="decoder load",
            )
        decoder = self.decoder
        pipeline = LTX2Orchestrator(
            model_path=self.model_path,
            transformer=self.transformer_adapter,
            vae=getattr(decoder, "vae", None),
            audio_vae=getattr(decoder, "audio_vae", None),
            vocoder=getattr(decoder, "vocoder", None),
            video_processor=getattr(decoder, "video_processor", None),
            host_pipeline=decoder,
            dtype=self.dtype,
            height=self.shape["height"],
            width=self.shape["width"],
            num_frames=self.shape["num_frames"],
            text_seq_len=self.text_seq_len,
            frame_rate=self.frame_rate,
        )
        self._check_orchestrator_geometry(pipeline)
        self._stamp_denoise_iterations(pipeline)
        self.pipeline = pipeline

    def _check_orchestrator_geometry(self, pipeline) -> None:
        """The orchestrator derives its latent shapes from the VAE ratios (diffusers defaults
        on ranks without the decode components); they must be the DiT's."""
        cfg = self.config
        mismatch = {
            name: (getattr(pipeline, name), value)
            for name, value in (("video_seq_len", cfg.video_seq_len),
                                ("audio_seq_len", cfg.audio_seq_len),
                                ("latent_num_frames", cfg.latent_num_frames))
            if int(getattr(pipeline, name)) != int(value)
        }
        if mismatch:
            raise ValueError(f"LTX-2 orchestrator and DiT disagree on the geometry: {mismatch}")

    def _stamp_denoise_iterations(self, pipeline) -> None:
        """Call ``step_hook`` after each audio scheduler step: once per denoise iteration."""
        scheduler_step = pipeline._scheduler_step

        def stamped_scheduler_step(scheduler, *args: Any, **kwargs: Any) -> torch.Tensor:
            latents = scheduler_step(scheduler, *args, **kwargs)
            if self.step_hook is not None and scheduler is not pipeline.scheduler:
                self.step_hook()
            return latents

        pipeline._scheduler_step = stamped_scheduler_step

    # ------------------------------------------------------------- evidence

    @property
    def phase_seconds(self) -> dict[str, float]:
        seconds = dict(self.transformer.phase_seconds) if self.transformer is not None else {}
        if self._text_encoder_load_seconds is not None:
            seconds["text encoder load"] = self._text_encoder_load_seconds
        return seconds

    @property
    def compiled_blocks(self) -> list[str]:
        return list(self.transformer.compiled_blocks) if self.transformer is not None else []

    @property
    def unwarmed_shapes(self) -> list:
        return list(self.transformer.unwarmed_shapes) if self.transformer is not None else []

    def dit_input_contract(self) -> dict[str, dict[str, Any]]:
        """The Trainium contract's ten tensors at the DiT batch; the DiT reads seven (the masks
        and sigma are accepted and dropped, see ``_OnDeviceLtx2Transformer``)."""
        cfg = self.config
        b = cfg.dit_batch
        text = int(cfg.text_seq_len)
        return {
            "hidden_states": {"shape": (b, cfg.video_seq_len, int(cfg.in_channels)),
                              "dtype": self.dtype},
            "audio_hidden_states": {"shape": (b, cfg.audio_seq_len, int(cfg.audio_in_channels)),
                                    "dtype": self.dtype},
            "encoder_hidden_states": {"shape": (b, text, cfg.video_text_dim),
                                      "dtype": self.dtype},
            "audio_encoder_hidden_states": {"shape": (b, text, cfg.audio_text_dim),
                                            "dtype": self.dtype},
            "timestep": {"shape": (b,), "dtype": self.dtype},
            "sigma": {"shape": (b,), "dtype": self.dtype},
            "encoder_attention_mask": {"shape": (b, text), "dtype": torch.bool},
            "audio_encoder_attention_mask": {"shape": (b, text), "dtype": torch.bool},
            "video_coords": {"shape": (b, 3, cfg.video_seq_len, 2), "dtype": torch.float32},
            "audio_coords": {"shape": (b, 1, cfg.audio_seq_len, 2), "dtype": torch.float32},
        }

    # ------------------------------------------------------------- inference

    def _require_dit(self) -> _OnDeviceLtx2Transformer:
        if self.transformer_adapter is None:
            raise RuntimeError(
                f"{type(self).__name__}: the DiT is not loaded (call load() first; "
                "enable_transformer=False never loads one)"
            )
        return self.transformer_adapter

    def forward_dit(self, bundle: LTX2DiTInputBundle):
        """One DiT forward on a host bundle: ``(video [B, S_v, 128], audio [B, S_a, 128])``."""
        return self._require_dit()(bundle)

    def encode_prompt(self, prompt, negative_prompt=None, guidance_scale: float = 1.0):
        """The four conditioning tensors (see ``_HostLtx2TextEncoder``), identical on every rank.
        Every rank must call it."""
        if self.text_encoder is None:
            raise RuntimeError(
                f"{type(self).__name__}: no text encoder (enable_text_encoder=False); pass "
                "encoder_hidden_states (and the audio stream and masks) instead of a prompt"
            )
        if isinstance(prompt, (list, tuple)):
            if len(prompt) != 1:
                raise NotImplementedError(
                    f"the neuron LTX-2 application runs batch 1, got {len(prompt)} prompts"
                )
            prompt = prompt[0]
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a string, got {type(prompt).__name__}")
        if isinstance(negative_prompt, (list, tuple)):
            negative_prompt = negative_prompt[0] if negative_prompt else None
        return self.text_encoder(prompt, negative_prompt, guidance_scale)

    def generate(self, *, prompt=None, negative_prompt=None, seed: int | None = None,
                 **kwargs: Any):
        """The orchestrator's ``__call__`` (``num_inference_steps``, ``guidance_scale``,
        ``generator``, ``output_type`` 'latent' | 'pt', ...), with:

        * ``prompt`` / ``negative_prompt``: encoded by ``encode_prompt`` (rank 0) into the four
          conditioning arguments, ``[negative, positive]`` when ``guidance_scale`` > 1;
        * ``seed``: a ``torch.Generator`` seeded with it (pass one or ``generator``);
        * the initial noise: drawn on rank 0 by the orchestrator's own ``prepare_latents`` then
          ``prepare_audio_latents`` from ``generator`` (the Trainium path's order) and
          broadcast, so a TP group denoises the same latents even when the global RNGs differ.

        Under CFG parallelism a call without CFG is refused: the DiT then takes only the
        batch-2 pair.
        """
        if self.pipeline is None:
            raise RuntimeError(f"{type(self).__name__} is not loaded; call load() first")
        guidance_scale = float(kwargs.get("guidance_scale", 1.0))
        audio_guidance = kwargs.get("audio_guidance_scale")
        audio_guidance = guidance_scale if audio_guidance is None else float(audio_guidance)
        do_cfg = guidance_scale > 1.0 or audio_guidance > 1.0
        if self.cfg_parallel_enabled and not do_cfg:
            raise ValueError(
                "cfg_parallel_enabled splits the two branches of classifier-free guidance over "
                "the CFG group; pass guidance_scale > 1, or run without CFG parallelism"
            )
        if prompt is not None:
            if kwargs.get("encoder_hidden_states") is not None:
                raise ValueError("pass prompt or encoder_hidden_states, not both")
            video, audio, video_mask, audio_mask = self.encode_prompt(
                prompt, negative_prompt, max(guidance_scale, audio_guidance))
            kwargs.update(encoder_hidden_states=video, audio_encoder_hidden_states=audio,
                          encoder_attention_mask=video_mask,
                          audio_encoder_attention_mask=audio_mask)
        elif negative_prompt is not None:
            raise ValueError("negative_prompt needs prompt")
        if seed is not None:
            if kwargs.get("generator") is not None:
                raise ValueError("pass seed or generator, not both")
            kwargs["generator"] = torch.Generator().manual_seed(int(seed))
        if kwargs.get("bundle") is None and kwargs.get("latents") is None:
            kwargs["latents"], kwargs["audio_latents"] = self._initial_latents(kwargs)
        kwargs.setdefault("num_inference_steps", _DEFAULT_STEPS)
        return self.pipeline(**kwargs)

    def _initial_latents(self, kwargs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Rank 0's packed initial video and audio noise, broadcast (see ``generate``)."""
        pipeline = self.pipeline
        generator = kwargs.get("generator")

        def draw() -> tuple[torch.Tensor, torch.Tensor]:
            video = pipeline.prepare_latents(
                batch_size=1, channels=int(kwargs.get("video_channels", _LATENT_CHANNELS)),
                dtype=pipeline.dtype, generator=generator)
            audio = pipeline.prepare_audio_latents(
                batch_size=1, channels=int(kwargs.get("audio_channels", _AUDIO_LATENT_CHANNELS)),
                dtype=pipeline.dtype, generator=generator)
            return video.contiguous(), audio.contiguous()

        if world_info()[1] == 1:
            return draw()
        drawn = rank0_call(draw, device=self.device, what="initial latents")
        video = broadcast_tensor(None if drawn is None else drawn[0], device=self.device).cpu()
        audio = broadcast_tensor(None if drawn is None else drawn[1], device=self.device).cpu()
        return video, audio

    def __call__(self, *args: Any, **kwargs: Any):
        """``app(bundle)`` runs the DiT; ``app("prompt", ...)`` and every keyword-only call run
        ``generate`` (the orchestrator)."""
        if len(args) == 1 and isinstance(args[0], LTX2DiTInputBundle) and not kwargs:
            return self.forward_dit(args[0])
        if len(args) == 1 and isinstance(args[0], (str, list)):
            if "prompt" in kwargs:
                raise TypeError("prompt given both positionally and as a keyword")
            kwargs["prompt"] = args[0]
            args = ()
        if args:
            raise TypeError(
                f"{type(self).__name__}: positional arguments are a DiT bundle or a prompt; got "
                f"{len(args)}"
            )
        return self.generate(**kwargs)


__all__ = ["TorchNeuronLtx2Application"]
