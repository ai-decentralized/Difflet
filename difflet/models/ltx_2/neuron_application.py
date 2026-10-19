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

The lifecycle, the evidence properties, the hooks, the rank-0 encoder plumbing and the
initial-latents broadcast are the composed neuron application's (``TorchNeuronModelApplication``,
``difflet/backends/neuron/core/model_application.py``); what is LTX-2's is here.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from difflet.backends.neuron.core.application_base import _normalize_dtype
from difflet.backends.neuron.core.distributed import rank0_call, world_info
from difflet.backends.neuron.core.model_application import (
    DeviceComponent,
    DeviceTextEncoder,
    HostTextEncoder,
    OnDeviceDiT,
    TorchNeuronModelApplication,
    broadcast_from_rank0,
    intra_op_threads,
)
from difflet.backends.neuron.ltx_2.config import LTX_2_TEXT_SEQ_LEN, NeuronLtx2Config
from difflet.backends.neuron.ltx_2.transformer import NeuronLtx2TransformerApplication
from difflet.models.ltx_2.contract import (
    LTX_2_DEFAULT_HEIGHT,
    LTX_2_DEFAULT_NUM_FRAMES,
    LTX_2_DEFAULT_WIDTH,
    LTX2DiTInputBundle,
)

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
    "text_encoder_device": "host",
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
#: The host encoder's discarded warm-up encode at load (``HostTextEncoder.warmup_args``).
_WARMUP_PROMPT = "a photo"
_LATENT_CHANNELS = 128
_AUDIO_LATENT_CHANNELS = 8
#: The four conditioning tensors' indices that are bool masks (they travel as int32).
_MASKS = (2, 3)


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


class _HostLtx2TextEncoder(HostTextEncoder):
    """The LTX-2 text path on rank 0's host; every rank receives its output by broadcast.

    ``__call__(prompt, negative_prompt, guidance_scale)`` returns what
    ``LTX2Orchestrator.prepare_conditioning`` returns on Trainium: ``(video_text [B, L, 3840],
    audio_text [B, L, 3840], video_mask [B, L] bool, audio_mask [B, L] bool)``, with B = 2
    (``[negative, positive]``) when ``guidance_scale`` > 1, else 1, and L = ``seq_len``. Rank 0
    computes it with the same call on the same diffusers modules (``_host_ltx2_pipeline``), so
    the bits are the Trainium host's for the same torch build (``HostTextEncoder``: with
    ``threads`` intra-op threads, inside ``rank0_call``).

    ``load()`` reads the encoder on rank 0 and, as for the Qwen-Image host encoder (6d38344),
    spends one discarded bf16 encode (``warmup_args``), so every request is encoded by a
    warmed-up process.
    """

    warmup_args = (_WARMUP_PROMPT, None, 1.0)

    def __init__(self, model_path, *, seq_len: int, dtype: torch.dtype, device, threads: int,
                 frame_rate: float, shape: dict[str, int]):
        super().__init__(model_path, device=device, threads=threads)
        self.seq_len = int(seq_len)
        self.dtype = dtype
        self.frame_rate = float(frame_rate)
        self.shape = dict(shape)
        self.orchestrator = None  # rank 0 only, after load()

    def _load(self) -> None:
        from difflet.models.ltx_2.pipeline import LTX2Orchestrator

        pipe = _host_ltx2_pipeline(self.model_path, dtype=self.dtype, text=True, decode=False)
        self.orchestrator = LTX2Orchestrator(
            model_path=self.model_path, host_pipeline=pipe, dtype=self.dtype,
            height=self.shape["height"], width=self.shape["width"],
            num_frames=self.shape["num_frames"], text_seq_len=self.seq_len,
            frame_rate=self.frame_rate,
        )

    def __call__(self, prompt: str, negative_prompt: str | None,
                 guidance_scale: float) -> tuple[torch.Tensor, ...]:
        encoded = self._rank0_encode(prompt, negative_prompt, guidance_scale)
        return tuple(broadcast_from_rank0(encoded, 4, device=self.device, bools=_MASKS))

    def _encode(self, prompt, negative_prompt, guidance_scale):
        if self.orchestrator is None:
            raise RuntimeError("the text encoder is not loaded on rank 0; call load() first")
        with self._pinned():
            return tuple(t.contiguous() for t in self.orchestrator.prepare_conditioning(
                prompt=prompt, negative_prompt=negative_prompt,
                guidance_scale=float(guidance_scale), max_sequence_length=self.seq_len,
            ))


class _DeviceLtx2TextEncoder(DeviceTextEncoder):
    """The LTX-2 text path with Gemma-3's language model on the device
    (``NeuronLtx2TextEncoderApplication``, TP-sharded beside the DiT), called like
    ``_HostLtx2TextEncoder``: ``(prompt, negative_prompt, guidance_scale)`` in, the four
    conditioning tensors out, identical on every rank.

    Every rank loads the tokenizer (``DeviceTextEncoder``, configured as diffusers'
    ``LTX2Pipeline._get_gemma_prompt_embeds`` uses it: left padding, the EOS token as the pad
    token when there is none) and tokenizes the stripped prompt, left-padded to ``seq_len``,
    truncated, special tokens added; then it moves the valid tokens to the front (the device
    encoder takes right-padded tokens, see its modeling module) and runs the encoder: its TP
    collectives need every rank. Rank 0 alone restores the left-padded layout of the 49 stacked
    states (the valid rows at the end, zeros before them: the connectors mask the padding out),
    runs diffusers' ``LTX2TextConnectors`` on its host (bf16, 1.4 B parameters, with ``threads``
    intra-op threads) on the ``[negative, positive]`` batch when ``guidance_scale`` > 1, and
    broadcasts the result as the host encoder does.
    """

    def __init__(self, app, *, model_path, seq_len: int, device, connectors_dtype: torch.dtype,
                 threads: int) -> None:
        super().__init__(app, model_path=model_path, device=device)
        self.seq_len = int(seq_len)
        self.connectors_dtype = connectors_dtype
        self.threads = int(threads)
        self.connectors = None  # rank 0 only, after load()

    def _configure_tokenizer(self, tokenizer) -> None:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def load(self, skip_warmup: bool = False) -> None:
        super().load(skip_warmup=skip_warmup)
        self.connectors = rank0_call(self._load_connectors, device=self.device,
                                     what="connectors load")

    def _load_connectors(self):
        from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

        return LTX2TextConnectors.from_pretrained(
            os.path.join(self.model_path, "connectors"), torch_dtype=self.connectors_dtype
        ).eval().requires_grad_(False)

    def _tokens(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """``(right-padded input_ids [1, L], diffusers' left-padded attention mask [1, L])``."""
        tokens = self.tokenizer(prompt.strip(), padding="max_length", max_length=self.seq_len,
                                truncation=True, add_special_tokens=True, return_tensors="pt")
        ids, mask = tokens.input_ids, tokens.attention_mask
        valid = ids[mask.bool()]
        right = torch.full_like(ids, int(self.tokenizer.pad_token_id))
        right[0, : valid.numel()] = valid
        return right.to(torch.int64), mask

    def _encode_one(self, prompt: str) -> tuple[torch.Tensor | None, torch.Tensor]:
        ids, mask = self._tokens(prompt)
        states = self.app(ids.to(self.device))
        if self.device.type == "neuron":
            torch.neuron.synchronize()
        if world_info()[0] != 0:
            return None, mask
        states = states.cpu()
        valid = int(mask.sum())
        left = torch.zeros_like(states)
        left[:, self.seq_len - valid:] = states[:, :valid]
        return left.flatten(2, 3), mask

    def __call__(self, prompt: str, negative_prompt: str | None,
                 guidance_scale: float) -> tuple[torch.Tensor, ...]:
        self._require_tokenizer()
        prompts = ([negative_prompt or "", prompt] if float(guidance_scale) > 1.0 else [prompt])
        encoded = [self._encode_one(p) for p in prompts]
        result = rank0_call(self._connect, encoded, device=self.device, what="connectors")
        return tuple(broadcast_from_rank0(result, 4, device=self.device, bools=_MASKS))

    def _connect(self, encoded):
        embeds = torch.cat([e for e, _ in encoded], dim=0).to(self.connectors_dtype)
        mask = torch.cat([m for _, m in encoded], dim=0)
        with intra_op_threads(self.threads), torch.no_grad():
            video, audio, connector_mask = self.connectors(embeds, mask, padding_side="left")
        connector_mask = connector_mask.to(torch.bool)
        return video.contiguous(), audio.contiguous(), connector_mask, connector_mask.clone()


class _OnDeviceLtx2Transformer(OnDeviceDiT):
    """The orchestrator's DiT callable (``OnDeviceDiT``): an ``LTX2DiTInputBundle`` in, the host
    ``(video, audio)`` velocity pair out.

    The five model tensors go to the device; the RoPE coordinates stay on the host as fp32 (the
    module computes the rotary tables there, see ``difflet/backends/neuron/ltx_2/transformer.py``);
    the masks and sigma are not sent (the DiT attends the text unmasked; sigma is the timestep).
    ``.dtype``, ``.config`` and ``cfg_parallel_enabled`` are what the orchestrator reads.
    """

    supports_ltx_2_extra_kwargs = False
    teacache_probe_fused = False

    def __init__(self, module, config: NeuronLtx2Config, dtype, device):
        super().__init__(module, config, dtype, device)
        self.cfg_parallel_enabled = bool(config.cfg_parallel_enabled)

    def __call__(self, bundle: LTX2DiTInputBundle, **kwargs: Any):
        if kwargs:
            raise NotImplementedError(
                f"the LTX-2 DiT takes no extra kwargs on the neuron backend: {sorted(kwargs)} "
                "(STG and modality guidance are not supported)"
            )
        if not isinstance(bundle, LTX2DiTInputBundle):
            raise TypeError(f"expected an LTX2DiTInputBundle, got {type(bundle).__name__}")
        _check_dit_shapes(self.config, bundle)
        coords = tuple(t.to(torch.float32).contiguous()
                       for t in (bundle.video_coords, bundle.audio_coords))
        return self._forward(
            (bundle.hidden_states, bundle.audio_hidden_states, bundle.encoder_hidden_states,
             bundle.audio_encoder_hidden_states, bundle.timestep),
            extra=coords,
        )


class _DeviceLtx2VaeDecoder(DeviceComponent):
    """The orchestrator's video VAE on the device (``NeuronLtx2VaeDecoderApplication``).

    Exposes what ``LTX2Orchestrator._decode_latents`` reads from a diffusers VAE: ``dtype``,
    ``config`` (``scaling_factor``, ``timestep_conditioning``), ``latents_mean`` /
    ``latents_std`` (read from the checkpoint, on the host) and ``decode(z, timestep,
    return_dict=False)``, which moves the un-normalised host latents to the device, decodes and
    returns the frames ``[B, 3, F, H, W]`` on the host as a one-tuple.
    """

    def __init__(self, app, *, device) -> None:
        from types import SimpleNamespace

        from safetensors import safe_open

        super().__init__(app, device=device)
        self.dtype = app.dtype
        self.config = SimpleNamespace(**app.vae_config)
        path = os.path.join(str(app.model_path), "diffusion_pytorch_model.safetensors")
        with safe_open(path, framework="pt", device="cpu") as handle:
            self.latents_mean = handle.get_tensor("latents_mean")
            self.latents_std = handle.get_tensor("latents_std")

    def decode(self, z: torch.Tensor, timestep=None, return_dict: bool = False):
        if timestep is not None:
            raise NotImplementedError("the LTX-2 device decoder takes no decode timestep")
        frames = self.app(z.to(self.dtype).contiguous().to(self.device))
        if self.device.type == "neuron":
            torch.neuron.synchronize()
        frames = frames.cpu()
        return (frames,) if not return_dict else {"sample": frames}


def _host_ltx2_audio_decoder(model_path: str, *, dtype: torch.dtype):
    """The audio VAE and the vocoder on the host CPU (diffusers' own modules, as the Trainium
    decode loads them) and the video post-processor, without the video VAE."""
    from diffusers import AutoencoderKLLTX2Audio
    from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
    from diffusers.video_processor import VideoProcessor

    audio_vae = AutoencoderKLLTX2Audio.from_pretrained(
        os.path.join(model_path, "audio_vae"), torch_dtype=dtype).eval()
    vocoder = LTX2Vocoder.from_pretrained(os.path.join(model_path, "vocoder"),
                                          torch_dtype=dtype).eval()
    return audio_vae, vocoder, VideoProcessor(vae_scale_factor=32)


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


class TorchNeuronLtx2Application(TorchNeuronModelApplication):
    """Host text path on rank 0 + the DiT on each rank's core + ``LTX2Orchestrator``, on every
    rank.

    Factory signature (``difflet/registry.py`` ``create_application``): ``model_path`` is the
    diffusers snapshot root, ``shape`` the request's ``height``/``width``/``num_frames``. Keyword
    arguments and their defaults are ``_KWARG_DEFAULTS``; ``device`` is ``"neuron"`` except in
    CPU tests (``"cpu"`` on gloo).
    """

    MODEL_LABEL = "LTX-2"
    KWARG_DEFAULTS = _KWARG_DEFAULTS
    TEACACHE_KWARGS = _TEACACHE_KWARGS
    DIT_BUNDLE = LTX2DiTInputBundle
    DIT_CALL = "a DiT bundle"

    def __init__(self, *, model_path, parallel, dtype: Any, shape: dict[str, int | None],
                 **kwargs: Any) -> None:
        super().__init__(model_path=model_path, parallel=parallel, dtype=dtype, shape=shape,
                         **kwargs)
        options = self.options
        self.frame_rate = float(options["frame_rate"])
        self.batch_size = 1
        transformer_path = os.path.join(self.model_path, "transformer")
        overrides: dict[str, Any] = dict(
            height=self.shape["height"], width=self.shape["width"],
            num_frames=self.shape["num_frames"], frame_rate=self.frame_rate,
            text_seq_len=self.text_seq_len, batch_size=1,
            tp_degree=int(parallel.tp_degree),
            cfg_parallel_enabled=bool(getattr(parallel, "cfg_parallel_enabled", False)),
            context_parallel_enabled=int(getattr(parallel, "cp_degree", 1) or 1) > 1,
            cp_mode=str(getattr(parallel, "cp_mode", "gather_kv") or "gather_kv"),
            torch_dtype=self.dtype,
        )
        if options["num_layers"] is not None:
            overrides["num_layers"] = int(options["num_layers"])
        self.config = NeuronLtx2Config.from_pretrained(transformer_path, **overrides)
        self.cfg_parallel_enabled = bool(self.config.cfg_parallel_enabled)
        if options["enable_transformer"]:
            self.transformer = NeuronLtx2TransformerApplication(
                model_path=transformer_path, config=self.config, parallel=parallel,
                dtype=self.dtype, exec_mode=options["exec_mode"], device=self.device,
            )
            self.config = self.transformer.config  # cp_degree filled from the parallel config
        enable_text = options["enable_text_encoder"]
        if options["enable_host_pipeline"] is not None:
            enable_text = bool(options["enable_host_pipeline"])
        if enable_text and options["text_encoder_device"] == "neuron":
            from difflet.backends.neuron.ltx_2.text_encoder import (
                NeuronLtx2TextEncoderApplication,
            )

            encoder_dtype = _normalize_dtype(options["text_encoder_dtype"])
            self._check_device_text_encoder_fits(encoder_dtype)
            self.text_encoder = _DeviceLtx2TextEncoder(
                NeuronLtx2TextEncoderApplication(
                    model_path=os.path.join(self.model_path, "text_encoder"),
                    seq_len=self.text_seq_len, parallel=parallel, dtype=encoder_dtype,
                    exec_mode=options["exec_mode"], device=self.device,
                ),
                model_path=self.model_path, seq_len=self.text_seq_len, device=self.device,
                connectors_dtype=encoder_dtype, threads=int(options["encoder_threads"]),
            )
        elif enable_text:
            self.text_encoder = _HostLtx2TextEncoder(
                self.model_path, seq_len=self.text_seq_len,
                dtype=_normalize_dtype(options["text_encoder_dtype"]), device=self.device,
                threads=int(options["encoder_threads"]), frame_rate=self.frame_rate,
                shape=self.shape,
            )
        self.enable_decode_components = bool(options["enable_decode_components"])
        self.decoder = None  # rank 0's host LTX2Pipeline decode components, after load()

    def _resolve_shape(self, shape: dict[str, int | None], shapes) -> dict[str, int]:
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
                raise ValueError(
                    f"the request shape {resolved} is not the one compiled shape {bucket}")
        return resolved

    def _check_device_text_encoder_fits(self, encoder_dtype: torch.dtype) -> None:
        """The device text encoder shares each core with this rank's DiT shard: refuse before
        anything loads when the two clearly cannot fit together
        (``difflet/backends/neuron/ltx_2/hbm.py``, ``_check_components_fit``): at tp4 about
        5.5 GiB of bf16 Gemma-3 beside 8.8 GiB of DiT; at tp2 (CFG2 on four cores, dp2tp2) twice
        both, over a 24 GiB core."""
        from difflet.backends.neuron.ltx_2.hbm import device_text_encoder_components

        self._check_components_fit(
            device_text_encoder_components(encoder_dtype.itemsize, self.dtype.itemsize),
            what=f"Gemma-3 ({encoder_dtype}) beside the LTX-2 DiT on one core",
            advice="encode on the host (text_encoder_device='host', no --device-text-encoder), "
            "or give the TP group more cores (tp_degree={tp} or more)",
        )

    # --------------------------------------------------------------------- load

    def _orchestrator_class(self) -> type:
        from difflet.models.ltx_2.pipeline import LTX2Orchestrator

        return LTX2Orchestrator

    def _dit_adapter(self, app) -> _OnDeviceLtx2Transformer:
        return _OnDeviceLtx2Transformer(app, self.config, self.dtype, self.device)

    def _load_decoders(self, skip_warmup: bool) -> None:
        """With ``enable_decode_components``, the decode path of diffusers' pipeline on rank
        0's host (the Trainium CLI's placement), in the application dtype."""
        del skip_warmup
        if self.enable_decode_components:
            self.decoder = rank0_call(
                _host_ltx2_pipeline, self.model_path, dtype=self.dtype, text=False, decode=True,
                device=self.device, what="decoder load",
            )

    def _build_pipeline(self, orchestrator_class: type):
        decoder = self.decoder
        pipeline = orchestrator_class(
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
        return pipeline

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

    def _ends_iteration(self, pipeline, args: tuple) -> bool:
        """The orchestrator steps the video scheduler, then the audio one, per iteration
        (``LTX2Orchestrator._scheduler_step(scheduler, ...)``): the audio step ends it."""
        return args[0] is not pipeline.scheduler

    # ----------------------------------------------------------------- evidence

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

    # ---------------------------------------------------------------- inference

    def encode_prompt(self, prompt, negative_prompt=None, guidance_scale: float = 1.0):
        """The four conditioning tensors (see ``_HostLtx2TextEncoder``), identical on every rank.
        Every rank must call it."""
        if self.text_encoder is None:
            raise RuntimeError(
                f"{type(self).__name__}: no text encoder (enable_text_encoder=False); pass "
                "encoder_hidden_states (and the audio stream and masks) instead of a prompt"
            )
        prompt = self._single_prompt(prompt)
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
        self._require_loaded()
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
        self._apply_seed(kwargs, seed)
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

        return self._draw_on_rank0(draw, count=2)


__all__ = ["TorchNeuronLtx2Application"]
