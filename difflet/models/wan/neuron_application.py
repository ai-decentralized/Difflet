"""Wan application for the neuron backend (TorchNeuron, one process per NeuronCore).

``TorchNeuronWanApplication`` keeps the outward contract of the Trainium ``NeuronWanApplication``
(``difflet/models/wan/application.py``, which already owns that name): the registry factory
signature, ``compile``/``has_compiled_artifacts``/``load``, ``.pipeline``, and a ``__call__`` that
sends a positional three-tensor call to the DiT and every other call -- the CLI stage's
``app(generator=..., prompt=..., ...)`` -- to the ``WanOrchestrator``. Every rank of the torchrun
launch builds one and holds:

* ``transformer``: ``NeuronWanTransformerApplication``, the DiT on this rank's core;
* ``transformer_2`` (``enable_transformer_2=True``, Wan 2.2 A14B): the low-noise expert, a second
  ``NeuronWanTransformerApplication`` on ``<model>/transformer_2`` with the same architecture.
  The orchestrator runs it for timesteps below ``boundary_ratio`` x 1000 (``model_index.json``)
  and ``transformer`` above. Both stay resident; the second loads after the first (runtime and
  mesh set-up are idempotent) and, its blocks having the first expert's graph, compiles no new
  NEFF;
* ``text_encoder``: ``_HostTextEncoder``, umT5 in fp32 on rank 0's host only, its embeddings
  broadcast to every rank (one fp32 copy of the 22.7 GB encoder instead of four); or, with
  ``text_encoder_device="neuron"``, ``_DeviceTextEncoder``: umT5 on the device in fp32, as on
  the host, sharded across the TP group with the AoT path's layout
  (``NeuronWanTextEncoderApplication``, about a quarter of the 22.7 GB fp32 encoder per rank),
  every rank encoding, the embeddings cast to the DiT's dtype;
* ``pipeline``: the backend-neutral ``WanOrchestrator`` (fp32 latents, UniPC on the host), which
  every rank runs on identical inputs; its DiT calls go through ``_OnDeviceTransformer``.

* ``vae_decoder`` (``enable_vae_decoder=True``): ``_DeviceVaeDecoder``, the VAE decoder on every
  rank's core in fp32, eager (``NeuronWanVaeDecoderApplication``); ``output_type="pt"`` then
  returns decoded frames. Without it the VAE decodes on the host outside this application
  (``--host-vae``).

Phase 1 limits, each rejected in ``__init__``: one static shape, batch 1, no TeaCache, tensor
parallelism only.

Timing hooks (for the benchmark): ``step_hook`` is called with no arguments once per denoise
iteration, right after the orchestrator's scheduler step (``load`` wraps
``WanOrchestrator._scheduler_step`` on this instance), never per DiT call: at guidance > 1 one
iteration is two DiT forwards plus the host UniPC step. ``transformer_adapter.forward_hook`` is
called after every DiT forward with that call's wall seconds (host -> device copies, forward,
device sync, copy back), whichever expert ran it; ``expert_calls`` counts the forwards per expert.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from collections.abc import Callable
from typing import Any

import torch

from difflet.backends.neuron.core.application_base import _normalize_dtype
from difflet.backends.neuron.core.checkpoint import _resolve_device
from difflet.backends.neuron.core.distributed import broadcast_tensor, rank0_call
from difflet.backends.neuron.runtime import check_parallel_supported
from difflet.backends.neuron.wan.config import NeuronWanConfig
from difflet.backends.neuron.wan.text_encoder import NeuronWanTextEncoderApplication
from difflet.backends.neuron.wan.transformer import NeuronWanTransformerApplication
from difflet.backends.neuron.core.weights_bf16 import resolve_component_dir
from difflet.backends.neuron.wan.vae import NeuronWanVaeDecoderApplication, default_vae_dtype

logger = logging.getLogger(__name__)

#: The keyword arguments the application accepts, with their defaults: the ones the Wan CLI
#: stage passes the Trainium application (difflet/cli/orchestrators/wan.py) plus exec_mode,
#: device and encoder_threads. Anything else is a TypeError rather than silently ignored.
_KWARG_DEFAULTS: dict[str, Any] = {
    "text_seq_len": 512,
    "batch_size": 1,
    "enable_text_encoder": True,
    "enable_transformer": True,
    "enable_transformer_2": False,
    "enable_vae_decoder": False,
    "exec_mode": None,
    "device": "neuron",
    "encoder_threads": 12,
    "text_encoder_device": "host",
    # umT5 in bf16, as the Trainium path runs it; the VAE's precision follows the clip length as on
    # the Trainium path (default_vae_dtype: bf16 up to 9 frames, fp32 beyond). Both overridable.
    "text_encoder_dtype": torch.bfloat16,
    "vae_dtype": None,
    "shapes": None,
    "teacache_cadence": None,
    "teacache_online_delta_alpha": None,
    "teacache_calibration_path": None,
}
_TEACACHE_KWARGS = ("teacache_cadence", "teacache_online_delta_alpha", "teacache_calibration_path")


class _HostTextEncoder:
    """umT5 in fp32 on rank 0's host; every rank receives the embeddings by broadcast.

    ``WanOrchestrator.encode_prompt`` tokenizes on every rank (cheap and deterministic) and
    calls this with ``(input_ids, attention_mask)`` padded on the right to ``seq_len``. Rank 0
    encodes inside ``rank0_call``, so a failing encode makes every rank raise together instead
    of leaving them in the broadcast; the result is broadcast in ``dtype`` and returned on the
    host. Only the valid prefix is encoded and the result re-padded with zeros: T5 attention is
    masked, so the padding never reaches the real tokens, and the orchestrator zeroes the padded
    rows anyway (``_zero_padding_embeds``). The encode runs with ``threads`` intra-op threads
    (the other ranks are waiting in the status all-reduce meanwhile) and restores the previous
    count after.

    ``load()`` reads the encoder on rank 0 inside its own status-synced phase, ``"text encoder
    load"``, so a rank 0 that cannot read it fails every rank there.
    """

    def __init__(self, model_path, *, text_dim: int, seq_len: int, dtype, device, threads: int,
                 encoder_dtype=torch.bfloat16):
        self.model_path = str(model_path)
        self.text_dim = int(text_dim)
        self.seq_len = int(seq_len)
        self.dtype = dtype
        self.encoder_dtype = encoder_dtype
        self.device = torch.device(device)
        self.threads = int(threads)
        self.model = None  # rank 0 only, after load()

    def load(self) -> None:
        self.model = rank0_call(self._load_model, device=self.device, what="text encoder load")

    def _load_model(self):
        from transformers import UMT5EncoderModel

        model = UMT5EncoderModel.from_pretrained(
            os.path.join(self.model_path, "text_encoder"), dtype=self.encoder_dtype
        )
        return model.eval().requires_grad_(False)

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embeds = rank0_call(
            self._encode, input_ids, attention_mask, device=self.device, what="prompt encode"
        )
        return broadcast_tensor(embeds, device=self.device).cpu()

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("the text encoder is not loaded on rank 0; call load() first")
        valid = max(int(attention_mask.sum(dim=-1).max()), 1)
        if attention_mask[:, valid:].any():
            raise ValueError("expected right-padded prompts (the tokenizer pads on the right)")
        previous = torch.get_num_threads()
        torch.set_num_threads(self.threads)
        try:
            with torch.no_grad():
                hidden = self.model(
                    input_ids[:, :valid].to(torch.int64),
                    attention_mask[:, :valid].to(torch.int32),
                ).last_hidden_state
        finally:
            torch.set_num_threads(previous)
        embeds = torch.zeros(input_ids.shape[0], self.seq_len, self.text_dim, dtype=self.dtype)
        embeds[:, :valid] = hidden.to(self.dtype)
        return embeds


class _DeviceTextEncoder:
    """umT5 on the device (``NeuronWanTextEncoderApplication``), called like ``_HostTextEncoder``.

    Every rank encodes (the encoder's TP collectives need all of them) and every rank gets the
    full embeddings, so nothing is broadcast. Each prompt row is encoded alone at the static
    ``(1, seq_len)`` shape, padding included and masked; the rows past the valid prefix are then
    zeroed, as the host path returns them (the orchestrator zeroes them as well).
    """

    def __init__(self, app: NeuronWanTextEncoderApplication, *, dtype, device):
        self.app = app
        self.dtype = dtype
        self.device = torch.device(device)

    def load(self, skip_warmup: bool = False) -> None:
        self.app.load(skip_warmup=skip_warmup)

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        rows = []
        for ids, mask in zip(input_ids, attention_mask):
            valid = max(int(mask.sum()), 1)
            if mask[valid:].any():
                raise ValueError("expected right-padded prompts (the tokenizer pads on the right)")
            hidden = self.app(
                ids[None].to(self.device, torch.int64), mask[None].to(self.device, torch.int32)
            )
            row = hidden.to("cpu", self.dtype)
            row[:, valid:] = 0
            rows.append(row)
        return torch.cat(rows)


class _DeviceVaeDecoder:
    """The orchestrator's VAE decoder: host latents in (already un-normalised), host frames out.

    Exposes what ``WanOrchestrator._decode_latents`` reads: ``dtype`` (fp32, the decode
    precision) and ``config`` (``latents_mean``/``latents_std``).
    """

    def __init__(self, app: NeuronWanVaeDecoderApplication, *, device):
        self.app = app
        self.dtype = app.dtype
        self.config = app.vae_config
        self.device = torch.device(device)

    def load(self, skip_warmup: bool = False) -> None:
        self.app.load(skip_warmup=skip_warmup)

    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        return self.app(latents.to(self.device, self.dtype)).cpu()


class _OnDeviceTransformer:
    """The orchestrator's DiT callable: host tensors in, host tensor out.

    Each input is cast to ``dtype`` and made contiguous on the host before it moves to
    ``device``: a strided host view (the timestep ``_batch_timestep`` expands from a 0-dim
    tensor, a transposed slice) would otherwise be restrided on the device through a CPU round
    trip (decision D16). ``.dtype`` and ``.config`` are what the orchestrator reads
    (``_component_dtype``; ``cfg_parallel_enabled``). The latents stay fp32 on the host between
    steps, as on TPU, because UniPC's corrector collapses in bf16.
    """

    def __init__(self, module: Callable[..., torch.Tensor], config, dtype, device, hook_from=None):
        self.module = module
        self.config = config
        self.dtype = dtype
        self.device = torch.device(device)
        #: Called after every forward with its wall seconds; see the module docstring.
        self.forward_hook: Callable[[float], None] | None = None
        #: Another adapter whose ``forward_hook`` this one reports to (the low-noise expert's
        #: adapter reports to the high-noise one's, so one hook times every DiT forward).
        self.hook_from = hook_from
        self.calls = 0

    def __call__(self, hidden_states, timestep, encoder_hidden_states) -> torch.Tensor:
        start = time.perf_counter()
        args = tuple(
            tensor.to(self.dtype).contiguous().to(self.device)
            for tensor in (hidden_states, timestep, encoder_hidden_states)
        )
        out = self.module(*args)
        if self.device.type == "neuron":
            torch.neuron.synchronize()
        out = out.cpu()
        self.calls += 1
        hook = self.forward_hook if self.hook_from is None else self.hook_from.forward_hook
        if hook is not None:
            hook(time.perf_counter() - start)
        return out


def _resolve_shape(shape: dict[str, int | None], shapes) -> dict[str, int]:
    if shapes:
        shapes = list(shapes)
        if len(shapes) > 1:
            raise NotImplementedError(
                "the neuron backend compiles one static shape per process (phase 1); "
                f"got shapes={shapes}"
            )
        height, width, num_frames = (int(v) for v in shapes[0])
        return {"height": height, "width": width, "num_frames": num_frames}
    return {
        "height": int(shape.get("height") or 480),
        "width": int(shape.get("width") or 832),
        "num_frames": int(shape.get("num_frames") or 9),
    }


def _reject_unsupported(options: dict[str, Any]) -> None:
    if int(options["batch_size"]) != 1:
        raise NotImplementedError(
            f"the neuron Wan application runs batch_size=1 only (phase 1), got "
            f"batch_size={options['batch_size']}"
        )
    if options["text_encoder_device"] not in ("host", "neuron"):
        raise ValueError(
            f"text_encoder_device must be 'host' or 'neuron', got {options['text_encoder_device']!r}"
        )


class TorchNeuronWanApplication(torch.nn.Module):
    """Host umT5 on rank 0 + the DiT on each rank's core + ``WanOrchestrator``, on every rank.

    Factory signature (``difflet/registry.py`` ``create_application``): ``model_path`` is the
    diffusers snapshot root, ``shape`` the request's ``height``/``width``/``num_frames``. Keyword
    arguments and their defaults are ``_KWARG_DEFAULTS``; ``device`` is ``"neuron"`` except in
    CPU tests (``"cpu"`` on gloo).
    """

    def __init__(
        self,
        *,
        model_path,
        parallel,
        dtype: Any,
        shape: dict[str, int | None],
        **kwargs: Any,
    ) -> None:
        super().__init__()
        unknown = sorted(set(kwargs) - set(_KWARG_DEFAULTS))
        if unknown:
            raise TypeError(
                f"{type(self).__name__} got unexpected keyword arguments: {', '.join(unknown)}"
            )
        options = {**_KWARG_DEFAULTS, **kwargs}
        _reject_unsupported(options)
        self.options = options  # read again at load (TeaCache)
        check_parallel_supported(parallel)
        self.model_path = str(model_path)
        self.parallel = parallel
        self.dtype = _normalize_dtype(dtype)
        self.device = _resolve_device(options["device"])
        self.shape = _resolve_shape(shape, options["shapes"])
        self.text_seq_len = int(options["text_seq_len"])
        transformer_path = os.path.join(self.model_path, "transformer")
        config_overrides = dict(
            **self.shape,
            batch_size=1,
            text_seq_len=self.text_seq_len,
            tp_degree=int(parallel.tp_degree),
            cfg_parallel_enabled=bool(getattr(parallel, "cfg_parallel_enabled", False)),
            context_parallel_enabled=int(getattr(parallel, "cp_degree", 1) or 1) > 1,
            sp_enabled=bool(getattr(parallel, "sp_enabled", False)),
            cp_mode=str(getattr(parallel, "cp_mode", "gather_kv")),
            torch_dtype=self.dtype,
        )
        self.config = NeuronWanConfig.from_pretrained(transformer_path, **config_overrides)
        self.transformer: NeuronWanTransformerApplication | None = None
        if options["enable_transformer"]:
            self.transformer = NeuronWanTransformerApplication(
                model_path=resolve_component_dir(transformer_path, self.dtype),  # the bf16 copy if made
                config=self.config,
                parallel=parallel,
                dtype=self.dtype,
                exec_mode=options["exec_mode"],
                device=self.device,
            )
        self.transformer_2: NeuronWanTransformerApplication | None = None
        if options["enable_transformer_2"]:
            transformer_2_path = self._low_noise_expert_path(options)
            config_2 = NeuronWanConfig.from_pretrained(transformer_2_path, **config_overrides)
            _require_same_architecture(self.config, config_2)
            self.transformer_2 = NeuronWanTransformerApplication(
                model_path=resolve_component_dir(transformer_2_path, self.dtype),
                config=config_2,
                parallel=parallel,
                dtype=self.dtype,
                exec_mode=options["exec_mode"],
                device=self.device,
            )
        self.text_encoder: _HostTextEncoder | _DeviceTextEncoder | None = None
        text_encoder_dtype = _normalize_dtype(options["text_encoder_dtype"])
        if (options["enable_text_encoder"] and options["text_encoder_device"] == "neuron"
                and text_encoder_dtype == torch.float32
                and (self.config.cfg_parallel_enabled or self.config.context_parallel_enabled)):
            raise ValueError(
                "CFG and context parallelism narrow the TP degree, so a core would hold a TP-"
                f"{parallel.tp_degree} shard of the fp32 text encoder beside a TP-"
                f"{parallel.tp_degree} shard of the DiT, more than its HBM; encode on the host "
                "(text_encoder_device='host', --host-text-encoder)"
            )
        if options["enable_text_encoder"] and options["text_encoder_device"] == "neuron":
            self.text_encoder = _DeviceTextEncoder(
                NeuronWanTextEncoderApplication(
                    model_path=os.path.join(self.model_path, "text_encoder"),
                    text_seq_len=self.text_seq_len,
                    batch_size=1,
                    parallel=parallel,
                    dtype=text_encoder_dtype,
                    exec_mode=options["exec_mode"],
                    device=self.device,
                ),
                dtype=self.dtype,
                device=self.device,
            )
        elif options["enable_text_encoder"]:
            self.text_encoder = _HostTextEncoder(
                self.model_path,
                text_dim=int(self.config.text_dim),
                seq_len=self.text_seq_len,
                dtype=self.dtype,
                device=self.device,
                threads=int(options["encoder_threads"]),
                encoder_dtype=text_encoder_dtype,
            )
        self.vae_decoder: _DeviceVaeDecoder | None = None
        if options["enable_vae_decoder"]:
            cfg = self.config
            self.vae_decoder = _DeviceVaeDecoder(
                NeuronWanVaeDecoderApplication(
                    model_path=os.path.join(self.model_path, "vae"),
                    latent_shape=(1, int(cfg.out_channels), cfg.latent_frames,
                                  cfg.latent_height, cfg.latent_width),
                    parallel=parallel,
                    dtype=(_normalize_dtype(options["vae_dtype"]) if options["vae_dtype"] is not None
                           else default_vae_dtype((int(cfg.latent_frames) - 1) * 4 + 1)),
                    device=self.device,
                ),
                device=self.device,
            )
        self.transformer_adapter: _OnDeviceTransformer | None = None
        self.transformer_2_adapter: _OnDeviceTransformer | None = None
        self.pipeline = None
        #: Called with no arguments once per denoise iteration; see the module docstring.
        self.step_hook: Callable[[], None] | None = None
        self._text_encoder_load_seconds: float | None = None

    def _low_noise_expert_path(self, options: dict[str, Any]) -> str:
        if not options["enable_transformer"]:
            raise ValueError(
                "enable_transformer_2 requires enable_transformer: the low-noise expert runs "
                "below the boundary timestep and the high-noise expert above it"
            )
        path = os.path.join(self.model_path, "transformer_2")
        if not os.path.isfile(os.path.join(path, "config.json")):
            raise ValueError(
                f"enable_transformer_2: {path} has no config.json; only two-expert checkpoints "
                "(Wan 2.2 A14B) have a low-noise expert (transformer_2)"
            )
        return path

    # --------------------------------------------------- DiffletPipeline API

    def compile(self, compiled_model_path, debug: bool = False) -> None:
        """No-op: the neuron backend compiles nothing ahead of time (see ``load``)."""
        del compiled_model_path, debug

    def has_compiled_artifacts(self, compiled_model_path) -> bool:
        del compiled_model_path
        return True

    def load(
        self,
        compiled_model_path=None,
        start_rank_id=None,
        local_ranks_size=None,
        skip_warmup: bool = False,
    ) -> None:
        """DiT (runtime init, shards, per-block compile, warm-up), then the low-noise expert if
        enabled, then umT5 on rank 0, then the orchestrator. Every rank must call it; the steps
        are status-synced phases."""
        del compiled_model_path, start_rank_id, local_ranks_size
        if self.pipeline is not None:
            return
        from difflet.models.wan.pipeline import WanOrchestrator

        if self.transformer is not None:
            self.transformer.load(skip_warmup=skip_warmup)
            self.transformer_adapter = _OnDeviceTransformer(
                self.transformer, self.config, self.dtype, self.device
            )
        if self.transformer_2 is not None:
            self.transformer_2.load(skip_warmup=skip_warmup)
            self.transformer_2_adapter = _OnDeviceTransformer(
                self.transformer_2, self.transformer_2.config, self.dtype, self.device,
                hook_from=self.transformer_adapter,
            )
        if isinstance(self.text_encoder, _DeviceTextEncoder):
            self.text_encoder.load(skip_warmup=skip_warmup)
        elif self.text_encoder is not None:
            start = time.perf_counter()
            self.text_encoder.load()
            self._text_encoder_load_seconds = time.perf_counter() - start
        if self.vae_decoder is not None:
            self.vae_decoder.load(skip_warmup=skip_warmup)
        pipeline = WanOrchestrator(
            model_path=self.model_path,
            text_encoder=self.text_encoder,
            transformer=self.transformer_adapter,
            transformer_2=self.transformer_2_adapter,
            vae_decoder=self.vae_decoder,
            dtype=self.dtype,
            max_text_length=self.text_seq_len,
            teacache_calibration_path=self.options["teacache_calibration_path"],
            teacache_probes=self._teacache_probes(),
            teacache_cadence=self.options["teacache_cadence"],
            teacache_online_delta_alpha=self.options["teacache_online_delta_alpha"],
            **self.shape,
        )
        self._stamp_denoise_iterations(pipeline)
        self.pipeline = pipeline
        logger.info("%s loaded: %s", type(self).__name__,
                    {k: round(v, 2) for k, v in self.phase_seconds.items()})

    def _teacache_probes(self) -> dict[int, Any]:
        """One TeaCache probe per loaded expert, keyed as WanOrchestrator looks them up (the id of
        the transformer callable it was given), when the calibration needs a per-step signal; so
        the Trainium CPU shadow is never built. Empty for the probe-free modes and without TeaCache."""
        if not self.options["teacache_calibration_path"]:
            return {}
        from difflet.backends.neuron.core.teacache import NeuronTeaCacheProbe

        probes = {}
        for adapter, app in ((getattr(self, "transformer_adapter", None), getattr(self, "transformer", None)),
                             (getattr(self, "transformer_2_adapter", None), getattr(self, "transformer_2", None))):
            if adapter is not None and app is not None:
                probes[id(adapter)] = NeuronTeaCacheProbe(app.teacache_mod_input)
        return probes

    def _stamp_denoise_iterations(self, pipeline) -> None:
        """Call ``step_hook`` after each scheduler step: exactly once per denoise iteration, in
        every branch of ``WanOrchestrator._denoise`` (CFG, cfg-parallel, a TeaCache skip)."""
        scheduler_step = pipeline._scheduler_step

        def stamped_scheduler_step(*args: Any, **kwargs: Any) -> torch.Tensor:
            latents = scheduler_step(*args, **kwargs)
            if self.step_hook is not None:
                self.step_hook()
            return latents

        pipeline._scheduler_step = stamped_scheduler_step

    # ------------------------------------------------------------- evidence

    @property
    def phase_seconds(self) -> dict[str, float]:
        """The DiT's load phases, the low-noise expert's as ``"transformer_2 <phase>"``, the
        device umT5's as ``"text_encoder <phase>"``, or for the host umT5 ``"text encoder load"``
        (rank 0's read, which the other ranks wait out in that phase's status sync)."""
        seconds = dict(self.transformer.phase_seconds) if self.transformer is not None else {}
        if self.transformer_2 is not None:
            seconds.update(
                {f"transformer_2 {k}": v for k, v in self.transformer_2.phase_seconds.items()}
            )
        if isinstance(self.text_encoder, _DeviceTextEncoder):
            seconds.update(
                {f"text_encoder {k}": v for k, v in self.text_encoder.app.phase_seconds.items()}
            )
        if self.vae_decoder is not None:
            seconds.update({f"vae {k}": v for k, v in self.vae_decoder.app.phase_seconds.items()})
        if self._text_encoder_load_seconds is not None:
            seconds["text encoder load"] = self._text_encoder_load_seconds
        return seconds

    @property
    def compiled_blocks(self) -> list[str]:
        blocks = list(self.transformer.compiled_blocks) if self.transformer is not None else []
        if self.transformer_2 is not None:
            blocks += [f"transformer_2.{name}" for name in self.transformer_2.compiled_blocks]
        if isinstance(self.text_encoder, _DeviceTextEncoder):
            blocks += [f"text_encoder.{name}" for name in self.text_encoder.app.compiled_blocks]
        return blocks

    @property
    def unwarmed_shapes(self) -> list:
        shapes = list(self.transformer.unwarmed_shapes) if self.transformer is not None else []
        if self.transformer_2 is not None:
            shapes += list(self.transformer_2.unwarmed_shapes)
        if isinstance(self.text_encoder, _DeviceTextEncoder):
            shapes += list(self.text_encoder.app.unwarmed_shapes)
        if self.vae_decoder is not None:
            shapes += list(self.vae_decoder.app.unwarmed_shapes)
        return shapes

    @property
    def expert_calls(self) -> dict[str, int]:
        """DiT forwards per expert since load: which expert the boundary actually selected."""
        calls = {}
        if self.transformer_adapter is not None:
            calls["transformer"] = self.transformer_adapter.calls
        if self.transformer_2_adapter is not None:
            calls["transformer_2"] = self.transformer_2_adapter.calls
        return calls

    def dit_input_contract(self) -> dict[str, dict[str, Any]]:
        cfg = self.config
        batch = int(cfg.batch_size)
        return {
            "hidden_states": {
                "shape": (
                    batch,
                    int(cfg.in_channels),
                    cfg.latent_frames,
                    cfg.latent_height,
                    cfg.latent_width,
                ),
                "dtype": self.dtype,
            },
            "timestep": {"shape": (batch,), "dtype": self.dtype},
            "encoder_hidden_states": {
                "shape": (batch, int(cfg.text_seq_len), int(cfg.text_dim)),
                "dtype": self.dtype,
            },
        }

    def __call__(self, *args: Any, **kwargs: Any):
        """``app(latents, timestep, text)`` runs the DiT (the high-noise expert when there are
        two); anything else runs the orchestrator."""
        if len(args) >= 3:
            if self.transformer_adapter is None:
                raise RuntimeError(
                    f"{type(self).__name__}: the DiT is not loaded (call load() first; "
                    "enable_transformer=False never loads one)"
                )
            return self.transformer_adapter(*args, **kwargs)
        if self.pipeline is None:
            raise RuntimeError(f"{type(self).__name__} is not loaded; call load() first")
        return self.pipeline(*args, **kwargs)


def _require_same_architecture(config, config_2) -> None:
    """The two experts must share one architecture: one block graph, one input contract."""
    first, second = dataclasses.asdict(config), dataclasses.asdict(config_2)
    differ = sorted(name for name in first if first[name] != second.get(name))
    if differ:
        raise ValueError(
            "transformer and transformer_2 configs differ in "
            + ", ".join(f"{name} ({first[name]!r} vs {second.get(name)!r})" for name in differ)
        )


__all__ = ["TorchNeuronWanApplication"]
