"""Wan application for the neuron backend (TorchNeuron, one process per NeuronCore).

``TorchNeuronWanApplication`` keeps the outward contract of the Trainium ``NeuronWanApplication``
(``difflet/models/wan/application.py``, which already owns that name): the registry factory
signature, ``compile``/``has_compiled_artifacts``/``load``, ``.pipeline``, and a ``__call__`` that
sends a positional three-tensor call to the DiT and every other call -- the CLI stage's
``app(generator=..., prompt=..., ...)`` -- to the ``WanOrchestrator``. Every rank of the torchrun
launch builds one and holds:

* ``transformer``: ``NeuronWanTransformerApplication``, the DiT on this rank's core;
* ``text_encoder``: ``_HostTextEncoder``, umT5 in fp32 on rank 0's host only, its embeddings
  broadcast to every rank (one fp32 copy of the 22.7 GB encoder instead of four);
* ``pipeline``: the backend-neutral ``WanOrchestrator`` (fp32 latents, UniPC on the host), which
  every rank runs on identical inputs; its DiT calls go through ``_OnDeviceTransformer``.

The VAE decodes on the host outside this application (``--host-vae``). Phase 1 limits, each
rejected in ``__init__``: one expert (no ``transformer_2``), no device VAE, one static shape,
batch 1, no TeaCache, tensor parallelism only.

Timing hooks (for the benchmark): ``step_hook`` is called with no arguments once per denoise
iteration, right after the orchestrator's scheduler step (``load`` wraps
``WanOrchestrator._scheduler_step`` on this instance), never per DiT call: at guidance > 1 one
iteration is two DiT forwards plus the host UniPC step. ``transformer_adapter.forward_hook`` is
called after every DiT forward with that call's wall seconds (host -> device copies, forward,
device sync, copy back).
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
from difflet.backends.neuron.core.distributed import broadcast_tensor, rank0_call
from difflet.backends.neuron.runtime import check_parallel_supported
from difflet.backends.neuron.wan.config import NeuronWanConfig
from difflet.backends.neuron.wan.transformer import NeuronWanTransformerApplication

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

    def __init__(self, model_path, *, text_dim: int, seq_len: int, dtype, device, threads: int):
        self.model_path = str(model_path)
        self.text_dim = int(text_dim)
        self.seq_len = int(seq_len)
        self.dtype = dtype
        self.device = torch.device(device)
        self.threads = int(threads)
        self.model = None  # rank 0 only, after load()

    def load(self) -> None:
        self.model = rank0_call(self._load_model, device=self.device, what="text encoder load")

    def _load_model(self):
        from transformers import UMT5EncoderModel

        model = UMT5EncoderModel.from_pretrained(
            os.path.join(self.model_path, "text_encoder"), dtype=torch.float32
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


class _OnDeviceTransformer:
    """The orchestrator's DiT callable: host tensors in, host tensor out.

    Each input is cast to ``dtype`` and made contiguous on the host before it moves to
    ``device``: a strided host view (the timestep ``_batch_timestep`` expands from a 0-dim
    tensor, a transposed slice) would otherwise be restrided on the device through a CPU round
    trip (decision D16). ``.dtype`` and ``.config`` are what the orchestrator reads
    (``_component_dtype``; ``cfg_parallel_enabled``). The latents stay fp32 on the host between
    steps, as on TPU, because UniPC's corrector collapses in bf16.
    """

    def __init__(self, module: Callable[..., torch.Tensor], config, dtype, device):
        self.module = module
        self.config = config
        self.dtype = dtype
        self.device = torch.device(device)
        #: Called after every forward with its wall seconds; see the module docstring.
        self.forward_hook: Callable[[float], None] | None = None

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
        if self.forward_hook is not None:
            self.forward_hook(time.perf_counter() - start)
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
    if options["enable_transformer_2"]:
        raise NotImplementedError(
            "the neuron Wan application holds one expert; transformer_2 (Wan2.2 A14B's "
            "low-noise expert) is not supported in phase 1"
        )
    if options["enable_vae_decoder"]:
        raise NotImplementedError(
            "the neuron backend decodes the VAE on the host (--host-vae); there is no device "
            "VAE in phase 1"
        )
    teacache = [name for name in _TEACACHE_KWARGS if options[name] is not None]
    if teacache:
        raise NotImplementedError(
            f"TeaCache is not supported on the neuron backend in phase 1 ({', '.join(teacache)})"
        )
    if int(options["batch_size"]) != 1:
        raise NotImplementedError(
            f"the neuron Wan application runs batch_size=1 only (phase 1), got "
            f"batch_size={options['batch_size']}"
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
        check_parallel_supported(parallel)
        self.model_path = str(model_path)
        self.parallel = parallel
        self.dtype = _normalize_dtype(dtype)
        self.device = _resolve_device(options["device"])
        self.shape = _resolve_shape(shape, options["shapes"])
        self.text_seq_len = int(options["text_seq_len"])
        transformer_path = os.path.join(self.model_path, "transformer")
        self.config = NeuronWanConfig.from_pretrained(
            transformer_path,
            **self.shape,
            batch_size=1,
            text_seq_len=self.text_seq_len,
            tp_degree=int(parallel.tp_degree),
            torch_dtype=self.dtype,
        )
        self.transformer: NeuronWanTransformerApplication | None = None
        if options["enable_transformer"]:
            self.transformer = NeuronWanTransformerApplication(
                model_path=transformer_path,
                config=self.config,
                parallel=parallel,
                dtype=self.dtype,
                exec_mode=options["exec_mode"],
                device=self.device,
            )
        self.text_encoder: _HostTextEncoder | None = None
        if options["enable_text_encoder"]:
            self.text_encoder = _HostTextEncoder(
                self.model_path,
                text_dim=int(self.config.text_dim),
                seq_len=self.text_seq_len,
                dtype=self.dtype,
                device=self.device,
                threads=int(options["encoder_threads"]),
            )
        self.transformer_adapter: _OnDeviceTransformer | None = None
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

    def load(
        self,
        compiled_model_path=None,
        start_rank_id=None,
        local_ranks_size=None,
        skip_warmup: bool = False,
    ) -> None:
        """DiT (runtime init, shards, per-block compile, warm-up), then umT5 on rank 0, then
        the orchestrator. Every rank must call it; the steps are status-synced phases."""
        del compiled_model_path, start_rank_id, local_ranks_size
        if self.pipeline is not None:
            return
        from difflet.models.wan.pipeline import WanOrchestrator

        if self.transformer is not None:
            self.transformer.load(skip_warmup=skip_warmup)
            self.transformer_adapter = _OnDeviceTransformer(
                self.transformer, self.config, self.dtype, self.device
            )
        if self.text_encoder is not None:
            start = time.perf_counter()
            self.text_encoder.load()
            self._text_encoder_load_seconds = time.perf_counter() - start
        pipeline = WanOrchestrator(
            model_path=self.model_path,
            text_encoder=self.text_encoder,
            transformer=self.transformer_adapter,
            transformer_2=None,
            vae_decoder=None,
            dtype=self.dtype,
            max_text_length=self.text_seq_len,
            **self.shape,
        )
        self._stamp_denoise_iterations(pipeline)
        self.pipeline = pipeline
        logger.info("%s loaded: %s", type(self).__name__,
                    {k: round(v, 2) for k, v in self.phase_seconds.items()})

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
        """The DiT's load phases plus ``"text encoder load"`` (rank 0's umT5 read, which the
        other ranks wait out in that phase's status sync)."""
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
        """``app(latents, timestep, text)`` runs the DiT; anything else runs the orchestrator."""
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


__all__ = ["TorchNeuronWanApplication"]
