"""The composed model application of the neuron backend, shared by Wan, Qwen-Image and LTX-2.

Each model's ``TorchNeuron<Model>Application`` (``difflet/models/<model>/neuron_application.py``)
is the registry's application on every rank of a torchrun launch: the DiT on this rank's core
(``TorchNeuronApplicationBase`` subclasses, ``core/application_base.py``), a text encoder on rank
0's host or on the device, optionally a device VAE, and the backend-neutral orchestrator, which
every rank runs on identical inputs. Their host-side plumbing is the same, and lives here:

* ``TorchNeuronModelApplication``: the keyword-argument contract (``KWARG_DEFAULTS``; anything
  else is a TypeError) and the phase-1 refusals (TeaCache, batch > 1, an unknown text encoder
  placement), the load order (DiT(s), then the text encoder, then the decoders, then the
  orchestrator; every step a status-synced phase), the evidence the CLI stage and the
  benchmarks read (``phase_seconds``, ``compiled_blocks``, ``unwarmed_shapes``), the timing
  hooks (``step_hook`` once per denoise iteration, ``transformer_adapter.forward_hook`` once per
  DiT forward), the HBM check of a device text encoder beside the DiT, ``seed=`` and the initial
  latents drawn on rank 0 and broadcast, and the ``__call__`` dispatch between a DiT call and
  ``generate``;
* ``OnDeviceDiT``: the orchestrator's DiT callable, host tensors in and host tensors out, each
  cast and made contiguous on the host before it moves (decision D16), timed and counted;
* ``HostTextEncoder``: an encoder on rank 0's host, loaded in its own status-synced phase
  (``"text encoder load"``), encoding with a pinned intra-op thread count inside ``rank0_call``
  (a failing encode fails every rank instead of leaving them in the broadcast), its output
  broadcast to every rank;
* ``DeviceComponent`` / ``DeviceTextEncoder``: the adapters around a component application on
  the device, the latter loading the tokenizer on every rank first.

What each model keeps in its own file: its components and their dtypes, its orchestrator and
how to build it, its input contract and the static-shape checks, the encode semantics, and the
reasons for each of those choices.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, ClassVar

import torch

from difflet.backends.neuron.core.application_base import (
    TorchNeuronApplicationBase,
    _normalize_dtype,
)
from difflet.backends.neuron.core.checkpoint import _resolve_device
from difflet.backends.neuron.core.distributed import (
    broadcast_tensor,
    collective_phase,
    rank0_call,
    world_info,
)
from difflet.backends.neuron.hbm import Component, budget_enforced, check_fits, components_bytes
from difflet.backends.neuron.runtime import check_parallel_supported

logger = logging.getLogger(__name__)

__all__ = [
    "DeviceComponent",
    "DeviceTextEncoder",
    "HostTextEncoder",
    "OnDeviceDiT",
    "TorchNeuronModelApplication",
    "broadcast_from_rank0",
    "intra_op_threads",
]


@contextmanager
def intra_op_threads(threads: int) -> Iterator[None]:
    """Run the block with ``threads`` intra-op threads and restore the previous count after.

    The host encoders pin it: the fp32 umT5's output bits depend on the count (D70), and the
    other ranks wait in a status all-reduce meanwhile, so the encode may take the host."""
    previous = torch.get_num_threads()
    torch.set_num_threads(int(threads))
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def broadcast_from_rank0(tensors: Sequence[torch.Tensor] | None, count: int, *, device,
                         bools: Sequence[int] = ()) -> list[torch.Tensor]:
    """Rank 0's ``count`` tensors (None on the other ranks) on every rank, on the host.

    ``broadcast_tensor`` carries no bool, so a bool tensor travels as int32; the indices in
    ``bools`` are turned back into bool on every rank (the other ranks cannot know the dtype)."""
    received = []
    for i in range(count):
        tensor = None if tensors is None else tensors[i]
        if tensor is not None and tensor.dtype == torch.bool:
            tensor = tensor.to(torch.int32)
        tensor = broadcast_tensor(tensor, device=device).cpu()
        received.append(tensor.to(torch.bool) if i in bools else tensor)
    return received


class OnDeviceDiT:
    """The orchestrator's DiT callable around the DiT application: host tensors in, host tensors
    out.

    ``_forward`` casts each model input to ``dtype`` and makes it contiguous on the host before
    it moves to ``device`` (``_device_input``): a strided host view (the timestep
    ``_batch_timestep`` expands from a 0-dim tensor, a transposed slice) would otherwise be
    restrided on the device through a CPU round trip (decision D16). It synchronises the device, copies the output back, counts the
    call and reports its wall seconds (host -> device copies, forward, sync, copy back) to
    ``forward_hook``. ``.dtype`` and ``.config`` are what the orchestrators read. The models'
    subclasses take their own call signature and check their one static shape first.
    """

    def __init__(self, module: Callable[..., Any], config, dtype, device) -> None:
        self.module = module
        self.config = config
        self.dtype = dtype
        self.device = torch.device(device)
        #: Called after every forward with its wall seconds.
        self.forward_hook: Callable[[float], None] | None = None
        self.calls = 0

    def _forward(self, tensors: Sequence[torch.Tensor], extra: Sequence[Any] = ()) -> Any:
        """``module(*tensors on the device, *extra)`` with its output on the host: a tensor, or
        a tuple of them."""
        start = time.perf_counter()
        args = tuple(self._device_input(tensor) for tensor in tensors)
        out = self.module(*args, *extra)
        if self.device.type == "neuron":
            torch.neuron.synchronize()
        out = out.cpu() if isinstance(out, torch.Tensor) else tuple(o.cpu() for o in out)
        self.calls += 1
        hook = self._forward_hook()
        if hook is not None:
            hook(time.perf_counter() - start)
        return out

    def _device_input(self, tensor: torch.Tensor) -> torch.Tensor:
        """One model input on the device: cast to ``dtype`` and made contiguous on the host
        first. A model whose module takes an integer tensor (a mask) keeps it by overriding
        this."""
        return tensor.to(self.dtype).contiguous().to(self.device)

    def _forward_hook(self) -> Callable[[float], None] | None:
        return self.forward_hook


class HostTextEncoder:
    """A text encoder on rank 0's host only; every rank receives its output by broadcast.

    ``load()`` runs ``_load`` on rank 0 inside its own status-synced phase (``"text encoder
    load"``), so a rank 0 that cannot read the encoder fails every rank there, then, when
    ``warmup_args`` is set, one discarded ``_encode(*warmup_args)``: a fresh process's first
    bf16 encode is not reproducible when the host CPUs are busy (it differed in 4 of 51 fresh
    processes on a trn2.3xlarge, no later encode did), so the request is never a process's
    first. ``_rank0_encode`` runs ``_encode`` on rank 0 inside ``rank0_call`` (a failing encode
    makes every rank raise together instead of leaving them in the broadcast); the subclass
    broadcasts the result (``broadcast_from_rank0``). ``_encode`` runs under ``_pinned()``.
    """

    #: The discarded encode ``load`` spends on rank 0, or None for none.
    warmup_args: ClassVar[tuple | None] = None

    def __init__(self, model_path, *, device, threads: int) -> None:
        self.model_path = str(model_path)
        self.device = torch.device(device)
        self.threads = int(threads)

    def load(self) -> None:
        rank0_call(self._load_on_rank0, device=self.device, what="text encoder load")

    def _load_on_rank0(self) -> None:
        self._load()
        if self.warmup_args is not None:
            self._encode(*self.warmup_args)

    def _load(self) -> None:
        """Read the encoder on rank 0 and keep it on this object."""
        raise NotImplementedError

    def _encode(self, *args: Any) -> Any:
        raise NotImplementedError

    def _rank0_encode(self, *args: Any) -> Any:
        """``_encode(*args)`` on rank 0 (None elsewhere), every rank status-synced."""
        return rank0_call(self._encode, *args, device=self.device, what="prompt encode")

    @contextmanager
    def _pinned(self) -> Iterator[None]:
        """``threads`` intra-op threads and no autograd, for the encode."""
        with intra_op_threads(self.threads), torch.no_grad():
            yield


class DeviceComponent:
    """The orchestrator-facing adapter around one component application on this rank's core
    (``app``, a ``TorchNeuronApplicationBase``); its load is the application's."""

    def __init__(self, app, *, device) -> None:
        self.app = app
        self.device = torch.device(device)

    def load(self, skip_warmup: bool = False) -> None:
        self.app.load(skip_warmup=skip_warmup)


class DeviceTextEncoder(DeviceComponent):
    """A text encoder on the device whose tokenizer every rank runs: the encoder's TP
    collectives need every rank, and every rank then holds the full hidden states.

    ``load`` reads ``<model>/tokenizer`` on every rank in its own status-synced phase
    (``"tokenizer load"``, configured by ``_configure_tokenizer``), then loads the encoder."""

    def __init__(self, app, *, model_path, device) -> None:
        super().__init__(app, device=device)
        self.model_path = str(model_path)
        self.tokenizer = None  # every rank, after load()

    def load(self, skip_warmup: bool = False) -> None:
        with collective_phase("tokenizer load", device=self.device):
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.model_path, "tokenizer"))
            self._configure_tokenizer(tokenizer)
            self.tokenizer = tokenizer
        super().load(skip_warmup=skip_warmup)

    def _configure_tokenizer(self, tokenizer) -> None:
        """Adjust the tokenizer the encode expects (padding side, pad token); nothing by
        default."""

    def _require_tokenizer(self) -> None:
        if self.tokenizer is None:
            raise RuntimeError("the device text encoder is not loaded; call load() first")


class TorchNeuronModelApplication(torch.nn.Module):
    """The composed application: the DiT(s) on this rank's core, a text encoder, optionally a
    device VAE, and the model's orchestrator, on every rank.

    Factory signature (``difflet/registry.py`` ``create_application``): ``model_path`` is the
    diffusers snapshot root, ``shape`` the request's shape. Keyword arguments and their defaults
    are the subclass's ``KWARG_DEFAULTS``; ``device`` is ``"neuron"`` except in CPU tests
    (``"cpu"`` on gloo). ``__init__`` leaves the merged options in ``self.options`` for the
    subclass to build its components from, after the common refusals; ``load`` brings them up.

    A subclass sets ``MODEL_LABEL``, ``KWARG_DEFAULTS``, ``TEACACHE_KWARGS`` (and, for a
    bundle-based DiT call, ``DIT_BUNDLE``/``DIT_CALL``) and implements ``_resolve_shape``,
    ``_orchestrator_class``, ``_dit_adapter`` and ``_build_pipeline``; it overrides the other
    hooks where its model differs.
    """

    #: The model's name in messages ("Wan", "Qwen-Image", "LTX-2").
    MODEL_LABEL: ClassVar[str] = ""
    #: The keyword arguments the application accepts, with their defaults.
    KWARG_DEFAULTS: ClassVar[dict[str, Any]] = {}
    #: The TeaCache keyword arguments, refused when not None (phase 1).
    TEACACHE_KWARGS: ClassVar[tuple[str, ...]] = ()
    #: The DiT input bundle a one-argument call is a DiT forward for, if any.
    DIT_BUNDLE: ClassVar[type | tuple[type, ...]] = ()
    #: What a positional DiT call looks like, for ``__call__``'s message.
    DIT_CALL: ClassVar[str] = "a DiT bundle"

    def __init__(self, *, model_path, parallel, dtype: Any, shape: dict[str, int | None],
                 **kwargs: Any) -> None:
        super().__init__()
        unknown = sorted(set(kwargs) - set(self.KWARG_DEFAULTS))
        if unknown:
            raise TypeError(
                f"{type(self).__name__} got unexpected keyword arguments: {', '.join(unknown)}"
            )
        options = {**self.KWARG_DEFAULTS, **kwargs}
        self._reject_unsupported(options)
        check_parallel_supported(parallel)
        self.options = options
        self.model_path = str(model_path)
        self.parallel = parallel
        self.dtype = _normalize_dtype(dtype)
        self.device = _resolve_device(options["device"])
        self.shape = self._resolve_shape(shape, options["shapes"])
        self.text_seq_len = int(options["text_seq_len"])
        # The components the subclass builds, and what load() fills in.
        self.transformer: TorchNeuronApplicationBase | None = None
        self.text_encoder: Any = None
        self.vae_decoder: DeviceComponent | None = None
        self.transformer_adapter: OnDeviceDiT | None = None
        self.pipeline = None
        #: Called with no arguments once per denoise iteration (``_stamp_denoise_iterations``).
        self.step_hook: Callable[[], None] | None = None
        self._text_encoder_load_seconds: float | None = None

    # ------------------------------------------------------------ construction hooks

    def _reject_unsupported(self, options: dict[str, Any]) -> None:
        """The phase-1 refusals, before anything is read: TeaCache, batch > 1, then the model's
        own (``_reject_model_options``), then an unknown text encoder placement."""
        teacache = [name for name in self.TEACACHE_KWARGS if options[name] is not None]
        if options.get("teacache_fused"):
            teacache.append("teacache_fused")
        if teacache:
            raise NotImplementedError(
                "TeaCache is not supported on the neuron backend in phase 1 "
                f"({', '.join(teacache)})"
            )
        if int(options["batch_size"]) != 1:
            raise NotImplementedError(
                f"the neuron {self.MODEL_LABEL} application runs batch_size=1 only (phase 1), "
                f"got batch_size={options['batch_size']}"
            )
        self._reject_model_options(options)
        if options["text_encoder_device"] not in ("host", "neuron"):
            raise ValueError(
                "text_encoder_device must be 'host' or 'neuron', got "
                f"{options['text_encoder_device']!r}"
            )

    def _reject_model_options(self, options: dict[str, Any]) -> None:
        """The model's own option refusals; none by default."""

    def _resolve_shape(self, shape: dict[str, int | None], shapes) -> dict[str, Any]:
        """The one static shape of this process from the request and ``shapes``."""
        raise NotImplementedError

    def _check_components_fit(self, components: Sequence[Component], *, what: str,
                              advice: str) -> None:
        """Raise before anything loads when ``components`` (a device text encoder beside the
        DiT) clearly cannot share one core (``difflet/backends/neuron/hbm.py``): the base's
        load-time check sees one component at a time, so it would pass the DiT and refuse the
        encoder only after the DiT had loaded. Like that check, this one runs on the neuron
        device, or elsewhere when ``DIFFLET_NEURON_HBM_GIB`` sets a budget."""
        if not budget_enforced(self.device.type):
            return
        tp = int(self.parallel.tp_degree)
        check_fits(components_bytes(self.model_path, components, tp_degree=tp), what=what,
                   tp_degree=tp, advice=advice)

    # --------------------------------------------------- DiffletPipeline API

    def compile(self, compiled_model_path, debug: bool = False) -> None:
        """No-op: the neuron backend compiles nothing ahead of time (see ``load``)."""
        del compiled_model_path, debug

    def has_compiled_artifacts(self, compiled_model_path) -> bool:
        del compiled_model_path
        return True

    def load(self, compiled_model_path=None, start_rank_id=None, local_ranks_size=None,
             skip_warmup: bool = False) -> None:
        """The DiT(s) (runtime init, shards, per-block compile, warm-up), then the text encoder
        (rank 0's host, or the device), then the decoders, then the orchestrator. Every rank
        must call it; the steps are status-synced phases."""
        del compiled_model_path, start_rank_id, local_ranks_size
        if self.pipeline is not None:
            return
        orchestrator_class = self._orchestrator_class()
        self._load_transformers(skip_warmup)
        self._load_text_encoder(skip_warmup)
        self._load_decoders(skip_warmup)
        pipeline = self._build_pipeline(orchestrator_class)
        self._stamp_denoise_iterations(pipeline)
        self.pipeline = pipeline
        logger.info("%s loaded: %s", type(self).__name__,
                    {k: round(v, 2) for k, v in self.phase_seconds.items()})

    def _orchestrator_class(self) -> type:
        """The orchestrator class, imported before anything loads."""
        raise NotImplementedError

    def _dit_adapter(self, app: TorchNeuronApplicationBase) -> OnDeviceDiT:
        raise NotImplementedError

    def _build_pipeline(self, orchestrator_class: type):
        raise NotImplementedError

    def _load_transformers(self, skip_warmup: bool) -> None:
        if self.transformer is not None:
            self.transformer.load(skip_warmup=skip_warmup)
            self.transformer_adapter = self._dit_adapter(self.transformer)

    def _load_text_encoder(self, skip_warmup: bool) -> None:
        if isinstance(self.text_encoder, DeviceComponent):
            self.text_encoder.load(skip_warmup=skip_warmup)
        elif self.text_encoder is not None:
            start = time.perf_counter()
            self.text_encoder.load()
            self._text_encoder_load_seconds = time.perf_counter() - start

    def _load_decoders(self, skip_warmup: bool) -> None:
        if self.vae_decoder is not None:
            self.vae_decoder.load(skip_warmup=skip_warmup)

    def _stamp_denoise_iterations(self, pipeline) -> None:
        """Call ``step_hook`` after each scheduler step that ends a denoise iteration
        (``_ends_iteration``): exactly once per iteration, in every branch of the denoise loop."""
        scheduler_step = pipeline._scheduler_step

        def stamped_scheduler_step(*args: Any, **kwargs: Any) -> torch.Tensor:
            latents = scheduler_step(*args, **kwargs)
            if self.step_hook is not None and self._ends_iteration(pipeline, args):
                self.step_hook()
            return latents

        pipeline._scheduler_step = stamped_scheduler_step

    def _ends_iteration(self, pipeline, args: tuple) -> bool:
        """Whether the scheduler step called with ``args`` ends a denoise iteration: every step
        does, unless the orchestrator steps several schedulers per iteration."""
        del pipeline, args
        return True

    # ------------------------------------------------------------- evidence

    def _device_apps(self) -> list[tuple[str, TorchNeuronApplicationBase]]:
        """``(prefix, application)`` of every component on the device, in load order; the DiT's
        prefix is empty."""
        apps: list[tuple[str, TorchNeuronApplicationBase]] = []
        if self.transformer is not None:
            apps.append(("", self.transformer))
        if isinstance(self.text_encoder, DeviceComponent):
            apps.append(("text_encoder", self.text_encoder.app))
        if self.vae_decoder is not None:
            apps.append(("vae", self.vae_decoder.app))
        return apps

    @property
    def phase_seconds(self) -> dict[str, float]:
        """The DiT's load phases, every other device component's as ``"<prefix> <phase>"``
        (``"text_encoder load checkpoint"``, ``"vae warmup forward"``), and for a host text
        encoder ``"text encoder load"`` (rank 0's read, which the other ranks wait out in that
        phase's status sync)."""
        seconds: dict[str, float] = {}
        for prefix, app in self._device_apps():
            seconds.update({f"{prefix} {k}" if prefix else k: v
                            for k, v in app.phase_seconds.items()})
        if self._text_encoder_load_seconds is not None:
            seconds["text encoder load"] = self._text_encoder_load_seconds
        return seconds

    @property
    def compiled_blocks(self) -> list[str]:
        return [f"{prefix}.{name}" if prefix else name
                for prefix, app in self._device_apps() for name in app.compiled_blocks]

    @property
    def unwarmed_shapes(self) -> list:
        return [shape for _, app in self._device_apps() for shape in app.unwarmed_shapes]

    # ------------------------------------------------------------- inference

    def _require_loaded(self) -> None:
        if self.pipeline is None:
            raise RuntimeError(f"{type(self).__name__} is not loaded; call load() first")

    def _require_dit(self) -> OnDeviceDiT:
        if self.transformer_adapter is None:
            raise RuntimeError(
                f"{type(self).__name__}: the DiT is not loaded (call load() first; "
                "enable_transformer=False never loads one)"
            )
        return self.transformer_adapter

    def forward_dit(self, bundle):
        """One DiT forward on a host bundle, the output on the host."""
        return self._require_dit()(bundle)

    def _single_prompt(self, prompt) -> str:
        """The one prompt of a batch-1 call (a string, or a one-element list)."""
        if isinstance(prompt, (list, tuple)):
            if len(prompt) != 1:
                raise NotImplementedError(
                    f"the neuron {self.MODEL_LABEL} application runs batch 1, got "
                    f"{len(prompt)} prompts"
                )
            prompt = prompt[0]
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a string, got {type(prompt).__name__}")
        return prompt

    @staticmethod
    def _apply_seed(kwargs: dict[str, Any], seed: int | None) -> None:
        """``seed``: a ``torch.Generator`` seeded with it, as ``kwargs["generator"]``."""
        if seed is not None:
            if kwargs.get("generator") is not None:
                raise ValueError("pass seed or generator, not both")
            kwargs["generator"] = torch.Generator().manual_seed(int(seed))

    def _draw_on_rank0(self, draw: Callable[[], Any], count: int | None = None) -> Any:
        """``draw()`` (the initial noise, from the caller's generator or the global RNG) on
        rank 0, broadcast: every rank of the TP/CP/CFG group denoises the same latents even when
        the ranks' generators or global RNGs differ. Rank 0's draw is the one the orchestrator
        itself would have made, so one rank, or ranks given identical generators, get the same
        latents as without it. ``count`` is the number of tensors ``draw`` returns as a tuple,
        None for one tensor."""
        if world_info()[1] == 1:
            return draw()
        drawn = rank0_call(draw, device=self.device, what="initial latents")
        if count is None:
            return broadcast_from_rank0(None if drawn is None else (drawn,), 1,
                                        device=self.device)[0]
        return tuple(broadcast_from_rank0(drawn, count, device=self.device))

    def _is_dit_call(self, args: tuple, kwargs: dict[str, Any]) -> bool:
        """Whether a call other than ``app(bundle)`` is a direct DiT call; none by default."""
        del args, kwargs
        return False

    def generate(self, **kwargs: Any):
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any):
        """``app(bundle)`` (and the model's other DiT calls, ``_is_dit_call``) run the DiT;
        ``app("prompt", ...)`` and every keyword-only call run ``generate`` (the orchestrator).
        """
        if len(args) == 1 and isinstance(args[0], self.DIT_BUNDLE) and not kwargs:
            return self.forward_dit(args[0])
        if self._is_dit_call(args, kwargs):
            return self._require_dit()(*args, **kwargs)
        if len(args) == 1 and isinstance(args[0], (str, list)):
            if "prompt" in kwargs:
                raise TypeError("prompt given both positionally and as a keyword")
            kwargs["prompt"] = args[0]
            args = ()
        if args:
            raise TypeError(
                f"{type(self).__name__}: positional arguments are {self.DIT_CALL} or a "
                f"prompt; got {len(args)}"
            )
        return self.generate(**kwargs)
