"""HunyuanVideo application for the neuron backend (TorchNeuron, one process per NeuronCore).

``TorchNeuronHunyuanVideoApplication`` keeps the outward contract of the Trainium
``NeuronHunyuanVideoApplication`` (``difflet/models/hunyuan_video/application.py``): the
registry factory signature, ``compile``/``has_compiled_artifacts``/``load``, ``.pipeline``,
``dit_input_contract()``, ``forward_dit(bundle)``, and a ``__call__`` that sends a
``HunyuanVideoDiTInputBundle`` to the DiT and every other call to the
``HunyuanVideoOrchestrator``. Every rank of the torchrun launch builds one and holds:

* ``transformer``: ``NeuronHunyuanVideoTransformerApplication``, the DiT on this rank's core
  (Difflet's port, TP shards, with SP or CP in any ``cp_mode``);
* ``text_encoder``: ``_HostHunyuanVideoTextEncoder``, on rank 0's host only: diffusers' own
  ``HunyuanVideoPipeline.encode_prompt`` (the Llama-3 8B encoder with the prompt template, the
  95-token crop and hidden layer -3, right-padded to ``text_seq_len``; CLIP-L's pooled output
  at 77 tokens) in the application dtype, so the tensors are the ones the AoT path's ``llama``
  and ``clip`` stages produce, without NxD. Rank 0 broadcasts the embeddings, the int64 mask
  and the pooled projection;
* ``pipeline``: the backend-neutral ``HunyuanVideoOrchestrator`` (FlowMatch Euler on the host,
  the guidance ``scale x 1000`` in the timestep embedding: the model is guidance-distilled, one
  DiT forward per step, no CFG), which every rank runs on identical inputs; its DiT calls go
  through ``_OnDeviceHunyuanVideoTransformer``.

The decode is not part of this application: the VAE runs after the generate launch has exited,
in its own one-core launch (as for HunyuanVideo 1.5, LTX-2 and Wan), so ``enable_vae_decoder``
is refused. ``output_type="latent"`` is the generate stage's output.

Parallelism: tensor parallelism, with or without sequence parallelism, and context parallelism
in every ``cp_mode`` (gather_kv, ulysses, ring). CFG parallelism does not apply (the entry and
the config refuse it); data parallelism is separate replicas. Each rank starts from rank 0's
initial noise (``generate``).

Phase 1 limits, each rejected in ``__init__`` (or, for the shape, at every DiT call): one
static shape, batch 1, the text encoder on the host. TeaCache runs, every mode the orchestrator
has: the probe-free fixed cadence and online delta, and adaptive TeaCache, whose per-step signal
is ``_HunyuanVideoTeaCacheProbe`` on the DiT's own modules (block 0's modulated input, the
absolute L2 distance HunyuanVideo's calibrations were fit on), fed to the orchestrator's fused
probe branch.

Timing hooks (for the benchmark): ``step_hook`` is called with no arguments once per denoise
iteration, after the scheduler step; ``transformer_adapter.forward_hook`` is called after every
DiT forward with that call's wall seconds.

The lifecycle, the evidence properties, the hooks, the rank-0 encoder plumbing and the
initial-latents broadcast are the composed neuron application's (``TorchNeuronModelApplication``,
``difflet/backends/neuron/core/model_application.py``); what is HunyuanVideo's is here.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from difflet.backends.neuron.core.model_application import (
    HostTextEncoder,
    OnDeviceDiT,
    TorchNeuronModelApplication,
    broadcast_from_rank0,
)
from difflet.backends.neuron.core.teacache import NeuronTeaCacheProbe
from difflet.backends.neuron.hunyuan_video.config import NeuronHunyuanVideoConfig
from difflet.backends.neuron.hunyuan_video.transformer import (
    NeuronHunyuanVideoTransformerApplication,
)
from difflet.models.hunyuan_video.application import HunyuanVideoDiTInputBundle

#: The keyword arguments the application accepts, with their defaults: the ones the HunyuanVideo
#: CLI stage passes the Trainium application (difflet/cli/orchestrators/hunyuan_video.py) plus
#: exec_mode, device, batch_size, text_encoder_device and encoder_threads. Anything else is a
#: TypeError rather than silently ignored.
_KWARG_DEFAULTS: dict[str, Any] = {
    "text_seq_len": 256,
    "batch_size": 1,
    "shapes": None,
    "enable_vae_decoder": False,
    "exec_mode": None,
    "device": "neuron",
    "encoder_threads": 12,
    "text_encoder_device": "host",
    # The Trainium CLI's opt-in for its separately compiled probe NEFF. Here the probe is the
    # DiT's own modules, attached whenever adaptive TeaCache runs (teacache_speedup).
    "enable_teacache_probe": None,
    "teacache_speedup": None,
    "teacache_calibration_path": None,
    "teacache_cadence": None,
    "teacache_online_delta_alpha": None,
}
#: The registry's default shape.
_DEFAULT_SHAPE = {"height": 320, "width": 512, "num_frames": 61}
#: The orchestrator's default step count and guidance scale when the call names none.
_DEFAULT_STEPS = 4
_DEFAULT_GUIDANCE = 6.0
#: The host encoder's discarded warm-up encode at load (``HostTextEncoder.warmup_args``).
_WARMUP_PROMPT = "a photo"


class _HostHunyuanVideoTextEncoder(HostTextEncoder):
    """The HunyuanVideo text path on rank 0's host; every rank receives its output by broadcast.

    ``__call__(prompt)`` returns ``(embeds [1, seq_len, 4096], mask [1, seq_len] int64, pooled
    [1, 768])``: diffusers' ``HunyuanVideoPipeline.encode_prompt`` (loaded without its
    transformer and VAE, in ``dtype``) on rank 0 (``HostTextEncoder``: with ``threads`` intra-op
    threads, inside ``rank0_call``), the mask cast to int64 (the DiT's warm-up dtype) before the
    broadcast. The Llama prompt is right-padded to ``seq_len`` after the template's crop, so the
    mask is a valid-token prefix.

    ``load()`` reads the encoders on rank 0 and, as for the Qwen-Image and LTX-2 host encoders,
    spends one discarded encode (``warmup_args``): a fresh process's first bf16 encode is not
    reproducible on a busy host, so no request is a process's first.
    """

    warmup_args = (_WARMUP_PROMPT,)

    def __init__(self, model_path, *, seq_len: int, dtype: torch.dtype, device, threads: int):
        super().__init__(model_path, device=device, threads=threads)
        self.seq_len = int(seq_len)
        self.dtype = dtype
        self.pipe = None  # rank 0 only, after load()

    def _load(self) -> None:
        from diffusers import HunyuanVideoPipeline

        pipe = HunyuanVideoPipeline.from_pretrained(self.model_path, transformer=None, vae=None,
                                                    torch_dtype=self.dtype)
        pipe.to("cpu")
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

    def __call__(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self._rank0_encode(prompt)
        return tuple(broadcast_from_rank0(encoded, 3, device=self.device))

    def _encode(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.pipe is None:
            raise RuntimeError("the text encoder is not loaded on rank 0; call load() first")
        with self._pinned():
            embeds, pooled, mask = self.pipe.encode_prompt(
                prompt=prompt, prompt_2=None, device=torch.device("cpu"),
                max_sequence_length=self.seq_len,
            )
        return (embeds.to(self.dtype).contiguous(), mask.to(torch.int64).contiguous(),
                pooled.to(self.dtype).contiguous())


class _HunyuanVideoTeaCacheProbe(NeuronTeaCacheProbe):
    """The shared probe with HunyuanVideo's metric: ``delta_t = ||m_t - m_{t-1}||_2`` (fp32),
    the absolute L2 distance the controller's host diff computes and the model's calibrations
    were fit on (``difflet/pipeline/teacache.py``, the Trainium probe), not the shared relative
    L1. ``m_{t-1}`` starts at zero, as in the shared probe; the controller's warm-up absorbs that
    first delta."""

    @torch.no_grad()
    def teacache_delta(self, *inputs):
        m = self.mod_input(*inputs)
        prev = self._prev if self._prev is not None else torch.zeros_like(m)
        delta = torch.linalg.vector_norm((m.float() - prev.float()).reshape(-1))
        self._prev = m
        return delta


class _OnDeviceHunyuanVideoTransformer(OnDeviceDiT):
    """The orchestrator's DiT callable (``OnDeviceDiT``): a ``HunyuanVideoDiTInputBundle`` in,
    the host noise prediction out.

    A call at any other shape than the static one raises (``_check_dit_shapes``) instead of
    compiling new block graphs. The six tensors go to the device in the bundle's order; the
    floats in ``dtype``, the int64 text mask as it is (``_device_input``): the warm-up compiled
    the blocks for an int64 mask, so a cast one would compile anew. With ``probe`` (adaptive
    TeaCache), ``teacache_probe_fused`` is True and ``teacache_delta(bundle)`` returns the
    probe's per-step signal: the orchestrator's fused branch.
    """

    def __init__(self, module, config: NeuronHunyuanVideoConfig, dtype, device,
                 probe: _HunyuanVideoTeaCacheProbe | None = None) -> None:
        super().__init__(module, config, dtype, device)
        self.probe = probe

    @property
    def teacache_probe_fused(self) -> bool:
        return self.probe is not None

    def __call__(self, bundle: HunyuanVideoDiTInputBundle) -> torch.Tensor:
        if not isinstance(bundle, HunyuanVideoDiTInputBundle):
            raise TypeError(f"expected a HunyuanVideoDiTInputBundle, got {type(bundle).__name__}")
        _check_dit_shapes(self.config, bundle)
        return self._forward(bundle.as_model_inputs())

    def _device_input(self, tensor: torch.Tensor) -> torch.Tensor:
        if not tensor.is_floating_point():
            return tensor.contiguous().to(self.device)
        return super()._device_input(tensor)

    def teacache_delta(self, bundle: HunyuanVideoDiTInputBundle) -> torch.Tensor:
        if self.probe is None:
            raise RuntimeError("no TeaCache probe: adaptive TeaCache (teacache_speedup) is off")
        return self.probe.teacache_delta(
            *(t.contiguous() for t in (bundle.hidden_states, bundle.timestep,
                                       bundle.pooled_projections, bundle.guidance)))


def dit_input_shapes(config: NeuronHunyuanVideoConfig) -> dict[str, tuple[int, ...]]:
    """The per-call DiT input shapes at the config's one static shape, batch 1: exactly the
    warm-up's (``get_example_inputs``)."""
    batch, text = int(config.batch_size), int(config.text_seq_len)
    return {
        "hidden_states": (batch, int(config.in_channels), config.latent_frames,
                          config.latent_height, config.latent_width),
        "timestep": (batch,),
        "encoder_hidden_states": (batch, text, int(config.text_embed_dim)),
        "encoder_attention_mask": (batch, text),
        "pooled_projections": (batch, int(config.pooled_projection_dim)),
        "guidance": (batch,),
    }


def _check_dit_shapes(config: NeuronHunyuanVideoConfig,
                      bundle: HunyuanVideoDiTInputBundle) -> None:
    """Refuse a DiT call at any other shape than the one the warm-up compiled.

    In compile mode a new shape does not fail on its own: every compiled block traces and
    compiles a new graph, and Dynamo fails only after ``recompile_limit`` shapes. In eager mode
    the rotary tables and the CP/SP sequence split assume the static shape too. Every rank
    receives the same call, so every rank raises here, before any device work or collective.
    """
    for name, shape in dit_input_shapes(config).items():
        actual = tuple(getattr(bundle, name).shape)
        if actual != shape:
            raise ValueError(
                f"HunyuanVideo DiT input {name!r} has shape {actual}, expected {shape} (the one "
                f"static shape, {config.height}x{config.width}x{config.num_frames}, "
                f"text_seq_len={config.text_seq_len}); the neuron backend compiles one static "
                "shape per process"
            )


class TorchNeuronHunyuanVideoApplication(TorchNeuronModelApplication):
    """Host Llama/CLIP encode on rank 0 + the DiT on each rank's core +
    ``HunyuanVideoOrchestrator``, on every rank.

    Factory signature (``difflet/registry.py`` ``create_application``): ``model_path`` is the
    diffusers snapshot root, ``shape`` the request's ``height``/``width``/``num_frames``. Keyword
    arguments and their defaults are ``_KWARG_DEFAULTS``; ``device`` is ``"neuron"`` except in
    CPU tests (``"cpu"`` on gloo).
    """

    MODEL_LABEL = "HunyuanVideo"
    KWARG_DEFAULTS = _KWARG_DEFAULTS
    #: Nothing refused: TeaCache runs on HunyuanVideo's neuron path (the options reach the
    #: orchestrator in ``_build_pipeline``; the adaptive probe is ``_dit_adapter``'s).
    TEACACHE_KWARGS = ()
    DIT_BUNDLE = HunyuanVideoDiTInputBundle
    DIT_CALL = "a HunyuanVideoDiTInputBundle"

    def __init__(self, *, model_path, parallel, dtype: Any, shape: dict[str, int | None],
                 **kwargs: Any) -> None:
        super().__init__(model_path=model_path, parallel=parallel, dtype=dtype, shape=shape,
                         **kwargs)
        options = self.options
        transformer_path = os.path.join(self.model_path, "transformer")
        config = NeuronHunyuanVideoConfig.from_pretrained(
            transformer_path, **self.shape, batch_size=1, text_seq_len=self.text_seq_len,
            tp_degree=int(parallel.tp_degree),
            context_parallel_enabled=int(getattr(parallel, "cp_degree", 1) or 1) > 1,
            cp_mode=str(getattr(parallel, "cp_mode", "gather_kv") or "gather_kv"),
            sp_enabled=bool(getattr(parallel, "sp_enabled", False)),
            cfg_parallel_enabled=bool(getattr(parallel, "cfg_parallel_enabled", False)),
        )
        self.transformer = NeuronHunyuanVideoTransformerApplication(
            model_path=transformer_path, config=config, parallel=parallel, dtype=self.dtype,
            exec_mode=options["exec_mode"], device=self.device,
        )
        self.config = self.transformer.config  # cp_degree filled from the parallel config
        self.text_encoder = _HostHunyuanVideoTextEncoder(
            self.model_path, seq_len=self.text_seq_len, dtype=self.dtype, device=self.device,
            threads=int(options["encoder_threads"]),
        )

    def _reject_model_options(self, options: dict[str, Any]) -> None:
        if options["text_encoder_device"] != "host":
            raise NotImplementedError(
                "the neuron HunyuanVideo application encodes the prompt on rank 0's host only "
                "(Llama and CLIP through diffusers' HunyuanVideoPipeline.encode_prompt); got "
                f"text_encoder_device={options['text_encoder_device']!r}"
            )
        if options["enable_vae_decoder"]:
            raise NotImplementedError(
                "the neuron HunyuanVideo application does not decode: the VAE runs in its own "
                "one-core vae launch after the generate launch (enable_vae_decoder=True)"
            )

    def _resolve_shape(self, shape: dict[str, int | None], shapes) -> dict[str, int]:
        resolved = {name: int(shape.get(name) or default)
                    for name, default in _DEFAULT_SHAPE.items()}
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

    # --------------------------------------------------------------------- load

    def _orchestrator_class(self) -> type:
        from difflet.models.hunyuan_video.pipeline import HunyuanVideoOrchestrator

        return HunyuanVideoOrchestrator

    def _dit_adapter(self, app) -> _OnDeviceHunyuanVideoTransformer:
        """The adapter, with the TeaCache probe when adaptive TeaCache runs (the only mode the
        orchestrator asks for a per-step signal in)."""
        probe = None
        if self.options["teacache_speedup"] is not None:
            probe = _HunyuanVideoTeaCacheProbe(app.teacache_mod_input)
        return _OnDeviceHunyuanVideoTransformer(app, self.config, self.dtype, self.device,
                                                probe=probe)

    def _build_pipeline(self, orchestrator_class: type):
        options = self.options
        return orchestrator_class(
            model_path=self.model_path,
            transformer=self.transformer_adapter,
            dtype=self.dtype,
            teacache_speedup=options["teacache_speedup"],
            teacache_calibration_path=options["teacache_calibration_path"],
            teacache_cadence=options["teacache_cadence"],
            teacache_online_delta_alpha=options["teacache_online_delta_alpha"],
            **self.shape,
        )

    # ----------------------------------------------------------------- evidence

    def dit_input_contract(self) -> dict[str, dict[str, Any]]:
        """The Trainium contract's six tensors at the static shape (``dit_input_shapes``)."""
        dtypes = {"encoder_attention_mask": torch.int64}
        return {name: {"shape": shape, "dtype": dtypes.get(name, self.dtype)}
                for name, shape in dit_input_shapes(self.config).items()}

    # ---------------------------------------------------------------- inference

    def encode_prompt(self, prompt) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(embeds, mask int64, pooled)`` (see ``_HostHunyuanVideoTextEncoder``), identical on
        every rank. Every rank must call it."""
        return self.text_encoder(self._single_prompt(prompt))

    def generate(self, *, prompt=None, seed: int | None = None,
                 num_inference_steps: int = _DEFAULT_STEPS,
                 guidance_scale: float = _DEFAULT_GUIDANCE, output_type: str = "latent",
                 **kwargs: Any):
        """The orchestrator's ``__call__`` (``bundle`` or ``latents`` and the three text
        tensors, ``timesteps``, ``return_trajectory``, ...), with:

        * ``prompt``: encoded by ``encode_prompt`` (rank 0) into ``encoder_hidden_states``,
          ``encoder_attention_mask`` and ``pooled_projections``;
        * ``seed``: a ``torch.Generator`` seeded with it (pass one or ``generator``);
        * the initial noise: without ``latents`` or ``bundle``, fp32 ``randn`` at the static
          latent shape from ``generator`` (diffusers' ``prepare_latents`` draw) on rank 0,
          broadcast, so a TP/CP group denoises the same latents even when the global RNGs
          differ;
        * the TeaCache probe's previous signal cleared, so each request starts as the first.
        """
        self._require_loaded()
        if prompt is not None:
            if kwargs.get("bundle") is not None or kwargs.get("encoder_hidden_states") is not None:
                raise ValueError("pass prompt or the text tensors (or a bundle), not both")
            embeds, mask, pooled = self.encode_prompt(prompt)
            kwargs.update(encoder_hidden_states=embeds, encoder_attention_mask=mask,
                          pooled_projections=pooled)
        self._apply_seed(kwargs, seed)
        generator = kwargs.pop("generator", None)
        if kwargs.get("bundle") is None and kwargs.get("latents") is None:
            kwargs["latents"] = self._initial_latents(generator)
        probe = self._require_dit().probe
        if probe is not None:
            probe.reset()
        return self.pipeline(num_inference_steps=num_inference_steps,
                             guidance_scale=guidance_scale, output_type=output_type, **kwargs)

    def _initial_latents(self, generator) -> torch.Tensor:
        """Rank 0's fp32 initial noise at the static latent shape, broadcast (see ``generate``)."""
        shape = dit_input_shapes(self.config)["hidden_states"]

        def draw() -> torch.Tensor:
            return torch.randn(shape, generator=generator, dtype=torch.float32)

        return self._draw_on_rank0(draw)


__all__ = ["TorchNeuronHunyuanVideoApplication", "dit_input_shapes"]
