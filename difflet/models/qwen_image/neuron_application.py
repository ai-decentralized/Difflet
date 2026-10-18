"""Qwen-Image application for the neuron backend (TorchNeuron, one process per NeuronCore).

``TorchNeuronQwenImageApplication`` keeps the outward contract of the Trainium
``NeuronQwenImageApplication`` (``difflet/models/qwen_image/application.py``, which already owns
that name) and of ``TpuQwenImageApplication``: the registry factory signature,
``compile``/``has_compiled_artifacts``/``load``, ``.pipeline``, ``dit_input_contract()``,
``forward_dit(bundle)``, and a ``__call__`` that sends a ``QwenImageDiTInputBundle`` or a
positional DiT call to the DiT and every other call to the ``QwenImageOrchestrator``. Every rank
of the torchrun launch builds one and holds:

* ``transformer``: ``NeuronQwenImageTransformerApplication``, the DiT on this rank's core (TP
  shards, and under CP this rank's image tokens);
* ``text_encoder``: ``_HostQwenTextEncoder``, Qwen2.5-VL on rank 0's host only (bf16 by
  default, one copy instead of one per rank), its embeddings broadcast to every rank. It
  reproduces the Trainium text stage's embeddings (``difflet/cli/orchestrators/qwen_image.py``
  ``_stage_text``): see that class for the semantics and the divergences from diffusers. Or,
  with ``text_encoder_device="neuron"`` (opt-in; the host stays the default),
  ``_DeviceQwenTextEncoder``: Qwen2.5-VL's language model on the device in ``text_encoder_dtype``
  (bf16), sharded across the TP group (``NeuronQwenTextEncoderApplication``, about 3.3 GiB per
  rank at tp4), every rank tokenizing and encoding the same prompt at the static
  ``(1, ENC_SEQ)`` shape, with the host encoder's post-processing. ``__init__`` refuses it when
  it cannot fit beside the DiT on one core at this tp (``_check_device_text_encoder_fits``: tp2,
  so CP2 or dp2tp2, does not fit);
* ``vae_decoder`` (``enable_vae_decoder=True``): ``_DeviceQwenVaeDecoder``, the decoder on every
  rank's core in fp32, eager. Qwen-Image's VAE (``AutoencoderKLQwenImage``) is diffusers' Wan 2.1
  VAE under another name, so this reuses ``NeuronWanVaeDecoderApplication``, as the Trainium
  VAE stage reuses its Wan decoder; the checkpoint's 108 decoder tensors map key for key and
  shape for shape onto ``WanVAEDecoderModel``. ``output_type="pt"`` then returns decoded images
  ``[B, 3, H, W]`` in [-1, 1]. Without it the VAE decodes on the host outside this application;
* ``pipeline``: the backend-neutral ``QwenImageOrchestrator`` (FlowMatch Euler on the host, the
  latents on the host in the application dtype, as on Trainium), which every rank runs on
  identical inputs; its DiT calls go through ``_OnDeviceTransformer``.

The text mask is dropped by the DiT, as on Trainium and TPU: the embeddings are zero-padded to
``text_seq_len`` and attended unmasked (``difflet/backends/neuron/qwen_image/transformer.py``).
So ``encoder_hidden_states`` must arrive zero-padded to exactly ``text_seq_len`` rows; a shorter
diffusers-style tensor is rejected rather than silently mis-shaped.

Parallelism: tensor parallelism, with or without sequence parallelism (Megatron SP within the
TP group: the ``QwenImageSPTransformer2DModel`` fork, see
``difflet/backends/neuron/qwen_image/transformer.py``), and context parallelism in every
``cp_mode`` (gather_kv, ulysses, ring); SP and CP are mutually exclusive, as for Wan
(``NeuronQwenImageConfig``).

Guidance: one DiT forward per step by default (the guidance scale is dropped: the checkpoint
has no guidance embedder). True classifier-free guidance is opt-in, ``true_cfg_scale`` > 1 with
a ``negative_prompt`` (``QwenImageOrchestrator``, diffusers' formula): two forwards per step, or
with CFG parallelism (``cfg_parallel_enabled``, not with CP) one batch-2 call whose branches run
on the two CFG groups. CFG parallelism refuses a call without true CFG.

Phase 1 limits, each rejected in ``__init__``: one static shape, batch 1, no TeaCache and no DP
within one application.

Timing hooks (for the benchmark): ``step_hook`` is called with no arguments once per denoise
iteration, right after the orchestrator's scheduler step (``load`` wraps
``QwenImageOrchestrator._scheduler_step`` on this instance); ``transformer_adapter.forward_hook``
is called after every DiT forward with that call's wall seconds (host -> device copies, forward,
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
from difflet.backends.neuron.core.distributed import broadcast_tensor, rank0_call, world_info
from difflet.backends.neuron.hbm import budget_enforced, check_fits
from difflet.backends.neuron.qwen_image.config import NeuronQwenImageConfig
from difflet.backends.neuron.qwen_image.hbm import device_text_encoder_bytes
from difflet.backends.neuron.qwen_image.text_encoder import NeuronQwenTextEncoderApplication
from difflet.backends.neuron.qwen_image.transformer import NeuronQwenImageTransformerApplication
from difflet.backends.neuron.runtime import check_parallel_supported
from difflet.backends.neuron.wan.vae import NeuronWanVaeDecoderApplication
from difflet.common.orchestrators.qwen_image import (
    ENC_SEQ,
    QWEN_DROP_IDX,
    QWEN_TEMPLATE,
    TEXT_SEQ_LEN,
)
from difflet.models.qwen_image.contract import QwenImageDiTInputBundle, normalize_dtype

logger = logging.getLogger(__name__)

#: The keyword arguments the application accepts, with their defaults: the ones the Qwen-Image
#: CLI generate stage passes the Trainium application (difflet/cli/orchestrators/qwen_image.py
#: ``_stage_generate``) and DiffletPipeline's TeaCache merge (``teacache_calibration``), plus
#: exec_mode, device, encoder_threads, text_encoder_dtype, text_encoder_device,
#: enable_text_encoder and enable_vae_decoder. Anything else is a TypeError rather than
#: silently ignored.
_KWARG_DEFAULTS: dict[str, Any] = {
    "text_seq_len": TEXT_SEQ_LEN,
    "batch_size": 1,
    "shapes": None,
    "enable_text_encoder": True,
    "enable_transformer": True,
    "enable_vae_decoder": False,
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
#: The dtype the text embeddings are zero-padded in: the Trainium text stage's (CLI and
#: serving) and the TPU encode's, whatever the encoder computed in.
_EMBED_DTYPE = torch.bfloat16
#: The host encoder's discarded warm-up encode at load (``_HostQwenTextEncoder._load``).
_WARMUP_PROMPT = "a photo"
#: The orchestrator's and the CLI's default step count, used to set the shifted schedule when
#: the caller passes neither ``timesteps`` nor ``num_inference_steps``.
_DEFAULT_STEPS = 4
#: Keyword names of a direct DiT call (``app(hidden_states=..., timestep=..., ...)``).
_DIT_KEYS = frozenset({"hidden_states", "timestep", "encoder_hidden_states"})


def qwen_prompt_embeds(
    hidden: torch.Tensor, valid: int, *, seq_len: int, drop_idx: int = QWEN_DROP_IDX
) -> tuple[torch.Tensor, int]:
    """The Trainium text stage's post-processing of Qwen2.5-VL's last hidden state.

    ``hidden`` is ``[1, >= valid, D]`` for the templated prompt; the template's first
    ``drop_idx`` rows (system prompt and user header) are dropped, the prompt's rows up to the
    valid length kept and zero-padded to ``seq_len`` in bf16. Returns the embeddings and the
    number of kept rows (the mask's ``True`` prefix, ``qwen_text_mask``).
    """
    used = hidden[:, drop_idx:valid]
    seq = int(used.shape[1])
    if seq > seq_len:
        raise ValueError(
            f"the prompt encodes to {seq} tokens after the {drop_idx} template tokens, more than "
            f"text_seq_len={seq_len}"
        )
    embeds = torch.zeros(hidden.shape[0], seq_len, hidden.shape[-1], dtype=_EMBED_DTYPE)
    embeds[:, :seq] = used.to(_EMBED_DTYPE)
    return embeds, seq


def qwen_text_mask(seq: int, seq_len: int, batch: int = 1) -> torch.Tensor:
    """The bool mask the text stage stores beside the embeddings: the first ``seq`` rows."""
    mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    mask[:, :seq] = True
    return mask


def tokenize_qwen_prompt(tokenizer, prompt: str, enc_seq_len: int) -> tuple[torch.Tensor, int]:
    """The Trainium text stage's tokenization, shared by the host and the device encoder:
    ``QWEN_TEMPLATE.format(prompt)`` tokenized to ``max_length=enc_seq_len``, right-padded and
    truncated to it. Returns ``input_ids [1, enc_seq_len]`` (int64) and the valid length; a
    prompt the budget truncates logs a warning (the CLI truncates silently)."""
    text = QWEN_TEMPLATE.format(prompt)
    tokens = tokenizer(
        text,
        max_length=enc_seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_ids, attention_mask = tokens.input_ids, tokens.attention_mask
    if input_ids.shape[0] != 1:
        raise ValueError("the neuron Qwen-Image application encodes one prompt (batch 1)")
    valid = int(attention_mask.sum())
    if attention_mask[:, valid:].any():
        raise ValueError("expected a right-padded prompt (the tokenizer pads on the right)")
    if valid == enc_seq_len:
        full = len(tokenizer(text, truncation=False).input_ids)
        if full > enc_seq_len:
            logger.warning(
                "Qwen-Image prompt is %d tokens with the template, more than the %d the "
                "encoder reads (ENC_SEQ); its tail is truncated, as on Trainium",
                full, enc_seq_len,
            )
    return input_ids.to(torch.int64), valid


class _HostQwenTextEncoder:
    """Qwen2.5-VL on rank 0's host; every rank receives the embeddings by broadcast.

    Embedding semantics are the Trainium text stage's (``_stage_text`` in
    ``difflet/cli/orchestrators/qwen_image.py``; ``_encode_prompt`` in
    ``difflet/serving/orchestrators/qwen_image.py``): the prompt is formatted into
    ``QWEN_TEMPLATE``, tokenized with ``<model>/tokenizer`` to ``max_length=ENC_SEQ`` (256),
    padded on the right to that length and truncated to it, and the last hidden state (after the
    final norm) is post-processed by ``qwen_prompt_embeds``: template rows dropped, the valid
    rows kept, zero-padded to ``text_seq_len`` in bf16, with a bool mask of the kept rows. The
    stage takes NxDI's captured ``norm`` output at the full padded length with ``arange``
    positions; here only the valid prefix is encoded (the language model is causal and the
    prompt right-padded, so the valid rows are unchanged, and no 256-row forward is wasted) and
    through ``model.model`` (``Qwen2_5_VLModel``), whose ``last_hidden_state`` is diffusers'
    ``hidden_states[-1]`` without the 152k-vocabulary ``lm_head`` or the per-layer states.

    Against diffusers 0.38's ``QwenImagePipeline._get_qwen_prompt_embeds`` (same template, same
    ``drop_idx`` 34, same ``hidden_states[-1]``) the Trainium semantics differ, and win:

    * token budget: diffusers truncates at ``1024 + 34`` tokens; here (as on Trainium) at
      ``ENC_SEQ`` = 256, so a prompt longer than about 217 tokens loses its tail (and the
      template's closing tokens). The CLI truncates silently; this logs a warning;
    * padding: diffusers pads to the batch's longest prompt and masks the padding in the DiT;
      here the embeddings are zero-padded to ``text_seq_len`` and the DiT attends them unmasked
      (see the module docstring);
    * dtype: diffusers keeps the encoder's dtype; here the embeddings are bf16 (the DiT's
      dtype on the device) whatever ``dtype`` the encoder ran in.

    Rank 0 tokenizes and encodes inside ``rank0_call``, so a failing encode makes every rank
    raise together instead of leaving them in the broadcast; the embeddings and the kept-row
    count are then broadcast (``broadcast_tensor`` carries no bool, so the mask is rebuilt from
    the count on every rank). The encode runs with ``threads`` intra-op threads (the other ranks
    are waiting in the status all-reduce meanwhile) and restores the previous count after.

    ``dtype`` is the encoder's compute dtype: bf16 by default (the Trainium stage's, and half the
    host memory of the 7B encoder: about 16.6 GB), float32 allowed (bf16 matmuls are emulated on
    a host without AVX512-BF16, where fp32 can be the faster of the two).

    ``load()`` reads the tokenizer and the encoder on rank 0 inside their own status-synced
    phase, ``"text encoder load"``, so a rank 0 that cannot read them fails every rank there,
    and runs one discarded encode (see ``_load``), so a request is never a process's first.
    """

    def __init__(
        self,
        model_path,
        *,
        text_dim: int,
        seq_len: int,
        dtype: torch.dtype,
        device,
        threads: int,
        enc_seq_len: int = ENC_SEQ,
    ):
        self.model_path = str(model_path)
        self.text_dim = int(text_dim)
        self.seq_len = int(seq_len)
        self.dtype = dtype
        self.device = torch.device(device)
        self.threads = int(threads)
        self.enc_seq_len = int(enc_seq_len)
        self.tokenizer = None  # rank 0 only, after load()
        self.model = None  # rank 0 only, after load()

    def load(self) -> None:
        loaded = rank0_call(self._load, device=self.device, what="text encoder load")
        if loaded is not None:
            self.tokenizer, self.model = loaded

    def _load(self):
        from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

        tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.model_path, "tokenizer"))
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            os.path.join(self.model_path, "text_encoder"), dtype=self.dtype
        ).eval().requires_grad_(False)
        # The first encode of a fresh process is not reproducible when the host CPUs are busy
        # (another DP replica, compiles, decodes): its bf16 result differed in 4 of 51 fresh
        # processes on a trn2.3xlarge, while none of the later encodes in those processes did,
        # whatever the prompt of the first one. So one encode is spent here and discarded, and
        # every request is encoded by a warmed-up process.
        self.tokenizer, self.model = tokenizer, model
        self._encode(_WARMUP_PROMPT)
        return tokenizer, model

    def __call__(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """``(encoder_hidden_states [1, seq_len, D] bf16, mask [1, seq_len] bool)`` on the host,
        identical on every rank. Every rank must call it."""
        encoded = rank0_call(self._encode, prompt, device=self.device, what="prompt encode")
        embeds = broadcast_tensor(None if encoded is None else encoded[0], device=self.device)
        count = broadcast_tensor(
            None if encoded is None else torch.tensor([encoded[1]], dtype=torch.int32),
            device=self.device,
        )
        return embeds.cpu(), qwen_text_mask(int(count.cpu()[0]), self.seq_len)

    def _encode(self, prompt: str) -> tuple[torch.Tensor, int]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("the text encoder is not loaded on rank 0; call load() first")
        input_ids, valid = tokenize_qwen_prompt(self.tokenizer, prompt, self.enc_seq_len)
        previous = torch.get_num_threads()
        torch.set_num_threads(self.threads)
        try:
            with torch.no_grad():
                hidden = self.model.model(
                    input_ids=input_ids[:, :valid],
                    attention_mask=torch.ones_like(input_ids[:, :valid]),
                    use_cache=False,
                ).last_hidden_state
        finally:
            torch.set_num_threads(previous)
        if hidden.shape[-1] != self.text_dim:
            raise ValueError(
                f"the text encoder's hidden size is {hidden.shape[-1]}, but the DiT expects "
                f"joint_attention_dim={self.text_dim}"
            )
        return qwen_prompt_embeds(hidden, valid, seq_len=self.seq_len)


class _DeviceQwenTextEncoder:
    """Qwen2.5-VL's language model on the device (``NeuronQwenTextEncoderApplication``), called
    like ``_HostQwenTextEncoder``: a prompt in, ``(embeddings, mask)`` out on the host.

    Every rank loads the tokenizer, tokenizes the prompt as the host encoder does
    (``tokenize_qwen_prompt``) and encodes it: the encoder's TP collectives need every rank, and
    every rank then holds the full last hidden state, so nothing is broadcast. The prompt is
    encoded at the static ``(1, enc_seq_len)`` shape, padding included and unmasked (causal
    attention keeps it out of the valid rows), and post-processed by ``qwen_prompt_embeds``:
    the template rows dropped, the valid rows kept and zero-padded to ``seq_len`` in bf16.
    """

    def __init__(self, app, *, model_path, text_dim: int, seq_len: int, device):
        self.app = app
        self.model_path = str(model_path)
        self.text_dim = int(text_dim)
        self.seq_len = int(seq_len)
        self.device = torch.device(device)
        self.tokenizer = None  # every rank, after load()

    def load(self, skip_warmup: bool = False) -> None:
        from difflet.backends.neuron.core.distributed import collective_phase

        with collective_phase("tokenizer load", device=self.device):
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(self.model_path, "tokenizer")
            )
        self.app.load(skip_warmup=skip_warmup)

    def __call__(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("the device text encoder is not loaded; call load() first")
        input_ids, valid = tokenize_qwen_prompt(self.tokenizer, prompt, self.app.enc_seq_len)
        hidden = self.app(input_ids.to(self.device))
        if self.device.type == "neuron":
            torch.neuron.synchronize()
        hidden = hidden.cpu()
        if hidden.shape[-1] != self.text_dim:
            raise ValueError(
                f"the text encoder's hidden size is {hidden.shape[-1]}, but the DiT expects "
                f"joint_attention_dim={self.text_dim}"
            )
        embeds, seq = qwen_prompt_embeds(hidden, valid, seq_len=self.seq_len)
        return embeds, qwen_text_mask(seq, self.seq_len)


class _DeviceQwenVaeDecoder:
    """The orchestrator's VAE: host latents in (already un-normalised), host images out.

    Exposes what ``QwenImageOrchestrator._decode_latents`` reads: ``dtype`` (fp32, the decode
    precision; the orchestrator un-normalises in it) and ``config`` (``latents_mean`` /
    ``latents_std``), and no ``decode`` attribute, so the orchestrator calls it directly with
    ``[B, z_dim, 1, h, w]``. The Wan decoder returns ``[B, 3, 1, H, W]``; the frame axis is
    dropped here, as diffusers' QwenImage pipeline and the Trainium VAE stage drop it.
    """

    def __init__(self, app: NeuronWanVaeDecoderApplication, *, device):
        self.app = app
        self.dtype = app.dtype
        self.config = app.vae_config
        self.device = torch.device(device)

    def load(self, skip_warmup: bool = False) -> None:
        self.app.load(skip_warmup=skip_warmup)

    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        frames = self.app(latents.to(self.dtype).contiguous().to(self.device)).cpu()
        return frames[:, :, 0]


class _OnDeviceTransformer:
    """The orchestrator's DiT callable: a ``QwenImageDiTInputBundle`` (the orchestrator's call)
    or positional tensors in, a host tensor out.

    Only ``hidden_states``, ``timestep`` and ``encoder_hidden_states`` go to the device; the
    mask and guidance are dropped (the DiT attends the zero-padded text unmasked and has no
    guidance embedder). Each input is cast to ``dtype`` and made contiguous on the host before it
    moves to ``device``: a strided host view (the timestep ``_batch_timestep`` expands from a
    0-dim tensor) would otherwise be restrided on the device through a CPU round trip (decision
    D16). ``.dtype`` and ``.config`` are what the orchestrator reads (``_component_dtype``).
    """

    def __init__(self, module: Callable[..., torch.Tensor], config, dtype, device):
        self.module = module
        self.config = config
        self.dtype = dtype
        self.device = torch.device(device)
        #: Called after every forward with its wall seconds; see the module docstring.
        self.forward_hook: Callable[[float], None] | None = None
        self.calls = 0

    def __call__(
        self,
        hidden_states,
        timestep=None,
        encoder_hidden_states=None,
        encoder_hidden_states_mask=None,
        guidance=None,
    ) -> torch.Tensor:
        del encoder_hidden_states_mask, guidance
        if isinstance(hidden_states, QwenImageDiTInputBundle):
            bundle = hidden_states
            hidden_states = bundle.hidden_states
            timestep, encoder_hidden_states = bundle.timestep, bundle.encoder_hidden_states
        if timestep is None or encoder_hidden_states is None:
            raise TypeError("the Qwen-Image DiT needs hidden_states, timestep and "
                            "encoder_hidden_states")
        _check_dit_shapes(self.config, hidden_states, encoder_hidden_states)
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
        if self.forward_hook is not None:
            self.forward_hook(time.perf_counter() - start)
        return out


def _check_dit_shapes(config, hidden_states: torch.Tensor, text: torch.Tensor) -> None:
    """The one static shape: the RoPE tables are precomputed for it, so another shape would fail
    deep in the attention (or, for the text, attend the wrong rows). The batch is 1, or the
    CFG-parallel ``[negative, conditional]`` pair of 2 (``NeuronQwenImageConfig.dit_batch``)."""
    batch = int(getattr(config, "dit_batch", config.batch_size))
    hidden = (batch, int(config.image_seq_len), int(config.in_channels))
    if tuple(hidden_states.shape) != hidden:
        raise ValueError(
            f"Qwen-Image DiT input 'hidden_states' has shape {tuple(hidden_states.shape)}, "
            f"expected {hidden} (the one static shape, {config.height}x{config.width})"
        )
    expected = (batch, int(config.text_seq_len), int(config.joint_attention_dim))
    if tuple(text.shape) != expected:
        raise ValueError(
            f"Qwen-Image DiT input 'encoder_hidden_states' has shape {tuple(text.shape)}, "
            f"expected {expected}: zero-pad the prompt embeddings to text_seq_len="
            f"{config.text_seq_len} (the DiT attends them unmasked)"
        )


def _resolve_shape(shape: dict[str, int | None], shapes) -> dict[str, Any]:
    height = int(shape.get("height") or 1024)
    width = int(shape.get("width") or 1024)
    if shapes:
        shapes = [tuple(entry) for entry in shapes]
        if len(shapes) > 1:
            raise NotImplementedError(
                "the neuron backend compiles one static shape per process (phase 1); "
                f"got shapes={shapes}"
            )
        bucket = (int(shapes[0][0]), int(shapes[0][1]))
        requested = (shape.get("height"), shape.get("width"))
        if any(v is not None for v in requested) and (height, width) != bucket:
            raise ValueError(
                f"the request shape {height}x{width} is not the one compiled shape "
                f"{bucket[0]}x{bucket[1]}"
            )
        height, width = bucket
    return {"height": height, "width": width, "num_frames": None}


def _reject_unsupported(options: dict[str, Any], parallel) -> None:
    teacache = [name for name in _TEACACHE_KWARGS if options[name] is not None]
    if options["teacache_fused"]:
        teacache.append("teacache_fused")
    if teacache:
        raise NotImplementedError(
            f"TeaCache is not supported on the neuron backend in phase 1 ({', '.join(teacache)})"
        )
    if int(options["batch_size"]) != 1:
        raise NotImplementedError(
            f"the neuron Qwen-Image application runs batch_size=1 only (phase 1), got "
            f"batch_size={options['batch_size']}"
        )
    dtype = normalize_dtype(options["text_encoder_dtype"])
    if dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"text_encoder_dtype must be bfloat16 or float32, got {dtype}")
    if options["text_encoder_device"] not in ("host", "neuron"):
        raise ValueError(
            "text_encoder_device must be 'host' or 'neuron', got "
            f"{options['text_encoder_device']!r}"
        )


def _runs_true_cfg(kwargs: dict[str, Any]) -> bool:
    """Whether an orchestrator call runs true CFG: ``true_cfg_scale`` > 1 and negative
    embeddings (``_negative_bundle``'s rule)."""
    scale = kwargs.get("true_cfg_scale")
    return (scale is not None and float(scale) > 1.0
            and kwargs.get("negative_encoder_hidden_states") is not None)


def _flow_match_mu(scheduler_config, image_seq_len: int) -> float:
    """The dynamic-shift ``mu`` the CLI generate stage and diffusers (``calculate_shift``) set
    the FlowMatch schedule with: linear in the packed image sequence length."""
    slope = (scheduler_config.max_shift - scheduler_config.base_shift) / (
        scheduler_config.max_image_seq_len - scheduler_config.base_image_seq_len
    )
    return image_seq_len * slope + (
        scheduler_config.base_shift - slope * scheduler_config.base_image_seq_len
    )


class TorchNeuronQwenImageApplication(torch.nn.Module):
    """Host Qwen2.5-VL on rank 0 + the DiT on each rank's core + ``QwenImageOrchestrator``, on
    every rank.

    Factory signature (``difflet/registry.py`` ``create_application``): ``model_path`` is the
    diffusers snapshot root, ``shape`` the request's ``height``/``width`` (``num_frames`` is
    ignored). Keyword arguments and their defaults are ``_KWARG_DEFAULTS``; ``device`` is
    ``"neuron"`` except in CPU tests (``"cpu"`` on gloo).
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
        _reject_unsupported(options, parallel)
        check_parallel_supported(parallel)
        self.model_path = str(model_path)
        self.parallel = parallel
        self.dtype = _normalize_dtype(dtype)
        self.device = _resolve_device(options["device"])
        self.shape = _resolve_shape(shape, options["shapes"])
        self.text_seq_len = int(options["text_seq_len"])
        self.batch_size = 1
        transformer_path = os.path.join(self.model_path, "transformer")
        cp = int(getattr(parallel, "cp_degree", 1) or 1)
        self.config = NeuronQwenImageConfig.from_pretrained(
            transformer_path,
            height=self.shape["height"],
            width=self.shape["width"],
            text_seq_len=self.text_seq_len,
            batch_size=1,
            tp_degree=int(parallel.tp_degree),
            context_parallel_enabled=cp > 1,
            cp_mode=str(getattr(parallel, "cp_mode", "gather_kv")),
            sp_enabled=bool(getattr(parallel, "sp_enabled", False)),
            cfg_parallel_enabled=bool(getattr(parallel, "cfg_parallel_enabled", False)),
            torch_dtype=self.dtype,
        )
        self.transformer: NeuronQwenImageTransformerApplication | None = None
        if options["enable_transformer"]:
            self.transformer = NeuronQwenImageTransformerApplication(
                model_path=transformer_path,
                config=self.config,
                parallel=parallel,
                dtype=self.dtype,
                exec_mode=options["exec_mode"],
                device=self.device,
            )
            self.config = self.transformer.config  # cp_degree filled from the parallel config
        self.text_encoder: _HostQwenTextEncoder | _DeviceQwenTextEncoder | None = None
        if options["enable_text_encoder"] and options["text_encoder_device"] == "neuron":
            encoder_dtype = normalize_dtype(options["text_encoder_dtype"])
            self._check_device_text_encoder_fits(encoder_dtype)
            self.text_encoder = _DeviceQwenTextEncoder(
                NeuronQwenTextEncoderApplication(
                    model_path=os.path.join(self.model_path, "text_encoder"),
                    parallel=parallel,
                    dtype=encoder_dtype,
                    exec_mode=options["exec_mode"],
                    device=self.device,
                ),
                model_path=self.model_path,
                text_dim=int(self.config.joint_attention_dim),
                seq_len=self.text_seq_len,
                device=self.device,
            )
        elif options["enable_text_encoder"]:
            self.text_encoder = _HostQwenTextEncoder(
                self.model_path,
                text_dim=int(self.config.joint_attention_dim),
                seq_len=self.text_seq_len,
                dtype=normalize_dtype(options["text_encoder_dtype"]),
                device=self.device,
                threads=int(options["encoder_threads"]),
            )
        self.vae_decoder: _DeviceQwenVaeDecoder | None = None
        if options["enable_vae_decoder"]:
            cfg = self.config
            self.vae_decoder = _DeviceQwenVaeDecoder(
                NeuronWanVaeDecoderApplication(
                    model_path=os.path.join(self.model_path, "vae"),
                    latent_shape=(1, int(cfg.out_channels), 1, cfg.latent_height,
                                  cfg.latent_width),
                    parallel=parallel,
                    dtype=torch.float32,  # see NeuronWanVaeDecoderApplication
                    device=self.device,
                ),
                device=self.device,
            )
        self.transformer_adapter: _OnDeviceTransformer | None = None
        self.pipeline = None
        #: Called with no arguments once per denoise iteration; see the module docstring.
        self.step_hook: Callable[[], None] | None = None
        self._text_encoder_load_seconds: float | None = None

    def _check_device_text_encoder_fits(self, encoder_dtype: torch.dtype) -> None:
        """The device text encoder shares each core with this rank's DiT shard. Raise before
        anything loads when the two clearly cannot fit together
        (``difflet/backends/neuron/qwen_image/hbm.py``): at tp2 (CP2 or CFG2 on four cores,
        dp2tp2 replicas) about 6.6 GiB of bf16 encoder beside 19.0 GiB of DiT, over a 24 GiB
        core; at tp4 about 3.3 + 9.5 GiB. The base's load-time check sees one component at a
        time, so it would pass the DiT and refuse the encoder only after the DiT had loaded.
        Like that check, this one runs on the neuron device, or elsewhere when
        ``DIFFLET_NEURON_HBM_GIB`` sets a budget."""
        if not budget_enforced(self.device.type):
            return
        tp = int(self.parallel.tp_degree)
        estimate = device_text_encoder_bytes(
            self.model_path, tp_degree=tp, encoder_dtype_bytes=encoder_dtype.itemsize,
            dit_dtype_bytes=self.dtype.itemsize,
        )
        check_fits(
            estimate,
            what=f"Qwen2.5-VL ({encoder_dtype}) beside the Qwen-Image DiT on one core",
            tp_degree=tp,
            advice="encode on the host (text_encoder_device='host', no --device-text-encoder), "
            "or give the TP group more cores (tp_degree={tp} or more)",
        )

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
        """DiT (runtime init, shards, per-block compile, warm-up), then Qwen2.5-VL on rank 0,
        then the VAE decoder if enabled, then the orchestrator. Every rank must call it; the
        steps are status-synced phases."""
        del compiled_model_path, start_rank_id, local_ranks_size
        if self.pipeline is not None:
            return
        from difflet.models.qwen_image.pipeline import QwenImageOrchestrator

        if self.transformer is not None:
            self.transformer.load(skip_warmup=skip_warmup)
            self.transformer_adapter = _OnDeviceTransformer(
                self.transformer, self.config, self.dtype, self.device
            )
        if isinstance(self.text_encoder, _DeviceQwenTextEncoder):
            self.text_encoder.load(skip_warmup=skip_warmup)
        elif self.text_encoder is not None:
            start = time.perf_counter()
            self.text_encoder.load()
            self._text_encoder_load_seconds = time.perf_counter() - start
        if self.vae_decoder is not None:
            self.vae_decoder.load(skip_warmup=skip_warmup)
        pipeline = QwenImageOrchestrator(
            model_path=self.model_path,
            transformer=self.transformer_adapter,
            vae=self.vae_decoder,
            dtype=self.dtype,
            height=self.shape["height"],
            width=self.shape["width"],
            text_seq_len=self.text_seq_len,
        )
        self._stamp_denoise_iterations(pipeline)
        self.pipeline = pipeline
        logger.info("%s loaded: %s", type(self).__name__,
                    {k: round(v, 2) for k, v in self.phase_seconds.items()})

    def _stamp_denoise_iterations(self, pipeline) -> None:
        """Call ``step_hook`` after each scheduler step: exactly once per denoise iteration."""
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
        """The DiT's load phases, ``"text encoder load"`` (rank 0's read, which the other ranks
        wait out in that phase's status sync) or the device text encoder's as ``"text_encoder
        <phase>"``, and the device VAE's as ``"vae <phase>"``."""
        seconds = dict(self.transformer.phase_seconds) if self.transformer is not None else {}
        if isinstance(self.text_encoder, _DeviceQwenTextEncoder):
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
        if isinstance(self.text_encoder, _DeviceQwenTextEncoder):
            blocks += [f"text_encoder.{name}" for name in self.text_encoder.app.compiled_blocks]
        return blocks

    @property
    def unwarmed_shapes(self) -> list:
        shapes = list(self.transformer.unwarmed_shapes) if self.transformer is not None else []
        if isinstance(self.text_encoder, _DeviceQwenTextEncoder):
            shapes += list(self.text_encoder.app.unwarmed_shapes)
        if self.vae_decoder is not None:
            shapes += list(self.vae_decoder.app.unwarmed_shapes)
        return shapes

    def dit_input_contract(self) -> dict[str, dict[str, Any]]:
        """The Trainium/TPU contract, five tensors; the DiT reads the first three (the mask and
        guidance are accepted and dropped, see the module docstring)."""
        cfg = self.config
        batch = cfg.dit_batch
        return {
            "hidden_states": {
                "shape": (batch, int(cfg.image_seq_len), int(cfg.in_channels)),
                "dtype": self.dtype,
            },
            "timestep": {"shape": (batch,), "dtype": self.dtype},
            "encoder_hidden_states": {
                "shape": (batch, int(cfg.text_seq_len), int(cfg.joint_attention_dim)),
                "dtype": self.dtype,
            },
            "encoder_hidden_states_mask": {
                "shape": (batch, int(cfg.text_seq_len)),
                "dtype": torch.bool,
            },
            "guidance": {"shape": (batch,), "dtype": self.dtype},
        }

    # ------------------------------------------------------------- inference

    def _require_dit(self) -> _OnDeviceTransformer:
        if self.transformer_adapter is None:
            raise RuntimeError(
                f"{type(self).__name__}: the DiT is not loaded (call load() first; "
                "enable_transformer=False never loads one)"
            )
        return self.transformer_adapter

    def forward_dit(self, bundle: QwenImageDiTInputBundle) -> torch.Tensor:
        """One DiT forward on a host bundle; ``[B, image_seq_len, patch^2 * out_channels]``."""
        return self._require_dit()(bundle)

    def encode_prompt(self, prompt: str | list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """``(encoder_hidden_states, encoder_hidden_states_mask)`` for one prompt, identical on
        every rank (rank 0 encodes, see ``_HostQwenTextEncoder``). Every rank must call it."""
        if self.text_encoder is None:
            raise RuntimeError(
                f"{type(self).__name__}: no text encoder (enable_text_encoder=False); pass "
                "encoder_hidden_states and encoder_hidden_states_mask instead of a prompt"
            )
        if self.pipeline is None:
            raise RuntimeError(f"{type(self).__name__} is not loaded; call load() first")
        if isinstance(prompt, (list, tuple)):
            if len(prompt) != 1:
                raise NotImplementedError(
                    f"the neuron Qwen-Image application runs batch 1, got {len(prompt)} prompts"
                )
            prompt = prompt[0]
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a string, got {type(prompt).__name__}")
        return self.text_encoder(prompt)

    def generate(self, *, prompt: str | list[str] | None = None, seed: int | None = None,
                 negative_prompt: str | list[str] | None = None, **kwargs: Any):
        """The orchestrator's ``__call__`` (its keyword arguments: ``encoder_hidden_states``,
        ``encoder_hidden_states_mask``, ``guidance``/``guidance_scale``, ``timesteps``,
        ``num_inference_steps``, ``latents``, ``generator``, ``output_type`` 'latent' | 'pt',
        ...), with three additions:

        * ``prompt``: encoded by ``encode_prompt`` into the two ``encoder_hidden_states*``
          arguments (pass one or the other);
        * ``seed``: a ``torch.Generator`` seeded with it (pass one or ``generator``);
        * ``negative_prompt``: encoded like ``prompt`` into ``negative_encoder_hidden_states*``
          for true classifier-free guidance, which the orchestrator runs when
          ``true_cfg_scale`` > 1 (opt-in; diffusers' QwenImagePipeline semantics). Under CFG
          parallelism a call without true CFG is refused: the DiT then takes only the batch-2
          ``[negative, conditional]`` pair;
        * the schedule: without ``timesteps``, the FlowMatch schedule is set as the CLI generate
          stage sets it (``sigmas = linspace(1, 1/N, N)`` shifted by ``_flow_match_mu`` at this
          shape) and passed as ``timesteps``; the orchestrator alone would set it without
          ``mu``, which Qwen-Image's dynamic-shifting scheduler rejects.

        The initial noise is drawn on rank 0 (from ``generator``, or the global RNG as the CLI's
        ``torch.manual_seed`` leaves it) and broadcast, so the ranks of a TP/CP group denoise
        the same latents even when their global RNGs differ.
        """
        if self.pipeline is None:
            raise RuntimeError(f"{type(self).__name__} is not loaded; call load() first")
        if prompt is not None:
            if kwargs.get("encoder_hidden_states") is not None:
                raise ValueError("pass prompt or encoder_hidden_states, not both")
            embeds, mask = self.encode_prompt(prompt)
            kwargs["encoder_hidden_states"] = embeds
            kwargs["encoder_hidden_states_mask"] = mask
        if negative_prompt is not None:
            if kwargs.get("negative_encoder_hidden_states") is not None:
                raise ValueError("pass negative_prompt or negative_encoder_hidden_states, not both")
            embeds, mask = self.encode_prompt(negative_prompt)
            kwargs["negative_encoder_hidden_states"] = embeds
            kwargs["negative_encoder_hidden_states_mask"] = mask
        if self.config.cfg_parallel_enabled and not _runs_true_cfg(kwargs):
            raise ValueError(
                "cfg_parallel_enabled splits the two branches of true classifier-free guidance "
                "over the CFG group; pass true_cfg_scale > 1 and a negative_prompt (or "
                "negative_encoder_hidden_states), or run without CFG parallelism"
            )
        if seed is not None:
            if kwargs.get("generator") is not None:
                raise ValueError("pass seed or generator, not both")
            kwargs["generator"] = torch.Generator().manual_seed(int(seed))
        if kwargs.get("bundle") is None and kwargs.get("latents") is None:
            kwargs["latents"] = self._initial_latents(kwargs)
        scheduler = self.pipeline.scheduler
        if kwargs.get("timesteps") is None and scheduler is not None:
            import numpy as np

            steps = max(int(kwargs.get("num_inference_steps") or _DEFAULT_STEPS), 1)
            kwargs["num_inference_steps"] = steps
            sigmas = np.linspace(1.0, 1.0 / steps, steps).tolist()
            scheduler.set_timesteps(
                sigmas=sigmas,
                mu=_flow_match_mu(scheduler.config, int(self.config.image_seq_len)),
                device="cpu",
            )
            kwargs["timesteps"] = scheduler.timesteps
        return self.pipeline(**kwargs)

    def _initial_latents(self, kwargs: dict[str, Any]) -> torch.Tensor:
        """Rank 0's packed initial noise, broadcast (see ``generate``)."""
        pipeline = self.pipeline

        def draw() -> torch.Tensor:
            return pipeline.prepare_latents(
                batch_size=int(kwargs.get("batch_size", 1)),
                channels=int(kwargs.get("channels", 16)),
                dtype=pipeline.dtype,
                generator=kwargs.get("generator"),
            )

        if world_info()[1] == 1:
            return draw()
        latents = rank0_call(draw, device=self.device, what="initial latents")
        return broadcast_tensor(latents, device=self.device).cpu()

    def __call__(self, *args: Any, **kwargs: Any):
        """``app(bundle)``, ``app(latents, timestep, text[, mask, guidance])`` or
        ``app(hidden_states=..., timestep=..., encoder_hidden_states=..., ...)`` run the DiT;
        ``app("prompt", ...)`` and every keyword-only call run ``generate`` (the orchestrator).
        """
        if len(args) == 1 and isinstance(args[0], QwenImageDiTInputBundle) and not kwargs:
            return self.forward_dit(args[0])
        if len(args) >= 3:
            return self._require_dit()(*args, **kwargs)
        if not args and _DIT_KEYS <= kwargs.keys():
            return self._require_dit()(**kwargs)
        if len(args) == 1 and isinstance(args[0], (str, list)):
            if "prompt" in kwargs:
                raise TypeError("prompt given both positionally and as a keyword")
            kwargs["prompt"] = args[0]
            args = ()
        if args:
            raise TypeError(
                f"{type(self).__name__}: positional arguments are a DiT call (a bundle or three "
                f"tensors) or a prompt; got {len(args)}"
            )
        return self.generate(**kwargs)


__all__ = [
    "TorchNeuronQwenImageApplication",
    "qwen_prompt_embeds",
    "qwen_text_mask",
    "tokenize_qwen_prompt",
]
