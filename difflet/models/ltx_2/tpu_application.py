"""LTX-2 on the TPU backend: the audiovisual DiT on the chips, the rest on the host.

Fourth model on the TPU backend. Structure follows ``NeuronLTX2Application``:
the application *is* the orchestrator's ``transformer`` (it takes an
``LTX2DiTInputBundle`` and returns ``(video, audio)``), and the diffusers
``LTX2Pipeline`` with ``transformer=None`` supplies the host pieces — Gemma-3
text encoder, connectors, video/audio VAEs, vocoder — exactly as on Trainium.

Host memory shapes the one TPU-specific choice: Gemma-3 12B is loaded on XLA
ordinal 0 only (fp32, ~48 GB; bf16 is emulated and ~4x slower on the EPYC
host) and its packed hidden states are broadcast to the other ranks, the
same trick Wan uses for umT5 and HunyuanVideo for Llama. Four copies would
not fit beside four DiT ranks on a 188 GB box.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import torch

from difflet.models.ltx_2.application import (
    LTX_2_DEFAULT_HEIGHT,
    LTX_2_DEFAULT_NUM_FRAMES,
    LTX_2_DEFAULT_TEXT_SEQ_LEN,
    LTX_2_DEFAULT_WIDTH,
    LTX2DiTInputBundle,
    validate_ltx_2_dit_inputs,
)

logger = logging.getLogger(__name__)

_ENCODE_OK = 0
_ENCODE_FAILED = 1


def _normalize_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    text = str(dtype).lower().removeprefix("torch.")
    if text in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if text in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def _is_encoder_rank() -> bool:
    import torch_xla.runtime as xr

    return int(xr.global_ordinal()) == 0


def load_tpu_host_pipeline(model_path: str, *, dtype: torch.dtype, encoder_dtype=torch.float32):
    """diffusers ``LTX2Pipeline`` minus the transformer, text encoder on ordinal 0 only.

    ``_get_gemma_prompt_embeds`` is replaced by a broadcast wrapper so every
    rank's ``encode_prompt`` returns the same packed embeddings without every
    rank holding Gemma. The connectors, VAEs and vocoder load on every rank
    (they are small and the decode runs on the primary replica anyway).
    """
    from difflet.models.ltx_2.pipeline import disable_ltx_2_xla_lazy_import

    disable_ltx_2_xla_lazy_import()
    from diffusers import LTX2Pipeline

    encoder = None
    is_encoder = _is_encoder_rank()
    if is_encoder:
        from transformers import Gemma3ForConditionalGeneration

        encoder = Gemma3ForConditionalGeneration.from_pretrained(
            os.path.join(model_path, "text_encoder"), dtype=encoder_dtype
        ).eval()
        encoder.requires_grad_(False)
    # fp32 on the host, whatever the DiT runs in: bf16 is emulated on the
    # EPYC host, and measured in the worker the bf16 audio VAE decode took
    # 114 s and the vocoder 196 s where fp32 takes 0.1 s and 0.6 s. The
    # orchestrator casts what it hands the DiT to `dtype` itself.
    pipe = LTX2Pipeline.from_pretrained(
        model_path, torch_dtype=torch.float32, transformer=None, text_encoder=encoder
    )
    pipe.to("cpu")
    pipe.set_progress_bar_config(disable=True)
    # The wire/return dtype follows the host pipeline (fp32), not the DiT.
    _install_broadcast_prompt_encoder(
        pipe, is_encoder=is_encoder, dtype=torch.float32, packed_width=packed_text_dim(model_path)
    )
    return pipe


def _install_broadcast_prompt_encoder(
    pipe, *, is_encoder: bool, dtype: torch.dtype, packed_width: int
) -> None:
    original = pipe._get_gemma_prompt_embeds
    packed_dim = packed_width
    pipe_dtype = dtype

    # Same signature diffusers' LTX2Pipeline uses when it calls
    # _get_gemma_prompt_embeds(..., device=..., dtype=...) from encode_prompt.
    def _encode(prompt, num_videos_per_prompt=1, max_sequence_length=1024, scale_factor=8,
                device=None, dtype=None):
        import torch_xla
        import torch_xla.core.xla_model as xm

        wire_dtype = pipe_dtype  # what crosses the interconnect; cast to `dtype` after
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        batch = len(prompts) * int(num_videos_per_prompt)
        xla = torch_xla.device()
        status = _ENCODE_OK
        if is_encoder:
            try:
                embeds, mask = original(
                    prompt, num_videos_per_prompt=num_videos_per_prompt,
                    max_sequence_length=max_sequence_length, scale_factor=scale_factor,
                    device=torch.device("cpu"), dtype=wire_dtype,
                )
                embeds = embeds.to(wire_dtype)
                mask = mask.to(torch.int32)
            except Exception:  # noqa: BLE001 - re-raised below on every rank
                logger.exception("ltx_2.tpu_prompt_encode_failed")
                status = _ENCODE_FAILED
                embeds = torch.zeros(batch, max_sequence_length, packed_dim, dtype=wire_dtype)
                mask = torch.zeros(batch, max_sequence_length, dtype=torch.int32)
        else:
            embeds = torch.zeros(batch, max_sequence_length, packed_dim, dtype=wire_dtype)
            mask = torch.zeros(batch, max_sequence_length, dtype=torch.int32)
        payload = [embeds.to(xla), mask.to(xla), torch.tensor([status], dtype=torch.int32).to(xla)]
        xm.collective_broadcast(payload, root_ordinal=0)
        xm.mark_step()
        if int(payload[2].cpu()[0]) != _ENCODE_OK:
            raise RuntimeError("LTX-2 prompt encoding failed on the encoder rank")
        # Returned in the host pipeline's dtype, not the caller's: the fp32
        # connectors consume these next (F.linear rejects a bf16 input on fp32
        # weights), and the orchestrator casts the connector output to the DiT
        # dtype when it builds the bundle.
        del dtype
        return payload[0].cpu(), payload[1].cpu().to(torch.int64)

    pipe._get_gemma_prompt_embeds = _encode


def packed_text_dim(model_path: str) -> int:
    """hidden_size x (num_hidden_layers + 1) of the Gemma-3 text encoder."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(os.path.join(model_path, "text_encoder"))
    tc = getattr(config, "text_config", config)
    return int(tc.hidden_size) * (int(tc.num_hidden_layers) + 1)


class TpuDeviceVideoVae:
    """LTX-2's video VAE on the chip, behind the host-side call the orchestrator makes.

    ``_decode_latents`` calls ``vae.decode(latents, timestep, return_dict=False)`` with
    host tensors and reads ``vae.config`` / ``vae.dtype``; this moves the call onto the
    device and the frames back. Measured on a v5e: 121 frames at 512x768 decode in
    0.7 s (29 s first compile) at 4.0 GB HBM peak, where the fp32 host decode did
    not finish in 14 minutes -- the reason LTX-2's serving smoke timed out twice.
    Only the primary replica decodes (the others return latents), so only it holds
    the 2.4 GB of VAE weights next to its DiT shard.
    """

    def __init__(self, vae, dtype: torch.dtype):
        import torch_xla

        self.config = vae.config
        self.dtype = dtype
        self._device = torch_xla.device()
        # The host pipeline is fp32 (see load_tpu_host_pipeline); on the chip
        # the VAE runs in the DiT's dtype.
        self._vae = vae.to(dtype).to(self._device).eval()

    def decode(self, latents, timestep=None, return_dict=False):
        import torch_xla.core.xla_model as xm

        with torch.no_grad():
            out = self._vae.decode(
                latents.to(self.dtype).to(self._device),
                None if timestep is None else timestep.to(self.dtype).to(self._device),
                return_dict=False,
            )[0]
        xm.mark_step()
        frames = out.float().cpu()
        return (frames,) if not return_dict else frames


def _is_primary_replica() -> bool:
    return int(os.environ.get("DIFFLET_REPLICA_RANK", "0")) == 0


class TpuLTX2Application(torch.nn.Module):
    supports_ltx_2_extra_kwargs = False

    def __init__(
        self,
        *,
        model_path: str,
        parallel,
        dtype: Any,
        shape: dict[str, int | None],
        **kwargs: Any,
    ) -> None:
        super().__init__()
        from difflet.backends.tpu.ltx_2.config import TpuLTX2Config
        from difflet.backends.tpu.ltx_2.transformer import TpuLTX2TransformerApplication

        self.model_path = model_path
        self.parallel = parallel
        self.dtype = _normalize_dtype(dtype)
        self.shape = {
            "height": int(shape.get("height") or LTX_2_DEFAULT_HEIGHT),
            "width": int(shape.get("width") or LTX_2_DEFAULT_WIDTH),
            "num_frames": int(shape.get("num_frames") or LTX_2_DEFAULT_NUM_FRAMES),
        }
        self.kwargs = kwargs
        self.text_seq_len = int(kwargs.get("text_seq_len", LTX_2_DEFAULT_TEXT_SEQ_LEN))
        self.audio_text_seq_len = int(kwargs.get("audio_text_seq_len", self.text_seq_len))
        self.audio_num_frames = kwargs.get("audio_num_frames")
        self.frame_rate = float(kwargs.get("frame_rate", 24.0))
        self.batch_size = int(kwargs.get("batch_size", 1))
        self.cfg_parallel_enabled = False
        self.transformer_path = os.path.join(model_path, "transformer")
        self.transformer = None
        self.host_pipeline = None
        self.pipeline = None
        self._device_module = None

        if not os.path.exists(os.path.join(self.transformer_path, "config.json")):
            return
        self.config = TpuLTX2Config.from_pretrained(
            self.transformer_path,
            height=self.shape["height"],
            width=self.shape["width"],
            num_frames=self.shape["num_frames"],
            batch_size=self.batch_size,
            text_seq_len=self.text_seq_len,
            audio_text_seq_len=self.audio_text_seq_len,
            audio_num_frames=self.audio_num_frames,
            frame_rate=self.frame_rate,
            tp_degree=int(parallel.tp_degree),
            torch_dtype=self.dtype,
            context_parallel_enabled=int(getattr(parallel, "cp_degree", 1)) > 1,
            cfg_parallel_enabled=bool(getattr(parallel, "cfg_parallel_enabled", False)),
            sp_enabled=bool(getattr(parallel, "sp_enabled", False)),
        )
        self.transformer = TpuLTX2TransformerApplication(
            model_path=self.transformer_path, config=self.config
        )

    # --------------------------------------------------------------- lifecycle

    def _require_transformer(self, action: str):
        if self.transformer is None:
            raise NotImplementedError(
                f"LTX-2 {action} requires transformer/config.json under {self.model_path!r}"
            )
        return self.transformer

    def load_eager(self) -> None:
        """DiT to the chips (with the gated warmup compile), then the host pipeline
        and the backend-neutral orchestrator — the path ``difflet serve`` uses."""
        import torch_xla
        import torch_xla.core.xla_model as xm

        from difflet.models.ltx_2.pipeline import LTX2Orchestrator

        transformer = self._require_transformer("load")
        device = torch_xla.device()
        module = transformer._prepare_module().to(device)
        transformer.module = module
        xm.mark_step()
        xm.wait_device_ops()
        seconds = transformer.warmup_eager(module, device)
        print(f"[ltx_2] tpu warmup (first compile) in {seconds:.1f}s", flush=True)
        self._device_module = module

        if bool(self.kwargs.get("enable_host_pipeline", True)):
            mark = time.monotonic()
            self.host_pipeline = load_tpu_host_pipeline(self.model_path, dtype=self.dtype)
            print(
                f"[ltx_2] host pipeline loaded in {time.monotonic() - mark:.1f}s "
                f"(text encoder on this rank: {self.host_pipeline.text_encoder is not None})",
                flush=True,
            )
        pipe = self.host_pipeline
        vae = getattr(pipe, "vae", None)
        if vae is not None and _is_primary_replica() and bool(self.kwargs.get("device_vae", True)):
            vae = TpuDeviceVideoVae(vae, self.dtype)
            print("[ltx_2] video VAE resident on the chip (primary replica)", flush=True)
        self.pipeline = LTX2Orchestrator(
            model_path=self.model_path,
            transformer=self,
            vae=vae,
            audio_vae=getattr(pipe, "audio_vae", None),
            vocoder=getattr(pipe, "vocoder", None),
            video_processor=getattr(pipe, "video_processor", None),
            host_pipeline=pipe,
            dtype=self.dtype,
            height=self.shape["height"],
            width=self.shape["width"],
            num_frames=self.shape["num_frames"],
            text_seq_len=self.text_seq_len,
            audio_text_seq_len=self.audio_text_seq_len,
            audio_num_frames=self.audio_num_frames,
            frame_rate=self.frame_rate,
            teacache_cadence=self.kwargs.get("teacache_cadence"),
            teacache_online_delta_alpha=self.kwargs.get("teacache_online_delta_alpha"),
        )

    def dit_input_contract(self) -> dict[str, dict[str, Any]]:
        cfg = self._require_transformer("input contract").config
        b = int(cfg.batch_size)
        return {
            "hidden_states": {"shape": (b, cfg.video_seq_len, int(cfg.in_channels)), "dtype": self.dtype},
            "audio_hidden_states": {
                "shape": (b, cfg.audio_seq_len, int(cfg.audio_in_channels)), "dtype": self.dtype,
            },
            "encoder_hidden_states": {
                "shape": (b, int(cfg.text_seq_len), int(cfg.video_text_dim)), "dtype": self.dtype,
            },
            "audio_encoder_hidden_states": {
                "shape": (b, int(cfg.audio_text_seq_len), int(cfg.audio_text_dim)), "dtype": self.dtype,
            },
            "timestep": {"shape": (b,), "dtype": self.dtype},
            "sigma": {"shape": (b,), "dtype": self.dtype},
            "encoder_attention_mask": {"shape": (b, int(cfg.text_seq_len)), "dtype": torch.bool},
            "audio_encoder_attention_mask": {
                "shape": (b, int(cfg.audio_text_seq_len)), "dtype": torch.bool,
            },
            "video_coords": {"shape": (b, 3, cfg.video_seq_len, 2), "dtype": torch.float32},
            "audio_coords": {"shape": (b, 1, cfg.audio_seq_len, 2), "dtype": torch.float32},
        }

    # ----------------------------------------------------------------- forward

    def forward_dit(self, bundle: LTX2DiTInputBundle):
        import torch_xla
        import torch_xla.core.xla_model as xm

        transformer = self._require_transformer("forward")
        validate_ltx_2_dit_inputs(bundle, config=transformer.config, dtype=self.dtype)
        device = torch_xla.device()
        with torch.no_grad():
            video, audio = transformer.forward(*[t.to(device) for t in bundle.as_model_inputs()])
        xm.mark_step()
        return video.cpu(), audio.cpu()

    def __call__(self, *args: Any, **kwargs: Any):
        if len(args) == 1 and isinstance(args[0], LTX2DiTInputBundle):
            return self.forward_dit(args[0])
        direct_keys = {
            "hidden_states", "audio_hidden_states", "encoder_hidden_states",
            "audio_encoder_hidden_states", "timestep", "sigma", "encoder_attention_mask",
            "audio_encoder_attention_mask", "video_coords", "audio_coords",
        }
        if not args and direct_keys.issubset(kwargs):
            return self.forward_dit(LTX2DiTInputBundle(**kwargs))
        if self.pipeline is not None and self.pipeline.has_runtime_components():
            return self.pipeline(*args, **kwargs)
        raise NotImplementedError("LTX-2 TPU application: call load_eager() first")


__all__ = ["TpuDeviceVideoVae", "TpuLTX2Application", "load_tpu_host_pipeline", "packed_text_dim"]
