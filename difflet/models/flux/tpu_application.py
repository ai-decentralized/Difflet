"""FLUX.1-dev on the TPU backend: the DiT and the VAE on the chips, the text encoders on the host.

Fifth model on the TPU backend, option (b) of
docs/plans/2026-09-11-tpu-port-hunyuan-ltx2-flux.md: diffusers'
``FluxTransformer2DModel`` TP-sharded per rank (``models/flux/tp_sharding.py``,
``backends/tpu/flux/transformer.py``) — the Trainium Flux code (the legacy NxDI
fork under ``modeling_flux.py``) is not touched.

Placement follows the other ports:

* T5-XXL (4.7B) runs on the host in fp32 on XLA ordinal 0 only and its hidden
  state is broadcast (``TpuBroadcastT5Encoder``); one fp32 copy is ~19 GB, and
  bf16 is emulated on the EPYC host (Qwen measured 4.5x slower);
* CLIP-L (123M) runs on the host on every rank — the pooled projection is
  cheaper to recompute than to broadcast in lockstep;
* the 2D VAE (84M) is resident on the chip of the primary replica
  (``TpuDeviceImageVae``): the DiT shard leaves ~9.7 GB of HBM free, and a
  1024x1024 decode fits comfortably (Qwen had ~6 GB free and had to park its
  VAE on the host between requests);
* the denoise loop is device-resident like Qwen's ``_tpu_denoise_loop``: every
  per-step scalar is a device tensor so XLA compiles one graph per step kind,
  and the probe-free TeaCache controller's residual stays on the chip.
"""

from __future__ import annotations

import logging
import os
import time
from types import SimpleNamespace
from typing import Any

import torch

logger = logging.getLogger(__name__)

FLUX_DEFAULT_HEIGHT = 1024
FLUX_DEFAULT_WIDTH = 1024
FLUX_TEXT_SEQ_LEN = 512  # diffusers' max_sequence_length, difflet's MAX_SEQUENCE_LENGTH
FLUX_CLIP_SEQ_LEN = 77
FLUX_VAE_SCALE_FACTOR = 8

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


def _is_primary_replica() -> bool:
    return int(os.environ.get("DIFFLET_REPLICA_RANK", "0")) == 0


# ------------------------------------------------------------------ geometry


def packed_latent_grid(height: int, width: int) -> tuple[int, int]:
    """(latent_height, latent_width) before 2x2 packing, as ``prepare_latents`` sizes them."""
    return (
        2 * (int(height) // (FLUX_VAE_SCALE_FACTOR * 2)),
        2 * (int(width) // (FLUX_VAE_SCALE_FACTOR * 2)),
    )


def image_seq_len(height: int, width: int) -> int:
    lh, lw = packed_latent_grid(height, width)
    return (lh // 2) * (lw // 2)


def flux_sigmas_and_mu(scheduler_config, num_inference_steps: int, seq_len: int):
    """``(sigmas, mu)`` exactly as ``FluxPipeline.__call__`` derives them."""
    import numpy as np
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift

    sigmas = np.linspace(1.0, 1.0 / int(num_inference_steps), int(num_inference_steps))
    get = scheduler_config.get if hasattr(scheduler_config, "get") else lambda k, d: getattr(scheduler_config, k, d)
    mu = calculate_shift(
        int(seq_len),
        get("base_image_seq_len", 256),
        get("max_image_seq_len", 4096),
        get("base_shift", 0.5),
        get("max_shift", 1.15),
    )
    return sigmas, float(mu)


def prepare_packed_latents(height: int, width: int, *, in_channels_latent: int, generator):
    """Seed-compatible with diffusers: the noise is drawn in the (B, C, H, W) latent
    shape and then packed, so the same seed gives the same starting point."""
    from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
    from diffusers.utils.torch_utils import randn_tensor

    lh, lw = packed_latent_grid(height, width)
    noise = randn_tensor((1, int(in_channels_latent), lh, lw), generator=generator, dtype=torch.float32)
    packed = FluxPipeline._pack_latents(noise, 1, int(in_channels_latent), lh, lw)
    img_ids = FluxPipeline._prepare_latent_image_ids(1, lh // 2, lw // 2, torch.device("cpu"), torch.float32)
    return packed, img_ids


def unpack_latents(packed, height: int, width: int):
    from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

    return FluxPipeline._unpack_latents(packed, int(height), int(width), FLUX_VAE_SCALE_FACTOR)


def image_tensor_to_pil(image):
    """``VaeImageProcessor.postprocess(output_type="pil")`` for one (1, 3, H, W) decode."""
    from PIL import Image

    arr = (image[0].float() / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((arr * 255).round().astype("uint8"))


def device_euler_loop(forward, latents, timesteps, deltas, *, controller, mark_step):
    """Device-resident flow-matching Euler loop with optional TeaCache skipping.

    ``forward(latents, timestep)`` returns the velocity for one step. Same
    structure and reasoning as Qwen's ``_tpu_denoise_loop``: latents and every
    per-step scalar live on the device, the update is ``x + (s_next - s) * v``
    in fp32, and the controller's residual stays lazy on the chip so a skipped
    step is one elementwise op.
    """
    steps = len(timesteps)
    if controller is not None:
        from difflet.pipeline.teacache import sync_probe_free_num_steps

        sync_probe_free_num_steps(controller, steps)
        controller.reset()
    for index in range(steps):
        if controller is not None and controller.should_skip(index, None):
            velocity = controller.skip_noise_pred()
        else:
            velocity = forward(latents, timesteps[index]).to(torch.float32)
            if controller is not None:
                controller.record_full_step(velocity)
        latents = latents + deltas[index] * velocity
        mark_step()
    return latents


# ------------------------------------------------------------ host encoders


class TpuBroadcastT5Encoder:
    """T5-XXL prompt embeds computed on XLA ordinal 0 and broadcast to every rank.

    Mirrors diffusers' ``_get_t5_prompt_embeds``: ``tokenizer_2`` at
    ``padding="max_length"`` / ``truncation=True`` over ``seq_len`` tokens, then
    ``text_encoder_2(ids)[0]``. Every rank calls this in lockstep — the
    collective must be reached by all of them, so the encoder rank never raises
    before it (failures travel as a status code and re-raise everywhere).
    """

    def __init__(self, model_path: str, *, seq_len: int, dtype: torch.dtype,
                 encoder_dtype: torch.dtype = torch.float32) -> None:
        from transformers import AutoConfig

        self.seq_len = int(seq_len)
        self.dtype = dtype
        self.is_encoder = _is_encoder_rank()
        self.model = None
        self.tokenizer = None
        config = AutoConfig.from_pretrained(os.path.join(model_path, "text_encoder_2"))
        self.hidden_size = int(config.d_model)
        if self.is_encoder:
            from transformers import AutoTokenizer, T5EncoderModel

            self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer_2"))
            self.model = T5EncoderModel.from_pretrained(
                os.path.join(model_path, "text_encoder_2"), dtype=encoder_dtype
            ).eval()
            self.model.requires_grad_(False)

    def encode_local(self, prompt: str) -> torch.Tensor:
        ids = self.tokenizer(
            [prompt], padding="max_length", max_length=self.seq_len, truncation=True,
            return_overflowing_tokens=False, return_length=False, return_tensors="pt",
        ).input_ids
        with torch.no_grad():
            return self.model(ids, output_hidden_states=False)[0]

    def __call__(self, prompt: str) -> torch.Tensor:
        import torch_xla
        import torch_xla.core.xla_model as xm

        device = torch_xla.device()
        embeds = torch.zeros(1, self.seq_len, self.hidden_size, dtype=self.dtype)
        status = _ENCODE_OK
        if self.is_encoder:
            try:
                embeds = self.encode_local(prompt).to(self.dtype)
            except Exception:  # noqa: BLE001 - re-raised below on every rank
                logger.exception("flux.tpu_t5_encode_failed")
                status = _ENCODE_FAILED
                embeds = torch.zeros(1, self.seq_len, self.hidden_size, dtype=self.dtype)
        payload = [embeds.to(device), torch.tensor([status], dtype=torch.int32).to(device)]
        xm.collective_broadcast(payload, root_ordinal=0)
        xm.mark_step()
        if int(payload[1].cpu()[0]) != _ENCODE_OK:
            raise RuntimeError("Flux T5 prompt encoding failed on the encoder rank")
        return payload[0].cpu()


class HostClipPooler:
    """CLIP-L pooled projection on the host (``_get_clip_prompt_embeds``)."""

    def __init__(self, model_path: str, *, dtype: torch.dtype, encoder_dtype: torch.dtype = torch.float32):
        from transformers import AutoTokenizer, CLIPTextModel

        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))
        self.model = CLIPTextModel.from_pretrained(
            os.path.join(model_path, "text_encoder"), dtype=encoder_dtype
        ).eval()
        self.model.requires_grad_(False)

    def __call__(self, prompt: str) -> torch.Tensor:
        ids = self.tokenizer(
            [prompt], padding="max_length", max_length=FLUX_CLIP_SEQ_LEN, truncation=True,
            return_overflowing_tokens=False, return_length=False, return_tensors="pt",
        ).input_ids
        with torch.no_grad():
            pooled = self.model(ids, output_hidden_states=False).pooler_output
        return pooled.to(self.dtype)


class TpuDeviceImageVae:
    """Flux's ``AutoencoderKL`` decoder on the chip, with the scaling/shift applied on device."""

    def __init__(self, vae, dtype: torch.dtype):
        import torch_xla

        self.config = vae.config
        self.dtype = dtype
        self.scaling_factor = float(vae.config.scaling_factor)
        self.shift_factor = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
        self._device = torch_xla.device()
        self._vae = vae.to(dtype).to(self._device).eval()

    def decode(self, latents):
        """(1, C, H, W) *scaled* latents (as the DiT produces them) -> (1, 3, 8H, 8W) fp32 host."""
        import torch_xla.core.xla_model as xm

        z = latents.to(torch.float32).to(self._device)
        z = (z / self.scaling_factor + self.shift_factor).to(self.dtype)
        with torch.no_grad():
            out = self._vae.decode(z, return_dict=False)[0]
        xm.mark_step()
        return out.float().cpu()


# -------------------------------------------------------------- application


class TpuFluxApplication(torch.nn.Module):
    """One rank of FLUX on TPU; ``load_eager`` then call like a diffusers pipeline."""

    def __init__(self, *, model_path: str, parallel, dtype: Any, shape: dict[str, int | None],
                 **kwargs: Any) -> None:
        super().__init__()
        from difflet.backends.tpu.flux.config import TpuFluxConfig
        from difflet.backends.tpu.flux.transformer import TpuFluxTransformerApplication

        self.model_path = model_path
        self.parallel = parallel
        self.dtype = _normalize_dtype(dtype)
        self.shape = {
            "height": int(shape.get("height") or FLUX_DEFAULT_HEIGHT),
            "width": int(shape.get("width") or FLUX_DEFAULT_WIDTH),
        }
        self.kwargs = kwargs
        self.text_seq_len = int(kwargs.get("text_seq_len", FLUX_TEXT_SEQ_LEN))
        self.batch_size = int(kwargs.get("batch_size", 1))
        self.transformer_path = os.path.join(model_path, "transformer")
        self.transformer = None
        self.config = None
        self.t5 = None
        self.clip = None
        self.vae = None
        self.scheduler = None
        self.teacache = None
        self.teacache_last_stats: dict[str, Any] | None = None
        self.last_timings: dict[str, float] = {}
        self._device_module = None

        if not os.path.exists(os.path.join(self.transformer_path, "config.json")):
            return
        self.config = TpuFluxConfig.from_pretrained(
            self.transformer_path,
            height=self.shape["height"],
            width=self.shape["width"],
            batch_size=self.batch_size,
            text_seq_len=self.text_seq_len,
            tp_degree=int(parallel.tp_degree),
            torch_dtype=self.dtype,
            context_parallel_enabled=int(getattr(parallel, "cp_degree", 1)) > 1,
            cfg_parallel_enabled=bool(getattr(parallel, "cfg_parallel_enabled", False)),
            sp_enabled=bool(getattr(parallel, "sp_enabled", False)),
        )
        self.transformer = TpuFluxTransformerApplication(
            model_path=self.transformer_path, config=self.config
        )

    # --------------------------------------------------------------- lifecycle

    def _require_transformer(self, action: str):
        if self.transformer is None:
            raise NotImplementedError(
                f"Flux {action} requires transformer/config.json under {self.model_path!r}"
            )
        return self.transformer

    def _build_teacache(self):
        cadence = self.kwargs.get("teacache_cadence")
        alpha = self.kwargs.get("teacache_online_delta_alpha")
        if cadence is None and alpha is None:
            return None
        from difflet.pipeline.teacache import build_probe_free_controller

        return build_probe_free_controller(
            model="flux",
            shape_label=f"{self.shape['height']}x{self.shape['width']}",
            cadence=cadence,
            online_delta_alpha=alpha,
        )

    def load_eager(self) -> None:
        """DiT to the chip (gated warmup compile), then the host encoders, the
        scheduler and — on the primary replica — the VAE on the chip."""
        import torch_xla
        import torch_xla.core.xla_model as xm
        from diffusers import FlowMatchEulerDiscreteScheduler

        transformer = self._require_transformer("load")
        device = torch_xla.device()
        mark = time.monotonic()
        module = transformer._prepare_module().to(device)
        transformer.module = module
        xm.mark_step()
        xm.wait_device_ops()
        self.last_timings["dit_load_s"] = time.monotonic() - mark
        seconds = transformer.warmup_eager(module, device)
        self.last_timings["warmup_s"] = seconds
        print(f"[flux] tpu warmup (first compile) in {seconds:.1f}s", flush=True)
        self._device_module = module

        if bool(self.kwargs.get("enable_host_pipeline", True)):
            mark = time.monotonic()
            self.t5 = TpuBroadcastT5Encoder(self.model_path, seq_len=self.text_seq_len, dtype=self.dtype)
            self.clip = HostClipPooler(self.model_path, dtype=self.dtype)
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                os.path.join(self.model_path, "scheduler")
            )
            if _is_primary_replica() and bool(self.kwargs.get("device_vae", True)):
                from diffusers import AutoencoderKL

                vae = AutoencoderKL.from_pretrained(os.path.join(self.model_path, "vae"), torch_dtype=self.dtype)
                self.vae = TpuDeviceImageVae(vae, self.dtype)
                print("[flux] VAE resident on the chip (primary replica)", flush=True)
            self.last_timings["host_load_s"] = time.monotonic() - mark
            print(
                f"[flux] host encoders loaded in {self.last_timings['host_load_s']:.1f}s "
                f"(T5 on this rank: {self.t5.model is not None})",
                flush=True,
            )
        self.teacache = self._build_teacache()

    # ---------------------------------------------------------------- contract

    def dit_input_contract(self) -> dict[str, dict[str, Any]]:
        cfg = self._require_transformer("input contract").config
        b = int(cfg.batch_size)
        return {
            "hidden_states": {"shape": (b, cfg.image_seq_len, int(cfg.in_channels)), "dtype": self.dtype},
            "encoder_hidden_states": {
                "shape": (b, int(cfg.text_seq_len), int(cfg.joint_attention_dim)), "dtype": self.dtype,
            },
            "pooled_projections": {"shape": (b, int(cfg.pooled_projection_dim)), "dtype": self.dtype},
            "timestep": {"shape": (b,), "dtype": self.dtype},
            "img_ids": {"shape": (cfg.image_seq_len, 3), "dtype": torch.float32},
            "txt_ids": {"shape": (int(cfg.text_seq_len), 3), "dtype": torch.float32},
            "guidance": {"shape": (b,), "dtype": self.dtype},
        }

    # ----------------------------------------------------------------- stages

    def encode_prompt(self, prompt: str) -> dict[str, torch.Tensor]:
        if self.t5 is None or self.clip is None:
            raise RuntimeError("Flux TPU application: call load_eager() first")
        return {
            "encoder_hidden_states": self.t5(prompt),
            "pooled_projections": self.clip(prompt),
            "txt_ids": torch.zeros(self.text_seq_len, 3, dtype=torch.float32),
        }

    def forward_dit(self, **inputs) -> torch.Tensor:
        """One DiT forward with host tensors in and out (oracle / harness surface)."""
        import torch_xla
        import torch_xla.core.xla_model as xm

        transformer = self._require_transformer("forward")
        device = torch_xla.device()
        order = ("hidden_states", "encoder_hidden_states", "pooled_projections", "timestep",
                 "img_ids", "txt_ids", "guidance")
        args = []
        for name in order:
            value = inputs[name]
            if value is not None and name not in {"img_ids", "txt_ids"}:
                value = value.to(self.dtype)
            args.append(None if value is None else value.to(device))
        with torch.no_grad():
            out = transformer.forward(*args)
        xm.mark_step()
        return out.cpu()

    def denoise(self, text: dict[str, torch.Tensor], *, num_inference_steps: int, guidance_scale: float,
                generator, height: int | None = None, width: int | None = None) -> torch.Tensor:
        """Packed fp32 latents on the host after the device-resident loop."""
        import torch_xla
        import torch_xla.core.xla_model as xm

        transformer = self._require_transformer("denoise")
        if self.scheduler is None:
            raise RuntimeError("Flux TPU application: call load_eager() first")
        height = int(height or self.shape["height"])
        width = int(width or self.shape["width"])
        if (height, width) != (self.shape["height"], self.shape["width"]):
            raise ValueError(
                f"Flux TPU application is loaded for {self.shape['height']}x{self.shape['width']}, "
                f"got {height}x{width}"
            )
        cfg = transformer.config
        device = torch_xla.device()
        packed, img_ids = prepare_packed_latents(
            height, width, in_channels_latent=int(cfg.in_channels) // 4, generator=generator
        )
        seq = int(packed.shape[1])
        sigmas_np, mu = flux_sigmas_and_mu(self.scheduler.config, num_inference_steps, seq)
        self.scheduler.set_timesteps(sigmas=sigmas_np.tolist(), mu=mu, device="cpu")
        sigmas = self.scheduler.sigmas.to(torch.float32)
        steps = len(self.scheduler.timesteps)
        # Every per-step scalar as a DEVICE tensor (a Python float would be
        # constant-folded and recompile every step; see Qwen's _denoise_tpu).
        deltas = [(sigmas[i + 1] - sigmas[i]).reshape(1).to(device) for i in range(steps)]
        timesteps = [(t / 1000).to(self.dtype).reshape(1).to(device) for t in self.scheduler.timesteps]
        latents = packed.to(device)
        states = text["encoder_hidden_states"].to(self.dtype).to(device)
        pooled = text["pooled_projections"].to(self.dtype).to(device)
        txt_ids = text["txt_ids"].to(torch.float32).to(device)
        img_ids = img_ids.to(device)
        guidance = None
        if bool(getattr(cfg, "guidance_embeds", True)):
            guidance = torch.full([1], float(guidance_scale), dtype=torch.float32).to(self.dtype).to(device)

        def forward(x, t):
            return transformer.forward(x.to(self.dtype), states, pooled, t, img_ids, txt_ids, guidance)

        with torch.no_grad():
            latents = device_euler_loop(
                forward, latents, timesteps, deltas, controller=self.teacache, mark_step=xm.mark_step
            )
        out = latents.cpu()
        if self.teacache is not None:
            self.teacache_last_stats = self.teacache.stats()
            # print, not logger.info: the serving worker has no logging handler.
            print(f"[teacache] stats: {self.teacache_last_stats}", flush=True)
        return out

    def decode(self, packed: torch.Tensor, *, height: int, width: int):
        if self.vae is None:
            raise RuntimeError("Flux VAE is not resident on this replica")
        latents = unpack_latents(packed.to(torch.float32), height, width)
        return image_tensor_to_pil(self.vae.decode(latents))

    # ------------------------------------------------------------------- call

    def __call__(self, prompt: str, num_inference_steps: int = 28, height: int | None = None,
                 width: int | None = None, guidance_scale: float = 3.5, generator=None,
                 teacache_enabled: bool = True, output_type: str = "pil", **ignored: Any):
        """diffusers-shaped call returning ``.images`` (PIL) or ``.latents`` (packed fp32).

        ``teacache_enabled`` is the adaptive-TeaCache switch of the Trainium
        pipeline; on TPU only the probe-free controller exists and it is
        configured at load time, so the flag is accepted and ignored.
        """
        del teacache_enabled, ignored
        height = int(height or self.shape["height"])
        width = int(width or self.shape["width"])
        if generator is None:
            generator = torch.Generator("cpu").manual_seed(0)
        timings: dict[str, float] = {}
        mark = time.monotonic()
        text = self.encode_prompt(prompt)
        timings["encode_s"] = time.monotonic() - mark
        mark = time.monotonic()
        packed = self.denoise(
            text, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
            generator=generator, height=height, width=width,
        )
        timings["denoise_s"] = time.monotonic() - mark
        self.last_timings.update(timings)
        if output_type == "latent":
            return SimpleNamespace(latents=packed, images=None)
        if self.vae is None:
            # Non-primary replicas: the engine keeps the primary's output only,
            # so hand back a correctly sized placeholder instead of decoding.
            from PIL import Image

            return SimpleNamespace(images=[Image.new("RGB", (width, height))], latents=packed)
        mark = time.monotonic()
        image = self.decode(packed, height=height, width=width)
        self.last_timings["decode_s"] = time.monotonic() - mark
        return SimpleNamespace(images=[image], latents=packed)


__all__ = [
    "FLUX_CLIP_SEQ_LEN",
    "FLUX_DEFAULT_HEIGHT",
    "FLUX_DEFAULT_WIDTH",
    "FLUX_TEXT_SEQ_LEN",
    "HostClipPooler",
    "TpuBroadcastT5Encoder",
    "TpuDeviceImageVae",
    "TpuFluxApplication",
    "device_euler_loop",
    "flux_sigmas_and_mu",
    "image_seq_len",
    "image_tensor_to_pil",
    "packed_latent_grid",
    "prepare_packed_latents",
    "unpack_latents",
]
