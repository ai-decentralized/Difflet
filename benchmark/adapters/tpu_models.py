"""Per-model drivers behind ``benchmark/adapters/tpu.py``.

One driver per model type. Each drives the objects ``difflet serve`` loads on
the TPU backend -- the same application factories, orchestrators, host
encoders and VAEs -- so the benchmark measures what a served request costs,
not a hand-rolled loop. The per-model runners this replaces
(``benchmark/{wan,hunyuan,ltx2,flux}_tpu_run.py``) each re-implemented a slice
of that wiring and none of them ran the trn2 cold/warm protocol; folding them
here is what puts every v5e row through the one harness path every trn2 row
went through.

A driver has four phases, called by the worker in ``tpu.py``:

    load()          -> ordered {stage: seconds}, per process. XLA's
                       first-execution compile lands here where the model's
                       load_eager warms up (HunyuanVideo, LTX-2, FLUX) and in
                       the first request otherwise (Qwen-Image, Wan).
    encode(prompt)  -> text conditioning; runs on every rank (collectives).
    denoise(text)   -> latents on the host; every rank.
    decode(latents) -> (tensor, layout, value_range); primary replica only,
                       exactly where serving decodes.

torch / torch_xla are imported inside methods: the module is imported by the
spawned worker after ``pjrt.initialize_multiprocess``, and the pure helpers at
the bottom are unit-tested without a chip.
"""

from __future__ import annotations

import io
import time
from collections import OrderedDict
from typing import Any

# --------------------------------------------------------------------------- #
# Timing hooks
# --------------------------------------------------------------------------- #


class _TimedCallable:
    """Call passthrough that stamps the step timer after every DiT call.

    Attribute reads (``dtype``, ``config``, ...) go to the wrapped object; the
    orchestrators read those off their transformer.
    """

    def __init__(self, inner, timer):
        self._inner = inner
        self._timer = timer

    def __call__(self, *args, **kwargs):
        out = self._inner(*args, **kwargs)
        self._timer.step()
        return out

    def __getattr__(self, item):
        return getattr(self._inner, item)


def _timed_method(fn, timer):
    def timed(*args, **kwargs):
        out = fn(*args, **kwargs)
        timer.step()
        return out

    return timed


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #


class _Driver:
    output_kind = "video"
    fps = 24

    def __init__(self, payload: dict, rank: int, world: int, timer) -> None:
        self.payload = payload
        self.rank = rank
        self.world = world
        self.timer = timer
        # Serving keeps the primary replica's output only and decodes there
        # alone (``_is_primary_replica`` in difflet/serving/models/*): rank 0.
        self.primary = rank == 0
        self.model_dir = payload["model_dir"]
        self.height = int(payload["height"])
        self.width = int(payload["width"])
        self.num_frames = payload.get("num_frames")
        self.steps = int(payload["steps"])
        self.seed = int(payload["seed"])
        self.guidance = float(payload["guidance_scale"])

    # -- shared pieces ----------------------------------------------------- #
    def _parallel(self):
        from difflet.pipeline.parallel_config import DiffletParallelConfig

        return DiffletParallelConfig(tp_degree=self.world)

    def _teacache_kwargs(self) -> dict[str, Any]:
        return {
            "teacache_cadence": self.payload.get("teacache_cadence"),
            "teacache_online_delta_alpha": self.payload.get("teacache_online_delta"),
        }

    def _generator(self):
        import torch

        return torch.Generator().manual_seed(self.seed)

    def _shape(self) -> dict[str, int | None]:
        return {"height": self.height, "width": self.width, "num_frames": self.num_frames}

    # -- contract ---------------------------------------------------------- #
    def load(self) -> "OrderedDict[str, float]":
        raise NotImplementedError

    def encode(self, prompt: str):
        raise NotImplementedError

    def denoise(self, text):
        raise NotImplementedError

    def decode(self, latents):
        raise NotImplementedError

    def teacache_stats(self):
        return None


class QwenImageDriver(_Driver):
    """``QwenImageServingStageAdapter`` -- the three serving stages, as before."""

    output_kind = "image"

    def load(self):
        from types import SimpleNamespace

        from difflet.serving.orchestrators.qwen_image import QwenImageServingStageAdapter
        from difflet.serving.types import ParallelTopology

        profile = SimpleNamespace(
            height=self.height, width=self.width, num_frames=None,
            parallel=ParallelTopology(tp_degree=self.world, cp_degree=1, world_size=self.world),
            world_size=self.world, teacache_speedup=None, teacache_calibration_data=None,
            teacache_cadence=self.payload.get("teacache_cadence"),
            teacache_online_delta=self.payload.get("teacache_online_delta"),
            shape_dict=lambda: {"height": self.height, "width": self.width, "num_frames": None},
        )
        adapter = QwenImageServingStageAdapter()
        adapter._tpu = True
        adapter.model_dir = self.model_dir
        adapter.active_profile = profile
        stages: "OrderedDict[str, float]" = OrderedDict()
        for name, fn in (
            ("text_encoder (Qwen2.5-VL 7B, host fp32, ordinal 0)", adapter._load_text_stage),
            ("transformer (DiT -> chips)", adapter._load_denoiser_stage),
            ("vae (host; moved to the chip per decode)", adapter._load_vae_stage),
        ):
            mark = time.monotonic()
            fn(profile)
            stages[name] = time.monotonic() - mark
        adapter._tpu_module = _TimedCallable(adapter._tpu_module, self.timer)
        self.adapter = adapter
        return stages

    def _request(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            prompt=self.payload["prompt"], num_inference_steps=self.steps, seed=self.seed,
            guidance_scale=self.guidance, request_id="bench", height=self.height, width=self.width,
        )

    def encode(self, prompt):
        return self.adapter._encode_prompt(prompt)

    def denoise(self, text):
        return self.adapter._denoise(text, self._request())

    def decode(self, latents):
        png = self.adapter._decode(latents, self._request())
        return _png_to_tensor(png), "BCHW", "zero_to_one"

    def teacache_stats(self):
        return getattr(self.adapter, "_tpu_teacache_last_stats", None)


class FluxDriver(_Driver):
    """``TpuFluxApplication``: T5 on ordinal 0 + broadcast, CLIP everywhere,
    DiT sharded, VAE on the primary's chip, device-resident Euler loop."""

    output_kind = "image"

    def load(self):
        import torch

        from difflet.models.flux.entry import create_flux_application

        app = create_flux_application(
            model_path=self.model_dir, parallel=self._parallel(), dtype=torch.bfloat16,
            shape={"height": self.height, "width": self.width, "num_frames": None},
            backend="tpu", **self._teacache_kwargs(),
        )
        mark = time.monotonic()
        app.load_eager()
        total = time.monotonic() - mark
        timings = dict(app.last_timings)
        stages: "OrderedDict[str, float]" = OrderedDict()
        stages["transformer (DiT -> chips)"] = float(timings.get("dit_load_s", 0.0))
        stages["transformer XLA first-execution compile (warmup)"] = float(timings.get("warmup_s", 0.0))
        stages["text_encoder_t5 (host fp32, ordinal 0) + text_encoder_clip + vae (primary chip)"] = float(
            timings.get("host_load_s", 0.0)
        )
        rest = total - sum(stages.values())
        if rest > 0.5:
            stages["load_eager residual"] = rest
        app.transformer.forward = _timed_method(app.transformer.forward, self.timer)
        self.app = app
        return stages

    def encode(self, prompt):
        return self.app.encode_prompt(prompt)

    def denoise(self, text):
        return self.app.denoise(
            text, num_inference_steps=self.steps, guidance_scale=self.guidance,
            generator=self._generator(), height=self.height, width=self.width,
        )

    def decode(self, latents):
        image = self.app.decode(latents, height=self.height, width=self.width)
        return _pil_to_tensor(image), "BCHW", "zero_to_one"

    def teacache_stats(self):
        return self.app.teacache_last_stats


class WanDriver(_Driver):
    """``TpuWanApplication.load_eager`` (single expert, umT5 on ordinal 0 +
    broadcast) and the serving adapter's host fp32 VAE decode."""

    fps = 16

    def load(self):
        import torch

        from difflet.models.wan.entry import create_wan_application
        from difflet.serving.models.wan import _application_kwargs, _load_host_vae

        app = create_wan_application(
            model_path=self.model_dir, parallel=self._parallel(), dtype=torch.bfloat16,
            shape=self._shape(), backend="tpu", **_application_kwargs(), **self._teacache_kwargs(),
        )
        stages: "OrderedDict[str, float]" = OrderedDict()
        mark = time.monotonic()
        app.load_eager()
        stages["transformer (DiT -> chips) + text_encoder (umT5, host fp32, ordinal 0)"] = (
            time.monotonic() - mark
        )
        self.vae = None
        if self.primary:
            mark = time.monotonic()
            self.vae = _load_host_vae(self.model_dir)
            stages["vae (host fp32, primary replica)"] = time.monotonic() - mark
        orchestrator = app.pipeline
        orchestrator.transformer = _TimedCallable(orchestrator.transformer, self.timer)
        self.app, self.orchestrator = app, orchestrator
        return stages

    def encode(self, prompt):
        # WanPromptEncoderStageRunner: a negative prompt only when guidance > 1.
        return self.orchestrator.encode_prompt(prompt=prompt)

    def denoise(self, text):
        out = self.orchestrator(
            prompt_embeds=text, num_inference_steps=self.steps, guidance_scale=self.guidance,
            generator=self._generator(), output_type="latent",
        )
        return out.latents

    def decode(self, latents):
        import torch

        # WanHostDecoderStageRunner, verbatim.
        vae = self.vae
        z_dim = int(vae.config.z_dim)
        x = latents.detach().to(device="cpu", dtype=torch.float32)
        mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1)
        inverse_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1)
        x = x / inverse_std + mean
        with torch.no_grad():
            frames = vae.decode(x, return_dict=False)[0]
        return frames.to(torch.float32).clamp(-1.0, 1.0), "BCTHW", "minus_one_to_one"

    def teacache_stats(self):
        return getattr(self.orchestrator, "_teacache_last_stats", None)


class HunyuanVideoDriver(_Driver):
    """``TpuHunyuanVideoApplication.load_eager`` plus the serving adapter's host
    stages: CLIP-L, Llama-3 on ordinal 0 + broadcast, fp32 tiled VAE."""

    fps = 24

    def load(self):
        import torch

        from difflet.models.hunyuan_video.entry import create_hunyuan_video_application
        from difflet.models.hunyuan_video.tpu_application import TpuBroadcastLlamaEncoder
        from difflet.serving.models import hunyuan_video as serving

        self.serving = serving
        app = create_hunyuan_video_application(
            model_path=self.model_dir, parallel=self._parallel(), dtype=torch.bfloat16,
            shape=self._shape(), backend="tpu", text_seq_len=serving._TEXT_SEQ_LEN,
            **self._teacache_kwargs(),
        )
        stages: "OrderedDict[str, float]" = OrderedDict()
        mark = time.monotonic()
        app.load_eager()
        stages["transformer (DiT -> chips + XLA first-execution compile)"] = time.monotonic() - mark
        mark = time.monotonic()
        self.clip_tokenizer, self.clip = serving._load_host_clip(self.model_dir)
        stages["text_encoder_2 (CLIP-L, host fp32)"] = time.monotonic() - mark
        mark = time.monotonic()
        self.llama = TpuBroadcastLlamaEncoder(
            self.model_dir, seq_len=serving._LLAMA_SEQ_LEN,
            capture_layer=int(serving._LLAMA_CAPTURE.rsplit(".", 1)[1]), dtype=torch.bfloat16,
        )
        self.llama_tokenizer = serving._load_llama_tokenizer(self.model_dir)
        stages["text_encoder (Llama-3 8B, host fp32, ordinal 0)"] = time.monotonic() - mark
        self.vae = None
        if self.primary:
            mark = time.monotonic()
            self.vae = serving._load_host_vae(self.model_dir)
            stages["vae (host fp32 tiled, primary replica)"] = time.monotonic() - mark
        orchestrator = app.pipeline
        orchestrator.transformer = _TimedCallable(orchestrator.transformer, self.timer)
        self.app, self.orchestrator = app, orchestrator
        return stages

    def encode(self, prompt):
        import torch

        serving = self.serving
        # HunyuanVideoHostClipStageRunner
        clip_in = self.clip_tokenizer(
            prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            clip_out = self.clip(input_ids=clip_in.input_ids, attention_mask=clip_in.attention_mask)
        pooled = clip_out.pooler_output.to(torch.bfloat16).cpu().reshape(1, -1)
        # HunyuanVideoLlamaStageRunner
        tokenized = self.llama_tokenizer(
            serving._LLAMA_TEMPLATE.format(prompt), max_length=serving._LLAMA_SEQ_LEN,
            padding="max_length", truncation=True, return_tensors="pt", return_attention_mask=True,
        )
        out = self.llama(
            input_ids=tokenized.input_ids, attention_mask=tokenized.attention_mask,
            position_ids=torch.arange(serving._LLAMA_SEQ_LEN, dtype=torch.int32).unsqueeze(0),
            sampling_params=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        )
        hidden = out.captured_tensors[0][:, serving._LLAMA_CROP_START:].to(torch.bfloat16).cpu()
        mask = tokenized.attention_mask[:, serving._LLAMA_CROP_START:].to(torch.int64)
        return {"encoder_hidden_states": hidden, "encoder_attention_mask": mask,
                "pooled_projections": pooled}

    def denoise(self, text):
        import torch

        from difflet.models.hunyuan_video.application import HunyuanVideoDiTInputBundle

        # HunyuanVideoDenoiserStageRunner, verbatim.
        latent_frames = (int(self.num_frames) - 1) // 4 + 1
        noise = torch.randn(
            1, 16, latent_frames, self.height // 8, self.width // 8,
            dtype=torch.bfloat16, generator=self._generator(),
        )
        bundle = HunyuanVideoDiTInputBundle(
            hidden_states=noise, timestep=torch.zeros(1, dtype=torch.bfloat16),
            encoder_hidden_states=text["encoder_hidden_states"],
            encoder_attention_mask=text["encoder_attention_mask"],
            pooled_projections=text["pooled_projections"],
            guidance=torch.full([1], self.guidance * 1000.0, dtype=torch.bfloat16),
        )
        out = self.orchestrator(
            bundle=bundle, num_inference_steps=self.steps, output_type="latent",
            return_trajectory=False,
        )
        return out.latents

    def decode(self, latents):
        import torch

        # HunyuanVideoHostDecoderStageRunner, verbatim.
        vae = self.vae
        scaling = float(getattr(vae.config, "scaling_factor", 1.0))
        x = latents.detach().to(device="cpu", dtype=torch.float32)
        with torch.no_grad():
            frames = vae.decode(x / scaling, return_dict=False)[0]
        return frames.to(torch.float32).clamp(-1.0, 1.0), "BCTHW", "minus_one_to_one"

    def teacache_stats(self):
        controller = getattr(self.orchestrator, "teacache_controller", None)
        return controller.stats() if controller is not None else None


class LTX2Driver(_Driver):
    """``TpuLTX2Application.load_eager``: Gemma-3 on ordinal 0 + broadcast,
    sharded DiT, video VAE on the primary's chip, audio VAE + vocoder host."""

    fps = 24

    def load(self):
        import torch

        from difflet.models.ltx_2.entry import create_ltx_2_application

        app = create_ltx_2_application(
            model_path=self.model_dir, parallel=self._parallel(), dtype=torch.bfloat16,
            shape=self._shape(), backend="tpu", enable_host_pipeline=True,
            **self._teacache_kwargs(),
        )
        stages: "OrderedDict[str, float]" = OrderedDict()
        mark = time.monotonic()
        app.load_eager()
        stages[
            "transformer (DiT -> chips + XLA first-execution compile) + Gemma-3 12B "
            "(host fp32, ordinal 0) + connectors + VAEs (video VAE on the primary chip)"
        ] = time.monotonic() - mark
        app.forward_dit = _timed_method(app.forward_dit, self.timer)
        self.app, self.orchestrator = app, app.pipeline
        return stages

    def encode(self, prompt):
        orchestrator = self.orchestrator
        return orchestrator.prepare_conditioning(
            prompt=prompt, guidance_scale=self.guidance,
            max_sequence_length=orchestrator.text_seq_len,
        )

    def denoise(self, text):
        states, audio_states, mask, audio_mask = text
        out = self.orchestrator(
            encoder_hidden_states=states, audio_encoder_hidden_states=audio_states,
            encoder_attention_mask=mask, audio_encoder_attention_mask=audio_mask,
            num_inference_steps=self.steps, guidance_scale=self.guidance,
            generator=self._generator(), output_type="latent",
        )
        self._audio_latents = out.audio_latents
        return out.latents

    def decode(self, latents):
        import torch

        video, _audio = self.orchestrator._decode_latents(latents, self._audio_latents)
        # postprocess_video(output_type="pt"): BFCHW in [0, 1].
        return video.to(torch.float32), "BFCHW", "zero_to_one"

    def teacache_stats(self):
        controller = getattr(self.orchestrator, "_teacache_controller", None)
        return controller.stats() if controller is not None else None


DRIVERS: dict[str, type[_Driver]] = {
    "qwen_image": QwenImageDriver,
    "flux": FluxDriver,
    "wan": WanDriver,
    "hunyuan_video": HunyuanVideoDriver,
    "ltx_2": LTX2Driver,
}


def make_driver(payload: dict, rank: int, world: int, timer) -> _Driver:
    model_type = payload["model_type"]
    if model_type not in DRIVERS:
        raise NotImplementedError(
            f"no TPU benchmark driver for model_type {model_type!r}; known: {sorted(DRIVERS)}"
        )
    return DRIVERS[model_type](payload, rank, world, timer)


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested without a chip)
# --------------------------------------------------------------------------- #


def output_info(tensor, note: str = "") -> dict:
    """The harness ``OutputInfo`` block for a tensor: shape / dtype / finite /
    range -- the same fields the trn2 adapter fills from its saved ``.pt``."""
    import torch

    x = tensor.detach().float().cpu()
    return {
        "shape": list(x.shape), "dtype": str(tensor.dtype),
        "finite": bool(torch.isfinite(x).all()),
        "min": float(x.min()), "max": float(x.max()),
        "mean": float(x.mean()), "std": float(x.std()), "note": note,
    }


def frames_uint8(tensor, layout: str, value_range: str):
    """Decoded model output -> ``[F, H, W, 3]`` uint8 frames.

    Layouts are the ones ``difflet/serving/video_media.py`` accepts from the
    adapters (``BCTHW`` for Wan/HunyuanVideo, ``BFCHW`` for LTX-2) plus
    ``BCHW`` for the image models.
    """
    import torch

    x = tensor.detach().float().cpu()
    if layout == "BCTHW":
        x = x[0].permute(1, 2, 3, 0)
    elif layout == "BFCHW":
        x = x[0].permute(0, 2, 3, 1)
    elif layout == "BCHW":
        x = x[0].permute(1, 2, 0).unsqueeze(0)
    else:
        raise ValueError(f"unknown layout {layout!r}")
    if value_range == "minus_one_to_one":
        x = (x.clamp(-1.0, 1.0) + 1.0) / 2.0
    elif value_range == "zero_to_one":
        x = x.clamp(0.0, 1.0)
    else:
        raise ValueError(f"unknown value_range {value_range!r}")
    return (x * 255.0).round().to(torch.uint8).numpy()


def save_media(frames, stem: str, *, kind: str, fps: int = 24) -> list[str]:
    """Write ``<stem>.png`` for an image, ``<stem>.mp4`` plus a first/middle/last
    contact sheet for a video. Returns the paths written."""
    from PIL import Image

    written: list[str] = []
    if kind == "image" or frames.shape[0] == 1:
        path = f"{stem}.png"
        Image.fromarray(frames[0]).save(path)
        written.append(path)
        return written
    try:
        import imageio

        path = f"{stem}.mp4"
        imageio.mimsave(path, list(frames), fps=fps)
        written.append(path)
    except Exception as exc:  # noqa: BLE001 - the contact sheet still lands
        print(f"[bench] mp4 write skipped: {exc}", flush=True)
    picks = [0, frames.shape[0] // 2, frames.shape[0] - 1]
    tiles = [Image.fromarray(frames[i]) for i in picks]
    width, height = tiles[0].size
    scale = min(1.0, 512.0 / width)
    tile_w, tile_h = int(width * scale), int(height * scale)
    sheet = Image.new("RGB", (tile_w * len(tiles), tile_h))
    for n, tile in enumerate(tiles):
        sheet.paste(tile.resize((tile_w, tile_h)), (n * tile_w, 0))
    path = f"{stem}_frames_{'_'.join(str(i) for i in picks)}.png"
    sheet.save(path)
    written.append(path)
    return written


def _pil_to_tensor(image):
    import numpy as np
    import torch

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def _png_to_tensor(png_bytes: bytes):
    from PIL import Image

    with Image.open(io.BytesIO(png_bytes)) as image:
        return _pil_to_tensor(image)


__all__ = [
    "DRIVERS", "make_driver", "output_info", "frames_uint8", "save_media",
]
