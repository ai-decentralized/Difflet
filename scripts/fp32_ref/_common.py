"""What phase G's CPU fp32 reference scripts (``scripts/fp32_ref/<model>.py``) share: the
device's initial-noise law, the md5 each script asserts its noise with, the thread count, the
guard that keeps the math fp32, and saving a run the way the device run saved it, so
``scripts/fp32_ref/score.py`` (through ``scripts/compare_difflet_runs.py``) reads the reference
exactly as it reads a device run.

The reference differs from the device only in dtype (fp32) and device (the host CPU); a starting
noise that differed by one bf16 rounding (rel L2 1.7e-3) would be scored as device error after
50 chaotic steps, and an output encoded differently would measure the encoder.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import diffusers
import numpy as np
import torch
from diffusers.utils import export_to_video
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

#: The fps the device's video orchestrators encode at, by score.py's model key: wan.py's
#: ``_save_video`` default (both Wan versions), ltx_2.py's, hunyuan_video.py's and
#: hunyuan_video_15.py's ``_FPS``. Only the MP4 container reads it, but a clip encoded at another
#: rate is another file.
DEVICE_FPS = {"ltx_2": 24, "wan21": 16, "wan22": 16, "hunyuan_video": 24,
              "hunyuan_video_15": 24}


def device_noise(shape, seed: int, *, bf16_draw: bool, generator_device="cpu") -> torch.Tensor:
    """The device run's initial noise as fp32: diffusers' ``randn_tensor`` from a generator on
    ``generator_device`` seeded ``seed``, in bf16 when ``bf16_draw``. FLUX, Qwen-Image, LTX-2 and
    HunyuanVideo-1.5 draw in their bf16 application dtype on the host (the fp32 draw rounded
    to bf16, the research checked), so the reference upcasts that draw, which is exact; Wan and
    HunyuanVideo draw in fp32, as diffusers' fp32 pipeline does."""
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    dtype = torch.bfloat16 if bf16_draw else torch.float32
    draw = randn_tensor(tuple(shape), generator=generator, device=torch.device("cpu"), dtype=dtype)
    return draw.float()


def tensor_md5(t: torch.Tensor) -> str:
    """md5 of the tensor's bytes in its own dtype and C order: the convention of the research's
    noise md5s (a bf16 draw is hashed as bf16, so its upcast is hashed after ``.to(bf16)``)."""
    data = t.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.md5(data.numpy().tobytes()).hexdigest()


def set_threads(n: int) -> None:
    """torch's intra-op threads, to the vCPUs of the one NUMA node the caller pinned the process
    to with taskset (two references share the host, one per node)."""
    if n < 1:
        raise ValueError(f"threads must be at least 1, not {n}")
    torch.set_num_threads(n)


#: The variables oneDNN takes its default fp32 math mode from (either prefix, any case).
FPMATH_MODE_ENV = ("ONEDNN_DEFAULT_FPMATH_MODE", "DNNL_DEFAULT_FPMATH_MODE")


def require_fp32_math() -> None:
    """Refuse fp32 matmuls or convs allowed to run in reduced precision, which on this AMX-BF16
    host would make the reference silently not fp32: torch's oneDNN fp32_precision 'bf16' or
    'tf32' ('none' inherits, 'ieee' is fp32; float32_matmul_precision 'medium' and 'high' set
    mkldnn.matmul to 'bf16' and 'tf32', so they are refused too), or a oneDNN default fpmath
    mode other than strict, which torch leaves to every primitive it gives no mode (bf16 takes
    an fp32 conv here from 2e-7 to 2e-3 rel error: the VAEs' convs). Each script's main calls
    it first, so a refused run has loaded nothing."""
    mkldnn = torch.backends.mkldnn
    found = {"torch.backends": torch.backends.fp32_precision, "mkldnn": mkldnn.fp32_precision,
             "mkldnn.matmul": mkldnn.matmul.fp32_precision,
             "mkldnn.conv": mkldnn.conv.fp32_precision, "mkldnn.rnn": mkldnn.rnn.fp32_precision}
    reduced = {name: value for name, value in found.items() if value not in ("none", "ieee")}
    reduced |= {name: os.environ[name] for name in FPMATH_MODE_ENV
                if os.environ.get(name, "strict").lower() != "strict"}
    if reduced:
        raise RuntimeError(f"fp32 math runs in reduced precision: {reduced}")


def _check_frames(frames: np.ndarray) -> None:
    """Refuse a clip ``export_to_video`` would encode as something else: it multiplies float
    frames by 255 and truncates, so uint8 frames wrap and [-1, 1] frames (a raw VAE decode) clip;
    a channel-first or batched array is not a list of HxWx3 frames."""
    if frames.dtype.kind != "f":
        raise ValueError(f"video frames must be float in [0, 1], not {frames.dtype}")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"video frames must be (F, H, W, 3), not {frames.shape}")
    if frames.min() < 0.0 or frames.max() > 1.0:
        raise ValueError(f"video frames must be in [0, 1], not [{frames.min()}, {frames.max()}]")


def save_run(out_dir, *, latents, output, settings: dict, noise_md5: str, timings: dict,
             fps=None) -> None:
    """Write a reference run into ``out_dir`` as score.py reads it: ``latents.pt`` (the final
    latents as given: the device's layout, fp32), the decoded output (a PIL image to ``out.png``;
    a clip's float [0, 1] frames (F, H, W, 3) to ``out.mp4`` at ``fps`` with diffusers'
    ``export_to_video``, the writer every difflet video orchestrator encodes with, libx264 at
    imageio quality 5) and ``run.json`` (the settings, the injected noise's md5, the timings and
    the versions and threads that made it). A clip's frames are also kept lossless in
    ``out.pt``, (1, F, 3, H, W) float [0, 1], the layout ``compare_difflet_runs`` reads a ``.pt``
    clip in: a codec-free comparison (device latents decoded on the host) then needs no rerun of
    a reference that took hours."""
    out_dir = Path(out_dir)
    lossless = None
    if isinstance(output, Image.Image):
        name = "out.png"
    else:
        if fps is None:
            raise ValueError("a video output needs the device's fps (DEVICE_FPS)")
        frames = np.asarray(output)
        _check_frames(frames)
        name, lossless = "out.mp4", "out.pt"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(latents, out_dir / "latents.pt")
    if name == "out.png":
        output.save(out_dir / name)
    else:
        frames = frames.astype(np.float32, copy=False)
        export_to_video(list(frames), str(out_dir / name), fps=fps)
        clip = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].contiguous()
        torch.save(clip, out_dir / lossless)
    record = {"settings": settings, "noise_md5": noise_md5, "timings": timings, "output": name,
              "lossless": lossless, "fps": fps, "threads": torch.get_num_threads(),
              "torch": torch.__version__, "diffusers": diffusers.__version__}
    (out_dir / "run.json").write_text(json.dumps(record, indent=1) + "\n")
