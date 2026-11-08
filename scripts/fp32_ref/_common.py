"""What phase G's CPU fp32 reference scripts (``scripts/fp32_ref/<model>.py``) share: the
device's initial-noise law, the md5 each script asserts its noise with, the thread count, and
saving a run the way the device run saved it, so ``scripts/fp32_ref/score.py`` (through
``scripts/compare_difflet_runs.py``) reads the reference exactly as it reads a device run.

The reference differs from the device only in dtype (fp32) and device (the host CPU); a starting
noise that differed by one bf16 rounding (rel L2 1.7e-3) would be scored as device error after
50 chaotic steps, and an output encoded differently would measure the encoder.
"""

from __future__ import annotations

import hashlib
import json
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
    the versions and threads that made it)."""
    out_dir = Path(out_dir)
    if isinstance(output, Image.Image):
        name = "out.png"
    else:
        if fps is None:
            raise ValueError("a video output needs the device's fps (DEVICE_FPS)")
        frames = np.asarray(output)
        _check_frames(frames)
        name = "out.mp4"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(latents, out_dir / "latents.pt")
    if name == "out.png":
        output.save(out_dir / name)
    else:
        export_to_video(list(frames.astype(np.float32, copy=False)), str(out_dir / name), fps=fps)
    record = {"settings": settings, "noise_md5": noise_md5, "timings": timings, "output": name,
              "fps": fps, "threads": torch.get_num_threads(), "torch": torch.__version__,
              "diffusers": diffusers.__version__}
    (out_dir / "run.json").write_text(json.dumps(record, indent=1) + "\n")
