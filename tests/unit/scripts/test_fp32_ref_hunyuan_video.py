"""Phase G's HunyuanVideo reference (``scripts/fp32_ref/hunyuan_video.py``) against the device runs
it is scored with: its settings against the orchestrator's constants and the two arms' command
lines, its decode tiles against the device decoder's, its noise against the device's fp32 draw,
its latents layout and clip format against the device files, and its frames through ``save_run``
against the device's own writer, byte for byte. No model is loaded."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
from pathlib import Path

import imageio.v3 as iio
import pytest
import torch

_DIR = Path(__file__).resolve().parents[3] / "scripts" / "fp32_ref"
_RUNS = Path("/home/ubuntu/fp8-bench/hv1_tp4")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"fp32_ref_{name}", _DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = _load("hunyuan_video")
common = _load("_common")
score = _load("score")


def _gen() -> torch.Generator:
    return torch.Generator().manual_seed(42)


def test_settings_are_the_orchestrators():
    from difflet.cli.orchestrators import hunyuan_video as orch

    assert (ref.HEIGHT, ref.WIDTH, ref.NUM_FRAMES) == orch._DEFAULT_SHAPE
    assert ref.GUIDANCE_SCALE == orch._DEFAULT_GUIDANCE
    assert ref.TEXT_SEQ_LEN == orch._TEXT_SEQ_LEN
    assert ref.FPS == orch._FPS == common.DEVICE_FPS["hunyuan_video"]
    assert (ref.STEPS, ref.SEED, ref.TRUE_CFG_SCALE) == (50, 42, 1.0)


@pytest.mark.parametrize("script", ["run_bf16_arms.sh", "run_loo.sh"])
def test_settings_are_the_device_command_lines(script):
    """Both arms' launch scripts (bf16; V15, the shipped MXFP8 default): the model, revision,
    shape, steps, guidance, seed and prompt the reference copies, and no negative prompt."""
    path = _RUNS / script
    if not path.is_file():
        pytest.skip(f"no device run script {path} on this host")
    text = path.read_text()
    for words in (f"--model-id {ref.MODEL_ID}", f"REV={ref.REVISION}",
                  f"--height {ref.HEIGHT} --width {ref.WIDTH} --num-frames {ref.NUM_FRAMES}",
                  f"--steps {ref.STEPS} --guidance-scale {ref.GUIDANCE_SCALE} --seed {ref.SEED}",
                  f'--prompt "{ref.PROMPT}"'):
        assert words in text
    assert "--negative-prompt" not in text and "--true-cfg" not in text


def test_vae_tiles_are_the_device_decoders():
    from difflet.backends.neuron.hunyuan_video.vae import NeuronHunyuanVideoVaeDecoderApplication

    params = inspect.signature(NeuronHunyuanVideoVaeDecoderApplication).parameters
    assert (ref.VAE_TILE, ref.VAE_TILE_FRAMES) == (params["tile_size"].default,
                                                   params["tile_frames"].default)


def test_noise_is_the_device_fp32_draw():
    """difflet's ``_initial_latents`` law (``torch.randn`` fp32 from a CPU generator seeded 42),
    diffusers' ``prepare_latents`` at the device shape, and the research's md5."""
    from diffusers import HunyuanVideoPipeline

    noise = ref.initial_noise()
    assert noise.dtype == torch.float32 and tuple(noise.shape) == ref.LATENT_SHAPE
    assert torch.equal(noise, torch.randn(ref.LATENT_SHAPE, generator=_gen(), dtype=torch.float32))
    pipe = HunyuanVideoPipeline.__new__(HunyuanVideoPipeline)
    pipe.vae_scale_factor_temporal, pipe.vae_scale_factor_spatial = 4, 8
    drawn = pipe.prepare_latents(1, 16, ref.HEIGHT, ref.WIDTH, ref.NUM_FRAMES, torch.float32,
                                 torch.device("cpu"), _gen())
    assert torch.equal(noise, drawn)
    assert ref.NOISE_MD5 == "fa10d6052fb81005792bbdab0b95d5d3"  # research: noise.evidence


def test_initial_noise_refuses_another_law(monkeypatch):
    monkeypatch.setattr(ref, "NOISE_MD5", "0" * 32)
    with pytest.raises(RuntimeError, match="md5"):
        ref.initial_noise()


@pytest.mark.parametrize("arm", ["bf16", "mx"])
def test_latents_layout_is_the_device_files(arm):
    path = score.BENCH / score.DEVICE_OUTPUTS["hunyuan_video"][arm]["latents"]
    if not path.is_file():
        pytest.skip(f"no device latents {path} on this host")
    device = torch.load(path, weights_only=True)
    assert type(device) is torch.Tensor and device.dtype == torch.float32
    assert tuple(device.shape) == ref.LATENT_SHAPE


@pytest.mark.parametrize("arm", ["bf16", "mx"])
def test_clip_format_is_the_device_files(arm):
    path = score.BENCH / score.DEVICE_OUTPUTS["hunyuan_video"][arm]["output"]
    if not path.is_file():
        pytest.skip(f"no device clip {path} on this host")
    meta = iio.immeta(path)
    assert meta["fps"] == ref.FPS and tuple(meta["size"]) == (ref.WIDTH, ref.HEIGHT)
    assert sum(1 for _ in iio.imiter(path)) == ref.NUM_FRAMES


def test_saved_clip_is_the_device_writers_byte_for_byte(tmp_path):
    """A decode-shaped clip ([1, 3, F, H, W], values past [-1, 1] included) through the reference's
    frames and ``save_run`` gives the MP4 the device's ``_write_video`` writes."""
    from diffusers.video_processor import VideoProcessor

    from difflet.cli.orchestrators.hunyuan_video import _write_video

    video = torch.rand(1, 3, 5, 32, 48, generator=_gen()) * 2.4 - 1.2
    common.save_run(tmp_path / "ref", latents=torch.zeros(1), settings={}, noise_md5="",
                    timings={}, fps=ref.FPS,
                    output=ref.clip_frames(video, VideoProcessor(vae_scale_factor=8)))
    _write_video(video, str(tmp_path / "device.mp4"))
    digests = {hashlib.md5(p.read_bytes()).hexdigest()
               for p in (tmp_path / "ref" / "out.mp4", tmp_path / "device.mp4")}
    assert len(digests) == 1
