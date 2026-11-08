"""Phase G's HunyuanVideo-1.5 reference (``scripts/fp32_ref/hunyuan_video_15.py``) against the
device runs it is scored with: its settings against the transformer stage's command line (what
the orchestrator launched for the arms' flags), the arms' launch scripts and the snapshot's
guider, its decode tiles against the device decoder's, its noise against the device's bf16 draw,
and its latents layout and clip format against the device files. No model is loaded."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import imageio.v3 as iio
import numpy as np
import pytest
import torch

_DIR = Path(__file__).resolve().parents[3] / "scripts" / "fp32_ref"
_RUNS = Path("/home/ubuntu/fp8-bench/hv15")
_SNAPSHOT = (Path.home() / ".cache/huggingface/hub"
             / "models--hunyuanvideo-community--HunyuanVideo-1.5-Diffusers-480p_t2v/snapshots"
             / "286be7ce72277246578a3e3cc2487e95ddae5bcf")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"fp32_ref_{name}", _DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = _load("hunyuan_video_15")
common = _load("_common")
score = _load("score")


def _gen() -> torch.Generator:
    return torch.Generator().manual_seed(42)


def test_settings_are_the_transformer_stages():
    """The stage line the orchestrator built for the arms' flags (--num-frames 9 --steps 50
    --seed 42 --prompt; no shape, guidance or negative prompt) carries the reference's values."""
    from difflet.cli.orchestrators import hunyuan_video_15 as orch

    args = argparse.Namespace(height=None, width=None, num_frames=9, steps=50, seed=42,
                              prompt=ref.PROMPT)
    parts = orch.HunyuanVideo15Orchestrator._stage_cli_args(
        SimpleNamespace(args=args), "generate", "/work", 2, True, 1)
    def value(flag: str) -> str:
        return parts[parts.index(flag) + 1]

    assert value("--model-id") == ref.MODEL_ID
    assert (value("--height"), value("--width"), value("--num-frames")) == (
        str(ref.HEIGHT), str(ref.WIDTH), str(ref.NUM_FRAMES))
    assert (value("--steps"), value("--seed"), value("--prompt")) == (
        str(ref.STEPS), str(ref.SEED), ref.PROMPT)
    assert "--guidance-scale" not in parts and "--negative-prompt" not in parts
    assert ref.STEPS == orch._DEFAULT_STEPS
    assert ref.FPS == orch._FPS == common.DEVICE_FPS["hunyuan_video_15"]
    assert ref.NEGATIVE_PROMPT == ""  # _stage_encode: ``req.negative_prompt or ""``


@pytest.mark.parametrize("script, markers", [
    ("run_f6a.sh", ("gen bf16_9 a 0 --num-frames 9",)),
    ("run_nod4.sh", ("--num-frames 9)", "Q=(--quant mxfp8)", "unset DIFFLET_QUANT_TARGETS")),
])
def test_settings_are_the_device_command_lines(script, markers):
    """Both arms' launch scripts (bf16_9; nod4's mx, the shipped MXFP8 default on the tree without
    D4, so no target override): the model, steps, seed, prompt and 9 frames, and none of the
    flags the reference leaves at the stage defaults."""
    path = _RUNS / script
    if not path.is_file():
        pytest.skip(f"no device run script {path} on this host")
    text = path.read_text()
    assert f"--model-id {ref.MODEL_ID}" in text and all(m in text for m in markers)
    assert f'--steps {ref.STEPS} --seed {ref.SEED} --prompt "{ref.PROMPT}"' in text
    for flag in ("--height", "--width", "--guidance-scale", "--negative-prompt", "--revision"):
        assert flag not in text


def test_guidance_is_the_snapshots():
    from difflet.backends.neuron.hunyuan_video15.pipeline import load_guidance

    if not _SNAPSHOT.is_dir():
        pytest.skip(f"no snapshot {_SNAPSHOT} on this host")
    assert load_guidance(str(_SNAPSHOT)) == ref.GUIDANCE_SCALE


def test_vae_tile_is_the_device_decoders():
    from difflet.backends.neuron.hunyuan_video15.transformer import VAE_SPATIAL
    from difflet.backends.neuron.hunyuan_video15.vae import (
        NeuronHunyuanVideo15VaeDecoderApplication,
    )

    params = inspect.signature(NeuronHunyuanVideo15VaeDecoderApplication).parameters
    assert ref.VAE_TILE == params["tile_size"].default
    assert ref.VAE_SPATIAL == VAE_SPATIAL


def test_noise_is_the_device_bf16_draw():
    """The research's checks: difflet's draw (``randn_tensor`` in bf16, CPU generator, seed 42)
    is diffusers' ``prepare_latents`` in bf16 (A) and its fp32 draw rounded to bf16 (B), not the
    fp32 draw itself (C, rel L2 1.66e-3), with the first values the research printed."""
    from diffusers import HunyuanVideo15Pipeline
    from diffusers.utils.torch_utils import randn_tensor

    noise = ref.initial_noise()
    assert noise.dtype == torch.float32 and tuple(noise.shape) == ref.LATENT_SHAPE
    pipe = HunyuanVideo15Pipeline.__new__(HunyuanVideo15Pipeline)
    pipe.vae_scale_factor_temporal, pipe.vae_scale_factor_spatial = 4, ref.VAE_SPATIAL
    args = (1, 32, ref.HEIGHT, ref.WIDTH, ref.NUM_FRAMES)
    cpu = torch.device("cpu")
    assert torch.equal(noise, pipe.prepare_latents(*args, torch.bfloat16, cpu, _gen()).float())
    fp32 = pipe.prepare_latents(*args, torch.float32, cpu, _gen())
    assert torch.equal(noise, fp32.to(torch.bfloat16).float())
    assert not torch.equal(noise, fp32)
    assert float((noise - fp32).norm() / fp32.norm()) == pytest.approx(1.663e-3, rel=1e-3)
    assert noise.flatten()[:4].tolist() == pytest.approx([1.9297, 1.4844, 0.9023, -2.1094],
                                                         abs=1e-4)
    device_law = randn_tensor(ref.LATENT_SHAPE, generator=_gen(), device=cpu,
                              dtype=torch.bfloat16)
    assert common.tensor_md5(device_law) == ref.NOISE_MD5


def test_initial_noise_refuses_another_law(monkeypatch):
    monkeypatch.setattr(ref, "NOISE_MD5", "0" * 32)
    with pytest.raises(RuntimeError, match="md5"):
        ref.initial_noise()


@pytest.mark.parametrize("arm", ["bf16", "mx"])
def test_latents_layout_is_the_device_files(arm):
    """The device keeps bf16 latents; the reference's fp32 ones have the same unpacked shape,
    which is all compare_difflet_runs needs (it compares both in float64)."""
    path = score.BENCH / score.DEVICE_OUTPUTS["hunyuan_video_15"][arm]["latents"]
    if not path.is_file():
        pytest.skip(f"no device latents {path} on this host")
    device = torch.load(path, weights_only=True)
    assert type(device) is torch.Tensor and tuple(device.shape) == ref.LATENT_SHAPE


#: The 9-frame clips of the tree that ships (F7 reverted D4, ``dd3eed2``; ``nod4/summary.txt``):
#: bf16 is the same clip with D4 or without (``bf16_9`` = ``nod4/bf16_gena.mp4``), the MXFP8
#: default is not (``u3``'s ``222a660d…`` is the same set and directory with D4's rotary).
_SHIPPED_MD5 = {"bf16": "9ab6e81c5613651def4bb391d2f6da01",
                "mx": "7a9ddae792a5e728b2affc880851d3fa"}


@pytest.mark.parametrize("arm", ["bf16", "mx"])
def test_device_clips_are_the_shipped_trees(arm):
    path = score.BENCH / score.DEVICE_OUTPUTS["hunyuan_video_15"][arm]["output"]
    if not path.is_file():
        pytest.skip(f"no device clip {path} on this host")
    assert hashlib.md5(path.read_bytes()).hexdigest() == _SHIPPED_MD5[arm]


@pytest.mark.parametrize("arm", ["bf16", "mx"])
def test_clip_format_is_the_device_files(arm):
    path = score.BENCH / score.DEVICE_OUTPUTS["hunyuan_video_15"][arm]["output"]
    if not path.is_file():
        pytest.skip(f"no device clip {path} on this host")
    meta = iio.immeta(path)
    assert meta["fps"] == ref.FPS and tuple(meta["size"]) == (ref.WIDTH, ref.HEIGHT)
    assert sum(1 for _ in iio.imiter(path)) == ref.NUM_FRAMES


def test_clip_frames_are_the_device_vae_stages():
    """The frames the device's VAE stage hands export_to_video: its processor (the pipeline
    module's HunyuanVideo15ImageProcessor at scale 16) on batch 0, float [0, 1] (F, H, W, 3)."""
    import diffusers

    module = inspect.getmodule(diffusers.HunyuanVideo15Pipeline)
    processor = module.HunyuanVideo15ImageProcessor(vae_scale_factor=16)
    video = torch.rand(1, 3, 3, 32, 48, generator=_gen()) * 2.4 - 1.2
    frames = ref.clip_frames(video, processor)
    expected = (video[0].permute(1, 2, 3, 0) * 0.5 + 0.5).clamp(0, 1).numpy()
    assert frames.dtype == np.float32 and np.array_equal(frames, expected)
