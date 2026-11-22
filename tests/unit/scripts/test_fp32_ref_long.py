"""The CPU fp32 references at the const-max campaign's long shapes (``scripts/fp32_ref/*_long.py``)
against the device runs they score: each arm's command line and request lines, Wan 2.2's expert
split at 40 steps by difflet's own rule, the noise (md5, the device code's own draw and
diffusers') and the device latents' layout. No model loads; the device-run checks skip on a host
without ``/mnt/nvme/difflet-exp/constmax-models``."""

from __future__ import annotations

import importlib
import inspect
import json
import re
import shlex
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

RUNS = Path("/mnt/nvme/difflet-exp/constmax-models")
ARMS = "ABCD"

wan21 = importlib.import_module("scripts.fp32_ref.wan21_long")
wan22 = importlib.import_module("scripts.fp32_ref.wan22_long")
hv1 = importlib.import_module("scripts.fp32_ref.hunyuan_video_long")
hv15 = importlib.import_module("scripts.fp32_ref.hunyuan_video_15_long")
_wan = importlib.import_module("scripts.fp32_ref._wan")


def _gen() -> torch.Generator:
    return torch.Generator().manual_seed(42)


def _flags(tokens: list[str]) -> dict:
    out, i = {}, tokens.index("generate") + 1
    while i < len(tokens):
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            out[tokens[i]], i = tokens[i + 1], i + 2
        else:
            out[tokens[i]], i = True, i + 1
    return out


def _device(model: str, arm: str) -> tuple[dict, list[dict]]:
    """The timed run's generate flags and request lines. Wan kept its command %q-quoted
    (timed.cmd.txt); the HunyuanVideo lanes only in /usr/bin/time's line (no spaced values)."""
    run = RUNS / model / arm
    if not (run / "requests.jsonl").is_file():
        pytest.skip(f"no device run {run} on this host")
    if (run / "timed.cmd.txt").is_file():
        tokens = shlex.split((run / "timed.cmd.txt").read_text(encoding="utf-8"))
    else:
        log = (run / "timed.log").read_text(encoding="utf-8", errors="replace")
        tokens = re.search(r'Command being timed: "([^"]*)"', log).group(1).split()
    requests = [json.loads(line) for line in
                (run / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    return _flags(tokens), requests


def _check_quant(flags: dict, arm: str) -> None:
    assert ("--quant" in flags) == (arm in "CD") and flags.get("--quant", "mxfp8") == "mxfp8"


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("model, ref", [("wan21", wan21), ("wan22", wan22)], ids=["wan21", "wan22"])
def test_wan_settings_are_the_device_runs(model, ref, arm):
    flags, requests = _device(model, arm)
    s = ref.SETTINGS
    assert flags["--model-id"] == s["model_id"] and flags["--revision"] == s["revision"]
    assert (flags["--height"], flags["--width"], flags["--num-frames"]) == (
        str(s["height"]), str(s["width"]), str(s["num_frames"]))
    assert ("--dual-expert" in flags) == (s["boundary_ratio"] is not None)
    assert flags["--steps"] == str(s["steps"]) and flags["--seed"] == str(s["seed"])
    assert float(flags["--guidance-scale"]) == s["guidance_scale"]
    assert flags.get("--guidance-scale-2") == (
        None if s["guidance_scale_2"] is None else str(s["guidance_scale_2"]))
    assert flags["--negative-prompt"] == s["negative_prompt"]
    _check_quant(flags, arm)
    for request in requests:
        assert (request["prompt"], request["seed"], request["steps"],
                request["negative_prompt"]) == (s["prompt"], s["seed"], s["steps"],
                                                s["negative_prompt"])


@pytest.mark.parametrize("arm", ARMS)
def test_hv1_settings_are_the_device_runs(arm):
    flags, requests = _device("hunyuan_video", arm)
    b, long = hv1.base, hv1.LONG
    assert flags["--model-id"] == b.MODEL_ID and flags["--revision"] == b.REVISION
    assert (flags["--height"], flags["--width"], flags["--num-frames"]) == (
        str(long["HEIGHT"]), str(long["WIDTH"]), str(long["NUM_FRAMES"]))
    assert flags["--steps"] == str(b.STEPS) and flags["--seed"] == str(b.SEED)
    assert float(flags["--guidance-scale"]) == b.GUIDANCE_SCALE
    assert "--negative-prompt" not in flags and "--true-cfg" not in flags
    _check_quant(flags, arm)
    assert all((r["prompt"], r["seed"], r["steps"]) == (long["PROMPT"], b.SEED, b.STEPS)
               and "negative_prompt" not in r for r in requests)


def _hv1_main(monkeypatch, argv) -> dict:
    """``hv1.main(argv)`` with the run itself stubbed: the directory and prompt it would use."""
    seen = {}
    for name in hv1.LONG:  # main sets these on base; registered here, monkeypatch restores them
        monkeypatch.setattr(hv1.base, name, getattr(hv1.base, name))
    monkeypatch.setattr(hv1.base, "require_fp32_math", lambda: None)
    monkeypatch.setattr(hv1.base, "set_threads", lambda n: None)
    monkeypatch.setattr(hv1.base, "run", lambda out_dir, steps: seen.update(
        out_dir=out_dir, prompt=hv1.base.PROMPT))
    assert hv1.main(argv) == 0
    return seen


def test_hv1_default_prompt_is_the_pinned_one_in_the_given_directory(monkeypatch, tmp_path):
    out = tmp_path / "hunyuan_video_long"
    for argv in (["--out-dir", str(out)],
                 ["--out-dir", str(out), "--prompt", hv1.LONG["PROMPT"]]):
        assert _hv1_main(monkeypatch, argv) == {"out_dir": out, "prompt": hv1.LONG["PROMPT"]}


def test_hv1_another_prompt_is_recorded_and_writes_to_another_key(monkeypatch, tmp_path):
    """A --prompt run can never land in the pinned prompt's directory (a 23 h reference), two
    prompts never share one, and run.json's settings carry the prompt the run used."""
    assert '"prompt": PROMPT' in inspect.getsource(hv1.base.run)  # read before the stub replaces it
    out = tmp_path / "hunyuan_video_long"
    cat = _hv1_main(monkeypatch, ["--prompt", "a cat walking through a garden",
                                  "--out-dir", f"{out}/", "--steps", "1"])
    dog = _hv1_main(monkeypatch, ["--out-dir", str(out), "--prompt", "a dog"])
    # an abbreviation base.main would accept must not slip past the suffix
    assert _hv1_main(monkeypatch, ["--prompt", "a dog", "--out", str(out)]) == dog
    assert (cat["prompt"], dog["prompt"]) == ("a cat walking through a garden", "a dog")
    assert len({out, cat["out_dir"], dog["out_dir"]}) == 3
    for seen in (cat, dog):
        assert seen["out_dir"].parent == tmp_path
        assert seen["out_dir"].name.startswith("hunyuan_video_long_")


@pytest.mark.parametrize("arm", ARMS)
def test_hv15_settings_are_the_device_runs(arm):
    """No --guidance-scale (the snapshot guider's CFG 6.0, which base.run checks) and no
    negative prompt (the encode stage encodes "")."""
    flags, requests = _device("hunyuan_video_15", arm)
    b = hv15.base
    assert flags["--model-id"] == b.MODEL_ID and flags["--revision"] == b.REVISION
    assert (flags["--height"], flags["--width"], flags["--num-frames"]) == (
        str(b.HEIGHT), str(b.WIDTH), str(hv15.LONG["NUM_FRAMES"]))
    assert flags["--steps"] == str(b.STEPS) and flags["--seed"] == str(b.SEED)
    assert "--guidance-scale" not in flags and "--negative-prompt" not in flags
    _check_quant(flags, arm)
    assert all((r["prompt"], r["seed"], r["steps"]) == (b.PROMPT, b.SEED, b.STEPS)
               and "negative_prompt" not in r for r in requests)


def test_wan22_dit_forwards_are_the_devices_expert_split():
    """difflet's expert rule over the snapshot's 40 UniPC timesteps (t >= 875 high noise), two
    forwards (cond, uncond) per step at guidance 4.0 / 3.0."""
    snapshot = (Path.home() / ".cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers"
                / "snapshots" / wan22.SETTINGS["revision"])
    if not snapshot.is_dir():
        pytest.skip(f"no snapshot {snapshot} on this host")
    from difflet.models.wan.pipeline import WanOrchestrator

    orchestrator = WanOrchestrator(model_path=str(snapshot), transformer="transformer",
                                   transformer_2="transformer_2")
    scheduler = orchestrator.scheduler
    scheduler.set_timesteps(wan22.SETTINGS["steps"])
    boundary = orchestrator.boundary_ratio * scheduler.config.num_train_timesteps
    experts = Counter(orchestrator._select_transformer(t, boundary) for t in scheduler.timesteps)
    assert {k: 2 * v for k, v in experts.items()} == wan22.DIT_FORWARDS
    assert wan21.DIT_FORWARDS == {"transformer": 2 * wan21.SETTINGS["steps"]}


@pytest.mark.parametrize("ref", [wan21, wan22], ids=["wan21", "wan22"])
def test_wan_noise_is_the_devices_draw(ref, tmp_path):
    """Equal to WanOrchestrator.prepare_latents' draw at the request's shape, and its md5
    (both asserted inside reference_noise)."""
    noise = _wan.reference_noise(ref.SETTINGS, tmp_path)
    assert tuple(noise.shape) == ref.SETTINGS["noise_shape"] and noise.dtype == torch.float32


def test_wan_noise_refuses_another_md5(tmp_path):
    with pytest.raises(AssertionError, match="md5"):
        _wan.reference_noise({**wan21.SETTINGS, "noise_md5": "0" * 32}, tmp_path)


def test_hv1_noise_is_the_devices_draw(monkeypatch):
    """md5 (asserted by initial_noise), difflet's ``_initial_latents`` at the device config and
    diffusers' ``prepare_latents``."""
    from diffusers import HunyuanVideoPipeline

    from difflet.backends.neuron.hunyuan_video.config import NeuronHunyuanVideoConfig
    from difflet.models.hunyuan_video.neuron_application import (
        TorchNeuronHunyuanVideoApplication,
    )

    for name, value in hv1.LONG.items():
        monkeypatch.setattr(hv1.base, name, value)
    noise = hv1.base.initial_noise()
    config = NeuronHunyuanVideoConfig(height=hv1.LONG["HEIGHT"], width=hv1.LONG["WIDTH"],
                                      num_frames=hv1.LONG["NUM_FRAMES"])
    stub = SimpleNamespace(config=config, _draw_on_rank0=lambda draw: draw())
    assert torch.equal(noise, TorchNeuronHunyuanVideoApplication._initial_latents(stub, _gen()))
    pipe = HunyuanVideoPipeline.__new__(HunyuanVideoPipeline)
    pipe.vae_scale_factor_temporal, pipe.vae_scale_factor_spatial = 4, 8
    assert torch.equal(noise, pipe.prepare_latents(
        1, 16, config.height, config.width, config.num_frames, torch.float32,
        torch.device("cpu"), _gen()))


def test_hv15_noise_is_the_devices_bf16_draw(monkeypatch):
    """md5 (asserted by initial_noise), pipeline15's draw (randn_tensor in bf16 on a CPU
    generator at its VAE geometry) and diffusers' bf16 ``prepare_latents``."""
    from diffusers import HunyuanVideo15Pipeline
    from diffusers.utils.torch_utils import randn_tensor

    from difflet.models.hunyuan_video.pipeline15 import VAE_SPATIAL, VAE_TEMPORAL

    for name, value in hv15.LONG.items():
        monkeypatch.setattr(hv15.base, name, value)
    noise = hv15.base.initial_noise()
    b = hv15.base
    shape = (1, 32, (b.NUM_FRAMES - 1) // VAE_TEMPORAL + 1, b.HEIGHT // VAE_SPATIAL,
             b.WIDTH // VAE_SPATIAL)
    assert shape == b.LATENT_SHAPE
    if torch.__version__ < "2.13":
        return  # older torch draws bf16 by another law (scripts/fp32_ref/_common.device_noise)
    bf16 = torch.bfloat16
    assert torch.equal(noise, randn_tensor(shape, generator=_gen(), device=torch.device("cpu"),
                                           dtype=bf16).float())
    pipe = HunyuanVideo15Pipeline.__new__(HunyuanVideo15Pipeline)
    pipe.vae_scale_factor_temporal, pipe.vae_scale_factor_spatial = VAE_TEMPORAL, VAE_SPATIAL
    assert torch.equal(noise, pipe.prepare_latents(
        1, 32, b.HEIGHT, b.WIDTH, b.NUM_FRAMES, bf16, torch.device("cpu"), _gen()).float())


@pytest.mark.parametrize("model, shape, dtype", [
    ("wan21", wan21.SETTINGS["noise_shape"], torch.float32),
    ("wan22", wan22.SETTINGS["noise_shape"], torch.float32),
    ("hunyuan_video", hv1.LONG["LATENT_SHAPE"], torch.float32),
    ("hunyuan_video_15", hv15.LONG["LATENT_SHAPE"], torch.bfloat16),
])
@pytest.mark.parametrize("arm", ARMS)
def test_latents_layout_is_the_device_files(model, shape, dtype, arm):
    """The final latents a scorer compares the reference's latents.pt with."""
    path = RUNS / model / arm / "work" / "worker_0" / "latents_req0001.pt"
    if not path.is_file():
        pytest.skip(f"no device latents {path} on this host")
    device = torch.load(path, map_location="cpu", weights_only=True)
    assert (tuple(device.shape), device.dtype) == (shape, dtype)
