"""Phase G's LTX-2 CPU fp32 reference (``scripts/fp32_ref/ltx_2.py``) without a model: its
settings against the device code that ran ltx2_tp4/{bf16,mx}_gen50a, the injected noise against
the device's own draw at the device shape, the schedule against the device's, the fp32 guard,
the frames handed to the mp4 writer against the device's, and the step callback's latents."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, LTX2Pipeline
from diffusers.pipelines.ltx2.pipeline_ltx2 import calculate_shift, retrieve_timesteps

from difflet.cli.orchestrators import ltx_2 as device_cli
from difflet.models.ltx_2 import neuron_application
from difflet.models.ltx_2.pipeline import LTX2Orchestrator

_DIR = Path(__file__).resolve().parents[3] / "scripts" / "fp32_ref"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"fp32_ref_{name}", _DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = _load("ltx_2")
common = _load("_common")


def test_settings_are_the_device_runs():
    """The device command's values (research: device_settings.source) and the defaults the
    device code applied where the command gave none."""
    assert (ref.MODEL_ID, ref.REVISION) == (device_cli._HF_MODEL_ID,
                                            "64f65d09e0b76c96202b5bc09997673184cf9b5d")
    assert (ref.HEIGHT, ref.WIDTH, ref.NUM_FRAMES) == (512, 768, 121) == device_cli._DEFAULT_SHAPE
    assert (ref.STEPS, ref.SEED) == (50, 42)
    assert ref.PROMPT == "a cat walking through a garden" and ref.NEGATIVE_PROMPT == ""
    assert ref.GUIDANCE_SCALE == device_cli._DEFAULT_GUIDANCE == 3.5
    assert ref.MAX_SEQUENCE_LENGTH == device_cli._TORCHRUN_APPLICATION_KWARGS["text_seq_len"]
    assert ref.FRAME_RATE == neuron_application._KWARG_DEFAULTS["frame_rate"]
    assert common.DEVICE_FPS["ltx_2"] == ref.FRAME_RATE == device_cli._FPS
    defaults = inspect.signature(LTX2Orchestrator.__call__).parameters
    assert {name: defaults[name].default for name in (
        "guidance_rescale", "stg_scale", "modality_scale", "use_cross_timestep")} == {
        "guidance_rescale": ref.GUIDANCE_RESCALE, "stg_scale": ref.STG_SCALE,
        "modality_scale": ref.MODALITY_SCALE, "use_cross_timestep": ref.USE_CROSS_TIMESTEP}


def test_call_kwargs_override_diffusers_defaults_with_the_devices():
    params = inspect.signature(LTX2Pipeline.__call__).parameters
    kwargs = ref.call_kwargs(ref.STEPS)
    assert set(kwargs) <= set(params)
    # diffusers would run 40 steps at guidance 4.0.
    assert (params["num_inference_steps"].default, params["guidance_scale"].default) == (40, 4.0)
    assert kwargs["num_inference_steps"] == 50 and kwargs["guidance_scale"] == 3.5
    assert kwargs["audio_guidance_scale"] == 3.5 and kwargs["negative_prompt"] == ""
    assert (kwargs["height"], kwargs["width"], kwargs["num_frames"]) == (512, 768, 121)
    assert kwargs["frame_rate"] == 24.0 and kwargs["max_sequence_length"] == 1024
    assert kwargs["output_type"] == "pt" and "latents" not in kwargs
    assert "audio_latents" not in kwargs  # diffusers would normalize them by the VAE's stats


@pytest.mark.skipif(not (ref.SNAPSHOT / "scheduler").is_dir(), reason="no LTX-2 snapshot")
def test_diffusers_default_schedule_is_the_devices():
    """The reference passes no sigmas: diffusers' own (linspace, mu at max_image_seq_len) must be
    the device orchestrator's for the snapshot's scheduler config."""
    device = LTX2Orchestrator(model_path=str(ref.SNAPSHOT), dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(ref.SNAPSHOT / "scheduler")
    config = scheduler.config
    mu = calculate_shift(config.get("max_image_seq_len", 4096),
                         config.get("base_image_seq_len", 1024),
                         config.get("max_image_seq_len", 4096), config.get("base_shift", 0.95),
                         config.get("max_shift", 2.05))
    timesteps, _ = retrieve_timesteps(scheduler, ref.STEPS, "cpu", None,
                                      sigmas=np.linspace(1.0, 1 / ref.STEPS, ref.STEPS), mu=mu)
    assert torch.equal(device._timesteps(ref.STEPS, device="cpu"), timesteps)
    assert torch.equal(device.scheduler.sigmas, scheduler.sigmas)


def test_noise_is_the_devices_draw_at_the_device_shape():
    noise = ref.reference_noise()
    device = ref.device_initial_latents()
    assert noise.keys() == device.keys() == {"video", "audio"}
    assert noise["video"].shape == (1, 6144, 128) and noise["audio"].shape == (1, 126, 128)
    for key in noise:
        assert device[key].dtype == torch.bfloat16 and noise[key].dtype == torch.float32
        assert torch.equal(noise[key], device[key].float())
        assert common.tensor_md5(device[key]) == ref.NOISE_MD5[key]
    # The video is the generator's first draw: the shared helper's law, packed.
    video = common.device_noise(ref.VIDEO_NOISE_SHAPE, ref.SEED, bf16_draw=True)
    assert torch.equal(noise["video"], LTX2Pipeline._pack_latents(video))
    fp32 = torch.randn(ref.VIDEO_NOISE_SHAPE, generator=torch.Generator().manual_seed(ref.SEED))
    assert not torch.equal(noise["video"], LTX2Pipeline._pack_latents(fp32))


def test_assert_fp32_refuses_a_component_left_in_bf16():
    class Pipe:
        components = {"vae": torch.nn.Linear(2, 2), "tokenizer": object(),
                      "vocoder": torch.nn.Linear(2, 2).to(torch.bfloat16)}

    with pytest.raises(RuntimeError, match="vocoder holds"):
        ref.assert_fp32(Pipe())
    Pipe.components["vocoder"] = Pipe.components["vocoder"].float()
    ref.assert_fp32(Pipe())


def test_device_clip_is_what_the_device_hands_the_mp4_writer(monkeypatch, tmp_path):
    import diffusers.utils

    handed = []
    monkeypatch.setattr(diffusers.utils, "export_to_video",
                        lambda frames, path, fps: handed.append((np.stack(frames), fps)))
    frames = torch.rand(1, 3, 3, 8, 16) * 1.2 - 0.1  # (B, F, 3, H, W), a little out of range
    assert device_cli._save_video(frames, str(tmp_path / "x.mp4"))
    clip = ref.device_clip(frames)
    assert clip.shape == (3, 8, 16, 3) and clip.dtype == np.float32
    assert np.array_equal(clip, handed[0][0]) and handed[0][1] == common.DEVICE_FPS["ltx_2"]


def test_last_step_keeps_the_last_steps_packed_latents():
    recorder = ref.LastStep(2)
    for step in range(2):
        kwargs = {"latents": torch.full((1, 4, 2), float(step)),
                  "audio_latents": torch.full((1, 3, 2), -float(step))}
        assert recorder(None, step, None, kwargs) is kwargs
    assert len(recorder.step_s) == 2
    assert torch.equal(recorder.latents["video"], torch.ones(1, 4, 2))
    assert torch.equal(recorder.latents["audio"], -torch.ones(1, 3, 2))


def test_one_step_is_refused_before_any_load(tmp_path, capsys):
    """diffusers' one-step LTX-2 schedule is NaN; the refusal comes before the weights load."""
    with pytest.raises(SystemExit):
        ref.main(["--out-dir", str(tmp_path), "--steps", "1",
                  "--model-path", str(tmp_path / "absent")])
    assert "at least 2 steps" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())
