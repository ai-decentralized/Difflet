"""Phase G's Wan 2.1 CPU fp32 reference (``scripts/fp32_ref/wan21.py``) and the Wan helper it
shares with Wan 2.2 (``_wan.py``): the settings against the device runs' (the research object,
the device command lines, difflet's CLI defaults, the snapshot the device loaded), the noise
against the device's draw at the device shape, and the fp32-pipeline guard. No model loads."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler

_DIR = Path(__file__).resolve().parents[3] / "scripts" / "fp32_ref"
BENCH = Path("/home/ubuntu/fp8-bench")
RESEARCH = BENCH / "phase_g_research.json"
#: The command lines of the two scored device runs: bf16 (h5) and the shipped MXFP8 (final).
COMMANDS = (BENCH / "wan_h3" / "verify_h5.sh", BENCH / "wan_h3" / "final_reruns.sh")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"fp32_ref_{name}", _DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wan21 = _load("wan21")
_wan = _load("_wan")
S = wan21.SETTINGS
SNAPSHOT = (Path.home() / ".cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-14B-Diffusers"
            / "snapshots" / S["revision"])


def _research() -> dict:
    if not RESEARCH.is_file():
        pytest.skip(f"no phase G research {RESEARCH} on this host")
    objects = [o for o in json.loads(RESEARCH.read_text())
               if o["device_settings"]["model_id"] == S["model_id"]]
    assert len(objects) == 1
    return objects[0]


def test_settings_are_the_research_device_settings():
    research = _research()
    device = research["device_settings"]
    for name in ("model_id", "height", "width", "num_frames", "steps", "seed", "prompt"):
        assert S[name] == device[name], name
    assert device["revision"].startswith(S["revision"])
    assert device["guidance"].startswith(f"{S['guidance_scale']}, so there is no CFG")
    assert device["negative_prompt"].startswith("None")
    assert "no boundary_ratio" in research["model"] and S["boundary_ratio"] is None
    assert S["guidance_scale_2"] is None
    assert f"flow_shift {S['flow_shift']}" in device["other"]
    assert f"text_seq_len {S['max_sequence_length']}" in device["other"]
    assert str(S["noise_shape"]) in research["latents_layout"]


@pytest.mark.parametrize("command", COMMANDS, ids=lambda p: p.name)
def test_settings_are_the_device_command_lines(command):
    """Each scored run's flags, and no --guidance-scale / --negative-prompt: the CLI's
    guidance 1.0 and no negative prompt are what the device ran."""
    if not command.is_file():
        pytest.skip(f"no device command {command} on this host")
    text = command.read_text()
    assert f"--model-id {S['model_id']}" in text
    assert f"--height {S['height']} --width {S['width']} --num-frames {S['num_frames']}" in text
    assert f'--steps {S["steps"]} --seed {S["seed"]} --prompt "{S["prompt"]}"' in text
    assert "--guidance-scale" not in text and "--negative-prompt" not in text


def test_text_length_is_the_clis():
    from difflet.cli.orchestrators import wan

    assert S["max_sequence_length"] == wan._TORCHRUN_APPLICATION_KWARGS["text_seq_len"]


def test_snapshot_is_one_expert_at_the_devices_flow_shift():
    """What the device loaded from the snapshot: no boundary_ratio (so one expert, one DiT
    forward per step at guidance 1.0) and the snapshot's UniPC at flow shift 3.0."""
    if not SNAPSHOT.is_dir():
        pytest.skip(f"no snapshot {SNAPSHOT} on this host")
    from difflet.models.wan.experts import has_low_noise_expert, read_boundary_ratio

    assert read_boundary_ratio(SNAPSHOT) is None and not has_low_noise_expert(SNAPSHOT)
    assert wan21.DIT_FORWARDS == {"transformer": S["steps"]}
    config = json.loads((SNAPSHOT / "scheduler" / "scheduler_config.json").read_text())
    assert config["_class_name"] == "UniPCMultistepScheduler"
    assert config["flow_shift"] == S["flow_shift"]


def test_noise_is_the_devices_draw(tmp_path):
    """The device shape's fp32 draw, equal to difflet's (asserted inside) and to the values
    the research printed from it."""
    noise = _wan.reference_noise(S, tmp_path)
    assert tuple(noise.shape) == S["noise_shape"] and noise.dtype == torch.float32
    assert noise[0, 0, 0, 0, :4].tolist() == [1.9269152879714966, 1.4872840642929077,
                                              0.9007171988487244, -2.1055209636688232]
    assert float(noise.double().sum()) == pytest.approx(-378.333338, abs=1e-6)


def test_noise_refuses_a_shape_the_device_does_not_draw(tmp_path):
    with pytest.raises(AssertionError, match="not the device's draw"):
        _wan.reference_noise({**S, "num_frames": 13}, tmp_path)


def _pipe(vae_dtype=torch.float32, scheduler=None, boundary_ratio=None, transformer_2=None):
    scheduler = scheduler or UniPCMultistepScheduler(use_flow_sigmas=True, flow_shift=3.0,
                                                     prediction_type="flow_prediction")
    components = {"vae": torch.nn.Linear(2, 2).to(vae_dtype), "scheduler": scheduler,
                  "tokenizer": object(), "transformer": torch.nn.Linear(2, 2),
                  "transformer_2": transformer_2}
    return SimpleNamespace(components=components, scheduler=scheduler,
                           transformer_2=transformer_2,
                           config=SimpleNamespace(boundary_ratio=boundary_ratio))


def test_check_pipeline_accepts_the_devices_fp32_model():
    _wan.check_pipeline(_pipe(), S)


@pytest.mark.parametrize("pipe, match", [
    (_pipe(vae_dtype=torch.bfloat16), "vae holds"),
    (_pipe(boundary_ratio=0.875), "boundary_ratio"),
    (_pipe(transformer_2=torch.nn.Linear(2, 2)), "low-noise expert"),
    (_pipe(scheduler=UniPCMultistepScheduler(use_flow_sigmas=True, flow_shift=5.0)),
     "flow_shift 5.0"),
    (_pipe(scheduler=FlowMatchEulerDiscreteScheduler(shift=3.0)), "FlowMatchEuler"),
])
def test_check_pipeline_refuses_what_the_device_did_not_run(pipe, match):
    with pytest.raises(ValueError, match=match):
        _wan.check_pipeline(pipe, S)
