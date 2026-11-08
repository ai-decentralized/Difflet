"""Phase G's Wan 2.2 (both experts) CPU fp32 reference (``scripts/fp32_ref/wan22.py``): the
settings against the device runs' (the research object, the device command line, the snapshot
the device loaded and difflet's expert rule on its scheduler) and the noise against the
device's draw at the device shape. No model loads."""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest
import torch

_DIR = Path(__file__).resolve().parents[3] / "scripts" / "fp32_ref"
BENCH = Path("/home/ubuntu/fp8-bench")
RESEARCH = BENCH / "phase_g_research.json"
#: The command line of both scored device runs (arms wan22d_bf16 and wan22d_mx).
COMMAND = BENCH / "wan_h3" / "final_reruns.sh"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"fp32_ref_{name}", _DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wan22 = _load("wan22")
_wan = _load("_wan")
S = wan22.SETTINGS
SNAPSHOT = (Path.home() / ".cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers"
            / "snapshots" / S["revision"])


def _snapshot() -> Path:
    if not SNAPSHOT.is_dir():
        pytest.skip(f"no snapshot {SNAPSHOT} on this host")
    return SNAPSHOT


def test_settings_are_the_research_device_settings():
    if not RESEARCH.is_file():
        pytest.skip(f"no phase G research {RESEARCH} on this host")
    objects = [o for o in json.loads(RESEARCH.read_text())
               if o["device_settings"]["model_id"] == S["model_id"]]
    assert len(objects) == 1
    research = objects[0]
    device = research["device_settings"]
    for name in ("model_id", "height", "width", "num_frames", "steps", "seed", "prompt"):
        assert S[name] == device[name], name
    assert device["revision"].startswith(S["revision"])
    assert device["guidance"].startswith(f"{S['guidance_scale']} on both experts, so no CFG")
    assert S["guidance_scale_2"] == S["guidance_scale"]
    assert device["negative_prompt"].startswith("None")
    assert f"boundary_ratio {S['boundary_ratio']}" in device["other"]
    assert f"flow_shift {S['flow_shift']}" in device["other"]
    assert f"max_sequence_length / text_seq_len is {S['max_sequence_length']}" in device["other"]
    assert str(S["noise_shape"]) in research["latents_layout"]


def test_settings_are_the_device_command_line():
    """Both experts, the shape and request flags, and no --guidance-scale(-2) /
    --negative-prompt: the CLI's guidance 1.0 on both experts and no negative prompt."""
    if not COMMAND.is_file():
        pytest.skip(f"no device command {COMMAND} on this host")
    text = COMMAND.read_text()
    assert f"--model-id {S['model_id']} --dual-expert" in text
    assert f"--height {S['height']} --width {S['width']} --num-frames {S['num_frames']}" in text
    assert f'--steps {S["steps"]} --seed {S["seed"]} --prompt "{S["prompt"]}"' in text
    assert "--guidance-scale" not in text and "--negative-prompt" not in text


def test_text_length_is_the_clis():
    from difflet.cli.orchestrators import wan

    assert S["max_sequence_length"] == wan._TORCHRUN_APPLICATION_KWARGS["text_seq_len"]


def test_snapshot_is_two_experts_at_the_devices_boundary_and_flow_shift():
    snapshot = _snapshot()
    from difflet.models.wan.experts import has_low_noise_expert, read_boundary_ratio

    assert read_boundary_ratio(snapshot) == S["boundary_ratio"]
    assert has_low_noise_expert(snapshot)
    config = json.loads((snapshot / "scheduler" / "scheduler_config.json").read_text())
    assert config["_class_name"] == "UniPCMultistepScheduler"
    assert config["flow_shift"] == S["flow_shift"]


def test_dit_forwards_are_the_devices_expert_split():
    """difflet's own expert rule over the snapshot's 50 UniPC timesteps: the high-noise expert
    for t >= 875 (step 15's t = 875 included), the low-noise one after: 16 and 34 forwards at
    guidance 1.0, which the reference's hook counts must equal."""
    from difflet.models.wan.pipeline import WanOrchestrator

    orchestrator = WanOrchestrator(model_path=str(_snapshot()), transformer="transformer",
                                   transformer_2="transformer_2")
    scheduler = orchestrator.scheduler
    scheduler.set_timesteps(S["steps"])
    boundary = orchestrator.boundary_ratio * scheduler.config.num_train_timesteps
    experts = Counter(orchestrator._select_transformer(t, boundary)
                      for t in scheduler.timesteps)
    assert experts == wan22.DIT_FORWARDS == {"transformer": 16, "transformer_2": 34}


def test_noise_is_the_devices_draw(tmp_path):
    """The device shape's fp32 draw, equal to difflet's (asserted inside) and to the values the
    research printed from it (the same seed and shape as Wan 2.1, so the same tensor)."""
    noise = _wan.reference_noise(S, tmp_path)
    assert tuple(noise.shape) == S["noise_shape"] and noise.dtype == torch.float32
    assert noise[0, 0, 0, 0, :5].tolist() == pytest.approx(
        [1.92692, 1.48728, 0.90072, -2.10552, 0.67842], abs=1e-5)
    assert float(noise.double().sum()) == pytest.approx(-378.3333, abs=1e-4)
