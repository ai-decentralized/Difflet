"""CPU tests for the Wan TPU application's orchestrator contract.

No chip needed: ``build_wan_orchestrator`` is the piece of ``load_eager`` that
decides what the backend-neutral ``WanOrchestrator`` is told, and TeaCache is
the part of that contract that was silently missing — ``difflet serve
--teacache-cadence`` on TPU built an orchestrator with no controller at all.
"""

from __future__ import annotations

import torch

from difflet.models.wan.tpu_application import build_wan_orchestrator


def _build(tmp_path, **kwargs):
    return build_wan_orchestrator(
        model_path=str(tmp_path),
        text_encoder=None,
        transformer=object(),
        transformer_2=None,
        dtype=torch.bfloat16,
        shape={"height": 480, "width": 832, "num_frames": 9},
        text_seq_len=512,
        kwargs=kwargs,
    )


def test_orchestrator_has_no_teacache_controller_by_default(tmp_path):
    orchestrator = _build(tmp_path)
    assert orchestrator._teacache_controller is None
    assert orchestrator.vae_decoder is None  # host decode, see load_eager
    assert (orchestrator.height, orchestrator.width, orchestrator.num_frames) == (480, 832, 9)
    assert orchestrator.max_text_length == 512


def test_orchestrator_receives_fixed_cadence(tmp_path):
    orchestrator = _build(tmp_path, teacache_cadence=2)
    cal = orchestrator._teacache_controller.calibration
    assert (cal.model, cal.cadence, cal.online_delta_alpha) == ("wan", 2, 0.0)
    # Probe-free: no block-0 CPU shadow is ever built for this controller.
    assert orchestrator._teacache_controller.needs_signal() is False


def test_orchestrator_receives_online_delta(tmp_path):
    orchestrator = _build(tmp_path, teacache_online_delta_alpha=0.6)
    cal = orchestrator._teacache_controller.calibration
    assert (cal.cadence, cal.online_delta_alpha) == (0, 0.6)


def test_orchestrator_ignores_adaptive_calibration_on_tpu(tmp_path):
    """The adaptive path needs the Trainium CPU-shadow probe; on TPU the kwarg
    is deliberately not forwarded rather than forwarded and failed late."""
    orchestrator = _build(tmp_path, teacache_calibration_path="/nope.json")
    assert orchestrator.teacache_calibration_path is None
    assert orchestrator._teacache_controller is None
