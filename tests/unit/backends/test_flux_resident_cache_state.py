from __future__ import annotations

import torch

from difflet.backends.trainium.flux.resident_cache_state_model import (
    ANCHOR_ACTION,
    PREDICT_ACTION,
    RESET_ACTION,
    FluxResidentCacheStateModel,
)


def _call(model, candidate, coefficients, action):
    output, checksum, new0, new1 = model(
        candidate,
        torch.tensor(coefficients, dtype=torch.float32),
        torch.tensor([action], dtype=torch.int32),
    )
    with torch.no_grad():
        model.anchor0.copy_(new0)
        model.anchor1.copy_(new1)
    return output, checksum


def test_resident_state_anchor_predict_reset_sequence():
    model = FluxResidentCacheStateModel(seq_len=3, channels=2, dtype=torch.bfloat16)
    zeros = torch.zeros((1, 3, 2), dtype=torch.bfloat16)
    anchor_a = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2).to(torch.bfloat16)
    anchor_b = (anchor_a.float() * 2.0 + 1.0).to(torch.bfloat16)
    anchor_c = (anchor_a.float() * -0.5 + 3.0).to(torch.bfloat16)

    reset, checksum = _call(model, zeros, [0.0, 0.0], RESET_ACTION)
    assert torch.equal(reset, zeros)
    assert checksum.item() == 0.0

    output_a, _ = _call(model, anchor_a, [0.0, 0.0], ANCHOR_ACTION)
    output_b, _ = _call(model, anchor_b, [0.0, 0.0], ANCHOR_ACTION)
    assert torch.equal(output_a, anchor_a)
    assert torch.equal(output_b, anchor_b)

    predicted, _ = _call(model, zeros, [-0.5, 1.5], PREDICT_ACTION)
    expected = (anchor_a.float() * -0.5 + anchor_b.float() * 1.5).to(torch.bfloat16)
    assert torch.equal(predicted, expected)

    _call(model, anchor_c, [0.0, 0.0], ANCHOR_ACTION)
    predicted_after_shift, _ = _call(model, zeros, [-1.0, 2.0], PREDICT_ACTION)
    shifted_expected = (anchor_b.float() * -1.0 + anchor_c.float() * 2.0).to(
        torch.bfloat16
    )
    assert torch.equal(predicted_after_shift, shifted_expected)

    _call(model, zeros, [0.0, 0.0], RESET_ACTION)
    predicted_after_reset, _ = _call(model, zeros, [-1.0, 2.0], PREDICT_ACTION)
    assert torch.equal(predicted_after_reset, zeros)


def test_predict_does_not_change_resident_anchors():
    model = FluxResidentCacheStateModel(seq_len=2, channels=2, dtype=torch.bfloat16)
    anchor_a = torch.ones((1, 2, 2), dtype=torch.bfloat16)
    anchor_b = torch.full((1, 2, 2), 4.0, dtype=torch.bfloat16)
    zeros = torch.zeros_like(anchor_a)
    _call(model, anchor_a, [0.0, 0.0], ANCHOR_ACTION)
    _call(model, anchor_b, [0.0, 0.0], ANCHOR_ACTION)
    before = (model.anchor0.detach().clone(), model.anchor1.detach().clone())
    first, _ = _call(model, zeros, [0.25, 0.75], PREDICT_ACTION)
    second, _ = _call(model, zeros, [0.25, 0.75], PREDICT_ACTION)
    assert torch.equal(first, second)
    assert torch.equal(model.anchor0, before[0])
    assert torch.equal(model.anchor1, before[1])
