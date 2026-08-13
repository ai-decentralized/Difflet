from __future__ import annotations

import torch

from difflet.backends.trainium.flux.resident_cross_graph_model import (
    ResidentAnchorUpdateModel,
    ResidentConsumeModel,
    ResidentPredictModel,
    ResidentResetModel,
)


def _share_state(models):
    anchor0 = models[0].anchor0
    anchor1 = models[0].anchor1
    for model in models[1:]:
        model.anchor0 = anchor0
        model.anchor1 = anchor1


def _commit(model, outputs):
    with torch.no_grad():
        model.anchor0.copy_(outputs[2])
        model.anchor1.copy_(outputs[3])
    return outputs[0], outputs[1]


def test_cross_graph_update_predict_consume_and_reset():
    models = [
        ResidentAnchorUpdateModel(seq_len=3, channels=2, dtype=torch.bfloat16),
        ResidentPredictModel(seq_len=3, channels=2, dtype=torch.bfloat16),
        ResidentConsumeModel(seq_len=3, channels=2, dtype=torch.bfloat16),
        ResidentResetModel(seq_len=3, channels=2, dtype=torch.bfloat16),
    ]
    _share_state(models)
    update, predict, consume, reset = models
    anchor_a = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2).to(torch.bfloat16)
    anchor_b = (anchor_a.float() * 2.0 + 1.0).to(torch.bfloat16)

    _commit(update, update(anchor_a))
    _commit(update, update(anchor_b))
    predicted, _ = _commit(
        predict, predict(torch.tensor([-0.5, 1.5], dtype=torch.float32))
    )
    expected = (anchor_a.float() * -0.5 + anchor_b.float() * 1.5).to(torch.bfloat16)
    assert torch.equal(predicted, expected)

    consumed, checksum = _commit(
        consume, consume(predicted, torch.ones((1,), dtype=torch.float32))
    )
    assert torch.equal(consumed, expected)
    assert checksum.item() == expected.float().square().mean().item()

    _commit(reset, reset(torch.tensor([7], dtype=torch.int32)))
    after_reset, _ = _commit(
        predict, predict(torch.tensor([-1.0, 2.0], dtype=torch.float32))
    )
    assert torch.count_nonzero(after_reset).item() == 0


def test_entry_point_input_signatures_are_distinct():
    full = (1, 4096, 64)
    signatures = {
        (full,),
        ((2,),),
        (full, (1,)),
        ((1,),),
    }
    assert len(signatures) == 4
