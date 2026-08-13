from __future__ import annotations

import torch

from difflet.backends.trainium.flux.resident_cache_step_model import (
    ResidentCacheAnchorStepModel,
    ResidentCacheFinalizeModel,
    ResidentCacheInitializeModel,
    ResidentCachePredictOnlyModel,
    ResidentCacheSchedulerStepModel,
    ResidentCacheSkipSegmentScanModel,
    ResidentCacheSkipStepBarrierModel,
    ResidentCacheSkipStepModel,
)


def _share_state(models):
    anchor0, anchor1, latent = models[0].anchor0, models[0].anchor1, models[0].latent
    for model in models[1:]:
        model.anchor0 = anchor0
        model.anchor1 = anchor1
        model.latent = latent


def _commit(model, outputs):
    with torch.no_grad():
        model.anchor0.copy_(outputs[2])
        model.anchor1.copy_(outputs[3])
        model.latent.copy_(outputs[4])
    return outputs[0]


def test_resident_cache_step_matches_flow_euler_sequence():
    models = [
        ResidentCacheInitializeModel(seq_len=2, channels=2, dtype=torch.bfloat16),
        ResidentCacheAnchorStepModel(seq_len=2, channels=2, dtype=torch.bfloat16),
        ResidentCacheSkipStepModel(seq_len=2, channels=2, dtype=torch.bfloat16),
        ResidentCacheFinalizeModel(seq_len=2, channels=2, dtype=torch.bfloat16),
    ]
    _share_state(models)
    initialize, anchor, skip, finalize = models
    initial = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2).to(torch.bfloat16)
    noise_a = torch.full_like(initial, 2.0)
    noise_b = torch.full_like(initial, 4.0)

    actual = _commit(
        initialize, initialize(initial, torch.tensor([1, 0], dtype=torch.int32))
    )
    expected = initial
    assert torch.equal(actual, expected)

    for noise, delta in ((noise_a, -0.1), (noise_b, -0.2)):
        expected = (expected.float() + delta * noise.float()).to(torch.bfloat16)
        actual = _commit(
            anchor, anchor(noise, torch.tensor([delta], dtype=torch.float32))
        )
        assert torch.equal(actual, expected)

    predicted = (noise_a.float() * -0.5 + noise_b.float() * 1.5).to(torch.bfloat16)
    expected = (expected.float() + -0.3 * predicted.float()).to(torch.bfloat16)
    actual = _commit(
        skip,
        skip(
            torch.tensor([-0.5, 1.5], dtype=torch.float32),
            torch.tensor([-0.3], dtype=torch.float32),
        ),
    )
    assert torch.equal(actual, expected)
    final = _commit(finalize, finalize(torch.tensor([0, 0, 1], dtype=torch.int32)))
    assert torch.equal(final, expected)


def test_split_predict_scheduler_preserves_bf16_boundary():
    models = [
        ResidentCacheInitializeModel(seq_len=2, channels=2, dtype=torch.bfloat16),
        ResidentCacheAnchorStepModel(seq_len=2, channels=2, dtype=torch.bfloat16),
        ResidentCachePredictOnlyModel(seq_len=2, channels=2, dtype=torch.bfloat16),
        ResidentCacheSchedulerStepModel(seq_len=2, channels=2, dtype=torch.bfloat16),
    ]
    _share_state(models)
    initialize, anchor, predict, scheduler = models
    initial = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2).to(torch.bfloat16)
    noise_a = torch.full_like(initial, 1.1015625)
    noise_b = torch.full_like(initial, 1.8984375)
    _commit(initialize, initialize(initial, torch.tensor([1, 0], dtype=torch.int32)))
    _commit(anchor, anchor(noise_a, torch.tensor([-0.1], dtype=torch.float32)))
    current = _commit(anchor, anchor(noise_b, torch.tensor([-0.2], dtype=torch.float32)))
    coefficients = torch.tensor([-1.0, 2.0], dtype=torch.float32)
    predicted = _commit(predict, predict(coefficients))
    expected_prediction = (
        noise_a.float() * -1.0 + noise_b.float() * 2.0
    ).to(torch.bfloat16)
    assert torch.equal(predicted, expected_prediction)
    actual = _commit(
        scheduler,
        scheduler(predicted, torch.tensor([-0.3], dtype=torch.float32)),
    )
    expected = (current.float() + -0.3 * expected_prediction.float()).to(torch.bfloat16)
    assert torch.equal(actual, expected)


def test_barrier_skip_preserves_host_bf16_contract_on_cpu():
    models = [
        ResidentCacheInitializeModel(seq_len=2, channels=2, dtype=torch.bfloat16),
        ResidentCacheAnchorStepModel(seq_len=2, channels=2, dtype=torch.bfloat16),
        ResidentCacheSkipStepBarrierModel(seq_len=2, channels=2, dtype=torch.bfloat16),
    ]
    _share_state(models)
    initialize, anchor, skip = models
    initial = torch.tensor([0.2, -0.4, 0.6, -0.8]).reshape(1, 2, 2).to(torch.bfloat16)
    noise_a = torch.tensor([1.1015625, -2.203125, 3.296875, -4.40625]).reshape(
        1, 2, 2
    ).to(torch.bfloat16)
    noise_b = torch.tensor([1.8984375, -1.3046875, 2.703125, -3.59375]).reshape(
        1, 2, 2
    ).to(torch.bfloat16)
    _commit(initialize, initialize(initial, torch.tensor([1, 0], dtype=torch.int32)))
    _commit(anchor, anchor(noise_a, torch.tensor([-0.1], dtype=torch.float32)))
    current = _commit(anchor, anchor(noise_b, torch.tensor([-0.2], dtype=torch.float32)))
    coefficients = torch.tensor([-1.0, 2.0], dtype=torch.float32)
    prediction = (
        noise_a.float() * coefficients[0] + noise_b.float() * coefficients[1]
    ).to(torch.bfloat16)
    expected = (current.float() + -0.3 * prediction.float()).to(torch.bfloat16)
    actual = _commit(
        skip,
        skip(coefficients, torch.tensor([-0.3], dtype=torch.float32)),
    )
    assert torch.equal(actual, expected)


def test_scan_segment_preserves_each_bf16_scheduler_update_on_cpu():
    models = [
        ResidentCacheInitializeModel(seq_len=2, channels=2, dtype=torch.bfloat16),
        ResidentCacheAnchorStepModel(seq_len=2, channels=2, dtype=torch.bfloat16),
        ResidentCacheSkipSegmentScanModel(seq_len=2, channels=2, dtype=torch.bfloat16),
    ]
    _share_state(models)
    initialize, anchor, scan = models
    initial = torch.tensor([0.2, -0.4, 0.6, -0.8]).reshape(1, 2, 2).to(torch.bfloat16)
    noise_a = torch.tensor([1.1015625, -2.203125, 3.296875, -4.40625]).reshape(
        1, 2, 2
    ).to(torch.bfloat16)
    noise_b = torch.tensor([1.8984375, -1.3046875, 2.703125, -3.59375]).reshape(
        1, 2, 2
    ).to(torch.bfloat16)
    _commit(initialize, initialize(initial, torch.tensor([1, 0], dtype=torch.int32)))
    _commit(anchor, anchor(noise_a, torch.tensor([-0.1], dtype=torch.float32)))
    current = _commit(anchor, anchor(noise_b, torch.tensor([-0.2], dtype=torch.float32)))
    coefficients = torch.zeros((9, 2), dtype=torch.float32)
    deltas = torch.zeros((9, 1), dtype=torch.float32)
    coefficients[:3] = torch.tensor([[-1.0, 2.0], [-2.0, 3.0], [-3.0, 4.0]])
    deltas[:3, 0] = torch.tensor([-0.3, -0.25, -0.2])
    expected = current
    for index in range(3):
        predicted = (
            noise_a.float() * coefficients[index, 0]
            + noise_b.float() * coefficients[index, 1]
        ).to(torch.bfloat16)
        expected = (expected.float() + deltas[index, 0] * predicted.float()).to(
            torch.bfloat16
        )
    actual = _commit(scan, scan(coefficients, deltas))
    assert torch.equal(actual, expected)
