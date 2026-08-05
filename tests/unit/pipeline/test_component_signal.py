from __future__ import annotations

import pytest
import torch

from difflet.pipeline.cache.component_signal import (
    relative_component_drift,
    residual_input_samples,
    residual_input_sketch,
)


def test_residual_input_sketch_preserves_layout_and_energy() -> None:
    hidden = torch.ones(2, 8 + 16, 8)
    hidden[:, 8:] *= 2

    sketch = residual_input_sketch(
        hidden,
        text_token_count=8,
        image_height=4,
        image_width=4,
        text_regions=2,
        image_region_rows=2,
        image_region_columns=2,
        channel_groups=2,
    )

    assert sketch.shape == (2, 6, 2, 2)
    assert torch.equal(sketch[:, :2, :, 0], torch.ones(2, 2, 2))
    assert torch.equal(sketch[:, :2, :, 1], torch.ones(2, 2, 2))
    assert torch.equal(sketch[:, 2:, :, 0], torch.full((2, 4, 2), 2.0))
    assert torch.equal(sketch[:, 2:, :, 1], torch.full((2, 4, 2), 2.0))


def test_relative_component_drift_is_scale_interpretable() -> None:
    anchor = torch.full((2, 3, 4), 2.0)
    current = anchor.clone()
    current[0] = 3.0
    current[1] = 4.0

    drift = relative_component_drift(current, anchor)

    assert drift.tolist() == pytest.approx([0.5, 1.0])


def test_residual_input_samples_selects_region_and_channel_centers() -> None:
    hidden = torch.arange(24 * 8, dtype=torch.float32).reshape(1, 24, 8)

    samples = residual_input_samples(
        hidden,
        text_token_count=8,
        image_height=4,
        image_width=4,
        text_regions=2,
        image_region_rows=2,
        image_region_columns=2,
        channel_groups=2,
    )

    assert samples.shape == (1, 6, 2)
    assert torch.equal(
        samples,
        hidden[
            :,
            torch.tensor([2, 6, 13, 15, 21, 23]),
        ][:, :, torch.tensor([2, 6])],
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"text_token_count": 7}, "sequence"),
        ({"text_regions": 3}, "divisible"),
        ({"channel_groups": 3}, "channels"),
    ),
)
def test_residual_input_sketch_rejects_invalid_layout(kwargs, message: str) -> None:
    values = {
        "text_token_count": 8,
        "image_height": 4,
        "image_width": 4,
        "text_regions": 2,
        "image_region_rows": 2,
        "image_region_columns": 2,
        "channel_groups": 2,
        **kwargs,
    }

    with pytest.raises(ValueError, match=message):
        residual_input_sketch(torch.ones(1, 24, 8), **values)

    with pytest.raises(ValueError, match=message):
        residual_input_samples(torch.ones(1, 24, 8), **values)
