from __future__ import annotations

import torch

from difflet.backends.trainium.flux.request_context_stage_model import (
    FluxRequestContextStageModel,
)


def test_request_context_stage_is_bit_exact_and_independent():
    model = FluxRequestContextStageModel()
    sources = (
        torch.randn((1, 3, 4), dtype=torch.bfloat16),
        torch.randn((1, 4), dtype=torch.bfloat16),
        torch.randn((1,), dtype=torch.bfloat16),
        torch.randn((5, 2, 2), dtype=torch.bfloat16),
    )
    staged = model(*sources, torch.zeros((1,), dtype=torch.int32))

    for source, output in zip(sources, staged, strict=True):
        assert torch.equal(output, source)
        assert output.data_ptr() != source.data_ptr()

    sources[0].zero_()
    assert torch.count_nonzero(staged[0]).item() > 0
