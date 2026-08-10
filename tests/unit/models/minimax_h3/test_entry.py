from __future__ import annotations

import pytest

from difflet.models.minimax_h3.entry import create_minimax_h3_application
from difflet.pipeline.parallel_config import DiffletParallelConfig


def _create(tmp_path, parallel):
    return create_minimax_h3_application(
        model_path=str(tmp_path),
        parallel=parallel,
        dtype="bf16",
        shape={"height": 768, "width": 1344, "num_frames": 124},
    )


def test_minimax_h3_application_exposes_static_dit_contract(tmp_path):
    app = _create(tmp_path, DiffletParallelConfig(tp_degree=4))

    assert app.shape == {"height": 768, "width": 1344, "num_frames": 124}
    assert app.layout.num_latent_frames == 37
    contract = app.dit_input_contract()
    assert contract["hidden_states"]["shape"] == (1, 37296, 96)
    assert contract["audio_hidden_states"]["shape"] == (1, 414, 32)
    assert contract["encoder_hidden_states"]["shape"] == (1, 1024, 5120)
    assert contract["position_ids"]["shape"] == (38784, 3)


@pytest.mark.parametrize(
    "parallel, message",
    [
        (DiffletParallelConfig(tp_degree=2), "fixed graph is TP4"),
        (DiffletParallelConfig(tp_degree=4, cp_degree=2), "context parallelism"),
        (
            DiffletParallelConfig(tp_degree=4, cfg_parallel_enabled=True),
            "guidance-distilled",
        ),
    ],
)
def test_minimax_h3_rejects_unqualified_topologies(tmp_path, parallel, message):
    with pytest.raises(NotImplementedError, match=message):
        _create(tmp_path, parallel)
