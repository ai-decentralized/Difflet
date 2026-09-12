"""CPU tests for the LTX-2 TPU application's construction contract."""

from __future__ import annotations

import json

import pytest
import torch

from difflet.models.ltx_2.entry import create_ltx_2_application
from difflet.models.ltx_2.tpu_application import TpuLTX2Application
from difflet.pipeline.parallel_config import DiffletParallelConfig
from tests.unit.backends.test_tpu_ltx_2_config import _UPSTREAM


@pytest.fixture
def snapshot(tmp_path):
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text(json.dumps(_UPSTREAM))
    return tmp_path


def test_entry_routes_tpu_to_the_tpu_application(snapshot):
    app = create_ltx_2_application(
        model_path=str(snapshot), parallel=DiffletParallelConfig(tp_degree=4), dtype="bf16",
        shape={"height": 512, "width": 768, "num_frames": 121}, backend="tpu", teacache_cadence=2,
    )
    assert isinstance(app, TpuLTX2Application)
    assert app.dtype is torch.bfloat16
    assert app.kwargs["teacache_cadence"] == 2
    assert app.pipeline is None and app.host_pipeline is None  # load_eager builds both


def test_entry_still_refuses_cp_on_tpu(snapshot):
    with pytest.raises(NotImplementedError, match="CP"):
        create_ltx_2_application(
            model_path=str(snapshot), parallel=DiffletParallelConfig(tp_degree=2, cp_degree=2),
            dtype="bf16", shape={"height": 512, "width": 768, "num_frames": 121}, backend="tpu",
        )


def test_dit_input_contract_matches_the_bundle(snapshot):
    app = create_ltx_2_application(
        model_path=str(snapshot), parallel=DiffletParallelConfig(tp_degree=4), dtype=torch.bfloat16,
        shape={"height": 512, "width": 768, "num_frames": 121}, backend="tpu",
    )
    contract = app.dit_input_contract()
    assert contract["hidden_states"]["shape"] == (1, 6144, 128)
    assert contract["audio_hidden_states"]["shape"] == (1, 126, 128)
    assert contract["encoder_hidden_states"]["shape"] == (1, 1024, 3840)
    assert contract["video_coords"] == {"shape": (1, 3, 6144, 2), "dtype": torch.float32}
    assert contract["encoder_attention_mask"]["dtype"] is torch.bool
    assert len(contract) == 10
