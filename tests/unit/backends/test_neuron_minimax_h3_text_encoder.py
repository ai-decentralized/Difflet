"""L1 (CPU, gloo): H3's Qwen3-VL conditioner as a neuron application vs transformers.

A tiny Qwen3-VL (4 layers, 8 q heads / 4 kv heads x 16, interleaved mrope config) saved by
transformers; the application builds the first 3 layers and returns their output on the valid
rows of a right-padded prompt, unmasked and causal, against transformers' masked
``hidden_states[3]``. tp1/tp2/tp4 shard the embedding, the GQA heads and the MLP.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen3_vl")

from types import SimpleNamespace  # noqa: E402

from tests.unit.backends._neuron_gloo import run_ranks  # noqa: E402
from tests.unit.backends._neuron_h3_toy import (  # noqa: E402
    TE_LAYER,
    te_reference,
    te_worker,
    write_tiny_qwen3vl,
)

OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    pytest.importorskip("accelerate")
    encoder = write_tiny_qwen3vl(tmp_path_factory.mktemp("tiny_qwen3vl"))
    return SimpleNamespace(dir=str(encoder), ref=te_reference(encoder))


@pytest.mark.parametrize("tp,exec_mode", [(1, "eager"), (2, "eager"), (4, "eager"),
                                          (2, "compile")])
def test_text_encoder_matches_transformers(tiny, tp, exec_mode):
    results = run_ranks(te_worker, tiny.dir, exec_mode, world_size=tp, env=OFFLINE)
    for r in results:
        torch.testing.assert_close(torch.tensor(r["hidden"]), tiny.ref, atol=2e-5, rtol=2e-5)
        assert r["layers"] == TE_LAYER
        assert r["report"]["missing"] == [] and r["report"]["unexpected"] == []
        skipped = r["report"]["skipped"]
        assert "lm_head.weight" in skipped
        assert any(k.startswith("model.language_model.layers.3.") for k in skipped)
        assert any(k.startswith("model.visual.") for k in skipped)
