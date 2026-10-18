"""The LTX-2 device text encoder (Gemma-3's language model) as a neuron-backend application,
on CPU against transformers' ``Gemma3TextModel``.

transformers encodes the LEFT-padded prompt with its attention mask (LTX-2's tokenization); the
application encodes the same tokens RIGHT-padded, causally, unmasked. The valid rows of every one
of the stacked hidden states must agree (RoPE is relative, so the shifted positions only change
rounding); the hidden-state order (embeddings, every layer's output but the last, the last after
the final norm) is transformers' own. Nothing here touches the device.
"""

import pytest

torch = pytest.importorskip("torch")

from tests.unit.backends._neuron_gloo import run_ranks  # noqa: E402
from tests.unit.backends._neuron_gemma3_toy import (  # noqa: E402
    SEQ_LEN,
    TINY_GEMMA3,
    VALID,
    gemma3_worker,
    write_tiny_gemma3,
)

OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors.torch")
    root = tmp_path_factory.mktemp("tiny_gemma3")
    return root, write_tiny_gemma3(root)


@pytest.mark.parametrize("world_size", [1, 2])
def test_valid_rows_match_transformers(tiny, world_size):
    root, reference = tiny
    results = run_ranks(gemma3_worker, str(root), world_size=world_size, env=OFFLINE)
    layers = TINY_GEMMA3["num_hidden_layers"]
    for result in results:
        states = torch.tensor(result["states"])
        assert tuple(states.shape) == (1, SEQ_LEN, TINY_GEMMA3["hidden_size"], layers + 1)
        # Valid rows: the first VALID here (right padding), the last VALID in transformers'.
        torch.testing.assert_close(states[:, :VALID], reference[:, SEQ_LEN - VALID:],
                                   atol=2e-5, rtol=2e-5)
        assert result["report"]["missing"] == [] and result["report"]["unexpected"] == []
        assert any(k.startswith("vision_tower.") for k in result["report"]["skipped"])
        assert result["heads"] == 4 // world_size and result["kv_heads"] == 2 // world_size
    if world_size > 1:
        assert results[0]["compiled_blocks"] == [f"layers.{i}" for i in range(layers)]
        assert results[0]["states"] == results[1]["states"]


def test_encoder_refuses_a_sequence_longer_than_the_sliding_window(tmp_path):
    from difflet.models.ltx_2.modeling_gemma3_text_encoder import (
        Gemma3TextEncoder,
        Gemma3TextEncoderConfig,
    )

    config = Gemma3TextEncoderConfig(**{k: v for k, v in TINY_GEMMA3.items() if k in {
        "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim", "sliding_window"}})
    with pytest.raises(NotImplementedError, match="exact only up to the window"):
        Gemma3TextEncoder(config, max_positions=32)
