"""L1 unit tests (CPU only): the MiniMax-H3 DiT as a neuron-backend application.

The model is the tiny diffusers-layout H3 of tests/unit/backends/_neuron_h3_toy.py (2 blocks, 1
refiner block, 4 heads x 16, a sharded checkpoint with an index). The reference is diffusers'
``MiniMaxH3Transformer3DModel`` itself, reloaded from the same files with its ``adaln_proj``
branch. Every forward runs in spawned gloo processes: at TP2/TP4 the head shards, the split
SwiGLU (a ``CheckpointSlice`` of the fused weight per rank) and the all-reduces run, and with
``precomputed_adaln`` the host-built modulation table replaces the branch. Nothing here touches
the device.
"""

import pytest

torch = pytest.importorskip("torch")

from types import SimpleNamespace  # noqa: E402

from difflet.backends.neuron.minimax_h3.transformer import (  # noqa: E402
    NeuronMiniMaxH3Config,
    checkpoint_key,
    swiglu_slices,
)
from difflet.backends.neuron.ops_impl import parallel_mesh  # noqa: E402
from tests.unit.backends._neuron_gloo import run_ranks  # noqa: E402
from tests.unit.backends._neuron_h3_toy import (  # noqa: E402
    TINY,
    TINY_SHAPE,
    TINY_TEXT_SEQ_LEN,
    diffusers_reference,
    l1_h3_worker,
    write_tiny_h3_transformer,
)

ATOL = RTOL = 2e-5
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


@pytest.fixture(autouse=True)
def _clean_mesh(monkeypatch):
    monkeypatch.delenv("DIFFLET_EXEC_MODE", raising=False)
    parallel_mesh.destroy_parallel_mesh()
    yield
    parallel_mesh.destroy_parallel_mesh()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    pytest.importorskip("safetensors.torch")
    pytest.importorskip("accelerate")
    pytest.importorskip("diffusers")
    transformer = write_tiny_h3_transformer(tmp_path_factory.mktemp("tiny_h3"))
    video, audio = diffusers_reference(transformer)
    return SimpleNamespace(transformer=str(transformer), video=video, audio=audio)


def _run(tiny, *, tp, precomputed=True, exec_mode="eager"):
    return run_ranks(l1_h3_worker, tiny.transformer, precomputed, exec_mode, world_size=tp,
                     env=OFFLINE)


def _assert_close(results, tiny):
    for r in results:
        assert torch.equal(torch.tensor(r["video"]), torch.tensor(results[0]["video"]))
        torch.testing.assert_close(torch.tensor(r["video"]), tiny.video, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(torch.tensor(r["audio"]), tiny.audio, atol=ATOL, rtol=RTOL)
        assert r["other_differs"]


def test_config_reads_checkpoint_and_derives_static_shapes(tiny):
    config = NeuronMiniMaxH3Config.from_pretrained(
        tiny.transformer, **TINY_SHAPE, text_seq_len=TINY_TEXT_SEQ_LEN, tp_degree=2)
    assert config.hidden_size == TINY["hidden_size"]
    assert config.checkpoint_num_layers == 2
    assert (config.video_seq_len, config.video_patch_dim, config.audio_seq_len) == (37, 96, 414)
    config.validate()


def test_checkpoint_keys_and_swiglu_slices():
    assert checkpoint_key("transformer.transformer_blocks.3.ff.down_proj.weight") == (
        "transformer_blocks.3.ff.net.2.weight")
    assert checkpoint_key("transformer.proj_in.weight") == "proj_in.weight"
    config = NeuronMiniMaxH3Config(num_layers=1, num_refiner_layers=1, ffn_dim=8)
    slices = swiglu_slices(config)
    up = slices["transformer.transformer_blocks.0.ff.up_proj.weight"]
    gate = slices["transformer.transformer_blocks.0.ff.gate_proj.weight"]
    assert (up.key, up.dim, up.start, up.length) == ("transformer_blocks.0.ff.net.0.proj.weight",
                                                     0, 0, 8)
    assert (gate.start, gate.length) == (8, 8)
    assert len(slices) == 4  # one refiner block + one transformer block, two halves each


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_precomputed_adaln_matches_diffusers(tiny, tp):
    results = _run(tiny, tp=tp)
    _assert_close(results, tiny)
    r = results[0]
    assert r["params_on_meta"] == 0
    assert set(r["fp32"].values()) == {"torch.float32"}
    assert not r["has_adaln"]
    assert r["local_heads"] == TINY["num_attention_heads"] // tp
    assert r["load_report"]["missing"] == [] and r["load_report"]["unexpected"] == []
    assert len(r["load_report"]["skipped"]) == 2 * 2  # adaln weight + bias per block
    if tp > 1:
        assert r["processor"] == "MiniMaxH3TPAttnProcessor"


def test_resident_adaln_matches_diffusers(tiny):
    results = _run(tiny, tp=2, precomputed=False)
    _assert_close(results, tiny)
    assert results[0]["has_adaln"] and not results[0]["load_report"].get("skipped")


def test_compile_mode_matches_eager(tiny):
    results = _run(tiny, tp=2, exec_mode="compile")
    _assert_close(results, tiny)
    assert results[0]["compiled_blocks"]
