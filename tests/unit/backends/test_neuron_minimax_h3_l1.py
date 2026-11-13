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
    text_bucket,
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


def _run(tiny, *, tp, precomputed=True, exec_mode="eager", cp=1, cp_mode="gather_kv"):
    return run_ranks(l1_h3_worker, tiny.transformer, precomputed, exec_mode, cp, cp_mode,
                     world_size=tp * cp, env=OFFLINE)


def _assert_close(results, tiny):
    for r in results:
        assert r["wide_max_abs"] < 1e-5, r["wide_max_abs"]
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


@pytest.mark.parametrize(
    "tp, cp, mode",
    [
        (2, 2, "ulysses"),  # 4 heads -> 2 per TP rank -> 1 per Ulysses rank
        (1, 2, "ulysses"),
        (2, 2, "gather_kv"),
        (1, 4, "gather_kv"),
        (1, 3, "gather_kv"),  # 3 does not divide the packed sequence: tail padding path
    ],
)
def test_context_parallel_matches_diffusers(tiny, tp, cp, mode):
    """The block stack sequence-sharded over the CP group (Ulysses all-to-all, or local
    queries against all-gathered K/V) gives the cp1 result on every rank."""
    results = _run(tiny, tp=tp, cp=cp, cp_mode=mode)
    _assert_close(results, tiny)
    r = results[0]
    assert r["processor"] == "_NeuronH3UlyssesAttnProcessor" and r["cp_mode"] == mode
    assert r["local_heads"] == TINY["num_attention_heads"] // tp


def test_context_parallel_compile_mode_matches_eager(tiny):
    results = _run(tiny, tp=2, cp=2, cp_mode="ulysses", exec_mode="compile")
    _assert_close(results, tiny)
    assert results[0]["compiled_blocks"]


def test_ulysses_refuses_heads_that_do_not_divide_tp_times_cp(tiny):
    # 4 heads over tp2 x cp4 = 8: Ulysses cannot shard them (gather_kv can).
    with pytest.raises(AssertionError, match="divisible by tp\\*cp"):
        _run(tiny, tp=2, cp=4, cp_mode="ulysses")


def test_text_bucket_is_the_smallest_that_holds_the_prompt():
    assert [text_bucket(n) for n in (1, 27, 128, 129, 600, 1024)] == [128, 128, 128, 256,
                                                                       1024, 1024]
    with pytest.raises(ValueError):
        text_bucket(1025)


def test_neuron_h3_attends_with_the_scalar_engine_exp_unless_configured():
    """The neuron trace module binds vector_exp=False into every block's attention processor
    (config.attention_vector_exp=False, the default); True keeps the plain difflet.ops.attention."""
    import functools

    from difflet.backends.neuron.minimax_h3.transformer import (
        NeuronMiniMaxH3Config,
        _trace_module_class,
    )

    def processors(module):
        tr = module.transformer
        return [b.attn.processor for b in (*tr.token_refiner.refiner_blocks, *tr.transformer_blocks)]

    tiny = dict(num_attention_heads=2, attention_head_dim=96, hidden_size=192, num_layers=2,
                num_refiner_layers=1, ffn_dim=256, text_dim=64, time_embed_hidden_dim=64,
                time_embed_dim=32, height=64, width=64, num_frames=124, text_seq_len=16,
                tp_degree=1, precomputed_adaln=False)
    with torch.device("meta"):
        scalar = _trace_module_class()(NeuronMiniMaxH3Config(**tiny))
        vector = _trace_module_class()(NeuronMiniMaxH3Config(**tiny, attention_vector_exp=True))
    for p in processors(scalar):
        assert isinstance(p._attention, functools.partial)
        assert p._attention.keywords == {"vector_exp": False}
    for p in processors(vector):
        assert not isinstance(p._attention, functools.partial)
