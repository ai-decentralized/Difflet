"""L1 unit tests (CPU only): the LTX-2 DiT as a neuron-backend application.

The model is the tiny diffusers-layout LTX-2 of tests/unit/backends/_neuron_ltx2_toy.py (2
blocks, 4 video heads x 8, 4 audio heads x 4, split RoPE, a sharded checkpoint with an index).
The reference is diffusers' ``LTX2VideoTransformer3DModel`` itself, reloaded from the same files
and called without text masks. Every forward runs in a spawned gloo process
(``_neuron_gloo.run_ranks``): at TP2 the column/row-parallel shards, the q/k RMS all-reduce and
the output all-reduces run, under CFG2 the batch scatter and gather. Nothing here touches the
device.
"""

import pytest

torch = pytest.importorskip("torch")

import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from difflet.backends.neuron.ltx_2.config import NeuronLtx2Config  # noqa: E402
from difflet.backends.neuron.ltx_2.transformer import (  # noqa: E402
    NeuronLtx2TransformerApplication,
    checkpoint_key,
)
from difflet.backends.neuron.ops_impl import parallel_mesh  # noqa: E402
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from tests.unit.backends._neuron_gloo import run_ranks  # noqa: E402
from tests.unit.backends._neuron_ltx2_toy import (  # noqa: E402
    LTX2_SNAPSHOT,
    TINY_AUDIO_SEQ_LEN,
    TINY_SHAPE,
    TINY_TEXT_SEQ_LEN,
    TINY_VIDEO_SEQ_LEN,
    checkpoint_keys,
    diffusers_reference,
    l1_ltx2_worker,
    l1_real_keys_worker,
    tiny_ltx2_config,
    write_tiny_ltx2_model,
)

TINY_BLOCK_NAMES = [f"transformer.transformer_blocks.{i}" for i in range(2)]
CLEAN_REPORT = {"missing": [], "unexpected": []}
ATOL = RTOL = 2e-5
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
ROPES = ("rope", "audio_rope", "cross_attn_rope", "cross_attn_audio_rope")


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
    root = tmp_path_factory.mktemp("tiny_ltx2")
    video, audio = write_tiny_ltx2_model(root)
    # At tp > 1 the shared TP patch rounds the RoPE tables to bf16 (as on Trainium), so the TP
    # runs are held to the reference with the same rounding; tp1 keeps diffusers' fp32 tables.
    video_tp, audio_tp = diffusers_reference(root / "transformer", bf16_rope=True)
    return SimpleNamespace(dir=str(root), transformer=root / "transformer", video=video,
                           audio=audio, video_tp=video_tp, audio_tp=audio_tp)


def _run(tiny, *, world_size=1, tp=1, exec_mode="eager", num_layers=None, cfg=False,
         cp_mode="gather_kv"):
    return run_ranks(l1_ltx2_worker, tiny.dir, tp, exec_mode, num_layers, cfg, cp_mode,
                     world_size=world_size, env=OFFLINE)


def _assert_replicated_and_close(results, video_ref, audio_ref):
    videos = [torch.tensor(r["video"]) for r in results]
    audios = [torch.tensor(r["audio"]) for r in results]
    for video, audio in zip(videos, audios):
        assert torch.equal(video, videos[0]) and torch.equal(audio, audios[0])
    torch.testing.assert_close(videos[0], video_ref, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(audios[0], audio_ref, atol=ATOL, rtol=RTOL)
    return videos[0], audios[0]


# ---- (a) config --------------------------------------------------------------------------


def test_config_from_pretrained_and_geometry(tiny):
    raw = json.loads((tiny.transformer / "config.json").read_text())
    assert "_class_name" in raw
    config = NeuronLtx2Config.from_pretrained(tiny.transformer, tp_degree=2,
                                              text_seq_len=TINY_TEXT_SEQ_LEN, **TINY_SHAPE)
    assert config.num_layers == 2 and config.checkpoint_num_layers == 2
    assert (config.latent_num_frames, config.latent_height, config.latent_width) == (3, 2, 3)
    assert config.video_seq_len == TINY_VIDEO_SEQ_LEN
    assert config.audio_num_frames == TINY_AUDIO_SEQ_LEN == config.audio_seq_len
    assert config.video_text_dim == config.audio_text_dim == 24
    assert config.rope_type == "split" and config.torch_dtype is torch.bfloat16


def test_config_defaults_are_ltx_2_and_match_the_trainium_geometry():
    default = NeuronLtx2Config(tp_degree=4)
    default.validate()
    assert (default.num_layers, default.num_attention_heads, default.attention_head_dim) == (
        48, 32, 128
    )
    # The registry's default shape: 512x768x121 -> 16 x 16 x 24 latent, 121/24 s of audio.
    assert default.video_seq_len == 16 * 16 * 24 and default.audio_num_frames == 126
    bench = NeuronLtx2Config(height=480, width=704, num_frames=49)
    assert bench.video_seq_len == 7 * 15 * 22 and bench.audio_seq_len == 51


def test_config_defaults_are_the_real_checkpoints():
    if not LTX2_SNAPSHOT.is_dir():
        pytest.skip("the real LTX-2 snapshot is not on this machine")
    default = NeuronLtx2Config(tp_degree=4)
    real = NeuronLtx2Config.from_pretrained(LTX2_SNAPSHOT / "transformer")
    for name in ("in_channels", "num_layers", "attention_head_dim", "num_attention_heads",
                 "cross_attention_dim", "audio_num_attention_heads", "audio_attention_head_dim",
                 "audio_cross_attention_dim", "caption_channels", "rope_type",
                 "rope_double_precision", "vae_scale_factors"):
        assert getattr(real, name) == getattr(default, name), name


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        ({"tp_degree": 3}, ValueError, "does not divide tp=3"),
        ({"height": 500}, ValueError, "multiple of 32"),
        ({"num_frames": 48}, ValueError, "8k\\+1"),
        ({"num_layers": 3, "checkpoint_num_layers": 2}, ValueError, "exceeds the checkpoint"),
        ({"gated_attn": True}, NotImplementedError, "gated attention"),
        ({"perturbed_attn": True}, NotImplementedError, "STG"),
        ({"dp_degree": 2}, NotImplementedError, "separate replicas"),
        ({"context_parallel_enabled": True, "cp_mode": "ring"}, NotImplementedError, "ring"),
        ({"context_parallel_enabled": True, "cp_degree": 4}, ValueError, "does not divide"),
        ({"context_parallel_enabled": True, "cp_degree": 2, "cfg_parallel_enabled": True},
         ValueError, "mutually exclusive"),
        ({"cp_degree": 2}, ValueError, "needs context_parallel_enabled"),
        ({"sp_enabled": True}, NotImplementedError, "sequence parallelism"),
    ],
)
def test_config_rejects(overrides, error, match):
    with pytest.raises(error, match=match):
        tiny_ltx2_config(**overrides).validate()


def test_application_refuses_a_disagreeing_parallel_config(tiny):
    config = tiny_ltx2_config(tp_degree=2)
    with pytest.raises(ValueError, match="tp_degree=2 but the parallel config has tp_degree=1"):
        NeuronLtx2TransformerApplication(model_path=tiny.transformer, config=config,
                                         parallel=DiffletParallelConfig(), dtype=torch.float32,
                                         device="cpu")
    with pytest.raises(ValueError, match="context_parallel_enabled=False but the parallel"):
        NeuronLtx2TransformerApplication(
            model_path=tiny.transformer, config=tiny_ltx2_config(tp_degree=2),
            parallel=DiffletParallelConfig(tp_degree=2, cp_degree=2), dtype=torch.float32,
            device="cpu")
    with pytest.raises(ValueError, match="cfg_parallel_enabled"):
        NeuronLtx2TransformerApplication(
            model_path=tiny.transformer, config=tiny_ltx2_config(),
            parallel=DiffletParallelConfig(cfg_parallel_enabled=True), dtype=torch.float32,
            device="cpu")


def test_checkpoint_key_strips_the_wrapper_prefix():
    assert checkpoint_key("transformer.transformer_blocks.0.attn1.to_q.weight") == (
        "transformer_blocks.0.attn1.to_q.weight")
    assert checkpoint_key("transformer.proj_out.bias") == "proj_out.bias"


# ---- (b) tp1 / tp2 against diffusers, eager and compiled -----------------------------------


def test_tp1_eager_matches_diffusers(tiny):
    [result] = _run(tiny)
    _assert_replicated_and_close([result], tiny.video, tiny.audio)
    assert result["load_report"] == CLEAN_REPORT
    assert result["params_on_meta"] == 0 and result["compiled_blocks"] == []
    assert result["rope_types"] == {attr: "_HostRope" for attr in ROPES}
    # The feed-forward GELU runs in fp32 on this backend (the device's bf16 tanh GELU is less
    # accurate than the CPU's; _FP32_GELU), with diffusers' parameters and keys.
    assert result["ff_activations"] == ["Fp32GELU"]
    # Warm-up, three forwards on the same coordinates: each table computed once.
    assert result["rope_computed"] == {attr: 1 for attr in ROPES}
    assert result["other_differs"] and result["again_equal"]


def test_tp2_eager_matches_diffusers_on_every_rank(tiny):
    results = _run(tiny, world_size=2, tp=2)
    _assert_replicated_and_close(results, tiny.video_tp, tiny.audio_tp)
    # The bf16 RoPE rounding is the only difference from diffusers' fp32 tables.
    assert not torch.allclose(torch.tensor(results[0]["video"]), tiny.video, atol=ATOL, rtol=RTOL)
    for result in results:
        assert result["mesh"] == "MeshSpec(dp=1, cfg=1, cp=1, tp=2)"
        assert result["processor"] == "_LTX2TPAttnProcessor"
        assert result["local_heads"] == {name: 2 for name in result["local_heads"]}
        assert result["load_report"] == CLEAN_REPORT
    # Every rank holds half the attention/FFN shards plus the replicated rest.
    [tp1] = _run(tiny)
    assert results[0]["param_numel"] < tp1["param_numel"]
    assert results[0]["weight_bytes"] < tp1["weight_bytes"]


def test_tp2_compile_is_one_graph_and_equals_eager(tiny):
    eager = _run(tiny, world_size=2, tp=2)
    compiled = _run(tiny, world_size=2, tp=2, exec_mode="compile")
    for e, c in zip(eager, compiled):
        assert c["compiled_blocks"] == TINY_BLOCK_NAMES
        assert c["unique_graphs_after_warmup"] == 1 == c["unique_graphs_after_forwards"]
        assert c["graph_breaks"] == 0 and c["unwarmed_shapes"] == []
        # The block's collectives are in the graph: q/k RMS (12), attention and FFN outputs (8).
        assert c["collectives_per_graph"] == [20]
        torch.testing.assert_close(torch.tensor(c["video"]), torch.tensor(e["video"]),
                                   atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(torch.tensor(c["audio"]), torch.tensor(e["audio"]),
                                   atol=ATOL, rtol=RTOL)
    _assert_replicated_and_close(compiled, tiny.video_tp, tiny.audio_tp)


def test_num_layers_override_loads_the_first_blocks(tiny):
    [result] = _run(tiny, num_layers=1)
    assert result["num_blocks"] == 1
    assert result["load_report"]["missing"] == [] and result["load_report"]["unexpected"] == []
    assert result["load_report"]["skipped"] and all(
        key.startswith("transformer_blocks.1.") for key in result["load_report"]["skipped"])
    video, audio = diffusers_reference(tiny.transformer, num_layers=1)
    _assert_replicated_and_close([result], video, audio)


# ---- (c) context and CFG parallelism -------------------------------------------------------


@pytest.mark.parametrize(("tp", "cp_mode"), [(1, "gather_kv"), (1, "ulysses"),
                                              (2, "gather_kv"), (2, "ulysses")])
def test_context_parallel_shards_the_video_tokens(tiny, tp, cp_mode):
    """CP2 over the 18 video tokens: each rank's blocks get 9 video tokens and the whole audio
    stream; the gathered output matches diffusers (with the TP path's bf16 RoPE at tp2)."""
    world = 2 * tp
    results = _run(tiny, world_size=world, tp=tp, cp_mode=cp_mode)
    video_ref, audio_ref = (tiny.video, tiny.audio) if tp == 1 else (tiny.video_tp, tiny.audio_tp)
    _assert_replicated_and_close(results, video_ref, audio_ref)
    mode = "ulysses" if cp_mode == "ulysses" else "gather"
    for result in results:
        assert result["block_tokens"] == [[TINY_VIDEO_SEQ_LEN // 2, TINY_AUDIO_SEQ_LEN]] * 3
        assert result["block_rows"] == [[1, TINY_VIDEO_SEQ_LEN // 2]] * 3
        assert result["kv_modes"] == {"attn1": mode, "video_to_audio_attn": "gather",
                                      "audio_attn1": None, "attn2": None, "audio_attn2": None,
                                      "audio_to_video_attn": None}


def test_context_parallel_compiles_one_graph(tiny):
    eager = _run(tiny, world_size=4, tp=2)
    compiled = _run(tiny, world_size=4, tp=2, exec_mode="compile")
    for e, c in zip(eager, compiled):
        assert c["unique_graphs_after_warmup"] == 1 == c["unique_graphs_after_forwards"]
        assert c["graph_breaks"] == 0
        torch.testing.assert_close(torch.tensor(c["video"]), torch.tensor(e["video"]),
                                   atol=ATOL, rtol=RTOL)


def test_cfg_parallel_splits_the_pair_and_gathers_both_branches(tiny):
    results = _run(tiny, world_size=2, tp=1, cfg=True)
    video, audio = diffusers_reference(tiny.transformer, batch=2)
    _assert_replicated_and_close(results, video, audio)
    assert results[0]["warmup_shapes"][0] == (2, TINY_VIDEO_SEQ_LEN, 128)
    # Review T3: the module splits the pair, so each rank's blocks see one branch (batch 1) of
    # all the video tokens; a module running both branches on every rank would also match.
    for result in results:
        assert result["block_rows"] == [[1, TINY_VIDEO_SEQ_LEN]] * 3


# ---- (d) the real checkpoint's keys --------------------------------------------------------


@pytest.mark.skipif(not LTX2_SNAPSHOT.is_dir(), reason="LTX-2 snapshot not on this host")
def test_real_checkpoint_keys_are_covered_at_tp4():
    transformer = LTX2_SNAPSHOT / "transformer"
    results = run_ranks(l1_real_keys_worker, str(transformer), world_size=4, env=OFFLINE,
                        timeout=600.0)
    assert set(results[0]["keys"]) == checkpoint_keys(transformer)
    shards = results[0]["shards"]
    prefix = "transformer.transformer_blocks.0."
    assert shards[prefix + "attn1.to_q.weight"][:2] == [0, [1024, 4096]]
    assert shards[prefix + "attn1.to_out.0.weight"][:2] == [1, [4096, 1024]]
    assert shards[prefix + "audio_to_video_attn.to_k.weight"][:2] == [0, [512, 2048]]
    assert shards[prefix + "video_to_audio_attn.to_v.bias"][:2] == [0, [512]]
    assert shards[prefix + "ff.net.0.proj.weight"][:2] == [0, [4096, 4096]]
    assert shards[prefix + "audio_ff.net.2.weight"][:2] == [1, [2048, 2048]]
    # Replicated: the q/k RMS weight (sliced per forward), the modulation tables, the top level.
    assert shards[prefix + "attn1.norm_q.weight"][:2] == [None, [4096]]
    assert shards["transformer.proj_in.weight"][:2] == [None, [4096, 128]]
    assert results[0]["video_seq_len"] == 2310 and results[0]["audio_seq_len"] == 51
    # Every rank's bf16 share: about 10 GiB, under a 24 GiB core.
    assert 9 * 2**30 < results[0]["weight_bytes"] < 11 * 2**30
