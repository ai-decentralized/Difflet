"""Q1 unit tests (CPU only): the Qwen-Image DiT as a neuron-backend application.

The model is the tiny diffusers-layout Qwen-Image of tests/unit/backends/_neuron_qwen_toy.py
(2 blocks, 4 heads x 32, a sharded checkpoint with an index). The reference is diffusers'
``QwenImageTransformer2DModel`` itself, reloaded from the same files and called with
``encoder_hidden_states_mask=None``: the unmasked joint attention the Difflet modeling runs on
every backend. Every forward runs in a spawned gloo process (``_neuron_gloo.run_ranks``, one rank
for the single-rank cases): at TP2 the column/row-parallel shards and their all-gathers and
all-reduces run, at CP2 the image-token scatter, the K/V all-gather and the output gather. Nothing
here touches the device.
"""

import pytest

# importorskip first, as in test_neuron_wan_w1.py: it keeps torch_neuronx's import-time
# nki.jit DeprecationWarning out of the report when this module is collected first.
torch = pytest.importorskip("torch")

import json  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from difflet.backends.neuron import runtime  # noqa: E402
from difflet.backends.neuron.ops_impl import parallel_mesh  # noqa: E402
from difflet.backends.neuron.qwen_image.config import NeuronQwenImageConfig  # noqa: E402
from difflet.backends.neuron.qwen_image.transformer import (  # noqa: E402
    NeuronQwenImageTransformerApplication,
    checkpoint_key,
)
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from tests.unit.backends._neuron_gloo import run_ranks  # noqa: E402
from tests.unit.backends._neuron_qwen_toy import (  # noqa: E402
    QWEN_SNAPSHOT,
    TINY_HIDDEN_SHAPE,
    TINY_IMAGE_SEQ_LEN,
    TINY_OUT_SHAPE,
    TINY_QWEN_TRANSFORMER,
    TINY_SHAPE,
    TINY_TEXT_SEQ_LEN,
    TINY_TEXT_SHAPE,
    checkpoint_keys,
    diffusers_reference,
    q1_qwen_worker,
    q1_real_keys_worker,
    spec_mesh,
    tiny_qwen_config,
    write_tiny_qwen_model,
)

LOAD_PHASES = {
    "runtime init", "build on meta", "load checkpoint", "eval", "compile", "warmup forward"
}
TINY_BLOCK_NAMES = [f"transformer.transformer_blocks.{i}" for i in range(2)]
WARMUP_SHAPES = [TINY_HIDDEN_SHAPE, (1,), TINY_TEXT_SHAPE]
CLEAN_REPORT = {"missing": [], "unexpected": []}
# Tight fp32: the same math as diffusers, except real-valued RoPE for its complex product.
ATOL = RTOL = 1e-5
# No rank may reach for the Hub: every model file is local.
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
    root = tmp_path_factory.mktemp("tiny_qwen")
    ref = write_tiny_qwen_model(root)
    return SimpleNamespace(dir=str(root), transformer=root / "transformer", ref=ref)


def _run(tiny, *, world_size=1, tp=1, cp_mode="gather_kv", exec_mode="eager", num_layers=None,
         model_dir=None, sp=False):
    return run_ranks(q1_qwen_worker, model_dir or tiny.dir, tp, cp_mode, exec_mode, num_layers,
                     sp, world_size=world_size, env=OFFLINE)


def _assert_replicated_and_close(results, reference):
    outs = [torch.tensor(result["out"]) for result in results]
    for out in outs:
        assert tuple(out.shape) == TINY_OUT_SHAPE
        assert torch.equal(out, outs[0])  # every rank holds the same (gathered/reduced) output
    torch.testing.assert_close(outs[0], reference, atol=ATOL, rtol=RTOL)
    return outs[0]


# ---- (a) config --------------------------------------------------------------------------


def test_config_from_pretrained_ignores_unknown_keys(tiny):
    raw = json.loads((tiny.transformer / "config.json").read_text())
    assert raw["pooled_projection_dim"] == 768 and "_class_name" in raw  # really on disk
    config = NeuronQwenImageConfig.from_pretrained(
        tiny.transformer, tp_degree=2, text_seq_len=TINY_TEXT_SEQ_LEN, **TINY_SHAPE
    )
    assert config.axes_dims_rope == (8, 12, 12) and config.num_layers == 2
    assert config.checkpoint_num_layers == 2 and config.out_channels == 4
    assert (config.latent_height, config.latent_width) == (8, 12)
    assert (config.packed_height, config.packed_width) == (4, 6)
    assert config.image_seq_len == TINY_IMAGE_SEQ_LEN
    assert config.torch_dtype is torch.bfloat16  # the dtype is the caller's, not the file's
    # out_channels defaults to in_channels // 4 when the file leaves it null.
    (tiny.transformer.parent / "nulled").mkdir(exist_ok=True)
    (tiny.transformer.parent / "nulled" / "config.json").write_text(
        json.dumps({**TINY_QWEN_TRANSFORMER, "out_channels": None})
    )
    assert NeuronQwenImageConfig.from_pretrained(tiny.transformer.parent / "nulled",
                                                 **TINY_SHAPE).out_channels == 4


def test_config_defaults_are_qwen_image():
    default = NeuronQwenImageConfig(tp_degree=4)
    default.validate()
    assert (default.num_layers, default.num_attention_heads, default.attention_head_dim) == (
        60, 24, 128
    )
    assert default.image_seq_len == 4096 and default.text_seq_len == 1024
    if QWEN_SNAPSHOT.is_dir():
        real = NeuronQwenImageConfig.from_pretrained(QWEN_SNAPSHOT / "transformer")
        for name in ("patch_size", "in_channels", "out_channels", "num_layers",
                     "attention_head_dim", "num_attention_heads", "joint_attention_dim",
                     "guidance_embeds", "axes_dims_rope"):
            assert getattr(real, name) == getattr(default, name), name


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        ({"tp_degree": 3}, ValueError, "does not divide tp=3"),
        ({"axes_dims_rope": (8, 12, 10)}, ValueError, "must sum to attention_head_dim"),
        ({"axes_dims_rope": (7, 13, 12)}, ValueError, "must be even"),
        ({"patch_size": 1}, NotImplementedError, "patch_size=2"),
        ({"guidance_embeds": True}, NotImplementedError, "guidance_embeds=True"),
        ({"height": 72}, ValueError, "height=72 must be a positive multiple of 16"),
        ({"width": 100}, ValueError, "width=100 must be a positive multiple of 16"),
        ({"dp_degree": 2}, NotImplementedError, "dp_degree=2"),
        # SP shards the packed image tokens and the padded text rows over tp: 24 and 8 rows.
        ({"sp_enabled": True, "tp_degree": 4, "height": 48}, ValueError,
         r"image_seq_len=18 \(3x6 packed\) over tp=4"),
        ({"sp_enabled": True, "tp_degree": 4, "text_seq_len": 6}, ValueError,
         "text_seq_len=6 over tp=4"),
        ({"sp_enabled": True, "context_parallel_enabled": True}, ValueError,
         "sp_enabled and context_parallel_enabled are mutually exclusive"),
        ({"sp_enabled": True, "zero_cond_t": True}, NotImplementedError, "zero_cond_t"),
        ({"cfg_parallel_enabled": True, "context_parallel_enabled": True}, ValueError,
         "cfg_parallel_enabled and context_parallel_enabled are mutually exclusive"),
        ({"cp_mode": "tree"}, ValueError, "cp_mode must be one of"),
        ({"cp_degree": 2}, ValueError, "needs context_parallel_enabled"),
        # 4 heads / tp 2 = 2 local heads do not split over cp 4.
        ({"context_parallel_enabled": True, "cp_mode": "ulysses", "tp_degree": 2,
          "cp_degree": 4}, ValueError, "num_attention_heads must be divisible by tp_degree"),
        # 64x80 -> packed 4x5 = 20 image tokens, not divisible by cp 8.
        ({"context_parallel_enabled": True, "cp_degree": 8, "width": 80}, ValueError,
         "image_seq_len=20"),
        ({"num_layers": 0}, ValueError, "num_layers=0"),
        ({"num_layers": 3, "checkpoint_num_layers": 2}, ValueError, "exceeds the checkpoint"),
    ],
)
def test_config_validate_rejects(overrides, error, match):
    with pytest.raises(error, match=match):
        tiny_qwen_config(**overrides).validate()


def test_config_validate_accepts_supported_topologies():
    for overrides in (
        {"tp_degree": 4},
        {"context_parallel_enabled": True, "cp_mode": "gather_kv", "cp_degree": 2},
        {"context_parallel_enabled": True, "cp_mode": "ring", "tp_degree": 2, "cp_degree": 2},
        {"context_parallel_enabled": True, "cp_mode": "ulysses", "tp_degree": 2, "cp_degree": 2},
        # cp unknown: the cp checks wait for the application (or the mesh).
        {"context_parallel_enabled": True, "cp_mode": "ulysses", "tp_degree": 4},
        {"num_layers": 1, "checkpoint_num_layers": 2},
        {"sp_enabled": True, "tp_degree": 2},
        {"sp_enabled": True, "tp_degree": 4},
        {"cfg_parallel_enabled": True, "tp_degree": 2},
        {"cfg_parallel_enabled": True, "sp_enabled": True, "tp_degree": 2},
        # tp1: the fork is the dense model, so nothing has to divide.
        {"sp_enabled": True, "text_seq_len": 7},
    ):
        tiny_qwen_config(**overrides).validate()


# ---- the application's guards --------------------------------------------------------------


def test_application_guards_topology_and_dtype(tiny):
    def make(config, parallel=None, dtype=torch.float32, device="cpu"):
        return NeuronQwenImageTransformerApplication(
            model_path=tiny.transformer, config=config, parallel=parallel, dtype=dtype,
            device=device,
        )

    with pytest.raises(ValueError, match="tp_degree=2.*tp_degree=1"):
        make(tiny_qwen_config(tp_degree=2), DiffletParallelConfig(tp_degree=1))
    # CFG parallelism changes the call's batch (the [negative, conditional] pair): the config
    # and the parallel config must agree on it.
    with pytest.raises(ValueError, match="cfg_parallel_enabled=False.*cfg_parallel_enabled=True"):
        make(tiny_qwen_config(tp_degree=2),
             DiffletParallelConfig(tp_degree=2, cfg_parallel_enabled=True))
    # The mesh carries no SP flag, so the config and the parallel config must agree on it.
    with pytest.raises(ValueError, match="sp_enabled=False.*sp_enabled=True"):
        make(tiny_qwen_config(tp_degree=2), DiffletParallelConfig(tp_degree=2, sp_enabled=True))
    with pytest.raises(ValueError, match="sp_enabled=True.*sp_enabled=False"):
        make(tiny_qwen_config(tp_degree=2, sp_enabled=True), DiffletParallelConfig(tp_degree=2))
    assert make(tiny_qwen_config(tp_degree=2, sp_enabled=True),
                DiffletParallelConfig(tp_degree=2, sp_enabled=True)).config.sp_enabled
    with pytest.raises(ValueError, match="context_parallel_enabled=False.*cp_degree=2"):
        make(tiny_qwen_config(), DiffletParallelConfig(cp_degree=2))
    with pytest.raises(ValueError, match="context_parallel_enabled=True.*cp_degree=1"):
        make(tiny_qwen_config(context_parallel_enabled=True), DiffletParallelConfig())
    with pytest.raises(ValueError, match="cp_mode='ring'.*cp_mode='gather_kv'"):
        make(tiny_qwen_config(context_parallel_enabled=True, cp_mode="ring"),
             DiffletParallelConfig(cp_degree=2))
    with pytest.raises(ValueError, match="cp_degree=4.*cp_degree=2"):
        make(tiny_qwen_config(context_parallel_enabled=True, cp_degree=4),
             DiffletParallelConfig(cp_degree=2))
    # cp_degree comes from the parallel config, so validate() sees the ulysses head split.
    with pytest.raises(ValueError, match="num_attention_heads must be divisible"):
        make(tiny_qwen_config(context_parallel_enabled=True, cp_mode="ulysses", tp_degree=4),
             DiffletParallelConfig(tp_degree=4, cp_degree=2, cp_mode="ulysses"))
    config = tiny_qwen_config(context_parallel_enabled=True)
    app = make(config, DiffletParallelConfig(cp_degree=2))
    assert app.config.cp_degree == 2 and config.cp_degree is None  # filled on a copy
    # Decision D42: bf16 is the only dtype on the device.
    with pytest.raises(ValueError, match="bfloat16"):
        make(tiny_qwen_config(), device="neuron")
    app = make(tiny_qwen_config(), dtype=torch.bfloat16, device="neuron")
    assert app.device.type == "neuron" and app.is_loaded is False
    assert app.block_attrs == ("transformer.transformer_blocks",)
    assert app.skipped_checkpoint_prefixes == ()
    assert runtime._neuron_runtime_initialized() is False  # constructing touched nothing


def test_example_inputs_are_the_per_step_shapes(tiny):
    app = NeuronQwenImageTransformerApplication(
        model_path=tiny.transformer, config=tiny_qwen_config(), dtype=torch.float32, device="cpu"
    )
    assert [(tuple(t.shape), t.dtype) for t in app.get_example_inputs()] == [
        (shape, torch.float32) for shape in WARMUP_SHAPES
    ]
    # The real geometry at 1024x1024: 64x64 packed tokens, 1024 padded text rows.
    real = NeuronQwenImageTransformerApplication(
        model_path="/unused", config=NeuronQwenImageConfig(), dtype=torch.bfloat16, device="cpu"
    )
    assert [(tuple(t.shape), t.dtype) for t in real.get_example_inputs()] == [
        ((1, 4096, 64), torch.bfloat16), ((1,), torch.bfloat16), ((1, 1024, 3584), torch.bfloat16)
    ]


# ---- (b) checkpoint keys -------------------------------------------------------------------


def test_checkpoint_key_strips_only_the_wrapper_prefix():
    for name in (
        "transformer_blocks.0.img_mod.1.weight",
        "transformer_blocks.59.txt_mod.1.bias",
        "transformer_blocks.3.img_mlp.net.0.proj.weight",
        "transformer_blocks.3.txt_mlp.net.2.bias",
        "transformer_blocks.0.attn.to_out.0.weight",
        "transformer_blocks.0.attn.to_add_out.bias",
        "transformer_blocks.0.attn.norm_added_k.weight",
        "norm_out.linear.weight",
        "time_text_embed.timestep_embedder.linear_1.bias",
        "img_in.weight",
        "txt_norm.weight",
        "proj_out.bias",
    ):
        assert checkpoint_key("transformer." + name) == name


def test_checkpoint_key_covers_the_module_and_the_tiny_checkpoint(tiny):
    from difflet.backends.neuron.core.checkpoint import build_on_meta

    app = NeuronQwenImageTransformerApplication(
        model_path=tiny.transformer, config=tiny_qwen_config(), dtype=torch.float32, device="cpu"
    )
    with spec_mesh(tp=1):
        module = build_on_meta(app.build_module)
    names = list(module.state_dict())
    keys = [app.checkpoint_key(n) for n in names]
    assert len(set(keys)) == len(keys)  # one source per parameter
    assert set(keys) == checkpoint_keys(tiny.transformer)  # no missing, no unexpected
    # Every parameter is in the state dict; the static RoPE tables are not (computed, real).
    assert {n for n, _ in module.named_parameters()} == set(names)
    assert all(p.is_meta for p in module.parameters())
    buffers = dict(module.named_buffers())
    assert sorted(buffers) == [
        "transformer.pos_embed.img_cos_4x6", "transformer.pos_embed.img_sin_4x6",
        "transformer.pos_embed.txt_cos", "transformer.pos_embed.txt_sin",
    ]
    assert not any(b.is_meta for b in buffers.values())
    assert tuple(buffers["transformer.pos_embed.img_cos_4x6"].shape) == (TINY_IMAGE_SEQ_LEN, 32)


@pytest.mark.skipif(not QWEN_SNAPSHOT.is_dir(), reason="the Qwen-Image snapshot is not cached")
def test_checkpoint_key_matches_the_real_qwen_image_index_at_tp2():
    """The real 60-block DiT built on meta at TP2 maps onto exactly the 1933 keys of the
    snapshot's index, TP-replaced linears included; only the index json is read."""
    transformer_dir = QWEN_SNAPSHOT / "transformer"
    index = json.loads(
        (transformer_dir / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )
    results = run_ranks(q1_real_keys_worker, str(transformer_dir), world_size=2, env=OFFLINE)
    keys = results[0]["keys"]
    assert len(keys) == len(set(keys)) == results[1]["num_names"] == 1933
    assert set(keys) == set(index["weight_map"])
    shards = results[0]["shards"]
    prefix = "transformer.transformer_blocks.0."
    # Column-parallel: output rows split (dim 0); row-parallel: input columns split (dim 1).
    assert shards[prefix + "img_mod.1.weight"] == [0, [3072 * 6 // 2, 3072], "ColumnParallelLinear"]
    assert shards[prefix + "txt_mod.1.weight"][0] == 0
    assert shards[prefix + "img_mlp.net.0.proj.weight"] == [0, [12288 // 2, 3072],
                                                             "ColumnParallelLinear"]
    assert shards[prefix + "img_mlp.net.2.weight"] == [1, [3072, 12288 // 2], "RowParallelLinear"]
    assert shards[prefix + "attn.to_q.weight"] == [0, [1536, 3072], "ColumnParallelLinear"]
    assert shards[prefix + "attn.to_out.0.weight"] == [1, [3072, 1536], "RowParallelLinear"]
    assert shards[prefix + "attn.to_add_out.weight"][0] == 1
    assert shards["transformer.norm_out.linear.weight"][0] == 0
    assert shards["transformer.time_text_embed.timestep_embedder.linear_1.weight"][0] == 0
    assert shards["transformer.img_in.weight"] == [0, [1536, 64], "ColumnParallelLinear"]
    assert shards["transformer.proj_out.weight"] == [None, [64, 3072], "Linear"]  # replicated
    # The static RoPE stays real (fp32, host) on the meta build: 64x64 image rows, 1024 text.
    assert results[0]["buffers"] == {
        f"transformer.pos_embed.{name}": ["cpu", "torch.float32"]
        for name in ("img_cos_64x64", "img_sin_64x64", "txt_cos", "txt_sin")
    }
    assert results[0]["image_seq_len"] == 4096


# ---- (c) single rank against diffusers --------------------------------------------------


def test_single_rank_eager_matches_diffusers(tiny):
    (result,) = _run(tiny)
    _assert_replicated_and_close([result], tiny.ref)
    assert not torch.equal(tiny.ref, torch.zeros_like(tiny.ref))
    assert result["load_report"] == CLEAN_REPORT
    assert result["warmup_shapes"] == WARMUP_SHAPES
    assert result["compiled_blocks"] == [] and result["unwarmed_shapes"] == []
    assert result["mesh"] == str(DiffletParallelConfig().mesh_spec)
    assert result["local_heads"] == 4 and result["num_blocks"] == 2
    assert result["processor"] == ["_QwenImageTrainiumAttnProcessor", False, "gather_kv"]
    assert result["params_on_meta"] == 0
    assert result["five_arg_equal"] is True  # the mask is dropped, as documented
    assert result["other_differs"] is True


# ---- (d) TP2 ---------------------------------------------------------------------------------


@pytest.mark.parametrize("exec_mode", ["eager", "compile"])
def test_tp2_matches_tp1_and_diffusers(tiny, exec_mode):
    (single,) = _run(tiny)
    results = _run(tiny, world_size=2, tp=2, exec_mode=exec_mode)
    out = _assert_replicated_and_close(results, tiny.ref)
    torch.testing.assert_close(out, torch.tensor(single["out"]), atol=ATOL, rtol=RTOL)
    for rank, result in enumerate(results):
        assert (result["rank"], result["world_size"]) == (rank, 2)
        assert result["load_report"] == CLEAN_REPORT
        assert result["local_heads"] == 2  # 4 heads over 2 ranks
        assert result["param_numel"] < single["param_numel"]  # sharded, not two TP1 copies
        assert result["unwarmed_shapes"] == []
        if exec_mode == "compile":
            assert result["compiled_blocks"] == TINY_BLOCK_NAMES
            # Both blocks share one graph, and the second forward's values reuse it.
            assert result["unique_graphs_after_warmup"] == 1
            assert result["unique_graphs_after_forwards"] == 1
            assert result["graph_breaks"] == 0
            # The TP collectives run inside the compiled block: the img/txt modulation
            # all-gathers, both attention out-projections' and both MLPs' all-reduces.
            assert result["collectives_per_graph"] == [6]
        assert result["five_arg_equal"] is True


# ---- (d2) TP2 with sequence parallelism ------------------------------------------------------


@pytest.mark.parametrize("exec_mode", ["eager", "compile"])
def test_tp2_sequence_parallel_matches_tp2_and_tp1(tiny, exec_mode):
    """Megatron SP over the TP group (the QwenImageSPTransformer2DModel fork): both residual
    streams sharded between the blocks, all-gather in and reduce-scatter out of every TP region.
    fp32: equal to TP2 without SP, TP1 and diffusers, and identical on every rank."""
    (single,) = _run(tiny)
    dense = _run(tiny, world_size=2, tp=2, exec_mode=exec_mode)
    results = _run(tiny, world_size=2, tp=2, exec_mode=exec_mode, sp=True)
    out = _assert_replicated_and_close(results, tiny.ref)
    torch.testing.assert_close(out, torch.tensor(single["out"]), atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(out, torch.tensor(dense[0]["out"]), atol=ATOL, rtol=RTOL)
    assert dense[0]["transformer_class"] == "QwenImageTransformer2DModel"
    for result in results:
        assert result["transformer_class"] == "QwenImageSPTransformer2DModel"
        assert result["sp_active"] is True  # the fork saw tp=2 (it falls back to dense at tp1)
        assert result["load_report"] == CLEAN_REPORT  # same keys as diffusers' class
        assert result["local_heads"] == 2
        assert result["warmup_shapes"] == WARMUP_SHAPES  # the shard is taken inside the module
        assert result["unwarmed_shapes"] == []
        assert result["five_arg_equal"] is True and result["other_differs"] is True
        if exec_mode == "eager":
            # Every forward's first block sees this rank's half of the 24 image tokens and of
            # the 8 text rows.
            assert result["block_rows"] == [[12, 4]] * 3
        else:
            assert result["compiled_blocks"] == TINY_BLOCK_NAMES
            assert result["unique_graphs_after_warmup"] == 1
            assert result["unique_graphs_after_forwards"] == 1
            assert result["graph_breaks"] == 0
            # The two modulation all-gathers, then per TP region (joint attention, image MLP,
            # text MLP) an all-gather of each stream entering it and a reduce-scatter of each
            # leaving it: 2 + (2 + 2) + (1 + 1) + (1 + 1), against TP2's 2 + 4 all-reduces.
            assert result["collectives_per_graph"] == [10]


# ---- (e) CP2 ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("world_size", "tp", "exec_mode"), [(2, 1, "eager"), (2, 1, "compile"), (4, 2, "eager")]
)
def test_cp2_gather_kv_matches_tp1(tiny, world_size, tp, exec_mode):
    """The image tokens and their RoPE rows scattered over the CP group, image K/V all-gathered,
    the output gathered back: equal to the unsharded run on every rank."""
    (single,) = _run(tiny)
    results = _run(tiny, world_size=world_size, tp=tp, cp_mode="gather_kv", exec_mode=exec_mode)
    out = _assert_replicated_and_close(results, tiny.ref)
    torch.testing.assert_close(out, torch.tensor(single["out"]), atol=ATOL, rtol=RTOL)
    for result in results:
        # The trace module's own init_parallel_mesh(config) agreed with the base's spec.
        expected = DiffletParallelConfig(tp_degree=tp, cp_degree=2).mesh_spec
        assert result["mesh"] == str(expected)
        assert result["processor"] == ["_QwenImageTrainiumAttnProcessor", True, "gather_kv"]
        assert result["local_heads"] == 4 // tp
        assert result["load_report"] == CLEAN_REPORT  # SPMDRank holds no checkpoint state
        assert result["warmup_shapes"] == WARMUP_SHAPES  # every rank gets the full sequence
        assert result["unwarmed_shapes"] == []
        if exec_mode == "compile":
            # One graph for both blocks; its only collective is the image K/V all-gather (the
            # token scatter is a local narrow and the output gather is outside the blocks).
            assert result["compiled_blocks"] == TINY_BLOCK_NAMES
            assert result["unique_graphs_after_forwards"] == 1
            assert result["collectives_per_graph"] == [1]


# ---- (f) compile mode ------------------------------------------------------------------------


def test_compile_mode_matches_eager(tiny):
    (eager,) = _run(tiny)
    (compiled,) = _run(tiny, exec_mode="compile")
    torch.testing.assert_close(
        torch.tensor(compiled["out"]), torch.tensor(eager["out"]), atol=ATOL, rtol=RTOL
    )
    assert compiled["compiled_blocks"] == TINY_BLOCK_NAMES
    assert compiled["unique_graphs_after_warmup"] == 1
    assert compiled["unique_graphs_after_forwards"] == 1
    assert compiled["graph_breaks"] == 0
    assert compiled["unwarmed_shapes"] == []
    assert compiled["collectives_per_graph"] == [0]  # tp=1: the layers skip every collective
    assert compiled["load_report"] == CLEAN_REPORT


# ---- (g) num_layers override -----------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny3(tmp_path_factory):
    pytest.importorskip("safetensors.torch")
    pytest.importorskip("diffusers")
    root = tmp_path_factory.mktemp("tiny_qwen3")
    write_tiny_qwen_model(root, num_layers=3)
    return SimpleNamespace(dir=str(root), transformer=root / "transformer")


def test_num_layers_override_loads_the_first_blocks_and_skips_the_rest(tiny3):
    config = NeuronQwenImageConfig.from_pretrained(tiny3.transformer, num_layers=2, **TINY_SHAPE)
    assert (config.num_layers, config.checkpoint_num_layers) == (2, 3)
    app = NeuronQwenImageTransformerApplication(
        model_path=tiny3.transformer, config=config, dtype=torch.float32, device="cpu"
    )
    assert app.skipped_checkpoint_prefixes == ("transformer_blocks.2.",)
    with pytest.raises(ValueError, match="exceeds the checkpoint's 3 blocks"):
        NeuronQwenImageConfig.from_pretrained(tiny3.transformer, num_layers=4)

    reference = diffusers_reference(tiny3.transformer, num_layers=2)
    full = diffusers_reference(tiny3.transformer)
    (result,) = _run(None, num_layers=2, model_dir=tiny3.dir)
    torch.testing.assert_close(torch.tensor(result["out"]), reference, atol=ATOL, rtol=RTOL)
    assert not torch.allclose(reference, full, atol=1e-3)  # the third block mattered
    assert result["num_blocks"] == 2
    block2 = sorted(k for k in checkpoint_keys(tiny3.transformer)
                    if k.startswith("transformer_blocks.2."))
    assert len(block2) == 32  # every tensor of one Qwen-Image block
    assert result["load_report"] == {"missing": [], "unexpected": [], "skipped": block2}
    # Without the override the same checkpoint loads clean.
    (full_run,) = _run(None, model_dir=tiny3.dir)
    torch.testing.assert_close(torch.tensor(full_run["out"]), full, atol=ATOL, rtol=RTOL)
    assert full_run["load_report"] == CLEAN_REPORT
    assert Path(tiny3.dir, "transformer", "config.json").is_file()
