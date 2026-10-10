"""W1 unit tests (CPU only): the neuron Wan DiT application and the composed Wan application.

The model is the tiny diffusers-layout Wan2.1 of tests/unit/backends/_neuron_wan_toy.py (4 heads,
2 blocks, a one-layer umT5). Multi-rank cases run on 4 gloo processes through
tests/unit/backends/_neuron_gloo.py: at TP4 the SPMDRank weight slice, the qk-norm all-reduces
and the row-parallel all-reduces all run, which TP1 skips. Nothing here touches the device.
"""

import pytest

# importorskip first, as in test_neuron_application_c8.py: it keeps torch_neuronx's import-time
# nki.jit DeprecationWarning out of the report when this module is collected first.
torch = pytest.importorskip("torch")

import inspect  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from difflet.backends.neuron import runtime  # noqa: E402
from difflet.backends.neuron.ops_impl import parallel_mesh  # noqa: E402
from difflet.backends.neuron.wan.config import NeuronWanConfig  # noqa: E402
from difflet.backends.neuron.wan.transformer import (  # noqa: E402
    NeuronWanTransformerApplication,
    checkpoint_key,
)
from difflet.models.wan import neuron_application as na  # noqa: E402
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from tests.unit.backends._neuron_gloo import run_ranks  # noqa: E402
from tests.unit.backends._neuron_toy import tp1_mesh  # noqa: E402
from tests.unit.backends._neuron_wan_toy import (  # noqa: E402
    TINY_BLOCK_NAMES,
    TINY_LATENT_SHAPE,
    TINY_SHAPE,
    TINY_TEXT_SEQ_LEN,
    TINY_TEXT_SHAPE,
    tiny_wan_config,
    tiny_wan_pipeline_reference,
    tiny_wan_tp1_model,
    write_tiny_wan_model,
)
from tests.unit.backends._neuron_workers import (  # noqa: E402
    w1_fake_hidden,
    w1_host_text_encoder_worker,
    w1_wan_lifecycle_worker,
    w1_wan_one_graph_worker,
    w1_wan_pipeline_worker,
)

LOAD_PHASES = {
    "runtime init", "build on meta", "load checkpoint", "eval", "compile", "warmup forward"
}
CONTRACT_KEYS = ["hidden_states", "timestep", "encoder_hidden_states"]
# No rank may reach for the Hub: every model file is local.
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
HOST_ENCODER_TIMEOUT = 120.0


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
    pytest.importorskip("transformers")
    root = tmp_path_factory.mktemp("tiny_wan")
    ref = write_tiny_wan_model(root)
    return SimpleNamespace(dir=str(root), transformer=root / "transformer", ref=ref)


def _composed(tiny, **overrides):
    kwargs = dict(
        model_path=tiny.dir,
        parallel=DiffletParallelConfig(),
        dtype=torch.float32,
        shape=dict(TINY_SHAPE),
        text_seq_len=TINY_TEXT_SEQ_LEN,
        exec_mode="eager",
        device="cpu",
    )
    kwargs.update(overrides)
    return na.TorchNeuronWanApplication(**kwargs)


def _shapes_and_dtypes(tensors):
    return [(tuple(t.shape), t.dtype) for t in tensors]


def test_config_validates_tp_and_shape(tiny):
    config = NeuronWanConfig.from_pretrained(
        tiny.transformer, tp_degree=4, text_seq_len=TINY_TEXT_SEQ_LEN, **TINY_SHAPE
    )
    assert config.patch_size == (1, 2, 2) and config.num_layers == 2
    geometry = (config.latent_frames, config.latent_height, config.latent_width)
    assert geometry == TINY_LATENT_SHAPE[2:] and config.image_seq_len == 12
    with pytest.raises(ValueError, match="does not divide tp=3"):
        NeuronWanConfig.from_pretrained(tiny.transformer, tp_degree=3)
    with pytest.raises(ValueError, match=r"num_frames=6 must satisfy \(num_frames - 1\) % 4"):
        NeuronWanConfig.from_pretrained(tiny.transformer, num_frames=6)
    for flag in ("context_parallel_enabled", "cfg_parallel_enabled", "sp_enabled"):
        with pytest.raises(NotImplementedError, match="neuron backend supports tensor parallel"):
            NeuronWanConfig.from_pretrained(tiny.transformer, **{flag: True})
    # The defaults are Wan2.1-T2V-14B at the benchmark shape 480x832x9.
    default = NeuronWanConfig(tp_degree=4)
    default.validate()
    geometry = (default.latent_frames, default.latent_height, default.latent_width)
    assert geometry == (3, 60, 104) and default.image_seq_len == 4680


def test_checkpoint_key_round_trip(tiny):
    assert checkpoint_key("blocks.0.ffn.net_in.weight") == "blocks.0.ffn.net.0.proj.weight"
    assert checkpoint_key("blocks.39.ffn.net_out.bias") == "blocks.39.ffn.net.2.bias"
    for name in (
        "blocks.0.attn1.to_q.weight",
        "blocks.0.attn2.to_out.0.bias",
        "blocks.1.attn1.norm_k.weight",
        "blocks.1.norm2.weight",
        "blocks.1.scale_shift_table",
        "patch_embedding.weight",
        "condition_embedder.time_embedder.linear_1.weight",
        "proj_out.bias",
        "scale_shift_table",
    ):
        assert checkpoint_key(name) == name
    app = NeuronWanTransformerApplication(
        model_path=tiny.transformer, config=tiny_wan_config(), dtype=torch.float32, device="cpu"
    )
    assert app.checkpoint_key("blocks.0.ffn.net_in.bias") == "blocks.0.ffn.net.0.proj.bias"
    # Every module parameter maps onto exactly the diffusers keys on disk, and back.
    from safetensors import safe_open

    with safe_open(str(tiny.transformer / "diffusion_pytorch_model.safetensors"), "pt") as f:
        on_disk = set(f.keys())
    with tp1_mesh():
        names = list(tiny_wan_tp1_model().state_dict())
    assert sorted(app.checkpoint_key(n) for n in names) == sorted(on_disk)
    assert any(".ffn.net.2." in key for key in on_disk)


def test_example_inputs_match_the_contract(tiny):
    app = _composed(tiny)
    contract = app.dit_input_contract()
    assert list(contract) == CONTRACT_KEYS
    expected = [(c["shape"], c["dtype"]) for c in contract.values()]
    assert expected == [
        (TINY_LATENT_SHAPE, torch.float32), ((1,), torch.float32), (TINY_TEXT_SHAPE, torch.float32)
    ]
    assert _shapes_and_dtypes(app.transformer.get_example_inputs()) == expected
    # At the real geometry the warm-up compiles exactly the per-step shapes (one graph).
    real = NeuronWanTransformerApplication(
        model_path="/unused", config=NeuronWanConfig(), dtype=torch.bfloat16, device="cpu"
    )
    assert _shapes_and_dtypes(real.get_example_inputs()) == [
        ((1, 16, 3, 60, 104), torch.bfloat16),
        ((1,), torch.bfloat16),
        ((1, 512, 4096), torch.bfloat16),
    ]


def test_transformer_application_guards_tp_and_dtype(tiny):
    with pytest.raises(ValueError, match="tp_degree=4.*tp_degree=1"):
        NeuronWanTransformerApplication(
            model_path=tiny.transformer, config=tiny_wan_config(tp_degree=4),
            parallel=DiffletParallelConfig(tp_degree=1), dtype=torch.float32, device="cpu",
        )
    # Decision D42: bf16 is the only dtype on the device (reduce_dtype == activation dtype).
    with pytest.raises(ValueError, match="bfloat16"):
        NeuronWanTransformerApplication(
            model_path=tiny.transformer, config=tiny_wan_config(), dtype=torch.float32,
            device="neuron",
        )
    app = NeuronWanTransformerApplication(
        model_path=tiny.transformer, config=tiny_wan_config(), dtype=torch.bfloat16,
        device="neuron",
    )
    assert app.device.type == "neuron" and app.is_loaded is False
    assert runtime._neuron_runtime_initialized() is False  # constructing touched nothing


@pytest.mark.parametrize("exec_mode", ["eager", "compile"])
def test_four_rank_lifecycle_matches_tp1(tiny, exec_mode):
    results = run_ranks(w1_wan_lifecycle_worker, tiny.dir, exec_mode, world_size=4, env=OFFLINE)
    with tp1_mesh():
        full_numel = sum(p.numel() for p in tiny_wan_tp1_model().parameters())
    outs = [torch.tensor(result["out"]) for result in results]
    for rank, (result, out) in enumerate(zip(results, outs, strict=True)):
        assert (result["rank"], result["world_size"]) == (rank, 4)
        assert result["load_report"] == {"missing": [], "unexpected": []}
        assert result["warmup_shapes"] == [TINY_LATENT_SHAPE, (1,), TINY_TEXT_SHAPE]
        assert result["compiled_blocks"] == (TINY_BLOCK_NAMES if exec_mode == "compile" else [])
        assert result["unwarmed_shapes"] == []
        assert result["local_heads"] == 1  # 4 heads over 4 ranks
        assert result["param_numel"] < full_numel  # sharded, not four TP1 copies
        assert torch.equal(out, outs[0])  # the all-reduced output is replicated
        torch.testing.assert_close(out, tiny.ref, atol=1e-4, rtol=1e-4)


def test_compile_mode_traces_one_graph_for_all_blocks(tiny):
    # Errata P1-3: at TP4, where SPMDRank.get_rank(), the weight narrow and the functional
    # all-reduces are inside the block; tp=1 skips all three.
    results = run_ranks(w1_wan_one_graph_worker, tiny.dir, world_size=4, env=OFFLINE)
    for result in results:
        assert result["compiled_blocks"] == TINY_BLOCK_NAMES
        assert result["unique_graphs_after_warmup"] == 1  # both blocks share the warm-up graph
        assert result["unique_graphs_after_forwards"] == 1  # and the CFG forwards reuse it
        assert result["graph_breaks"] == 0
        assert result["recorded_graphs"] == 1
        # q/k qk-norm sums and the to_out all-reduce of each attention, plus the FFN's.
        assert result["collectives_per_graph"] == [7]
        assert result["unwarmed_shapes"] == []


def test_adapter_passes_contiguous_inputs():
    received = []

    class Recorder:
        def __call__(self, *args):
            received.append(args)
            return torch.ones(1, 4, 2, 4, 6)

    config = SimpleNamespace(cfg_parallel_enabled=False)
    adapter = na._OnDeviceTransformer(Recorder(), config, torch.bfloat16, "cpu")
    assert adapter.dtype is torch.bfloat16 and adapter.config is config
    seconds = []
    adapter.forward_hook = seconds.append
    sent = []
    # Strided host views as the orchestrator can hand them over: transposed latents and text,
    # and _batch_timestep's expanded 0-dim timestep (stride 0). The second call is already in
    # the adapter dtype, so only the adapter's .contiguous() can make it dense.
    for dtype, batch in ((torch.float32, 1), (torch.bfloat16, 2)):
        latents = torch.randn(batch, 4, 2, 6, 4).to(dtype).transpose(-1, -2)
        timestep = torch.tensor(999.0, dtype=dtype).expand(batch)
        text = torch.randn(batch, 24, 8).to(dtype).transpose(1, 2)
        assert not latents.is_contiguous() and not text.is_contiguous()
        assert timestep.stride() == (0,)
        out = adapter(latents, timestep, text)
        assert out.device.type == "cpu" and torch.equal(out, torch.ones(1, 4, 2, 4, 6))
        sent.append((latents, timestep, text))
    assert len(received) == 2
    for args, originals in zip(received, sent, strict=True):
        for got, original in zip(args, originals, strict=True):
            assert got.is_contiguous() and got.dtype is torch.bfloat16
            assert torch.equal(got, original.to(torch.bfloat16))
    assert received[1][1].stride() == (1,)  # the batch-2 timestep was materialised
    assert len(seconds) == 2 and all(s >= 0.0 for s in seconds)


def test_host_text_encoder_broadcasts_rank0_result():
    start = time.monotonic()
    results = run_ranks(
        w1_host_text_encoder_worker, world_size=4, timeout=HOST_ENCODER_TIMEOUT, env=OFFLINE
    )
    assert time.monotonic() - start < HOST_ENCODER_TIMEOUT
    # Rank 0 encoded only the 6 valid tokens, with its own thread count, then re-padded to 8.
    ids = torch.tensor([[3, 4, 5, 6, 7, 1]])
    expected = torch.zeros(TINY_TEXT_SHAPE)
    expected[:, :6] = w1_fake_hidden(ids, TINY_TEXT_SHAPE[-1])
    expected = expected.to(torch.bfloat16)
    for rank, result in enumerate(results):
        for call in ("ok", "recovered"):
            assert result[call]["outcome"] == "ok" and result[call]["dtype"] == "torch.bfloat16"
            assert torch.equal(torch.tensor(result[call]["embeds"]).to(torch.bfloat16), expected)
        assert result["threads_restored"] is True
        if rank == 0:
            assert result["calls"] == [((1, 6), 2)] * 3
            assert result["failed"]["outcome"] == "ValueError: fake umT5 failure"
            assert result["load"]["outcome"].split(":")[0] != "RankFailureError"
            assert result["load"]["outcome"] != "ok"
        else:
            assert result["calls"] == []  # only rank 0 encodes
            assert result["failed"]["outcome"].startswith(
                "RankFailureError: prompt encode: failed on rank 0"
            )
            assert result["load"]["outcome"].startswith(
                "RankFailureError: text encoder load: failed on rank 0"
            )
        assert result["loaded_model"] is False


def test_application_call_dispatch(tiny):
    app = _composed(tiny)
    with pytest.raises(RuntimeError, match="not loaded"):
        app(prompt="a cat")
    with pytest.raises(RuntimeError, match="not loaded"):
        app(torch.zeros(1), torch.zeros(1), torch.zeros(1))
    app.transformer_adapter = lambda *args, **kwargs: ("adapter", len(args), kwargs)
    app.pipeline = lambda *args, **kwargs: ("pipeline", len(args), kwargs)
    x = torch.zeros(1)
    assert app(x, x, x) == ("adapter", 3, {})
    assert app(prompt="a cat", num_inference_steps=2) == (
        "pipeline", 0, {"prompt": "a cat", "num_inference_steps": 2}
    )
    assert app("a cat", guidance_scale=5.0) == ("pipeline", 1, {"guidance_scale": 5.0})
    # The DiffletPipeline contract (difflet_pipeline.py inspects these parameter names).
    assert list(inspect.signature(app.load).parameters) == [
        "compiled_model_path", "start_rank_id", "local_ranks_size", "skip_warmup"
    ]
    assert list(inspect.signature(app.compile).parameters) == ["compiled_model_path", "debug"]
    assert app.compile("/unused") is None and app.has_compiled_artifacts("/unused") is True
    # The orchestrator's shapes argument with one entry is the shape.
    single = _composed(tiny, shape={"height": None, "width": None, "num_frames": None},
                       shapes=((32, 48, 5),))
    assert single.config.image_seq_len == 12
    for kwargs, match in (
        ({"enable_transformer_2": True}, "transformer_2"),
        ({"enable_vae_decoder": True}, "host"),
        ({"teacache_cadence": 2}, "TeaCache"),
        ({"teacache_online_delta_alpha": 0.5}, "TeaCache"),
        ({"teacache_calibration_path": "/x.json"}, "TeaCache"),
        ({"shapes": ((32, 48, 5), (32, 48, 9))}, "one static shape"),
        ({"batch_size": 2}, "batch_size"),
        ({"parallel": DiffletParallelConfig(cp_degree=2)}, "tensor parallelism only"),
    ):
        with pytest.raises(NotImplementedError, match=match):
            _composed(tiny, **kwargs)
    with pytest.raises(TypeError, match="encoder_thread"):
        _composed(tiny, encoder_thread=4)


def test_prompt_goes_through_the_tokenizer_and_the_host_encoder(tiny):
    # One process (no process group): the CLI's app(prompt=...) path up to the DiT, with
    # enable_transformer=False so nothing but the text encoder loads.
    from transformers import UMT5EncoderModel

    app = _composed(tiny, enable_transformer=False, encoder_threads=2)
    assert app.transformer is None
    app.load()
    assert set(app.phase_seconds) == {"text encoder load"} and app.compiled_blocks == []
    out = app(prompt="a cat on the moon", num_inference_steps=1, output_type="latent")
    embeds = out.prompt_embeds
    assert tuple(embeds.shape) == TINY_TEXT_SHAPE and embeds.dtype == torch.float32
    # 5 words + </s> are valid; the 2 padded rows are zero.
    assert torch.count_nonzero(embeds[:, 6:]) == 0 and bool(embeds[:, :6].abs().sum(-1).all())
    encoder = UMT5EncoderModel.from_pretrained(
        str(Path(tiny.dir, "text_encoder")), dtype=torch.float32
    ).eval()
    ids = torch.tensor([[3, 4, 5, 6, 7, 1, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
    with torch.no_grad():
        full = encoder(ids, mask).last_hidden_state  # untrimmed, as diffusers encodes
    torch.testing.assert_close(embeds[:, :6], full[:, :6], atol=1e-5, rtol=1e-5)
    with pytest.raises(RuntimeError, match="DiT is not loaded"):
        app(torch.zeros(1), torch.zeros(1), torch.zeros(1))


def test_end_to_end_two_steps_on_cpu_matches_tp1(tiny):
    reference = tiny_wan_pipeline_reference(tiny.dir, num_inference_steps=2, guidance_scale=5.0)
    results = run_ranks(w1_wan_pipeline_worker, tiny.dir, world_size=4, env=OFFLINE)
    latents = [torch.tensor(result["latents"]) for result in results]
    for rank, result in enumerate(results):
        assert result["dtype"] == "torch.float32"
        assert result["sha256"] == results[0]["sha256"]  # bit-identical on every rank
        assert torch.equal(latents[rank], latents[0])
        # step_hook: one stamp per denoise iteration, after both CFG forwards (errata P0-2).
        assert result["steps"] == [2, 4]
        assert len(result["forward_seconds"]) == 4
        # The "" negative prompt went through the rank-0 host encoder (1 valid token, </s>),
        # not the orchestrator's zeros fallback.
        assert result["encoder_inputs"] == ([(1, 1)] if rank == 0 else [])
        assert len(result["negative_abs_sums"]) == 1 and result["negative_abs_sums"][0] > 0
        assert set(result["phase_seconds"]) == LOAD_PHASES | {"text encoder load"}
        assert result["unwarmed_shapes"] == []
    assert tuple(latents[0].shape) == TINY_LATENT_SHAPE
    torch.testing.assert_close(latents[0], reference, atol=1e-4, rtol=1e-4)
    assert not torch.equal(reference, torch.zeros_like(reference))
    assert Path(tiny.dir, "tokenizer", "tokenizer.json").is_file()
