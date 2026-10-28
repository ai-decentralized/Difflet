"""Q2 unit tests (CPU only): the end-to-end Qwen-Image application on the neuron backend.

``TorchNeuronQwenImageApplication`` (difflet/models/qwen_image/neuron_application.py) composes
Q1's DiT application, the rank-0 host Qwen2.5-VL, the optional device VAE (the Wan decoder) and
the backend-neutral ``QwenImageOrchestrator``. The model is the tiny snapshot of
tests/unit/backends/_neuron_qwen_app_toy.py around Q1's tiny DiT. Multi-rank runs go through
``_neuron_gloo.run_ranks`` (fresh spawned processes, ``DIFFLET_BACKEND=neuron``); the 7B encoder
is never loaded: the broadcast tests use a stub, the prompt tests a seeded 2-layer Qwen2.5-VL
with the real tokenizer (skipped when the real snapshot's tokenizer is not on this machine).
Nothing here touches the device.
"""

import pytest

# importorskip first, as in test_neuron_wan_w1.py: it keeps torch_neuronx's import-time
# nki.jit DeprecationWarning out of the report when this module is collected first.
torch = pytest.importorskip("torch")

import inspect  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from difflet.backends.neuron.ops_impl import parallel_mesh  # noqa: E402
from difflet.common.orchestrators import qwen_image as qwen_common  # noqa: E402
from difflet.models.qwen_image import neuron_application as na  # noqa: E402
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from tests.unit.backends._neuron_gloo import run_ranks  # noqa: E402
from tests.unit.backends._neuron_qwen_app_toy import (  # noqa: E402
    TEXT_DIM,
    TINY_IMAGE_SHAPE,
    TINY_PACKED_LATENT_SHAPE,
    TINY_PROMPT,
    TINY_VAE_CONFIG,
    TINY_Z_DIM,
    StubQwenTokenizer,
    StubQwenVL,
    q2_difflet_pipeline_worker,
    q2_host_text_encoder_worker,
    q2_pipeline_worker,
    stub_hidden,
    tiny_text_inputs,
    tokenizer_available,
    write_tiny_qwen_snapshot,
)
from tests.unit.backends._neuron_qwen_toy import (  # noqa: E402
    TINY_PACKED,
    TINY_SHAPE,
    TINY_TEXT_SEQ_LEN,
)

LOAD_PHASES = {
    "runtime init", "build on meta", "load checkpoint", "eval", "compile", "warmup forward"
}
# fp32 TP-N / CP-N against TP1 and diffusers differ by reduction order only.
FP32_TOL = {"atol": 1e-4, "rtol": 1e-4}
# No rank may reach for the Hub: every model file is local.
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
needs_tokenizer = pytest.mark.skipif(
    not tokenizer_available(), reason="the real Qwen-Image tokenizer is not on this machine"
)


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
    pytest.importorskip("transformers")
    root = tmp_path_factory.mktemp("tiny_qwen_app")
    write_tiny_qwen_snapshot(root)
    return SimpleNamespace(dir=str(root))


def _composed(tiny, **overrides):
    kwargs = dict(
        model_path=tiny.dir,
        parallel=DiffletParallelConfig(),
        dtype=torch.float32,
        shape=dict(TINY_SHAPE),
        text_seq_len=TINY_TEXT_SEQ_LEN,
        exec_mode="eager",
        device="cpu",
        enable_text_encoder=False,
    )
    kwargs.update(overrides)
    return na.TorchNeuronQwenImageApplication(**kwargs)


def _host_reference(model_dir, embeds, mask, *, steps=2, seed=7):
    """diffusers' own two-step loop: the tiny DiT on the right-padded ``embeds`` with their
    ``mask`` (the padded rows masked, as QwenImagePipeline masks them) under Qwen-Image's
    FlowMatch schedule, shifted as QwenImagePipeline shifts it, from the noise
    ``torch.Generator().manual_seed(seed)`` draws."""
    import numpy as np
    from diffusers import (
        FlowMatchEulerDiscreteScheduler,
        QwenImagePipeline,
        QwenImageTransformer2DModel,
    )
    from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift

    model = QwenImageTransformer2DModel.from_pretrained(
        str(Path(model_dir, "transformer")), torch_dtype=torch.float32
    ).eval()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(Path(model_dir, "scheduler")))
    latent_hw = (2 * TINY_PACKED[0], 2 * TINY_PACKED[1])
    noise = torch.randn((1, 1, TINY_Z_DIM, *latent_hw), generator=torch.Generator().manual_seed(
        seed), dtype=torch.float32)
    latents = QwenImagePipeline._pack_latents(noise, 1, TINY_Z_DIM, *latent_hw)
    image_seq_len = latents.shape[1]
    cfg = scheduler.config
    mu = calculate_shift(image_seq_len, cfg.base_image_seq_len, cfg.max_image_seq_len,
                         cfg.base_shift, cfg.max_shift)
    scheduler.set_timesteps(sigmas=np.linspace(1.0, 1.0 / steps, steps), mu=mu)
    with torch.no_grad():
        for t in scheduler.timesteps:
            noise_pred = model(
                hidden_states=latents,
                timestep=t.expand(1) / 1000,
                encoder_hidden_states=embeds.float(),
                encoder_hidden_states_mask=mask,
                img_shapes=[[(1, *TINY_PACKED)]],
                return_dict=False,
            )[0]
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    return latents, [float(t) for t in scheduler.timesteps]


# ---- shared constants ------------------------------------------------------------------


def test_template_is_shared_and_matches_diffusers():
    """One template for every Qwen-Image text path, equal to diffusers'."""
    from difflet.cli.orchestrators import qwen_image as cli_qwen
    from difflet.serving.orchestrators import qwen_image as serving_qwen

    for module in (cli_qwen, serving_qwen):
        assert module._QWEN_TEMPLATE is qwen_common.QWEN_TEMPLATE
        assert module._QWEN_DROP_IDX == qwen_common.QWEN_DROP_IDX == 34
    pytest.importorskip("diffusers")
    from diffusers import QwenImagePipeline

    # __init__ needs every component and sets the template there, so read it from the source.
    source = inspect.getsource(QwenImagePipeline.__init__)
    assert repr(qwen_common.QWEN_TEMPLATE)[1:-1] in source
    assert "prompt_template_encode_start_idx = 34" in source


# ---- factory, registry, kwargs -----------------------------------------------------------


def test_entry_routes_neuron_and_registry_lists_it(tiny):
    from difflet.models.qwen_image.entry import create_qwen_image_application
    from difflet.registry import resolve_model

    assert resolve_model("Qwen/Qwen-Image").backends == ("trainium", "tpu", "neuron")
    app = create_qwen_image_application(
        model_path=tiny.dir, parallel=DiffletParallelConfig(), dtype=torch.float32,
        shape=dict(TINY_SHAPE), backend="neuron", device="cpu", exec_mode="eager",
        text_seq_len=TINY_TEXT_SEQ_LEN, enable_text_encoder=False,
    )
    assert type(app) is na.TorchNeuronQwenImageApplication
    assert app.config.image_seq_len == TINY_PACKED[0] * TINY_PACKED[1]
    assert app.config.cp_degree == 1  # filled from the parallel config by the DiT application
    # CFG parallelism serves the neuron application's opt-in true CFG; the Trainium and TPU
    # paths run one forward per step and still refuse it.
    cfg_app = create_qwen_image_application(
        model_path=tiny.dir, parallel=DiffletParallelConfig(cfg_parallel_enabled=True),
        dtype=torch.float32, shape=dict(TINY_SHAPE), backend="neuron", device="cpu",
        text_seq_len=TINY_TEXT_SEQ_LEN, enable_text_encoder=False,
    )
    assert cfg_app.config.cfg_parallel_enabled and cfg_app.config.dit_batch == 2
    for backend in ("trainium", "tpu"):
        with pytest.raises(NotImplementedError, match="guidance-distilled"):
            create_qwen_image_application(
                model_path=tiny.dir, parallel=DiffletParallelConfig(cfg_parallel_enabled=True),
                dtype=torch.float32, shape=dict(TINY_SHAPE), backend=backend,
            )
    with pytest.raises(NotImplementedError, match="trainium, tpu and neuron backends, got 'cuda'"):
        create_qwen_image_application(
            model_path=tiny.dir, parallel=DiffletParallelConfig(), dtype=torch.float32,
            shape=dict(TINY_SHAPE), backend="cuda",
        )


def test_kwargs_mirror_the_cli_and_reject_phase1_limits(tiny):
    # Every keyword the CLI generate stage passes the Trainium application is accepted.
    app = _composed(
        tiny, shapes=((TINY_SHAPE["height"], TINY_SHAPE["width"], None),),
        enable_transformer=True, teacache_fused=False, teacache_speedup=None,
        teacache_calibration_path=None, teacache_cadence=None, teacache_online_delta_alpha=None,
    )
    assert app.shape == {**TINY_SHAPE, "num_frames": None}
    # The DiffletPipeline contract (difflet_pipeline.py inspects these parameter names).
    assert list(inspect.signature(app.load).parameters) == [
        "compiled_model_path", "start_rank_id", "local_ranks_size", "skip_warmup"
    ]
    assert list(inspect.signature(app.compile).parameters) == ["compiled_model_path", "debug"]
    assert app.compile("/unused") is None and app.has_compiled_artifacts("/unused") is True
    # One shape from shapes alone, (h, w) entries included.
    single = _composed(tiny, shape={"height": None, "width": None, "num_frames": None},
                       shapes=[[32, 48]])
    assert single.config.image_seq_len == 6 and single.shape["height"] == 32
    for kwargs, match in (
        ({"teacache_cadence": 2}, "TeaCache"),
        ({"teacache_online_delta_alpha": 0.5}, "TeaCache"),
        ({"teacache_calibration_path": "/x.json"}, "TeaCache"),
        ({"teacache_speedup": 1.5}, "TeaCache"),
        ({"teacache_calibration": object()}, "TeaCache"),
        ({"teacache_fused": True}, "teacache_fused"),
        ({"shapes": ((64, 96, None), (32, 48, None))}, "one static shape"),
        ({"batch_size": 2}, "batch_size"),
        ({"parallel": DiffletParallelConfig(dp_degree=2)}, "dp_degree=2"),
    ):
        with pytest.raises(NotImplementedError, match=match):
            _composed(tiny, **kwargs)
    # Sequence parallelism reaches the DiT's config (SP with CP is refused by the parallel
    # config itself and again by NeuronQwenImageConfig, as for Wan).
    assert _composed(tiny, parallel=DiffletParallelConfig(sp_enabled=True)).config.sp_enabled
    with pytest.raises(TypeError, match="encoder_thread"):
        _composed(tiny, encoder_thread=4)
    with pytest.raises(ValueError, match="text_encoder_dtype"):
        _composed(tiny, text_encoder_dtype=torch.float16)
    with pytest.raises(ValueError, match="not the one compiled shape"):
        _composed(tiny, shapes=((32, 48, None),))
    with pytest.raises(ValueError, match="bfloat16 on the neuron device"):
        _composed(tiny, device="neuron")  # D42 (no device is touched: the check is in __init__)
    fp32 = _composed(tiny, text_encoder_dtype="float32", enable_text_encoder=True)
    assert fp32.text_encoder.dtype is torch.float32
    assert _composed(tiny, enable_text_encoder=True).text_encoder.dtype is torch.bfloat16
    assert fp32.text_encoder.enc_seq_len == qwen_common.ENC_SEQ == 256


def test_dit_contract_and_call_dispatch(tiny):
    app = _composed(tiny)
    contract = app.dit_input_contract()
    assert list(contract) == ["hidden_states", "timestep", "encoder_hidden_states",
                              "encoder_hidden_states_mask", "guidance"]
    assert contract["hidden_states"]["shape"] == TINY_PACKED_LATENT_SHAPE
    assert contract["encoder_hidden_states"]["shape"] == (1, TINY_TEXT_SEQ_LEN, TEXT_DIM)
    assert contract["encoder_hidden_states_mask"]["dtype"] is torch.bool
    with pytest.raises(RuntimeError, match="not loaded"):
        app(prompt="a cat")
    with pytest.raises(RuntimeError, match="DiT is not loaded"):
        app(torch.zeros(1), torch.zeros(1), torch.zeros(1))
    app.transformer_adapter = lambda *args, **kwargs: ("dit", len(args), sorted(kwargs))
    app.generate = lambda **kwargs: ("generate", sorted(kwargs))
    x = torch.zeros(1)
    bundle = na.QwenImageDiTInputBundle(x, x, x, x, x)
    assert app(bundle) == ("dit", 1, [])
    assert app.forward_dit(bundle) == ("dit", 1, [])
    assert app(x, x, x) == ("dit", 3, [])
    assert app(x, x, x, x, x) == ("dit", 5, [])
    assert app(hidden_states=x, timestep=x, encoder_hidden_states=x) == (
        "dit", 0, ["encoder_hidden_states", "hidden_states", "timestep"]
    )
    assert app("a cat", num_inference_steps=2) == ("generate", ["num_inference_steps", "prompt"])
    assert app(encoder_hidden_states=x, encoder_hidden_states_mask=x) == (
        "generate", ["encoder_hidden_states", "encoder_hidden_states_mask"]
    )
    with pytest.raises(TypeError, match="positional arguments"):
        app(x, x)


def test_adapter_takes_the_bundle_passes_the_mask_drops_guidance_and_checks_shapes():
    received = []

    def recorder(*args):
        received.append(args)
        return torch.ones(1, 24, 16)

    config = SimpleNamespace(batch_size=1, image_seq_len=24, in_channels=16, text_seq_len=8,
                             joint_attention_dim=TEXT_DIM, height=64, width=96)
    adapter = na._OnDeviceTransformer(recorder, config, torch.bfloat16, "cpu")
    seconds = []
    adapter.forward_hook = seconds.append
    hidden = torch.randn(1, 16, 24).transpose(1, 2)  # a strided host view
    timestep = torch.tensor(0.5).expand(1)
    text = torch.randn(1, 8, TEXT_DIM)
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[:, :5] = True  # a right-padded prompt: 5 valid rows
    out = adapter(na.QwenImageDiTInputBundle(hidden, timestep, text, mask, torch.zeros(1)))
    assert torch.equal(out, torch.ones(1, 24, 16)) and adapter.calls == 1 and len(seconds) == 1
    adapter(hidden, timestep, text, mask, torch.zeros(1))
    adapter(hidden, timestep, text)  # no mask: every text row valid
    assert len(received) == 3
    for args in received:
        assert len(args) == 4  # the mask reaches the DiT; guidance never does
        assert all(a.is_contiguous() and a.dtype is torch.bfloat16 for a in args[:3])
        assert torch.equal(args[0], hidden.to(torch.bfloat16))
        assert args[3].dtype is torch.int64 and args[3].is_contiguous()
    assert received[0][3].tolist() == received[1][3].tolist() == [[1] * 5 + [0] * 3]
    assert received[2][3].tolist() == [[1] * 8]
    # The DiT masks by valid-row count, so only a prefix of valid rows is accepted.
    holes = mask.clone()
    holes[:, 1] = False
    with pytest.raises(ValueError, match="prefix of valid rows"):
        adapter(hidden, timestep, text, holes)
    with pytest.raises(ValueError, match="'encoder_hidden_states_mask' has shape"):
        adapter(hidden, timestep, text, mask[:, :5])
    with pytest.raises(ValueError, match="zero-pad the prompt embeddings to text_seq_len=8"):
        adapter(hidden, timestep, text[:, :5])
    with pytest.raises(ValueError, match="'hidden_states' has shape"):
        adapter(hidden[:, :12], timestep, text)


# ---- the text encoder ---------------------------------------------------------------------


def test_prompt_embeds_post_processing_is_the_trainium_stage():
    """``qwen_prompt_embeds`` against the CLI's ``_stage_text`` arithmetic, written out."""
    hidden = torch.randn(1, 256, TEXT_DIM)
    valid = 41
    embeds, seq = na.qwen_prompt_embeds(hidden, valid, seq_len=TINY_TEXT_SEQ_LEN)
    # _stage_text: dev = hs[:, 34:valid]; ehs = zeros(1, TEXT_SEQ_LEN, D, bf16); ehs[:, :seq] = dev
    dev = hidden.float()[:, 34:valid]
    expected = torch.zeros(1, TINY_TEXT_SEQ_LEN, TEXT_DIM, dtype=torch.bfloat16)
    expected[:, : dev.shape[1]] = dev.to(torch.bfloat16)
    assert seq == 7 and embeds.dtype is torch.bfloat16 and torch.equal(embeds, expected)
    mask = na.qwen_text_mask(seq, TINY_TEXT_SEQ_LEN)
    assert mask.dtype is torch.bool and mask.tolist() == [[True] * 7 + [False]]
    with pytest.raises(ValueError, match="more than text_seq_len=8"):
        na.qwen_prompt_embeds(hidden, 34 + 9, seq_len=TINY_TEXT_SEQ_LEN)


def test_host_text_encoder_with_the_stub_on_one_rank(caplog):
    encoder = na._HostQwenTextEncoder("/unused", text_dim=TEXT_DIM, seq_len=TINY_TEXT_SEQ_LEN,
                                      dtype=torch.bfloat16, device="cpu", threads=2)
    with pytest.raises(RuntimeError, match="not loaded on rank 0"):
        encoder("a cat")
    encoder.tokenizer, encoder.model = StubQwenTokenizer(n_prompt=3), StubQwenVL()
    embeds, mask = encoder("a red fox")
    text, kwargs = encoder.tokenizer.calls[0]
    assert text == qwen_common.QWEN_TEMPLATE.format("a red fox")
    assert kwargs == {"max_length": 256, "padding": "max_length", "truncation": True,
                      "return_tensors": "pt", "return_attention_mask": True}
    assert encoder.model.calls == [((1, 37), 2)]  # only the valid prefix is encoded
    expected = torch.zeros(1, TINY_TEXT_SEQ_LEN, TEXT_DIM, dtype=torch.bfloat16)
    expected[:, :3] = stub_hidden(torch.arange(100, 137)[None])[:, 34:].to(torch.bfloat16)
    assert torch.equal(embeds, expected)
    assert mask.tolist() == [[True] * 3 + [False] * 5]
    # A prompt longer than ENC_SEQ is truncated, as on Trainium, with a warning.
    encoder.tokenizer = StubQwenTokenizer(n_prompt=300)
    encoder.seq_len = 256
    with caplog.at_level("WARNING"):
        embeds, mask = encoder("x " * 300)
    assert "its tail is truncated" in caplog.text
    assert int(mask.sum()) == 256 - 34
    encoder.model = StubQwenVL(text_dim=TEXT_DIM + 1)
    encoder.seq_len = TINY_TEXT_SEQ_LEN
    encoder.tokenizer = StubQwenTokenizer(n_prompt=3)
    with pytest.raises(ValueError, match="joint_attention_dim=24"):
        encoder("a cat")


def test_host_text_encoder_load_spends_one_encode_before_any_request(monkeypatch):
    """The first encode of a fresh process was not reproducible under host CPU contention (a
    dp2tp2 replica's first request on the trn2, 2026-10-18; 4 of 51 fresh CPU processes), later
    ones always were: load() encodes a fixed prompt once and discards it, with the encode
    threads, so the first request is the process's second encode and its result is unchanged."""
    import transformers

    tok, vl = StubQwenTokenizer(n_prompt=3), StubQwenVL()
    vl.eval = lambda: vl
    vl.requires_grad_ = lambda flag: vl
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tok)
    monkeypatch.setattr(transformers.Qwen2_5_VLForConditionalGeneration, "from_pretrained",
                        lambda *a, **k: vl)
    encoder = na._HostQwenTextEncoder("/unused", text_dim=TEXT_DIM, seq_len=TINY_TEXT_SEQ_LEN,
                                      dtype=torch.bfloat16, device="cpu", threads=2)
    encoder.load()
    assert encoder.tokenizer is tok and encoder.model is vl
    assert [text for text, _ in tok.calls] == [qwen_common.QWEN_TEMPLATE.format(na._WARMUP_PROMPT)]
    assert vl.calls == [((1, 37), 2)]  # one forward, with the encode threads
    embeds, mask = encoder("a red fox")
    assert len(vl.calls) == 2 and tok.calls[-1][0] == qwen_common.QWEN_TEMPLATE.format("a red fox")
    expected = torch.zeros(1, TINY_TEXT_SEQ_LEN, TEXT_DIM, dtype=torch.bfloat16)
    expected[:, :3] = stub_hidden(torch.arange(100, 137)[None])[:, 34:].to(torch.bfloat16)
    assert torch.equal(embeds, expected) and mask.tolist() == [[True] * 3 + [False] * 5]


def test_host_text_encoder_broadcasts_rank0_result_to_two_ranks():
    results = run_ranks(q2_host_text_encoder_worker, world_size=2, timeout=120.0, env=OFFLINE)
    expected = torch.zeros(1, TINY_TEXT_SEQ_LEN, TEXT_DIM)
    expected[:, :3] = stub_hidden(torch.arange(100, 137)[None])[:, 34:]
    expected = expected.to(torch.bfloat16)
    for rank, result in enumerate(results):
        for call in ("ok", "recovered"):
            assert result[call]["outcome"] == "ok"
            assert result[call]["dtype"] == "torch.bfloat16"
            assert result[call]["mask_dtype"] == "torch.bool"
            assert torch.equal(torch.tensor(result[call]["embeds"]).to(torch.bfloat16), expected)
            assert result[call]["mask"] == [[True] * 3 + [False] * 5]
        assert result["ok"] == results[0]["ok"]  # identical on both ranks
        assert result["threads_restored"] is True
        if rank == 0:
            assert result["encoder_calls"] == [((1, 37), 2)] * 3
            assert len(result["tokenizer_calls"]) == 3
            assert result["failed"]["outcome"] == "ValueError: fake Qwen2.5-VL failure"
            assert result["load"]["outcome"].split(":")[0] not in ("ok", "RankFailureError")
        else:
            assert result["encoder_calls"] == [] and result["tokenizer_calls"] == []
            assert result["failed"]["outcome"].startswith(
                "RankFailureError: prompt encode: failed on rank 0"
            )
            assert result["load"]["outcome"].startswith(
                "RankFailureError: text encoder load: failed on rank 0"
            )
        assert result["loaded"] is False


@needs_tokenizer
def test_host_text_encoder_matches_diffusers_and_the_padded_encode(tiny):
    """The real load path (tokenizer + a tiny Qwen2.5-VL) in fp32: the kept rows equal diffusers'
    ``_get_qwen_prompt_embeds`` and the Trainium stage's full 256-token padded encode."""
    from diffusers import QwenImagePipeline
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    encoder = na._HostQwenTextEncoder(tiny.dir, text_dim=TEXT_DIM, seq_len=TINY_TEXT_SEQ_LEN,
                                      dtype=torch.float32, device="cpu", threads=2)
    encoder.load()
    assert isinstance(encoder.model, Qwen2_5_VLForConditionalGeneration)
    embeds, mask = encoder(TINY_PROMPT)
    assert mask.tolist() == [[True] * 7 + [False]]
    assert torch.count_nonzero(embeds[:, 7:]) == 0

    tokenizer = AutoTokenizer.from_pretrained(str(Path(tiny.dir, "tokenizer")))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(Path(tiny.dir, "text_encoder")), dtype=torch.float32
    ).eval()
    pipe = SimpleNamespace(
        tokenizer=tokenizer, text_encoder=model, tokenizer_max_length=1024,
        prompt_template_encode=qwen_common.QWEN_TEMPLATE, prompt_template_encode_start_idx=34,
        _extract_masked_hidden=lambda h, m: QwenImagePipeline._extract_masked_hidden(None, h, m),
    )
    with torch.no_grad():
        ref, ref_mask = QwenImagePipeline._get_qwen_prompt_embeds(
            pipe, TINY_PROMPT, device=torch.device("cpu"), dtype=torch.float32
        )
    assert tuple(ref.shape) == (1, 7, TEXT_DIM) and ref_mask.tolist() == [[1] * 7]
    torch.testing.assert_close(embeds[:, :7], ref.to(torch.bfloat16), atol=0, rtol=0)
    # The Trainium stage: 256 padded tokens, the last (normed) hidden state, rows 34..valid.
    tokens = tokenizer(qwen_common.QWEN_TEMPLATE.format(TINY_PROMPT), max_length=256,
                       padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        full = model(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask,
                     output_hidden_states=True).hidden_states[-1]
    valid = int(tokens.attention_mask.sum())
    torch.testing.assert_close(embeds[:, :7].float(),
                               full[:, 34:valid].to(torch.bfloat16).float(), atol=0, rtol=0)


# ---- the VAE --------------------------------------------------------------------------------


def test_qwen_vae_loads_into_the_wan_decoder_and_decodes_like_diffusers(tiny):
    """Qwen-Image's VAE through ``NeuronWanVaeDecoderApplication`` (one CPU rank, fp32 eager):
    every decoder tensor loads, and the decode equals ``AutoencoderKLQwenImage.decode``."""
    from diffusers import AutoencoderKLQwenImage

    app = _composed(tiny, enable_transformer=False, enable_vae_decoder=True,
                    vae_dtype=torch.float32)
    vae = app.vae_decoder
    assert vae.dtype is torch.float32 and vae.app.exec_mode == "eager"
    assert vae.app.latent_shape == (1, TINY_Z_DIM, 1, 2 * TINY_PACKED[0], 2 * TINY_PACKED[1])
    assert tuple(vae.config.latents_std) == tuple(TINY_VAE_CONFIG["latents_std"])
    assert not hasattr(vae, "decode")  # the orchestrator calls it directly
    app.load()
    assert vae.app.load_report["missing"] == [] and vae.app.load_report["unexpected"] == []
    assert vae.app.load_report["skipped"]  # the encoder half
    latents = torch.randn(1, TINY_Z_DIM, 1, 8, 12, generator=torch.Generator().manual_seed(8))
    images = vae(latents)
    assert tuple(images.shape) == TINY_IMAGE_SHAPE  # the frame axis is dropped
    reference = AutoencoderKLQwenImage.from_pretrained(str(Path(tiny.dir, "vae"))).eval()
    with torch.no_grad():
        expected = reference.decode(latents, return_dict=False)[0][:, :, 0]
    torch.testing.assert_close(images, expected, **FP32_TOL)
    assert vae.app.dtype is torch.float32  # asked for: the default is bf16


def test_qwen_vae_bf16_is_the_default_and_close_to_the_fp32_decode(tiny):
    """bf16 is the decoder's default (``vae_dtype=torch.float32``, the CLI's --vae-dtype fp32,
    keeps fp32): the decoder loads in bf16, the adapter still takes fp32 latents (the
    orchestrator un-normalises in fp32) and returns fp32 images (the decoder's fp32 output head),
    within bf16's error of the fp32 decode of the same latents. 40 dB (images in [-1, 1])
    catches a broken path; the device gate on the real decoder passed at 56.7 dB."""
    import math

    default = _composed(tiny, enable_transformer=False, enable_vae_decoder=True).vae_decoder
    assert default.app.dtype is torch.bfloat16 and default.dtype is torch.float32
    latents = torch.randn(1, TINY_Z_DIM, 1, 8, 12, generator=torch.Generator().manual_seed(8))
    images = {}
    for dtype in (torch.float32, torch.bfloat16):
        app = _composed(tiny, enable_transformer=False, enable_vae_decoder=True, vae_dtype=dtype)
        vae = app.vae_decoder
        assert vae.app.dtype is dtype and vae.dtype is torch.float32
        app.load()
        assert {p.dtype for p in vae.app.module.parameters()} == {dtype}
        images[dtype] = vae(latents)
    fp32, bf16 = images[torch.float32], images[torch.bfloat16]
    assert fp32.dtype == bf16.dtype == torch.float32 and bf16.shape == TINY_IMAGE_SHAPE
    mse = float(((bf16.double() - fp32.double()) ** 2).mean())
    assert 0 < mse and 10 * math.log10(4.0 / mse) >= 40.0
    with pytest.raises(ValueError, match="vae_dtype must be bfloat16 or float32"):
        _composed(tiny, enable_vae_decoder=True, vae_dtype=torch.float16)


@pytest.mark.skipif(
    not Path("/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen-Image/snapshots/"
             "75e0b4be04f60ec59a75f475837eced720f823b6/vae/config.json").is_file(),
    reason="the real Qwen-Image snapshot is not on this machine",
)
def test_real_qwen_vae_checkpoint_maps_onto_the_wan_decoder():
    """Header-only: the real VAE's decoder tensors are exactly WanVAEDecoderModel's, by name and
    shape (nothing is loaded)."""
    import json
    import struct

    from tests.unit.backends._neuron_qwen_toy import QWEN_SNAPSHOT
    from difflet.models.wan.vae.modeling_vae import WanVAEDecoderConfig, WanVAEDecoderModel

    vae_dir = QWEN_SNAPSHOT / "vae"
    with open(vae_dir / "diffusion_pytorch_model.safetensors", "rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    header.pop("__metadata__", None)
    config = WanVAEDecoderConfig.from_diffusers_dict(json.loads((vae_dir / "config.json")
                                                                .read_text()))
    assert config.z_dim == 16 and len(config.latents_mean) == 16
    with torch.device("meta"):
        model = WanVAEDecoderModel(config)
    params = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    decoder = {k: tuple(v["shape"]) for k, v in header.items()
               if not k.startswith(("encoder.", "quant_conv."))}
    assert params == decoder and len(params) == 108


# ---- end to end -----------------------------------------------------------------------------


@pytest.mark.parametrize(("world_size", "tp", "cp_mode"), [(1, 1, "gather_kv"), (2, 2, "gather_kv")])
def test_end_to_end_two_steps_on_fixed_embeddings_matches_diffusers(tiny, world_size, tp, cp_mode):
    embeds, mask = tiny_text_inputs()
    reference, timesteps = _host_reference(tiny.dir, embeds, mask)
    results = run_ranks(q2_pipeline_worker, tiny.dir, tp, cp_mode, "eager", False,
                        world_size=world_size, env=OFFLINE)
    for result in results:
        assert result["sha256"] == results[0]["sha256"]  # bit-identical on every rank
        assert result["dtype"] == "torch.float32"
        assert result["steps"] == [1, 2] and result["forwards"] == 2  # step_hook once per step
        assert result["timesteps"] == pytest.approx(timesteps)  # the shifted schedule
        assert set(result["phase_seconds"]) == LOAD_PHASES
        assert result["unwarmed_shapes"] == []
    latents = torch.tensor(results[0]["latents"])
    assert tuple(latents.shape) == TINY_PACKED_LATENT_SHAPE
    torch.testing.assert_close(latents, reference, **FP32_TOL)


@pytest.mark.parametrize("exec_mode", ["eager", "compile"])
def test_end_to_end_sequence_parallel_tp2_matches_diffusers(tiny, exec_mode):
    """Two steps with Megatron SP over TP2 (the DiT's QwenImageSPTransformer2DModel fork):
    identical on both ranks, equal to TP2 without SP and to diffusers' loop."""
    embeds, mask = tiny_text_inputs()
    reference, _ = _host_reference(tiny.dir, embeds, mask)
    dense = run_ranks(q2_pipeline_worker, tiny.dir, 2, "gather_kv", exec_mode, False,
                      world_size=2, env=OFFLINE)
    results = run_ranks(q2_pipeline_worker, tiny.dir, 2, "gather_kv", exec_mode, False, "latent",
                        True, world_size=2, env=OFFLINE)
    for result in results:
        assert result["sha256"] == results[0]["sha256"]
        assert result["steps"] == [1, 2] and result["unwarmed_shapes"] == []
        assert len(result["compiled_blocks"]) == (2 if exec_mode == "compile" else 0)
    latents = torch.tensor(results[0]["latents"])
    torch.testing.assert_close(latents, torch.tensor(dense[0]["latents"]), **FP32_TOL)
    torch.testing.assert_close(latents, reference, **FP32_TOL)


@needs_tokenizer
@pytest.mark.parametrize(
    ("world_size", "tp", "cp_mode", "exec_mode"),
    [(1, 1, "gather_kv", "eager"), (2, 2, "gather_kv", "eager"), (2, 1, "gather_kv", "compile")],
)
def test_end_to_end_prompt_on_gloo_ranks_matches_tp1(tiny, world_size, tp, cp_mode, exec_mode):
    """``app(prompt=...)``: the tiny Qwen2.5-VL on rank 0, broadcast, two steps; TP1, TP2 and
    CP2 (compiled blocks) agree, and with diffusers' loop on the same embeddings."""
    results = run_ranks(q2_pipeline_worker, tiny.dir, tp, cp_mode, exec_mode, True,
                        world_size=world_size, env=OFFLINE)
    for result in results:
        assert result["sha256"] == results[0]["sha256"]
        assert result["embeds"] == results[0]["embeds"] and result["mask"] == results[0]["mask"]
        assert result["mask"] == [[True] * 7 + [False]]
        assert result["steps"] == [1, 2]
        assert set(result["phase_seconds"]) == LOAD_PHASES | {"text encoder load"}
        assert result["unwarmed_shapes"] == []
        assert len(result["compiled_blocks"]) == (2 if exec_mode == "compile" else 0)
    embeds = torch.tensor(results[0]["embeds"]).to(torch.bfloat16)
    reference, _ = _host_reference(tiny.dir, embeds, torch.tensor(results[0]["mask"]))
    torch.testing.assert_close(torch.tensor(results[0]["latents"]), reference, **FP32_TOL)


def test_end_to_end_with_the_device_vae_decodes_images(tiny):
    """``output_type="pt"`` with ``enable_vae_decoder``: the orchestrator un-normalises with the
    VAE's latents_mean/std and the decoder returns ``[1, 3, H, W]``, as diffusers decodes."""
    from diffusers import AutoencoderKLQwenImage, QwenImagePipeline

    results = run_ranks(q2_pipeline_worker, tiny.dir, 2, "gather_kv", "eager", False, "pt",
                        world_size=2, env=OFFLINE)
    assert results[0]["images"] == results[1]["images"]
    images = torch.tensor(results[0]["images"])
    assert tuple(images.shape) == TINY_IMAGE_SHAPE
    assert {f"vae {phase}" for phase in LOAD_PHASES} <= set(results[0]["phase_seconds"])
    latents = torch.tensor(results[0]["latents"])
    vae = AutoencoderKLQwenImage.from_pretrained(str(Path(tiny.dir, "vae"))).eval()
    unpacked = QwenImagePipeline._unpack_latents(latents, TINY_SHAPE["height"],
                                                 TINY_SHAPE["width"], 8)
    mean = torch.tensor(vae.config.latents_mean).view(1, TINY_Z_DIM, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, TINY_Z_DIM, 1, 1, 1)
    with torch.no_grad():
        expected = vae.decode(unpacked * std + mean, return_dict=False)[0][:, :, 0]
    torch.testing.assert_close(images, expected, **FP32_TOL)


def test_difflet_pipeline_builds_the_neuron_app_without_aot(tiny, tmp_path):
    cache = tmp_path / "compiled"
    (result,) = run_ranks(q2_difflet_pipeline_worker, tiny.dir, str(cache), world_size=1,
                          env=OFFLINE)
    assert result["app"] == "TorchNeuronQwenImageApplication" and result["backend"] == "neuron"
    assert result["compiled_dir_exists"] is False  # nothing is compiled ahead of time
    embeds, mask = tiny_text_inputs()
    reference, _ = _host_reference(tiny.dir, embeds, mask)
    torch.testing.assert_close(torch.tensor(result["latents"]), reference, **FP32_TOL)
