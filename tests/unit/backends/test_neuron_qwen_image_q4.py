"""Q4 unit tests (CPU only): Qwen-Image's prompt encoder on the neuron backend (opt-in).

``NeuronQwenTextEncoderApplication`` (difflet/backends/neuron/qwen_image/text_encoder.py) puts
Qwen2.5-VL's language model (``QwenTextEncoderModel``, difflet/models/qwen_image/
modeling_qwen_text_encoder.py) on the device, TP-sharded, and the composed application's
``text_encoder_device="neuron"`` uses it in place of the rank-0 host encoder. Here a seeded tiny
Qwen2.5-VL with grouped-query attention (4 query heads over 2 kv heads) and a three-axis mrope
section runs on gloo ranks in fp32 against transformers' own model and against the host encoder.
The 7B encoder is never loaded; the real checkpoint is read for its headers only.
"""

import pytest

torch = pytest.importorskip("torch")

import json  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from difflet.backends.neuron.hbm import GIB, HbmBudgetError  # noqa: E402
from difflet.backends.neuron.ops_impl import parallel_mesh  # noqa: E402
from difflet.backends.neuron.qwen_image import hbm as qwen_hbm  # noqa: E402
from difflet.models.qwen_image import neuron_application as na  # noqa: E402
from difflet.models.qwen_image.modeling_qwen_text_encoder import (  # noqa: E402
    QwenTextEncoderConfig,
    rotary_tables,
)
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from tests.unit.backends._neuron_gloo import run_ranks  # noqa: E402
from tests.unit.backends._neuron_qwen_app_toy import (  # noqa: E402
    TEXT_DIM,
    TINY_PROMPT,
    TP_TEXT_ENCODER,
    q2_pipeline_worker,
    q4_device_prompt_worker,
    q4_text_encoder_worker,
    tokenizer_available,
    write_tiny_qwen_snapshot,
    write_tiny_text_encoder,
)
from tests.unit.backends._neuron_qwen_toy import (  # noqa: E402
    QWEN_SNAPSHOT,
    TINY_SHAPE,
    TINY_TEXT_SEQ_LEN,
)

OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
FP32_TOL = {"atol": 1e-5, "rtol": 1e-5}
ENC = 48  # a short static encode for the modeling tests (the application's default is 256)
needs_tokenizer = pytest.mark.skipif(
    not tokenizer_available(), reason="the real Qwen-Image tokenizer is not on this machine"
)


@pytest.fixture(autouse=True)
def _clean_mesh(monkeypatch):
    monkeypatch.delenv("DIFFLET_EXEC_MODE", raising=False)
    monkeypatch.delenv("DIFFLET_NEURON_HBM_GIB", raising=False)
    parallel_mesh.destroy_parallel_mesh()
    yield
    parallel_mesh.destroy_parallel_mesh()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """The tiny snapshot with a TP2-able text encoder (always written; the tokenizer only when
    the real one is on this machine)."""
    pytest.importorskip("safetensors.torch")
    pytest.importorskip("accelerate")
    pytest.importorskip("diffusers")
    pytest.importorskip("transformers")
    root = tmp_path_factory.mktemp("tiny_qwen_q4")
    write_tiny_qwen_snapshot(root, text_encoder_overrides=TP_TEXT_ENCODER)
    if not (root / "text_encoder").is_dir():
        write_tiny_text_encoder(root / "text_encoder", **TP_TEXT_ENCODER)
    return SimpleNamespace(dir=str(root), text_encoder=root / "text_encoder")


def _transformers_lm(text_encoder_dir):
    from transformers import Qwen2_5_VLForConditionalGeneration

    return Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(text_encoder_dir), dtype=torch.float32
    ).eval()


def _ids(n=ENC):
    return torch.randint(0, 151936, (1, n), generator=torch.Generator().manual_seed(9))


# ---- config and rotary embedding -----------------------------------------------------------


def test_config_reads_the_text_config_and_validates_the_tp_split(tiny):
    config = QwenTextEncoderConfig.from_pretrained(tiny.text_encoder)
    assert (config.hidden_size, config.num_attention_heads, config.num_key_value_heads) == (
        TEXT_DIM, 4, 2)
    assert config.head_dim == 6 and config.num_hidden_layers == 2
    config.validate(2)
    with pytest.raises(ValueError, match="num_key_value_heads=2 does not divide tp=4"):
        config.validate(4)
    with pytest.raises(NotImplementedError, match="rope_scaling type 'yarn'"):
        QwenTextEncoderConfig.from_dict({"text_config": {"rope_scaling": {"rope_type": "yarn"}}})
    with pytest.raises(NotImplementedError, match="use_sliding_window"):
        QwenTextEncoderConfig.from_dict({"use_sliding_window": True})
    if (QWEN_SNAPSHOT / "text_encoder" / "config.json").is_file():
        real = QwenTextEncoderConfig.from_pretrained(QWEN_SNAPSHOT / "text_encoder")
        assert real == QwenTextEncoderConfig()  # the defaults are Qwen-Image's Qwen2.5-VL-7B
        real.validate(4)  # 28 / 4 query heads, 4 / 4 kv heads
        with pytest.raises(ValueError, match="num_attention_heads=28 does not divide tp=8"):
            real.validate(8)


def test_mrope_with_text_positions_is_the_1d_rope():
    """transformers' multimodal RoPE with the three identical position axes a text prompt gets
    equals ``rotary_tables``' 1-D RoPE, bit for bit (Qwen-Image's 16/24/24 sections)."""
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLRotaryEmbedding,
        apply_multimodal_rotary_pos_emb,
    )
    from transformers import Qwen2_5_VLTextConfig

    hf_config = Qwen2_5_VLTextConfig(
        hidden_size=3584, num_attention_heads=28, rope_theta=1000000.0,
        rope_scaling={"type": "default", "rope_type": "default", "mrope_section": [16, 24, 24]},
    )
    positions = torch.arange(256)[None].expand(3, 1, 256)  # get_rope_index for text
    q = torch.randn(1, 28, 256, 128, generator=torch.Generator().manual_seed(1))
    k = torch.randn(1, 4, 256, 128, generator=torch.Generator().manual_seed(2))
    hf_cos, hf_sin = Qwen2_5_VLRotaryEmbedding(hf_config)(q, positions)
    hf_q, hf_k = apply_multimodal_rotary_pos_emb(q, k, hf_cos, hf_sin, [16, 24, 24])
    cos, sin = rotary_tables(128, 256, 1000000.0)
    assert torch.equal(cos, hf_cos[0, 0]) and torch.equal(sin, hf_sin[0, 0])

    def rotate(x):
        x1, x2 = x[..., :64], x[..., 64:]
        return x * cos + torch.cat((-x2, x1), dim=-1) * sin

    assert torch.equal(rotate(q), hf_q) and torch.equal(rotate(k), hf_k)


# ---- the encoder against transformers --------------------------------------------------------


@pytest.mark.parametrize(("world_size", "exec_mode"),
                         [(1, "eager"), (2, "eager"), (1, "compile"), (2, "compile")])
def test_device_encoder_matches_transformers(tiny, world_size, exec_mode):
    """TP1 and TP2 (4 query heads over 2 kv heads: one group per rank), eager and with the
    decoder layers compiled into one aot_eager graph, against transformers' Qwen2.5-VL language
    model on the same ids, fp32."""
    results = run_ranks(q4_text_encoder_worker, str(tiny.text_encoder), exec_mode, ENC,
                        world_size=world_size, env=OFFLINE)
    with torch.no_grad():
        expected = _transformers_lm(tiny.text_encoder).model(
            input_ids=_ids(), use_cache=False).last_hidden_state
    for result in results:
        hidden = torch.tensor(result["hidden"])
        assert hidden.shape == (1, ENC, TEXT_DIM)
        torch.testing.assert_close(hidden, expected, **FP32_TOL)
        assert result["hidden"] == results[0]["hidden"]  # identical on every rank
        assert result["repeat_equal"] is True
        assert result["heads"] == [4 // world_size, 2 // world_size]
        report = result["load_report"]
        assert report["missing"] == [] and report["unexpected"] == []
        assert {k.split(".")[0] for k in report["skipped"]} == {"visual", "lm_head"}
        assert result["warmup_shapes"] == [(1, ENC)] and result["unwarmed_shapes"] == []
        if exec_mode == "compile":
            assert result["compiled_blocks"] == ["layers.0", "layers.1"]
            assert result["unique_graphs"] == 1 and result["graph_breaks"] == 0
        else:
            assert result["compiled_blocks"] == []


def test_causal_attention_keeps_the_padding_out_of_the_valid_rows(tiny):
    """Why the device encoder needs no mask: on a right-padded prompt, transformers' masked
    encode of the valid prefix and the unmasked causal encode of the whole padded length agree
    on every valid row."""
    model = _transformers_lm(tiny.text_encoder)
    ids = _ids()
    valid = 30
    padded = ids.clone()
    padded[:, valid:] = 151643  # the pad token
    with torch.no_grad():
        prefix = model.model(input_ids=ids[:, :valid], use_cache=False).last_hidden_state
        masked = model.model(input_ids=padded, attention_mask=(torch.arange(ENC) < valid)[None]
                             .long(), use_cache=False).last_hidden_state
        unmasked = model.model(input_ids=padded, use_cache=False).last_hidden_state
    torch.testing.assert_close(unmasked[:, :valid], prefix, **FP32_TOL)
    torch.testing.assert_close(masked[:, :valid], prefix, **FP32_TOL)


# ---- the composed application ------------------------------------------------------------


@needs_tokenizer
@pytest.mark.parametrize("world_size", [1, 2])
def test_device_prompt_embeds_equal_the_host_encoders(tiny, world_size):
    """The full post-processing: the device encoder's embeddings and mask (every rank) against
    the rank-0 host encoder's, both fp32, on the real tokenizer and the template."""
    host = na._HostQwenTextEncoder(tiny.dir, text_dim=TEXT_DIM, seq_len=TINY_TEXT_SEQ_LEN,
                                   dtype=torch.float32, device="cpu", threads=2)
    host.load()
    host_embeds, host_mask = host(TINY_PROMPT)
    results = run_ranks(q4_device_prompt_worker, tiny.dir, False, world_size=world_size,
                        env=OFFLINE)
    for result in results:
        assert result["encoder"] == "_DeviceQwenTextEncoder"
        assert result["dtype"] == "torch.bfloat16"
        assert result["mask"] == host_mask.tolist() == [[True] * 7 + [False]]
        assert result["embeds"] == results[0]["embeds"]
        torch.testing.assert_close(torch.tensor(result["embeds"]).to(torch.bfloat16),
                                   host_embeds, atol=0, rtol=0)
        assert {"text_encoder load checkpoint", "text_encoder warmup forward"} <= set(
            result["phase_seconds"])


@needs_tokenizer
def test_end_to_end_with_the_device_encoder_matches_the_host_encoder_run(tiny):
    """Two steps from a prompt at TP2, both encoders in fp32: the device encoder (every rank
    encodes) and the host encoder (rank 0 encodes, broadcast) give the same latents, because
    they give the same embeddings bit for bit (the test above)."""
    device = run_ranks(q4_device_prompt_worker, tiny.dir, True, world_size=2, env=OFFLINE)
    host = run_ranks(q2_pipeline_worker, tiny.dir, 2, "gather_kv", "eager", True, "latent",
                     False, "float32", world_size=2, env=OFFLINE)
    assert device[0]["sha256"] == device[1]["sha256"]
    assert device[0]["embeds"] == host[0]["embeds"]
    assert device[0]["sha256"] == host[0]["sha256"]


def _composed(tiny, **overrides):
    kwargs = dict(model_path=tiny.dir, parallel=DiffletParallelConfig(), dtype=torch.float32,
                  shape=dict(TINY_SHAPE), text_seq_len=TINY_TEXT_SEQ_LEN, exec_mode="eager",
                  device="cpu", text_encoder_device="neuron")
    kwargs.update(overrides)
    return na.TorchNeuronQwenImageApplication(**kwargs)


def test_application_options_and_the_hbm_refusal(tiny, monkeypatch):
    app = _composed(tiny)
    assert isinstance(app.text_encoder, na._DeviceQwenTextEncoder)
    assert app.text_encoder.app.dtype is torch.bfloat16  # text_encoder_dtype's default
    assert app.text_encoder.app.enc_seq_len == 256
    assert isinstance(_composed(tiny, text_encoder_device="host").text_encoder,
                      na._HostQwenTextEncoder)
    assert na._KWARG_DEFAULTS["text_encoder_device"] == "host"  # opt-in
    with pytest.raises(ValueError, match="text_encoder_device must be 'host' or 'neuron'"):
        _composed(tiny, text_encoder_device="tpu")
    # The DiT and the encoder together, before anything loads (a tiny budget refuses them).
    monkeypatch.setenv("DIFFLET_NEURON_HBM_GIB", "0.000001")
    with pytest.raises(HbmBudgetError, match="Qwen2.5-VL .* beside the Qwen-Image DiT"):
        _composed(tiny)
    _composed(tiny, text_encoder_device="host", enable_transformer=False)  # host: no check
    monkeypatch.setenv("DIFFLET_NEURON_HBM_GIB", "24")
    _composed(tiny)


def test_hbm_estimates_count_the_language_model_only(tiny, monkeypatch):
    from transformers import Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(tiny.text_encoder))
    lm = sum(p.numel() for n, p in model.named_parameters()
             if not n.startswith(("model.visual.", "lm_head.")))
    assert qwen_hbm.text_encoder_lm_numel(tiny.dir) == lm
    if not (QWEN_SNAPSHOT / "text_encoder").is_dir():
        return
    # The real checkpoint (headers only): 7.07 B language-model parameters, 13.2 GiB in bf16.
    assert qwen_hbm.text_encoder_lm_numel(QWEN_SNAPSHOT) == 7_070_619_136
    monkeypatch.setenv("DIFFLET_NEURON_HBM_GIB", "24")
    fits4, est4, limit = qwen_hbm.device_text_encoder_fits(QWEN_SNAPSHOT, tp_degree=4)
    fits2, est2, _ = qwen_hbm.device_text_encoder_fits(QWEN_SNAPSHOT, tp_degree=2)
    assert fits4 and round(est4 / GIB, 1) == 12.8 and round(limit / GIB, 1) == 22.8
    assert not fits2 and round(est2 / GIB, 1) == 25.6


def test_real_text_encoder_checkpoint_keys_map_onto_the_module():
    """Header-only: every language-model tensor of the real checkpoint is a parameter of the
    module (through ``checkpoint_key``) with the same shape at tp1, and only the vision tower and
    lm_head are left over."""
    if not (QWEN_SNAPSHOT / "text_encoder" / "model.safetensors.index.json").is_file():
        pytest.skip("the real Qwen-Image snapshot is not on this machine")
    import struct

    from difflet.backends.neuron.qwen_image.text_encoder import NeuronQwenTextEncoderApplication
    from difflet.models.qwen_image.modeling_qwen_text_encoder import QwenTextEncoderModel

    shapes = {}
    for path in sorted((QWEN_SNAPSHOT / "text_encoder").glob("*.safetensors")):
        with open(path, "rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        shapes.update({k: tuple(v["shape"]) for k, v in header.items() if k != "__metadata__"})
    app = NeuronQwenTextEncoderApplication(model_path=QWEN_SNAPSHOT / "text_encoder",
                                           dtype=torch.bfloat16, device="cpu")
    with torch.device("meta"):
        module = QwenTextEncoderModel(app.encoder_config, max_positions=8)
    params = {app.checkpoint_key(n): tuple(p.shape) for n, p in module.named_parameters()}
    assert all(shapes[k] == v for k, v in params.items())
    left = set(shapes) - set(params)
    assert left and all(k.startswith(app.skipped_checkpoint_prefixes) for k in left)
    assert Path(app.model_path).name == "text_encoder"
