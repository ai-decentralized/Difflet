"""Device-local rotary initialization and request-scoped Qwen attention policy."""

from types import SimpleNamespace
import importlib

import pytest
import torch
import transformers

# The CUDA stack's Transformers (5.x) has the Qwen2.5-VL rotary API the conditioning uses;
# the Trainium stacks pin 4.x.
pytestmark = pytest.mark.skipif(
    int(transformers.__version__.split(".")[0]) < 5,
    reason="HunyuanVideo 1.5 CUDA conditioning needs Transformers 5",
)


def conditioning():
    return importlib.import_module("difflet.backends.cuda.hunyuan_video_15.conditioning")


def tiny_qwen():
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel

    config = Qwen2_5_VLTextConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1000000.0,
            "mrope_section": [1, 1, 2],
        },
    )
    return Qwen2_5_VLTextModel(config).eval()


def test_rotary_initialization_uses_model_device_without_changing_parameters():
    m = conditioning()
    qwen = tiny_qwen()
    before = {k: p.detach().clone() for k, p in qwen.named_parameters()}
    config = qwen.config.to_dict()
    expected, scale = qwen.rotary_emb.compute_default_rope_parameters(qwen.rotary_emb.config)
    m.initialize_qwen_rotary(qwen)
    for name in ("inv_freq", "original_inv_freq"):
        value = getattr(qwen.rotary_emb, name)
        assert value.dtype == torch.float32 and value.device == next(qwen.parameters()).device
        assert torch.equal(value, expected)
        assert qwen.rotary_emb._buffers[name] is value
        assert name in qwen.rotary_emb._non_persistent_buffers_set
    assert qwen.rotary_emb.attention_scaling == scale
    assert qwen.config.to_dict() == config
    assert all(torch.equal(p, before[k]) for k, p in qwen.named_parameters())


def test_actual_qwen_encoding_uses_policy_and_restores_it(monkeypatch):
    m = conditioning()
    qwen = tiny_qwen()
    seen = []

    def encode_prompt(input_ids):
        seen.append(torch.backends.cuda.cudnn_sdp_enabled())
        return qwen(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False
        ).last_hidden_state

    pipe = SimpleNamespace(text_encoder=qwen, encode_prompt=encode_prompt)
    old = torch.backends.cuda.cudnn_sdp_enabled()
    torch.backends.cuda.enable_cudnn_sdp(True)
    try:
        m.configure_conditioning(pipe)
        got = pipe.encode_prompt(torch.tensor([[1, 2, 3]]))
        assert got.shape == (1, 3, 16) and torch.isfinite(got).all()
        assert seen == [False] and torch.backends.cuda.cudnn_sdp_enabled()
    finally:
        torch.backends.cuda.enable_cudnn_sdp(old)


@pytest.mark.parametrize("initial", [False, True])
def test_conditioning_policy_restores_on_exception(initial):
    m = conditioning()
    qwen = tiny_qwen()

    def broken():
        assert not torch.backends.cuda.cudnn_sdp_enabled()
        raise RuntimeError("encoding failed")

    pipe = SimpleNamespace(text_encoder=qwen, encode_prompt=broken)
    old = torch.backends.cuda.cudnn_sdp_enabled()
    torch.backends.cuda.enable_cudnn_sdp(initial)
    try:
        m.configure_conditioning(pipe)
        with pytest.raises(RuntimeError, match="encoding failed"):
            pipe.encode_prompt()
        assert torch.backends.cuda.cudnn_sdp_enabled() == initial
    finally:
        torch.backends.cuda.enable_cudnn_sdp(old)


def test_application_load_wires_policy_for_direct_pipe_and_generate(monkeypatch):
    from difflet.backends.cuda.hunyuan_video_15.application import CudaHunyuanVideo15Application
    import diffusers

    seen = []
    qwen = tiny_qwen()

    class VAE(torch.nn.Identity):
        def enable_tiling(self):
            pass

    class Pipe:
        def __init__(self, **components):
            self.components = components
            self.__dict__.update(components)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls(
                text_encoder=qwen,
                text_encoder_2=torch.nn.Identity(),
                vae=VAE(),
                tokenizer=object(),
                guider=SimpleNamespace(config={}),
            )

        def encode_prompt(self, *args, **kwargs):
            seen.append(("encode", torch.backends.cuda.cudnn_sdp_enabled()))

        def __call__(self, **kwargs):
            self.encode_prompt(kwargs["prompt"])
            seen.append(("after_encode", torch.backends.cuda.cudnn_sdp_enabled()))
            return SimpleNamespace(frames=torch.zeros(1))

    monkeypatch.setattr(diffusers, "HunyuanVideo15Pipeline", Pipe)
    app = object.__new__(CudaHunyuanVideo15Application)
    app.model_path = "unused"
    app.dtype = torch.float32
    app.device = torch.device("cpu")
    app.local_files_only = True
    app.config = SimpleNamespace(out_channels=2)
    app.adapter = object()
    app.shape = dict(height=480, width=848, num_frames=121)
    old = torch.backends.cuda.cudnn_sdp_enabled()
    torch.backends.cuda.enable_cudnn_sdp(True)
    try:
        app.load_companions()
        app.pipe.encode_prompt("a cat")
        app.generate("a cat", output_type="latent")
        assert seen == [("encode", False), ("encode", False), ("after_encode", True)]
        assert torch.backends.cuda.cudnn_sdp_enabled()
    finally:
        torch.backends.cuda.enable_cudnn_sdp(old)


def test_rotary_initializer_uses_encoder_device_instead_of_ambient_context():
    m = conditioning()
    qwen = tiny_qwen()
    with torch.device("meta"):
        m.initialize_qwen_rotary(qwen)
    assert qwen.rotary_emb.inv_freq.device.type == "cpu"
    assert torch.isfinite(qwen.rotary_emb.inv_freq).all()


def test_conditioning_preserves_other_sdpa_flags_and_other_instances():
    m = conditioning()
    states = lambda: (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
    )
    expected = states()
    seen = []
    configured = SimpleNamespace(
        text_encoder=tiny_qwen(), encode_prompt=lambda: seen.append(states())
    )
    untouched = SimpleNamespace(encode_prompt=lambda: torch.backends.cuda.cudnn_sdp_enabled())
    before = torch.backends.cuda.cudnn_sdp_enabled()
    m.configure_conditioning(configured)
    configured.encode_prompt()
    assert seen == [expected] and states() == expected and untouched.encode_prompt() == before
