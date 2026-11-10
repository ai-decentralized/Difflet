"""Standard CFG and the real HunyuanVideo 1.5 driver, without encoder downloads."""

from types import MethodType, SimpleNamespace

import pytest
import torch
from diffusers import (
    ClassifierFreeGuidance,
    FlowMatchEulerDiscreteScheduler,
    HunyuanVideo15Pipeline,
)
from diffusers.models.modeling_utils import ModelMixin

from tests.unit.backends.test_cuda_hunyuan15_transformer import _component, _inputs


def _application(tmp_path):
    from difflet.backends.cuda.hunyuan_video_15.application import CudaHunyuanVideo15Application
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    return CudaHunyuanVideo15Application(
        model_path=str(tmp_path),
        parallel=DiffletParallelConfig(tp_degree=1),
        dtype="fp32",
        shape=dict(height=32, width=32, num_frames=9),
        device="cpu",
    )


class TinyVAE(ModelMixin):
    temporal_compression_ratio = 4
    spatial_compression_ratio = 16

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(latent_channels=4)


def _pipeline(transformer):
    pipe = HunyuanVideo15Pipeline(
        transformer=transformer,
        vae=TinyVAE(),
        scheduler=FlowMatchEulerDiscreteScheduler(shift=5),
        text_encoder=None,
        tokenizer=None,
        text_encoder_2=None,
        tokenizer_2=None,
        guider=ClassifierFreeGuidance(guidance_scale=6.0),
    )
    pipe.vision_num_semantic_tokens = 3

    def encode(self, prompt=None, **kwargs):
        inputs = _inputs()
        # Deterministic conditioning distinguishes prompt, glyph and negative branches.
        offset = sum(ord(c) for c in (prompt or "")) % 17 / 17
        glyph = '"' in (prompt or "")
        return (
            (inputs["encoder_hidden_states"] + offset).to(kwargs["dtype"]),
            inputs["encoder_attention_mask"],
            (inputs["encoder_hidden_states_2"] - offset).to(kwargs["dtype"]),
            (
                inputs["encoder_attention_mask_2"]
                if glyph
                else torch.zeros_like(inputs["encoder_attention_mask_2"])
            ),
        )

    pipe.encode_prompt = MethodType(encode, pipe)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _assert_driver_close(actual, expected, dtype, *, bf16_fraction):
    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    else:
        bound = bf16_fraction * expected.abs().max().item()
        torch.testing.assert_close(actual, expected, atol=bound, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hv15_real_driver_cfg_aba_and_negative_prompts(tmp_path, monkeypatch, dtype):
    from difflet.backends.cuda.hunyuan_video_15.application import HunyuanVideo15DiTAdapter

    reference, component = _component(tmp_path, dtype=dtype)
    reference.to(dtype)
    if dtype == torch.bfloat16:
        import difflet.backends.cuda.hunyuan_video_15.modeling as modeling
        from difflet.backends.cuda.ops_impl.attention import attention

        monkeypatch.setattr(modeling, "attention", attention)
    app = _application(tmp_path)
    app.config = component.dit_config
    app.transformer = component
    app.adapter = HunyuanVideo15DiTAdapter(component)
    app.pipe = _pipeline(app.adapter)
    app.load_text_encoders = True
    # Encoding is substituted only at the text-encoder boundary above.
    app.pipe.tokenizer = object()
    app._guider_config = dict(app.pipe.guider.config)
    noise = torch.randn(1, 4, 3, 2, 2, generator=torch.Generator().manual_seed(8))
    outputs = []
    for prompt, negative, scale in [
        ("a cat", "", 6.0),
        ('a sign "你好"', "blur", 1.0),
        ("a cat", "", 6.0),
        ("a cat", "motion", 6.0),
    ]:
        stock = _pipeline(reference)
        stock.guider = ClassifierFreeGuidance(guidance_scale=scale)
        # Capture each actual denoiser input/output, hence every scheduler transition.
        stock_calls = []
        candidate_calls = []
        ref_hook = reference.register_forward_hook(
            lambda m, a, k, y: stock_calls.append(y[0].clone()), with_kwargs=True
        )
        candidate_hook = component.module.register_forward_hook(
            lambda m, a, k, y: candidate_calls.append(y[0].clone()), with_kwargs=True
        )
        try:
            with torch.no_grad():
                expected = stock(
                    prompt=prompt,
                    negative_prompt=negative,
                    latents=noise.clone(),
                    height=32,
                    width=32,
                    num_frames=9,
                    num_inference_steps=3,
                    output_type="latent",
                ).frames
                actual = app.generate(
                    prompt,
                    negative_prompt=negative,
                    guidance_scale=scale,
                    latents=noise.clone(),
                    num_inference_steps=3,
                    output_type="latent",
                )
        finally:
            ref_hook.remove()
            candidate_hook.remove()
        assert len(stock_calls) == len(candidate_calls) == (3 if scale == 1 else 6)
        # FP32 checks the driver, its CFG scale and negative prompts included, to FP32
        # rounding. In BF16 the valid-prefix attention rounds differently from stock's masked
        # attention, by an amount that depends on the CPU's BF16 kernels and scales with the
        # largest values rather than each element; on this tiny model that can exceed what a
        # wrong scale or prompt changes, so BF16 checks only that the path stays within BF16
        # resolution of stock: each denoiser call to 2^-6 of its peak magnitude, and the
        # final latents, where CFG at scale s can multiply a per-call difference by up to
        # 2s - 1, to 2^-4.
        for a, b in zip(candidate_calls, stock_calls):
            _assert_driver_close(a, b, dtype, bf16_fraction=2**-6)
        _assert_driver_close(actual, expected, dtype, bf16_fraction=2**-4)
        outputs.append(actual)
    torch.testing.assert_close(outputs[0], outputs[2], atol=0, rtol=0)
    assert not torch.equal(outputs[0], outputs[3])


class Pipe:
    tokenizer = object()

    def __init__(self):
        self.guider = ClassifierFreeGuidance(guidance_scale=6)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append((kwargs, self.guider))
        return SimpleNamespace(frames=torch.full((1, 9, 3, 32, 32), 0.5))


def test_hv15_generation_contract_and_guider_reset(tmp_path):
    app = _application(tmp_path)
    app.config = SimpleNamespace(out_channels=4, latent_frames=3, latent_height=2, latent_width=2)
    app.pipe = Pipe()
    app._guider_config = dict(app.pipe.guider.config)
    output = app.generate('a sign "你好"', negative_prompt="blur", seed=7, num_inference_steps=3)
    assert output.shape == (1, 3, 9, 32, 32) and output.dtype == torch.float32
    assert torch.count_nonzero(output) == 0
    first = app.pipe.calls[0][1]
    app.generate("a cat", guidance_scale=1)
    assert app.pipe.calls[1][1] is not first
    assert app.pipe.calls[1][1].config.guidance_scale == 1
    assert app.pipe.calls[0][0]["negative_prompt"] == "blur"
    assert app.pipe.calls[0][0]["prompt"] == 'a sign "你好"'


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(num_inference_steps=0),
        dict(width=48),
        dict(num_frames=8),
        dict(output_type="pil"),
        dict(latents=torch.zeros(1, 4, 3, 2, 3)),
    ],
)
def test_hv15_generation_refusals_precede_driver(tmp_path, kwargs):
    app = _application(tmp_path)
    app.pipe = Pipe()
    app.config = SimpleNamespace(out_channels=4, latent_frames=3, latent_height=2, latent_width=2)
    with pytest.raises(ValueError):
        app.generate("a cat", **kwargs)
    with pytest.raises(ValueError, match="one prompt"):
        app.generate(["a cat", "a dog"])
    assert not app.pipe.calls


def test_hv15_load_companions_preserves_reference_text_contract(tmp_path, monkeypatch):
    from difflet.backends.cuda.hunyuan_video_15 import application
    from tests.unit.models.hunyuan_video.test_native_vae_tiling import _vae

    _, component = _component(tmp_path)
    app = _application(tmp_path)
    app.config = component.dit_config
    app.transformer = component
    app.adapter = application.HunyuanVideo15DiTAdapter(component)
    monkeypatch.setattr(application, "configure_conditioning", lambda pipe: None)

    def load(*args, **kwargs):
        assert kwargs["transformer"] is None
        # The real constructor, so the reference's text contract is what is checked.
        return HunyuanVideo15Pipeline(
            transformer=None,
            vae=_vae(),
            text_encoder=torch.nn.Identity(),
            tokenizer=None,
            text_encoder_2=torch.nn.Identity(),
            tokenizer_2=None,
            scheduler=FlowMatchEulerDiscreteScheduler(shift=5),
            guider=ClassifierFreeGuidance(guidance_scale=6),
        )

    monkeypatch.setattr(HunyuanVideo15Pipeline, "from_pretrained", load)
    app.load_companions()
    assert app.pipe.prompt_template_encode_start_idx == 108
    assert app.pipe.tokenizer_max_length == 1000 and app.pipe.tokenizer_2_max_length == 256
    assert app.pipe.vision_num_semantic_tokens == 729
