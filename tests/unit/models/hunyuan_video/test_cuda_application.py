"""The CUDA adapter and application contracts, without encoder downloads."""

from types import SimpleNamespace
from contextlib import nullcontext

import pytest
import torch

from tests.unit.backends.test_cuda_hunyuan_transformer import _component, _inputs


def _application(tmp_path):
    from difflet.backends.cuda.hunyuan_video.application import CudaHunyuanVideoApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    app = CudaHunyuanVideoApplication(
        model_path=str(tmp_path),
        parallel=DiffletParallelConfig(tp_degree=1),
        dtype="bf16",
        shape=dict(height=32, width=32, num_frames=9),
        device="cpu",
    )
    app.config = SimpleNamespace(in_channels=2, latent_frames=3, latent_height=4, latent_width=4)
    return app


class Tokenizer:
    def __call__(self, text, **kwargs):
        # Token count includes the reference template prefix, which validation drops.
        user = text.split("\n\n")[-1].removesuffix("<|eot_id|>")
        return SimpleNamespace(input_ids=torch.ones(1, 95 + len(user.split()), dtype=torch.int64))


class Pipeline:
    tokenizer = Tokenizer()

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["output_type"] == "latent":
            return SimpleNamespace(frames=kwargs["latents"])
        frames = torch.tensor([0.0, 0.5, 1.0])[None, None, :, None, None]
        return SimpleNamespace(frames=frames.expand(1, 9, 3, 32, 32))


def test_adapter_matches_reference_and_keeps_timestep_precision(tmp_path):
    from difflet.backends.cuda.hunyuan_video.application import HunyuanVideoDiTAdapter

    ref, component = _component(tmp_path, mask_mode="compact")
    adapter = HunyuanVideoDiTAdapter(component)
    inputs = _inputs()
    with adapter.cache_context("cond"), torch.no_grad():
        expected = ref(*inputs, return_dict=False)[0]
        actual = adapter(*inputs, return_dict=False)[0]
        structured = adapter(*inputs).sample
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(structured, actual, atol=0, rtol=0)
    assert adapter.device == torch.device("cpu")


def test_application_returns_shared_video_layout_and_preserves_generation_settings(tmp_path):
    app = _application(tmp_path)
    app.pipe = Pipeline()
    noise = torch.randn(1, 2, 3, 4, 4)
    frames = app.generate("a cat", seed=9, latents=noise, num_inference_steps=7)
    assert frames.shape == (1, 3, 9, 32, 32) and frames.dtype == torch.float32
    torch.testing.assert_close(frames[0, :, 0, 0, 0], torch.tensor([-1.0, 0.0, 1.0]))
    call = app.pipe.calls[0]
    assert call["latents"] is noise and call.get("generator") is None
    assert call["guidance_scale"] == 6.0 and call["true_cfg_scale"] == 1.0
    assert call["num_inference_steps"] == 7 and call["max_sequence_length"] == 256
    assert app.generate("a cat", latents=noise, output_type="latent") is noise


def test_long_clip_prompt_is_not_rejected_or_truncated_for_llama(tmp_path):
    app = _application(tmp_path)
    app.pipe = Pipeline()
    text = " ".join(["cat"] * 100)
    app.generate(text)
    assert app.pipe.calls[-1]["prompt"] == text
    with pytest.raises(ValueError, match="256"):
        app.generate(" ".join(["cat"] * 257))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(num_frames=13),
        dict(height=48),
        dict(num_inference_steps=0),
        dict(num_inference_steps=1.9),
        dict(num_inference_steps=True),
        dict(guidance_scale=float("nan")),
        dict(negative_prompt="blurry"),
        dict(true_cfg_scale=2),
        dict(guidance_scale_2=3),
    ],
)
def test_invalid_or_unsupported_generation_is_refused_before_pipeline_call(tmp_path, kwargs):
    app = _application(tmp_path)
    app.pipe = Pipeline()
    with pytest.raises((ValueError, NotImplementedError)):
        app.generate("a cat", **kwargs)
    assert app.pipe.calls == []


def test_batch_and_unloaded_generation_are_explicit_errors(tmp_path):
    app = _application(tmp_path)
    with pytest.raises(RuntimeError, match="load"):
        app.generate("a cat")
    app.pipe = Pipeline()
    with pytest.raises(NotImplementedError, match="one prompt"):
        app.generate(["a cat", "a dog"])


def test_spec_template_matches_pinned_reference_without_importing_trainium():
    from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import DEFAULT_PROMPT_TEMPLATE
    from difflet.backends.cuda.hunyuan_video.prompt import LLAMA_TEMPLATE, LLAMA_CROP_START

    assert LLAMA_TEMPLATE == DEFAULT_PROMPT_TEMPLATE["template"]
    assert LLAMA_CROP_START == DEFAULT_PROMPT_TEMPLATE["crop_start"]


@pytest.mark.parametrize("flag", ["load_text_encoders", "load_vae"])
def test_generation_without_required_companions_errors_early(tmp_path, flag):
    app = _application(tmp_path)
    app.pipe = Pipeline()
    setattr(app, flag, False)
    with pytest.raises(RuntimeError, match="requires"):
        app.generate("a cat")
    assert app.pipe.calls == []


def test_real_diffusers_driver_reconstructed_with_adapter_matches_tiny_latent_trajectory(
    tmp_path, monkeypatch
):
    from diffusers import FlowMatchEulerDiscreteScheduler, HunyuanVideoPipeline
    from difflet.backends.cuda.hunyuan_video.application import HunyuanVideoDiTAdapter

    ref, component = _component(tmp_path, mask_mode="compact")
    app = _application(tmp_path)
    app.dtype = torch.float32
    app.config = component.dit_config
    app.transformer = component
    app.adapter = HunyuanVideoDiTAdapter(component)
    app.load_text_encoders = app.load_vae = False

    def make_pipeline(transformer):
        return HunyuanVideoPipeline(
            transformer=transformer,
            text_encoder=None,
            tokenizer=None,
            text_encoder_2=None,
            tokenizer_2=None,
            vae=None,
            scheduler=FlowMatchEulerDiscreteScheduler(shift=7.0),
        )

    def load_base(*args, **kwargs):
        assert (
            kwargs["transformer"]
            is kwargs["text_encoder"]
            is kwargs["text_encoder_2"]
            is kwargs["vae"]
            is None
        )
        return make_pipeline(None)

    monkeypatch.setattr(HunyuanVideoPipeline, "from_pretrained", load_base)
    app.load_companions()
    reference = make_pipeline(ref)
    noise, _, text, mask, pooled, _ = _inputs()
    kwargs = dict(
        prompt_embeds=text,
        prompt_attention_mask=mask,
        pooled_prompt_embeds=pooled,
        latents=noise,
        height=32,
        width=32,
        num_frames=9,
        num_inference_steps=3,
        max_sequence_length=5,
        guidance_scale=6.0,
        output_type="latent",
    )
    actual_trajectory, ref_trajectory = [], []

    def capture(destination):
        def callback(pipe, step, time, values):
            destination.append(values["latents"].clone())
            return values

        return callback

    with torch.no_grad():
        expected = reference(**kwargs, callback_on_step_end=capture(ref_trajectory)).frames
        actual = app.pipe(**kwargs, callback_on_step_end=capture(actual_trajectory)).frames
    assert len(actual_trajectory) == len(ref_trajectory) == 3
    for a, b in zip(actual_trajectory, ref_trajectory):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
        assert a.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_real_bf16_driver_preserves_reference_loop_rounding_and_actual_timestep_values():
    from diffusers import FlowMatchEulerDiscreteScheduler, HunyuanVideoPipeline
    from difflet.backends.cuda.hunyuan_video.application import HunyuanVideoDiTAdapter

    class Predictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
            self.config = self.dit_config = SimpleNamespace(in_channels=2)
            self.observed = []

        def cache_context(self, name):
            return nullcontext()

        @property
        def dtype(self):
            return self.anchor.dtype

        @property
        def device(self):
            return self.anchor.device

        def forward(
            self,
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            pooled_projections,
            guidance,
            return_dict=False,
            **kwargs,
        ):
            self.observed.append((timestep.dtype, float(timestep[0])))
            return (torch.zeros_like(hidden_states),)

    def pipe(transformer):
        return HunyuanVideoPipeline(
            transformer=transformer,
            text_encoder=None,
            tokenizer=None,
            text_encoder_2=None,
            tokenizer_2=None,
            vae=None,
            scheduler=FlowMatchEulerDiscreteScheduler(shift=7.0),
        )

    reference, component = Predictor(), Predictor()
    stock, adapted = pipe(reference), pipe(HunyuanVideoDiTAdapter(component))
    noise, _, text, mask, pooled, _ = _inputs()
    kw = dict(
        latents=noise,
        prompt_embeds=text,
        prompt_attention_mask=mask,
        pooled_prompt_embeds=pooled,
        height=32,
        width=32,
        num_frames=9,
        num_inference_steps=3,
        guidance_scale=6.0,
        output_type="latent",
    )
    with torch.no_grad():
        expected, actual = stock(**kw).frames, adapted(**kw).frames
    assert expected.dtype == actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert [d for d, _ in reference.observed] == [torch.float32, torch.bfloat16, torch.bfloat16]
    assert [d for d, _ in component.observed] == [torch.float32] * 3
    assert (
        [v for _, v in reference.observed]
        == [v for _, v in component.observed]
        == [1000.0, 932.0, 776.0]
    )
    assert float(stock.scheduler.timesteps[1]) != component.observed[1][1]
