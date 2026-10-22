"""CPU coverage for the Flux GPU application's FluxPipeline wiring.

No GPU: ``device="cpu"`` runs the same load path a GPU host runs. A tiny diffusers
``FluxTransformer2DModel`` checkpoint (the recipe of
``tests/unit/backends/test_cuda_flux_transformer.py``) is the oracle; the text encoders and
VAE are never loaded here — the real ``FluxPipeline.from_pretrained`` is replaced by a
recorder that returns a component-less pipeline.
"""

from __future__ import annotations

import inspect
import json
import os

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel
from safetensors.torch import save_file

# With no backend set, bind difflet.ops to the CPU ops for the modules imported below (a
# Trainium host would bind the Neuron ops).
_PINNED = "DIFFLET_BACKEND" not in os.environ
os.environ.setdefault("DIFFLET_BACKEND", "cpu")

from difflet.backends.gpu.application import create_gpu_application  # noqa: E402
import difflet.models.flux.modeling_flux  # noqa: E402,F401  (bound to the CPU ops)
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402

if _PINNED:
    del os.environ["DIFFLET_BACKEND"]

AXES_DIMS_ROPE = (2, 2, 4)  # even axes summing to attention_head_dim
TINY = dict(
    patch_size=1,
    in_channels=4,
    num_layers=1,
    num_single_layers=1,
    attention_head_dim=8,
    num_attention_heads=2,
    joint_attention_dim=16,
    pooled_projection_dim=8,
)


@pytest.fixture(autouse=True)
def _cpu_backend(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")


def _build(tmp_path, *, dtype="bf16", **kwargs):
    return create_gpu_application(
        "flux",
        model_path=str(tmp_path),
        parallel=DiffletParallelConfig(tp_degree=1),
        dtype=dtype,
        shape={"height": 64, "width": 64},
        **kwargs,
    )


def _write_tiny_checkpoint(root, *, guidance_embeds=True) -> FluxTransformer2DModel:
    """A diffusers ``transformer/`` dir (config.json + one unindexed shard)."""
    torch.manual_seed(0)
    reference = FluxTransformer2DModel(
        **TINY, axes_dims_rope=AXES_DIMS_ROPE, guidance_embeds=guidance_embeds
    ).eval()
    tdir = root / "transformer"
    tdir.mkdir(parents=True)
    config = {**TINY, "axes_dims_rope": list(AXES_DIMS_ROPE), "guidance_embeds": guidance_embeds}
    (tdir / "config.json").write_text(json.dumps(config))
    state = {k: v.detach().clone().contiguous() for k, v in reference.state_dict().items()}
    save_file(state, str(tdir / "diffusion_pytorch_model.safetensors"))
    return reference


@pytest.fixture
def from_pretrained_recorder(monkeypatch):
    """Replace FluxPipeline.from_pretrained: record the kwargs, return a bare pipeline."""
    calls = []

    def fake(model_path, **kwargs):
        calls.append((model_path, kwargs))
        return FluxPipeline(
            scheduler=FlowMatchEulerDiscreteScheduler(),
            vae=None,
            text_encoder=None,
            tokenizer=None,
            text_encoder_2=None,
            tokenizer_2=None,
            transformer=None,
        )

    monkeypatch.setattr(FluxPipeline, "from_pretrained", staticmethod(fake))
    return calls


def _loaded_tiny_app(tmp_path, *, guidance_embeds=True):
    reference = _write_tiny_checkpoint(tmp_path, guidance_embeds=guidance_embeds)
    app = _build(tmp_path, dtype="fp32", device="cpu", load_text_encoders=False, load_vae=False)
    cache = tmp_path / "cache"
    app.compile(str(cache))
    assert app.has_compiled_artifacts(str(cache))
    app.load(str(cache), skip_warmup=True)
    return reference, app


def _dit_inputs():
    torch.manual_seed(1)
    # 64x64 pixels -> 8x8 VAE latent -> 4x4 packed tokens = 16 image tokens.
    return dict(
        hidden_states=torch.randn(1, 16, 4),
        encoder_hidden_states=torch.randn(1, 6, 16),
        pooled_projections=torch.randn(1, 8),
        timestep=torch.tensor([0.5]),
        img_ids=FluxPipeline._prepare_latent_image_ids(1, 4, 4, "cpu", torch.float32),
        txt_ids=torch.zeros(6, 3),
    )


# ----------------------------------------------------------------- application


def test_load_builds_flux_pipeline_around_the_adapter(tmp_path, from_pretrained_recorder):
    _, app = _loaded_tiny_app(tmp_path)
    ((model_path, kwargs),) = from_pretrained_recorder
    assert model_path == str(tmp_path)
    # diffusers refuses a non-ModelMixin transformer in from_pretrained (see the next test),
    # so the transformer is skipped there and the adapter registered on the pipeline.
    assert kwargs["transformer"] is None
    assert kwargs["torch_dtype"] is torch.float32 and kwargs["local_files_only"] is True
    # load_text_encoders=False / load_vae=False: those components are not loaded at all.
    assert kwargs["text_encoder"] is None and kwargs["text_encoder_2"] is None
    assert kwargs["vae"] is None
    assert app.pipe.transformer is app.adapter
    assert app.adapter.app is app.transformer


def test_flux_pipeline_drives_the_adapter_like_the_diffusers_transformer(
    tmp_path, from_pretrained_recorder, monkeypatch
):
    """FluxPipeline built with the adapter as its transformer: registered, survives
    ``pipe.to``, reports a device, and denoises to the same latents as the same pipeline
    around diffusers' own transformer."""
    # CPU tensors: keep FluxPipeline off torch_xla's mark_step on a Neuron host.
    monkeypatch.setattr("diffusers.pipelines.flux.pipeline_flux.XLA_AVAILABLE", False)
    reference, app = _loaded_tiny_app(tmp_path)
    adapter = app.adapter
    pipe = FluxPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        vae=None,
        text_encoder=None,
        tokenizer=None,
        text_encoder_2=None,
        tokenizer_2=None,
        transformer=adapter,
    )
    assert pipe.transformer is adapter
    assert pipe.to("cpu") is pipe and pipe.transformer is adapter
    assert pipe.device == torch.device("cpu")  # the adapter is the pipeline's only module

    oracle = FluxPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        vae=None,
        text_encoder=None,
        tokenizer=None,
        text_encoder_2=None,
        tokenizer_2=None,
        transformer=reference,
    )
    torch.manual_seed(2)
    common = dict(
        prompt_embeds=torch.randn(1, 6, 16),
        pooled_prompt_embeds=torch.randn(1, 8),
        latents=torch.randn(1, 16, 4),
        height=64,
        width=64,
        num_inference_steps=2,
        guidance_scale=3.5,
        output_type="latent",
    )
    with torch.no_grad():
        actual = pipe(**common).images
        expected = oracle(**common).images
    assert actual.shape == expected.shape == (1, 16, 4)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_dit_contract_and_forward_dit_match_diffusers(tmp_path, from_pretrained_recorder):
    from difflet.models.flux.contract import FluxDiTInputBundle

    reference, app = _loaded_tiny_app(tmp_path)
    contract = app.dit_input_contract()
    assert contract["hidden_states"]["shape"] == (1, 16, 4)
    assert contract["encoder_hidden_states"]["shape"] == (1, 512, 16)
    assert contract["pooled_projections"]["shape"] == (1, 8)
    assert contract["timestep"]["shape"] == (1,)
    assert contract["guidance"]["shape"] == (1,)
    assert contract["image_rotary_emb"]["shape"] == (512 + 16, 8, 2)
    assert all(spec["dtype"] is torch.float32 for spec in contract.values())

    inputs = _dit_inputs()
    guidance = torch.tensor([3.5])
    bundle = FluxDiTInputBundle(
        hidden_states=inputs["hidden_states"],
        encoder_hidden_states=inputs["encoder_hidden_states"],
        pooled_projections=inputs["pooled_projections"],
        timestep=inputs["timestep"],
        guidance=guidance,
        image_rotary_emb=app.adapter.rotary_embedding(inputs["txt_ids"], inputs["img_ids"]),
    )
    with torch.no_grad():
        expected = reference(**inputs, guidance=guidance, return_dict=False)[0]
        from_bundle = app(bundle)
        from_kwargs = app(**{f: getattr(bundle, f) for f in bundle.__dataclass_fields__})
    torch.testing.assert_close(from_bundle, expected, rtol=1e-4, atol=1e-4)
    assert torch.equal(from_kwargs, from_bundle)


def test_bundle_order_is_the_dit_positional_order():
    from difflet.models.flux.contract import FluxDiTInputBundle
    from difflet.models.flux.modeling_flux import NeuronFluxTransformer2DModel

    names = list(inspect.signature(NeuronFluxTransformer2DModel.forward).parameters)[1:7]
    tensors = {name: torch.tensor([float(i)]) for i, name in enumerate(names)}
    bundle = FluxDiTInputBundle(**tensors)
    assert [float(t) for t in bundle.as_model_inputs()] == [float(i) for i in range(6)]


def test_schnell_contract_has_an_empty_guidance(tmp_path, from_pretrained_recorder):
    _, app = _loaded_tiny_app(tmp_path, guidance_embeds=False)
    assert app.dit_input_contract()["guidance"]["shape"] == (0,)


class _RecordingPipe:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return type("Out", (), {"images": ["image"]})()


def test_generate_calls_flux_pipeline_with_the_planned_arguments(tmp_path):
    app = _build(tmp_path, device="cpu")
    app.pipe = _RecordingPipe()
    noise = torch.randn(1, 16, 4)
    images = app.generate("a cat", seed=7, latents=noise)
    assert images == ["image"]
    (kw,) = app.pipe.calls
    assert kw["prompt"] == "a cat" and kw["latents"] is noise
    assert kw["height"] == 64 and kw["width"] == 64  # the application's shape by default
    assert kw["num_inference_steps"] == 28 and kw["guidance_scale"] == 3.5
    assert kw["output_type"] == "pil" and kw["max_sequence_length"] == 512
    expected = torch.Generator(device="cpu").manual_seed(7)
    assert torch.equal(kw["generator"].get_state(), expected.get_state())


def test_generate_forwards_overrides_and_returns_the_output_for_pt(tmp_path):
    app = _build(tmp_path, device="cpu")
    app.pipe = _RecordingPipe()
    out = app.generate(
        "a cat", num_inference_steps=4, guidance_scale=5.0, height=128, width=96, output_type="pt"
    )
    assert out.images == ["image"]  # "pt"/"latent": the FluxPipelineOutput itself
    (kw,) = app.pipe.calls
    assert kw["num_inference_steps"] == 4 and kw["guidance_scale"] == 5.0
    assert (kw["height"], kw["width"]) == (128, 96) and kw["generator"] is None
