"""NeuronHunyuanVideoApplication.__call__ dispatch routing.

Closes the cclog 29c day-3 item ``__call__ / app routing``: verify
external callers can drive the full hybrid path via the public app
``__call__`` (the path ``NovaPipeline.from_pretrained(...).__call__``
hands kwargs through to). Distinct from
``test_hunyuan_video_pipeline_orchestrator.py`` which exercises the
orchestrator in isolation — this file checks the dispatch *into* the
orchestrator from the application layer.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from nova.models.hunyuan_video.application import (
    HunyuanVideoDiTInputBundle,
    NeuronHunyuanVideoApplication,
)
from nova.models.hunyuan_video.pipeline import (
    HunyuanVideoOrchestrator,
    HunyuanVideoPipelineOutput,
)
from nova.pipeline.parallel_config import NovaParallelConfig


def _fake_transformer_config() -> SimpleNamespace:
    """Minimal config mirroring fields read by validate_hunyuan_video_dit_inputs."""
    return SimpleNamespace(
        neuron_config=SimpleNamespace(batch_size=1),
        text_seq_len=4,
        in_channels=16,
        latent_frames=2,
        latent_height=2,
        latent_width=2,
        text_embed_dim=8,
        pooled_projection_dim=5,
    )


class _FakeTransformer:
    def __init__(self, value: float = 0.25):
        self.dtype = torch.float32
        self.value = value
        self.config = _fake_transformer_config()
        self.bundles: list[HunyuanVideoDiTInputBundle] = []

    def __call__(self, *args, **kwargs):
        # Accept either positional 6-tuple (forward_dit -> transformer(*bundle.as_model_inputs()))
        # or a positional Bundle from orchestrator path.
        if len(args) == 6:
            bundle = HunyuanVideoDiTInputBundle(*args)
        elif len(args) == 1 and isinstance(args[0], HunyuanVideoDiTInputBundle):
            bundle = args[0]
        else:
            raise AssertionError(f"unexpected transformer call args={args} kwargs={kwargs}")
        self.bundles.append(bundle)
        return {"sample": torch.full_like(bundle.hidden_states, self.value)}


class _FakeVAE:
    def __init__(self):
        self.dtype = torch.float32
        self.config = SimpleNamespace(scaling_factor=2.0)
        self.inputs: list[torch.Tensor] = []

    def decode(self, latents, return_dict):
        assert return_dict is False
        self.inputs.append(latents.detach().clone())
        return (latents[:, :3],)


def _build_app(tmp_path) -> NeuronHunyuanVideoApplication:
    """Skeleton app with no transformer/VAE compiled artifacts."""
    return NeuronHunyuanVideoApplication(
        model_path=str(tmp_path),
        parallel=NovaParallelConfig(tp_degree=1, cp_enabled=False),
        dtype=torch.float32,
        shape={"height": 320, "width": 512, "num_frames": 61},
        text_seq_len=4,
    )


def _install_fake_components(
    app: NeuronHunyuanVideoApplication,
    *,
    transformer: _FakeTransformer | None = None,
    vae: _FakeVAE | None = None,
) -> tuple[_FakeTransformer | None, _FakeVAE | None]:
    transformer = transformer or _FakeTransformer()
    vae = vae or _FakeVAE()
    app.transformer = transformer
    app.pipeline = HunyuanVideoOrchestrator(
        model_path=app.model_path,
        transformer=transformer,
        vae=vae,
        dtype=torch.float32,
    )
    return transformer, vae


def _bundle() -> HunyuanVideoDiTInputBundle:
    return HunyuanVideoDiTInputBundle(
        hidden_states=torch.zeros((1, 16, 2, 2, 2), dtype=torch.float32),
        timestep=torch.zeros([1], dtype=torch.float32),
        encoder_hidden_states=torch.ones((1, 4, 8), dtype=torch.float32),
        encoder_attention_mask=torch.ones((1, 4), dtype=torch.int64),
        pooled_projections=torch.ones((1, 5), dtype=torch.float32),
        guidance=torch.ones([1], dtype=torch.float32) * 6000.0,
    )


def test_app_call_routes_positional_bundle_to_forward_dit(tmp_path):
    """`app(bundle)` as a positional Bundle → forward_dit, not orchestrator."""
    app = _build_app(tmp_path)
    transformer, _ = _install_fake_components(app)

    out = app(_bundle())
    assert isinstance(out, dict) and "sample" in out
    assert len(transformer.bundles) == 1


def test_app_call_routes_named_tensors_to_forward_dit(tmp_path):
    """`app(hidden_states=..., timestep=..., ...)` with 6 tensor kwargs → forward_dit."""
    app = _build_app(tmp_path)
    transformer, _ = _install_fake_components(app)
    bundle = _bundle()

    out = app(
        hidden_states=bundle.hidden_states,
        timestep=bundle.timestep,
        encoder_hidden_states=bundle.encoder_hidden_states,
        encoder_attention_mask=bundle.encoder_attention_mask,
        pooled_projections=bundle.pooled_projections,
        guidance=bundle.guidance,
    )
    assert isinstance(out, dict) and "sample" in out
    assert len(transformer.bundles) == 1


def test_app_call_routes_bundle_kwarg_through_orchestrator_pt(tmp_path):
    """`app(bundle=..., output_type='pt')` exercises the full hybrid path:
    app.__call__ → orchestrator.__call__ → fake DiT loop → fake VAE decode.
    This is the public M3 v0 end-to-end entry point."""
    app = _build_app(tmp_path)
    transformer, vae = _install_fake_components(app)

    output = app(
        bundle=_bundle(),
        timesteps=torch.tensor([1000.0, 500.0]),
        output_type="pt",
    )
    assert isinstance(output, HunyuanVideoPipelineOutput)
    # FakeVAE returns latents[:, :3]; with output_type='pt' frames go through decode.
    assert output.frames.shape == (1, 3, 2, 2, 2)
    # 2-step denoise → 2 transformer calls + 1 VAE decode.
    assert len(transformer.bundles) == 2
    assert len(vae.inputs) == 1


def test_app_call_routes_bundle_kwarg_through_orchestrator_latent(tmp_path):
    """`app(bundle=..., output_type='latent')` returns latents without VAE."""
    app = _build_app(tmp_path)
    transformer, vae = _install_fake_components(app)

    output = app(
        bundle=_bundle(),
        timesteps=torch.tensor([1000.0]),
        output_type="latent",
    )
    assert isinstance(output, HunyuanVideoPipelineOutput)
    assert output.frames.shape == (1, 16, 2, 2, 2)
    assert len(transformer.bundles) == 1
    assert len(vae.inputs) == 0  # VAE not called when output_type='latent'


def test_app_call_raises_when_pipeline_has_no_runtime_components(tmp_path):
    """Skeleton app: pipeline exists but has no transformer/VAE.
    `has_runtime_components()` is False → unsupported kwargs raise
    NotImplementedError instead of silently no-op."""
    app = _build_app(tmp_path)
    # Skeleton app: transformer is None; pipeline exists but has no runtime components.
    assert app.transformer is None
    assert app.pipeline is not None
    assert app.pipeline.has_runtime_components() is False

    with pytest.raises(NotImplementedError, match="HunyuanVideo end-to-end inference"):
        app(prompt="a cat walking")


def test_app_call_orchestrator_path_rejects_prompt_string(tmp_path):
    """Per cclog 29 §0/§1.1, text encoding is host-side (cache helper),
    not embedded in the orchestrator. `app(prompt=...)` must not silently
    succeed — it must raise so external callers see the contract."""
    app = _build_app(tmp_path)
    _install_fake_components(app)

    # Orchestrator.__call__ doesn't accept `prompt` as a parameter, so
    # routing through `app.pipeline(prompt=...)` raises TypeError.
    with pytest.raises(TypeError):
        app(prompt="a cat walking")
