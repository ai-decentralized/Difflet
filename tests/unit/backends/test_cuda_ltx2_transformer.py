"""CPU oracle coverage for the LTX-2 GPU transformer component."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

GEOMETRY = dict(height=64, width=96, num_frames=17, text_seq_len=6)
TINY = dict(
    in_channels=8,
    out_channels=8,
    num_attention_heads=4,
    attention_head_dim=8,
    cross_attention_dim=32,
    audio_in_channels=8,
    audio_out_channels=8,
    audio_num_attention_heads=4,
    audio_attention_head_dim=4,
    audio_cross_attention_dim=16,
    caption_channels=24,
    num_layers=2,
    rope_type="split",
)


@pytest.fixture(autouse=True)
def _clean_gpu_mesh():
    from difflet.backends.cuda.ops_impl import parallel_mesh

    parallel_mesh.destroy_parallel_mesh()
    yield
    parallel_mesh.destroy_parallel_mesh()


def _api():
    assert importlib.util.find_spec("difflet.backends.cuda.ltx_2.dit_config") is not None
    from difflet.backends.cuda.ltx_2.dit_config import Ltx2DiTConfig
    from difflet.backends.cuda.ltx_2.transformer import (
        CudaLtx2TransformerApplication,
        Ltx2DiTAdapter,
    )

    return Ltx2DiTConfig, CudaLtx2TransformerApplication, Ltx2DiTAdapter


def _checkpoint(root):
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        source = LTX2VideoTransformer3DModel(**TINY).eval().requires_grad_(False)
        with torch.no_grad():
            for param in source.parameters():
                param.add_(0.03 * torch.randn_like(param))
    source.save_pretrained(root, max_shard_size="60KB")
    assert (root / "diffusion_pytorch_model.safetensors.index.json").is_file()
    # Independently reload through upstream, never through Difflet's loader/model builder.
    return LTX2VideoTransformer3DModel.from_pretrained(root).eval().requires_grad_(False)


def _bundle(reference, *, scale=1.0):
    from difflet.models.ltx_2.contract import LTX2DiTInputBundle

    gen = torch.Generator().manual_seed(456)
    video_coords = reference.rope.prepare_video_coords(1, 3, 2, 3, device="cpu", fps=24.0)
    audio_coords = reference.audio_rope.prepare_audio_coords(1, 18, device="cpu")
    return LTX2DiTInputBundle(
        torch.randn(1, 18, 8, generator=gen) * scale,
        torch.randn(1, 18, 8, generator=gen) * scale,
        torch.randn(1, 6, 24, generator=gen) * scale,
        torch.randn(1, 6, 24, generator=gen) * scale,
        torch.tensor([700.125 * scale]),
        torch.tensor([0.700125 * scale]),
        torch.tensor([[True, False, True, True, False, True]]),
        torch.tensor([[False, True, True, False, True, False]]),
        video_coords,
        audio_coords,
    )


def _kwargs(bundle):
    return dict(
        hidden_states=bundle.hidden_states,
        audio_hidden_states=bundle.audio_hidden_states,
        encoder_hidden_states=bundle.encoder_hidden_states,
        audio_encoder_hidden_states=bundle.audio_encoder_hidden_states,
        timestep=bundle.timestep,
        audio_timestep=bundle.timestep,
        sigma=bundle.sigma,
        audio_sigma=bundle.sigma,
        encoder_attention_mask=bundle.encoder_attention_mask,
        audio_encoder_attention_mask=bundle.audio_encoder_attention_mask,
        video_coords=bundle.video_coords,
        audio_coords=bundle.audio_coords,
        num_frames=3,
        height=2,
        width=3,
        audio_num_frames=18,
        fps=24.0,
        isolate_modalities=False,
        use_cross_timestep=False,
        spatio_temporal_guidance_blocks=None,
        perturbation_mask=None,
        attention_kwargs=None,
        return_dict=False,
    )


def _loaded(tmp_path, *, compile_blocks=False, dtype=torch.float32):
    Config, Component, Adapter = _api()
    directory = tmp_path / "transformer"
    reference = _checkpoint(directory).to(dtype=dtype)
    cfg = Config.from_pretrained(directory, torch_dtype=dtype, **GEOMETRY)
    app = Component(
        model_path=str(directory), config=cfg, device="cpu", torch_compile=compile_blocks
    )
    cache = tmp_path / "cache"
    app.compile(cache)
    app.load(cache, skip_warmup=compile_blocks)
    return reference, app, Adapter(app)


def test_feature_available():
    _api()


def test_strict_sharded_load_and_masked_dual_stream_oracle(tmp_path):
    import diffusers.models.transformers.transformer_ltx2 as upstream

    original_rope = upstream.apply_split_rotary_emb
    reference, app, _ = _loaded(tmp_path)
    assert upstream.apply_split_rotary_emb is original_rope
    assert app.load_report == {"missing": [], "unexpected": []}
    assert app.has_compiled_artifacts(tmp_path / "cache")
    assert not app.module.training
    assert all(not p.requires_grad and not p.is_meta for p in app.module.parameters())
    prior = None
    for scale in (1.0, 0.37):
        bundle = _bundle(reference, scale=scale)
        with torch.no_grad():
            expected = reference(**_kwargs(bundle))
            actual = app(*bundle.as_model_inputs())
        for got, want in zip(actual, expected, strict=True):
            assert got.shape == (1, 18, 8)
            torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)
        if prior is not None:
            assert all(not torch.allclose(a, b) for a, b in zip(prior, actual, strict=True))
        prior = actual
    # Ragged masks must affect the outputs, so silently dropping masks is caught.
    kwargs = _kwargs(bundle)
    kwargs.update(encoder_attention_mask=None, audio_encoder_attention_mask=None)
    with torch.no_grad():
        unmasked = reference(**kwargs)
    assert all(not torch.allclose(a, b) for a, b in zip(actual, unmasked, strict=True))


def test_adapter_preserves_upstream_kwargs_outputs_and_float_timestep(tmp_path):
    reference, app, adapter = _loaded(tmp_path)
    assert adapter.config.in_channels == 8
    assert adapter.dtype == torch.float32 and adapter.device == torch.device("cpu")
    coords = adapter.rope.prepare_video_coords(1, 3, 2, 3, device="cpu", fps=19.0)
    assert coords.shape == (1, 3, 18, 2)
    assert adapter.audio_rope.prepare_audio_coords(1, 18, device="cpu").shape == (1, 1, 18, 2)
    bundle = _bundle(reference)
    kwargs = _kwargs(bundle)
    kwargs.update(
        audio_timestep=torch.tensor([123.375]),
        audio_sigma=torch.tensor([0.123375]),
        isolate_modalities=True,
        use_cross_timestep=True,
        video_coords=coords,
        fps=19.0,
    )
    with torch.no_grad(), adapter.cache_context("cond"):
        actual = adapter(**kwargs)
        expected = reference(**kwargs)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)
    kwargs["return_dict"] = True
    with torch.no_grad():
        got, want = adapter(**kwargs), reference(**kwargs)
    torch.testing.assert_close(got.sample, want.sample)
    torch.testing.assert_close(got.audio_sample, want.audio_sample)
    # Switching just the video timestep moves both coupled outputs; no dtype rounding.
    kwargs["return_dict"] = False
    kwargs["isolate_modalities"] = False
    kwargs["timestep"] = torch.tensor([700.499])
    with torch.no_grad():
        changed = adapter(**kwargs)
        wanted = reference(**kwargs)
    for got, want in zip(changed, wanted, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("broken", ["missing", "shape", "unexpected"])
def test_malformed_checkpoint_is_refused(tmp_path, broken):
    from safetensors.torch import load_file, save_file

    Config, Component, _ = _api()
    directory = tmp_path / "transformer"
    _checkpoint(directory)
    index_path = directory / "diffusion_pytorch_model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    key = "proj_in.weight"
    shard_path = directory / index["weight_map"][key]
    shard = load_file(shard_path)
    if broken == "missing":
        del index["weight_map"][key]
        del shard[key]
    elif broken == "shape":
        shard[key] = shard[key][:1].clone()
    else:
        shard["unrecognized.weight"] = torch.ones(1)
        index["weight_map"]["unrecognized.weight"] = shard_path.name
    save_file(shard, shard_path)
    index_path.write_text(json.dumps(index))
    app = Component(
        model_path=str(directory),
        config=Config.from_pretrained(directory, torch_dtype=torch.float32, **GEOMETRY),
        device="cpu",
    )
    app.compile(tmp_path / "cache")
    with pytest.raises(ValueError, match="checkpoint|parameter"):
        app.load(tmp_path / "cache")
    assert not app.is_loaded


@pytest.mark.parametrize(
    "change",
    [
        {"height": 65},
        {"width": 0},
        {"num_frames": 16},
        {"num_frames": 0},
    ],
)
def test_runtime_config_refusals(tmp_path, change):
    Config, _, _ = _api()
    directory = tmp_path / "transformer"
    _checkpoint(directory)
    with pytest.raises((ValueError, NotImplementedError)):
        Config.from_pretrained(directory, **{**GEOMETRY, **change})


@pytest.mark.parametrize(
    "change",
    [
        {"_class_name": "OtherTransformer"},
        {"gated_attn": True},
        {"audio_cross_attn_mod": True},
        {"use_prompt_embeddings": False},
        {"patch_size": 2},
        {"vae_scale_factors": [8, 16, 16]},
    ],
)
def test_checkpoint_config_refusals(tmp_path, change):
    Config, _, _ = _api()
    directory = tmp_path / "transformer"
    _checkpoint(directory)
    path = directory / "config.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **change}))
    with pytest.raises((ValueError, NotImplementedError)):
        Config.from_pretrained(directory, **GEOMETRY)


def test_lifecycle_and_ten_tensor_examples(tmp_path):
    Config, Component, Adapter = _api()
    directory = tmp_path / "transformer"
    _checkpoint(directory)
    cfg = Config.from_pretrained(directory, **GEOMETRY)
    assert (cfg.latent_num_frames, cfg.latent_height, cfg.latent_width) == (3, 2, 3)
    assert (cfg.video_seq_len, cfg.audio_seq_len) == (18, 18)
    app = Component(model_path=str(directory), config=cfg, device="cpu")
    with pytest.raises(RuntimeError, match="load"):
        app(*app.get_example_inputs())
    with pytest.raises(RuntimeError, match="load"):
        Adapter(app)
    tensors = app.get_example_inputs()
    assert [tuple(t.shape) for t in tensors] == [
        (1, 18, 8),
        (1, 18, 8),
        (1, 6, 24),
        (1, 6, 24),
        (1,),
        (1,),
        (1, 6),
        (1, 6),
        (1, 3, 18, 2),
        (1, 1, 18, 2),
    ]
    assert [t.dtype for t in tensors] == [
        *([torch.bfloat16] * 4),
        torch.float32,
        torch.float32,
        torch.bool,
        torch.bool,
        torch.float32,
        torch.float32,
    ]
    with pytest.raises(FileNotFoundError):
        app.load(tmp_path / "cache")


def test_compiled_blocks_execute_and_match_oracle(tmp_path, monkeypatch):
    # Exercise the actual Dynamo wrapper and resulting graphs on CPU without Inductor.
    real_compile = torch.compile
    graphs = []

    def backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    def cpu_compile(module, **kwargs):
        kwargs.pop("mode", None)
        kwargs.pop("options", None)
        return real_compile(module, backend=backend, **kwargs)

    monkeypatch.setattr(torch, "compile", cpu_compile)
    reference, app, _ = _loaded(tmp_path, compile_blocks=True)
    from torch._dynamo.eval_frame import OptimizedModule

    blocks = app.module.transformer.transformer_blocks
    assert len(blocks) == 2 and all(isinstance(b, OptimizedModule) for b in blocks)
    bundle = _bundle(reference)
    with torch.no_grad():
        actual, expected = app(*bundle.as_model_inputs()), reference(**_kwargs(bundle))
    assert graphs
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)


def test_inductor_bfloat16_blocks_preserve_eager_rounding(tmp_path):
    # Removing LTX-2's per-compile precision policy must fail this regression.
    # Use real Inductor: a graph.forward backend cannot expose fused bf16 casts.
    from dataclasses import replace

    from torch._inductor import metrics

    torch._dynamo.reset()
    kernels_before = metrics.generated_kernel_count
    with torch._inductor.config.patch(emulate_precision_casts=False, compile_threads=1):
        reference, app, _ = _loaded(tmp_path, compile_blocks=True, dtype=torch.bfloat16)
        bundle = _bundle(reference)
        bundle = replace(
            bundle,
            **{
                name: getattr(bundle, name).to(torch.bfloat16)
                for name in (
                    "hidden_states",
                    "audio_hidden_states",
                    "encoder_hidden_states",
                    "audio_encoder_hidden_states",
                )
            },
        )
        with torch.no_grad():
            expected = reference(**_kwargs(bundle))
            actual = app(*bundle.as_model_inputs())
            repeated = app(*bundle.as_model_inputs())
        assert metrics.generated_kernel_count > kernels_before
        assert not torch._inductor.config.emulate_precision_casts
        for got, want, again in zip(actual, expected, repeated, strict=True):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
            torch.testing.assert_close(again, want, rtol=0, atol=0)


def _cuda_binding_check(tmp_path):
    import diffusers.models.transformers.transformer_ltx2 as upstream

    from difflet.models.ltx_2 import modeling_ltx_2

    assert upstream.apply_split_rotary_emb.__module__ == upstream.__name__
    reference, app, _ = _loaded(tmp_path)
    assert modeling_ltx_2.get_tensor_model_parallel_size.__module__.startswith(
        "difflet.backends.cuda"
    )
    bundle = _bundle(reference)
    with torch.no_grad():
        actual, expected = app(*bundle.as_model_inputs()), reference(**_kwargs(bundle))
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)
    assert not any(
        name.startswith(
            (
                "neuronx_distributed",
                "torch_neuronx",
                "difflet.backends.neuron.",
                "difflet.backends.trainium.",
            )
        )
        for name in sys.modules
    )


def test_cuda_dispatch_with_cpu_tensors_isolated(tmp_path):
    _api()
    script = (
        "import runpy; from pathlib import Path; ns=runpy.run_path("
        + repr(__file__)
        + "); ns['_cuda_binding_check'](Path("
        + repr(str(tmp_path))
        + "))"
    )
    env = {
        **os.environ,
        "DIFFLET_BACKEND": "cuda",
        "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
    }
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_bfloat16_adapter_keeps_fractional_fp32_timesteps(tmp_path):
    from dataclasses import replace

    reference, app, adapter = _loaded(tmp_path, dtype=torch.bfloat16)
    bundle = _bundle(reference)
    bundle = replace(
        bundle,
        **{
            name: getattr(bundle, name).to(torch.bfloat16)
            for name in (
                "hidden_states",
                "audio_hidden_states",
                "encoder_hidden_states",
                "audio_encoder_hidden_states",
            )
        },
    )
    kwargs = _kwargs(bundle)
    with torch.no_grad():
        actual = adapter(**kwargs)
        expected = reference(**kwargs)
        rounded = reference(
            **{
                **kwargs,
                "timestep": bundle.timestep.to(torch.bfloat16),
                "audio_timestep": bundle.timestep.to(torch.bfloat16),
            }
        )
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert any(
        not torch.equal(want, altered) for want, altered in zip(expected, rounded, strict=True)
    )


def test_unknown_checkpoint_keys_are_ignored(tmp_path):
    # A field a newer diffusers adds must not break loading; strict weight loading still
    # refuses a checkpoint whose layers differ.
    Config, _, _ = _api()
    directory = tmp_path / "transformer"
    _checkpoint(directory)
    before = Config.from_pretrained(directory, **GEOMETRY)
    path = directory / "config.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "future_field": 1}))
    assert Config.from_pretrained(directory, **GEOMETRY) == before


def test_adapter_reports_loaded_tensor_dtype_and_device_after_to(tmp_path):
    _, app, adapter = _loaded(tmp_path)
    adapter.to(dtype=torch.bfloat16)
    assert next(app.module.transformer.parameters()).dtype == torch.bfloat16
    assert adapter.dtype == torch.bfloat16
    # Meta is sufficient to exercise device reporting without requiring a GPU.
    adapter.to(device="meta")
    assert next(app.module.transformer.parameters()).device == torch.device("meta")
    assert adapter.device == torch.device("meta")


def test_official_gate_policy_covers_component_warmup_and_adapter_without_sigma_rescaling(
    tmp_path, monkeypatch
):
    from difflet.backends.cuda.ltx_2 import official_compatibility as policy

    Config, Component, Adapter = _api()
    directory = tmp_path / "transformer"
    reference = _checkpoint(directory).bfloat16()
    cfg = Config.from_pretrained(directory, torch_dtype=torch.bfloat16, **GEOMETRY)
    seen = []
    configure = policy.configure_official_forward

    def instrument(model):
        result = configure(model)
        for name in ["av_cross_attn_video_a2v_gate", "av_cross_attn_audio_v2a_gate"]:
            getattr(model, name).register_forward_hook(
                lambda module, args, result, name=name: seen.append(
                    (name, args[0].detach().clone())
                )
            )
        return result

    monkeypatch.setattr(policy, "configure_official_forward", instrument)
    app = Component(
        model_path=str(directory), config=cfg, device="cpu", generation_profile="ltx_official_bf16"
    )
    cache = tmp_path / "cache"
    app.compile(cache)
    app.load(cache)
    assert seen  # The actual component warmup reached both gates.
    factor = cfg.cross_attn_timestep_scale_multiplier / cfg.timestep_scale_multiplier
    expected = app.get_example_inputs()[5].flatten() * factor
    assert all(torch.equal(timestep, expected) for _, timestep in seen)
    seen.clear()
    bundle = _bundle(reference)
    inputs = tuple(
        value.bfloat16() if index < 4 else value
        for index, value in enumerate(bundle.as_model_inputs())
    )
    with torch.no_grad():
        app(*inputs)
    assert len(seen) == 2 and all(
        torch.equal(timestep, inputs[5].flatten() * factor) for _, timestep in seen
    )
    seen.clear()
    kwargs = _kwargs(bundle)
    for key in [
        "hidden_states",
        "audio_hidden_states",
        "encoder_hidden_states",
        "audio_encoder_hidden_states",
    ]:
        kwargs[key] = kwargs[key].bfloat16()
    kwargs["sigma"] = torch.tensor([0.123])
    kwargs["audio_sigma"] = torch.tensor([0.456])
    with torch.no_grad():
        Adapter(app)(**kwargs)
    assert len(seen) == 2
    assert torch.equal(seen[0][1], kwargs["audio_sigma"] * factor)
    assert torch.equal(seen[1][1], kwargs["sigma"] * factor)
