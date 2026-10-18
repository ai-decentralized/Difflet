"""CPU coverage for the Wan CUDA application (CUDA plan F1, task 3).

No GPU: ``device="cpu"`` runs the load path the pod runs. The two experts are tiny diffusers
``WanTransformer3DModel`` twins (the recipe of ``tests/unit/backends/test_cuda_wan_transformer.py``)
saved as ``transformer/`` and ``transformer_2/``; ``model_index.json`` carries
``boundary_ratio: 0.875`` and ``scheduler/`` a UniPC flow config whose four steps are
999, 900, 750, 500 — two on each side of the boundary. UMT5 is never loaded: a fake text
encoder is injected with ``load_text_encoder=False, text_encoder=...`` (the UMT5 callable is
tested on its own against a fake ``UMT5EncoderModel``). The VAE is a tiny real
``AutoencoderKLWan`` where decoding is under test, and absent otherwise.

The spies wrap each expert's ``forward``, so every assertion is about calls that went through
the application's own device-moving DiT callable into the Difflet module.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest
import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanTransformer3DModel

from difflet.models.wan.entry import create_wan_application
from difflet.pipeline.parallel_config import DiffletParallelConfig

# Task 2's tiny expert (tests/unit/backends/test_cuda_wan_transformer.py).
TINY = dict(
    patch_size=(1, 2, 2),
    num_attention_heads=2,
    attention_head_dim=8,
    in_channels=4,
    out_channels=4,
    text_dim=16,
    freq_dim=32,
    ffn_dim=32,
    num_layers=1,
    cross_attn_norm=True,
    qk_norm="rms_norm_across_heads",
    eps=1e-6,
)
# 16x16 pixels, 5 frames -> a 2x2x2 latent (vae 8x spatial, 4x temporal).
SHAPE = {"height": 16, "width": 16, "num_frames": 5}
TEXT_SEQ_LEN = 6
LATENT_SHAPE = (1, 4, 2, 2, 2)
BOUNDARY = 0.875 * 1000
TINY_VAE = dict(
    base_dim=8,
    z_dim=4,
    dim_mult=[1, 1, 1, 1],
    num_res_blocks=1,
    temperal_downsample=[False, True, True],
    latents_mean=[0.1, -0.2, 0.3, 0.0],
    latents_std=[1.5, 0.5, 2.0, 1.0],
)


@pytest.fixture(autouse=True)
def _cpu_backend(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")


# ------------------------------------------------------------------ fixtures


def _write_model(root, *, transformer_2=True, vae=False):
    """A diffusers-shaped Wan 2.2 snapshot: experts, model_index, scheduler, optional VAE."""
    for seed, name in ((0, "transformer"), (1, "transformer_2")):
        if name == "transformer_2" and not transformer_2:
            continue
        torch.manual_seed(seed)
        WanTransformer3DModel(**TINY).eval().save_pretrained(str(root / name))
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "WanPipeline", "boundary_ratio": 0.875})
    )
    UniPCMultistepScheduler(
        prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0
    ).save_pretrained(str(root / "scheduler"))
    reference_vae = None
    if vae:
        torch.manual_seed(2)
        reference_vae = AutoencoderKLWan(**TINY_VAE).eval()
        reference_vae.save_pretrained(str(root / "vae"))
    return reference_vae


class FakeTextEncoder:
    """Embeddings that depend on the token ids, so cond and uncond differ."""

    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, attention_mask):
        self.calls.append((input_ids.clone(), attention_mask.clone()))
        ramp = torch.linspace(-1.0, 1.0, TINY["text_dim"])
        return torch.sin(input_ids.float()[..., None] * ramp)


class StubTokenizer:
    """``""`` is one token; any other prompt is three."""

    def __call__(self, prompts, padding, truncation, max_length, return_tensors):
        ids = torch.zeros(len(prompts), max_length, dtype=torch.int64)
        mask = torch.zeros(len(prompts), max_length, dtype=torch.int64)
        for row, prompt in enumerate(prompts):
            tokens = [1] if prompt == "" else [5, 6, 7]
            ids[row, : len(tokens)] = torch.tensor(tokens)
            mask[row, : len(tokens)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def _build(root, *, parallel=None, dtype="bf16", **kwargs):
    return create_wan_application(
        model_path=str(root),
        parallel=parallel or DiffletParallelConfig(tp_degree=1),
        dtype=dtype,
        shape=dict(SHAPE),
        backend="cuda",
        text_seq_len=TEXT_SEQ_LEN,
        **kwargs,
    )


def _loaded_app(tmp_path, *, vae=False, **kwargs):
    root = tmp_path / "wan"
    reference_vae = _write_model(root, vae=vae)
    text_encoder = FakeTextEncoder()
    app = _build(
        root,
        dtype="fp32",
        device="cpu",
        load_text_encoder=False,
        text_encoder=text_encoder,
        load_vae=vae,
        **kwargs,
    )
    cache = tmp_path / "cache"
    app.compile(str(cache))
    app.load(str(cache), skip_warmup=True)
    app.pipeline._tokenizer = StubTokenizer()
    return app, text_encoder, reference_vae


def _spy_experts(monkeypatch, app):
    calls = []
    for name in ("transformer", "transformer_2"):
        expert = getattr(app, name)
        if expert is None:
            continue
        original = expert.forward

        def spy(hidden_states, timestep, encoder_hidden_states, _name=name, _orig=original):
            out = _orig(hidden_states, timestep, encoder_hidden_states)
            calls.append(
                SimpleNamespace(
                    expert=_name,
                    timestep=float(timestep[0]),
                    text=encoder_hidden_states.detach().clone(),
                    out=out.detach().clone(),
                )
            )
            return out

        monkeypatch.setattr(expert, "forward", spy)
    return calls


def _spy_scheduler_steps(monkeypatch, pipeline):
    steps = []
    original = pipeline._scheduler_step

    def spy(noise_pred, timestep, latents, num_inference_steps):
        steps.append(SimpleNamespace(timestep=float(timestep), noise_pred=noise_pred.clone()))
        return original(noise_pred, timestep, latents, num_inference_steps)

    monkeypatch.setattr(pipeline, "_scheduler_step", spy)
    return steps


def _expected_timesteps(root, steps=4):
    scheduler = UniPCMultistepScheduler.from_pretrained(str(root / "scheduler"))
    scheduler.set_timesteps(steps)
    return [float(t) for t in scheduler.timesteps]


def _encoded(prompt):
    """What the orchestrator should hand the DiT for ``prompt`` (zero-padded fake embeds)."""
    tokenized = StubTokenizer()([prompt], "max_length", True, TEXT_SEQ_LEN, "pt")
    embeds = FakeTextEncoder()(tokenized["input_ids"], tokenized["attention_mask"])
    return embeds * tokenized["attention_mask"][..., None]


# ------------------------------------------------------------------ residency / lifecycle


def test_both_experts_are_resident_by_default_on_cuda(tmp_path):
    from difflet.backends.cuda.wan.transformer import CudaWanTransformerApplication

    root = tmp_path / "wan"
    _write_model(root)
    app = _build(root, device="cpu")
    assert app.enable_transformer_2 is True
    assert isinstance(app.transformer, CudaWanTransformerApplication)
    assert isinstance(app.transformer_2, CudaWanTransformerApplication)
    # Each expert owns its own diffusers directory (Task 2 ruling).
    assert app.transformer.model_path == str(root / "transformer")
    assert app.transformer_2.model_path == str(root / "transformer_2")
    assert app.transformer_2.dit_config.text_seq_len == TEXT_SEQ_LEN

    assert _build(root, device="cpu", enable_transformer_2=False).transformer_2 is None

    single = tmp_path / "wan21"
    _write_model(single, transformer_2=False)  # a Wan 2.1 snapshot has one expert
    assert _build(single, device="cpu").transformer_2 is None


def test_compile_and_load_cover_both_experts(tmp_path):
    root = tmp_path / "wan"
    _write_model(root)
    app = _build(root, dtype="fp32", device="cpu", load_text_encoder=False, load_vae=False)
    cache = tmp_path / "cache"
    assert app.has_compiled_artifacts(str(cache)) is False
    app.compile(str(cache))
    # One artifact directory per expert: two shared would overwrite each other.
    assert (cache / "transformer" / "cuda_manifest.json").is_file()
    assert (cache / "transformer_2" / "cuda_manifest.json").is_file()
    assert app.has_compiled_artifacts(str(cache)) is True
    app.load(str(cache))  # with warm-up: each expert runs its example inputs
    assert app.transformer.is_loaded and app.transformer_2.is_loaded
    assert app.pipeline.transformer is not None and app.pipeline.transformer_2 is not None
    assert app.pipeline.boundary_ratio == pytest.approx(0.875)

    (cache / "transformer_2" / "cuda_manifest.json").unlink()
    assert app.has_compiled_artifacts(str(cache)) is False


def test_lifecycle_without_a_transformer_is_an_explicit_error(tmp_path):
    app = _build(tmp_path)  # no transformer/config.json: the early return
    assert app.transformer is None and app.transformer_2 is None
    assert app.has_compiled_artifacts(str(tmp_path)) is False
    with pytest.raises(NotImplementedError, match="transformer/config.json"):
        app.compile(str(tmp_path))
    with pytest.raises(NotImplementedError, match="transformer/config.json"):
        app.load(str(tmp_path))
    with pytest.raises(RuntimeError, match="load"):
        app.generate("a prompt")


def test_cuda_application_signatures_match_the_pipeline_contract(tmp_path):
    app = _build(tmp_path)
    assert "debug" in inspect.signature(app.compile).parameters
    load_params = inspect.signature(app.load).parameters
    for name in ("start_rank_id", "local_ranks_size", "skip_warmup"):
        assert name in load_params


def test_dit_input_contract_matches_the_expert(tmp_path):
    root = tmp_path / "wan"
    _write_model(root)
    contract = _build(root, device="cpu").dit_input_contract()
    assert contract["hidden_states"] == {"shape": LATENT_SHAPE, "dtype": torch.bfloat16}
    assert contract["timestep"] == {"shape": (1,), "dtype": torch.bfloat16}
    assert contract["encoder_hidden_states"] == {
        "shape": (1, TEXT_SEQ_LEN, TINY["text_dim"]),
        "dtype": torch.bfloat16,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"teacache_cadence": 2},
        {"teacache_speedup": 1.5},
        {"teacache_calibration_path": "/x.json"},
        {"shapes": [[480, 832, 9], [480, 832, 17]]},
        {"torch_compile": True},
    ],
)
def test_cuda_wan_application_refuses_f1b_kwargs(tmp_path, kwargs):
    with pytest.raises(NotImplementedError, match="F1b"):
        _build(tmp_path, **kwargs)


def test_off_values_of_f1b_kwargs_are_accepted(tmp_path):
    # The CLI/pipeline pass several of these with their "off" value; only a request is refused.
    app = _build(tmp_path, teacache_cadence=None, teacache_probe_enabled=False, shapes=None)
    assert type(app).__name__ == "CudaWanApplication"


# ------------------------------------------------------------------ the denoise loop


@pytest.mark.parametrize("enable_transformer_2", [True, False])
def test_generate_routes_steps_to_the_boundary_expert(tmp_path, monkeypatch, enable_transformer_2):
    app, _, _ = _loaded_app(tmp_path, enable_transformer_2=enable_transformer_2)
    calls = _spy_experts(monkeypatch, app)
    latents = app.generate(
        "a cat",
        num_inference_steps=4,
        guidance_scale=1.0,
        guidance_scale_2=1.0,
        seed=0,
        output_type="latent",
    )
    assert latents.shape == LATENT_SHAPE and latents.dtype == torch.float32

    timesteps = _expected_timesteps(tmp_path / "wan")
    assert timesteps == [999.0, 900.0, 750.0, 500.0]  # two steps on each side of 875
    assert [c.timestep for c in calls] == timesteps  # guidance 1.0: one pass per step
    if enable_transformer_2:
        expected = ["transformer" if t >= BOUNDARY else "transformer_2" for t in timesteps]
    else:
        expected = ["transformer"] * 4  # the high-noise expert runs every step
    assert [c.expert for c in calls] == expected


def test_generate_runs_cond_and_uncond_passes(tmp_path, monkeypatch):
    app, text_encoder, _ = _loaded_app(tmp_path)
    calls = _spy_experts(monkeypatch, app)
    steps = _spy_scheduler_steps(monkeypatch, app.pipeline)
    app.generate(
        "a cat",
        num_inference_steps=4,
        guidance_scale=4.0,
        guidance_scale_2=4.0,
        seed=0,
        output_type="latent",
    )

    assert len(calls) == 8  # cond + uncond per step
    cond_text, uncond_text = _encoded("a cat"), _encoded("")
    assert not torch.equal(cond_text, uncond_text)
    for index, step in enumerate(steps):
        cond, uncond = calls[2 * index], calls[2 * index + 1]
        assert cond.timestep == uncond.timestep == step.timestep
        assert cond.expert == uncond.expert
        torch.testing.assert_close(cond.text, cond_text)
        # negative_prompt=None: the orchestrator encodes "" (a real EOS embedding).
        torch.testing.assert_close(uncond.text, uncond_text)
        torch.testing.assert_close(step.noise_pred, uncond.out + 4.0 * (cond.out - uncond.out))
    # The prompt and the empty negative are each encoded once per call to the encoder.
    assert {int(mask.sum()) for _, mask in text_encoder.calls} == {1, 3}


def test_guidance_scale_2_is_used_below_the_boundary(tmp_path, monkeypatch):
    app, _, _ = _loaded_app(tmp_path)
    calls = _spy_experts(monkeypatch, app)
    steps = _spy_scheduler_steps(monkeypatch, app.pipeline)
    app.generate(
        "a cat",
        num_inference_steps=4,
        guidance_scale=4.0,
        guidance_scale_2=2.0,
        seed=0,
        output_type="latent",
    )

    assert [s.timestep for s in steps] == _expected_timesteps(tmp_path / "wan")
    for index, step in enumerate(steps):
        cond, uncond = calls[2 * index], calls[2 * index + 1]
        scale = 4.0 if step.timestep >= BOUNDARY else 2.0
        assert cond.expert == ("transformer" if scale == 4.0 else "transformer_2")
        torch.testing.assert_close(step.noise_pred, uncond.out + scale * (cond.out - uncond.out))
    # The low-noise steps really did not use 4.0 (the outputs differ enough to tell).
    low_cond, low_uncond = calls[-2], calls[-1]
    assert not torch.allclose(
        steps[-1].noise_pred, low_uncond.out + 4.0 * (low_cond.out - low_uncond.out)
    )


def test_generate_is_seeded_on_the_host_and_one_shape(tmp_path):
    app, _, _ = _loaded_app(tmp_path)
    kwargs = dict(num_inference_steps=2, guidance_scale=1.0, guidance_scale_2=1.0)
    first = app.generate("a cat", seed=7, output_type="latent", **kwargs)
    again = app.generate("a cat", seed=7, output_type="latent", **kwargs)
    torch.testing.assert_close(first, again)
    noise = torch.randn(LATENT_SHAPE, generator=torch.Generator().manual_seed(3))
    verbatim = app.generate("a cat", latents=noise, output_type="latent", **kwargs)
    torch.testing.assert_close(
        verbatim, app.generate("a cat", latents=noise.clone(), output_type="latent", **kwargs)
    )
    with pytest.raises(ValueError, match="num_frames"):
        app.generate("a cat", num_frames=9, output_type="latent", **kwargs)
    # The orchestrator's own refusal: no decoder was loaded or injected.
    with pytest.raises(ValueError, match="VAE decoder"):
        app.generate("a cat", seed=0, output_type="pt", **kwargs)


def test_call_with_a_prompt_dispatches_to_generate(tmp_path):
    app, _, _ = _loaded_app(tmp_path)
    kwargs = dict(num_inference_steps=1, guidance_scale=1.0, guidance_scale_2=1.0, seed=0)
    torch.testing.assert_close(
        app(prompt="a cat", output_type="latent", **kwargs),
        app.generate("a cat", output_type="latent", **kwargs),
    )


# ------------------------------------------------------------------ the callables


def test_dit_callable_moves_to_the_model_dtype_and_returns_host_tensors():
    from difflet.models.wan.cuda_application import _CudaDiT

    seen = []

    def expert(hidden_states, timestep, encoder_hidden_states):
        seen.append((hidden_states, timestep, encoder_hidden_states))
        return hidden_states * 2

    expert.dit_config = SimpleNamespace(cfg_parallel_enabled=False)
    dit = _CudaDiT(expert, torch.bfloat16, torch.device("cpu"))
    assert dit.dtype is torch.bfloat16  # the orchestrator casts per step from this
    assert dit.config.cfg_parallel_enabled is False
    out = dit(torch.ones(LATENT_SHAPE), torch.tensor([999.0]), torch.ones(1, 6, 16))
    assert {t.dtype for t in seen[0]} == {torch.bfloat16}
    assert out.device.type == "cpu" and out.dtype == torch.bfloat16
    assert torch.equal(out, torch.full(LATENT_SHAPE, 2.0, dtype=torch.bfloat16))


def test_text_encoder_callable_pads_to_text_seq_len(tmp_path, monkeypatch):
    import transformers

    from difflet.models.wan.cuda_application import _CudaTextEncoder

    class FakeUMT5(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, input_ids, attention_mask):
            self.calls.append((input_ids.clone(), attention_mask.clone()))
            hidden = input_ids.float()[..., None] + torch.arange(16.0) / 16  # fp32
            return SimpleNamespace(last_hidden_state=hidden)

    fake = FakeUMT5()
    loads = []

    def from_pretrained(path, **kwargs):
        loads.append((path, kwargs))
        return fake

    monkeypatch.setattr(transformers.UMT5EncoderModel, "from_pretrained", from_pretrained)
    encoder = _CudaTextEncoder(str(tmp_path), 16, 8, torch.bfloat16, torch.device("cpu"))
    ((path, load_kwargs),) = loads
    assert path == str(tmp_path / "text_encoder")
    assert load_kwargs["dtype"] is torch.bfloat16

    input_ids = torch.tensor([[5, 6, 7, 0, 0, 0, 0, 0], [8, 9, 0, 0, 0, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]])
    embeds = encoder(input_ids, mask)

    # The model sees only the valid prefix (the longest row's), not the 512-style padding.
    ((seen_ids, seen_mask),) = fake.calls
    assert seen_ids.shape == (2, 3) and torch.equal(seen_mask, mask[:, :3])
    assert embeds.shape == (2, 8, 16)
    assert embeds.dtype is torch.bfloat16 and embeds.device.type == "cpu"
    expected = (input_ids.float()[..., None] + torch.arange(16.0) / 16) * mask[..., None]
    torch.testing.assert_close(embeds, expected.to(torch.bfloat16))
    # Zero past each row's mask, including row 1's position 2 inside the trimmed prefix.
    assert embeds[0, 3:].eq(0).all() and embeds[1, 2:].eq(0).all()


def test_vae_decoder_callable_exposes_config_and_decodes(tmp_path):
    from difflet.models.wan.cuda_application import _CudaVaeDecoder

    torch.manual_seed(2)
    vae = AutoencoderKLWan(**TINY_VAE).eval()
    decoder = _CudaVaeDecoder(vae)
    # _decode_latents reads these to denormalise exactly as diffusers' WanPipeline does.
    assert list(decoder.config.latents_mean) == TINY_VAE["latents_mean"]
    assert list(decoder.config.latents_std) == TINY_VAE["latents_std"]
    assert decoder.dtype is torch.float32

    latents = torch.randn(LATENT_SHAPE)
    frames = decoder(latents)
    with torch.no_grad():
        expected = vae.decode(latents, return_dict=False)[0]
    assert frames.shape == (1, 3, 5, 16, 16)
    assert frames.dtype is torch.float32 and frames.device.type == "cpu"
    torch.testing.assert_close(frames, expected)  # nothing else: no post-processing
    assert frames.min() >= -1.0 and frames.max() <= 1.0


def test_generate_pt_decodes_through_the_vae(tmp_path, monkeypatch):
    app, _, reference_vae = _loaded_app(tmp_path, vae=True)
    assert app.pipeline.vae_decoder is app.vae_decoder
    kwargs = dict(num_inference_steps=2, guidance_scale=1.0, guidance_scale_2=1.0, seed=0)
    latents = app.generate("a cat", output_type="latent", **kwargs)
    frames = app.generate("a cat", output_type="pt", **kwargs)

    mean = torch.tensor(TINY_VAE["latents_mean"]).view(1, -1, 1, 1, 1)
    std = torch.tensor(TINY_VAE["latents_std"]).view(1, -1, 1, 1, 1)
    with torch.no_grad():
        expected = reference_vae.decode(latents * std + mean, return_dict=False)[0]
    assert frames.shape == (1, 3, 5, 16, 16) and frames.dtype is torch.float32
    torch.testing.assert_close(frames, expected)
