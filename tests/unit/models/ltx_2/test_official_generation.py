"""Host arithmetic and request contracts for LTX-2's explicit BF16 profile."""

import importlib.util
import inspect
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch


def _api():
    assert importlib.util.find_spec("difflet.backends.cuda.ltx_2.official") is not None
    from difflet.backends.cuda.ltx_2 import official

    return official


CONFIG = dict(
    base_image_seq_len=1024,
    max_image_seq_len=4096,
    base_shift=0.95,
    max_shift=2.05,
    shift_terminal=0.1,
)


@pytest.mark.parametrize(
    "steps, bits",
    [
        (1, [1036831952, 0]),
        (2, [1065353208, 1036831952, 0]),
        (4, [1065353213, 1063123240, 1059171967, 1036831952, 0]),
        (
            40,
            [
                1065353215,
                1065293644,
                1065231358,
                1065166170,
                1065097869,
                1065026231,
                1064951001,
                1064871903,
                1064788638,
                1064700856,
                1064608186,
                1064510209,
                1064406452,
                1064296390,
                1064179430,
                1064054900,
                1063922045,
                1063779999,
                1063627774,
                1063464236,
                1063288075,
                1063097767,
                1062891544,
                1062667318,
                1062422624,
                1062154525,
                1061859494,
                1061533263,
                1061170600,
                1060765046,
                1060308510,
                1059790736,
                1059198520,
                1058514557,
                1057715744,
                1056576386,
                1054304366,
                1051522368,
                1047497964,
                1036831952,
                0,
            ],
        ),
    ],
)
def test_schedule_preserves_fp32_shift_and_terminal_rounding(steps, bits):
    # Literal float32 bits from the LTX one-stage schedule, including its rounded start.
    actual = _api().official_sigmas(CONFIG, steps, device="cpu")
    assert actual.dtype == torch.float32
    assert actual.view(torch.int32).tolist() == bits


def test_cfg_and_euler_preserve_each_bf16_materialization():
    api = _api()
    state = torch.tensor([1.125, -0.2421875, 0.70703125, 2.359375], dtype=torch.bfloat16)
    state = state.reshape(1, 2, 2).transpose(1, 2)
    positive = torch.tensor([0.3125, -0.7109375, 0.12109375, 1.515625], dtype=torch.bfloat16)
    negative = torch.tensor([-0.21875, 0.271484375, 0.828125, -0.09375], dtype=torch.bfloat16)
    positive, negative = positive.reshape(1, 2, 2), negative.reshape(1, 2, 2)
    sigmas = torch.tensor([0.713579, 0.1, 0.0])
    expected_velocity = [[16338, -16311, -16430, 16560], [16337, -16282, -16430, 16556]]
    expected_state = [[15858, 16425, 16196, -16510], [-17095, 16448, 16238, -16441]]
    for index in range(2):
        velocity = api.guided_velocity(state, positive, negative, sigmas[index], 3.5)
        assert velocity.dtype == torch.bfloat16
        assert velocity.view(torch.int16).flatten().tolist() == expected_velocity[index]
        state = api.euler_step(state, velocity, sigmas[index], sigmas[index + 1])
        assert state.dtype == torch.bfloat16
        assert state.view(torch.int16).flatten().tolist() == expected_state[index]


def test_final_sigmas_are_not_shifted_and_own_the_step_count():
    api = _api()
    actual = api.resolve_sigmas(CONFIG, 2, [0.8, 0.3, 0.0], device="cpu")
    assert actual.tolist() == torch.tensor([0.8, 0.3, 0.0]).tolist()
    # Omitting the terminal boundary has the same public meaning.
    torch.testing.assert_close(
        api.resolve_sigmas(CONFIG, 2, [0.8, 0.3], device="cpu"), actual, rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "sigmas",
    [
        [],
        [0.8],
        [0.8, 0.8, 0],
        [0.3, 0.8, 0],
        [0.8, 0, 0],
        [0.8, -0.2, 0],
        [float("nan"), 0.3, 0],
        [1, float("inf"), 0],
        [[0.8, 0.3, 0]],
        [True, 0.3, 0],
    ],
)
def test_invalid_final_sigmas_are_rejected(sigmas):
    with pytest.raises(ValueError):
        _api().resolve_sigmas(CONFIG, 2, sigmas, device="cpu")


def test_sigma_length_cannot_silently_change_requested_steps():
    with pytest.raises(ValueError, match="num_inference_steps"):
        _api().resolve_sigmas(CONFIG, 4, [0.8, 0.3, 0], device="cpu")


class _Transformer:
    dtype = torch.bfloat16
    config = SimpleNamespace(in_channels=2)

    def __init__(self):
        self.calls = []
        self.rope = SimpleNamespace(
            prepare_video_coords=lambda b, f, h, w, d, fps: torch.zeros(b, 3, f * h * w, 2)
        )
        self.audio_rope = SimpleNamespace(
            prepare_audio_coords=lambda b, f, d: torch.zeros(b, 1, f, 2)
        )

    def cache_context(self, name):
        return nullcontext()

    def __call__(self, **kwargs):
        from diffusers import LTX2VideoTransformer3DModel

        # A fake velocity function must not hide unsupported production kwargs.
        inspect.signature(LTX2VideoTransformer3DModel.forward).bind(self, **kwargs)
        self.calls.append(kwargs)
        video = kwargs["hidden_states"] * 0.25 + kwargs["encoder_hidden_states"][:, :1, :1]
        audio = (
            kwargs["audio_hidden_states"] * 0.25 + kwargs["audio_encoder_hidden_states"][:, :1, :1]
        )
        return video, audio


def _pipe():
    from diffusers import LTX2Pipeline

    transformer = _Transformer()
    connector_batches = []

    def connectors(embeddings, mask, padding_side):
        assert padding_side == "left"
        connector_batches.append(embeddings.shape[0])
        # Batch-sensitive outputs catch accidentally merging connector evaluations.
        return embeddings + embeddings.shape[0], 2 * embeddings + embeddings.shape[0], mask

    video = SimpleNamespace(
        dtype=torch.bfloat16,
        latents_mean=torch.zeros(2, dtype=torch.bfloat16),
        latents_std=torch.ones(2, dtype=torch.bfloat16),
        config=SimpleNamespace(scaling_factor=1.0, timestep_conditioning=False),
        decode=lambda x, timestep, return_dict: (x,),
    )
    audio = SimpleNamespace(
        dtype=torch.bfloat16,
        mel_compression_ratio=2,
        latents_mean=torch.zeros((), dtype=torch.bfloat16),
        latents_std=torch.ones((), dtype=torch.bfloat16),
        config=SimpleNamespace(mel_bins=4, latent_channels=1),
        decode=lambda x, return_dict: (x,),
    )
    pipe = SimpleNamespace(
        transformer=transformer,
        connectors=connectors,
        text_encoder=None,
        tokenizer=None,
        vae=video,
        audio_vae=audio,
        vocoder=lambda mel: mel.flatten(1),
        scheduler=SimpleNamespace(config=CONFIG),
        vae_spatial_compression_ratio=32,
        vae_temporal_compression_ratio=8,
        transformer_spatial_patch_size=1,
        transformer_temporal_patch_size=1,
        audio_sampling_rate=128,
        audio_hop_length=24,
        audio_vae_temporal_compression_ratio=1,
        audio_vae_mel_compression_ratio=2,
        video_processor=SimpleNamespace(
            postprocess_video=lambda x, output_type: x.permute(0, 2, 1, 3, 4)
        ),
        connector_batches=connector_batches,
    )
    for name in (
        "_pack_latents",
        "_unpack_latents",
        "_normalize_latents",
        "_denormalize_latents",
        "_pack_audio_latents",
        "_unpack_audio_latents",
        "_normalize_audio_latents",
        "_denormalize_audio_latents",
    ):
        setattr(pipe, name, getattr(LTX2Pipeline, name))
    return pipe


def _request():
    noise = torch.tensor([1, -0.5, 0.75, 2], dtype=torch.bfloat16).reshape(1, 2, 2)
    return dict(
        prompt=None,
        negative_prompt=None,
        height=32,
        width=32,
        num_frames=9,
        frame_rate=24,
        num_inference_steps=2,
        guidance_scale=3.5,
        generator=torch.Generator().manual_seed(12),
        latents=noise.clone(),
        audio_latents=noise.clone(),
        prompt_embeds=torch.ones(1, 4, 2, dtype=torch.bfloat16),
        negative_prompt_embeds=-torch.ones(1, 4, 2, dtype=torch.bfloat16),
        prompt_attention_mask=torch.tensor([[0, 1, 1, 1]]),
        negative_prompt_attention_mask=torch.tensor([[0, 0, 1, 1]]),
        output_type="latent",
        max_sequence_length=4,
        device="cpu",
        sigmas=[0.8, 0.3, 0],
    )


def test_profile_driver_matches_materialized_states_and_preserves_public_outputs():
    api, pipe = _api(), _pipe()
    assert hasattr(api, "generate")
    video, audio = api.generate(pipe, **_request())
    assert pipe.connector_batches == [1, 1]
    first, second = pipe.transformer.calls
    assert first["encoder_hidden_states"][:, 0, 0].tolist() == [2.0, 0.0]
    assert first["audio_encoder_hidden_states"][:, 0, 0].tolist() == [3.0, -1.0]
    assert first["encoder_attention_mask"] is None
    assert first["audio_encoder_attention_mask"] is None
    assert first["timestep"].shape == (2, 2)
    assert first["audio_timestep"].shape == (2, 2)
    assert first["timestep"].dtype == torch.float32
    assert first["timestep"].tolist() == [[800.0, 800.0], [800.0, 800.0]]
    assert second["hidden_states"][0].flatten().tolist() == [-2.625, -3.9375, -2.84375, -1.75]
    assert second["audio_hidden_states"][0].flatten().tolist() == [-5.625, -6.9375, -5.84375, -4.75]
    assert video.shape == (1, 2, 2, 1, 1) and audio.shape == (1, 1, 2, 2)
    assert video.dtype == audio.dtype == torch.bfloat16
    assert video.flatten().tolist() == [-4.5625, -4.75, -5.78125, -3.71875]
    assert audio.flatten().tolist() == [-9.0625, -10.25, -9.375, -8.25]


def test_profile_driver_decodes_both_streams_with_existing_components():
    api, pipe = _api(), _pipe()
    assert hasattr(api, "generate")
    video, audio = api.generate(pipe, **{**_request(), "output_type": "pt"})
    assert video.shape == (1, 2, 2, 1, 1)
    assert audio.tolist() == [[-9.0625, -10.25, -9.375, -8.25]]


def test_profile_noise_draws_packed_bf16_and_resets_each_request():
    api, pipe = _api(), _pipe()
    assert hasattr(api, "generate")
    request = {**_request(), "latents": None, "audio_latents": None}
    api.generate(pipe, **request)
    rng = torch.Generator().manual_seed(12)
    expected_video = torch.randn(1, 2, 2, dtype=torch.bfloat16, generator=rng)
    expected_audio = torch.randn(1, 2, 2, dtype=torch.bfloat16, generator=rng)
    first = pipe.transformer.calls[0]
    torch.testing.assert_close(first["hidden_states"][0:1], expected_video, rtol=0, atol=0)
    torch.testing.assert_close(first["audio_hidden_states"][0:1], expected_audio, rtol=0, atol=0)
    request["generator"] = torch.Generator().manual_seed(12)
    api.generate(pipe, **request)
    torch.testing.assert_close(
        pipe.transformer.calls[2]["hidden_states"], first["hidden_states"], rtol=0, atol=0
    )


class _Tokenizer:
    padding_side = "right"
    pad_token = None
    eos_token = "eos"
    bos_token_id = 7

    def __call__(self, text, **kwargs):
        # Positive has BOS already; negative requires insertion.
        return SimpleNamespace(input_ids=[7, 2] if text == "positive" else [3])

    def pad(self, payload, **kwargs):
        assert payload["input_ids"] == [[7, 2], [7, 3]]
        return SimpleNamespace(
            input_ids=torch.tensor([[0, 0, 7, 2], [0, 0, 7, 3]]),
            attention_mask=torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]]),
        )


class _TextEncoder:
    dtype = torch.bfloat16

    def __init__(self):
        self.batches = []

    def __call__(self, input_ids, attention_mask, output_hidden_states):
        self.batches.append(input_ids.shape[0])
        hidden = input_ids.to(torch.bfloat16).unsqueeze(-1)
        return SimpleNamespace(hidden_states=(hidden, hidden + 1))


def test_profile_combines_raw_prompts_before_gemma_and_keeps_connectors_separate():
    api, pipe = _api(), _pipe()
    assert hasattr(api, "generate")
    pipe.tokenizer, pipe.text_encoder = _Tokenizer(), _TextEncoder()
    request = {**_request(), "prompt": " positive ", "negative_prompt": "negative"}
    for key in (
        "prompt_embeds",
        "negative_prompt_embeds",
        "prompt_attention_mask",
        "negative_prompt_attention_mask",
    ):
        request[key] = None
    api.generate(pipe, **request)
    assert pipe.text_encoder.batches == [2]
    assert pipe.connector_batches == [1, 1]
    assert pipe.tokenizer.padding_side == "left" and pipe.tokenizer.pad_token == "eos"


def _application(tmp_path, **kwargs):
    from difflet.backends.cuda.ltx_2.application import CudaLtx2Application
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    app = CudaLtx2Application(
        model_path=str(tmp_path),
        parallel=DiffletParallelConfig(tp_degree=1),
        dtype="bf16",
        shape=dict(height=32, width=32, num_frames=9),
        device="cpu",
        text_seq_len=4,
        **kwargs,
    )
    app.pipe = _pipe()
    app.transformer = SimpleNamespace(
        dit_config=SimpleNamespace(
            video_seq_len=2,
            in_channels=2,
            latent_num_frames=2,
            latent_height=1,
            latent_width=1,
            audio_seq_len=2,
        )
    )
    return app


def _public_request():
    return {
        key: value
        for key, value in _request().items()
        if key not in {"generator", "max_sequence_length", "device"}
    }


def test_application_selects_explicit_bf16_profile_without_changing_output_contract(tmp_path):
    app = _application(tmp_path, generation_profile="ltx_official_bf16")
    output = app.generate_audio_video(**_public_request(), seed=12)
    assert output.frames is None and output.audio is None and output.sample_rate is None
    assert output.video_latents.flatten().tolist() == [-4.5625, -4.75, -5.78125, -3.71875]
    assert output.audio_latents.flatten().tolist() == [-9.0625, -10.25, -9.375, -8.25]


def test_stock_default_still_calls_existing_pipeline_with_no_profile_arguments(tmp_path):
    app = _application(tmp_path)

    class StockPipe(SimpleNamespace):
        def __call__(self, **kwargs):
            self.seen = kwargs
            return SimpleNamespace(frames="video", audio="audio")

    app.pipe = StockPipe(**vars(app.pipe))
    request = _public_request()
    request.pop("sigmas")
    output = app.generate_audio_video(**request)
    assert output.video_latents == "video" and output.audio_latents == "audio"
    assert "generation_profile" not in app.pipe.seen and "sigmas" not in app.pipe.seen
    assert app.pipe.seen["max_sequence_length"] == 4
    assert app.pipe.seen["latents"] is request["latents"]


def test_stock_profile_refuses_final_sigmas_instead_of_reshifting_them(tmp_path):
    app = _application(tmp_path)
    with pytest.raises(ValueError, match="ltx_official_bf16"):
        app.generate_audio_video(**_public_request())


@pytest.mark.parametrize("profile", ["mystery", True, ["ltx_official_bf16"]])
def test_application_rejects_unknown_generation_profile_before_io(tmp_path, profile):
    with pytest.raises(ValueError, match="generation_profile"):
        _application(tmp_path, generation_profile=profile)


def test_bf16_profile_refuses_float32_instead_of_silently_changing_semantics(tmp_path):
    from difflet.backends.cuda.ltx_2.application import CudaLtx2Application
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    with pytest.raises(ValueError, match="bfloat16"):
        CudaLtx2Application(
            model_path=str(tmp_path),
            parallel=DiffletParallelConfig(tp_degree=1),
            dtype="fp32",
            shape=dict(height=32, width=32, num_frames=9),
            device="cpu",
            generation_profile="ltx_official_bf16",
        )


def test_native_cfg_zero_requires_negative_conditioning(tmp_path):
    app = _application(tmp_path, generation_profile="ltx_official_bf16")
    request = {**_public_request(), "guidance_scale": 0.0, "negative_prompt_embeds": None}
    with pytest.raises(RuntimeError, match="prompt embeddings"):
        app.generate_audio_video(**request)


def test_invalid_schedule_is_rejected_before_conditioning_or_noise_draws():
    api, pipe = _api(), _pipe()
    with pytest.raises(ValueError):
        api.generate(pipe, **{**_request(), "sigmas": [0.8, 0.8, 0]})
    assert pipe.connector_batches == [] and pipe.transformer.calls == []


def test_degenerate_terminal_stretch_is_rejected_before_generation():
    with pytest.raises(ValueError, match="schedule"):
        _api().resolve_sigmas({**CONFIG, "shift_terminal": 1.0}, 2, None, device="cpu")


def test_profile_rejects_missing_prompt_before_using_loaded_text_encoder(tmp_path):
    app = _application(tmp_path, generation_profile="ltx_official_bf16")
    app.pipe.text_encoder = _TextEncoder()
    request = {**_public_request(), "prompt_embeds": None}
    with pytest.raises(ValueError, match="prompt"):
        app.generate_audio_video(**request)


@pytest.mark.parametrize("field", ["prompt_embeds", "negative_prompt_embeds"])
def test_profile_rejects_nonfinite_or_nonfloating_embeddings_before_connectors(tmp_path, field):
    app = _application(tmp_path, generation_profile="ltx_official_bf16")
    for malformed in (torch.full((1, 4, 2), float("nan")), torch.ones(1, 4, 2, dtype=torch.int32)):
        with pytest.raises(ValueError, match="embeddings"):
            app.generate_audio_video(**{**_public_request(), field: malformed})
    assert app.pipe.connector_batches == []


@pytest.mark.parametrize("guidance", [1.0, 3.5])
def test_profile_uses_real_adapter_with_token_timesteps_and_cache_context(monkeypatch, guidance):
    from difflet.backends.cuda.ltx_2.dit_config import Ltx2DiTConfig
    from difflet.backends.cuda.ltx_2.modeling import CudaLtx2TransformerModule
    from difflet.backends.cuda.ltx_2.transformer import Ltx2DiTAdapter

    config = Ltx2DiTConfig(
        in_channels=2,
        out_channels=2,
        num_attention_heads=2,
        attention_head_dim=8,
        cross_attention_dim=16,
        audio_in_channels=2,
        audio_out_channels=2,
        audio_num_attention_heads=2,
        audio_attention_head_dim=8,
        audio_cross_attention_dim=16,
        caption_channels=2,
        num_layers=1,
        height=32,
        width=32,
        num_frames=9,
        text_seq_len=4,
        audio_num_frames=2,
        torch_dtype=torch.bfloat16,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(84)
        module = CudaLtx2TransformerModule(config)
    module = module.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    pipe = _pipe()
    pipe.transformer = Ltx2DiTAdapter(
        SimpleNamespace(is_loaded=True, module=module, generation_profile="diffusers")
    )
    contexts, inputs = [], []
    original_context = module.transformer.cache_context

    @contextmanager
    def observed_context(name):
        contexts.append(("enter", name))
        with original_context(name):
            yield
        contexts.append(("exit", name))

    monkeypatch.setattr(module.transformer, "cache_context", observed_context)
    hook = module.transformer.register_forward_pre_hook(
        lambda model, args, kwargs: inputs.append(kwargs), with_kwargs=True
    )
    try:
        video, audio = _api().generate(pipe, **{**_request(), "guidance_scale": guidance})
    finally:
        hook.remove()
    assert contexts == [("enter", "guided"), ("exit", "guided")] * 2
    assert len(inputs) == 2
    batch = 1 if guidance == 1 else 2
    assert inputs[0]["timestep"].shape == (batch, 2)
    assert inputs[0]["audio_timestep"].shape == (batch, 2)
    assert inputs[0]["timestep"].dtype == inputs[0]["audio_timestep"].dtype == torch.float32
    assert video.shape == (1, 2, 2, 1, 1) and audio.shape == (1, 1, 2, 2)
    assert video.dtype == audio.dtype == torch.bfloat16
    assert torch.isfinite(video).all() and torch.isfinite(audio).all()


def test_cfg_off_keeps_native_video_layout_at_the_transformer_boundary():
    api, pipe = _api(), _pipe()
    api.generate(pipe, **{**_request(), "guidance_scale": 1.0})
    first = pipe.transformer.calls[0]["hidden_states"]
    assert first.shape == (1, 2, 2)
    assert first.stride() == (4, 1, 2)
    assert not first.is_contiguous()
