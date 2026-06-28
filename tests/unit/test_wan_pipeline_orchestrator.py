from types import SimpleNamespace

import pytest
import torch

from difflet.models.wan.pipeline import WanOrchestrator, WanPipelineOutput


class FakeTransformer:
    def __init__(self, *, bias: float = 0.0):
        self.dtype = torch.float32
        self.bias = bias
        self.calls = []

    def __call__(self, hidden_states, timestep, encoder_hidden_states):
        self.calls.append(
            {
                "hidden_states": hidden_states.detach().clone(),
                "timestep": timestep.detach().clone(),
                "encoder_hidden_states": encoder_hidden_states.detach().clone(),
            }
        )
        return torch.ones_like(hidden_states) * self.bias


class FakeTextEncoder:
    def __init__(self):
        self.dtype = torch.float32
        self.calls = []

    def __call__(self, input_ids, attention_mask):
        self.calls.append((input_ids.detach().clone(), attention_mask.detach().clone()))
        batch, seq_len = input_ids.shape
        return torch.ones((batch, seq_len, 8), dtype=torch.float32)


class FakeVAE:
    def __init__(self):
        self.dtype = torch.float32
        self.config = SimpleNamespace(
            latents_mean=[0.5] * 16,
            latents_std=[2.0] * 16,
        )
        self.inputs = []

    def __call__(self, latents):
        self.inputs.append(latents.detach().clone())
        return latents[:, :3]


def test_prepare_latents_uses_wan_temporal_and_spatial_scale(tmp_path):
    pipeline = WanOrchestrator(model_path=str(tmp_path), dtype=torch.float32)

    latents = pipeline.prepare_latents(batch_size=2, height=480, width=832, num_frames=9)

    assert latents.shape == (2, 16, 3, 60, 104)
    assert latents.dtype == torch.float32


def test_orchestrator_runs_fallback_denoise_with_prompt_embeds(tmp_path):
    transformer = FakeTransformer(bias=0.25)
    pipeline = WanOrchestrator(
        model_path=str(tmp_path),
        transformer=transformer,
        dtype=torch.float32,
        boundary_ratio=None,
    )
    latents = torch.zeros((1, 16, 3, 4, 4), dtype=torch.float32)
    prompt_embeds = torch.ones((1, 5, 8), dtype=torch.float32)

    output = pipeline(
        prompt_embeds=prompt_embeds,
        latents=latents,
        height=32,
        width=32,
        num_frames=9,
        num_inference_steps=2,
        output_type="latent",
    )

    assert isinstance(output, WanPipelineOutput)
    assert output.frames.shape == latents.shape
    assert torch.allclose(output.frames, torch.full_like(latents, -0.25))
    assert len(transformer.calls) == 2
    assert transformer.calls[0]["timestep"].shape == (1,)


def test_orchestrator_routes_late_steps_to_transformer_2(tmp_path):
    high_noise = FakeTransformer(bias=0.0)
    low_noise = FakeTransformer(bias=0.0)
    pipeline = WanOrchestrator(
        model_path=str(tmp_path),
        transformer=high_noise,
        transformer_2=low_noise,
        dtype=torch.float32,
        boundary_ratio=0.875,
    )

    pipeline(
        prompt_embeds=torch.ones((1, 5, 8), dtype=torch.float32),
        latents=torch.zeros((1, 16, 3, 4, 4), dtype=torch.float32),
        height=32,
        width=32,
        num_frames=9,
        num_inference_steps=2,
    )

    assert len(high_noise.calls) == 1
    assert len(low_noise.calls) == 1
    assert high_noise.calls[0]["timestep"].item() == pytest.approx(999.0)
    assert low_noise.calls[0]["timestep"].item() == pytest.approx(0.0)


def test_orchestrator_supports_text_encoder_inputs(tmp_path):
    text_encoder = FakeTextEncoder()
    transformer = FakeTransformer(bias=0.0)
    pipeline = WanOrchestrator(
        model_path=str(tmp_path),
        text_encoder=text_encoder,
        transformer=transformer,
        dtype=torch.float32,
    )
    input_ids = torch.arange(6, dtype=torch.int64).view(1, 6)

    pipeline(
        input_ids=input_ids,
        latents=torch.zeros((1, 16, 3, 4, 4), dtype=torch.float32),
        height=32,
        width=32,
        num_frames=9,
    )

    assert len(text_encoder.calls) == 1
    assert torch.equal(text_encoder.calls[0][1], torch.ones_like(input_ids, dtype=torch.int32))
    assert transformer.calls[0]["encoder_hidden_states"].shape == (1, 6, 8)


def test_orchestrator_decodes_pt_output_with_vae_latent_stats(tmp_path):
    vae = FakeVAE()
    pipeline = WanOrchestrator(
        model_path=str(tmp_path),
        vae_decoder=vae,
        dtype=torch.float32,
    )
    latents = torch.ones((1, 16, 3, 4, 4), dtype=torch.float32)

    output = pipeline(
        latents=latents,
        height=32,
        width=32,
        num_frames=9,
        output_type="pt",
        return_dict=False,
    )

    assert isinstance(output, tuple)
    assert output[0].shape == (1, 3, 3, 4, 4)
    assert torch.allclose(vae.inputs[0], torch.full_like(latents, 2.5))


def test_prompt_string_without_text_encoder_raises(tmp_path):
    pipeline = WanOrchestrator(model_path=str(tmp_path), dtype=torch.float32)

    with pytest.raises(ValueError, match="no Wan text_encoder"):
        pipeline(prompt="a test prompt")


def test_prompt_string_routes_through_tokenizer_when_text_encoder_active(tmp_path):
    text_encoder = FakeTextEncoder()

    class _StubTokenizer:
        def __call__(self, prompts, padding, truncation, max_length, return_tensors):
            assert padding == "max_length"
            assert truncation is True
            assert return_tensors == "pt"
            ids = torch.zeros((len(prompts), max_length), dtype=torch.int64)
            mask = torch.ones((len(prompts), max_length), dtype=torch.int32)
            return {"input_ids": ids, "attention_mask": mask}

    pipeline = WanOrchestrator(
        model_path=str(tmp_path),
        text_encoder=text_encoder,
        dtype=torch.float32,
        max_text_length=4,
    )
    pipeline._tokenizer = _StubTokenizer()

    out = pipeline.encode_prompt(prompt="a test prompt")
    assert torch.is_tensor(out)
    assert len(text_encoder.calls) == 1
    received_input_ids, _ = text_encoder.calls[0]
    assert received_input_ids.shape == (1, 4)


def test_negative_prompt_string_routes_through_tokenizer(tmp_path):
    text_encoder = FakeTextEncoder()
    transformer = FakeTransformer(bias=0.5)

    class _StubTokenizer:
        def __call__(self, prompts, padding, truncation, max_length, return_tensors):
            ids = torch.zeros((len(prompts), max_length), dtype=torch.int64)
            mask = torch.ones((len(prompts), max_length), dtype=torch.int32)
            return {"input_ids": ids, "attention_mask": mask}

    pipeline = WanOrchestrator(
        model_path=str(tmp_path),
        text_encoder=text_encoder,
        transformer=transformer,
        dtype=torch.float32,
        max_text_length=4,
        height=8,
        width=8,
        num_frames=5,
    )
    pipeline._tokenizer = _StubTokenizer()

    latents = torch.zeros((1, 16, 2, 1, 1), dtype=torch.float32)
    pipeline(
        prompt="a positive",
        negative_prompt="a negative",
        latents=latents,
        num_inference_steps=1,
        guidance_scale=2.0,
        output_type="latent",
    )
    # encode_prompt invoked once for positive and once for negative.
    assert len(text_encoder.calls) == 2
    # Transformer was called twice per step (cond + uncond) because guidance_scale > 1.
    assert len(transformer.calls) == 2


def test_application_uses_video_to_latent_frame_formula_for_dit_compile_shape():
    from difflet.models.wan.application import _latent_num_frames

    assert _latent_num_frames(1) == 1
    assert _latent_num_frames(5) == 2
    assert _latent_num_frames(9) == 3
    assert _latent_num_frames(49) == 13
    assert _latent_num_frames(81) == 21


def test_application_clamps_single_core_component_load_range():
    from difflet.models.wan.application import NeuronWanApplication

    component = SimpleNamespace(
        config=SimpleNamespace(neuron_config=SimpleNamespace(world_size=1))
    )

    assert NeuronWanApplication._component_load_rank_range(
        component,
        start_rank_id=0,
        local_ranks_size=4,
    ) == (0, 1)
