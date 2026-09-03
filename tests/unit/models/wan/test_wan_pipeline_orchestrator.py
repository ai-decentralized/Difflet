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


class FakeCfgParallelTransformer:
    """Emulates the gathered batch=2 output of a CFG-parallel transformer.

    The real model scatters [uncond, cond] to two ranks and gathers the result
    back to batch=2; this fake just returns a batch=2 tensor with distinct
    uncond/cond values so the CFG combine can be checked.
    """

    def __init__(self, *, uncond: float = 1.0, cond: float = 3.0):
        self.dtype = torch.float32
        self.config = SimpleNamespace(cfg_parallel_enabled=True)
        self.uncond = uncond
        self.cond = cond
        self.calls = []

    def __call__(self, hidden_states, timestep, encoder_hidden_states):
        self.calls.append(
            {
                "hidden_states": hidden_states.detach().clone(),
                "timestep": timestep.detach().clone(),
                "encoder_hidden_states": encoder_hidden_states.detach().clone(),
            }
        )
        out = torch.empty_like(hidden_states)
        out[0:1] = self.uncond
        out[1:2] = self.cond
        return out


def test_orchestrator_cfg_parallel_uses_single_batched_call(tmp_path):
    transformer = FakeCfgParallelTransformer(uncond=1.0, cond=3.0)
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
        num_inference_steps=1,
        guidance_scale=5.0,
        output_type="latent",
    )

    # One batched call per step (not two serial uncond/cond passes), batch=2.
    assert len(transformer.calls) == 1
    assert transformer.calls[0]["hidden_states"].shape[0] == 2
    assert transformer.calls[0]["timestep"].shape == (2,)
    assert transformer.calls[0]["encoder_hidden_states"].shape[0] == 2
    # noise_pred = uncond + scale*(cond-uncond) = 1 + 5*(3-1) = 11; scheduler
    # fallback (no scheduler) is latents - noise_pred/steps = 0 - 11/1 = -11.
    assert torch.allclose(output.frames, torch.full_like(latents, -11.0))


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


def test_encode_prompt_zeroes_padding_positions(tmp_path):
    # Reference Wan convention (diffusers _get_t5_prompt_embeds): embeddings at
    # padding positions are ZEROED. The DiT cross-attends unmasked over the full
    # sequence and was trained with zero-padded embeds — raw UMT5 pad-token
    # outputs at 500+ positions poison cross-attention and yield noise videos.
    text_encoder = FakeTextEncoder()  # returns all-ones embeddings
    pipeline = WanOrchestrator(
        model_path=str(tmp_path), text_encoder=text_encoder, dtype=torch.float32,
    )
    input_ids = torch.ones((1, 8), dtype=torch.int64)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.int32)
    out = pipeline.encode_prompt(input_ids=input_ids, attention_mask=attention_mask)
    assert torch.all(out[0, :3] == 1.0)   # valid tokens untouched
    assert torch.all(out[0, 3:] == 0.0)   # padding rows zeroed


class _FakeTokenizer:
    def __call__(self, prompts, **kw):
        n = len(prompts)
        ids = torch.ones((n, 8), dtype=torch.int64)
        mask = torch.zeros((n, 8), dtype=torch.int32)
        mask[:, 0] = 1  # empty prompt -> single valid (EOS) token
        return {"input_ids": ids, "attention_mask": mask}


def test_dense_cfg_encodes_empty_prompt_negative(tmp_path):
    # diffusers encodes negative_prompt="" through the text encoder (then
    # zero-pads); zeros_like is not a valid embedding and washes out CFG.
    transformer = FakeTransformer(bias=0.25)
    text_encoder = FakeTextEncoder()  # returns ones
    pipeline = WanOrchestrator(
        model_path=str(tmp_path), transformer=transformer,
        text_encoder=text_encoder, dtype=torch.float32,
    )
    pipeline._tokenizer = _FakeTokenizer()
    prompt_embeds = torch.full((1, 8, 8), 2.0)
    pipeline(prompt_embeds=prompt_embeds, latents=torch.zeros((1, 16, 3, 60, 104)),
             num_inference_steps=1, guidance_scale=4.0, output_type="latent")
    # dense CFG: cond call then uncond call
    assert len(transformer.calls) == 2
    uncond = transformer.calls[1]["encoder_hidden_states"]
    assert torch.all(uncond[0, 0] == 1.0)      # encoded empty-prompt token
    assert torch.all(uncond[0, 1:] == 0.0)     # zero-padded tail
    assert not torch.all(uncond == 0.0)        # NOT the zeros_like fallback


def test_loop_latents_held_in_float32(tmp_path):
    # diffusers holds loop latents in fp32 (UniPC order-2 corrector algebra
    # collapses in bf16); the model input is cast per step.
    transformer = FakeTransformer(bias=0.5)
    transformer.dtype = torch.bfloat16  # model contract: inputs cast per step
    pipeline = WanOrchestrator(
        model_path=str(tmp_path), transformer=transformer, dtype=torch.bfloat16,
    )
    out = pipeline(prompt_embeds=torch.ones((1, 8, 8), dtype=torch.bfloat16),
                   num_inference_steps=2, guidance_scale=1.0, output_type="latent",
                   generator=torch.Generator().manual_seed(0))
    assert out.latents.dtype == torch.float32
    assert transformer.calls[0]["hidden_states"].dtype == torch.bfloat16


# ------------------------------------------------------ probe-free TeaCache

def _run_cadence(pipeline, *, steps, guidance_scale=1.0):
    return pipeline(
        prompt_embeds=torch.ones((1, 5, 8), dtype=torch.float32),
        negative_prompt_embeds=torch.zeros((1, 5, 8), dtype=torch.float32),
        latents=torch.zeros((1, 16, 3, 4, 4), dtype=torch.float32),
        height=32, width=32, num_frames=9,
        num_inference_steps=steps, guidance_scale=guidance_scale,
        output_type="latent",
    )


def test_fixed_cadence_skips_dit_steps_and_prints_stats(tmp_path, capsys):
    # Regression for the CLI-dropped flag: the pipeline must build a probe-free
    # controller from teacache_cadence (no calibration file, no CPU shadow),
    # sync num_steps to the request, and report the skip stats.
    transformer = FakeTransformer(bias=0.25)
    pipeline = WanOrchestrator(
        model_path=str(tmp_path), transformer=transformer, dtype=torch.float32,
        boundary_ratio=None, teacache_cadence=2,
    )
    assert pipeline._teacache_controller is not None
    assert pipeline._teacache_shadows == {}

    # 14 steps, warmup/cooldown 5, cadence 2 -> skips at steps 6 and 8.
    _run_cadence(pipeline, steps=14, guidance_scale=4.0)
    stats = pipeline._teacache_last_stats
    assert stats["skipped_steps"] == 2
    assert stats["full_steps"] == 12
    # Dense CFG: a full step is 2 DiT calls (cond + uncond); a skip saves both.
    assert len(transformer.calls) == 24
    assert pipeline._teacache_controller.calibration.num_steps == 14
    out = capsys.readouterr().out
    assert "[teacache] probe-free controller enabled" in out
    assert "[teacache] stats: {'full_steps': 12, 'skipped_steps': 2" in out


def test_fixed_cadence_controller_resets_between_requests(tmp_path):
    transformer = FakeTransformer(bias=0.25)
    pipeline = WanOrchestrator(
        model_path=str(tmp_path), transformer=transformer, dtype=torch.float32,
        boundary_ratio=None, teacache_cadence=2,
    )
    _run_cadence(pipeline, steps=14)
    _run_cadence(pipeline, steps=12)  # window [5, 7): one skip (step 6)
    assert pipeline._teacache_last_stats == {
        **pipeline._teacache_last_stats, "full_steps": 11, "skipped_steps": 1,
    }
    assert pipeline._teacache_controller.calibration.num_steps == 12


def test_probe_free_modes_are_exclusive_with_calibration_path(tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        WanOrchestrator(
            model_path=str(tmp_path), transformer=FakeTransformer(), dtype=torch.float32,
            boundary_ratio=None, teacache_cadence=2,
            teacache_calibration_path=str(tmp_path / "cal.json"),
        )


def test_online_delta_mode_builds_probe_free_controller(tmp_path):
    pipeline = WanOrchestrator(
        model_path=str(tmp_path), transformer=FakeTransformer(), dtype=torch.float32,
        boundary_ratio=None, teacache_online_delta_alpha=0.6,
    )
    assert pipeline._teacache_controller.needs_signal() is False
    assert pipeline._teacache_controller.calibration.online_delta_alpha == pytest.approx(0.6)
