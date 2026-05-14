from types import SimpleNamespace

import pytest
import torch

from nova.models.ltx_2.application import LTX2DiTInputBundle
from nova.models.ltx_2.pipeline import (
    LTX2Orchestrator,
    LTX2PipelineOutput,
    ltx_2_scheduler_mu,
    make_ltx_2_audio_coords,
    make_ltx_2_video_coords,
    pack_ltx_2_audio_latents,
    pack_ltx_2_video_latents,
    rescale_ltx_2_noise_cfg,
    unpack_ltx_2_audio_latents,
    unpack_ltx_2_video_latents,
    _convert_velocity_to_x0,
    _convert_x0_to_velocity,
)


class FakeDualStreamTransformer:
    def __init__(self, *, video_value: float = 0.25, audio_value: float = 0.5):
        self.dtype = torch.float32
        self.video_value = video_value
        self.audio_value = audio_value
        self.calls = []

    def __call__(self, bundle: LTX2DiTInputBundle):
        self.calls.append(bundle)
        return (
            torch.ones_like(bundle.hidden_states) * self.video_value,
            torch.ones_like(bundle.audio_hidden_states) * self.audio_value,
        )


class FakeCfgTransformer:
    dtype = torch.float32

    def __init__(self):
        self.calls = []

    def __call__(self, bundle: LTX2DiTInputBundle):
        self.calls.append(bundle)
        batch = bundle.hidden_states.shape[0]
        assert batch % 2 == 0
        half = batch // 2
        video = torch.empty_like(bundle.hidden_states)
        audio = torch.empty_like(bundle.audio_hidden_states)
        video[:half].fill_(0.25)
        video[half:].fill_(1.0)
        audio[:half].fill_(0.5)
        audio[half:].fill_(2.0)
        return video, audio


class FakeRescaleCfgTransformer:
    dtype = torch.float32

    def __init__(self):
        self.calls = []

    def __call__(self, bundle: LTX2DiTInputBundle):
        self.calls.append(bundle)
        assert bundle.hidden_states.shape[0] == 2
        video_pattern = torch.linspace(
            0.0,
            1.0,
            steps=bundle.hidden_states[0].numel(),
            dtype=bundle.hidden_states.dtype,
            device=bundle.hidden_states.device,
        ).reshape_as(bundle.hidden_states[0])
        audio_pattern = torch.linspace(
            -1.0,
            1.0,
            steps=bundle.audio_hidden_states[0].numel(),
            dtype=bundle.audio_hidden_states.dtype,
            device=bundle.audio_hidden_states.device,
        ).reshape_as(bundle.audio_hidden_states[0])
        video = torch.stack([torch.zeros_like(video_pattern), video_pattern], dim=0)
        audio = torch.stack([torch.zeros_like(audio_pattern), audio_pattern], dim=0)
        return video, audio


class FakeExtraGuidanceTransformer:
    dtype = torch.float32
    supports_ltx_2_extra_kwargs = True

    def __init__(self):
        self.calls = []

    def __call__(self, bundle: LTX2DiTInputBundle, **kwargs):
        self.calls.append((bundle, kwargs))
        if kwargs.get("spatio_temporal_guidance_blocks") is not None:
            return (
                torch.ones_like(bundle.hidden_states) * 0.5,
                torch.ones_like(bundle.audio_hidden_states) * 0.5,
            )
        if kwargs.get("isolate_modalities"):
            return (
                torch.ones_like(bundle.hidden_states) * 0.75,
                torch.ones_like(bundle.audio_hidden_states) * 0.75,
            )
        batch = bundle.hidden_states.shape[0]
        assert batch % 2 == 0
        half = batch // 2
        video = torch.empty_like(bundle.hidden_states)
        audio = torch.empty_like(bundle.audio_hidden_states)
        video[:half].fill_(0.25)
        video[half:].fill_(1.0)
        audio[:half].fill_(0.25)
        audio[half:].fill_(1.0)
        return video, audio


class FakeVideoVAE:
    def __init__(self):
        self.dtype = torch.float32
        self.config = SimpleNamespace()
        self.calls = []

    def decode(self, latents, timestep=None, return_dict=False):
        assert return_dict is False
        self.calls.append((latents, timestep))
        return (latents[:, :3],)


class FakeAudioVAE:
    def __init__(self):
        self.dtype = torch.float32
        self.config = SimpleNamespace()
        self.calls = []

    def decode(self, latents, return_dict):
        assert return_dict is False
        self.calls.append(latents)
        return (latents.mean(dim=1),)


class FakeVocoder:
    def __call__(self, mel):
        return mel.flatten(1)


class FakeVideoProcessor:
    def __init__(self):
        self.calls = []

    def postprocess_video(self, video, output_type):
        self.calls.append((video, output_type))
        return video + 1


class FakeHostPipeline:
    def __init__(self):
        self._execution_device = torch.device("cpu")
        self.dtype = torch.float32
        self.tokenizer = SimpleNamespace(padding_side="left")
        self.encode_calls = []
        self.connector_calls = []

    def encode_prompt(
        self,
        *,
        prompt,
        negative_prompt,
        do_classifier_free_guidance,
        num_videos_per_prompt,
        prompt_embeds,
        negative_prompt_embeds,
        prompt_attention_mask,
        negative_prompt_attention_mask,
        max_sequence_length,
        device,
        dtype,
    ):
        self.encode_calls.append(
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "do_classifier_free_guidance": do_classifier_free_guidance,
                "num_videos_per_prompt": num_videos_per_prompt,
                "max_sequence_length": max_sequence_length,
                "device": device,
                "dtype": dtype,
            }
        )
        del prompt_embeds, negative_prompt_embeds, prompt_attention_mask
        del negative_prompt_attention_mask
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        batch_size *= int(num_videos_per_prompt)
        embeds = torch.ones((batch_size, max_sequence_length, 12), dtype=dtype)
        mask = torch.ones((batch_size, max_sequence_length), dtype=torch.long)
        if not do_classifier_free_guidance:
            return embeds, mask, None, None
        negative_embeds = torch.zeros((batch_size, max_sequence_length, 12), dtype=dtype)
        negative_mask = torch.ones((batch_size, max_sequence_length), dtype=torch.long)
        return embeds, mask, negative_embeds, negative_mask

    def connectors(self, prompt_embeds, prompt_attention_mask, *, padding_side):
        self.connector_calls.append(
            {
                "prompt_embeds": prompt_embeds,
                "prompt_attention_mask": prompt_attention_mask,
                "padding_side": padding_side,
            }
        )
        video = torch.ones((prompt_embeds.shape[0], prompt_embeds.shape[1], 32))
        audio = torch.full((prompt_embeds.shape[0], prompt_embeds.shape[1], 32), 2.0)
        return video, audio, prompt_attention_mask


class FakeScheduler:
    config = {}
    order = 1

    def __init__(self):
        self.calls = []
        self.timesteps = torch.empty(0)

    def set_timesteps(self, num_inference_steps=None, *, device=None, sigmas=None, mu=None):
        self.calls.append(
            {
                "num_inference_steps": num_inference_steps,
                "device": device,
                "sigmas": sigmas,
                "mu": mu,
            }
        )
        if sigmas is not None:
            self.timesteps = torch.as_tensor(sigmas, dtype=torch.float32)
        else:
            self.timesteps = torch.linspace(
                1.0,
                1.0 / float(num_inference_steps),
                steps=int(num_inference_steps),
                dtype=torch.float32,
            )

    def step(self, noise_pred, timestep, latents, return_dict=False):
        assert return_dict is False
        return (latents - noise_pred,)


class FakeStatefulScheduler(FakeScheduler):
    def __init__(self):
        super().__init__()
        self.timesteps = torch.tensor([1.0, 0.5], dtype=torch.float32)
        self.sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)
        self.step_index = 0

    def step(self, noise_pred, timestep, latents, return_dict=False):
        assert return_dict is False
        del timestep
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        self.step_index += 1
        return (latents + (sigma_next - sigma) * noise_pred,)


def _bundle(
    latents: torch.Tensor | None = None,
    audio_latents: torch.Tensor | None = None,
) -> LTX2DiTInputBundle:
    latents = torch.zeros((1, 6, 128), dtype=torch.float32) if latents is None else latents
    audio_latents = (
        torch.zeros((1, 4, 128), dtype=torch.float32) if audio_latents is None else audio_latents
    )
    return LTX2DiTInputBundle(
        hidden_states=latents,
        audio_hidden_states=audio_latents,
        encoder_hidden_states=torch.ones((1, 5, 32), dtype=torch.float32),
        audio_encoder_hidden_states=torch.ones((1, 5, 32), dtype=torch.float32),
        timestep=torch.zeros([1], dtype=torch.float32),
        sigma=torch.zeros([1], dtype=torch.float32),
        encoder_attention_mask=torch.ones((1, 5), dtype=torch.bool),
        audio_encoder_attention_mask=torch.ones((1, 5), dtype=torch.bool),
        video_coords=torch.zeros((1, 3, latents.shape[1], 2), dtype=torch.float32),
        audio_coords=torch.zeros((1, 1, audio_latents.shape[1], 2), dtype=torch.float32),
    )


def test_pack_and_unpack_ltx_2_video_latents_round_trip():
    latents = torch.arange(1 * 2 * 3 * 4 * 5, dtype=torch.float32).reshape(1, 2, 3, 4, 5)

    packed = pack_ltx_2_video_latents(latents)
    unpacked = unpack_ltx_2_video_latents(packed, num_frames=3, height=4, width=5)

    assert packed.shape == (1, 3 * 4 * 5, 2)
    assert torch.equal(unpacked, latents)


def test_pack_and_unpack_ltx_2_audio_latents_round_trip():
    latents = torch.arange(1 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4)

    packed = pack_ltx_2_audio_latents(latents)
    unpacked = unpack_ltx_2_audio_latents(packed, latent_length=3, num_mel_bins=4)

    assert packed.shape == (1, 3, 8)
    assert torch.equal(unpacked, latents)


def test_ltx_2_video_coords_match_upstream_boundary_formula():
    coords = make_ltx_2_video_coords(
        batch_size=1,
        num_frames=2,
        height=2,
        width=2,
        device="cpu",
        fps=24.0,
    )

    assert coords.shape == (1, 3, 8, 2)
    assert torch.allclose(coords[0, 0, 0], torch.tensor([0.0, 1.0 / 24.0]))
    assert torch.allclose(coords[0, 0, 4], torch.tensor([1.0 / 24.0, 9.0 / 24.0]))
    assert torch.equal(coords[0, 1, 0], torch.tensor([0.0, 32.0]))
    assert torch.equal(coords[0, 2, 1], torch.tensor([32.0, 64.0]))


def test_ltx_2_audio_coords_match_upstream_boundary_formula():
    coords = make_ltx_2_audio_coords(batch_size=1, audio_num_frames=3, device="cpu")

    assert coords.shape == (1, 1, 3, 2)
    expected = torch.tensor(
        [
            [0.0, 0.01],
            [0.01, 0.05],
            [0.05, 0.09],
        ]
    )
    assert torch.allclose(coords[0, 0], expected)


def test_ltx_2_scheduler_mu_matches_upstream_defaults():
    assert ltx_2_scheduler_mu({}) == pytest.approx(2.05)
    assert ltx_2_scheduler_mu(
        {
            "base_image_seq_len": 256,
            "max_image_seq_len": 4096,
            "base_shift": 0.5,
            "max_shift": 1.15,
        }
    ) == pytest.approx(1.15)


def test_ltx_2_orchestrator_timesteps_pass_ltx2_scheduler_shift(tmp_path):
    scheduler = FakeScheduler()
    pipeline = LTX2Orchestrator(
        model_path=str(tmp_path),
        dtype=torch.float32,
        scheduler=scheduler,
    )

    timesteps = pipeline._timesteps(4, device=torch.device("cpu"))

    assert torch.allclose(timesteps, torch.tensor([1.0, 0.75, 0.5, 0.25]))
    assert len(scheduler.calls) == 1
    assert scheduler.calls[0]["device"] == "cpu"
    assert scheduler.calls[0]["mu"] == pytest.approx(2.05)
    assert scheduler.calls[0]["sigmas"].tolist() == pytest.approx([1.0, 0.75, 0.5, 0.25])


def test_ltx_2_orchestrator_rejects_one_step_diffusers_scheduler(tmp_path):
    scheduler = FakeScheduler()
    pipeline = LTX2Orchestrator(
        model_path=str(tmp_path),
        dtype=torch.float32,
        scheduler=scheduler,
    )

    with pytest.raises(ValueError, match="num_inference_steps >= 2"):
        pipeline._timesteps(1, device=torch.device("cpu"))

    assert scheduler.calls == []


def test_ltx_2_orchestrator_prepares_dual_stream_latents(tmp_path):
    pipeline = LTX2Orchestrator(
        model_path=str(tmp_path),
        dtype=torch.float32,
        height=64,
        width=96,
        num_frames=17,
        audio_num_frames=4,
    )

    video = pipeline.prepare_latents(batch_size=2)
    audio = pipeline.prepare_audio_latents(batch_size=2)

    assert video.shape == (2, 3 * 2 * 3, 128)
    assert audio.shape == (2, 4, 128)
    assert video.dtype == torch.float32
    assert audio.dtype == torch.float32


def test_ltx_2_orchestrator_runs_bundle_denoise_with_fallback_scheduler(tmp_path):
    transformer = FakeDualStreamTransformer(video_value=0.5, audio_value=0.25)
    with pytest.warns(RuntimeWarning, match="scheduler_config.json"):
        pipeline = LTX2Orchestrator(
            model_path=str(tmp_path),
            transformer=transformer,
            dtype=torch.float32,
            height=64,
            width=96,
            num_frames=17,
            audio_num_frames=4,
        )

    output = pipeline(
        bundle=_bundle(),
        timesteps=torch.tensor([1.0, 0.5]),
        return_trajectory=True,
    )

    assert isinstance(output, LTX2PipelineOutput)
    assert torch.allclose(output.latents, torch.full((1, 6, 128), -0.5))
    assert torch.allclose(output.audio_latents, torch.full((1, 4, 128), -0.25))
    assert len(transformer.calls) == 2
    assert transformer.calls[0].timestep.shape == (1,)
    assert transformer.calls[0].audio_encoder_attention_mask.dtype == torch.bool
    assert output.trajectory is not None
    assert len(output.trajectory) == 3


def test_ltx_2_orchestrator_uses_independent_audio_scheduler_state(tmp_path):
    transformer = FakeDualStreamTransformer(video_value=1.0, audio_value=2.0)
    scheduler = FakeStatefulScheduler()
    pipeline = LTX2Orchestrator(
        model_path=str(tmp_path),
        transformer=transformer,
        dtype=torch.float32,
        scheduler=scheduler,
    )

    output = pipeline(bundle=_bundle(), timesteps=scheduler.timesteps)

    assert scheduler.step_index == 2
    assert torch.allclose(output.latents, torch.full((1, 6, 128), -1.0))
    assert torch.allclose(output.audio_latents, torch.full((1, 4, 128), -2.0))


def test_ltx_2_orchestrator_builds_bundle_from_named_tensors(tmp_path):
    transformer = FakeDualStreamTransformer(video_value=1.0, audio_value=1.0)
    pipeline = LTX2Orchestrator(
        model_path=str(tmp_path),
        transformer=transformer,
        dtype=torch.float32,
        height=64,
        width=96,
        num_frames=17,
        text_seq_len=5,
        audio_num_frames=4,
        scheduler=None,
    )

    pipeline(
        latents=torch.zeros((1, 18, 128), dtype=torch.float32),
        audio_latents=torch.zeros((1, 4, 128), dtype=torch.float32),
        timesteps=torch.tensor([1.0]),
        encoder_hidden_states=torch.ones((1, 5, 32), dtype=torch.float32),
    )

    assert len(transformer.calls) == 1
    assert transformer.calls[0].audio_encoder_hidden_states.shape == (1, 5, 32)
    assert transformer.calls[0].encoder_attention_mask.shape == (1, 5)


def test_ltx_2_orchestrator_prepares_prompt_conditioning_with_host_pipeline(tmp_path):
    transformer = FakeDualStreamTransformer(video_value=1.0, audio_value=1.0)
    host_pipeline = FakeHostPipeline()
    pipeline = LTX2Orchestrator(
        model_path=str(tmp_path),
        transformer=transformer,
        host_pipeline=host_pipeline,
        dtype=torch.float32,
        height=64,
        width=96,
        num_frames=17,
        text_seq_len=5,
        audio_num_frames=4,
        scheduler=None,
    )

    output = pipeline(
        prompt=["a", "b"],
        num_videos_per_prompt=1,
        timesteps=torch.tensor([1.0]),
        guidance_scale=1.0,
    )

    assert output.latents.shape == (2, 18, 128)
    assert output.audio_latents.shape == (2, 4, 128)
    assert len(transformer.calls) == 1
    assert torch.equal(transformer.calls[0].encoder_attention_mask, torch.ones((2, 5), dtype=torch.bool))
    assert torch.all(transformer.calls[0].audio_encoder_hidden_states == 2.0)
    assert transformer.calls[0].video_coords.shape == (2, 3, 18, 2)
    assert transformer.calls[0].audio_coords.shape == (2, 1, 4, 2)
    assert host_pipeline.encode_calls[0]["do_classifier_free_guidance"] is False
    assert host_pipeline.connector_calls[0]["padding_side"] == "left"


def test_ltx_2_orchestrator_runs_host_prompt_cfg(tmp_path):
    transformer = FakeCfgTransformer()
    pipeline = LTX2Orchestrator(
        model_path=str(tmp_path),
        transformer=transformer,
        host_pipeline=FakeHostPipeline(),
        dtype=torch.float32,
        height=64,
        width=96,
        num_frames=17,
        text_seq_len=5,
        audio_num_frames=4,
        scheduler=None,
    )

    output = pipeline(
        prompt="a",
        latents=torch.zeros((1, 18, 128), dtype=torch.float32),
        audio_latents=torch.zeros((1, 4, 128), dtype=torch.float32),
        guidance_scale=4.0,
        audio_guidance_scale=3.0,
        timesteps=torch.tensor([1.0]),
    )

    assert torch.allclose(output.latents, torch.full((1, 18, 128), -3.25))
    assert torch.allclose(output.audio_latents, torch.full((1, 4, 128), -5.0))
    assert transformer.calls[0].hidden_states.shape == (2, 18, 128)
    assert transformer.calls[0].audio_hidden_states.shape == (2, 4, 128)
    assert transformer.calls[0].encoder_hidden_states.shape == (2, 5, 32)
    assert transformer.calls[0].video_coords.shape == (2, 3, 18, 2)
    assert transformer.calls[0].audio_coords.shape == (2, 1, 4, 2)


def test_ltx_2_orchestrator_applies_video_and_audio_guidance_rescale(tmp_path):
    transformer = FakeRescaleCfgTransformer()
    pipeline = LTX2Orchestrator(
        model_path=str(tmp_path),
        transformer=transformer,
        host_pipeline=FakeHostPipeline(),
        dtype=torch.float32,
        height=64,
        width=96,
        num_frames=17,
        text_seq_len=5,
        audio_num_frames=4,
        scheduler=None,
    )

    video_pattern = torch.linspace(0.0, 1.0, steps=18 * 128).reshape(1, 18, 128)
    audio_pattern = torch.linspace(-1.0, 1.0, steps=4 * 128).reshape(1, 4, 128)
    expected_video_pred = rescale_ltx_2_noise_cfg(
        video_pattern * 4.0,
        video_pattern,
        guidance_rescale=0.25,
    )
    expected_audio_pred = rescale_ltx_2_noise_cfg(
        audio_pattern * 3.0,
        audio_pattern,
        guidance_rescale=0.5,
    )

    output = pipeline(
        prompt="a",
        latents=torch.zeros((1, 18, 128), dtype=torch.float32),
        audio_latents=torch.zeros((1, 4, 128), dtype=torch.float32),
        guidance_scale=4.0,
        audio_guidance_scale=3.0,
        guidance_rescale=0.25,
        audio_guidance_rescale=0.5,
        timesteps=torch.tensor([1.0]),
    )

    assert torch.allclose(output.latents, -expected_video_pred)
    assert torch.allclose(output.audio_latents, -expected_audio_pred)


def test_ltx_2_orchestrator_runs_stg_and_modality_guidance(tmp_path):
    transformer = FakeExtraGuidanceTransformer()
    pipeline = LTX2Orchestrator(
        model_path=str(tmp_path),
        transformer=transformer,
        host_pipeline=FakeHostPipeline(),
        dtype=torch.float32,
        height=64,
        width=96,
        num_frames=17,
        text_seq_len=5,
        audio_num_frames=4,
        scheduler=None,
    )

    output = pipeline(
        prompt="a",
        latents=torch.zeros((1, 18, 128), dtype=torch.float32),
        audio_latents=torch.zeros((1, 4, 128), dtype=torch.float32),
        guidance_scale=2.0,
        audio_guidance_scale=2.0,
        stg_scale=0.5,
        audio_stg_scale=0.5,
        modality_scale=1.5,
        audio_modality_scale=1.5,
        spatio_temporal_guidance_blocks=[0],
        use_cross_timestep=True,
        attention_kwargs={"scale": 1.0},
        timesteps=torch.tensor([1.0]),
    )

    assert torch.allclose(output.latents, torch.full((1, 18, 128), -2.125))
    assert torch.allclose(output.audio_latents, torch.full((1, 4, 128), -2.125))
    assert len(transformer.calls) == 3
    assert transformer.calls[1][0].hidden_states.shape == (1, 18, 128)
    assert transformer.calls[1][1]["spatio_temporal_guidance_blocks"] == [0]
    assert transformer.calls[1][1]["use_cross_timestep"] is True
    assert transformer.calls[1][1]["attention_kwargs"] == {"scale": 1.0}
    assert transformer.calls[2][1]["isolate_modalities"] is True


def test_ltx_2_orchestrator_rejects_stg_without_blocks(tmp_path):
    pipeline = LTX2Orchestrator(
        model_path=str(tmp_path),
        transformer=FakeExtraGuidanceTransformer(),
        host_pipeline=FakeHostPipeline(),
        dtype=torch.float32,
        scheduler=None,
    )

    with pytest.raises(ValueError, match="spatio_temporal_guidance_blocks"):
        pipeline(prompt="a", stg_scale=1.0)


def test_ltx_2_orchestrator_rejects_extra_guidance_for_bundle_only_transformer(tmp_path):
    pipeline = LTX2Orchestrator(
        model_path=str(tmp_path),
        transformer=FakeDualStreamTransformer(),
        host_pipeline=FakeHostPipeline(),
        dtype=torch.float32,
        height=64,
        width=96,
        num_frames=17,
        text_seq_len=5,
        audio_num_frames=4,
        scheduler=None,
    )

    with pytest.raises(NotImplementedError, match="extra LTX-2 guidance kwargs"):
        pipeline(
            prompt="a",
            latents=torch.zeros((1, 18, 128), dtype=torch.float32),
            audio_latents=torch.zeros((1, 4, 128), dtype=torch.float32),
            stg_scale=1.0,
            spatio_temporal_guidance_blocks=[0],
            timesteps=torch.tensor([1.0]),
        )


def test_ltx_2_velocity_x0_conversion_round_trip():
    sample = torch.tensor([[[1.0, 2.0, 3.0]]])
    velocity = torch.tensor([[[0.5, -1.0, 2.0]]])
    sigma = torch.tensor(0.25)

    x0 = _convert_velocity_to_x0(sample, velocity, sigma)
    recovered = _convert_x0_to_velocity(sample, x0, sigma)

    assert torch.allclose(x0, torch.tensor([[[0.875, 2.25, 2.5]]]))
    assert torch.allclose(recovered, velocity)


def test_ltx_2_orchestrator_requires_encoder_hidden_states(tmp_path):
    pipeline = LTX2Orchestrator(model_path=str(tmp_path), dtype=torch.float32)

    with pytest.raises(ValueError, match="encoder_hidden_states"):
        pipeline()


def test_ltx_2_orchestrator_decodes_pt_output_with_required_components(tmp_path):
    vae = FakeVideoVAE()
    audio_vae = FakeAudioVAE()
    processor = FakeVideoProcessor()
    pipeline = LTX2Orchestrator(
        model_path=str(tmp_path),
        vae=vae,
        audio_vae=audio_vae,
        vocoder=FakeVocoder(),
        video_processor=processor,
        dtype=torch.float32,
        height=64,
        width=96,
        num_frames=17,
        audio_num_frames=4,
    )

    output = pipeline(
        bundle=_bundle(
            latents=torch.ones((1, 18, 128), dtype=torch.float32),
            audio_latents=torch.ones((1, 4, 128), dtype=torch.float32),
        ),
        output_type="pt",
        return_dict=False,
    )

    assert isinstance(output, tuple)
    assert output[0].shape == (1, 3, 3, 2, 3)
    assert output[1].shape == (1, 64)
    assert len(vae.calls) == 1
    assert vae.calls[0][1] is None
    assert processor.calls[0][1] == "pt"


def test_ltx_2_orchestrator_denormalizes_latents_before_decode(tmp_path):
    vae = FakeVideoVAE()
    vae.latents_mean = torch.full((128,), 2.0)
    vae.latents_std = torch.full((128,), 4.0)
    vae.config = SimpleNamespace(scaling_factor=2.0, timestep_conditioning=True)
    audio_vae = FakeAudioVAE()
    audio_vae.latents_mean = torch.full((128,), 3.0)
    audio_vae.latents_std = torch.full((128,), 5.0)
    pipeline = LTX2Orchestrator(
        model_path=str(tmp_path),
        vae=vae,
        audio_vae=audio_vae,
        vocoder=FakeVocoder(),
        dtype=torch.float32,
        height=64,
        width=96,
        num_frames=17,
        audio_num_frames=4,
    )

    pipeline(
        bundle=_bundle(
            latents=torch.ones((1, 18, 128), dtype=torch.float32),
            audio_latents=torch.ones((1, 4, 128), dtype=torch.float32),
        ),
        output_type="pt",
        decode_timestep=0.25,
        decode_noise_scale=0.0,
    )

    assert torch.allclose(vae.calls[0][0], torch.full((1, 128, 3, 2, 3), 4.0))
    assert torch.allclose(vae.calls[0][1], torch.tensor([0.25]))
    assert torch.allclose(audio_vae.calls[0], torch.full((1, 8, 4, 16), 8.0))


def test_ltx_2_orchestrator_applies_decode_noise_before_video_denormalize(tmp_path):
    vae = FakeVideoVAE()
    vae.latents_mean = torch.full((128,), 2.0)
    vae.latents_std = torch.full((128,), 4.0)
    vae.config = SimpleNamespace(scaling_factor=2.0, timestep_conditioning=True)
    audio_vae = FakeAudioVAE()
    pipeline = LTX2Orchestrator(
        model_path=str(tmp_path),
        vae=vae,
        audio_vae=audio_vae,
        vocoder=FakeVocoder(),
        dtype=torch.float32,
        height=64,
        width=96,
        num_frames=17,
        audio_num_frames=4,
    )
    generator = torch.Generator(device="cpu").manual_seed(123)

    pipeline(
        bundle=_bundle(
            latents=torch.ones((1, 18, 128), dtype=torch.float32),
            audio_latents=torch.ones((1, 4, 128), dtype=torch.float32),
        ),
        output_type="pt",
        decode_timestep=0.5,
        decode_noise_scale=1.0,
        generator=generator,
    )

    expected_generator = torch.Generator(device="cpu").manual_seed(123)
    expected_noise = torch.randn(
        (1, 128, 3, 2, 3),
        generator=expected_generator,
        dtype=torch.float32,
    )
    expected = expected_noise * 4.0 / 2.0 + 2.0
    assert torch.allclose(vae.calls[0][0], expected)
    assert torch.allclose(vae.calls[0][1], torch.tensor([0.5]))
