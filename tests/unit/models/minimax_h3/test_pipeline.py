from __future__ import annotations

import torch

from difflet.models.minimax_h3.pipeline import (
    denoise_minimax_h3_t2va,
    patchify_video_latents,
    unpatchify_video_latents,
)
from difflet.models.minimax_h3.scheduling_minimax_h3 import MiniMaxH3Scheduler


class _ZeroTransformer:
    def __init__(self):
        self.calls = []

    def __call__(self, *inputs):
        self.calls.append(inputs)
        return torch.zeros_like(inputs[0], dtype=torch.float32), torch.zeros_like(
            inputs[1], dtype=torch.float32
        )


def test_h3_video_patchify_round_trip():
    latents = torch.arange(1 * 3 * 2 * 4 * 6).reshape(1, 3, 2, 4, 6)

    rows = patchify_video_latents(latents, patch_size=(1, 2, 2))
    restored = unpatchify_video_latents(
        rows,
        channels=3,
        num_latent_frames=2,
        latent_height=4,
        latent_width=6,
        patch_size=(1, 2, 2),
    )

    assert rows.shape == (1, 12, 12)
    assert torch.equal(restored, latents)


def test_h3_scheduler_uses_data_ward_velocity_and_terminal_zero():
    scheduler = MiniMaxH3Scheduler(shift=1.0)
    scheduler.set_timesteps(3)
    sample = torch.tensor([2.0])
    velocity = torch.tensor([4.0])

    first = scheduler.step(velocity, scheduler.timesteps[0], sample, return_dict=False)[0]
    second = scheduler.step(velocity, scheduler.timesteps[1], first, return_dict=False)[0]

    assert scheduler.timesteps.tolist() == [0.0, 0.5]
    assert torch.allclose(first, torch.tensor([4.0]))
    assert torch.allclose(second, torch.tensor([6.0]))


def test_h3_t2va_runtime_keeps_static_two_timestep_graph_and_unpacks_latents():
    transformer = _ZeroTransformer()
    embeddings = torch.zeros(1, 8, 16, dtype=torch.bfloat16)
    text_mask = torch.tensor([[True, True, True, False, False, False, False, False]])

    output = denoise_minimax_h3_t2va(
        transformer,
        encoder_hidden_states=embeddings,
        encoder_attention_mask=text_mask,
        num_text_tokens=3,
        height=32,
        width=32,
        num_frames=124,
        num_inference_steps=3,
        seed=7,
        video_scheduler=MiniMaxH3Scheduler(shift=12.0),
        audio_scheduler=MiniMaxH3Scheduler(shift=3.0),
    )

    assert len(transformer.calls) == 2
    first = transformer.calls[0]
    assert first[0].shape == (1, 37, 96)
    assert first[1].shape == (1, 414, 32)
    assert first[3].shape == (2,)
    assert first[4].shape == (512,)
    assert first[-1].shape == (1, 512)
    assert int(first[-1].sum()) == 3 + 414 + 37
    assert output.video_latents.shape == (1, 24, 37, 2, 2)
    assert output.audio_latents.shape == (2, 32, 207)


def test_h3_t2va_runtime_is_seed_deterministic():
    kwargs = dict(
        encoder_hidden_states=torch.zeros(1, 8, 16),
        encoder_attention_mask=torch.tensor(
            [[True, True, False, False, False, False, False, False]]
        ),
        num_text_tokens=2,
        height=32,
        width=32,
        num_frames=124,
        num_inference_steps=2,
        seed=11,
    )

    first = denoise_minimax_h3_t2va(
        _ZeroTransformer(),
        video_scheduler=MiniMaxH3Scheduler(shift=12.0),
        audio_scheduler=MiniMaxH3Scheduler(shift=3.0),
        **kwargs,
    )
    second = denoise_minimax_h3_t2va(
        _ZeroTransformer(),
        video_scheduler=MiniMaxH3Scheduler(shift=12.0),
        audio_scheduler=MiniMaxH3Scheduler(shift=3.0),
        **kwargs,
    )

    assert torch.equal(first.video_latents, second.video_latents)
    assert torch.equal(first.audio_latents, second.audio_latents)
