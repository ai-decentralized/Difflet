"""MiniMax-H3 FL2VA (first/last keyframe) host-side contract.

The values the official diffusers ``modular_pipelines/minimax_h3`` produces were checked on trn3
against diffusers main (keyframes and layout bit-exact, conditions to 1e-5); these tests pin the
structure on CPU.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch
from PIL import Image

from difflet.models.minimax_h3.contracts import (
    AUDIO_TAG,
    KEYFRAME_NOISE_AUG,
    TEXT_TAG,
    VIDEO_TAG,
    build_padded_t2va_layout,
    build_t2va_layout,
    keyframe_anchor_time,
    resolve_canvas_size,
)
from difflet.models.minimax_h3.keyframes import (
    build_fl2va_presentation,
    encode_keyframe_condition,
    keyframe_pixels,
    prepare_keyframes,
)
from difflet.models.minimax_h3.pipeline import (
    denoise_minimax_h3_t2va,
    patchify_video_latents,
    step_timestep_slots,
)
from difflet.models.minimax_h3.scheduling_minimax_h3 import MiniMaxH3Scheduler

H, W, F = 64, 128, 124  # latent 4x8 -> 2x4 = 8 rows per frame


# ------------------------------------------------------------------------------ layout


def test_fl2va_layout_puts_one_frame_block_per_keyframe_between_text_and_audio():
    tags = torch.tensor([TEXT_TAG, VIDEO_TAG, VIDEO_TAG, TEXT_TAG, TEXT_TAG])
    t2va = build_t2va_layout(num_text_tokens=5, height=H, width=W, num_frames=F)
    layout = build_t2va_layout(num_text_tokens=5, height=H, width=W, num_frames=F,
                               text_token_tags=tags, keyframe_anchors=("first", "last"))
    rows = 8
    assert layout.num_condition_video_rows == 2 * rows
    assert layout.sequence_length == t2va.sequence_length + 2 * rows
    # [text | conditions | audio | video]; the video indices lead with the condition rows.
    assert layout.video_indices[: 2 * rows].tolist() == list(range(5, 5 + 2 * rows))
    assert int(layout.audio_indices[0]) == 5 + 2 * rows
    assert torch.equal(layout.token_tags[:5], tags)
    assert (layout.token_tags[layout.video_indices] == VIDEO_TAG).all()
    assert (layout.token_tags[layout.audio_indices] == AUDIO_TAG).all()
    # The generated rows keep the T2VA positions. The first block sits at the first frame's
    # time; the last block at the clip's end less one pixel frame (5/3), which is the last
    # latent frame's time plus its span less 5/3 (the reference's arithmetic).
    gen = layout.video_indices[2 * rows:]
    assert torch.equal(layout.position_ids[gen], t2va.position_ids[t2va.video_indices])
    first = layout.position_ids[5:5 + rows]
    last = layout.position_ids[5 + rows:5 + 2 * rows]
    gen_pos = layout.position_ids[gen]
    assert torch.equal(first, gen_pos[:rows])
    assert torch.equal(last[:, 1:], gen_pos[-rows:, 1:])
    last_span = 5.0 / 3.0 * (1, 4, 4, 4, 4)[(37 - 1) % 5]
    assert float(last[0, 0]) == pytest.approx(float(gen_pos[-1, 0]) + last_span - 5.0 / 3.0)
    assert float(last[0, 0]) == keyframe_anchor_time("last", num_text_tokens=5, num_latent_frames=37)
    assert torch.equal(t2va.token_tags, build_t2va_layout(
        num_text_tokens=5, height=H, width=W, num_frames=F).token_tags)


def test_fl2va_layout_rejects_two_keyframes_at_one_end():
    with pytest.raises(ValueError, match="one 'first' and one 'last'"):
        build_t2va_layout(num_text_tokens=3, height=H, width=W, num_frames=F,
                          keyframe_anchors=("first", "first"))
    with pytest.raises(ValueError, match="'first' or 'last'"):
        keyframe_anchor_time("middle", num_text_tokens=3, num_latent_frames=37)


def test_padded_fl2va_layout_keeps_the_text_padding_at_the_tail():
    layout = build_padded_t2va_layout(num_text_tokens=5, max_text_tokens=16, height=H, width=W,
                                      num_frames=F, sequence_alignment=128,
                                      keyframe_anchors=("last",))
    live = build_t2va_layout(num_text_tokens=5, height=H, width=W, num_frames=F,
                             keyframe_anchors=("last",))
    assert layout.num_condition_video_rows == 8 and layout.keyframe_anchors == ("last",)
    assert torch.equal(layout.video_indices, live.video_indices)
    assert layout.sequence_length % 128 == 0
    assert not layout.attention_mask[live.sequence_length:].any()


def test_canvas_size_matches_the_released_768_short_edge_geometry():
    assert resolve_canvas_size(1280, 720) == (768, 1344)
    assert resolve_canvas_size(720, 1280) == (1344, 768)
    assert resolve_canvas_size(1, 1) == (768, 768)
    assert resolve_canvas_size(4, 1) == (512, 2016)  # area-capped: 768x3072 > 768x1344
    with pytest.raises(ValueError, match="aspect ratios"):
        resolve_canvas_size(5, 1)


# ------------------------------------------------------------------------------ keyframes


def _image(width, height, seed=0, mode="RGB"):
    rng = np.random.default_rng(seed)
    channels = 4 if mode == "RGBA" else 3
    return Image.fromarray(rng.integers(0, 255, (height, width, channels), dtype=np.uint8), mode)


def test_prepare_keyframes_stretches_the_anchor_and_cover_crops_the_follower():
    first, last = _image(1280, 720), _image(1000, 800, seed=1)
    keyframes, anchors, height, width = prepare_keyframes(first, last)
    assert (height, width) == (768, 1344) and anchors == ("first", "last")
    assert keyframes[0].size == (1344, 768)
    assert np.array_equal(np.array(keyframes[0]),
                          np.array(first.resize((1344, 768), Image.Resampling.LANCZOS)))
    # Cover: scale = max(1344/1000, 768/800) = 1.344 -> 1344x1075, centre crop rows 153..921.
    resized = last.resize((1344, 1075), Image.Resampling.LANCZOS)
    assert np.array_equal(np.array(keyframes[1]), np.array(resized.crop((0, 153, 1344, 921))))


def test_prepare_keyframes_last_only_and_explicit_canvas():
    keyframes, anchors, height, width = prepare_keyframes(
        None, _image(300, 200, mode="RGBA"), height=256, width=448)
    assert anchors == ("last",) and (height, width) == (256, 448)
    assert keyframes[0].mode == "RGB" and keyframes[0].size == (448, 256)
    with pytest.raises(ValueError, match="first and/or a last"):
        prepare_keyframes(None, None)
    with pytest.raises(ValueError, match="together"):
        prepare_keyframes(_image(64, 64), None, height=256)


class _Tokenizer:
    _special = {"<|vision_start|>": 900, "<|image_pad|>": 901, "<|vision_end|>": 902}

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [100 + len(word) for word in text.split()]}

    def convert_tokens_to_ids(self, token):
        return self._special[token]


class _ImageProcessor:
    merge_size = 2

    def __call__(self, images, return_tensors="pt"):
        grids = torch.tensor([[1, img.size[1] // 16, img.size[0] // 16] for img in images])
        return {"pixel_values": torch.zeros(int(grids.prod(-1).sum()), 4), "image_grid_thw": grids}


def test_presentation_labels_each_keyframe_and_tags_its_vision_block_as_video():
    keyframes = [_image(64, 32), _image(64, 32, seed=1)]
    ids, tags, vision = build_fl2va_presentation(_Tokenizer(), _ImageProcessor(), "a b c", keyframes)
    per_image = (32 // 16) * (64 // 16) // 4  # merged image tokens
    label = len("<Picture 1>: ".split())
    block = label + 1 + per_image + 1
    assert len(ids) == 2 * block + 3 and tags.shape == (len(ids),)
    assert ids[label:label + per_image + 2] == [900] + [901] * per_image + [902]
    assert (tags[:label] == TEXT_TAG).all() and (tags[label:block] == VIDEO_TAG).all()
    assert (tags[-3:] == TEXT_TAG).all()
    assert set(vision) == {"pixel_values", "image_grid_thw"}
    text_ids, text_tags, none = build_fl2va_presentation(_Tokenizer(), _ImageProcessor(), "a b", [])
    assert len(text_ids) == 2 and (text_tags == TEXT_TAG).all() and none == {}


def test_condition_encode_is_seeded_independently_of_the_request():
    pixels = keyframe_pixels(_image(32, 16))
    assert pixels.shape == (1, 3, 1, 16, 32) and pixels.dtype == torch.uint8
    moments = lambda x: torch.cat([x.mean(1, keepdim=True).repeat(1, 2, 1, 1, 1),
                                   torch.full_like(x[:, :2], -2.0)], dim=1)
    kwargs = dict(latents_mean=[0.5, -0.5], latents_std=[2.0, 4.0])
    torch.manual_seed(0)
    a = encode_keyframe_condition(moments, pixels, **kwargs)
    torch.manual_seed(123)
    b = encode_keyframe_condition(moments, pixels, **kwargs)
    c = encode_keyframe_condition(moments, pixels, encode_seed=7, **kwargs)
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert a.dtype == torch.float32 and a.shape == (1, 2, 1, 16, 32)
    # The sampled latent is rounded through fp16 before the normalization.
    raw = a * torch.tensor([2.0, 4.0]).view(1, -1, 1, 1, 1) + torch.tensor([0.5, -0.5]).view(1, -1, 1, 1, 1)
    assert torch.allclose(raw, raw.half().float(), atol=1e-5)


# ------------------------------------------------------------------------------ denoise


def test_step_timestep_slots_add_the_condition_slot_held_at_0999():
    video = torch.tensor([0.0, 0.5, 0.9995])
    audio = torch.tensor([0.0, 0.3, 0.9])
    t2va = step_timestep_slots(video, audio)
    fl2va = step_timestep_slots(video, audio, keyframes=True)
    assert t2va.shape == (3, 2) and fl2va.shape == (3, 3)
    assert torch.equal(fl2va[:, :2], t2va)
    assert fl2va[:, 2].tolist() == pytest.approx([KEYFRAME_NOISE_AUG, KEYFRAME_NOISE_AUG, 0.9995])


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *inputs):
        self.calls.append(inputs)
        video, audio = inputs[0], inputs[1]
        return torch.ones_like(video, dtype=torch.float32), torch.zeros_like(audio, dtype=torch.float32)


def _fl2va_run(transformer, conditions, anchors, seed=3, steps=3, **kwargs):
    n = 6
    return denoise_minimax_h3_t2va(
        transformer, encoder_hidden_states=torch.zeros(1, n, 8),
        encoder_attention_mask=torch.ones(1, n, dtype=torch.bool), num_text_tokens=n,
        height=H, width=W, num_frames=F, num_inference_steps=steps, seed=seed,
        video_scheduler=MiniMaxH3Scheduler(shift=12.0), audio_scheduler=MiniMaxH3Scheduler(shift=3.0),
        model_dtype=torch.float32, condition_latents=conditions, keyframe_anchors=anchors, **kwargs)


def test_fl2va_denoise_draws_conditions_first_and_never_steps_them():
    conditions = [torch.randn(1, 24, 1, H // 16, W // 16, generator=torch.Generator().manual_seed(i))
                  for i in range(2)]
    tags = torch.tensor([TEXT_TAG, VIDEO_TAG, VIDEO_TAG, VIDEO_TAG, TEXT_TAG, TEXT_TAG])
    recorder = _Recorder()
    _fl2va_run(recorder, conditions, ("first", "last"), text_token_tags=tags)
    rows = 8
    first_video = recorder.calls[0][0][0]
    # The generator draws one noise per condition first, then the video, then the audio.
    gen = torch.Generator().manual_seed(3)
    expected = []
    for condition in conditions:
        noise = torch.randn(condition.shape, generator=gen)
        mixed = KEYFRAME_NOISE_AUG * condition + (1 - KEYFRAME_NOISE_AUG) * noise
        expected.append(patchify_video_latents(mixed)[0])
    video_noise = torch.randn(1, 24, 37, H // 16, W // 16, generator=gen)
    assert torch.allclose(first_video[: 2 * rows], torch.cat(expected), atol=1e-6)
    assert torch.equal(first_video[2 * rows:], patchify_video_latents(video_noise)[0])
    # The condition rows reach every call unchanged, while the generated rows move.
    for call in recorder.calls[1:]:
        assert torch.equal(call[0][0][: 2 * rows], first_video[: 2 * rows])
        assert not torch.equal(call[0][0][2 * rows:], first_video[2 * rows:])
    # Three timestep slots; the condition rows read slot 2, held at max(t, 0.999).
    timestep, indices, token_tags = recorder.calls[0][3], recorder.calls[0][4], recorder.calls[0][5]
    assert timestep.shape == (3,) and timestep[2] == pytest.approx(max(float(timestep[1]), 0.999))
    assert (indices[6:6 + 2 * rows] == 2).all() and (indices[:6] == 1).all()
    assert torch.equal(token_tags[:6], tags)


def test_fl2va_denoise_checks_conditions_against_the_layout_and_the_table():
    with pytest.raises(ValueError, match="condition latents for"):
        _fl2va_run(_Recorder(), [torch.zeros(1, 24, 1, 4, 8)], ("first", "last"))
    with pytest.raises(ValueError, match="encoded at"):
        _fl2va_run(_Recorder(), [torch.zeros(1, 24, 1, 8, 8)], ("first",))
    table = torch.zeros(2, 1, 6, 6, 4)  # a T2VA (two-slot) table
    with pytest.raises(ValueError, match="3 modalities"):
        _fl2va_run(_Recorder(), [torch.zeros(1, 24, 1, 4, 8)], ("first",),
                   adaln_modulation_table=table)


# ------------------------------------------------------------------------------ host encoder


def test_host_video_encoder_tiles_like_one_pass_when_the_frame_fits_a_tile():
    from difflet.models.minimax_h3.host_components import HostVideoEncoder

    conv = torch.nn.Conv3d(3, 4, kernel_size=(1, 16, 16), stride=(1, 16, 16))
    cfg = {"spatial_downsample_factors": [2, 2, 2, 2, 1, 1], "latents_mean": [0.0, 0.0],
           "latents_std": [1.0, 1.0]}
    encoder = HostVideoEncoder(conv.eval(), cfg)
    x = torch.randn(1, 3, 1, 64, 128)
    assert torch.equal(encoder.moments(x), conv(x))
    tiled = HostVideoEncoder(conv.eval(), cfg, tile=64, min_overlap=16).moments(x)
    assert tiled.shape == conv(x).shape
    assert torch.allclose(tiled, conv(x), atol=1e-5)  # a per-patch conv blends exactly
    with pytest.raises(ValueError, match="single keyframes"):
        encoder.moments(torch.zeros(1, 3, 2, 64, 64))


def test_vision_pos_embed_resampling_is_bilinear_align_corners_in_fp32():
    from difflet.models.minimax_h3.host_components import _pos_embed_interpolate_fp32_weights

    side, channels, merge = 6, 5, 2
    table = torch.randn(side * side, channels)
    vision = types.SimpleNamespace(
        num_grid_per_side=side, config=types.SimpleNamespace(spatial_merge_size=merge),
        pos_embed=torch.nn.Embedding.from_pretrained(table))
    grid = torch.tensor([[1, 4, 8], [2, 6, 4]])
    out = _pos_embed_interpolate_fp32_weights(vision, grid)
    pieces = []
    for t, h, w in grid.tolist():
        dense = torch.nn.functional.interpolate(
            table.T.reshape(1, channels, side, side), size=(h, w), mode="bilinear",
            align_corners=True)[0].permute(1, 2, 0)
        merged = dense.reshape(h // merge, merge, w // merge, merge, channels).permute(0, 2, 1, 3, 4)
        pieces.append(merged.reshape(-1, channels).repeat(t, 1))
    assert torch.allclose(out, torch.cat(pieces), atol=1e-5)


# ------------------------------------------------------------------------------ neuron config


def test_neuron_config_packs_the_keyframe_rows_and_a_third_slot():
    from difflet.backends.neuron.minimax_h3.transformer import NeuronMiniMaxH3Config

    t2va = NeuronMiniMaxH3Config(height=256, width=448, num_frames=124, text_seq_len=128)
    fl2va = NeuronMiniMaxH3Config(height=256, width=448, num_frames=124, text_seq_len=128,
                                  num_keyframes=2)
    rows = (256 // 32) * (448 // 32)
    assert t2va.num_timestep_slots == 2 and fl2va.num_timestep_slots == 3
    assert fl2va.video_seq_len == t2va.video_seq_len + 2 * rows
    layout = fl2va.layout(64)
    assert layout.num_condition_video_rows == 2 * rows
    assert layout.video_indices.numel() == fl2va.video_seq_len
    with pytest.raises(ValueError, match="num_keyframes"):
        NeuronMiniMaxH3Config(num_keyframes=3).validate()
