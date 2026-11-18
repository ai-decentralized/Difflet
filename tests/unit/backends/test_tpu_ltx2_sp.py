"""CPU checks of LTX-2's sequence-parallel block forward (backends/tpu/ltx_2/sp.py).

The collective matmuls are JAX rings that only run on the chips; here they are
replaced by their tp=1 meaning (all-gather = identity, reduce-scatter = the
matmul), so the SP model must reproduce diffusers' own forward. The video
sequence is odd, so it is padded (to a multiple of 2*tp) and the pad keys must
be masked for the result to match.
"""

from __future__ import annotations

import importlib

import pytest
import torch
import torch.nn.functional as F

from difflet.pipeline.parallel_mesh import MeshSpec


@pytest.fixture(autouse=True)
def _mesh():
    # Resolved at run time: another test may drop difflet.backends.tpu.* from
    # sys.modules, and sp.install imports parallel_mesh when it runs.
    pm = importlib.import_module("difflet.backends.tpu.ops_impl.parallel_mesh")
    pm.destroy_parallel_mesh()
    pm.init_parallel_mesh(MeshSpec(tp=1))
    yield
    pm.destroy_parallel_mesh()


def _tiny():
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

    torch.manual_seed(0)
    return LTX2VideoTransformer3DModel(
        in_channels=8, out_channels=8, patch_size=1, patch_size_t=1,
        num_attention_heads=2, attention_head_dim=8, cross_attention_dim=16,
        vae_scale_factors=(8, 32, 32), pos_embed_max_pos=20, base_height=2048,
        base_width=2048, audio_in_channels=8, audio_out_channels=8,
        audio_patch_size=16, audio_patch_size_t=1, audio_num_attention_heads=2,
        audio_attention_head_dim=8, audio_cross_attention_dim=16, audio_scale_factor=4,
        audio_pos_embed_max_pos=20, audio_sampling_rate=16000, audio_hop_length=160,
        num_layers=2, caption_channels=16, rope_double_precision=False,
        use_prompt_embeddings=True, rope_type="split",
    ).eval().double()


def _fake_rings(monkeypatch, cm):
    def ag(x, linears, head_dims=None):  # the real signature: a fake that leaks still fits
        return [F.linear(x, lin.weight, lin.bias) for lin in linears]

    def rs(x, linear):
        return F.linear(x, linear.weight, linear.bias)

    monkeypatch.setattr(cm, "all_gather_matmul", ag)
    monkeypatch.setattr(cm, "matmul_reduce_scatter", rs)
    monkeypatch.setattr(cm, "prepare", lambda ops=None: None)


def _inputs(frames=1, height=3, width=3, audio_frames=5, text=6, batch=2):
    g = torch.Generator().manual_seed(1)
    seq = frames * height * width
    mask = torch.zeros(batch, text, dtype=torch.int64)
    mask[0, 2:] = 1  # left padding, a different length per row
    mask[1, 4:] = 1
    return dict(
        hidden_states=torch.randn(batch, seq, 8, generator=g, dtype=torch.float64),
        audio_hidden_states=torch.randn(batch, audio_frames, 8, generator=g, dtype=torch.float64),
        encoder_hidden_states=torch.randn(batch, text, 16, generator=g, dtype=torch.float64),
        audio_encoder_hidden_states=torch.randn(batch, text, 16, generator=g, dtype=torch.float64),
        timestep=torch.tensor([300.0, 700.0], dtype=torch.float64),
        sigma=torch.tensor([0.3, 0.7], dtype=torch.float64),
        encoder_attention_mask=mask, audio_encoder_attention_mask=mask,
        num_frames=frames, height=height, width=width, audio_num_frames=audio_frames,
        return_dict=False,
    )


def _coords(model, inputs):
    return model.rope.prepare_video_coords(
        inputs["hidden_states"].shape[0], inputs["num_frames"], inputs["height"],
        inputs["width"], inputs["hidden_states"].device, fps=24.0)


def _sp_forward(sp, model, inputs):
    """What the TPU application does around the compiled model: pad eagerly,
    run, cut the video output back."""
    seq, padded = model._difflet_sp_rows
    hidden, coords = sp.pad_video_inputs(inputs["hidden_states"], _coords(model, inputs), padded)
    video, audio = model(**dict(inputs, hidden_states=hidden, video_coords=coords))
    return sp.unpad_video_output(video, seq), audio


@pytest.mark.parametrize("dims,shared", [
    ((1, 3, 3), "1"),  # 9 rows, padded to 10
    ((2, 2, 2), "1"),  # 8 rows, no pad
    ((1, 3, 3), "0"),  # a2v and v2a each through its own ring
])
def test_sp_forward_reproduces_diffusers_at_tp1(monkeypatch, dims, shared):
    sp = importlib.import_module("difflet.backends.tpu.ltx_2.sp")
    monkeypatch.setenv(sp.SHARED_CA_GATHER_ENV, shared)
    cm = importlib.import_module("difflet.backends.tpu.ops_impl.collective_matmul")
    _fake_rings(monkeypatch, cm)
    frames, height, width = dims
    ref_model = _tiny()
    inputs = _inputs(frames, height, width)
    with torch.no_grad():
        want = ref_model(**inputs)
    model = _tiny()
    seq = frames * height * width
    assert sp.install(model, video_seq_len=seq)
    assert model._difflet_sp_rows == (seq, sp.padded_length(seq, 1))
    with torch.no_grad():
        got = _sp_forward(sp, model, inputs)
    for g, w in zip(got, want):
        assert g.shape == w.shape
        # 1e-6, not float64 rounding: difflet's CPU attention op takes its
        # softmax in fp32 (diffusers' SDPA stays in float64), ~5e-8 here.
        torch.testing.assert_close(g, w, rtol=1e-6, atol=1e-6)


def test_padding_without_dropping_the_pad_keys_would_differ(monkeypatch):
    """Guards the pad-key handling: attending over the pad rows changes the result."""
    sp = importlib.import_module("difflet.backends.tpu.ltx_2.sp")
    cm = importlib.import_module("difflet.backends.tpu.ops_impl.collective_matmul")
    _fake_rings(monkeypatch, cm)
    inputs = _inputs(1, 3, 3)
    with torch.no_grad():
        want = _tiny()(**inputs)
    model = _tiny()
    sp.install(model, video_seq_len=9)
    real = sp._attend
    monkeypatch.setattr(sp, "_attend", lambda *a, valid_keys=None, **k: real(*a, **k))
    with torch.no_grad():
        got = _sp_forward(sp, model, inputs)
    assert (got[0] - want[0]).abs().max() > 1e-3


def test_unpadded_stream_is_refused(monkeypatch):
    """Padding inside the compiled graph broke the rings on torch_tpu; the first
    block insists on an already padded stream."""
    sp = importlib.import_module("difflet.backends.tpu.ltx_2.sp")
    cm = importlib.import_module("difflet.backends.tpu.ops_impl.collective_matmul")
    _fake_rings(monkeypatch, cm)
    model = _tiny()
    sp.install(model, video_seq_len=9)
    with pytest.raises(ValueError, match="padded to 10"), torch.no_grad():
        model(**_inputs(1, 3, 3))


def test_pad_and_unpad_video_rows():
    sp = importlib.import_module("difflet.backends.tpu.ltx_2.sp")
    h = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    c = torch.arange(2 * 3 * 3 * 2, dtype=torch.float32).reshape(2, 3, 3, 2)
    hp, cp = sp.pad_video_inputs(h, c, 4)
    assert hp.shape == (2, 4, 2) and not hp[:, 3].any()
    assert cp.shape == (2, 3, 4, 2) and torch.equal(cp[:, :, 3], c[:, :, 2])
    assert torch.equal(sp.unpad_video_output(hp, 3), h)
    assert sp.pad_video_inputs(h, c, 3)[0] is h


def test_padded_length():
    sp = importlib.import_module("difflet.backends.tpu.ltx_2.sp")
    assert sp.padded_length(2310, 4) == 2312
    assert sp.padded_length(6144, 4) == 6144
    assert sp.padded_length(9, 1) == 10
    # ring halves on whole bf16 tiles: 2*tp*16 rows (LTX-2 480x704x49: halves of 304)
    assert sp.padded_length(2310, 4, row_align=16) == 2432
    assert sp.padded_length(2304, 4, row_align=16) == 2304
