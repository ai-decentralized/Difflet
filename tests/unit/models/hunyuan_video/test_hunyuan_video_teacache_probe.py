"""CPU parity tests for the HunyuanVideo TeaCache probe model.

These tests live in ``difflet/backends/trainium/`` namespace by way of the
``HunyuanVideoTeacacheProbeModel`` class, but they exercise only the CPU
computational path — no Trainium load, no NEFF compile. The goal is to
prove that the probe's forward is numerically identical to calling
``HunyuanVideoTransformer3DModel.teacache_mod_input`` directly, before we
trust the probe NEFF for hardware calibration.

Pattern mirrors ``tests/unit/test_hunyuan_video_attention.py``.
"""

from __future__ import annotations

import torch


def _tiny_kwargs() -> dict:
    return {
        "in_channels": 2,
        "out_channels": 2,
        "num_attention_heads": 2,
        "attention_head_dim": 6,
        "num_layers": 1,
        "num_single_layers": 1,
        "num_refiner_layers": 1,
        "mlp_ratio": 2.0,
        "patch_size": 2,
        "patch_size_t": 1,
        "guidance_embeds": True,
        "text_embed_dim": 6,
        "pooled_projection_dim": 5,
        "rope_axes_dim": (2, 2, 2),
    }


def _tiny_bundle():
    torch.manual_seed(123)
    hidden_states = torch.randn(1, 2, 2, 4, 4)
    timestep = torch.tensor([7], dtype=torch.long)
    encoder_hidden_states = torch.randn(1, 4, 6)
    encoder_attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
    pooled_projections = torch.randn(1, 5)
    guidance = torch.tensor([3], dtype=torch.long)
    return (
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        pooled_projections,
        guidance,
    )


def test_probe_mod_input_matches_cpu_teacache_mod_input(monkeypatch):
    """The probe's mod_input output must be bit-identical to the CPU model's
    own teacache_mod_input, weight-for-weight."""
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")

    from difflet.backends.trainium.hunyuan_video.teacache_probe_model import (
        HunyuanVideoTeacacheProbeModel,
    )
    from difflet.models.hunyuan_video.modeling_hunyuan_video import (
        HunyuanVideoTransformer3DModel,
    )

    kwargs = _tiny_kwargs()
    torch.manual_seed(20)
    reference = HunyuanVideoTransformer3DModel(**kwargs).eval()
    probe = HunyuanVideoTeacacheProbeModel(reference.config).eval()
    probe.model.load_state_dict(reference.state_dict())

    (
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        pooled_projections,
        guidance,
    ) = _tiny_bundle()

    # Build a zero prev_mod_input matching mod_input shape so the delta is
    # exactly ||mod_input||.
    p, p_t = kwargs["patch_size"], kwargs["patch_size_t"]
    latent_frames, latent_h, latent_w = 2, 4, 4
    seq_len = (latent_frames // p_t) * (latent_h // p) * (latent_w // p)
    inner_dim = kwargs["num_attention_heads"] * kwargs["attention_head_dim"]
    zero_prev = torch.zeros(1, seq_len, inner_dim)

    with torch.no_grad():
        ref_mod_input = reference.teacache_mod_input(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            pooled_projections,
            guidance,
        )
        delta, probe_mod_input = probe(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            pooled_projections,
            guidance,
            zero_prev,
        )

    assert torch.equal(probe_mod_input, ref_mod_input)

    expected_delta = torch.linalg.vector_norm((ref_mod_input - zero_prev).reshape(-1))
    assert torch.allclose(delta, expected_delta, atol=0.0, rtol=0.0)


def test_probe_delta_uses_caller_prev_mod_input(monkeypatch):
    """Delta must be a function of (mod_input, prev_mod_input), not just
    mod_input. Pass two different prev tensors and confirm the deltas differ."""
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")

    from difflet.backends.trainium.hunyuan_video.teacache_probe_model import (
        HunyuanVideoTeacacheProbeModel,
    )
    from difflet.models.hunyuan_video.modeling_hunyuan_video import (
        HunyuanVideoTransformer3DModel,
    )

    kwargs = _tiny_kwargs()
    torch.manual_seed(20)
    reference = HunyuanVideoTransformer3DModel(**kwargs).eval()
    probe = HunyuanVideoTeacacheProbeModel(reference.config).eval()
    probe.model.load_state_dict(reference.state_dict())

    inputs = _tiny_bundle()
    seq_len = 2 * (4 // 2) * (4 // 2)  # latent_frames * (h/p) * (w/p)
    inner_dim = kwargs["num_attention_heads"] * kwargs["attention_head_dim"]

    torch.manual_seed(99)
    prev_a = torch.randn(1, seq_len, inner_dim)
    prev_b = torch.randn(1, seq_len, inner_dim)

    with torch.no_grad():
        delta_a, mod_a = probe(*inputs, prev_a)
        delta_b, mod_b = probe(*inputs, prev_b)

    assert torch.equal(mod_a, mod_b)  # mod_input independent of prev
    assert not torch.allclose(delta_a, delta_b)  # delta depends on prev


def test_probe_delta_zero_when_prev_equals_current(monkeypatch):
    """Sanity check: delta is exactly 0 when prev_mod_input == mod_input."""
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")

    from difflet.backends.trainium.hunyuan_video.teacache_probe_model import (
        HunyuanVideoTeacacheProbeModel,
    )
    from difflet.models.hunyuan_video.modeling_hunyuan_video import (
        HunyuanVideoTransformer3DModel,
    )

    kwargs = _tiny_kwargs()
    torch.manual_seed(20)
    reference = HunyuanVideoTransformer3DModel(**kwargs).eval()
    probe = HunyuanVideoTeacacheProbeModel(reference.config).eval()
    probe.model.load_state_dict(reference.state_dict())

    inputs = _tiny_bundle()
    with torch.no_grad():
        mod_input = reference.teacache_mod_input(*inputs)
        delta, probe_mod = probe(*inputs, mod_input)

    assert torch.equal(probe_mod, mod_input)
    assert delta.item() == 0.0


def test_probe_module_state_dict_matches_wrapped_model(monkeypatch):
    """The probe wrapper holds a full HunyuanVideoTransformer3DModel under the
    ``model.`` prefix; this lets the production HF state dict load by simply
    prefixing keys. Verify that round-trip."""
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")

    from difflet.backends.trainium.hunyuan_video.teacache_probe_model import (
        HunyuanVideoTeacacheProbeModel,
    )
    from difflet.models.hunyuan_video.modeling_hunyuan_video import (
        HunyuanVideoTransformer3DModel,
    )

    kwargs = _tiny_kwargs()
    torch.manual_seed(20)
    reference = HunyuanVideoTransformer3DModel(**kwargs).eval()
    probe = HunyuanVideoTeacacheProbeModel(reference.config).eval()

    reference_keys = set(reference.state_dict().keys())
    probe_keys = set(probe.state_dict().keys())
    expected_probe_keys = {f"model.{key}" for key in reference_keys}
    assert probe_keys == expected_probe_keys
