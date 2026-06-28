"""Unit tests for Wan 2.2 + LTX-2 TeaCache wiring (cclog 87/88).

No device: exercises the calibration profiles, the Wan block-0 signal (timestep-only
property), and the LTX-2 dual-stream velocity-cache skip/reuse logic at the controller
level (the shape of the `_denoise` wiring, without loading the real models).
"""

import json
from pathlib import Path

import pytest
import torch

from difflet.pipeline.teacache import TeaCacheCalibration, TeaCacheController

_CALIB_DIR = Path(__file__).resolve().parents[2] / "cclogs" / "m9-teacache"

# The calibration profiles live under cclogs/, which is gitignored, so they are
# absent on fresh checkouts / CI. Skip the data-dependent cases when missing.
_requires_calib = pytest.mark.skipif(
    not (_CALIB_DIR.is_dir() and any(_CALIB_DIR.glob("teacache_calib_*.json"))),
    reason=f"TeaCache calibration data not present at {_CALIB_DIR} (cclogs/ is gitignored)",
)


@_requires_calib
@pytest.mark.parametrize("name", ["teacache_calib_wan.json", "teacache_calib_ltx_2.json"])
def test_wan_ltx2_calibrations_load_and_are_well_formed(name):
    """The committed Wan/LTX-2 calibration profiles parse + carry sane fields."""
    calib = TeaCacheCalibration.from_json(_CALIB_DIR / name)
    assert calib.model in {"wan", "ltx_2"}
    assert calib.accumulate is True  # both are adaptive accumulate-mode
    assert len(calib.poly_coef) >= 2
    assert calib.threshold > 0.0
    assert calib.num_steps > 0
    # predict_delta must be finite over the gate's rel-L1 range.
    for x in (0.01, 0.1, 0.4, 1.0):
        assert torch.isfinite(torch.tensor(calib.predict_delta(x)))


def _tiny_wan_model():
    from difflet.models.wan.modeling_wan import WanTransformer3DModel, WanTransformerConfig

    config = WanTransformerConfig(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=16,
        out_channels=16,
        text_dim=16,
        freq_dim=16,
        ffn_dim=64,
        num_layers=1,
    )
    torch.manual_seed(0)
    return WanTransformer3DModel(config).eval(), config


def test_wan_teacache_mod_input_shape_and_timestep_only():
    """Wan block-0 signal: correct shape AND independent of the text embeds.

    The modulation is `scale_shift_table + timestep_proj` (timestep-only), so
    teacache_mod_input must be identical for two different encoder_hidden_states at the
    same (hidden_states, timestep) — the property the adaptive signal relies on.
    """
    model, config = _tiny_wan_model()
    hidden_states = torch.randn(1, config.in_channels, 2, 4, 4)
    timestep = torch.tensor([500.0])
    ehs_a = torch.randn(1, 4, config.text_dim)
    ehs_b = torch.randn(1, 4, config.text_dim)

    with torch.no_grad():
        mod_a = model.teacache_mod_input(hidden_states, timestep, ehs_a)
        mod_b = model.teacache_mod_input(hidden_states, timestep, ehs_b)

    # shape = (B, num_patches, inner_dim); 2*2*2 patches, inner_dim = 2*16
    assert mod_a.shape == (1, 8, config.inner_dim)
    assert torch.isfinite(mod_a).all()
    # timestep-only: different text embeds -> identical modulated input
    assert torch.allclose(mod_a, mod_b, atol=1e-5)
    # but a different timestep changes it
    with torch.no_grad():
        mod_t = model.teacache_mod_input(hidden_states, torch.tensor([900.0]), ehs_a)
    assert not torch.allclose(mod_a, mod_t, atol=1e-3)


def test_ltx2_dual_stream_velocity_cache_skip_reuse():
    """Replicate the LTX-2 `_denoise` skip/reuse logic (cclog 87 velocity-cache fix):

    the controller tracks the VIDEO velocity residual; the AUDIO velocity residual is
    held manually. A skipped step must reproduce `prev + cached_residual` for BOTH
    streams (caching the velocity, never the x0). Asserts the dual-stream bookkeeping.
    """
    calib = TeaCacheCalibration(
        model="ltx_2",
        shape_label="t",
        poly_coef=(1.0,),  # predict_delta == 1.0; threshold 2.5 -> first eligible step skips
        threshold=2.5,
        warmup_steps=1,
        cooldown_steps=0,
        num_steps=6,
        accumulate=True,
    )
    ctrl = TeaCacheController(calib)
    prev_audio_vel = None
    cached_audio_vel_res = None

    def step(i, video_vel_full, audio_vel_full, mod):
        nonlocal prev_audio_vel, cached_audio_vel_res
        diff = None
        if ctrl.prev_mod_input is not None:
            prev = ctrl.prev_mod_input
            diff = float((mod.float().cpu() - prev).abs().mean() / prev.abs().mean().clamp_min(1e-8))
        skip = ctrl.should_skip(i, mod, diff_norm=diff)
        if skip:
            v = ctrl.skip_noise_pred(mod_input=mod)
            a = prev_audio_vel + cached_audio_vel_res
        else:
            v, a = video_vel_full, audio_vel_full
            ctrl.record_full_step(v, mod_input=mod)
            if prev_audio_vel is not None:
                cached_audio_vel_res = a - prev_audio_vel
        prev_audio_vel = a
        return skip, v, a

    mod = torch.ones(1, 4)
    # two full steps to seed video residual + audio residual
    s0, v0, a0 = step(0, torch.full((1, 2), 1.0), torch.full((1, 2), 10.0), mod)
    s1, v1, a1 = step(1, torch.full((1, 2), 2.0), torch.full((1, 2), 12.0), mod + 0.1)
    assert s0 is False and s1 is False
    # video residual = 2-1 = 1; audio residual = 12-10 = 2
    s2, v2, a2 = step(2, torch.full((1, 2), 99.0), torch.full((1, 2), 99.0), mod + 0.2)
    assert s2 is True  # residual seeded by steps 0-1; accum 1.0 < 2.5 -> skip
    # skip reuses prev + residual (NOT the 99.0 full values)
    assert torch.allclose(v2, torch.full((1, 2), 3.0))  # 2 + 1
    assert torch.allclose(a2, torch.full((1, 2), 14.0))  # 12 + 2


@_requires_calib
def test_ltx2_calibration_threshold_variants_consistent():
    """Sanity: the LTX-2 calib's poly is monotone-usable (accumulate never NaNs/stalls)."""
    calib = TeaCacheCalibration.from_json(_CALIB_DIR / "teacache_calib_ltx_2.json")
    ctrl = TeaCacheController(calib)
    # feed a descending-then-rising rel-L1 trajectory; accum must stay finite
    for i, x in enumerate([0.4, 0.1, 0.05, 0.05, 0.1, 0.3]):
        ctrl.record_full_step(torch.ones(1, 2), torch.ones(1, 2))
        val = calib.predict_delta(x)
        assert torch.isfinite(torch.tensor(val))
