import argparse

import pytest

from difflet.cli.modes import ModeConfig, model_class, resolve_mode

WAN = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
FLUX = "black-forest-labs/FLUX.1-dev"
HYV = "hunyuanvideo-community/HunyuanVideo"
HYV15 = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
QWEN = "Qwen/Qwen-Image"
LTX = "Lightricks/LTX-2"


def _args(**kw):
    ns = argparse.Namespace(dp=None, cp_degree=None, cfg_parallel=False)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_all_six_models_classified():
    for mid in (WAN, FLUX, HYV, HYV15, QWEN, LTX):
        assert model_class(mid) in {"distilled", "true_cfg", "true_cfg_no_cp"}


@pytest.mark.parametrize("mid", [FLUX, HYV, HYV15, QWEN])
def test_distilled_never_gets_cfg2(mid):
    for mode in ("latency", "throughput", "mixed"):
        assert resolve_mode(mid, mode, _args()).cfg_parallel is False


def test_mode_table_matches_spec():
    assert resolve_mode(FLUX, "latency", _args()) == ModeConfig(1, False, 4)
    assert resolve_mode(FLUX, "throughput", _args()) == ModeConfig(4, False, 1)
    assert resolve_mode(FLUX, "mixed", _args()) == ModeConfig(2, False, 2)
    assert resolve_mode(WAN, "latency", _args()) == ModeConfig(1, True, 1)
    assert resolve_mode(WAN, "throughput", _args()) == ModeConfig(4, False, 1)
    assert resolve_mode(WAN, "mixed", _args()) == ModeConfig(2, True, 1)


def test_ltx2_cp_always_capped():
    for mode in ("latency", "throughput", "mixed"):
        cfg = resolve_mode(LTX, mode, _args())
        assert cfg.cp_degree == 1


def test_explicit_flags_override_mode():
    cfg = resolve_mode(FLUX, "throughput", _args(dp=2, cp_degree=2))
    assert cfg == ModeConfig(2, False, 2)


def test_no_mode_returns_none_and_unknown_raises():
    assert resolve_mode(FLUX, None, _args()) is None
    with pytest.raises(ValueError, match="unknown mode"):
        resolve_mode(FLUX, "warp", _args())
    with pytest.raises(ValueError, match="unknown model"):
        resolve_mode("nope/nope", "latency", _args())
