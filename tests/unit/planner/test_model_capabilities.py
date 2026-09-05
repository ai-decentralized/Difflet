"""Registry ModelCapabilities: the single source of truth for model x strategy.

The support matrix asserted here is the one documented in README.md's "Feature
support" table and exercised on device by scripts/verify_cli.py. Head counts are
read from each model's shipped transformer/config.json.
"""
from __future__ import annotations

import pytest

from difflet.pipeline.parallel_config import CP_MODES
from difflet.registry import ModelCapabilities, registered_models, resolve_model

# model name -> (heads, distilled, supports_cp, supports_sp)
EXPECTED = {
    "flux": (24, True, True, True),
    "qwen_image": (24, True, True, True),
    "wan": (40, False, True, True),
    "hunyuan_video": (24, True, True, True),
    "hunyuan_video_15": (16, True, False, False),
    "ltx_2": (32, False, False, False),
}


def _capabilities(name: str) -> ModelCapabilities:
    return resolve_model("", model_type=name).require_capabilities()


def test_every_builtin_model_declares_capabilities():
    """Only the builtins: other suites register dummy entries into the registry.

    ``capabilities`` stays optional on ``ModelEntry`` precisely so those test
    fixtures (and any out-of-tree model) need not fill it in; the planner calls
    ``require_capabilities()`` and fails loudly instead of guessing.
    """

    builtins = {entry.name for entry in registered_models()} & set(EXPECTED)
    assert builtins == set(EXPECTED), "a builtin model disappeared from the registry"
    missing = [name for name in EXPECTED if resolve_model("", model_type=name).capabilities is None]
    assert missing == []


@pytest.mark.parametrize("name,expected", EXPECTED.items())
def test_declared_capabilities_match_the_documented_matrix(name, expected):
    heads, distilled, supports_cp, supports_sp = expected
    caps = _capabilities(name)
    assert caps.num_attention_heads == heads
    assert caps.is_distilled is distilled
    assert caps.supports_cp is supports_cp
    assert caps.supports_sp is supports_sp


@pytest.mark.parametrize("name", EXPECTED)
def test_cfg_parallel_is_the_complement_of_distillation(name):
    caps = _capabilities(name)
    assert caps.supports_cfg_parallel is not caps.is_distilled


@pytest.mark.parametrize("name", EXPECTED)
def test_cp_modes_track_cp_support(name):
    caps = _capabilities(name)
    assert bool(caps.cp_modes) is caps.supports_cp
    if caps.supports_cp:
        assert caps.cp_modes == frozenset(CP_MODES)


def test_head_counts_permit_the_four_core_configs():
    """Every tp/cp split verify_cli runs on four cores must divide the heads.

    tp2cp2ulysses is the tight one: ulysses shards heads over cp on top of the
    TP head shard, so it needs heads % (tp * cp) == 0, i.e. heads % 4 here.
    """

    for name, (heads, _, supports_cp, _) in EXPECTED.items():
        assert heads % 4 == 0, f"{name}: tp=4 does not divide {heads} heads"
        assert heads % 2 == 0, f"{name}: tp=2 does not divide {heads} heads"
        if supports_cp:
            assert heads % (2 * 2) == 0, f"{name}: tp2cp2 ulysses needs heads % 4 == 0"


# ------------------------------------------------------------- invariants


def test_rejects_cp_modes_without_cp_support():
    with pytest.raises(ValueError, match="cp_modes must be empty"):
        ModelCapabilities(
            num_attention_heads=24, is_distilled=True,
            supports_cp=False, supports_sp=False,
        )


def test_rejects_cp_support_without_any_mode():
    with pytest.raises(ValueError, match="at least one cp_mode"):
        ModelCapabilities(
            num_attention_heads=24, is_distilled=True,
            supports_cp=True, supports_sp=False, cp_modes=frozenset(),
        )


def test_rejects_unknown_cp_mode():
    with pytest.raises(ValueError, match="unknown cp_modes"):
        ModelCapabilities(
            num_attention_heads=24, is_distilled=True,
            supports_cp=True, supports_sp=False, cp_modes=frozenset({"pipefusion"}),
        )


def test_rejects_zero_heads():
    with pytest.raises(ValueError, match="num_attention_heads"):
        ModelCapabilities(
            num_attention_heads=0, is_distilled=True,
            supports_cp=False, supports_sp=False, cp_modes=frozenset(),
        )


# ------------------------------------------------------ consumers stay in sync


def test_mode_classification_derives_from_capabilities():
    from difflet.cli.modes import model_class

    assert model_class("black-forest-labs/FLUX.1-dev") == "distilled"
    assert model_class("Wan-AI/Wan2.2-T2V-A14B-Diffusers") == "true_cfg"
    assert model_class("Lightricks/LTX-2") == "true_cfg_no_cp"


def test_cli_sp_validation_reads_the_registry():
    import argparse

    from difflet.cli.main import _validate_sp

    args = argparse.Namespace(
        sp_enabled=True, cp_degree=1, model_id="Lightricks/LTX-2",
    )
    with pytest.raises(SystemExit):
        _validate_sp(args)

    args.model_id = "black-forest-labs/FLUX.1-dev"
    _validate_sp(args)  # supported: must not raise

    args.model_id = "Qwen/Qwen-Image"
    _validate_sp(args)  # supported since the modeling_qwen SP fork landed


def test_cli_cfg_parallel_validation_reads_the_registry():
    import argparse

    from difflet.cli.main import _validate_cfg_parallel

    args = argparse.Namespace(
        cfg_parallel=True, cp_degree=1, model_id="black-forest-labs/FLUX.1-dev",
    )
    with pytest.raises(SystemExit):
        _validate_cfg_parallel(args)

    args.model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    _validate_cfg_parallel(args)  # true-CFG: must not raise
