"""Staged compiled-artifact dirs must be keyed on cp_mode.

The staged dirs are named from tp/cp/cfg/sp + shape. They did NOT carry cp_mode, so
two compiles differing only in --cp-mode landed in the same ~/.cache/difflet dir and
silently reused each other's artifact — even though their compile-cache hashes differ.
That was already latently true for ring; a third mode makes it acute.

These tests pin both halves of the fix: distinct modes get distinct dirs, and the
gather_kv default keeps its historical name so existing caches stay valid.
"""

import argparse

import pytest

from difflet.cli.orchestrators.base import cp_mode_token
from difflet.cli.orchestrators.hunyuan_video import HunyuanVideoOrchestrator
from difflet.cli.orchestrators.qwen_image import QwenImageOrchestrator
from difflet.cli.orchestrators.wan import WanOrchestrator
from difflet.pipeline.parallel_config import CP_MODES


def _args(**kw):
    base = dict(
        cache_dir="/tmp/cache", tp_degree=2, cp_degree=2, cp_mode="gather_kv",
        cfg_parallel=False, sp_enabled=False,
        height=None, width=None, num_frames=None, model_id=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


# (orchestrator, model_id, the CP-bearing stage)
CASES = [
    (WanOrchestrator, "Wan-AI/Wan2.2-T2V-A14B-Diffusers", "transformer"),
    (QwenImageOrchestrator, "Qwen/Qwen-Image", "generate"),
    (HunyuanVideoOrchestrator, "hunyuanvideo-community/HunyuanVideo", "generate"),
]


def test_cp_mode_token_is_empty_only_for_the_default():
    assert cp_mode_token(_args(cp_mode="gather_kv")) == ""
    assert cp_mode_token(_args(cp_mode="ring")) == "ring"
    assert cp_mode_token(_args(cp_mode="ulysses")) == "ulysses"


@pytest.mark.parametrize("orch_cls,model_id,stage", CASES)
def test_each_cp_mode_gets_a_distinct_staged_dir(orch_cls, model_id, stage):
    orch = orch_cls(_args(model_id=model_id))
    dirs = {
        mode: orch._stage_compiled_dir(stage, _args(model_id=model_id, cp_mode=mode))
        for mode in CP_MODES
    }
    assert len(set(dirs.values())) == len(CP_MODES), (
        f"cp_modes collide in the staged cache: {dirs}"
    )


@pytest.mark.parametrize("orch_cls,model_id,stage", CASES)
def test_gather_kv_staged_dir_name_is_unchanged(orch_cls, model_id, stage):
    # Back-compat: the default must not gain a token, or every existing compiled
    # artifact in ~/.cache/difflet would be orphaned.
    orch = orch_cls(_args(model_id=model_id))
    got = orch._stage_compiled_dir(stage, _args(model_id=model_id, cp_mode="gather_kv"))
    assert "gather_kv" not in got.name
    for mode in ("ring", "ulysses"):
        assert mode not in got.name
