from __future__ import annotations

import argparse

from difflet.cli.orchestrators.hunyuan_video_15 import HunyuanVideo15Orchestrator


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model_id="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        tp_degree=None, cp_degree=None, cp_mode="gather_kv",
        height=None, width=None, num_frames=None,
        cache_dir=None, prompt=None, output=None,
        steps=None, guidance_scale=None, seed=42,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_shared_cli_args_defaults():
    parts = HunyuanVideo15Orchestrator(_args())._shared_cli_args("compile")
    assert "--model-id" in parts
    assert "--tp-degree" in parts
    assert parts[parts.index("--height") + 1] == "480"
    assert parts[parts.index("--width") + 1] == "848"
    assert parts[parts.index("--num-frames") + 1] == "121"
    assert parts[parts.index("--steps") + 1] == "4"
    assert parts[parts.index("--guidance-scale") + 1] == "6.0"
    assert parts[parts.index("--stage-mode") + 1] == "compile"
    # Optional flags absent when their args are None.
    assert "--prompt" not in parts
    assert "--output" not in parts
    assert "--cache-dir" not in parts
    assert "--work-dir" not in parts


def test_shared_cli_args_with_all_optionals():
    args = _args(
        tp_degree=8, cp_degree=2, cp_mode="ring",
        height=720, width=1280, num_frames=61,
        cache_dir="/tmp/cache", prompt="a cat", output="/tmp/o.mp4",
        steps=10, guidance_scale=7.5, seed=1,
    )
    parts = HunyuanVideo15Orchestrator(args)._shared_cli_args("generate", work_dir="/tmp/wd")
    assert parts[parts.index("--tp-degree") + 1] == "8"
    assert parts[parts.index("--cp-degree") + 1] == "2"
    assert parts[parts.index("--cp-mode") + 1] == "ring"
    assert parts[parts.index("--prompt") + 1] == "a cat"
    assert parts[parts.index("--output") + 1] == "/tmp/o.mp4"
    assert parts[parts.index("--cache-dir") + 1] == "/tmp/cache"
    assert parts[parts.index("--work-dir") + 1] == "/tmp/wd"
    assert parts[parts.index("--stage-mode") + 1] == "generate"
