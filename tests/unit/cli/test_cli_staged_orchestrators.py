from __future__ import annotations
import argparse
import pytest


# ---------------------------------------------------------------- Wan helpers

def _wan_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers", tp_degree=4, cp_degree=1,
        height=480, width=832, num_frames=9,
        cache_dir=None, force=False, revision=None,
        prompt="a cat walking", output="/tmp/wan.mp4",
        steps=2, guidance_scale=1.0, seed=42,
        work_dir=None, keep_work_dir=False,
        teacache_cadence=None, teacache_online_delta=None,
        teacache_speedup=None, teacache_calibration=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestWanOrchestrator:

    def test_generate_spawns_transformer_then_vae(self, monkeypatch, tmp_path):
        from difflet.cli.orchestrators.wan import WanOrchestrator
        stages = []
        monkeypatch.setattr(
            "difflet.cli.runner.run_stage",
            lambda orch, stage, **kw: stages.append(stage),
        )
        args = _wan_args(work_dir=str(tmp_path))
        (tmp_path).mkdir(parents=True, exist_ok=True)
        WanOrchestrator(args).generate()
        assert stages == ["transformer", "vae"]

    def test_compile_spawns_transformer_then_vae_in_compile_mode(self, monkeypatch):
        from difflet.cli.orchestrators.wan import WanOrchestrator
        calls = []
        monkeypatch.setattr(
            "difflet.cli.runner.run_stage",
            lambda orch, stage, **kw: calls.append((stage, kw.get("cli_args", []))),
        )
        WanOrchestrator(_wan_args()).compile()
        assert [s for s, _ in calls] == ["transformer", "vae"]
        for stage, cli_args in calls:
            assert "--stage-mode" in cli_args
            idx = cli_args.index("--stage-mode")
            assert cli_args[idx + 1] == "compile"

    def test_transformer_stage_uses_tp_times_cp_cores(self, monkeypatch):
        from difflet.cli.orchestrators.wan import WanOrchestrator
        core_counts = {}
        monkeypatch.setattr(
            "difflet.cli.runner.run_stage",
            lambda orch, stage, num_cores, **kw: core_counts.update({stage: num_cores}),
        )
        WanOrchestrator(_wan_args(tp_degree=4, cp_degree=2)).generate()
        assert core_counts["transformer"] == 8

    def test_vae_stage_uses_one_core(self, monkeypatch):
        from difflet.cli.orchestrators.wan import WanOrchestrator
        core_counts = {}
        monkeypatch.setattr(
            "difflet.cli.runner.run_stage",
            lambda orch, stage, num_cores, **kw: core_counts.update({stage: num_cores}),
        )
        WanOrchestrator(_wan_args()).generate()
        assert core_counts["vae"] == 1

    def test_wan_does_not_set_virtual_core_size(self, monkeypatch):
        from difflet.cli.orchestrators.wan import WanOrchestrator
        vcs_values = {}
        monkeypatch.setattr(
            "difflet.cli.runner.run_stage",
            lambda orch, stage, virtual_core_size, **kw: vcs_values.update({stage: virtual_core_size}),
        )
        WanOrchestrator(_wan_args()).generate()
        assert all(v is None for v in vcs_values.values())


# ---------------------------------------------------------------- HunyuanVideo helpers

def _hv_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model_id="hunyuanvideo-community/HunyuanVideo", tp_degree=4, cp_degree=1,
        height=320, width=512, num_frames=61,
        cache_dir=None, force=False, revision=None,
        prompt="a cat", output="/tmp/hv.mp4",
        steps=4, guidance_scale=6.0, seed=42,
        work_dir=None, keep_work_dir=False,
        teacache_cadence=None, teacache_online_delta=None,
        teacache_speedup=None, teacache_calibration=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestHunyuanVideoOrchestrator:

    def test_generate_spawns_clip_llama_generate_in_order(self, monkeypatch, tmp_path):
        from difflet.cli.orchestrators.hunyuan_video import HunyuanVideoOrchestrator
        stages = []
        monkeypatch.setattr(
            "difflet.cli.runner.run_stage",
            lambda orch, stage, **kw: stages.append(stage),
        )
        HunyuanVideoOrchestrator(_hv_args(work_dir=str(tmp_path))).generate()
        assert stages == ["clip", "llama", "generate"]

    def test_compile_spawns_all_three_stages_in_compile_mode(self, monkeypatch):
        from difflet.cli.orchestrators.hunyuan_video import HunyuanVideoOrchestrator
        calls = []
        monkeypatch.setattr(
            "difflet.cli.runner.run_stage",
            lambda orch, stage, **kw: calls.append((stage, kw.get("cli_args", []))),
        )
        HunyuanVideoOrchestrator(_hv_args()).compile()
        assert [s for s, _ in calls] == ["clip", "llama", "generate"]
        for _, cli_args in calls:
            idx = cli_args.index("--stage-mode")
            assert cli_args[idx + 1] == "compile"

    def test_clip_stage_uses_one_core(self, monkeypatch):
        from difflet.cli.orchestrators.hunyuan_video import HunyuanVideoOrchestrator
        cores = {}
        monkeypatch.setattr(
            "difflet.cli.runner.run_stage",
            lambda orch, stage, num_cores, **kw: cores.update({stage: num_cores}),
        )
        HunyuanVideoOrchestrator(_hv_args()).generate()
        assert cores["clip"] == 1

    def test_llama_and_generate_stages_use_tp_times_cp(self, monkeypatch):
        from difflet.cli.orchestrators.hunyuan_video import HunyuanVideoOrchestrator
        cores = {}
        monkeypatch.setattr(
            "difflet.cli.runner.run_stage",
            lambda orch, stage, num_cores, **kw: cores.update({stage: num_cores}),
        )
        HunyuanVideoOrchestrator(_hv_args(tp_degree=4, cp_degree=2)).generate()
        assert cores["llama"] == 8
        assert cores["generate"] == 8

    def test_all_stages_use_virtual_core_size_2(self, monkeypatch):
        from difflet.cli.orchestrators.hunyuan_video import HunyuanVideoOrchestrator
        vcs = {}
        monkeypatch.setattr(
            "difflet.cli.runner.run_stage",
            lambda orch, stage, virtual_core_size, **kw: vcs.update({stage: virtual_core_size}),
        )
        HunyuanVideoOrchestrator(_hv_args()).generate()
        assert all(v == 2 for v in vcs.values())


# ---------------------------------------------------------------- QwenImage helpers

def _qwen_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model_id="Qwen/Qwen-Image", tp_degree=4, cp_degree=1,
        height=1024, width=1024, num_frames=None,
        cache_dir=None, force=False, revision=None,
        prompt="a cat", output="/tmp/qwen.png",
        steps=4, guidance_scale=4.0, seed=42,
        work_dir=None, keep_work_dir=False,
        teacache_cadence=None, teacache_online_delta=None,
        teacache_speedup=None, teacache_calibration=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestQwenImageOrchestrator:

    def test_generate_spawns_text_generate_vae_in_order(self, monkeypatch, tmp_path):
        from difflet.cli.orchestrators.qwen_image import QwenImageOrchestrator
        stages = []
        monkeypatch.setattr(
            "difflet.cli.runner.run_stage",
            lambda orch, stage, **kw: stages.append(stage),
        )
        QwenImageOrchestrator(_qwen_args(work_dir=str(tmp_path))).generate()
        assert stages == ["text", "generate", "vae"]

    def test_vae_stage_uses_one_core(self, monkeypatch):
        from difflet.cli.orchestrators.qwen_image import QwenImageOrchestrator
        cores = {}
        monkeypatch.setattr(
            "difflet.cli.runner.run_stage",
            lambda orch, stage, num_cores, **kw: cores.update({stage: num_cores}),
        )
        QwenImageOrchestrator(_qwen_args()).generate()
        assert cores["vae"] == 1

    def test_text_and_generate_stages_use_tp_times_cp(self, monkeypatch):
        from difflet.cli.orchestrators.qwen_image import QwenImageOrchestrator
        cores = {}
        monkeypatch.setattr(
            "difflet.cli.runner.run_stage",
            lambda orch, stage, num_cores, **kw: cores.update({stage: num_cores}),
        )
        QwenImageOrchestrator(_qwen_args(tp_degree=4, cp_degree=2)).generate()
        assert cores["text"] == 8
        assert cores["generate"] == 8

    def test_all_stages_use_virtual_core_size_2(self, monkeypatch):
        from difflet.cli.orchestrators.qwen_image import QwenImageOrchestrator
        vcs = {}
        monkeypatch.setattr(
            "difflet.cli.runner.run_stage",
            lambda orch, stage, virtual_core_size, **kw: vcs.update({stage: virtual_core_size}),
        )
        QwenImageOrchestrator(_qwen_args()).generate()
        assert all(v == 2 for v in vcs.values())

    def test_compile_spawns_all_three_in_compile_mode(self, monkeypatch):
        from difflet.cli.orchestrators.qwen_image import QwenImageOrchestrator
        calls = []
        monkeypatch.setattr(
            "difflet.cli.runner.run_stage",
            lambda orch, stage, **kw: calls.append((stage, kw.get("cli_args", []))),
        )
        QwenImageOrchestrator(_qwen_args()).compile()
        assert [s for s, _ in calls] == ["text", "generate", "vae"]
        for _, cli_args in calls:
            idx = cli_args.index("--stage-mode")
            assert cli_args[idx + 1] == "compile"
