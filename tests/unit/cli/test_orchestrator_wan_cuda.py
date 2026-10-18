"""Wan 2.2 on CUDA through the CLI: one process, DiffletPipeline + generate().

CPU-only: DiffletPipeline.from_pretrained is replaced by a recorder; no weights, no GPU.
Mirrors tests/unit/cli/test_orchestrator_flux_cuda.py; the differences are the video output
(``OUTPUT_TYPE = "pt"``, MP4 through the Trainium orchestrator's ``_save_video``, ``.pt``
fallback), ``--num-frames`` and the 480x832x9 default shape.
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from difflet.cli.orchestrators import wan as trainium_wan
from difflet.cli.orchestrators import wan_cuda as mod
from difflet.cli.orchestrators.qwen_image_cuda import CudaOneProcessOrchestrator

WAN22 = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
WAN21 = "Wan-AI/Wan2.1-T2V-14B-Diffusers"


def _args(**over):
    d = dict(
        model_id=WAN22,
        tp_degree=None,
        cp_degree=None,
        cp_mode="gather_kv",
        cfg_parallel=False,
        sp_enabled=False,
        dp=None,
        height=None,
        width=None,
        num_frames=None,
        shapes=None,
        cache_dir=None,
        force=False,
        revision=None,
        prompt="a cat",
        output="/tmp/x/wan.mp4",
        steps=None,
        guidance_scale=None,
        seed=42,
        requests=None,
        requests_dir=None,
        teacache_cadence=None,
        teacache_online_delta=None,
        teacache_speedup=None,
        teacache_calibration=None,
        host_vae=False,
        taef1=False,
        taef1_path=None,
        torch_compile=False,
    )
    d.update(over)
    return argparse.Namespace(**d)


def _video(frames: int = 9) -> torch.Tensor:
    """The application's ``output_type="pt"`` result: (B, 3, F, H, W) in [-1, 1]."""
    torch.manual_seed(0)
    return torch.rand(1, 3, frames, 16, 16) * 2 - 1


class _App:
    def __init__(self):
        self.calls = []
        self.video = _video()

    def generate(self, prompt, **kw):
        self.calls.append((prompt, kw))
        return self.video


class _Pipe:
    def __init__(self):
        self.app = _App()
        self.compiled_path = "/fake/compiled"


@pytest.fixture
def recorder(monkeypatch):
    calls = []

    def fake(model_id, **kw):
        calls.append((model_id, kw))
        return _Pipe()

    monkeypatch.setattr(
        "difflet.pipeline.difflet_pipeline.DiffletPipeline.from_pretrained", staticmethod(fake)
    )
    return calls


@pytest.fixture
def saved(monkeypatch):
    """Record _save_video calls; write a marker file so the output exists."""
    calls = []

    def fake(tensor, output_path):
        calls.append((tensor, output_path))
        Path(output_path).write_bytes(b"mp4")
        return True

    monkeypatch.setattr(mod, "_save_video", fake)
    return calls


def test_wan_orchestrator_is_the_shared_one_process_orchestrator():
    assert issubclass(mod.WanCudaOrchestrator, CudaOneProcessOrchestrator)
    assert mod.WanCudaOrchestrator.HF_MODEL_ID == WAN22
    assert set(mod.WanCudaOrchestrator.MODEL_IDS) == {WAN22, WAN21}
    assert mod.WanCudaOrchestrator.DEFAULT_STEPS == 40
    assert mod.WanCudaOrchestrator.ACCEPTS_GUIDANCE_SCALE is True
    assert mod.WanCudaOrchestrator.DEFAULT_GUIDANCE_SCALE == 4.0
    assert mod.WanCudaOrchestrator.OUTPUT_TYPE == "pt"
    # The image models keep PIL output.
    assert CudaOneProcessOrchestrator.OUTPUT_TYPE == "pil"


def test_the_mp4_writer_is_the_trainium_orchestrator_s_helper():
    # Reused by import, not copied.
    assert mod._save_video is trainium_wan._save_video


def test_compile_builds_the_cuda_pipeline_at_480x832x9_without_loading(recorder):
    mod.WanCudaOrchestrator(_args()).compile()
    model_id, kw = recorder[0]
    assert model_id == WAN22
    assert kw["backend"] == "cuda" and kw["load"] is False
    assert kw["parallel"].tp_degree == 1 and kw["dtype"] == "bf16"
    assert kw["local_files_only"] is True
    assert (kw["height"], kw["width"], kw["num_frames"]) == (480, 832, 9)
    assert "application_kwargs" not in kw


def test_height_width_num_frames_cache_dir_and_force_reach_from_pretrained(recorder):
    args = _args(cache_dir="/c", force=True, height=320, width=512, num_frames=17)
    mod.WanCudaOrchestrator(args).compile()
    _, kw = recorder[0]
    assert kw["compile_cache_dir"] == "/c" and kw["force_compile"] is True
    assert (kw["height"], kw["width"], kw["num_frames"]) == (320, 512, 17)


def test_generate_writes_the_mp4_through_save_video(recorder, saved, tmp_path):
    out = tmp_path / "deep" / "wan.mp4"
    orch = mod.WanCudaOrchestrator(_args(output=str(out)))
    orch.generate()
    _, kw = recorder[0]
    assert kw["load"] is True and kw["force_compile"] is False
    # The whole (B, 3, F, H, W) tensor goes to the helper, which takes frames[0] itself.
    ((tensor, path),) = saved
    assert tensor is orch._last_pipe.app.video and path == str(out)
    assert out.is_file() and not out.with_suffix(".pt").exists()


def test_a_failed_mp4_export_falls_back_to_a_pt_tensor(recorder, monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "_save_video", lambda tensor, path: False)
    out = tmp_path / "wan.mp4"
    orch = mod.WanCudaOrchestrator(_args(output=str(out)))
    orch.generate()
    pt = out.with_suffix(".pt")
    assert pt.is_file() and not out.exists()
    assert torch.equal(torch.load(pt), orch._last_pipe.app.video)


def test_a_non_mp4_output_is_saved_as_the_tensor(recorder, saved, tmp_path):
    out = tmp_path / "wan.pt"
    orch = mod.WanCudaOrchestrator(_args(output=str(out)))
    orch.generate()
    assert saved == [] and torch.equal(torch.load(out), orch._last_pipe.app.video)


def test_generate_writes_a_real_mp4(recorder, tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    out = tmp_path / "wan.mp4"
    mod.WanCudaOrchestrator(_args(output=str(out))).generate()
    assert out.is_file() and out.stat().st_size > 0 and not out.with_suffix(".pt").exists()


def test_steps_default_to_40_guidance_to_4_and_frames_to_9(recorder, saved, tmp_path):
    orch = mod.WanCudaOrchestrator(_args(output=str(tmp_path / "w.mp4")))
    orch.generate()
    prompt, kw = orch._last_pipe.app.calls[0]
    assert prompt == "a cat" and kw["seed"] == 42 and kw["output_type"] == "pt"
    assert kw["num_inference_steps"] == mod.DEFAULT_STEPS == 40
    assert kw["guidance_scale"] == mod.DEFAULT_GUIDANCE_SCALE == 4.0
    assert kw["num_frames"] == 9
    # No --negative-prompt flag: the application's default (None → "") applies.
    assert "negative_prompt" not in kw


def test_steps_seed_guidance_and_frames_are_forwarded(recorder, saved, tmp_path):
    orch = mod.WanCudaOrchestrator(
        _args(output=str(tmp_path / "w.mp4"), steps=7, seed=3, guidance_scale=5.0, num_frames=17)
    )
    orch.generate()
    _, kw = orch._last_pipe.app.calls[0]
    assert kw["num_inference_steps"] == 7 and kw["seed"] == 3 and kw["guidance_scale"] == 5.0
    assert kw["num_frames"] == 17
    assert recorder[0][1]["num_frames"] == 17


def test_wan21_is_built_from_its_own_repo(recorder, monkeypatch):
    args = _args(model_id=WAN21)
    mod.WanCudaOrchestrator(args).compile()
    assert recorder[0][0] == WAN21
    seen = {}
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda mid, **kw: seen.update(mid=mid) or "/snap",
    )
    mod.WanCudaOrchestrator(args).download()
    assert seen["mid"] == WAN21


@pytest.mark.parametrize(
    "over,expected",
    [
        ({"tp_degree": 2}, "not supported"),
        ({"sp_enabled": True}, "not supported"),
        ({"cp_degree": 2}, "not supported"),
        ({"shapes": "480x832x9,480x832x17"}, "not supported"),
        ({"teacache_cadence": 2}, "not supported"),
        ({"teacache_speedup": 1.5}, "not supported"),
        ({"teacache_online_delta": 0.1}, "not supported"),
        ({"teacache_calibration": "/x.json"}, "not supported"),
        ({"taef1": True}, "not supported"),
        ({"taef1_path": "madebyollin/taef1"}, "not supported"),
        ({"host_vae": True}, "not supported"),
        ({"torch_compile": True}, "not supported"),
    ],
)
def test_unsupported_flags_are_refused_before_any_load(recorder, over, expected):
    with pytest.raises(SystemExit) as exc:
        mod.WanCudaOrchestrator(_args(**over)).generate()
    assert expected in str(exc.value) and "[wan/cuda]" in str(exc.value) and recorder == []


@pytest.mark.parametrize(
    "over,flag",
    [
        ({"shapes": "480x832x9"}, "--shapes"),
        ({"teacache_cadence": 2}, "--teacache-cadence"),
        ({"teacache_calibration": "/x.json"}, "--teacache-calibration"),
        ({"taef1_path": "madebyollin/taef1"}, "--taef1"),
        ({"host_vae": True}, "--host-vae"),
        ({"torch_compile": True}, "--torch-compile"),
    ],
)
def test_refusals_name_the_flag(recorder, over, flag):
    with pytest.raises(SystemExit) as exc:
        mod.WanCudaOrchestrator(_args(**over)).compile()
    assert flag in str(exc.value) and recorder == []


def test_download_resolves_the_hub_snapshot(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda mid, **kw: seen.update(mid=mid, **kw) or "/snap",
    )
    mod.WanCudaOrchestrator(_args()).download()
    assert seen["mid"] == WAN22 and seen["local_files_only"] is False


def test_run_is_download_compile_generate(recorder, monkeypatch, tmp_path):
    order = []
    orch = mod.WanCudaOrchestrator(_args(output=str(tmp_path / "w.mp4")))
    monkeypatch.setattr(orch, "download", lambda: order.append("download"))
    monkeypatch.setattr(orch, "compile", lambda: order.append("compile"))
    monkeypatch.setattr(orch, "generate", lambda: order.append("generate"))
    orch.run()
    assert order == ["download", "compile", "generate"]


def test_run_refuses_unsupported_flags_before_any_download(recorder, monkeypatch):
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda *a, **kw: pytest.fail("must refuse before resolving the snapshot"),
    )
    with pytest.raises(SystemExit) as exc:
        mod.WanCudaOrchestrator(_args(host_vae=True)).run()
    assert "not supported" in str(exc.value) and recorder == []


def test_lone_teacache_calibration_is_refused_by_name(recorder):
    """``main._validate_teacache`` lets a lone ``--teacache-calibration`` through and Wan on
    CUDA forwards nothing, so without this refusal the calibration file is silently dropped."""
    with pytest.raises(SystemExit) as exc:
        mod.WanCudaOrchestrator(_args(teacache_calibration="/x.json")).generate()
    assert "--teacache-calibration" in str(exc.value) and "not supported" in str(exc.value)
    assert recorder == []


# ------------------------------------------------- CFG-parallel under torchrun

_USER_ARGV = [
    "/venv/bin/difflet",
    "run",
    "--model-id",
    WAN22,
    "--cfg-parallel",
    "--prompt",
    "a cat",
    "--output",
    "out/wan.mp4",
]


@pytest.fixture
def launches(monkeypatch):
    """Record ``run_torchrun`` calls instead of launching; the return code is settable."""
    calls = []
    state = {"code": 0}

    def fake(world_size, argv, env=None):
        calls.append((world_size, argv, env))
        return state["code"]

    monkeypatch.setattr("difflet.cli.runner.run_torchrun", fake)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    return calls, state


def test_wan_accepts_cfg_parallel():
    assert mod.WanCudaOrchestrator.ACCEPTS_CFG_PARALLEL is True
    assert CudaOneProcessOrchestrator.ACCEPTS_CFG_PARALLEL is False


@pytest.mark.parametrize(
    "over,world_size",
    [({}, 1), ({"cfg_parallel": True}, 2), ({"cfg_parallel": True, "tp_degree": 1}, 2)],
)
def test_world_size_is_cfg_times_cp_times_tp(over, world_size):
    assert mod.WanCudaOrchestrator(_args(**over)).world_size == world_size


def test_world_size_counts_cp_and_tp_degrees():
    # cp > 1 and tp > 1 are refused today; the product is still the launch size.
    assert mod.WanCudaOrchestrator(_args(cp_degree=2)).world_size == 2
    assert mod.WanCudaOrchestrator(_args(tp_degree=2, cfg_parallel=True)).world_size == 4


def test_compile_with_cfg_parallel_builds_the_cfg_parallel_config(recorder, launches):
    mod.WanCudaOrchestrator(_args(cfg_parallel=True)).compile()
    _, kw = recorder[0]
    assert kw["parallel"].cfg_parallel_enabled is True and kw["parallel"].tp_degree == 1
    assert kw["load"] is False
    assert launches[0] == []  # compile never re-execs


def test_cfg_parallel_off_keeps_the_single_branch_config(recorder):
    mod.WanCudaOrchestrator(_args()).compile()
    assert recorder[0][1]["parallel"].cfg_parallel_enabled is False


def test_generate_with_cfg_parallel_re_execs_under_torchrun(recorder, saved, launches, tmp_path):
    calls, _ = launches
    orch = mod.WanCudaOrchestrator(_args(cfg_parallel=True, output=str(tmp_path / "w.mp4")))
    user = [*_USER_ARGV[:1], "generate", *_USER_ARGV[2:]]
    orch.generate(argv=user)
    ((world_size, argv, _env),) = calls
    assert world_size == 2
    # The user's own arguments after the command word, verbatim.
    assert argv == ["-m", "difflet.cli.main", "generate", *_USER_ARGV[2:]]
    # The parent loads nothing and writes nothing: the ranks do the work.
    assert recorder == [] and saved == []


def test_a_run_becomes_generate_in_the_ranks(recorder, saved, launches):
    calls, _ = launches
    mod.WanCudaOrchestrator(_args(cfg_parallel=True)).generate(argv=list(_USER_ARGV))
    ((_, argv, _),) = calls
    assert argv[:3] == ["-m", "difflet.cli.main", "generate"] and "run" not in argv
    assert argv[3:] == _USER_ARGV[2:]


def test_generate_reads_sys_argv_by_default(recorder, saved, launches, monkeypatch):
    calls, _ = launches
    monkeypatch.setattr(sys, "argv", list(_USER_ARGV))
    mod.WanCudaOrchestrator(_args(cfg_parallel=True)).generate()
    assert calls[0][1] == ["-m", "difflet.cli.main", "generate", *_USER_ARGV[2:]]


def test_a_failed_launch_exits_with_the_child_return_code(recorder, saved, launches):
    _, state = launches
    state["code"] = 7
    with pytest.raises(SystemExit) as exc:
        mod.WanCudaOrchestrator(_args(cfg_parallel=True)).generate(argv=list(_USER_ARGV))
    assert exc.value.code == 7


def test_run_downloads_and_compiles_in_the_parent_then_launches_generate(
    recorder, saved, launches, monkeypatch
):
    calls, _ = launches
    pulled = []
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda mid, **kw: pulled.append(mid) or "/snap",
    )
    monkeypatch.setattr(sys, "argv", list(_USER_ARGV))
    mod.WanCudaOrchestrator(_args(cfg_parallel=True)).run()
    assert pulled == [WAN22]
    # One parent-side build (compile, load=False), then the launch; no parent-side load.
    assert [kw["load"] for _, kw in recorder] == [False]
    assert recorder[0][1]["parallel"].cfg_parallel_enabled is True
    assert [(ws, argv[2]) for ws, argv, _ in calls] == [(2, "generate")]


def test_one_process_generate_never_launches(recorder, saved, launches, tmp_path):
    calls, _ = launches
    mod.WanCudaOrchestrator(_args(output=str(tmp_path / "w.mp4"))).generate(argv=list(_USER_ARGV))
    assert calls == [] and [kw["load"] for _, kw in recorder] == [True] and len(saved) == 1


def test_a_refused_flag_is_refused_before_any_launch(recorder, launches):
    calls, _ = launches
    with pytest.raises(SystemExit) as exc:
        mod.WanCudaOrchestrator(_args(cfg_parallel=True, host_vae=True)).generate(argv=[])
    assert "not supported" in str(exc.value) and calls == [] and recorder == []


@pytest.mark.parametrize("rank", ["0", "1"])
def test_inside_a_rank_generate_runs_in_process(
    recorder, saved, launches, monkeypatch, tmp_path, capsys, rank
):
    calls, _ = launches
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", rank)
    out = tmp_path / "deep" / "w.mp4"
    orch = mod.WanCudaOrchestrator(_args(cfg_parallel=True, output=str(out)))
    orch.generate(argv=list(_USER_ARGV))
    # SPMD: every rank loads the cfg-parallel pipeline and runs the full generate.
    assert calls == []
    ((_, kw),) = recorder
    assert kw["load"] is True and kw["parallel"].cfg_parallel_enabled is True
    assert len(orch._last_pipe.app.calls) == 1
    printed = capsys.readouterr().out
    if rank == "0":
        assert [path for _, path in saved] == [str(out)] and out.is_file()
        assert "[difflet] wrote" in printed
    else:
        # Only rank 0 writes: rank 1 touches neither the output nor its directory.
        assert saved == [] and not out.exists() and not out.parent.exists()
        assert "wrote" not in printed


def test_is_rank_zero_reads_rank(monkeypatch):
    orch = mod.WanCudaOrchestrator(_args())
    monkeypatch.delenv("RANK", raising=False)
    assert orch.is_rank_zero is True
    monkeypatch.setenv("RANK", "0")
    assert orch.is_rank_zero is True
    monkeypatch.setenv("RANK", "1")
    assert orch.is_rank_zero is False


# ------------------------------------------------- batch mode: a dp worker


def _manifest(tmp_path, specs):
    from difflet.cli.dp.requests_io import RequestSpec, write_manifest

    requests_dir = tmp_path / "requests"
    reqs = [
        RequestSpec(index=i, output=str(tmp_path / "out" / f"r{i}.mp4"), **spec)
        for i, spec in enumerate(specs)
    ]
    write_manifest(reqs, requests_dir)
    return requests_dir


def _batch_args(requests_dir, **over):
    batch = dict(
        prompt=None,
        output=None,
        requests_dir=str(requests_dir),
        worker_index=0,
        dp_schedule="least_loaded",
        work_dir=str(requests_dir.parent / "worker_0"),
    )
    return _args(**{**batch, **over})


def test_batch_mode_forwards_each_request_s_overrides(recorder, saved, tmp_path):
    from difflet.cli.dp.claims import summarize

    requests_dir = _manifest(
        tmp_path,
        [
            dict(prompt="p0", seed=0),
            dict(prompt="p1", seed=1, steps=3, guidance_scale=2.5, negative_prompt="blurry"),
        ],
    )
    orch = mod.WanCudaOrchestrator(_batch_args(requests_dir, num_frames=5))
    orch.generate(argv=[])
    assert [kw["load"] for _, kw in recorder] == [True]
    (p0, kw0), (p1, kw1) = orch._last_pipe.app.calls
    assert (p0, kw0["seed"], kw0["num_inference_steps"], kw0["guidance_scale"]) == (
        "p0",
        0,
        mod.DEFAULT_STEPS,
        mod.DEFAULT_GUIDANCE_SCALE,
    )
    assert "negative_prompt" not in kw0 and kw0["num_frames"] == 5
    assert (p1, kw1["seed"], kw1["num_inference_steps"], kw1["guidance_scale"]) == (
        "p1",
        1,
        3,
        2.5,
    )
    assert kw1["negative_prompt"] == "blurry" and kw1["num_frames"] == 5
    assert [path for _, path in saved] == [str(tmp_path / "out" / f"r{i}.mp4") for i in (0, 1)]
    assert summarize(requests_dir).done == [0, 1]


def test_batch_mode_a_failing_request_is_marked_failed_and_the_loop_continues(
    recorder, saved, monkeypatch, tmp_path
):
    from difflet.cli.dp.claims import summarize

    def generate(self, prompt, **kw):
        self.calls.append((prompt, kw))
        if prompt == "boom":
            raise RuntimeError("generation exploded")
        return self.video

    monkeypatch.setattr(_App, "generate", generate)
    requests_dir = _manifest(tmp_path, [dict(prompt="boom"), dict(prompt="p1"), dict(prompt="p2")])
    orch = mod.WanCudaOrchestrator(_batch_args(requests_dir))
    orch.generate(argv=[])
    assert [p for p, _ in orch._last_pipe.app.calls] == ["boom", "p1", "p2"]
    s = summarize(requests_dir)
    assert s.done == [1, 2] and list(s.failed) == [0] and "generation exploded" in s.failed[0]
    assert len(saved) == 2


def test_a_cfg_parallel_worker_re_execs_torchrun_with_its_worker_flags(
    recorder, saved, launches, tmp_path
):
    # A dp replica with --cfg-parallel is itself a two-rank torchrun world: the worker's
    # --requests-dir/--worker-index/--dp-schedule reach the ranks verbatim.
    calls, _ = launches
    requests_dir = _manifest(tmp_path, [dict(prompt="p0")])
    worker_argv = [
        "/venv/lib/difflet/cli/main.py",
        "generate",
        "--model-id",
        WAN22,
        "--cfg-parallel",
        "--requests-dir",
        str(requests_dir),
        "--worker-index",
        "0",
        "--dp-schedule",
        "least_loaded",
    ]
    mod.WanCudaOrchestrator(_batch_args(requests_dir, cfg_parallel=True)).generate(argv=worker_argv)
    ((world_size, argv, _),) = calls
    assert world_size == 2 and argv[3:] == worker_argv[2:]
    # The parent worker claims nothing: the ranks do.
    assert recorder == [] and not (requests_dir / "req_0000.claim").exists()


@pytest.fixture
def broadcasts(monkeypatch):
    """Fake ``torch.distributed.broadcast_object_list`` for a rank of a 2-rank world.

    On rank 0 it records what was sent; on other ranks it delivers ``state["feed"]``
    one item per call (what rank 0 would have sent).
    """
    import torch.distributed

    state = {"sent": [], "feed": []}

    def fake(object_list, src=0, group=None, device=None):
        assert src == 0
        if os.environ.get("RANK") == "0":
            state["sent"].append(object_list[0])
        else:
            object_list[0] = state["feed"].pop(0)

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", fake)
    return state


def test_inside_a_cfg_world_rank_zero_claims_and_broadcasts(
    recorder, saved, launches, broadcasts, monkeypatch, tmp_path
):
    from difflet.cli.dp.claims import summarize

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    requests_dir = _manifest(tmp_path, [dict(prompt="p0"), dict(prompt="p1")])
    orch = mod.WanCudaOrchestrator(_batch_args(requests_dir, cfg_parallel=True))
    orch.generate(argv=[])
    assert [r.prompt if r else None for r in broadcasts["sent"]] == ["p0", "p1", None]
    assert [p for p, _ in orch._last_pipe.app.calls] == ["p0", "p1"]
    assert len(saved) == 2 and summarize(requests_dir).done == [0, 1]


def test_inside_a_cfg_world_rank_one_follows_rank_zero_and_writes_nothing(
    recorder, saved, launches, broadcasts, monkeypatch, tmp_path
):
    from difflet.cli.dp.claims import summarize
    from difflet.cli.dp.requests_io import read_manifest

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    requests_dir = _manifest(tmp_path, [dict(prompt="p0"), dict(prompt="p1")])
    broadcasts["feed"] = [*read_manifest(requests_dir), None]
    orch = mod.WanCudaOrchestrator(_batch_args(requests_dir, cfg_parallel=True))
    orch.generate(argv=[])
    # Same requests, same order as rank 0 (SPMD), but no claims, markers or outputs.
    assert [p for p, _ in orch._last_pipe.app.calls] == ["p0", "p1"]
    assert saved == [] and not list(requests_dir.glob("*.claim"))
    s = summarize(requests_dir)
    assert s.done == [] and not s.failed and s.unfinished == [0, 1]


def _exploding_generate(message):
    def generate(self, prompt, **kw):
        self.calls.append((prompt, kw))
        if prompt == "boom":
            raise RuntimeError(message)
        return self.video

    return generate


def test_inside_a_cfg_world_a_failure_on_rank_one_re_raises_and_marks_nothing(
    recorder, saved, launches, broadcasts, monkeypatch, tmp_path
):
    # A one-rank failure must not leave the ranks' collectives out of step (rank 0 in a CFG
    # gather, this rank in the next broadcast): the follower re-raises, torchrun tears the
    # world down, and the router marks the claimed request failed.
    from difflet.cli.dp.requests_io import read_manifest

    monkeypatch.setattr(_App, "generate", _exploding_generate("rank 1 exploded"))
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    requests_dir = _manifest(tmp_path, [dict(prompt="boom"), dict(prompt="p1")])
    broadcasts["feed"] = [*read_manifest(requests_dir), None]
    orch = mod.WanCudaOrchestrator(_batch_args(requests_dir, cfg_parallel=True))
    with pytest.raises(RuntimeError, match="rank 1 exploded"):
        orch.generate(argv=[])
    assert [p for p, _ in orch._last_pipe.app.calls] == ["boom"]  # no next request
    assert not list(requests_dir.glob("*.failed")) and not list(requests_dir.glob("*.done"))
    assert saved == []


def test_inside_a_cfg_world_a_failure_on_rank_zero_marks_failed_then_exits(
    recorder, saved, launches, broadcasts, monkeypatch, tmp_path
):
    from difflet.cli.dp.claims import summarize

    monkeypatch.setattr(_App, "generate", _exploding_generate("rank 0 exploded"))
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    requests_dir = _manifest(tmp_path, [dict(prompt="boom"), dict(prompt="p1")])
    orch = mod.WanCudaOrchestrator(_batch_args(requests_dir, cfg_parallel=True))
    with pytest.raises(SystemExit) as exc:
        orch.generate(argv=[])
    assert exc.value.code == 1
    # .failed is written before the exit; the next request is never broadcast or claimed.
    s = summarize(requests_dir)
    assert list(s.failed) == [0] and "rank 0 exploded" in s.failed[0]
    assert s.done == [] and s.unfinished == [1]
    assert not (requests_dir / "req_0001.claim").exists()
    assert [r.prompt for r in broadcasts["sent"]] == ["boom"]
    assert [p for p, _ in orch._last_pipe.app.calls] == ["boom"] and saved == []


_FORBIDDEN = (
    "neuronx_distributed",
    "torch_neuronx",
    "torch_xla",
    "difflet.backends.trainium",
    "difflet.backends.tpu",
)


def test_wan_cuda_imports_nothing_from_neuron_or_the_sibling_backends():
    tree = ast.parse(Path(mod.__file__).read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    bad = [m for m in imported if any(m == p or m.startswith(p + ".") for p in _FORBIDDEN)]
    assert bad == []


def test_importing_the_cuda_and_trainium_wan_orchestrators_pulls_no_neuron_module():
    # _save_video lives in the Trainium orchestrator: importing it must stay Neuron-free.
    code = textwrap.dedent("""
        import importlib.abc, sys
        BLOCKED = {"neuronx_distributed", "torch_neuronx", "neuronxcc", "nki", "torch_xla",
                   "libneuronxla"}
        BLOCKED_PREFIXES = ("difflet.backends.trainium", "difflet.backends.tpu")
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in BLOCKED or name.startswith(BLOCKED_PREFIXES):
                    raise ImportError("blocked: " + name)
        sys.meta_path.insert(0, Block())
        import difflet.cli.orchestrators.wan_cuda
        import difflet.cli.orchestrators.wan
        print("OK")
        """)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "DIFFLET_BACKEND": "cpu"},
    )
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stderr[-2000:]
