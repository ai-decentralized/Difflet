"""Wan 2.2 on the GPU CLI flow: what only Wan does. The video output (shared MP4 encoder,
``.pt`` fallback), CFG-parallel under ``torchrun`` and per-request
negative prompts. The flow itself is tested once per model in test_gpu_cli.py.

CPU-only: DiffletPipeline.from_pretrained is replaced by a recorder; no weights, no GPU.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

from difflet.serving import video_media
from difflet.models.wan.spec import SPEC
from tests.unit.cli import test_gpu_cli as gpu_cli_tests
from tests.unit.cli.test_gpu_cli import _App, _batch, _exploding_generate, _manifest, _orch

recorder = gpu_cli_tests.recorder
saved = gpu_cli_tests.saved
WAN22 = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"


@pytest.fixture
def spec():
    return SPEC


def test_generate_writes_the_mp4_through_save_video(recorder, saved, tmp_path):
    out = tmp_path / "wan.mp4"
    orch = _orch(SPEC, output=str(out))
    orch.generate()
    # The whole (B, 3, F, H, W) tensor goes to the helper, which takes frames[0] itself.
    ((tensor, path),) = saved
    assert tensor is orch._last_pipe.app.result and path == str(out)
    assert out.is_file() and not out.with_suffix(".pt").exists()


def test_a_failed_mp4_export_falls_back_to_a_pt_tensor(recorder, monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise video_media.VideoMediaEncodingError("encoding failed")

    monkeypatch.setattr(video_media, "encode_tensor_to_mp4", fail)
    out = tmp_path / "wan.mp4"
    orch = _orch(SPEC, output=str(out))
    orch.generate()
    pt = out.with_suffix(".pt")
    assert pt.is_file() and not out.exists()
    assert torch.equal(torch.load(pt), orch._last_pipe.app.result)


def test_a_non_mp4_output_is_saved_as_the_tensor(recorder, saved, tmp_path):
    out = tmp_path / "wan.pt"
    orch = _orch(SPEC, output=str(out))
    orch.generate()
    assert saved == [] and torch.equal(torch.load(out), orch._last_pipe.app.result)


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


@pytest.mark.parametrize(
    "over,world_size",
    [({}, 1), ({"cfg_parallel": True}, 2), ({"cfg_parallel": True, "tp_degree": 1}, 2)],
)
def test_world_size_is_cfg_times_cp_times_tp(over, world_size):
    assert _orch(SPEC, **over).world_size == world_size


def test_compile_with_cfg_parallel_builds_the_cfg_parallel_config(recorder, launches):
    _orch(SPEC, cfg_parallel=True).compile()
    _, kw = recorder[0]
    assert kw["parallel"].cfg_parallel_enabled is True and kw["parallel"].tp_degree == 1
    assert kw["load"] is False
    assert launches[0] == []  # compile never re-execs


def test_generate_with_cfg_parallel_re_execs_under_torchrun(recorder, saved, launches, tmp_path):
    calls, _ = launches
    orch = _orch(SPEC, cfg_parallel=True, output=str(tmp_path / "w.mp4"))
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
    _orch(SPEC, cfg_parallel=True).generate(argv=list(_USER_ARGV))
    ((_, argv, _),) = calls
    assert argv[:3] == ["-m", "difflet.cli.main", "generate"] and "run" not in argv
    assert argv[3:] == _USER_ARGV[2:]


def test_generate_reads_sys_argv_by_default(recorder, saved, launches, monkeypatch):
    calls, _ = launches
    monkeypatch.setattr(sys, "argv", list(_USER_ARGV))
    _orch(SPEC, cfg_parallel=True).generate()
    assert calls[0][1] == ["-m", "difflet.cli.main", "generate", *_USER_ARGV[2:]]


def test_a_failed_launch_exits_with_the_child_return_code(recorder, saved, launches):
    _, state = launches
    state["code"] = 7
    with pytest.raises(SystemExit) as exc:
        _orch(SPEC, cfg_parallel=True).generate(argv=list(_USER_ARGV))
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
    _orch(SPEC, cfg_parallel=True).run()
    assert pulled == [WAN22]
    # One parent-side build (compile, load=False), then the launch; no parent-side load.
    assert [kw["load"] for _, kw in recorder] == [False]
    assert recorder[0][1]["parallel"].cfg_parallel_enabled is True
    assert [(ws, argv[2]) for ws, argv, _ in calls] == [(2, "generate")]


def test_one_process_generate_never_launches(recorder, saved, launches, tmp_path):
    calls, _ = launches
    _orch(SPEC, output=str(tmp_path / "w.mp4")).generate(argv=list(_USER_ARGV))
    assert calls == [] and [kw["load"] for _, kw in recorder] == [True] and len(saved) == 1


def test_a_refused_flag_is_refused_before_any_launch(recorder, launches):
    calls, _ = launches
    with pytest.raises(SystemExit) as exc:
        _orch(SPEC, cfg_parallel=True, host_vae=True).generate(argv=[])
    assert "not supported" in str(exc.value) and calls == [] and recorder == []


@pytest.mark.parametrize("rank", ["0", "1"])
def test_inside_a_rank_generate_runs_in_process(
    recorder, saved, launches, monkeypatch, tmp_path, capsys, rank
):
    calls, _ = launches
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", rank)
    out = tmp_path / "deep" / "w.mp4"
    orch = _orch(SPEC, cfg_parallel=True, output=str(out))
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


# ------------------------------------------------- batch mode: a dp worker


def test_batch_mode_forwards_each_request_s_overrides(recorder, saved, tmp_path):
    from difflet.cli.dp.claims import summarize

    requests_dir = _manifest(
        SPEC,
        tmp_path,
        [
            dict(prompt="p0", seed=0),
            dict(prompt="p1", seed=1, steps=3, guidance_scale=2.5, negative_prompt="blurry"),
        ],
    )
    orch = _orch(SPEC, **_batch(requests_dir, num_frames=5))
    orch.generate(argv=[])
    (p0, kw0), (p1, kw1) = orch._last_pipe.app.calls
    assert (p0, kw0["guidance_scale"]) == ("p0", SPEC.default_guidance_scale)
    assert "negative_prompt" not in kw0 and kw0["num_frames"] == 5
    assert (p1, kw1["guidance_scale"], kw1["negative_prompt"]) == ("p1", 2.5, "blurry")
    assert kw1["num_frames"] == 5
    assert [path for _, path in saved] == [str(tmp_path / "out" / f"r{i}.mp4") for i in (0, 1)]
    assert summarize(requests_dir).done == [0, 1]


def test_a_cfg_parallel_worker_re_execs_torchrun_with_its_worker_flags(
    recorder, saved, launches, tmp_path
):
    # A dp replica with --cfg-parallel is itself a two-rank torchrun world: the worker's
    # --requests-dir/--worker-index/--dp-schedule reach the ranks verbatim.
    calls, _ = launches
    requests_dir = _manifest(SPEC, tmp_path, [dict(prompt="p0")])
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
    _orch(SPEC, **_batch(requests_dir, cfg_parallel=True)).generate(argv=worker_argv)
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
    requests_dir = _manifest(SPEC, tmp_path, [dict(prompt="p0"), dict(prompt="p1")])
    orch = _orch(SPEC, **_batch(requests_dir, cfg_parallel=True))
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
    requests_dir = _manifest(SPEC, tmp_path, [dict(prompt="p0"), dict(prompt="p1")])
    broadcasts["feed"] = [*read_manifest(requests_dir), None]
    orch = _orch(SPEC, **_batch(requests_dir, cfg_parallel=True))
    orch.generate(argv=[])
    # Same requests, same order as rank 0 (SPMD), but no claims, markers or outputs.
    assert [p for p, _ in orch._last_pipe.app.calls] == ["p0", "p1"]
    assert saved == [] and not list(requests_dir.glob("*.claim"))
    s = summarize(requests_dir)
    assert s.done == [] and not s.failed and s.unfinished == [0, 1]


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
    requests_dir = _manifest(SPEC, tmp_path, [dict(prompt="boom"), dict(prompt="p1")])
    broadcasts["feed"] = [*read_manifest(requests_dir), None]
    orch = _orch(SPEC, **_batch(requests_dir, cfg_parallel=True))
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
    requests_dir = _manifest(SPEC, tmp_path, [dict(prompt="boom"), dict(prompt="p1")])
    orch = _orch(SPEC, **_batch(requests_dir, cfg_parallel=True))
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
