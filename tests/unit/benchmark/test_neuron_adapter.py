"""CPU tests of the neuron benchmark adapter: launch environments, command lines, the CLI log
parser, the per-launch summary, the counter rule, and how a launch is stopped (W6)."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from benchmark.adapters import neuron as na
from benchmark.models import MATRIX
from tests.unit.benchmark import _neuron_results as fake

CACHE_VARS = ("TORCH_NEURONX_NEFF_CACHE_DIR", "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR",
              "TORCH_NEURONX_HLO_CACHE_DIR")


# ------------------------------------------------------------------------- environments


def _shell():
    return {"PATH": "/usr/bin", "HOME": "/home/u",
            "TORCH_NEURONX_NEFF_CACHE_DIR": "/elsewhere/neff",
            "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR": "/elsewhere/local",
            "TORCH_NEURONX_HLO_CACHE_DIR": "/elsewhere/hlo",
            "WORLD_SIZE": "8", "RANK": "3", "DIFFLET_BACKEND": "trainium",
            "DIFFLET_EXEC_MODE": "eager", "NEURON_RT_NUM_CORES": "2",
            "OMP_NUM_THREADS": "12"}


def test_bench_env_never_inherits_the_shells_cache_locations(tmp_path):
    env = na.launch_env(_shell(), cache_root=tmp_path / "row", export_cache_dirs=True,
                        omp_threads=na.BENCH_OMP_THREADS)
    root = str(tmp_path / "row")
    assert env["DIFFLET_COMPILE_CACHE"] == root
    assert env["TORCH_NEURONX_NEFF_CACHE_DIR"] == os.path.join(root, "neuron", "neff")
    assert env["TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR"] == os.path.join(root, "neuron", "neff_local")
    assert env["TORCH_NEURONX_HLO_CACHE_DIR"] == os.path.join(root, "neuron", "hlo")
    assert env["NKI_ENABLE_TRACE_CACHE"] == "0"
    for name in ("WORLD_SIZE", "RANK", "DIFFLET_BACKEND", "DIFFLET_EXEC_MODE",
                 "NEURON_RT_NUM_CORES"):
        assert name not in env
    assert env["OMP_NUM_THREADS"] == "3"  # brief 8.5, not the shell's
    assert env["HF_HUB_OFFLINE"] == "1" and env["DIFFLET_NO_JEMALLOC"] == "1"
    assert env["DIFFLET_NEURON_PG_TIMEOUT_S"] == "3600"
    assert env["PYTHONSAFEPATH"] == "1" and env["PYTHONPATH"] == str(na.REPO_ROOT)
    assert env["PATH"] == "/usr/bin"
    assert "TORCH_NEURONX_ENABLED_METRIC_TABLES" not in env


def test_cli_env_leaves_the_cache_locations_to_the_launcher(tmp_path):
    env = na.launch_env(_shell(), cache_root=tmp_path / "row", export_cache_dirs=False)
    assert env["DIFFLET_COMPILE_CACHE"] == str(tmp_path / "row")
    assert not any(name in env for name in CACHE_VARS)


def test_cli_env_is_the_users_default_threads(tmp_path):
    # D82: no OMP_NUM_THREADS, not even the shell's 12, so torchrun and the driver keep theirs
    env = na.launch_env(_shell(), cache_root=tmp_path, export_cache_dirs=False)
    assert "OMP_NUM_THREADS" not in env
    assert "OMP_NUM_THREADS" not in na.LAUNCH_ENV and na.BENCH_OMP_THREADS == "3"
    assert "OMP_NUM_THREADS" in na.RECORDED_ENV  # recorded as None for a CLI launch


def test_metrics_dir_turns_on_the_graph_stats_table(tmp_path):
    env = na.launch_env({}, cache_root=tmp_path, export_cache_dirs=True,
                        metrics_dir=tmp_path / "m")
    assert env["TORCH_NEURONX_ENABLED_METRIC_TABLES"] == "graph_stats"
    assert env["TORCH_NEURONX_METRICS_DIR"] == str(tmp_path / "m")


def test_cache_dirs_layout(tmp_path):
    dirs = na.cache_dirs(tmp_path)
    assert dirs == {"neff": tmp_path / "neuron" / "neff",
                    "neff_local": tmp_path / "neuron" / "neff_local",
                    "hlo": tmp_path / "neuron" / "hlo"}


# ----------------------------------------------------------------------------- commands


def test_single_backend_flag_dedupes_and_checks():
    flags = MATRIX["wan_2_1_neuron"].extra_generate_flags
    out = na.single_backend_flag(flags)
    assert out.count("--backend") == 1 and out[out.index("--backend") + 1] == "neuron"
    assert na.single_backend_flag(["--host-vae"]) == ["--host-vae", "--backend", "neuron"]
    assert na.single_backend_flag(["--backend=neuron", "--backend", "neuron"]) == [
        "--backend", "neuron"]
    with pytest.raises(ValueError, match="trainium"):
        na.single_backend_flag(["--backend", "trainium"])


def test_cli_generate_command_for_the_matrix_entry(tmp_path):
    cfg = MATRIX["wan_2_1_neuron"]
    cmd = na.cli_generate_command("/venv/bin/python", cfg, output=tmp_path / "o.mp4",
                                  cache_dir=tmp_path / "c", work_dir=tmp_path / "w")
    assert cmd[:4] == ["/venv/bin/python", "-m", "difflet.cli.main", "generate"]
    assert cmd.count("--backend") == 1
    for flag in ("--host-vae", "--keep-work-dir"):
        assert flag in cmd
    pairs = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1) if cmd[i].startswith("--")}
    assert pairs["--exec-mode"] == "compile" and pairs["--backend"] == "neuron"
    assert pairs["--revision"] == cfg.revision and pairs["--steps"] == "50"
    assert pairs["--guidance-scale"] == "1" and pairs["--seed"] == "42"
    assert pairs["--tp-degree"] == "4" and pairs["--cp-degree"] == "1"
    assert (pairs["--height"], pairs["--width"], pairs["--num-frames"]) == ("480", "832", "9")
    assert pairs["--prompt"] == cfg.prompt
    assert pairs["--cache-dir"] == str(tmp_path / "c")
    assert not any("nxd_inference" in part for part in cmd)


def test_bench_command_is_a_four_rank_torchrun_of_the_bench_script(tmp_path):
    cmd = na.bench_command("/venv/bin/python", model_dir="/snap", out_dir=tmp_path,
                           steps=50, guidance_scale=5.0, iters=3, prompt="p", seed=42,
                           height=480, width=832, num_frames=9, expect="warm")
    assert cmd[:6] == ["/venv/bin/python", "-m", "torch.distributed.run", "--standalone",
                       "--nproc-per-node", "4"]
    assert cmd[6] == str(na.BENCH_SCRIPT) and na.BENCH_SCRIPT.is_file()
    pairs = {cmd[i]: cmd[i + 1] for i in range(7, len(cmd) - 1, 2)}
    assert pairs["--guidance-scale"] == "5" and pairs["--iters"] == "3"
    assert pairs["--expect"] == "warm" and pairs["--model-dir"] == "/snap"


# ------------------------------------------------------------------------------ CLI log


CLI_LOG = """\
[difflet] something
[wan] neuron rank 1/4: exec_mode=compile compiled_blocks=40 unwarmed_shapes=[] fallbacks: load=[] generate=[] load+warmup 28.4s, stage 83.6s
[wan] neuron rank 2/4: exec_mode=compile compiled_blocks=40 unwarmed_shapes=[] fallbacks: load=[] generate=[] load+warmup 28.1s, stage 83.5s
[wan] neuron rank 3/4: exec_mode=compile compiled_blocks=40 unwarmed_shapes=[] fallbacks: load=[] generate=[] load+warmup 27.9s, stage 83.4s
[wan] latents saved to /w/latents.pt
[wan] neuron rank 0/4: exec_mode=compile compiled_blocks=40 unwarmed_shapes=[] fallbacks: load=[] generate=[] load+warmup 29.0s, stage 84.0s
[wan] video saved to /o/out.mp4
"""


def test_parse_cli_log_reads_the_rank_lines():
    parsed = na.parse_cli_log(CLI_LOG)
    assert sorted(parsed["ranks"]) == [0, 1, 2, 3]
    assert parsed["ranks"][0]["load_warmup_s"] == 29.0
    assert parsed["ranks"][3]["stage_s"] == 83.4
    assert parsed["latents_saved"] == ["/w/latents.pt"]
    assert parsed["video_saved"] == ["/o/out.mp4"]
    assert parsed["failure_lines"] == 0
    assert all(ok for _, ok, _ in na.cli_checks(parsed))


def test_parse_cli_log_survives_interleaved_rank_output():
    # torchrun's ranks run with python -u: a print's text and its newline are two writes, so
    # another rank's line can land in between (seen in W6 g5.0/cli-warm1).
    lines = CLI_LOG.splitlines()
    rank3, rank2 = lines[3], lines[2]
    text = "\n".join([lines[0], lines[1], rank3 + rank2, "", lines[4],
                      lines[5] + "[wan] video saved to /o/out.mp4"]) + "\n"
    parsed = na.parse_cli_log(text)
    assert sorted(parsed["ranks"]) == [0, 1, 2, 3]
    assert parsed["ranks"][3]["stage_s"] == 83.4 and parsed["ranks"][2]["stage_s"] == 83.5
    assert parsed["latents_saved"] == ["/w/latents.pt"]
    assert parsed["ranks"][0]["stage_s"] == 84.0
    assert parsed["video_saved"] == ["/o/out.mp4"]
    assert all(ok for _, ok, _ in na.cli_checks(parsed))


def test_parse_cli_log_keeps_nested_unwarmed_shapes_and_flags_them():
    line = ("[wan] neuron rank 0/4: exec_mode=compile compiled_blocks=40 unwarmed_shapes="
            "[((1, 16, 3, 60, 104), (1,), (1, 512, 4096))] fallbacks: load=[] "
            "generate=['aten::foo'] load+warmup 30.0s, stage 90.0s\n")
    parsed = na.parse_cli_log(line + "Traceback (most recent call last):\n")
    rank = parsed["ranks"][0]
    assert rank["unwarmed_shapes"] == "[((1, 16, 3, 60, 104), (1,), (1, 512, 4096))]"
    assert rank["generate_fallbacks"] == "['aten::foo']"
    checks = {name: ok for name, ok, _ in na.cli_checks(parsed)}
    assert checks == {"rank_lines": False, "compile_clean": False, "one_writer": False,
                      "video_saved": False, "no_failure": False}


def test_read_rank_results_requires_every_rank(tmp_path):
    for rank in (0, 1, 3):
        (tmp_path / f"result-rank{rank}.json").write_text(json.dumps({"rank": rank}))
    with pytest.raises(ValueError, match="found ranks \\[0, 1, 3\\]"):
        na.read_rank_results(tmp_path)
    (tmp_path / "result-rank2.json").write_text(json.dumps({"rank": 2}))
    assert [r["rank"] for r in na.read_rank_results(tmp_path)] == [0, 1, 2, 3]


# ------------------------------------------------------------------------- step checks


def test_step_series_checks_pass_a_flat_series():
    out = na.step_series_checks([1.07] * 49)
    assert out["ok"] and out["tail_ratio"] == 1.0 and out["drift"] == 0.0


def test_step_series_checks_flag_a_tail_and_a_drift():
    tail = [1.0] * 40 + [1.2] * 9
    assert na.step_series_checks(tail)["tail_ratio"] == pytest.approx(1.2)
    assert not na.step_series_checks(tail)["ok"]
    drift = [1.0] * 10 + [1.0] * 29 + [1.06] * 10
    out = na.step_series_checks(drift)
    assert out["drift"] == pytest.approx(0.06) and not out["ok"]
    assert out["tail_ratio"] <= 1.10
    short = na.step_series_checks([1.0] * 5)
    assert not short["ok"] and math.isnan(short["tail_ratio"])


def test_throttle_fields_ignore_the_window_timestamps():
    idle = ("entries=1 mean=128/255(50%) max=128/255(50%) min=128/255(50%) first_ts=1 "
            "last_ts=1")
    busy = ("entries=1 mean=164/255(64%) max=255/255(100%) min=128/255(50%) first_ts=9 "
            "last_ts=9")
    later_idle = idle.replace("_ts=1", "_ts=7")
    assert na.throttle_fields({"neuron_core0": idle, "power_utilization": "x"}) == {
        "neuron_core0": {"entries": "1", "max": "128/255"}}
    assert not na.throttle_changed({"neuron_core0": idle}, {"neuron_core0": later_idle})
    assert na.throttle_changed({"neuron_core0": busy}, {"neuron_core0": idle})
    assert not na.throttle_changed(None, None)


def test_spread():
    assert na.spread([1.0, 1.01, 1.02]) == pytest.approx(0.02)
    assert math.isnan(na.spread([1.0]))


# ---------------------------------------------------------------------- counter rule


def _summary(results, **kwargs):
    return na.summarize_bench_launch(results, steps=kwargs.get("steps", 50),
                                     guidance_scale=kwargs.get("guidance", 5.0))


def test_counter_rule_warm():
    summary = _summary(fake.launch_results())
    ok, text = na.counter_rule(summary, "warm", 0)
    assert ok and "r0 0/39/7459" in text and "new .neff 0" in text
    assert not na.counter_rule(summary, "warm", 1)[0]  # a new NEFF file
    compiled = fake.launch_results()
    compiled[2]["counters"]["run"] = fake.counters(total=1, hits=39)
    assert not na.counter_rule(_summary(compiled), "warm", 0)[0]
    no_hits = fake.launch_results(run_counters=fake.counters(in_memory=10))
    assert not na.counter_rule(_summary(no_hits), "warm", 0)[0]


def test_counter_rule_cold_is_decided_by_any_rank_compiling():
    summary = _summary(fake.cold_results())
    ok, _ = na.counter_rule(summary, "cold", 47)
    assert ok
    assert not na.counter_rule(summary, "cold", 0)[0]
    nothing = _summary(fake.launch_results())  # only persistent hits: that is a warm launch
    assert not na.counter_rule(nothing, "cold", 47)[0]
    with pytest.raises(ValueError):
        na.counter_rule(summary, "lukewarm", 0)


def test_compile_split_tells_block_from_eager_op_neffs():
    split = na.compile_split(fake.compile_info(block_ms=55_000, eager_ms=(800, 0, 400)))
    assert split["block_compiled"] == 1 and split["block_compile_s"] == 55.0
    assert split["eager_compiled"] == 2 and split["eager_compile_s"] == pytest.approx(1.2)
    assert split["eager_loaded"] == 1 and split["dynamo_graphs"] == 1
    assert split["compile_time_s"] == pytest.approx(56.2)
    assert split["dynamo_lower_s"] == pytest.approx(0.187)
    loaded = na.compile_split(fake.compile_info())
    assert loaded["block_loaded"] == 1 and loaded["block_compiled"] == 0
    assert na.compile_split(None)["dynamo_graphs"] == 0


# --------------------------------------------------------------------------- summaries


def test_summarize_bench_launch_numbers():
    results = fake.launch_results()
    results[2]["iterations"][0]["step_deltas_s"][5] = 1.30  # one slow step on rank 2
    summary = _summary(results)
    assert summary["ok"]
    first = summary["iterations"][0]
    assert first["ok"] and first["no_compile"]
    assert summary["e2e_script_s"] == pytest.approx(40.0 + 56.0 + 53.0)
    assert summary["e2e_latents_s"] == pytest.approx(96.0)
    assert first["step"]["n"] == 49 and first["step"]["median"] == pytest.approx(1.07)
    assert first["step_rank_max"]["max"] == pytest.approx(1.30)  # elementwise max over ranks
    assert first["forward"]["n"] == 100
    assert first["text_encode_s"] == 2.0 and first["negative_encode_s"] == 1.5
    assert first["host_decode_s"] == 53.0 and first["decode_threads"] == 12
    assert summary["rank_max"]["warmup_forward_s"] == 13.0
    assert summary["text_encoder_load_s"] == 0.7
    assert summary["resident_within_2pct"] is True
    assert summary["checks"]["hlo_locks_in_cache"]


def test_summarize_flags_what_invalidates_a_series():
    one_forward_short = fake.launch_results(guidance=1.0)
    assert _summary(one_forward_short, guidance=1.0)["ok"]  # 50 forwards at g=1.0
    assert not _summary(one_forward_short, guidance=5.0)["iterations"][0]["checks"]["forwards"]

    differ = fake.launch_results()
    differ[3]["iterations"][1]["latents_sha256"] = "0" * 64
    summary = _summary(differ)
    assert not summary["iterations"][1]["checks"]["latents_equal_on_ranks"]
    assert not summary["checks"]["latents_equal_across_iterations"]
    assert not summary["ok"]

    fell_back = fake.launch_results()
    fell_back[1]["iterations"][2]["fallbacks"] = ["aten::index"]
    fell_back[0]["iterations"][2]["unwarmed_shapes"] = [[[1, 16, 3, 60, 104]]]
    it = _summary(fell_back)["iterations"][2]
    assert not it["checks"]["fallbacks"] and not it["checks"]["unwarmed_shapes"]

    compiled = fake.launch_results(generate_counters=fake.counters(total=2))
    out = _summary(compiled)
    assert out["ok"] and not out["iterations"][0]["no_compile"]  # judged by the driver

    locks = fake.launch_results()
    locks[0]["hlo_lock_dir_at_import"] = "/tmp/hlo_cache"
    assert not _summary(locks)["checks"]["hlo_locks_in_cache"]


def test_resident_rule_flags_a_spread_above_two_percent():
    results = fake.launch_results()
    for r in results:
        r["iterations"][2]["step_deltas_s"] = [1.07 * 1.03] * 49
    summary = _summary(results)
    assert summary["resident_within_2pct"] is False
    assert summary["resident_step_median_spread"] == pytest.approx(0.03)


# ----------------------------------------------------------------------------- devices


def test_device_and_ps_busy_detection():
    idle = json.dumps([{"neuron_device": 0, "neuron_processes": []}])
    busy = json.dumps([{"neuron_device": 0, "neuron_processes": [{"pid": 7}]}])
    assert na.device_processes(idle) == [] and na.device_processes(busy) == [{"pid": 7}]
    ps = ("    PID COMMAND\n"
          "      1 /sbin/init\n"
          "    100 /venv/bin/python -m torch.distributed.run --standalone x\n"
          "    101 /venv/bin/python /wt/benchmark/neuron_wan_bench.py --inspect-latents a\n"
          "    102 /venv/bin/python -m benchmark.neuron_cold_warm --record-dir r\n"
          "    103 /venv/bin/python /wt/benchmark/neuron_wan_bench.py --model-dir s\n")
    assert na.busy_processes(ps) == [
        "100 /venv/bin/python -m torch.distributed.run --standalone x",
        "103 /venv/bin/python /wt/benchmark/neuron_wan_bench.py --model-dir s"]
    assert na.busy_processes(ps, ignore_pids=[100, 103]) == []
    # an operator's `ps | grep -E 'torchrun|...'` is not a device job (M-7)
    grep = "    PID COMMAND\n    104 grep -E torchrun|neuron_wan_bench\n" \
           "    105 /usr/bin/grep --color=auto torchrun\n"
    assert na.busy_processes(grep) == []


def test_descendants_walks_the_whole_tree():
    parents = {10: 1, 11: 10, 12: 10, 13: 11, 14: 13, 20: 1}
    assert na.descendants(10, parents) == [11, 12, 13, 14]
    assert na.descendants(20, parents) == []
    assert na.process_parents()[os.getpid()] == os.getppid()


class _FakeProc:
    pid = 4242

    def __init__(self, waits):
        self.waits = list(waits)

    def wait(self, timeout=None):
        outcome = self.waits.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def test_run_logged_stops_the_launch_before_reraising_an_interrupt(tmp_path, monkeypatch):
    stopped = []
    monkeypatch.setattr(na, "stop_launch", lambda proc, grace: stopped.append(grace) or -2)
    proc = _FakeProc([KeyboardInterrupt()])
    popen_kwargs = {}

    def popen(cmd, **kwargs):
        popen_kwargs.update(kwargs)
        return proc

    with pytest.raises(KeyboardInterrupt):
        na.run_logged(["torchrun"], {}, tmp_path / "x.log", timeout=10, stop_grace=5,
                      popen=popen)
    assert stopped == [5]  # the ranks were stopped before the interrupt propagated
    assert popen_kwargs["start_new_session"] is True  # only run_logged signals the launch

    stopped.clear()
    out = na.run_logged(["torchrun"], {}, tmp_path / "y.log", timeout=10, stop_grace=5,
                        popen=lambda cmd, **kw: _FakeProc([subprocess.TimeoutExpired("x", 10)]))
    assert out["timed_out"] and out["exit_code"] == -2 and stopped == [5]


def test_stop_launch_kills_ranks_that_left_the_process_group(tmp_path):
    # A stand-in for torchrun: ignores SIGINT and starts a "rank" in a session of its own
    # (as torchrun does) that ignores SIGINT too; neither stops by itself.
    rank_pid = tmp_path / "rank.pid"
    code = (
        "import signal, subprocess, sys, time\n"
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "rank = subprocess.Popen([sys.executable, '-c', 'import signal, time; "
        "signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(120)'], "
        "start_new_session=True)\n"
        f"open({str(rank_pid)!r}, 'w').write(str(rank.pid))\n"
        "time.sleep(120)\n")
    out = na.run_logged([sys.executable, "-c", code], dict(os.environ), tmp_path / "l.log",
                        timeout=3, stop_grace=1)
    assert out["timed_out"] and out["exit_code"] == -signal.SIGKILL
    pid = int(rank_pid.read_text())
    deadline = time.time() + 10
    while time.time() < deadline and _alive(pid):
        time.sleep(0.1)
    assert not _alive(pid)  # the rank did not outlive its launcher


def test_a_repeated_interrupt_does_not_abandon_the_wait():
    proc = _FakeProc([KeyboardInterrupt(), KeyboardInterrupt(), 0])
    assert na.wait_through_interrupts(proc, 30) == 0  # two more Ctrl-C, still waited
    late = _FakeProc([KeyboardInterrupt(), subprocess.TimeoutExpired("x", 1)])
    assert na.wait_through_interrupts(late, 1) is None  # -> the SIGKILL fallback follows


def test_stop_launch_kills_a_rank_left_behind_by_a_launcher_that_exited(tmp_path):
    # The launcher exits on SIGINT within the grace period but leaves its session-detached
    # "rank" (which ignores SIGINT) running, as a CLI that gave up on its torchrun would.
    rank_pid = tmp_path / "rank.pid"
    code = (
        "import signal, subprocess, sys, time\n"
        "signal.signal(signal.SIGINT, lambda *a: sys.exit(0))\n"
        "rank = subprocess.Popen([sys.executable, '-c', 'import signal, time; "
        "signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(120)'], "
        "start_new_session=True)\n"
        f"open({str(rank_pid)!r}, 'w').write(str(rank.pid))\n"
        "time.sleep(120)\n")
    out = na.run_logged([sys.executable, "-c", code], dict(os.environ), tmp_path / "l.log",
                        timeout=3, stop_grace=30)
    assert out["timed_out"] and out["exit_code"] == 0  # the launcher stopped by itself
    pid = int(rank_pid.read_text())
    deadline = time.time() + 10
    while time.time() < deadline and _alive(pid):
        time.sleep(0.1)
    assert not _alive(pid)


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state != "Z"


def test_cli_timeline_and_torchrun_note(tmp_path):
    latents, mp4 = tmp_path / "latents.pt", tmp_path / "out.mp4"
    latents.write_text("x")
    mp4.write_text("y")
    os.utime(latents, (1000.0, 1000.0))
    os.utime(mp4, (1060.0, 1060.0))
    assert na.file_timeline({"start_epoch": 900.0, "end_epoch": 1061.0}, latents, mp4) == {
        "to_latents_s": 100.0, "decode_s": 60.0, "after_mp4_s": 1.0}
    assert na.file_timeline({"start_epoch": 0, "end_epoch": 1}, tmp_path / "no", mp4) is None
    log = "a\nSetting OMP_NUM_THREADS environment variable for each process to be 1 ...\nb\n"
    assert na.torchrun_omp_lines(log) == [
        "Setting OMP_NUM_THREADS environment variable for each process to be 1 ..."]


def test_bench_all_skips_entries_for_other_backends(monkeypatch):
    from benchmark import bench, result_cache

    ran = []
    monkeypatch.setattr(bench, "run_one", lambda slug, backend, **kw: ran.append(slug))
    monkeypatch.setattr(bench, "write_outputs", lambda slug, res, **kw: None)
    # no host probes; --force: the committed results' cache state must not skip a slug
    monkeypatch.setattr(result_cache, "instance_type", lambda: "test")
    monkeypatch.setattr(result_cache, "system_sdk", lambda: {})
    monkeypatch.setattr(result_cache, "platform_identity", lambda adapter, **kw: {})
    monkeypatch.setattr(sys, "argv", ["bench", "--all", "--backend", "trainium", "--force"])
    assert bench.main() == 0
    assert "wan_2_1_neuron" not in ran and "wan_2_1" in ran
    ran.clear()
    monkeypatch.setattr(sys, "argv", ["bench", "--all", "--backend", "neuron", "--force"])
    bench.main()
    assert "wan_2_1_neuron" in ran and "wan_2_1" in ran  # backends=None: any backend
    assert MATRIX["wan_2_1_neuron"].backends == ("neuron",)
    assert all(cfg.backends is None for slug, cfg in MATRIX.items() if slug != "wan_2_1_neuron")


def test_adapter_compiles_nothing_ahead_of_time(tmp_path):
    adapter = na.NeuronAdapter(cache_dir=str(tmp_path), log_dir=str(tmp_path / "logs"))
    assert adapter.compile(MATRIX["wan_2_1_neuron"]) == (0.0, {"aot": 0.0})
    assert adapter.name == "neuron" and adapter.cache_dir == str(tmp_path)


def test_bench_selects_the_neuron_adapter():
    from benchmark import bench

    assert type(bench._make_adapter("neuron")).__name__ == "NeuronAdapter"


def test_matrix_entry_matches_wan_2_1_but_for_steps_and_flags():
    ref, cfg = MATRIX["wan_2_1"], MATRIX["wan_2_1_neuron"]
    for field in ("model_id", "revision", "model_type", "tp", "cp", "height", "width",
                  "num_frames", "prompt", "seed", "dtype", "guidance_scale", "output_kind"):
        assert getattr(cfg, field) == getattr(ref, field), field
    assert cfg.steps == 50 and cfg.guidance_scale == 1.0
    assert cfg.extra_generate_flags == ["--host-vae", "--backend", "neuron", "--exec-mode",
                                        "compile"]
    assert cfg.config_label == ("tp=4, bf16, TorchNeuron per-block torch.compile, host fp32 "
                                "umT5 on rank 0, host VAE")


def test_adapter_module_imports_no_torch():
    import subprocess
    import sys

    code = ("import sys; import benchmark.adapters.neuron, benchmark.neuron_cold_warm; "
            "print([m for m in ('torch', 'torch_neuronx') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(na.REPO_ROOT)}, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
