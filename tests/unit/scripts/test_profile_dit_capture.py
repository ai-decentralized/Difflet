"""scripts/profile_dit/capture.sh: the commands a lane runs and the block NEFFs it keeps, driven
with a recording PY, a fake neuron-explorer and a fake fuser on PATH (no device, no model), as
test_quant_bench.py drives bench.sh. A non-block NEFF profiled as a block, a busy device used
anyway or a wrong flag costs a device lane and a misleading breakdown, so those are pinned here;
system_profile.py's step window is pinned on a synthetic system profile."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PD = REPO / "scripts" / "profile_dit"
#: The configuration table: height, width, frames (None: an image model) and block NEFFs per arm.
TABLE = {"qwen_image": ("1024", "1024", None, 1), "flux": ("1024", "1024", None, 2),
         "ltx_2": ("512", "768", "121", 1), "wan21_official": ("480", "832", "81", 1),
         "wan22_official": ("480", "832", "81", 1), "hunyuan_video_L": ("544", "960", "129", 2),
         "hv15_official": ("480", "848", "121", 1)}
ENV = ("NEURON_RT_VISIBLE_CORES", "NEURON_RT_INSPECT_DEVICE_PROFILE")

#: The lane's PY: a `-m difflet.cli.main` command is recorded (argv and ENV), prints the
#: runtime's dropped-events warning for process FAKE_DROPPED_PID and the line FAKE_STEP_LINE if
#: set, and writes its --output; anything else (system_profile.py, the signature check) runs on
#: python.
FAKE_PY = """#!{exe}
import json, os, sys
if sys.argv[1:3] == ["-m", "difflet.cli.main"]:
    args = sys.argv[3:]
    with open(os.environ["PY_LOG"], "a") as f:
        f.write(json.dumps({{"argv": args, "env": {{k: os.environ.get(k) for k in {env!r}}}}}) + "\\n")
    pid = os.environ.get("FAKE_DROPPED_PID")
    if pid:
        print(f"2026-Nov-13 14:31:13.844683 {{pid}}:{{pid}}  WARN   NRT:dump_trace_info"
              "    System profile events were dropped due to full ring buffers (max 6403834"
              " dropped on a single NC).")
    if os.environ.get("FAKE_STEP_LINE"):
        print(os.environ["FAKE_STEP_LINE"])
    open(args[args.index("--output") + 1], "w").write("out")
    sys.exit(0)
os.execv(sys.executable, [sys.executable] + sys.argv[1:])
"""

#: neuron-explorer: `inspect` writes the dump FAKE_DUMP describes (per rank, {model id:
#: "<marker>[+<KiB>]"}: a NEFF is its marker in a 1024-byte header, then a tar.gz of a debug-info
#: file naming the marker's source files and 1 MiB (+ KiB) of incompressible bytes) and runs the
#: userscript, taking FAKE_DUMP_S seconds (its -o, start and end appended to DUMP_SPANS if set);
#: `capture` takes FAKE_CAPTURE_S seconds (its start and end appended to SPANS if
#: set) and writes the two NTFFs; `view` prints a summary, writes a system profile
#: (each NEFF once in a warm-up and once in the step, on both physical cores as the runtime
#: records it) or an Instruction table whose rows the NEFF's marker picks.
FAKE_X = """#!{exe}
import gzip, io, json, os, random, subprocess, sys, tarfile
from pathlib import Path
args = sys.argv[1:]
parent = open(f"/proc/{{os.getppid()}}/cmdline", "rb").read().split(b"\\0")[:4]  # its timeout
with open(os.environ["X_LOG"], "a") as f:
    f.write(json.dumps({{"argv": args, "cwd": os.getcwd(),
                        "parent": [p.decode() for p in parent],
                        "cores": os.environ.get("NEURON_RT_VISIBLE_CORES"),
                        "comm": os.environ.get("NEURON_RT_ROOT_COMM_ID")}}) + "\\n")
def opt(name):
    return args[args.index(name) + 1] if name in args else None
ATTN = "/venv/site-packages/nkilib/core/attention/attention_cte.py"
LINEAR = "/repo/difflet/backends/neuron/ops_impl/linear.py"
HV = "/repo/difflet/models/hunyuan_video/modeling_hunyuan_video.py"
VAE = "/repo/difflet/models/hunyuan_video/vae/modeling_vae.py"
DUAL = [("HunyuanVideoAttention[attn][0]_custom-call.411", ATTN)] * 3 + [
    ("HunyuanVideoFeedForward[ff][0]/FP8RowParallelLinear[2][0]_custom-call.552", LINEAR)]
#: marker -> (the source files its debug info names, its Instruction rows)
MARKERS = {{"block": (["/repo/difflet/models/toy/modeling_toy.py", ATTN, LINEAR],
                     [("Attn[attn][0]_custom-call.1", ATTN), ("Mlp[ff][0]_dot.2", LINEAR)]),
           "dual": ([HV, ATTN, LINEAR], DUAL),
           "vaerows": ([HV, ATTN], DUAL + [("Unknown", VAE)] * 9),
           "vae": ([VAE], None), "bare": ([], None)}}
def neff(spec):
    marker, _, kib = spec.partition("+")
    tar = io.BytesIO()
    with tarfile.open(fileobj=tar, mode="w") as t:
        for name, data in (("sg00/debug_info_asm_PE.dbg",
                            "\\n".join(MARKERS[marker.rstrip("0123456789")][0]).encode()),
                           ("sg00/PE0.bin", random.Random(marker).randbytes((1 << 20) + 1024 * int(kib or 0)))):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return marker.encode().ljust(1024, b"\\0") + gzip.compress(tar.getvalue(), mtime=0)
if args[0] == "inspect":
    out = Path(opt("-o"))
    for rank, neffs in enumerate(json.loads(os.environ["FAKE_DUMP"])):
        d = out / f"i-0test_pid_{{1000 + rank}}" / "17"
        d.mkdir(parents=True)
        for model_id, spec in neffs.items():
            (d / f"neff_{{model_id}}_vnc_0.neff").write_bytes(neff(spec))
    import time
    t0 = time.time()
    time.sleep(float(os.environ.get("FAKE_DUMP_S", 0)))
    rc = subprocess.call(args[args.index("-o") + 2:])
    if os.environ.get("DUMP_SPANS"):
        with open(os.environ["DUMP_SPANS"], "a") as f:
            f.write(json.dumps({{"out": str(out), "t0": t0, "t1": time.time()}}) + "\\n")
    sys.exit(rc)
if args[0] == "capture":
    import time
    ntffs = [Path(opt("-s")).with_name(f"profile_rank_0_exec_{{e}}.ntff") for e in (4, 8)]
    if os.environ.get("FAKE_NQ_OVERFLOW"):  # HV1's dual block with --enable-dge-notifs, 2026-11-13
        if "--enable-dge-notifs" in args:
            for n in ntffs:
                n.write_text("partial")
            print("ERROR  TDRV:exec_request_process_errors [ND 12][NC 7] notification queue overflow"
                  " on model module.neff. Too many notifications generated during execution.")
            print("ERROR  NMGR:tpb_xu_step Failed to progress exec state, Execution Unit[3101]"
                  " encountered fatal error [NRT_EXEC_SW_NQ_OVERFLOW].")
            sys.exit(1)
        if any(n.exists() for n in ntffs):
            sys.exit(3)  # the failed capture's partial NTFFs are still there
    t0 = time.time()
    time.sleep(float(os.environ.get("FAKE_CAPTURE_S", 0)))
    if os.environ.get("SPANS"):
        with open(os.environ["SPANS"], "a") as f:
            f.write(f"{{t0}} {{time.time()}}\\n")
    for e in (4, 8):
        Path(opt("-s")).with_name(f"profile_rank_0_exec_{{e}}.ntff").write_text("ntff")
    sys.exit(0)
fmt = opt("--output-format")
if fmt == "summary-json":
    print(json.dumps({{"device_profile": {{"total_time": 0.01}}}}))
elif fmt == "json":
    names, events = {{}}, []
    for k, neff in enumerate(sorted(Path(opt("-d")).rglob("neff_*.neff"))):
        mid = int(neff.name.split("_")[1])
        names[str(mid)] = "module." + neff.name
        for phase, t0 in ((0, 1e9), (1, 30e9)):
            for core in (4, 5):
                events.append({{"name": "nc_exec_running", "model_id": mid, "exec_id": 2 * k + phase,
                               "timestamp": int(t0 + k * 1e6), "duration": 500000,
                               "device_core_idx": core, "worker_gid": 0}})
    Path(opt("--output-file")).write_text(json.dumps({{"device_names": [names], "trace_event": events}}))
elif fmt == "parquet":
    import pandas as pd
    marker = Path(opt("-n")).read_bytes()[:1024].rstrip(b"\\0").decode()
    rows = [(layer, f"{{src}}:120") for layer, src in MARKERS[marker.rstrip("0123456789")][1]]
    out = Path(opt("--output-file"))
    out.mkdir(parents=True)
    pd.DataFrame(rows, columns=["layer", "nki_source_location"]).to_parquet(out / "Instruction.parquet")
"""

#: fuser: busy (exit 0, a pid printed) on its FUSER_BUSY_AT-th call, idle (exit 1) otherwise.
FAKE_FUSER = """#!/usr/bin/env bash
echo "$*" >> "$FUSER_LOG"
n=$(wc -l < "$FUSER_LOG")
if [ "$n" = "${FUSER_BUSY_AT:-0}" ]; then echo " 4242"; exit 0; fi
exit 1
"""

#: One kind per key whose rows the "block" marker gives (the real file's entries come from
#: real parquets; the HV1 dual-stream test reads the real file).
TEST_SIGNATURES = {key: {"block": {"require": ["Attn[attn]", "Mlp[ff]"], "attention_kernel": True}}
                   for key in TABLE}

needs_tools = pytest.mark.skipif(not (shutil.which("bash") and shutil.which("taskset")),
                                 reason="capture.sh runs its commands under taskset")


def _dump(key: str) -> list[dict[str, str]]:
    """Every rank holds the key's block NEFFs (one model id each)."""
    return [{str(100 + i): f"block{i}" for i in range(TABLE[key][3])} for _ in range(4)]


def _capture(root: Path, key: str, arm: str = "bf16", device: int = 1, dump=None,
             signatures: dict | None = TEST_SIGNATURES, out: str | None = None,
             extra: tuple[str, ...] = (), **env: str):
    """Run capture.sh with the fakes under ``root`` (its working directory; ``--out`` defaults
    to ``root/out``, absolute; ``extra`` arguments appended); returns (process, arm dir, PY
    lines, neuron-explorer lines, fuser lines)."""
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name, text in (("neuron-explorer", FAKE_X.format(exe=sys.executable)),
                       ("fuser", FAKE_FUSER), ("py", FAKE_PY.format(exe=sys.executable, env=ENV))):
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    logs = {k: root / f"{k}.jsonl" for k in ("PY_LOG", "X_LOG", "FUSER_LOG")}
    full = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "PY": str(bin_dir / "py"),
            "FAKE_DUMP": json.dumps(dump or _dump(key)), **{k: str(v) for k, v in logs.items()},
            "CAPTURE_LOCK": str(root / "capture.lock"), "DUMP_SLOT": str(root / "dump-slot"),
            **env}
    if signatures is not None:
        (root / "signatures.json").write_text(json.dumps(signatures))
        full["SIGNATURES"] = str(root / "signatures.json")
    cpu = str(min(os.sched_getaffinity(0)))
    done = subprocess.run(["bash", str(PD / "capture.sh"), "--key", key, "--arm", arm,
                           "--device", str(device), "--cpus", cpu, "--out", out or str(root / "out"),
                           *extra],
                          env=full, cwd=root, capture_output=True, text=True, timeout=300)

    def lines(k):
        p = logs[k]
        return [json.loads(ln) if k != "FUSER_LOG" else ln
                for ln in p.read_text().splitlines()] if p.exists() else []
    return done, root / "out" / key / arm, lines("PY_LOG"), lines("X_LOG"), lines("FUSER_LOG")


def _flag(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else None


def _status(arm_dir: Path) -> str:
    return (arm_dir / "STATUS").read_text().splitlines()[0]


@needs_tools
@pytest.mark.parametrize("key", TABLE)
def test_each_key_dumps_one_step_at_its_shape_and_captures_its_blocks_as_four_workers(
        tmp_path, key):
    height, width, frames, blocks = TABLE[key]
    # one step's sigma is NaN for the schedulers with a shift_terminal (capture.sh's key_for)
    steps = "2" if key in ("qwen_image", "ltx_2") else "1"
    arms = ("bf16",) if key == "hv15_official" else ("bf16", "mxfp8")
    for arm in arms:
        done, arm_dir, py, x, fuser = _capture(tmp_path / arm, key, arm, device=3)
        assert done.returncode == 0, done.stderr
        assert _status(arm_dir) == "DONE"
        [gen] = py
        argv = gen["argv"]
        assert argv[0] == "generate" and _flag(argv, "--steps") == steps
        assert (_flag(argv, "--height"), _flag(argv, "--width"), _flag(argv, "--num-frames")) == (
            height, width, frames)
        assert ("--quant" in argv) == (arm == "mxfp8") and _flag(argv, "--quant") in (None, "mxfp8")
        cache = "difflet-cache-fp8" if key == "qwen_image" else "difflet-cache-video"  # R7
        assert Path(_flag(argv, "--cache-dir")).name == cache
        assert gen["env"] == {"NEURON_RT_VISIBLE_CORES": "12-15",
                              "NEURON_RT_INSPECT_DEVICE_PROFILE": "1"}
        captures = [ln["argv"] for ln in x if ln["argv"][0] == "capture"]
        assert len(captures) == blocks
        # capture 1200 s; each block view 2700 s (ruling R8: HV1.5's took 481-555 s); rank 0's
        # system view 900 s; each killed 60 s after its signal if it ignores it (ruling R9)
        assert {tuple(ln["parent"]) for ln in x if ln["argv"][0] == "capture"} == {
            ("timeout", "-k", "60", "1200")}
        block_views = [ln for ln in x if ln["argv"][0] == "view" and "-n" in ln["argv"]]
        assert len(block_views) == 3 * blocks
        assert {tuple(ln["parent"]) for ln in block_views} == {("timeout", "-k", "60", "2700")}
        [system_view] = [ln for ln in x if ln["argv"][0] == "view" and "-d" in ln["argv"]]
        assert tuple(system_view["parent"]) == ("timeout", "-k", "60", "900")
        # HV1 without the DGE notifications from the start (ruling R23: its 225-235 MB blocks
        # overflow the notification queue with them); every other key with them
        dge = key != "hunyuan_video_L"
        for c in captures:
            assert {"--collectives-worker-count=4", "--collectives-workers-per-node=4",
                    "--num-exec=8", "--profile-nth-exec=4"} <= set(c)
            assert ("--enable-dge-notifs" in c) == dge
        assert fuser == ["/dev/neuron3"] * (1 + blocks)  # before the dump and each capture
        for i in range(1, blocks + 1):
            blk = arm_dir / f"blk{i}"
            for name in ("summary.json", "summary_exec8.json", "pq/trace/Instruction.parquet",
                         "source.json"):
                assert (blk / name).exists(), (arm, i, name)
            source = json.loads((blk / "source.json").read_text())
            assert len(source["ranks"]) == (2 if key == "hv15_official" else 4)
            assert source["neff_name"] == Path(source["ranks"][0]).name
            assert (source["per_rank"], source["dge_notifs"]) == (False, dge)
        summary = json.loads((arm_dir / "system_summary.json").read_text())
        assert summary["step"]["steps"] == int(steps)
        assert sorted(summary["blocks"]) == sorted(
            json.loads((arm_dir / f"blk{i}" / "source.json").read_text())["neff_name"]
            for i in range(1, blocks + 1))


@needs_tools
def test_a_neff_only_rank_0_holds_is_not_captured(tmp_path):
    dump = _dump("flux")
    dump[0]["999"] = "block9+512"  # model code, the largest NEFF of the dump, on rank 0 only
    done, arm_dir, _, x, _ = _capture(tmp_path, "flux", dump=dump)
    assert done.returncode == 0, done.stderr
    assert _status(arm_dir) == "DONE"
    assert len([ln for ln in x if ln["argv"][0] == "capture"]) == 2
    kept = sorted((arm_dir / f"blk{i}" / "block.neff").read_bytes()[:6] for i in (1, 2))
    assert kept == [b"block0", b"block1"] and not (arm_dir / "blk3").exists()


@needs_tools
def test_neffs_on_every_rank_that_are_not_model_code_are_not_candidates(tmp_path):
    """FLUX's VAE pieces run on every rank, larger than its blocks: one names the VAE's code,
    another no source file at all (2026-11-13 dump)."""
    dump = [{**ranks, "200": "vae+512", "300": "bare+512"} for ranks in _dump("flux")]
    done, arm_dir, _, x, _ = _capture(tmp_path, "flux", dump=dump)
    assert done.returncode == 0, done.stderr
    assert _status(arm_dir) == "DONE"
    assert len([ln for ln in x if ln["argv"][0] == "capture"]) == 2
    assert (arm_dir / "lane.log").read_text().count("not model code") == 2


@needs_tools
def test_a_count_other_than_the_tables_fails_before_any_capture(tmp_path):
    done, arm_dir, _, x, _ = _capture(tmp_path, "qwen_image", dump=_dump("flux"))
    assert done.returncode != 0
    assert _status(arm_dir).startswith("FAILED candidates")
    assert not [ln for ln in x if ln["argv"][0] == "capture"]


@needs_tools
def test_a_candidate_whose_rows_come_from_the_vae_is_rejected(tmp_path):
    """Against the real signatures.json: the second NEFF passes model_code and carries the HV1
    dual-stream roots and attention_cte rows too, so only its dominant source location (the
    VAE's) rejects it."""
    dump = [{"100": "dual+64", "200": "vaerows"} for _ in range(4)]
    done, arm_dir, *_ = _capture(tmp_path, "hunyuan_video_L", dump=dump, signatures=None)
    assert done.returncode != 0
    status = _status(arm_dir)
    assert status.startswith("FAILED signature blk2") and "vae/modeling_vae.py" in status
    assert json.loads((arm_dir / "blk1" / "source.json").read_text())["kind"] == "dual"
    assert not (arm_dir / "blk2" / "source.json").exists()


@needs_tools
@pytest.mark.parametrize("busy_at, captures", [(1, 0), (3, 1)])
def test_a_busy_device_writes_busy_and_runs_nothing_more_on_it(tmp_path, busy_at, captures):
    """Busy before the dump (fuser's first call) or before the second capture (its third)."""
    done, arm_dir, py, x, _ = _capture(tmp_path, "flux", FUSER_BUSY_AT=str(busy_at))
    assert done.returncode == 0, done.stderr
    assert _status(arm_dir) == "BUSY"
    assert len(py) == (busy_at > 1)
    device = [ln["argv"][0] for ln in x if ln["argv"][0] in ("inspect", "capture")]
    assert device == ["inspect"] * (busy_at > 1) + ["capture"] * captures
    assert not (arm_dir / "blk2" / "summary.json").exists()
    assert "4242" in (arm_dir / "lane.log").read_text()


@needs_tools
def test_two_lanes_never_capture_at_the_same_time(tmp_path):
    """Ruling R17: one `neuron-explorer capture` at a time host-wide among our lanes (a capture
    lock); two FLUX lanes start together, each capture takes 0.5 s."""
    env = {"CAPTURE_LOCK": str(tmp_path / "capture.lock"), "SPANS": str(tmp_path / "spans"),
           "FAKE_CAPTURE_S": "0.5"}
    with ThreadPoolExecutor(2) as pool:
        lanes = list(pool.map(lambda d: _capture(tmp_path / d, "flux", device=int(d), **env),
                              ["1", "2"]))
    for done, arm_dir, *_ in lanes:
        assert _status(arm_dir) == "DONE", done.stderr
    spans = sorted(tuple(map(float, ln.split()))
                   for ln in (tmp_path / "spans").read_text().splitlines())
    assert len(spans) == 4
    assert all(b[0] >= a[1] for a, b in zip(spans, spans[1:])), spans


def _max_overlap(spans: list[tuple[float, float]]) -> int:
    edges = sorted([(a, 1) for a, _ in spans] + [(b, -1) for _, b in spans])  # ends first on ties
    running = top = 0
    for _, step in edges:
        running += step
        top = max(top, running)
    return top


@needs_tools
def test_at_most_four_heavy_dumps_run_at_once_and_light_ones_never_wait(tmp_path):
    """Ruling R20: a compile inside a heavy key's dump (HV1 544x960x129, Wan 81f, HV1.5 121f)
    must not run 13-wide. Seven heavy lanes and two light ones start together, each dump 2 s;
    the slots are held for the dump only."""
    heavy = [(k, a) for k in ("hunyuan_video_L", "wan21_official", "wan22_official")
             for a in ("bf16", "mxfp8")] + [("hv15_official", "bf16")]
    light = [("qwen_image", "bf16"), ("flux", "bf16")]
    env = {"DUMP_SLOT": str(tmp_path / "dump-slot"), "DUMP_SPANS": str(tmp_path / "dumps"),
           "FAKE_DUMP_S": "2"}
    lanes = heavy + light
    with ThreadPoolExecutor(len(lanes)) as pool:
        results = list(pool.map(lambda i: _capture(tmp_path / str(i), *lanes[i], device=i + 1, **env),
                                range(len(lanes))))
    for done, arm_dir, *_ in results:
        assert _status(arm_dir) == "DONE", done.stderr
    spans = {}
    for ln in (tmp_path / "dumps").read_text().splitlines():
        d = json.loads(ln)
        spans[tuple(Path(d["out"]).parts[-3:-1])] = (d["t0"], d["t1"])
    assert _max_overlap([spans[lane] for lane in heavy]) == 4
    first_heavy_end = min(spans[lane][1] for lane in heavy)
    assert all(spans[lane][0] < first_heavy_end for lane in light)


@needs_tools
def test_the_lanes_own_leftover_processes_are_killed_before_a_device_command(tmp_path):
    arm_dir = tmp_path / "out" / "qwen_image" / "bf16"
    orphan = subprocess.run(["bash", "-c", "sleep 300 > /dev/null 2>&1 & echo $!"], text=True,
                            capture_output=True, env={**os.environ, "DIFFLET_PROF_LANE": str(arm_dir)})
    pid = int(orphan.stdout)
    done, arm_dir, *_ = _capture(tmp_path, "qwen_image")
    assert done.returncode == 0, done.stderr
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@needs_tools
@pytest.mark.parametrize("pid, status", [("1000", "FAILED system profile"), ("1004", "DONE")])
def test_dropped_system_events_fail_the_lane_only_on_rank_0(tmp_path, pid, status):
    """A full ring buffer drops system-profile events: on rank 0 (pid 1000 here) its step counts
    would come out short; HV1.5's VAE decoder process, after the ranks, dropped 6.4 M on
    2026-11-13 without touching them."""
    done, arm_dir, *_ = _capture(tmp_path, "qwen_image", FAKE_DROPPED_PID=pid)
    assert _status(arm_dir).startswith(status), done.stderr


@needs_tools
@pytest.mark.parametrize("steps, status", [(1, "DONE"), (2, "FAILED system profile: ")])
def test_a_dump_that_ran_another_step_count_than_the_lane_fails_before_any_capture(
        tmp_path, steps, status):
    """Ruling R9: the per-step numbers divide by the lane's step count; HunyuanVideo-1.5's
    orchestrator prints the steps it ran (line format of the 2026-11-13 smoke's dump.log)."""
    line = f"[hv15] request 0: denoise 162.5 s, {steps} steps, median 162527 ms (first 162527 ms)"
    done, arm_dir, _, x, _ = _capture(tmp_path, "hv15_official", FAKE_STEP_LINE=line)
    assert _status(arm_dir).startswith(status), done.stderr
    if status == "DONE":
        summary = json.loads((arm_dir / "system_summary.json").read_text())
        assert summary["step"]["dump_steps"] == 1
    else:
        assert "2 steps" in _status(arm_dir)
        assert not [ln for ln in x if ln["argv"][0] == "capture"]


@needs_tools
def test_a_notification_queue_overflow_is_captured_again_without_dge_notifs(tmp_path):
    """Ruling R22: HV1's 235 MB dual block overflowed the runtime's notification queue with
    --enable-dge-notifs (NRT_EXEC_SW_NQ_OVERFLOW, 2026-11-13); no runtime setting raises that
    queue (NEURON_RT_PROFILE_BUF_DMA_MB=1024 overflowed the same), so the block is captured again
    without the flag, its partial NTFFs deleted first, and source.json says so. HV1 now starts
    without it (ruling R23); the retry stays for any other key, here FLUX's two blocks."""
    done, arm_dir, _, x, _ = _capture(tmp_path, "flux", FAKE_NQ_OVERFLOW="1")
    assert done.returncode == 0, done.stderr
    assert _status(arm_dir) == "DONE"
    dge = ["--enable-dge-notifs" in ln["argv"] for ln in x if ln["argv"][0] == "capture"]
    assert dge == [True, False, True, False]
    for i in (1, 2):
        blk = arm_dir / f"blk{i}"
        assert json.loads((blk / "source.json").read_text())["dge_notifs"] is False
        assert "NRT_EXEC_SW_NQ_OVERFLOW" in (blk / "capture_dge.log").read_text()
        assert (blk / "profile_rank_0_exec_4.ntff").read_text() == "ntff"


def _opts(argv: list[str]) -> dict[str, str]:
    return dict(a.split("=", 1) for a in argv if a.startswith("--") and "=" in a)


@needs_tools
def test_neffs_that_differ_per_rank_are_replayed_one_worker_per_rank(tmp_path):
    """Ruling R21: LTX-2 and Wan compile another block NEFF on every rank (2026-11-13 run, ltx_2
    bf16: 6,105,602 / 6,103,355 / 6,103,814 / 6,103,754 B, four sha256). The replay runs one
    process per worker, worker r on rank r's NEFF and logical core 4d+r, one root communicator;
    worker 0 is profiled and analyzed."""
    dump = [{str(100 + r): f"block{r}"} for r in range(4)]
    done, arm_dir, _, x, _ = _capture(tmp_path, "ltx_2", device=3, dump=dump)
    assert done.returncode == 0, done.stderr
    assert _status(arm_dir) == "DONE"
    captures = sorted((ln for ln in x if ln["argv"][0] == "capture"),
                      key=lambda ln: int(_opts(ln["argv"])["--collectives-worker-start-id"]))
    assert len(captures) == 4
    assert len({c["comm"] for c in captures}) == 1 and captures[0]["comm"]
    for r, c in enumerate(captures):
        assert _opts(c["argv"]) | {"--enable-dge-notifs": None} == {
            "--collectives-worker-count": "4", "--collectives-workers-per-node": "1",
            "--collectives-worker-start-id": str(r), "--collectives-profile-id": "0",
            "--num-exec": "8", "--profile-nth-exec": "4", "--enable-dge-notifs": None}
        assert _flag(c["argv"], "-n") == f"rank{r}.neff" and c["cores"] == str(12 + r)
        assert tuple(c["parent"]) == ("timeout", "-k", "60", "1200")
    blk = arm_dir / "blk1"
    for r in range(4):
        assert (blk / f"rank{r}.neff").read_bytes()[:6] == f"block{r}".encode()
    assert (blk / "block.neff").read_bytes() == (blk / "rank0.neff").read_bytes()
    source = json.loads((blk / "source.json").read_text())
    assert [Path(p).parts[-3] for p in source["ranks"]] == [f"i-0test_pid_{1000 + r}" for r in range(4)]
    assert source["per_rank"] is True and len(set(source["sha256_by_rank"])) == 4
    assert source["sha256"] == source["sha256_by_rank"][0]
    summary = json.loads((arm_dir / "system_summary.json").read_text())
    assert summary["blocks"] == [source["neff_name"]] == ["neff_100_vnc_0.neff"]


@needs_tools
def test_per_rank_roles_are_matched_by_size(tmp_path):
    """Two block kinds per rank, every NEFF its own: the larger of each rank is one role."""
    dump = [{str(100 + r): f"block{r}+512", str(200 + r): f"block{4 + r}"} for r in range(4)]
    done, arm_dir, _, x, _ = _capture(tmp_path, "flux", dump=dump)
    assert done.returncode == 0, done.stderr
    assert _status(arm_dir) == "DONE"
    assert len([ln for ln in x if ln["argv"][0] == "capture"]) == 8
    for i, first in ((1, 0), (2, 4)):
        for r in range(4):
            assert (arm_dir / f"blk{i}" / f"rank{r}.neff").read_bytes()[:6] == f"block{first + r}".encode()


@needs_tools
@pytest.mark.parametrize("change, reason", [
    ({2: {"999": "block9"}}, "rank 2"),       # a rank with another count
    ({1: {"101": "block1+512"}}, "sizes"),    # one role's sizes 50% apart
])
def test_per_rank_candidates_that_do_not_line_up_fail_before_any_capture(tmp_path, change, reason):
    dump = [{str(100 + r): f"block{r}"} for r in range(4)]
    for r, neffs in change.items():
        dump[r] = {**({} if "101" in neffs else dump[r]), **neffs}
    done, arm_dir, _, x, _ = _capture(tmp_path, "ltx_2", dump=dump)
    assert _status(arm_dir).startswith("FAILED candidates") and reason in _status(arm_dir)
    assert not [ln for ln in x if ln["argv"][0] == "capture"]


@needs_tools
def test_candidates_mode_prints_the_selection_of_a_kept_dump_and_runs_nothing(tmp_path):
    """The device-free check of a lane's kept dump (`--candidates`): no device, no STATUS."""
    dump = [{str(100 + r): f"block{r}"} for r in range(4)]
    done, arm_dir, py, x, fuser = _capture(tmp_path, "ltx_2", dump=dump)
    assert _status(arm_dir) == "DONE"
    lane_log = (arm_dir / "lane.log").read_text()
    done, _, py2, x2, fuser2 = _capture(tmp_path, "ltx_2", dump=dump, extra=("--candidates",))
    assert done.returncode == 0, done.stderr
    [line] = done.stdout.splitlines()
    size, sha, mode, *paths = line.split()
    assert mode == "rank" and [Path(p).parts[-3] for p in paths] == [
        f"i-0test_pid_{1000 + r}" for r in range(4)]
    assert (py2, x2, fuser2) == (py, x, fuser)  # nothing run
    assert _status(arm_dir) == "DONE" and (arm_dir / "lane.log").read_text() == lane_log


@needs_tools
def test_hv15_profiles_the_second_cfg_half_when_its_neff_differs(tmp_path):
    dump = [{"100": "block0"}, {"100": "block0"}, {"300": "block1"}, {"300": "block1"}]
    done, arm_dir, _, x, _ = _capture(tmp_path, "hv15_official", dump=dump)
    assert done.returncode == 0, done.stderr
    assert _status(arm_dir) == "DONE"
    assert len([ln for ln in x if ln["argv"][0] == "capture"]) == 2
    ranks = [json.loads((arm_dir / f"blk{i}" / "source.json").read_text())["ranks"] for i in (1, 2)]
    assert [[Path(r).parts[-3] for r in rs] for rs in ranks] == [
        ["i-0test_pid_1000", "i-0test_pid_1001"], ["i-0test_pid_1002", "i-0test_pid_1003"]]
    # system_summary covers rank 0, which runs only its own group's NEFF
    summary = json.loads((arm_dir / "system_summary.json").read_text())
    assert summary["blocks"] == ["neff_100_vnc_0.neff"]


@needs_tools
def test_a_relative_out_is_the_callers_not_the_repos(tmp_path):
    """The dump runs from the checkout: a relative --out taken as given would put the dump, the
    work dir and the NTFFs in the repo, on the root disk, and leave the lane no rank dumps."""
    done, arm_dir, py, _, _ = _capture(tmp_path, "qwen_image", out="out")
    assert done.returncode == 0, done.stderr
    assert _status(arm_dir) == "DONE"
    assert Path(_flag(py[0]["argv"], "--work-dir")) == arm_dir / "work"
    assert not (REPO / "out").exists()


@needs_tools
def test_a_rerun_skips_the_dump_and_the_captures_already_done(tmp_path):
    done, arm_dir, _, x, _ = _capture(tmp_path, "qwen_image")
    assert done.returncode == 0 and _status(arm_dir) == "DONE"
    first = len(x)
    done, arm_dir, _, x, _ = _capture(tmp_path, "qwen_image")
    assert done.returncode == 0, done.stderr
    assert _status(arm_dir) == "DONE"
    assert not [ln for ln in x[first:] if ln["argv"][0] in ("inspect", "capture")]


def test_hv15_mxfp8_is_refused_naming_the_pair(tmp_path):
    done = subprocess.run(["bash", str(PD / "capture.sh"), "--key", "hv15_official", "--arm",
                           "mxfp8", "--device", "1", "--cpus", "0", "--out", str(tmp_path / "out")],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 2
    assert "hv15_official mxfp8" in done.stderr
    assert not (tmp_path / "out").exists()


def _loop_profile(tmp_path: Path) -> tuple[Path, Path]:
    """A warm-up forward, 1 s of prompt encoding, then two forwards 12 ms apart (one step's CFG
    pair, or two steps), as Qwen-Image's 2-step dump; every execution recorded on both physical
    cores of the logical core, the run's first without an exec_id (as in HV1's profile). The
    loop's blocks: 2032-2067.001 and 2082-2117.001 ms. Returns (system.json, rank 0's dump)."""
    events, exec_id = [], 0

    def run(model_id, start_ms, dur_ms):
        nonlocal exec_id
        exec_id += 1
        for core, skew in ((4, 0), (5, 1000)):  # the second core ends 1 us later
            e = {"name": "nc_exec_running", "model_id": model_id, "exec_id": exec_id,
                 "flow_id": [str(exec_id)], "timestamp": int(start_ms * 1e6),
                 "duration": int(dur_ms * 1e6) + skew, "device_core_idx": core, "worker_gid": 0}
            if exec_id == 1:
                del e["exec_id"]
            events.append(e)
    run(33, 10.0, 1.0)  # a NEFF of the load, before the warm-up
    t = 1000.0
    for _ in range(3):  # the warm-up forward, ends at 1029.001
        run(11, t, 9.0)
        t += 10
    t = 2030.0  # 1 s later the loop: per forward a prefix NEFF and three blocks
    for _ in range(2):
        run(22, t, 1.0)
        t += 2
        for dur in (10.0, 12.0, 11.0):
            run(11, t, dur)
            t += dur + 1
        t += 12
    events.append({"name": "cc_running", "timestamp": int(2040e6), "duration": 500000})
    events.append({"name": "cc_running", "timestamp": int(1005e6), "duration": 500000})  # warm-up
    profile = tmp_path / "system.json"
    profile.write_text(json.dumps({"device_names": [{"11": "m.b", "22": "m.p"}],
                                   "trace_event": events}))
    rank0 = tmp_path / "rank0" / "17"
    rank0.mkdir(parents=True)
    for mid in (11, 22):
        (rank0 / f"neff_{mid}_vnc_0.neff").write_bytes(b"x")
    return profile, tmp_path / "rank0"


def _system_profile(profile: Path, rank0: Path, steps: int, log: str = ""):
    """system_profile.py on ``profile`` with a dump.log holding ``log``."""
    dump_log = profile.with_name("dump.log")
    dump_log.write_text(log + "\ndump exit 0\n")
    return subprocess.run([sys.executable, str(PD / "system_profile.py"), str(profile), str(rank0),
                           str(steps), str(dump_log), "neff_11_vnc_0.neff"],
                          capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("steps", [1, 2])
def test_system_profile_counts_the_step_not_the_warmup(tmp_path, steps):
    done = _system_profile(*_loop_profile(tmp_path), steps)
    assert done.returncode == 0, done.stderr
    s = json.loads(done.stdout)
    block, prefix = s["neffs"]["neff_11_vnc_0.neff"], s["neffs"]["neff_22_vnc_0.neff"]
    # the window runs from the loop's first block to its last: the first forward's prefix
    # NEFF falls before it
    assert block["execs_per_step"] == 6 // steps and prefix["execs_per_step"] == 1 / steps
    assert block["in_model_median_ms"] == pytest.approx(11.001)
    assert s["non_block_ms"] == pytest.approx(1.001 / steps)
    # 0.999 ms after each block but a forward's last and after the prefix; 12.999 ms between
    # the forwards
    assert s["host_gap_ms"] == pytest.approx((5 * 0.999 + 12.999) / steps)
    # the loop: the first block's start to the last one's end; the dump printed no step count
    assert s["step"] == {"wall_ms": pytest.approx((2117.001 - 2032) / steps), "steps": steps,
                         "gap_s": 0.2, "dump_steps": None}
    assert s["cc_running"] == {"n": 1 / steps, "total_ms": pytest.approx(0.5 / steps)}
    assert s["blocks"] == ["neff_11_vnc_0.neff"]


#: The dump's own step lines (ruling R9): HunyuanVideo-1.5's orchestrator prints the steps it ran;
#: the torchrun and FLUX orchestrators print the median of the step intervals and their count
#: (steps - 1), only when there are two steps or more (Qwen-Image's 2-step smoke, 2026-11-13).
HV15_LINE = "[hv15] request 0: denoise 162.5 s, {n} steps, median 162527 ms (first 162527 ms)"
MEDIAN_LINE = ("[qwen_image] request: 0.84 s; to the first step's end 0.53 s (prompt encode + "
               "step 1); step median {ms} ms (n={k}, max {ms}); after the last step 0.00 s")


@pytest.mark.parametrize("steps, log, error", [
    (2, HV15_LINE.format(n=2), None),
    (1, HV15_LINE.format(n=2), "the dump ran 2 steps"),
    (2, MEDIAN_LINE.format(ms=50.0, k=1), None),
    (1, MEDIAN_LINE.format(ms=50.0, k=1), "the dump ran 2 steps"),
    # 1 x 90 ms between the two steps' ends, the window 85 ms: a step lies outside it
    (2, MEDIAN_LINE.format(ms=90.0, k=1), "outside the loop window"),
])
def test_system_profile_cross_checks_the_dumps_own_step_count(tmp_path, steps, log, error):
    done = _system_profile(*_loop_profile(tmp_path), steps, log)
    if error is None:
        assert done.returncode == 0, done.stderr
        assert json.loads(done.stdout)["step"]["dump_steps"] == 2
    else:
        assert done.returncode != 0 and error in done.stderr, done.stderr


def test_system_profile_refuses_a_profile_that_is_not_rank_0s(tmp_path):
    profile = tmp_path / "system.json"
    profile.write_text(json.dumps({"device_names": [{}], "trace_event": [
        {"name": "nc_exec_running", "model_id": 11, "exec_id": 1, "timestamp": 0,
         "duration": 1, "device_core_idx": 6, "worker_gid": 1}]}))
    (tmp_path / "rank0").mkdir()
    done = _system_profile(profile, tmp_path / "rank0", 1)
    assert done.returncode != 0 and "worker_gid" in done.stderr
