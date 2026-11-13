"""scripts/profile_dit/run_all.sh: the 13 lanes of the DiT block profile's device run, scheduled
on idle devices, driven with a fake capture.sh (a lane's outcome scripted per run), a fake
neuron-explorer, fuser, df and taskset on PATH, a recording PY and a fake HOME holding the
root-disk cache entries the lanes load. No device, no model. A lane on a busy device, two lanes
on one CPU, a lane re-quantizing on the root disk or a run that never ends costs a device run, so
those are pinned here."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PD = REPO / "scripts" / "profile_dit"
LANES = [(k, a) for k in ("qwen_image", "flux", "ltx_2", "wan21_official", "wan22_official",
                          "hunyuan_video_L") for a in ("bf16", "mxfp8")]
LANES.append(("hv15_official", "bf16"))
BLOCKS = {"flux": 2, "hunyuan_video_L": 2}
#: The lanes whose cache entries are keyed by the root hub's files (run_all.sh's hub_of): Qwen's
#: quantized digest by its blob names (the NVMe copy has plain files), Wan 2.2's bf16 copies by
#: their root source; LTX-2 and Wan 2.1 are only on root.
ROOT_HUB = {("qwen_image", "mxfp8"), *((k, a) for k in ("ltx_2", "wan21_official", "wan22_official")
                                      for a in ("bf16", "mxfp8"))}
NVME_HUB = "/mnt/nvme/difflet-quant-bench-hub"

#: capture.sh: one JSON line per run to CAPTURE_LOG (lane, device, cpus, the hub and trace-cache
#: environment, the dump's start and end); the lane's n-th run ends as FAKE_OUTCOMES {"<key>
#: <arm>": [outcome, ...]} says (default DONE), after FAKE_LANE_S seconds of "dump"; outcome
#: "EXIT<rc>" exits rc with no STATUS written. The DISK_DROP_LANE lane's start creates DF_LOW_FILE.
FAKE_CAPTURE = """#!{exe}
import json, os, sys, time
from pathlib import Path
a = dict(zip(sys.argv[1::2], sys.argv[2::2]))
lane_id = a["--key"] + " " + a["--arm"]
lane = Path(a["--out"], a["--key"], a["--arm"])
lane.mkdir(parents=True, exist_ok=True)
log = Path(os.environ["CAPTURE_LOG"])
runs = log.read_text().splitlines() if log.exists() else []
n = sum(json.loads(ln)["lane"] == lane_id for ln in runs)
outcome = (json.loads(os.environ.get("FAKE_OUTCOMES", "{{}}")).get(lane_id, [])[n:] or ["DONE"])[0]
t0 = time.time()
if os.environ.get("DISK_DROP_LANE") == lane_id:
    Path(os.environ["DF_LOW_FILE"]).touch()
if outcome.startswith("EXIT"):
    with open(log, "a") as f:
        f.write(json.dumps({{"lane": lane_id, "device": int(a["--device"]), "t0": t0}}) + "\\n")
    sys.exit(int(outcome[4:]))
if outcome != "BUSY":
    (lane / "dump.log").write_text("loading\\n")
    time.sleep(float(os.environ.get("FAKE_LANE_S", "0")))
    (lane / "dump.log").write_text("loading\\ndump exit 0\\n")
    for i in range(1, {blocks!r}.get(a["--key"], 1) + 1):
        (lane / f"blk{{i}}").mkdir(exist_ok=True)
with open(log, "a") as f:
    f.write(json.dumps({{"lane": lane_id, "device": int(a["--device"]), "cpus": a["--cpus"],
                        "hub": os.environ.get("HF_HUB_CACHE"),
                        "trace": os.environ.get("DIFFLET_NKI_TRACE_CACHE"),
                        "t0": t0, "t1": time.time()}}) + "\\n")
(lane / "STATUS").write_text(outcome + "\\n")
"""

#: neuron-explorer: `capture` (logged with its time and cores) writes the two NTFFs; `view`
#: prints execution 4's or 8's summary.
FAKE_X = """#!{exe}
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["X_LOG"], "a") as f:
    f.write(json.dumps({{"argv": args, "t": time.time(), "cwd": os.getcwd(),
                        "cores": os.environ.get("NEURON_RT_VISIBLE_CORES")}}) + "\\n")
opt = lambda name: args[args.index(name) + 1]
if args[0] == "capture":
    for e in (4, 8):
        Path(opt("-s")).with_name(f"profile_rank_0_exec_{{e}}.ntff").write_text("ntff")
elif args[0] == "view":
    e = 4 if "exec_4" in opt("-s") else 8
    print(json.dumps({{"device_profile_x": {{"total_time": 0.005 + e * 1e-5,
                                            "throttle_avg_util_limit_nc6_percent": 0.8,
                                            "throttle_avg_util_limit_nc7_percent": 0.6}}}}))
"""

#: PY: analyze.py's argv to ANALYZE_LOG (exit FAKE_ANALYZE_RC); anything else runs on python.
FAKE_PY = """#!{exe}
import json, os, sys
if sys.argv[1:2] and sys.argv[1].endswith("analyze.py"):
    with open(os.environ["ANALYZE_LOG"], "a") as f:
        f.write(json.dumps(sys.argv[2:]) + "\\n")
    sys.exit(int(os.environ.get("FAKE_ANALYZE_RC", "0")))
os.execv(sys.executable, [sys.executable] + sys.argv[1:])
"""

FAKE_FUSER = """#!/usr/bin/env bash
echo "$*" >> "$FUSER_LOG"
case " $FUSER_BUSY " in *" ${1#/dev/neuron} "*) echo " 4242"; exit 0 ;; esac
exit 1
"""
#: df: DF_AVAIL GB free on / (default 500), 59 once DF_LOW_FILE exists
FAKE_DF = """#!/usr/bin/env bash
echo " Avail"
if [ -n "${DF_LOW_FILE:-}" ] && [ -e "$DF_LOW_FILE" ]; then echo 59G; else echo "${DF_AVAIL:-500}G"; fi
"""
#: taskset: the CPU list and command to TASKSET_LOG, then runs the command (no affinity)
FAKE_TASKSET = """#!/usr/bin/env bash
echo "$*" >> "$TASKSET_LOG"
shift 2
exec "$@"
"""

#: The root-disk cache entries the lanes load (run_all.sh's lane_files), relative to HOME.
V, W = "difflet-cache-video", "difflet-cache-video/neuron/weights"
ENTRIES = [
    "difflet-cache-fp8/quantized/Qwen--Qwen-Image__75e0b4be/transformer/mxfp8-65db3df4b2e34023",
    f"{V}/quantized/black-forest-labs--FLUX.1-dev__3de623fc/transformer/mxfp8-809df193b011ba1f",
    f"{V}/quantized/Lightricks--LTX-2__64f65d09/transformer/mxfp8-916a153a8eeb2fa3",
    f"{W}/Lightricks--LTX-2__text_encoder__64f65d09__bfloat16__tp4__9aad1a74d1afb88f",
    f"{V}/weights-bf16/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/transformer",
    f"{V}/quantized/Wan-AI--Wan2.1-T2V-14B-Diffusers__38ec498c/transformer/mxfp8-6d28df7cc78386b5",
    f"{W}/Wan-AI--Wan2.1-T2V-14B-Diffusers__text_encoder__38ec498c__bfloat16__tp4__91b0beceaabd499e",
    f"{W}/Wan-AI--Wan2.1-T2V-14B-Diffusers__vae__38ec498c__bfloat16__tp1__fa7540f14397355b",
    f"{V}/weights-bf16/5be7df9619b54f4e2667b2755bc6a756675b5cd7/transformer",
    f"{V}/weights-bf16/5be7df9619b54f4e2667b2755bc6a756675b5cd7/transformer_2",
    f"{V}/quantized/Wan-AI--Wan2.2-T2V-A14B-Diffusers__5be7df96/transformer/mxfp8-372c52a68c440756",
    f"{V}/quantized/Wan-AI--Wan2.2-T2V-A14B-Diffusers__5be7df96/transformer_2/mxfp8-03f6d63f26aab3cf",
    f"{W}/Wan-AI--Wan2.2-T2V-A14B-Diffusers__vae__5be7df96__bfloat16__tp1__c4319fcfc5a85228",
    f"{V}/quantized/hunyuanvideo-community--HunyuanVideo__e8c2aaa6/transformer/mxfp8-46e0952c1c5a9f74",
    f"{W}/hunyuanvideo-community--HunyuanVideo-1.5-Diffusers-480p_t2v__transformer__286be7ce__"
    "bfloat16__tp2__f620128f4bf3c6b4"]
HUB = ".cache/huggingface/hub"
SNAPSHOTS = {
    "models--Qwen--Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6":
        ["text_encoder", "transformer", "vae"],
    "models--Lightricks--LTX-2/snapshots/64f65d09e0b76c96202b5bc09997673184cf9b5d":
        ["connectors", "text_encoder", "transformer", "vae"],
    "models--Wan-AI--Wan2.1-T2V-14B-Diffusers/snapshots/38ec498cb3208fb688890f8cc7e94ede2cbd7f68":
        ["text_encoder", "transformer", "vae"],
    "models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7":
        ["text_encoder", "transformer", "transformer_2", "vae"]}

needs_tools = pytest.mark.skipif(not (shutil.which("bash") and shutil.which("flock")),
                                 reason="run_all.sh runs bash and flock")


def _home(home: Path) -> None:
    for rel in ENTRIES:
        (home / rel).mkdir(parents=True)
        (home / rel / "w.safetensors").write_text("w")
    for snap, comps in SNAPSHOTS.items():
        for c in comps:
            (home / HUB / snap / c).mkdir(parents=True)
            (home / HUB / snap / c / "w.safetensors").write_text("w")
        (home / HUB / snap / "model_index.json").write_text("{}")


def _run_all(tmp_path: Path, *args: str, timeout: int = 120, **env: str):
    """run_all.sh with the fakes (written once per tmp_path); returns (process, run root)."""
    bin_dir, home, root = tmp_path / "bin", tmp_path / "home", tmp_path / "run"
    if not bin_dir.exists():
        bin_dir.mkdir()
        for name, text in (("capture.sh", FAKE_CAPTURE.format(exe=sys.executable, blocks=BLOCKS)),
                           ("neuron-explorer", FAKE_X.format(exe=sys.executable)),
                           ("py", FAKE_PY.format(exe=sys.executable)), ("fuser", FAKE_FUSER),
                           ("df", FAKE_DF), ("taskset", FAKE_TASKSET)):
            (bin_dir / name).write_text(text)
            (bin_dir / name).chmod(0o755)
        _home(home)
    logs = {k: str(tmp_path / f"{k.lower()}") for k in
            ("CAPTURE_LOG", "X_LOG", "ANALYZE_LOG", "FUSER_LOG", "TASKSET_LOG")}
    full = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": str(home),
            "ROOT": str(root), "PY": str(bin_dir / "py"), "CAPTURE": str(bin_dir / "capture.sh"),
            "CAPTURE_LOCK": str(tmp_path / "capture.lock"), "BASELINE_NEFF": "", "POLL": "0.1",
            "FUSER_BUSY": "0 6", **logs, **env}
    done = subprocess.run(["bash", str(PD / "run_all.sh"), *args], env=full, cwd=tmp_path,
                          capture_output=True, text=True, timeout=timeout)
    return done, root


def _jsonl(path: Path) -> list:
    return [json.loads(ln) for ln in path.read_text().splitlines()] if path.exists() else []


def _cpus(spec: str) -> set[int]:
    out = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out |= set(range(int(lo), int(hi or lo) + 1))
    return out


def _status(root: Path, key: str, arm: str) -> str:
    return (root / key / arm / "STATUS").read_text().splitlines()[0]


@needs_tools
def test_every_lane_runs_once_on_its_own_idle_device_cpu_slice_and_hub(tmp_path):
    done, root = _run_all(tmp_path, FAKE_LANE_S="0.5")
    assert done.returncode == 0, done.stderr
    runs = _jsonl(tmp_path / "capture_log")
    assert sorted(r["lane"] for r in runs) == sorted(f"{k} {a}" for k, a in LANES)
    # 14 idle devices for 13 lanes: all at once, never 0 or 6 (busy), never two on one device
    devices = [r["device"] for r in runs]
    assert len(set(devices)) == 13 and not {0, 6} & set(devices)
    for r in runs:
        # the device's 12-thread slice of its NUMA node (devices 0-7: node 1, 48-95,144-191)
        k, base = r["device"] % 8, (48 if r["device"] < 8 else 0)
        assert _cpus(r["cpus"]) == set(range(base + 6 * k, base + 6 * k + 6)) | set(
            range(base + 96 + 6 * k, base + 102 + 6 * k)), r
        key, arm = r["lane"].split()
        assert r["hub"] == (str(tmp_path / "home" / HUB) if (key, arm) in ROOT_HUB else NVME_HUB)
        assert r["trace"] == ("1" if key.startswith("wan") and arm == "bf16" else None), r
    sets = [_cpus(r["cpus"]) for r in runs]
    assert all(not a & b for i, a in enumerate(sets) for b in sets[i + 1:])
    # each DONE lane's blocks analyzed with its key, on its CPU slice
    analyses = _jsonl(tmp_path / "analyze_log")
    assert sorted((Path(a[0]).parent.parent.name, Path(a[0]).parent.name, Path(a[0]).name, a[2])
                  for a in analyses) == sorted(
        (k, a, f"blk{i}", k) for k, a in LANES for i in range(1, BLOCKS.get(k, 1) + 1))
    assert all(a[1] == "--model" for a in analyses)
    by_lane = {tuple(r["lane"].split()): r["cpus"] for r in runs}
    taskset = (tmp_path / "taskset_log").read_text().splitlines()
    for k, a in LANES:
        assert any(ln.startswith(f"-c {by_lane[k, a]} ") and f"/{k}/{a}/blk1" in ln
                   for ln in taskset), (k, a)
    marker = (root / "RUN_DONE").read_text()
    for k, a in LANES:
        assert f"{k} {a}: DONE" in marker


@needs_tools
def test_lanes_queue_behind_the_first_to_finish_when_few_devices_are_idle(tmp_path):
    busy = " ".join(str(d) for d in range(16) if d not in (3, 9, 12))
    done, root = _run_all(tmp_path, FAKE_LANE_S="0.3", FUSER_BUSY=busy)
    assert done.returncode == 0, done.stderr
    runs = _jsonl(tmp_path / "capture_log")
    assert len(runs) == 13 and {r["device"] for r in runs} == {3, 9, 12}
    edges = sorted([(r["t0"], 1) for r in runs] + [(r["t1"], -1) for r in runs])
    running = [sum(e for _, e in edges[:i + 1]) for i in range(len(edges))]
    assert max(running) == 3
    assert all(_status(root, k, a) == "DONE" for k, a in LANES)


@needs_tools
def test_a_busy_lane_is_requeued_on_another_device_never_the_same(tmp_path):
    busy = " ".join(str(d) for d in range(16) if d not in (4, 5))
    done, root = _run_all(tmp_path, FUSER_BUSY=busy,
                          FAKE_OUTCOMES=json.dumps({"flux bf16": ["BUSY", "BUSY", "DONE"]}))
    assert done.returncode == 0, done.stderr
    flux = [r["device"] for r in _jsonl(tmp_path / "capture_log") if r["lane"] == "flux bf16"]
    # BUSY on one of the two idle devices, then on the other; no idle device it has not tried
    # is left, so once nothing else runs the run ends instead of waiting forever
    assert len(flux) == 2 and flux[0] != flux[1]
    assert _status(root, "flux", "bf16") == "BUSY"
    assert "flux bf16: BUSY" in (root / "RUN_DONE").read_text()


@needs_tools
def test_a_lane_busy_once_ends_done_on_the_next_idle_device(tmp_path):
    done, root = _run_all(tmp_path, FAKE_OUTCOMES=json.dumps({"ltx_2 mxfp8": ["BUSY"]}))
    assert done.returncode == 0, done.stderr
    ltx = [r["device"] for r in _jsonl(tmp_path / "capture_log") if r["lane"] == "ltx_2 mxfp8"]
    assert len(ltx) == 2 and ltx[0] != ltx[1]
    assert _status(root, "ltx_2", "mxfp8") == "DONE"
    assert (root / "ltx_2" / "mxfp8" / "busy_devices").read_text().split() == [str(ltx[0])]


@needs_tools
@pytest.mark.parametrize("outcomes", [["EXIT3"], ["BUSY", "EXIT3"]])
def test_a_capture_that_exits_without_a_terminal_status_fails_the_lane(tmp_path, outcomes):
    """Ruling R19 (M3): capture.sh dying before its STATUS (none, or the BUSY of the run before)
    would requeue the lane forever."""
    done, root = _run_all(tmp_path, FAKE_OUTCOMES=json.dumps({"flux bf16": outcomes}))
    assert done.returncode == 0, done.stderr
    assert _status(root, "flux", "bf16") == "FAILED capture.sh exit 3"
    runs = [r for r in _jsonl(tmp_path / "capture_log") if r["lane"] == "flux bf16"]
    assert len(runs) == len(outcomes)
    assert "flux bf16: FAILED capture.sh exit 3" in (root / "RUN_DONE").read_text()


@needs_tools
def test_each_lane_runs_whole_under_its_cpu_slice(tmp_path):
    """Ruling R19 (M5): capture.sh and all it runs (the signature check, system_profile.py)
    inherit the lane's CPU slice."""
    done, root = _run_all(tmp_path)
    assert done.returncode == 0, done.stderr
    taskset = (tmp_path / "taskset_log").read_text().splitlines()
    for r in _jsonl(tmp_path / "capture_log"):
        key, arm = r["lane"].split()
        assert [ln for ln in taskset if ln.startswith(f"-c {r['cpus']} ")
                and str(tmp_path / "bin" / "capture.sh") in ln
                and f"--key {key} --arm {arm} " in ln], r["lane"]


@needs_tools
def test_the_disk_is_checked_every_poll_and_a_drop_stops_new_lanes(tmp_path):
    """Ruling R19 (M8): / falls under 60 GB as flux bf16, the third lane on the three idle devices,
    starts; the rest still queue: none of them starts, one DISK LOW line, the run still ends."""
    low = tmp_path / "df_low"
    busy = " ".join(str(d) for d in range(16) if d not in (3, 9, 12))
    done, root = _run_all(tmp_path, FUSER_BUSY=busy, FAKE_LANE_S="0.3", DF_LOW_FILE=str(low),
                          DISK_DROP_LANE="flux bf16")
    assert done.returncode == 0, done.stderr
    dropped = low.stat().st_mtime
    runs = _jsonl(tmp_path / "capture_log")
    assert sorted(r["lane"] for r in runs) == ["flux bf16", "qwen_image bf16", "qwen_image mxfp8"]
    # mtime is the kernel's coarse clock: a few ms behind time.time(); a later lane would start
    # 0.3 s after the drop at the earliest (when a device frees)
    assert all(r["t0"] <= dropped + 0.05 for r in runs)
    assert {_status(root, k, a) for k, a in LANES} == {"DONE", "FAILED disk"}
    log = (root / "run_all.log").read_text()
    assert len([ln for ln in log.splitlines() if "DISK LOW" in ln]) == 1
    assert (root / "RUN_DONE").exists()


@needs_tools
def test_below_60_gb_on_the_root_disk_no_lane_starts(tmp_path):
    done, root = _run_all(tmp_path, DF_AVAIL="59")
    assert done.returncode == 0, done.stderr
    assert not _jsonl(tmp_path / "capture_log")
    assert all(_status(root, k, a) == "FAILED disk" for k, a in LANES)
    assert (root / "RUN_DONE").exists()


@needs_tools
def test_a_lane_whose_cache_entry_is_missing_fails_before_it_loads(tmp_path):
    """A missing quantized directory would be re-quantized on the root disk (tens of GB)."""
    _run_all(tmp_path, DF_AVAIL="59")  # writes the fakes and HOME, starts nothing
    shutil.rmtree(tmp_path / "run")
    shutil.rmtree(tmp_path / "home" / ENTRIES[1])  # FLUX's quantized transformer
    done, root = _run_all(tmp_path)
    assert done.returncode == 0, done.stderr
    assert _status(root, "flux", "mxfp8").startswith("FAILED cache miss")
    assert ENTRIES[1] in _status(root, "flux", "mxfp8")
    assert "flux mxfp8" not in [r["lane"] for r in _jsonl(tmp_path / "capture_log")]
    assert _status(root, "flux", "bf16") == "DONE"


@needs_tools
def test_the_prewarm_reads_the_lanes_root_files_and_not_what_it_does_not_load(tmp_path):
    done, root = _run_all(tmp_path)
    assert done.returncode == 0, done.stderr
    log = (root / "run_all.log").read_text()
    [line] = [ln for ln in log.splitlines() if "prewarm wan22_official bf16:" in ln
              and "/weights-bf16/" in ln]
    snap = next(s for s in SNAPSHOTS if "Wan2.2" in s)
    for part in ("text_encoder", "vae", "model_index.json"):
        assert f"{snap}/{part}" in line
    for part in ("transformer", "transformer_2"):  # the fp32 experts: the bf16 copies load
        assert f"{snap}/{part} " not in line + " "
        assert f"weights-bf16/5be7df9619b54f4e2667b2755bc6a756675b5cd7/{part}" in line
    assert not [ln for ln in log.splitlines() if "prewarm qwen_image bf16:" in ln and "/" in ln]


@needs_tools
def test_a_restart_skips_finished_lanes_adopts_running_ones_and_redoes_missing_analyses(tmp_path):
    done, root = _run_all(tmp_path)
    assert done.returncode == 0, done.stderr
    first, analyzed = len(_jsonl(tmp_path / "capture_log")), len(_jsonl(tmp_path / "analyze_log"))
    (root / "qwen_image" / "bf16" / "blk1" / "analyze.log").unlink()
    (root / "flux" / "mxfp8" / "STATUS").write_text("FAILED signature blk2: x\n")
    # wan21 bf16 is running under an earlier run_all: its lock is held; it ends DONE in 2 s
    lane = root / "wan21_official" / "bf16"
    (lane / "STATUS").unlink()
    holder = subprocess.Popen(
        ["flock", str(lane / ".lock"), "bash", "-c",
         f"sleep 2; echo DONE > {lane}/STATUS"])
    try:
        done, root = _run_all(tmp_path)
    finally:
        holder.wait(timeout=30)
    assert done.returncode == 0, done.stderr
    assert len(_jsonl(tmp_path / "capture_log")) == first  # nothing captured again
    redone = _jsonl(tmp_path / "analyze_log")[analyzed:]
    assert [(Path(a[0]).parent.parent.name, Path(a[0]).parent.name) for a in redone] == [
        ("qwen_image", "bf16")]
    marker = (root / "RUN_DONE").read_text()
    assert "flux mxfp8: FAILED signature blk2: x" in marker
    assert "wan21_official bf16: DONE" in marker


@needs_tools
def test_the_in_run_baseline_capture_runs_once_while_four_lanes_dump(tmp_path):
    """Ruling R17's third capture of the Qwen-Image smoke block NEFF: on an idle device no lane
    holds, while at least four lanes are in their dump; its execution 4 and 8 times recorded."""
    neff = tmp_path / "smoke.neff"
    neff.write_bytes(b"neff")
    done, root = _run_all(tmp_path, FAKE_LANE_S="3", BASELINE_NEFF=str(neff))
    assert done.returncode == 0, done.stderr
    [capture] = [x for x in _jsonl(tmp_path / "x_log") if x["argv"][0] == "capture"]
    assert {"--collectives-worker-count=4", "--collectives-workers-per-node=4", "--num-exec=8",
            "--profile-nth-exec=4", "--enable-dge-notifs"} <= set(capture["argv"])
    result = json.loads((root / "baseline" / "inrun" / "result.json").read_text())
    assert result["lanes_dumping"] >= 4
    runs = _jsonl(tmp_path / "capture_log")
    assert sum(r["t0"] <= capture["t"] <= r["t1"] for r in runs) >= 4
    d = result["device"]
    assert capture["cores"] == f"{4 * d}-{4 * d + 3}"
    assert not [r for r in runs if r["device"] == d and r["t0"] <= capture["t"] <= r["t1"]]
    assert result["exec4"] == {"total_time_ms": pytest.approx(5.04), "avg_util_limit":
                               pytest.approx(0.7)}
    assert result["exec8"]["total_time_ms"] == pytest.approx(5.08)
    assert "baseline inrun: DONE" in (root / "RUN_DONE").read_text()


@needs_tools
def test_a_lone_baseline_capture_runs_the_same_command_on_the_given_device(tmp_path):
    neff = tmp_path / "smoke.neff"
    neff.write_bytes(b"neff")
    done, root = _run_all(tmp_path, "baseline", "7", "lone1", BASELINE_NEFF=str(neff))
    assert done.returncode == 0, done.stderr
    assert not _jsonl(tmp_path / "capture_log")
    [capture] = [x for x in _jsonl(tmp_path / "x_log") if x["argv"][0] == "capture"]
    assert capture["cores"] == "28-31"
    result = json.loads((root / "baseline" / "lone1" / "result.json").read_text())
    assert (result["device"], result["lanes_dumping"]) == (7, 0)
    assert (tmp_path / "fuser_log").read_text().split() == ["/dev/neuron7"]
