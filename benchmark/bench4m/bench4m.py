"""Four-model native vs AoT TP4 benchmark harness (2026-10-26; not part of the repo).

Phase 2 (2026-10-26): hunyuan_video added (AoT on main). A q* run of it also exports
HV1_DUMP_LATENTS_DIR (the probe's rank-0 initial/final latent dump; main has none for HV) and
checks out.pt with check_pt_hv.py. The file before this change: bench4m.as-used-before-hv-aot-2026-10-26.py.

Phase 2 (2026-10-26, later): wan2_1 added (Wan2.1 14B AoT on main, device bf16 VAE). Same q*
handling as hunyuan_video (probe dump; main has no Wan latent dump; out.pt is the VAE stage's raw
(1, C, T, H, W) [-1, 1] frames, checked with check_pt_hv.py, the same map as main's Wan
_save_video). A model may set "root_prefix" so its roots are roots/<root_prefix>_<cell> (Wan:
"wan21", the root name the stage names). The file before this change:
bench4m.as-used-hv-aot-2026-10-26.py.

Phase 2 (2026-10-26, native HV/Wan hash runs): --preread-files GLOB (repeatable, "{root}" = the
run's root) makes --preread read those files instead of the model's checkpoint list, for a run
that reads a weight store in its root (native Wan: umT5 + VAE from the checkpoint, the DiT from
the root's bf16 store); their residency right before the launch is preread_files_resident_before.
Additive; the file before this change: bench4m.as-used-wan-aot-2026-10-26.py.

Phase 2 (2026-10-26, LTX-2 native full cell): --root-prefix P overrides the model's root_prefix
for that invocation, so the root is roots/<P>_<cell> (LTX-2 native: --root-prefix ltx2 --cell
native_tp4 -> roots/ltx2_native_tp4, the root the stage names; logs stay under B/ltx_2/<cell>/).
Without it nothing changes. The file before this change: bench4m.as-used-hvwan-native-2026-10-26.py.

Derived from bench-2026-10-25/scripts/wbench.py (the Wan2.1 harness, itself from qbench.py); one
harness for Qwen-Image and LTX-2 (more models can be added to MODELS). Differences from wbench:

- AoT runs import difflet from the detached origin/main worktree /home/ubuntu/aot-main
  (PYTHONPATH=<probe>:/home/ubuntu/aot-main, cwd /home/ubuntu/aot-main); native runs from the
  hunyuan-neuron worktree. Before every run the harness runs `python -c 'import difflet;
  print(difflet.__file__)'` with the run's exact env and cwd, records it (difflet_file_precheck)
  and refuses to start an AoT run whose difflet is not under /home/ubuntu/aot-main. The probe also
  records difflet.__file__ in every process (probe/*.json "difflet_file").
- head / dirty are those of the worktree the run imports (aot-main for AoT).
- A run whose name starts with "q" is a quality run: DIFFLET_DUMP_LATENTS_DIR=<run> and
  DIFFLET_STEP_TIMES_FILE=<run>/steps.jsonl are exported (main's opt-in dumps), and the output is
  lossless: out.png for Qwen-Image (the CLI's PNG; the packed final latents are <run>/final.pt),
  out.pt for LTX-2 (the CLI saves the decoded frames tensor when the suffix is not .mp4). Timed
  runs (compile, cold, gen*) write the model's normal output (Qwen png, LTX-2 mp4) and export
  neither variable.
- After a compile, the MODULE_* directories of the root's neuron compile cache are listed
  (hlo_modules) so the DiT HLO hash can be compared with the single-core one.

  bench4m.py cell --model qwen_image|ltx_2 --cell C --venv aot|native
                  --runs compile,cold,gen0,gen1,gen2,gen3,q1 [--preread] [--env K=V ...] -- <flags>
  bench4m.py run  --model M --cell C --run R --mode compile|generate --venv aot|native
                  [--drop-cache] [--preread] [--env K=V ...] -- <flags>
  bench4m.py dry  --model M --cell C --run R --mode compile|generate --venv aot|native [--env ...] -- <flags>
      (prints the command, env and difflet.__file__ precheck; never touches the device)

<flags> are the cell's own flags (e.g. --backend trainium --tp-degree 4 --cp-degree 1); the harness
adds the model, the pinned revision, the shape, --cache-dir <root> and, for a generate, --steps 50
--guidance-scale <model's> --seed 42 --prompt ... --output <run>/out.<ext> --work-dir <run>/work
--keep-work-dir. Roots: BENCH/roots/<model>_<cell>. "compile" needs a fresh root. "cold" is a
generate after `sync; echo 3 > /proc/sys/vm/drop_caches` (passwordless sudo); meminfo (Cached) is
recorded right after the drop. --preread (compile only): drop the page cache, read the checkpoint
files the run reads twice, record mincore residency. For AoT the harness points
DIFFLET_SHARED_WEIGHTS_DIR and NEURON_COMPILE_CACHE_URL into the root (every run of the cell).
Every run waits for /dev/neuron0 to be idle before it starts and checks that it was released after.
"""
import argparse
import ctypes
import glob
import hashlib
import json
import mmap
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

# Host paths (2026-11-03, moved into the repo; nothing else changed). The data root and the two
# venvs are required settings; the worktrees default to this checkout; the HF hub cache is
# HF_HOME's (HF_HUB_CACHE is dropped from every run's env below, so it is not read here either).
# The helper scripts, the probe and nmon.json sit next to this file (they were under
# BENCH/scripts and BENCH/probe).
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def _required(name):
    value = os.environ.get(name)
    if not value:
        sys.exit(f"bench4m: set {name} (see benchmark/bench4m/README.md)")
    return value


BENCH = _required("BENCH4M_DIR")  # run dirs, summary.jsonl and roots/ live here
WTS = {"native": os.environ.get("BENCH4M_NATIVE_WT") or REPO,
       "aot": os.environ.get("BENCH4M_AOT_WT") or REPO}
VENVS = {"native": _required("BENCH4M_NATIVE_VENV"), "aot": _required("BENCH4M_AOT_VENV")}
PROMPT = "a cinematic shot of a red fox running through a snowy forest"
CHECK_MP4 = os.path.join(HERE, "check_mp4.py")
CHECK_PT = os.path.join(HERE, "check_pt_video.py")
HF = os.path.join(os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")

MODELS = {
    "qwen_image": {
        "id": "Qwen/Qwen-Image", "rev": "75e0b4be04f60ec59a75f475837eced720f823b6",
        "snap": f"{HF}/models--Qwen--Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6",
        "ckpt": ["transformer/*.safetensors", "text_encoder/*.safetensors", "vae/*.safetensors"],
        "shape": ["--height", "1024", "--width", "1024"], "guidance": "4.0",
        "ext": "png", "q_ext": "png", "cli_prefix": ("[qwen_image]", "[generate]", "[vae]", "[text]"),
    },
    "ltx_2": {
        "id": "Lightricks/LTX-2", "rev": "47da56e2ad66ce4125a9922b4a8826bf407f9d0a",
        "snap": f"{HF}/models--Lightricks--LTX-2/snapshots/47da56e2ad66ce4125a9922b4a8826bf407f9d0a",
        # what the AoT run reads: the DiT shards (compile), Gemma-3 via transformers'
        # model.safetensors.index.json (model-*-of-00011), connectors, video/audio VAE, vocoder
        "ckpt": ["transformer/*.safetensors", "text_encoder/model-*.safetensors",
                 "connectors/*.safetensors", "vae/*.safetensors", "audio_vae/*.safetensors",
                 "vocoder/*.safetensors"],
        "shape": ["--height", "480", "--width", "704", "--num-frames", "49"], "guidance": "1.0",
        "ext": "mp4", "q_ext": "pt", "cli_prefix": ("[ltx_2]", "[difflet]", "[teacache]"),
    },
    # phase 2, 2026-10-26: HunyuanVideo 1.0 AoT on main (clip, llama, generate stage processes).
    # Main's stages resolve the model with resolve_model_path(id, local_files_only=True) and no
    # revision; the HF cache holds only this snapshot and refs/main names it.
    "hunyuan_video": {
        "id": "hunyuanvideo-community/HunyuanVideo", "rev": "e8c2aaa66fe3742a32c11a6766aecbf07c56e773",
        "snap": f"{HF}/models--hunyuanvideo-community--HunyuanVideo/snapshots/"
                "e8c2aaa66fe3742a32c11a6766aecbf07c56e773",
        "ckpt": ["transformer/*.safetensors", "text_encoder/*.safetensors",
                 "text_encoder_2/*.safetensors", "vae/*.safetensors"],
        "shape": ["--height", "320", "--width", "512", "--num-frames", "61"], "guidance": "6.0",
        "ext": "mp4", "q_ext": "pt",
        "cli_prefix": ("[hunyuan_video]", "[difflet]", "[clip]", "[llama]", "[generate]"),
        # main's generate stage saves the raw VAE output (1, C, T, H, W) in [-1, 1] when the
        # --output suffix is not .mp4; HV1_DUMP_LATENTS_DIR turns on the probe's latent dump
        "check_pt": "check_pt_hv.py", "probe_dump": True,
    },
    # phase 2, 2026-10-26: Wan2.1 14B AoT on main (transformer stage: umT5 + DiT at TP4; vae stage:
    # the compiled bf16 VAE on one core). The checkpoint the runs read: DiT and umT5 (fp32) and VAE.
    "wan2_1": {
        "id": "Wan-AI/Wan2.1-T2V-14B-Diffusers", "rev": "38ec498cb3208fb688890f8cc7e94ede2cbd7f68",
        "snap": f"{HF}/models--Wan-AI--Wan2.1-T2V-14B-Diffusers/snapshots/"
                "38ec498cb3208fb688890f8cc7e94ede2cbd7f68",
        "ckpt": ["transformer/*.safetensors", "text_encoder/*.safetensors", "vae/*.safetensors"],
        "shape": ["--height", "480", "--width", "832", "--num-frames", "9"], "guidance": "5.0",
        "ext": "mp4", "q_ext": "pt", "cli_prefix": ("[wan]", "[difflet]"),
        "check_pt": "check_pt_hv.py", "probe_dump": True, "root_prefix": "wan21",
    },
}


def log(msg):
    print(f"[bench4m {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def device_users():
    r = subprocess.run(["fuser", "/dev/neuron0"], capture_output=True, text=True)
    return [p for p in r.stdout.split() if p.strip()]


def wait_idle(timeout=180):
    t = time.time()
    while True:
        users = device_users()
        if not users:
            return True
        if time.time() - t > timeout:
            return users
        time.sleep(5)


def meminfo():
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            if k in ("MemFree", "Cached", "MemAvailable", "Dirty"):
                out[k + "_GiB"] = round(int(v.split()[0]) / 2**20, 2)
    return out


def sha(path):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ckpt_files(cfg):
    fs = []
    for g in cfg["ckpt"]:
        fs += sorted(glob.glob(os.path.join(cfg["snap"], g)))
    return [os.path.realpath(f) for f in fs]


_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                       ctypes.c_long]
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]


def resident_fraction(files):
    """Bytes of `files` in the page cache (mincore), over their size."""
    page = os.sysconf("SC_PAGE_SIZE")
    tot = res = 0
    for f in files:
        size = os.path.getsize(f)
        tot += size
        if size == 0:
            continue
        with open(f, "rb") as fh:
            addr = _libc.mmap(None, ctypes.c_size_t(size), mmap.PROT_READ, mmap.MAP_SHARED,
                              fh.fileno(), ctypes.c_long(0))
            if addr in (None, ctypes.c_void_p(-1).value):
                continue
            n = (size + page - 1) // page
            vec = (ctypes.c_ubyte * n)()
            rc = _libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec)
            if rc == 0:
                res += bytes(vec).count(1) * page
            _libc.munmap(ctypes.c_void_p(addr), ctypes.c_size_t(size))
    return {"bytes_gib": round(tot / 2**30, 2), "resident_frac": round(min(res, tot) / tot, 4)
            if tot else None, "n_files": len(files)}


def drop_cache():
    subprocess.run(["sync"], check=True)
    subprocess.run(["sudo", "-n", "tee", "/proc/sys/vm/drop_caches"], input="3\n",
                   text=True, check=True, stdout=subprocess.DEVNULL)


def preread(cfg, globs=None):
    """Drop the page cache, then read the files twice. ``globs`` (absolute patterns, phase 2
    native hash runs) replaces the model's checkpoint list: what that run actually reads."""
    files = ckpt_files(cfg) if not globs else sorted(
        {os.path.realpath(f) for g in globs for f in glob.glob(g, recursive=True)})
    rec = {"dropped_at": time.time()}
    if globs:
        rec["globs"] = list(globs)
        rec["files"] = files
    drop_cache()
    rec["after_drop"] = resident_fraction(files)
    rec["meminfo_after_drop"] = meminfo()
    for i in (1, 2):
        t = time.time()
        for f in files:
            with open(f, "rb") as fh:
                while fh.read(64 << 20):
                    pass
        rec[f"read{i}_s"] = round(time.time() - t, 1)
    rec["after_reads"] = resident_fraction(files)
    rec["meminfo"] = meminfo()
    return rec


def nmon_peaks(path):
    """Per-core peak of neuron-monitor's memory_used breakdown (sum of its categories), GiB."""
    peaks = {}
    if not os.path.exists(path):
        return peaks
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            per_core = {}
            for rt in d.get("neuron_runtime_data") or []:
                mu = (rt.get("report") or {}).get("memory_used") or {}
                ncu = ((mu.get("neuron_runtime_used_bytes") or {}).get("usage_breakdown") or {}) \
                    .get("neuroncore_memory_usage") or {}
                for core, cats in ncu.items():
                    tot = sum(v for v in cats.values() if isinstance(v, (int, float)))
                    per_core[core] = per_core.get(core, 0) + tot
            for core, tot in per_core.items():
                peaks[core] = max(peaks.get(core, 0), tot)
    return {k: round(v / 2**30, 2) for k, v in sorted(peaks.items())}


def probe_summary(pdir):
    recs = []
    for p in sorted(glob.glob(os.path.join(pdir, "*.json"))):
        try:
            recs.append(json.load(open(p)))
        except ValueError:
            pass
    if not recs:
        return {}
    t_first = min(r["t0"] for r in recs)
    procs = []
    for r in sorted(recs, key=lambda r: (r["t0"], str(r.get("rank")))):
        ev = r["events"]
        pr = {"pid": r["pid"], "rank": r["rank"], "argv": " ".join(r["argv"][:2])[-90:],
              "argv_full": " ".join(r["argv"])[-300:],
              "difflet_file": r.get("difflet_file"),
              "start_s": round(r["t0"] - t_first, 2), "exit_s": round(r["t_exit"] - t_first, 2),
              "peak_hbm_gib": (round(r["peak_device_mem_gib"], 2)
                               if r.get("peak_device_mem_gib") is not None else None)}
        spans, starts = [], {}
        for e in ev:
            n = e["name"]
            if n.endswith(":start"):
                starts[n[:-6]] = e["t"]
            elif n.endswith(":end") and n[:-4] in starts:
                k = n[:-4]
                s = starts.pop(k)
                spans.append({"name": k, "at_s": round(s - t_first, 2),
                              "dur_s": round(e["t"] - s, 2)})
            elif n == "app_phase_seconds":
                pr["phase_seconds"] = e.get("phases")
            elif n == "probe_patch_error":
                pr.setdefault("probe_patch_errors", []).append(e)
        pr["spans"] = spans
        st = [e["t"] for e in ev if e["name"] == "scheduler_step"]
        if st:
            deltas = [b - a for a, b in zip(st, st[1:])]
            den = [e["t"] for e in ev if e["name"] == "denoise:start"]
            pr["steps"] = len(st)
            if den:
                pr["step0_s"] = round(st[0] - den[-1], 3)
            if deltas:
                pr["step_deltas_ms"] = [round(d * 1000, 1) for d in deltas]
                pr["step_median_ms"] = round(statistics.median(deltas) * 1000, 1)
        procs.append(pr)
    return {"t_first": t_first, "processes": procs}


def check_mp4(run_dir):
    mp4 = os.path.join(run_dir, "out.mp4")
    if not os.path.exists(mp4):
        return None
    r = subprocess.run([os.path.join(VENVS["native"], "bin", "python"), CHECK_MP4, mp4,
                        os.path.join(run_dir, "mid.png")], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("MP4 "):
            return json.loads(line[4:])
    return {"error": (r.stdout + r.stderr)[-500:]}


def check_pt(run_dir, model=None):
    pt = os.path.join(run_dir, "out.pt")
    if not os.path.exists(pt):
        return None
    chk = os.path.join(HERE, MODELS[model]["check_pt"]) if model and \
        MODELS[model].get("check_pt") else CHECK_PT
    r = subprocess.run([os.path.join(VENVS["aot"], "bin", "python"), "-I", chk, pt,
                        os.path.join(run_dir, "mid.png")], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("PT "):
            return json.loads(line[3:])
    return {"error": (r.stdout + r.stderr)[-500:]}


def check_png(path):
    if not os.path.exists(path):
        return None
    code = ("import json,sys,numpy as np;from PIL import Image;"
            "a=np.asarray(Image.open(sys.argv[1]).convert('RGB')).astype(np.float32);"
            "print('PNG '+json.dumps({'h':a.shape[0],'w':a.shape[1],'std':float(a.std()),"
            "'mean':float(a.mean()),'neighbour_absdiff':float(np.abs(np.diff(a,axis=1)).mean())}))")
    r = subprocess.run([os.path.join(VENVS["native"], "bin", "python"), "-c", code, path],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("PNG "):
            return json.loads(line[4:])
    return {"error": (r.stdout + r.stderr)[-500:]}


def build(model, cell, run, mode, venv, flags, root_cell=None, extra_env=None, root_prefix=None):
    cfg = MODELS[model]
    run_dir = os.path.join(BENCH, model, cell, run)
    prefix = root_prefix or cfg.get("root_prefix", model)
    root = os.path.join(BENCH, "roots", f"{prefix}_{root_cell or cell}")
    wt = WTS[venv]
    V = VENVS[venv]
    quality = run.startswith("q")
    env = dict(os.environ)
    for k in ("DIFFLET_NEURON_PG_TIMEOUT_S", "DIFFLET_BACKEND", "DIFFLET_SHARED_WEIGHTS_DIR",
              "NEURON_COMPILE_CACHE_URL", "DIFFLET_COMPILE_CACHE", "DIFFLET_NEURON_WEIGHT_STORE",
              "DIFFLET_NEURON_WEIGHT_STORE_ALL", "NEURON_RT_VIRTUAL_CORE_SIZE",
              "DIFFLET_DUMP_LATENTS_DIR", "DIFFLET_STEP_TIMES_FILE", "HF_HUB_CACHE",
              "PYTHONPATH", "NEURON_RT_VISIBLE_CORES", "NEURON_RT_NUM_CORES",
              "HV1_DUMP_LATENTS_DIR"):
        env.pop(k, None)
    env.update({
        "PYTHONPATH": f"{HERE}/probe:{wt}",
        "PATH": f"{V}/bin:" + env.get("PATH", "/usr/bin:/bin"),
        "HF_HUB_OFFLINE": "1",
        "HV1_PROBE_OUT": os.path.join(run_dir, "probe"),
    })
    if venv == "native":
        env["DIFFLET_COMPILE_CACHE"] = root
    else:
        env["DIFFLET_SHARED_WEIGHTS_DIR"] = os.path.join(root, "_shared_weights")
        env["NEURON_COMPILE_CACHE_URL"] = os.path.join(root, "neuron-compile-cache")
    if quality and mode == "generate":
        env["DIFFLET_DUMP_LATENTS_DIR"] = run_dir
        env["DIFFLET_STEP_TIMES_FILE"] = os.path.join(run_dir, "steps.jsonl")
        if cfg.get("probe_dump"):  # main has no dump for this model: the probe's (rank 0)
            env["HV1_DUMP_LATENTS_DIR"] = run_dir
    env.update(extra_env or {})
    cmd = [f"{V}/bin/python", "-m", "difflet.cli.main", mode, "--model-id", cfg["id"],
           "--revision", cfg["rev"], *cfg["shape"], "--cache-dir", root, *flags]
    if mode == "generate":
        ext = cfg["q_ext"] if quality else cfg["ext"]
        cmd += ["--steps", "50", "--guidance-scale", cfg["guidance"], "--seed", "42",
                "--prompt", PROMPT, "--output", os.path.join(run_dir, f"out.{ext}"),
                "--work-dir", os.path.join(run_dir, "work"), "--keep-work-dir"]
    return cfg, run_dir, root, wt, env, cmd


def difflet_precheck(venv, env, wt):
    r = subprocess.run([f"{VENVS[venv]}/bin/python", "-c",
                        "import difflet; print(difflet.__file__)"],
                       cwd=wt, env={k: v for k, v in env.items() if k != "HV1_PROBE_OUT"},
                       capture_output=True, text=True)
    return (r.stdout.strip().splitlines() or [""])[-1], r.returncode, r.stderr[-300:]


def wt_head(wt):
    head = subprocess.run(["git", "-C", wt, "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", wt, "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    return head, bool(dirty)


def run_one(model, cell, run, mode, venv, flags, drop=False, root_cell=None, do_preread=False,
            extra_env=None, dry=False, preread_globs=None, root_prefix=None):
    cfg, run_dir, root, wt, env, cmd = build(model, cell, run, mode, venv, flags, root_cell,
                                             extra_env, root_prefix)
    # "{root}" in a --preread-files pattern is the run's root (phase 2 native hash runs)
    preread_globs = [g.replace("{root}", root) for g in (preread_globs or [])] or None
    dfile, drc, derr = difflet_precheck(venv, env, wt)
    head, dirty = wt_head(wt)
    if dry:
        print(json.dumps({"run_dir": run_dir, "root": root, "cwd": wt, "head": head,
                          "dirty": dirty, "difflet_file_precheck": dfile, "precheck_rc": drc,
                          "precheck_stderr": derr if drc else "",
                          "env_set": {k: env[k] for k in sorted(env) if k.startswith(
                              ("DIFFLET", "NEURON", "HF_", "PYTHONPATH", "HV1"))},
                          "cmd": cmd, "ckpt": resident_fraction(ckpt_files(cfg)),
                          "preread_globs": preread_globs,
                          "preread_files": resident_fraction(sorted(
                              {os.path.realpath(f) for g in preread_globs
                               for f in glob.glob(g, recursive=True)})) if preread_globs else None},
                         indent=1))
        return 0 if drc == 0 else 1
    if os.path.exists(run_dir):
        raise SystemExit(f"{run_dir} exists; refusing to overwrite evidence")
    if mode == "compile" and os.path.exists(root):
        raise SystemExit(f"compile needs a fresh root; {root} exists")
    if drc != 0 or not dfile.startswith(wt + "/"):
        raise SystemExit(f"difflet precheck failed: {dfile!r} not under {wt} (rc={drc}) {derr}")
    free_gb = shutil.disk_usage("/").free / 1e9
    # 2026-10-28 (phase 7): 80 GB for a compile (it writes the weight store, Wan 38 GB); 40 GB
    # for the other runs, which write only outputs and logs (the Wan p7 cell stopped after its
    # compile at 55.8 GB free; no existing root is deleted to make room).
    min_free = 80 if mode == "compile" else 40
    if free_gb < min_free:
        raise SystemExit(f"only {free_gb:.1f} GB free on / (< {min_free}); stopping")
    # another session may run unit tests that open /dev/neuron0: wait up to 30 min for it
    first = device_users()
    if first:
        log(f"/dev/neuron0 held by {first}; waiting (up to 1800 s)")
    busy = wait_idle(timeout=1800)
    if busy is not True:
        raise SystemExit(f"/dev/neuron0 busy: pids {busy}")
    os.makedirs(run_dir)
    meta = {"model": model, "root_cell": root_cell, "cell": cell, "run": run, "mode": mode,
            "venv": venv, "drop_cache": drop, "preread": do_preread, "head": head, "dirty": dirty,
            "wt": wt, "difflet_file_precheck": dfile, "cmd": cmd, "root": root,
            "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "env_extra": {k: env[k] for k in ("DIFFLET_COMPILE_CACHE", "DIFFLET_SHARED_WEIGHTS_DIR",
                                              "NEURON_COMPILE_CACHE_URL", "DIFFLET_DUMP_LATENTS_DIR",
                                              "HV1_DUMP_LATENTS_DIR", "DIFFLET_STEP_TIMES_FILE", "PYTHONPATH",
                                              "HF_HUB_OFFLINE") if k in env}}
    meta["env_extra"].update(extra_env or {})
    if do_preread:
        log(f"{model}/{cell}/{run}: dropping the page cache and reading the checkpoint twice")
        meta["preread_rec"] = preread(cfg, preread_globs)
        log(f"preread: {meta['preread_rec']}")
    late = device_users()  # re-check after the (minutes-long) preread, before the drop
    if late:
        meta["device_busy_before_launch"] = late
        log(f"/dev/neuron0 held by {late} after preread; waiting (up to 1800 s)")
        if wait_idle(timeout=1800) is not True:
            raise SystemExit(f"/dev/neuron0 busy before launch: pids {device_users()}")
    if drop:
        drop_cache()
        meta["dropped_at"] = time.time()
    files = ckpt_files(cfg)
    meta["ckpt_resident_before"] = resident_fraction(files)
    if preread_globs:  # what the run actually reads, right before the launch
        meta["preread_files_resident_before"] = resident_fraction(sorted(
            {os.path.realpath(f) for g in preread_globs for f in glob.glob(g, recursive=True)}))
    meta["meminfo_before"] = meminfo()
    meta["df_free_gb_before"] = round(shutil.disk_usage("/").free / 1e9, 1)
    log(f"{model}/{cell}/{run} {mode} start: {' '.join(cmd)}")
    nmon_f = open(os.path.join(run_dir, "nmon.jsonl"), "w")
    nmon = subprocess.Popen(["/opt/aws/neuron/bin/neuron-monitor", "-c",
                             os.path.join(HERE, "nmon.json")], stdout=nmon_f,
                            stderr=subprocess.DEVNULL)
    t = time.time()
    with open(os.path.join(run_dir, "run.log"), "w") as lf:
        lf.write(f"date: {meta['date']}\nwt: {wt}\nhead: {head} dirty={int(dirty)}\n"
                 f"difflet_file_precheck: {dfile}\n"
                 f"env_extra: {json.dumps(meta['env_extra'])}\ncmd: {' '.join(cmd)}\n")
        lf.flush()
        rc = subprocess.run(["/usr/bin/time", "-v", *cmd], cwd=wt, env=env, stdout=lf,
                            stderr=subprocess.STDOUT).returncode
    wall = time.time() - t
    nmon.terminate()
    nmon.wait()
    nmon_f.close()
    meta.update({"rc": rc, "wall_s": round(wall, 2), "t_start": t, "meminfo_after": meminfo(),
                 "df_free_gb_after": round(shutil.disk_usage("/").free / 1e9, 1)})
    after = wait_idle(timeout=120)
    meta["device_released"] = after is True
    if after is not True:
        meta["stragglers"] = after
    meta["nmon_peak_gib"] = nmon_peaks(os.path.join(run_dir, "nmon.jsonl"))
    meta["probe"] = probe_summary(os.path.join(run_dir, "probe"))
    if meta["probe"]:
        meta["probe_t_first_minus_start_s"] = round(meta["probe"]["t_first"] - t, 2)
        meta["difflet_files_seen"] = sorted({p["difflet_file"] for p in meta["probe"]["processes"]
                                             if p.get("difflet_file")})
    if mode == "compile":
        meta["hlo_modules"] = sorted(
            os.path.basename(p) for p in glob.glob(os.path.join(root, "neuron-compile-cache", "*",
                                                                "MODULE_*")))
        meta["root_du_gb"] = round(sum(os.path.getsize(os.path.join(dp, f))
                                       for dp, _, fs in os.walk(root) for f in fs) / 1e9, 1)
    if mode == "generate":
        for ext in ("png", "mp4", "pt"):
            p = os.path.join(run_dir, f"out.{ext}")
            if os.path.exists(p):
                meta[f"sha256_out_{ext}"] = sha(p)
        if os.path.exists(os.path.join(run_dir, "out.png")):
            meta["png"] = check_png(os.path.join(run_dir, "out.png"))
        if os.path.exists(os.path.join(run_dir, "out.mp4")):
            meta["mp4"] = check_mp4(run_dir)
        if os.path.exists(os.path.join(run_dir, "out.pt")):
            meta["pt"] = check_pt(run_dir, model)
        meta["sha256_mid_png"] = sha(os.path.join(run_dir, "mid.png"))
        lat = glob.glob(os.path.join(run_dir, "work", "**", "latents.pt"), recursive=True)
        meta["sha256_latents_pt"] = sha(lat[0]) if lat else None
        meta["dumps"] = {os.path.basename(p): sha(p) for p in sorted(
            glob.glob(os.path.join(run_dir, "initial*.pt")) + glob.glob(
                os.path.join(run_dir, "final*.pt")))}
        sj = os.path.join(run_dir, "steps.jsonl")
        if os.path.exists(sj):
            rows = [json.loads(x) for x in open(sj) if x.strip()]
            d = [r["step_s"] for r in rows if r["step"] > 0]
            meta["steps_jsonl"] = {"n": len(rows), "median_ms_excl0": round(
                statistics.median(d) * 1000, 1) if d else None}
    with open(os.path.join(run_dir, "run.log")) as lf:
        txt = lf.read()
    m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", txt)
    meta["time_v_elapsed"] = m.group(1) if m else None
    meta["cli_lines"] = [ln for ln in txt.splitlines()
                         if ln.startswith(cfg["cli_prefix"]) or "load+warmup" in ln
                         or "Reusing pre-sharded" in ln or "Traceback" in ln
                         or "Error" in ln][:80]
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(meta, f, indent=1)
    slim = {k: v for k, v in meta.items() if k not in ("probe", "cli_lines")}
    r0 = [p for p in (meta["probe"] or {}).get("processes", []) if p.get("steps")]
    r0 = [p for p in r0 if p.get("rank") in ("0", None)]
    if r0:
        slim["step_median_ms"] = r0[0].get("step_median_ms")
        slim["steps"] = r0[0].get("steps")
    slim["peak_hbm_gib"] = [p["peak_hbm_gib"] for p in (meta["probe"] or {}).get("processes", [])
                            if p.get("peak_hbm_gib") is not None]
    os.makedirs(os.path.join(BENCH, model), exist_ok=True)
    with open(os.path.join(BENCH, model, "summary.jsonl"), "a") as f:
        f.write(json.dumps(slim) + "\n")
    outsha = meta.get("sha256_out_png") or meta.get("sha256_out_mp4") or meta.get("sha256_out_pt")
    log(f"{model}/{cell}/{run} {mode} rc={rc} wall={wall:.1f}s step_median={slim.get('step_median_ms')}"
        f" steps={slim.get('steps')} nmon={meta['nmon_peak_gib']} released={meta['device_released']}"
        f" out={str(outsha)[:8]} difflet={meta.get('difflet_files_seen')}")
    return rc


def main():
    argv = sys.argv[1:]
    flags = []
    if "--" in argv:
        i = argv.index("--")
        argv, flags = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("run", "cell", "dry"))
    p.add_argument("--model", choices=tuple(MODELS), required=True)
    p.add_argument("--cell", required=True)
    p.add_argument("--venv", choices=tuple(VENVS), default="aot")
    p.add_argument("--run")
    p.add_argument("--mode", choices=("compile", "generate"))
    p.add_argument("--drop-cache", action="store_true")
    p.add_argument("--preread", action="store_true")
    p.add_argument("--runs", default="compile,cold,gen0,gen1,gen2,gen3,q1")
    p.add_argument("--root-cell", default=None, help="reuse roots/<model>_<this> (no compile run)")
    p.add_argument("--env", action="append", default=[], help="K=V for every run of the cell")
    p.add_argument("--preread-files", action="append", default=[],
                   help="absolute glob ({root} = the run's root) read by --preread instead of the "
                        "model's checkpoint list (run/dry only; phase 2 native hash runs)")
    p.add_argument("--root-prefix", default=None,
                   help="override the model's root_prefix: roots/<this>_<cell> (LTX-2 native)")
    a = p.parse_args(argv)
    extra = dict(kv.split("=", 1) for kv in a.env)
    if a.action in ("run", "dry"):
        sys.exit(run_one(a.model, a.cell, a.run, a.mode, a.venv, flags, a.drop_cache, a.root_cell,
                         a.preread, extra, dry=(a.action == "dry"),
                         preread_globs=a.preread_files or None, root_prefix=a.root_prefix))
    for run in a.runs.split(","):
        mode = "compile" if run.startswith("compile") else "generate"
        rc = run_one(a.model, a.cell, run, mode, a.venv, flags, drop=(run == "cold"),
                     root_cell=a.root_cell, do_preread=(a.preread and mode == "compile"),
                     extra_env=extra, root_prefix=a.root_prefix)
        if rc != 0:
            log(f"{a.model}/{a.cell}/{run} failed rc={rc}; stopping the cell")
            print("CELL_DONE FAIL", flush=True)
            sys.exit(rc)
    print("CELL_DONE OK", flush=True)


if __name__ == "__main__":
    main()
