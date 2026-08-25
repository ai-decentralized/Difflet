#!/usr/bin/env python3
"""Phase-broken stable timing for every feasible 4-core parallel config.

Extends ``scripts/flux_parallel_sweep.py`` (which measured per-step only) with
the full phase breakdown the planner's cost model needs, for FLUX.1-dev and
Wan 2.2 T2V A14B on one trn2.3xlarge:

  compile   — cold ``difflet compile`` wall + per-component sub-phases
              (``benchmark.parse_compile``: module load / HLO gen / HLO compile)
  e2e cold  — one ``difflet generate`` immediately after
              ``sync; echo 3 > /proc/sys/vm/drop_caches`` (true cold weight
              load), broken into per-stage load_s / shard_s
              (``benchmark.parse_generate``)
  e2e warm  — N (=3) further generates; median +/- spread of the wall and of
              every stage's load (the *stable* numbers)
  per-step  — in-process real-generate inter-step deltas, step 0 excluded
              (``benchmark/step_realloop.py``'s method): FLUX wraps the
              backbone app exactly like the flux sweep; Wan drives the real
              ``transformer`` stage via ``WanOrchestrator._run_stage_internal``
              with ``NeuronWanBackboneApplication.__call__`` wrapped, so the
              loop, inputs, and artifacts are the CLI's own.

DP rows (dp2tp2, dp2tp2sp) are generate-only: compile resolves to the dp=1
tp2/tp2sp artifact (compile-once-load-k) and the per-step graph is that base
config's, so only their e2e (through the real ``--dp 2`` router, two requests)
is measured; the JSON records the derivation.

Results land in ``artifacts/parallel_phase_sweep/<model>/<label>.json`` (one
file per config, written incrementally after every phase so a crash loses
nothing) plus ``summary.{json,md}``. Resume: rerun with the same --model; a
phase already present in the JSON is skipped unless --phase is given.

    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python scripts/parallel_phase_sweep.py --model flux
    python scripts/parallel_phase_sweep.py --model wan
    python scripts/parallel_phase_sweep.py --model flux --only tp4 --phase step
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ART_ROOT = REPO / "artifacts" / "parallel_phase_sweep"

WARM_ITERS = 3
COMPILE_TIMEOUT = 6 * 3600
GENERATE_TIMEOUT = 2 * 3600
WORKER_TIMEOUT = 2 * 3600
# A cache-hit compile returns in well under 5 minutes; a cold one never does.
COMPILE_HIT_WALL_S = 300


class Spec(dict):
    """Benchmark spec for one model (see SPECS below for the fields)."""


SPECS: dict[str, Spec] = {
    "flux": Spec(
        model_id="black-forest-labs/FLUX.1-dev",
        revision="3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
        prompt="a cat sitting on a bench",
        steps=28,
        guidance=3.5,
        seed=42,
        height=1024,
        width=1024,
        num_frames=None,
        output_ext="png",
    ),
    "wan": Spec(
        model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        revision="5be7df9619b54f4e2667b2755bc6a756675b5cd7",
        prompt="a cat walking through a garden",
        steps=20,
        guidance=4.0,  # true-CFG path, same as verify_cli's wan cell (g=1.0
        # would leave the cfg-parallel branch degenerate)
        seed=42,
        height=480,
        width=832,
        num_frames=9,
        output_ext="mp4",
    ),
}


def _cfg(tp=1, cp=1, cp_mode="gather_kv", cfg=False, sp=False, dp=1) -> dict:
    return {
        "tp_degree": tp, "cp_degree": cp, "cp_mode": cp_mode,
        "cfg_parallel_enabled": cfg, "sp_enabled": sp, "dp_degree": dp,
    }


# Exactly the host-filling configs the planner enumerates on 4 cores, plus the
# under-filling tp2/tp2sp bases their DP rows load (D35's "traversal set").
FLUX_CONFIGS: dict[str, dict] = {
    "tp4": _cfg(tp=4),
    "tp4sp": _cfg(tp=4, sp=True),
    "tp2cp2": _cfg(tp=2, cp=2),
    "tp2cp2ring": _cfg(tp=2, cp=2, cp_mode="ring"),
    "tp2cp2ulysses": _cfg(tp=2, cp=2, cp_mode="ulysses"),
    "tp2": _cfg(tp=2),
    "tp2sp": _cfg(tp=2, sp=True),
}
WAN_CONFIGS: dict[str, dict] = dict(FLUX_CONFIGS, tp2cfg=_cfg(tp=2, cfg=True))

# label -> (artifact base, dp degree). Compile/per-step are the base's.
DP_ROWS: dict[str, tuple[str, int]] = {
    "dp2tp2": ("tp2", 2),
    "dp2tp2sp": ("tp2sp", 2),
}


def world_size(cfg: dict) -> int:
    return (
        cfg["tp_degree"] * cfg["cp_degree"]
        * (2 if cfg["cfg_parallel_enabled"] else 1)
    )


def config_flags(cfg: dict) -> list[str]:
    f = ["--tp-degree", str(cfg["tp_degree"])]
    if cfg["cp_degree"] > 1:
        f += ["--cp-degree", str(cfg["cp_degree"])]
        if cfg["cp_mode"] != "gather_kv":
            f += ["--cp-mode", cfg["cp_mode"]]
    if cfg["cfg_parallel_enabled"]:
        f.append("--cfg-parallel")
    if cfg["sp_enabled"]:
        f.append("--sp")
    if cfg["dp_degree"] > 1:
        f += ["--dp", str(cfg["dp_degree"])]
    return f


# ------------------------------------------------------------------ helpers


def _stats(samples: list[float]) -> dict:
    if not samples:
        return {"n": 0}
    s = sorted(samples)
    return {
        "n": len(s),
        "mean": statistics.fmean(s),
        "median": statistics.median(s),
        "min": s[0],
        "max": s[-1],
        "p90": s[min(len(s) - 1, max(0, int(round(0.9 * (len(s) - 1)))))],
        **({"std": statistics.pstdev(s)} if len(s) > 1 else {}),
    }


def _run(cmd: list[str], log: Path, timeout: float, env: dict | None = None) -> tuple[float, str]:
    t0 = time.perf_counter()
    with log.open("w") as fh:
        proc = subprocess.run(
            cmd, stdout=fh, stderr=subprocess.STDOUT, text=True,
            timeout=timeout, env=env)
    text = log.read_text(encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = "\n".join(text.splitlines()[-25:])
        raise RuntimeError(
            f"failed ({proc.returncode}): {' '.join(cmd)}\n{tail}")
    return time.perf_counter() - t0, text


def _venv_difflet() -> str:
    which = shutil.which("difflet") or ""
    if "/aws_neuronx_venv" not in which:
        raise SystemExit(
            "difflet not on PATH from the Neuron venv — run inside "
            "`source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate`")
    return which


def _median_stage_loads(breakdowns: list[dict]) -> dict:
    """Median per-stage load/shard across warm runs (stage i matched by index)."""
    n_stages = min(len(b.get("stages", [])) for b in breakdowns)
    out: dict = {"stages": []}
    for i in range(n_stages):
        names = {b["stages"][i].get("stage") for b in breakdowns}
        loads = [b["stages"][i]["load_s"] for b in breakdowns]
        shards = [b["stages"][i].get("shard_s", 0.0) for b in breakdowns]
        stage = {
            "stage": sorted(names)[0] if len(names) == 1 else f"stage_{i}",
            "load_s_median": statistics.median(loads),
            "shard_s_median": statistics.median(shards),
        }
        out["stages"].append(stage)
    out["weights_load_total_s_median"] = round(
        statistics.median([b["weights_load_total_s"] for b in breakdowns]), 3)
    return out


# ------------------------------------------------------------------ phases


def phase_compile(model: str, label: str, cfg: dict, spec: Spec, out: dict,
                  log_dir: Path) -> None:
    difflet = _venv_difflet()
    cmd = [difflet, "compile", "--model-id", spec["model_id"]]
    if spec["revision"]:
        cmd += ["--revision", spec["revision"]]
    cmd += config_flags(cfg)
    cmd += ["--height", str(spec["height"]), "--width", str(spec["width"])]
    if spec["num_frames"]:
        cmd += ["--num-frames", str(spec["num_frames"])]
    wall, text = _run(cmd, log_dir / f"{label}_compile.log", COMPILE_TIMEOUT)
    from benchmark.parse_compile import parse
    breakdown = parse(text)
    cold = bool(breakdown.get("wall_total_s")) or wall >= COMPILE_HIT_WALL_S
    out["compile"] = {
        "wall_s": round(wall, 1),
        "cache_hit": not cold,
        "breakdown_s": breakdown,
    }


def _generate_cmd(difflet: str, label_cfg: dict, spec: Spec, output: str,
                  *, requests: Path | None = None) -> list[str]:
    cmd = [difflet, "generate", "--model-id", spec["model_id"]]
    if spec["revision"]:
        cmd += ["--revision", spec["revision"]]
    cmd += config_flags(label_cfg)
    cmd += ["--height", str(spec["height"]), "--width", str(spec["width"])]
    if spec["num_frames"]:
        cmd += ["--num-frames", str(spec["num_frames"])]
    cmd += ["--steps", str(spec["steps"]),
            "--guidance-scale", str(spec["guidance"]),
            "--seed", str(spec["seed"])]
    if requests is not None:
        cmd += ["--requests", str(requests)]
    else:
        cmd += ["--prompt", spec["prompt"], "--output", output]
    return cmd


def _parse_generate(text: str, wall: float) -> dict:
    from benchmark.parse_generate import parse
    return parse(text, wall)


def _drop_page_cache() -> bool:
    try:
        proc = subprocess.run(
            ["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
            capture_output=True, text=True, timeout=120)
        return proc.returncode == 0
    except Exception:
        return False


def phase_generate(model: str, label: str, cfg: dict, spec: Spec, out: dict,
                   log_dir: Path, out_dir: Path) -> None:
    """One cold + WARM_ITERS warm generates, with per-stage load breakdowns."""
    difflet = _venv_difflet()

    dropped = _drop_page_cache()
    if not dropped:
        out.setdefault("notes", []).append(
            "cold run skipped: sudo drop_caches unavailable — cold numbers "
            "come from the first post-compile generate instead")
        cold_note = "page cache NOT dropped (sudo unavailable)"
    else:
        cold_note = "page cache dropped (sync; echo 3 > drop_caches)"
    ext = spec["output_ext"]
    cmd = _generate_cmd(difflet, cfg, spec, str(out_dir / f"{label}_cold.{ext}"))
    wall, text = _run(cmd, log_dir / f"{label}_gen_cold.log", GENERATE_TIMEOUT)
    out["e2e_cold"] = _parse_generate(text, wall)
    out["e2e_cold"]["note"] = cold_note

    walls: list[float] = []
    breakdowns: list[dict] = []
    for i in range(WARM_ITERS):
        cmd = _generate_cmd(
            difflet, cfg, spec, str(out_dir / f"{label}_warm{i}.{ext}"))
        wall, text = _run(
            cmd, log_dir / f"{label}_gen_warm{i}.log", GENERATE_TIMEOUT)
        walls.append(wall)
        breakdowns.append(_parse_generate(text, wall))
    out["e2e_warm"] = {
        "wall_s": _stats(walls),
        **_median_stage_loads(breakdowns),
        "runs": [round(w, 3) for w in walls],
    }


def phase_generate_dp(model: str, label: str, base: str, dp: int, spec: Spec,
                      out: dict, log_dir: Path, out_dir: Path) -> None:
    """DP rows: e2e through the real router with one request per replica."""
    difflet = _venv_difflet()
    cfg = dict(_CFG_REGISTRY[model][base])
    assert cfg["dp_degree"] == 1
    ext = spec["output_ext"]
    stem = out_dir / label

    def requests_file(runs: int) -> Path:
        path = out_dir / f"{label}_requests.jsonl"
        path.write_text("\n".join(
            json.dumps({
                "prompt": f"{spec['prompt']} (variant {r})",
                "output": str(out_dir / f"{label}_w{runs}_dp{r}.{ext}"),
                "seed": spec["seed"] + r,
            }) for r in range(dp)) + "\n", encoding="utf-8")
        return path

    gen_cfg = dict(cfg, dp_degree=dp)  # the router needs --dp N; compile did not
    dropped = _drop_page_cache()
    cmd = _generate_cmd(
        difflet, gen_cfg, spec, "", requests=requests_file(0))
    wall, text = _run(cmd, log_dir / f"{label}_gen_cold.log", GENERATE_TIMEOUT)
    out["e2e_cold"] = _parse_generate(text, wall)
    out["e2e_cold"]["note"] = (
        f"{dp} requests through the --dp {dp} router, one per replica"
        + ("; page cache dropped" if dropped else "; page cache NOT dropped"))

    walls = []
    for i in range(WARM_ITERS):
        cmd = _generate_cmd(
            difflet, gen_cfg, spec, "", requests=requests_file(i + 1))
        wall, _ = _run(cmd, log_dir / f"{label}_gen_warm{i}.log",
                       GENERATE_TIMEOUT)
        walls.append(wall)
    out["e2e_warm"] = {"wall_s": _stats(walls),
                       "runs": [round(w, 3) for w in walls]}
    out.setdefault("notes", []).append(
        f"DERIVED per-step: dp{dp} replicas load and execute the same {base} "
        f"artifact, so the DiT-step latency is the {base} row's; dp adds only "
        f"router/process overhead, which is e2e, not step, time")


# ------------------------------------------------------------------ workers


def _worker_env(cfg: dict) -> dict:
    import os
    env = dict(os.environ)
    world = world_size(cfg)
    env["NEURON_RT_NUM_CORES"] = str(world)
    if world < 4:
        # In-process loads otherwise boot a 4-rank group and the ranks beyond
        # world_size never find a root (same pinning the DP router uses).
        env["NEURON_RT_VISIBLE_CORES"] = ",".join(str(i) for i in range(world))
    return env


def worker_flux(label: str) -> int:
    """In-process realloop per-step for one flux config (flux sweep's method)."""
    import torch
    from difflet.pipeline.difflet_pipeline import DiffletPipeline
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    spec = SPECS["flux"]
    cfg = FLUX_CONFIGS[label]
    parallel = DiffletParallelConfig(**{k: v for k, v in cfg.items()})
    cache = Path("~/.cache/difflet").expanduser()
    print(f"[step:{label}] building pipeline {parallel}...", flush=True)
    t0 = time.perf_counter()
    pipe = DiffletPipeline.from_pretrained(
        spec["model_id"], model_type="flux", parallel=parallel,
        dtype=torch.bfloat16, height=spec["height"], width=spec["width"],
        compile_cache_dir=str(cache), revision=spec["revision"],
        skip_compile=True)
    build_s = time.perf_counter() - t0
    dit_cls = type(pipe.app.pipe.transformer)
    print(f"[step:{label}] pipeline ready in {build_s:.1f}s", flush=True)

    stamps: list[float] = []
    orig = dit_cls.__call__

    def timed(self, *a, **k):
        r = orig(self, *a, **k)
        stamps.append(time.perf_counter())  # host tensors returned -> synced
        return r

    gen_kwargs = dict(
        prompt=spec["prompt"], num_inference_steps=spec["steps"],
        height=spec["height"], width=spec["width"],
        guidance_scale=spec["guidance"],
        generator=torch.Generator().manual_seed(spec["seed"]),
        output_type="pt")
    dit_cls.__call__ = timed
    try:
        pipe(**gen_kwargs)            # warm-up: page cache + NEFF dispatch
        stamps.clear()
        t1 = time.perf_counter()
        result = pipe(**gen_kwargs)   # timed
        gen_s = time.perf_counter() - t1
    finally:
        dit_cls.__call__ = orig

    deltas = [stamps[i] - stamps[i - 1] for i in range(1, len(stamps))]
    finite = None
    try:
        img = getattr(result, "images", None) or result[0]
        if isinstance(img, torch.Tensor):
            finite = bool(torch.isfinite(img).all())
    except Exception as exc:
        print(f"[step:{label}] finite-check skipped: {exc}", flush=True)
    print("STEP_RESULT " + json.dumps({
        "label": label, "build_s": build_s, "generate_wall_s": gen_s,
        "finite": finite, "step_s": deltas}))
    return 0


def worker_wan(label: str) -> int:
    """Real transformer stage of the Wan CLI, backbone __call__ wrapped."""
    import argparse as _ap
    import torch

    from difflet.backends.trainium.wan.backbone import (
        NeuronWanBackboneApplication,
    )
    from difflet.cli.orchestrators.wan import WanOrchestrator

    spec = SPECS["wan"]
    cfg = WAN_CONFIGS[label]
    work_dir = ART_ROOT / "wan" / "work" / label
    work_dir.mkdir(parents=True, exist_ok=True)

    parent = _ap.Namespace(
        model_id=spec["model_id"], revision=spec["revision"],
        tp_degree=cfg["tp_degree"], cp_degree=cfg["cp_degree"],
        cp_mode=cfg["cp_mode"], height=spec["height"], width=spec["width"],
        num_frames=spec["num_frames"], steps=spec["steps"],
        guidance_scale=spec["guidance"], seed=spec["seed"],
        cfg_parallel=cfg["cfg_parallel_enabled"], sp_enabled=cfg["sp_enabled"],
        prompt=spec["prompt"],
        output=str(work_dir / "latents_out.mp4"),
        cache_dir=None, work_dir=str(work_dir), requests_dir=None,
        worker_index=0, dp_schedule="round_robin", host_vae=False,
        dp_degree=cfg["dp_degree"])
    orch = WanOrchestrator(parent)
    argv = orch._shared_cli_args(stage_mode="generate", work_dir=str(work_dir))

    from difflet.cli.stage import _build_stage_parser
    ns = _build_stage_parser().parse_args(
        ["--orchestrator", spec["model_id"], "--stage", "transformer"] + argv)

    stamps: list[float] = []
    orig = NeuronWanBackboneApplication.__call__

    def timed(self, *a, **k):
        r = orig(self, *a, **k)
        stamps.append(time.perf_counter())  # host tensors returned -> synced
        return r

    build_s: list[float] = []
    NeuronWanBackboneApplication.__call__ = timed
    try:
        t0 = time.perf_counter()
        orch._run_stage_internal("transformer", ns)   # warm-up (loads too)
        build_s.append(time.perf_counter() - t0)
        stamps.clear()
        t1 = time.perf_counter()
        orch._run_stage_internal("transformer", ns)   # timed
        gen_s = time.perf_counter() - t1
    finally:
        NeuronWanBackboneApplication.__call__ = orig

    deltas = [stamps[i] - stamps[i - 1] for i in range(1, len(stamps))]
    finite = None
    try:
        latents = torch.load(work_dir / "latents.pt", map_location="cpu")
        finite = bool(torch.isfinite(latents).all())
    except Exception as exc:
        print(f"[step:{label}] finite-check skipped: {exc}", flush=True)
    print("STEP_RESULT " + json.dumps({
        "label": label, "build_s": build_s[0], "generate_wall_s": gen_s,
        "finite": finite, "step_s": deltas}))
    return 0


_CFG_REGISTRY = {"flux": FLUX_CONFIGS, "wan": WAN_CONFIGS}
_WORKERS = {"flux": worker_flux, "wan": worker_wan}


def phase_step(model: str, label: str, cfg: dict, spec: Spec, out: dict,
               log_dir: Path) -> None:
    env = _worker_env(cfg)
    log = log_dir / f"{label}_step.log"
    with log.open("w") as fh:
        proc = subprocess.run(
            [sys.executable, __file__, "--step-worker", model, label],
            stdout=fh, stderr=subprocess.STDOUT, text=True,
            timeout=WORKER_TIMEOUT, env=env)
    text = log.read_text(encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = "\n".join(text.splitlines()[-25:])
        raise RuntimeError(f"[{label}] step worker failed\n{tail}")
    line = next(l for l in text.splitlines() if l.startswith("STEP_RESULT "))
    payload = json.loads(line[len("STEP_RESULT "):])
    payload.pop("label")
    payload["step_s"] = _stats(payload["step_s"])
    payload["step_s"] = {k: (round(v, 6) if isinstance(v, float) else v)
                         for k, v in payload["step_s"].items()}
    payload["build_s"] = round(payload["build_s"], 1)
    payload["generate_wall_s"] = round(payload["generate_wall_s"], 1)
    out["step"] = payload


# ------------------------------------------------------------------ driver


def _load_result(model: str, label: str) -> dict:
    path = ART_ROOT / model / f"{label}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"model": model, "label": label}


def _save_result(model: str, label: str, result: dict) -> None:
    path = ART_ROOT / model / f"{label}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


def run_model(model: str, only: list[str] | None, phases: list[str] | None) -> int:
    spec = SPECS[model]
    configs = _CFG_REGISTRY[model]
    labels = [l for l in configs if not only or l in only]
    labels += [l for l in DP_ROWS if (not only or l in only)]
    log_dir = ART_ROOT / model / "logs"
    out_dir = ART_ROOT / model / "out"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    for label in labels:
        result = _load_result(model, label)
        result.setdefault("parallel", configs.get(label) or dict(
            _CFG_REGISTRY[model][DP_ROWS[label][0]], dp_degree=DP_ROWS[label][1]))
        result["spec"] = {k: spec[k] for k in
                          ("model_id", "revision", "steps", "guidance", "seed",
                           "height", "width", "num_frames", "prompt")}
        is_dp = label in DP_ROWS

        def want(phase: str) -> bool:
            return (phases is None or phase in phases) and phase not in result

        try:
            if want("compile"):
                cfg = configs[DP_ROWS[label][0]] if is_dp else configs[label]
                print(f">>> [{model}:{label}] compile ...", flush=True)
                phase_compile(model, label, cfg, spec, result, log_dir)
                _save_result(model, label, result)
                c = result["compile"]
                print(f"    wall {c['wall_s']:.0f}s "
                      f"({'cache hit' if c['cache_hit'] else 'cold'})", flush=True)
            if want("generate"):
                print(f">>> [{model}:{label}] generate (1 cold + "
                      f"{WARM_ITERS} warm) ...", flush=True)
                if is_dp:
                    base, dp = DP_ROWS[label]
                    phase_generate_dp(model, label, base, dp, spec, result,
                                      log_dir, out_dir)
                else:
                    phase_generate(model, label, configs[label], spec, result,
                                   log_dir, out_dir)
                _save_result(model, label, result)
                w = result["e2e_warm"]["wall_s"]
                print(f"    warm e2e median {w['median']:.1f}s "
                      f"(n={w['n']})", flush=True)
            if want("step") and not is_dp:
                print(f">>> [{model}:{label}] per-step realloop ...", flush=True)
                phase_step(model, label, configs[label], spec, result, log_dir)
                _save_result(model, label, result)
                s = result["step"]["step_s"]
                print(f"    step median {s['median'] * 1000:.1f} ms "
                      f"(n={s['n']})", flush=True)
        except Exception as exc:
            result.setdefault("errors", {})[
                _phase_name(want, is_dp)] = str(exc)[:2000]
            _save_result(model, label, result)
            print(f"    [{model}:{label}] ERROR: {exc}", flush=True)
            continue

    write_summary(model)
    return 0


def _phase_name(want, is_dp: bool) -> str:
    # best-effort label for the error record (the failing phase)
    for phase in ("compile", "generate", "step"):
        if want(phase):
            return phase
    return "unknown"


def write_summary(model: str) -> None:
    root = ART_ROOT / model
    rows = []
    for path in sorted(root.glob("*.json")):
        if path.name == "summary.json":
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        c = r.get("compile", {})
        cold = r.get("e2e_cold", {})
        warm = r.get("e2e_warm", {})
        step = r.get("step", {})
        sw = warm.get("wall_s", {})
        ss = step.get("step_s", {})
        rows.append({
            "label": r.get("label", path.stem),
            "compile_wall_s": c.get("wall_s"),
            "compile_cache_hit": c.get("cache_hit"),
            "e2e_cold_s": cold.get("wall_total_s"),
            "load_cold_s": cold.get("weights_load_total_s"),
            "e2e_warm_median_s": sw.get("median"),
            "e2e_warm_std_s": sw.get("std"),
            "load_warm_median_s": warm.get("weights_load_total_s_median"),
            "step_median_ms": round(ss["median"] * 1000, 1) if ss.get("median") else None,
            "step_n": ss.get("n"),
            "finite": step.get("finite"),
            "errors": list(r.get("errors", {}).keys()) or None,
        })
    (root / "summary.json").write_text(
        json.dumps(rows, indent=1), encoding="utf-8")

    lines = [
        f"# Parallel phase sweep — {model} "
        f"({SPECS[model]['model_id']}, {SPECS[model]['steps']} steps, "
        f"guidance {SPECS[model]['guidance']})",
        "",
        "| config | compile s (cold) | load cold s | load warm s | e2e cold s | "
        "e2e warm s (median) | step ms (median, n) | finite | err |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in sorted(rows, key=lambda x: (x["e2e_warm_median_s"] is None,
                                         x["e2e_warm_median_s"] or 0)):
        lines.append(
            "| {label} | {c} | {lc} | {lw} | {ec} | {ew} | {st} | {fin} | {er} |".format(
                label=r["label"],
                c=r["compile_wall_s"] if not r["compile_cache_hit"] else "cache",
                lc=r["load_cold_s"], lw=r["load_warm_median_s"],
                ec=r["e2e_cold_s"],
                ew=(f"{r['e2e_warm_median_s']:.1f}"
                    f" ±{r['e2e_warm_std_s']:.1f}"
                    if r["e2e_warm_median_s"] is not None else "—"),
                st=(f"{r['step_median_ms']} (n={r['step_n']})"
                    if r["step_median_ms"] else "derived"),
                fin=r["finite"], er=",".join(r["errors"] or [])))
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"\nsummary -> {root / 'summary.md'}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=sorted(SPECS))
    p.add_argument("--only", nargs="*", help="restrict to these config labels")
    p.add_argument("--phase", nargs="*",
                   help="restrict to these phases (compile/generate/step); "
                        "default: every phase missing from the saved JSON")
    p.add_argument("--step-worker", nargs=2, metavar=("MODEL", "LABEL"),
                   help=argparse.SUPPRESS)  # internal: in-process measurement
    args = p.parse_args()
    if args.step_worker:
        return _WORKERS[args.step_worker[0]](args.step_worker[1])
    if not args.model:
        p.error("--model is required")
    return run_model(args.model, args.only, args.phase)


if __name__ == "__main__":
    raise SystemExit(main())
