"""Neuron (TorchNeuron) backend adapter for the benchmark harness, plus the launch helpers and
the pure log/JSON summaries ``benchmark/neuron_cold_warm.py`` builds on.

The neuron backend compiles nothing ahead of time: each process compiles (or loads from the
persistent NEFF cache) its per-block graph inside ``load()``'s warm-up forward. So, like the
TPU adapter, ``compile()`` returns 0 and the compile cost lands in the first launch on an
empty cache (``neuron_cold_warm.py`` measures it as cold minus warm warm-up).

``run_generate()`` launches ``benchmark/neuron_wan_bench.py`` under torchrun (one rank per
NeuronCore) and is driven by the ranks' ``result-rank<r>.json``, never by log parsing: the
per-step series exists only there. ``run_cli_generate()`` times the user-facing
``difflet generate --backend neuron`` (``<python> -m difflet.cli.main``, the venv this module
runs in; never the Trainium adapter's NXD venv or its PATH edit) and parses the stage's
``[wan]`` lines.

Imports no torch: the driver process never touches the device.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark.harness import BackendAdapter, OutputInfo, Stats
from difflet.backends.neuron.compile import CACHE_ENV_DIRS, neuron_cache_env

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_SCRIPT = REPO_ROOT / "benchmark" / "neuron_wan_bench.py"
WORLD = 4

#: torch-neuronx's three cache locations; a launch takes them from its row's cache root only.
CACHE_ENV_VARS = tuple(CACHE_ENV_DIRS)
#: torchrun owns these; a copy inherited from the shell must not reach a launch.
TORCHRUN_RANK_ENV = ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR",
                     "MASTER_PORT")
#: Inherited values that would change what a launch runs; dropped from every launch env.
#: ``OMP_NUM_THREADS`` too: a launch sets it only when asked (``launch_env(omp_threads=...)``).
DROPPED_ENV = (*CACHE_ENV_VARS, *TORCHRUN_RANK_ENV, "DIFFLET_BACKEND", "DIFFLET_EXEC_MODE",
               "NEURON_RT_NUM_CORES", "NEURON_RT_VIRTUAL_CORE_SIZE",
               "TORCH_NEURONX_ENABLED_METRIC_TABLES", "TORCH_NEURONX_METRICS_DIR",
               "OMP_NUM_THREADS")
#: Set in every launch env (brief section 7 P2).
LAUNCH_ENV = {
    "HF_HUB_OFFLINE": "1",
    "DIFFLET_NO_JEMALLOC": "1",
    "DIFFLET_NEURON_PG_TIMEOUT_S": "3600",
    "PYTHONSAFEPATH": "1",
}
#: The bench script's ranks run their host work at 3 threads (brief 8.5). The CLI launches
#: export nothing (ruling D82): torchrun gives its ranks its default and the ``difflet
#: generate`` driver keeps torch's, so the CLI rows show what a user gets.
BENCH_OMP_THREADS = "3"
#: Variables every launch record lists (``None`` = not set in that launch's env).
RECORDED_ENV = (*LAUNCH_ENV, "OMP_NUM_THREADS", "DIFFLET_COMPILE_CACHE", *CACHE_ENV_VARS,
                "NKI_ENABLE_TRACE_CACHE", "PYTHONPATH", "NEURON_CC_FLAGS")

# Not anchored to line starts: torchrun runs the ranks with ``python -u``, so a rank's print
# reaches the shared log as two writes (the text, then its newline) and another rank's line
# can land between them ("... stage 91.6s[wan] neuron rank 2/4: ..."). Each record's own text
# stays contiguous, so the groups are bounded by the next keyword instead.
_RANK_LINE = re.compile(
    r"\[wan\] (?P<backend>\w+) rank (?P<rank>\d+)/(?P<world>\d+): "
    r"exec_mode=(?P<exec_mode>\S+) compiled_blocks=(?P<compiled_blocks>\d+) "
    r"unwarmed_shapes=(?P<unwarmed_shapes>.*?) fallbacks: load=(?P<load_fallbacks>.*?) "
    r"generate=(?P<generate_fallbacks>.*?) load\+warmup (?P<load_warmup_s>\d+(?:\.\d+)?)s, "
    r"stage (?P<stage_s>\d+(?:\.\d+)?)s"
)
_LATENTS_SAVED = re.compile(r"\[wan\] latents saved to (?P<path>\S+?\.pt)(?=\s|\[wan\]|$)")
_VIDEO_SAVED = re.compile(r"\[wan\] video saved to (?P<path>\S+?\.(?:mp4|pt))(?=\s|\[wan\]|$)")
_HOST_VAE_NOTE = "decodes the VAE on the host: using --host-vae"
_FAILURE_MARKERS = ("Traceback (most recent call last)", "ChildFailedError")


# ------------------------------------------------------------------------- environments


def launch_env(base: Mapping[str, str], *, cache_root: str | os.PathLike[str],
               export_cache_dirs: bool, metrics_dir: str | os.PathLike[str] | None = None,
               omp_threads: str | None = None) -> dict[str, str]:
    """The environment of one launch.

    ``DIFFLET_COMPILE_CACHE`` is ``cache_root`` (one root per benchmark row, D66), the three
    torch-neuronx cache variables are never inherited from ``base`` (brief section 8.2), nor
    are torchrun's rank variables, ``DIFFLET_BACKEND`` / ``DIFFLET_EXEC_MODE`` (the command
    names them), a stale core count or ``OMP_NUM_THREADS``, which is set only to
    ``omp_threads`` when given (the bench launch: ``BENCH_OMP_THREADS``; a CLI launch: unset,
    D82). ``export_cache_dirs`` (the bare torchrun bench launch) then exports the three
    locations under ``cache_root`` plus ``NKI_ENABLE_TRACE_CACHE=0``, exactly what
    ``difflet.cli.runner.run_stage`` exports to its ranks, so they are set before the ranks
    import torch. A CLI launch leaves that to the launcher. ``metrics_dir`` turns on
    torch-neuronx's per-graph CSV (``graph_stats``) into that directory.
    """
    env = {k: v for k, v in base.items() if k not in DROPPED_ENV}
    leaked = [name for name in (*CACHE_ENV_VARS, "OMP_NUM_THREADS") if name in env]
    if leaked:  # pragma: no cover - DROPPED_ENV covers them; kept as the stated invariant
        raise AssertionError(f"inherited variables survived: {leaked}")
    env.update(LAUNCH_ENV)
    if omp_threads is not None:
        env["OMP_NUM_THREADS"] = str(omp_threads)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["DIFFLET_COMPILE_CACHE"] = os.path.abspath(os.fspath(cache_root))
    if export_cache_dirs:
        from difflet.backends.neuron.compile import configure_compile_cache

        configure_compile_cache(env=env)
        expected = neuron_cache_env(root=env["DIFFLET_COMPILE_CACHE"], env={})
        if any(env[name] != expected[name] for name in CACHE_ENV_VARS):
            raise AssertionError(f"cache locations {[env[n] for n in CACHE_ENV_VARS]} are not "
                                 f"under {env['DIFFLET_COMPILE_CACHE']}")
    if metrics_dir is not None:
        env["TORCH_NEURONX_ENABLED_METRIC_TABLES"] = "graph_stats"
        env["TORCH_NEURONX_METRICS_DIR"] = os.path.abspath(os.fspath(metrics_dir))
    return env


def cache_dirs(cache_root: str | os.PathLike[str]) -> dict[str, Path]:
    """``neff``, ``neff_local`` and ``hlo`` under ``<cache_root>/neuron``."""
    locations = neuron_cache_env(root=os.fspath(cache_root), env={})
    return {CACHE_ENV_DIRS[name]: Path(path) for name, path in locations.items()}


# ----------------------------------------------------------------------------- commands


def bench_command(python: str, *, model_dir: str | os.PathLike[str], out_dir, steps: int,
                  guidance_scale: float, iters: int, prompt: str, seed: int,
                  height: int, width: int, num_frames: int, expect: str | None = None,
                  revision: str | None = None) -> list[str]:
    """torchrun (``-m torch.distributed.run``) of ``neuron_wan_bench.py``, 4 ranks."""
    cmd = [python, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node",
           str(WORLD), str(BENCH_SCRIPT), "--model-dir", os.fspath(model_dir),
           "--out-dir", os.fspath(out_dir), "--steps", str(steps),
           "--guidance-scale", f"{guidance_scale:g}", "--iters", str(iters),
           "--prompt", prompt, "--seed", str(seed), "--height", str(height),
           "--width", str(width), "--num-frames", str(num_frames)]
    if revision:
        cmd += ["--revision", revision]
    if expect:
        cmd += ["--expect", expect]
    return cmd


def single_backend_flag(flags: Sequence[str], backend: str = "neuron") -> list[str]:
    """``flags`` with ``--backend <backend>`` exactly once (the MATRIX entry already carries it,
    so the adapter must not add it again); any other backend named there is an error."""
    out: list[str] = []
    seen = False
    i = 0
    while i < len(flags):
        flag = flags[i]
        if flag == "--backend" or flag.startswith("--backend="):
            value = flag.split("=", 1)[1] if "=" in flag else (
                flags[i + 1] if i + 1 < len(flags) else "")
            if value != backend:
                raise ValueError(f"extra_generate_flags name --backend {value!r}, not {backend!r}")
            i += 1 if "=" in flag else 2
            if not seen:
                out += ["--backend", backend]
                seen = True
            continue
        out.append(flag)
        i += 1
    if not seen:
        out += ["--backend", backend]
    return out


def cli_generate_command(python: str, cfg, *, output, cache_dir, work_dir) -> list[str]:
    """``<python> -m difflet.cli.main generate`` for a ``BenchConfig`` (its own guidance)."""
    cmd = [python, "-m", "difflet.cli.main", "generate", "--model-id", cfg.model_id]
    if cfg.revision:
        cmd += ["--revision", cfg.revision]
    cmd += [*cfg.parallel_flags(), *cfg.shape_flags(), "--steps", str(cfg.steps)]
    if cfg.guidance_scale is not None:
        cmd += ["--guidance-scale", f"{cfg.guidance_scale:g}"]
    cmd += ["--seed", str(cfg.seed), "--prompt", cfg.prompt, "--output", os.fspath(output),
            "--cache-dir", os.fspath(cache_dir), "--work-dir", os.fspath(work_dir),
            "--keep-work-dir"]
    return cmd + single_backend_flag(cfg.extra_generate_flags)


# ------------------------------------------------------------------------------- parsing


def parse_cli_log(text: str) -> dict[str, Any]:
    """The ``[wan]`` lines of a ``difflet generate --backend neuron`` log."""
    ranks: dict[int, dict[str, Any]] = {}
    for m in _RANK_LINE.finditer(text):
        ranks[int(m["rank"])] = {
            "backend": m["backend"], "world": int(m["world"]), "exec_mode": m["exec_mode"],
            "compiled_blocks": int(m["compiled_blocks"]),
            "unwarmed_shapes": m["unwarmed_shapes"].strip(),
            "load_fallbacks": m["load_fallbacks"].strip(),
            "generate_fallbacks": m["generate_fallbacks"].strip(),
            "load_warmup_s": float(m["load_warmup_s"]), "stage_s": float(m["stage_s"]),
        }
    return {
        "ranks": dict(sorted(ranks.items())),
        "latents_saved": [m["path"] for m in _LATENTS_SAVED.finditer(text)],
        "video_saved": [m["path"] for m in _VIDEO_SAVED.finditer(text)],
        "host_vae_note": _HOST_VAE_NOTE in text,
        "failure_lines": sum(text.count(marker) for marker in _FAILURE_MARKERS),
    }


def cli_checks(parsed: Mapping[str, Any], *, world: int = WORLD,
               compiled_blocks: int = 40) -> list[tuple[str, bool, str]]:
    """What one CLI launch must show: every rank's line in compile mode with 40 compiled
    blocks, no unwarmed shape, no fallback; exactly one ``latents saved``; one video."""
    ranks = parsed["ranks"]
    lines_ok = sorted(ranks) == list(range(world))
    clean = all(r["exec_mode"] == "compile" and r["compiled_blocks"] == compiled_blocks
                and r["unwarmed_shapes"] == "[]" and r["load_fallbacks"] == "[]"
                and r["generate_fallbacks"] == "[]" for r in ranks.values())
    return [
        ("rank_lines", lines_ok, f"ranks {sorted(ranks)}"),
        ("compile_clean", lines_ok and clean,
         "exec_mode/compiled_blocks/unwarmed/fallbacks per rank: "
         + "; ".join(f"r{k} {v['exec_mode']}/{v['compiled_blocks']}/{v['unwarmed_shapes']}/"
                     f"{v['load_fallbacks']}/{v['generate_fallbacks']}" for k, v in ranks.items())),
        ("one_writer", len(parsed["latents_saved"]) == 1,
         f"{len(parsed['latents_saved'])} 'latents saved' lines"),
        ("video_saved", len(parsed["video_saved"]) == 1, f"{parsed['video_saved']}"),
        ("no_failure", parsed["failure_lines"] == 0, f"{parsed['failure_lines']} failure lines"),
    ]


def read_rank_results(out_dir: str | os.PathLike[str], world: int = WORLD) -> list[dict]:
    """Every rank's ``result-rank<r>.json``, by rank; all ``world`` ranks must be there."""
    paths = sorted(Path(out_dir).glob("result-rank*.json"))
    results = sorted((json.loads(p.read_text()) for p in paths), key=lambda r: r["rank"])
    found = [r["rank"] for r in results]
    if found != list(range(world)):
        raise ValueError(f"{out_dir}: expected result-rank0..{world - 1}.json, found ranks {found}")
    return results


# ------------------------------------------------------------------------------ summaries


def stats_dict(samples: Sequence[float]) -> dict[str, Any]:
    return Stats.from_samples(list(samples)).__dict__


def step_series_checks(deltas: Sequence[float], *, max_tail_ratio: float = 1.10,
                       max_drift: float = 0.05, window: int = 10) -> dict[str, Any]:
    """The statistical throttle checks of one step series (research-benchmark 3.3):
    ``p90 / median <= 1.10`` and the median of the first ``window`` deltas vs the last
    ``window`` within 5 %."""
    if len(deltas) < 2 * window:
        return {"ok": False, "tail_ratio": math.nan, "drift": math.nan,
                "detail": f"only {len(deltas)} deltas (need {2 * window})"}
    st = Stats.from_samples(list(deltas))
    first = Stats.from_samples(list(deltas[:window])).median
    last = Stats.from_samples(list(deltas[-window:])).median
    tail = st.p90 / st.median
    drift = abs(last - first) / first
    return {"ok": tail <= max_tail_ratio and drift <= max_drift, "tail_ratio": tail,
            "drift": drift, "first10_median": first, "last10_median": last,
            "detail": f"p90/median {tail:.4f} (max {max_tail_ratio}); first-10 vs last-10 "
                      f"median {drift * 100:.2f}% (max {max_drift * 100:.0f}%)"}


_THROTTLE_FIELD = re.compile(r"(?P<key>entries|max)=(?P<value>[^\s(]+)")


def throttle_fields(raw: Mapping[str, str] | None) -> dict[str, dict[str, str]]:
    """``entries`` and ``max`` of every core's sysfs ``ham_throttle`` string (the window
    timestamps change on every read and are dropped)."""
    out = {}
    for core, text in (raw or {}).items():
        if core.startswith("neuron_core"):
            out[core] = {m["key"]: m["value"] for m in _THROTTLE_FIELD.finditer(text)}
    return out


def throttle_changed(before: Mapping[str, str] | None, after: Mapping[str, str] | None) -> bool:
    """Whether any core's ``entries`` or ``max`` differs: a flag, not a verdict (their
    semantics are undocumented, research-benchmark 3.3)."""
    return throttle_fields(before) != throttle_fields(after)


def spread(values: Sequence[float]) -> float:
    """(max - min) / min; nan for fewer than two values."""
    values = [v for v in values if v is not None]
    if len(values) < 2 or min(values) <= 0:
        return math.nan
    return (max(values) - min(values)) / min(values)


def compile_split(compile_info: Mapping[str, Any] | None) -> dict[str, Any]:
    """One rank's NEFFs by origin, from the bench script's ``compile_info()``: an entry whose
    cache key is a Dynamo graph id is a block NEFF, any other an eager-op NEFF; a positive
    ``compilation_time_ms`` means this rank compiled it, 0 that it loaded it from disk."""
    info = compile_info or {}
    graphs = info.get("dynamo_graphs") or []
    keys = {g["graph_id"] for g in graphs}
    out = {"block_compiled": 0, "block_compile_s": 0.0, "block_loaded": 0,
           "eager_compiled": 0, "eager_compile_s": 0.0, "eager_loaded": 0,
           "compile_time_s": ((info.get("compilation_time") or {}).get("accumulator_s") or 0.0),
           "dynamo_graphs": len(graphs),
           "dynamo_fx_passes_s": sum(g.get("torch_neuronx_fx_passes_us", 0) for g in graphs) / 1e6,
           "dynamo_lower_s": sum(g.get("torch_neuronx_lower_us", 0) for g in graphs) / 1e6,
           # Dynamo's own frame compile (bytecode trace + the backend call), seconds
           "dynamo_compile_inner_s": ((info.get("dynamo_phases_s") or {})
                                      .get("_compile.compile_inner") or {}).get("total_s", 0.0),
           "error": info.get("error")}
    for entry in info.get("cache_entries") or []:
        kind = "block" if entry["cache_key"] in keys else "eager"
        if entry["compilation_time_ms"] > 0:
            out[f"{kind}_compiled"] += 1
            out[f"{kind}_compile_s"] += entry["compilation_time_ms"] / 1000.0
        else:
            out[f"{kind}_loaded"] += 1
    return out


def summarize_bench_launch(results: Sequence[Mapping[str, Any]], *, steps: int,
                           guidance_scale: float) -> dict[str, Any]:
    """One launch's numbers and checks from its four ``result-rank<r>.json``.

    Rank 0's clocks give the e2e numbers (``prepare_runtime`` start -> mp4 saved, and ->
    latents ready); the DiT step series is rank 0's, with the per-step rank-max (the
    elementwise max over ranks: what the request waits for) beside it.
    """
    rank0 = results[0]
    forwards_per_step = 2 if guidance_scale > 1.0 else 1
    ranks = []
    for r in results:
        phases = r["phase_seconds"]
        ranks.append({
            "rank": r["rank"], "imports_s": r["imports_s"],
            "prepare_runtime_s": r["prepare_runtime_s"], "load_s": r["load_s"],
            "phase_seconds": phases, "counters": r["counters"],
            "compiled_blocks": r["compiled_blocks"], "load_fallbacks": r["load_fallbacks"],
            "peak_device_mem_gib": r["peak_device_mem_gib"],
            "threads": r["torch_threads"], "env": r["env"],
            "hlo_lock_dir_at_import": r["hlo_lock_dir_at_import"],
            "metrics_setup": r["metrics_setup"],
            "compile": compile_split(r.get("compile_end")),
            "neff_files_end": r["snapshots"]["end"]["neff_files"],
            "hlo_files_end": r["snapshots"]["end"]["hlo_files"],
            "dynamo_unique_graphs": r["snapshots"]["end"]["dynamo"]["unique_graphs"],
            "graph_breaks": r["snapshots"]["end"]["dynamo"]["graph_breaks"],
        })

    def rank_max(getter):
        values = [getter(r) for r in results]
        values = [v for v in values if v is not None]
        return max(values) if values else None

    iterations = []
    n_iters = len(rank0["iterations"])
    for i in range(n_iters):
        per_rank = [r["iterations"][i] for r in results]
        it0 = per_rank[0]
        deltas0 = it0["step_deltas_s"]
        n = min(len(p["step_deltas_s"]) for p in per_rank)
        rank_max_series = [max(p["step_deltas_s"][k] for p in per_rank) for k in range(n)]
        hashes = [p["latents_sha256"] for p in per_rank]
        prompt_encodes = [e["seconds"] for e in it0["encodes"] if e["tag"] == "prompt"]
        negative_encodes = [e["seconds"] for e in it0["encodes"] if e["tag"] == "negative"]
        encodes = {"prompt": prompt_encodes[0] if prompt_encodes else None,
                   "negative": sum(negative_encodes) if negative_encodes else None}
        decode = it0.get("decode") or {}
        timings = decode.get("timings") or {}
        checks = {
            "stamps": all(p["stamps"] == steps for p in per_rank),
            "deltas": all(len(p["step_deltas_s"]) == steps - 1 for p in per_rank),
            "forwards": all(len(p["forwards_s"]) == steps * forwards_per_step for p in per_rank),
            "fallbacks": all(p["fallbacks"] == [] for p in per_rank),
            "unwarmed_shapes": all(p["unwarmed_shapes"] == [] for p in per_rank),
            "latents_equal_on_ranks": len(set(hashes)) == 1,
            "mp4_written": bool(decode.get("mp4_exists")),
        }
        # Judged by the driver: required of a warm launch, evidence only on a cold one (its
        # first generate may still compile eager-op NEFFs the warm-up did not need).
        no_compile = all(p["counters"]["TotalCompilations"] == 0 for p in per_rank)
        iterations.append({
            "index": it0["index"],
            "generate_s": it0["generate_s"],
            "generate_s_rank_max": max(p["generate_s"] for p in per_rank),
            "t_start_s": it0["t_start_s"], "t_latents_s": it0["t_latents_s"],
            "t_mp4_saved_s": it0.get("t_mp4_saved_s"),
            "text_encode_s": encodes.get("prompt"),
            "negative_encode_s": encodes.get("negative"),
            "umt5_encodes": it0["umt5_encodes"],
            "denoise_s": it0["denoise_s"],
            "iteration0_s": it0["iteration0_s"],
            "vae_decode_s": timings.get("vae_decode_s"),
            "host_decode_s": timings.get("decode_latents_host_s"),
            "mp4_write_s": timings.get("mp4_write_s"),
            "decode_threads": decode.get("threads"),
            "step": stats_dict(deltas0),
            "step_rank_max": stats_dict(rank_max_series),
            "step_median_by_rank": [Stats.from_samples(p["step_deltas_s"]).median
                                    for p in per_rank],
            "step_checks": step_series_checks(deltas0),
            "forward": stats_dict(it0["forwards_s"]),
            "counters_by_rank": [p["counters"] for p in per_rank],
            "fallbacks_by_rank": [p["fallbacks"] for p in per_rank],
            "unwarmed_by_rank": [p["unwarmed_shapes"] for p in per_rank],
            "latents_sha256": hashes[0] if len(set(hashes)) == 1 else hashes,
            "latents_stats": it0["latents_stats"],
            "latents_pt_sha256": it0.get("latents_pt_sha256"),
            "frames": decode.get("frames"),
            "mp4": decode.get("mp4"), "mp4_sha256": decode.get("mp4_sha256"),
            "mp4_bytes": decode.get("mp4_bytes"),
            "throttle_before": it0.get("throttle_before"),
            "throttle_after": it0.get("throttle_after"),
            "throttle_after_decode": it0.get("throttle_after_decode"),
            "throttle_changed": throttle_changed(it0.get("throttle_before"),
                                                 it0.get("throttle_after")),
            "max_memory_allocated_gib_by_rank": [p["max_memory_allocated_gib"]
                                                 for p in per_rank],
            "waited_for_rank0_s": [p.get("waited_for_rank0_s") for p in per_rank[1:]],
            "no_compile": no_compile,
            "checks": checks,
            "ok": all(checks.values()),
        })

    first = iterations[0] if iterations else {}
    medians = [it["step"]["median"] for it in iterations]
    latent_hashes = {it["latents_sha256"] if isinstance(it["latents_sha256"], str) else None
                     for it in iterations}
    checks = {
        "every_iteration_ok": all(it["ok"] for it in iterations),
        "load_fallbacks": all(r["load_fallbacks"] == [] for r in results),
        "compiled_blocks": all(r["compiled_blocks"] == 40 for r in results),
        "latents_equal_across_iterations": len(latent_hashes) == 1 and None not in latent_hashes,
        "hlo_locks_in_cache": all(r["hlo_lock_dir_at_import"]
                                  == r["env"]["TORCH_NEURONX_HLO_CACHE_DIR"] for r in results),
    }
    return {
        "ranks": ranks,
        "rank_max": {
            "warmup_forward_s": rank_max(lambda r: r["phase_seconds"].get("warmup forward")),
            "load_checkpoint_s": rank_max(lambda r: r["phase_seconds"].get("load checkpoint")),
            "load_s": rank_max(lambda r: r["load_s"]),
            "prepare_runtime_s": rank_max(lambda r: r["prepare_runtime_s"]),
            "peak_device_mem_gib": rank_max(lambda r: r["peak_device_mem_gib"]),
        },
        "text_encoder_load_s": rank0["phase_seconds"].get("text encoder load"),
        "e2e_script_s": first.get("t_mp4_saved_s"),
        "e2e_latents_s": first.get("t_latents_s"),
        "prepare_start_epoch": rank0["epoch"]["prepare_start"],
        "script_start_epoch": rank0["epoch"]["script_start"],
        "iterations": iterations,
        "resident_step_median_spread": spread(medians),
        "resident_within_2pct": (spread(medians) <= 0.02) if len(medians) > 1 else None,
        "checks": checks,
        "ok": all(checks.values()),
    }


def counter_rule(summary: Mapping[str, Any], expect: str, new_neffs: int) -> tuple[bool, str]:
    """C9's rule on the whole process's counters (``pre_prepare`` -> end) of every rank.

    warm: ``TotalCompilations == 0 and PersistentHits >= 1`` on every rank and no new
    ``.neff`` file in the row's cache. cold: at least one new ``.neff`` and at least one rank
    compiled (``TotalCompilations >= 1``); in a cold launch a rank that waited on another's
    lock shows only persistent hits, so cold is decided by any rank compiling.
    """
    runs = [r["counters"]["run"] for r in summary["ranks"]]
    text = " ".join(f"r{r['rank']} {c['TotalCompilations']}/{c['PersistentHits']}/"
                    f"{c['InMemoryHits']}" for r, c in zip(summary["ranks"], runs))
    text = f"Total/PersistentHits/InMemoryHits {text}; new .neff {new_neffs}"
    if expect == "warm":
        ok = new_neffs == 0 and all(c["TotalCompilations"] == 0 and c["PersistentHits"] >= 1
                                    for c in runs)
    elif expect == "cold":
        ok = new_neffs >= 1 and any(c["TotalCompilations"] >= 1 for c in runs) and all(
            c["TotalCompilations"] + c["PersistentHits"] >= 1 for c in runs)
    else:
        raise ValueError(f"expect must be 'cold' or 'warm', got {expect!r}")
    return ok, text


# ------------------------------------------------------------------------------ processes


#: How long a stopped launch gets to stop its ranks before it is killed: torchrun forwards the
#: signal, waits TORCH_ELASTIC_SHUTDOWN_TIMEOUT (30 s) and SIGKILLs the ranks still running;
#: ``difflet generate`` waits for its torchrun up to 60 s more (difflet/cli/runner.py).
STOP_GRACE_S = 90.0


def descendants(root: int, parents: Mapping[int, int]) -> list[int]:
    """Every pid below ``root`` in ``parents`` (pid -> parent pid), breadth first."""
    children: dict[int, list[int]] = {}
    for pid, ppid in parents.items():
        children.setdefault(ppid, []).append(pid)
    out, queue = [], list(children.get(root, []))
    while queue:
        pid = queue.pop(0)
        if pid not in out:
            out.append(pid)
            queue.extend(children.get(pid, []))
    return out


def process_parents() -> dict[int, int]:
    """pid -> parent pid of every process, from ``/proc/<pid>/stat``."""
    parents = {}
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            text = stat.read_text()
        except OSError:
            continue
        # "pid (comm) state ppid ...": comm may contain spaces and parentheses
        parents[int(text.split(" ", 1)[0])] = int(text.rsplit(")", 1)[1].split()[1])
    return parents


def process_stat(pid: int) -> tuple[str, str] | None:
    """``(state, start time in clock ticks)`` of a live process, from ``/proc/<pid>/stat``;
    None when it is gone. The start time tells a pid from a later process reusing it."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return None
    return fields[0], fields[19]


def _tree(pid: int) -> dict[int, str | None]:
    """The descendants of ``pid`` with their start times."""
    out = {}
    for child in descendants(pid, process_parents()):
        stat = process_stat(child)
        out[child] = stat[1] if stat else None
    return out


def wait_through_interrupts(proc, timeout: float | None) -> int | None:
    """``proc.wait(timeout)`` that keeps waiting through repeated interrupts: another Ctrl-C or
    SIGTERM while a launch is being stopped must not abandon the stop (as
    ``difflet.cli.runner._await_torchrun_exit`` keeps waiting). None when ``timeout`` expires;
    ``timeout=None`` waits for the exit."""
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        remaining = None if deadline is None else max(deadline - time.monotonic(), 0.0)
        try:
            return proc.wait(timeout=remaining)
        except KeyboardInterrupt:
            print(f"[neuron bench] interrupted again; still stopping launch pid {proc.pid}",
                  file=sys.stderr, flush=True)
        except subprocess.TimeoutExpired:
            return None


def stop_launch(proc: subprocess.Popen, grace: float = STOP_GRACE_S) -> int:
    """Stop a launch started in a session of its own (``run_logged``) without orphaning ranks.

    SIGINT goes to the launch's whole process group, as a terminal's Ctrl-C would: torchrun
    (or ``difflet generate`` and the torchrun it started) forwards it to its ranks, which run
    in sessions of their own. After ``grace`` seconds whatever is left gets SIGKILL: the
    process group and every descendant seen before or after the SIGINT (the ranks are not in
    the group), with a warning naming them. Repeated interrupts do not cut either wait short.
    Once the launch has exited, by itself or killed, every descendant seen before that is
    still running (the same pid with the same start time) is killed too: a launcher that gave
    up on its torchrun can leave ranks behind. Returns the launch's exit code."""
    import signal

    seen = _tree(proc.pid)
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    code = wait_through_interrupts(proc, grace)
    if code is None:
        seen.update(_tree(proc.pid))
        print(f"[neuron bench] launch pid {proc.pid} did not stop {grace:.0f}s after SIGINT; "
              f"killing its process group and descendants {sorted(seen)}", file=sys.stderr,
              flush=True)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        code = wait_through_interrupts(proc, None)
    left = []
    for pid, start in sorted(seen.items()):
        stat = process_stat(pid)
        if start is not None and stat is not None and stat[1] == start and stat[0] != "Z":
            left.append(pid)
    if left:
        print(f"[neuron bench] launch pid {proc.pid} exited and left {left} running; killing "
              "them", file=sys.stderr, flush=True)
        for pid in left:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    return code


def run_logged(cmd: Sequence[str], env: Mapping[str, str], log: Path, *, timeout: float,
               stop_grace: float = STOP_GRACE_S, popen=subprocess.Popen) -> dict[str, Any]:
    """Run ``cmd`` with stdout+stderr into ``log``; wall clock and epochs around it.

    The launch runs in a session of its own, so only this function signals it. On timeout,
    and when this process is interrupted (KeyboardInterrupt; the driver maps SIGTERM to it),
    ``stop_launch`` stops it the way ``difflet.cli.runner._run_torchrun`` stops torchrun:
    SIGINT to the launch, a bounded wait for its ranks to stop, then SIGKILL of what is left
    (D39). An interrupt is re-raised after the launch has stopped."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as handle:
        start_epoch = time.time()
        start = time.perf_counter()
        proc = popen(list(cmd), env=dict(env), stdout=handle, stderr=subprocess.STDOUT,
                     start_new_session=True)
        timed_out = False
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            code = stop_launch(proc, stop_grace)
        except BaseException:
            stop_launch(proc, stop_grace)
            raise
        wall = time.perf_counter() - start
    return {"cmd": list(cmd), "exit_code": code, "wall_s": wall, "start_epoch": start_epoch,
            "end_epoch": start_epoch + wall, "timed_out": timed_out, "log": str(log)}


def inspect_outputs(python: str, env: Mapping[str, str], latents: Path,
                    mp4: Path | None = None) -> dict[str, Any]:
    """The bench script's CPU ``--inspect-latents`` helper in a subprocess (this process
    stays torch-free): tensor/file sha256 of a latents.pt, the mp4 read back."""
    cmd = [python, str(BENCH_SCRIPT), "--inspect-latents", str(latents)]
    if mp4 is not None:
        cmd += ["--inspect-mp4", str(mp4)]
    out = subprocess.run(cmd, env=dict(env), capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        return {"error": out.stderr.strip().splitlines()[-5:]}
    return json.loads(out.stdout.strip().splitlines()[-1])


def device_processes(neuron_ls_json: str) -> list[Any]:
    """The processes ``neuron-ls -j`` lists on any device (``[]`` = cores free)."""
    devices = json.loads(neuron_ls_json)
    return [p for d in devices for p in (d.get("neuron_processes") or [])]


_BUSY = re.compile(r"torchrun|torch\.distributed\.run|difflet\.cli\.stage|"
                   r"tests/manual/check_neuron|neuron_oracle|neuron_wan_bench")


def busy_processes(ps_output: str, ignore_pids: Sequence[int] = ()) -> list[str]:
    """``ps -eo pid,args`` lines of device jobs (the brief's D9 pattern plus
    ``torch.distributed.run``, the module form both launchers use), minus our own."""
    lines = []
    for line in ps_output.splitlines()[1:]:
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit() or int(parts[0]) in ignore_pids:
            continue
        if os.path.basename(parts[1].split(None, 1)[0]) == "grep":
            continue  # someone's `ps | grep -E 'torchrun|...'` (the D9 check drops it too)
        if _BUSY.search(parts[1]) and "neuron_wan_bench.py --inspect" not in parts[1]:
            lines.append(line.strip())
    return lines


# -------------------------------------------------------------------------------- adapter


class NeuronAdapter(BackendAdapter):
    """``benchmark.bench --backend neuron``: one bench launch per ``run_generate``.

    The NEFF cache is ``<cache_dir>/neuron`` (default ``~/.cache/difflet``), so the first
    launch on a fresh ``cache_dir`` is cold and the later ones warm. For the full cold / warm /
    resident / CLI matrix use ``python -m benchmark.neuron_cold_warm``.
    """

    name = "neuron"

    def __init__(self, *, cache_dir: str | None = None, log_dir: str | None = None,
                 python: str | None = None, iters: int = 1):
        from benchmark.models import logs_dir

        self.cache_dir = os.path.abspath(os.path.expanduser(cache_dir or "~/.cache/difflet"))
        self.log_dir = Path(log_dir or logs_dir()).absolute()
        self.python = python or sys.executable
        self.iters = iters
        self._model_dir: str | None = None
        self._launches = 0
        self._torch_threads: dict[str | None, int | None] = {}

    def device_info(self) -> str:
        try:
            out = subprocess.run(["/opt/aws/neuron/bin/neuron-ls", "-j"], capture_output=True,
                                 text=True, timeout=30).stdout
            d = json.loads(out)[0]
            gib = round(int(d.get("memory_size", 0)) / 2**30)
            return (f"{d.get('instance_type', 'trn2')} / {d.get('nc_count', '?')} NeuronCores"
                    f" (logical config {d.get('logical_neuroncore_config', '?')}) / "
                    f"{gib} GiB/device")
        except Exception:  # noqa: BLE001
            return "Trainium (neuron-ls unavailable)"

    def toolchain(self) -> dict[str, str]:
        code = ("import importlib.metadata as m\n"
                "for p in ['torch','torch-neuronx','neuronx-cc','nki','diffusers',"
                "'transformers','numpy']:\n"
                "    try: print(p+'='+m.version(p))\n"
                "    except Exception: pass")
        try:
            out = subprocess.run([self.python, "-I", "-c", code], capture_output=True,
                                 text=True, timeout=60).stdout
            return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
        except Exception:  # noqa: BLE001
            return {}

    def model_dir(self, spec) -> str:
        if self._model_dir is None:
            from difflet.pipeline.path_resolver import resolve_model_path

            self._model_dir = resolve_model_path(spec.model_id, revision=spec.revision,
                                                 local_files_only=True)
        return self._model_dir

    def prepare(self, spec) -> None:
        try:
            self.model_dir(spec)
        except Exception:  # noqa: BLE001 - not in the local HF cache yet
            cmd = [self.python, "-m", "difflet.cli.main", "download", "--model-id",
                   spec.model_id] + (["--revision", spec.revision] if spec.revision else [])
            subprocess.run(cmd, check=True, env={**os.environ, "PYTHONPATH": str(REPO_ROOT)})
            self.model_dir(spec)

    def compile(self, spec) -> tuple[float, dict[str, float]]:
        # Nothing is built ahead of time: the per-block compile runs inside the first launch's
        # warm-up forward (neuron_cold_warm.py measures it as cold minus warm).
        return 0.0, {"aot": 0.0}

    def run_bench(self, spec, out_dir: Path, *, cache_root: str | None = None,
                  expect: str | None = None, iters: int | None = None,
                  timeout: float = 3600.0) -> dict[str, Any]:
        """One torchrun launch of the bench script; returns the launch record + summary."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        root = cache_root or self.cache_dir
        env = launch_env(os.environ, cache_root=root, export_cache_dirs=True,
                         metrics_dir=out_dir, omp_threads=BENCH_OMP_THREADS)
        env["DIFFLET_DISABLE_PREWARM"] = "1"
        cmd = bench_command(self.python, model_dir=self.model_dir(spec), out_dir=out_dir,
                            steps=spec.steps, guidance_scale=spec.guidance_scale or 1.0,
                            iters=iters or self.iters, prompt=spec.prompt, seed=spec.seed,
                            height=spec.height, width=spec.width,
                            num_frames=spec.num_frames, expect=expect,
                            revision=spec.revision)
        launch = run_logged(cmd, env, out_dir / "launch.log", timeout=timeout)
        launch["env"] = {k: env.get(k) for k in RECORDED_ENV}
        if launch["exit_code"] != 0:
            return {"launch": launch, "summary": None}
        try:
            results = read_rank_results(out_dir)
            summary = summarize_bench_launch(results, steps=spec.steps,
                                             guidance_scale=spec.guidance_scale or 1.0)
        except Exception as exc:  # noqa: BLE001 - e.g. a rank wrote no JSON despite exit 0
            return {"launch": launch, "summary": None,
                    "error": f"{type(exc).__name__}: {exc}"}
        return {"launch": launch, "summary": summary}

    def run_generate(self, spec) -> dict:
        self._launches += 1
        out_dir = self.log_dir / f"neuron_{spec.model_type}_launch{self._launches}"
        record = self.run_bench(spec, out_dir)
        summary = record["summary"]
        if summary is None:
            raise RuntimeError(f"bench launch failed ({record['launch']['exit_code']}); "
                               f"see {record['launch']['log']}")
        first = summary["iterations"][0]
        frames = first.get("frames") or {}
        return {
            "wall_seconds": summary["e2e_script_s"],
            "launch_wall_seconds": record["launch"]["wall_s"],
            "load_seconds": summary["rank_max"]["load_checkpoint_s"],
            "step_seconds": first["step"]["samples"],
            "step_basis": "real loop, neuron synchronize both sides, step 0 excluded, "
                          f"n={first['step']['n']}",
            "peak_mem_gb": summary["rank_max"]["peak_device_mem_gib"],
            "output": OutputInfo(shape=frames.get("shape"), dtype=frames.get("dtype"),
                                 finite=frames.get("finite"), min=frames.get("min"),
                                 max=frames.get("max"), mean=frames.get("mean"),
                                 std=frames.get("std"),
                                 note=f"{first.get('mp4_bytes')} byte mp4").__dict__,
            "stage_seconds": {
                "text_encoder_load_s": summary["text_encoder_load_s"],
                "text_encode": first["text_encode_s"],
                "denoise": first["denoise_s"],
                "vae_decode": first["host_decode_s"],
            },
            "log": record["launch"]["log"],
        }

    def run_cli_generate(self, spec, out_dir: Path, *, cache_root: str | None = None,
                         command: list[str] | None = None, timeout: float = 3600.0
                         ) -> dict[str, Any]:
        """One ``difflet generate --backend neuron`` (or ``command``) in the user's default
        environment (no ``OMP_NUM_THREADS``, D82): wall, parsed lines, hashes, and what a
        re-render needs without touching the files again (``timeline``, ``threads``)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        root = cache_root or self.cache_dir
        env = launch_env(os.environ, cache_root=root, export_cache_dirs=False)
        mp4, work = out_dir / "out.mp4", out_dir / "work"
        cmd = command or cli_generate_command(self.python, spec, output=mp4, cache_dir=root,
                                              work_dir=work)
        launch = run_logged(cmd, env, out_dir / "cli.log", timeout=timeout)
        launch["env"] = {k: env.get(k) for k in RECORDED_ENV}
        text = Path(launch["log"]).read_text(errors="replace")
        parsed = parse_cli_log(text)
        latents = work / "latents.pt"
        inspected = (inspect_outputs(self.python, env, latents, mp4 if mp4.exists() else None)
                     if latents.exists() else {"error": "no latents.pt"})
        return {"launch": launch, "parsed": parsed, "checks": cli_checks(parsed),
                "inspect": inspected, "mp4": str(mp4), "latents": str(latents),
                "files": {"latents": "work/latents.pt", "mp4": "out.mp4", "log": "cli.log"},
                "timeline": file_timeline(launch, latents, mp4),
                "threads": {"launch_env_omp_num_threads": env.get("OMP_NUM_THREADS"),
                            "driver_torch_threads": self.torch_threads(env),
                            "rank_omp_lines": torchrun_omp_lines(text)}}

    def torch_threads(self, env: Mapping[str, str]) -> int | None:
        """``torch.get_num_threads()`` of a process started with ``env`` (the CLI driver's,
        which decodes the VAE on the host): measured once per ``OMP_NUM_THREADS`` value."""
        key = env.get("OMP_NUM_THREADS")
        if key not in self._torch_threads:
            out = subprocess.run([self.python, "-I", "-c",
                                  "import torch; print(torch.get_num_threads())"],
                                 env=dict(env), capture_output=True, text=True, timeout=300)
            lines = out.stdout.strip().splitlines()
            self._torch_threads[key] = int(lines[-1]) if out.returncode == 0 and lines else None
        return self._torch_threads[key]


def file_timeline(launch: Mapping[str, Any], latents: Path, mp4: Path) -> dict[str, float] | None:
    """A CLI launch split by its files' mtimes, taken right after it: process start ->
    latents.pt saved by rank 0 (spawn, load, generate), latents.pt -> mp4 written (rank
    teardown, then the driver's host VAE load + decode + export), mp4 -> process exit."""
    if not latents.is_file() or not mp4.is_file():
        return None
    t_latents, t_mp4 = latents.stat().st_mtime, mp4.stat().st_mtime
    return {"to_latents_s": t_latents - launch["start_epoch"], "decode_s": t_mp4 - t_latents,
            "after_mp4_s": launch["end_epoch"] - t_mp4}


def torchrun_omp_lines(text: str) -> list[str]:
    """torchrun's own note when it sets ``OMP_NUM_THREADS=1`` for its ranks (unset in env)."""
    return [line.strip() for line in text.splitlines() if "OMP_NUM_THREADS" in line][:2]


__all__ = ["NeuronAdapter"]
