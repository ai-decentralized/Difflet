"""Synthetic ``result-rank<r>.json`` payloads, shaped like ``benchmark/neuron_wan_bench.py``'s,
and the launch records ``benchmark/neuron_cold_warm.py`` keeps, for the CPU tests."""

from __future__ import annotations

import copy
from typing import Any

LATENTS = "8e5eaffed4cf6f3a79a94dc11664c4f6b49767375fc6f5c451a39330bed990f6"
MP4 = "a9f9004fc1343eab167e70f4522f382243f40b7b83765b726e01d02cb01b7f34"
HLO = "/records/caches/g5.0-main/neuron/hlo"
COUNTER_KEYS = ("TotalCompilations", "PersistentHits", "PersistentMisses", "InMemoryHits",
                "InMemoryMisses")


def counters(total: int = 0, hits: int = 0, in_memory: int = 0) -> dict[str, int]:
    return {"TotalCompilations": total, "PersistentHits": hits, "PersistentMisses": 0,
            "InMemoryHits": in_memory, "InMemoryMisses": 0}


def compile_info(*, block_ms: int = 0, eager_ms: tuple[int, ...] = ()) -> dict[str, Any]:
    entries = [{"cache_key": "block-key", "neff_size_bytes": 5_050_000, "access_count": 80,
                "compilation_time_ms": block_ms}]
    entries += [{"cache_key": f"eager-{i}", "neff_size_bytes": 20_000, "access_count": 3,
                 "compilation_time_ms": ms} for i, ms in enumerate(eager_ms)]
    accumulated = (block_ms + sum(eager_ms)) / 1000.0
    return {
        "compilation_time": {"accumulator_s": accumulated, "total_samples": len(entries),
                             "samples_s": []},
        "cache_entries": entries,
        "dynamo_graphs": [{"graph_id": "block-key", "torch_neuronx_fx_passes_us": 1100,
                           "torch_neuronx_lower_us": 187000,
                           "torch_neuronx_compile_us": block_ms * 1000}],
        "dynamo_phases_s": {},
    }


def rank_result(rank: int, *, steps: int = 50, guidance: float = 5.0, iters: int = 3,
                warmup: float = 13.0, load_checkpoint: float = 3.6, step: float = 1.07,
                forward: float = 0.53, latents: str = LATENTS, run_counters=None,
                generate_counters=None, block_ms: int = 0, eager_ms: tuple[int, ...] = (),
                fallbacks: tuple[str, ...] = (), unwarmed=(), prepare_epoch: float = 1000.0,
                deltas: list[float] | None = None) -> dict[str, Any]:
    forwards_per_step = 2 if guidance > 1.0 else 1
    series = list(deltas) if deltas is not None else [step] * (steps - 1)
    iterations = []
    t = 40.0
    for index in range(1, iters + 1):
        it: dict[str, Any] = {
            "index": index, "t_start_s": t, "t_latents_s": t + 56.0, "generate_s": 56.0,
            "encodes": [{"tag": "prompt", "t_s": t, "seconds": 2.0}]
            + ([{"tag": "negative", "t_s": t + 3, "seconds": 1.5}] if guidance > 1.0 else []),
            "umt5_encodes": [{"tokens": 16, "threads": 12, "seconds": 1.9}] if rank == 0 else [],
            "denoise_s": 53.5, "iteration0_s": 2.6, "stamps": steps,
            "step_deltas_s": list(series),
            "forwards_s": [forward] * (steps * forwards_per_step),
            "counters": dict(generate_counters or counters(in_memory=7406)),
            "fallbacks": list(fallbacks), "unwarmed_shapes": list(unwarmed),
            "max_memory_allocated_gib": 8.03, "latents_sha256": latents,
            "latents_stats": {"shape": [1, 16, 3, 60, 104], "dtype": "torch.float32",
                              "finite": True, "min": -3.0, "max": 3.0, "mean": 0.0, "std": 1.0},
            "throttle_before": ({"neuron_core0": "entries=1 mean=128/255(50%)",
                                 "power_utilization": "POWER_STATUS_VALID,1,2,3"}
                                if rank == 0 else None),
            "host_memory": {},
        }
        if rank == 0:
            it.update({
                "latents_pt": f"/out/iter{index}_latents.pt",
                "latents_pt_sha256": "e639f314",
                "decode": {"timings": {"vae_decode_s": 51.9, "mp4_write_s": 0.2,
                                       "decode_latents_host_s": 53.0},
                           "frames": {"shape": [1, 3, 9, 480, 832], "dtype": "torch.float32",
                                      "finite": True, "min": -1.1, "max": 1.0, "mean": -0.2,
                                      "std": 0.4, "per_frame_std": [0.4] * 9,
                                      "saved_mp4": True},
                           "threads": 12, "mp4": f"/out/iter{index}.mp4", "mp4_exists": True,
                           "mp4_bytes": 101970, "mp4_sha256": MP4, "wall_s": 53.1},
                "t_mp4_saved_s": t + 56.0 + 53.0,
                "throttle_after": {"neuron_core0": "entries=1 mean=128/255(50%)",
                                   "power_utilization": "POWER_STATUS_VALID,4,5,6"},
            })
        else:
            it["waited_for_rank0_s"] = 53.0
        iterations.append(it)
        t += 56.0 + 53.0 + 0.5
    run = run_counters or counters(hits=39, in_memory=7459)
    env = {"OMP_NUM_THREADS": "3", "TORCH_NEURONX_HLO_CACHE_DIR": HLO}
    return {
        "rank": rank, "world_size": 4, "args": {"steps": steps, "guidance_scale": guidance},
        "env": env, "hlo_lock_dir_at_import": HLO,
        "torch_threads": {"default": 3, "encoder": 12, "decode": 12 if rank == 0 else None},
        "epoch": {"script_start": prepare_epoch - 9.0, "prepare_start": prepare_epoch},
        "imports_s": 8.0, "prepare_runtime_s": 2.0, "load_s": 30.0,
        "metrics_setup": {"enabled_before_prepare_runtime": True,
                          "runtime_initialized_when_enabled": False},
        "phase_seconds": {"runtime init": 0.03, "build on meta": 0.08,
                          "load checkpoint": load_checkpoint, "eval": 0.01, "compile": 0.3,
                          "warmup forward": warmup, "text encoder load": 0.7},
        "load_fallbacks": [], "compiled_blocks": 40,
        "warmup_shapes": [[1, 16, 3, 60, 104], [1], [1, 512, 4096]],
        "load_report": {"missing": 0, "unexpected": 0},
        "snapshots": {"end": {"dynamo": {"unique_graphs": 1, "graph_breaks": 0},
                              "neff_files": 47, "hlo_files": 132}},
        "counters": {"prepare_runtime": counters(hits=1), "load": counters(hits=38),
                     "run": run},
        "compile_after_load": compile_info(block_ms=block_ms, eager_ms=eager_ms),
        "compile_end": compile_info(block_ms=block_ms, eager_ms=eager_ms),
        "iterations": iterations,
        "evidence": {"fallbacks": [], "unwarmed_shapes": []},
        "peak_device_mem_gib": 8.03, "t_end_s": 400.0,
    }


def launch_results(**kwargs) -> list[dict[str, Any]]:
    return [rank_result(rank, **kwargs) for rank in range(4)]


def cold_results(**kwargs) -> list[dict[str, Any]]:
    """Cold: rank 0 compiles the block NEFF and the eager-op NEFFs, ranks 1-3 compile their own
    block NEFF (rank-specific) and load the shared eager-op NEFFs."""
    out = []
    for rank in range(4):
        out.append(rank_result(
            rank, warmup=78.0,
            run_counters=counters(total=44 if rank == 0 else 1, hits=1 if rank == 0 else 44,
                                  in_memory=7459),
            block_ms=55_000, eager_ms=(800,) * 43 if rank == 0 else (),
            **kwargs))
    return out


HOST = {"device": "trn2.3xlarge / 4 NeuronCores (logical config 2) / 96 GiB/device",
        "toolchain": {"torch": "2.14.0+cpu"}, "sudo_available": True, "epoch": 1.0}
TREE = "54da3e39d41e0e55e42e9ccd2c4f20fd83a80551"


def provenance(head: str = "3763193977f4a508f72f713444d100f65fa88b08") -> dict[str, Any]:
    return {"head": head, "worktree": "/wt", "status": "?? benchmark/neuron_wan_bench.py",
            "difflet_tree": TREE, "difflet_diff_vs_head": "",
            "files_sha256": {"benchmark/neuron_wan_bench.py": "aa"},
            "loaded_sha256": {"benchmark/neuron_cold_warm.py": "bb"},
            "neuronx_cc": ["NeuronX Compiler version 2.28"], "neuron_cc_flags": None,
            "python": "/venv/bin/python"}


def launch_record(summary, *, name: str, guidance: float, kind: str = "bench",
                  expect: str = "warm", wall: float = 400.0, new_neffs=None,
                  start_epoch: float = 990.0, checks=None, ok: bool = True,
                  omp: str | None = "3") -> dict[str, Any]:
    return {
        "launch": {"name": name, "kind": kind, "guidance": guidance, "cache": "main",
                   "expect": expect, "fresh_cache": expect == "cold", "drop_page_cache": False},
        "run": {"exit_code": 0, "wall_s": wall, "start_epoch": start_epoch,
                "end_epoch": start_epoch + wall, "timed_out": False,
                "env": {"OMP_NUM_THREADS": omp}},
        "summary": copy.deepcopy(summary),
        "state_before": {"meminfo": {"Cached_gib": 80.0}},
        "new_neffs": dict(new_neffs or {}),
        "checks": list(checks or [{"name": "exit", "ok": True, "detail": "", "hard": True}]),
        "ok": ok, "host": dict(HOST), "provenance": provenance(),
    }


def cli_record(*, name: str, guidance: float, wall: float, kind: str = "cli",
               expect: str = "warm", new_neffs=None, tensor: str = LATENTS,
               ok: bool = True, drop=None, omp: str | None = None,
               driver_threads: int = 6,
               head: str = "386a03181a10a1106fe00d1cf0b29b21e605ac99") -> dict[str, Any]:
    rank_lines = {str(r): {"backend": "neuron", "world": 4, "exec_mode": "compile",
                           "compiled_blocks": 40, "unwarmed_shapes": "[]",
                           "load_fallbacks": "[]", "generate_fallbacks": "[]",
                           "load_warmup_s": 30.0, "stage_s": 90.0} for r in range(4)}
    record = {
        "launch": {"name": name, "kind": kind, "guidance": guidance, "cache": "main",
                   "expect": expect, "fresh_cache": False,
                   "drop_page_cache": drop is not None},
        "run": {"exit_code": 0, "wall_s": wall, "start_epoch": 0.0, "end_epoch": wall,
                "timed_out": False},
        "cli": {"launch": {"env": {"HF_HUB_OFFLINE": "1", "DIFFLET_NO_JEMALLOC": "1",
                                   "DIFFLET_NEURON_PG_TIMEOUT_S": "3600",
                                   "PYTHONSAFEPATH": "1", "OMP_NUM_THREADS": omp}},
                "parsed": {"ranks": rank_lines}, "inspect": {
            "tensor_sha256": tensor, "file_sha256": "e639f3146c3619406267c317c9798b8ead7300535c7835a7d7c7c0de54769028",
            "mp4": {"sha256": MP4}}},
        "new_neffs": dict(new_neffs or {}),
        "checks": [{"name": "exit", "ok": ok, "detail": "", "hard": True}],
        "ok": ok, "host": dict(HOST), "provenance": provenance(head),
        "timeline": {"to_latents_s": wall - 61.0, "decode_s": 60.0, "after_mp4_s": 1.0},
        "threads": {"launch_env_omp_num_threads": omp, "driver_torch_threads": driver_threads,
                    "rank_omp_lines": []},
    }
    if drop is not None:
        record["page_cache_drop"] = drop
    return record
