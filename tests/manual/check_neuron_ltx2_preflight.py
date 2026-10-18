"""Manual check: the LTX-2 DiT through the neuron application on torchrun ranks.

The device preflight of the LTX-2 port to the PyTorch-native neuron backend, modelled on
``tests/manual/check_neuron_qwen_preflight.py`` (it imports the Wan preflight's measurement
helpers, so the gates count NEFFs, compilations and memory the same way): it builds
``NeuronLtx2TransformerApplication`` under ``get_backend('neuron').prepare_runtime``, loads the
first ``--num-layers`` blocks of the real checkpoint (48 = the whole model), warms up, runs one
DiT forward per reference timestep through the composed application's host<->device adapter
(``_OnDeviceLtx2Transformer``, the orchestrator's own DiT call) under ``track_fallbacks()``,
then times the device forward on device-resident inputs with ``tests/numerical/
neuron_benchmark.py``.

Inputs come from ``--reference``, written beforehand by ``tests/manual/ltx2_dit_reference.py``
(seeded video and audio latents, real connector outputs of a prompt, bf16 timesteps, diffusers
fp32 and bf16 outputs of the first N blocks). When the reference has the same ``--num-layers``
its outputs gate parity, for the video and the audio stream separately; otherwise its inputs are
used and parity is reported as not gated.

Launch (one device job at a time), e.g. TP4 eager on 4 layers:

    HF_HUB_OFFLINE=1 DIFFLET_DISABLE_PREWARM=1 DIFFLET_COMPILE_CACHE=<cache root> \\
    PYTHONPATH=<worktree> <venv>/bin/torchrun --standalone --nproc-per-node 4 \\
        <worktree>/tests/manual/check_neuron_ltx2_preflight.py --model-dir <snapshot> \\
        --reference W/ref-l4.pt --num-layers 4 --exec-mode eager \\
        --save W/l4-eager/out.pt --out-json W/l4-eager/result.json

and compile mode against the eager outputs: ``--exec-mode compile --compare eager=W/l4-eager/
out.pt --expect cold``.

Checks (rank 0 prints one PASS/FAIL line each): checkpoint (no missing / unexpected tensor),
heads_per_rank, fallbacks (none in the warm-up, the parity forwards or the benchmark),
unwarmed_shapes, forward_shapes, finite, parity_fp32_<stream>_t<t> (``neuron_parity.check``
against the fp32 reference with the bf16 control), host_rope (each of the four RoPE tables
computed once in the process: the warm-up's coordinates are the forwards'), no_compile_in_
forwards, instrumentation, hbm (``max_memory_allocated`` at most ``--hbm-limit-gib``), in compile
mode compiled_blocks / one_graph / warmup_neff / block_neff, in eager mode no compiled block,
ranks_identical (sha256 of every rank's outputs equal to rank 0's, over a gloo side group), and
``--compare`` / ``--equal-to`` / ``--expect`` as in the Qwen preflight.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ["DIFFLET_BACKEND"] = "neuron"

import torch

from difflet.backends.neuron import runtime
from difflet.backends.neuron.compile import NEFF_CACHE_ENV, configure_compile_cache
from difflet.backends.neuron.ltx_2.config import NeuronLtx2Config
from difflet.backends.neuron.ltx_2.transformer import ROPE_ATTRS, NeuronLtx2TransformerApplication
from difflet.backends.neuron.runtime import track_fallbacks
from difflet.backends.registry import get_backend
from difflet.models.ltx_2.contract import LTX2DiTInputBundle
from difflet.models.ltx_2.neuron_application import _OnDeviceLtx2Transformer
from difflet.pipeline.parallel_config import DiffletParallelConfig
from tests.manual import check_neuron_wan_preflight as wanpf
from tests.numerical import neuron_benchmark as bench
from tests.numerical import neuron_parity as parity

VS_OTHER_MIN_COSINE = wanpf.VS_EAGER_MIN_COSINE
VS_OTHER_MAX_RATIO_VS_CONTROL = wanpf.VS_EAGER_MAX_RATIO_VS_CONTROL
STREAMS = ("video", "audio")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--num-layers", type=int, default=48)
    p.add_argument("--exec-mode", choices=("eager", "compile"), required=True)
    p.add_argument("--save", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--compare", action="append", default=[], metavar="NAME=PATH")
    p.add_argument("--equal-to", type=Path, default=None)
    p.add_argument("--expect", choices=("cold", "warm"), default=None)
    p.add_argument("--bench-warmup", type=int, default=2)
    p.add_argument("--bench-iters", type=int, default=5)
    p.add_argument("--bench-rounds", type=int, default=2)
    p.add_argument("--hbm-limit-gib", type=float, default=24.0)
    args = p.parse_args()
    compares = {}
    for item in args.compare:
        name, sep, path = item.partition("=")
        if not sep or not name:
            p.error(f"--compare wants NAME=PATH, got {item!r}")
        compares[name] = Path(path)
    args.compare = compares
    return args


def _load_saved(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, weights_only=False)["outputs"]


def _stats(t: torch.Tensor) -> dict[str, Any]:
    t = t.double()
    return {"finite": bool(torch.isfinite(t).all()), "absmean": float(t.abs().mean()),
            "std": float(t.std()), "min": float(t.min()), "max": float(t.max())}


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    ref = torch.load(args.reference, weights_only=False)
    gated = int(ref["num_layers"]) == args.num_layers
    runs = ref["runs"]
    others = {name: _load_saved(path) for name, path in args.compare.items()}
    equal_to = _load_saved(args.equal_to) if args.equal_to is not None else None

    configure_compile_cache()
    import torch_neuronx
    import torch_neuronx.metrics as metrics

    metrics.set_enabled(True)
    metrics_setup = {"enabled_before_prepare_runtime": bool(metrics.is_enabled()),
                     "runtime_initialized_when_enabled":
                         bool(torch_neuronx.is_neuron_runtime_initialized())}
    neffs_before = wanpf.neff_files()
    snaps = {"pre_prepare": wanpf.snapshot(t0, device=False, runtime_started=False)}
    world = int(os.environ.get("WORLD_SIZE", "1"))
    tp = world
    parallel = DiffletParallelConfig(tp_degree=tp)
    get_backend("neuron").prepare_runtime(parallel)
    prepare_s = time.perf_counter() - t0

    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    logical_core = runtime.parse_core_list(os.environ.get(runtime.VISIBLE_CORES_ENV))
    logical_core = logical_core[0] if len(logical_core) == 1 else None
    snaps["start"] = wanpf.snapshot(t0)

    transformer_dir = args.model_dir / "transformer"
    config = NeuronLtx2Config.from_pretrained(
        transformer_dir, num_layers=args.num_layers, height=int(ref["height"]),
        width=int(ref["width"]), num_frames=int(ref["num_frames"]), tp_degree=tp,
        torch_dtype=torch.bfloat16,
    )
    app = NeuronLtx2TransformerApplication(
        model_path=transformer_dir, config=config, parallel=parallel, dtype=torch.bfloat16,
        exec_mode=args.exec_mode, device="neuron",
    )
    app.load(skip_warmup=True)
    snaps["after_load"] = wanpf.snapshot(t0)
    wanpf._progress(rank, "loaded", app, snaps["after_load"])
    with track_fallbacks() as warmup_fallbacks:
        app.warmup()
    snaps["after_warmup"] = wanpf.snapshot(t0)
    wanpf._progress(rank, "warmed up", app, snaps["after_warmup"])
    breakdown = {"after_warmup": wanpf.compile_breakdown()} if args.exec_mode == "compile" else {}

    forward_shapes: list[list[tuple[int, ...]]] = []

    def module(*device_args: torch.Tensor):
        forward_shapes.append([tuple(a.shape) for a in device_args])
        return app(*device_args)

    adapter = _OnDeviceLtx2Transformer(module, config, torch.bfloat16, app.device)
    adapter_s: list[float] = []
    adapter.forward_hook = adapter_s.append
    outputs: dict[str, torch.Tensor] = {}
    with track_fallbacks() as fallbacks:
        for run in runs:
            t = run["dit_timestep"]
            bundle = LTX2DiTInputBundle(
                hidden_states=ref["hidden_states"], audio_hidden_states=ref["audio_hidden_states"],
                encoder_hidden_states=ref["encoder_hidden_states"],
                audio_encoder_hidden_states=ref["audio_encoder_hidden_states"],
                timestep=t, sigma=t,
                encoder_attention_mask=torch.ones(1, config.text_seq_len, dtype=torch.bool),
                audio_encoder_attention_mask=torch.ones(1, config.text_seq_len, dtype=torch.bool),
                video_coords=ref["video_coords"], audio_coords=ref["audio_coords"],
            )
            video, audio = adapter(bundle)
            outputs[f"video/{run['timestep']:g}"] = video.float()
            outputs[f"audio/{run['timestep']:g}"] = audio.float()
    snaps["after_forwards"] = wanpf.snapshot(t0)
    wanpf._progress(rank, "parity forwards done", app, snaps["after_forwards"])

    bench_result = None
    with track_fallbacks() as bench_fallbacks:
        if args.bench_iters > 0:
            device_args = tuple(
                t.to(torch.bfloat16).contiguous().to(app.device)
                for t in (ref["hidden_states"], ref["audio_hidden_states"],
                          ref["encoder_hidden_states"], ref["audio_encoder_hidden_states"],
                          runs[0]["dit_timestep"]))
            coords = (ref["video_coords"], ref["audio_coords"])
            [bench_result] = bench.run({"dit_forward": lambda: app(*device_args, *coords)},
                                       warmup=args.bench_warmup, iters=args.bench_iters,
                                       rounds=args.bench_rounds,
                                       memory_stats=bench.neuron_memory_stats)
            print(f"[rank {rank}] {bench_result}", flush=True)
    snaps["end"] = wanpf.snapshot(t0)
    if args.exec_mode == "compile":
        breakdown["end"] = wanpf.compile_breakdown()
    evidence = parity.device_evidence(app, fallbacks)
    neffs_after = wanpf.neff_files()
    output_hashes = {key: wanpf.sha256(out) for key, out in outputs.items()}
    all_hashes = wanpf.gather_hashes(output_hashes, world)

    checks: list[tuple[str, bool, str]] = []
    report = app.load_report or {}
    checks.append(("checkpoint", not report.get("missing") and not report.get("unexpected"),
                   f"missing={len(report.get('missing', []))} "
                   f"unexpected={len(report.get('unexpected', []))} "
                   f"skipped={len(report.get('skipped', []))}"))
    block0 = app.module.transformer.transformer_blocks[0]
    heads = {name: int(getattr(block0, name).heads)
             for name in ("attn1", "audio_attn1", "attn2", "audio_attn2", "audio_to_video_attn",
                          "video_to_audio_attn")}
    checks.append(("heads_per_rank",
                   heads["attn1"] == config.num_attention_heads // tp
                   and heads["audio_attn1"] == config.audio_num_attention_heads // tp,
                   f"{heads} (32 / tp {tp})"))
    checks.append(("fallbacks",
                   evidence["fallbacks"] == [] and warmup_fallbacks == []
                   and bench_fallbacks == [],
                   f"forwards={evidence['fallbacks']} warm-up={warmup_fallbacks} "
                   f"benchmark={bench_fallbacks}"))
    checks.append(("unwarmed_shapes", evidence["unwarmed_shapes"] == [],
                   f"{evidence['unwarmed_shapes']} warmup={app.warmup_shapes}"))
    checks.append(("forward_shapes",
                   len(forward_shapes) == len(runs)
                   and all(shapes == app.warmup_shapes for shapes in forward_shapes),
                   f"{len(forward_shapes)} forwards at {sorted(set(map(str, forward_shapes)))} "
                   f"warmup={app.warmup_shapes}"))
    differ = sorted(r for r, h in enumerate(all_hashes) if h != all_hashes[0])
    checks.append(("ranks_identical", len(all_hashes) == world and not differ,
                   f"{len(all_hashes)} ranks, sha256 of every output equal to rank 0's"
                   + (f"; differ on ranks {differ}" if differ else "")))
    ropes = {attr: int(getattr(app.module.transformer, attr).computed) for attr in ROPE_ATTRS}
    checks.append(("host_rope", all(n == 1 for n in ropes.values()),
                   f"tables computed per RoPE module: {ropes}"))
    out_stats = {key: _stats(out) for key, out in outputs.items()}
    checks.append(("finite", all(s["finite"] for s in out_stats.values()),
                   " ".join(f"{key}: absmean={s['absmean']:.4f}" for key, s in out_stats.items())))

    parity_results: dict[str, Any] = {}
    for run in runs:
        for stream in STREAMS:
            key = f"{stream}/{run['timestep']:g}"
            out, fp32, bf16 = outputs[key], run[stream], run[f"{stream}_bf16"]
            entry: dict[str, Any] = {"gated": gated}
            if not gated:
                entry["note"] = (f"reference has {ref['num_layers']} layers, this run "
                                 f"{args.num_layers}: inputs only, parity not gated")
                parity_results[key] = entry
                continue
            m = parity.metrics(out, fp32, control=bf16)
            control = None
            notes: list[str] = []
            if bf16 is not None:
                m["vs_bf16"] = parity.metrics(out, bf16)
                control = parity.metrics(bf16, fp32)
            ok, reasons = parity.check(m, control=control, notes=notes)
            entry.update({"vs_fp32": m, "control": control, "ok": ok, "reasons": reasons,
                          "notes": notes})
            detail = (f"cos={m['cosine']:.6f} rel_l1={m['rel_l1']:.3e} rel_l2={m['rel_l2']:.3e} "
                      f"rel_max={m['rel_max']:.3e}")
            if control is not None:
                entry["ratio_vs_control"] = m["rel_l1"] / control["rel_l1"]
                detail += (f" | {entry['ratio_vs_control']:.3f}x control rel_l1 "
                           f"{control['rel_l1']:.3e} (cos {control['cosine']:.6f}) "
                           f"err_cos={m['err_cos']:.3f} {parity.same_dtype_label(m, control)}")
            parity_results[key] = entry
            checks.append((f"parity_fp32_{stream}_t{run['timestep']:g}", ok,
                           detail + (f" {reasons}" if reasons else "")
                           + (f" {notes}" if notes else "")))

    for name, saved in others.items():
        for key, out in outputs.items():
            other = saved[key]
            m = parity.metrics(out, other)
            control = parity_results.get(key, {}).get("control")
            ok = m["cosine"] >= VS_OTHER_MIN_COSINE
            detail = f"cos={m['cosine']:.6f} (min {VS_OTHER_MIN_COSINE}) rel_l1={m['rel_l1']:.3e}"
            if control is not None:
                limit = VS_OTHER_MAX_RATIO_VS_CONTROL * control["rel_l1"]
                ok = ok and m["rel_l1"] <= limit
                detail += (f" ({m['rel_l1'] / control['rel_l1']:.3f}x control "
                           f"{control['rel_l1']:.3e}, max {VS_OTHER_MAX_RATIO_VS_CONTROL}x)")
            detail += f" bit_identical={torch.equal(out, other)}"
            parity_results.setdefault(key, {})[f"vs_{name}"] = m
            checks.append((f"vs_{name}_{key}", ok, detail))
    if equal_to is not None:
        for key, out in outputs.items():
            checks.append((f"equal_to_{key}", torch.equal(out, equal_to[key]),
                           f"max_abs={float((out - equal_to[key]).abs().max()):.3e}"))

    delta = wanpf.delta
    load_delta = delta(snaps["after_load"], snaps["start"])
    warmup_delta = delta(snaps["after_warmup"], snaps["after_load"])
    forwards_delta = delta(snaps["end"], snaps["after_warmup"])
    run_delta = delta(snaps["end"], snaps["pre_prepare"])
    added_neffs = sorted(set(neffs_after) - set(neffs_before))
    added = {
        "prepare_runtime": wanpf.added_entries(snaps["start"], snaps["pre_prepare"]),
        "load": wanpf.added_entries(snaps["after_load"], snaps["start"]),
        "warmup": wanpf.added_entries(snaps["after_warmup"], snaps["after_load"]),
        "forwards": wanpf.added_entries(snaps["end"], snaps["after_warmup"]),
    }
    graphs_warm, graphs_end = snaps["after_warmup"]["dynamo"], snaps["end"]["dynamo"]
    forward_neffs = forwards_delta["TotalCompilations"] + forwards_delta["PersistentHits"]
    no_new_graph = (args.exec_mode != "compile"
                    or graphs_end["unique_graphs"] == graphs_warm["unique_graphs"])
    checks.append(("no_compile_in_forwards", forward_neffs == 0 and no_new_graph,
                   f"forward+benchmark counters {forwards_delta}, unique_graphs "
                   f"{graphs_warm['unique_graphs']} -> {graphs_end['unique_graphs']}"))
    peak_gib = snaps["end"]["device_memory"]["max_allocated_gib"]
    checks.append(("instrumentation",
                   *wanpf.instrumentation_check(metrics_setup, forwards_delta, peak_gib)))
    checks.append(("hbm", peak_gib <= args.hbm_limit_gib,
                   f"max_memory_allocated after load="
                   f"{snaps['after_load']['device_memory']['max_allocated_gib']:.2f} warm-up="
                   f"{snaps['after_warmup']['device_memory']['max_allocated_gib']:.2f} end="
                   f"{peak_gib:.2f} GiB (limit {args.hbm_limit_gib}); weights allocated "
                   f"{snaps['after_load']['device_memory']['allocated_gib']:.2f} GiB"))
    blocks: list[dict[str, Any]] = []
    if args.exec_mode == "compile":
        checks.append(("compiled_blocks", len(app.compiled_blocks) == config.num_layers,
                       f"{len(app.compiled_blocks)} of {config.num_layers}"))
        checks.append(("one_graph",
                       graphs_warm["unique_graphs"] == 1 and graphs_end["unique_graphs"] == 1
                       and graphs_end["graph_breaks"] == 0,
                       f"unique_graphs after warm-up={graphs_warm['unique_graphs']} "
                       f"end={graphs_end['unique_graphs']} graph_breaks="
                       f"{graphs_end['graph_breaks']} for {config.num_layers} blocks"))
        checks.append(("warmup_neff",
                       warmup_delta["TotalCompilations"] + warmup_delta["PersistentHits"] >= 1,
                       f"warm-up counters {warmup_delta}"))
        graphs = breakdown["end"].get("neuron_graphs", [])
        blocks = wanpf.block_neffs(graphs, added)
        checks.append(("block_neff",
                       len(graphs) == graphs_end["unique_graphs"] and len(blocks) == len(graphs)
                       and all(b["phase"] == "warmup" for b in blocks),
                       f"{len(graphs)} neuron graph(s), entries "
                       + str([(b["phase"], b["key"][:12], b["neff_size_bytes"],
                               round(b["compilation_time_ms"] / 1e3, 1)) for b in blocks])
                       + f" (phase, key, bytes, compile s){breakdown['end'].get('error', '')}"))
    else:
        checks.append(("eager_compiles_no_block",
                       not app.compiled_blocks and graphs_end["unique_graphs"] == 0,
                       f"compiled_blocks={len(app.compiled_blocks)} "
                       f"unique_graphs={graphs_end['unique_graphs']}"))
    if args.expect == "cold" and rank == 0:
        checks.append(("neff_cache_cold", len(added_neffs) >= 1,
                       f"new_neffs={len(added_neffs)} run={run_delta}"))
    if args.expect == "warm":
        checks.append(("warm_counters",
                       run_delta["TotalCompilations"] == 0 and run_delta["PersistentHits"] >= 1
                       and not added_neffs,
                       f"run counters {run_delta} new_neffs={len(added_neffs)}"))

    ok = all(passed for _, passed, _ in checks)
    if rank == 0:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"outputs": outputs, "num_layers": args.num_layers,
                    "exec_mode": args.exec_mode, "tp": tp,
                    "reference": str(args.reference)}, args.save)

    warmup_s = app.phase_seconds.get("warmup forward")
    result = {
        "rank": rank, "world_size": world, "logical_core": logical_core,
        "exec_mode": args.exec_mode, "expect": args.expect, "ok": ok,
        "args": {k: (str(v) if isinstance(v, Path) else
                     {n: str(p) for n, p in v.items()} if isinstance(v, dict) else v)
                 for k, v in vars(args).items()},
        "neff_cache_dir": os.environ.get(NEFF_CACHE_ENV),
        "metrics_setup": metrics_setup,
        "pg_timeout": wanpf.pg_timeout(),
        "torch_threads": torch.get_num_threads(),
        "config": {"num_layers": config.num_layers,
                   "checkpoint_num_layers": config.checkpoint_num_layers,
                   "tp": tp, "video_seq_len": config.video_seq_len,
                   "audio_seq_len": config.audio_seq_len, "text_seq_len": config.text_seq_len,
                   "local_heads": heads},
        "rank_output_hashes": all_hashes,
        "rope_computed": ropes,
        "reference": {"path": str(args.reference), "num_layers": int(ref["num_layers"]),
                      "timesteps": [r["timestep"] for r in runs], "parity_gated": gated},
        "checks": [{"name": n, "ok": p, "detail": d} for n, p, d in checks],
        "parity": parity_results,
        "output_stats": out_stats,
        "device_evidence": evidence,
        "warmup_fallbacks": list(warmup_fallbacks),
        "benchmark_fallbacks": list(bench_fallbacks),
        "phase_seconds": dict(app.phase_seconds),
        "timers_s": {"prepare_runtime": prepare_s, "adapter_forwards": adapter_s,
                     "total": time.perf_counter() - t0},
        "benchmark": None if bench_result is None else asdict(bench_result),
        "max_memory_allocated_gib": peak_gib,
        "max_memory_allocated_gib_by_phase": {
            name: snaps[name]["device_memory"]["max_allocated_gib"]
            for name in ("start", "after_load", "after_warmup", "after_forwards", "end")},
        "weights_allocated_gib": snaps["after_load"]["device_memory"]["allocated_gib"],
        "weight_bytes_estimate": app.weight_bytes,
        "host_vmhwm_gib_by_phase": {name: snap["host_memory"]["VmHWM_gib"]
                                    for name, snap in snaps.items()},
        "counters": {"load": load_delta, "warmup": warmup_delta,
                     "forwards_and_benchmark": forwards_delta, "run": run_delta},
        "cache_entries_added": added,
        "compile_breakdown": breakdown,
        "block_neffs": blocks,
        "forward_shapes": forward_shapes,
        "new_neffs": len(added_neffs),
        "sysfs_device_memory": wanpf.sysfs_device_memory(logical_core),
        "outputs": {key: {"sha256": wanpf.sha256(out), "shape": list(out.shape)}
                    for key, out in outputs.items()},
        "compiled_blocks": len(app.compiled_blocks),
        "load_report": {k: len(v) for k, v in report.items()},
    }
    path = args.out_json if rank == 0 else args.out_json.with_name(
        f"{args.out_json.stem}.rank{rank}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=str))

    if rank == 0:
        for name, passed, detail in checks:
            print(f"{'PASS' if passed else 'FAIL'} {name:28s} {detail}", flush=True)
    failed = [name for name, passed, _ in checks if not passed]
    bench_line = ""
    if bench_result is not None:
        bench_line = (f"bench median={bench_result.median_ms:.1f}ms p10={bench_result.p10_ms:.1f} "
                      f"p90={bench_result.p90_ms:.1f} n={len(bench_result.samples_ms)} ")
    adapter_line = (f"adapter={[round(s * 1e3, 1) for s in adapter_s]}ms "
                    f"(median {statistics.median(adapter_s) * 1e3:.1f}) " if adapter_s else "")
    print(f"[rank {rank}] {'PASS' if ok else 'FAIL'} {args.exec_mode}"
          f"{'/' + args.expect if args.expect else ''} layers={args.num_layers} tp={tp} "
          f"load_checkpoint={app.phase_seconds.get('load checkpoint', 0):.1f}s "
          f"warmup_forward={warmup_s:.1f}s {adapter_line}{bench_line}"
          f"run_counters={run_delta} new_neffs={len(added_neffs)} "
          f"peak_hbm={peak_gib:.2f}GiB VmHWM={snaps['end']['host_memory']['VmHWM_gib']:.2f}GiB"
          + (f" failed={failed}" if failed else ""), flush=True)
    if ok and dist.is_initialized():
        dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
