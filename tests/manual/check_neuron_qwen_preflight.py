"""Manual check (V1): the Qwen-Image DiT through the neuron application on torchrun ranks.

The device preflight of the Qwen-Image port to the PyTorch-native neuron backend, modelled on
``tests/manual/check_neuron_wan_preflight.py`` (whose measurement helpers it imports, so the two
gates count NEFFs, compilations and memory the same way): it builds
``NeuronQwenImageTransformerApplication`` under ``get_backend('neuron').prepare_runtime``,
loads the first ``--num-layers`` blocks of the real checkpoint (60 = the whole model), warms
up, runs one DiT forward per reference timestep through the composed application's
host<->device adapter (``_OnDeviceTransformer``, the orchestrator's own DiT call: inputs cast
to bf16 on the host, moved, forward, ``torch.neuron.synchronize``, copied back) under
``track_fallbacks()``, then times the device forward on device-resident inputs with
``tests/numerical/neuron_benchmark.py`` (synchronised both sides, interleaved rounds).

Inputs come from ``--reference``, written beforehand by ``tests/manual/qwen_image_dit_reference.py``
(seeded packed latents, real Qwen2.5-VL embeddings of a fixed prompt zero-padded to
``text_seq_len``, the DiT timesteps t/1000 in bf16, diffusers fp32 and bf16 outputs). When the
reference was computed with the same ``--num-layers`` its outputs gate parity; otherwise its
inputs are used and parity is reported as not gated (a 60-layer launch with a 4-layer
reference proves finiteness, timing and memory only).

Launch (one device job at a time; ``W`` a records directory), e.g. TP4 eager:

    HF_HUB_OFFLINE=1 OMP_NUM_THREADS=3 DIFFLET_NO_JEMALLOC=1 DIFFLET_DISABLE_PREWARM=1 \\
    DIFFLET_COMPILE_CACHE=<cache root> PYTHONPATH=<worktree> \\
    <venv>/bin/torchrun --standalone --nproc-per-node 4 \\
        <worktree>/tests/manual/check_neuron_qwen_preflight.py --model-dir <snapshot> \\
        --reference W/ref-l4.pt --num-layers 4 --exec-mode eager \\
        --save W/l4-eager/out.pt --out-json W/l4-eager/result.json

and compile mode against the eager outputs: ``--exec-mode compile --compare eager=W/l4-eager/out.pt
--expect cold`` (``--expect warm --equal-to W/l4-cold/out.pt`` on the second launch). Add
``DIFFLET_NEURON_PG_TIMEOUT_S=7200`` for a launch that cold-compiles. ``--sp`` builds the
Megatron-SP fork (``QwenImageSPTransformer2DModel``) over the TP group; gate it against the
non-SP launch of the same tp with ``--compare tp=W/l4-eager/out.pt``.

Checks (every rank; rank 0 prints one PASS/FAIL line each, every rank a summary line):

* checkpoint (no missing / unexpected tensor; blocks past ``--num-layers`` are ``skipped``),
  heads_per_rank (24 / tp, or / (tp * cp) under Ulysses' head shard: the module's own count);
* fallbacks: ``track_fallbacks()`` around the warm-up, the parity forwards and the benchmark
  is ``[]``; unwarmed_shapes ``[]``; forward_shapes: every parity forward's device inputs have
  the warm-up's shapes;
* finite: every output is finite;
* parity_fp32_t<t> (reference with the same ``--num-layers`` only): ``neuron_parity.check``
  against the fp32 reference with the bf16 control (cosine >= 0.995, rel_l1 <= 1.15x the
  control's, the same-dtype limit per D68); cosine, rel_l1, rel_l2 and rel_max are printed;
* no_compile_in_forwards: the parity forwards and the benchmark neither compiled nor loaded a
  NEFF and, in compile mode, added no Dynamo graph; instrumentation: the counters were live;
* hbm: ``max_memory_allocated`` at the end at most ``--hbm-limit-gib`` (default 24, one
  logical core's share of the trn2.3xlarge's 96 GB);
* compile mode: compiled_blocks (all of them), one_graph (one Dynamo graph, no break),
  warmup_neff, block_neff (the Dynamo graph's NEFF is a runtime cache entry the warm-up added);
  eager mode: no block compiled, no Dynamo graph;
* ranks_identical: the sha256 of every rank's outputs equals rank 0's (TP gathers the final
  projection and CP the image tokens, so each rank holds the full output), all-gathered over
  a gloo side group; the per-rank sequence/head arithmetic of the CP mode is recorded as
  ``shape_rules``; cp_scatter: the diffusers model (eager, inside the trace module) receives
  ``image_seq_len / cp`` image tokens per rank in every parity forward;
* ``--compare NAME=PATH`` (another launch's ``--save``, e.g. eager): vs_NAME_t<t>, cosine >=
  0.999 and rel_l1 <= 2x the bf16 control's rel_l1 from fp32 (the Wan preflight's D58 guard;
  without a control only the cosine);
* ``--equal-to PATH``: outputs ``torch.equal`` to that launch's (a warm launch of the same NEFFs);
* ``--expect cold``: rank 0 finds new ``.neff`` files; ``--expect warm``: no compilation and no
  new ``.neff`` on every rank, at least one persistent hit.

Rank 0 writes ``--out-json``, rank r > 0 ``<stem>.rank<r>.json`` next to it: the checks, the
parity metrics, phase_seconds, the forward timers and the benchmark (median / p10 / p90 /
every sample), the compilation counters per phase, the runtime cache entries each phase added,
the compile breakdown, the new ``.neff`` files, device memory per phase, the driver's sysfs
memory counters and host memory. Rank 0 also saves its outputs as fp32 to ``--save``.
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
from difflet.backends.neuron.qwen_image.config import NeuronQwenImageConfig
from difflet.backends.neuron.qwen_image.transformer import NeuronQwenImageTransformerApplication
from difflet.backends.neuron.runtime import track_fallbacks
from difflet.backends.registry import get_backend
from difflet.models.qwen_image.neuron_application import _OnDeviceTransformer
from difflet.pipeline.parallel_config import DiffletParallelConfig
from tests.manual import check_neuron_wan_preflight as wanpf
from tests.numerical import neuron_benchmark as bench
from tests.numerical import neuron_parity as parity

#: The Wan preflight's compiled-vs-eager sanity guard (decision D58), reused unchanged.
VS_OTHER_MIN_COSINE = wanpf.VS_EAGER_MIN_COSINE
VS_OTHER_MAX_RATIO_VS_CONTROL = wanpf.VS_EAGER_MAX_RATIO_VS_CONTROL


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-dir", type=Path, required=True,
                   help="diffusers snapshot root holding transformer/")
    p.add_argument("--reference", type=Path, required=True,
                   help="tests/manual/qwen_image_dit_reference.py output (inputs, references)")
    p.add_argument("--num-layers", type=int, default=60)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--text-seq-len", type=int, default=1024)
    p.add_argument("--exec-mode", choices=("eager", "compile"), required=True)
    p.add_argument("--tp-degree", type=int, default=None, help="default: WORLD_SIZE / cp")
    p.add_argument("--cp-degree", type=int, default=1)
    p.add_argument("--cp-mode", choices=("gather_kv", "ulysses", "ring"), default="gather_kv")
    p.add_argument("--sp", action="store_true",
                   help="sequence parallelism within the TP group (not with --cp-degree > 1)")
    p.add_argument("--save", type=Path, required=True, help="rank 0's outputs (fp32, .pt)")
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--compare", action="append", default=[], metavar="NAME=PATH",
                   help="gate against another launch's --save (e.g. eager=...)")
    p.add_argument("--equal-to", type=Path, default=None,
                   help="another launch's --save; outputs must be torch.equal")
    p.add_argument("--expect", choices=("cold", "warm"), default=None)
    p.add_argument("--bench-warmup", type=int, default=2)
    p.add_argument("--bench-iters", type=int, default=5,
                   help="timed device forwards per round (0: no benchmark)")
    p.add_argument("--bench-rounds", type=int, default=2)
    p.add_argument("--hbm-limit-gib", type=float, default=24.0)
    args = p.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world % args.cp_degree:
        p.error(f"WORLD_SIZE {world} is not divisible by --cp-degree {args.cp_degree}")
    if args.tp_degree is None:
        args.tp_degree = world // args.cp_degree
    if args.tp_degree * args.cp_degree != world:
        p.error(f"--tp-degree {args.tp_degree} x --cp-degree {args.cp_degree} != WORLD_SIZE "
                f"{world}")
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


def gather_hashes(hashes: dict[str, str], world: int) -> list[dict[str, str]]:
    """Every rank's output hashes, in rank order, over a gloo side group (host objects only:
    the neuron group carries device tensors)."""
    import torch.distributed as dist

    if world <= 1 or not dist.is_initialized():
        return [hashes]
    group = dist.new_group(backend="gloo")
    gathered: list[Any] = [None] * world
    dist.all_gather_object(gathered, hashes, group=group)
    dist.destroy_process_group(group)
    return gathered


def shape_rules(config: NeuronQwenImageConfig, tp: int, cp: int, cp_mode: str) -> dict[str, Any]:
    """The per-rank sequence and head arithmetic of the CP mode, as recorded evidence.

    The image tokens are scattered over cp (text replicated). Ring attends the joint
    ``[txt ‖ img/cp]`` query to each K/V block through the flash kernel's logsumexp path, which
    needs the query length padded to a multiple of 128 (``_attention_with_lse`` pads it);
    Ulysses swaps to ``heads / (tp * cp)`` heads at the full joint length.
    """
    img, txt = int(config.image_seq_len), int(config.text_seq_len)
    heads = int(config.num_attention_heads)
    img_rank = img // cp
    rules: dict[str, Any] = {
        "image_seq_len": img, "text_seq_len": txt, "image_tokens_per_rank": img_rank,
        "heads_per_tp_rank": heads // tp, "image_divisible_by_cp": img % cp == 0,
    }
    if cp > 1 and cp_mode == "ring":
        q_len = txt + img_rank
        rules.update({"ring_query_len": q_len, "ring_query_len_mod_128": q_len % 128,
                      "ring_query_padded_to": -(-q_len // 128) * 128,
                      "ring_key_blocks": [txt] + [img_rank] * cp})
    if cp > 1 and cp_mode == "ulysses":
        rules.update({"ulysses_heads_per_rank": (heads // tp) // cp,
                      "ulysses_heads_divisible": (heads // tp) % cp == 0,
                      "ulysses_attention_len": img + txt})
    if cp > 1 and cp_mode == "gather_kv":
        rules.update({"gather_kv_query_len": txt + img_rank, "gather_kv_key_len": txt + img})
    return rules


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    # Host only, before any device work: a wrong reference must not cost a 60-layer load.
    ref = torch.load(args.reference, weights_only=False)
    for key in ("height", "width", "text_seq_len"):
        if int(ref[key]) != int(getattr(args, key)):
            raise ValueError(f"{args.reference} has {key}={ref[key]}, this run "
                             f"--{key.replace('_', '-')} {getattr(args, key)}")
    gated = int(ref["num_layers"]) == args.num_layers
    runs = ref["runs"]
    others = {name: _load_saved(path) for name, path in args.compare.items()}
    equal_to = _load_saved(args.equal_to) if args.equal_to is not None else None

    configure_compile_cache()  # what prepare_runtime does; resolved now to count the NEFFs
    import torch_neuronx
    import torch_neuronx.metrics as metrics

    metrics.set_enabled(True)
    metrics_setup = {"enabled_before_prepare_runtime": bool(metrics.is_enabled()),
                     "runtime_initialized_when_enabled":
                         bool(torch_neuronx.is_neuron_runtime_initialized())}
    neffs_before = wanpf.neff_files()
    snaps = {"pre_prepare": wanpf.snapshot(t0, device=False, runtime_started=False)}
    world = int(os.environ.get("WORLD_SIZE", "1"))
    tp, cp = args.tp_degree, args.cp_degree
    parallel = DiffletParallelConfig(tp_degree=tp, cp_degree=cp, cp_mode=args.cp_mode,
                                     sp_enabled=args.sp)
    get_backend("neuron").prepare_runtime(parallel)
    prepare_s = time.perf_counter() - t0

    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    logical_core = runtime.parse_core_list(os.environ.get(runtime.VISIBLE_CORES_ENV))
    logical_core = logical_core[0] if len(logical_core) == 1 else None
    snaps["start"] = wanpf.snapshot(t0)

    transformer_dir = args.model_dir / "transformer"
    config = NeuronQwenImageConfig.from_pretrained(
        transformer_dir, num_layers=args.num_layers, height=args.height, width=args.width,
        text_seq_len=args.text_seq_len, tp_degree=tp, context_parallel_enabled=cp > 1,
        cp_mode=args.cp_mode, sp_enabled=args.sp, torch_dtype=torch.bfloat16,
    )
    app = NeuronQwenImageTransformerApplication(
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

    def module(*device_args: torch.Tensor) -> torch.Tensor:
        forward_shapes.append([tuple(a.shape) for a in device_args])
        return app(*device_args)

    # The diffusers model runs eager inside the trace module, after the CP scatter: the
    # sequence it receives is this rank's image shard (positive evidence the scatter happened).
    seen_image_tokens: list[int] = []

    def _record(_mod, _args, kwargs):
        seen_image_tokens.append(int(kwargs["hidden_states"].shape[1]))

    scatter_hook = app.module.transformer.register_forward_pre_hook(_record, with_kwargs=True)
    adapter = _OnDeviceTransformer(module, config, torch.bfloat16, app.device)
    adapter_s: list[float] = []
    adapter.forward_hook = adapter_s.append
    outputs: dict[str, torch.Tensor] = {}
    with track_fallbacks() as fallbacks:
        for run in runs:
            out = adapter(ref["hidden_states"], run["dit_timestep"],
                          ref["encoder_hidden_states"])
            outputs[f"{run['timestep']:g}"] = out.float()
    scatter_hook.remove()
    snaps["after_forwards"] = wanpf.snapshot(t0)
    wanpf._progress(rank, "parity forwards done", app, snaps["after_forwards"])

    bench_result = None
    with track_fallbacks() as bench_fallbacks:
        if args.bench_iters > 0:
            device_args = tuple(
                t.to(torch.bfloat16).contiguous().to(app.device)
                for t in (ref["hidden_states"], runs[0]["dit_timestep"],
                          ref["encoder_hidden_states"]))
            [bench_result] = bench.run({"dit_forward": lambda: app(*device_args)},
                                       warmup=args.bench_warmup, iters=args.bench_iters,
                                       rounds=args.bench_rounds,
                                       memory_stats=bench.neuron_memory_stats)
            print(f"[rank {rank}] {bench_result}", flush=True)
    snaps["end"] = wanpf.snapshot(t0)
    if args.exec_mode == "compile":
        breakdown["end"] = wanpf.compile_breakdown()
    evidence = parity.device_evidence(app, fallbacks)
    neffs_after = wanpf.neff_files()
    output_hashes = {ts: wanpf.sha256(out) for ts, out in outputs.items()}
    all_hashes = gather_hashes(output_hashes, world)

    # ------------------------------------------------------------------ checks
    checks: list[tuple[str, bool, str]] = []
    report = app.load_report or {}
    checks.append(("checkpoint", not report.get("missing") and not report.get("unexpected"),
                   f"missing={len(report.get('missing', []))} "
                   f"unexpected={len(report.get('unexpected', []))} "
                   f"skipped={len(report.get('skipped', []))}"))
    attn = app.module.transformer.transformer_blocks[0].attn
    heads_per_rank = config.num_attention_heads // tp
    checks.append(("heads_per_rank", int(attn.heads) == heads_per_rank,
                   f"attn.heads={attn.heads} (={config.num_attention_heads}/tp {tp})"))
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
    # TP gathers the final projection and CP gathers the image tokens back, so every rank
    # holds the full output: each must be bit-identical to rank 0's.
    differ = sorted(r for r, h in enumerate(all_hashes) if h != all_hashes[0])
    checks.append(("ranks_identical", len(all_hashes) == world and not differ,
                   f"{len(all_hashes)} ranks, sha256 of every output equal to rank 0's"
                   + (f"; differ on ranks {differ}" if differ else "")
                   + f" (t{next(iter(output_hashes))} {next(iter(output_hashes.values()))[:12]})"))
    expected_tokens = config.image_seq_len // cp
    checks.append(("cp_scatter",
                   len(seen_image_tokens) == len(runs)
                   and all(n == expected_tokens for n in seen_image_tokens),
                   f"image tokens per rank into the blocks {seen_image_tokens} (expected "
                   f"{config.image_seq_len}/cp {cp} = {expected_tokens}; output "
                   f"{[list(o.shape) for o in outputs.values()][0]})"))
    out_stats = {ts: _stats(out) for ts, out in outputs.items()}
    checks.append(("finite", all(s["finite"] for s in out_stats.values()),
                   " ".join(f"t{ts}: absmean={s['absmean']:.4f} std={s['std']:.4f} "
                            f"[{s['min']:.3f}, {s['max']:.3f}]"
                            for ts, s in out_stats.items())))

    parity_results: dict[str, Any] = {}
    for run in runs:
        ts = f"{run['timestep']:g}"
        out, fp32, bf16 = outputs[ts], run["reference"], run["reference_bf16"]
        entry: dict[str, Any] = {"gated": gated}
        if not gated:
            entry["note"] = (f"reference has {ref['num_layers']} layers, this run "
                             f"{args.num_layers}: inputs only, parity not gated")
            parity_results[ts] = entry
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
            entry["ratio_l2_vs_control"] = m["rel_l2"] / control["rel_l2"]
            entry["same_dtype_gated"] = parity.same_dtype_gated(m)
            detail += (f" | {entry['ratio_vs_control']:.3f}x control rel_l1 "
                       f"{control['rel_l1']:.3e} (rel_l2 {control['rel_l2']:.3e}, cos "
                       f"{control['cosine']:.6f}) err_cos={m['err_cos']:.3f} "
                       f"{parity.same_dtype_label(m, control)}")
        parity_results[ts] = entry
        checks.append((f"parity_fp32_t{ts}", ok, detail + (f" {reasons}" if reasons else "")
                       + (f" {notes}" if notes else "")))

    for name, saved in others.items():
        for ts, out in outputs.items():
            other = saved[ts]
            m = parity.metrics(out, other)
            control = parity_results.get(ts, {}).get("control")
            ok = m["cosine"] >= VS_OTHER_MIN_COSINE
            detail = f"cos={m['cosine']:.6f} (min {VS_OTHER_MIN_COSINE}) rel_l1={m['rel_l1']:.3e}"
            if control is not None:
                limit = VS_OTHER_MAX_RATIO_VS_CONTROL * control["rel_l1"]
                ok = ok and m["rel_l1"] <= limit
                detail += (f" ({m['rel_l1'] / control['rel_l1']:.3f}x control "
                           f"{control['rel_l1']:.3e}, max {VS_OTHER_MAX_RATIO_VS_CONTROL}x)")
            detail += (f" rel_l2={m['rel_l2']:.3e} rel_max={m['rel_max']:.3e} "
                       f"bit_identical={torch.equal(out, other)}")
            parity_results.setdefault(ts, {})[f"vs_{name}"] = m
            checks.append((f"vs_{name}_t{ts}", ok, detail))
    if equal_to is not None:
        for ts, out in outputs.items():
            equal = torch.equal(out, equal_to[ts])
            checks.append((f"equal_to_t{ts}", equal,
                           f"max_abs={float((out - equal_to[ts]).abs().max()):.3e}"))

    delta = wanpf.delta
    prepare_delta = delta(snaps["start"], snaps["pre_prepare"])
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
                    "exec_mode": args.exec_mode, "tp": tp, "cp": cp, "cp_mode": args.cp_mode,
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
        "env": {k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "HF_HUB_OFFLINE", "DIFFLET_NO_JEMALLOC", runtime.PG_TIMEOUT_ENV,
            runtime.VISIBLE_CORES_ENV, "DIFFLET_DISABLE_PREWARM", "DIFFLET_COMPILE_CACHE")},
        "torch_threads": torch.get_num_threads(),
        "config": {"num_layers": config.num_layers,
                   "checkpoint_num_layers": config.checkpoint_num_layers,
                   "heads": config.num_attention_heads, "head_dim": config.attention_head_dim,
                   "heads_per_rank": heads_per_rank, "tp": tp, "cp": cp,
                   "cp_mode": args.cp_mode, "image_seq_len": config.image_seq_len,
                   "text_seq_len": config.text_seq_len},
        "shape_rules": shape_rules(config, tp, cp, args.cp_mode),
        "rank_output_hashes": all_hashes,
        "image_tokens_into_blocks": seen_image_tokens,
        "reference": {"path": str(args.reference), "num_layers": int(ref["num_layers"]),
                      "prompt": ref.get("prompt"), "valid_rows": ref.get("valid_rows"),
                      "timesteps": [r["timestep"] for r in runs],
                      "dit_timesteps": [float(r["dit_timestep"][0]) for r in runs],
                      "parity_gated": gated},
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
        "host_vmhwm_gib_by_phase": {name: snap["host_memory"]["VmHWM_gib"]
                                    for name, snap in snaps.items()},
        "counters": {"prepare_runtime": prepare_delta, "load": load_delta,
                     "warmup": warmup_delta, "forwards_and_benchmark": forwards_delta,
                     "run": run_delta},
        "cache_entries_added": added,
        "compile_breakdown": breakdown,
        "block_neffs": blocks,
        "forward_shapes": forward_shapes,
        "neffs_before_prepare_runtime": len(neffs_before),
        "new_neffs": len(added_neffs),
        "new_neff_files": {path: neffs_after[path] for path in added_neffs},
        "snapshots": {
            name: {**{k: v for k, v in snap.items() if k != "cache_entries"},
                   "cache_entry_count": None if snap["cache_entries"] is None
                   else len(snap["cache_entries"])}
            for name, snap in snaps.items()},
        "sysfs_device_memory": wanpf.sysfs_device_memory(logical_core),
        "outputs": {ts: {"sha256": wanpf.sha256(out), "shape": list(out.shape)}
                    for ts, out in outputs.items()},
        "compiled_blocks": len(app.compiled_blocks),
        "load_report": {k: len(v) for k, v in report.items()},
    }
    path = args.out_json if rank == 0 else args.out_json.with_name(
        f"{args.out_json.stem}.rank{rank}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))

    if rank == 0:
        for name, passed, detail in checks:
            print(f"{'PASS' if passed else 'FAIL'} {name:24s} {detail}", flush=True)
    failed = [name for name, passed, _ in checks if not passed]
    bench_line = ""
    if bench_result is not None:
        bench_line = (f"bench median={bench_result.median_ms:.1f}ms p10={bench_result.p10_ms:.1f} "
                      f"p90={bench_result.p90_ms:.1f} n={len(bench_result.samples_ms)} ")
    adapter_line = (f"adapter={[round(s * 1e3, 1) for s in adapter_s]}ms "
                    f"(median {statistics.median(adapter_s) * 1e3:.1f}) " if adapter_s else "")
    print(f"[rank {rank}] {'PASS' if ok else 'FAIL'} {args.exec_mode}"
          f"{'/' + args.expect if args.expect else ''} layers={args.num_layers} tp={tp} cp={cp} "
          f"{(args.cp_mode + ' ') if cp > 1 else ''}"
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
