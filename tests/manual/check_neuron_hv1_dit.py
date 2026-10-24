"""HunyuanVideo 1.0's DiT application on the device: parity against an fp32 diffusers oracle at
any depth and any layout (TP, TP+SP, TPxCP gather_kv / ulysses / ring), and time per forward.

Phase 1, host CPU (no device, no torchrun): diffusers' ``HunyuanVideoTransformer3DModel`` built
with the first ``--layers`` dual-stream and ``--single-layers`` single-stream blocks of the real
checkpoint (only those blocks are read; default the whole model, about 52 GB of fp32 host RAM)
runs one forward per ``--timesteps`` entry in fp32 (the reference) and in bf16 (the control:
what bf16 alone costs) on seeded inputs: the latents of a ``--frames`` x ``--height`` x
``--width`` clip (5 x 320 x 512 = 2 x 20 x 32 = 1,280 image tokens by default), a 256-token
right-padded text with ``--valid-text`` valid tokens (default 20), the pooled projection and the
guidance (6.0 x 1000). Every float input is rounded to bf16 first, so the reference sees exactly
what the device sees. Inputs, outputs, depth and shape are saved together.

    PYTHONPATH=. python tests/manual/check_neuron_hv1_dit.py --make-oracle W/hv1_dit_l2s2.pt \\
        --layers 2 --single-layers 2

Phase 2, device: ``NeuronHunyuanVideoTransformerApplication`` at the oracle's depth and shape
under torchrun (``world = tp x cp``), after ``prepare_runtime`` (core binding, the NEFF cache
and the numerics workarounds), through the orchestrator's own host<->device adapter
(``_OnDeviceHunyuanVideoTransformer``: floats in bf16, the int64 mask as it is). Gate per
timestep (``tests/numerical/neuron_parity.py`` ``check``): rel-L1 vs fp32 at most 1.15x the bf16
control's, cosine at least 0.995; plus no CPU fallback, no unwarmed shape, every rank's output
identical, and in compile mode every block compiled. ``--iters`` then times the device forward
on device-resident inputs (``tests/numerical/neuron_benchmark.py``). Launched without torchrun,
the script builds the torchrun command for the layout and runs it (``--dry-run`` prints it):

    PYTHONPATH=. python tests/manual/check_neuron_hv1_dit.py --oracle W/hv1_dit_l2s2.pt \\
        --tp-degree 2 --exec-mode compile --out-json W/l2s2-tp2.json
    PYTHONPATH=. python tests/manual/check_neuron_hv1_dit.py --oracle W/hv1_dit_l2s2.pt \\
        --tp-degree 2 --cp-degree 2 --cp-mode ring

The oracle's depth, shape and valid-text count win over the flags. ``--oracle none``: timing
only at the flags' depth and shape (finite output, time per forward). Exit status 0 only when
every check passes.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ["DIFFLET_BACKEND"] = "neuron"

import torch

from tests.numerical import neuron_benchmark as bench
from tests.numerical import neuron_parity as parity

DEFAULT_MODEL_DIR = ("/home/ubuntu/.cache/huggingface/hub/models--hunyuanvideo-community--"
                     "HunyuanVideo/snapshots/e8c2aaa66fe3742a32c11a6766aecbf07c56e773")
H, W, FRAMES = 320, 512, 5
TEXT_SEQ_LEN = 256
VALID_TEXT = 20
TIMESTEPS = (900.0,)
GUIDANCE = 6000.0  # guidance_scale 6.0 x 1000, as the pipeline sends it
SEED = 0
REPO_ROOT = Path(__file__).resolve().parents[2]
#: Environment a hand launch needs that the CLI launcher would otherwise set (an exported value
#: wins): offline hub, no prewarm thread, the CLI's 3 h process-group timeout.
LAUNCH_ENV = {"HF_HUB_OFFLINE": "1", "DIFFLET_DISABLE_PREWARM": "1",
              "DIFFLET_NEURON_PG_TIMEOUT_S": "10800"}


# ---- shared: geometry, inputs, launch -------------------------------------------------------


def transformer_dir(model_dir) -> Path:
    return Path(model_dir).expanduser() / "transformer"


def config_for(model_dir, *, height: int, width: int, frames: int, layers: int | None,
               single_layers: int | None, **topology):
    """The neuron config at this shape and depth (validated: 4k+1 frames, 16-aligned pixels,
    the layout's splits). ``None`` depth keeps the checkpoint's."""
    from difflet.backends.neuron.hunyuan_video.config import NeuronHunyuanVideoConfig

    depth = {k: v for k, v in (("num_layers", layers), ("num_single_layers", single_layers))
             if v is not None}
    return NeuronHunyuanVideoConfig.from_pretrained(
        transformer_dir(model_dir), height=height, width=width, num_frames=frames,
        text_seq_len=TEXT_SEQ_LEN, batch_size=1, **depth, **topology)


def _bf16_values(t: torch.Tensor) -> torch.Tensor:
    """``t`` rounded to bf16, kept in fp32: the reference then sees exactly the device's input."""
    return t.to(torch.bfloat16).to(torch.float32)


def make_inputs(config, valid_text: int, seed: int = SEED) -> dict[str, torch.Tensor]:
    """Seeded DiT inputs at ``config``'s static shape, batch 1, without the timestep."""
    if not 0 <= valid_text <= config.text_seq_len:
        raise SystemExit(f"--valid-text {valid_text} is outside [0, {config.text_seq_len}]")
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(1, config.text_seq_len, dtype=torch.int64)
    mask[:, :valid_text] = 1
    latent = (1, config.in_channels, config.latent_frames, config.latent_height,
              config.latent_width)
    return {
        "hidden_states": _bf16_values(torch.randn(latent, generator=g)),
        "encoder_hidden_states": _bf16_values(
            torch.randn(1, config.text_seq_len, config.text_embed_dim, generator=g)),
        "encoder_attention_mask": mask,
        "pooled_projections": _bf16_values(
            torch.randn(1, config.pooled_projection_dim, generator=g)),
        "guidance": _bf16_values(torch.tensor([GUIDANCE])),
    }


def layout_name(tp: int, cp: int, cp_mode: str, sp: bool) -> str:
    return f"TP{tp}" + (" + SP" if sp else "") + (f" x CP{cp} ({cp_mode})" if cp > 1 else "")


def torchrun_command(nproc: int, script: str, argv: list[str]) -> list[str]:
    """``torchrun --standalone --nproc-per-node N script argv`` with this interpreter."""
    return [sys.executable, "-m", "torch.distributed.run", "--standalone",
            "--nproc-per-node", str(nproc), script, *argv]


def launch(nproc: int, argv: list[str], *, dry_run: bool, script: str = __file__) -> int:
    """Re-run ``script`` (this one by default) under torchrun on ``nproc`` ranks with ``argv``
    minus ``--dry-run``. Printed first, with the environment it adds, so a log holds the exact
    command."""
    argv = [a for a in argv if a != "--dry-run"]
    cmd = torchrun_command(nproc, str(Path(script).resolve()), argv)
    env = dict(os.environ)
    added = {k: v for k, v in LAUNCH_ENV.items() if not env.get(k)}
    env.update(added)
    path = [str(REPO_ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(path))
    shown = " ".join(f"{k}={shlex.quote(v)}" for k, v in
                     {**added, "PYTHONPATH": env["PYTHONPATH"]}.items())
    print(f"launch: {shown} {shlex.join(cmd)}", flush=True)
    if dry_run:
        return 0
    import subprocess

    return subprocess.call(cmd, env=env)


# ---- phase 1: the oracle --------------------------------------------------------------------


def make_oracle(args) -> int:
    from diffusers import HunyuanVideoTransformer3DModel
    from diffusers.utils import logging as diffusers_logging

    # The cut blocks' keys are reported as unused weights; they are checked below instead.
    diffusers_logging.set_verbosity_error()
    torch.set_num_threads(args.threads)
    config = config_for(args.model_dir, height=args.height, width=args.width, frames=args.frames,
                        layers=args.layers, single_layers=args.single_layers)
    layers, single = int(config.num_layers), int(config.num_single_layers)
    inputs = make_inputs(config, args.valid_text)
    t0 = time.perf_counter()
    model, info = HunyuanVideoTransformer3DModel.from_pretrained(
        str(transformer_dir(args.model_dir)), num_layers=layers, num_single_layers=single,
        torch_dtype=torch.float32, low_cpu_mem_usage=True, output_loading_info=True)
    model = model.eval().requires_grad_(False)
    load_s = time.perf_counter() - t0
    # The blocks past the cut are in the checkpoint on purpose.
    cut = tuple(f"transformer_blocks.{i}." for i in range(layers, config.checkpoint_num_layers))
    cut += tuple(f"single_transformer_blocks.{i}."
                 for i in range(single, config.checkpoint_num_single_layers))
    unexpected = [k for k in info.get("unexpected_keys", []) if not k.startswith(cut)]
    missing = list(info.get("missing_keys", []))
    if missing or unexpected:
        raise SystemExit(f"checkpoint mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")

    def forward(dtype: torch.dtype) -> list[torch.Tensor]:
        m = model.to(dtype)
        outs = []
        for t in args.timesteps:
            x = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in inputs.items()}
            with torch.no_grad():
                out = m(timestep=_bf16_values(torch.tensor([t])).to(dtype), **x,
                        return_dict=False)[0]
            outs.append(out.float())
        return outs

    seconds = {"load": load_s}
    t0 = time.perf_counter()
    fp32 = forward(torch.float32)
    seconds["fp32_forwards"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    bf16 = forward(torch.bfloat16)
    seconds["bf16_forwards"] = time.perf_counter() - t0
    runs = []
    for t, ref, ctl in zip(args.timesteps, fp32, bf16):
        control = parity.metrics(ctl, ref)
        runs.append({"timestep": float(t), "ref_fp32": ref, "control_bf16": ctl,
                     "control_metrics": control})
        print(f"t={t:g}: ref absmean={float(ref.abs().mean()):.4f} finite="
              f"{bool(torch.isfinite(ref).all())}; control rel_l1={control['rel_l1']:.3e} "
              f"cos={control['cosine']:.6f}", flush=True)
    out_path = Path(args.make_oracle).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "kind": "hv1_dit_oracle", "inputs": inputs, "runs": runs,
        "num_layers": layers, "num_single_layers": single,
        "checkpoint_num_layers": int(config.checkpoint_num_layers),
        "checkpoint_num_single_layers": int(config.checkpoint_num_single_layers),
        "height": int(config.height), "width": int(config.width),
        "num_frames": int(config.num_frames), "text_seq_len": int(config.text_seq_len),
        "valid_text": int(args.valid_text), "num_image_tokens": int(config.num_image_tokens),
        "seed": SEED, "model_dir": str(args.model_dir), "seconds": seconds,
        "build": _build_info(),
    }, out_path)
    print(f"wrote {out_path}: layers={layers} single_layers={single} "
          f"{config.num_frames}x{config.height}x{config.width} ({config.num_image_tokens} image "
          f"tokens) valid_text={args.valid_text} timesteps={list(args.timesteps)} "
          f"load={load_s:.1f}s fp32={seconds['fp32_forwards']:.1f}s "
          f"bf16={seconds['bf16_forwards']:.1f}s", flush=True)
    return 0


def _build_info() -> dict[str, Any]:
    from tests.numerical.buildinfo import build_info

    return build_info()


# ---- phase 2: the device check --------------------------------------------------------------


def resolve_layout(args, world: int) -> tuple[int, int]:
    """``(tp, cp)`` for ``world`` ranks: ``--cp-degree`` defaults to ``world / tp``."""
    tp = int(args.tp_degree)
    cp = int(args.cp_degree) if args.cp_degree is not None else max(1, world // tp)
    if tp * cp != world:
        raise SystemExit(f"--tp-degree {tp} x --cp-degree {cp} = {tp * cp} ranks, but "
                         f"WORLD_SIZE={world}")
    if cp > 1 and args.sp:
        raise SystemExit("--sp and --cp-degree > 1 are mutually exclusive (both shard the "
                         "image sequence)")
    return tp, cp


def check(args) -> int:
    t_start = time.perf_counter()
    oracle = None
    if args.oracle != "none":
        oracle = torch.load(Path(args.oracle).expanduser(), weights_only=False)
        if oracle.get("kind") != "hv1_dit_oracle":
            raise SystemExit(f"{args.oracle} is not a check_neuron_hv1_dit.py oracle")
        height, width, frames = oracle["height"], oracle["width"], oracle["num_frames"]
        layers, single = oracle["num_layers"], oracle["num_single_layers"]
        valid_text = oracle["valid_text"]
    else:
        height, width, frames = args.height, args.width, args.frames
        layers, single, valid_text = args.layers, args.single_layers, args.valid_text
    world = int(os.environ.get("WORLD_SIZE", "1"))
    tp, cp = resolve_layout(args, world)
    cp_mode = args.cp_mode if cp > 1 else "gather_kv"
    layout = layout_name(tp, cp, cp_mode, args.sp)

    from difflet.backends.neuron.compile import configure_compile_cache

    configure_compile_cache()
    import torch_neuronx  # noqa: F401  (registers the device; the runtime starts later)
    import torch_neuronx.metrics as neuron_metrics

    neuron_metrics.set_enabled(True)  # device_evidence's compilation counters

    from difflet.backends.neuron.hunyuan_video.transformer import (
        NeuronHunyuanVideoTransformerApplication,
    )
    from difflet.backends.neuron.runtime import track_fallbacks
    from difflet.backends.registry import get_backend
    from difflet.models.hunyuan_video.application import HunyuanVideoDiTInputBundle
    from difflet.models.hunyuan_video.neuron_application import (
        _OnDeviceHunyuanVideoTransformer,
    )
    from difflet.pipeline.parallel_config import DiffletParallelConfig
    from tests.manual import check_neuron_wan_preflight as wanpf

    parallel = DiffletParallelConfig(tp_degree=tp, cp_degree=cp, cp_mode=cp_mode,
                                     sp_enabled=bool(args.sp))
    get_backend("neuron").prepare_runtime(parallel)  # core binding + the numerics workarounds
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    say = print if rank == 0 else (lambda *a, **k: None)
    config = config_for(args.model_dir, height=height, width=width, frames=frames,
                        layers=layers, single_layers=single, tp_degree=tp,
                        context_parallel_enabled=cp > 1, cp_mode=cp_mode,
                        sp_enabled=bool(args.sp))
    app = NeuronHunyuanVideoTransformerApplication(
        model_path=transformer_dir(args.model_dir), config=config, parallel=parallel,
        dtype=torch.bfloat16, exec_mode=args.exec_mode, device="neuron")
    config = app.config
    t = time.perf_counter()
    with track_fallbacks() as warmup_fallbacks:
        app.load()
    load_s = time.perf_counter() - t
    say(f"{layout} {args.exec_mode}: layers={config.num_layers}+{config.num_single_layers} "
        f"{frames}x{height}x{width} ({config.num_image_tokens} image tokens) "
        f"valid_text={valid_text}; load+warmup {load_s:.1f} s phases="
        f"{ {k: round(v, 1) for k, v in app.phase_seconds.items()} }", flush=True)

    inputs = oracle["inputs"] if oracle is not None else make_inputs(config, valid_text)
    runs = oracle["runs"] if oracle is not None else [{"timestep": t} for t in args.timesteps]
    adapter = _OnDeviceHunyuanVideoTransformer(app, config, torch.bfloat16, app.device)
    adapter_s: list[float] = []
    adapter.forward_hook = adapter_s.append
    outputs: dict[str, torch.Tensor] = {}
    with track_fallbacks() as fallbacks:
        for run in runs:
            bundle = HunyuanVideoDiTInputBundle(
                timestep=_bf16_values(torch.tensor([float(run["timestep"])])), **inputs)
            outputs[f"t{float(run['timestep']):g}"] = adapter(bundle).float()

    bench_result = None
    with track_fallbacks() as bench_fallbacks:
        if args.iters > 0:
            dev = app.device
            device_args = tuple(
                (x.to(torch.bfloat16) if x.is_floating_point() else x).contiguous().to(dev)
                for x in HunyuanVideoDiTInputBundle(
                    timestep=_bf16_values(torch.tensor([float(runs[0]["timestep"])])),
                    **inputs).as_model_inputs())
            [bench_result] = bench.run({"dit_forward": lambda: app(*device_args)},
                                       warmup=1, iters=args.iters, rounds=args.rounds,
                                       memory_stats=bench.neuron_memory_stats)
            say(str(bench_result), flush=True)

    evidence = parity.device_evidence(app, fallbacks)
    hashes = wanpf.gather_hashes({k: wanpf.sha256(v) for k, v in outputs.items()}, world)
    checks: list[tuple[str, bool, str]] = []
    report = app.load_report or {}
    checks.append(("checkpoint", not report.get("missing") and not report.get("unexpected"),
                   f"missing={len(report.get('missing', []))} unexpected="
                   f"{len(report.get('unexpected', []))} skipped={len(report.get('skipped', []))}"))
    checks.append(("fallbacks", not (warmup_fallbacks or fallbacks or bench_fallbacks),
                   f"warm-up={list(warmup_fallbacks)} forwards={list(fallbacks)} "
                   f"benchmark={list(bench_fallbacks)}"))
    checks.append(("unwarmed_shapes", evidence["unwarmed_shapes"] == [],
                   f"{evidence['unwarmed_shapes']} warmup={app.warmup_shapes}"))
    differ = [r for r, h in enumerate(hashes) if h != hashes[0]]
    checks.append(("ranks_identical", len(hashes) == world and not differ,
                   f"{len(hashes)} ranks" + (f"; differ on ranks {differ}" if differ else "")))
    if args.exec_mode == "compile":
        expected = int(config.num_layers) + int(config.num_single_layers)
        checks.append(("compiled_blocks", len(app.compiled_blocks) == expected,
                       f"{len(app.compiled_blocks)} of {expected}"))
    else:
        checks.append(("eager_compiles_no_block", not app.compiled_blocks,
                       f"compiled_blocks={len(app.compiled_blocks)}"))
    for key, out in outputs.items():
        checks.append((f"finite_{key}", bool(torch.isfinite(out).all()),
                       f"absmean={float(out.double().abs().mean()):.4f} shape={tuple(out.shape)}"))

    parity_results: dict[str, Any] = {}
    for run in runs if oracle is not None else []:
        key = f"t{float(run['timestep']):g}"
        out, ref, ctl = outputs[key], run["ref_fp32"], run["control_bf16"]
        m = parity.metrics(out, ref, control=ctl)
        m["vs_bf16"] = parity.metrics(out, ctl)
        control = parity.metrics(ctl, ref)
        notes: list[str] = []
        ok, reasons = parity.check(m, control=control, notes=notes)
        ratio = m["rel_l1"] / control["rel_l1"] if control["rel_l1"] > 0 else float("inf")
        parity_results[key] = {"vs_fp32": m, "control": control, "ratio_vs_control": ratio,
                               "ok": ok, "reasons": reasons, "notes": notes}
        checks.append((f"parity_{key}", ok,
                       f"rel_l1={m['rel_l1']:.3e} ({ratio:.3f}x control {control['rel_l1']:.3e}"
                       f", gate {parity.MAX_RATIO_VS_CONTROL}x) cos={m['cosine']:.6f} "
                       f"err_cos={m['err_cos']:.3f} {parity.same_dtype_label(m, control)}"
                       + (f" {reasons}" if reasons else "") + (f" {notes}" if notes else "")))

    ok = all(passed for _, passed, _ in checks)
    result = {
        "rank": rank, "world_size": world, "layout": layout, "tp": tp, "cp": cp,
        "cp_mode": cp_mode, "sp": bool(args.sp), "exec_mode": args.exec_mode, "ok": ok,
        "oracle": None if oracle is None else str(args.oracle),
        "depth": {"num_layers": config.num_layers, "num_single_layers": config.num_single_layers},
        "shape": {"height": height, "width": width, "num_frames": frames,
                  "num_image_tokens": config.num_image_tokens, "valid_text": valid_text},
        "checks": [{"name": n, "ok": p, "detail": d} for n, p, d in checks],
        "parity": parity_results,
        "device_evidence": evidence,
        "adapter_seconds": adapter_s,
        "load_seconds": load_s,
        "benchmark": None if bench_result is None else asdict(bench_result),
        "rank_output_hashes": hashes,
        "total_seconds": time.perf_counter() - t_start,
    }
    if args.out_json is not None:
        path = Path(args.out_json).expanduser()
        if rank:
            path = path.with_name(f"{path.stem}.rank{rank}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, default=str))
    if args.save is not None and rank == 0:
        torch.save({"outputs": outputs, "layout": layout, "exec_mode": args.exec_mode},
                   Path(args.save).expanduser())
    for name, passed, detail in checks:
        say(f"{'PASS' if passed else 'FAIL'} {name:24s} {detail}", flush=True)
    failed = [name for name, passed, _ in checks if not passed]
    peak = evidence["peak_device_mem_gb"]
    print(f"[rank {rank}] {'PASS' if ok else 'FAIL'} {args.exec_mode} {layout} "
          f"layers={config.num_layers}+{config.num_single_layers} valid_text={valid_text} "
          f"load+warmup={load_s:.1f}s adapter={[round(s * 1e3, 1) for s in adapter_s]}ms "
          + (f"bench median={bench_result.median_ms:.1f}ms " if bench_result else "")
          + (f"peak_hbm={peak:.2f}GiB " if peak is not None else "")
          + (f"failed={failed}" if failed else ""), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0 if ok else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    summary, details = __doc__.split("\n\n", 1)
    p = argparse.ArgumentParser(description=" ".join(summary.split()), epilog=details,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="the diffusers snapshot root")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--make-oracle", metavar="PATH", help="phase 1: write the host oracle")
    mode.add_argument("--oracle", metavar="PATH|none",
                      help="phase 2: check the device against this oracle ('none': timing only)")
    p.add_argument("--iters", type=int, default=5, help="timed device forwards per round")
    p.add_argument("--rounds", type=int, default=2, help="benchmark rounds")
    p.add_argument("--exec-mode", choices=("compile", "eager"), default="compile")
    p.add_argument("--tp-degree", type=int, default=2)
    p.add_argument("--cp-degree", type=int, default=None,
                   help="context parallelism over the image tokens (default: world / tp)")
    p.add_argument("--cp-mode", choices=("gather_kv", "ulysses", "ring"), default="gather_kv")
    p.add_argument("--sp", action="store_true", help="sequence parallelism within the TP group")
    p.add_argument("--frames", type=int, default=FRAMES)
    p.add_argument("--height", type=int, default=H)
    p.add_argument("--width", type=int, default=W)
    p.add_argument("--valid-text", type=int, default=VALID_TEXT,
                   help=f"valid tokens of the {TEXT_SEQ_LEN}-token text (right-padded)")
    p.add_argument("--layers", type=int, default=None,
                   help="dual-stream blocks (default: the checkpoint's 20)")
    p.add_argument("--single-layers", type=int, default=None,
                   help="single-stream blocks (default: the checkpoint's 40)")
    p.add_argument("--timesteps", type=float, nargs="+", default=list(TIMESTEPS),
                   help="oracle timesteps, one gated forward each")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 1,
                   help="host threads for the oracle")
    p.add_argument("--out-json", default=None, help="phase 2: the result (rank 0; others .rankN)")
    p.add_argument("--save", default=None, help="phase 2: rank 0's outputs (torch.save)")
    p.add_argument("--dry-run", action="store_true",
                   help="phase 2 outside torchrun: print the torchrun command, do not run it")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    if args.make_oracle:
        return make_oracle(args)
    if "WORLD_SIZE" not in os.environ:
        cp = args.cp_degree if args.cp_degree is not None else 1
        return launch(int(args.tp_degree) * int(cp), argv, dry_run=args.dry_run)
    return check(args)


if __name__ == "__main__":
    sys.exit(main())
