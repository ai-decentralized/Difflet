#!/usr/bin/env python3
"""Phase-0 FP8 PTQ probe on a tiny Wan backbone: compile, load, numerics vs CPU.

Runs the production path (offline fp8 checkpoint -> NxD quantized parallel
linears -> neuronx-cc -> load -> forward) on a random 2-block Wan transformer,
so the design-spec assumptions are checked in minutes instead of after a 14B
compile:

  A1  scale layout / key names load into NxD's quantized layers
  A2  dynamic activation quantization matches the CPU reference (device-fp8 vs
      CPU-fp8 cosine ~ 1; both differ from bf16 by the same quantization error)
  A4  fp8 tensors survive sharding + presharded safetensors
  A5  the fp8 compiler flag reaches neuronx-cc

It also times N forwards of the bf16 and fp8 NEFFs. At this tiny shape that is
a sanity signal only; the per-step number that matters comes from the 14B A/B
(scripts/ptq_fp8_ab.py).

    PYTHONPATH=$PWD python scripts/ptq_fp8_device_probe.py --work-dir /tmp/ptq_probe \\
        [--quant-granularity tensor|channel] [--only bf16|fp8|both]

The CPU reference and the device build run in separate processes (the op
dispatch is frozen at first import per process: a process that bound the CPU
backend cannot build the Neuron model). ``--stage cpu`` / ``--stage device`` are
those halves; the default ``--stage all`` runs both and writes the report.
Exit 0 when the fp8 arm compiles, loads, and matches the CPU fp8 reference
(cosine >= --min-cosine); the JSON report is written either way.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NEURON_VENV = Path("/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference")
NEURON_PYTHON = NEURON_VENV / "bin" / "python"

TINY_CONFIG = {
    "_class_name": "WanTransformer3DModel",
    "patch_size": [1, 2, 2],
    "num_attention_heads": 4,
    "attention_head_dim": 32,
    "in_channels": 16,
    "out_channels": 16,
    "text_dim": 64,
    "freq_dim": 64,
    "ffn_dim": 256,
    "num_layers": 2,
    "cross_attn_norm": True,
    "qk_norm": "rms_norm_across_heads",
    "rope_max_seq_len": 1024,
    "eps": 1e-6,
}
# Difflet <-> diffusers FFN naming (the on-device loader reads diffusers keys).
_TO_DIFFUSERS = ((".ffn.net_in.", ".ffn.net.0.proj."), (".ffn.net_out.", ".ffn.net.2."))
REPORT = "ptq_probe_report.json"


def ensure_runtime_python() -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        if Path(sys.executable) != NEURON_PYTHON and NEURON_PYTHON.exists():
            env = os.environ.copy()
            env["PATH"] = f"{NEURON_VENV / 'bin'}:{env.get('PATH', '')}"
            env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
            os.execve(str(NEURON_PYTHON), [str(NEURON_PYTHON), *sys.argv], env)
        raise


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", type=Path, default=Path("/tmp/difflet_ptq_probe"))
    p.add_argument("--stage", choices=["all", "cpu", "device"], default="all")
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--text-seq-len", type=int, default=512, help="traced text length (production default 512)")
    p.add_argument("--quant-granularity", choices=["tensor", "channel"], default="tensor")
    p.add_argument("--only", choices=["bf16", "fp8", "both"], default="both")
    p.add_argument("--iters", type=int, default=10, help="timed forwards per arm (after 2 warmups)")
    p.add_argument("--min-cosine", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force-clean", action="store_true")
    return p


def _spec(args):
    from difflet.quant.spec import QuantSpec

    return QuantSpec(weight_granularity=args.quant_granularity)


def _first(value):
    return value[0] if isinstance(value, (list, tuple)) else value


# ------------------------------------------------------------------ cpu stage


def stage_cpu(args) -> int:
    """Random tiny Wan transformer in HF layout + the CPU bf16 / fp8 references."""
    os.environ["DIFFLET_BACKEND"] = "cpu"
    import torch
    from safetensors.torch import save_file

    import difflet.models.wan.modeling_wan as wan
    from difflet.quant.fake_linear import quantize_module_

    cfg = wan.WanTransformerConfig.from_diffusers_dict(TINY_CONFIG)
    torch.manual_seed(args.seed)
    model = wan.WanTransformer3DModel(cfg).to(torch.bfloat16).eval()
    state = {}
    for key, value in model.state_dict().items():
        if key.endswith(".rank"):
            continue
        for ours, theirs in _TO_DIFFUSERS:
            key = key.replace(ours, theirs)
        state[key] = value.contiguous()
    transformer_dir = args.work_dir / "tiny_wan" / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    save_file(state, str(transformer_dir / "diffusion_pytorch_model.safetensors"))
    (transformer_dir / "config.json").write_text(json.dumps(TINY_CONFIG, indent=2))

    g = torch.Generator().manual_seed(args.seed + 1)
    latents = torch.randn(1, cfg.in_channels, 1, args.height // 8, args.width // 8, generator=g)
    text = torch.randn(1, args.text_seq_len, cfg.text_dim, generator=g)
    inputs = (latents.to(torch.bfloat16), torch.tensor([500.0]).to(torch.bfloat16), text.to(torch.bfloat16))
    with torch.no_grad():
        cpu_bf16 = _first(model(*inputs)).float()
        quantize_module_(model, _spec(args))
        cpu_fp8 = _first(model(*inputs)).float()
    torch.save({"inputs": inputs, "cpu_bf16": cpu_bf16, "cpu_fp8": cpu_fp8}, args.work_dir / "cpu_reference.pt")
    print(f"[probe:cpu] tiny model + references written under {args.work_dir}", flush=True)
    return 0


# --------------------------------------------------------------- device stage


def _device_arm(name, transformer_dir, args, inputs, spec, cache_dir):
    """Compile + load + time one arm on the device; returns (output, record)."""
    import torch

    from difflet.backends.trainium.wan.backbone import NeuronWanBackboneApplication
    from difflet.models.wan.application import create_wan_backbone_config
    from difflet.quant.checkpoint import ensure_quantized_checkpoint, quantized_checkpoint_dir

    record = {"arm": name, "compiled": False, "loaded": False}
    quant_dir = None
    if spec is not None:
        quant_dir = quantized_checkpoint_dir(cache_dir, transformer_dir, spec)
        started = time.perf_counter()
        ensure_quantized_checkpoint(transformer_dir, quant_dir, spec, create=True)
        record["quantize_seconds"] = round(time.perf_counter() - started, 3)
        record["quantized_checkpoint"] = str(quant_dir)
    config = create_wan_backbone_config(
        model_path=str(transformer_dir.parent), world_size=1, tp_degree=1, dtype=torch.bfloat16,
        height=args.height, width=args.width, num_frames=1, batch_size=1,
        quant=spec, quant_checkpoint_dir=quant_dir,
    )
    # The backbone traces its text input at config.text_seq_len (512 unless
    # set); the loaded graph rejects any other shape, so trace at the probe's.
    config.text_seq_len = args.text_seq_len
    app = NeuronWanBackboneApplication(model_path=str(transformer_dir), config=config)
    record["compiler_args"] = app.get_compiler_args()
    compiled_dir = args.work_dir / f"compiled_{name}"
    started = time.perf_counter()
    app.compile(str(compiled_dir))
    record["compile_seconds"] = round(time.perf_counter() - started, 3)
    record["compiled"] = True
    started = time.perf_counter()
    app.load(str(compiled_dir), start_rank_id=0, local_ranks_size=1, skip_warmup=True)
    record["load_seconds"] = round(time.perf_counter() - started, 3)
    record["loaded"] = True
    for _ in range(2):
        _first(app(*inputs))
    samples = []
    output = None
    for _ in range(max(1, args.iters)):
        started = time.perf_counter()
        output = _first(app(*inputs))
        samples.append((time.perf_counter() - started) * 1000.0)
    record["forward_ms"] = {"n": len(samples), "mean": statistics.fmean(samples),
                            "median": statistics.median(samples), "min": min(samples), "max": max(samples)}
    output = output.detach().cpu().float()
    torch.save(output, args.work_dir / f"device_{name}_output.pt")  # kept for NaN / pattern inspection
    record["output_nonfinite"] = int((~torch.isfinite(output)).sum())
    return output, record


def stage_device(args) -> int:
    os.environ.setdefault("NEURON_RT_NUM_CORES", "1")
    import torch

    from difflet.quant.metrics import tensor_error_metrics

    reference = torch.load(args.work_dir / "cpu_reference.pt", map_location="cpu")
    inputs, cpu_bf16, cpu_fp8 = reference["inputs"], reference["cpu_bf16"], reference["cpu_fp8"]
    spec = _spec(args)
    report_path = args.work_dir / REPORT
    report = {
        "spec": spec.to_dict(),
        "shape": {"height": args.height, "width": args.width, "text_seq_len": args.text_seq_len},
        "cpu_fp8_vs_cpu_bf16": tensor_error_metrics(cpu_bf16, cpu_fp8),
        "arms": {},
        "checks": {},
        "passed": False,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    outputs = {}
    transformer_dir = args.work_dir / "tiny_wan" / "transformer"
    for name in ("bf16", "fp8"):
        if args.only not in ("both", name):
            continue
        try:
            output, record = _device_arm(name, transformer_dir, args, inputs,
                                         spec if name == "fp8" else None, args.work_dir / "cache")
            outputs[name] = output
        except Exception as exc:  # keep the partial report: the failure IS the finding
            import traceback

            record = {"arm": name, "error": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc()}
        report["arms"][name] = record
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    checks = report["checks"]
    if "bf16" in outputs:
        checks["device_bf16_vs_cpu_bf16"] = tensor_error_metrics(cpu_bf16, outputs["bf16"])
    if "fp8" in outputs:
        checks["device_fp8_vs_cpu_fp8"] = tensor_error_metrics(cpu_fp8, outputs["fp8"])
        checks["device_fp8_vs_cpu_bf16"] = tensor_error_metrics(cpu_bf16, outputs["fp8"])
    if "bf16" in outputs and "fp8" in outputs:
        checks["device_fp8_vs_device_bf16"] = tensor_error_metrics(outputs["bf16"], outputs["fp8"])
    if args.only == "bf16":
        report["passed"] = "bf16" in outputs
    else:
        report["passed"] = bool(
            report["arms"].get("fp8", {}).get("loaded", False)
            and checks.get("device_fp8_vs_cpu_fp8", {}).get("cosine", 0.0) >= args.min_cosine
        )
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps({k: v for k, v in report.items() if k != "arms"}, indent=2))
    for name, record in report["arms"].items():
        summary = {k: v for k, v in record.items() if k not in ("traceback",)}
        print(f"[probe] {name}: {json.dumps(summary)}")
        if "traceback" in record:
            print(record["traceback"])
    print(f"[probe] report: {report_path}  passed={report['passed']}")
    return 0 if report["passed"] else 1


def main() -> int:
    ensure_runtime_python()
    args = build_parser().parse_args()
    if args.stage == "cpu":
        return stage_cpu(args)
    if args.stage == "device":
        return stage_device(args)
    if args.force_clean and args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    forwarded = [a for a in sys.argv[1:] if a not in ("--force-clean",)]
    base = [sys.executable, str(Path(__file__).resolve()), *forwarded]
    env = dict(os.environ, PYTHONPATH=f"{ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    cpu = subprocess.run(base + ["--stage", "cpu"], env=dict(env, DIFFLET_BACKEND="cpu"))
    if cpu.returncode != 0:
        return cpu.returncode
    device_env = dict(env)
    device_env.pop("DIFFLET_BACKEND", None)  # auto-detects trainium in the Neuron venv
    return subprocess.run(base + ["--stage", "device"], env=device_env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
