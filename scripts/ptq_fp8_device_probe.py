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
        [--quant-granularity tensor|channel] [--quant-act dynamic|none] [--only bf16|fp8|both]

Exit 0 when the fp8 arm compiles, loads, and matches the CPU fp8 reference
(cosine >= --min-cosine); the JSON report is written either way.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
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
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--text-seq-len", type=int, default=16)
    p.add_argument("--quant-granularity", choices=["tensor", "channel"], default="tensor")
    p.add_argument("--quant-act", choices=["dynamic", "none"], default="dynamic")
    p.add_argument("--only", choices=["bf16", "fp8", "both"], default="both")
    p.add_argument("--iters", type=int, default=10, help="timed forwards per arm (after 2 warmups)")
    p.add_argument("--min-cosine", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force-clean", action="store_true")
    return p


def _write_tiny_model(model_dir: Path, seed: int):
    """Random tiny Wan transformer in HF layout + the CPU bf16/fp8 references."""
    import torch
    from safetensors.torch import save_file

    os.environ["DIFFLET_BACKEND"] = "cpu"
    import importlib

    import difflet.ops as ops

    importlib.reload(ops)
    import difflet.models.wan.modeling_wan as wan

    importlib.reload(wan)
    from difflet.quant.fake_linear import quantize_module_
    from difflet.quant.spec import QuantSpec

    cfg = wan.WanTransformerConfig.from_diffusers_dict(TINY_CONFIG)
    torch.manual_seed(seed)
    model = wan.WanTransformer3DModel(cfg).to(torch.bfloat16).eval()
    state = {}
    for key, value in model.state_dict().items():
        if key.endswith(".rank"):
            continue
        for ours, theirs in _TO_DIFFUSERS:
            key = key.replace(ours, theirs)
        state[key] = value.contiguous()
    transformer_dir = model_dir / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    save_file(state, str(transformer_dir / "diffusion_pytorch_model.safetensors"))
    (transformer_dir / "config.json").write_text(json.dumps(TINY_CONFIG, indent=2))
    return model, cfg, QuantSpec, quantize_module_


def _inputs(cfg, args, seed):
    import torch

    g = torch.Generator().manual_seed(seed + 1)
    latents = torch.randn(1, cfg.in_channels, 1, args.height // 8, args.width // 8, generator=g)
    text = torch.randn(1, args.text_seq_len, cfg.text_dim, generator=g)
    timestep = torch.tensor([500.0])
    return latents.to(torch.bfloat16), timestep.to(torch.bfloat16), text.to(torch.bfloat16)


def _first(value):
    return value[0] if isinstance(value, (list, tuple)) else value


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
    return output.detach().cpu(), record


def main() -> int:
    ensure_runtime_python()
    args = build_parser().parse_args()
    os.environ.setdefault("NEURON_RT_NUM_CORES", "1")
    if args.force_clean and args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.work_dir / "ptq_probe_report.json"

    import torch

    from difflet.quant.metrics import tensor_error_metrics

    model_dir = args.work_dir / "tiny_wan"
    model, cfg, QuantSpec, quantize_module_ = _write_tiny_model(model_dir, args.seed)
    spec = QuantSpec(weight_granularity=args.quant_granularity, activation=args.quant_act)
    inputs = _inputs(cfg, args, args.seed)
    with torch.no_grad():
        cpu_bf16 = _first(model(*inputs)).float()
        quantize_module_(model, spec)
        cpu_fp8 = _first(model(*inputs)).float()
    report = {
        "spec": spec.to_dict(),
        "shape": {"height": args.height, "width": args.width, "text_seq_len": args.text_seq_len},
        "cpu_fp8_vs_cpu_bf16": tensor_error_metrics(cpu_bf16, cpu_fp8),
        "arms": {},
        "checks": {},
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    os.environ["DIFFLET_BACKEND"] = "trainium"
    cache_dir = args.work_dir / "cache"
    outputs = {}
    for name in ("bf16", "fp8"):
        if args.only not in ("both", name):
            continue
        try:
            output, record = _device_arm(name, model_dir / "transformer", args, inputs,
                                         spec if name == "fp8" else None, cache_dir)
            outputs[name] = output.float()
        except Exception as exc:  # keep the partial report: the failure IS the finding
            import traceback

            record = {"arm": name, "error": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc()}
        report["arms"][name] = record
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    if "bf16" in outputs:
        report["checks"]["device_bf16_vs_cpu_bf16"] = tensor_error_metrics(cpu_bf16, outputs["bf16"])
    if "fp8" in outputs:
        report["checks"]["device_fp8_vs_cpu_fp8"] = tensor_error_metrics(cpu_fp8, outputs["fp8"])
        report["checks"]["device_fp8_vs_cpu_bf16"] = tensor_error_metrics(cpu_bf16, outputs["fp8"])
    if "bf16" in outputs and "fp8" in outputs:
        report["checks"]["device_fp8_vs_device_bf16"] = tensor_error_metrics(outputs["bf16"], outputs["fp8"])
    fp8_ok = (
        report["arms"].get("fp8", {}).get("loaded", False)
        and report["checks"].get("device_fp8_vs_cpu_fp8", {}).get("cosine", 0.0) >= args.min_cosine
    )
    report["passed"] = bool(fp8_ok) if args.only != "bf16" else bool(outputs.get("bf16") is not None)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps({k: v for k, v in report.items() if k != "arms"}, indent=2))
    for name, record in report["arms"].items():
        summary = {k: v for k, v in record.items() if k not in ("traceback", "compiler_args")}
        print(f"[probe] {name}: {json.dumps(summary)}")
        if "traceback" in record:
            print(record["traceback"])
    print(f"[probe] report: {report_path}  passed={report['passed']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
