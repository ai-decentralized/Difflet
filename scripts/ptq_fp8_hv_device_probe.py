#!/usr/bin/env python3
"""FP8 PTQ probe on a tiny HunyuanVideo 1.0 backbone: compile, load, numerics vs CPU.

The HunyuanVideo twin of ``scripts/ptq_fp8_device_probe.py`` (Wan). It runs the
production path (offline fp8 checkpoint -> NxD quantized parallel linears ->
neuronx-cc -> load -> forward) on a random tiny HunyuanVideo transformer with a
**padded** text sequence (3/4 of the rows masked, like the Llama stage's 256-row
input), so the model-specific pieces are checked in minutes:

  * dual-stream blocks (``add_*`` / ``to_add_out`` / ``ff_context``) and
    single-stream blocks with the fused ``proj_out`` split into
    ``proj_out_attn`` (``skip_bias_add``, ``reduce_output=False``) + ``proj_out_mlp``
  * the token refiner left in bf16
  * the attention mask over padded text rows
  * Difflet's per-tensor dynamic activation path (W8A8)

    PYTHONPATH=$PWD python scripts/ptq_fp8_hv_device_probe.py --work-dir /tmp/ptq_hv_probe \\
        [--quant-granularity tensor|channel] [--only bf16|fp8|both] [--num-single-layers 2] [--text-seq-len 64]

The CPU reference and the device build run in separate processes (op dispatch is
frozen at first import). Exit 0 when the fp8 arm compiles, loads, and matches the
CPU fp8 reference (cosine >= --min-cosine); the JSON report is written either way.
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
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NEURON_VENV = Path("/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference")
NEURON_PYTHON = NEURON_VENV / "bin" / "python"
REPORT = "ptq_hv_probe_report.json"


def tiny_config(args) -> dict:
    if getattr(args, "real_model_dir", None):
        # The real transformer config, truncated to the probe's block counts: a
        # device-vs-CPU check with the real weights' activation statistics.
        raw = json.loads((args.real_model_dir / "config.json").read_text())
        raw["num_layers"] = args.num_layers
        raw["num_single_layers"] = args.num_single_layers
        return raw
    return {
        "_class_name": "HunyuanVideoTransformer3DModel",
        # 128 = the real head dim; attention_cte rejects 32 once heads are sharded (tp > 1)
        "attention_head_dim": args.head_dim,
        "guidance_embeds": True,
        "in_channels": 16,
        "mlp_ratio": 4.0,
        "num_attention_heads": 4,
        "num_layers": args.num_layers,
        "num_refiner_layers": 1,
        "num_single_layers": args.num_single_layers,
        "out_channels": 16,
        "patch_size": 2,
        "patch_size_t": 1,
        "pooled_projection_dim": 32,
        "qk_norm": "rms_norm",
        "rope_axes_dim": [8, 12, 12] if args.head_dim == 32 else [16, 56, 56],
        "rope_theta": 256.0,
        "text_embed_dim": 64,
    }


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
    p.add_argument("--work-dir", type=Path, default=Path("/tmp/difflet_ptq_hv_probe"))
    p.add_argument("--stage", choices=["all", "cpu", "device"], default="all")
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--num-frames", type=int, default=5, help="pixel frames (latent frames = (n-1)//4+1)")
    p.add_argument("--text-seq-len", type=int, default=64)
    p.add_argument("--valid-text-rows", type=int, default=16, help="unmasked text rows (the rest are pads)")
    p.add_argument("--pad-value", type=float, default=0.0,
                   help="pad rows are N(0,1)*pad_value; 0 = zeroed pads (the pipeline's behaviour)")
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--num-single-layers", type=int, default=2)
    p.add_argument("--quant-granularity", choices=["tensor", "channel"], default="tensor")
    p.add_argument("--only", choices=["bf16", "fp8", "both"], default="both")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--min-cosine", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--real-model-dir", type=Path, default=None,
                   help="HF transformer/ dir: use the REAL config and weights, truncated to --num-layers / "
                        "--num-single-layers (device-vs-CPU check with real activation statistics)")
    p.add_argument("--text-pt", type=Path, default=None,
                   help="real conditioning {encoder_hidden_states, encoder_attention_mask, pooled_projections}; "
                        "pass --text-seq-len / --valid-text-rows matching it for the device stage")
    p.add_argument("--tp-degree", type=int, default=1, help="device tensor-parallel degree (production: 4)")
    p.add_argument("--text-pad-to", type=int, default=0,
                   help="with --text-pt: zero-pad the text to this many rows (masked); pass the same --text-seq-len")
    p.add_argument("--head-dim", type=int, default=32, help="tiny-model attention head dim (128 for tp > 1)")
    p.add_argument("--targets", default=None,
                   help="comma-separated fp8 target globs replacing the model's set (layer-type bisection)")
    p.add_argument("--force-clean", action="store_true")
    return p


def _spec(args):
    from difflet.quant.spec import QuantSpec

    spec = QuantSpec.for_model("hunyuan_video", weight_granularity=args.quant_granularity)
    if getattr(args, "targets", None):
        # Restrict the fp8 layer set (layer-type bisection); same matching rules.
        spec = QuantSpec(weight_granularity=args.quant_granularity,
                         targets=tuple(t for t in args.targets.split(",") if t))
    return spec


def _first(value):
    if isinstance(value, dict):  # the device backbone returns {"sample": tensor}
        return value["sample"] if "sample" in value else next(iter(value.values()))
    return value[0] if isinstance(value, (list, tuple)) else value


# ------------------------------------------------------------------ cpu stage


def _to_diffusers_layout(state: dict, num_single_layers: int) -> dict:
    """Inverse of the device converter: fuse proj_out_attn/proj_out_mlp back into proj_out."""
    import torch

    out = {k: v.contiguous() for k, v in state.items() if not k.endswith(".rank")}
    for i in range(num_single_layers):
        p = f"single_transformer_blocks.{i}"
        attn_w = out.pop(f"{p}.proj_out_attn.weight")
        mlp_w = out.pop(f"{p}.proj_out_mlp.weight")
        out[f"{p}.proj_out.weight"] = torch.cat([attn_w, mlp_w], dim=1).contiguous()
        out[f"{p}.proj_out.bias"] = out.pop(f"{p}.proj_out_attn.bias")
    return out


def _real_truncated_state(model_dir, num_layers: int, num_single_layers: int) -> dict:
    """The real HF transformer tensors for the first ``num_layers`` double and
    ``num_single_layers`` single blocks plus every non-block tensor (bf16)."""
    import torch
    from safetensors import safe_open

    index = json.loads((model_dir / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]

    def keep(key: str) -> bool:
        for prefix, limit in (("transformer_blocks.", num_layers), ("single_transformer_blocks.", num_single_layers)):
            if key.startswith(prefix):
                return int(key[len(prefix):].split(".")[0]) < limit
        return True

    wanted: dict[str, list[str]] = {}
    for key, shard in index.items():
        if keep(key):
            wanted.setdefault(shard, []).append(key)
    state = {}
    for shard, keys in wanted.items():
        with safe_open(str(model_dir / shard), "pt") as f:
            for key in keys:
                state[key] = f.get_tensor(key).to(torch.bfloat16).contiguous()
    return state


def stage_cpu(args) -> int:
    os.environ["DIFFLET_BACKEND"] = "cpu"
    import torch
    from safetensors.torch import save_file

    import difflet.models.hunyuan_video.modeling_hunyuan_video as hv
    from difflet.quant.fake_linear import quantize_module_

    raw = tiny_config(args)
    cfg = hv.HunyuanVideoTransformerConfig.from_diffusers_dict(raw)
    torch.manual_seed(args.seed)
    model = hv.HunyuanVideoTransformer3DModel(cfg).to(torch.bfloat16).eval()
    transformer_dir = args.work_dir / "tiny_hv" / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    if args.real_model_dir:
        state = _real_truncated_state(args.real_model_dir, cfg.num_layers, cfg.num_single_layers)
        save_file(state, str(transformer_dir / "diffusion_pytorch_model.safetensors"))
        from difflet.backends.trainium.hunyuan_video.backbone import NeuronHunyuanVideoBackboneApplication

        converted = NeuronHunyuanVideoBackboneApplication.convert_hf_to_neuron_state_dict(
            dict(state), types.SimpleNamespace(
                num_attention_heads=cfg.num_attention_heads, attention_head_dim=cfg.attention_head_dim,
                num_single_layers=cfg.num_single_layers, neuron_config=types.SimpleNamespace(world_size=1)))
        converted = {k: v for k, v in converted.items() if not k.endswith(".rank")}
        missing, unexpected = model.load_state_dict(converted, strict=False)
        print(f"[probe:cpu] real weights: {len(state)} HF tensors, missing {len(missing)}, unexpected {len(unexpected)}",
              flush=True)
    else:
        save_file(_to_diffusers_layout(model.state_dict(), cfg.num_single_layers),
                  str(transformer_dir / "diffusion_pytorch_model.safetensors"))
    (transformer_dir / "config.json").write_text(json.dumps(raw, indent=2))

    g = torch.Generator().manual_seed(args.seed + 1)
    latent_frames = (args.num_frames - 1) // 4 + 1
    latents = torch.randn(1, cfg.in_channels, latent_frames, args.height // 8, args.width // 8, generator=g)
    if args.text_pt:
        # Real conditioning (the pipeline zeroes the pad rows before the DiT).
        real = torch.load(args.text_pt, map_location="cpu")
        text = real["encoder_hidden_states"].float()
        mask = real["encoder_attention_mask"].to(torch.int64)
        text = text * mask.unsqueeze(-1).to(text.dtype)
        pooled = real["pooled_projections"].float()
        if args.text_pad_to and args.text_pad_to > text.shape[1]:
            # extra zero rows, masked out: same tokens, a longer text sequence (shape test)
            extra = args.text_pad_to - text.shape[1]
            text = torch.cat([text, text.new_zeros(text.shape[0], extra, text.shape[2])], dim=1)
            mask = torch.cat([mask, mask.new_zeros(mask.shape[0], extra)], dim=1)
        args.text_seq_len = int(text.shape[1])
        args.valid_text_rows = int(mask.sum())
    else:
        text = torch.randn(1, args.text_seq_len, cfg.text_embed_dim, generator=g)
        mask = torch.zeros(1, args.text_seq_len, dtype=torch.int64)
        mask[:, : args.valid_text_rows] = 1
        text[:, args.valid_text_rows:] *= args.pad_value
        pooled = torch.randn(1, cfg.pooled_projection_dim, generator=g)
    inputs = (
        latents.to(torch.bfloat16),
        torch.tensor([500.0]).to(torch.bfloat16),
        text.to(torch.bfloat16),
        mask,
        pooled.to(torch.bfloat16),
        torch.tensor([6000.0]).to(torch.bfloat16),
    )
    with torch.no_grad():
        cpu_bf16 = _first(model(*inputs, return_dict=False)).float()
        report = quantize_module_(model, _spec(args))
        cpu_fp8 = _first(model(*inputs, return_dict=False)).float()
    torch.save({"inputs": inputs, "cpu_bf16": cpu_bf16, "cpu_fp8": cpu_fp8}, args.work_dir / "cpu_reference.pt")
    print(f"[probe:cpu] tiny HunyuanVideo + references under {args.work_dir}; "
          f"{report['num_quantized']} linears fake-quantized", flush=True)
    return 0


# --------------------------------------------------------------- device stage


def _device_arm(name, transformer_dir, args, inputs, spec, cache_dir):
    import torch

    from difflet.backends.trainium.hunyuan_video.backbone import NeuronHunyuanVideoBackboneApplication
    from difflet.models.hunyuan_video.application import create_hunyuan_video_backbone_config
    from difflet.quant.checkpoint import ensure_quantized_checkpoint, quantized_checkpoint_dir

    record = {"arm": name, "compiled": False, "loaded": False}
    quant_dir = None
    if spec is not None:
        quant_dir = quantized_checkpoint_dir(cache_dir, transformer_dir, spec)
        started = time.perf_counter()
        ensure_quantized_checkpoint(transformer_dir, quant_dir, spec, create=True)
        record["quantize_seconds"] = round(time.perf_counter() - started, 3)
        record["quantized_checkpoint"] = str(quant_dir)
    config = create_hunyuan_video_backbone_config(
        model_path=str(transformer_dir.parent), world_size=args.tp_degree, tp_degree=args.tp_degree, dtype=torch.bfloat16,
        height=args.height, width=args.width, num_frames=args.num_frames, text_seq_len=args.text_seq_len,
        batch_size=1, quant=spec, quant_checkpoint_dir=quant_dir,
    )
    app = NeuronHunyuanVideoBackboneApplication(model_path=str(transformer_dir), config=config)
    record["compiler_args"] = app.get_compiler_args()
    compiled_dir = args.work_dir / f"compiled_{name}"
    started = time.perf_counter()
    app.compile(str(compiled_dir))
    record["compile_seconds"] = round(time.perf_counter() - started, 3)
    record["compiled"] = True
    started = time.perf_counter()
    app.load(str(compiled_dir), start_rank_id=0, local_ranks_size=args.tp_degree, skip_warmup=True)
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
    torch.save(output, args.work_dir / f"device_{name}_output.pt")
    record["output_nonfinite"] = int((~torch.isfinite(output)).sum())
    record["output_absmax"] = float(output.abs().max())
    return output, record


def stage_device(args) -> int:
    os.environ.setdefault("NEURON_RT_NUM_CORES", str(args.tp_degree))
    import torch

    from difflet.quant.metrics import tensor_error_metrics

    reference = torch.load(args.work_dir / "cpu_reference.pt", map_location="cpu")
    inputs, cpu_bf16, cpu_fp8 = reference["inputs"], reference["cpu_bf16"], reference["cpu_fp8"]
    spec = _spec(args)
    report_path = args.work_dir / REPORT
    report = {
        "spec": spec.to_dict(),
        "shape": {"height": args.height, "width": args.width, "num_frames": args.num_frames,
                  "text_seq_len": args.text_seq_len, "valid_text_rows": args.valid_text_rows,
                  "pad_value": args.pad_value, "num_layers": args.num_layers,
                  "num_single_layers": args.num_single_layers},
        "cpu_fp8_vs_cpu_bf16": tensor_error_metrics(cpu_bf16, cpu_fp8),
        "arms": {},
        "checks": {},
        "passed": False,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    outputs = {}
    transformer_dir = args.work_dir / "tiny_hv" / "transformer"
    for name in ("bf16", "fp8"):
        if args.only not in ("both", name):
            continue
        try:
            output, record = _device_arm(name, transformer_dir, args, inputs,
                                         spec if name == "fp8" else None, args.work_dir / "cache")
            outputs[name] = output
        except Exception as exc:  # keep the partial report: the failure IS the finding
            import traceback

            record = {"arm": name, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
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
    device_env.pop("DIFFLET_BACKEND", None)
    return subprocess.run(base + ["--stage", "device"], env=device_env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
