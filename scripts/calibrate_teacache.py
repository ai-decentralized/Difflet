#!/usr/bin/env python3
"""Collect or fit TeaCache calibration evidence.

The default path fits the polynomial used by ``TeaCacheController`` from an
explicit ``--pairs-json`` artifact. Hardware collection is available only when
the caller passes explicit foreground acknowledgement flags.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from nova.pipeline.teacache import CALIBRATION_SCHEMA, TeaCacheCalibration


PAIRS_SCHEMA = "nova-m9-teacache-pairs-v1"
FOREGROUND_ACK = "I am running TeaCache T0 in the foreground"


def _load_pairs(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != PAIRS_SCHEMA:
        raise ValueError(f"unsupported TeaCache pairs schema: {doc.get('schema')!r}")
    samples = doc.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("pairs JSON must contain a non-empty samples list")
    return doc


def _sample_xy(samples: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    x_values: list[float] = []
    y_values: list[float] = []
    for sample in samples:
        x_values.append(float(sample["mod_input_diff_norm"]))
        y_values.append(float(sample["noise_pred_diff_norm"]))
    x = torch.tensor(x_values, dtype=torch.float64)
    y = torch.tensor(y_values, dtype=torch.float64)
    return x, y


def _poly_design(x: torch.Tensor, degree: int) -> torch.Tensor:
    columns = [torch.ones_like(x)]
    for power in range(1, int(degree) + 1):
        columns.append(x.pow(power))
    return torch.stack(columns, dim=1)


def _fit_poly(x: torch.Tensor, y: torch.Tensor, degree: int) -> torch.Tensor:
    design = _poly_design(x, degree)
    return torch.linalg.lstsq(design, y[:, None]).solution[: degree + 1, 0]


def _r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    residual = torch.sum((y_true - y_pred).pow(2))
    centered = torch.sum((y_true - torch.mean(y_true)).pow(2))
    if centered.item() == 0:
        return 1.0 if residual.item() == 0 else 0.0
    return float(1.0 - residual.item() / centered.item())


def _threshold_for_target(
    predicted_delta: torch.Tensor,
    *,
    num_steps: int,
    warmup_steps: int,
    cooldown_steps: int,
    target_speedup: float,
) -> float:
    if target_speedup <= 1.0:
        raise ValueError("--target-speedup must be > 1.0")
    skippable_steps = max(int(num_steps) - int(warmup_steps) - int(cooldown_steps), 1)
    target_skips = max(int(round(num_steps - num_steps / target_speedup)), 1)
    target_fraction = min(max(target_skips / skippable_steps, 0.0), 1.0)
    q = predicted_delta.new_tensor(target_fraction)
    return float(torch.quantile(predicted_delta, q).item())


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _mark_step() -> None:
    try:
        import torch_xla.core.xla_model as xm
    except ImportError:
        return
    xm.mark_step()


def _load_hv_bundle(bundle_path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    from safetensors.torch import load_file as load_safetensors_file

    meta_path = Path(str(bundle_path) + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tensors = {
        key: value.detach().cpu()
        for key, value in load_safetensors_file(str(bundle_path), device="cpu").items()
    }
    return meta, tensors


def _hv_shape_label(meta: dict[str, Any]) -> str:
    return f"{int(meta['height'])}x{int(meta['width'])}x{int(meta['num_frames'])}"


def _build_hv_bundle(meta: dict[str, Any], tensors: dict[str, torch.Tensor]):
    from nova.models.hunyuan_video.application import HunyuanVideoDiTInputBundle

    return HunyuanVideoDiTInputBundle(
        hidden_states=tensors["latents_init"],
        timestep=tensors["timesteps"][:1].clone(),
        encoder_hidden_states=tensors["encoder_hidden_states"],
        encoder_attention_mask=tensors["encoder_attention_mask"],
        pooled_projections=tensors["pooled_projections"],
        guidance=tensors["guidance"],
    )


def _load_hv_app(args: argparse.Namespace, meta: dict[str, Any]):
    from nova.models.hunyuan_video.application import NeuronHunyuanVideoApplication
    from nova.pipeline.parallel_config import NovaParallelConfig

    return NeuronHunyuanVideoApplication(
        model_path=args.source_dir,
        parallel=NovaParallelConfig(tp_degree=args.tp_degree, cp_enabled=False),
        dtype=_dtype_from_name(args.dtype),
        shape={
            "height": int(meta["height"]),
            "width": int(meta["width"]),
            "num_frames": int(meta["num_frames"]),
        },
        text_seq_len=int(meta["text_seq_len"]),
        enable_vae_decoder=False,
    )


def _require_hv_teacache_mod_input(app: Any) -> None:
    # cclog 74 architecture: the outer NeuronHunyuanVideoApplication delegates
    # teacache_mod_input to a sibling NeuronHunyuanVideoTeacacheProbeApplication.
    # The probe instance lives on app.teacache_probe; the outer app surfaces the
    # method via delegation.
    if not hasattr(app, "teacache_mod_input"):
        raise RuntimeError(
            "HunyuanVideo application does not expose teacache_mod_input. "
            "T0 must not load hardware until a real block-0 modulated-input "
            "hook or dedicated mod-input probe NEFF is available."
        )
    if getattr(app, "teacache_probe", None) is None:
        raise RuntimeError(
            "HunyuanVideo TeaCache probe NEFF application is not active. The "
            "outer application must instantiate NeuronHunyuanVideoTeacacheProbeApplication "
            "to expose the device-side block-0 modulated input. See cclog 74."
        )


def _collect_hv_bundle_pairs(
    args: argparse.Namespace,
    *,
    bundle_path: Path,
    split: str,
    app: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from nova.models.hunyuan_video.application import HunyuanVideoDiTInputBundle
    from nova.models.hunyuan_video.pipeline import (
        _batch_timestep,
        _component_dtype,
        _first_tensor,
        _teacache_mod_input,
    )

    meta, tensors = _load_hv_bundle(bundle_path)
    if app is None:
        app = _load_hv_app(args, meta)
        _require_hv_teacache_mod_input(app)
        app.load(args.compiled_dir, skip_warmup=bool(args.skip_warmup))
    pipe = app.pipeline
    if pipe is None or pipe.transformer is None:
        raise RuntimeError("HunyuanVideo pipeline transformer is unavailable after load")

    bundle = _build_hv_bundle(meta, tensors)
    timesteps = pipe._prepare_explicit_timesteps(  # noqa: SLF001
        tensors["timesteps"],
        device=tensors["latents_init"].device,
    )
    num_steps = int(args.num_inference_steps or int(timesteps.numel()))
    timesteps = timesteps[:num_steps]
    latents = bundle.hidden_states
    prev_mod_input: torch.Tensor | None = None
    prev_noise_pred: torch.Tensor | None = None
    samples: list[dict[str, Any]] = []

    for step_index, timestep in enumerate(timesteps):
        model_dtype = _component_dtype(pipe.transformer, pipe.dtype)
        timestep_batch = _batch_timestep(
            timestep,
            batch_size=latents.shape[0],
            device=latents.device,
            dtype=model_dtype,
        )
        model_bundle = HunyuanVideoDiTInputBundle(
            hidden_states=latents.to(dtype=model_dtype),
            timestep=timestep_batch,
            encoder_hidden_states=bundle.encoder_hidden_states.to(dtype=model_dtype),
            encoder_attention_mask=bundle.encoder_attention_mask,
            pooled_projections=bundle.pooled_projections.to(dtype=model_dtype),
            guidance=bundle.guidance.to(dtype=model_dtype),
        )
        mod_input = _teacache_mod_input(
            pipe.transformer,
            model_bundle,
            source="block0_modulated_input",
        ).detach()
        noise_pred = _first_tensor(pipe.transformer(model_bundle)).detach()
        _mark_step()
        if prev_mod_input is not None and prev_noise_pred is not None:
            samples.append(
                {
                    "bundle": str(bundle_path),
                    "split": split,
                    "step_index": int(step_index),
                    "mod_input_diff_norm": float(
                        torch.linalg.vector_norm(
                            mod_input.float().cpu() - prev_mod_input.float().cpu()
                        ).item()
                    ),
                    "noise_pred_diff_norm": float(
                        torch.linalg.vector_norm(
                            noise_pred.float().cpu() - prev_noise_pred.float().cpu()
                        ).item()
                    ),
                }
            )
        prev_mod_input = mod_input.float().cpu()
        prev_noise_pred = noise_pred.float().cpu()
        latents = pipe._scheduler_step(  # noqa: SLF001
            noise_pred,
            timestep,
            latents,
            int(timesteps.numel()),
        )
        _mark_step()
        print(
            f"[m9-t0] {bundle_path.name} {split} step "
            f"{step_index + 1}/{int(timesteps.numel())}",
            flush=True,
        )
    return meta, samples


def _collect_hunyuan_video_pairs(args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_hardware:
        raise RuntimeError("--collect-hv-pairs-out requires --allow-hardware")
    if args.foreground_ack != FOREGROUND_ACK:
        raise RuntimeError(f"--foreground-ack must equal {FOREGROUND_ACK!r}")
    if not args.source_dir or not args.compiled_dir:
        raise ValueError("--source-dir and --compiled-dir are required for HV collection")
    if not args.bundle:
        raise ValueError("--bundle is required for HV collection")

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    all_samples: list[dict[str, Any]] = []
    first_meta: dict[str, Any] | None = None

    # Load the application ONCE — reloading per bundle leaks Neuron RT resources
    # and crashes with c10::Error after ~5 iterations on HV N4 4d8s1r (cclog 75).
    first_bundle_meta, _ = _load_hv_bundle(Path((args.bundle or args.holdout_bundle)[0]))
    app = _load_hv_app(args, first_bundle_meta)
    _require_hv_teacache_mod_input(app)
    app.load(args.compiled_dir, skip_warmup=bool(args.skip_warmup))

    for bundle in args.bundle:
        meta, samples = _collect_hv_bundle_pairs(
            args,
            bundle_path=Path(bundle),
            split="train",
            app=app,
        )
        first_meta = first_meta or meta
        all_samples.extend(samples)
    for bundle in args.holdout_bundle:
        meta, samples = _collect_hv_bundle_pairs(
            args,
            bundle_path=Path(bundle),
            split="holdout",
            app=app,
        )
        first_meta = first_meta or meta
        all_samples.extend(samples)
    if first_meta is None:
        raise ValueError("no bundles were collected")
    if not all_samples:
        raise RuntimeError("collection produced no adjacent-step samples")

    num_steps = int(args.num_inference_steps or first_meta["num_inference_steps"])
    return {
        "schema": PAIRS_SCHEMA,
        "model": "hunyuan_video",
        "shape_label": _hv_shape_label(first_meta),
        "num_steps": num_steps,
        "mod_input_source": "block0_modulated_input",
        "hardware_measured": True,
        "started_at": started,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "samples": all_samples,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-json", default=None, help="Collected scalar pair JSON")
    parser.add_argument("--out", default=None, help="Output calibration JSON")
    parser.add_argument("--model", default=None, help="Override model label from pairs JSON")
    parser.add_argument("--shape-label", default=None, help="Override shape label")
    parser.add_argument("--num-steps", type=int, default=None, help="Override step count")
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--target-speedup", type=float, default=1.5)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument(
        "--skip-run-length",
        type=int,
        default=1,
        help="cclog 78: commit to N consecutive skips per probe decision (1=probe every step)",
    )
    parser.add_argument("--cooldown-steps", type=int, default=5)
    parser.add_argument(
        "--mod-input-source",
        default=None,
        choices=("block0_modulated_input", "hidden_states_proxy"),
        help=(
            "Override source recorded in calibration JSON. Real gates require "
            "block0_modulated_input; hidden_states_proxy is test-only."
        ),
    )
    parser.add_argument("--collect-hv-pairs-out", default=None)
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--compiled-dir", default=None)
    parser.add_argument("--bundle", action="append", default=[])
    parser.add_argument("--holdout-bundle", action="append", default=[])
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--allow-hardware", action="store_true")
    parser.add_argument("--foreground-ack", default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.collect_hv_pairs_out:
        pairs_doc = _collect_hunyuan_video_pairs(args)
        collect_out = Path(args.collect_hv_pairs_out)
        collect_out.parent.mkdir(parents=True, exist_ok=True)
        collect_out.write_text(json.dumps(pairs_doc, indent=2, sort_keys=True) + "\n")
        if args.out is None:
            print(json.dumps(pairs_doc, indent=2, sort_keys=True))
            return 0

    if args.pairs_json is None:
        if args.collect_hv_pairs_out is None:
            raise ValueError("--pairs-json is required unless --collect-hv-pairs-out is used")
        args.pairs_json = args.collect_hv_pairs_out
    if args.out is None:
        raise ValueError("--out is required when fitting a calibration JSON")

    pairs_path = Path(args.pairs_json)
    doc = _load_pairs(pairs_path)
    samples = doc["samples"]
    train_samples = [s for s in samples if s.get("split", "train") == "train"]
    holdout_samples = [s for s in samples if s.get("split") == "holdout"]
    if not train_samples:
        raise ValueError("pairs JSON has no train samples")

    train_x, train_y = _sample_xy(train_samples)
    coef = _fit_poly(train_x, train_y, args.degree)
    eval_samples = holdout_samples or train_samples
    eval_x, eval_y = _sample_xy(eval_samples)
    eval_pred = _poly_design(eval_x, args.degree).matmul(coef)
    fit_r2 = _r2_score(eval_y, eval_pred)
    train_pred = _poly_design(train_x, args.degree).matmul(coef)

    model = args.model or doc.get("model")
    shape_label = args.shape_label or doc.get("shape_label")
    num_steps = args.num_steps or doc.get("num_steps")
    mod_input_source = args.mod_input_source or doc.get(
        "mod_input_source",
        "block0_modulated_input",
    )
    if model is None or shape_label is None or num_steps is None:
        raise ValueError("model, shape_label, and num_steps are required")

    calibration = TeaCacheCalibration(
        model=str(model),
        shape_label=str(shape_label),
        num_steps=int(num_steps),
        poly_coef=tuple(float(item) for item in coef.tolist()),
        threshold=_threshold_for_target(
            train_pred,
            num_steps=int(num_steps),
            warmup_steps=args.warmup_steps,
            cooldown_steps=args.cooldown_steps,
            target_speedup=args.target_speedup,
        ),
        warmup_steps=int(args.warmup_steps),
        cooldown_steps=int(args.cooldown_steps),
        target_speedup=float(args.target_speedup),
        fit_r2=float(fit_r2),
        n_samples=len(samples),
        mod_input_source=str(mod_input_source),
        skip_run_length=int(args.skip_run_length),
    )
    out_doc = calibration.to_dict()
    out_doc["hardware_measured"] = bool(doc.get("hardware_measured", False))
    out_doc["source_pairs_json"] = str(pairs_path)
    out_doc["schema"] = CALIBRATION_SCHEMA
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_doc, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out_doc, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
