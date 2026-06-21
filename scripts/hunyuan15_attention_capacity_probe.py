#!/usr/bin/env python3
"""Compile a standalone HunyuanVideo 1.5-size attention graph on Trainium.

This is a capacity probe, not a parity gate. It separates the production 480p
attention operator from the rest of the transformer block so we can tell
whether segmented block compilation is enough, or whether attention itself
needs a different kernel/lowering.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

NEURON_VENV = Path("/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference")
NEURON_PYTHON = NEURON_VENV / "bin" / "python"
ROOT = Path(__file__).resolve().parents[1]


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


ensure_runtime_python()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("sdpa", "nki", "manual-stats", "manual-stats-masked"),
        default="sdpa",
    )
    parser.add_argument("--cache-dir", default="/tmp/difflet_hunyuan15_attention_capacity_cache")
    parser.add_argument("--query-len", type=int, default=31 * 30 * 53 + 10)
    parser.add_argument("--key-len", type=int, default=31 * 30 * 53 + 10)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--layout",
        choices=("bhsd", "bshd"),
        default="bshd",
        help="Input/output attention layout. bshd matches diffusers dispatch_attention_fn.",
    )
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--force-clean", action="store_true")
    parser.add_argument(
        "--run-stream-merge-check",
        action="store_true",
        help="After compile/load, compare tiled manual-stats attention against CPU SDPA.",
    )
    parser.add_argument("--stream-query-tiles", type=int, default=1)
    parser.add_argument("--stream-key-tiles", type=int, default=2)
    parser.add_argument(
        "--skip-stream-reference",
        action="store_true",
        help="Run tiled attention without constructing a full CPU SDPA reference.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument(
        "--compiler-args",
        default=(
            "--model-type=generic -O1 --auto-cast=none "
            "--internal-hlo2tensorizer-options='--verify-hlo=true'"
        ),
    )
    return parser


class _AttentionCapacityModule(nn.Module):
    def __init__(self, backend: str, layout: str) -> None:
        super().__init__()
        self.backend = backend
        self.layout = layout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_valid: torch.Tensor | None = None,
        key_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.layout == "bshd":
            query = query.permute(0, 2, 1, 3)
            key = key.permute(0, 2, 1, 3)
            value = value.permute(0, 2, 1, 3)

        if self.backend in {"manual-stats", "manual-stats-masked"}:
            scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
            scores = scores * (1 / math.sqrt(query.shape[-1]))
            if self.backend == "manual-stats-masked":
                if query_valid is None or key_valid is None:
                    raise ValueError("manual-stats-masked requires query_valid and key_valid")
                key_valid = key_valid.to(dtype=torch.bool).view(key.shape[0], 1, 1, key.shape[2])
                scores = torch.where(
                    key_valid,
                    scores,
                    torch.full_like(scores, -1.0e9),
                )
            max_score = scores.max(dim=-1).values
            weights = torch.exp(scores - max_score.unsqueeze(-1))
            denom = weights.sum(dim=-1)
            numerator = torch.matmul(weights.to(value.dtype), value)
            if self.backend == "manual-stats-masked":
                query_valid = query_valid.to(dtype=torch.bool).view(
                    query.shape[0],
                    1,
                    query.shape[2],
                )
                numerator = torch.where(query_valid.unsqueeze(-1), numerator, torch.zeros_like(numerator))
                max_score = torch.where(query_valid, max_score, torch.full_like(max_score, -float("inf")))
                denom = torch.where(query_valid, denom, torch.zeros_like(denom))
            if self.layout == "bshd":
                numerator = numerator.permute(0, 2, 1, 3)
                max_score = max_score.permute(0, 2, 1)
                denom = denom.permute(0, 2, 1)
            return numerator, max_score, denom

        if self.backend == "sdpa":
            hidden_states = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
            )
            if self.layout == "bshd":
                hidden_states = hidden_states.permute(0, 2, 1, 3)
            return hidden_states

        from difflet.ops import attention

        batch_size, heads, query_len, head_dim = query.shape
        key_len = key.shape[2]
        value_len = value.shape[2]
        query = query.reshape(batch_size * heads, query_len, head_dim)
        key = key.reshape(batch_size * heads, key_len, head_dim)
        value = value.reshape(batch_size * heads, value_len, head_dim)
        hidden_states = attention(
            query,
            key,
            value,
            scale=1 / math.sqrt(head_dim),
            causal=False,
            tp_q=True,
            tp_k=True,
            tp_out=False,
        )
        hidden_states = hidden_states.reshape(batch_size, heads, query_len, head_dim)
        if self.layout == "bshd":
            hidden_states = hidden_states.permute(0, 2, 1, 3)
        return hidden_states


def merge_manual_stats_tiles(
    tile_numerators: list[torch.Tensor],
    tile_max_scores: list[torch.Tensor],
    tile_denoms: list[torch.Tensor],
) -> torch.Tensor:
    """Merge streaming-softmax tile statistics across K/V chunks."""

    if not tile_numerators:
        raise ValueError("at least one attention tile is required")
    stacked_max = torch.stack(tile_max_scores, dim=0)
    global_max = stacked_max.max(dim=0).values
    numerator_accum = torch.zeros_like(tile_numerators[0])
    denom_accum = torch.zeros_like(tile_denoms[0])
    finite_global = torch.isfinite(global_max)
    for numerator, max_score, denom in zip(tile_numerators, tile_max_scores, tile_denoms):
        scale = torch.where(
            finite_global,
            torch.exp(max_score - global_max),
            torch.zeros_like(max_score),
        )
        numerator_accum = numerator_accum + numerator * scale.unsqueeze(-1)
        denom_accum = denom_accum + denom * scale
    return torch.where(
        denom_accum.unsqueeze(-1) > 0,
        numerator_accum / denom_accum.unsqueeze(-1),
        torch.zeros_like(numerator_accum),
    )


def run_streaming_manual_stats_attention(
    app: object,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_tile_size: int,
    key_tile_size: int,
    collect_output: bool = True,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, dict[str, object]]:
    """Run a compiled manual-stats tile module over full BSHD Q/K/V tensors."""

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must be BSHD tensors")
    if key.shape != value.shape:
        raise ValueError(f"key/value shape mismatch: {tuple(key.shape)} vs {tuple(value.shape)}")
    if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError(f"incompatible query/key shapes: {tuple(query.shape)} vs {tuple(key.shape)}")
    if query.shape[1] % query_tile_size != 0:
        raise ValueError("query sequence length must be divisible by query_tile_size")
    if key.shape[1] % key_tile_size != 0:
        raise ValueError("key sequence length must be divisible by key_tile_size")
    valid_mask_bool = None
    if valid_mask is not None:
        if tuple(valid_mask.shape) != (query.shape[0], key.shape[1]):
            raise ValueError(
                f"valid_mask must have shape {(query.shape[0], key.shape[1])}, "
                f"got {tuple(valid_mask.shape)}"
            )
        valid_mask_bool = valid_mask.to(dtype=torch.bool)
        valid_mask = valid_mask.to(dtype=torch.int64)

    forward_elapsed = 0.0
    output_chunks = []
    checksum = 0.0
    output_absmax = 0.0
    stream_query_tiles = query.shape[1] // query_tile_size
    stream_key_tiles = key.shape[1] // key_tile_size
    for query_index in range(stream_query_tiles):
        query_start = query_index * query_tile_size
        query_end = query_start + query_tile_size
        query_chunk = query[:, query_start:query_end].contiguous()
        tile_numerators = []
        tile_max_scores = []
        tile_denoms = []
        for key_index in range(stream_key_tiles):
            key_start = key_index * key_tile_size
            key_end = key_start + key_tile_size
            t0 = time.perf_counter()
            with torch.no_grad():
                if valid_mask is None:
                    numerator, max_score, denom = app(
                        query_chunk,
                        key[:, key_start:key_end].contiguous(),
                        value[:, key_start:key_end].contiguous(),
                    )
                else:
                    query_valid = valid_mask[:, query_start:query_end].contiguous()
                    key_valid = valid_mask[:, key_start:key_end].contiguous()
                    numerator, max_score, denom = app(
                        query_chunk,
                        key[:, key_start:key_end].contiguous(),
                        value[:, key_start:key_end].contiguous(),
                        query_valid,
                        key_valid,
                    )
            forward_elapsed += time.perf_counter() - t0
            numerator = numerator.detach().cpu().float()
            max_score = max_score.detach().cpu().float()
            denom = denom.detach().cpu().float()
            if valid_mask_bool is not None and not bool(
                valid_mask_bool[:, key_start:key_end].any()
            ):
                numerator.zero_()
                max_score.fill_(-float("inf"))
                denom.zero_()
            tile_numerators.append(numerator)
            tile_max_scores.append(max_score)
            tile_denoms.append(denom)

        merged = merge_manual_stats_tiles(tile_numerators, tile_max_scores, tile_denoms)
        if valid_mask_bool is not None:
            merged = torch.where(
                valid_mask_bool[:, query_start:query_end].unsqueeze(-1).unsqueeze(-1).cpu(),
                merged,
                torch.zeros_like(merged),
            )
        checksum += float(merged.sum())
        output_absmax = max(output_absmax, float(merged.abs().max()))
        if collect_output:
            output_chunks.append(merged)

    metrics = {
        "stream_query_total": int(query.shape[1]),
        "stream_key_total": int(key.shape[1]),
        "stream_query_tiles": int(stream_query_tiles),
        "stream_key_tiles": int(stream_key_tiles),
        "stream_tile_calls": int(stream_query_tiles * stream_key_tiles),
        "stream_forward_elapsed_s": forward_elapsed,
        "stream_output_checksum": checksum,
        "stream_output_absmax": output_absmax,
    }
    output = torch.cat(output_chunks, dim=1) if collect_output else None
    return output, metrics


def build_attention_capacity_application(args: argparse.Namespace) -> tuple[object, Path]:
    os.environ.setdefault("DIFFLET_BACKEND", "trainium")
    os.environ["LOCAL_WORLD_SIZE"] = str(args.tp_degree)

    from difflet.backends.trainium.core.application_base import NeuronApplicationBase
    from difflet.backends.trainium.core.config import InferenceConfig, NeuronConfig
    from difflet.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper

    class AttentionCapacityConfig(InferenceConfig):
        def get_required_attributes(self):
            return ["query_len", "key_len", "heads", "head_dim", "backend", "layout"]

    class AttentionCapacityWrapper(ModelWrapper):
        def input_generator(self):
            dtype = self.config.neuron_config.torch_dtype
            if self.config.layout == "bshd":
                shape_q = [1, self.config.query_len, self.config.heads, self.config.head_dim]
                shape_kv = [1, self.config.key_len, self.config.heads, self.config.head_dim]
            else:
                shape_q = [1, self.config.heads, self.config.query_len, self.config.head_dim]
                shape_kv = [1, self.config.heads, self.config.key_len, self.config.head_dim]
            inputs = (
                torch.randn(shape_q, dtype=dtype),
                torch.randn(shape_kv, dtype=dtype),
                torch.randn(shape_kv, dtype=dtype),
            )
            if self.config.backend == "manual-stats-masked":
                inputs = (
                    *inputs,
                    torch.ones([1, self.config.query_len], dtype=torch.int64),
                    torch.ones([1, self.config.key_len], dtype=torch.int64),
                )
            return [inputs]

        def get_model_instance(self):
            def _create_model():
                model = _AttentionCapacityModule(self.config.backend, self.config.layout)
                model = model.to(dtype=self.config.neuron_config.torch_dtype)
                model.eval()
                return model

            return BaseModelInstance(module_cls=_create_model, input_output_aliases={})

        def forward(self, *model_inputs):
            if self.model is None:
                raise RuntimeError("Forward called before load. Run load() first.")
            return self._forward(*model_inputs)

    class AttentionCapacityApplication(NeuronApplicationBase):
        _model_cls = object

        def __init__(self, *app_args, **app_kwargs):
            super().__init__(*app_args, **app_kwargs)
            self.model = AttentionCapacityWrapper(
                config=self.config,
                model_cls=self._model_cls,
                tag="HunyuanVideo15AttentionCapacity",
                compiler_args=args.compiler_args,
                priority_model_idx=0,
            )
            self.models.append(self.model)

        @classmethod
        def get_config_cls(cls):
            return AttentionCapacityConfig

        @classmethod
        def get_state_dict(cls, model_name_or_path: str, config: InferenceConfig) -> dict:
            del model_name_or_path, config
            return {}

        @staticmethod
        def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
            del config
            return state_dict

        @staticmethod
        def update_state_dict_for_tied_weights(state_dict):
            pass

        def forward(self, *model_inputs, **kwargs):
            return self.models[0](*model_inputs, **kwargs)

    cache_dir = Path(args.cache_dir)
    if args.force_clean and cache_dir.exists():
        import shutil

        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    config = AttentionCapacityConfig(
        neuron_config=NeuronConfig(
            batch_size=1,
            tp_degree=args.tp_degree,
            world_size=args.tp_degree,
            torch_dtype=torch.bfloat16,
            skip_sharding=True,
        ),
        query_len=args.query_len,
        key_len=args.key_len,
        heads=args.heads,
        head_dim=args.head_dim,
        backend=args.backend,
        layout=args.layout,
    )
    app = AttentionCapacityApplication(model_path=str(cache_dir), config=config)
    return app, cache_dir


def _compile(args: argparse.Namespace) -> dict[str, object]:
    app, cache_dir = build_attention_capacity_application(args)
    t0 = time.perf_counter()
    app.compile(str(cache_dir), debug=False)
    elapsed = time.perf_counter() - t0
    model_pt = cache_dir / "model.pt"
    metrics = {
        "backend": args.backend,
        "layout": args.layout,
        "query_len": args.query_len,
        "key_len": args.key_len,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "tp_degree": args.tp_degree,
        "compiler_args": args.compiler_args,
        "compile_elapsed_s": elapsed,
        "compiled_path": str(cache_dir),
        "model_pt_bytes": model_pt.stat().st_size if model_pt.exists() else None,
    }
    if args.run_stream_merge_check:
        if args.backend != "manual-stats":
            raise ValueError("--run-stream-merge-check requires --backend manual-stats")
        if args.layout != "bshd":
            raise ValueError("--run-stream-merge-check currently expects --layout bshd")
        metrics.update(_run_stream_merge_check(args, app, cache_dir))
    return metrics


def _run_stream_merge_check(
    args: argparse.Namespace,
    app: object,
    cache_dir: Path,
) -> dict[str, object]:
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    dtype = torch.bfloat16
    query_total = args.query_len * args.stream_query_tiles
    key_total = args.key_len * args.stream_key_tiles
    query = torch.randn([1, query_total, args.heads, args.head_dim], generator=generator, dtype=dtype)
    key = torch.randn([1, key_total, args.heads, args.head_dim], generator=generator, dtype=dtype)
    value = torch.randn([1, key_total, args.heads, args.head_dim], generator=generator, dtype=dtype)

    app.load(str(cache_dir), skip_warmup=True)
    merged, metrics = run_streaming_manual_stats_attention(
        app,
        query,
        key,
        value,
        query_tile_size=args.query_len,
        key_tile_size=args.key_len,
        collect_output=not args.skip_stream_reference,
    )
    if args.skip_stream_reference:
        return metrics

    reference = torch.nn.functional.scaled_dot_product_attention(
        query.permute(0, 2, 1, 3).float(),
        key.permute(0, 2, 1, 3).float(),
        value.permute(0, 2, 1, 3).float(),
        dropout_p=0.0,
        is_causal=False,
    ).permute(0, 2, 1, 3)
    diff = (merged - reference).abs()
    cosine = torch.nn.functional.cosine_similarity(
        merged.reshape(-1),
        reference.reshape(-1),
        dim=0,
    ).item()
    metrics.update(
        {
            "stream_cosine": cosine,
            "stream_max_abs": float(diff.max()),
            "stream_mean_abs": float(diff.mean()),
        }
    )
    return metrics


def main() -> int:
    args = build_parser().parse_args()
    try:
        metrics = _compile(args)
        status = 0
    except Exception as exc:
        metrics = {
            "backend": args.backend,
            "layout": args.layout,
            "query_len": args.query_len,
            "key_len": args.key_len,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "tp_degree": args.tp_degree,
            "compiler_args": args.compiler_args,
            "compiled_path": args.cache_dir,
            "status": "fail",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-20:],
        }
        status = 1
    else:
        metrics["status"] = "pass"
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    if args.metrics_out:
        path = Path(args.metrics_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[hunyuan15-attn-capacity] metrics -> {path}", flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
