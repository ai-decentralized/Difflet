#!/usr/bin/env python3
"""Compile split HunyuanVideo 1.5 transformer-block segments on Trainium.

This is a capacity probe for the 480p block-closure path. The monolithic
HunyuanVideo 1.5 transformer block fails at 480p because attention lowering
blows past compiler limits. This script checks whether the same block can be
split into compileable segments around attention:

* ``pre-qkv``: AdaLN + Q/K/V projections + q/k norm + RoPE + context Q/K/V.
* ``post``: attention output projections + residuals + norm2 + FFNs.

The attention tile itself is covered by ``hunyuan15_attention_capacity_probe``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
    parser.add_argument("--cache-dir", default="/tmp/difflet_hunyuan15_block_split_capacity_cache")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Optional HunyuanVideo 1.5 model directory for real block weights.",
    )
    parser.add_argument("--transformer-subfolder", default="transformer")
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--part", choices=("pre-qkv", "post", "both"), default="both")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--text-seq-len", type=int, default=1000)
    parser.add_argument("--text-seq-len-2", type=int, default=256)
    parser.add_argument("--image-seq-len", type=int, default=729)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--qk-norm", default="rms_norm")
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--patch-size-t", type=int, default=1)
    parser.add_argument("--spatial-compression-ratio", type=int, default=16)
    parser.add_argument("--temporal-compression-ratio", type=int, default=4)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--force-clean", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument(
        "--run-parity",
        action="store_true",
        help="Load the compiled segment and compare against the same CPU module.",
    )
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument(
        "--compiler-args",
        default=(
            "--model-type=transformer -O1 --auto-cast=none "
            "--internal-hlo2tensorizer-options='--verify-hlo=true'"
        ),
    )
    return parser


def _transformer_dir(args: argparse.Namespace) -> Path:
    if args.model_dir is None:
        raise ValueError("--model-dir is required for real-weight parity")
    return Path(args.model_dir) / args.transformer_subfolder


def _load_block_state_dict_from_dir(
    transformer_dir: Path,
    block_index: int,
    *,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    from difflet.backends.trainium.core.modules.checkpoint import load_state_dict

    state_dict = load_state_dict(str(transformer_dir))
    prefix = f"transformer_blocks.{block_index}."
    block_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            tensor = value.to(dtype=dtype) if dtype is not None and torch.is_floating_point(value) else value
            block_state_dict[f"block.{key[len(prefix):]}"] = tensor
    if not block_state_dict:
        raise KeyError(f"no state_dict keys found for {prefix!r} under {transformer_dir}")
    return block_state_dict


def _latent_frames(num_frames: int, temporal_compression_ratio: int) -> int:
    return (int(num_frames) - 1) // int(temporal_compression_ratio) + 1


def _shape_meta(args: argparse.Namespace) -> dict[str, int]:
    latent_frames = _latent_frames(args.num_frames, args.temporal_compression_ratio)
    latent_height = args.height // args.spatial_compression_ratio
    latent_width = args.width // args.spatial_compression_ratio
    if latent_frames % args.patch_size_t != 0:
        raise ValueError("latent frame count must be divisible by patch_size_t")
    if latent_height % args.patch_size != 0 or latent_width % args.patch_size != 0:
        raise ValueError("latent height/width must be divisible by patch_size")
    latent_seq_len = (
        latent_frames
        // args.patch_size_t
        * latent_height
        // args.patch_size
        * latent_width
        // args.patch_size
    )
    context_seq_len = args.text_seq_len + args.text_seq_len_2 + args.image_seq_len
    inner_dim = args.heads * args.head_dim
    return {
        "latent_frames": latent_frames,
        "latent_height": latent_height,
        "latent_width": latent_width,
        "latent_seq_len": latent_seq_len,
        "context_seq_len": context_seq_len,
        "total_seq_len": latent_seq_len + context_seq_len,
        "inner_dim": inner_dim,
    }


class _Hunyuan15BlockPreQkvModule(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        from diffusers.models.transformers.transformer_hunyuan_video15 import (
            HunyuanVideo15TransformerBlock,
        )

        self.block = HunyuanVideo15TransformerBlock(
            args.heads,
            args.head_dim,
            mlp_ratio=args.mlp_ratio,
            qk_norm=args.qk_norm,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from diffusers.models.embeddings import apply_rotary_emb

        norm_hidden_states, *_ = self.block.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, *_ = self.block.norm1_context(encoder_hidden_states, emb=temb)
        attn = self.block.attn

        query = attn.to_q(norm_hidden_states).unflatten(2, (attn.heads, -1))
        key = attn.to_k(norm_hidden_states).unflatten(2, (attn.heads, -1))
        value = attn.to_v(norm_hidden_states).unflatten(2, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        query = apply_rotary_emb(query, (freqs_cos, freqs_sin), sequence_dim=1)
        key = apply_rotary_emb(key, (freqs_cos, freqs_sin), sequence_dim=1)

        encoder_query = attn.add_q_proj(norm_encoder_hidden_states).unflatten(2, (attn.heads, -1))
        encoder_key = attn.add_k_proj(norm_encoder_hidden_states).unflatten(2, (attn.heads, -1))
        encoder_value = attn.add_v_proj(norm_encoder_hidden_states).unflatten(2, (attn.heads, -1))
        if attn.norm_added_q is not None:
            encoder_query = attn.norm_added_q(encoder_query)
        if attn.norm_added_k is not None:
            encoder_key = attn.norm_added_k(encoder_key)

        return (
            torch.cat([query, encoder_query], dim=1),
            torch.cat([key, encoder_key], dim=1),
            torch.cat([value, encoder_value], dim=1),
        )


class _Hunyuan15BlockPostModule(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        from diffusers.models.transformers.transformer_hunyuan_video15 import (
            HunyuanVideo15TransformerBlock,
        )

        self.block = HunyuanVideo15TransformerBlock(
            args.heads,
            args.head_dim,
            mlp_ratio=args.mlp_ratio,
            qk_norm=args.qk_norm,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent_seq_len = hidden_states.shape[1]
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.block.norm1(
            hidden_states,
            emb=temb,
        )
        (
            norm_encoder_hidden_states,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = self.block.norm1_context(encoder_hidden_states, emb=temb)
        del norm_hidden_states, norm_encoder_hidden_states

        attention_states = attention_states.flatten(2, 3).to(hidden_states.dtype)
        attn_output = attention_states[:, :latent_seq_len]
        context_attn_output = attention_states[:, latent_seq_len:]
        attn_output = self.block.attn.to_out[0](attn_output)
        attn_output = self.block.attn.to_out[1](attn_output)
        context_attn_output = self.block.attn.to_add_out(context_attn_output)

        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)

        norm_hidden_states = self.block.norm2(hidden_states)
        norm_encoder_hidden_states = self.block.norm2_context(encoder_hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        )

        ff_output = self.block.ff(norm_hidden_states)
        context_ff_output = self.block.ff_context(norm_encoder_hidden_states)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        return hidden_states, encoder_hidden_states


def _make_cpu_module(
    args: argparse.Namespace,
    part: str,
    *,
    dtype: torch.dtype,
) -> nn.Module:
    module = (
        _Hunyuan15BlockPreQkvModule(args)
        if part == "pre-qkv"
        else _Hunyuan15BlockPostModule(args)
    )
    if args.model_dir is not None:
        module.load_state_dict(
            _load_block_state_dict_from_dir(
                _transformer_dir(args),
                args.block_index,
                dtype=dtype,
            ),
            strict=True,
        )
    return module.to(dtype=dtype).eval()


def _make_part_inputs(
    part: str,
    meta: dict[str, int],
    args: argparse.Namespace,
    *,
    dtype: torch.dtype,
    seed: int = 0,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden_shape = [1, meta["latent_seq_len"], meta["inner_dim"]]
    context_shape = [1, meta["context_seq_len"], meta["inner_dim"]]
    temb_shape = [1, meta["inner_dim"]]
    if part == "pre-qkv":
        return (
            torch.randn(hidden_shape, generator=generator, dtype=dtype),
            torch.randn(context_shape, generator=generator, dtype=dtype),
            torch.randn(temb_shape, generator=generator, dtype=dtype),
            torch.randn([meta["latent_seq_len"], args.head_dim], generator=generator, dtype=dtype),
            torch.randn([meta["latent_seq_len"], args.head_dim], generator=generator, dtype=dtype),
        )
    return (
        torch.randn(hidden_shape, generator=generator, dtype=dtype),
        torch.randn(context_shape, generator=generator, dtype=dtype),
        torch.randn(temb_shape, generator=generator, dtype=dtype),
        torch.randn(
            [1, meta["total_seq_len"], args.heads, args.head_dim],
            generator=generator,
            dtype=dtype,
        ),
    )


def _flatten_tensors(value: object) -> list[torch.Tensor]:
    if torch.is_tensor(value):
        return [value.detach().cpu()]
    if isinstance(value, (tuple, list)):
        tensors = []
        for item in value:
            tensors.extend(_flatten_tensors(item))
        return tensors
    raise TypeError(f"expected tensor output, got {type(value)!r}")


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().reshape(-1),
        b.float().reshape(-1),
        dim=0,
    ).item()


def build_block_split_application(
    args: argparse.Namespace,
    part: str,
    meta: dict[str, int],
) -> tuple[object, Path]:
    os.environ.setdefault("DIFFLET_BACKEND", "trainium")
    os.environ["LOCAL_WORLD_SIZE"] = str(args.tp_degree)

    from difflet.backends.trainium.core.application_base import NeuronApplicationBase
    from difflet.backends.trainium.core.config import InferenceConfig, NeuronConfig
    from difflet.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper

    class BlockSplitConfig(InferenceConfig):
        def get_required_attributes(self):
            return [
                "part",
                "latent_seq_len",
                "context_seq_len",
                "total_seq_len",
                "inner_dim",
                "heads",
                "head_dim",
            ]

    class BlockSplitWrapper(ModelWrapper):
        def input_generator(self):
            dtype = self.config.neuron_config.torch_dtype
            hidden_shape = [1, self.config.latent_seq_len, self.config.inner_dim]
            context_shape = [1, self.config.context_seq_len, self.config.inner_dim]
            temb_shape = [1, self.config.inner_dim]
            if self.config.part == "pre-qkv":
                return [
                    (
                        torch.randn(hidden_shape, dtype=dtype),
                        torch.randn(context_shape, dtype=dtype),
                        torch.randn(temb_shape, dtype=dtype),
                        torch.randn([self.config.latent_seq_len, self.config.head_dim], dtype=dtype),
                        torch.randn([self.config.latent_seq_len, self.config.head_dim], dtype=dtype),
                    )
                ]
            return [
                (
                    torch.randn(hidden_shape, dtype=dtype),
                    torch.randn(context_shape, dtype=dtype),
                    torch.randn(temb_shape, dtype=dtype),
                    torch.randn(
                        [1, self.config.total_seq_len, self.config.heads, self.config.head_dim],
                        dtype=dtype,
                    ),
                )
            ]

        def get_model_instance(self):
            def _create_model():
                module = (
                    _Hunyuan15BlockPreQkvModule(args)
                    if self.config.part == "pre-qkv"
                    else _Hunyuan15BlockPostModule(args)
                )
                module = module.to(dtype=self.config.neuron_config.torch_dtype)
                module.eval()
                return module

            return BaseModelInstance(module_cls=_create_model, input_output_aliases={})

        def forward(self, *model_inputs):
            if self.model is None:
                raise RuntimeError("Forward called before load. Run load() first.")
            return self._forward(*model_inputs)

    class BlockSplitApplication(NeuronApplicationBase):
        _model_cls = object

        def __init__(self, *app_args, **app_kwargs):
            super().__init__(*app_args, **app_kwargs)
            self.model = BlockSplitWrapper(
                config=self.config,
                model_cls=self._model_cls,
                tag=f"HunyuanVideo15BlockSplit_{part}",
                compiler_args=args.compiler_args,
                priority_model_idx=0,
            )
            self.models.append(self.model)

        @classmethod
        def get_config_cls(cls):
            return BlockSplitConfig

        @classmethod
        def get_state_dict(cls, model_name_or_path: str, config: InferenceConfig) -> dict:
            del model_name_or_path
            source_model_dir = getattr(config, "source_model_dir", None)
            if not source_model_dir:
                return {}
            return _load_block_state_dict_from_dir(
                Path(source_model_dir) / getattr(config, "transformer_subfolder", "transformer"),
                int(getattr(config, "block_index", 0)),
                dtype=config.neuron_config.torch_dtype,
            )

        @staticmethod
        def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
            del config
            return state_dict

        @staticmethod
        def update_state_dict_for_tied_weights(state_dict):
            pass

        def forward(self, *model_inputs, **kwargs):
            return self.models[0](*model_inputs, **kwargs)

    part_dir = Path(args.cache_dir) / part
    if args.force_clean and part_dir.exists():
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)
    config = BlockSplitConfig(
        neuron_config=NeuronConfig(
            batch_size=1,
            tp_degree=args.tp_degree,
            world_size=args.tp_degree,
            torch_dtype=torch.bfloat16,
            skip_sharding=True,
        ),
        part=part,
        latent_seq_len=meta["latent_seq_len"],
        context_seq_len=meta["context_seq_len"],
        total_seq_len=meta["total_seq_len"],
        inner_dim=meta["inner_dim"],
        heads=args.heads,
        head_dim=args.head_dim,
        source_model_dir=args.model_dir,
        transformer_subfolder=args.transformer_subfolder,
        block_index=args.block_index,
    )
    app = BlockSplitApplication(model_path=str(part_dir), config=config)
    return app, part_dir


def _compile_part(args: argparse.Namespace, part: str, meta: dict[str, int]) -> dict[str, object]:
    app, part_dir = build_block_split_application(args, part, meta)
    t0 = time.perf_counter()
    if not args.skip_compile:
        app.compile(str(part_dir), debug=False)
        elapsed = time.perf_counter() - t0
    else:
        elapsed = None
    model_pt = part_dir / "model.pt"
    metrics = {
        "part": part,
        "status": "pass",
        "compiled_path": str(part_dir),
        "compile_elapsed_s": elapsed,
        "model_pt_bytes": model_pt.stat().st_size if model_pt.exists() else None,
    }
    if args.run_parity:
        app.load(str(part_dir), skip_warmup=True)
        inputs = _make_part_inputs(part, meta, args, dtype=torch.bfloat16)
        t1 = time.perf_counter()
        with torch.no_grad():
            trainium_outputs = _flatten_tensors(app(*inputs))
        trainium_elapsed = time.perf_counter() - t1
        cpu_module = _make_cpu_module(args, part, dtype=torch.bfloat16)
        t2 = time.perf_counter()
        with torch.no_grad():
            reference_outputs = _flatten_tensors(cpu_module(*inputs))
        reference_elapsed = time.perf_counter() - t2
        output_metrics = []
        for index, (actual, expected) in enumerate(zip(trainium_outputs, reference_outputs)):
            diff = (actual.float() - expected.float()).abs()
            output_metrics.append(
                {
                    "index": index,
                    "shape": list(actual.shape),
                    "cosine": _cosine(actual, expected),
                    "max_abs": float(diff.max()),
                    "mean_abs": float(diff.mean()),
                }
            )
        metrics.update(
            {
                "trainium_forward_elapsed_s": trainium_elapsed,
                "reference_elapsed_s": reference_elapsed,
                "outputs": output_metrics,
            }
        )
    return metrics


def main() -> int:
    args = build_parser().parse_args()
    meta = _shape_meta(args)
    parts = ["pre-qkv", "post"] if args.part == "both" else [args.part]
    metrics: dict[str, object] = {
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "text_seq_len": args.text_seq_len,
        "text_seq_len_2": args.text_seq_len_2,
        "image_seq_len": args.image_seq_len,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "tp_degree": args.tp_degree,
        "compiler_args": args.compiler_args,
        **meta,
        "parts": [],
    }
    status = 0
    for part in parts:
        try:
            part_metrics = _compile_part(args, part, meta)
        except Exception as exc:
            part_metrics = {
                "part": part,
                "status": "fail",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-20:],
            }
            status = 1
        metrics["parts"].append(part_metrics)
        if status:
            break

    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    if args.metrics_out:
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[hunyuan15-block-split] metrics -> {metrics_path}", flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
