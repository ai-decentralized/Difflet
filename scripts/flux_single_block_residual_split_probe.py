#!/usr/bin/env python3
"""Compile and exercise a component-aligned FLUX single-block cache split.

This is an engineering/identifiability probe, not a serving benchmark.  It
builds two static Trainium graphs around one real FLUX single-stream block:

* ``anchor`` computes attention and MLP, returning the globally reduced MLP
  branch as a cache payload plus a small sketch of the MLP input;
* ``hybrid`` recomputes attention but consumes a supplied MLP payload.

The synthetic perturbation sweep tests whether cheap MLP-input sketch drift is
monotone with the true MLP-output cache error.  A later trajectory experiment
must repeat the test on real denoising hidden states before any quality claim.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, List

NEURON_VENV = Path("/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference")
NEURON_PYTHON = NEURON_VENV / "bin" / "python"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
DEFAULT_MODEL_DIR = (
    Path("/home/ubuntu/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev")
    / "snapshots"
    / DEFAULT_REVISION
)
HARDWARE_ACK = "I am running the FLUX single-block residual split probe"


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

from difflet.models.flux.modeling_flux import (  # noqa: E402
    NeuronFluxSingleTransformerBlock,
)
from difflet.ops import reduce_from_tensor_model_parallel_region  # noqa: E402
from difflet.pipeline.cache.component_signal import (  # noqa: E402
    relative_component_drift,
    residual_input_samples,
    residual_input_sketch,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument(
        "--cache-dir",
        default="/home/ubuntu/difflet-artifacts/flux-single-block-residual-split-20260804/compiled",
    )
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--text-token-count", type=int, default=512)
    parser.add_argument("--text-regions", type=int, default=8)
    parser.add_argument("--image-region-rows", type=int, default=4)
    parser.add_argument("--image-region-columns", type=int, default=4)
    parser.add_argument("--channel-groups", type=int, default=32)
    parser.add_argument(
        "--perturbation-scales",
        default="0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2",
    )
    parser.add_argument("--perturbation-seeds", type=int, default=3)
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument(
        "--execution-mode",
        choices=("bucketed", "bucketed-bank", "separate"),
        default="bucketed-bank",
        help=(
            "Use a bucketed model with one moments signal, a four-signal stability bank, "
            "or the original two-app diagnostic."
        ),
    )
    parser.add_argument("--metrics-out")
    parser.add_argument("--allow-hardware", action="store_true")
    parser.add_argument("--foreground-ack")
    return parser


def _strict_scales(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(not math.isfinite(item) or item <= 0.0 for item in result):
        raise ValueError("perturbation scales must be finite positive numbers")
    if tuple(sorted(set(result))) != result:
        raise ValueError("perturbation scales must be unique and increasing")
    return result


def _model_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "transformer" / "config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_single_block_state_dict(
    model_dir: Path,
    block_index: int,
    *,
    inner_dim: int,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    transformer_dir = model_dir / "transformer"
    index_path = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    prefix = f"single_transformer_blocks.{block_index}."
    weight_map = {
        key: value
        for key, value in index["weight_map"].items()
        if key.startswith(prefix)
    }
    if not weight_map:
        raise KeyError(f"no FLUX weights found for {prefix!r}")
    shards: dict[str, list[str]] = {}
    for key, filename in weight_map.items():
        shards.setdefault(str(filename), []).append(key)
    state: dict[str, torch.Tensor] = {}
    for filename, keys in shards.items():
        with safe_open(transformer_dir / filename, framework="pt", device="cpu") as handle:
            for key in keys:
                value = handle.get_tensor(key)
                if value.is_floating_point():
                    value = value.to(dtype=dtype)
                state[f"block.{key[len(prefix):]}"] = value.contiguous()

    combined_weight = state.pop("block.proj_out.weight")
    combined_bias = state.pop("block.proj_out.bias")
    state["block.proj_out_attn.weight"] = combined_weight[:, :inner_dim].contiguous()
    state["block.proj_out_attn.bias"] = combined_bias.contiguous()
    state["block.proj_out_mlp.weight"] = combined_weight[:, inner_dim:].contiguous()
    return state


class _FluxSingleBlockSplit(nn.Module):
    def __init__(self, config: Any, *, part: str) -> None:
        super().__init__()
        self.part = part
        self.text_token_count = int(config.text_token_count)
        self.image_height = int(config.image_height)
        self.image_width = int(config.image_width)
        self.text_regions = int(config.text_regions)
        self.image_region_rows = int(config.image_region_rows)
        self.image_region_columns = int(config.image_region_columns)
        self.channel_groups = int(config.channel_groups)
        self.block = NeuronFluxSingleTransformerBlock(
            dim=int(config.inner_dim),
            num_attention_heads=int(config.num_attention_heads),
            attention_head_dim=int(config.attention_head_dim),
            reduce_dtype=config.neuron_config.torch_dtype,
            mlp_ratio=float(config.mlp_ratio),
            context_parallel_enabled=False,
            sp_enabled=False,
        )

    def _prefix(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized, gate = self.block.norm(hidden_states, emb=temb)
        attention = self.block.attn(
            hidden_states=normalized,
            image_rotary_emb=image_rotary_emb,
        )
        out_attn, bias = self.block.proj_out_attn(attention)
        attn_global = reduce_from_tensor_model_parallel_region(
            out_attn,
            process_group=self.block.proj_out_attn.tensor_parallel_group,
        )
        sketch = residual_input_sketch(
            normalized,
            text_token_count=self.text_token_count,
            image_height=self.image_height,
            image_width=self.image_width,
            text_regions=self.text_regions,
            image_region_rows=self.image_region_rows,
            image_region_columns=self.image_region_columns,
            channel_groups=self.channel_groups,
        )
        return normalized, gate.unsqueeze(1), attn_global, bias, sketch

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        cached_mlp_global: torch.Tensor | None = None,
    ):
        residual = hidden_states
        normalized, gate, attn_global, bias, sketch = self._prefix(
            hidden_states, temb, image_rotary_emb
        )
        if self.part == "anchor":
            mlp_hidden = self.block.act_mlp(self.block.proj_mlp(normalized))
            out_mlp = self.block.proj_out_mlp(mlp_hidden)
            mlp_global = reduce_from_tensor_model_parallel_region(
                out_mlp,
                process_group=self.block.proj_out_mlp.tensor_parallel_group,
            )
            output = residual + gate * (attn_global + mlp_global + bias)
            return output, mlp_global, sketch
        if cached_mlp_global is None:
            raise ValueError("hybrid block requires cached_mlp_global")
        output = residual + gate * (attn_global + cached_mlp_global + bias)
        return output, sketch


def _bucket_kernel():
    @torch.jit.script
    def select(inputs: List[torch.Tensor]):
        bucket_idx = 0
        if inputs[-1].shape[0] > 1:
            bucket_idx = 1
        return inputs, torch.tensor([bucket_idx], dtype=torch.int32)

    return select


def _relative_rms_bfloat16(current: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    numerator = (current - anchor).square().mean().sqrt()
    denominator = anchor.square().mean().sqrt().clamp_min(1e-6)
    return (numerator / denominator).reshape(1)


class _FluxSingleBlockBucketed(_FluxSingleBlockSplit):
    """Anchor/hybrid buckets sharing MLP payload and input sketch in HBM."""

    def __init__(self, config: Any) -> None:
        super().__init__(config, part="anchor")
        sequence = self.text_token_count + self.image_height * self.image_width
        inner_dim = int(config.inner_dim)
        sketch_regions = self.text_regions + self.image_region_rows * self.image_region_columns
        self.cached_mlp_global = nn.Parameter(
            torch.zeros(1, sequence, inner_dim), requires_grad=False
        )
        self.cached_input_sketch = nn.Parameter(
            torch.zeros(1, sketch_regions, self.channel_groups, 2), requires_grad=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        route: torch.Tensor,
    ):
        residual = hidden_states
        normalized, gate, attn_global, bias, sketch = self._prefix(
            hidden_states, temb, image_rotary_emb
        )
        online_signal = relative_component_drift(sketch, self.cached_input_sketch)
        if self.part == "anchor":
            mlp_hidden = self.block.act_mlp(self.block.proj_mlp(normalized))
            out_mlp = self.block.proj_out_mlp(mlp_hidden)
            mlp_global = reduce_from_tensor_model_parallel_region(
                out_mlp,
                process_group=self.block.proj_out_mlp.tensor_parallel_group,
            )
            teacher_error = _relative_rms_bfloat16(
                mlp_global, self.cached_mlp_global
            ).float()
            output = residual + gate * (attn_global + mlp_global + bias)
            marker = route[:1].to(output.dtype) * 0 + 10
            return output, online_signal, teacher_error, marker, mlp_global, sketch
        output = residual + gate * (attn_global + self.cached_mlp_global + bias)
        teacher_error = online_signal * 0
        marker = route[:1].to(output.dtype) * 0 + 20
        return (
            output,
            online_signal,
            teacher_error,
            marker,
            self.cached_mlp_global,
            self.cached_input_sketch,
        )


SIGNAL_BANK_NAMES = ("raw_input_samples", "normalized_input_samples")


class _FluxSingleBlockBucketedSignalBank(_FluxSingleBlockSplit):
    """Exercise raw and normalized representative block-input samples."""

    def __init__(self, config: Any) -> None:
        super().__init__(config, part="anchor")
        sequence = self.text_token_count + self.image_height * self.image_width
        inner_dim = int(config.inner_dim)
        sketch_regions = self.text_regions + self.image_region_rows * self.image_region_columns
        samples_shape = (1, sketch_regions, self.channel_groups)
        self.cached_mlp_global = nn.Parameter(
            torch.zeros(1, sequence, inner_dim), requires_grad=False
        )
        self.cached_raw_samples = nn.Parameter(
            torch.zeros(samples_shape), requires_grad=False
        )
        self.cached_normalized_samples = nn.Parameter(
            torch.zeros(samples_shape), requires_grad=False
        )

    def _raw_samples(self, hidden_states: torch.Tensor) -> torch.Tensor:
        kwargs = {
            "text_token_count": self.text_token_count,
            "image_height": self.image_height,
            "image_width": self.image_width,
            "text_regions": self.text_regions,
            "image_region_rows": self.image_region_rows,
            "image_region_columns": self.image_region_columns,
            "channel_groups": self.channel_groups,
        }
        return residual_input_samples(hidden_states, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        route: torch.Tensor,
    ):
        residual = hidden_states
        normalized, gate, attn_global, bias, _ = self._prefix(
            hidden_states, temb, image_rotary_emb
        )
        raw_samples = self._raw_samples(hidden_states)
        normalized_samples = self._raw_samples(normalized)
        online_signals = torch.cat(
            (
                relative_component_drift(
                    raw_samples, self.cached_raw_samples
                ).reshape(-1, 1),
                relative_component_drift(
                    normalized_samples, self.cached_normalized_samples
                ).reshape(-1, 1),
            ),
            dim=1,
        )
        if self.part == "anchor":
            mlp_hidden = self.block.act_mlp(self.block.proj_mlp(normalized))
            out_mlp = self.block.proj_out_mlp(mlp_hidden)
            mlp_global = reduce_from_tensor_model_parallel_region(
                out_mlp,
                process_group=self.block.proj_out_mlp.tensor_parallel_group,
            )
            teacher_error = _relative_rms_bfloat16(
                mlp_global, self.cached_mlp_global
            ).float()
            output = residual + gate * (attn_global + mlp_global + bias)
            marker = route[:1].to(output.dtype) * 0 + 10
            return (
                output,
                online_signals,
                teacher_error,
                marker,
                mlp_global,
                raw_samples,
                normalized_samples,
            )
        output = residual + gate * (attn_global + self.cached_mlp_global + bias)
        teacher_error = online_signals[:, :1] * 0
        marker = route[:1].to(output.dtype) * 0 + 20
        return (
            output,
            online_signals,
            teacher_error,
            marker,
            self.cached_mlp_global,
            self.cached_raw_samples,
            self.cached_normalized_samples,
        )


def _make_ids_and_rotary(
    *, image_height: int, image_width: int, text_token_count: int, dtype: torch.dtype
) -> torch.Tensor:
    from difflet.layers.embeddings import FluxPosEmbed

    text_ids = torch.zeros(text_token_count, 3, dtype=dtype)
    image_ids = torch.zeros(image_height, image_width, 3, dtype=dtype)
    image_ids[..., 1] += torch.arange(image_height, dtype=dtype)[:, None]
    image_ids[..., 2] += torch.arange(image_width, dtype=dtype)[None, :]
    ids = torch.cat((text_ids, image_ids.reshape(-1, 3)), dim=0)
    return torch.stack(FluxPosEmbed(theta=10000, axes_dim=(16, 56, 56))(ids), dim=2).to(
        dtype=dtype
    )


def _make_inputs(config: Any, *, seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dtype = config.neuron_config.torch_dtype
    sequence = int(config.text_token_count) + int(config.image_height) * int(config.image_width)
    return (
        torch.randn([1, sequence, int(config.inner_dim)], generator=generator, dtype=dtype),
        torch.randn([1, int(config.inner_dim)], generator=generator, dtype=dtype),
        _make_ids_and_rotary(
            image_height=int(config.image_height),
            image_width=int(config.image_width),
            text_token_count=int(config.text_token_count),
            dtype=dtype,
        ),
    )


def build_application(args: argparse.Namespace, part: str):
    os.environ.setdefault("DIFFLET_BACKEND", "trainium")
    os.environ["LOCAL_WORLD_SIZE"] = str(args.tp_degree)

    from difflet.backends.trainium.core.application_base import NeuronApplicationBase
    from difflet.backends.trainium.core.config import InferenceConfig, NeuronConfig
    from difflet.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper

    source_config = _model_config(Path(args.model_dir))
    inner_dim = int(source_config["num_attention_heads"]) * int(
        source_config["attention_head_dim"]
    )
    image_height = args.height // 16
    image_width = args.width // 16

    class SplitConfig(InferenceConfig):
        def get_required_attributes(self):
            return [
                "part",
                "inner_dim",
                "num_attention_heads",
                "attention_head_dim",
                "mlp_ratio",
                "text_token_count",
                "image_height",
                "image_width",
                "text_regions",
                "image_region_rows",
                "image_region_columns",
                "channel_groups",
                "source_model_dir",
                "block_index",
            ]

    class SplitWrapper(ModelWrapper):
        def input_generator(self):
            values = _make_inputs(self.config, seed=0)
            if self.config.part == "hybrid":
                sequence = int(self.config.text_token_count) + int(
                    self.config.image_height
                ) * int(self.config.image_width)
                values = (
                    *values,
                    torch.randn(
                        [1, sequence, int(self.config.inner_dim)],
                        dtype=self.config.neuron_config.torch_dtype,
                    ),
                )
            return [values]

        def get_model_instance(self):
            def create_model():
                return (
                    _FluxSingleBlockSplit(self.config, part=self.config.part)
                    .to(dtype=self.config.neuron_config.torch_dtype)
                    .eval()
                )

            return BaseModelInstance(module_cls=create_model, input_output_aliases={})

        def forward(self, *model_inputs):
            if self.model is None:
                raise RuntimeError("Forward called before load")
            return self._forward(*model_inputs)

    class SplitApplication(NeuronApplicationBase):
        _model_cls = object

        def __init__(self, *app_args, **app_kwargs):
            super().__init__(*app_args, **app_kwargs)
            self.model = SplitWrapper(
                config=self.config,
                model_cls=self._model_cls,
                tag=f"FluxSingleBlockResidualSplit_{part}",
                compiler_args=(
                    "--model-type=transformer -O1 --auto-cast=none "
                    "--internal-hlo2tensorizer-options='--verify-hlo=true'"
                ),
                priority_model_idx=0,
            )
            self.models.append(self.model)

        @classmethod
        def get_config_cls(cls):
            return SplitConfig

        @classmethod
        def get_state_dict(cls, model_name_or_path: str, config: InferenceConfig):
            del model_name_or_path
            return _load_single_block_state_dict(
                Path(config.source_model_dir),
                int(config.block_index),
                inner_dim=int(config.inner_dim),
                dtype=config.neuron_config.torch_dtype,
            )

        @staticmethod
        def convert_hf_to_neuron_state_dict(state_dict, config):
            del config
            return state_dict

        @staticmethod
        def update_state_dict_for_tied_weights(state_dict):
            del state_dict

        def forward(self, *model_inputs, **kwargs):
            del kwargs
            return self.models[0](*model_inputs)

    config = SplitConfig(
        neuron_config=NeuronConfig(
            batch_size=1,
            tp_degree=args.tp_degree,
            world_size=args.tp_degree,
            torch_dtype=torch.bfloat16,
        ),
        part=part,
        inner_dim=inner_dim,
        num_attention_heads=int(source_config["num_attention_heads"]),
        attention_head_dim=int(source_config["attention_head_dim"]),
        mlp_ratio=float(source_config.get("mlp_ratio", 4.0)),
        text_token_count=args.text_token_count,
        image_height=image_height,
        image_width=image_width,
        text_regions=args.text_regions,
        image_region_rows=args.image_region_rows,
        image_region_columns=args.image_region_columns,
        channel_groups=args.channel_groups,
        source_model_dir=str(Path(args.model_dir).resolve()),
        block_index=args.block_index,
    )
    part_dir = Path(args.cache_dir).expanduser().resolve() / part
    app = SplitApplication(model_path=str(part_dir), config=config)
    return app, config, part_dir


def build_bucketed_application(args: argparse.Namespace, *, signal_bank: bool = False):
    os.environ.setdefault("DIFFLET_BACKEND", "trainium")
    os.environ["LOCAL_WORLD_SIZE"] = str(args.tp_degree)

    from torch_neuronx import BucketModelConfig
    from neuronx_distributed.trace.model_builder import BaseModelInstance

    from difflet.backends.trainium.core.application_base import NeuronApplicationBase
    from difflet.backends.trainium.core.config import InferenceConfig, NeuronConfig
    from difflet.backends.trainium.core.model_wrapper import ModelWrapper

    source_config = _model_config(Path(args.model_dir))
    inner_dim = int(source_config["num_attention_heads"]) * int(
        source_config["attention_head_dim"]
    )
    image_height = args.height // 16
    image_width = args.width // 16
    sequence = args.text_token_count + image_height * image_width
    sketch_regions = (
        args.text_regions + args.image_region_rows * args.image_region_columns
    )

    class BucketedSplitConfig(InferenceConfig):
        def get_required_attributes(self):
            return [
                "inner_dim",
                "num_attention_heads",
                "attention_head_dim",
                "mlp_ratio",
                "text_token_count",
                "image_height",
                "image_width",
                "text_regions",
                "image_region_rows",
                "image_region_columns",
                "channel_groups",
                "source_model_dir",
                "block_index",
            ]

    class BucketedSplitInstance(BaseModelInstance):
        def __init__(self, config):
            self.config = config

            def create_model():
                return (
                    (
                        _FluxSingleBlockBucketedSignalBank(config)
                        if signal_bank
                        else _FluxSingleBlockBucketed(config)
                    )
                    .to(dtype=config.neuron_config.torch_dtype)
                    .eval()
                )

            super().__init__(
                create_model,
                input_output_aliases={},
            )

        def get(self, bucket_rank, **kwargs):
            del bucket_rank
            self.module.part = str(kwargs["part"])
            if signal_bank:
                return self.module, {
                    self.module.cached_mlp_global: 4,
                    self.module.cached_raw_samples: 5,
                    self.module.cached_normalized_samples: 6,
                }
            return self.module, {
                self.module.cached_mlp_global: 4,
                self.module.cached_input_sketch: 5,
            }

    class BucketedSplitWrapper(ModelWrapper):
        def __init__(self, *wrapper_args, **wrapper_kwargs):
            super().__init__(*wrapper_args, **wrapper_kwargs)
            dtype = self.config.neuron_config.torch_dtype
            moments_shape = (
                1,
                sketch_regions,
                args.channel_groups,
                2,
            )
            samples_shape = (1, sketch_regions, args.channel_groups)
            if signal_bank:
                shared_state_buffer = [
                    torch.zeros(1, sequence, inner_dim, dtype=dtype),
                    torch.zeros(samples_shape, dtype=dtype),
                    torch.zeros(samples_shape, dtype=dtype),
                ]
            else:
                shared_state_buffer = [
                    torch.zeros(1, sequence, inner_dim, dtype=dtype),
                    torch.zeros(moments_shape, dtype=dtype),
                ]
            self.bucket_config = BucketModelConfig(
                _bucket_kernel,
                shared_state_buffer=shared_state_buffer,
                func_kwargs=[{"part": "anchor"}, {"part": "hybrid"}],
            )

        def input_generator(self):
            values = _make_inputs(self.config, seed=0)
            return [
                (*values, torch.tensor([0], dtype=torch.int32)),
                (*values, torch.tensor([1, 1], dtype=torch.int32)),
            ]

        def get_model_instance(self):
            return BucketedSplitInstance(self.config)

        def forward(self, *model_inputs):
            if self.model is None:
                raise RuntimeError("Forward called before load")
            return self._forward(*model_inputs)

    class BucketedSplitApplication(NeuronApplicationBase):
        _model_cls = (
            _FluxSingleBlockBucketedSignalBank
            if signal_bank
            else _FluxSingleBlockBucketed
        )

        def __init__(self, *app_args, **app_kwargs):
            super().__init__(*app_args, **app_kwargs)
            self.model = BucketedSplitWrapper(
                config=self.config,
                model_cls=self._model_cls,
                tag=(
                    "FluxSingleBlockResidualSplit_bucketed_bank_v2"
                    if signal_bank
                    else "FluxSingleBlockResidualSplit_bucketed"
                ),
                compiler_args=(
                    "--model-type=transformer -O1 --auto-cast=none "
                    "--internal-hlo2tensorizer-options='--verify-hlo=true'"
                ),
                priority_model_idx=0,
            )
            self.models.append(self.model)

        @classmethod
        def get_config_cls(cls):
            return BucketedSplitConfig

        @classmethod
        def get_state_dict(cls, model_name_or_path: str, config: InferenceConfig):
            del model_name_or_path
            return _load_single_block_state_dict(
                Path(config.source_model_dir),
                int(config.block_index),
                inner_dim=int(config.inner_dim),
                dtype=config.neuron_config.torch_dtype,
            )

        @staticmethod
        def convert_hf_to_neuron_state_dict(state_dict, config):
            del config
            return state_dict

        @staticmethod
        def update_state_dict_for_tied_weights(state_dict):
            del state_dict

        def forward(self, *model_inputs, **kwargs):
            del kwargs
            return self.models[0](*model_inputs)

    config = BucketedSplitConfig(
        neuron_config=NeuronConfig(
            batch_size=1,
            tp_degree=args.tp_degree,
            world_size=args.tp_degree,
            torch_dtype=torch.bfloat16,
        ),
        inner_dim=inner_dim,
        num_attention_heads=int(source_config["num_attention_heads"]),
        attention_head_dim=int(source_config["attention_head_dim"]),
        mlp_ratio=float(source_config.get("mlp_ratio", 4.0)),
        text_token_count=args.text_token_count,
        image_height=image_height,
        image_width=image_width,
        text_regions=args.text_regions,
        image_region_rows=args.image_region_rows,
        image_region_columns=args.image_region_columns,
        channel_groups=args.channel_groups,
        source_model_dir=str(Path(args.model_dir).resolve()),
        block_index=args.block_index,
    )
    bucket_name = "bucketed-bank-v2" if signal_bank else "bucketed"
    part_dir = Path(args.cache_dir).expanduser().resolve() / bucket_name
    app = BucketedSplitApplication(model_path=str(part_dir), config=config)
    return app, config, part_dir


def _compile_and_load(args: argparse.Namespace, part: str):
    app, config, part_dir = build_application(args, part)
    if args.force_compile and part_dir.exists():
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if not (part_dir / "model.pt").exists():
        app.compile(str(part_dir), debug=False)
        compile_seconds = time.perf_counter() - started
    else:
        compile_seconds = 0.0
    load_started = time.perf_counter()
    app.load(str(part_dir), skip_warmup=bool(args.skip_warmup))
    return app, config, part_dir, compile_seconds, time.perf_counter() - load_started


def _compile_and_load_bucketed(
    args: argparse.Namespace, *, signal_bank: bool = False
):
    app, config, part_dir = build_bucketed_application(args, signal_bank=signal_bank)
    if args.force_compile and part_dir.exists():
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if not (part_dir / "model.pt").exists():
        app.compile(str(part_dir), debug=False)
        compile_seconds = time.perf_counter() - started
    else:
        compile_seconds = 0.0
    load_started = time.perf_counter()
    app.load(str(part_dir), skip_warmup=bool(args.skip_warmup))
    return app, config, part_dir, compile_seconds, time.perf_counter() - load_started


def _as_tuple(value: Any) -> tuple[torch.Tensor, ...]:
    if torch.is_tensor(value):
        return (value,)
    if isinstance(value, (tuple, list)) and all(torch.is_tensor(item) for item in value):
        return tuple(value)
    raise TypeError(f"unexpected split-block output: {type(value)!r}")


def _clone_outputs(value: Any) -> tuple[torch.Tensor, ...]:
    """Materialize outputs before the next Neuron call can reuse host buffers."""
    return tuple(item.detach().cpu().clone() for item in _as_tuple(value))


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            left.detach().float().cpu().reshape(-1),
            right.detach().float().cpu().reshape(-1),
            dim=0,
        )
    )


def _relative_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(relative_component_drift(left.detach().cpu(), right.detach().cpu())[0])


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0
        for index in order[cursor:end]:
            result[index] = rank
        cursor = end
    return result


def _pearson(left: list[float], right: list[float]) -> float:
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_energy = sum((x - left_mean) ** 2 for x in left)
    right_energy = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_energy * right_energy)
    return numerator / denominator if denominator else 0.0


def _spearman(left: list[float], right: list[float]) -> float:
    return _pearson(_ranks(left), _ranks(right))


def _validate_hardware_args(args: argparse.Namespace) -> tuple[float, ...]:
    if not args.allow_hardware or args.foreground_ack != HARDWARE_ACK:
        raise RuntimeError(
            f"hardware execution requires --allow-hardware and --foreground-ack {HARDWARE_ACK!r}"
        )
    scales = _strict_scales(args.perturbation_scales)
    if args.perturbation_seeds <= 0:
        raise ValueError("perturbation-seeds must be positive")
    return scales


def _run_bucketed(args: argparse.Namespace) -> dict[str, Any]:
    scales = _validate_hardware_args(args)
    signal_bank = args.execution_mode == "bucketed-bank"
    signal_names = SIGNAL_BANK_NAMES if signal_bank else ("normalized_input_moments",)
    app, config, compiled_dir, compile_seconds, load_seconds = (
        _compile_and_load_bucketed(args, signal_bank=signal_bank)
    )
    base_inputs = _make_inputs(config, seed=1729)
    route_anchor = torch.tensor([0], dtype=torch.int32)
    route_hybrid = torch.tensor([1, 1], dtype=torch.int32)

    with torch.no_grad():
        base_full, _, _, anchor_marker = _clone_outputs(
            app(*base_inputs, route_anchor)
        )
        same_hybrid, same_signal, _, hybrid_marker = _clone_outputs(
            app(*base_inputs, route_hybrid)
        )
        app(*base_inputs, route_anchor)
        anchor_started = time.perf_counter()
        app(*base_inputs, route_anchor)
        anchor_seconds = time.perf_counter() - anchor_started
        hybrid_started = time.perf_counter()
        app(*base_inputs, route_hybrid)
        hybrid_seconds = time.perf_counter() - hybrid_started

    same_step = {
        "output_cosine": _cosine(base_full, same_hybrid),
        "output_relative_rms_error": _relative_error(base_full, same_hybrid),
        "online_signals": {
            name: float(value)
            for name, value in zip(
                signal_names,
                same_signal.detach().float().cpu().reshape(-1).tolist(),
            )
        },
        "anchor_bucket_marker": float(anchor_marker.detach().float().cpu().reshape(-1)[0]),
        "hybrid_bucket_marker": float(hybrid_marker.detach().float().cpu().reshape(-1)[0]),
        "output_device": str(base_full.device),
        "aliased_state_returned_to_host": False,
    }
    rows = []
    for perturbation_seed in range(args.perturbation_seeds):
        generator = torch.Generator(device="cpu").manual_seed(2801 + perturbation_seed)
        noise = torch.randn(base_inputs[0].shape, generator=generator, dtype=base_inputs[0].dtype)
        for scale in scales:
            perturbed_hidden = (base_inputs[0].float() + scale * noise.float()).to(
                base_inputs[0].dtype
            )
            current_inputs = (perturbed_hidden, base_inputs[1], base_inputs[2])
            with torch.no_grad():
                app(*base_inputs, route_anchor)
                stale_hybrid, online_signal, _, hybrid_marker = _clone_outputs(
                    app(*current_inputs, route_hybrid)
                )
                current_full, anchor_signal, teacher_error, anchor_marker = _clone_outputs(
                    app(*current_inputs, route_anchor)
                )
            row = {
                "perturbation_seed": perturbation_seed,
                "perturbation_scale": scale,
                "online_signals": {
                    name: float(value)
                    for name, value in zip(
                        signal_names,
                        online_signal.detach().float().cpu().reshape(-1).tolist(),
                    )
                },
                "anchor_signals": {
                    name: float(value)
                    for name, value in zip(
                        signal_names,
                        anchor_signal.detach().float().cpu().reshape(-1).tolist(),
                    )
                },
                "true_mlp_cache_error": float(
                    teacher_error.detach().float().cpu().reshape(-1)[0]
                ),
                "block_output_error": _relative_error(current_full, stale_hybrid),
                "block_output_cosine": _cosine(current_full, stale_hybrid),
                "anchor_bucket_marker": float(
                    anchor_marker.detach().float().cpu().reshape(-1)[0]
                ),
                "hybrid_bucket_marker": float(
                    hybrid_marker.detach().float().cpu().reshape(-1)[0]
                ),
            }
            row["online_sketch_drift"] = row["online_signals"][signal_names[0]]
            row["anchor_sketch_drift"] = row["anchor_signals"][signal_names[0]]
            rows.append(row)
            print(
                f"[flux-single-split-bucketed] seed={perturbation_seed} scale={scale:g} "
                f"signal={row['online_sketch_drift']:.6f} "
                f"mlp_error={row['true_mlp_cache_error']:.6f}",
                flush=True,
            )

    mlp_errors = [float(row["true_mlp_cache_error"]) for row in rows]
    block_errors = [float(row["block_output_error"]) for row in rows]
    signal_results = {}
    for signal_name in signal_names:
        signals = [float(row["online_signals"][signal_name]) for row in rows]
        signal_results[signal_name] = {
            "same_step_floor": float(same_step["online_signals"][signal_name]),
            "vs_true_mlp_error_spearman": _spearman(signals, mlp_errors),
            "vs_block_output_error_spearman": _spearman(signals, block_errors),
            "maximum_online_vs_anchor_abs_difference": max(
                abs(
                    float(row["online_signals"][signal_name])
                    - float(row["anchor_signals"][signal_name])
                )
                for row in rows
            ),
        }
    sequence = int(config.text_token_count) + int(config.image_height) * int(
        config.image_width
    )
    payload_bytes = 2 * sequence * int(config.inner_dim)
    sketch_regions = int(config.text_regions) + int(config.image_region_rows) * int(
        config.image_region_columns
    )
    moments_bytes = 2 * sketch_regions * int(config.channel_groups) * 2
    samples_bytes = 2 * sketch_regions * int(config.channel_groups)
    sketch_bytes = 2 * samples_bytes if signal_bank else moments_bytes
    return {
        "schema": "difflet-flux-single-block-residual-split-probe",
        "schema_revision": 2,
        "serving_claim": False,
        "execution_mode": (
            "bucketed_shared_hbm_state_signal_bank"
            if signal_bank
            else "bucketed_shared_hbm_state"
        ),
        "model_revision": DEFAULT_REVISION,
        "block_index": args.block_index,
        "shape": {
            "hidden": list(base_inputs[0].shape),
            "mlp_cache_payload": [1, sequence, int(config.inner_dim)],
            "signal_state_shapes": (
                {
                    "raw_input_samples": [
                        1,
                        sketch_regions,
                        int(config.channel_groups),
                    ],
                    "normalized_input_samples": [
                        1,
                        sketch_regions,
                        int(config.channel_groups),
                    ],
                }
                if signal_bank
                else {
                    "normalized_input_moments": [
                        1,
                        sketch_regions,
                        int(config.channel_groups),
                        2,
                    ]
                }
            ),
        },
        "compiled": {
            "path": str(compiled_dir),
            "compile_seconds": compile_seconds,
            "load_seconds": load_seconds,
            "bucket_count": 2,
            "routing": "route tensor shape 1 selects anchor; shape 2 selects hybrid",
            "shared_state": (
                ["cached_mlp_global", *signal_names]
                if signal_bank
                else ["cached_mlp_global", "cached_input_sketch"]
            ),
        },
        "same_step_parity": same_step,
        "measurement_cost": {
            "anchor_call_seconds": anchor_seconds,
            "hybrid_call_seconds": hybrid_seconds,
            "single_block_payload_bytes": payload_bytes,
            "single_block_sketch_bytes": sketch_bytes,
            "sketch_to_payload_fraction": sketch_bytes / payload_bytes,
            "cache_payload_host_transfer_bytes_per_call": 0,
            "latency_is_serving_claim": False,
        },
        "synthetic_identifiability": {
            "sample_count": len(rows),
            "signal_families": signal_results,
            "preselected_signal": (
                "raw_input_samples; normalized_input_samples is a post-opened AOT-parity follow-up"
                if signal_bank
                else "normalized_input_moments"
            ),
            "opened_synthetic_data": True,
            "trajectory_quality_claim": False,
        },
        "rows": rows,
        "next_gate": (
            "Only bucket parity, HBM-state persistence, and strong synthetic monotonicity permit "
            "a separately registered real-denoising-state collection."
        ),
    }


def _run_separate(args: argparse.Namespace) -> dict[str, Any]:
    scales = _validate_hardware_args(args)
    anchor, config, anchor_dir, anchor_compile, anchor_load = _compile_and_load(args, "anchor")
    hybrid, _, hybrid_dir, hybrid_compile, hybrid_load = _compile_and_load(args, "hybrid")
    base_inputs = _make_inputs(config, seed=1729)

    with torch.no_grad():
        anchor_started = time.perf_counter()
        base_full, base_mlp, base_sketch = _as_tuple(anchor(*base_inputs))
        anchor_seconds = time.perf_counter() - anchor_started
        hybrid_started = time.perf_counter()
        same_hybrid, same_sketch = _as_tuple(hybrid(*base_inputs, base_mlp))
        hybrid_seconds = time.perf_counter() - hybrid_started

    same_step = {
        "output_cosine": _cosine(base_full, same_hybrid),
        "output_relative_rms_error": _relative_error(base_full, same_hybrid),
        "sketch_relative_rms_error": _relative_error(base_sketch, same_sketch),
        "anchor_output_device": str(base_full.device),
        "cache_payload_device": str(base_mlp.device),
        "hybrid_output_device": str(same_hybrid.device),
    }
    rows = []
    for perturbation_seed in range(args.perturbation_seeds):
        generator = torch.Generator(device="cpu").manual_seed(2801 + perturbation_seed)
        noise = torch.randn(base_inputs[0].shape, generator=generator, dtype=base_inputs[0].dtype)
        for scale in scales:
            perturbed_hidden = (base_inputs[0].float() + scale * noise.float()).to(
                base_inputs[0].dtype
            )
            current_inputs = (perturbed_hidden, base_inputs[1], base_inputs[2])
            with torch.no_grad():
                current_full, current_mlp, current_sketch = _as_tuple(anchor(*current_inputs))
                stale_hybrid, online_sketch = _as_tuple(
                    hybrid(*current_inputs, base_mlp)
                )
            rows.append(
                {
                    "perturbation_seed": perturbation_seed,
                    "perturbation_scale": scale,
                    "online_sketch_drift": _relative_error(current_sketch, base_sketch),
                    "hybrid_sketch_identity_error": _relative_error(
                        current_sketch, online_sketch
                    ),
                    "true_mlp_cache_error": _relative_error(current_mlp, base_mlp),
                    "block_output_error": _relative_error(current_full, stale_hybrid),
                    "block_output_cosine": _cosine(current_full, stale_hybrid),
                }
            )
            print(
                f"[flux-single-split] seed={perturbation_seed} scale={scale:g} "
                f"signal={rows[-1]['online_sketch_drift']:.6f} "
                f"mlp_error={rows[-1]['true_mlp_cache_error']:.6f}",
                flush=True,
            )

    signals = [float(row["online_sketch_drift"]) for row in rows]
    mlp_errors = [float(row["true_mlp_cache_error"]) for row in rows]
    block_errors = [float(row["block_output_error"]) for row in rows]
    payload_bytes = base_mlp.numel() * base_mlp.element_size()
    sketch_bytes = base_sketch.numel() * base_sketch.element_size()
    return {
        "schema": "difflet-flux-single-block-residual-split-probe",
        "schema_revision": 1,
        "serving_claim": False,
        "model_revision": DEFAULT_REVISION,
        "block_index": args.block_index,
        "shape": {
            "hidden": list(base_inputs[0].shape),
            "mlp_cache_payload": list(base_mlp.shape),
            "mlp_input_sketch": list(base_sketch.shape),
        },
        "compiled": {
            "anchor_path": str(anchor_dir),
            "hybrid_path": str(hybrid_dir),
            "anchor_compile_seconds": anchor_compile,
            "hybrid_compile_seconds": hybrid_compile,
            "anchor_load_seconds": anchor_load,
            "hybrid_load_seconds": hybrid_load,
        },
        "same_step_parity": same_step,
        "measurement_cost": {
            "anchor_call_seconds": anchor_seconds,
            "hybrid_call_seconds": hybrid_seconds,
            "single_block_payload_bytes": payload_bytes,
            "single_block_sketch_bytes": sketch_bytes,
            "sketch_to_payload_fraction": sketch_bytes / payload_bytes,
            "latency_is_serving_claim": False,
        },
        "synthetic_identifiability": {
            "sample_count": len(rows),
            "sketch_drift_vs_true_mlp_error_spearman": _spearman(signals, mlp_errors),
            "sketch_drift_vs_block_output_error_spearman": _spearman(signals, block_errors),
            "opened_synthetic_data": True,
            "trajectory_quality_claim": False,
        },
        "rows": rows,
        "next_gate": (
            "Only same-step numerical parity and strong synthetic monotonicity permit a separately "
            "registered real-denoising-state collection."
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.execution_mode in {"bucketed", "bucketed-bank"}:
        return _run_bucketed(args)
    return _run_separate(args)


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run(args)
        status = 0
    except Exception as exc:
        result = {
            "schema": "difflet-flux-single-block-residual-split-probe",
            "schema_revision": 1,
            "serving_claim": False,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-30:],
        }
        status = 1
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(text, flush=True)
    if args.metrics_out:
        output = Path(args.metrics_out).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"[flux-single-split] metrics={output}", flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
