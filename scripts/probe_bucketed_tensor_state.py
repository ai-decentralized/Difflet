#!/usr/bin/env python3
"""Verify large BF16 shared-state aliases across two Trainium AOT buckets.

This isolates BucketModel state transport from FLUX math.  The anchor bucket
writes deterministic payload/sketch tensors into shared HBM buffers; the
hybrid bucket reads them and compares them with identical explicit inputs.
Zero error proves that any same-input FLUX discrepancy comes from compiling
the branch math into two graphs, rather than from the shared-state alias.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List

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

PAYLOAD_SHAPE = (1, 4608, 3072)
SKETCH_SHAPE = (1, 24, 32, 2)


def _bucket_kernel():
    @torch.jit.script
    def select(inputs: List[torch.Tensor]):
        bucket_idx = 0
        if inputs[-1].shape[0] > 1:
            bucket_idx = 1
        return inputs, torch.tensor([bucket_idx], dtype=torch.int32)

    return select


def _relative_rms(current: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    difference = current.float() - expected.float()
    numerator = difference.square().mean().sqrt()
    denominator = expected.float().square().mean().sqrt().clamp_min(1e-12)
    return (numerator / denominator).reshape(1)


class _BucketTensorState(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.payload_state = nn.Parameter(
            torch.zeros(PAYLOAD_SHAPE, dtype=torch.bfloat16), requires_grad=False
        )
        self.sketch_state = nn.Parameter(
            torch.zeros(SKETCH_SHAPE, dtype=torch.bfloat16), requires_grad=False
        )
        self.part = "anchor"

    def forward(
        self,
        expected_payload: torch.Tensor,
        expected_sketch: torch.Tensor,
        route: torch.Tensor,
    ):
        if self.part == "anchor":
            route_zero = route[:1].to(expected_payload.dtype).reshape(1, 1, 1) * 0
            # Read the aliased parameters so both state inputs remain in the
            # anchor HLO lowering context.  The registered shared buffers start
            # at zero and this probe intentionally performs one anchor write.
            payload = expected_payload + self.payload_state + route_zero
            sketch = (
                expected_sketch
                + self.sketch_state
                + route_zero.reshape(1, 1, 1, 1)
            )
            marker = route[:1].to(torch.float32) * 0 + 10
        else:
            payload = self.payload_state
            sketch = self.sketch_state
            marker = route[:1].to(torch.float32) * 0 + 20
        payload_error = _relative_rms(payload, expected_payload)
        sketch_error = _relative_rms(sketch, expected_sketch)
        return payload_error, sketch_error, marker, payload, sketch


def build_app(cache_dir: Path, *, tp_degree: int):
    from neuronx_distributed.trace.model_builder import BaseModelInstance
    from torch_neuronx import BucketModelConfig

    from difflet.backends.trainium.core.application_base import NeuronApplicationBase
    from difflet.backends.trainium.core.config import InferenceConfig, NeuronConfig
    from difflet.backends.trainium.core.model_wrapper import ModelWrapper

    class TensorStateConfig(InferenceConfig):
        def get_required_attributes(self):
            return []

    class TensorStateInstance(BaseModelInstance):
        def __init__(self):
            super().__init__(_BucketTensorState, input_output_aliases={})

        def get(self, bucket_rank, **kwargs):
            del bucket_rank
            self.module.part = str(kwargs["part"])
            return self.module, {
                self.module.payload_state: 3,
                self.module.sketch_state: 4,
            }

    class TensorStateWrapper(ModelWrapper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.bucket_config = BucketModelConfig(
                _bucket_kernel,
                shared_state_buffer=[
                    torch.zeros(PAYLOAD_SHAPE, dtype=torch.bfloat16),
                    torch.zeros(SKETCH_SHAPE, dtype=torch.bfloat16),
                ],
                func_kwargs=[{"part": "anchor"}, {"part": "hybrid"}],
            )

        def input_generator(self):
            payload = torch.randn(PAYLOAD_SHAPE, dtype=torch.bfloat16)
            sketch = torch.randn(SKETCH_SHAPE, dtype=torch.bfloat16)
            return [
                (payload, sketch, torch.tensor([0], dtype=torch.int32)),
                (payload, sketch, torch.tensor([1, 1], dtype=torch.int32)),
            ]

        def get_model_instance(self):
            return TensorStateInstance()

        def forward(self, *inputs):
            return self._forward(*inputs)

    class TensorStateApplication(NeuronApplicationBase):
        _model_cls = _BucketTensorState

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model = TensorStateWrapper(
                config=self.config,
                model_cls=self._model_cls,
                tag="BucketedPersistentTensorState",
                compiler_args="--model-type=transformer -O1 --auto-cast=none",
                priority_model_idx=0,
            )
            self.models.append(self.model)

        @classmethod
        def get_config_cls(cls):
            return TensorStateConfig

        @classmethod
        def get_state_dict(cls, model_name_or_path, config):
            del model_name_or_path, config
            return {}

        @staticmethod
        def convert_hf_to_neuron_state_dict(state_dict, config):
            del state_dict, config
            return {}

        @staticmethod
        def update_state_dict_for_tied_weights(state_dict):
            del state_dict

        def forward(self, *inputs):
            return self.models[0](*inputs)

    config = TensorStateConfig(
        neuron_config=NeuronConfig(
            batch_size=1,
            tp_degree=tp_degree,
            world_size=tp_degree,
            torch_dtype=torch.bfloat16,
        )
    )
    return TensorStateApplication(model_path=str(cache_dir), config=config)


def _clone_outputs(output) -> tuple[torch.Tensor, ...]:
    if not isinstance(output, (tuple, list)) or len(output) != 3:
        raise TypeError("tensor-state probe must return two errors and one marker")
    return tuple(value.detach().cpu().clone() for value in output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default="/home/ubuntu/difflet-artifacts/bucketed-tensor-state-20260804",
    )
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--force-compile", action="store_true")
    args = parser.parse_args()
    if args.tp_degree <= 0:
        raise ValueError("tp-degree must be positive")
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    if args.force_compile and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LOCAL_WORLD_SIZE"] = str(args.tp_degree)
    app = build_app(cache_dir, tp_degree=args.tp_degree)
    if not (cache_dir / "model.pt").exists():
        app.compile(str(cache_dir), debug=False)
    app.load(str(cache_dir), skip_warmup=True)

    generator = torch.Generator(device="cpu").manual_seed(4101)
    payload = torch.randn(PAYLOAD_SHAPE, generator=generator, dtype=torch.bfloat16)
    sketch = torch.randn(SKETCH_SHAPE, generator=generator, dtype=torch.bfloat16)
    route_anchor = torch.tensor([0], dtype=torch.int32)
    route_hybrid = torch.tensor([1, 1], dtype=torch.int32)
    anchor_outputs = _clone_outputs(app(payload, sketch, route_anchor))
    hybrid_outputs = _clone_outputs(app(payload, sketch, route_hybrid))
    result = {
        "tp_degree": args.tp_degree,
        "payload_shape": PAYLOAD_SHAPE,
        "sketch_shape": SKETCH_SHAPE,
        "anchor_payload_relative_rms": float(anchor_outputs[0].reshape(-1)[0]),
        "anchor_sketch_relative_rms": float(anchor_outputs[1].reshape(-1)[0]),
        "anchor_marker": float(anchor_outputs[2].reshape(-1)[0]),
        "hybrid_payload_relative_rms": float(hybrid_outputs[0].reshape(-1)[0]),
        "hybrid_sketch_relative_rms": float(hybrid_outputs[1].reshape(-1)[0]),
        "hybrid_marker": float(hybrid_outputs[2].reshape(-1)[0]),
    }
    result["shared_state_passed"] = bool(
        result["anchor_payload_relative_rms"] == 0.0
        and result["anchor_sketch_relative_rms"] == 0.0
        and result["hybrid_payload_relative_rms"] == 0.0
        and result["hybrid_sketch_relative_rms"] == 0.0
        and result["anchor_marker"] == 10.0
        and result["hybrid_marker"] == 20.0
    )
    print(result, flush=True)
    return 0 if result["shared_state_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
