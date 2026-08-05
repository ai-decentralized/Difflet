#!/usr/bin/env python3
"""Verify that two bucketed NEFFs share one aliased HBM state buffer."""

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


def _bucket_kernel():
    @torch.jit.script
    def select(inputs: List[torch.Tensor]):
        bucket_idx = 0
        if inputs[-1].shape[0] > 1:
            bucket_idx = 1
        return inputs, torch.tensor([bucket_idx], dtype=torch.int32)

    return select


class _BucketCounter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.state = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=False)
        self.part = "anchor"

    def forward(self, increment: torch.Tensor, route: torch.Tensor):
        if self.part == "anchor":
            new_state = self.state + increment.reshape(-1)[0]
            marker = route.to(torch.float32) * 0.0 + 10.0
        else:
            new_state = self.state
            marker = route.to(torch.float32) * 0.0 + 20.0
        readable = new_state + 100.0
        return readable, route, marker, new_state


def build_app(cache_dir: Path):
    from torch_neuronx import BucketModelConfig
    from neuronx_distributed.trace.model_builder import BaseModelInstance

    from difflet.backends.trainium.core.application_base import NeuronApplicationBase
    from difflet.backends.trainium.core.config import InferenceConfig, NeuronConfig
    from difflet.backends.trainium.core.model_wrapper import ModelWrapper

    class CounterConfig(InferenceConfig):
        def get_required_attributes(self):
            return []

    class CounterInstance(BaseModelInstance):
        def __init__(self):
            super().__init__(_BucketCounter, input_output_aliases={})

        def get(self, bucket_rank, **kwargs):
            del bucket_rank
            self.module.part = str(kwargs["part"])
            return self.module, {self.module.state: 3}

    class CounterWrapper(ModelWrapper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.bucket_config = BucketModelConfig(
                _bucket_kernel,
                shared_state_buffer=[torch.zeros(1, dtype=torch.float32)],
                func_kwargs=[{"part": "anchor"}, {"part": "hybrid"}],
            )

        def input_generator(self):
            return [
                (
                    torch.ones(1, dtype=torch.float32),
                    torch.tensor([0], dtype=torch.int32),
                ),
                (
                    torch.ones(1, dtype=torch.float32),
                    torch.tensor([1, 1], dtype=torch.int32),
                ),
            ]

        def get_model_instance(self):
            return CounterInstance()

        def forward(self, increment, route):
            return self._forward(increment, route)

    class CounterApplication(NeuronApplicationBase):
        _model_cls = _BucketCounter

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model = CounterWrapper(
                config=self.config,
                model_cls=self._model_cls,
                tag="BucketedPersistentCounter",
                compiler_args="--model-type=transformer -O1 --auto-cast=none",
                priority_model_idx=0,
            )
            self.models.append(self.model)

        @classmethod
        def get_config_cls(cls):
            return CounterConfig

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

    config = CounterConfig(
        neuron_config=NeuronConfig(
            batch_size=1,
            tp_degree=1,
            world_size=1,
            torch_dtype=torch.float32,
        )
    )
    return CounterApplication(model_path=str(cache_dir), config=config)


def _read(output) -> tuple[float, float]:
    if not isinstance(output, (tuple, list)) or len(output) != 3:
        raise TypeError("bucket counter must return readable, route, and marker")
    return (
        float(output[0].detach().cpu().reshape(-1)[0]),
        float(output[2].detach().cpu().reshape(-1)[0]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default="/home/ubuntu/difflet-artifacts/bucketed-persistent-state-20260804",
    )
    parser.add_argument("--force-compile", action="store_true")
    args = parser.parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    if args.force_compile and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LOCAL_WORLD_SIZE"] = "1"
    app = build_app(cache_dir)
    if not (cache_dir / "model.pt").exists():
        app.compile(str(cache_dir), debug=False)
    app.load(str(cache_dir), skip_warmup=True)

    anchor = torch.tensor([3.0], dtype=torch.float32)
    ignored = torch.tensor([99.0], dtype=torch.float32)
    route_anchor = torch.tensor([0], dtype=torch.int32)
    route_hybrid = torch.tensor([1, 1], dtype=torch.int32)
    observations = [
        _read(app(anchor, route_anchor)),
        _read(app(ignored, route_hybrid)),
        _read(app(anchor, route_anchor)),
        _read(app(ignored, route_hybrid)),
    ]
    expected = [(103.0, 10.0), (103.0, 20.0), (106.0, 10.0), (106.0, 20.0)]
    print(
        {
            "observations": observations,
            "expected": expected,
            "shared_state_passed": observations == expected,
        }
    )
    return 0 if observations == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
