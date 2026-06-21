#!/usr/bin/env python3
"""cclog 79 Test 2 (v2): persistent device state across nrt_execute via the
PRODUCTION alias pattern.

v1 failed (KeyError placeholder) for two reasons, now fixed:
  1. the state buffer was loaded from the checkpoint → identity changed → no
     longer matched the traced placeholder. Fix: keep state as a runtime buffer
     (not in the checkpoint), like KV-cache past_key_values.
  2. used the generic static-dict alias. Fix: a custom ModelInstance whose
     get() builds {self.module.state: output_index} AFTER load_module, exactly
     like DecoderModelInstance.get() (model_wrapper.py:1626-1631).

Counter model: state buffer (0). forward returns new = state + 1 (output 0),
aliased back to state. 5 calls should read 1,2,3,4,5.
"""

from __future__ import annotations

import os
import sys
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

OUT_DIR = ROOT / ".difflet-cache" / "f3_hunyuan_n4_4d8s1r" / "compiled_counter"


class CounterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # tkg-nki3 recipe: state MUST be nn.Parameter, not register_buffer —
        # the alias mechanism in hlo_conversion.py only iterates
        # named_parameters() to build aliased_inputs. requires_grad=False.
        self.state = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=False)

    def forward(self, increment: torch.Tensor):
        # increment is USED (an unused input gets pruned → empty flattener input
        # → "List trace inputs must have elements"). Real fused-A uses latent etc.
        new = self.state + increment.reshape(-1)[0]
        # output 0 = readable (distinct tensor, +100 so it can't dedup with
        # output 1); output 1 = new state aliased back to the Parameter.
        readable = new + 100.0
        return readable, new


def main() -> int:
    from neuronx_distributed.trace.model_builder import BaseModelInstance
    from difflet.backends.trainium.core.application_base import NeuronApplicationBase
    from difflet.backends.trainium.core.model_wrapper import ModelWrapper
    from difflet.backends.trainium.hunyuan_video.backbone import (
        HunyuanVideoBackboneInferenceConfig,
    )
    from difflet.models.hunyuan_video.application import create_hunyuan_video_backbone_config

    class _PersistentStateInstance(BaseModelInstance):
        """Builds the alias post-load referencing self.module's runtime buffer,
        mirroring DecoderModelInstance.get() for KV-cache."""

        def __init__(self, module_cls):
            super().__init__(module_cls, input_output_aliases={})

        def get(self, bucket_rank, **kwargs):
            # output 0 = readable value (kept for user); output 1 = new state,
            # aliased back to the state Parameter (written in-place in HBM).
            aliases = {self.module.state: 1}
            return self.module, aliases

    class _CounterWrapper(ModelWrapper):
        def __init__(self, config, model_cls, tag="", compiler_args=None,
                     priority_model_idx=None, model_init_kwargs=None):
            super().__init__(config, model_cls, tag, compiler_args,
                             priority_model_idx, model_init_kwargs or {})
            self.bucket_config = None

        def input_generator(self):
            return [(torch.ones([1], dtype=torch.float32),)]

        def get_model_instance(self):
            return _PersistentStateInstance(module_cls=CounterModel)

        def forward(self, x):
            return self._forward(x)

    class _CounterApp(NeuronApplicationBase):
        _model_cls = CounterModel

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.model = _CounterWrapper(
                config=self.config, model_cls=self._model_cls, tag="CounterModel",
                compiler_args="--model-type=transformer -O1 --auto-cast=none",
                priority_model_idx=0,
            )
            self.models.append(self.model)
            self.dtype = self.config.neuron_config.torch_dtype
            os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)

        @classmethod
        def get_config_cls(cls):
            return HunyuanVideoBackboneInferenceConfig

        def forward(self, *mi, **kw):
            return self.models[0](*mi, **kw)

        @staticmethod
        def convert_hf_to_neuron_state_dict(sd, config):
            del config, sd
            return {}  # state stays a runtime buffer; nothing loaded

        @staticmethod
        def update_state_dict_for_tied_weights(sd):
            pass

    source = ROOT / ".difflet-cache" / "f3_hunyuan_n4_4d8s1r" / "source"
    config = create_hunyuan_video_backbone_config(
        model_path=str(source), world_size=4, tp_degree=4, dtype=torch.bfloat16,
        height=320, width=512, num_frames=61, text_seq_len=256, batch_size=1,
    )
    app = _CounterApp(model_path=str(source), config=config)
    app.compile(str(OUT_DIR))
    app.load(str(OUT_DIR), skip_warmup=True)

    dummy = torch.ones([1], dtype=torch.float32)
    reads = []
    for i in range(5):
        out = app.models[0](dummy)
        if isinstance(out, (tuple, list)):
            out = out[0]
        val = float(out.detach().cpu().reshape(-1)[0].item())
        reads.append(val)
        print(f"[test2] call {i + 1}: returned {val}", flush=True)

    # readable = new + 100 where new = state + 1; if state accumulates 1..5,
    # readable = 101,102,103,104,105.
    accumulates = reads == [101.0, 102.0, 103.0, 104.0, 105.0]
    print(f"[test2] reads={reads}", flush=True)
    print(f"[test2] PERSISTENT_STATE_WORKS={accumulates}", flush=True)
    return 0 if accumulates else 1


if __name__ == "__main__":
    sys.exit(main())
