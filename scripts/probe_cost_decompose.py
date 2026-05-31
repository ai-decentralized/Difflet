#!/usr/bin/env python3
"""cclog 79 Test 1: decompose the ~51 ms/step TeaCache probe NEFF cost.

Runs ONE variant per process (avoids the multi-load c10::Error seen in
cclog 75). Variants:

  probe_full  — current probe NEFF: returns (delta, mod_input 63 MB)
  delta_only  — same compute, returns ONLY the scalar delta (no mod_input out)
  trivial     — near-empty NEFF (sum of small input): pure dispatch + mark_step floor
  markstep    — xm.mark_step() in a loop, no NEFF: mark_step floor alone

Decomposition:
  probe_full - delta_only      = mod_input 63 MB output write cost
  trivial                       = dispatch + mark_step floor
  markstep                      = mark_step alone
  delta_only - trivial          = probe compute + delta output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
from safetensors.torch import load_file as load_safetensors_file  # noqa: E402

SOURCE = ROOT / ".nova-cache" / "f3_hunyuan_n4_4d8s1r" / "source"
COMPILED = ROOT / ".nova-cache" / "f3_hunyuan_n4_4d8s1r" / "compiled"
BUNDLE = ROOT / ".nova-cache" / "hunyuan_dit_inputs" / "cat_walking_4step.safetensors"
META = Path(str(BUNDLE) + ".meta.json")
N_ITERS = 20


def _median_ms(times: list[float]) -> float:
    return sorted(times)[len(times) // 2] * 1000.0


def _bundle_inputs():
    tensors = {
        k: v.to(dtype=torch.bfloat16) if v.is_floating_point() else v
        for k, v in load_safetensors_file(str(BUNDLE), device="cpu").items()
    }
    return tensors


def _run_markstep() -> dict:
    import torch_xla.core.xla_model as xm

    x = torch.ones(1024, device=xm.xla_device())
    for _ in range(5):
        x = x + 1.0
        xm.mark_step()
    times = []
    for _ in range(N_ITERS):
        t = time.perf_counter()
        xm.mark_step()
        times.append(time.perf_counter() - t)
    return {"variant": "markstep", "median_ms": _median_ms(times)}


def _build_probe_app(meta, *, model_cls=None):
    from nova.models.hunyuan_video.application import NeuronHunyuanVideoApplication
    from nova.pipeline.parallel_config import NovaParallelConfig

    app = NeuronHunyuanVideoApplication(
        model_path=str(SOURCE),
        parallel=NovaParallelConfig(tp_degree=4, cp_enabled=False),
        dtype=torch.bfloat16,
        shape={"height": int(meta["height"]), "width": int(meta["width"]),
               "num_frames": int(meta["num_frames"])},
        text_seq_len=int(meta["text_seq_len"]),
        enable_vae_decoder=False,
    )
    if model_cls is not None:
        app.teacache_probe._model_cls = model_cls
        app.teacache_probe.models[0].model_cls = model_cls
    return app


def _run_probe_full(meta, tensors) -> dict:
    from nova.models.hunyuan_video.application import HunyuanVideoDiTInputBundle

    app = _build_probe_app(meta)
    app.load(str(COMPILED), skip_warmup=True)
    bundle = HunyuanVideoDiTInputBundle(
        hidden_states=tensors["latents_init"],
        timestep=tensors["timesteps"][:1].clone(),
        encoder_hidden_states=tensors["encoder_hidden_states"],
        encoder_attention_mask=tensors["encoder_attention_mask"],
        pooled_projections=tensors["pooled_projections"],
        guidance=tensors["guidance"],
    )
    app.teacache_mod_input(bundle)  # warm
    times = []
    for _ in range(N_ITERS):
        t = time.perf_counter()
        out = app.teacache_mod_input(bundle)
        _ = out.detach().cpu()  # materialize like real device-probe handoff
        times.append(time.perf_counter() - t)
    return {"variant": "probe_full", "median_ms": _median_ms(times)}


def _run_delta_only(meta, tensors, out_dir: Path) -> dict:
    from nova.backends.trainium.hunyuan_video.teacache_probe_costmodels import (
        HunyuanVideoTeacacheProbeDeltaOnly,
    )

    app = _build_probe_app(meta, model_cls=HunyuanVideoTeacacheProbeDeltaOnly)
    # compile the delta-only probe to its own dir
    app.teacache_probe.compile(str(out_dir))
    app.teacache_probe.load(str(out_dir), skip_warmup=True)
    seq_len = (
        (int(app.teacache_probe.config.latent_frames) // int(app.teacache_probe.config.patch_size_t))
        * (int(app.teacache_probe.config.latent_height) // int(app.teacache_probe.config.patch_size))
        * (int(app.teacache_probe.config.latent_width) // int(app.teacache_probe.config.patch_size))
    )
    inner = int(app.teacache_probe.config.inner_dim)
    prev = torch.zeros((1, seq_len, inner), dtype=torch.bfloat16)
    args = (
        tensors["latents_init"], tensors["timesteps"][:1].clone(),
        tensors["encoder_hidden_states"], tensors["encoder_attention_mask"],
        tensors["pooled_projections"], tensors["guidance"], prev,
    )
    app.teacache_probe.models[0](*args)  # warm
    times = []
    for _ in range(N_ITERS):
        t = time.perf_counter()
        out = app.teacache_probe.models[0](*args)
        _ = float(out.detach().cpu().item())
        times.append(time.perf_counter() - t)
    return {"variant": "delta_only", "median_ms": _median_ms(times)}


def _run_trivial(meta, tensors, out_dir: Path) -> dict:
    import os as _os

    from nova.backends.trainium.core.application_base import NeuronApplicationBase
    from nova.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper
    from nova.backends.trainium.hunyuan_video.backbone import (
        HunyuanVideoBackboneInferenceConfig,
    )
    from nova.backends.trainium.hunyuan_video.teacache_probe_costmodels import (
        HunyuanVideoTrivialNEFF,
    )
    from nova.models.hunyuan_video.application import create_hunyuan_video_backbone_config
    from nova.pipeline.parallel_config import NovaParallelConfig

    class _TrivialWrapper(ModelWrapper):
        def __init__(self, config, model_cls, tag="", compiler_args=None,
                     priority_model_idx=None, model_init_kwargs=None):
            super().__init__(config, model_cls, tag, compiler_args,
                             priority_model_idx, model_init_kwargs or {})
            self.bucket_config = None

        def input_generator(self):
            return [(torch.randn([1, 64], dtype=torch.bfloat16),)]

        def get_model_instance(self):
            def _create():
                m = self.model_cls(self.config)
                return m.to(dtype=self.config.neuron_config.torch_dtype).eval()
            return BaseModelInstance(module_cls=_create, input_output_aliases={})

        def forward(self, x):
            return self._forward(x)

    class _TrivialApp(NeuronApplicationBase):
        _model_cls = HunyuanVideoTrivialNEFF

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.model = _TrivialWrapper(
                config=self.config, model_cls=self._model_cls,
                tag="HunyuanVideoTrivialNEFF",
                compiler_args=(
                    "--model-type=transformer -O1 --auto-cast=none"
                ),
                priority_model_idx=0,
            )
            self.models.append(self.model)
            self.dtype = self.config.neuron_config.torch_dtype
            _os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)

        @classmethod
        def get_config_cls(cls):
            return HunyuanVideoBackboneInferenceConfig

        def forward(self, *mi, **kw):
            return self.models[0](*mi, **kw)

        @staticmethod
        def convert_hf_to_neuron_state_dict(sd, config):
            del config
            return {f"scale": sd.get("scale", torch.ones(1))}

        @staticmethod
        def update_state_dict_for_tied_weights(sd):
            pass

    config = create_hunyuan_video_backbone_config(
        model_path=str(SOURCE), world_size=4, tp_degree=4, dtype=torch.bfloat16,
        height=int(meta["height"]), width=int(meta["width"]),
        num_frames=int(meta["num_frames"]), text_seq_len=int(meta["text_seq_len"]),
        batch_size=1,
    )
    app = _TrivialApp(model_path=str(SOURCE), config=config)
    app.compile(str(out_dir))
    app.load(str(out_dir), skip_warmup=True)
    x = torch.randn([1, 64], dtype=torch.bfloat16)
    app.models[0](x)  # warm
    times = []
    for _ in range(N_ITERS):
        t = time.perf_counter()
        out = app.models[0](x)
        _ = float(out.detach().cpu().reshape(-1)[0].item())
        times.append(time.perf_counter() - t)
    return {"variant": "trivial", "median_ms": _median_ms(times)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True,
                   choices=("probe_full", "delta_only", "trivial", "markstep"))
    p.add_argument("--out", required=True)
    args = p.parse_args()

    meta = json.loads(META.read_text())
    tensors = _bundle_inputs()

    if args.variant == "markstep":
        result = _run_markstep()
    elif args.variant == "probe_full":
        result = _run_probe_full(meta, tensors)
    elif args.variant == "delta_only":
        result = _run_delta_only(meta, tensors, ROOT / ".nova-cache" / "f3_hunyuan_n4_4d8s1r" / "compiled_delta_only")
    elif args.variant == "trivial":
        result = _run_trivial(meta, tensors, ROOT / ".nova-cache" / "f3_hunyuan_n4_4d8s1r" / "compiled_trivial")
    else:
        raise SystemExit(f"unknown variant {args.variant}")

    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(f"[cost] {result['variant']}: {result['median_ms']:.2f} ms/call", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
