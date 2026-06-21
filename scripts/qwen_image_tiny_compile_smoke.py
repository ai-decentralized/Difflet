#!/usr/bin/env python3
"""Compile/load smoke for the Qwen-Image Trainium transformer boundary."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

NEURON_VENV = Path("/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference")
NEURON_PYTHON = NEURON_VENV / "bin" / "python"


def ensure_runtime_python() -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        if Path(sys.executable) != NEURON_PYTHON and NEURON_PYTHON.exists():
            env = os.environ.copy()
            env["PATH"] = f"{NEURON_VENV / 'bin'}:{env.get('PATH', '')}"
            env["PYTHONPATH"] = f"{Path(__file__).resolve().parents[1]}{os.pathsep}{env.get('PYTHONPATH', '')}"
            os.execve(str(NEURON_PYTHON), [str(NEURON_PYTHON), *sys.argv], env)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="/tmp/difflet_qwen_image_tiny_model")
    parser.add_argument("--cache-dir", default="/tmp/difflet_qwen_image_tiny_cache")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--text-seq-len", type=int, default=16)
    parser.add_argument("--tp-degree", type=int, default=1)
    parser.add_argument("--force-clean", action="store_true")
    parser.add_argument("--load", action="store_true", help="Load and run one Neuron forward.")
    parser.add_argument("--compare-cpu", action="store_true", help="Compare Neuron output to CPU.")
    return parser


def create_tiny_model(model_dir: Path) -> None:
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

    transformer_dir = model_dir / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    model = QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=64,
        out_channels=16,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=32,
        guidance_embeds=False,
        axes_dims_rope=(2, 2, 4),
    )
    model.save_pretrained(transformer_dir, safe_serialization=True)


def main() -> None:
    ensure_runtime_python()
    import torch

    from difflet import DiffletParallelConfig, DiffletPipeline
    from difflet.models.qwen_image.application import QwenImageDiTInputBundle

    args = build_parser().parse_args()
    model_dir = Path(args.model_dir)
    cache_dir = Path(args.cache_dir)
    if args.force_clean:
        for path in (model_dir, cache_dir):
            if path.exists():
                shutil.rmtree(path)
    create_tiny_model(model_dir)

    pipe = DiffletPipeline.from_pretrained(
        str(model_dir),
        model_type="qwen_image",
        parallel=DiffletParallelConfig(tp_degree=args.tp_degree),
        dtype=torch.bfloat16,
        height=args.height,
        width=args.width,
        compile_cache_dir=str(cache_dir),
        load=args.load,
        skip_warmup=True,
        application_kwargs={"text_seq_len": args.text_seq_len},
    )

    result = {
        "compiled_path": str(pipe.compiled_path),
        "components": [spec.name for spec in pipe.app.components()],
        "artifact_ready": pipe.app.has_compiled_artifacts(str(pipe.compiled_path)),
    }
    if args.load:
        from safetensors.torch import load_file

        from difflet.backends.trainium.qwen_image.transformer import _QwenImageTransformerTraceModule

        contract = pipe.app.dit_input_contract()
        bundle = QwenImageDiTInputBundle(
            hidden_states=torch.randn(contract["hidden_states"]["shape"], dtype=torch.bfloat16),
            timestep=torch.ones(contract["timestep"]["shape"], dtype=torch.bfloat16),
            encoder_hidden_states=torch.randn(
                contract["encoder_hidden_states"]["shape"],
                dtype=torch.bfloat16,
            ),
            encoder_hidden_states_mask=torch.ones(
                contract["encoder_hidden_states_mask"]["shape"],
                dtype=torch.bool,
            ),
            guidance=torch.ones(contract["guidance"]["shape"], dtype=torch.bfloat16),
        )
        output = pipe(bundle)
        pipeline_output = pipe(
            latents=bundle.hidden_states,
            timesteps=torch.ones(1, dtype=torch.bfloat16),
            encoder_hidden_states=bundle.encoder_hidden_states,
            encoder_hidden_states_mask=bundle.encoder_hidden_states_mask,
            guidance=bundle.guidance,
            output_type="latent",
        )
        result.update(
            {
                "output_shape": list(output.shape),
                "output_dtype": str(output.dtype),
                "output_mean": float(output.float().mean()),
                "pipeline_output_shape": list(pipeline_output.latents.shape),
                "pipeline_output_dtype": str(pipeline_output.latents.dtype),
            }
        )
        if args.compare_cpu:
            cpu_model = _QwenImageTransformerTraceModule(pipe.app.transformer.config)
            state_dict = load_file(model_dir / "transformer" / "diffusion_pytorch_model.safetensors")
            cpu_model.load_state_dict(
                {f"transformer.{key}": value for key, value in state_dict.items()},
                strict=False,
            )
            cpu_model = cpu_model.eval().to(dtype=torch.bfloat16)
            with torch.no_grad():
                expected = cpu_model(*bundle.as_model_inputs())
            actual_flat = output.float().flatten()
            expected_flat = expected.float().flatten()
            cosine = torch.nn.functional.cosine_similarity(
                actual_flat.unsqueeze(0),
                expected_flat.unsqueeze(0),
            ).item()
            result.update(
                {
                    "cpu_cosine": cosine,
                    "cpu_mean_abs": float((output.float() - expected.float()).abs().mean()),
                    "cpu_max_abs": float((output.float() - expected.float()).abs().max()),
                }
            )

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
