#!/usr/bin/env python3
"""Compile/load smoke for the LTX-2 Trainium transformer boundary."""

from __future__ import annotations

import argparse
import json
import os
import shutil
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="/tmp/difflet_ltx2_tiny_model")
    parser.add_argument("--cache-dir", default="/tmp/difflet_ltx2_tiny_cache")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--num-frames", type=int, default=9)
    parser.add_argument("--text-seq-len", type=int, default=4)
    parser.add_argument("--audio-num-frames", type=int, default=2)
    parser.add_argument("--tp-degree", type=int, default=1)
    parser.add_argument("--force-clean", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--load", action="store_true", help="Load and run one Neuron forward.")
    parser.add_argument("--compare-cpu", action="store_true", help="Compare Neuron output to CPU.")
    return parser


def create_tiny_model(model_dir: Path) -> None:
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

    transformer_dir = model_dir / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    model = LTX2VideoTransformer3DModel(
        in_channels=8,
        out_channels=8,
        patch_size=1,
        patch_size_t=1,
        num_attention_heads=2,
        attention_head_dim=4,
        cross_attention_dim=8,
        vae_scale_factors=(8, 32, 32),
        pos_embed_max_pos=20,
        base_height=2048,
        base_width=2048,
        audio_in_channels=8,
        audio_out_channels=8,
        audio_patch_size=16,
        audio_patch_size_t=1,
        audio_num_attention_heads=2,
        audio_attention_head_dim=4,
        audio_cross_attention_dim=8,
        audio_scale_factor=4,
        audio_pos_embed_max_pos=20,
        audio_sampling_rate=16000,
        audio_hop_length=160,
        num_layers=1,
        caption_channels=8,
        rope_double_precision=False,
        use_prompt_embeddings=True,
    )
    model.save_pretrained(transformer_dir, safe_serialization=True)


def _make_bundle(pipe):
    import torch

    from difflet.models.ltx_2.application import LTX2DiTInputBundle
    from difflet.models.ltx_2.pipeline import make_ltx_2_audio_coords, make_ltx_2_video_coords

    contract = pipe.app.dit_input_contract()
    cfg = pipe.app.transformer.config
    return LTX2DiTInputBundle(
        hidden_states=torch.randn(contract["hidden_states"]["shape"], dtype=torch.bfloat16),
        audio_hidden_states=torch.randn(
            contract["audio_hidden_states"]["shape"],
            dtype=torch.bfloat16,
        ),
        encoder_hidden_states=torch.randn(
            contract["encoder_hidden_states"]["shape"],
            dtype=torch.bfloat16,
        ),
        audio_encoder_hidden_states=torch.randn(
            contract["audio_encoder_hidden_states"]["shape"],
            dtype=torch.bfloat16,
        ),
        timestep=torch.ones(contract["timestep"]["shape"], dtype=torch.bfloat16),
        sigma=torch.ones(contract["sigma"]["shape"], dtype=torch.bfloat16),
        encoder_attention_mask=torch.ones(
            contract["encoder_attention_mask"]["shape"],
            dtype=torch.bool,
        ),
        audio_encoder_attention_mask=torch.ones(
            contract["audio_encoder_attention_mask"]["shape"],
            dtype=torch.bool,
        ),
        video_coords=make_ltx_2_video_coords(
            batch_size=contract["video_coords"]["shape"][0],
            num_frames=int(cfg.latent_num_frames),
            height=int(cfg.latent_height),
            width=int(cfg.latent_width),
            device="cpu",
            patch_size=int(cfg.patch_size),
            patch_size_t=int(cfg.patch_size_t),
            scale_factors=tuple(cfg.vae_scale_factors),
            causal_offset=int(getattr(cfg, "causal_offset", 1)),
            fps=float(getattr(cfg, "frame_rate", 24.0)),
        ),
        audio_coords=make_ltx_2_audio_coords(
            batch_size=contract["audio_coords"]["shape"][0],
            audio_num_frames=int(cfg.audio_num_frames),
            device="cpu",
            patch_size_t=int(cfg.audio_patch_size_t),
            scale_factor=int(cfg.audio_scale_factor),
            causal_offset=int(getattr(cfg, "causal_offset", 1)),
            sampling_rate=int(cfg.audio_sampling_rate),
            hop_length=int(cfg.audio_hop_length),
        ),
    )


def main() -> None:
    ensure_runtime_python()
    import torch

    from difflet import DiffletParallelConfig, DiffletPipeline

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
        model_type="ltx_2",
        parallel=DiffletParallelConfig(tp_degree=args.tp_degree),
        dtype=torch.bfloat16,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        compile_cache_dir=str(cache_dir),
        skip_compile=args.skip_compile,
        load=args.load,
        skip_warmup=True,
        application_kwargs={
            "text_seq_len": args.text_seq_len,
            "audio_num_frames": args.audio_num_frames,
        },
    )

    result = {
        "compiled_path": str(pipe.compiled_path),
        "components": [spec.name for spec in pipe.app.components()],
        "artifact_ready": pipe.app.has_compiled_artifacts(str(pipe.compiled_path)),
    }
    if args.load:
        from safetensors.torch import load_file

        from difflet.backends.trainium.ltx_2.transformer import _LTX2TransformerTraceModule

        bundle = _make_bundle(pipe)
        video_out, audio_out = pipe(bundle)
        result.update(
            {
                "video_output_shape": list(video_out.shape),
                "audio_output_shape": list(audio_out.shape),
                "video_output_dtype": str(video_out.dtype),
                "audio_output_dtype": str(audio_out.dtype),
                "video_output_mean": float(video_out.float().mean()),
                "audio_output_mean": float(audio_out.float().mean()),
            }
        )
        if args.compare_cpu:
            cpu_model = _LTX2TransformerTraceModule(pipe.app.transformer.config)
            state_dict = load_file(
                model_dir / "transformer" / "diffusion_pytorch_model.safetensors"
            )
            cpu_model.load_state_dict(
                {f"transformer.{key}": value for key, value in state_dict.items()},
                strict=False,
            )
            cpu_model = cpu_model.eval().to(dtype=torch.bfloat16)
            with torch.no_grad():
                expected_video, expected_audio = cpu_model(*bundle.as_model_inputs())
            video_cosine = torch.nn.functional.cosine_similarity(
                video_out.float().flatten().unsqueeze(0),
                expected_video.float().flatten().unsqueeze(0),
            ).item()
            audio_cosine = torch.nn.functional.cosine_similarity(
                audio_out.float().flatten().unsqueeze(0),
                expected_audio.float().flatten().unsqueeze(0),
            ).item()
            result.update(
                {
                    "cpu_video_cosine": video_cosine,
                    "cpu_audio_cosine": audio_cosine,
                    "cpu_video_mean_abs": float(
                        (video_out.float() - expected_video.float()).abs().mean()
                    ),
                    "cpu_audio_mean_abs": float(
                        (audio_out.float() - expected_audio.float()).abs().mean()
                    ),
                    "cpu_video_max_abs": float(
                        (video_out.float() - expected_video.float()).abs().max()
                    ),
                    "cpu_audio_max_abs": float(
                        (audio_out.float() - expected_audio.float()).abs().max()
                    ),
                }
            )

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
