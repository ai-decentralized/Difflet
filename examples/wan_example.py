"""Wan 2.2 text-to-video inference on Trainium via Nova.

Single-instance launch (text + DiT, 4 NeuronCores, save latents):

    NEURON_RT_NUM_CORES=4 python examples/wan_example.py \\
        --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \\
        --tp-degree 4 --skip-warmup \\
        --num-inference-steps 2 \\
        --num-frames 9 --height 480 --width 832 \\
        --prompt "a cat walking" \\
        --save-latents /tmp/wan_latents.pt \\
        --output-type latent \\
        --enable-text --enable-transformer

Single-instance VAE decode (1 NeuronCore, decode saved latents):

    NEURON_RT_NUM_CORES=1 python examples/wan_example.py \\
        --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \\
        --tp-degree 1 --skip-warmup \\
        --num-frames 9 --height 480 --width 832 \\
        --load-latents /tmp/wan_latents.pt \\
        --output-type pt --output /tmp/wan_smoke.mp4 \\
        --enable-vae

Larger Trainium instance (text + DiT + VAE in one process):

    NEURON_RT_NUM_CORES=8 python examples/wan_example.py \\
        --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \\
        --tp-degree 4 --skip-warmup \\
        --num-frames 9 --prompt "a cat walking" \\
        --output /tmp/wan_smoke.mp4

The 4-core ``trn3pd98.3xlarge`` cannot fit TP=4 transformer + TP=1 VAE in
one process (cclogs/16 §7.7); use ``scripts/wan_smoke.sh`` to drive the
two-stage sequential split.

Cache key composition mirrors Flux (cclogs/16 §8.6):
    .nova-cache/wan_text_encoder_smoke / wan_backbone_smoke /
    wan_backbone_2_smoke / wan_vae_decoder_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

from nova.models.wan.application import NeuronWanApplication, _latent_num_frames
from nova.pipeline.parallel_config import NovaParallelConfig
from nova.pipeline.path_resolver import resolve_model_path


_DEFAULT_TEXT_DIR = ".nova-cache/wan_text_encoder_smoke"
_DEFAULT_TRANSFORMER_DIR = ".nova-cache/wan_backbone_smoke"
_DEFAULT_TRANSFORMER_2_DIR = ".nova-cache/wan_backbone_2_smoke"
_DEFAULT_VAE_DIR = ".nova-cache/wan_vae_decoder_smoke"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True,
                   help="HF model id (e.g. Wan-AI/Wan2.2-T2V-A14B-Diffusers) or local path")
    p.add_argument("--tp-degree", type=int, default=4)
    p.add_argument("--cp-degree", type=int, default=1,
                   help="Context-parallel degree (1 = disabled)")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=9)
    p.add_argument("--text-seq-len", type=int, default=512)
    p.add_argument("--num-inference-steps", type=int, default=2)
    p.add_argument("--guidance-scale", type=float, default=1.0)

    p.add_argument("--prompt", default=None)
    p.add_argument("--negative-prompt", default=None)
    p.add_argument("--seed", type=int, default=20260510)

    p.add_argument("--output", default=None,
                   help="Path to write final tensor (.pt) or video (.mp4 if vae enabled)")
    p.add_argument("--output-type", choices=["latent", "pt"], default="pt")
    p.add_argument("--save-latents", default=None,
                   help="Write the post-denoise latent tensor to this path (.pt)")
    p.add_argument("--load-latents", default=None,
                   help="Skip text+denoise and feed this latent tensor into VAE decode")

    p.add_argument("--enable-text", dest="enable_text", action="store_true", default=None)
    p.add_argument("--no-text", dest="enable_text", action="store_false")
    p.add_argument("--enable-transformer", dest="enable_transformer", action="store_true", default=None)
    p.add_argument("--no-transformer", dest="enable_transformer", action="store_false")
    p.add_argument("--enable-transformer-2", dest="enable_transformer_2", action="store_true", default=None)
    p.add_argument("--no-transformer-2", dest="enable_transformer_2", action="store_false")
    p.add_argument("--enable-vae", dest="enable_vae", action="store_true", default=None)
    p.add_argument("--no-vae", dest="enable_vae", action="store_false")

    p.add_argument("--text-dir", default=_DEFAULT_TEXT_DIR)
    p.add_argument("--transformer-dir", default=_DEFAULT_TRANSFORMER_DIR)
    p.add_argument("--transformer-2-dir", default=_DEFAULT_TRANSFORMER_2_DIR)
    p.add_argument("--vae-dir", default=_DEFAULT_VAE_DIR)
    p.add_argument("--compiled-dir", default=".nova-cache/wan_smoke_run",
                   help="Working directory; component artifacts are symlinked under it")

    p.add_argument("--local-files-only", action="store_true", default=True)
    p.add_argument("--allow-remote", dest="local_files_only", action="store_false")
    p.add_argument("--download-weights", action="store_true",
                   help="Download missing transformer/text_encoder/tokenizer/vae files from HF")
    p.add_argument("--skip-warmup", action="store_true")
    p.add_argument("--teacache-calibration", default=None,
                   help="path to a TeaCache calibration JSON; enables adaptive step skipping")

    return p.parse_args(argv)


def _resolve_components(args: argparse.Namespace, has_latents: bool) -> dict:
    """Defaults: use whatever dirs exist, but honor --no-* / --enable-* overrides."""
    comp = {
        "text": args.enable_text,
        "transformer": args.enable_transformer,
        "transformer_2": args.enable_transformer_2,
        "vae": args.enable_vae,
    }
    paths = {
        "text": Path(args.text_dir),
        "transformer": Path(args.transformer_dir),
        "transformer_2": Path(args.transformer_2_dir),
        "vae": Path(args.vae_dir),
    }
    for k, v in comp.items():
        if v is None:
            comp[k] = paths[k].exists()
    if has_latents:
        # When loading latents from disk, text/transformer must not run.
        comp["text"] = False
        comp["transformer"] = False
        comp["transformer_2"] = False
    if args.output_type == "latent":
        comp["vae"] = False
    return comp


def _read_neuron_config(path: Path) -> dict:
    config_path = path / "neuron_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _check_artifact_shape(path: Path, expected: dict, label: str) -> None:
    cfg = _read_neuron_config(path)
    actual = {k: cfg.get(k) for k in expected}
    if actual != expected:
        raise RuntimeError(
            f"{label} artifact shape mismatch: expected {expected}, got {actual}. "
            f"Recompile the component."
        )


def _has_component_weights(path: Path) -> bool:
    names = (
        "model.safetensors",
        "model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return any((path / name).exists() for name in names)


def _has_tokenizer_assets(path: Path) -> bool:
    return any(
        (path / name).exists()
        for name in ("spiece.model", "tokenizer.model", "tokenizer.json")
    )


def _download_files(repo_id: str, prefix: str, *, suffixes: tuple[str, ...]) -> None:
    from huggingface_hub import hf_hub_download, list_repo_files

    candidates = [
        name for name in list_repo_files(repo_id)
        if name.startswith(prefix) and name.endswith(suffixes)
    ]
    if not candidates:
        raise FileNotFoundError(f"no remote files for {repo_id}:{prefix}")
    print(f"[wan] downloading {len(candidates)} files for {prefix}", flush=True)
    for filename in candidates:
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"[wan] downloaded {filename} -> {path}", flush=True)


def _ensure_weights(model_dir: Path, repo_id: str, components: dict, args: argparse.Namespace) -> None:
    """Download missing component weights and tokenizer assets if requested."""
    requirements: list[tuple[str, str, tuple[str, ...]]] = []
    if components["text"]:
        requirements.append(("text_encoder",
                             "text_encoder/",
                             (".safetensors", ".safetensors.index.json", ".bin", ".bin.index.json")))
        if args.prompt is not None and not _has_tokenizer_assets(model_dir / "tokenizer"):
            requirements.append(("tokenizer", "tokenizer/", (".model", ".json")))
    if components["transformer"]:
        requirements.append(("transformer",
                             "transformer/",
                             (".safetensors", ".safetensors.index.json", ".bin", ".bin.index.json")))
    if components["transformer_2"]:
        requirements.append(("transformer_2",
                             "transformer_2/",
                             (".safetensors", ".safetensors.index.json", ".bin", ".bin.index.json")))
    if components["vae"]:
        requirements.append(("vae",
                             "vae/",
                             (".safetensors", ".safetensors.index.json", ".bin", ".bin.index.json")))
    for label, prefix, suffixes in requirements:
        local = model_dir / label
        if label == "tokenizer":
            ok = _has_tokenizer_assets(local)
        else:
            ok = _has_component_weights(local)
        if ok:
            continue
        if not args.download_weights:
            raise FileNotFoundError(
                f"missing {label} files under {local}. "
                f"Re-run with --download-weights to fetch them from {repo_id}."
            )
        _download_files(repo_id, prefix, suffixes=suffixes)


def _stage_components(compiled_dir: Path, components: dict, args: argparse.Namespace) -> None:
    if compiled_dir.exists() or compiled_dir.is_symlink():
        if compiled_dir.is_symlink() or compiled_dir.is_file():
            compiled_dir.unlink()
        else:
            shutil.rmtree(compiled_dir)
    compiled_dir.mkdir(parents=True)

    sources = {
        "text": (Path(args.text_dir), "text_encoder"),
        "transformer": (Path(args.transformer_dir), "transformer"),
        "transformer_2": (Path(args.transformer_2_dir), "transformer_2"),
        "vae": (Path(args.vae_dir), "vae_decoder"),
    }
    for key, enabled in components.items():
        if not enabled:
            continue
        src, name = sources[key]
        os.symlink(src.resolve(), compiled_dir / name, target_is_directory=True)
        print(f"[wan] staged {name} -> {src.resolve()}", flush=True)


def _save_video(tensor: torch.Tensor, output_path: str) -> bool:
    """Write `(1, C, T, H, W)` tensor to MP4. Returns True on success."""
    try:
        from diffusers.utils import export_to_video
    except ImportError:
        print("[wan] diffusers.utils.export_to_video unavailable; saving as .pt", flush=True)
        return False
    frames = tensor.detach().to(torch.float32).clamp(-1, 1)
    frames = ((frames + 1.0) / 2.0).clamp(0, 1)
    frames = (frames[0].permute(1, 2, 3, 0).cpu().numpy() * 255).round().astype("uint8")
    try:
        export_to_video(list(frames), output_path, fps=16)
    except Exception as exc:  # pragma: no cover - exec env dependent
        print(f"[wan] mp4 export failed ({exc}); skipping", flush=True)
        return False
    print(f"[wan] video saved to {output_path}", flush=True)
    return True


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    has_latents = bool(args.load_latents)
    components = _resolve_components(args, has_latents)
    if not any(components.values()):
        raise RuntimeError(
            "No components enabled. Pass --enable-text/--enable-transformer/--enable-vae "
            "or precompile component artifacts first."
        )

    model_dir = Path(resolve_model_path(args.model, local_files_only=args.local_files_only))
    repo_id = args.model if "/" in args.model and not Path(args.model).exists() else "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    latent_frames = _latent_num_frames(args.num_frames)

    print(f"[wan] model_dir       = {model_dir}", flush=True)
    print(f"[wan] components      = {components}", flush=True)
    print(f"[wan] video shape     = ({args.height}, {args.width}, {args.num_frames})", flush=True)
    print(f"[wan] latent frames   = {latent_frames}", flush=True)

    if components["transformer"]:
        _check_artifact_shape(
            Path(args.transformer_dir),
            {"height": args.height, "width": args.width, "num_frames": latent_frames},
            "transformer",
        )
    if components["transformer_2"]:
        _check_artifact_shape(
            Path(args.transformer_2_dir),
            {"height": args.height, "width": args.width, "num_frames": latent_frames},
            "transformer_2",
        )
    if components["vae"]:
        _check_artifact_shape(
            Path(args.vae_dir),
            {"height": args.height, "width": args.width, "num_frames": args.num_frames},
            "vae_decoder",
        )

    _ensure_weights(model_dir, repo_id, components, args)
    compiled_dir = Path(args.compiled_dir)
    _stage_components(compiled_dir, components, args)

    parallel = NovaParallelConfig(tp_degree=args.tp_degree, cp_degree=args.cp_degree)
    app = NeuronWanApplication(
        model_path=str(model_dir),
        parallel=parallel,
        dtype=torch.bfloat16,
        shape={"height": args.height, "width": args.width, "num_frames": args.num_frames},
        text_seq_len=args.text_seq_len,
        batch_size=1,
        enable_text_encoder=components["text"],
        enable_transformer=components["transformer"],
        enable_transformer_2=components["transformer_2"],
        enable_vae_decoder=components["vae"],
        teacache_calibration_path=args.teacache_calibration,
    )

    t0 = time.monotonic()
    app.load(
        str(compiled_dir),
        start_rank_id=0,
        local_ranks_size=parallel.world_size,
        skip_warmup=args.skip_warmup,
    )
    print(f"[wan] load elapsed    = {time.monotonic() - t0:.3f}s", flush=True)

    torch.manual_seed(args.seed)
    if has_latents:
        latents = torch.load(args.load_latents, map_location="cpu")
        if latents.dtype != torch.bfloat16:
            latents = latents.to(torch.bfloat16)
        print(f"[wan] loaded latents  = {args.load_latents}", flush=True)
    else:
        latents = torch.randn(
            1, 16, latent_frames, args.height // 8, args.width // 8,
            dtype=torch.bfloat16,
        ) * 0.1

    call_kwargs: dict = {
        "latents": latents,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "output_type": args.output_type,
    }
    if components["text"]:
        if args.prompt is None:
            call_kwargs["input_ids"] = torch.zeros((1, args.text_seq_len), dtype=torch.int64)
            call_kwargs["attention_mask"] = torch.ones((1, args.text_seq_len), dtype=torch.int32)
        else:
            call_kwargs["prompt"] = args.prompt
            if args.negative_prompt:
                call_kwargs["negative_prompt"] = args.negative_prompt
    elif components["transformer"]:
        text_dim = int(getattr(app.transformer.config, "text_dim", 4096))
        call_kwargs["prompt_embeds"] = torch.zeros((1, args.text_seq_len, text_dim), dtype=torch.bfloat16)

    t1 = time.monotonic()
    out = app(**call_kwargs)
    print(f"[wan] forward elapsed = {time.monotonic() - t1:.3f}s", flush=True)
    frames = out.frames if hasattr(out, "frames") else out[0]
    print(f"[wan] output shape    = {tuple(frames.shape)}", flush=True)
    print(f"[wan] output dtype    = {frames.dtype}", flush=True)
    print(f"[wan] output mean     = {float(frames.float().mean()):.10g}", flush=True)
    print(f"[wan] output std      = {float(frames.float().std()):.10g}", flush=True)

    if args.save_latents:
        latents_to_save = out.latents if hasattr(out, "latents") else frames
        Path(args.save_latents).parent.mkdir(parents=True, exist_ok=True)
        torch.save(latents_to_save.detach().cpu(), args.save_latents)
        print(f"[wan] saved latents   = {args.save_latents}", flush=True)

    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix == ".mp4" and args.output_type == "pt" and components["vae"]:
            if not _save_video(frames, str(target)):
                pt_target = target.with_suffix(".pt")
                torch.save(frames.detach().cpu(), pt_target)
                print(f"[wan] tensor saved to {pt_target} (mp4 export skipped)", flush=True)
        else:
            torch.save(frames.detach().cpu(), target)
            print(f"[wan] tensor saved to {target}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
