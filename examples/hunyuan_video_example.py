"""HunyuanVideo text-to-video inference on Trainium via Nova (fully on-device text encode).

Unlike the M3 v0 hybrid path (host HF text encoders), this example runs Llama-3 and CLIP
on Trainium. Capacity forces staging: the 8B Llama encoder and the 13B DiT cannot co-fit on
one 4-core card, and CLIP's tp=1 differs from Llama's tp=4 world_size -- so each runs as its
own process and passes tensors through ``--work-dir`` files (mirroring examples/wan_example.py).

Stage 1 -- CLIP pooled projections (1 NeuronCore):

    NEURON_RT_NUM_CORES=1 python examples/hunyuan_video_example.py --stage clip \\
        --prompt "a cat walking in a sunlit garden"

Stage 2 -- Llama-3 prompt embeddings (4 NeuronCores, TP=4):

    NEURON_RT_NUM_CORES=4 NEURON_RT_VIRTUAL_CORE_SIZE=2 \\
    python examples/hunyuan_video_example.py --stage llama \\
        --prompt "a cat walking in a sunlit garden"

Stage 3 -- DiT denoise (TP=4) + on-device VAE decode -> video (4 NeuronCores):

    NEURON_RT_NUM_CORES=4 NEURON_RT_VIRTUAL_CORE_SIZE=2 \\
    python examples/hunyuan_video_example.py --stage generate \\
        --num-inference-steps 4 --output /tmp/hunyuan.mp4

This is now a fully on-device pipeline: Llama-3 + CLIP + DiT + VAE all run on Trainium.
The DiT artifact must be pre-compiled and --dit-compiled must point at a parent dir that
contains BOTH ``transformer/`` (masked-SDPA NEFF) and ``vae_decoder/`` (the Nova Trainium
VAE NEFF), so the VAE decode also runs on device. Pass ``--cpu-vae`` to fall back to the
HF CPU VAE decoder (the old hybrid behaviour). The Llama and CLIP NEFFs compile on first
run into --llama-compiled / --clip-compiled.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_DEFAULT_CACHE = Path(__file__).resolve().parent.parent / ".nova-cache"

# HunyuanVideo Llama prompt template (diffusers DEFAULT_PROMPT_TEMPLATE, crop_start=95).
LLAMA_TEMPLATE = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the "
    "following aspects: 1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)
LLAMA_CROP_START = 95
# hidden_states[-(num_hidden_layers_to_skip + 1)] with skip=2 -> output of decoder layer 29.
LLAMA_CAPTURE = "layers.29"
LLAMA_HIDDEN_LAYERS_TO_SKIP = 2


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=["clip", "llama", "generate"])
    p.add_argument("--prompt", default="a cat walking in a sunlit garden")
    p.add_argument(
        "--model-dir",
        default=None,
        help="HF snapshot dir with text_encoder/ text_encoder_2/ tokenizer/ tokenizer_2/. "
        "Defaults to the local hunyuanvideo-community/HunyuanVideo snapshot.",
    )
    p.add_argument(
        "--dit-source",
        default=str(_DEFAULT_CACHE / "hunyuan_n4_20d40s2r" / "source"),
        help="Dir with transformer/ (real weights) + vae/ + scheduler/ for the DiT stage.",
    )
    p.add_argument(
        "--dit-compiled",
        default=str(_DEFAULT_CACHE / "hunyuan_sdpa_20d40s2r" / "compiled"),
        help="Pre-compiled masked-SDPA DiT NEFF parent dir (contains transformer/).",
    )
    p.add_argument("--llama-compiled", default=str(_DEFAULT_CACHE / "hunyuan_llama_enc_l29_351"))
    p.add_argument("--clip-compiled", default=str(_DEFAULT_CACHE / "hunyuan_clip_enc"))
    p.add_argument("--work-dir", default="/tmp/hunyuan_example")
    p.add_argument("--output", default="/tmp/hunyuan.mp4")
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num-frames", type=int, default=61)
    p.add_argument("--num-inference-steps", type=int, default=4)
    p.add_argument("--text-seq-len", type=int, default=256)
    p.add_argument("--guidance-scale", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tp-degree", type=int, default=4)
    p.add_argument(
        "--cpu-vae",
        action="store_true",
        help="Decode the VAE on the HF CPU reference instead of the Trainium vae_decoder/ NEFF "
        "(falls back to the old hybrid behaviour; the pipeline is otherwise fully on-device).",
    )
    return p.parse_args()


def _resolve_model_dir(model_dir: str | None) -> str:
    if model_dir:
        return model_dir
    import glob

    hits = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/"
            "models--hunyuanvideo-community--HunyuanVideo/snapshots/*"
        )
    )
    if not hits:
        raise SystemExit("Pass --model-dir; no local HunyuanVideo snapshot found.")
    return hits[0]


def stage_clip(args: argparse.Namespace) -> None:
    from transformers import CLIPTokenizer

    from nova.backends.trainium.core.config import NeuronConfig
    from nova.models.flux.clip.modeling_clip import (
        CLIPInferenceConfig,
        NeuronClipApplication,
    )
    from nova.utils.diffusers_adapter import load_diffusers_config

    model_dir = _resolve_model_dir(args.model_dir)
    clip_path = os.path.join(model_dir, "text_encoder_2")

    config = CLIPInferenceConfig(
        neuron_config=NeuronConfig(tp_degree=1, world_size=1, torch_dtype=torch.bfloat16),
        load_config=load_diffusers_config(clip_path),
    )
    # HunyuanVideo's CLIP config.json omits these HF-runtime flags the modeling reads.
    for key, val in {
        "output_attentions": False,
        "output_hidden_states": False,
        "use_return_dict": True,
    }.items():
        setattr(config, key, val)

    app = NeuronClipApplication(model_path=clip_path, config=config)
    if not os.path.exists(os.path.join(args.clip_compiled, "model.pt")):
        print("[clip] compiling ...", flush=True)
        app.compile(args.clip_compiled)
    app.load(args.clip_compiled)

    tok = CLIPTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer_2"))
    ids = tok(
        args.prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    ).input_ids.to(torch.int64)
    out = app(ids)
    pooled = out.pooler_output.to(torch.bfloat16).cpu().reshape(1, -1)

    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    torch.save({"pooled_projections": pooled}, os.path.join(args.work_dir, "clip.pt"))
    print(f"[clip] pooled_projections {tuple(pooled.shape)} -> {args.work_dir}/clip.pt", flush=True)


def stage_llama(args: argparse.Namespace) -> None:
    from transformers import AutoConfig, AutoTokenizer

    from neuronx_distributed_inference.models.config import (
        NeuronConfig,
        TensorCaptureConfig,
    )
    from neuronx_distributed_inference.models.llama.modeling_llama import (
        NeuronLlamaForCausalLM,
    )
    from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

    model_dir = _resolve_model_dir(args.model_dir)
    enc_path = os.path.join(model_dir, "text_encoder")
    seq = args.text_seq_len + LLAMA_CROP_START

    hf_cfg = AutoConfig.from_pretrained(enc_path)
    if hf_cfg.pad_token_id is None:
        hf_cfg.pad_token_id = 0
    hf_cfg.tie_word_embeddings = True  # checkpoint is bare LlamaModel; satisfy the loader

    neuron_config = NeuronConfig(
        tp_degree=args.tp_degree,
        batch_size=1,
        seq_len=seq,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config={},
        tensor_capture_config=TensorCaptureConfig(modules_to_capture=[LLAMA_CAPTURE]),
    )
    config = NeuronLlamaForCausalLM.get_config_cls()(
        neuron_config, load_config=load_pretrained_config(hf_config=hf_cfg)
    )
    app = NeuronLlamaForCausalLM(enc_path, config)
    if not os.path.exists(os.path.join(args.llama_compiled, "model.pt")):
        print(f"[llama] compiling @seq={seq} capturing {LLAMA_CAPTURE} ...", flush=True)
        app.compile(args.llama_compiled)
    app.load(args.llama_compiled)

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    text_inputs = tok(
        LLAMA_TEMPLATE.format(args.prompt),
        max_length=seq,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_ids = text_inputs.input_ids.to(torch.int32)
    attn = text_inputs.attention_mask.to(torch.int32)
    position_ids = torch.arange(seq, dtype=torch.int32).unsqueeze(0)

    out = app(
        input_ids=input_ids,
        attention_mask=attn,
        position_ids=position_ids,
        sampling_params=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
    )
    hidden = out.captured_tensors[0][:, LLAMA_CROP_START :].to(torch.bfloat16).cpu()
    mask = attn[:, LLAMA_CROP_START :].to(torch.int64)

    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    torch.save(
        {"encoder_hidden_states": hidden, "encoder_attention_mask": mask},
        os.path.join(args.work_dir, "llama.pt"),
    )
    print(
        f"[llama] encoder_hidden_states {tuple(hidden.shape)} valid={int(mask.sum())} "
        f"-> {args.work_dir}/llama.pt",
        flush=True,
    )


def stage_generate(args: argparse.Namespace) -> None:
    from nova.models.hunyuan_video.application import (
        HunyuanVideoDiTInputBundle,
        NeuronHunyuanVideoApplication,
    )
    from nova.pipeline.parallel_config import NovaParallelConfig

    llama = torch.load(os.path.join(args.work_dir, "llama.pt"))
    clip = torch.load(os.path.join(args.work_dir, "clip.pt"))

    latent_frames = (args.num_frames - 1) // 4 + 1
    torch.manual_seed(args.seed)
    latents = torch.randn(
        1, 16, latent_frames, args.height // 8, args.width // 8, dtype=torch.bfloat16
    )
    guidance = torch.full([1], args.guidance_scale * 1000.0, dtype=torch.bfloat16)

    app = NeuronHunyuanVideoApplication(
        model_path=args.dit_source,
        parallel=NovaParallelConfig(tp_degree=args.tp_degree, cp_degree=1),
        dtype=torch.bfloat16,
        shape={"height": args.height, "width": args.width, "num_frames": args.num_frames},
        text_seq_len=args.text_seq_len,
        enable_vae_decoder=not args.cpu_vae,
    )
    app.teacache_probe = None  # not running teacache; no probe NEFF is compiled
    t0 = time.monotonic()
    app.load(args.dit_compiled, skip_warmup=True)
    print(f"[generate] DiT load = {time.monotonic() - t0:.1f}s", flush=True)

    timesteps = _hunyuan_timesteps(app, args.num_inference_steps, latents.device)
    bundle = HunyuanVideoDiTInputBundle(
        hidden_states=latents,
        timestep=timesteps[:1].clone(),
        encoder_hidden_states=llama["encoder_hidden_states"],
        encoder_attention_mask=llama["encoder_attention_mask"].to(torch.int64),
        pooled_projections=clip["pooled_projections"],
        guidance=guidance,
    )
    t0 = time.monotonic()
    output = app(
        bundle=bundle,
        timesteps=timesteps,
        num_inference_steps=args.num_inference_steps,
        output_type="pt",
        return_trajectory=False,
    )
    print(f"[generate] denoise+decode = {time.monotonic() - t0:.1f}s", flush=True)

    frames = output.frames
    print(
        f"[generate] video {tuple(frames.shape)} finite={bool(torch.isfinite(frames).all())} "
        f"mean/std={frames.float().mean():.3f}/{frames.float().std():.3f}",
        flush=True,
    )
    out_path = Path(args.output)
    tensor_path = out_path.with_suffix(".pt")
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(frames.cpu(), tensor_path)
    print(f"[generate] saved video tensor -> {tensor_path}", flush=True)


def _hunyuan_timesteps(app, num_steps: int, device) -> torch.Tensor:
    import numpy as np

    from nova.models.hunyuan_video.pipeline import _retrieve_timesteps

    sigmas = np.linspace(1.0, 0.0, num_steps + 1)[:-1]
    timesteps, _ = _retrieve_timesteps(app.pipeline.scheduler, num_steps, "cpu", sigmas=sigmas)
    return timesteps.to(device=device)


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("NOVA_BACKEND", "trainium")
    if args.stage == "clip":
        stage_clip(args)
    elif args.stage == "llama":
        stage_llama(args)
    else:
        stage_generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
