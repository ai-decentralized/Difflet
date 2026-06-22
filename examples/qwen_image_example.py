"""Qwen-Image text-to-image inference on Trainium via Difflet (fully on-device).

Qwen2.5-VL text encoder, the DiT, and the VAE all run on Trainium. Capacity forces
staging (the 7B Qwen2.5-VL encoder and the 60-layer DiT cannot co-fit on 4 cores, and the
VAE runs tp=1), so each runs as its own process and passes tensors through --work-dir files
(mirroring examples/hunyuan_video_example.py and examples/wan_example.py).

Stage 1 -- Qwen2.5-VL prompt embeddings (4 NeuronCores, TP=4):

    NEURON_RT_NUM_CORES=4 NEURON_RT_VIRTUAL_CORE_SIZE=2 \\
    python examples/qwen_image_example.py --stage text \\
        --prompt "a small red cabin beside a lake, crisp morning light"

Stage 2 -- DiT denoise (TP=4) -> packed latents (4 NeuronCores):

    NEURON_RT_NUM_CORES=4 NEURON_RT_VIRTUAL_CORE_SIZE=2 \\
    python examples/qwen_image_example.py --stage generate --num-inference-steps 4

Stage 3 -- VAE decode -> image (1 NeuronCore):

    NEURON_RT_NUM_CORES=1 python examples/qwen_image_example.py --stage vae \\
        --output /tmp/qwen.png

The Qwen2.5-VL encoder NEFF compiles on first run; the DiT compiles into --dit-cache (a
content-addressed Difflet compile cache); the VAE reuses Difflet's Wan VAE decoder port (the
Qwen-Image VAE config is identical to Wan's).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import torch

_DEFAULT_CACHE = Path(__file__).resolve().parent.parent / ".difflet-cache"

# Qwen-Image Llama/Qwen2.5-VL prompt template (diffusers prompt_template_encode).
QWEN_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
QWEN_DROP_IDX = 34  # prompt_template_encode_start_idx


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=["text", "generate", "vae"])
    p.add_argument("--prompt", default="a small red cabin beside a lake, crisp morning light")
    p.add_argument("--model-dir", default=None, help="Qwen/Qwen-Image snapshot dir. Defaults to local.")
    p.add_argument("--enc-compiled", default=str(_DEFAULT_CACHE / "qwen_qwen25vl_enc"))
    p.add_argument(
        "--dit-compiled",
        default=str(_DEFAULT_CACHE / "qwen_ondevice" / "dit"),
        help="Qwen-Image DiT NEFF parent dir (contains transformer/). "
        "Compiled here on first run if missing.",
    )
    p.add_argument("--vae-compiled", default=str(_DEFAULT_CACHE / "qwen_vae_dec"))
    p.add_argument("--work-dir", default="/tmp/qwen_example")
    p.add_argument("--output", default="/tmp/qwen.png")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--text-seq-len", type=int, default=1024)
    p.add_argument("--enc-seq", type=int, default=256, help="device bucket for the templated prompt")
    p.add_argument("--num-inference-steps", type=int, default=4)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tp-degree", type=int, default=4)
    p.add_argument(
        "--cp-degree",
        type=int,
        default=1,
        help="Context-parallel degree (1 = disabled). world_size = tp_degree * cp_degree.",
    )
    return p.parse_args()


def _model_dir(model_dir: str | None) -> str:
    if model_dir:
        return model_dir
    hits = glob.glob(
        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen-Image/snapshots/*")
    )
    if not hits:
        raise SystemExit("Pass --model-dir; no local Qwen-Image snapshot found.")
    return hits[0]


def stage_text(args: argparse.Namespace) -> None:
    from transformers import AutoConfig, AutoTokenizer

    from neuronx_distributed_inference.models.config import NeuronConfig, TensorCaptureConfig
    from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_text import (
        NeuronQwen2VLTextForCausalLM,
    )
    from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

    model_dir = _model_dir(args.model_dir)
    enc_path = os.path.join(model_dir, "text_encoder")

    text_cfg = AutoConfig.from_pretrained(enc_path).text_config
    if getattr(text_cfg, "pad_token_id", None) is None:
        text_cfg.pad_token_id = 0

    neuron_config = NeuronConfig(
        tp_degree=args.tp_degree,
        batch_size=1,
        seq_len=args.enc_seq,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config={},
        tensor_capture_config=TensorCaptureConfig(modules_to_capture=["norm"]),
    )
    config = NeuronQwen2VLTextForCausalLM.get_config_cls()(
        neuron_config, load_config=load_pretrained_config(hf_config=text_cfg)
    )
    app = NeuronQwen2VLTextForCausalLM(enc_path, config)
    if not os.path.exists(os.path.join(args.enc_compiled, "model.pt")):
        print("[text] compiling Qwen2.5-VL encoder ...", flush=True)
        app.compile(args.enc_compiled)
    app.load(args.enc_compiled)

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    ti = tok(
        QWEN_TEMPLATE.format(args.prompt),
        max_length=args.enc_seq,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_ids = ti.input_ids.to(torch.int32)
    attn = ti.attention_mask.to(torch.int32)
    valid = int(attn.sum())
    out = app(
        input_ids=input_ids,
        attention_mask=attn,
        position_ids=torch.arange(args.enc_seq, dtype=torch.int32).unsqueeze(0),
        sampling_params=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
    )
    hs = out.captured_tensors[0].float()  # (1, enc_seq, 3584)
    dev = hs[:, QWEN_DROP_IDX:valid]  # valid prompt tokens after template drop
    # pad to text_seq_len
    seq = dev.shape[1]
    ehs = torch.zeros(1, args.text_seq_len, dev.shape[-1], dtype=torch.bfloat16)
    ehs[:, :seq] = dev.to(torch.bfloat16)
    mask = torch.zeros(1, args.text_seq_len, dtype=torch.bool)
    mask[:, :seq] = True

    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    torch.save(
        {"encoder_hidden_states": ehs, "encoder_hidden_states_mask": mask},
        os.path.join(args.work_dir, "text.pt"),
    )
    print(f"[text] encoder_hidden_states {tuple(ehs.shape)} valid={seq} -> {args.work_dir}/text.pt", flush=True)


def stage_generate(args: argparse.Namespace) -> None:
    os.environ.setdefault("DIFFLET_BACKEND", "trainium")
    from difflet.models.qwen_image.application import NeuronQwenImageApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    model_dir = _model_dir(args.model_dir)
    text = torch.load(os.path.join(args.work_dir, "text.pt"))
    guidance = torch.full([1], float(args.guidance_scale), dtype=torch.bfloat16)

    app = NeuronQwenImageApplication(
        model_path=model_dir,
        parallel=DiffletParallelConfig(tp_degree=args.tp_degree, cp_degree=args.cp_degree),
        dtype=torch.bfloat16,
        shape={"height": args.height, "width": args.width, "num_frames": None},
        text_seq_len=args.text_seq_len,
        enable_transformer=True,
    )
    if not app.has_compiled_artifacts(args.dit_compiled):
        print(f"[generate] compiling DiT into {args.dit_compiled} (first run) ...", flush=True)
        tc = time.time()
        app.compile(args.dit_compiled)
        print(f"[generate] DiT compile = {time.time() - tc:.1f}s", flush=True)
    t0 = time.time()
    app.load(args.dit_compiled, skip_warmup=True)
    print(f"[generate] DiT load = {time.time() - t0:.1f}s", flush=True)

    # Qwen scheduler uses dynamic shifting -> set_timesteps needs mu(image_seq_len).
    import numpy as np

    sched = app.pipeline.scheduler
    sc = sched.config
    image_seq_len = (args.height // 16) * (args.width // 16)
    slope = (sc.max_shift - sc.base_shift) / (sc.max_image_seq_len - sc.base_image_seq_len)
    mu = image_seq_len * slope + (sc.base_shift - slope * sc.base_image_seq_len)
    sigmas = np.linspace(1.0, 1.0 / args.num_inference_steps, args.num_inference_steps).tolist()
    sched.set_timesteps(sigmas=sigmas, mu=mu, device="cpu")
    timesteps = sched.timesteps

    torch.manual_seed(args.seed)
    out = app.pipeline(
        encoder_hidden_states=text["encoder_hidden_states"],
        encoder_hidden_states_mask=text["encoder_hidden_states_mask"],
        guidance=guidance,
        timesteps=timesteps,
        num_inference_steps=args.num_inference_steps,
        output_type="latent",
    )
    packed = out.latents.cpu()
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    torch.save(packed, os.path.join(args.work_dir, "latents.pt"))
    print(f"[generate] packed latents {tuple(packed.shape)} -> {args.work_dir}/latents.pt", flush=True)


def stage_vae(args: argparse.Namespace) -> None:
    from difflet.backends.trainium.core.config import NeuronConfig
    from difflet.backends.trainium.wan.vae import (
        NeuronWanVAEDecoderApplication,
        WanVAEDecoderInferenceConfig,
    )
    from difflet.utils.diffusers_adapter import load_diffusers_config

    model_dir = _model_dir(args.model_dir)
    vae_path = os.path.join(model_dir, "vae")
    packed = torch.load(os.path.join(args.work_dir, "latents.pt")).float()

    # unpack (1, seq, 64) -> (1, 16, 1, H/8, W/8)
    b, seq, _ = packed.shape
    hh = ww = int(seq**0.5)
    z = packed.view(b, hh, ww, 16, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(b, 16, hh * 2, ww * 2)
    z = z.unsqueeze(2)

    config = WanVAEDecoderInferenceConfig(
        neuron_config=NeuronConfig(tp_degree=1, world_size=1, torch_dtype=torch.bfloat16),
        load_config=load_diffusers_config(vae_path),
        height=args.height,
        width=args.width,
        num_frames=1,
    )
    # apply Qwen latent normalization (latents_mean/std) before decode
    mean = torch.tensor(config.latents_mean).view(1, -1, 1, 1, 1)
    std = torch.tensor(config.latents_std).view(1, -1, 1, 1, 1)
    z = (z * std + mean).to(torch.bfloat16) if len(config.latents_mean) else z.to(torch.bfloat16)

    app = NeuronWanVAEDecoderApplication(model_path=vae_path, config=config)
    if not os.path.exists(os.path.join(args.vae_compiled, "model.pt")):
        print("[vae] compiling ...", flush=True)
        app.compile(args.vae_compiled)
    app.load(args.vae_compiled)

    img = app(z)
    img = (img[0] if isinstance(img, (tuple, list)) else img).float().cpu()
    img = img[:, :, 0]  # drop temporal frame -> (1, 3, H, W)
    print(f"[vae] image {tuple(img.shape)} finite={bool(torch.isfinite(img).all())} "
          f"mean/std={img.mean():.3f}/{img.std():.3f}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(img, out_path.with_suffix(".pt"))
    try:
        from torchvision.utils import save_image

        save_image((img[0] * 0.5 + 0.5).clamp(0, 1), str(out_path))
        print(f"[vae] saved image -> {out_path}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[vae] saved tensor -> {out_path.with_suffix('.pt')} (png skipped: {exc})", flush=True)


def main() -> int:
    args = _parse_args()
    {"text": stage_text, "generate": stage_generate, "vae": stage_vae}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
