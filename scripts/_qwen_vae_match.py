"""Q2: on-device Qwen-Image VAE decode via Difflet's Wan VAE port, vs HF.

AutoencoderKLQwenImage is architecturally identical to AutoencoderKLWan (same config:
z_dim=16, base_dim=96, dim_mult=[1,2,4,4], temperal_downsample=[False,True,True]; same
decoder.* / post_quant_conv.* keys). So Difflet's NeuronWanVAEDecoderApplication loads the
Qwen VAE weights directly. We decode the same unpacked latent on device and via HF and
compare.
"""

import glob
import os
import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from difflet.backends.trainium.core.config import NeuronConfig
from difflet.backends.trainium.wan.vae import (
    NeuronWanVAEDecoderApplication,
    WanVAEDecoderInferenceConfig,
)
from difflet.utils.diffusers_adapter import load_diffusers_config

SNAP = glob.glob("/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen-Image/snapshots/*")[0]
VAE = f"{SNAP}/vae"
OUT = "/home/ubuntu/difflet/.difflet-cache/qwen_vae_dec"
BUNDLE = "/home/ubuntu/difflet/.difflet-cache/qwen_image_dit_inputs/full_1024_4step.safetensors"
H = W = 1024


def unpack(latents):  # (1, 4096, 64) -> (1, 16, 1, 128, 128)
    b, seq, _ = latents.shape
    hh = ww = int(seq**0.5)  # 64
    x = latents.view(b, hh, ww, 16, 2, 2).permute(0, 3, 1, 4, 2, 5)
    x = x.reshape(b, 16, hh * 2, ww * 2)
    return x.unsqueeze(2)  # add temporal frame


config = WanVAEDecoderInferenceConfig(
    neuron_config=NeuronConfig(tp_degree=1, world_size=1, torch_dtype=torch.bfloat16),
    load_config=load_diffusers_config(VAE),
    height=H,
    width=W,
    num_frames=1,
)
app = NeuronWanVAEDecoderApplication(model_path=VAE, config=config)
if not os.path.exists(os.path.join(OUT, "model.pt")):
    print("[q2] compiling Qwen VAE (Wan port) ...", flush=True)
    t0 = time.time()
    app.compile(OUT)
    print(f"[q2] compiled {time.time() - t0:.1f}s", flush=True)
app.load(OUT)

z = unpack(load_file(BUNDLE)["latents_init"].float()).to(torch.bfloat16)
print(f"[q2] latent z {tuple(z.shape)}", flush=True)

dev = app(z)
dev = (dev[0] if isinstance(dev, (tuple, list)) else dev).float().cpu()
print(f"[q2] device decode {tuple(dev.shape)}", flush=True)

# HF reference
from diffusers import AutoencoderKLQwenImage

hf = AutoencoderKLQwenImage.from_pretrained(VAE, torch_dtype=torch.float32).eval()
with torch.inference_mode():
    ref = hf.decode(z.float(), return_dict=False)[0]
ref = torch.clamp(ref, -1.0, 1.0).float().cpu()
print(f"[q2] HF decode {tuple(ref.shape)}", flush=True)

n = min(dev.numel(), ref.numel())
cos = F.cosine_similarity(dev.reshape(-1)[:n], ref.reshape(-1)[:n], dim=0).item()
print(f"\n=== Q2 VAE decode cosine(device vs HF) = {cos:.6f} ===", flush=True)
print("RESULT:", "PASS" if cos >= 0.99 else "FAIL", flush=True)
