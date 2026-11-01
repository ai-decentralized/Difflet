"""Frames 0, 1 and 8 of: diffusers fp32 dual (row 1), diffusers bf16 dual (row 2), AOT dual
(row 3), every one decoded from its latents by the same fp32 VAE -> PNG."""
import sys

import numpy as np
import torch
from diffusers import AutoencoderKLWan
from PIL import Image

C = "/home/ubuntu/campaigns/wan-dual-2026-10-31"
SNAP = ("/home/ubuntu/.cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/"
        "5be7df9619b54f4e2667b2755bc6a756675b5cd7")
vae = AutoencoderKLWan.from_pretrained(f"{SNAP}/vae", torch_dtype=torch.float32).eval()
mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
rows = []
for path in ("ref/diffusers_fp32_wan_dual_s42_latents.pt", "ref/diffusers_bf16_wan_dual_s42_latents.pt",
             "work/dual_gen1/latents.pt"):
    z = torch.load(f"{C}/{path}", map_location="cpu").float()
    with torch.no_grad():
        video = vae.decode(z * std + mean, return_dict=False)[0]
    frames = ((video.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)[0].permute(1, 2, 3, 0).numpy()
    rows.append(np.concatenate([frames[i] for i in (0, 1, 8)], axis=1))
Image.fromarray(np.concatenate(rows, axis=0)).resize((3 * 416, 3 * 240)).save(sys.argv[1])
print(sys.argv[1])
