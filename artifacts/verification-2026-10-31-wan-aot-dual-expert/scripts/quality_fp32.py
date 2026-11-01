"""Quality against the CPU fp32 diffusers dual-expert baseline.

Every latents tensor is decoded by the same fp32 diffusers VAE on the host, then compared with
scripts/quality_vs_reference.py's metrics (latent cosine / rel-L2; uint8 per-frame PSNR and
Gaussian SSIM).

    python quality_fp32.py <snapshot dir> <campaign dir>
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/home/ubuntu/Difflet/.claude/worktrees/wan-aot-parity-plan/scripts")
from quality_vs_reference import frame_metrics, latent_metrics  # noqa: E402

snapshot, campaign = Path(sys.argv[1]), Path(sys.argv[2])
from diffusers import AutoencoderKLWan  # noqa: E402

vae = AutoencoderKLWan.from_pretrained(snapshot / "vae", torch_dtype=torch.float32).eval()
mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)


def frames(latents):
    with torch.no_grad():
        video = vae.decode(latents.float() * std + mean, return_dict=False)[0]
    video = ((video.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)
    return video[0].permute(1, 2, 3, 0).numpy()


FP32 = "ref/diffusers_fp32_wan_dual_s42_latents.pt"
pairs = {
    "diffusers_bf16_dual_vs_fp32_dual": (FP32, "ref/diffusers_bf16_wan_dual_s42_latents.pt"),
    "aot_dual_vs_fp32_dual": (FP32, "work/dual_gen1/latents.pt"),
    "aot_single_vs_fp32_dual": (FP32, "work/br_gen1/latents.pt"),
    "aot_single_vs_aot_dual": ("work/dual_gen1/latents.pt", "work/br_gen1/latents.pt"),
}
decoded, report = {}, {}
for name, (ref, cand) in pairs.items():
    tensors = {key: torch.load(campaign / key, map_location="cpu").float() for key in (ref, cand)}
    for key, value in tensors.items():
        if key not in decoded:
            decoded[key] = frames(value)
    metrics = {**latent_metrics(tensors[ref], tensors[cand]),
               **frame_metrics(decoded[ref], decoded[cand])}
    report[name] = {k: (round(v, 4) if isinstance(v, float) else [round(x, 2) for x in v])
                    for k, v in metrics.items()}
    print(name, json.dumps(report[name]), flush=True)
(campaign / "logs" / "quality_fp32.json").write_text(json.dumps(report, indent=1))
