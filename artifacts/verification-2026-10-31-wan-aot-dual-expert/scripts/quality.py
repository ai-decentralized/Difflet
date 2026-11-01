"""Quality of the AOT latents against the CPU bf16 diffusers references.

Every latents tensor (AOT device runs and CPU references alike) is decoded by the same fp32
diffusers VAE on the host (un-normalise, decode, [0, 1]), then compared with
scripts/quality_vs_reference.py's metrics (latent cosine / rel-L2; uint8 per-frame PSNR and
Gaussian SSIM). Run after the device and CPU jobs (it uses the CPU).

    python quality.py <snapshot dir> <campaign dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/ubuntu/Difflet/.claude/worktrees/wan-aot-parity-plan/scripts")
from quality_vs_reference import frame_metrics, latent_metrics  # noqa: E402

snapshot, campaign = Path(sys.argv[1]), Path(sys.argv[2])
from diffusers import AutoencoderKLWan  # noqa: E402

vae = AutoencoderKLWan.from_pretrained(snapshot / "vae", torch_dtype=torch.float32).eval()
mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)


def frames(latents: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        video = vae.decode(latents.float() * std + mean, return_dict=False)[0]  # [B, C, F, H, W]
    video = ((video.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)
    return video[0].permute(1, 2, 3, 0).numpy()  # [F, H, W, 3]


def load(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu").float()


pairs = {
    # the gate: AOT dual expert vs diffusers dual expert
    "aot_dual_vs_ref_dual": ("work/dual_gen1/latents.pt", "ref/diffusers_bf16_wan_dual_s42_latents.pt"),
    # calibration: AOT single expert vs diffusers single expert
    "aot_single_vs_ref_single": ("work/br_gen1/latents.pt", "ref/diffusers_bf16_wan_s42_latents.pt"),
    # sanity: a wrong-expert run must be far from the dual reference
    "aot_single_vs_ref_dual": ("work/br_gen1/latents.pt", "ref/diffusers_bf16_wan_dual_s42_latents.pt"),
}
decoded, report = {}, {}
for name, (cand, ref) in pairs.items():
    cand_path, ref_path = campaign / cand, campaign / ref
    if not (cand_path.exists() and ref_path.exists()):
        report[name] = "missing input"
        continue
    c, r = load(cand_path), load(ref_path)
    for key, value in ((cand, c), (ref, r)):
        if key not in decoded:
            decoded[key] = frames(value)
    metrics = {**latent_metrics(r, c), **frame_metrics(decoded[ref], decoded[cand])}
    report[name] = {k: (round(v, 4) if isinstance(v, float) else [round(x, 2) for x in v])
                    for k, v in metrics.items()}
    print(name, json.dumps(report[name]), flush=True)
(campaign / "logs" / "quality.json").write_text(json.dumps(report, indent=1))
