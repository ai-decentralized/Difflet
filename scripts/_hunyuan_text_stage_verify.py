"""M3: verify a DEVICE-encoded bundle drives the DiT to the same latents as the HF bundle.

Builds a bundle = HF bundle with encoder_hidden_states / encoder_attention_mask /
pooled_projections replaced by the on-device Llama+CLIP outputs (saved by M1/M2), keeping
the identical noise/timesteps/guidance. Runs the DiT and compares the final latent to the
HF-bundle DiT run (/tmp/hy_e2e_latents.pt).
"""

import json
import os

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from nova.models.hunyuan_video.application import (
    HunyuanVideoDiTInputBundle,
    NeuronHunyuanVideoApplication,
)
from nova.pipeline.parallel_config import NovaParallelConfig

ROOT = "/home/ubuntu/nova"
SOURCE = f"{ROOT}/.nova-cache/hunyuan_n4_20d40s2r/source"
COMPILED = f"{ROOT}/.nova-cache/hunyuan_sdpa_20d40s2r/compiled"
BUNDLE = f"{ROOT}/.nova-cache/hunyuan_dit_inputs/cat_walking_4step.safetensors"

meta = json.loads(open(BUNDLE + ".meta.json").read())
hf = load_file(BUNDLE)
llama = torch.load("/tmp/hy_dev_llama.pt")
clip = torch.load("/tmp/hy_dev_clip.pt")

# device-encoded inputs; identical noise/timesteps/guidance from HF bundle
ehs = llama["encoder_hidden_states"]
emask = llama["encoder_attention_mask"]
pooled = clip["pooled_projections"]
print(f"[m3] device ehs{tuple(ehs.shape)} mask{tuple(emask.shape)} pooled{tuple(pooled.shape)}", flush=True)
print(f"[m3] mask valid tokens (device)={int(emask.sum())} vs HF bundle={int(hf['encoder_attention_mask'].sum())}", flush=True)

app = NeuronHunyuanVideoApplication(
    model_path=SOURCE,
    parallel=NovaParallelConfig(tp_degree=4, cp_degree=1),
    dtype=torch.bfloat16,
    shape={"height": meta["height"], "width": meta["width"], "num_frames": meta["num_frames"]},
    text_seq_len=meta["text_seq_len"],
)
app.teacache_probe = None
app.load(COMPILED, skip_warmup=True)

bundle = HunyuanVideoDiTInputBundle(
    hidden_states=hf["latents_init"],
    timestep=hf["timesteps"][:1].clone(),
    encoder_hidden_states=ehs,
    encoder_attention_mask=emask.to(torch.int64),
    pooled_projections=pooled,
    guidance=hf["guidance"],
)
with torch.inference_mode():
    out = app.pipeline(bundle=bundle, timesteps=hf["timesteps"], output_type="latent", return_trajectory=False)
lat = out.latents.float().cpu()
print(f"[m3] device-bundle DiT latent {tuple(lat.shape)} std={lat.std():.4f}", flush=True)

ref = torch.load("/tmp/hy_e2e_latents.pt").float()  # HF-bundle DiT latents (sdpa run)
cos = F.cosine_similarity(lat.reshape(-1), ref.reshape(-1), dim=0).item()
print(f"\n=== M3 final-latent cosine(device-encoded bundle vs HF-bundle) = {cos:.6f} ===", flush=True)
print("RESULT:", "PASS" if cos >= 0.999 else "FAIL", flush=True)
