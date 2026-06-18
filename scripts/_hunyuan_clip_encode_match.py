"""M2: on-device CLIP (text_encoder_2) -> pooled_projections, vs HF bundle.

HF _get_clip_prompt_embeds: tokenize RAW prompt with tokenizer_2 (CLIP, max_length=77),
CLIPTextModel(...).pooler_output -> (1, 768). We reuse Nova's Flux CLIP port
(NeuronClipApplication) pointed at HunyuanVideo's text_encoder_2.
"""

import glob
import json
import os
import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import CLIPTokenizer

from nova.backends.trainium.core.config import NeuronConfig
from nova.models.flux.clip.modeling_clip import CLIPInferenceConfig, NeuronClipApplication
from nova.utils.diffusers_adapter import load_diffusers_config

SNAP = glob.glob(
    "/home/ubuntu/.cache/huggingface/hub/"
    "models--hunyuanvideo-community--HunyuanVideo/snapshots/*"
)[0]
CLIP = f"{SNAP}/text_encoder_2"
TOK2 = f"{SNAP}/tokenizer_2"
OUT = "/home/ubuntu/nova/.nova-cache/hunyuan_clip_enc"
BUNDLE = "/home/ubuntu/nova/.nova-cache/hunyuan_dit_inputs/cat_walking_4step.safetensors"

meta = json.loads(open(BUNDLE + ".meta.json").read())
prompt = meta["prompt"][0]
print(f"[m2] prompt={prompt!r}", flush=True)

neuron_config = NeuronConfig(tp_degree=1, world_size=1, torch_dtype=torch.bfloat16)
clip_config = CLIPInferenceConfig(neuron_config=neuron_config, load_config=load_diffusers_config(CLIP))
# HunyuanVideo's CLIP config.json omits these HF-runtime flags the modeling reads.
for _k, _v in {"output_attentions": False, "output_hidden_states": False, "use_return_dict": True}.items():
    setattr(clip_config, _k, _v)
app = NeuronClipApplication(model_path=CLIP, config=clip_config)
if not os.path.exists(os.path.join(OUT, "model.pt")):
    print("[m2] compiling CLIP ...", flush=True)
    t0 = time.time(); app.compile(OUT); print(f"[m2] compiled {time.time()-t0:.1f}s", flush=True)
app.load(OUT)

tok = CLIPTokenizer.from_pretrained(TOK2)
ti = tok(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
input_ids = ti.input_ids.to(torch.int64)
print(f"[m2] input_ids {tuple(input_ids.shape)}", flush=True)

out = app(input_ids)
pooled = getattr(out, "pooler_output", None)
if pooled is None and isinstance(out, (tuple, list)):
    pooled = out[1] if len(out) > 1 else out[0]
if pooled is None:
    print("[m2] out type:", type(out), "attrs:", [a for a in dir(out) if not a.startswith("_")][:20], flush=True)
    raise SystemExit("no pooler_output")
pooled = pooled.float().cpu().reshape(1, -1)
print(f"[m2] device pooler_output {tuple(pooled.shape)}", flush=True)

ref = load_file(BUNDLE)["pooled_projections"].float().reshape(1, -1)
print(f"[m2] bundle pooled_projections {tuple(ref.shape)}", flush=True)
cos = F.cosine_similarity(pooled.reshape(-1), ref.reshape(-1), dim=0).item()
print(f"\n=== M2 pooled_projections cosine(device vs HF bundle) = {cos:.6f} ===", flush=True)
print("RESULT:", "PASS" if cos >= 0.999 else "FAIL", flush=True)

torch.save({"pooled_projections": pooled.to(torch.bfloat16)}, "/tmp/hy_dev_clip.pt")
print("[m2] saved -> /tmp/hy_dev_clip.pt", flush=True)
