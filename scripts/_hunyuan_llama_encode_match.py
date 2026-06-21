"""M1: on-device Llama encode that reproduces HunyuanVideo's HF encode_prompt.

HF _get_llama_prompt_embeds: format prompt into DEFAULT_PROMPT_TEMPLATE, tokenize to
(256 + crop_start=95) = 351 tokens, take hidden_states[-3] (num_hidden_layers_to_skip=2
=> output of decoder layer 29), crop the first 95 -> (1,256,4096).

We capture `layers.29` on the device Llama (seq_len=351) and apply the same host-side
template/tokenize/crop, then compare to the cached bundle's encoder_hidden_states.
"""

import glob
import json
import os
import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer

from neuronx_distributed_inference.models.config import NeuronConfig, TensorCaptureConfig
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

SNAP = glob.glob(
    "/home/ubuntu/.cache/huggingface/hub/"
    "models--hunyuanvideo-community--HunyuanVideo/snapshots/*"
)[0]
ENC = f"{SNAP}/text_encoder"
TOK = f"{SNAP}/tokenizer"
OUT = "/home/ubuntu/difflet/.difflet-cache/hunyuan_llama_enc_l29_351"
BUNDLE = "/home/ubuntu/difflet/.difflet-cache/hunyuan_dit_inputs/cat_walking_4step.safetensors"

CROP_START = 95
TEXT_SEQ = 256
SEQ = TEXT_SEQ + CROP_START  # 351
CAPTURE = "layers.29"  # hidden_states[-3] for 32-layer Llama (skip last 2)
TEMPLATE = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)

meta = json.loads(open(BUNDLE + ".meta.json").read())
prompt = meta["prompt"][0]
print(f"[m1] prompt={prompt!r} seq={SEQ} capture={CAPTURE}", flush=True)

hf_cfg = AutoConfig.from_pretrained(ENC)
if hf_cfg.pad_token_id is None:
    hf_cfg.pad_token_id = 0
hf_cfg.tie_word_embeddings = True

nc = NeuronConfig(
    tp_degree=4, batch_size=1, seq_len=SEQ, torch_dtype=torch.bfloat16,
    on_device_sampling_config={},
    tensor_capture_config=TensorCaptureConfig(modules_to_capture=[CAPTURE]),
)
config = NeuronLlamaForCausalLM.get_config_cls()(nc, load_config=load_pretrained_config(hf_config=hf_cfg))
model = NeuronLlamaForCausalLM(ENC, config)
if not os.path.exists(os.path.join(OUT, "model.pt")):
    print("[m1] compiling @351 capturing layers.29 ...", flush=True)
    t0 = time.time(); model.compile(OUT); print(f"[m1] compiled {time.time()-t0:.1f}s", flush=True)
model.load(OUT)

# host: HF template + tokenize (exactly as _get_llama_prompt_embeds)
tok = AutoTokenizer.from_pretrained(TOK)
templated = TEMPLATE.format(prompt)
ti = tok(templated, max_length=SEQ, padding="max_length", truncation=True,
         return_tensors="pt", return_attention_mask=True)
input_ids = ti.input_ids.to(torch.int32)
attn = ti.attention_mask.to(torch.int32)
position_ids = torch.arange(SEQ, dtype=torch.int32).unsqueeze(0)
print(f"[m1] tokenized: input_ids{tuple(input_ids.shape)} valid={int(attn.sum())}", flush=True)

out = model(input_ids=input_ids, attention_mask=attn, position_ids=position_ids,
            sampling_params=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32))
cap = getattr(out, "captured_tensors", None)
print("[m1] captured:", None if not cap else [tuple(t.shape) for t in cap], flush=True)
if not cap:
    for n, _ in model.named_modules():
        if "29" in n:
            print("   module:", n, flush=True)
    raise SystemExit("no capture")

hs = cap[0].float().cpu()  # (1, 351, 4096)
hs_crop = hs[:, CROP_START:]  # (1, 256, 4096)
print(f"[m1] device hidden[-3] {tuple(hs.shape)} -> crop {tuple(hs_crop.shape)}", flush=True)

ref = load_file(BUNDLE)["encoder_hidden_states"].float()  # (1,256,4096), HF host
print(f"[m1] bundle encoder_hidden_states {tuple(ref.shape)}", flush=True)
cos = F.cosine_similarity(hs_crop.reshape(-1), ref.reshape(-1), dim=0).item()
# also per-token cosine over valid region
valid = int(attn.sum()) - CROP_START
pt = F.cosine_similarity(hs_crop[0, :max(valid, 1)], ref[0, :max(valid, 1)], dim=-1).mean().item()
print(f"\n=== M1 encoder_hidden_states cosine(device vs HF bundle) = {cos:.6f} (valid-token mean {pt:.6f}) ===", flush=True)
print("RESULT:", "PASS" if pt >= 0.999 else "FAIL", flush=True)

torch.save(
    {"encoder_hidden_states": hs_crop.to(torch.bfloat16),
     "encoder_attention_mask": attn[:, CROP_START:].clone()},
    "/tmp/hy_dev_llama.pt",
)
print("[m1] saved -> /tmp/hy_dev_llama.pt", flush=True)
