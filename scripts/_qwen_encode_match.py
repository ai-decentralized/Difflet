"""Q1: on-device Qwen2.5-VL encode reproducing Qwen-Image's HF encode_prompt.

HF _get_qwen_prompt_embeds: format prompt into prompt_template_encode, tokenize, run
Qwen2.5-VL with output_hidden_states -> hidden_states[-1] (final norm), extract valid
tokens via mask, drop the first prompt_template_encode_start_idx=34, pad. dim=3584, no CLIP.

We capture `norm` on the device Qwen2.5-VL text model, drop 34, and compare the valid
tokens to the cached Qwen-Image bundle's encoder_hidden_states.
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
from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_text import (
    NeuronQwen2VLTextForCausalLM,
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

SNAP = glob.glob(
    "/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen-Image/snapshots/*"
)[0]
ENC = f"{SNAP}/text_encoder"
TOK = f"{SNAP}/tokenizer"
OUT = "/home/ubuntu/difflet/.difflet-cache/qwen_qwen25vl_enc"
BUNDLE = "/home/ubuntu/difflet/.difflet-cache/qwen_image_dit_inputs/full_1024_4step.safetensors"

DROP_IDX = 34  # prompt_template_encode_start_idx
SEQ = 256  # device bucket (templated prompt is short; valid tokens are mask-independent of pad)
TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)

meta = json.loads(open(BUNDLE + ".meta.json").read())
prompt = meta["prompt"][0]
print(f"[q1] prompt={prompt!r} seq={SEQ}", flush=True)

full_cfg = AutoConfig.from_pretrained(ENC)
text_cfg = full_cfg.text_config  # the LM config (mrope, 28 layers, 3584, GQA-4)
if getattr(text_cfg, "pad_token_id", None) is None:
    text_cfg.pad_token_id = 0

nc = NeuronConfig(
    tp_degree=4, batch_size=1, seq_len=SEQ, torch_dtype=torch.bfloat16,
    on_device_sampling_config={},
    tensor_capture_config=TensorCaptureConfig(modules_to_capture=["norm"]),
)
config = NeuronQwen2VLTextForCausalLM.get_config_cls()(
    nc, load_config=load_pretrained_config(hf_config=text_cfg)
)
model = NeuronQwen2VLTextForCausalLM(ENC, config)
if not os.path.exists(os.path.join(OUT, "model.pt")):
    print("[q1] compiling Qwen2.5-VL text @256 capturing norm ...", flush=True)
    t0 = time.time(); model.compile(OUT); print(f"[q1] compiled {time.time()-t0:.1f}s", flush=True)
model.load(OUT)

tok = AutoTokenizer.from_pretrained(TOK)
templated = TEMPLATE.format(prompt)
ti = tok(templated, max_length=SEQ, padding="max_length", truncation=True,
         return_tensors="pt", return_attention_mask=True)
input_ids = ti.input_ids.to(torch.int32)
attn = ti.attention_mask.to(torch.int32)
position_ids = torch.arange(SEQ, dtype=torch.int32).unsqueeze(0)
valid = int(attn.sum())
print(f"[q1] tokenized valid={valid} (drop {DROP_IDX} -> {valid - DROP_IDX} output tokens)", flush=True)

out = model(input_ids=input_ids, attention_mask=attn, position_ids=position_ids,
            sampling_params=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32))
cap = getattr(out, "captured_tensors", None)
print("[q1] captured:", None if not cap else [tuple(t.shape) for t in cap], flush=True)
if not cap:
    raise SystemExit("no capture")

hs = cap[0].float().cpu()  # (1, 256, 3584)
# extract valid tokens then drop the template prefix
dev = hs[:, DROP_IDX:valid]  # (1, valid-34, 3584)
print(f"[q1] device valid-region {tuple(dev.shape)}", flush=True)

ref_all = load_file(BUNDLE)["encoder_hidden_states"].float()  # (1,1024,3584) padded
ref_mask = load_file(BUNDLE)["encoder_hidden_states_mask"]
ref_valid = int(ref_mask.sum())
ref = ref_all[:, :ref_valid]  # (1, ref_valid, 3584)
print(f"[q1] bundle valid={ref_valid} region {tuple(ref.shape)}", flush=True)

n = min(dev.shape[1], ref.shape[1])
cos = F.cosine_similarity(dev[:, :n].reshape(-1), ref[:, :n].reshape(-1), dim=0).item()
pt = F.cosine_similarity(dev[0, :n], ref[0, :n], dim=-1).mean().item()
print(f"\n=== Q1 encoder_hidden_states cosine(device vs HF bundle) = {cos:.6f} (per-token mean {pt:.6f}) ===", flush=True)
print("RESULT:", "PASS" if pt >= 0.999 else "FAIL", flush=True)
