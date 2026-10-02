"""Dump HunyuanVideo's real text conditioning for the CPU DiT probe: Llama hidden_states[-3]
(cropped 95, 256 rows, bf16) + mask, and CLIP pooled projections."""
import glob
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPTextModel, CLIPTokenizer

sys.path.insert(0, "/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan")
from difflet.cli.orchestrators import hunyuan_video as hv  # noqa: E402

OUT = "/home/ubuntu/.claude/jobs/b5f130d0/tmp/hv_text.pt"
snap = glob.glob("/home/ubuntu/.cache/huggingface/hub/models--hunyuanvideo-community--HunyuanVideo/snapshots/*")[0]
seq = hv._TEXT_SEQ_LEN + hv._LLAMA_CROP_START
prompt = "a cinematic shot of a red fox running through a snowy forest"

clip_tok = CLIPTokenizer.from_pretrained(f"{snap}/tokenizer_2")
clip = CLIPTextModel.from_pretrained(f"{snap}/text_encoder_2", torch_dtype=torch.bfloat16).eval()
ci = clip_tok(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
with torch.no_grad():
    pooled = clip(input_ids=ci.input_ids, attention_mask=ci.attention_mask).pooler_output
print("clip pooled", tuple(pooled.shape), flush=True)

tok = AutoTokenizer.from_pretrained(f"{snap}/tokenizer")
ti = tok(hv._LLAMA_TEMPLATE.format(prompt), max_length=seq, padding="max_length", truncation=True,
         return_tensors="pt", return_attention_mask=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(f"{snap}/text_encoder", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
print("llama loaded in", round(time.time() - t0, 1), "s", flush=True)
with torch.no_grad():
    out = model(input_ids=ti.input_ids, attention_mask=ti.attention_mask, output_hidden_states=True)
hidden = out.hidden_states[-3][:, hv._LLAMA_CROP_START:].to(torch.bfloat16).contiguous()
mask = ti.attention_mask[:, hv._LLAMA_CROP_START:].to(torch.int64)
torch.save({"encoder_hidden_states": hidden, "encoder_attention_mask": mask, "pooled_projections": pooled.to(torch.bfloat16)}, OUT)
print("saved", OUT, tuple(hidden.shape), "valid", int(mask.sum()), flush=True)
