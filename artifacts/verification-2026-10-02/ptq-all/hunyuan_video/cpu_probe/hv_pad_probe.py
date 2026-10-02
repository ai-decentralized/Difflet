"""CPU probe: magnitude of HunyuanVideo's Llama text-encoder hidden states at pad rows vs valid rows.

Replicates the CLI llama stage (template, max_length=256+95, hidden_states layer capture, crop 95)
with HF transformers in bf16 on the host; prints per-row norm statistics.
"""
import glob
import sys
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan")
from difflet.cli.orchestrators import hunyuan_video as hv  # noqa: E402

snap = glob.glob("/home/ubuntu/.cache/huggingface/hub/models--hunyuanvideo-community--HunyuanVideo/snapshots/*")[0]
seq = hv._TEXT_SEQ_LEN + hv._LLAMA_CROP_START
prompt = "a cinematic shot of a red fox running through a snowy forest"
tok = AutoTokenizer.from_pretrained(f"{snap}/tokenizer")
ti = tok(hv._LLAMA_TEMPLATE.format(prompt), max_length=seq, padding="max_length", truncation=True,
         return_tensors="pt", return_attention_mask=True)
print("capture", hv._LLAMA_CAPTURE, "seq", seq, "valid tokens", int(ti.attention_mask.sum()), flush=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(f"{snap}/text_encoder", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.eval()
print("loaded in", round(time.time() - t0, 1), "s", flush=True)
with torch.no_grad():
    out = model(input_ids=ti.input_ids, attention_mask=ti.attention_mask, output_hidden_states=True)
# diffusers HunyuanVideo uses hidden_states[-(num_hidden_layers_to_skip + 1)] with skip=2 -> hidden_states[-3]
hs = out.hidden_states
for label, h in (("hidden_states[-3] (diffusers default)", hs[-3]), ("hidden_states[-1]", hs[-1])):
    h = h[:, hv._LLAMA_CROP_START:].float()
    m = ti.attention_mask[:, hv._LLAMA_CROP_START:].bool()[0]
    norms = h[0].norm(dim=-1)
    absmax = h[0].abs().amax(dim=-1)
    print(f"== {label}: rows {h.shape[1]} valid {int(m.sum())} pad {int((~m).sum())}")
    print(f"   valid rows: norm mean {norms[m].mean():.1f} max {norms[m].max():.1f}; absmax mean {absmax[m].mean():.2f} max {absmax[m].max():.2f}")
    print(f"   pad rows:   norm mean {norms[~m].mean():.1f} max {norms[~m].max():.1f}; absmax mean {absmax[~m].mean():.2f} max {absmax[~m].max():.2f}")
    print(f"   per-tensor absmax over all rows {h.abs().amax():.2f} vs valid-only {h[0][m].abs().amax():.2f}")
