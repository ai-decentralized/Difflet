"""One device run: compile Llama-3 encoder with hidden-states output (tensor_capture
on the final `norm`), load HF weights, run a forward, and check cosine parity of the
all-position hidden states vs the HF CPU LlamaModel reference.

Parity uses identical input_ids on both sides (tokenizer/prompt-template fidelity is a
separate host-side concern, not what this verifies).
"""

import glob
import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoConfig, LlamaModel

from neuronx_distributed_inference.models.config import (
    NeuronConfig,
    TensorCaptureConfig,
)
from neuronx_distributed_inference.models.llama.modeling_llama import (
    NeuronLlamaForCausalLM,
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

SNAP = glob.glob(
    "/home/ubuntu/.cache/huggingface/hub/"
    "models--hunyuanvideo-community--HunyuanVideo/snapshots/*/text_encoder"
)[0]
OUT = "/home/ubuntu/difflet/.difflet-cache/hunyuan_llama_enc_hs"
SEQ = 256
os.makedirs(OUT, exist_ok=True)

print("model_path:", SNAP, flush=True)
hf_cfg = AutoConfig.from_pretrained(SNAP)
if hf_cfg.pad_token_id is None:
    hf_cfg.pad_token_id = 0
# Checkpoint is a bare LlamaModel (no lm_head). We capture the pre-lm_head `norm`
# output, so tie lm_head -> embed_tokens just to satisfy the weight loader.
hf_cfg.tie_word_embeddings = True

neuron_config = NeuronConfig(
    tp_degree=4,
    batch_size=1,
    seq_len=SEQ,
    torch_dtype=torch.bfloat16,
    on_device_sampling_config={},  # tensor_capture requires this even at defaults
    tensor_capture_config=TensorCaptureConfig(modules_to_capture=["norm"]),
)
config = NeuronLlamaForCausalLM.get_config_cls()(
    neuron_config, load_config=load_pretrained_config(hf_config=hf_cfg)
)
model = NeuronLlamaForCausalLM(SNAP, config)

if not os.path.exists(os.path.join(OUT, "model.pt")):
    print("compiling (with tensor_capture on `norm`) ...", flush=True)
    t0 = time.time()
    model.compile(OUT)
    print(f"compiled in {time.time() - t0:.1f}s", flush=True)

print("loading weights onto device ...", flush=True)
t0 = time.time()
model.load(OUT)
print(f"loaded in {time.time() - t0:.1f}s", flush=True)

# --- fixed identical input on both sides ---
torch.manual_seed(0)
input_ids = torch.randint(1, 10000, (1, SEQ), dtype=torch.int32)
attention_mask = torch.ones((1, SEQ), dtype=torch.int32)
position_ids = torch.arange(SEQ, dtype=torch.int32).unsqueeze(0)

print("running Neuron forward ...", flush=True)
out = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    position_ids=position_ids,
    sampling_params=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
)
print("output type:", type(out), flush=True)
captured = getattr(out, "captured_tensors", None)
print("captured_tensors:", None if captured is None else [tuple(t.shape) for t in captured], flush=True)
if not captured:
    print("[DIAG] top-level submodule names containing 'norm':", flush=True)
    for n, _ in model.named_modules():
        if n.endswith("norm") or n == "norm":
            print("   ", n, flush=True)
    raise SystemExit("no captured tensors — fix module name")

neuron_hs = captured[0].float().cpu()  # (1, SEQ, 4096)
print("neuron hidden states:", tuple(neuron_hs.shape), neuron_hs.dtype, flush=True)

# --- HF CPU golden ---
print("running HF CPU reference ...", flush=True)
hf = LlamaModel.from_pretrained(SNAP, torch_dtype=torch.float32).eval()
with torch.no_grad():
    hf_hs = hf(input_ids=input_ids.long(), attention_mask=attention_mask).last_hidden_state.float()
print("hf hidden states:", tuple(hf_hs.shape), flush=True)

# --- parity ---
cos = F.cosine_similarity(neuron_hs.reshape(-1), hf_hs.reshape(-1), dim=0).item()
max_abs = (neuron_hs - hf_hs).abs().max().item()
rel = ((neuron_hs - hf_hs).norm() / hf_hs.norm()).item()
print(f"\n=== PARITY: cosine={cos:.6f}  max_abs_err={max_abs:.4f}  rel_l2={rel:.4f} ===", flush=True)
print("RESULT:", "PASS" if cos >= 0.99 else "FAIL", flush=True)
