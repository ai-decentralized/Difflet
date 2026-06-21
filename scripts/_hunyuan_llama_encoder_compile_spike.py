"""Spike: compile HunyuanVideo's Llama-3 text encoder via NxDI's NeuronLlamaForCausalLM.

Goal: prove neuronx-cc lowers the Llama-3-8B graph at the HunyuanVideo text_encoder
config (TP=4, seq_len=256). Weights are NOT needed to compile (NxDI traces from
config only); the in-flight weight download is for the later load+forward smoke.
"""

import glob
import os
import time

import torch
from transformers import AutoConfig

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.llama.modeling_llama import (
    NeuronLlamaForCausalLM,
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

SNAP = glob.glob(
    "/home/ubuntu/.cache/huggingface/hub/"
    "models--hunyuanvideo-community--HunyuanVideo/snapshots/*/text_encoder"
)[0]
OUT = "/home/ubuntu/difflet/.difflet-cache/hunyuan_llama_enc_compile"
os.makedirs(OUT, exist_ok=True)

print("model_path:", SNAP, flush=True)
hf_cfg = AutoConfig.from_pretrained(SNAP)
if hf_cfg.pad_token_id is None:
    hf_cfg.pad_token_id = 0
    print("injected pad_token_id=0", flush=True)

neuron_config = NeuronConfig(
    tp_degree=4,
    batch_size=1,
    seq_len=256,
    torch_dtype=torch.bfloat16,
)
config = NeuronLlamaForCausalLM.get_config_cls()(
    neuron_config, load_config=load_pretrained_config(hf_config=hf_cfg)
)
print("building NeuronLlamaForCausalLM ...", flush=True)
model = NeuronLlamaForCausalLM(SNAP, config)

print(f"compiling -> {OUT}", flush=True)
t0 = time.time()
model.compile(OUT)
print(f"COMPILE OK in {time.time() - t0:.1f}s -> {OUT}", flush=True)
