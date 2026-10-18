"""A tiny transformers-layout Gemma-3 text checkpoint for the LTX-2 device text encoder tests,
and their gloo worker (not collected by pytest).

``write_tiny_gemma3(root)`` writes ``root/text_encoder``: a ``config.json`` with a
``text_config`` (Gemma-3's multimodal layout) and ``model.safetensors`` holding a seeded,
perturbed transformers ``Gemma3TextModel`` under ``language_model.model.*`` plus a stray
``vision_tower.*`` tensor (both as in the real checkpoint), and returns transformers' own fp32
hidden states of a LEFT-padded prompt with its attention mask: the 49-state stack LTX-2's
pipeline reads, here ``num_layers + 1`` states.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

TINY_GEMMA3 = {
    "vocab_size": 96, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 3,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0, "rope_local_base_freq": 10_000.0,
    "rope_scaling": {"factor": 8.0, "rope_type": "linear"}, "query_pre_attn_scalar": 16,
    "sliding_window": 16, "hidden_activation": "gelu_pytorch_tanh", "attention_bias": False,
    "attn_logit_softcapping": None, "final_logit_softcapping": None,
    "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
    "max_position_embeddings": 128, "pad_token_id": 0, "use_bidirectional_attention": False,
}
SEQ_LEN = 12
VALID = 7


def tiny_tokens():
    """``(left-padded ids [1, L], mask [1, L], right-padded ids [1, L])`` with VALID tokens."""
    gen = torch.Generator().manual_seed(5)
    valid = torch.randint(1, TINY_GEMMA3["vocab_size"], (VALID,), generator=gen)
    left = torch.zeros(1, SEQ_LEN, dtype=torch.int64)
    left[0, SEQ_LEN - VALID:] = valid
    mask = torch.zeros(1, SEQ_LEN, dtype=torch.int64)
    mask[0, SEQ_LEN - VALID:] = 1
    right = torch.zeros(1, SEQ_LEN, dtype=torch.int64)
    right[0, :VALID] = valid
    return left, mask, right


def write_tiny_gemma3(root) -> torch.Tensor:
    from safetensors.torch import save_file
    from transformers import Gemma3TextConfig, Gemma3TextModel

    config = Gemma3TextConfig(**TINY_GEMMA3)
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = Gemma3TextModel(config).eval()
        with torch.no_grad():
            for param in model.parameters():
                param.add_(0.05 * torch.randn_like(param))
    directory = Path(root) / "text_encoder"
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {f"language_model.model.{k}": v.contiguous()
               for k, v in model.state_dict().items()}
    tensors["vision_tower.vision_model.post_layernorm.weight"] = torch.ones(8)
    save_file(tensors, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps(
        {"model_type": "gemma3", "architectures": ["Gemma3ForConditionalGeneration"],
         "text_config": {**TINY_GEMMA3, "model_type": "gemma3_text"}}))
    left, mask, _ = tiny_tokens()
    with torch.no_grad():
        out = model(input_ids=left, attention_mask=mask, output_hidden_states=True)
    return torch.stack(out.hidden_states, dim=-1)


def gemma3_worker(rank: int, world_size: int, root: str) -> dict:
    """The device text encoder application on CPU fp32 at TP=world: the stacked hidden states
    of the right-padded prompt, and the lifecycle."""
    from difflet.backends.neuron.ltx_2.text_encoder import NeuronLtx2TextEncoderApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    app = NeuronLtx2TextEncoderApplication(
        model_path=str(Path(root) / "text_encoder"), seq_len=SEQ_LEN,
        parallel=DiffletParallelConfig(tp_degree=world_size), dtype=torch.float32,
        exec_mode="compile" if world_size > 1 else "eager", device="cpu")
    app.load()
    _, _, right = tiny_tokens()
    states = app(right)
    return {"states": states.tolist(), "report": app.load_report,
            "compiled_blocks": list(app.compiled_blocks),
            "heads": int(app.module.layers[0].self_attn.heads),
            "kv_heads": int(app.module.layers[0].self_attn.kv_heads)}
