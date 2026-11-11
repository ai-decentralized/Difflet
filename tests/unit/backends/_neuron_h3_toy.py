"""A tiny diffusers-layout MiniMax-H3 DiT for the neuron H3 tests, and their gloo workers (not a
test module).

Every run goes through ``_neuron_gloo.run_ranks`` in a fresh process with
``DIFFLET_BACKEND=neuron``, which binds ``difflet.ops`` to the neuron implementations on CPU/gloo.
"""

from __future__ import annotations

from pathlib import Path

import torch

TINY = dict(
    num_attention_heads=4,
    attention_head_dim=16,
    hidden_size=32,
    num_layers=2,
    num_refiner_layers=1,
    ffn_dim=64,
    in_channels=24,
    audio_in_channels=32,
    patch_size=(1, 2, 2),
    text_dim=16,
    freq_dim=8,
    time_embed_hidden_dim=32,
    time_embed_dim=16,
    rope_freq_dim=2,
    rope_theta=10000.0,
)
TINY_SHAPE = dict(height=32, width=32, num_frames=124)
TINY_TEXT_SEQ_LEN = 8
LIVE_TEXT = 5
TIMESTEP = (0.3, 0.7)


def _model(precomputed_adaln: bool = False):
    from difflet.models.minimax_h3.modeling_minimax_h3 import MiniMaxH3Transformer3DModel

    return MiniMaxH3Transformer3DModel(**TINY, precomputed_adaln=precomputed_adaln)


def write_tiny_h3_transformer(root) -> Path:
    """Random weights (fp32 everywhere, scaled so the outputs are not dominated by noise),
    saved sharded with an index as the real checkpoint is."""
    torch.manual_seed(0)
    model = _model()
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0.0, 0.2)
    out = Path(root) / "transformer"
    model.save_pretrained(out, max_shard_size="200KB", safe_serialization=True)
    return out


def tiny_inputs(*, scale: float = 1.0):
    """The DiT call at the tiny shape (bool masks, as the denoise loop builds them)."""
    from difflet.models.minimax_h3.contracts import build_padded_t2va_layout

    layout = build_padded_t2va_layout(
        num_text_tokens=LIVE_TEXT, max_text_tokens=TINY_TEXT_SEQ_LEN, sequence_alignment=128,
        **TINY_SHAPE,
    )
    gen = torch.Generator().manual_seed(1)
    video_rows = int(layout.video_indices.numel())
    audio_rows = int(layout.audio_indices.numel())
    timestep_indices = torch.ones(layout.sequence_length, dtype=torch.int64)
    timestep_indices[layout.audio_indices] = 0
    text_mask = torch.zeros(1, TINY_TEXT_SEQ_LEN, dtype=torch.bool)
    text_mask[:, :LIVE_TEXT] = True
    return (
        scale * torch.randn(1, video_rows, 96, generator=gen),
        scale * torch.randn(1, audio_rows, 32, generator=gen),
        scale * torch.randn(1, TINY_TEXT_SEQ_LEN, 16, generator=gen),
        torch.tensor(TIMESTEP, dtype=torch.float32),
        timestep_indices,
        layout.token_tags.to(torch.int64),
        layout.position_ids.to(torch.float32),
        layout.video_indices.to(torch.int64),
        layout.audio_indices.to(torch.int64),
        layout.text_indices.to(torch.int64),
        text_mask,
        layout.attention_mask.unsqueeze(0),
    )


def diffusers_reference(transformer_dir):
    """diffusers' model on the same files, fp32, with the adaln branch (no precompute)."""
    from difflet.models.minimax_h3.modeling_minimax_h3 import MiniMaxH3Transformer3DModel

    model = MiniMaxH3Transformer3DModel.from_pretrained(transformer_dir, torch_dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        video, audio = model(*tiny_inputs(), return_dict=False)
    return video, audio


def modulation_table(transformer_dir) -> torch.Tensor:
    from difflet.models.minimax_h3.pipeline import build_adaln_modulation_table

    table = build_adaln_modulation_table(
        str(transformer_dir), torch.tensor([TIMESTEP]), num_layers=TINY["num_layers"],
        freq_dim=TINY["freq_dim"], time_embed_hidden_dim=TINY["time_embed_hidden_dim"],
        time_embed_dim=TINY["time_embed_dim"], hidden_size=TINY["hidden_size"],
        dtype=torch.float32,
    )
    return table[0]


def as_neuron_call(inputs, table=None):
    """The neuron application's call: masks as int32, the step's modulation table appended."""
    *head, text_mask, mask = inputs
    call = (*head, text_mask.to(torch.int32), mask.to(torch.int32))
    return call if table is None else (*call, table)


# --------------------------------------------------------------------------- gloo workers


def l1_h3_worker(rank: int, world_size: int, transformer_dir: str, precomputed: bool,
                 exec_mode: str) -> dict:
    from difflet.backends.neuron.minimax_h3.transformer import (
        NeuronMiniMaxH3Config,
        NeuronMiniMaxH3TransformerApplication,
    )
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    parallel = DiffletParallelConfig(tp_degree=world_size)
    config = NeuronMiniMaxH3Config.from_pretrained(
        transformer_dir, **TINY_SHAPE, text_seq_len=TINY_TEXT_SEQ_LEN, tp_degree=world_size,
        precomputed_adaln=precomputed,
    )
    app = NeuronMiniMaxH3TransformerApplication(
        model_path=transformer_dir, config=config, parallel=parallel, dtype=torch.float32,
        exec_mode=exec_mode, device="cpu",
    )
    torch._dynamo.reset()
    app.load()
    table = modulation_table(transformer_dir) if precomputed else None
    video, audio = app(*as_neuron_call(tiny_inputs(), table))
    other, _ = app(*as_neuron_call(tiny_inputs(scale=0.5), table))
    t = app.module.transformer
    block0 = t.transformer_blocks[0]
    return {
        "load_report": app.load_report,
        "fp32": {name: str(getattr(t, name).weight.dtype if hasattr(getattr(t, name), "weight")
                           else next(getattr(t, name).parameters()).dtype)
                 for name in ("proj_in", "audio_proj_in", "time_embedder", "proj_out",
                              "audio_proj_out")},
        "block_dtype": str(block0.attn.to_q.weight.dtype),
        "local_heads": int(block0.attn.heads),
        "processor": type(block0.attn.processor).__name__,
        "has_adaln": block0.adaln_proj is not None,
        "compiled_blocks": list(app.compiled_blocks),
        "params_on_meta": sum(p.is_meta for p in app.module.parameters()),
        "video": video.float().tolist(),
        "audio": audio.float().tolist(),
        "other_differs": not bool(torch.equal(video, other)),
    }


def vae_worker(rank: int, world_size: int, vae_dir: str, exec_mode: str) -> dict:
    """The tiny H3 video VAE as a neuron application vs the host reference decoder, one tile."""
    from difflet.backends.neuron.minimax_h3.vae import (
        TILE_LATENT_SHAPE,
        DeviceTileDecoder,
        NeuronMiniMaxH3VideoVaeApplication,
    )
    from difflet.models.minimax_h3.host_components import load_host_video_decoder

    app = NeuronMiniMaxH3VideoVaeApplication(model_path=vae_dir, dtype=torch.float32,
                                             exec_mode=exec_mode, device="cpu")
    torch._dynamo.reset()
    app.load()
    tile = torch.randn(TILE_LATENT_SHAPE, generator=torch.Generator().manual_seed(3))
    out = DeviceTileDecoder(app)(tile)
    ref = load_host_video_decoder(str(Path(vae_dir).parent))(tile)
    return {"max_abs": float((out - ref).abs().max()), "shape": list(out.shape),
            "dtype": str(out.dtype), "report": app.load_report,
            "norm_dtype": str(app.module.decoder.transformer_blocks[0].norm1.weight.dtype)}
