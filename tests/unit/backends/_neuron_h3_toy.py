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


def tiny_inputs(*, scale: float = 1.0, max_text: int = TINY_TEXT_SEQ_LEN):
    """The DiT call at the tiny shape (bool masks, as the denoise loop builds them); the text
    bucket is ``max_text`` rows, the first LIVE_TEXT live (the same values for any bucket)."""
    from difflet.models.minimax_h3.contracts import build_padded_t2va_layout

    layout = build_padded_t2va_layout(
        num_text_tokens=LIVE_TEXT, max_text_tokens=max_text, sequence_alignment=128,
        **TINY_SHAPE,
    )
    gen = torch.Generator().manual_seed(1)
    video_rows = int(layout.video_indices.numel())
    audio_rows = int(layout.audio_indices.numel())
    timestep_indices = torch.ones(layout.sequence_length, dtype=torch.int64)
    timestep_indices[layout.audio_indices] = 0
    text_mask = torch.zeros(1, max_text, dtype=torch.bool)
    text_mask[:, :LIVE_TEXT] = True
    video = scale * torch.randn(1, video_rows, 96, generator=gen)
    audio = scale * torch.randn(1, audio_rows, 32, generator=gen)
    text = torch.zeros(1, max_text, 16)
    text[:, :LIVE_TEXT] = scale * torch.randn(1, LIVE_TEXT, 16, generator=gen)
    return (
        video,
        audio,
        text,
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
                 exec_mode: str, cp: int = 1, cp_mode: str = "gather_kv") -> dict:
    """``world_size = tp * cp``; ``cp > 1`` runs the block stack context-parallel."""
    from difflet.backends.neuron.minimax_h3.transformer import (
        NeuronMiniMaxH3Config,
        NeuronMiniMaxH3TransformerApplication,
    )
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    tp = world_size // cp
    parallel = DiffletParallelConfig(tp_degree=tp, cp_degree=cp, cp_mode=cp_mode)
    config = NeuronMiniMaxH3Config.from_pretrained(
        transformer_dir, **TINY_SHAPE, text_seq_len=TINY_TEXT_SEQ_LEN, tp_degree=tp,
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
    # A wider text bucket (more padding rows) gives the same outputs.
    wide_v, wide_a = app(*as_neuron_call(tiny_inputs(max_text=2 * TINY_TEXT_SEQ_LEN), table))
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
        "cp_mode": getattr(block0.attn.processor, "mode", None),
        "has_adaln": block0.adaln_proj is not None,
        "compiled_blocks": list(app.compiled_blocks),
        "params_on_meta": sum(p.is_meta for p in app.module.parameters()),
        "video": video.float().tolist(),
        "audio": audio.float().tolist(),
        "other_differs": not bool(torch.equal(video, other)),
        "wide_max_abs": max(float((wide_v - video).abs().max()), float((wide_a - audio).abs().max())),
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


# --------------------------------------------------------------------------- text encoder

TINY_QWEN3VL_TEXT = dict(
    vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
    num_attention_heads=8, num_key_value_heads=4, head_dim=16, rms_norm_eps=1e-6,
    rope_theta=5000000.0, max_position_embeddings=64,
    rope_scaling={"rope_type": "default", "mrope_section": [3, 3, 2], "mrope_interleaved": True},
    pad_token_id=0,
)
TE_LAYER = 3
TE_SEQ = 16
TE_LIVE = 11


def write_tiny_qwen3vl(root) -> Path:
    """A tiny Qwen3VLForConditionalGeneration saved by transformers (``model.language_model.*``,
    ``model.visual.*``, ``lm_head``), as H3's ``text_encoder/``."""
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    torch.manual_seed(0)
    config = Qwen3VLConfig(
        text_config=TINY_QWEN3VL_TEXT,
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                           out_hidden_size=64, patch_size=4, spatial_merge_size=2,
                           temporal_patch_size=2, deepstack_visual_indexes=[0],
                           num_position_embeddings=16),
    )
    model = Qwen3VLForConditionalGeneration(config).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.1)
    out = Path(root) / "text_encoder"
    model.save_pretrained(out, safe_serialization=True)
    return out


def tiny_te_ids() -> torch.Tensor:
    ids = torch.zeros(1, TE_SEQ, dtype=torch.int64)
    ids[0, :TE_LIVE] = torch.randint(1, 128, (TE_LIVE,), generator=torch.Generator().manual_seed(2))
    return ids


def te_reference(encoder_dir) -> torch.Tensor:
    """transformers' model: hidden_states[TE_LAYER] (layer TE_LAYER-1's output), with the mask."""
    from transformers import Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(encoder_dir,
                                                            torch_dtype=torch.float32).eval()
    ids = tiny_te_ids()
    mask = (torch.arange(TE_SEQ) < TE_LIVE).to(torch.int64)[None]
    with torch.no_grad():
        out = model.model.language_model(input_ids=ids, attention_mask=mask,
                                         output_hidden_states=True, use_cache=False)
    return out.hidden_states[TE_LAYER][:, :TE_LIVE]


def te_worker(rank: int, world_size: int, encoder_dir: str, exec_mode: str) -> dict:
    from difflet.backends.neuron.minimax_h3.text_encoder import (
        NeuronMiniMaxH3TextEncoderApplication,
    )
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    app = NeuronMiniMaxH3TextEncoderApplication(
        model_path=encoder_dir, seq_len=TE_SEQ, layer=TE_LAYER,
        parallel=DiffletParallelConfig(tp_degree=world_size), dtype=torch.float32,
        exec_mode=exec_mode, device="cpu",
    )
    torch._dynamo.reset()
    from difflet.backends.neuron.core.application_base import TorchNeuronApplicationBase

    TorchNeuronApplicationBase.load(app)  # no tokenizer in the tiny snapshot
    hidden = app(tiny_te_ids())
    return {"hidden": hidden[:, :TE_LIVE].tolist(), "report": app.load_report,
            "layers": len(app.module.layers)}


def vae_ranks_worker(rank: int, world_size: int, vae_dir: str, scratch: str) -> dict:
    """``decode_video_on_ranks`` over the gloo ranks vs one process decoding every tile."""
    from difflet.backends.neuron.minimax_h3.vae import (
        DeviceTileDecoder,
        NeuronMiniMaxH3VideoVaeApplication,
        decode_video_on_ranks,
    )
    from difflet.models.minimax_h3.pipeline import decode_minimax_h3_video
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    app = NeuronMiniMaxH3VideoVaeApplication(
        model_path=vae_dir, parallel=DiffletParallelConfig(tp_degree=world_size),
        dtype=torch.float32, exec_mode="eager", device="cpu")
    app.load(skip_warmup=True)
    decoder = DeviceTileDecoder(app)
    latents = torch.randn(1, 24, 10, 16, 28, generator=torch.Generator().manual_seed(5))
    cfg = app.vae_config
    calls = []
    counted = lambda tile: (calls.append(1), decoder(tile))[1]  # noqa: E731
    video = decode_video_on_ranks(counted, latents, cfg, rank=rank, world=world_size,
                                  scratch_dir=scratch, device=torch.device("cpu"))
    result = {"tiles_decoded_here": len(calls)}
    if rank == 0:
        ref = decode_minimax_h3_video(decoder, latents, latents_mean=cfg["latents_mean"],
                                      latents_std=cfg["latents_std"],
                                      clip_length=int(cfg["clip_length"]),
                                      token_drop=int(cfg["token_drop"]))
        result.update(equal=bool(torch.equal(video, ref)), shape=list(video.shape),
                      leftovers=sorted(p.name for p in Path(scratch).iterdir()))
    else:
        result["video_is_none"] = video is None
    return result
