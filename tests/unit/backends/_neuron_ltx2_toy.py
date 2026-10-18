"""A tiny diffusers-layout LTX-2 DiT for the neuron LTX-2 tests, and their gloo workers (not
collected by pytest).

``write_tiny_ltx2_model(root)`` writes ``root/transformer``: the ``config.json`` and a sharded
safetensors checkpoint (with its ``.safetensors.index.json``, the real snapshot's layout) of a
seeded diffusers ``LTX2VideoTransformer3DModel`` (2 blocks; 4 video heads x 8 and 4 audio heads
x 4, split RoPE as in the real checkpoint), and returns diffusers' own fp32 outputs on
``tiny_ltx2_inputs()``: the model reloaded from those files and called without text masks, the
unmasked text attention every Difflet LTX-2 backend runs. The reference never touches
``difflet.ops``.

Every run goes through ``_neuron_gloo.run_ranks`` in a fresh process with
``DIFFLET_BACKEND=neuron``, which binds ``difflet.ops`` (resolved when the LTX-2 modeling is
imported) to the neuron backend and keeps Dynamo's caches out of the parent.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

TINY_LTX2_TRANSFORMER: dict = {
    "_class_name": "LTX2VideoTransformer3DModel",
    "_diffusers_version": "0.37.0.dev0",
    "activation_fn": "gelu-approximate",
    "attention_bias": True,
    "attention_head_dim": 8,
    "attention_out_bias": True,
    "audio_attention_head_dim": 4,
    "audio_cross_attention_dim": 16,
    "audio_hop_length": 160,
    "audio_in_channels": 128,
    "audio_num_attention_heads": 4,
    "audio_out_channels": 128,
    "audio_patch_size": 1,
    "audio_patch_size_t": 1,
    "audio_pos_embed_max_pos": 20,
    "audio_sampling_rate": 16000,
    "audio_scale_factor": 4,
    "base_height": 2048,
    "base_width": 2048,
    "caption_channels": 24,
    "causal_offset": 1,
    "cross_attention_dim": 32,
    "cross_attn_timestep_scale_multiplier": 1000,
    "in_channels": 128,
    "norm_elementwise_affine": False,
    "norm_eps": 1e-06,
    "num_attention_heads": 4,
    "num_layers": 2,
    "out_channels": 128,
    "patch_size": 1,
    "patch_size_t": 1,
    "pos_embed_max_pos": 20,
    "qk_norm": "rms_norm_across_heads",
    "rope_double_precision": True,
    "rope_theta": 10000.0,
    "rope_type": "split",
    "timestep_scale_multiplier": 1000,
    "vae_scale_factors": [8, 32, 32],
}
#: 64x96x17 -> latent 3 frames x 2 x 3 = 18 video tokens; round(17/24 * 25) = 18 audio frames.
TINY_SHAPE = {"height": 64, "width": 96, "num_frames": 17}
TINY_VIDEO_SEQ_LEN = 18
TINY_AUDIO_SEQ_LEN = 18
TINY_TEXT_SEQ_LEN = 6
TINY_WEIGHT_SEED = 0
TINY_INPUT_SEED = 1
#: A mid-schedule DiT timestep (the orchestrator passes the scheduler's 0..1000 timesteps).
TINY_TIMESTEP = 700.0
TINY_MAX_SHARD_SIZE = "60KB"

LTX2_SNAPSHOT = Path(
    "/home/ubuntu/.cache/huggingface/hub/models--Lightricks--LTX-2/snapshots/"
    "47da56e2ad66ce4125a9922b4a8826bf407f9d0a"
)


def tiny_ltx2_config(**overrides):
    """``NeuronLtx2Config`` of the tiny model at ``TINY_SHAPE``, fp32."""
    from difflet.backends.neuron.ltx_2.config import NeuronLtx2Config

    geometry = {k: v for k, v in TINY_LTX2_TRANSFORMER.items() if not k.startswith("_")}
    geometry["vae_scale_factors"] = tuple(geometry["vae_scale_factors"])
    kwargs = {**geometry, **TINY_SHAPE, "text_seq_len": TINY_TEXT_SEQ_LEN,
              "torch_dtype": torch.float32}
    kwargs.update(overrides)
    return NeuronLtx2Config(**kwargs)


def tiny_ltx2_inputs(*, batch: int = 1, dtype: torch.dtype = torch.float32, scale: float = 1.0):
    """``(hidden, audio_hidden, text, audio_text, timestep, video_coords, audio_coords)`` at the
    tiny shape, identical in every process (``scale`` varies the values, not the shapes)."""
    from difflet.backends.neuron.ltx_2.transformer import example_coords

    cfg = tiny_ltx2_config()
    gen = torch.Generator().manual_seed(TINY_INPUT_SEED)
    hidden = torch.randn(batch, TINY_VIDEO_SEQ_LEN, cfg.in_channels, generator=gen) * scale
    audio = torch.randn(batch, TINY_AUDIO_SEQ_LEN, cfg.audio_in_channels, generator=gen) * scale
    text = torch.randn(batch, TINY_TEXT_SEQ_LEN, cfg.caption_channels, generator=gen) * scale
    audio_text = torch.randn(batch, TINY_TEXT_SEQ_LEN, cfg.caption_channels, generator=gen) * scale
    timestep = torch.full((batch,), TINY_TIMESTEP * scale)
    video_coords, audio_coords = example_coords(cfg, batch)
    tensors = [t.to(dtype) for t in (hidden, audio, text, audio_text, timestep)]
    return (*tensors, video_coords, audio_coords)


def _diffusers_model(num_layers: int, seed: int):
    """A seeded diffusers model whose every parameter is perturbed off its init, so a weight
    that fails to load (or loads into the wrong slot) moves the output."""
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

    config = {k: v for k, v in TINY_LTX2_TRANSFORMER.items() if not k.startswith("_")}
    config["num_layers"] = num_layers
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = LTX2VideoTransformer3DModel(**config)
        with torch.no_grad():
            for param in model.parameters():
                param.add_(0.05 * torch.randn_like(param))
    return model.eval().requires_grad_(False)


def _round_rope_to_bf16(model) -> None:
    """Round the four RoPE modules' tables to bf16 (kept as fp32 values): what the shared TP
    patch (``_patch_ltx2_rope_for_tp``) does at tp > 1 on every backend, Trainium included."""
    for attr in ("rope", "audio_rope", "cross_attn_rope", "cross_attn_audio_rope"):
        rope = getattr(model, attr)

        def wrapped(*args, _forward=rope.forward, **kwargs):
            cos, sin = _forward(*args, **kwargs)
            return cos.to(torch.bfloat16).float(), sin.to(torch.bfloat16).float()

        rope.forward = wrapped


def diffusers_reference(transformer_dir, *, num_layers: int | None = None, batch: int = 1,
                        scale: float = 1.0, bf16_rope: bool = False):
    """diffusers' fp32 ``(video, audio)`` on ``tiny_ltx2_inputs()`` from the saved files, without
    text masks; with ``num_layers`` only the first that many blocks run; with ``bf16_rope`` the
    RoPE tables are rounded to bf16, as the TP path rounds them."""
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

    model = LTX2VideoTransformer3DModel.from_pretrained(
        str(transformer_dir), torch_dtype=torch.float32
    ).eval()
    if num_layers is not None:
        model.transformer_blocks = model.transformer_blocks[:num_layers]
    if bf16_rope:
        _round_rope_to_bf16(model)
    hidden, audio, text, audio_text, timestep, vcoords, acoords = tiny_ltx2_inputs(
        batch=batch, scale=scale)
    with torch.no_grad():
        video_out, audio_out = model(
            hidden_states=hidden, audio_hidden_states=audio, encoder_hidden_states=text,
            audio_encoder_hidden_states=audio_text, timestep=timestep, sigma=timestep,
            video_coords=vcoords, audio_coords=acoords, return_dict=False,
        )
    return video_out, audio_out


def write_tiny_ltx2_model(root, *, num_layers: int = 2):
    """Write ``root/transformer`` (sharded safetensors + index); return diffusers' fp32
    ``(video, audio)`` on ``tiny_ltx2_inputs()``."""
    transformer_dir = Path(root) / "transformer"
    model = _diffusers_model(num_layers, TINY_WEIGHT_SEED)
    model.save_pretrained(
        str(transformer_dir), safe_serialization=True, max_shard_size=TINY_MAX_SHARD_SIZE
    )
    if not list(transformer_dir.glob("*.safetensors.index.json")):
        raise AssertionError("the tiny checkpoint must be sharded with an index, as the real one")
    config = json.loads((transformer_dir / "config.json").read_text())
    config.update({k: v for k, v in TINY_LTX2_TRANSFORMER.items() if k != "num_layers"})
    config["num_layers"] = num_layers
    (transformer_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return diffusers_reference(transformer_dir)


def checkpoint_keys(transformer_dir) -> set[str]:
    from difflet.backends.neuron.core.checkpoint import build_weight_map

    return set(build_weight_map(transformer_dir))


# --------------------------------------------------------------------------
# gloo workers: fn(rank, world_size, *args) -> plain Python data
# --------------------------------------------------------------------------


def _recording_aot_eager(graphs: list):
    aot_eager = torch._dynamo.lookup_backend("aot_eager")

    def backend(gm, example_inputs):
        graphs.append(gm)
        return aot_eager(gm, example_inputs)

    return backend


def _collective_count(gm) -> int:
    return sum(
        1
        for node in gm.graph.nodes
        if node.op == "call_function"
        and "_c10d_functional" in str(node.target)
        and "wait_tensor" not in str(node.target)
    )


def l1_ltx2_worker(
    rank: int,
    world_size: int,
    model_dir: str,
    tp: int,
    exec_mode: str,
    num_layers: int | None = None,
    cfg: bool = False,
    cp_mode: str = "gather_kv",
) -> dict:
    """Load the tiny DiT at TP=tp (x CFG 2 when ``cfg``, else x CP world/tp) on CPU fp32, run the
    fixed inputs twice (the second time with other values at the same shapes), report the
    outputs and the lifecycle. In eager mode the video tokens the first block receives are
    recorded (this rank's shard under CP)."""
    from torch._dynamo.utils import counters

    from difflet.backends.neuron.ltx_2.config import NeuronLtx2Config
    from difflet.backends.neuron.ltx_2.transformer import (
        ROPE_ATTRS,
        NeuronLtx2TransformerApplication,
    )
    from difflet.backends.neuron.ops_impl import parallel_mesh
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    cp = 1 if cfg else world_size // tp
    parallel = DiffletParallelConfig(tp_degree=tp, cfg_parallel_enabled=cfg, cp_degree=cp,
                                     cp_mode=cp_mode if cp > 1 else "gather_kv")
    overrides = dict(TINY_SHAPE, text_seq_len=TINY_TEXT_SEQ_LEN, tp_degree=tp,
                     cfg_parallel_enabled=cfg, context_parallel_enabled=cp > 1,
                     cp_mode=parallel.cp_mode, torch_dtype=torch.float32)
    if num_layers is not None:
        overrides["num_layers"] = num_layers
    config = NeuronLtx2Config.from_pretrained(Path(model_dir) / "transformer", **overrides)
    app = NeuronLtx2TransformerApplication(
        model_path=Path(model_dir) / "transformer", config=config, parallel=parallel,
        dtype=torch.float32, exec_mode=exec_mode, device="cpu",
    )
    graphs: list = []
    if exec_mode == "compile":
        app.compile_backend = _recording_aot_eager(graphs)
    torch._dynamo.reset()
    counters.clear()
    app.load()
    after_warmup = counters["stats"]["unique_graphs"]
    block_tokens: list = []
    if exec_mode == "eager":
        app.module.transformer.transformer_blocks[0].register_forward_pre_hook(
            lambda module, args, kwargs: block_tokens.append(
                [int(kwargs["hidden_states"].shape[1]),
                 int(kwargs["audio_hidden_states"].shape[1])]),
            with_kwargs=True,
        )
    batch = 2 if cfg else 1
    out = app(*tiny_ltx2_inputs(batch=batch))
    other = app(*tiny_ltx2_inputs(batch=batch, scale=0.5))
    again = app(*tiny_ltx2_inputs(batch=batch))
    block0 = app.module.transformer.transformer_blocks[0]
    ropes = {attr: getattr(app.module.transformer, attr) for attr in ROPE_ATTRS}
    return {
        "rank": app.rank,
        "world_size": app.world_size,
        "mesh": str(parallel_mesh.get_mesh_spec()),
        "load_report": app.load_report,
        "warmup_shapes": app.warmup_shapes,
        "compiled_blocks": list(app.compiled_blocks),
        "unwarmed_shapes": list(app.unwarmed_shapes),
        "unique_graphs_after_warmup": after_warmup,
        "unique_graphs_after_forwards": counters["stats"]["unique_graphs"],
        "graph_breaks": sum(counters["graph_break"].values()),
        "collectives_per_graph": [_collective_count(gm) for gm in graphs],
        "num_blocks": len(app.module.transformer.transformer_blocks),
        "local_heads": {name: int(getattr(block0, name).heads)
                        for name in ("attn1", "audio_attn1", "attn2", "audio_attn2",
                                     "audio_to_video_attn", "video_to_audio_attn")},
        "processor": type(block0.attn1.processor).__name__,
        "kv_modes": {name: getattr(getattr(block0, name).processor, "kv_mode", "-")
                     for name in ("attn1", "audio_attn1", "attn2", "audio_attn2",
                                  "audio_to_video_attn", "video_to_audio_attn")},
        "block_tokens": block_tokens,
        "ff_activations": sorted({type(b.ff.net[0]).__name__ for b in
                                  app.module.transformer.transformer_blocks}
                                 | {type(b.audio_ff.net[0]).__name__ for b in
                                    app.module.transformer.transformer_blocks}),
        "rope_types": {attr: type(m).__name__ for attr, m in ropes.items()},
        "rope_computed": {attr: int(getattr(m, "computed", -1)) for attr, m in ropes.items()},
        "params_on_meta": sum(p.is_meta for p in app.module.parameters()),
        "param_numel": sum(p.numel() for p in app.module.parameters()),
        "weight_bytes": app.weight_bytes,
        "video": out[0].float().tolist(),
        "audio": out[1].float().tolist(),
        "other_differs": not bool(torch.equal(out[0], other[0])),
        "again_equal": bool(torch.equal(out[0], again[0]) and torch.equal(out[1], again[1])),
    }


def l1_real_keys_worker(rank: int, world_size: int, transformer_dir: str) -> dict:
    """Build the real LTX-2 DiT (48 blocks) on meta at TP=world and map its parameters through
    the application's ``checkpoint_key``; nothing is loaded."""
    from difflet.backends.neuron.core.application_base import rank_weight_bytes
    from difflet.backends.neuron.core.checkpoint import build_on_meta, shard_dim
    from difflet.backends.neuron.ltx_2.config import NeuronLtx2Config
    from difflet.backends.neuron.ltx_2.transformer import NeuronLtx2TransformerApplication
    from difflet.backends.neuron.ops_impl import parallel_mesh
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    parallel = DiffletParallelConfig(tp_degree=world_size)
    config = NeuronLtx2Config.from_pretrained(transformer_dir, tp_degree=world_size,
                                              height=480, width=704, num_frames=49)
    app = NeuronLtx2TransformerApplication(
        model_path=transformer_dir, config=config, parallel=parallel, dtype=torch.bfloat16,
        device="cpu",
    )
    parallel_mesh.init_parallel_mesh(parallel.mesh_spec)
    module = build_on_meta(app.build_module)
    names = list(module.state_dict())
    probes = (
        "transformer.transformer_blocks.0.attn1.to_q.weight",
        "transformer.transformer_blocks.0.attn1.to_out.0.weight",
        "transformer.transformer_blocks.0.audio_to_video_attn.to_k.weight",
        "transformer.transformer_blocks.0.video_to_audio_attn.to_v.bias",
        "transformer.transformer_blocks.0.ff.net.0.proj.weight",
        "transformer.transformer_blocks.0.audio_ff.net.2.weight",
        "transformer.transformer_blocks.0.attn1.norm_q.weight",
        "transformer.transformer_blocks.0.scale_shift_table",
        "transformer.proj_in.weight",
        "transformer.caption_projection.linear_1.weight",
        "transformer.proj_out.weight",
    )
    params = dict(module.named_parameters())
    return {
        "keys": sorted(app.checkpoint_key(n) for n in names) if rank == 0 else [],
        "num_names": len(names),
        "shards": {
            p: [shard_dim(module, p), list(params[p].shape),
                type(module.get_submodule(p.rpartition(".")[0])).__name__]
            for p in probes
        },
        "weight_bytes": rank_weight_bytes(module, torch.bfloat16),
        "video_seq_len": config.video_seq_len,
        "audio_seq_len": config.audio_seq_len,
    }


# --------------------------------------------------------------------------
# the composed application (L2)
# --------------------------------------------------------------------------

#: The real checkpoint's scheduler/scheduler_config.json (the orchestrator reads it).
LTX2_SCHEDULER_CONFIG: dict = {
    "_class_name": "FlowMatchEulerDiscreteScheduler", "_diffusers_version": "0.37.0.dev0",
    "base_image_seq_len": 1024, "base_shift": 0.95, "invert_sigmas": False,
    "max_image_seq_len": 4096, "max_shift": 2.05, "num_train_timesteps": 1000, "shift": 1.0,
    "shift_terminal": 0.1, "stochastic_sampling": False, "time_shift_type": "exponential",
    "use_beta_sigmas": False, "use_dynamic_shifting": True, "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


def write_tiny_ltx2_snapshot(root) -> None:
    """``write_tiny_ltx2_model`` plus the scheduler config: what the composed application reads
    when its text encoder and decoders are off."""
    write_tiny_ltx2_model(root)
    scheduler = Path(root) / "scheduler"
    scheduler.mkdir(parents=True, exist_ok=True)
    (scheduler / "scheduler_config.json").write_text(json.dumps(LTX2_SCHEDULER_CONFIG))


def tiny_conditioning(batch: int):
    """Text streams and masks for ``generate``: ``[negative, positive]`` when ``batch`` is 2."""
    gen = torch.Generator().manual_seed(7)
    shape = (batch, TINY_TEXT_SEQ_LEN, TINY_LTX2_TRANSFORMER["caption_channels"])
    mask = torch.ones(batch, TINY_TEXT_SEQ_LEN, dtype=torch.bool)
    return {"encoder_hidden_states": torch.randn(shape, generator=gen),
            "audio_encoder_hidden_states": torch.randn(shape, generator=gen),
            "encoder_attention_mask": mask, "audio_encoder_attention_mask": mask.clone()}


class _DiffusersDiT:
    """The orchestrator's DiT callable around diffusers' model itself (the reference): a bundle
    in, ``(video, audio)`` out, no masks, sigma = timestep."""

    supports_ltx_2_extra_kwargs = False
    teacache_probe_fused = False
    cfg_parallel_enabled = False

    def __init__(self, transformer_dir, *, bf16_rope: bool = False):
        from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

        self.model = LTX2VideoTransformer3DModel.from_pretrained(
            str(transformer_dir), torch_dtype=torch.float32).eval()
        if bf16_rope:
            _round_rope_to_bf16(self.model)
        self.dtype = torch.float32

    def __call__(self, bundle):
        with torch.no_grad():
            return self.model(
                hidden_states=bundle.hidden_states, audio_hidden_states=bundle.audio_hidden_states,
                encoder_hidden_states=bundle.encoder_hidden_states,
                audio_encoder_hidden_states=bundle.audio_encoder_hidden_states,
                timestep=bundle.timestep, sigma=bundle.timestep,
                video_coords=bundle.video_coords, audio_coords=bundle.audio_coords,
                return_dict=False,
            )


def reference_generate(model_dir, *, guidance_scale: float, steps: int, seed: int,
                       bf16_rope: bool = False) -> dict:
    """``LTX2Orchestrator`` driving diffusers' model on CPU fp32: the latents the application
    must reproduce (``bf16_rope``: with the TP path's bf16 RoPE tables)."""
    from difflet.models.ltx_2.pipeline import LTX2Orchestrator

    orchestrator = LTX2Orchestrator(
        model_path=str(model_dir),
        transformer=_DiffusersDiT(Path(model_dir) / "transformer", bf16_rope=bf16_rope),
        dtype=torch.float32, text_seq_len=TINY_TEXT_SEQ_LEN, **TINY_SHAPE,
    )
    batch = 2 if guidance_scale > 1.0 else 1
    out = orchestrator(**tiny_conditioning(batch), num_inference_steps=steps,
                       guidance_scale=guidance_scale,
                       generator=torch.Generator().manual_seed(seed), output_type="latent")
    return {"video": out.latents, "audio": out.audio_latents}


def l2_app_worker(rank: int, world_size: int, model_dir: str, tp: int, cfg: bool,
                  guidance_scale: float, steps: int, exec_mode: str = "eager") -> dict:
    """The composed ``TorchNeuronLtx2Application`` (text encoder off, the conditioning passed
    in) on CPU fp32 at TP=tp (x CFG 2 when ``cfg``): one ``generate``; the latents and the
    lifecycle."""
    from difflet.models.ltx_2.neuron_application import TorchNeuronLtx2Application
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    app = TorchNeuronLtx2Application(
        model_path=model_dir,
        parallel=DiffletParallelConfig(tp_degree=tp, cfg_parallel_enabled=cfg),
        dtype=torch.float32, shape=dict(TINY_SHAPE), text_seq_len=TINY_TEXT_SEQ_LEN,
        enable_text_encoder=False, exec_mode=exec_mode, device="cpu",
    )
    app.load()
    steps_seen: list = []
    app.step_hook = lambda: steps_seen.append(1)
    calls: list = []
    app.transformer_adapter.forward_hook = calls.append
    batch = 2 if guidance_scale > 1.0 else 1
    out = app.generate(**tiny_conditioning(batch), num_inference_steps=steps,
                       guidance_scale=guidance_scale, seed=0, output_type="latent")
    contract = app.dit_input_contract()
    return {
        "video": out.latents.tolist(),
        "audio": out.audio_latents.tolist(),
        "steps": len(steps_seen),
        "dit_calls": len(calls),
        "contract_hidden": list(contract["hidden_states"]["shape"]),
        "phase_seconds": sorted(app.phase_seconds),
    }


def vae_decoder_worker(rank: int, world_size: int, vae_dir: str, latent_shape) -> dict:
    """``NeuronLtx2VaeDecoderApplication`` on CPU fp32: one decode of seeded latents."""
    from difflet.backends.neuron.ltx_2.vae import NeuronLtx2VaeDecoderApplication

    app = NeuronLtx2VaeDecoderApplication(model_path=vae_dir, latent_shape=tuple(latent_shape),
                                          dtype=torch.float32, device="cpu")
    app.load()
    z = torch.randn(tuple(latent_shape), generator=torch.Generator().manual_seed(3))
    return {"out": app(z).tolist(), "report": app.load_report,
            "warmup_shapes": app.warmup_shapes}
