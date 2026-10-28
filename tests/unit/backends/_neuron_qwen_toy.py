"""A tiny diffusers-layout Qwen-Image DiT for the Q1 tests, and their gloo workers (not collected
by pytest).

``write_tiny_qwen_model(root)`` writes ``root/transformer``: the ``config.json`` and a sharded
safetensors checkpoint (with its ``.safetensors.index.json``, the real snapshot's layout) of a
seeded diffusers ``QwenImageTransformer2DModel`` (2 blocks, 4 heads x 32, RoPE axes (8, 12, 12)),
and returns diffusers' own fp32 output on ``tiny_qwen_inputs()``: the model reloaded from those
files, called with ``encoder_hidden_states_mask=None`` and the packed ``img_shapes`` (every text
row of ``tiny_qwen_inputs()`` is valid). The reference never touches ``difflet.ops``.

The workers live here rather than in ``_neuron_workers.py`` so this suite can grow without
editing a file other suites append to; ``_neuron_gloo.run_ranks`` only needs them importable.
Every run, single-rank ones included, goes through ``run_ranks``: a fresh process with
``DIFFLET_BACKEND=neuron`` binds ``difflet.ops`` (which the Qwen modeling resolves at import) to
the neuron backend whatever an earlier test module in the pytest process selected, and keeps
Dynamo's caches out of the parent.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import torch

#: A diffusers transformer/config.json at toy size. ``pooled_projection_dim`` and the
#: underscore keys are in the real file and must be ignored by the config loader.
TINY_QWEN_TRANSFORMER: dict = {
    "_class_name": "QwenImageTransformer2DModel",
    "_diffusers_version": "0.34.0.dev0",
    "attention_head_dim": 32,
    "axes_dims_rope": [8, 12, 12],
    "guidance_embeds": False,
    "in_channels": 16,
    "joint_attention_dim": 24,
    "num_attention_heads": 4,
    "num_layers": 2,
    "out_channels": 4,
    "patch_size": 2,
    "pooled_projection_dim": 768,
}
#: 64x96 -> latent 8x12 -> packed 4x6 = 24 image tokens (even, so CP2 splits them); 8 text rows.
TINY_SHAPE = {"height": 64, "width": 96}
TINY_PACKED = (4, 6)
TINY_TEXT_SEQ_LEN = 8
TINY_IMAGE_SEQ_LEN = TINY_PACKED[0] * TINY_PACKED[1]
TINY_HIDDEN_SHAPE = (1, TINY_IMAGE_SEQ_LEN, TINY_QWEN_TRANSFORMER["in_channels"])
TINY_TEXT_SHAPE = (1, TINY_TEXT_SEQ_LEN, TINY_QWEN_TRANSFORMER["joint_attention_dim"])
TINY_OUT_SHAPE = (1, TINY_IMAGE_SEQ_LEN, 4 * TINY_QWEN_TRANSFORMER["out_channels"])
TINY_WEIGHT_SEED = 0
TINY_INPUT_SEED = 1
#: The orchestrator passes sigma-scale timesteps (t / 1000); diffusers' embedder scales by 1000.
TINY_TIMESTEP = 0.75
#: Valid rows of the right-padded prompt the masked forwards use (``tiny_padded_text``).
TINY_PADDED_VALID_ROWS = 5
#: Small enough that the tiny checkpoint spans several files plus an index.
TINY_MAX_SHARD_SIZE = "40KB"

QWEN_SNAPSHOT = Path(
    "/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen-Image/snapshots/"
    "75e0b4be04f60ec59a75f475837eced720f823b6"
)


def tiny_qwen_inputs(*, dtype: torch.dtype = torch.float32):
    """The fixed DiT inputs ``(hidden_states, timestep, encoder_hidden_states)``, identical in
    every process. The text has no padded rows: all eight are attended, as the DiT does."""
    gen = torch.Generator().manual_seed(TINY_INPUT_SEED)
    hidden = torch.randn(TINY_HIDDEN_SHAPE, generator=gen)
    text = torch.randn(TINY_TEXT_SHAPE, generator=gen)
    timestep = torch.tensor([TINY_TIMESTEP])
    return hidden.to(dtype), timestep.to(dtype), text.to(dtype)


def tiny_padded_text(text: torch.Tensor, valid: int = TINY_PADDED_VALID_ROWS):
    """``text`` right-padded as the text encoder writes a short prompt: rows past ``valid``
    zeroed, and its bool mask ``[1, TINY_TEXT_SEQ_LEN]`` (the first ``valid`` rows True)."""
    padded = text.clone()
    padded[:, valid:] = 0
    mask = torch.zeros(text.shape[:2], dtype=torch.bool)
    mask[:, :valid] = True
    return padded, mask


def tiny_qwen_config(**overrides):
    """``NeuronQwenImageConfig`` of the tiny model at ``TINY_SHAPE``, fp32."""
    from difflet.backends.neuron.qwen_image.config import NeuronQwenImageConfig

    geometry = {
        k: v for k, v in TINY_QWEN_TRANSFORMER.items()
        if not k.startswith("_") and k != "pooled_projection_dim"
    }
    geometry["axes_dims_rope"] = tuple(geometry["axes_dims_rope"])
    kwargs = {**geometry, **TINY_SHAPE, "text_seq_len": TINY_TEXT_SEQ_LEN,
              "torch_dtype": torch.float32}
    kwargs.update(overrides)
    return NeuronQwenImageConfig(**kwargs)


def _diffusers_model(num_layers: int, seed: int):
    """A seeded diffusers model whose every parameter is perturbed off its init, so a weight
    that fails to load (or loads into the wrong slot) moves the output."""
    from diffusers import QwenImageTransformer2DModel

    config = {k: v for k, v in TINY_QWEN_TRANSFORMER.items()
              if not k.startswith("_") and k != "pooled_projection_dim"}
    config["num_layers"] = num_layers
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = QwenImageTransformer2DModel(**config)
        with torch.no_grad():
            for param in model.parameters():
                param.add_(0.05 * torch.randn_like(param))
    return model.eval().requires_grad_(False)


def diffusers_reference(
    transformer_dir, *, num_layers: int | None = None, valid_rows: int | None = None
) -> torch.Tensor:
    """diffusers' fp32 output on ``tiny_qwen_inputs()`` from the saved files, unmasked; with
    ``num_layers`` only the first that many blocks run (the override's semantics). With
    ``valid_rows`` the prompt is its first ``valid_rows`` text rows only, diffusers' own
    unpadded call: what diffusers' masked call on ``tiny_padded_text`` computes too."""
    from diffusers import QwenImageTransformer2DModel

    model = QwenImageTransformer2DModel.from_pretrained(
        str(transformer_dir), torch_dtype=torch.float32
    ).eval()
    if num_layers is not None:
        model.transformer_blocks = model.transformer_blocks[:num_layers]
    hidden, timestep, text = tiny_qwen_inputs()
    if valid_rows is not None:
        text = text[:, :valid_rows]
    with torch.no_grad():
        return model(
            hidden_states=hidden,
            timestep=timestep,
            encoder_hidden_states=text,
            encoder_hidden_states_mask=None,
            img_shapes=[[(1, *TINY_PACKED)]],
            return_dict=False,
        )[0]


def write_tiny_qwen_model(root, *, num_layers: int = 2) -> torch.Tensor:
    """Write ``root/transformer`` (config.json with the real file's extra keys, sharded
    safetensors + index); return diffusers' fp32 output on ``tiny_qwen_inputs()``."""
    transformer_dir = Path(root) / "transformer"
    model = _diffusers_model(num_layers, TINY_WEIGHT_SEED)
    model.save_pretrained(
        str(transformer_dir), safe_serialization=True, max_shard_size=TINY_MAX_SHARD_SIZE
    )
    if not list(transformer_dir.glob("*.safetensors.index.json")):
        raise AssertionError("the tiny checkpoint must be sharded with an index, as the real one")
    # diffusers writes its own config.json; restore the real file's extra keys.
    config = json.loads((transformer_dir / "config.json").read_text())
    config.update({k: v for k, v in TINY_QWEN_TRANSFORMER.items() if k != "num_layers"})
    config["num_layers"] = num_layers
    (transformer_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return diffusers_reference(transformer_dir)


def checkpoint_keys(transformer_dir) -> set[str]:
    """Every tensor name in a diffusers checkpoint directory (index or single file)."""
    from difflet.backends.neuron.core.checkpoint import build_weight_map

    return set(build_weight_map(transformer_dir))


@contextmanager
def spec_mesh(**axes):
    """A spec-only neuron mesh (no process group), e.g. ``spec_mesh(tp=1)``."""
    import torch.distributed as dist

    from difflet.backends.neuron.ops_impl import parallel_mesh
    from difflet.pipeline.parallel_mesh import MeshSpec

    if dist.is_available() and dist.is_initialized():
        raise RuntimeError("spec_mesh() must run without a process group")
    parallel_mesh.destroy_parallel_mesh()
    parallel_mesh.init_parallel_mesh(MeshSpec(**axes))
    try:
        yield
    finally:
        parallel_mesh.destroy_parallel_mesh()


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


def q1_qwen_worker(
    rank: int,
    world_size: int,
    model_dir: str,
    tp: int,
    cp_mode: str,
    exec_mode: str,
    num_layers: int | None = None,
    sp: bool = False,
) -> dict:
    """Load the tiny DiT at TP=tp x CP=world/tp on CPU fp32 (with Megatron SP over the TP group
    when ``sp``), run the fixed inputs twice (the second time with other values at the same
    shapes), report the output and the lifecycle. In eager mode the rows the first block
    receives per stream are recorded: under SP each rank's sequence shard."""
    from torch._dynamo.utils import counters

    from difflet.backends.neuron.ops_impl import parallel_mesh
    from difflet.backends.neuron.qwen_image.config import NeuronQwenImageConfig
    from difflet.backends.neuron.qwen_image.transformer import (
        NeuronQwenImageTransformerApplication,
    )
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    cp = world_size // tp
    parallel = DiffletParallelConfig(
        tp_degree=tp, cp_degree=cp, cp_mode=cp_mode if cp > 1 else "gather_kv", sp_enabled=sp
    )
    overrides = dict(
        TINY_SHAPE,
        text_seq_len=TINY_TEXT_SEQ_LEN,
        tp_degree=tp,
        context_parallel_enabled=cp > 1,
        cp_mode=parallel.cp_mode,
        sp_enabled=sp,
        torch_dtype=torch.float32,
    )
    if num_layers is not None:
        overrides["num_layers"] = num_layers
    config = NeuronQwenImageConfig.from_pretrained(Path(model_dir) / "transformer", **overrides)
    app = NeuronQwenImageTransformerApplication(
        model_path=Path(model_dir) / "transformer",
        config=config,
        parallel=parallel,
        dtype=torch.float32,
        exec_mode=exec_mode,
        device="cpu",
    )
    graphs: list = []
    if exec_mode == "compile":
        app.compile_backend = _recording_aot_eager(graphs)
    torch._dynamo.reset()
    counters.clear()
    app.load()
    after_warmup = counters["stats"]["unique_graphs"]
    block_rows: list = []
    if exec_mode == "eager":
        app.module.transformer.transformer_blocks[0].register_forward_pre_hook(
            lambda module, args, kwargs: block_rows.append(
                [int(kwargs["hidden_states"].shape[1]),
                 int(kwargs["encoder_hidden_states"].shape[1])]
            ),
            with_kwargs=True,
        )
    hidden, timestep, text = tiny_qwen_inputs()
    out = app(hidden, timestep, text)
    other = app(hidden, timestep * 0.5, -text)
    # The orchestrator's five-argument call with an all-ones float mask: cast to the warm-up's
    # int64 and guidance dropped before the base's shape bookkeeping, so in compile mode it is
    # no unwarmed shape either.
    out_five = app(hidden, timestep, text, torch.ones(1, TINY_TEXT_SEQ_LEN), None)
    # A right-padded prompt and its mask: only the valid rows may be attended.
    padded, mask = tiny_padded_text(text)
    out_masked = app(hidden, timestep, padded, mask)
    out_unmasked_padded = app(hidden, timestep, padded)
    unwarmed = list(app.unwarmed_shapes)
    graphs_after_forwards = counters["stats"]["unique_graphs"]
    block0 = app.module.transformer.transformer_blocks[0]
    return {
        "rank": app.rank,
        "world_size": app.world_size,
        "mesh": str(parallel_mesh.get_mesh_spec()),
        "load_report": app.load_report,
        "warmup_shapes": app.warmup_shapes,
        "compiled_blocks": list(app.compiled_blocks),
        "unwarmed_shapes": unwarmed,
        "unique_graphs_after_warmup": after_warmup,
        "unique_graphs_after_forwards": graphs_after_forwards,
        "graph_breaks": sum(counters["graph_break"].values()),
        "collectives_per_graph": [_collective_count(gm) for gm in graphs],
        "num_blocks": len(app.module.transformer.transformer_blocks),
        "transformer_class": type(app.module.transformer).__name__,
        "sp_active": bool(getattr(app.module.transformer, "sp_enabled", False)),
        "block_rows": block_rows,
        "local_heads": int(block0.attn.heads),
        "processor": [type(block0.attn.processor).__name__,
                      block0.attn.processor.context_parallel_enabled,
                      block0.attn.processor.cp_mode],
        "param_numel": sum(p.numel() for p in app.module.parameters()),
        "params_on_meta": sum(p.is_meta for p in app.module.parameters()),
        "out": out.float().tolist(),
        "five_arg_equal": bool(torch.equal(out, out_five)),
        "out_masked": out_masked.float().tolist(),
        "out_unmasked_padded": out_unmasked_padded.float().tolist(),
        "other_differs": not bool(torch.equal(out, other)),
    }


def q1_real_keys_worker(rank: int, world_size: int, transformer_dir: str) -> dict:
    """Build the real Qwen-Image DiT (60 blocks) on meta at TP=world and map its parameters
    through the application's ``checkpoint_key``; nothing is loaded."""
    from difflet.backends.neuron.core.checkpoint import build_on_meta, shard_dim
    from difflet.backends.neuron.ops_impl import parallel_mesh
    from difflet.backends.neuron.qwen_image.config import NeuronQwenImageConfig
    from difflet.backends.neuron.qwen_image.transformer import (
        NeuronQwenImageTransformerApplication,
    )
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    parallel = DiffletParallelConfig(tp_degree=world_size)
    config = NeuronQwenImageConfig.from_pretrained(transformer_dir, tp_degree=world_size)
    app = NeuronQwenImageTransformerApplication(
        model_path=transformer_dir, config=config, parallel=parallel,
        dtype=torch.bfloat16, device="cpu",
    )
    parallel_mesh.init_parallel_mesh(parallel.mesh_spec)
    module = build_on_meta(app.build_module)
    names = list(module.state_dict())
    probes = (
        "transformer.transformer_blocks.0.img_mod.1.weight",
        "transformer.transformer_blocks.0.txt_mod.1.weight",
        "transformer.transformer_blocks.0.img_mlp.net.0.proj.weight",
        "transformer.transformer_blocks.0.img_mlp.net.2.weight",
        "transformer.transformer_blocks.0.attn.to_q.weight",
        "transformer.transformer_blocks.0.attn.to_out.0.weight",
        "transformer.transformer_blocks.0.attn.to_add_out.weight",
        "transformer.norm_out.linear.weight",
        "transformer.time_text_embed.timestep_embedder.linear_1.weight",
        "transformer.img_in.weight",
        "transformer.proj_out.weight",
    )
    params = dict(module.named_parameters())
    return {
        "keys": sorted(app.checkpoint_key(n) for n in names) if rank == 0 else [],
        "num_names": len(names),
        "shards": {
            p: [shard_dim(module, p), list(params[p].shape), type(
                module.get_submodule(p.rpartition(".")[0])).__name__]
            for p in probes
        },
        "buffers": {n: [str(b.device), str(b.dtype)] for n, b in module.named_buffers()},
        "image_seq_len": config.image_seq_len,
    }
