"""The structural checks every neuron DiT config and DiT application runs. Torch-free.

Wan, Qwen-Image and LTX-2 each validate their config before the mesh and the meta build, so a
wrong shape or topology fails with a message instead of building a module that ignores it or
fails deep in its first forward (a ragged split surfaces as a shape error in a collective, or
in the block graph's trace). Most rules are the same for the three: the block count against the
checkpoint's, the heads TP shards, the CP mode, the cp degree against the CP flag, the
sequence a CP or SP group splits, Ulysses' head split, and the pairs of modes that claim the
same lanes. Their applications also check that the config, which sizes the module, and the
parallel config, which sizes the mesh, describe the same topology. Those rules live here, each
with the model's name in its message.

What differs stays in each model's ``validate()``: the order of the checks, the patch and VAE
geometry, the length CP splits (Wan's and Qwen-Image's image tokens, LTX-2's video tokens), and
the modes a model does not run (LTX-2's ring and SP, Qwen-Image's zero_cond_t under SP).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from difflet.pipeline.parallel_config import CP_MODES


def check_layer_count(model: str, num_layers: Any, checkpoint_num_layers: Any) -> None:
    """At least one block, and no more than the checkpoint holds (``num_layers=N`` builds the
    first N blocks of the checkpoint for an on-device parity check)."""
    if int(num_layers) < 1:
        raise ValueError(f"{model} num_layers={num_layers} must be >= 1")
    if checkpoint_num_layers is not None and int(num_layers) > int(checkpoint_num_layers):
        raise ValueError(
            f"{model} num_layers={num_layers} exceeds the checkpoint's "
            f"{checkpoint_num_layers} blocks"
        )


def check_heads_divide_tp(model: str, heads: Any, tp: Any, *, what: str = "") -> None:
    """TP shards the attention heads (and with them every column-parallel projection), so it
    has to divide them evenly; a ragged split would only surface deep in the weight load.
    ``what`` names the stream when a model has several ("video", "audio")."""
    if int(heads) % int(tp):
        label = f"{heads} {what} attention heads" if what else f"{heads} attention heads"
        raise ValueError(f"{model} has {label}, which does not divide tp={tp}")


def check_cp_mode(cp_mode: str) -> None:
    if cp_mode not in CP_MODES:
        raise ValueError(f"cp_mode must be one of {CP_MODES}, got {cp_mode!r}")


def check_cp_enabled(cp: int | None, enabled: bool) -> None:
    """A CP degree above 1 without the CP flag would build a module that runs the whole
    sequence on every rank of the CP group."""
    if cp is not None and cp > 1 and not enabled:
        raise ValueError(
            f"cp_degree={cp} needs context_parallel_enabled=True (the module would run "
            "the full sequence on every rank of the CP group)"
        )


def exclusive_modes(model: str, first: str, second: str, reason: str) -> ValueError:
    """The error for two modes that claim the same dimension or the same lanes, for the caller
    to raise when both are on."""
    return ValueError(f"{model} {first} and {second} are mutually exclusive ({reason})")


def check_cp_split(model: str, name: str, length: int, layout: str, cp: int) -> None:
    """CP scatters ``length`` tokens evenly over the CP group (no padding)."""
    if length % cp:
        raise ValueError(
            f"{model} {name}={length} ({layout}) is not divisible by cp_degree={cp}"
        )


def check_even_split(what: str, length: int, axis: str, size: int) -> None:
    """A sequence ``scatter_to_process_group_spmd`` splits over a group of ``size`` (no
    padding): ``what`` describes it ("Wan sequence parallelism splits image_seq_len=4680")."""
    if length % size:
        raise ValueError(f"{what} over {axis}={size}, which does not divide it")


def check_ulysses_heads(heads: int, tp: int, cp: int) -> None:
    """Ulysses trades this rank's sequence shard for a head shard over the CP group: the
    rank's ``heads / tp`` heads must split evenly over cp."""
    if (heads // tp) % cp:
        raise ValueError(
            f"cp_mode='ulysses' shards this rank's {heads // tp} heads (heads={heads} / "
            f"tp={tp}) over cp_degree={cp}; num_attention_heads must be divisible by "
            "tp_degree * cp_degree"
        )


# ---------------------------------------------------------- config against parallel config


def check_tp_matches(name: str, config: Any, parallel: Any) -> None:
    """The config sizes the module and the parallel config the mesh: their tp must agree."""
    if int(config.tp_degree) != int(parallel.tp_degree):
        raise ValueError(
            f"{name} has tp_degree={config.tp_degree} but the parallel config has "
            f"tp_degree={parallel.tp_degree}; the module would be sharded for one and "
            "the mesh built for the other"
        )


def check_flag_matches(name: str, config: Any, parallel: Any, field: str) -> None:
    """A mode flag (``cfg_parallel_enabled``, ``sp_enabled``) the module and the mesh must
    agree on; a parallel config without the attribute has the mode off."""
    mine, theirs = getattr(config, field), getattr(parallel, field, False)
    if bool(mine) != bool(theirs):
        raise ValueError(
            f"{name} has {field}={mine} but the parallel config has {field}={theirs}"
        )


def resolve_cp_degree(name: str, config: Any, parallel: Any) -> Any:
    """``config`` with ``cp_degree`` filled from ``parallel`` (a copy) when it is None, after
    checking that both describe the same CP group: the CP flag on iff ``parallel.cp_degree >
    1``, the same ``cp_mode``, the same degree when the config names one. The modeling calls
    ``init_parallel_mesh(config)`` again under CP, which is a no-op only when both describe the
    same mesh, and a filled ``cp_degree`` lets ``validate()`` run the cp divisibility checks."""
    cp = int(getattr(parallel, "cp_degree", 1) or 1)
    if bool(config.context_parallel_enabled) != (cp > 1):
        raise ValueError(
            f"{name} has context_parallel_enabled={config.context_parallel_enabled} "
            f"but the parallel config has cp_degree={cp}"
        )
    if cp > 1 and str(config.cp_mode) != str(parallel.cp_mode):
        raise ValueError(
            f"{name} has cp_mode={config.cp_mode!r} but the parallel config has "
            f"cp_mode={parallel.cp_mode!r}"
        )
    if config.cp_degree is None:
        return dataclasses.replace(config, cp_degree=cp)
    if int(config.cp_degree) != cp:
        raise ValueError(
            f"{name} has cp_degree={config.cp_degree} but the parallel config has "
            f"cp_degree={cp}"
        )
    return config


def require_bf16_on_device(model: str, device_type: str, dtype: Any) -> None:
    """Decision D42: the neuron TP linears all-reduce in the activation dtype, which was
    checked against NxD for bf16 only, so no other dtype reaches the device unexamined."""
    import torch

    if device_type == "neuron" and dtype is not torch.bfloat16:
        raise ValueError(
            f"the {model} DiT runs in torch.bfloat16 on the neuron device, got {dtype} "
            "(decision D42: the TP layers all-reduce in the activation dtype, checked "
            "against NxD for bf16 only)"
        )


def truncated_block_prefixes(config: Any, block_prefix: str) -> tuple[str, ...]:
    """The checkpoint prefixes of the blocks past ``config.num_layers`` when it was cut below
    the checkpoint's own count: in the checkpoint on purpose, so reported as skipped rather
    than unexpected."""
    total = config.checkpoint_num_layers
    if total is None or int(total) <= int(config.num_layers):
        return ()
    return tuple(f"{block_prefix}{i}." for i in range(int(config.num_layers), int(total)))


__all__ = [
    "check_cp_enabled",
    "check_cp_mode",
    "check_cp_split",
    "check_even_split",
    "check_flag_matches",
    "check_heads_divide_tp",
    "check_layer_count",
    "check_tp_matches",
    "check_ulysses_heads",
    "exclusive_modes",
    "require_bf16_on_device",
    "resolve_cp_degree",
    "truncated_block_prefixes",
]
