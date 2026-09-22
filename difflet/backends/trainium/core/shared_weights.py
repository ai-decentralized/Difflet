"""Shape-independent store for pre-sharded weights, shared via hardlinks.

An AOT artifact is keyed by the whole configuration, but only the NEFF actually
depends on shape. The sharded weights depend on model, dtype, tp_degree and
whether a rank marker is baked in — nothing else. Measured on HunyuanVideo at
tp=4: ``weights/tp{0..3}_sharded_checkpoint.safetensors`` are byte-identical
across 320x512x61 and 512x320x61 (11,634,459,808 B each, 46.5 GB per shape),
while ``model.pt`` differs. Every extra resolution therefore costs ~46.5 GB of
duplicate bytes for ~74 MB of new information.

This module keeps one copy under ``<DIFFLET_COMPILE_CACHE>/_shared_weights/``
and hardlinks it into each artifact's ``weights/`` directory. The names the
loader expects are unchanged, so ``application_base`` 's presharded read path
(and its missing-shard fallback) needs no modification.

Two axes collapse at once:

* across shapes — every resolution links the same inodes;
* across ranks — when no rank marker is present, ``cp``/``cfg``/``dp``/``pp``/
  ``ep`` only replicate the tp-sharded weights, so ``tp{r}`` links to
  ``shard{r % tp_degree}``.

**Hardlinks make in-place writes contagious.** ``safetensors.save_file``
truncates an existing path, and ``--force`` recompiles in place, so a naive
rewrite would silently mutate every shape sharing the inode. Everything here
unlinks before it writes, and :func:`prepare_for_write` must be called before
any code re-shards into a directory that may already hold links.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterable

from difflet import envs
from difflet.pipeline.compile_cache import normalize_dtype

logger = logging.getLogger(__name__)

SHARD_TEMPLATE = "tp{rank}_sharded_checkpoint.safetensors"
SHARD_GLOB = "tp*_sharded_checkpoint.safetensors"
_CANONICAL_TEMPLATE = "shard{index}.safetensors"
_STORE_DIRNAME = "_shared_weights"

# Bump when the key inputs or the on-disk layout change, so old entries are
# simply ignored rather than mis-reused.
_SCHEME = 2

# Set by NxD when the caller overrides weight layout options; it rewrites the
# shard files in place (application_base.shard_weights), which is exactly the
# mutation hardlinks cannot tolerate.
_LAYOUT_OVERRIDE_ENV = "NXD_LAYOUT_TRANSFORMATION_OPTIONS"


def has_rank_marker(config: Any) -> bool:
    """True when the checkpoint carries a per-rank ``global_rank.rank`` tensor.

    CP, CFG and Megatron-SP each make the backbone bake ``arange(world_size)``
    into the state dict so every rank reads its own index. Measured: with any
    of them on, ``tp2`` no longer equals ``tp0``, so the rank-periodic link
    rule below does not hold and the shards must be stored per rank.
    """
    return bool(
        getattr(config, "context_parallel_enabled", False)
        or getattr(config, "cfg_parallel_enabled", False)
        or getattr(config, "sp_enabled", False)
    )


def _key_inputs(app: Any) -> dict[str, Any]:
    """Everything that could plausibly change the shard files.

    Deliberately conservative. Measurement says a narrower key would still
    share correctly today — cp/cfg/sp all add the same 4-byte rank marker, and
    that marker's value is the global rank, so a cp=2 checkpoint is a prefix of
    a cp=4 one. But that is an empirical property of the current modelling
    code, not a contract: a model that ever baked something world_size-shaped
    into its state dict would make the narrower key silently serve the wrong
    weights. Naming every parallel axis instead makes the key auditable by
    inspection, and costs only the rare case of one model compiled at two cp
    degrees. Cross-shape sharing — the reason this store exists — is unaffected
    either way, since shape appears in neither form.
    """
    config, neuron_config = app.config, app.neuron_config
    inputs = {
        "scheme": _SCHEME,
        # Resolved source checkpoint. For a HuggingFace cache this pins both
        # the repo and the revision (…/snapshots/<sha>/<component>). The store
        # is a machine-local dedup device, so a machine-local key is fine —
        # artifacts stay self-contained because hardlinks are regular files.
        "source": os.path.realpath(str(app.model_path)),
        "dtype": normalize_dtype(neuron_config.torch_dtype),
        "tp_degree": int(neuron_config.tp_degree),
        # world_size = dp x cfg x cp x tp, so it is what separates cp=2 from
        # cp=4; the backbone only ever sees the booleans below, not the degree.
        "world_size": int(neuron_config.world_size),
        "context_parallel": bool(getattr(config, "context_parallel_enabled", False)),
        "sequence_parallel": bool(getattr(config, "sp_enabled", False)),
        "cfg_parallel": bool(getattr(config, "cfg_parallel_enabled", False)),
    }
    # Small prefix-only probes must not publish incomplete shards into the
    # backbone's store. Omit for ordinary apps to preserve existing cache keys.
    layout = getattr(app, "shared_weights_layout", None)
    if layout is not None:
        inputs["weight_layout"] = layout
    return inputs


def store_key(app: Any) -> str:
    raw = json.dumps(_key_inputs(app), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


_SAFE_LABEL = re.compile(r"[^A-Za-z0-9._-]+")
_LABEL_MAX = 180  # leave room for the digest suffix inside a 255-byte name


def _describe_source(model_path: str) -> list[str]:
    """Readable fragments for a checkpoint directory.

    A HuggingFace cache entry looks like
    ``…/models--<org>--<name>/snapshots/<revision>/<component>``, which carries
    everything worth naming. Anything else falls back to the last two path
    components.
    """
    source = Path(os.path.realpath(model_path))
    parts = source.parts
    for index, part in enumerate(parts):
        if not part.startswith("models--"):
            continue
        bits = [part[len("models--"):]]
        rest = parts[index + 1:]
        if len(rest) >= 2 and rest[0] == "snapshots":
            if rest[2:]:
                bits.append("-".join(rest[2:]))
            bits.append(rest[1][:8])
        elif rest:
            bits.append("-".join(rest))
        return bits
    parent = source.parent.name
    return [parent, source.name] if parent else [source.name]


def store_label(app: Any) -> str:
    """Human-readable prefix for the store directory.

    Naming is cosmetic — the digest appended by :func:`store_dir` is what makes
    the directory unique. This exists so that ``ls _shared_weights`` says which
    model, component and parallel layout each copy belongs to.
    """
    config, neuron_config = app.config, app.neuron_config
    bits = _describe_source(str(app.model_path))
    bits.append(normalize_dtype(neuron_config.torch_dtype))
    layout = f"tp{int(neuron_config.tp_degree)}"
    world = int(neuron_config.world_size)
    if world != int(neuron_config.tp_degree):
        layout += f"w{world}"
    for flag, suffix in (
        ("context_parallel_enabled", "cp"),
        ("sp_enabled", "sp"),
        ("cfg_parallel_enabled", "cfg"),
    ):
        if getattr(config, flag, False):
            layout += f"-{suffix}"
    bits.append(layout)
    label = _SAFE_LABEL.sub("-", "__".join(str(b) for b in bits if b)).strip("-_")
    return label[:_LABEL_MAX] or "model"


def store_dir(app: Any) -> Path | None:
    """Where this application's canonical shards live, or None if disabled.

    Returns None when sharing is switched off, or when the layout-override env
    var is set — that path rewrites shards in place and would corrupt peers.
    """
    if not envs.DIFFLET_SHARE_WEIGHTS:
        return None
    if _LAYOUT_OVERRIDE_ENV in os.environ:
        logger.info(
            "%s is set; skipping the shared weight store because that path "
            "rewrites shard files in place.",
            _LAYOUT_OVERRIDE_ENV,
        )
        return None
    try:
        override = envs.DIFFLET_SHARED_WEIGHTS_DIR
        root = (
            Path(override).expanduser()
            if override
            else Path(envs.DIFFLET_COMPILE_CACHE).expanduser() / _STORE_DIRNAME
        )
        key = store_key(app)
        target = root / f"{store_label(app)}__{key}"
        # Entries written before the readable prefix existed were named by the
        # digest alone. Adopt one rather than stranding tens of GB.
        legacy = root / key
        if legacy.is_dir() and not target.exists():
            try:
                legacy.rename(target)
                logger.info("Renamed shared weight store %s -> %s", legacy, target.name)
            except OSError:
                return legacy
        return target
    except Exception:  # noqa: BLE001 - never let sharing break a compile
        logger.warning("Could not derive a shared weight store key", exc_info=True)
        return None


def ranks_for(app: Any) -> list[int]:
    """The rank range whose shards this process writes.

    Mirrors ``ModelBuilder.shard_checkpoint``, which iterates
    ``range(start_rank_id, start_rank_id + local_ranks_size)``.
    """
    neuron_config = app.neuron_config
    start = int(neuron_config.start_rank_id)
    return list(range(start, start + int(neuron_config.local_ranks_size)))


def canonical_index(rank: int, app: Any) -> int:
    """Which canonical shard a given rank's file is a copy of.

    Without a rank marker the content depends only on the tp coordinate, so
    ranks repeat with period ``tp_degree`` — measured: tp=2/world=4 gives
    ``tp2 == tp0`` and ``tp3 == tp1``. With a rank marker every rank differs.
    """
    if has_rank_marker(app.config):
        return rank
    return rank % int(app.neuron_config.tp_degree)


def prepare_for_write(weights_dir: str | os.PathLike[str]) -> None:
    """Drop existing shard files so a re-shard cannot mutate shared inodes.

    ``safetensors.save_file`` opens for truncate; if the target is a hardlink
    into the store, writing through it rewrites every artifact that shares the
    inode. Unlinking first makes the subsequent write land on a fresh inode.
    """
    directory = Path(weights_dir)
    if not directory.is_dir():
        return
    for path in sorted(directory.glob(SHARD_GLOB)):
        try:
            path.unlink()
        except OSError:
            logger.warning("Could not unlink %s before re-sharding", path, exc_info=True)


def _relink(source: Path, target: Path) -> None:
    """Hardlink source -> target, replacing whatever target was."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    os.link(source, target)


def link_from_store(
    store: Path, weights_dir: str | os.PathLike[str], app: Any
) -> bool:
    """Populate ``weights_dir`` from the store. False if the store can't serve.

    False means "shard normally" — a missing or partial store, a cross-device
    cache dir, or any link failure. Never raises: sharing is an optimisation.
    """
    ranks = ranks_for(app)
    needed = {canonical_index(rank, app) for rank in ranks}
    sources = {index: store / _CANONICAL_TEMPLATE.format(index=index) for index in needed}
    if not all(path.is_file() for path in sources.values()):
        return False

    directory = Path(weights_dir)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        for rank in ranks:
            _relink(sources[canonical_index(rank, app)], directory / SHARD_TEMPLATE.format(rank=rank))
    except OSError as exc:
        # EXDEV (cache dir on another filesystem) is the expected one.
        logger.info("Cannot hardlink from the shared weight store (%s); sharding normally.", exc)
        prepare_for_write(directory)
        return False

    total = sum(path.stat().st_size for path in sources.values())
    logger.info(
        "Reused %d shard(s) (%.2f GB) from the shared weight store %s",
        len(ranks), total / 1e9, store,
    )
    return True


def publish_to_store(
    store: Path, weights_dir: str | os.PathLike[str], app: Any
) -> None:
    """Adopt freshly sharded files into the store and relink them in place.

    After this the artifact's shards and the store's canonical shards are the
    same inodes, so the next shape links instead of re-sharding, and duplicate
    ranks within this artifact collapse onto one copy too. Best effort.
    """
    directory = Path(weights_dir)
    try:
        store.mkdir(parents=True, exist_ok=True)
        produced: dict[int, Path] = {}
        for rank in ranks_for(app):
            path = directory / SHARD_TEMPLATE.format(rank=rank)
            if not path.is_file():
                continue
            index = canonical_index(rank, app)
            canonical = store / _CANONICAL_TEMPLATE.format(index=index)
            if index not in produced:
                if not canonical.is_file():
                    _relink(path, canonical)
                produced[index] = canonical
            # Point the artifact at the canonical inode. For a duplicate rank
            # this also reclaims the redundant copy we just wrote.
            _relink(produced[index], path)
    except OSError as exc:
        logger.info("Could not publish shards to the shared weight store (%s).", exc)


def describe(store: Path | None, ranks: Iterable[int]) -> str:
    if store is None:
        return "shared weight store disabled"
    return f"shared weight store {store} for ranks {sorted(ranks)}"
