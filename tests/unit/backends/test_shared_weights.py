import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from difflet.backends.trainium.core import shared_weights


def make_app(
    tmp_path: Path,
    *,
    tp_degree: int = 4,
    world_size: int | None = None,
    local_ranks_size: int | None = None,
    start_rank_id: int = 0,
    dtype=torch.bfloat16,
    source: str = "model",
    **flags,
):
    """Minimal stand-in for NeuronApplicationBase's sharing-relevant surface."""
    src = tmp_path / source
    src.mkdir(parents=True, exist_ok=True)
    world = tp_degree if world_size is None else world_size
    return SimpleNamespace(
        model_path=str(src),
        neuron_config=SimpleNamespace(
            tp_degree=tp_degree,
            world_size=world,
            torch_dtype=dtype,
            start_rank_id=start_rank_id,
            local_ranks_size=world if local_ranks_size is None else local_ranks_size,
        ),
        config=SimpleNamespace(**flags),
    )


def write_shards(weights_dir: Path, ranks, payload=lambda r: None) -> None:
    weights_dir.mkdir(parents=True, exist_ok=True)
    for rank in ranks:
        body = payload(rank)
        if body is None:
            body = f"rank-{rank}".encode()
        (weights_dir / shared_weights.SHARD_TEMPLATE.format(rank=rank)).write_bytes(body)


@pytest.fixture(autouse=True)
def _cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("DIFFLET_SHARE_WEIGHTS", raising=False)
    monkeypatch.delenv("DIFFLET_SHARED_WEIGHTS_DIR", raising=False)
    monkeypatch.delenv("NXD_LAYOUT_TRANSFORMATION_OPTIONS", raising=False)


# ---------------------------------------------------------------- rank marker

@pytest.mark.parametrize(
    "flag",
    ["context_parallel_enabled", "cfg_parallel_enabled", "sp_enabled"],
)
def test_rank_marker_set_by_cp_cfg_and_sp(flag):
    assert shared_weights.has_rank_marker(SimpleNamespace(**{flag: True}))


def test_rank_marker_absent_by_default():
    assert not shared_weights.has_rank_marker(SimpleNamespace())


# ---------------------------------------------------------------- key

def test_key_changes_with_source_dtype_and_every_parallel_axis(tmp_path):
    base = shared_weights.store_key(make_app(tmp_path))
    assert base == shared_weights.store_key(make_app(tmp_path))  # stable

    for label, other in [
        ("source", make_app(tmp_path, source="other")),
        ("dtype", make_app(tmp_path, dtype=torch.float32)),
        ("tp", make_app(tmp_path, tp_degree=2)),
        ("world", make_app(tmp_path, world_size=8)),
        ("cp", make_app(tmp_path, context_parallel_enabled=True)),
        ("sp", make_app(tmp_path, sp_enabled=True)),
        ("cfg", make_app(tmp_path, cfg_parallel_enabled=True)),
    ]:
        assert shared_weights.store_key(other) != base, label


def test_cp_degrees_do_not_share_a_key(tmp_path):
    """cp=2 and cp=4 shards happen to be compatible today, but that is an
    empirical property of the modelling code rather than a contract, so the
    key separates them instead of relying on it."""
    cp2 = make_app(tmp_path, tp_degree=1, world_size=2, context_parallel_enabled=True)
    cp4 = make_app(tmp_path, tp_degree=1, world_size=4, context_parallel_enabled=True)
    assert shared_weights.store_key(cp2) != shared_weights.store_key(cp4)


def test_key_ignores_shape_and_batch(tmp_path):
    """The whole point: resolution must not partition the store."""
    base = make_app(tmp_path)
    other_shape = make_app(tmp_path)
    other_shape.config.height, other_shape.config.width = 512, 320
    assert shared_weights.store_key(other_shape) == shared_weights.store_key(base)


def test_label_records_the_parallel_layout(tmp_path):
    assert "tp4" in shared_weights.store_label(make_app(tmp_path))
    assert "w8" in shared_weights.store_label(make_app(tmp_path, world_size=8))
    assert "-cp" in shared_weights.store_label(make_app(tmp_path, context_parallel_enabled=True))
    assert "-sp" in shared_weights.store_label(make_app(tmp_path, sp_enabled=True))


# ---------------------------------------------------------------- store_dir

def test_store_dir_disabled_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DIFFLET_SHARE_WEIGHTS", "0")
    assert shared_weights.store_dir(make_app(tmp_path)) is None


def test_store_dir_disabled_when_layout_override_set(tmp_path, monkeypatch):
    monkeypatch.setenv("NXD_LAYOUT_TRANSFORMATION_OPTIONS", "{}")
    assert shared_weights.store_dir(make_app(tmp_path)) is None


def test_store_dir_lives_under_compile_cache(tmp_path):
    store = shared_weights.store_dir(make_app(tmp_path))
    assert store is not None
    assert store.parent.name == "_shared_weights"
    assert str(store).startswith(str(tmp_path / "cache"))


def test_store_dir_name_is_readable_and_digest_suffixed(tmp_path):
    hf = tmp_path / "hub" / "models--acme--Widget-XL" / "snapshots" / "abcdef1234567890" / "transformer"
    hf.mkdir(parents=True)
    app = make_app(tmp_path)
    app.model_path = str(hf)

    store = shared_weights.store_dir(app)
    assert store is not None
    name = store.name
    assert name.startswith("acme--Widget-XL__transformer__abcdef12__bfloat16__tp4__")
    assert name.endswith(shared_weights.store_key(app))


def test_store_label_falls_back_for_non_hf_paths(tmp_path):
    app = make_app(tmp_path, source="my_model")
    label = shared_weights.store_label(app)
    assert "my_model" in label
    assert "/" not in label and " " not in label


def test_store_label_is_bounded(tmp_path):
    deep = tmp_path / ("x" * 200) / "transformer"
    deep.mkdir(parents=True)
    app = make_app(tmp_path)
    app.model_path = str(deep)
    assert len(shared_weights.store_label(app)) <= 180


def test_legacy_digest_only_store_is_adopted(tmp_path):
    """Entries written before the readable prefix must not be stranded."""
    app = make_app(tmp_path, tp_degree=2)
    key = shared_weights.store_key(app)
    root = tmp_path / "cache" / "_shared_weights"
    legacy = root / key
    legacy.mkdir(parents=True)
    (legacy / "shard0.safetensors").write_bytes(b"rank-0")
    (legacy / "shard1.safetensors").write_bytes(b"rank-1")

    store = shared_weights.store_dir(app)
    assert store is not None
    assert store.name != key and store.name.endswith(key)
    assert not legacy.exists()
    assert (store / "shard0.safetensors").read_bytes() == b"rank-0"
    assert shared_weights.link_from_store(store, tmp_path / "w", app) is True


def test_store_dir_honours_explicit_override(tmp_path, monkeypatch):
    """--cache-dir does not move the store; this env var does."""
    elsewhere = tmp_path / "elsewhere"
    monkeypatch.setenv("DIFFLET_SHARED_WEIGHTS_DIR", str(elsewhere))
    store = shared_weights.store_dir(make_app(tmp_path))
    assert store is not None
    assert store.parent == elsewhere


# ---------------------------------------------------------------- rank mapping

def test_canonical_index_is_periodic_without_rank_marker(tmp_path):
    app = make_app(tmp_path, tp_degree=2, local_ranks_size=4)
    assert [shared_weights.canonical_index(r, app) for r in range(4)] == [0, 1, 0, 1]


def test_canonical_index_is_identity_with_rank_marker(tmp_path):
    app = make_app(tmp_path, tp_degree=2, local_ranks_size=4, sp_enabled=True)
    assert [shared_weights.canonical_index(r, app) for r in range(4)] == [0, 1, 2, 3]


# ---------------------------------------------------------------- publish/link

def test_publish_then_link_shares_one_inode(tmp_path):
    app = make_app(tmp_path, tp_degree=2)
    store = shared_weights.store_dir(app)
    first = tmp_path / "shape_a" / "weights"
    write_shards(first, [0, 1])
    shared_weights.publish_to_store(store, first, app)

    second = tmp_path / "shape_b" / "weights"
    assert shared_weights.link_from_store(store, second, app) is True

    for rank in (0, 1):
        a = first / shared_weights.SHARD_TEMPLATE.format(rank=rank)
        b = second / shared_weights.SHARD_TEMPLATE.format(rank=rank)
        assert b.read_bytes() == a.read_bytes()
        assert os.stat(a).st_ino == os.stat(b).st_ino


def test_link_from_empty_store_reports_failure(tmp_path):
    app = make_app(tmp_path, tp_degree=2)
    store = shared_weights.store_dir(app)
    assert shared_weights.link_from_store(store, tmp_path / "w", app) is False


def test_link_from_partial_store_reports_failure(tmp_path):
    app = make_app(tmp_path, tp_degree=2)
    store = shared_weights.store_dir(app)
    first = tmp_path / "a" / "weights"
    write_shards(first, [0, 1])
    shared_weights.publish_to_store(store, first, app)
    (store / "shard1.safetensors").unlink()

    assert shared_weights.link_from_store(store, tmp_path / "b" / "weights", app) is False


def test_publish_collapses_duplicate_ranks(tmp_path):
    """world_size > tp_degree replicates shards; they should end up one inode."""
    app = make_app(tmp_path, tp_degree=2, local_ranks_size=4)
    store = shared_weights.store_dir(app)
    weights = tmp_path / "artifact" / "weights"
    write_shards(weights, [0, 1, 2, 3], payload=lambda r: f"rank-{r % 2}".encode())

    shared_weights.publish_to_store(store, weights, app)

    name = shared_weights.SHARD_TEMPLATE.format
    assert os.stat(weights / name(rank=0)).st_ino == os.stat(weights / name(rank=2)).st_ino
    assert os.stat(weights / name(rank=1)).st_ino == os.stat(weights / name(rank=3)).st_ino
    assert sorted(p.name for p in store.glob("*.safetensors")) == [
        "shard0.safetensors",
        "shard1.safetensors",
    ]


# ------------------------------------------------- the --force corruption hazard

def test_prepare_for_write_unlinks_existing_shards(tmp_path):
    weights = tmp_path / "weights"
    write_shards(weights, [0, 1])
    shared_weights.prepare_for_write(weights)
    assert list(weights.glob(shared_weights.SHARD_GLOB)) == []


def test_rewriting_after_prepare_does_not_touch_peer_artifacts(tmp_path):
    """A --force recompile must not mutate other shapes sharing the inode.

    safetensors.save_file truncates an existing path, so without the unlink
    the rewrite would travel through the hardlink into every peer.
    """
    app = make_app(tmp_path, tp_degree=2)
    store = shared_weights.store_dir(app)
    name = shared_weights.SHARD_TEMPLATE.format

    first = tmp_path / "shape_a" / "weights"
    write_shards(first, [0, 1])
    shared_weights.publish_to_store(store, first, app)
    second = tmp_path / "shape_b" / "weights"
    assert shared_weights.link_from_store(store, second, app)

    # Simulate the recompile path: unlink, then write fresh content.
    shared_weights.prepare_for_write(second)
    write_shards(second, [0, 1], payload=lambda r: b"recompiled")

    assert (first / name(rank=0)).read_bytes() == b"rank-0"
    assert (store / "shard0.safetensors").read_bytes() == b"rank-0"
    assert (second / name(rank=0)).read_bytes() == b"recompiled"


def test_link_failure_leaves_no_partial_directory(tmp_path, monkeypatch):
    app = make_app(tmp_path, tp_degree=2)
    store = shared_weights.store_dir(app)
    first = tmp_path / "a" / "weights"
    write_shards(first, [0, 1])
    shared_weights.publish_to_store(store, first, app)

    real_link = os.link
    calls = {"n": 0}

    def flaky_link(src, dst):
        calls["n"] += 1
        if calls["n"] > 1:  # fail partway, as EXDEV would
            raise OSError(18, "Invalid cross-device link")
        return real_link(src, dst)

    monkeypatch.setattr(shared_weights.os, "link", flaky_link)
    target = tmp_path / "b" / "weights"
    assert shared_weights.link_from_store(store, target, app) is False
    assert list(target.glob(shared_weights.SHARD_GLOB)) == []
