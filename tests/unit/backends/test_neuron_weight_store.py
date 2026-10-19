"""The neuron backend's pre-sharded weight store (``difflet/backends/neuron/core/weight_store.py``).

On CPU (and gloo for the multi-rank cases), with ToyApplication over a fp32 checkpoint: the
store round trip loads the same bits as the checkpoint; the entry key separates dtype, layer
count, TP degree, checkpoint revision and store format; a corrupted, partial or mismatched
entry falls back to the checkpoint and is rewritten; concurrent writers of one entry leave one
valid copy; the off switch and the disk-space check skip the store. The composed applications
of the three models are pinned in ``test_neuron_e2e_pins.py``; ``test_e2e_pins_with_the_store``
there runs some of those pins again through a written and then a read store.
"""

import fcntl
import json
import multiprocessing as mp
import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from difflet.backends.neuron.core import weight_store  # noqa: E402
from difflet.backends.neuron.ops_impl import parallel_mesh  # noqa: E402
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from tests.unit.backends._neuron_gloo import run_ranks  # noqa: E402
from tests.unit.backends._neuron_toy import (  # noqa: E402
    TOY_APP_BLOCKS,
    TOY_APP_DIM,
    TOY_APP_HIDDEN,
    ToyApplication,
    toy_blocks_input,
    toy_blocks_reference,
    write_toy_checkpoint,
)
from tests.unit.backends._neuron_workers import (  # noqa: E402
    ws_concurrent_launch,
    ws_toy_worker,
)


@pytest.fixture(autouse=True)
def _clean_mesh(monkeypatch):
    monkeypatch.delenv("DIFFLET_EXEC_MODE", raising=False)
    parallel_mesh.destroy_parallel_mesh()
    yield
    parallel_mesh.destroy_parallel_mesh()


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    pytest.importorskip("safetensors.torch")
    pytest.importorskip("accelerate")
    x = toy_blocks_input(1, 16, TOY_APP_DIM)
    weights, _, _ = toy_blocks_reference(TOY_APP_BLOCKS, TOY_APP_DIM, TOY_APP_HIDDEN, x)
    directory = tmp_path_factory.mktemp("ws_ckpt")
    write_toy_checkpoint(directory, weights, num_files=2)
    return SimpleNamespace(dir=str(directory), x=x, weights=weights)


@pytest.fixture
def store(tmp_path, monkeypatch):
    root = tmp_path / "store"
    monkeypatch.setenv("DIFFLET_NEURON_WEIGHT_STORE", str(root))
    return root


def _load(toy, dtype=torch.bfloat16, model_dir=None, **kwargs):
    parallel_mesh.destroy_parallel_mesh()
    app = ToyApplication(model_path=model_dir or toy.dir, parallel=DiffletParallelConfig(),
                         dtype=dtype, exec_mode="eager", device="cpu", **kwargs)
    app.load()
    return app


def _state(app):
    return {name: t.detach().clone() for name, t in app.module.state_dict().items()}


def _assert_same_bits(a, b):
    assert a.keys() == b.keys()
    for name in a:
        assert a[name].dtype == b[name].dtype, name
        assert torch.equal(a[name].view(torch.uint8) if a[name].dtype == torch.bool else a[name],
                           b[name]), name


def _entry_files(root):
    (entry,) = list(root.iterdir())
    return entry, sorted(p.name for p in entry.iterdir() if not p.name.endswith(".lock"))


# ------------------------------------------------------------------------------ round trip


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_round_trip_is_bit_identical_to_the_checkpoint_load(toy, store, monkeypatch, dtype):
    if dtype == torch.float32:  # the fp32 checkpoint in fp32: written only when asked for
        monkeypatch.setenv("DIFFLET_NEURON_WEIGHT_STORE_ALL", "1")
    written = _load(toy, dtype)
    assert written.weight_store.status == "written"
    entry, files = _entry_files(store)
    assert files == ["replicated.safetensors", "replicated.safetensors.manifest.json"]
    assert f"__{str(dtype).removeprefix('torch.')}__tp1__" in entry.name

    hit = _load(toy, dtype)
    assert hit.weight_store.status == "hit"
    assert hit.weight_store.bytes == (entry / "replicated.safetensors").stat().st_size

    monkeypatch.setenv("DIFFLET_NEURON_WEIGHT_STORE", "0")
    direct = _load(toy, dtype)
    assert direct.weight_store.status == "off"
    _assert_same_bits(_state(hit), _state(direct))
    _assert_same_bits(_state(written), _state(direct))
    assert hit.load_report == direct.load_report == {"missing": [], "unexpected": []}
    assert all(p.dtype == dtype for p in hit.module.parameters())  # the load dtype is kept
    x = toy.x.to(dtype)
    with torch.no_grad():
        assert torch.equal(hit(x), direct(x))


def test_a_checkpoint_already_in_the_load_dtype_is_not_written(toy, store, monkeypatch):
    # The same bytes on the same disk load no faster (Qwen-Image's bf16 DiT on trn2: 311 s cold
    # and 3.6 s warm with or without the store), so the store would only cost disk and a write.
    app = _load(toy, torch.float32)
    assert app.weight_store.status == "not written"
    assert app.weight_store.detail.startswith("no gain: the checkpoint holds these tensors")
    assert "DIFFLET_NEURON_WEIGHT_STORE_ALL=1" in app.weight_store.detail
    assert not store.exists() or not any(store.iterdir())
    monkeypatch.setenv("DIFFLET_NEURON_WEIGHT_STORE_ALL", "1")
    assert _load(toy, torch.float32).weight_store.status == "written"
    assert _load(toy, torch.float32).weight_store.status == "hit"
    monkeypatch.setenv("DIFFLET_NEURON_WEIGHT_STORE_ALL", "0")
    assert _load(toy, torch.float32).weight_store.status == "hit"  # an entry is always read


def test_the_store_file_is_plain_safetensors_in_the_load_dtype(toy, store):
    from safetensors.torch import load_file

    app = _load(toy, torch.bfloat16)
    entry, _ = _entry_files(store)
    stored = load_file(str(entry / "replicated.safetensors"))
    assert stored.keys() == app.module.state_dict().keys()
    assert {t.dtype for t in stored.values()} == {torch.bfloat16}
    manifest = json.loads((entry / "replicated.safetensors.manifest.json").read_text())
    assert manifest["format"] == weight_store.STORE_FORMAT
    assert manifest["bytes"] == (entry / "replicated.safetensors").stat().st_size
    assert [t[0] for t in manifest["tensors"]] == list(stored)  # module order is file order
    assert manifest["load_report"] == {"missing": [], "unexpected": []}


def test_the_load_report_is_replayed_on_a_hit(toy, store):
    # Two blocks of a four-block checkpoint: the other blocks' tensors are "unexpected".
    written = _load(toy, n_blocks=2)
    hit = _load(toy, n_blocks=2)
    assert hit.weight_store.status == "hit"
    assert hit.load_report == written.load_report
    assert hit.load_report["unexpected"] and not hit.load_report["missing"]


# ------------------------------------------------------------------------------------- key


def _entry_names(root):
    return sorted(p.name for p in root.iterdir())


def test_the_key_separates_dtype_layers_and_format(toy, store, monkeypatch):
    monkeypatch.setenv("DIFFLET_NEURON_WEIGHT_STORE_ALL", "1")  # fp32 is written too
    _load(toy, torch.bfloat16)
    _load(toy, torch.float32)
    _load(toy, torch.bfloat16, n_blocks=2)
    assert len(_entry_names(store)) == 3
    _load(toy, torch.bfloat16)  # the first again: a hit, no new entry
    assert len(_entry_names(store)) == 3
    monkeypatch.setattr(weight_store, "STORE_FORMAT", weight_store.STORE_FORMAT + 1)
    app = _load(toy, torch.bfloat16)
    assert app.weight_store.status == "written" and len(_entry_names(store)) == 4


def test_the_key_follows_the_checkpoint_revision(toy, store, tmp_path):
    copy = tmp_path / "ckpt_copy"
    write_toy_checkpoint(copy, toy.weights, num_files=2)
    first = _load(toy, model_dir=str(copy))
    assert first.weight_store.status == "written"
    assert _load(toy, model_dir=str(copy)).weight_store.status == "hit"
    # A new revision of a file (here: new content, new mtime) is a new entry, never a stale hit.
    changed = dict(toy.weights)
    changed["blocks.0.norm1.weight"] = changed["blocks.0.norm1.weight"] + 1
    write_toy_checkpoint(copy, changed, num_files=2)
    second = _load(toy, model_dir=str(copy))
    assert second.weight_store.status == "written"
    assert second.weight_store.directory != first.weight_store.directory
    assert torch.equal(second.module.blocks[0].norm1.weight,
                       changed["blocks.0.norm1.weight"].to(torch.bfloat16))


def test_the_key_separates_tp_degrees(toy, store):
    env = {"DIFFLET_NEURON_WEIGHT_STORE": str(store)}
    tp4 = run_ranks(ws_toy_worker, toy.dir, "bfloat16", 4, world_size=4, env=env)
    tp2 = run_ranks(ws_toy_worker, toy.dir, "bfloat16", 2, world_size=2, env=env)
    assert len({r["directory"] for r in tp4}) == 1 and len({r["directory"] for r in tp2}) == 1
    assert tp4[0]["directory"] != tp2[0]["directory"]
    assert "__tp4__" in tp4[0]["directory"] and "__tp2__" in tp2[0]["directory"]


# ------------------------------------------------------------------- multi-rank, concurrency


def test_four_gloo_ranks_write_their_slices_then_load_them_bit_identically(toy, store):
    env = {"DIFFLET_NEURON_WEIGHT_STORE": str(store)}
    first = run_ranks(ws_toy_worker, toy.dir, "bfloat16", 4, world_size=4, env=env)
    assert [r["status"] for r in first] == ["written"] * 4
    entry, files = _entry_files(store)
    shards = [f"tp{r}-of-4.safetensors" for r in range(4)]
    assert files == sorted(["replicated.safetensors", "replicated.safetensors.manifest.json",
                            *shards, *(f"{s}.manifest.json" for s in shards)])
    # The replicated tensors are written once, by one rank.
    assert sum("replicated" in r["detail"] for r in first) == 1

    second = run_ranks(ws_toy_worker, toy.dir, "bfloat16", 4, world_size=4, env=env)
    assert [r["status"] for r in second] == ["hit"] * 4
    direct = run_ranks(ws_toy_worker, toy.dir, "bfloat16", 4, world_size=4,
                       env={"DIFFLET_NEURON_WEIGHT_STORE": "off"})
    assert [r["status"] for r in direct] == ["off"] * 4
    for a, b, c in zip(first, second, direct):
        assert a["tensors"] == b["tensors"] == c["tensors"]
        assert a["out"] == b["out"] == c["out"]
        assert a["report"] == b["report"] == c["report"]
    assert len({r["out"] for r in second}) == 1


def test_two_processes_of_one_tp_rank_share_its_file(toy, store):
    # tp2 x CFG2: global ranks 0 and 2 (and 1 and 3) hold the same TP rank; one of each pair
    # writes the file (the other skips or finds it written), and all four load the same bits.
    env = {"DIFFLET_NEURON_WEIGHT_STORE": str(store)}
    first = run_ranks(ws_toy_worker, toy.dir, "bfloat16", 2, world_size=4, env=env)
    _, files = _entry_files(store)
    assert files == sorted(["replicated.safetensors", "replicated.safetensors.manifest.json",
                            "tp0-of-2.safetensors", "tp0-of-2.safetensors.manifest.json",
                            "tp1-of-2.safetensors", "tp1-of-2.safetensors.manifest.json"])
    second = run_ranks(ws_toy_worker, toy.dir, "bfloat16", 2, world_size=4, env=env)
    assert [r["status"] for r in second] == ["hit"] * 4
    assert [r["tensors"] for r in first] == [r["tensors"] for r in second]
    assert first[0]["tensors"] == first[2]["tensors"] and first[1]["tensors"] == first[3]["tensors"]


def test_concurrent_launches_leave_one_valid_entry(toy, store):
    ctx = mp.get_context("spawn")
    barrier, results = ctx.Barrier(4), ctx.Queue()
    procs = [ctx.Process(target=ws_concurrent_launch, args=(toy.dir, str(store), barrier, results))
             for _ in range(4)]
    for p in procs:
        p.start()
    got = [results.get(timeout=120) for _ in procs]
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0
    statuses = [s for s, _ in got]
    assert statuses.count("written") >= 1 and set(statuses) <= {"written", "not written", "hit"}
    assert len({d for _, d in got}) == 1
    _, files = _entry_files(store)
    assert files == ["replicated.safetensors", "replicated.safetensors.manifest.json"]
    assert _load(toy).weight_store.status == "hit"


def test_a_file_locked_by_another_writer_is_not_written(toy, store):
    first = _load(toy)
    entry, _ = _entry_files(store)
    (entry / "replicated.safetensors.manifest.json").unlink()
    fd = os.open(entry / "replicated.safetensors.lock", os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        app = _load(toy)
    finally:
        os.close(fd)
    assert app.weight_store.status == "not written"
    assert "being written by another process" in app.weight_store.detail
    _assert_same_bits(_state(app), _state(first))


# --------------------------------------------------------------- invalid entries fall back


def _corrupt_truncate(entry):
    with open(entry / "replicated.safetensors", "r+b") as f:
        f.truncate(os.path.getsize(entry / "replicated.safetensors") - 7)


def _corrupt_data(entry):
    # The first data bytes, which the sampled checksum always covers.
    path = entry / "replicated.safetensors"
    with open(path, "r+b") as f:
        length = int.from_bytes(f.read(8), "little")
        f.seek(8 + length)
        f.write(b"\xff" * 64)


def _corrupt_header(entry):
    with open(entry / "replicated.safetensors", "r+b") as f:
        f.seek(9)
        f.write(b"#")


def _drop_manifest(entry):  # a data file without its manifest: a write that never committed
    (entry / "replicated.safetensors.manifest.json").unlink()


def _drop_data(entry):
    (entry / "replicated.safetensors").unlink()


def _mismatched_manifest(entry):
    path = entry / "replicated.safetensors.manifest.json"
    body = json.loads(path.read_text())
    body["tensors"][0][2] = [1, *body["tensors"][0][2]]
    path.write_text(json.dumps(body))


def _garbage_manifest(entry):
    (entry / "replicated.safetensors.manifest.json").write_text("{not json")


@pytest.mark.parametrize("corrupt, reason", [
    (_corrupt_truncate, "bytes, manifest says"),
    (_corrupt_data, "data checksum mismatch"),
    (_corrupt_header, "header differs"),
    (_drop_manifest, "no entry"),
    (_drop_data, "data file unreadable"),
    (_mismatched_manifest, "manifest tensors differ"),
    (_garbage_manifest, "unreadable manifest"),
])
def test_a_bad_entry_falls_back_and_is_rewritten(toy, store, corrupt, reason):
    reference = _state(_load(toy))
    entry, _ = _entry_files(store)
    corrupt(entry)
    app = _load(toy)
    assert app.weight_store.status == "written"
    if reason != "no entry":
        assert reason in app.weight_store.detail
    _assert_same_bits(_state(app), reference)
    again = _load(toy)
    assert again.weight_store.status == "hit"
    _assert_same_bits(_state(again), reference)


def test_a_dead_writers_temp_file_is_ignored_and_cleaned(toy, store):
    reference = _state(_load(toy))
    entry, _ = _entry_files(store)
    _drop_manifest(entry)
    stale = entry / ".replicated.safetensors.tmp-1-deadbeef"
    stale.write_bytes(b"partial")
    app = _load(toy)
    assert app.weight_store.status == "written" and not stale.exists()
    _assert_same_bits(_state(app), reference)


def test_a_failed_store_read_falls_back_to_the_checkpoint(toy, store, monkeypatch):
    reference = _state(_load(toy))

    def broken(self, targets):
        raise OSError("injected read failure")

    monkeypatch.setattr(weight_store._StoreFile, "read_into", broken)
    app = _load(toy)
    assert app.weight_store.status == "not written"
    assert "injected read failure" in app.weight_store.detail
    _assert_same_bits(_state(app), reference)


def test_a_failed_store_write_never_fails_the_load(toy, store, monkeypatch):
    def full_disk(self, name, tensor):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(weight_store._FileWriter, "add", full_disk)
    app = _load(toy)
    assert app.weight_store.status == "not written"
    assert "write failed" in app.weight_store.detail and "No space left" in app.weight_store.detail
    entry, files = _entry_files(store)
    assert files == []  # the temp file is gone and nothing was published
    monkeypatch.undo()
    monkeypatch.setenv("DIFFLET_NEURON_WEIGHT_STORE", "0")
    _assert_same_bits(_state(app), _state(_load(toy)))


def test_a_failed_checkpoint_load_leaves_no_entry(toy, store, tmp_path):
    ckpt = tmp_path / "partial_ckpt"
    partial = {k: v for k, v in toy.weights.items() if k != "blocks.3.mlp.down.bias"}
    write_toy_checkpoint(ckpt, partial)
    with pytest.raises(ValueError, match="had no checkpoint entry"):
        _load(toy, model_dir=str(ckpt))
    (entry,) = list(store.iterdir())
    assert sorted(p.name for p in entry.iterdir() if not p.name.endswith(".lock")) == []


# ----------------------------------------------------------------- off switch, disk space


@pytest.mark.parametrize("value", ["0", "off", "false", "none", "no", "OFF"])
def test_the_off_switch(toy, tmp_path, monkeypatch, value):
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("DIFFLET_NEURON_WEIGHT_STORE", value)
    assert weight_store.store_root() is None
    assert _load(toy).weight_store.status == "off"
    assert not (tmp_path / "cache").exists()


def test_the_default_location_is_under_the_compile_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("DIFFLET_NEURON_WEIGHT_STORE")
    assert weight_store.store_root() == tmp_path / "cache" / "neuron" / "weights"
    monkeypatch.setenv("DIFFLET_NEURON_WEIGHT_STORE", "")
    assert weight_store.store_root() == tmp_path / "cache" / "neuron" / "weights"
    monkeypatch.setenv("DIFFLET_NEURON_WEIGHT_STORE", str(tmp_path / "elsewhere"))
    assert weight_store.store_root() == tmp_path / "elsewhere"


def test_a_short_disk_skips_the_write(toy, store, monkeypatch):
    reference = _state(_load(toy))  # written
    for path in store.iterdir():
        for f in path.iterdir():
            f.unlink()
    usage = SimpleNamespace(total=100 * 1024**3, used=0, free=weight_store.RESERVE_BYTES)
    monkeypatch.setattr(weight_store.shutil, "disk_usage", lambda path: usage)
    app = _load(toy)
    assert app.weight_store.status == "not written"
    assert "GB free" in app.weight_store.detail and "reserve" in app.weight_store.detail
    entry, files = _entry_files(store)
    assert files == []
    _assert_same_bits(_state(app), reference)


def test_the_evidence_line(toy, store, capsys):
    _load(toy)
    _load(toy)
    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("[weight store]")]
    assert len(lines) == 2
    assert lines[0].startswith("[weight store] rank 0/1 ToyApplication: written ")
    assert lines[1].startswith("[weight store] rank 0/1 ToyApplication: hit ")
