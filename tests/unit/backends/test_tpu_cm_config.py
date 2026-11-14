"""backends/tpu/ops_impl/cm_config.py: per-shape collective-matmul parameters.

With no table and the cost model off the caller's default comes back unchanged
(today's behaviour); a table entry wins for its exact shape only; the shape log
records each resolution; the table tag changes with the table; and the cost
model, when switched on, picks the ring where a hop's compute does not hide its
transfer (Qwen-Image's 256-row text stream, q/k/v only, 2026-11-13) and the
kernel where it does (FLUX's single block, q/k/v plus the MLP up-projection).
Per hop the ratio compute/transfer is ~ 2 * sum(widths) * link_bw / flops, so
it is the widths, not the rows, that decide.
"""

from __future__ import annotations

import json

import pytest

from difflet.backends.tpu.ops_impl import cm_config

CHIP = "TPU v5 lite"
QWEN_TXT = cm_config.AgShape(CHIP, 1, 256, 3072, (768, 768, 768), "bfloat16", 4)
FLUX_SINGLE = cm_config.AgShape(CHIP, 1, 1152, 3072, (768, 768, 768, 3072), "bfloat16", 4)
DEFAULT = cm_config.AgConfig(use_kernel=True, tile_width=256, resident=False)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (cm_config.TUNING_ENV, cm_config.COST_MODEL_ENV, cm_config.LOG_SHAPES_ENV):
        monkeypatch.delenv(name, raising=False)
    cm_config._load_table.cache_clear()


def test_default_passes_through():
    assert cm_config.choose_ag(QWEN_TXT, DEFAULT) == (DEFAULT, "default")
    rs = cm_config.RsShape(CHIP, 1, 4096, (768,), 3072, "bfloat16", 4)
    rs_default = cm_config.RsConfig(use_kernel=False)
    assert cm_config.choose_rs(rs, rs_default) == (rs_default, "default")
    assert cm_config.table_tag() == ""


def _write_table(tmp_path, tile=768):
    path = tmp_path / "tuning.json"
    path.write_text(json.dumps({"ag": [{
        "shape": {"chip": CHIP, "batch": 1, "rows": 256, "k": 3072,
                  "widths": [768, 768, 768], "dtype": "bfloat16", "ring": 4},
        "config": {"use_kernel": True, "tile_width": tile, "resident": True}}]}))
    return path


def test_table_hit_only_for_its_shape(monkeypatch, tmp_path):
    monkeypatch.setenv(cm_config.TUNING_ENV, str(_write_table(tmp_path)))
    config, source = cm_config.choose_ag(QWEN_TXT, DEFAULT)
    assert source == "table" and config == cm_config.AgConfig(True, 768, True)
    assert cm_config.choose_ag(FLUX_SINGLE, DEFAULT) == (DEFAULT, "default")


def test_table_tag_follows_the_table(monkeypatch, tmp_path):
    monkeypatch.setenv(cm_config.TUNING_ENV, str(_write_table(tmp_path, 768)))
    first = cm_config.table_tag()
    _write_table(tmp_path, 512)
    assert first and cm_config.table_tag() != first


def test_shape_log(monkeypatch, tmp_path):
    log = tmp_path / "shapes.jsonl"
    monkeypatch.setenv(cm_config.LOG_SHAPES_ENV, str(log))
    cm_config.choose_ag(QWEN_TXT, DEFAULT)
    record = json.loads(log.read_text().splitlines()[0])
    assert record["kind"] == "ag" and record["source"] == "default"
    assert record["shape"]["rows"] == 256 and record["config"]["tile_width"] == 256


def test_cost_model_ring_for_short_streams_kernel_for_long(monkeypatch):
    monkeypatch.setenv(cm_config.COST_MODEL_ENV, "1")
    text, source = cm_config.choose_ag(QWEN_TXT, DEFAULT)
    assert source == "model" and not text.use_kernel
    image, _ = cm_config.choose_ag(FLUX_SINGLE, DEFAULT)
    assert image.use_kernel
    assert cm_config.table_tag()  # the cost model is part of the ops' names too


def test_cost_model_parts(monkeypatch):
    """``tile`` alone keeps the caller's kernel choice and only sizes the tile;
    each set of parts names its ops differently; unknown parts are refused."""
    monkeypatch.setenv(cm_config.COST_MODEL_ENV, "tile")
    text, _ = cm_config.choose_ag(QWEN_TXT, DEFAULT)
    assert text.use_kernel and text.tile_width == 384  # 768 = 2 x 384: no 512 divides it
    tile_tag = cm_config.table_tag()
    monkeypatch.setenv(cm_config.COST_MODEL_ENV, "kernel,tile")
    assert cm_config.table_tag() not in ("", tile_tag)
    monkeypatch.setenv(cm_config.COST_MODEL_ENV, "tiles")
    with pytest.raises(ValueError, match="unknown parts"):
        cm_config.choose_ag(QWEN_TXT, DEFAULT)


def test_tile_largest_that_fits(monkeypatch, tmp_path):
    """Largest multiple of 128 dividing every width, up to max_tile; smaller when the
    VMEM estimate does not fit (torch_tpu's CreateTiling, shrink until it fits)."""
    monkeypatch.setenv(cm_config.COST_MODEL_ENV, "tile")
    ltx_ffn = cm_config.AgShape(CHIP, 1, 578, 4096, (4096,), "bfloat16", 4)
    assert cm_config.choose_ag(ltx_ffn, DEFAULT)[0].tile_width == 512   # the default cap
    path = tmp_path / "hw.json"
    path.write_text(json.dumps({"hw": {"max_tile": 4096}}))
    monkeypatch.setenv(cm_config.TUNING_ENV, str(path))
    assert cm_config.choose_ag(ltx_ffn, DEFAULT)[0].tile_width == 4096  # 87 MiB < 96
    path.write_text(json.dumps({"hw": {"max_tile": 4096, "vmem_bytes": 48 * 2**20}}))
    cm_config._load_table.cache_clear()
    # 2048 needs 55 MiB (halves 14, two weight tiles 34, outputs 5, product 2)
    assert cm_config.choose_ag(ltx_ffn, DEFAULT)[0].tile_width == 1024  # 48 MiB budget


def test_vmem_budget_per_chip():
    assert cm_config.vmem_budget("TPU v5 lite") == 96 * 2**20
    assert cm_config.vmem_budget("TPU v5p") == 95 * 2**20 * 3 // 4
    assert cm_config.vmem_budget("TPU v5 lite", {"vmem_bytes": 1}) == 1


def test_op_lookup_reads_no_table(monkeypatch):
    """A built op is found without calling table_tag(): it reads the table file, and
    file I/O inside the compiled forward broke Dynamo's graph at every call
    (LTX-2 with a table: 25 320 modules instead of 60, 2026-11-13)."""
    from difflet.backends.tpu.ops_impl import collective_matmul as cm

    def no_io():
        raise AssertionError("table_tag() called on the forward path")

    monkeypatch.setattr(cm_config, "table_tag", no_io)
    key = ("rs", 1, cm.sp_rs_kernel_mode(), (), "abcd1234")
    monkeypatch.setattr(cm, "_STATE", {"ops": {key: "built"}, "tuning": "abcd1234"})
    assert cm._op("rs", 1) == "built"
