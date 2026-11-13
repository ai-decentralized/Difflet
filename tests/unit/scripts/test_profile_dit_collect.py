"""scripts/profile_dit/collect.py: merges block breakdowns, system summaries and the evidence doc's
step times and quality rows into results.json and the page's datasets.

The breakdowns and lane directories here are synthetic, written into ``tmp_path``; the evidence
doc is the real one. The last test runs the real assets when the lanes are on this host."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "profile_dit_collect", REPO / "scripts" / "profile_dit" / "collect.py")
collect = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(collect)

EVIDENCE = REPO / "docs" / "verification" / "2026-11-07-mxfp8-video-models.md"
RUN = Path("/mnt/nvme/difflet-prof/run")


def breakdown(block_ms=10.0, engines=None, attention=None, detail=None, mfu=0.4):
    """A breakdown.json as analyze.py writes it, small."""
    engines = engines or {"tensor": 6.0, "vector": 4.0, "scalar": 1.0, "hbm_dma": 2.0,
                          "collective": 1.0}
    return {
        "block_ms": block_ms, "replay_spread_ms": 0.01, "block_neff_sha256": "ab",
        "engines_ms": engines,
        "components": {"attention": attention or {"tensor": 2.0, "vector": 3.0},
                       "rope": {"vector": 0.5}, "unattributed": {"tensor": 1.0}},
        "engines_check": {"summary_ms": {"scalar_engine_active_time": 99.0}},
        "attention_detail": detail or {"attn": {"qk_matmul": {"tensor": 1.0},
                                                "mask_softmax": {"vector": 3.0}}},
        "memory_bytes": {"hbm_read": 10, "psum_read": 5, "psum_to_sbuf": 4},
        "psum_readout_copy_ms": 0.2, "psum_readout_fused_ms": 0.3,
        "psum_readout_copy_ms_by_engine": {"scalar": 0.2},
        "psum_readout_fused_ms_by_engine": {"vector": 0.3},
        "mfu": mfu, "mbu": 0.1, "throttle": {"avg_util_limit": 0.8, "limited_time_share": 0.3},
        "other_share": {}, "unattributed_share": {}, "warnings": ["w"], "dge_notifs": True,
        "unattributed_by_source": {"(none)": {"tensor": 1.0}},
        "component_overlap": {"tensor": 1.162, "vector": 1.04, "scalar": 1.0, "hbm_dma": 1.0},
        "timeline": [{"component": "attention", "engine_class": "tensor", "start_us": 0.0,
                      "end_us": 5.0, "busy_us": 4.0}],
    }


def lane(tmp_path, key, arm, blocks, kinds=None):
    """``blocks``: [(breakdown or None, execs_per_step, in_model_median_ms)] -> (assets, run)."""
    assets, run = tmp_path / "assets", tmp_path / "run"
    names = [f"neff_{i}_vnc_0.neff" for i in range(1, len(blocks) + 1)]
    for i, (b, _, _) in enumerate(blocks, 1):
        if b is not None:
            (assets / key / arm).mkdir(parents=True, exist_ok=True)
            (assets / key / arm / f"blk{i}.json").write_text(json.dumps(b))
        (run / key / arm / f"blk{i}").mkdir(parents=True)
        (run / key / arm / f"blk{i}" / "source.json").write_text(json.dumps(
            {"neff_name": names[i - 1], "kind": (kinds or ["block"] * len(blocks))[i - 1]}))
    (run / key / arm / "system_summary.json").write_text(json.dumps({
        "blocks": names, "step": {"wall_ms": 200.0, "steps": 1},
        "neffs": {n: {"execs_per_step": e, "in_model_median_ms": t}
                  for n, (_, e, t) in zip(names, blocks)},
        "non_block_ms": 1.5, "host_gap_ms": 0.5}))
    return assets, run


def one(assets, run, key="m", n=1, steps=None):
    configs = {key: ("Model", "8x8", "tp4", ("bf16",), n)}
    steps = {(key, "bf16"): (221, "11b", "A", 50)} if steps is None else steps
    return collect.collect(assets, run, configs, steps)["configs"][0]


def test_bottleneck_is_the_largest_engine_class_after_the_dma_split(tmp_path):
    # ScalarE's summary time (99 ms) still holds its DMA; engines_ms has it as hbm_dma
    b = breakdown(engines={"scalar": 3.0, "hbm_dma": 4.0, "tensor": 3.5}, mfu=0.21)
    c = one(*lane(tmp_path, "m", "bf16", [(b, 60, 4.0)]))
    assert c["bottleneck"] == {"engine_class": "hbm_dma", "share": 0.4, "mfu": 0.21}
    assert c["blocks"][0]["bottleneck"]["engine_class"] == "hbm_dma"
    # RoPE (0.5 ms, VectorE only) is read on its own engine, not as 0 on the bottleneck's
    assert (c["rope_engine"], c["rope_share"]) == ("vector", 0.05)


def test_attention_share_is_on_the_bottleneck_engine_class(tmp_path):
    b = breakdown(engines={"tensor": 3.0, "vector": 6.0}, attention={"tensor": 5.0, "vector": 2.0},
                  detail={"attn1": {"mask_softmax": {"vector": 1.5}},
                          "attn2": {"mask_softmax": {"vector": 0.5}}})
    c = one(*lane(tmp_path, "m", "bf16", [(b, 60, 4.0)]))
    assert c["bottleneck"]["engine_class"] == "vector"
    assert c["attention_share"] == pytest.approx(0.2)
    # Plan B routes attn1, not the cross-attention attn2: 2.0 x 1.5 / 2.0 of the vector time
    assert c["addressable_attention_share"] == pytest.approx(0.15)
    assert c["cross_attention_share"] == pytest.approx(0.05)


def test_no_evidence_step_time_is_not_measured(tmp_path):
    c = one(*lane(tmp_path, "m", "bf16", [(breakdown(), 60, 4.0)]), steps={})
    assert c["step_ms"] is None
    assert c["step_source"] == "not measured"
    assert c["outside_blocks_ms"] is None
    assert c["blocks_ms"] == pytest.approx(240.0)


def test_outside_blocks_is_the_step_less_executions_times_in_model_medians(tmp_path):
    assets, run = lane(tmp_path, "m", "bf16", [(breakdown(), 19, 4.0), (breakdown(), 38, 3.0)],
                       kinds=["dual", "single"])
    c = one(assets, run, n=2)
    assert c["step_source"] == {"section": "11b", "host": "A", "steps": 50}
    assert c["blocks_ms"] == pytest.approx(19 * 4.0 + 38 * 3.0)
    assert c["outside_blocks_ms"] == pytest.approx(221 - 190.0)
    assert (c["non_block_ms"], c["host_gap_ms"], c["profile_step_ms"]) == (1.5, 0.5, 200.0)
    assert [(b["kind"], b["execs_per_step"]) for b in c["blocks"]] == [("dual", 19), ("single", 38)]


def test_a_config_weighs_its_blocks_by_executions(tmp_path):
    tensor = breakdown(block_ms=4.0, engines={"tensor": 3.0, "vector": 1.0}, mfu=0.5)
    vector = breakdown(block_ms=2.0, engines={"tensor": 0.5, "vector": 1.5}, mfu=0.2)
    c = one(*lane(tmp_path, "m", "bf16", [(tensor, 1, 4.0), (vector, 10, 2.0)]), n=2)
    # tensor 3 + 10 x 0.5 = 8 < vector 1 + 10 x 1.5 = 16, of 4 + 10 x 2 = 24 ms
    assert c["bottleneck"]["engine_class"] == "vector"
    assert c["bottleneck"]["share"] == pytest.approx(16 / 24)
    assert c["bottleneck"]["mfu"] == pytest.approx((4 * 0.5 + 20 * 0.2) / 24)


def test_a_missing_breakdown_is_pending_and_never_estimated(tmp_path):
    assets, run = lane(tmp_path, "m", "bf16", [(breakdown(), 20, 450.0), (None, 40, 230.0)],
                       kinds=["dual", "single"])
    c = one(assets, run, n=2)
    pending = c["blocks"][1]
    assert pending["status"] == "pending" and "block_ms" not in pending
    assert (pending["kind"], pending["execs_per_step"]) == ("single", 40)
    assert c["status"] == "pending"
    assert c["bottleneck"] is None and c["attention_share"] is None
    assert c["outside_blocks_ms"] == pytest.approx(221 - (20 * 450.0 + 40 * 230.0))
    data = collect.datasets(collect.collect(assets, run, {"m": ("Model", "8x8", "tp4", ("bf16",), 2)},
                                            {}))
    assert "timeline_m_bf16_blk1" in data and "timeline_m_bf16_blk2" not in data
    row = [r for r in data["blocks"] if r["blk"] == "blk2"][0]
    assert row["status"] == "pending" and row["block_ms"] is None
    assert not [r for r in data["engines"] if r["blk"] == "blk2"]


def test_a_lane_with_nothing_on_disk_is_pending(tmp_path):
    configs = {"m": ("Model", "8x8", "tp4", ("bf16", "mxfp8"), 2)}
    c = collect.collect(tmp_path / "a", tmp_path / "r", configs, {})["configs"]
    assert [x["arm"] for x in c] == ["bf16", "mxfp8"]
    assert all(x["status"] == "pending" and x["blocks_ms"] is None for x in c)
    assert [b["blk"] for b in c[0]["blocks"]] == [1, 2]


def test_datasets_are_rows_and_one_time_line_per_block(tmp_path):
    assets, run = lane(tmp_path, "m", "bf16", [(breakdown(), 60, 4.0)])
    data = collect.datasets(collect.collect(assets, run, {"m": ("Model", "8x8", "tp4", ("bf16",), 1)},
                                            {("m", "bf16"): (221, "11b", "A", 50)}))
    assert set(data) == {"overview", "blocks", "engines", "components", "attention_phases",
                         "memory", "unattributed", "quality", "timeline_m_bf16_blk1"}
    assert all(isinstance(rows, list) and all(isinstance(r, dict) for r in rows)
               for rows in data.values())
    o = data["overview"][0]
    assert (o["step_ms"], o["step_section"], o["bottleneck"], o["blocks"]) == (221, "11b", "tensor",
                                                                              "block x60")
    assert {r["engine_class"]: r["ms"] for r in data["engines"]}["hbm_dma"] == 2.0
    assert {(r["module"], r["routed"], r["phase"]) for r in data["attention_phases"]} == {
        ("attn", True, "qk_matmul"), ("attn", True, "mask_softmax")}
    psum = {(r["metric"], r["engine_class"]): r["value"] for r in data["memory"]}
    assert psum[("psum_readout_copy", "scalar")] == 0.2 and psum[("hbm_read", "")] == 10
    assert data["timeline_m_bf16_blk1"] == breakdown()["timeline"]


def test_component_overlap_is_noted_per_engine_over_1_05(tmp_path):
    """Ruling R28: an engine whose components' unions sum over 1.05 x its union (interleaved
    branches, HV1's single block TensorE 1.16) gets a note; every engine row carries its ratio."""
    assets, run = lane(tmp_path, "m", "bf16", [(breakdown(), 60, 4.0)])
    data = collect.datasets(collect.collect(assets, run, {"m": ("Model", "8x8", "tp4", ("bf16",), 1)},
                                            {("m", "bf16"): (221, "11b", "A", 50)}))
    assert data["blocks"][0]["component_overlap"] == "tensor 1.162x"
    overlap = {r["engine_class"]: r["component_overlap"] for r in data["engines"]}
    assert (overlap["tensor"], overlap["vector"], overlap["collective"]) == (1.162, 1.04, None)


def test_store_and_doc_tables(tmp_path):
    assets, run = lane(tmp_path, "m", "bf16", [(breakdown(), 60, 4.0), (None, 2, 1.0)])
    results = collect.collect(assets, run, {"m": ("Model", "8x8", "tp4", ("bf16",), 2)}, {})
    data = collect.datasets(results)
    docs = collect.store(results, data, "<meta>", "now")
    assert set(docs) == {"dash/meta", "files/index.html", *(f"datasets/{i}" for i in data)}
    assert docs["datasets/timeline_m_bf16_blk1"]["source"]["url"] == "<URL:timeline_m_bf16_blk1>"
    assert docs["datasets/timeline_m_bf16_blk1"]["title"] == "Time line, Model bf16 blk1 (block)"
    assert all(d["fixed"] for k, d in docs.items() if k.startswith("datasets/"))
    md = collect.markdown(results)
    assert "| Model | bf16 | blk2 (block) | pending |" in md  # never a number for a pending block
    assert "not measured" in md


def _section(heading: str) -> str:
    """The evidence doc's text from ``### <heading>`` to the next heading."""
    text = EVIDENCE.read_text()
    return re.search(rf"^### {re.escape(heading)}\..*?(?=^##)", text, re.S | re.M).group(0)


def test_quality_rows_carry_shape_section_and_profiled_shape():
    rows = collect.collect(Path("/nonexistent"), Path("/nonexistent"))["quality"]
    assert {r["section"] for r in rows} == {"7f", "10c", "11c", "12c"}
    assert {r["key"] for r in rows} == set(collect.CONFIGS)
    for r in rows:
        shape = collect.CONFIGS[r["key"]][1]
        assert r["shape"] and r["profiled_shape"] is (r["shape"] == shape)
        text = _section(r["section"])  # each transcribed value is in its section
        for f, fmt in (("psnr", "{:.2f}"), ("psnr_worst", "{:.2f}"), ("ssim", "{:.3f}"),
                       ("lpips", "{:.3f}"), ("bf16_psnr", "{:.2f}"), ("delta_psnr", "{:+.2f}")):
            if r[f] is not None:
                assert fmt.format(r[f]) in text, (r["key"], r["section"], f)
    assert [r["profiled_shape"] for r in rows if r["section"] in ("7f", "12c")] == [True] * 4


def test_step_times_are_the_evidence_docs():
    for (key, arm), (ms, section, host, steps) in collect.STEP_TIMES.items():
        assert key in collect.CONFIGS and arm in collect.CONFIGS[key][3]
        text = _section(section)
        assert str(ms) in text or f"{ms:,}" in text, (key, arm)
        assert host == ("B" if section.startswith("12") else "A")


@pytest.mark.skipif(not RUN.is_dir(), reason="the profile lanes are not on this host")
def test_the_real_profile():
    results = collect.collect(collect.ASSETS, RUN)
    by = {(c["key"], c["arm"]): c for c in results["configs"]}
    assert len(by) == 13
    for (key, arm), c in by.items():
        assert c["step_ms"] is not None
        if c["status"] == "ok":
            assert 0 < c["bottleneck"]["share"] <= 1
            assert 0 <= c["addressable_attention_share"] <= c["attention_share"] <= 1
    assert by["ltx_2", "bf16"]["blocks"][0]["execs_per_step"] == 96  # 48 layers x 2 CFG forwards
    for rows in collect.datasets(results).values():
        assert len(json.dumps(rows)) < 10e6
