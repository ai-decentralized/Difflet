"""scripts/profile_dit/analyze.py: per-engine, per-component and per-phase time of one block NEFF.

The blocks here are synthetic: tiny ``Instruction`` / ``CcOp`` / ``ActiveTime`` / ``Warning``
parquets and a ``summary.json`` written into ``tmp_path``. Layers are Qwen-Image module paths that
``components.classify`` puts in a known component. ``hlo_name`` and ``hlo_attrs`` are null, as
most rows of a real parquet are (ruling R13). The last test runs the captured blocks when they are
on this host."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "profile_dit_analyze", REPO / "scripts" / "profile_dit" / "analyze.py")
analyze = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analyze)

CTE = ("/opt/aws_neuronx_venv_torch_neuronx_2_14_0/lib/python3.12/site-packages"
       "/nkilib/core/attention/attention_cte.py")
ATTN = "Attention[attn][0]_custom-call.5"  # attention; module attn
MLP_UP = "FeedForward[img_mlp][0]/GELU[0][0]/ColumnParallelLinear[proj][0]_dot.1"
MLP_ACT = "FeedForward[img_mlp][0]/GELU[0][0]_tanh.1"
MOD = "Sequential[img_mod][0]/ColumnParallelLinear[1][0]_dot.91"
OTHER = "Mystery[x][0]_add.1"  # no rule names it

#: Step 4 (2026-11-13): per engine and physical core, the union of a block's instruction intervals
#: without EVENT_SEMAPHORE* rows against ActiveTime.parquet differed by at most 0.113% (FLUX single,
#: GpSimd) over HV1 dual and the Task 1 smokes (Qwen-Image, FLUX dual and single, HV1.5).
ACTIVE_TIME_TOLERANCE = 0.005


def row(engine, start, end, layer=MLP_UP, opcode="MATMUL", pcore=0, src="", psum=0.0, attrs=None):
    return {"start_ts": start, "end_ts": end, "duration_ns": end - start, "engine": engine,
            "pcore_idx": pcore, "opcode": opcode, "layer": layer, "hlo_name": None,
            "nki_source_location": src or None, "hlo_attrs": attrs, "psum_read_bytes": psum}


def block(tmp_path, rows, cc=(), total_time=1e-3, exec8=None, source=True, **summary):
    """A block directory as Task 1 writes it."""
    trace = tmp_path / "pq" / "trace"
    trace.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(trace / "Instruction.parquet")
    pd.DataFrame({"start_ts": [s for s, _ in cc], "end_ts": [e for _, e in cc]},
                 dtype="int64").to_parquet(trace / "CcOp.parquet")
    pd.DataFrame({"start_ts": [0], "end_ts": [7], "engine": ["tensor"], "pcore_idx": [0],
                  "duration_ns": [7]}).to_parquet(trace / "ActiveTime.parquet")
    pd.DataFrame({"message": ["Missing additional dynamic DMA metadata"],
                  "category": ["Missing Data"]}).to_parquet(trace / "Warning.parquet")
    fields = {"total_time": total_time, "mfu_estimated_percent": 0.4, "mbu_estimated_percent": 0.1,
              "tensor_engine_active_time": 5e-4, "dma_active_time": 3e-4,
              "static_dma_active_time": 1e-4, "hardware_dynamic_dma_active_time": 1e-4,
              "software_dynamic_dma_active_time": 1e-4, "hbm_read_bytes": 10,
              "hbm_write_bytes": 11, "sbuf_read_bytes": 12, "sbuf_write_bytes": 13,
              "psum_read_bytes": 14, "psum_write_bytes": 15, "psum_read_sbuf_write_bytes": 16,
              "spill_save_bytes": 17, "spill_reload_bytes": 18, **summary}
    (tmp_path / "summary.json").write_text(
        json.dumps({"device_profile_profile_rank_0_exec_4": fields}))
    if exec8 is not None:
        (tmp_path / "summary_exec8.json").write_text(
            json.dumps({"device_profile_profile_rank_0_exec_8": {**fields, "total_time": exec8}}))
    (tmp_path / "block.neff").write_bytes(b"neff")
    if source:
        (tmp_path / "source.json").write_text(json.dumps({"neff_name": "n.neff", "sha256": "abc"}))
    return tmp_path


def run(tmp_path, rows, **kw):
    return analyze.analyze(block(tmp_path, rows, **kw), "qwen_image")


def test_a_dma_instruction_is_hbm_traffic_not_the_issuing_engines_compute(tmp_path):
    out = run(tmp_path, [row("Scalar", 0, 1000, opcode="DMA_DIRECT2D"),
                         row("Scalar", 1000, 1500, MLP_ACT, opcode="ACTIVATE")])
    assert out["engines_ms"]["hbm_dma"] == pytest.approx(1e-3)
    assert out["engines_ms"]["scalar"] == pytest.approx(5e-4)
    assert out["components"]["mlp_up"] == {"hbm_dma": pytest.approx(1e-3)}


def test_overlapping_rows_count_once_and_the_two_physical_cores_are_averaged(tmp_path):
    out = run(tmp_path, [row("Tensor", 0, 1000), row("Tensor", 500, 1500),
                         row("Tensor", 0, 500, pcore=1)])
    assert out["engines_ms"]["tensor"] == pytest.approx((1500 + 500) / 2 / 1e6)


def test_components_sum_to_no_more_than_their_engine_and_the_block(tmp_path):
    out = run(tmp_path / "a", [row("Tensor", 0, 1000), row("Tensor", 950, 2000, MOD)],
              total_time=3e-6)
    total = sum(c.get("tensor", 0) for c in out["components"].values())
    assert total <= 1.10 * out["engines_ms"]["tensor"] and total <= out["block_ms"]
    with pytest.raises(ValueError, match="tensor"):  # two components on the same busy time
        run(tmp_path / "b", [row("Tensor", 0, 1000), row("Tensor", 0, 1000, MOD)])


def test_a_semaphore_wait_adds_nothing(tmp_path):
    out = run(tmp_path, [row("Tensor", 0, 1000),
                         row("Tensor", 1000, 9000, opcode="EVENT_SEMAPHORE"),
                         row("Sync", 0, 9000, opcode="EVENT_SEMAPHORE_RANGE_CLEAR")])
    assert out["engines_ms"] == {"tensor": pytest.approx(1e-3), "collective": 0.0}
    assert [(s["start_us"], s["end_us"]) for s in out["timeline"]] == [(0.0, 1.0)]


def test_psum_read_out_splits_into_pure_copies_and_fused_compute(tmp_path):
    out = run(tmp_path, [row("Vector", 0, 1000, MLP_ACT, opcode="RANGE_SELECT", psum=64.0),
                         row("Vector", 2000, 2500, MLP_ACT, opcode="COPY", psum=64.0),
                         row("Scalar", 2200, 2400, MLP_ACT, opcode="COPY", psum=32.0),
                         row("Scalar", 3000, 3300, MLP_ACT, opcode="CAST", psum=32.0),
                         row("Scalar", 4000, 9000, MLP_ACT, opcode="COPY")])  # reads no PSUM
    assert out["psum_readout_fused_ms"] == pytest.approx(1e-3)
    assert out["psum_readout_copy_ms"] == pytest.approx(8e-4)  # across engines: 2200-2400 once
    # ruling R16: per engine, so each sits inside its VectorE / ScalarE bar
    assert out["psum_readout_copy_ms_by_engine"] == {"vector": pytest.approx(5e-4),
                                                     "scalar": pytest.approx(5e-4)}
    assert out["psum_readout_fused_ms_by_engine"] == {"vector": pytest.approx(1e-3)}
    assert out["memory_bytes"] == {
        "hbm_read": 10, "hbm_write": 11, "sbuf_read": 12, "sbuf_write": 13, "psum_read": 14,
        "psum_write": 15, "psum_to_sbuf": 16, "spill_save": 17, "spill_reload": 18}


def test_throttle_averages_every_cores_fields(tmp_path):
    out = run(tmp_path, [row("Tensor", 0, 1000)],
              throttle_avg_util_limit_nc6_percent=0.8, throttle_avg_util_limit_nc7_percent=0.6,
              throttle_activity_1_active_time_nc6_percent=0.3,
              throttle_activity_1_active_time_nc7_percent=0.5,
              throttle_activity_1_avg_util_limit_nc6_percent=0.5,  # not the overall limit
              throttle_activity_0_active_time_nc6_percent=0.9)  # not the limited state
    assert out["throttle"] == {"avg_util_limit": pytest.approx(0.7),
                               "limited_time_share": pytest.approx(0.4)}
    assert (out["mfu"], out["mbu"]) == (0.4, 0.1)


def test_other_over_10_percent_of_an_engines_attributed_time_is_refused_naming_it(tmp_path):
    # 200 of the 1200 ns attributed: 17%; of all 11200 ns 1.8%: Unknown rows must not dilute it
    rows = [row("Vector", 0, 10_000, "Unknown"), row("Vector", 10_000, 10_200, OTHER),
            row("Vector", 10_200, 11_200, MLP_ACT)]
    with pytest.raises(ValueError, match="vector"):
        run(tmp_path, rows)


def test_rows_with_no_module_path_are_unattributed_not_other(tmp_path):
    out = run(tmp_path, [row("Vector", 0, 3000, "Unknown"), row("Vector", 3000, 3950, MLP_ACT),
                         row("Vector", 3950, 4000, OTHER)])
    assert out["components"]["unattributed"] == {"vector": pytest.approx(3e-3)}
    assert out["unattributed_share"]["vector"] == pytest.approx(0.75)
    assert out["other_share"]["vector"] == pytest.approx(0.05)


def test_unattributed_time_is_split_by_its_source_file_and_function(tmp_path):
    """Ruling R25: Unknown rows stay unattributed; unattributed_by_source says which source file
    (and function, where the file can be read and parsed) their busy time came from, on the
    same per-physical-core union basis, with "(none)" for rows that name no source."""
    mod = tmp_path / "mod.py"
    mod.write_text("import os\n\n\nclass A:\n    def f(self):\n        return os.sep\n")
    lin = '[{"op_type":"","source_file":"/elsewhere/lin.py","source_line":"7"}]'
    out = run(tmp_path / "blk", [
        row("Tensor", 0, 1000, "Unknown", src=f"{mod}:6"),
        row("Tensor", 500, 1500, "Unknown", src=f"{mod}:6"),  # overlaps the first: once
        row("Tensor", 2000, 3000, "Unknown", pcore=1, src=f"{mod}:6"),
        row("Vector", 0, 300, "Unknown", src="/gone/elsewhere/thing.py:10"),  # no such file
        row("Vector", 400, 500, "Unknown", attrs=lin),  # its source only in hlo_attrs
        row("Scalar", 0, 200, "Unknown"),
        row("Tensor", 5000, 6000)])  # attributed (mlp_up): not in it
    assert out["unattributed_by_source"] == {
        "mod.py:A.f": {"tensor": pytest.approx((1500 + 1000) / 2 / 1e6)},
        "thing.py": {"vector": pytest.approx(300 / 2 / 1e6)},
        "lin.py": {"vector": pytest.approx(100 / 2 / 1e6)},
        "(none)": {"scalar": pytest.approx(200 / 2 / 1e6)}}
    assert out["components"]["unattributed"]["tensor"] == pytest.approx(2500 / 2 / 1e6)


def test_unmapped_attention_lines_over_10_percent_of_attention_are_refused(tmp_path):
    rows = [row("Tensor", 0, 1000, ATTN, src=f"{CTE}:4128"),
            row("Vector", 1000, 1200, ATTN, src=f"{CTE}:9999")]
    with pytest.raises(ValueError, match="9999"):
        run(tmp_path, rows)


def test_attention_loads_and_write_back_are_phase_other_not_unmapped(tmp_path):
    # ruling R11: 3772 (output write-back) is phase "other" by design, however large
    out = run(tmp_path, [row("Tensor", 0, 1000, ATTN, src=f"{CTE}:4128"),
                         row("Sync", 1000, 3000, ATTN, opcode="DMA_DIRECT2D", src=f"{CTE}:3772"),
                         row("Vector", 1000, 1300, ATTN, src=f"{CTE}:3381")])
    assert out["attention_detail"] == {"attn": {"qk_matmul": {"tensor": pytest.approx(1e-3)},
                                                "other": {"hbm_dma": pytest.approx(2e-3)},
                                                "mask_softmax": {"vector": pytest.approx(3e-4)}}}


def test_collectives_take_their_time_from_the_cc_ops(tmp_path):
    out = run(tmp_path, [row("Tensor", 0, 1000)], cc=[(0, 1000), (500, 2000), (5000, 5500)])
    assert out["engines_ms"]["collective"] == pytest.approx(2.5e-3)
    assert out["components"]["collective"] == {"collective": pytest.approx(2.5e-3)}


def test_replay_spread_and_neff_hash_come_from_what_the_block_directory_has(tmp_path):
    out = run(tmp_path / "t1", [row("Tensor", 0, 1000)], total_time=1e-3, exec8=1.2e-3)
    assert (out["block_ms"], out["replay_spread_ms"], out["block_neff_sha256"]) == (
        pytest.approx(1.0), pytest.approx(0.2), "abc")
    assert out["dge_notifs"] is True  # captured with --enable-dge-notifs unless source.json says not
    out = run(tmp_path / "hv1", [row("Tensor", 0, 1000)], source=False)  # last campaign's layout
    assert out["replay_spread_ms"] is None and out["dge_notifs"] is None
    assert out["block_neff_sha256"] == hashlib.sha256(b"neff").hexdigest()
    assert out["warnings"] == ["Missing additional dynamic DMA metadata"]
    (tmp_path / "hv1" / "block.neff").unlink()
    assert analyze.analyze(tmp_path / "hv1", "qwen_image")["block_neff_sha256"] is None
    # ruling R22: captured again without --enable-dge-notifs after a notification queue overflow
    (tmp_path / "t1" / "source.json").write_text(json.dumps({"sha256": "abc", "dge_notifs": False}))
    assert analyze.analyze(tmp_path / "t1", "qwen_image")["dge_notifs"] is False


def test_the_timeline_is_physical_core_0_in_execution_order_with_gaps_under_5_us_merged(tmp_path):
    out = run(tmp_path, [row("Tensor", 0, 1000), row("Tensor", 3000, 4000),
                         row("Tensor", 3500, 3800), row("Tensor", 10_000, 11_000),
                         row("Vector", 500, 800, MLP_ACT), row("Tensor", 20_000, 30_000, pcore=1)],
              cc=[(1500, 2500)])
    # busy_us (ruling R16): the busy runs inside a segment, overlaps once (0-1 and 3-4 us)
    assert out["timeline"] == [
        {"component": "mlp_up", "engine_class": "tensor", "start_us": 0.0, "end_us": 4.0,
         "busy_us": 2.0},
        {"component": "mlp_act", "engine_class": "vector", "start_us": 0.5, "end_us": 0.8,
         "busy_us": pytest.approx(0.3)},
        {"component": "collective", "engine_class": "collective", "start_us": 1.5, "end_us": 2.5,
         "busy_us": 1.0},
        {"component": "mlp_up", "engine_class": "tensor", "start_us": 10.0, "end_us": 11.0,
         "busy_us": 1.0}]


def test_a_long_blocks_timeline_is_bounded_and_keeps_each_engine_classs_span(tmp_path):
    # ruling R15: 20,000 short rows, 26 us apart per engine, over a 200 ms block -> gaps under
    # 200 ms / 4000 = 50 us merge
    kinds = [("Tensor", MLP_UP, "MATMUL"), ("Vector", MLP_ACT, "TENSOR_TENSOR"),
             ("Scalar", MLP_ACT, "DMA_DIRECT2D")]
    rows = [row(eng, i * 10_000, i * 10_000 + 4_000, layer, opcode=op)
            for i in range(20_000) for eng, layer, op in [kinds[i % 3]]]
    timeline = run(tmp_path, rows, total_time=0.2)["timeline"]
    assert len(timeline) <= 4000
    for ec, first, last, rows_ in [("tensor", 0, 199_984, 6667), ("vector", 10, 199_994, 6667),
                                   ("hbm_dma", 20, 199_974, 6666)]:
        spans = [(s["start_us"], s["end_us"]) for s in timeline if s["engine_class"] == ec]
        assert (min(spans)[0], max(e for _, e in spans)) == (first, last), ec
        # the merged segments still carry every row's 4 us of busy time (ruling R16)
        assert sum(s["busy_us"] for s in timeline if s["engine_class"] == ec) == pytest.approx(
            4.0 * rows_), ec


def test_the_cli_writes_breakdown_json_with_the_fields_the_page_reads(tmp_path):
    analyze.main([str(block(tmp_path, [row("Tensor", 0, 1000)])), "--model", "qwen_image"])
    out = json.loads((tmp_path / "breakdown.json").read_text())
    assert set(out) == {
        "block_ms", "replay_spread_ms", "block_neff_sha256", "engines_ms", "components",
        "engines_check", "attention_detail", "memory_bytes", "psum_readout_copy_ms",
        "psum_readout_fused_ms", "psum_readout_copy_ms_by_engine",
        "psum_readout_fused_ms_by_engine", "mfu", "mbu", "throttle", "other_share",
        "unattributed_share", "warnings", "timeline", "dge_notifs", "unattributed_by_source"}
    summary_ms = out["engines_check"]["summary_ms"]
    assert summary_ms["dma_active_time"] == pytest.approx(0.3)
    assert {"static_dma_active_time", "hardware_dynamic_dma_active_time",
            "software_dynamic_dma_active_time"} <= set(summary_ms)


CAPTURED = {"hv1_dual": ("/home/ubuntu/fp8-bench/hv1_tp4/profsp_V15/blk1", "hunyuan_video"),
            "qwen": ("/mnt/nvme/difflet-prof/t1-smoke/qwen_image/bf16/blk1", "qwen_image"),
            "flux_dual": ("/mnt/nvme/difflet-prof/t1-smoke-final/flux/bf16/blk1", "flux"),
            "flux_single": ("/mnt/nvme/difflet-prof/t1-smoke-final/flux/bf16/blk2", "flux")}


@pytest.mark.parametrize("name", CAPTURED)
def test_each_engines_union_reproduces_the_profilers_active_time(name):
    path, model = CAPTURED[name]
    if not (Path(path) / "pq" / "trace" / "ActiveTime.parquet").exists():
        pytest.skip(f"{path} is not on this host")
    check = analyze.analyze(Path(path), model)["engines_check"]
    for engine, ms in check["union_ms"].items():
        expected = check["active_time_ms"][engine]
        assert ms == pytest.approx(expected, rel=ACTIVE_TIME_TOLERANCE), engine
