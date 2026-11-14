"""scripts/mx_attention/probe.py: the one-core attention probe's CLI, its shape table, its fp32
reference per head and its timing summary. Nothing here touches a device: the device workers are
replaced by a fake that writes what a worker writes."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "mx_attention_probe", REPO / "scripts" / "mx_attention" / "probe.py")
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)

SEVEN = {"wan_81f", "hv1_L", "hv15_121f", "ltx2_video", "flux_1024", "qwen_1024", "ltx2_audio"}


def test_the_shape_table_has_the_seven_rows():
    shapes = probe.load_shapes()
    assert set(shapes) == SEVEN
    for name, row in shapes.items():
        assert row["path"] in ("bounded", "unbounded"), name
        if row["path"] == "bounded":
            assert 0 < row["valid_k"] < row["s"] and row["valid_k"] % 32, name
        if name != "ltx2_audio":
            assert row["routed"] is True and row["d"] == 128, name
    assert shapes["ltx2_audio"]["d"] == 64 and shapes["ltx2_audio"]["routed"] is False


@pytest.mark.parametrize("bad", [["--shape", "sd3_1024", "--impl", "cte"],
                                 ["--shape", "wan_81f", "--impl", "sdpa"],
                                 ["--shape", "wan_81f", "--impl", "cte", "--pv-mx"]])
def test_the_cli_rejects_what_it_cannot_run(bad, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        probe.main([*bad, "--out", str(tmp_path / "o.json")])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice" in err or "need --impl fork" in err


def test_fork_stops_with_a_message_while_the_kernel_module_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "FORK_MODULE", "difflet.backends.neuron.ops_impl._no_such_kernel")
    with pytest.raises(SystemExit) as exc:
        probe.main(["--shape", "flux_1024", "--impl", "fork", "--pv-mx",
                    "--out", str(tmp_path / "o.json")])
    assert "_no_such_kernel" in str(exc.value) and "Task 3" in str(exc.value)


TINY = {"tiny": {"bh": 3, "s": 700, "valid_k": 613, "d": 64, "path": "bounded", "routed": True},
        "tiny_u": {"bh": 2, "s": 640, "d": 64, "path": "unbounded", "routed": True}}


@pytest.mark.parametrize("name", sorted(TINY))
def test_the_reference_is_fp32_attention_per_head(name):
    row = {"name": name, **TINY[name]}
    q, k, v, bounds = probe.make_inputs(row)
    assert q.dtype == torch.bfloat16 and q.shape == (row["bh"], row["s"], row["d"])
    mask = None
    if bounds:
        lo, hi = bounds
        assert lo.dtype == torch.int32 and lo.shape == (row["bh"], row["s"], 1)
        assert (lo == 0).all() and (hi == row["valid_k"]).all()
        mask = torch.arange(row["s"]) < row["valid_k"]
    expected = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=mask)
    for h in range(row["bh"]):
        torch.testing.assert_close(probe.reference(row, q, k, v, bounds, h), expected[h],
                                   atol=1e-5, rtol=1e-5)


def test_compare_reports_each_head_and_the_whole():
    row = {"name": "tiny", **TINY["tiny"]}
    q, k, v, bounds = probe.make_inputs(row)
    exact = torch.stack([probe.reference(row, q, k, v, bounds, h) for h in range(row["bh"])])
    device = exact.clone()
    device[1] *= 1.01
    overall, heads = probe.compare(device, lambda h: exact[h], row["bh"])
    assert heads[0] == {"rel_l2": 0.0, "cosine": pytest.approx(1.0, abs=1e-12)}
    assert heads[1]["rel_l2"] == pytest.approx(0.01, rel=1e-6)
    assert 0 < overall["rel_l2"] < heads[1]["rel_l2"]


def test_the_timing_is_the_median_and_spread_of_the_processes():
    assert probe.summarize([10.0, 12.0, 11.0]) == {"median_ms": 11.0, "spread": 2.0 / 11.0}


def test_main_runs_the_workers_and_scores_the_first_ones_output(tmp_path, monkeypatch):
    """--runs worker processes, the first saving the device output; here a fake worker writes
    the exact reference with one head off, and a ms per call per run."""
    monkeypatch.setattr(probe, "load_shapes", lambda: {k: dict(r) for k, r in TINY.items()})
    monkeypatch.setattr(probe, "require_idle_core", lambda: {"device": 1, "idle": True})
    row = {"name": "tiny", **TINY["tiny"]}
    q, k, v, bounds = probe.make_inputs(row)
    calls = []

    def fake_worker(argv, log):
        calls.append(argv)
        run = Path(argv[argv.index("--worker") + 1])
        if "--save-output" in argv:
            out = torch.stack([probe.reference(row, q, k, v, bounds, h) for h in range(3)])
            out[2] *= -1
            torch.save(out.bfloat16(), probe.output_path(Path(argv[argv.index("--out") + 1])))
        run.write_text(json.dumps({"ms": 5.0 + len(calls), "first_call_s": 1.0}))

    monkeypatch.setattr(probe, "run_worker", fake_worker)
    out = tmp_path / "tiny.json"
    assert probe.main(["--shape", "tiny", "--impl", "cte", "--out", str(out)]) == 0
    assert len(calls) == 3 and sum("--save-output" in c for c in calls) == 1
    result = json.loads(out.read_text())
    assert result["runs_ms"] == [6.0, 7.0, 8.0] and result["median_ms"] == 7.0
    assert result["spread"] == pytest.approx(2.0 / 7.0)
    assert [h["cosine"] for h in result["per_head"]] == pytest.approx([1, 1, -1], abs=1e-2)
    assert result["shape"]["name"] == "tiny" and result["reference"] == "fp32"
    assert len(result["fuser"]) == 3
