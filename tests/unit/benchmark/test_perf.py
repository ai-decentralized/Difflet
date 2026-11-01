"""benchmark.perf: bench JSON -> record, validation, and the page data build."""

from __future__ import annotations

import copy
import json

import pytest

from benchmark import perf

BENCH = {
    "backend": "cuda",
    "device": "CUDA / NVIDIA H100 PCIe",
    "device_slug": "h100",
    "config_slug": "flux_1_dev",
    "model_id": "black-forest-labs/FLUX.1-dev",
    "shape": {"height": 1024, "width": 1024, "num_frames": None},
    "steps": 28,
    "seed": 42,
    "guidance_scale": 3.5,
    "dtype": "bf16",
    "parallel": {"tp_degree": 4, "cp_degree": 1},
    "status": "ok",
    "notes": [],
    "e2e_cold_seconds": 15.880184779999581,
    "compile_seconds": 0.0,
    "timestamp": "2026-06-27 06:22 UTC",
    "toolchain": {"torch": "2.9.1+cu128", "diffusers": "0.38.0"},
    "step_latency": {"mean": 0.3108234684444445, "median": 0.3107889489997433},
    "e2e_warm": {"mean": 15.76907642299966},
}


def _rec(**over):
    rec = perf.record_from_bench(copy.deepcopy(BENCH))
    rec.update(over)
    return rec


def test_record_from_cuda_bench():
    rec = perf.record_from_bench(BENCH)
    assert perf.validate(rec) == []
    assert rec["platform"] == "h100" and rec["backend"] == "cuda" and rec["model"] == "flux_1_dev"
    assert rec["device"] == "NVIDIA H100 PCIe"
    assert rec["parallel"] == "1 GPU"  # single-GPU dense, whatever the config says
    assert rec["metrics"]["dit_step_ms"] == pytest.approx(310.82, abs=0.01)
    assert rec["metrics"]["warm_e2e_s"] == pytest.approx(15.77)
    assert rec["metrics"]["cold_e2e_s"] == pytest.approx(15.88)
    assert rec["metrics"]["compile_s"] is None  # eager: nothing to compile


def test_record_from_trainium_bench_is_aot_with_compile():
    d = dict(
        BENCH,
        backend="trainium",
        device="trn3pd98.3xlarge / 4 NeuronCores",
        device_slug="trn3",
        compile_seconds=6378.79,
        parallel={"tp_degree": 4, "cp_degree": 2},
    )
    rec = perf.record_from_bench(d)
    assert rec["backend"] == "aot" and rec["parallel"] == "TP4 CP2"
    assert rec["device"] == "trn3pd98.3xlarge / 4 NeuronCores"
    assert rec["metrics"]["compile_s"] == 6378.8


def test_failed_oom_run_and_unknown_adapter():
    d = dict(
        BENCH,
        status="failed",
        step_latency=None,
        e2e_warm=None,
        e2e_cold_seconds=None,
        notes=[
            "H100 runs single-GPU dense. Long methodology text.",
            "FAILED: CUDA out of memory. Tried to allocate 51.05 GiB.",
        ],
    )
    rec = perf.record_from_bench(d)
    assert rec["status"] == "oom" and perf.validate(rec) == []
    assert rec["notes"] == [
        "FAILED: CUDA out of memory"
    ]  # only the failure reason reaches the page
    assert perf.record_from_bench(dict(BENCH, notes=["methodology"]))["notes"] == []
    with pytest.raises(ValueError, match="pass --backend"):
        perf.record_from_bench(dict(BENCH, backend="mystery"))
    assert (
        perf.record_from_bench(dict(BENCH, backend="mystery"), backend="native")["backend"]
        == "native"
    )


@pytest.mark.parametrize(
    "over, msg",
    [
        ({"backend": "gpu"}, "backend"),
        ({"status": "pending"}, "status"),
        ({"date": "27/06/2026"}, "date"),
        ({"steps": 0}, "steps"),
        ({"shape": {"height": 1024}}, "shape"),
        ({"metrics": {"dit_step_ms": "fast"}}, "dit_step_ms"),
        ({"metrics": {"latency": 1.0}}, "unknown metric"),
        ({"metrics": {"dit_step_ms": None}}, "every metric is null"),
    ],
)
def test_validate_rejects(over, msg):
    errs = perf.validate(_rec(**over))
    assert errs and any(msg in e for e in errs), errs


def test_validate_missing_field():
    rec = _rec()
    del rec["toolchain"]
    assert perf.validate(rec) == ["missing field 'toolchain'"]


def test_write_record_names_and_refuses_overwrite(tmp_path):
    path = perf.write_record(_rec(), tmp_path)
    assert path.name == "2026-06-27_0622_h100_cuda_flux_1_dev.json"
    with pytest.raises(FileExistsError):
        perf.write_record(_rec(), tmp_path)
    perf.write_record(_rec(), tmp_path, force=True)
    with pytest.raises(ValueError, match="invalid perf record"):
        perf.write_record(_rec(status="pending"), tmp_path)


def test_build_keeps_newest_per_model_platform_backend(tmp_path):
    old = _rec(date="2026-06-27 06:22 UTC")
    new = _rec(date="2026-10-26")
    new["metrics"] = dict(new["metrics"], dit_step_ms=300.0, psnr_db=21.5)
    trn = _rec(
        platform="trn2", backend="native", device="trn2.3xlarge", parallel="TP4", date="2026-10-26"
    )
    for r in (old, new, trn):
        perf.write_record(r, tmp_path)
    data = perf.build(tmp_path, repo_root=tmp_path)
    assert [c["label"] for c in data["configs"]] == ["H100", "trn2 Native"]  # fixed platform order
    assert data["models"] == [{"slug": "flux_1_dev", "name": "FLUX.1-dev", "shape": "1024×1024"}]
    h100 = [r for r in data["results"] if r["config"] == "h100/cuda"]
    assert (
        len(h100) == 1
        and h100[0]["metrics"]["dit_step_ms"] == 300.0
        and h100[0]["date"] == "2026-10-26"
    )
    assert h100[0]["metrics"]["psnr_db"] == 21.5 and h100[0]["metrics"]["ssim"] is None
    assert h100[0]["source"] == h100[0]["record"] == "2026-10-26_h100_cuda_flux_1_dev.json"
    assert data["generated"] == "2026-10-26"


def test_build_reports_every_invalid_file(tmp_path):
    (tmp_path / "a.json").write_text("{not json")
    (tmp_path / "b.json").write_text(json.dumps(_rec(status="pending")))
    with pytest.raises(ValueError) as e:
        perf.build(tmp_path)
    assert "a.json" in str(e.value) and "b.json" in str(e.value)


def test_cli_add_check_build(tmp_path, capsys):
    bench = tmp_path / "flux.json"
    bench.write_text(json.dumps(BENCH))
    res = tmp_path / "results"
    assert (
        perf.main(
            [
                "--results-dir",
                str(res),
                "add",
                str(bench),
                "--psnr",
                "30.5",
                "--ssim",
                "0.95",
                "--note",
                "host VAE",
            ]
        )
        == 0
    )
    rec = json.loads(next(res.glob("*.json")).read_text())
    assert rec["metrics"]["psnr_db"] == 30.5 and rec["notes"] == ["host VAE"]
    assert perf.main(["--results-dir", str(res), "check"]) == 0
    out = tmp_path / "perf.json"
    assert perf.main(["--results-dir", str(res), "build", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["results"][0]["metrics"]["ssim"] == 0.95


def test_committed_records_are_valid():
    """Every record in benchmark/results/ must load, so the Pages build cannot break on main."""
    data = perf.build()
    assert data["results"], "no committed perf records"
