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
    assert h100[0]["device"] == "NVIDIA H100 PCIe"
    assert h100[0]["guidance_scale"] == 3.5  # part of the workload the page compares


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


@pytest.mark.parametrize(
    "adapter, backend",
    [("trainium", "aot"), ("neuron", "native"), ("cuda", "cuda"), ("tpu", "tpu")],
)
def test_adapter_maps_to_backend(adapter, backend):
    assert perf.record_from_bench(dict(BENCH, backend=adapter))["backend"] == backend


def test_backend_variant_slug_maps_to_its_page_model():
    d = dict(BENCH, backend="neuron", config_slug="wan_2_1_neuron", device_slug="trn2")
    rec = perf.record_from_bench(d)
    assert rec["model"] == "wan_2_1" and rec["backend"] == "native"


def test_every_matrix_slug_is_a_page_model():
    """A new MATRIX entry must get a page name (perf.MODELS) or an alias, not show as a raw slug."""
    from benchmark.models import MATRIX

    assert [s for s in MATRIX if perf.MODEL_ALIASES.get(s, s) not in perf.MODELS] == []


def test_sync_record_rewrites_in_place_and_keeps_manual_fields(tmp_path):
    bench, res = tmp_path / "flux.json", tmp_path / "results"
    failed = dict(BENCH, status="failed", notes=["FAILED: boom"], step_latency=None, e2e_warm=None)
    bench.write_text(json.dumps(failed))
    first = perf.sync_record(bench, source="benchmark/h100/flux_1_dev.md", results_dir=res)
    # PSNR and a note added by hand
    argv = ["--results-dir", str(res), "add", str(bench), "--force", "--psnr", "30.5"]
    assert perf.main(argv + ["--note", "host VAE"]) == 0
    # a follow-up script patches the bench JSON (same timestamp): the run now has a warm number
    bench.write_text(json.dumps(dict(BENCH, e2e_warm={"mean": 14.0})))
    second = perf.sync_record(bench, results_dir=res)
    assert second == first and len(list(res.glob("*.json"))) == 1
    rec = json.loads(second.read_text())
    assert rec["status"] == "ok" and rec["metrics"]["warm_e2e_s"] == 14.0
    assert rec["metrics"]["psnr_db"] == 30.5 and rec["metrics"]["ssim"] is None
    assert rec["notes"] == ["host VAE"]  # the stale failure reason is gone
    assert rec["source"] == "benchmark/h100/flux_1_dev.md"


def test_sync_quietly_reports_instead_of_raising(tmp_path, capsys):
    assert perf.sync_quietly(tmp_path / "missing.json", results_dir=tmp_path) is None
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(dict(BENCH, backend="mystery")))
    assert perf.sync_quietly(bad, results_dir=tmp_path) is None
    assert capsys.readouterr().out.count("no perf record written") == 2
    assert list(tmp_path.glob("*_h100_*.json")) == []


def test_flagged_run_is_published_with_its_flag_note():
    """neuron_cold_warm marks a completed run 'flagged' when only soft checks fail."""
    d = dict(
        BENCH, status="flagged", notes=["Methodology.", "FLAGGED (soft): warm spread 6%. More."]
    )
    rec = perf.record_from_bench(d)
    assert rec["status"] == "ok" and rec["notes"] == ["FLAGGED (soft): warm spread 6%"]


def test_cache_hit_compile_is_not_a_compile_time():
    trn = dict(BENCH, backend="trainium", device_slug="trn3")
    hit = perf.record_from_bench(
        dict(trn, compile_seconds=5.83, compile_breakdown={"wall_total_s": 5.83})
    )
    assert hit["metrics"]["compile_s"] is None and hit["notes"] == [perf.CACHE_HIT_NOTE]
    built = {"transformer": {"wall_s": 900.0}, "wall_total_s": 901.8}
    real = perf.record_from_bench(dict(trn, compile_seconds=901.84, compile_breakdown=built))
    assert real["metrics"]["compile_s"] == 901.8 and real["notes"] == []


def test_add_force_note_replaces_old_notes(tmp_path):
    bench, res = tmp_path / "flux.json", tmp_path / "results"
    bench.write_text(json.dumps(BENCH))
    base = ["--results-dir", str(res), "add", str(bench)]
    perf.main(base + ["--note", "VAE kept in fp16", "--psnr", "30.5"])
    perf.main(base + ["--force", "--note", "VAE kept in fp32"])
    rec = json.loads(next(res.glob("*.json")).read_text())
    assert rec["notes"] == ["VAE kept in fp32"] and rec["metrics"]["psnr_db"] == 30.5
    perf.main(base + ["--force"])  # no --note: the notes are kept
    assert json.loads(next(res.glob("*.json")).read_text())["notes"] == ["VAE kept in fp32"]


def test_unreadable_record_is_replaced(tmp_path):
    bench, res = tmp_path / "flux.json", tmp_path / "results"
    bench.write_text(json.dumps(BENCH))
    path = perf.sync_record(bench, results_dir=res)
    path.write_text("<<<<<<< HEAD\n{}\n=======\n[]\n>>>>>>> branch\n")  # merge conflict
    assert perf.main(["--results-dir", str(res), "add", str(bench), "--force"]) == 0
    assert perf.validate(json.loads(path.read_text())) == []
    path.write_text("[]")  # valid JSON, not a record
    assert perf.sync_record(bench, results_dir=res) == path
    assert perf.validate(json.loads(path.read_text())) == []


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_metric_is_rejected(bad):
    rec = _rec()
    rec["metrics"]["warm_e2e_s"] = bad
    assert any("finite" in e for e in perf.validate(rec))


@pytest.mark.parametrize(
    "g, ok",
    [
        (None, True),
        (4, True),
        (3.5, True),
        ("4.0", False),
        (True, False),
        (-1, False),
        (float("nan"), False),
    ],
)
def test_guidance_scale_must_be_a_number(g, ok):
    assert (perf.validate(_rec(guidance_scale=g)) == []) is ok
