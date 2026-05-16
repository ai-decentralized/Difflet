import json

from nova.pipeline.parallel_config import NovaParallelConfig


def _rows():
    return [
        {"block": 0, "linear": "to_q", "cosine": 0.9998},
        {"block": 0, "linear": "to_out.0", "cosine": 0.9994},
        {"block": 1, "linear": "ff.net.2", "cosine": 0.9989},
    ]


def test_precision_schedule_round_trips(tmp_path):
    from nova.pipeline.precision_schedule import PrecisionSchedule

    schedule = PrecisionSchedule(
        model_id="model",
        bundle="bundle.safetensors",
        tau=0.9995,
        assignments={
            "0:to_q": "mxfp8_e4m3",
            "0:to_out.0": "bf16",
        },
        metadata={"source": "unit"},
    )
    path = tmp_path / "schedule.json"

    schedule.write_json(path)
    loaded = PrecisionSchedule.read_json(path)

    assert loaded == schedule
    assert loaded.mx_coverage == 0.5
    assert loaded.precision_for(0, "to_q") == "mxfp8_e4m3"


def test_synthesize_schedule_thresholds_by_cell_cosine():
    from nova.pipeline.precision_schedule import PRECISION_BF16, PRECISION_MXFP8_E4M3
    from nova.pipeline.precision_schedule import synthesize_schedule

    schedule = synthesize_schedule(
        _rows(),
        0.9995,
        model_id="model",
        bundle="bundle",
    )

    assert schedule.assignments == {
        "0:to_q": PRECISION_MXFP8_E4M3,
        "0:to_out.0": PRECISION_BF16,
        "1:ff.net.2": PRECISION_BF16,
    }
    assert schedule.mx_coverage == 1 / 3


def test_schedule_frontier_includes_extremes_and_taus():
    from nova.pipeline.precision_schedule import schedule_frontier

    schedules = schedule_frontier(_rows(), [0.9990, 0.9995], model_id="model", bundle="bundle")

    assert [schedule.mx_coverage for schedule in schedules[:2]] == [0.0, 1.0]
    assert [schedule.tau for schedule in schedules[2:]] == [0.9990, 0.9995]


def test_compile_cache_manifest_records_optional_precision_schedule(tmp_path):
    from nova.pipeline.compile_cache import CacheSpec, read_manifest, write_manifest

    spec = CacheSpec(
        model_id="org/test-model",
        model_path="/tmp/model",
        model_name="unit_dummy",
        parallel=NovaParallelConfig(tp_degree=1),
        dtype="bf16",
        precision_schedule={
            "path": "/tmp/schedule.json",
            "model_id": "org/test-model",
            "bundle": "bundle.safetensors",
            "tau": 0.9995,
        },
    )

    write_manifest(tmp_path, spec)
    manifest = read_manifest(tmp_path)

    assert manifest["precision_schedule"]["path"] == "/tmp/schedule.json"
    assert manifest["precision_schedule"]["tau"] == 0.9995


def test_old_compile_cache_manifest_still_reads_but_is_not_valid(tmp_path):
    from nova.pipeline.compile_cache import CacheSpec, has_valid_manifest, read_manifest

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "cache_key": "old",
                "cache_inputs": {},
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    spec = CacheSpec(
        model_id="org/test-model",
        model_path="/tmp/model",
        model_name="unit_dummy",
        parallel=NovaParallelConfig(tp_degree=1),
        dtype="bf16",
    )

    assert read_manifest(tmp_path)["schema_version"] == 3
    assert has_valid_manifest(tmp_path, spec) is False
