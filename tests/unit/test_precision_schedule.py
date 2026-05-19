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


def _rows_two_dtype():
    # block 0 to_q: E4M3 already good -> E4M3
    # block 1 to_v: E4M3 below tau_hi but E5M2 recovers -> E5M2
    # block 2 ff.net.0.proj: both below thresholds -> BF16
    return [
        {
            "block": 0,
            "linear": "to_q",
            "cosine": 0.99990,
            "metrics": {
                "e4m3": {"cosine": 0.99990, "mean_abs": 0.0, "max_abs": 0.0},
                "e5m2": {"cosine": 0.99970, "mean_abs": 0.0, "max_abs": 0.0},
            },
        },
        {
            "block": 1,
            "linear": "to_v",
            "cosine": 0.99930,
            "metrics": {
                "e4m3": {"cosine": 0.99930, "mean_abs": 0.0, "max_abs": 0.0},
                "e5m2": {"cosine": 0.99960, "mean_abs": 0.0, "max_abs": 0.0},
            },
        },
        {
            "block": 2,
            "linear": "ff.net.0.proj",
            "cosine": 0.99500,
            "metrics": {
                "e4m3": {"cosine": 0.99500, "mean_abs": 0.0, "max_abs": 0.0},
                "e5m2": {"cosine": 0.99700, "mean_abs": 0.0, "max_abs": 0.0},
            },
        },
    ]


def test_two_threshold_lattice_routes_by_both_cosines():
    from nova.pipeline.precision_schedule import (
        PRECISION_BF16,
        PRECISION_MXFP8_E4M3,
        PRECISION_MXFP8_E5M2,
        synthesize_two_threshold_schedule,
    )

    schedule = synthesize_two_threshold_schedule(
        _rows_two_dtype(),
        tau_hi=0.9998,
        tau_lo=0.9995,
        model_id="m",
        bundle="b",
    )

    assert schedule.assignments == {
        "0:to_q": PRECISION_MXFP8_E4M3,
        "1:to_v": PRECISION_MXFP8_E5M2,
        "2:ff.net.0.proj": PRECISION_BF16,
    }
    # non-BF16 coverage = E4M3 ∪ E5M2 = 2/3
    assert schedule.mx_coverage == 2 / 3
    assert schedule.coverage_by_precision[PRECISION_MXFP8_E5M2] == 1 / 3
    assert schedule.metadata["tau_hi"] == 0.9998
    assert schedule.metadata["tau_lo"] == 0.9995


def test_two_threshold_synthesis_is_deterministic():
    from nova.pipeline.precision_schedule import synthesize_two_threshold_schedule

    a = synthesize_two_threshold_schedule(
        _rows_two_dtype(), 0.9998, 0.9995, model_id="m", bundle="b"
    )
    b = synthesize_two_threshold_schedule(
        _rows_two_dtype(), 0.9998, 0.9995, model_id="m", bundle="b"
    )
    assert a == b


def test_two_threshold_frontier_has_three_extremes_plus_grid():
    from nova.pipeline.precision_schedule import two_threshold_frontier

    schedules = two_threshold_frontier(
        _rows_two_dtype(),
        [(0.9998, 0.9995), (0.9990, 0.9980)],
        model_id="m",
        bundle="b",
    )

    # all-bf16, all-e4m3, all-e5m2, then the 2 grid points
    assert [s.mx_coverage for s in schedules[:3]] == [0.0, 1.0, 1.0]
    assert len(schedules) == 5
    assert all(s.tau is not None for s in schedules[3:])


def test_old_two_level_schedule_still_deserializes(tmp_path):
    # A schedule written before E5M2 existed (schema 1, bf16/e4m3 only)
    # must still round-trip unchanged under the extended vocabulary.
    from nova.pipeline.precision_schedule import PrecisionSchedule

    legacy = {
        "schema_version": 1,
        "model_id": "legacy",
        "bundle": "b.safetensors",
        "tau": 0.9995,
        "mx_coverage": 0.5,
        "assignments": {"0:to_q": "mxfp8_e4m3", "0:to_v": "bf16"},
        "metadata": {},
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    loaded = PrecisionSchedule.read_json(path)
    assert loaded.assignments == {"0:to_q": "mxfp8_e4m3", "0:to_v": "bf16"}
    assert loaded.mx_coverage == 0.5


def test_single_threshold_degrades_on_schema1_table():
    # Schema-1 rows (no "metrics") -> E5M2 unknown -> two-threshold
    # never selects E5M2, degrading to single-threshold behavior.
    from nova.pipeline.precision_schedule import (
        PRECISION_BF16,
        PRECISION_MXFP8_E4M3,
        synthesize_two_threshold_schedule,
    )

    schedule = synthesize_two_threshold_schedule(
        _rows(),
        tau_hi=0.9995,
        tau_lo=0.0,
        model_id="m",
        bundle="b",
    )
    assert set(schedule.assignments.values()) <= {
        PRECISION_BF16,
        PRECISION_MXFP8_E4M3,
    }


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
