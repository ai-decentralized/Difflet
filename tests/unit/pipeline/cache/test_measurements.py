from __future__ import annotations

import pytest
import torch

from difflet.pipeline.cache import (
    CACHE_MEASUREMENT_SCHEMA,
    CACHE_MEASUREMENT_SCHEMA_REVISION,
    SPATIAL_MEASUREMENT_SCHEMA,
    SPATIAL_MEASUREMENT_SCHEMA_REVISION,
    CacheRunner,
    CacheStepContext,
    ExplicitMaskPolicy,
    InMemoryMeasurementSink,
    InMemorySpatialMeasurementSink,
    ResolvedCacheSession,
    TaylorSeerPredictor,
    load_cache_measurements,
    load_spatial_measurements,
    measure_latent_update,
    measure_spatial_error,
    resolve_cache_config,
    SpatialMeasurementLayout,
)


def _context(step_index: int, num_steps: int) -> CacheStepContext:
    return CacheStepContext(
        step_index=step_index,
        num_steps=num_steps,
        timestep=float(100 - step_index * 3),
        sigma=float(num_steps - step_index) / num_steps,
    )


def _run_quadratic_trajectory(runner: CacheRunner, mask: tuple[bool, ...]):
    decisions = []
    outputs = []
    for step_index in range(len(mask)):
        context = _context(step_index, len(mask))
        decision = runner.decide(context)
        decisions.append(decision)
        if decision.should_skip:
            output = runner.predict(context)
        else:
            output = torch.tensor([float(step_index * step_index)])
            runner.record_anchor(context, output)
        outputs.append(output.detach().clone())
    return decisions, outputs


def test_anchor_measurement_compares_estimate_before_recording_real_anchor():
    mask = (True, True, False, False, True)
    sink = InMemoryMeasurementSink()
    runner = CacheRunner(
        ExplicitMaskPolicy(mask),
        TaylorSeerPredictor(order=1),
        measurement_sink=sink,
    )

    _run_quadratic_trajectory(runner, mask)

    measurements = sink.anchor_measurements()
    assert [record.step_index for record in measurements] == [0, 1, 4]
    assert [record.estimate_status for record in measurements] == [
        "history_not_ready",
        "history_not_ready",
        "measured",
    ]
    measured = measurements[-1]
    assert measured.history_size == 2
    assert measured.estimated_steps_since_anchor == 2
    assert measured.anchor_step_gap == 3
    assert measured.output_shape == (1,)
    assert measured.output_dtype == "float32"
    assert measured.output_norm == pytest.approx(16.0)
    assert measured.relative_output_change == pytest.approx(15.0)
    assert measured.relative_output_curvature == pytest.approx(0.8)
    assert measured.estimate_relative_error == pytest.approx(0.75)
    assert measured.estimate_seconds is not None
    assert measured.estimate_seconds >= 0.0
    assert measured.measurement_seconds >= measured.estimate_seconds
    assert measured.numerically_valid is True
    assert measured.to_dict()["output_shape"] == [1]


def test_enabling_measurements_preserves_decisions_outputs_history_and_stats():
    mask = (True, True, False, True, False, False, True)
    plain = CacheRunner(
        ExplicitMaskPolicy(mask),
        TaylorSeerPredictor(order=1),
    )
    sink = InMemoryMeasurementSink()
    measured = CacheRunner(
        ExplicitMaskPolicy(mask),
        TaylorSeerPredictor(order=1),
        measurement_sink=sink,
    )

    plain_decisions, plain_outputs = _run_quadratic_trajectory(plain, mask)
    measured_decisions, measured_outputs = _run_quadratic_trajectory(measured, mask)

    assert measured_decisions == plain_decisions
    assert all(
        torch.equal(measured_output, plain_output)
        for measured_output, plain_output in zip(measured_outputs, plain_outputs)
    )
    assert measured.stats() == plain.stats()
    assert [anchor.step_index for anchor in measured.history] == [
        anchor.step_index for anchor in plain.history
    ]
    assert all(
        torch.equal(measured_anchor.output, plain_anchor.output)
        for measured_anchor, plain_anchor in zip(measured.history, plain.history)
    )
    assert len(sink.anchor_measurements()) == sum(mask)


def test_runner_produces_anchor_measurements_for_a_controlling_policy_without_a_sink():
    class MeasuringPolicy:
        def __init__(self):
            self.measurements = []

        def should_skip(self, context, history, observation):
            del context, history, observation
            return False

        def observe_anchor_measurement(self, measurement):
            self.measurements.append(measurement)

        def reset(self):
            self.measurements.clear()

    policy = MeasuringPolicy()
    runner = CacheRunner(policy, TaylorSeerPredictor(order=1))
    for step_index in range(3):
        context = _context(step_index, 3)
        assert runner.decide(context).should_compute
        runner.record_anchor(context, torch.tensor([float(step_index)]))

    assert [row.step_index for row in policy.measurements] == [0, 1, 2]
    assert policy.measurements[-1].estimate_status == "measured"


def test_prediction_configuration_error_is_recorded_without_losing_anchor():
    class InvalidPredictor:
        required_history = 1
        max_consecutive_predictions = None

        def predict(self, context, history):
            del context, history
            raise ValueError("invalid coordinate configuration")

    sink = InMemoryMeasurementSink()
    runner = CacheRunner(
        ExplicitMaskPolicy((True, True)),
        InvalidPredictor(),
        history_capacity=2,
        measurement_sink=sink,
    )

    for step_index in range(2):
        context = _context(step_index, 2)
        assert runner.decide(context).should_compute
        runner.record_anchor(context, torch.tensor([float(step_index)]))

    assert runner.stats()["full_steps"] == 2
    assert [anchor.step_index for anchor in runner.history] == [0, 1]
    assert sink.anchor_measurements()[-1].estimate_status == "prediction_error"
    assert sink.anchor_measurements()[-1].estimate_relative_error is None
    assert sink.anchor_measurements()[-1].estimate_seconds is not None
    assert sink.anchor_measurements()[-1].numerically_valid is False


def test_resolved_session_threads_and_resets_request_measurements():
    sink = InMemoryMeasurementSink()
    resolved = resolve_cache_config(
        num_steps=3,
        mask=(True, True, True),
        predictor=TaylorSeerPredictor(order=1),
    )
    session = ResolvedCacheSession(resolved, measurement_sink=sink)

    for step_index in range(3):
        assert session.decide_step(step_index).should_compute
        session.record_anchor(step_index, torch.tensor([float(step_index)]))

    assert len(sink.anchor_measurements()) == 3
    session.clear_request_state()
    assert sink.anchor_measurements() == ()
    assert sink.latent_updates() == ()
    assert session.statistics()["full_steps"] == 0


def test_request_measurement_report_roundtrips_with_strict_fields(tmp_path):
    sink = InMemoryMeasurementSink()
    runner = CacheRunner(
        ExplicitMaskPolicy((True, True, True)),
        TaylorSeerPredictor(order=1),
        measurement_sink=sink,
    )
    _run_quadratic_trajectory(runner, (True, True, True))
    report = sink.build_report(num_steps=3, configuration_source="unit_test")

    document = report.to_dict()
    assert document["schema"] == CACHE_MEASUREMENT_SCHEMA
    assert document["schema_revision"] == CACHE_MEASUREMENT_SCHEMA_REVISION
    assert load_cache_measurements(document) == report

    path = tmp_path / "cache-measurements.json"
    report.write_json(path)
    assert load_cache_measurements(path) == report

    document["unknown"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        load_cache_measurements(document)


def test_history_barrier_preserves_earlier_request_measurements():
    sink = InMemoryMeasurementSink()
    runner = CacheRunner(
        ExplicitMaskPolicy((True, True, True)),
        TaylorSeerPredictor(order=1),
        measurement_sink=sink,
    )

    for step_index in range(3):
        context = CacheStepContext(
            step_index=step_index,
            num_steps=3,
            is_barrier=step_index == 1,
        )
        runner.decide(context)
        runner.record_anchor(context, torch.tensor([float(step_index)]))

    assert [record.step_index for record in sink.anchor_measurements()] == [0, 1, 2]
    assert sink.anchor_measurements()[1].decision_reason == "barrier"
    assert sink.anchor_measurements()[1].history_size == 0


def test_latent_update_measurement_uses_float32_l2_relative_change():
    measurement = measure_latent_update(
        context=_context(1, 3),
        before=torch.tensor([3.0, 4.0], dtype=torch.bfloat16),
        after=torch.tensor([0.0, 8.0], dtype=torch.bfloat16),
        decision_reason="policy_skip",
        used_estimate=True,
    )

    assert measurement.latent_shape == (2,)
    assert measurement.latent_dtype == "bfloat16"
    assert measurement.before_norm == pytest.approx(5.0)
    assert measurement.after_norm == pytest.approx(8.0)
    assert measurement.update_norm == pytest.approx(5.0)
    assert measurement.relative_update == pytest.approx(1.0)
    assert measurement.numerically_valid is True


def test_session_records_actual_scheduler_action_once_per_step():
    sink = InMemoryMeasurementSink()
    resolved = resolve_cache_config(
        num_steps=4,
        mask=(True, True, False, True),
        predictor=TaylorSeerPredictor(order=1),
    )
    session = ResolvedCacheSession(resolved, measurement_sink=sink)

    for step_index in range(4):
        decision = session.decide_step(step_index)
        if decision.should_skip:
            session.estimate_output(step_index)
        else:
            session.record_anchor(step_index, torch.tensor([float(step_index)]))
        session.record_latent_update(
            step_index,
            torch.tensor([float(step_index + 1)]),
            torch.tensor([float(step_index + 2)]),
        )

    updates = sink.latent_updates()
    assert [record.step_index for record in updates] == [0, 1, 2, 3]
    assert [record.decision_reason for record in updates] == [
        "policy_compute",
        "policy_compute",
        "policy_skip",
        "policy_compute",
    ]
    assert [record.used_estimate for record in updates] == [False, False, True, False]
    with pytest.raises(RuntimeError, match="measured twice"):
        session.record_latent_update(3, torch.tensor([1.0]), torch.tensor([2.0]))


def test_disabled_session_latent_measurement_is_a_tensor_free_noop():
    resolved = resolve_cache_config(
        num_steps=1,
        mask=(True,),
        predictor=TaylorSeerPredictor(order=1),
    )
    session = ResolvedCacheSession(resolved)

    assert session.measurements_enabled is False
    session.record_latent_update(99, object(), object())


def test_spatial_measurement_records_exact_region_energies_with_one_layout():
    actual = torch.arange(1, 17, dtype=torch.float32).reshape(1, 4, 4)
    estimate = actual.clone()
    estimate[:, 0, :] += 1.0
    estimate[:, 3, :] += 2.0
    layout = SpatialMeasurementLayout(
        token_height=2,
        token_width=2,
        region_rows=2,
        region_columns=2,
        token_axis=1,
    )

    measurement = measure_spatial_error(
        context=_context(2, 5),
        estimate=estimate,
        actual=actual,
        layout=layout,
    )

    assert measurement.error_energy == pytest.approx((4.0, 0.0, 0.0, 16.0))
    assert measurement.reference_energy == pytest.approx((30.0, 174.0, 446.0, 846.0))
    assert measurement.measurement_seconds >= 0.0


def test_spatial_observer_is_read_only_and_report_roundtrips(tmp_path):
    layout = SpatialMeasurementLayout(2, 2, 2, 2, token_axis=1)
    sink = InMemorySpatialMeasurementSink(layout)
    runner = CacheRunner(
        ExplicitMaskPolicy((True, True, True)),
        TaylorSeerPredictor(order=1),
        measurement_sink=sink,
    )
    outputs = []
    for step_index in range(3):
        context = _context(step_index, 3)
        output = torch.full((1, 4, 2), float(step_index * step_index))
        outputs.append(output.clone())
        assert runner.decide(context).should_compute
        runner.record_anchor(context, output)

    retained_outputs = outputs[-len(runner.history) :]
    assert all(
        torch.equal(anchor.output, expected)
        for anchor, expected in zip(runner.history, retained_outputs)
    )
    assert [record.step_index for record in sink.anchor_measurements()] == [0, 1, 2]
    assert [record.step_index for record in sink.spatial_errors()] == [2]
    report = sink.build_spatial_report(
        num_steps=3,
        configuration_source="unit_test",
    )
    document = report.to_dict()
    assert document["schema"] == SPATIAL_MEASUREMENT_SCHEMA
    assert document["schema_revision"] == SPATIAL_MEASUREMENT_SCHEMA_REVISION
    assert load_spatial_measurements(document) == report

    path = tmp_path / "spatial-measurements.json"
    report.write_json(path)
    assert load_spatial_measurements(path) == report
    sink.clear()
    assert sink.anchor_measurements() == ()
    assert sink.spatial_errors() == ()
