from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

from token_prediction.dataset import PredictionPoint, PredictionPosition, PredictionTarget
from token_prediction.estimators.base import (
    FitContext,
    ObservedTransition,
    RunContext,
    SessionSeed,
    TokenForecast,
    TrainingExample,
    TrainingView,
)
from token_prediction.estimators.cross_position_deduct import (
    CROSS_POSITION_INPUT_CONTRACT_HASH,
    CrossPositionDeductEstimator,
)


TARGET = PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS
CONDITION = "condition:cross-position"
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def _point(
    index: int,
    *,
    position: PredictionPosition,
    offset: int | None = 0,
    missing_counter: int | None = 0,
    task_id: str = "task",
    trajectory_id: str = "trajectory",
    run_id: str = "run",
    condition_id: str = CONDITION,
    target: PredictionTarget = TARGET,
) -> PredictionPoint:
    features = (
        {}
        if missing_counter is None
        else {"missing_usage_attempts": missing_counter}
    )
    return PredictionPoint(
        point_id=f"{position.value}-point-{index}",
        source_event_id=f"event-{index}",
        task_id=task_id,
        trajectory_id=trajectory_id,
        run_id=run_id,
        prediction_context_id=f"context-{index}",
        condition_id=condition_id,
        logical_call_id=f"call-{index}",
        attempt_id=None,
        cutoff_event_seq=index * 10,
        position=position,
        target=target,
        features=features,
        known_offset_tokens=offset,
    )


def _seed(
    *,
    point: PredictionPoint | None = None,
    forecast: TokenForecast | None = None,
) -> SessionSeed:
    seed_point = point or _point(0, position=PredictionPosition.TASK_PRE, offset=10)
    seed_forecast = forecast or TokenForecast(
        point_id=seed_point.point_id,
        target=seed_point.target,
        lower=0.0,
        point=100.0,
        upper=100.0,
        raw_lower=-10.0,
        raw_point=100.0,
        raw_upper=80.0,
    )
    return SessionSeed(
        task_pre_point=seed_point,
        forecast=seed_forecast,
        initializer_id="initializer:inner-oof",
        initializer_hash=HASH_A,
        inner_split_id="outer-0/inner-ensemble",
        component_bundle_hashes=(HASH_B, HASH_C),
        seed_policy_id="uncalibrated-repaired-quantiles-v1",
        seed_policy_hash=HASH_D,
    )


def _view(*, condition_id: str = CONDITION) -> TrainingView:
    examples = tuple(
        TrainingExample(
            point=_point(
                index + 10,
                position=PredictionPosition.TASK_UPDATE,
                task_id=f"train-task-{index}",
                trajectory_id=f"train-trajectory-{index}",
                run_id=f"train-run-{index}",
                condition_id=condition_id,
            ),
            target_value=100.0 + index,
            sample_weight=1.0,
        )
        for index in range(2)
    )
    return TrainingView(
        dataset_id="dataset:v2",
        position=PredictionPosition.TASK_UPDATE,
        target=TARGET,
        examples=examples,
    )


def _fitted():
    view = _view()
    return CrossPositionDeductEstimator().fit(view, view, FitContext(7, 0))


def _context(*, seed: SessionSeed | None = None, **changes: object) -> RunContext:
    values: dict[str, object] = {
        "task_id": "task",
        "trajectory_id": "trajectory",
        "run_id": "run",
        "dataset_id": "dataset:v2",
        "condition_id": CONDITION,
        "target": TARGET,
        "runtime_mode": "offline",
        "input_contract_hash": CROSS_POSITION_INPUT_CONTRACT_HASH,
        "session_seed": seed or _seed(),
    }
    values.update(changes)
    return RunContext(**values)  # type: ignore[arg-type]


class SessionSeedTests(unittest.TestCase):
    def test_requires_task_pre_provider_target_and_matching_raw_repair(self) -> None:
        with self.assertRaisesRegex(ValueError, "Task-pre"):
            _seed(point=_point(0, position=PredictionPosition.TASK_UPDATE))

        point = _point(0, position=PredictionPosition.TASK_PRE)
        wrong_target = replace(point, target=PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS)
        with self.assertRaisesRegex(ValueError, "provider_accounted"):
            _seed(
                point=wrong_target,
                forecast=TokenForecast(
                    wrong_target.point_id,
                    wrong_target.target,
                    1.0,
                    2.0,
                    3.0,
                    raw_lower=1.0,
                    raw_point=2.0,
                    raw_upper=3.0,
                ),
            )

        calibrated = TokenForecast(
            point.point_id,
            point.target,
            0.0,
            100.0,
            150.0,
            raw_lower=-10.0,
            raw_point=100.0,
            raw_upper=80.0,
        )
        with self.assertRaisesRegex(ValueError, "repair"):
            _seed(point=point, forecast=calibrated)

        without_raw = TokenForecast(point.point_id, point.target, 1.0, 2.0, 3.0)
        with self.assertRaisesRegex(ValueError, "raw quantiles"):
            _seed(point=point, forecast=without_raw)

    def test_hashes_are_strict_and_seed_identity_is_stable(self) -> None:
        first = _seed()
        second = _seed()
        self.assertEqual(first.content_hash, second.content_hash)
        with self.assertRaisesRegex(ValueError, "initializer_hash"):
            replace(first, initializer_hash="not-a-hash")
        with self.assertRaisesRegex(ValueError, "component_bundle_hashes"):
            replace(first, component_bundle_hashes=())


class CrossPositionDeductTests(unittest.TestCase):
    def test_first_update_uses_task_pre_seed_and_exact_offset_algebra(self) -> None:
        session = _fitted().start(_context())
        first_update = _point(1, position=PredictionPosition.TASK_UPDATE, offset=20)
        session.observe(ObservedTransition("task_pre-point-0", first_update.point_id, 30))
        first = session.predict(first_update)

        self.assertEqual((first.lower, first.point, first.upper), (0.0, 60.0, 60.0))
        self.assertEqual(
            (first.raw_lower, first.raw_point, first.raw_upper),
            (-40.0, 60.0, 60.0),
        )

        second_update = _point(2, position=PredictionPosition.TASK_UPDATE, offset=5)
        session.observe(ObservedTransition(first.point_id, second_update.point_id, 25))
        second = session.predict(second_update)
        self.assertEqual((second.lower, second.point, second.upper), (0.0, 50.0, 50.0))

    def test_missing_counter_growth_carries_then_equal_counter_recovers(self) -> None:
        session = _fitted().start(_context())
        polluted = _point(
            1,
            position=PredictionPosition.TASK_UPDATE,
            missing_counter=1,
        )
        session.observe(ObservedTransition("task_pre-point-0", polluted.point_id, 40))
        carried = session.predict(polluted)
        self.assertEqual((carried.lower, carried.point, carried.upper), (0.0, 100.0, 100.0))
        self.assertEqual(session.fallback_count, 1)
        self.assertEqual(session.last_fallback_reason, "missing_usage_counter_increased")

        recovered = _point(
            2,
            position=PredictionPosition.TASK_UPDATE,
            missing_counter=1,
        )
        session.observe(ObservedTransition(polluted.point_id, recovered.point_id, 30))
        resumed = session.predict(recovered)
        self.assertEqual((resumed.lower, resumed.point, resumed.upper), (0.0, 70.0, 70.0))
        self.assertIsNone(session.last_fallback_reason)

    def test_missing_operands_are_not_imputed_and_counter_decrease_fails(self) -> None:
        session = _fitted().start(_context())
        missing = _point(1, position=PredictionPosition.TASK_UPDATE, offset=None)
        session.observe(ObservedTransition("task_pre-point-0", missing.point_id, None))
        carried = session.predict(missing)
        self.assertEqual(carried.point, 100.0)
        self.assertEqual(
            session.last_fallback_reason,
            "missing_observed_spend_and_current_offset",
        )

        seed = _seed(
            point=_point(
                0,
                position=PredictionPosition.TASK_PRE,
                missing_counter=2,
            )
        )
        decreasing_session = _fitted().start(_context(seed=seed))
        current = _point(
            1,
            position=PredictionPosition.TASK_UPDATE,
            missing_counter=1,
        )
        decreasing_session.observe(
            ObservedTransition(seed.task_pre_point.point_id, current.point_id, 1)
        )
        with self.assertRaisesRegex(ValueError, "decreased"):
            decreasing_session.predict(current)

    def test_absent_counters_on_both_points_do_not_block_visible_spend(self) -> None:
        seed = _seed(
            point=_point(
                0,
                position=PredictionPosition.TASK_PRE,
                offset=0,
                missing_counter=None,
            )
        )
        session = _fitted().start(_context(seed=seed))
        current = _point(
            1,
            position=PredictionPosition.TASK_UPDATE,
            offset=0,
            missing_counter=None,
        )
        session.observe(ObservedTransition(seed.task_pre_point.point_id, current.point_id, 25))
        self.assertEqual(session.predict(current).point, 75.0)

    def test_context_seed_and_prediction_scope_mismatches_fail_closed(self) -> None:
        fitted = _fitted()
        cases = (
            ("dataset_id", "other-dataset", "dataset_id"),
            ("condition_id", "condition:other", "condition_id"),
            ("target", PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS, "target"),
            ("input_contract_hash", HASH_A, "input_contract_hash"),
            ("session_seed", None, "session_seed"),
        )
        for name, value, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    fitted.start(_context(**{name: value}))

        wrong_seed = _seed(
            point=_point(
                0,
                position=PredictionPosition.TASK_PRE,
                task_id="other-task",
            )
        )
        with self.assertRaisesRegex(ValueError, "task_id"):
            fitted.start(_context(seed=wrong_seed))

        session = fitted.start(_context())
        wrong_point = _point(
            1,
            position=PredictionPosition.TASK_UPDATE,
            condition_id="condition:other",
        )
        session.observe(ObservedTransition("task_pre-point-0", wrong_point.point_id, 1))
        with self.assertRaisesRegex(ValueError, "condition_id"):
            session.predict(wrong_point)

    def test_sequence_protocol_fails_closed_and_calibration_cannot_feed_back(self) -> None:
        session = _fitted().start(_context())
        first_point = _point(1, position=PredictionPosition.TASK_UPDATE)
        with self.assertRaisesRegex(RuntimeError, "observe"):
            session.predict(first_point)
        with self.assertRaisesRegex(ValueError, "from_point_id"):
            session.observe(ObservedTransition("wrong", first_point.point_id, 1))

        session.observe(ObservedTransition("task_pre-point-0", first_point.point_id, 10))
        first = session.predict(first_point)
        calibrated_copy = replace(first, lower=0.0, upper=10_000.0)
        self.assertEqual(calibrated_copy.upper, 10_000.0)

        second_point = _point(2, position=PredictionPosition.TASK_UPDATE)
        session.observe(ObservedTransition(first.point_id, second_point.point_id, 10))
        second = session.predict(second_point)
        self.assertEqual(second.upper, 90.0)

    def test_fit_binds_dataset_position_target_and_condition(self) -> None:
        view = _view()
        fitted = CrossPositionDeductEstimator(
            expected_condition_id=CONDITION
        ).fit(view, view, FitContext(1, 0))
        self.assertEqual(fitted.dataset_id, "dataset:v2")
        self.assertEqual(fitted.input_contract_hash, CROSS_POSITION_INPUT_CONTRACT_HASH)

        wrong_position = replace(view, position=PredictionPosition.TASK_PRE)
        with self.assertRaisesRegex(ValueError, "Task-update"):
            CrossPositionDeductEstimator().fit(
                wrong_position, wrong_position, FitContext(1, 0)
            )
        wrong_target = replace(view, target=PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS)
        with self.assertRaisesRegex(ValueError, "provider_accounted"):
            CrossPositionDeductEstimator().fit(
                wrong_target, wrong_target, FitContext(1, 0)
            )
        with self.assertRaisesRegex(ValueError, "expected_condition_id"):
            CrossPositionDeductEstimator(
                expected_condition_id="condition:other"
            ).fit(view, view, FitContext(1, 0))

    def test_fit_binds_real_lifecycle_input_contract_when_supplied(self) -> None:
        sequence = SimpleNamespace(
            dataset_id="dataset:v2",
            condition_id=CONDITION,
            target=TARGET,
            input_contract_hash=HASH_A,
        )
        view = replace(_view(), lifecycle_sequences=(sequence,))
        fitted = CrossPositionDeductEstimator().fit(view, view, FitContext(1, 0))
        self.assertEqual(fitted.input_contract_hash, HASH_A)

        wrong_contract = replace(
            _view(),
            lifecycle_sequences=(
                SimpleNamespace(
                    dataset_id="dataset:v2",
                    condition_id=CONDITION,
                    target=TARGET,
                    input_contract_hash=HASH_B,
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "input contracts"):
            CrossPositionDeductEstimator().fit(view, wrong_contract, FitContext(1, 0))

        with self.assertRaisesRegex(ValueError, "expected_input_contract_hash"):
            CrossPositionDeductEstimator(
                expected_input_contract_hash=HASH_C
            ).fit(view, view, FitContext(1, 0))

    def test_base_extensions_remain_backward_compatible(self) -> None:
        legacy = RunContext("task", "trajectory", "run")
        self.assertEqual(legacy.runtime_mode, "offline")
        self.assertIsNone(legacy.session_seed)

        view = replace(_view(), lifecycle_sequences=(("context-step",),))
        self.assertEqual(view.lifecycle_sequences, (("context-step",),))
        self.assertEqual(len(view.sequences()), len(view.examples))
        with self.assertRaisesRegex(TypeError, "lifecycle_sequences"):
            replace(view, lifecycle_sequences=[])  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
