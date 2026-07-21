from __future__ import annotations

import math
import unittest
from dataclasses import replace

from token_prediction.contracts import EventType
from token_prediction.dataset import (
    DatasetRow,
    LabelStatus,
    PredictionPoint,
    PredictionPosition,
    PredictionTarget,
    build_supervised_dataset,
    make_task_split_plan,
)
from token_prediction.estimators.base import (
    FitContext,
    ObservedTransition,
    RunContext,
    TrainingExample,
    TrainingView,
)
from token_prediction.estimators.deduct import DeductOnlyEstimator
from token_prediction.estimators.registry import EstimatorRegistry
from token_prediction.experiment import (
    CandidateRole,
    CandidateSpec,
    ExperimentRunner,
    ExperimentSpec,
)
from token_prediction.features import NO_FEATURES
from token_prediction.trajectory import Trajectory

from tests.helpers import event


CONDITION = "condition:deduct-test"


def _point(
    index: int,
    *,
    offset: int | None,
    task_id: str = "task",
    trajectory_id: str = "trajectory",
    run_id: str = "run",
    condition_id: str = CONDITION,
) -> PredictionPoint:
    return PredictionPoint(
        point_id=f"point-{index}",
        source_event_id=f"event-{index}",
        task_id=task_id,
        trajectory_id=trajectory_id,
        run_id=run_id,
        prediction_context_id=f"context-{index}",
        condition_id=condition_id,
        logical_call_id=f"call-{index}",
        attempt_id=None,
        cutoff_event_seq=index * 10,
        position=PredictionPosition.TASK_UPDATE,
        target=PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
        features={},
        known_offset_tokens=offset,
    )


def _view(
    values: tuple[float, ...] = (100.0, 200.0, 300.0, 400.0),
    weights: tuple[float, ...] | None = None,
    *,
    condition_id: str = CONDITION,
) -> TrainingView:
    actual_weights = weights or tuple(1.0 for _ in values)
    return TrainingView(
        dataset_id="dataset",
        position=PredictionPosition.TASK_UPDATE,
        target=PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
        examples=tuple(
            TrainingExample(
                point=_point(
                    index + 100,
                    offset=10,
                    task_id=f"train-task-{index}",
                    trajectory_id=f"train-trajectory-{index}",
                    run_id=f"train-run-{index}",
                    condition_id=condition_id,
                ),
                target_value=value,
                sample_weight=actual_weights[index],
            )
            for index, value in enumerate(values)
        ),
    )


def _fitted(*, alpha: float = 0.5):
    view = _view()
    return DeductOnlyEstimator().fit(
        view,
        view,
        FitContext(seed=7, fold=0, interval_alpha=alpha),
    )


def _session():
    return _fitted().start(RunContext("task", "trajectory", "run"))


def _three_call_trajectory(task_index: int, run_index: int) -> Trajectory:
    prefix = f"deduct-task{task_index}-run{run_index}"
    offsets = (8 + task_index, 11 + task_index, 13 + task_index)
    outputs = (5 + run_index, 7 + run_index, 9 + run_index)
    events = [
        event(
            prefix,
            0,
            EventType.TASK_STARTED,
            payload={
                "task_id": f"task-{task_index}",
                "run_id": prefix,
                "condition_id": CONDITION,
                "model_id": "fixture-model",
                "agent_id": "fixture-agent",
            },
        )
    ]
    for call_index, (offset, output) in enumerate(zip(offsets, outputs)):
        call_id = f"{prefix}-call{call_index}"
        attempt_id = f"attempt-{call_index}"
        request_seq = 1 + call_index * 3
        provider_input = offset + 2
        events.extend(
            (
                event(
                    prefix,
                    request_seq,
                    EventType.REQUEST_BUILT,
                    call_id=call_id,
                    payload={"request_tokens_local": offset},
                ),
                event(
                    prefix,
                    request_seq + 1,
                    EventType.API_ATTEMPT_STARTED,
                    call_id=call_id,
                    attempt_id=attempt_id,
                ),
                event(
                    prefix,
                    request_seq + 2,
                    EventType.API_COMPLETED,
                    call_id=call_id,
                    attempt_id=attempt_id,
                    payload={
                        "usage": {
                            "input_tokens": provider_input,
                            "output_tokens": output,
                            "total_tokens": provider_input + output,
                        }
                    },
                ),
            )
        )
    events.append(event(prefix, 10, EventType.TASK_FINISHED))
    return Trajectory.from_events(events)


def _four_call_trajectory_with_early_missing_usage(
    task_index: int,
    run_index: int,
) -> Trajectory:
    prefix = f"deduct-missing-task{task_index}-run{run_index}"
    events = [
        event(
            prefix,
            0,
            EventType.TASK_STARTED,
            payload={
                "task_id": f"missing-task-{task_index}",
                "run_id": prefix,
                "condition_id": CONDITION,
            },
        )
    ]
    for call_index in range(4):
        call_id = f"{prefix}-call{call_index}"
        attempt_id = f"attempt-{call_index}"
        request_seq = 1 + call_index * 3
        offset = 8 + task_index + call_index * 3
        events.extend(
            (
                event(
                    prefix,
                    request_seq,
                    EventType.REQUEST_BUILT,
                    call_id=call_id,
                    payload={"request_tokens_local": offset},
                ),
                event(
                    prefix,
                    request_seq + 1,
                    EventType.API_ATTEMPT_STARTED,
                    call_id=call_id,
                    attempt_id=attempt_id,
                ),
            )
        )
        if call_index == 0:
            events.append(
                event(
                    prefix,
                    request_seq + 2,
                    EventType.API_FAILED,
                    call_id=call_id,
                    attempt_id=attempt_id,
                    payload={"usage": {"input_tokens": offset + 2}},
                )
            )
        else:
            output = 5 + run_index + call_index * 2
            provider_input = offset + 2
            events.append(
                event(
                    prefix,
                    request_seq + 2,
                    EventType.API_COMPLETED,
                    call_id=call_id,
                    attempt_id=attempt_id,
                    payload={
                        "usage": {
                            "input_tokens": provider_input,
                            "output_tokens": output,
                            "total_tokens": provider_input + output,
                        }
                    },
                )
            )
    events.append(event(prefix, 13, EventType.TASK_FINISHED))
    return Trajectory.from_events(events)


class DeductOnlyEstimatorTests(unittest.TestCase):
    def test_first_task_update_is_outer_train_cold_start_not_task_pre(self) -> None:
        # alpha=.5 makes the fitted q25/q50/q75 values exactly 100/200/300.
        # No Task-pre point or forecast is supplied to this within-cell session.
        forecast = _session().predict(_point(1, offset=10))
        self.assertEqual(
            (forecast.lower, forecast.point, forecast.upper),
            (100.0, 200.0, 300.0),
        )

    def test_exact_offset_algebra_and_multi_point_jump(self) -> None:
        session = _session()
        first = session.predict(_point(1, offset=10))
        session.observe(ObservedTransition(first.point_id, "point-4", 30))
        second = session.predict(_point(4, offset=20))
        self.assertEqual(
            (second.lower, second.point, second.upper),
            (60.0, 160.0, 260.0),
        )

        # A jump from cutoff 40 to 90 is valid when spend aggregates the gap.
        session.observe(ObservedTransition(second.point_id, "point-9", 200))
        third = session.predict(_point(9, offset=5))
        self.assertEqual(
            (third.lower, third.point, third.upper),
            (0.0, 0.0, 75.0),
        )
        self.assertLessEqual(third.lower, third.point)
        self.assertLessEqual(third.point, third.upper)

    def test_missing_offset_carries_forward_then_resumes(self) -> None:
        session = _session()
        first = session.predict(_point(1, offset=None))
        session.observe(ObservedTransition(first.point_id, "point-2", 30))
        second = session.predict(_point(2, offset=20))
        self.assertEqual(
            (second.lower, second.point, second.upper),
            (first.lower, first.point, first.upper),
        )
        self.assertEqual(session.fallback_count, 1)
        self.assertEqual(session.last_fallback_reason, "missing_previous_offset")

        session.observe(ObservedTransition(second.point_id, "point-3", 40))
        third = session.predict(_point(3, offset=10))
        self.assertEqual(
            (third.lower, third.point, third.upper),
            (70.0, 170.0, 270.0),
        )
        self.assertIsNone(session.last_fallback_reason)

    def test_missing_spend_is_not_imputed_as_zero(self) -> None:
        session = _session()
        first = session.predict(_point(1, offset=10))
        session.observe(ObservedTransition(first.point_id, "point-2", None))
        second = session.predict(_point(2, offset=20))
        self.assertEqual(
            (second.lower, second.point, second.upper),
            (first.lower, first.point, first.upper),
        )
        self.assertEqual(session.last_fallback_reason, "missing_observed_spend")

        session.observe(ObservedTransition(second.point_id, "point-3", 40))
        third = session.predict(_point(3, offset=10))
        self.assertEqual(
            (third.lower, third.point, third.upper),
            (70.0, 170.0, 270.0),
        )

    def test_sequence_and_cell_mismatches_fail_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "before the first"):
            _session().observe(ObservedTransition("point-1", "point-2", 1))

        session = _session()
        session.predict(_point(1, offset=10))
        with self.assertRaisesRegex(ValueError, "from_point_id"):
            session.observe(ObservedTransition("wrong", "point-2", 1))
        with self.assertRaisesRegex(RuntimeError, "observe must"):
            session.predict(_point(2, offset=10))

        session.observe(ObservedTransition("point-1", "point-2", 1))
        with self.assertRaisesRegex(ValueError, "to_point_id"):
            session.predict(_point(3, offset=10))

        for mismatched in (
            replace(_point(1, offset=10), condition_id="condition:other"),
            replace(_point(1, offset=10), task_id="other-task"),
            replace(_point(1, offset=10), run_id="other-run"),
            replace(
                _point(1, offset=10),
                target=PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
            ),
        ):
            with self.subTest(point=mismatched):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    _session().predict(mismatched)

    def test_alpha_is_derived_from_fit_context_and_legacy_mismatch_fails(self) -> None:
        view = _view()
        fitted = DeductOnlyEstimator().fit(
            view,
            view,
            FitContext(seed=1, fold=0, interval_alpha=0.5),
        )
        self.assertEqual(
            (fitted.initial_lower, fitted.initial_point, fitted.initial_upper),
            (100.0, 200.0, 300.0),
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            DeductOnlyEstimator(alpha=0.1).fit(
                view,
                view,
                FitContext(seed=1, fold=0, interval_alpha=0.2),
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            DeductOnlyEstimator(alpha=math.nan)

    def test_fit_and_predict_bind_position_target_and_condition(self) -> None:
        mixed_examples = list(_view().examples)
        mixed_examples[-1] = replace(
            mixed_examples[-1],
            point=replace(mixed_examples[-1].point, condition_id="condition:other"),
        )
        mixed = replace(_view(), examples=tuple(mixed_examples))
        with self.assertRaisesRegex(ValueError, "exactly one condition_id"):
            DeductOnlyEstimator().fit(mixed, _view(), FitContext(1, 0))

        wrong_position = replace(_view(), position=PredictionPosition.TASK_PRE)
        with self.assertRaisesRegex(ValueError, "Task-update"):
            DeductOnlyEstimator().fit(wrong_position, wrong_position, FitContext(1, 0))

    def test_provider_accounted_target_uses_the_same_within_cell_algebra(self) -> None:
        target = PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS
        base = _view()
        examples = tuple(
            replace(
                example,
                point=replace(example.point, target=target, known_offset_tokens=0),
            )
            for example in base.examples
        )
        view = replace(base, target=target, examples=examples)
        fitted = DeductOnlyEstimator().fit(
            view,
            view,
            FitContext(seed=7, fold=0, interval_alpha=0.5),
        )
        session = fitted.start(RunContext("task", "trajectory", "run"))
        first_point = replace(_point(1, offset=0), target=target)
        first = session.predict(first_point)
        session.observe(ObservedTransition(first.point_id, "point-2", 30))
        second = session.predict(replace(_point(2, offset=0), target=target))
        self.assertEqual(
            (second.lower, second.point, second.upper),
            (70.0, 170.0, 270.0),
        )

    def test_forecast_has_no_evaluation_label_dependency(self) -> None:
        point = _point(1, offset=10)
        low_label = DatasetRow(point, 1, LabelStatus.OBSERVED)
        high_label = DatasetRow(point, 999_999, LabelStatus.OBSERVED)
        low_forecast = _session().predict(low_label.point)
        high_forecast = _session().predict(high_label.point)
        self.assertEqual(low_forecast, high_forecast)

    def test_experiment_runner_uses_local_registry_and_exact_online_updates(self) -> None:
        dataset = build_supervised_dataset(
            _three_call_trajectory(task, run)
            for task in range(5)
            for run in range(2)
        )
        split = make_task_split_plan(
            dataset.task_ids,
            dataset_id=dataset.dataset_id,
            folds=5,
            seed=29,
        )
        registry = EstimatorRegistry()
        registry.register(
            "deduct_only", lambda params: DeductOnlyEstimator(**dict(params))
        )
        result = ExperimentRunner(registry).run(
            dataset,
            split,
            ExperimentSpec(
                "deduct-only-contract",
                PredictionPosition.TASK_UPDATE,
                PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
                (
                    CandidateSpec(
                        "deduct",
                        "deduct_only",
                        NO_FEATURES,
                        role=CandidateRole.BASELINE,
                    ),
                ),
                alpha=0.5,
                calibrator_id="none",
            ),
            seed=29,
        )[0]

        selected = dataset.select(
            PredictionPosition.TASK_UPDATE,
            PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
        )
        rows = {row.point.point_id: row for row in selected.rows}
        forecasts = {
            record.point_id: record.forecast for record in result.predictions
        }
        self.assertEqual(set(forecasts), set(rows))
        by_trajectory: dict[str, list[DatasetRow]] = {}
        for row in selected.rows:
            by_trajectory.setdefault(row.point.trajectory_id, []).append(row)
        for sequence in by_trajectory.values():
            first, second = sorted(
                sequence, key=lambda row: row.point.cutoff_event_seq
            )
            prior = forecasts[first.point.point_id]
            current = forecasts[second.point.point_id]
            spend = (
                int(second.point.features["cumulative_provider_input_tokens"])
                + int(second.point.features["cumulative_provider_output_tokens"])
                - int(first.point.features["cumulative_provider_input_tokens"])
                - int(first.point.features["cumulative_provider_output_tokens"])
            )
            delta = (
                int(first.point.known_offset_tokens or 0)
                - spend
                - int(second.point.known_offset_tokens or 0)
            )
            self.assertEqual(current.lower, max(0.0, prior.lower + delta))
            self.assertEqual(current.point, max(0.0, prior.point + delta))
            self.assertEqual(current.upper, max(0.0, prior.upper + delta))

    def test_runner_recovers_transition_spend_after_earlier_missing_usage(self) -> None:
        dataset = build_supervised_dataset(
            _four_call_trajectory_with_early_missing_usage(task, run)
            for task in range(5)
            for run in range(2)
        )
        split = make_task_split_plan(
            dataset.task_ids,
            dataset_id=dataset.dataset_id,
            folds=5,
            seed=37,
        )
        registry = EstimatorRegistry()
        registry.register("deduct_only", lambda params: DeductOnlyEstimator())
        result = ExperimentRunner(registry).run(
            dataset,
            split,
            ExperimentSpec(
                "deduct-missing-recovery",
                PredictionPosition.TASK_UPDATE,
                PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
                (
                    CandidateSpec(
                        "deduct",
                        "deduct_only",
                        NO_FEATURES,
                        role=CandidateRole.BASELINE,
                    ),
                ),
                alpha=0.5,
                calibrator_id="none",
            ),
            seed=37,
        )[0]

        selected = dataset.select(
            PredictionPosition.TASK_UPDATE,
            PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
        )
        forecasts = {
            record.point_id: record.forecast for record in result.predictions
        }
        by_trajectory: dict[str, list[DatasetRow]] = {}
        for row in selected.rows:
            by_trajectory.setdefault(row.point.trajectory_id, []).append(row)
        for sequence in by_trajectory.values():
            ordered = sorted(sequence, key=lambda row: row.point.cutoff_event_seq)
            self.assertEqual(len(ordered), 3)
            self.assertEqual(
                {row.point.features["missing_usage_attempts"] for row in ordered},
                {1},
            )
            for previous, current in zip(ordered, ordered[1:]):
                previous_forecast = forecasts[previous.point.point_id]
                current_forecast = forecasts[current.point.point_id]
                spend = sum(
                    int(current.point.features[name])
                    - int(previous.point.features[name])
                    for name in (
                        "cumulative_provider_input_tokens",
                        "cumulative_provider_output_tokens",
                    )
                )
                delta = (
                    int(previous.point.known_offset_tokens or 0)
                    - spend
                    - int(current.point.known_offset_tokens or 0)
                )
                self.assertEqual(
                    current_forecast.point,
                    max(0.0, previous_forecast.point + delta),
                )


if __name__ == "__main__":
    unittest.main()
