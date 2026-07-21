from __future__ import annotations

import unittest

from token_prediction.dataset import PredictionPoint, PredictionPosition, PredictionTarget
from token_prediction.estimators import (
    EmpiricalQuantileEstimator,
    FitContext,
    LengthOnlyEstimator,
    RunContext,
    TrainingExample,
    TrainingView,
)


def _point(index: int) -> PredictionPoint:
    return PredictionPoint(
        point_id=f"point-{index}",
        source_event_id=f"event-{index}",
        task_id=f"task-{index}",
        trajectory_id=f"trajectory-{index}",
        run_id=f"run-{index}",
        prediction_context_id=f"context-{index}",
        condition_id="condition",
        logical_call_id=None,
        attempt_id=None,
        cutoff_event_seq=0,
        position=PredictionPosition.TASK_PRE,
        target=PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
        features={"current_request_tokens_local": index * 10},
        known_offset_tokens=0,
    )


def _view() -> TrainingView:
    return TrainingView(
        dataset_id="dataset",
        position=PredictionPosition.TASK_PRE,
        target=PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
        examples=tuple(
            TrainingExample(_point(index), float(index * 10), 1.0)
            for index in range(4)
        ),
    )


class IntervalContractTests(unittest.TestCase):
    def test_fit_context_validates_and_defaults_alpha(self) -> None:
        self.assertEqual(FitContext(1, 0).interval_alpha, 0.10)
        for invalid in (0.0, 1.0, -0.1, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    FitContext(1, 0, interval_alpha=invalid)

    def test_empirical_baseline_uses_context_alpha(self) -> None:
        view = _view()
        fitted = EmpiricalQuantileEstimator().fit(
            view,
            view,
            FitContext(1, 0, interval_alpha=0.5),
        )
        forecast = fitted.start(RunContext("task", "trajectory", "run")).predict(
            _point(10)
        )
        self.assertEqual((forecast.lower, forecast.point, forecast.upper), (0.0, 10.0, 20.0))

    def test_legacy_explicit_alpha_must_match_context(self) -> None:
        view = _view()
        context = FitContext(1, 0, interval_alpha=0.2)
        estimators = (
            EmpiricalQuantileEstimator(alpha=0.1),
            LengthOnlyEstimator(alpha=0.1),
        )
        for estimator in estimators:
            with self.subTest(estimator=estimator.estimator_id):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    estimator.fit(view, view, context)


if __name__ == "__main__":
    unittest.main()
