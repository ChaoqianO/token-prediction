from __future__ import annotations

import unittest

from token_prediction.dataset import PredictionPoint, PredictionPosition, PredictionTarget
from token_prediction.estimators import (
    DirectFeatureEstimator,
    FitContext,
    RunContext,
    TrainingExample,
    TrainingView,
    builtin_registry,
)


TARGET = PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS


def _point(index: int, estimate: float) -> PredictionPoint:
    return PredictionPoint(
        point_id=f"p-{index}",
        source_event_id=f"e-{index}",
        task_id=f"t-{index}",
        trajectory_id=f"tr-{index}",
        run_id=f"r-{index}",
        prediction_context_id=f"c-{index}",
        condition_id="condition",
        logical_call_id=None,
        attempt_id=None,
        cutoff_event_seq=0,
        position=PredictionPosition.TASK_LAUNCH,
        target=TARGET,
        features={"llm_self_estimated_total_tokens": estimate},
        known_offset_tokens=0,
    )


def _view(start: int, stop: int) -> TrainingView:
    return TrainingView(
        dataset_id="dataset",
        position=PredictionPosition.TASK_LAUNCH,
        target=TARGET,
        examples=tuple(
            TrainingExample(_point(index, index * 10.0), index * 10.0 + index, 1.0)
            for index in range(start, stop)
        ),
    )


class DirectFeatureEstimatorTests(unittest.TestCase):
    def test_point_prediction_is_not_fitted_or_rescaled(self) -> None:
        fitted = DirectFeatureEstimator(
            feature_name="llm_self_estimated_total_tokens"
        ).fit(_view(1, 11), _view(11, 15), FitContext(7, 0, 0.2))
        point = _point(20, 123.5)
        forecast = fitted.start(RunContext("t", "tr", "r")).predict(point)
        self.assertEqual(forecast.point, 123.5)
        self.assertLessEqual(forecast.lower, forecast.point)
        self.assertGreaterEqual(forecast.upper, forecast.point)

    def test_registry_exposes_direct_feature_baseline(self) -> None:
        estimator = builtin_registry().create(
            "direct_feature",
            {"feature_name": "llm_self_estimated_total_tokens"},
        )
        self.assertIsInstance(estimator, DirectFeatureEstimator)


if __name__ == "__main__":
    unittest.main()
