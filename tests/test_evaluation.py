from __future__ import annotations

import unittest

from token_prediction.dataset import PredictionTarget
from token_prediction.estimators import TokenForecast
from token_prediction.evaluation import (
    CalibrationExample,
    FittedExpansionCalibrator,
    IdentityCalibrator,
    ScoredForecast,
    TaskMaxConformalCalibrator,
    evaluate_forecasts,
    evaluate_task_forecasts,
)


class EvaluationTests(unittest.TestCase):
    def test_identity_calibrator_rejects_nonzero_expansion(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly zero"):
            FittedExpansionCalibrator("none", 0.1, 1.0)

    def test_fitted_calibrator_round_trip_is_strict_and_replays(self) -> None:
        target = PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS
        fitted = FittedExpansionCalibrator("task_max_conformal", 0.1, 3.5)
        loaded = FittedExpansionCalibrator.from_dict(fitted.to_dict())
        forecast = TokenForecast("p", target, 5, 10, 12)
        self.assertEqual(loaded, fitted)
        self.assertEqual(loaded.transform(forecast), fitted.transform(forecast))
        for tampered in (
            {**fitted.to_dict(), "extra": True},
            {**fitted.to_dict(), "expansion": -1.0},
            {**fitted.to_dict(), "interval_alpha": float("nan")},
            {**fitted.to_dict(), "calibrator_id": "unknown"},
        ):
            with self.assertRaises((TypeError, ValueError)):
                FittedExpansionCalibrator.from_dict(tampered)

    def test_metric_suite_matches_hand_calculation(self) -> None:
        target = PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS
        rows = (
            ScoredForecast("t0", "r0", TokenForecast("p0", target, 7, 8, 11), 10, 1),
            ScoredForecast("t1", "r1", TokenForecast("p1", target, 22, 25, 28), 20, 1),
        )
        metrics = evaluate_forecasts(rows, alpha=0.1)
        self.assertAlmostEqual(float(metrics["mae"]), 3.5)
        self.assertAlmostEqual(float(metrics["wape"]), 7 / 30)
        self.assertAlmostEqual(float(metrics["coverage"]), 0.5)
        self.assertAlmostEqual(float(metrics["task_simultaneous_coverage"]), 0.5)
        self.assertAlmostEqual(float(metrics["mean_bias"]), 1.5)
        self.assertAlmostEqual(float(metrics["underestimation_rate"]), 0.5)
        self.assertAlmostEqual(float(metrics["interval_score"]), 25.0)
        self.assertAlmostEqual(float(metrics["normalized_interval_width"]), 0.35)

    def test_zero_target_does_not_divide_by_zero(self) -> None:
        target = PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS
        metrics = evaluate_forecasts(
            (ScoredForecast("t", "r", TokenForecast("p", target, 0, 0, 1), 0, 1),)
        )
        self.assertEqual(metrics["wape"], 0.0)
        self.assertEqual(metrics["normalized_interval_width"], 1.0)

    def test_task_metrics_are_label_free_weighted_aggregates(self) -> None:
        target = PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS
        rows = (
            ScoredForecast("t0", "r0", TokenForecast("p0", target, 5, 8, 12), 10, 1),
            ScoredForecast("t0", "r1", TokenForecast("p1", target, 15, 18, 25), 20, 3),
            ScoredForecast("t1", "r2", TokenForecast("p2", target, 0, 2, 4), 3, 2),
        )
        metrics = evaluate_task_forecasts(rows, alpha=0.1)
        self.assertEqual(tuple(metrics), ("t0", "t1"))
        self.assertEqual(metrics["t0"].n_points, 2)
        self.assertEqual(metrics["t0"].n_trajectories, 2)
        self.assertEqual(metrics["t0"].weight_sum, 4)
        self.assertEqual(metrics["t0"].weighted_mae, 2)
        self.assertEqual(metrics["t0"].weighted_coverage, 1)
        self.assertNotIn("target_value", metrics["t0"].to_dict())

    def test_task_max_calibration_uses_one_conservative_score_per_task(self) -> None:
        target = PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS
        examples = (
            CalibrationExample("t0", TokenForecast("p0", target, 8, 10, 12), 20),
            CalibrationExample("t0", TokenForecast("p1", target, 8, 10, 12), 14),
            CalibrationExample("t1", TokenForecast("p2", target, 8, 10, 12), 13),
        )
        fitted = TaskMaxConformalCalibrator(alpha=0.1).fit(examples)
        transformed = fitted.transform(TokenForecast("test", target, 8, 10, 12))
        self.assertEqual((transformed.lower, transformed.upper), (0.0, 20.0))
        self.assertEqual(
            (transformed.raw_lower, transformed.raw_point, transformed.raw_upper),
            (8, 10, 12),
        )

    def test_calibration_preserves_estimator_native_quantiles(self) -> None:
        target = PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS
        forecast = TokenForecast(
            "point",
            target,
            0,
            9,
            9,
            raw_lower=-2,
            raw_point=9,
            raw_upper=8,
        )
        transformed = IdentityCalibrator().fit(()).transform(forecast)
        self.assertEqual(
            (transformed.raw_lower, transformed.raw_point, transformed.raw_upper),
            (-2, 9, 8),
        )

    def test_raw_interval_metrics_use_pre_calibration_quantiles(self) -> None:
        target = PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS
        rows = (
            ScoredForecast(
                "t0",
                "r0",
                TokenForecast(
                    "p0",
                    target,
                    5,
                    8,
                    15,
                    raw_lower=7,
                    raw_point=8,
                    raw_upper=11,
                ),
                10,
                1,
            ),
            ScoredForecast(
                "t1",
                "r1",
                TokenForecast(
                    "p1",
                    target,
                    10,
                    25,
                    30,
                    raw_lower=22,
                    raw_point=25,
                    raw_upper=28,
                ),
                20,
                1,
            ),
        )
        metrics = evaluate_forecasts(rows, alpha=0.1)
        self.assertEqual(metrics["raw_coverage"], 0.5)
        self.assertEqual(metrics["raw_task_simultaneous_coverage"], 0.5)
        self.assertEqual(metrics["raw_interval_score"], 25.0)
        self.assertEqual(metrics["raw_median_interval_width"], 4.0)
        self.assertAlmostEqual(metrics["raw_normalized_interval_width"], 0.35)
        self.assertEqual(metrics["quantile_crossing_rate"], 0.0)

    def test_raw_crossing_is_counted_before_validity_repair(self) -> None:
        target = PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS
        metrics = evaluate_forecasts(
            (
                ScoredForecast(
                    "task",
                    "trajectory",
                    TokenForecast(
                        "point",
                        target,
                        0,
                        10,
                        20,
                        raw_lower=20,
                        raw_point=10,
                        raw_upper=5,
                    ),
                    10,
                    1,
                ),
            )
        )
        self.assertEqual(metrics["quantile_crossing_rate"], 1.0)
        self.assertGreaterEqual(float(metrics["raw_interval_score"]), 0.0)

    def test_task_simultaneous_coverage_requires_every_point(self) -> None:
        target = PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS
        rows = (
            ScoredForecast(
                "t0",
                "r0",
                TokenForecast(
                    "p0",
                    target,
                    0,
                    5,
                    10,
                    raw_lower=0,
                    raw_point=5,
                    raw_upper=10,
                ),
                5,
                1,
            ),
            ScoredForecast(
                "t0",
                "r0",
                TokenForecast(
                    "p1",
                    target,
                    0,
                    5,
                    10,
                    raw_lower=0,
                    raw_point=15,
                    raw_upper=30,
                ),
                20,
                1,
            ),
            ScoredForecast(
                "t1",
                "r1",
                TokenForecast(
                    "p2",
                    target,
                    0,
                    5,
                    10,
                    raw_lower=10,
                    raw_point=15,
                    raw_upper=20,
                ),
                5,
                1,
            ),
        )
        metrics = evaluate_forecasts(rows)
        self.assertAlmostEqual(float(metrics["coverage"]), 2 / 3)
        self.assertEqual(metrics["task_simultaneous_coverage"], 0.5)
        self.assertAlmostEqual(float(metrics["raw_coverage"]), 2 / 3)
        self.assertEqual(metrics["raw_task_simultaneous_coverage"], 0.5)

    def test_partial_or_mixed_raw_forecasts_fail_closed(self) -> None:
        target = PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS
        with self.assertRaisesRegex(ValueError, "provided together"):
            TokenForecast("partial", target, 0, 1, 2, raw_lower=0)
        with self.assertRaisesRegex(ValueError, "all rows or no rows"):
            evaluate_forecasts(
                (
                    ScoredForecast(
                        "t0",
                        "r0",
                        TokenForecast(
                            "raw",
                            target,
                            0,
                            1,
                            2,
                            raw_lower=0,
                            raw_point=1,
                            raw_upper=2,
                        ),
                        1,
                        1,
                    ),
                    ScoredForecast(
                        "t1", "r1", TokenForecast("plain", target, 0, 1, 2), 1, 1
                    ),
                )
            )


if __name__ == "__main__":
    unittest.main()
