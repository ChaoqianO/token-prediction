from __future__ import annotations

import unittest

from token_prediction.dataset import PredictionPosition, PredictionTarget
from token_prediction.estimators import TokenForecast
from token_prediction.evaluation import (
    assert_calibration_raw_prediction_parity,
    compare_matched_coverage,
    compare_same_tasks_across_conditions,
)
from token_prediction.experiment import CandidateResult, PredictionRecord


TARGET = PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS


def _result(
    candidate_id: str,
    *,
    condition_id: str = "condition:a",
    calibrator_id: str = "task_max_conformal",
    lower: float = 5,
    upper: float = 15,
    raw_lower: float = 7,
    raw_upper: float = 13,
    coverage: float = 0.90,
    interval_score: float = 20,
    task_maes: tuple[float, float] = (1, 2),
) -> CandidateResult:
    records = tuple(
        PredictionRecord(
            candidate_id=candidate_id,
            point_id=f"{condition_id}-point-{index}",
            task_id=f"task-{index}",
            trajectory_id=f"trajectory-{index}",
            condition_id=condition_id,
            fold=index,
            target=TARGET,
            forecast=TokenForecast(
                f"{condition_id}-point-{index}",
                TARGET,
                lower,
                10,
                upper,
                raw_lower=raw_lower,
                raw_point=10,
                raw_upper=raw_upper,
            ),
            sample_weight=0.5,
        )
        for index in range(2)
    )
    return CandidateResult(
        candidate_id=candidate_id,
        candidate_hash=f"hash-{candidate_id}",
        dataset_id="dataset",
        split_plan_id="split",
        eligibility_hash=f"eligible-{condition_id}",
        position=PredictionPosition.TASK_UPDATE,
        target=TARGET,
        condition_id=condition_id,
        calibrator_id=calibrator_id,
        alpha=0.1,
        metric_suite_id="metrics",
        predictions=records,
        metrics={
            "task_simultaneous_coverage": coverage,
            "interval_score": interval_score,
        },
        task_metrics={
            f"task-{index}": {
                "weighted_mae": task_maes[index],
                "n_points": 1,
                "n_trajectories": 1,
                "weight_sum": 0.5,
                "weighted_interval_score": interval_score,
                "weighted_coverage": coverage,
            }
            for index in range(2)
        },
    )


class Stage4EvaluationTests(unittest.TestCase):
    def test_calibration_comparison_requires_exact_raw_prediction_parity(self) -> None:
        reference = _result(
            "model",
            calibrator_id="task_max_conformal",
            lower=5,
            upper=15,
        )
        candidate = _result(
            "model",
            calibrator_id="none",
            lower=7,
            upper=13,
        )
        assert_calibration_raw_prediction_parity(reference, candidate)
        changed = _result(
            "model",
            calibrator_id="none",
            lower=7,
            upper=13,
            raw_lower=8,
        )
        with self.assertRaisesRegex(ValueError, "raw predictions"):
            assert_calibration_raw_prediction_parity(reference, changed)

    def test_interval_score_winner_is_reported_only_at_matched_coverage(self) -> None:
        reference = _result(
            "model",
            calibrator_id="task_max_conformal",
            coverage=0.90,
            interval_score=20,
        )
        matched = _result(
            "model",
            calibrator_id="none",
            lower=7,
            upper=13,
            coverage=0.91,
            interval_score=18,
        )
        comparison = compare_matched_coverage(reference, matched, tolerance=0.02)
        self.assertTrue(comparison.matched)
        self.assertEqual(comparison.winner, "none")

        unmatched = _result(
            "model",
            calibrator_id="none",
            lower=7,
            upper=13,
            coverage=0.70,
            interval_score=10,
        )
        comparison = compare_matched_coverage(reference, unmatched, tolerance=0.02)
        self.assertFalse(comparison.matched)
        self.assertIsNone(comparison.winner)

    def test_same_task_pairing_averages_conditions_before_bootstrap(self) -> None:
        pairs = {
            "condition:a": (
                _result(
                    "candidate",
                    condition_id="condition:a",
                    task_maes=(1, 3),
                ),
                _result(
                    "reference",
                    condition_id="condition:a",
                    task_maes=(2, 4),
                ),
            ),
            "condition:b": (
                _result(
                    "candidate",
                    condition_id="condition:b",
                    task_maes=(3, 5),
                ),
                _result(
                    "reference",
                    condition_id="condition:b",
                    task_maes=(4, 6),
                ),
            ),
        }
        comparison = compare_same_tasks_across_conditions(
            pairs,
            iterations=200,
            seed=17,
        )
        self.assertEqual(comparison.condition_count, 2)
        self.assertEqual(comparison.task_count, 2)
        self.assertEqual(comparison.mean_mae_delta, -1)
        self.assertEqual(comparison.mae_delta_ci_lower, -1)
        self.assertEqual(comparison.mae_delta_ci_upper, -1)
        self.assertEqual(comparison.candidate_win_probability, 1)


if __name__ == "__main__":
    unittest.main()
