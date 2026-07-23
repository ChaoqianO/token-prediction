from __future__ import annotations

import unittest

from token_prediction.dataset import PredictionTarget
from token_prediction.estimators import TokenForecast
from token_prediction.evaluation import ScoredForecast, evaluate_budget_scenarios


TARGET = PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS


def _row(index: int, truth: float, lower: float, point: float, upper: float) -> ScoredForecast:
    return ScoredForecast(
        task_id=f"task-{index}",
        trajectory_id=f"trajectory-{index}",
        forecast=TokenForecast(
            point_id=f"point-{index}",
            target=TARGET,
            lower=lower,
            point=point,
            upper=upper,
        ),
        target_value=truth,
        sample_weight=1.0,
    )


class BudgetEvaluationTests(unittest.TestCase):
    def test_fixed_budget_decisions_and_interval_uncertainty(self) -> None:
        rows = (
            _row(0, 150, 110, 120, 140),
            _row(1, 50, 80, 130, 160),
        )
        report = evaluate_budget_scenarios(rows, budgets=(100,))
        metrics = report["scenarios"]["100"]
        self.assertEqual(report["threshold_policy"], "explicit_fixed_remaining_token_budgets_v1")
        self.assertEqual(metrics["actual_overrun_rate"], 0.5)
        self.assertEqual(metrics["predicted_overrun_rate"], 1.0)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["false_positive_rate"], 1.0)
        self.assertEqual(metrics["interval_definite_overrun_rate"], 0.5)
        self.assertEqual(metrics["interval_uncertain_rate"], 0.5)

    def test_budget_thresholds_are_explicit_unique_positive_integers(self) -> None:
        rows = (_row(0, 150, 110, 120, 140),)
        for budgets in ((), (100, 100), (0,), (100.0,)):
            with self.subTest(budgets=budgets):
                with self.assertRaisesRegex(ValueError, "budgets"):
                    evaluate_budget_scenarios(rows, budgets=budgets)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
