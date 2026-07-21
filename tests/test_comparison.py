from __future__ import annotations

import unittest

from token_prediction.dataset import PredictionPosition, PredictionTarget
from token_prediction.estimators import TokenForecast
from token_prediction.evaluation import paired_task_bootstrap, paired_task_metric_bootstrap
from token_prediction.experiment import CandidateResult, PredictionRecord


TARGET = PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS


def _record(
    candidate_id: str,
    point_id: str,
    task_id: str,
    prediction: float,
    *,
    trajectory_id: str | None = None,
    weight: float = 0.5,
    condition_id: str = "condition-a",
) -> PredictionRecord:
    return PredictionRecord(
        candidate_id=candidate_id,
        point_id=point_id,
        task_id=task_id,
        trajectory_id=trajectory_id or f"trajectory-{task_id}",
        condition_id=condition_id,
        fold=0,
        target=TARGET,
        forecast=TokenForecast(point_id, TARGET, prediction, prediction, prediction),
        sample_weight=weight,
    )


def _result(
    candidate_id: str,
    records: tuple[PredictionRecord, ...],
    *,
    task_maes: dict[str, float] | None = None,
) -> CandidateResult:
    task_metrics = (
        {
            "task-a": {
                "n_points": 2,
                "n_trajectories": 1,
                "weight_sum": 0.5,
                "weighted_mae": task_maes["task-a"],
                "weighted_interval_score": 0.0,
                "weighted_coverage": 1.0,
            },
            "task-b": {
                "n_points": 1,
                "n_trajectories": 1,
                "weight_sum": 0.5,
                "weighted_mae": task_maes["task-b"],
                "weighted_interval_score": 0.0,
                "weighted_coverage": 1.0,
            },
        }
        if task_maes is not None
        else {}
    )
    return CandidateResult(
        candidate_id=candidate_id,
        candidate_hash=f"hash-{candidate_id}",
        dataset_id="dataset-a",
        split_plan_id="split-a",
        eligibility_hash="eligible-a",
        position=PredictionPosition.TASK_UPDATE,
        target=TARGET,
        condition_id="condition-a",
        calibrator_id="none",
        alpha=0.1,
        metric_suite_id="metrics-a",
        predictions=records,
        metrics={},
        task_metrics=task_metrics,
    )


class PairedTaskBootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.truth = {"a-1": 10.0, "a-2": 20.0, "b-1": 30.0}
        self.candidate_records = (
            _record("candidate", "a-1", "task-a", 9.0, weight=0.25),
            _record("candidate", "a-2", "task-a", 19.0, weight=0.25),
            _record("candidate", "b-1", "task-b", 29.0, weight=0.5),
        )
        self.reference_records = (
            _record("reference", "a-1", "task-a", 8.0, weight=0.25),
            _record("reference", "a-2", "task-a", 18.0, weight=0.25),
            _record("reference", "b-1", "task-b", 28.0, weight=0.5),
        )

    def test_candidate_results_produce_task_clustered_paired_comparison(self) -> None:
        comparison = paired_task_bootstrap(
            _result("candidate", self.candidate_records),
            _result("reference", self.reference_records),
            self.truth,
            iterations=500,
            seed=17,
        )
        self.assertEqual(comparison.n_points, 3)
        self.assertEqual(comparison.n_tasks, 2)
        self.assertEqual(comparison.candidate_mae, 1.0)
        self.assertEqual(comparison.reference_mae, 2.0)
        self.assertEqual(comparison.mae_delta, -1.0)
        self.assertEqual(comparison.mae_delta_ci_lower, -1.0)
        self.assertEqual(comparison.mae_delta_ci_upper, -1.0)
        self.assertEqual(comparison.candidate_win_probability, 1.0)

    def test_prediction_sequences_are_reproducible_and_order_independent(self) -> None:
        first = paired_task_bootstrap(
            self.candidate_records,
            self.reference_records,
            self.truth,
            iterations=257,
            seed=91,
        )
        second = paired_task_bootstrap(
            tuple(reversed(self.candidate_records)),
            tuple(reversed(self.reference_records)),
            self.truth,
            iterations=257,
            seed=91,
        )
        self.assertEqual(first, second)

    def test_label_free_task_metrics_reproduce_the_paired_mae_comparison(self) -> None:
        candidate = _result(
            "candidate",
            self.candidate_records,
            task_maes={"task-a": 1.0, "task-b": 1.0},
        )
        reference = _result(
            "reference",
            self.reference_records,
            task_maes={"task-a": 2.0, "task-b": 2.0},
        )
        comparison = paired_task_metric_bootstrap(
            candidate,
            reference,
            iterations=500,
            seed=17,
        )
        self.assertEqual(comparison.n_points, 3)
        self.assertEqual(comparison.n_tasks, 2)
        self.assertEqual(comparison.candidate_mae, 1.0)
        self.assertEqual(comparison.reference_mae, 2.0)
        self.assertEqual(comparison.mae_delta, -1.0)
        self.assertEqual(comparison.candidate_win_probability, 1.0)

        incompatible = CandidateResult(
            **{
                **reference.__dict__,
                "dataset_id": "another-dataset",
            }
        )
        with self.assertRaisesRegex(ValueError, "same experiment cohort"):
            paired_task_metric_bootstrap(candidate, incompatible, iterations=10)

    def test_rejects_point_task_and_weight_mismatches(self) -> None:
        with self.assertRaisesRegex(ValueError, "cohorts differ"):
            paired_task_bootstrap(
                self.candidate_records[:-1],
                self.reference_records,
                self.truth,
                iterations=10,
            )

        wrong_task = (*self.reference_records[:-1], _record("reference", "b-1", "other", 28.0))
        with self.assertRaisesRegex(ValueError, "cohort metadata differs"):
            paired_task_bootstrap(
                self.candidate_records,
                wrong_task,
                self.truth,
                iterations=10,
            )

        wrong_weight = (
            *self.reference_records[:-1],
            _record("reference", "b-1", "task-b", 28.0, weight=0.25),
        )
        with self.assertRaisesRegex(ValueError, "sample weights differ"):
            paired_task_bootstrap(
                self.candidate_records,
                wrong_weight,
                self.truth,
                iterations=10,
            )

    def test_rejects_truth_and_candidate_result_cohort_mismatches(self) -> None:
        with self.assertRaisesRegex(ValueError, "truth mapping"):
            paired_task_bootstrap(
                self.candidate_records,
                self.reference_records,
                {**self.truth, "extra": 4.0},
                iterations=10,
            )

        reference = _result("reference", self.reference_records)
        incompatible = CandidateResult(
            **{
                **reference.__dict__,
                "dataset_id": "another-dataset",
            }
        )
        with self.assertRaisesRegex(ValueError, "same experiment cohort"):
            paired_task_bootstrap(
                _result("candidate", self.candidate_records),
                incompatible,
                self.truth,
                iterations=10,
            )


if __name__ == "__main__":
    unittest.main()
