from __future__ import annotations

import copy
import unittest
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

from scripts import run_stage4_experiments as stage4
from token_prediction.dataset import PredictionPosition, PredictionTarget
from token_prediction.estimators import TokenForecast
from token_prediction.experiment import CandidateResult, PredictionRecord


TARGET = PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS


def _result(*, latency_ms: float = 1.0, prediction: float = 90.0) -> CandidateResult:
    record = PredictionRecord(
        candidate_id="candidate",
        point_id="point",
        task_id="private-task",
        trajectory_id="private-trajectory",
        condition_id="condition:test",
        fold=0,
        target=TARGET,
        forecast=TokenForecast(
            "point",
            TARGET,
            80,
            prediction,
            100,
            raw_lower=81,
            raw_point=prediction,
            raw_upper=99,
            latency_ms=latency_ms,
        ),
        sample_weight=1.0,
    )
    return CandidateResult(
        candidate_id="candidate",
        candidate_hash="a" * 64,
        dataset_id="b" * 64,
        split_plan_id="c" * 64,
        eligibility_hash="d" * 64,
        position=PredictionPosition.TASK_UPDATE,
        target=TARGET,
        condition_id="condition:test",
        calibrator_id="task_max_conformal",
        alpha=0.1,
        metric_suite_id="token_prediction_metrics_v2",
        predictions=(record,),
        metrics={"mae": 1.0},
        task_metrics={
            "private-task": {
                "n_points": 1,
                "n_trajectories": 1,
                "weight_sum": 1.0,
                "weighted_mae": 1.0,
                "weighted_interval_score": 2.0,
                "weighted_coverage": 1.0,
            }
        },
    )


def _results_document() -> dict[str, object]:
    value: dict[str, object] = {
        "results_schema_version": stage4.STAGE4_RESULTS_SCHEMA_VERSION,
        "stage_name": stage4.STAGE4_STAGE_NAME,
        "run_policy_id": stage4.STAGE4_RUN_POLICY_ID,
        "artifact_layout_id": stage4.STAGE4_ARTIFACT_LAYOUT_ID,
        "checkpoint_policy_id": stage4.STAGE4_CHECKPOINT_POLICY_ID,
        "run_id": "run",
        "source": {},
        "data_foundation": {},
        "code_binding": {},
        "runtime_versions": {},
        "dataset": {},
        "development_protocol": {},
        "matrix": {},
        "experiments": [],
        "matched_coverage_calibration": [],
        "paired_same_task_across_conditions": [],
        "summary": {},
        "final_holdout": {
            "evaluated": False,
            "prediction_count": 0,
            "target_values_used_for_fit_calibration_scoring": False,
            "selection_claim": "none",
        },
    }
    value["results_payload_sha256"] = stage4._semantic_sha256(value)
    return value


class Stage4RunnerTests(unittest.TestCase):
    def test_neural_and_lifecycle_candidates_require_reloadable_bundles(self) -> None:
        point_mlp = SimpleNamespace(
            estimator_id="independent_mlp",
            graph=SimpleNamespace(is_lifecycle=False),
        )
        lifecycle = SimpleNamespace(
            estimator_id="cross_position_deduct",
            graph=SimpleNamespace(is_lifecycle=True),
        )
        stateless = SimpleNamespace(
            estimator_id="empirical_quantile",
            graph=SimpleNamespace(is_lifecycle=False),
        )
        self.assertTrue(stage4._requires_reloadable_bundle(point_mlp))
        self.assertTrue(stage4._requires_reloadable_bundle(lifecycle))
        self.assertFalse(stage4._requires_reloadable_bundle(stateless))

    def test_prediction_projection_excludes_latency_but_binds_forecasts(self) -> None:
        first = _result(latency_ms=1.0)
        second = _result(latency_ms=999.0)
        changed = _result(latency_ms=1.0, prediction=91.0)
        self.assertEqual(
            stage4.prediction_projection_sha256(first),
            stage4.prediction_projection_sha256(second),
        )
        self.assertNotEqual(
            stage4.prediction_projection_sha256(first),
            stage4.prediction_projection_sha256(changed),
        )
        self.assertEqual(
            stage4.cohort_projection_sha256(first),
            stage4.cohort_projection_sha256(changed),
        )

    def test_task_metric_projection_pseudonymizes_private_ids(self) -> None:
        projection = stage4._task_metric_projection(_result())
        self.assertEqual(len(projection), 1)
        self.assertNotIn("private-task", str(projection))
        self.assertRegex(str(projection[0]["task_pseudonym"]), r"^[0-9a-f]{64}$")

    def test_results_digest_holdout_and_private_fields_fail_closed(self) -> None:
        value = _results_document()
        self.assertEqual(
            stage4.verify_stage4_results_document(value),
            value["results_payload_sha256"],
        )
        tampered = copy.deepcopy(value)
        tampered["run_id"] = "changed"
        with self.assertRaisesRegex(stage4.Stage4ExperimentError, "does not close"):
            stage4.verify_stage4_results_document(tampered)
        wrong_checkpoint_policy = copy.deepcopy(value)
        wrong_checkpoint_policy["checkpoint_policy_id"] = "save_sometimes"
        with self.assertRaisesRegex(stage4.Stage4ExperimentError, "policy identity"):
            stage4.verify_stage4_results_document(wrong_checkpoint_policy)
        malformed_digest = copy.deepcopy(value)
        malformed_digest["results_payload_sha256"] = "G" * 64
        with self.assertRaisesRegex(stage4.Stage4ExperimentError, "lowercase SHA-256"):
            stage4.verify_stage4_results_document(malformed_digest)
        opened = copy.deepcopy(value)
        opened["final_holdout"]["evaluated"] = True
        opened["results_payload_sha256"] = stage4._semantic_sha256(
            {key: item for key, item in opened.items() if key != "results_payload_sha256"}
        )
        with self.assertRaisesRegex(stage4.Stage4ExperimentError, "opened final holdout"):
            stage4.verify_stage4_results_document(opened)
        private = copy.deepcopy(value)
        private["source"]["task_id"] = "private-task"
        private["results_payload_sha256"] = stage4._semantic_sha256(
            {key: item for key, item in private.items() if key != "results_payload_sha256"}
        )
        with self.assertRaisesRegex(stage4.Stage4ExperimentError, "forbidden raw field"):
            stage4.verify_stage4_results_document(private)

    def test_output_and_checkpoint_roots_are_restricted(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for relative, prefix in (
            ("workspace/stage4/runs", stage4.ALLOWED_OUTPUT_PREFIX),
            ("workspace/stage4/checkpoints", stage4.ALLOWED_CHECKPOINT_PREFIX),
        ):
            canonical, resolved = stage4._safe_workspace_root(
                root,
                relative,
                label="test root",
                prefix=prefix,
            )
            self.assertEqual(canonical, relative)
            self.assertTrue(resolved.is_absolute())
        unsafe_cases = (
            ("workspace/stage4", stage4.ALLOWED_OUTPUT_PREFIX),
            ("workspace/stage4/checkpoints", stage4.ALLOWED_OUTPUT_PREFIX),
            ("workspace/stage4/runs", stage4.ALLOWED_CHECKPOINT_PREFIX),
            ("../workspace/stage4/runs", stage4.ALLOWED_OUTPUT_PREFIX),
            ("C:/outside", stage4.ALLOWED_OUTPUT_PREFIX),
        )
        for unsafe, prefix in unsafe_cases:
            with self.subTest(path=unsafe, prefix=prefix):
                with self.assertRaises((ValueError, stage4.Stage4ExperimentError)):
                    stage4._safe_workspace_root(
                        root,
                        unsafe,
                        label="test root",
                        prefix=prefix,
                    )

    def test_nested_bundle_fits_windows_legacy_path_budget(self) -> None:
        root = PureWindowsPath(r"E:\kabuda\token prediction")
        experiment_key = stage4._artifact_key("e", "x" * 200)
        candidate_key = stage4._artifact_key("c", "y" * 200)
        deepest = (
            root
            / "workspace"
            / "stage4"
            / "runs"
            / stage4._output_key("f" * 24)
            / "fold_artifacts"
            / experiment_key
            / candidate_key
            / "seed_20260719"
            / "fold_0"
            / "bundle"
            / "components"
            / ("f" * 64)
            / "model-0.txt"
        )
        self.assertLess(len(str(deepest)), 260)


if __name__ == "__main__":
    unittest.main()
