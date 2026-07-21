from __future__ import annotations

import copy
import unittest
from pathlib import Path, PureWindowsPath

from scripts import run_stage3_experiments as stage3
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
        "results_schema_version": stage3.STAGE3_RESULTS_SCHEMA_VERSION,
        "stage_name": stage3.STAGE3_STAGE_NAME,
        "run_policy_id": stage3.STAGE3_RUN_POLICY_ID,
        "artifact_layout_id": stage3.STAGE3_ARTIFACT_LAYOUT_ID,
        "run_id": "run",
        "source": {},
        "data_foundation": {},
        "code_binding": {},
        "runtime_versions": {},
        "dataset": {},
        "development_protocol": {},
        "matrix": {},
        "experiments": [],
        "gates": [],
        "summary": {},
        "final_holdout": {
            "evaluated": False,
            "prediction_count": 0,
            "target_values_used_for_fit_calibration_scoring": False,
            "selection_claim": "none",
        },
    }
    value["results_payload_sha256"] = stage3._semantic_sha256(value)
    return value


class Stage3RunnerTests(unittest.TestCase):
    def test_prediction_projection_excludes_latency_but_binds_forecasts(self) -> None:
        first = _result(latency_ms=1.0)
        second = _result(latency_ms=999.0)
        changed = _result(latency_ms=1.0, prediction=91.0)
        self.assertEqual(
            stage3.prediction_projection_sha256(first),
            stage3.prediction_projection_sha256(second),
        )
        self.assertNotEqual(
            stage3.prediction_projection_sha256(first),
            stage3.prediction_projection_sha256(changed),
        )
        self.assertEqual(
            stage3.cohort_projection_sha256(first),
            stage3.cohort_projection_sha256(changed),
        )

    def test_task_metric_projection_pseudonymizes_private_ids(self) -> None:
        projection = stage3._task_metric_projection(_result())
        self.assertEqual(len(projection), 1)
        self.assertNotIn("private-task", str(projection))
        self.assertRegex(str(projection[0]["task_pseudonym"]), r"^[0-9a-f]{64}$")

    def test_results_digest_holdout_and_private_fields_fail_closed(self) -> None:
        value = _results_document()
        self.assertEqual(
            stage3.verify_stage3_results_document(value),
            value["results_payload_sha256"],
        )
        tampered = copy.deepcopy(value)
        tampered["run_id"] = "changed"
        with self.assertRaisesRegex(stage3.Stage3ExperimentError, "does not close"):
            stage3.verify_stage3_results_document(tampered)
        opened = copy.deepcopy(value)
        opened["final_holdout"]["evaluated"] = True
        opened["results_payload_sha256"] = stage3._semantic_sha256(
            {
                key: item
                for key, item in opened.items()
                if key != "results_payload_sha256"
            }
        )
        with self.assertRaisesRegex(stage3.Stage3ExperimentError, "not sealed"):
            stage3.verify_stage3_results_document(opened)
        private = copy.deepcopy(value)
        private["source"]["task_id"] = "private-task"
        private["results_payload_sha256"] = stage3._semantic_sha256(
            {
                key: item
                for key, item in private.items()
                if key != "results_payload_sha256"
            }
        )
        with self.assertRaisesRegex(stage3.Stage3ExperimentError, "forbidden raw field"):
            stage3.verify_stage3_results_document(private)

    def test_output_root_is_restricted_to_ignored_stage3_tree(self) -> None:
        root = Path(__file__).resolve().parents[1]
        relative, resolved = stage3._safe_output_root(
            root,
            "workspace/stage3/runs",
        )
        self.assertEqual(relative, "workspace/stage3/runs")
        self.assertTrue(resolved.is_absolute())
        for unsafe in (
            "workspace/stage3",
            "workspace/elsewhere",
            "../workspace/stage3/runs",
            "C:/outside",
        ):
            with self.subTest(path=unsafe):
                with self.assertRaises((ValueError, stage3.Stage3ExperimentError)):
                    stage3._safe_output_root(root, unsafe)

    def test_nested_gru_bundle_fits_windows_legacy_path_budget(self) -> None:
        root = PureWindowsPath(r"E:\kabuda\token prediction")
        experiment_key = stage3._artifact_key("e", "x" * 200)
        candidate_key = stage3._artifact_key("c", "y" * 200)
        deepest = (
            root
            / "workspace"
            / "stage3"
            / "runs"
            / stage3._output_key("f" * 24)
            / "fold_artifacts"
            / experiment_key
            / candidate_key
            / "seed_20260719"
            / "fold_0"
            / "bundle"
            / "components"
            / ("f" * 64)
            / "model"
            / "gru"
            / "weights.safetensors"
        )
        self.assertLess(len(str(deepest)), 260)


if __name__ == "__main__":
    unittest.main()
