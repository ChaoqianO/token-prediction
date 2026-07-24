from __future__ import annotations

import copy
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from scripts.prepare_stage4_selection import SOURCE_ARTIFACTS
from scripts.run_stage4_final import (
    FINAL_COHORT_PROJECTION_ID,
    FINAL_RESULTS_SCHEMA_VERSION,
    FINAL_RUN_POLICY_ID,
    FINAL_SCORE_PROJECTION_ID,
    FINAL_STAGE_NAME,
    SELECTION_LOCK_POLICY_ID,
    SELECTION_LOCK_SCHEMA_VERSION,
    SELECTION_TAG,
    Stage4FinalError,
    _directory_projection_sha256,
    _validate_selection_lock_document,
    verify_final_results_document,
)
from token_prediction.final_ensemble import semantic_sha256


def _selection_lock() -> dict[str, object]:
    return {
        "selection_lock_schema_version": SELECTION_LOCK_SCHEMA_VERSION,
        "policy_id": SELECTION_LOCK_POLICY_ID,
        "selection_tag": SELECTION_TAG,
        "selection_artifact": {
            "path": "workspace/stage4/selection/s4sel-example",
            "artifact_id": "a" * 64,
            "run_id": "b" * 24,
            "selection_id": "c" * 64,
            "selection_payload_sha256": "d" * 64,
            "selection_code_commit": "e" * 40,
            "selection_code_tree_sha256": "f" * 64,
        },
        "source_artifacts": [asdict(value) for value in SOURCE_ARTIFACTS],
        "protocol": {
            "selection_policy_id": "stage4_development_only_stability_guard_v1",
            "ensemble_policy_id": "development_three_seed_five_fold_mean_v1",
            "final_holdout_evaluation_count": 1,
            "refit_selected_learned_models": False,
            "calibration_application_count": 1,
            "resume_policy_id": FINAL_RUN_POLICY_ID,
        },
    }


def _checkpoint(index: int, selection_id: str) -> dict[str, object]:
    value: dict[str, object] = {
        "checkpoint_schema_version": 1,
        "run_policy_id": FINAL_RUN_POLICY_ID,
        "selection_id": selection_id,
        "cell_id": semantic_sha256(["cell", index]),
        "source_name": f"source-{index % 4}",
        "source_id": f"source-id-{index % 4}",
        "condition_id": f"condition-{index}",
        "position": "call_pre",
        "target": "call_billable_total_tokens",
        "candidate_id": "lightgbm_history",
        "candidate_hash": "a" * 64,
        "calibrator_id": "task_max_conformal",
        "alpha": 0.1,
        "final_dataset": {},
        "model_execution": {
            "member_count": 15,
        },
        "metrics": {},
        "task_metrics": [],
        "diagnostics": {},
        "prediction_projection_id": FINAL_SCORE_PROJECTION_ID,
        "prediction_projection_sha256": "b" * 64,
        "cohort_projection_id": FINAL_COHORT_PROJECTION_ID,
        "cohort_projection_sha256": "c" * 64,
        "prediction_count": index + 1,
    }
    value["checkpoint_payload_sha256"] = semantic_sha256(value)
    return value


def _final_results() -> dict[str, object]:
    selection_id = "d" * 64
    cells = [_checkpoint(index, selection_id) for index in range(29)]
    prediction_count = sum(value["prediction_count"] for value in cells)
    base: dict[str, object] = {
        "results_schema_version": FINAL_RESULTS_SCHEMA_VERSION,
        "stage_name": FINAL_STAGE_NAME,
        "run_policy_id": FINAL_RUN_POLICY_ID,
        "run_id": "run-id",
        "selection": {"selection_id": selection_id},
        "evaluation_code_binding": {},
        "datasets": [{}, {}, {}, {}],
        "cells": cells,
        "summary": {
            "source_count": 4,
            "cell_count": 29,
            "ensemble_member_count": 435,
            "prediction_count": prediction_count,
        },
        "final_holdout": {
            "evaluated": True,
            "evaluation_count": 1,
            "prediction_count": prediction_count,
            "target_values_used_for_fit": False,
            "target_values_used_for_calibration": False,
            "target_values_used_for_scoring": True,
            "model_selection_after_open": False,
        },
    }
    base["results_payload_sha256"] = semantic_sha256(base)
    return base


class Stage4FinalTests(unittest.TestCase):
    def test_selection_lock_requires_exact_one_time_no_refit_protocol(self) -> None:
        _validate_selection_lock_document(_selection_lock())
        tampered = _selection_lock()
        tampered["protocol"]["final_holdout_evaluation_count"] = 2
        with self.assertRaisesRegex(Stage4FinalError, "protocol"):
            _validate_selection_lock_document(tampered)

    def test_final_results_close_over_all_cells_and_predictions(self) -> None:
        results = _final_results()
        self.assertEqual(
            verify_final_results_document(results),
            results["results_payload_sha256"],
        )

    def test_final_results_reject_post_open_selection_claim(self) -> None:
        results = copy.deepcopy(_final_results())
        results["final_holdout"]["model_selection_after_open"] = True
        payload = dict(results)
        payload.pop("results_payload_sha256")
        results["results_payload_sha256"] = semantic_sha256(payload)
        with self.assertRaisesRegex(Stage4FinalError, "protocol"):
            verify_final_results_document(results)

    def test_bundle_directory_projection_is_path_and_content_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text("one", encoding="utf-8")
            (root / "nested").mkdir()
            (root / "nested" / "weights.bin").write_bytes(b"two")
            first = _directory_projection_sha256(root)
            (root / "nested" / "weights.bin").write_bytes(b"three")
            second = _directory_projection_sha256(root)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
