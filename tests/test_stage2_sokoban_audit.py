from __future__ import annotations

import unittest

from scripts.audit_stage2_sokoban import (
    EXPECTED_STAGE1_ARTIFACT_ID,
    SOKOBAN_AUDIT_POLICY_ID,
    SOKOBAN_AUDIT_SCHEMA_VERSION,
    SOKOBAN_AUDIT_STAGE_NAME,
    verify_sokoban_audit_results,
)
from scripts.run_stage2_experiments import Stage2ExperimentError, _semantic_sha256


def _valid_results() -> dict[str, object]:
    value: dict[str, object] = {
        "audit_schema_version": SOKOBAN_AUDIT_SCHEMA_VERSION,
        "stage_name": SOKOBAN_AUDIT_STAGE_NAME,
        "policy_id": SOKOBAN_AUDIT_POLICY_ID,
        "source": {
            "source_id": "bagen_sokoban_dialogues_v1",
            "revision": "1" * 64,
            "source_descriptor_hash": "2" * 64,
            "capability_contract_hash": "3" * 64,
            "manifest_path": "configs/stage2_auxiliary_sources.json",
            "manifest_sha256": "4" * 64,
            "raw_artifact_sha256": "5" * 64,
        },
        "dataset": {
            "dataset_id": "6" * 64,
            "row_count": 40,
            "trajectory_count": 20,
            "task_count": 20,
            "task_set_sha256": "7" * 64,
            "condition_count": 1,
            "status_counts": {
                "observed": 30,
                "censored": 0,
                "missing": 10,
                "invalid": 0,
            },
        },
        "lifecycle_compatibility": {
            "condition_count": 1,
            "sequence_count": 20,
            "step_count": 30,
            "scored_step_count": 20,
            "unscored_step_count": 10,
            "offline_shadow_prediction_count": 2,
            "offline_shadow_exact": True,
            "prediction_projection_sha256": "8" * 64,
        },
        "development_gate": {
            "status": "estimable",
            "reason": "nested_five_fold_protocol_available",
            "protocol_id": "9" * 64,
            "development_task_count": 15,
            "final_holdout_task_count": 5,
            "outer_folds": 5,
            "inner_folds": 5,
            "folds_reduced": False,
            "target_values_used_for_gate": False,
        },
        "stage1_regression": {
            "artifact_id": EXPECTED_STAGE1_ARTIFACT_ID,
            "artifact_manifest_sha256": "a" * 64,
            "bundle_count": 20,
            "parity_record_count": 992,
            "parity_mismatch_count": 0,
            "parity_sha256": "b" * 64,
            "historical_source_binding_status": "unrecoverable",
        },
        "final_holdout": {
            "evaluated": False,
            "prediction_count": 0,
            "target_values_used_for_fit_calibration_scoring": False,
            "selection_claim": "none",
        },
    }
    value["results_payload_sha256"] = _semantic_sha256(value)
    return value


class Stage2SokobanAuditTests(unittest.TestCase):
    def test_valid_aggregate_evidence_closes(self) -> None:
        value = _valid_results()
        self.assertEqual(
            verify_sokoban_audit_results(value),
            value["results_payload_sha256"],
        )

    def test_gate_and_final_holdout_tampering_fail_closed(self) -> None:
        value = _valid_results()
        value["development_gate"]["status"] = "not_estimable"
        value["results_payload_sha256"] = _semantic_sha256(
            {key: item for key, item in value.items() if key != "results_payload_sha256"}
        )
        with self.assertRaisesRegex(Stage2ExperimentError, "estimability"):
            verify_sokoban_audit_results(value)

        value = _valid_results()
        value["final_holdout"]["evaluated"] = True
        value["results_payload_sha256"] = _semantic_sha256(
            {key: item for key, item in value.items() if key != "results_payload_sha256"}
        )
        with self.assertRaisesRegex(Stage2ExperimentError, "holdout"):
            verify_sokoban_audit_results(value)

    def test_payload_tampering_fails_closed(self) -> None:
        value = _valid_results()
        value["dataset"]["row_count"] = 41
        with self.assertRaises(Stage2ExperimentError):
            verify_sokoban_audit_results(value)


if __name__ == "__main__":
    unittest.main()
