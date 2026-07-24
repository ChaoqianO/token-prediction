from __future__ import annotations

import copy
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from scripts.prepare_stage4_selection import SOURCE_ARTIFACTS
from scripts.run_stage4_final import (
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SELECTION_LOCK,
    FINAL_COHORT_PROJECTION_ID,
    FINAL_EVALUATION_EXPLICIT_PATHS,
    FINAL_RESULTS_SCHEMA_VERSION,
    FINAL_RUN_POLICY_ID,
    FINAL_SCORE_PROJECTION_ID,
    FINAL_STAGE_NAME,
    SELECTION_LOCK_POLICY_ID,
    SELECTION_LOCK_SCHEMA_VERSION,
    SELECTION_TAG,
    SelectionLockContext,
    Stage4FinalError,
    _directory_projection_sha256,
    _exclusive_final_process_lock,
    _evaluate_missing_cells,
    _open_final_tombstone,
    _publish_final_tombstone,
    _require_canonical_final_arguments,
    _verify_checkpoint_selection_binding,
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
    def test_final_paths_are_fixed_and_alternate_roots_are_rejected(self) -> None:
        _require_canonical_final_arguments(
            selection_lock=DEFAULT_SELECTION_LOCK,
            output_root=DEFAULT_OUTPUT_ROOT,
            checkpoint_root=DEFAULT_CHECKPOINT_ROOT,
        )
        with self.assertRaisesRegex(Stage4FinalError, "exactly"):
            _require_canonical_final_arguments(
                selection_lock=DEFAULT_SELECTION_LOCK,
                output_root=f"{DEFAULT_OUTPUT_ROOT}/alternate",
                checkpoint_root=DEFAULT_CHECKPOINT_ROOT,
            )
        with self.assertRaisesRegex(Stage4FinalError, "exactly"):
            _require_canonical_final_arguments(
                selection_lock=DEFAULT_SELECTION_LOCK,
                output_root=DEFAULT_OUTPUT_ROOT,
                checkpoint_root=f"{DEFAULT_CHECKPOINT_ROOT}/alternate",
            )

    def test_final_process_lock_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with _exclusive_final_process_lock(root):
                with self.assertRaisesRegex(Stage4FinalError, "already holds"):
                    with _exclusive_final_process_lock(root):
                        self.fail("nested final process lock unexpectedly succeeded")

    def test_tombstone_fails_closed_without_ledger_and_publishes_once(self) -> None:
        selection_id = "a" * 64
        selection_commit = "b" * 40
        run_id = "c" * 24
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = root / DEFAULT_CHECKPOINT_ROOT / run_id / "ledger.json"
            tombstone = _open_final_tombstone(
                root,
                selection_id=selection_id,
                selection_commit=selection_commit,
                run_id=run_id,
                ledger_path=ledger,
            )
            with self.assertRaisesRegex(Stage4FinalError, "no resumable canonical ledger"):
                _open_final_tombstone(
                    root,
                    selection_id=selection_id,
                    selection_commit=selection_commit,
                    run_id=run_id,
                    ledger_path=ledger,
                )
            ledger.parent.mkdir(parents=True)
            ledger.write_text("{}", encoding="utf-8")
            self.assertEqual(
                _open_final_tombstone(
                    root,
                    selection_id=selection_id,
                    selection_commit=selection_commit,
                    run_id=run_id,
                    ledger_path=ledger,
                ),
                tombstone,
            )
            _publish_final_tombstone(
                tombstone,
                selection_id=selection_id,
                selection_commit=selection_commit,
                run_id=run_id,
                final_artifact_id="d" * 64,
            )
            with self.assertRaisesRegex(Stage4FinalError, "published"):
                _open_final_tombstone(
                    root,
                    selection_id=selection_id,
                    selection_commit=selection_commit,
                    run_id=run_id,
                    ledger_path=ledger,
                )

    def test_final_code_closure_contains_all_executed_loaders_and_controls(self) -> None:
        self.assertTrue(
            {
                "scripts/run_data_foundation_baseline.py",
                "scripts/run_stage2_experiments.py",
                "scripts/run_stage3_experiments.py",
                "scripts/run_stage4_experiments.py",
                "configs/data_foundation_v2_baseline.json",
                "configs/stage2_auxiliary_sources.json",
                "configs/source_descriptors/bagen_swebench.json",
                "configs/source_descriptors/spend_openhands.json",
            }
            <= FINAL_EVALUATION_EXPLICIT_PATHS
        )

    def test_resume_rejects_ledger_checkpoint_disagreement(self) -> None:
        selection_id = "a" * 64
        lock = SelectionLockContext(
            path=DEFAULT_SELECTION_LOCK,
            sha256="b" * 64,
            document={},
            selection_root=Path("."),
            selection_manifest_id="c" * 64,
            selection={"selection_id": selection_id, "cells": []},
            selection_commit="d" * 40,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoints" / ("e" * 24)
            checkpoint.mkdir(parents=True)
            ledger = checkpoint / "ledger.json"
            ledger.write_text(
                '{"completed_cell_ids":["orphaned-cell"]}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Stage4FinalError, "cannot be reopened"):
                _evaluate_missing_cells(root, lock, checkpoint, ledger)

    def test_resume_checkpoint_is_bound_to_selected_model_and_dataset(self) -> None:
        member_hash = "1" * 64
        cell = {
            "cell_id": "2" * 64,
            "source_name": "source-0",
            "source_id": "source-id-0",
            "condition_id": "condition-0",
            "position": "call_pre",
            "target": "call_billable_total_tokens",
            "candidate_id": "lightgbm_history",
            "candidate_hash": "3" * 64,
            "calibrator_id": "task_max_conformal",
            "alpha": 0.1,
            "selected_artifact_key": "stage4_bagen_sokoban",
            "members": [{"member_sha256": member_hash}],
        }
        selection = {
            "source_artifacts": [
                {
                    "source_name": "source-0",
                    "derived_dataset_id": "4" * 64,
                    "development_protocol_id": "5" * 64,
                }
            ]
        }
        checkpoint = {
            key: cell[key]
            for key in (
                "cell_id",
                "source_name",
                "source_id",
                "condition_id",
                "position",
                "target",
                "candidate_id",
                "candidate_hash",
                "calibrator_id",
                "alpha",
            )
        }
        checkpoint.update(
            {
                "final_dataset": {
                    "dataset_id": "6" * 64,
                    "parent_dataset_id": "4" * 64,
                    "task_count": 1,
                    "trajectory_count": 1,
                    "scored_point_count": 1,
                },
                "model_execution": {
                    "ensemble_policy_id": (
                        "development_three_seed_five_fold_mean_v1"
                    ),
                    "member_count": 1,
                    "member_projection_sha256": semantic_sha256([member_hash]),
                    "execution_mode": "strict_loaded_bundle_only",
                    "refit": False,
                    "calibration_application_count": 1,
                },
                "metrics": {"mae": 1.0},
                "task_metrics": [{"task": "pseudonym"}],
                "diagnostics": {"status": "point_cell"},
                "prediction_projection_sha256": "7" * 64,
                "cohort_projection_sha256": "8" * 64,
                "prediction_count": 1,
            }
        )
        _verify_checkpoint_selection_binding(
            checkpoint,
            selection=selection,
            cell=cell,
        )
        checkpoint["candidate_hash"] = "9" * 64
        with self.assertRaisesRegex(Stage4FinalError, "candidate_hash"):
            _verify_checkpoint_selection_binding(
                checkpoint,
                selection=selection,
                cell=cell,
            )

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
