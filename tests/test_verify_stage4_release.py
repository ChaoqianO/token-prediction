from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from scripts.prepare_stage4_selection import SOURCE_ARTIFACTS
from scripts.verify_stage4_release import (
    EVALUATION_CODE_POLICY_ID,
    EXPECTED_CELL_COUNT,
    EXPECTED_MEMBER_COUNT,
    EXPECTED_PREDICTION_COUNT,
    HISTORICAL_ADDED_EXPLICIT_PATHS,
    HISTORICAL_AMENDED_CODE_POLICY_ID,
    HISTORICAL_CLOSURE_POLICY_ID,
    HISTORICAL_OBSERVATION_POLICY_ID,
    MAX_RELEASE_JSON_BYTES,
    METADATA_AMENDMENTS,
    PROTECTED_RELEASE_TAGS,
    Stage4ReleaseError,
    _bundle_tree_projection,
    _evaluation_code_binding_at_commit,
    _historical_added_path_records,
    _historical_amended_code_binding_at_commit,
    _load_json,
    _selection_code_binding_at_commit,
    _validate_release_document,
    _verify_final_cell_bindings,
    _verify_ledger,
    _verify_metadata_amendment_documents,
    _verify_member_shape,
    _verify_selection_code_binding_from_git,
)
from token_prediction.final_ensemble import semantic_sha256


ROOT = Path(__file__).resolve().parents[1]
SHA256 = "0" * 64
SELECTION_ID = "1" * 64
SELECTION_COMMIT = "8b31e252d86e7288ce9d08a0829dc7f5b8bb5270"
FINAL_ARTIFACT_ID = "2" * 64
FINAL_RUN_ID = "synthetic-final-run"


def _release() -> dict[str, object]:
    return {
        "release_schema_version": 1,
        "stage_name": "stage4_final_holdout",
        "policy_id": "stage4_commit_bound_single_final_holdout_release_v1",
        "selection": {
            "lock_path": "configs/stage4_selection.json",
            "lock_sha256": SHA256,
            "tag": "stage4-final-selection-v1",
            "commit": SELECTION_COMMIT,
            "artifact": {
                "path": "workspace/stage4/selection/s4sel-synthetic",
                "artifact_id": "3" * 64,
                "run_id": "synthetic-selection-run",
                "selection_id": SELECTION_ID,
                "selection_payload_sha256": "4" * 64,
                "manifest_file_count": 16,
            },
        },
        "final_artifact": {
            "path": "workspace/stage4/final/s4final-synthetic",
            "artifact_id": FINAL_ARTIFACT_ID,
            "run_id": FINAL_RUN_ID,
            "selection_id": SELECTION_ID,
            "results_payload_sha256": "5" * 64,
            "manifest_file_count": 2,
        },
        "evaluation_code_binding": {
            "policy_id": EVALUATION_CODE_POLICY_ID,
            "git_commit": SELECTION_COMMIT,
            "code_tree_sha256": "6" * 64,
        },
        "historical_code_closure_amendments": [
            {
                "policy_id": HISTORICAL_CLOSURE_POLICY_ID,
                "selection_commit": SELECTION_COMMIT,
                "selection_code_commit": "e" * 40,
                "original_binding": {
                    "policy_id": EVALUATION_CODE_POLICY_ID,
                    "git_commit": SELECTION_COMMIT,
                    "code_tree_sha256": "6" * 64,
                    "path_count": 71,
                },
                "amended_binding": {
                    "policy_id": HISTORICAL_AMENDED_CODE_POLICY_ID,
                    "git_commit": SELECTION_COMMIT,
                    "code_tree_sha256": "8" * 64,
                    "path_count": 80,
                    "path_projection_sha256": "9" * 64,
                },
                "added_paths": [
                    {
                        "path": path,
                        "git_blob_oid": "a" * 40,
                        "sha256": "b" * 64,
                    }
                    for path in HISTORICAL_ADDED_EXPLICIT_PATHS
                ],
                "execution_observation": {
                    "policy_id": HISTORICAL_OBSERVATION_POLICY_ID,
                    "tracked_worktree_clean": True,
                    "observed_process_count": 1,
                    "recorded_pid": 4984,
                    "canonical_output_root": "workspace/stage4/final",
                    "canonical_checkpoint_root": (
                        "workspace/stage4/final-checkpoints"
                    ),
                    "canonical_final_artifact_count": 1,
                    "canonical_checkpoint_run_count": 1,
                    "checkpoint_cell_count": EXPECTED_CELL_COUNT,
                },
                "impact": (
                    "code_provenance_only_no_prediction_selection_calibration_"
                    "or_score_change"
                ),
                "residual_limitation": (
                    "historical_runner_lacked_intrinsic_exclusive_open_and_"
                    "complete_import_origin_binding"
                ),
            }
        ],
        "remote_controls": {
            "provider": "github",
            "repository": "ChaoqianO/token-prediction",
            "tag_ruleset_id": 19652329,
            "ruleset_name": "Protect immutable experiment tags",
            "final_release_tag": "stage4-final-release-v1",
            "target": "tag",
            "enforcement": "active",
            "bypass_actor_count": 0,
            "rules": ["update", "deletion"],
            "protected_tags": list(PROTECTED_RELEASE_TAGS),
        },
        "source_artifacts": [asdict(item) for item in SOURCE_ARTIFACTS],
        "metadata_amendments": copy.deepcopy(METADATA_AMENDMENTS),
        "protocol": {
            "run_policy_id": "stage4_single_open_resumable_final_holdout_v1",
            "selection_policy_id": "stage4_development_only_stability_guard_v1",
            "ensemble_policy_id": "development_three_seed_five_fold_mean_v1",
            "final_holdout_evaluation_count": 1,
            "final_holdout_prediction_count": EXPECTED_PREDICTION_COUNT,
            "selected_cell_count": EXPECTED_CELL_COUNT,
            "ensemble_member_count": EXPECTED_MEMBER_COUNT,
            "member_count_per_cell": 15,
            "refit_selected_learned_models": False,
            "calibration_application_count": 1,
            "verification_mode": "artifact_only_no_source_replay_v1",
        },
        "ledger": {
            "path": (
                f"workspace/stage4/final-checkpoints/{FINAL_RUN_ID}/ledger.json"
            ),
            "schema_version": 1,
            "status": "published",
            "completed_cell_count": EXPECTED_CELL_COUNT,
            "final_artifact_id": FINAL_ARTIFACT_ID,
        },
        "report": {
            "path": "docs/stage-4-final-report.md",
            "sha256": "7" * 64,
        },
    }


def _selection_and_final_cells() -> tuple[dict[str, object], dict[str, object]]:
    selected: list[dict[str, object]] = []
    final: list[dict[str, object]] = []
    for index in range(EXPECTED_CELL_COUNT):
        cell_id = hashlib.sha256(f"cell-{index}".encode()).hexdigest()
        member_hashes = [
            hashlib.sha256(f"member-{index}-{member}".encode()).hexdigest()
            for member in range(15)
        ]
        selected_cell = {
            "cell_id": cell_id,
            "source_name": "synthetic",
            "source_id": "synthetic-source",
            "condition_id": "condition:synthetic",
            "position": "call_pre",
            "target": "call_billable_total_tokens",
            "candidate_id": "lightgbm_history",
            "candidate_hash": "8" * 64,
            "calibrator_id": "task_max_conformal",
            "alpha": 0.1,
            "selected_artifact_key": "stage4_bagen_sokoban",
            "members": [
                {"member_sha256": member_hash} for member_hash in member_hashes
            ],
        }
        prediction_count = (
            EXPECTED_PREDICTION_COUNT - EXPECTED_CELL_COUNT + 1
            if index == 0
            else 1
        )
        final_cell = {
            key: selected_cell[key]
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
        final_cell.update(
            {
                "model_execution": {
                    "ensemble_policy_id": (
                        "development_three_seed_five_fold_mean_v1"
                    ),
                    "member_count": 15,
                    "member_projection_sha256": semantic_sha256(member_hashes),
                    "execution_mode": "strict_loaded_bundle_only",
                    "refit": False,
                    "calibration_application_count": 1,
                },
                "prediction_count": prediction_count,
            }
        )
        selected.append(selected_cell)
        final.append(final_cell)
    return {"cells": selected}, {"cells": final}


class Stage4ReleaseVerifierTests(unittest.TestCase):
    def test_release_schema_and_cardinalities_close(self) -> None:
        _validate_release_document(_release())

        extra = _release()
        extra["unexpected"] = True
        with self.assertRaisesRegex(Stage4ReleaseError, "keys"):
            _validate_release_document(extra)

        changed = copy.deepcopy(_release())
        changed["protocol"]["final_holdout_prediction_count"] -= 1
        with self.assertRaisesRegex(Stage4ReleaseError, "protocol"):
            _validate_release_document(changed)

        changed = copy.deepcopy(_release())
        changed["selection"]["artifact"]["manifest_file_count"] = 15
        with self.assertRaisesRegex(Stage4ReleaseError, "cardinality"):
            _validate_release_document(changed)

        changed = copy.deepcopy(_release())
        changed["metadata_amendments"][0]["impact"] = "prediction_changed"
        with self.assertRaisesRegex(Stage4ReleaseError, "amendments"):
            _validate_release_document(changed)

    def test_release_source_and_path_tampering_fail_closed(self) -> None:
        changed = copy.deepcopy(_release())
        changed["source_artifacts"][0]["source_commit"] = "f" * 40
        with self.assertRaisesRegex(Stage4ReleaseError, "source artifacts"):
            _validate_release_document(changed)

        changed = copy.deepcopy(_release())
        changed["report"]["path"] = "../stage-4-final-report.md"
        with self.assertRaisesRegex(Stage4ReleaseError, "safe relative path"):
            _validate_release_document(changed)

        changed = copy.deepcopy(_release())
        changed["ledger"]["path"] = "workspace/stage4/final/ledger.json"
        with self.assertRaisesRegex(Stage4ReleaseError, "ledger binding"):
            _validate_release_document(changed)

    def test_strict_json_rejects_duplicate_and_nonfinite_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "release.json"
            path.write_text('{"a":1,"a":2}\n', encoding="utf-8")
            with self.assertRaisesRegex(Stage4ReleaseError, "duplicate"):
                _load_json(
                    path,
                    maximum_bytes=MAX_RELEASE_JSON_BYTES,
                    description="synthetic release",
                )

            path.write_text('{"value":NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(Stage4ReleaseError, "non-finite"):
                _load_json(
                    path,
                    maximum_bytes=MAX_RELEASE_JSON_BYTES,
                    description="synthetic release",
                )

            path.write_text('{"value":1e999}\n', encoding="utf-8")
            with self.assertRaisesRegex(Stage4ReleaseError, "non-finite"):
                _load_json(
                    path,
                    maximum_bytes=MAX_RELEASE_JSON_BYTES,
                    description="synthetic release",
                )

    def test_member_checksum_and_shape_are_strict(self) -> None:
        cell = {
            "selected_artifact_key": "stage4_spend_aggregate",
            "target": "task_total_accounted_tokens",
        }
        member = {
            "origin": "selection_artifact",
            "bundle_kind": "empirical_json",
            "split_seed": 20260719,
            "split_plan_id": "9" * 64,
            "fold": 0,
            "state_path": "empirical/state.json",
            "state_sha256": "a" * 64,
        }
        member["member_sha256"] = semantic_sha256(member)
        parsed, kind = _verify_member_shape(member, cell=cell)
        self.assertEqual(parsed, member)
        self.assertEqual(kind, "empirical_json")

        changed = dict(member)
        changed["fold"] = 1
        with self.assertRaisesRegex(Stage4ReleaseError, "checksum"):
            _verify_member_shape(changed, cell=cell)

        changed = dict(member)
        changed["unexpected"] = True
        with self.assertRaisesRegex(Stage4ReleaseError, "keys"):
            _verify_member_shape(changed, cell=cell)

    def test_bundle_projection_detects_tree_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            (bundle / "manifest.json").write_text("{}", encoding="utf-8")
            digest, count = _bundle_tree_projection(bundle)
            self.assertEqual(count, 1)
            self.assertEqual(
                digest,
                semantic_sha256(
                    [
                        {
                            "path": "manifest.json",
                            "sha256": hashlib.sha256(b"{}").hexdigest(),
                        }
                    ]
                ),
            )
            (bundle / "extra.txt").write_text("tamper", encoding="utf-8")
            changed, changed_count = _bundle_tree_projection(bundle)
            self.assertNotEqual(changed, digest)
            self.assertEqual(changed_count, 2)

    def test_final_cell_binding_rejects_refit_and_cardinality_drift(self) -> None:
        selection, final = _selection_and_final_cells()
        _verify_final_cell_bindings(selection, final)

        changed = copy.deepcopy(final)
        changed["cells"][0]["model_execution"]["refit"] = True
        with self.assertRaisesRegex(Stage4ReleaseError, "execution protocol"):
            _verify_final_cell_bindings(selection, changed)

        changed = copy.deepcopy(final)
        changed["cells"][0]["prediction_count"] -= 1
        with self.assertRaisesRegex(Stage4ReleaseError, "prediction cardinality"):
            _verify_final_cell_bindings(selection, changed)

    def test_published_ledger_must_cover_all_final_cells(self) -> None:
        release = _release()
        _selection, final = _selection_and_final_cells()
        expected_cells = sorted(cell["cell_id"] for cell in final["cells"])
        ledger = {
            "ledger_schema_version": 1,
            "run_policy_id": "stage4_single_open_resumable_final_holdout_v1",
            "run_id": FINAL_RUN_ID,
            "selection_id": SELECTION_ID,
            "status": "published",
            "completed_cell_ids": expected_cells,
            "final_artifact_id": FINAL_ARTIFACT_ID,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / release["ledger"]["path"]
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(ledger, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            _verify_ledger(root, release, final)

            ledger["completed_cell_ids"] = expected_cells[:-1]
            path.write_text(
                json.dumps(ledger, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Stage4ReleaseError, "complete publication"):
                _verify_ledger(root, release, final)

    def test_metadata_amendment_closes_authoritative_and_artifact_counts(self) -> None:
        assignments = [
            {
                "cohort": "final_holdout" if index < 14 else "development",
                "task_pseudonym": hashlib.sha256(
                    f"task-{index}".encode()
                ).hexdigest(),
            }
            for index in range(16)
        ]
        source = {
            "development_protocol": {
                "permanent_holdout": {"assignments": assignments}
            }
        }
        final = {
            "datasets": [
                {"source_name": "bagen_swebench", "task_count": 13}
            ],
            "cells": [
                {
                    "source_name": "bagen_swebench",
                    "final_dataset": {"task_count": count},
                    "metrics": {"n_tasks": count},
                }
                for count in (12, 13, 14)
            ],
        }
        amendment = METADATA_AMENDMENTS[0]
        _verify_metadata_amendment_documents(
            source,
            final,
            amendment=amendment,
        )

        changed_source = copy.deepcopy(source)
        changed_source["development_protocol"]["permanent_holdout"][
            "assignments"
        ][0]["cohort"] = "development"
        with self.assertRaisesRegex(Stage4ReleaseError, "authoritative task count"):
            _verify_metadata_amendment_documents(
                changed_source,
                final,
                amendment=amendment,
            )

        changed_final = copy.deepcopy(final)
        changed_final["datasets"][0]["task_count"] = 14
        with self.assertRaisesRegex(Stage4ReleaseError, "artifact field"):
            _verify_metadata_amendment_documents(
                source,
                changed_final,
                amendment=amendment,
            )

        changed_final = copy.deepcopy(final)
        changed_final["cells"][-1]["final_dataset"]["task_count"] = 13
        changed_final["cells"][-1]["metrics"]["n_tasks"] = 13
        with self.assertRaisesRegex(Stage4ReleaseError, "cell coverage"):
            _verify_metadata_amendment_documents(
                source,
                changed_final,
                amendment=amendment,
            )

        changed_final = copy.deepcopy(final)
        changed_final["cells"][0]["metrics"]["n_tasks"] = 11
        with self.assertRaisesRegex(Stage4ReleaseError, "task counts disagree"):
            _verify_metadata_amendment_documents(
                source,
                changed_final,
                amendment=amendment,
            )

    def test_historical_evaluation_code_binding_matches_frozen_tag(self) -> None:
        commit = subprocess.run(
            [
                "git",
                "rev-parse",
                "--verify",
                "refs/tags/stage4-final-selection-v1^{commit}",
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        binding = _evaluation_code_binding_at_commit(ROOT, commit)
        self.assertEqual(binding["git_commit"], SELECTION_COMMIT)
        self.assertEqual(
            binding["code_tree_sha256"],
            "42e35f8a03414175865246c484aab829d76aaef5e34f94c5eca91da153e2b94c",
        )
        self.assertIn("scripts/run_stage4_final.py", binding["paths"])
        self.assertIn("configs/stage4_selection.json", binding["paths"])
        self.assertGreater(len(binding["paths"]), 50)

    def test_historical_amended_closure_binds_every_executed_loader(self) -> None:
        binding = _historical_amended_code_binding_at_commit(
            ROOT,
            SELECTION_COMMIT,
        )
        self.assertEqual(
            binding["code_tree_sha256"],
            "ed201c1ba0d0ba475c13bb003d683fbb73ffa2604ae294a643f443ae9fc6d6ca",
        )
        self.assertEqual(binding["path_count"], 80)
        self.assertEqual(
            binding["path_projection_sha256"],
            "a0919c24a594a156684f12a2b32356493aef85dcfb9f2457feab5473612e8d19",
        )
        records = _historical_added_path_records(ROOT, SELECTION_COMMIT)
        self.assertEqual(
            [record["path"] for record in records],
            list(HISTORICAL_ADDED_EXPLICIT_PATHS),
        )
        self.assertTrue(
            all(len(record["sha256"]) == 64 for record in records)
        )

    def test_selection_code_binding_is_recomputed_from_git_blobs(self) -> None:
        selection_lock = json.loads(
            (ROOT / "configs/stage4_selection.json").read_text(encoding="utf-8")
        )
        locked_artifact = selection_lock["selection_artifact"]
        binding = _selection_code_binding_at_commit(
            ROOT,
            locked_artifact["selection_code_commit"],
        )
        self.assertEqual(
            binding["code_tree_sha256"],
            locked_artifact["selection_code_tree_sha256"],
        )
        _verify_selection_code_binding_from_git(ROOT, binding)
        changed = copy.deepcopy(binding)
        changed["code_tree_sha256"] = "f" * 64
        with self.assertRaisesRegex(Stage4ReleaseError, "committed Git blobs"):
            _verify_selection_code_binding_from_git(ROOT, changed)


if __name__ == "__main__":
    unittest.main()
