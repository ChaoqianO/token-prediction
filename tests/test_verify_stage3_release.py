from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_stage3_experiments import SOURCE_NAMES, Stage3ExperimentError
from scripts.verify_stage3_release import (
    DEFAULT_RELEASE_LOCK,
    MAX_RELEASE_JSON_BYTES,
    _code_hash_at_commit,
    _load_json,
    _require_measured_latency,
    _require_shared_stage2_stage3_cohort,
    _resolve_artifact_source,
    _stage2_regression_neutral,
    _task_metric_multiset,
    _validate_release_document,
    _verify_fold_provenance,
    verify_stage3_release,
)
from token_prediction.contracts import SourceCapabilities, SourceDescriptor


ROOT = Path(__file__).resolve().parents[1]
SHA256 = "0" * 64


def _artifact(name: str, *, kind: str) -> dict[str, object]:
    is_gate = kind == "gate"
    identity = hashlib.sha256(name.encode("ascii")).hexdigest()
    return {
        "kind": kind,
        "path": f"workspace/stage3/runs/s3-{name.replace('_', '-')}",
        "source_id": SOURCE_NAMES[name],
        "source_descriptor_hash": SHA256,
        "run_id": identity[:24],
        "artifact_id": identity,
        "results_payload_sha256": SHA256,
        "base_dataset_id": SHA256,
        "derived_dataset_id": SHA256,
        "development_dataset_id": SHA256,
        "development_protocol_id": SHA256,
        "matrix_id": SHA256,
        "experiment_count": 0 if is_gate else 1,
        "candidate_seed_run_count": 0 if is_gate else 21,
        "lifecycle_candidate_seed_run_count": 0 if is_gate else 12,
        "exact_lifecycle_reload_fold_count": 0 if is_gate else 60,
        "reloadable_bundle_fold_count": 0 if is_gate else 90,
        "independently_loaded_bundle_count": 0 if is_gate else 90,
        "stage2_regression_candidate_seed_run_count": 0 if is_gate else 12,
        "manifest_file_count": 1 if is_gate else 100,
    }


def _release() -> dict[str, object]:
    artifacts = {
        "spend_aggregate": _artifact("spend_aggregate", kind="gate"),
        "bagen_sokoban": _artifact("bagen_sokoban", kind="experiment"),
        "bagen_swebench": _artifact("bagen_swebench", kind="experiment"),
        "spend_openhands": _artifact("spend_openhands", kind="experiment"),
    }
    return {
        "release_schema_version": 1,
        "stage_name": "stage3_development",
        "policy_id": "stage3_commit_bound_four_source_release_v1",
        "code_binding": {
            "artifact_git_commit": "1" * 40,
            "artifact_git_tag": "stage3-artifact-source-v1",
            "code_tree_sha256": SHA256,
        },
        "protocol": {
            "outer_folds": 5,
            "inner_folds": 5,
            "split_seeds": [20260719, 20260720, 20260721],
            "run_policy_id": "stage3_source_three_seed_cuda_resumable_nested_cv_v2",
            "checkpoint_policy_id": "atomic_candidate_and_every_neural_epoch_v1",
            "checkpoint_interval_epochs": 1,
            "neural_training_device": "cuda",
            "neural_inference_device": "cpu",
            "calibrator_id": "task_max_conformal",
            "alpha": 0.1,
            "budget_thresholds": [16384, 32768, 65536, 131072],
            "progress_checkpoints": [0.25, 0.5, 0.75],
            "final_holdout_evaluated": False,
            "final_holdout_prediction_count": 0,
        },
        "stage2_regression": {
            "release_lock_path": "configs/stage2_release.json",
            "release_lock_sha256": SHA256,
            "exact_candidate_ids": [
                "cross_position_deduct",
                "empirical",
                "lightgbm_history",
            ],
            "runtime_scoped_candidate_ids": ["mlp_history"],
            "normalization_policy_id": ("exact_non_neural_and_stage3_mlp_contract_v2"),
            "candidate_seed_run_count": 36,
        },
        "artifacts": artifacts,
        "totals": {
            "artifact_count": 4,
            "experiment_artifact_count": 3,
            "gate_artifact_count": 1,
            "experiment_count": 3,
            "candidate_seed_run_count": 63,
            "lifecycle_candidate_seed_run_count": 36,
            "exact_lifecycle_reload_fold_count": 180,
            "reloadable_bundle_fold_count": 270,
            "independently_loaded_bundle_count": 270,
            "stage2_regression_candidate_seed_run_count": 36,
            "manifest_file_count": 301,
        },
        "report": {
            "path": "docs/stage-3-report.md",
            "sha256": SHA256,
        },
    }


class Stage3ReleaseVerifierTests(unittest.TestCase):
    def test_repository_release_controls_and_source_tree_close(self) -> None:
        result = verify_stage3_release(
            ROOT,
            tracked_only=True,
            require_git_clean=False,
        )
        self.assertEqual(result.locked_artifact_count, 4)
        self.assertEqual(result.verified_artifact_count, 0)
        self.assertEqual(result.stage2_regression_candidate_seed_run_count, 0)
        self.assertFalse(result.final_holdout_evaluated)
        self.assertEqual(
            result.code_tree_sha256,
            "18cb3fdc20df475f556683d3b1db7f83f549aa94442a39fe59225a084fdfba26",
        )

        release = _load_json(
            ROOT / DEFAULT_RELEASE_LOCK,
            maximum_bytes=MAX_RELEASE_JSON_BYTES,
            description="Stage 3 release test lock",
        )
        code = release["code_binding"]
        code_hash, paths = _code_hash_at_commit(ROOT, code["artifact_git_commit"])
        self.assertEqual(code_hash, code["code_tree_sha256"])
        self.assertGreater(len(paths), 50)

        status, reproduced_paths = _resolve_artifact_source(
            ROOT,
            code["artifact_git_commit"],
            code["artifact_git_tag"],
            code["code_tree_sha256"],
        )
        self.assertEqual(
            status,
            "artifact_tag_commit_and_source_tree_verified",
        )
        self.assertEqual(reproduced_paths, paths)

    def test_frozen_release_shape_closes(self) -> None:
        _validate_release_document(_release())

    def test_unknown_fields_and_total_drift_fail_closed(self) -> None:
        extra = _release()
        extra["unexpected"] = True
        with self.assertRaisesRegex(Stage3ExperimentError, "keys"):
            _validate_release_document(extra)

        changed = copy.deepcopy(_release())
        changed["totals"]["reloadable_bundle_fold_count"] = 269
        with self.assertRaisesRegex(Stage3ExperimentError, "totals"):
            _validate_release_document(changed)

    def test_final_holdout_claim_fails_closed(self) -> None:
        changed = copy.deepcopy(_release())
        changed["protocol"]["final_holdout_evaluated"] = True
        with self.assertRaisesRegex(Stage3ExperimentError, "protocol"):
            _validate_release_document(changed)

    def test_gate_and_lifecycle_cardinalities_fail_closed(self) -> None:
        gate_changed = copy.deepcopy(_release())
        gate_changed["artifacts"]["spend_aggregate"]["experiment_count"] = 1
        gate_changed["totals"]["experiment_count"] = 4
        with self.assertRaisesRegex(Stage3ExperimentError, "gate artifact"):
            _validate_release_document(gate_changed)

        lifecycle_changed = copy.deepcopy(_release())
        lifecycle_changed["artifacts"]["bagen_sokoban"]["exact_lifecycle_reload_fold_count"] = 59
        lifecycle_changed["totals"]["exact_lifecycle_reload_fold_count"] = 179
        with self.assertRaisesRegex(Stage3ExperimentError, "cardinalities"):
            _validate_release_document(lifecycle_changed)

        regression_changed = copy.deepcopy(_release())
        regression_changed["stage2_regression"]["candidate_seed_run_count"] = 35
        with self.assertRaisesRegex(Stage3ExperimentError, "regression binding count"):
            _validate_release_document(regression_changed)

    def test_source_identity_reuse_and_missing_latency_fail_closed(self) -> None:
        wrong_source = copy.deepcopy(_release())
        wrong_source["artifacts"]["bagen_sokoban"]["source_id"] = SOURCE_NAMES["bagen_swebench"]
        with self.assertRaisesRegex(Stage3ExperimentError, "source id"):
            _validate_release_document(wrong_source)

        reused_run = copy.deepcopy(_release())
        reused_run["artifacts"]["bagen_sokoban"]["run_id"] = reused_run["artifacts"][
            "bagen_swebench"
        ]["run_id"]
        with self.assertRaisesRegex(Stage3ExperimentError, "not unique"):
            _validate_release_document(reused_run)

        with self.assertRaisesRegex(Stage3ExperimentError, "measured prediction latency"):
            _require_measured_latency(
                {"latency_p50_ms": 0.0, "latency_p95_ms": 0.0},
                description="synthetic candidate",
            )
        _require_measured_latency(
            {"latency_p50_ms": 0.01, "latency_p95_ms": 0.02},
            description="synthetic candidate",
        )

    def test_fold_provenance_is_bound_to_release_scope(self) -> None:
        source_id = SOURCE_NAMES["bagen_sokoban"]
        descriptor = SourceDescriptor(
            source_id=source_id,
            revision="synthetic-revision",
            manifest_path="workspace/synthetic/manifest.json",
            manifest_sha256="2" * 64,
            capabilities=SourceCapabilities(
                source_id=source_id,
                observables=frozenset(),
            ),
        )
        experiment = {"condition_id": "condition:synthetic"}
        candidate = {
            "candidate_id": "gru_residual",
            "candidate_hash": "3" * 64,
            "estimator_id": "gru_residual",
        }
        seed_result = {"split_plan_id": "4" * 64}
        provenance = {
            "candidate_id": candidate["candidate_id"],
            "candidate_hash": candidate["candidate_hash"],
            "dataset_id": "5" * 64,
            "condition_id": experiment["condition_id"],
            "source_descriptor": descriptor.to_dict(),
            "source_descriptor_hash": descriptor.descriptor_hash,
            "capability_contract_hash": descriptor.capabilities.contract_hash,
            "input_contract_hash": "6" * 64,
            "code_hash": "7" * 64,
            "split_plan_id": seed_result["split_plan_id"],
            "position": "task_update",
            "target": "task_provider_accounted_remaining_tokens",
            "calibrator_id": "task_max_conformal",
            "interval_alpha": 0.1,
            "outer_fold": 2,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "provenance.json").write_text(
                json.dumps(provenance),
                encoding="utf-8",
            )
            observed = _verify_fold_provenance(
                root,
                experiment=experiment,
                candidate=candidate,
                seed_result=seed_result,
                fold=2,
                is_lifecycle=True,
                expected_source_id=source_id,
                expected_source_descriptor_hash=descriptor.descriptor_hash,
                expected_capability_contract_hash=descriptor.capabilities.contract_hash,
                expected_code_hash="7" * 64,
                expected_dataset_id="5" * 64,
                expected_input_contract_hash="6" * 64,
            )
            self.assertEqual(observed, provenance)

            provenance["code_hash"] = "8" * 64
            (root / "provenance.json").write_text(
                json.dumps(provenance),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Stage3ExperimentError, "release scope"):
                _verify_fold_provenance(
                    root,
                    experiment=experiment,
                    candidate=candidate,
                    seed_result=seed_result,
                    fold=2,
                    is_lifecycle=True,
                    expected_source_id=source_id,
                    expected_source_descriptor_hash=descriptor.descriptor_hash,
                    expected_capability_contract_hash=(descriptor.capabilities.contract_hash),
                    expected_code_hash="7" * 64,
                    expected_dataset_id="5" * 64,
                    expected_input_contract_hash="6" * 64,
                )

            lightgbm_candidate = {
                "candidate_id": "lightgbm_history",
                "candidate_hash": "9" * 64,
                "estimator_id": "lightgbm_quantile",
            }
            provenance["candidate_id"] = lightgbm_candidate["candidate_id"]
            provenance["candidate_hash"] = lightgbm_candidate["candidate_hash"]
            provenance["code_hash"] = "7" * 64
            del provenance["source_descriptor"]
            provenance["fold"] = provenance.pop("outer_fold")
            (root / "provenance.json").write_text(
                json.dumps(provenance),
                encoding="utf-8",
            )
            _verify_fold_provenance(
                root,
                experiment=experiment,
                candidate=lightgbm_candidate,
                seed_result=seed_result,
                fold=2,
                is_lifecycle=False,
                expected_source_id=source_id,
                expected_source_descriptor_hash=descriptor.descriptor_hash,
                expected_capability_contract_hash=descriptor.capabilities.contract_hash,
                expected_code_hash="7" * 64,
                expected_dataset_id="5" * 64,
                expected_input_contract_hash="6" * 64,
            )
            with self.assertRaisesRegex(Stage3ExperimentError, "lacks a source descriptor"):
                _verify_fold_provenance(
                    root,
                    experiment=experiment,
                    candidate={**lightgbm_candidate, "estimator_id": "gru_residual"},
                    seed_result=seed_result,
                    fold=2,
                    is_lifecycle=False,
                    expected_source_id=source_id,
                    expected_source_descriptor_hash=descriptor.descriptor_hash,
                    expected_capability_contract_hash=(descriptor.capabilities.contract_hash),
                    expected_code_hash="7" * 64,
                    expected_dataset_id="5" * 64,
                    expected_input_contract_hash="6" * 64,
                )

    def test_stage2_regression_normalizes_only_timing_and_task_salt(self) -> None:
        first = {
            "latency_p50_ms": 0.01,
            "latency_p95_ms": 0.02,
            "weighted_mae": 12.5,
        }
        second = {
            "latency_p50_ms": 9.0,
            "latency_p95_ms": 10.0,
            "weighted_mae": 12.5,
        }
        self.assertEqual(
            _stage2_regression_neutral(first),
            _stage2_regression_neutral(second),
        )
        self.assertNotEqual(
            _stage2_regression_neutral(first),
            _stage2_regression_neutral({**second, "weighted_mae": 12.6}),
        )

        stage2_tasks = [
            {"task_pseudonym": "a" * 64, "weighted_mae": 1.0},
            {"task_pseudonym": "b" * 64, "weighted_mae": 2.0},
        ]
        stage3_tasks = [
            {"task_pseudonym": "d" * 64, "weighted_mae": 2.0},
            {"task_pseudonym": "c" * 64, "weighted_mae": 1.0},
        ]
        self.assertEqual(
            _task_metric_multiset(stage2_tasks, description="Stage 2 fixture"),
            _task_metric_multiset(stage3_tasks, description="Stage 3 fixture"),
        )

    def test_stage_specific_cohort_projection_namespaces_are_explicit(self) -> None:
        shared = {
            "split_plan_id": "1" * 64,
            "comparability_key": ["same", "cohort"],
            "prediction_count": 12,
        }
        stage2 = {
            **shared,
            "cohort_projection_id": "stage2_prediction_cohort_projection_v1",
            "cohort_projection_sha256": "2" * 64,
        }
        stage3 = {
            **shared,
            "cohort_projection_id": "stage3_prediction_cohort_projection_v1",
            "cohort_projection_sha256": "3" * 64,
        }
        _require_shared_stage2_stage3_cohort(
            stage2,
            stage3,
            description="synthetic regression",
        )

        changed = {**stage3, "prediction_count": 13}
        with self.assertRaisesRegex(Stage3ExperimentError, "cohort differs"):
            _require_shared_stage2_stage3_cohort(
                stage2,
                changed,
                description="synthetic regression",
            )

        wrong_namespace = {**stage3, "cohort_projection_id": "unexpected"}
        with self.assertRaisesRegex(Stage3ExperimentError, "namespace"):
            _require_shared_stage2_stage3_cohort(
                stage2,
                wrong_namespace,
                description="synthetic regression",
            )


if __name__ == "__main__":
    unittest.main()
