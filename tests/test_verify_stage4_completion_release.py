from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.verify_stage4_completion_release import (
    DIAGNOSTICS_POLICY_ID,
    EXPECTED_CANDIDATE_SEED_RUN_COUNT,
    EXPECTED_EXPERIMENT_COUNT,
    EXPECTED_RELOADABLE_BUNDLE_FOLD_COUNT,
    MAX_RELEASE_JSON_BYTES,
    PARENT_FINAL_ARTIFACT_ID,
    SOURCE_CODE_POLICY_ID,
    SOURCE_TAG,
    Stage4CompletionReleaseError,
    _code_binding_at_commit,
    _load_json,
    _validate_release_document,
    _verify_progress_diagnostics,
    _verify_result_coverage,
    _verify_run_dispersion,
)
from token_prediction.evaluation.stratification import (
    PROGRESS_STRATIFICATION_ID,
    RUN_DISPERSION_EXTENSION_ID,
    RUN_VARIANCE_ID,
)
from token_prediction.stage2_matrix import SPEND_AGGREGATE_SOURCE_ID
from token_prediction.stage4_matrix import (
    FROZEN_STAGE4_SOURCE_CONDITIONS,
    STAGE4_MATRIX_POLICY_ID,
    STAGE4_MATRIX_SCHEMA_VERSION,
    STAGE4_MISSING_MASK_INVARIANT_ID,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "c1ac2484f44ed65705cdd00eba7b70a739a3ac0b"
SHA256 = "0" * 64


def _release() -> dict[str, object]:
    scopes = (
        (
            "spend_aggregate",
            "spend_your_money_aggregate_v1",
            3,
            15,
        ),
        ("bagen_sokoban", "bagen_sokoban_dialogues_v1", 7, 66),
        ("bagen_swebench", "bagen_swebench_traj_v2", 35, 330),
        ("spend_openhands", "openhands_archive_trajectory_v3", 7, 66),
    )
    artifacts = []
    for index, (name, source_id, experiments, candidate_seeds) in enumerate(
        scopes
    ):
        run_id = hashlib.sha256(name.encode()).hexdigest()[:24]
        artifacts.append(
            {
                "source_name": name,
                "source_id": source_id,
                "path": f"workspace/stage4/runs/s4-{run_id[:20]}",
                "artifact_id": str(index + 1) * 64,
                "run_id": run_id,
                "results_payload_sha256": chr(ord("a") + index) * 64,
                "manifest_sha256": str(5 + index) * 64,
                "matrix_id": chr(ord("1") + index) * 64,
                "experiment_count": experiments,
                "candidate_seed_run_count": candidate_seeds,
                "manifest_file_count": 10 + index,
            }
        )
    diagnostics_coverage = {
        "bound_source_artifact_count": 4,
        "lifecycle_source_count": 3,
        "lifecycle_condition_count": 7,
        "lifecycle_candidate_count": 2,
        "lifecycle_candidate_cell_count": 14,
        "lifecycle_candidate_seed_count": 42,
        "lifecycle_bundle_count": 210,
        "replayed_run_count": 420,
        "scored_run_count": 400,
        "scored_boundary_count": 4_000,
        "unscored_context_boundary_count": 200,
    }
    return {
        "release_schema_version": 1,
        "stage_name": "stage4_development_completion_supplement",
        "policy_id": "stage4_development_only_completion_release_v1",
        "source_binding": {
            "policy_id": SOURCE_CODE_POLICY_ID,
            "git_commit": SOURCE_COMMIT,
            "code_tree_sha256": SHA256,
            "source_tag": SOURCE_TAG,
        },
        "parent_final_release": {
            "lock_path": "configs/stage4_release.json",
            "lock_sha256": SHA256,
            "final_release_tag": "stage4-final-release-v1",
            "final_artifact_id": PARENT_FINAL_ARTIFACT_ID,
            "final_holdout_evaluation_count": 1,
            "final_holdout_prediction_count": 86_335,
        },
        "artifacts": artifacts,
        "diagnostics_artifact": {
            "path": (
                "workspace/stage4/completion_diagnostics/"
                "s4diag-0123456789abcdef0123"
            ),
            "artifact_id": "9" * 64,
            "manifest_sha256": "8" * 64,
            "results_payload_sha256": "7" * 64,
            "training_source_commit": SOURCE_COMMIT,
            "diagnostics_code_binding": {
                "git_commit": "1" * 40,
                "code_tree_sha256": "2" * 64,
                "code_paths": [
                    "scripts/run_stage4_completion_diagnostics.py",
                    "src/token_prediction/evaluation/stratification.py",
                    "src/token_prediction/lifecycle.py",
                    "src/token_prediction/lifecycle_bundle.py",
                ],
                "source_tag": "stage4-completion-diagnostics-source-v1",
            },
            "source_artifact_ids": {
                item["source_name"]: item["artifact_id"] for item in artifacts
            },
            "coverage": diagnostics_coverage,
        },
        "protocol": {
            "development_only": True,
            "artifact_count": 4,
            "experiment_count": EXPECTED_EXPERIMENT_COUNT,
            "candidate_seed_run_count": EXPECTED_CANDIDATE_SEED_RUN_COUNT,
            "reloadable_bundle_fold_count": (
                EXPECTED_RELOADABLE_BUNDLE_FOLD_COUNT
            ),
            "call_pre_mlp_cell_count": 21,
            "call_pre_mlp_bundle_fold_count": 315,
            "seed_policy_cell_count": 7,
            "seed_policy_bundle_fold_count": 210,
            "diagnostics_artifact_count": 1,
            "diagnostics_bound_source_artifact_count": 4,
            "diagnostics_lifecycle_source_count": 3,
            "diagnostics_lifecycle_condition_count": 7,
            "diagnostics_candidate_count": 2,
            "diagnostics_candidate_cell_count": 14,
            "diagnostics_candidate_seed_count": 42,
            "diagnostics_bundle_count": 210,
            "split_seeds": [20260719, 20260720, 20260721],
            "outer_folds": 5,
            "inner_folds": 5,
            "final_holdout_evaluated": False,
            "final_holdout_prediction_count": 0,
            "final_holdout_target_values_used_for_fit_calibration_scoring": (
                False
            ),
            "final_holdout_selection_claim": "none",
        },
        "report": {
            "path": "docs/stage-4-completion-supplement.md",
            "sha256": "6" * 64,
        },
    }


def _metrics() -> dict[str, object]:
    return {
        "interval_diagnostics_id": "weighted_interval_tail_and_reserve_v1",
        "interval_below_truth_rate": 0.1,
        "interval_above_truth_rate": 0.1,
        "target_exceeds_upper_rate": 0.1,
        "mean_extra_reserved_tokens": 1.0,
        "raw_interval_below_truth_rate": 0.1,
        "raw_interval_above_truth_rate": 0.1,
        "raw_target_exceeds_upper_rate": 0.1,
        "raw_mean_extra_reserved_tokens": 1.0,
    }


def _candidate(
    candidate_id: str,
    estimator_id: str,
    *,
    index: int,
) -> dict[str, object]:
    reloadable = estimator_id != "empirical_quantile"
    graph = {
        "initializer_estimator_id": "none",
        "updater_estimator_id": estimator_id,
        "lifecycle_schema_id": "point_cell_v1",
        "seed_policy_id": "none",
        "inner_split_policy_id": "none",
    }
    seed_results = []
    for seed in (20260719, 20260720, 20260721):
        seed_results.append(
            {
                "split_seed": seed,
                "split_plan_id": hashlib.sha256(
                    f"{candidate_id}-{seed}".encode()
                ).hexdigest(),
                "metrics": _metrics(),
                "fold_metrics": {
                    str(fold): _metrics() for fold in range(5)
                },
                "reloadable_bundle_folds": list(range(5)) if reloadable else [],
                "fold_artifact_count": 5 if reloadable else 0,
                "bundle_reload_parity": (
                    {"fold_count": 5, "status": "exact_during_execution"}
                    if reloadable
                    else {
                        "fold_count": 0,
                        "status": "not_applicable_stateless",
                    }
                ),
            }
        )
    return {
        "artifact_key": f"c_{index:016x}",
        "candidate_id": candidate_id,
        "candidate_hash": hashlib.sha256(candidate_id.encode()).hexdigest(),
        "estimator_id": estimator_id,
        "feature_set_id": f"feature-{candidate_id}",
        "feature_set_hash": hashlib.sha256(
            f"feature-{candidate_id}".encode()
        ).hexdigest(),
        "candidate_graph": graph,
        "seed_results": seed_results,
    }


def _aggregate_results() -> dict[str, object]:
    condition_id = next(
        iter(FROZEN_STAGE4_SOURCE_CONDITIONS[SPEND_AGGREGATE_SOURCE_ID])
    )
    specifications = (
        (
            "calibration-none",
            "none",
            (("lightgbm_structured", "lightgbm_quantile"),),
        ),
        (
            "calibration-task-max",
            "task_max_conformal",
            (("lightgbm_structured", "lightgbm_quantile"),),
        ),
        (
            "method",
            "task_max_conformal",
            (
                ("empirical", "empirical_quantile"),
                ("task_chars_length", "lightgbm_quantile"),
                ("lightgbm_structured", "lightgbm_quantile"),
            ),
        ),
    )
    experiments = []
    plans = []
    candidate_index = 0
    for experiment_index, (suffix, calibrator, candidate_specs) in enumerate(
        specifications
    ):
        candidates = []
        for candidate_id, estimator_id in candidate_specs:
            candidates.append(
                _candidate(
                    candidate_id,
                    estimator_id,
                    index=candidate_index,
                )
            )
            candidate_index += 1
        experiment_id = f"synthetic-{suffix}"
        role = "ablation" if calibrator == "none" else "primary"
        axis = "calibration" if role == "ablation" else None
        reference = (
            "synthetic-calibration-task-max" if role == "ablation" else None
        )
        allowed = ["calibrator_id"] if role == "ablation" else []
        experiment = {
            "artifact_key": f"e_{experiment_index:016x}",
            "experiment_id": experiment_id,
            "position": "task_launch",
            "target": "task_total_accounted_tokens",
            "condition_id": condition_id,
            "alpha": 0.1,
            "calibrator_id": calibrator,
            "plan_role": role,
            "axis": axis,
            "reference_experiment_id": reference,
            "allowed_config_paths": allowed,
            "candidates": candidates,
        }
        experiments.append(experiment)
        plans.append(
            {
                "spec": {
                    "experiment_id": experiment_id,
                    "position": experiment["position"],
                    "target": experiment["target"],
                    "condition_id": condition_id,
                    "alpha": 0.1,
                    "calibrator_id": calibrator,
                    "candidates": [
                        {
                            "candidate_id": item["candidate_id"],
                            "candidate_hash": item["candidate_hash"],
                            "estimator_id": item["estimator_id"],
                            "feature_set_id": item["feature_set_id"],
                            "feature_set_hash": item["feature_set_hash"],
                            "graph": item["candidate_graph"],
                        }
                        for item in candidates
                    ],
                },
                "role": role,
                "axis": axis,
                "reference_experiment_id": reference,
                "allowed_config_paths": allowed,
            }
        )
    return {
        "source": {"source_id": SPEND_AGGREGATE_SOURCE_ID},
        "summary": {},
        "matrix": {
            "schema_version": STAGE4_MATRIX_SCHEMA_VERSION,
            "policy_id": STAGE4_MATRIX_POLICY_ID,
            "source_id": SPEND_AGGREGATE_SOURCE_ID,
            "safety_invariants": [
                {
                    "invariant_id": STAGE4_MISSING_MASK_INVARIANT_ID,
                    "estimator_ids": ["gru_residual", "independent_mlp"],
                    "required_behavior": (
                        "neural_inputs_keep_explicit_missing_indicators_and_"
                        "history_ablations_keep_missing_usage_attempts"
                    ),
                    "prohibited_ablation": (
                        "disable_or_remove_missing_telemetry_masks"
                    ),
                    "violation_action": "fail_closed",
                }
            ],
            "plans": plans,
        },
        "experiments": experiments,
    }


class Stage4CompletionReleaseVerifierTests(unittest.TestCase):
    def test_release_schema_closes_and_rejects_drift(self) -> None:
        _validate_release_document(_release())

        changed = copy.deepcopy(_release())
        changed["protocol"]["reloadable_bundle_fold_count"] -= 1
        with self.assertRaisesRegex(
            Stage4CompletionReleaseError,
            "protocol differs",
        ):
            _validate_release_document(changed)

        changed = copy.deepcopy(_release())
        changed["diagnostics_artifact"]["coverage"][
            "lifecycle_bundle_count"
        ] -= 1
        with self.assertRaisesRegex(
            Stage4CompletionReleaseError,
            "fixed coverage",
        ):
            _validate_release_document(changed)

        changed = copy.deepcopy(_release())
        changed["diagnostics_artifact"]["training_source_commit"] = "f" * 40
        with self.assertRaisesRegex(
            Stage4CompletionReleaseError,
            "training source commit",
        ):
            _validate_release_document(changed)

        changed = copy.deepcopy(_release())
        changed["diagnostics_artifact"]["diagnostics_code_binding"][
            "code_paths"
        ] = list(
            reversed(
                changed["diagnostics_artifact"]["diagnostics_code_binding"][
                    "code_paths"
                ]
            )
        )
        with self.assertRaisesRegex(
            Stage4CompletionReleaseError,
            "code paths",
        ):
            _validate_release_document(changed)

        changed = copy.deepcopy(_release())
        changed["parent_final_release"]["final_holdout_evaluation_count"] = 2
        with self.assertRaisesRegex(
            Stage4CompletionReleaseError,
            "parent final release",
        ):
            _validate_release_document(changed)

        changed = copy.deepcopy(_release())
        changed["artifacts"][0]["manifest_sha256"] = "../manifest.json"
        with self.assertRaisesRegex(
            Stage4CompletionReleaseError,
            "lowercase SHA-256",
        ):
            _validate_release_document(changed)

    def test_strict_json_rejects_duplicate_and_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "release.json"
            path.write_text('{"a":1,"a":2}\n', encoding="utf-8")
            with self.assertRaisesRegex(
                Stage4CompletionReleaseError,
                "duplicate",
            ):
                _load_json(
                    path,
                    maximum_bytes=MAX_RELEASE_JSON_BYTES,
                    description="synthetic completion release",
                )

            path.write_text('{"value":NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(
                Stage4CompletionReleaseError,
                "non-finite",
            ):
                _load_json(
                    path,
                    maximum_bytes=MAX_RELEASE_JSON_BYTES,
                    description="synthetic completion release",
                )

    def test_source_commit_reproduces_frozen_code_tree(self) -> None:
        binding = _code_binding_at_commit(ROOT, SOURCE_COMMIT)
        self.assertEqual(binding["git_commit"], SOURCE_COMMIT)
        self.assertEqual(
            binding["code_tree_sha256"],
            "6418545afa08a39df1797486e4c845063c2de13b29f20c81500933fad2201757",
        )
        self.assertIn(
            "scripts/run_stage4_experiments.py",
            binding["code_paths"],
        )

    def test_aggregate_matrix_recomputes_candidate_and_bundle_counts(
        self,
    ) -> None:
        results = _aggregate_results()
        coverage = _verify_result_coverage(
            results,
            source_name="spend_aggregate",
        )
        self.assertEqual(coverage.experiment_count, 3)
        self.assertEqual(coverage.candidate_seed_run_count, 15)
        self.assertEqual(coverage.reloadable_bundle_fold_count, 60)

        changed = copy.deepcopy(results)
        changed["experiments"][0]["candidates"][0]["seed_results"][0][
            "reloadable_bundle_folds"
        ] = [0, 1, 2, 3]
        with self.assertRaisesRegex(
            Stage4CompletionReleaseError,
            "reloadable fold",
        ):
            _verify_result_coverage(
                changed,
                source_name="spend_aggregate",
            )

    def test_progress_and_run_dispersion_policies_are_strict(self) -> None:
        progress = {
            "stratification_id": PROGRESS_STRATIFICATION_ID,
            "selection_policy": (
                "first_boundary_at_or_after_sequence_fraction_v1"
            ),
            "strata": {
                key: {
                    "checkpoint": checkpoint,
                    "n_sequences": 2,
                    "n_selected_boundaries": 2,
                    "n_scored": 1,
                    "n_unscored": 1,
                    "metrics": None,
                }
                for key, checkpoint in (
                    ("p25", 0.25),
                    ("p50", 0.5),
                    ("p75", 0.75),
                )
            },
        }
        self.assertEqual(_verify_progress_diagnostics(progress), 2)

        dispersion = {
            "run_variance_id": RUN_VARIANCE_ID,
            "run_dispersion_extension_id": RUN_DISPERSION_EXTENSION_ID,
            "n_tasks": 2,
            "n_scored_runs": 2,
            "n_repeated_tasks": 1,
            "status": "estimable",
            "mean_within_task_run_mae_variance": 1.0,
            "median_within_task_run_mae_variance": 1.0,
            "max_within_task_run_mae_variance": 1.0,
            "mean_within_task_run_mae_iqr": 1.0,
            "median_within_task_run_mae_iqr": 1.0,
            "max_within_task_run_mae_iqr": 1.0,
            "mean_within_task_run_mae_max_minus_min": 2.0,
            "median_within_task_run_mae_max_minus_min": 2.0,
            "max_within_task_run_mae_max_minus_min": 2.0,
        }
        self.assertEqual(_verify_run_dispersion(dispersion), 2)
        changed = dict(dispersion)
        changed["run_dispersion_extension_id"] = "legacy"
        with self.assertRaisesRegex(
            Stage4CompletionReleaseError,
            "dispersion policy",
        ):
            _verify_run_dispersion(changed)

    def test_diagnostics_policy_constant_is_frozen(self) -> None:
        self.assertEqual(
            DIAGNOSTICS_POLICY_ID,
            "stage4_completion_development_lifecycle_replay_v1",
        )


if __name__ == "__main__":
    unittest.main()
