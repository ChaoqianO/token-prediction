from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.verify_stage4_completion_release import (
    COMPLETION_RELEASE_TAG,
    DIAGNOSTICS_POLICY_ID,
    EXPECTED_CANDIDATE_SEED_RUN_COUNT,
    EXPECTED_EXPERIMENT_COUNT,
    EXPECTED_RELOADABLE_BUNDLE_FOLD_COUNT,
    MAX_RELEASE_JSON_BYTES,
    PARENT_FINAL_ARTIFACT_ID,
    RELEASE_CONTROL_PATHS,
    SOURCE_CODE_POLICY_ID,
    SOURCE_TAG,
    Stage4CompletionReleaseError,
    _code_binding_at_commit,
    _checkpoint_expectation,
    _expected_artifact_key,
    _load_json,
    _require_exact_artifact_payloads,
    _semantic_sha256,
    _training_run_semantic_sha256,
    _validate_release_document,
    _verify_shared_development_task_projection,
    _verify_release_control_tag,
    _verify_result_coverage,
    _verify_tracked_bindings,
)
from token_prediction.evaluation import METRIC_SUITE_ID
from token_prediction.features import NO_FEATURES
from token_prediction.stage2_matrix import (
    SPEND_AGGREGATE_SOURCE_ID,
    SPEND_AGGREGATE_STRUCTURED_FEATURES,
    SPEND_AGGREGATE_TASK_CHARS,
)
from token_prediction.stage4_matrix import (
    FROZEN_STAGE4_SOURCE_CONDITIONS,
    STAGE4_MATRIX_POLICY_ID,
    STAGE4_MATRIX_SCHEMA_VERSION,
    STAGE4_MISSING_MASK_INVARIANT_ID,
    STAGE4_MIN_DEVELOPMENT_TASKS,
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
        "checkpoint_verified_candidate_seed_count": 42,
        "lifecycle_replayed_candidate_seed_count": 0,
        "lifecycle_metrics_unavailable_candidate_seed_count": 42,
    }
    return {
        "release_schema_version": 2,
        "stage_name": "stage4_development_completion_supplement",
        "policy_id": "stage4_development_only_completion_release_v1",
        "release_control": {
            "release_tag": COMPLETION_RELEASE_TAG,
        },
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
            "diagnostics_checkpoint_verified_candidate_seed_count": 42,
            "diagnostics_lifecycle_replayed_candidate_seed_count": 0,
            "diagnostics_lifecycle_metrics_unavailable_candidate_seed_count": (
                42
            ),
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


def _release_control_payloads() -> dict[str, bytes]:
    return {
        relative: f"frozen:{relative}\n".encode()
        for relative in RELEASE_CONTROL_PATHS
    }


def _write_release_controls(
    root: Path,
    payloads: dict[str, bytes],
) -> None:
    for relative, payload in payloads.items():
        path = root / Path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def _release_control_git(
    payloads: dict[str, bytes],
    *,
    commit: str = "a" * 40,
):
    def fake_git(
        _root: Path,
        *arguments: str,
        maximum_bytes: int = 64 * 1024 * 1024,
    ) -> bytes:
        del maximum_bytes
        if arguments == (
            "rev-parse",
            "--verify",
            f"refs/tags/{COMPLETION_RELEASE_TAG}^{{commit}}",
        ):
            return f"{commit}\n".encode()
        if len(arguments) == 2 and arguments[0] == "show":
            tagged_commit, relative = arguments[1].split(":", 1)
            if tagged_commit != commit:
                raise AssertionError("unexpected release-control commit")
            return payloads[relative]
        raise AssertionError(f"unexpected Git arguments: {arguments!r}")

    return fake_git


def _checkpoint_fixture() -> dict[str, object]:
    split_plan_id = "1" * 64
    eligibility_hash = "2" * 64
    experiment = {
        "experiment_id": "experiment:v1",
        "position": "task_update",
        "target": "task_total_accounted_tokens",
        "condition_id": "condition:v1",
        "calibrator_id": "task_max_conformal",
        "alpha": 0.1,
    }
    candidate = {
        "candidate_id": "raw_seed_history",
        "candidate_hash": "3" * 64,
    }
    seed_result = {
        "split_seed": 20260719,
        "split_plan_id": split_plan_id,
        "comparability_key": [
            "dataset:v1",
            split_plan_id,
            eligibility_hash,
            experiment["position"],
            experiment["target"],
            experiment["condition_id"],
            experiment["calibrator_id"],
            str(experiment["alpha"]),
            METRIC_SUITE_ID,
        ],
    }
    return {
        "results": {"run_id": "4" * 24},
        "experiment": experiment,
        "candidate": candidate,
        "seed_result": seed_result,
        "source_provenance_hash": "5" * 64,
        "source_run_semantic_sha256": "6" * 64,
    }


def _checkpoint_execution_hash(fixture: dict[str, object]) -> str:
    experiment = fixture["experiment"]
    candidate = fixture["candidate"]
    seed_result = fixture["seed_result"]
    comparability = seed_result["comparability_key"]
    return _semantic_sha256(
        {
            "experiment_id": experiment["experiment_id"],
            "candidate_id": candidate["candidate_id"],
            "candidate_hash": candidate["candidate_hash"],
            "dataset_id": comparability[0],
            "split_plan_id": seed_result["split_plan_id"],
            "split_seed": seed_result["split_seed"],
            "eligibility_hash": comparability[2],
            "position": experiment["position"],
            "target": experiment["target"],
            "condition_id": experiment["condition_id"],
            "calibrator_id": experiment["calibrator_id"],
            "alpha": experiment["alpha"],
            "source_provenance_hash": fixture[
                "source_provenance_hash"
            ],
        }
    )


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
    feature_set = {
        "empirical": NO_FEATURES,
        "task_chars_length": SPEND_AGGREGATE_TASK_CHARS,
        "lightgbm_structured": SPEND_AGGREGATE_STRUCTURED_FEATURES,
    }[candidate_id]
    params = {"alpha": 0.1}
    candidate_hash = _semantic_sha256(
        {
            "estimator_id": estimator_id,
            "feature_set_hash": feature_set.content_hash,
            "params": params,
            "initializer_params": {},
            "graph": graph,
        }
    )
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
        "artifact_key": _expected_artifact_key("c", candidate_hash),
        "candidate_id": candidate_id,
        "candidate_hash": candidate_hash,
        "estimator_id": estimator_id,
        "feature_set_id": feature_set.feature_set_id,
        "feature_set_hash": feature_set.content_hash,
        "role": "baseline" if candidate_id == "empirical" else "model",
        "params": params,
        "initializer_params": {},
        "seed_policy_hash": None,
        "ablation": None,
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
            "artifact_key": _expected_artifact_key("e", experiment_id),
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
                            "role": item["role"],
                            "feature_set_id": item["feature_set_id"],
                            "feature_set_hash": item["feature_set_hash"],
                            "params": item["params"],
                            "initializer_params": item[
                                "initializer_params"
                            ],
                            "graph": item["candidate_graph"],
                            "seed_policy_hash": item["seed_policy_hash"],
                            "ablation": item["ablation"],
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
    matrix = {
        "schema_version": STAGE4_MATRIX_SCHEMA_VERSION,
        "policy_id": STAGE4_MATRIX_POLICY_ID,
        "source_id": SPEND_AGGREGATE_SOURCE_ID,
        "development_protocol_id": "d" * 64,
        "capability_contract_hash": "c" * 64,
        "minimum_development_tasks": STAGE4_MIN_DEVELOPMENT_TASKS,
        "plans": plans,
        "gates": [],
        "telemetry_decisions": [],
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
    }
    matrix["matrix_id"] = _semantic_sha256(matrix)
    return {
        "source": {"source_id": SPEND_AGGREGATE_SOURCE_ID},
        "summary": {"matrix_id": matrix["matrix_id"]},
        "matrix": matrix,
        "experiments": experiments,
    }


class Stage4CompletionReleaseVerifierTests(unittest.TestCase):
    def test_shared_development_task_projection_is_required(self) -> None:
        projection = "a" * 64
        records = [
            {
                "source_name": "source",
                "condition_id": "condition",
                "experiment_id": "experiment",
                "candidate_id": candidate_id,
                "split_seed": split_seed,
                "checkpoint_parity": {
                    "development_task_count": 10,
                    "development_task_projection_sha256": projection,
                },
            }
            for candidate_id in (
                "cross_position_deduct_raw_repaired_oof_seed",
                "cross_position_deduct_point_only_oof_seed",
            )
            for split_seed in (20260719, 20260720, 20260721)
        ]
        _verify_shared_development_task_projection(records)
        records[0]["checkpoint_parity"][
            "development_task_projection_sha256"
        ] = "b" * 64
        with self.assertRaisesRegex(
            Stage4CompletionReleaseError,
            "development task identity projection",
        ):
            _verify_shared_development_task_projection(records)

    def test_release_schema_closes_and_rejects_drift(self) -> None:
        _validate_release_document(_release())

        changed = copy.deepcopy(_release())
        changed["release_control"]["release_tag"] = "moved-release-tag"
        with self.assertRaisesRegex(
            Stage4CompletionReleaseError,
            "release-control tag",
        ):
            _validate_release_document(changed)

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
            "checkpoint-only diagnostics coverage",
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

    def test_release_control_tag_binds_every_current_control_byte(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = _release_control_payloads()
            _write_release_controls(root, payloads)
            with patch(
                "scripts.verify_stage4_completion_release._git",
                side_effect=_release_control_git(payloads),
            ):
                tagged_commit = _verify_release_control_tag(
                    root,
                    _release(),
                    release_relative=(
                        "configs/stage4_completion_release.json"
                    ),
                )
            self.assertEqual(tagged_commit, "a" * 40)

    def test_release_control_tag_missing_fails_closed(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "scripts.verify_stage4_completion_release._git",
                side_effect=Stage4CompletionReleaseError(
                    "synthetic missing tag"
                ),
            ),
        ):
            with self.assertRaisesRegex(
                Stage4CompletionReleaseError,
                "tag is missing or invalid",
            ):
                _verify_release_control_tag(
                    Path(temporary),
                    _release(),
                    release_relative=(
                        "configs/stage4_completion_release.json"
                    ),
                )

    def test_tracked_only_binding_cannot_skip_a_missing_release_tag(
        self,
    ) -> None:
        controls = (
            *RELEASE_CONTROL_PATHS,
            "configs/stage4_release.json",
        )

        def fake_git(
            _root: Path,
            *arguments: str,
            maximum_bytes: int = 64 * 1024 * 1024,
        ) -> bytes:
            del maximum_bytes
            if arguments[:3] == ("ls-files", "-z", "--"):
                return b"\0".join(
                    relative.encode() for relative in controls
                ) + b"\0"
            if arguments[:5] == (
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--",
            ):
                return b""
            if arguments == (
                "rev-parse",
                "--verify",
                f"refs/tags/{COMPLETION_RELEASE_TAG}^{{commit}}",
            ):
                raise Stage4CompletionReleaseError("synthetic missing tag")
            raise AssertionError(f"unexpected Git arguments: {arguments!r}")

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "scripts.verify_stage4_completion_release._git",
                side_effect=fake_git,
            ),
            self.assertRaisesRegex(
                Stage4CompletionReleaseError,
                "tag is missing or invalid",
            ),
        ):
            _verify_tracked_bindings(
                Path(temporary),
                _release(),
                release_relative=(
                    "configs/stage4_completion_release.json"
                ),
                require_git_clean=True,
                require_release_control_tag=True,
            )

    def test_release_control_tag_move_with_changed_bytes_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current = _release_control_payloads()
            tagged = dict(current)
            changed_path = "scripts/freeze_stage4_completion.py"
            tagged[changed_path] = b"moved-tag-freezer\n"
            _write_release_controls(root, current)
            with (
                patch(
                    "scripts.verify_stage4_completion_release._git",
                    side_effect=_release_control_git(
                        tagged,
                        commit="b" * 40,
                    ),
                ),
                self.assertRaisesRegex(
                    Stage4CompletionReleaseError,
                    "tag moved.*freeze_stage4_completion.py",
                ),
            ):
                _verify_release_control_tag(
                    root,
                    _release(),
                    release_relative=(
                        "configs/stage4_completion_release.json"
                    ),
                )

    def test_release_control_current_bytes_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tagged = _release_control_payloads()
            current = dict(tagged)
            changed_path = "scripts/verify_stage4_completion_release.py"
            current[changed_path] = b"tampered-current-verifier\n"
            _write_release_controls(root, current)
            with (
                patch(
                    "scripts.verify_stage4_completion_release._git",
                    side_effect=_release_control_git(tagged),
                ),
                self.assertRaisesRegex(
                    Stage4CompletionReleaseError,
                    "current bytes differ.*verify_stage4_completion_release.py",
                ),
            ):
                _verify_release_control_tag(
                    root,
                    _release(),
                    release_relative=(
                        "configs/stage4_completion_release.json"
                    ),
                )

    def test_release_control_workflow_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tagged = _release_control_payloads()
            current = dict(tagged)
            current[".github/workflows/ci.yml"] = (
                b"name: CI\n# verification removed\n"
            )
            _write_release_controls(root, current)
            with (
                patch(
                    "scripts.verify_stage4_completion_release._git",
                    side_effect=_release_control_git(tagged),
                ),
                self.assertRaisesRegex(
                    Stage4CompletionReleaseError,
                    r"current bytes differ.*\.github/workflows/ci\.yml",
                ),
            ):
                _verify_release_control_tag(
                    root,
                    _release(),
                    release_relative=(
                        "configs/stage4_completion_release.json"
                    ),
                )

    def test_full_verifier_compares_all_nine_checkpoint_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(9):
                with self.subTest(index=index):
                    fixture = copy.deepcopy(_checkpoint_fixture())
                    fixture["seed_result"]["comparability_key"][index] = (
                        f"tampered-{index}"
                    )
                    expected_error = (
                        "checkpoint verification failed"
                        if index in {0, 2}
                        else "comparability key differs"
                    )
                    with self.assertRaisesRegex(
                        Stage4CompletionReleaseError,
                        expected_error,
                    ):
                        _checkpoint_expectation(root, **fixture)

    def test_source_run_semantic_sha256_must_close_run_id(self) -> None:
        semantic = {
            "source_name": "bagen_sokoban",
            "matrix_id": "1" * 64,
        }
        semantic_sha256 = _semantic_sha256(semantic)
        metadata = {
            "run_id": semantic_sha256[:24],
            "run_semantic": semantic,
            "results_payload_sha256": "2" * 64,
        }
        self.assertEqual(
            _training_run_semantic_sha256(
                metadata,
                expected_run_id=semantic_sha256[:24],
                expected_results_payload_sha256="2" * 64,
                description="synthetic source",
            ),
            semantic_sha256,
        )

        changed = copy.deepcopy(metadata)
        changed["run_semantic"]["matrix_id"] = "3" * 64
        with self.assertRaisesRegex(
            Stage4CompletionReleaseError,
            "run semantic SHA-256 differs",
        ):
            _training_run_semantic_sha256(
                changed,
                expected_run_id=semantic_sha256[:24],
                expected_results_payload_sha256="2" * 64,
                description="synthetic source",
            )

    def test_checkpoint_manifest_metadata_is_exact_and_semantic_bound(
        self,
    ) -> None:
        fixture = _checkpoint_fixture()
        expected_metadata = {
            "candidate_execution_hash": _checkpoint_execution_hash(fixture),
            "run_id": fixture["results"]["run_id"],
            "run_semantic_sha256": fixture[
                "source_run_semantic_sha256"
            ],
        }
        cases = {
            "extra key": {**expected_metadata, "extra": "forbidden"},
            "wrong semantic": {
                **expected_metadata,
                "run_semantic_sha256": "f" * 64,
            },
            "wrong execution": {
                **expected_metadata,
                "candidate_execution_hash": "e" * 64,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            for name, metadata in cases.items():
                with (
                    self.subTest(case=name),
                    patch(
                        "scripts.verify_stage4_completion_release.verify_artifact",
                        return_value=SimpleNamespace(
                            stage_name="development_candidate_checkpoint",
                            schema_version=1,
                            files={"candidate_result.json": "0" * 64},
                            metadata=metadata,
                        ),
                    ),
                    self.assertRaisesRegex(
                        Stage4CompletionReleaseError,
                        "checkpoint identity differs",
                    ),
                ):
                    _checkpoint_expectation(
                        Path(temporary),
                        **copy.deepcopy(fixture),
                    )

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

    def test_manifest_payload_topology_rejects_secrets_and_checkpoints(
        self,
    ) -> None:
        expected = {"results.json", "fold_artifacts/e/c/provenance.json"}
        with self.assertRaisesRegex(
            Stage4CompletionReleaseError,
            "extra=.*secret",
        ):
            _require_exact_artifact_payloads(
                {
                    **{path: SHA256 for path in expected},
                    "secret.env": SHA256,
                },
                expected,
                description="synthetic artifact",
            )
        with self.assertRaisesRegex(
            Stage4CompletionReleaseError,
            "extra=.*checkpoint",
        ):
            _require_exact_artifact_payloads(
                {
                    **{path: SHA256 for path in expected},
                    "checkpoints/unreferenced.json": SHA256,
                },
                expected,
                description="synthetic artifact",
            )

    def test_feature_and_candidate_hashes_cannot_be_tampered_together(
        self,
    ) -> None:
        changed = copy.deepcopy(_aggregate_results())
        result_candidate = changed["experiments"][0]["candidates"][0]
        plan_candidate = changed["matrix"]["plans"][0]["spec"][
            "candidates"
        ][0]
        forged_feature_hash = "f" * 64
        for candidate in (result_candidate, plan_candidate):
            candidate["feature_set_hash"] = forged_feature_hash
        forged_candidate_hash = _semantic_sha256(
            {
                "estimator_id": plan_candidate["estimator_id"],
                "feature_set_hash": forged_feature_hash,
                "params": plan_candidate["params"],
                "initializer_params": plan_candidate[
                    "initializer_params"
                ],
                "graph": plan_candidate["graph"],
            }
        )
        result_candidate["candidate_hash"] = forged_candidate_hash
        plan_candidate["candidate_hash"] = forged_candidate_hash
        result_candidate["artifact_key"] = _expected_artifact_key(
            "c",
            forged_candidate_hash,
        )
        matrix = changed["matrix"]
        matrix.pop("matrix_id")
        matrix["matrix_id"] = _semantic_sha256(matrix)
        changed["summary"]["matrix_id"] = matrix["matrix_id"]
        with self.assertRaisesRegex(
            Stage4CompletionReleaseError,
            "feature set hash is not the frozen implementation",
        ):
            _verify_result_coverage(
                changed,
                source_name="spend_aggregate",
            )

    def test_artifact_keys_are_recomputed_from_semantic_identity(self) -> None:
        changed = copy.deepcopy(_aggregate_results())
        changed["experiments"][0]["artifact_key"] = "e_" + "0" * 16
        with self.assertRaisesRegex(
            Stage4CompletionReleaseError,
            "experiment artifact key does not close",
        ):
            _verify_result_coverage(
                changed,
                source_name="spend_aggregate",
            )

    def test_diagnostics_policy_constant_is_frozen(self) -> None:
        self.assertEqual(
            DIAGNOSTICS_POLICY_ID,
            "stage4_completion_artifact_checkpoint_only_v2",
        )


if __name__ == "__main__":
    unittest.main()
