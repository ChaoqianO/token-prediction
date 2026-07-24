from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.freeze_stage4_completion import (
    DIAGNOSTICS_POLICY_ID,
    DIAGNOSTICS_STAGE_NAME,
    EXPECTED_CODE_TREE_SHA256,
    EXPECTED_DIAGNOSTICS_SOURCE_TAG,
    EXPECTED_FINAL_HOLDOUT,
    EXPECTED_SOURCE_COMMIT,
    EXPECTED_SOURCE_TAG,
    POINT_ONLY_SEED_CANDIDATE_ID,
    RAW_SEED_CANDIDATE_ID,
    DIAGNOSTICS_LIFECYCLE_UNAVAILABLE_REASON,
    DIAGNOSTICS_UNAVAILABLE_LIFECYCLE_METRICS,
    SOURCE_EXPECTATIONS,
    Stage4CompletionFreezeError,
    _audit_diagnostics_code_binding,
    _require_safe_output_ancestors,
    _semantic_sha256,
    freeze_completion_release,
)
from scripts.run_stage4_experiments import _framed_code_hash
from scripts.verify_stage4_completion_release import _validate_release_document
from token_prediction.lineage import publish_artifact


DIAGNOSTICS_CODE_BINDING = {
    "git_commit": "d" * 40,
    "code_tree_sha256": "e" * 64,
    "code_paths": [
        "scripts/run_stage4_completion_diagnostics.py",
        "src/token_prediction/evaluation/stratification.py",
        "src/token_prediction/lifecycle.py",
        "src/token_prediction/lifecycle_bundle.py",
    ],
}
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _seed_results(*, learned: bool) -> list[dict[str, object]]:
    folds = list(range(5)) if learned else []
    return [
        {
            "split_seed": split_seed,
            "fold_artifact_count": len(folds),
            "reloadable_bundle_folds": folds,
        }
        for split_seed in (20260719, 20260720, 20260721)
    ]


def _candidate(candidate_id: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "seed_results": _seed_results(learned=candidate_id != "empirical"),
    }


def _profiles(source_name: str) -> list[tuple[str, list[str]]]:
    if source_name == "spend_aggregate":
        return [
            ("task_launch", ["empirical", "model-a"]),
            ("task_launch", ["model-b", "model-c"]),
            ("task_launch", ["model-d"]),
        ]
    call = [
        (
            "call_pre",
            [
                "empirical",
                "pre_request_char_message_length",
                "lightgbm_history",
                "mlp_history",
            ],
        )
        for _ in range(15 if source_name == "bagen_swebench" else 3)
    ]
    seed = [
        (
            "task_update",
            [RAW_SEED_CANDIDATE_ID, POINT_ONLY_SEED_CANDIDATE_ID],
        )
        for _ in range(5 if source_name == "bagen_swebench" else 1)
    ]
    if source_name == "bagen_swebench":
        remaining = [
            ("task_pre", ["empirical", f"remaining-{index}-a"])
            for index in range(5)
        ] + [
            (
                "task_pre",
                [
                    f"remaining-{index}-a",
                    f"remaining-{index}-b",
                    f"remaining-{index}-c",
                ],
            )
            for index in range(5, 15)
        ]
    else:
        remaining = [
            ("task_pre", ["empirical", "remaining-a", "remaining-b"]),
            ("task_pre", ["remaining-c", "remaining-d", "remaining-e"]),
            ("task_pre", ["remaining-f", "remaining-g"]),
        ]
    return call + seed + remaining


def _training_results(
    source_index: int,
    *,
    final_holdout: dict[str, object] | None = None,
) -> tuple[str, dict[str, object], dict[str, object]]:
    expectation = SOURCE_EXPECTATIONS[source_index]
    experiments = [
        {
            "experiment_id": f"{expectation.source_name}-experiment-{index}",
            "position": position,
            "target": "synthetic_target",
            "condition_id": f"condition:{index}",
            "candidates": [_candidate(candidate_id) for candidate_id in candidate_ids],
        }
        for index, (position, candidate_ids) in enumerate(
            _profiles(expectation.source_name)
        )
    ]
    plans = [
        {"spec": {"experiment_id": experiment["experiment_id"]}}
        for experiment in experiments
    ]
    matrix: dict[str, object] = {
        "schema_version": 2,
        "policy_id": "stage4_single_axis_condition_position_target_matrix_v2",
        "source_id": expectation.source_id,
        "capability_contract_hash": "a" * 64,
        "development_protocol_id": "b" * 64,
        "minimum_development_tasks": 40,
        "plans": plans,
        "gates": [],
        "telemetry_decisions": [],
        "safety_invariants": [],
    }
    matrix["matrix_id"] = _semantic_sha256(matrix)
    run_semantic = {
        "source_name": expectation.source_name,
        "source_id": expectation.source_id,
        "matrix_id": matrix["matrix_id"],
        "git_commit": EXPECTED_SOURCE_COMMIT,
        "code_tree_sha256": EXPECTED_CODE_TREE_SHA256,
    }
    run_id = _semantic_sha256(run_semantic)[:24]
    results: dict[str, object] = {
        "results_schema_version": 1,
        "stage_name": "stage4_development_source",
        "run_policy_id": "stage4_source_three_seed_single_axis_v1",
        "artifact_layout_id": "stage4_compact_fold_artifact_layout_v1",
        "checkpoint_policy_id": "atomic_candidate_and_every_neural_epoch_v1",
        "run_id": run_id,
        "source": {
            "source_name": expectation.source_name,
            "source_id": expectation.source_id,
        },
        "data_foundation": {},
        "code_binding": {
            "git_commit": EXPECTED_SOURCE_COMMIT,
            "code_tree_sha256": EXPECTED_CODE_TREE_SHA256,
        },
        "runtime_versions": {},
        "dataset": {},
        "development_protocol": {"protocol_id": "b" * 64},
        "matrix": matrix,
        "experiments": experiments,
        "matched_coverage_calibration": [],
        "paired_same_task_across_conditions": [],
        "summary": {
            "experiment_count": expectation.experiment_count,
            "candidate_seed_run_count": expectation.candidate_seed_run_count,
            "split_seeds": [20260719, 20260720, 20260721],
            "outer_folds": 5,
            "inner_folds": 5,
            "gate_count": 0,
        },
        "final_holdout": (
            dict(EXPECTED_FINAL_HOLDOUT)
            if final_holdout is None
            else final_holdout
        ),
    }
    results["results_payload_sha256"] = _semantic_sha256(results)
    return run_id, results, run_semantic


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _publish_training_artifact(
    root: Path,
    source_index: int,
    *,
    final_holdout: dict[str, object] | None = None,
) -> tuple[Path, str]:
    run_id, results, run_semantic = _training_results(
        source_index,
        final_holdout=final_holdout,
    )
    path = root / "workspace" / "stage4" / "runs" / f"s4-{run_id[:20]}"
    path.mkdir(parents=True)
    _write_json(path / "results.json", results)
    manifest = publish_artifact(
        path,
        stage_name="stage4_development_source",
        metadata={
            "run_id": run_id,
            "run_semantic": run_semantic,
            "results_payload_sha256": results["results_payload_sha256"],
        },
    )
    return path, manifest.artifact_id


def _diagnostic_fixture() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[tuple[object, ...], dict[str, object]],
    dict[tuple[object, ...], dict[str, object]],
]:
    diagnostics: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []
    expected_diagnostics: dict[tuple[object, ...], dict[str, object]] = {}
    expected_inventory: dict[tuple[object, ...], dict[str, object]] = {}
    for source_name, condition_count in (
        ("bagen_sokoban", 1),
        ("bagen_swebench", 5),
        ("spend_openhands", 1),
    ):
        for condition_index in range(condition_count):
            condition_id = f"{source_name}:condition:{condition_index}"
            experiment_id = f"{source_name}:experiment:{condition_index}"
            for candidate_id in (
                RAW_SEED_CANDIDATE_ID,
                POINT_ONLY_SEED_CANDIDATE_ID,
            ):
                candidate_hash = _semantic_sha256(
                    [source_name, condition_id, candidate_id]
                )
                for split_seed in (20260719, 20260720, 20260721):
                    split_plan_id = _semantic_sha256(
                        [source_name, candidate_id, split_seed]
                    )
                    base = {
                        "source_name": source_name,
                        "condition_id": condition_id,
                        "experiment_id": experiment_id,
                        "candidate_id": candidate_id,
                        "candidate_hash": candidate_hash,
                        "split_seed": split_seed,
                        "split_plan_id": split_plan_id,
                    }
                    identity = tuple(base.values())
                    fold_projection = []
                    for fold in range(5):
                        manifest_sha256 = _semantic_sha256(
                            [*identity, fold, "manifest"]
                        )
                        item = {
                            **base,
                            "fold": fold,
                            "bundle_relative_path": (
                                f"fold_artifacts/e_fixture/c_fixture/"
                                f"seed_{split_seed}/fold_{fold}/bundle"
                            ),
                            "bundle_manifest_sha256": manifest_sha256,
                            "bundle_file_count": 4,
                            "load_status": "safe_loaded",
                        }
                        inventory.append(item)
                        expected_inventory[(*identity, fold)] = item
                        fold_projection.append(item)
                    projection = _semantic_sha256(
                        [source_name, candidate_id, split_seed, "prediction"]
                    )
                    cohort = _semantic_sha256(
                        [source_name, candidate_id, split_seed, "cohort"]
                    )
                    aggregate = _semantic_sha256(
                        [source_name, candidate_id, split_seed, "aggregate"]
                    )
                    parity = {
                        "status": "exact",
                        "checkpoint_artifact_id": _semantic_sha256(
                            [*identity, "checkpoint"]
                        ),
                        "checkpoint_result_sha256": _semantic_sha256(
                            [*identity, "result"]
                        ),
                        "prediction_count": 1,
                        "expected_prediction_count": 1,
                        "prediction_projection_sha256": projection,
                        "expected_prediction_projection_sha256": projection,
                        "cohort_projection_sha256": cohort,
                        "expected_cohort_projection_sha256": cohort,
                        "aggregate_metrics_projection_sha256": aggregate,
                        "expected_aggregate_metrics_projection_sha256": (
                            aggregate
                        ),
                        "development_cohort_status": "development_only",
                        "development_task_count": 1,
                        "development_task_projection_sha256": (
                            _semantic_sha256([*identity, "development"])
                        ),
                    }
                    record = {
                        **base,
                        "bundle_folds": list(range(5)),
                        "bundle_projection_sha256": _semantic_sha256(
                            fold_projection
                        ),
                        "checkpoint_parity": parity,
                        "lifecycle_metrics": {
                            "status": "unavailable",
                            "reason_code": (
                                DIAGNOSTICS_LIFECYCLE_UNAVAILABLE_REASON
                            ),
                            "labels_present": False,
                            "lifecycle_sequences_present": False,
                            "unavailable_metrics": list(
                                DIAGNOSTICS_UNAVAILABLE_LIFECYCLE_METRICS
                            ),
                            "historical_stage3_reference": None,
                        },
                    }
                    diagnostics.append(record)
                    expected_diagnostics[identity] = {
                        **base,
                        "_checkpoint_parity": parity,
                    }
    return diagnostics, inventory, expected_diagnostics, expected_inventory


def _publish_diagnostics_artifact(
    root: Path,
    source_artifact_ids: dict[str, str],
    *,
    final_holdout: dict[str, object] | None = None,
    lifecycle_reason: str = DIAGNOSTICS_LIFECYCLE_UNAVAILABLE_REASON,
    extra_payload: str | None = None,
) -> Path:
    run_id = "f" * 24
    coverage = {
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
    diagnostics, inventory, _, _ = _diagnostic_fixture()
    if lifecycle_reason != DIAGNOSTICS_LIFECYCLE_UNAVAILABLE_REASON:
        diagnostics[0]["lifecycle_metrics"]["reason_code"] = lifecycle_reason
    source_artifacts = []
    for source_index, expectation in enumerate(SOURCE_EXPECTATIONS):
        training_run_id, training, _run_semantic = _training_results(
            source_index
        )
        source_artifacts.append(
            {
                "source_name": expectation.source_name,
                "source_id": expectation.source_id,
                "run_id": training_run_id,
                "artifact_id": source_artifact_ids[expectation.source_name],
                "results_payload_sha256": training["results_payload_sha256"],
                "matrix_id": training["matrix"]["matrix_id"],
                "development_protocol_id": training["development_protocol"][
                    "protocol_id"
                ],
                "lifecycle_status": (
                    "not_applicable_no_lifecycle"
                    if expectation.source_name == "spend_aggregate"
                    else "unavailable_no_presealed_replay_projection"
                ),
            }
        )
    source_artifacts.sort(key=lambda item: item["source_name"])
    results: dict[str, object] = {
        "results_schema_version": 2,
        "stage_name": DIAGNOSTICS_STAGE_NAME,
        "policy_id": DIAGNOSTICS_POLICY_ID,
        "source_binding": {
            "git_commit": EXPECTED_SOURCE_COMMIT,
            "code_tree_sha256": EXPECTED_CODE_TREE_SHA256,
        },
        "diagnostics_code_binding": dict(DIAGNOSTICS_CODE_BINDING),
        "source_artifacts": source_artifacts,
        "coverage": coverage,
        "bundle_inventory": inventory,
        "diagnostics": diagnostics,
        "final_holdout": (
            dict(EXPECTED_FINAL_HOLDOUT)
            if final_holdout is None
            else final_holdout
        ),
    }
    results["results_payload_sha256"] = _semantic_sha256(results)
    path = (
        root
        / "workspace"
        / "stage4"
        / "completion_diagnostics"
        / f"s4diag-{run_id[:20]}"
    )
    path.mkdir(parents=True)
    _write_json(path / "results.json", results)
    if extra_payload is not None:
        (path / extra_payload).write_bytes(b"must-not-be-published\n")
    publish_artifact(
        path,
        stage_name=DIAGNOSTICS_STAGE_NAME,
        schema_version=2,
        metadata={
            "run_id": run_id,
            "run_semantic": {
                "results_schema_version": 2,
                "policy_id": DIAGNOSTICS_POLICY_ID,
                "source_binding": {
                    "git_commit": EXPECTED_SOURCE_COMMIT,
                    "code_tree_sha256": EXPECTED_CODE_TREE_SHA256,
                },
                "diagnostics_code_binding": dict(DIAGNOSTICS_CODE_BINDING),
                "source_artifacts": [
                    {
                        "source_name": item["source_name"],
                        "artifact_id": item["artifact_id"],
                        "results_payload_sha256": item[
                            "results_payload_sha256"
                        ],
                    }
                    for item in source_artifacts
                ],
                "diagnostics_runner_sha256": "b" * 64,
                "final_holdout": dict(EXPECTED_FINAL_HOLDOUT),
            },
            "results_payload_sha256": results["results_payload_sha256"],
            "source_git_commit": EXPECTED_SOURCE_COMMIT,
            "source_code_tree_sha256": EXPECTED_CODE_TREE_SHA256,
            "diagnostics_code_binding": dict(DIAGNOSTICS_CODE_BINDING),
            "source_artifact_ids": [
                source_artifact_ids[name] for name in sorted(source_artifact_ids)
            ],
            "coverage": coverage,
            "diagnostics_runner_sha256": "b" * 64,
        },
    )
    return path


def _write_parent_and_report(root: Path) -> None:
    parent = json.loads(
        (REPOSITORY_ROOT / "configs" / "stage4_release.json").read_text(
            encoding="utf-8"
        )
    )
    (root / "configs").mkdir()
    _write_json(root / "configs" / "stage4_release.json", parent)
    (root / "docs").mkdir()
    (root / "docs" / "stage-4-completion-supplement.md").write_bytes(
        b"# Development-only completion supplement\n"
    )


def _fixture(root: Path) -> tuple[list[Path], Path]:
    artifact_paths: list[Path] = []
    artifact_ids: dict[str, str] = {}
    for source_index, expectation in enumerate(SOURCE_EXPECTATIONS):
        path, artifact_id = _publish_training_artifact(root, source_index)
        artifact_paths.append(path)
        artifact_ids[expectation.source_name] = artifact_id
    diagnostics = _publish_diagnostics_artifact(root, artifact_ids)
    _write_parent_and_report(root)
    return artifact_paths, diagnostics


class FreezeStage4CompletionTests(unittest.TestCase):
    def setUp(self) -> None:
        _, _, expected_diagnostics, expected_inventory = (
            _diagnostic_fixture()
        )
        patchers = (
            patch(
                "scripts.freeze_stage4_completion._audit_diagnostics_code_binding",
                return_value={
                    **DIAGNOSTICS_CODE_BINDING,
                    "source_tag": EXPECTED_DIAGNOSTICS_SOURCE_TAG,
                },
            ),
            patch(
                "scripts.freeze_stage4_completion._code_binding_at_commit",
                return_value={
                    "git_commit": EXPECTED_SOURCE_COMMIT,
                    "code_tree_sha256": EXPECTED_CODE_TREE_SHA256,
                },
            ),
            patch(
                "scripts.freeze_stage4_completion._verify_result_coverage",
                side_effect=self._coverage,
            ),
            patch(
                "scripts.freeze_stage4_completion._load_declared_bundles",
                side_effect=lambda _root, results, **_kwargs: next(
                    item.reloadable_bundle_fold_count
                    for item in SOURCE_EXPECTATIONS
                    if item.source_name
                    == results["source"]["source_name"]
                ),
            ),
            patch(
                "scripts.freeze_stage4_completion._canonical_report_payload",
                return_value=(
                    b"# Development-only completion supplement\n"
                ),
            ),
            patch(
                "scripts.freeze_stage4_completion._expected_diagnostics_scope",
                return_value=(expected_diagnostics, expected_inventory),
            ),
        )
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def _coverage(
        _results: object,
        *,
        source_name: str,
    ) -> SimpleNamespace:
        expectation = next(
            item
            for item in SOURCE_EXPECTATIONS
            if item.source_name == source_name
        )
        if source_name == "spend_aggregate":
            call_cells = 0
            seed_cells = 0
        elif source_name == "bagen_swebench":
            call_cells = 15
            seed_cells = 5
        else:
            call_cells = 3
            seed_cells = 1
        return SimpleNamespace(
            experiment_count=expectation.experiment_count,
            candidate_seed_run_count=expectation.candidate_seed_run_count,
            reloadable_bundle_fold_count=(
                expectation.reloadable_bundle_fold_count
            ),
            call_pre_mlp_cell_count=call_cells,
            call_pre_mlp_bundle_fold_count=call_cells * 15,
            seed_policy_cell_count=seed_cells,
            seed_policy_bundle_fold_count=seed_cells * 30,
        )

    def test_freezes_deterministic_development_only_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts, diagnostics = _fixture(root)
            arguments = {
                "repository_root": root,
                "artifact_paths": artifacts,
                "diagnostics_artifact_path": diagnostics,
                "source_tag": EXPECTED_SOURCE_TAG,
                "diagnostics_source_tag": EXPECTED_DIAGNOSTICS_SOURCE_TAG,
                "output_path": "configs/stage4_completion_release.json",
            }
            with patch(
                "scripts.freeze_stage4_completion._tag_commit",
                return_value=EXPECTED_SOURCE_COMMIT,
            ):
                first = freeze_completion_release(**arguments)
                second = freeze_completion_release(**arguments)

            self.assertEqual(first, second)
            self.assertEqual(first["release_schema_version"], 2)
            self.assertEqual(
                [item["source_name"] for item in first["artifacts"]],
                [item.source_name for item in SOURCE_EXPECTATIONS],
            )
            self.assertEqual(first["protocol"]["experiment_count"], 52)
            self.assertEqual(first["protocol"]["candidate_seed_run_count"], 477)
            self.assertEqual(
                first["protocol"]["reloadable_bundle_fold_count"],
                1_950,
            )
            self.assertEqual(first["protocol"]["diagnostics_bundle_count"], 210)
            self.assertFalse(first["protocol"]["final_holdout_evaluated"])
            self.assertEqual(
                first["release_control"]["release_tag"],
                "stage4-completion-release-v1",
            )
            self.assertEqual(
                first["diagnostics_artifact"]["coverage"][
                    "checkpoint_verified_candidate_seed_count"
                ],
                42,
            )
            self.assertEqual(
                first["diagnostics_artifact"]["coverage"][
                    "lifecycle_replayed_candidate_seed_count"
                ],
                0,
            )
            _validate_release_document(first)

    def test_diagnostics_code_binding_requires_tagged_closed_tree(self) -> None:
        paths = list(DIAGNOSTICS_CODE_BINDING["code_paths"])
        payloads = {path: f"payload:{path}".encode() for path in paths}
        binding = {
            "git_commit": DIAGNOSTICS_CODE_BINDING["git_commit"],
            "code_tree_sha256": _framed_code_hash(
                [(path, payloads[path]) for path in paths]
            ),
            "code_paths": paths,
        }
        with (
            patch(
                "scripts.freeze_stage4_completion._tag_commit",
                return_value=DIAGNOSTICS_CODE_BINDING["git_commit"],
            ),
            patch(
                "scripts.freeze_stage4_completion._git_file",
                side_effect=lambda _root, _commit, path: payloads[path],
            ),
            patch(
                "scripts.freeze_stage4_completion._diagnostics_code_paths_at_commit",
                return_value=tuple(paths),
            ),
        ):
            audited = _audit_diagnostics_code_binding(
                Path("."),
                binding,
                source_tag=EXPECTED_DIAGNOSTICS_SOURCE_TAG,
            )
            self.assertEqual(
                audited["source_tag"],
                EXPECTED_DIAGNOSTICS_SOURCE_TAG,
            )
            tampered = {**binding, "code_tree_sha256": "0" * 64}
            with self.assertRaisesRegex(
                Stage4CompletionFreezeError,
                "code tree does not close",
            ):
                _audit_diagnostics_code_binding(
                    Path("."),
                    tampered,
                    source_tag=EXPECTED_DIAGNOSTICS_SOURCE_TAG,
                )

    def test_rejects_training_artifact_that_claims_final_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_paths: list[Path] = []
            artifact_ids: dict[str, str] = {}
            for source_index, expectation in enumerate(SOURCE_EXPECTATIONS):
                final = (
                    {**EXPECTED_FINAL_HOLDOUT, "evaluated": True}
                    if source_index == 0
                    else None
                )
                path, artifact_id = _publish_training_artifact(
                    root,
                    source_index,
                    final_holdout=final,
                )
                artifact_paths.append(path)
                artifact_ids[expectation.source_name] = artifact_id
            diagnostics = _publish_diagnostics_artifact(root, artifact_ids)
            _write_parent_and_report(root)
            with (
                patch(
                    "scripts.freeze_stage4_completion._tag_commit",
                    return_value=EXPECTED_SOURCE_COMMIT,
                ),
                self.assertRaisesRegex(
                    Stage4CompletionFreezeError,
                    "results verification failed",
                ),
            ):
                freeze_completion_release(
                    repository_root=root,
                    artifact_paths=artifact_paths,
                    diagnostics_artifact_path=diagnostics,
                    source_tag=EXPECTED_SOURCE_TAG,
                    diagnostics_source_tag=EXPECTED_DIAGNOSTICS_SOURCE_TAG,
                )

    def test_rejects_diagnostics_that_fabricate_lifecycle_availability(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_paths: list[Path] = []
            artifact_ids: dict[str, str] = {}
            for source_index, expectation in enumerate(SOURCE_EXPECTATIONS):
                path, artifact_id = _publish_training_artifact(root, source_index)
                artifact_paths.append(path)
                artifact_ids[expectation.source_name] = artifact_id
            diagnostics = _publish_diagnostics_artifact(
                root,
                artifact_ids,
                lifecycle_reason="fabricated_replay",
            )
            _write_parent_and_report(root)
            with (
                patch(
                    "scripts.freeze_stage4_completion._tag_commit",
                    return_value=EXPECTED_SOURCE_COMMIT,
                ),
                self.assertRaisesRegex(
                    Stage4CompletionFreezeError,
                    "lifecycle-unavailable",
                ),
            ):
                freeze_completion_release(
                    repository_root=root,
                    artifact_paths=artifact_paths,
                    diagnostics_artifact_path=diagnostics,
                    source_tag=EXPECTED_SOURCE_TAG,
                    diagnostics_source_tag=EXPECTED_DIAGNOSTICS_SOURCE_TAG,
                )

    def test_rejects_diagnostics_that_claim_final_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_paths: list[Path] = []
            artifact_ids: dict[str, str] = {}
            for source_index, expectation in enumerate(SOURCE_EXPECTATIONS):
                path, artifact_id = _publish_training_artifact(root, source_index)
                artifact_paths.append(path)
                artifact_ids[expectation.source_name] = artifact_id
            diagnostics = _publish_diagnostics_artifact(
                root,
                artifact_ids,
                final_holdout={**EXPECTED_FINAL_HOLDOUT, "evaluated": True},
            )
            _write_parent_and_report(root)
            with (
                patch(
                    "scripts.freeze_stage4_completion._tag_commit",
                    return_value=EXPECTED_SOURCE_COMMIT,
                ),
                self.assertRaisesRegex(
                    Stage4CompletionFreezeError,
                    "opened final data",
                ),
            ):
                freeze_completion_release(
                    repository_root=root,
                    artifact_paths=artifact_paths,
                    diagnostics_artifact_path=diagnostics,
                    source_tag=EXPECTED_SOURCE_TAG,
                    diagnostics_source_tag=EXPECTED_DIAGNOSTICS_SOURCE_TAG,
                )

    def test_rejects_manifest_declared_diagnostics_secret(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_paths: list[Path] = []
            artifact_ids: dict[str, str] = {}
            for source_index, expectation in enumerate(SOURCE_EXPECTATIONS):
                path, artifact_id = _publish_training_artifact(
                    root,
                    source_index,
                )
                artifact_paths.append(path)
                artifact_ids[expectation.source_name] = artifact_id
            diagnostics = _publish_diagnostics_artifact(
                root,
                artifact_ids,
                extra_payload="secret.env",
            )
            _write_parent_and_report(root)
            with (
                patch(
                    "scripts.freeze_stage4_completion._tag_commit",
                    return_value=EXPECTED_SOURCE_COMMIT,
                ),
                self.assertRaisesRegex(
                    Stage4CompletionFreezeError,
                    "topology differs",
                ),
            ):
                freeze_completion_release(
                    repository_root=root,
                    artifact_paths=artifact_paths,
                    diagnostics_artifact_path=diagnostics,
                    source_tag=EXPECTED_SOURCE_TAG,
                    diagnostics_source_tag=EXPECTED_DIAGNOSTICS_SOURCE_TAG,
                )

    def test_report_must_equal_canonical_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts, diagnostics = _fixture(root)
            report = root / "docs" / "stage-4-completion-supplement.md"
            report.write_bytes(b"# plausible but noncanonical\n")
            with (
                patch(
                    "scripts.freeze_stage4_completion._tag_commit",
                    return_value=EXPECTED_SOURCE_COMMIT,
                ),
                self.assertRaisesRegex(
                    Stage4CompletionFreezeError,
                    "canonical artifact summary",
                ),
            ):
                freeze_completion_release(
                    repository_root=root,
                    artifact_paths=artifacts,
                    diagnostics_artifact_path=diagnostics,
                    source_tag=EXPECTED_SOURCE_TAG,
                    diagnostics_source_tag=EXPECTED_DIAGNOSTICS_SOURCE_TAG,
                )

    def test_formal_output_rejects_linked_ancestor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            outside = Path(directory) / "outside"
            root.mkdir()
            outside.mkdir()
            try:
                os.symlink(
                    outside,
                    root / "configs",
                    target_is_directory=True,
                )
            except OSError as exc:
                self.skipTest(f"directory symlink unavailable: {exc}")
            with self.assertRaisesRegex(
                Stage4CompletionFreezeError,
                "linked",
            ):
                _require_safe_output_ancestors(
                    root,
                    root / "configs" / "stage4_completion_release.json",
                )

    def test_rejects_wrong_source_order_and_source_tag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts, diagnostics = _fixture(root)
            with patch(
                "scripts.freeze_stage4_completion._tag_commit",
                return_value=EXPECTED_SOURCE_COMMIT,
            ):
                with self.assertRaisesRegex(
                    Stage4CompletionFreezeError,
                    "source identity differs",
                ):
                    freeze_completion_release(
                        repository_root=root,
                        artifact_paths=[
                            artifacts[1],
                            artifacts[0],
                            *artifacts[2:],
                        ],
                        diagnostics_artifact_path=diagnostics,
                        source_tag=EXPECTED_SOURCE_TAG,
                        diagnostics_source_tag=EXPECTED_DIAGNOSTICS_SOURCE_TAG,
                    )
                with self.assertRaisesRegex(
                    Stage4CompletionFreezeError,
                    "source tag identity differs",
                ):
                    freeze_completion_release(
                        repository_root=root,
                        artifact_paths=artifacts,
                        diagnostics_artifact_path=diagnostics,
                        source_tag="another-tag",
                        diagnostics_source_tag=EXPECTED_DIAGNOSTICS_SOURCE_TAG,
                    )

    def test_lock_output_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts, diagnostics = _fixture(root)
            output = root / "configs" / "stage4_completion_release.json"
            output.write_text("{}\n", encoding="utf-8")
            with (
                patch(
                    "scripts.freeze_stage4_completion._tag_commit",
                    return_value=EXPECTED_SOURCE_COMMIT,
                ),
                self.assertRaisesRegex(
                    Stage4CompletionFreezeError,
                    "immutable and differs",
                ),
            ):
                freeze_completion_release(
                    repository_root=root,
                    artifact_paths=artifacts,
                    diagnostics_artifact_path=diagnostics,
                    source_tag=EXPECTED_SOURCE_TAG,
                    diagnostics_source_tag=EXPECTED_DIAGNOSTICS_SOURCE_TAG,
                    output_path="configs/stage4_completion_release.json",
                )


if __name__ == "__main__":
    unittest.main()
