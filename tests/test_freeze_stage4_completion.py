from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
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
    RUN_DISPERSION_EXTENSION_ID,
    RUN_VARIANCE_ID,
    SOURCE_EXPECTATIONS,
    Stage4CompletionFreezeError,
    _audit_diagnostics_code_binding,
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
) -> tuple[str, dict[str, object]]:
    expectation = SOURCE_EXPECTATIONS[source_index]
    run_id = f"{source_index + 1:x}" * 24
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
        "plans": plans,
        "gates": [],
        "telemetry_decisions": [],
        "safety_invariants": [],
    }
    matrix["matrix_id"] = _semantic_sha256(matrix)
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
        "development_protocol": {},
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
    return run_id, results


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
    run_id, results = _training_results(
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
            "results_payload_sha256": results["results_payload_sha256"],
        },
    )
    return path, manifest.artifact_id


def _run_dispersion(
    *,
    extension_id: str = RUN_DISPERSION_EXTENSION_ID,
) -> dict[str, object]:
    return {
        "run_variance_id": RUN_VARIANCE_ID,
        "run_dispersion_extension_id": extension_id,
        "n_tasks": 1,
        "n_scored_runs": 1,
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


def _publish_diagnostics_artifact(
    root: Path,
    source_artifact_ids: dict[str, str],
    *,
    final_holdout: dict[str, object] | None = None,
    extension_id: str = RUN_DISPERSION_EXTENSION_ID,
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
        "replayed_run_count": 42,
        "scored_run_count": 42,
        "scored_boundary_count": 42,
        "unscored_context_boundary_count": 0,
    }
    results: dict[str, object] = {
        "results_schema_version": 1,
        "stage_name": DIAGNOSTICS_STAGE_NAME,
        "policy_id": DIAGNOSTICS_POLICY_ID,
        "source_binding": {
            "git_commit": EXPECTED_SOURCE_COMMIT,
            "code_tree_sha256": EXPECTED_CODE_TREE_SHA256,
        },
        "diagnostics_code_binding": dict(DIAGNOSTICS_CODE_BINDING),
        "source_artifacts": [
            {
                "source_name": expectation.source_name,
                "artifact_id": source_artifact_ids[expectation.source_name],
            }
            for expectation in sorted(
                SOURCE_EXPECTATIONS,
                key=lambda item: item.source_name,
            )
        ],
        "coverage": coverage,
        "bundle_inventory": [{"index": index} for index in range(210)],
        "diagnostics": [
            {
                "index": index,
                "run_variance": _run_dispersion(extension_id=extension_id),
            }
            for index in range(42)
        ],
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
    publish_artifact(
        path,
        stage_name=DIAGNOSTICS_STAGE_NAME,
        metadata={
            "run_id": run_id,
            "run_semantic": {
                "results_schema_version": 1,
                "policy_id": DIAGNOSTICS_POLICY_ID,
                "source_binding": {
                    "git_commit": EXPECTED_SOURCE_COMMIT,
                    "code_tree_sha256": EXPECTED_CODE_TREE_SHA256,
                },
                "diagnostics_code_binding": dict(DIAGNOSTICS_CODE_BINDING),
                "source_artifacts": [
                    {
                        "source_name": expectation.source_name,
                        "artifact_id": source_artifact_ids[
                            expectation.source_name
                        ],
                        "results_payload_sha256": "a" * 64,
                    }
                    for expectation in sorted(
                        SOURCE_EXPECTATIONS,
                        key=lambda item: item.source_name,
                    )
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
    (root / "docs" / "stage-4-completion-supplement.md").write_text(
        "# Development-only completion supplement\n",
        encoding="utf-8",
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
        patcher = patch(
            "scripts.freeze_stage4_completion._audit_diagnostics_code_binding",
            return_value={
                **DIAGNOSTICS_CODE_BINDING,
                "source_tag": EXPECTED_DIAGNOSTICS_SOURCE_TAG,
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)

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
                "output_path": "configs/fixture-completion.json",
            }
            with patch(
                "scripts.freeze_stage4_completion._tag_commit",
                return_value=EXPECTED_SOURCE_COMMIT,
            ):
                first = freeze_completion_release(**arguments)
                second = freeze_completion_release(**arguments)

            self.assertEqual(first, second)
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
                first["diagnostics_artifact"]["coverage"]["scored_run_count"],
                42,
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

    def test_rejects_diagnostics_without_dispersion_extension(self) -> None:
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
                extension_id="legacy_variance_only",
            )
            _write_parent_and_report(root)
            with (
                patch(
                    "scripts.freeze_stage4_completion._tag_commit",
                    return_value=EXPECTED_SOURCE_COMMIT,
                ),
                self.assertRaisesRegex(
                    Stage4CompletionFreezeError,
                    "dispersion extension",
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
            output = root / "configs" / "fixture-completion.json"
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
                    output_path="configs/fixture-completion.json",
                )


if __name__ == "__main__":
    unittest.main()
