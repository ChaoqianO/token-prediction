from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts import run_stage4_completion_diagnostics as diagnostics
from scripts.run_stage2_experiments import SOURCE_NAMES
from scripts.summarize_stage4_completion import (
    load_completion_diagnostics_artifact,
)
from token_prediction.lineage import publish_artifact


CONDITIONS = {
    "bagen_sokoban": ("condition:sokoban",),
    "bagen_swebench": tuple(f"condition:swe-{index}" for index in range(5)),
    "spend_openhands": ("condition:openhands",),
}


def _source_artifacts() -> list[dict[str, object]]:
    values = []
    for index, source_name in enumerate(sorted(SOURCE_NAMES)):
        values.append(
            {
                "source_name": source_name,
                "source_id": SOURCE_NAMES[source_name],
                "run_id": f"{index + 1:024x}",
                "artifact_id": f"{index + 1:064x}",
                "results_payload_sha256": f"{index + 11:064x}",
                "matrix_id": f"{index + 21:064x}",
                "development_protocol_id": f"{index + 31:064x}",
                "lifecycle_status": (
                    "not_applicable_no_lifecycle"
                    if source_name == "spend_aggregate"
                    else "complete"
                ),
            }
        )
    return values


def _progress() -> dict[str, object]:
    return {
        "stratification_id": diagnostics.PROGRESS_ID,
        "selection_policy": "first_boundary_at_or_after_sequence_fraction_v1",
        "strata": {
            key: {
                "checkpoint": checkpoint,
                "n_sequences": 3,
                "n_selected_boundaries": 3,
                "n_scored": 2,
                "n_unscored": 1,
                "metrics": {"mae": 1.0},
            }
            for key, checkpoint in (("p25", 0.25), ("p50", 0.5), ("p75", 0.75))
        },
    }


def _termination() -> dict[str, object]:
    return {
        "stratification_id": diagnostics.TERMINATION_ID,
        "strata": {
            "observed_termination": {
                "n_sequences": 3,
                "n_tasks": 3,
                "n_update_boundaries": 3,
                "n_scored": 2,
                "n_context_only": 1,
                "metrics": {"mae": 1.0},
            }
        },
    }


def _run_variance() -> dict[str, object]:
    return {
        "run_variance_id": diagnostics.RUN_VARIANCE_ID,
        "run_dispersion_extension_id": diagnostics.RUN_DISPERSION_EXTENSION_ID,
        "n_tasks": 3,
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


def _diagnostics_and_inventory() -> tuple[
    list[dict[str, object]], list[dict[str, object]]
]:
    diagnostic_rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []
    for source_name in sorted(CONDITIONS):
        for condition_index, condition_id in enumerate(CONDITIONS[source_name]):
            experiment_id = f"experiment:{source_name}:{condition_index}"
            for candidate_index, candidate_id in enumerate(
                sorted(diagnostics.EXPECTED_LIFECYCLE_CANDIDATES)
            ):
                candidate_hash = (
                    f"{100 + condition_index * 10 + candidate_index:064x}"
                )
                for split_seed in diagnostics.STAGE_SPLIT_SEEDS:
                    seed_inventory = []
                    for fold in range(5):
                        item = {
                            "source_name": source_name,
                            "condition_id": condition_id,
                            "experiment_id": experiment_id,
                            "candidate_id": candidate_id,
                            "candidate_hash": candidate_hash,
                            "split_seed": split_seed,
                            "split_plan_id": f"{split_seed + condition_index:064x}",
                            "fold": fold,
                            "bundle_relative_path": (
                                f"fold_artifacts/e_0000000000000000/"
                                f"c_{candidate_index:016x}/seed_{split_seed}/"
                                f"fold_{fold}/bundle"
                            ),
                            "bundle_manifest_sha256": (
                                f"{split_seed + condition_index + candidate_index + fold:064x}"
                            ),
                            "bundle_file_count": 13,
                            "load_status": "safe_loaded",
                        }
                        inventory_rows.append(item)
                        seed_inventory.append(item)
                    projection = diagnostics._bundle_projection(seed_inventory)
                    prediction_projection = (
                        f"{split_seed + condition_index + candidate_index + 500:064x}"
                    )
                    diagnostic_rows.append(
                        {
                            "source_name": source_name,
                            "condition_id": condition_id,
                            "experiment_id": experiment_id,
                            "candidate_id": candidate_id,
                            "candidate_hash": candidate_hash,
                            "split_seed": split_seed,
                            "split_plan_id": f"{split_seed + condition_index:064x}",
                            "bundle_folds": list(range(5)),
                            "bundle_projection_sha256": projection,
                            "reload_parity": {
                                "status": "exact",
                                "scored_prediction_count": 2,
                                "expected_prediction_count": 2,
                                "prediction_projection_sha256": (
                                    prediction_projection
                                ),
                                "expected_prediction_projection_sha256": (
                                    prediction_projection
                                ),
                            },
                            "progress": _progress(),
                            "termination": _termination(),
                            "run_variance": _run_variance(),
                        }
                    )
    diagnostic_rows.sort(key=diagnostics._diagnostic_identity)
    inventory_rows.sort(key=diagnostics._inventory_identity)
    return diagnostic_rows, inventory_rows


def _results() -> dict[str, object]:
    diagnostic_rows, inventory_rows = _diagnostics_and_inventory()
    return diagnostics._build_results(
        source_binding={
            "git_commit": diagnostics.EXPECTED_SOURCE_COMMIT,
            "code_tree_sha256": diagnostics.EXPECTED_SOURCE_CODE_TREE_SHA256,
        },
        diagnostics_code_binding={
            "git_commit": "a" * 40,
            "code_tree_sha256": "b" * 64,
            "code_paths": sorted(diagnostics.DIAGNOSTICS_DIRECT_CODE_PATHS),
        },
        source_artifacts=_source_artifacts(),
        diagnostics=diagnostic_rows,
        inventory=inventory_rows,
        replay_totals=(126, 84, 84, 42),
    )


def _reclose(value: dict[str, object]) -> None:
    value.pop("results_payload_sha256", None)
    value["results_payload_sha256"] = diagnostics._semantic_sha256(value)


class Stage4CompletionDiagnosticsTests(unittest.TestCase):
    def test_complete_schema_closes_exact_coverage(self) -> None:
        results = _results()
        self.assertEqual(
            diagnostics.verify_diagnostics_results_document(results),
            results["results_payload_sha256"],
        )
        self.assertEqual(len(results["diagnostics"]), 42)
        self.assertEqual(len(results["bundle_inventory"]), 210)
        self.assertEqual(
            results["coverage"],
            {
                "bound_source_artifact_count": 4,
                "lifecycle_source_count": 3,
                "lifecycle_condition_count": 7,
                "lifecycle_candidate_count": 2,
                "lifecycle_candidate_cell_count": 14,
                "lifecycle_candidate_seed_count": 42,
                "lifecycle_bundle_count": 210,
                "replayed_run_count": 126,
                "scored_run_count": 84,
                "scored_boundary_count": 84,
                "unscored_context_boundary_count": 42,
            },
        )

    def test_final_parity_dispersion_and_inventory_tampering_fail(self) -> None:
        cases = []
        opened = copy.deepcopy(_results())
        opened["final_holdout"]["evaluated"] = True
        _reclose(opened)
        cases.append((opened, "final holdout"))

        parity = copy.deepcopy(_results())
        parity["diagnostics"][0]["reload_parity"]["status"] = "close_enough"
        _reclose(parity)
        cases.append((parity, "reload parity"))

        dispersion = copy.deepcopy(_results())
        dispersion["diagnostics"][0]["run_variance"][
            "run_dispersion_extension_id"
        ] = "old"
        _reclose(dispersion)
        cases.append((dispersion, "evaluator identity"))

        inventory = copy.deepcopy(_results())
        inventory["bundle_inventory"][0]["load_status"] = "unchecked"
        _reclose(inventory)
        cases.append((inventory, "inventory entry"))

        for value, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(
                    diagnostics.Stage4CompletionDiagnosticsError,
                    message,
                ):
                    diagnostics.verify_diagnostics_results_document(value)

    def test_dual_code_binding_is_required(self) -> None:
        results = _results()
        results["diagnostics_code_binding"]["code_paths"] = [
            "scripts/run_stage4_completion_diagnostics.py"
        ]
        _reclose(results)
        with self.assertRaisesRegex(
            diagnostics.Stage4CompletionDiagnosticsError,
            "code paths",
        ):
            diagnostics.verify_diagnostics_results_document(results)

    def test_output_root_is_fixed_and_bundle_projection_is_canonical(self) -> None:
        root = Path(__file__).resolve().parents[1]
        expected = diagnostics._safe_output_parent(
            root,
            diagnostics.DEFAULT_OUTPUT_ROOT,
        )
        self.assertEqual(
            expected,
            root / "workspace" / "stage4" / "completion_diagnostics",
        )
        with self.assertRaisesRegex(
            diagnostics.Stage4CompletionDiagnosticsError,
            "must be",
        ):
            diagnostics._safe_output_parent(
                root,
                "workspace/stage4/final",
            )
        _, inventory = _diagnostics_and_inventory()
        first = inventory[:5]
        self.assertEqual(
            diagnostics._bundle_projection(first),
            diagnostics._bundle_projection(tuple(reversed(first))),
        )

    def test_summarizer_fully_verifies_the_immutable_supplement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = (
                root
                / "workspace"
                / "stage4"
                / "completion_diagnostics"
                / "s4diag-synthetic"
            )
            artifact.mkdir(parents=True)
            results = _results()
            (artifact / "results.json").write_text(
                json.dumps(results, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            manifest = publish_artifact(
                artifact,
                stage_name=diagnostics.STAGE_NAME,
                schema_version=diagnostics.ARTIFACT_SCHEMA_VERSION,
                metadata={
                    "results_payload_sha256": results[
                        "results_payload_sha256"
                    ]
                },
            )
            loaded = load_completion_diagnostics_artifact(
                artifact,
                repo_root=root,
                diagnostics_root=(
                    root / "workspace" / "stage4" / "completion_diagnostics"
                ),
            )
            self.assertEqual(loaded.artifact_id, manifest.artifact_id)
            self.assertEqual(
                loaded.results_payload_sha256,
                results["results_payload_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
