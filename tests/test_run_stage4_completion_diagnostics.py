from __future__ import annotations

import ast
import copy
import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts import run_stage4_completion_diagnostics as diagnostics
from scripts.run_stage2_experiments import SOURCE_NAMES
from token_prediction.dataset import PredictionPosition, PredictionTarget
from token_prediction.estimators.base import TokenForecast
from token_prediction.experiment import CandidateResult, PredictionRecord
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
                    else "unavailable_no_presealed_replay_projection"
                ),
            }
        )
    return values


def _diagnostics_and_inventory() -> tuple[
    list[dict[str, object]], list[dict[str, object]]
]:
    diagnostic_rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []
    for source_name in sorted(CONDITIONS):
        for condition_index, condition_id in enumerate(CONDITIONS[source_name]):
            experiment_id = f"experiment:{source_name}:{condition_index}"
            development_task_projection = diagnostics._semantic_sha256(
                {
                    "policy_id": diagnostics.DEVELOPMENT_TASK_PSEUDONYM_POLICY_ID,
                    "source_name": source_name,
                    "condition_id": condition_id,
                    "task_pseudonyms": ["task-a", "task-b"],
                }
            )
            for candidate_index, candidate_id in enumerate(
                sorted(diagnostics.EXPECTED_LIFECYCLE_CANDIDATES)
            ):
                candidate_hash = (
                    f"{100 + condition_index * 10 + candidate_index:064x}"
                )
                experiment_key = diagnostics._artifact_key("e", experiment_id)
                candidate_key = diagnostics._artifact_key("c", candidate_hash)
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
                                f"fold_artifacts/{experiment_key}/"
                                f"{candidate_key}/seed_{split_seed}/"
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
                            "checkpoint_parity": {
                                "status": "exact",
                                "checkpoint_artifact_id": (
                                    f"{split_seed + candidate_index + 700:064x}"
                                ),
                                "checkpoint_result_sha256": (
                                    f"{split_seed + candidate_index + 800:064x}"
                                ),
                                "prediction_count": 2,
                                "expected_prediction_count": 2,
                                "prediction_projection_sha256": (
                                    prediction_projection
                                ),
                                "expected_prediction_projection_sha256": (
                                    prediction_projection
                                ),
                                "cohort_projection_sha256": (
                                    f"{split_seed + candidate_index + 900:064x}"
                                ),
                                "expected_cohort_projection_sha256": (
                                    f"{split_seed + candidate_index + 900:064x}"
                                ),
                                "aggregate_metrics_projection_sha256": (
                                    f"{split_seed + candidate_index + 1000:064x}"
                                ),
                                "expected_aggregate_metrics_projection_sha256": (
                                    f"{split_seed + candidate_index + 1000:064x}"
                                ),
                                "development_cohort_status": "development_only",
                                "development_task_count": 2,
                                "development_task_projection_sha256": (
                                    development_task_projection
                                ),
                            },
                            "lifecycle_metrics": {
                                "status": "unavailable",
                                "reason_code": (
                                    diagnostics.LIFECYCLE_UNAVAILABLE_REASON
                                ),
                                "labels_present": False,
                                "lifecycle_sequences_present": False,
                                "unavailable_metrics": list(
                                    diagnostics.UNAVAILABLE_LIFECYCLE_METRICS
                                ),
                                "historical_stage3_reference": None,
                            },
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
                "checkpoint_verified_candidate_seed_count": 42,
                "lifecycle_replayed_candidate_seed_count": 0,
                "lifecycle_metrics_unavailable_candidate_seed_count": 42,
            },
        )

    def test_final_parity_dispersion_and_inventory_tampering_fail(self) -> None:
        cases = []
        opened = copy.deepcopy(_results())
        opened["final_holdout"]["evaluated"] = True
        _reclose(opened)
        cases.append((opened, "final holdout"))

        parity = copy.deepcopy(_results())
        parity["diagnostics"][0]["checkpoint_parity"]["status"] = "close_enough"
        _reclose(parity)
        cases.append((parity, "checkpoint parity"))

        task_identity = copy.deepcopy(_results())
        task_identity["diagnostics"][0]["checkpoint_parity"][
            "development_task_projection_sha256"
        ] = "f" * 64
        _reclose(task_identity)
        cases.append((task_identity, "development task identity projection"))

        dispersion = copy.deepcopy(_results())
        dispersion["diagnostics"][0]["lifecycle_metrics"][
            "unavailable_metrics"
        ] = ["progress"]
        _reclose(dispersion)
        cases.append((dispersion, "unavailable lifecycle"))

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

    def test_raw_and_final_access_paths_are_absent_from_diagnostics(self) -> None:
        module_source = Path(diagnostics.__file__).read_text(encoding="utf-8")
        tree = ast.parse(module_source)
        forbidden_symbols = {
            "Stage2LoadedSource",
            "Stage4Matrix",
            "_count_termination",
            "_prediction_result",
            "_replay_candidate_seed",
            "_validate_reconstructed_source",
            "build_lifecycle_slice",
            "evaluate_progress_checkpoints",
            "evaluate_same_task_run_variance",
            "evaluate_termination_strata",
            "load_stage2_source",
        }
        referenced_names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        imported_names = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        defined_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(
                node,
                (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            )
        }
        self.assertFalse(
            forbidden_symbols & (referenced_names | imported_names | defined_names)
        )
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertFalse(
            {
                module
                for module in imported_modules
                if module.endswith("run_stage2_experiments")
                or module.endswith("stage4_matrix")
            }
        )
        for forbidden in forbidden_symbols:
            with self.subTest(forbidden_symbol=forbidden):
                self.assertNotIn(forbidden, module_source)

        source = "\n".join(
            inspect.getsource(value)
            for value in (
                diagnostics.run_completion_diagnostics,
                diagnostics._source_diagnostics,
                diagnostics._audit_candidate_seed_without_raw,
                diagnostics._checkpoint_parity,
            )
        )
        for forbidden in (
            "load_stage2_source",
            "workspace/external",
            "stage4/final",
            "tarfile",
            "gzip",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

        experiment = {
            "experiment_id": "experiment:lifecycle",
            "artifact_key": "e_" + "0" * 16,
            "position": "task_update",
            "target": "task_provider_accounted_remaining_tokens",
            "condition_id": "condition:test",
            "calibrator_id": "task_max_conformal",
            "alpha": 0.1,
            "candidates": [
                {
                    "candidate_id": candidate_id,
                    "candidate_hash": f"{index + 1:064x}",
                    "artifact_key": f"c_{index + 1:016x}",
                    "seed_results": [
                        {"split_seed": seed}
                        for seed in diagnostics.STAGE_SPLIT_SEEDS
                    ],
                }
                for index, candidate_id in enumerate(
                    sorted(diagnostics.EXPECTED_LIFECYCLE_CANDIDATES)
                )
            ],
        }
        artifact = diagnostics.VerifiedSourceArtifact(
            SimpleNamespace(
                document={
                    "run_id": "1" * 24,
                    "source": {
                        "source_name": "bagen_sokoban",
                        "source_id": SOURCE_NAMES["bagen_sokoban"],
                    },
                    "matrix": {"matrix_id": "2" * 64},
                    "development_protocol": {"protocol_id": "3" * 64},
                    "experiments": [experiment],
                },
                artifact_id="4" * 64,
                results_payload_sha256="5" * 64,
            ),
            SimpleNamespace(),
        )
        audited = diagnostics.AuditedDiagnostic(
            document={"safe": True},
            inventory=(),
        )
        with (
            mock.patch.object(
                diagnostics,
                "_audit_candidate_seed_without_raw",
                return_value=audited,
            ) as audit,
            mock.patch.object(
                Path,
                "open",
                side_effect=AssertionError("raw/final path opened"),
            ),
            mock.patch.object(
                Path,
                "read_bytes",
                side_effect=AssertionError("raw/final bytes read"),
            ),
            mock.patch.object(
                Path,
                "read_text",
                side_effect=AssertionError("raw/final text read"),
            ),
        ):
            document, rows, inventory = diagnostics._source_diagnostics(
                root=Path("unused"),
                artifact=artifact,
            )
        self.assertEqual(
            document["lifecycle_status"],
            "unavailable_no_presealed_replay_projection",
        )
        self.assertEqual(len(rows), 6)
        self.assertEqual(inventory, [])
        self.assertEqual(audit.call_count, 6)

    def test_arbitrary_json_and_final_results_are_rejected_before_read(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace" / "stage4" / "runs").mkdir(parents=True)
            arbitrary = root / "raw-secret.json"
            arbitrary.write_text('{"secret": true}\n', encoding="utf-8")
            final = root / "workspace" / "stage4" / "final" / "results.json"
            final.parent.mkdir(parents=True)
            final.write_text('{"evaluated": true}\n', encoding="utf-8")
            for supplied in (arbitrary, final):
                with self.subTest(supplied=supplied):
                    with (
                        mock.patch.object(
                            Path,
                            "read_bytes",
                            side_effect=AssertionError("payload bytes read"),
                        ),
                        mock.patch.object(
                            Path,
                            "read_text",
                            side_effect=AssertionError("payload text read"),
                        ),
                        self.assertRaises(
                            diagnostics.Stage4CompletionDiagnosticsError
                        ),
                    ):
                        diagnostics._direct_artifact_references(
                            root,
                            [supplied],
                        )

    def test_runs_root_link_or_reparse_is_rejected_lexically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_runs = root / "real-runs"
            real_runs.mkdir()
            runs = root / "workspace" / "stage4" / "runs"
            runs.parent.mkdir(parents=True)
            try:
                os.symlink(real_runs, runs, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlink unavailable: {exc}")
            artifact = runs / ("s4-" + "0" * 20)
            with self.assertRaisesRegex(
                diagnostics.Stage4CompletionDiagnosticsError,
                "symlink|junction|reparse",
            ):
                diagnostics._direct_artifact_references(root, [artifact])

    def test_existing_output_is_recomputed_before_it_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_parent = root / "diagnostics"
            run_semantic = {"fixture": "existing-recompute"}
            run_id = diagnostics._semantic_sha256(run_semantic)[:24]
            output = output_parent / diagnostics._output_key(run_id)
            output.mkdir(parents=True)
            loaded = SimpleNamespace(
                path=root / "source",
                artifact_id="a" * 64,
                results_payload_sha256="b" * 64,
                document={},
            )
            artifact = diagnostics.VerifiedSourceArtifact(
                loaded,
                SimpleNamespace(),
            )
            expected_results = {"recomputed": True}
            summary = diagnostics.CompletionDiagnosticsSummary(
                run_id=run_id,
                output_dir=output,
                artifact_id="c" * 64,
                results_payload_sha256="d" * 64,
                diagnostic_count=42,
                bundle_count=210,
            )
            order: list[str] = []

            def audit(
                **_kwargs: object,
            ) -> tuple[dict[str, object], list[object], list[object]]:
                order.append("audit")
                return ({"source_name": "fixture"}, [], [])

            def load(
                *_args: object,
                **_kwargs: object,
            ) -> diagnostics.CompletionDiagnosticsSummary:
                order.append("load")
                return summary

            with (
                mock.patch.object(diagnostics, "_verify_runner_origin"),
                mock.patch.object(
                    diagnostics,
                    "_safe_output_parent",
                    return_value=output_parent,
                ),
                mock.patch.object(
                    diagnostics,
                    "_direct_artifact_references",
                    return_value=(SimpleNamespace(),),
                ),
                mock.patch.object(
                    diagnostics,
                    "_verify_source_artifacts",
                    return_value=(artifact,),
                ),
                mock.patch.object(
                    diagnostics,
                    "_source_binding",
                    return_value=({"git_commit": "x"}, ()),
                ),
                mock.patch.object(
                    diagnostics,
                    "capture_diagnostics_code_binding",
                    return_value={"code": "binding"},
                ),
                mock.patch.object(
                    diagnostics,
                    "_runner_sha256",
                    return_value="e" * 64,
                ),
                mock.patch.object(
                    diagnostics,
                    "_run_semantic",
                    return_value=run_semantic,
                ),
                mock.patch.object(
                    diagnostics,
                    "_source_diagnostics",
                    side_effect=audit,
                ),
                mock.patch.object(
                    diagnostics,
                    "_build_results",
                    return_value=expected_results,
                ),
                mock.patch.object(
                    diagnostics,
                    "verify_diagnostics_results_document",
                    return_value="f" * 64,
                ),
                mock.patch.object(
                    diagnostics,
                    "verify_artifact",
                    return_value=SimpleNamespace(artifact_id="a" * 64),
                ),
                mock.patch.object(
                    diagnostics,
                    "_load_existing",
                    side_effect=load,
                ) as existing,
            ):
                actual = diagnostics.run_completion_diagnostics(
                    repository_root=root,
                    artifact_inputs=["unused"],
                )
            self.assertEqual(actual, summary)
            self.assertEqual(order, ["audit", "load"])
            self.assertEqual(
                existing.call_args.kwargs["expected_results"],
                expected_results,
            )

    def test_source_manifest_run_semantic_and_scope_close_exactly(self) -> None:
        results_sha256 = "1" * 64
        runtime_versions = {"python_version": "3.12.10"}
        run_semantic = {
            "results_schema_version": 1,
            "run_policy_id": "run-policy",
            "checkpoint_policy_id": "checkpoint-policy",
            "source_name": "bagen_sokoban",
            "source_id": "source-id",
            "revision": "revision",
            "raw_artifact_sha256": "2" * 64,
            "data_foundation_baseline_lock_sha256": "3" * 64,
            "base_dataset_id": "4" * 64,
            "derived_dataset_id": "5" * 64,
            "development_protocol_id": "6" * 64,
            "matrix_id": "7" * 64,
            "git_commit": diagnostics.EXPECTED_SOURCE_COMMIT,
            "code_tree_sha256": diagnostics.EXPECTED_SOURCE_CODE_TREE_SHA256,
            "runtime_versions": runtime_versions,
        }
        run_id = diagnostics._semantic_sha256(run_semantic)[:24]
        loaded = SimpleNamespace(
            document={
                "results_schema_version": 1,
                "run_id": run_id,
                "source": {
                    "source_name": "bagen_sokoban",
                    "source_id": "source-id",
                    "revision": "revision",
                    "raw_artifact_sha256": "2" * 64,
                },
                "dataset": {
                    "base_dataset_id": "4" * 64,
                    "derived_dataset_id": "5" * 64,
                },
                "matrix": {"matrix_id": "7" * 64},
                "development_protocol": {"protocol_id": "6" * 64},
                "code_binding": {
                    "git_commit": diagnostics.EXPECTED_SOURCE_COMMIT,
                    "code_tree_sha256": (
                        diagnostics.EXPECTED_SOURCE_CODE_TREE_SHA256
                    ),
                },
                "runtime_versions": runtime_versions,
            },
            results_payload_sha256=results_sha256,
        )
        manifest = SimpleNamespace(
            stage_name=diagnostics.SOURCE_STAGE_NAME,
            schema_version=diagnostics.SOURCE_ARTIFACT_SCHEMA_VERSION,
            files={
                "results.json": "8" * 64,
                "fold_artifacts/e/c/seed/fold/bundle/manifest.json": "9" * 64,
            },
            metadata={
                "run_id": run_id,
                "run_semantic": run_semantic,
                "results_payload_sha256": results_sha256,
            },
        )
        artifact = diagnostics.VerifiedSourceArtifact(loaded, manifest)
        diagnostics._verify_source_manifest_scope(artifact)

        manifest.metadata = {**manifest.metadata, "extra": "forbidden"}
        with self.assertRaisesRegex(
            diagnostics.Stage4CompletionDiagnosticsError,
            "metadata scope",
        ):
            diagnostics._verify_source_manifest_scope(artifact)

    def test_joint_source_reseal_cannot_replace_frozen_identity(self) -> None:
        source_name = "bagen_sokoban"
        expected = diagnostics.EXPECTED_SOURCE_ARTIFACT_IDENTITIES[source_name]
        artifact = diagnostics.VerifiedSourceArtifact(
            SimpleNamespace(
                path=Path("workspace/stage4/runs")
                / ("s4-" + expected["run_id"][:20]),
                document={"run_id": expected["run_id"]},
                artifact_id="f" * 64,
                results_payload_sha256=expected["results_payload_sha256"],
            ),
            SimpleNamespace(),
        )
        with self.assertRaisesRegex(
            diagnostics.Stage4CompletionDiagnosticsError,
            "frozen completion identity",
        ):
            diagnostics._verify_pinned_source_identity(
                artifact,
                source_name=source_name,
            )

    def test_joint_identity_inventory_tamper_breaks_cell_topology(self) -> None:
        results = copy.deepcopy(_results())
        first = results["diagnostics"][0]
        source_name = first["source_name"]
        condition_id = first["condition_id"]
        candidate_id = first["candidate_id"]
        replacement_experiment = first["experiment_id"] + ":tampered"
        for item in results["diagnostics"]:
            if (
                item["source_name"] == source_name
                and item["condition_id"] == condition_id
                and item["candidate_id"] == candidate_id
            ):
                item["experiment_id"] = replacement_experiment
        for item in results["bundle_inventory"]:
            if (
                item["source_name"] == source_name
                and item["condition_id"] == condition_id
                and item["candidate_id"] == candidate_id
            ):
                item["experiment_id"] = replacement_experiment
                item["bundle_relative_path"] = (
                    "fold_artifacts/"
                    f"{diagnostics._artifact_key('e', replacement_experiment)}/"
                    f"{diagnostics._artifact_key('c', item['candidate_hash'])}/"
                    f"seed_{item['split_seed']}/fold_{item['fold']}/bundle"
                )
        for item in results["diagnostics"]:
            matching = [
                inventory
                for inventory in results["bundle_inventory"]
                if diagnostics._diagnostic_identity(inventory)
                == diagnostics._diagnostic_identity(item)
            ]
            item["bundle_projection_sha256"] = diagnostics._bundle_projection(
                matching
            )
        results["diagnostics"].sort(key=diagnostics._diagnostic_identity)
        results["bundle_inventory"].sort(key=diagnostics._inventory_identity)
        _reclose(results)
        with self.assertRaisesRegex(
            diagnostics.Stage4CompletionDiagnosticsError,
            "condition-cell topology",
        ):
            diagnostics.verify_diagnostics_results_document(results)

    def test_checkpoint_projection_is_independently_bound_and_development_only(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_plan_id = "1" * 64
            source_provenance_hash = "2" * 64
            candidate_hash = "3" * 64
            task_id = "task-a"
            condition_id = "condition:test"
            experiment_id = "experiment:test"
            source_run_semantic = {"fixture": "checkpoint-parity"}
            run_semantic_sha256 = diagnostics._semantic_sha256(
                source_run_semantic
            )
            run_id = run_semantic_sha256[:24]
            source_results_sha256 = "7" * 64
            candidate_id = diagnostics.RAW_SEED_CANDIDATE_ID
            target = PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS
            record = PredictionRecord(
                candidate_id=candidate_id,
                point_id="point-a",
                task_id=task_id,
                trajectory_id="trajectory-a",
                condition_id=condition_id,
                fold=0,
                target=target,
                forecast=TokenForecast(
                    point_id="point-a",
                    target=target,
                    lower=1.0,
                    point=2.0,
                    upper=3.0,
                    latency_ms=0.1,
                    raw_lower=1.0,
                    raw_point=2.0,
                    raw_upper=3.0,
                ),
                sample_weight=1.0,
            )
            task_metric = {
                "n_points": 1,
                "n_trajectories": 1,
                "weight_sum": 1.0,
                "weighted_mae": 1.0,
                "weighted_coverage": 1.0,
                "weighted_interval_score": 2.0,
            }
            result = CandidateResult(
                candidate_id=candidate_id,
                candidate_hash=candidate_hash,
                dataset_id="dataset:test",
                split_plan_id=split_plan_id,
                eligibility_hash="5" * 64,
                position=PredictionPosition.TASK_UPDATE,
                target=target,
                condition_id=condition_id,
                calibrator_id="task_max_conformal",
                alpha=0.1,
                metric_suite_id=diagnostics.METRIC_SUITE_ID,
                predictions=(record,),
                metrics={"mae": 1.0},
                fold_metrics={0: {"mae": 1.0}},
                task_metrics={task_id: task_metric},
            )
            task_projection = diagnostics._task_metric_projection(result)
            seed_result = {
                "split_plan_id": split_plan_id,
                "prediction_count": 1,
                "prediction_projection_sha256": (
                    diagnostics.prediction_projection_sha256(result)
                ),
                "cohort_projection_sha256": (
                    diagnostics.cohort_projection_sha256(result)
                ),
                "metrics": {"mae": 1.0},
                "fold_metrics": {"0": {"mae": 1.0}},
                "task_metrics": task_projection,
                "comparability_key": [
                    "dataset:test",
                    split_plan_id,
                    "5" * 64,
                    "task_update",
                    target.value,
                    condition_id,
                    "task_max_conformal",
                    "0.1",
                    diagnostics.METRIC_SUITE_ID,
                ],
            }
            experiment = {
                "experiment_id": experiment_id,
                "position": "task_update",
                "target": target.value,
                "condition_id": condition_id,
                "calibrator_id": "task_max_conformal",
                "alpha": 0.1,
            }
            candidate = {
                "candidate_id": candidate_id,
                "candidate_hash": candidate_hash,
            }
            execution_key = {
                "experiment_id": experiment_id,
                "candidate_id": candidate_id,
                "candidate_hash": candidate_hash,
                "dataset_id": "dataset:test",
                "split_plan_id": split_plan_id,
                "split_seed": diagnostics.STAGE_SPLIT_SEEDS[0],
                "eligibility_hash": "5" * 64,
                "position": "task_update",
                "target": target.value,
                "condition_id": condition_id,
                "calibrator_id": "task_max_conformal",
                "alpha": 0.1,
                "source_provenance_hash": source_provenance_hash,
            }
            execution_hash = diagnostics._semantic_sha256(execution_key)
            checkpoint = (
                root
                / "workspace"
                / "stage4"
                / "checkpoints"
                / run_id
                / "candidates"
                / execution_hash
            )
            checkpoint.mkdir(parents=True)
            result_document = {
                "candidate_id": candidate_id,
                "candidate_hash": candidate_hash,
                "dataset_id": "dataset:test",
                "split_plan_id": split_plan_id,
                "eligibility_hash": "5" * 64,
                "position": "task_update",
                "target": target.value,
                "condition_id": condition_id,
                "calibrator_id": "task_max_conformal",
                "alpha": 0.1,
                "metric_suite_id": diagnostics.METRIC_SUITE_ID,
                "predictions": [
                    {
                        "candidate_id": candidate_id,
                        "point_id": "point-a",
                        "task_id": task_id,
                        "trajectory_id": "trajectory-a",
                        "condition_id": condition_id,
                        "fold": 0,
                        "target": target.value,
                        "forecast": {
                            "point_id": "point-a",
                            "target": target.value,
                            "lower": 1.0,
                            "point": 2.0,
                            "upper": 3.0,
                            "latency_ms": 0.1,
                            "overhead_input_tokens": 0,
                            "overhead_output_tokens": 0,
                            "raw_lower": 1.0,
                            "raw_point": 2.0,
                            "raw_upper": 3.0,
                        },
                        "sample_weight": 1.0,
                    }
                ],
                "metrics": {"mae": 1.0},
                "fold_metrics": {"0": {"mae": 1.0}},
                "task_metrics": {task_id: task_metric},
                "fold_artifacts": [
                    {
                        "fold": fold,
                        "encoder": None,
                        "fit_report": None,
                        "feature_importance": None,
                        "model_strings": None,
                        "bundle_files": None,
                        "calibrator": None,
                        "provenance": None,
                    }
                    for fold in range(5)
                ],
            }
            wrapper = {
                "checkpoint_schema_version": 1,
                "execution_key": execution_key,
                "result": result_document,
                "result_sha256": diagnostics._semantic_sha256(result_document),
            }
            (checkpoint / "candidate_result.json").write_text(
                json.dumps(wrapper, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            publish_artifact(
                checkpoint,
                stage_name="development_candidate_checkpoint",
                schema_version=1,
                metadata={
                    "candidate_execution_hash": execution_hash,
                    "run_id": run_id,
                    "run_semantic_sha256": run_semantic_sha256,
                },
            )
            development_pseudonym = diagnostics.hashlib.sha256(
                (
                    f"{diagnostics.DEVELOPMENT_TASK_PSEUDONYM_POLICY_ID}\0"
                    f"{task_id}"
                ).encode()
            ).hexdigest()
            artifact = diagnostics.VerifiedSourceArtifact(
                SimpleNamespace(
                    document={
                        "run_id": run_id,
                        "development_protocol": {
                            "permanent_holdout": {
                                "assignments": [
                                    {
                                        "task_pseudonym": development_pseudonym,
                                        "cohort": "development",
                                    }
                                ]
                            }
                        },
                    },
                    results_payload_sha256=source_results_sha256,
                ),
                SimpleNamespace(
                    metadata={
                        "run_id": run_id,
                        "run_semantic": source_run_semantic,
                        "results_payload_sha256": source_results_sha256,
                    }
                ),
            )
            parity = diagnostics._checkpoint_parity(
                root=root,
                artifact=artifact,
                experiment=experiment,
                candidate=candidate,
                split_seed=diagnostics.STAGE_SPLIT_SEEDS[0],
                seed_result=seed_result,
                source_provenance_hash=source_provenance_hash,
            )
            self.assertEqual(parity["status"], "exact")
            self.assertEqual(
                parity["development_cohort_status"], "development_only"
            )

            tampered = copy.deepcopy(seed_result)
            tampered["prediction_projection_sha256"] = "f" * 64
            with self.assertRaisesRegex(
                diagnostics.Stage4CompletionDiagnosticsError,
                "finalized seed result",
            ):
                diagnostics._checkpoint_parity(
                    root=root,
                    artifact=artifact,
                    experiment=experiment,
                    candidate=candidate,
                    split_seed=diagnostics.STAGE_SPLIT_SEEDS[0],
                    seed_result=tampered,
                    source_provenance_hash=source_provenance_hash,
                )

            tampered_comparability = copy.deepcopy(seed_result)
            tampered_comparability["comparability_key"][8] = "wrong-suite"
            with self.assertRaisesRegex(
                diagnostics.Stage4CompletionDiagnosticsError,
                "comparability",
            ):
                diagnostics._checkpoint_parity(
                    root=root,
                    artifact=artifact,
                    experiment=experiment,
                    candidate=candidate,
                    split_seed=diagnostics.STAGE_SPLIT_SEEDS[0],
                    seed_result=tampered_comparability,
                    source_provenance_hash=source_provenance_hash,
                )

            checkpoint_manifest = diagnostics.verify_artifact(checkpoint)
            extra_metadata_manifest = SimpleNamespace(
                stage_name=checkpoint_manifest.stage_name,
                schema_version=checkpoint_manifest.schema_version,
                files=checkpoint_manifest.files,
                metadata={**checkpoint_manifest.metadata, "extra": "forbidden"},
                artifact_id=checkpoint_manifest.artifact_id,
            )
            with (
                mock.patch.object(
                    diagnostics,
                    "verify_artifact",
                    return_value=extra_metadata_manifest,
                ),
                self.assertRaisesRegex(
                    diagnostics.Stage4CompletionDiagnosticsError,
                    "identity",
                ),
            ):
                diagnostics._checkpoint_parity(
                    root=root,
                    artifact=artifact,
                    experiment=experiment,
                    candidate=candidate,
                    split_seed=diagnostics.STAGE_SPLIT_SEEDS[0],
                    seed_result=seed_result,
                    source_provenance_hash=source_provenance_hash,
                )

            artifact.loaded.document["development_protocol"][
                "permanent_holdout"
            ]["assignments"][0]["cohort"] = "final_holdout"
            with self.assertRaisesRegex(
                diagnostics.Stage4CompletionDiagnosticsError,
                "final or unassigned",
            ):
                diagnostics._checkpoint_parity(
                    root=root,
                    artifact=artifact,
                    experiment=experiment,
                    candidate=candidate,
                    split_seed=diagnostics.STAGE_SPLIT_SEEDS[0],
                    seed_result=seed_result,
                    source_provenance_hash=source_provenance_hash,
                )

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

    def test_immutable_supplement_closes_under_its_v2_verifier(self) -> None:
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
            verified = diagnostics.verify_artifact(artifact)
            self.assertEqual(verified.artifact_id, manifest.artifact_id)
            self.assertEqual(
                diagnostics.verify_diagnostics_results_document(results),
                results["results_payload_sha256"],
            )

    def test_existing_artifact_rejects_jointly_reclosed_parity_tamper(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            expected = _results()
            tampered = copy.deepcopy(expected)
            parity = tampered["diagnostics"][0]["checkpoint_parity"]
            parity["prediction_projection_sha256"] = "f" * 64
            parity["expected_prediction_projection_sha256"] = "f" * 64
            _reclose(tampered)
            diagnostics.verify_diagnostics_results_document(tampered)
            (output / "results.json").write_text(
                json.dumps(tampered, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            run_semantic = {"fixture": "joint-tamper"}
            run_id = diagnostics._semantic_sha256(run_semantic)[:24]
            publish_artifact(
                output,
                stage_name=diagnostics.STAGE_NAME,
                schema_version=diagnostics.ARTIFACT_SCHEMA_VERSION,
                metadata={
                    "run_id": run_id,
                    "run_semantic": run_semantic,
                    "results_payload_sha256": tampered[
                        "results_payload_sha256"
                    ],
                },
            )
            with self.assertRaisesRegex(
                diagnostics.Stage4CompletionDiagnosticsError,
                "complete recomputation",
            ):
                diagnostics._load_existing(
                    output,
                    run_id=run_id,
                    run_semantic=run_semantic,
                    expected_results=expected,
                )


if __name__ == "__main__":
    unittest.main()
