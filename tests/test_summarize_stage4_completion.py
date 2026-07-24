from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.summarize_stage4_completion import (
    COMPLETION_DIAGNOSTICS_ARTIFACT_SCHEMA_VERSION,
    CompletionSummaryError,
    EXPECTED_FINAL_HOLDOUT,
    FROZEN_SPLIT_SEEDS,
    INTERVAL_RESERVE_FIELDS,
    POINT_ONLY_SEED_CANDIDATE_ID,
    POINT_ONLY_SEED_POLICY_ID,
    RAW_SEED_CANDIDATE_ID,
    RAW_SEED_POLICY_ID,
    RUN_DISPERSION_FIELDS,
    ArtifactReference,
    LoadedDiagnosticsArtifact,
    build_completion_summary,
    load_completion_diagnostics_artifact,
    load_development_artifact,
    render_markdown,
    resolve_artifact_references,
)

SOKOBAN_SOURCE = (
    "bagen_sokoban",
    "bagen_sokoban_dialogues_v1",
    ("condition:effa60eb1d4380d124bf",),
)
SWE_SOURCE = (
    "bagen_swebench",
    "bagen_swebench_traj_v2",
    (
        "condition:54cb50fce273f0aa2d74",
        "condition:949ac3b7a342718cd505",
        "condition:d94078c05d91b0d58aee",
        "condition:dce86ced00dc11c77205",
        "condition:f95ae2a5e11682f6b7fc",
    ),
)
SPEND_SOURCE = (
    "spend_openhands",
    "openhands_archive_trajectory_v3",
    ("condition:b407e0d1ec34f386ebc4",),
)
SEED_POLICY_SOURCE_GROUPS = (SOKOBAN_SOURCE, SWE_SOURCE, SPEND_SOURCE)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _metrics(
    mae: float,
    *,
    coverage: float = 0.9,
    task_simultaneous_coverage: float | None = None,
    interval_score: float = 10.0,
    upper_tail_rate: float = 0.1,
    n_points: int = 13,
    weight_sum: float = 10.0,
) -> dict[str, object]:
    return {
        "mae": mae,
        "coverage": coverage,
        "task_simultaneous_coverage": (
            coverage
            if task_simultaneous_coverage is None
            else task_simultaneous_coverage
        ),
        "interval_score": interval_score,
        "n_points": n_points,
        "weight_sum": weight_sum,
        "interval_diagnostics_id": "weighted_interval_tail_and_reserve_v1",
        "interval_below_truth_rate": 0.1,
        "interval_above_truth_rate": 0.2,
        "target_exceeds_upper_rate": upper_tail_rate,
        "mean_extra_reserved_tokens": 3.0,
        "raw_interval_below_truth_rate": 0.1,
        "raw_interval_above_truth_rate": 0.2,
        "raw_target_exceeds_upper_rate": 0.1,
        "raw_mean_extra_reserved_tokens": 2.0,
    }


def _task_metrics(
    *,
    mae: float,
    coverage: float,
    interval_score: float,
) -> list[dict[str, object]]:
    covered_task_count = round(coverage * 10)
    if not 0 <= covered_task_count <= 10 or not abs(
        coverage - covered_task_count / 10
    ) < 1e-12:
        raise ValueError("test task coverage must be an exact tenth")
    return [
        {
            "task_pseudonym": f"task-{chr(ord('a') + index)}",
            "n_points": (2, 3, 1, 1, 1, 1, 1, 1, 1, 1)[index],
            "n_trajectories": 1,
            "weight_sum": 1.0,
            "weighted_mae": mae + (-0.5 if index < 5 else 0.5),
            "weighted_coverage": float(index < covered_task_count),
            "weighted_interval_score": interval_score
            + (-0.5 if index < 5 else 0.5),
        }
        for index in range(10)
    ]


def _set_task_coverage(
    seed: dict[str, object],
    covered: list[bool] | tuple[bool, ...],
) -> None:
    rows = seed["task_metrics"]
    if len(rows) != len(covered):
        raise ValueError("test task coverage shape differs")
    for row, is_covered in zip(rows, covered, strict=True):
        row["weighted_coverage"] = float(is_covered)
    weight_sum = sum(float(row["weight_sum"]) for row in rows)
    seed["metrics"]["coverage"] = sum(
        float(row["weight_sum"]) * float(row["weighted_coverage"]) for row in rows
    ) / weight_sum
    seed["metrics"]["task_simultaneous_coverage"] = sum(covered) / len(covered)


def _dispersion() -> dict[str, object]:
    return {
        "run_variance_id": "same_task_run_mae_variance_v1",
        "run_dispersion_extension_id": "same_task_run_mae_iqr_max_minus_min_v1",
        "mean_within_task_run_mae_iqr": 1.0,
        "median_within_task_run_mae_iqr": 1.0,
        "max_within_task_run_mae_iqr": 1.0,
        "mean_within_task_run_mae_max_minus_min": 2.0,
        "median_within_task_run_mae_max_minus_min": 2.0,
        "max_within_task_run_mae_max_minus_min": 2.0,
    }


def _seed_result(
    candidate_id: str,
    split_seed: int,
    mae: float,
    *,
    lifecycle: bool = False,
    reference_id: str | None = None,
    reference_mae: float | None = None,
    lower: float = -2.0,
    upper: float = -0.5,
) -> dict[str, object]:
    value: dict[str, object] = {
        "candidate_id": candidate_id,
        "split_seed": split_seed,
        "metrics": _metrics(mae),
    }
    if lifecycle:
        value["stage4_evaluation"] = {
            "lifecycle": {"run_variance": _dispersion()}
        }
    if reference_id is not None:
        assert reference_mae is not None
        value["paired_vs_reference"] = {
            "candidate_id": candidate_id,
            "reference_id": reference_id,
            "candidate_mae": mae,
            "reference_mae": reference_mae,
            "mae_delta": mae - reference_mae,
            "mae_delta_ci_lower": lower,
            "mae_delta_ci_upper": upper,
            "candidate_win_probability": 0.9 if upper < 0 else 0.1,
        }
    return value


def _candidate(
    candidate_id: str,
    maes: tuple[float, float, float],
    *,
    lifecycle: bool = False,
    seed_policy_id: str = "none",
    reference_id: str | None = None,
    reference_maes: tuple[float, float, float] | None = None,
    axis: str = "method",
    upper: float = -0.5,
) -> dict[str, object]:
    candidate_hash = hashlib.sha256(f"candidate:{candidate_id}".encode()).hexdigest()
    feature_set_hash = hashlib.sha256(b"feature-set:none").hexdigest()
    return {
        "candidate_id": candidate_id,
        "candidate_hash": candidate_hash,
        "estimator_id": "cross_position_deduct" if lifecycle else candidate_id,
        "feature_set_hash": feature_set_hash,
        "feature_set_id": "none",
        "role": "ablation" if reference_id is not None else "model",
        "candidate_graph": {
            "initializer_estimator_id": "empirical_quantile" if lifecycle else "none",
            "updater_estimator_id": "cross_position_deduct" if lifecycle else candidate_id,
            "lifecycle_schema_id": "task_lifecycle_sequence_v1" if lifecycle else "point_cell_v1",
            "seed_policy_id": seed_policy_id,
            "inner_split_policy_id": "task_hash_inner_five_fold_v1" if lifecycle else "none",
        },
        "ablation": (
            {
                "reference_candidate_id": reference_id,
                "axis": axis,
                "allowed_config_paths": (
                    ["graph.seed_policy_id"] if axis == "seed_policy" else ["estimator_id"]
                ),
            }
            if reference_id is not None
            else None
        ),
        "seed_results": [
            _seed_result(
                candidate_id,
                split_seed,
                mae,
                lifecycle=lifecycle,
                reference_id=reference_id,
                reference_mae=reference_maes[index] if reference_maes else None,
                upper=upper,
            )
            for index, (split_seed, mae) in enumerate(
                zip(FROZEN_SPLIT_SEEDS, maes, strict=True)
            )
        ],
    }


def _call_experiment() -> dict[str, object]:
    reference_maes = (10.0, 11.0, 12.0)
    return {
        "experiment_id": "call-cell",
        "position": "call_pre",
        "target": "call_billable_total_tokens",
        "condition_id": "condition:call",
        "candidates": [
            _candidate("lightgbm_history", reference_maes),
            _candidate(
                "mlp_history",
                (9.0, 12.0, 12.0),
                reference_id="lightgbm_history",
                reference_maes=reference_maes,
                upper=0.2,
            ),
        ],
    }


def _seed_policy_experiment(
    condition_id: str = SOKOBAN_SOURCE[2][0],
    *,
    upper: float = 0.5,
) -> dict[str, object]:
    reference_maes = (10.0, 10.0, 10.0)
    experiment = {
        "experiment_id": f"seed-policy-cell-{condition_id.removeprefix('condition:')}",
        "position": "task_update",
        "target": "task_provider_accounted_remaining_tokens",
        "condition_id": condition_id,
        "alpha": 0.1,
        "calibrator_id": "task_max_conformal",
        "axis": None,
        "plan_role": "primary",
        "reference_experiment_id": None,
        "allowed_config_paths": [],
        "candidates": [
            _candidate(
                RAW_SEED_CANDIDATE_ID,
                reference_maes,
                lifecycle=True,
                seed_policy_id=RAW_SEED_POLICY_ID,
            ),
            _candidate(
                POINT_ONLY_SEED_CANDIDATE_ID,
                reference_maes,
                lifecycle=True,
                seed_policy_id=POINT_ONLY_SEED_POLICY_ID,
                reference_id=RAW_SEED_CANDIDATE_ID,
                reference_maes=reference_maes,
                axis="seed_policy",
                upper=upper,
            ),
        ],
    }
    reference, candidate = experiment["candidates"]
    for reference_seed, candidate_seed in zip(
        reference["seed_results"], candidate["seed_results"], strict=True
    ):
        split_seed = reference_seed["split_seed"]
        split_plan_id = hashlib.sha256(f"split:{split_seed}".encode()).hexdigest()
        cohort_projection_sha256 = hashlib.sha256(
            f"cohort:{split_seed}".encode()
        ).hexdigest()
        comparability_key = [
            hashlib.sha256(b"dataset").hexdigest(),
            split_plan_id,
            hashlib.sha256(b"input-contract").hexdigest(),
            "task_update",
            "task_provider_accounted_remaining_tokens",
            condition_id,
            "task_max_conformal",
            "0.1",
            "token_prediction_metrics_v2",
        ]
        shared_contract = {
            "comparability_key": comparability_key,
            "split_plan_id": split_plan_id,
            "cohort_projection_id": "stage4_prediction_cohort_projection_v1",
            "cohort_projection_sha256": cohort_projection_sha256,
            "prediction_count": 13,
            "task_metric_policy_id": "stage4_task_pseudonym_v1",
        }
        reference_seed.update(copy.deepcopy(shared_contract))
        candidate_seed.update(copy.deepcopy(shared_contract))
        reference_seed["metrics"] = _metrics(
            10.0,
            coverage=0.9,
            interval_score=10.0,
            upper_tail_rate=0.09,
        )
        candidate_seed["metrics"] = _metrics(
            10.0,
            coverage=0.9,
            interval_score=8.0,
            upper_tail_rate=0.10,
        )
        reference_seed["task_metrics"] = _task_metrics(
            mae=10.0,
            coverage=0.9,
            interval_score=10.0,
        )
        candidate_seed["task_metrics"] = _task_metrics(
            mae=10.0,
            coverage=0.9,
            interval_score=8.0,
        )
    return experiment


def _seed_policy_experiments(
    condition_ids: tuple[str, ...] = SOKOBAN_SOURCE[2],
) -> list[dict[str, object]]:
    return [_seed_policy_experiment(condition_id) for condition_id in condition_ids]


def _seed_policy_plan(experiment: dict[str, object]) -> dict[str, object]:
    plan_candidates = []
    for candidate in experiment["candidates"]:
        plan_candidates.append(
            {
                "candidate_id": candidate["candidate_id"],
                "candidate_hash": candidate["candidate_hash"],
                "estimator_id": candidate["estimator_id"],
                "feature_set_hash": candidate["feature_set_hash"],
                "feature_set_id": candidate["feature_set_id"],
                "role": candidate["role"],
                "ablation": copy.deepcopy(candidate["ablation"]),
                "graph": copy.deepcopy(candidate["candidate_graph"]),
            }
        )
    return {
        "role": experiment["plan_role"],
        "axis": experiment["axis"],
        "reference_experiment_id": experiment["reference_experiment_id"],
        "allowed_config_paths": copy.deepcopy(
            experiment["allowed_config_paths"]
        ),
        "spec": {
            "alpha": experiment["alpha"],
            "calibrator_id": experiment["calibrator_id"],
            "condition_id": experiment["condition_id"],
            "experiment_id": experiment["experiment_id"],
            "position": experiment["position"],
            "target": experiment["target"],
            "required_features": [],
            "candidates": plan_candidates,
        },
    }


def _results(
    source_name: str,
    run_id: str,
    experiments: list[dict[str, object]],
    *,
    source_id: str | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "results_schema_version": 1,
        "stage_name": "stage4_development_source",
        "run_policy_id": "stage4_source_three_seed_single_axis_v1",
        "artifact_layout_id": "stage4_compact_fold_artifact_layout_v1",
        "checkpoint_policy_id": "atomic_candidate_and_every_neural_epoch_v1",
        "run_id": run_id,
        "source": {
            "source_name": source_name,
            "source_id": source_id or f"{source_name}_test_source_v1",
        },
        "data_foundation": {},
        "code_binding": {},
        "runtime_versions": {},
        "dataset": {},
        "development_protocol": {},
        "matrix": {
            "plans": [
                _seed_policy_plan(experiment)
                for experiment in experiments
                if any(
                    candidate["candidate_id"] == POINT_ONLY_SEED_CANDIDATE_ID
                    for candidate in experiment["candidates"]
                )
            ]
        },
        "experiments": experiments,
        "matched_coverage_calibration": [],
        "paired_same_task_across_conditions": [],
        "summary": {},
        "final_holdout": dict(EXPECTED_FINAL_HOLDOUT),
    }
    value["results_payload_sha256"] = hashlib.sha256(
        _canonical_bytes(value)
    ).hexdigest()
    return value


def _publish(
    path: Path,
    source_name: str,
    run_id: str,
    experiments: list[dict[str, object]],
    *,
    source_id: str | None = None,
) -> str:
    path.mkdir(parents=True)
    results = _results(
        source_name,
        run_id,
        experiments,
        source_id=source_id,
    )
    results_payload = (
        json.dumps(
            results,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode()
        + b"\n"
    )
    metadata = {
        "run_id": run_id,
        "results_payload_sha256": results["results_payload_sha256"],
    }
    semantic = {
        "stage_name": "stage4_development_source",
        "schema_version": 1,
        "files": {"results.json": hashlib.sha256(results_payload).hexdigest()},
        "metadata": metadata,
    }
    artifact_id = hashlib.sha256(_canonical_bytes(semantic)).hexdigest()
    manifest = {"artifact_id": artifact_id, **semantic}
    (path / "results.json").write_bytes(results_payload)
    (path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    (path / "_SUCCESS").write_text(artifact_id + "\n", encoding="ascii")
    return artifact_id


def _publish_seed_source(
    path: Path,
    source: tuple[str, str, tuple[str, ...]],
    run_id: str,
    experiments: list[dict[str, object]] | None = None,
) -> str:
    return _publish(
        path,
        source[0],
        run_id,
        experiments if experiments is not None else _seed_policy_experiments(source[2]),
        source_id=source[1],
    )


def _write_diagnostics_artifact_shell(path: Path) -> bytes:
    path.mkdir(parents=True)
    payload = b"{}"
    (path / "results.json").write_bytes(payload)
    (path / "manifest.json").write_bytes(b"{}")
    (path / "_SUCCESS").write_bytes(b"placeholder\n")
    return payload


def _completion_release_document(
    artifact: Path,
    *,
    artifact_id: str,
    results: dict[str, object],
) -> dict[str, object]:
    source = results["source"]
    assert isinstance(source, dict)
    return {
        "release_schema_version": 2,
        "stage_name": "stage4_development_completion_supplement",
        "policy_id": "stage4_development_only_completion_release_v1",
        "release_control": {},
        "source_binding": {},
        "parent_final_release": {},
        "artifacts": [
            {
                "source_name": source["source_name"],
                "source_id": source["source_id"],
                "path": artifact.relative_to(artifact.parents[3]).as_posix(),
                "artifact_id": artifact_id,
                "run_id": results["run_id"],
                "results_payload_sha256": results["results_payload_sha256"],
                "manifest_sha256": hashlib.sha256(
                    (artifact / "manifest.json").read_bytes()
                ).hexdigest(),
                "matrix_id": "d" * 64,
                "experiment_count": 1,
                "candidate_seed_run_count": 1,
                "manifest_file_count": 1,
            }
        ],
        "diagnostics_artifact": {},
        "protocol": {
            "development_only": True,
            "final_holdout_evaluated": False,
            "final_holdout_prediction_count": 0,
            "final_holdout_target_values_used_for_fit_calibration_scoring": False,
            "final_holdout_selection_claim": "none",
        },
        "report": {},
    }


def _diagnostics_for_artifacts(artifacts: tuple) -> LoadedDiagnosticsArtifact:
    source_artifacts = []
    rows = []
    for artifact in artifacts:
        source_name = artifact.document["source"]["source_name"]
        source_artifacts.append(
            {
                "source_name": source_name,
                "artifact_id": artifact.artifact_id,
                "results_payload_sha256": artifact.results_payload_sha256,
            }
        )
        for experiment in artifact.document["experiments"]:
            lifecycle_candidates = [
                candidate
                for candidate in experiment["candidates"]
                if candidate["candidate_graph"]["initializer_estimator_id"] != "none"
            ]
            if not lifecycle_candidates:
                continue
            task_rows = lifecycle_candidates[0]["seed_results"][0]["task_metrics"]
            task_projection = hashlib.sha256(
                _canonical_bytes(
                    {
                        "policy_id": "sha256_development_task_pseudonym_v1",
                        "source_name": source_name,
                        "condition_id": experiment["condition_id"],
                        "task_pseudonyms": sorted(
                            row["task_pseudonym"] for row in task_rows
                        ),
                    }
                )
            ).hexdigest()
            for candidate in lifecycle_candidates:
                for seed in candidate["seed_results"]:
                    rows.append(
                        {
                            "source_name": source_name,
                            "condition_id": experiment["condition_id"],
                            "experiment_id": experiment["experiment_id"],
                            "candidate_id": candidate["candidate_id"],
                            "split_seed": seed["split_seed"],
                            "checkpoint_parity": {
                                "status": "exact",
                                "development_cohort_status": "development_only",
                                "development_task_count": len(task_rows),
                                "development_task_projection_sha256": task_projection,
                            },
                            "lifecycle_metrics": {
                                "status": "unavailable",
                                "reason_code": (
                                    "no_presealed_development_lifecycle_projection_v1"
                                ),
                                "labels_present": False,
                                "lifecycle_sequences_present": False,
                                "unavailable_metrics": [
                                    "progress",
                                    "run_variance_iqr_max_minus_min",
                                    "termination",
                                ],
                                "historical_stage3_reference": None,
                            },
                        }
                    )
    return LoadedDiagnosticsArtifact(
        path=Path("synthetic-diagnostics"),
        artifact_id="d" * 64,
        results_payload_sha256="e" * 64,
        document={
            "source_artifacts": source_artifacts,
            "diagnostics": rows,
        },
    )


class Stage4CompletionSummaryTests(unittest.TestCase):
    def test_comparisons_rule_coverage_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "workspace" / "stage4" / "runs"
            paths = [runs / f"run-{index}" for index in range(4)]
            _publish(
                paths[0],
                SOKOBAN_SOURCE[0],
                "sokoban-run",
                [_call_experiment(), *_seed_policy_experiments(SOKOBAN_SOURCE[2])],
                source_id=SOKOBAN_SOURCE[1],
            )
            _publish(
                paths[1],
                SWE_SOURCE[0],
                "swe-run",
                _seed_policy_experiments(SWE_SOURCE[2]),
                source_id=SWE_SOURCE[1],
            )
            _publish(
                paths[2],
                SPEND_SOURCE[0],
                "spend-run",
                _seed_policy_experiments(SPEND_SOURCE[2]),
                source_id=SPEND_SOURCE[1],
            )
            _publish(paths[3], "empty-b", "empty-run-b", [])
            references = resolve_artifact_references(
                [str(path) for path in paths],
                repo_root=root,
                development_runs_root=runs,
            )
            loaded = tuple(load_development_artifact(item) for item in references)
            summary = build_completion_summary(
                loaded,
                diagnostics=_diagnostics_for_artifacts(loaded),
            )

            call = summary["call_pre_mlp_vs_lightgbm"]
            self.assertEqual(len(call), 1)
            self.assertEqual(len(call[0]["seed_results"]), 3)
            self.assertEqual(call[0]["seed_results"][0]["mae_winner"], "candidate")
            self.assertNotIn("seed_outcomes", call[0])
            self.assertEqual(
                call[0]["mae_seed_outcomes_role"],
                "primary_comparison_evidence",
            )
            seed_policy = summary["seed_policy_point_only_vs_raw_repaired"]
            self.assertEqual(len(seed_policy), 7)
            self.assertTrue(seed_policy[0]["replacement_rule"]["replace_reference"])
            self.assertTrue(seed_policy[0]["seed_results"][0]["seed_pass"])
            self.assertNotIn("seed_outcomes", seed_policy[0])
            self.assertEqual(
                seed_policy[0]["mae_seed_outcomes_role"],
                "parity_report_only_not_selection",
            )
            self.assertEqual(
                seed_policy[0]["replacement_rule"]["coverage_metric"],
                "task_simultaneous_coverage",
            )
            self.assertEqual(
                seed_policy[0]["seed_results"][0][
                    "candidate_task_simultaneous_coverage"
                ],
                0.9,
            )
            self.assertEqual(
                seed_policy[0]["seed_results"][0]["bootstrap_iterations"],
                10_000,
            )
            self.assertEqual(
                seed_policy[0]["seed_results"][0]["mae_parity_role"],
                "reported_only_not_used_for_selection",
            )
            self.assertEqual(
                seed_policy[0]["seed_results"][0][
                    "raw_forecast_interval_metric_role"
                ],
                "diagnostic_only_not_used_for_selection",
            )
            self.assertNotIn("raw_metric_role", seed_policy[0]["seed_results"][0])
            self.assertEqual(
                seed_policy[0]["replacement_rule"][
                    "raw_forecast_interval_metric_role"
                ],
                "diagnostic_only_not_selection",
            )
            self.assertNotIn("raw_metric_role", seed_policy[0]["replacement_rule"])
            self.assertEqual(
                summary["seed_policy_frozen_replacement_rule"]["decision"],
                "prospectively_replace_raw_repaired_reference",
            )
            frozen_rule = summary["seed_policy_frozen_replacement_rule"]
            self.assertTrue(frozen_rule["condition_set_complete"])
            self.assertEqual(
                frozen_rule["coverage_aggregation"],
                "equal_weight_per_task_all_points_covered",
            )
            self.assertEqual(
                frozen_rule["raw_forecast_interval_metric_role"],
                "diagnostic_only_not_selection",
            )
            self.assertNotIn("raw_metric_role", frozen_rule)
            self.assertEqual(frozen_rule["missing_semantic_cells"], [])
            self.assertEqual(
                {
                    (
                        cell["source_name"],
                        cell["source_id"],
                        cell["condition_id"],
                        cell["axis"],
                        cell["position"],
                        cell["target"],
                        cell["calibrator_id"],
                        cell["alpha"],
                    )
                    for cell in frozen_rule["observed_semantic_cells"]
                },
                {
                    (
                        source_name,
                        source_id,
                        condition_id,
                        None,
                        "task_update",
                        "task_provider_accounted_remaining_tokens",
                        "task_max_conformal",
                        0.1,
                    )
                    for source_name, source_id, conditions in SEED_POLICY_SOURCE_GROUPS
                    for condition_id in conditions
                },
            )
            self.assertFalse(
                summary["seed_policy_frozen_replacement_rule"][
                    "parent_final_reselected"
                ]
            )
            coverage = summary["metric_coverage"]
            self.assertTrue(coverage["interval_reserve"]["complete"])
            self.assertTrue(coverage["repeated_run_dispersion"]["complete"])
            self.assertFalse(summary["final_holdout_access"]["accessed"])
            self.assertEqual(
                summary["final_holdout_access"]["attestation_scope"],
                "this_summary_reader_process_only",
            )
            self.assertTrue(
                summary["final_holdout_access"]["upstream_loader_audit_disclosure"][
                    "raw_final_task_records_may_have_been_parsed"
                ]
            )
            markdown = render_markdown(summary)
            self.assertIn("Call-pre MLP vs LightGBM", markdown)
            self.assertIn("Final holdout (summary-reader scope)", markdown)
            self.assertIn("mixed raw source payloads", markdown)
            self.assertIn("20260719:", markdown)
            self.assertIn("task simultaneous coverage delta=", markdown)
            self.assertIn("parent final selection is unchanged", markdown)

    def test_seed_policy_rejects_cross_seed_task_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workspace" / "stage4" / "runs" / "run"
            _publish_seed_source(path, SOKOBAN_SOURCE, "run")
            loaded = load_development_artifact(ArtifactReference(path))
            supplement = _diagnostics_for_artifacts((loaded,))
            supplement.document["diagnostics"][0]["checkpoint_parity"][
                "development_task_projection_sha256"
            ] = "f" * 64
            with self.assertRaisesRegex(
                CompletionSummaryError,
                "development task identity projection",
            ):
                build_completion_summary((loaded,), diagnostics=supplement)

    def test_seed_policy_rejects_unpaired_task_clusters(self) -> None:
        mutations = {
            "pseudonym sets differ": lambda candidate: candidate["task_metrics"][
                0
            ].update({"task_pseudonym": "task-z"}),
            "n_points differ": lambda candidate: candidate["task_metrics"][0].update(
                {"n_points": 99}
            ),
            "n_trajectories differ": lambda candidate: candidate["task_metrics"][
                0
            ].update({"n_trajectories": 2}),
            "must be a positive integer": lambda candidate: candidate["task_metrics"][
                0
            ].update({"n_trajectories": 0}),
            "weight_sum differs": lambda candidate: candidate["task_metrics"][
                0
            ].update({"weight_sum": 2.0}),
        }
        for expected, mutate in mutations.items():
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                path = (
                    Path(directory)
                    / "workspace"
                    / "stage4"
                    / "runs"
                    / "run"
                )
                experiment = _seed_policy_experiment()
                candidate = experiment["candidates"][1]["seed_results"][0]
                mutate(candidate)
                _publish_seed_source(path, SOKOBAN_SOURCE, "run", [experiment])
                loaded = load_development_artifact(ArtifactReference(path))
                with self.assertRaisesRegex(CompletionSummaryError, expected):
                    build_completion_summary((loaded,))

    def test_seed_policy_rejects_seed_comparability_field_tampering(self) -> None:
        mutations = (
            (
                "comparability-key",
                "comparability_key semantic suffix differs",
                lambda seed: seed["comparability_key"].__setitem__(
                    8, "other_metric_suite"
                ),
            ),
            (
                "split-plan",
                "comparability_key does not bind split_plan_id",
                lambda seed: seed.update({"split_plan_id": "f" * 64}),
            ),
            (
                "cohort-projection",
                "paired seed comparability fields differ",
                lambda seed: seed.update({"cohort_projection_sha256": "e" * 64}),
            ),
            (
                "prediction-count",
                "paired seed comparability fields differ",
                lambda seed: seed.update({"prediction_count": 6}),
            ),
            (
                "task-policy",
                "task_metric_policy_id differs",
                lambda seed: seed.update(
                    {"task_metric_policy_id": "other_task_policy"}
                ),
            ),
        )
        for case, expected, mutate in mutations:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "workspace" / "stage4" / "runs" / "run"
                experiment = _seed_policy_experiment()
                mutate(experiment["candidates"][1]["seed_results"][0])
                _publish_seed_source(path, SOKOBAN_SOURCE, "run", [experiment])
                loaded = load_development_artifact(ArtifactReference(path))
                with self.assertRaisesRegex(CompletionSummaryError, expected):
                    build_completion_summary((loaded,))

    def test_seed_policy_rejects_cross_seed_task_cohort_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workspace" / "stage4" / "runs" / "run"
            experiment = _seed_policy_experiment()
            for candidate in experiment["candidates"]:
                task_metrics = candidate["seed_results"][1]["task_metrics"]
                task_metrics[0]["n_points"] = 1
                task_metrics[1]["n_points"] = 4
            _publish_seed_source(path, SOKOBAN_SOURCE, "run", [experiment])
            loaded = load_development_artifact(ArtifactReference(path))
            with self.assertRaisesRegex(
                CompletionSummaryError, "cohort differs across split seeds"
            ):
                build_completion_summary((loaded,))

    def test_seed_policy_rejects_non_integer_split_seed(self) -> None:
        for invalid in (float(FROZEN_SPLIT_SEEDS[0]), True):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "workspace" / "stage4" / "runs" / "run"
                experiment = _seed_policy_experiment()
                for candidate in experiment["candidates"]:
                    candidate["seed_results"][0]["split_seed"] = invalid
                _publish_seed_source(path, SOKOBAN_SOURCE, "run", [experiment])
                loaded = load_development_artifact(ArtifactReference(path))
                with self.assertRaisesRegex(
                    CompletionSummaryError, "must be a positive integer"
                ):
                    build_completion_summary((loaded,))

    def test_seed_policy_rejects_semantic_cell_and_matrix_tampering(self) -> None:
        cases = ("semantic_cell", "axis", "matrix")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "workspace" / "stage4" / "runs" / "run"
                experiment = _seed_policy_experiment()
                if case == "semantic_cell":
                    condition_id = "condition:00000000000000000000"
                    experiment["condition_id"] = condition_id
                    for candidate in experiment["candidates"]:
                        for seed in candidate["seed_results"]:
                            seed["comparability_key"][5] = condition_id
                elif case == "axis":
                    experiment["axis"] = "seed_policy"
                results = _results(
                    SOKOBAN_SOURCE[0],
                    "run",
                    [experiment],
                    source_id=SOKOBAN_SOURCE[1],
                )
                if case == "semantic_cell":
                    results["matrix"]["plans"][0]["spec"][
                        "condition_id"
                    ] = condition_id
                elif case == "axis":
                    results["matrix"]["plans"][0]["axis"] = "seed_policy"
                else:
                    results["matrix"]["plans"][0]["spec"]["target"] = "tampered"
                results_without_digest = dict(results)
                results_without_digest.pop("results_payload_sha256")
                results["results_payload_sha256"] = hashlib.sha256(
                    _canonical_bytes(results_without_digest)
                ).hexdigest()
                _publish_seed_source(path, SOKOBAN_SOURCE, "run", [experiment])
                # Republish the intentionally self-consistent results document.
                results_payload = (
                    json.dumps(results, sort_keys=True, indent=2).encode() + b"\n"
                )
                (path / "results.json").write_bytes(results_payload)
                manifest = json.loads(
                    (path / "manifest.json").read_text(encoding="utf-8")
                )
                manifest["files"]["results.json"] = hashlib.sha256(
                    results_payload
                ).hexdigest()
                manifest["metadata"]["results_payload_sha256"] = results[
                    "results_payload_sha256"
                ]
                semantic = {
                    key: manifest[key]
                    for key in ("stage_name", "schema_version", "files", "metadata")
                }
                manifest["artifact_id"] = hashlib.sha256(
                    _canonical_bytes(semantic)
                ).hexdigest()
                (path / "manifest.json").write_text(
                    json.dumps(manifest, sort_keys=True), encoding="utf-8"
                )
                (path / "_SUCCESS").write_text(
                    manifest["artifact_id"] + "\n", encoding="ascii"
                )
                loaded = load_development_artifact(ArtifactReference(path))
                expected = (
                    "not a frozen seed-policy cell"
                    if case == "semantic_cell"
                    else "axis differs"
                    if case == "axis"
                    else "matrix plan spec.target does not close"
                )
                with self.assertRaisesRegex(CompletionSummaryError, expected):
                    build_completion_summary((loaded,))

    def test_seed_policy_coverage_and_upper_tail_guards(self) -> None:
        cases = (
            "minimum_task_coverage",
            "aggregate_task_coverage_delta",
            "task_coverage_ci_lower",
            "relative_upper_tail",
            "absolute_upper_tail",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                path = (
                    Path(directory)
                    / "workspace"
                    / "stage4"
                    / "runs"
                    / "run"
                )
                experiment = _seed_policy_experiment()
                candidate = experiment["candidates"][1]["seed_results"][0]
                reference = experiment["candidates"][0]["seed_results"][0]
                if case == "minimum_task_coverage":
                    _set_task_coverage(candidate, [True] * 8 + [False] * 2)
                    _set_task_coverage(reference, [True] * 8 + [False] * 2)
                elif case == "aggregate_task_coverage_delta":
                    _set_task_coverage(reference, [True] * 8 + [False] * 2)
                elif case == "task_coverage_ci_lower":
                    _set_task_coverage(candidate, [False] + [True] * 9)
                    _set_task_coverage(
                        reference,
                        [True, False] + [True] * 8,
                    )
                elif case == "relative_upper_tail":
                    candidate["metrics"]["target_exceeds_upper_rate"] = 0.11
                    reference["metrics"]["target_exceeds_upper_rate"] = 0.08
                else:
                    candidate["metrics"]["target_exceeds_upper_rate"] = 0.13
                    reference["metrics"]["target_exceeds_upper_rate"] = 0.13
                _publish_seed_source(path, SOKOBAN_SOURCE, "run", [experiment])
                loaded = load_development_artifact(ArtifactReference(path))
                summary = build_completion_summary((loaded,))
                seed = summary["seed_policy_point_only_vs_raw_repaired"][0][
                    "seed_results"
                ][0]
                self.assertFalse(seed["seed_pass"])
                guards = seed["selection_guards"]
                if case == "minimum_task_coverage":
                    self.assertFalse(
                        guards[
                            "candidate_task_simultaneous_coverage_at_least_"
                            "nominal_minus_tolerance"
                        ]
                    )
                    self.assertTrue(
                        guards[
                            "aggregate_task_simultaneous_coverage_delta_"
                            "within_tolerance"
                        ]
                    )
                    self.assertTrue(
                        guards[
                            "task_simultaneous_coverage_delta_ci_lower_at_"
                            "least_negative_tolerance"
                        ]
                    )
                elif case == "aggregate_task_coverage_delta":
                    self.assertTrue(
                        guards[
                            "candidate_task_simultaneous_coverage_at_least_"
                            "nominal_minus_tolerance"
                        ]
                    )
                    self.assertFalse(
                        guards[
                            "aggregate_task_simultaneous_coverage_delta_"
                            "within_tolerance"
                        ]
                    )
                    self.assertTrue(
                        guards[
                            "task_simultaneous_coverage_delta_ci_lower_at_"
                            "least_negative_tolerance"
                        ]
                    )
                elif case == "task_coverage_ci_lower":
                    self.assertTrue(
                        guards[
                            "candidate_task_simultaneous_coverage_at_least_"
                            "nominal_minus_tolerance"
                        ]
                    )
                    self.assertTrue(
                        guards[
                            "aggregate_task_simultaneous_coverage_delta_"
                            "within_tolerance"
                        ]
                    )
                    self.assertFalse(
                        guards[
                            "task_simultaneous_coverage_delta_ci_lower_at_"
                            "least_negative_tolerance"
                        ]
                    )
                elif case == "relative_upper_tail":
                    self.assertFalse(
                        guards["upper_tail_rate_not_worse_beyond_tolerance"]
                    )
                else:
                    self.assertTrue(
                        guards["upper_tail_rate_not_worse_beyond_tolerance"]
                    )
                    self.assertFalse(
                        guards[
                            "candidate_upper_tail_at_most_alpha_plus_tolerance"
                        ]
                    )

    def test_real_openhands_task_coverage_counts_block_replacement(self) -> None:
        experiment = _seed_policy_experiment(SPEND_SOURCE[2][0])
        # Captured from the three real OpenHands paired task cohorts as
        # (candidate covered, reference covered) counts.
        paired_counts = (
            {(False, False): 35, (False, True): 6, (True, False): 2, (True, True): 354},
            {(False, False): 38, (False, True): 4, (True, False): 4, (True, True): 351},
            {(False, False): 36, (False, True): 1, (True, False): 2, (True, True): 358},
        )
        reference_seeds = experiment["candidates"][0]["seed_results"]
        candidate_seeds = experiment["candidates"][1]["seed_results"]
        for candidate_seed, reference_seed, counts in zip(
            candidate_seeds,
            reference_seeds,
            paired_counts,
            strict=True,
        ):
            pairs = [
                pair
                for pair, count in counts.items()
                for _ in range(count)
            ]
            self.assertEqual(len(pairs), 397)
            for seed, side, interval_score in (
                (candidate_seed, 0, 8.0),
                (reference_seed, 1, 10.0),
            ):
                covered = [pair[side] for pair in pairs]
                seed["prediction_count"] = len(pairs)
                seed["task_metrics"] = [
                    {
                        "task_pseudonym": f"task-{index:03d}",
                        "n_points": 1,
                        "n_trajectories": 1,
                        "weight_sum": 1.0,
                        "weighted_mae": 10.0,
                        "weighted_coverage": float(is_covered),
                        "weighted_interval_score": interval_score,
                    }
                    for index, is_covered in enumerate(covered)
                ]
                seed["metrics"] = _metrics(
                    10.0,
                    coverage=sum(covered) / len(covered),
                    task_simultaneous_coverage=sum(covered) / len(covered),
                    interval_score=interval_score,
                    upper_tail_rate=0.10 if side == 0 else 0.09,
                    n_points=len(pairs),
                    weight_sum=float(len(pairs)),
                )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workspace" / "stage4" / "runs" / "openhands"
            _publish_seed_source(
                path,
                SPEND_SOURCE,
                "real-openhands-coverage-regression",
                [experiment],
            )
            loaded = load_development_artifact(ArtifactReference(path))
            comparison = build_completion_summary((loaded,))[
                "seed_policy_point_only_vs_raw_repaired"
            ][0]

        first_seed = comparison["seed_results"][0]
        self.assertLessEqual(
            abs(first_seed["task_simultaneous_coverage_delta"]),
            0.02,
        )
        self.assertLess(
            first_seed["task_simultaneous_coverage_delta_ci_lower"],
            -0.02,
        )
        self.assertFalse(
            first_seed["selection_guards"][
                "task_simultaneous_coverage_delta_ci_lower_at_"
                "least_negative_tolerance"
            ]
        )
        self.assertFalse(first_seed["seed_pass"])
        self.assertFalse(comparison["replacement_rule"]["replace_reference"])
        self.assertEqual(
            comparison["mae_seed_outcomes_role"],
            "parity_report_only_not_selection",
        )
        self.assertNotIn("seed_outcomes", comparison)

    def test_seed_policy_bootstrap_seed_is_identity_bound_and_deterministic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory) / "workspace" / "stage4" / "runs"
            sokoban_path = runs / "sokoban"
            swe_path = runs / "swe"
            _publish_seed_source(sokoban_path, SOKOBAN_SOURCE, "sokoban-run")
            _publish_seed_source(
                swe_path,
                SWE_SOURCE,
                "swe-run",
                _seed_policy_experiments((SWE_SOURCE[2][0],)),
            )
            loaded = (
                load_development_artifact(ArtifactReference(sokoban_path)),
                load_development_artifact(ArtifactReference(swe_path)),
            )
            first = build_completion_summary(loaded)
            second = build_completion_summary(loaded)
            first_comparisons = first[
                "seed_policy_point_only_vs_raw_repaired"
            ]
            second_comparisons = second[
                "seed_policy_point_only_vs_raw_repaired"
            ]
            first_seed = first_comparisons[0]["seed_results"][0]
            repeated_seed = second_comparisons[0]["seed_results"][0]
            different_identity_seed = first_comparisons[1]["seed_results"][0]
            self.assertEqual(
                first_seed["bootstrap_seed_derivation_policy_id"],
                "stage4-seed-policy-interval-and-task-coverage-bootstrap-v3",
            )
            expected_material = {
                "policy_id": (
                    "stage4-seed-policy-interval-and-task-coverage-bootstrap-v3"
                ),
                "split_seed": FROZEN_SPLIT_SEEDS[0],
                "experiment_id": (
                    "seed-policy-cell-effa60eb1d4380d124bf"
                ),
                "candidate_id": POINT_ONLY_SEED_CANDIDATE_ID,
                "reference_id": RAW_SEED_CANDIDATE_ID,
                "source_name": SOKOBAN_SOURCE[0],
                "source_id": SOKOBAN_SOURCE[1],
                "condition_id": SOKOBAN_SOURCE[2][0],
                "position": "task_update",
                "target": "task_provider_accounted_remaining_tokens",
                "calibrator_id": "task_max_conformal",
                "alpha": 0.1,
                "paired_task_cohort_sha256": first_seed[
                    "paired_task_cohort_sha256"
                ],
            }
            expected_material_sha256 = hashlib.sha256(
                _canonical_bytes(expected_material)
            ).hexdigest()
            expected_seed = int(expected_material_sha256[:16], 16)
            self.assertEqual(
                first_seed["bootstrap_seed_material_sha256"],
                expected_material_sha256,
            )
            self.assertEqual(first_seed["bootstrap_random_seed"], expected_seed)
            self.assertEqual(
                first_seed["bootstrap_random_seed"],
                repeated_seed["bootstrap_random_seed"],
            )
            self.assertEqual(
                first_seed["interval_score_delta_ci_upper"],
                repeated_seed["interval_score_delta_ci_upper"],
            )
            self.assertEqual(
                first_seed["task_simultaneous_coverage_delta_ci_lower"],
                repeated_seed["task_simultaneous_coverage_delta_ci_lower"],
            )
            self.assertNotEqual(
                first_seed["bootstrap_random_seed"],
                different_identity_seed["bootstrap_random_seed"],
            )

    def test_seed_policy_interval_score_win_and_one_seed_loss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            passing_path = root / "workspace" / "stage4" / "runs" / "passing"
            passing = _seed_policy_experiment()
            _publish_seed_source(
                passing_path, SOKOBAN_SOURCE, "run-pass", [passing]
            )
            passing_summary = build_completion_summary(
                (load_development_artifact(ArtifactReference(passing_path)),)
            )
            passing_seed = passing_summary[
                "seed_policy_point_only_vs_raw_repaired"
            ][0]["seed_results"][0]
            self.assertLess(passing_seed["interval_score_delta_ci_upper"], 0)
            self.assertTrue(
                passing_seed["selection_guards"][
                    "interval_score_delta_ci_upper_below_zero"
                ]
            )
            self.assertEqual(passing_seed["mae_delta"], 0.0)

            losing_path = root / "workspace" / "stage4" / "runs" / "losing"
            losing = _seed_policy_experiment()
            losing_seed = losing["candidates"][1]["seed_results"][1]
            losing_seed["metrics"]["interval_score"] = 12.0
            losing_seed["task_metrics"] = _task_metrics(
                mae=10.0,
                coverage=0.9,
                interval_score=12.0,
            )
            _publish_seed_source(
                losing_path, SOKOBAN_SOURCE, "run-lose", [losing]
            )
            losing_summary = build_completion_summary(
                (load_development_artifact(ArtifactReference(losing_path)),)
            )
            comparison = losing_summary[
                "seed_policy_point_only_vs_raw_repaired"
            ][0]
            self.assertFalse(comparison["seed_results"][1]["seed_pass"])
            self.assertFalse(
                comparison["replacement_rule"]["all_three_seeds_pass"]
            )
            self.assertFalse(comparison["replacement_rule"]["replace_reference"])

    def test_seed_policy_requires_all_seven_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "workspace" / "stage4" / "runs"
            incomplete_paths = (runs / "incomplete-sokoban", runs / "incomplete-swe")
            _publish_seed_source(
                incomplete_paths[0], SOKOBAN_SOURCE, "incomplete-sokoban"
            )
            _publish_seed_source(
                incomplete_paths[1], SWE_SOURCE, "incomplete-swe"
            )
            incomplete = build_completion_summary(
                tuple(
                    load_development_artifact(ArtifactReference(path))
                    for path in incomplete_paths
                )
            )["seed_policy_frozen_replacement_rule"]
            self.assertFalse(incomplete["condition_set_complete"])
            self.assertEqual(len(incomplete["missing_semantic_cells"]), 1)
            self.assertEqual(
                incomplete["decision"],
                "retain_raw_repaired_reference_for_prospective_runs",
            )

            failing_paths = tuple(
                runs / f"failing-{source[0]}" for source in SEED_POLICY_SOURCE_GROUPS
            )
            failing_groups = [
                _seed_policy_experiments(source[2])
                for source in SEED_POLICY_SOURCE_GROUPS
            ]
            losing_seed = failing_groups[-1][0]["candidates"][1]["seed_results"][2]
            losing_seed["metrics"]["interval_score"] = 12.0
            losing_seed["task_metrics"] = _task_metrics(
                mae=10.0,
                coverage=0.9,
                interval_score=12.0,
            )
            for path, source, experiments in zip(
                failing_paths,
                SEED_POLICY_SOURCE_GROUPS,
                failing_groups,
                strict=True,
            ):
                _publish_seed_source(
                    path,
                    source,
                    f"failing-{source[0]}",
                    experiments,
                )
            failing_loaded = tuple(
                    load_development_artifact(ArtifactReference(path))
                    for path in failing_paths
                )
            failing = build_completion_summary(
                failing_loaded,
                diagnostics=_diagnostics_for_artifacts(failing_loaded),
            )["seed_policy_frozen_replacement_rule"]
            self.assertTrue(failing["condition_set_complete"])
            self.assertEqual(failing["passing_condition_count"], 6)
            self.assertFalse(failing["all_conditions_pass"])
            self.assertEqual(
                failing["decision"],
                "retain_raw_repaired_reference_for_prospective_runs",
            )

    def test_missing_metrics_are_reported_not_zero_filled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workspace" / "stage4" / "runs" / "run"
            experiment = _seed_policy_experiment()
            del experiment["candidates"][0]["seed_results"][0]["metrics"][
                INTERVAL_RESERVE_FIELDS[1]
            ]
            del experiment["candidates"][1]["seed_results"][0]["stage4_evaluation"][
                "lifecycle"
            ]["run_variance"][RUN_DISPERSION_FIELDS[1]]
            _publish_seed_source(path, SOKOBAN_SOURCE, "run", [experiment])
            loaded = load_development_artifact(ArtifactReference(path))
            summary = build_completion_summary((loaded,))
            artifact_coverage = summary["metric_coverage"]["artifacts"][0]
            self.assertEqual(
                artifact_coverage["interval_reserve_missing"][0]["missing_fields"],
                [INTERVAL_RESERVE_FIELDS[1]],
            )
            self.assertEqual(
                artifact_coverage["run_dispersion_missing"][0]["missing_fields"],
                [RUN_DISPERSION_FIELDS[1]],
            )
            self.assertFalse(summary["completion_status"]["metric_coverage_complete"])

    def test_diagnostics_loader_accepts_only_the_exact_runner_topology(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            diagnostics_root = (
                repo_root / "workspace" / "stage4" / "completion_diagnostics"
            )
            artifact = diagnostics_root / "artifact"
            payload = _write_diagnostics_artifact_shell(artifact)
            results_digest = "a" * 64
            manifest = SimpleNamespace(
                stage_name="stage4_completion_diagnostics",
                schema_version=COMPLETION_DIAGNOSTICS_ARTIFACT_SCHEMA_VERSION,
                files={"results.json": hashlib.sha256(payload).hexdigest()},
                metadata={"results_payload_sha256": results_digest},
                artifact_id="b" * 64,
            )
            with (
                mock.patch(
                    "scripts.summarize_stage4_completion.verify_artifact",
                    return_value=manifest,
                ),
                mock.patch(
                    "scripts.run_stage4_completion_diagnostics."
                    "verify_diagnostics_results_document",
                    return_value=results_digest,
                ),
            ):
                loaded = load_completion_diagnostics_artifact(
                    artifact,
                    repo_root=repo_root,
                    diagnostics_root=diagnostics_root,
                )
            self.assertEqual(loaded.path, artifact)
            self.assertEqual(loaded.document, {})

    def test_diagnostics_loader_rejects_extra_files_and_directories_pre_hash(
        self,
    ) -> None:
        for extra_kind in ("regular_file", "directory"):
            with (
                self.subTest(extra_kind=extra_kind),
                tempfile.TemporaryDirectory() as directory,
            ):
                repo_root = Path(directory)
                diagnostics_root = (
                    repo_root / "workspace" / "stage4" / "completion_diagnostics"
                )
                artifact = diagnostics_root / "artifact"
                _write_diagnostics_artifact_shell(artifact)
                extra = artifact / "copied-final-labels.bin"
                if extra_kind == "regular_file":
                    extra.write_bytes(b"secret")
                else:
                    extra.mkdir()
                with mock.patch(
                    "scripts.summarize_stage4_completion.verify_artifact",
                    side_effect=AssertionError(
                        "generic verifier must not hash unexpected topology"
                    ),
                ) as verifier:
                    with self.assertRaisesRegex(
                        CompletionSummaryError,
                        "physical topology differs",
                    ):
                        load_completion_diagnostics_artifact(
                            artifact,
                            repo_root=repo_root,
                            diagnostics_root=diagnostics_root,
                        )
                verifier.assert_not_called()

    def test_diagnostics_loader_rejects_lexically_unsafe_ancestors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            diagnostics_root = (
                repo_root / "workspace" / "stage4" / "completion_diagnostics"
            )
            artifact = diagnostics_root / "artifact"
            _write_diagnostics_artifact_shell(artifact)
            unsafe_ancestor = diagnostics_root.parent
            with (
                mock.patch(
                    "scripts.summarize_stage4_completion._is_link_or_reparse",
                    side_effect=lambda path: path == unsafe_ancestor,
                ),
                mock.patch(
                    "scripts.summarize_stage4_completion.verify_artifact",
                    side_effect=AssertionError("verifier must not run"),
                ) as verifier,
            ):
                with self.assertRaisesRegex(
                    CompletionSummaryError,
                    "traverses a symlink, junction, or reparse point",
                ):
                    load_completion_diagnostics_artifact(
                        artifact,
                        repo_root=repo_root,
                        diagnostics_root=diagnostics_root,
                    )
            verifier.assert_not_called()

    def test_diagnostics_loader_rejects_hard_linked_required_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            diagnostics_root = (
                repo_root / "workspace" / "stage4" / "completion_diagnostics"
            )
            artifact = diagnostics_root / "artifact"
            artifact.mkdir(parents=True)
            outside_payload = repo_root / "outside-results.json"
            outside_payload.write_bytes(b"{}")
            os.link(outside_payload, artifact / "results.json")
            (artifact / "manifest.json").write_bytes(b"{}")
            (artifact / "_SUCCESS").write_bytes(b"placeholder\n")
            with mock.patch(
                "scripts.summarize_stage4_completion.verify_artifact",
                side_effect=AssertionError("verifier must not hash hard links"),
            ) as verifier:
                with self.assertRaisesRegex(
                    CompletionSummaryError,
                    "physical topology differs",
                ):
                    load_completion_diagnostics_artifact(
                        artifact,
                        repo_root=repo_root,
                        diagnostics_root=diagnostics_root,
                    )
            verifier.assert_not_called()

    def test_diagnostics_v2_unavailable_lifecycle_is_reported_not_imputed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workspace" / "stage4" / "runs" / "run"
            experiment = _seed_policy_experiment()
            for candidate in experiment["candidates"]:
                for seed in candidate["seed_results"]:
                    seed.pop("stage4_evaluation")
            artifact_id = _publish_seed_source(
                path, SOKOBAN_SOURCE, "run", [experiment]
            )
            loaded = load_development_artifact(ArtifactReference(path))
            supplement_rows = []
            development_task_projection = hashlib.sha256(
                b"synthetic-development-task-projection"
            ).hexdigest()
            for candidate in experiment["candidates"]:
                for seed in candidate["seed_results"]:
                    supplement_rows.append(
                        {
                            "source_name": SOKOBAN_SOURCE[0],
                            "condition_id": experiment["condition_id"],
                            "experiment_id": experiment["experiment_id"],
                            "candidate_id": candidate["candidate_id"],
                            "split_seed": seed["split_seed"],
                            "checkpoint_parity": {
                                "status": "exact",
                                "development_cohort_status": "development_only",
                                "development_task_count": 10,
                                "development_task_projection_sha256": (
                                    development_task_projection
                                ),
                            },
                            "lifecycle_metrics": {
                                "status": "unavailable",
                                "reason_code": (
                                    "no_presealed_development_lifecycle_projection_v1"
                                ),
                                "labels_present": False,
                                "lifecycle_sequences_present": False,
                                "unavailable_metrics": [
                                    "progress",
                                    "run_variance_iqr_max_minus_min",
                                    "termination",
                                ],
                                "historical_stage3_reference": None,
                            },
                        }
                    )
            supplement = LoadedDiagnosticsArtifact(
                path=Path(directory) / "diagnostics",
                artifact_id="d" * 64,
                results_payload_sha256="e" * 64,
                document={
                    "source_artifacts": [
                        {
                            "source_name": SOKOBAN_SOURCE[0],
                            "artifact_id": artifact_id,
                            "results_payload_sha256": (
                                loaded.results_payload_sha256
                            ),
                        }
                    ],
                    "diagnostics": supplement_rows,
                },
            )
            summary = build_completion_summary(
                (loaded,),
                diagnostics=supplement,
            )
            coverage = summary["metric_coverage"]["repeated_run_dispersion"]
            self.assertFalse(coverage["complete"])
            self.assertEqual(coverage["complete_count"], 0)
            self.assertEqual(coverage["declared_unavailable_count"], 6)
            self.assertEqual(coverage["status"], "declared_unavailable")
            self.assertEqual(coverage["supplement_artifact_id"], "d" * 64)
            self.assertEqual(
                summary["metric_coverage"]["artifacts"][0][
                    "run_dispersion_missing"
                ][0]["availability_status"],
                "declared_unavailable",
            )
            self.assertEqual(
                summary["diagnostics_supplement"]["artifact_id"],
                "d" * 64,
            )

    def test_release_lock_is_safe_and_binds_expected_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "workspace" / "stage4" / "runs"
            run_id = "a" * 24
            artifact = runs / f"s4-{run_id[:20]}"
            artifact_id = _publish(artifact, "source", run_id, [])
            results = json.loads((artifact / "results.json").read_text(encoding="utf-8"))
            lock = root / "configs" / "stage4_completion_release.json"
            lock.parent.mkdir()
            document = _completion_release_document(
                artifact,
                artifact_id=artifact_id,
                results=results,
            )
            lock.write_text(
                json.dumps(document),
                encoding="utf-8",
            )
            references = resolve_artifact_references(
                [lock],
                repo_root=root,
                development_runs_root=runs,
            )
            loaded = load_development_artifact(references[0])
            self.assertEqual(loaded.artifact_id, artifact_id)

            document["artifacts"][0]["path"] = "workspace/stage4/final/forbidden"
            lock.write_text(
                json.dumps(document),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                CompletionSummaryError, "workspace/stage4/runs"
            ):
                resolve_artifact_references(
                    [lock],
                    repo_root=root,
                    development_runs_root=runs,
                )

    def test_release_lock_rejects_noncanonical_json_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arbitrary = root / "raw-source.json"
            arbitrary.write_text('{"artifacts": []}', encoding="utf-8")
            canonical = root / "configs" / "stage4_completion_release.json"
            canonical.parent.mkdir()
            canonical.write_text("{}", encoding="utf-8")
            aliases = (
                (
                    arbitrary,
                    "only configs/stage4_completion_release.json",
                ),
                (
                    root
                    / "configs"
                    / ".."
                    / "configs"
                    / "stage4_completion_release.json",
                    "path is not canonical",
                ),
            )
            for supplied, expected in aliases:
                with (
                    self.subTest(supplied=supplied),
                    mock.patch(
                        "scripts.summarize_stage4_completion._read_regular_bytes",
                        side_effect=AssertionError(
                            "noncanonical JSON must not be read"
                        ),
                    ) as reader,
                ):
                    with self.assertRaisesRegex(CompletionSummaryError, expected):
                        resolve_artifact_references([supplied], repo_root=root)
                reader.assert_not_called()

    def test_release_lock_requires_exact_identity_schema_and_final_safe_protocol(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "workspace" / "stage4" / "runs"
            run_id = "a" * 24
            artifact = runs / f"s4-{run_id[:20]}"
            artifact_id = _publish(artifact, "source", run_id, [])
            results = json.loads((artifact / "results.json").read_text(encoding="utf-8"))
            lock = root / "configs" / "stage4_completion_release.json"
            lock.parent.mkdir()
            pristine = _completion_release_document(
                artifact,
                artifact_id=artifact_id,
                results=results,
            )
            cases = (
                (
                    "schema_version",
                    "schema or identity differs",
                    lambda value: value.update({"release_schema_version": 1}),
                ),
                (
                    "stage_name",
                    "schema or identity differs",
                    lambda value: value.update({"stage_name": "other"}),
                ),
                (
                    "policy_id",
                    "schema or identity differs",
                    lambda value: value.update({"policy_id": "other"}),
                ),
                (
                    "extra_top_level",
                    "schema or identity differs",
                    lambda value: value.update({"labels": [123]}),
                ),
                (
                    "final_evaluated",
                    "protocol is not final-safe",
                    lambda value: value["protocol"].update(
                        {"final_holdout_evaluated": True}
                    ),
                ),
                (
                    "false_encoded_as_zero",
                    "protocol is not final-safe",
                    lambda value: value["protocol"].update(
                        {"final_holdout_evaluated": 0}
                    ),
                ),
                (
                    "artifact_missing_field",
                    "schema differs from the formal release",
                    lambda value: value["artifacts"][0].pop("manifest_sha256"),
                ),
                (
                    "artifact_extra_field",
                    "schema differs from the formal release",
                    lambda value: value["artifacts"][0].update({"label": 123}),
                ),
            )
            for case, expected, mutate in cases:
                with self.subTest(case=case):
                    document = copy.deepcopy(pristine)
                    mutate(document)
                    lock.write_text(json.dumps(document), encoding="utf-8")
                    with self.assertRaisesRegex(CompletionSummaryError, expected):
                        resolve_artifact_references(
                            [lock],
                            repo_root=root,
                            development_runs_root=runs,
                        )

    def test_release_lock_rejects_unsafe_ancestry_and_hard_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configs = root / "configs"
            configs.mkdir()
            lock = configs / "stage4_completion_release.json"
            lock.write_text("{}", encoding="utf-8")
            with (
                mock.patch(
                    "scripts.summarize_stage4_completion._is_link_or_reparse",
                    side_effect=lambda path: path == configs,
                ),
                mock.patch(
                    "scripts.summarize_stage4_completion._read_regular_bytes",
                    side_effect=AssertionError("unsafe ancestry must fail pre-read"),
                ) as reader,
            ):
                with self.assertRaisesRegex(
                    CompletionSummaryError,
                    "traverses a symlink, junction, or reparse point",
                ):
                    resolve_artifact_references([lock], repo_root=root)
            reader.assert_not_called()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configs = root / "configs"
            configs.mkdir()
            outside = root / "outside.json"
            outside.write_text("{}", encoding="utf-8")
            lock = configs / "stage4_completion_release.json"
            try:
                os.link(outside, lock)
            except OSError as exc:
                self.skipTest(f"hard links are unavailable: {exc}")
            with self.assertRaisesRegex(CompletionSummaryError, "regular file"):
                resolve_artifact_references([lock], repo_root=root)

    def test_final_claim_and_tampering_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "workspace" / "stage4" / "runs"
            path = runs / "run"
            _publish(path, "source", "run", [])
            outside = root / "workspace" / "stage4" / "final" / "run"
            outside.mkdir(parents=True)
            with self.assertRaisesRegex(CompletionSummaryError, "never opened"):
                resolve_artifact_references(
                    [outside],
                    repo_root=root,
                    development_runs_root=runs,
                )

            results = json.loads((path / "results.json").read_text(encoding="utf-8"))
            opened = copy.deepcopy(results)
            opened["final_holdout"]["evaluated"] = True
            opened_without_digest = dict(opened)
            opened_without_digest.pop("results_payload_sha256")
            opened["results_payload_sha256"] = hashlib.sha256(
                _canonical_bytes(opened_without_digest)
            ).hexdigest()
            results_payload = (
                json.dumps(opened, sort_keys=True, indent=2).encode() + b"\n"
            )
            (path / "results.json").write_bytes(results_payload)
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            manifest["files"]["results.json"] = hashlib.sha256(
                results_payload
            ).hexdigest()
            manifest["metadata"]["results_payload_sha256"] = opened[
                "results_payload_sha256"
            ]
            semantic = {
                key: manifest[key]
                for key in ("stage_name", "schema_version", "files", "metadata")
            }
            manifest["artifact_id"] = hashlib.sha256(
                _canonical_bytes(semantic)
            ).hexdigest()
            (path / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True), encoding="utf-8"
            )
            (path / "_SUCCESS").write_text(
                manifest["artifact_id"] + "\n", encoding="ascii"
            )
            with self.assertRaisesRegex(CompletionSummaryError, "final-holdout"):
                load_development_artifact(ArtifactReference(path))


if __name__ == "__main__":
    unittest.main()
