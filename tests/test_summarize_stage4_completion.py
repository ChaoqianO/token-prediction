from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_stage4_completion import (
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
    load_development_artifact,
    render_markdown,
    resolve_artifact_references,
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _metrics(mae: float) -> dict[str, object]:
    return {
        "mae": mae,
        "interval_diagnostics_id": "weighted_interval_tail_and_reserve_v1",
        "interval_below_truth_rate": 0.1,
        "interval_above_truth_rate": 0.2,
        "target_exceeds_upper_rate": 0.1,
        "mean_extra_reserved_tokens": 3.0,
        "raw_interval_below_truth_rate": 0.1,
        "raw_interval_above_truth_rate": 0.2,
        "raw_target_exceeds_upper_rate": 0.1,
        "raw_mean_extra_reserved_tokens": 2.0,
    }


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
    return {
        "candidate_id": candidate_id,
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


def _seed_policy_experiment(*, upper: float = -0.5) -> dict[str, object]:
    reference_maes = (10.0, 10.0, 10.0)
    return {
        "experiment_id": "seed-policy-cell",
        "position": "task_update",
        "target": "task_provider_accounted_remaining_tokens",
        "condition_id": "condition:seed",
        "candidates": [
            _candidate(
                RAW_SEED_CANDIDATE_ID,
                reference_maes,
                lifecycle=True,
                seed_policy_id=RAW_SEED_POLICY_ID,
            ),
            _candidate(
                POINT_ONLY_SEED_CANDIDATE_ID,
                (9.0, 9.0, 9.0),
                lifecycle=True,
                seed_policy_id=POINT_ONLY_SEED_POLICY_ID,
                reference_id=RAW_SEED_CANDIDATE_ID,
                reference_maes=reference_maes,
                axis="seed_policy",
                upper=upper,
            ),
        ],
    }


def _results(
    source_name: str,
    run_id: str,
    experiments: list[dict[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {
        "results_schema_version": 1,
        "stage_name": "stage4_development_source",
        "run_policy_id": "stage4_source_three_seed_single_axis_v1",
        "artifact_layout_id": "stage4_compact_fold_artifact_layout_v1",
        "checkpoint_policy_id": "atomic_candidate_and_every_neural_epoch_v1",
        "run_id": run_id,
        "source": {"source_name": source_name},
        "data_foundation": {},
        "code_binding": {},
        "runtime_versions": {},
        "dataset": {},
        "development_protocol": {},
        "matrix": {},
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
) -> str:
    path.mkdir(parents=True)
    results = _results(source_name, run_id, experiments)
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


class Stage4CompletionSummaryTests(unittest.TestCase):
    def test_comparisons_rule_coverage_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "workspace" / "stage4" / "runs"
            paths = [runs / f"run-{index}" for index in range(4)]
            _publish(paths[0], "call-source", "call-run", [_call_experiment()])
            _publish(
                paths[1],
                "seed-source",
                "seed-run",
                [_seed_policy_experiment()],
            )
            _publish(paths[2], "empty-a", "empty-run-a", [])
            _publish(paths[3], "empty-b", "empty-run-b", [])
            references = resolve_artifact_references(
                [str(path) for path in paths],
                repo_root=root,
                development_runs_root=runs,
            )
            loaded = tuple(load_development_artifact(item) for item in references)
            summary = build_completion_summary(loaded)

            call = summary["call_pre_mlp_vs_lightgbm"]
            self.assertEqual(len(call), 1)
            self.assertEqual(len(call[0]["seed_results"]), 3)
            self.assertEqual(call[0]["seed_results"][0]["mae_winner"], "candidate")
            seed_policy = summary["seed_policy_point_only_vs_raw_repaired"]
            self.assertEqual(len(seed_policy), 1)
            self.assertTrue(seed_policy[0]["replacement_rule"]["replace_reference"])
            self.assertEqual(
                summary["seed_policy_frozen_replacement_rule"]["decision"],
                "replace_raw_repaired_reference",
            )
            coverage = summary["metric_coverage"]
            self.assertTrue(coverage["interval_reserve"]["complete"])
            self.assertTrue(coverage["repeated_run_dispersion"]["complete"])
            self.assertFalse(summary["final_holdout_access"]["accessed"])
            markdown = render_markdown(summary)
            self.assertIn("Call-pre MLP vs LightGBM", markdown)
            self.assertIn("Final holdout: not accessed", markdown)
            self.assertIn("20260719:", markdown)

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
            _publish(path, "source", "run", [experiment])
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

    def test_diagnostics_supplement_closes_lifecycle_dispersion_coverage(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workspace" / "stage4" / "runs" / "run"
            experiment = _seed_policy_experiment()
            for candidate in experiment["candidates"]:
                for seed in candidate["seed_results"]:
                    seed.pop("stage4_evaluation")
            artifact_id = _publish(path, "source", "run", [experiment])
            loaded = load_development_artifact(ArtifactReference(path))
            supplement_rows = []
            for candidate in experiment["candidates"]:
                for seed in candidate["seed_results"]:
                    supplement_rows.append(
                        {
                            "source_name": "source",
                            "experiment_id": "seed-policy-cell",
                            "candidate_id": candidate["candidate_id"],
                            "split_seed": seed["split_seed"],
                            "run_variance": _dispersion(),
                        }
                    )
            supplement = LoadedDiagnosticsArtifact(
                path=Path(directory) / "diagnostics",
                artifact_id="d" * 64,
                results_payload_sha256="e" * 64,
                document={
                    "source_artifacts": [
                        {
                            "source_name": "source",
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
            self.assertTrue(coverage["complete"])
            self.assertEqual(coverage["complete_count"], 6)
            self.assertEqual(coverage["supplement_artifact_id"], "d" * 64)
            self.assertEqual(
                summary["diagnostics_supplement"]["artifact_id"],
                "d" * 64,
            )

    def test_release_lock_is_safe_and_binds_expected_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "workspace" / "stage4" / "runs"
            artifact = runs / "run"
            artifact_id = _publish(artifact, "source", "run", [])
            results = json.loads((artifact / "results.json").read_text(encoding="utf-8"))
            lock = root / "configs" / "stage4_completion_release.json"
            lock.parent.mkdir()
            lock.write_text(
                json.dumps(
                    {
                        "development_artifacts": [
                            {
                                "path": "workspace/stage4/runs/run",
                                "artifact_id": artifact_id,
                                "results_payload_sha256": results[
                                    "results_payload_sha256"
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            references = resolve_artifact_references(
                [lock],
                repo_root=root,
                development_runs_root=runs,
            )
            loaded = load_development_artifact(references[0])
            self.assertEqual(loaded.artifact_id, artifact_id)

            lock.write_text(
                json.dumps(
                    {
                        "development_artifacts": [
                            {"path": "workspace/stage4/final/forbidden"}
                        ]
                    }
                ),
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
