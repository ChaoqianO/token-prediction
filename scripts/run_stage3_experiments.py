"""Run one immutable, source-bound Stage 3 development experiment artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from token_prediction.dataset import build_lifecycle_slice
from token_prediction.development import STAGE_SPLIT_SEEDS, build_development_protocol
from token_prediction.evaluation import (
    ScoredForecast,
    evaluate_budget_scenarios,
    evaluate_progress_checkpoints,
    evaluate_same_task_run_variance,
    evaluate_termination_strata,
    paired_task_metric_bootstrap,
)
from token_prediction.experiment import CandidateResult, CandidateSpec, ExperimentSpec
from token_prediction.lifecycle_bundle import load_lifecycle_bundle
from token_prediction.lineage import publish_artifact, verify_artifact
from token_prediction.pipeline import (
    DevelopmentExperimentResults,
    _write_fold_artifacts,
    run_development_experiments,
)
from token_prediction.stage3_matrix import (
    STAGE3_BUDGET_THRESHOLDS,
    Stage3Matrix,
    build_stage3_matrix,
)

if __package__:
    from scripts.run_data_foundation_baseline import (
        DEFAULT_BASELINE_LOCK,
        DataFoundationBaselineError,
        LockContext,
        _is_link_or_reparse,
        _repo_path,
        _safe_relative,
        load_lock_context,
    )
    from scripts.run_stage2_experiments import (
        DATA_FOUNDATION_BASELINE_RELATIVE,
        SOURCE_NAMES,
        STAGE1_VERIFIER_RELATIVE,
        STAGE2_AUXILIARY_MANIFEST_RELATIVE,
        STAGE2_METADATA_EXTRACTOR_RELATIVE,
        STAGE2_RUNNER_RELATIVE,
        STAGE2_SOKOBAN_AUDITOR_RELATIVE,
        Stage2ExperimentError,
        Stage2LoadedSource,
        _runtime_versions,
        _verify_source_inputs,
        load_stage2_source,
    )
else:  # pragma: no cover - production CLI invocation
    from run_data_foundation_baseline import (
        DEFAULT_BASELINE_LOCK,
        DataFoundationBaselineError,
        LockContext,
        _is_link_or_reparse,
        _repo_path,
        _safe_relative,
        load_lock_context,
    )
    from run_stage2_experiments import (
        DATA_FOUNDATION_BASELINE_RELATIVE,
        SOURCE_NAMES,
        STAGE1_VERIFIER_RELATIVE,
        STAGE2_AUXILIARY_MANIFEST_RELATIVE,
        STAGE2_METADATA_EXTRACTOR_RELATIVE,
        STAGE2_RUNNER_RELATIVE,
        STAGE2_SOKOBAN_AUDITOR_RELATIVE,
        Stage2ExperimentError,
        Stage2LoadedSource,
        _runtime_versions,
        _verify_source_inputs,
        load_stage2_source,
    )


STAGE3_RESULTS_SCHEMA_VERSION = 1
STAGE3_ARTIFACT_SCHEMA_VERSION = 1
STAGE3_STAGE_NAME = "stage3_development_source"
STAGE3_RUN_POLICY_ID = "stage3_source_three_seed_nested_cv_v1"
STAGE3_PREDICTION_PROJECTION_ID = "stage3_calibrated_prediction_projection_v1"
STAGE3_COHORT_PROJECTION_ID = "stage3_prediction_cohort_projection_v1"
STAGE3_TASK_PSEUDONYM_POLICY_ID = "stage3_task_pseudonym_v1"
STAGE3_ARTIFACT_LAYOUT_ID = "stage3_compact_fold_artifact_layout_v1"
STAGE3_OUTPUT_KEY_HEX_LENGTH = 20
STAGE3_RUNNER_RELATIVE = "scripts/run_stage3_experiments.py"
DEFAULT_OUTPUT_ROOT = "workspace/stage3/runs"
ALLOWED_OUTPUT_PREFIX = "workspace/stage3/runs/"
_FORBIDDEN_RESULT_KEYS = frozenset(
    {
        "attempt_id",
        "label",
        "logical_call_id",
        "point_id",
        "source_event_id",
        "target_value",
        "task_id",
        "trajectory_id",
        "truth",
    }
)


class Stage3ExperimentError(RuntimeError):
    """A Stage 3 run cannot be executed or published safely."""


@dataclass(frozen=True)
class Stage3CodeBinding:
    git_commit: str
    code_tree_sha256: str
    paths: tuple[str, ...]


@dataclass(frozen=True)
class Stage3SourceSummary:
    source_name: str
    source_id: str
    run_id: str
    output_dir: Path
    artifact_id: str
    results_payload_sha256: str
    matrix_id: str
    development_protocol_id: str
    experiment_count: int
    candidate_seed_run_count: int


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Stage3ExperimentError("Stage 3 metadata is not finite canonical JSON") from exc


def _semantic_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _required_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Stage3ExperimentError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _assert_aggregate_safe(value: object, *, path: str = "results") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise Stage3ExperimentError(f"{path} contains a non-string key")
            if key.casefold() in _FORBIDDEN_RESULT_KEYS:
                raise Stage3ExperimentError(f"{path} contains forbidden raw field {key!r}")
            if key.endswith("_path") and item is not None:
                try:
                    _safe_relative(item, label=f"{path}.{key}")
                except DataFoundationBaselineError as exc:
                    raise Stage3ExperimentError(
                        f"{path}.{key} is not a safe repository-relative path"
                    ) from exc
            _assert_aggregate_safe(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_aggregate_safe(item, path=f"{path}[{index}]")
        return
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    raise Stage3ExperimentError(f"{path} contains unsupported aggregate value")


def _git(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-c", "core.quotepath=false", *arguments],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise Stage3ExperimentError(f"Git command failed: {message}")
    return completed.stdout


def _stage3_code_paths(root: Path) -> tuple[str, ...]:
    dependencies = {
        STAGE3_RUNNER_RELATIVE,
        STAGE2_RUNNER_RELATIVE,
        STAGE2_METADATA_EXTRACTOR_RELATIVE,
        STAGE2_SOKOBAN_AUDITOR_RELATIVE,
        STAGE1_VERIFIER_RELATIVE,
        DATA_FOUNDATION_BASELINE_RELATIVE,
        STAGE2_AUXILIARY_MANIFEST_RELATIVE,
    }
    raw = _git(root, "ls-files", "-z", "--", "src/token_prediction", *sorted(dependencies))
    paths = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        try:
            relative = item.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise Stage3ExperimentError("Git returned a non-UTF-8 code path") from exc
        relative = _safe_relative(relative, label="Stage 3 code path")
        if relative in dependencies or (
            relative.startswith("src/token_prediction/") and relative.endswith(".py")
        ):
            paths.append(relative)
    resolved = tuple(sorted(set(paths)))
    if not dependencies <= set(resolved) or not any(
        path.startswith("src/token_prediction/") for path in resolved
    ):
        raise Stage3ExperimentError("HEAD does not contain the Stage 3 runner and package")
    return resolved


def _framed_code_hash(items: Iterable[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256(b"token-prediction-stage3-code-tree-v1\0")
    for relative, payload in items:
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def capture_stage3_code_binding(root: Path) -> Stage3CodeBinding:
    commit = _git(root, "rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise Stage3ExperimentError("HEAD is not a full Git commit id")
    paths = _stage3_code_paths(root)
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--",
        *paths,
    )
    if status:
        raise Stage3ExperimentError("Stage 3 runner and package must be clean at HEAD")
    workspace_items = []
    commit_items = []
    for relative in paths:
        path = _repo_path(root, relative, label="Stage 3 code path")
        if not path.is_file() or _is_link_or_reparse(path):
            raise Stage3ExperimentError("Stage 3 code binding contains an unsafe file")
        workspace_items.append((relative, path.read_bytes()))
        commit_items.append((relative, _git(root, "show", f"{commit}:{relative}")))
    workspace_hash = _framed_code_hash(workspace_items)
    if workspace_hash != _framed_code_hash(commit_items):
        raise Stage3ExperimentError("Stage 3 workspace code differs from HEAD blobs")
    return Stage3CodeBinding(commit, workspace_hash, paths)


def _verify_runner_origin(root: Path) -> None:
    expected = _repo_path(root, STAGE3_RUNNER_RELATIVE, label="Stage 3 runner")
    actual = Path(__file__)
    if _is_link_or_reparse(actual) or actual.resolve() != expected.resolve():
        raise Stage3ExperimentError("executing Stage 3 runner is outside repository_root")


def _task_pseudonym(task_id: str, *, split_plan_id: str) -> str:
    return hashlib.sha256(
        f"{STAGE3_TASK_PSEUDONYM_POLICY_ID}\0{split_plan_id}\0{task_id}".encode("utf-8")
    ).hexdigest()


def _artifact_key(kind: str, identity: str) -> str:
    if kind not in {"e", "c"} or not str(identity).strip():
        raise Stage3ExperimentError("invalid compact artifact identity")
    digest = hashlib.sha256(
        f"{STAGE3_ARTIFACT_LAYOUT_ID}\0{kind}\0{identity}".encode("utf-8")
    ).hexdigest()
    return f"{kind}_{digest[:16]}"


def _output_key(run_id: str) -> str:
    if len(run_id) < STAGE3_OUTPUT_KEY_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in run_id
    ):
        raise Stage3ExperimentError("Stage 3 run id is not hexadecimal")
    return f"s3-{run_id[:STAGE3_OUTPUT_KEY_HEX_LENGTH]}"


def _prediction_document(result: CandidateResult, record: Any) -> dict[str, object]:
    forecast = record.forecast
    return {
        "candidate_id": result.candidate_id,
        "candidate_hash": result.candidate_hash,
        "point_id": record.point_id,
        "task_id": record.task_id,
        "trajectory_id": record.trajectory_id,
        "condition_id": record.condition_id,
        "fold": record.fold,
        "target": record.target.value,
        "lower": forecast.lower,
        "point": forecast.point,
        "upper": forecast.upper,
        "raw_lower": forecast.raw_lower,
        "raw_point": forecast.raw_point,
        "raw_upper": forecast.raw_upper,
        "overhead_input_tokens": forecast.overhead_input_tokens,
        "overhead_output_tokens": forecast.overhead_output_tokens,
        "sample_weight": record.sample_weight,
    }


def prediction_projection_sha256(result: CandidateResult) -> str:
    digest = hashlib.sha256(f"{STAGE3_PREDICTION_PROJECTION_ID}\0".encode("ascii"))
    for record in sorted(result.predictions, key=lambda item: item.point_id):
        payload = _canonical_json_bytes(_prediction_document(result, record))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def cohort_projection_sha256(result: CandidateResult) -> str:
    digest = hashlib.sha256(f"{STAGE3_COHORT_PROJECTION_ID}\0".encode("ascii"))
    for record in sorted(result.predictions, key=lambda item: item.point_id):
        payload = _canonical_json_bytes(
            {
                "point_id": record.point_id,
                "task_id": record.task_id,
                "trajectory_id": record.trajectory_id,
                "condition_id": record.condition_id,
                "fold": record.fold,
                "target": record.target.value,
                "sample_weight": record.sample_weight,
            }
        )
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _task_metric_projection(result: CandidateResult) -> list[dict[str, object]]:
    return [
        {
            "task_pseudonym": _task_pseudonym(
                task_id,
                split_plan_id=result.split_plan_id,
            ),
            **dict(metrics),
        }
        for task_id, metrics in sorted(result.task_metrics.items())
    ]


def _numeric_seed_aggregate(
    seed_metrics: Sequence[Mapping[str, float | int | str]],
) -> dict[str, Mapping[str, float]]:
    if not seed_metrics:
        raise Stage3ExperimentError("cannot aggregate an empty Stage 3 seed set")
    common = set(seed_metrics[0])
    for metrics in seed_metrics[1:]:
        common &= set(metrics)
    result: dict[str, Mapping[str, float]] = {}
    for key in sorted(common):
        values = [metrics[key] for metrics in seed_metrics]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            continue
        numeric = [float(value) for value in values]
        if any(not math.isfinite(value) for value in numeric):
            raise Stage3ExperimentError("Stage 3 seed metrics contain non-finite values")
        mean = sum(numeric) / len(numeric)
        variance = sum((value - mean) ** 2 for value in numeric) / len(numeric)
        result[key] = {
            "mean": mean,
            "minimum": min(numeric),
            "maximum": max(numeric),
            "population_stddev": math.sqrt(variance),
        }
    return result


def _budget_rows(
    result: CandidateResult,
    spec: ExperimentSpec,
    execution: DevelopmentExperimentResults,
) -> tuple[ScoredForecast, ...]:
    cell = execution.protocol.development_dataset.select(
        spec.position,
        spec.target,
        condition_id=spec.condition_id,
    )
    labels = {
        row.point.point_id: float(row.label)
        for row in cell.rows
        if row.label is not None
    }
    if set(labels) != {record.point_id for record in result.predictions}:
        raise Stage3ExperimentError("Stage 3 budget cohort differs from scored predictions")
    return tuple(
        ScoredForecast(
            task_id=record.task_id,
            trajectory_id=record.trajectory_id,
            forecast=record.forecast,
            target_value=labels[record.point_id],
            sample_weight=record.sample_weight,
        )
        for record in result.predictions
    )


def _lifecycle_diagnostics(
    result: CandidateResult,
    candidate: CandidateSpec,
    spec: ExperimentSpec,
    execution: DevelopmentExperimentResults,
    *,
    split_seed: int,
    source_provenance: Mapping[str, object],
) -> Mapping[str, object]:
    if not candidate.graph.is_lifecycle:
        return {"status": "not_applicable_point_candidate"}
    split_plan = next(
        (plan for plan in execution.protocol.outer_plans if plan.seed == split_seed),
        None,
    )
    if split_plan is None:
        raise Stage3ExperimentError("Stage 3 diagnostic split plan is missing")
    lifecycle_slice = build_lifecycle_slice(
        execution.protocol.development_dataset,
        target=spec.target,
        condition_id=spec.condition_id,
    )
    artifacts = {artifact.fold: artifact for artifact in result.fold_artifacts}
    if set(artifacts) != set(range(5)):
        raise Stage3ExperimentError("lifecycle diagnostics require five fold artifacts")
    records = {record.point_id: record for record in result.predictions}
    replayed = []
    replayed_scored_ids: set[str] = set()
    for fold in range(5):
        bundle = dict(artifacts[fold].bundle_files or {})
        if not bundle:
            raise Stage3ExperimentError("lifecycle diagnostic bundle is missing")
        loaded = load_lifecycle_bundle(
            bundle,
            expected_source_provenance=source_provenance,
        )
        test_tasks = split_plan.partition(fold).test_tasks
        sequences = tuple(
            sequence
            for sequence in lifecycle_slice.sequences
            if sequence.task_id in test_tasks
        )
        if not sequences:
            raise Stage3ExperimentError("lifecycle diagnostic fold has no test sequences")
        fold_runs = loaded.run_calibrated(sequences)
        for run in fold_runs:
            for prediction in run.scored_predictions:
                point_id = prediction.step.point.point_id
                record = records.get(point_id)
                if record is None or record.fold != fold:
                    raise Stage3ExperimentError(
                        "lifecycle replay differs from the scored result cohort"
                    )
                if prediction.forecast != record.forecast:
                    raise Stage3ExperimentError(
                        "lifecycle replay differs from the calibrated result trajectory"
                    )
                replayed_scored_ids.add(point_id)
        replayed.extend(fold_runs)
    if replayed_scored_ids != set(records):
        raise Stage3ExperimentError("lifecycle replay did not cover every scored prediction")
    return {
        "status": "complete_calibrated_trajectory_replay_exact",
        "run_count": len(replayed),
        "update_boundary_count": sum(len(run.predictions) for run in replayed),
        "scored_boundary_count": sum(len(run.scored_predictions) for run in replayed),
        "progress": evaluate_progress_checkpoints(replayed, alpha=spec.alpha),
        "termination": evaluate_termination_strata(replayed, alpha=spec.alpha),
        "run_variance": evaluate_same_task_run_variance(replayed),
    }


def _result_document(
    result: CandidateResult,
    candidate: CandidateSpec,
    spec: ExperimentSpec,
    execution: DevelopmentExperimentResults,
    *,
    split_seed: int,
    source_provenance: Mapping[str, object],
) -> dict[str, object]:
    bundle_folds = [
        artifact.fold
        for artifact in result.fold_artifacts
        if artifact.bundle_files is not None
    ]
    require_bundle = candidate.estimator_id in {
        "independent_mlp",
        "lightgbm_quantile",
        "gru_residual",
    } or candidate.graph.is_lifecycle
    if require_bundle and bundle_folds != list(range(5)):
        raise Stage3ExperimentError(
            f"candidate {result.candidate_id!r} lacks five reloadable fold bundles"
        )
    budget = evaluate_budget_scenarios(
        _budget_rows(result, spec, execution),
        budgets=STAGE3_BUDGET_THRESHOLDS,
    )
    lifecycle = _lifecycle_diagnostics(
        result,
        candidate,
        spec,
        execution,
        split_seed=split_seed,
        source_provenance=source_provenance,
    )
    return {
        "candidate_id": result.candidate_id,
        "candidate_hash": result.candidate_hash,
        "comparability_key": list(result.comparability_key),
        "split_plan_id": result.split_plan_id,
        "prediction_count": len(result.predictions),
        "prediction_projection_id": STAGE3_PREDICTION_PROJECTION_ID,
        "prediction_projection_sha256": prediction_projection_sha256(result),
        "cohort_projection_id": STAGE3_COHORT_PROJECTION_ID,
        "cohort_projection_sha256": cohort_projection_sha256(result),
        "metrics": dict(result.metrics),
        "fold_metrics": {
            str(fold): dict(metrics) for fold, metrics in result.fold_metrics.items()
        },
        "task_metric_policy_id": STAGE3_TASK_PSEUDONYM_POLICY_ID,
        "task_metrics": _task_metric_projection(result),
        "fold_artifact_count": len(result.fold_artifacts),
        "reloadable_bundle_folds": bundle_folds,
        "bundle_reload_parity": {
            "status": (
                "exact_during_execution"
                if require_bundle
                else "not_applicable_stateless_or_mechanical"
            ),
            "fold_count": len(bundle_folds),
        },
        "stage3_evaluation": {
            "budget": budget,
            "lifecycle": lifecycle,
        },
    }


def build_stage3_results(
    execution: DevelopmentExperimentResults,
    matrix: Stage3Matrix,
    *,
    source_name: str,
    loaded: Stage2LoadedSource,
    lock_context: LockContext,
    code_binding: Stage3CodeBinding,
    runtime_versions: Mapping[str, str],
    run_id: str,
) -> dict[str, object]:
    if execution.protocol.protocol_id != matrix.development_protocol_id:
        raise Stage3ExperimentError("Stage 3 execution and matrix protocol ids differ")
    if tuple(item.split_seed for item in execution.seed_results) != STAGE_SPLIT_SEEDS:
        raise Stage3ExperimentError("Stage 3 execution does not contain all frozen seeds")
    source_provenance = {
        "source_descriptor": loaded.source_lock.descriptor.to_dict(),
        "source_descriptor_hash": loaded.source_lock.descriptor.descriptor_hash,
        "code_hash": code_binding.code_tree_sha256,
        "runtime_versions": dict(runtime_versions),
    }

    experiments = []
    candidate_seed_run_count = 0
    for spec_index, spec in enumerate(matrix.experiments):
        candidate_documents = []
        for candidate in spec.candidates:
            per_seed = []
            raw_seed_metrics = []
            for seed_result in execution.seed_results:
                group = seed_result.result_groups[spec_index]
                result = next(
                    (item for item in group if item.candidate_id == candidate.candidate_id),
                    None,
                )
                reference = next(
                    (item for item in group if item.candidate_id == "empirical"),
                    None,
                )
                if result is None or reference is None:
                    raise Stage3ExperimentError("Stage 3 candidate/reference result is missing")
                result_document = _result_document(
                    result,
                    candidate,
                    spec,
                    execution,
                    split_seed=seed_result.split_seed,
                    source_provenance=source_provenance,
                )
                result_document["split_seed"] = seed_result.split_seed
                if candidate.candidate_id != "empirical":
                    comparison_seed = int(
                        hashlib.sha256(
                            (
                                f"stage3-paired-bootstrap-v1\0{seed_result.split_seed}\0"
                                f"{spec.experiment_id}\0{candidate.candidate_id}"
                            ).encode("utf-8")
                        ).hexdigest()[:16],
                        16,
                    )
                    result_document["paired_vs_empirical"] = asdict(
                        paired_task_metric_bootstrap(
                            result,
                            reference,
                            iterations=10_000,
                            seed=comparison_seed,
                        )
                    )
                per_seed.append(result_document)
                raw_seed_metrics.append(result.metrics)
                candidate_seed_run_count += 1
            candidate_documents.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "candidate_hash": candidate.content_hash,
                    "artifact_key": _artifact_key("c", candidate.content_hash),
                    "estimator_id": candidate.estimator_id,
                    "feature_set_id": candidate.feature_set.feature_set_id,
                    "feature_set_hash": candidate.feature_set.content_hash,
                    "candidate_graph": candidate.graph.to_dict(),
                    "role": candidate.role.value,
                    "ablation": (
                        {
                            "reference_candidate_id": (
                                candidate.ablation.reference_candidate_id
                            ),
                            "axis": candidate.ablation.axis.value,
                            "allowed_config_paths": sorted(
                                candidate.ablation.allowed_config_paths
                            ),
                        }
                        if candidate.ablation is not None
                        else None
                    ),
                    "seed_results": per_seed,
                    "cross_seed_metrics": _numeric_seed_aggregate(raw_seed_metrics),
                }
            )
        experiments.append(
            {
                "experiment_id": spec.experiment_id,
                "artifact_key": _artifact_key("e", spec.experiment_id),
                "position": spec.position.value,
                "target": spec.target.value,
                "condition_id": spec.condition_id,
                "alpha": spec.alpha,
                "calibrator_id": spec.calibrator_id,
                "candidates": candidate_documents,
            }
        )

    results: dict[str, object] = {
        "results_schema_version": STAGE3_RESULTS_SCHEMA_VERSION,
        "stage_name": STAGE3_STAGE_NAME,
        "run_policy_id": STAGE3_RUN_POLICY_ID,
        "artifact_layout_id": STAGE3_ARTIFACT_LAYOUT_ID,
        "run_id": run_id,
        "source": {
            "source_name": source_name,
            "source_id": loaded.source_lock.descriptor.source_id,
            "revision": loaded.source_lock.descriptor.revision,
            "source_descriptor_hash": loaded.source_lock.descriptor.descriptor_hash,
            "capability_contract_hash": (
                loaded.source_lock.descriptor.capabilities.contract_hash
            ),
            "manifest_path": loaded.source_lock.manifest_path,
            "manifest_sha256": loaded.source_lock.manifest_sha256,
            "raw_artifact_sha256": loaded.source_lock.raw_artifact_sha256,
        },
        "data_foundation": {
            "baseline_lock_path": lock_context.baseline_lock_path,
            "baseline_lock_file_sha256": lock_context.baseline_lock_file_sha256,
            "audit_payload_sha256": lock_context.audit_payload_sha256,
        },
        "code_binding": {
            "git_commit": code_binding.git_commit,
            "code_tree_sha256": code_binding.code_tree_sha256,
            "code_paths": list(code_binding.paths),
        },
        "runtime_versions": dict(runtime_versions),
        "dataset": {
            "base_dataset_id": loaded.base_dataset_id,
            "derived_dataset_id": loaded.derived_dataset.dataset_id,
            "development_dataset_id": execution.protocol.development_dataset.dataset_id,
            "base_row_count": loaded.base_row_count,
            "derived_row_count": len(loaded.derived_dataset.rows),
            "development_row_count": len(execution.protocol.development_dataset.rows),
            "input_projection": loaded.projection_id,
        },
        "development_protocol": execution.audit_document,
        "matrix": matrix.identity_document(),
        "experiments": experiments,
        "gates": [gate.to_dict() for gate in matrix.gates],
        "summary": {
            "experiment_count": len(matrix.experiments),
            "candidate_seed_run_count": candidate_seed_run_count,
            "split_seeds": list(STAGE_SPLIT_SEEDS),
            "outer_folds": 5,
            "inner_folds": 5,
            "budget_thresholds": list(STAGE3_BUDGET_THRESHOLDS),
        },
        "final_holdout": {
            "evaluated": False,
            "prediction_count": 0,
            "target_values_used_for_fit_calibration_scoring": False,
            "selection_claim": "none",
        },
    }
    _assert_aggregate_safe(results)
    results["results_payload_sha256"] = _semantic_sha256(results)
    return results


def verify_stage3_results_document(value: Mapping[str, object]) -> str:
    required = {
        "results_schema_version",
        "stage_name",
        "run_policy_id",
        "artifact_layout_id",
        "run_id",
        "source",
        "data_foundation",
        "code_binding",
        "runtime_versions",
        "dataset",
        "development_protocol",
        "matrix",
        "experiments",
        "gates",
        "summary",
        "final_holdout",
        "results_payload_sha256",
    }
    if set(value) != required:
        raise Stage3ExperimentError("Stage 3 results keys do not match the schema")
    if value["results_schema_version"] != STAGE3_RESULTS_SCHEMA_VERSION:
        raise Stage3ExperimentError("unsupported Stage 3 results schema")
    if value["stage_name"] != STAGE3_STAGE_NAME or value["run_policy_id"] != STAGE3_RUN_POLICY_ID:
        raise Stage3ExperimentError("Stage 3 results policy identity is invalid")
    if value["artifact_layout_id"] != STAGE3_ARTIFACT_LAYOUT_ID:
        raise Stage3ExperimentError("Stage 3 artifact layout identity is invalid")
    _assert_aggregate_safe(value)
    holdout = value["final_holdout"]
    if not isinstance(holdout, Mapping) or holdout != {
        "evaluated": False,
        "prediction_count": 0,
        "target_values_used_for_fit_calibration_scoring": False,
        "selection_claim": "none",
    }:
        raise Stage3ExperimentError("Stage 3 final holdout is not sealed")
    expected = dict(value)
    declared = _required_sha256(
        expected.pop("results_payload_sha256"),
        name="Stage 3 results payload SHA-256",
    )
    if _semantic_sha256(expected) != declared:
        raise Stage3ExperimentError("Stage 3 results payload SHA-256 does not close")
    return declared


def _write_results(path: Path, results: Mapping[str, object]) -> None:
    path.write_bytes(
        json.dumps(
            dict(results),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _write_execution_artifacts(
    root: Path,
    execution: DevelopmentExperimentResults,
    matrix: Stage3Matrix,
) -> None:
    experiment_keys: set[str] = set()
    for spec_index, spec in enumerate(matrix.experiments):
        experiment_key = _artifact_key("e", spec.experiment_id)
        if experiment_key in experiment_keys:
            raise Stage3ExperimentError("compact Stage 3 experiment artifact key collided")
        experiment_keys.add(experiment_key)
        candidate_keys: set[str] = set()
        for candidate in spec.candidates:
            candidate_key = _artifact_key("c", candidate.content_hash)
            if candidate_key in candidate_keys:
                raise Stage3ExperimentError("compact Stage 3 candidate artifact key collided")
            candidate_keys.add(candidate_key)
            for seed_result in execution.seed_results:
                result = next(
                    item
                    for item in seed_result.result_groups[spec_index]
                    if item.candidate_id == candidate.candidate_id
                )
                _write_fold_artifacts(
                    root,
                    experiment_id=experiment_key,
                    candidate_id=candidate_key,
                    split_seed=seed_result.split_seed,
                    artifacts=result.fold_artifacts,
                )


def _safe_output_root(root: Path, relative: str) -> tuple[str, Path]:
    try:
        canonical = _safe_relative(relative, label="Stage 3 output root")
    except DataFoundationBaselineError as exc:
        raise Stage3ExperimentError("Stage 3 output root is not a safe relative path") from exc
    prefix = ALLOWED_OUTPUT_PREFIX.rstrip("/")
    if canonical != prefix and not canonical.startswith(ALLOWED_OUTPUT_PREFIX):
        raise Stage3ExperimentError(
            f"Stage 3 output root must be {prefix!r} or a descendant"
        )
    try:
        resolved = _repo_path(root, canonical, label="Stage 3 output root")
    except DataFoundationBaselineError as exc:
        raise Stage3ExperimentError("Stage 3 output root escapes the repository") from exc
    return canonical, resolved


def _run_semantic(
    *,
    source_name: str,
    loaded: Stage2LoadedSource,
    lock_context: LockContext,
    code_binding: Stage3CodeBinding,
    runtime_versions: Mapping[str, str],
    matrix: Stage3Matrix,
) -> dict[str, object]:
    return {
        "results_schema_version": STAGE3_RESULTS_SCHEMA_VERSION,
        "run_policy_id": STAGE3_RUN_POLICY_ID,
        "source_name": source_name,
        "source_id": loaded.source_lock.descriptor.source_id,
        "revision": loaded.source_lock.descriptor.revision,
        "raw_artifact_sha256": loaded.source_lock.raw_artifact_sha256,
        "data_foundation_baseline_lock_sha256": (
            lock_context.baseline_lock_file_sha256
        ),
        "base_dataset_id": loaded.base_dataset_id,
        "derived_dataset_id": loaded.derived_dataset.dataset_id,
        "development_protocol_id": matrix.development_protocol_id,
        "matrix_id": matrix.matrix_id,
        "git_commit": code_binding.git_commit,
        "code_tree_sha256": code_binding.code_tree_sha256,
        "runtime_versions": dict(runtime_versions),
    }


def _existing_summary(
    output: Path,
    *,
    source_name: str,
    run_id: str,
    expected_semantic: Mapping[str, object],
) -> Stage3SourceSummary:
    manifest = verify_artifact(output)
    if manifest.metadata.get("run_id") != run_id or manifest.metadata.get(
        "run_semantic"
    ) != dict(expected_semantic):
        raise Stage3ExperimentError("existing Stage 3 artifact has another identity")
    try:
        results = json.loads((output / "results.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Stage3ExperimentError("existing Stage 3 results are unreadable") from exc
    if not isinstance(results, dict):
        raise Stage3ExperimentError("existing Stage 3 results must be an object")
    payload_hash = verify_stage3_results_document(results)
    summary = results["summary"]
    source = results["source"]
    matrix = results["matrix"]
    protocol = results["development_protocol"]
    if not all(isinstance(item, Mapping) for item in (summary, source, matrix, protocol)):
        raise Stage3ExperimentError("existing Stage 3 result sections are invalid")
    return Stage3SourceSummary(
        source_name=source_name,
        source_id=str(source["source_id"]),
        run_id=run_id,
        output_dir=output,
        artifact_id=manifest.artifact_id,
        results_payload_sha256=payload_hash,
        matrix_id=str(matrix["matrix_id"]),
        development_protocol_id=str(protocol["protocol_id"]),
        experiment_count=int(summary["experiment_count"]),
        candidate_seed_run_count=int(summary["candidate_seed_run_count"]),
    )


def run_stage3_source(
    *,
    repository_root: str | Path,
    source_name: str,
    baseline_lock: str = DEFAULT_BASELINE_LOCK,
    output_root: str = DEFAULT_OUTPUT_ROOT,
) -> Stage3SourceSummary:
    supplied_root = Path(repository_root)
    if _is_link_or_reparse(supplied_root):
        raise Stage3ExperimentError("repository root must not be linked or reparse-backed")
    root = supplied_root.resolve()
    if not root.is_dir():
        raise Stage3ExperimentError("repository root is not a directory")
    _verify_runner_origin(root)
    _canonical_output_root, output_parent = _safe_output_root(root, output_root)
    code_binding = capture_stage3_code_binding(root)
    lock_context = load_lock_context(root, baseline_lock)
    loaded = load_stage2_source(root, lock_context, source_name=source_name)
    protocol = build_development_protocol(loaded.derived_dataset)
    matrix = build_stage3_matrix(
        protocol,
        source_id=loaded.source_lock.descriptor.source_id,
    )
    runtime_versions = _runtime_versions()
    run_semantic = _run_semantic(
        source_name=source_name,
        loaded=loaded,
        lock_context=lock_context,
        code_binding=code_binding,
        runtime_versions=runtime_versions,
        matrix=matrix,
    )
    run_id = _semantic_sha256(run_semantic)[:24]
    output = output_parent / _output_key(run_id)
    if output.exists():
        return _existing_summary(
            output,
            source_name=source_name,
            run_id=run_id,
            expected_semantic=run_semantic,
        )

    source_provenance = {
        "source_descriptor": loaded.source_lock.descriptor.to_dict(),
        "source_descriptor_hash": loaded.source_lock.descriptor.descriptor_hash,
        "code_hash": code_binding.code_tree_sha256,
        "runtime_versions": runtime_versions,
    }
    execution = run_development_experiments(
        loaded.derived_dataset,
        matrix.experiments,
        source_provenance=source_provenance,
        protocol=protocol,
    )
    results = build_stage3_results(
        execution,
        matrix,
        source_name=source_name,
        loaded=loaded,
        lock_context=lock_context,
        code_binding=code_binding,
        runtime_versions=runtime_versions,
        run_id=run_id,
    )
    results_payload_sha256 = verify_stage3_results_document(results)

    output_parent.mkdir(parents=True, exist_ok=True)
    if _is_link_or_reparse(output_parent):
        raise Stage3ExperimentError("Stage 3 output parent is unsafe")
    temporary = Path(tempfile.mkdtemp(prefix=".s3-", dir=output_parent))
    try:
        _write_results(temporary / "results.json", results)
        _write_execution_artifacts(temporary, execution, matrix)
        if capture_stage3_code_binding(root) != code_binding:
            raise Stage3ExperimentError("Stage 3 code changed during execution")
        _verify_source_inputs(root, lock_context, loaded)
        manifest = publish_artifact(
            temporary,
            stage_name=STAGE3_STAGE_NAME,
            schema_version=STAGE3_ARTIFACT_SCHEMA_VERSION,
            metadata={
                "run_id": run_id,
                "run_semantic": run_semantic,
                "results_payload_sha256": results_payload_sha256,
            },
        )
        if capture_stage3_code_binding(root) != code_binding:
            raise Stage3ExperimentError(
                "Stage 3 code changed during artifact publication"
            )
        _verify_source_inputs(root, lock_context, loaded)
        if output.exists():
            raise FileExistsError(f"Stage 3 artifact destination appeared: {output}")
        os.replace(temporary, output)
        if verify_artifact(output) != manifest:
            raise Stage3ExperimentError("published Stage 3 artifact failed verification")
    finally:
        if temporary.exists():
            try:
                temporary.resolve().relative_to(output_parent.resolve())
            except ValueError as exc:
                raise Stage3ExperimentError("temporary artifact escaped output root") from exc
            shutil.rmtree(temporary)

    return Stage3SourceSummary(
        source_name=source_name,
        source_id=loaded.source_lock.descriptor.source_id,
        run_id=run_id,
        output_dir=output,
        artifact_id=manifest.artifact_id,
        results_payload_sha256=results_payload_sha256,
        matrix_id=matrix.matrix_id,
        development_protocol_id=protocol.protocol_id,
        experiment_count=len(matrix.experiments),
        candidate_seed_run_count=(
            len(STAGE_SPLIT_SEEDS)
            * sum(len(spec.candidates) for spec in matrix.experiments)
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one immutable Stage 3 source experiment artifact."
    )
    parser.add_argument(
        "--repository-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument("--source", required=True, choices=sorted(SOURCE_NAMES))
    parser.add_argument("--baseline-lock", default=DEFAULT_BASELINE_LOCK)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = run_stage3_source(
            repository_root=args.repository_root,
            source_name=args.source,
            baseline_lock=args.baseline_lock,
            output_root=args.output_root,
        )
    except (
        DataFoundationBaselineError,
        Stage2ExperimentError,
        Stage3ExperimentError,
        ValueError,
    ) as exc:
        raise SystemExit(f"Stage 3 experiment failed: {exc}") from exc
    print(
        json.dumps(
            {
                **asdict(summary),
                "output_dir": summary.output_dir.as_posix(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
