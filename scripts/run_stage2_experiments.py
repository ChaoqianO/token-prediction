"""Run one immutable, source-bound Stage 2 development experiment artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from token_prediction.collection import (
    BagenSwebenchReader,
    OpenHandsArchiveMetadata,
    OpenHandsArchiveReader,
)
from token_prediction.dataset import (
    SupervisedDataset,
    augment_request_shape_features,
    build_capability_supervised_dataset,
)
from token_prediction.development import STAGE_SPLIT_SEEDS, build_development_protocol
from token_prediction.evaluation import paired_task_metric_bootstrap
from token_prediction.experiment import CandidateResult
from token_prediction.lineage import publish_artifact, verify_artifact
from token_prediction.pipeline import (
    DevelopmentExperimentResults,
    _write_fold_artifacts,
    run_development_experiments,
)
from token_prediction.stage2_matrix import (
    BAGEN_SOURCE_ID,
    SPEND_SOURCE_ID,
    Stage2Matrix,
    build_stage2_matrix,
)

if __package__:
    from scripts.run_data_foundation_baseline import (
        DEFAULT_BASELINE_LOCK,
        DataFoundationBaselineError,
        LockContext,
        SourceLock,
        _is_link_or_reparse,
        _load_bagen_manifest,
        _repo_path,
        _safe_relative,
        _verify_realized_dataset,
        _verify_spend_archive,
        load_lock_context,
    )
else:  # pragma: no cover - exercised by the production CLI invocation
    from run_data_foundation_baseline import (
        DEFAULT_BASELINE_LOCK,
        DataFoundationBaselineError,
        LockContext,
        SourceLock,
        _is_link_or_reparse,
        _load_bagen_manifest,
        _repo_path,
        _safe_relative,
        _verify_realized_dataset,
        _verify_spend_archive,
        load_lock_context,
    )


STAGE2_RESULTS_SCHEMA_VERSION = 1
STAGE2_ARTIFACT_SCHEMA_VERSION = 1
STAGE2_STAGE_NAME = "stage2_development_source"
STAGE2_RUN_POLICY_ID = "stage2_source_three_seed_nested_cv_v1"
STAGE2_PREDICTION_PROJECTION_ID = "stage2_calibrated_prediction_projection_v1"
STAGE2_COHORT_PROJECTION_ID = "stage2_prediction_cohort_projection_v1"
STAGE2_TASK_PSEUDONYM_POLICY_ID = "stage2_task_pseudonym_v1"
STAGE2_RUNNER_RELATIVE = "scripts/run_stage2_experiments.py"
DEFAULT_OUTPUT_ROOT = "workspace/stage2/experiments"
ALLOWED_OUTPUT_PREFIX = "workspace/stage2/experiments/"
SOURCE_NAMES = {
    "bagen_swebench": BAGEN_SOURCE_ID,
    "spend_openhands": SPEND_SOURCE_ID,
}
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


class Stage2ExperimentError(RuntimeError):
    """A Stage 2 run cannot be executed or published safely."""


@dataclass(frozen=True)
class Stage2CodeBinding:
    git_commit: str
    code_tree_sha256: str
    paths: tuple[str, ...]


@dataclass(frozen=True)
class Stage2LoadedSource:
    source_name: str
    source_lock: SourceLock
    base_dataset_id: str
    base_row_count: int
    derived_dataset: SupervisedDataset
    raw_paths: tuple[Path, ...]


@dataclass(frozen=True)
class Stage2SourceSummary:
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
        raise Stage2ExperimentError("Stage 2 metadata is not finite canonical JSON") from exc


def _semantic_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _required_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Stage2ExperimentError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _assert_aggregate_safe(value: object, *, path: str = "results") -> None:
    """Reject raw row identities, labels, and unsafe paths from public results."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise Stage2ExperimentError(f"{path} contains a non-string key")
            if key.casefold() in _FORBIDDEN_RESULT_KEYS:
                raise Stage2ExperimentError(f"{path} contains forbidden raw field {key!r}")
            if key.endswith("_path") and item is not None:
                try:
                    _safe_relative(item, label=f"{path}.{key}")
                except DataFoundationBaselineError as exc:
                    raise Stage2ExperimentError(
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
    raise Stage2ExperimentError(f"{path} contains unsupported aggregate value")


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
        raise Stage2ExperimentError(f"Git command failed: {message}")
    return completed.stdout


def _stage2_code_paths(root: Path) -> tuple[str, ...]:
    raw = _git(
        root,
        "ls-files",
        "-z",
        "--",
        "src/token_prediction",
        STAGE2_RUNNER_RELATIVE,
    )
    paths: list[str] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        try:
            relative = item.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise Stage2ExperimentError("Git returned a non-UTF-8 code path") from exc
        relative = _safe_relative(relative, label="Stage 2 code path")
        if relative == STAGE2_RUNNER_RELATIVE or (
            relative.startswith("src/token_prediction/") and relative.endswith(".py")
        ):
            paths.append(relative)
    resolved = tuple(sorted(set(paths)))
    if STAGE2_RUNNER_RELATIVE not in resolved or not any(
        path.startswith("src/token_prediction/") for path in resolved
    ):
        raise Stage2ExperimentError("HEAD does not contain the Stage 2 runner and package")
    return resolved


def _framed_code_hash(items: Iterable[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256(b"token-prediction-stage2-code-tree-v1\0")
    for relative, payload in items:
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def capture_stage2_code_binding(root: Path) -> Stage2CodeBinding:
    commit = _git(root, "rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise Stage2ExperimentError("HEAD is not a full Git commit id")
    paths = _stage2_code_paths(root)
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--",
        "src/token_prediction",
        STAGE2_RUNNER_RELATIVE,
    )
    if status:
        raise Stage2ExperimentError("Stage 2 runner and package must be clean at HEAD")
    workspace_items: list[tuple[str, bytes]] = []
    commit_items: list[tuple[str, bytes]] = []
    for relative in paths:
        path = _repo_path(root, relative, label="Stage 2 code path")
        if not path.is_file() or _is_link_or_reparse(path):
            raise Stage2ExperimentError("Stage 2 code binding contains an unsafe file")
        workspace_items.append((relative, path.read_bytes()))
        commit_items.append((relative, _git(root, "show", f"{commit}:{relative}")))
    workspace_hash = _framed_code_hash(workspace_items)
    if workspace_hash != _framed_code_hash(commit_items):
        raise Stage2ExperimentError("Stage 2 workspace code differs from HEAD blobs")
    return Stage2CodeBinding(commit, workspace_hash, paths)


def _installed_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def _module_version(distribution: str, module_name: str) -> str:
    try:
        module = import_module(module_name)
    except (ImportError, OSError):
        return "not-installed"
    value = getattr(module, "__version__", None)
    if not isinstance(value, str) or not value.strip():
        return _installed_version(distribution)
    return value.strip()


def _runtime_versions() -> dict[str, str]:
    versions = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "token_prediction_version": _installed_version("token-prediction"),
        "numpy_version": _module_version("numpy", "numpy"),
        "lightgbm_version": _module_version("lightgbm", "lightgbm"),
        "torch_version": _module_version("torch", "torch"),
        "safetensors_version": _module_version("safetensors", "safetensors"),
    }
    required = {
        "numpy_version",
        "lightgbm_version",
        "torch_version",
        "safetensors_version",
    }
    missing = sorted(name for name in required if versions[name] == "not-installed")
    if missing:
        raise Stage2ExperimentError(
            "Stage 2 optional runtime is incomplete: " + ", ".join(missing)
        )
    return versions


def _verify_runner_origin(root: Path) -> None:
    expected = _repo_path(root, STAGE2_RUNNER_RELATIVE, label="Stage 2 runner")
    actual = Path(__file__)
    if _is_link_or_reparse(actual) or actual.resolve() != expected.resolve():
        raise Stage2ExperimentError("executing Stage 2 runner is outside repository_root")


def load_stage2_source(
    root: Path,
    lock_context: LockContext,
    *,
    source_name: str,
) -> Stage2LoadedSource:
    try:
        expected_source_id = SOURCE_NAMES[source_name]
        source = lock_context.sources[source_name]
    except KeyError as exc:
        raise Stage2ExperimentError(f"unsupported Stage 2 source {source_name!r}") from exc
    if source.descriptor.source_id != expected_source_id:
        raise Stage2ExperimentError("Stage 2 source id differs from its frozen contract")

    if source_name == "bagen_swebench":
        paths = _load_bagen_manifest(root, source)
        reader = BagenSwebenchReader()
        trajectories = tuple(reader.read(path) for path in paths)
        raw_paths = paths
    else:
        archive = _verify_spend_archive(root, source)
        reader = OpenHandsArchiveReader()
        trajectories = tuple(
            reader.iter_archive(
                archive,
                OpenHandsArchiveMetadata(archive_identity=source.raw_artifact_sha256),
            )
        )
        raw_paths = (archive,)
    if not trajectories:
        raise Stage2ExperimentError("Stage 2 source contains no trajectories")
    base_dataset = build_capability_supervised_dataset(trajectories, source.descriptor)
    _verify_realized_dataset(base_dataset, source)
    derived_dataset = augment_request_shape_features(base_dataset, trajectories)
    if (
        derived_dataset.source_descriptor_hash != base_dataset.source_descriptor_hash
        or derived_dataset.capability_contract_hash
        != base_dataset.capability_contract_hash
        or derived_dataset.dataset_id == base_dataset.dataset_id
    ):
        raise Stage2ExperimentError("Stage 2 request-shape projection changed source identity")
    return Stage2LoadedSource(
        source_name=source_name,
        source_lock=source,
        base_dataset_id=base_dataset.dataset_id,
        base_row_count=len(base_dataset.rows),
        derived_dataset=derived_dataset,
        raw_paths=raw_paths,
    )


def _verify_source_inputs(
    root: Path,
    lock_context: LockContext,
    loaded: Stage2LoadedSource,
) -> None:
    current_context = load_lock_context(root, lock_context.baseline_lock_path)
    if current_context != lock_context:
        raise Stage2ExperimentError("Data Foundation lock changed during Stage 2 execution")
    if loaded.source_name == "bagen_swebench":
        if _load_bagen_manifest(root, loaded.source_lock) != loaded.raw_paths:
            raise Stage2ExperimentError("BAGEN raw membership changed during Stage 2 execution")
    else:
        if (_verify_spend_archive(root, loaded.source_lock),) != loaded.raw_paths:
            raise Stage2ExperimentError("Spend archive identity changed during Stage 2 execution")


def _task_pseudonym(task_id: str, *, split_plan_id: str) -> str:
    return hashlib.sha256(
        f"{STAGE2_TASK_PSEUDONYM_POLICY_ID}\0{split_plan_id}\0{task_id}".encode("utf-8")
    ).hexdigest()


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
    digest = hashlib.sha256(f"{STAGE2_PREDICTION_PROJECTION_ID}\0".encode("ascii"))
    for record in sorted(result.predictions, key=lambda item: item.point_id):
        payload = _canonical_json_bytes(_prediction_document(result, record))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def cohort_projection_sha256(result: CandidateResult) -> str:
    digest = hashlib.sha256(f"{STAGE2_COHORT_PROJECTION_ID}\0".encode("ascii"))
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
    projected: list[dict[str, object]] = []
    for task_id, metrics in sorted(result.task_metrics.items()):
        projected.append(
            {
                "task_pseudonym": _task_pseudonym(
                    task_id,
                    split_plan_id=result.split_plan_id,
                ),
                **dict(metrics),
            }
        )
    return sorted(projected, key=lambda item: str(item["task_pseudonym"]))


def _numeric_seed_aggregate(
    seed_metrics: Sequence[Mapping[str, float | int | str]],
) -> dict[str, Mapping[str, float]]:
    if not seed_metrics:
        raise Stage2ExperimentError("cannot aggregate an empty Stage 2 seed set")
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
            raise Stage2ExperimentError("Stage 2 seed metrics contain non-finite values")
        mean = sum(numeric) / len(numeric)
        variance = sum((value - mean) ** 2 for value in numeric) / len(numeric)
        result[key] = {
            "mean": mean,
            "minimum": min(numeric),
            "maximum": max(numeric),
            "population_stddev": math.sqrt(variance),
        }
    return result


def _comparison_document(value: Any) -> dict[str, object]:
    return dict(asdict(value))


def _result_document(
    result: CandidateResult,
    *,
    require_reloadable_bundle: bool,
) -> dict[str, object]:
    bundle_folds = [
        artifact.fold
        for artifact in result.fold_artifacts
        if artifact.bundle_files is not None
    ]
    if require_reloadable_bundle and bundle_folds != list(range(5)):
        raise Stage2ExperimentError(
            f"candidate {result.candidate_id!r} lacks five reloadable fold bundles"
        )
    return {
        "candidate_id": result.candidate_id,
        "candidate_hash": result.candidate_hash,
        "comparability_key": list(result.comparability_key),
        "split_plan_id": result.split_plan_id,
        "prediction_count": len(result.predictions),
        "prediction_projection_id": STAGE2_PREDICTION_PROJECTION_ID,
        "prediction_projection_sha256": prediction_projection_sha256(result),
        "cohort_projection_id": STAGE2_COHORT_PROJECTION_ID,
        "cohort_projection_sha256": cohort_projection_sha256(result),
        "metrics": dict(result.metrics),
        "fold_metrics": {
            str(fold): dict(metrics) for fold, metrics in result.fold_metrics.items()
        },
        "task_metric_policy_id": STAGE2_TASK_PSEUDONYM_POLICY_ID,
        "task_metrics": _task_metric_projection(result),
        "fold_artifact_count": len(result.fold_artifacts),
        "reloadable_bundle_folds": bundle_folds,
        "bundle_reload_parity": {
            "status": (
                "exact_during_execution"
                if require_reloadable_bundle
                else "not_applicable_stateless_or_mechanical"
            ),
            "fold_count": len(bundle_folds),
        },
    }


def build_stage2_results(
    execution: DevelopmentExperimentResults,
    matrix: Stage2Matrix,
    *,
    source_name: str,
    loaded: Stage2LoadedSource,
    lock_context: LockContext,
    code_binding: Stage2CodeBinding,
    runtime_versions: Mapping[str, str],
    run_id: str,
) -> dict[str, object]:
    if execution.protocol.protocol_id != matrix.development_protocol_id:
        raise Stage2ExperimentError("Stage 2 execution and matrix protocol ids differ")
    if tuple(item.split_seed for item in execution.seed_results) != STAGE_SPLIT_SEEDS:
        raise Stage2ExperimentError("Stage 2 execution does not contain all frozen seeds")

    experiments: list[dict[str, object]] = []
    candidate_seed_run_count = 0
    for spec_index, spec in enumerate(matrix.experiments):
        candidate_documents: list[dict[str, object]] = []
        for candidate in spec.candidates:
            per_seed: list[dict[str, object]] = []
            raw_seed_metrics: list[Mapping[str, float | int | str]] = []
            for seed_result in execution.seed_results:
                group = seed_result.result_groups[spec_index]
                result = next(
                    (item for item in group if item.candidate_id == candidate.candidate_id),
                    None,
                )
                if result is None:
                    raise Stage2ExperimentError("Stage 2 candidate result is missing")
                reference = next(
                    (item for item in group if item.candidate_id == "empirical"),
                    None,
                )
                if reference is None:
                    raise Stage2ExperimentError("Stage 2 empirical reference is missing")
                requires_bundle = candidate.estimator_id in {
                    "independent_mlp",
                    "lightgbm_quantile",
                } or candidate.graph.is_lifecycle
                result_document = _result_document(
                    result,
                    require_reloadable_bundle=requires_bundle,
                )
                result_document["split_seed"] = seed_result.split_seed
                if candidate.candidate_id != "empirical":
                    comparison_seed = int(
                        hashlib.sha256(
                            (
                                f"stage2-paired-bootstrap-v1\0{seed_result.split_seed}\0"
                                f"{spec.experiment_id}\0{candidate.candidate_id}"
                            ).encode("utf-8")
                        ).hexdigest()[:16],
                        16,
                    )
                    comparison = paired_task_metric_bootstrap(
                        result,
                        reference,
                        iterations=10_000,
                        seed=comparison_seed,
                    )
                    result_document["paired_vs_empirical"] = _comparison_document(comparison)
                per_seed.append(result_document)
                raw_seed_metrics.append(result.metrics)
                candidate_seed_run_count += 1
            candidate_documents.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "candidate_hash": candidate.content_hash,
                    "estimator_id": candidate.estimator_id,
                    "feature_set_id": candidate.feature_set.feature_set_id,
                    "feature_set_hash": candidate.feature_set.content_hash,
                    "candidate_graph": candidate.graph.to_dict(),
                    "seed_results": per_seed,
                    "cross_seed_metrics": _numeric_seed_aggregate(raw_seed_metrics),
                }
            )
        experiments.append(
            {
                "experiment_id": spec.experiment_id,
                "position": spec.position.value,
                "target": spec.target.value,
                "condition_id": spec.condition_id,
                "alpha": spec.alpha,
                "calibrator_id": spec.calibrator_id,
                "candidates": candidate_documents,
            }
        )

    results: dict[str, object] = {
        "results_schema_version": STAGE2_RESULTS_SCHEMA_VERSION,
        "stage_name": STAGE2_STAGE_NAME,
        "run_policy_id": STAGE2_RUN_POLICY_ID,
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
            "request_shape_projection": "request_boundary_shape_v1",
        },
        "development_protocol": execution.audit_document,
        "matrix": matrix.identity_document(),
        "experiments": experiments,
        "summary": {
            "experiment_count": len(matrix.experiments),
            "candidate_seed_run_count": candidate_seed_run_count,
            "split_seeds": list(STAGE_SPLIT_SEEDS),
            "outer_folds": 5,
            "inner_folds": 5,
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


def verify_stage2_results_document(value: Mapping[str, object]) -> str:
    required = {
        "results_schema_version",
        "stage_name",
        "run_policy_id",
        "run_id",
        "source",
        "data_foundation",
        "code_binding",
        "runtime_versions",
        "dataset",
        "development_protocol",
        "matrix",
        "experiments",
        "summary",
        "final_holdout",
        "results_payload_sha256",
    }
    if set(value) != required:
        raise Stage2ExperimentError("Stage 2 results keys do not match the schema")
    if value["results_schema_version"] != STAGE2_RESULTS_SCHEMA_VERSION:
        raise Stage2ExperimentError("unsupported Stage 2 results schema")
    if value["stage_name"] != STAGE2_STAGE_NAME or value["run_policy_id"] != STAGE2_RUN_POLICY_ID:
        raise Stage2ExperimentError("Stage 2 results policy identity is invalid")
    _assert_aggregate_safe(value)
    holdout = value["final_holdout"]
    if not isinstance(holdout, Mapping) or holdout != {
        "evaluated": False,
        "prediction_count": 0,
        "target_values_used_for_fit_calibration_scoring": False,
        "selection_claim": "none",
    }:
        raise Stage2ExperimentError("Stage 2 final holdout is not sealed")
    expected = dict(value)
    declared = _required_sha256(
        expected.pop("results_payload_sha256"),
        name="Stage 2 results payload SHA-256",
    )
    if _semantic_sha256(expected) != declared:
        raise Stage2ExperimentError("Stage 2 results payload SHA-256 does not close")
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
    matrix: Stage2Matrix,
) -> None:
    for spec_index, spec in enumerate(matrix.experiments):
        for candidate in spec.candidates:
            for seed_result in execution.seed_results:
                result = next(
                    item
                    for item in seed_result.result_groups[spec_index]
                    if item.candidate_id == candidate.candidate_id
                )
                _write_fold_artifacts(
                    root,
                    experiment_id=spec.experiment_id,
                    candidate_id=result.candidate_id,
                    split_seed=seed_result.split_seed,
                    artifacts=result.fold_artifacts,
                )


def _safe_output_root(root: Path, relative: str) -> tuple[str, Path]:
    try:
        canonical = _safe_relative(relative, label="Stage 2 output root")
    except DataFoundationBaselineError as exc:
        raise Stage2ExperimentError("Stage 2 output root is not a safe relative path") from exc
    prefix = ALLOWED_OUTPUT_PREFIX.rstrip("/")
    if canonical != prefix and not canonical.startswith(ALLOWED_OUTPUT_PREFIX):
        raise Stage2ExperimentError(
            f"Stage 2 output root must be {prefix!r} or a descendant"
        )
    try:
        resolved = _repo_path(root, canonical, label="Stage 2 output root")
    except DataFoundationBaselineError as exc:
        raise Stage2ExperimentError("Stage 2 output root escapes the repository") from exc
    return canonical, resolved


def _run_semantic(
    *,
    source_name: str,
    loaded: Stage2LoadedSource,
    lock_context: LockContext,
    code_binding: Stage2CodeBinding,
    runtime_versions: Mapping[str, str],
    matrix: Stage2Matrix,
) -> dict[str, object]:
    return {
        "results_schema_version": STAGE2_RESULTS_SCHEMA_VERSION,
        "run_policy_id": STAGE2_RUN_POLICY_ID,
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
) -> Stage2SourceSummary:
    manifest = verify_artifact(output)
    metadata = manifest.metadata
    if metadata.get("run_id") != run_id or metadata.get("run_semantic") != dict(
        expected_semantic
    ):
        raise Stage2ExperimentError("existing Stage 2 artifact has another identity")
    results_path = output / "results.json"
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Stage2ExperimentError("existing Stage 2 results are unreadable") from exc
    if not isinstance(results, dict):
        raise Stage2ExperimentError("existing Stage 2 results must be an object")
    payload_hash = verify_stage2_results_document(results)
    summary = results["summary"]
    source = results["source"]
    matrix = results["matrix"]
    protocol = results["development_protocol"]
    if not all(isinstance(item, Mapping) for item in (summary, source, matrix, protocol)):
        raise Stage2ExperimentError("existing Stage 2 result sections are invalid")
    return Stage2SourceSummary(
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


def run_stage2_source(
    *,
    repository_root: str | Path,
    source_name: str,
    baseline_lock: str = DEFAULT_BASELINE_LOCK,
    output_root: str = DEFAULT_OUTPUT_ROOT,
) -> Stage2SourceSummary:
    supplied_root = Path(repository_root)
    if _is_link_or_reparse(supplied_root):
        raise Stage2ExperimentError("repository root must not be linked or reparse-backed")
    root = supplied_root.resolve()
    if not root.is_dir():
        raise Stage2ExperimentError("repository root is not a directory")
    _verify_runner_origin(root)
    _canonical_output_root, output_parent = _safe_output_root(root, output_root)
    code_binding = capture_stage2_code_binding(root)
    lock_context = load_lock_context(root, baseline_lock)
    loaded = load_stage2_source(root, lock_context, source_name=source_name)
    protocol = build_development_protocol(loaded.derived_dataset)
    matrix = build_stage2_matrix(
        protocol,
        source_id=loaded.source_lock.descriptor.source_id,
    )
    if not matrix.experiments:
        raise Stage2ExperimentError("Stage 2 matrix has no estimable experiments")
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
    output = output_parent / f"{source_name}-{run_id}"
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
    results = build_stage2_results(
        execution,
        matrix,
        source_name=source_name,
        loaded=loaded,
        lock_context=lock_context,
        code_binding=code_binding,
        runtime_versions=runtime_versions,
        run_id=run_id,
    )
    results_payload_sha256 = verify_stage2_results_document(results)

    output_parent.mkdir(parents=True, exist_ok=True)
    if _is_link_or_reparse(output_parent):
        raise Stage2ExperimentError("Stage 2 output parent is unsafe")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{source_name}-{run_id}.tmp-", dir=output_parent)
    )
    try:
        _write_results(temporary / "results.json", results)
        _write_execution_artifacts(temporary, execution, matrix)
        before_publish_code = capture_stage2_code_binding(root)
        if before_publish_code != code_binding:
            raise Stage2ExperimentError("Stage 2 code changed during execution")
        _verify_source_inputs(root, lock_context, loaded)
        manifest = publish_artifact(
            temporary,
            stage_name=STAGE2_STAGE_NAME,
            schema_version=STAGE2_ARTIFACT_SCHEMA_VERSION,
            metadata={
                "run_id": run_id,
                "run_semantic": run_semantic,
                "results_payload_sha256": results_payload_sha256,
            },
        )
        if capture_stage2_code_binding(root) != code_binding:
            raise Stage2ExperimentError("Stage 2 code changed during artifact publication")
        _verify_source_inputs(root, lock_context, loaded)
        if output.exists():
            raise FileExistsError(f"Stage 2 artifact destination appeared: {output}")
        os.replace(temporary, output)
        if verify_artifact(output) != manifest:
            raise Stage2ExperimentError("published Stage 2 artifact failed verification")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    return Stage2SourceSummary(
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
        description="Run one immutable Stage 2 source experiment artifact."
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
        summary = run_stage2_source(
            repository_root=args.repository_root,
            source_name=args.source,
            baseline_lock=args.baseline_lock,
            output_root=args.output_root,
        )
    except (DataFoundationBaselineError, Stage2ExperimentError, ValueError) as exc:
        raise SystemExit(f"Stage 2 experiment failed: {exc}") from exc
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
