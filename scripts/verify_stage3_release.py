"""Verify the frozen Stage 3 release lock, report, and local artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from token_prediction.contracts import SourceDescriptor
from token_prediction.development import STAGE_SPLIT_SEEDS
from token_prediction.estimators.lightgbm_bundle import load_lightgbm_bundle
from token_prediction.estimators.neural_bundle import load_neural_bundle
from token_prediction.lifecycle_bundle import load_lifecycle_bundle
from token_prediction.lineage import verify_artifact
from token_prediction.stage3_matrix import (
    FROZEN_STAGE3_SOURCE_CONDITIONS,
    STAGE3_BUDGET_THRESHOLDS,
    STAGE3_MIN_DEVELOPMENT_TASKS,
)

if __package__:
    from scripts.run_stage2_experiments import (
        STAGE2_COHORT_PROJECTION_ID,
        Stage2ExperimentError as Stage2BaselineError,
        verify_stage2_results_document,
    )
    from scripts.verify_stage2_release import (
        _validate_release_document as _validate_stage2_release_document,
    )
    from scripts.run_stage3_experiments import (
        DATA_FOUNDATION_BASELINE_RELATIVE,
        STAGE1_VERIFIER_RELATIVE,
        STAGE2_AUXILIARY_MANIFEST_RELATIVE,
        STAGE2_METADATA_EXTRACTOR_RELATIVE,
        STAGE2_RUNNER_RELATIVE,
        STAGE2_SOKOBAN_AUDITOR_RELATIVE,
        STAGE3_ARTIFACT_SCHEMA_VERSION,
        STAGE3_CHECKPOINT_POLICY_ID,
        STAGE3_COHORT_PROJECTION_ID,
        STAGE3_RUNNER_RELATIVE,
        STAGE3_RUN_POLICY_ID,
        STAGE3_STAGE_NAME,
        SOURCE_NAMES,
        Stage3ExperimentError,
        _framed_code_hash,
        _git,
        _is_link_or_reparse,
        _repo_path,
        _required_sha256,
        _safe_relative,
        verify_stage3_results_document,
    )
else:  # pragma: no cover - production CLI invocation
    from run_stage2_experiments import (
        STAGE2_COHORT_PROJECTION_ID,
        Stage2ExperimentError as Stage2BaselineError,
        verify_stage2_results_document,
    )
    from verify_stage2_release import (
        _validate_release_document as _validate_stage2_release_document,
    )
    from run_stage3_experiments import (
        DATA_FOUNDATION_BASELINE_RELATIVE,
        STAGE1_VERIFIER_RELATIVE,
        STAGE2_AUXILIARY_MANIFEST_RELATIVE,
        STAGE2_METADATA_EXTRACTOR_RELATIVE,
        STAGE2_RUNNER_RELATIVE,
        STAGE2_SOKOBAN_AUDITOR_RELATIVE,
        STAGE3_ARTIFACT_SCHEMA_VERSION,
        STAGE3_CHECKPOINT_POLICY_ID,
        STAGE3_COHORT_PROJECTION_ID,
        STAGE3_RUNNER_RELATIVE,
        STAGE3_RUN_POLICY_ID,
        STAGE3_STAGE_NAME,
        SOURCE_NAMES,
        Stage3ExperimentError,
        _framed_code_hash,
        _git,
        _is_link_or_reparse,
        _repo_path,
        _required_sha256,
        _safe_relative,
        verify_stage3_results_document,
    )


DEFAULT_RELEASE_LOCK = "configs/stage3_release.json"
STAGE2_RELEASE_LOCK = "configs/stage2_release.json"
STAGE2_REGRESSION_POLICY_ID = "exact_non_neural_and_stage3_mlp_contract_v2"
RELEASE_SCHEMA_VERSION = 1
RELEASE_STAGE_NAME = "stage3_development"
RELEASE_POLICY_ID = "stage3_commit_bound_four_source_release_v1"
RELEASE_TAG = "stage3-artifact-source-v1"
ARTIFACT_NAMES = frozenset(
    {"spend_aggregate", "bagen_sokoban", "bagen_swebench", "spend_openhands"}
)
GATE_ARTIFACT_NAME = "spend_aggregate"
EXPECTED_CANDIDATES = {
    "empirical": "empirical_quantile",
    "cross_position_deduct": "cross_position_deduct",
    "lightgbm_history": "lightgbm_quantile",
    "mlp_history": "independent_mlp",
    "gru_residual": "gru_residual",
    "gru_no_recurrence": "gru_residual",
    "gru_zero_residual": "gru_residual",
}
LIFECYCLE_CANDIDATES = frozenset(
    {
        "cross_position_deduct",
        "gru_residual",
        "gru_no_recurrence",
        "gru_zero_residual",
    }
)
STAGE2_EXACT_REGRESSION_CANDIDATES = frozenset(
    {
        "empirical",
        "cross_position_deduct",
        "lightgbm_history",
    }
)
STAGE2_RUNTIME_SCOPED_CANDIDATES = frozenset({"mlp_history"})
STAGE2_REGRESSION_CANDIDATES = STAGE2_EXACT_REGRESSION_CANDIDATES | STAGE2_RUNTIME_SCOPED_CANDIDATES
MAX_RELEASE_JSON_BYTES = 1024 * 1024
MAX_RESULTS_JSON_BYTES = 32 * 1024 * 1024
MAX_REPORT_BYTES = 1024 * 1024
MAX_FOLD_PROVENANCE_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class Stage3ReleaseVerification:
    lock_path: str
    report_path: str
    code_tree_sha256: str
    artifact_commit_status: str
    locked_artifact_count: int
    verified_artifact_count: int
    manifest_file_count: int
    candidate_seed_run_count: int
    lifecycle_candidate_seed_run_count: int
    exact_lifecycle_reload_fold_count: int
    reloadable_bundle_fold_count: int
    independently_loaded_bundle_count: int
    stage2_regression_candidate_seed_run_count: int
    final_holdout_evaluated: bool


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Stage3ExperimentError("Stage 3 release JSON contains duplicate keys")
        result[key] = value
    return result


def _constant(value: str) -> Any:
    raise Stage3ExperimentError(f"Stage 3 release JSON contains {value}")


def _load_json(path: Path, *, maximum_bytes: int, description: str) -> Mapping[str, Any]:
    if _is_link_or_reparse(path) or not path.is_file():
        raise Stage3ExperimentError(f"{description} must be a regular non-link file")
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise Stage3ExperimentError(f"{description} has an invalid size")
    try:
        value = json.loads(
            path.read_bytes().decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage3ExperimentError(f"{description} is not strict UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise Stage3ExperimentError(f"{description} must contain a JSON object")
    return value


def _exact(value: object, keys: set[str], *, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise Stage3ExperimentError(f"{description} keys do not match")
    return value


def _integer(value: object, *, description: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Stage3ExperimentError(f"{description} must be an integer >= {minimum}")
    return value


def _text(value: object, *, description: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise Stage3ExperimentError(f"{description} must be non-empty normalized text")
    return value


def _require_measured_latency(value: object, *, description: str) -> None:
    if not isinstance(value, Mapping):
        raise Stage3ExperimentError(f"{description} metrics must be an object")
    p50 = value.get("latency_p50_ms")
    p95 = value.get("latency_p95_ms")
    if (
        isinstance(p50, bool)
        or not isinstance(p50, (int, float))
        or isinstance(p95, bool)
        or not isinstance(p95, (int, float))
        or not math.isfinite(float(p50))
        or not math.isfinite(float(p95))
        or float(p50) <= 0
        or float(p95) < float(p50)
    ):
        raise Stage3ExperimentError(f"{description} lacks measured prediction latency")


def _latency_neutral(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _latency_neutral(item)
            for key, item in value.items()
            if key not in {"latency_p50_ms", "latency_p95_ms"}
        }
    if isinstance(value, list):
        return [_latency_neutral(item) for item in value]
    return value


def _stage2_regression_neutral(value: object) -> object:
    """Remove only run-local timing and pseudonym salts from regression evidence."""

    if isinstance(value, Mapping):
        return {
            key: _stage2_regression_neutral(item)
            for key, item in value.items()
            if key
            not in {
                "latency_p50_ms",
                "latency_p95_ms",
                "task_pseudonym",
            }
        }
    if isinstance(value, list):
        return [_stage2_regression_neutral(item) for item in value]
    return value


def _task_metric_multiset(value: object, *, description: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, Mapping) for item in value)
    ):
        raise Stage3ExperimentError(f"{description} task metrics are invalid")
    rendered = []
    for item in value:
        normalized = _stage2_regression_neutral(item)
        try:
            rendered.append(
                json.dumps(
                    normalized,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError) as exc:
            raise Stage3ExperimentError(
                f"{description} task metrics are not finite canonical JSON"
            ) from exc
    return tuple(sorted(rendered))


def _require_shared_stage2_stage3_cohort(
    stage2_seed: Mapping[str, Any],
    stage3_seed: Mapping[str, Any],
    *,
    description: str,
) -> None:
    """Validate one cohort across the stage-specific projection namespaces."""

    if any(
        stage2_seed.get(field) != stage3_seed.get(field)
        for field in (
            "split_plan_id",
            "comparability_key",
            "prediction_count",
        )
    ):
        raise Stage3ExperimentError(f"{description} cohort differs")
    if (
        stage2_seed.get("cohort_projection_id") != STAGE2_COHORT_PROJECTION_ID
        or stage3_seed.get("cohort_projection_id") != STAGE3_COHORT_PROJECTION_ID
    ):
        raise Stage3ExperimentError(f"{description} cohort projection namespace differs")
    _required_sha256(
        stage2_seed.get("cohort_projection_sha256"),
        name=f"{description} Stage 2 cohort projection",
    )
    _required_sha256(
        stage3_seed.get("cohort_projection_sha256"),
        name=f"{description} Stage 3 cohort projection",
    )


_ARTIFACT_KEYS = {
    "kind",
    "path",
    "source_id",
    "source_descriptor_hash",
    "run_id",
    "artifact_id",
    "results_payload_sha256",
    "base_dataset_id",
    "derived_dataset_id",
    "development_dataset_id",
    "development_protocol_id",
    "matrix_id",
    "experiment_count",
    "candidate_seed_run_count",
    "lifecycle_candidate_seed_run_count",
    "exact_lifecycle_reload_fold_count",
    "reloadable_bundle_fold_count",
    "independently_loaded_bundle_count",
    "stage2_regression_candidate_seed_run_count",
    "manifest_file_count",
}
_TOTAL_KEYS = {
    "artifact_count",
    "experiment_artifact_count",
    "gate_artifact_count",
    "experiment_count",
    "candidate_seed_run_count",
    "lifecycle_candidate_seed_run_count",
    "exact_lifecycle_reload_fold_count",
    "reloadable_bundle_fold_count",
    "independently_loaded_bundle_count",
    "stage2_regression_candidate_seed_run_count",
    "manifest_file_count",
}


def _validate_release_document(value: Mapping[str, Any]) -> None:
    _exact(
        value,
        {
            "release_schema_version",
            "stage_name",
            "policy_id",
            "code_binding",
            "protocol",
            "stage2_regression",
            "artifacts",
            "totals",
            "report",
        },
        description="Stage 3 release lock",
    )
    if (
        value["release_schema_version"] != RELEASE_SCHEMA_VERSION
        or value["stage_name"] != RELEASE_STAGE_NAME
        or value["policy_id"] != RELEASE_POLICY_ID
    ):
        raise Stage3ExperimentError("Stage 3 release lock identity is invalid")
    code = _exact(
        value["code_binding"],
        {"artifact_git_commit", "artifact_git_tag", "code_tree_sha256"},
        description="Stage 3 release code binding",
    )
    commit = _text(code["artifact_git_commit"], description="artifact Git commit")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise Stage3ExperimentError("artifact Git commit must be a full lowercase SHA")
    if code["artifact_git_tag"] != RELEASE_TAG:
        raise Stage3ExperimentError("Stage 3 artifact release tag is invalid")
    _required_sha256(code["code_tree_sha256"], name="Stage 3 code tree")

    protocol = _exact(
        value["protocol"],
        {
            "outer_folds",
            "inner_folds",
            "split_seeds",
            "run_policy_id",
            "checkpoint_policy_id",
            "checkpoint_interval_epochs",
            "neural_training_device",
            "neural_inference_device",
            "calibrator_id",
            "alpha",
            "budget_thresholds",
            "progress_checkpoints",
            "final_holdout_evaluated",
            "final_holdout_prediction_count",
        },
        description="Stage 3 release protocol",
    )
    if protocol != {
        "outer_folds": 5,
        "inner_folds": 5,
        "split_seeds": [20260719, 20260720, 20260721],
        "run_policy_id": STAGE3_RUN_POLICY_ID,
        "checkpoint_policy_id": STAGE3_CHECKPOINT_POLICY_ID,
        "checkpoint_interval_epochs": 1,
        "neural_training_device": "cuda",
        "neural_inference_device": "cpu",
        "calibrator_id": "task_max_conformal",
        "alpha": 0.1,
        "budget_thresholds": list(STAGE3_BUDGET_THRESHOLDS),
        "progress_checkpoints": [0.25, 0.5, 0.75],
        "final_holdout_evaluated": False,
        "final_holdout_prediction_count": 0,
    }:
        raise Stage3ExperimentError("Stage 3 release protocol is not frozen")

    regression = _exact(
        value["stage2_regression"],
        {
            "release_lock_path",
            "release_lock_sha256",
            "exact_candidate_ids",
            "runtime_scoped_candidate_ids",
            "normalization_policy_id",
            "candidate_seed_run_count",
        },
        description="Stage 2 regression binding",
    )
    if (
        regression["release_lock_path"] != STAGE2_RELEASE_LOCK
        or regression["exact_candidate_ids"] != sorted(STAGE2_EXACT_REGRESSION_CANDIDATES)
        or regression["runtime_scoped_candidate_ids"] != sorted(STAGE2_RUNTIME_SCOPED_CANDIDATES)
        or regression["normalization_policy_id"] != STAGE2_REGRESSION_POLICY_ID
    ):
        raise Stage3ExperimentError("Stage 2 regression binding is invalid")
    _required_sha256(
        regression["release_lock_sha256"],
        name="Stage 2 release lock",
    )
    _integer(
        regression["candidate_seed_run_count"],
        description="Stage 2 regression candidate-seed run count",
    )

    artifacts = value["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != ARTIFACT_NAMES:
        raise Stage3ExperimentError("Stage 3 release artifact set is incomplete")
    summed = {key: 0 for key in _TOTAL_KEYS}
    summed["artifact_count"] = len(artifacts)
    artifact_paths: set[str] = set()
    artifact_ids: set[str] = set()
    run_ids: set[str] = set()
    for name, raw in artifacts.items():
        entry = _exact(raw, _ARTIFACT_KEYS, description=f"Stage 3 artifact {name}")
        expected_kind = "gate" if name == GATE_ARTIFACT_NAME else "experiment"
        if entry["kind"] != expected_kind:
            raise Stage3ExperimentError(f"Stage 3 artifact {name} has another kind")
        relative = _safe_relative(entry["path"], label=f"Stage 3 artifact {name} path")
        if not relative.startswith("workspace/stage3/runs/s3-"):
            raise Stage3ExperimentError(f"Stage 3 artifact {name} path is outside release root")
        if entry["source_id"] != SOURCE_NAMES[name]:
            raise Stage3ExperimentError(f"Stage 3 artifact {name} source id differs")
        _required_sha256(
            entry["source_descriptor_hash"],
            name=f"Stage 3 artifact {name} source descriptor",
        )
        run_id = _text(entry["run_id"], description=f"Stage 3 artifact {name} run id")
        if len(run_id) != 24 or any(character not in "0123456789abcdef" for character in run_id):
            raise Stage3ExperimentError(f"Stage 3 artifact {name} run id is invalid")
        for field in ("artifact_id", "results_payload_sha256"):
            _required_sha256(entry[field], name=f"Stage 3 artifact {name} {field}")
        for field in (
            "base_dataset_id",
            "derived_dataset_id",
            "development_dataset_id",
            "development_protocol_id",
            "matrix_id",
        ):
            identifier = _text(entry[field], description=f"Stage 3 artifact {name} {field}")
            _required_sha256(
                identifier.removeprefix("spend-your-money:"),
                name=f"Stage 3 artifact {name} {field}",
            )
        count_fields = (
            "experiment_count",
            "candidate_seed_run_count",
            "lifecycle_candidate_seed_run_count",
            "exact_lifecycle_reload_fold_count",
            "reloadable_bundle_fold_count",
            "independently_loaded_bundle_count",
            "stage2_regression_candidate_seed_run_count",
        )
        for field in count_fields:
            _integer(entry[field], description=f"Stage 3 artifact {name} {field}")
            summed[field] += int(entry[field])
        file_count = _integer(
            entry["manifest_file_count"],
            description=f"Stage 3 artifact {name} manifest file count",
            minimum=1,
        )
        summed["manifest_file_count"] += file_count
        artifact_paths.add(relative)
        artifact_ids.add(str(entry["artifact_id"]))
        run_ids.add(run_id)
        if expected_kind == "gate":
            summed["gate_artifact_count"] += 1
            if any(entry[field] != 0 for field in count_fields):
                raise Stage3ExperimentError("Stage 3 gate artifact contains model runs")
        else:
            summed["experiment_artifact_count"] += 1
            experiment_count = int(entry["experiment_count"])
            expected_counts = {
                "candidate_seed_run_count": 21 * experiment_count,
                "lifecycle_candidate_seed_run_count": 12 * experiment_count,
                "exact_lifecycle_reload_fold_count": 60 * experiment_count,
                "reloadable_bundle_fold_count": 90 * experiment_count,
                "independently_loaded_bundle_count": 90 * experiment_count,
                "stage2_regression_candidate_seed_run_count": 12 * experiment_count,
            }
            if experiment_count < 1 or any(
                entry[field] != expected for field, expected in expected_counts.items()
            ):
                raise Stage3ExperimentError(
                    "Stage 3 experiment artifact cardinalities differ from the frozen matrix"
                )

    if not (
        len(artifact_paths) == len(artifacts)
        and len(artifact_ids) == len(artifacts)
        and len(run_ids) == len(artifacts)
    ):
        raise Stage3ExperimentError("Stage 3 release artifact identities are not unique")

    totals = _exact(value["totals"], _TOTAL_KEYS, description="Stage 3 release totals")
    normalized_totals = {
        key: _integer(raw, description=f"Stage 3 total {key}") for key, raw in totals.items()
    }
    if normalized_totals != summed:
        raise Stage3ExperimentError("Stage 3 release totals do not close over artifacts")
    if (
        regression["candidate_seed_run_count"]
        != normalized_totals["stage2_regression_candidate_seed_run_count"]
    ):
        raise Stage3ExperimentError("Stage 2 regression binding count differs")
    if (
        normalized_totals["artifact_count"] != 4
        or normalized_totals["experiment_artifact_count"] != 3
        or normalized_totals["gate_artifact_count"] != 1
        or normalized_totals["exact_lifecycle_reload_fold_count"]
        != 5 * normalized_totals["lifecycle_candidate_seed_run_count"]
        or normalized_totals["independently_loaded_bundle_count"]
        != normalized_totals["reloadable_bundle_fold_count"]
        or normalized_totals["stage2_regression_candidate_seed_run_count"]
        != 12 * normalized_totals["experiment_count"]
    ):
        raise Stage3ExperimentError("Stage 3 release cardinalities are invalid")

    report = _exact(value["report"], {"path", "sha256"}, description="Stage 3 report")
    if _safe_relative(report["path"], label="Stage 3 report path") != "docs/stage-3-report.md":
        raise Stage3ExperimentError("Stage 3 report path is invalid")
    _required_sha256(report["sha256"], name="Stage 3 report")


_EXPLICIT_CODE_PATHS = frozenset(
    {
        DATA_FOUNDATION_BASELINE_RELATIVE,
        STAGE1_VERIFIER_RELATIVE,
        STAGE2_AUXILIARY_MANIFEST_RELATIVE,
        STAGE2_METADATA_EXTRACTOR_RELATIVE,
        STAGE2_RUNNER_RELATIVE,
        STAGE2_SOKOBAN_AUDITOR_RELATIVE,
        STAGE3_RUNNER_RELATIVE,
    }
)


def _code_paths_at_commit(root: Path, commit: str) -> tuple[str, ...]:
    raw = _git(
        root,
        "ls-tree",
        "-r",
        "--name-only",
        "-z",
        commit,
        "--",
        "src/token_prediction",
        *sorted(_EXPLICIT_CODE_PATHS),
    )
    paths = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        try:
            relative = item.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise Stage3ExperimentError("Git returned a non-UTF-8 Stage 3 path") from exc
        relative = _safe_relative(relative, label="Stage 3 historical code path")
        if relative in _EXPLICIT_CODE_PATHS or (
            relative.startswith("src/token_prediction/") and relative.endswith(".py")
        ):
            paths.append(relative)
    resolved = tuple(sorted(set(paths)))
    if not _EXPLICIT_CODE_PATHS <= set(resolved) or not any(
        path.startswith("src/token_prediction/") for path in resolved
    ):
        raise Stage3ExperimentError("historical commit lacks the complete Stage 3 source set")
    return resolved


def _code_hash_at_commit(root: Path, commit: str) -> tuple[str, tuple[str, ...]]:
    paths = _code_paths_at_commit(root, commit)
    items = [(relative, _git(root, "show", f"{commit}:{relative}")) for relative in paths]
    return _framed_code_hash(items), paths


def _resolve_artifact_source(
    root: Path,
    commit: str,
    tag: str,
    expected_code_hash: str,
) -> tuple[str, tuple[str, ...]]:
    tagged = _git(root, "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}")
    if tagged.decode("ascii").strip() != commit:
        raise Stage3ExperimentError("Stage 3 release tag does not point to artifact commit")
    actual, paths = _code_hash_at_commit(root, commit)
    if actual != expected_code_hash:
        raise Stage3ExperimentError("artifact Git commit does not reproduce the code tree")
    return "artifact_tag_commit_and_source_tree_verified", paths


def _require_tracked_clean(root: Path, paths: Sequence[str]) -> None:
    tracked = {
        item.decode("utf-8", errors="strict")
        for item in _git(root, "ls-files", "-z", "--", *paths).split(b"\0")
        if item
    }
    if tracked != set(paths):
        raise Stage3ExperimentError("Stage 3 release controls must be tracked")
    if _git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--",
        *paths,
    ):
        raise Stage3ExperimentError("Stage 3 release controls must be clean at HEAD")


def _bundle_mapping(directory: Path) -> Mapping[str, bytes]:
    if _is_link_or_reparse(directory) or not directory.is_dir():
        raise Stage3ExperimentError("lifecycle bundle directory is unsafe")
    files: dict[str, bytes] = {}
    for path in sorted(directory.rglob("*")):
        if _is_link_or_reparse(path):
            raise Stage3ExperimentError("lifecycle bundle contains a link/reparse point")
        if path.is_dir():
            continue
        if not path.is_file():
            raise Stage3ExperimentError("lifecycle bundle contains a special node")
        relative = path.relative_to(directory).as_posix()
        files[relative] = path.read_bytes()
    if not files:
        raise Stage3ExperimentError("lifecycle bundle directory is empty")
    return files


def _verify_fold_provenance(
    fold_root: Path,
    *,
    experiment: Mapping[str, Any],
    candidate: Mapping[str, Any],
    seed_result: Mapping[str, Any],
    fold: int,
    is_lifecycle: bool,
    expected_source_id: str,
    expected_source_descriptor_hash: str,
    expected_capability_contract_hash: str,
    expected_code_hash: str,
    expected_dataset_id: str,
    expected_input_contract_hash: str,
) -> Mapping[str, Any]:
    provenance = _load_json(
        fold_root / "provenance.json",
        maximum_bytes=MAX_FOLD_PROVENANCE_BYTES,
        description="Stage 3 fold provenance",
    )
    required = {
        "candidate_id": candidate["candidate_id"],
        "candidate_hash": candidate["candidate_hash"],
        "dataset_id": expected_dataset_id,
        "condition_id": experiment["condition_id"],
        "source_descriptor_hash": expected_source_descriptor_hash,
        "capability_contract_hash": expected_capability_contract_hash,
        "input_contract_hash": expected_input_contract_hash,
        "code_hash": expected_code_hash,
        "split_plan_id": seed_result["split_plan_id"],
        "position": "task_update",
        "target": "task_provider_accounted_remaining_tokens",
        "calibrator_id": "task_max_conformal",
        "interval_alpha": 0.1,
    }
    if any(provenance.get(key) != value for key, value in required.items()):
        raise Stage3ExperimentError("Stage 3 fold provenance differs from release scope")
    fold_field = "outer_fold" if is_lifecycle else "fold"
    if provenance.get(fold_field) != fold:
        raise Stage3ExperimentError("Stage 3 fold provenance index differs")
    descriptor_document = provenance.get("source_descriptor")
    if descriptor_document is None:
        if candidate["estimator_id"] != "lightgbm_quantile":
            raise Stage3ExperimentError("Stage 3 fold provenance lacks a source descriptor")
    else:
        if not isinstance(descriptor_document, Mapping):
            raise Stage3ExperimentError("Stage 3 fold provenance source descriptor is invalid")
        try:
            descriptor = SourceDescriptor.from_dict(descriptor_document)
        except (TypeError, ValueError) as exc:
            raise Stage3ExperimentError(
                "Stage 3 fold provenance source descriptor is invalid"
            ) from exc
        if (
            descriptor.source_id != expected_source_id
            or descriptor.descriptor_hash != expected_source_descriptor_hash
            or descriptor.capabilities.contract_hash != expected_capability_contract_hash
        ):
            raise Stage3ExperimentError(
                "Stage 3 fold provenance source descriptor differs from release scope"
            )
    return provenance


def _load_declared_bundles(
    artifact_path: Path,
    experiment: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    expected_source_id: str,
    expected_source_descriptor_hash: str,
    expected_capability_contract_hash: str,
    expected_code_hash: str,
    expected_dataset_id: str,
    expected_input_contract_hash: str,
) -> int:
    estimator_id = str(candidate["estimator_id"])
    graph = candidate["candidate_graph"]
    is_lifecycle = graph["initializer_estimator_id"] != "none"
    count = 0
    for seed_result in candidate["seed_results"]:
        seed = int(seed_result["split_seed"])
        for fold in seed_result["reloadable_bundle_folds"]:
            fold_root = (
                artifact_path
                / "fold_artifacts"
                / str(experiment["artifact_key"])
                / str(candidate["artifact_key"])
                / f"seed_{seed}"
                / f"fold_{int(fold)}"
            )
            provenance = _verify_fold_provenance(
                fold_root,
                experiment=experiment,
                candidate=candidate,
                seed_result=seed_result,
                fold=int(fold),
                is_lifecycle=is_lifecycle,
                expected_source_id=expected_source_id,
                expected_source_descriptor_hash=expected_source_descriptor_hash,
                expected_capability_contract_hash=expected_capability_contract_hash,
                expected_code_hash=expected_code_hash,
                expected_dataset_id=expected_dataset_id,
                expected_input_contract_hash=expected_input_contract_hash,
            )
            bundle = fold_root / "bundle"
            if is_lifecycle:
                loaded = load_lifecycle_bundle(_bundle_mapping(bundle))
                if dict(loaded.manifest) != dict(provenance):
                    raise Stage3ExperimentError(
                        "Stage 3 lifecycle bundle provenance differs from fold provenance"
                    )
            elif estimator_id == "independent_mlp":
                loaded = load_neural_bundle(bundle)
                if dict(loaded.provenance or {}) != dict(provenance):
                    raise Stage3ExperimentError(
                        "Stage 3 neural bundle provenance differs from fold provenance"
                    )
            elif estimator_id == "lightgbm_quantile":
                load_lightgbm_bundle(bundle)
            else:
                raise Stage3ExperimentError(
                    "Stage 3 result declares a bundle for an unsupported estimator"
                )
            count += 1
    return count


def _load_stage2_regression_results(
    root: Path,
    name: str,
    entry: Mapping[str, Any],
) -> Mapping[str, Any]:
    relative = _safe_relative(entry.get("path"), label=f"Stage 2 artifact {name} path")
    if not relative.startswith("workspace/stage2/experiments/s2-"):
        raise Stage3ExperimentError(f"Stage 2 artifact {name} path is outside release root")
    artifact_path = _repo_path(root, relative, label=f"Stage 2 artifact {name}")
    manifest = verify_artifact(artifact_path)
    artifact_id = _required_sha256(
        entry.get("artifact_id"),
        name=f"Stage 2 artifact {name} id",
    )
    results_sha256 = _required_sha256(
        entry.get("results_payload_sha256"),
        name=f"Stage 2 artifact {name} results payload",
    )
    if (
        manifest.artifact_id != artifact_id
        or manifest.metadata.get("results_payload_sha256") != results_sha256
    ):
        raise Stage3ExperimentError(f"Stage 2 artifact {name} differs from its release lock")
    results = _load_json(
        artifact_path / "results.json",
        maximum_bytes=MAX_RESULTS_JSON_BYTES,
        description=f"Stage 2 artifact {name} results",
    )
    if verify_stage2_results_document(results) != results_sha256:
        raise Stage3ExperimentError(f"Stage 2 artifact {name} results hash differs")
    return results


def _load_stage3_regression_results(
    root: Path,
    name: str,
    entry: Mapping[str, Any],
) -> Mapping[str, Any]:
    artifact_path = _repo_path(root, entry["path"], label=f"Stage 3 artifact {name}")
    results = _load_json(
        artifact_path / "results.json",
        maximum_bytes=MAX_RESULTS_JSON_BYTES,
        description=f"Stage 3 artifact {name} regression results",
    )
    if verify_stage3_results_document(results) != entry["results_payload_sha256"]:
        raise Stage3ExperimentError(f"Stage 3 artifact {name} regression hash differs")
    return results


def _verify_stage2_regression(
    root: Path,
    release: Mapping[str, Any],
    stage2_release: Mapping[str, Any],
) -> int:
    stage2_artifacts = stage2_release.get("artifacts")
    if not isinstance(stage2_artifacts, Mapping):
        raise Stage3ExperimentError("Stage 2 release artifact set is invalid")

    compared = 0
    for name in sorted(ARTIFACT_NAMES - {GATE_ARTIFACT_NAME}):
        source_start = compared
        raw_stage2_entry = stage2_artifacts.get(name)
        raw_stage3_entry = release["artifacts"][name]
        if not isinstance(raw_stage2_entry, Mapping) or not isinstance(raw_stage3_entry, Mapping):
            raise Stage3ExperimentError(f"Stage 2 regression artifact {name} is missing")
        stage2_results = _load_stage2_regression_results(
            root,
            name,
            raw_stage2_entry,
        )
        stage3_results = _load_stage3_regression_results(
            root,
            name,
            raw_stage3_entry,
        )
        stage2_dataset = stage2_results["dataset"]
        stage3_dataset = stage3_results["dataset"]
        if (
            stage2_results["source"]["source_id"] != raw_stage3_entry["source_id"]
            or stage2_results["source"]["source_descriptor_hash"]
            != raw_stage3_entry["source_descriptor_hash"]
            or any(
                stage2_dataset[field] != stage3_dataset[field]
                for field in (
                    "base_dataset_id",
                    "derived_dataset_id",
                    "development_dataset_id",
                )
            )
            or stage2_results["development_protocol"]["protocol_id"]
            != stage3_results["development_protocol"]["protocol_id"]
        ):
            raise Stage3ExperimentError(
                f"Stage 2 regression artifact {name} cohort identity differs"
            )

        stage2_experiments = {
            (
                experiment["condition_id"],
                experiment["position"],
                experiment["target"],
            ): experiment
            for experiment in stage2_results["experiments"]
        }
        for stage3_experiment in stage3_results["experiments"]:
            key = (
                stage3_experiment["condition_id"],
                stage3_experiment["position"],
                stage3_experiment["target"],
            )
            stage2_experiment = stage2_experiments.get(key)
            if stage2_experiment is None:
                raise Stage3ExperimentError(
                    f"Stage 2 regression artifact {name} lacks a Stage 3 cell"
                )
            if any(
                stage2_experiment[field] != stage3_experiment[field]
                for field in ("alpha", "calibrator_id")
            ):
                raise Stage3ExperimentError(
                    f"Stage 2 regression artifact {name} calibration contract differs"
                )
            stage2_candidates = {
                candidate["candidate_id"]: candidate
                for candidate in stage2_experiment["candidates"]
            }
            stage3_candidates = {
                candidate["candidate_id"]: candidate
                for candidate in stage3_experiment["candidates"]
            }
            if not STAGE2_REGRESSION_CANDIDATES <= set(stage2_candidates) or not (
                STAGE2_REGRESSION_CANDIDATES <= set(stage3_candidates)
            ):
                raise Stage3ExperimentError(
                    f"Stage 2 regression artifact {name} lacks shared candidates"
                )
            for candidate_id in sorted(STAGE2_REGRESSION_CANDIDATES):
                stage2_candidate = stage2_candidates[candidate_id]
                stage3_candidate = stage3_candidates[candidate_id]
                shared_identity_fields = (
                    "estimator_id",
                    "feature_set_hash",
                    "feature_set_id",
                    "candidate_graph",
                )
                if any(
                    stage2_candidate[field] != stage3_candidate[field]
                    for field in shared_identity_fields
                ):
                    raise Stage3ExperimentError(
                        f"Stage 2 regression artifact {name} candidate identity differs"
                    )
                exact_regression = candidate_id in STAGE2_EXACT_REGRESSION_CANDIDATES
                if exact_regression:
                    if stage2_candidate["candidate_hash"] != stage3_candidate[
                        "candidate_hash"
                    ] or _stage2_regression_neutral(
                        stage2_candidate["cross_seed_metrics"]
                    ) != _stage2_regression_neutral(stage3_candidate["cross_seed_metrics"]):
                        raise Stage3ExperimentError(
                            f"Stage 2 regression artifact {name} candidate identity differs"
                        )
                elif stage2_candidate["candidate_hash"] == stage3_candidate["candidate_hash"]:
                    raise Stage3ExperimentError(
                        f"Stage 2 regression artifact {name} Stage 3 runtime identity is invalid"
                    )
                stage2_seeds = {
                    seed_result["split_seed"]: seed_result
                    for seed_result in stage2_candidate["seed_results"]
                }
                stage3_seeds = {
                    seed_result["split_seed"]: seed_result
                    for seed_result in stage3_candidate["seed_results"]
                }
                if set(stage2_seeds) != set(STAGE_SPLIT_SEEDS) or set(stage3_seeds) != set(
                    STAGE_SPLIT_SEEDS
                ):
                    raise Stage3ExperimentError(
                        f"Stage 2 regression artifact {name} split seeds differ"
                    )
                for split_seed in STAGE_SPLIT_SEEDS:
                    stage2_seed = stage2_seeds[split_seed]
                    stage3_seed = stage3_seeds[split_seed]
                    _require_shared_stage2_stage3_cohort(
                        stage2_seed,
                        stage3_seed,
                        description=f"Stage 2 regression artifact {name}",
                    )
                    if exact_regression and any(
                        _stage2_regression_neutral(stage2_seed[field])
                        != _stage2_regression_neutral(stage3_seed[field])
                        for field in ("metrics", "fold_metrics")
                    ):
                        raise Stage3ExperimentError(
                            f"Stage 2 regression artifact {name} numerical metrics differ"
                        )
                    if exact_regression and _task_metric_multiset(
                        stage2_seed["task_metrics"],
                        description=f"Stage 2 artifact {name}",
                    ) != _task_metric_multiset(
                        stage3_seed["task_metrics"],
                        description=f"Stage 3 artifact {name}",
                    ):
                        raise Stage3ExperimentError(
                            f"Stage 2 regression artifact {name} task metrics differ"
                        )
                    compared += 1

        expected = raw_stage3_entry["stage2_regression_candidate_seed_run_count"]
        source_compared = compared - source_start
        if source_compared != expected:
            raise Stage3ExperimentError(
                f"Stage 2 regression artifact {name} count differs from release lock"
            )
    return compared


def _verify_experiment_entry(
    root: Path,
    name: str,
    entry: Mapping[str, Any],
    *,
    expected_code_hash: str,
    expected_commit: str,
    expected_code_paths: tuple[str, ...],
) -> tuple[int, int, int, int, int, int]:
    artifact_path = _repo_path(root, entry["path"], label=f"Stage 3 artifact {name}")
    manifest = verify_artifact(artifact_path)
    if (
        manifest.stage_name != STAGE3_STAGE_NAME
        or manifest.schema_version != STAGE3_ARTIFACT_SCHEMA_VERSION
        or manifest.artifact_id != entry["artifact_id"]
        or len(manifest.files) != entry["manifest_file_count"]
        or manifest.metadata.get("run_id") != entry["run_id"]
        or manifest.metadata.get("results_payload_sha256") != entry["results_payload_sha256"]
    ):
        raise Stage3ExperimentError(f"Stage 3 artifact {name} manifest differs from lock")
    results = _load_json(
        artifact_path / "results.json",
        maximum_bytes=MAX_RESULTS_JSON_BYTES,
        description=f"Stage 3 artifact {name} results",
    )
    if verify_stage3_results_document(results) != entry["results_payload_sha256"]:
        raise Stage3ExperimentError(f"Stage 3 artifact {name} results hash differs")
    code = results["code_binding"]
    source = results["source"]
    dataset = results["dataset"]
    protocol = results["development_protocol"]
    matrix = results["matrix"]
    summary = results["summary"]
    if (
        results["run_id"] != entry["run_id"]
        or code["git_commit"] != expected_commit
        or code["code_tree_sha256"] != expected_code_hash
        or tuple(code["code_paths"]) != expected_code_paths
        or source["source_id"] != entry["source_id"]
        or source["source_descriptor_hash"] != entry["source_descriptor_hash"]
        or dataset["base_dataset_id"] != entry["base_dataset_id"]
        or dataset["derived_dataset_id"] != entry["derived_dataset_id"]
        or dataset["development_dataset_id"] != entry["development_dataset_id"]
        or protocol["protocol_id"] != entry["development_protocol_id"]
        or matrix["matrix_id"] != entry["matrix_id"]
        or summary["experiment_count"] != entry["experiment_count"]
        or summary["candidate_seed_run_count"] != entry["candidate_seed_run_count"]
    ):
        raise Stage3ExperimentError(f"Stage 3 artifact {name} identity differs from lock")
    experiments = results["experiments"]
    if (
        not isinstance(experiments, list)
        or not all(isinstance(experiment, Mapping) for experiment in experiments)
        or len(experiments) != entry["experiment_count"]
    ):
        raise Stage3ExperimentError(f"Stage 3 artifact {name} experiment count differs")
    gates = results["gates"]
    if (
        not isinstance(gates, list)
        or not all(isinstance(gate, Mapping) for gate in gates)
        or any(
            experiment.get("position") != "task_update"
            or experiment.get("target") != "task_provider_accounted_remaining_tokens"
            for experiment in experiments
        )
        or any(
            gate.get("source_id") != SOURCE_NAMES[name]
            or gate.get("position") != "task_update"
            or gate.get("target") != "task_provider_accounted_remaining_tokens"
            for gate in gates
        )
    ):
        raise Stage3ExperimentError(f"Stage 3 artifact {name} cell semantics differ")
    experiment_conditions = [experiment["condition_id"] for experiment in experiments]
    gate_conditions = [gate["condition_id"] for gate in gates]
    expected_conditions = FROZEN_STAGE3_SOURCE_CONDITIONS[SOURCE_NAMES[name]]
    if len(set(experiment_conditions + gate_conditions)) != len(experiment_conditions) + len(
        gate_conditions
    ) or set(experiment_conditions + gate_conditions) != set(expected_conditions):
        raise Stage3ExperimentError(f"Stage 3 artifact {name} condition coverage differs")
    if name == GATE_ARTIFACT_NAME:
        if (
            experiments
            or len(gates) != 1
            or gates[0].get("reason") != "aggregate_source_has_no_request_boundary_lifecycle"
            or gates[0].get("development_task_count") != 0
            or gates[0].get("eligible_point_count") != 0
        ):
            raise Stage3ExperimentError(f"Stage 3 artifact {name} gate semantics differ")
    else:
        for gate in gates:
            task_count = _integer(
                gate.get("development_task_count"),
                description=f"Stage 3 artifact {name} gate task count",
            )
            _integer(
                gate.get("eligible_point_count"),
                description=f"Stage 3 artifact {name} gate point count",
            )
            reason = gate.get("reason")
            if reason == "insufficient_development_tasks_for_five_fold_cv":
                if not 0 < task_count < STAGE3_MIN_DEVELOPMENT_TASKS:
                    raise Stage3ExperimentError(
                        f"Stage 3 artifact {name} sparse gate count differs"
                    )
            elif reason != "capability_or_observed_development_lifecycle_unavailable":
                raise Stage3ExperimentError(f"Stage 3 artifact {name} gate reason differs")
    if results["final_holdout"] != {
        "evaluated": False,
        "prediction_count": 0,
        "selection_claim": "none",
        "target_values_used_for_fit_calibration_scoring": False,
    }:
        raise Stage3ExperimentError(f"Stage 3 artifact {name} final holdout is not sealed")

    candidate_runs = 0
    lifecycle_runs = 0
    lifecycle_folds = 0
    reloadable_folds = 0
    loaded_bundles = 0
    for experiment in experiments:
        if experiment["alpha"] != 0.1 or experiment["calibrator_id"] != "task_max_conformal":
            raise Stage3ExperimentError(f"Stage 3 artifact {name} calibration differs")
        candidates = experiment["candidates"]
        if (
            not isinstance(candidates, list)
            or not all(isinstance(candidate, Mapping) for candidate in candidates)
            or {candidate["candidate_id"] for candidate in candidates} != set(EXPECTED_CANDIDATES)
            or any(
                candidate["estimator_id"] != EXPECTED_CANDIDATES[candidate["candidate_id"]]
                for candidate in candidates
            )
        ):
            raise Stage3ExperimentError(f"Stage 3 artifact {name} candidate set differs")
        for candidate in candidates:
            seed_results = candidate["seed_results"]
            if [item["split_seed"] for item in seed_results] != list(STAGE_SPLIT_SEEDS):
                raise Stage3ExperimentError(f"Stage 3 artifact {name} seeds differ")
            is_lifecycle = candidate["candidate_graph"]["initializer_estimator_id"] != "none"
            if is_lifecycle != (candidate["candidate_id"] in LIFECYCLE_CANDIDATES):
                raise Stage3ExperimentError(f"Stage 3 artifact {name} candidate DAG differs")
            for seed_result in seed_results:
                candidate_runs += 1
                _require_measured_latency(
                    seed_result["metrics"],
                    description=(
                        f"Stage 3 artifact {name} candidate "
                        f"{candidate['candidate_id']} seed {seed_result['split_seed']}"
                    ),
                )
                parity = seed_result["bundle_reload_parity"]
                folds = _integer(
                    parity["fold_count"],
                    description=f"Stage 3 artifact {name} reload fold count",
                )
                if parity["status"] == "exact_during_execution":
                    if folds != 5 or seed_result["reloadable_bundle_folds"] != [0, 1, 2, 3, 4]:
                        raise Stage3ExperimentError(
                            f"Stage 3 artifact {name} exact reload omitted folds"
                        )
                    reloadable_folds += folds
                elif parity["status"] == "not_applicable_stateless_or_mechanical":
                    if folds != 0 or seed_result["reloadable_bundle_folds"]:
                        raise Stage3ExperimentError(
                            f"Stage 3 artifact {name} non-bundle candidate has bundles"
                        )
                else:
                    raise Stage3ExperimentError(
                        f"Stage 3 artifact {name} has unsupported reload status"
                    )
                lifecycle = seed_result["stage3_evaluation"]["lifecycle"]
                budget = seed_result["stage3_evaluation"]["budget"]
                scenarios = budget["scenarios"]
                if (
                    budget["threshold_policy"] != "explicit_fixed_remaining_token_budgets_v1"
                    or not isinstance(scenarios, Mapping)
                    or set(scenarios) != {str(threshold) for threshold in STAGE3_BUDGET_THRESHOLDS}
                ):
                    raise Stage3ExperimentError(f"Stage 3 artifact {name} budget protocol differs")
                if is_lifecycle:
                    if lifecycle["status"] != "complete_calibrated_trajectory_replay_exact":
                        raise Stage3ExperimentError(
                            f"Stage 3 artifact {name} lacks trajectory replay parity"
                        )
                    progress = lifecycle["progress"]
                    if set(progress["strata"]) != {"p25", "p50", "p75"}:
                        raise Stage3ExperimentError(
                            f"Stage 3 artifact {name} progress strata differ"
                        )
                    if lifecycle["termination"]["stratification_id"] != (
                        "lifecycle_termination_strata_v1"
                    ):
                        raise Stage3ExperimentError(
                            f"Stage 3 artifact {name} termination strata differ"
                        )
                    lifecycle_runs += 1
                    lifecycle_folds += folds
                elif lifecycle != {"status": "not_applicable_point_candidate"}:
                    raise Stage3ExperimentError(
                        f"Stage 3 artifact {name} point lifecycle status differs"
                    )
            loaded_bundles += _load_declared_bundles(
                artifact_path,
                experiment,
                candidate,
                expected_source_id=entry["source_id"],
                expected_source_descriptor_hash=entry["source_descriptor_hash"],
                expected_capability_contract_hash=source["capability_contract_hash"],
                expected_code_hash=expected_code_hash,
                expected_dataset_id=entry["development_dataset_id"],
                expected_input_contract_hash=protocol["development_dataset"]["input_contract_hash"],
            )
        by_candidate = {candidate["candidate_id"]: candidate for candidate in candidates}
        deduct_results = by_candidate["cross_position_deduct"]["seed_results"]
        zero_results = by_candidate["gru_zero_residual"]["seed_results"]
        for deduct, zero in zip(deduct_results, zero_results, strict=True):
            if (
                deduct["split_seed"] != zero["split_seed"]
                or deduct["cohort_projection_sha256"] != zero["cohort_projection_sha256"]
                or deduct["task_metrics"] != zero["task_metrics"]
                or any(
                    _latency_neutral(deduct[field]) != _latency_neutral(zero[field])
                    for field in ("metrics", "fold_metrics", "stage3_evaluation")
                )
            ):
                raise Stage3ExperimentError(
                    f"Stage 3 artifact {name} zero residual does not reduce to Deduct"
                )
    if (
        candidate_runs != entry["candidate_seed_run_count"]
        or lifecycle_runs != entry["lifecycle_candidate_seed_run_count"]
        or lifecycle_folds != entry["exact_lifecycle_reload_fold_count"]
        or reloadable_folds != entry["reloadable_bundle_fold_count"]
        or loaded_bundles != entry["independently_loaded_bundle_count"]
    ):
        raise Stage3ExperimentError(f"Stage 3 artifact {name} run totals do not close")
    return (
        len(manifest.files),
        candidate_runs,
        lifecycle_runs,
        lifecycle_folds,
        reloadable_folds,
        loaded_bundles,
    )


def verify_stage3_release(
    repository_root: str | Path,
    *,
    lock_path: str = DEFAULT_RELEASE_LOCK,
    tracked_only: bool = False,
    require_git_clean: bool = True,
) -> Stage3ReleaseVerification:
    root = Path(repository_root).resolve()
    relative_lock = _safe_relative(lock_path, label="Stage 3 release lock path")
    lock_file = _repo_path(root, relative_lock, label="Stage 3 release lock")
    release = _load_json(
        lock_file,
        maximum_bytes=MAX_RELEASE_JSON_BYTES,
        description="Stage 3 release lock",
    )
    _validate_release_document(release)
    regression = release["stage2_regression"]
    stage2_lock_path = _repo_path(
        root,
        regression["release_lock_path"],
        label="Stage 2 release lock",
    )
    if (
        _is_link_or_reparse(stage2_lock_path)
        or not stage2_lock_path.is_file()
        or not 0 < stage2_lock_path.stat().st_size <= MAX_RELEASE_JSON_BYTES
        or hashlib.sha256(stage2_lock_path.read_bytes()).hexdigest()
        != regression["release_lock_sha256"]
    ):
        raise Stage3ExperimentError("Stage 2 release lock differs from regression binding")
    stage2_release = _load_json(
        stage2_lock_path,
        maximum_bytes=MAX_RELEASE_JSON_BYTES,
        description="Stage 2 release lock",
    )
    _validate_stage2_release_document(stage2_release)
    report = release["report"]
    report_path = _repo_path(root, report["path"], label="Stage 3 report")
    if (
        _is_link_or_reparse(report_path)
        or not report_path.is_file()
        or not 0 < report_path.stat().st_size <= MAX_REPORT_BYTES
    ):
        raise Stage3ExperimentError("Stage 3 report is not a safe bounded file")
    report_payload = report_path.read_bytes()
    if (
        hashlib.sha256(report_payload).hexdigest() != report["sha256"]
        or b"<PENDING_" in report_payload
    ):
        raise Stage3ExperimentError("Stage 3 report differs from release lock")
    controls = (
        relative_lock,
        str(report["path"]),
        "scripts/verify_stage3_release.py",
        STAGE2_RELEASE_LOCK,
    )
    if require_git_clean:
        _require_tracked_clean(root, controls)

    code = release["code_binding"]
    expected_code = str(code["code_tree_sha256"])
    expected_commit = str(code["artifact_git_commit"])
    commit_status, expected_paths = _resolve_artifact_source(
        root,
        expected_commit,
        str(code["artifact_git_tag"]),
        expected_code,
    )

    verified_artifacts = 0
    totals = {
        "manifest_file_count": 0,
        "candidate_seed_run_count": 0,
        "lifecycle_candidate_seed_run_count": 0,
        "exact_lifecycle_reload_fold_count": 0,
        "reloadable_bundle_fold_count": 0,
        "independently_loaded_bundle_count": 0,
        "stage2_regression_candidate_seed_run_count": 0,
    }
    if not tracked_only:
        for name in sorted(ARTIFACT_NAMES):
            counts = _verify_experiment_entry(
                root,
                name,
                release["artifacts"][name],
                expected_code_hash=expected_code,
                expected_commit=expected_commit,
                expected_code_paths=expected_paths,
            )
            (
                manifest_file_count,
                candidate_seed_run_count,
                lifecycle_candidate_seed_run_count,
                exact_lifecycle_reload_fold_count,
                reloadable_bundle_fold_count,
                independently_loaded_bundle_count,
            ) = counts
            totals["manifest_file_count"] += manifest_file_count
            totals["candidate_seed_run_count"] += candidate_seed_run_count
            totals["lifecycle_candidate_seed_run_count"] += lifecycle_candidate_seed_run_count
            totals["exact_lifecycle_reload_fold_count"] += exact_lifecycle_reload_fold_count
            totals["reloadable_bundle_fold_count"] += reloadable_bundle_fold_count
            totals["independently_loaded_bundle_count"] += independently_loaded_bundle_count
            verified_artifacts += 1
        totals["stage2_regression_candidate_seed_run_count"] = _verify_stage2_regression(
            root, release, stage2_release
        )
        release_totals = release["totals"]
        for key in (
            "manifest_file_count",
            "candidate_seed_run_count",
            "lifecycle_candidate_seed_run_count",
            "exact_lifecycle_reload_fold_count",
            "reloadable_bundle_fold_count",
            "independently_loaded_bundle_count",
            "stage2_regression_candidate_seed_run_count",
        ):
            if totals[key] != release_totals[key]:
                raise Stage3ExperimentError("Stage 3 verified totals differ from release lock")

    return Stage3ReleaseVerification(
        lock_path=relative_lock,
        report_path=str(report["path"]),
        code_tree_sha256=expected_code,
        artifact_commit_status=commit_status,
        locked_artifact_count=len(release["artifacts"]),
        verified_artifact_count=verified_artifacts,
        manifest_file_count=totals["manifest_file_count"],
        candidate_seed_run_count=totals["candidate_seed_run_count"],
        lifecycle_candidate_seed_run_count=totals["lifecycle_candidate_seed_run_count"],
        exact_lifecycle_reload_fold_count=totals["exact_lifecycle_reload_fold_count"],
        reloadable_bundle_fold_count=totals["reloadable_bundle_fold_count"],
        independently_loaded_bundle_count=totals["independently_loaded_bundle_count"],
        stage2_regression_candidate_seed_run_count=totals[
            "stage2_regression_candidate_seed_run_count"
        ],
        final_holdout_evaluated=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the frozen Stage 3 release")
    parser.add_argument(
        "--repository-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument("--lock", default=DEFAULT_RELEASE_LOCK)
    parser.add_argument(
        "--tracked-only",
        action="store_true",
        help="verify tracked controls and code binding without ignored local artifacts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = verify_stage3_release(
            args.repository_root,
            lock_path=args.lock,
            tracked_only=args.tracked_only,
        )
    except (OSError, Stage2BaselineError, Stage3ExperimentError, ValueError) as exc:
        raise SystemExit(f"Stage 3 release verification failed: {exc}") from exc
    print(json.dumps(asdict(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
