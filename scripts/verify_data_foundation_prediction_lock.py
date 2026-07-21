from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from scripts import run_data_foundation_baseline as runner
    from scripts.audit_data_foundation_v2 import verify_audit_payload
    from scripts.verify_data_foundation_baseline import (
        _assert_privacy_safe,
        _validate_baseline,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    import run_data_foundation_baseline as runner  # type: ignore[no-redef]
    from audit_data_foundation_v2 import verify_audit_payload  # type: ignore[no-redef]
    from verify_data_foundation_baseline import (  # type: ignore[no-redef]
        _assert_privacy_safe,
        _validate_baseline,
    )


LOCK_SCHEMA_VERSION = 1
LOCK_TYPE = "data_foundation_prediction_baseline"
DEFAULT_LOCK = Path("configs/data_foundation_prediction_baseline.json")
DEFAULT_ARTIFACT = Path(runner.DEFAULT_OUTPUT)
REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BUNDLE_COUNT = 90
EXPECTED_CELL_COUNT = 6
EXPECTED_GATED_CONDITION_COUNT = 4
EXPECTED_SOURCE_NAMES = {"bagen_swebench", "spend_openhands"}

_LOCK_KEYS = {
    "artifact",
    "conditions",
    "data_foundation",
    "holdout",
    "lock_type",
    "metrics",
    "prediction_lock_schema_version",
    "protocol",
    "runner",
    "sources",
}
_ARTIFACT_KEYS = {
    "artifact_id",
    "artifact_schema_version",
    "baseline_id",
    "manifest_bytes",
    "manifest_sha256",
    "relative_path",
    "results_bytes",
    "results_file_sha256",
    "results_payload_sha256",
}
_RUNNER_KEYS = {
    "code_blob_count",
    "code_paths_sha256",
    "code_tree_sha256",
    "git_commit",
    "runner_path",
    "tracked_control_blob_count",
    "tracked_control_paths_sha256",
    "tracked_control_tree_sha256",
}
_DATA_FOUNDATION_KEYS = {
    "audit_file_sha256",
    "audit_git_commit",
    "audit_path",
    "audit_payload_sha256",
    "audit_source_tree_sha256",
    "baseline_lock_file_sha256",
    "baseline_lock_path",
}
_PROTOCOL_KEYS = {
    "alpha",
    "calibrator_id",
    "candidate_id",
    "evaluation_scope",
    "final_holdout_evaluated",
    "final_holdout_prediction_count",
    "final_holdout_target_values_used_for_fit_calibration_scoring",
    "final_model_selection_claim",
    "fold_count",
    "metric_suite_id",
    "split_assignment_policy_id",
    "split_seeds",
    "weighting_id",
}
_HOLDOUT_KEYS = {
    "assignment_inputs",
    "bucket_count",
    "final_holdout_bucket_threshold_exclusive",
    "independent_of_split_seed_labels_and_suffixes",
    "policy_id",
    "policy_payload_sha256",
    "salt",
}
_CONDITION_KEYS = {
    "condition_gate_policy_id",
    "condition_gate_policy_sha256",
    "condition_projection_sha256",
    "estimable_condition_count",
    "estimable_condition_counts_by_source",
    "gated_condition_count",
    "gated_condition_counts_by_source",
}
_METRIC_KEYS = {
    "aggregate_metrics_sha256",
    "bundle_count",
    "cell_count",
    "prediction_count",
}
_SOURCE_KEYS = {
    "capability_contract_hash",
    "dataset_id",
    "dataset_row_count",
    "descriptor_file_sha256",
    "descriptor_path",
    "manifest_path",
    "manifest_sha256",
    "raw_artifact_bytes",
    "raw_artifact_path",
    "raw_artifact_sha256",
    "raw_artifact_sha256_kind",
    "revision",
    "source_descriptor_hash",
    "source_id",
}


class PredictionLockError(RuntimeError):
    """The production prediction artifact does not close to its tracked lock."""


@dataclass(frozen=True)
class GitTreeEvidence:
    commit: str
    paths: tuple[str, ...]
    payloads: tuple[tuple[str, bytes], ...]


def _exact(value: Any, keys: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PredictionLockError(f"{label} must be an object")
    actual = set(value)
    if actual != keys:
        raise PredictionLockError(
            f"{label} keys are not exact "
            f"(missing={sorted(keys - actual)!r}, extra={sorted(actual - keys)!r})"
        )
    return value


def _text(value: Any, *, label: str) -> str:
    try:
        return runner._require_text(value, label=label)
    except runner.DataFoundationBaselineError as exc:
        raise PredictionLockError(str(exc)) from exc


def _sha256(value: Any, *, label: str) -> str:
    try:
        return runner._require_sha256(value, label=label)
    except runner.DataFoundationBaselineError as exc:
        raise PredictionLockError(str(exc)) from exc


def _commit(value: Any, *, label: str) -> str:
    try:
        return runner._require_commit(value, label=label)
    except runner.DataFoundationBaselineError as exc:
        raise PredictionLockError(str(exc)) from exc


def _count(value: Any, *, label: str) -> int:
    try:
        return runner._require_non_negative_int(value, label=label)
    except runner.DataFoundationBaselineError as exc:
        raise PredictionLockError(str(exc)) from exc


def _number(value: Any, *, label: str) -> float:
    try:
        return runner._require_finite_number(value, label=label)
    except runner.DataFoundationBaselineError as exc:
        raise PredictionLockError(str(exc)) from exc


def _relative(value: Any, *, label: str) -> str:
    try:
        return runner._safe_relative(value, label=label)
    except runner.DataFoundationBaselineError as exc:
        raise PredictionLockError(str(exc)) from exc


def _repo_path(root: Path, relative: Any, *, label: str) -> Path:
    canonical = _relative(relative, label=label)
    try:
        return runner._repo_path(root, canonical, label=label)
    except runner.DataFoundationBaselineError as exc:
        raise PredictionLockError(str(exc)) from exc


def _regular_file(root: Path, relative: Any, *, label: str) -> Path:
    path = _repo_path(root, relative, label=label)
    if not path.is_file() or runner._is_link_or_reparse(path):
        raise PredictionLockError(f"{label} must be one regular, non-reparse file")
    return path


def _regular_directory(root: Path, relative: Any, *, label: str) -> Path:
    path = _repo_path(root, relative, label=label)
    if not path.is_dir() or runner._is_link_or_reparse(path):
        raise PredictionLockError(f"{label} must be one regular, non-reparse directory")
    return path


def _assert_regular_tree(root: Path, *, label: str) -> None:
    """Reject links, reparse points, and non-file/non-directory entries."""

    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise PredictionLockError(f"cannot enumerate {label}") from exc
        for entry in entries:
            path = Path(entry.path)
            if runner._is_link_or_reparse(path):
                raise PredictionLockError(
                    f"{label} must not contain symlinks, junctions, or reparse points"
                )
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError as exc:
                raise PredictionLockError(f"cannot inspect {label} member") from exc
            if stat.S_ISDIR(mode):
                pending.append(path)
            elif not stat.S_ISREG(mode):
                raise PredictionLockError(
                    f"{label} must contain only regular files and directories"
                )


def _file_sha256(path: Path) -> str:
    try:
        return runner._sha256_file(path)
    except runner.DataFoundationBaselineError as exc:
        raise PredictionLockError(str(exc)) from exc


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        return runner._load_json(path, label=label)
    except runner.DataFoundationBaselineError as exc:
        raise PredictionLockError(str(exc)) from exc


def _semantic_sha256(value: Any) -> str:
    try:
        return runner._semantic_sha256(value)
    except runner.DataFoundationBaselineError as exc:
        raise PredictionLockError(str(exc)) from exc


def _source_counts(values: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        source: sum(value.get("source_name") == source for value in values)
        for source in sorted(EXPECTED_SOURCE_NAMES)
    }


def _seed_controls(results: Mapping[str, Any]) -> tuple[str, str]:
    split_policies: set[str] = set()
    weighting_ids: set[str] = set()
    for cell in results["cells"]:
        for seed in cell["seed_results"]:
            split_policies.add(_text(seed.get("split_assignment_policy_id"), label="split policy"))
            weighting_ids.add(_text(seed.get("weighting_id"), label="weighting id"))
    if len(split_policies) != 1 or len(weighting_ids) != 1:
        raise PredictionLockError("artifact seed controls are not uniform")
    return next(iter(split_policies)), next(iter(weighting_ids))


def _control_paths(results: Mapping[str, Any]) -> tuple[str, ...]:
    binding = results["source_binding"]
    values = [binding["baseline_lock_path"]]
    values.extend(source["descriptor_path"] for source in results["sources"].values())
    paths = tuple(sorted(_relative(value, label="tracked control path") for value in values))
    if len(set(paths)) != len(paths) or len(paths) != 3:
        raise PredictionLockError(
            "tracked controls must be exactly the baseline lock and two descriptors"
        )
    return paths


def _condition_projection(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for status, entries in (
        ("estimable", results["cells"]),
        ("not_estimable", results["not_estimable_conditions"]),
    ):
        for item in entries:
            values.append(
                {
                    "condition_id": item["condition_id"],
                    "position": item["position"],
                    "source_name": item["source_name"],
                    "status": status,
                    "target": item["target"],
                }
            )
    return sorted(values, key=lambda item: (item["source_name"], item["condition_id"]))


def _metrics_projection(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "aggregate_metrics": cell["aggregate_metrics"],
                "condition_id": cell["condition_id"],
                "position": cell["position"],
                "source_name": cell["source_name"],
                "target": cell["target"],
            }
            for cell in results["cells"]
        ],
        key=lambda item: (item["source_name"], item["condition_id"]),
    )


def _prediction_count(results: Mapping[str, Any]) -> int:
    return sum(
        _count(seed["prediction_count"], label="seed prediction count")
        for cell in results["cells"]
        for seed in cell["seed_results"]
    )


def build_lock_projection(
    repo_root: Path,
    *,
    artifact_relative: str,
    manifest: Mapping[str, Any],
    results: Mapping[str, Any],
) -> dict[str, Any]:
    """Project an already verified artifact into the exact tracked lock schema."""

    root = repo_root.resolve(strict=True)
    relative = _relative(artifact_relative, label="prediction artifact path")
    artifact = _regular_directory(root, relative, label="prediction artifact")
    manifest_path = artifact / "manifest.json"
    results_path = artifact / "results.json"
    binding = results["source_binding"]
    holdout = results["permanent_final_holdout_policy"]
    gate_policy = results["condition_gate_policy"]
    cells = list(results["cells"])
    gates = list(results["not_estimable_conditions"])
    split_policy, weighting_id = _seed_controls(results)
    code_paths = tuple(binding["code_paths"])
    control_paths = _control_paths(results)
    return {
        "prediction_lock_schema_version": LOCK_SCHEMA_VERSION,
        "lock_type": LOCK_TYPE,
        "artifact": {
            "relative_path": relative,
            "artifact_schema_version": manifest["artifact_schema_version"],
            "baseline_id": manifest["baseline_id"],
            "artifact_id": manifest["artifact_id"],
            "manifest_bytes": manifest_path.stat().st_size,
            "manifest_sha256": _file_sha256(manifest_path),
            "results_bytes": results_path.stat().st_size,
            "results_file_sha256": _file_sha256(results_path),
            "results_payload_sha256": results["results_payload_sha256"],
        },
        "runner": {
            "runner_path": runner.RUNNER_RELATIVE,
            "git_commit": binding["git_commit"],
            "code_tree_sha256": binding["runner_and_src_code_tree_sha256"],
            "code_blob_count": len(code_paths),
            "code_paths_sha256": _semantic_sha256(list(code_paths)),
            "tracked_control_tree_sha256": binding["tracked_control_tree_sha256"],
            "tracked_control_blob_count": len(control_paths),
            "tracked_control_paths_sha256": _semantic_sha256(list(control_paths)),
        },
        "data_foundation": {
            "baseline_lock_path": binding["baseline_lock_path"],
            "baseline_lock_file_sha256": binding["baseline_lock_file_sha256"],
            "audit_path": binding["data_foundation_audit_path"],
            "audit_file_sha256": binding["data_foundation_audit_file_sha256"],
            "audit_payload_sha256": binding["data_foundation_audit_payload_sha256"],
            "audit_git_commit": binding["data_foundation_audit_git_commit"],
            "audit_source_tree_sha256": binding[
                "data_foundation_audit_source_tree_sha256"
            ],
        },
        "protocol": {
            "candidate_id": results["candidate_id"],
            "evaluation_scope": results["evaluation_scope"],
            "final_holdout_evaluated": results["final_holdout_evaluated"],
            "final_holdout_prediction_count": results["final_holdout_prediction_count"],
            "final_holdout_target_values_used_for_fit_calibration_scoring": results[
                "final_holdout_target_values_used_for_fit_calibration_scoring"
            ],
            "final_model_selection_claim": results["final_model_selection_claim"],
            "fold_count": results["fold_count"],
            "split_seeds": list(results["split_seeds"]),
            "alpha": results["alpha"],
            "calibrator_id": results["calibrator_id"],
            "metric_suite_id": results["metric_suite_id"],
            "split_assignment_policy_id": split_policy,
            "weighting_id": weighting_id,
        },
        "holdout": {
            "policy_id": holdout["policy_id"],
            "salt": holdout["salt"],
            "bucket_count": holdout["bucket_count"],
            "final_holdout_bucket_threshold_exclusive": holdout[
                "final_holdout_bucket_threshold_exclusive"
            ],
            "assignment_inputs": holdout["assignment_inputs"],
            "independent_of_split_seed_labels_and_suffixes": holdout[
                "independent_of_split_seed_labels_and_suffixes"
            ],
            "policy_payload_sha256": _semantic_sha256(holdout),
        },
        "conditions": {
            "condition_gate_policy_id": gate_policy["policy_id"],
            "condition_gate_policy_sha256": _semantic_sha256(gate_policy),
            "condition_projection_sha256": _semantic_sha256(
                _condition_projection(results)
            ),
            "estimable_condition_count": len(cells),
            "gated_condition_count": len(gates),
            "estimable_condition_counts_by_source": _source_counts(cells),
            "gated_condition_counts_by_source": _source_counts(gates),
        },
        "metrics": {
            "aggregate_metrics_sha256": _semantic_sha256(_metrics_projection(results)),
            "bundle_count": results["bundle_count"],
            "cell_count": len(cells),
            "prediction_count": _prediction_count(results),
        },
        "sources": {
            name: dict(value) for name, value in sorted(results["sources"].items())
        },
    }


def _validate_lock(value: Any) -> Mapping[str, Any]:
    try:
        _assert_privacy_safe(value, label="prediction lock")
    except Exception as exc:
        if isinstance(exc, PredictionLockError):
            raise
        raise PredictionLockError(str(exc)) from exc
    lock = _exact(value, _LOCK_KEYS, label="prediction lock")
    if lock["prediction_lock_schema_version"] != LOCK_SCHEMA_VERSION:
        raise PredictionLockError("prediction lock schema version is unsupported")
    if lock["lock_type"] != LOCK_TYPE:
        raise PredictionLockError("prediction lock type is unsupported")
    artifact = _exact(lock["artifact"], _ARTIFACT_KEYS, label="prediction lock.artifact")
    _relative(artifact["relative_path"], label="prediction artifact path")
    if (
        _count(artifact["artifact_schema_version"], label="artifact schema version")
        != runner.BASELINE_ARTIFACT_SCHEMA_VERSION
        or _text(artifact["baseline_id"], label="baseline id")
        != runner.BASELINE_ID
    ):
        raise PredictionLockError("prediction artifact schema or baseline id is unsupported")
    for key in (
        "artifact_id",
        "manifest_sha256",
        "results_file_sha256",
        "results_payload_sha256",
    ):
        _sha256(artifact[key], label=f"prediction lock.artifact.{key}")
    for key in ("manifest_bytes", "results_bytes"):
        _count(artifact[key], label=f"prediction lock.artifact.{key}")

    runner_lock = _exact(lock["runner"], _RUNNER_KEYS, label="prediction lock.runner")
    if _relative(runner_lock["runner_path"], label="runner path") != runner.RUNNER_RELATIVE:
        raise PredictionLockError("prediction lock runner path is unsupported")
    _commit(runner_lock["git_commit"], label="runner commit")
    for key in (
        "code_paths_sha256",
        "code_tree_sha256",
        "tracked_control_paths_sha256",
        "tracked_control_tree_sha256",
    ):
        _sha256(runner_lock[key], label=f"prediction lock.runner.{key}")
    for key in ("code_blob_count", "tracked_control_blob_count"):
        if _count(runner_lock[key], label=f"prediction lock.runner.{key}") == 0:
            raise PredictionLockError(f"prediction lock.runner.{key} must be positive")

    data = _exact(
        lock["data_foundation"], _DATA_FOUNDATION_KEYS, label="prediction lock.data_foundation"
    )
    for key in ("baseline_lock_path", "audit_path"):
        _relative(data[key], label=f"prediction lock.data_foundation.{key}")
    _commit(data["audit_git_commit"], label="Data Foundation audit commit")
    for key in (
        "audit_file_sha256",
        "audit_payload_sha256",
        "audit_source_tree_sha256",
        "baseline_lock_file_sha256",
    ):
        _sha256(data[key], label=f"prediction lock.data_foundation.{key}")

    protocol = _exact(lock["protocol"], _PROTOCOL_KEYS, label="prediction lock.protocol")
    for key in (
        "calibrator_id",
        "candidate_id",
        "evaluation_scope",
        "final_model_selection_claim",
        "metric_suite_id",
        "split_assignment_policy_id",
        "weighting_id",
    ):
        _text(protocol[key], label=f"prediction lock.protocol.{key}")
    if not isinstance(protocol["split_seeds"], list) or any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in protocol["split_seeds"]
    ):
        raise PredictionLockError("prediction lock split seeds are invalid")
    if (
        protocol["candidate_id"] != runner.CANDIDATE_ID
        or protocol["calibrator_id"] != runner.CALIBRATOR_ID
        or protocol["evaluation_scope"] != "development_cross_validation_only"
        or protocol["metric_suite_id"] != runner.METRIC_SUITE_ID
        or protocol["split_assignment_policy_id"]
        != runner.SPLIT_ASSIGNMENT_POLICY_ID
        or protocol["weighting_id"] != "task_run_point_equal_v1"
        or protocol["split_seeds"] != list(runner.SPLIT_SEEDS)
        or _count(protocol["fold_count"], label="prediction lock fold count")
        != runner.FOLDS
        or _number(protocol["alpha"], label="prediction lock alpha")
        != runner.ALPHA
    ):
        raise PredictionLockError("prediction lock protocol is not frozen")
    if (
        protocol["final_holdout_evaluated"] is not False
        or protocol["final_holdout_prediction_count"] != 0
        or protocol["final_holdout_target_values_used_for_fit_calibration_scoring"]
        is not False
        or protocol["final_model_selection_claim"] != "none"
    ):
        raise PredictionLockError("prediction lock makes an unauthorized final-holdout claim")

    holdout = _exact(lock["holdout"], _HOLDOUT_KEYS, label="prediction lock.holdout")
    for key in ("assignment_inputs", "policy_id", "salt"):
        _text(holdout[key], label=f"prediction lock.holdout.{key}")
    for key in ("bucket_count", "final_holdout_bucket_threshold_exclusive"):
        _count(holdout[key], label=f"prediction lock.holdout.{key}")
    if (
        holdout["independent_of_split_seed_labels_and_suffixes"] is not True
        or holdout["assignment_inputs"] != "task_id_only"
        or holdout["policy_id"] != runner.FINAL_HOLDOUT_POLICY_ID
        or holdout["salt"] != runner.FINAL_HOLDOUT_SALT
        or holdout["bucket_count"] != runner.FINAL_HOLDOUT_BUCKET_COUNT
        or holdout["final_holdout_bucket_threshold_exclusive"]
        != runner.FINAL_HOLDOUT_BUCKET_THRESHOLD
    ):
        raise PredictionLockError("prediction lock holdout policy is not frozen")
    _sha256(holdout["policy_payload_sha256"], label="holdout policy payload SHA-256")

    conditions = _exact(
        lock["conditions"], _CONDITION_KEYS, label="prediction lock.conditions"
    )
    if (
        _text(conditions["condition_gate_policy_id"], label="condition gate policy id")
        != "frozen_condition_minimum_cohort_gate_v1"
    ):
        raise PredictionLockError("prediction lock condition gate policy is not frozen")
    for key in ("condition_gate_policy_sha256", "condition_projection_sha256"):
        _sha256(conditions[key], label=f"prediction lock.conditions.{key}")
    if _count(conditions["estimable_condition_count"], label="estimable count") != EXPECTED_CELL_COUNT:
        raise PredictionLockError("prediction lock estimable condition count is not six")
    if (
        _count(conditions["gated_condition_count"], label="gated count")
        != EXPECTED_GATED_CONDITION_COUNT
    ):
        raise PredictionLockError("prediction lock gated condition count is not four")
    for key, expected in (
        ("estimable_condition_counts_by_source", {"bagen_swebench": 5, "spend_openhands": 1}),
        ("gated_condition_counts_by_source", {"bagen_swebench": 4, "spend_openhands": 0}),
    ):
        counts = _exact(conditions[key], EXPECTED_SOURCE_NAMES, label=key)
        if dict(counts) != expected:
            raise PredictionLockError(f"prediction lock {key} is not frozen")

    metrics = _exact(lock["metrics"], _METRIC_KEYS, label="prediction lock.metrics")
    _sha256(metrics["aggregate_metrics_sha256"], label="aggregate metrics SHA-256")
    if _count(metrics["bundle_count"], label="bundle count") != EXPECTED_BUNDLE_COUNT:
        raise PredictionLockError("prediction lock bundle count is not ninety")
    if _count(metrics["cell_count"], label="cell count") != EXPECTED_CELL_COUNT:
        raise PredictionLockError("prediction lock cell count is not six")
    if _count(metrics["prediction_count"], label="prediction count") == 0:
        raise PredictionLockError("prediction lock prediction count must be positive")

    sources = _exact(lock["sources"], EXPECTED_SOURCE_NAMES, label="prediction lock.sources")
    for name, source_value in sources.items():
        source = _exact(source_value, _SOURCE_KEYS, label=f"prediction lock.sources.{name}")
        for key in ("descriptor_path", "manifest_path", "raw_artifact_path"):
            _relative(source[key], label=f"prediction lock.sources.{name}.{key}")
        for key in (
            "capability_contract_hash",
            "dataset_id",
            "descriptor_file_sha256",
            "manifest_sha256",
            "raw_artifact_sha256",
            "source_descriptor_hash",
        ):
            _sha256(source[key], label=f"prediction lock.sources.{name}.{key}")
        for key in ("revision", "raw_artifact_sha256_kind", "source_id"):
            _text(source[key], label=f"prediction lock.sources.{name}.{key}")
        _count(source["dataset_row_count"], label=f"{name} dataset rows")
        _count(source["raw_artifact_bytes"], label=f"{name} raw bytes")
    return lock


def _git(
    root: Path,
    arguments: Sequence[str],
    *,
    binary: bool = False,
) -> str | bytes:
    kwargs: dict[str, Any] = {"capture_output": True, "check": False, "timeout": 30}
    if not binary:
        kwargs.update({"text": True, "encoding": "utf-8", "errors": "strict"})
    try:
        result = subprocess.run(
            [runner._git_executable(), "-C", str(root), *arguments],
            **kwargs,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise PredictionLockError("cannot inspect prediction baseline Git objects") from exc
    if result.returncode != 0:
        raise PredictionLockError("Git could not inspect prediction baseline objects")
    output = result.stdout
    if not isinstance(output, bytes if binary else str):
        raise PredictionLockError("Git returned invalid prediction baseline output")
    return output


def frozen_git_tree(root: Path, *, commit: str, paths: Sequence[str]) -> GitTreeEvidence:
    resolved_root = root.resolve(strict=True)
    frozen_commit = _commit(commit, label="prediction runner commit")
    top = str(_git(resolved_root, ("rev-parse", "--show-toplevel"))).strip()
    if Path(top).resolve(strict=True) != resolved_root:
        raise PredictionLockError("repository root is not the Git worktree top-level")
    resolved_commit = str(
        _git(resolved_root, ("rev-parse", "--verify", f"{frozen_commit}^{{commit}}"))
    ).strip()
    if resolved_commit != frozen_commit:
        raise PredictionLockError("prediction runner commit object does not resolve exactly")
    canonical_paths = tuple(sorted(_relative(path, label="frozen Git path") for path in paths))
    if len(set(canonical_paths)) != len(canonical_paths) or not canonical_paths:
        raise PredictionLockError("frozen Git paths are empty or duplicated")
    payloads: list[tuple[str, bytes]] = []
    for path in canonical_paths:
        raw = str(_git(resolved_root, ("ls-tree", "-z", frozen_commit, "--", path)))
        records = [record for record in raw.split("\0") if record]
        if len(records) != 1:
            raise PredictionLockError("frozen Git path does not resolve to exactly one object")
        try:
            header, returned_path = records[0].split("\t", 1)
            mode, object_type, object_id = header.split(" ", 2)
        except ValueError as exc:
            raise PredictionLockError("frozen Git tree entry is malformed") from exc
        if (
            returned_path != path
            or mode not in {"100644", "100755"}
            or object_type != "blob"
        ):
            raise PredictionLockError("frozen Git path is not one regular blob")
        blob = _git(resolved_root, ("cat-file", "blob", object_id), binary=True)
        assert isinstance(blob, bytes)
        payloads.append((path, blob))
    return GitTreeEvidence(frozen_commit, canonical_paths, tuple(payloads))


def _frozen_runner_paths(root: Path, *, commit: str) -> tuple[str, ...]:
    resolved_root = root.resolve(strict=True)
    frozen_commit = _commit(commit, label="prediction runner commit")
    raw = str(
        _git(
            resolved_root,
            (
                "ls-tree",
                "-r",
                "-z",
                frozen_commit,
                "--",
                "src/token_prediction",
                runner.RUNNER_RELATIVE,
            ),
        )
    )
    if raw and not raw.endswith("\0"):
        raise PredictionLockError("frozen runner Git tree output is truncated")
    enumerated: list[str] = []
    for record in raw[:-1].split("\0") if raw else ():
        try:
            header, path_value = record.split("\t", 1)
            mode, object_type, _ = header.split(" ", 2)
        except ValueError as exc:
            raise PredictionLockError("frozen runner Git tree entry is malformed") from exc
        path = _relative(path_value, label="frozen runner code path")
        if path != runner.RUNNER_RELATIVE and not (
            path.startswith("src/token_prediction/") and path.endswith(".py")
        ):
            continue
        if mode not in {"100644", "100755"} or object_type != "blob":
            raise PredictionLockError("frozen runner code path is not a regular blob")
        enumerated.append(path)
    exact_paths = tuple(sorted(enumerated))
    if (
        len(set(exact_paths)) != len(exact_paths)
        or runner.RUNNER_RELATIVE not in exact_paths
        or not any(path.startswith("src/token_prediction/") for path in exact_paths)
    ):
        raise PredictionLockError("frozen runner/source tree is incomplete")
    return exact_paths


def frozen_runner_code_tree(
    root: Path,
    *,
    commit: str,
    claimed_paths: Sequence[str],
) -> GitTreeEvidence:
    """Require the artifact code path list to equal the full frozen runner tree."""

    resolved_root = root.resolve(strict=True)
    frozen_commit = _commit(commit, label="prediction runner commit")
    exact_paths = _frozen_runner_paths(resolved_root, commit=frozen_commit)
    claimed = tuple(sorted(_relative(path, label="claimed code path") for path in claimed_paths))
    if claimed != exact_paths:
        raise PredictionLockError(
            "artifact code paths do not equal the full frozen runner/source tree"
        )
    return frozen_git_tree(resolved_root, commit=frozen_commit, paths=exact_paths)


def verify_tracked_prediction_lock(root: Path, relative: str, path: Path) -> None:
    """Require the trust-root lock itself to be a clean file tracked by HEAD."""

    resolved_root = root.resolve(strict=True)
    canonical = _relative(relative, label="tracked prediction lock path")
    top = str(_git(resolved_root, ("rev-parse", "--show-toplevel"))).strip()
    if Path(top).resolve(strict=True) != resolved_root:
        raise PredictionLockError("prediction lock repository root is not Git top-level")
    listed = str(
        _git(resolved_root, ("ls-files", "--error-unmatch", "--", canonical))
    ).strip()
    if listed != canonical:
        raise PredictionLockError("prediction lock is not tracked exactly by Git")
    status = str(
        _git(
            resolved_root,
            (
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--",
                canonical,
            ),
        )
    )
    if status:
        raise PredictionLockError("tracked prediction lock is dirty")
    committed = _git(resolved_root, ("show", f"HEAD:{canonical}"), binary=True)
    assert isinstance(committed, bytes)
    if committed != path.read_bytes():
        raise PredictionLockError("prediction lock bytes differ from the HEAD blob")


def _control_tree_sha256(items: Iterable[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256(b"data-foundation-baseline-controls-v1\0")
    for relative, payload in items:
        for framed in (relative.encode("utf-8"), payload):
            digest.update(len(framed).to_bytes(8, "big"))
            digest.update(framed)
    return digest.hexdigest()


def _workspace_code_tree(root: Path) -> tuple[tuple[str, ...], str]:
    resolved_root = root.resolve(strict=True)
    source_root = _repo_path(resolved_root, "src", label="workspace source root")
    package_root = _repo_path(
        resolved_root, "src/token_prediction", label="workspace package root"
    )
    for label, directory in (("source root", source_root), ("package root", package_root)):
        if not directory.is_dir() or runner._is_link_or_reparse(directory):
            raise PredictionLockError(f"workspace {label} is linked, reparse-backed, or missing")
    paths: list[str] = []
    for directory_name, directory_names, file_names in os.walk(
        package_root, topdown=True, followlinks=False
    ):
        directory = Path(directory_name)
        if runner._is_link_or_reparse(directory):
            raise PredictionLockError("workspace source tree contains a linked directory")
        for child_name in directory_names:
            if runner._is_link_or_reparse(directory / child_name):
                raise PredictionLockError(
                    "workspace source tree contains a junction or reparse point"
                )
        for file_name in file_names:
            if file_name.endswith(".py"):
                path = directory / file_name
                status = path.lstat()
                if runner._is_link_or_reparse(path) or not stat.S_ISREG(status.st_mode):
                    raise PredictionLockError("workspace source tree contains a linked file")
                paths.append(path.relative_to(resolved_root).as_posix())
    paths.append(runner.RUNNER_RELATIVE)
    canonical = tuple(sorted(_relative(path, label="workspace code path") for path in paths))
    items = [
        (
            relative,
            _regular_file(resolved_root, relative, label=f"workspace code {relative}").read_bytes(),
        )
        for relative in canonical
    ]
    return canonical, runner._framed_code_hash(items)


def _verify_bound_inputs(root: Path, results: Mapping[str, Any]) -> None:
    binding = results["source_binding"]
    data_lock = _regular_file(
        root, binding["baseline_lock_path"], label="bound Data Foundation lock"
    )
    _load_json(data_lock, label="bound Data Foundation lock")
    if _file_sha256(data_lock) != binding["baseline_lock_file_sha256"]:
        raise PredictionLockError("bound Data Foundation lock SHA-256 differs")
    audit = _regular_file(root, binding["data_foundation_audit_path"], label="bound audit")
    if _file_sha256(audit) != binding["data_foundation_audit_file_sha256"]:
        raise PredictionLockError("bound Data Foundation audit file SHA-256 differs")
    audit_value = _load_json(audit, label="bound Data Foundation audit")
    try:
        verify_audit_payload(audit_value)
    except Exception as exc:
        raise PredictionLockError(f"bound audit payload is invalid: {exc}") from exc
    if audit_value.get("audit_payload_sha256") != binding[
        "data_foundation_audit_payload_sha256"
    ]:
        raise PredictionLockError("bound Data Foundation audit payload SHA-256 differs")

    for name, source in results["sources"].items():
        descriptor = _regular_file(
            root, source["descriptor_path"], label=f"{name} descriptor"
        )
        if _file_sha256(descriptor) != source["descriptor_file_sha256"]:
            raise PredictionLockError(f"{name} descriptor SHA-256 differs")
        manifest = _regular_file(root, source["manifest_path"], label=f"{name} manifest")
        if _file_sha256(manifest) != source["manifest_sha256"]:
            raise PredictionLockError(f"{name} manifest SHA-256 differs")
        raw = _repo_path(root, source["raw_artifact_path"], label=f"{name} raw artifact")
        if not raw.exists() or runner._is_link_or_reparse(raw):
            raise PredictionLockError(f"{name} raw artifact path is missing or reparse-backed")


def verify_tracked_prediction_lock_only(
    repo_root: Path,
    *,
    lock_path: Path = DEFAULT_LOCK,
    require_workspace_source_match: bool = False,
) -> dict[str, Any]:
    """Verify the production lock and immutable Git bindings without raw artifacts."""

    root = repo_root.resolve(strict=True)
    lock_relative = _relative(lock_path.as_posix(), label="prediction lock argument")
    lock_file = _regular_file(root, lock_relative, label="prediction lock")
    verify_tracked_prediction_lock(root, lock_relative, lock_file)
    lock = _validate_lock(_load_json(lock_file, label="prediction lock"))
    if lock["artifact"]["relative_path"] != runner.DEFAULT_OUTPUT:
        raise PredictionLockError("tracked prediction artifact path is not the frozen default")

    runner_lock = lock["runner"]
    frozen_paths = _frozen_runner_paths(root, commit=runner_lock["git_commit"])
    frozen_code = frozen_runner_code_tree(
        root,
        commit=runner_lock["git_commit"],
        claimed_paths=frozen_paths,
    )
    if (
        len(frozen_paths) != runner_lock["code_blob_count"]
        or _semantic_sha256(list(frozen_paths)) != runner_lock["code_paths_sha256"]
        or runner._framed_code_hash(frozen_code.payloads)
        != runner_lock["code_tree_sha256"]
    ):
        raise PredictionLockError("tracked lock runner/source Git binding does not close")

    control_paths = tuple(
        sorted(
            [lock["data_foundation"]["baseline_lock_path"]]
            + [source["descriptor_path"] for source in lock["sources"].values()]
        )
    )
    if (
        len(set(control_paths)) != len(control_paths)
        or len(control_paths) != runner_lock["tracked_control_blob_count"]
        or _semantic_sha256(list(control_paths))
        != runner_lock["tracked_control_paths_sha256"]
    ):
        raise PredictionLockError("tracked lock control path binding does not close")
    head_commit = _commit(
        str(_git(root, ("rev-parse", "--verify", "HEAD^{commit}"))).strip(),
        label="HEAD",
    )
    try:
        control_tree_hash = runner.tracked_control_tree_sha256(
            root,
            control_paths,
            git_commit=head_commit,
        )
    except runner.DataFoundationBaselineError as exc:
        raise PredictionLockError(str(exc)) from exc
    if control_tree_hash != runner_lock["tracked_control_tree_sha256"]:
        raise PredictionLockError("tracked lock control tree binding does not close")

    data_lock = _regular_file(
        root,
        lock["data_foundation"]["baseline_lock_path"],
        label="tracked Data Foundation lock",
    )
    data_lock_value = _load_json(data_lock, label="tracked Data Foundation lock")
    try:
        _validate_baseline(data_lock_value)
    except Exception as exc:
        raise PredictionLockError(f"tracked Data Foundation lock is invalid: {exc}") from exc
    if _file_sha256(data_lock) != lock["data_foundation"]["baseline_lock_file_sha256"]:
        raise PredictionLockError("tracked Data Foundation lock SHA-256 differs")
    implementation = _exact(
        data_lock_value.get("implementation"),
        {
            "git_commit",
            "git_source_binding_policy",
            "source_blob_count",
            "source_tree_sha256",
        },
        label="tracked Data Foundation implementation",
    )
    production_audit = _exact(
        data_lock_value.get("production_audit"),
        {
            "audit_payload_sha256",
            "bytes",
            "deterministic_run_count",
            "file_sha256",
            "relative_path",
            "rerun_byte_identical",
            "rerun_bytes",
            "rerun_file_sha256",
            "rerun_relative_path",
        },
        label="tracked Data Foundation production audit",
    )
    data_binding = lock["data_foundation"]
    if (
        implementation["git_commit"] != data_binding["audit_git_commit"]
        or implementation["source_tree_sha256"]
        != data_binding["audit_source_tree_sha256"]
        or production_audit["relative_path"] != data_binding["audit_path"]
        or production_audit["file_sha256"] != data_binding["audit_file_sha256"]
        or production_audit["audit_payload_sha256"]
        != data_binding["audit_payload_sha256"]
    ):
        raise PredictionLockError("tracked Data Foundation identity differs")
    audit_commit = _commit(implementation["git_commit"], label="Data Foundation commit")
    resolved_audit_commit = str(
        _git(root, ("rev-parse", "--verify", f"{audit_commit}^{{commit}}"))
    ).strip()
    if resolved_audit_commit != audit_commit:
        raise PredictionLockError("Data Foundation commit object does not resolve exactly")

    data_sources = _exact(
        data_lock_value.get("sources"),
        EXPECTED_SOURCE_NAMES,
        label="tracked Data Foundation sources",
    )
    for name, source in lock["sources"].items():
        descriptor = _regular_file(
            root,
            source["descriptor_path"],
            label=f"tracked {name} descriptor",
        )
        _load_json(descriptor, label=f"tracked {name} descriptor")
        if _file_sha256(descriptor) != source["descriptor_file_sha256"]:
            raise PredictionLockError(f"tracked {name} descriptor SHA-256 differs")
        data_source = data_sources[name]
        comparisons = {
            "capability_contract_hash": "capability_contract_hash",
            "dataset_id": "dataset_id",
            "dataset_row_count": "row_count",
            "descriptor_file_sha256": "descriptor_file_sha256",
            "manifest_sha256": "manifest_sha256",
            "revision": "revision",
            "source_descriptor_hash": "source_descriptor_hash",
            "source_id": "source_id",
        }
        if any(source[left] != data_source[right] for left, right in comparisons.items()):
            raise PredictionLockError(f"tracked {name} Data Foundation binding differs")

    workspace_paths, workspace_hash = _workspace_code_tree(root)
    workspace_matches = (
        workspace_paths == frozen_paths
        and workspace_hash == runner_lock["code_tree_sha256"]
    )
    if require_workspace_source_match and not workspace_matches:
        raise PredictionLockError("workspace runner/source tree differs from frozen lock")
    summary = {
        "artifact_id": lock["artifact"]["artifact_id"],
        "baseline_id": lock["artifact"]["baseline_id"],
        "bundle_count": lock["metrics"]["bundle_count"],
        "estimable_condition_count": lock["conditions"]["estimable_condition_count"],
        "gated_condition_count": lock["conditions"]["gated_condition_count"],
        "prediction_count": lock["metrics"]["prediction_count"],
        "verification_scope": "tracked_lock_and_git_objects",
        "workspace_source_matches_frozen": workspace_matches,
    }
    try:
        _assert_privacy_safe(summary, label="tracked prediction lock summary")
    except Exception as exc:
        raise PredictionLockError(str(exc)) from exc
    return summary


def verify_prediction_lock(
    repo_root: Path,
    *,
    lock_path: Path = DEFAULT_LOCK,
    artifact_path: Path = DEFAULT_ARTIFACT,
    require_workspace_source_match: bool = False,
) -> dict[str, Any]:
    root = repo_root.resolve(strict=True)
    lock_relative = _relative(lock_path.as_posix(), label="prediction lock argument")
    lock_file = _regular_file(root, lock_relative, label="prediction lock")
    verify_tracked_prediction_lock(root, lock_relative, lock_file)
    lock = _validate_lock(_load_json(lock_file, label="prediction lock"))
    artifact_relative = _relative(
        artifact_path.as_posix(), label="prediction artifact argument"
    )
    if artifact_relative != lock["artifact"]["relative_path"]:
        raise PredictionLockError("prediction artifact argument differs from tracked lock")
    artifact = _regular_directory(root, artifact_relative, label="prediction artifact")
    _assert_regular_tree(artifact, label="prediction artifact")
    try:
        manifest = runner.verify_artifact(artifact)
    except runner.DataFoundationBaselineError as exc:
        raise PredictionLockError(f"prediction artifact verification failed: {exc}") from exc
    results = _load_json(artifact / "results.json", label="prediction results")
    projection = build_lock_projection(
        root,
        artifact_relative=artifact_relative,
        manifest=manifest,
        results=results,
    )
    if dict(lock) != projection:
        raise PredictionLockError("prediction artifact projection differs from tracked lock")

    _verify_bound_inputs(root, results)
    binding = results["source_binding"]
    code_paths = tuple(binding["code_paths"])
    code_git = frozen_runner_code_tree(
        root,
        commit=binding["git_commit"],
        claimed_paths=code_paths,
    )
    if (
        len(code_git.paths) != lock["runner"]["code_blob_count"]
        or runner._framed_code_hash(code_git.payloads)
        != binding["runner_and_src_code_tree_sha256"]
    ):
        raise PredictionLockError("frozen runner/source Git blobs do not close")
    control_paths = _control_paths(results)
    control_git = frozen_git_tree(root, commit=binding["git_commit"], paths=control_paths)
    if (
        len(control_git.paths) != lock["runner"]["tracked_control_blob_count"]
        or _control_tree_sha256(control_git.payloads) != binding["tracked_control_tree_sha256"]
    ):
        raise PredictionLockError("frozen tracked-control Git blobs do not close")
    actual_control_items = tuple(
        (
            relative,
            _regular_file(root, relative, label=f"tracked control {relative}").read_bytes(),
        )
        for relative in control_paths
    )
    if _control_tree_sha256(actual_control_items) != binding["tracked_control_tree_sha256"]:
        raise PredictionLockError("current tracked-control bytes differ from frozen commit")

    workspace_paths, workspace_hash = _workspace_code_tree(root)
    workspace_matches = (
        workspace_paths == code_git.paths
        and workspace_hash == binding["runner_and_src_code_tree_sha256"]
    )
    if require_workspace_source_match and not workspace_matches:
        raise PredictionLockError("workspace runner/source tree differs from frozen artifact")

    summary = {
        "artifact_id": manifest["artifact_id"],
        "baseline_id": manifest["baseline_id"],
        "bundle_count": results["bundle_count"],
        "cell_count": len(results["cells"]),
        "estimable_condition_count": len(results["cells"]),
        "gated_condition_count": len(results["not_estimable_conditions"]),
        "prediction_count": _prediction_count(results),
        "results_payload_sha256": results["results_payload_sha256"],
        "workspace_source_matches_frozen": workspace_matches,
    }
    try:
        _assert_privacy_safe(summary, label="prediction verification summary")
    except Exception as exc:
        raise PredictionLockError(str(exc)) from exc
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the tracked Data Foundation prediction-baseline lock."
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument(
        "--tracked-only",
        action="store_true",
        help="verify the tracked lock and immutable Git bindings without workspace artifacts",
    )
    parser.add_argument("--require-workspace-source-match", action="store_true")
    args = parser.parse_args()
    try:
        if args.tracked_only:
            summary = verify_tracked_prediction_lock_only(
                REPO_ROOT,
                lock_path=args.lock,
                require_workspace_source_match=args.require_workspace_source_match,
            )
        else:
            summary = verify_prediction_lock(
                REPO_ROOT,
                lock_path=args.lock,
                artifact_path=args.artifact,
                require_workspace_source_match=args.require_workspace_source_match,
            )
    except (PredictionLockError, OSError, ValueError) as exc:
        parser.exit(2, f"Data Foundation prediction lock verification failed: {exc}\n")
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
