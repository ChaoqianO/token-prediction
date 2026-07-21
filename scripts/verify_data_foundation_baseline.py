from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from scripts.audit_data_foundation_v2 import (
        AUDIT_SCHEMA_VERSION,
        BUILD_COMMAND,
        CAPABILITY_DATASET_SCHEMA_VERSION,
        DEFAULT_OUTPUT,
        REPO_ROOT,
        DataFoundationAuditError,
        _default_git_executable,
        _is_relevant_source_path,
        _assert_aggregate_safe,
        _require_git_commit,
        _require_list,
        _require_mapping,
        _require_non_negative_int,
        _require_sha256,
        _require_text,
        _source_tree_hash_from_file_hashes,
        load_bagen_source_summary,
        load_source_descriptor,
        load_spend_source_summary,
        load_strict_json,
        verify_audit_payload,
        verify_file,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from audit_data_foundation_v2 import (  # type: ignore[no-redef]
        AUDIT_SCHEMA_VERSION,
        BUILD_COMMAND,
        CAPABILITY_DATASET_SCHEMA_VERSION,
        DEFAULT_OUTPUT,
        REPO_ROOT,
        DataFoundationAuditError,
        _default_git_executable,
        _is_relevant_source_path,
        _assert_aggregate_safe,
        _require_git_commit,
        _require_list,
        _require_mapping,
        _require_non_negative_int,
        _require_sha256,
        _require_text,
        _source_tree_hash_from_file_hashes,
        load_bagen_source_summary,
        load_source_descriptor,
        load_spend_source_summary,
        load_strict_json,
        verify_audit_payload,
        verify_file,
    )
from token_prediction.contracts import canonical_input_path, resolve_canonical_input_file
from token_prediction.dataset import (
    LabelStatus,
    PredictionPosition,
    PredictionTarget,
    decide_target_capability,
)


BASELINE_SCHEMA_VERSION = 1
BASELINE_TYPE = "data_foundation_v2"
DEFAULT_BASELINE = Path("configs/data_foundation_v2_baseline.json")
GIT_SOURCE_BINDING_POLICY = "tracked_clean_head_blob_tree_v1"
FROZEN_SOURCE_BLOB_COUNT = 42

_BASELINE_KEYS = {
    "baseline_schema_version",
    "baseline_type",
    "build_command",
    "implementation",
    "production_audit",
    "sources",
}
_BASELINE_IMPLEMENTATION_KEYS = {
    "git_commit",
    "git_source_binding_policy",
    "source_blob_count",
    "source_tree_sha256",
}
_PRODUCTION_AUDIT_KEYS = {
    "audit_payload_sha256",
    "bytes",
    "deterministic_run_count",
    "file_sha256",
    "relative_path",
    "rerun_byte_identical",
    "rerun_bytes",
    "rerun_file_sha256",
    "rerun_relative_path",
}
_BASELINE_SOURCE_KEYS = {
    "capability_contract_hash",
    "condition_count",
    "dataset_id",
    "dataset_status_counts",
    "descriptor_file_sha256",
    "manifest_sha256",
    "revision",
    "row_count",
    "run_count",
    "source_descriptor_hash",
    "source_id",
    "task_count",
    "trajectory_count",
}
_AUDIT_KEYS = {
    "audit_payload_sha256",
    "build_command",
    "data_foundation_v2_audit_schema_version",
    "dataset_schema_version",
    "implementation",
    "source_count",
    "sources",
}
_AUDIT_IMPLEMENTATION_KEYS = {
    "git_commit",
    "git_source_binding_policy",
    "runtime",
    "source_tree_sha256",
}
_RUNTIME_KEYS = {"python_implementation", "python_version"}
_AUDIT_SOURCE_KEYS = {
    "artifacts",
    "build_command",
    "capability_contract_hash",
    "capability_decision_matrix",
    "dataset",
    "identity_counts",
    "source_descriptor",
    "source_descriptor_hash",
    "source_name",
}
_ARTIFACT_KEYS = {"bytes", "file_count", "path", "sha256", "sha256_kind"}
_DATASET_KEYS = {
    "by_position",
    "by_position_target",
    "by_target",
    "capability_contract_hash",
    "dataset_id",
    "row_count",
    "schema_version",
    "source_descriptor_hash",
    "status_counts",
}
_IDENTITY_KEYS = {"condition_count", "run_count", "task_count", "trajectory_count"}
_STATUS_KEYS = {status.value for status in LabelStatus}
_DECISION_KEYS = {
    "available",
    "capability_contract_hash",
    "missing_observables",
    "position",
    "reason",
    "required_observables",
    "source_id",
    "target",
}
_FORBIDDEN_IDENTITY_KEYS = {
    "attempt",
    "attempt_id",
    "call_id",
    "condition_id",
    "event",
    "event_id",
    "event_seq",
    "logical_call",
    "logical_call_id",
    "message_id",
    "point",
    "point_id",
    "prediction_point_id",
    "raw_message",
    "raw_messages",
    "raw_ref",
    "raw_reference",
    "request_id",
    "response_id",
    "run_id",
    "task_id",
    "trace_id",
    "trajectory_id",
}
FULL_SOURCE_LOADERS: Mapping[str, Callable[[Path], dict[str, Any]]] = {
    "bagen_swebench": load_bagen_source_summary,
    "spend_openhands": load_spend_source_summary,
}


@dataclass(frozen=True)
class GitSourceEvidence:
    git_commit: str
    paths: tuple[str, ...]
    source_tree_sha256: str

    @property
    def blob_count(self) -> int:
        return len(self.paths)


def _require_exact_keys(
    value: Any,
    expected: set[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    mapping = _require_mapping(value, label)
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise DataFoundationAuditError(
            f"{label} keys do not match the frozen schema "
            f"(missing={missing!r}, extra={extra!r})"
        )
    return mapping


def _assert_privacy_safe(value: Any, *, label: str) -> None:
    def walk(item: Any, *, item_label: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                normalized = str(key).casefold().replace("-", "_")
                if normalized in _FORBIDDEN_IDENTITY_KEYS:
                    raise DataFoundationAuditError(
                        f"{item_label} contains forbidden row-level identity key {key!r}"
                    )
                walk(child, item_label=f"{item_label}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                walk(child, item_label=f"{item_label}[{index}]")

    walk(value, item_label=label)
    _assert_aggregate_safe(value, label=label)


def _require_exact_int(value: Any, expected: int, *, label: str) -> int:
    resolved = _require_non_negative_int(value, label)
    if resolved != expected:
        raise DataFoundationAuditError(f"{label} must equal {expected}")
    return resolved


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise DataFoundationAuditError(f"{label} must be a boolean")
    return value


def _require_status_counts(value: Any, *, label: str) -> dict[str, int]:
    counts = _require_exact_keys(value, _STATUS_KEYS, label=label)
    return {
        status: _require_non_negative_int(counts[status], f"{label}.{status}")
        for status in sorted(_STATUS_KEYS)
    }


def _canonical_path(value: object, *, label: str) -> str:
    try:
        return canonical_input_path(value, context=label)
    except ValueError as exc:
        raise DataFoundationAuditError(str(exc)) from exc


def _require_relative_argument(path: Path, *, label: str) -> str:
    return _canonical_path(path.as_posix(), label=label)


def _resolve_strict_repo_file(
    repo_root: Path,
    relative_path: str,
    *,
    label: str,
) -> Path:
    canonical = _canonical_path(relative_path, label=label)
    try:
        return resolve_canonical_input_file(repo_root, canonical, context=label)
    except (OSError, ValueError) as exc:
        raise DataFoundationAuditError(str(exc)) from exc


def verify_frozen_git_source_tree(
    repo_root: Path,
    *,
    git_commit: str,
    expected_source_tree_sha256: str,
    expected_blob_count: int,
    git_executable: str | Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> GitSourceEvidence:
    """Rebuild the frozen source hash from blobs in the named commit object."""

    root = repo_root.resolve(strict=True)
    commit = _require_git_commit(git_commit, "frozen Git commit")
    expected_hash = _require_sha256(
        expected_source_tree_sha256, "frozen Git source tree SHA-256"
    )
    blob_count = _require_non_negative_int(expected_blob_count, "source blob count")
    if blob_count == 0:
        raise DataFoundationAuditError("source blob count must be positive")
    executable = str(git_executable or _default_git_executable())

    def run_git(
        arguments: Sequence[str],
        *,
        label: str,
        text_output: bool,
    ) -> str | bytes:
        kwargs: dict[str, Any] = {
            "capture_output": True,
            "check": False,
            "timeout": 30,
        }
        if text_output:
            kwargs.update(
                {
                    "text": True,
                    "encoding": "utf-8",
                    "errors": "strict",
                }
            )
        try:
            result = runner(
                [executable, "-C", str(root), *arguments],
                **kwargs,
            )
        except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
            raise DataFoundationAuditError(f"cannot inspect frozen Git {label}") from exc
        if result.returncode != 0:
            raise DataFoundationAuditError(f"Git could not inspect frozen {label}")
        output = result.stdout
        expected_type = str if text_output else bytes
        if not isinstance(output, expected_type):
            raise DataFoundationAuditError(f"Git returned invalid frozen {label} output")
        return output

    top_level = str(
        run_git(("rev-parse", "--show-toplevel"), label="repository root", text_output=True)
    ).strip()
    try:
        if Path(top_level).resolve(strict=True) != root:
            raise DataFoundationAuditError(
                "configured root is not the frozen Git worktree top-level"
            )
    except OSError as exc:
        raise DataFoundationAuditError("frozen Git worktree root is invalid") from exc

    resolved_commit = str(
        run_git(
            ("rev-parse", "--verify", f"{commit}^{{commit}}"),
            label="commit object",
            text_output=True,
        )
    ).strip()
    if resolved_commit != commit:
        raise DataFoundationAuditError(
            "frozen Git commit did not resolve to the exact pinned commit object"
        )

    tree_output = str(
        run_git(
            (
                "ls-tree",
                "-r",
                "-z",
                commit,
                "--",
                "src/token_prediction",
                "scripts/audit_data_foundation_v2.py",
            ),
            label="source tree",
            text_output=True,
        )
    )
    if tree_output and not tree_output.endswith("\0"):
        raise DataFoundationAuditError("frozen Git source tree output is truncated")
    object_ids: dict[str, str] = {}
    for record in tree_output[:-1].split("\0") if tree_output else ():
        try:
            header, raw_path = record.split("\t", 1)
            mode, object_type, object_id = header.split(" ", 2)
        except ValueError as exc:
            raise DataFoundationAuditError(
                "frozen Git source tree contains a malformed entry"
            ) from exc
        path = _canonical_path(raw_path, label="frozen Git source path")
        if not _is_relevant_source_path(path):
            continue
        if mode not in {"100644", "100755"} or object_type != "blob":
            raise DataFoundationAuditError(
                "frozen Git relevant source entry is not a regular blob"
            )
        _require_git_commit(object_id, "frozen Git blob object id")
        if path in object_ids:
            raise DataFoundationAuditError("frozen Git source tree repeats a path")
        object_ids[path] = object_id
    paths = tuple(sorted(object_ids))
    if len(paths) != blob_count:
        raise DataFoundationAuditError(
            "frozen Git relevant source blob count does not match baseline"
        )

    file_hashes: dict[str, str] = {}
    for path in paths:
        blob = run_git(
            ("cat-file", "blob", object_ids[path]),
            label=f"blob {path}",
            text_output=False,
        )
        assert isinstance(blob, bytes)
        file_hashes[path] = hashlib.sha256(blob).hexdigest()
    rebuilt_hash = _source_tree_hash_from_file_hashes(file_hashes)
    if rebuilt_hash != expected_hash:
        raise DataFoundationAuditError(
            "frozen Git blob source tree hash does not match baseline"
        )
    return GitSourceEvidence(commit, paths, rebuilt_hash)


def _is_link_or_reparse_point(path: Path, status: os.stat_result) -> bool:
    if stat.S_ISLNK(status.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if reparse_flag and getattr(status, "st_file_attributes", 0) & reparse_flag:
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def strict_workspace_source_tree_sha256(
    repo_root: Path,
) -> str:
    """Hash the workspace source tree after rejecting every linked component."""

    root = repo_root.resolve(strict=True)
    source_root = root / "src"
    package_root = root / "src" / "token_prediction"
    for label, directory in (("source root", source_root), ("source package", package_root)):
        try:
            directory_status = directory.lstat()
        except OSError as exc:
            raise DataFoundationAuditError(f"workspace {label} is missing") from exc
        if _is_link_or_reparse_point(directory, directory_status) or not stat.S_ISDIR(
            directory_status.st_mode
        ):
            raise DataFoundationAuditError(
                f"workspace {label} must not be a symlink, junction, or reparse point"
            )

    actual_paths: list[str] = []
    for directory_name, directory_names, file_names in os.walk(
        package_root, topdown=True, followlinks=False
    ):
        directory = Path(directory_name)
        status = directory.lstat()
        if _is_link_or_reparse_point(directory, status):
            raise DataFoundationAuditError(
                "workspace source tree contains a symlink, junction, or reparse point"
            )
        for child_name in sorted(directory_names):
            child = directory / child_name
            child_status = child.lstat()
            if _is_link_or_reparse_point(child, child_status):
                raise DataFoundationAuditError(
                    "workspace source tree contains a symlink, junction, or reparse point"
                )
        for file_name in sorted(file_names):
            if not file_name.endswith(".py"):
                continue
            path = directory / file_name
            file_status = path.lstat()
            if _is_link_or_reparse_point(path, file_status) or not stat.S_ISREG(
                file_status.st_mode
            ):
                raise DataFoundationAuditError(
                    "workspace source tree contains a linked or non-regular Python file"
                )
            actual_paths.append(path.relative_to(root).as_posix())
    actual_paths.append("scripts/audit_data_foundation_v2.py")
    canonical_actual = tuple(
        sorted(_canonical_path(path, label="workspace source path") for path in actual_paths)
    )
    if len(set(canonical_actual)) != len(canonical_actual) or not canonical_actual:
        raise DataFoundationAuditError("workspace source paths are empty or duplicated")

    file_hashes: dict[str, str] = {}
    for relative_path in canonical_actual:
        path = _resolve_strict_repo_file(
            root,
            relative_path,
            label=f"workspace source {relative_path}",
        )
        file_hashes[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return _source_tree_hash_from_file_hashes(file_hashes)


def _validate_artifact_entry(value: Any, *, label: str) -> Mapping[str, Any]:
    artifact = _require_exact_keys(value, _ARTIFACT_KEYS, label=label)
    _canonical_path(artifact["path"], label=f"{label}.path")
    _require_non_negative_int(artifact["bytes"], f"{label}.bytes")
    file_count = _require_non_negative_int(artifact["file_count"], f"{label}.file_count")
    if file_count == 0:
        raise DataFoundationAuditError(f"{label}.file_count must be positive")
    _require_sha256(artifact["sha256"], f"{label}.sha256")
    sha_kind = _require_text(artifact["sha256_kind"], f"{label}.sha256_kind")
    if sha_kind not in {"file_bytes", "framed_file_index_v1"}:
        raise DataFoundationAuditError(f"{label}.sha256_kind is unsupported")
    if sha_kind == "file_bytes" and file_count != 1:
        raise DataFoundationAuditError(f"{label} file_bytes evidence must name one file")
    return artifact


def _validate_status_row(value: Any, keys: set[str], *, label: str) -> Mapping[str, Any]:
    row = _require_exact_keys(value, keys | {"row_count", "status_counts"}, label=label)
    row_count = _require_non_negative_int(row["row_count"], f"{label}.row_count")
    counts = _require_status_counts(row["status_counts"], label=f"{label}.status_counts")
    if sum(counts.values()) != row_count:
        raise DataFoundationAuditError(f"{label} status counts do not sum to row_count")
    for key in keys:
        _require_text(row[key], f"{label}.{key}")
    return row


def _validate_dataset(value: Any, *, label: str) -> Mapping[str, Any]:
    dataset = _require_exact_keys(value, _DATASET_KEYS, label=label)
    _require_sha256(dataset["dataset_id"], f"{label}.dataset_id")
    _require_exact_int(
        dataset["schema_version"],
        CAPABILITY_DATASET_SCHEMA_VERSION,
        label=f"{label}.schema_version",
    )
    _require_sha256(
        dataset["source_descriptor_hash"], f"{label}.source_descriptor_hash"
    )
    _require_sha256(
        dataset["capability_contract_hash"], f"{label}.capability_contract_hash"
    )
    row_count = _require_non_negative_int(dataset["row_count"], f"{label}.row_count")
    status_counts = _require_status_counts(
        dataset["status_counts"], label=f"{label}.status_counts"
    )
    if sum(status_counts.values()) != row_count:
        raise DataFoundationAuditError(f"{label} status counts do not sum to row_count")

    expected_positions = {position.value for position in PredictionPosition}
    expected_targets = {target.value for target in PredictionTarget}
    expected_cells = {
        (position, target)
        for position in expected_positions
        for target in expected_targets
    }

    by_position = _require_list(dataset["by_position"], f"{label}.by_position")
    position_rows: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(by_position):
        row = _validate_status_row(
            item, {"position"}, label=f"{label}.by_position[{index}]"
        )
        position = row["position"]
        if position in position_rows:
            raise DataFoundationAuditError(f"{label}.by_position contains a duplicate")
        position_rows[position] = row
    if set(position_rows) != expected_positions:
        raise DataFoundationAuditError(f"{label}.by_position is incomplete")

    by_target = _require_list(dataset["by_target"], f"{label}.by_target")
    target_rows: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(by_target):
        row = _validate_status_row(item, {"target"}, label=f"{label}.by_target[{index}]")
        target = row["target"]
        if target in target_rows:
            raise DataFoundationAuditError(f"{label}.by_target contains a duplicate")
        target_rows[target] = row
    if set(target_rows) != expected_targets:
        raise DataFoundationAuditError(f"{label}.by_target is incomplete")

    by_cell = _require_list(
        dataset["by_position_target"], f"{label}.by_position_target"
    )
    cell_rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, item in enumerate(by_cell):
        row = _validate_status_row(
            item,
            {"position", "target"},
            label=f"{label}.by_position_target[{index}]",
        )
        cell = (row["position"], row["target"])
        if cell in cell_rows:
            raise DataFoundationAuditError(
                f"{label}.by_position_target contains a duplicate"
            )
        cell_rows[cell] = row
    if set(cell_rows) != expected_cells:
        raise DataFoundationAuditError(f"{label}.by_position_target is incomplete")

    for rows, dimension in (
        (position_rows.values(), "by_position"),
        (target_rows.values(), "by_target"),
        (cell_rows.values(), "by_position_target"),
    ):
        if sum(int(row["row_count"]) for row in rows) != row_count:
            raise DataFoundationAuditError(f"{label}.{dimension} does not sum to row_count")
        for status in _STATUS_KEYS:
            if sum(int(row["status_counts"][status]) for row in rows) != status_counts[status]:
                raise DataFoundationAuditError(
                    f"{label}.{dimension} status totals do not close"
                )
    for position, aggregate in position_rows.items():
        cells = [row for (cell_position, _), row in cell_rows.items() if cell_position == position]
        if sum(int(row["row_count"]) for row in cells) != aggregate["row_count"]:
            raise DataFoundationAuditError(
                f"{label}.by_position does not match position-target cell margins"
            )
        for status in _STATUS_KEYS:
            if (
                sum(int(row["status_counts"][status]) for row in cells)
                != aggregate["status_counts"][status]
            ):
                raise DataFoundationAuditError(
                    f"{label}.by_position status does not match cell margins"
                )
    for target, aggregate in target_rows.items():
        cells = [row for (_, cell_target), row in cell_rows.items() if cell_target == target]
        if sum(int(row["row_count"]) for row in cells) != aggregate["row_count"]:
            raise DataFoundationAuditError(
                f"{label}.by_target does not match position-target cell margins"
            )
        for status in _STATUS_KEYS:
            if (
                sum(int(row["status_counts"][status]) for row in cells)
                != aggregate["status_counts"][status]
            ):
                raise DataFoundationAuditError(
                    f"{label}.by_target status does not match cell margins"
                )
    return dataset


def _validate_capability_decisions(
    value: Any,
    *,
    source_id: str,
    capability_contract_hash: str,
    label: str,
) -> dict[tuple[str, str], bool]:
    decisions = _require_list(value, label)
    expected_cells = {
        (position.value, target.value)
        for position in PredictionPosition
        for target in PredictionTarget
    }
    cells: set[tuple[str, str]] = set()
    for index, item in enumerate(decisions):
        item_label = f"{label}[{index}]"
        decision = _require_exact_keys(item, _DECISION_KEYS, label=item_label)
        position = _require_text(decision["position"], f"{item_label}.position")
        target = _require_text(decision["target"], f"{item_label}.target")
        cell = (position, target)
        if cell in cells:
            raise DataFoundationAuditError(f"{label} contains a duplicate cell")
        cells.add(cell)
        if decision["source_id"] != source_id:
            raise DataFoundationAuditError(f"{item_label}.source_id does not close")
        if decision["capability_contract_hash"] != capability_contract_hash:
            raise DataFoundationAuditError(
                f"{item_label}.capability_contract_hash does not close"
            )
        available = _require_bool(decision["available"], label=f"{item_label}.available")
        required = _require_list(
            decision["required_observables"], f"{item_label}.required_observables"
        )
        missing = _require_list(
            decision["missing_observables"], f"{item_label}.missing_observables"
        )
        for list_name, entries in (("required", required), ("missing", missing)):
            if any(not isinstance(entry, str) or not entry for entry in entries):
                raise DataFoundationAuditError(
                    f"{item_label}.{list_name}_observables contains invalid values"
                )
            if len(set(entries)) != len(entries):
                raise DataFoundationAuditError(
                    f"{item_label}.{list_name}_observables contains duplicates"
                )
        if not set(missing).issubset(required) or (available and missing):
            raise DataFoundationAuditError(f"{item_label} availability does not close")
        _require_text(decision["reason"], f"{item_label}.reason")
    if cells != expected_cells:
        raise DataFoundationAuditError(f"{label} is incomplete")
    return {
        (decision["position"], decision["target"]): bool(decision["available"])
        for decision in decisions
    }


def _validate_baseline(value: Any) -> Mapping[str, Any]:
    _assert_privacy_safe(value, label="baseline")
    baseline = _require_exact_keys(value, _BASELINE_KEYS, label="baseline")
    _require_exact_int(
        baseline["baseline_schema_version"],
        BASELINE_SCHEMA_VERSION,
        label="baseline.baseline_schema_version",
    )
    if baseline["baseline_type"] != BASELINE_TYPE:
        raise DataFoundationAuditError("baseline.baseline_type is unsupported")
    build_command = _require_text(baseline["build_command"], "baseline.build_command")
    if build_command != BUILD_COMMAND:
        raise DataFoundationAuditError("baseline.build_command is not the frozen command")
    _assert_aggregate_safe(build_command, label="baseline.build_command")

    implementation = _require_exact_keys(
        baseline["implementation"],
        _BASELINE_IMPLEMENTATION_KEYS,
        label="baseline.implementation",
    )
    _require_git_commit(implementation["git_commit"], "baseline implementation commit")
    if implementation["git_source_binding_policy"] != GIT_SOURCE_BINDING_POLICY:
        raise DataFoundationAuditError("baseline implementation policy is unsupported")
    _require_exact_int(
        implementation["source_blob_count"],
        FROZEN_SOURCE_BLOB_COUNT,
        label="baseline.implementation.source_blob_count",
    )
    _require_sha256(
        implementation["source_tree_sha256"],
        "baseline.implementation.source_tree_sha256",
    )

    production = _require_exact_keys(
        baseline["production_audit"],
        _PRODUCTION_AUDIT_KEYS,
        label="baseline.production_audit",
    )
    _canonical_path(
        production["relative_path"], label="baseline.production_audit.relative_path"
    )
    _require_non_negative_int(production["bytes"], "baseline.production_audit.bytes")
    _require_sha256(
        production["file_sha256"], "baseline.production_audit.file_sha256"
    )
    _require_sha256(
        production["audit_payload_sha256"],
        "baseline.production_audit.audit_payload_sha256",
    )
    rerun_relative = _canonical_path(
        production["rerun_relative_path"],
        label="baseline.production_audit.rerun_relative_path",
    )
    if rerun_relative == production["relative_path"]:
        raise DataFoundationAuditError(
            "baseline production rerun path must differ from the primary audit path"
        )
    _require_non_negative_int(
        production["rerun_bytes"], "baseline.production_audit.rerun_bytes"
    )
    _require_sha256(
        production["rerun_file_sha256"],
        "baseline.production_audit.rerun_file_sha256",
    )
    run_count = _require_non_negative_int(
        production["deterministic_run_count"],
        "baseline.production_audit.deterministic_run_count",
    )
    if run_count < 2:
        raise DataFoundationAuditError(
            "baseline production audit requires at least two deterministic runs"
        )
    if not _require_bool(
        production["rerun_byte_identical"],
        label="baseline.production_audit.rerun_byte_identical",
    ):
        raise DataFoundationAuditError("baseline production rerun is not byte-identical")

    sources = _require_mapping(baseline["sources"], "baseline.sources")
    if not sources:
        raise DataFoundationAuditError("baseline.sources must not be empty")
    for name, source_value in sources.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise DataFoundationAuditError("baseline source name is invalid")
        source = _require_exact_keys(
            source_value, _BASELINE_SOURCE_KEYS, label=f"baseline.sources.{name}"
        )
        for field in (
            "capability_contract_hash",
            "dataset_id",
            "descriptor_file_sha256",
            "manifest_sha256",
            "source_descriptor_hash",
        ):
            _require_sha256(source[field], f"baseline.sources.{name}.{field}")
        for field in ("revision", "source_id"):
            _require_text(source[field], f"baseline.sources.{name}.{field}")
        for field in (
            "condition_count",
            "row_count",
            "run_count",
            "task_count",
            "trajectory_count",
        ):
            _require_non_negative_int(source[field], f"baseline.sources.{name}.{field}")
        status = _require_status_counts(
            source["dataset_status_counts"],
            label=f"baseline.sources.{name}.dataset_status_counts",
        )
        if sum(status.values()) != source["row_count"]:
            raise DataFoundationAuditError(
                f"baseline.sources.{name} status counts do not sum to row_count"
            )
    return baseline


def _validate_audit(value: Any) -> Mapping[str, Any]:
    _assert_privacy_safe(value, label="audit")
    audit = _require_exact_keys(value, _AUDIT_KEYS, label="audit")
    _require_exact_int(
        audit["data_foundation_v2_audit_schema_version"],
        AUDIT_SCHEMA_VERSION,
        label="audit.data_foundation_v2_audit_schema_version",
    )
    _require_exact_int(
        audit["dataset_schema_version"],
        CAPABILITY_DATASET_SCHEMA_VERSION,
        label="audit.dataset_schema_version",
    )
    _require_sha256(audit["audit_payload_sha256"], "audit.audit_payload_sha256")
    _require_text(audit["build_command"], "audit.build_command")
    implementation = _require_exact_keys(
        audit["implementation"],
        _AUDIT_IMPLEMENTATION_KEYS,
        label="audit.implementation",
    )
    _require_git_commit(implementation["git_commit"], "audit implementation commit")
    _require_sha256(
        implementation["source_tree_sha256"], "audit.implementation.source_tree_sha256"
    )
    if implementation["git_source_binding_policy"] != GIT_SOURCE_BINDING_POLICY:
        raise DataFoundationAuditError("audit implementation policy is unsupported")
    runtime = _require_exact_keys(
        implementation["runtime"], _RUNTIME_KEYS, label="audit.implementation.runtime"
    )
    for key in sorted(_RUNTIME_KEYS):
        _require_text(runtime[key], f"audit.implementation.runtime.{key}")

    sources = _require_mapping(audit["sources"], "audit.sources")
    source_count = _require_non_negative_int(audit["source_count"], "audit.source_count")
    if source_count != len(sources) or not sources:
        raise DataFoundationAuditError("audit.source_count does not close")
    for name, source_value in sources.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise DataFoundationAuditError("audit source name is invalid")
        source = _require_exact_keys(
            source_value, _AUDIT_SOURCE_KEYS, label=f"audit.sources.{name}"
        )
        if source["source_name"] != name:
            raise DataFoundationAuditError(f"audit.sources.{name}.source_name mismatch")
        source_build_command = _require_text(
            source["build_command"], f"audit.sources.{name}.build_command"
        )
        if source_build_command != audit["build_command"]:
            raise DataFoundationAuditError(
                f"audit.sources.{name}.build_command does not close"
            )
        descriptor_hash = _require_sha256(
            source["source_descriptor_hash"],
            f"audit.sources.{name}.source_descriptor_hash",
        )
        contract_hash = _require_sha256(
            source["capability_contract_hash"],
            f"audit.sources.{name}.capability_contract_hash",
        )
        descriptor = _require_mapping(
            source["source_descriptor"], f"audit.sources.{name}.source_descriptor"
        )
        source_id = _require_text(
            descriptor.get("source_id"), f"audit.sources.{name}.source_descriptor.source_id"
        )
        decisions = _validate_capability_decisions(
            source["capability_decision_matrix"],
            source_id=source_id,
            capability_contract_hash=contract_hash,
            label=f"audit.sources.{name}.capability_decision_matrix",
        )
        dataset = _validate_dataset(
            source["dataset"], label=f"audit.sources.{name}.dataset"
        )
        cells = {
            (cell["position"], cell["target"]): cell
            for cell in dataset["by_position_target"]
        }
        for cell, available in decisions.items():
            if available:
                continue
            row = cells[cell]
            if row["row_count"] != 0 or any(row["status_counts"].values()):
                raise DataFoundationAuditError(
                    f"audit.sources.{name} capability-unavailable cell emitted rows"
                )
        if dataset["source_descriptor_hash"] != descriptor_hash:
            raise DataFoundationAuditError(
                f"audit.sources.{name} dataset descriptor hash does not close"
            )
        if dataset["capability_contract_hash"] != contract_hash:
            raise DataFoundationAuditError(
                f"audit.sources.{name} dataset capability hash does not close"
            )
        identity = _require_exact_keys(
            source["identity_counts"],
            _IDENTITY_KEYS,
            label=f"audit.sources.{name}.identity_counts",
        )
        for key in sorted(_IDENTITY_KEYS):
            _require_non_negative_int(
                identity[key], f"audit.sources.{name}.identity_counts.{key}"
            )
        artifacts = _require_mapping(
            source["artifacts"], f"audit.sources.{name}.artifacts"
        )
        if "descriptor" not in artifacts:
            raise DataFoundationAuditError(
                f"audit.sources.{name}.artifacts is missing descriptor evidence"
            )
        for artifact_name, artifact in artifacts.items():
            if not isinstance(artifact_name, str) or not re.fullmatch(
                r"[a-z][a-z0-9_]*", artifact_name
            ):
                raise DataFoundationAuditError(
                    f"audit.sources.{name} artifact name is invalid"
                )
            _validate_artifact_entry(
                artifact, label=f"audit.sources.{name}.artifacts.{artifact_name}"
            )
    return audit


def verify_data_foundation_baseline(
    repo_root: Path,
    *,
    baseline_path: Path = DEFAULT_BASELINE,
    audit_path: Path = DEFAULT_OUTPUT,
    full_source_verify: bool = False,
    require_workspace_source_match: bool = False,
) -> dict[str, Any]:
    """Verify a tracked baseline lock against one immutable production audit."""

    root = repo_root.resolve(strict=True)
    baseline_relative = _require_relative_argument(baseline_path, label="baseline argument")
    baseline_file = _resolve_strict_repo_file(root, baseline_relative, label="baseline")
    baseline = _validate_baseline(load_strict_json(baseline_file, label="baseline"))

    audit_relative = _require_relative_argument(audit_path, label="audit argument")
    production = _require_mapping(baseline["production_audit"], "baseline.production_audit")
    if audit_relative != production["relative_path"]:
        raise DataFoundationAuditError(
            "audit argument does not match baseline.production_audit.relative_path"
        )
    strict_audit_file = _resolve_strict_repo_file(
        root, audit_relative, label="production audit"
    )
    audit_file, _ = verify_file(
        root,
        audit_relative,
        expected_sha256=production["file_sha256"],
        expected_bytes=production["bytes"],
        label="production audit",
    )
    if audit_file != strict_audit_file:
        raise DataFoundationAuditError("production audit resolution changed during verification")
    rerun_relative = production["rerun_relative_path"]
    strict_rerun_file = _resolve_strict_repo_file(
        root, rerun_relative, label="production audit rerun"
    )
    rerun_file, _ = verify_file(
        root,
        rerun_relative,
        expected_sha256=production["rerun_file_sha256"],
        expected_bytes=production["rerun_bytes"],
        label="production audit rerun",
    )
    if rerun_file != strict_rerun_file:
        raise DataFoundationAuditError(
            "production audit rerun resolution changed during verification"
        )
    try:
        if rerun_file.samefile(audit_file):
            raise DataFoundationAuditError(
                "production audit rerun must be a distinct file from the primary audit"
            )
    except OSError as exc:
        raise DataFoundationAuditError(
            "cannot compare production audit and rerun file identities"
        ) from exc
    if rerun_file.read_bytes() != audit_file.read_bytes():
        raise DataFoundationAuditError(
            "production audit rerun bytes differ from the primary audit"
        )
    loaded_audit = load_strict_json(audit_file, label="production audit")
    verify_audit_payload(loaded_audit)
    audit = _validate_audit(loaded_audit)
    if audit["audit_payload_sha256"] != production["audit_payload_sha256"]:
        raise DataFoundationAuditError("production audit payload hash does not match baseline")
    if audit["build_command"] != baseline["build_command"]:
        raise DataFoundationAuditError("production audit build command does not match baseline")

    baseline_implementation = _require_mapping(
        baseline["implementation"], "baseline.implementation"
    )
    audit_implementation = _require_mapping(audit["implementation"], "audit.implementation")
    for field in ("git_commit", "git_source_binding_policy", "source_tree_sha256"):
        if audit_implementation[field] != baseline_implementation[field]:
            raise DataFoundationAuditError(
                f"production audit implementation {field} does not match baseline"
            )
    git_source = verify_frozen_git_source_tree(
        root,
        git_commit=baseline_implementation["git_commit"],
        expected_source_tree_sha256=baseline_implementation["source_tree_sha256"],
        expected_blob_count=baseline_implementation["source_blob_count"],
    )
    workspace_tree_hash = strict_workspace_source_tree_sha256(root)
    workspace_source_matches_frozen = (
        workspace_tree_hash == baseline_implementation["source_tree_sha256"]
    )
    if (
        require_workspace_source_match or full_source_verify
    ) and not workspace_source_matches_frozen:
        raise DataFoundationAuditError(
            "current relevant source tree does not match the frozen implementation"
        )

    baseline_sources = _require_mapping(baseline["sources"], "baseline.sources")
    audit_sources = _require_mapping(audit["sources"], "audit.sources")
    if set(audit_sources) != set(baseline_sources):
        raise DataFoundationAuditError("production audit source set does not match baseline")

    output_sources: dict[str, Any] = {}
    for name in sorted(baseline_sources):
        locked = _require_mapping(baseline_sources[name], f"baseline.sources.{name}")
        source = _require_mapping(audit_sources[name], f"audit.sources.{name}")
        artifacts = _require_mapping(source["artifacts"], f"audit.sources.{name}.artifacts")
        descriptor_artifact = _require_mapping(
            artifacts["descriptor"], f"audit.sources.{name}.artifacts.descriptor"
        )
        if descriptor_artifact["sha256"] != locked["descriptor_file_sha256"]:
            raise DataFoundationAuditError(
                f"source {name} descriptor artifact SHA-256 does not match baseline"
            )
        strict_descriptor_file = _resolve_strict_repo_file(
            root,
            descriptor_artifact["path"],
            label=f"source {name} descriptor",
        )
        descriptor, descriptor_evidence = load_source_descriptor(
            root,
            descriptor_artifact["path"],
            expected_sha256=locked["descriptor_file_sha256"],
        )
        if strict_descriptor_file.stat().st_size != descriptor_evidence.bytes:
            raise DataFoundationAuditError(
                f"source {name} descriptor resolution changed during verification"
            )
        if descriptor_evidence.bytes != descriptor_artifact["bytes"]:
            raise DataFoundationAuditError(
                f"source {name} descriptor artifact byte size does not close"
            )
        if descriptor_artifact["file_count"] != 1 or descriptor_artifact["sha256_kind"] != "file_bytes":
            raise DataFoundationAuditError(
                f"source {name} descriptor artifact evidence is not a single file"
            )
        if descriptor.to_dict() != source["source_descriptor"]:
            raise DataFoundationAuditError(
                f"source {name} embedded descriptor differs from tracked descriptor"
            )

        expected_descriptor_fields = {
            "source_id": descriptor.source_id,
            "revision": descriptor.revision,
            "manifest_sha256": descriptor.manifest_sha256,
            "capability_contract_hash": descriptor.capabilities.contract_hash,
            "source_descriptor_hash": descriptor.descriptor_hash,
        }
        for field, actual in expected_descriptor_fields.items():
            if locked[field] != actual:
                raise DataFoundationAuditError(
                    f"source {name} {field} does not match the tracked descriptor"
                )
        if source["source_descriptor_hash"] != descriptor.descriptor_hash:
            raise DataFoundationAuditError(
                f"source {name} audit descriptor hash does not close"
            )
        if source["capability_contract_hash"] != descriptor.capabilities.contract_hash:
            raise DataFoundationAuditError(
                f"source {name} audit capability hash does not close"
            )
        canonical_decisions = [
            decide_target_capability(
                descriptor.capabilities,
                position,
                target,
            ).to_dict()
            for position in PredictionPosition
            for target in PredictionTarget
        ]
        if source["capability_decision_matrix"] != canonical_decisions:
            raise DataFoundationAuditError(
                f"source {name} capability decision matrix is not derived from "
                "the tracked descriptor"
            )

        manifest_matches = [
            _require_mapping(item, f"audit.sources.{name}.artifacts.{artifact_name}")
            for artifact_name, item in artifacts.items()
            if _require_mapping(
                item, f"audit.sources.{name}.artifacts.{artifact_name}"
            )["path"]
            == descriptor.manifest_path
        ]
        if len(manifest_matches) != 1:
            raise DataFoundationAuditError(
                f"source {name} must contain exactly one manifest artifact"
            )
        manifest_artifact = manifest_matches[0]
        if (
            manifest_artifact["sha256"] != descriptor.manifest_sha256
            or manifest_artifact["sha256_kind"] != "file_bytes"
            or manifest_artifact["file_count"] != 1
        ):
            raise DataFoundationAuditError(
                f"source {name} manifest artifact evidence does not close"
            )
        strict_manifest_file = _resolve_strict_repo_file(
            root,
            descriptor.manifest_path,
            label=f"source {name} manifest",
        )
        verified_manifest_file, manifest_evidence = verify_file(
            root,
            descriptor.manifest_path,
            expected_sha256=descriptor.manifest_sha256,
            expected_bytes=manifest_artifact["bytes"],
            label=f"source {name} manifest",
        )
        if verified_manifest_file != strict_manifest_file:
            raise DataFoundationAuditError(
                f"source {name} manifest resolution changed during verification"
            )
        if manifest_evidence.sha256 != locked["manifest_sha256"]:
            raise DataFoundationAuditError(
                f"source {name} manifest SHA-256 does not match baseline"
            )

        dataset = _require_mapping(source["dataset"], f"audit.sources.{name}.dataset")
        identity = _require_mapping(
            source["identity_counts"], f"audit.sources.{name}.identity_counts"
        )
        comparisons = {
            "dataset_id": dataset["dataset_id"],
            "row_count": dataset["row_count"],
            "dataset_status_counts": dataset["status_counts"],
            "task_count": identity["task_count"],
            "trajectory_count": identity["trajectory_count"],
            "run_count": identity["run_count"],
            "condition_count": identity["condition_count"],
        }
        for field, actual in comparisons.items():
            if locked[field] != actual:
                raise DataFoundationAuditError(
                    f"source {name} {field} does not match baseline"
                )
        output_sources[name] = {
            "dataset_id": dataset["dataset_id"],
            "identity_counts": dict(identity),
            "row_count": dataset["row_count"],
            "status_counts": dict(dataset["status_counts"]),
        }

    if full_source_verify:
        if set(audit_sources) != set(FULL_SOURCE_LOADERS):
            raise DataFoundationAuditError(
                "full source verification has no exact loader set for frozen sources"
            )
        for name in sorted(audit_sources):
            rebuilt = FULL_SOURCE_LOADERS[name](root)
            if rebuilt != audit_sources[name]:
                raise DataFoundationAuditError(
                    f"full source verification mismatch for {name}"
                )

    summary = {
        "audit_payload_sha256": audit["audit_payload_sha256"],
        "deterministic_run_count": production["deterministic_run_count"],
        "implementation": {
            "source_blob_count": git_source.blob_count,
            "git_commit": baseline_implementation["git_commit"],
            "source_tree_sha256": baseline_implementation["source_tree_sha256"],
        },
        "rerun_byte_identical": True,
        "raw_artifacts_rehashed": full_source_verify,
        "source_count": len(output_sources),
        "sources": output_sources,
        "workspace_source_matches_frozen": workspace_source_matches_frozen,
    }
    _assert_aggregate_safe(summary, label="baseline verification summary")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the immutable Data Foundation v2 baseline lock."
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--full-source-verify",
        action="store_true",
        help="rehash and rebuild all raw source summaries (includes the multi-GB archive)",
    )
    parser.add_argument(
        "--require-workspace-source-match",
        action="store_true",
        help="fail if the current source tree differs from the frozen implementation",
    )
    args = parser.parse_args()
    try:
        summary = verify_data_foundation_baseline(
            REPO_ROOT,
            baseline_path=args.baseline,
            audit_path=args.audit,
            full_source_verify=args.full_source_verify,
            require_workspace_source_match=args.require_workspace_source_match,
        )
    except (DataFoundationAuditError, OSError, ValueError) as exc:
        parser.exit(2, f"Data Foundation baseline verification failed: {exc}\n")
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
