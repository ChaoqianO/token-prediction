from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


HANDOFF_SCHEMA_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = REPO_ROOT / "workspace"
DEFAULT_MANIFEST_SUMMARY = DEFAULT_WORKSPACE / "external" / "bagen" / "manifest_summary.json"
DEFAULT_BAGEN_AUDIT_DIR = DEFAULT_WORKSPACE / "external" / "bagen" / "audits"
DEFAULT_BAGEN_COMBINED_AUDIT = (
    DEFAULT_WORKSPACE / "external" / "bagen" / "combined_swebench_audit.json"
)
DEFAULT_SPEND_INVENTORY = (
    DEFAULT_WORKSPACE / "external" / "spend_your_money" / "gpt_5.2_inventory.json"
)
DEFAULT_SPEND_TRAJECTORY_AUDIT = (
    DEFAULT_WORKSPACE
    / "external"
    / "spend_your_money"
    / "gpt_5.2_trajectory_audit.json"
)
DEFAULT_OUTPUT = DEFAULT_WORKSPACE / "handoffs" / "trajectory_ingestion_summary.json"

BAGEN_REPO = "MLL-Lab/BAGEN"
BAGEN_REVISION = "58189576e54b675fdd0e1d6c1c9f189c2992732f"
BAGEN_FAMILY_AUDITS = {
    "claude-opus4.7": "claude-opus4.7.json",
    "claude-sonnet4.6": "claude-sonnet4.6.json",
    "gemini3.1": "gemini3.1.json",
    "gpt5.2instant": "gpt5.2instant.json",
    "qwen3-235b": "qwen3-235b.json",
}
BAGEN_FAMILY_ROOTS = {
    family: f"swebench-origin-{family}" for family in BAGEN_FAMILY_AUDITS
}
BAGEN_SOURCE_PATHS = {
    "reader": "src/token_prediction/collection/bagen_swebench.py",
    "builder": "src/token_prediction/dataset/builder.py",
    "labels": "src/token_prediction/dataset/labels.py",
    "audit": "scripts/audit_bagen_combined.py",
    "family_audit": "scripts/audit_bagen_swebench.py",
}
BAGEN_TASK_CROSS_FAMILY_DISTRIBUTION = {"4": 4, "5": 60}
BAGEN_COMBINED_DATASET_ID = "c845574fd0c0e3da3b6a4d1782787d3d53a1b71db738314836f08419bcb57a60"
BAGEN_COMBINED_AUDIT_SCHEMA_VERSION = 1
BAGEN_COMBINED_AUDIT_SOURCE_ID = "bagen_swebench_combined_audit_v1"
BAGEN_COMBINED_EXPECTED_COUNTS = {
    "task_id_count": 64,
    "run_id_count": 316,
    "trajectory_id_count": 316,
    "condition_id_count": 9,
    "dataset_row_count": 45_564,
    "raw_file_count": 316,
    "raw_bytes": 263_785_722,
}

SPEND_REPO = "loong0814/openhands_trajectories"
SPEND_REVISION = "fa9cbb063f770df596da95af24f7af3b8f595778"
SPEND_INVENTORY_SOURCE_ID = (
    "spend_your_money/openhands_trajectories:gpt_5.2_4runs"
)
SPEND_ARCHIVE_BYTES = 2_908_192_516
SPEND_ARCHIVE_SHA256 = "993abcb55aae423f9067d5e6c8e1aeaccf83b9ce31474a215982686527934214"
SPEND_ARCHIVE_XET_ETAG = "5824153171526bdfb245b74fca532407cf68add02079b4fa0f7c1cf47ea1c1c8"
SPEND_ARCHIVE_NAME = "gpt_5.2_4runs.tar.gz"
SPEND_ARCHIVE_WRAPPER = "gpt_5.2_4runs"
SPEND_READER_SOURCE_ID = "openhands_archive_trajectory_v2"
SPEND_TRAJECTORY_AUDIT_SCHEMA_VERSION = 1
SPEND_INVENTORY_SCHEMA_VERSION = 2
SPEND_DATASET_SCHEMA_VERSION = 1
SPEND_FEATURE_SCHEMA_VERSION = 2

# These are semantic provenance pins, not merely informational hashes. A reader,
# label, builder, or audit change requires regenerating the trajectory audit and
# deliberately updating this map before a new handoff can be frozen.
SPEND_CODE_ARTIFACT_PINS = {
    "reader": {
        "path": "src/token_prediction/collection/openhands_trajectory.py",
        "sha256": "67d719eb8182a8dd7339e1fa107300caaeafa46943a4b59c814812004e40cb07",
    },
    "builder": {
        "path": "src/token_prediction/dataset/builder.py",
        "sha256": "8a3ec9166d2ceec0f6060fb88152c34e6b6223974ad940d8f68dc40c0c03f528",
    },
    "labels": {
        "path": "src/token_prediction/dataset/labels.py",
        "sha256": "bd29dff3622e1faa03f6b052b49b5887a505d1b5926605ead930cef0feee4a51",
    },
    "audit": {
        "path": "scripts/audit_openhands_trajectory.py",
        "sha256": "f0e7a6e2beae4470148b52455a4932c08c2257e30453b63d992fc99ec5c95747",
    },
}

POSITIONS = ("task_launch", "task_pre", "task_update", "call_pre", "call_update")
TARGETS = (
    "task_total_accounted_tokens",
    "task_unknown_remaining_tokens",
    "call_unknown_billable_tokens",
    "call_billable_output_tokens",
    "call_final_response_output_tokens",
    "call_remaining_output_tokens",
)
STATUSES = ("observed", "missing", "censored", "invalid")
BUILDER_CELLS = frozenset(
    {
        ("task_launch", "task_total_accounted_tokens"),
        ("task_pre", "task_unknown_remaining_tokens"),
        ("task_update", "task_unknown_remaining_tokens"),
        ("call_pre", "call_unknown_billable_tokens"),
        ("call_pre", "call_billable_output_tokens"),
        ("call_pre", "call_final_response_output_tokens"),
        ("call_update", "call_remaining_output_tokens"),
    }
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

SPEND_DECLARED_OBSERVABLES = (
    "attempt_usage",
    "call_usage",
    "request_messages",
    "task_usage",
    "tool_events",
)
SPEND_CAPABILITY_FIELDS = {
    "request_tokens_local": {
        "available",
        "observed_count",
        "missing_count",
        "reason",
        "gates_targets",
    },
    "attempt_usage": {
        "available",
        "complete_count",
        "missing_count",
        "invalid_count",
        "scope",
    },
    "task_usage": {
        "available",
        "complete_count",
        "missing_count",
        "invalid_count",
        "scope",
        "explicit_zero_call_count",
        "explicit_zero_call_source",
        "explicit_zero_call_never_imputed",
        "explicit_zero_call_criteria",
        "all_preserved_sessions_count",
        "current_session_without_completion_boundaries_count",
        "missing_incomplete_extra_session_count",
        "missing_no_completion_or_task_metrics_count",
    },
    "retry": {"supported", "retry_count", "reason"},
    "tool_events": {
        "available",
        "started_count",
        "completed_count",
        "failed_count",
        "failure_observable_count",
        "failure_unobservable_count",
        "failure_status_scope",
    },
    "errors": {
        "task_error_available",
        "task_error_count",
        "attempt_error_available",
        "attempt_error_count",
        "provider_error_envelope_available",
        "provider_error_envelope_count",
        "provider_error_envelope_semantics",
        "reason",
    },
    "task_termination": {
        "available",
        "finished_count",
        "aborted_count",
        "observed_lifecycle_count",
        "censored_lifecycle_count",
        "task_log_observed_count",
        "task_log_missing_count",
        "source",
    },
    "generation_checkpoint": {
        "available",
        "observed_count",
        "reason",
        "gates_position",
    },
    "session_reconciliation": {
        "completion_logging_complete_count",
        "completion_logging_incomplete_count",
        "task_usage_reconciled_count",
        "task_usage_unreconciled_count",
        "metrics_completion_extra_task_count",
        "metrics_completion_extra_count",
        "metrics_missing_completion_task_count",
        "metrics_missing_completion_count",
        "history_llm_metrics_ledger_match_count",
        "history_llm_metrics_ledger_mismatch_count",
        "message_prefix_reset_count",
        "repeated_request_snapshot_count",
        "response_not_materialized_in_next_request_count",
        "reasoning_subset_anomaly_count",
    },
}
SPEND_CAPABILITY_METRICS = {
    "request_tokens_local": {
        "observed_count": "request_tokens_local_observed_count",
        "missing_count": "request_tokens_local_missing_count",
    },
    "attempt_usage": {
        "complete_count": "attempt_usage_complete_count",
        "missing_count": "attempt_usage_missing_count",
        "invalid_count": "attempt_usage_invalid_count",
    },
    "task_usage": {
        "complete_count": "task_usage_complete_count",
        "missing_count": "task_usage_missing_count",
        "invalid_count": "task_usage_invalid_count",
        "explicit_zero_call_count": "task_usage_explicit_zero_call_count",
        "all_preserved_sessions_count": "task_usage_all_preserved_sessions_count",
        "current_session_without_completion_boundaries_count": (
            "task_usage_current_session_only_count"
        ),
        "missing_incomplete_extra_session_count": "task_usage_missing_extra_session_count",
        "missing_no_completion_or_task_metrics_count": "task_usage_missing_no_evidence_count",
    },
    "tool_events": {
        "started_count": "tool_started_count",
        "completed_count": "tool_completed_count",
        "failed_count": "tool_failed_count",
        "failure_observable_count": "tool_terminal_failure_observable_count",
        "failure_unobservable_count": "tool_terminal_failure_unobservable_count",
    },
    "errors": {
        "task_error_count": "task_error_count",
        "attempt_error_count": "api_failed_count",
        "provider_error_envelope_count": "provider_error_envelope_count",
    },
    "task_termination": {
        "finished_count": "task_finished_count",
        "aborted_count": "task_aborted_count",
        "observed_lifecycle_count": "task_lifecycle_observed_count",
        "censored_lifecycle_count": "task_lifecycle_censored_count",
        "task_log_observed_count": "task_log_observed_count",
        "task_log_missing_count": "task_log_missing_count",
    },
    "generation_checkpoint": {"observed_count": "generation_checkpoint_count"},
    "session_reconciliation": {
        "completion_logging_complete_count": "completion_logging_complete_count",
        "completion_logging_incomplete_count": "completion_logging_incomplete_count",
        "task_usage_reconciled_count": "task_usage_reconciled_count",
        "task_usage_unreconciled_count": "task_usage_unreconciled_count",
        "metrics_completion_extra_task_count": "metrics_completion_extra_task_count",
        "metrics_completion_extra_count": "metrics_completion_extra_count",
        "metrics_missing_completion_task_count": "metrics_missing_completion_task_count",
        "metrics_missing_completion_count": "metrics_missing_completion_count",
        "history_llm_metrics_ledger_match_count": "history_llm_metrics_ledger_match_count",
        "history_llm_metrics_ledger_mismatch_count": (
            "history_llm_metrics_ledger_mismatch_count"
        ),
        "message_prefix_reset_count": "message_prefix_reset_count",
        "repeated_request_snapshot_count": "repeated_request_snapshot_count",
        "response_not_materialized_in_next_request_count": (
            "response_not_materialized_in_next_request_count"
        ),
        "reasoning_subset_anomaly_count": "reasoning_subset_anomaly_count",
    },
}

DEFAULT_CHANGED_FILES = (
    "src/token_prediction/collection/__init__.py",
    "src/token_prediction/collection/bagen_swebench.py",
    "src/token_prediction/collection/openhands_trajectory.py",
    "scripts/audit_bagen_manifest.py",
    "scripts/audit_bagen_combined.py",
    "scripts/audit_bagen_swebench.py",
    "scripts/audit_openhands_archive.py",
    "scripts/audit_openhands_trajectory.py",
    "scripts/download_bagen.py",
    "scripts/download_spend_archive.py",
    "scripts/freeze_trajectory_handoff.py",
    "tests/test_bagen_swebench_reader.py",
    "tests/test_bagen_combined_audit.py",
    "tests/test_openhands_archive_audit.py",
    "tests/test_openhands_trajectory_audit.py",
    "tests/test_openhands_trajectory_reader.py",
    "tests/test_freeze_trajectory_handoff.py",
    "docs/trajectory-data-audit.md",
)


class TrajectoryHandoffError(ValueError):
    """Raised when the evidence is incomplete or mutually inconsistent."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _semantic_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _update_framed_hash(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def _bagen_canonical_family_sha256(hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256(b"bagen-swebench-canonical-family-v1\0")
    for relative_path in sorted(hashes):
        _update_framed_hash(digest, relative_path.encode("utf-8"))
        _update_framed_hash(digest, bytes.fromhex(hashes[relative_path]))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> Any:
    raise TrajectoryHandoffError(f"JSON contains a non-finite constant: {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise TrajectoryHandoffError(f"JSON contains a duplicate field: {key!r}")
        value[key] = item
    return value


def _reject_non_finite_numbers(value: Any, *, location: str = "JSON") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TrajectoryHandoffError(f"{location} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_non_finite_numbers(item, location=f"{location}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_finite_numbers(item, location=f"{location}[{index}]")


def _strict_json_loads(value: str, *, label: str) -> Any:
    try:
        result = json.loads(
            value,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise TrajectoryHandoffError(f"{label} is not strict JSON") from exc
    _reject_non_finite_numbers(result, location=label)
    return result


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise TrajectoryHandoffError(f"{label} is missing: {source.name}")
    try:
        with source.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrajectoryHandoffError(f"{label} is not valid UTF-8 JSON") from exc
    _reject_non_finite_numbers(value, location=label)
    if not isinstance(value, dict):
        raise TrajectoryHandoffError(f"{label} must contain one JSON object")
    return value


def _required_mapping(value: Mapping[str, Any], key: str, *, label: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise TrajectoryHandoffError(f"{label}.{key} must be an object")
    return result


def _required_list(value: Mapping[str, Any], key: str, *, label: str) -> list[Any]:
    result = value.get(key)
    if not isinstance(result, list):
        raise TrajectoryHandoffError(f"{label}.{key} must be a list")
    return result


def _required_text(value: Mapping[str, Any], key: str, *, label: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip() or "\x00" in result:
        raise TrajectoryHandoffError(f"{label}.{key} must be non-empty text")
    return result


def _required_int(value: Mapping[str, Any], key: str, *, label: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int) or result < 0:
        raise TrajectoryHandoffError(f"{label}.{key} must be a non-negative integer")
    return result


def _required_bool(value: Mapping[str, Any], key: str, *, label: str) -> bool:
    result = value.get(key)
    if not isinstance(result, bool):
        raise TrajectoryHandoffError(f"{label}.{key} must be boolean")
    return result


def _required_sha(value: Mapping[str, Any], key: str, *, label: str) -> str:
    result = _required_text(value, key, label=label)
    if SHA256_RE.fullmatch(result) is None:
        raise TrajectoryHandoffError(f"{label}.{key} must be a lowercase SHA256")
    return result


def _safe_relative(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise TrajectoryHandoffError(f"{label} must be a canonical POSIX relative path")
    windows = PureWindowsPath(value)
    path = PurePosixPath(value)
    if (
        "\\" in value
        or path.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or value.startswith(("/", "\\"))
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise TrajectoryHandoffError(f"{label} must be a canonical POSIX relative path")
    return path.as_posix()


def _safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise TrajectoryHandoffError(f"{label} must be a non-empty bounded string")
    if any(character in value for character in ("\x00", "\n", "\r")):
        raise TrajectoryHandoffError(f"{label} contains a control character")
    return value


def _safe_segment(value: Any, *, label: str) -> str:
    segment = _safe_id(value, label=label)
    if segment in {".", ".."} or "/" in segment or "\\" in segment:
        raise TrajectoryHandoffError(f"{label} must be one canonical path segment")
    return segment


def _workspace_path(path: Path, workspace_root: Path, *, label: str) -> str:
    resolved = Path(path).resolve()
    root = Path(workspace_root).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise TrajectoryHandoffError(f"{label} must be inside the workspace root") from exc
    return _safe_relative(
        (PurePosixPath("workspace") / PurePosixPath(relative.as_posix())).as_posix(),
        label=label,
    )


def _verify_file(path: Path, *, expected_bytes: int, expected_sha256: str, label: str) -> None:
    if not Path(path).is_file():
        raise TrajectoryHandoffError(f"{label} is missing")
    actual_bytes = Path(path).stat().st_size
    if actual_bytes != expected_bytes:
        raise TrajectoryHandoffError(
            f"{label} size mismatch: expected {expected_bytes}, got {actual_bytes}"
        )
    actual_sha256 = _file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise TrajectoryHandoffError(f"{label} SHA256 mismatch")


def _git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {Path(path).stat().st_size}\0".encode("ascii"))
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bagen_manifest_index(path: Path) -> dict[str, int]:
    index: dict[str, int] = {}
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise TrajectoryHandoffError(
                        f"BAGEN manifest line {line_number} is blank"
                    )
                raw = _strict_json_loads(
                    line, label=f"BAGEN manifest line {line_number}"
                )
                if not isinstance(raw, Mapping):
                    raise TrajectoryHandoffError(
                        f"BAGEN manifest line {line_number} must be an object"
                    )
                label = f"BAGEN manifest line {line_number}"
                member_path = _safe_relative(
                    _required_text(raw, "path", label=label), label=f"{label}.path"
                )
                size_bytes = _required_int(raw, "size_bytes", label=label)
                if member_path in index:
                    raise TrajectoryHandoffError(
                        f"BAGEN manifest repeats path {member_path!r}"
                    )
                index[member_path] = size_bytes
    except (OSError, UnicodeError) as exc:
        raise TrajectoryHandoffError("BAGEN manifest is not valid UTF-8 JSONL") from exc
    if not index:
        raise TrajectoryHandoffError("BAGEN manifest is empty")
    return index


def _status_counts(value: Mapping[str, Any], *, label: str) -> dict[str, int]:
    if set(value) != set(STATUSES):
        raise TrajectoryHandoffError(
            f"{label} must explicitly contain observed/missing/censored/invalid"
        )
    return {status: _required_int(value, status, label=label) for status in STATUSES}


def _normalize_spend_inventory_status_counts(
    inventory: Mapping[str, Any], *, trajectory_count: int
) -> dict[str, Any]:
    raw_telemetry = _required_mapping(
        inventory, "telemetry_status_counts", label="Spend inventory"
    )
    expected_telemetry = {"history", "metrics", "task_error_nonempty", "usage"}
    if set(raw_telemetry) != expected_telemetry:
        raise TrajectoryHandoffError("Spend inventory telemetry status fields disagree")
    telemetry: dict[str, Any] = {}
    for field in ("history", "metrics", "usage"):
        counts = _status_counts(
            _required_mapping(
                raw_telemetry,
                field,
                label="Spend inventory.telemetry_status_counts",
            ),
            label=f"Spend inventory.telemetry_status_counts.{field}",
        )
        if sum(counts.values()) != trajectory_count:
            raise TrajectoryHandoffError(
                f"Spend inventory telemetry status counts do not close for {field}"
            )
        telemetry[field] = counts
    telemetry["task_error_nonempty"] = _required_int(
        raw_telemetry,
        "task_error_nonempty",
        label="Spend inventory.telemetry_status_counts",
    )
    if telemetry["task_error_nonempty"] > trajectory_count:
        raise TrajectoryHandoffError("Spend inventory task-error count does not close")

    raw_labels = _required_mapping(
        inventory, "label_status_counts", label="Spend inventory"
    )
    if set(raw_labels) != {"evaluator_accuracy", "task_termination"}:
        raise TrajectoryHandoffError("Spend inventory label status fields disagree")
    labels: dict[str, dict[str, int]] = {}
    for field in ("evaluator_accuracy", "task_termination"):
        counts = _status_counts(
            _required_mapping(
                raw_labels, field, label="Spend inventory.label_status_counts"
            ),
            label=f"Spend inventory.label_status_counts.{field}",
        )
        if sum(counts.values()) != trajectory_count:
            raise TrajectoryHandoffError(
                f"Spend inventory label status counts do not close for {field}"
            )
        labels[field] = counts
    return {"telemetry": telemetry, "labels": labels}


def _classify_changed_files(paths: Sequence[str]) -> dict[str, list[str]]:
    output = {"source": [], "scripts": [], "tests": [], "docs": [], "other": []}
    seen: set[str] = set()
    for raw_path in paths:
        path = _safe_relative(raw_path, label="changed_file")
        if path in seen:
            raise TrajectoryHandoffError(f"changed_file is duplicated: {path}")
        seen.add(path)
        if path.startswith("src/"):
            category = "source"
        elif path.startswith("scripts/"):
            category = "scripts"
        elif path.startswith("tests/"):
            category = "tests"
        elif path.startswith("docs/"):
            category = "docs"
        else:
            category = "other"
        output[category].append(path)
    if not seen:
        raise TrajectoryHandoffError("changed_files must not be empty")
    for values in output.values():
        values.sort()
    return output


def _freeze_code_artifacts(
    *,
    repo_root: Path,
    records: Mapping[str, Mapping[str, Any]],
    expected_roles: set[str],
    label: str,
) -> dict[str, dict[str, Any]]:
    if set(records) != expected_roles:
        raise TrajectoryHandoffError(
            f"{label} code artifact roles disagree: expected {sorted(expected_roles)}"
        )
    output: dict[str, dict[str, Any]] = {}
    for role in sorted(records):
        record = records[role]
        path = _safe_relative(
            _required_text(record, "path", label=f"{label}.{role}"),
            label=f"{label}.{role}.path",
        )
        sha256 = _required_sha(record, "sha256", label=f"{label}.{role}")
        source = Path(repo_root).joinpath(*PurePosixPath(path).parts)
        if not source.is_file():
            raise TrajectoryHandoffError(f"{label}.{role} source file is missing")
        size_bytes = source.stat().st_size
        if "bytes" in record and _required_int(
            record, "bytes", label=f"{label}.{role}"
        ) != size_bytes:
            raise TrajectoryHandoffError(f"{label}.{role} source byte count disagrees")
        if _file_sha256(source) != sha256:
            raise TrajectoryHandoffError(f"{label}.{role} source SHA256 disagrees")
        output[role] = {"path": path, "bytes": size_bytes, "sha256": sha256}
    return output


def _freeze_pinned_code_artifacts(
    *, repo_root: Path, pins: Mapping[str, Mapping[str, str]], label: str
) -> dict[str, dict[str, Any]]:
    return _freeze_code_artifacts(
        repo_root=repo_root,
        records=pins,
        expected_roles=set(pins),
        label=label,
    )


def _validate_bagen_combined_audit(
    *,
    path: Path,
    repo_root: Path,
    workspace_root: Path,
    manifest_summary_path: Path,
    raw_manifest_path: Path,
    manifest_etag: str,
    manifest_counts: Mapping[str, int],
    families: Sequence[Mapping[str, Any]],
    task_mapping: Sequence[Mapping[str, Any]],
    identity_counts: Mapping[str, int],
    matrix_row_count: int,
    matrix_status_counts: Mapping[str, int],
    expected_dataset_id: str,
    expected_counts: Mapping[str, int],
    expected_task_cross_family_distribution: Mapping[str, int],
) -> dict[str, Any]:
    audit_path = Path(path)
    audit = _load_json(audit_path, label="BAGEN combined SWE-bench audit")
    expected_audit_fields = {
        "combined_audit_schema_version",
        "source_id",
        "hub",
        "manifest",
        "families",
        "family_audit_index",
        "family_dataset_id_index",
        "counts",
        "combined_dataset",
        "task_cross_family_distribution",
        "task_family_mapping",
        "condition_trajectory_counts",
        "canonical_family_index",
        "canonical_family_index_sha256",
        "canonical_trajectory_index_sha256",
        "source_files",
        "source_hashes",
        "construction",
        "audit_payload_sha256",
    }
    if set(audit) != expected_audit_fields:
        raise TrajectoryHandoffError("BAGEN combined audit fields disagree with schema v1")
    payload_sha256 = _required_sha(
        audit, "audit_payload_sha256", label="BAGEN combined audit"
    )
    payload = dict(audit)
    del payload["audit_payload_sha256"]
    if _semantic_sha256(payload) != payload_sha256:
        raise TrajectoryHandoffError("BAGEN combined audit payload SHA256 is invalid")
    if _required_int(
        audit, "combined_audit_schema_version", label="BAGEN combined audit"
    ) != BAGEN_COMBINED_AUDIT_SCHEMA_VERSION:
        raise TrajectoryHandoffError("BAGEN combined audit schema version disagrees")
    if _required_text(audit, "source_id", label="BAGEN combined audit") != (
        BAGEN_COMBINED_AUDIT_SOURCE_ID
    ):
        raise TrajectoryHandoffError("BAGEN combined audit source_id disagrees")
    hub = _required_mapping(audit, "hub", label="BAGEN combined audit")
    if (
        _required_text(hub, "repo", label="BAGEN combined audit.hub") != BAGEN_REPO
        or _required_text(
            hub, "resolved_revision", label="BAGEN combined audit.hub"
        )
        != BAGEN_REVISION
    ):
        raise TrajectoryHandoffError("BAGEN combined audit Hub pin disagrees")

    manifest = _required_mapping(audit, "manifest", label="BAGEN combined audit")
    manifest_summary = _required_mapping(
        manifest, "summary", label="BAGEN combined audit.manifest"
    )
    manifest_raw = _required_mapping(
        manifest, "raw", label="BAGEN combined audit.manifest"
    )
    expected_manifest = {
        "summary_path": _workspace_path(
            manifest_summary_path, workspace_root, label="BAGEN manifest summary"
        ),
        "summary_bytes": manifest_summary_path.stat().st_size,
        "summary_sha256": _file_sha256(manifest_summary_path),
        "raw_manifest_path": _workspace_path(
            raw_manifest_path, workspace_root, label="BAGEN raw manifest"
        ),
        "raw_manifest_bytes": raw_manifest_path.stat().st_size,
        "raw_manifest_sha256": _file_sha256(raw_manifest_path),
        "manifest_etag": manifest_etag,
        **manifest_counts,
    }
    actual_manifest = {
        "summary_path": _safe_relative(
            _required_text(
                manifest_summary, "path", label="BAGEN combined audit.manifest.summary"
            ),
            label="BAGEN combined audit.manifest.summary_path",
        ),
        "summary_bytes": _required_int(
            manifest_summary, "bytes", label="BAGEN combined audit.manifest.summary"
        ),
        "summary_sha256": _required_sha(
            manifest_summary, "sha256", label="BAGEN combined audit.manifest.summary"
        ),
        "raw_manifest_path": _safe_relative(
            _required_text(
                manifest_raw, "path", label="BAGEN combined audit.manifest.raw"
            ),
            label="BAGEN combined audit.manifest.raw_manifest_path",
        ),
        "raw_manifest_bytes": _required_int(
            manifest_raw, "bytes", label="BAGEN combined audit.manifest.raw"
        ),
        "raw_manifest_sha256": _required_sha(
            manifest_raw, "sha256", label="BAGEN combined audit.manifest.raw"
        ),
        "manifest_etag": _required_text(
            manifest_raw, "git_blob_etag", label="BAGEN combined audit.manifest.raw"
        ),
        "file_count": _required_int(
            manifest_raw, "file_count", label="BAGEN combined audit.manifest.raw"
        ),
        "total_bytes": _required_int(
            manifest_raw, "total_bytes", label="BAGEN combined audit.manifest.raw"
        ),
        "traj_json_count": _required_int(
            manifest_raw, "traj_json_count", label="BAGEN combined audit.manifest.raw"
        ),
        "traj_json_bytes": _required_int(
            manifest_raw, "traj_json_bytes", label="BAGEN combined audit.manifest.raw"
        ),
    }
    if actual_manifest != expected_manifest:
        raise TrajectoryHandoffError("BAGEN combined audit manifest evidence disagrees")

    raw_families = _required_list(audit, "families", label="BAGEN combined audit")
    if len(raw_families) != len(families):
        raise TrajectoryHandoffError("BAGEN combined audit family count disagrees")
    combined_families: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(raw_families):
        if not isinstance(raw, Mapping):
            raise TrajectoryHandoffError(
                f"BAGEN combined audit families[{index}] must be an object"
            )
        family = _required_text(raw, "family", label=f"BAGEN combined families[{index}]")
        if family in combined_families:
            raise TrajectoryHandoffError("BAGEN combined audit repeats a family")
        combined_families[family] = raw
    if set(combined_families) != {str(item["family"]) for item in families}:
        raise TrajectoryHandoffError("BAGEN combined audit family identities disagree")

    for family_record in families:
        family = str(family_record["family"])
        raw = combined_families[family]
        label = f"BAGEN combined family {family}"
        dataset = _required_mapping(raw, "dataset", label=label)
        expected = {
            "family": family,
            "family_root": family_record["family_root"],
            "local_relative_root": family_record["local_relative_root"],
            "audit_path": family_record["audit"]["local_relative_path"],
            "audit_bytes": family_record["audit"]["bytes"],
            "audit_sha256": family_record["audit"]["sha256"],
            "raw_file_count": family_record["raw_file_count"],
            "raw_bytes": family_record["raw_bytes"],
            "task_count": family_record["counts"]["task_id_count"],
            "run_count": family_record["counts"]["trajectory_id_count"],
            "trajectory_count": family_record["counts"]["trajectory_id_count"],
            "condition_count": family_record["counts"]["condition_id_count"],
            "dataset_id": family_record["dataset"]["dataset_id"],
            "dataset_row_count": family_record["dataset"]["row_count"],
            "dataset_schema_version": SPEND_DATASET_SCHEMA_VERSION,
            "feature_schema_version": SPEND_FEATURE_SCHEMA_VERSION,
            "canonical_content_sha256": family_record["canonical"]["aggregate_sha256"],
        }
        actual = {
            "family": family,
            "family_root": _required_text(raw, "family_root", label=label),
            "local_relative_root": _safe_relative(
                _required_text(raw, "local_relative_root", label=label),
                label=f"{label}.local_relative_root",
            ),
            "audit_path": _safe_relative(
                _required_text(raw, "audit_path", label=label),
                label=f"{label}.audit_path",
            ),
            "audit_bytes": _required_int(raw, "audit_bytes", label=label),
            "audit_sha256": _required_sha(raw, "audit_sha256", label=label),
            "raw_file_count": _required_int(raw, "raw_file_count", label=label),
            "raw_bytes": _required_int(raw, "raw_bytes", label=label),
            "task_count": _required_int(raw, "task_count", label=label),
            "run_count": _required_int(raw, "run_count", label=label),
            "trajectory_count": _required_int(raw, "trajectory_count", label=label),
            "condition_count": _required_int(raw, "condition_count", label=label),
            "dataset_id": _required_sha(dataset, "dataset_id", label=f"{label}.dataset"),
            "dataset_row_count": _required_int(
                dataset, "row_count", label=f"{label}.dataset"
            ),
            "dataset_schema_version": _required_int(
                dataset, "schema_version", label=f"{label}.dataset"
            ),
            "feature_schema_version": _required_int(
                dataset, "feature_schema_version", label=f"{label}.dataset"
            ),
            "canonical_content_sha256": _required_sha(
                raw, "canonical_content_sha256", label=label
            ),
        }
        if actual != expected:
            raise TrajectoryHandoffError(
                f"BAGEN combined audit family evidence disagrees for {family}"
            )

    expected_family_audit_index = {
        str(item["family"]): {
            "path": item["audit"]["local_relative_path"],
            "bytes": item["audit"]["bytes"],
            "sha256": item["audit"]["sha256"],
        }
        for item in families
    }
    if _required_mapping(
        audit, "family_audit_index", label="BAGEN combined audit"
    ) != expected_family_audit_index:
        raise TrajectoryHandoffError("BAGEN combined family audit index disagrees")
    expected_family_dataset_index = {
        str(item["family"]): item["dataset"]["dataset_id"] for item in families
    }
    if _required_mapping(
        audit, "family_dataset_id_index", label="BAGEN combined audit"
    ) != expected_family_dataset_index:
        raise TrajectoryHandoffError("BAGEN combined family dataset index disagrees")

    derived_counts = {
        "task_id_count": identity_counts["task_id_count"],
        "run_id_count": identity_counts["run_id_count"],
        "trajectory_id_count": identity_counts["trajectory_id_count"],
        "condition_id_count": identity_counts["condition_id_count"],
        "dataset_row_count": matrix_row_count,
        "raw_file_count": sum(int(item["raw_file_count"]) for item in families),
        "raw_bytes": sum(int(item["raw_bytes"]) for item in families),
    }
    if set(expected_counts) != set(derived_counts):
        raise TrajectoryHandoffError("BAGEN production count pins are incomplete")
    for key, expected_value in expected_counts.items():
        if key not in derived_counts or derived_counts[key] != expected_value:
            raise TrajectoryHandoffError(
                f"BAGEN derived combined count {key} disagrees with the production pin"
            )
    counts = _required_mapping(audit, "counts", label="BAGEN combined audit")
    actual_counts = {
        key: _required_int(counts, key, label="BAGEN combined audit.counts")
        for key in derived_counts
    }
    if actual_counts != derived_counts:
        raise TrajectoryHandoffError("BAGEN combined audit counts do not close")

    combined_dataset = _required_mapping(
        audit, "combined_dataset", label="BAGEN combined audit"
    )
    if (
        _required_sha(
            combined_dataset, "dataset_id", label="BAGEN combined audit.combined_dataset"
        )
        != expected_dataset_id
        or _required_int(
            combined_dataset, "row_count", label="BAGEN combined audit.combined_dataset"
        )
        != matrix_row_count
        or _required_int(
            combined_dataset,
            "schema_version",
            label="BAGEN combined audit.combined_dataset",
        )
        != SPEND_DATASET_SCHEMA_VERSION
        or _required_int(
            combined_dataset,
            "feature_schema_version",
            label="BAGEN combined audit.combined_dataset",
        )
        != SPEND_FEATURE_SCHEMA_VERSION
    ):
        raise TrajectoryHandoffError("BAGEN combined dataset identity/schema disagrees")
    if sum(matrix_status_counts.values()) != matrix_row_count:
        raise TrajectoryHandoffError("BAGEN combined matrix statuses do not close")

    family_index = sorted(
        [
            {
                "family": str(item["family"]),
                "canonical_content_sha256": str(item["canonical"]["aggregate_sha256"]),
            }
            for item in families
        ],
        key=lambda item: item["family"],
    )
    expected_family_index_sha = _semantic_sha256(family_index)
    if _required_list(
        audit, "canonical_family_index", label="BAGEN combined audit"
    ) != family_index:
        raise TrajectoryHandoffError("BAGEN canonical family index disagrees")
    if _required_sha(
        audit, "canonical_family_index_sha256", label="BAGEN combined audit"
    ) != expected_family_index_sha:
        raise TrajectoryHandoffError("BAGEN canonical family index SHA256 disagrees")
    trajectory_index = sorted(
        [
            {
                "family": str(family["family"]),
                "path": str(file_record["hub_relative_path"]),
                "canonical_content_sha256": str(
                    file_record["canonical_content_sha256"]
                ),
            }
            for family in families
            for file_record in family["trajectory_files"]
        ],
        key=lambda item: (item["family"], item["path"]),
    )
    expected_trajectory_index_sha = _semantic_sha256(trajectory_index)
    if _required_sha(
        audit, "canonical_trajectory_index_sha256", label="BAGEN combined audit"
    ) != expected_trajectory_index_sha:
        raise TrajectoryHandoffError("BAGEN canonical trajectory index SHA256 disagrees")

    raw_source_files = _required_mapping(
        audit, "source_files", label="BAGEN combined audit"
    )
    if set(raw_source_files) != set(BAGEN_SOURCE_PATHS) or not all(
        isinstance(value, Mapping) for value in raw_source_files.values()
    ):
        raise TrajectoryHandoffError("BAGEN combined source file roles are invalid")
    source_files = _freeze_code_artifacts(
        repo_root=repo_root,
        records=raw_source_files,  # type: ignore[arg-type]
        expected_roles={"reader", "builder", "labels", "audit", "family_audit"},
        label="BAGEN combined audit.source_files",
    )
    if (
        {role: record["path"] for role, record in source_files.items()}
        != BAGEN_SOURCE_PATHS
        or len({record["path"] for record in source_files.values()})
        != len(source_files)
    ):
        raise TrajectoryHandoffError("BAGEN combined source role paths disagree with pins")
    source_hashes = _required_mapping(
        audit, "source_hashes", label="BAGEN combined audit"
    )
    expected_source_hashes = {
        record["path"]: record["sha256"] for record in source_files.values()
    }
    actual_source_hashes = {
        _safe_relative(str(key), label="BAGEN combined source_hashes path"): _required_sha(
            source_hashes, key, label="BAGEN combined audit.source_hashes"
        )
        for key in source_hashes
    }
    if actual_source_hashes != expected_source_hashes:
        raise TrajectoryHandoffError("BAGEN combined source_hashes disagree")
    construction = _required_mapping(audit, "construction", label="BAGEN combined audit")
    expected_construction = {
        "command": "$env:PYTHONPATH='src'; python scripts/audit_bagen_combined.py",
        "reader": "BagenSwebenchReader",
        "dataset_builder": "build_supervised_dataset",
        "family_order": sorted(BAGEN_FAMILY_AUDITS),
        "output": "workspace/external/bagen/combined_swebench_audit.json",
    }
    if construction != expected_construction:
        raise TrajectoryHandoffError("BAGEN combined construction command disagrees")
    command = str(expected_construction["command"])

    family_distribution = Counter(
        int(task["family_count"]) for task in task_mapping
    )
    expected_family_distribution = {
        str(key): family_distribution[key] for key in sorted(family_distribution)
    }
    raw_family_distribution = _required_mapping(
        audit, "task_cross_family_distribution", label="BAGEN combined audit"
    )
    actual_family_distribution = {
        str(key): _required_int(
            raw_family_distribution,
            key,
            label="BAGEN combined audit.task_cross_family_distribution",
        )
        for key in raw_family_distribution
    }
    normalized_distribution_pin = {
        str(key): int(value)
        for key, value in expected_task_cross_family_distribution.items()
    }
    if (
        actual_family_distribution != expected_family_distribution
        or expected_family_distribution != normalized_distribution_pin
    ):
        raise TrajectoryHandoffError("BAGEN task cross-family distribution disagrees")

    expected_task_family_mapping = [
        {
            "task_id": str(task["task_id"]),
            "family_count": int(task["family_count"]),
            "families": sorted(
                str(trajectory["family"]) for trajectory in task["trajectories"]
            ),
            "trajectories": [
                {
                    "family": str(trajectory["family"]),
                    "run_id": str(trajectory["run_id"]),
                    "trajectory_id": str(trajectory["trajectory_id"]),
                    "condition_id": str(trajectory["condition_id"]),
                }
                for trajectory in task["trajectories"]
            ],
        }
        for task in task_mapping
    ]
    if _required_list(
        audit, "task_family_mapping", label="BAGEN combined audit"
    ) != expected_task_family_mapping:
        raise TrajectoryHandoffError("BAGEN combined task-family mapping disagrees")
    condition_counts = Counter(
        str(trajectory["condition_id"])
        for task in task_mapping
        for trajectory in task["trajectories"]
    )
    expected_condition_counts = dict(sorted(condition_counts.items()))
    if _required_mapping(
        audit, "condition_trajectory_counts", label="BAGEN combined audit"
    ) != expected_condition_counts:
        raise TrajectoryHandoffError("BAGEN combined condition counts disagree")
    return {
        "local_relative_path": _workspace_path(
            audit_path, workspace_root, label="BAGEN combined audit"
        ),
        "bytes": audit_path.stat().st_size,
        "sha256": _file_sha256(audit_path),
        "payload_sha256": payload_sha256,
        "schema_version": BAGEN_COMBINED_AUDIT_SCHEMA_VERSION,
        "counts": derived_counts,
        "dataset_id": expected_dataset_id,
        "dataset_row_count": matrix_row_count,
        "label_status_counts": {
            status: int(matrix_status_counts[status]) for status in STATUSES
        },
        "task_cross_family_distribution": expected_family_distribution,
        "canonical_family_index_sha256": expected_family_index_sha,
        "canonical_trajectory_index_sha256": expected_trajectory_index_sha,
        "source_files": source_files,
        "construction_command": command,
    }


def _validation_output(validation_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not validation_results:
        raise TrajectoryHandoffError("validation_results must not be empty")
    records: list[dict[str, str]] = []
    names: set[str] = set()
    for index, raw in enumerate(validation_results):
        label = f"validation_results[{index}]"
        name = _required_text(raw, "name", label=label)
        if name in names:
            raise TrajectoryHandoffError(f"duplicate validation result: {name}")
        names.add(name)
        command = _required_text(raw, "command", label=label)
        status = _required_text(raw, "status", label=label)
        result = _required_text(raw, "result", label=label)
        if status not in {"passed", "failed"}:
            raise TrajectoryHandoffError(f"{label}.status must be passed or failed")
        absolute_pattern = r"(?:[A-Za-z]:[\\/]|(?:^|\s)/(?:Users|home|tmp)/)"
        if re.search(absolute_pattern, command):
            raise TrajectoryHandoffError(f"{label}.command contains an absolute path")
        if re.search(absolute_pattern, result):
            raise TrajectoryHandoffError(f"{label}.result contains an absolute path")
        if len(command) > 2_000 or len(result) > 2_000:
            raise TrajectoryHandoffError(f"{label} command/result must be concise summaries")
        records.append(
            {"name": name, "command": command, "status": status, "result": result}
        )
    _assert_no_absolute_paths(records, location="validation_results")
    failed = sorted(record["name"] for record in records if record["status"] != "passed")
    if failed:
        raise TrajectoryHandoffError(
            "freeze validation gates failed: " + ", ".join(failed)
        )
    return {
        "all_passed": True,
        "results": sorted(records, key=lambda item: item["name"]),
    }


def _contains_absolute_local_path(value: str) -> bool:
    without_urls = re.sub(r"https?://[^\s]+", "", value)
    if re.search(r"(?i)(?<![A-Za-z0-9])[A-Z]:[\\/]", without_urls):
        return True
    if re.search(
        r"(?:^|[\s\"'(<=>:])(?:\\\\|//)[^\\/\s]+[\\/][^\\/\s]+",
        without_urls,
    ):
        return True
    if re.search(r"(?:^|[\s\"'(<=>:])\\(?!\\)[^\\\s]", without_urls):
        return True
    if re.search(r"(?:^|[\s\"'(<=>:])/(?!/)[^\s]*", without_urls):
        return True
    stripped = without_urls.strip(" \t\r\n\"'()[]{}<>,;:")
    if stripped and (
        PurePosixPath(stripped).is_absolute()
        or PureWindowsPath(stripped).is_absolute()
    ):
        return True
    return False


def _assert_no_absolute_paths(value: Any, *, location: str = "handoff") -> None:
    if isinstance(value, str):
        if _contains_absolute_local_path(value):
            raise TrajectoryHandoffError(f"{location} contains an absolute local path")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and _contains_absolute_local_path(key):
                raise TrajectoryHandoffError(
                    f"{location} contains an absolute local path in a mapping key"
                )
            _assert_no_absolute_paths(item, location=f"{location}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_absolute_paths(item, location=f"{location}[{index}]")


def _bagen_cell_capability(position: str, target: str, row_count: int) -> dict[str, str]:
    if (position, target) not in BUILDER_CELLS:
        return {
            "capability": "not_applicable",
            "eligibility": "not_applicable",
            "reason": "builder_does_not_emit_this_position_target_pair",
        }
    if position == "call_update" or target == "call_remaining_output_tokens":
        return {
            "capability": "unsupported",
            "eligibility": "gated",
            "reason": "no_generation_checkpoint_or_streaming_delta",
        }
    if target in {"task_unknown_remaining_tokens", "call_unknown_billable_tokens"}:
        return {
            "capability": "provider_input_proxy_only",
            "eligibility": "proxy_only_separate_experiment",
            "reason": "request_tokens_local_is_not_preserved; provider_input_tokens_is_a_proxy",
        }
    eligibility = "eligible_when_status_observed" if row_count > 0 else "no_rows_emitted"
    return {
        "capability": "exact_provider_usage",
        "eligibility": eligibility,
        "reason": "target_is_derived_from_recorded_provider_usage",
    }


def _bagen_reason_counts(
    position: str, target: str, counts: Mapping[str, int]
) -> tuple[dict[str, int], str]:
    reasons: Counter[str] = Counter()
    if counts["missing"]:
        reason = (
            "missing_task_usage"
            if position == "task_launch" and target == "task_total_accounted_tokens"
            else "missing_usage"
        )
        reasons[reason] += counts["missing"]
    if counts["censored"]:
        reasons["max_turns"] += counts["censored"]
    if counts["invalid"]:
        reasons["audit_did_not_retain_cell_reason"] += counts["invalid"]
    return (
        dict(sorted(reasons.items())),
        (
            "deterministic reader label contract (missing_usage/missing_task_usage/max_turns); "
            "the family audit stores status counts but not a per-cell reason table"
        ),
    )


def _spend_cell_capability(
    position: str,
    target: str,
    cell: Mapping[str, Any],
    *,
    request_tokens_local_available: bool,
    generation_checkpoint_available: bool,
) -> dict[str, str]:
    structural = cell.get("structurally_emitted_by_builder")
    if not isinstance(structural, bool):
        raise TrajectoryHandoffError(
            f"Spend label_matrix.{position}.{target} lacks structural eligibility"
        )
    if not structural:
        return {
            "capability": "not_applicable",
            "eligibility": "not_applicable",
            "reason": "builder_does_not_emit_this_position_target_pair",
        }
    if position == "call_update" or target == "call_remaining_output_tokens":
        return {
            "capability": "exact" if generation_checkpoint_available else "unsupported",
            "eligibility": (
                "eligible_when_status_observed" if generation_checkpoint_available else "gated"
            ),
            "reason": (
                "generation_checkpoint_observed"
                if generation_checkpoint_available
                else "no_generation_checkpoint_or_streaming_delta"
            ),
        }
    if target in {"task_unknown_remaining_tokens", "call_unknown_billable_tokens"}:
        return {
            "capability": "exact" if request_tokens_local_available else "unsupported",
            "eligibility": (
                "eligible_when_status_observed" if request_tokens_local_available else "gated"
            ),
            "reason": (
                "request_tokens_local_observed"
                if request_tokens_local_available
                else "request_tokens_local_not_preserved"
            ),
        }
    return {
        "capability": "exact_recorded_usage",
        "eligibility": "eligible_when_status_observed",
        "reason": "target_is_derived_from_recorded_usage_when_complete",
    }


def _build_bagen(
    *,
    manifest_path: Path,
    combined_audit_path: Path,
    audit_paths: Mapping[str, Path],
    workspace_root: Path,
    repo_root: Path,
    expected_combined_dataset_id: str,
    expected_combined_counts: Mapping[str, int],
    expected_task_cross_family_distribution: Mapping[str, int],
) -> dict[str, Any]:
    manifest = _load_json(manifest_path, label="BAGEN manifest summary")
    if _required_text(manifest, "resolved_revision", label="BAGEN manifest") != BAGEN_REVISION:
        raise TrajectoryHandoffError("BAGEN manifest revision disagrees with the frozen pin")
    source_url = _required_text(manifest, "source_url", label="BAGEN manifest")
    if source_url != f"https://huggingface.co/datasets/{BAGEN_REPO}":
        raise TrajectoryHandoffError("BAGEN manifest source URL disagrees with the frozen repository")
    manifest_sha = _required_sha(manifest, "manifest_sha256", label="BAGEN manifest")
    manifest_bytes = _required_int(manifest, "manifest_bytes", label="BAGEN manifest")
    raw_manifest_path = manifest_path.with_name(
        _required_text(manifest, "manifest_file", label="BAGEN manifest")
    )
    _verify_file(
        raw_manifest_path,
        expected_bytes=manifest_bytes,
        expected_sha256=manifest_sha,
        label="BAGEN raw manifest",
    )
    manifest_etag = _required_text(manifest, "manifest_etag", label="BAGEN manifest")
    if _git_blob_sha1(raw_manifest_path) != manifest_etag:
        raise TrajectoryHandoffError("BAGEN manifest Git blob ETag is invalid")
    manifest_index = _load_bagen_manifest_index(raw_manifest_path)
    if _required_int(manifest, "file_count", label="BAGEN manifest") != len(manifest_index):
        raise TrajectoryHandoffError("BAGEN manifest summary file count disagrees with JSONL")
    if _required_int(manifest, "total_bytes", label="BAGEN manifest") != sum(
        manifest_index.values()
    ):
        raise TrajectoryHandoffError("BAGEN manifest summary byte count disagrees with JSONL")
    manifest_trajectories = {
        path: size for path, size in manifest_index.items() if path.endswith(".traj.json")
    }
    if _required_int(manifest, "traj_json_count", label="BAGEN manifest") != len(
        manifest_trajectories
    ) or _required_int(manifest, "traj_json_bytes", label="BAGEN manifest") != sum(
        manifest_trajectories.values()
    ):
        raise TrajectoryHandoffError("BAGEN manifest trajectory totals disagree with JSONL")
    if set(audit_paths) != set(BAGEN_FAMILY_AUDITS):
        raise TrajectoryHandoffError("exactly the five pinned BAGEN family audits are required")
    if SHA256_RE.fullmatch(expected_combined_dataset_id) is None:
        raise TrajectoryHandoffError("BAGEN combined dataset ID must be a lowercase SHA256")

    bagen_root = Path(workspace_root) / "external" / "bagen"
    families: list[dict[str, Any]] = []
    all_task_records: dict[str, list[dict[str, str]]] = defaultdict(list)
    trajectory_ids: set[str] = set()
    condition_ids: set[str] = set()
    condition_trajectory_counts: Counter[str] = Counter()
    matrix_counts: dict[tuple[str, str], Counter[str]] = {
        (position, target): Counter() for position in POSITIONS for target in TARGETS
    }
    total_calls = 0
    total_attempts = 0
    complete_usage = 0
    missing_usage = 0
    retries = 0
    within_call_retries = 0
    tool_events = 0
    tool_failures = 0
    terminal_events: Counter[str] = Counter()
    exit_statuses: Counter[str] = Counter()
    selected_trajectory_hub_paths: set[str] = set()

    for family in sorted(BAGEN_FAMILY_AUDITS):
        audit_path = Path(audit_paths[family])
        audit = _load_json(audit_path, label=f"BAGEN {family} audit")
        if _required_text(audit, "family", label=f"BAGEN {family}") != family:
            raise TrajectoryHandoffError(f"BAGEN audit identity mismatch for {family}")
        if _required_int(
            audit, "audit_schema_version", label=f"BAGEN {family}"
        ) != 1:
            raise TrajectoryHandoffError(f"BAGEN audit schema version mismatch for {family}")
        if _required_text(audit, "source_id", label=f"BAGEN {family}") != (
            "bagen_swebench_traj_v1"
        ) or _required_text(audit, "reader_version", label=f"BAGEN {family}") != (
            "bagen_swebench_traj_v1"
        ):
            raise TrajectoryHandoffError(f"BAGEN source_id mismatch for {family}")
        if not _required_bool(
            audit, "canonical_rerun_consistent", label=f"BAGEN {family}"
        ):
            raise TrajectoryHandoffError(f"BAGEN canonical rerun is inconsistent for {family}")
        family_root = _safe_id(
            _required_text(audit, "family_root", label=f"BAGEN {family}"),
            label=f"BAGEN {family}.family_root",
        )
        if family_root != BAGEN_FAMILY_ROOTS[family]:
            raise TrajectoryHandoffError(f"BAGEN family root mismatch for {family}")
        local_root = bagen_root / "origin" / family_root
        if not local_root.is_dir():
            raise TrajectoryHandoffError(f"BAGEN family root is missing for {family}")
        raw_files = _required_list(audit, "raw_files", label=f"BAGEN {family}")
        output_files: list[dict[str, Any]] = []
        family_tasks: set[str] = set()
        family_trajectories: set[str] = set()
        family_conditions: set[str] = set()
        seen_paths: set[str] = set()
        canonical_hashes: dict[str, str] = {}
        raw_source_hashes: dict[str, str] = {}
        for index, raw_value in enumerate(raw_files):
            if not isinstance(raw_value, Mapping):
                raise TrajectoryHandoffError(f"BAGEN {family}.raw_files[{index}] must be an object")
            raw_label = f"BAGEN {family}.raw_files[{index}]"
            path = _safe_relative(_required_text(raw_value, "path", label=raw_label), label=raw_label)
            if not path.endswith(".traj.json"):
                raise TrajectoryHandoffError(f"{raw_label} is not a *.traj.json member")
            if path in seen_paths:
                raise TrajectoryHandoffError(f"BAGEN {family} repeats raw path {path}")
            seen_paths.add(path)
            size_bytes = _required_int(raw_value, "bytes", label=raw_label)
            raw_sha = _required_sha(raw_value, "sha256", label=raw_label)
            canonical_sha = _required_sha(
                raw_value, "canonical_content_sha256", label=raw_label
            )
            if not _required_bool(
                raw_value, "canonical_rerun_consistent", label=raw_label
            ):
                raise TrajectoryHandoffError(
                    f"BAGEN per-file canonical rerun is inconsistent for {family}/{path}"
                )
            canonical_hashes[path] = canonical_sha
            raw_source_hashes[path] = raw_sha
            task_id = _safe_id(raw_value.get("task_id"), label=f"{raw_label}.task_id")
            trajectory_id = _safe_id(
                raw_value.get("trajectory_id"), label=f"{raw_label}.trajectory_id"
            )
            condition_id = _safe_id(
                raw_value.get("condition_id"), label=f"{raw_label}.condition_id"
            )
            local_file = local_root.joinpath(*PurePosixPath(path).parts)
            _verify_file(
                local_file,
                expected_bytes=size_bytes,
                expected_sha256=raw_sha,
                label=f"BAGEN raw file {family}/{path}",
            )
            if trajectory_id in trajectory_ids:
                raise TrajectoryHandoffError(f"duplicate BAGEN trajectory_id: {trajectory_id}")
            trajectory_ids.add(trajectory_id)
            family_tasks.add(task_id)
            family_trajectories.add(trajectory_id)
            family_conditions.add(condition_id)
            condition_ids.add(condition_id)
            condition_trajectory_counts[condition_id] += 1
            hub_relative = f"origin/{family_root}/{path}"
            if manifest_index.get(hub_relative) != size_bytes:
                raise TrajectoryHandoffError(
                    f"BAGEN manifest path/size mismatch for {hub_relative}"
                )
            if hub_relative in selected_trajectory_hub_paths:
                raise TrajectoryHandoffError(
                    f"BAGEN selected trajectory Hub path is duplicated: {hub_relative}"
                )
            selected_trajectory_hub_paths.add(hub_relative)
            local_relative = f"workspace/external/bagen/{hub_relative}"
            run_id = f"{family_root}/{path}"
            record = {
                "relative_path": path,
                "local_relative_path": local_relative,
                "hub_relative_path": hub_relative,
                "bytes": size_bytes,
                "sha256": raw_sha,
                "task_id": task_id,
                "run_id": run_id,
                "trajectory_id": trajectory_id,
                "condition_id": condition_id,
                "canonical_content_sha256": canonical_sha,
            }
            output_files.append(record)
            all_task_records[task_id].append(
                {
                    "family": family,
                    "run_id": run_id,
                    "trajectory_id": trajectory_id,
                    "condition_id": condition_id,
                    "raw_file": local_relative,
                }
            )

        declared_raw_count = _required_int(audit, "raw_file_count", label=f"BAGEN {family}")
        declared_raw_bytes = _required_int(audit, "raw_bytes", label=f"BAGEN {family}")
        if declared_raw_count != len(output_files) or declared_raw_bytes != sum(
            item["bytes"] for item in output_files
        ):
            raise TrajectoryHandoffError(f"BAGEN raw-file totals disagree for {family}")
        declared_counts = {
            "task_id_count": _required_int(audit, "task_count", label=f"BAGEN {family}"),
            "trajectory_id_count": _required_int(
                audit, "trajectory_count", label=f"BAGEN {family}"
            ),
            "condition_id_count": _required_int(
                audit, "condition_count", label=f"BAGEN {family}"
            ),
        }
        actual_counts = {
            "task_id_count": len(family_tasks),
            "trajectory_id_count": len(family_trajectories),
            "condition_id_count": len(family_conditions),
        }
        if declared_counts != actual_counts:
            raise TrajectoryHandoffError(f"BAGEN identity totals disagree for {family}")
        declared_source_hashes = _required_mapping(
            audit, "source_hashes", label=f"BAGEN {family}"
        )
        if {
            _safe_relative(str(path), label=f"BAGEN {family}.source_hashes path"): (
                _required_sha(
                    declared_source_hashes,
                    path,
                    label=f"BAGEN {family}.source_hashes",
                )
            )
            for path in declared_source_hashes
        } != raw_source_hashes:
            raise TrajectoryHandoffError(f"BAGEN source hashes disagree for {family}")
        canonical_family_sha = _bagen_canonical_family_sha256(canonical_hashes)
        if (
            _required_sha(audit, "canonical_content_sha256", label=f"BAGEN {family}")
            != canonical_family_sha
            or _required_sha(
                audit, "canonical_rerun_content_sha256", label=f"BAGEN {family}"
            )
            != canonical_family_sha
        ):
            raise TrajectoryHandoffError(
                f"BAGEN canonical family/rerun hashes disagree for {family}"
            )

        dataset = _required_mapping(audit, "dataset", label=f"BAGEN {family}")
        cells = _required_list(dataset, "by_position_target", label=f"BAGEN {family}.dataset")
        seen_cells: set[tuple[str, str]] = set()
        family_matrix_status: Counter[str] = Counter()
        family_matrix_row_count = 0
        for cell_index, raw_cell in enumerate(cells):
            if not isinstance(raw_cell, Mapping):
                raise TrajectoryHandoffError(
                    f"BAGEN {family}.dataset.by_position_target[{cell_index}] must be an object"
                )
            cell_label = f"BAGEN {family}.dataset.by_position_target[{cell_index}]"
            position = _required_text(raw_cell, "position", label=cell_label)
            target = _required_text(raw_cell, "target", label=cell_label)
            key = (position, target)
            if key not in matrix_counts or key in seen_cells:
                raise TrajectoryHandoffError(f"BAGEN {family} has unknown or repeated matrix cell")
            seen_cells.add(key)
            counts = _status_counts(
                _required_mapping(raw_cell, "status_counts", label=cell_label), label=cell_label
            )
            row_count = _required_int(raw_cell, "row_count", label=cell_label)
            if row_count != sum(counts.values()):
                raise TrajectoryHandoffError(f"BAGEN {family} matrix row count does not close")
            if key not in BUILDER_CELLS and row_count:
                raise TrajectoryHandoffError(
                    f"BAGEN {family} non-structural matrix cell contains rows"
                )
            matrix_counts[key].update(counts)
            family_matrix_status.update(counts)
            family_matrix_row_count += row_count
        if seen_cells != set(matrix_counts):
            raise TrajectoryHandoffError(f"BAGEN {family} matrix is incomplete")
        family_dataset_row_count = _required_int(
            dataset, "row_count", label=f"BAGEN {family}.dataset"
        )
        family_dataset_status = _status_counts(
            _required_mapping(dataset, "status_counts", label=f"BAGEN {family}.dataset"),
            label=f"BAGEN {family}.dataset.status_counts",
        )
        if (
            family_dataset_row_count != family_matrix_row_count
            or family_dataset_row_count != sum(family_dataset_status.values())
            or family_dataset_status
            != {status: family_matrix_status[status] for status in STATUSES}
        ):
            raise TrajectoryHandoffError(
                f"BAGEN {family} dataset/matrix status totals do not close"
            )

        family_calls = _required_int(audit, "call_count", label=f"BAGEN {family}")
        family_attempts = _required_int(audit, "attempt_count", label=f"BAGEN {family}")
        family_complete_usage = _required_int(
            audit, "complete_usage_attempts", label=f"BAGEN {family}"
        )
        family_missing_usage = _required_int(
            audit, "missing_usage_attempts", label=f"BAGEN {family}"
        )
        family_retries = _required_int(audit, "retry_count", label=f"BAGEN {family}")
        family_within_call_retries = _required_int(
            audit, "within_call_retry_count", label=f"BAGEN {family}"
        )
        family_tool_events = _required_int(
            audit, "tool_event_count", label=f"BAGEN {family}"
        )
        family_tool_failures = _required_int(
            audit, "tool_failure_count", label=f"BAGEN {family}"
        )
        if (
            family_calls != family_attempts
            or family_complete_usage + family_missing_usage != family_attempts
            or family_retries > family_attempts
            or family_within_call_retries > family_retries
            or family_tool_failures > family_tool_events
        ):
            raise TrajectoryHandoffError(f"BAGEN telemetry counts do not close for {family}")
        total_calls += family_calls
        total_attempts += family_attempts
        complete_usage += family_complete_usage
        missing_usage += family_missing_usage
        retries += family_retries
        within_call_retries += family_within_call_retries
        tool_events += family_tool_events
        tool_failures += family_tool_failures
        distributions = _required_mapping(audit, "distributions", label=f"BAGEN {family}")
        terminal_raw = _required_mapping(
            distributions, "task_terminal_event", label=f"BAGEN {family}.distributions"
        )
        exit_raw = _required_mapping(
            distributions, "exit_status", label=f"BAGEN {family}.distributions"
        )
        if sum(
            _required_int(terminal_raw, key, label="task_terminal_event")
            for key in terminal_raw
        ) != len(family_trajectories) or sum(
            _required_int(exit_raw, key, label="exit_status") for key in exit_raw
        ) != len(family_trajectories):
            raise TrajectoryHandoffError(
                f"BAGEN task termination counts do not close for {family}"
            )
        terminal_events.update(
            {
                _safe_id(key, label="task_terminal_event"): _required_int(
                    terminal_raw, key, label="task_terminal_event"
                )
                for key in terminal_raw
            }
        )
        exit_statuses.update(
            {
                _safe_id(key, label="exit_status"): _required_int(
                    exit_raw, key, label="exit_status"
                )
                for key in exit_raw
            }
        )

        families.append(
            {
                "family": family,
                "family_root": family_root,
                "local_relative_root": f"workspace/external/bagen/origin/{family_root}",
                "hub_relative_root": f"origin/{family_root}",
                "counts": actual_counts,
                "raw_file_count": len(output_files),
                "raw_bytes": sum(item["bytes"] for item in output_files),
                "trajectory_files": sorted(output_files, key=lambda item: item["relative_path"]),
                "canonical": {
                    "aggregate_sha256": canonical_family_sha,
                    "rerun_aggregate_sha256": canonical_family_sha,
                    "rerun_consistent": True,
                },
                "dataset": {
                    "dataset_id": _required_sha(dataset, "dataset_id", label=f"BAGEN {family}.dataset"),
                    "row_count": family_dataset_row_count,
                    "schema_version": SPEND_DATASET_SCHEMA_VERSION,
                    "feature_schema_version": SPEND_FEATURE_SCHEMA_VERSION,
                    "status_counts": family_dataset_status,
                },
                "audit": {
                    "local_relative_path": _workspace_path(
                        audit_path, workspace_root, label=f"BAGEN {family} audit"
                    ),
                    "bytes": audit_path.stat().st_size,
                    "sha256": _file_sha256(audit_path),
                    "schema_version": 1,
                    "rebuild_command": (
                        "$env:PYTHONPATH='src'; python scripts/audit_bagen_swebench.py "
                        f"workspace/external/bagen/origin/{family_root} "
                        f"workspace/external/bagen/audits/{BAGEN_FAMILY_AUDITS[family]}"
                    ),
                },
            }
        )

    task_mapping = []
    for task_id in sorted(all_task_records):
        trajectories = sorted(all_task_records[task_id], key=lambda item: item["family"])
        if len({item["family"] for item in trajectories}) != len(trajectories):
            raise TrajectoryHandoffError(f"BAGEN task repeats a family mapping: {task_id}")
        task_mapping.append(
            {
                "task_id": task_id,
                "family_count": len(trajectories),
                "trajectories": trajectories,
            }
        )

    gpt_family = next(item for item in families if item["family"] == "gpt5.2instant")
    gpt_root = bagen_root / "origin" / gpt_family["family_root"]
    aux_paths = sorted(
        path for path in gpt_root.rglob("*") if path.is_file() and not path.name.endswith(".traj.json")
    )
    if len(aux_paths) != 5:
        raise TrajectoryHandoffError(
            f"BAGEN GPT-5.2 Instant auxiliary slice must contain exactly 5 files, got {len(aux_paths)}"
        )
    gpt_auxiliary_files = []
    selected_auxiliary_hub_paths: set[str] = set()
    for path in aux_paths:
        relative = path.relative_to(gpt_root).as_posix()
        hub_relative = f"origin/{gpt_family['family_root']}/{relative}"
        if manifest_index.get(hub_relative) != path.stat().st_size:
            raise TrajectoryHandoffError(
                f"BAGEN manifest path/size mismatch for GPT auxiliary {hub_relative}"
            )
        selected_auxiliary_hub_paths.add(hub_relative)
        gpt_auxiliary_files.append(
            {
                "relative_path": _safe_relative(relative, label="BAGEN GPT auxiliary path"),
                "local_relative_path": (
                    f"workspace/external/bagen/origin/{gpt_family['family_root']}/{relative}"
                ),
                "hub_relative_path": hub_relative,
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    gpt_family["auxiliary_files"] = gpt_auxiliary_files

    family_prefixes = tuple(
        f"origin/{item['family_root']}/" for item in families
    )
    manifest_selected_trajectories = {
        path
        for path in manifest_trajectories
        if path.startswith(family_prefixes)
    }
    if manifest_selected_trajectories != selected_trajectory_hub_paths:
        raise TrajectoryHandoffError(
            "BAGEN manifest trajectory members do not exactly match the five audited families"
        )
    if len(selected_trajectory_hub_paths) != sum(
        item["raw_file_count"] for item in families
    ):
        raise TrajectoryHandoffError("BAGEN selected trajectory Hub path count does not close")
    gpt_manifest_paths = {
        path
        for path in manifest_index
        if path.startswith(f"origin/{gpt_family['family_root']}/")
    }
    if gpt_manifest_paths != selected_auxiliary_hub_paths | {
        path
        for path in selected_trajectory_hub_paths
        if path.startswith(f"origin/{gpt_family['family_root']}/")
    }:
        raise TrajectoryHandoffError(
            "BAGEN GPT-5.2 Instant manifest slice is not exactly 64 trajectories plus 5 auxiliaries"
        )

    matrix: dict[str, dict[str, Any]] = {}
    aggregate_status_counts: Counter[str] = Counter()
    for position in POSITIONS:
        matrix[position] = {}
        for target in TARGETS:
            counts = {status: matrix_counts[(position, target)][status] for status in STATUSES}
            row_count = sum(counts.values())
            reason_counts, reason_provenance = _bagen_reason_counts(position, target, counts)
            aggregate_status_counts.update(counts)
            matrix[position][target] = {
                "row_count": row_count,
                "status_counts": counts,
                "reason_counts": reason_counts,
                "reason_count_provenance": reason_provenance,
                **_bagen_cell_capability(position, target, row_count),
            }

    family_hash_index = [
        {
            "family": item["family"],
            "canonical_aggregate_sha256": item["canonical"]["aggregate_sha256"],
            "dataset_id": item["dataset"]["dataset_id"],
        }
        for item in families
    ]
    combined_row_count = sum(aggregate_status_counts.values())
    if combined_row_count != sum(int(item["dataset"]["row_count"]) for item in families):
        raise TrajectoryHandoffError("BAGEN combined family/matrix row totals do not close")
    identity_counts = {
        "task_id_count": len(all_task_records),
        "run_id_count": len(trajectory_ids),
        "trajectory_id_count": len(trajectory_ids),
        "condition_id_count": len(condition_ids),
    }
    combined_audit = _validate_bagen_combined_audit(
        path=combined_audit_path,
        repo_root=repo_root,
        workspace_root=workspace_root,
        manifest_summary_path=manifest_path,
        raw_manifest_path=raw_manifest_path,
        manifest_etag=manifest_etag,
        manifest_counts={
            "file_count": len(manifest_index),
            "total_bytes": sum(manifest_index.values()),
            "traj_json_count": len(manifest_trajectories),
            "traj_json_bytes": sum(manifest_trajectories.values()),
        },
        families=families,
        task_mapping=task_mapping,
        identity_counts=identity_counts,
        matrix_row_count=combined_row_count,
        matrix_status_counts=aggregate_status_counts,
        expected_dataset_id=expected_combined_dataset_id,
        expected_counts=expected_combined_counts,
        expected_task_cross_family_distribution=(
            expected_task_cross_family_distribution
        ),
    )
    return {
        "source_id": "bagen_swebench_traj_v1",
        "schema_pins": {
            "family_audit_schema_version": 1,
            "combined_audit_schema_version": BAGEN_COMBINED_AUDIT_SCHEMA_VERSION,
            "dataset_schema_version": SPEND_DATASET_SCHEMA_VERSION,
            "feature_schema_version": SPEND_FEATURE_SCHEMA_VERSION,
        },
        "hub": {
            "repo_id": BAGEN_REPO,
            "repo_type": "dataset",
            "resolved_revision": BAGEN_REVISION,
            "url": f"https://huggingface.co/datasets/{BAGEN_REPO}",
            "pinned_tree_url": (
                f"https://huggingface.co/datasets/{BAGEN_REPO}/tree/{BAGEN_REVISION}"
            ),
        },
        "manifest": {
            "local_relative_path": _workspace_path(
                raw_manifest_path,
                workspace_root,
                label="BAGEN manifest",
            ),
            "summary_local_relative_path": _workspace_path(
                manifest_path, workspace_root, label="BAGEN manifest summary"
            ),
            "bytes": manifest_bytes,
            "sha256": manifest_sha,
            "git_blob_etag": manifest_etag,
            "pinned_url": (
                f"https://huggingface.co/datasets/{BAGEN_REPO}/resolve/"
                f"{BAGEN_REVISION}/manifest.jsonl"
            ),
            "summary_bytes": manifest_path.stat().st_size,
            "summary_sha256": _file_sha256(manifest_path),
        },
        "combined_audit": combined_audit,
        "families": families,
        "identity": {
            **identity_counts,
            "condition_trajectory_counts": dict(sorted(condition_trajectory_counts.items())),
            "run_id_semantics": (
                "one source-relative *.traj.json path per trajectory; family is the "
                "cross-condition grouping axis"
            ),
            "task_cross_family_mapping": task_mapping,
        },
        "telemetry_capability": {
            "request_tokens_local": {
                "available_exact": False,
                "proxy_available": complete_usage > 0,
                "proxy": "provider_input_tokens",
                "proxy_observed_count": complete_usage,
                "proxy_missing_count": missing_usage,
                "eligibility": "proxy_only_separate_experiment",
            },
            "attempt_usage": {
                "available": complete_usage > 0,
                "scope": "logical_call_attempt",
                "complete_count": complete_usage,
                "missing_count": missing_usage,
                "invalid_count": 0,
            },
            "retry": {
                "provider_transport_ledger_available": False,
                "within_call_retry_count": within_call_retries,
                "format_error_recovery_count": retries,
                "format_error_recovery_is_provider_retry": False,
            },
            "tool_events": {
                "available": tool_events > 0,
                "terminal_event_count": tool_events,
                "failure_observed_count": tool_failures,
                "start_timestamp_available": False,
            },
            "errors": {
                "format_error_recovery_available": retries > 0,
                "format_error_recovery_count": retries,
                "provider_transport_error_ledger_available": False,
            },
            "task_termination": {
                "available": sum(terminal_events.values()) == len(trajectory_ids),
                "event_counts": dict(sorted(terminal_events.items())),
                "exit_status_counts": dict(sorted(exit_statuses.items())),
                "limits_exceeded_is_censored": True,
            },
            "generation_checkpoint": {
                "available": False,
                "observed_count": 0,
                "reason": "no_streaming_generation_deltas_or_checkpoints",
            },
            "logical_call_count": total_calls,
            "attempt_count": total_attempts,
        },
        "position_target_matrix": matrix,
        "label_status_totals": {
            status: aggregate_status_counts[status] for status in STATUSES
        },
        "canonical": {
            "per_trajectory_hashes_in_family_files": True,
            "family_index": family_hash_index,
            "family_index_sha256": _semantic_sha256(family_hash_index),
        },
        "dataset": {
            "combined_dataset_id": expected_combined_dataset_id,
            "row_count": combined_row_count,
            "schema_version": SPEND_DATASET_SCHEMA_VERSION,
            "feature_schema_version": SPEND_FEATURE_SCHEMA_VERSION,
            "family_dataset_ids": {
                item["family"]: item["dataset"]["dataset_id"] for item in families
            },
            "construction_command": (
                "$env:PYTHONPATH='src'; python scripts/audit_bagen_swebench.py "
                "workspace/external/bagen/origin/<family-root> "
                "workspace/external/bagen/audits/<family>.json"
            ),
            "family_construction_commands": {
                item["family"]: item["audit"]["rebuild_command"] for item in families
            },
            "combined_dataset_id_provenance": (
                "required combined_swebench_audit payload tied to all five family audit SHA256s"
            ),
        },
    }


def _inventory_jsonl_member(
    raw: Mapping[str, Any],
    *,
    wrapper: str,
    run_basename: str,
    expected_filename: str,
    label: str,
) -> dict[str, Any]:
    filename = _safe_segment(
        _required_text(raw, "filename", label=label), label=f"{label}.filename"
    )
    if filename != expected_filename:
        raise TrajectoryHandoffError(f"{label}.filename disagrees with the pinned member name")
    files = _required_list(raw, "files", label=label)
    if len(files) != 1 or not isinstance(files[0], Mapping):
        raise TrajectoryHandoffError(f"{label} must point to exactly one JSONL member")
    file_record = files[0]
    member_path = f"{wrapper}/{run_basename}/{filename}"
    member_path_hash = hashlib.sha256(member_path.encode("utf-8")).hexdigest()
    frozen_path_hash = _required_sha(file_record, "member_path_sha256", label=f"{label}.files[0]")
    if frozen_path_hash != member_path_hash:
        raise TrajectoryHandoffError(f"{label} reconstructed member path hash disagrees")
    return {
        "archive_internal_path": member_path,
        "bytes": _required_int(file_record, "bytes", label=f"{label}.files[0]"),
        "sha256": _required_sha(file_record, "sha256", label=f"{label}.files[0]"),
        "record_count": _required_int(
            file_record, "record_count", label=f"{label}.files[0]"
        ),
        "member_path_sha256": frozen_path_hash,
    }


def _validate_spend_source_capability(
    raw: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    trajectory_count: int,
) -> dict[str, Any]:
    if _required_int(
        metrics, "trajectory_count", label="Spend trajectory audit.metrics"
    ) != trajectory_count:
        raise TrajectoryHandoffError("Spend capability metrics trajectory count disagrees")
    expected_top_level = {
        "source_id",
        "declared_observables",
        *SPEND_CAPABILITY_FIELDS,
    }
    if set(raw) != expected_top_level:
        raise TrajectoryHandoffError(
            "Spend source_capability fields disagree with the pinned reader schema"
        )
    source_id = _required_text(raw, "source_id", label="Spend source_capability")
    if source_id != SPEND_READER_SOURCE_ID:
        raise TrajectoryHandoffError("Spend trajectory reader source_id disagrees with pin")
    declared = _required_list(raw, "declared_observables", label="Spend source_capability")
    if declared != list(SPEND_DECLARED_OBSERVABLES):
        raise TrajectoryHandoffError("Spend declared observables disagree with the reader pin")

    sections: dict[str, Mapping[str, Any]] = {}
    for section_name, expected_fields in SPEND_CAPABILITY_FIELDS.items():
        section = _required_mapping(
            raw, section_name, label="Spend source_capability"
        )
        if set(section) != expected_fields:
            raise TrajectoryHandoffError(
                f"Spend source_capability.{section_name} fields disagree with the pin"
            )
        sections[section_name] = section
        for field, metric_name in SPEND_CAPABILITY_METRICS.get(
            section_name, {}
        ).items():
            if _required_int(
                section,
                field,
                label=f"Spend source_capability.{section_name}",
            ) != _required_int(metrics, metric_name, label="Spend trajectory audit.metrics"):
                raise TrajectoryHandoffError(
                    f"Spend source_capability.{section_name}.{field} disagrees with metrics"
                )

    request = sections["request_tokens_local"]
    request_observed = _required_int(
        request, "observed_count", label="Spend source_capability.request_tokens_local"
    )
    request_missing = _required_int(
        request, "missing_count", label="Spend source_capability.request_tokens_local"
    )
    if (
        _required_bool(
            request, "available", label="Spend source_capability.request_tokens_local"
        )
        != (request_observed > 0)
        or request_observed + request_missing
        != _required_int(metrics, "request_count", label="Spend trajectory audit.metrics")
        or request.get("gates_targets")
        != ["task_unknown_remaining_tokens", "call_unknown_billable_tokens"]
        or request.get("reason")
        != (
            "no_local_tokenizer_count_in_archive"
            if request_observed == 0
            else "partially_observed"
        )
    ):
        raise TrajectoryHandoffError("Spend request_tokens_local capability does not close")

    attempt = sections["attempt_usage"]
    attempt_counts = [
        _required_int(attempt, field, label="Spend source_capability.attempt_usage")
        for field in ("complete_count", "missing_count", "invalid_count")
    ]
    if (
        _required_bool(
            attempt, "available", label="Spend source_capability.attempt_usage"
        )
        != (attempt_counts[0] > 0)
        or sum(attempt_counts)
        != _required_int(metrics, "attempt_count", label="Spend trajectory audit.metrics")
        or attempt.get("scope") != "current_response_only"
    ):
        raise TrajectoryHandoffError("Spend attempt_usage capability does not close")

    task_usage = sections["task_usage"]
    task_usage_counts = [
        _required_int(task_usage, field, label="Spend source_capability.task_usage")
        for field in ("complete_count", "missing_count", "invalid_count")
    ]
    explicit_zero_call_count = _required_int(
        task_usage,
        "explicit_zero_call_count",
        label="Spend source_capability.task_usage",
    )
    all_preserved_sessions_count = _required_int(
        task_usage,
        "all_preserved_sessions_count",
        label="Spend source_capability.task_usage",
    )
    if (
        _required_bool(
            task_usage, "available", label="Spend source_capability.task_usage"
        )
        != (task_usage_counts[0] > 0)
        or sum(task_usage_counts) != trajectory_count
        or task_usage.get("scope")
        != (
            "output.metrics current-session aggregate plus complete preserved "
            "completion extras, plus source-reported explicit zero-call usage; "
            "never backfilled into attempt events"
        )
        or task_usage.get("explicit_zero_call_source")
        != "output.metrics.accumulated_token_usage"
        or task_usage.get("explicit_zero_call_never_imputed") is not True
        or task_usage.get("explicit_zero_call_criteria")
        != (
            "usage_scope=explicit_zero_call_task, accounted and reported total "
            "tokens both zero, completion_snapshot_count=0"
        )
        or explicit_zero_call_count > task_usage_counts[0]
        or explicit_zero_call_count > all_preserved_sessions_count
    ):
        raise TrajectoryHandoffError("Spend task_usage capability does not close")

    retry = sections["retry"]
    if (
        _required_bool(retry, "supported", label="Spend source_capability.retry")
        or retry.get("retry_count") is not None
        or retry.get("reason") != "provider_transport_retry_ledger_not_preserved"
    ):
        raise TrajectoryHandoffError("Spend retry capability disagrees with the reader pin")

    tools = sections["tool_events"]
    tool_counts = {
        field: _required_int(tools, field, label="Spend source_capability.tool_events")
        for field in (
            "started_count",
            "completed_count",
            "failed_count",
            "failure_observable_count",
            "failure_unobservable_count",
        )
    }
    if (
        _required_bool(tools, "available", label="Spend source_capability.tool_events")
        != (sum(tool_counts.values()) > 0)
        or tool_counts["completed_count"] + tool_counts["failed_count"]
        != tool_counts["started_count"]
        or tool_counts["failure_observable_count"]
        + tool_counts["failure_unobservable_count"]
        != tool_counts["started_count"]
        or tools.get("failure_status_scope") != "explicit_output_jsonl_only"
    ):
        raise TrajectoryHandoffError("Spend tool_events capability does not close")

    errors = sections["errors"]
    for prefix in ("task_error", "attempt_error", "provider_error_envelope"):
        count = _required_int(
            errors, f"{prefix}_count", label="Spend source_capability.errors"
        )
        if _required_bool(
            errors, f"{prefix}_available", label="Spend source_capability.errors"
        ) != (count > 0):
            raise TrajectoryHandoffError(
                f"Spend errors.{prefix} availability/count disagrees"
            )
    if (
        errors.get("provider_error_envelope_semantics")
        != (
            "preserved on a completed response; not classified as API_FAILED "
            "or a transport retry"
        )
        or errors.get("reason")
        != "task_errors_attempt_failures_and_provider_envelopes_are_distinct"
    ):
        raise TrajectoryHandoffError("Spend error capability semantics disagree with pin")

    termination = sections["task_termination"]
    lifecycle_observed = _required_int(
        termination,
        "observed_lifecycle_count",
        label="Spend source_capability.task_termination",
    )
    lifecycle_censored = _required_int(
        termination,
        "censored_lifecycle_count",
        label="Spend source_capability.task_termination",
    )
    finished = _required_int(
        termination, "finished_count", label="Spend source_capability.task_termination"
    )
    aborted = _required_int(
        termination, "aborted_count", label="Spend source_capability.task_termination"
    )
    task_log_observed = _required_int(
        termination,
        "task_log_observed_count",
        label="Spend source_capability.task_termination",
    )
    task_log_missing = _required_int(
        termination,
        "task_log_missing_count",
        label="Spend source_capability.task_termination",
    )
    if (
        _required_bool(
            termination, "available", label="Spend source_capability.task_termination"
        )
        != (lifecycle_observed > 0)
        or finished + aborted != lifecycle_observed
        or lifecycle_observed + lifecycle_censored != trajectory_count
        or task_log_observed + task_log_missing != trajectory_count
        or termination.get("source")
        != "output.jsonl_when_present_else_censored_logging_incomplete"
    ):
        raise TrajectoryHandoffError("Spend task_termination capability does not close")

    generation = sections["generation_checkpoint"]
    checkpoint_count = _required_int(
        generation,
        "observed_count",
        label="Spend source_capability.generation_checkpoint",
    )
    if (
        _required_bool(
            generation,
            "available",
            label="Spend source_capability.generation_checkpoint",
        )
        != (checkpoint_count > 0)
        or generation.get("gates_position") != "call_update"
        or generation.get("reason")
        != (
            "no_streaming_generation_deltas_or_checkpoints_in_archive"
            if checkpoint_count == 0
            else "observed"
        )
    ):
        raise TrajectoryHandoffError("Spend generation_checkpoint capability does not close")

    reconciliation = sections["session_reconciliation"]
    reconciliation_pairs = (
        ("completion_logging_complete_count", "completion_logging_incomplete_count"),
        ("task_usage_reconciled_count", "task_usage_unreconciled_count"),
        (
            "history_llm_metrics_ledger_match_count",
            "history_llm_metrics_ledger_mismatch_count",
        ),
    )
    for complete_field, incomplete_field in reconciliation_pairs:
        if (
            _required_int(
                reconciliation,
                complete_field,
                label="Spend source_capability.session_reconciliation",
            )
            + _required_int(
                reconciliation,
                incomplete_field,
                label="Spend source_capability.session_reconciliation",
            )
            != trajectory_count
        ):
            raise TrajectoryHandoffError(
                "Spend session_reconciliation trajectory counts do not close"
            )

    safe: dict[str, Any] = {
        "source_id": source_id,
        "declared_observables": list(SPEND_DECLARED_OBSERVABLES),
    }
    for section_name in sorted(SPEND_CAPABILITY_FIELDS):
        section = sections[section_name]
        safe[section_name] = {
            field: section[field] for field in sorted(SPEND_CAPABILITY_FIELDS[section_name])
        }
    return safe


def _normalize_spend_matrix(
    raw_matrix: Mapping[str, Any],
    source_capability: Mapping[str, Any],
    *,
    matrix_label: str,
) -> dict[str, Any]:
    request = _required_mapping(
        source_capability, "request_tokens_local", label="Spend source_capability"
    )
    generation = _required_mapping(
        source_capability, "generation_checkpoint", label="Spend source_capability"
    )
    request_available = _required_bool(
        request, "available", label="Spend source_capability.request_tokens_local"
    )
    generation_available = _required_bool(
        generation, "available", label="Spend source_capability.generation_checkpoint"
    )
    matrix: dict[str, Any] = {}
    if set(raw_matrix) != set(POSITIONS):
        raise TrajectoryHandoffError(
            f"{matrix_label} does not contain every prediction position"
        )
    for position in POSITIONS:
        raw_targets = raw_matrix.get(position)
        if not isinstance(raw_targets, Mapping) or set(raw_targets) != set(TARGETS):
            raise TrajectoryHandoffError(f"{matrix_label}.{position} is incomplete")
        matrix[position] = {}
        for target in TARGETS:
            raw_cell = raw_targets[target]
            if not isinstance(raw_cell, Mapping):
                raise TrajectoryHandoffError(
                    f"{matrix_label}.{position}.{target} must be an object"
                )
            label = f"{matrix_label}.{position}.{target}"
            status_counts = _status_counts(
                _required_mapping(raw_cell, "status_counts", label=label), label=label
            )
            row_count = _required_int(raw_cell, "row_count", label=label)
            if row_count != sum(status_counts.values()):
                raise TrajectoryHandoffError(f"{label} row count does not close")
            reason_counts_raw = _required_mapping(raw_cell, "reason_counts", label=label)
            reason_counts = {
                _safe_id(key, label=f"{label}.reason"): _required_int(
                    reason_counts_raw, key, label=f"{label}.reason_counts"
                )
                for key in sorted(reason_counts_raw)
            }
            non_observed = sum(status_counts[status] for status in STATUSES if status != "observed")
            if sum(reason_counts.values()) != non_observed:
                raise TrajectoryHandoffError(f"{label} reason counts do not close")
            structural = _required_bool(
                raw_cell, "structurally_emitted_by_builder", label=label
            )
            if structural != ((position, target) in BUILDER_CELLS):
                raise TrajectoryHandoffError(f"{label} structural eligibility disagrees")
            if not structural and row_count:
                raise TrajectoryHandoffError(f"{label} non-structural cell contains rows")
            eligible_row_count = _required_int(
                raw_cell, "eligible_row_count", label=label
            )
            eligible = _required_bool(
                raw_cell, "eligible_for_supervised_training", label=label
            )
            if (
                eligible_row_count != status_counts["observed"]
                or eligible != (status_counts["observed"] > 0)
            ):
                raise TrajectoryHandoffError(f"{label} eligible rows/boolean disagree")
            capability_unavailable = (
                (
                    target
                    in {"task_unknown_remaining_tokens", "call_unknown_billable_tokens"}
                    and not request_available
                )
                or (
                    (position == "call_update" or target == "call_remaining_output_tokens")
                    and not generation_available
                )
            )
            if capability_unavailable and status_counts["observed"]:
                raise TrajectoryHandoffError(
                    f"{label} has observed rows for unavailable telemetry"
                )
            matrix[position][target] = {
                "structurally_emitted_by_builder": structural,
                "row_count": row_count,
                "status_counts": status_counts,
                "reason_counts": reason_counts,
                "eligible_row_count": eligible_row_count,
                "eligible_for_supervised_training": eligible,
                **_spend_cell_capability(
                    position,
                    target,
                    raw_cell,
                    request_tokens_local_available=request_available,
                    generation_checkpoint_available=generation_available,
                ),
            }
    return matrix


def _build_spend(
    *,
    inventory_path: Path,
    trajectory_audit_path: Path,
    workspace_root: Path,
    repo_root: Path,
    code_artifact_pins: Mapping[str, Mapping[str, str]],
    expected_repo: str,
    expected_revision: str,
    expected_archive_bytes: int,
    expected_archive_sha256: str,
    archive_xet_etag: str,
) -> dict[str, Any]:
    inventory = _load_json(inventory_path, label="Spend archive inventory")
    trajectory_audit = _load_json(
        trajectory_audit_path,
        label=(
            "Spend trajectory audit (run scripts/audit_openhands_trajectory.py before freezing)"
        ),
    )
    inventory_relative_path = _workspace_path(
        inventory_path, workspace_root, label="Spend inventory"
    )
    trajectory_audit_relative_path = _workspace_path(
        trajectory_audit_path, workspace_root, label="Spend trajectory audit"
    )
    if inventory_relative_path != (
        "workspace/external/spend_your_money/gpt_5.2_inventory.json"
    ) or trajectory_audit_relative_path != (
        "workspace/external/spend_your_money/gpt_5.2_trajectory_audit.json"
    ):
        raise TrajectoryHandoffError(
            "Spend inventory/audit must use their pinned ignored-workspace paths"
        )
    if _required_int(
        inventory, "inventory_schema_version", label="Spend inventory"
    ) != SPEND_INVENTORY_SCHEMA_VERSION:
        raise TrajectoryHandoffError("Spend inventory schema version disagrees with pin")
    if _required_text(inventory, "source_id", label="Spend inventory") != (
        SPEND_INVENTORY_SOURCE_ID
    ):
        raise TrajectoryHandoffError("Spend inventory source_id disagrees with pin")
    if _required_int(
        trajectory_audit,
        "trajectory_audit_schema_version",
        label="Spend trajectory audit",
    ) != SPEND_TRAJECTORY_AUDIT_SCHEMA_VERSION:
        raise TrajectoryHandoffError("Spend trajectory audit schema version disagrees with pin")
    if set(code_artifact_pins) != {"reader", "builder", "labels", "audit"}:
        raise TrajectoryHandoffError("Spend semantic code pin roles are incomplete")
    code_artifacts = _freeze_pinned_code_artifacts(
        repo_root=repo_root,
        pins=code_artifact_pins,
        label="Spend semantic code pins",
    )
    inventory_sha = _file_sha256(inventory_path)
    audit_inventory = _required_mapping(
        trajectory_audit, "inventory", label="Spend trajectory audit"
    )
    if _safe_relative(
        _required_text(
            audit_inventory,
            "local_relative_path",
            label="Spend trajectory audit.inventory",
        ),
        label="Spend trajectory audit.inventory.local_relative_path",
    ) != inventory_relative_path:
        raise TrajectoryHandoffError("Spend trajectory audit inventory path disagrees")
    if _required_sha(audit_inventory, "sha256", label="Spend trajectory audit.inventory") != (
        inventory_sha
    ):
        raise TrajectoryHandoffError("Spend trajectory audit points to a different inventory")
    if _required_int(
        audit_inventory,
        "inventory_schema_version",
        label="Spend trajectory audit.inventory",
    ) != SPEND_INVENTORY_SCHEMA_VERSION:
        raise TrajectoryHandoffError("Spend trajectory audit inventory schema pin disagrees")
    audit_hash = trajectory_audit.get("audit_payload_sha256")
    if not isinstance(audit_hash, str) or SHA256_RE.fullmatch(audit_hash) is None:
        raise TrajectoryHandoffError("Spend trajectory audit lacks its payload SHA256")
    unhashed_audit = dict(trajectory_audit)
    del unhashed_audit["audit_payload_sha256"]
    if _semantic_sha256(unhashed_audit) != audit_hash:
        raise TrajectoryHandoffError("Spend trajectory audit payload SHA256 is invalid")

    inventory_repo = _required_text(inventory, "hub_repo", label="Spend inventory")
    inventory_revision = _required_text(
        inventory, "resolved_revision", label="Spend inventory"
    )
    if inventory_repo != expected_repo or inventory_revision != expected_revision:
        raise TrajectoryHandoffError("Spend inventory Hub identity disagrees with the frozen pin")
    archive_bytes = _required_int(inventory, "archive_bytes", label="Spend inventory")
    archive_sha = _required_sha(inventory, "archive_sha256", label="Spend inventory")
    if archive_bytes != expected_archive_bytes or archive_sha != expected_archive_sha256:
        raise TrajectoryHandoffError("Spend archive identity disagrees with the frozen pin")
    archive_path_text = _safe_relative(
        _required_text(inventory, "archive_path", label="Spend inventory"),
        label="Spend inventory.archive_path",
    )
    expected_archive_path_text = (
        "workspace/external/spend_your_money/gpt_5.2_4runs.tar.gz"
    )
    if archive_path_text != expected_archive_path_text:
        raise TrajectoryHandoffError(
            "Spend archive must remain at its pinned ignored-workspace path"
        )
    archive_path = Path(workspace_root).parent.joinpath(*PurePosixPath(archive_path_text).parts)
    _verify_file(
        archive_path,
        expected_bytes=archive_bytes,
        expected_sha256=archive_sha,
        label="Spend archive",
    )
    extracted_root = archive_path.parent / SPEND_ARCHIVE_WRAPPER
    if extracted_root.exists():
        raise TrajectoryHandoffError(
            "Spend extracted archive root is present; freeze requires archive-only workspace storage"
        )
    audit_archive = _required_mapping(
        trajectory_audit, "archive", label="Spend trajectory audit"
    )
    if (
        _safe_relative(
            _required_text(
                audit_archive,
                "local_relative_path",
                label="Spend trajectory audit.archive",
            ),
            label="Spend trajectory audit.archive.local_relative_path",
        )
        != archive_path_text
        or _required_int(audit_archive, "bytes", label="Spend trajectory audit.archive")
        != archive_bytes
        or _required_sha(audit_archive, "sha256", label="Spend trajectory audit.archive")
        != archive_sha
        or _required_text(audit_archive, "hub_repo", label="Spend trajectory audit.archive")
        != expected_repo
        or _required_text(
            audit_archive, "resolved_revision", label="Spend trajectory audit.archive"
        )
        != expected_revision
    ):
        raise TrajectoryHandoffError("Spend trajectory audit archive identity disagrees")

    counts = _required_mapping(trajectory_audit, "counts", label="Spend trajectory audit")
    identity_counts = {
        "task_id_count": _required_int(counts, "task_id_count", label="Spend counts"),
        "run_id_count": _required_int(counts, "run_id_count", label="Spend counts"),
        "trajectory_id_count": _required_int(
            counts, "trajectory_id_count", label="Spend counts"
        ),
        "condition_id_count": _required_int(counts, "condition_id_count", label="Spend counts"),
    }
    task_mapping_raw = _required_list(
        trajectory_audit, "task_run_mapping", label="Spend trajectory audit"
    )
    task_mapping: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    seen_trajectories: set[str] = set()
    seen_conditions: set[str] = set()
    condition_trajectory_counts: Counter[str] = Counter()
    mapping_quadruples: set[tuple[str, str, str, str]] = set()
    mapping_task_runs: set[tuple[str, str]] = set()
    for index, raw_task in enumerate(task_mapping_raw):
        if not isinstance(raw_task, Mapping):
            raise TrajectoryHandoffError(f"Spend task_run_mapping[{index}] must be an object")
        task_label = f"Spend task_run_mapping[{index}]"
        task_id = _safe_id(raw_task.get("task_id"), label=f"{task_label}.task_id")
        if task_id in seen_tasks:
            raise TrajectoryHandoffError(f"Spend task mapping repeats task_id {task_id}")
        seen_tasks.add(task_id)
        raw_runs = _required_list(raw_task, "runs", label=task_label)
        runs: list[dict[str, str]] = []
        for run_index, raw_run in enumerate(raw_runs):
            if not isinstance(raw_run, Mapping):
                raise TrajectoryHandoffError(f"{task_label}.runs[{run_index}] must be an object")
            run_label = f"{task_label}.runs[{run_index}]"
            run_id = _safe_id(raw_run.get("run_id"), label=f"{run_label}.run_id")
            trajectory_id = _safe_id(
                raw_run.get("trajectory_id"), label=f"{run_label}.trajectory_id"
            )
            condition_id = _safe_id(
                raw_run.get("condition_id"), label=f"{run_label}.condition_id"
            )
            if trajectory_id in seen_trajectories:
                raise TrajectoryHandoffError(f"Spend repeats trajectory_id {trajectory_id}")
            seen_trajectories.add(trajectory_id)
            seen_conditions.add(condition_id)
            condition_trajectory_counts[condition_id] += 1
            quadruple = (task_id, run_id, trajectory_id, condition_id)
            if quadruple in mapping_quadruples or (task_id, run_id) in mapping_task_runs:
                raise TrajectoryHandoffError(
                    "Spend task_run_mapping repeats an identity quadruple or task/run"
                )
            mapping_quadruples.add(quadruple)
            mapping_task_runs.add((task_id, run_id))
            runs.append(
                {
                    "run_id": run_id,
                    "trajectory_id": trajectory_id,
                    "condition_id": condition_id,
                }
            )
        if [item["run_id"] for item in runs] != ["run_1", "run_2", "run_3", "run_4"]:
            raise TrajectoryHandoffError(f"Spend task {task_id} does not map to exactly four runs")
        task_mapping.append({"task_id": task_id, "runs": runs})
    actual_counts = {
        "task_id_count": len(seen_tasks),
        "run_id_count": 4,
        "trajectory_id_count": len(seen_trajectories),
        "condition_id_count": len(seen_conditions),
    }
    if identity_counts != actual_counts:
        raise TrajectoryHandoffError("Spend identity counts do not close over task_run_mapping")
    inventory_status_counts = _normalize_spend_inventory_status_counts(
        inventory, trajectory_count=actual_counts["trajectory_id_count"]
    )
    if (
        _required_int(inventory, "task_count", label="Spend inventory")
        != actual_counts["task_id_count"]
        or _required_int(inventory, "trajectory_count", label="Spend inventory")
        != actual_counts["trajectory_id_count"]
        or _required_int(inventory, "run_count", label="Spend inventory")
        != actual_counts["run_id_count"]
    ):
        raise TrajectoryHandoffError("Spend inventory identity counts disagree")
    run_ids = _required_list(trajectory_audit, "run_ids", label="Spend trajectory audit")
    if run_ids != ["run_1", "run_2", "run_3", "run_4"]:
        raise TrajectoryHandoffError("Spend trajectory audit run_ids disagree with pin")
    raw_global_condition_counts = _required_mapping(
        trajectory_audit, "condition_counts", label="Spend trajectory audit"
    )
    global_condition_counts = {
        _safe_id(key, label="Spend condition_id"): _required_int(
            raw_global_condition_counts, key, label="Spend condition_counts"
        )
        for key in sorted(raw_global_condition_counts)
    }
    if global_condition_counts != dict(sorted(condition_trajectory_counts.items())):
        raise TrajectoryHandoffError("Spend global condition counts disagree with task mapping")
    source_capability = _required_mapping(
        trajectory_audit, "source_capability", label="Spend trajectory audit"
    )

    inventory_runs = _required_list(inventory, "runs", label="Spend inventory")
    audit_per_run = _required_mapping(
        trajectory_audit, "per_run", label="Spend trajectory audit"
    )
    if len(inventory_runs) != 4 or set(audit_per_run) != {
        "run_1",
        "run_2",
        "run_3",
        "run_4",
    }:
        raise TrajectoryHandoffError("Spend must contain exactly runs 1 through 4")
    runs_output: list[dict[str, Any]] = []
    inventory_report_totals: Counter[str] = Counter()
    evaluator_report_totals: Counter[str] = Counter()
    summed_run_matrix_counts: dict[tuple[str, str], Counter[str]] = {
        (position, target): Counter() for position in POSITIONS for target in TARGETS
    }
    for expected_run_id, raw_run in enumerate(inventory_runs, start=1):
        if not isinstance(raw_run, Mapping):
            raise TrajectoryHandoffError("Spend inventory run entry must be an object")
        label = f"Spend inventory.runs[{expected_run_id - 1}]"
        run_id = _required_int(raw_run, "run_id", label=label)
        if run_id != expected_run_id:
            raise TrajectoryHandoffError("Spend inventory runs are not ordered 1 through 4")
        run_basename = _safe_segment(
            _required_text(raw_run, "run_basename", label=label),
            label=f"{label}.run_basename",
        )
        run_key = f"run_{run_id}"
        per_run = audit_per_run[run_key]
        if not isinstance(per_run, Mapping):
            raise TrajectoryHandoffError(f"Spend trajectory audit.per_run.{run_key} must be an object")
        output_jsonl = _inventory_jsonl_member(
            _required_mapping(raw_run, "output_jsonl", label=label),
            wrapper=SPEND_ARCHIVE_WRAPPER,
            run_basename=run_basename,
            expected_filename="output.jsonl",
            label=f"{label}.output_jsonl",
        )
        output_swebench_jsonl = _inventory_jsonl_member(
            _required_mapping(raw_run, "output_swebench_jsonl", label=label),
            wrapper=SPEND_ARCHIVE_WRAPPER,
            run_basename=run_basename,
            expected_filename="output.swebench.jsonl",
            label=f"{label}.output_swebench_jsonl",
        )
        run_dataset = _required_mapping(per_run, "dataset", label=f"Spend {run_key}")
        run_matrix = _normalize_spend_matrix(
            _required_mapping(per_run, "label_matrix", label=f"Spend {run_key}"),
            source_capability,
            matrix_label=f"Spend {run_key}.label_matrix",
        )
        run_matrix_status: Counter[str] = Counter()
        for position in POSITIONS:
            for target in TARGETS:
                cell_counts = run_matrix[position][target]["status_counts"]
                run_matrix_status.update(cell_counts)
                summed_run_matrix_counts[(position, target)].update(cell_counts)
        run_dataset_row_count = _required_int(
            run_dataset, "row_count", label=f"Spend {run_key}.dataset"
        )
        if sum(run_matrix_status.values()) != run_dataset_row_count:
            raise TrajectoryHandoffError(f"Spend {run_key} matrix/dataset rows disagree")
        run_task_count = _required_int(per_run, "task_count", label=f"Spend {run_key}")
        run_trajectory_count = _required_int(
            per_run, "trajectory_count", label=f"Spend {run_key}"
        )
        if (
            _required_int(raw_run, "task_count", label=label) != run_task_count
            or _required_int(raw_run, "task_run_count", label=label) != run_task_count
            or run_task_count != run_trajectory_count
            or output_jsonl["record_count"] != run_trajectory_count
            or output_swebench_jsonl["record_count"] != run_trajectory_count
        ):
            raise TrajectoryHandoffError(f"Spend {run_key} inventory task counts disagree")
        task_report_count = _required_int(raw_run, "task_report_count", label=label)
        aggregate_report_count = _required_int(
            raw_run, "aggregate_report_count", label=label
        )
        report_count = _required_int(raw_run, "report_count", label=label)
        run_metrics = _required_mapping(per_run, "metrics", label=f"Spend {run_key}")
        if _required_int(
            raw_run, "llm_completions_count", label=label
        ) != _required_int(
            run_metrics, "logical_call_count", label=f"Spend {run_key}.metrics"
        ):
            raise TrajectoryHandoffError(
                f"Spend {run_key} completion/logical-call counts disagree"
            )
        evaluator_observed = _required_int(
            run_metrics,
            "evaluator_report_observed_count",
            label=f"Spend {run_key}.metrics",
        )
        evaluator_missing = _required_int(
            run_metrics,
            "evaluator_report_missing_count",
            label=f"Spend {run_key}.metrics",
        )
        if (
            report_count != task_report_count + aggregate_report_count
            or aggregate_report_count != 1
            or task_report_count != evaluator_observed
            or evaluator_observed + evaluator_missing != run_trajectory_count
        ):
            raise TrajectoryHandoffError(f"Spend {run_key} report counts do not close")
        evaluator_report_totals.update(
            {"observed": evaluator_observed, "missing": evaluator_missing}
        )
        task_report_bytes = _required_int(raw_run, "task_report_bytes", label=label)
        aggregate_report_bytes = _required_int(
            raw_run, "aggregate_report_bytes", label=label
        )
        report_bytes = _required_int(raw_run, "report_bytes", label=label)
        if report_bytes != task_report_bytes + aggregate_report_bytes:
            raise TrajectoryHandoffError(f"Spend {run_key} report bytes do not close")
        inventory_report_totals.update(
            {
                "task_report_count": task_report_count,
                "aggregate_report_count": aggregate_report_count,
                "report_count": report_count,
                "task_report_bytes": task_report_bytes,
                "aggregate_report_bytes": aggregate_report_bytes,
                "report_bytes": report_bytes,
            }
        )
        raw_condition_counts = _required_mapping(
            per_run, "condition_counts", label=f"Spend {run_key}"
        )
        condition_counts = {
            _safe_id(key, label=f"Spend {run_key}.condition_id"): _required_int(
                raw_condition_counts, key, label=f"Spend {run_key}.condition_counts"
            )
            for key in sorted(raw_condition_counts)
        }
        if sum(condition_counts.values()) != run_trajectory_count:
            raise TrajectoryHandoffError(f"Spend {run_key} condition counts do not close")
        runs_output.append(
            {
                "run_id": run_key,
                "archive_internal_root": f"{SPEND_ARCHIVE_WRAPPER}/{run_basename}",
                "output_jsonl": output_jsonl,
                "output_swebench_jsonl": output_swebench_jsonl,
                "counts": {
                    "task_id_count": _required_int(per_run, "task_count", label=f"Spend {run_key}"),
                    "trajectory_id_count": _required_int(
                        per_run, "trajectory_count", label=f"Spend {run_key}"
                    ),
                    "llm_completion_member_count": _required_int(
                        raw_run, "llm_completions_count", label=label
                    ),
                    "task_report_count": task_report_count,
                    "aggregate_report_count": aggregate_report_count,
                    "report_count": report_count,
                    "evaluator_report_missing_count": evaluator_missing,
                    "task_report_bytes": task_report_bytes,
                    "aggregate_report_bytes": aggregate_report_bytes,
                    "report_bytes": report_bytes,
                },
                "condition_counts": condition_counts,
                "canonical_aggregate_sha256": _required_sha(
                    per_run, "canonical_aggregate_sha256", label=f"Spend {run_key}"
                ),
                "dataset": {
                    "dataset_id": _required_sha(
                        run_dataset, "dataset_id", label=f"Spend {run_key}.dataset"
                    ),
                    "row_count": run_dataset_row_count,
                    "label_status_counts": {
                        status: run_matrix_status[status] for status in STATUSES
                    },
                },
            }
        )

    global_report_counts = {
        key: _required_int(inventory, key, label="Spend inventory")
        for key in (
            "task_report_count",
            "aggregate_report_count",
            "report_count",
            "task_report_bytes",
            "aggregate_report_bytes",
            "report_bytes",
        )
    }
    if global_report_counts["report_count"] != (
        global_report_counts["task_report_count"]
        + global_report_counts["aggregate_report_count"]
    ) or global_report_counts["report_bytes"] != (
        global_report_counts["task_report_bytes"]
        + global_report_counts["aggregate_report_bytes"]
    ) or global_report_counts["aggregate_report_count"] != 4:
        raise TrajectoryHandoffError("Spend global report counts/bytes do not close")
    if global_report_counts != {
        key: inventory_report_totals[key] for key in global_report_counts
    }:
        raise TrajectoryHandoffError("Spend global/per-run report totals disagree")
    trajectory_metrics = _required_mapping(
        trajectory_audit, "metrics", label="Spend trajectory audit"
    )
    evaluator_observed = _required_int(
        trajectory_metrics,
        "evaluator_report_observed_count",
        label="Spend trajectory audit.metrics",
    )
    evaluator_missing = _required_int(
        trajectory_metrics,
        "evaluator_report_missing_count",
        label="Spend trajectory audit.metrics",
    )
    if (
        evaluator_observed != global_report_counts["task_report_count"]
        or evaluator_observed != evaluator_report_totals["observed"]
        or evaluator_missing != evaluator_report_totals["missing"]
        or evaluator_observed + evaluator_missing
        != identity_counts["trajectory_id_count"]
    ):
        raise TrajectoryHandoffError(
            "Spend canonical evaluator reports disagree with inventory task reports"
        )
    evaluator_label_status = inventory_status_counts["labels"]["evaluator_accuracy"]
    if (
        evaluator_label_status["observed"] != evaluator_observed
        or sum(evaluator_label_status.values())
        != identity_counts["trajectory_id_count"]
    ):
        raise TrajectoryHandoffError(
            "Spend evaluator label statuses disagree with report evidence"
        )
    safe_source_capability = _validate_spend_source_capability(
        source_capability,
        trajectory_metrics,
        trajectory_count=identity_counts["trajectory_id_count"],
    )
    if inventory_status_counts["telemetry"]["task_error_nonempty"] != (
        safe_source_capability["errors"]["task_error_count"]
    ):
        raise TrajectoryHandoffError(
            "Spend inventory and trajectory task-error counts disagree"
        )
    reader_source_id = safe_source_capability["source_id"]
    matrix = _normalize_spend_matrix(
        _required_mapping(trajectory_audit, "label_matrix", label="Spend trajectory audit"),
        source_capability,
        matrix_label="Spend label_matrix",
    )
    aggregate_status_counts: Counter[str] = Counter()
    for targets in matrix.values():
        for cell in targets.values():
            aggregate_status_counts.update(cell["status_counts"])
    dataset = _required_mapping(trajectory_audit, "dataset", label="Spend trajectory audit")
    dataset_row_count = _required_int(dataset, "row_count", label="Spend dataset")
    if (
        _required_int(dataset, "schema_version", label="Spend dataset")
        != SPEND_DATASET_SCHEMA_VERSION
        or _required_int(dataset, "feature_schema_version", label="Spend dataset")
        != SPEND_FEATURE_SCHEMA_VERSION
    ):
        raise TrajectoryHandoffError("Spend dataset/feature schema versions disagree with pins")
    if (
        sum(aggregate_status_counts.values()) != dataset_row_count
        or _required_int(counts, "dataset_row_count", label="Spend counts")
        != dataset_row_count
        or sum(int(item["dataset"]["row_count"]) for item in runs_output)
        != dataset_row_count
    ):
        raise TrajectoryHandoffError("Spend global/per-run matrix and dataset rows disagree")
    for position in POSITIONS:
        for target in TARGETS:
            global_counts = matrix[position][target]["status_counts"]
            run_counts = summed_run_matrix_counts[(position, target)]
            if global_counts != {status: run_counts[status] for status in STATUSES}:
                raise TrajectoryHandoffError(
                    f"Spend global/per-run matrix cell disagrees: {position}/{target}"
                )
    canonical_trajectories = _required_list(
        trajectory_audit, "canonical_trajectories", label="Spend trajectory audit"
    )
    if len(canonical_trajectories) != identity_counts["trajectory_id_count"]:
        raise TrajectoryHandoffError("Spend canonical trajectory hash index is incomplete")
    safe_canonical: list[dict[str, str]] = []
    canonical_quadruples: set[tuple[str, str, str, str]] = set()
    canonical_trajectory_ids: set[str] = set()
    canonical_task_runs: set[tuple[str, str]] = set()
    canonical_condition_counts: Counter[str] = Counter()
    for index, raw in enumerate(canonical_trajectories):
        if not isinstance(raw, Mapping):
            raise TrajectoryHandoffError(f"Spend canonical_trajectories[{index}] must be an object")
        label = f"Spend canonical_trajectories[{index}]"
        record = {
            "task_id": _safe_id(raw.get("task_id"), label=f"{label}.task_id"),
            "run_id": _safe_id(raw.get("run_id"), label=f"{label}.run_id"),
            "trajectory_id": _safe_id(
                raw.get("trajectory_id"), label=f"{label}.trajectory_id"
            ),
            "condition_id": _safe_id(
                raw.get("condition_id"), label=f"{label}.condition_id"
            ),
            "canonical_sha256": _required_sha(raw, "canonical_sha256", label=label),
        }
        quadruple = (
            record["task_id"],
            record["run_id"],
            record["trajectory_id"],
            record["condition_id"],
        )
        task_run = (record["task_id"], record["run_id"])
        if (
            quadruple in canonical_quadruples
            or record["trajectory_id"] in canonical_trajectory_ids
            or task_run in canonical_task_runs
        ):
            raise TrajectoryHandoffError(
                "Spend canonical_trajectories repeats a quadruple, trajectory, or task/run"
            )
        canonical_quadruples.add(quadruple)
        canonical_trajectory_ids.add(record["trajectory_id"])
        canonical_task_runs.add(task_run)
        canonical_condition_counts[record["condition_id"]] += 1
        safe_canonical.append(record)
    if canonical_quadruples != mapping_quadruples:
        raise TrajectoryHandoffError(
            "Spend canonical_trajectories and task_run_mapping quadruples disagree"
        )
    if dict(sorted(canonical_condition_counts.items())) != global_condition_counts:
        raise TrajectoryHandoffError("Spend canonical condition counts disagree")
    safe_canonical.sort(key=lambda item: item["trajectory_id"])
    canonical_source_sha = _semantic_sha256(safe_canonical)
    if _required_sha(
        trajectory_audit,
        "canonical_source_aggregate_sha256",
        label="Spend trajectory audit",
    ) != canonical_source_sha:
        raise TrajectoryHandoffError("Spend canonical source aggregate SHA256 disagrees")
    run_canonical_sha: dict[str, str] = {}
    for run_key in ("run_1", "run_2", "run_3", "run_4"):
        run_records = [item for item in safe_canonical if item["run_id"] == run_key]
        run_sha = _semantic_sha256(run_records)
        run_canonical_sha[run_key] = run_sha
        frozen_run = next(item for item in runs_output if item["run_id"] == run_key)
        if (
            len(run_records) != frozen_run["counts"]["trajectory_id_count"]
            or len({item["task_id"] for item in run_records})
            != frozen_run["counts"]["task_id_count"]
        ):
            raise TrajectoryHandoffError(f"Spend {run_key} canonical counts disagree")
        per_run = audit_per_run[run_key]
        if not isinstance(per_run, Mapping) or _required_sha(
            per_run, "canonical_aggregate_sha256", label=f"Spend {run_key}"
        ) != run_sha:
            raise TrajectoryHandoffError(
                f"Spend {run_key} canonical aggregate SHA256 disagrees"
            )
        run_conditions = Counter(item["condition_id"] for item in run_records)
        frozen_conditions = frozen_run["condition_counts"]
        if dict(sorted(run_conditions.items())) != frozen_conditions:
            raise TrajectoryHandoffError(f"Spend {run_key} canonical conditions disagree")

    return {
        "source_id": SPEND_INVENTORY_SOURCE_ID,
        "trajectory_reader_source_id": reader_source_id,
        "schema_pins": {
            "trajectory_audit_schema_version": SPEND_TRAJECTORY_AUDIT_SCHEMA_VERSION,
            "inventory_schema_version": SPEND_INVENTORY_SCHEMA_VERSION,
            "dataset_schema_version": SPEND_DATASET_SCHEMA_VERSION,
            "feature_schema_version": SPEND_FEATURE_SCHEMA_VERSION,
        },
        "semantic_code_artifacts": code_artifacts,
        "hub": {
            "repo_id": expected_repo,
            "repo_type": "dataset",
            "resolved_revision": expected_revision,
            "url": f"https://huggingface.co/datasets/{expected_repo}",
            "pinned_tree_url": (
                f"https://huggingface.co/datasets/{expected_repo}/tree/{expected_revision}"
            ),
        },
        "archive": {
            "local_relative_path": archive_path_text,
            "bytes": archive_bytes,
            "sha256": archive_sha,
            "xet_etag": archive_xet_etag,
            "pinned_url": (
                f"https://huggingface.co/datasets/{expected_repo}/resolve/"
                f"{expected_revision}/{SPEND_ARCHIVE_NAME}"
            ),
            "extracted_copy_present": False,
        },
        "inventory": {
            "local_relative_path": _workspace_path(
                inventory_path, workspace_root, label="Spend inventory"
            ),
            "bytes": inventory_path.stat().st_size,
            "sha256": inventory_sha,
            "schema_version": SPEND_INVENTORY_SCHEMA_VERSION,
            "report_evidence": {
                **global_report_counts,
                "evaluator_report_observed_count": evaluator_observed,
                "evaluator_report_missing_count": evaluator_missing,
            },
        },
        "trajectory_audit": {
            "local_relative_path": _workspace_path(
                trajectory_audit_path, workspace_root, label="Spend trajectory audit"
            ),
            "bytes": trajectory_audit_path.stat().st_size,
            "sha256": _file_sha256(trajectory_audit_path),
            "payload_sha256": audit_hash,
            "schema_version": SPEND_TRAJECTORY_AUDIT_SCHEMA_VERSION,
        },
        "runs": runs_output,
        "identity": {
            **identity_counts,
            "condition_trajectory_counts": dict(sorted(condition_trajectory_counts.items())),
            "task_cross_run_mapping": task_mapping,
        },
        "telemetry_capability": safe_source_capability,
        "position_target_matrix": matrix,
        "label_status_totals": {
            status: aggregate_status_counts[status] for status in STATUSES
        },
        "canonical": {
            "source_aggregate_sha256": canonical_source_sha,
            "per_run_aggregate_sha256": run_canonical_sha,
            "trajectory_hashes": safe_canonical,
            "audit_payload_sha256": audit_hash,
        },
        "dataset": {
            "dataset_id": _required_sha(dataset, "dataset_id", label="Spend dataset"),
            "row_count": dataset_row_count,
            "schema_version": SPEND_DATASET_SCHEMA_VERSION,
            "feature_schema_version": SPEND_FEATURE_SCHEMA_VERSION,
            "construction": _required_text(dataset, "construction", label="Spend dataset"),
            "construction_command": (
                "$env:PYTHONPATH='src'; python scripts/audit_openhands_trajectory.py "
                "--archive workspace/external/spend_your_money/gpt_5.2_4runs.tar.gz "
                "--inventory workspace/external/spend_your_money/gpt_5.2_inventory.json "
                "--output workspace/external/spend_your_money/gpt_5.2_trajectory_audit.json"
            ),
        },
        "inventory_status_counts": inventory_status_counts,
    }


def _experiment_recommendations() -> dict[str, Any]:
    return {
        "immediate": [
            {
                "id": "exact_call_output",
                "sources": [
                    "bagen_swebench_traj_v1",
                    "spend_your_money/openhands_trajectories:gpt_5.2_4runs",
                ],
                "positions": ["call_pre"],
                "targets": [
                    "call_billable_output_tokens",
                    "call_final_response_output_tokens",
                ],
                "guard": "train/evaluate only rows whose label status is observed",
            },
            {
                "id": "bagen_task_total",
                "sources": ["bagen_swebench_traj_v1"],
                "positions": ["task_launch"],
                "targets": ["task_total_accounted_tokens"],
                "guard": "group splits by task_id and retain condition_id",
            },
            {
                "id": "spend_task_total_observed_subset",
                "sources": ["spend_your_money/openhands_trajectories:gpt_5.2_4runs"],
                "positions": ["task_launch"],
                "targets": ["task_total_accounted_tokens"],
                "eligibility_evidence": {
                    "observed_rows": 1_896,
                    "excluded_censored_rows": 104,
                    "excluded_censored_reason": "task_error",
                },
                "guard": (
                    "train/evaluate only the 1,896 observed rows; exclude the 104 "
                    "task-error rows as censored; group splits by task_id and retain "
                    "condition_id"
                ),
            },
            {
                "id": "paired_task_across_family_or_run",
                "sources": [
                    "bagen_swebench_traj_v1",
                    "spend_your_money/openhands_trajectories:gpt_5.2_4runs",
                ],
                "guard": (
                    "use frozen task_cross_family_mapping/task_cross_run_mapping; never split "
                    "one task across train and evaluation"
                ),
            },
            {
                "id": "spend_evaluator_observed_subset",
                "sources": ["spend_your_money/openhands_trajectories:gpt_5.2_4runs"],
                "guard": "use only observed evaluator labels; keep missing/censored/invalid distinct",
            },
        ],
        "gated": [
            {
                "id": "exact_unknown_token_targets",
                "gate": (
                    "request_tokens_local is absent in Spend and only a provider-input proxy in BAGEN"
                ),
            },
            {
                "id": "provider_retry_prediction",
                "gate": (
                    "neither source preserves a provider transport retry ledger; BAGEN FormatError "
                    "recovery is a different phenomenon"
                ),
            },
            {
                "id": "task_lifecycle_feature_leakage",
                "gate": (
                    "task_error and terminal lifecycle are post-launch outcome metadata; "
                    "do not use them as predictors for task-launch task-total labels"
                ),
            },
            {
                "id": "generation_checkpoint_prediction",
                "gate": "neither source preserves streaming generation checkpoints",
            },
            {
                "id": "cross_source_tool_failure_prediction",
                "gate": (
                    "cross-source tool instrumentation and failure semantics are not "
                    "aligned: BAGEN emits terminal-only events without trustworthy "
                    "starts and derives failures mainly from return codes, while Spend "
                    "preserves explicit action/observation pairs; comparisons require "
                    "scope normalization"
                ),
            },
            {
                "id": "full_coverage_spend_accuracy",
                "gate": "Spend evaluator outcomes contain missing and invalid labels; never fill with zero",
            },
        ],
    }


def build_handoff(
    manifest_summary_path: Path,
    bagen_combined_audit_path: Path,
    bagen_audit_paths: Mapping[str, Path],
    spend_inventory_path: Path,
    spend_trajectory_audit_path: Path,
    *,
    workspace_root: Path,
    changed_files: Sequence[str],
    validation_results: Sequence[Mapping[str, Any]],
    repo_root: Path = REPO_ROOT,
    bagen_expected_combined_dataset_id: str = BAGEN_COMBINED_DATASET_ID,
    bagen_expected_combined_counts: Mapping[str, int] = BAGEN_COMBINED_EXPECTED_COUNTS,
    bagen_expected_task_cross_family_distribution: Mapping[
        str, int
    ] = BAGEN_TASK_CROSS_FAMILY_DISTRIBUTION,
    spend_code_artifact_pins: Mapping[
        str, Mapping[str, str]
    ] = SPEND_CODE_ARTIFACT_PINS,
    spend_expected_repo: str = SPEND_REPO,
    spend_expected_revision: str = SPEND_REVISION,
    spend_expected_archive_bytes: int = SPEND_ARCHIVE_BYTES,
    spend_expected_archive_sha256: str = SPEND_ARCHIVE_SHA256,
    spend_archive_xet_etag: str = SPEND_ARCHIVE_XET_ETAG,
) -> dict[str, Any]:
    """Build a content-free, deterministic handoff from frozen audit artifacts."""

    validation = _validation_output(validation_results)
    bagen = _build_bagen(
        manifest_path=Path(manifest_summary_path),
        combined_audit_path=Path(bagen_combined_audit_path),
        audit_paths={key: Path(value) for key, value in bagen_audit_paths.items()},
        workspace_root=Path(workspace_root),
        repo_root=Path(repo_root),
        expected_combined_dataset_id=bagen_expected_combined_dataset_id,
        expected_combined_counts=bagen_expected_combined_counts,
        expected_task_cross_family_distribution=(
            bagen_expected_task_cross_family_distribution
        ),
    )
    spend = _build_spend(
        inventory_path=Path(spend_inventory_path),
        trajectory_audit_path=Path(spend_trajectory_audit_path),
        workspace_root=Path(workspace_root),
        repo_root=Path(repo_root),
        code_artifact_pins=spend_code_artifact_pins,
        expected_repo=spend_expected_repo,
        expected_revision=spend_expected_revision,
        expected_archive_bytes=spend_expected_archive_bytes,
        expected_archive_sha256=spend_expected_archive_sha256,
        archive_xet_etag=spend_archive_xet_etag,
    )
    handoff: dict[str, Any] = {
        "handoff_schema_version": HANDOFF_SCHEMA_VERSION,
        "policy": {
            "license_status": "not independently verified",
            "redistribution_allowed": False,
            "raw_archives_and_trajectories": "ignored workspace only",
            "contains_raw_message_response_or_tool_content": False,
            "contains_auth_material": False,
            "task_ids": "retained only in ignored workspace handoff for mainline grouping",
            "missing_value_policy": "never impute zero; preserve observed/missing/censored/invalid",
        },
        "sources": {
            "bagen": bagen,
            "spend_your_money": spend,
        },
        "source_position_target_matrix": {
            "bagen": bagen["position_target_matrix"],
            "spend_your_money": spend["position_target_matrix"],
        },
        "reproducibility": {
            "commands": [
                (
                    "$env:PYTHONPATH='src'; python scripts/audit_bagen_manifest.py "
                    "workspace/external/bagen/manifest.jsonl "
                    "workspace/external/bagen/manifest_summary.json"
                ),
                (
                    "$env:PYTHONPATH='src'; python scripts/audit_openhands_archive.py "
                    "--archive workspace/external/spend_your_money/gpt_5.2_4runs.tar.gz "
                    "--output workspace/external/spend_your_money/gpt_5.2_inventory.json"
                ),
                bagen["dataset"]["construction_command"],
                bagen["combined_audit"]["construction_command"],
                spend["dataset"]["construction_command"],
                (
                    "$env:PYTHONPATH='src'; python scripts/freeze_trajectory_handoff.py "
                    "--ruff-status passed --ruff-result '<ruff summary>' "
                    "--pytest-status passed --pytest-result '<pytest summary>'"
                ),
            ],
            "input_artifact_hashes": {
                "bagen_manifest": bagen["manifest"]["sha256"],
                "bagen_manifest_summary": bagen["manifest"]["summary_sha256"],
                "bagen_family_audits": {
                    item["family"]: item["audit"]["sha256"] for item in bagen["families"]
                },
                "bagen_combined_audit": bagen["combined_audit"]["sha256"],
                "spend_archive": spend["archive"]["sha256"],
                "spend_inventory": spend["inventory"]["sha256"],
                "spend_trajectory_audit": spend["trajectory_audit"]["sha256"],
            },
        },
        "implementation_validation": {
            "changed_files": _classify_changed_files(changed_files),
            "tests": validation,
        },
        "recommended_experiments": _experiment_recommendations(),
    }
    _assert_no_absolute_paths(handoff)
    handoff["handoff_payload_sha256"] = _semantic_sha256(handoff)
    return handoff


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                value,
                handle,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _default_audits() -> dict[str, Path]:
    return {
        family: DEFAULT_BAGEN_AUDIT_DIR / filename
        for family, filename in BAGEN_FAMILY_AUDITS.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the audited BAGEN and Spend trajectory branch for mainline consumption."
    )
    parser.add_argument("--manifest-summary", type=Path, default=DEFAULT_MANIFEST_SUMMARY)
    parser.add_argument(
        "--bagen-combined-audit", type=Path, default=DEFAULT_BAGEN_COMBINED_AUDIT
    )
    parser.add_argument("--spend-inventory", type=Path, default=DEFAULT_SPEND_INVENTORY)
    parser.add_argument(
        "--spend-trajectory-audit", type=Path, default=DEFAULT_SPEND_TRAJECTORY_AUDIT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--changed-file", action="append", dest="changed_files")
    parser.add_argument("--ruff-status", choices=("passed", "failed"), required=True)
    parser.add_argument("--ruff-result", required=True)
    parser.add_argument("--pytest-status", choices=("passed", "failed"), required=True)
    parser.add_argument("--pytest-result", required=True)
    args = parser.parse_args()

    validation = [
        {
            "name": "ruff",
            "command": "$env:PYTHONPATH='src'; python -m ruff check src tests scripts",
            "status": args.ruff_status,
            "result": args.ruff_result,
        },
        {
            "name": "pytest",
            "command": "$env:PYTHONPATH='src'; python -m pytest -q",
            "status": args.pytest_status,
            "result": args.pytest_result,
        },
    ]
    try:
        output_relative = _workspace_path(
            args.output, DEFAULT_WORKSPACE, label="handoff output"
        )
        handoff = build_handoff(
            args.manifest_summary,
            args.bagen_combined_audit,
            _default_audits(),
            args.spend_inventory,
            args.spend_trajectory_audit,
            workspace_root=DEFAULT_WORKSPACE,
            changed_files=args.changed_files or DEFAULT_CHANGED_FILES,
            validation_results=validation,
        )
        atomic_write_json(args.output, handoff)
    except (OSError, TrajectoryHandoffError) as exc:
        parser.exit(2, f"handoff freeze failed: {exc}\n")
    print(
        json.dumps(
            {
                "output": output_relative,
                "handoff_payload_sha256": handoff["handoff_payload_sha256"],
                "bagen_trajectory_count": handoff["sources"]["bagen"]["identity"][
                    "trajectory_id_count"
                ],
                "spend_trajectory_count": handoff["sources"]["spend_your_money"]["identity"][
                    "trajectory_id_count"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
