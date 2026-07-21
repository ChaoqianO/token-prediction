from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Iterable, Mapping, Sequence

from token_prediction.collection import (
    BagenSwebenchReader,
    OpenHandsArchiveMetadata,
    OpenHandsArchiveReader,
)
from token_prediction.contracts import SourceCapabilities, SourceDescriptor
from token_prediction.dataset import (
    CAPABILITY_DATASET_SCHEMA_VERSION,
    LabelStatus,
    PredictionPosition,
    PredictionTarget,
    build_capability_supervised_dataset,
    decide_target_capability,
)
from token_prediction.trajectory import Trajectory


AUDIT_SCHEMA_VERSION = 1
READ_BUFFER_BYTES = 1024 * 1024
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BAGEN_DESCRIPTOR = Path("configs/source_descriptors/bagen_swebench.json")
DEFAULT_BAGEN_COMBINED_AUDIT = Path(
    "workspace/external/bagen/combined_swebench_audit.json"
)
DEFAULT_SPEND_DESCRIPTOR = Path("configs/source_descriptors/spend_openhands.json")
DEFAULT_OUTPUT = Path(
    "workspace/data_foundation/data_foundation_v2_audit.json"
)

BAGEN_DESCRIPTOR_SHA256 = (
    "fa54a5c80e386a3bd12c00e8525f3c119c88407c8e88de8e2f97d06f2596a97d"
)
SPEND_DESCRIPTOR_SHA256 = (
    "1306c6b4c74b0af72ade350edb0eefc6146d35db465cd5a88543583477386c1a"
)
BAGEN_COMBINED_AUDIT_SHA256 = (
    "2d8f3abe10b526f80488554d672039c9f9bc81b31e230b7bb6b14c94b0ffaea5"
)
BAGEN_COMBINED_AUDIT_SOURCE_ID = "bagen_swebench_combined_audit_v1"
BAGEN_MANIFEST_SHA256 = (
    "f5900dead3a32ca303d500f123ee96b89e6797527cbb99fef0cd9beaf2a00071"
)
SPEND_ARCHIVE_SHA256 = (
    "993abcb55aae423f9067d5e6c8e1aeaccf83b9ce31474a215982686527934214"
)
SPEND_ARCHIVE_BYTES = 2_908_192_516
EXPECTED_BAGEN_TRAJECTORIES = 316
EXPECTED_SPEND_TRAJECTORIES = 2_000
SPEND_INVENTORY_SOURCE_ID = (
    "spend_your_money/openhands_trajectories:gpt_5.2_4runs"
)

BUILD_COMMAND = (
    "$env:PYTHONPATH='src'; python scripts/audit_data_foundation_v2.py"
)


class DataFoundationAuditError(ValueError):
    """Raised when schema-v2 source evidence does not close exactly."""


@dataclass(frozen=True)
class ArtifactEvidence:
    path: str
    bytes: int
    sha256: str
    file_count: int = 1
    sha256_kind: str = "file_bytes"

    def __post_init__(self) -> None:
        _canonical_relative_path(self.path, label="artifact path")
        _require_non_negative_int(self.bytes, "artifact bytes")
        if self.file_count <= 0:
            raise DataFoundationAuditError("artifact file_count must be positive")
        _require_sha256(self.sha256, "artifact sha256")
        if self.sha256_kind not in {"file_bytes", "framed_file_index_v1"}:
            raise DataFoundationAuditError("artifact sha256_kind is unsupported")

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "sha256_kind": self.sha256_kind,
            "file_count": self.file_count,
        }


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataFoundationAuditError(f"{label} must be an object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise DataFoundationAuditError(f"{label} must be a list")
    return value


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataFoundationAuditError(f"{label} must be a non-empty string")
    return value


def _require_source_id(
    value: Mapping[str, Any],
    *,
    expected: str,
    label: str,
) -> None:
    actual = _require_text(value.get("source_id"), f"{label}.source_id")
    if actual != expected:
        raise DataFoundationAuditError(f"{label} source_id mismatch")


def _require_non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DataFoundationAuditError(f"{label} must be a non-negative integer")
    return value


def _require_sha256(value: Any, label: str) -> str:
    digest = _require_text(value, label)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise DataFoundationAuditError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _require_git_commit(value: Any, label: str = "Git commit") -> str:
    commit = _require_text(value, label)
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit):
        raise DataFoundationAuditError(
            f"{label} must be one full lowercase Git object id"
        )
    return commit


def _canonical_relative_path(value: Any, *, label: str) -> str:
    path = _require_text(value, label)
    posix = PurePosixPath(path)
    windows = PureWindowsPath(path)
    if (
        "\\" in path
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part in {"", ".", ".."} for part in posix.parts)
        or posix.as_posix() != path
    ):
        raise DataFoundationAuditError(
            f"{label} must be a canonical relative POSIX path"
        )
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(READ_BUFFER_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DataFoundationAuditError(f"value is not canonical JSON: {exc}") from exc
    return rendered.encode("utf-8")


def _semantic_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DataFoundationAuditError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise DataFoundationAuditError(f"JSON contains non-finite constant {value}")


def _reject_non_finite(value: Any, *, label: str) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DataFoundationAuditError(f"{label} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_non_finite(item, label=f"{label}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_finite(item, label=f"{label}[{index}]")


def _strict_json_loads(value: str, *, label: str) -> Any:
    try:
        parsed = json.loads(
            value,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except DataFoundationAuditError:
        raise
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise DataFoundationAuditError(f"{label} is not strict UTF-8 JSON") from exc
    _reject_non_finite(parsed, label=label)
    return parsed


def load_strict_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise DataFoundationAuditError(f"cannot read {label}") from exc
    parsed = _strict_json_loads(text, label=label)
    if not isinstance(parsed, dict):
        raise DataFoundationAuditError(f"{label} must contain one JSON object")
    return parsed


def _assert_no_symlink_components(root: Path, path: Path, *, label: str) -> None:
    current = root
    relative = path.relative_to(root)
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise DataFoundationAuditError(f"{label} must not traverse a symlink")


def _resolve_repo_file(repo_root: Path, relative_path: str, *, label: str) -> Path:
    canonical = _canonical_relative_path(relative_path, label=label)
    root = repo_root.resolve()
    candidate = root.joinpath(*PurePosixPath(canonical).parts)
    _assert_no_symlink_components(root, candidate, label=label)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise DataFoundationAuditError(f"{label} is missing") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise DataFoundationAuditError(f"{label} must resolve to a regular repository file")
    return resolved


def _argument_relative_path(repo_root: Path, path: Path, *, label: str) -> str:
    root = repo_root.resolve()
    if not path.is_absolute():
        _canonical_relative_path(path.as_posix(), label=label)
    candidate = path if path.is_absolute() else root / path
    lexical = Path(os.path.abspath(candidate))
    if not lexical.is_relative_to(root):
        raise DataFoundationAuditError(f"{label} must stay inside the repository")
    _assert_no_symlink_components(root, lexical, label=label)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise DataFoundationAuditError(f"{label} is missing") from exc
    if not resolved.is_relative_to(root):
        raise DataFoundationAuditError(f"{label} must stay inside the repository")
    _assert_no_symlink_components(root, resolved, label=label)
    return _canonical_relative_path(resolved.relative_to(root).as_posix(), label=label)


def verify_file(
    repo_root: Path,
    relative_path: str,
    *,
    expected_sha256: str,
    expected_bytes: int | None = None,
    label: str,
) -> tuple[Path, ArtifactEvidence]:
    expected_digest = _require_sha256(expected_sha256, f"{label} expected SHA-256")
    path = _resolve_repo_file(repo_root, relative_path, label=label)
    size = path.stat().st_size
    if expected_bytes is not None and size != _require_non_negative_int(
        expected_bytes, f"{label} expected bytes"
    ):
        raise DataFoundationAuditError(f"{label} byte size does not match its frozen pin")
    actual_digest = _sha256_file(path)
    if actual_digest != expected_digest:
        raise DataFoundationAuditError(f"{label} SHA-256 does not match its frozen pin")
    return path, ArtifactEvidence(relative_path, size, actual_digest)


def load_source_descriptor(
    repo_root: Path,
    relative_path: str,
    *,
    expected_sha256: str,
    expected_capabilities: SourceCapabilities | None = None,
) -> tuple[SourceDescriptor, ArtifactEvidence]:
    path, evidence = verify_file(
        repo_root,
        relative_path,
        expected_sha256=expected_sha256,
        label="source descriptor",
    )
    payload = load_strict_json(path, label="source descriptor")
    try:
        descriptor = SourceDescriptor.from_dict(payload)
    except (TypeError, ValueError) as exc:
        raise DataFoundationAuditError(f"source descriptor is invalid: {exc}") from exc
    if payload != descriptor.to_dict():
        raise DataFoundationAuditError("source descriptor is not in canonical schema form")
    if expected_capabilities is not None and descriptor.capabilities != expected_capabilities:
        raise DataFoundationAuditError(
            "source descriptor capability contract disagrees with the active reader"
        )
    return descriptor, evidence


def _status_counts(counter: Mapping[str, int]) -> dict[str, int]:
    return {status.value: int(counter.get(status.value, 0)) for status in LabelStatus}


def _position_target_counts(rows: Sequence[Any]) -> dict[str, object]:
    status_counter: Counter[str] = Counter(row.status.value for row in rows)
    return {
        "row_count": len(rows),
        "status_counts": _status_counts(status_counter),
    }


def _contains_absolute_local_path(value: str) -> bool:
    if re.search(r"(?i)(?<![A-Za-z0-9])[A-Z]:[\\/]", value):
        return True
    stripped = value.strip(" \t\r\n\"'()[]{}<>,;:")
    return bool(
        stripped
        and (
            PurePosixPath(stripped).is_absolute()
            or PureWindowsPath(stripped).is_absolute()
        )
    )


def _assert_aggregate_safe(value: Any, *, label: str = "audit") -> None:
    forbidden_identity_keys = {
        "task_id",
        "trajectory_id",
        "run_id",
        "condition_id",
        "raw_message",
        "raw_messages",
        "messages",
    }
    if isinstance(value, str):
        if _contains_absolute_local_path(value):
            raise DataFoundationAuditError(f"{label} contains an absolute local path")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in forbidden_identity_keys:
                raise DataFoundationAuditError(
                    f"{label} contains forbidden row-level identity key {key!r}"
                )
            _assert_aggregate_safe(item, label=f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_aggregate_safe(item, label=f"{label}[{index}]")


def build_source_audit(
    *,
    source_name: str,
    trajectories: Iterable[Trajectory],
    descriptor: SourceDescriptor,
    artifacts: Mapping[str, ArtifactEvidence],
    build_command: str = BUILD_COMMAND,
) -> dict[str, Any]:
    """Build one aggregate-only schema-v2 source audit from canonical trajectories."""

    if not re.fullmatch(r"[a-z][a-z0-9_]*", source_name):
        raise DataFoundationAuditError("source_name must be a lowercase identifier")
    if not artifacts or "descriptor" not in artifacts:
        raise DataFoundationAuditError("source artifacts must include the descriptor")
    command = _require_text(build_command, "build command")
    _assert_aggregate_safe(command, label="build command")
    resolved_trajectories = tuple(trajectories)
    if not resolved_trajectories:
        raise DataFoundationAuditError("source must contain at least one trajectory")
    if len({item.trajectory_id for item in resolved_trajectories}) != len(
        resolved_trajectories
    ):
        raise DataFoundationAuditError("source contains duplicate trajectory identities")

    dataset = build_capability_supervised_dataset(
        resolved_trajectories,
        descriptor,
    )
    if dataset.schema_version != CAPABILITY_DATASET_SCHEMA_VERSION:
        raise DataFoundationAuditError("capability builder did not produce schema v2")
    if dataset.source_descriptor_hash != descriptor.descriptor_hash:
        raise DataFoundationAuditError("dataset source descriptor hash does not close")
    if dataset.capability_contract_hash != descriptor.capabilities.contract_hash:
        raise DataFoundationAuditError("dataset capability contract hash does not close")

    decisions = {
        (position, target): decide_target_capability(
            descriptor.capabilities,
            position,
            target,
        )
        for position in PredictionPosition
        for target in PredictionTarget
    }
    rows_by_cell: dict[tuple[PredictionPosition, PredictionTarget], list[Any]] = {
        key: [] for key in decisions
    }
    for row in dataset.rows:
        key = (row.point.position, row.point.target)
        if key not in rows_by_cell or not decisions[key].available:
            raise DataFoundationAuditError(
                "dataset emitted a row for a capability-gated position/target cell"
            )
        rows_by_cell[key].append(row)

    by_position_target = []
    for position in PredictionPosition:
        for target in PredictionTarget:
            cell = _position_target_counts(rows_by_cell[(position, target)])
            by_position_target.append(
                {
                    "position": position.value,
                    "target": target.value,
                    **cell,
                }
            )

    by_position = []
    for position in PredictionPosition:
        rows = [row for row in dataset.rows if row.point.position == position]
        by_position.append({"position": position.value, **_position_target_counts(rows)})
    by_target = []
    for target in PredictionTarget:
        rows = [row for row in dataset.rows if row.point.target == target]
        by_target.append({"target": target.value, **_position_target_counts(rows)})

    summary = {
        "source_name": source_name,
        "source_descriptor": descriptor.to_dict(),
        "source_descriptor_hash": descriptor.descriptor_hash,
        "capability_contract_hash": descriptor.capabilities.contract_hash,
        "capability_decision_matrix": [
            decisions[(position, target)].to_dict()
            for position in PredictionPosition
            for target in PredictionTarget
        ],
        "dataset": {
            "dataset_id": dataset.dataset_id,
            "schema_version": dataset.schema_version,
            "source_descriptor_hash": dataset.source_descriptor_hash,
            "capability_contract_hash": dataset.capability_contract_hash,
            **_position_target_counts(dataset.rows),
            "by_position": by_position,
            "by_target": by_target,
            "by_position_target": by_position_target,
        },
        "identity_counts": {
            "task_count": len({item.task_id for item in resolved_trajectories}),
            "trajectory_count": len(resolved_trajectories),
            "run_count": len({item.run_id for item in resolved_trajectories}),
            "condition_count": len(
                {item.condition_id for item in resolved_trajectories}
            ),
        },
        "artifacts": {
            name: artifacts[name].to_dict() for name in sorted(artifacts)
        },
        "build_command": command,
    }
    _assert_aggregate_safe(summary)
    return summary


def build_data_foundation_audit(
    source_summaries: Mapping[str, Mapping[str, Any]],
    *,
    git_commit: str,
    source_tree_sha256: str,
    runtime: Mapping[str, str],
    build_command: str = BUILD_COMMAND,
) -> dict[str, Any]:
    """Assemble deterministic source summaries and bind them with one payload hash."""

    if not source_summaries:
        raise DataFoundationAuditError("at least one source summary is required")
    resolved_git_commit = _require_git_commit(git_commit)
    tree_hash = _require_sha256(source_tree_sha256, "source tree SHA-256")
    runtime_output = {
        str(key): _require_text(value, f"runtime.{key}")
        for key, value in sorted(runtime.items())
    }
    if not runtime_output:
        raise DataFoundationAuditError("runtime must not be empty")
    sources: dict[str, Mapping[str, Any]] = {}
    for name in sorted(source_summaries):
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise DataFoundationAuditError("source summary key is invalid")
        summary = source_summaries[name]
        if summary.get("source_name") != name:
            raise DataFoundationAuditError("source summary key/name mismatch")
        sources[name] = summary
    payload: dict[str, Any] = {
        "data_foundation_v2_audit_schema_version": AUDIT_SCHEMA_VERSION,
        "dataset_schema_version": CAPABILITY_DATASET_SCHEMA_VERSION,
        "source_count": len(sources),
        "sources": sources,
        "implementation": {
            "git_commit": resolved_git_commit,
            "git_source_binding_policy": "tracked_clean_head_blob_tree_v1",
            "source_tree_sha256": tree_hash,
            "runtime": runtime_output,
        },
        "build_command": _require_text(build_command, "build command"),
    }
    _assert_aggregate_safe(payload)
    payload["audit_payload_sha256"] = _semantic_sha256(payload)
    return payload


def verify_audit_payload(value: Mapping[str, Any]) -> None:
    declared = _require_sha256(
        value.get("audit_payload_sha256"), "audit payload SHA-256"
    )
    payload = dict(value)
    payload.pop("audit_payload_sha256")
    if _semantic_sha256(payload) != declared:
        raise DataFoundationAuditError("audit payload SHA-256 does not match")


def _update_framed_hash(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def _framed_file_index_evidence(
    *,
    scope_path: str,
    files: Mapping[str, tuple[int, str]],
) -> ArtifactEvidence:
    if not files:
        raise DataFoundationAuditError("framed artifact index must not be empty")
    digest = hashlib.sha256(b"data-foundation-file-index-v1\0")
    total_bytes = 0
    for relative_path in sorted(files):
        path = _canonical_relative_path(relative_path, label="indexed artifact path")
        size, sha256 = files[relative_path]
        total_bytes += _require_non_negative_int(size, "indexed artifact bytes")
        _update_framed_hash(digest, path.encode("utf-8"))
        _update_framed_hash(digest, str(size).encode("ascii"))
        _update_framed_hash(
            digest,
            bytes.fromhex(_require_sha256(sha256, "indexed artifact SHA-256")),
        )
    return ArtifactEvidence(
        path=_canonical_relative_path(scope_path, label="artifact scope path"),
        bytes=total_bytes,
        sha256=digest.hexdigest(),
        file_count=len(files),
        sha256_kind="framed_file_index_v1",
    )


def _verify_embedded_payload_hash(value: Mapping[str, Any], *, label: str) -> None:
    declared = _require_sha256(
        value.get("audit_payload_sha256"), f"{label}.audit_payload_sha256"
    )
    payload = dict(value)
    payload.pop("audit_payload_sha256")
    if _semantic_sha256(payload) != declared:
        raise DataFoundationAuditError(f"{label} embedded payload SHA-256 is invalid")


def _load_manifest(path: Path) -> dict[str, int]:
    entries: dict[str, int] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise DataFoundationAuditError(
                        f"BAGEN manifest line {line_number} is blank"
                    )
                value = _strict_json_loads(
                    line,
                    label=f"BAGEN manifest line {line_number}",
                )
                entry = _require_mapping(
                    value, f"BAGEN manifest line {line_number}"
                )
                relative = _canonical_relative_path(
                    entry.get("path"),
                    label=f"BAGEN manifest line {line_number}.path",
                )
                if relative in entries:
                    raise DataFoundationAuditError(
                        f"BAGEN manifest repeats path {relative!r}"
                    )
                entries[relative] = _require_non_negative_int(
                    entry.get("size_bytes"),
                    f"BAGEN manifest line {line_number}.size_bytes",
                )
    except (OSError, UnicodeError) as exc:
        raise DataFoundationAuditError("cannot read BAGEN manifest") from exc
    return entries


def _family_raw_index(
    repo_root: Path,
    combined: Mapping[str, Any],
) -> tuple[dict[str, tuple[int, str]], ArtifactEvidence]:
    raw_index: dict[str, tuple[int, str]] = {}
    audit_index: dict[str, tuple[int, str]] = {}
    families = _require_list(combined.get("families"), "BAGEN combined families")
    if len(families) != 5:
        raise DataFoundationAuditError("BAGEN combined audit must contain five families")
    for index, raw_family in enumerate(families):
        family = _require_mapping(raw_family, f"BAGEN family {index}")
        family_name = _require_text(family.get("family"), f"BAGEN family {index}.family")
        audit_relative = _canonical_relative_path(
            family.get("audit_path"),
            label=f"BAGEN family {family_name}.audit_path",
        )
        audit_sha = _require_sha256(
            family.get("audit_sha256"), f"BAGEN family {family_name}.audit_sha256"
        )
        audit_bytes = _require_non_negative_int(
            family.get("audit_bytes"), f"BAGEN family {family_name}.audit_bytes"
        )
        audit_path, _ = verify_file(
            repo_root,
            audit_relative,
            expected_sha256=audit_sha,
            expected_bytes=audit_bytes,
            label=f"BAGEN family audit {family_name}",
        )
        audit_index[audit_relative] = (audit_bytes, audit_sha)
        audit = load_strict_json(audit_path, label=f"BAGEN family audit {family_name}")
        _require_source_id(
            audit,
            expected=BagenSwebenchReader.source_id,
            label="BAGEN family audit",
        )
        family_root = _canonical_relative_path(
            family.get("local_relative_root"),
            label=f"BAGEN family {family_name}.local_relative_root",
        )
        if PurePosixPath(family_root).name != audit.get("family_root"):
            raise DataFoundationAuditError("BAGEN family root/audit mismatch")
        raw_files = _require_list(audit.get("raw_files"), "BAGEN family raw_files")
        source_hashes = _require_mapping(
            audit.get("source_hashes"), "BAGEN family source_hashes"
        )
        family_paths: set[str] = set()
        family_bytes = 0
        for raw_index_number, raw_value in enumerate(raw_files):
            raw = _require_mapping(raw_value, "BAGEN family raw file")
            family_relative = _canonical_relative_path(
                raw.get("path"),
                label=f"BAGEN family raw_files[{raw_index_number}].path",
            )
            if family_relative in family_paths:
                raise DataFoundationAuditError("BAGEN family audit repeats a raw path")
            family_paths.add(family_relative)
            raw_sha = _require_sha256(
                raw.get("sha256"), "BAGEN family raw SHA-256"
            )
            if source_hashes.get(family_relative) != raw_sha:
                raise DataFoundationAuditError(
                    "BAGEN family raw_files/source_hashes mismatch"
                )
            raw_bytes = _require_non_negative_int(
                raw.get("bytes"), "BAGEN family raw bytes"
            )
            family_bytes += raw_bytes
            repo_relative = f"{family_root}/{family_relative}"
            if repo_relative in raw_index:
                raise DataFoundationAuditError("BAGEN family audits overlap raw paths")
            raw_index[repo_relative] = (raw_bytes, raw_sha)
        if set(source_hashes) != family_paths:
            raise DataFoundationAuditError(
                "BAGEN family source_hashes contains missing or extra paths"
            )
        if len(raw_files) != _require_non_negative_int(
            audit.get("raw_file_count"), "BAGEN family raw_file_count"
        ) or family_bytes != _require_non_negative_int(
            audit.get("raw_bytes"), "BAGEN family raw_bytes"
        ):
            raise DataFoundationAuditError("BAGEN family raw counts do not close")
    return raw_index, _framed_file_index_evidence(
        scope_path="workspace/external/bagen/audits",
        files=audit_index,
    )


def load_bagen_source_summary(
    repo_root: Path,
    *,
    descriptor_relative: str = DEFAULT_BAGEN_DESCRIPTOR.as_posix(),
    combined_audit_relative: str = DEFAULT_BAGEN_COMBINED_AUDIT.as_posix(),
) -> dict[str, Any]:
    descriptor, descriptor_evidence = load_source_descriptor(
        repo_root,
        descriptor_relative,
        expected_sha256=BAGEN_DESCRIPTOR_SHA256,
        expected_capabilities=BagenSwebenchReader.capabilities,
    )
    if descriptor.source_id != BagenSwebenchReader.source_id:
        raise DataFoundationAuditError("BAGEN descriptor source_id mismatch")
    combined_path, combined_evidence = verify_file(
        repo_root,
        combined_audit_relative,
        expected_sha256=BAGEN_COMBINED_AUDIT_SHA256,
        label="BAGEN combined audit",
    )
    combined = load_strict_json(combined_path, label="BAGEN combined audit")
    _verify_embedded_payload_hash(combined, label="BAGEN combined audit")
    _require_source_id(
        combined,
        expected=BAGEN_COMBINED_AUDIT_SOURCE_ID,
        label="BAGEN combined audit",
    )

    manifest_info = _require_mapping(combined.get("manifest"), "BAGEN manifest")
    manifest_raw = _require_mapping(manifest_info.get("raw"), "BAGEN manifest.raw")
    manifest_sha = _require_sha256(
        manifest_raw.get("sha256"), "BAGEN combined manifest SHA-256"
    )
    if manifest_sha != descriptor.manifest_sha256 or manifest_sha != BAGEN_MANIFEST_SHA256:
        raise DataFoundationAuditError("BAGEN descriptor/combined manifest SHA-256 mismatch")
    if descriptor.manifest_path != manifest_raw.get("path"):
        raise DataFoundationAuditError("BAGEN descriptor/combined manifest path mismatch")
    manifest_path, manifest_evidence = verify_file(
        repo_root,
        descriptor.manifest_path,
        expected_sha256=manifest_sha,
        expected_bytes=_require_non_negative_int(
            manifest_raw.get("bytes"), "BAGEN manifest bytes"
        ),
        label="BAGEN manifest",
    )
    manifest_entries = _load_manifest(manifest_path)
    if len(manifest_entries) != _require_non_negative_int(
        manifest_raw.get("file_count"), "BAGEN manifest file_count"
    ) or sum(manifest_entries.values()) != _require_non_negative_int(
        manifest_raw.get("total_bytes"), "BAGEN manifest total_bytes"
    ):
        raise DataFoundationAuditError("BAGEN manifest aggregate counts do not close")

    family_raw_index, family_audit_evidence = _family_raw_index(repo_root, combined)
    manifest_parent = PurePosixPath(descriptor.manifest_path).parent.as_posix()
    manifest_trajectories = {
        f"{manifest_parent}/{relative}": size
        for relative, size in manifest_entries.items()
        if relative.endswith(".traj.json")
    }
    if set(manifest_trajectories) != set(family_raw_index):
        raise DataFoundationAuditError(
            "BAGEN manifest and frozen family audits disagree on trajectory paths"
        )
    if len(manifest_trajectories) != EXPECTED_BAGEN_TRAJECTORIES:
        raise DataFoundationAuditError("BAGEN trajectory count is not the frozen 316")
    if len(manifest_trajectories) != _require_non_negative_int(
        manifest_raw.get("traj_json_count"), "BAGEN manifest traj_json_count"
    ) or sum(manifest_trajectories.values()) != _require_non_negative_int(
        manifest_raw.get("traj_json_bytes"), "BAGEN manifest traj_json_bytes"
    ):
        raise DataFoundationAuditError("BAGEN trajectory manifest totals do not close")

    verified_raw: dict[str, tuple[int, str]] = {}
    trajectory_paths: list[Path] = []
    for relative_path in sorted(family_raw_index):
        expected_bytes, expected_sha = family_raw_index[relative_path]
        if manifest_trajectories[relative_path] != expected_bytes:
            raise DataFoundationAuditError("BAGEN manifest/family raw byte mismatch")
        raw_path, _ = verify_file(
            repo_root,
            relative_path,
            expected_sha256=expected_sha,
            expected_bytes=expected_bytes,
            label="BAGEN raw trajectory",
        )
        verified_raw[relative_path] = (expected_bytes, expected_sha)
        trajectory_paths.append(raw_path)

    raw_evidence = _framed_file_index_evidence(
        scope_path="workspace/external/bagen/origin",
        files=verified_raw,
    )
    combined_counts = _require_mapping(combined.get("counts"), "BAGEN counts")
    if raw_evidence.file_count != _require_non_negative_int(
        combined_counts.get("raw_file_count"), "BAGEN combined raw_file_count"
    ) or raw_evidence.bytes != _require_non_negative_int(
        combined_counts.get("raw_bytes"), "BAGEN combined raw_bytes"
    ):
        raise DataFoundationAuditError("BAGEN combined raw counts do not close")

    reader = BagenSwebenchReader()
    summary = build_source_audit(
        source_name="bagen_swebench",
        trajectories=(reader.read(path) for path in trajectory_paths),
        descriptor=descriptor,
        artifacts={
            "combined_audit": combined_evidence,
            "descriptor": descriptor_evidence,
            "family_audits": family_audit_evidence,
            "manifest": manifest_evidence,
            "raw_trajectories": raw_evidence,
        },
    )
    identity = _require_mapping(summary["identity_counts"], "BAGEN identity counts")
    expected_identity = {
        "task_count": _require_non_negative_int(
            combined_counts.get("task_id_count"), "BAGEN task_id_count"
        ),
        "trajectory_count": _require_non_negative_int(
            combined_counts.get("trajectory_id_count"), "BAGEN trajectory_id_count"
        ),
        "run_count": _require_non_negative_int(
            combined_counts.get("run_id_count"), "BAGEN run_id_count"
        ),
        "condition_count": _require_non_negative_int(
            combined_counts.get("condition_id_count"), "BAGEN condition_id_count"
        ),
    }
    if dict(identity) != expected_identity:
        raise DataFoundationAuditError("BAGEN schema-v2 identities disagree with frozen audit")
    return summary


def load_spend_source_summary(
    repo_root: Path,
    *,
    descriptor_relative: str = DEFAULT_SPEND_DESCRIPTOR.as_posix(),
) -> dict[str, Any]:
    descriptor, descriptor_evidence = load_source_descriptor(
        repo_root,
        descriptor_relative,
        expected_sha256=SPEND_DESCRIPTOR_SHA256,
        expected_capabilities=OpenHandsArchiveReader.capabilities,
    )
    if descriptor.source_id != OpenHandsArchiveReader.source_id:
        raise DataFoundationAuditError("Spend descriptor source_id mismatch")
    inventory_path, inventory_evidence = verify_file(
        repo_root,
        descriptor.manifest_path,
        expected_sha256=descriptor.manifest_sha256,
        label="Spend inventory",
    )
    inventory = load_strict_json(inventory_path, label="Spend inventory")
    _require_source_id(
        inventory,
        expected=SPEND_INVENTORY_SOURCE_ID,
        label="Spend inventory",
    )
    if inventory.get("resolved_revision") != descriptor.revision:
        raise DataFoundationAuditError("Spend inventory revision mismatch")
    archive_relative = _canonical_relative_path(
        inventory.get("archive_path"), label="Spend archive path"
    )
    archive_sha = _require_sha256(
        inventory.get("archive_sha256"), "Spend archive SHA-256"
    )
    archive_bytes = _require_non_negative_int(
        inventory.get("archive_bytes"), "Spend archive bytes"
    )
    if archive_sha != SPEND_ARCHIVE_SHA256 or archive_bytes != SPEND_ARCHIVE_BYTES:
        raise DataFoundationAuditError("Spend archive identity is not the frozen source")
    archive_path, archive_evidence = verify_file(
        repo_root,
        archive_relative,
        expected_sha256=archive_sha,
        expected_bytes=archive_bytes,
        label="Spend archive",
    )
    expected_trajectory_count = _require_non_negative_int(
        inventory.get("trajectory_count"), "Spend trajectory_count"
    )
    if expected_trajectory_count != EXPECTED_SPEND_TRAJECTORIES:
        raise DataFoundationAuditError("Spend trajectory count is not the frozen 2000")

    reader = OpenHandsArchiveReader()
    summary = build_source_audit(
        source_name="spend_openhands",
        trajectories=reader.iter_archive(
            archive_path,
            OpenHandsArchiveMetadata(archive_identity=archive_sha),
        ),
        descriptor=descriptor,
        artifacts={
            "archive": archive_evidence,
            "descriptor": descriptor_evidence,
            "inventory": inventory_evidence,
        },
    )
    identity = _require_mapping(summary["identity_counts"], "Spend identity counts")
    expected_identity = {
        "task_count": _require_non_negative_int(
            inventory.get("task_count"), "Spend task_count"
        ),
        "trajectory_count": expected_trajectory_count,
        "run_count": _require_non_negative_int(
            inventory.get("run_count"), "Spend run_count"
        ),
    }
    for key, value in expected_identity.items():
        if identity.get(key) != value:
            raise DataFoundationAuditError(
                f"Spend schema-v2 {key} disagrees with frozen inventory"
            )
    return summary


def source_tree_sha256(repo_root: Path) -> str:
    root = repo_root.resolve()
    paths = _relevant_workspace_source_paths(root)
    file_hashes = {
        path.relative_to(root).as_posix(): _sha256_file(path) for path in paths
    }
    return _source_tree_hash_from_file_hashes(file_hashes)


def _is_relevant_source_path(path: str) -> bool:
    return path == "scripts/audit_data_foundation_v2.py" or (
        path.startswith("src/token_prediction/") and path.endswith(".py")
    )


def _relevant_workspace_source_paths(repo_root: Path) -> tuple[Path, ...]:
    root = repo_root.resolve()
    package = root / "src" / "token_prediction"
    script = root / "scripts" / "audit_data_foundation_v2.py"
    if not package.is_dir() or not script.is_file():
        raise DataFoundationAuditError("relevant source tree is incomplete")
    if any(path.is_symlink() for path in package.rglob("*")):
        raise DataFoundationAuditError("relevant source tree must not contain symlinks")
    paths = tuple(
        sorted(
            [path for path in package.rglob("*.py") if path.is_file()] + [script],
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )
    if not paths:
        raise DataFoundationAuditError("source tree contains no Python files")
    for path in paths:
        _assert_no_symlink_components(root, path, label="source tree file")
    return paths


def _source_tree_hash_from_file_hashes(
    file_hashes: Mapping[str, str],
) -> str:
    if not file_hashes:
        raise DataFoundationAuditError("source tree file index must not be empty")
    digest = hashlib.sha256(b"data-foundation-source-tree-v1\0")
    for relative_path in sorted(file_hashes):
        canonical = _canonical_relative_path(
            relative_path, label="source tree file path"
        )
        if not _is_relevant_source_path(canonical):
            raise DataFoundationAuditError(
                "source tree file index contains an irrelevant path"
            )
        digest_value = _require_sha256(
            file_hashes[relative_path], "source tree file SHA-256"
        )
        _update_framed_hash(digest, canonical.encode("utf-8"))
        _update_framed_hash(digest, bytes.fromhex(digest_value))
    return digest.hexdigest()


def runtime_info() -> dict[str, str]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": ".".join(str(value) for value in sys.version_info[:3]),
    }


def _default_git_executable() -> str:
    discovered = shutil.which("git")
    if discovered:
        return discovered
    if os.name == "nt":
        candidates = [
            Path(os.environ.get("ProgramFiles", "C:/Program Files"))
            / "Git"
            / "cmd"
            / "git.exe",
            Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
            / "Git"
            / "cmd"
            / "git.exe",
            Path(os.environ.get("LOCALAPPDATA", ""))
            / "Programs"
            / "Git"
            / "cmd"
            / "git.exe",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
    raise DataFoundationAuditError("Git executable is unavailable")


def resolve_git_commit(
    repo_root: Path,
    *,
    git_executable: str | Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> str:
    """Resolve and type-check the repository's current commit without a shell."""

    root = repo_root.resolve()
    if not root.is_dir():
        raise DataFoundationAuditError("Git repository root is missing")
    executable = str(git_executable or _default_git_executable())

    def run_git(arguments: Sequence[str], *, label: str) -> str:
        try:
            result = runner(
                [executable, "-C", str(root), *arguments],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="strict",
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
            raise DataFoundationAuditError(f"cannot resolve {label} with Git") from exc
        if result.returncode != 0:
            raise DataFoundationAuditError(f"Git could not resolve {label}")
        if not isinstance(result.stdout, str):
            raise DataFoundationAuditError(f"Git returned invalid {label} output")
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            raise DataFoundationAuditError(f"Git returned ambiguous {label} output")
        return lines[0]

    top_level = run_git(("rev-parse", "--show-toplevel"), label="repository root")
    try:
        resolved_top_level = Path(top_level).resolve(strict=True)
    except OSError as exc:
        raise DataFoundationAuditError("Git repository root is not a resolvable path") from exc
    if resolved_top_level != root:
        raise DataFoundationAuditError(
            "configured repository root is not the Git worktree top-level"
        )
    commit = run_git(
        ("rev-parse", "--verify", "HEAD^{commit}"),
        label="HEAD commit",
    )
    return _require_git_commit(commit, "resolved Git commit")


def _parse_nul_paths(value: str, *, label: str) -> tuple[str, ...]:
    if not value:
        return ()
    if not value.endswith("\0"):
        raise DataFoundationAuditError(f"Git returned truncated {label} output")
    paths = value[:-1].split("\0")
    if any(not path for path in paths):
        raise DataFoundationAuditError(f"Git returned an empty {label} path")
    canonical = tuple(
        _canonical_relative_path(path, label=f"Git {label} path") for path in paths
    )
    if len(set(canonical)) != len(canonical):
        raise DataFoundationAuditError(f"Git returned duplicate {label} paths")
    return canonical


def verify_git_source_binding(
    repo_root: Path,
    *,
    git_commit: str,
    workspace_source_tree_sha256: str,
    git_executable: str | Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> None:
    """Bind the relevant clean worktree bytes to the exact current HEAD blobs."""

    root = repo_root.resolve()
    expected_commit = _require_git_commit(git_commit)
    expected_workspace_hash = _require_sha256(
        workspace_source_tree_sha256, "workspace source tree SHA-256"
    )
    executable = str(git_executable or _default_git_executable())
    current_commit = resolve_git_commit(
        root,
        git_executable=executable,
        runner=runner,
    )
    if current_commit != expected_commit:
        raise DataFoundationAuditError("Git HEAD changed during source binding")

    actual_workspace_hash = source_tree_sha256(root)
    if actual_workspace_hash != expected_workspace_hash:
        raise DataFoundationAuditError(
            "workspace source tree changed during source binding"
        )

    pathspec = (
        "src/token_prediction",
        "scripts/audit_data_foundation_v2.py",
    )

    def run_git_text(arguments: Sequence[str], *, label: str) -> str:
        try:
            result = runner(
                [executable, "-C", str(root), *arguments],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="strict",
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
            raise DataFoundationAuditError(f"cannot inspect Git {label}") from exc
        if result.returncode != 0 or not isinstance(result.stdout, str):
            raise DataFoundationAuditError(f"Git could not inspect {label}")
        return result.stdout

    staged_paths = _parse_nul_paths(
        run_git_text(
            (
                "diff",
                "--cached",
                "--name-only",
                "-z",
                expected_commit,
                "--",
                *pathspec,
            ),
            label="staged relevant changes",
        ),
        label="staged change",
    )
    if any(_is_relevant_source_path(path) for path in staged_paths):
        raise DataFoundationAuditError("relevant source tree has staged changes")

    unstaged_paths = _parse_nul_paths(
        run_git_text(
            ("diff", "--name-only", "-z", "--", *pathspec),
            label="unstaged relevant changes",
        ),
        label="unstaged change",
    )
    if any(_is_relevant_source_path(path) for path in unstaged_paths):
        raise DataFoundationAuditError("relevant source tree has unstaged changes")

    untracked_paths = _parse_nul_paths(
        run_git_text(
            (
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                *pathspec,
            ),
            label="untracked relevant files",
        ),
        label="untracked file",
    )
    if any(_is_relevant_source_path(path) for path in untracked_paths):
        raise DataFoundationAuditError("relevant source tree has untracked files")

    head_paths = _parse_nul_paths(
        run_git_text(
            (
                "ls-tree",
                "-r",
                "-z",
                "--name-only",
                expected_commit,
                "--",
                *pathspec,
            ),
            label="HEAD relevant paths",
        ),
        label="HEAD tree",
    )
    relevant_head_paths = tuple(
        sorted(path for path in head_paths if _is_relevant_source_path(path))
    )
    relevant_workspace_paths = tuple(
        path.relative_to(root).as_posix()
        for path in _relevant_workspace_source_paths(root)
    )
    if relevant_head_paths != relevant_workspace_paths:
        raise DataFoundationAuditError(
            "relevant workspace source paths are not all tracked by HEAD"
        )

    head_file_hashes: dict[str, str] = {}
    for relative_path in relevant_head_paths:
        try:
            result = runner(
                [
                    executable,
                    "-C",
                    str(root),
                    "cat-file",
                    "blob",
                    f"{expected_commit}:{relative_path}",
                ],
                capture_output=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise DataFoundationAuditError("cannot read relevant HEAD blob") from exc
        if result.returncode != 0 or not isinstance(result.stdout, bytes):
            raise DataFoundationAuditError("Git could not read a relevant HEAD blob")
        head_file_hashes[relative_path] = hashlib.sha256(result.stdout).hexdigest()
    head_source_tree_hash = _source_tree_hash_from_file_hashes(head_file_hashes)
    if head_source_tree_hash != expected_workspace_hash:
        raise DataFoundationAuditError(
            "HEAD blob source tree hash does not match the clean workspace"
        )


def atomic_write_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    force: bool = False,
) -> None:
    """Atomically create JSON; refuse overwrite unless ``force`` is explicit."""

    destination = path.resolve()
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
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if force:
            os.replace(temporary_path, destination)
        else:
            try:
                os.link(temporary_path, destination)
            except FileExistsError as exc:
                raise DataFoundationAuditError(
                    "audit output already exists; pass --force to replace it"
                ) from exc
            temporary_path.unlink()
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _output_path(repo_root: Path, path: Path) -> Path:
    root = repo_root.resolve()
    if not path.is_absolute():
        _canonical_relative_path(path.as_posix(), label="audit output")
    candidate = path if path.is_absolute() else root / path
    lexical = Path(os.path.abspath(candidate))
    if not lexical.is_relative_to(root):
        raise DataFoundationAuditError("audit output must stay inside the repository")
    _assert_no_symlink_components(root, lexical.parent, label="audit output parent")
    resolved_parent = lexical.parent.resolve()
    if not resolved_parent.is_relative_to(root):
        raise DataFoundationAuditError("audit output must stay inside the repository")
    _assert_no_symlink_components(root, resolved_parent, label="audit output parent")
    relative = lexical.relative_to(root).as_posix()
    _canonical_relative_path(relative, label="audit output")
    if lexical.is_symlink():
        raise DataFoundationAuditError("audit output must not be a symlink")
    return lexical.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the aggregate-only schema-v2 Data Foundation audit from frozen sources."
        )
    )
    parser.add_argument(
        "--source",
        choices=("all", "bagen", "spend"),
        default="all",
        help="source subset; production freeze uses all",
    )
    parser.add_argument(
        "--bagen-descriptor", type=Path, default=DEFAULT_BAGEN_DESCRIPTOR
    )
    parser.add_argument(
        "--bagen-combined-audit", type=Path, default=DEFAULT_BAGEN_COMBINED_AUDIT
    )
    parser.add_argument(
        "--spend-descriptor", type=Path, default=DEFAULT_SPEND_DESCRIPTOR
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="atomically replace an existing output instead of failing closed",
    )
    args = parser.parse_args()

    try:
        git_commit = resolve_git_commit(REPO_ROOT)
        workspace_source_tree_hash = source_tree_sha256(REPO_ROOT)
        verify_git_source_binding(
            REPO_ROOT,
            git_commit=git_commit,
            workspace_source_tree_sha256=workspace_source_tree_hash,
        )
        summaries: dict[str, Mapping[str, Any]] = {}
        if args.source in {"all", "bagen"}:
            summaries["bagen_swebench"] = load_bagen_source_summary(
                REPO_ROOT,
                descriptor_relative=_argument_relative_path(
                    REPO_ROOT, args.bagen_descriptor, label="BAGEN descriptor"
                ),
                combined_audit_relative=_argument_relative_path(
                    REPO_ROOT,
                    args.bagen_combined_audit,
                    label="BAGEN combined audit",
                ),
            )
        if args.source in {"all", "spend"}:
            summaries["spend_openhands"] = load_spend_source_summary(
                REPO_ROOT,
                descriptor_relative=_argument_relative_path(
                    REPO_ROOT, args.spend_descriptor, label="Spend descriptor"
                ),
            )
        verify_git_source_binding(
            REPO_ROOT,
            git_commit=git_commit,
            workspace_source_tree_sha256=workspace_source_tree_hash,
        )
        audit = build_data_foundation_audit(
            summaries,
            git_commit=git_commit,
            source_tree_sha256=workspace_source_tree_hash,
            runtime=runtime_info(),
        )
        output = _output_path(REPO_ROOT, args.output)
        atomic_write_json(output, audit, force=args.force)
    except (DataFoundationAuditError, OSError, ValueError) as exc:
        parser.exit(2, f"Data Foundation v2 audit failed: {exc}\n")
    print(
        json.dumps(
            {
                "audit_payload_sha256": audit["audit_payload_sha256"],
                "source_count": audit["source_count"],
                "sources": sorted(audit["sources"]),
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
