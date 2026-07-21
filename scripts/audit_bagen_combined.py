from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from token_prediction.collection import BagenSwebenchReader, BagenSwebenchSchemaError
from token_prediction.dataset import build_supervised_dataset
from token_prediction.dataset.schema import DATASET_SCHEMA_VERSION
from token_prediction.features import FEATURE_SCHEMA_VERSION
from token_prediction.trajectory import Trajectory


COMBINED_AUDIT_SCHEMA_VERSION = 1
SOURCE_ID = "bagen_swebench_combined_audit_v1"
BAGEN_REPO = "MLL-Lab/BAGEN"
BAGEN_REVISION = "58189576e54b675fdd0e1d6c1c9f189c2992732f"
BAGEN_COMBINED_DATASET_ID = (
    "c845574fd0c0e3da3b6a4d1782787d3d53a1b71db738314836f08419bcb57a60"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BAGEN_ROOT = REPO_ROOT / "workspace" / "external" / "bagen"
DEFAULT_MANIFEST_SUMMARY = BAGEN_ROOT / "manifest_summary.json"
DEFAULT_MANIFEST = BAGEN_ROOT / "manifest.jsonl"
DEFAULT_OUTPUT = BAGEN_ROOT / "combined_swebench_audit.json"

FAMILY_SPECS: dict[str, tuple[str, str]] = {
    "claude-opus4.7": (
        "swebench-origin-claude-opus4.7",
        "claude-opus4.7.json",
    ),
    "claude-sonnet4.6": (
        "swebench-origin-claude-sonnet4.6",
        "claude-sonnet4.6.json",
    ),
    "gemini3.1": ("swebench-origin-gemini3.1", "gemini3.1.json"),
    "gpt5.2instant": (
        "swebench-origin-gpt5.2instant",
        "gpt5.2instant.json",
    ),
    "qwen3-235b": ("swebench-origin-qwen3-235b", "qwen3-235b.json"),
}

DEFAULT_FAMILY_ROOTS = {
    family: BAGEN_ROOT / "origin" / family_root
    for family, (family_root, _) in FAMILY_SPECS.items()
}
DEFAULT_FAMILY_AUDITS = {
    family: BAGEN_ROOT / "audits" / filename
    for family, (_, filename) in FAMILY_SPECS.items()
}

PRODUCTION_EXPECTED_COUNTS = {
    "task_id_count": 64,
    "run_id_count": 316,
    "trajectory_id_count": 316,
    "condition_id_count": 9,
    "dataset_row_count": 45_564,
    "raw_file_count": 316,
    "raw_bytes": 263_785_722,
}
PRODUCTION_TASK_CROSS_FAMILY_DISTRIBUTION = {"4": 4, "5": 60}

SOURCE_PATHS = {
    "reader": "src/token_prediction/collection/bagen_swebench.py",
    "builder": "src/token_prediction/dataset/builder.py",
    "labels": "src/token_prediction/dataset/labels.py",
    "audit": "scripts/audit_bagen_combined.py",
    "family_audit": "scripts/audit_bagen_swebench.py",
}
CONSTRUCTION_COMMAND = (
    "$env:PYTHONPATH='src'; python scripts/audit_bagen_combined.py"
)

_MANIFEST_FIELDS = frozenset(
    {
        "download_url",
        "extension",
        "path",
        "relative_path",
        "size_bytes",
        "top_level",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_ETAG_RE = re.compile(r"[0-9a-f]{40}\Z")
_SAFE_PATH_RE = re.compile(r"[A-Za-z0-9._/-]+\Z")
_READ_BUFFER_BYTES = 1024 * 1024


class BagenCombinedAuditError(ValueError):
    """Raised when the five-family evidence cannot be frozen consistently."""


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
        raise BagenCombinedAuditError(f"value is not canonical JSON: {exc}") from exc
    return rendered.encode("utf-8")


def _semantic_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_BUFFER_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_etag(path: Path) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_BUFFER_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise BagenCombinedAuditError(f"non-finite JSON number is forbidden: {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise BagenCombinedAuditError(f"duplicate JSON field: {key!r}")
        value[key] = item
    return value


def _decode_json(value: str, *, label: str) -> Any:
    try:
        return json.loads(
            value,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise BagenCombinedAuditError(f"{label} is not valid JSON: {exc}") from exc
    except BagenCombinedAuditError as exc:
        raise BagenCombinedAuditError(f"{label}: {exc}") from exc


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise BagenCombinedAuditError(f"{label} is not a file: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = _decode_json(handle.read(), label=label)
    except (OSError, UnicodeError) as exc:
        raise BagenCombinedAuditError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise BagenCombinedAuditError(f"{label} must be a JSON object")
    return value


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BagenCombinedAuditError(f"{label} must be an object")
    return value


def _require_list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise BagenCombinedAuditError(f"{label} must be a list")
    return value


def _require_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BagenCombinedAuditError(f"{label} must be a non-empty string")
    return value


def _require_int(value: Any, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise BagenCombinedAuditError(f"{label} must be a non-negative integer")
    return value


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise BagenCombinedAuditError(f"{label} must be a boolean")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    rendered = _require_text(value, label=label)
    if _SHA256_RE.fullmatch(rendered) is None:
        raise BagenCombinedAuditError(f"{label} must be a lowercase SHA256")
    return rendered


def _canonical_posix_path(value: Any, *, label: str) -> str:
    rendered = _require_text(value, label=label)
    if _SAFE_PATH_RE.fullmatch(rendered) is None:
        raise BagenCombinedAuditError(f"{label} is not a safe path")
    path = PurePosixPath(rendered)
    if (
        path.is_absolute()
        or path.as_posix() != rendered
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise BagenCombinedAuditError(f"{label} is not a canonical relative path")
    return rendered


def _read_manifest(path: Path, *, expected_repo: str) -> dict[str, int]:
    if not path.is_file():
        raise BagenCombinedAuditError(f"raw manifest is not a file: {path}")
    entries: dict[str, int] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                value = _decode_json(raw_line, label=f"manifest line {line_number}")
                item = _require_mapping(value, label=f"manifest line {line_number}")
                if set(item) != _MANIFEST_FIELDS:
                    raise BagenCombinedAuditError(
                        f"manifest line {line_number} has unsupported fields"
                    )
                item_path = _canonical_posix_path(
                    item.get("path"), label=f"manifest line {line_number}.path"
                )
                if item_path in entries:
                    raise BagenCombinedAuditError(
                        f"manifest contains duplicate path: {item_path}"
                    )
                top_level = _require_text(
                    item.get("top_level"),
                    label=f"manifest line {line_number}.top_level",
                )
                if top_level not in {"origin", "estimation"}:
                    raise BagenCombinedAuditError(
                        f"manifest line {line_number}.top_level is unsupported"
                    )
                if not item_path.startswith(f"{top_level}/"):
                    raise BagenCombinedAuditError(
                        f"manifest line {line_number} has inconsistent top_level"
                    )
                relative_path = _canonical_posix_path(
                    item.get("relative_path"),
                    label=f"manifest line {line_number}.relative_path",
                )
                if relative_path != item_path[len(top_level) + 1 :]:
                    raise BagenCombinedAuditError(
                        f"manifest line {line_number} has inconsistent relative_path"
                    )
                extension = _require_text(
                    item.get("extension"),
                    label=f"manifest line {line_number}.extension",
                )
                if extension != PurePosixPath(item_path).suffix:
                    raise BagenCombinedAuditError(
                        f"manifest line {line_number} has inconsistent extension"
                    )
                size_bytes = _require_int(
                    item.get("size_bytes"),
                    label=f"manifest line {line_number}.size_bytes",
                )
                expected_url = (
                    f"https://huggingface.co/datasets/{expected_repo}/resolve/main/"
                    f"{quote(item_path, safe='/')}"
                )
                if item.get("download_url") != expected_url:
                    raise BagenCombinedAuditError(
                        f"manifest line {line_number} has an unexpected download URL"
                    )
                entries[item_path] = size_bytes
    except (OSError, UnicodeError) as exc:
        raise BagenCombinedAuditError(f"cannot read raw manifest: {exc}") from exc
    if not entries:
        raise BagenCombinedAuditError("raw manifest is empty")
    return entries


def _verify_manifest_summary(
    summary_path: Path,
    manifest_path: Path,
    manifest_entries: Mapping[str, int],
    *,
    expected_repo: str,
    expected_revision: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = _load_json_object(summary_path, label="manifest summary")
    expected_source_url = f"https://huggingface.co/datasets/{expected_repo}"
    if summary.get("source_url") != expected_source_url:
        raise BagenCombinedAuditError("manifest summary source_url does not match the Hub repo")
    if summary.get("source_id") != f"{expected_repo}@main":
        raise BagenCombinedAuditError("manifest summary source_id does not match the Hub repo")
    if summary.get("resolved_revision") != expected_revision:
        raise BagenCombinedAuditError("manifest summary revision is not the pinned revision")
    if summary.get("manifest_file") != manifest_path.name:
        raise BagenCombinedAuditError("manifest summary names a different raw manifest")

    manifest_bytes = manifest_path.stat().st_size
    manifest_sha256 = _file_sha256(manifest_path)
    manifest_etag = _git_blob_etag(manifest_path)
    if _require_int(summary.get("manifest_bytes"), label="manifest summary.manifest_bytes") != (
        manifest_bytes
    ):
        raise BagenCombinedAuditError("manifest summary byte count does not match the file")
    if _require_sha256(
        summary.get("manifest_sha256"), label="manifest summary.manifest_sha256"
    ) != manifest_sha256:
        raise BagenCombinedAuditError("manifest summary SHA256 does not match the file")
    declared_etag = _require_text(
        summary.get("manifest_etag"), label="manifest summary.manifest_etag"
    )
    if _GIT_ETAG_RE.fullmatch(declared_etag) is None or declared_etag != manifest_etag:
        raise BagenCombinedAuditError("manifest summary Git ETag does not match the file")

    traj_entries = {
        item_path: size
        for item_path, size in manifest_entries.items()
        if item_path.endswith(".traj.json")
    }
    expected_summary_counts = {
        "file_count": len(manifest_entries),
        "total_bytes": sum(manifest_entries.values()),
        "traj_json_count": len(traj_entries),
        "traj_json_bytes": sum(traj_entries.values()),
    }
    for key, expected in expected_summary_counts.items():
        if _require_int(summary.get(key), label=f"manifest summary.{key}") != expected:
            raise BagenCombinedAuditError(f"manifest summary {key} does not close")

    summary_evidence = {
        "path": "workspace/external/bagen/manifest_summary.json",
        "bytes": summary_path.stat().st_size,
        "sha256": _file_sha256(summary_path),
    }
    manifest_evidence = {
        "path": "workspace/external/bagen/manifest.jsonl",
        "bytes": manifest_bytes,
        "sha256": manifest_sha256,
        "git_blob_etag": manifest_etag,
        **expected_summary_counts,
    }
    return summary_evidence, manifest_evidence


def _update_framed_hash(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def _canonical_trajectory_hash(trajectory: Trajectory) -> str:
    digest = hashlib.sha256(b"bagen-swebench-canonical-trajectory-v1\0")
    for event in trajectory.events:
        _update_framed_hash(digest, _canonical_bytes(event.to_dict()))
    return digest.hexdigest()


def _canonical_family_hash(hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256(b"bagen-swebench-canonical-family-v1\0")
    for relative_path in sorted(hashes):
        _update_framed_hash(digest, relative_path.encode("utf-8"))
        _update_framed_hash(digest, bytes.fromhex(hashes[relative_path]))
    return digest.hexdigest()


def _audit_raw_index(audit: Mapping[str, Any], *, family: str) -> dict[str, Mapping[str, Any]]:
    raw_values = _require_list(audit.get("raw_files"), label=f"{family} audit.raw_files")
    index: dict[str, Mapping[str, Any]] = {}
    for raw_index, raw_value in enumerate(raw_values):
        label = f"{family} audit.raw_files[{raw_index}]"
        raw = _require_mapping(raw_value, label=label)
        relative_path = _canonical_posix_path(raw.get("path"), label=f"{label}.path")
        if relative_path in index:
            raise BagenCombinedAuditError(
                f"{family} audit has duplicate raw path: {relative_path}"
            )
        index[relative_path] = raw
    return index


def _verify_family_audit_header(
    audit: Mapping[str, Any],
    *,
    family: str,
    family_root_name: str,
) -> None:
    if _require_int(audit.get("audit_schema_version"), label=f"{family}.audit_schema_version") != 1:
        raise BagenCombinedAuditError(f"{family} audit schema version is unsupported")
    if audit.get("source_id") != BagenSwebenchReader.source_id:
        raise BagenCombinedAuditError(f"{family} audit source_id is unsupported")
    if audit.get("reader_version") != BagenSwebenchReader.source_id:
        raise BagenCombinedAuditError(f"{family} audit reader_version is unsupported")
    if audit.get("family") != family or audit.get("family_root") != family_root_name:
        raise BagenCombinedAuditError(f"{family} audit identity does not match its input slot")
    if not _require_bool(
        audit.get("canonical_rerun_consistent"),
        label=f"{family}.canonical_rerun_consistent",
    ):
        raise BagenCombinedAuditError(f"{family} audit did not pass its canonical rerun")
    canonical = _require_sha256(
        audit.get("canonical_content_sha256"),
        label=f"{family}.canonical_content_sha256",
    )
    rerun = _require_sha256(
        audit.get("canonical_rerun_content_sha256"),
        label=f"{family}.canonical_rerun_content_sha256",
    )
    if canonical != rerun:
        raise BagenCombinedAuditError(f"{family} canonical rerun hashes disagree")


def _verify_family(
    family: str,
    root: Path,
    audit_path: Path,
    manifest_entries: Mapping[str, int],
) -> tuple[dict[str, Any], list[Trajectory], list[dict[str, str]]]:
    family_root_name, audit_filename = FAMILY_SPECS[family]
    if not root.is_dir():
        raise BagenCombinedAuditError(f"{family} root is not a directory: {root}")
    if root.name != family_root_name:
        raise BagenCombinedAuditError(
            f"{family} root basename must be {family_root_name!r}"
        )
    if audit_path.name != audit_filename:
        raise BagenCombinedAuditError(
            f"{family} audit basename must be {audit_filename!r}"
        )
    audit = _load_json_object(audit_path, label=f"{family} family audit")
    _verify_family_audit_header(
        audit,
        family=family,
        family_root_name=family_root_name,
    )
    audit_raw = _audit_raw_index(audit, family=family)

    raw_paths = sorted(
        (path for path in root.rglob("*.traj.json") if path.is_file()),
        key=lambda item: item.as_posix(),
    )
    if not raw_paths:
        raise BagenCombinedAuditError(f"{family} root has no trajectory files")
    try:
        trajectories = list(BagenSwebenchReader().iter_directory(root))
    except BagenSwebenchSchemaError as exc:
        raise BagenCombinedAuditError(f"{family} reader failed: {exc}") from exc
    if len(raw_paths) != len(trajectories):
        raise BagenCombinedAuditError(f"{family} raw and canonical trajectory counts disagree")

    canonical_hashes: dict[str, str] = {}
    manifest_paths: list[dict[str, str]] = []
    task_ids: set[str] = set()
    run_ids: set[str] = set()
    trajectory_ids: set[str] = set()
    condition_ids: set[str] = set()
    task_trajectory_counts: Counter[str] = Counter()
    raw_bytes = 0
    for raw_path, trajectory in zip(raw_paths, trajectories, strict=True):
        relative_path = raw_path.relative_to(root).as_posix()
        raw_record = audit_raw.get(relative_path)
        if raw_record is None:
            raise BagenCombinedAuditError(
                f"{family} audit is missing raw path {relative_path!r}"
            )
        size_bytes = raw_path.stat().st_size
        sha256 = _file_sha256(raw_path)
        canonical_sha256 = _canonical_trajectory_hash(trajectory)
        checks: tuple[tuple[str, Any, Any], ...] = (
            ("bytes", raw_record.get("bytes"), size_bytes),
            ("sha256", raw_record.get("sha256"), sha256),
            ("task_id", raw_record.get("task_id"), trajectory.task_id),
            ("run_id", trajectory.run_id, f"{family_root_name}/{relative_path}"),
            ("trajectory_id", raw_record.get("trajectory_id"), trajectory.trajectory_id),
            ("condition_id", raw_record.get("condition_id"), trajectory.condition_id),
            (
                "canonical_content_sha256",
                raw_record.get("canonical_content_sha256"),
                canonical_sha256,
            ),
        )
        for field, declared, actual in checks:
            if declared != actual:
                raise BagenCombinedAuditError(
                    f"{family} {relative_path} {field} does not match canonical evidence"
                )
        if not _require_bool(
            raw_record.get("canonical_rerun_consistent"),
            label=f"{family} {relative_path}.canonical_rerun_consistent",
        ):
            raise BagenCombinedAuditError(
                f"{family} {relative_path} did not pass its canonical rerun"
            )
        trajectory_dataset = build_supervised_dataset((trajectory,))
        if _require_int(
            raw_record.get("dataset_row_count"),
            label=f"{family} {relative_path}.dataset_row_count",
        ) != len(trajectory_dataset.rows):
            raise BagenCombinedAuditError(
                f"{family} {relative_path} dataset row count changed"
            )

        hub_path = f"origin/{family_root_name}/{relative_path}"
        manifest_size = manifest_entries.get(hub_path)
        if manifest_size is None:
            raise BagenCombinedAuditError(f"pinned manifest is missing {hub_path!r}")
        if manifest_size != size_bytes:
            raise BagenCombinedAuditError(f"pinned manifest byte count changed for {hub_path!r}")

        canonical_hashes[relative_path] = canonical_sha256
        manifest_paths.append(
            {
                "family": family,
                "path": hub_path,
                "canonical_content_sha256": canonical_sha256,
            }
        )
        raw_bytes += size_bytes
        task_ids.add(trajectory.task_id)
        run_ids.add(trajectory.run_id)
        trajectory_ids.add(trajectory.trajectory_id)
        condition_ids.add(trajectory.condition_id)
        task_trajectory_counts[trajectory.task_id] += 1

    if set(audit_raw) != set(canonical_hashes):
        raise BagenCombinedAuditError(f"{family} audit contains stale or extra raw paths")
    source_hashes = _require_mapping(audit.get("source_hashes"), label=f"{family}.source_hashes")
    expected_source_hashes = {
        relative_path: str(audit_raw[relative_path].get("sha256"))
        for relative_path in sorted(audit_raw)
    }
    if dict(source_hashes) != expected_source_hashes:
        raise BagenCombinedAuditError(f"{family} audit source_hashes index disagrees")

    family_dataset = build_supervised_dataset(trajectories)
    dataset = _require_mapping(audit.get("dataset"), label=f"{family}.dataset")
    family_canonical_sha256 = _canonical_family_hash(canonical_hashes)
    expected_task_counts = {
        task_id: task_trajectory_counts[task_id] for task_id in sorted(task_trajectory_counts)
    }
    aggregate_checks: tuple[tuple[str, Any, Any], ...] = (
        ("raw_file_count", audit.get("raw_file_count"), len(raw_paths)),
        ("raw_bytes", audit.get("raw_bytes"), raw_bytes),
        ("task_count", audit.get("task_count"), len(task_ids)),
        ("trajectory_count", audit.get("trajectory_count"), len(trajectory_ids)),
        ("condition_count", audit.get("condition_count"), len(condition_ids)),
        ("dataset.row_count", dataset.get("row_count"), len(family_dataset.rows)),
        ("dataset.dataset_id", dataset.get("dataset_id"), family_dataset.dataset_id),
        (
            "canonical_content_sha256",
            audit.get("canonical_content_sha256"),
            family_canonical_sha256,
        ),
        ("task_trajectory_counts", audit.get("task_trajectory_counts"), expected_task_counts),
    )
    for field, declared, actual in aggregate_checks:
        if declared != actual:
            raise BagenCombinedAuditError(f"{family} audit {field} does not close")

    family_summary = {
        "family": family,
        "family_root": family_root_name,
        "local_relative_root": f"workspace/external/bagen/origin/{family_root_name}",
        "audit_path": f"workspace/external/bagen/audits/{audit_filename}",
        "audit_bytes": audit_path.stat().st_size,
        "audit_sha256": _file_sha256(audit_path),
        "raw_file_count": len(raw_paths),
        "raw_bytes": raw_bytes,
        "task_count": len(task_ids),
        "run_count": len(run_ids),
        "trajectory_count": len(trajectory_ids),
        "condition_count": len(condition_ids),
        "dataset": {
            "dataset_id": family_dataset.dataset_id,
            "row_count": len(family_dataset.rows),
            "schema_version": DATASET_SCHEMA_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
        },
        "canonical_content_sha256": family_canonical_sha256,
    }
    return family_summary, trajectories, manifest_paths


def _source_file_evidence(
    source_paths: Mapping[str, tuple[str, Path]] | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    resolved_sources = source_paths or {
        name: (relative_path, REPO_ROOT / relative_path)
        for name, relative_path in SOURCE_PATHS.items()
    }
    required = {"reader", "builder", "labels", "audit", "family_audit"}
    if set(resolved_sources) != required:
        raise BagenCombinedAuditError(
            "source_paths must contain reader, builder, labels, audit, and family_audit"
        )
    output: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for name in sorted(resolved_sources):
        relative_path, physical_path = resolved_sources[name]
        safe_path = _canonical_posix_path(relative_path, label=f"source_files.{name}.path")
        if not physical_path.is_file():
            raise BagenCombinedAuditError(f"source file does not exist: {physical_path}")
        sha256 = _file_sha256(physical_path)
        output[name] = {
            "path": safe_path,
            "bytes": physical_path.stat().st_size,
            "sha256": sha256,
        }
        hashes[safe_path] = sha256
    return output, dict(sorted(hashes.items()))


def _normalize_expected_counts(value: Mapping[str, int]) -> dict[str, int]:
    if set(value) != set(PRODUCTION_EXPECTED_COUNTS):
        raise BagenCombinedAuditError("expected_counts has unsupported keys")
    return {
        key: _require_int(value[key], label=f"expected_counts.{key}")
        for key in PRODUCTION_EXPECTED_COUNTS
    }


def build_combined_audit(
    manifest_summary_path: Path = DEFAULT_MANIFEST_SUMMARY,
    manifest_path: Path = DEFAULT_MANIFEST,
    family_roots: Mapping[str, Path] = DEFAULT_FAMILY_ROOTS,
    family_audits: Mapping[str, Path] = DEFAULT_FAMILY_AUDITS,
    *,
    expected_repo: str = BAGEN_REPO,
    expected_revision: str = BAGEN_REVISION,
    expected_counts: Mapping[str, int] = PRODUCTION_EXPECTED_COUNTS,
    expected_task_cross_family_distribution: Mapping[
        str, int
    ] = PRODUCTION_TASK_CROSS_FAMILY_DISTRIBUTION,
    expected_dataset_id: str | None = BAGEN_COMBINED_DATASET_ID,
    source_paths: Mapping[str, tuple[str, Path]] | None = None,
) -> dict[str, Any]:
    """Build and strictly verify the deterministic five-family freeze payload.

    Family audit JSON is loaded in memory, but raw ``*.traj.json`` files are
    consumed exclusively through :class:`BagenSwebenchReader`'s streaming parser.
    """

    if set(family_roots) != set(FAMILY_SPECS) or set(family_audits) != set(FAMILY_SPECS):
        raise BagenCombinedAuditError("exactly the five pinned BAGEN families are required")
    repo = _require_text(expected_repo, label="expected_repo")
    revision = _require_text(expected_revision, label="expected_revision")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise BagenCombinedAuditError("expected_revision must be a pinned 40-character commit")

    manifest_entries = _read_manifest(manifest_path, expected_repo=repo)
    summary_evidence, manifest_evidence = _verify_manifest_summary(
        manifest_summary_path,
        manifest_path,
        manifest_entries,
        expected_repo=repo,
        expected_revision=revision,
    )

    families: list[dict[str, Any]] = []
    all_trajectories: list[Trajectory] = []
    canonical_trajectory_index: list[dict[str, str]] = []
    family_by_task: defaultdict[str, set[str]] = defaultdict(set)
    trajectories_by_task: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    global_run_ids: set[str] = set()
    global_trajectory_ids: set[str] = set()
    global_condition_ids: set[str] = set()
    audited_manifest_paths: set[str] = set()

    for family in sorted(FAMILY_SPECS):
        family_summary, trajectories, trajectory_index = _verify_family(
            family,
            family_roots[family],
            family_audits[family],
            manifest_entries,
        )
        families.append(family_summary)
        all_trajectories.extend(trajectories)
        canonical_trajectory_index.extend(trajectory_index)
        for item in trajectory_index:
            if item["path"] in audited_manifest_paths:
                raise BagenCombinedAuditError(
                    f"trajectory Hub path appears in multiple families: {item['path']}"
                )
            audited_manifest_paths.add(item["path"])
        for trajectory in trajectories:
            if trajectory.run_id in global_run_ids:
                raise BagenCombinedAuditError(f"duplicate run_id: {trajectory.run_id}")
            if trajectory.trajectory_id in global_trajectory_ids:
                raise BagenCombinedAuditError(
                    f"duplicate trajectory_id: {trajectory.trajectory_id}"
                )
            global_run_ids.add(trajectory.run_id)
            global_trajectory_ids.add(trajectory.trajectory_id)
            global_condition_ids.add(trajectory.condition_id)
            family_by_task[trajectory.task_id].add(family)
            trajectories_by_task[trajectory.task_id].append(
                {
                    "family": family,
                    "run_id": trajectory.run_id,
                    "trajectory_id": trajectory.trajectory_id,
                    "condition_id": trajectory.condition_id,
                }
            )

    manifest_trajectory_paths = {
        item_path for item_path in manifest_entries if item_path.endswith(".traj.json")
    }
    if audited_manifest_paths != manifest_trajectory_paths:
        missing = sorted(manifest_trajectory_paths - audited_manifest_paths)
        extra = sorted(audited_manifest_paths - manifest_trajectory_paths)
        raise BagenCombinedAuditError(
            "five-family raw paths do not exactly cover pinned manifest trajectories; "
            f"missing={missing[:3]!r}, extra={extra[:3]!r}"
        )

    combined_dataset = build_supervised_dataset(all_trajectories)
    counts = {
        "task_id_count": len(family_by_task),
        "run_id_count": len(global_run_ids),
        "trajectory_id_count": len(global_trajectory_ids),
        "condition_id_count": len(global_condition_ids),
        "dataset_row_count": len(combined_dataset.rows),
        "raw_file_count": sum(int(item["raw_file_count"]) for item in families),
        "raw_bytes": sum(int(item["raw_bytes"]) for item in families),
    }
    normalized_expected_counts = _normalize_expected_counts(expected_counts)
    if counts != normalized_expected_counts:
        raise BagenCombinedAuditError(
            f"combined counts do not match the frozen expectation: {counts!r}"
        )
    if expected_dataset_id is not None:
        expected_dataset_sha = _require_sha256(
            expected_dataset_id, label="expected_dataset_id"
        )
        if combined_dataset.dataset_id != expected_dataset_sha:
            raise BagenCombinedAuditError("combined dataset ID changed")

    cross_family_counter = Counter(str(len(families_for_task)) for families_for_task in family_by_task.values())
    task_cross_family_distribution = {
        family_count: cross_family_counter[family_count]
        for family_count in sorted(cross_family_counter, key=int)
    }
    normalized_expected_distribution = {
        _require_text(key, label="expected distribution key"): _require_int(
            count, label=f"expected distribution.{key}"
        )
        for key, count in expected_task_cross_family_distribution.items()
    }
    if task_cross_family_distribution != normalized_expected_distribution:
        raise BagenCombinedAuditError("task cross-family distribution changed")

    task_family_mapping = [
        {
            "task_id": task_id,
            "family_count": len(family_by_task[task_id]),
            "families": sorted(family_by_task[task_id]),
            "trajectories": sorted(
                trajectories_by_task[task_id],
                key=lambda item: (item["family"], item["run_id"]),
            ),
        }
        for task_id in sorted(family_by_task)
    ]
    canonical_family_index = [
        {
            "family": str(family_summary["family"]),
            "canonical_content_sha256": str(
                family_summary["canonical_content_sha256"]
            ),
        }
        for family_summary in families
    ]
    canonical_trajectory_index.sort(key=lambda item: (item["family"], item["path"]))
    family_audit_index = {
        str(item["family"]): {
            "path": str(item["audit_path"]),
            "bytes": int(item["audit_bytes"]),
            "sha256": str(item["audit_sha256"]),
        }
        for item in families
    }
    family_dataset_id_index = {
        str(item["family"]): str(_require_mapping(item["dataset"], label="dataset")["dataset_id"])
        for item in families
    }
    source_files, source_hashes = _source_file_evidence(source_paths)

    payload: dict[str, Any] = {
        "combined_audit_schema_version": COMBINED_AUDIT_SCHEMA_VERSION,
        "source_id": SOURCE_ID,
        "hub": {"repo": repo, "resolved_revision": revision},
        "manifest": {
            "summary": summary_evidence,
            "raw": manifest_evidence,
        },
        "families": families,
        "family_audit_index": family_audit_index,
        "family_dataset_id_index": family_dataset_id_index,
        "counts": counts,
        "combined_dataset": {
            "dataset_id": combined_dataset.dataset_id,
            "row_count": len(combined_dataset.rows),
            "schema_version": DATASET_SCHEMA_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
        },
        "task_cross_family_distribution": task_cross_family_distribution,
        "task_family_mapping": task_family_mapping,
        "condition_trajectory_counts": {
            condition_id: sum(
                trajectory.condition_id == condition_id for trajectory in all_trajectories
            )
            for condition_id in sorted(global_condition_ids)
        },
        "canonical_family_index": canonical_family_index,
        "canonical_family_index_sha256": _semantic_sha256(canonical_family_index),
        "canonical_trajectory_index_sha256": _semantic_sha256(
            canonical_trajectory_index
        ),
        "source_files": source_files,
        "source_hashes": source_hashes,
        "construction": {
            "command": CONSTRUCTION_COMMAND,
            "reader": "BagenSwebenchReader",
            "dataset_builder": "build_supervised_dataset",
            "family_order": [str(item["family"]) for item in families],
            "output": "workspace/external/bagen/combined_swebench_audit.json",
        },
    }
    payload["audit_payload_sha256"] = _semantic_sha256(payload)
    return payload


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
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
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _parse_overrides(values: Sequence[str] | None, *, label: str) -> dict[str, Path] | None:
    if values is None:
        return None
    result: dict[str, Path] = {}
    for value in values:
        family, separator, raw_path = value.partition("=")
        if not separator or family not in FAMILY_SPECS or not raw_path:
            raise BagenCombinedAuditError(
                f"{label} must use one FAMILY=PATH entry for every pinned family"
            )
        if family in result:
            raise BagenCombinedAuditError(f"{label} repeats family {family!r}")
        result[family] = Path(raw_path)
    if set(result) != set(FAMILY_SPECS):
        raise BagenCombinedAuditError(f"{label} must specify all five pinned families")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and freeze the combined five-family BAGEN SWE-bench dataset."
    )
    parser.add_argument("--manifest-summary", type=Path, default=DEFAULT_MANIFEST_SUMMARY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--family-root",
        action="append",
        help="test/relocation override in FAMILY=PATH form; provide all five",
    )
    parser.add_argument(
        "--family-audit",
        action="append",
        help="test/relocation override in FAMILY=PATH form; provide all five",
    )
    args = parser.parse_args()
    try:
        family_roots = _parse_overrides(args.family_root, label="--family-root")
        family_audits = _parse_overrides(args.family_audit, label="--family-audit")
        audit = build_combined_audit(
            args.manifest_summary,
            args.manifest,
            family_roots or DEFAULT_FAMILY_ROOTS,
            family_audits or DEFAULT_FAMILY_AUDITS,
        )
        atomic_write_json(args.output, audit)
    except (BagenCombinedAuditError, OSError, ValueError) as exc:
        parser.exit(2, f"combined BAGEN audit failed: {exc}\n")
    print(
        json.dumps(
            {
                "audit_payload_sha256": audit["audit_payload_sha256"],
                "combined_dataset_id": audit["combined_dataset"]["dataset_id"],
                "trajectory_id_count": audit["counts"]["trajectory_id_count"],
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
