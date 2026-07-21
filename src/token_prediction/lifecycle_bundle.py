"""Strict persistence for Stage 2 lifecycle composite estimators.

The loader intentionally supports only the fully specified Stage 2 baseline:
five empirical Task-pre initializers, a cross-position Deduct updater, and a
fitted interval calibrator.  Unknown component formats fail closed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from token_prediction import __version__ as TOKEN_PREDICTION_VERSION
from token_prediction.contracts import SourceDescriptor
from token_prediction.crossfit import (
    SEED_POLICY_HASH,
    SEED_POLICY_ID,
    ensemble_repaired_forecasts,
)
from token_prediction.dataset import (
    CAPABILITY_DATASET_SCHEMA_VERSION,
    INNER_FOLDS,
    INNER_FOLD_POLICY_ID,
    LIFECYCLE_SCHEMA_VERSION,
    LIFECYCLE_WEIGHTING_ID,
    LifecycleSequence,
    PredictionPoint,
    PredictionPosition,
    PredictionTarget,
    decide_target_capability,
    supported_input_contract_hashes_from_capability,
)
from token_prediction.development import OUTER_FOLDS
from token_prediction.estimators import RunContext, SessionSeed, TokenForecast
from token_prediction.estimators.cross_position_deduct import (
    FittedCrossPositionDeduct,
)
from token_prediction.evaluation import FittedExpansionCalibrator
from token_prediction.features import FEATURE_SCHEMA_VERSION, FeatureGroup, FeatureSet
from token_prediction.lifecycle import (
    LifecyclePrediction,
    LifecycleRun,
    RuntimeMode,
    run_lifecycle_batch,
)


LIFECYCLE_COMPOSITE_BUNDLE_SCHEMA_VERSION = 2
LIFECYCLE_COMPONENT_SCHEMA_VERSION = 2
EMPIRICAL_INITIALIZER_FORMAT = "empirical_quantile_fitted_v1"
CROSS_POSITION_DEDUCT_FORMAT = "cross_position_deduct_fitted_v1"
OPAQUE_AUDIT_FORMAT = "opaque_audit_only_v1"

_MANIFEST = "manifest.json"
_MANIFEST_HASH = "manifest.sha256"
_CALIBRATOR = "calibrator.json"
_MAX_ENTRIES = 128
_MAX_DEPTH = 8
_MAX_FILE_BYTES = 64 * 1024 * 1024
_MAX_TOTAL_BYTES = 256 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024
_SHA256_CHARS = frozenset("0123456789abcdef")
_PROTOCOL_FEATURES = frozenset(
    {
        "missing_usage_attempts",
        "cumulative_provider_input_tokens",
        "cumulative_provider_output_tokens",
    }
)
_TASK_LIFECYCLE_SCHEMA_ID = f"task_lifecycle_v{LIFECYCLE_SCHEMA_VERSION}"
_CANDIDATE_GRAPH_KEYS = {
    "initializer_estimator_id",
    "updater_estimator_id",
    "lifecycle_schema_id",
    "seed_policy_id",
    "inner_split_policy_id",
}
_MANIFEST_KEYS = {
    "bundle_schema_version",
    "bundle_kind",
    "candidate_id",
    "candidate_hash",
    "candidate_graph",
    "dataset_id",
    "dataset_schema_version",
    "source_descriptor",
    "source_descriptor_hash",
    "capability_contract_hash",
    "code_hash",
    "runtime_versions",
    "split_plan_id",
    "eligibility_hash",
    "position",
    "target",
    "condition_id",
    "outer_fold",
    "outer_task_partitions_sha256",
    "feature_set",
    "feature_set_hash",
    "feature_schema_version",
    "protocol_features",
    "lifecycle_schema_id",
    "lifecycle_schema_version",
    "lifecycle_weighting_id",
    "lifecycle_context_hash",
    "lifecycle_scored_hash",
    "input_contract_hash",
    "initializer_hash",
    "initializer_components",
    "updater_component_hash",
    "inner_split_policy_id",
    "inner_split_id",
    "inner_task_assignments",
    "inner_task_assignments_sha256",
    "seed_policy_id",
    "seed_policy_hash",
    "seed_set_hash",
    "calibrator_id",
    "interval_alpha",
    "files",
}


class LifecycleBundleError(ValueError):
    """The lifecycle bundle is unsafe, corrupted, or unsupported."""


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
        raise LifecycleBundleError("bundle metadata is not canonical JSON data") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_constant(value: str) -> None:
    raise LifecycleBundleError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LifecycleBundleError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _parse_json(payload: bytes, *, description: str) -> Any:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LifecycleBundleError(f"{description} is not valid UTF-8 JSON") from exc
    if _canonical_json_bytes(value) != payload:
        raise LifecycleBundleError(f"{description} is not canonical JSON")
    return value


def _mapping(value: Any, *, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise LifecycleBundleError(f"{description} must be a JSON object")
    return value


def _keys(value: Mapping[str, Any], expected: set[str], *, description: str) -> None:
    if set(value) != expected:
        raise LifecycleBundleError(
            f"{description} keys do not match schema; "
            f"missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _string(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise LifecycleBundleError(f"{description} must be a non-empty trimmed string")
    return value


def _integer(value: Any, *, description: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LifecycleBundleError(f"{description} must be an integer >= {minimum}")
    return value


def _floating(value: Any, *, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LifecycleBundleError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise LifecycleBundleError(f"{description} must be finite")
    return result


def _checksum(value: Any, *, description: str) -> str:
    result = _string(value, description=description)
    if len(result) != 64 or set(result) - _SHA256_CHARS:
        raise LifecycleBundleError(f"{description} is not a lowercase SHA-256")
    return result


def _version_prefix(
    value: str,
    *,
    components: int,
    description: str,
) -> tuple[int, ...]:
    parts = value.split(".")
    if len(parts) < components:
        raise LifecycleBundleError(f"{description} is not a compatible version string")
    try:
        prefix = tuple(int(part) for part in parts[:components])
    except ValueError as exc:
        raise LifecycleBundleError(
            f"{description} is not a compatible version string"
        ) from exc
    if any(part < 0 for part in prefix):
        raise LifecycleBundleError(f"{description} is not a compatible version string")
    return prefix


def _validate_runtime_versions(value: Any) -> dict[str, str]:
    runtime = _mapping(value, description="runtime versions")
    if not runtime:
        raise LifecycleBundleError("runtime versions must not be empty")
    normalized: dict[str, str] = {}
    for name, version in sorted(runtime.items()):
        normalized[_string(name, description="runtime name")] = _string(
            version, description=f"runtime version {name!r}"
        )
    required = {"python_version", "token_prediction_version"}
    if not required <= set(normalized):
        raise LifecycleBundleError(
            "runtime versions must bind Python and token-prediction"
        )
    if _version_prefix(
        normalized["python_version"],
        components=2,
        description="bundle Python version",
    ) != _version_prefix(
        platform.python_version(),
        components=2,
        description="current Python version",
    ):
        raise LifecycleBundleError("bundle Python major/minor version is incompatible")
    if _version_prefix(
        normalized["token_prediction_version"],
        components=2,
        description="bundle token-prediction version",
    ) != _version_prefix(
        TOKEN_PREDICTION_VERSION,
        components=2,
        description="current token-prediction version",
    ):
        raise LifecycleBundleError("bundle token-prediction version is incompatible")
    return normalized


def _validate_lifecycle_capabilities(descriptor: SourceDescriptor) -> None:
    for position in (PredictionPosition.TASK_PRE, PredictionPosition.TASK_UPDATE):
        decision = decide_target_capability(
            descriptor.capabilities,
            position,
            PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
        )
        if not decision.available:
            raise LifecycleBundleError(
                "source capability contract cannot produce the lifecycle target"
            )


def _safe_relative_path(value: Any, *, description: str) -> str:
    rendered = _string(value, description=description)
    if "\\" in rendered:
        raise LifecycleBundleError(f"{description} must not contain backslashes")
    posix = PurePosixPath(rendered)
    windows = PureWindowsPath(rendered)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or not posix.parts
        or any(part in {"", ".", ".."} for part in posix.parts)
        or posix.as_posix() != rendered
        or len(posix.parts) > _MAX_DEPTH
    ):
        raise LifecycleBundleError(
            f"{description} must be a normalized relative POSIX path"
        )
    return rendered


def _is_reparse(stat_result: os.stat_result) -> bool:
    attributes = int(getattr(stat_result, "st_file_attributes", 0))
    flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return bool(attributes & flag)


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    if stat.S_IFMT(left.st_mode) != stat.S_IFMT(right.st_mode):
        return False
    left_identity = (int(left.st_dev), int(left.st_ino))
    right_identity = (int(right.st_dev), int(right.st_ino))
    return (
        not all(left_identity)
        or not all(right_identity)
        or left_identity == right_identity
    )


def _snapshot(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        stat.S_IFMT(metadata.st_mode),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _lstat(path: Path, *, description: str) -> os.stat_result:
    try:
        return path.lstat()
    except OSError as exc:
        raise LifecycleBundleError(f"cannot inspect {description}: {path}") from exc


def _require_directory(path: Path, *, description: str) -> os.stat_result:
    metadata = _lstat(path, description=description)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise LifecycleBundleError(f"{description} must be a real directory: {path}")
    return metadata


def _require_regular_file(path: Path, *, description: str) -> os.stat_result:
    metadata = _lstat(path, description=description)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise LifecycleBundleError(
            f"{description} must be a regular non-link file: {path}"
        )
    return metadata


def _read_regular_file(
    path: Path,
    *,
    expected_metadata: os.stat_result,
    maximum_bytes: int,
    description: str,
) -> bytes:
    before = _require_regular_file(path, description=description)
    if _snapshot(before) != _snapshot(expected_metadata):
        raise LifecycleBundleError(f"{description} changed before reading: {path}")
    if before.st_size < 0 or before.st_size > maximum_bytes:
        raise LifecycleBundleError(f"{description} exceeds its safe size limit: {path}")

    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(
        getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise LifecycleBundleError(f"cannot open {description}: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(opened.st_mode)
            or _is_reparse(opened)
            or not stat.S_ISREG(opened.st_mode)
            or not _same_identity(opened, before)
        ):
            raise LifecycleBundleError(
                f"{description} changed identity or resolved through a link: {path}"
            )
        chunks: list[bytes] = []
        total = 0
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            while True:
                chunk = handle.read(_READ_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum_bytes:
                    raise LifecycleBundleError(
                        f"{description} exceeds its safe size limit while reading: {path}"
                    )
                chunks.append(chunk)
            opened_after = os.fstat(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    after = _require_regular_file(path, description=description)
    if (
        total != before.st_size
        or not _same_identity(opened_after, before)
        or opened_after.st_size != before.st_size
        or opened_after.st_mtime_ns != before.st_mtime_ns
        or _snapshot(after) != _snapshot(before)
    ):
        raise LifecycleBundleError(f"{description} changed while being read: {path}")
    return b"".join(chunks)


def _read_directory(root: Path) -> dict[str, bytes]:
    root_stat = _require_directory(root, description="lifecycle bundle root")

    files: dict[str, bytes] = {}
    file_metadata: dict[str, os.stat_result] = {}
    directory_metadata: dict[str, os.stat_result] = {"": root_stat}
    total = 0
    entry_count = 0

    def visit(
        directory: Path,
        parts: tuple[str, ...],
        expected_metadata: os.stat_result,
    ) -> None:
        nonlocal entry_count, total
        before = _require_directory(directory, description="bundle directory")
        if _snapshot(before) != _snapshot(expected_metadata):
            raise LifecycleBundleError(
                f"bundle directory changed before enumeration: {directory}"
            )
        try:
            iterator = os.scandir(directory)
        except OSError as exc:
            raise LifecycleBundleError(f"cannot enumerate lifecycle bundle: {exc}") from exc
        with iterator:
            entries: list[
                tuple[str, Path, tuple[str, ...], os.stat_result]
            ] = []
            for entry in iterator:
                entry_count += 1
                if entry_count > _MAX_ENTRIES:
                    raise LifecycleBundleError(
                        "lifecycle bundle contains too many entries"
                    )
                relative_parts = (*parts, entry.name)
                relative = "/".join(relative_parts)
                _safe_relative_path(relative, description="bundle entry")
                entry_path = Path(entry.path)
                entry_stat = _lstat(
                    entry_path,
                    description=f"bundle entry {relative!r}",
                )
                entries.append((entry.name, entry_path, relative_parts, entry_stat))
            entries.sort(key=lambda item: item[0])
        if parts and not entries:
            raise LifecycleBundleError(
                f"empty bundle directory is forbidden: {'/'.join(parts)}"
            )
        for _name, entry_path, relative_parts, entry_stat in entries:
            relative = "/".join(relative_parts)
            if stat.S_ISLNK(entry_stat.st_mode) or _is_reparse(entry_stat):
                raise LifecycleBundleError(f"symlink/reparse bundle entry is forbidden: {relative}")
            if stat.S_ISDIR(entry_stat.st_mode):
                directory_metadata[relative] = entry_stat
                visit(entry_path, relative_parts, entry_stat)
                continue
            if not stat.S_ISREG(entry_stat.st_mode):
                raise LifecycleBundleError(f"non-regular bundle entry is forbidden: {relative}")
            if entry_stat.st_size > _MAX_FILE_BYTES:
                raise LifecycleBundleError(f"bundle file is too large: {relative}")
            file_metadata[relative] = entry_stat
            payload = _read_regular_file(
                entry_path,
                expected_metadata=entry_stat,
                maximum_bytes=_MAX_FILE_BYTES,
                description=f"bundle entry {relative!r}",
            )
            total += len(payload)
            if total > _MAX_TOTAL_BYTES:
                raise LifecycleBundleError("lifecycle bundle is too large")
            files[relative] = payload

        after = _require_directory(directory, description="bundle directory")
        if _snapshot(after) != _snapshot(before):
            raise LifecycleBundleError(
                f"bundle directory changed while being enumerated: {directory}"
            )

    visit(root, (), root_stat)

    seen_files: set[str] = set()
    seen_directories: set[str] = {""}

    def verify(directory: Path, parts: tuple[str, ...]) -> None:
        relative_directory = "/".join(parts)
        expected_directory = directory_metadata.get(relative_directory)
        if expected_directory is None:
            raise LifecycleBundleError("lifecycle bundle contains an extra directory")
        before = _require_directory(directory, description="bundle directory")
        if _snapshot(before) != _snapshot(expected_directory):
            raise LifecycleBundleError("lifecycle bundle directory changed after reading")
        try:
            iterator = os.scandir(directory)
        except OSError as exc:
            raise LifecycleBundleError(f"cannot re-enumerate lifecycle bundle: {exc}") from exc
        with iterator:
            entries = sorted(iterator, key=lambda item: item.name)
        for entry in entries:
            relative_parts = (*parts, entry.name)
            relative = "/".join(relative_parts)
            _safe_relative_path(relative, description="bundle entry")
            entry_path = Path(entry.path)
            metadata = _lstat(entry_path, description=f"bundle entry {relative!r}")
            if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
                raise LifecycleBundleError(
                    f"symlink/reparse bundle entry is forbidden: {relative}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                expected = directory_metadata.get(relative)
                if expected is None or _snapshot(metadata) != _snapshot(expected):
                    raise LifecycleBundleError(
                        "lifecycle bundle directory changed after reading"
                    )
                seen_directories.add(relative)
                verify(entry_path, relative_parts)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise LifecycleBundleError(
                    f"non-regular bundle entry is forbidden: {relative}"
                )
            expected = file_metadata.get(relative)
            if expected is None or _snapshot(metadata) != _snapshot(expected):
                raise LifecycleBundleError("lifecycle bundle file changed after reading")
            seen_files.add(relative)
        after = _require_directory(directory, description="bundle directory")
        if _snapshot(after) != _snapshot(expected_directory):
            raise LifecycleBundleError("lifecycle bundle directory changed during verification")

    verify(root, ())
    if seen_files != set(file_metadata) or seen_directories != set(directory_metadata):
        raise LifecycleBundleError("lifecycle bundle membership changed after reading")
    return files


def _normalize_files(source: str | os.PathLike[str] | Mapping[str, bytes]) -> dict[str, bytes]:
    if isinstance(source, Mapping):
        files: dict[str, bytes] = {}
        total = 0
        for raw_name, payload in source.items():
            name = _safe_relative_path(raw_name, description="bundle file name")
            if name in files:
                raise LifecycleBundleError(f"duplicate bundle file: {name}")
            if not isinstance(payload, bytes):
                raise LifecycleBundleError("bundle mapping values must be bytes")
            if len(payload) > _MAX_FILE_BYTES:
                raise LifecycleBundleError(f"bundle file is too large: {name}")
            total += len(payload)
            files[name] = payload
        if len(files) > _MAX_ENTRIES or total > _MAX_TOTAL_BYTES:
            raise LifecycleBundleError("lifecycle bundle exceeds safety limits")
        return files
    return _read_directory(Path(source))


def _feature_set_document(feature_set: FeatureSet) -> dict[str, Any]:
    return {
        "feature_set_id": feature_set.feature_set_id,
        "include_all": feature_set.include_all,
        "include_groups": sorted(value.value for value in feature_set.include_groups),
        "exclude_groups": sorted(value.value for value in feature_set.exclude_groups),
        "include_subgroups": sorted(feature_set.include_subgroups),
        "exclude_subgroups": sorted(feature_set.exclude_subgroups),
        "include_features": sorted(feature_set.include_features),
        "exclude_features": sorted(feature_set.exclude_features),
    }


def feature_set_document(feature_set: FeatureSet) -> Mapping[str, Any]:
    """Return the exact portable FeatureSet configuration."""

    return MappingProxyType(_feature_set_document(feature_set))


def _load_feature_set(value: Any, *, expected_hash: str) -> FeatureSet:
    document = _mapping(value, description="feature set")
    _keys(
        document,
        {
            "feature_set_id",
            "include_all",
            "include_groups",
            "exclude_groups",
            "include_subgroups",
            "exclude_subgroups",
            "include_features",
            "exclude_features",
        },
        description="feature set",
    )
    include_all = document["include_all"]
    if not isinstance(include_all, bool):
        raise LifecycleBundleError("feature set include_all must be boolean")

    def strings(name: str) -> list[str]:
        raw = document[name]
        if not isinstance(raw, list) or any(
            not isinstance(item, str) or not item for item in raw
        ):
            raise LifecycleBundleError(f"feature set {name} must be a string list")
        if raw != sorted(set(raw)):
            raise LifecycleBundleError(f"feature set {name} must be sorted and unique")
        return raw

    try:
        result = FeatureSet(
            feature_set_id=_string(
                document["feature_set_id"], description="feature set id"
            ),
            include_all=include_all,
            include_groups=frozenset(FeatureGroup(item) for item in strings("include_groups")),
            exclude_groups=frozenset(FeatureGroup(item) for item in strings("exclude_groups")),
            include_subgroups=frozenset(strings("include_subgroups")),
            exclude_subgroups=frozenset(strings("exclude_subgroups")),
            include_features=frozenset(strings("include_features")),
            exclude_features=frozenset(strings("exclude_features")),
        )
    except (TypeError, ValueError) as exc:
        raise LifecycleBundleError("feature set is invalid") from exc
    if result.content_hash != expected_hash:
        raise LifecycleBundleError("feature set content hash does not match")
    return result


def validate_source_provenance(
    value: Mapping[str, Any],
    *,
    source_descriptor_hash: str,
    capability_contract_hash: str,
    require_lifecycle_capabilities: bool = False,
) -> dict[str, Any]:
    """Validate the complete source/code/runtime provenance supplied by a caller."""

    document = _mapping(value, description="source provenance")
    _keys(
        document,
        {
            "source_descriptor",
            "source_descriptor_hash",
            "code_hash",
            "runtime_versions",
        },
        description="source provenance",
    )
    descriptor_document = _mapping(
        document["source_descriptor"], description="source descriptor"
    )
    try:
        descriptor = SourceDescriptor.from_dict(descriptor_document)
    except (TypeError, ValueError) as exc:
        raise LifecycleBundleError("source descriptor is invalid") from exc
    declared_descriptor_hash = _checksum(
        document["source_descriptor_hash"],
        description="source descriptor hash",
    )
    if descriptor.descriptor_hash != declared_descriptor_hash:
        raise LifecycleBundleError("source descriptor document/hash mismatch")
    if declared_descriptor_hash != source_descriptor_hash:
        raise LifecycleBundleError("source descriptor differs from lifecycle dataset")
    if descriptor.capabilities.contract_hash != capability_contract_hash:
        raise LifecycleBundleError("capability contract differs from lifecycle dataset")
    if require_lifecycle_capabilities:
        _validate_lifecycle_capabilities(descriptor)
    code_hash = _checksum(document["code_hash"], description="code hash")
    normalized_runtime = _validate_runtime_versions(document["runtime_versions"])
    return {
        "source_descriptor": descriptor.to_dict(),
        "source_descriptor_hash": declared_descriptor_hash,
        "code_hash": code_hash,
        "runtime_versions": normalized_runtime,
    }


@dataclass(frozen=True)
class _LoadedEmpiricalInitializer:
    estimator_id: str
    target: PredictionTarget
    lower: float
    point: float
    upper: float

    def start(self, context: RunContext) -> "_LoadedEmpiricalSession":
        del context
        return _LoadedEmpiricalSession(
            self.estimator_id,
            self.target,
            self.lower,
            self.point,
            self.upper,
        )


@dataclass
class _LoadedEmpiricalSession:
    estimator_id: str
    target: PredictionTarget
    lower: float
    point: float
    upper: float

    def predict(self, point: PredictionPoint) -> TokenForecast:
        return TokenForecast(
            point_id=point.point_id,
            target=self.target,
            lower=self.lower,
            point=self.point,
            upper=self.upper,
        )

    def observe(self, transition: Any) -> None:
        del transition


@dataclass(frozen=True)
class LoadedInitializer:
    inner_fold: int
    component_hash: str
    bundle_hashes: tuple[str, ...]
    fitted: _LoadedEmpiricalInitializer


@dataclass(frozen=True)
class LoadedLifecycleBundle:
    manifest: Mapping[str, Any]
    feature_set: FeatureSet
    initializers: tuple[LoadedInitializer, ...]
    updater: FittedCrossPositionDeduct
    calibrator: FittedExpansionCalibrator

    def _initializer_point(self, point: PredictionPoint) -> PredictionPoint:
        selected = self.feature_set.select(point.features)
        for name in _PROTOCOL_FEATURES:
            if name in point.features:
                selected[name] = point.features[name]
        return point.with_features(selected)

    def _updater_point(self, point: PredictionPoint) -> PredictionPoint:
        selected = self.feature_set.select(point.features)
        for name in _PROTOCOL_FEATURES:
            if name in point.features:
                selected[name] = point.features[name]
        return point.with_features(selected)

    def external_seeds(
        self,
        sequences: Sequence[LifecycleSequence],
    ) -> Mapping[str, SessionSeed]:
        if not sequences:
            raise LifecycleBundleError("lifecycle reload batch is empty")
        manifest = self.manifest
        seeds: dict[str, SessionSeed] = {}
        for sequence in sequences:
            if (
                sequence.dataset_id != manifest["dataset_id"]
                or sequence.condition_id != manifest["condition_id"]
                or sequence.target.value != manifest["target"]
                or sequence.input_contract_hash != manifest["input_contract_hash"]
                or sequence.schema_version != manifest["lifecycle_schema_version"]
            ):
                raise LifecycleBundleError(
                    "lifecycle sequence does not match the loaded bundle scope"
                )
            point = self._initializer_point(sequence.steps[0].point)
            forecasts: list[TokenForecast] = []
            for component in self.initializers:
                session = component.fitted.start(
                    RunContext(
                        point.task_id,
                        point.trajectory_id,
                        point.run_id,
                        dataset_id=sequence.dataset_id,
                        condition_id=sequence.condition_id,
                        target=sequence.target,
                        input_contract_hash=sequence.input_contract_hash,
                    )
                )
                forecast = session.predict(point)
                repaired_point = max(0.0, forecast.point)
                forecasts.append(
                    TokenForecast(
                        point_id=point.point_id,
                        target=point.target,
                        lower=min(max(0.0, forecast.lower), repaired_point),
                        point=repaired_point,
                        upper=max(max(0.0, forecast.upper), repaired_point),
                        raw_lower=forecast.lower,
                        raw_point=forecast.point,
                        raw_upper=forecast.upper,
                    )
                )
            ensemble = ensemble_repaired_forecasts(point, forecasts)
            bundle_hashes = tuple(
                digest
                for component in self.initializers
                for digest in component.bundle_hashes
            )
            seed = SessionSeed(
                task_pre_point=point,
                forecast=ensemble,
                initializer_id="empirical_quantile",
                initializer_hash=str(manifest["initializer_hash"]),
                inner_split_id=str(manifest["inner_split_id"]),
                component_bundle_hashes=bundle_hashes,
                seed_policy_id=SEED_POLICY_ID,
                seed_policy_hash=SEED_POLICY_HASH,
            )
            if point.point_id in seeds:
                raise LifecycleBundleError("duplicate Task-pre point in reload batch")
            seeds[point.point_id] = seed
        return MappingProxyType(seeds)

    def run_calibrated(
        self,
        sequences: Sequence[LifecycleSequence],
        *,
        runtime_mode: RuntimeMode = "offline",
    ) -> tuple[LifecycleRun, ...]:
        """Restore and replay every update boundary, including unscored context."""

        runs = run_lifecycle_batch(
            self.updater,
            sequences,
            self.external_seeds(sequences),
            runtime_mode=runtime_mode,
            select_point=self._updater_point,
        )
        calibrated: list[LifecycleRun] = []
        for run in runs:
            predictions = tuple(
                LifecyclePrediction(
                    item.step,
                    self.calibrator.transform(item.forecast),
                    item.transition,
                )
                for item in run.predictions
            )
            calibrated.append(
                LifecycleRun(
                    run.sequence,
                    run.runtime_mode,
                    run.seed,
                    predictions,
                )
            )
        return tuple(calibrated)


def _load_empirical_state(payload: bytes) -> _LoadedEmpiricalInitializer:
    state = _mapping(_parse_json(payload, description="empirical state"), description="empirical state")
    _keys(
        state,
        {
            "state_schema_version",
            "estimator_id",
            "target",
            "lower",
            "point",
            "upper",
        },
        description="empirical state",
    )
    if _integer(state["state_schema_version"], description="state schema", minimum=1) != 1:
        raise LifecycleBundleError("unsupported empirical state schema")
    if state["estimator_id"] != "empirical_quantile":
        raise LifecycleBundleError("empirical state estimator id is invalid")
    try:
        target = PredictionTarget(_string(state["target"], description="state target"))
    except ValueError as exc:
        raise LifecycleBundleError("empirical state target is invalid") from exc
    lower = _floating(state["lower"], description="empirical lower")
    point = _floating(state["point"], description="empirical point")
    upper = _floating(state["upper"], description="empirical upper")
    if not 0 <= lower <= point <= upper:
        raise LifecycleBundleError("empirical quantiles are invalid")
    return _LoadedEmpiricalInitializer(
        "empirical_quantile", target, lower, point, upper
    )


def _load_cross_position_state(payload: bytes) -> FittedCrossPositionDeduct:
    state = _mapping(
        _parse_json(payload, description="cross-position state"),
        description="cross-position state",
    )
    _keys(
        state,
        {
            "state_schema_version",
            "estimator_id",
            "dataset_id",
            "target",
            "condition_id",
            "input_contract_hash",
        },
        description="cross-position state",
    )
    if _integer(state["state_schema_version"], description="state schema", minimum=1) != 1:
        raise LifecycleBundleError("unsupported cross-position state schema")
    if state["estimator_id"] != "cross_position_deduct":
        raise LifecycleBundleError("cross-position state estimator id is invalid")
    try:
        target = PredictionTarget(_string(state["target"], description="state target"))
    except ValueError as exc:
        raise LifecycleBundleError("cross-position state target is invalid") from exc
    return FittedCrossPositionDeduct(
        "cross_position_deduct",
        _string(state["dataset_id"], description="state dataset id"),
        target,
        _string(state["condition_id"], description="state condition id"),
        _checksum(state["input_contract_hash"], description="state input contract"),
    )


def _validate_component_files(
    files: Mapping[str, bytes],
    manifest_files: Mapping[str, str],
    *,
    component_hash: str,
) -> tuple[Mapping[str, Any], dict[str, bytes], tuple[str, ...]]:
    prefix = f"components/{component_hash}/"
    descriptor_path = f"{prefix}component.json"
    if descriptor_path not in files:
        raise LifecycleBundleError("component descriptor is missing")
    descriptor = _mapping(
        _parse_json(files[descriptor_path], description="component descriptor"),
        description="component descriptor",
    )
    role = descriptor.get("role")
    if role == "inner_initializer":
        _keys(
            descriptor,
            {
                "component_schema_version",
                "role",
                "estimator_id",
                "serialization_format",
                "initializer_hash",
                "inner_split_id",
                "inner_task_assignments_sha256",
                "outer_fold",
                "inner_fold",
                "feature_set_hash",
                "task_partitions_sha256",
                "model_files",
                "calibration",
                "component_hash",
            },
            description="initializer component descriptor",
        )
    elif role == "lifecycle_updater":
        _keys(
            descriptor,
            {
                "component_schema_version",
                "role",
                "estimator_id",
                "serialization_format",
                "candidate_hash",
                "outer_fold",
                "model_files",
                "component_hash",
            },
            description="updater component descriptor",
        )
    else:
        raise LifecycleBundleError("component role is unsupported")
    if descriptor["component_schema_version"] != LIFECYCLE_COMPONENT_SCHEMA_VERSION:
        raise LifecycleBundleError("unsupported lifecycle component schema")
    declared_hash = _checksum(
        descriptor.get("component_hash"), description="component hash"
    )
    if declared_hash != component_hash:
        raise LifecycleBundleError("component directory/hash mismatch")
    semantic = dict(descriptor)
    semantic.pop("component_hash")
    if _sha256_bytes(_canonical_json_bytes(semantic)) != component_hash:
        raise LifecycleBundleError("component semantic hash does not match")
    model_hashes = _mapping(
        descriptor.get("model_files"), description="component model files"
    )
    payloads: dict[str, bytes] = {}
    for raw_name, raw_hash in model_hashes.items():
        name = _safe_relative_path(raw_name, description="component model file")
        expected = _checksum(raw_hash, description="component model checksum")
        path = f"{prefix}model/{name}"
        if path not in files or manifest_files.get(path) != expected:
            raise LifecycleBundleError("component model file is missing or inconsistent")
        payloads[name] = files[path]
    actual_paths = {
        name for name in files if name.startswith(prefix) and name != descriptor_path
    }
    expected_paths = {f"{prefix}model/{name}" for name in payloads}
    if actual_paths != expected_paths:
        raise LifecycleBundleError("component contains missing or extra files")
    bundle_hashes = tuple(
        sorted(_sha256_bytes(files[name]) for name in {descriptor_path, *expected_paths})
    )
    return descriptor, payloads, bundle_hashes


def load_lifecycle_bundle(
    source: str | os.PathLike[str] | Mapping[str, bytes],
    *,
    expected_source_provenance: Mapping[str, Any] | None = None,
) -> LoadedLifecycleBundle:
    """Load a safe, complete lifecycle bundle from a directory or byte mapping."""

    files = _normalize_files(source)
    if _MANIFEST not in files or _MANIFEST_HASH not in files:
        raise LifecycleBundleError("lifecycle bundle manifest is missing")
    manifest_bytes = files[_MANIFEST]
    expected_manifest_hash = f"{_sha256_bytes(manifest_bytes)}\n".encode("ascii")
    if files[_MANIFEST_HASH] != expected_manifest_hash:
        raise LifecycleBundleError("lifecycle bundle manifest checksum does not match")
    manifest = _mapping(
        _parse_json(manifest_bytes, description="lifecycle manifest"),
        description="lifecycle manifest",
    )
    _keys(manifest, _MANIFEST_KEYS, description="lifecycle manifest")
    if manifest["bundle_schema_version"] != LIFECYCLE_COMPOSITE_BUNDLE_SCHEMA_VERSION:
        raise LifecycleBundleError("unsupported lifecycle bundle schema")
    if manifest["bundle_kind"] != "lifecycle_composite":
        raise LifecycleBundleError("unsupported lifecycle bundle kind")

    declared_files_raw = _mapping(manifest["files"], description="manifest files")
    declared_files: dict[str, str] = {}
    for raw_name, raw_hash in declared_files_raw.items():
        name = _safe_relative_path(raw_name, description="manifest file name")
        if name in {_MANIFEST, _MANIFEST_HASH}:
            raise LifecycleBundleError("manifest cannot hash itself")
        declared_files[name] = _checksum(raw_hash, description="manifest file checksum")
    if set(files) != { _MANIFEST, _MANIFEST_HASH, *declared_files }:
        raise LifecycleBundleError("lifecycle bundle contains missing or extra files")
    for name, expected in declared_files.items():
        if _sha256_bytes(files[name]) != expected:
            raise LifecycleBundleError(f"bundle checksum mismatch: {name}")

    for name in (
        "candidate_hash",
        "source_descriptor_hash",
        "capability_contract_hash",
        "code_hash",
        "split_plan_id",
        "eligibility_hash",
        "feature_set_hash",
        "lifecycle_context_hash",
        "lifecycle_scored_hash",
        "input_contract_hash",
        "initializer_hash",
        "updater_component_hash",
        "inner_split_id",
        "seed_policy_hash",
        "seed_set_hash",
    ):
        _checksum(manifest[name], description=f"manifest {name}")
    _string(manifest["candidate_id"], description="manifest candidate id")
    _string(manifest["dataset_id"], description="manifest dataset id")
    _string(manifest["condition_id"], description="manifest condition id")
    graph = _mapping(manifest["candidate_graph"], description="candidate graph")
    _keys(graph, _CANDIDATE_GRAPH_KEYS, description="candidate graph")
    expected_graph = {
        "initializer_estimator_id": "empirical_quantile",
        "updater_estimator_id": "cross_position_deduct",
        "lifecycle_schema_id": _TASK_LIFECYCLE_SCHEMA_ID,
        "seed_policy_id": SEED_POLICY_ID,
        "inner_split_policy_id": INNER_FOLD_POLICY_ID,
    }
    if dict(graph) != expected_graph:
        raise LifecycleBundleError("unsupported lifecycle candidate graph")
    descriptor_document = _mapping(
        manifest["source_descriptor"], description="manifest source descriptor"
    )
    try:
        descriptor = SourceDescriptor.from_dict(descriptor_document)
    except (TypeError, ValueError) as exc:
        raise LifecycleBundleError("manifest source descriptor is invalid") from exc
    if descriptor.descriptor_hash != manifest["source_descriptor_hash"]:
        raise LifecycleBundleError("manifest source descriptor hash does not match")
    if descriptor.capabilities.contract_hash != manifest["capability_contract_hash"]:
        raise LifecycleBundleError("manifest capability contract hash does not match")
    _validate_lifecycle_capabilities(descriptor)
    allowed_input_contract_hashes = supported_input_contract_hashes_from_capability(
        str(manifest["capability_contract_hash"])
    )
    if manifest["input_contract_hash"] not in allowed_input_contract_hashes:
        raise LifecycleBundleError(
            "manifest input contract does not match its capability contract"
        )
    runtime = _validate_runtime_versions(manifest["runtime_versions"])
    if expected_source_provenance is not None:
        expected = validate_source_provenance(
            expected_source_provenance,
            source_descriptor_hash=str(manifest["source_descriptor_hash"]),
            capability_contract_hash=str(manifest["capability_contract_hash"]),
            require_lifecycle_capabilities=True,
        )
        for name, actual in (
            ("source_descriptor", descriptor.to_dict()),
            ("source_descriptor_hash", manifest["source_descriptor_hash"]),
            ("code_hash", manifest["code_hash"]),
            ("runtime_versions", runtime),
        ):
            if expected[name] != actual:
                raise LifecycleBundleError(
                    f"bundle {name.replace('_', ' ')} differs from the expected provenance"
                )
    if manifest["position"] != PredictionPosition.TASK_UPDATE.value:
        raise LifecycleBundleError("lifecycle bundle position is invalid")
    if manifest["target"] != PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS.value:
        raise LifecycleBundleError("lifecycle bundle target is invalid")
    if manifest["seed_policy_id"] != SEED_POLICY_ID:
        raise LifecycleBundleError("lifecycle seed policy id is invalid")
    if manifest["seed_policy_hash"] != SEED_POLICY_HASH:
        raise LifecycleBundleError("lifecycle seed policy hash is invalid")
    if manifest["lifecycle_schema_version"] != LIFECYCLE_SCHEMA_VERSION:
        raise LifecycleBundleError("lifecycle schema version is invalid")
    if manifest["lifecycle_schema_id"] != _TASK_LIFECYCLE_SCHEMA_ID:
        raise LifecycleBundleError("lifecycle schema id is invalid")
    if manifest["lifecycle_weighting_id"] != LIFECYCLE_WEIGHTING_ID:
        raise LifecycleBundleError("lifecycle weighting policy is invalid")
    if manifest["inner_split_policy_id"] != INNER_FOLD_POLICY_ID:
        raise LifecycleBundleError("inner split policy is invalid")
    if manifest["protocol_features"] != sorted(_PROTOCOL_FEATURES):
        raise LifecycleBundleError("lifecycle protocol features are invalid")
    if (
        _integer(manifest["dataset_schema_version"], description="dataset schema", minimum=1)
        != CAPABILITY_DATASET_SCHEMA_VERSION
    ):
        raise LifecycleBundleError("unsupported lifecycle dataset schema")
    if (
        _integer(manifest["feature_schema_version"], description="feature schema", minimum=1)
        != FEATURE_SCHEMA_VERSION
    ):
        raise LifecycleBundleError("unsupported lifecycle feature schema")
    if _integer(manifest["outer_fold"], description="outer fold") >= OUTER_FOLDS:
        raise LifecycleBundleError("lifecycle outer fold is out of range")
    outer_partitions = _mapping(
        manifest["outer_task_partitions_sha256"],
        description="outer task partitions",
    )
    _keys(
        outer_partitions,
        {"train", "validation", "calibration", "test"},
        description="outer task partitions",
    )
    outer_partition_sets: dict[str, frozenset[str]] = {}
    for role, pseudonyms in outer_partitions.items():
        if not isinstance(pseudonyms, list) or not pseudonyms:
            raise LifecycleBundleError(f"outer {role} task partition is invalid")
        checksums = [
            _checksum(item, description=f"outer {role} task pseudonym")
            for item in pseudonyms
        ]
        if checksums != sorted(set(checksums)):
            raise LifecycleBundleError(f"outer {role} task partition is not canonical")
        outer_partition_sets[role] = frozenset(checksums)
    outer_roles = tuple(outer_partition_sets)
    for index, left_role in enumerate(outer_roles):
        for right_role in outer_roles[index + 1 :]:
            if outer_partition_sets[left_role] & outer_partition_sets[right_role]:
                raise LifecycleBundleError("outer task partitions overlap")

    assignment_records = manifest["inner_task_assignments"]
    if not isinstance(assignment_records, list) or not assignment_records:
        raise LifecycleBundleError("inner task assignments are invalid")
    inner_mapping_list: list[tuple[str, int]] = []
    for record in assignment_records:
        item = _mapping(record, description="inner task assignment")
        _keys(item, {"task_pseudonym", "fold"}, description="inner task assignment")
        inner_mapping_list.append(
            (
                _checksum(
                    item["task_pseudonym"],
                    description="inner task assignment pseudonym",
                ),
                _integer(item["fold"], description="inner task assignment fold"),
            )
        )
    inner_mapping = tuple(inner_mapping_list)
    if inner_mapping != tuple(sorted(inner_mapping)):
        raise LifecycleBundleError("inner task assignments are not canonical")
    inner_tasks = [task for task, _fold in inner_mapping]
    if len(inner_tasks) != len(set(inner_tasks)):
        raise LifecycleBundleError("inner task assignments contain duplicate tasks")
    if {fold for _task, fold in inner_mapping} != set(range(INNER_FOLDS)):
        raise LifecycleBundleError("inner task assignments require exactly five folds")
    inner_task_set = frozenset(inner_tasks)
    condition_outer_train = outer_partition_sets["train"]
    if not condition_outer_train <= inner_task_set:
        raise LifecycleBundleError(
            "inner task assignments do not cover condition outer train tasks"
        )
    if inner_task_set & (
        outer_partition_sets["validation"]
        | outer_partition_sets["calibration"]
        | outer_partition_sets["test"]
    ):
        raise LifecycleBundleError(
            "inner task assignments include a condition outer evaluation task"
        )
    if _sha256_bytes(_canonical_json_bytes(inner_mapping)) != manifest[
        "inner_task_assignments_sha256"
    ]:
        raise LifecycleBundleError("inner task assignment hash does not match")
    inner_task_to_fold = {
        task: fold
        for task, fold in inner_mapping
        if task in condition_outer_train
    }

    feature_set = _load_feature_set(
        manifest["feature_set"], expected_hash=str(manifest["feature_set_hash"])
    )
    calibrator_document = _mapping(
        _parse_json(files[_CALIBRATOR], description="calibrator"),
        description="calibrator",
    )
    try:
        calibrator = FittedExpansionCalibrator.from_dict(calibrator_document)
    except (TypeError, ValueError) as exc:
        raise LifecycleBundleError("fitted calibrator is invalid") from exc
    alpha = _floating(manifest["interval_alpha"], description="interval alpha")
    if (
        calibrator.calibrator_id != manifest["calibrator_id"]
        or not math.isclose(calibrator.interval_alpha, alpha, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise LifecycleBundleError("calibrator does not match manifest")

    summaries = manifest["initializer_components"]
    if not isinstance(summaries, list) or len(summaries) != 5:
        raise LifecycleBundleError("lifecycle bundle requires five initializer components")
    initializers: list[LoadedInitializer] = []
    for summary in summaries:
        summary = _mapping(summary, description="initializer summary")
        _keys(
            summary,
            {"inner_fold", "component_hash", "bundle_hashes"},
            description="initializer summary",
        )
        inner_fold = _integer(summary["inner_fold"], description="inner fold")
        component_hash = _checksum(
            summary["component_hash"], description="initializer component hash"
        )
        descriptor, payloads, actual_bundle_hashes = _validate_component_files(
            files,
            declared_files,
            component_hash=component_hash,
        )
        if descriptor.get("role") != "inner_initializer":
            raise LifecycleBundleError("initializer component role is invalid")
        if descriptor.get("serialization_format") != EMPIRICAL_INITIALIZER_FORMAT:
            raise LifecycleBundleError("unsupported lifecycle initializer component")
        if descriptor.get("estimator_id") != "empirical_quantile":
            raise LifecycleBundleError("initializer estimator id is invalid")
        if descriptor.get("initializer_hash") != manifest["initializer_hash"]:
            raise LifecycleBundleError("initializer identity differs from manifest")
        if descriptor.get("inner_split_id") != manifest["inner_split_id"]:
            raise LifecycleBundleError("initializer split differs from manifest")
        if (
            descriptor.get("inner_task_assignments_sha256")
            != manifest["inner_task_assignments_sha256"]
        ):
            raise LifecycleBundleError(
                "initializer task assignment identity differs from manifest"
            )
        if descriptor.get("outer_fold") != manifest["outer_fold"]:
            raise LifecycleBundleError("initializer outer fold differs from manifest")
        if descriptor.get("inner_fold") != inner_fold:
            raise LifecycleBundleError("initializer inner fold differs from summary")
        if descriptor.get("feature_set_hash") != manifest["feature_set_hash"]:
            raise LifecycleBundleError("initializer feature set differs from manifest")
        if descriptor.get("calibration") != "none":
            raise LifecycleBundleError("initializer must remain uncalibrated")
        task_partitions = _mapping(
            descriptor.get("task_partitions_sha256"),
            description="initializer task partitions",
        )
        _keys(
            task_partitions,
            {"fit", "validation", "holdout"},
            description="initializer task partitions",
        )
        normalized_partitions: dict[str, frozenset[str]] = {}
        for role, pseudonyms in task_partitions.items():
            if not isinstance(pseudonyms, list) or not pseudonyms:
                raise LifecycleBundleError(
                    f"initializer {role} task partition is invalid"
                )
            checksums = [
                _checksum(item, description=f"initializer {role} task pseudonym")
                for item in pseudonyms
            ]
            if checksums != sorted(set(checksums)):
                raise LifecycleBundleError(
                    f"initializer {role} task partition is not canonical"
                )
            normalized_partitions[role] = frozenset(checksums)
        partition_roles = tuple(normalized_partitions)
        for index, left_role in enumerate(partition_roles):
            for right_role in partition_roles[index + 1 :]:
                if normalized_partitions[left_role] & normalized_partitions[right_role]:
                    raise LifecycleBundleError("initializer task partitions overlap")
        expected_holdout = frozenset(
            task for task, fold in inner_task_to_fold.items() if fold == inner_fold
        )
        expected_validation = frozenset(
            task
            for task, fold in inner_task_to_fold.items()
            if fold == (inner_fold + 1) % INNER_FOLDS
        )
        expected_fit = frozenset(inner_task_to_fold) - expected_holdout - expected_validation
        expected_partitions = {
            "fit": expected_fit,
            "validation": expected_validation,
            "holdout": expected_holdout,
        }
        if normalized_partitions != expected_partitions:
            raise LifecycleBundleError(
                "initializer task partitions differ from the frozen inner assignment"
            )
        if set(payloads) != {"state.json"}:
            raise LifecycleBundleError("empirical initializer state files are invalid")
        declared_bundle_hashes = summary["bundle_hashes"]
        if not isinstance(declared_bundle_hashes, list) or tuple(
            _checksum(item, description="initializer bundle hash")
            for item in declared_bundle_hashes
        ) != actual_bundle_hashes:
            raise LifecycleBundleError("initializer bundle hashes do not match")
        fitted = _load_empirical_state(payloads["state.json"])
        if fitted.target.value != manifest["target"]:
            raise LifecycleBundleError("initializer target differs from manifest")
        initializers.append(
            LoadedInitializer(inner_fold, component_hash, actual_bundle_hashes, fitted)
        )
    if [item.inner_fold for item in initializers] != list(range(5)):
        raise LifecycleBundleError("initializer folds must be exactly 0..4")

    updater_hash = str(manifest["updater_component_hash"])
    updater_descriptor, updater_payloads, _ = _validate_component_files(
        files,
        declared_files,
        component_hash=updater_hash,
    )
    if updater_descriptor.get("role") != "lifecycle_updater":
        raise LifecycleBundleError("updater component role is invalid")
    if updater_descriptor.get("serialization_format") != CROSS_POSITION_DEDUCT_FORMAT:
        raise LifecycleBundleError("unsupported lifecycle updater component")
    if updater_descriptor.get("estimator_id") != "cross_position_deduct":
        raise LifecycleBundleError("updater estimator id is invalid")
    if updater_descriptor.get("candidate_hash") != manifest["candidate_hash"]:
        raise LifecycleBundleError("updater candidate identity differs from manifest")
    if updater_descriptor.get("outer_fold") != manifest["outer_fold"]:
        raise LifecycleBundleError("updater outer fold differs from manifest")
    if set(updater_payloads) != {"state.json"}:
        raise LifecycleBundleError("cross-position updater state files are invalid")
    updater = _load_cross_position_state(updater_payloads["state.json"])
    if (
        updater.dataset_id != manifest["dataset_id"]
        or updater.condition_id != manifest["condition_id"]
        or updater.target.value != manifest["target"]
        or updater.input_contract_hash != manifest["input_contract_hash"]
    ):
        raise LifecycleBundleError("updater state differs from manifest scope")

    return LoadedLifecycleBundle(
        MappingProxyType(dict(manifest)),
        feature_set,
        tuple(initializers),
        updater,
        calibrator,
    )


__all__ = [
    "CROSS_POSITION_DEDUCT_FORMAT",
    "EMPIRICAL_INITIALIZER_FORMAT",
    "LIFECYCLE_COMPONENT_SCHEMA_VERSION",
    "LIFECYCLE_COMPOSITE_BUNDLE_SCHEMA_VERSION",
    "LifecycleBundleError",
    "LoadedLifecycleBundle",
    "OPAQUE_AUDIT_FORMAT",
    "feature_set_document",
    "load_lifecycle_bundle",
    "validate_source_provenance",
]
