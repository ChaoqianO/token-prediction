"""Portable, fail-closed persistence for fitted LightGBM quantile models.

Bundles are directories with an exact file allow-list.  LightGBM boosters are
stored in their documented text representation, never as Python pickles.  The
manifest checksum detects accidental manifest edits; it is not a signature and
does not protect against an attacker who can rewrite the entire directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from token_prediction.dataset import PredictionPosition, PredictionTarget

from .lightgbm import (
    LIGHTGBM_ESTIMATOR_VERSION,
    FittedLightGBMQuantiles,
    LightGBMFitReport,
    QuantileFitReport,
    _load_optional_dependencies,
)
from .tabular_encoder import ENCODER_SCHEMA_VERSION, FoldTabularEncoder


BUNDLE_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
MANIFEST_HASH_FILENAME = "manifest.sha256"
ENCODER_FILENAME = "encoder.json"
_ESTIMATOR_ID = "lightgbm_quantile"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_VERSION = re.compile(r"(?P<major>[0-9]+)(?:\.|\Z)")
_MAX_BUNDLE_ENTRIES = 16
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_ENCODER_BYTES = 64 * 1024 * 1024
_MAX_MODEL_BYTES = 512 * 1024 * 1024


class LightGBMBundleError(ValueError):
    """The requested bundle is incomplete, inconsistent, or unsupported."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise LightGBMBundleError("bundle metadata is not canonical JSON") from exc
    return (rendered + "\n").encode("utf-8")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LightGBMBundleError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _invalid_json_constant(value: str) -> None:
    raise LightGBMBundleError(f"non-finite JSON number {value!r}")


def _parse_json(payload: bytes, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_invalid_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LightGBMBundleError(f"{description} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise LightGBMBundleError(f"{description} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, description: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise LightGBMBundleError(
            f"{description} keys do not match schema; missing={missing}, extra={extra}"
        )


def _require_mapping(value: Any, *, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LightGBMBundleError(f"{description} must be an object")
    return value


def _require_list(value: Any, *, description: str) -> list[Any]:
    if not isinstance(value, list):
        raise LightGBMBundleError(f"{description} must be an array")
    return value


def _require_string(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise LightGBMBundleError(f"{description} must be a non-empty string")
    return value


def _require_int(value: Any, *, description: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LightGBMBundleError(
            f"{description} must be an integer greater than or equal to {minimum}"
        )
    return value


def _require_float(value: Any, *, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LightGBMBundleError(f"{description} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise LightGBMBundleError(f"{description} must be a finite number")
    return parsed


def _version_major(value: Any, *, description: str) -> int:
    version = _require_string(value, description=description)
    matched = _VERSION.match(version)
    if matched is None:
        raise LightGBMBundleError(f"{description} is not a supported version string")
    return int(matched.group("major"))


def _quantile_id(quantile: float) -> str:
    """Return a collision-free identifier for the quantile's IEEE-754 value."""

    return f"q{struct.pack('>d', quantile).hex()}"


def _model_filename(quantile_id: str) -> str:
    return f"model-{quantile_id}.txt"


def _validate_quantiles(values: tuple[float, ...]) -> None:
    if (
        len(values) != 3
        or not 0 < values[0] < values[1] < values[2] < 1
        or not math.isclose(values[1], 0.5, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise LightGBMBundleError(
            "bundle quantiles must be ordered (lower, 0.5, upper) values in (0, 1)"
        )
    if len({_quantile_id(value) for value in values}) != len(values):
        raise LightGBMBundleError("bundle quantile identifiers collide")


def _validate_encoder_payload(value: Mapping[str, Any]) -> None:
    _require_exact_keys(
        value,
        {
            "schema_version",
            "columns",
            "category_vocabularies",
            "vector_dimensions",
            "dropped_all_missing_vectors",
        },
        description="encoder schema",
    )
    _require_int(value["schema_version"], description="encoder schema version", minimum=1)
    for index, item in enumerate(_require_list(value["columns"], description="columns")):
        column = _require_mapping(item, description=f"encoder column {index}")
        _require_exact_keys(
            column,
            {"name", "source_feature", "dtype", "vector_index"},
            description=f"encoder column {index}",
        )
    for index, item in enumerate(
        _require_list(value["category_vocabularies"], description="vocabularies")
    ):
        vocabulary = _require_mapping(item, description=f"category vocabulary {index}")
        _require_exact_keys(
            vocabulary,
            {"feature_name", "values"},
            description=f"category vocabulary {index}",
        )
    for index, item in enumerate(
        _require_list(value["vector_dimensions"], description="vector dimensions")
    ):
        vector = _require_mapping(item, description=f"vector dimension {index}")
        _require_exact_keys(
            vector,
            {"feature_name", "width"},
            description=f"vector dimension {index}",
        )
    _require_list(
        value["dropped_all_missing_vectors"], description="dropped vector features"
    )


def _fit_report_manifest(fitted: FittedLightGBMQuantiles) -> dict[str, Any]:
    report = fitted.fit_report
    quantile_reports: dict[str, Any] = {}
    for quantile_report in report.quantiles:
        quantile_id = _quantile_id(quantile_report.quantile)
        quantile_reports[quantile_id] = {
            "seed": quantile_report.seed,
            "best_validation_loss": quantile_report.best_validation_loss,
            "validation_history": list(quantile_report.validation_history),
            "parameters": dict(quantile_report.parameters),
        }
    return {
        "train_point_hash": report.train_point_hash,
        "validation_point_hash": report.validation_point_hash,
        "train_point_count": report.train_point_count,
        "validation_point_count": report.validation_point_count,
        "platform": report.platform,
        "quantile_reports": quantile_reports,
    }


def _validate_fitted(fitted: FittedLightGBMQuantiles) -> None:
    if not isinstance(fitted, FittedLightGBMQuantiles):
        raise TypeError("save_lightgbm_bundle requires FittedLightGBMQuantiles")
    if fitted.estimator_id != _ESTIMATOR_ID:
        raise LightGBMBundleError(f"unsupported estimator id {fitted.estimator_id!r}")
    if fitted.fit_report.estimator_version != LIGHTGBM_ESTIMATOR_VERSION:
        raise LightGBMBundleError("fitted estimator version is unsupported")
    quantiles = tuple(float(value) for value in fitted.quantiles)
    _validate_quantiles(quantiles)
    if set(fitted.boosters) != set(quantiles):
        raise LightGBMBundleError("booster keys do not match fitted quantiles")
    if set(fitted.best_iterations) != set(quantiles):
        raise LightGBMBundleError("best-iteration keys do not match fitted quantiles")
    if tuple(report.quantile for report in fitted.fit_report.quantiles) != quantiles:
        raise LightGBMBundleError("fit-report quantiles do not match fitted quantiles")
    for report in fitted.fit_report.quantiles:
        if report.best_iteration != fitted.best_iterations[report.quantile]:
            raise LightGBMBundleError("fit-report best iteration is inconsistent")
    if fitted.fit_report.encoder_schema_hash != fitted.encoder.schema.content_hash:
        raise LightGBMBundleError("fit-report encoder hash is inconsistent")
    _version_major(
        fitted.fit_report.lightgbm_version,
        description="training LightGBM version",
    )
    _version_major(fitted.fit_report.numpy_version, description="training NumPy version")


def lightgbm_bundle_files(fitted: FittedLightGBMQuantiles) -> Mapping[str, bytes]:
    """Build the complete canonical bundle file set without touching the filesystem."""

    _validate_fitted(fitted)
    lgb, np = _load_optional_dependencies()
    if _version_major(
        fitted.fit_report.lightgbm_version,
        description="training LightGBM version",
    ) != _version_major(lgb.__version__, description="runtime LightGBM version"):
        raise LightGBMBundleError("training and save-time LightGBM major versions differ")
    if _version_major(
        fitted.fit_report.numpy_version,
        description="training NumPy version",
    ) != _version_major(np.__version__, description="runtime NumPy version"):
        raise LightGBMBundleError("training and save-time NumPy major versions differ")

    encoder_payload = fitted.encoder.to_dict()
    encoder_bytes = _json_bytes(encoder_payload)
    files: dict[str, bytes] = {ENCODER_FILENAME: encoder_bytes}
    quantile_records: list[dict[str, Any]] = []
    best_iterations: dict[str, int] = {}
    model_records: dict[str, dict[str, str]] = {}
    for quantile in fitted.quantiles:
        quantile_id = _quantile_id(quantile)
        filename = _model_filename(quantile_id)
        best_iteration = int(fitted.best_iterations[quantile])
        if best_iteration <= 0:
            raise LightGBMBundleError("best iterations must be positive")
        model_bytes = fitted.boosters[quantile].model_to_string(
            num_iteration=best_iteration
        ).encode("utf-8")
        files[filename] = model_bytes
        quantile_records.append({"id": quantile_id, "value": quantile})
        best_iterations[quantile_id] = best_iteration
        model_records[quantile_id] = {
            "filename": filename,
            "sha256": _sha256(model_bytes),
        }

    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "estimator": {
            "id": fitted.estimator_id,
            "version": fitted.fit_report.estimator_version,
        },
        "scope": {
            "dataset_id": fitted.dataset_id,
            "position": fitted.position.value,
            "prediction_target": fitted.target.value,
            "condition_ids": list(fitted.allowed_condition_ids),
        },
        "quantiles": quantile_records,
        "best_iterations": best_iterations,
        "encoder": {
            "filename": ENCODER_FILENAME,
            "sha256": _sha256(encoder_bytes),
            "content_hash": fitted.encoder.schema.content_hash,
            "schema_version": fitted.encoder.schema.schema_version,
        },
        "models": model_records,
        "runtime": {
            "lightgbm_version": fitted.fit_report.lightgbm_version,
            "numpy_version": fitted.fit_report.numpy_version,
        },
        "fit_report": _fit_report_manifest(fitted),
    }
    manifest_bytes = _json_bytes(manifest)
    files[MANIFEST_FILENAME] = manifest_bytes
    files[MANIFEST_HASH_FILENAME] = f"{_sha256(manifest_bytes)}\n".encode("ascii")
    return MappingProxyType(files)


def save_lightgbm_bundle(
    fitted: FittedLightGBMQuantiles, directory: str | Path
) -> Path:
    """Save ``fitted`` into a new directory and return its resolved path.

    The destination must not exist.  This prevents stale model files from being
    silently retained when a bundle is replaced.
    """

    destination = Path(directory).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"bundle destination already exists: {destination}")
    files = lightgbm_bundle_files(fitted)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    )
    try:
        for filename, payload in files.items():
            (temporary / filename).write_bytes(payload)
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _is_reparse(metadata: os.stat_result) -> bool:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return bool(attributes & flag)


def _reliable_identity(metadata: os.stat_result) -> tuple[int, int] | None:
    device = int(getattr(metadata, "st_dev", 0))
    inode = int(getattr(metadata, "st_ino", 0))
    links = int(getattr(metadata, "st_nlink", 0))
    if device < 0 or inode <= 0 or links <= 0:
        return None
    return device, inode


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    left_identity = _reliable_identity(left)
    right_identity = _reliable_identity(right)
    return (
        stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
        and left_identity is not None
        and left_identity == right_identity
    )


def _snapshot(
    metadata: os.stat_result,
) -> tuple[int, int | None, tuple[int, int] | None, int, int]:
    return (
        stat.S_IFMT(metadata.st_mode),
        int(metadata.st_size) if stat.S_ISREG(metadata.st_mode) else None,
        _reliable_identity(metadata),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _same_snapshot(left: os.stat_result, right: os.stat_result) -> bool:
    left_snapshot = _snapshot(left)
    return left_snapshot[2] is not None and left_snapshot == _snapshot(right)


def _require_reliable_identity(
    metadata: os.stat_result, *, description: str
) -> os.stat_result:
    if _reliable_identity(metadata) is None:
        raise LightGBMBundleError(
            f"{description} has no reliable filesystem identity"
        )
    return metadata


def _require_directory(path: Path, *, description: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise LightGBMBundleError(f"cannot inspect {description}") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise LightGBMBundleError(
            f"{description} must be a real directory, not a symlink/reparse point"
        )
    return _require_reliable_identity(metadata, description=description)


def _require_regular_file(path: Path, *, description: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise LightGBMBundleError(f"cannot inspect {description}") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise LightGBMBundleError(
            f"{description} must be a regular non-link file"
        )
    return _require_reliable_identity(metadata, description=description)


def _read_regular_limited(
    path: Path,
    *,
    maximum_bytes: int,
    description: str,
    expected_metadata: os.stat_result,
) -> bytes:
    current = _require_regular_file(path, description=description)
    if not _same_snapshot(current, expected_metadata):
        raise LightGBMBundleError(f"{description} changed after enumeration")
    if current.st_size < 0 or current.st_size > maximum_bytes:
        raise LightGBMBundleError(f"{description} exceeds its safe size limit")
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise LightGBMBundleError(
            f"{description} could not be opened safely"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _is_reparse(opened)
            or not _same_identity(opened, current)
        ):
            raise LightGBMBundleError(f"{description} changed before it was opened")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(maximum_bytes + 1)
        finished = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if not _same_snapshot(opened, finished):
        raise LightGBMBundleError(f"{description} changed while being read")
    if len(payload) != opened.st_size or len(payload) > maximum_bytes:
        raise LightGBMBundleError(
            f"{description} changed while being read or is oversized"
        )
    after = _require_regular_file(path, description=description)
    if not _same_snapshot(after, expected_metadata):
        raise LightGBMBundleError(f"{description} changed after it was read")
    return payload


@dataclass(frozen=True)
class _BundleFile:
    metadata: os.stat_result
    payload: bytes


@dataclass(frozen=True)
class _BundleTree:
    ancestors: Mapping[Path, os.stat_result]
    root: os.stat_result
    files: Mapping[str, _BundleFile]


def _bundle_tree(root: Path) -> _BundleTree:
    ancestors: dict[Path, os.stat_result] = {}
    chain = tuple(reversed((root, *root.parents)))
    for path in chain:
        description = "bundle root" if path == root else "bundle ancestor"
        ancestors[path] = _require_directory(path, description=description)
    root_metadata = ancestors[root]
    enumerated: dict[str, os.stat_result] = {}
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if len(enumerated) >= _MAX_BUNDLE_ENTRIES:
                    raise LightGBMBundleError(
                        "bundle exceeds its safe entry-count limit"
                    )
                metadata = entry.stat(follow_symlinks=False)
                if (
                    stat.S_ISLNK(metadata.st_mode)
                    or _is_reparse(metadata)
                    or not stat.S_ISREG(metadata.st_mode)
                ):
                    raise LightGBMBundleError(
                        f"bundle entry is not a regular non-link file: {entry.name}"
                    )
                path_metadata = _require_regular_file(
                    Path(entry.path), description=f"bundle file {entry.name!r}"
                )
                entry_identity = _reliable_identity(metadata)
                if entry_identity is not None and not _same_identity(
                    metadata, path_metadata
                ):
                    raise LightGBMBundleError(
                        "bundle file changed after directory enumeration"
                    )
                enumerated[entry.name] = path_metadata
    except OSError as exc:
        raise LightGBMBundleError("cannot enumerate bundle root") from exc

    files: dict[str, _BundleFile] = {}
    for filename, metadata in enumerated.items():
        maximum_bytes = (
            65
            if filename == MANIFEST_HASH_FILENAME
            else _MAX_MANIFEST_BYTES
            if filename == MANIFEST_FILENAME
            else _MAX_ENCODER_BYTES
            if filename == ENCODER_FILENAME
            else _MAX_MODEL_BYTES
            if filename.startswith("model-") and filename.endswith(".txt")
            else _MAX_MANIFEST_BYTES
        )
        payload = _read_regular_limited(
            root / filename,
            maximum_bytes=maximum_bytes,
            description=f"bundle file {filename!r}",
            expected_metadata=metadata,
        )
        files[filename] = _BundleFile(metadata, payload)
    tree = _BundleTree(
        ancestors=MappingProxyType(ancestors),
        root=root_metadata,
        files=MappingProxyType(files),
    )
    _verify_bundle_tree(root, tree)
    return tree


def _verify_bundle_tree(root: Path, tree: _BundleTree) -> None:
    for path, expected in tree.ancestors.items():
        description = "bundle root" if path == root else "bundle ancestor"
        actual = _require_directory(path, description=description)
        if path == root:
            if not _same_snapshot(actual, expected):
                raise LightGBMBundleError("bundle root changed during load")
        elif not _same_identity(actual, expected):
            raise LightGBMBundleError("bundle ancestor changed during load")
    try:
        actual_names = {entry.name for entry in os.scandir(root)}
    except OSError as exc:
        raise LightGBMBundleError("cannot re-enumerate bundle root") from exc
    if actual_names != set(tree.files):
        raise LightGBMBundleError("bundle file set changed during load")
    for filename, expected in tree.files.items():
        actual = _require_regular_file(
            root / filename, description=f"bundle file {filename!r}"
        )
        if not _same_snapshot(actual, expected.metadata):
            raise LightGBMBundleError("bundle file changed during load")


def _validated_manifest(tree: _BundleTree) -> dict[str, Any]:
    for filename in (MANIFEST_FILENAME, MANIFEST_HASH_FILENAME):
        if filename not in tree.files:
            raise LightGBMBundleError(f"required regular file is missing: {filename}")
    checksum_bytes = tree.files[MANIFEST_HASH_FILENAME].payload
    if len(checksum_bytes) != 65 or not checksum_bytes.endswith(b"\n"):
        raise LightGBMBundleError(
            "manifest checksum must be 64 lowercase hex bytes plus newline"
        )
    try:
        checksum = checksum_bytes[:64].decode("ascii")
    except UnicodeDecodeError as exc:
        raise LightGBMBundleError("manifest checksum is not ASCII") from exc
    if _SHA256.fullmatch(checksum) is None:
        raise LightGBMBundleError("manifest checksum is malformed")
    manifest_bytes = tree.files[MANIFEST_FILENAME].payload
    if _sha256(manifest_bytes) != checksum:
        raise LightGBMBundleError("manifest checksum mismatch")
    manifest = _parse_json(manifest_bytes, description="bundle manifest")
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "estimator",
            "scope",
            "quantiles",
            "best_iterations",
            "encoder",
            "models",
            "runtime",
            "fit_report",
        },
        description="bundle manifest",
    )
    if (
        _require_int(manifest["schema_version"], description="bundle schema version")
        != BUNDLE_SCHEMA_VERSION
    ):
        raise LightGBMBundleError("unsupported bundle schema version")
    return manifest


def _parse_quantiles(manifest: Mapping[str, Any]) -> tuple[tuple[float, ...], tuple[str, ...]]:
    quantiles: list[float] = []
    identifiers: list[str] = []
    for index, raw in enumerate(
        _require_list(manifest["quantiles"], description="quantiles")
    ):
        record = _require_mapping(raw, description=f"quantile {index}")
        _require_exact_keys(record, {"id", "value"}, description=f"quantile {index}")
        quantile = _require_float(record["value"], description=f"quantile {index} value")
        identifier = _require_string(record["id"], description=f"quantile {index} id")
        if identifier != _quantile_id(quantile):
            raise LightGBMBundleError("quantile identifier does not match its value")
        quantiles.append(quantile)
        identifiers.append(identifier)
    values = tuple(quantiles)
    _validate_quantiles(values)
    if len(identifiers) != len(set(identifiers)):
        raise LightGBMBundleError("quantile identifiers must be unique")
    return values, tuple(identifiers)


def _parse_fit_report(
    value: Any,
    *,
    quantiles: tuple[float, ...],
    identifiers: tuple[str, ...],
    best_iterations: Mapping[str, int],
    encoder_hash: str,
    lightgbm_version: str,
    numpy_version: str,
) -> LightGBMFitReport:
    report = _require_mapping(value, description="fit report")
    _require_exact_keys(
        report,
        {
            "train_point_hash",
            "validation_point_hash",
            "train_point_count",
            "validation_point_count",
            "platform",
            "quantile_reports",
        },
        description="fit report",
    )
    quantile_reports_raw = _require_mapping(
        report["quantile_reports"], description="quantile fit reports"
    )
    if set(quantile_reports_raw) != set(identifiers):
        raise LightGBMBundleError("fit-report quantile mapping is incomplete")
    quantile_reports: list[QuantileFitReport] = []
    for quantile, identifier in zip(quantiles, identifiers):
        raw = _require_mapping(
            quantile_reports_raw[identifier], description=f"fit report {identifier}"
        )
        _require_exact_keys(
            raw,
            {"seed", "best_validation_loss", "validation_history", "parameters"},
            description=f"fit report {identifier}",
        )
        history = tuple(
            _require_float(entry, description=f"validation history {identifier}")
            for entry in _require_list(
                raw["validation_history"],
                description=f"validation history {identifier}",
            )
        )
        parameters = _require_mapping(
            raw["parameters"], description=f"parameters {identifier}"
        )
        if _require_float(
            parameters.get("alpha"), description=f"parameter alpha {identifier}"
        ) != quantile:
            raise LightGBMBundleError("fit-report alpha does not match quantile")
        quantile_reports.append(
            QuantileFitReport(
                quantile=quantile,
                seed=_require_int(raw["seed"], description=f"seed {identifier}"),
                best_iteration=best_iterations[identifier],
                best_validation_loss=_require_float(
                    raw["best_validation_loss"],
                    description=f"best validation loss {identifier}",
                ),
                validation_history=history,
                parameters=parameters,
            )
        )
    return LightGBMFitReport(
        estimator_version=LIGHTGBM_ESTIMATOR_VERSION,
        encoder_schema_hash=encoder_hash,
        train_point_hash=_require_string(
            report["train_point_hash"], description="train point hash"
        ),
        validation_point_hash=_require_string(
            report["validation_point_hash"], description="validation point hash"
        ),
        train_point_count=_require_int(
            report["train_point_count"], description="train point count", minimum=1
        ),
        validation_point_count=_require_int(
            report["validation_point_count"],
            description="validation point count",
            minimum=1,
        ),
        lightgbm_version=lightgbm_version,
        numpy_version=numpy_version,
        platform=_require_string(report["platform"], description="training platform"),
        quantiles=tuple(quantile_reports),
    )


def load_lightgbm_bundle(directory: str | Path) -> FittedLightGBMQuantiles:
    """Load and validate a bundle.

    Extra files, symlinks, checksum mismatches, schema drift, an incompatible
    LightGBM/NumPy major version, and scope/model inconsistencies are rejected.
    """

    source = Path(directory).expanduser().absolute()
    if not source.exists() and not source.is_symlink():
        raise LightGBMBundleError(f"bundle directory does not exist: {source}")
    tree = _bundle_tree(source)
    manifest = _validated_manifest(tree)
    # Manifest parsing is an intentional mutation boundary.  Revalidate before
    # trusting any filename or scope selected by its contents.
    _verify_bundle_tree(source, tree)

    estimator = _require_mapping(manifest["estimator"], description="estimator")
    _require_exact_keys(estimator, {"id", "version"}, description="estimator")
    if _require_string(estimator["id"], description="estimator id") != _ESTIMATOR_ID:
        raise LightGBMBundleError("unsupported bundle estimator")
    if (
        _require_int(estimator["version"], description="estimator version")
        != LIGHTGBM_ESTIMATOR_VERSION
    ):
        raise LightGBMBundleError("unsupported estimator version")

    quantiles, identifiers = _parse_quantiles(manifest)
    best_raw = _require_mapping(manifest["best_iterations"], description="best iterations")
    if set(best_raw) != set(identifiers):
        raise LightGBMBundleError("best-iteration mapping does not match quantiles")
    best_iterations_by_id = {
        identifier: _require_int(
            best_raw[identifier], description=f"best iteration {identifier}", minimum=1
        )
        for identifier in identifiers
    }

    scope = _require_mapping(manifest["scope"], description="scope")
    _require_exact_keys(
        scope,
        {"dataset_id", "position", "prediction_target", "condition_ids"},
        description="scope",
    )
    dataset_id = _require_string(scope["dataset_id"], description="dataset id")
    try:
        position = PredictionPosition(
            _require_string(scope["position"], description="prediction position")
        )
        target = PredictionTarget(
            _require_string(scope["prediction_target"], description="prediction target")
        )
    except ValueError as exc:
        raise LightGBMBundleError("bundle scope contains an unsupported enum value") from exc
    condition_ids = tuple(
        _require_string(value, description="condition id")
        for value in _require_list(scope["condition_ids"], description="condition ids")
    )
    if not condition_ids or tuple(sorted(set(condition_ids))) != condition_ids:
        raise LightGBMBundleError("condition ids must be non-empty, sorted, and unique")

    encoder_record = _require_mapping(manifest["encoder"], description="encoder")
    _require_exact_keys(
        encoder_record,
        {"filename", "sha256", "content_hash", "schema_version"},
        description="encoder",
    )
    if encoder_record["filename"] != ENCODER_FILENAME:
        raise LightGBMBundleError("unsupported encoder filename")
    if (
        _require_int(encoder_record["schema_version"], description="encoder version")
        != ENCODER_SCHEMA_VERSION
    ):
        raise LightGBMBundleError("unsupported encoder schema version")
    encoder_sha = _require_string(encoder_record["sha256"], description="encoder SHA256")
    encoder_hash = _require_string(
        encoder_record["content_hash"], description="encoder content hash"
    )
    if _SHA256.fullmatch(encoder_sha) is None or _SHA256.fullmatch(encoder_hash) is None:
        raise LightGBMBundleError("encoder hash is malformed")

    models_raw = _require_mapping(manifest["models"], description="models")
    if set(models_raw) != set(identifiers):
        raise LightGBMBundleError("model mapping does not match quantiles")
    model_records: dict[str, tuple[str, str]] = {}
    for identifier in identifiers:
        model = _require_mapping(models_raw[identifier], description=f"model {identifier}")
        _require_exact_keys(model, {"filename", "sha256"}, description=f"model {identifier}")
        filename = _require_string(model["filename"], description=f"filename {identifier}")
        checksum = _require_string(model["sha256"], description=f"SHA256 {identifier}")
        if filename != _model_filename(identifier) or Path(filename).name != filename:
            raise LightGBMBundleError("model filename does not match quantile")
        if _SHA256.fullmatch(checksum) is None:
            raise LightGBMBundleError("model SHA256 is malformed")
        model_records[identifier] = (filename, checksum)

    expected_files = {
        MANIFEST_FILENAME,
        MANIFEST_HASH_FILENAME,
        ENCODER_FILENAME,
        *(filename for filename, _ in model_records.values()),
    }
    actual_files = set(tree.files)
    if actual_files != expected_files:
        raise LightGBMBundleError(
            "bundle file set does not match manifest; "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )
    encoder_bytes = tree.files[ENCODER_FILENAME].payload
    if _sha256(encoder_bytes) != encoder_sha:
        raise LightGBMBundleError("encoder file checksum mismatch")
    encoder_payload = _parse_json(encoder_bytes, description="encoder schema")
    _validate_encoder_payload(encoder_payload)
    try:
        encoder = FoldTabularEncoder.from_dict(encoder_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise LightGBMBundleError("encoder schema is invalid") from exc
    if encoder.schema.content_hash != encoder_hash:
        raise LightGBMBundleError("encoder content hash mismatch")

    runtime = _require_mapping(manifest["runtime"], description="runtime")
    _require_exact_keys(
        runtime, {"lightgbm_version", "numpy_version"}, description="runtime"
    )
    lightgbm_version = _require_string(
        runtime["lightgbm_version"], description="training LightGBM version"
    )
    numpy_version = _require_string(
        runtime["numpy_version"], description="training NumPy version"
    )
    lgb, np = _load_optional_dependencies()
    if _version_major(
        lightgbm_version, description="training LightGBM version"
    ) != _version_major(lgb.__version__, description="runtime LightGBM version"):
        raise LightGBMBundleError("bundle LightGBM major version is incompatible")
    if _version_major(
        numpy_version, description="training NumPy version"
    ) != _version_major(np.__version__, description="runtime NumPy version"):
        raise LightGBMBundleError("bundle NumPy major version is incompatible")

    fit_report = _parse_fit_report(
        manifest["fit_report"],
        quantiles=quantiles,
        identifiers=identifiers,
        best_iterations=best_iterations_by_id,
        encoder_hash=encoder_hash,
        lightgbm_version=lightgbm_version,
        numpy_version=numpy_version,
    )

    boosters: dict[float, Any] = {}
    best_iterations: dict[float, int] = {}
    for quantile, identifier in zip(quantiles, identifiers):
        filename, checksum = model_records[identifier]
        model_bytes = tree.files[filename].payload
        if _sha256(model_bytes) != checksum:
            raise LightGBMBundleError(f"model file checksum mismatch: {filename}")
        try:
            model_text = model_bytes.decode("utf-8")
            booster = lgb.Booster(model_str=model_text)
        except (UnicodeDecodeError, lgb.basic.LightGBMError) as exc:
            raise LightGBMBundleError(f"LightGBM model is invalid: {filename}") from exc
        best_iteration = best_iterations_by_id[identifier]
        if booster.num_feature() != len(encoder.schema.feature_names):
            raise LightGBMBundleError("model feature count does not match encoder")
        if tuple(booster.feature_name()) != encoder.schema.feature_names:
            raise LightGBMBundleError("model feature names do not match encoder")
        if booster.current_iteration() != best_iteration:
            raise LightGBMBundleError("model iteration count does not match manifest")
        objective = str(booster.params.get("objective") or "")
        try:
            model_alpha = float(booster.params.get("alpha"))
        except (TypeError, ValueError) as exc:
            raise LightGBMBundleError("model does not declare a valid quantile alpha") from exc
        if objective != "quantile" or not math.isclose(
            model_alpha, quantile, rel_tol=0.0, abs_tol=1e-15
        ):
            raise LightGBMBundleError("model objective/alpha does not match quantile mapping")
        boosters[quantile] = booster
        best_iterations[quantile] = best_iteration

    # Booster construction can execute native parsing code for long enough for
    # a concurrent actor to replace a member or the directory.  The returned
    # model is accepted only if the complete on-disk tree still has the same
    # identities and metadata as the bytes from which it was constructed.
    _verify_bundle_tree(source, tree)

    return FittedLightGBMQuantiles(
        estimator_id=_ESTIMATOR_ID,
        target=target,
        position=position,
        dataset_id=dataset_id,
        allowed_condition_ids=condition_ids,
        encoder=encoder,
        boosters=boosters,
        best_iterations=best_iterations,
        quantiles=quantiles,
        fit_report=fit_report,
    )


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "LightGBMBundleError",
    "lightgbm_bundle_files",
    "load_lightgbm_bundle",
    "save_lightgbm_bundle",
]
