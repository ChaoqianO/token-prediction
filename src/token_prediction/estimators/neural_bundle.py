"""Portable, fail-closed persistence for Independent MLP estimators.

Only canonical JSON and ``safetensors`` are accepted.  Bundle traversal,
backslashes, absolute paths, symlinks/reparse points, stale files, missing
files, and checksum drift all fail closed before model construction.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

from token_prediction import __version__ as TOKEN_PREDICTION_VERSION
from token_prediction.contracts import SourceDescriptor
from token_prediction.dataset import (
    CAPABILITY_DATASET_SCHEMA_VERSION,
    PredictionPosition,
    PredictionTarget,
    decide_target_capability,
    supported_input_contract_hashes_from_capability,
)
from token_prediction.features import FEATURE_SCHEMA_VERSION

from .mlp import (
    INDEPENDENT_MLP_ESTIMATOR_VERSION,
    FittedIndependentMLP,
    MLPArchitecture,
    MLPFitReport,
    _build_network,
    _load_neural_dependencies,
    _valid_quantiles,
    _validate_fit_report_semantics,
)
from .neural_encoder import (
    NEURAL_ENCODER_SCHEMA_VERSION,
    NeuralFeatureEncoder,
    OptionalNeuralDependencyError,
)


NEURAL_BUNDLE_SCHEMA_VERSION = 2
NEURAL_BUNDLE_LEGACY_SCHEMA_VERSION = 1
NEURAL_COMPONENT_SCHEMA_VERSION = 1
_ESTIMATOR_ID = "independent_mlp"
_MANIFEST = "manifest.json"
_MANIFEST_HASH = "manifest.sha256"
_CALIBRATOR = "calibrator.json"
_COMPONENT_DESCRIPTOR = "component.json"
_ENCODER = "encoder.json"
_ARCHITECTURE = "architecture.json"
_WEIGHTS = "weights.safetensors"
_SHA256_LENGTH = 64
_MAX_MANIFEST_BYTES = 1_048_576
_MAX_JSON_COMPONENT_BYTES = 1_048_576
_MAX_ENCODER_BYTES = 16_777_216
_MAX_WEIGHTS_BYTES = 536_870_912
_MAX_BUNDLE_ENTRIES = 64
_MAX_BUNDLE_DEPTH = 8
_POINT_PROVENANCE_KEYS_V1 = {
    "bundle_role",
    "candidate_id",
    "candidate_hash",
    "candidate_graph",
    "dataset_id",
    "dataset_schema_version",
    "source_descriptor_hash",
    "capability_contract_hash",
    "input_contract_hash",
    "split_plan_id",
    "eligibility_hash",
    "feature_set_hash",
    "feature_schema_version",
    "position",
    "target",
    "condition_id",
    "fold",
    "interval_alpha",
    "calibrator_id",
    "code_hash",
}
_POINT_PROVENANCE_KEYS_V2 = _POINT_PROVENANCE_KEYS_V1 | {"source_descriptor"}
_CANDIDATE_GRAPH_KEYS = {
    "initializer_estimator_id",
    "updater_estimator_id",
    "lifecycle_schema_id",
    "seed_policy_id",
    "inner_split_policy_id",
}


class NeuralBundleError(ValueError):
    pass


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NeuralBundleError("bundle metadata is not canonical JSON data") from exc


def _reject_constant(value: str) -> None:
    raise NeuralBundleError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NeuralBundleError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _parse_canonical_json(payload: bytes, *, description: str) -> Any:
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NeuralBundleError(f"{description} is not valid UTF-8 JSON") from exc
    if _json_bytes(document) != payload:
        raise NeuralBundleError(f"{description} is not canonical JSON")
    return document


def _mapping(value: Any, *, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise NeuralBundleError(f"{description} must be a JSON object")
    return value


def _list(value: Any, *, description: str) -> list[Any]:
    if not isinstance(value, list):
        raise NeuralBundleError(f"{description} must be a JSON list")
    return value


def _keys(value: Mapping[str, Any], expected: set[str], *, description: str) -> None:
    if set(value) != expected:
        raise NeuralBundleError(
            f"{description} keys do not match schema; "
            f"missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _string(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise NeuralBundleError(f"{description} must be a non-empty string")
    return value


def _integer(value: Any, *, description: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise NeuralBundleError(f"{description} must be an integer >= {minimum}")
    return value


def _floating(value: Any, *, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NeuralBundleError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise NeuralBundleError(f"{description} must be finite")
    return result


def _checksum(value: Any, *, description: str) -> str:
    result = _string(value, description=description)
    if len(result) != _SHA256_LENGTH or any(character not in "0123456789abcdef" for character in result):
        raise NeuralBundleError(f"{description} is not a lowercase SHA-256")
    return result


def _relative_posix(value: Any, *, description: str) -> str:
    result = _string(value, description=description)
    if "\\" in result:
        raise NeuralBundleError(f"{description} must not contain backslashes")
    path = PurePosixPath(result)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise NeuralBundleError(f"{description} must be a normalized relative POSIX path")
    if path.as_posix() != result:
        raise NeuralBundleError(f"{description} must be a normalized relative POSIX path")
    return result


def _version_major(value: str, *, description: str) -> int:
    try:
        result = int(value.split(".", 1)[0])
    except (TypeError, ValueError) as exc:
        raise NeuralBundleError(f"{description} has no numeric major version") from exc
    if result < 0:
        raise NeuralBundleError(f"{description} has an invalid major version")
    return result


def _version_prefix(
    value: str,
    *,
    components: int,
    description: str,
) -> tuple[int, ...]:
    parts = value.split(".")
    if len(parts) < components:
        raise NeuralBundleError(f"{description} is not a compatible version string")
    try:
        result = tuple(int(part) for part in parts[:components])
    except ValueError as exc:
        raise NeuralBundleError(
            f"{description} is not a compatible version string"
        ) from exc
    if any(part < 0 for part in result):
        raise NeuralBundleError(f"{description} is not a compatible version string")
    return result


def _runtime_versions(value: Any, *, legacy: bool) -> dict[str, str]:
    runtime = _mapping(value, description="runtime")
    expected = (
        {"numpy_version", "torch_version", "safetensors_version"}
        if legacy
        else {
            "python_version",
            "token_prediction_version",
            "numpy_version",
            "torch_version",
            "safetensors_version",
        }
    )
    _keys(runtime, expected, description="runtime")
    normalized = {
        name: _string(runtime[name], description=f"runtime {name}")
        for name in sorted(expected)
    }
    if not legacy:
        if _version_prefix(
            normalized["python_version"],
            components=2,
            description="bundle Python version",
        ) != _version_prefix(
            platform.python_version(),
            components=2,
            description="current Python version",
        ):
            raise NeuralBundleError("bundle Python major/minor version is incompatible")
        if _version_prefix(
            normalized["token_prediction_version"],
            components=2,
            description="bundle token-prediction version",
        ) != _version_prefix(
            TOKEN_PREDICTION_VERSION,
            components=2,
            description="current token-prediction version",
        ):
            raise NeuralBundleError(
                "bundle token-prediction major/minor version is incompatible"
            )
    return normalized


def _load_safetensors() -> tuple[Any, Any, str]:
    try:
        import safetensors
        from safetensors.torch import load as load_tensors
        from safetensors.torch import save as save_tensors
    except ModuleNotFoundError as exc:  # pragma: no cover - base-only CI exercises this
        raise OptionalNeuralDependencyError(
            "neural bundle persistence requires optional dependencies; "
            "install token-prediction[neural]"
        ) from exc
    return load_tensors, save_tensors, str(safetensors.__version__)


def _quantile_id(value: float) -> str:
    # The canonical decimal representation is collision-free for binary64.
    return f"q-{value.hex().replace('.', '_').replace('+', 'p').replace('-', 'm')}"


def _identity_calibrator(interval_alpha: float) -> dict[str, Any]:
    return {
        "calibrator_schema_version": 1,
        "calibrator_id": "none",
        "interval_alpha": interval_alpha,
        "expansion": 0.0,
    }


def _validated_point_provenance(
    value: Any,
    *,
    dataset_id: str,
    input_contract_hash: str | None,
    position: PredictionPosition,
    target: PredictionTarget,
    condition_ids: tuple[str, ...],
    calibrator_id: str,
    interval_alpha: float,
    bundle_schema_version: int = NEURAL_BUNDLE_SCHEMA_VERSION,
) -> Mapping[str, Any]:
    provenance = _mapping(value, description="provenance")
    legacy = bundle_schema_version == NEURAL_BUNDLE_LEGACY_SCHEMA_VERSION
    _keys(
        provenance,
        _POINT_PROVENANCE_KEYS_V1 if legacy else _POINT_PROVENANCE_KEYS_V2,
        description="point-model provenance",
    )
    if provenance["bundle_role"] != "point_model":
        raise NeuralBundleError("unsupported neural bundle provenance role")
    _string(provenance["candidate_id"], description="provenance candidate id")
    _checksum(provenance["candidate_hash"], description="provenance candidate hash")
    graph = _mapping(provenance["candidate_graph"], description="candidate graph")
    _keys(graph, _CANDIDATE_GRAPH_KEYS, description="candidate graph")
    graph_values = {
        key: _string(graph[key], description=f"candidate graph {key}")
        for key in _CANDIDATE_GRAPH_KEYS
    }
    if graph_values != {
        "initializer_estimator_id": "none",
        "updater_estimator_id": _ESTIMATOR_ID,
        "lifecycle_schema_id": "point_cell_v1",
        "seed_policy_id": "none",
        "inner_split_policy_id": "none",
    }:
        raise NeuralBundleError(
            "Independent MLP point bundle has an incompatible candidate graph"
        )
    if _string(provenance["dataset_id"], description="provenance dataset id") != dataset_id:
        raise NeuralBundleError("provenance dataset id does not match bundle scope")
    dataset_schema_version = _integer(
        provenance["dataset_schema_version"],
        description="provenance dataset schema version",
        minimum=1,
    )
    if not legacy and dataset_schema_version != CAPABILITY_DATASET_SCHEMA_VERSION:
        raise NeuralBundleError("unsupported provenance dataset schema version")
    for key in (
        "source_descriptor_hash",
        "capability_contract_hash",
        "split_plan_id",
        "eligibility_hash",
        "feature_set_hash",
        "code_hash",
    ):
        _checksum(provenance[key], description=f"provenance {key}")
    source_descriptor_hash = _checksum(
        provenance["source_descriptor_hash"],
        description="provenance source descriptor hash",
    )
    capability_contract_hash = _checksum(
        provenance["capability_contract_hash"],
        description="provenance capability contract hash",
    )
    declared_input_contract = _checksum(
        provenance["input_contract_hash"],
        description="provenance input contract hash",
    )
    if input_contract_hash is None or declared_input_contract != input_contract_hash:
        raise NeuralBundleError(
            "provenance input contract hash does not match bundle scope"
        )
    feature_schema_version = _integer(
        provenance["feature_schema_version"],
        description="provenance feature schema version",
        minimum=1,
    )
    if not legacy and feature_schema_version != FEATURE_SCHEMA_VERSION:
        raise NeuralBundleError("unsupported provenance feature schema version")
    if provenance["position"] != position.value:
        raise NeuralBundleError("provenance position does not match bundle scope")
    if provenance["target"] != target.value:
        raise NeuralBundleError("provenance target does not match bundle scope")
    if not legacy:
        descriptor_document = _mapping(
            provenance["source_descriptor"],
            description="provenance source descriptor",
        )
        try:
            descriptor = SourceDescriptor.from_dict(descriptor_document)
        except (TypeError, ValueError) as exc:
            raise NeuralBundleError("provenance source descriptor is invalid") from exc
        if descriptor.descriptor_hash != source_descriptor_hash:
            raise NeuralBundleError("provenance source descriptor hash does not match")
        if descriptor.capabilities.contract_hash != capability_contract_hash:
            raise NeuralBundleError("provenance capability contract hash does not match")
        decision = decide_target_capability(descriptor.capabilities, position, target)
        if not decision.available:
            raise NeuralBundleError(
                "source capability contract cannot produce the neural bundle target"
            )
        allowed_input_contracts = supported_input_contract_hashes_from_capability(
            capability_contract_hash
        )
        if declared_input_contract not in allowed_input_contracts:
            raise NeuralBundleError(
                "provenance input contract does not match its capability contract"
            )
    condition_id = _string(
        provenance["condition_id"], description="provenance condition id"
    )
    if condition_ids != (condition_id,):
        raise NeuralBundleError(
            "point-model provenance must bind exactly one bundle condition"
        )
    _integer(provenance["fold"], description="provenance fold")
    declared_alpha = _floating(
        provenance["interval_alpha"], description="provenance interval alpha"
    )
    if not math.isclose(
        declared_alpha,
        interval_alpha,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise NeuralBundleError("provenance alpha does not match calibrator")
    if (
        _string(provenance["calibrator_id"], description="provenance calibrator id")
        != calibrator_id
    ):
        raise NeuralBundleError("provenance calibrator id does not match calibrator")
    return provenance


def _validated_expected_source_provenance(value: Any) -> dict[str, Any]:
    document = _mapping(value, description="expected source provenance")
    _keys(
        document,
        {
            "source_descriptor",
            "source_descriptor_hash",
            "code_hash",
            "runtime_versions",
        },
        description="expected source provenance",
    )
    descriptor_document = _mapping(
        document["source_descriptor"],
        description="expected source descriptor",
    )
    try:
        descriptor = SourceDescriptor.from_dict(descriptor_document)
    except (TypeError, ValueError) as exc:
        raise NeuralBundleError("expected source descriptor is invalid") from exc
    descriptor_hash = _checksum(
        document["source_descriptor_hash"],
        description="expected source descriptor hash",
    )
    if descriptor.descriptor_hash != descriptor_hash:
        raise NeuralBundleError("expected source descriptor document/hash mismatch")
    runtime = _mapping(
        document["runtime_versions"],
        description="expected source runtime versions",
    )
    required_runtime = {"python_version", "token_prediction_version"}
    if not required_runtime <= set(runtime):
        raise NeuralBundleError(
            "expected source runtime must bind Python and token-prediction"
        )
    normalized_runtime = {
        _string(name, description="expected runtime name"): _string(
            version,
            description=f"expected runtime version {name!r}",
        )
        for name, version in runtime.items()
    }
    return {
        "source_descriptor": descriptor.to_dict(),
        "source_descriptor_hash": descriptor_hash,
        "code_hash": _checksum(document["code_hash"], description="expected code hash"),
        "runtime_versions": normalized_runtime,
    }


def _fit_report_document(report: MLPFitReport) -> dict[str, Any]:
    return {
        "train_point_hash": report.train_point_hash,
        "validation_point_hash": report.validation_point_hash,
        "train_point_count": report.train_point_count,
        "validation_point_count": report.validation_point_count,
        "seed": report.seed,
        "best_epoch": report.best_epoch,
        "best_validation_loss": report.best_validation_loss,
        "validation_history": list(report.validation_history),
        "parameters": dict(report.parameters),
        "platform": report.platform,
    }


def _state_for_safetensors(fitted: FittedIndependentMLP, torch: Any) -> dict[str, Any]:
    reference = _build_network(torch, fitted.architecture)
    expected = reference.state_dict()
    actual = fitted.model.state_dict()
    if set(actual) != set(expected):
        raise NeuralBundleError("model state keys do not match the declared architecture")
    tensors: dict[str, Any] = {}
    for name in sorted(expected):
        tensor = actual[name]
        if tuple(tensor.shape) != tuple(expected[name].shape):
            raise NeuralBundleError(f"model tensor shape does not match architecture: {name}")
        if tensor.dtype != torch.float32:
            raise NeuralBundleError(f"model tensor must use float32: {name}")
        tensor = tensor.detach().to(device="cpu").contiguous()
        if not bool(torch.isfinite(tensor).all()):
            raise NeuralBundleError(f"model tensor is non-finite: {name}")
        tensors[name] = tensor
    return tensors


def neural_bundle_files(
    fitted: FittedIndependentMLP,
    *,
    calibrator: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Mapping[str, bytes]:
    """Return the complete nested bundle as a relative-POSIX file mapping."""

    np, torch = _load_neural_dependencies()
    _, save_tensors, safetensors_version = _load_safetensors()
    calibrator = (
        calibrator
        if calibrator is not None
        else fitted.calibrator_document
        if fitted.calibrator_document is not None
        else _identity_calibrator(2 * fitted.quantiles[0])
    )
    provenance_document = (
        provenance
        if provenance is not None
        else fitted.provenance
        if fitted.provenance is not None
        else {}
    )
    calibrator = _mapping(calibrator, description="calibrator document")
    try:
        from token_prediction.evaluation.calibration import FittedExpansionCalibrator

        parsed_calibrator = FittedExpansionCalibrator.from_dict(calibrator)
    except (KeyError, TypeError, ValueError) as exc:
        raise NeuralBundleError("calibrator document is invalid") from exc
    if not math.isclose(
        parsed_calibrator.interval_alpha,
        2 * fitted.quantiles[0],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise NeuralBundleError("calibrator alpha does not match fitted quantiles")
    provenance_document = _validated_point_provenance(
        provenance_document,
        dataset_id=fitted.dataset_id,
        input_contract_hash=fitted.input_contract_hash,
        position=fitted.position,
        target=fitted.target,
        condition_ids=fitted.allowed_condition_ids,
        calibrator_id=parsed_calibrator.calibrator_id,
        interval_alpha=parsed_calibrator.interval_alpha,
    )
    encoder_bytes = _json_bytes(fitted.encoder.to_dict())
    architecture_bytes = _json_bytes(fitted.architecture.to_dict())
    try:
        weights_bytes = save_tensors(_state_for_safetensors(fitted, torch))
    except Exception as exc:
        if isinstance(exc, NeuralBundleError):
            raise
        raise NeuralBundleError("could not serialize model weights as safetensors") from exc
    component = {
        "schema_version": NEURAL_COMPONENT_SCHEMA_VERSION,
        "type": _ESTIMATOR_ID,
        "estimator_version": INDEPENDENT_MLP_ESTIMATOR_VERSION,
        "files": {
            "architecture": {"path": _ARCHITECTURE, "sha256": _sha256(architecture_bytes)},
            "encoder": {
                "path": _ENCODER,
                "sha256": _sha256(encoder_bytes),
                "content_hash": fitted.encoder.schema.content_hash,
                "schema_version": fitted.encoder.schema.schema_version,
            },
            "weights": {"path": _WEIGHTS, "sha256": _sha256(weights_bytes)},
        },
    }
    component_bytes = _json_bytes(component)
    component_hash = _sha256(component_bytes)
    component_path = f"components/{component_hash}"
    calibrator_bytes = _json_bytes(dict(calibrator))
    quantile_records = [
        {"id": _quantile_id(value), "value": value} for value in fitted.quantiles
    ]
    manifest = {
        "schema_version": NEURAL_BUNDLE_SCHEMA_VERSION,
        "bundle_type": "composite_neural",
        "estimator": {
            "id": fitted.estimator_id,
            "version": fitted.fit_report.estimator_version,
        },
        "scope": {
            "dataset_id": fitted.dataset_id,
            "input_contract_hash": fitted.input_contract_hash,
            "position": fitted.position.value,
            "prediction_target": fitted.target.value,
            "condition_ids": list(fitted.allowed_condition_ids),
        },
        "quantiles": quantile_records,
        "component": {
            "path": component_path,
            "descriptor": f"{component_path}/{_COMPONENT_DESCRIPTOR}",
            "content_hash": component_hash,
        },
        "calibrator": {"path": _CALIBRATOR, "sha256": _sha256(calibrator_bytes)},
        "provenance": dict(provenance_document),
        "runtime": {
            "python_version": platform.python_version(),
            "token_prediction_version": TOKEN_PREDICTION_VERSION,
            "numpy_version": fitted.fit_report.numpy_version,
            "torch_version": fitted.fit_report.torch_version,
            "safetensors_version": safetensors_version,
        },
        "fit_report": _fit_report_document(fitted.fit_report),
    }
    if _version_major(str(np.__version__), description="runtime NumPy version") != _version_major(
        fitted.fit_report.numpy_version, description="training NumPy version"
    ):
        raise NeuralBundleError("training and save-time NumPy major versions differ")
    if _version_major(str(torch.__version__), description="runtime PyTorch version") != _version_major(
        fitted.fit_report.torch_version, description="training PyTorch version"
    ):
        raise NeuralBundleError("training and save-time PyTorch major versions differ")
    manifest_bytes = _json_bytes(manifest)
    if len(manifest_bytes) > _MAX_MANIFEST_BYTES:
        raise NeuralBundleError("generated bundle manifest exceeds its safe size limit")
    if len(encoder_bytes) > _MAX_ENCODER_BYTES:
        raise NeuralBundleError("generated encoder exceeds its safe size limit")
    if len(weights_bytes) > _MAX_WEIGHTS_BYTES:
        raise NeuralBundleError("generated weights exceed their safe size limit")
    if any(
        len(payload) > _MAX_JSON_COMPONENT_BYTES
        for payload in (architecture_bytes, component_bytes, calibrator_bytes)
    ):
        raise NeuralBundleError("generated bundle JSON component exceeds its safe size limit")
    files = {
        _MANIFEST: manifest_bytes,
        _MANIFEST_HASH: f"{_sha256(manifest_bytes)}\n".encode("ascii"),
        _CALIBRATOR: calibrator_bytes,
        f"{component_path}/{_COMPONENT_DESCRIPTOR}": component_bytes,
        f"{component_path}/{_ENCODER}": encoder_bytes,
        f"{component_path}/{_ARCHITECTURE}": architecture_bytes,
        f"{component_path}/{_WEIGHTS}": weights_bytes,
    }
    for path in files:
        _relative_posix(path, description="generated bundle path")
    return MappingProxyType(files)


def save_neural_bundle(
    fitted: FittedIndependentMLP,
    directory: str | Path,
    *,
    calibrator: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(directory).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"bundle destination already exists: {destination}")
    files = neural_bundle_files(
        fitted,
        calibrator=calibrator,
        provenance=provenance,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    )
    try:
        for relative, payload in files.items():
            path = temporary.joinpath(*PurePosixPath(relative).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
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
    """Return an identity only when the platform supplied all identity fields.

    Some Windows filesystems expose ``st_dev == st_ino == st_nlink == 0`` for
    ``DirEntry.stat`` results.  Treating those sentinel values (or timestamps
    paired with them) as a strong identity makes an unchanged entry appear to
    drift when it is inspected later with ``Path.lstat``.
    """

    device = int(getattr(metadata, "st_dev", 0))
    inode = int(getattr(metadata, "st_ino", 0))
    links = int(getattr(metadata, "st_nlink", 0))
    # A zero device number is valid on some Windows volumes.  A zero inode (or
    # link count) is the sentinel that prevents us from proving file identity.
    if device < 0 or inode <= 0 or links <= 0:
        return None
    return device, inode


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    if stat.S_IFMT(left.st_mode) != stat.S_IFMT(right.st_mode):
        return False
    left_identity = _reliable_identity(left)
    right_identity = _reliable_identity(right)
    return (
        left_identity is not None
        and right_identity is not None
        and left_identity == right_identity
    )


def _snapshot(
    metadata: os.stat_result,
) -> tuple[int, int | None, tuple[int, int] | None, int, int]:
    """Capture metadata only after a reliable path/handle identity is known."""

    file_type = stat.S_IFMT(metadata.st_mode)
    identity = _reliable_identity(metadata)
    size = int(metadata.st_size) if stat.S_ISREG(metadata.st_mode) else None
    return (
        file_type,
        size,
        identity,
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _same_snapshot(left: os.stat_result, right: os.stat_result) -> bool:
    left_snapshot = _snapshot(left)
    right_snapshot = _snapshot(right)
    return (
        left_snapshot[2] is not None
        and right_snapshot[2] is not None
        and left_snapshot == right_snapshot
    )


def _require_reliable_identity(
    metadata: os.stat_result,
    *,
    description: str,
) -> os.stat_result:
    if _reliable_identity(metadata) is None:
        raise NeuralBundleError(
            f"{description} has no reliable file identity; refusing unsafe disk load"
        )
    return metadata


def _require_directory(path: Path, *, description: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise NeuralBundleError(f"cannot inspect {description}") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise NeuralBundleError(f"{description} must be a real directory")
    return metadata


def _require_regular_file(path: Path, *, description: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise NeuralBundleError(f"cannot inspect {description}") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise NeuralBundleError(f"{description} must be a regular non-link file")
    return metadata


def _read_regular_limited(
    path: Path,
    *,
    maximum_bytes: int,
    description: str,
    expected_metadata: os.stat_result,
) -> bytes:
    _require_reliable_identity(expected_metadata, description=description)
    metadata = _require_reliable_identity(
        _require_regular_file(path, description=description),
        description=description,
    )
    if not _same_snapshot(metadata, expected_metadata):
        raise NeuralBundleError(f"{description} changed after bundle enumeration")
    if metadata.st_size < 0 or metadata.st_size > maximum_bytes:
        raise NeuralBundleError(f"{description} exceeds its safe size limit")
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise NeuralBundleError(f"{description} could not be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(opened.st_mode)
            or _is_reparse(opened)
            or not stat.S_ISREG(opened.st_mode)
            or _reliable_identity(opened) is None
            or not _same_identity(opened, metadata)
        ):
            raise NeuralBundleError(f"{description} changed before it was opened")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(maximum_bytes + 1)
        finished = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if not _same_snapshot(opened, finished):
        raise NeuralBundleError(f"{description} changed while being read")
    if len(payload) != opened.st_size or len(payload) > maximum_bytes:
        raise NeuralBundleError(f"{description} changed while being read or is oversized")
    after = _require_reliable_identity(
        _require_regular_file(path, description=description),
        description=description,
    )
    if not _same_snapshot(after, expected_metadata):
        raise NeuralBundleError(f"{description} changed after it was read")
    return payload


@dataclass(frozen=True)
class _BundleFile:
    metadata: os.stat_result
    payload: bytes


@dataclass(frozen=True)
class _BundleTree:
    files: Mapping[str, _BundleFile]
    directories: Mapping[str, os.stat_result]


def _bundle_entries(root: Path) -> _BundleTree:
    root_metadata = _require_reliable_identity(
        _require_directory(root, description="bundle root"),
        description="bundle root",
    )
    files: dict[str, _BundleFile] = {}
    directories: dict[str, os.stat_result] = {"": root_metadata}
    entry_count = 0

    def visit(
        directory: Path,
        relative: PurePosixPath,
        expected_metadata: os.stat_result,
    ) -> None:
        nonlocal entry_count
        before = _require_reliable_identity(
            _require_directory(directory, description="bundle directory"),
            description="bundle directory",
        )
        if not _same_identity(before, expected_metadata):
            raise NeuralBundleError("bundle directory changed before enumeration")
        with os.scandir(directory) as entries:
            for entry in entries:
                entry_count += 1
                if entry_count > _MAX_BUNDLE_ENTRIES:
                    raise NeuralBundleError("bundle exceeds its safe entry-count limit")
                entry_metadata = entry.stat(follow_symlinks=False)
                child_relative = relative / entry.name
                child = child_relative.as_posix()
                _relative_posix(child, description="bundle entry")
                if len(child_relative.parts) > _MAX_BUNDLE_DEPTH:
                    raise NeuralBundleError("bundle exceeds its safe directory-depth limit")
                if stat.S_ISLNK(entry_metadata.st_mode) or _is_reparse(entry_metadata):
                    raise NeuralBundleError(f"bundle entry is a symlink/reparse point: {child}")
                path = Path(entry.path)
                if stat.S_ISDIR(entry_metadata.st_mode):
                    path_metadata = _require_reliable_identity(
                        _require_directory(path, description="bundle directory"),
                        description="bundle directory",
                    )
                    entry_identity = _reliable_identity(entry_metadata)
                    if entry_identity is not None and not _same_identity(
                        entry_metadata, path_metadata
                    ):
                        raise NeuralBundleError(
                            "bundle directory changed after directory enumeration"
                        )
                    directories[child] = path_metadata
                    visit(path, child_relative, path_metadata)
                elif stat.S_ISREG(entry_metadata.st_mode):
                    path_metadata = _require_reliable_identity(
                        _require_regular_file(path, description="bundle file"),
                        description="bundle file",
                    )
                    entry_identity = _reliable_identity(entry_metadata)
                    if entry_identity is not None and not _same_identity(
                        entry_metadata, path_metadata
                    ):
                        raise NeuralBundleError(
                            "bundle file changed after directory enumeration"
                        )
                    maximum_bytes = (
                        _MAX_MANIFEST_BYTES
                        if child == _MANIFEST
                        else 65
                        if child == _MANIFEST_HASH
                        else _MAX_WEIGHTS_BYTES
                        if child.endswith(f"/{_WEIGHTS}")
                        else _MAX_ENCODER_BYTES
                        if child.endswith(f"/{_ENCODER}")
                        else _MAX_JSON_COMPONENT_BYTES
                    )
                    payload = _read_regular_limited(
                        path,
                        maximum_bytes=maximum_bytes,
                        description=f"bundle file {child!r}",
                        expected_metadata=path_metadata,
                    )
                    files[child] = _BundleFile(path_metadata, payload)
                else:
                    raise NeuralBundleError(f"bundle entry is not a regular file: {child}")
        after = _require_reliable_identity(
            _require_directory(directory, description="bundle directory"),
            description="bundle directory",
        )
        if not _same_snapshot(after, before):
            raise NeuralBundleError("bundle directory changed while being enumerated")

    visit(root, PurePosixPath(), root_metadata)
    return _BundleTree(
        files=MappingProxyType(files),
        directories=MappingProxyType(directories),
    )


def _verify_bundle_tree(root: Path, tree: _BundleTree) -> None:
    for relative, expected in tree.directories.items():
        path = root if not relative else root.joinpath(*PurePosixPath(relative).parts)
        actual = _require_directory(path, description="bundle directory")
        if not _same_snapshot(actual, expected):
            raise NeuralBundleError("bundle directory changed during load")
    for relative, expected in tree.files.items():
        path = root.joinpath(*PurePosixPath(relative).parts)
        actual = _require_reliable_identity(
            _require_regular_file(path, description="bundle file"),
            description="bundle file",
        )
        if not _same_snapshot(actual, expected.metadata):
            raise NeuralBundleError("bundle file changed during load")


def _validated_manifest(root: Path, tree: _BundleTree) -> Mapping[str, Any]:
    del root
    for filename in (_MANIFEST, _MANIFEST_HASH):
        if filename not in tree.files:
            raise NeuralBundleError(f"required regular file is missing: {filename}")
    checksum_bytes = tree.files[_MANIFEST_HASH].payload
    if len(checksum_bytes) != 65 or not checksum_bytes.endswith(b"\n"):
        raise NeuralBundleError("manifest checksum must be 64 lowercase hex bytes plus newline")
    try:
        checksum = checksum_bytes[:64].decode("ascii")
    except UnicodeDecodeError as exc:
        raise NeuralBundleError("manifest checksum is not ASCII") from exc
    _checksum(checksum, description="manifest checksum")
    manifest_bytes = tree.files[_MANIFEST].payload
    if _sha256(manifest_bytes) != checksum:
        raise NeuralBundleError("manifest checksum mismatch")
    manifest = _mapping(
        _parse_canonical_json(manifest_bytes, description="bundle manifest"),
        description="bundle manifest",
    )
    _keys(
        manifest,
        {
            "schema_version",
            "bundle_type",
            "estimator",
            "scope",
            "quantiles",
            "component",
            "calibrator",
            "provenance",
            "runtime",
            "fit_report",
        },
        description="bundle manifest",
    )
    if manifest["bundle_type"] != "composite_neural":
        raise NeuralBundleError("unsupported bundle type")
    return manifest


def _parse_quantiles(value: Any) -> tuple[float, float, float]:
    records = _list(value, description="quantiles")
    if len(records) != 3:
        raise NeuralBundleError("bundle must contain exactly three quantiles")
    values: list[float] = []
    for index, raw in enumerate(records):
        record = _mapping(raw, description=f"quantile {index}")
        _keys(record, {"id", "value"}, description=f"quantile {index}")
        quantile = _floating(record["value"], description=f"quantile {index} value")
        if record["id"] != _quantile_id(quantile):
            raise NeuralBundleError("quantile identifier does not match its value")
        values.append(quantile)
    result = tuple(values)
    if not _valid_quantiles(result):
        raise NeuralBundleError("bundle quantiles are not symmetric around 0.5")
    return result  # type: ignore[return-value]


def _parse_fit_report(
    value: Any,
    *,
    encoder_hash: str,
    torch_version: str,
    numpy_version: str,
) -> MLPFitReport:
    report = _mapping(value, description="fit report")
    _keys(
        report,
        {
            "train_point_hash",
            "validation_point_hash",
            "train_point_count",
            "validation_point_count",
            "seed",
            "best_epoch",
            "best_validation_loss",
            "validation_history",
            "parameters",
            "platform",
        },
        description="fit report",
    )
    history = tuple(
        _floating(item, description="validation history value")
        for item in _list(report["validation_history"], description="validation history")
    )
    parameters = _mapping(report["parameters"], description="fit parameters")
    try:
        return MLPFitReport(
            estimator_version=INDEPENDENT_MLP_ESTIMATOR_VERSION,
            encoder_schema_hash=encoder_hash,
            train_point_hash=_checksum(report["train_point_hash"], description="train point hash"),
            validation_point_hash=_checksum(
                report["validation_point_hash"], description="validation point hash"
            ),
            train_point_count=_integer(
                report["train_point_count"], description="train point count", minimum=1
            ),
            validation_point_count=_integer(
                report["validation_point_count"],
                description="validation point count",
                minimum=1,
            ),
            seed=_integer(report["seed"], description="fit seed"),
            best_epoch=_integer(report["best_epoch"], description="best epoch", minimum=1),
            best_validation_loss=_floating(
                report["best_validation_loss"], description="best validation loss"
            ),
            validation_history=history,
            parameters=parameters,
            torch_version=torch_version,
            numpy_version=numpy_version,
            platform=_string(report["platform"], description="training platform"),
        )
    except ValueError as exc:
        raise NeuralBundleError("fit report is internally inconsistent") from exc


def load_neural_bundle(
    directory: str | Path,
    *,
    expected_source_provenance: Mapping[str, Any] | None = None,
    allow_legacy_v1: bool = False,
) -> FittedIndependentMLP:
    source = Path(directory).expanduser().absolute()
    if not source.exists() and not source.is_symlink():
        raise NeuralBundleError(f"bundle directory does not exist: {source}")
    tree = _bundle_entries(source)
    actual_files = set(tree.files)
    actual_directories = set(tree.directories) - {""}
    manifest = _validated_manifest(source, tree)
    bundle_schema_version = _integer(
        manifest["schema_version"],
        description="bundle schema version",
        minimum=1,
    )
    legacy = bundle_schema_version == NEURAL_BUNDLE_LEGACY_SCHEMA_VERSION
    if bundle_schema_version != NEURAL_BUNDLE_SCHEMA_VERSION and not (
        legacy and allow_legacy_v1
    ):
        raise NeuralBundleError("unsupported neural bundle schema version")
    if legacy and expected_source_provenance is not None:
        raise NeuralBundleError(
            "legacy neural bundle cannot satisfy expected source provenance"
        )
    # The manifest read is an intentional hook boundary: revalidate the tree
    # before following any manifest-selected component path.  In particular,
    # this prevents a component directory swapped to a symlink/reparse point
    # after enumeration from being traversed for descriptor or weight reads.
    _verify_bundle_tree(source, tree)

    estimator = _mapping(manifest["estimator"], description="estimator")
    _keys(estimator, {"id", "version"}, description="estimator")
    if estimator["id"] != _ESTIMATOR_ID:
        raise NeuralBundleError("unsupported neural estimator id")
    if _integer(estimator["version"], description="estimator version", minimum=1) != INDEPENDENT_MLP_ESTIMATOR_VERSION:
        raise NeuralBundleError("unsupported neural estimator version")

    component = _mapping(manifest["component"], description="component")
    _keys(component, {"path", "descriptor", "content_hash"}, description="component")
    component_path = _relative_posix(component["path"], description="component path")
    component_hash = _checksum(component["content_hash"], description="component hash")
    if component_path != f"components/{component_hash}":
        raise NeuralBundleError("component path does not match its content hash")
    descriptor_path = _relative_posix(
        component["descriptor"], description="component descriptor path"
    )
    if descriptor_path != f"{component_path}/{_COMPONENT_DESCRIPTOR}":
        raise NeuralBundleError("component descriptor path is invalid")

    calibrator_record = _mapping(manifest["calibrator"], description="calibrator")
    _keys(calibrator_record, {"path", "sha256"}, description="calibrator")
    calibrator_path = _relative_posix(
        calibrator_record["path"], description="calibrator path"
    )
    if calibrator_path != _CALIBRATOR:
        raise NeuralBundleError("unsupported calibrator path")
    calibrator_sha = _checksum(
        calibrator_record["sha256"], description="calibrator checksum"
    )

    expected_files = {
        _MANIFEST,
        _MANIFEST_HASH,
        calibrator_path,
        descriptor_path,
        f"{component_path}/{_ENCODER}",
        f"{component_path}/{_ARCHITECTURE}",
        f"{component_path}/{_WEIGHTS}",
    }
    expected_directories = {"components", component_path}
    if actual_files != expected_files or actual_directories != expected_directories:
        raise NeuralBundleError(
            "bundle file set does not match manifest; "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}, "
            f"directory_extra={sorted(actual_directories - expected_directories)}"
        )

    descriptor_bytes = tree.files[descriptor_path].payload
    if _sha256(descriptor_bytes) != component_hash:
        raise NeuralBundleError("component descriptor content hash mismatch")
    descriptor = _mapping(
        _parse_canonical_json(descriptor_bytes, description="component descriptor"),
        description="component descriptor",
    )
    _keys(
        descriptor,
        {"schema_version", "type", "estimator_version", "files"},
        description="component descriptor",
    )
    if _integer(descriptor["schema_version"], description="component schema version", minimum=1) != NEURAL_COMPONENT_SCHEMA_VERSION:
        raise NeuralBundleError("unsupported neural component schema version")
    if descriptor["type"] != _ESTIMATOR_ID or descriptor["estimator_version"] != INDEPENDENT_MLP_ESTIMATOR_VERSION:
        raise NeuralBundleError("component estimator identity is inconsistent")
    records = _mapping(descriptor["files"], description="component files")
    _keys(records, {"architecture", "encoder", "weights"}, description="component files")

    architecture_record = _mapping(records["architecture"], description="architecture file")
    _keys(architecture_record, {"path", "sha256"}, description="architecture file")
    encoder_record = _mapping(records["encoder"], description="encoder file")
    _keys(
        encoder_record,
        {"path", "sha256", "content_hash", "schema_version"},
        description="encoder file",
    )
    weights_record = _mapping(records["weights"], description="weights file")
    _keys(weights_record, {"path", "sha256"}, description="weights file")
    component_files = (
        (architecture_record, _ARCHITECTURE, "architecture"),
        (encoder_record, _ENCODER, "encoder"),
        (weights_record, _WEIGHTS, "weights"),
    )
    payloads: dict[str, bytes] = {}
    for record, expected_name, description in component_files:
        relative_name = _relative_posix(record["path"], description=f"{description} path")
        if relative_name != expected_name:
            raise NeuralBundleError(f"unsupported {description} filename")
        checksum = _checksum(record["sha256"], description=f"{description} checksum")
        payload = tree.files[f"{component_path}/{relative_name}"].payload
        if _sha256(payload) != checksum:
            raise NeuralBundleError(f"{description} file checksum mismatch")
        payloads[description] = payload

    if _integer(encoder_record["schema_version"], description="encoder schema version", minimum=1) != NEURAL_ENCODER_SCHEMA_VERSION:
        raise NeuralBundleError("unsupported neural encoder schema version")
    encoder_hash = _checksum(encoder_record["content_hash"], description="encoder content hash")
    encoder_document = _mapping(
        _parse_canonical_json(payloads["encoder"], description="encoder schema"),
        description="encoder schema",
    )
    try:
        encoder = NeuralFeatureEncoder.from_dict(encoder_document)
    except (KeyError, TypeError, ValueError) as exc:
        raise NeuralBundleError("encoder schema is invalid") from exc
    if encoder.schema.content_hash != encoder_hash:
        raise NeuralBundleError("encoder content hash mismatch")
    architecture_document = _mapping(
        _parse_canonical_json(payloads["architecture"], description="architecture"),
        description="architecture",
    )
    try:
        architecture = MLPArchitecture.from_dict(architecture_document)
    except (TypeError, ValueError) as exc:
        raise NeuralBundleError("MLP architecture is invalid") from exc
    if architecture.input_dim != encoder.schema.output_width:
        raise NeuralBundleError("architecture input width does not match encoder")

    scope = _mapping(manifest["scope"], description="scope")
    _keys(
        scope,
        {
            "dataset_id",
            "input_contract_hash",
            "position",
            "prediction_target",
            "condition_ids",
        },
        description="scope",
    )
    dataset_id = _string(scope["dataset_id"], description="dataset id")
    input_contract_hash = (
        None
        if scope["input_contract_hash"] is None
        else _checksum(
            scope["input_contract_hash"],
            description="input contract hash",
        )
    )
    try:
        position = PredictionPosition(_string(scope["position"], description="position"))
        target = PredictionTarget(
            _string(scope["prediction_target"], description="prediction target")
        )
    except ValueError as exc:
        raise NeuralBundleError("bundle scope contains an unsupported enum") from exc
    conditions = tuple(
        _string(item, description="condition id")
        for item in _list(scope["condition_ids"], description="condition ids")
    )
    if not conditions or tuple(sorted(set(conditions))) != conditions:
        raise NeuralBundleError("condition ids must be non-empty, sorted, and unique")
    quantiles = _parse_quantiles(manifest["quantiles"])

    runtime = _runtime_versions(manifest["runtime"], legacy=legacy)
    numpy_version = runtime["numpy_version"]
    torch_version = runtime["torch_version"]
    safetensors_version = runtime["safetensors_version"]
    np, torch = _load_neural_dependencies()
    load_tensors, _, current_safetensors_version = _load_safetensors()
    version_pairs = (
        (numpy_version, str(np.__version__), "NumPy"),
        (torch_version, str(torch.__version__), "PyTorch"),
        (safetensors_version, current_safetensors_version, "safetensors"),
    )
    for trained, current, description in version_pairs:
        if _version_major(trained, description=f"training {description}") != _version_major(
            current, description=f"runtime {description}"
        ):
            raise NeuralBundleError(f"bundle {description} major version is incompatible")

    try:
        tensors = load_tensors(payloads["weights"])
    except Exception as exc:
        raise NeuralBundleError("weights.safetensors is invalid") from exc
    model = _build_network(torch, architecture)
    expected_state = model.state_dict()
    if set(tensors) != set(expected_state):
        raise NeuralBundleError("weight tensor names do not match architecture")
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != tuple(expected_state[name].shape):
            raise NeuralBundleError(f"weight tensor shape does not match architecture: {name}")
        if tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
            raise NeuralBundleError(f"weight tensor must be finite float32: {name}")
    try:
        model.load_state_dict(tensors, strict=True)
    except RuntimeError as exc:
        raise NeuralBundleError("weight tensors cannot be loaded into architecture") from exc
    model.eval()

    calibrator_bytes = tree.files[calibrator_path].payload
    if _sha256(calibrator_bytes) != calibrator_sha:
        raise NeuralBundleError("calibrator file checksum mismatch")
    calibrator_document = _mapping(
        _parse_canonical_json(calibrator_bytes, description="calibrator document"),
        description="calibrator document",
    )
    try:
        from token_prediction.evaluation.calibration import FittedExpansionCalibrator

        parsed_calibrator = FittedExpansionCalibrator.from_dict(calibrator_document)
    except (KeyError, TypeError, ValueError) as exc:
        raise NeuralBundleError("calibrator document is invalid") from exc
    expected_alpha = 2 * quantiles[0]
    if not math.isclose(
        parsed_calibrator.interval_alpha,
        expected_alpha,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise NeuralBundleError("calibrator alpha does not match bundle quantiles")
    provenance = _validated_point_provenance(
        manifest["provenance"],
        dataset_id=dataset_id,
        input_contract_hash=input_contract_hash,
        position=position,
        target=target,
        condition_ids=conditions,
        calibrator_id=parsed_calibrator.calibrator_id,
        interval_alpha=parsed_calibrator.interval_alpha,
        bundle_schema_version=bundle_schema_version,
    )
    if expected_source_provenance is not None:
        expected = _validated_expected_source_provenance(expected_source_provenance)
        actual_descriptor = _mapping(
            provenance["source_descriptor"],
            description="provenance source descriptor",
        )
        for name, actual in (
            ("source_descriptor", dict(actual_descriptor)),
            ("source_descriptor_hash", provenance["source_descriptor_hash"]),
            ("code_hash", provenance["code_hash"]),
        ):
            if expected[name] != actual:
                raise NeuralBundleError(
                    f"bundle {name.replace('_', ' ')} differs from expected provenance"
                )
        expected_runtime = expected["runtime_versions"]
        for name, actual_version in runtime.items():
            if (
                name not in expected_runtime
                or expected_runtime[name] != actual_version
            ):
                raise NeuralBundleError(
                    "bundle runtime versions differ from expected provenance"
                )
    fit_report = _parse_fit_report(
        manifest["fit_report"],
        encoder_hash=encoder_hash,
        torch_version=torch_version,
        numpy_version=numpy_version,
    )
    try:
        _validate_fit_report_semantics(fit_report, architecture, quantiles)
    except ValueError as exc:
        raise NeuralBundleError("fit report does not match the model architecture") from exc
    loaded = FittedIndependentMLP(
        estimator_id=_ESTIMATOR_ID,
        target=target,
        position=position,
        dataset_id=dataset_id,
        input_contract_hash=input_contract_hash,
        allowed_condition_ids=conditions,
        encoder=encoder,
        architecture=architecture,
        model=model,
        quantiles=quantiles,
        fit_report=fit_report,
        calibrator_document=calibrator_document,
        provenance=provenance,
    )
    _verify_bundle_tree(source, tree)
    return loaded


__all__ = [
    "NEURAL_BUNDLE_SCHEMA_VERSION",
    "NeuralBundleError",
    "load_neural_bundle",
    "neural_bundle_files",
    "save_neural_bundle",
]
