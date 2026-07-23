"""Safe canonical-JSON/safetensors persistence for GRU residual updaters."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Mapping

from token_prediction import __version__ as TOKEN_PREDICTION_VERSION
from token_prediction.dataset import PredictionPosition, PredictionTarget

from .gru import (
    GRU_RESIDUAL_ESTIMATOR_VERSION,
    FittedGRUResidual,
    GRUArchitecture,
    GRUFitReport,
    _build_network,
    _load_neural_dependencies,
    _valid_quantiles,
    _validate_fit_report,
)
from .neural_encoder import NEURAL_ENCODER_SCHEMA_VERSION, NeuralFeatureEncoder


GRU_BUNDLE_SCHEMA_VERSION = 1
GRU_COMPONENT_SCHEMA_VERSION = 1
GRU_BUNDLE_FORMAT = "gru_residual_safetensors_v1"

_MANIFEST = "manifest.json"
_MANIFEST_HASH = "manifest.sha256"
_CALIBRATOR = "calibrator.json"
_COMPONENT_DESCRIPTOR = "component.json"
_ENCODER = "encoder.json"
_ARCHITECTURE = "architecture.json"
_WEIGHTS = "weights.safetensors"
_MAX_ENTRIES = 64
_MAX_DEPTH = 8
_MAX_MANIFEST_BYTES = 1_048_576
_MAX_JSON_BYTES = 16_777_216
_MAX_WEIGHTS_BYTES = 536_870_912
_MAX_TOTAL_BYTES = 768 * 1024 * 1024
_SHA256_CHARS = frozenset("0123456789abcdef")
_PROVENANCE_KEYS = {
    "role",
    "candidate_id",
    "candidate_hash",
    "candidate_graph",
    "dataset_id",
    "split_plan_id",
    "eligibility_hash",
    "lifecycle_context_hash",
    "lifecycle_scored_hash",
    "outer_fold",
    "outer_task_partitions_sha256",
    "initializer_hash",
    "inner_split_id",
    "seed_set_hash",
    "interval_alpha",
    "calibrator_id",
}
_GRAPH_KEYS = {
    "initializer_estimator_id",
    "updater_estimator_id",
    "lifecycle_schema_id",
    "seed_policy_id",
    "inner_split_policy_id",
}


class GRUBundleError(ValueError):
    """The GRU bundle is unsafe, corrupted, or incompatible."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: object) -> bytes:
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
        raise GRUBundleError("GRU bundle metadata is not canonical JSON data") from exc


def _reject_constant(value: str) -> None:
    raise GRUBundleError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GRUBundleError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _parse_json(payload: bytes, *, description: str) -> Any:
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GRUBundleError(f"{description} is not valid UTF-8 JSON") from exc
    if _json_bytes(document) != payload:
        raise GRUBundleError(f"{description} is not canonical JSON")
    return document


def _mapping(value: Any, *, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise GRUBundleError(f"{description} must be a JSON object")
    return value


def _list(value: Any, *, description: str) -> list[Any]:
    if not isinstance(value, list):
        raise GRUBundleError(f"{description} must be a JSON list")
    return value


def _keys(value: Mapping[str, Any], expected: set[str], *, description: str) -> None:
    if set(value) != expected:
        raise GRUBundleError(
            f"{description} keys do not match schema; "
            f"missing={sorted(expected - set(value))}, extra={sorted(set(value) - expected)}"
        )


def _string(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise GRUBundleError(f"{description} must be a non-empty string")
    return value


def _integer(value: Any, *, description: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GRUBundleError(f"{description} must be an integer >= {minimum}")
    return value


def _floating(value: Any, *, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GRUBundleError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise GRUBundleError(f"{description} must be finite")
    return result


def _checksum(value: Any, *, description: str) -> str:
    result = _string(value, description=description)
    if len(result) != 64 or any(character not in _SHA256_CHARS for character in result):
        raise GRUBundleError(f"{description} must be a lowercase SHA-256")
    return result


def _safe_path(value: Any, *, description: str) -> str:
    result = _string(value, description=description)
    posix = PurePosixPath(result)
    windows = PureWindowsPath(result)
    if (
        "\\" in result
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or not posix.parts
        or len(posix.parts) > _MAX_DEPTH
        or any(part in {"", ".", ".."} for part in posix.parts)
        or posix.as_posix() != result
    ):
        raise GRUBundleError(f"{description} must be a safe relative POSIX path")
    return result


def _version_prefix(value: str, count: int, *, description: str) -> tuple[int, ...]:
    parts = value.split(".")
    if len(parts) < count:
        raise GRUBundleError(f"{description} is not a compatible version")
    try:
        result = tuple(int(part) for part in parts[:count])
    except ValueError as exc:
        raise GRUBundleError(f"{description} is not a compatible version") from exc
    if any(item < 0 for item in result):
        raise GRUBundleError(f"{description} is not a compatible version")
    return result


def _identity_calibrator(alpha: float) -> dict[str, Any]:
    return {
        "calibrator_schema_version": 1,
        "calibrator_id": "none",
        "interval_alpha": alpha,
        "expansion": 0.0,
    }


def _validate_provenance(
    value: Any,
    *,
    fitted: FittedGRUResidual,
    calibrator_id: str,
    interval_alpha: float,
) -> Mapping[str, Any]:
    document = _mapping(value, description="GRU provenance")
    _keys(document, _PROVENANCE_KEYS, description="GRU provenance")
    if document["role"] != "lifecycle_updater":
        raise GRUBundleError("GRU provenance role is invalid")
    if document["dataset_id"] != fitted.dataset_id:
        raise GRUBundleError("GRU provenance dataset differs from fitted scope")
    for name in ("candidate_hash", "split_plan_id", "eligibility_hash"):
        _checksum(document[name], description=f"provenance {name}")
    for name in (
        "lifecycle_context_hash",
        "lifecycle_scored_hash",
        "initializer_hash",
        "inner_split_id",
        "seed_set_hash",
    ):
        _checksum(document[name], description=f"provenance {name}")
    _string(document["candidate_id"], description="provenance candidate id")
    _integer(document["outer_fold"], description="provenance outer fold")
    graph = _mapping(document["candidate_graph"], description="provenance graph")
    _keys(graph, _GRAPH_KEYS, description="provenance graph")
    if graph["updater_estimator_id"] != fitted.estimator_id:
        raise GRUBundleError("GRU provenance graph has the wrong updater")
    if graph["initializer_estimator_id"] == "none":
        raise GRUBundleError("GRU lifecycle updater requires an initializer")
    partitions = _mapping(
        document["outer_task_partitions_sha256"],
        description="provenance outer partitions",
    )
    _keys(partitions, {"train", "validation", "calibration", "test"}, description="outer partitions")
    for role, values in partitions.items():
        if not isinstance(values, (tuple, list)) or not values:
            raise GRUBundleError(f"outer {role} partition is invalid")
        normalized = [_checksum(item, description=f"outer {role} task") for item in values]
        if normalized != sorted(set(normalized)):
            raise GRUBundleError(f"outer {role} partition is not canonical")
    alpha = _floating(document["interval_alpha"], description="provenance alpha")
    if not math.isclose(alpha, interval_alpha, rel_tol=0.0, abs_tol=1e-12):
        raise GRUBundleError("GRU provenance alpha differs from calibrator")
    if document["calibrator_id"] != calibrator_id:
        raise GRUBundleError("GRU provenance calibrator differs from bundle")
    return document


def _fit_report_document(report: GRUFitReport) -> dict[str, Any]:
    return {
        "train_sequence_hash": report.train_sequence_hash,
        "validation_sequence_hash": report.validation_sequence_hash,
        "train_sequence_count": report.train_sequence_count,
        "validation_sequence_count": report.validation_sequence_count,
        "train_scored_point_count": report.train_scored_point_count,
        "validation_scored_point_count": report.validation_scored_point_count,
        "seed": report.seed,
        "best_epoch": report.best_epoch,
        "best_validation_loss": report.best_validation_loss,
        "validation_history": list(report.validation_history),
        "target_scale": report.target_scale,
        "parameters": dict(report.parameters),
        "platform": report.platform,
    }


def _state_for_safetensors(fitted: FittedGRUResidual, torch: Any) -> dict[str, Any]:
    expected = _build_network(torch, fitted.architecture).state_dict()
    actual = fitted.model.state_dict()
    if set(actual) != set(expected):
        raise GRUBundleError("GRU state keys do not match the declared architecture")
    tensors: dict[str, Any] = {}
    for name in sorted(expected):
        tensor = actual[name]
        if tuple(tensor.shape) != tuple(expected[name].shape):
            raise GRUBundleError(f"GRU tensor shape differs from architecture: {name}")
        if tensor.dtype != torch.float32:
            raise GRUBundleError(f"GRU tensor must be float32: {name}")
        tensor = tensor.detach().to(device="cpu").contiguous()
        if not bool(torch.isfinite(tensor).all()):
            raise GRUBundleError(f"GRU tensor is non-finite: {name}")
        tensors[name] = tensor
    return tensors


def _safetensors() -> tuple[Any, Any, str]:
    try:
        import safetensors
        from safetensors.torch import load as load_tensors
        from safetensors.torch import save as save_tensors
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency gate
        from .neural_encoder import OptionalNeuralDependencyError

        raise OptionalNeuralDependencyError(
            "GRU bundle persistence requires token-prediction[neural]"
        ) from exc
    return load_tensors, save_tensors, str(safetensors.__version__)


def gru_bundle_files(
    fitted: FittedGRUResidual,
    *,
    calibrator: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Mapping[str, bytes]:
    np, torch = _load_neural_dependencies()
    _load_tensors, save_tensors, safetensors_version = _safetensors()
    calibrator_document = (
        calibrator
        if calibrator is not None
        else fitted.calibrator_document
        if fitted.calibrator_document is not None
        else _identity_calibrator(2 * fitted.quantiles[0])
    )
    try:
        from token_prediction.evaluation.calibration import FittedExpansionCalibrator

        parsed_calibrator = FittedExpansionCalibrator.from_dict(calibrator_document)
    except (KeyError, TypeError, ValueError) as exc:
        raise GRUBundleError("GRU calibrator document is invalid") from exc
    if not math.isclose(
        parsed_calibrator.interval_alpha,
        2 * fitted.quantiles[0],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise GRUBundleError("GRU calibrator alpha differs from quantiles")
    provenance_document = _validate_provenance(
        provenance if provenance is not None else fitted.provenance or {},
        fitted=fitted,
        calibrator_id=parsed_calibrator.calibrator_id,
        interval_alpha=parsed_calibrator.interval_alpha,
    )
    encoder_bytes = _json_bytes(fitted.encoder.to_dict())
    architecture_bytes = _json_bytes(fitted.architecture.to_dict())
    weights_bytes = save_tensors(_state_for_safetensors(fitted, torch))
    component = {
        "schema_version": GRU_COMPONENT_SCHEMA_VERSION,
        "type": fitted.estimator_id,
        "estimator_version": GRU_RESIDUAL_ESTIMATOR_VERSION,
        "files": {
            "architecture": {
                "path": _ARCHITECTURE,
                "sha256": _sha256(architecture_bytes),
            },
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
    calibrator_bytes = _json_bytes(dict(calibrator_document))
    manifest = {
        "schema_version": GRU_BUNDLE_SCHEMA_VERSION,
        "bundle_type": "gru_residual_neural",
        "estimator": {
            "id": fitted.estimator_id,
            "version": GRU_RESIDUAL_ESTIMATOR_VERSION,
        },
        "scope": {
            "dataset_id": fitted.dataset_id,
            "input_contract_hash": fitted.input_contract_hash,
            "position": PredictionPosition.TASK_UPDATE.value,
            "prediction_target": fitted.target.value,
            "condition_id": fitted.condition_id,
        },
        "quantiles": list(fitted.quantiles),
        "target_scale": fitted.target_scale,
        "residual_scale": fitted.residual_scale,
        "no_recurrence": fitted.no_recurrence,
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
    if _version_prefix(str(np.__version__), 1, description="NumPy version") != _version_prefix(
        fitted.fit_report.numpy_version,
        1,
        description="training NumPy version",
    ):
        raise GRUBundleError("training and save-time NumPy major versions differ")
    if _version_prefix(str(torch.__version__), 1, description="PyTorch version") != _version_prefix(
        fitted.fit_report.torch_version,
        1,
        description="training PyTorch version",
    ):
        raise GRUBundleError("training and save-time PyTorch major versions differ")
    manifest_bytes = _json_bytes(manifest)
    files = {
        _MANIFEST: manifest_bytes,
        _MANIFEST_HASH: f"{_sha256(manifest_bytes)}\n".encode("ascii"),
        _CALIBRATOR: calibrator_bytes,
        f"{component_path}/{_COMPONENT_DESCRIPTOR}": component_bytes,
        f"{component_path}/{_ENCODER}": encoder_bytes,
        f"{component_path}/{_ARCHITECTURE}": architecture_bytes,
        f"{component_path}/{_WEIGHTS}": weights_bytes,
    }
    _validate_file_limits(files)
    return MappingProxyType(files)


def _validate_file_limits(files: Mapping[str, bytes]) -> None:
    if not files or len(files) > _MAX_ENTRIES:
        raise GRUBundleError("GRU bundle entry count is invalid")
    total = 0
    for raw_name, payload in files.items():
        name = _safe_path(raw_name, description="GRU bundle path")
        if not isinstance(payload, bytes):
            raise GRUBundleError(f"GRU bundle payload must be bytes: {name}")
        maximum = (
            _MAX_MANIFEST_BYTES
            if name == _MANIFEST
            else 65
            if name == _MANIFEST_HASH
            else _MAX_WEIGHTS_BYTES
            if name.endswith(f"/{_WEIGHTS}")
            else _MAX_JSON_BYTES
        )
        if len(payload) > maximum:
            raise GRUBundleError(f"GRU bundle file exceeds its size limit: {name}")
        total += len(payload)
    if total > _MAX_TOTAL_BYTES:
        raise GRUBundleError("GRU bundle exceeds its total size limit")


@dataclass(frozen=True)
class _NormalizedSource:
    files: Mapping[str, bytes]
    root: Path | None = None
    tree: Any | None = None

    def verify(self) -> None:
        if self.root is not None and self.tree is not None:
            from .neural_bundle import _verify_bundle_tree

            _verify_bundle_tree(self.root, self.tree)


def _normalize_source(
    source: str | os.PathLike[str] | Mapping[str, bytes],
) -> _NormalizedSource:
    if isinstance(source, Mapping):
        files = dict(source)
        _validate_file_limits(files)
        return _NormalizedSource(MappingProxyType(files))
    root = Path(source).expanduser().absolute()
    if not root.exists() and not root.is_symlink():
        raise GRUBundleError(f"GRU bundle directory does not exist: {root}")
    try:
        from .neural_bundle import NeuralBundleError, _bundle_entries

        tree = _bundle_entries(root)
    except (OSError, NeuralBundleError) as exc:
        raise GRUBundleError("GRU bundle directory is unsafe") from exc
    files = {name: item.payload for name, item in tree.files.items()}
    _validate_file_limits(files)
    return _NormalizedSource(MappingProxyType(files), root, tree)


def _parse_fit_report(
    value: Any,
    *,
    encoder_hash: str,
    torch_version: str,
    numpy_version: str,
) -> GRUFitReport:
    document = _mapping(value, description="GRU fit report")
    expected = {
        "train_sequence_hash",
        "validation_sequence_hash",
        "train_sequence_count",
        "validation_sequence_count",
        "train_scored_point_count",
        "validation_scored_point_count",
        "seed",
        "best_epoch",
        "best_validation_loss",
        "validation_history",
        "target_scale",
        "parameters",
        "platform",
    }
    _keys(document, expected, description="GRU fit report")
    try:
        return GRUFitReport(
            estimator_version=GRU_RESIDUAL_ESTIMATOR_VERSION,
            encoder_schema_hash=encoder_hash,
            train_sequence_hash=_checksum(
                document["train_sequence_hash"],
                description="train sequence hash",
            ),
            validation_sequence_hash=_checksum(
                document["validation_sequence_hash"],
                description="validation sequence hash",
            ),
            train_sequence_count=_integer(
                document["train_sequence_count"],
                description="train sequence count",
                minimum=1,
            ),
            validation_sequence_count=_integer(
                document["validation_sequence_count"],
                description="validation sequence count",
                minimum=1,
            ),
            train_scored_point_count=_integer(
                document["train_scored_point_count"],
                description="train scored point count",
                minimum=1,
            ),
            validation_scored_point_count=_integer(
                document["validation_scored_point_count"],
                description="validation scored point count",
                minimum=1,
            ),
            seed=_integer(document["seed"], description="fit seed"),
            best_epoch=_integer(document["best_epoch"], description="best epoch", minimum=1),
            best_validation_loss=_floating(
                document["best_validation_loss"],
                description="best validation loss",
            ),
            validation_history=tuple(
                _floating(item, description="validation history")
                for item in _list(
                    document["validation_history"],
                    description="validation history",
                )
            ),
            target_scale=_floating(document["target_scale"], description="target scale"),
            parameters=_mapping(document["parameters"], description="fit parameters"),
            torch_version=torch_version,
            numpy_version=numpy_version,
            platform=_string(document["platform"], description="training platform"),
        )
    except ValueError as exc:
        raise GRUBundleError("GRU fit report is inconsistent") from exc


def load_gru_bundle(
    source: str | os.PathLike[str] | Mapping[str, bytes],
    *,
    expected_provenance: Mapping[str, Any] | None = None,
    apply_calibrator: bool = True,
) -> FittedGRUResidual:
    normalized = _normalize_source(source)
    files = normalized.files
    if _MANIFEST not in files or _MANIFEST_HASH not in files:
        raise GRUBundleError("GRU bundle manifest is missing")
    manifest_bytes = files[_MANIFEST]
    if files[_MANIFEST_HASH] != f"{_sha256(manifest_bytes)}\n".encode("ascii"):
        raise GRUBundleError("GRU manifest checksum does not match")
    manifest = _mapping(
        _parse_json(manifest_bytes, description="GRU manifest"),
        description="GRU manifest",
    )
    _keys(
        manifest,
        {
            "schema_version",
            "bundle_type",
            "estimator",
            "scope",
            "quantiles",
            "target_scale",
            "residual_scale",
            "no_recurrence",
            "component",
            "calibrator",
            "provenance",
            "runtime",
            "fit_report",
        },
        description="GRU manifest",
    )
    if manifest["schema_version"] != GRU_BUNDLE_SCHEMA_VERSION:
        raise GRUBundleError("unsupported GRU bundle schema")
    if manifest["bundle_type"] != "gru_residual_neural":
        raise GRUBundleError("unsupported GRU bundle type")
    estimator = _mapping(manifest["estimator"], description="GRU estimator")
    _keys(estimator, {"id", "version"}, description="GRU estimator")
    if estimator != {
        "id": "gru_residual",
        "version": GRU_RESIDUAL_ESTIMATOR_VERSION,
    }:
        raise GRUBundleError("unsupported GRU estimator identity")

    component = _mapping(manifest["component"], description="GRU component")
    _keys(component, {"path", "descriptor", "content_hash"}, description="GRU component")
    component_hash = _checksum(component["content_hash"], description="component hash")
    component_path = _safe_path(component["path"], description="component path")
    descriptor_path = _safe_path(component["descriptor"], description="descriptor path")
    if component_path != f"components/{component_hash}" or descriptor_path != (
        f"{component_path}/{_COMPONENT_DESCRIPTOR}"
    ):
        raise GRUBundleError("GRU component path does not match its hash")
    calibrator_record = _mapping(manifest["calibrator"], description="GRU calibrator")
    _keys(calibrator_record, {"path", "sha256"}, description="GRU calibrator")
    calibrator_path = _safe_path(calibrator_record["path"], description="calibrator path")
    if calibrator_path != _CALIBRATOR:
        raise GRUBundleError("unsupported GRU calibrator path")
    expected_files = {
        _MANIFEST,
        _MANIFEST_HASH,
        _CALIBRATOR,
        descriptor_path,
        f"{component_path}/{_ENCODER}",
        f"{component_path}/{_ARCHITECTURE}",
        f"{component_path}/{_WEIGHTS}",
    }
    if set(files) != expected_files:
        raise GRUBundleError("GRU bundle contains missing or extra files")
    normalized.verify()

    descriptor_bytes = files[descriptor_path]
    if _sha256(descriptor_bytes) != component_hash:
        raise GRUBundleError("GRU component descriptor hash does not match")
    descriptor = _mapping(
        _parse_json(descriptor_bytes, description="GRU component descriptor"),
        description="GRU component descriptor",
    )
    _keys(
        descriptor,
        {"schema_version", "type", "estimator_version", "files"},
        description="GRU component descriptor",
    )
    if (
        descriptor["schema_version"] != GRU_COMPONENT_SCHEMA_VERSION
        or descriptor["type"] != "gru_residual"
        or descriptor["estimator_version"] != GRU_RESIDUAL_ESTIMATOR_VERSION
    ):
        raise GRUBundleError("GRU component identity is incompatible")
    records = _mapping(descriptor["files"], description="GRU component files")
    _keys(records, {"architecture", "encoder", "weights"}, description="component files")
    payloads: dict[str, bytes] = {}
    for role, filename in (
        ("architecture", _ARCHITECTURE),
        ("encoder", _ENCODER),
        ("weights", _WEIGHTS),
    ):
        record = _mapping(records[role], description=f"{role} record")
        expected_keys = (
            {"path", "sha256", "content_hash", "schema_version"}
            if role == "encoder"
            else {"path", "sha256"}
        )
        _keys(record, expected_keys, description=f"{role} record")
        if _safe_path(record["path"], description=f"{role} path") != filename:
            raise GRUBundleError(f"unsupported GRU {role} filename")
        payload = files[f"{component_path}/{filename}"]
        if _sha256(payload) != _checksum(record["sha256"], description=f"{role} checksum"):
            raise GRUBundleError(f"GRU {role} checksum does not match")
        payloads[role] = payload
    encoder_record = _mapping(records["encoder"], description="encoder record")
    if encoder_record["schema_version"] != NEURAL_ENCODER_SCHEMA_VERSION:
        raise GRUBundleError("unsupported GRU encoder schema")
    try:
        encoder = NeuralFeatureEncoder.from_dict(
            _mapping(
                _parse_json(payloads["encoder"], description="GRU encoder"),
                description="GRU encoder",
            )
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GRUBundleError("GRU encoder is invalid") from exc
    if encoder.schema.content_hash != _checksum(
        encoder_record["content_hash"],
        description="encoder content hash",
    ):
        raise GRUBundleError("GRU encoder content hash does not match")
    try:
        architecture = GRUArchitecture.from_dict(
            _mapping(
                _parse_json(payloads["architecture"], description="GRU architecture"),
                description="GRU architecture",
            )
        )
    except (TypeError, ValueError) as exc:
        raise GRUBundleError("GRU architecture is invalid") from exc
    if architecture.point_input_dim != encoder.schema.output_width:
        raise GRUBundleError("GRU architecture and encoder widths differ")

    scope = _mapping(manifest["scope"], description="GRU scope")
    _keys(
        scope,
        {
            "dataset_id",
            "input_contract_hash",
            "position",
            "prediction_target",
            "condition_id",
        },
        description="GRU scope",
    )
    if scope["position"] != PredictionPosition.TASK_UPDATE.value:
        raise GRUBundleError("GRU bundle position is invalid")
    try:
        target = PredictionTarget(
            _string(scope["prediction_target"], description="prediction target")
        )
    except ValueError as exc:
        raise GRUBundleError("GRU bundle target is invalid") from exc
    if target != PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS:
        raise GRUBundleError("unsupported GRU prediction target")
    dataset_id = _string(scope["dataset_id"], description="dataset id")
    input_contract_hash = _checksum(
        scope["input_contract_hash"],
        description="input contract hash",
    )
    condition_id = _string(scope["condition_id"], description="condition id")
    quantiles = tuple(
        _floating(item, description="GRU quantile")
        for item in _list(manifest["quantiles"], description="GRU quantiles")
    )
    if not _valid_quantiles(quantiles):
        raise GRUBundleError("GRU quantiles are invalid")
    target_scale = _floating(manifest["target_scale"], description="target scale")
    residual_scale = _floating(manifest["residual_scale"], description="residual scale")
    no_recurrence = manifest["no_recurrence"]
    if target_scale < 1 or residual_scale < 0 or not isinstance(no_recurrence, bool):
        raise GRUBundleError("GRU scale/state policy is invalid")

    calibrator_bytes = files[_CALIBRATOR]
    if _sha256(calibrator_bytes) != _checksum(
        calibrator_record["sha256"],
        description="calibrator checksum",
    ):
        raise GRUBundleError("GRU calibrator checksum does not match")
    calibrator_document = _mapping(
        _parse_json(calibrator_bytes, description="GRU calibrator"),
        description="GRU calibrator",
    )
    try:
        from token_prediction.evaluation.calibration import FittedExpansionCalibrator

        parsed_calibrator = FittedExpansionCalibrator.from_dict(calibrator_document)
    except (KeyError, TypeError, ValueError) as exc:
        raise GRUBundleError("GRU calibrator is invalid") from exc
    if not math.isclose(
        parsed_calibrator.interval_alpha,
        2 * quantiles[0],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise GRUBundleError("GRU calibrator alpha differs from quantiles")

    runtime = _mapping(manifest["runtime"], description="GRU runtime")
    _keys(
        runtime,
        {
            "python_version",
            "token_prediction_version",
            "numpy_version",
            "torch_version",
            "safetensors_version",
        },
        description="GRU runtime",
    )
    normalized_runtime = {
        name: _string(value, description=f"runtime {name}")
        for name, value in runtime.items()
    }
    if _version_prefix(
        normalized_runtime["python_version"],
        2,
        description="bundle Python version",
    ) != _version_prefix(platform.python_version(), 2, description="runtime Python version"):
        raise GRUBundleError("GRU bundle Python major/minor is incompatible")
    if _version_prefix(
        normalized_runtime["token_prediction_version"],
        2,
        description="bundle package version",
    ) != _version_prefix(
        TOKEN_PREDICTION_VERSION,
        2,
        description="runtime package version",
    ):
        raise GRUBundleError("GRU bundle package version is incompatible")
    np, torch = _load_neural_dependencies()
    load_tensors, _save_tensors, current_safetensors = _safetensors()
    for trained, current, description in (
        (normalized_runtime["numpy_version"], str(np.__version__), "NumPy"),
        (normalized_runtime["torch_version"], str(torch.__version__), "PyTorch"),
        (normalized_runtime["safetensors_version"], current_safetensors, "safetensors"),
    ):
        if _version_prefix(trained, 1, description=f"bundle {description}") != _version_prefix(
            current,
            1,
            description=f"runtime {description}",
        ):
            raise GRUBundleError(f"GRU bundle {description} major version is incompatible")
    try:
        tensors = load_tensors(payloads["weights"])
    except Exception as exc:
        raise GRUBundleError("GRU weights.safetensors is invalid") from exc
    model = _build_network(torch, architecture)
    expected_state = model.state_dict()
    if set(tensors) != set(expected_state):
        raise GRUBundleError("GRU weight tensor names differ from architecture")
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != tuple(expected_state[name].shape):
            raise GRUBundleError(f"GRU tensor shape differs from architecture: {name}")
        if tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
            raise GRUBundleError(f"GRU tensor must be finite float32: {name}")
    try:
        model.load_state_dict(tensors, strict=True)
    except RuntimeError as exc:
        raise GRUBundleError("GRU tensors cannot be loaded") from exc
    model.eval()
    fit_report = _parse_fit_report(
        manifest["fit_report"],
        encoder_hash=encoder.schema.content_hash,
        torch_version=normalized_runtime["torch_version"],
        numpy_version=normalized_runtime["numpy_version"],
    )
    try:
        _validate_fit_report(fit_report, architecture, quantiles)
    except ValueError as exc:
        raise GRUBundleError("GRU fit report differs from architecture") from exc
    if not math.isclose(fit_report.target_scale, target_scale, rel_tol=0.0, abs_tol=0.0):
        raise GRUBundleError("GRU fit report target scale differs from manifest")
    if (
        not math.isclose(
            float(fit_report.parameters["residual_scale"]),
            residual_scale,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or bool(fit_report.parameters["no_recurrence"]) != no_recurrence
    ):
        raise GRUBundleError("GRU fit report state policy differs from manifest")
    provisional = FittedGRUResidual(
        estimator_id="gru_residual",
        target=target,
        dataset_id=dataset_id,
        input_contract_hash=input_contract_hash,
        condition_id=condition_id,
        encoder=encoder,
        architecture=architecture,
        model=model,
        quantiles=quantiles,
        target_scale=target_scale,
        residual_scale=residual_scale,
        no_recurrence=no_recurrence,
        fit_report=fit_report,
    )
    provenance = _validate_provenance(
        manifest["provenance"],
        fitted=provisional,
        calibrator_id=parsed_calibrator.calibrator_id,
        interval_alpha=parsed_calibrator.interval_alpha,
    )
    if expected_provenance is not None and _json_bytes(dict(provenance)) != _json_bytes(
        dict(expected_provenance)
    ):
        raise GRUBundleError("GRU provenance differs from the lifecycle manifest")
    loaded = FittedGRUResidual(
        **{
            **provisional.__dict__,
            "calibrator_document": (
                dict(calibrator_document) if apply_calibrator else None
            ),
            "provenance": dict(provenance),
        }
    )
    normalized.verify()
    return loaded


def save_gru_bundle(
    fitted: FittedGRUResidual,
    directory: str | os.PathLike[str],
    *,
    calibrator: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(directory).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"GRU bundle destination already exists: {destination}")
    files = gru_bundle_files(fitted, calibrator=calibrator, provenance=provenance)
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


__all__ = [
    "GRUBundleError",
    "GRU_BUNDLE_FORMAT",
    "GRU_BUNDLE_SCHEMA_VERSION",
    "gru_bundle_files",
    "load_gru_bundle",
    "save_gru_bundle",
]
