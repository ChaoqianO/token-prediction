"""Safe, atomic checkpoints for long development experiments.

Checkpoints live only in ignored mutable workspace storage.  They never use
pickle: result objects are canonical JSON, while neural tensor state is
supplied by estimators as safetensors plus canonical JSON metadata.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import uuid
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

from token_prediction.dataset import PredictionPosition, PredictionTarget
from token_prediction.estimators import FitCheckpoint, TokenForecast
from token_prediction.experiment import (
    CandidateExecutionKey,
    CandidateResult,
    CandidateResultStore,
    FoldArtifact,
    PredictionRecord,
)
from token_prediction.lineage import publish_artifact, verify_artifact


CANDIDATE_CHECKPOINT_SCHEMA_VERSION = 1
FIT_CHECKPOINT_SCHEMA_VERSION = 1
_CANDIDATE_STAGE = "development_candidate_checkpoint"
_FIT_STAGE = "neural_fit_epoch_checkpoint"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_MAX_RESULT_BYTES = 2 * 1024**3
_MAX_FIT_FILE_BYTES = 2 * 1024**3


class CheckpointError(RuntimeError):
    """A checkpoint is corrupt, unsafe, or belongs to another execution."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CheckpointError("checkpoint value is not finite canonical JSON") from exc


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int)):
                raise CheckpointError("checkpoint mapping keys must be strings or integers")
            result[str(key)] = _json_value(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise CheckpointError("checkpoint values must be finite")
        return value
    raise CheckpointError(f"unsupported checkpoint value: {type(value).__name__}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError(f"duplicate checkpoint JSON key: {key!r}")
        result[key] = value
    return result


def _strict_json(payload: bytes, *, description: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                CheckpointError(f"non-finite checkpoint constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"{description} is not valid UTF-8 JSON") from exc


def _mapping(value: Any, *, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointError(f"{description} must be an object")
    return value


def _sequence(value: Any, *, description: str) -> list[Any]:
    if not isinstance(value, list):
        raise CheckpointError(f"{description} must be an array")
    return value


def _finite(value: Any, *, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CheckpointError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CheckpointError(f"{description} must be finite")
    return result


def _integer(value: Any, *, description: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CheckpointError(f"{description} must be an integer >= {minimum}")
    return value


def _text(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CheckpointError(f"{description} must be non-empty text")
    return value


def _safe_relative(value: str, *, description: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or "\\" in value:
        raise CheckpointError(f"{description} is not a safe relative path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or posix.as_posix() != value
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise CheckpointError(f"{description} is not a safe relative path")
    return value


def _is_reparse(metadata: os.stat_result) -> bool:
    flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return bool(int(getattr(metadata, "st_file_attributes", 0)) & flag)


def _require_plain_directory(path: Path, *, description: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CheckpointError(f"cannot inspect {description}") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise CheckpointError(f"{description} must be a plain directory")


def _write_new(path: Path, payload: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | int(getattr(os, "O_BINARY", 0))
        | int(getattr(os, "O_NOFOLLOW", 0))
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise CheckpointError("short checkpoint write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_limited(path: Path, *, limit: int, description: str) -> bytes:
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size < 0
        or metadata.st_size > limit
    ):
        raise CheckpointError(f"unsafe or oversized {description}")
    payload = path.read_bytes()
    if len(payload) != metadata.st_size:
        raise CheckpointError(f"{description} changed while being read")
    return payload


def _key_from_dict(value: Mapping[str, Any]) -> CandidateExecutionKey:
    expected = {
        "experiment_id",
        "candidate_id",
        "candidate_hash",
        "dataset_id",
        "split_plan_id",
        "split_seed",
        "eligibility_hash",
        "position",
        "target",
        "condition_id",
        "calibrator_id",
        "alpha",
        "source_provenance_hash",
    }
    if set(value) != expected:
        raise CheckpointError("candidate execution key schema does not match")
    return CandidateExecutionKey(
        experiment_id=_text(value["experiment_id"], description="experiment_id"),
        candidate_id=_text(value["candidate_id"], description="candidate_id"),
        candidate_hash=_text(value["candidate_hash"], description="candidate_hash"),
        dataset_id=_text(value["dataset_id"], description="dataset_id"),
        split_plan_id=_text(value["split_plan_id"], description="split_plan_id"),
        split_seed=_integer(value["split_seed"], description="split_seed"),
        eligibility_hash=_text(value["eligibility_hash"], description="eligibility_hash"),
        position=PredictionPosition(_text(value["position"], description="position")),
        target=PredictionTarget(_text(value["target"], description="target")),
        condition_id=_text(value["condition_id"], description="condition_id"),
        calibrator_id=_text(value["calibrator_id"], description="calibrator_id"),
        alpha=_finite(value["alpha"], description="alpha"),
        source_provenance_hash=_text(
            value["source_provenance_hash"], description="source_provenance_hash"
        ),
    )


def _forecast_document(forecast: TokenForecast) -> dict[str, Any]:
    return {
        "point_id": forecast.point_id,
        "target": forecast.target.value,
        "lower": forecast.lower,
        "point": forecast.point,
        "upper": forecast.upper,
        "latency_ms": forecast.latency_ms,
        "overhead_input_tokens": forecast.overhead_input_tokens,
        "overhead_output_tokens": forecast.overhead_output_tokens,
        "raw_lower": forecast.raw_lower,
        "raw_point": forecast.raw_point,
        "raw_upper": forecast.raw_upper,
    }


def _forecast_from_dict(value: Mapping[str, Any]) -> TokenForecast:
    expected = {
        "point_id",
        "target",
        "lower",
        "point",
        "upper",
        "latency_ms",
        "overhead_input_tokens",
        "overhead_output_tokens",
        "raw_lower",
        "raw_point",
        "raw_upper",
    }
    if set(value) != expected:
        raise CheckpointError("forecast checkpoint schema does not match")
    raw = []
    for name in ("raw_lower", "raw_point", "raw_upper"):
        item = value[name]
        raw.append(None if item is None else _finite(item, description=name))
    return TokenForecast(
        point_id=_text(value["point_id"], description="forecast point_id"),
        target=PredictionTarget(_text(value["target"], description="forecast target")),
        lower=_finite(value["lower"], description="forecast lower"),
        point=_finite(value["point"], description="forecast point"),
        upper=_finite(value["upper"], description="forecast upper"),
        latency_ms=_finite(value["latency_ms"], description="forecast latency"),
        overhead_input_tokens=_integer(
            value["overhead_input_tokens"], description="forecast input overhead"
        ),
        overhead_output_tokens=_integer(
            value["overhead_output_tokens"], description="forecast output overhead"
        ),
        raw_lower=raw[0],
        raw_point=raw[1],
        raw_upper=raw[2],
    )


def _artifact_document(artifact: FoldArtifact) -> dict[str, Any]:
    return {
        "fold": artifact.fold,
        "encoder": None if artifact.encoder is None else _json_value(artifact.encoder),
        "fit_report": (None if artifact.fit_report is None else _json_value(artifact.fit_report)),
        "feature_importance": (
            None
            if artifact.feature_importance is None
            else [_json_value(item) for item in artifact.feature_importance]
        ),
        "model_strings": (
            None if artifact.model_strings is None else _json_value(artifact.model_strings)
        ),
        "bundle_files": (
            None
            if artifact.bundle_files is None
            else {
                name: base64.b64encode(payload).decode("ascii")
                for name, payload in sorted(artifact.bundle_files.items())
            }
        ),
        "calibrator": (None if artifact.calibrator is None else _json_value(artifact.calibrator)),
        "provenance": (None if artifact.provenance is None else _json_value(artifact.provenance)),
    }


def _optional_mapping(value: Any, *, description: str) -> Mapping[str, Any] | None:
    return None if value is None else _mapping(value, description=description)


def _artifact_from_dict(value: Mapping[str, Any]) -> FoldArtifact:
    expected = {
        "fold",
        "encoder",
        "fit_report",
        "feature_importance",
        "model_strings",
        "bundle_files",
        "calibrator",
        "provenance",
    }
    if set(value) != expected:
        raise CheckpointError("fold artifact checkpoint schema does not match")
    importance_value = value["feature_importance"]
    importance = None
    if importance_value is not None:
        importance = tuple(
            dict(_mapping(item, description="feature importance row"))
            for item in _sequence(importance_value, description="feature importance")
        )
    bundle_value = value["bundle_files"]
    bundle = None
    if bundle_value is not None:
        bundle = {}
        for raw_name, raw_payload in _mapping(bundle_value, description="bundle files").items():
            name = _safe_relative(raw_name, description="bundle file name")
            if not isinstance(raw_payload, str):
                raise CheckpointError("bundle checkpoint payload must be base64 text")
            try:
                bundle[name] = base64.b64decode(raw_payload, validate=True)
            except ValueError as exc:
                raise CheckpointError("bundle checkpoint payload is invalid base64") from exc
    model_strings = _optional_mapping(value["model_strings"], description="model strings")
    if model_strings is not None and any(
        not isinstance(name, str) or not isinstance(item, str)
        for name, item in model_strings.items()
    ):
        raise CheckpointError("model string checkpoint entries must be text")
    return FoldArtifact(
        fold=_integer(value["fold"], description="artifact fold"),
        encoder=_optional_mapping(value["encoder"], description="encoder"),
        fit_report=_optional_mapping(value["fit_report"], description="fit report"),
        feature_importance=importance,
        model_strings=model_strings,
        bundle_files=bundle,
        calibrator=_optional_mapping(value["calibrator"], description="calibrator"),
        provenance=_optional_mapping(value["provenance"], description="provenance"),
    )


def _result_document(result: CandidateResult) -> dict[str, Any]:
    return {
        "candidate_id": result.candidate_id,
        "candidate_hash": result.candidate_hash,
        "dataset_id": result.dataset_id,
        "split_plan_id": result.split_plan_id,
        "eligibility_hash": result.eligibility_hash,
        "position": result.position.value,
        "target": result.target.value,
        "condition_id": result.condition_id,
        "calibrator_id": result.calibrator_id,
        "alpha": result.alpha,
        "metric_suite_id": result.metric_suite_id,
        "predictions": [
            {
                "candidate_id": record.candidate_id,
                "point_id": record.point_id,
                "task_id": record.task_id,
                "trajectory_id": record.trajectory_id,
                "condition_id": record.condition_id,
                "fold": record.fold,
                "target": record.target.value,
                "forecast": _forecast_document(record.forecast),
                "sample_weight": record.sample_weight,
            }
            for record in result.predictions
        ],
        "metrics": dict(result.metrics),
        "fold_metrics": {str(fold): dict(metrics) for fold, metrics in result.fold_metrics.items()},
        "task_metrics": {task: dict(metrics) for task, metrics in result.task_metrics.items()},
        "fold_artifacts": [_artifact_document(item) for item in result.fold_artifacts],
    }


def _result_from_dict(value: Mapping[str, Any]) -> CandidateResult:
    expected = {
        "candidate_id",
        "candidate_hash",
        "dataset_id",
        "split_plan_id",
        "eligibility_hash",
        "position",
        "target",
        "condition_id",
        "calibrator_id",
        "alpha",
        "metric_suite_id",
        "predictions",
        "metrics",
        "fold_metrics",
        "task_metrics",
        "fold_artifacts",
    }
    if set(value) != expected:
        raise CheckpointError("candidate result checkpoint schema does not match")
    predictions: list[PredictionRecord] = []
    for raw in _sequence(value["predictions"], description="candidate predictions"):
        item = _mapping(raw, description="candidate prediction")
        if set(item) != {
            "candidate_id",
            "point_id",
            "task_id",
            "trajectory_id",
            "condition_id",
            "fold",
            "target",
            "forecast",
            "sample_weight",
        }:
            raise CheckpointError("candidate prediction checkpoint schema does not match")
        predictions.append(
            PredictionRecord(
                candidate_id=_text(item["candidate_id"], description="candidate_id"),
                point_id=_text(item["point_id"], description="point_id"),
                task_id=_text(item["task_id"], description="task_id"),
                trajectory_id=_text(item["trajectory_id"], description="trajectory_id"),
                condition_id=_text(item["condition_id"], description="condition_id"),
                fold=_integer(item["fold"], description="prediction fold"),
                target=PredictionTarget(_text(item["target"], description="target")),
                forecast=_forecast_from_dict(_mapping(item["forecast"], description="forecast")),
                sample_weight=_finite(item["sample_weight"], description="sample weight"),
            )
        )
    fold_metrics: dict[int, dict[str, Any]] = {}
    for fold, metrics in _mapping(value["fold_metrics"], description="fold metrics").items():
        if not isinstance(fold, str) or not fold.isdecimal():
            raise CheckpointError("fold metric keys must be decimal integers")
        fold_metrics[_integer(int(fold), description="fold metric key")] = dict(
            _mapping(metrics, description="fold metrics")
        )
    return CandidateResult(
        candidate_id=_text(value["candidate_id"], description="candidate_id"),
        candidate_hash=_text(value["candidate_hash"], description="candidate_hash"),
        dataset_id=_text(value["dataset_id"], description="dataset_id"),
        split_plan_id=_text(value["split_plan_id"], description="split_plan_id"),
        eligibility_hash=_text(value["eligibility_hash"], description="eligibility_hash"),
        position=PredictionPosition(_text(value["position"], description="position")),
        target=PredictionTarget(_text(value["target"], description="target")),
        condition_id=_text(value["condition_id"], description="condition_id"),
        calibrator_id=_text(value["calibrator_id"], description="calibrator_id"),
        alpha=_finite(value["alpha"], description="alpha"),
        metric_suite_id=_text(value["metric_suite_id"], description="metric_suite_id"),
        predictions=tuple(predictions),
        metrics=dict(_mapping(value["metrics"], description="metrics")),
        fold_metrics=fold_metrics,
        task_metrics={
            _text(task, description="task metric key"): dict(
                _mapping(metrics, description="task metrics")
            )
            for task, metrics in _mapping(value["task_metrics"], description="task metrics").items()
        },
        fold_artifacts=tuple(
            _artifact_from_dict(_mapping(item, description="fold artifact"))
            for item in _sequence(value["fold_artifacts"], description="fold artifacts")
        ),
    )


class _AtomicFitCheckpoint(FitCheckpoint):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        _require_plain_directory(self.root, description="fit checkpoint root")

    @staticmethod
    def _identity(value: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
        normalized = dict(value)
        digest = _semantic_hash(normalized)
        return normalized, digest

    def _generations(self) -> list[Path]:
        return sorted(
            (
                path
                for path in self.root.iterdir()
                if path.is_dir() and path.name.startswith("epoch-")
            ),
            key=lambda path: path.name,
        )

    def load(self, identity: Mapping[str, Any]) -> Mapping[str, bytes] | None:
        expected_identity, expected_hash = self._identity(identity)
        candidates: list[tuple[int, Path, Mapping[str, Any]]] = []
        for path in self._generations():
            manifest = verify_artifact(path)
            metadata = manifest.metadata
            if (
                metadata.get("fit_identity_sha256") != expected_hash
                or metadata.get("fit_identity") != expected_identity
            ):
                raise CheckpointError("fit checkpoint identity differs from the requested fit")
            epoch = _integer(metadata.get("epoch"), description="fit checkpoint epoch", minimum=1)
            candidates.append((epoch, path, manifest.files))
        if not candidates:
            return None
        _epoch, path, files = max(candidates, key=lambda item: item[0])
        return {
            name: _read_limited(
                path.joinpath(*PurePosixPath(name).parts),
                limit=_MAX_FIT_FILE_BYTES,
                description=f"fit checkpoint file {name}",
            )
            for name in sorted(files)
        }

    def save(
        self,
        identity: Mapping[str, Any],
        *,
        epoch: int,
        files: Mapping[str, bytes],
    ) -> None:
        normalized, identity_hash = self._identity(identity)
        epoch = _integer(epoch, description="fit checkpoint epoch", minimum=1)
        if not files:
            raise CheckpointError("fit checkpoint file set must not be empty")
        temporary = Path(tempfile.mkdtemp(prefix=".tmp-fit-", dir=self.root))
        try:
            for raw_name, payload in sorted(files.items()):
                name = _safe_relative(raw_name, description="fit checkpoint file")
                if not isinstance(payload, bytes) or len(payload) > _MAX_FIT_FILE_BYTES:
                    raise CheckpointError("fit checkpoint payload is not safe bytes")
                destination = temporary.joinpath(*PurePosixPath(name).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                _write_new(destination, payload)
            manifest = publish_artifact(
                temporary,
                stage_name=_FIT_STAGE,
                schema_version=FIT_CHECKPOINT_SCHEMA_VERSION,
                metadata={
                    "fit_identity": normalized,
                    "fit_identity_sha256": identity_hash,
                    "epoch": epoch,
                },
            )
            destination = self.root / f"epoch-{epoch:06d}-{manifest.artifact_id[:16]}"
            if destination.exists():
                if verify_artifact(destination) != manifest:
                    raise CheckpointError("fit checkpoint epoch already has different bytes")
                shutil.rmtree(temporary)
            else:
                os.replace(temporary, destination)
                if verify_artifact(destination) != manifest:
                    raise CheckpointError("published fit checkpoint failed verification")
            for previous in self._generations():
                if previous == destination:
                    continue
                obsolete = self.root / f".obsolete-{uuid.uuid4().hex}"
                os.replace(previous, obsolete)
                shutil.rmtree(obsolete)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def clear(self) -> None:
        for previous in self._generations():
            obsolete = self.root / f".obsolete-{uuid.uuid4().hex}"
            os.replace(previous, obsolete)
            shutil.rmtree(obsolete)


class CandidateCheckpointStore(CandidateResultStore):
    """Run-bound candidate and per-epoch neural checkpoint store."""

    def __init__(
        self,
        root: str | Path,
        *,
        run_id: str,
        run_semantic: Mapping[str, Any],
    ) -> None:
        if not _SAFE_ID.fullmatch(run_id):
            raise ValueError("checkpoint run_id is unsafe")
        self.run_id = run_id
        self.run_semantic = dict(run_semantic)
        self.run_semantic_sha256 = _semantic_hash(self.run_semantic)
        self.root = Path(os.path.abspath(os.fspath(Path(root).expanduser()))) / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        _require_plain_directory(self.root, description="candidate checkpoint root")
        (self.root / "candidates").mkdir(exist_ok=True)
        (self.root / "fits").mkdir(exist_ok=True)
        _require_plain_directory(
            self.root / "candidates", description="candidate checkpoint directory"
        )
        _require_plain_directory(self.root / "fits", description="fit checkpoint directory")

    def _candidate_path(self, key: CandidateExecutionKey) -> Path:
        return self.root / "candidates" / key.content_hash

    def load(self, key: CandidateExecutionKey) -> CandidateResult | None:
        path = self._candidate_path(key)
        if not path.exists():
            return None
        manifest = verify_artifact(path)
        if (
            manifest.stage_name != _CANDIDATE_STAGE
            or manifest.schema_version != CANDIDATE_CHECKPOINT_SCHEMA_VERSION
            or manifest.metadata
            != {
                "candidate_execution_hash": key.content_hash,
                "run_id": self.run_id,
                "run_semantic_sha256": self.run_semantic_sha256,
            }
            or set(manifest.files) != {"candidate_result.json"}
        ):
            raise CheckpointError("candidate checkpoint manifest identity is invalid")
        payload = _read_limited(
            path / "candidate_result.json",
            limit=_MAX_RESULT_BYTES,
            description="candidate result checkpoint",
        )
        document = _mapping(
            _strict_json(payload, description="candidate result checkpoint"),
            description="candidate result checkpoint",
        )
        if set(document) != {
            "checkpoint_schema_version",
            "execution_key",
            "result",
            "result_sha256",
        }:
            raise CheckpointError("candidate checkpoint document schema does not match")
        if document["checkpoint_schema_version"] != CANDIDATE_CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointError("unsupported candidate checkpoint schema")
        actual_key = _key_from_dict(
            _mapping(document["execution_key"], description="execution key")
        )
        if actual_key != key or actual_key.content_hash != key.content_hash:
            raise CheckpointError("candidate checkpoint belongs to another execution")
        result_document = _mapping(document["result"], description="candidate result")
        declared = document["result_sha256"]
        if not isinstance(declared, str) or _SHA256.fullmatch(declared) is None:
            raise CheckpointError("candidate result checksum is invalid")
        if _semantic_hash(result_document) != declared:
            raise CheckpointError("candidate result checksum does not close")
        return _result_from_dict(result_document)

    def save(self, key: CandidateExecutionKey, result: CandidateResult) -> None:
        path = self._candidate_path(key)
        if path.exists():
            existing = self.load(key)
            if existing is None or _result_document(existing) != _result_document(result):
                raise CheckpointError("candidate checkpoint already has different content")
            return
        result_document = _result_document(result)
        document = {
            "checkpoint_schema_version": CANDIDATE_CHECKPOINT_SCHEMA_VERSION,
            "execution_key": key.to_dict(),
            "result": result_document,
            "result_sha256": _semantic_hash(result_document),
        }
        temporary = Path(tempfile.mkdtemp(prefix=".tmp-candidate-", dir=path.parent))
        try:
            _write_new(temporary / "candidate_result.json", _canonical_bytes(document) + b"\n")
            manifest = publish_artifact(
                temporary,
                stage_name=_CANDIDATE_STAGE,
                schema_version=CANDIDATE_CHECKPOINT_SCHEMA_VERSION,
                metadata={
                    "candidate_execution_hash": key.content_hash,
                    "run_id": self.run_id,
                    "run_semantic_sha256": self.run_semantic_sha256,
                },
            )
            os.replace(temporary, path)
            if verify_artifact(path) != manifest:
                raise CheckpointError("published candidate checkpoint failed verification")
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        loaded = self.load(key)
        if loaded is None or _result_document(loaded) != result_document:
            raise CheckpointError("candidate checkpoint round-trip changed the result")
        fit_root = self.root / "fits" / key.content_hash
        if fit_root.exists():
            checkpoint = _AtomicFitCheckpoint(fit_root)
            checkpoint.clear()

    def fit_checkpoint(
        self,
        key: CandidateExecutionKey,
        fold: int,
    ) -> FitCheckpoint:
        fold = _integer(fold, description="fit checkpoint fold")
        return _AtomicFitCheckpoint(self.root / "fits" / key.content_hash / f"fold-{fold:02d}")


__all__ = [
    "CANDIDATE_CHECKPOINT_SCHEMA_VERSION",
    "FIT_CHECKPOINT_SCHEMA_VERSION",
    "CandidateCheckpointStore",
    "CheckpointError",
]
