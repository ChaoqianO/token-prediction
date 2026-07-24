from __future__ import annotations

import hashlib
import inspect
import json
import math
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum, StrEnum
from functools import lru_cache
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

from token_prediction.dataset import (
    DatasetRow,
    DatasetSlice,
    PredictionPoint,
    PredictionPosition,
    PredictionTarget,
    SplitPlan,
    SupervisedDataset,
    assign_inner_task_folds,
    build_lifecycle_slice,
)
from token_prediction.dataset.lifecycle import lifecycle_condition_task_ids
from token_prediction.development import (
    OUTER_FOLDS,
    DevelopmentProtocol,
    NestedDevelopmentPlan,
    OuterInnerPlan,
)
from token_prediction.estimators import (
    EstimatorRegistry,
    FitCheckpoint,
    FitContext,
    ObservedTransition,
    RunContext,
    TokenForecast,
    TrainingExample,
    TrainingView,
)
from token_prediction.evaluation import (
    METRIC_SUITE_ID,
    CalibrationExample,
    IdentityCalibrator,
    ScoredForecast,
    TaskMaxConformalCalibrator,
    evaluate_forecasts,
    evaluate_task_forecasts,
)
from token_prediction.features import FEATURE_SCHEMA_VERSION, FeatureSet
from token_prediction.lifecycle import visible_spend_delta


class CandidateRole(StrEnum):
    BASELINE = "baseline"
    MODEL = "model"
    ABLATION = "ablation"


class AblationAxis(StrEnum):
    METHOD = "method"
    FEATURE_SET = "feature_set"
    STATE_UPDATE = "state_update"
    SEED_POLICY = "seed_policy"
    PROBE_INTERVAL = "probe_interval"
    CALIBRATION = "calibration"


POINT_LIFECYCLE_SCHEMA_ID = "point_cell_v1"
NO_INITIALIZER_ID = "none"
NO_SEED_POLICY_ID = "none"
NO_INNER_SPLIT_POLICY_ID = "none"


@lru_cache(maxsize=1)
def _source_tree_hash() -> str:
    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class CandidateGraph:
    """The initializer/updater DAG and lifecycle policies in candidate identity."""

    updater_estimator_id: str
    initializer_estimator_id: str = NO_INITIALIZER_ID
    lifecycle_schema_id: str = POINT_LIFECYCLE_SCHEMA_ID
    seed_policy_id: str = NO_SEED_POLICY_ID
    inner_split_policy_id: str = NO_INNER_SPLIT_POLICY_ID

    def __post_init__(self) -> None:
        values = (
            self.updater_estimator_id,
            self.initializer_estimator_id,
            self.lifecycle_schema_id,
            self.seed_policy_id,
            self.inner_split_policy_id,
        )
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("candidate graph identities must be non-empty strings")
        if self.initializer_estimator_id == NO_INITIALIZER_ID:
            if (
                self.lifecycle_schema_id != POINT_LIFECYCLE_SCHEMA_ID
                or self.seed_policy_id != NO_SEED_POLICY_ID
                or self.inner_split_policy_id != NO_INNER_SPLIT_POLICY_ID
            ):
                raise ValueError("point candidates cannot declare lifecycle seed policies")
        elif (
            self.lifecycle_schema_id == POINT_LIFECYCLE_SCHEMA_ID
            or self.seed_policy_id == NO_SEED_POLICY_ID
            or self.inner_split_policy_id == NO_INNER_SPLIT_POLICY_ID
        ):
            raise ValueError("lifecycle candidates require lifecycle, seed, and inner-split ids")

    @property
    def is_lifecycle(self) -> bool:
        return self.initializer_estimator_id != NO_INITIALIZER_ID

    def to_dict(self) -> dict[str, str]:
        return {
            "initializer_estimator_id": self.initializer_estimator_id,
            "updater_estimator_id": self.updater_estimator_id,
            "lifecycle_schema_id": self.lifecycle_schema_id,
            "seed_policy_id": self.seed_policy_id,
            "inner_split_policy_id": self.inner_split_policy_id,
        }


@dataclass(frozen=True)
class AblationSpec:
    reference_candidate_id: str
    axis: AblationAxis
    allowed_config_paths: frozenset[str]

    def __post_init__(self) -> None:
        if not self.reference_candidate_id or not self.allowed_config_paths:
            raise ValueError("ablation reference and allowed config paths are required")


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    estimator_id: str
    feature_set: FeatureSet
    params: Mapping[str, Any] = field(default_factory=dict)
    initializer_params: Mapping[str, Any] = field(default_factory=dict)
    graph: CandidateGraph | None = None
    role: CandidateRole = CandidateRole.MODEL
    ablation: AblationSpec | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.estimator_id:
            raise ValueError("candidate_id and estimator_id are required")
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "initializer_params", dict(self.initializer_params))
        try:
            json.dumps(
                {
                    "updater": self.params,
                    "initializer": self.initializer_params,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("candidate params must be finite canonical JSON") from exc
        graph = self.graph or CandidateGraph(updater_estimator_id=self.estimator_id)
        if graph.updater_estimator_id != self.estimator_id:
            raise ValueError("candidate graph updater must match estimator_id")
        if not graph.is_lifecycle and self.initializer_params:
            raise ValueError("point candidates cannot declare initializer_params")
        object.__setattr__(self, "graph", graph)
        if self.role == CandidateRole.ABLATION and self.ablation is None:
            raise ValueError("ablation candidates require an AblationSpec")
        if self.role != CandidateRole.ABLATION and self.ablation is not None:
            raise ValueError("only ablation candidates may declare an AblationSpec")

    @property
    def content_hash(self) -> str:
        payload = {
            "estimator_id": self.estimator_id,
            "feature_set_hash": self.feature_set.content_hash,
            "params": self.params,
            "initializer_params": self.initializer_params,
            "graph": self.graph.to_dict(),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    position: PredictionPosition
    target: PredictionTarget
    candidates: tuple[CandidateSpec, ...]
    alpha: float = 0.10
    calibrator_id: str = "task_max_conformal"
    required_features: frozenset[str] = frozenset()
    condition_id: str | None = None

    def __post_init__(self) -> None:
        if not self.experiment_id or not self.candidates:
            raise ValueError("experiment id and candidates are required")
        if len({candidate.candidate_id for candidate in self.candidates}) != len(self.candidates):
            raise ValueError("candidate ids must be unique")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")


@dataclass(frozen=True)
class PredictionRecord:
    candidate_id: str
    point_id: str
    task_id: str
    trajectory_id: str
    condition_id: str
    fold: int
    target: PredictionTarget
    forecast: TokenForecast
    sample_weight: float


@dataclass(frozen=True)
class FoldArtifact:
    """Optional, estimator-provided audit material for one fitted fold."""

    fold: int
    encoder: Mapping[str, Any] | None = None
    fit_report: Mapping[str, Any] | None = None
    feature_importance: tuple[Mapping[str, Any], ...] | None = None
    model_strings: Mapping[str, str] | None = None
    bundle_files: Mapping[str, bytes] | None = None
    calibrator: Mapping[str, Any] | None = None
    provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.fold < 0:
            raise ValueError("fold must be non-negative")
        if self.encoder is not None:
            object.__setattr__(self, "encoder", MappingProxyType(dict(self.encoder)))
        if self.fit_report is not None:
            object.__setattr__(self, "fit_report", MappingProxyType(dict(self.fit_report)))
        if self.feature_importance is not None:
            object.__setattr__(
                self,
                "feature_importance",
                tuple(MappingProxyType(dict(record)) for record in self.feature_importance),
            )
        if self.model_strings is not None:
            models = dict(self.model_strings)
            if any(not isinstance(name, str) or not name for name in models):
                raise TypeError("model string names must be non-empty strings")
            if any(not isinstance(value, str) for value in models.values()):
                raise TypeError("model_strings() values must be strings")
            object.__setattr__(self, "model_strings", MappingProxyType(models))
        if self.calibrator is not None:
            object.__setattr__(
                self,
                "calibrator",
                MappingProxyType(dict(self.calibrator)),
            )
        if self.provenance is not None:
            object.__setattr__(
                self,
                "provenance",
                MappingProxyType(dict(self.provenance)),
            )
        if self.bundle_files is not None:
            bundle = dict(self.bundle_files)
            for name, payload in bundle.items():
                if (
                    not isinstance(name, str)
                    or not name
                    or name in {".", ".."}
                    or "\\" in name
                    or name != name.strip()
                ):
                    raise ValueError("bundle file names must be safe relative POSIX paths")
                relative = PurePosixPath(name)
                windows = PureWindowsPath(name)
                if (
                    relative.is_absolute()
                    or windows.is_absolute()
                    or windows.drive
                    or any(part in {"", ".", ".."} for part in relative.parts)
                    or relative.as_posix() != name
                ):
                    raise ValueError("bundle file names must be safe relative POSIX paths")
                if not isinstance(payload, bytes):
                    raise TypeError("bundle_files() values must be bytes")
            object.__setattr__(self, "bundle_files", MappingProxyType(bundle))

    @property
    def has_payload(self) -> bool:
        return any(
            value is not None
            for value in (
                self.encoder,
                self.fit_report,
                self.feature_importance,
                self.model_strings,
                self.bundle_files,
            )
        )


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    candidate_hash: str
    dataset_id: str
    split_plan_id: str
    eligibility_hash: str
    position: PredictionPosition
    target: PredictionTarget
    condition_id: str
    calibrator_id: str
    alpha: float
    metric_suite_id: str
    predictions: tuple[PredictionRecord, ...]
    metrics: Mapping[str, float | int | str]
    fold_metrics: Mapping[int, Mapping[str, float | int | str]] = field(default_factory=dict)
    task_metrics: Mapping[str, Mapping[str, float | int]] = field(default_factory=dict)
    fold_artifacts: tuple[FoldArtifact, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        normalized_fold_metrics: dict[int, Mapping[str, float | int | str]] = {}
        for fold, metrics in self.fold_metrics.items():
            fold_number = int(fold)
            if fold_number < 0:
                raise ValueError("fold metric keys must be non-negative")
            normalized_fold_metrics[fold_number] = MappingProxyType(dict(metrics))
        object.__setattr__(
            self,
            "fold_metrics",
            MappingProxyType(dict(sorted(normalized_fold_metrics.items()))),
        )
        normalized_task_metrics: dict[str, Mapping[str, float | int]] = {}
        for task_id, metrics in self.task_metrics.items():
            if not isinstance(task_id, str) or not task_id:
                raise ValueError("task metric keys must be non-empty task ids")
            normalized_task_metrics[task_id] = MappingProxyType(dict(metrics))
        if normalized_task_metrics:
            predicted_tasks = {record.task_id for record in self.predictions}
            if set(normalized_task_metrics) != predicted_tasks:
                raise ValueError("task metrics must exactly cover predicted tasks")
            if sum(int(item["n_points"]) for item in normalized_task_metrics.values()) != len(
                self.predictions
            ):
                raise ValueError("task metric point counts do not close")
        object.__setattr__(
            self,
            "task_metrics",
            MappingProxyType(dict(sorted(normalized_task_metrics.items()))),
        )
        artifacts = tuple(self.fold_artifacts)
        if len({artifact.fold for artifact in artifacts}) != len(artifacts):
            raise ValueError("candidate fold artifacts must have unique fold ids")
        object.__setattr__(
            self, "fold_artifacts", tuple(sorted(artifacts, key=lambda item: item.fold))
        )

    @property
    def comparability_key(self) -> tuple[str, ...]:
        return (
            self.dataset_id,
            self.split_plan_id,
            self.eligibility_hash,
            self.position.value,
            self.target.value,
            self.condition_id,
            self.calibrator_id,
            str(self.alpha),
            self.metric_suite_id,
        )


def _json_compatible(value: Any, *, context: str) -> Any:
    """Convert audit values without relying on deepcopy-based dataclasses.asdict."""

    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_compatible(getattr(value, item.name), context=context)
            for item in fields(value)
        }
    if isinstance(value, Enum):
        return _json_compatible(value.value, context=context)
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, Enum):
                key = key.value
            if not isinstance(key, (str, int, float, bool)):
                raise TypeError(f"{context} contains a non-JSON mapping key")
            converted[str(key)] = _json_compatible(item, context=context)
        return converted
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item, context=context) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [_json_compatible(item, context=context) for item in value]
        return sorted(converted, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"{context} contains unsupported value {type(value).__name__}")


def _mapping_artifact(value: Any, *, context: str) -> Mapping[str, Any]:
    converted = _json_compatible(value, context=context)
    if not isinstance(converted, dict):
        raise TypeError(f"{context} must serialize to a JSON object")
    return converted


@dataclass(frozen=True)
class CandidateExecutionKey:
    """Complete semantic identity for one resumable candidate/seed execution."""

    experiment_id: str
    candidate_id: str
    candidate_hash: str
    dataset_id: str
    split_plan_id: str
    split_seed: int
    eligibility_hash: str
    position: PredictionPosition
    target: PredictionTarget
    condition_id: str
    calibrator_id: str
    alpha: float
    source_provenance_hash: str

    def __post_init__(self) -> None:
        for name in (
            "experiment_id",
            "candidate_id",
            "dataset_id",
            "condition_id",
            "calibrator_id",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"candidate execution {name} is required")
        for name in (
            "candidate_hash",
            "split_plan_id",
            "eligibility_hash",
            "source_provenance_hash",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or value != value.lower()
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"candidate execution {name} must be a SHA-256 digest")
        if isinstance(self.split_seed, bool) or not isinstance(self.split_seed, int):
            raise ValueError("candidate execution split_seed must be an integer")
        if not math.isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise ValueError("candidate execution alpha must be in (0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "candidate_id": self.candidate_id,
            "candidate_hash": self.candidate_hash,
            "dataset_id": self.dataset_id,
            "split_plan_id": self.split_plan_id,
            "split_seed": self.split_seed,
            "eligibility_hash": self.eligibility_hash,
            "position": self.position.value,
            "target": self.target.value,
            "condition_id": self.condition_id,
            "calibrator_id": self.calibrator_id,
            "alpha": self.alpha,
            "source_provenance_hash": self.source_provenance_hash,
        }

    @property
    def content_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class CandidateResultStore(Protocol):
    """Safe persistence boundary used to resume long development matrices."""

    def load(self, key: CandidateExecutionKey) -> CandidateResult | None: ...

    def save(self, key: CandidateExecutionKey, result: CandidateResult) -> None: ...

    def fit_checkpoint(
        self,
        key: CandidateExecutionKey,
        fold: int,
    ) -> FitCheckpoint | None: ...


def _source_provenance_hash(value: Mapping[str, object] | None) -> str:
    normalized = _json_compatible(
        dict(value or {}),
        context="candidate execution source provenance",
    )
    payload = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _candidate_execution_key(
    *,
    spec: ExperimentSpec,
    candidate: CandidateSpec,
    dataset_slice: DatasetSlice,
    split_plan: SplitPlan,
    seed: int,
    source_provenance: Mapping[str, object] | None,
) -> CandidateExecutionKey:
    return CandidateExecutionKey(
        experiment_id=spec.experiment_id,
        candidate_id=candidate.candidate_id,
        candidate_hash=candidate.content_hash,
        dataset_id=dataset_slice.dataset_id,
        split_plan_id=split_plan.split_plan_id,
        split_seed=seed,
        eligibility_hash=dataset_slice.eligibility_hash,
        position=dataset_slice.position,
        target=dataset_slice.target,
        condition_id=dataset_slice.condition_id,
        calibrator_id=spec.calibrator_id,
        alpha=spec.alpha,
        source_provenance_hash=_source_provenance_hash(source_provenance),
    )


def _validate_resumed_candidate_result(
    result: CandidateResult,
    *,
    key: CandidateExecutionKey,
    dataset_slice: DatasetSlice,
    split_plan: SplitPlan,
) -> None:
    """Recompute every scored projection before accepting a checkpoint."""

    expected_identity = (
        key.candidate_id,
        key.candidate_hash,
        key.dataset_id,
        key.split_plan_id,
        key.eligibility_hash,
        key.position,
        key.target,
        key.condition_id,
        key.calibrator_id,
        key.alpha,
        METRIC_SUITE_ID,
    )
    actual_identity = (
        result.candidate_id,
        result.candidate_hash,
        result.dataset_id,
        result.split_plan_id,
        result.eligibility_hash,
        result.position,
        result.target,
        result.condition_id,
        result.calibrator_id,
        result.alpha,
        result.metric_suite_id,
    )
    if actual_identity != expected_identity:
        raise ValueError("resumed candidate result identity differs from its execution key")

    rows = {row.point.point_id: row for row in dataset_slice.rows}
    weighted = {
        item.row.point.point_id: item.sample_weight for item in dataset_slice.weighted_rows()
    }
    records = {record.point_id: record for record in result.predictions}
    if len(records) != len(result.predictions) or set(records) != set(rows):
        raise ValueError("resumed candidate result changed the eligible point cohort")

    scored: list[ScoredForecast] = []
    scored_by_fold: dict[int, list[ScoredForecast]] = defaultdict(list)
    assignment = split_plan.task_to_fold
    for point_id, row in rows.items():
        record = records[point_id]
        point = row.point
        expected_fold = assignment.get(point.task_id)
        if expected_fold is None:
            raise ValueError("resumed candidate task is absent from the split plan")
        if (
            record.task_id != point.task_id
            or record.trajectory_id != point.trajectory_id
            or record.condition_id != point.condition_id
            or record.fold != expected_fold
            or record.target != point.target
            or record.forecast.point_id != point_id
            or record.forecast.target != point.target
            or not math.isclose(
                record.sample_weight,
                weighted[point_id],
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise ValueError("resumed candidate prediction differs from its frozen cohort")
        if row.label is None:
            raise ValueError("resumed eligible candidate row has no label")
        item = ScoredForecast(
            task_id=record.task_id,
            trajectory_id=record.trajectory_id,
            forecast=record.forecast,
            target_value=float(row.label),
            sample_weight=record.sample_weight,
        )
        scored.append(item)
        scored_by_fold[record.fold].append(item)

    expected_metrics = evaluate_forecasts(scored, alpha=key.alpha)
    if dict(result.metrics) != dict(expected_metrics):
        raise ValueError("resumed candidate aggregate metrics do not recompute")
    expected_fold_metrics = {
        fold: dict(evaluate_forecasts(scored_by_fold[fold], alpha=key.alpha))
        for fold in range(split_plan.folds)
    }
    if {fold: dict(value) for fold, value in result.fold_metrics.items()} != (
        expected_fold_metrics
    ):
        raise ValueError("resumed candidate fold metrics do not recompute")
    expected_task_metrics = {
        task_id: metric.to_dict()
        for task_id, metric in evaluate_task_forecasts(scored, alpha=key.alpha).items()
    }
    if {task: dict(value) for task, value in result.task_metrics.items()} != (
        expected_task_metrics
    ):
        raise ValueError("resumed candidate task metrics do not recompute")


def _collect_fold_artifact(
    fitted: Any,
    *,
    fold: int,
    calibrator: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> FoldArtifact | None:
    encoder_payload: Mapping[str, Any] | None = None
    encoder = getattr(fitted, "encoder", None)
    encoder_to_dict = getattr(encoder, "to_dict", None)
    if callable(encoder_to_dict):
        encoder_payload = _mapping_artifact(encoder_to_dict(), context="fitted encoder.to_dict()")

    fit_report_payload: Mapping[str, Any] | None = None
    fit_report = getattr(fitted, "fit_report", None)
    if fit_report is not None:
        fit_report_payload = _mapping_artifact(fit_report, context="fitted fit_report")

    importance_payload: tuple[Mapping[str, Any], ...] | None = None
    importance_method = getattr(fitted, "source_feature_importance", None)
    if callable(importance_method):
        converted = _json_compatible(
            importance_method(), context="fitted source_feature_importance()"
        )
        if not isinstance(converted, list) or any(
            not isinstance(record, dict) for record in converted
        ):
            raise TypeError("fitted source_feature_importance() must serialize to JSON objects")
        importance_payload = tuple(converted)

    model_payload: Mapping[str, str] | None = None
    model_method = getattr(fitted, "model_strings", None)
    if callable(model_method):
        converted = _mapping_artifact(model_method(), context="fitted model_strings()")
        if any(not isinstance(value, str) for value in converted.values()):
            raise TypeError("fitted model_strings() values must be strings")
        model_payload = converted

    bundle_payload: Mapping[str, bytes] | None = None
    bundle_method = getattr(fitted, "bundle_files", None)
    if callable(bundle_method):
        parameters = inspect.signature(bundle_method).parameters.values()
        supports_keywords = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
        )
        names = {parameter.name for parameter in parameters}
        if supports_keywords or {"calibrator", "provenance"} <= names:
            if calibrator is None or provenance is None:
                raise ValueError("composite bundle creation requires calibrator and provenance")
            raw_bundle = bundle_method(
                calibrator=dict(calibrator),
                provenance=dict(provenance),
            )
        else:
            raw_bundle = bundle_method()
        if not isinstance(raw_bundle, Mapping):
            raise TypeError("fitted bundle_files() must return a mapping")
        bundle_payload = dict(raw_bundle)

    artifact = FoldArtifact(
        fold=fold,
        encoder=encoder_payload,
        fit_report=fit_report_payload,
        feature_importance=importance_payload,
        model_strings=model_payload,
        bundle_files=bundle_payload,
        calibrator=calibrator,
        provenance=provenance,
    )
    return artifact if artifact.has_payload else None


def _resolved_candidate(candidate: CandidateSpec) -> dict[str, Any]:
    resolved = {
        "estimator_id": candidate.estimator_id,
        "feature_set": candidate.feature_set.content_hash,
        **{f"params.{key}": value for key, value in sorted(candidate.params.items())},
        **{
            f"initializer_params.{key}": value
            for key, value in sorted(candidate.initializer_params.items())
        },
    }
    resolved.update({f"graph.{key}": value for key, value in candidate.graph.to_dict().items()})
    return resolved


def validate_ablation_specs(candidates: Sequence[CandidateSpec]) -> None:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    for candidate in candidates:
        if candidate.ablation is None:
            continue
        try:
            reference = by_id[candidate.ablation.reference_candidate_id]
        except KeyError as exc:
            raise ValueError(
                f"ablation reference {candidate.ablation.reference_candidate_id!r} is missing"
            ) from exc
        before = _resolved_candidate(reference)
        after = _resolved_candidate(candidate)
        paths = set(before) | set(after)
        actual = {path for path in paths if before.get(path) != after.get(path)}
        if actual != set(candidate.ablation.allowed_config_paths):
            raise ValueError(
                f"ablation {candidate.candidate_id!r} changed {sorted(actual)}, "
                f"expected {sorted(candidate.ablation.allowed_config_paths)}"
            )


def _select_point(point: PredictionPoint, feature_set: FeatureSet) -> PredictionPoint:
    return point.with_features(feature_set.select(point.features))


def _transition_spend(
    previous: PredictionPoint,
    current: PredictionPoint,
) -> int | None:
    """Compatibility wrapper for the shared offline/shadow lifecycle driver."""

    return visible_spend_delta(previous, current)


def _predict_rows(
    fitted: Any,
    rows: Sequence[DatasetRow],
    *,
    dataset_slice: DatasetSlice,
    feature_set: FeatureSet,
) -> dict[str, TokenForecast]:
    by_trajectory: dict[str, list[DatasetRow]] = defaultdict(list)
    for row in rows:
        by_trajectory[row.point.trajectory_id].append(row)
    predictions: dict[str, TokenForecast] = {}
    for trajectory_id in sorted(by_trajectory):
        sequence = sorted(
            by_trajectory[trajectory_id],
            key=lambda row: (row.point.cutoff_event_seq, row.point.point_id),
        )
        first = sequence[0].point
        session = fitted.start(
            RunContext(
                first.task_id,
                trajectory_id,
                first.run_id,
                dataset_id=dataset_slice.dataset_id,
                condition_id=dataset_slice.condition_id,
                target=dataset_slice.target,
                input_contract_hash=dataset_slice.input_contract_hash,
            )
        )
        previous: PredictionPoint | None = None
        for row in sequence:
            selected = _select_point(row.point, feature_set)
            if previous is not None:
                spend = _transition_spend(previous, row.point)
                session.observe(ObservedTransition(previous.point_id, selected.point_id, spend))
            started = time.perf_counter_ns()
            forecast = session.predict(selected)
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            if forecast.point_id != selected.point_id or forecast.target != selected.target:
                raise ValueError("estimator returned a forecast for the wrong point or target")
            if forecast.point_id in predictions:
                raise ValueError("estimator returned a duplicate prediction")
            predictions[forecast.point_id] = forecast.with_latency(elapsed_ms)
            previous = row.point
    expected = {row.point.point_id for row in rows}
    if set(predictions) != expected:
        raise ValueError("estimator did not return exactly one forecast per requested point")
    return predictions


def _verify_point_fold_reload(
    artifact: FoldArtifact | None,
    *,
    fitted: Any,
    dataset_slice: DatasetSlice,
    feature_set: FeatureSet,
    test_rows: Sequence[DatasetRow],
    raw_test_forecasts: Mapping[str, TokenForecast],
    calibrator: Any,
    expected_provenance: Mapping[str, Any],
    expected_source_provenance: Mapping[str, Any],
) -> None:
    """Load a serialized learned point model and replay its calibrated test cell."""

    estimator_id = getattr(fitted, "estimator_id", None)
    if estimator_id not in {"independent_mlp", "lightgbm_quantile"}:
        return
    if artifact is None or artifact.bundle_files is None:
        raise ValueError(f"{estimator_id} fold did not produce a reloadable bundle")
    if artifact.calibrator is None or dict(artifact.calibrator) != calibrator.to_dict():
        raise ValueError(f"{estimator_id} fold changed its calibrator document")
    if artifact.provenance is None:
        raise ValueError(f"{estimator_id} fold did not retain provenance")
    actual_artifact_provenance = _json_compatible(
        artifact.provenance,
        context=f"{estimator_id} artifact provenance",
    )
    wanted_provenance = _json_compatible(
        expected_provenance,
        context=f"expected {estimator_id} provenance",
    )
    if actual_artifact_provenance != wanted_provenance:
        raise ValueError(f"{estimator_id} fold changed provenance")

    if estimator_id == "independent_mlp":
        # Imported lazily so point baselines remain free of neural dependencies.
        from token_prediction.estimators.neural_bundle import load_neural_bundle
    else:
        from token_prediction.estimators.lightgbm_bundle import load_lightgbm_bundle

    with tempfile.TemporaryDirectory(prefix="token-prediction-point-reload-") as temporary:
        root = Path(temporary)
        for name, payload in sorted(artifact.bundle_files.items()):
            destination = root.joinpath(*PurePosixPath(name).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        if estimator_id == "independent_mlp":
            reloaded = load_neural_bundle(
                root,
                expected_source_provenance=expected_source_provenance,
            )
            actual_provenance = _json_compatible(
                reloaded.provenance,
                context="reloaded Independent MLP provenance",
            )
            if actual_provenance != wanted_provenance:
                raise ValueError("Independent MLP bundle reload changed fold provenance")
        else:
            reloaded = load_lightgbm_bundle(root)

    replayed = _predict_rows(
        reloaded,
        test_rows,
        dataset_slice=dataset_slice,
        feature_set=feature_set,
    )
    if set(replayed) != set(raw_test_forecasts):
        raise ValueError(f"{estimator_id} bundle reload changed the test point cohort")
    for point_id, raw_forecast in raw_test_forecasts.items():
        expected = calibrator.transform(raw_forecast)
        actual_raw = replayed[point_id]
        actual = (
            actual_raw if estimator_id == "independent_mlp" else calibrator.transform(actual_raw)
        )
        # Latency is measured around each invocation and is intentionally not a
        # serialized model output.  Replacing only that measurement leaves every
        # calibrated/raw quantile, scope field, and overhead counter under exact
        # equality.
        if replace(expected, latency_ms=actual.latency_ms) != actual:
            raise ValueError(f"{estimator_id} bundle reload changed a calibrated test forecast")


def _make_training_view(
    dataset_slice: DatasetSlice,
    rows: Sequence[DatasetRow],
    weights: Mapping[str, float],
    feature_set: FeatureSet,
) -> TrainingView:
    return TrainingView(
        dataset_id=dataset_slice.dataset_id,
        position=dataset_slice.position,
        target=dataset_slice.target,
        input_contract_hash=dataset_slice.input_contract_hash,
        examples=tuple(
            TrainingExample(
                point=_select_point(row.point, feature_set),
                target_value=float(row.label),
                sample_weight=weights[row.point.point_id],
            )
            for row in rows
            if row.label is not None
        ),
    )


def run_candidate_cv(
    dataset_slice: DatasetSlice,
    split_plan: SplitPlan,
    candidate: CandidateSpec,
    registry: EstimatorRegistry,
    *,
    alpha: float,
    calibrator_id: str,
    seed: int,
    source_provenance: Mapping[str, object] | None = None,
    fit_checkpoint_factory: Callable[[int], FitCheckpoint | None] | None = None,
) -> CandidateResult:
    if split_plan.dataset_id != dataset_slice.dataset_id:
        raise ValueError("split plan belongs to another dataset")
    split_plan.validate_tasks(
        (row.point.task_id for row in dataset_slice.rows), require_exact=False
    )
    if not dataset_slice.rows:
        raise ValueError("experiment cell has no eligible points")
    if candidate.estimator_id == "independent_mlp" and source_provenance is None:
        raise ValueError("Independent MLP experiments require source provenance")
    weighted = dataset_slice.weighted_rows()
    weights = {item.row.point.point_id: item.sample_weight for item in weighted}
    all_records: list[PredictionRecord] = []
    all_scored: list[ScoredForecast] = []
    fold_metrics: dict[int, Mapping[str, float | int | str]] = {}
    fold_artifacts: list[FoldArtifact] = []

    for fold in range(split_plan.folds):
        partition = split_plan.partition(fold)
        train_rows = [
            row for row in dataset_slice.rows if row.point.task_id in partition.train_tasks
        ]
        validation_rows = [
            row for row in dataset_slice.rows if row.point.task_id in partition.validation_tasks
        ]
        calibration_rows = [
            row for row in dataset_slice.rows if row.point.task_id in partition.calibration_tasks
        ]
        test_rows = [row for row in dataset_slice.rows if row.point.task_id in partition.test_tasks]
        if not train_rows or not validation_rows or not calibration_rows or not test_rows:
            raise ValueError(
                f"fold {fold} has an empty train/validation/calibration/test partition"
            )
        estimator = registry.create(candidate.estimator_id, candidate.params)
        fitted = estimator.fit(
            _make_training_view(dataset_slice, train_rows, weights, candidate.feature_set),
            _make_training_view(dataset_slice, validation_rows, weights, candidate.feature_set),
            FitContext(
                seed=seed,
                fold=fold,
                interval_alpha=alpha,
                checkpoint=(
                    fit_checkpoint_factory(fold) if fit_checkpoint_factory is not None else None
                ),
            ),
        )
        calibration_forecasts = _predict_rows(
            fitted,
            calibration_rows,
            dataset_slice=dataset_slice,
            feature_set=candidate.feature_set,
        )
        calibration_examples = [
            CalibrationExample(
                task_id=row.point.task_id,
                forecast=calibration_forecasts[row.point.point_id],
                target_value=float(row.label),
            )
            for row in calibration_rows
            if row.label is not None
        ]
        if calibrator_id == "task_max_conformal":
            calibrator = TaskMaxConformalCalibrator(alpha=alpha).fit(calibration_examples)
        elif calibrator_id == "none":
            calibrator = IdentityCalibrator(alpha=alpha).fit(calibration_examples)
        else:
            raise ValueError(f"unknown calibrator {calibrator_id!r}")
        calibrator_document = calibrator.to_dict()
        fold_provenance = {
            "bundle_role": "point_model",
            "candidate_id": candidate.candidate_id,
            "candidate_hash": candidate.content_hash,
            "candidate_graph": candidate.graph.to_dict(),
            "dataset_id": dataset_slice.dataset_id,
            "dataset_schema_version": dataset_slice.dataset_schema_version,
            "source_descriptor_hash": dataset_slice.source_descriptor_hash,
            "capability_contract_hash": dataset_slice.capability_contract_hash,
            "split_plan_id": split_plan.split_plan_id,
            "eligibility_hash": dataset_slice.eligibility_hash,
            "feature_set_hash": candidate.feature_set.content_hash,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "position": dataset_slice.position.value,
            "target": dataset_slice.target.value,
            "condition_id": dataset_slice.condition_id,
            "input_contract_hash": dataset_slice.input_contract_hash,
            "fold": fold,
            "interval_alpha": alpha,
            "calibrator_id": calibrator_id,
            "code_hash": (
                source_provenance["code_hash"]
                if source_provenance is not None
                else _source_tree_hash()
            ),
        }
        if candidate.estimator_id == "independent_mlp":
            assert source_provenance is not None
            fold_provenance["source_descriptor"] = source_provenance["source_descriptor"]
        fold_artifact = _collect_fold_artifact(
            fitted,
            fold=fold,
            calibrator=calibrator_document,
            provenance=fold_provenance,
        )
        if fold_artifact is not None:
            fold_artifacts.append(fold_artifact)
        raw_test_forecasts = _predict_rows(
            fitted,
            test_rows,
            dataset_slice=dataset_slice,
            feature_set=candidate.feature_set,
        )
        _verify_point_fold_reload(
            fold_artifact,
            fitted=fitted,
            dataset_slice=dataset_slice,
            feature_set=candidate.feature_set,
            test_rows=test_rows,
            raw_test_forecasts=raw_test_forecasts,
            calibrator=calibrator,
            expected_provenance=fold_provenance,
            expected_source_provenance=(source_provenance if source_provenance is not None else {}),
        )
        fold_scored: list[ScoredForecast] = []
        for row in test_rows:
            if row.label is None:
                raise AssertionError("eligible test row has no label")
            forecast = calibrator.transform(raw_test_forecasts[row.point.point_id])
            record = PredictionRecord(
                candidate_id=candidate.candidate_id,
                point_id=row.point.point_id,
                task_id=row.point.task_id,
                trajectory_id=row.point.trajectory_id,
                condition_id=row.point.condition_id,
                fold=fold,
                target=row.point.target,
                forecast=forecast,
                sample_weight=weights[row.point.point_id],
            )
            all_records.append(record)
            scored = ScoredForecast(
                task_id=row.point.task_id,
                trajectory_id=row.point.trajectory_id,
                forecast=forecast,
                target_value=float(row.label),
                sample_weight=weights[row.point.point_id],
            )
            fold_scored.append(scored)
            all_scored.append(scored)
        fold_metrics[fold] = evaluate_forecasts(fold_scored, alpha=alpha)

    expected = {row.point.point_id for row in dataset_slice.rows}
    actual = [record.point_id for record in all_records]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("cross-validation must predict every eligible point exactly once")
    metrics = evaluate_forecasts(all_scored, alpha=alpha)
    return CandidateResult(
        candidate_id=candidate.candidate_id,
        candidate_hash=candidate.content_hash,
        dataset_id=dataset_slice.dataset_id,
        split_plan_id=split_plan.split_plan_id,
        eligibility_hash=dataset_slice.eligibility_hash,
        position=dataset_slice.position,
        target=dataset_slice.target,
        condition_id=dataset_slice.condition_id,
        calibrator_id=calibrator_id,
        alpha=alpha,
        metric_suite_id=METRIC_SUITE_ID,
        predictions=tuple(sorted(all_records, key=lambda record: record.point_id)),
        metrics=metrics,
        fold_metrics=fold_metrics,
        task_metrics={
            task_id: task_metric.to_dict()
            for task_id, task_metric in evaluate_task_forecasts(
                all_scored,
                alpha=alpha,
            ).items()
        },
        fold_artifacts=tuple(fold_artifacts),
    )


def compare_candidate_results(results: Sequence[CandidateResult]) -> dict[str, Mapping[str, Any]]:
    if not results:
        raise ValueError("no candidate results to compare")
    key = results[0].comparability_key
    if any(result.comparability_key != key for result in results[1:]):
        raise ValueError("candidate results are not comparable")
    point_ids = {record.point_id for record in results[0].predictions}
    for result in results[1:]:
        if {record.point_id for record in result.predictions} != point_ids:
            raise ValueError("candidate prediction cohorts differ")
    return {result.candidate_id: result.metrics for result in results}


class ExperimentRunner:
    def __init__(self, registry: EstimatorRegistry) -> None:
        self.registry = registry

    def run(
        self,
        dataset: SupervisedDataset,
        split_plan: SplitPlan,
        spec: ExperimentSpec,
        *,
        seed: int,
        development_protocol: DevelopmentProtocol | None = None,
        source_provenance: Mapping[str, object] | None = None,
        result_store: CandidateResultStore | None = None,
    ) -> tuple[CandidateResult, ...]:
        nested_plan: NestedDevelopmentPlan | None = None
        if development_protocol is not None:
            nested_plan = development_protocol.nested_plan_for(split_plan)
            if dataset != development_protocol.development_dataset:
                raise ValueError("production experiments require the sealed development dataset")
            if seed != nested_plan.split_seed:
                raise ValueError("production model seed must equal the frozen split seed")
            if dataset.task_ids & development_protocol.final_holdout_tasks:
                raise ValueError("final-holdout tasks leaked into production input")
        validate_ablation_specs(spec.candidates)
        dataset_slice = dataset.select(
            spec.position,
            spec.target,
            required_features=spec.required_features,
            condition_id=spec.condition_id,
        )
        lifecycle_slice = None
        if any(candidate.graph.is_lifecycle for candidate in spec.candidates):
            if spec.required_features:
                raise ValueError(
                    "lifecycle experiments keep missing features as masked context; "
                    "required_features must be empty"
                )
            if spec.position != PredictionPosition.TASK_UPDATE:
                raise ValueError("lifecycle candidates require the Task-update position")
            condition_tasks = lifecycle_condition_task_ids(
                dataset,
                target=spec.target,
                condition_id=spec.condition_id,
            )
            planned_tasks = frozenset(task for task, _fold in split_plan.assignments)
            if not condition_tasks <= planned_tasks:
                raise ValueError("lifecycle condition contains tasks outside the split plan")
            lifecycle_slice = build_lifecycle_slice(
                dataset,
                target=spec.target,
                condition_id=spec.condition_id,
                task_ids=condition_tasks,
            )
            if source_provenance is None:
                raise ValueError("lifecycle experiments require source provenance")
            if nested_plan is None:
                inner_plans = {
                    outer_fold: OuterInnerPlan(
                        split_seed=split_plan.seed,
                        outer_test_fold=outer_fold,
                        outer_split_plan_id=split_plan.split_plan_id,
                        assignment=assign_inner_task_folds(
                            split_plan.partition(outer_fold).train_tasks,
                            seed=split_plan.seed,
                        ),
                    )
                    for outer_fold in range(OUTER_FOLDS)
                }
            else:
                inner_plans = {plan.outer_test_fold: plan for plan in nested_plan.inner_plans}
        resolved: list[CandidateResult] = []
        for candidate in spec.candidates:
            execution_key = _candidate_execution_key(
                spec=spec,
                candidate=candidate,
                dataset_slice=dataset_slice,
                split_plan=split_plan,
                seed=seed,
                source_provenance=source_provenance,
            )
            cached = result_store.load(execution_key) if result_store is not None else None
            if cached is not None:
                _validate_resumed_candidate_result(
                    cached,
                    key=execution_key,
                    dataset_slice=dataset_slice,
                    split_plan=split_plan,
                )
                resolved.append(cached)
                continue
            fit_checkpoint_factory = (
                (lambda fold, *, _key=execution_key: result_store.fit_checkpoint(_key, fold))
                if result_store is not None
                else None
            )
            if candidate.graph.is_lifecycle:
                if lifecycle_slice is None:
                    raise AssertionError("lifecycle slice was not constructed")
                # Imported lazily to avoid the result-contract module cycle.
                from token_prediction.lifecycle_experiment import (
                    run_lifecycle_candidate_cv,
                )

                result = run_lifecycle_candidate_cv(
                    dataset,
                    lifecycle_slice,
                    split_plan,
                    candidate,
                    self.registry,
                    alpha=spec.alpha,
                    calibrator_id=spec.calibrator_id,
                    seed=seed,
                    inner_plans=inner_plans,
                    source_provenance=source_provenance,
                    fit_checkpoint_factory=fit_checkpoint_factory,
                )
            else:
                result = run_candidate_cv(
                    dataset_slice,
                    split_plan,
                    candidate,
                    self.registry,
                    alpha=spec.alpha,
                    calibrator_id=spec.calibrator_id,
                    seed=seed,
                    source_provenance=source_provenance,
                    fit_checkpoint_factory=fit_checkpoint_factory,
                )
            if result_store is not None:
                result_store.save(execution_key, result)
                persisted = result_store.load(execution_key)
                if persisted is None:
                    raise RuntimeError("candidate checkpoint disappeared after publication")
                _validate_resumed_candidate_result(
                    persisted,
                    key=execution_key,
                    dataset_slice=dataset_slice,
                    split_plan=split_plan,
                )
                result = persisted
            resolved.append(result)
        results = tuple(resolved)
        compare_candidate_results(results)
        return results
