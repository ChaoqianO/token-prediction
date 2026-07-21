from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum, StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from token_prediction.dataset import (
    DatasetRow,
    DatasetSlice,
    PredictionPoint,
    PredictionPosition,
    PredictionTarget,
    SplitPlan,
    SupervisedDataset,
)
from token_prediction.estimators import (
    EstimatorRegistry,
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
)
from token_prediction.features import FeatureSet


class CandidateRole(StrEnum):
    BASELINE = "baseline"
    MODEL = "model"
    ABLATION = "ablation"


class AblationAxis(StrEnum):
    FEATURE_SET = "feature_set"
    STATE_UPDATE = "state_update"
    PROBE_INTERVAL = "probe_interval"
    CALIBRATION = "calibration"


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
    role: CandidateRole = CandidateRole.MODEL
    ablation: AblationSpec | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.estimator_id:
            raise ValueError("candidate_id and estimator_id are required")
        object.__setattr__(self, "params", dict(self.params))
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
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
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
        if len({candidate.candidate_id for candidate in self.candidates}) != len(
            self.candidates
        ):
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
        if self.bundle_files is not None:
            bundle = dict(self.bundle_files)
            for name, payload in bundle.items():
                if (
                    not isinstance(name, str)
                    or not name
                    or name in {".", ".."}
                    or "/" in name
                    or "\\" in name
                ):
                    raise ValueError("bundle file names must be safe basenames")
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
    fold_metrics: Mapping[int, Mapping[str, float | int | str]] = field(
        default_factory=dict
    )
    fold_artifacts: tuple[FoldArtifact, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        normalized_fold_metrics: dict[
            int, Mapping[str, float | int | str]
        ] = {}
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


def _collect_fold_artifact(fitted: Any, *, fold: int) -> FoldArtifact | None:
    encoder_payload: Mapping[str, Any] | None = None
    encoder = getattr(fitted, "encoder", None)
    encoder_to_dict = getattr(encoder, "to_dict", None)
    if callable(encoder_to_dict):
        encoder_payload = _mapping_artifact(
            encoder_to_dict(), context="fitted encoder.to_dict()"
        )

    fit_report_payload: Mapping[str, Any] | None = None
    fit_report = getattr(fitted, "fit_report", None)
    if fit_report is not None:
        fit_report_payload = _mapping_artifact(
            fit_report, context="fitted fit_report"
        )

    importance_payload: tuple[Mapping[str, Any], ...] | None = None
    importance_method = getattr(fitted, "source_feature_importance", None)
    if callable(importance_method):
        converted = _json_compatible(
            importance_method(), context="fitted source_feature_importance()"
        )
        if not isinstance(converted, list) or any(
            not isinstance(record, dict) for record in converted
        ):
            raise TypeError(
                "fitted source_feature_importance() must serialize to JSON objects"
            )
        importance_payload = tuple(converted)

    model_payload: Mapping[str, str] | None = None
    model_method = getattr(fitted, "model_strings", None)
    if callable(model_method):
        converted = _mapping_artifact(
            model_method(), context="fitted model_strings()"
        )
        if any(not isinstance(value, str) for value in converted.values()):
            raise TypeError("fitted model_strings() values must be strings")
        model_payload = converted

    bundle_payload: Mapping[str, bytes] | None = None
    bundle_method = getattr(fitted, "bundle_files", None)
    if callable(bundle_method):
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
    )
    return artifact if artifact.has_payload else None


def _resolved_candidate(candidate: CandidateSpec) -> dict[str, Any]:
    return {
        "estimator_id": candidate.estimator_id,
        "feature_set": candidate.feature_set.content_hash,
        **{f"params.{key}": value for key, value in sorted(candidate.params.items())},
    }


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
    """Return newly observed billable spend, recovering after an earlier gap.

    Cumulative known usage remains usable after a missing attempt because the
    unknown historical amount cancels when the missing-attempt counter is equal
    at both endpoints.  A counter increase inside this transition makes only
    this transition unknown; it must not poison every later transition.
    """

    previous_missing = previous.features.get("missing_usage_attempts")
    current_missing = current.features.get("missing_usage_attempts")
    if not isinstance(previous_missing, int) or not isinstance(current_missing, int):
        return None
    if current_missing < previous_missing:
        raise ValueError("missing usage attempt count decreased within a trajectory")
    if current_missing != previous_missing:
        return None

    names = (
        "cumulative_provider_input_tokens",
        "cumulative_provider_output_tokens",
    )
    previous_values = tuple(previous.features.get(name) for name in names)
    current_values = tuple(current.features.get(name) for name in names)
    if not all(isinstance(value, int) for value in (*previous_values, *current_values)):
        return None
    spend = sum(int(value) for value in current_values) - sum(
        int(value) for value in previous_values
    )
    if spend < 0:
        raise ValueError("cumulative spend decreased within a trajectory")
    return spend


def _predict_rows(
    fitted: Any,
    rows: Sequence[DatasetRow],
    *,
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
            RunContext(first.task_id, trajectory_id, first.run_id)
        )
        previous: PredictionPoint | None = None
        for row in sequence:
            selected = _select_point(row.point, feature_set)
            if previous is not None:
                spend = _transition_spend(previous, row.point)
                session.observe(
                    ObservedTransition(previous.point_id, selected.point_id, spend)
                )
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
) -> CandidateResult:
    if split_plan.dataset_id != dataset_slice.dataset_id:
        raise ValueError("split plan belongs to another dataset")
    split_plan.validate_tasks(
        (row.point.task_id for row in dataset_slice.rows), require_exact=False
    )
    if not dataset_slice.rows:
        raise ValueError("experiment cell has no eligible points")
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
            row
            for row in dataset_slice.rows
            if row.point.task_id in partition.validation_tasks
        ]
        calibration_rows = [
            row
            for row in dataset_slice.rows
            if row.point.task_id in partition.calibration_tasks
        ]
        test_rows = [
            row for row in dataset_slice.rows if row.point.task_id in partition.test_tasks
        ]
        if not train_rows or not validation_rows or not calibration_rows or not test_rows:
            raise ValueError(
                f"fold {fold} has an empty train/validation/calibration/test partition"
            )
        estimator = registry.create(candidate.estimator_id, candidate.params)
        fitted = estimator.fit(
            _make_training_view(dataset_slice, train_rows, weights, candidate.feature_set),
            _make_training_view(
                dataset_slice, validation_rows, weights, candidate.feature_set
            ),
            FitContext(seed=seed, fold=fold, interval_alpha=alpha),
        )
        fold_artifact = _collect_fold_artifact(fitted, fold=fold)
        if fold_artifact is not None:
            fold_artifacts.append(fold_artifact)
        calibration_forecasts = _predict_rows(
            fitted, calibration_rows, feature_set=candidate.feature_set
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
            calibrator = IdentityCalibrator().fit(calibration_examples)
        else:
            raise ValueError(f"unknown calibrator {calibrator_id!r}")
        raw_test_forecasts = _predict_rows(
            fitted, test_rows, feature_set=candidate.feature_set
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
    ) -> tuple[CandidateResult, ...]:
        validate_ablation_specs(spec.candidates)
        dataset_slice = dataset.select(
            spec.position,
            spec.target,
            required_features=spec.required_features,
            condition_id=spec.condition_id,
        )
        results = tuple(
            run_candidate_cv(
                dataset_slice,
                split_plan,
                candidate,
                self.registry,
                alpha=spec.alpha,
                calibrator_id=spec.calibrator_id,
                seed=seed,
            )
            for candidate in spec.candidates
        )
        compare_candidate_results(results)
        return results
