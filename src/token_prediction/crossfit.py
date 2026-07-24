from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from token_prediction.dataset import PredictionPoint, PredictionPosition, PredictionTarget
from token_prediction.estimators.base import (
    FittedEstimator,
    RunContext,
    SessionSeed,
    TokenForecast,
)


SEED_POLICY_ID = "inner_oof_uncalibrated_repaired_quantile_mean_v1"
POINT_ONLY_SEED_POLICY_ID = "inner_oof_uncalibrated_repaired_point_only_mean_v1"
SEED_POLICY_DOCUMENT = {
    "seed_policy_id": SEED_POLICY_ID,
    "oof_rule": "exactly_one_inner_holdout_component",
    "external_rule": "all_inner_components",
    "ensemble_rule": "arithmetic_mean_per_repaired_quantile",
    "calibration": "none",
    "repair": "non_negative_ordered_before_propagation",
}
POINT_ONLY_SEED_POLICY_DOCUMENT = {
    "seed_policy_id": POINT_ONLY_SEED_POLICY_ID,
    "oof_rule": "exactly_one_inner_holdout_component",
    "external_rule": "all_inner_components",
    "ensemble_rule": "arithmetic_mean_of_repaired_raw_point_broadcast",
    "calibration": "none",
    "repair": "non_negative_raw_point_broadcast_before_propagation",
}
SEED_POLICY_DOCUMENTS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        SEED_POLICY_ID: MappingProxyType(SEED_POLICY_DOCUMENT),
        POINT_ONLY_SEED_POLICY_ID: MappingProxyType(POINT_ONLY_SEED_POLICY_DOCUMENT),
    }
)
_SHA256 = frozenset("0123456789abcdef")


def _require_sha256(value: str, *, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or set(value) - _SHA256:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _semantic_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def seed_policy_document(seed_policy_id: str) -> Mapping[str, str]:
    try:
        return SEED_POLICY_DOCUMENTS[seed_policy_id]
    except KeyError as exc:
        raise ValueError(f"unsupported crossfit seed policy {seed_policy_id!r}") from exc


def seed_policy_hash(seed_policy_id: str) -> str:
    return _semantic_hash(dict(seed_policy_document(seed_policy_id)))


SEED_POLICY_HASH = seed_policy_hash(SEED_POLICY_ID)
POINT_ONLY_SEED_POLICY_HASH = seed_policy_hash(POINT_ONLY_SEED_POLICY_ID)
SUPPORTED_SEED_POLICY_IDS = frozenset(SEED_POLICY_DOCUMENTS)


@dataclass(frozen=True)
class InitializerComponent:
    """One inner-fold initializer and the exact tasks that produced it."""

    inner_fold: int
    component_id: str
    component_hash: str
    bundle_hashes: tuple[str, ...]
    fit_tasks: frozenset[str]
    validation_tasks: frozenset[str]
    holdout_tasks: frozenset[str]
    fitted: FittedEstimator = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.inner_fold < 0:
            raise ValueError("inner fold must be non-negative")
        if not self.component_id.strip():
            raise ValueError("initializer component id is required")
        _require_sha256(self.component_hash, name="component_hash")
        if not isinstance(self.bundle_hashes, tuple) or not self.bundle_hashes:
            raise ValueError("initializer component requires bundle hashes")
        for index, digest in enumerate(self.bundle_hashes):
            _require_sha256(digest, name=f"bundle_hashes[{index}]")
        groups = (self.fit_tasks, self.validation_tasks, self.holdout_tasks)
        if any(not group for group in groups):
            raise ValueError("inner fit/validation/holdout task groups must be non-empty")
        if any(
            left & right
            for index, left in enumerate(groups)
            for right in groups[index + 1 :]
        ):
            raise ValueError("inner task groups must be disjoint")


@dataclass(frozen=True)
class CrossfitSeedRecord:
    point_id: str
    task_id: str
    trajectory_id: str
    role: str
    producer_inner_folds: tuple[int, ...]
    seed: SessionSeed

    def __post_init__(self) -> None:
        point = self.seed.task_pre_point
        if (
            self.point_id != point.point_id
            or self.task_id != point.task_id
            or self.trajectory_id != point.trajectory_id
        ):
            raise ValueError("seed record identity does not match its Task-pre point")
        if self.role not in {"inner_oof", "external_ensemble"}:
            raise ValueError("seed record role is unsupported")
        if not self.producer_inner_folds or len(set(self.producer_inner_folds)) != len(
            self.producer_inner_folds
        ):
            raise ValueError("seed producer folds must be unique and non-empty")
        if self.role == "inner_oof" and len(self.producer_inner_folds) != 1:
            raise ValueError("an OOF seed must come from exactly one inner fold")

    def identity_projection(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "task_id": self.task_id,
            "trajectory_id": self.trajectory_id,
            "role": self.role,
            "producer_inner_folds": self.producer_inner_folds,
            "seed_content_hash": self.seed.content_hash,
        }


@dataclass(frozen=True)
class CrossfitSeedSet:
    initializer_id: str
    initializer_hash: str
    inner_split_id: str
    records: tuple[CrossfitSeedRecord, ...]
    by_point_id: Mapping[str, SessionSeed] = field(init=False, repr=False)
    seed_policy_id: str = field(init=False)
    seed_policy_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.initializer_id.strip() or not self.inner_split_id.strip():
            raise ValueError("initializer and inner split identities are required")
        _require_sha256(self.initializer_hash, name="initializer_hash")
        if not self.records:
            raise ValueError("crossfit seed set is empty")
        point_ids = [record.point_id for record in self.records]
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("crossfit seed point ids must be unique")
        if any(
            record.seed.initializer_id != self.initializer_id
            or record.seed.initializer_hash != self.initializer_hash
            or record.seed.inner_split_id != self.inner_split_id
            for record in self.records
        ):
            raise ValueError("seed records do not share initializer/split identity")
        policy_ids = {record.seed.seed_policy_id for record in self.records}
        policy_hashes = {record.seed.seed_policy_hash for record in self.records}
        if len(policy_ids) != 1 or len(policy_hashes) != 1:
            raise ValueError("seed records do not share one seed policy")
        policy_id = next(iter(policy_ids))
        policy_hash = next(iter(policy_hashes))
        if policy_id not in SUPPORTED_SEED_POLICY_IDS:
            raise ValueError("seed records use an unsupported seed policy")
        if policy_hash != seed_policy_hash(policy_id):
            raise ValueError("seed record policy hash does not match its policy")
        object.__setattr__(self, "seed_policy_id", policy_id)
        object.__setattr__(self, "seed_policy_hash", policy_hash)
        object.__setattr__(
            self,
            "by_point_id",
            MappingProxyType({record.point_id: record.seed for record in self.records}),
        )

    @property
    def content_hash(self) -> str:
        return _semantic_hash(
            {
                "initializer_id": self.initializer_id,
                "initializer_hash": self.initializer_hash,
                "inner_split_id": self.inner_split_id,
                "seed_policy_id": self.seed_policy_id,
                "seed_policy_hash": self.seed_policy_hash,
                "records": [
                    record.identity_projection()
                    for record in sorted(self.records, key=lambda item: item.point_id)
                ],
            }
        )


def prepare_seed_forecast(
    forecast: TokenForecast,
    *,
    seed_policy_id: str = SEED_POLICY_ID,
) -> TokenForecast:
    """Normalize a raw initializer output according to a frozen seed policy."""

    seed_policy_document(seed_policy_id)
    raw = (forecast.raw_lower, forecast.raw_point, forecast.raw_upper)
    if all(value is not None for value in raw):
        raw_lower, raw_point, raw_upper = (float(value) for value in raw)
    else:
        raw_lower, raw_point, raw_upper = (
            forecast.lower,
            forecast.point,
            forecast.upper,
        )
    repaired_point = max(0.0, raw_point)
    repaired_lower = min(max(0.0, raw_lower), repaired_point)
    repaired_upper = max(max(0.0, raw_upper), repaired_point)
    if seed_policy_id == POINT_ONLY_SEED_POLICY_ID:
        repaired_lower = repaired_point
        repaired_upper = repaired_point
        raw_lower = raw_point
        raw_upper = raw_point
    return TokenForecast(
        point_id=forecast.point_id,
        target=forecast.target,
        lower=repaired_lower,
        point=repaired_point,
        upper=repaired_upper,
        raw_lower=raw_lower,
        raw_point=raw_point,
        raw_upper=raw_upper,
    )


def _seed_forecast(forecast: TokenForecast) -> TokenForecast:
    """Backward-compatible name for the frozen primary seed policy."""

    return prepare_seed_forecast(forecast, seed_policy_id=SEED_POLICY_ID)


def ensemble_repaired_forecasts(
    point: PredictionPoint,
    forecasts: Sequence[TokenForecast],
    *,
    seed_policy_id: str = SEED_POLICY_ID,
) -> TokenForecast:
    seed_policy_document(seed_policy_id)
    if not forecasts:
        raise ValueError("cannot ensemble an empty initializer forecast set")
    if any(
        forecast.point_id != point.point_id or forecast.target != point.target
        for forecast in forecasts
    ):
        raise ValueError("initializer forecasts do not match the Task-pre point")
    count = len(forecasts)
    lower = math.fsum(forecast.lower for forecast in forecasts) / count
    center = math.fsum(forecast.point for forecast in forecasts) / count
    upper = math.fsum(forecast.upper for forecast in forecasts) / count
    if not 0 <= lower <= center <= upper:
        raise ValueError("repaired initializer quantiles lost ordering during ensemble")
    if seed_policy_id == POINT_ONLY_SEED_POLICY_ID and not lower == center == upper:
        raise ValueError("point-only initializer forecasts must remain broadcast")
    # The ensemble consumes already-repaired component forecasts.  Its raw
    # diagnostics intentionally equal that repaired ensemble, proving that no
    # conformal expansion entered the recurrent seed.
    return TokenForecast(
        point_id=point.point_id,
        target=point.target,
        lower=lower,
        point=center,
        upper=upper,
        raw_lower=lower,
        raw_point=center,
        raw_upper=upper,
    )


def _predict_component(
    component: InitializerComponent,
    point: PredictionPoint,
    *,
    dataset_id: str,
    input_contract_hash: str,
    seed_policy_id: str,
) -> TokenForecast:
    session = component.fitted.start(
        RunContext(
            point.task_id,
            point.trajectory_id,
            point.run_id,
            dataset_id=dataset_id,
            condition_id=point.condition_id,
            target=point.target,
            runtime_mode="offline",
            input_contract_hash=input_contract_hash,
        )
    )
    forecast = session.predict(point)
    if forecast.point_id != point.point_id or forecast.target != point.target:
        raise ValueError("initializer returned a forecast for the wrong Task-pre point")
    return prepare_seed_forecast(forecast, seed_policy_id=seed_policy_id)


def generate_crossfit_seeds(
    task_pre_points: Iterable[PredictionPoint],
    components: Sequence[InitializerComponent],
    *,
    dataset_id: str,
    input_contract_hash: str,
    initializer_id: str,
    initializer_hash: str,
    inner_split_id: str,
    oof_tasks: frozenset[str],
    external_tasks: frozenset[str],
    seed_policy_id: str = SEED_POLICY_ID,
) -> CrossfitSeedSet:
    """Generate leakage-safe OOF and external-ensemble Task-pre seeds."""

    if not components:
        raise ValueError("inner initializer component set is empty")
    policy_hash = seed_policy_hash(seed_policy_id)
    if len({component.inner_fold for component in components}) != len(components):
        raise ValueError("inner initializer folds must be unique")
    _require_sha256(initializer_hash, name="initializer_hash")
    _require_sha256(input_contract_hash, name="input_contract_hash")
    if oof_tasks & external_tasks:
        raise ValueError("OOF and external task sets overlap")
    inner_task_universe = frozenset(
        task
        for component in components
        for task in (
            component.fit_tasks | component.validation_tasks | component.holdout_tasks
        )
    )
    if inner_task_universe != oof_tasks:
        raise ValueError("inner component task universe does not equal OOF tasks")
    holdout_count = {
        task: sum(task in component.holdout_tasks for component in components)
        for task in oof_tasks
    }
    if any(count != 1 for count in holdout_count.values()):
        raise ValueError("each OOF task must enter exactly one inner holdout")

    points = tuple(sorted(task_pre_points, key=lambda point: point.point_id))
    if not points:
        raise ValueError("Task-pre point set is empty")
    point_ids = [point.point_id for point in points]
    if len(point_ids) != len(set(point_ids)):
        raise ValueError("Task-pre point ids must be unique")
    point_tasks = {point.task_id for point in points}
    if point_tasks != oof_tasks | external_tasks:
        raise ValueError("Task-pre points do not exactly cover OOF and external tasks")
    for point in points:
        if (
            point.position != PredictionPosition.TASK_PRE
            or point.target
            != PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS
        ):
            raise ValueError("crossfit seeds require Task-pre provider-accounted points")

    ordered_components = tuple(sorted(components, key=lambda item: item.inner_fold))
    records: list[CrossfitSeedRecord] = []
    for point in points:
        if point.task_id in oof_tasks:
            selected = tuple(
                component
                for component in ordered_components
                if point.task_id in component.holdout_tasks
            )
            if len(selected) != 1:
                raise ValueError("OOF task did not resolve to exactly one holdout component")
            component = selected[0]
            if point.task_id in component.fit_tasks or point.task_id in component.validation_tasks:
                raise ValueError("OOF task leaked into its initializer fit or validation set")
            forecast = _predict_component(
                component,
                point,
                dataset_id=dataset_id,
                input_contract_hash=input_contract_hash,
                seed_policy_id=seed_policy_id,
            )
            producer_folds = (component.inner_fold,)
            bundle_hashes = component.bundle_hashes
            role = "inner_oof"
        else:
            if point.task_id in inner_task_universe:
                raise ValueError("external task entered an inner initializer partition")
            component_forecasts = tuple(
                _predict_component(
                    component,
                    point,
                    dataset_id=dataset_id,
                    input_contract_hash=input_contract_hash,
                    seed_policy_id=seed_policy_id,
                )
                for component in ordered_components
            )
            forecast = ensemble_repaired_forecasts(
                point,
                component_forecasts,
                seed_policy_id=seed_policy_id,
            )
            producer_folds = tuple(
                component.inner_fold for component in ordered_components
            )
            bundle_hashes = tuple(
                digest
                for component in ordered_components
                for digest in component.bundle_hashes
            )
            role = "external_ensemble"
        seed = SessionSeed(
            task_pre_point=point,
            forecast=forecast,
            initializer_id=initializer_id,
            initializer_hash=initializer_hash,
            inner_split_id=inner_split_id,
            component_bundle_hashes=bundle_hashes,
            seed_policy_id=seed_policy_id,
            seed_policy_hash=policy_hash,
        )
        records.append(
            CrossfitSeedRecord(
                point_id=point.point_id,
                task_id=point.task_id,
                trajectory_id=point.trajectory_id,
                role=role,
                producer_inner_folds=producer_folds,
                seed=seed,
            )
        )
    return CrossfitSeedSet(
        initializer_id=initializer_id,
        initializer_hash=initializer_hash,
        inner_split_id=inner_split_id,
        records=tuple(records),
    )


__all__ = [
    "CrossfitSeedRecord",
    "CrossfitSeedSet",
    "InitializerComponent",
    "POINT_ONLY_SEED_POLICY_DOCUMENT",
    "POINT_ONLY_SEED_POLICY_HASH",
    "POINT_ONLY_SEED_POLICY_ID",
    "SEED_POLICY_DOCUMENT",
    "SEED_POLICY_DOCUMENTS",
    "SEED_POLICY_HASH",
    "SEED_POLICY_ID",
    "SUPPORTED_SEED_POLICY_IDS",
    "ensemble_repaired_forecasts",
    "generate_crossfit_seeds",
    "prepare_seed_forecast",
    "seed_policy_document",
    "seed_policy_hash",
]
