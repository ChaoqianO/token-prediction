from __future__ import annotations

import hashlib
import math
import platform
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from token_prediction.dataset import PredictionPoint, PredictionPosition, PredictionTarget

from .base import FitContext, ObservedTransition, RunContext, TokenForecast, TrainingView
from .tabular_encoder import EncoderSchema, FoldTabularEncoder


LIGHTGBM_ESTIMATOR_VERSION = 1
_PROTECTED_PARAMS = frozenset(
    {
        "objective",
        "metric",
        "alpha",
        "device",
        "device_type",
        "deterministic",
        "force_col_wise",
        "force_row_wise",
        "num_threads",
        "num_thread",
        "n_jobs",
        "seed",
        "random_seed",
        "data_random_seed",
        "feature_fraction_seed",
        "bagging_seed",
        "drop_seed",
    }
)


class OptionalEstimatorDependencyError(RuntimeError):
    pass


def _load_optional_dependencies() -> tuple[Any, Any]:
    try:
        import lightgbm as lgb
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover - tested by the base-only CI job
        raise OptionalEstimatorDependencyError(
            "LightGBM estimation requires optional dependencies; "
            "install token-prediction[estimators]"
        ) from exc
    return lgb, np


def _point_hash(point_ids: tuple[str, ...]) -> str:
    encoded = "\n".join(sorted(point_ids)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _view_condition_ids(view: TrainingView, *, name: str) -> tuple[str, ...]:
    conditions: set[str] = set()
    for example in view.examples:
        point = example.point
        if point.position != view.position or point.target != view.target:
            raise ValueError(f"{name} point does not match its TrainingView cell")
        conditions.add(point.condition_id)
    if len(conditions) != 1:
        raise ValueError(f"{name} view must contain exactly one condition scope")
    return tuple(sorted(conditions))


def _derived_seed(seed: int, fold: int, quantile: float) -> int:
    payload = f"lightgbm-quantile-v{LIGHTGBM_ESTIMATOR_VERSION}:{seed}:{fold}:{quantile:.8f}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big") % (
        2**31 - 1
    )


@dataclass(frozen=True)
class QuantileFitReport:
    quantile: float
    seed: int
    best_iteration: int
    best_validation_loss: float
    validation_history: tuple[float, ...]
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


@dataclass(frozen=True)
class LightGBMFitReport:
    estimator_version: int
    encoder_schema_hash: str
    train_point_hash: str
    validation_point_hash: str
    train_point_count: int
    validation_point_count: int
    lightgbm_version: str
    numpy_version: str
    platform: str
    quantiles: tuple[QuantileFitReport, ...]


@dataclass(frozen=True)
class RawQuantileDiagnostics:
    q05: float
    q50: float
    q95: float
    crossed: bool


@dataclass(frozen=True)
class FeatureImportanceRecord:
    quantile: float
    expanded_feature_name: str
    source_feature_name: str
    gain: float
    split_count: int
    normalized_gain: float


@dataclass(frozen=True)
class SourceFeatureImportanceRecord:
    quantile: float
    source_feature_name: str
    gain: float
    split_count: int
    normalized_gain: float


@dataclass
class LightGBMQuantileSession:
    target: PredictionTarget
    position: PredictionPosition
    allowed_condition_ids: tuple[str, ...]
    encoder: FoldTabularEncoder
    boosters: Mapping[float, Any]
    best_iterations: Mapping[float, int]
    quantiles: tuple[float, float, float]
    last_raw_quantiles: RawQuantileDiagnostics | None = None
    prediction_count: int = 0
    raw_crossing_count: int = 0

    def predict(self, point: PredictionPoint) -> TokenForecast:
        if point.target != self.target:
            raise ValueError(
                f"bundle target is {self.target.value!r}, got {point.target.value!r}"
            )
        if point.position != self.position:
            raise ValueError(
                f"bundle position is {self.position.value!r}, got {point.position.value!r}"
            )
        if point.condition_id not in self.allowed_condition_ids:
            raise ValueError(
                f"condition_id {point.condition_id!r} is outside the bundle scope"
            )
        batch = self.encoder.transform((point,))
        raw = tuple(
            float(
                self.boosters[quantile].predict(
                    batch.matrix,
                    num_iteration=self.best_iterations[quantile],
                    num_threads=1,
                )[0]
            )
            for quantile in self.quantiles
        )
        crossed = raw[0] > raw[1] or raw[1] > raw[2]
        self.last_raw_quantiles = RawQuantileDiagnostics(
            q05=raw[0], q50=raw[1], q95=raw[2], crossed=crossed
        )
        self.prediction_count += 1
        self.raw_crossing_count += int(crossed)

        point_value = max(0.0, raw[1])
        lower = min(max(0.0, raw[0]), point_value)
        upper = max(max(0.0, raw[2]), point_value)
        return TokenForecast(
            point_id=point.point_id,
            target=self.target,
            lower=lower,
            point=point_value,
            upper=upper,
            raw_lower=raw[0],
            raw_point=raw[1],
            raw_upper=raw[2],
        )

    def observe(self, transition: ObservedTransition) -> None:
        del transition


@dataclass(frozen=True)
class FittedLightGBMQuantiles:
    estimator_id: str
    target: PredictionTarget
    position: PredictionPosition
    dataset_id: str
    allowed_condition_ids: tuple[str, ...]
    encoder: FoldTabularEncoder
    boosters: Mapping[float, Any]
    best_iterations: Mapping[float, int]
    quantiles: tuple[float, float, float]
    fit_report: LightGBMFitReport

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("dataset_id is required for a fitted LightGBM model")
        if not self.allowed_condition_ids:
            raise ValueError("at least one allowed condition_id is required")
        if any(not condition_id.strip() for condition_id in self.allowed_condition_ids):
            raise ValueError("allowed condition ids must be non-empty")
        if tuple(sorted(set(self.allowed_condition_ids))) != self.allowed_condition_ids:
            raise ValueError("allowed condition ids must be sorted and unique")
        object.__setattr__(self, "boosters", MappingProxyType(dict(self.boosters)))
        object.__setattr__(
            self, "best_iterations", MappingProxyType(dict(self.best_iterations))
        )

    def start(self, context: RunContext) -> LightGBMQuantileSession:
        del context
        return LightGBMQuantileSession(
            target=self.target,
            position=self.position,
            allowed_condition_ids=self.allowed_condition_ids,
            encoder=self.encoder,
            boosters=self.boosters,
            best_iterations=self.best_iterations,
            quantiles=self.quantiles,
        )

    def model_strings(self) -> dict[str, str]:
        return {
            f"q{int(round(quantile * 100)):02d}": self.boosters[quantile].model_to_string(
                num_iteration=self.best_iterations[quantile]
            )
            for quantile in self.quantiles
        }

    def bundle_files(self) -> Mapping[str, bytes]:
        """Return the strict deployable bundle for generic fold-artifact writers."""

        from .lightgbm_bundle import lightgbm_bundle_files

        return lightgbm_bundle_files(self)

    def feature_importance(self) -> tuple[FeatureImportanceRecord, ...]:
        records: list[FeatureImportanceRecord] = []
        names = self.encoder.schema.feature_names
        sources = self.encoder.schema.source_features
        for quantile in self.quantiles:
            booster = self.boosters[quantile]
            iteration = self.best_iterations[quantile]
            gains = [
                float(value)
                for value in booster.feature_importance(
                    importance_type="gain", iteration=iteration
                )
            ]
            splits = [
                int(value)
                for value in booster.feature_importance(
                    importance_type="split", iteration=iteration
                )
            ]
            if len(gains) != len(names) or len(splits) != len(names):
                raise AssertionError("LightGBM importance length does not match encoder schema")
            gain_total = sum(gains)
            records.extend(
                FeatureImportanceRecord(
                    quantile=quantile,
                    expanded_feature_name=name,
                    source_feature_name=source,
                    gain=gain,
                    split_count=split_count,
                    normalized_gain=(gain / gain_total if gain_total > 0 else 0.0),
                )
                for name, source, gain, split_count in zip(names, sources, gains, splits)
            )
        return tuple(records)

    def source_feature_importance(self) -> tuple[SourceFeatureImportanceRecord, ...]:
        expanded = self.feature_importance()
        totals = {
            quantile: sum(record.gain for record in expanded if record.quantile == quantile)
            for quantile in self.quantiles
        }
        grouped: dict[tuple[float, str], tuple[float, int]] = {}
        for record in expanded:
            key = (record.quantile, record.source_feature_name)
            gain, split_count = grouped.get(key, (0.0, 0))
            grouped[key] = (gain + record.gain, split_count + record.split_count)
        return tuple(
            SourceFeatureImportanceRecord(
                quantile=quantile,
                source_feature_name=source,
                gain=gain,
                split_count=split_count,
                normalized_gain=(gain / totals[quantile] if totals[quantile] > 0 else 0.0),
            )
            for (quantile, source), (gain, split_count) in sorted(grouped.items())
        )


class LightGBMQuantileEstimator:
    estimator_id = "lightgbm_quantile"

    def __init__(
        self,
        *,
        quantiles: tuple[float, float, float] | list[float] | None = None,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 30,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
        min_data_in_leaf: int = 5,
        max_depth: int = -1,
        feature_fraction: float = 1.0,
        bagging_fraction: float = 1.0,
        bagging_freq: int = 0,
        lambda_l1: float = 0.0,
        lambda_l2: float = 0.0,
        max_bin: int = 255,
        extra_params: Mapping[str, Any] | None = None,
    ) -> None:
        normalized_quantiles = (
            tuple(float(value) for value in quantiles)
            if quantiles is not None
            else None
        )
        if normalized_quantiles is not None:
            quantiles_are_ordered = (
                len(normalized_quantiles) == 3
                and 0
                < normalized_quantiles[0]
                < normalized_quantiles[1]
                < normalized_quantiles[2]
                < 1
            )
            if (
                not quantiles_are_ordered
                or not math.isclose(normalized_quantiles[1], 0.5)
            ):
                raise ValueError(
                    "quantiles must contain ordered (lower, 0.5, upper) values"
                )
        if num_boost_round <= 0 or early_stopping_rounds <= 0:
            raise ValueError("boost rounds and early-stopping rounds must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if num_leaves < 2 or min_data_in_leaf <= 0:
            raise ValueError("num_leaves must be >= 2 and min_data_in_leaf must be positive")
        if not 0 < feature_fraction <= 1 or not 0 < bagging_fraction <= 1:
            raise ValueError("feature_fraction and bagging_fraction must be in (0, 1]")
        if bagging_freq < 0 or lambda_l1 < 0 or lambda_l2 < 0 or max_bin < 2:
            raise ValueError("bagging/regularization/bin parameters are invalid")
        additional = dict(extra_params or {})
        protected = sorted(_PROTECTED_PARAMS & set(additional))
        if protected:
            raise ValueError(f"extra_params may not override deterministic controls: {protected}")

        self.quantiles = normalized_quantiles
        self.num_boost_round = int(num_boost_round)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.learning_rate = float(learning_rate)
        self.num_leaves = int(num_leaves)
        self.min_data_in_leaf = int(min_data_in_leaf)
        self.max_depth = int(max_depth)
        self.feature_fraction = float(feature_fraction)
        self.bagging_fraction = float(bagging_fraction)
        self.bagging_freq = int(bagging_freq)
        self.lambda_l1 = float(lambda_l1)
        self.lambda_l2 = float(lambda_l2)
        self.max_bin = int(max_bin)
        self.extra_params = additional

    def _parameters(self, *, quantile: float, context: FitContext) -> dict[str, Any]:
        seed = _derived_seed(context.seed, context.fold, quantile)
        params: dict[str, Any] = {
            "objective": "quantile",
            "metric": "quantile",
            "alpha": quantile,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "max_depth": self.max_depth,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "max_bin": self.max_bin,
            "device_type": "cpu",
            "deterministic": True,
            "force_col_wise": True,
            "num_threads": 1,
            "seed": seed,
            "data_random_seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "drop_seed": seed,
            "verbosity": -1,
        }
        params.update(self.extra_params)
        return params

    def fit(
        self,
        train: TrainingView,
        validation: TrainingView,
        context: FitContext,
    ) -> FittedLightGBMQuantiles:
        expected_quantiles = (
            context.interval_alpha / 2,
            0.5,
            1 - context.interval_alpha / 2,
        )
        if self.quantiles is not None and any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
            for actual, expected in zip(self.quantiles, expected_quantiles)
        ):
            raise ValueError(
                f"configured quantiles {self.quantiles} do not match experiment "
                f"interval_alpha {context.interval_alpha}"
            )
        resolved_quantiles = self.quantiles or expected_quantiles
        lgb, np = _load_optional_dependencies()
        if train.dataset_id != validation.dataset_id:
            raise ValueError("train and validation views belong to different datasets")
        if train.position != validation.position or train.target != validation.target:
            raise ValueError("train and validation views must share position and target")
        train_conditions = _view_condition_ids(train, name="train")
        validation_conditions = _view_condition_ids(validation, name="validation")
        if train_conditions != validation_conditions:
            raise ValueError("train and validation views must share condition scope")

        train_points = tuple(example.point for example in train.examples)
        validation_points = tuple(example.point for example in validation.examples)
        encoder = FoldTabularEncoder.fit(train_points)
        if not encoder.schema.columns:
            raise ValueError("LightGBM requires at least one usable train-fold feature")
        encoded_train = encoder.transform(train_points)
        encoded_validation = encoder.transform(validation_points)
        train_y = np.asarray([example.target_value for example in train.examples], dtype=float)
        validation_y = np.asarray(
            [example.target_value for example in validation.examples], dtype=float
        )
        train_weight = np.asarray(
            [example.sample_weight for example in train.examples], dtype=float
        )
        validation_weight = np.asarray(
            [example.sample_weight for example in validation.examples], dtype=float
        )

        boosters: dict[float, Any] = {}
        best_iterations: dict[float, int] = {}
        reports: list[QuantileFitReport] = []
        for quantile in resolved_quantiles:
            params = self._parameters(quantile=quantile, context=context)
            train_set = lgb.Dataset(
                encoded_train.matrix,
                label=train_y,
                weight=train_weight,
                feature_name=list(encoded_train.feature_names),
                categorical_feature=list(encoded_train.categorical_indices),
                free_raw_data=False,
            )
            validation_set = lgb.Dataset(
                encoded_validation.matrix,
                label=validation_y,
                weight=validation_weight,
                feature_name=list(encoded_validation.feature_names),
                categorical_feature=list(encoded_validation.categorical_indices),
                reference=train_set,
                free_raw_data=False,
            )
            evaluation_history: dict[str, dict[str, list[float]]] = {}
            booster = lgb.train(
                params,
                train_set,
                num_boost_round=self.num_boost_round,
                valid_sets=[validation_set],
                valid_names=["validation"],
                callbacks=[
                    lgb.early_stopping(
                        self.early_stopping_rounds,
                        first_metric_only=True,
                        verbose=False,
                    ),
                    lgb.log_evaluation(period=0),
                    lgb.record_evaluation(evaluation_history),
                ],
            )
            history = tuple(
                float(value)
                for value in evaluation_history.get("validation", {}).get("quantile", ())
            )
            best_iteration = int(booster.best_iteration or len(history) or self.num_boost_round)
            if best_iteration <= 0:
                raise AssertionError("LightGBM did not produce a positive best iteration")
            best_loss = (
                history[min(best_iteration, len(history)) - 1]
                if history
                else float(booster.best_score["validation"]["quantile"])
            )
            boosters[quantile] = booster
            best_iterations[quantile] = best_iteration
            reports.append(
                QuantileFitReport(
                    quantile=quantile,
                    seed=int(params["seed"]),
                    best_iteration=best_iteration,
                    best_validation_loss=best_loss,
                    validation_history=history,
                    parameters=params,
                )
            )

        fit_report = LightGBMFitReport(
            estimator_version=LIGHTGBM_ESTIMATOR_VERSION,
            encoder_schema_hash=encoder.schema.content_hash,
            train_point_hash=_point_hash(tuple(point.point_id for point in train_points)),
            validation_point_hash=_point_hash(
                tuple(point.point_id for point in validation_points)
            ),
            train_point_count=len(train_points),
            validation_point_count=len(validation_points),
            lightgbm_version=str(lgb.__version__),
            numpy_version=str(np.__version__),
            platform=platform.platform(),
            quantiles=tuple(reports),
        )
        return FittedLightGBMQuantiles(
            estimator_id=self.estimator_id,
            target=train.target,
            position=train.position,
            dataset_id=train.dataset_id,
            allowed_condition_ids=train_conditions,
            encoder=encoder,
            boosters=boosters,
            best_iterations=best_iterations,
            quantiles=resolved_quantiles,
            fit_report=fit_report,
        )


__all__ = [
    "EncoderSchema",
    "FeatureImportanceRecord",
    "FittedLightGBMQuantiles",
    "LIGHTGBM_ESTIMATOR_VERSION",
    "LightGBMFitReport",
    "LightGBMQuantileEstimator",
    "LightGBMQuantileSession",
    "OptionalEstimatorDependencyError",
    "QuantileFitReport",
    "RawQuantileDiagnostics",
    "SourceFeatureImportanceRecord",
]
