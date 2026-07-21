from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import EstimatorFactory, TokenEstimator
from .baselines import DirectFeatureEstimator, EmpiricalQuantileEstimator, LengthOnlyEstimator
from .deduct import DeductOnlyEstimator
from .cross_position_deduct import CrossPositionDeductEstimator
from .gru import GRUResidualEstimator
from .mlp import IndependentMLPQuantileEstimator


def _lightgbm_quantile_factory(params: Mapping[str, Any]) -> TokenEstimator:
    # Keep the core, zero-dependency baseline path importable without the optional
    # estimator extra. The implementation itself imports LightGBM only at fit time.
    from .lightgbm import LightGBMQuantileEstimator

    return LightGBMQuantileEstimator(**dict(params))


class EstimatorRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, EstimatorFactory] = {}

    def register(self, estimator_id: str, factory: EstimatorFactory) -> None:
        key = str(estimator_id).strip()
        if not key:
            raise ValueError("estimator_id is required")
        if key in self._factories:
            raise ValueError(f"estimator {key!r} is already registered")
        self._factories[key] = factory

    def create(self, estimator_id: str, params: Mapping[str, Any] | None = None) -> TokenEstimator:
        try:
            factory = self._factories[estimator_id]
        except KeyError as exc:
            raise KeyError(f"unknown estimator {estimator_id!r}") from exc
        estimator = factory(dict(params or {}))
        if estimator.estimator_id != estimator_id:
            raise ValueError("estimator factory returned a mismatched estimator_id")
        return estimator

    @property
    def estimator_ids(self) -> frozenset[str]:
        return frozenset(self._factories)


def builtin_registry() -> EstimatorRegistry:
    registry = EstimatorRegistry()
    registry.register(
        "empirical_quantile",
        lambda params: EmpiricalQuantileEstimator(**params),
    )
    registry.register(
        "length_only",
        lambda params: LengthOnlyEstimator(**params),
    )
    registry.register(
        "direct_feature",
        lambda params: DirectFeatureEstimator(**params),
    )
    registry.register(
        "deduct_only",
        lambda params: DeductOnlyEstimator(**params),
    )
    registry.register(
        "cross_position_deduct",
        lambda params: CrossPositionDeductEstimator(**params),
    )
    registry.register("lightgbm_quantile", _lightgbm_quantile_factory)
    registry.register(
        "independent_mlp",
        lambda params: IndependentMLPQuantileEstimator(**params),
    )
    registry.register(
        "gru_residual",
        lambda params: GRUResidualEstimator(**params),
    )
    return registry
