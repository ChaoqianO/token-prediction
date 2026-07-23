from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from token_prediction.contracts import SourceCapabilities
from token_prediction.dataset import PredictionPoint
from token_prediction.estimators import (
    FittedEstimator,
    ObservedTransition,
    SessionSeed,
    TokenForecast,
)
from token_prediction.lifecycle import LifecycleSessionDriver
from token_prediction.telemetry import (
    TelemetryDecision,
    TelemetrySurface,
    require_telemetry_surface,
)


@dataclass(frozen=True)
class ShadowPrediction:
    ordinal: int
    point_id: str
    forecast: TokenForecast
    transition: ObservedTransition

    def __post_init__(self) -> None:
        if self.ordinal < 1:
            raise ValueError("shadow prediction ordinal must be positive")
        if self.point_id != self.forecast.point_id:
            raise ValueError("shadow prediction point and forecast differ")
        if self.point_id != self.transition.to_point_id:
            raise ValueError("shadow prediction point and transition differ")


class OnlineShadowSession:
    """Label-free incremental adapter over the production lifecycle driver."""

    def __init__(
        self,
        fitted: FittedEstimator,
        *,
        capabilities: SourceCapabilities,
        dataset_id: str,
        input_contract_hash: str,
        condition_id: str,
        task_pre_point: PredictionPoint,
        seed: SessionSeed,
        select_point: Callable[[PredictionPoint], PredictionPoint] | None = None,
        sink: Callable[[ShadowPrediction], None] | None = None,
    ) -> None:
        self.capability_decision: TelemetryDecision = require_telemetry_surface(
            capabilities,
            TelemetrySurface.ONLINE_SHADOW,
        )
        self._driver = LifecycleSessionDriver(
            fitted,
            dataset_id=dataset_id,
            input_contract_hash=input_contract_hash,
            condition_id=condition_id,
            task_pre_point=task_pre_point,
            seed=seed,
            runtime_mode="shadow",
            select_point=select_point,
        )
        self._sink = sink
        self._ordinal = 0

    def observe_boundary(self, point: PredictionPoint) -> ShadowPrediction:
        forecast, transition = self._driver.advance(point)
        self._ordinal += 1
        result = ShadowPrediction(
            ordinal=self._ordinal,
            point_id=point.point_id,
            forecast=forecast,
            transition=transition,
        )
        if self._sink is not None:
            self._sink(result)
        return result


__all__ = [
    "OnlineShadowSession",
    "ShadowPrediction",
]
