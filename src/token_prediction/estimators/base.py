from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Mapping, Protocol

from token_prediction.dataset import PredictionPoint, PredictionPosition, PredictionTarget


@dataclass(frozen=True)
class TrainingExample:
    point: PredictionPoint
    target_value: float
    sample_weight: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.target_value) or self.target_value < 0:
            raise ValueError("training target must be finite and non-negative")
        if not math.isfinite(self.sample_weight) or self.sample_weight <= 0:
            raise ValueError("sample weight must be finite and positive")


@dataclass(frozen=True)
class TrainingView:
    dataset_id: str
    position: PredictionPosition
    target: PredictionTarget
    examples: tuple[TrainingExample, ...]

    def __post_init__(self) -> None:
        if not self.examples:
            raise ValueError("training view is empty")
        point_ids = [example.point.point_id for example in self.examples]
        if len(set(point_ids)) != len(point_ids):
            raise ValueError("training point ids must be unique")

    def sequences(self) -> tuple[tuple[TrainingExample, ...], ...]:
        """Return trajectory-local examples in online visibility order."""

        grouped: dict[str, list[TrainingExample]] = defaultdict(list)
        for example in self.examples:
            grouped[example.point.trajectory_id].append(example)
        return tuple(
            tuple(
                sorted(
                    grouped[trajectory_id],
                    key=lambda item: (
                        item.point.cutoff_event_seq,
                        item.point.point_id,
                    ),
                )
            )
            for trajectory_id in sorted(grouped)
        )


@dataclass(frozen=True)
class FitContext:
    seed: int
    fold: int
    interval_alpha: float = 0.10

    def __post_init__(self) -> None:
        if not math.isfinite(self.interval_alpha) or not 0 < self.interval_alpha < 1:
            raise ValueError("interval_alpha must be finite and in (0, 1)")


@dataclass(frozen=True)
class RunContext:
    task_id: str
    trajectory_id: str
    run_id: str


@dataclass(frozen=True)
class ObservedTransition:
    from_point_id: str
    to_point_id: str
    observed_spend_tokens: int | None

    def __post_init__(self) -> None:
        if self.observed_spend_tokens is not None and self.observed_spend_tokens < 0:
            raise ValueError("observed spend must be non-negative or missing")


@dataclass(frozen=True)
class TokenForecast:
    point_id: str
    target: PredictionTarget
    lower: float
    point: float
    upper: float
    latency_ms: float = 0.0
    overhead_input_tokens: int = 0
    overhead_output_tokens: int = 0
    raw_lower: float | None = None
    raw_point: float | None = None
    raw_upper: float | None = None

    def __post_init__(self) -> None:
        values = (self.lower, self.point, self.upper, self.latency_ms)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("forecast values must be finite")
        if not 0 <= self.lower <= self.point <= self.upper:
            raise ValueError("forecast must satisfy 0 <= lower <= point <= upper")
        if self.overhead_input_tokens < 0 or self.overhead_output_tokens < 0:
            raise ValueError("prediction overhead must be non-negative")
        raw = (self.raw_lower, self.raw_point, self.raw_upper)
        if any(value is not None for value in raw) and not all(
            value is not None for value in raw
        ):
            raise ValueError("raw forecast values must be provided together or omitted together")
        if all(value is not None for value in raw) and any(
            not math.isfinite(float(value)) for value in raw
        ):
            raise ValueError("raw forecast values must be finite")

    def with_latency(self, latency_ms: float) -> "TokenForecast":
        return replace(self, latency_ms=latency_ms)


class PredictionSession(Protocol):
    def predict(self, point: PredictionPoint) -> TokenForecast: ...

    def observe(self, transition: ObservedTransition) -> None: ...


class FittedEstimator(Protocol):
    estimator_id: str

    def start(self, context: RunContext) -> PredictionSession: ...


class TokenEstimator(Protocol):
    estimator_id: str

    def fit(
        self,
        train: TrainingView,
        validation: TrainingView,
        context: FitContext,
    ) -> FittedEstimator: ...


EstimatorParams = Mapping[str, Any]


class EstimatorFactory(Protocol):
    def __call__(self, params: EstimatorParams) -> TokenEstimator: ...
