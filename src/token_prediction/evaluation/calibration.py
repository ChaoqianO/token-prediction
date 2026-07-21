from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol, Sequence

from token_prediction.estimators import TokenForecast


@dataclass(frozen=True)
class CalibrationExample:
    task_id: str
    forecast: TokenForecast
    target_value: float


class FittedCalibrator(Protocol):
    calibrator_id: str

    def transform(self, forecast: TokenForecast) -> TokenForecast: ...


class IntervalCalibrator(Protocol):
    calibrator_id: str

    def fit(self, examples: Sequence[CalibrationExample]) -> FittedCalibrator: ...


@dataclass(frozen=True)
class _ExpansionCalibrator:
    calibrator_id: str
    expansion: float

    def transform(self, forecast: TokenForecast) -> TokenForecast:
        raw_lower = (
            forecast.raw_lower if forecast.raw_lower is not None else forecast.lower
        )
        raw_point = (
            forecast.raw_point if forecast.raw_point is not None else forecast.point
        )
        raw_upper = (
            forecast.raw_upper if forecast.raw_upper is not None else forecast.upper
        )
        return TokenForecast(
            point_id=forecast.point_id,
            target=forecast.target,
            lower=max(0.0, forecast.lower - self.expansion),
            point=forecast.point,
            upper=max(forecast.point, forecast.upper + self.expansion),
            latency_ms=forecast.latency_ms,
            overhead_input_tokens=forecast.overhead_input_tokens,
            overhead_output_tokens=forecast.overhead_output_tokens,
            raw_lower=raw_lower,
            raw_point=raw_point,
            raw_upper=raw_upper,
        )


class TaskMaxConformalCalibrator:
    """Conservative split conformal calibration using one score per task."""

    calibrator_id = "task_max_conformal"

    def __init__(self, *, alpha: float = 0.10) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha

    def fit(self, examples: Sequence[CalibrationExample]) -> _ExpansionCalibrator:
        if not examples:
            raise ValueError("calibration examples are empty")
        by_task: dict[str, list[float]] = defaultdict(list)
        for example in examples:
            y = float(example.target_value)
            forecast = example.forecast
            score = max(forecast.lower - y, y - forecast.upper, 0.0)
            by_task[example.task_id].append(score)
        task_scores = sorted(max(scores) for scores in by_task.values())
        rank = min(
            len(task_scores),
            max(1, math.ceil((len(task_scores) + 1) * (1 - self.alpha))),
        )
        return _ExpansionCalibrator(self.calibrator_id, task_scores[rank - 1])


@dataclass(frozen=True)
class IdentityCalibrator:
    calibrator_id: str = "none"

    def fit(self, examples: Sequence[CalibrationExample]) -> _ExpansionCalibrator:
        del examples
        return _ExpansionCalibrator(self.calibrator_id, 0.0)
