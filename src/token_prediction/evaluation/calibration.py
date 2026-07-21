from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from token_prediction.estimators import TokenForecast


CALIBRATOR_SCHEMA_VERSION = 1


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
class FittedExpansionCalibrator:
    calibrator_id: str
    interval_alpha: float
    expansion: float

    def __post_init__(self) -> None:
        if self.calibrator_id not in {"task_max_conformal", "none"}:
            raise ValueError("unsupported fitted calibrator id")
        if not math.isfinite(self.interval_alpha) or not 0 < self.interval_alpha < 1:
            raise ValueError("calibrator alpha must be finite and in (0, 1)")
        if not math.isfinite(self.expansion) or self.expansion < 0:
            raise ValueError("calibrator expansion must be finite and non-negative")
        if self.calibrator_id == "none" and self.expansion != 0.0:
            raise ValueError("identity calibrator expansion must be exactly zero")

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "calibrator_schema_version": CALIBRATOR_SCHEMA_VERSION,
            "calibrator_id": self.calibrator_id,
            "interval_alpha": self.interval_alpha,
            "expansion": self.expansion,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FittedExpansionCalibrator":
        expected = {
            "calibrator_schema_version",
            "calibrator_id",
            "interval_alpha",
            "expansion",
        }
        if set(value) != expected:
            raise ValueError("fitted calibrator has missing or extra fields")
        version = value.get("calibrator_schema_version")
        if isinstance(version, bool) or version != CALIBRATOR_SCHEMA_VERSION:
            raise ValueError("unsupported fitted calibrator schema version")
        calibrator_id = value.get("calibrator_id")
        if not isinstance(calibrator_id, str):
            raise TypeError("fitted calibrator id must be a string")
        alpha = value.get("interval_alpha")
        expansion = value.get("expansion")
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or isinstance(expansion, bool)
            or not isinstance(expansion, (int, float))
        ):
            raise TypeError("fitted calibrator numeric fields must be JSON numbers")
        return cls(calibrator_id, float(alpha), float(expansion))

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

    def fit(self, examples: Sequence[CalibrationExample]) -> FittedExpansionCalibrator:
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
        return FittedExpansionCalibrator(
            self.calibrator_id,
            self.alpha,
            task_scores[rank - 1],
        )


@dataclass(frozen=True)
class IdentityCalibrator:
    calibrator_id: str = "none"
    alpha: float = 0.10

    def __post_init__(self) -> None:
        if self.calibrator_id != "none":
            raise ValueError("identity calibrator id must be 'none'")
        if not math.isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise ValueError("identity calibrator alpha must be finite and in (0, 1)")

    def fit(self, examples: Sequence[CalibrationExample]) -> FittedExpansionCalibrator:
        del examples
        return FittedExpansionCalibrator(self.calibrator_id, self.alpha, 0.0)
