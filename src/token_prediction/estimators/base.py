from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping, Protocol

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
    lifecycle_sequences: tuple[Any, ...] | None = None
    input_contract_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.examples:
            raise ValueError("training view is empty")
        point_ids = [example.point.point_id for example in self.examples]
        if len(set(point_ids)) != len(point_ids):
            raise ValueError("training point ids must be unique")
        if self.lifecycle_sequences is not None and not isinstance(self.lifecycle_sequences, tuple):
            raise TypeError("lifecycle_sequences must be a tuple or None")
        if self.input_contract_hash is not None:
            _require_sha256(self.input_contract_hash, name="input_contract_hash")

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


class FitCheckpoint(Protocol):
    """Atomic, non-pickle persistence supplied to long neural fits."""

    def load(self, identity: Mapping[str, Any]) -> Mapping[str, bytes] | None: ...

    def save(
        self,
        identity: Mapping[str, Any],
        *,
        epoch: int,
        files: Mapping[str, bytes],
    ) -> None: ...

    def clear(self) -> None: ...


@dataclass(frozen=True)
class FitContext:
    seed: int
    fold: int
    interval_alpha: float = 0.10
    checkpoint: FitCheckpoint | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.interval_alpha) or not 0 < self.interval_alpha < 1:
            raise ValueError("interval_alpha must be finite and in (0, 1)")
        if self.checkpoint is not None and not all(
            callable(getattr(self.checkpoint, name, None)) for name in ("load", "save", "clear")
        ):
            raise TypeError("checkpoint must implement load/save/clear")


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
        if any(value is not None for value in raw) and not all(value is not None for value in raw):
            raise ValueError("raw forecast values must be provided together or omitted together")
        if all(value is not None for value in raw) and any(
            not math.isfinite(float(value)) for value in raw
        ):
            raise ValueError("raw forecast values must be finite")

    def with_latency(self, latency_ms: float) -> "TokenForecast":
        return replace(self, latency_ms=latency_ms)


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _require_identifier(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")


def _require_sha256(value: str, *, name: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class SessionSeed:
    """Strict, label-free provenance for a cross-position Task session.

    ``forecast`` is the initializer's *uncalibrated* forecast after the common
    non-negative/ordered repair.  Its raw values are mandatory so that a caller
    cannot accidentally feed a conformalized interval back into the lifecycle.
    """

    task_pre_point: PredictionPoint
    forecast: TokenForecast
    initializer_id: str
    initializer_hash: str
    inner_split_id: str
    component_bundle_hashes: tuple[str, ...]
    seed_policy_id: str
    seed_policy_hash: str

    def __post_init__(self) -> None:
        point = self.task_pre_point
        forecast = self.forecast
        if point.position != PredictionPosition.TASK_PRE:
            raise ValueError("session seed point must be at the Task-pre position")
        if point.target != PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS:
            raise ValueError(
                "session seed point must target task_provider_accounted_remaining_tokens"
            )
        if forecast.point_id != point.point_id:
            raise ValueError("session seed forecast point_id does not match Task-pre point")
        if forecast.target != point.target:
            raise ValueError("session seed forecast target does not match Task-pre point")
        if forecast.latency_ms < 0:
            raise ValueError("session seed forecast latency must be non-negative")

        raw = (forecast.raw_lower, forecast.raw_point, forecast.raw_upper)
        if not all(value is not None for value in raw):
            raise ValueError(
                "session seed forecast requires raw quantiles to prove uncalibrated repair"
            )
        raw_lower, raw_point, raw_upper = (float(value) for value in raw)
        repaired_point = max(0.0, raw_point)
        repaired_lower = min(max(0.0, raw_lower), repaired_point)
        repaired_upper = max(max(0.0, raw_upper), repaired_point)
        repaired = (forecast.lower, forecast.point, forecast.upper)
        expected = (repaired_lower, repaired_point, repaired_upper)
        if any(
            not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1e-12)
            for actual, wanted in zip(repaired, expected)
        ):
            raise ValueError(
                "session seed forecast is not the non-negative/ordered repair of its raw values"
            )

        for name in ("initializer_id", "inner_split_id", "seed_policy_id"):
            _require_identifier(getattr(self, name), name=name)
        _require_sha256(self.initializer_hash, name="initializer_hash")
        _require_sha256(self.seed_policy_hash, name="seed_policy_hash")
        if not isinstance(self.component_bundle_hashes, tuple):
            raise TypeError("component_bundle_hashes must be a tuple")
        if not self.component_bundle_hashes:
            raise ValueError("component_bundle_hashes must not be empty")
        for index, digest in enumerate(self.component_bundle_hashes):
            _require_sha256(digest, name=f"component_bundle_hashes[{index}]")

    @property
    def content_hash(self) -> str:
        """Return a stable identity without serializing point feature values as labels."""

        payload = {
            "task_pre_point_id": self.task_pre_point.point_id,
            "source_event_id": self.task_pre_point.source_event_id,
            "task_id": self.task_pre_point.task_id,
            "trajectory_id": self.task_pre_point.trajectory_id,
            "run_id": self.task_pre_point.run_id,
            "prediction_context_id": self.task_pre_point.prediction_context_id,
            "condition_id": self.task_pre_point.condition_id,
            "logical_call_id": self.task_pre_point.logical_call_id,
            "attempt_id": self.task_pre_point.attempt_id,
            "cutoff_event_seq": self.task_pre_point.cutoff_event_seq,
            "position": self.task_pre_point.position.value,
            "target": self.task_pre_point.target.value,
            "features": dict(self.task_pre_point.features),
            "known_offset_tokens": self.task_pre_point.known_offset_tokens,
            "forecast": {
                "raw_lower": self.forecast.raw_lower,
                "raw_point": self.forecast.raw_point,
                "raw_upper": self.forecast.raw_upper,
                "lower": self.forecast.lower,
                "point": self.forecast.point,
                "upper": self.forecast.upper,
                "latency_ms": self.forecast.latency_ms,
                "overhead_input_tokens": self.forecast.overhead_input_tokens,
                "overhead_output_tokens": self.forecast.overhead_output_tokens,
            },
            "initializer_id": self.initializer_id,
            "initializer_hash": self.initializer_hash,
            "inner_split_id": self.inner_split_id,
            "component_bundle_hashes": self.component_bundle_hashes,
            "seed_policy_id": self.seed_policy_id,
            "seed_policy_hash": self.seed_policy_hash,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RunContext:
    task_id: str
    trajectory_id: str
    run_id: str
    dataset_id: str | None = None
    condition_id: str | None = None
    target: PredictionTarget | None = None
    runtime_mode: Literal["offline", "shadow"] = "offline"
    input_contract_hash: str | None = None
    session_seed: SessionSeed | None = None

    def __post_init__(self) -> None:
        for name in ("task_id", "trajectory_id", "run_id"):
            _require_identifier(getattr(self, name), name=name)
        if self.dataset_id is not None:
            _require_identifier(self.dataset_id, name="dataset_id")
        if self.condition_id is not None:
            _require_identifier(self.condition_id, name="condition_id")
        if self.target is not None and not isinstance(self.target, PredictionTarget):
            raise TypeError("target must be a PredictionTarget or None")
        if self.runtime_mode not in {"offline", "shadow"}:
            raise ValueError("runtime_mode must be 'offline' or 'shadow'")
        if self.input_contract_hash is not None:
            _require_sha256(self.input_contract_hash, name="input_contract_hash")


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
