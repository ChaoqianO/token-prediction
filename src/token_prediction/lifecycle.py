from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Literal, Mapping, Sequence

from token_prediction.dataset import LifecycleSequence, LifecycleStep, PredictionPoint
from token_prediction.estimators import (
    FittedEstimator,
    ObservedTransition,
    RunContext,
    SessionSeed,
    TokenForecast,
)


RuntimeMode = Literal["offline", "shadow"]


def visible_spend_delta(
    previous: PredictionPoint,
    current: PredictionPoint,
) -> int | None:
    """Return a visible provider-spend delta without poisoning later recovery."""

    if previous.trajectory_id != current.trajectory_id:
        raise ValueError("lifecycle transition crossed trajectories")
    if current.cutoff_event_seq <= previous.cutoff_event_seq:
        raise ValueError("lifecycle transition did not advance visibility")
    previous_missing = previous.features.get("missing_usage_attempts")
    current_missing = current.features.get("missing_usage_attempts")
    if (
        not isinstance(previous_missing, int)
        or isinstance(previous_missing, bool)
        or not isinstance(current_missing, int)
        or isinstance(current_missing, bool)
    ):
        return None
    if previous_missing < 0 or current_missing < 0:
        raise ValueError("missing usage attempt count must be non-negative")
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
    values = (*previous_values, *current_values)
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return None
    spend = sum(int(value) for value in current_values) - sum(
        int(value) for value in previous_values
    )
    if spend < 0:
        raise ValueError("cumulative provider spend decreased within a trajectory")
    return spend


@dataclass(frozen=True)
class LifecyclePrediction:
    step: LifecycleStep
    forecast: TokenForecast
    transition: ObservedTransition

    def __post_init__(self) -> None:
        if self.forecast.point_id != self.step.point.point_id:
            raise ValueError("lifecycle forecast point identity is inconsistent")
        if self.forecast.target != self.step.point.target:
            raise ValueError("lifecycle forecast target is inconsistent")
        if self.transition.to_point_id != self.step.point.point_id:
            raise ValueError("lifecycle transition and prediction point differ")


@dataclass(frozen=True)
class LifecycleRun:
    sequence: LifecycleSequence
    runtime_mode: RuntimeMode
    seed: SessionSeed
    predictions: tuple[LifecyclePrediction, ...]

    def __post_init__(self) -> None:
        if self.runtime_mode not in {"offline", "shadow"}:
            raise ValueError("unsupported lifecycle runtime mode")
        expected = tuple(step.point.point_id for step in self.sequence.steps[1:])
        actual = tuple(item.step.point.point_id for item in self.predictions)
        if actual != expected:
            raise ValueError("lifecycle run did not predict every Task-update boundary")

    @property
    def by_point_id(self) -> Mapping[str, LifecyclePrediction]:
        return MappingProxyType(
            {prediction.step.point.point_id: prediction for prediction in self.predictions}
        )

    @property
    def scored_predictions(self) -> tuple[LifecyclePrediction, ...]:
        return tuple(
            prediction for prediction in self.predictions if prediction.step.score_mask
        )

    @property
    def loss_predictions(self) -> tuple[LifecyclePrediction, ...]:
        return tuple(
            prediction for prediction in self.predictions if prediction.step.loss_mask
        )


def run_lifecycle_sequence(
    fitted: FittedEstimator,
    sequence: LifecycleSequence,
    seed: SessionSeed,
    *,
    runtime_mode: RuntimeMode = "offline",
    select_point: Callable[[PredictionPoint], PredictionPoint] | None = None,
) -> LifecycleRun:
    """Drive one sequence through the same observe→predict order in both modes."""

    if runtime_mode not in {"offline", "shadow"}:
        raise ValueError("runtime_mode must be 'offline' or 'shadow'")
    first = sequence.steps[0]
    if first.point.point_id != seed.task_pre_point.point_id:
        raise ValueError("lifecycle sequence Task-pre point does not match its seed")
    if first.loss_mask or first.score_mask or first.sample_weight != 0:
        raise ValueError("Task-pre must be unscored updater context")
    selector = select_point or (lambda point: point)
    session = fitted.start(
        RunContext(
            sequence.task_id,
            sequence.trajectory_id,
            sequence.run_id,
            dataset_id=sequence.dataset_id,
            condition_id=sequence.condition_id,
            target=sequence.target,
            runtime_mode=runtime_mode,
            input_contract_hash=sequence.input_contract_hash,
            session_seed=seed,
        )
    )
    predictions: list[LifecyclePrediction] = []
    previous = first.point
    for step in sequence.steps[1:]:
        selected = selector(step.point)
        if (
            selected.point_id != step.point.point_id
            or selected.task_id != step.point.task_id
            or selected.trajectory_id != step.point.trajectory_id
            or selected.position != step.point.position
            or selected.target != step.point.target
        ):
            raise ValueError("feature selection changed lifecycle point identity")
        transition = ObservedTransition(
            from_point_id=previous.point_id,
            to_point_id=selected.point_id,
            observed_spend_tokens=visible_spend_delta(previous, step.point),
        )
        session.observe(transition)
        forecast = session.predict(selected)
        predictions.append(LifecyclePrediction(step, forecast, transition))
        previous = step.point
    return LifecycleRun(sequence, runtime_mode, seed, tuple(predictions))


def run_lifecycle_batch(
    fitted: FittedEstimator,
    sequences: Sequence[LifecycleSequence],
    seeds: Mapping[str, SessionSeed],
    *,
    runtime_mode: RuntimeMode = "offline",
    select_point: Callable[[PredictionPoint], PredictionPoint] | None = None,
) -> tuple[LifecycleRun, ...]:
    if not sequences:
        raise ValueError("lifecycle batch is empty")
    expected = {sequence.steps[0].point.point_id for sequence in sequences}
    if set(seeds) != expected:
        raise ValueError("lifecycle seed set does not exactly match Task-pre points")
    return tuple(
        run_lifecycle_sequence(
            fitted,
            sequence,
            seeds[sequence.steps[0].point.point_id],
            runtime_mode=runtime_mode,
            select_point=select_point,
        )
        for sequence in sorted(
            sequences,
            key=lambda item: (item.task_id, item.run_id, item.trajectory_id),
        )
    )


__all__ = [
    "LifecyclePrediction",
    "LifecycleRun",
    "RuntimeMode",
    "run_lifecycle_batch",
    "run_lifecycle_sequence",
    "visible_spend_delta",
]
