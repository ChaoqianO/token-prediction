from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from token_prediction.dataset import PredictionPoint, PredictionPosition, PredictionTarget

from .base import (
    FitContext,
    ObservedTransition,
    RunContext,
    SessionSeed,
    TokenForecast,
    TrainingView,
)


_INPUT_CONTRACT = {
    "estimator_id": "cross_position_deduct",
    "version": 1,
    "seed_position": PredictionPosition.TASK_PRE.value,
    "update_position": PredictionPosition.TASK_UPDATE.value,
    "target": PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS.value,
    "point_operands": ["known_offset_tokens", "features.missing_usage_attempts"],
    "transition_operands": ["observed_spend_tokens"],
}
CROSS_POSITION_INPUT_CONTRACT_HASH = hashlib.sha256(
    json.dumps(_INPUT_CONTRACT, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def _condition_id(view: TrainingView, *, name: str) -> str:
    conditions: set[str] = set()
    for example in view.examples:
        point = example.point
        if point.position != view.position or point.target != view.target:
            raise ValueError(f"{name} point does not match its TrainingView cell")
        conditions.add(point.condition_id)
    if len(conditions) != 1:
        raise ValueError(f"{name} view must contain exactly one condition_id")
    return next(iter(conditions))


def _lifecycle_input_contract_hash(view: TrainingView, *, name: str) -> str | None:
    if view.lifecycle_sequences is None:
        return None
    hashes: set[str] = set()
    for sequence in view.lifecycle_sequences:
        dataset_id = getattr(sequence, "dataset_id", None)
        condition_id = getattr(sequence, "condition_id", None)
        target = getattr(sequence, "target", None)
        contract_hash = getattr(sequence, "input_contract_hash", None)
        if dataset_id != view.dataset_id:
            raise ValueError(f"{name} lifecycle sequence belongs to another dataset")
        if target != view.target:
            raise ValueError(f"{name} lifecycle sequence target does not match its view")
        if condition_id not in {
            example.point.condition_id for example in view.examples
        }:
            raise ValueError(f"{name} lifecycle sequence condition does not match its view")
        if not isinstance(contract_hash, str) or (
            len(contract_hash) != 64
            or any(character not in "0123456789abcdef" for character in contract_hash)
        ):
            raise ValueError(f"{name} lifecycle input_contract_hash is invalid")
        hashes.add(contract_hash)
    if len(hashes) != 1:
        raise ValueError(f"{name} lifecycle sequences must share one input contract")
    return next(iter(hashes))


def _missing_usage_counter(point: PredictionPoint) -> int | None:
    value = point.features.get("missing_usage_attempts")
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("missing_usage_attempts must be a non-negative integer or missing")
    return value


@dataclass
class CrossPositionDeductSession:
    """Carry an uncalibrated Task-pre forecast through real request boundaries."""

    estimator_id: str
    target: PredictionTarget
    condition_id: str
    context: RunContext
    seed: SessionSeed
    fallback_count: int = 0
    last_fallback_reason: str | None = None
    _last_point: PredictionPoint | None = field(init=False, default=None)
    _last_forecast: TokenForecast | None = field(init=False, default=None)
    _pending_transition: ObservedTransition | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._last_point = self.seed.task_pre_point
        self._last_forecast = self.seed.forecast

    def _validate_point(self, point: PredictionPoint) -> None:
        if point.task_id != self.context.task_id:
            raise ValueError("prediction point task_id does not match the session")
        if point.trajectory_id != self.context.trajectory_id:
            raise ValueError("prediction point trajectory_id does not match the session")
        if point.run_id != self.context.run_id:
            raise ValueError("prediction point run_id does not match the session")
        if point.condition_id != self.condition_id:
            raise ValueError("prediction point condition_id does not match the fitted scope")
        if point.position != PredictionPosition.TASK_UPDATE:
            raise ValueError("cross_position_deduct predicts only Task-update points")
        if point.target != self.target:
            raise ValueError("prediction point target does not match the fitted scope")

    def observe(self, transition: ObservedTransition) -> None:
        if self._last_point is None:
            raise RuntimeError("cross-position session has no Task-pre seed point")
        if self._pending_transition is not None:
            raise RuntimeError("the previous transition has not been consumed by predict")
        if transition.from_point_id != self._last_point.point_id:
            raise ValueError("transition from_point_id does not match the previous point")
        if not transition.to_point_id or transition.to_point_id == transition.from_point_id:
            raise ValueError("transition must advance to a different non-empty point_id")
        self._pending_transition = transition

    def predict(self, point: PredictionPoint) -> TokenForecast:
        self._validate_point(point)
        previous_point = self._last_point
        previous_forecast = self._last_forecast
        if previous_point is None or previous_forecast is None:
            raise RuntimeError("cross-position session is missing its Task-pre seed")
        transition = self._pending_transition
        if transition is None:
            raise RuntimeError("observe must be called before each Task-update prediction")
        if transition.to_point_id != point.point_id:
            raise ValueError("transition to_point_id does not match the prediction point")
        if point.cutoff_event_seq <= previous_point.cutoff_event_seq:
            raise ValueError("prediction points must advance in visibility order")

        previous_counter = _missing_usage_counter(previous_point)
        current_counter = _missing_usage_counter(point)
        counter_increased = False
        counter_unavailable = (previous_counter is None) != (current_counter is None)
        if previous_counter is not None and current_counter is not None:
            if current_counter < previous_counter:
                raise ValueError("missing usage attempt count decreased within a trajectory")
            counter_increased = current_counter > previous_counter

        missing: list[str] = []
        if previous_point.known_offset_tokens is None:
            missing.append("previous_offset")
        if transition.observed_spend_tokens is None:
            missing.append("observed_spend")
        if point.known_offset_tokens is None:
            missing.append("current_offset")
        if counter_increased:
            missing.append("usage_counter_increased")
        elif counter_unavailable:
            missing.append("usage_counter")

        if missing:
            # The newly missing attempt is never treated as zero.  Advancing the
            # stored point lets a later equal counter resume with a visible delta.
            raw_lower = previous_forecast.lower
            raw_point = previous_forecast.point
            raw_upper = previous_forecast.upper
            self.fallback_count += 1
            self.last_fallback_reason = "missing_" + "_and_".join(missing)
        else:
            delta = (
                int(previous_point.known_offset_tokens)
                - int(transition.observed_spend_tokens)
                - int(point.known_offset_tokens)
            )
            raw_lower = previous_forecast.lower + delta
            raw_point = previous_forecast.point + delta
            raw_upper = previous_forecast.upper + delta
            self.last_fallback_reason = None

        center = max(0.0, raw_point)
        lower = min(max(0.0, raw_lower), center)
        upper = max(max(0.0, raw_upper), center)
        forecast = TokenForecast(
            point_id=point.point_id,
            target=self.target,
            lower=lower,
            point=center,
            upper=upper,
            raw_lower=raw_lower,
            raw_point=raw_point,
            raw_upper=raw_upper,
        )
        self._last_point = point
        self._last_forecast = forecast
        self._pending_transition = None
        return forecast


@dataclass(frozen=True)
class FittedCrossPositionDeduct:
    estimator_id: str
    dataset_id: str
    target: PredictionTarget
    condition_id: str
    input_contract_hash: str

    def start(self, context: RunContext) -> CrossPositionDeductSession:
        if context.dataset_id is None:
            raise ValueError("cross_position_deduct requires RunContext.dataset_id")
        if context.dataset_id != self.dataset_id:
            raise ValueError("RunContext dataset_id does not match the fitted dataset")
        if context.condition_id is None:
            raise ValueError("cross_position_deduct requires RunContext.condition_id")
        if context.condition_id != self.condition_id:
            raise ValueError("RunContext condition_id does not match the fitted scope")
        if context.target is None:
            raise ValueError("cross_position_deduct requires RunContext.target")
        if context.target != self.target:
            raise ValueError("RunContext target does not match the fitted scope")
        if context.input_contract_hash is None:
            raise ValueError("cross_position_deduct requires RunContext.input_contract_hash")
        if context.input_contract_hash != self.input_contract_hash:
            raise ValueError("RunContext input_contract_hash does not match the fitted contract")
        seed = context.session_seed
        if seed is None:
            raise ValueError("cross_position_deduct requires RunContext.session_seed")

        point = seed.task_pre_point
        for name in ("task_id", "trajectory_id", "run_id"):
            if getattr(point, name) != getattr(context, name):
                raise ValueError(f"session seed {name} does not match RunContext")
        if point.condition_id != self.condition_id:
            raise ValueError("session seed condition_id does not match the fitted scope")
        if point.target != self.target or seed.forecast.target != self.target:
            raise ValueError("session seed target does not match the fitted scope")

        return CrossPositionDeductSession(
            estimator_id=self.estimator_id,
            target=self.target,
            condition_id=self.condition_id,
            context=context,
            seed=seed,
        )


class CrossPositionDeductEstimator:
    """Mechanical Task-pre to Task-update baseline with no learned updater."""

    estimator_id = "cross_position_deduct"

    def __init__(
        self,
        *,
        expected_condition_id: str | None = None,
        expected_input_contract_hash: str | None = None,
    ) -> None:
        if expected_condition_id is not None and not expected_condition_id.strip():
            raise ValueError("expected_condition_id must be non-empty or None")
        if expected_input_contract_hash is not None and (
            len(expected_input_contract_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_input_contract_hash
            )
        ):
            raise ValueError("expected_input_contract_hash must be a lowercase SHA-256 digest")
        self.expected_condition_id = expected_condition_id
        self.expected_input_contract_hash = expected_input_contract_hash

    def fit(
        self,
        train: TrainingView,
        validation: TrainingView,
        context: FitContext,
    ) -> FittedCrossPositionDeduct:
        del context
        if train.position != PredictionPosition.TASK_UPDATE:
            raise ValueError("cross_position_deduct supports only the Task-update position")
        if train.target != PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS:
            raise ValueError(
                "cross_position_deduct supports only "
                "task_provider_accounted_remaining_tokens"
            )
        if validation.dataset_id != train.dataset_id:
            raise ValueError("train and validation views belong to different datasets")
        if validation.position != train.position or validation.target != train.target:
            raise ValueError("train and validation views must describe the same cell")
        train_condition = _condition_id(train, name="train")
        validation_condition = _condition_id(validation, name="validation")
        if train_condition != validation_condition:
            raise ValueError("train and validation condition_id values do not match")
        if (
            self.expected_condition_id is not None
            and train_condition != self.expected_condition_id
        ):
            raise ValueError("training condition_id does not match expected_condition_id")
        train_contract_hash = _lifecycle_input_contract_hash(train, name="train")
        validation_contract_hash = _lifecycle_input_contract_hash(
            validation, name="validation"
        )
        if train_contract_hash != validation_contract_hash:
            raise ValueError("train and validation lifecycle input contracts do not match")
        if (
            self.expected_input_contract_hash is not None
            and train_contract_hash is not None
            and self.expected_input_contract_hash != train_contract_hash
        ):
            raise ValueError(
                "lifecycle input contract does not match expected_input_contract_hash"
            )
        input_contract_hash = (
            train_contract_hash
            or self.expected_input_contract_hash
            or CROSS_POSITION_INPUT_CONTRACT_HASH
        )
        return FittedCrossPositionDeduct(
            estimator_id=self.estimator_id,
            dataset_id=train.dataset_id,
            target=train.target,
            condition_id=train_condition,
            input_contract_hash=input_contract_hash,
        )


__all__ = [
    "CROSS_POSITION_INPUT_CONTRACT_HASH",
    "CrossPositionDeductEstimator",
    "CrossPositionDeductSession",
    "FittedCrossPositionDeduct",
]
