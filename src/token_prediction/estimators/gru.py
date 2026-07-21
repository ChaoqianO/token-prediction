"""Deterministic recurrent residual updater for Task remaining forecasts.

The model is deliberately anchored to the exact cross-position mechanical
Deduct transition.  It consumes only prefix-visible points and observed spend,
rolls forward with its own previous forecast, and adds a learned residual.  A
``residual_scale`` of zero therefore reduces exactly to the mechanical updater
at every boundary.
"""

from __future__ import annotations

import hashlib
import math
import platform
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from token_prediction.dataset import PredictionPoint, PredictionPosition, PredictionTarget

from .base import (
    FitContext,
    ObservedTransition,
    RunContext,
    SessionSeed,
    TokenForecast,
    TrainingExample,
    TrainingView,
)
from .cross_position_deduct import mechanical_deduct_operands
from .mlp import (
    _configure_deterministic_cpu,
    _deep_freeze_json,
    _load_neural_dependencies,
    _valid_quantiles,
)
from .neural_encoder import NeuralFeatureEncoder


GRU_RESIDUAL_ESTIMATOR_VERSION = 1
GRU_STATE_FEATURE_DIM = 5
MAX_GRU_INPUT_DIMENSION = 1_000_000
MAX_GRU_HIDDEN_WIDTH = 4096
MAX_GRU_PARAMETERS = 50_000_000
MAX_GRU_ENCODED_CELLS = 25_000_000
_FIT_PARAMETER_KEYS = frozenset(
    {
        "quantiles",
        "transition_dim",
        "hidden_dim",
        "residual_head_dim",
        "learning_rate",
        "weight_decay",
        "max_epochs",
        "patience",
        "min_delta",
        "q50_huber_delta",
        "residual_scale",
        "no_recurrence",
        "device",
        "deterministic",
        "num_threads",
        "optimizer",
        "activation",
        "teacher_forcing",
    }
)


def _derived_seed(seed: int, fold: int) -> int:
    payload = f"gru-residual-v{GRU_RESIDUAL_ESTIMATOR_VERSION}:{seed}:{fold}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big") % (
        2**31 - 1
    )


def _semantic_hash(value: object) -> str:
    import json

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _repair_tensor(torch: Any, raw: Any) -> Any:
    center = torch.clamp(raw[1], min=0.0)
    lower = torch.minimum(torch.clamp(raw[0], min=0.0), center)
    upper = torch.maximum(torch.clamp(raw[2], min=0.0), center)
    return torch.stack((lower, center, upper))


def _repair_values(raw: Sequence[float]) -> tuple[float, float, float]:
    center = max(0.0, float(raw[1]))
    return (
        min(max(0.0, float(raw[0])), center),
        center,
        max(max(0.0, float(raw[2])), center),
    )


def _weighted_target_scale(examples: Sequence[TrainingExample]) -> float:
    ordered = sorted(
        ((float(example.target_value), float(example.sample_weight)) for example in examples),
        key=lambda item: item[0],
    )
    total = sum(weight for _value, weight in ordered)
    threshold = total / 2.0
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return max(1.0, value)
    raise AssertionError("weighted target scale did not close")


def _row_quantile_loss(
    torch: Any,
    prediction: Any,
    target: Any,
    quantiles: tuple[float, float, float],
    *,
    q50_huber_delta: float | None,
) -> Any:
    error = target - prediction
    q = torch.tensor(quantiles, dtype=prediction.dtype, device=prediction.device)
    losses = torch.maximum(q * error, (q - 1.0) * error)
    if q50_huber_delta is not None:
        median_error = prediction[1] - target
        absolute = torch.abs(median_error)
        delta = float(q50_huber_delta)
        huber = torch.where(
            absolute <= delta,
            0.5 * median_error.square() / delta,
            absolute - 0.5 * delta,
        )
        losses = losses.clone()
        losses[1] = 0.5 * huber
    return losses.mean()


@dataclass(frozen=True)
class GRUArchitecture:
    point_input_dim: int
    transition_dim: int = 64
    hidden_dim: int = 64
    residual_head_dim: int = 64
    state_feature_dim: int = GRU_STATE_FEATURE_DIM
    output_dim: int = 3
    activation: str = "silu"
    recurrent_cell: str = "gru_cell"
    architecture_version: int = 1

    def __post_init__(self) -> None:
        if self.architecture_version != 1:
            raise ValueError("unsupported GRU architecture version")
        dimensions = (
            self.point_input_dim,
            self.transition_dim,
            self.hidden_dim,
            self.residual_head_dim,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in dimensions
        ):
            raise ValueError("GRU architecture dimensions must be positive integers")
        if self.point_input_dim > MAX_GRU_INPUT_DIMENSION:
            raise ValueError("GRU point input dimension exceeds the safe bundle limit")
        if any(value > MAX_GRU_HIDDEN_WIDTH for value in dimensions[1:]):
            raise ValueError("GRU hidden width exceeds the safe bundle limit")
        if self.state_feature_dim != GRU_STATE_FEATURE_DIM:
            raise ValueError("unsupported GRU state feature width")
        if self.output_dim != 3:
            raise ValueError("GRU output_dim must be three quantile residuals")
        if self.activation != "silu" or self.recurrent_cell != "gru_cell":
            raise ValueError("GRU architecture activation/cell is unsupported")
        input_dim = self.transition_input_dim
        parameter_count = (
            input_dim * self.transition_dim
            + self.transition_dim
            + 3
            * (
                self.transition_dim * self.hidden_dim
                + self.hidden_dim * self.hidden_dim
                + 2 * self.hidden_dim
            )
            + self.hidden_dim * self.residual_head_dim
            + self.residual_head_dim
            + self.residual_head_dim * self.output_dim
            + self.output_dim
        )
        if parameter_count > MAX_GRU_PARAMETERS:
            raise ValueError("GRU architecture exceeds the safe parameter limit")

    @property
    def transition_input_dim(self) -> int:
        return self.point_input_dim * 3 + self.state_feature_dim

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture_version": self.architecture_version,
            "point_input_dim": self.point_input_dim,
            "transition_dim": self.transition_dim,
            "hidden_dim": self.hidden_dim,
            "residual_head_dim": self.residual_head_dim,
            "state_feature_dim": self.state_feature_dim,
            "output_dim": self.output_dim,
            "activation": self.activation,
            "recurrent_cell": self.recurrent_cell,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GRUArchitecture":
        expected = {
            "architecture_version",
            "point_input_dim",
            "transition_dim",
            "hidden_dim",
            "residual_head_dim",
            "state_feature_dim",
            "output_dim",
            "activation",
            "recurrent_cell",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("GRU architecture keys do not match schema")
        integer_names = expected - {"activation", "recurrent_cell"}
        if any(
            isinstance(value[name], bool) or not isinstance(value[name], int)
            for name in integer_names
        ):
            raise ValueError("GRU architecture dimensions must be integers")
        if not isinstance(value["activation"], str) or not isinstance(
            value["recurrent_cell"], str
        ):
            raise ValueError("GRU architecture identifiers must be strings")
        return cls(**dict(value))


def _build_network(torch: Any, architecture: GRUArchitecture) -> Any:
    class GRUResidualNetwork(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transition_encoder = torch.nn.Sequential(
                torch.nn.Linear(
                    architecture.transition_input_dim,
                    architecture.transition_dim,
                ),
                torch.nn.SiLU(),
            )
            self.recurrent = torch.nn.GRUCell(
                architecture.transition_dim,
                architecture.hidden_dim,
            )
            self.residual_head = torch.nn.Sequential(
                torch.nn.Linear(
                    architecture.hidden_dim,
                    architecture.residual_head_dim,
                ),
                torch.nn.SiLU(),
                torch.nn.Linear(architecture.residual_head_dim, architecture.output_dim),
            )
            # Start from the exact mechanical model.  Training can move away
            # from zero, while the explicit zero-residual ablation remains exact.
            torch.nn.init.zeros_(self.residual_head[-1].weight)
            torch.nn.init.zeros_(self.residual_head[-1].bias)

        def forward_step(self, inputs: Any, hidden: Any) -> tuple[Any, Any]:
            encoded = self.transition_encoder(inputs)
            updated = self.recurrent(encoded, hidden)
            return self.residual_head(updated), updated

    return GRUResidualNetwork().to(device="cpu", dtype=torch.float32)


@dataclass(frozen=True)
class GRUFitReport:
    estimator_version: int
    encoder_schema_hash: str
    train_sequence_hash: str
    validation_sequence_hash: str
    train_sequence_count: int
    validation_sequence_count: int
    train_scored_point_count: int
    validation_scored_point_count: int
    seed: int
    best_epoch: int
    best_validation_loss: float
    validation_history: tuple[float, ...]
    target_scale: float
    parameters: Mapping[str, Any]
    torch_version: str
    numpy_version: str
    platform: str

    def __post_init__(self) -> None:
        if self.estimator_version != GRU_RESIDUAL_ESTIMATOR_VERSION:
            raise ValueError("unsupported GRU residual estimator version")
        counts = (
            self.train_sequence_count,
            self.validation_sequence_count,
            self.train_scored_point_count,
            self.validation_scored_point_count,
        )
        if any(value <= 0 for value in counts):
            raise ValueError("GRU fit report counts must be positive")
        if self.best_epoch <= 0 or self.best_epoch > len(self.validation_history):
            raise ValueError("GRU fit report best epoch is invalid")
        if not math.isfinite(self.best_validation_loss):
            raise ValueError("GRU validation loss must be finite")
        if any(not math.isfinite(value) for value in self.validation_history):
            raise ValueError("GRU validation history must be finite")
        if not math.isfinite(self.target_scale) or self.target_scale < 1:
            raise ValueError("GRU target scale must be finite and at least one")
        for name, digest in (
            ("encoder_schema_hash", self.encoder_schema_hash),
            ("train_sequence_hash", self.train_sequence_hash),
            ("validation_sequence_hash", self.validation_sequence_hash),
        ):
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if set(self.parameters) != _FIT_PARAMETER_KEYS:
            raise ValueError("GRU fit report parameters do not match the estimator contract")
        object.__setattr__(self, "parameters", _deep_freeze_json(self.parameters))


def _validate_fit_report(
    report: GRUFitReport,
    architecture: GRUArchitecture,
    quantiles: tuple[float, float, float],
) -> None:
    parameters = report.parameters
    expected_dimensions = {
        "transition_dim": architecture.transition_dim,
        "hidden_dim": architecture.hidden_dim,
        "residual_head_dim": architecture.residual_head_dim,
    }
    if tuple(parameters["quantiles"]) != quantiles:
        raise ValueError("GRU fit report quantiles do not match")
    if any(parameters[name] != value for name, value in expected_dimensions.items()):
        raise ValueError("GRU fit report dimensions do not match")
    if parameters["activation"] != architecture.activation:
        raise ValueError("GRU fit report activation does not match")
    if parameters["teacher_forcing"] is not False:
        raise ValueError("GRU teacher forcing must remain disabled")
    if parameters["device"] != "cpu" or parameters["optimizer"] != "adamw":
        raise ValueError("GRU runtime/optimizer contract is invalid")
    if parameters["deterministic"] is not True or parameters["num_threads"] != 1:
        raise ValueError("GRU deterministic CPU contract is invalid")
    if not isinstance(parameters["no_recurrence"], bool):
        raise ValueError("GRU no_recurrence must be boolean")
    scale = parameters["residual_scale"]
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(float(scale))
        or float(scale) < 0
    ):
        raise ValueError("GRU residual_scale contract is invalid")
    max_epochs = parameters["max_epochs"]
    patience = parameters["patience"]
    if (
        isinstance(max_epochs, bool)
        or not isinstance(max_epochs, int)
        or not 1 <= max_epochs <= 200
        or isinstance(patience, bool)
        or not isinstance(patience, int)
        or not 1 <= patience <= max_epochs
        or len(report.validation_history) > max_epochs
    ):
        raise ValueError("GRU epoch/patience contract is invalid")


@dataclass(frozen=True)
class _PreparedSequence:
    context_hash: str
    points: tuple[PredictionPoint, ...]
    seed_forecast: tuple[float, float, float]
    encoded: Any
    transitions: tuple[ObservedTransition, ...]
    labels: tuple[float | None, ...]
    loss_masks: tuple[bool, ...]
    weights: tuple[float, ...]


def _sequence_collection(
    view: TrainingView,
    *,
    description: str,
) -> tuple[Any, ...]:
    sequences = tuple(view.lifecycle_sequences or ())
    if not sequences:
        raise ValueError(f"GRU {description} view requires lifecycle sequences")
    conditions: set[str] = set()
    loss_examples: dict[str, TrainingExample] = {}
    for example in view.examples:
        if example.point.position != view.position or example.point.target != view.target:
            raise ValueError(f"GRU {description} point does not match its TrainingView")
        if example.point.point_id in loss_examples:
            raise ValueError(f"GRU {description} examples repeat point ids")
        loss_examples[example.point.point_id] = example
        conditions.add(example.point.condition_id)
    if len(conditions) != 1:
        raise ValueError(f"GRU {description} view must contain exactly one condition")

    sequence_examples: dict[str, tuple[PredictionPoint, float, float]] = {}
    for sequence in sequences:
        if (
            getattr(sequence, "dataset_id", None) != view.dataset_id
            or getattr(sequence, "target", None) != view.target
            or getattr(sequence, "condition_id", None) not in conditions
            or getattr(sequence, "input_contract_hash", None) != view.input_contract_hash
        ):
            raise ValueError(f"GRU {description} lifecycle scope is inconsistent")
        steps = tuple(getattr(sequence, "steps", ()))
        seed = getattr(sequence, "session_seed", None)
        if not steps or not isinstance(seed, SessionSeed):
            raise ValueError(f"GRU {description} lifecycle seed is missing")
        first = steps[0]
        if (
            first.point.position != PredictionPosition.TASK_PRE
            or first.label is not None
            or first.invalid_reason != "redacted_task_pre_label"
        ):
            raise ValueError("GRU cannot inspect the Task-pre training label")
        if seed.task_pre_point != first.point:
            raise ValueError(f"GRU {description} seed point differs from sequence")
        for step in steps[1:]:
            if step.loss_mask:
                if step.label is None or step.sample_weight <= 0:
                    raise ValueError(f"GRU {description} loss step is invalid")
                sequence_examples[step.point.point_id] = (
                    step.point,
                    float(step.label),
                    float(step.sample_weight),
                )
    if set(sequence_examples) != set(loss_examples):
        raise ValueError(f"GRU {description} examples do not exactly match loss masks")
    for point_id, (point, label, weight) in sequence_examples.items():
        example = loss_examples[point_id]
        if (
            example.point != point
            or not math.isclose(example.target_value, label, rel_tol=0.0, abs_tol=0.0)
            or not math.isclose(example.sample_weight, weight, rel_tol=0.0, abs_tol=0.0)
        ):
            raise ValueError(f"GRU {description} example differs from lifecycle step")
    return sequences


def _prepare_sequences(
    sequences: Sequence[Any],
    encoder: NeuralFeatureEncoder,
    torch: Any,
) -> tuple[_PreparedSequence, ...]:
    # Imported lazily to avoid an estimators/lifecycle import cycle.
    from token_prediction.lifecycle import visible_spend_delta

    prepared: list[_PreparedSequence] = []
    for sequence in sequences:
        steps = tuple(sequence.steps)
        points = tuple(step.point for step in steps)
        encoded = torch.from_numpy(encoder.transform(points).matrix).to(dtype=torch.float32)
        transitions: list[ObservedTransition] = []
        for previous, current in zip(points, points[1:]):
            transitions.append(
                ObservedTransition(
                    previous.point_id,
                    current.point_id,
                    visible_spend_delta(previous, current),
                )
            )
        prepared.append(
            _PreparedSequence(
                context_hash=str(sequence.context_hash),
                points=points,
                seed_forecast=(
                    float(sequence.session_seed.forecast.lower),
                    float(sequence.session_seed.forecast.point),
                    float(sequence.session_seed.forecast.upper),
                ),
                encoded=encoded,
                transitions=tuple(transitions),
                labels=tuple(
                    None if step.label is None else float(step.label) for step in steps[1:]
                ),
                loss_masks=tuple(bool(step.loss_mask) for step in steps[1:]),
                weights=tuple(float(step.sample_weight) for step in steps[1:]),
            )
        )
    return tuple(prepared)


def _rollout_loss(
    torch: Any,
    model: Any,
    sequences: Sequence[_PreparedSequence],
    *,
    architecture: GRUArchitecture,
    target_scale: float,
    quantiles: tuple[float, float, float],
    q50_huber_delta: float | None,
    residual_scale: float,
    no_recurrence: bool,
) -> Any:
    weighted_losses: list[Any] = []
    weights: list[float] = []
    for sequence in sequences:
        previous_forecast = torch.tensor(sequence.seed_forecast, dtype=torch.float32)
        hidden = torch.zeros((1, architecture.hidden_dim), dtype=torch.float32)
        previous_encoded = sequence.encoded[0]
        for index, (previous_point, current_point, transition) in enumerate(
            zip(sequence.points, sequence.points[1:], sequence.transitions)
        ):
            operands = mechanical_deduct_operands(
                previous_point,
                current_point,
                transition,
            )
            raw_mechanical = (
                previous_forecast
                if operands.delta_tokens is None
                else previous_forecast + float(operands.delta_tokens)
            )
            mechanical = _repair_tensor(torch, raw_mechanical)
            current_encoded = sequence.encoded[index + 1]
            spend = transition.observed_spend_tokens
            state = torch.tensor(
                (
                    0.0 if spend is None else float(spend) / target_scale,
                    0.0 if spend is None else 1.0,
                    float(mechanical[0].detach()) / target_scale,
                    float(mechanical[1].detach()) / target_scale,
                    float(mechanical[2].detach()) / target_scale,
                ),
                dtype=torch.float32,
            )
            transition_input = torch.cat(
                (
                    previous_encoded,
                    current_encoded,
                    current_encoded - previous_encoded,
                    state,
                )
            ).unsqueeze(0)
            recurrent_input = (
                torch.zeros_like(hidden) if no_recurrence else hidden
            )
            residual, updated_hidden = model.forward_step(
                transition_input,
                recurrent_input,
            )
            raw = mechanical + residual[0] * (target_scale * residual_scale)
            prediction = _repair_tensor(torch, raw)
            if sequence.loss_masks[index]:
                label = sequence.labels[index]
                weight = sequence.weights[index]
                if label is None or weight <= 0:
                    raise ValueError("GRU loss mask is inconsistent with label/weight")
                target = torch.tensor(label, dtype=torch.float32)
                weighted_losses.append(
                    _row_quantile_loss(
                        torch,
                        prediction,
                        target,
                        quantiles,
                        q50_huber_delta=q50_huber_delta,
                    )
                    * weight
                )
                weights.append(weight)
            # This is the model's own repaired forecast.  No target value is
            # substituted here or at any later transition.
            previous_forecast = prediction
            previous_encoded = current_encoded
            hidden = updated_hidden
    if not weighted_losses or not weights:
        raise ValueError("GRU rollout contains no scored loss points")
    return torch.stack(weighted_losses).sum() / sum(weights)


@dataclass
class GRUResidualSession:
    context: RunContext
    target: PredictionTarget
    condition_id: str
    encoder: NeuralFeatureEncoder
    architecture: GRUArchitecture
    model: Any
    target_scale: float
    residual_scale: float
    no_recurrence: bool
    seed: SessionSeed
    calibrator: Any | None = None
    fallback_count: int = 0
    last_fallback_reason: str | None = None
    _last_point: PredictionPoint = field(init=False)
    _last_forecast: TokenForecast = field(init=False)
    _pending_transition: ObservedTransition | None = field(init=False, default=None)
    _hidden: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._last_point = self.seed.task_pre_point
        self._last_forecast = self.seed.forecast

    def _validate_point(self, point: PredictionPoint) -> None:
        for name in ("task_id", "trajectory_id", "run_id"):
            if getattr(point, name) != getattr(self.context, name):
                raise ValueError(f"prediction point {name} does not match the session")
        if point.condition_id != self.condition_id:
            raise ValueError("prediction point condition_id does not match the fitted scope")
        if point.position != PredictionPosition.TASK_UPDATE:
            raise ValueError("gru_residual predicts only Task-update points")
        if point.target != self.target:
            raise ValueError("prediction point target does not match the fitted scope")

    def observe(self, transition: ObservedTransition) -> None:
        if self._pending_transition is not None:
            raise RuntimeError("the previous transition has not been consumed by predict")
        if transition.from_point_id != self._last_point.point_id:
            raise ValueError("transition from_point_id does not match the previous point")
        if not transition.to_point_id or transition.to_point_id == transition.from_point_id:
            raise ValueError("transition must advance to a different point")
        self._pending_transition = transition

    def predict(self, point: PredictionPoint) -> TokenForecast:
        self._validate_point(point)
        transition = self._pending_transition
        if transition is None:
            raise RuntimeError("observe must be called before each Task-update prediction")
        operands = mechanical_deduct_operands(self._last_point, point, transition)
        previous = self._last_forecast
        previous_values = (previous.lower, previous.point, previous.upper)
        raw_mechanical = (
            previous_values
            if operands.delta_tokens is None
            else tuple(value + operands.delta_tokens for value in previous_values)
        )
        mechanical = _repair_values(raw_mechanical)
        if operands.delta_tokens is None:
            self.fallback_count += 1
            self.last_fallback_reason = operands.fallback_reason
        else:
            self.last_fallback_reason = None

        np, torch = _load_neural_dependencies()
        del np
        encoded = self.encoder.transform((self._last_point, point)).matrix
        previous_encoded = torch.from_numpy(encoded[0]).to(dtype=torch.float32)
        current_encoded = torch.from_numpy(encoded[1]).to(dtype=torch.float32)
        spend = transition.observed_spend_tokens
        state = torch.tensor(
            (
                0.0 if spend is None else float(spend) / self.target_scale,
                0.0 if spend is None else 1.0,
                mechanical[0] / self.target_scale,
                mechanical[1] / self.target_scale,
                mechanical[2] / self.target_scale,
            ),
            dtype=torch.float32,
        )
        inputs = torch.cat(
            (
                previous_encoded,
                current_encoded,
                current_encoded - previous_encoded,
                state,
            )
        ).unsqueeze(0)
        hidden = (
            torch.zeros((1, self.architecture.hidden_dim), dtype=torch.float32)
            if self._hidden is None or self.no_recurrence
            else self._hidden
        )
        self.model.eval()
        with torch.inference_mode():
            residual, updated_hidden = self.model.forward_step(inputs, hidden)
            raw = tuple(
                mechanical[index]
                + float(residual[0, index]) * self.target_scale * self.residual_scale
                for index in range(3)
            )
        repaired = _repair_values(raw)
        uncalibrated = TokenForecast(
            point_id=point.point_id,
            target=self.target,
            lower=repaired[0],
            point=repaired[1],
            upper=repaired[2],
            raw_lower=raw[0],
            raw_point=raw[1],
            raw_upper=raw[2],
        )
        # Preserve the raw updater state even when a standalone loaded bundle
        # exposes calibrated outputs.
        self._last_point = point
        self._last_forecast = uncalibrated
        self._hidden = updated_hidden.detach().cpu()
        self._pending_transition = None
        return (
            self.calibrator.transform(uncalibrated)
            if self.calibrator is not None
            else uncalibrated
        )


@dataclass(frozen=True)
class FittedGRUResidual:
    estimator_id: str
    target: PredictionTarget
    dataset_id: str
    input_contract_hash: str
    condition_id: str
    encoder: NeuralFeatureEncoder
    architecture: GRUArchitecture
    model: Any
    quantiles: tuple[float, float, float]
    target_scale: float
    residual_scale: float
    no_recurrence: bool
    fit_report: GRUFitReport
    calibrator_document: Mapping[str, Any] | None = None
    provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.estimator_id != "gru_residual":
            raise ValueError("unsupported GRU residual estimator id")
        if not self.dataset_id or not self.condition_id:
            raise ValueError("GRU fitted scope identifiers are required")
        if len(self.input_contract_hash) != 64 or any(
            char not in "0123456789abcdef" for char in self.input_contract_hash
        ):
            raise ValueError("GRU input contract hash is invalid")
        if self.target != PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS:
            raise ValueError("GRU residual supports provider-accounted Task remaining only")
        if self.architecture.point_input_dim != self.encoder.schema.output_width:
            raise ValueError("GRU architecture does not match encoder width")
        if not _valid_quantiles(self.quantiles):
            raise ValueError("GRU quantiles must be symmetric around 0.5")
        if not math.isfinite(self.target_scale) or self.target_scale < 1:
            raise ValueError("GRU target scale is invalid")
        if not math.isfinite(self.residual_scale) or self.residual_scale < 0:
            raise ValueError("GRU residual scale is invalid")
        if self.fit_report.encoder_schema_hash != self.encoder.schema.content_hash:
            raise ValueError("GRU fit report encoder hash is inconsistent")
        _validate_fit_report(self.fit_report, self.architecture, self.quantiles)
        if not math.isclose(
            self.fit_report.target_scale,
            self.target_scale,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("GRU fit report target scale is inconsistent")
        if not math.isclose(
            float(self.fit_report.parameters["residual_scale"]),
            self.residual_scale,
            rel_tol=0.0,
            abs_tol=0.0,
        ) or bool(self.fit_report.parameters["no_recurrence"]) != self.no_recurrence:
            raise ValueError("GRU fit report state policy is inconsistent")
        if self.calibrator_document is not None:
            object.__setattr__(
                self,
                "calibrator_document",
                MappingProxyType(dict(self.calibrator_document)),
            )
        if self.provenance is not None:
            object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def start(self, context: RunContext) -> GRUResidualSession:
        required = {
            "dataset_id": context.dataset_id,
            "condition_id": context.condition_id,
            "target": context.target,
            "input_contract_hash": context.input_contract_hash,
            "session_seed": context.session_seed,
        }
        for name, value in required.items():
            if value is None:
                raise ValueError(f"GRU residual requires RunContext.{name}")
        if context.dataset_id != self.dataset_id:
            raise ValueError("RunContext dataset_id does not match the fitted dataset")
        if context.condition_id != self.condition_id:
            raise ValueError("RunContext condition_id does not match the fitted scope")
        if context.target != self.target:
            raise ValueError("RunContext target does not match the fitted scope")
        if context.input_contract_hash != self.input_contract_hash:
            raise ValueError("RunContext input_contract_hash does not match the fitted contract")
        seed = context.session_seed
        if not isinstance(seed, SessionSeed):
            raise ValueError("GRU residual requires a valid SessionSeed")
        for name in ("task_id", "trajectory_id", "run_id"):
            if getattr(seed.task_pre_point, name) != getattr(context, name):
                raise ValueError(f"session seed {name} does not match RunContext")
        if seed.task_pre_point.condition_id != self.condition_id:
            raise ValueError("session seed condition_id does not match the fitted scope")
        calibrator = None
        if self.calibrator_document is not None:
            from token_prediction.evaluation.calibration import FittedExpansionCalibrator

            calibrator = FittedExpansionCalibrator.from_dict(self.calibrator_document)
        return GRUResidualSession(
            context=context,
            target=self.target,
            condition_id=self.condition_id,
            encoder=self.encoder,
            architecture=self.architecture,
            model=self.model,
            target_scale=self.target_scale,
            residual_scale=self.residual_scale,
            no_recurrence=self.no_recurrence,
            seed=seed,
            calibrator=calibrator,
        )

    def bundle_files(
        self,
        *,
        calibrator: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> Mapping[str, bytes]:
        from .gru_bundle import gru_bundle_files

        return gru_bundle_files(self, calibrator=calibrator, provenance=provenance)


class GRUResidualEstimator:
    estimator_id = "gru_residual"

    def __init__(
        self,
        *,
        quantiles: tuple[float, float, float] | list[float] | None = None,
        transition_dim: int = 64,
        hidden_dim: int = 64,
        residual_head_dim: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 200,
        patience: int = 20,
        min_delta: float = 0.0,
        q50_huber_delta: float | None = None,
        residual_scale: float = 1.0,
        no_recurrence: bool = False,
    ) -> None:
        if quantiles is not None and (
            not isinstance(quantiles, (tuple, list))
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in quantiles
            )
        ):
            raise ValueError("quantiles must be finite JSON numbers")
        normalized_quantiles = (
            tuple(float(value) for value in quantiles) if quantiles is not None else None
        )
        if normalized_quantiles is not None and not _valid_quantiles(normalized_quantiles):
            raise ValueError("quantiles must be symmetric and ordered around 0.5")
        for name, value in (
            ("transition_dim", transition_dim),
            ("hidden_dim", hidden_dim),
            ("residual_head_dim", residual_head_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value, positive in (
            ("learning_rate", learning_rate, True),
            ("weight_decay", weight_decay, False),
            ("min_delta", min_delta, False),
            ("residual_scale", residual_scale, False),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or (float(value) <= 0 if positive else float(value) < 0)
            ):
                qualifier = "positive" if positive else "non-negative"
                raise ValueError(f"{name} must be finite and {qualifier}")
        if isinstance(max_epochs, bool) or not isinstance(max_epochs, int) or not 1 <= max_epochs <= 200:
            raise ValueError("max_epochs must be an integer in [1, 200]")
        if isinstance(patience, bool) or not isinstance(patience, int) or not 1 <= patience <= max_epochs:
            raise ValueError("patience must be an integer in [1, max_epochs]")
        if q50_huber_delta is not None and (
            isinstance(q50_huber_delta, bool)
            or not isinstance(q50_huber_delta, (int, float))
            or not math.isfinite(float(q50_huber_delta))
            or q50_huber_delta <= 0
        ):
            raise ValueError("q50_huber_delta must be finite and positive when enabled")
        if not isinstance(no_recurrence, bool):
            raise ValueError("no_recurrence must be boolean")
        self.quantiles = normalized_quantiles
        self.transition_dim = transition_dim
        self.hidden_dim = hidden_dim
        self.residual_head_dim = residual_head_dim
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = float(min_delta)
        self.q50_huber_delta = (
            None if q50_huber_delta is None else float(q50_huber_delta)
        )
        self.residual_scale = float(residual_scale)
        self.no_recurrence = no_recurrence

    def fit(
        self,
        train: TrainingView,
        validation: TrainingView,
        context: FitContext,
    ) -> FittedGRUResidual:
        expected_quantiles = (
            context.interval_alpha / 2,
            0.5,
            1 - context.interval_alpha / 2,
        )
        if self.quantiles is not None and any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
            for actual, expected in zip(self.quantiles, expected_quantiles)
        ):
            raise ValueError("configured quantiles do not match experiment interval_alpha")
        quantiles = self.quantiles or expected_quantiles
        if train.dataset_id != validation.dataset_id:
            raise ValueError("GRU train and validation datasets differ")
        if train.input_contract_hash is None or (
            train.input_contract_hash != validation.input_contract_hash
        ):
            raise ValueError("GRU train and validation input contracts differ")
        if (
            train.position != PredictionPosition.TASK_UPDATE
            or validation.position != train.position
        ):
            raise ValueError("GRU residual supports only Task-update")
        if (
            train.target != PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS
            or validation.target != train.target
        ):
            raise ValueError("GRU residual supports provider-accounted Task remaining only")
        train_sequences = _sequence_collection(train, description="train")
        validation_sequences = _sequence_collection(validation, description="validation")
        train_condition = {sequence.condition_id for sequence in train_sequences}
        validation_condition = {sequence.condition_id for sequence in validation_sequences}
        if len(train_condition) != 1 or train_condition != validation_condition:
            raise ValueError("GRU train and validation condition scopes differ")

        np, torch = _load_neural_dependencies()
        seed = _derived_seed(context.seed, context.fold)
        _configure_deterministic_cpu(torch, seed)
        train_points = tuple(
            step.point for sequence in train_sequences for step in sequence.steps
        )
        validation_points = tuple(
            step.point for sequence in validation_sequences for step in sequence.steps
        )
        encoder = NeuralFeatureEncoder.fit(train_points)
        if encoder.schema.output_width <= 0:
            raise ValueError("GRU residual requires at least one train-fold feature")
        architecture = GRUArchitecture(
            point_input_dim=encoder.schema.output_width,
            transition_dim=self.transition_dim,
            hidden_dim=self.hidden_dim,
            residual_head_dim=self.residual_head_dim,
        )
        encoded_cells = (
            len(train_points) + len(validation_points)
        ) * architecture.point_input_dim
        if encoded_cells > MAX_GRU_ENCODED_CELLS:
            raise ValueError("GRU encoded matrices exceed the safe cell-count limit")
        prepared_train = _prepare_sequences(train_sequences, encoder, torch)
        prepared_validation = _prepare_sequences(validation_sequences, encoder, torch)
        target_scale = _weighted_target_scale(train.examples)
        model = _build_network(torch, architecture)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        history: list[float] = []
        best_loss = math.inf
        best_epoch = 0
        best_state: dict[str, Any] | None = None
        stale_epochs = 0
        rollout_kwargs = {
            "architecture": architecture,
            "target_scale": target_scale,
            "quantiles": quantiles,
            "q50_huber_delta": self.q50_huber_delta,
            "residual_scale": self.residual_scale,
            "no_recurrence": self.no_recurrence,
        }
        for epoch in range(1, self.max_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = _rollout_loss(
                torch,
                model,
                prepared_train,
                **rollout_kwargs,
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("GRU training loss became non-finite")
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.inference_mode():
                validation_loss = float(
                    _rollout_loss(
                        torch,
                        model,
                        prepared_validation,
                        **rollout_kwargs,
                    ).item()
                )
            if not math.isfinite(validation_loss):
                raise RuntimeError("GRU validation loss became non-finite")
            history.append(validation_loss)
            if validation_loss < best_loss - self.min_delta:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break
        if best_state is None:
            raise RuntimeError("GRU residual did not produce a valid checkpoint")
        model.load_state_dict(best_state, strict=True)
        model.eval()
        parameters: dict[str, Any] = {
            "quantiles": list(quantiles),
            "transition_dim": self.transition_dim,
            "hidden_dim": self.hidden_dim,
            "residual_head_dim": self.residual_head_dim,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "q50_huber_delta": self.q50_huber_delta,
            "residual_scale": self.residual_scale,
            "no_recurrence": self.no_recurrence,
            "device": "cpu",
            "deterministic": True,
            "num_threads": 1,
            "optimizer": "adamw",
            "activation": "silu",
            "teacher_forcing": False,
        }
        report = GRUFitReport(
            estimator_version=GRU_RESIDUAL_ESTIMATOR_VERSION,
            encoder_schema_hash=encoder.schema.content_hash,
            train_sequence_hash=_semantic_hash(
                [
                    {
                        "context_hash": sequence.context_hash,
                        "seed_hash": sequence.session_seed.content_hash,
                    }
                    for sequence in train_sequences
                ]
            ),
            validation_sequence_hash=_semantic_hash(
                [
                    {
                        "context_hash": sequence.context_hash,
                        "seed_hash": sequence.session_seed.content_hash,
                    }
                    for sequence in validation_sequences
                ]
            ),
            train_sequence_count=len(train_sequences),
            validation_sequence_count=len(validation_sequences),
            train_scored_point_count=len(train.examples),
            validation_scored_point_count=len(validation.examples),
            seed=seed,
            best_epoch=best_epoch,
            best_validation_loss=best_loss,
            validation_history=tuple(history),
            target_scale=target_scale,
            parameters=parameters,
            torch_version=str(torch.__version__),
            numpy_version=str(np.__version__),
            platform=platform.platform(),
        )
        return FittedGRUResidual(
            estimator_id=self.estimator_id,
            target=train.target,
            dataset_id=train.dataset_id,
            input_contract_hash=train.input_contract_hash,
            condition_id=next(iter(train_condition)),
            encoder=encoder,
            architecture=architecture,
            model=model,
            quantiles=quantiles,
            target_scale=target_scale,
            residual_scale=self.residual_scale,
            no_recurrence=self.no_recurrence,
            fit_report=report,
        )


__all__ = [
    "FittedGRUResidual",
    "GRUArchitecture",
    "GRUFitReport",
    "GRUResidualEstimator",
    "GRUResidualSession",
    "GRU_RESIDUAL_ESTIMATOR_VERSION",
    "MAX_GRU_ENCODED_CELLS",
    "MAX_GRU_HIDDEN_WIDTH",
    "MAX_GRU_INPUT_DIMENSION",
    "MAX_GRU_PARAMETERS",
]
