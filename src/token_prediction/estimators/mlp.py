"""Deterministic, position-local MLP quantile estimator.

PyTorch and NumPy are optional runtime dependencies.  They are imported only
when fitting or predicting, so the base package remains usable without the
``neural`` extra.
"""

from __future__ import annotations

import hashlib
import math
import platform
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from token_prediction.dataset import PredictionPoint, PredictionPosition, PredictionTarget

from .base import FitContext, ObservedTransition, RunContext, TokenForecast, TrainingView
from .neural_encoder import NeuralFeatureEncoder, OptionalNeuralDependencyError


INDEPENDENT_MLP_ESTIMATOR_VERSION = 1
MAX_MLP_DIMENSION = 1_000_000
MAX_MLP_HIDDEN_WIDTH = 4096
MAX_MLP_PARAMETERS = 50_000_000
MAX_MLP_ENCODED_CELLS = 25_000_000
_FIT_PARAMETER_KEYS = frozenset(
    {
        "quantiles",
        "hidden_dims",
        "learning_rate",
        "weight_decay",
        "max_epochs",
        "patience",
        "min_delta",
        "q50_huber_delta",
        "device",
        "deterministic",
        "num_threads",
        "optimizer",
        "activation",
    }
)


def _deep_freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("fit report parameter keys must be strings")
        return MappingProxyType(
            {key: _deep_freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("fit report parameters must be finite JSON values")
        return value
    raise ValueError("fit report parameters must contain only canonical JSON values")


def _valid_quantiles(values: tuple[float, ...]) -> bool:
    return (
        len(values) == 3
        and 0 < values[0] < values[1] < values[2] < 1
        and math.isclose(values[1], 0.5, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(values[2], 1.0 - values[0], rel_tol=0.0, abs_tol=1e-12)
    )


def _load_neural_dependencies() -> tuple[Any, Any]:
    try:
        import numpy as np
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - base-only CI exercises this
        raise OptionalNeuralDependencyError(
            "Independent MLP estimation requires optional dependencies; "
            "install token-prediction[neural]"
        ) from exc
    return np, torch


def _point_hash(point_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(point_ids)).encode("utf-8")).hexdigest()


def _derived_seed(seed: int, fold: int) -> int:
    payload = f"independent-mlp-v{INDEPENDENT_MLP_ESTIMATOR_VERSION}:{seed}:{fold}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big") % (
        2**31 - 1
    )


def _view_condition_ids(view: TrainingView, *, description: str) -> tuple[str, ...]:
    conditions: set[str] = set()
    for example in view.examples:
        if example.point.position != view.position or example.point.target != view.target:
            raise ValueError(f"{description} point does not match its TrainingView cell")
        conditions.add(example.point.condition_id)
    if len(conditions) != 1:
        raise ValueError(f"{description} view must contain exactly one condition scope")
    return tuple(sorted(conditions))


def _configure_deterministic_cpu(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting the inter-op pool only before parallel work.
        # A prior deterministic fit may already have fixed it to one thread.
        if torch.get_num_interop_threads() != 1:
            raise
    torch.use_deterministic_algorithms(True)


def _build_network(torch: Any, architecture: "MLPArchitecture") -> Any:
    dimensions = (architecture.input_dim, *architecture.hidden_dims, architecture.output_dim)
    layers: list[Any] = []
    for index, (input_width, output_width) in enumerate(zip(dimensions, dimensions[1:])):
        layers.append(torch.nn.Linear(input_width, output_width, bias=True))
        if index < len(dimensions) - 2:
            layers.append(torch.nn.SiLU())
    return torch.nn.Sequential(*layers).to(device="cpu", dtype=torch.float32)


def _weighted_quantile_loss(
    torch: Any,
    predictions: Any,
    targets: Any,
    weights: Any,
    quantiles: tuple[float, float, float],
    *,
    q50_huber_delta: float | None,
) -> Any:
    if predictions.ndim != 2 or predictions.shape[1] != 3:
        raise ValueError("quantile predictions must have shape (n, 3)")
    errors = targets.unsqueeze(1) - predictions
    q = torch.tensor(quantiles, dtype=predictions.dtype, device=predictions.device)
    losses = torch.maximum(q * errors, (q - 1.0) * errors)
    if q50_huber_delta is not None:
        median_error = predictions[:, 1] - targets
        absolute = torch.abs(median_error)
        delta = float(q50_huber_delta)
        huber = torch.where(
            absolute <= delta,
            0.5 * median_error.square() / delta,
            absolute - 0.5 * delta,
        )
        losses = losses.clone()
        losses[:, 1] = 0.5 * huber
    row_loss = losses.mean(dim=1)
    return torch.sum(row_loss * weights) / torch.sum(weights)


@dataclass(frozen=True)
class MLPArchitecture:
    input_dim: int
    hidden_dims: tuple[int, int] = (128, 64)
    output_dim: int = 3
    activation: str = "silu"
    architecture_version: int = 1

    def __post_init__(self) -> None:
        if self.architecture_version != 1:
            raise ValueError("unsupported MLP architecture version")
        if self.input_dim <= 0:
            raise ValueError("MLP input_dim must be positive")
        if self.input_dim > MAX_MLP_DIMENSION:
            raise ValueError("MLP input_dim exceeds the safe bundle limit")
        if len(self.hidden_dims) != 2 or any(width <= 0 for width in self.hidden_dims):
            raise ValueError("MLP requires exactly two positive hidden dimensions")
        if any(width > MAX_MLP_HIDDEN_WIDTH for width in self.hidden_dims):
            raise ValueError("MLP hidden width exceeds the safe bundle limit")
        if self.output_dim != 3:
            raise ValueError("MLP output_dim must be three quantiles")
        if self.activation != "silu":
            raise ValueError("MLP activation must be SiLU")
        dimensions = (self.input_dim, *self.hidden_dims, self.output_dim)
        parameter_count = sum(
            input_width * output_width + output_width
            for input_width, output_width in zip(dimensions, dimensions[1:])
        )
        if parameter_count > MAX_MLP_PARAMETERS:
            raise ValueError("MLP architecture exceeds the safe parameter limit")

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture_version": self.architecture_version,
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "output_dim": self.output_dim,
            "activation": self.activation,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MLPArchitecture":
        expected = {
            "architecture_version",
            "input_dim",
            "hidden_dims",
            "output_dim",
            "activation",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("MLP architecture keys do not match schema")
        integers = (value["architecture_version"], value["input_dim"], value["output_dim"])
        if any(isinstance(item, bool) or not isinstance(item, int) for item in integers):
            raise ValueError("MLP architecture dimensions must be integers")
        hidden = value["hidden_dims"]
        if not isinstance(hidden, list) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in hidden
        ):
            raise ValueError("MLP hidden_dims must be an integer list")
        if not isinstance(value["activation"], str):
            raise ValueError("MLP activation must be a string")
        return cls(
            architecture_version=value["architecture_version"],
            input_dim=value["input_dim"],
            hidden_dims=tuple(hidden),  # type: ignore[arg-type]
            output_dim=value["output_dim"],
            activation=value["activation"],
        )


@dataclass(frozen=True)
class MLPFitReport:
    estimator_version: int
    encoder_schema_hash: str
    train_point_hash: str
    validation_point_hash: str
    train_point_count: int
    validation_point_count: int
    seed: int
    best_epoch: int
    best_validation_loss: float
    validation_history: tuple[float, ...]
    parameters: Mapping[str, Any]
    torch_version: str
    numpy_version: str
    platform: str

    def __post_init__(self) -> None:
        if self.estimator_version != INDEPENDENT_MLP_ESTIMATOR_VERSION:
            raise ValueError("unsupported Independent MLP estimator version")
        if self.train_point_count <= 0 or self.validation_point_count <= 0:
            raise ValueError("fit report point counts must be positive")
        if self.best_epoch <= 0 or self.best_epoch > len(self.validation_history):
            raise ValueError("fit report best epoch is invalid")
        if not math.isfinite(self.best_validation_loss):
            raise ValueError("fit report validation loss must be finite")
        if any(not math.isfinite(value) for value in self.validation_history):
            raise ValueError("fit report validation history must be finite")
        if set(self.parameters) != _FIT_PARAMETER_KEYS:
            raise ValueError("fit report parameter keys do not match the estimator contract")
        object.__setattr__(self, "parameters", _deep_freeze_json(self.parameters))


def _validate_fit_report_semantics(
    report: MLPFitReport,
    architecture: MLPArchitecture,
    quantiles: tuple[float, float, float],
) -> None:
    parameters = report.parameters
    if tuple(parameters["quantiles"]) != quantiles:
        raise ValueError("fit report quantiles do not match the fitted model")
    if tuple(parameters["hidden_dims"]) != architecture.hidden_dims:
        raise ValueError("fit report hidden_dims do not match the architecture")
    if parameters["activation"] != architecture.activation:
        raise ValueError("fit report activation does not match the architecture")
    if parameters["device"] != "cpu" or parameters["optimizer"] != "adamw":
        raise ValueError("fit report runtime/optimizer contract is invalid")
    if parameters["deterministic"] is not True or parameters["num_threads"] != 1:
        raise ValueError("fit report deterministic CPU contract is invalid")
    max_epochs = parameters["max_epochs"]
    patience = parameters["patience"]
    if (
        isinstance(max_epochs, bool)
        or not isinstance(max_epochs, int)
        or not 1 <= max_epochs <= 200
        or isinstance(patience, bool)
        or not isinstance(patience, int)
        or not 1 <= patience <= max_epochs
    ):
        raise ValueError("fit report epoch/patience contract is invalid")
    if len(report.validation_history) > max_epochs:
        raise ValueError("fit report history exceeds max_epochs")
    for name, minimum, strict in (
        ("learning_rate", 0.0, True),
        ("weight_decay", 0.0, False),
        ("min_delta", 0.0, False),
    ):
        value = parameters[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or (float(value) <= minimum if strict else float(value) < minimum)
        ):
            raise ValueError(f"fit report {name} contract is invalid")
    huber = parameters["q50_huber_delta"]
    if huber is not None and (
        isinstance(huber, bool)
        or not isinstance(huber, (int, float))
        or not math.isfinite(float(huber))
        or float(huber) <= 0
    ):
        raise ValueError("fit report q50_huber_delta contract is invalid")


@dataclass
class IndependentMLPSession:
    context: RunContext
    target: PredictionTarget
    position: PredictionPosition
    allowed_condition_ids: tuple[str, ...]
    encoder: NeuralFeatureEncoder
    model: Any
    quantiles: tuple[float, float, float]
    calibrator: Any | None = None

    def predict(self, point: PredictionPoint) -> TokenForecast:
        if point.task_id != self.context.task_id:
            raise ValueError("prediction point task_id does not match the session")
        if point.trajectory_id != self.context.trajectory_id:
            raise ValueError("prediction point trajectory_id does not match the session")
        if point.run_id != self.context.run_id:
            raise ValueError("prediction point run_id does not match the session")
        if point.target != self.target:
            raise ValueError(f"bundle target is {self.target.value!r}, got {point.target.value!r}")
        if point.position != self.position:
            raise ValueError(
                f"bundle position is {self.position.value!r}, got {point.position.value!r}"
            )
        if point.condition_id not in self.allowed_condition_ids:
            raise ValueError(f"condition_id {point.condition_id!r} is outside the bundle scope")
        _, torch = _load_neural_dependencies()
        encoded = self.encoder.transform((point,))
        inputs = torch.from_numpy(encoded.matrix).to(device="cpu", dtype=torch.float32)
        self.model.eval()
        with torch.inference_mode():
            raw = tuple(float(value) for value in self.model(inputs)[0].tolist())
        repaired_point = max(0.0, raw[1])
        repaired = (
            min(max(0.0, raw[0]), repaired_point),
            repaired_point,
            max(max(0.0, raw[2]), repaired_point),
        )
        forecast = TokenForecast(
            point_id=point.point_id,
            target=self.target,
            lower=repaired[0],
            point=repaired[1],
            upper=repaired[2],
            raw_lower=raw[0],
            raw_point=raw[1],
            raw_upper=raw[2],
        )
        return self.calibrator.transform(forecast) if self.calibrator is not None else forecast

    def observe(self, transition: ObservedTransition) -> None:
        # An Independent MLP has no recurrent, teacher-forced, or spend state.
        del transition


@dataclass(frozen=True)
class FittedIndependentMLP:
    estimator_id: str
    target: PredictionTarget
    position: PredictionPosition
    dataset_id: str
    input_contract_hash: str | None
    allowed_condition_ids: tuple[str, ...]
    encoder: NeuralFeatureEncoder
    architecture: MLPArchitecture
    model: Any
    quantiles: tuple[float, float, float]
    fit_report: MLPFitReport
    calibrator_document: Mapping[str, Any] | None = None
    provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.estimator_id != "independent_mlp":
            raise ValueError("unsupported Independent MLP estimator id")
        if not self.dataset_id.strip():
            raise ValueError("dataset_id is required for a fitted Independent MLP")
        if tuple(sorted(set(self.allowed_condition_ids))) != self.allowed_condition_ids:
            raise ValueError("allowed condition ids must be sorted and unique")
        if not self.allowed_condition_ids or any(
            not value.strip() for value in self.allowed_condition_ids
        ):
            raise ValueError("at least one non-empty allowed condition id is required")
        if self.architecture.input_dim != self.encoder.schema.output_width:
            raise ValueError("MLP architecture does not match encoder output width")
        if not _valid_quantiles(self.quantiles):
            raise ValueError("MLP quantiles must be symmetric and ordered around 0.5")
        if self.fit_report.encoder_schema_hash != self.encoder.schema.content_hash:
            raise ValueError("MLP fit report encoder hash is inconsistent")
        _validate_fit_report_semantics(self.fit_report, self.architecture, self.quantiles)
        if self.calibrator_document is not None:
            object.__setattr__(
                self, "calibrator_document", MappingProxyType(dict(self.calibrator_document))
            )
        if self.provenance is not None:
            object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def start(self, context: RunContext) -> IndependentMLPSession:
        if context.dataset_id is None:
            raise ValueError("Independent MLP requires RunContext.dataset_id")
        if context.dataset_id != self.dataset_id:
            raise ValueError("RunContext dataset_id does not match the fitted dataset")
        if context.condition_id is None:
            raise ValueError("Independent MLP requires RunContext.condition_id")
        if context.condition_id not in self.allowed_condition_ids:
            raise ValueError("RunContext condition_id is outside the fitted scope")
        if context.target is None:
            raise ValueError("Independent MLP requires RunContext.target")
        if context.target != self.target:
            raise ValueError("RunContext target does not match the fitted target")
        if context.input_contract_hash is None:
            raise ValueError("Independent MLP requires RunContext.input_contract_hash")
        if self.input_contract_hash is None:
            raise ValueError("fitted MLP has no input contract for this RunContext")
        if context.input_contract_hash != self.input_contract_hash:
            raise ValueError("RunContext input_contract_hash does not match the fitted contract")
        calibrator = None
        if self.calibrator_document is not None:
            # Import locally to keep the estimator module safe in the minimal
            # dependency graph and to avoid an estimators/evaluation cycle.
            from token_prediction.evaluation.calibration import FittedExpansionCalibrator

            calibrator = FittedExpansionCalibrator.from_dict(self.calibrator_document)
        return IndependentMLPSession(
            context=context,
            target=self.target,
            position=self.position,
            allowed_condition_ids=self.allowed_condition_ids,
            encoder=self.encoder,
            model=self.model,
            quantiles=self.quantiles,
            calibrator=calibrator,
        )

    def bundle_files(
        self,
        *,
        calibrator: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> Mapping[str, bytes]:
        from .neural_bundle import neural_bundle_files

        return neural_bundle_files(
            self,
            calibrator=calibrator,
            provenance=provenance,
        )


class IndependentMLPQuantileEstimator:
    estimator_id = "independent_mlp"

    def __init__(
        self,
        *,
        quantiles: tuple[float, float, float] | list[float] | None = None,
        hidden_dims: tuple[int, int] | list[int] = (128, 64),
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 200,
        patience: int = 20,
        min_delta: float = 0.0,
        q50_huber_delta: float | None = None,
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
            raise ValueError(
                "quantiles must contain symmetric ordered (lower, 0.5, upper) values"
            )
        if not isinstance(hidden_dims, (tuple, list)) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in hidden_dims
        ):
            raise ValueError("hidden_dims must contain integer widths")
        normalized_hidden = tuple(hidden_dims)
        if len(normalized_hidden) != 2 or any(value <= 0 for value in normalized_hidden):
            raise ValueError("hidden_dims must contain exactly two positive widths")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not math.isfinite(float(learning_rate))
            or learning_rate <= 0
        ):
            raise ValueError("learning_rate must be finite and positive")
        if (
            isinstance(weight_decay, bool)
            or not isinstance(weight_decay, (int, float))
            or not math.isfinite(float(weight_decay))
            or weight_decay < 0
        ):
            raise ValueError("weight_decay must be finite and non-negative")
        if isinstance(max_epochs, bool) or not isinstance(max_epochs, int):
            raise ValueError("max_epochs must be an integer")
        if max_epochs <= 0 or max_epochs > 200:
            raise ValueError("max_epochs must be in [1, 200]")
        if isinstance(patience, bool) or not isinstance(patience, int):
            raise ValueError("patience must be an integer")
        if patience <= 0 or patience > max_epochs:
            raise ValueError("patience must be in [1, max_epochs]")
        if (
            isinstance(min_delta, bool)
            or not isinstance(min_delta, (int, float))
            or not math.isfinite(float(min_delta))
            or min_delta < 0
        ):
            raise ValueError("min_delta must be finite and non-negative")
        if q50_huber_delta is not None and (
            isinstance(q50_huber_delta, bool)
            or not isinstance(q50_huber_delta, (int, float))
            or not math.isfinite(float(q50_huber_delta))
            or q50_huber_delta <= 0
        ):
            raise ValueError("q50_huber_delta must be finite and positive when enabled")
        self.quantiles = normalized_quantiles
        self.hidden_dims = normalized_hidden
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.q50_huber_delta = (
            float(q50_huber_delta) if q50_huber_delta is not None else None
        )

    def fit(
        self,
        train: TrainingView,
        validation: TrainingView,
        context: FitContext,
    ) -> FittedIndependentMLP:
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
        if train.dataset_id != validation.dataset_id:
            raise ValueError("train and validation views belong to different datasets")
        if train.input_contract_hash != validation.input_contract_hash:
            raise ValueError("train and validation views have different input contracts")
        if train.position != validation.position or train.target != validation.target:
            raise ValueError("train and validation views must share position and target")
        train_conditions = _view_condition_ids(train, description="train")
        validation_conditions = _view_condition_ids(validation, description="validation")
        if train_conditions != validation_conditions:
            raise ValueError("train and validation views must share condition scope")
        np, torch = _load_neural_dependencies()
        seed = _derived_seed(context.seed, context.fold)
        _configure_deterministic_cpu(torch, seed)

        train_points = tuple(example.point for example in train.examples)
        validation_points = tuple(example.point for example in validation.examples)
        encoder = NeuralFeatureEncoder.fit(train_points)
        if encoder.schema.output_width <= 0:
            raise ValueError("Independent MLP requires at least one usable train-fold feature")
        architecture = MLPArchitecture(
            input_dim=encoder.schema.output_width,
            hidden_dims=self.hidden_dims,  # type: ignore[arg-type]
        )
        encoded_cells = (
            len(train_points) + len(validation_points)
        ) * architecture.input_dim
        if encoded_cells > MAX_MLP_ENCODED_CELLS:
            raise ValueError(
                "Independent MLP encoded matrices exceed the safe cell-count limit"
            )
        encoded_train = encoder.transform(train_points)
        encoded_validation = encoder.transform(validation_points)
        model = _build_network(torch, architecture)
        train_x = torch.from_numpy(encoded_train.matrix).to(dtype=torch.float32)
        validation_x = torch.from_numpy(encoded_validation.matrix).to(dtype=torch.float32)
        train_y = torch.tensor(
            [example.target_value for example in train.examples], dtype=torch.float32
        )
        validation_y = torch.tensor(
            [example.target_value for example in validation.examples], dtype=torch.float32
        )
        train_weight = torch.tensor(
            [example.sample_weight for example in train.examples], dtype=torch.float32
        )
        validation_weight = torch.tensor(
            [example.sample_weight for example in validation.examples], dtype=torch.float32
        )
        if not all(
            bool(torch.isfinite(tensor).all())
            for tensor in (
                train_x,
                validation_x,
                train_y,
                validation_y,
                train_weight,
                validation_weight,
            )
        ):
            raise ValueError("neural training tensors must be finite")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        history: list[float] = []
        best_loss = math.inf
        best_epoch = 0
        best_state: dict[str, Any] | None = None
        stale_epochs = 0
        for epoch in range(1, self.max_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            predictions = model(train_x)
            loss = _weighted_quantile_loss(
                torch,
                predictions,
                train_y,
                train_weight,
                resolved_quantiles,
                q50_huber_delta=self.q50_huber_delta,
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Independent MLP training loss became non-finite")
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.inference_mode():
                validation_loss = float(
                    _weighted_quantile_loss(
                        torch,
                        model(validation_x),
                        validation_y,
                        validation_weight,
                        resolved_quantiles,
                        q50_huber_delta=self.q50_huber_delta,
                    ).item()
                )
            if not math.isfinite(validation_loss):
                raise RuntimeError("Independent MLP validation loss became non-finite")
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
        if best_state is None or best_epoch <= 0:
            raise RuntimeError("Independent MLP did not produce a valid checkpoint")
        model.load_state_dict(best_state, strict=True)
        model.eval()
        parameters: dict[str, Any] = {
            "quantiles": list(resolved_quantiles),
            "hidden_dims": list(self.hidden_dims),
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "q50_huber_delta": self.q50_huber_delta,
            "device": "cpu",
            "deterministic": True,
            "num_threads": 1,
            "optimizer": "adamw",
            "activation": "silu",
        }
        report = MLPFitReport(
            estimator_version=INDEPENDENT_MLP_ESTIMATOR_VERSION,
            encoder_schema_hash=encoder.schema.content_hash,
            train_point_hash=_point_hash(tuple(point.point_id for point in train_points)),
            validation_point_hash=_point_hash(
                tuple(point.point_id for point in validation_points)
            ),
            train_point_count=len(train_points),
            validation_point_count=len(validation_points),
            seed=seed,
            best_epoch=best_epoch,
            best_validation_loss=best_loss,
            validation_history=tuple(history),
            parameters=parameters,
            torch_version=str(torch.__version__),
            numpy_version=str(np.__version__),
            platform=platform.platform(),
        )
        return FittedIndependentMLP(
            estimator_id=self.estimator_id,
            target=train.target,
            position=train.position,
            dataset_id=train.dataset_id,
            input_contract_hash=train.input_contract_hash,
            allowed_condition_ids=train_conditions,
            encoder=encoder,
            architecture=architecture,
            model=model,
            quantiles=resolved_quantiles,
            fit_report=report,
        )


__all__ = [
    "FittedIndependentMLP",
    "INDEPENDENT_MLP_ESTIMATOR_VERSION",
    "IndependentMLPQuantileEstimator",
    "IndependentMLPSession",
    "MAX_MLP_ENCODED_CELLS",
    "MAX_MLP_DIMENSION",
    "MAX_MLP_HIDDEN_WIDTH",
    "MAX_MLP_PARAMETERS",
    "MLPArchitecture",
    "MLPFitReport",
]
