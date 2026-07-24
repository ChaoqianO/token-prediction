from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Callable, Mapping

from token_prediction.contracts import SourceCapabilities
from token_prediction.dataset import PredictionPoint, PredictionTarget
from token_prediction.dataset.derived_features import (
    supported_input_contract_hashes_from_capability,
)
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


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        "online shadow fitted provenance contains a non-canonical value "
        f"of type {type(value).__name__}"
    )


def _semantic_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            _canonical_value(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _require_sha256(value: str, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def fitted_model_provenance_hash(fitted: FittedEstimator) -> str:
    """Return the online-safe identity of one fitted updater.

    Learned estimators must carry the provenance restored from their safe
    bundle.  Deterministic estimators without a ``provenance`` attribute are
    identified by their complete fitted scope.
    """

    estimator_id = getattr(fitted, "estimator_id", None)
    if not isinstance(estimator_id, str) or not estimator_id:
        raise ValueError("online shadow fitted estimator_id is missing")
    declared_provenance: Any = None
    if hasattr(fitted, "provenance"):
        declared_provenance = getattr(fitted, "provenance")
        if not isinstance(declared_provenance, Mapping) or not declared_provenance:
            raise ValueError(
                "online shadow learned estimator requires safe-bundle provenance"
            )
    scope: dict[str, Any] = {"estimator_id": estimator_id}
    for name in (
        "dataset_id",
        "input_contract_hash",
        "target",
        "position",
        "condition_id",
        "allowed_condition_ids",
        "architecture",
        "quantiles",
        "target_scale",
        "residual_scale",
        "no_recurrence",
        "fit_report",
    ):
        if hasattr(fitted, name):
            scope[name] = getattr(fitted, name)
    encoder = getattr(fitted, "encoder", None)
    encoder_schema = getattr(encoder, "schema", None)
    encoder_hash = getattr(encoder_schema, "content_hash", None)
    if encoder is not None:
        if not isinstance(encoder_hash, str):
            raise ValueError("online shadow fitted encoder identity is missing")
        _require_sha256(encoder_hash, name="fitted encoder content hash")
        scope["encoder_schema_hash"] = encoder_hash
    model = getattr(fitted, "model", None)
    if model is not None:
        state_dict_method = getattr(model, "state_dict", None)
        if not callable(state_dict_method):
            raise ValueError("online shadow fitted model state identity is missing")
        state = state_dict_method()
        if not isinstance(state, Mapping) or not state:
            raise ValueError("online shadow fitted model state is invalid")
        tensors: dict[str, object] = {}
        for name, tensor in sorted(state.items(), key=lambda pair: str(pair[0])):
            if not isinstance(name, str) or not name:
                raise ValueError("online shadow fitted model state key is invalid")
            try:
                normalized = tensor.detach().cpu().contiguous()
                payload = normalized.numpy().tobytes(order="C")
                shape = [int(value) for value in normalized.shape]
                dtype = str(normalized.dtype)
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"online shadow cannot fingerprint fitted tensor {name!r}"
                ) from exc
            tensors[name] = {
                "dtype": dtype,
                "shape": shape,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        scope["model_state"] = tensors
    model_strings_method = getattr(fitted, "model_strings", None)
    if callable(model_strings_method):
        model_strings = model_strings_method()
        if not isinstance(model_strings, Mapping) or not model_strings:
            raise ValueError("online shadow fitted booster identity is invalid")
        scope["model_strings_sha256"] = {
            str(name): hashlib.sha256(str(value).encode()).hexdigest()
            for name, value in sorted(
                model_strings.items(),
                key=lambda pair: str(pair[0]),
            )
        }
    if declared_provenance is not None:
        scope["bundle_provenance"] = declared_provenance
        calibrator = getattr(fitted, "calibrator_document", None)
        if calibrator is not None:
            if not isinstance(calibrator, Mapping):
                raise ValueError("online shadow fitted calibrator provenance is invalid")
            scope["calibrator"] = calibrator
    return _semantic_sha256(
        {
            "identity_schema_version": 1,
            "fitted_scope": scope,
        }
    )


@dataclass(frozen=True)
class OnlineShadowProvenance:
    """Frozen authorization binding for one label-free shadow session."""

    source_id: str
    capability_contract_hash: str
    dataset_id: str
    input_contract_hash: str
    condition_id: str
    target: PredictionTarget
    estimator_id: str
    fitted_model_provenance_hash: str
    seed_component_bundle_hashes: tuple[str, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported online shadow provenance schema version")
        for name in ("source_id", "dataset_id", "condition_id", "estimator_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"online shadow provenance {name} is required")
        for name in (
            "capability_contract_hash",
            "input_contract_hash",
            "fitted_model_provenance_hash",
        ):
            _require_sha256(getattr(self, name), name=name)
        if not isinstance(self.target, PredictionTarget):
            raise TypeError("online shadow provenance target must be a PredictionTarget")
        if (
            not isinstance(self.seed_component_bundle_hashes, tuple)
            or not self.seed_component_bundle_hashes
        ):
            raise ValueError(
                "online shadow provenance requires seed component bundle hashes"
            )
        for index, digest in enumerate(self.seed_component_bundle_hashes):
            _require_sha256(
                digest,
                name=f"seed_component_bundle_hashes[{index}]",
            )

    @property
    def contract_id(self) -> str:
        return _semantic_sha256(
            {
                "schema_version": self.schema_version,
                "source_id": self.source_id,
                "capability_contract_hash": self.capability_contract_hash,
                "dataset_id": self.dataset_id,
                "input_contract_hash": self.input_contract_hash,
                "condition_id": self.condition_id,
                "target": self.target.value,
                "estimator_id": self.estimator_id,
                "fitted_model_provenance_hash": (
                    self.fitted_model_provenance_hash
                ),
                "seed_component_bundle_hashes": list(
                    self.seed_component_bundle_hashes
                ),
            }
        )


def _validate_shadow_provenance(
    fitted: FittedEstimator,
    *,
    capabilities: SourceCapabilities,
    provenance: OnlineShadowProvenance,
    dataset_id: str,
    input_contract_hash: str,
    condition_id: str,
    task_pre_point: PredictionPoint,
    seed: SessionSeed,
) -> None:
    if capabilities.source_id != provenance.source_id:
        raise ValueError(
            "online shadow capabilities source_id differs from frozen provenance"
        )
    if capabilities.contract_hash != provenance.capability_contract_hash:
        raise ValueError(
            "online shadow capability contract differs from frozen provenance"
        )
    supported_contracts = supported_input_contract_hashes_from_capability(
        capabilities.contract_hash,
        capabilities=capabilities,
    )
    if input_contract_hash not in supported_contracts:
        raise ValueError(
            "online shadow input contract is not derived from the capability contract"
        )
    if dataset_id != provenance.dataset_id:
        raise ValueError("online shadow dataset differs from frozen provenance")
    if input_contract_hash != provenance.input_contract_hash:
        raise ValueError("online shadow input contract differs from frozen provenance")
    if condition_id != provenance.condition_id:
        raise ValueError("online shadow condition differs from frozen provenance")
    if task_pre_point.target != provenance.target:
        raise ValueError("online shadow target differs from frozen provenance")
    estimator_id = getattr(fitted, "estimator_id", None)
    if estimator_id != provenance.estimator_id:
        raise ValueError("online shadow estimator differs from frozen provenance")
    for name, expected in (
        ("dataset_id", dataset_id),
        ("input_contract_hash", input_contract_hash),
        ("target", provenance.target),
    ):
        actual = getattr(fitted, name, None)
        if actual != expected:
            raise ValueError(
                f"online shadow fitted {name} differs from frozen provenance"
            )
    fitted_condition = getattr(fitted, "condition_id", None)
    allowed_conditions = getattr(fitted, "allowed_condition_ids", ())
    if (
        fitted_condition != condition_id
        and condition_id not in allowed_conditions
    ):
        raise ValueError(
            "online shadow fitted condition differs from frozen provenance"
        )
    actual_model_hash = fitted_model_provenance_hash(fitted)
    if actual_model_hash != provenance.fitted_model_provenance_hash:
        raise ValueError(
            "online shadow fitted model provenance differs from frozen provenance"
        )
    if seed.component_bundle_hashes != provenance.seed_component_bundle_hashes:
        raise ValueError(
            "online shadow seed bundle identity differs from frozen provenance"
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
        provenance: OnlineShadowProvenance,
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
        if not isinstance(provenance, OnlineShadowProvenance):
            raise TypeError(
                "online shadow requires an OnlineShadowProvenance authorization"
            )
        _validate_shadow_provenance(
            fitted,
            capabilities=capabilities,
            provenance=provenance,
            dataset_id=dataset_id,
            input_contract_hash=input_contract_hash,
            condition_id=condition_id,
            task_pre_point=task_pre_point,
            seed=seed,
        )
        self.provenance = provenance
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
    "OnlineShadowProvenance",
    "OnlineShadowSession",
    "ShadowPrediction",
    "fitted_model_provenance_hash",
]
