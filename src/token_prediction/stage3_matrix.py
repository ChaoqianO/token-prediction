"""Frozen, development-only Stage 3 recurrent lifecycle matrix."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from token_prediction.crossfit import SEED_POLICY_ID
from token_prediction.dataset import (
    INNER_FOLD_POLICY_ID,
    PredictionPosition,
    PredictionTarget,
)
from token_prediction.development import DevelopmentProtocol
from token_prediction.experiment import (
    AblationAxis,
    AblationSpec,
    CandidateGraph,
    CandidateRole,
    CandidateSpec,
    ExperimentSpec,
    validate_ablation_specs,
)
from token_prediction.features import NO_FEATURES
from token_prediction.lifecycle_experiment import TASK_LIFECYCLE_SCHEMA_ID
from token_prediction.stage2_matrix import (
    BAGEN_SOKOBAN_SOURCE_ID,
    BAGEN_SOURCE_ID,
    FROZEN_SOURCE_CONDITIONS,
    SPEND_AGGREGATE_SOURCE_ID,
    SPEND_SOURCE_ID,
    STAGE2_HISTORY_FEATURES,
)


STAGE3_MATRIX_SCHEMA_VERSION = 1
STAGE3_MATRIX_POLICY_ID = "stage3_gru_lifecycle_ablation_matrix_v1"
STAGE3_MIN_DEVELOPMENT_TASKS = 10
STAGE3_ALPHA = 0.10
STAGE3_CALIBRATOR_ID = "task_max_conformal"
STAGE3_BUDGET_THRESHOLDS = (16_384, 32_768, 65_536, 131_072)

FROZEN_STAGE3_SOURCE_CONDITIONS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        source_id: conditions
        for source_id, conditions in FROZEN_SOURCE_CONDITIONS.items()
        if source_id
        in {
            BAGEN_SOURCE_ID,
            BAGEN_SOKOBAN_SOURCE_ID,
            SPEND_SOURCE_ID,
            SPEND_AGGREGATE_SOURCE_ID,
        }
    }
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class Stage3Gate:
    source_id: str
    condition_id: str
    reason: str
    development_task_count: int
    eligible_point_count: int

    def __post_init__(self) -> None:
        if not self.source_id or not self.condition_id or not self.reason:
            raise ValueError("Stage 3 gate identities and reason are required")
        if self.development_task_count < 0 or self.eligible_point_count < 0:
            raise ValueError("Stage 3 gate counts must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "condition_id": self.condition_id,
            "position": PredictionPosition.TASK_UPDATE.value,
            "target": (
                PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS.value
            ),
            "reason": self.reason,
            "development_task_count": self.development_task_count,
            "eligible_point_count": self.eligible_point_count,
        }


def _candidate_document(candidate: CandidateSpec) -> dict[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "candidate_hash": candidate.content_hash,
        "estimator_id": candidate.estimator_id,
        "role": candidate.role.value,
        "feature_set_id": candidate.feature_set.feature_set_id,
        "feature_set_hash": candidate.feature_set.content_hash,
        "params": dict(candidate.params),
        "initializer_params": dict(candidate.initializer_params),
        "graph": candidate.graph.to_dict(),
        "ablation": (
            {
                "reference_candidate_id": candidate.ablation.reference_candidate_id,
                "axis": candidate.ablation.axis.value,
                "allowed_config_paths": sorted(candidate.ablation.allowed_config_paths),
            }
            if candidate.ablation is not None
            else None
        ),
    }


def _spec_document(spec: ExperimentSpec) -> dict[str, object]:
    return {
        "experiment_id": spec.experiment_id,
        "position": spec.position.value,
        "target": spec.target.value,
        "condition_id": spec.condition_id,
        "alpha": spec.alpha,
        "calibrator_id": spec.calibrator_id,
        "required_features": sorted(spec.required_features),
        "candidates": [_candidate_document(candidate) for candidate in spec.candidates],
    }


@dataclass(frozen=True)
class Stage3Matrix:
    source_id: str
    development_protocol_id: str
    experiments: tuple[ExperimentSpec, ...]
    gates: tuple[Stage3Gate, ...]
    matrix_id: str
    schema_version: int = STAGE3_MATRIX_SCHEMA_VERSION
    policy_id: str = STAGE3_MATRIX_POLICY_ID

    def __post_init__(self) -> None:
        if self.schema_version != STAGE3_MATRIX_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 3 matrix schema version")
        if self.policy_id != STAGE3_MATRIX_POLICY_ID:
            raise ValueError("unsupported Stage 3 matrix policy")
        if self.source_id not in FROZEN_STAGE3_SOURCE_CONDITIONS:
            raise ValueError("Stage 3 matrix source is not frozen")
        if not self.development_protocol_id:
            raise ValueError("Stage 3 matrix requires a development protocol id")
        experiment_ids = [spec.experiment_id for spec in self.experiments]
        if len(experiment_ids) != len(set(experiment_ids)):
            raise ValueError("Stage 3 experiment ids must be unique")
        expected = _canonical_sha256(self.identity_document(include_matrix_id=False))
        if self.matrix_id != expected:
            raise ValueError("Stage 3 matrix id does not match its semantics")

    def identity_document(self, *, include_matrix_id: bool = True) -> dict[str, object]:
        document: dict[str, object] = {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "source_id": self.source_id,
            "development_protocol_id": self.development_protocol_id,
            "minimum_development_tasks": STAGE3_MIN_DEVELOPMENT_TASKS,
            "budget_thresholds": list(STAGE3_BUDGET_THRESHOLDS),
            "experiments": [_spec_document(spec) for spec in self.experiments],
            "gates": [gate.to_dict() for gate in self.gates],
        }
        if include_matrix_id:
            document["matrix_id"] = self.matrix_id
        return document


def _lightgbm_params() -> dict[str, object]:
    return {
        "quantiles": [STAGE3_ALPHA / 2, 0.5, 1 - STAGE3_ALPHA / 2],
        "num_boost_round": 500,
        "early_stopping_rounds": 30,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 5,
        "max_depth": -1,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "max_bin": 255,
    }


def _mlp_params() -> dict[str, object]:
    return {
        "quantiles": [STAGE3_ALPHA / 2, 0.5, 1 - STAGE3_ALPHA / 2],
        "hidden_dims": [128, 64],
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "max_epochs": 200,
        "patience": 20,
        "min_delta": 0.0,
        "q50_huber_delta": None,
    }


def _gru_params() -> dict[str, object]:
    return {
        "quantiles": [STAGE3_ALPHA / 2, 0.5, 1 - STAGE3_ALPHA / 2],
        "transition_dim": 64,
        "hidden_dim": 64,
        "residual_head_dim": 64,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "max_epochs": 200,
        "patience": 20,
        "min_delta": 0.0,
        "q50_huber_delta": None,
        "residual_scale": 1.0,
        "no_recurrence": False,
    }


def _lifecycle_graph(updater_id: str) -> CandidateGraph:
    return CandidateGraph(
        initializer_estimator_id="empirical_quantile",
        updater_estimator_id=updater_id,
        lifecycle_schema_id=TASK_LIFECYCLE_SCHEMA_ID,
        seed_policy_id=SEED_POLICY_ID,
        inner_split_policy_id=INNER_FOLD_POLICY_ID,
    )


def _stage3_candidates(
    *,
    condition_id: str,
    input_contract_hash: str,
) -> tuple[CandidateSpec, ...]:
    gru_params = _gru_params()
    no_recurrence = {**gru_params, "no_recurrence": True}
    zero_residual = {**gru_params, "residual_scale": 0.0}
    candidates = (
        CandidateSpec(
            "empirical",
            "empirical_quantile",
            NO_FEATURES,
            params={"alpha": STAGE3_ALPHA},
            role=CandidateRole.BASELINE,
        ),
        CandidateSpec(
            "cross_position_deduct",
            "cross_position_deduct",
            NO_FEATURES,
            params={
                "expected_condition_id": condition_id,
                "expected_input_contract_hash": input_contract_hash,
            },
            initializer_params={"alpha": STAGE3_ALPHA},
            graph=_lifecycle_graph("cross_position_deduct"),
            role=CandidateRole.BASELINE,
        ),
        CandidateSpec(
            "lightgbm_history",
            "lightgbm_quantile",
            STAGE2_HISTORY_FEATURES,
            params=_lightgbm_params(),
        ),
        CandidateSpec(
            "mlp_history",
            "independent_mlp",
            STAGE2_HISTORY_FEATURES,
            params=_mlp_params(),
        ),
        CandidateSpec(
            "gru_residual",
            "gru_residual",
            STAGE2_HISTORY_FEATURES,
            params=gru_params,
            initializer_params={"alpha": STAGE3_ALPHA},
            graph=_lifecycle_graph("gru_residual"),
        ),
        CandidateSpec(
            "gru_no_recurrence",
            "gru_residual",
            STAGE2_HISTORY_FEATURES,
            params=no_recurrence,
            initializer_params={"alpha": STAGE3_ALPHA},
            graph=_lifecycle_graph("gru_residual"),
            role=CandidateRole.ABLATION,
            ablation=AblationSpec(
                reference_candidate_id="gru_residual",
                axis=AblationAxis.STATE_UPDATE,
                allowed_config_paths=frozenset({"params.no_recurrence"}),
            ),
        ),
        CandidateSpec(
            "gru_zero_residual",
            "gru_residual",
            STAGE2_HISTORY_FEATURES,
            params=zero_residual,
            initializer_params={"alpha": STAGE3_ALPHA},
            graph=_lifecycle_graph("gru_residual"),
            role=CandidateRole.ABLATION,
            ablation=AblationSpec(
                reference_candidate_id="gru_residual",
                axis=AblationAxis.STATE_UPDATE,
                allowed_config_paths=frozenset({"params.residual_scale"}),
            ),
        ),
    )
    validate_ablation_specs(candidates)
    return candidates


def build_stage3_matrix(
    protocol: DevelopmentProtocol,
    *,
    source_id: str,
) -> Stage3Matrix:
    """Build Stage 3 exclusively from the sealed development dataset."""

    try:
        conditions = FROZEN_STAGE3_SOURCE_CONDITIONS[source_id]
    except KeyError as exc:
        raise ValueError(f"unsupported Stage 3 source_id {source_id!r}") from exc
    dataset = protocol.development_dataset
    if dataset.task_ids & protocol.final_holdout_tasks:
        raise ValueError("final-holdout tasks leaked into the Stage 3 matrix input")
    if dataset.input_contract_hash is None:
        raise ValueError("Stage 3 matrix requires an input contract hash")

    experiments: list[ExperimentSpec] = []
    gates: list[Stage3Gate] = []
    for condition_id in sorted(conditions):
        cell = dataset.select(
            PredictionPosition.TASK_UPDATE,
            PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
            condition_id=condition_id,
        )
        tasks = {row.point.task_id for row in cell.rows}
        task_count = len(tasks)
        point_count = len(cell.rows)
        if source_id == SPEND_AGGREGATE_SOURCE_ID:
            reason = "aggregate_source_has_no_request_boundary_lifecycle"
        elif task_count < STAGE3_MIN_DEVELOPMENT_TASKS:
            reason = (
                "capability_or_observed_development_lifecycle_unavailable"
                if point_count == 0
                else "insufficient_development_tasks_for_five_fold_cv"
            )
        else:
            reason = ""
        if reason:
            gates.append(
                Stage3Gate(
                    source_id=source_id,
                    condition_id=condition_id,
                    reason=reason,
                    development_task_count=task_count,
                    eligible_point_count=point_count,
                )
            )
            continue
        experiments.append(
            ExperimentSpec(
                experiment_id=(
                    f"stage3-{source_id}-{condition_id.removeprefix('condition:')}-"
                    "task_update-task_provider_accounted_remaining_tokens"
                ),
                position=PredictionPosition.TASK_UPDATE,
                target=(
                    PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS
                ),
                candidates=_stage3_candidates(
                    condition_id=condition_id,
                    input_contract_hash=dataset.input_contract_hash,
                ),
                alpha=STAGE3_ALPHA,
                calibrator_id=STAGE3_CALIBRATOR_ID,
                condition_id=condition_id,
            )
        )

    ordered_experiments = tuple(sorted(experiments, key=lambda item: item.experiment_id))
    ordered_gates = tuple(sorted(gates, key=lambda item: item.condition_id))
    semantic = {
        "schema_version": STAGE3_MATRIX_SCHEMA_VERSION,
        "policy_id": STAGE3_MATRIX_POLICY_ID,
        "source_id": source_id,
        "development_protocol_id": protocol.protocol_id,
        "minimum_development_tasks": STAGE3_MIN_DEVELOPMENT_TASKS,
        "budget_thresholds": list(STAGE3_BUDGET_THRESHOLDS),
        "experiments": [_spec_document(spec) for spec in ordered_experiments],
        "gates": [gate.to_dict() for gate in ordered_gates],
    }
    return Stage3Matrix(
        source_id=source_id,
        development_protocol_id=protocol.protocol_id,
        experiments=ordered_experiments,
        gates=ordered_gates,
        matrix_id=_canonical_sha256(semantic),
    )


__all__ = [
    "FROZEN_STAGE3_SOURCE_CONDITIONS",
    "STAGE3_ALPHA",
    "STAGE3_BUDGET_THRESHOLDS",
    "STAGE3_CALIBRATOR_ID",
    "STAGE3_MATRIX_POLICY_ID",
    "STAGE3_MATRIX_SCHEMA_VERSION",
    "STAGE3_MIN_DEVELOPMENT_TASKS",
    "Stage3Gate",
    "Stage3Matrix",
    "build_stage3_matrix",
]
