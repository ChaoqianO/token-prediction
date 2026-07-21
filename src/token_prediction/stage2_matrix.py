"""Frozen, capability-gated Stage 2 experiment matrix.

The matrix is constructed only from a sealed ``DevelopmentProtocol``.  Final
holdout rows are therefore unavailable to condition gating, candidate routing,
and matrix identity.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from token_prediction.crossfit import SEED_POLICY_ID
from token_prediction.dataset import (
    INNER_FOLD_POLICY_ID,
    PredictionPosition,
    PredictionTarget,
)
from token_prediction.development import DevelopmentProtocol
from token_prediction.experiment import (
    CandidateGraph,
    CandidateRole,
    CandidateSpec,
    ExperimentSpec,
)
from token_prediction.features import FeatureSet, NO_FEATURES, REQUEST_CONTENT_CHARS_ONLY
from token_prediction.lifecycle_experiment import TASK_LIFECYCLE_SCHEMA_ID


STAGE2_MATRIX_SCHEMA_VERSION = 1
STAGE2_MATRIX_POLICY_ID = "stage2_condition_target_candidate_matrix_v1"
STAGE2_MIN_DEVELOPMENT_TASKS = 10
STAGE2_ALPHA = 0.10
STAGE2_CALIBRATOR_ID = "task_max_conformal"

BAGEN_SOURCE_ID = "bagen_swebench_traj_v2"
SPEND_SOURCE_ID = "openhands_archive_trajectory_v3"

FROZEN_BAGEN_CONDITIONS = frozenset(
    {
        "condition:20f615a22697984db6cc",
        "condition:54cb50fce273f0aa2d74",
        "condition:562b4f6934238e459db9",
        "condition:686d78e7865f5e646e0b",
        "condition:8fe0be8b5f924006a166",
        "condition:949ac3b7a342718cd505",
        "condition:d94078c05d91b0d58aee",
        "condition:dce86ced00dc11c77205",
        "condition:f95ae2a5e11682f6b7fc",
    }
)
FROZEN_SPEND_CONDITIONS = frozenset({"condition:b407e0d1ec34f386ebc4"})
FROZEN_SOURCE_CONDITIONS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        BAGEN_SOURCE_ID: FROZEN_BAGEN_CONDITIONS,
        SPEND_SOURCE_ID: FROZEN_SPEND_CONDITIONS,
    }
)

STRUCTURED_FEATURE_NAMES = frozenset(
    {
        "task_tokens",
        "max_steps",
        "model_id",
        "agent_id",
        "reasoning_effort",
        "request_message_count",
        "request_content_chars",
    }
)
HISTORY_FEATURE_NAMES = STRUCTURED_FEATURE_NAMES | frozenset(
    {
        "completed_call_count",
        "completed_api_attempts",
        "failed_api_attempts",
        "completed_tool_calls",
        "failed_tool_calls",
        "known_usage_attempts",
        "missing_usage_attempts",
        "request_count",
        "step_progress_ratio",
        "cumulative_provider_input_tokens",
        "cumulative_provider_output_tokens",
        "last_call_output_tokens",
        "recent_generated_mean_3",
        "last_tool_type",
        "last_round_tool_error_count",
        "consecutive_error_rounds",
        "repeated_action_count_3",
    }
)
STAGE2_STRUCTURED_FEATURES = FeatureSet(
    "stage2_structured",
    include_all=False,
    include_features=STRUCTURED_FEATURE_NAMES,
)
STAGE2_HISTORY_FEATURES = FeatureSet(
    "stage2_structured_history",
    include_all=False,
    include_features=HISTORY_FEATURE_NAMES,
)

_CELL_TEMPLATES = (
    (
        PredictionPosition.TASK_LAUNCH,
        PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
    ),
    (
        PredictionPosition.TASK_UPDATE,
        PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
    ),
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
class Stage2Gate:
    source_id: str
    condition_id: str
    position: PredictionPosition
    target: PredictionTarget
    scope: str
    reason: str
    development_task_count: int
    eligible_point_count: int
    candidate_id: str | None = None

    def __post_init__(self) -> None:
        if self.scope not in {"cell", "candidate"}:
            raise ValueError("Stage 2 gate scope must be 'cell' or 'candidate'")
        if not self.source_id or not self.condition_id or not self.reason:
            raise ValueError("Stage 2 gate identities and reason are required")
        if self.development_task_count < 0 or self.eligible_point_count < 0:
            raise ValueError("Stage 2 gate counts must be non-negative")
        if (self.scope == "candidate") != (self.candidate_id is not None):
            raise ValueError("candidate gate scope requires exactly one candidate_id")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "condition_id": self.condition_id,
            "position": self.position.value,
            "target": self.target.value,
            "scope": self.scope,
            "candidate_id": self.candidate_id,
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
        "graph": candidate.graph.to_dict(),
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
class Stage2Matrix:
    source_id: str
    development_protocol_id: str
    experiments: tuple[ExperimentSpec, ...]
    gates: tuple[Stage2Gate, ...]
    matrix_id: str
    schema_version: int = STAGE2_MATRIX_SCHEMA_VERSION
    policy_id: str = STAGE2_MATRIX_POLICY_ID

    def __post_init__(self) -> None:
        if self.schema_version != STAGE2_MATRIX_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 2 matrix schema version")
        if self.policy_id != STAGE2_MATRIX_POLICY_ID:
            raise ValueError("unsupported Stage 2 matrix policy")
        if self.source_id not in FROZEN_SOURCE_CONDITIONS:
            raise ValueError("Stage 2 matrix source is not frozen")
        if not self.development_protocol_id:
            raise ValueError("Stage 2 matrix requires a development protocol id")
        experiment_ids = [spec.experiment_id for spec in self.experiments]
        if len(experiment_ids) != len(set(experiment_ids)):
            raise ValueError("Stage 2 experiment ids must be unique")
        expected = _canonical_sha256(self.identity_document(include_matrix_id=False))
        if self.matrix_id != expected:
            raise ValueError("Stage 2 matrix id does not match its semantics")

    def identity_document(self, *, include_matrix_id: bool = True) -> dict[str, object]:
        document: dict[str, object] = {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "source_id": self.source_id,
            "development_protocol_id": self.development_protocol_id,
            "minimum_development_tasks": STAGE2_MIN_DEVELOPMENT_TASKS,
            "experiments": [_spec_document(spec) for spec in self.experiments],
            "gates": [gate.to_dict() for gate in self.gates],
        }
        if include_matrix_id:
            document["matrix_id"] = self.matrix_id
        return document


def _mlp_params() -> dict[str, object]:
    return {
        "quantiles": [STAGE2_ALPHA / 2, 0.5, 1 - STAGE2_ALPHA / 2],
        "hidden_dims": [128, 64],
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "max_epochs": 200,
        "patience": 20,
        "min_delta": 0.0,
        "q50_huber_delta": None,
    }


def _lightgbm_params() -> dict[str, object]:
    return {
        "quantiles": [STAGE2_ALPHA / 2, 0.5, 1 - STAGE2_ALPHA / 2],
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


def _common_point_candidates() -> tuple[CandidateSpec, ...]:
    return (
        CandidateSpec(
            "empirical",
            "empirical_quantile",
            NO_FEATURES,
            params={"alpha": STAGE2_ALPHA},
            role=CandidateRole.BASELINE,
        ),
        CandidateSpec(
            "lightgbm_structured",
            "lightgbm_quantile",
            STAGE2_STRUCTURED_FEATURES,
            params=_lightgbm_params(),
        ),
        CandidateSpec(
            "lightgbm_history",
            "lightgbm_quantile",
            STAGE2_HISTORY_FEATURES,
            params=_lightgbm_params(),
        ),
        CandidateSpec(
            "mlp_structured",
            "independent_mlp",
            STAGE2_STRUCTURED_FEATURES,
            params=_mlp_params(),
        ),
        CandidateSpec(
            "mlp_history",
            "independent_mlp",
            STAGE2_HISTORY_FEATURES,
            params=_mlp_params(),
        ),
    )


def _update_candidates(
    *,
    condition_id: str,
    input_contract_hash: str,
    include_request_chars: bool,
) -> tuple[CandidateSpec, ...]:
    candidates = list(_common_point_candidates())
    candidates.insert(
        1,
        CandidateSpec(
            "within_cell_deduct",
            "deduct_only",
            NO_FEATURES,
            params={"alpha": STAGE2_ALPHA},
            role=CandidateRole.BASELINE,
        ),
    )
    candidates.insert(
        2,
        CandidateSpec(
            "cross_position_deduct",
            "cross_position_deduct",
            NO_FEATURES,
            params={
                "expected_condition_id": condition_id,
                "expected_input_contract_hash": input_contract_hash,
            },
            initializer_params={"alpha": STAGE2_ALPHA},
            graph=CandidateGraph(
                initializer_estimator_id="empirical_quantile",
                updater_estimator_id="cross_position_deduct",
                lifecycle_schema_id=TASK_LIFECYCLE_SCHEMA_ID,
                seed_policy_id=SEED_POLICY_ID,
                inner_split_policy_id=INNER_FOLD_POLICY_ID,
            ),
            role=CandidateRole.BASELINE,
        ),
    )
    if include_request_chars:
        candidates.insert(
            1,
            CandidateSpec(
                "request_chars_length",
                "length_only",
                REQUEST_CONTENT_CHARS_ONLY,
                params={
                    "feature_name": "request_content_chars",
                    "alpha": STAGE2_ALPHA,
                },
                role=CandidateRole.BASELINE,
            ),
        )
    return tuple(candidates)


def _has_complete_request_chars(rows: Sequence[object]) -> bool:
    if not rows:
        return False
    for row in rows:
        value = row.point.features.get("request_content_chars")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            return False
    return True


def build_stage2_matrix(
    protocol: DevelopmentProtocol,
    *,
    source_id: str,
) -> Stage2Matrix:
    """Build the frozen Stage 2 matrix from development rows only."""

    try:
        conditions = FROZEN_SOURCE_CONDITIONS[source_id]
    except KeyError as exc:
        raise ValueError(f"unsupported Stage 2 source_id {source_id!r}") from exc
    dataset = protocol.development_dataset
    if dataset.task_ids & protocol.final_holdout_tasks:
        raise ValueError("final-holdout tasks leaked into the Stage 2 matrix input")
    if dataset.input_contract_hash is None:
        raise ValueError("Stage 2 matrix requires an input contract hash")

    experiments: list[ExperimentSpec] = []
    gates: list[Stage2Gate] = []
    for condition_id in sorted(conditions):
        for position, target in _CELL_TEMPLATES:
            cell = dataset.select(position, target, condition_id=condition_id)
            tasks = {row.point.task_id for row in cell.rows}
            task_count = len(tasks)
            point_count = len(cell.rows)
            if task_count < STAGE2_MIN_DEVELOPMENT_TASKS:
                reason = (
                    "capability_or_observed_development_cell_unavailable"
                    if point_count == 0
                    else "insufficient_development_tasks_for_five_fold_cv"
                )
                gates.append(
                    Stage2Gate(
                        source_id=source_id,
                        condition_id=condition_id,
                        position=position,
                        target=target,
                        scope="cell",
                        reason=reason,
                        development_task_count=task_count,
                        eligible_point_count=point_count,
                    )
                )
                continue

            if position == PredictionPosition.TASK_LAUNCH:
                candidates = _common_point_candidates()
                gates.append(
                    Stage2Gate(
                        source_id=source_id,
                        condition_id=condition_id,
                        position=position,
                        target=target,
                        scope="candidate",
                        candidate_id="request_chars_length",
                        reason="no_prefix_causal_length_feature_at_task_launch",
                        development_task_count=task_count,
                        eligible_point_count=point_count,
                    )
                )
            else:
                complete_request_chars = _has_complete_request_chars(cell.rows)
                candidates = _update_candidates(
                    condition_id=condition_id,
                    input_contract_hash=dataset.input_contract_hash,
                    include_request_chars=complete_request_chars,
                )
                if not complete_request_chars:
                    gates.append(
                        Stage2Gate(
                            source_id=source_id,
                            condition_id=condition_id,
                            position=position,
                            target=target,
                            scope="candidate",
                            candidate_id="request_chars_length",
                            reason="request_content_chars_missing_on_scored_cohort",
                            development_task_count=task_count,
                            eligible_point_count=point_count,
                        )
                    )
            experiments.append(
                ExperimentSpec(
                    experiment_id=(
                        f"stage2-{source_id}-{condition_id.removeprefix('condition:')}-"
                        f"{position.value}-{target.value}"
                    ),
                    position=position,
                    target=target,
                    candidates=candidates,
                    alpha=STAGE2_ALPHA,
                    calibrator_id=STAGE2_CALIBRATOR_ID,
                    condition_id=condition_id,
                )
            )

    ordered_experiments = tuple(
        sorted(experiments, key=lambda item: item.experiment_id)
    )
    ordered_gates = tuple(
        sorted(
            gates,
            key=lambda item: (
                item.condition_id,
                item.position.value,
                item.target.value,
                item.scope,
                item.candidate_id or "",
            ),
        )
    )
    semantic = {
        "schema_version": STAGE2_MATRIX_SCHEMA_VERSION,
        "policy_id": STAGE2_MATRIX_POLICY_ID,
        "source_id": source_id,
        "development_protocol_id": protocol.protocol_id,
        "minimum_development_tasks": STAGE2_MIN_DEVELOPMENT_TASKS,
        "experiments": [_spec_document(spec) for spec in ordered_experiments],
        "gates": [gate.to_dict() for gate in ordered_gates],
    }
    return Stage2Matrix(
        source_id=source_id,
        development_protocol_id=protocol.protocol_id,
        experiments=ordered_experiments,
        gates=ordered_gates,
        matrix_id=_canonical_sha256(semantic),
    )


__all__ = [
    "BAGEN_SOURCE_ID",
    "FROZEN_BAGEN_CONDITIONS",
    "FROZEN_SOURCE_CONDITIONS",
    "FROZEN_SPEND_CONDITIONS",
    "SPEND_SOURCE_ID",
    "STAGE2_ALPHA",
    "STAGE2_CALIBRATOR_ID",
    "STAGE2_HISTORY_FEATURES",
    "STAGE2_MATRIX_POLICY_ID",
    "STAGE2_MATRIX_SCHEMA_VERSION",
    "STAGE2_MIN_DEVELOPMENT_TASKS",
    "STAGE2_STRUCTURED_FEATURES",
    "Stage2Gate",
    "Stage2Matrix",
    "build_stage2_matrix",
]
