"""Capability-gated Stage 4 ablation and multi-position experiment matrix."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from token_prediction.contracts import Observable, SourceCapabilities
from token_prediction.crossfit import (
    POINT_ONLY_SEED_POLICY_ID,
    SEED_POLICY_ID,
    seed_policy_hash,
)
from token_prediction.dataset import (
    INNER_FOLD_POLICY_ID,
    PredictionPosition,
    PredictionTarget,
)
from token_prediction.dataset.capabilities import decide_target_capability
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
from token_prediction.features import NO_FEATURES, FeatureSet
from token_prediction.lifecycle_experiment import TASK_LIFECYCLE_SCHEMA_ID
from token_prediction.stage2_matrix import (
    BAGEN_SOKOBAN_SOURCE_ID,
    BAGEN_SOURCE_ID,
    FROZEN_SOURCE_CONDITIONS,
    HISTORY_FEATURE_NAMES,
    SPEND_AGGREGATE_SOURCE_ID,
    SPEND_AGGREGATE_STRUCTURED_FEATURES,
    SPEND_AGGREGATE_TASK_CHARS,
    SPEND_SOURCE_ID,
    STAGE2_HISTORY_FEATURES,
    STAGE2_STRUCTURED_FEATURES,
)
from token_prediction.telemetry import (
    TELEMETRY_REQUIREMENTS,
    TelemetryDecision,
    TelemetrySurface,
    decide_telemetry_surface,
)


STAGE4_MATRIX_SCHEMA_VERSION = 2
STAGE4_MATRIX_POLICY_ID = "stage4_single_axis_condition_position_target_matrix_v2"
STAGE4_MIN_DEVELOPMENT_TASKS = 10
STAGE4_ALPHA = 0.10
STAGE4_PRIMARY_CALIBRATOR_ID = "task_max_conformal"
STAGE4_MISSING_MASK_INVARIANT_ID = "explicit_missing_telemetry_masks_required_v1"
STAGE4_CALL_PRE_TARGETS = (
    PredictionTarget.CALL_BILLABLE_TOTAL_TOKENS,
    PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS,
    PredictionTarget.CALL_FINAL_RESPONSE_OUTPUT_TOKENS,
)
STAGE4_TELEMETRY_SURFACES = tuple(TelemetrySurface)
STAGE4_G3_REQUIRED_OBSERVABLES = frozenset(
    TELEMETRY_REQUIREMENTS[TelemetrySurface.CALL_UPDATE]
    | TELEMETRY_REQUIREMENTS[TelemetrySurface.G3_ENTROPY_STOP]
    | TELEMETRY_REQUIREMENTS[TelemetrySurface.G3_HIDDEN_STATE]
)

_PROGRESS_FEATURES = frozenset(
    {
        "completed_call_count",
        "completed_api_attempts",
        "known_usage_attempts",
        "request_count",
        "step_progress_ratio",
        "cumulative_provider_input_tokens",
        "cumulative_provider_output_tokens",
        "last_call_output_tokens",
        "recent_generated_mean_3",
    }
)
_TOOLS_ERRORS_FEATURES = frozenset(
    {
        "failed_api_attempts",
        "completed_tool_calls",
        "failed_tool_calls",
        "last_tool_type",
        "last_round_tool_error_count",
        "consecutive_error_rounds",
        "repeated_action_count_3",
    }
)
STAGE4_NO_PROGRESS_FEATURES = FeatureSet(
    "stage4_history_without_progress",
    include_all=False,
    include_features=HISTORY_FEATURE_NAMES - _PROGRESS_FEATURES,
)
STAGE4_NO_TOOLS_ERRORS_FEATURES = FeatureSet(
    "stage4_history_without_tools_errors",
    include_all=False,
    include_features=HISTORY_FEATURE_NAMES - _TOOLS_ERRORS_FEATURES,
)
STAGE4_PRE_REQUEST_CHAR_MESSAGE_FEATURES = FeatureSet(
    "stage4_pre_request_char_message_length",
    include_all=False,
    include_features=frozenset(
        {
            "request_content_chars",
            "request_message_count",
        }
    ),
)
STAGE4_RETRIEVAL_HISTORY_FEATURES = FeatureSet(
    "stage4_history_with_fold_fitted_retrieval",
    include_all=False,
    include_features=HISTORY_FEATURE_NAMES
    | frozenset(
        {
            "similar_task_total_tokens_median",
            "similar_task_total_tokens_iqr",
            "similar_task_call_count_median",
            "similar_task_mean_similarity",
        }
    ),
)
STAGE4_G3_FEATURES = FeatureSet(
    "stage4_generation_checkpoint",
    include_all=False,
    include_features=HISTORY_FEATURE_NAMES
    | frozenset(
        {
            "generated_tokens_so_far",
            "stop_prob_mean_16",
            "next_token_entropy_mean_16",
            "hidden_state_projection",
        }
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
        ).encode()
    ).hexdigest()


def _lightgbm_params() -> dict[str, object]:
    return {
        "quantiles": [STAGE4_ALPHA / 2, 0.5, 1 - STAGE4_ALPHA / 2],
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
        "quantiles": [STAGE4_ALPHA / 2, 0.5, 1 - STAGE4_ALPHA / 2],
        "hidden_dims": [128, 64],
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "max_epochs": 200,
        "patience": 20,
        "min_delta": 0.0,
        "q50_huber_delta": None,
        "training_device": "cuda",
    }


def _method_ablation_paths(
    reference_params: Mapping[str, object],
    candidate_params: Mapping[str, object],
) -> frozenset[str]:
    changed_params = {
        f"params.{key}"
        for key in set(reference_params) | set(candidate_params)
        if reference_params.get(key) != candidate_params.get(key)
    }
    return frozenset(
        {
            "estimator_id",
            "graph.updater_estimator_id",
            *changed_params,
        }
    )


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
        "seed_policy_hash": (
            seed_policy_hash(candidate.graph.seed_policy_id)
            if candidate.graph.is_lifecycle
            else None
        ),
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


def _spec_document(
    spec: ExperimentSpec,
    *,
    include_experiment_id: bool = True,
) -> dict[str, object]:
    value: dict[str, object] = {
        "position": spec.position.value,
        "target": spec.target.value,
        "condition_id": spec.condition_id,
        "alpha": spec.alpha,
        "calibrator_id": spec.calibrator_id,
        "required_features": sorted(spec.required_features),
        "candidates": [_candidate_document(candidate) for candidate in spec.candidates],
    }
    if include_experiment_id:
        value["experiment_id"] = spec.experiment_id
    return value


class Stage4PlanRole(StrEnum):
    PRIMARY = "primary"
    ABLATION = "ablation"


@dataclass(frozen=True)
class Stage4ExperimentPlan:
    spec: ExperimentSpec
    role: Stage4PlanRole = Stage4PlanRole.PRIMARY
    reference_experiment_id: str | None = None
    axis: AblationAxis | None = None
    allowed_config_paths: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.role == Stage4PlanRole.PRIMARY:
            if (
                self.reference_experiment_id is not None
                or self.axis is not None
                or self.allowed_config_paths
            ):
                raise ValueError("primary Stage 4 plans cannot declare an ablation")
        elif (
            not self.reference_experiment_id
            or self.axis is None
            or not self.allowed_config_paths
        ):
            raise ValueError("Stage 4 ablation plans require reference, axis, and paths")

    def to_dict(self) -> dict[str, object]:
        return {
            "spec": _spec_document(self.spec),
            "role": self.role.value,
            "reference_experiment_id": self.reference_experiment_id,
            "axis": self.axis.value if self.axis is not None else None,
            "allowed_config_paths": sorted(self.allowed_config_paths),
        }


@dataclass(frozen=True)
class Stage4Gate:
    source_id: str
    condition_id: str
    surface: str
    reason: str
    capability_contract_hash: str
    development_task_count: int
    eligible_point_count: int
    position: PredictionPosition | None = None
    target: PredictionTarget | None = None

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (
                self.source_id,
                self.condition_id,
                self.surface,
                self.reason,
                self.capability_contract_hash,
            )
        ):
            raise ValueError("Stage 4 gate identities and reason are required")
        if self.development_task_count < 0 or self.eligible_point_count < 0:
            raise ValueError("Stage 4 gate counts must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "condition_id": self.condition_id,
            "surface": self.surface,
            "position": self.position.value if self.position is not None else None,
            "target": self.target.value if self.target is not None else None,
            "reason": self.reason,
            "capability_contract_hash": self.capability_contract_hash,
            "development_task_count": self.development_task_count,
            "eligible_point_count": self.eligible_point_count,
        }


@dataclass(frozen=True)
class Stage4SafetyInvariant:
    invariant_id: str
    estimator_ids: tuple[str, ...]
    required_behavior: str
    prohibited_ablation: str
    violation_action: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (
                self.invariant_id,
                self.required_behavior,
                self.prohibited_ablation,
                self.violation_action,
            )
        ):
            raise ValueError("Stage 4 safety invariant fields are required")
        if (
            not self.estimator_ids
            or self.estimator_ids != tuple(sorted(set(self.estimator_ids)))
            or any(not estimator_id for estimator_id in self.estimator_ids)
        ):
            raise ValueError("Stage 4 safety invariant estimators must be canonical")

    def to_dict(self) -> dict[str, object]:
        return {
            "invariant_id": self.invariant_id,
            "estimator_ids": list(self.estimator_ids),
            "required_behavior": self.required_behavior,
            "prohibited_ablation": self.prohibited_ablation,
            "violation_action": self.violation_action,
        }


def _missing_mask_safety_invariant() -> Stage4SafetyInvariant:
    return Stage4SafetyInvariant(
        invariant_id=STAGE4_MISSING_MASK_INVARIANT_ID,
        estimator_ids=("gru_residual", "independent_mlp"),
        required_behavior=(
            "neural_inputs_keep_explicit_missing_indicators_and_history_ablations_keep_"
            "missing_usage_attempts"
        ),
        prohibited_ablation="disable_or_remove_missing_telemetry_masks",
        violation_action="fail_closed",
    )


@dataclass(frozen=True)
class Stage4Matrix:
    source_id: str
    development_protocol_id: str
    capability_contract_hash: str
    plans: tuple[Stage4ExperimentPlan, ...]
    gates: tuple[Stage4Gate, ...]
    telemetry_decisions: tuple[TelemetryDecision, ...]
    safety_invariants: tuple[Stage4SafetyInvariant, ...]
    matrix_id: str
    schema_version: int = STAGE4_MATRIX_SCHEMA_VERSION
    policy_id: str = STAGE4_MATRIX_POLICY_ID

    def __post_init__(self) -> None:
        if self.schema_version != STAGE4_MATRIX_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 4 matrix schema version")
        if self.policy_id != STAGE4_MATRIX_POLICY_ID:
            raise ValueError("unsupported Stage 4 matrix policy")
        if self.source_id not in FROZEN_SOURCE_CONDITIONS:
            raise ValueError("Stage 4 matrix source is not frozen")
        experiment_ids = [plan.spec.experiment_id for plan in self.plans]
        if experiment_ids != sorted(experiment_ids):
            raise ValueError("Stage 4 plans must use canonical experiment order")
        if len(experiment_ids) != len(set(experiment_ids)):
            raise ValueError("Stage 4 experiment ids must be unique")
        if "missing_usage_attempts" not in STAGE4_NO_TOOLS_ERRORS_FEATURES.include_features:
            raise ValueError("Stage 4 telemetry missingness indicator was ablated")
        if self.safety_invariants != (_missing_mask_safety_invariant(),):
            raise ValueError("Stage 4 missing-mask safety invariant is not frozen")
        by_id = {plan.spec.experiment_id: plan for plan in self.plans}
        for plan in self.plans:
            validate_ablation_specs(plan.spec.candidates)
            for candidate in plan.spec.candidates:
                if candidate.estimator_id not in {
                    estimator_id
                    for invariant in self.safety_invariants
                    for estimator_id in invariant.estimator_ids
                }:
                    continue
                if any(
                    "missing_mask" in path
                    for path in (
                        candidate.ablation.allowed_config_paths
                        if candidate.ablation is not None
                        else ()
                    )
                ):
                    raise ValueError("Stage 4 cannot ablate required missing masks")
            if plan.role != Stage4PlanRole.ABLATION:
                continue
            reference = by_id.get(str(plan.reference_experiment_id))
            if reference is None:
                raise ValueError("Stage 4 experiment ablation reference is missing")
            before = _spec_document(reference.spec, include_experiment_id=False)
            after = _spec_document(plan.spec, include_experiment_id=False)
            actual = {
                key
                for key in set(before) | set(after)
                if before.get(key) != after.get(key)
            }
            if actual != set(plan.allowed_config_paths):
                raise ValueError(
                    f"Stage 4 experiment ablation changed {sorted(actual)}, "
                    f"expected {sorted(plan.allowed_config_paths)}"
                )
        expected = _canonical_sha256(self.identity_document(include_matrix_id=False))
        if self.matrix_id != expected:
            raise ValueError("Stage 4 matrix id does not match its semantics")

    @property
    def experiments(self) -> tuple[ExperimentSpec, ...]:
        return tuple(plan.spec for plan in self.plans)

    def identity_document(self, *, include_matrix_id: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "source_id": self.source_id,
            "development_protocol_id": self.development_protocol_id,
            "capability_contract_hash": self.capability_contract_hash,
            "minimum_development_tasks": STAGE4_MIN_DEVELOPMENT_TASKS,
            "plans": [plan.to_dict() for plan in self.plans],
            "gates": [gate.to_dict() for gate in self.gates],
            "telemetry_decisions": [
                decision.to_dict() for decision in self.telemetry_decisions
            ],
            "safety_invariants": [
                invariant.to_dict() for invariant in self.safety_invariants
            ],
        }
        if include_matrix_id:
            value["matrix_id"] = self.matrix_id
        return value


def _feature_ablation_candidates() -> tuple[CandidateSpec, ...]:
    params = _lightgbm_params()
    mlp_params = _mlp_params()
    candidates = (
        CandidateSpec(
            "empirical",
            "empirical_quantile",
            FeatureSet("none", include_all=False),
            params={"alpha": STAGE4_ALPHA},
            role=CandidateRole.BASELINE,
        ),
        CandidateSpec(
            "lightgbm_history",
            "lightgbm_quantile",
            STAGE2_HISTORY_FEATURES,
            params=params,
        ),
        CandidateSpec(
            "mlp_history",
            "independent_mlp",
            STAGE2_HISTORY_FEATURES,
            params=mlp_params,
            role=CandidateRole.ABLATION,
            ablation=AblationSpec(
                reference_candidate_id="lightgbm_history",
                axis=AblationAxis.METHOD,
                allowed_config_paths=_method_ablation_paths(params, mlp_params),
            ),
        ),
        CandidateSpec(
            "lightgbm_without_progress",
            "lightgbm_quantile",
            STAGE4_NO_PROGRESS_FEATURES,
            params=params,
            role=CandidateRole.ABLATION,
            ablation=AblationSpec(
                reference_candidate_id="lightgbm_history",
                axis=AblationAxis.FEATURE_SET,
                allowed_config_paths=frozenset({"feature_set"}),
            ),
        ),
        CandidateSpec(
            "lightgbm_without_tools_errors",
            "lightgbm_quantile",
            STAGE4_NO_TOOLS_ERRORS_FEATURES,
            params=params,
            role=CandidateRole.ABLATION,
            ablation=AblationSpec(
                reference_candidate_id="lightgbm_history",
                axis=AblationAxis.FEATURE_SET,
                allowed_config_paths=frozenset({"feature_set"}),
            ),
        ),
        CandidateSpec(
            "lightgbm_structured",
            "lightgbm_quantile",
            STAGE2_STRUCTURED_FEATURES,
            params=params,
            role=CandidateRole.ABLATION,
            ablation=AblationSpec(
                reference_candidate_id="lightgbm_history",
                axis=AblationAxis.FEATURE_SET,
                allowed_config_paths=frozenset({"feature_set"}),
            ),
        ),
    )
    validate_ablation_specs(candidates)
    return candidates


def _single_history_candidate() -> tuple[CandidateSpec, ...]:
    return (
        CandidateSpec(
            "lightgbm_history",
            "lightgbm_quantile",
            STAGE2_HISTORY_FEATURES,
            params=_lightgbm_params(),
        ),
    )


def _call_pre_candidates() -> tuple[CandidateSpec, ...]:
    lightgbm_params = _lightgbm_params()
    mlp_params = _mlp_params()
    candidates = (
        CandidateSpec(
            "empirical",
            "empirical_quantile",
            FeatureSet("none", include_all=False),
            params={"alpha": STAGE4_ALPHA},
            role=CandidateRole.BASELINE,
        ),
        CandidateSpec(
            "pre_request_char_message_length",
            "lightgbm_quantile",
            STAGE4_PRE_REQUEST_CHAR_MESSAGE_FEATURES,
            params=lightgbm_params,
            role=CandidateRole.BASELINE,
        ),
        CandidateSpec(
            "lightgbm_history",
            "lightgbm_quantile",
            STAGE2_HISTORY_FEATURES,
            params=lightgbm_params,
        ),
        CandidateSpec(
            "mlp_history",
            "independent_mlp",
            STAGE2_HISTORY_FEATURES,
            params=mlp_params,
            role=CandidateRole.ABLATION,
            ablation=AblationSpec(
                reference_candidate_id="lightgbm_history",
                axis=AblationAxis.METHOD,
                allowed_config_paths=_method_ablation_paths(
                    lightgbm_params,
                    mlp_params,
                ),
            ),
        ),
    )
    validate_ablation_specs(candidates)
    return candidates


def _seed_policy_candidates(
    *,
    condition_id: str,
    input_contract_hash: str,
) -> tuple[CandidateSpec, ...]:
    params = {
        "expected_condition_id": condition_id,
        "expected_input_contract_hash": input_contract_hash,
    }
    initializer_params = {"alpha": STAGE4_ALPHA}

    def graph(seed_policy_id: str) -> CandidateGraph:
        return CandidateGraph(
            initializer_estimator_id="empirical_quantile",
            updater_estimator_id="cross_position_deduct",
            lifecycle_schema_id=TASK_LIFECYCLE_SCHEMA_ID,
            seed_policy_id=seed_policy_id,
            inner_split_policy_id=INNER_FOLD_POLICY_ID,
        )

    candidates = (
        CandidateSpec(
            "cross_position_deduct_raw_repaired_oof_seed",
            "cross_position_deduct",
            NO_FEATURES,
            params=params,
            initializer_params=initializer_params,
            graph=graph(SEED_POLICY_ID),
        ),
        CandidateSpec(
            "cross_position_deduct_point_only_oof_seed",
            "cross_position_deduct",
            NO_FEATURES,
            params=params,
            initializer_params=initializer_params,
            graph=graph(POINT_ONLY_SEED_POLICY_ID),
            role=CandidateRole.ABLATION,
            ablation=AblationSpec(
                reference_candidate_id=(
                    "cross_position_deduct_raw_repaired_oof_seed"
                ),
                axis=AblationAxis.SEED_POLICY,
                allowed_config_paths=frozenset({"graph.seed_policy_id"}),
            ),
        ),
    )
    validate_ablation_specs(candidates)
    return candidates


def _aggregate_candidates() -> tuple[CandidateSpec, ...]:
    return (
        CandidateSpec(
            "empirical",
            "empirical_quantile",
            FeatureSet("none", include_all=False),
            params={"alpha": STAGE4_ALPHA},
            role=CandidateRole.BASELINE,
        ),
        CandidateSpec(
            "task_chars_length",
            "lightgbm_quantile",
            SPEND_AGGREGATE_TASK_CHARS,
            params=_lightgbm_params(),
            role=CandidateRole.BASELINE,
        ),
        CandidateSpec(
            "lightgbm_structured",
            "lightgbm_quantile",
            SPEND_AGGREGATE_STRUCTURED_FEATURES,
            params=_lightgbm_params(),
        ),
    )


def _single_aggregate_candidate() -> tuple[CandidateSpec, ...]:
    return (
        CandidateSpec(
            "lightgbm_structured",
            "lightgbm_quantile",
            SPEND_AGGREGATE_STRUCTURED_FEATURES,
            params=_lightgbm_params(),
        ),
    )


def _experiment_id(
    source_id: str,
    condition_id: str,
    position: PredictionPosition,
    target: PredictionTarget,
    suffix: str,
) -> str:
    return (
        f"stage4-{source_id}-{condition_id.removeprefix('condition:')}-"
        f"{position.value}-{target.value}-{suffix}"
    )


def _cell_counts(
    protocol: DevelopmentProtocol,
    *,
    position: PredictionPosition,
    target: PredictionTarget,
    condition_id: str,
) -> tuple[int, int]:
    cell = protocol.development_dataset.select(
        position,
        target,
        condition_id=condition_id,
    )
    return len({row.point.task_id for row in cell.rows}), len(cell.rows)


def _cell_gate_reason(task_count: int, point_count: int) -> str:
    if point_count == 0:
        return "capability_or_observed_development_cell_unavailable"
    if task_count < STAGE4_MIN_DEVELOPMENT_TASKS:
        return "insufficient_development_tasks_for_five_fold_cv"
    return ""


def _calibration_plans(
    *,
    source_id: str,
    condition_id: str,
    position: PredictionPosition,
    target: PredictionTarget,
    candidates: tuple[CandidateSpec, ...],
) -> tuple[Stage4ExperimentPlan, Stage4ExperimentPlan]:
    reference_id = _experiment_id(
        source_id,
        condition_id,
        position,
        target,
        "calibration-task-max",
    )
    reference = Stage4ExperimentPlan(
        ExperimentSpec(
            reference_id,
            position,
            target,
            candidates,
            alpha=STAGE4_ALPHA,
            calibrator_id=STAGE4_PRIMARY_CALIBRATOR_ID,
            condition_id=condition_id,
        )
    )
    ablation = Stage4ExperimentPlan(
        ExperimentSpec(
            _experiment_id(
                source_id,
                condition_id,
                position,
                target,
                "calibration-none",
            ),
            position,
            target,
            candidates,
            alpha=STAGE4_ALPHA,
            calibrator_id="none",
            condition_id=condition_id,
        ),
        role=Stage4PlanRole.ABLATION,
        reference_experiment_id=reference_id,
        axis=AblationAxis.CALIBRATION,
        allowed_config_paths=frozenset({"calibrator_id"}),
    )
    return reference, ablation


def build_stage4_matrix(
    protocol: DevelopmentProtocol,
    *,
    source_id: str,
    capabilities: SourceCapabilities,
) -> Stage4Matrix:
    if source_id not in FROZEN_SOURCE_CONDITIONS:
        raise ValueError(f"unsupported Stage 4 source_id {source_id!r}")
    if capabilities.source_id != source_id:
        raise ValueError("Stage 4 capability source_id differs from the matrix source")
    if protocol.parent_capability_contract_hash != capabilities.contract_hash:
        raise ValueError("Stage 4 capabilities differ from the development protocol")
    if protocol.development_dataset.task_ids & protocol.final_holdout_tasks:
        raise ValueError("final-holdout tasks leaked into the Stage 4 matrix input")

    plans: list[Stage4ExperimentPlan] = []
    gates: list[Stage4Gate] = []
    decisions = tuple(
        decide_telemetry_surface(capabilities, surface)
        for surface in STAGE4_TELEMETRY_SURFACES
    )
    for decision in decisions:
        if decision.gated:
            gates.append(
                Stage4Gate(
                    source_id=source_id,
                    condition_id="condition:all",
                    surface=decision.surface.value,
                    reason=decision.reason,
                    capability_contract_hash=capabilities.contract_hash,
                    development_task_count=0,
                    eligible_point_count=0,
                )
            )

    for condition_id in sorted(FROZEN_SOURCE_CONDITIONS[source_id]):
        if Observable.TASK_TEXT not in capabilities.observables:
            gates.append(
                Stage4Gate(
                    source_id=source_id,
                    condition_id=condition_id,
                    surface="fold_fitted_tfidf_retrieval",
                    reason="missing_observables:task_text",
                    capability_contract_hash=capabilities.contract_hash,
                    development_task_count=0,
                    eligible_point_count=0,
                )
            )

        if source_id == SPEND_AGGREGATE_SOURCE_ID:
            position = PredictionPosition.TASK_LAUNCH
            target = PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS
            task_count, point_count = _cell_counts(
                protocol,
                position=position,
                target=target,
                condition_id=condition_id,
            )
            reason = _cell_gate_reason(task_count, point_count)
            if reason:
                gates.append(
                    Stage4Gate(
                        source_id,
                        condition_id,
                        "task_launch",
                        reason,
                        capabilities.contract_hash,
                        task_count,
                        point_count,
                        position,
                        target,
                    )
                )
                continue
            plans.append(
                Stage4ExperimentPlan(
                    ExperimentSpec(
                        _experiment_id(
                            source_id,
                            condition_id,
                            position,
                            target,
                            "method",
                        ),
                        position,
                        target,
                        _aggregate_candidates(),
                        alpha=STAGE4_ALPHA,
                        calibrator_id=STAGE4_PRIMARY_CALIBRATOR_ID,
                        condition_id=condition_id,
                    )
                )
            )
            plans.extend(
                _calibration_plans(
                    source_id=source_id,
                    condition_id=condition_id,
                    position=position,
                    target=target,
                    candidates=_single_aggregate_candidate(),
                )
            )
            continue

        task_position = PredictionPosition.TASK_UPDATE
        task_target = PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS
        task_count, point_count = _cell_counts(
            protocol,
            position=task_position,
            target=task_target,
            condition_id=condition_id,
        )
        task_reason = _cell_gate_reason(task_count, point_count)
        if task_reason:
            gates.append(
                Stage4Gate(
                    source_id,
                    condition_id,
                    "task_feature_ablation",
                    task_reason,
                    capabilities.contract_hash,
                    task_count,
                    point_count,
                    task_position,
                    task_target,
                )
            )
        else:
            plans.append(
                Stage4ExperimentPlan(
                    ExperimentSpec(
                        _experiment_id(
                            source_id,
                            condition_id,
                            task_position,
                            task_target,
                            "feature-ablation",
                        ),
                        task_position,
                        task_target,
                        _feature_ablation_candidates(),
                        alpha=STAGE4_ALPHA,
                        calibrator_id=STAGE4_PRIMARY_CALIBRATOR_ID,
                        condition_id=condition_id,
                    )
                )
            )
            plans.extend(
                _calibration_plans(
                    source_id=source_id,
                    condition_id=condition_id,
                    position=task_position,
                    target=task_target,
                    candidates=_single_history_candidate(),
                )
            )
            input_contract_hash = protocol.development_dataset.input_contract_hash
            if input_contract_hash is None:
                gates.append(
                    Stage4Gate(
                        source_id,
                        condition_id,
                        "seed_policy_ablation",
                        "missing_input_contract_hash",
                        capabilities.contract_hash,
                        task_count,
                        point_count,
                        task_position,
                        task_target,
                    )
                )
            else:
                plans.append(
                    Stage4ExperimentPlan(
                        ExperimentSpec(
                            _experiment_id(
                                source_id,
                                condition_id,
                                task_position,
                                task_target,
                                "seed-policy-ablation",
                            ),
                            task_position,
                            task_target,
                            _seed_policy_candidates(
                                condition_id=condition_id,
                                input_contract_hash=input_contract_hash,
                            ),
                            alpha=STAGE4_ALPHA,
                            calibrator_id=STAGE4_PRIMARY_CALIBRATOR_ID,
                            condition_id=condition_id,
                        )
                    )
                )

        for target in STAGE4_CALL_PRE_TARGETS:
            decision = decide_target_capability(
                capabilities,
                PredictionPosition.CALL_PRE,
                target,
            )
            task_count, point_count = _cell_counts(
                protocol,
                position=PredictionPosition.CALL_PRE,
                target=target,
                condition_id=condition_id,
            )
            reason = decision.reason if decision.gated else _cell_gate_reason(
                task_count,
                point_count,
            )
            if reason:
                gates.append(
                    Stage4Gate(
                        source_id,
                        condition_id,
                        "call_pre",
                        reason,
                        capabilities.contract_hash,
                        task_count,
                        point_count,
                        PredictionPosition.CALL_PRE,
                        target,
                    )
                )
                continue
            plans.append(
                Stage4ExperimentPlan(
                    ExperimentSpec(
                        _experiment_id(
                            source_id,
                            condition_id,
                            PredictionPosition.CALL_PRE,
                            target,
                            "method",
                        ),
                        PredictionPosition.CALL_PRE,
                        target,
                        _call_pre_candidates(),
                        alpha=STAGE4_ALPHA,
                        calibrator_id=STAGE4_PRIMARY_CALIBRATOR_ID,
                        condition_id=condition_id,
                    )
                )
            )

        call_update_target = PredictionTarget.CALL_REMAINING_OUTPUT_TOKENS
        call_update_decision = decide_target_capability(
            capabilities,
            PredictionPosition.CALL_UPDATE,
            call_update_target,
        )
        update_task_count, update_point_count = _cell_counts(
            protocol,
            position=PredictionPosition.CALL_UPDATE,
            target=call_update_target,
            condition_id=condition_id,
        )
        update_reason = (
            call_update_decision.reason
            if call_update_decision.gated
            else _cell_gate_reason(update_task_count, update_point_count)
        )
        if update_reason:
            gates.append(
                Stage4Gate(
                    source_id,
                    condition_id,
                    "call_update",
                    update_reason,
                    capabilities.contract_hash,
                    update_task_count,
                    update_point_count,
                    PredictionPosition.CALL_UPDATE,
                    call_update_target,
                )
            )
        else:
            candidates = [
                CandidateSpec(
                    "empirical",
                    "empirical_quantile",
                    FeatureSet("none", include_all=False),
                    params={"alpha": STAGE4_ALPHA},
                    role=CandidateRole.BASELINE,
                )
            ]
            missing_g3_observables = tuple(
                sorted(
                    observable.value
                    for observable in (
                        STAGE4_G3_REQUIRED_OBSERVABLES
                        - capabilities.observables
                    )
                )
            )
            if missing_g3_observables:
                gates.append(
                    Stage4Gate(
                        source_id,
                        condition_id,
                        "g3_composite",
                        (
                            "missing_observables:"
                            f"{','.join(missing_g3_observables)}"
                        ),
                        capabilities.contract_hash,
                        update_task_count,
                        update_point_count,
                        PredictionPosition.CALL_UPDATE,
                        call_update_target,
                    )
                )
            else:
                candidates.append(
                    CandidateSpec(
                        "lightgbm_g3",
                        "lightgbm_quantile",
                        STAGE4_G3_FEATURES,
                        params=_lightgbm_params(),
                    )
                )
            plans.append(
                Stage4ExperimentPlan(
                    ExperimentSpec(
                        _experiment_id(
                            source_id,
                            condition_id,
                            PredictionPosition.CALL_UPDATE,
                            call_update_target,
                            "method",
                        ),
                        PredictionPosition.CALL_UPDATE,
                        call_update_target,
                        tuple(candidates),
                        alpha=STAGE4_ALPHA,
                        calibrator_id=STAGE4_PRIMARY_CALIBRATOR_ID,
                        condition_id=condition_id,
                    )
                )
            )

    ordered_plans = tuple(sorted(plans, key=lambda plan: plan.spec.experiment_id))
    ordered_gates = tuple(
        sorted(
            gates,
            key=lambda gate: (
                gate.condition_id,
                gate.surface,
                gate.position.value if gate.position is not None else "",
                gate.target.value if gate.target is not None else "",
                gate.reason,
            ),
        )
    )
    safety_invariants = (_missing_mask_safety_invariant(),)
    semantic = {
        "schema_version": STAGE4_MATRIX_SCHEMA_VERSION,
        "policy_id": STAGE4_MATRIX_POLICY_ID,
        "source_id": source_id,
        "development_protocol_id": protocol.protocol_id,
        "capability_contract_hash": capabilities.contract_hash,
        "minimum_development_tasks": STAGE4_MIN_DEVELOPMENT_TASKS,
        "plans": [plan.to_dict() for plan in ordered_plans],
        "gates": [gate.to_dict() for gate in ordered_gates],
        "telemetry_decisions": [decision.to_dict() for decision in decisions],
        "safety_invariants": [
            invariant.to_dict() for invariant in safety_invariants
        ],
    }
    return Stage4Matrix(
        source_id=source_id,
        development_protocol_id=protocol.protocol_id,
        capability_contract_hash=capabilities.contract_hash,
        plans=ordered_plans,
        gates=ordered_gates,
        telemetry_decisions=decisions,
        safety_invariants=safety_invariants,
        matrix_id=_canonical_sha256(semantic),
    )


FROZEN_STAGE4_SOURCE_CONDITIONS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        BAGEN_SOURCE_ID: FROZEN_SOURCE_CONDITIONS[BAGEN_SOURCE_ID],
        BAGEN_SOKOBAN_SOURCE_ID: FROZEN_SOURCE_CONDITIONS[BAGEN_SOKOBAN_SOURCE_ID],
        SPEND_SOURCE_ID: FROZEN_SOURCE_CONDITIONS[SPEND_SOURCE_ID],
        SPEND_AGGREGATE_SOURCE_ID: FROZEN_SOURCE_CONDITIONS[SPEND_AGGREGATE_SOURCE_ID],
    }
)


__all__ = [
    "FROZEN_STAGE4_SOURCE_CONDITIONS",
    "STAGE4_ALPHA",
    "STAGE4_CALL_PRE_TARGETS",
    "STAGE4_MATRIX_POLICY_ID",
    "STAGE4_MATRIX_SCHEMA_VERSION",
    "STAGE4_MISSING_MASK_INVARIANT_ID",
    "STAGE4_MIN_DEVELOPMENT_TASKS",
    "STAGE4_PRIMARY_CALIBRATOR_ID",
    "STAGE4_TELEMETRY_SURFACES",
    "Stage4ExperimentPlan",
    "Stage4Gate",
    "Stage4Matrix",
    "Stage4PlanRole",
    "Stage4SafetyInvariant",
    "build_stage4_matrix",
]
