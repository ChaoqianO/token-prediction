from __future__ import annotations

import unittest
from dataclasses import replace

from tests.test_stage2_matrix import (
    AGGREGATE_CONDITION,
    PRIMARY_CONDITION,
    _dataset,
    _point,
)

from token_prediction.contracts import Observable, SourceCapabilities
from token_prediction.crossfit import POINT_ONLY_SEED_POLICY_ID, SEED_POLICY_ID
from token_prediction.dataset import (
    DatasetRow,
    LabelStatus,
    PredictionPosition,
    PredictionTarget,
    SupervisedDataset,
)
from token_prediction.development import build_development_protocol
from token_prediction.experiment import AblationAxis, validate_ablation_specs
from token_prediction.stage2_matrix import (
    BAGEN_SOURCE_ID,
    SPEND_AGGREGATE_SOURCE_ID,
)
from token_prediction.stage4_matrix import (
    STAGE4_CALL_PRE_TARGETS,
    STAGE4_MISSING_MASK_INVARIANT_ID,
    Stage4Matrix,
    Stage4PlanRole,
    build_stage4_matrix,
)
from token_prediction.telemetry import TelemetrySurface


def _bagen_capabilities() -> SourceCapabilities:
    return SourceCapabilities(
        BAGEN_SOURCE_ID,
        frozenset(
            {
                Observable.ATTEMPT_USAGE,
                Observable.REQUEST_BOUNDARIES,
                Observable.REQUEST_MESSAGES,
                Observable.TASK_TERMINATION,
            }
        ),
    )


def _bagen_dataset() -> SupervisedDataset:
    capabilities = _bagen_capabilities()
    base = _dataset()
    rows = list(base.rows)
    for task_index in range(100):
        for target_index, target in enumerate(STAGE4_CALL_PRE_TARGETS):
            point = replace(
                _point(
                    task_index,
                    position=PredictionPosition.TASK_UPDATE,
                    target=target,
                    condition_id=PRIMARY_CONDITION,
                    request_chars=100 + task_index,
                ),
                point_id=f"call-point-{task_index:03d}-{target_index}",
                source_event_id=f"call-event-{task_index:03d}-{target_index}",
                position=PredictionPosition.CALL_PRE,
                cutoff_event_seq=20,
            )
            rows.append(
                DatasetRow(
                    point,
                    100 + task_index + target_index,
                    LabelStatus.OBSERVED,
                )
            )
    return replace(
        base,
        rows=tuple(rows),
        capability_contract_hash=capabilities.contract_hash,
    )


def _bagen_call_update_dataset(
    telemetry_observables: frozenset[Observable],
) -> tuple[SupervisedDataset, SourceCapabilities]:
    capabilities = SourceCapabilities(
        BAGEN_SOURCE_ID,
        _bagen_capabilities().observables | telemetry_observables,
    )
    base = _bagen_dataset()
    rows = list(base.rows)
    for task_index in range(100):
        point = replace(
            _point(
                task_index,
                position=PredictionPosition.TASK_UPDATE,
                target=PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
                condition_id=PRIMARY_CONDITION,
            ),
            point_id=f"call-update-point-{task_index:03d}",
            source_event_id=f"call-update-event-{task_index:03d}",
            logical_call_id=f"call-{task_index:03d}",
            cutoff_event_seq=30,
            position=PredictionPosition.CALL_UPDATE,
            target=PredictionTarget.CALL_REMAINING_OUTPUT_TOKENS,
        )
        rows.append(
            DatasetRow(
                point,
                100 + task_index,
                LabelStatus.OBSERVED,
            )
        )
    return (
        replace(
            base,
            rows=tuple(rows),
            capability_contract_hash=capabilities.contract_hash,
        ),
        capabilities,
    )


def _aggregate_dataset() -> tuple[SupervisedDataset, SourceCapabilities]:
    capabilities = SourceCapabilities(
        SPEND_AGGREGATE_SOURCE_ID,
        frozenset({Observable.TASK_AGGREGATE_USAGE}),
    )
    rows = tuple(
        DatasetRow(
            replace(
                _point(
                    index,
                    position=PredictionPosition.TASK_LAUNCH,
                    target=PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
                    condition_id=AGGREGATE_CONDITION,
                ),
                features={
                    "task_char_count": 100 + index,
                    "task_word_count": 20,
                    "task_line_count": 2,
                    "task_code_fence_count": 0,
                    "repo_id": "org/repo",
                    "model_id": "gpt-5.2",
                    "agent_id": "openhands",
                },
            ),
            1_000 + index,
            LabelStatus.OBSERVED,
        )
        for index in range(100)
    )
    return (
        SupervisedDataset(
            dataset_id="1" * 64,
            rows=rows,
            schema_version=2,
            source_descriptor_hash="2" * 64,
            capability_contract_hash=capabilities.contract_hash,
            input_contract_hash="4" * 64,
        ),
        capabilities,
    )


class Stage4MatrixTests(unittest.TestCase):
    def test_primary_condition_has_single_axis_features_calibration_and_three_call_targets(
        self,
    ) -> None:
        capabilities = _bagen_capabilities()
        matrix = build_stage4_matrix(
            build_development_protocol(_bagen_dataset()),
            source_id=BAGEN_SOURCE_ID,
            capabilities=capabilities,
        )
        primary = tuple(
            plan
            for plan in matrix.plans
            if plan.spec.condition_id == PRIMARY_CONDITION
        )
        self.assertEqual(len(primary), 7)
        self.assertEqual(
            {
                plan.spec.target
                for plan in primary
                if plan.spec.position == PredictionPosition.CALL_PRE
            },
            set(STAGE4_CALL_PRE_TARGETS),
        )
        self.assertTrue(
            all(
                len(
                    {
                        candidate_result.target
                        for candidate_result in (plan.spec,)
                    }
                )
                == 1
                for plan in primary
            )
        )

        feature_plan = next(
            plan
            for plan in primary
            if plan.spec.experiment_id.endswith("feature-ablation")
        )
        candidates = {
            candidate.candidate_id: candidate
            for candidate in feature_plan.spec.candidates
        }
        self.assertEqual(
            set(candidates),
            {
                "empirical",
                "lightgbm_history",
                "lightgbm_structured",
                "lightgbm_without_progress",
                "lightgbm_without_tools_errors",
                "mlp_history",
            },
        )

        call_pre_plans = tuple(
            plan
            for plan in primary
            if plan.spec.position == PredictionPosition.CALL_PRE
        )
        self.assertEqual(len(call_pre_plans), len(STAGE4_CALL_PRE_TARGETS))
        self.assertEqual(
            len({plan.spec.experiment_id for plan in call_pre_plans}),
            len(STAGE4_CALL_PRE_TARGETS),
        )
        for plan in call_pre_plans:
            by_id = {
                candidate.candidate_id: candidate
                for candidate in plan.spec.candidates
            }
            self.assertEqual(
                set(by_id),
                {
                    "empirical",
                    "pre_request_char_message_length",
                    "lightgbm_history",
                    "mlp_history",
                },
            )
            self.assertEqual(
                by_id["mlp_history"].ablation.axis,
                AblationAxis.METHOD,
            )
            self.assertEqual(
                by_id["mlp_history"].ablation.reference_candidate_id,
                "lightgbm_history",
            )
            self.assertEqual(plan.spec.alpha, 0.10)
            self.assertEqual(plan.spec.calibrator_id, "task_max_conformal")
            self.assertEqual(plan.spec.condition_id, PRIMARY_CONDITION)
            validate_ablation_specs(plan.spec.candidates)
        validate_ablation_specs(feature_plan.spec.candidates)
        self.assertIn(
            "missing_usage_attempts",
            candidates["lightgbm_without_tools_errors"].feature_set.include_features,
        )
        self.assertEqual(
            {
                candidate.candidate_id: candidate.ablation.axis
                for candidate in feature_plan.spec.candidates
                if candidate.ablation is not None
            },
            {
                "lightgbm_structured": AblationAxis.FEATURE_SET,
                "lightgbm_without_progress": AblationAxis.FEATURE_SET,
                "lightgbm_without_tools_errors": AblationAxis.FEATURE_SET,
                "mlp_history": AblationAxis.METHOD,
            },
        )

        calibration = tuple(
            plan
            for plan in primary
            if "calibration-" in plan.spec.experiment_id
        )
        self.assertEqual(len(calibration), 2)
        ablation = next(
            plan for plan in calibration if plan.role == Stage4PlanRole.ABLATION
        )
        self.assertEqual(ablation.axis, AblationAxis.CALIBRATION)
        self.assertEqual(
            ablation.allowed_config_paths,
            frozenset({"calibrator_id"}),
        )

        seed_plan = next(
            plan
            for plan in primary
            if plan.spec.experiment_id.endswith("seed-policy-ablation")
        )
        seed_candidates = {
            candidate.candidate_id: candidate
            for candidate in seed_plan.spec.candidates
        }
        self.assertEqual(
            set(seed_candidates),
            {
                "cross_position_deduct_raw_repaired_oof_seed",
                "cross_position_deduct_point_only_oof_seed",
            },
        )
        raw_seed = seed_candidates[
            "cross_position_deduct_raw_repaired_oof_seed"
        ]
        point_only_seed = seed_candidates[
            "cross_position_deduct_point_only_oof_seed"
        ]
        self.assertEqual(raw_seed.graph.seed_policy_id, SEED_POLICY_ID)
        self.assertEqual(
            point_only_seed.graph.seed_policy_id,
            POINT_ONLY_SEED_POLICY_ID,
        )
        self.assertEqual(
            point_only_seed.ablation.axis,
            AblationAxis.SEED_POLICY,
        )
        self.assertEqual(
            point_only_seed.ablation.allowed_config_paths,
            frozenset({"graph.seed_policy_id"}),
        )
        self.assertEqual(raw_seed.params, point_only_seed.params)
        self.assertEqual(
            raw_seed.initializer_params,
            point_only_seed.initializer_params,
        )
        self.assertEqual(
            seed_plan.spec.calibrator_id,
            "task_max_conformal",
        )
        validate_ablation_specs(seed_plan.spec.candidates)

        self.assertEqual(len(matrix.safety_invariants), 1)
        invariant = matrix.safety_invariants[0]
        self.assertEqual(
            invariant.invariant_id,
            STAGE4_MISSING_MASK_INVARIANT_ID,
        )
        self.assertEqual(
            invariant.estimator_ids,
            ("gru_residual", "independent_mlp"),
        )
        self.assertEqual(invariant.violation_action, "fail_closed")

    def test_missing_retrieval_call_update_and_g3_are_explicit_gates(self) -> None:
        matrix = build_stage4_matrix(
            build_development_protocol(_bagen_dataset()),
            source_id=BAGEN_SOURCE_ID,
            capabilities=_bagen_capabilities(),
        )
        primary_gates = tuple(
            gate for gate in matrix.gates if gate.condition_id == PRIMARY_CONDITION
        )
        retrieval = next(
            gate
            for gate in primary_gates
            if gate.surface == "fold_fitted_tfidf_retrieval"
        )
        self.assertEqual(retrieval.reason, "missing_observables:task_text")
        call_update = next(
            gate for gate in primary_gates if gate.surface == "call_update"
        )
        self.assertIn("output_deltas", call_update.reason)

        decisions = {
            decision.surface: decision for decision in matrix.telemetry_decisions
        }
        self.assertTrue(decisions[TelemetrySurface.ONLINE_SHADOW].available)
        self.assertFalse(decisions[TelemetrySurface.CALL_UPDATE].available)
        self.assertFalse(decisions[TelemetrySurface.G3_ENTROPY_STOP].available)
        self.assertFalse(decisions[TelemetrySurface.G3_HIDDEN_STATE].available)
        self.assertFalse(decisions[TelemetrySurface.G3_RESUMABLE_STATE].available)

    def test_composite_g3_candidate_requires_all_feature_telemetry(self) -> None:
        cases = (
            (
                "output-only",
                frozenset({Observable.OUTPUT_DELTAS}),
                "missing_observables:hidden_state,logprobs",
                {"empirical"},
                "g3_composite",
            ),
            (
                "logprobs-only",
                frozenset({Observable.LOGPROBS}),
                "missing_observables:output_deltas",
                set(),
                "call_update",
            ),
            (
                "hidden-only",
                frozenset({Observable.HIDDEN_STATE}),
                "missing_observables:output_deltas",
                set(),
                "call_update",
            ),
            (
                "full",
                frozenset(
                    {
                        Observable.OUTPUT_DELTAS,
                        Observable.LOGPROBS,
                        Observable.HIDDEN_STATE,
                    }
                ),
                None,
                {"empirical", "lightgbm_g3"},
                None,
            ),
        )
        for (
            name,
            telemetry,
            expected_reason,
            expected_candidates,
            expected_gate_surface,
        ) in cases:
            with self.subTest(name=name):
                dataset, capabilities = _bagen_call_update_dataset(telemetry)
                matrix = build_stage4_matrix(
                    build_development_protocol(dataset),
                    source_id=BAGEN_SOURCE_ID,
                    capabilities=capabilities,
                )
                plans = tuple(
                    plan
                    for plan in matrix.plans
                    if plan.spec.condition_id == PRIMARY_CONDITION
                    and plan.spec.position == PredictionPosition.CALL_UPDATE
                )
                self.assertEqual(bool(plans), bool(expected_candidates))
                if expected_candidates:
                    self.assertEqual(len(plans), 1)
                    self.assertEqual(
                        {
                            candidate.candidate_id
                            for candidate in plans[0].spec.candidates
                        },
                        expected_candidates,
                    )
                matching_gates = [
                    gate
                    for gate in matrix.gates
                    if gate.condition_id == PRIMARY_CONDITION
                    and gate.surface in {"call_update", "g3_composite"}
                ]
                if expected_gate_surface is None:
                    self.assertFalse(matching_gates)
                else:
                    gate = next(
                        gate
                        for gate in matching_gates
                        if gate.surface == expected_gate_surface
                    )
                    self.assertEqual(gate.reason, expected_reason)

    def test_aggregate_remains_task_launch_only_with_calibration_ablation(self) -> None:
        dataset, capabilities = _aggregate_dataset()
        matrix = build_stage4_matrix(
            build_development_protocol(dataset),
            source_id=SPEND_AGGREGATE_SOURCE_ID,
            capabilities=capabilities,
        )
        self.assertEqual(len(matrix.plans), 3)
        self.assertTrue(
            all(
                plan.spec.position == PredictionPosition.TASK_LAUNCH
                and plan.spec.target
                == PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS
                for plan in matrix.plans
            )
        )
        self.assertTrue(
            any(plan.role == Stage4PlanRole.ABLATION for plan in matrix.plans)
        )

    def test_holdout_suffix_changes_cannot_change_matrix_identity(self) -> None:
        dataset = _bagen_dataset()
        capabilities = _bagen_capabilities()
        first_protocol = build_development_protocol(dataset)
        changed_rows = tuple(
            replace(
                row,
                label=int(row.label or 0) + 10_000_000,
                point=replace(row.point, features={"suffix_only": 1}),
            )
            if row.point.task_id in first_protocol.final_holdout_tasks
            else row
            for row in dataset.rows
        )
        second_protocol = build_development_protocol(
            replace(dataset, dataset_id="f" * 64, rows=changed_rows)
        )
        self.assertEqual(
            build_stage4_matrix(
                first_protocol,
                source_id=BAGEN_SOURCE_ID,
                capabilities=capabilities,
            ),
            build_stage4_matrix(
                second_protocol,
                source_id=BAGEN_SOURCE_ID,
                capabilities=capabilities,
            ),
        )

    def test_capability_and_matrix_identity_tamper_fail_closed(self) -> None:
        protocol = build_development_protocol(_bagen_dataset())
        capabilities = _bagen_capabilities()
        matrix = build_stage4_matrix(
            protocol,
            source_id=BAGEN_SOURCE_ID,
            capabilities=capabilities,
        )
        with self.assertRaisesRegex(ValueError, "matrix id"):
            Stage4Matrix(
                source_id=matrix.source_id,
                development_protocol_id=matrix.development_protocol_id,
                capability_contract_hash=matrix.capability_contract_hash,
                plans=matrix.plans,
                gates=matrix.gates,
                telemetry_decisions=matrix.telemetry_decisions,
                safety_invariants=matrix.safety_invariants,
                matrix_id="0" * 64,
            )
        with self.assertRaisesRegex(ValueError, "missing-mask safety invariant"):
            replace(matrix, safety_invariants=())
        with self.assertRaisesRegex(ValueError, "capabilities differ"):
            build_stage4_matrix(
                protocol,
                source_id=BAGEN_SOURCE_ID,
                capabilities=SourceCapabilities(
                    BAGEN_SOURCE_ID,
                    frozenset({Observable.REQUEST_BOUNDARIES}),
                ),
            )


if __name__ == "__main__":
    unittest.main()
