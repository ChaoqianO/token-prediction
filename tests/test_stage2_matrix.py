from __future__ import annotations

import unittest
from dataclasses import replace

from token_prediction.dataset import (
    DatasetRow,
    LabelStatus,
    PredictionPoint,
    PredictionPosition,
    PredictionTarget,
    SupervisedDataset,
)
from token_prediction.development import build_development_protocol
from token_prediction.stage2_matrix import (
    BAGEN_SOKOBAN_SOURCE_ID,
    BAGEN_SOURCE_ID,
    FROZEN_BAGEN_CONDITIONS,
    SPEND_AGGREGATE_SOURCE_ID,
    Stage2Matrix,
    build_stage2_matrix,
)


PRIMARY_CONDITION = "condition:54cb50fce273f0aa2d74"
SPARSE_CONDITION = "condition:20f615a22697984db6cc"
AGGREGATE_CONDITION = "condition:spend-your-money:f06cc0d037ed16ff12db"


def _point(
    task_index: int,
    *,
    position: PredictionPosition,
    target: PredictionTarget,
    condition_id: str,
    request_chars: int | None = None,
) -> PredictionPoint:
    suffix = f"{condition_id[-4:]}-{position.value}-{target.value}"
    features: dict[str, int | str | None] = {
        "model_id": "fixture-model",
        "agent_id": "fixture-agent",
        "completed_call_count": 1 if position == PredictionPosition.TASK_UPDATE else 0,
        "missing_usage_attempts": 0,
        "cumulative_provider_input_tokens": 10,
        "cumulative_provider_output_tokens": 5,
    }
    if position == PredictionPosition.TASK_UPDATE:
        features["request_content_chars"] = request_chars
        features["request_message_count"] = 2
    return PredictionPoint(
        point_id=f"point-{task_index:03d}-{suffix}",
        source_event_id=f"event-{task_index:03d}-{suffix}",
        task_id=f"private-task-{task_index:03d}",
        trajectory_id=f"trajectory-{task_index:03d}",
        run_id=f"run-{task_index:03d}",
        prediction_context_id=f"context-{task_index:03d}-{suffix}",
        condition_id=condition_id,
        logical_call_id=(
            f"call-{task_index:03d}" if position == PredictionPosition.TASK_UPDATE else None
        ),
        attempt_id=None,
        cutoff_event_seq=(10 if position == PredictionPosition.TASK_UPDATE else 0),
        position=position,
        target=target,
        features=features,
        known_offset_tokens=0,
    )


def _dataset(*, request_chars: bool = True) -> SupervisedDataset:
    rows: list[DatasetRow] = []
    for task_index in range(100):
        for position, target in (
            (
                PredictionPosition.TASK_LAUNCH,
                PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
            ),
            (
                PredictionPosition.TASK_UPDATE,
                PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
            ),
        ):
            rows.append(
                DatasetRow(
                    _point(
                        task_index,
                        position=position,
                        target=target,
                        condition_id=PRIMARY_CONDITION,
                        request_chars=(100 + task_index if request_chars else None),
                    ),
                    1_000 + task_index,
                    LabelStatus.OBSERVED,
                )
            )
    for task_index in range(5):
        rows.append(
            DatasetRow(
                _point(
                    task_index,
                    position=PredictionPosition.TASK_UPDATE,
                    target=PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
                    condition_id=SPARSE_CONDITION,
                    request_chars=50,
                ),
                100,
                LabelStatus.OBSERVED,
            )
        )
    return SupervisedDataset(
        dataset_id="a" * 64,
        rows=tuple(rows),
        schema_version=2,
        source_descriptor_hash="b" * 64,
        capability_contract_hash="c" * 64,
        input_contract_hash="d" * 64,
    )


class Stage2MatrixTests(unittest.TestCase):
    def test_spend_aggregate_is_launch_only_with_real_task_length(self) -> None:
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
        dataset = SupervisedDataset(
            dataset_id="1" * 64,
            rows=rows,
            schema_version=2,
            source_descriptor_hash="2" * 64,
            capability_contract_hash="3" * 64,
            input_contract_hash="4" * 64,
        )
        matrix = build_stage2_matrix(
            build_development_protocol(dataset),
            source_id=SPEND_AGGREGATE_SOURCE_ID,
        )
        self.assertEqual(len(matrix.experiments), 1)
        self.assertFalse(matrix.gates)
        spec = matrix.experiments[0]
        self.assertEqual(spec.position, PredictionPosition.TASK_LAUNCH)
        self.assertEqual(
            {candidate.candidate_id for candidate in spec.candidates},
            {
                "empirical",
                "task_chars_length",
                "lightgbm_structured",
                "mlp_structured",
            },
        )

    def test_matrix_routes_launch_and_lifecycle_candidates_on_one_cohort(self) -> None:
        protocol = build_development_protocol(_dataset())
        matrix = build_stage2_matrix(protocol, source_id=BAGEN_SOURCE_ID)
        primary = tuple(
            spec for spec in matrix.experiments if spec.condition_id == PRIMARY_CONDITION
        )
        self.assertEqual(len(primary), 2)
        by_position = {spec.position: spec for spec in primary}
        launch_ids = {candidate.candidate_id for candidate in by_position[
            PredictionPosition.TASK_LAUNCH
        ].candidates}
        update_ids = {candidate.candidate_id for candidate in by_position[
            PredictionPosition.TASK_UPDATE
        ].candidates}
        self.assertEqual(
            launch_ids,
            {
                "empirical",
                "lightgbm_structured",
                "lightgbm_history",
                "mlp_structured",
                "mlp_history",
            },
        )
        self.assertTrue(
            {
                "request_chars_length",
                "within_cell_deduct",
                "cross_position_deduct",
            }
            <= update_ids
        )
        lifecycle = next(
            candidate
            for candidate in by_position[PredictionPosition.TASK_UPDATE].candidates
            if candidate.candidate_id == "cross_position_deduct"
        )
        self.assertTrue(lifecycle.graph.is_lifecycle)
        self.assertEqual(lifecycle.graph.initializer_estimator_id, "empirical_quantile")
        self.assertEqual(lifecycle.graph.updater_estimator_id, "cross_position_deduct")

    def test_sokoban_routes_the_full_launch_and_lifecycle_matrix(self) -> None:
        dataset = _dataset()
        rows = tuple(
            replace(
                row,
                point=replace(row.point, condition_id="condition:effa60eb1d4380d124bf"),
            )
            for row in dataset.rows
            if row.point.condition_id == PRIMARY_CONDITION
        )
        sokoban = replace(dataset, dataset_id="5" * 64, rows=rows)
        matrix = build_stage2_matrix(
            build_development_protocol(sokoban),
            source_id=BAGEN_SOKOBAN_SOURCE_ID,
        )
        self.assertEqual(len(matrix.experiments), 2)
        self.assertEqual(
            {spec.position for spec in matrix.experiments},
            {PredictionPosition.TASK_LAUNCH, PredictionPosition.TASK_UPDATE},
        )
        update = next(
            spec
            for spec in matrix.experiments
            if spec.position == PredictionPosition.TASK_UPDATE
        )
        self.assertIn(
            "cross_position_deduct",
            {candidate.candidate_id for candidate in update.candidates},
        )

    def test_sparse_and_absent_frozen_conditions_fail_closed(self) -> None:
        matrix = build_stage2_matrix(
            build_development_protocol(_dataset()),
            source_id=BAGEN_SOURCE_ID,
        )
        active_conditions = {spec.condition_id for spec in matrix.experiments}
        self.assertEqual(active_conditions, {PRIMARY_CONDITION})
        cell_gates = tuple(gate for gate in matrix.gates if gate.scope == "cell")
        self.assertEqual(
            {gate.condition_id for gate in cell_gates},
            FROZEN_BAGEN_CONDITIONS - {PRIMARY_CONDITION},
        )
        sparse_update = next(
            gate
            for gate in cell_gates
            if gate.condition_id == SPARSE_CONDITION
            and gate.position == PredictionPosition.TASK_UPDATE
        )
        self.assertEqual(
            sparse_update.reason,
            "insufficient_development_tasks_for_five_fold_cv",
        )

    def test_missing_request_chars_gates_only_the_length_candidate(self) -> None:
        matrix = build_stage2_matrix(
            build_development_protocol(_dataset(request_chars=False)),
            source_id=BAGEN_SOURCE_ID,
        )
        update = next(
            spec
            for spec in matrix.experiments
            if spec.condition_id == PRIMARY_CONDITION
            and spec.position == PredictionPosition.TASK_UPDATE
        )
        self.assertNotIn(
            "request_chars_length",
            {candidate.candidate_id for candidate in update.candidates},
        )
        gate = next(
            gate
            for gate in matrix.gates
            if gate.scope == "candidate"
            and gate.condition_id == PRIMARY_CONDITION
            and gate.position == PredictionPosition.TASK_UPDATE
        )
        self.assertEqual(gate.candidate_id, "request_chars_length")
        self.assertEqual(
            gate.reason,
            "request_content_chars_missing_on_scored_cohort",
        )

    def test_final_holdout_changes_cannot_change_the_matrix(self) -> None:
        dataset = _dataset()
        first_protocol = build_development_protocol(dataset)
        changed_rows = tuple(
            replace(
                row,
                label=int(row.label or 0) + 1_000_000,
                point=replace(row.point, features={"suffix_only": 999}),
            )
            if row.point.task_id in first_protocol.final_holdout_tasks
            else row
            for row in dataset.rows
        )
        second_protocol = build_development_protocol(
            replace(dataset, dataset_id="e" * 64, rows=changed_rows)
        )
        first = build_stage2_matrix(first_protocol, source_id=BAGEN_SOURCE_ID)
        second = build_stage2_matrix(second_protocol, source_id=BAGEN_SOURCE_ID)
        self.assertEqual(first, second)

    def test_identity_and_source_fail_closed(self) -> None:
        matrix = build_stage2_matrix(
            build_development_protocol(_dataset()),
            source_id=BAGEN_SOURCE_ID,
        )
        with self.assertRaisesRegex(ValueError, "matrix id"):
            Stage2Matrix(
                source_id=matrix.source_id,
                development_protocol_id=matrix.development_protocol_id,
                experiments=matrix.experiments,
                gates=matrix.gates,
                matrix_id="0" * 64,
            )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            build_stage2_matrix(
                build_development_protocol(_dataset()),
                source_id="unknown",
            )


if __name__ == "__main__":
    unittest.main()
