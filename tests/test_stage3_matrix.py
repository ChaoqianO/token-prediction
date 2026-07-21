from __future__ import annotations

import unittest
from dataclasses import replace

from tests.test_stage2_matrix import PRIMARY_CONDITION, _dataset

from token_prediction.development import build_development_protocol
from token_prediction.experiment import validate_ablation_specs
from token_prediction.stage2_matrix import (
    BAGEN_SOURCE_ID,
    FROZEN_BAGEN_CONDITIONS,
    SPEND_AGGREGATE_SOURCE_ID,
)
from token_prediction.stage3_matrix import Stage3Matrix, build_stage3_matrix


class Stage3MatrixTests(unittest.TestCase):
    def test_matrix_keeps_key_comparators_and_adds_single_axis_gru_ablations(self) -> None:
        matrix = build_stage3_matrix(
            build_development_protocol(_dataset()),
            source_id=BAGEN_SOURCE_ID,
        )
        self.assertEqual(len(matrix.experiments), 1)
        spec = matrix.experiments[0]
        by_id = {candidate.candidate_id: candidate for candidate in spec.candidates}
        self.assertEqual(
            set(by_id),
            {
                "empirical",
                "cross_position_deduct",
                "lightgbm_history",
                "mlp_history",
                "gru_residual",
                "gru_no_recurrence",
                "gru_zero_residual",
            },
        )
        self.assertEqual(by_id["gru_residual"].params["transition_dim"], 64)
        self.assertEqual(by_id["gru_residual"].params["hidden_dim"], 64)
        self.assertEqual(by_id["gru_residual"].params["residual_head_dim"], 64)
        self.assertEqual(
            by_id["gru_residual"].graph.initializer_estimator_id,
            "empirical_quantile",
        )
        self.assertEqual(
            by_id["gru_residual"].graph.updater_estimator_id,
            "gru_residual",
        )
        self.assertEqual(
            by_id["gru_no_recurrence"].ablation.allowed_config_paths,
            frozenset({"params.no_recurrence"}),
        )
        self.assertEqual(
            by_id["gru_zero_residual"].ablation.allowed_config_paths,
            frozenset({"params.residual_scale"}),
        )
        validate_ablation_specs(spec.candidates)

    def test_sparse_and_absent_conditions_fail_closed(self) -> None:
        matrix = build_stage3_matrix(
            build_development_protocol(_dataset()),
            source_id=BAGEN_SOURCE_ID,
        )
        self.assertEqual(
            {gate.condition_id for gate in matrix.gates},
            FROZEN_BAGEN_CONDITIONS - {PRIMARY_CONDITION},
        )
        self.assertTrue(
            all(
                gate.reason
                in {
                    "capability_or_observed_development_lifecycle_unavailable",
                    "insufficient_development_tasks_for_five_fold_cv",
                }
                for gate in matrix.gates
            )
        )

    def test_aggregate_source_is_explicitly_non_lifecycle(self) -> None:
        matrix = build_stage3_matrix(
            build_development_protocol(_dataset()),
            source_id=SPEND_AGGREGATE_SOURCE_ID,
        )
        self.assertFalse(matrix.experiments)
        self.assertEqual(len(matrix.gates), 1)
        self.assertEqual(
            matrix.gates[0].reason,
            "aggregate_source_has_no_request_boundary_lifecycle",
        )

    def test_final_holdout_changes_cannot_change_matrix_identity(self) -> None:
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
            replace(dataset, dataset_id="f" * 64, rows=changed_rows)
        )
        self.assertEqual(
            build_stage3_matrix(first_protocol, source_id=BAGEN_SOURCE_ID),
            build_stage3_matrix(second_protocol, source_id=BAGEN_SOURCE_ID),
        )

    def test_matrix_identity_and_unknown_source_fail_closed(self) -> None:
        matrix = build_stage3_matrix(
            build_development_protocol(_dataset()),
            source_id=BAGEN_SOURCE_ID,
        )
        with self.assertRaisesRegex(ValueError, "matrix id"):
            Stage3Matrix(
                source_id=matrix.source_id,
                development_protocol_id=matrix.development_protocol_id,
                experiments=matrix.experiments,
                gates=matrix.gates,
                matrix_id="0" * 64,
            )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            build_stage3_matrix(
                build_development_protocol(_dataset()),
                source_id="unknown",
            )


if __name__ == "__main__":
    unittest.main()
