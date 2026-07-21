from __future__ import annotations

import unittest

from token_prediction.dataset import (
    PredictionPosition,
    PredictionTarget,
    assign_task_folds,
    build_supervised_dataset,
    make_task_split_plan,
)
from token_prediction.contracts import EventType
from token_prediction.trajectory import Trajectory

from tests.helpers import event, make_two_call_trajectory


class DatasetContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = build_supervised_dataset(
            make_two_call_trajectory(task, run)
            for task in range(5)
            for run in range(2)
        )

    def test_explicit_point_join_builds_all_supported_request_cells(self) -> None:
        self.assertEqual(len(self.dataset.rows), 90)
        self.assertEqual(len({row.point.point_id for row in self.dataset.rows}), 90)
        task_pre = self.dataset.select(
            PredictionPosition.TASK_PRE,
            PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
        )
        task_update = self.dataset.select(
            PredictionPosition.TASK_UPDATE,
            PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
        )
        call_pre = self.dataset.select(
            PredictionPosition.CALL_PRE,
            PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS,
        )
        self.assertEqual(
            (len(task_pre.rows), len(task_update.rows), len(call_pre.rows)),
            (10, 10, 20),
        )
        self.assertTrue(
            all(row.point.source_event_id in row.point.point_id for row in self.dataset.rows)
        )

    def test_task_weight_is_conserved_despite_runs_and_points(self) -> None:
        selected = self.dataset.select(
            PredictionPosition.CALL_PRE,
            PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS,
        )
        totals: dict[str, float] = {}
        for weighted in selected.weighted_rows():
            task_id = weighted.row.point.task_id
            totals[task_id] = totals.get(task_id, 0.0) + weighted.sample_weight
        self.assertEqual(set(totals), {f"task-{index}" for index in range(5)})
        for total in totals.values():
            self.assertAlmostEqual(total, 1.0)

    def test_split_is_task_grouped_and_deterministic(self) -> None:
        first = make_task_split_plan(
            self.dataset.task_ids,
            dataset_id=self.dataset.dataset_id,
            folds=5,
            seed=7,
        )
        second = make_task_split_plan(
            reversed(sorted(self.dataset.task_ids)),
            dataset_id=self.dataset.dataset_id,
            folds=5,
            seed=7,
        )
        self.assertEqual(first, second)
        for fold in range(5):
            partition = first.partition(fold)
            self.assertEqual(len(partition.test_tasks), 1)
            self.assertEqual(len(partition.calibration_tasks), 1)
            self.assertEqual(len(partition.validation_tasks), 1)
            self.assertEqual(len(partition.train_tasks), 2)

    def test_task_assignment_can_be_frozen_before_point_expansion(self) -> None:
        trajectories = tuple(make_two_call_trajectory(task, 0) for task in range(5))
        assignment = assign_task_folds(
            (trajectory.task_id for trajectory in trajectories), folds=5, seed=9
        )
        later_dataset = build_supervised_dataset(trajectories)
        plan = assignment.bind(later_dataset.dataset_id)
        self.assertEqual(assignment.task_ids, later_dataset.task_ids)
        plan.validate_tasks(later_dataset.task_ids)

    def test_real_generation_checkpoint_builds_call_update_target(self) -> None:
        prefix = "checkpoint"
        trajectory = Trajectory.from_events(
            (
                event(prefix, 0, EventType.TASK_STARTED, payload={"task_id": "task"}),
                event(
                    prefix,
                    1,
                    EventType.REQUEST_BUILT,
                    call_id="call",
                    payload={"request_tokens_local": 10},
                ),
                event(
                    prefix,
                    2,
                    EventType.API_ATTEMPT_STARTED,
                    call_id="call",
                    attempt_id="attempt",
                ),
                event(
                    prefix,
                    3,
                    EventType.GENERATION_CHECKPOINT,
                    call_id="call",
                    attempt_id="attempt",
                    payload={
                        "generated_tokens_so_far": 4,
                        "stop_prob_mean_16": 0.2,
                        "next_token_entropy_mean_16": 1.5,
                    },
                ),
                event(
                    prefix,
                    4,
                    EventType.API_COMPLETED,
                    call_id="call",
                    attempt_id="attempt",
                    payload={"usage": {"input_tokens": 10, "output_tokens": 10}},
                ),
                event(prefix, 5, EventType.TASK_FINISHED),
            )
        )
        selected = build_supervised_dataset((trajectory,)).select(
            PredictionPosition.CALL_UPDATE,
            PredictionTarget.CALL_REMAINING_OUTPUT_TOKENS,
        )
        self.assertEqual(len(selected.rows), 1)
        self.assertEqual(selected.rows[0].label, 6)
        self.assertEqual(selected.rows[0].point.features["generated_tokens_so_far"], 4)

    def test_mixed_model_conditions_require_an_explicit_experiment_cell(self) -> None:
        first = make_two_call_trajectory(0, 0)
        second_source = make_two_call_trajectory(1, 0)
        second_events = list(second_source.events)
        second_events[0] = second_events[0].with_payload(
            {**second_events[0].payload, "condition_id": "condition:other-model"}
        )
        second = Trajectory.from_events(second_events)
        dataset = build_supervised_dataset((first, second))
        with self.assertRaisesRegex(ValueError, "mixes execution conditions"):
            dataset.select(
                PredictionPosition.CALL_PRE,
                PredictionTarget.CALL_UNKNOWN_BILLABLE_TOKENS,
            )
        selected = dataset.select(
            PredictionPosition.CALL_PRE,
            PredictionTarget.CALL_UNKNOWN_BILLABLE_TOKENS,
            condition_id=first.condition_id,
        )
        self.assertTrue(selected.rows)
        self.assertTrue(
            all(row.point.condition_id == first.condition_id for row in selected.rows)
        )

    def test_dataset_rejects_non_object_usage_instead_of_treating_it_as_missing(self) -> None:
        trajectory = Trajectory.from_events(
            (
                event("malformed", 0, EventType.TASK_STARTED, payload={"task_id": "task"}),
                event(
                    "malformed",
                    1,
                    EventType.REQUEST_BUILT,
                    call_id="call",
                ),
                event(
                    "malformed",
                    2,
                    EventType.API_ATTEMPT_STARTED,
                    call_id="call",
                    attempt_id="attempt",
                ),
                event(
                    "malformed",
                    3,
                    EventType.API_COMPLETED,
                    call_id="call",
                    attempt_id="attempt",
                    payload={"usage": "not-an-object"},
                ),
                event("malformed", 4, EventType.TASK_FINISHED),
            )
        )
        with self.assertRaisesRegex(ValueError, "token usage must be an object"):
            build_supervised_dataset((trajectory,))


if __name__ == "__main__":
    unittest.main()
