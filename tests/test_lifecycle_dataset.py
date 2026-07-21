from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace

from token_prediction.contracts import Observable, SourceCapabilities, SourceDescriptor
from token_prediction.dataset import (
    DatasetRow,
    LabelStatus,
    PredictionPosition,
    PredictionTarget,
    SupervisedDataset,
    build_capability_supervised_dataset,
    build_lifecycle_slice,
    build_prediction_points,
)
from token_prediction.dataset.lifecycle import lifecycle_condition_task_ids

from tests.helpers import make_two_call_trajectory


def _descriptor() -> SourceDescriptor:
    return SourceDescriptor(
        source_id="lifecycle-fixture",
        revision="revision-1",
        manifest_path="workspace/manifests/lifecycle-fixture.json",
        manifest_sha256="c" * 64,
        capabilities=SourceCapabilities(
            source_id="lifecycle-fixture",
            observables=frozenset(
                {
                    Observable.ATTEMPT_USAGE,
                    Observable.REQUEST_BOUNDARIES,
                    Observable.TASK_TERMINATION,
                    Observable.TASK_USAGE,
                }
            ),
        ),
    )


def _replace_rows(
    dataset: SupervisedDataset,
    replacements: dict[str, DatasetRow],
    *,
    dataset_id: str | None = None,
) -> SupervisedDataset:
    return replace(
        dataset,
        dataset_id=dataset_id or dataset.dataset_id,
        rows=tuple(replacements.get(row.point.point_id, row) for row in dataset.rows),
    )


class LifecycleDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trajectories = tuple(
            make_two_call_trajectory(task, run) for task in range(2) for run in range(2)
        )
        self.descriptor = _descriptor()
        self.point_set = build_prediction_points(
            self.trajectories,
            self.descriptor,
        )
        self.dataset = build_capability_supervised_dataset(
            self.trajectories,
            self.descriptor,
        )
        self.target = PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS

    def _target_rows(self) -> tuple[DatasetRow, ...]:
        return tuple(row for row in self.dataset.rows if row.point.target == self.target)

    def test_sequence_keeps_every_boundary_and_unscored_missing_context(self) -> None:
        target_rows = self._target_rows()
        missing_row = next(
            row
            for row in target_rows
            if row.point.task_id == "task-0"
            and row.point.run_id == "task0-run0"
            and row.point.position == PredictionPosition.TASK_UPDATE
        )
        censored_row = next(
            row
            for row in target_rows
            if row.point.task_id == "task-1"
            and row.point.run_id == "task1-run0"
            and row.point.position == PredictionPosition.TASK_PRE
        )
        changed = _replace_rows(
            self.dataset,
            {
                missing_row.point.point_id: replace(
                    missing_row,
                    label=None,
                    status=LabelStatus.MISSING,
                    invalid_reason="missing_usage",
                ),
                censored_row.point.point_id: replace(
                    censored_row,
                    label=None,
                    status=LabelStatus.CENSORED,
                    invalid_reason="step_limit",
                ),
            },
            dataset_id="d" * 64,
        )
        lifecycle = build_lifecycle_slice(changed, target=self.target)
        self.assertEqual(len(lifecycle.sequences), 4)
        self.assertTrue(all(len(sequence.steps) == 2 for sequence in lifecycle.sequences))
        by_id = {step.point.point_id: step for step in lifecycle.steps}
        for point_id in (missing_row.point.point_id, censored_row.point.point_id):
            step = by_id[point_id]
            self.assertTrue(step.is_context_only)
            self.assertEqual(step.sample_weight, 0.0)
        self.assertTrue(
            all(
                sequence.steps[0].is_context_only and sequence.steps[0].sample_weight == 0.0
                for sequence in lifecycle.sequences
            )
        )
        # Task-pre is seed/context only.  Task 0's missing-update run has no
        # updater loss; its other run owns the task's sole unit of weight.
        task_zero = [step for step in lifecycle.scored_steps if step.point.task_id == "task-0"]
        self.assertEqual([step.sample_weight for step in task_zero], [1.0])
        for task_id in lifecycle.task_ids:
            self.assertAlmostEqual(
                sum(
                    step.sample_weight
                    for step in lifecycle.scored_steps
                    if step.point.task_id == task_id
                ),
                1.0,
            )

    def test_label_value_mutation_does_not_change_context_or_scored_hash(self) -> None:
        baseline = build_lifecycle_slice(
            self.dataset,
            target=self.target,
            point_set=self.point_set,
        )
        without_explicit_points = build_lifecycle_slice(
            self.dataset,
            target=self.target,
        )
        self.assertEqual(
            baseline.input_contract_hash,
            without_explicit_points.input_contract_hash,
        )
        self.assertEqual(baseline.context_hash, without_explicit_points.context_hash)
        self.assertEqual(baseline.scored_hash, without_explicit_points.scored_hash)
        self.assertTrue(
            all(
                not sequence.steps[0].loss_mask
                and not sequence.steps[0].score_mask
                and sequence.steps[0].sample_weight == 0.0
                for sequence in baseline.sequences
            )
        )
        row = self._target_rows()[0]
        changed_row = replace(row, label=int(row.label or 0) + 999)
        changed = _replace_rows(
            self.dataset,
            {row.point.point_id: changed_row},
            dataset_id=hashlib.sha256(b"changed-label-dataset").hexdigest(),
        )
        mutated = build_lifecycle_slice(
            changed,
            target=self.target,
            point_set=self.point_set,
        )
        self.assertEqual(baseline.context_hash, mutated.context_hash)
        self.assertEqual(baseline.scored_hash, mutated.scored_hash)
        self.assertNotEqual(
            baseline.sequences[0].steps[0].label,
            mutated.sequences[0].steps[0].label,
        )

    def test_status_changes_only_scored_identity_not_context_identity(self) -> None:
        baseline = build_lifecycle_slice(self.dataset, target=self.target)
        row = next(
            current
            for current in self._target_rows()
            if current.point.position == PredictionPosition.TASK_UPDATE
        )
        changed = _replace_rows(
            self.dataset,
            {
                row.point.point_id: replace(
                    row,
                    label=None,
                    status=LabelStatus.CENSORED,
                    invalid_reason="step_limit",
                )
            },
            dataset_id="e" * 64,
        )
        mutated = build_lifecycle_slice(changed, target=self.target)
        self.assertEqual(baseline.context_hash, mutated.context_hash)
        self.assertNotEqual(baseline.scored_hash, mutated.scored_hash)

    def test_scored_cohort_can_use_other_tasks_as_unscored_context(self) -> None:
        lifecycle = build_lifecycle_slice(
            self.dataset,
            target=self.target,
            scored_task_ids={"task-0"},
        )
        task_one = [step for step in lifecycle.steps if step.point.task_id == "task-1"]
        self.assertTrue(task_one)
        self.assertTrue(all(step.is_context_only for step in task_one))
        self.assertTrue(all(step.sample_weight == 0.0 for step in task_one))

    def test_condition_cell_accepts_only_its_actual_task_subset(self) -> None:
        original_condition = self.dataset.rows[0].point.condition_id
        changed = replace(
            self.dataset,
            dataset_id="9" * 64,
            rows=tuple(
                replace(
                    row,
                    point=replace(row.point, condition_id="family-b"),
                )
                if row.point.task_id == "task-1"
                else row
                for row in self.dataset.rows
            ),
        )
        condition_tasks = lifecycle_condition_task_ids(
            changed,
            target=self.target,
            condition_id=original_condition,
        )
        self.assertEqual(condition_tasks, frozenset({"task-0"}))
        lifecycle = build_lifecycle_slice(
            changed,
            target=self.target,
            condition_id=original_condition,
            task_ids=condition_tasks,
        )
        self.assertEqual(lifecycle.task_ids, frozenset({"task-0"}))
        with self.assertRaisesRegex(ValueError, "absent from the selected condition"):
            build_lifecycle_slice(
                changed,
                target=self.target,
                condition_id=original_condition,
                task_ids=changed.task_ids,
            )

    def test_scored_hash_rejects_weight_tampering(self) -> None:
        lifecycle = build_lifecycle_slice(self.dataset, target=self.target)
        sequence = lifecycle.sequences[0]
        scored_index = next(index for index, step in enumerate(sequence.steps) if step.score_mask)
        steps = list(sequence.steps)
        steps[scored_index] = replace(
            steps[scored_index],
            sample_weight=steps[scored_index].sample_weight * 2,
        )
        with self.assertRaisesRegex(ValueError, "scored_hash"):
            replace(sequence, steps=tuple(steps))

    def test_structural_corruption_and_invalid_status_fail_closed(self) -> None:
        first = self._target_rows()[0]
        corrupted_point = replace(
            first.point,
            position=PredictionPosition.TASK_UPDATE,
        )
        corrupted = _replace_rows(
            self.dataset,
            {first.point.point_id: replace(first, point=corrupted_point)},
        )
        with self.assertRaisesRegex(ValueError, "must start at Task-pre"):
            build_lifecycle_slice(corrupted, target=self.target)

        invalid = _replace_rows(
            self.dataset,
            {
                first.point.point_id: replace(
                    first,
                    label=None,
                    status=LabelStatus.INVALID,
                    invalid_reason="broken_event_order",
                )
            },
        )
        with self.assertRaisesRegex(ValueError, "invalid trajectory"):
            build_lifecycle_slice(invalid, target=self.target)

    def test_prefix_point_join_is_exact_and_missing_features_are_not_dropped(self) -> None:
        lifecycle = build_lifecycle_slice(
            self.dataset,
            target=self.target,
            point_set=self.point_set,
        )
        first = lifecycle.sequences[0].steps[0]
        self.assertIn("last_call_output_tokens", first.point.features)
        self.assertIsNone(first.point.features["last_call_output_tokens"])
        incomplete_points = build_prediction_points(
            self.trajectories[:-1],
            self.descriptor,
        )
        with self.assertRaisesRegex(ValueError, "exact prefix point join"):
            build_lifecycle_slice(
                self.dataset,
                target=self.target,
                point_set=incomplete_points,
            )


if __name__ == "__main__":
    unittest.main()
