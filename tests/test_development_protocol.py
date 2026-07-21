from __future__ import annotations

import copy
import hashlib
import json
import unittest
from dataclasses import replace

from token_prediction.dataset.schema import (
    DatasetRow,
    LabelStatus,
    PredictionPoint,
    PredictionPosition,
    PredictionTarget,
    SupervisedDataset,
)
from token_prediction.development import (
    STAGE_SPLIT_SEEDS,
    build_development_protocol,
    verify_development_audit_document,
)


def _dataset(*, task_count: int = 100, dataset_id: str = "a" * 64) -> SupervisedDataset:
    rows: list[DatasetRow] = []
    for task_index in range(task_count):
        task_id = f"private-task-{task_index:03d}"
        for variant in range(4):
            rows.append(
                DatasetRow(
                    point=PredictionPoint(
                        point_id=f"point-{task_index:03d}-{variant}",
                        source_event_id=f"event-{task_index:03d}-{variant}",
                        task_id=task_id,
                        trajectory_id=f"trajectory-{task_index:03d}-{variant}",
                        run_id=f"run-{variant % 2}",
                        prediction_context_id=f"context-{task_index:03d}-{variant}",
                        condition_id=f"family-{variant // 2}",
                        logical_call_id=None,
                        attempt_id=None,
                        cutoff_event_seq=variant,
                        position=PredictionPosition.TASK_PRE,
                        target=(PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS),
                        features={"visible_prefix_value": task_index + variant},
                        known_offset_tokens=0,
                    ),
                    label=task_index + variant,
                    status=LabelStatus.OBSERVED,
                )
            )
    return SupervisedDataset(
        dataset_id=dataset_id,
        rows=tuple(reversed(rows)),
        schema_version=2,
        source_descriptor_hash="b" * 64,
        capability_contract_hash="c" * 64,
        input_contract_hash="e" * 64,
    )


class DevelopmentProtocolTests(unittest.TestCase):
    def test_task_only_assignment_ignores_labels_and_suffix_values(self) -> None:
        dataset = _dataset()
        first = build_development_protocol(dataset)
        changed_rows = tuple(
            replace(
                row,
                point=replace(
                    row.point,
                    source_event_id=f"{row.point.source_event_id}-changed-suffix",
                    cutoff_event_seq=row.point.cutoff_event_seq + 10_000,
                    features={"visible_prefix_value": -1, "new_suffix_feature": 999},
                ),
                label=1_000_000 + index,
            )
            for index, row in enumerate(dataset.rows)
        )
        changed = build_development_protocol(replace(dataset, rows=changed_rows))

        self.assertEqual(first.holdout_plan.assignments, changed.holdout_plan.assignments)
        self.assertEqual(first.holdout_plan.assignment_id, changed.holdout_plan.assignment_id)
        self.assertNotEqual(
            first.development_dataset.dataset_id,
            changed.development_dataset.dataset_id,
        )
        self.assertNotEqual(first.protocol_id, changed.protocol_id)
        self.assertEqual(
            tuple(plan.assignments for plan in first.outer_plans),
            tuple(plan.assignments for plan in changed.outer_plans),
        )
        self.assertEqual(
            tuple(plan.assignment.assignments for plan in first.inner_plans),
            tuple(plan.assignment.assignments for plan in changed.inner_plans),
        )

    def test_final_holdout_label_change_cannot_change_development_identity(self) -> None:
        dataset = _dataset()
        first = build_development_protocol(dataset)
        final_tasks = first.final_holdout_tasks
        changed_rows = tuple(
            replace(row, label=int(row.label or 0) + 10_000)
            if row.point.task_id in final_tasks
            else row
            for row in dataset.rows
        )
        changed = build_development_protocol(
            replace(dataset, dataset_id="d" * 64, rows=changed_rows)
        )

        self.assertNotEqual(first.parent_dataset_id, changed.parent_dataset_id)
        self.assertNotEqual(
            first.holdout_plan.holdout_plan_id,
            changed.holdout_plan.holdout_plan_id,
        )
        self.assertEqual(first.development_dataset, changed.development_dataset)
        self.assertEqual(first.protocol_id, changed.protocol_id)
        self.assertEqual(first.outer_plans, changed.outer_plans)
        self.assertEqual(first.inner_plans, changed.inner_plans)

    def test_all_runs_and_families_for_a_task_stay_in_one_cohort(self) -> None:
        dataset = _dataset()
        protocol = build_development_protocol(dataset)
        development_point_ids = {row.point.point_id for row in protocol.development_dataset.rows}
        for task_id in dataset.task_ids:
            task_point_ids = {
                row.point.point_id for row in dataset.rows if row.point.task_id == task_id
            }
            self.assertEqual(len(task_point_ids), 4)
            if task_id in protocol.holdout_plan.development_tasks:
                self.assertTrue(task_point_ids <= development_point_ids)
            else:
                self.assertFalse(task_point_ids & development_point_ids)

    def test_three_seed_nested_plans_are_reproducible_and_nonempty(self) -> None:
        first = build_development_protocol(_dataset())
        second = build_development_protocol(_dataset())
        self.assertEqual(first, second)
        self.assertEqual(first.split_seeds, STAGE_SPLIT_SEEDS)
        self.assertEqual(tuple(plan.seed for plan in first.outer_plans), STAGE_SPLIT_SEEDS)
        self.assertEqual(len(first.inner_plans), 15)
        self.assertEqual(
            tuple(plan.split_seed for plan in first.outer_inner_plans),
            STAGE_SPLIT_SEEDS,
        )
        self.assertTrue(all(len(plan.inner_plans) == 5 for plan in first.outer_inner_plans))
        for nested in first.outer_inner_plans:
            self.assertEqual(
                first.nested_plan_for(nested.outer_plan),
                nested,
            )

        inner_by_key = {(plan.split_seed, plan.outer_test_fold): plan for plan in first.inner_plans}
        for outer in first.outer_plans:
            self.assertEqual(outer.folds, 5)
            self.assertEqual(
                frozenset(task for task, _fold in outer.assignments),
                first.development_dataset.task_ids,
            )
            for outer_test_fold in range(5):
                outer_train = outer.partition(outer_test_fold).train_tasks
                inner = inner_by_key[(outer.seed, outer_test_fold)].assignment
                self.assertEqual(inner.task_ids, outer_train)
                for inner_holdout_fold in range(5):
                    partition = inner.partition(inner_holdout_fold)
                    self.assertTrue(partition.initializer_fit_tasks)
                    self.assertTrue(partition.validation_tasks)
                    self.assertTrue(partition.holdout_tasks)
                    self.assertEqual(
                        partition.initializer_fit_tasks
                        | partition.validation_tasks
                        | partition.holdout_tasks,
                        outer_train,
                    )

    def test_final_holdout_is_sealed_from_every_development_layer(self) -> None:
        dataset = _dataset()
        protocol = build_development_protocol(dataset)
        final_tasks = protocol.final_holdout_tasks
        self.assertTrue(final_tasks)
        self.assertFalse(final_tasks & protocol.development_dataset.task_ids)
        self.assertEqual(
            {row.point.task_id for row in dataset.rows if row.point.task_id in final_tasks},
            final_tasks,
        )
        for outer in protocol.outer_plans:
            self.assertFalse(final_tasks & frozenset(task for task, _fold in outer.assignments))
            for fold in range(5):
                partition = outer.partition(fold)
                self.assertFalse(final_tasks & partition.train_tasks)
                self.assertFalse(final_tasks & partition.validation_tasks)
                self.assertFalse(final_tasks & partition.calibration_tasks)
                self.assertFalse(final_tasks & partition.test_tasks)
        for inner in protocol.inner_plans:
            self.assertFalse(final_tasks & inner.assignment.task_ids)

        leaked_dataset = replace(
            protocol.development_dataset,
            rows=protocol.development_dataset.rows
            + (next(row for row in dataset.rows if row.point.task_id in final_tasks),),
        )
        with self.assertRaisesRegex(ValueError, "exact task projection"):
            replace(protocol, development_dataset=leaked_dataset)

    def test_development_identity_binds_projection_assignment_and_provenance(self) -> None:
        first = build_development_protocol(_dataset(dataset_id="a" * 64))
        different_parent = build_development_protocol(_dataset(dataset_id="d" * 64))
        self.assertEqual(
            first.holdout_plan.assignment_id, different_parent.holdout_plan.assignment_id
        )
        self.assertEqual(
            first.development_dataset.dataset_id,
            different_parent.development_dataset.dataset_id,
        )
        self.assertEqual(first.protocol_id, different_parent.protocol_id)
        self.assertNotEqual(
            first.holdout_plan.holdout_plan_id,
            different_parent.holdout_plan.holdout_plan_id,
        )
        self.assertEqual(first.development_dataset.schema_version, 2)
        self.assertEqual(first.development_dataset.source_descriptor_hash, "b" * 64)
        self.assertEqual(first.development_dataset.capability_contract_hash, "c" * 64)
        self.assertEqual(first.development_dataset.input_contract_hash, "e" * 64)

        different_input_contract = build_development_protocol(
            replace(_dataset(), input_contract_hash="f" * 64)
        )
        self.assertNotEqual(first.protocol_id, different_input_contract.protocol_id)
        self.assertNotEqual(
            first.development_dataset.dataset_id,
            different_input_contract.development_dataset.dataset_id,
        )

        different_assignment = build_development_protocol(
            _dataset(), holdout_salt="different-frozen-holdout-salt"
        )
        self.assertNotEqual(
            first.holdout_plan.assignment_id,
            different_assignment.holdout_plan.assignment_id,
        )
        self.assertNotEqual(
            first.development_dataset.dataset_id,
            different_assignment.development_dataset.dataset_id,
        )

    def test_small_development_cohort_fails_closed_before_nested_cv(self) -> None:
        with self.assertRaisesRegex(ValueError, "undersized|too few"):
            build_development_protocol(_dataset(task_count=12))

    def test_public_audit_is_pseudonymized_serializable_and_tamper_evident(self) -> None:
        protocol = build_development_protocol(_dataset())
        audit = protocol.to_audit_document()
        encoded = json.dumps(audit, ensure_ascii=False, sort_keys=True)
        for task_id, _cohort in protocol.holdout_plan.assignments:
            self.assertNotIn(task_id, encoded)
        json.loads(encoded)
        verify_development_audit_document(audit)

        changed = copy.deepcopy(audit)
        changed["outer_plans"][0]["assignments"][0]["fold"] = (
            changed["outer_plans"][0]["assignments"][0]["fold"] + 1
        ) % 5
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            verify_development_audit_document(changed)

        structurally_tampered = copy.deepcopy(audit)
        holdout_assignment = next(
            item
            for item in structurally_tampered["permanent_holdout"]["assignments"]
            if item["cohort"] == "final_holdout"
        )
        structurally_tampered["outer_plans"][0]["assignments"][0]["task_pseudonym"] = (
            holdout_assignment["task_pseudonym"]
        )
        del structurally_tampered["audit_sha256"]
        structurally_tampered["audit_sha256"] = hashlib.sha256(
            json.dumps(
                structurally_tampered,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "task universe|duplicate|canonical"):
            verify_development_audit_document(structurally_tampered)

    def test_protocol_and_audit_identity_reject_tampering(self) -> None:
        protocol = build_development_protocol(_dataset())
        with self.assertRaisesRegex(ValueError, "protocol id"):
            replace(protocol, protocol_id="0" * 64)

        audit = protocol.to_audit_document()
        tampered = copy.deepcopy(audit)
        tampered["protocol_id"] = "0" * 64
        del tampered["audit_sha256"]
        tampered["audit_sha256"] = hashlib.sha256(
            json.dumps(
                tampered,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "protocol id"):
            verify_development_audit_document(tampered)


if __name__ == "__main__":
    unittest.main()
