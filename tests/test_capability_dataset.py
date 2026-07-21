from __future__ import annotations

import unittest

from token_prediction.contracts import Observable, SourceCapabilities, SourceDescriptor
from token_prediction.dataset import (
    CAPABILITY_DATASET_SCHEMA_VERSION,
    V2_EXCLUDED_LOCAL_FEATURES,
    PredictionPosition,
    PredictionTarget,
    build_capability_supervised_dataset,
    decide_target_capability,
)
from token_prediction.trajectory import Trajectory

from tests.helpers import make_two_call_trajectory


def _descriptor(
    *,
    local_count: bool = False,
    extra: frozenset[Observable] = frozenset(),
) -> SourceDescriptor:
    observables = {
        Observable.ATTEMPT_USAGE,
        Observable.REQUEST_BOUNDARIES,
        Observable.TASK_TERMINATION,
        Observable.TASK_USAGE,
        *extra,
    }
    if local_count:
        observables.add(Observable.REQUEST_LOCAL_COUNT)
    capabilities = SourceCapabilities(
        source_id="fixture-source",
        observables=frozenset(observables),
    )
    return SourceDescriptor(
        source_id="fixture-source",
        revision="revision-1",
        manifest_path="workspace/manifests/fixture.json",
        manifest_sha256="a" * 64,
        capabilities=capabilities,
    )


def _change_request_local_counts(trajectory: Trajectory) -> Trajectory:
    changed = []
    for current in trajectory.events:
        if current.event_id.endswith("-e1"):
            current = current.with_payload(
                {**current.payload, "request_tokens_local": 9_001}
            )
        elif current.event_id.endswith("-e5"):
            current = current.with_payload(
                {**current.payload, "request_tokens_local": 9_999}
            )
        changed.append(current)
    return Trajectory.from_events(changed)


def _change_second_call_usage(trajectory: Trajectory) -> Trajectory:
    changed = []
    for current in trajectory.events:
        if current.event_id.endswith("-e7"):
            current = current.with_payload(
                {
                    "usage": {
                        "input_tokens": 180,
                        "output_tokens": 17,
                        "total_tokens": 197,
                    }
                }
            )
        changed.append(current)
    return Trajectory.from_events(changed)


class CapabilityDatasetTests(unittest.TestCase):
    def test_target_requirements_fail_closed_without_local_count(self) -> None:
        capabilities = _descriptor().capabilities
        local = decide_target_capability(
            capabilities,
            PredictionPosition.TASK_PRE,
            PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
        )
        provider = decide_target_capability(
            capabilities,
            PredictionPosition.TASK_PRE,
            PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
        )
        call_total = decide_target_capability(
            capabilities,
            PredictionPosition.CALL_PRE,
            PredictionTarget.CALL_BILLABLE_TOTAL_TOKENS,
        )
        unsupported = decide_target_capability(
            capabilities,
            PredictionPosition.CALL_PRE,
            PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
        )

        self.assertTrue(local.gated)
        self.assertEqual(local.missing_observables, ("request_local_count",))
        self.assertEqual(
            local.reason, "missing_observables:request_local_count"
        )
        self.assertTrue(provider.available)
        self.assertTrue(call_total.available)
        self.assertTrue(unsupported.gated)
        self.assertEqual(unsupported.reason, "unsupported_position_target")

    def test_provider_accounted_target_algebra_and_proxy_exclusion(self) -> None:
        dataset = build_capability_supervised_dataset(
            (make_two_call_trajectory(0),),
            _descriptor(),
        )
        self.assertEqual(dataset.schema_version, CAPABILITY_DATASET_SCHEMA_VERSION)
        self.assertEqual(len(dataset.rows), 9)
        targets = {row.point.target for row in dataset.rows}
        self.assertNotIn(PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS, targets)
        self.assertNotIn(PredictionTarget.CALL_UNKNOWN_BILLABLE_TOKENS, targets)

        task_rows = dataset.select(
            PredictionPosition.TASK_PRE,
            PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
        ).rows + dataset.select(
            PredictionPosition.TASK_UPDATE,
            PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
        ).rows
        call_rows = dataset.select(
            PredictionPosition.CALL_PRE,
            PredictionTarget.CALL_BILLABLE_TOTAL_TOKENS,
        ).rows
        self.assertEqual([row.label for row in task_rows], [260, 140])
        self.assertEqual([row.label for row in call_rows], [120, 140])
        self.assertTrue(
            all(row.point.known_offset_tokens == 0 for row in dataset.rows)
        )
        self.assertTrue(
            all(
                not (set(row.point.features) & V2_EXCLUDED_LOCAL_FEATURES)
                for row in dataset.rows
            )
        )

    def test_real_local_count_preserves_length_features_and_legacy_offsets(self) -> None:
        dataset = build_capability_supervised_dataset(
            (make_two_call_trajectory(0),),
            _descriptor(local_count=True),
        )
        task_local = dataset.select(
            PredictionPosition.TASK_PRE,
            PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
        ).rows + dataset.select(
            PredictionPosition.TASK_UPDATE,
            PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
        ).rows
        call_local = dataset.select(
            PredictionPosition.CALL_PRE,
            PredictionTarget.CALL_UNKNOWN_BILLABLE_TOKENS,
        ).rows
        provider = dataset.select(
            PredictionPosition.TASK_PRE,
            PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
        ).rows

        self.assertEqual([row.label for row in task_local], [162, 12])
        self.assertEqual(
            [row.point.known_offset_tokens for row in task_local], [98, 128]
        )
        self.assertEqual(
            [row.point.known_offset_tokens for row in call_local], [98, 128]
        )
        self.assertEqual(provider[0].point.known_offset_tokens, 0)
        output_only = [
            row
            for row in dataset.rows
            if row.point.target
            in {
                PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS,
                PredictionTarget.CALL_FINAL_RESPONSE_OUTPUT_TOKENS,
            }
        ]
        self.assertTrue(output_only)
        self.assertTrue(
            all(row.point.known_offset_tokens == 0 for row in output_only)
        )
        self.assertEqual(
            provider[0].point.features["current_request_tokens_local"], 98
        )

    def test_proxy_only_request_count_cannot_change_v2_rows_or_dataset_id(self) -> None:
        trajectory = make_two_call_trajectory(0)
        changed = _change_request_local_counts(trajectory)
        descriptor = _descriptor()
        first = build_capability_supervised_dataset((trajectory,), descriptor)
        second = build_capability_supervised_dataset((changed,), descriptor)
        self.assertEqual(first.dataset_id, second.dataset_id)
        self.assertEqual(first.rows, second.rows)

    def test_dataset_id_binds_the_full_capability_contract(self) -> None:
        trajectory = make_two_call_trajectory(0)
        first = build_capability_supervised_dataset((trajectory,), _descriptor())
        second = build_capability_supervised_dataset(
            (trajectory,),
            _descriptor(extra=frozenset({Observable.TOOL_EVENTS})),
        )
        self.assertNotEqual(first.capability_contract_hash, second.capability_contract_hash)
        self.assertNotEqual(first.dataset_id, second.dataset_id)

    def test_suffix_mutation_changes_suffix_label_but_not_prefix_features(self) -> None:
        trajectory = make_two_call_trajectory(0)
        changed = _change_second_call_usage(trajectory)
        descriptor = _descriptor()
        first = build_capability_supervised_dataset((trajectory,), descriptor)
        second = build_capability_supervised_dataset((changed,), descriptor)

        def first_task_row(dataset: object):
            return next(
                row
                for row in dataset.rows  # type: ignore[attr-defined]
                if row.point.position == PredictionPosition.TASK_PRE
                and row.point.target
                == PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS
            )

        first_row = first_task_row(first)
        changed_row = first_task_row(second)
        self.assertEqual(dict(first_row.point.features), dict(changed_row.point.features))
        self.assertEqual(first_row.label, 260)
        self.assertEqual(changed_row.label, 317)

        first_call = first.select(
            PredictionPosition.CALL_PRE,
            PredictionTarget.CALL_BILLABLE_TOTAL_TOKENS,
        ).rows[0]
        changed_call = second.select(
            PredictionPosition.CALL_PRE,
            PredictionTarget.CALL_BILLABLE_TOTAL_TOKENS,
        ).rows[0]
        self.assertEqual(first_call.label, changed_call.label)
        self.assertEqual(first_call.label, 120)


if __name__ == "__main__":
    unittest.main()
