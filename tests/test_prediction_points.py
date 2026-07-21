from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

from token_prediction.contracts import Observable, SourceCapabilities, SourceDescriptor
from token_prediction.dataset import (
    PredictionPosition,
    PredictionTarget,
    build_capability_supervised_dataset,
    build_prediction_points,
    prediction_input_contract_hash,
)

from tests.helpers import make_two_call_trajectory
from tests.test_capability_dataset import _change_second_call_usage


def _descriptor(*, local_count: bool = False) -> SourceDescriptor:
    observables = {
        Observable.ATTEMPT_USAGE,
        Observable.REQUEST_BOUNDARIES,
        Observable.TASK_TERMINATION,
        Observable.TASK_USAGE,
    }
    if local_count:
        observables.add(Observable.REQUEST_LOCAL_COUNT)
    return SourceDescriptor(
        source_id="point-fixture",
        revision="revision-1",
        manifest_path="workspace/manifests/point-fixture.json",
        manifest_sha256="b" * 64,
        capabilities=SourceCapabilities(
            source_id="point-fixture",
            observables=frozenset(observables),
        ),
    )


class PredictionPointTests(unittest.TestCase):
    def test_prefix_builder_never_calls_label_builders(self) -> None:
        trajectory = make_two_call_trajectory(0)
        descriptor = _descriptor()
        with (
            patch(
                "token_prediction.dataset.labels.build_prediction_labels",
                side_effect=AssertionError("labels are forbidden"),
            ),
            patch(
                "token_prediction.dataset.labels.build_task_aggregate_label",
                side_effect=AssertionError("labels are forbidden"),
            ),
            patch(
                "token_prediction.dataset.labels.build_generation_labels",
                side_effect=AssertionError("labels are forbidden"),
            ),
        ):
            result = build_prediction_points((trajectory,), descriptor)
        self.assertEqual(len(result.points), 9)
        self.assertEqual(
            result.input_contract_hash,
            prediction_input_contract_hash(descriptor),
        )

    def test_schema_v2_point_rows_and_dataset_ids_keep_regression_parity(self) -> None:
        trajectory = make_two_call_trajectory(0)
        expected = {
            False: "f1a59dfa102bf9422ea8646caaa581e11de9df80353a3f352e2938735769f6b5",
            True: "7f39d907ceb045ea965d1eaac1d8c5352a36d72c12b9c810e6f2f722d509b9ca",
        }
        # The descriptor identity differs from the older capability-dataset
        # fixture, so these constants freeze this point-specific fixture.
        for local_count in (False, True):
            descriptor = _descriptor(local_count=local_count)
            point_set = build_prediction_points((trajectory,), descriptor)
            dataset = build_capability_supervised_dataset((trajectory,), descriptor)
            self.assertEqual(
                point_set.points,
                tuple(row.point for row in dataset.rows),
            )
            self.assertEqual(dataset.dataset_id, expected[local_count])

    def test_suffix_observation_cannot_change_prefix_point_or_input_contract(self) -> None:
        original = make_two_call_trajectory(0)
        changed = _change_second_call_usage(original)
        descriptor = _descriptor()
        first = build_prediction_points((original,), descriptor)
        second = build_prediction_points((changed,), descriptor)

        def task_pre(result: object):
            return next(
                point
                for point in result.points  # type: ignore[attr-defined]
                if point.position == PredictionPosition.TASK_PRE
                and point.target
                == PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS
            )

        self.assertEqual(task_pre(first), task_pre(second))
        self.assertEqual(first.input_contract_hash, second.input_contract_hash)
        self.assertEqual(first.context_hash, second.context_hash)

    def test_input_contract_excludes_manifest_revision_provenance(self) -> None:
        first = _descriptor()
        changed = replace(
            first,
            revision="revision-2",
            manifest_sha256="f" * 64,
        )
        self.assertNotEqual(first.descriptor_hash, changed.descriptor_hash)
        self.assertEqual(
            prediction_input_contract_hash(first),
            prediction_input_contract_hash(changed),
        )

    def test_proxy_local_counts_are_removed_but_missing_features_remain_explicit(self) -> None:
        point_set = build_prediction_points(
            (make_two_call_trajectory(0),),
            _descriptor(),
        )
        task_pre = next(
            point
            for point in point_set.points
            if point.position == PredictionPosition.TASK_PRE
            and point.target
            == PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS
        )
        self.assertNotIn("current_request_tokens_local", task_pre.features)
        self.assertNotIn("request_delta_tokens", task_pre.features)
        self.assertIn("last_call_output_tokens", task_pre.features)
        self.assertIsNone(task_pre.features["last_call_output_tokens"])


if __name__ == "__main__":
    unittest.main()
