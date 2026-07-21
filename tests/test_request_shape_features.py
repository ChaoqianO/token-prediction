from __future__ import annotations

import unittest

from token_prediction.contracts import EventType, Observable, SourceCapabilities, SourceDescriptor
from token_prediction.dataset import (
    PredictionPosition,
    PredictionTarget,
    augment_request_shape_features,
    build_capability_supervised_dataset,
    build_lifecycle_slice,
)
from token_prediction.trajectory import Trajectory

from tests.helpers import make_two_call_trajectory


def _descriptor() -> SourceDescriptor:
    return SourceDescriptor(
        source_id="request-shape-fixture",
        revision="revision-1",
        manifest_path="workspace/manifests/request-shape.json",
        manifest_sha256="a" * 64,
        capabilities=SourceCapabilities(
            source_id="request-shape-fixture",
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


def _with_request_shape(trajectory: Trajectory) -> Trajectory:
    events = []
    request_index = 0
    for event in trajectory.events:
        if event.event_type == EventType.REQUEST_BUILT:
            payload = event.payload
            payload.update(
                {
                    "request_message_count": 2 + request_index,
                    "request_content_chars": 1000 + 250 * request_index,
                }
            )
            event = event.with_payload(payload)
            request_index += 1
        events.append(event)
    return Trajectory.from_events(events)


class RequestShapeFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trajectories = tuple(
            _with_request_shape(make_two_call_trajectory(task, run))
            for task in range(2)
            for run in range(2)
        )
        self.dataset = build_capability_supervised_dataset(
            self.trajectories,
            _descriptor(),
        )

    def test_projection_is_derived_and_visible_at_request_boundary(self) -> None:
        derived = augment_request_shape_features(self.dataset, self.trajectories)
        self.assertNotEqual(derived.dataset_id, self.dataset.dataset_id)
        self.assertNotEqual(derived.input_contract_hash, self.dataset.input_contract_hash)
        request_rows = [
            row
            for row in derived.rows
            if row.point.position
            in {
                PredictionPosition.TASK_PRE,
                PredictionPosition.TASK_UPDATE,
                PredictionPosition.CALL_PRE,
            }
        ]
        self.assertTrue(request_rows)
        self.assertTrue(
            all(
                isinstance(row.point.features["request_message_count"], int)
                and isinstance(row.point.features["request_content_chars"], int)
                for row in request_rows
            )
        )
        self.assertTrue(
            all(
                "request_content_chars" not in row.point.features
                for row in self.dataset.rows
            )
        )
        lifecycle = build_lifecycle_slice(
            derived,
            target=PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
        )
        self.assertEqual(lifecycle.input_contract_hash, derived.input_contract_hash)

    def test_future_usage_change_cannot_change_request_shape_inputs(self) -> None:
        baseline = augment_request_shape_features(self.dataset, self.trajectories)
        changed_events = []
        changed_once = False
        for event in self.trajectories[0].events:
            if event.event_type == EventType.API_COMPLETED and not changed_once:
                payload = event.payload
                usage = dict(payload["usage"])
                usage["output_tokens"] = int(usage["output_tokens"]) + 999
                usage["total_tokens"] = int(usage["input_tokens"]) + int(
                    usage["output_tokens"]
                )
                payload["usage"] = usage
                event = event.with_payload(payload)
                changed_once = True
            changed_events.append(event)
        changed_trajectories = (
            Trajectory.from_events(changed_events),
            *self.trajectories[1:],
        )
        changed_dataset = build_capability_supervised_dataset(
            changed_trajectories,
            _descriptor(),
        )
        changed = augment_request_shape_features(changed_dataset, changed_trajectories)
        baseline_features = {
            row.point.point_id: (
                row.point.features.get("request_message_count"),
                row.point.features.get("request_content_chars"),
            )
            for row in baseline.rows
        }
        changed_features = {
            row.point.point_id: (
                row.point.features.get("request_message_count"),
                row.point.features.get("request_content_chars"),
            )
            for row in changed.rows
        }
        self.assertEqual(baseline_features, changed_features)
        self.assertEqual(baseline.input_contract_hash, changed.input_contract_hash)
        self.assertNotEqual(baseline.dataset_id, changed.dataset_id)

    def test_invalid_or_incomplete_projection_fails_closed(self) -> None:
        first = self.trajectories[0]
        events = []
        changed = False
        for event in first.events:
            if event.event_type == EventType.REQUEST_BUILT and not changed:
                payload = event.payload
                payload["request_content_chars"] = -1
                event = event.with_payload(payload)
                changed = True
            events.append(event)
        invalid_trajectories = (Trajectory.from_events(events), *self.trajectories[1:])
        invalid_dataset = build_capability_supervised_dataset(
            invalid_trajectories,
            _descriptor(),
        )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            augment_request_shape_features(invalid_dataset, invalid_trajectories)
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            augment_request_shape_features(self.dataset, self.trajectories[:-1])


if __name__ == "__main__":
    unittest.main()
