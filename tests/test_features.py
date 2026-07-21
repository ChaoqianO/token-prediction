from __future__ import annotations

import unittest
from pathlib import Path

from token_prediction.contracts import CanonicalEvent, EventType
from token_prediction.features import replay_feature_snapshots
from token_prediction.pipeline import load_jsonl_events

from tests.helpers import event


FIXTURE = Path(__file__).parent / "fixtures" / "two_call_events.jsonl"


class FeatureCausalityTests(unittest.TestCase):
    def test_response_usage_is_not_visible_at_request_boundary(self) -> None:
        snapshots = replay_feature_snapshots(load_jsonl_events(FIXTURE))
        self.assertEqual(len(snapshots), 2)
        self.assertEqual(snapshots[0].values["completed_api_attempts"], 0)
        self.assertEqual(snapshots[0].values["cumulative_provider_input_tokens"], 0)
        self.assertIsNone(snapshots[0].values["request_delta_tokens"])
        self.assertEqual(snapshots[1].values["completed_api_attempts"], 1)
        self.assertEqual(snapshots[1].values["cumulative_provider_input_tokens"], 100)
        self.assertEqual(snapshots[0].point_event_id, "e1")
        self.assertEqual(snapshots[1].logical_call_id, "call-1")

    def test_identical_prefix_different_suffix_has_same_feature_hash(self) -> None:
        prefix = load_jsonl_events(FIXTURE)[:2]
        future_a = CanonicalEvent.create(
            trajectory_id="traj-1",
            event_seq=2,
            event_type=EventType.TASK_FINISHED,
            payload={"outcome": "success"},
        )
        future_b = CanonicalEvent.create(
            trajectory_id="traj-1",
            event_seq=2,
            event_type=EventType.TASK_ABORTED,
            payload={"outcome": "failure", "future_tokens": 999999},
        )
        snapshot_a = replay_feature_snapshots([*prefix, future_a])[0]
        snapshot_b = replay_feature_snapshots([*prefix, future_b])[0]
        self.assertEqual(snapshot_a.values, snapshot_b.values)
        self.assertEqual(snapshot_a.feature_hash, snapshot_b.feature_hash)

    def test_missing_request_count_is_not_silently_zero(self) -> None:
        events = [
            event("missing", 0, EventType.TASK_STARTED, payload={"task_id": "task"}),
            event("missing", 1, EventType.REQUEST_BUILT, call_id="call", payload={}),
        ]
        snapshot = replay_feature_snapshots(events)[0]
        self.assertIsNone(snapshot.values["current_request_tokens_local"])
        self.assertIsNone(snapshot.values["context_utilization"])

    def test_failed_attempt_usage_is_visible_at_next_boundary(self) -> None:
        events = [
            event("failed", 0, EventType.TASK_STARTED, payload={"task_id": "task"}),
            event("failed", 1, EventType.REQUEST_BUILT, call_id="call0"),
            event(
                "failed",
                2,
                EventType.API_ATTEMPT_STARTED,
                call_id="call0",
                attempt_id="a0",
            ),
            event(
                "failed",
                3,
                EventType.API_FAILED,
                call_id="call0",
                attempt_id="a0",
                payload={"usage": {"input_tokens": 11, "output_tokens": 2}},
            ),
            event("failed", 4, EventType.REQUEST_BUILT, call_id="call1"),
        ]
        snapshot = replay_feature_snapshots(events)[1]
        self.assertEqual(snapshot.values["cumulative_provider_input_tokens"], 11)
        self.assertEqual(snapshot.values["cumulative_provider_output_tokens"], 2)
        self.assertEqual(snapshot.values["failed_api_attempts"], 1)

    def test_feature_reducer_rejects_coerced_token_counts(self) -> None:
        for value in (1.5, "2", True):
            with self.subTest(value=value):
                events = [
                    event("strict", 0, EventType.TASK_STARTED, payload={"task_id": "task"}),
                    event(
                        "strict",
                        1,
                        EventType.REQUEST_BUILT,
                        call_id="call",
                        payload={"request_tokens_local": value},
                    ),
                ]
                with self.assertRaisesRegex(ValueError, "non-negative integers"):
                    replay_feature_snapshots(events)

    def test_feature_reducer_rejects_non_object_usage(self) -> None:
        events = [
            event("strict-usage", 0, EventType.TASK_STARTED),
            event(
                "strict-usage",
                1,
                EventType.REQUEST_BUILT,
                call_id="call",
            ),
            event(
                "strict-usage",
                2,
                EventType.API_COMPLETED,
                call_id="call",
                attempt_id="attempt",
                payload={"usage": []},
            ),
        ]
        with self.assertRaisesRegex(ValueError, "token usage must be an object"):
            replay_feature_snapshots(events)


if __name__ == "__main__":
    unittest.main()
