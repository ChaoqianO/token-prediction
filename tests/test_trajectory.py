from __future__ import annotations

import unittest

from token_prediction.contracts import EventType
from token_prediction.trajectory import Trajectory, TrajectoryValidationError

from tests.helpers import event, make_two_call_trajectory


class TrajectoryContractTests(unittest.TestCase):
    def test_valid_trajectory_exposes_task_and_run_identity(self) -> None:
        trajectory = make_two_call_trajectory(0, 1)
        self.assertEqual(trajectory.task_id, "task-0")
        self.assertEqual(trajectory.run_id, "task0-run1")

    def test_terminal_attempt_requires_start(self) -> None:
        prefix = "bad"
        events = (
            event(prefix, 0, EventType.TASK_STARTED, payload={"task_id": "task"}),
            event(
                prefix,
                1,
                EventType.REQUEST_BUILT,
                call_id="call",
                payload={"request_tokens_local": 3},
            ),
            event(
                prefix,
                2,
                EventType.API_COMPLETED,
                call_id="call",
                attempt_id="attempt",
                payload={"usage": {"input_tokens": 3, "output_tokens": 1}},
            ),
            event(prefix, 3, EventType.TASK_FINISHED),
        )
        with self.assertRaisesRegex(TrajectoryValidationError, "unstarted"):
            Trajectory.from_events(events)

    def test_finished_trajectory_rejects_dangling_attempt(self) -> None:
        prefix = "dangling"
        events = (
            event(prefix, 0, EventType.TASK_STARTED, payload={"task_id": "task"}),
            event(prefix, 1, EventType.REQUEST_BUILT, call_id="call"),
            event(
                prefix,
                2,
                EventType.API_ATTEMPT_STARTED,
                call_id="call",
                attempt_id="attempt",
            ),
            event(prefix, 3, EventType.TASK_FINISHED),
        )
        with self.assertRaisesRegex(TrajectoryValidationError, "unterminated"):
            Trajectory.from_events(events)

    def test_aborted_trajectory_may_preserve_dangling_attempt(self) -> None:
        prefix = "timeout"
        events = (
            event(prefix, 0, EventType.TASK_STARTED, payload={"task_id": "task"}),
            event(prefix, 1, EventType.REQUEST_BUILT, call_id="call"),
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
                EventType.TASK_ABORTED,
                payload={"reason": "timeout"},
            ),
        )
        self.assertEqual(Trajectory.from_events(events).task_id, "task")

    def test_logical_call_events_cannot_arrive_after_the_next_request(self) -> None:
        prefix = "interleaved"
        events = (
            event(prefix, 0, EventType.TASK_STARTED, payload={"task_id": "task"}),
            event(prefix, 1, EventType.REQUEST_BUILT, call_id="call-0"),
            event(
                prefix,
                2,
                EventType.API_ATTEMPT_STARTED,
                call_id="call-0",
                attempt_id="attempt-0",
            ),
            event(
                prefix,
                3,
                EventType.API_COMPLETED,
                call_id="call-0",
                attempt_id="attempt-0",
                payload={"usage": {"input_tokens": 3, "output_tokens": 1}},
            ),
            event(prefix, 4, EventType.REQUEST_BUILT, call_id="call-1"),
            event(
                prefix,
                5,
                EventType.TOOL_COMPLETED,
                call_id="call-0",
                payload={"tool_name": "late-tool"},
            ),
            event(prefix, 6, EventType.TASK_FINISHED),
        )
        with self.assertRaisesRegex(TrajectoryValidationError, "interleave"):
            Trajectory.from_events(events)

    def test_new_request_rejects_an_active_dangling_attempt(self) -> None:
        prefix = "overlap"
        events = (
            event(prefix, 0, EventType.TASK_STARTED, payload={"task_id": "task"}),
            event(prefix, 1, EventType.REQUEST_BUILT, call_id="call-0"),
            event(
                prefix,
                2,
                EventType.API_ATTEMPT_STARTED,
                call_id="call-0",
                attempt_id="attempt-0",
            ),
            event(prefix, 3, EventType.REQUEST_BUILT, call_id="call-1"),
            event(prefix, 4, EventType.TASK_ABORTED),
        )
        with self.assertRaisesRegex(TrajectoryValidationError, "cannot start"):
            Trajectory.from_events(events)


if __name__ == "__main__":
    unittest.main()
