from __future__ import annotations

import unittest

from token_prediction.contracts import EventType
from token_prediction.features import replay_feature_snapshots

from tests.helpers import event


def _task(prefix: str = "g1"):
    return event(prefix, 0, EventType.TASK_STARTED, payload={"task_id": "task"})


class LogicalCallFeatureTests(unittest.TestCase):
    def test_retry_outputs_are_summed_only_when_next_request_closes_call(self) -> None:
        events = [
            _task("retry"),
            event("retry", 1, EventType.REQUEST_BUILT, call_id="call0"),
            event(
                "retry",
                2,
                EventType.API_ATTEMPT_STARTED,
                call_id="call0",
                attempt_id="a0",
            ),
            event(
                "retry",
                3,
                EventType.API_FAILED,
                call_id="call0",
                attempt_id="a0",
                payload={"usage": {"input_tokens": 11, "output_tokens": 2}},
            ),
            event(
                "retry",
                4,
                EventType.API_ATTEMPT_STARTED,
                call_id="call0",
                attempt_id="a1",
            ),
            event(
                "retry",
                5,
                EventType.GENERATION_CHECKPOINT,
                call_id="call0",
                attempt_id="a1",
                payload={"generated_tokens_so_far": 1},
            ),
            event(
                "retry",
                6,
                EventType.API_COMPLETED,
                call_id="call0",
                attempt_id="a1",
                payload={"usage": {"input_tokens": 12, "output_tokens": 5}},
            ),
            event("retry", 7, EventType.REQUEST_BUILT, call_id="call1"),
        ]

        initial, checkpoint, next_request = replay_feature_snapshots(events)

        self.assertIsNone(initial.values["last_call_output_tokens"])
        self.assertIsNone(checkpoint.values["last_call_output_tokens"])
        self.assertEqual(checkpoint.values["cumulative_provider_output_tokens"], 2)
        self.assertEqual(checkpoint.values["consecutive_error_rounds"], 0)

        self.assertEqual(next_request.values["last_call_output_tokens"], 7)
        self.assertEqual(next_request.values["recent_generated_mean_3"], 7.0)
        self.assertEqual(next_request.values["cumulative_provider_input_tokens"], 23)
        self.assertEqual(next_request.values["cumulative_provider_output_tokens"], 7)
        self.assertEqual(next_request.values["completed_call_count"], 1)
        self.assertEqual(next_request.values["consecutive_error_rounds"], 1)

    def test_any_unknown_attempt_poisoning_ages_out_of_three_call_window(self) -> None:
        events = [
            _task("unknown"),
            event("unknown", 1, EventType.REQUEST_BUILT, call_id="call0"),
            event(
                "unknown",
                2,
                EventType.API_ATTEMPT_STARTED,
                call_id="call0",
                attempt_id="a0-missing",
            ),
            event(
                "unknown",
                3,
                EventType.API_FAILED,
                call_id="call0",
                attempt_id="a0-missing",
                payload={"usage": {"input_tokens": 4}},
            ),
            event(
                "unknown",
                4,
                EventType.API_ATTEMPT_STARTED,
                call_id="call0",
                attempt_id="a0-known",
            ),
            event(
                "unknown",
                5,
                EventType.API_COMPLETED,
                call_id="call0",
                attempt_id="a0-known",
                payload={"usage": {"input_tokens": 5, "output_tokens": 5}},
            ),
            event("unknown", 6, EventType.REQUEST_BUILT, call_id="call1"),
            event(
                "unknown",
                7,
                EventType.API_ATTEMPT_STARTED,
                call_id="call1",
                attempt_id="a1",
            ),
            event(
                "unknown",
                8,
                EventType.API_COMPLETED,
                call_id="call1",
                attempt_id="a1",
                payload={"usage": {"input_tokens": 10, "output_tokens": 10}},
            ),
            event("unknown", 9, EventType.REQUEST_BUILT, call_id="call2"),
            event(
                "unknown",
                10,
                EventType.API_ATTEMPT_STARTED,
                call_id="call2",
                attempt_id="a2",
            ),
            event(
                "unknown",
                11,
                EventType.API_COMPLETED,
                call_id="call2",
                attempt_id="a2",
                payload={"usage": {"input_tokens": 20, "output_tokens": 20}},
            ),
            event("unknown", 12, EventType.REQUEST_BUILT, call_id="call3"),
            event(
                "unknown",
                13,
                EventType.API_ATTEMPT_STARTED,
                call_id="call3",
                attempt_id="a3",
            ),
            event(
                "unknown",
                14,
                EventType.API_COMPLETED,
                call_id="call3",
                attempt_id="a3",
                payload={"usage": {"input_tokens": 30, "output_tokens": 30}},
            ),
            event("unknown", 15, EventType.REQUEST_BUILT, call_id="call4"),
        ]

        requests = replay_feature_snapshots(events)
        after_unknown, after_10, after_20, after_30 = requests[1:]

        self.assertIsNone(after_unknown.values["last_call_output_tokens"])
        self.assertIsNone(after_unknown.values["recent_generated_mean_3"])
        self.assertEqual(after_unknown.values["cumulative_provider_output_tokens"], 5)
        self.assertEqual(after_unknown.values["missing_usage_attempts"], 1)

        self.assertEqual(after_10.values["last_call_output_tokens"], 10)
        self.assertIsNone(after_10.values["recent_generated_mean_3"])
        self.assertEqual(after_20.values["last_call_output_tokens"], 20)
        self.assertIsNone(after_20.values["recent_generated_mean_3"])
        self.assertEqual(after_30.values["last_call_output_tokens"], 30)
        self.assertEqual(after_30.values["recent_generated_mean_3"], 20.0)

    def test_first_and_attemptless_calls_have_unknown_history(self) -> None:
        events = [
            _task("empty"),
            event("empty", 1, EventType.REQUEST_BUILT, call_id="call0"),
            event("empty", 2, EventType.REQUEST_BUILT, call_id="call1"),
        ]

        first, second = replay_feature_snapshots(events)
        self.assertIsNone(first.values["last_call_output_tokens"])
        self.assertIsNone(first.values["recent_generated_mean_3"])
        self.assertIsNone(first.values["last_tool_type"])
        self.assertIsNone(first.values["last_round_tool_error_count"])
        self.assertEqual(first.values["consecutive_error_rounds"], 0)
        self.assertEqual(first.values["repeated_action_count_3"], 0)

        self.assertIsNone(second.values["last_call_output_tokens"])
        self.assertIsNone(second.values["recent_generated_mean_3"])
        self.assertEqual(second.values["completed_call_count"], 1)

    def test_tool_error_rounds_action_priority_and_cross_call_reset(self) -> None:
        events = [
            _task("tools"),
            event("tools", 1, EventType.REQUEST_BUILT, call_id="call0"),
            event(
                "tools",
                2,
                EventType.API_ATTEMPT_STARTED,
                call_id="call0",
                attempt_id="a0",
            ),
            event(
                "tools",
                3,
                EventType.API_FAILED,
                call_id="call0",
                attempt_id="a0",
                payload={"usage": {"input_tokens": 1, "output_tokens": 1}},
            ),
            event(
                "tools",
                4,
                EventType.API_ATTEMPT_STARTED,
                call_id="call0",
                attempt_id="a0-retry",
            ),
            event(
                "tools",
                5,
                EventType.API_COMPLETED,
                call_id="call0",
                attempt_id="a0-retry",
                payload={"usage": {"input_tokens": 1, "output_tokens": 1}},
            ),
            event(
                "tools",
                6,
                EventType.TOOL_STARTED,
                call_id="call0",
                payload={
                    "tool_name": "shell",
                    "action_hash": "same-hash",
                    "action_name": "ignored-start",
                },
            ),
            event(
                "tools",
                7,
                EventType.TOOL_COMPLETED,
                call_id="call0",
                payload={
                    "tool_name": "shell",
                    "action_hash": "same-hash",
                    "action_name": "first-name",
                },
            ),
            event(
                "tools",
                8,
                EventType.TOOL_COMPLETED,
                call_id="call0",
                payload={
                    "tool_name": "reader",
                    "action_hash": "same-hash",
                    "action_name": "second-name",
                },
            ),
            event("tools", 9, EventType.REQUEST_BUILT, call_id="call1"),
            event(
                "tools",
                10,
                EventType.API_ATTEMPT_STARTED,
                call_id="call1",
                attempt_id="a1",
            ),
            event(
                "tools",
                11,
                EventType.API_COMPLETED,
                call_id="call1",
                attempt_id="a1",
                payload={"usage": {"input_tokens": 1, "output_tokens": 1}},
            ),
            event(
                "tools",
                12,
                EventType.TOOL_FAILED,
                call_id="call1",
                payload={
                    "tool_name": "writer",
                    "action_name": "same-action",
                    "action": "different-action-a",
                },
            ),
            event(
                "tools",
                13,
                EventType.TOOL_COMPLETED,
                call_id="call1",
                payload={
                    "tool_name": "formatter",
                    "action_name": "same-action",
                    "action": "different-action-b",
                },
            ),
            event("tools", 14, EventType.REQUEST_BUILT, call_id="call2"),
            event(
                "tools",
                15,
                EventType.API_ATTEMPT_STARTED,
                call_id="call2",
                attempt_id="a2",
            ),
            event(
                "tools",
                16,
                EventType.API_COMPLETED,
                call_id="call2",
                attempt_id="a2",
                payload={"usage": {"input_tokens": 1, "output_tokens": 1}},
            ),
            event("tools", 17, EventType.REQUEST_BUILT, call_id="call3"),
        ]

        initial, after_api_error, after_tool_error, after_clean = (
            replay_feature_snapshots(events)
        )

        self.assertIsNone(initial.values["last_tool_type"])
        self.assertEqual(after_api_error.values["last_tool_type"], "reader")
        self.assertEqual(after_api_error.values["last_round_tool_error_count"], 0)
        self.assertEqual(after_api_error.values["consecutive_error_rounds"], 1)
        # The start event is not double-counted, while action_hash wins over
        # the two deliberately different action_name values.
        self.assertEqual(after_api_error.values["repeated_action_count_3"], 1)

        self.assertEqual(after_tool_error.values["last_tool_type"], "formatter")
        self.assertEqual(after_tool_error.values["last_round_tool_error_count"], 1)
        self.assertEqual(after_tool_error.values["consecutive_error_rounds"], 2)
        # action_name wins over the deliberately different action values.
        self.assertEqual(after_tool_error.values["repeated_action_count_3"], 1)

        # `last_tool_type` means the most recent visible tool invocation, not
        # merely the last round's tool; a tool-free round must not erase it.
        self.assertEqual(after_clean.values["last_tool_type"], "formatter")
        self.assertEqual(after_clean.values["last_round_tool_error_count"], 0)
        self.assertEqual(after_clean.values["consecutive_error_rounds"], 0)
        self.assertEqual(after_clean.values["repeated_action_count_3"], 1)


if __name__ == "__main__":
    unittest.main()
