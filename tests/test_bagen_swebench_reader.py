from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from token_prediction.collection import (
    BagenSwebenchMetadata,
    BagenSwebenchReader,
    BagenSwebenchSchemaError,
)
from token_prediction.contracts import EventType, Observable
from token_prediction.dataset import (
    LabelStatus,
    PredictionPosition,
    PredictionTarget,
    build_supervised_dataset,
)
from token_prediction.features import replay_feature_snapshots


INSTANCE_ID = "django__django-12345"
SUBMISSION = "diff --git a/example.py b/example.py\n+fixed = True\n"


def _usage(
    input_tokens: int,
    output_tokens: int,
    *,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> dict[str, object]:
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "prompt_tokens_details": {"cached_tokens": cached_tokens},
        "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
    }


def _tool_call(tool_call_id: str, command: str) -> dict[str, object]:
    return {
        "function": {
            "arguments": json.dumps({"command": command}, separators=(",", ":")),
            "name": "bash",
        },
        "id": tool_call_id,
        "type": "function",
    }


def _assistant(
    request_id: str,
    input_tokens: int,
    output_tokens: int,
    *,
    content: str | None = None,
    tool_calls: list[dict[str, object]] | None = None,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
    timestamp: float = 1_700_000_000.0,
) -> dict[str, object]:
    calls = list(tool_calls or [])
    message: dict[str, object] = {
        "content": content,
        "role": "assistant",
        "tool_calls": calls,
        "function_call": None,
        "provider_specific_fields": {"refusal": None},
        "annotations": [],
    }
    provider_message = {
        key: copy.deepcopy(message[key])
        for key in ("role", "content", "tool_calls", "function_call")
    }
    message["extra"] = {
        "actions": [],
        "response": {
            "id": request_id,
            "created": int(timestamp),
            "model": "gpt-5.2-2025-12-11",
            "object": "chat.completion",
            "choices": [
                {
                    "finish_reason": "tool_calls" if calls else "stop",
                    "index": 0,
                    "message": provider_message,
                }
            ],
            "usage": _usage(
                input_tokens,
                output_tokens,
                cached_tokens=cached_tokens,
                reasoning_tokens=reasoning_tokens,
            ),
        },
        "cost": 0.01,
        "timestamp": timestamp,
    }
    return message


def _format_error(
    *,
    request_id: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> dict[str, object]:
    extra: dict[str, object] = {"interrupt_type": "FormatError"}
    if request_id is not None:
        extra["response"] = {
            "id": request_id,
            "created": 1_700_000_005,
            "model": "gpt-5.2-2025-12-11",
            "usage": _usage(
                input_tokens,
                output_tokens,
                cached_tokens=cached_tokens,
                reasoning_tokens=reasoning_tokens,
            ),
        }
        extra["timestamp"] = 1_700_000_005.5
    return {
        "role": "user",
        "content": "Tool call error: malformed arguments",
        "extra": extra,
    }


def _tool_result(
    tool_call_id: str,
    *,
    returncode: int,
    content: str = "<output>done</output>",
) -> dict[str, object]:
    return {
        "content": content,
        "extra": {
            "raw_output": content,
            "returncode": returncode,
            "timestamp": 1_700_000_001.0,
            "exception_info": "",
        },
        "tool_call_id": tool_call_id,
        "role": "tool",
    }


def _exit(submission: str = SUBMISSION) -> dict[str, object]:
    return {
        "role": "exit",
        "content": submission,
        "extra": {"exit_status": "Submitted", "submission": submission},
    }


def _payload(
    messages: list[dict[str, object]],
    *,
    instance_id: str = INSTANCE_ID,
    submission: str = SUBMISSION,
    include_instance_id: bool = True,
    include_trajectory_format: bool = True,
) -> dict[str, object]:
    api_calls = sum(
        message.get("role") == "assistant"
        or (
            message.get("role") == "user"
            and isinstance(message.get("extra"), dict)
            and message["extra"].get("interrupt_type") == "FormatError"  # type: ignore[union-attr]
        )
        for message in messages
    )
    payload: dict[str, object] = {
        "info": {
            "model_stats": {"instance_cost": 0.03, "api_calls": api_calls},
            "config": {
                "agent": {
                    "system_template": "You are a coding agent.",
                    "instance_template": "Fix {{task}}.",
                    "step_limit": 20,
                    "cost_limit": 3.0,
                    "output_path": None,
                },
                "agent_type": "tests.ProgressTrackingAgent",
                "model": {
                    "model_name": "gpt-5.2",
                    "model_kwargs": {
                        "drop_params": True,
                        "temperature": 1,
                        "parallel_tool_calls": True,
                        "max_completion_tokens": 800,
                        "reasoning_effort": "none",
                    },
                    "litellm_model_registry": None,
                    "set_cache_control": None,
                    "cost_tracking": "default",
                    "format_error_template": "Tool call error: {{error}}",
                    "observation_template": "{{output}}",
                    "multimodal_regex": "",
                },
                "model_type": "tests.LitellmModel",
                "environment": {
                    "image": "swebench/test:latest",
                    "cwd": "/testbed",
                    "env": {"PAGER": "cat"},
                    "forward_env": [],
                    "timeout": 60,
                    "executable": "docker",
                    "run_args": ["--rm"],
                    "container_timeout": "2h",
                    "pull_timeout": 120,
                    "interpreter": ["bash", "-c"],
                },
                "environment_type": "tests.DockerEnvironment",
            },
            "mini_version": "2.2.8",
            "exit_status": "Submitted",
            "submission": submission,
        },
        "messages": messages,
    }
    if include_trajectory_format:
        payload["trajectory_format"] = "mini-swe-agent-1.1"
    if include_instance_id:
        payload["instance_id"] = instance_id
    return payload


def _base_messages(*assistant_messages: dict[str, object]) -> list[dict[str, object]]:
    return [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Fix the reported bug."},
        *assistant_messages,
        _exit(),
    ]


def _write_trajectory(
    root: Path,
    payload: dict[str, object],
    *,
    instance_id: str = INSTANCE_ID,
) -> Path:
    directory = root / instance_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{instance_id}.traj.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _metadata(
    *,
    model_family: str = "gpt52-instant",
    provider: str = "openai",
    run_identity: str = "run-a",
) -> BagenSwebenchMetadata:
    return BagenSwebenchMetadata(
        model_family=model_family,
        provider=provider,
        run_identity=run_identity,
    )


class BagenSwebenchReaderTests(unittest.TestCase):
    def test_identity_condition_overrides_and_determinism(self) -> None:
        payload = _payload(
            _base_messages(_assistant("req-1", 100, 10, content="Done"))
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_trajectory(Path(temporary), payload)
            reader = BagenSwebenchReader()
            first = reader.read(path, _metadata())
            repeated = reader.read(path, _metadata())
            other_run = reader.read(path, _metadata(run_identity="run-b"))
            other_family = reader.read(path, _metadata(model_family="family-b"))
            other_provider = reader.read(path, _metadata(provider="provider-b"))

            changed_config = copy.deepcopy(payload)
            config = changed_config["info"]["config"]  # type: ignore[index]
            config["model"]["model_kwargs"]["temperature"] = 0.25  # type: ignore[index]
            changed_path = _write_trajectory(
                Path(temporary) / "changed-config", changed_config
            )
            changed_condition = reader.read(changed_path, _metadata())

        self.assertEqual(first.task_id, INSTANCE_ID)
        self.assertEqual(first.run_id, "run-a")
        self.assertEqual(first.events, repeated.events)
        self.assertEqual(
            [event.content_hash for event in first.events],
            [event.content_hash for event in repeated.events],
        )
        self.assertEqual(
            [event.event_seq for event in first.events], list(range(len(first.events)))
        )
        self.assertEqual(
            len({event.event_id for event in first.events}), len(first.events)
        )

        started = first.events[0].payload
        self.assertEqual(started["model_family"], "gpt52-instant")
        self.assertEqual(started["provider"], "openai")
        self.assertEqual(started["run_id"], "run-a")

        self.assertEqual(first.task_id, other_run.task_id)
        self.assertEqual(first.condition_id, other_run.condition_id)
        self.assertNotEqual(first.trajectory_id, other_run.trajectory_id)
        self.assertEqual(first.task_id, other_family.task_id)
        self.assertNotEqual(first.condition_id, other_family.condition_id)
        self.assertNotEqual(first.trajectory_id, other_family.trajectory_id)
        self.assertNotEqual(first.condition_id, other_provider.condition_id)
        self.assertNotEqual(first.condition_id, changed_condition.condition_id)

    def test_usage_closes_and_cached_or_reasoning_tokens_are_not_added_twice(
        self,
    ) -> None:
        call = _tool_call("tool-1", "python -m pytest -q")
        messages = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Fix the reported bug."},
            _assistant(
                "req-1",
                100,
                10,
                tool_calls=[call],
                cached_tokens=40,
                reasoning_tokens=3,
            ),
            _tool_result("tool-1", returncode=0),
            _assistant(
                "req-2",
                120,
                20,
                content="Implemented the fix.",
                cached_tokens=50,
                reasoning_tokens=7,
            ),
            _exit(),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_trajectory(Path(temporary), _payload(messages))
            trajectory = BagenSwebenchReader().read(path, _metadata())

        self.assertEqual(
            [event.event_type for event in trajectory.events],
            [
                EventType.TASK_STARTED,
                EventType.REQUEST_BUILT,
                EventType.API_ATTEMPT_STARTED,
                EventType.API_COMPLETED,
                EventType.TOOL_COMPLETED,
                EventType.REQUEST_BUILT,
                EventType.API_ATTEMPT_STARTED,
                EventType.API_COMPLETED,
                EventType.TASK_FINISHED,
            ],
        )
        starts = {
            (event.logical_call_id, event.attempt_id)
            for event in trajectory.events
            if event.event_type == EventType.API_ATTEMPT_STARTED
        }
        terminals = {
            (event.logical_call_id, event.attempt_id)
            for event in trajectory.events
            if event.event_type in {EventType.API_COMPLETED, EventType.API_FAILED}
        }
        self.assertEqual(starts, terminals)

        api_usages = [
            event.payload["usage"]
            for event in trajectory.events
            if event.event_type == EventType.API_COMPLETED
        ]
        self.assertEqual(
            sum(int(usage["input_tokens"]) for usage in api_usages), 220
        )
        self.assertEqual(
            sum(int(usage["output_tokens"]) for usage in api_usages), 30
        )
        task_usage = trajectory.events[-1].payload["usage"]
        self.assertEqual(task_usage["input_tokens"], 220)
        self.assertEqual(task_usage["output_tokens"], 30)
        self.assertEqual(task_usage["total_tokens"], 250)
        self.assertEqual(task_usage["cached_input_tokens"], 90)
        self.assertEqual(task_usage["reasoning_output_tokens"], 10)

        requests = [
            event
            for event in trajectory.events
            if event.event_type == EventType.REQUEST_BUILT
        ]
        self.assertEqual(
            [event.payload["request_tokens_local"] for event in requests], [None, None]
        )
        self.assertTrue(
            all(
                event.payload["request_token_count_source"] == "missing"
                for event in requests
            )
        )
        terminals = [
            event
            for event in trajectory.events
            if event.event_type == EventType.API_COMPLETED
        ]
        self.assertEqual(
            [
                event.payload["provider_input_tokens_post_response_audit"]
                for event in terminals
            ],
            [100, 120],
        )
        self.assertTrue(
            all(
                event.payload["provider_input_tokens_post_response_audit_source"]
                == "provider_response_usage"
                for event in terminals
            )
        )
        self.assertNotIn(
            Observable.REQUEST_LOCAL_COUNT,
            BagenSwebenchReader.capabilities.observables,
        )
        self.assertIn(
            Observable.REQUEST_BOUNDARIES,
            BagenSwebenchReader.capabilities.observables,
        )
        self.assertIn(
            Observable.TASK_TERMINATION,
            BagenSwebenchReader.capabilities.observables,
        )

        dataset = build_supervised_dataset((trajectory,))
        task_total = dataset.select(
            PredictionPosition.TASK_LAUNCH,
            PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
        )
        call_unknown = dataset.select(
            PredictionPosition.CALL_PRE,
            PredictionTarget.CALL_UNKNOWN_BILLABLE_TOKENS,
        )
        self.assertEqual(task_total.rows[0].label, 250)
        self.assertTrue(
            all(row.status == LabelStatus.MISSING for row in call_unknown.rows)
        )
        self.assertTrue(all(row.label is None for row in call_unknown.rows))
        self.assertTrue(
            all(
                row.invalid_reason == "missing_request_tokens_local"
                for row in call_unknown.rows
            )
        )
        self.assertTrue(
            all(row.point.known_offset_tokens is None for row in call_unknown.rows)
        )

    def test_format_errors_preserve_missing_usage_and_bill_known_usage(self) -> None:
        messages = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Fix the reported bug."},
            _format_error(),
            _format_error(
                request_id="format-2",
                input_tokens=50,
                output_tokens=5,
                cached_tokens=20,
                reasoning_tokens=2,
            ),
            _assistant("req-3", 70, 7, content="Recovered."),
            _exit(),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_trajectory(Path(temporary), _payload(messages))
            trajectory = BagenSwebenchReader().read(path, _metadata())

        failures = [
            event
            for event in trajectory.events
            if event.event_type == EventType.API_FAILED
        ]
        self.assertEqual(len(failures), 2)
        self.assertIsNone(failures[0].payload["usage"])
        self.assertEqual(failures[1].payload["usage"]["total_tokens"], 55)
        self.assertIsNone(
            failures[0].payload["provider_input_tokens_post_response_audit"]
        )
        self.assertEqual(
            failures[1].payload["provider_input_tokens_post_response_audit"], 50
        )
        self.assertTrue(all(event.payload["error_type"] == "FormatError" for event in failures))

        finished = trajectory.events[-1].payload
        self.assertIsNone(finished["usage"])
        self.assertEqual(finished["known_usage_attempts"], 2)
        self.assertEqual(finished["missing_usage_attempts"], 1)
        self.assertEqual(finished["format_error_recovery_calls"], 2)

        requests = [
            event
            for event in trajectory.events
            if event.event_type == EventType.REQUEST_BUILT
        ]
        dataset = build_supervised_dataset((trajectory,))

        first_unknown = next(
            row
            for row in dataset.rows
            if row.point.source_event_id == requests[0].event_id
            and row.point.target == PredictionTarget.CALL_UNKNOWN_BILLABLE_TOKENS
        )
        second_unknown = next(
            row
            for row in dataset.rows
            if row.point.source_event_id == requests[1].event_id
            and row.point.target == PredictionTarget.CALL_UNKNOWN_BILLABLE_TOKENS
        )
        third_update = next(
            row
            for row in dataset.rows
            if row.point.source_event_id == requests[2].event_id
            and row.point.position == PredictionPosition.TASK_UPDATE
        )
        task_launch = next(
            row
            for row in dataset.rows
            if row.point.position == PredictionPosition.TASK_LAUNCH
        )
        self.assertEqual(first_unknown.status, LabelStatus.MISSING)
        self.assertEqual(first_unknown.invalid_reason, "missing_usage")
        self.assertEqual(second_unknown.status, LabelStatus.MISSING)
        self.assertEqual(
            second_unknown.invalid_reason, "missing_request_tokens_local"
        )
        self.assertIsNone(second_unknown.label)
        self.assertIsNone(second_unknown.point.known_offset_tokens)
        self.assertEqual(task_launch.status, LabelStatus.MISSING)
        self.assertEqual(task_launch.invalid_reason, "missing_task_usage")
        self.assertEqual(third_update.point.features["failed_api_attempts"], 2)
        self.assertEqual(third_update.point.features["known_usage_attempts"], 1)
        self.assertEqual(third_update.point.features["missing_usage_attempts"], 1)
        self.assertEqual(
            third_update.point.features["cumulative_provider_input_tokens"], 50
        )
        self.assertEqual(
            third_update.point.features["cumulative_provider_output_tokens"], 5
        )

    def test_tool_calls_pair_once_and_preserve_failures_or_exit_intercept(self) -> None:
        failed_call = _tool_call("tool-failed", "false")
        submit_call = _tool_call("tool-submit", "submit patch")
        messages = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Fix the reported bug."},
            _assistant("req-1", 100, 10, tool_calls=[failed_call]),
            _tool_result("tool-failed", returncode=2, content="failed"),
            _assistant("req-2", 120, 20, tool_calls=[submit_call]),
            _exit(),
        ]
        payload = _payload(messages)
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_trajectory(Path(temporary), payload)
            trajectory = BagenSwebenchReader().read(path, _metadata())

            unmatched = copy.deepcopy(payload)
            unmatched["messages"][3]["tool_call_id"] = "not-pending"  # type: ignore[index]
            bad_path = _write_trajectory(Path(temporary) / "bad", unmatched)
            with self.assertRaises(BagenSwebenchSchemaError):
                BagenSwebenchReader().read(bad_path, _metadata())

        tool_events = [
            event
            for event in trajectory.events
            if event.event_type
            in {EventType.TOOL_STARTED, EventType.TOOL_COMPLETED, EventType.TOOL_FAILED}
        ]
        self.assertEqual(
            [event.event_type for event in tool_events],
            [EventType.TOOL_FAILED, EventType.TOOL_COMPLETED],
        )
        self.assertEqual(
            [event.payload["tool_call_id"] for event in tool_events],
            ["tool-failed", "tool-submit"],
        )
        self.assertEqual(tool_events[0].payload["returncode"], 2)
        self.assertTrue(tool_events[1].payload["terminal_intercept"])
        self.assertNotEqual(
            tool_events[0].logical_call_id, tool_events[1].logical_call_id
        )

    def test_future_usage_changes_labels_and_post_response_audit_not_prefix_features(
        self,
    ) -> None:
        first_call = _assistant("req-1", 100, 10, content="First response")
        base_messages = _base_messages(
            first_call,
            _assistant("req-2", 120, 20, content="Short suffix"),
        )
        base_payload = _payload(base_messages)
        changed_payload = copy.deepcopy(base_payload)
        changed_first = changed_payload["messages"][2]  # type: ignore[index]
        first_response = changed_first["extra"]["response"]  # type: ignore[index]
        first_response["usage"]["prompt_tokens"] = 900  # type: ignore[index]
        first_response["usage"]["total_tokens"] = 910  # type: ignore[index]
        changed_second = changed_payload["messages"][3]  # type: ignore[index]
        changed_second["content"] = "Different future response"
        response = changed_second["extra"]["response"]  # type: ignore[index]
        response["choices"][0]["message"]["content"] = "Different future response"  # type: ignore[index]
        response["usage"]["completion_tokens"] = 200  # type: ignore[index]
        response["usage"]["total_tokens"] = 320  # type: ignore[index]
        changed_payload["info"]["submission"] = "different future patch"  # type: ignore[index]
        changed_payload["messages"][-1] = _exit("different future patch")  # type: ignore[index]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_path = _write_trajectory(root / "first", base_payload)
            second_path = _write_trajectory(root / "second", changed_payload)
            reader = BagenSwebenchReader()
            first = reader.read(first_path, _metadata(run_identity="prefix-run"))
            changed = reader.read(second_path, _metadata(run_identity="prefix-run"))

        first_snapshot = replay_feature_snapshots(first.events)[0]
        changed_snapshot = replay_feature_snapshots(changed.events)[0]
        self.assertEqual(first_snapshot.feature_hash, changed_snapshot.feature_hash)
        self.assertIsNone(first_snapshot.values["current_request_tokens_local"])
        self.assertIsNone(changed_snapshot.values["current_request_tokens_local"])

        first_request = next(
            event for event in first.events if event.event_type == EventType.REQUEST_BUILT
        )
        changed_request = next(
            event for event in changed.events if event.event_type == EventType.REQUEST_BUILT
        )
        self.assertIsNone(first_request.payload["request_tokens_local"])
        self.assertIsNone(changed_request.payload["request_tokens_local"])
        first_terminal = next(
            event for event in first.events if event.event_type == EventType.API_COMPLETED
        )
        changed_terminal = next(
            event for event in changed.events if event.event_type == EventType.API_COMPLETED
        )
        self.assertEqual(
            first_terminal.payload["provider_input_tokens_post_response_audit"], 100
        )
        self.assertEqual(
            changed_terminal.payload["provider_input_tokens_post_response_audit"], 900
        )

        first_unknown = next(
            row
            for row in build_supervised_dataset((first,)).rows
            if row.point.position == PredictionPosition.TASK_PRE
            and row.point.target
            == PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS
        )
        changed_unknown = next(
            row
            for row in build_supervised_dataset((changed,)).rows
            if row.point.position == PredictionPosition.TASK_PRE
            and row.point.target
            == PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS
        )
        self.assertEqual(first_unknown.point.features, changed_unknown.point.features)
        self.assertEqual(first_unknown.status, LabelStatus.MISSING)
        self.assertEqual(changed_unknown.status, LabelStatus.MISSING)
        self.assertIsNone(first_unknown.label)
        self.assertIsNone(changed_unknown.label)
        self.assertIsNone(first_unknown.point.known_offset_tokens)
        self.assertIsNone(changed_unknown.point.known_offset_tokens)

        first_total = build_supervised_dataset((first,)).select(
            PredictionPosition.TASK_LAUNCH,
            PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
        ).rows[0]
        changed_total = build_supervised_dataset((changed,)).select(
            PredictionPosition.TASK_LAUNCH,
            PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
        ).rows[0]
        self.assertEqual(first_total.label, 250)
        self.assertEqual(changed_total.label, 1_230)

        forbidden = {
            "source_file_sha256",
            "total_turns",
            "relative_progress",
            "final_state",
            "rollout_success",
            "success",
            "target_output",
            "actual_tokens_used_so_far",
            "actual_remaining_total_tokens",
            "actual_can_finish",
            "prediction",
            "evaluator_metrics",
            "estimator_output",
            "suffix_messages",
            "future_tool_calls",
            "future_exit_status",
        }
        self.assertFalse(forbidden & set(first_unknown.point.features))

    def test_unknown_schema_fails_closed(self) -> None:
        original = _payload(
            _base_messages(_assistant("req-1", 100, 10, content="Done"))
        )
        cases: list[dict[str, object]] = []

        unknown_top_level = copy.deepcopy(original)
        unknown_top_level["predictions"] = []
        cases.append(unknown_top_level)

        unknown_message = copy.deepcopy(original)
        unknown_message["messages"][2]["evaluator_score"] = 1.0  # type: ignore[index]
        cases.append(unknown_message)

        unknown_info = copy.deepcopy(original)
        unknown_info["info"]["resolved"] = True  # type: ignore[index]
        cases.append(unknown_info)

        unsupported_format = copy.deepcopy(original)
        unsupported_format["trajectory_format"] = "sera-aggregate-1"
        cases.append(unsupported_format)

        with tempfile.TemporaryDirectory() as temporary:
            for index, payload in enumerate(cases):
                with self.subTest(case=index):
                    path = _write_trajectory(Path(temporary) / str(index), payload)
                    with self.assertRaises(BagenSwebenchSchemaError):
                        BagenSwebenchReader().read(path, _metadata())

    def test_duplicate_keys_and_non_finite_json_fail_closed(self) -> None:
        original = _payload(
            _base_messages(_assistant("req-1", 100, 10, content="Done"))
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate_path = _write_trajectory(root / "duplicate", original)
            raw = duplicate_path.read_text(encoding="utf-8")
            marker = '"prompt_tokens": 100'
            self.assertIn(marker, raw)
            duplicate_path.write_text(
                raw.replace(
                    marker,
                    '"prompt_tokens": 100, "prompt_tokens": 101',
                    1,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(BagenSwebenchSchemaError, "duplicate"):
                BagenSwebenchReader().read(duplicate_path, _metadata())

            for index, invalid in enumerate(
                (float("nan"), float("inf"), float("-inf")),
                start=1,
            ):
                payload = copy.deepcopy(original)
                usage = payload["messages"][2]["extra"]["response"]["usage"]  # type: ignore[index]
                usage["prompt_tokens"] = invalid  # type: ignore[index]
                path = _write_trajectory(root / f"non-finite-{index}", payload)
                with self.subTest(value=invalid), self.assertRaisesRegex(
                    BagenSwebenchSchemaError, "non-finite"
                ):
                    BagenSwebenchReader().read(path, _metadata())

    def test_non_negative_counts_require_json_integers_without_coercion(self) -> None:
        original = _payload(
            _base_messages(_assistant("req-1", 100, 10, content="Done"))
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, invalid in enumerate((True, 100.5, "100"), start=1):
                payload = copy.deepcopy(original)
                usage = payload["messages"][2]["extra"]["response"]["usage"]  # type: ignore[index]
                usage["prompt_tokens"] = invalid  # type: ignore[index]
                path = _write_trajectory(root / f"invalid-count-{index}", payload)
                with self.subTest(value=invalid), self.assertRaisesRegex(
                    BagenSwebenchSchemaError, "must be an integer"
                ):
                    BagenSwebenchReader().read(path, _metadata())

    def test_tool_argument_json_rejects_duplicate_keys(self) -> None:
        call = _tool_call("tool-1", "true")
        call["function"]["arguments"] = (  # type: ignore[index]
            '{"command":"true","command":"false"}'
        )
        payload = _payload(
            _base_messages(
                _assistant("req-1", 100, 10, tool_calls=[call]),
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_trajectory(Path(temporary), payload)
            with self.assertRaisesRegex(BagenSwebenchSchemaError, "duplicate"):
                BagenSwebenchReader().read(path, _metadata())

    def test_iter_directory_reads_only_exact_traj_json_files(self) -> None:
        messages = _base_messages(_assistant("req-1", 100, 10, content="Done"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_trajectory(
                root / "a",
                _payload(messages, instance_id="repo__project-1"),
                instance_id="repo__project-1",
            )
            _write_trajectory(
                root / "z",
                _payload(messages, instance_id="repo__project-2"),
                instance_id="repo__project-2",
            )
            (root / "preds.json").write_text("not valid JSON", encoding="utf-8")
            (root / "aggregate.json").write_text("not valid JSON", encoding="utf-8")
            (root / "ignored.traj.json.bak").write_text(
                "not valid JSON", encoding="utf-8"
            )
            (root / "directory.traj.json").mkdir()

            trajectories = tuple(
                BagenSwebenchReader().iter_directory(
                    root, _metadata(run_identity="directory-run")
                )
            )

        self.assertEqual(
            {trajectory.task_id for trajectory in trajectories},
            {"repo__project-1", "repo__project-2"},
        )

    def test_optional_identity_fields_and_streaming_read(self) -> None:
        payload = _payload(
            _base_messages(_assistant("req-1", 100, 10, content="Done")),
            include_instance_id=False,
            include_trajectory_format=False,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_trajectory(Path(temporary), payload)
            with (
                patch.object(
                    Path,
                    "read_text",
                    side_effect=AssertionError("Path.read_text loads the whole file"),
                ),
                patch.object(
                    json,
                    "load",
                    side_effect=AssertionError("json.load loads the whole document"),
                ),
            ):
                trajectory = BagenSwebenchReader().read(path, _metadata())

        self.assertEqual(trajectory.task_id, INSTANCE_ID)

    def test_inconsistent_provider_usage_total_is_rejected(self) -> None:
        assistant = _assistant("req-1", 100, 10, content="Done")
        usage = assistant["extra"]["response"]["usage"]  # type: ignore[index]
        usage["total_tokens"] = 999  # type: ignore[index]
        payload = _payload(_base_messages(assistant))
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_trajectory(Path(temporary), payload)
            with self.assertRaises(BagenSwebenchSchemaError):
                BagenSwebenchReader().read(path, _metadata())

    def test_limits_exceeded_is_censored_not_completed_failure(self) -> None:
        payload = _payload(
            _base_messages(_assistant("req-1", 100, 10, content="Still working")),
            submission="",
        )
        payload["info"]["exit_status"] = "LimitsExceeded"  # type: ignore[index]
        payload["messages"][-1] = {  # type: ignore[index]
            "role": "exit",
            "content": "LimitsExceeded",
            "extra": {"exit_status": "LimitsExceeded", "submission": ""},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_trajectory(Path(temporary), payload)
            trajectory = BagenSwebenchReader().read(path, _metadata())
            dataset = build_supervised_dataset((trajectory,))

        self.assertEqual(trajectory.events[-1].event_type, EventType.TASK_ABORTED)
        self.assertEqual(trajectory.events[-1].payload["reason"], "max_turns")
        task_rows = [
            row
            for row in dataset.rows
            if row.point.position
            in {PredictionPosition.TASK_LAUNCH, PredictionPosition.TASK_PRE}
        ]
        self.assertTrue(task_rows)
        self.assertTrue(all(row.status == LabelStatus.CENSORED for row in task_rows))
        self.assertTrue(all(row.invalid_reason == "max_turns" for row in task_rows))


if __name__ == "__main__":
    unittest.main()
