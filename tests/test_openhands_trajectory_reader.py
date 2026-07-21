from __future__ import annotations

import copy
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from token_prediction.collection.openhands_trajectory import (
    OpenHandsArchiveMetadata,
    OpenHandsArchiveReader,
    OpenHandsArchiveSchemaError,
)
from token_prediction.contracts import EventType, Observable
from token_prediction.features import replay_feature_snapshots


WRAPPER = "gpt_5.2_4runs"
TASK_ID = "django__django-12345"
MODEL = "gpt-5.2"
RAW_SYSTEM = "RAW_SYSTEM_PROMPT_SENTINEL"
RAW_USER = "RAW_USER_PROMPT_SENTINEL"
RAW_ANSWER = "RAW_ASSISTANT_ANSWER_SENTINEL"
RAW_ARGUMENTS = '{"cmd":"RAW_TOOL_ARGUMENT_SENTINEL"}'
RAW_TOOL_OUTPUT = "RAW_TOOL_OUTPUT_SENTINEL"
_MISSING = object()


def _run_directory(run: int) -> str:
    return f"{MODEL}_maxiter_500_N_v0.62.0-no-hint-run_{run}"


def _member(run: int, task_id: str, filename: str) -> str:
    return (
        f"{WRAPPER}/{_run_directory(run)}/llm_completions/"
        f"{task_id}/{filename}"
    )


def _report_member(run: int, task_id: str) -> str:
    return f"{WRAPPER}/{_run_directory(run)}/eval_outputs/{task_id}/report.json"


def _task_log_member(run: int) -> str:
    return f"{WRAPPER}/{_run_directory(run)}/output.jsonl"


def _text_part(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


def _base_messages() -> list[dict[str, object]]:
    return [
        {"role": "system", "content": _text_part(RAW_SYSTEM)},
        {"role": "user", "content": _text_part(RAW_USER)},
    ]


def _tool_call(
    tool_call_id: str,
    *,
    response_shape: bool,
) -> dict[str, object]:
    result: dict[str, object] = {
        "id": tool_call_id,
        "type": "function",
        "function": {"name": "shell", "arguments": RAW_ARGUMENTS},
    }
    if response_shape:
        result["index"] = 0
    return result


def _tool_definition() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run one synthetic command.",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
                "additionalProperties": False,
            },
        },
    }


def _usage(input_tokens: int, output_tokens: int) -> dict[str, object]:
    return {
        "completion_tokens": output_tokens,
        "prompt_tokens": input_tokens,
        "total_tokens": input_tokens + output_tokens,
        "completion_tokens_details": {
            "accepted_prediction_tokens": 0,
            "audio_tokens": 0,
            "reasoning_tokens": min(1, output_tokens),
            "rejected_prediction_tokens": 0,
            "text_tokens": output_tokens,
            "image_tokens": 0,
        },
        "prompt_tokens_details": {
            "audio_tokens": 0,
            "cached_tokens": min(2, input_tokens),
            "text_tokens": input_tokens,
            "image_tokens": 0,
        },
        "cost": 0.01,
        "is_byok": False,
        "cost_details": {
            "upstream_inference_cost": None,
            "upstream_inference_prompt_cost": None,
            "upstream_inference_completions_cost": None,
        },
    }


def _response(
    *,
    response_id: str,
    created: int,
    input_tokens: int = 10,
    output_tokens: int = 3,
    answer: str = RAW_ANSWER,
    tool_call_id: str | None = None,
    usage: object = _MISSING,
) -> dict[str, object]:
    tool_calls = (
        [_tool_call(tool_call_id, response_shape=True)]
        if tool_call_id is not None
        else []
    )
    result: dict[str, object] = {
        "id": response_id,
        "created": created,
        "model": "gpt-5.2-2025-12-11",
        "object": "chat.completion",
        "system_fingerprint": None,
        "choices": [
            {
                "finish_reason": "tool_calls" if tool_calls else "stop",
                "index": 0,
                "message": {
                    "content": answer,
                    "role": "assistant",
                    "tool_calls": tool_calls,
                    "function_call": None,
                    "reasoning_content": "synthetic reasoning",
                },
                "provider_specific_fields": {
                    "native_finish_reason": "tool_calls" if tool_calls else "stop"
                },
            }
        ],
        "provider": "openai",
    }
    if usage is _MISSING:
        result["usage"] = _usage(input_tokens, output_tokens)
    elif usage != "omit":
        result["usage"] = usage
    return result


def _historical_assistant(
    *,
    answer: str = RAW_ANSWER,
    tool_call_id: str | None = None,
) -> dict[str, object]:
    return {
        "content": _text_part(answer),
        "role": "assistant",
        **(
            {"tool_calls": [_tool_call(tool_call_id, response_shape=False)]}
            if tool_call_id is not None
            else {}
        ),
    }


def _tool_result(tool_call_id: str) -> dict[str, object]:
    return {
        "content": _text_part(RAW_TOOL_OUTPUT),
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": "shell",
    }


def _completion(
    *,
    timestamp: float,
    created: int,
    response_id: str,
    messages: list[dict[str, object]] | None = None,
    input_tokens: int = 10,
    output_tokens: int = 3,
    tool_call_id: str | None = None,
    usage: object = _MISSING,
) -> dict[str, object]:
    return {
        "messages": copy.deepcopy(messages if messages is not None else _base_messages()),
        "response": _response(
            response_id=response_id,
            created=created,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_call_id=tool_call_id,
            usage=usage,
        ),
        "args": [],
        "kwargs": {"tools": [_tool_definition()]},
        "timestamp": timestamp,
        "cost": 0.01,
    }


def _completion_name(timestamp: float) -> str:
    return f"openai__{MODEL}-{timestamp:.1f}.json"


def _report(task_id: str, *, resolved: bool = True) -> dict[str, object]:
    empty = {"success": [], "failure": []}
    return {
        task_id: {
            "patch_is_None": False,
            "patch_exists": True,
            "patch_successfully_applied": True,
            "resolved": resolved,
            "tests_status": {
                "FAIL_TO_PASS": {
                    "success": ["RAW_TEST_NAME_SENTINEL"],
                    "failure": [],
                },
                "PASS_TO_PASS": copy.deepcopy(empty),
                "FAIL_TO_FAIL": copy.deepcopy(empty),
                "PASS_TO_FAIL": copy.deepcopy(empty),
            },
        }
    }


def _metric_usage(
    response_id: str,
    input_tokens: int,
    output_tokens: int,
) -> dict[str, object]:
    return {
        "model": "gpt-5.2-2025-12-11",
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "cache_read_tokens": min(2, input_tokens),
        "cache_write_tokens": 0,
        "context_window": 400_000,
        "per_turn_token": input_tokens + output_tokens,
        "response_id": response_id,
    }


def _task_log_line(
    task_id: str,
    *,
    response_usages: tuple[tuple[str, int, int], ...] = (),
    finished: bool = True,
    error: str | None = None,
    history: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    if history is None:
        history = [
            {
                "id": 0,
                "timestamp": "2026-01-01T00:00:00+00:00",
                "source": "agent",
                "message": "synthetic system event",
                "action": "system",
                "args": {},
            }
        ]
        for index, _ in enumerate(response_usages, start=1):
            history.append(
                {
                    "id": index,
                    "timestamp": f"2026-01-01T00:00:{index:02d}+00:00",
                    "source": "agent",
                    "message": "synthetic model event",
                    "action": "think",
                    "args": {},
                    "llm_metrics": {},
                }
            )
        if finished:
            history.append(
                {
                    "id": len(history),
                    "timestamp": f"2026-01-01T00:01:{len(history):02d}+00:00",
                    "source": "agent",
                    "message": "synthetic finish event",
                    "action": "finish",
                    "args": {},
                }
            )
    input_total = sum(item[1] for item in response_usages)
    output_total = sum(item[2] for item in response_usages)
    accumulated_usage = _metric_usage("", input_total, output_total)
    accumulated_usage["cache_read_tokens"] = sum(
        min(2, input_tokens) for _, input_tokens, _ in response_usages
    )
    return {
        "instance_id": task_id,
        "test_result": {},
        "instruction": "RAW_TASK_INSTRUCTION_SENTINEL",
        "metadata": {"max_iterations": 500},
        "history": history,
        "metrics": {
            "accumulated_cost": 0.01 * len(response_usages),
            "max_budget_per_task": None,
            "accumulated_token_usage": accumulated_usage,
            "costs": [
                {
                    "model": "gpt-5.2-2025-12-11",
                    "cost": 0.01,
                    "timestamp": 1_700_000_000.0 + index,
                }
                for index, _ in enumerate(response_usages)
            ],
            "response_latencies": [
                {
                    "model": "gpt-5.2-2025-12-11",
                    "latency": 1.0,
                    "response_id": response_id,
                }
                for response_id, _, _ in response_usages
            ],
            "token_usages": [
                _metric_usage(response_id, input_tokens, output_tokens)
                for response_id, input_tokens, output_tokens in response_usages
            ],
            "condenser": [],
        },
        "error": error,
        "instance": {"instance_id": task_id},
    }


def _startup_error_task_log_line(task_id: str) -> dict[str, object]:
    return {
        "instance_id": task_id,
        "test_result": {},
        "instruction": None,
        "metadata": None,
        "history": None,
        "metrics": None,
        "error": "RAW_TOP_LEVEL_ERROR_SENTINEL",
        "instance": None,
    }


def _jsonl_bytes(*values: object) -> bytes:
    return b"\n".join(_json_bytes(value) for value in values) + b"\n"


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_archive(
    path: Path,
    members: list[tuple[str, object]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        for name, value in members:
            raw = value if isinstance(value, bytes) else _json_bytes(value)
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            info.mtime = 0
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(raw))
    return path


def _metadata() -> OpenHandsArchiveMetadata:
    return OpenHandsArchiveMetadata(archive_identity="a" * 64)


def _event_types(trajectory: object) -> list[EventType]:
    return [event.event_type for event in trajectory.events]  # type: ignore[attr-defined]


class OpenHandsTrajectoryReaderTests(unittest.TestCase):
    def test_four_runs_preserve_same_task_mapping_and_stable_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.tar.gz"
            members: list[tuple[str, object]] = []
            for run in range(1, 5):
                members.extend(
                    [
                        (_report_member(run, TASK_ID), _report(TASK_ID)),
                        (
                            _member(run, TASK_ID, _completion_name(1.0)),
                            _completion(
                                timestamp=1.0,
                                created=1_700_000_001,
                                response_id=f"response-run-{run}",
                            ),
                        ),
                    ]
                )
            _write_archive(path, members)

            first = OpenHandsArchiveReader().read(path, _metadata())
            second = OpenHandsArchiveReader().read(path, _metadata())

        self.assertEqual(len(first), 4)
        self.assertEqual({item.task_id for item in first}, {TASK_ID})
        self.assertEqual({item.run_id for item in first}, {f"run_{i}" for i in range(1, 5)})
        self.assertEqual(len({item.trajectory_id for item in first}), 4)
        self.assertEqual(len({item.condition_id for item in first}), 1)
        self.assertEqual(
            [[event.to_dict() for event in item.events] for item in first],
            [[event.to_dict() for event in item.events] for item in second],
        )
        for trajectory in first:
            self.assertEqual(
                [event.event_seq for event in trajectory.events],
                list(range(len(trajectory.events))),
            )
            self.assertEqual(
                [event.event_id for event in trajectory.events],
                [
                    f"{trajectory.trajectory_id}:event:{index}"
                    for index in range(len(trajectory.events))
                ],
            )

    def test_condition_identity_includes_response_provider_and_resolved_model(
        self,
    ) -> None:
        baseline = _completion(
            timestamp=1.0,
            created=1_700_000_001,
            response_id="response-baseline",
        )
        changed_provider = copy.deepcopy(baseline)
        changed_provider["response"]["id"] = "response-provider"  # type: ignore[index]
        changed_provider["response"]["provider"] = "azure"  # type: ignore[index]
        changed_model = copy.deepcopy(baseline)
        changed_model["response"]["id"] = "response-model"  # type: ignore[index]
        changed_model["response"]["model"] = "gpt-5.2-2026-01-15"  # type: ignore[index]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reader = OpenHandsArchiveReader()
            trajectories = []
            for name, completion in (
                ("baseline", baseline),
                ("provider", changed_provider),
                ("model", changed_model),
            ):
                path = _write_archive(
                    root / name / "fixture.tar.gz",
                    [(_member(1, TASK_ID, _completion_name(1.0)), completion)],
                )
                first = reader.read(path, _metadata())[0]
                second = reader.read(path, _metadata())[0]
                self.assertEqual(first.condition_id, second.condition_id)
                trajectories.append(first)

        self.assertEqual(len({item.condition_id for item in trajectories}), 3)
        started = [item.events[0].payload for item in trajectories]
        self.assertEqual(
            [payload["provider"] for payload in started],
            ["openai", "azure", "openai"],
        )
        self.assertEqual(
            [payload["resolved_model_id"] for payload in started],
            [
                "gpt-5.2-2025-12-11",
                "gpt-5.2-2025-12-11",
                "gpt-5.2-2026-01-15",
            ],
        )

    def test_archive_is_read_forward_only_without_extract_or_member_listing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_archive(
                Path(temporary) / "fixture.tar.gz",
                [
                    (
                        _member(1, TASK_ID, _completion_name(1.0)),
                        _completion(
                            timestamp=1.0,
                            created=1_700_000_001,
                            response_id="response-1",
                        ),
                    )
                ],
            )
            with (
                patch.object(
                    tarfile.TarFile,
                    "getmembers",
                    side_effect=AssertionError("getmembers is forbidden"),
                ) as getmembers,
                patch.object(
                    tarfile.TarFile,
                    "extract",
                    side_effect=AssertionError("extract is forbidden"),
                ) as extract,
                patch.object(
                    tarfile.TarFile,
                    "extractall",
                    side_effect=AssertionError("extractall is forbidden"),
                ) as extractall,
                patch(
                    "token_prediction.collection.openhands_trajectory.tarfile.open",
                    wraps=tarfile.open,
                ) as open_archive,
            ):
                trajectories = OpenHandsArchiveReader().read(path, _metadata())

        self.assertEqual(len(trajectories), 1)
        self.assertEqual(open_archive.call_args.kwargs["mode"], "r|gz")
        getmembers.assert_not_called()
        extract.assert_not_called()
        extractall.assert_not_called()

    def test_rejects_unsafe_archive_paths(self) -> None:
        unsafe_names = (
            f"../{_member(1, TASK_ID, _completion_name(1.0))}",
            f"/{_member(1, TASK_ID, _completion_name(1.0))}",
            _member(1, TASK_ID, _completion_name(1.0)).replace("/", "\\", 1),
        )
        for unsafe_name in unsafe_names:
            with self.subTest(unsafe_name=unsafe_name), tempfile.TemporaryDirectory() as temporary:
                path = _write_archive(
                    Path(temporary) / "unsafe.tar.gz",
                    [
                        (
                            unsafe_name,
                            _completion(
                                timestamp=1.0,
                                created=1_700_000_001,
                                response_id="response-1",
                            ),
                        )
                    ],
                )
                with self.assertRaises(OpenHandsArchiveSchemaError):
                    OpenHandsArchiveReader().read(path, _metadata())

    def test_timestamp_sources_sort_physical_reverse_order_and_usage_is_per_response(self) -> None:
        first_response = _completion(
            timestamp=1.0,
            created=1_700_000_001,
            response_id="response-0001",
            input_tokens=10,
            output_tokens=3,
            tool_call_id="raw-tool-call-1",
        )
        second_messages = [
            *_base_messages(),
            _historical_assistant(tool_call_id="raw-tool-call-1"),
            _tool_result("raw-tool-call-1"),
        ]
        second_response = _completion(
            timestamp=2.0,
            created=1_700_000_002,
            response_id="response-0002",
            messages=second_messages,
            input_tokens=20,
            output_tokens=5,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_archive(
                Path(temporary) / "fixture.tar.gz",
                [
                    (
                        _member(1, TASK_ID, _completion_name(2.0)),
                        second_response,
                    ),
                    (
                        _member(1, TASK_ID, _completion_name(1.0)),
                        first_response,
                    ),
                ],
            )
            trajectory = OpenHandsArchiveReader().read(path, _metadata())[0]

        completed = [
            event
            for event in trajectory.events
            if event.event_type == EventType.API_COMPLETED
        ]
        self.assertEqual(
            [event.payload["usage"]["input_tokens"] for event in completed],
            [10, 20],
        )
        self.assertEqual(
            [event.payload["usage"]["output_tokens"] for event in completed],
            [3, 5],
        )
        terminal = trajectory.events[-1]
        self.assertEqual(terminal.payload["usage"]["input_tokens"], 30)
        self.assertEqual(terminal.payload["usage"]["output_tokens"], 8)

    def test_conflicting_timestamp_rankings_and_duplicate_response_ids_fail_closed(self) -> None:
        first = _completion(
            timestamp=2.0,
            created=1_700_000_001,
            response_id="response-duplicate",
        )
        second_messages = [*_base_messages(), _historical_assistant()]
        second = _completion(
            timestamp=1.0,
            created=1_700_000_002,
            response_id="response-0002",
            messages=second_messages,
        )
        cases = {
            "timestamp ranking conflict": (first, second),
            "duplicate response id": (
                _completion(
                    timestamp=1.0,
                    created=1_700_000_001,
                    response_id="response-duplicate",
                ),
                _completion(
                    timestamp=2.0,
                    created=1_700_000_002,
                    response_id="response-duplicate",
                    messages=second_messages,
                ),
            ),
        }
        for label, payloads in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                path = _write_archive(
                    Path(temporary) / "fixture.tar.gz",
                    [
                        (_member(1, TASK_ID, _completion_name(1.0)), payloads[0]),
                        (_member(1, TASK_ID, _completion_name(2.0)), payloads[1]),
                    ],
                )
                with self.assertRaises(OpenHandsArchiveSchemaError):
                    OpenHandsArchiveReader().read(path, _metadata())

    def test_message_history_must_be_a_strict_exact_prefix(self) -> None:
        first = _completion(
            timestamp=1.0,
            created=1_700_000_001,
            response_id="response-0001",
        )
        valid_next_messages = [*_base_messages(), _historical_assistant()]
        cases = {
            "does not grow": _base_messages(),
            "mutates prefix": [
                {"role": "system", "content": _text_part("mutated")},
                _base_messages()[1],
                _historical_assistant(),
            ],
            "historical response mismatch": [
                *_base_messages(),
                _historical_assistant(answer="different answer"),
            ],
        }
        for label, next_messages in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                second = _completion(
                    timestamp=2.0,
                    created=1_700_000_002,
                    response_id="response-0002",
                    messages=next_messages,
                )
                path = _write_archive(
                    Path(temporary) / "fixture.tar.gz",
                    [
                        (_member(1, TASK_ID, _completion_name(1.0)), first),
                        (_member(1, TASK_ID, _completion_name(2.0)), second),
                    ],
                )
                with self.assertRaises(OpenHandsArchiveSchemaError):
                    OpenHandsArchiveReader().read(path, _metadata())

        with tempfile.TemporaryDirectory() as temporary:
            second = _completion(
                timestamp=2.0,
                created=1_700_000_002,
                response_id="response-0002",
                messages=valid_next_messages,
            )
            path = _write_archive(
                Path(temporary) / "fixture.tar.gz",
                [
                    (_member(1, TASK_ID, _completion_name(1.0)), first),
                    (_member(1, TASK_ID, _completion_name(2.0)), second),
                ],
            )
            self.assertEqual(len(OpenHandsArchiveReader().read(path, _metadata())), 1)

    def test_user_only_delta_preserves_two_calls_without_inferred_failure_or_tool(self) -> None:
        first_response_id = "response-0001"
        second_response_id = "response-0002"
        task_log = _task_log_line(
            TASK_ID,
            response_usages=(
                (first_response_id, 10, 3),
                (second_response_id, 10, 3),
            ),
        )
        first = _completion(
            timestamp=1.0,
            created=1_700_000_001,
            response_id=first_response_id,
            tool_call_id="unmaterialized-tool-call",
        )
        second = _completion(
            timestamp=2.0,
            created=1_700_000_002,
            response_id=second_response_id,
            messages=[
                *_base_messages(),
                {"role": "user", "content": _text_part("synthetic follow-up")},
            ],
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_archive(
                Path(temporary) / "fixture.tar.gz",
                [
                    (_task_log_member(1), _jsonl_bytes(task_log)),
                    (_member(1, TASK_ID, _completion_name(1.0)), first),
                    (_member(1, TASK_ID, _completion_name(2.0)), second),
                ],
            )
            trajectory = OpenHandsArchiveReader().read(path, _metadata())[0]

        types = _event_types(trajectory)
        self.assertEqual(types.count(EventType.REQUEST_BUILT), 2)
        self.assertEqual(types.count(EventType.API_ATTEMPT_STARTED), 2)
        self.assertEqual(types.count(EventType.API_COMPLETED), 2)
        self.assertNotIn(EventType.API_FAILED, types)
        self.assertNotIn(EventType.TOOL_STARTED, types)
        self.assertNotIn(EventType.TOOL_COMPLETED, types)
        self.assertNotIn(EventType.TOOL_FAILED, types)
        self.assertNotIn(EventType.GENERATION_CHECKPOINT, types)
        requests = [
            event
            for event in trajectory.events
            if event.event_type == EventType.REQUEST_BUILT
        ]
        attempts = [
            event
            for event in trajectory.events
            if event.event_type == EventType.API_ATTEMPT_STARTED
        ]
        self.assertEqual(len({event.logical_call_id for event in requests}), 2)
        self.assertEqual(
            {event.logical_call_id for event in requests},
            {event.logical_call_id for event in attempts},
        )
        self.assertTrue(all(event.payload["retry_observable"] is False for event in attempts))
        terminal = trajectory.events[-1]
        self.assertEqual(
            terminal.payload["response_not_materialized_in_next_request_count"],
            1,
        )
        self.assertEqual(terminal.payload["message_prefix_reset_count"], 0)
        self.assertEqual(terminal.payload["repeated_request_snapshot_count"], 0)

    def test_other_non_tool_non_user_deltas_remain_fail_closed(self) -> None:
        first = _completion(
            timestamp=1.0,
            created=1_700_000_001,
            response_id="response-0001",
        )
        cases = {
            "system-only delta": [
                *_base_messages(),
                {"role": "system", "content": _text_part("unexpected system")},
            ],
            "two-user delta": [
                *_base_messages(),
                {"role": "user", "content": _text_part("follow-up one")},
                {"role": "user", "content": _text_part("follow-up two")},
            ],
            "assistant after materialized response": [
                *_base_messages(),
                _historical_assistant(),
                _historical_assistant(),
            ],
        }
        for label, messages in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                second = _completion(
                    timestamp=2.0,
                    created=1_700_000_002,
                    response_id="response-0002",
                    messages=messages,
                )
                path = _write_archive(
                    Path(temporary) / "fixture.tar.gz",
                    [
                        (_member(1, TASK_ID, _completion_name(1.0)), first),
                        (_member(1, TASK_ID, _completion_name(2.0)), second),
                    ],
                )
                with self.assertRaises(OpenHandsArchiveSchemaError):
                    OpenHandsArchiveReader().read(path, _metadata())

    def test_missing_or_null_usage_is_missing_not_zero(self) -> None:
        for label, usage in (("missing", "omit"), ("null", None)):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                path = _write_archive(
                    Path(temporary) / "fixture.tar.gz",
                    [
                        (
                            _member(1, TASK_ID, _completion_name(1.0)),
                            _completion(
                                timestamp=1.0,
                                created=1_700_000_001,
                                response_id="response-1",
                                usage=usage,
                            ),
                        )
                    ],
                )
                trajectory = OpenHandsArchiveReader().read(path, _metadata())[0]

                completed = next(
                    event
                    for event in trajectory.events
                    if event.event_type == EventType.API_COMPLETED
                )
                for key in ("input_tokens", "output_tokens", "total_tokens"):
                    self.assertIsNone(completed.payload["usage"][key])
                    self.assertIsNone(trajectory.events[-1].payload["usage"][key])
                self.assertEqual(trajectory.events[-1].payload["known_usage_attempts"], 0)
                self.assertEqual(trajectory.events[-1].payload["missing_usage_attempts"], 1)

    def test_task_log_aggregate_does_not_backfill_missing_attempt_usage(self) -> None:
        task_log = _task_log_line(
            TASK_ID,
            response_usages=(("response-1", 10, 3),),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_archive(
                Path(temporary) / "fixture.tar.gz",
                [
                    (_task_log_member(1), _jsonl_bytes(task_log)),
                    (
                        _member(1, TASK_ID, _completion_name(1.0)),
                        _completion(
                            timestamp=1.0,
                            created=1_700_000_001,
                            response_id="response-1",
                            usage=None,
                        ),
                    ),
                ],
            )
            trajectory = OpenHandsArchiveReader().read(path, _metadata())[0]

        completed = next(
            event for event in trajectory.events if event.event_type == EventType.API_COMPLETED
        )
        self.assertIsNone(completed.payload["usage"]["input_tokens"])
        self.assertIsNone(completed.payload["usage"]["output_tokens"])
        terminal = trajectory.events[-1]
        self.assertEqual(terminal.payload["usage"]["input_tokens"], 10)
        self.assertEqual(terminal.payload["usage"]["output_tokens"], 3)
        self.assertEqual(terminal.payload["known_usage_attempts"], 0)
        self.assertEqual(terminal.payload["missing_usage_attempts"], 1)
        self.assertTrue(terminal.payload["task_usage_reconciled"])

    def test_provider_error_envelope_is_preserved_without_inventing_failure(self) -> None:
        completion = _completion(
            timestamp=1.0,
            created=1_700_000_001,
            response_id="response-provider-envelope",
        )
        provider_fields = completion["response"]["choices"][0][  # type: ignore[index]
            "provider_specific_fields"
        ]
        provider_fields["error"] = {  # type: ignore[index]
            "message": "RAW_PROVIDER_ERROR_SENTINEL",
            "code": 500,
            "metadata": {
                "provider_name": "openai",
                "raw": {"code": "upstream_error", "message": "RAW_UPSTREAM_SENTINEL"},
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_archive(
                Path(temporary) / "fixture.tar.gz",
                [(_member(1, TASK_ID, _completion_name(1.0)), completion)],
            )
            trajectory = OpenHandsArchiveReader().read(path, _metadata())[0]

        self.assertFalse(
            any(event.event_type == EventType.API_FAILED for event in trajectory.events)
        )
        completed = next(
            event for event in trajectory.events if event.event_type == EventType.API_COMPLETED
        )
        self.assertTrue(completed.payload["provider_error_envelope_present"])
        self.assertEqual(trajectory.events[-1].payload["provider_error_envelope_count"], 1)
        rendered = json.dumps(
            [event.to_dict() for event in trajectory.events], sort_keys=True
        )
        self.assertNotIn("RAW_PROVIDER_ERROR_SENTINEL", rendered)
        self.assertNotIn("RAW_UPSTREAM_SENTINEL", rendered)

    def test_tool_delta_emits_only_completed_and_no_unproved_attempt_facts(self) -> None:
        first = _completion(
            timestamp=1.0,
            created=1_700_000_001,
            response_id="response-0001",
            tool_call_id="raw-tool-call-1",
        )
        second = _completion(
            timestamp=2.0,
            created=1_700_000_002,
            response_id="response-0002",
            messages=[
                *_base_messages(),
                _historical_assistant(tool_call_id="raw-tool-call-1"),
                _tool_result("raw-tool-call-1"),
            ],
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_archive(
                Path(temporary) / "fixture.tar.gz",
                [
                    (_member(1, TASK_ID, _completion_name(1.0)), first),
                    (_member(1, TASK_ID, _completion_name(2.0)), second),
                ],
            )
            trajectory = OpenHandsArchiveReader().read(path, _metadata())[0]

        types = _event_types(trajectory)
        self.assertEqual(types.count(EventType.REQUEST_BUILT), 2)
        self.assertEqual(types.count(EventType.API_ATTEMPT_STARTED), 2)
        self.assertEqual(types.count(EventType.API_COMPLETED), 2)
        self.assertEqual(types.count(EventType.TOOL_COMPLETED), 1)
        self.assertNotIn(EventType.TOOL_STARTED, types)
        self.assertNotIn(EventType.TOOL_FAILED, types)
        self.assertNotIn(EventType.API_FAILED, types)
        self.assertNotIn(EventType.GENERATION_CHECKPOINT, types)
        tool_event = next(
            event for event in trajectory.events if event.event_type == EventType.TOOL_COMPLETED
        )
        self.assertFalse(tool_event.payload["failure_observable"])
        self.assertTrue(str(tool_event.payload["tool_call_id"]).startswith("tool:"))

    def test_capabilities_and_payload_do_not_claim_local_counts_or_leak_raw_content(self) -> None:
        reader = OpenHandsArchiveReader()
        self.assertIn(Observable.TASK_USAGE, reader.capabilities.observables)
        self.assertIn(Observable.CALL_USAGE, reader.capabilities.observables)
        self.assertIn(Observable.ATTEMPT_USAGE, reader.capabilities.observables)
        self.assertIn(Observable.REQUEST_MESSAGES, reader.capabilities.observables)
        self.assertIn(Observable.TOOL_EVENTS, reader.capabilities.observables)
        self.assertIn(Observable.REQUEST_BOUNDARIES, reader.capabilities.observables)
        self.assertIn(Observable.TASK_TERMINATION, reader.capabilities.observables)
        self.assertNotIn(Observable.REQUEST_LOCAL_COUNT, reader.capabilities.observables)
        self.assertNotIn(Observable.OUTPUT_DELTAS, reader.capabilities.observables)

        with tempfile.TemporaryDirectory() as temporary:
            path = _write_archive(
                Path(temporary) / "fixture.tar.gz",
                [
                    (_report_member(1, TASK_ID), _report(TASK_ID)),
                    (
                        _member(1, TASK_ID, _completion_name(1.0)),
                        _completion(
                            timestamp=1.0,
                            created=1_700_000_001,
                            response_id="RAW_RESPONSE_ID_SENTINEL",
                            tool_call_id="RAW_TOOL_CALL_ID_SENTINEL",
                        ),
                    ),
                    (
                        _member(1, TASK_ID, _completion_name(2.0)),
                        _completion(
                            timestamp=2.0,
                            created=1_700_000_002,
                            response_id="response-2",
                            messages=[
                                *_base_messages(),
                                _historical_assistant(
                                    tool_call_id="RAW_TOOL_CALL_ID_SENTINEL"
                                ),
                                _tool_result("RAW_TOOL_CALL_ID_SENTINEL"),
                            ],
                        ),
                    ),
                ],
            )
            trajectory = reader.read(path, _metadata())[0]

        request = next(
            event for event in trajectory.events if event.event_type == EventType.REQUEST_BUILT
        )
        self.assertNotIn("task_usage_observable", trajectory.events[0].payload)
        self.assertIsNone(request.payload["request_tokens_local"])
        self.assertEqual(request.payload["request_token_count_source"], "missing")
        rendered = json.dumps(
            [event.to_dict() for event in trajectory.events],
            ensure_ascii=False,
            sort_keys=True,
        )
        for secret in (
            RAW_SYSTEM,
            RAW_USER,
            RAW_ANSWER,
            RAW_ARGUMENTS,
            RAW_TOOL_OUTPUT,
            "RAW_RESPONSE_ID_SENTINEL",
            "RAW_TOOL_CALL_ID_SENTINEL",
            "RAW_TEST_NAME_SENTINEL",
        ):
            self.assertNotIn(secret, rendered)

    def test_report_is_evaluator_metadata_and_missing_report_remains_explicit(self) -> None:
        for present in (True, False):
            with self.subTest(report_present=present), tempfile.TemporaryDirectory() as temporary:
                members: list[tuple[str, object]] = []
                if present:
                    members.append((_report_member(1, TASK_ID), _report(TASK_ID)))
                members.append(
                    (
                        _member(1, TASK_ID, _completion_name(1.0)),
                        _completion(
                            timestamp=1.0,
                            created=1_700_000_001,
                            response_id="response-1",
                        ),
                    )
                )
                path = _write_archive(Path(temporary) / "fixture.tar.gz", members)
                terminal = OpenHandsArchiveReader().read(path, _metadata())[0].events[-1]

                self.assertEqual(terminal.event_type, EventType.TASK_ABORTED)
                self.assertEqual(terminal.payload["reason"], "logging_incomplete")
                self.assertEqual(terminal.payload["evaluator_report_present"], present)
                if present:
                    self.assertTrue(terminal.payload["evaluator_report"]["resolved"])
                    self.assertEqual(
                        terminal.payload["evaluator_report"]["tests_status_counts"]
                        ["FAIL_TO_PASS"],
                        {"success": 1, "failure": 0},
                    )
                else:
                    self.assertIsNone(terminal.payload["evaluator_report"])

    def test_output_jsonl_proves_finished_lifecycle_and_keeps_zero_call_task(self) -> None:
        zero_call_task = "sympy__sympy-54321"
        completed_task_log = _task_log_line(
            TASK_ID,
            response_usages=(("response-1", 10, 3),),
        )
        zero_call_log = _task_log_line(zero_call_task)
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_archive(
                Path(temporary) / "fixture.tar.gz",
                [
                    (
                        _task_log_member(1),
                        _jsonl_bytes(completed_task_log, zero_call_log),
                    ),
                    (
                        _member(1, TASK_ID, _completion_name(1.0)),
                        _completion(
                            timestamp=1.0,
                            created=1_700_000_001,
                            response_id="response-1",
                            input_tokens=10,
                            output_tokens=3,
                        ),
                    ),
                ],
            )
            trajectories = OpenHandsArchiveReader().read(path, _metadata())

        self.assertEqual({item.task_id for item in trajectories}, {TASK_ID, zero_call_task})
        by_task = {item.task_id: item for item in trajectories}
        completed_terminal = by_task[TASK_ID].events[-1]
        self.assertEqual(completed_terminal.event_type, EventType.TASK_FINISHED)
        self.assertEqual(completed_terminal.payload["reason"], "agent_finished")
        self.assertEqual(completed_terminal.payload["lifecycle_source"], "output.jsonl")
        self.assertTrue(completed_terminal.payload["completion_logging_complete"])
        self.assertTrue(completed_terminal.payload["task_usage_reconciled"])
        self.assertEqual(completed_terminal.payload["usage"]["total_tokens"], 13)
        self.assertNotIn(
            "RAW_TASK_INSTRUCTION_SENTINEL",
            json.dumps(
                [event.to_dict() for item in trajectories for event in item.events]
            ),
        )

        zero_call = by_task[zero_call_task]
        self.assertEqual(
            _event_types(zero_call),
            [EventType.TASK_STARTED, EventType.TASK_FINISHED],
        )
        self.assertEqual(zero_call.events[-1].payload["completion_snapshot_count"], 0)
        self.assertEqual(zero_call.events[-1].payload["known_usage_attempts"], 0)
        self.assertEqual(zero_call.events[-1].payload["missing_usage_attempts"], 0)
        self.assertEqual(zero_call.events[-1].payload["usage"]["total_tokens"], 0)

    def test_output_jsonl_explicit_startup_error_emits_zero_call_aborted(self) -> None:
        error_task = "pytest__pytest-22222"
        normal_task_log = _task_log_line(
            TASK_ID,
            response_usages=(("response-1", 10, 3),),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_archive(
                Path(temporary) / "fixture.tar.gz",
                [
                    (
                        _task_log_member(1),
                        _jsonl_bytes(
                            normal_task_log,
                            _startup_error_task_log_line(error_task),
                        ),
                    ),
                    (
                        _member(1, TASK_ID, _completion_name(1.0)),
                        _completion(
                            timestamp=1.0,
                            created=1_700_000_001,
                            response_id="response-1",
                        ),
                    ),
                ],
            )
            trajectories = OpenHandsArchiveReader().read(path, _metadata())

        error_trajectory = next(item for item in trajectories if item.task_id == error_task)
        self.assertEqual(
            _event_types(error_trajectory),
            [EventType.TASK_STARTED, EventType.TASK_ABORTED],
        )
        terminal = error_trajectory.events[-1]
        self.assertEqual(terminal.payload["outcome"], "error")
        self.assertEqual(terminal.payload["reason"], "task_error")
        self.assertIsNotNone(terminal.payload["task_error_hash"])
        self.assertEqual(
            terminal.payload["task_error_chars"],
            len("RAW_TOP_LEVEL_ERROR_SENTINEL"),
        )
        self.assertIsNone(terminal.payload["usage"]["input_tokens"])
        self.assertIsNone(terminal.payload["usage"]["output_tokens"])
        self.assertEqual(
            terminal.payload["usage_scope"],
            "missing_no_completion_or_task_metrics",
        )
        rendered = json.dumps([event.to_dict() for event in error_trajectory.events])
        self.assertNotIn("RAW_TOP_LEVEL_ERROR_SENTINEL", rendered)

    def test_output_jsonl_explicit_tool_failure_emits_started_and_failed(self) -> None:
        tool_call_id = "RAW_TOOL_CALL_ID_SENTINEL"
        history = [
            {
                "id": 0,
                "timestamp": "2026-01-01T00:00:00+00:00",
                "source": "agent",
                "message": "system",
                "action": "system",
                "args": {},
            },
            {
                "id": 1,
                "timestamp": "2026-01-01T00:00:01+00:00",
                "source": "agent",
                "message": "run tool",
                "action": "run",
                "args": {},
                "tool_call_metadata": {
                    "function_name": "shell",
                    "tool_call_id": tool_call_id,
                    "model_response": {"id": "response-1"},
                    "total_calls_in_response": 1,
                },
                "llm_metrics": {},
            },
            {
                "id": 2,
                "timestamp": "2026-01-01T00:00:02+00:00",
                "source": "environment",
                "message": "tool failed",
                "observation": "run",
                "content": "RAW_JSONL_TOOL_OUTPUT_SENTINEL",
                "extras": {},
                "cause": 1,
                "success": False,
                "tool_call_metadata": {
                    "function_name": "shell",
                    "tool_call_id": tool_call_id,
                    "model_response": {"id": "response-1"},
                    "total_calls_in_response": 1,
                },
            },
            {
                "id": 3,
                "timestamp": "2026-01-01T00:00:03+00:00",
                "source": "agent",
                "message": "finish",
                "action": "finish",
                "args": {},
            },
        ]
        task_log = _task_log_line(
            TASK_ID,
            response_usages=(("response-1", 10, 3),),
            history=history,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_archive(
                Path(temporary) / "fixture.tar.gz",
                [
                    (_task_log_member(1), _jsonl_bytes(task_log)),
                    (
                        _member(1, TASK_ID, _completion_name(1.0)),
                        _completion(
                            timestamp=1.0,
                            created=1_700_000_001,
                            response_id="response-1",
                            tool_call_id=tool_call_id,
                        ),
                    ),
                ],
            )
            trajectory = OpenHandsArchiveReader().read(path, _metadata())[0]

        types = _event_types(trajectory)
        self.assertEqual(types.count(EventType.TOOL_STARTED), 1)
        self.assertEqual(types.count(EventType.TOOL_FAILED), 1)
        self.assertNotIn(EventType.TOOL_COMPLETED, types)
        failed = next(
            event for event in trajectory.events if event.event_type == EventType.TOOL_FAILED
        )
        self.assertTrue(failed.payload["failure_observable"])
        self.assertEqual(failed.payload["failure_evidence"], "success_false")
        self.assertNotIn(
            "RAW_JSONL_TOOL_OUTPUT_SENTINEL",
            json.dumps([event.to_dict() for event in trajectory.events]),
        )

    def test_report_only_task_run_does_not_disappear_silently(self) -> None:
        report_only_task = "sympy__sympy-54321"
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_archive(
                Path(temporary) / "fixture.tar.gz",
                [
                    (_report_member(1, TASK_ID), _report(TASK_ID)),
                    (
                        _member(1, TASK_ID, _completion_name(1.0)),
                        _completion(
                            timestamp=1.0,
                            created=1_700_000_001,
                            response_id="response-1",
                        ),
                    ),
                    (
                        _report_member(2, report_only_task),
                        _report(report_only_task, resolved=False),
                    ),
                ],
            )
            with self.assertRaises(OpenHandsArchiveSchemaError):
                OpenHandsArchiveReader().read(path, _metadata())

    def test_unknown_completion_schema_fails_closed(self) -> None:
        cases: dict[str, dict[str, object]] = {}
        unexpected = _completion(
            timestamp=1.0,
            created=1_700_000_001,
            response_id="response-1",
        )
        unexpected["future_field"] = "unsupported"
        cases["unexpected root field"] = unexpected
        missing = _completion(
            timestamp=1.0,
            created=1_700_000_001,
            response_id="response-1",
        )
        del missing["kwargs"]
        cases["missing required root field"] = missing

        for label, payload in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                path = _write_archive(
                    Path(temporary) / "fixture.tar.gz",
                    [(_member(1, TASK_ID, _completion_name(1.0)), payload)],
                )
                with self.assertRaises(OpenHandsArchiveSchemaError):
                    OpenHandsArchiveReader().read(path, _metadata())

    def test_unknown_output_jsonl_schema_fails_closed(self) -> None:
        task_log = _task_log_line(TASK_ID)
        task_log["future_field"] = "unsupported"
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_archive(
                Path(temporary) / "fixture.tar.gz",
                [(_task_log_member(1), _jsonl_bytes(task_log))],
            )
            with self.assertRaises(OpenHandsArchiveSchemaError):
                OpenHandsArchiveReader().read(path, _metadata())

    def test_future_suffix_does_not_change_shared_prefix_features(self) -> None:
        first = _completion(
            timestamp=1.0,
            created=1_700_000_001,
            response_id="response-0001",
            input_tokens=10,
            output_tokens=3,
        )
        second = _completion(
            timestamp=2.0,
            created=1_700_000_002,
            response_id="response-0002",
            messages=[*_base_messages(), _historical_assistant()],
            input_tokens=20,
            output_tokens=5,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            short_path = _write_archive(
                root / "short" / "fixture.tar.gz",
                [(_member(1, TASK_ID, _completion_name(1.0)), first)],
            )
            long_path = _write_archive(
                root / "long" / "fixture.tar.gz",
                [
                    (_member(1, TASK_ID, _completion_name(1.0)), first),
                    (_member(1, TASK_ID, _completion_name(2.0)), second),
                ],
            )
            reader = OpenHandsArchiveReader()
            short = reader.read(short_path, _metadata())[0]
            long = reader.read(long_path, _metadata())[0]

        self.assertEqual(
            [event.to_dict() for event in short.events[:4]],
            [event.to_dict() for event in long.events[:4]],
        )
        short_features = replay_feature_snapshots(short.events)
        long_features = replay_feature_snapshots(long.events)
        self.assertEqual(short_features[0].feature_hash, long_features[0].feature_hash)
        self.assertEqual(short_features[0].values, long_features[0].values)

    def test_future_task_log_does_not_change_task_start_point(self) -> None:
        completion = _completion(
            timestamp=1.0,
            created=1_700_000_001,
            response_id="response-0001",
            input_tokens=10,
            output_tokens=3,
        )
        task_log = _task_log_line(
            TASK_ID,
            response_usages=(("response-0001", 10, 3),),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            without_log_path = _write_archive(
                root / "without-log" / "fixture.tar.gz",
                [(_member(1, TASK_ID, _completion_name(1.0)), completion)],
            )
            with_log_path = _write_archive(
                root / "with-log" / "fixture.tar.gz",
                [
                    (_task_log_member(1), _jsonl_bytes(task_log)),
                    (_member(1, TASK_ID, _completion_name(1.0)), completion),
                ],
            )
            reader = OpenHandsArchiveReader()
            without_log = reader.read(without_log_path, _metadata())[0]
            with_log = reader.read(with_log_path, _metadata())[0]

        self.assertEqual(
            without_log.events[0].to_dict(),
            with_log.events[0].to_dict(),
        )
        self.assertNotIn("task_usage_observable", without_log.events[0].payload)
        self.assertNotIn("task_usage_observable", with_log.events[0].payload)
        without_start = replay_feature_snapshots(
            without_log.events,
            include_task_started=True,
        )[0]
        with_start = replay_feature_snapshots(
            with_log.events,
            include_task_started=True,
        )[0]
        self.assertEqual(without_start.feature_hash, with_start.feature_hash)
        self.assertEqual(without_start.values, with_start.values)
        self.assertEqual(without_log.events[-1].payload["reason"], "logging_incomplete")
        self.assertEqual(with_log.events[-1].payload["reason"], "agent_finished")


if __name__ == "__main__":
    unittest.main()
