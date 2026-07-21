from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, Mapping, TextIO

from token_prediction.contracts import (
    CanonicalEvent,
    EventType,
    Observable,
    SourceCapabilities,
)
from token_prediction.trajectory import Trajectory


class BagenSwebenchSchemaError(ValueError):
    """Raised when a BAGEN mini-SWE-agent trajectory is not safely normalizable."""


@dataclass(frozen=True)
class BagenSwebenchMetadata:
    benchmark_id: str = "bagen-swebench"
    model_family: str | None = None
    provider: str | None = None
    run_identity: str | None = None
    instance_id: str | None = None
    condition_id: str | None = None
    use_provider_input_proxy: bool = True


@dataclass(frozen=True)
class _Usage:
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cached_input_tokens: int | None
    cache_creation_input_tokens: int | None
    cache_read_input_tokens: int | None
    reasoning_output_tokens: int | None
    total_source: str

    @property
    def complete(self) -> bool:
        return self.input_tokens is not None and self.output_tokens is not None

    def to_payload(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_input_tokens": self.cache_creation_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
            "total_source": self.total_source,
        }


@dataclass(frozen=True)
class _ToolCall:
    tool_call_id: str
    logical_call_id: str
    tool_name: str
    arguments_hash: str
    arguments_chars: int


class BagenSwebenchReader:
    """Stream one preserved mini-SWE-agent ``*.traj.json`` into canonical events.

    The parser reads one top-level message at a time.  It never calls
    ``Path.read_text`` or ``json.load`` and therefore does not retain an entire
    large Gemini trajectory in memory.

    Provider input usage is available only after a response.  When enabled it
    is retained only as explicitly post-response audit telemetry on the attempt
    terminal; ``request_tokens_local`` remains missing and the reader never
    advertises ``REQUEST_LOCAL_COUNT``.
    """

    source_id = "bagen_swebench_traj_v1"
    capabilities = SourceCapabilities(
        source_id=source_id,
        observables=frozenset(
            {
                Observable.TASK_USAGE,
                Observable.CALL_USAGE,
                Observable.ATTEMPT_USAGE,
                Observable.REQUEST_BOUNDARIES,
                Observable.TASK_TERMINATION,
                Observable.REQUEST_MESSAGES,
                Observable.TOOL_EVENTS,
            }
        ),
        source="declared",
    )

    def read(
        self,
        location: str | Path,
        metadata: BagenSwebenchMetadata | None = None,
    ) -> Trajectory:
        source = Path(location).resolve()
        if not source.is_file() or not source.name.endswith(".traj.json"):
            raise BagenSwebenchSchemaError("source must be one *.traj.json file")
        resolved = metadata or BagenSwebenchMetadata()
        instance_id = resolved.instance_id or _instance_id_from_path(source)
        model_family = resolved.model_family or _family_from_path(source)
        run_identity = resolved.run_identity or _source_identity(source)
        trajectory_semantic = {
            "source_id": self.source_id,
            "benchmark_id": resolved.benchmark_id,
            "instance_id": instance_id,
            "model_family": model_family,
            "run_identity": run_identity,
        }
        trajectory_id = (
            f"{resolved.benchmark_id}:trajectory:{instance_id}:{model_family}:"
            f"{_semantic_hash(trajectory_semantic)[:20]}"
        )
        source_sha256 = _file_sha256(source)

        try:
            with source.open("r", encoding="utf-8") as handle:
                parser = _JsonCursor(handle)
                raw = _read_top_level(parser)
                top_metadata = raw.metadata
                info = _require_mapping(top_metadata.get("info"), "top-level info")
                _validate_optional_identity(top_metadata, instance_id)
                identity = _condition_identity(info, model_family, resolved)
                condition_id = resolved.condition_id or (
                    f"condition:{_semantic_hash(identity)[:20]}"
                )
                normalizer = _Normalizer(
                    source=source,
                    source_sha256=source_sha256,
                    instance_id=instance_id,
                    trajectory_id=trajectory_id,
                    run_identity=run_identity,
                    model_family=model_family,
                    provider=str(identity["provider"]),
                    condition_id=condition_id,
                    info=info,
                    use_provider_input_proxy=resolved.use_provider_input_proxy,
                )
                for message_index, message in raw.messages:
                    normalizer.consume(message_index, message)
                trajectory = normalizer.finish()
                _validate_optional_identity(raw.trailing_metadata, instance_id)
                raw.validate_complete()
                return trajectory
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise BagenSwebenchSchemaError(f"cannot read BAGEN trajectory: {exc}") from exc

    def iter_directory(
        self,
        location: str | Path,
        metadata: BagenSwebenchMetadata | None = None,
    ) -> Iterator[Trajectory]:
        root = Path(location).resolve()
        if not root.is_dir():
            raise BagenSwebenchSchemaError("trajectory root must be a directory")
        seen: set[str] = set()
        paths = sorted(
            (path for path in root.rglob("*.traj.json") if path.is_file()),
            key=lambda item: item.as_posix(),
        )
        if not paths:
            raise BagenSwebenchSchemaError("trajectory root contains no *.traj.json files")
        for path in paths:
            trajectory = self.read(path, metadata)
            if trajectory.trajectory_id in seen:
                raise BagenSwebenchSchemaError(
                    f"duplicate normalized trajectory id for {path.name!r}"
                )
            seen.add(trajectory.trajectory_id)
            yield trajectory


@dataclass
class _RawTopLevel:
    metadata: dict[str, Any]
    messages: Iterator[tuple[int, Mapping[str, Any]]]
    trailing_metadata: dict[str, Any]
    _state: dict[str, Any]

    def validate_complete(self) -> None:
        if not self._state.get("completed"):
            raise BagenSwebenchSchemaError("messages iterator was not fully consumed")
        seen = set(self.metadata) | set(self.trailing_metadata) | {"messages"}
        required = {"info", "messages"}
        missing = required - seen
        if missing:
            raise BagenSwebenchSchemaError(
                f"trajectory is missing top-level fields: {', '.join(sorted(missing))}"
            )


def _read_top_level(parser: "_JsonCursor") -> _RawTopLevel:
    parser.expect("{")
    before: dict[str, Any] = {}
    after: dict[str, Any] = {}
    state: dict[str, Any] = {"completed": False}
    allowed = {"instance_id", "info", "messages", "trajectory_format"}

    first = True
    while True:
        marker = parser.peek_non_whitespace()
        if marker == "}":
            parser.expect("}")
            raise BagenSwebenchSchemaError("trajectory has no messages array")
        if not first:
            parser.expect(",")
        key = parser.read_value()
        if not isinstance(key, str) or key not in allowed:
            raise BagenSwebenchSchemaError("trajectory has an unsupported top-level field")
        if key in before:
            raise BagenSwebenchSchemaError(f"duplicate top-level field: {key}")
        parser.expect(":")
        if key == "messages":
            break
        before[key] = parser.read_value()
        first = False

    def iterate() -> Iterator[tuple[int, Mapping[str, Any]]]:
        parser.expect("[")
        index = 0
        first_message = True
        while parser.peek_non_whitespace() != "]":
            if not first_message:
                parser.expect(",")
            value = parser.read_value()
            if not isinstance(value, Mapping):
                raise BagenSwebenchSchemaError(f"message {index} is not an object")
            yield index, value
            index += 1
            first_message = False
        parser.expect("]")
        if index == 0:
            raise BagenSwebenchSchemaError("messages must be a non-empty array")

        while parser.peek_non_whitespace() != "}":
            parser.expect(",")
            trailing_key = parser.read_value()
            if not isinstance(trailing_key, str) or trailing_key not in allowed - {"messages"}:
                raise BagenSwebenchSchemaError(
                    "trajectory has an unsupported trailing top-level field"
                )
            if trailing_key in before or trailing_key in after:
                raise BagenSwebenchSchemaError(
                    f"duplicate top-level field: {trailing_key}"
                )
            parser.expect(":")
            after[trailing_key] = parser.read_value()
        parser.expect("}")
        parser.require_eof()
        state["completed"] = True

    return _RawTopLevel(before, iterate(), after, state)


class _Normalizer:
    def __init__(
        self,
        *,
        source: Path,
        source_sha256: str,
        instance_id: str,
        trajectory_id: str,
        run_identity: str,
        model_family: str,
        provider: str,
        condition_id: str,
        info: Mapping[str, Any],
        use_provider_input_proxy: bool,
    ) -> None:
        self.source = source
        self.instance_id = instance_id
        self.trajectory_id = trajectory_id
        self.run_identity = run_identity
        self.model_family = model_family
        self.provider = provider
        self.condition_id = condition_id
        self.info = dict(info)
        self.use_provider_input_proxy = use_provider_input_proxy
        self.events: list[CanonicalEvent] = []
        self.prefix_fingerprints: list[str] = []
        self.prefix_roles: Counter[str] = Counter()
        self.prefix_content_chars = 0
        self.pending_tools: dict[str, _ToolCall] = {}
        self.seen_tool_ids: set[str] = set()
        self.usages: list[_Usage] = []
        self.call_count = 0
        self.format_error_count = 0
        self.tool_terminal_count = 0
        self.tool_failure_count = 0
        self.seen_exit = False
        self.message_count = 0
        self.resolved_models: set[str] = set()

        config = _require_mapping(self.info.get("config"), "info.config")
        agent = _require_mapping(config.get("agent"), "info.config.agent")
        model = _require_mapping(config.get("model"), "info.config.model")
        self.agent_id = "mini-swe-agent"
        self.agent_type = _required_text(config, "agent_type", "info.config")
        self.mini_version = _required_text(self.info, "mini_version", "info")
        self.configured_model = _required_text(model, "model_name", "info.config.model")
        self.agent_config_hash = _semantic_hash(_agent_behavior(agent))
        self.behavior_config_hash = _semantic_hash(_behavior_config(self.info))

        self._append(
            EventType.TASK_STARTED,
            payload={
                "task_id": instance_id,
                "run_id": run_identity,
                "prediction_context_id": f"{instance_id}:initial",
                "condition_id": condition_id,
                "benchmark_id": "swe-bench",
                "model_family": model_family,
                "model_id": self.configured_model,
                "provider": provider,
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "agent_version": self.mini_version,
                "mini_version": self.mini_version,
                "agent_config_hash": self.agent_config_hash,
                "behavior_config_hash": self.behavior_config_hash,
                "source_file_sha256": source_sha256,
                "canonical_time_source": "synthetic_order_only",
                "task_tokens": None,
            },
            raw_ref=f"{source.name}#/info",
        )

    def consume(self, index: int, message_value: Mapping[str, Any]) -> None:
        if self.seen_exit:
            raise BagenSwebenchSchemaError("exit message must be final")
        message = dict(message_value)
        role = str(message.get("role") or "").strip()
        if role not in {"system", "user", "assistant", "tool", "exit"}:
            raise BagenSwebenchSchemaError(f"message {index} has unsupported role")
        if index == 0 and role != "system":
            raise BagenSwebenchSchemaError("first message must have role=system")
        if index == 1 and role != "user":
            raise BagenSwebenchSchemaError("second message must have role=user")
        if index > 1 and role == "system":
            raise BagenSwebenchSchemaError("system message may appear only first")

        if role == "assistant":
            self._consume_assistant(index, message)
        elif role == "tool":
            self._consume_tool(index, message)
        elif role == "exit":
            self._consume_exit(index, message)
        elif role == "user" and index > 1:
            self._consume_interruption(index, message)

        self._record_prefix(message)
        self.message_count += 1

    def finish(self) -> Trajectory:
        if not self.seen_exit:
            raise BagenSwebenchSchemaError("trajectory requires one final exit message")
        if self.pending_tools:
            raise BagenSwebenchSchemaError("trajectory has tool calls without results")
        model_stats = _require_mapping(self.info.get("model_stats"), "info.model_stats")
        _validate_exact_keys(model_stats, {"api_calls", "instance_cost"}, "info.model_stats")
        reported_calls = _non_negative_int(
            model_stats.get("api_calls"), "info.model_stats.api_calls"
        )
        if reported_calls != self.call_count:
            raise BagenSwebenchSchemaError(
                "info.model_stats.api_calls disagrees with evidenced API calls"
            )
        return Trajectory.from_events(self.events)

    def _consume_assistant(self, index: int, message: Mapping[str, Any]) -> None:
        allowed = {
            "annotations",
            "content",
            "extra",
            "function_call",
            "provider_specific_fields",
            "reasoning",
            "reasoning_content",
            "reasoning_details",
            "refusal",
            "role",
            "thinking_blocks",
            "tool_calls",
        }
        _validate_exact_keys(
            message,
            allowed,
            f"assistant message {index}",
            optional={
                "annotations",
                "function_call",
                "provider_specific_fields",
                "reasoning",
                "reasoning_content",
                "reasoning_details",
                "refusal",
                "thinking_blocks",
            },
        )
        if message.get("function_call") is not None:
            raise BagenSwebenchSchemaError("legacy assistant function_call is unsupported")
        extra = _require_mapping(message.get("extra"), f"assistant message {index}.extra")
        response = _require_mapping(
            extra.get("response"), f"assistant message {index}.extra.response"
        )
        usage = _usage_from_response(response, f"assistant message {index}")
        self.usages.append(usage)
        call_id = f"{self.trajectory_id}:call:{self.call_count}"
        attempt_id = f"{call_id}:attempt:0"
        provider_input_audit = (
            usage.input_tokens
            if self.use_provider_input_proxy and usage.input_tokens is not None
            else None
        )
        self._append_request(call_id, index)
        self._append(
            EventType.API_ATTEMPT_STARTED,
            logical_call_id=call_id,
            attempt_id=attempt_id,
            payload={
                "provider": self.provider,
                "configured_model": self.configured_model,
                "request_content_hash": self._prefix_hash(),
                "source_created": _optional_number(response.get("created")),
            },
            raw_ref=f"{self.source.name}#/messages/{index}/extra/response",
        )
        request_id = _required_text(response, "id", f"assistant message {index}.response")
        resolved_model = _required_text(
            response, "model", f"assistant message {index}.response"
        )
        self.resolved_models.add(resolved_model)
        finish_reason = _validate_chat_response(response, message, index)
        self._append(
            EventType.API_COMPLETED,
            logical_call_id=call_id,
            attempt_id=attempt_id,
            payload={
                "provider": self.provider,
                "model": resolved_model,
                "request_id": request_id,
                "finish_reason": finish_reason,
                "usage": usage.to_payload(),
                "source_timestamp": _optional_number(extra.get("timestamp")),
                "response_content_hash": _content_summary(message.get("content"))[0],
                "response_content_chars": _content_summary(message.get("content"))[1],
                "provider_input_tokens_post_response_audit": provider_input_audit,
                "provider_input_tokens_post_response_audit_source": (
                    "provider_response_usage"
                    if provider_input_audit is not None
                    else "missing"
                ),
            },
            raw_ref=f"{self.source.name}#/messages/{index}/extra/response",
        )
        self.call_count += 1
        for call_value in _tool_calls(message, index):
            if call_value.tool_call_id in self.seen_tool_ids:
                raise BagenSwebenchSchemaError("tool_call_id must be unique")
            self.seen_tool_ids.add(call_value.tool_call_id)
            self.pending_tools[call_value.tool_call_id] = _ToolCall(
                tool_call_id=call_value.tool_call_id,
                logical_call_id=call_id,
                tool_name=call_value.tool_name,
                arguments_hash=call_value.arguments_hash,
                arguments_chars=call_value.arguments_chars,
            )

    def _consume_interruption(self, index: int, message: Mapping[str, Any]) -> None:
        allowed = {"content", "extra", "role"}
        _validate_exact_keys(message, allowed, f"user interruption {index}")
        extra = _require_mapping(message.get("extra"), f"user interruption {index}.extra")
        interrupt_type = str(extra.get("interrupt_type") or "")
        if interrupt_type != "FormatError":
            raise BagenSwebenchSchemaError("unexpected mid-trajectory user message")
        _validate_exact_keys(
            extra,
            {"interrupt_type", "response", "timestamp"},
            f"user interruption {index}.extra",
            optional={"response", "timestamp"},
        )
        response_value = extra.get("response")
        response = (
            _require_mapping(response_value, f"user interruption {index}.extra.response")
            if response_value is not None
            else None
        )
        usage = (
            _usage_from_response(response, f"user interruption {index}")
            if response is not None
            else _Usage(None, None, None, None, None, None, None, "missing")
        )
        self.usages.append(usage)
        call_id = f"{self.trajectory_id}:call:{self.call_count}"
        attempt_id = f"{call_id}:attempt:0"
        provider_input_audit = (
            usage.input_tokens
            if self.use_provider_input_proxy and usage.input_tokens is not None
            else None
        )
        self._append_request(call_id, index)
        self._append(
            EventType.API_ATTEMPT_STARTED,
            logical_call_id=call_id,
            attempt_id=attempt_id,
            payload={
                "provider": self.provider,
                "configured_model": self.configured_model,
                "request_content_hash": self._prefix_hash(),
                "source_created": (
                    _optional_number(response.get("created")) if response else None
                ),
            },
            raw_ref=f"{self.source.name}#/messages/{index}",
        )
        content_hash, content_chars, _ = _content_summary(message.get("content"))
        request_id = str(response.get("id") or "") if response else ""
        resolved_model = str(response.get("model") or "") if response else ""
        if resolved_model:
            self.resolved_models.add(resolved_model)
        self._append(
            EventType.API_FAILED,
            logical_call_id=call_id,
            attempt_id=attempt_id,
            payload={
                "provider": self.provider,
                "model": resolved_model or None,
                "request_id": request_id or None,
                "usage": usage.to_payload() if response is not None else None,
                "error_type": interrupt_type,
                "error_content_hash": content_hash,
                "error_content_chars": content_chars,
                "status_code": None,
                "retryable": True,
                "source_timestamp": _optional_number(extra.get("timestamp")),
                "provider_input_tokens_post_response_audit": provider_input_audit,
                "provider_input_tokens_post_response_audit_source": (
                    "provider_response_usage"
                    if provider_input_audit is not None
                    else "missing"
                ),
            },
            raw_ref=f"{self.source.name}#/messages/{index}",
        )
        self.call_count += 1
        self.format_error_count += 1

    def _consume_tool(self, index: int, message: Mapping[str, Any]) -> None:
        allowed = {"content", "extra", "role", "tool_call_id"}
        _validate_exact_keys(message, allowed, f"tool message {index}")
        tool_call_id = _required_text(message, "tool_call_id", f"tool message {index}")
        tool = self.pending_tools.pop(tool_call_id, None)
        if tool is None:
            raise BagenSwebenchSchemaError("tool result does not match a pending tool call")
        extra = _require_mapping(message.get("extra"), f"tool message {index}.extra")
        allowed_extra = {
            "exception",
            "exception_info",
            "exception_type",
            "raw_output",
            "returncode",
            "timestamp",
        }
        if not set(extra) <= allowed_extra:
            raise BagenSwebenchSchemaError("tool result has unsupported extra fields")
        returncode = _optional_int(extra.get("returncode"), "tool returncode")
        exception_text = str(extra.get("exception_info") or extra.get("exception") or "")
        failed = (returncode is not None and returncode != 0) or bool(exception_text.strip())
        output_hash, output_chars, output_bytes = _content_summary(message.get("content"))
        self._append(
            EventType.TOOL_FAILED if failed else EventType.TOOL_COMPLETED,
            logical_call_id=tool.logical_call_id,
            payload={
                "tool_call_id": tool.tool_call_id,
                "tool_name": tool.tool_name,
                "arguments_hash": tool.arguments_hash,
                "arguments_chars": tool.arguments_chars,
                "output_content_hash": output_hash,
                "output_content_chars": output_chars,
                "output_content_bytes": output_bytes,
                "returncode": returncode,
                "exception_type": extra.get("exception_type"),
                "source_timestamp": _optional_number(extra.get("timestamp")),
            },
            raw_ref=f"{self.source.name}#/messages/{index}",
        )
        self.tool_terminal_count += 1
        self.tool_failure_count += int(failed)

    def _consume_exit(self, index: int, message: Mapping[str, Any]) -> None:
        allowed = {"content", "extra", "role"}
        _validate_exact_keys(message, allowed, f"exit message {index}")
        extra = _require_mapping(message.get("extra"), f"exit message {index}.extra")
        _validate_exact_keys(extra, {"exit_status", "submission"}, f"exit message {index}.extra")
        exit_status = _required_text(extra, "exit_status", f"exit message {index}.extra")
        info_status = _required_text(self.info, "exit_status", "info")
        if exit_status != info_status:
            raise BagenSwebenchSchemaError("top-level and exit-message status disagree")
        if _canonical_json(extra.get("submission")) != _canonical_json(
            self.info.get("submission")
        ):
            raise BagenSwebenchSchemaError("top-level and exit-message submission disagree")

        if self.pending_tools:
            if len(self.pending_tools) != 1:
                raise BagenSwebenchSchemaError("exit cannot resolve multiple pending tool calls")
            tool = self.pending_tools.pop(next(iter(self.pending_tools)))
            output_hash, output_chars, output_bytes = _content_summary(message.get("content"))
            tool_failed = exit_status != "Submitted"
            self._append(
                EventType.TOOL_FAILED if tool_failed else EventType.TOOL_COMPLETED,
                logical_call_id=tool.logical_call_id,
                payload={
                    "tool_call_id": tool.tool_call_id,
                    "tool_name": tool.tool_name,
                    "arguments_hash": tool.arguments_hash,
                    "arguments_chars": tool.arguments_chars,
                    "output_content_hash": output_hash,
                    "output_content_chars": output_chars,
                    "output_content_bytes": output_bytes,
                    "returncode": None,
                    "source_timestamp": None,
                    "terminal_intercept": True,
                },
                raw_ref=f"{self.source.name}#/messages/{index}",
            )
            self.tool_terminal_count += 1
            self.tool_failure_count += int(tool_failed)

        if exit_status == "Submitted":
            terminal_type = EventType.TASK_FINISHED
            outcome = "submitted"
            reason = "agent_finished"
        elif exit_status == "LimitsExceeded":
            terminal_type = EventType.TASK_ABORTED
            outcome = "limits_exceeded"
            reason = "max_turns"
        else:
            raise BagenSwebenchSchemaError(f"unsupported exit status: {exit_status!r}")
        aggregate = _aggregate_usage(self.usages)
        submission_hash, submission_chars, submission_bytes = _content_summary(
            extra.get("submission")
        )
        model_stats = _require_mapping(self.info.get("model_stats"), "info.model_stats")
        self._append(
            terminal_type,
            payload={
                "outcome": outcome,
                "reason": reason,
                "usage": aggregate.to_payload() if aggregate else None,
                "known_usage_attempts": sum(usage.complete for usage in self.usages),
                "missing_usage_attempts": sum(not usage.complete for usage in self.usages),
                "format_error_recovery_calls": self.format_error_count,
                "tool_terminal_count": self.tool_terminal_count,
                "tool_failure_count": self.tool_failure_count,
                "reported_api_calls": model_stats.get("api_calls"),
                "reported_instance_cost": model_stats.get("instance_cost"),
                "exit_status": exit_status,
                "submission_content_hash": submission_hash,
                "submission_content_chars": submission_chars,
                "submission_content_bytes": submission_bytes,
                "resolved_models": sorted(self.resolved_models),
            },
            raw_ref=f"{self.source.name}#/messages/{index}",
        )
        self.seen_exit = True

    def _append_request(self, call_id: str, index: int) -> None:
        self._append(
            EventType.REQUEST_BUILT,
            payload={
                "request_tokens_local": None,
                "request_token_count_source": "missing",
                "prediction_context_id": f"{self.instance_id}:call:{self.call_count}",
                "request_message_count": len(self.prefix_fingerprints),
                "request_content_chars": self.prefix_content_chars,
                "request_role_counts": dict(sorted(self.prefix_roles.items())),
                "request_content_hash": self._prefix_hash(),
            },
            logical_call_id=call_id,
            raw_ref=f"{self.source.name}#/messages/{index}",
        )

    def _record_prefix(self, message: Mapping[str, Any]) -> None:
        projection = _message_projection(message)
        self.prefix_fingerprints.append(_semantic_hash(projection))
        role = str(message.get("role") or "")
        self.prefix_roles[role] += 1
        self.prefix_content_chars += int(projection["content_chars"])

    def _prefix_hash(self) -> str:
        return _semantic_hash({"message_fingerprints": self.prefix_fingerprints})

    def _append(
        self,
        event_type: EventType,
        *,
        payload: Mapping[str, Any],
        raw_ref: str,
        logical_call_id: str | None = None,
        attempt_id: str | None = None,
    ) -> None:
        seq = len(self.events)
        occurred_at = (datetime(2000, 1, 1, tzinfo=UTC) + timedelta(microseconds=seq)).isoformat()
        self.events.append(
            CanonicalEvent.create(
                schema_version=1,
                event_id=f"{self.trajectory_id}:event:{seq}",
                trajectory_id=self.trajectory_id,
                event_seq=seq,
                event_type=event_type,
                occurred_at=occurred_at,
                payload=payload,
                logical_call_id=logical_call_id,
                attempt_id=attempt_id,
                raw_ref=raw_ref,
            )
        )


@dataclass(frozen=True)
class _ParsedToolCall:
    tool_call_id: str
    tool_name: str
    arguments_hash: str
    arguments_chars: int


def _tool_calls(message: Mapping[str, Any], index: int) -> tuple[_ParsedToolCall, ...]:
    value = message.get("tool_calls")
    if value is None:
        return ()
    if not isinstance(value, list):
        raise BagenSwebenchSchemaError(f"assistant message {index} tool_calls is not a list")
    result: list[_ParsedToolCall] = []
    for tool_index, raw_call in enumerate(value):
        call = _require_mapping(raw_call, f"assistant message {index} tool call {tool_index}")
        _validate_exact_keys(
            call,
            {"caller", "function", "id", "index", "type"},
            "assistant tool call",
            optional={"caller", "index"},
        )
        if call.get("type") != "function":
            raise BagenSwebenchSchemaError("unsupported assistant tool-call type")
        function = _require_mapping(call.get("function"), "assistant tool call function")
        _validate_exact_keys(function, {"arguments", "name"}, "assistant tool function")
        call_id = _required_text(call, "id", "assistant tool call")
        name = _required_text(function, "name", "assistant tool function")
        arguments = _required_text(function, "arguments", "assistant tool function")
        parsed_arguments = _strict_json_loads(
            arguments,
            context="tool-call arguments",
        )
        if not isinstance(parsed_arguments, Mapping):
            raise BagenSwebenchSchemaError("tool-call arguments must decode to an object")
        result.append(
            _ParsedToolCall(
                tool_call_id=call_id,
                tool_name=name,
                arguments_hash=hashlib.sha256(arguments.encode("utf-8")).hexdigest(),
                arguments_chars=len(arguments),
            )
        )
    return tuple(result)


def _validate_chat_response(
    response: Mapping[str, Any], message: Mapping[str, Any], index: int
) -> str | None:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise BagenSwebenchSchemaError(f"assistant message {index} requires one response choice")
    choice = _require_mapping(choices[0], f"assistant message {index} response choice")
    response_message = _require_mapping(
        choice.get("message"), f"assistant message {index} response choice message"
    )
    for key in ("role", "content", "tool_calls", "function_call"):
        if _canonical_json(response_message.get(key)) != _canonical_json(message.get(key)):
            raise BagenSwebenchSchemaError(
                f"assistant message {index} disagrees with embedded provider response"
            )
    finish_reason = choice.get("finish_reason")
    return str(finish_reason) if finish_reason is not None else None


def _usage_from_response(response: Mapping[str, Any], context: str) -> _Usage:
    raw = response.get("usage")
    if raw is None:
        return _Usage(None, None, None, None, None, None, None, "missing")
    usage = _require_mapping(raw, f"{context}.usage")
    input_tokens = _aliased_optional_int(usage, ("prompt_tokens", "input_tokens"), context)
    output_tokens = _aliased_optional_int(
        usage, ("completion_tokens", "output_tokens"), context
    )
    total_tokens = _aliased_optional_int(usage, ("total_tokens",), context)
    if input_tokens is not None and output_tokens is not None:
        accounted = input_tokens + output_tokens
        if total_tokens is not None and total_tokens != accounted:
            raise BagenSwebenchSchemaError(f"{context} reported token total is inconsistent")
        total_source = "reported" if total_tokens is not None else "derived_input_plus_output"
        total_tokens = accounted
    else:
        total_source = "reported_partial" if total_tokens is not None else "missing"

    prompt_details = usage.get("prompt_tokens_details")
    prompt_details = (
        _require_mapping(prompt_details, f"{context}.prompt_tokens_details")
        if prompt_details is not None
        else {}
    )
    completion_details = usage.get("completion_tokens_details") or usage.get(
        "output_tokens_details"
    )
    completion_details = (
        _require_mapping(completion_details, f"{context}.completion_tokens_details")
        if completion_details is not None
        else {}
    )
    cache_read = _first_optional_int(
        (
            usage.get("cache_read_input_tokens"),
            usage.get("cache_read_tokens"),
            prompt_details.get("cached_tokens"),
        ),
        f"{context}.cache_read_tokens",
    )
    cache_creation = _first_optional_int(
        (
            usage.get("cache_creation_input_tokens"),
            usage.get("cache_write_input_tokens"),
        ),
        f"{context}.cache_creation_tokens",
    )
    cached = _first_optional_int(
        (usage.get("cached_input_tokens"), cache_read), f"{context}.cached_input_tokens"
    )
    reasoning = _first_optional_int(
        (
            usage.get("reasoning_tokens"),
            completion_details.get("reasoning_tokens"),
        ),
        f"{context}.reasoning_tokens",
    )
    return _Usage(
        input_tokens,
        output_tokens,
        total_tokens,
        cached,
        cache_creation,
        cache_read,
        reasoning,
        total_source,
    )


def _aggregate_usage(usages: list[_Usage]) -> _Usage | None:
    if not usages or any(not usage.complete for usage in usages):
        return None

    def optional_sum(name: str) -> int | None:
        values = [getattr(usage, name) for usage in usages]
        return sum(int(value) for value in values if value is not None) if all(
            value is not None for value in values
        ) else None

    input_tokens = sum(int(usage.input_tokens or 0) for usage in usages)
    output_tokens = sum(int(usage.output_tokens or 0) for usage in usages)
    return _Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cached_input_tokens=optional_sum("cached_input_tokens"),
        cache_creation_input_tokens=optional_sum("cache_creation_input_tokens"),
        cache_read_input_tokens=optional_sum("cache_read_input_tokens"),
        reasoning_output_tokens=optional_sum("reasoning_output_tokens"),
        total_source="derived_complete_attempt_sum",
    )


def _condition_identity(
    info: Mapping[str, Any], model_family: str, metadata: BagenSwebenchMetadata
) -> dict[str, Any]:
    _validate_exact_keys(
        info,
        {
            "config",
            "exception_str",
            "exit_status",
            "mini_version",
            "model_stats",
            "submission",
            "traceback",
        },
        "info",
        optional={"exception_str", "traceback"},
    )
    config = _require_mapping(info.get("config"), "info.config")
    model = _require_mapping(config.get("model"), "info.config.model")
    configured_model = _required_text(model, "model_name", "info.config.model")
    provider = metadata.provider or _provider(model_family, configured_model)
    return {
        "provider": provider,
        "model": configured_model,
        "model_family": model_family,
        "agent_id": "mini-swe-agent",
        "agent_type": _required_text(config, "agent_type", "info.config"),
        "mini_version": _required_text(info, "mini_version", "info"),
        "behavior_config_hash": _semantic_hash(_behavior_config(info)),
    }


def _behavior_config(info: Mapping[str, Any]) -> dict[str, Any]:
    config = _require_mapping(info.get("config"), "info.config")
    _validate_exact_keys(
        config,
        {"agent", "agent_type", "environment", "environment_type", "model", "model_type"},
        "info.config",
    )
    agent = _require_mapping(config.get("agent"), "info.config.agent")
    model = _require_mapping(config.get("model"), "info.config.model")
    environment = _require_mapping(config.get("environment"), "info.config.environment")
    return {
        "agent": _agent_behavior(agent),
        "agent_type": _required_text(config, "agent_type", "info.config"),
        "model": _model_behavior(model),
        "model_type": _required_text(config, "model_type", "info.config"),
        "environment": _environment_behavior(environment),
        "environment_type": _required_text(config, "environment_type", "info.config"),
        "mini_version": _required_text(info, "mini_version", "info"),
    }


def _agent_behavior(agent: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"cost_limit", "instance_template", "output_path", "step_limit", "system_template"}
    _validate_exact_keys(agent, allowed, "info.config.agent")
    return {
        "system_template_hash": _text_hash(agent.get("system_template")),
        "instance_template_hash": _text_hash(agent.get("instance_template")),
        "step_limit": _optional_non_negative_int(agent.get("step_limit"), "agent step_limit"),
        "cost_limit": _optional_number(agent.get("cost_limit")),
    }


def _model_behavior(model: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "cost_tracking",
        "format_error_template",
        "litellm_model_registry",
        "model_kwargs",
        "model_name",
        "multimodal_regex",
        "observation_template",
        "set_cache_control",
    }
    _validate_exact_keys(
        model,
        allowed,
        "info.config.model",
        optional={
            "cost_tracking",
            "format_error_template",
            "litellm_model_registry",
            "multimodal_regex",
            "observation_template",
            "set_cache_control",
        },
    )
    if model.get("litellm_model_registry") is not None:
        raise BagenSwebenchSchemaError("non-default model registry is unsupported")
    kwargs = _require_mapping(model.get("model_kwargs"), "info.config.model.model_kwargs")
    allowed_kwargs = {
        "drop_params",
        "frequency_penalty",
        "max_completion_tokens",
        "max_tokens",
        "parallel_tool_calls",
        "presence_penalty",
        "output_config",
        "reasoning_effort",
        "response_format",
        "seed",
        "stop",
        "temperature",
        "thinking",
        "top_p",
    }
    if not set(kwargs) <= allowed_kwargs:
        raise BagenSwebenchSchemaError("model kwargs contain unsupported fields")
    return {
        "model_name": _required_text(model, "model_name", "info.config.model"),
        "model_kwargs": dict(sorted(kwargs.items())),
        "cost_tracking": model.get("cost_tracking"),
        "set_cache_control": model.get("set_cache_control"),
        "format_error_template_hash": _text_hash(model.get("format_error_template")),
        "observation_template_hash": _text_hash(model.get("observation_template")),
        "multimodal_regex_hash": _text_hash(model.get("multimodal_regex")),
    }


def _environment_behavior(environment: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "container_timeout",
        "cwd",
        "env",
        "executable",
        "forward_env",
        "image",
        "interpreter",
        "pull_timeout",
        "run_args",
        "timeout",
    }
    _validate_exact_keys(environment, allowed, "info.config.environment")
    env = _require_mapping(environment.get("env"), "info.config.environment.env")
    safe_env = {"LESS", "MANPAGER", "PAGER", "PIP_PROGRESS_BAR", "TQDM_DISABLE"}
    if not set(env) <= safe_env:
        raise BagenSwebenchSchemaError("environment contains unsupported variables")
    forward_env = environment.get("forward_env")
    if not isinstance(forward_env, list) or any(not isinstance(item, str) for item in forward_env):
        raise BagenSwebenchSchemaError("environment.forward_env must be a string list")
    return {
        "cwd": environment.get("cwd"),
        "env": dict(sorted(env.items())),
        "forward_env_names_hash": _semantic_hash({"names": sorted(forward_env)}),
        "timeout": environment.get("timeout"),
        "executable": environment.get("executable"),
        "run_args": environment.get("run_args"),
        "container_timeout": environment.get("container_timeout"),
        "pull_timeout": environment.get("pull_timeout"),
        "interpreter": environment.get("interpreter"),
    }


def _message_projection(message: Mapping[str, Any]) -> dict[str, Any]:
    content_hash, content_chars, content_bytes = _content_summary(message.get("content"))
    tool_calls = message.get("tool_calls")
    tool_projection: list[dict[str, Any]] = []
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if isinstance(call, Mapping):
                function = call.get("function")
                function = function if isinstance(function, Mapping) else {}
                arguments = str(function.get("arguments") or "")
                tool_projection.append(
                    {
                        "id": call.get("id"),
                        "name": function.get("name"),
                        "arguments_hash": hashlib.sha256(arguments.encode("utf-8")).hexdigest(),
                        "arguments_chars": len(arguments),
                    }
                )
    extra = message.get("extra")
    extra = extra if isinstance(extra, Mapping) else {}
    return {
        "role": message.get("role"),
        "content_hash": content_hash,
        "content_chars": content_chars,
        "content_bytes": content_bytes,
        "tool_call_id": message.get("tool_call_id"),
        "tool_calls": tool_projection,
        "interrupt_type": extra.get("interrupt_type"),
    }


def _content_summary(value: Any) -> tuple[str, int, int]:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        text = _canonical_json(value)
    encoded = text.encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), len(text), len(encoded)


def _validate_optional_identity(metadata: Mapping[str, Any], instance_id: str) -> None:
    if "instance_id" in metadata:
        raw_instance = str(metadata.get("instance_id") or "").strip()
        if not raw_instance or raw_instance != instance_id:
            raise BagenSwebenchSchemaError("top-level instance_id disagrees with source identity")
    if "trajectory_format" in metadata:
        value = str(metadata.get("trajectory_format") or "")
        if value != "mini-swe-agent-1.1":
            raise BagenSwebenchSchemaError("unsupported trajectory_format")


def _instance_id_from_path(path: Path) -> str:
    suffix = ".traj.json"
    instance_id = path.name[: -len(suffix)]
    if not instance_id:
        raise BagenSwebenchSchemaError("trajectory filename does not contain an instance_id")
    if path.parent.name and "__" in path.parent.name and path.parent.name != instance_id:
        raise BagenSwebenchSchemaError("trajectory directory and filename instance ids disagree")
    return instance_id


def _family_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        lowered = part.lower()
        marker = "swebench-origin-"
        if lowered.startswith(marker):
            family = lowered[len(marker) :]
            if family:
                return family
    raise BagenSwebenchSchemaError("model_family is not encoded in source path; pass metadata")


def _source_identity(path: Path) -> str:
    parts = path.parts
    for index, part in enumerate(parts):
        if part.lower().startswith("swebench-origin-"):
            return Path(*parts[index:]).as_posix()
    return path.name


def _provider(model_family: str, configured_model: str) -> str:
    lowered = f"{model_family}/{configured_model}".lower()
    if "claude" in lowered or lowered.startswith("anthropic/"):
        return "anthropic"
    if "gemini" in lowered:
        return "google"
    if "qwen" in lowered or "openrouter/" in lowered:
        return "openrouter"
    if "gpt" in lowered or "openai" in lowered:
        return "openai"
    raise BagenSwebenchSchemaError("provider cannot be inferred; pass metadata")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _text_hash(value: Any) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BagenSwebenchSchemaError(f"{context} must be an object")
    return value


def _required_text(value: Mapping[str, Any], key: str, context: str) -> str:
    text = str(value.get(key) or "").strip()
    if not text:
        raise BagenSwebenchSchemaError(f"{context}.{key} must be non-empty")
    return text


def _validate_exact_keys(
    value: Mapping[str, Any],
    allowed: set[str],
    context: str,
    *,
    optional: set[str] | None = None,
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise BagenSwebenchSchemaError(f"{context} contains unsupported fields")
    required = allowed - (optional or set())
    missing = required - set(value)
    if missing:
        raise BagenSwebenchSchemaError(f"{context} is missing required fields")


def _non_negative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BagenSwebenchSchemaError(f"{context} must be an integer")
    if value < 0:
        raise BagenSwebenchSchemaError(f"{context} must be non-negative")
    return value


def _optional_non_negative_int(value: Any, context: str) -> int | None:
    return None if value is None else _non_negative_int(value, context)


def _optional_int(value: Any, context: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise BagenSwebenchSchemaError(f"{context} must be an integer")
    return value


def _optional_number(value: Any) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BagenSwebenchSchemaError("numeric telemetry field has invalid type")
    if isinstance(value, float) and not math.isfinite(value):
        raise BagenSwebenchSchemaError("numeric telemetry field must be finite")
    return value


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise BagenSwebenchSchemaError(f"JSON contains a duplicate object key: {key!r}")
        value[key] = item
    return value


def _reject_non_finite_json(value: Any) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BagenSwebenchSchemaError("JSON contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_non_finite_json(item)
        return
    if isinstance(value, list):
        for item in value:
            _reject_non_finite_json(item)


def _strict_json_loads(value: str, *, context: str) -> Any:
    try:
        parsed = json.loads(
            value,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                BagenSwebenchSchemaError(
                    f"{context} contains a non-finite constant: {constant}"
                )
            ),
        )
    except BagenSwebenchSchemaError:
        raise
    except json.JSONDecodeError as exc:
        raise BagenSwebenchSchemaError(f"{context} is not valid JSON") from exc
    _reject_non_finite_json(parsed)
    return parsed


def _aliased_optional_int(
    value: Mapping[str, Any], aliases: tuple[str, ...], context: str
) -> int | None:
    observed = [value[key] for key in aliases if key in value and value[key] is not None]
    if not observed:
        return None
    parsed = [_non_negative_int(item, context) for item in observed]
    if len(set(parsed)) != 1:
        raise BagenSwebenchSchemaError(f"{context} has conflicting usage aliases")
    return parsed[0]


def _first_optional_int(values: tuple[Any, ...], context: str) -> int | None:
    observed = [_non_negative_int(value, context) for value in values if value is not None]
    if not observed:
        return None
    if len(set(observed)) != 1:
        raise BagenSwebenchSchemaError(f"{context} has conflicting values")
    return observed[0]


class _JsonCursor:
    """Small standard-library incremental JSON cursor for top-level array streaming."""

    def __init__(self, handle: TextIO, chunk_size: int = 1024 * 1024) -> None:
        self.handle = handle
        self.chunk_size = chunk_size
        self.buffer = ""
        self.position = 0
        self.eof = False

    def _fill(self) -> bool:
        if self.position < len(self.buffer):
            return True
        chunk = self.handle.read(self.chunk_size)
        self.buffer = chunk
        self.position = 0
        if not chunk:
            self.eof = True
            return False
        return True

    def _take(self) -> str:
        if not self._fill():
            raise BagenSwebenchSchemaError("unexpected end of JSON")
        value = self.buffer[self.position]
        self.position += 1
        return value

    def peek_non_whitespace(self) -> str:
        while True:
            if not self._fill():
                return ""
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer):
                return self.buffer[self.position]

    def expect(self, expected: str) -> None:
        observed = self.peek_non_whitespace()
        if observed != expected:
            raise BagenSwebenchSchemaError(
                f"expected JSON delimiter {expected!r}, found {observed!r}"
            )
        self.position += 1

    def read_value(self) -> Any:
        first = self.peek_non_whitespace()
        if not first:
            raise BagenSwebenchSchemaError("unexpected end of JSON value")
        if first in "{[":
            text = self._read_container()
        elif first == '"':
            text = self._read_string()
        else:
            text = self._read_scalar()
        return _strict_json_loads(text, context="trajectory JSON value")

    def _read_container(self) -> str:
        pieces: list[str] = []
        depth = 0
        in_string = False
        escaped = False
        segment_start = self.position
        while True:
            if not self._fill():
                raise BagenSwebenchSchemaError("unterminated JSON container")
            segment_start = self.position
            while self.position < len(self.buffer):
                char = self.buffer[self.position]
                self.position += 1
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char in "{[":
                    depth += 1
                elif char in "}]":
                    depth -= 1
                    if depth == 0:
                        pieces.append(self.buffer[segment_start : self.position])
                        return "".join(pieces)
                    if depth < 0:
                        raise BagenSwebenchSchemaError("unbalanced JSON container")
            pieces.append(self.buffer[segment_start : self.position])

    def _read_string(self) -> str:
        pieces: list[str] = []
        escaped = False
        started = False
        while True:
            if not self._fill():
                raise BagenSwebenchSchemaError("unterminated JSON string")
            segment_start = self.position
            while self.position < len(self.buffer):
                char = self.buffer[self.position]
                self.position += 1
                if not started:
                    if char != '"':
                        raise BagenSwebenchSchemaError("invalid JSON string")
                    started = True
                    continue
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    pieces.append(self.buffer[segment_start : self.position])
                    return "".join(pieces)
            pieces.append(self.buffer[segment_start : self.position])

    def _read_scalar(self) -> str:
        pieces: list[str] = []
        while True:
            if not self._fill():
                break
            segment_start = self.position
            while self.position < len(self.buffer):
                char = self.buffer[self.position]
                if char.isspace() or char in ",]}":
                    pieces.append(self.buffer[segment_start : self.position])
                    return "".join(pieces)
                self.position += 1
            pieces.append(self.buffer[segment_start : self.position])
        text = "".join(pieces)
        if not text:
            raise BagenSwebenchSchemaError("empty JSON scalar")
        return text

    def require_eof(self) -> None:
        if self.peek_non_whitespace():
            raise BagenSwebenchSchemaError("trailing content follows top-level JSON")
