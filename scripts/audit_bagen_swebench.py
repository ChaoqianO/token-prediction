from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from token_prediction.collection import BagenSwebenchReader, BagenSwebenchSchemaError
from token_prediction.contracts import CanonicalEvent, EventType
from token_prediction.dataset import (
    LabelStatus,
    PredictionPosition,
    PredictionTarget,
    build_supervised_dataset,
)
from token_prediction.trajectory import Trajectory


AUDIT_SCHEMA_VERSION = 1
_READ_BUFFER_BYTES = 1024 * 1024
_TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "reasoning_output_tokens",
)
_USAGE_FIELDS = frozenset((*_TOKEN_FIELDS, "total_source"))
_ATTEMPT_TOTAL_SOURCES = frozenset(
    {
        "derived_input_plus_output",
        "missing",
        "reported",
        "reported_partial",
    }
)
_SUPPORTED_EVENT_TYPES = frozenset(
    {
        EventType.TASK_STARTED,
        EventType.REQUEST_BUILT,
        EventType.API_ATTEMPT_STARTED,
        EventType.API_COMPLETED,
        EventType.API_FAILED,
        EventType.TOOL_COMPLETED,
        EventType.TOOL_FAILED,
        EventType.TASK_FINISHED,
        EventType.TASK_ABORTED,
    }
)


class BagenSwebenchAuditError(ValueError):
    """Raised when canonical evidence does not close under the audit contract."""


@dataclass(frozen=True)
class _UsageEvidence:
    values: Mapping[str, int | None]
    total_source: str
    mapping_present: bool

    @property
    def complete(self) -> bool:
        return self.values["input_tokens"] is not None and self.values["output_tokens"] is not None


class _AuditState:
    def __init__(self) -> None:
        self.raw_files: list[dict[str, Any]] = []
        self.task_ids: set[str] = set()
        self.trajectory_ids: set[str] = set()
        self.condition_tasks: dict[str, set[str]] = defaultdict(set)
        self.condition_trajectories: Counter[str] = Counter()
        self.trajectories_per_task: Counter[str] = Counter()

        self.call_count = 0
        self.attempt_count = 0
        self.complete_usage_attempts = 0
        self.missing_usage_attempts = 0
        self.retry_count = 0
        self.within_call_retry_count = 0
        self.format_error_recovery_calls = 0
        self.tool_event_count = 0
        self.tool_failure_count = 0
        self.tool_terminal_intercept_count = 0

        self.token_sums: Counter[str] = Counter()
        self.token_sum_coverage: Counter[str] = Counter()
        self.attempt_total_closure: Counter[str] = Counter()
        self.task_total_closure: Counter[str] = Counter()
        self.attempt_status: Counter[str] = Counter()
        self.attempt_usage_total_source: Counter[str] = Counter()
        self.api_error_type: Counter[str] = Counter()
        self.api_error_status_code: Counter[str] = Counter()
        self.api_error_retryable: Counter[str] = Counter()

        self.messages_per_trajectory: Counter[int] = Counter()
        self.message_roles: Counter[str] = Counter()
        self.exit_status: Counter[str] = Counter()
        self.task_terminal_event: Counter[str] = Counter()
        self.model_family: Counter[str] = Counter()
        self.provider: Counter[str] = Counter()
        self.configured_model: Counter[str] = Counter()
        self.resolved_model_trajectory_presence: Counter[str] = Counter()
        self.resolved_model_attempts: Counter[str] = Counter()
        self.agent_id: Counter[str] = Counter()
        self.agent_type: Counter[str] = Counter()
        self.agent_version: Counter[str] = Counter()
        self.mini_version: Counter[str] = Counter()
        self.format_errors_per_trajectory: Counter[int] = Counter()
        self.tool_events_per_trajectory: Counter[int] = Counter()
        self.tool_failures_per_trajectory: Counter[int] = Counter()
        self.tool_name: Counter[str] = Counter()

        self.dataset_row_count = 0
        self.dataset_counts: Counter[tuple[str, str, str]] = Counter()
        self.dataset_invalid_reasons: Counter[str] = Counter()
        self.canonical_hashes: dict[str, str] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_BUFFER_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_framed_hash(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def _canonical_trajectory_hash(trajectory: Trajectory) -> str:
    digest = hashlib.sha256(b"bagen-swebench-canonical-trajectory-v1\0")
    for event in trajectory.events:
        encoded = json.dumps(
            event.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        _update_framed_hash(digest, encoded)
    return digest.hexdigest()


def _canonical_family_hash(hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256(b"bagen-swebench-canonical-family-v1\0")
    for relative_path in sorted(hashes):
        _update_framed_hash(digest, relative_path.encode("utf-8"))
        _update_framed_hash(digest, bytes.fromhex(hashes[relative_path]))
    return digest.hexdigest()


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BagenSwebenchAuditError(f"{context} must be an object")
    return value


def _require_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BagenSwebenchAuditError(f"{context} must be a non-empty string")
    return value


def _require_non_negative_int(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise BagenSwebenchAuditError(f"{context} must be a non-negative integer")
    return value


def _optional_non_negative_int(value: Any, context: str) -> int | None:
    if value is None:
        return None
    return _require_non_negative_int(value, context)


def _require_string_list(value: Any, context: str) -> list[str]:
    if not isinstance(value, list):
        raise BagenSwebenchAuditError(f"{context} must be a list")
    result = [_require_text(item, f"{context} item") for item in value]
    if result != sorted(set(result)):
        raise BagenSwebenchAuditError(f"{context} must be sorted and unique")
    return result


def _usage_evidence(
    value: Any,
    context: str,
    *,
    task_aggregate: bool = False,
) -> _UsageEvidence:
    if value is None:
        return _UsageEvidence(
            values={field: None for field in _TOKEN_FIELDS},
            total_source="missing",
            mapping_present=False,
        )
    usage = _require_mapping(value, context)
    if set(usage) != _USAGE_FIELDS:
        missing = sorted(_USAGE_FIELDS - set(usage))
        unexpected = sorted(set(usage) - _USAGE_FIELDS)
        details: list[str] = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected {', '.join(unexpected)}")
        raise BagenSwebenchAuditError(f"{context} has unsupported fields: {'; '.join(details)}")

    values = {
        field: _optional_non_negative_int(usage.get(field), f"{context}.{field}")
        for field in _TOKEN_FIELDS
    }
    total_source = _require_text(usage.get("total_source"), f"{context}.total_source")
    if total_source not in _ATTEMPT_TOTAL_SOURCES | {"derived_complete_attempt_sum"}:
        raise BagenSwebenchAuditError(f"{context}.total_source is unsupported: {total_source!r}")
    if values["cache_write_input_tokens"] != values["cache_creation_input_tokens"]:
        raise BagenSwebenchAuditError(
            f"{context} cache_write_input_tokens and cache_creation_input_tokens disagree"
        )

    evidence = _UsageEvidence(values, total_source, True)
    if evidence.complete:
        accounted = int(values["input_tokens"] or 0) + int(values["output_tokens"] or 0)
        if values["total_tokens"] != accounted:
            raise BagenSwebenchAuditError(
                f"{context} total_tokens does not equal input_tokens + output_tokens"
            )
        complete_sources = {"derived_input_plus_output", "reported"}
        if task_aggregate:
            complete_sources.add("derived_complete_attempt_sum")
        if total_source not in complete_sources:
            raise BagenSwebenchAuditError(
                f"{context} has an invalid total_source for complete usage"
            )
    elif total_source not in {"missing", "reported_partial"}:
        raise BagenSwebenchAuditError(f"{context} has an invalid total_source for incomplete usage")
    return evidence


def _event_call_id(event: CanonicalEvent, context: str) -> str:
    return _require_text(event.logical_call_id, f"{context}.logical_call_id")


def _event_attempt_key(event: CanonicalEvent, context: str) -> tuple[str, str]:
    return (
        _event_call_id(event, context),
        _require_text(event.attempt_id, f"{context}.attempt_id"),
    )


def _raw_file_evidence(root: Path, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BagenSwebenchAuditError(f"trajectory path is not a file: {path}")
    relative_path = path.relative_to(root).as_posix()
    return {
        "path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _trajectory_paths(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*.traj.json") if path.is_file()),
        key=lambda item: item.as_posix(),
    )


def _paired_trajectories(root: Path, paths: Sequence[Path]) -> Iterator[tuple[Path, Trajectory]]:
    trajectories = iter(BagenSwebenchReader().iter_directory(root))
    for path in paths:
        try:
            trajectory = next(trajectories)
        except StopIteration as exc:
            raise BagenSwebenchAuditError(
                "reader returned fewer trajectories than the raw file inventory"
            ) from exc
        yield path, trajectory
    try:
        next(trajectories)
    except StopIteration:
        return
    raise BagenSwebenchAuditError("reader returned more trajectories than the raw file inventory")


def _expected_task_usage(attempt_usages: Sequence[_UsageEvidence]) -> dict[str, int | None]:
    expected: dict[str, int | None] = {}
    for field in _TOKEN_FIELDS:
        values = [usage.values[field] for usage in attempt_usages]
        expected[field] = (
            sum(int(value) for value in values if value is not None)
            if all(value is not None for value in values)
            else None
        )
    return expected


def _verify_task_usage(
    task_usage_value: Any,
    attempt_usages: Sequence[_UsageEvidence],
    context: str,
) -> str:
    if not attempt_usages:
        if task_usage_value is not None:
            raise BagenSwebenchAuditError(f"{context} must be null when there are no attempts")
        return "no_attempts"
    if any(not usage.complete for usage in attempt_usages):
        if task_usage_value is not None:
            raise BagenSwebenchAuditError(
                f"{context} must be null when any attempt usage is incomplete"
            )
        return "missing_attempt_usage"

    task_usage = _usage_evidence(task_usage_value, context, task_aggregate=True)
    if not task_usage.mapping_present or not task_usage.complete:
        raise BagenSwebenchAuditError(f"{context} must contain complete aggregate usage")
    if task_usage.total_source != "derived_complete_attempt_sum":
        raise BagenSwebenchAuditError(
            f"{context}.total_source must be 'derived_complete_attempt_sum'"
        )
    expected = _expected_task_usage(attempt_usages)
    if dict(task_usage.values) != expected:
        raise BagenSwebenchAuditError(
            f"{context} does not equal the field-wise complete attempt sum"
        )
    return "closed"


def _dataset_counts(
    trajectory: Trajectory,
) -> tuple[int, Counter[tuple[str, str, str]], Counter[str]]:
    dataset = build_supervised_dataset((trajectory,))
    counts: Counter[tuple[str, str, str]] = Counter()
    invalid_reasons: Counter[str] = Counter()
    for row in dataset.rows:
        if row.point.trajectory_id != trajectory.trajectory_id:
            raise BagenSwebenchAuditError("dataset row changed trajectory identity")
        key = (row.point.position.value, row.point.target.value, row.status.value)
        counts[key] += 1
        if row.invalid_reason:
            invalid_reasons[row.invalid_reason] += 1
    return len(dataset.rows), counts, invalid_reasons


def _audit_trajectory(
    root: Path,
    path: Path,
    trajectory: Trajectory,
    state: _AuditState,
) -> None:
    raw = _raw_file_evidence(root, path)
    relative_path = str(raw["path"])
    if relative_path in state.canonical_hashes:
        raise BagenSwebenchAuditError(f"duplicate raw path: {relative_path}")
    if trajectory.trajectory_id in state.trajectory_ids:
        raise BagenSwebenchAuditError(
            f"duplicate canonical trajectory id: {trajectory.trajectory_id}"
        )

    events = trajectory.events
    if not events:
        raise BagenSwebenchAuditError(f"{relative_path} has no canonical events")
    for event in events:
        if event.schema_version != 1:
            raise BagenSwebenchAuditError(
                f"{relative_path} has unsupported canonical schema version"
            )
        if event.event_type not in _SUPPORTED_EVENT_TYPES:
            raise BagenSwebenchAuditError(
                f"{relative_path} has unsupported event type: {event.event_type.value}"
            )
    if events[0].event_type != EventType.TASK_STARTED:
        raise BagenSwebenchAuditError(f"{relative_path} does not start with task_started")
    if events[-1].event_type not in {EventType.TASK_FINISHED, EventType.TASK_ABORTED}:
        raise BagenSwebenchAuditError(
            f"{relative_path} does not finish with a supported task terminal"
        )

    started = events[0].payload
    finished = events[-1].payload
    task_id = _require_text(started.get("task_id"), f"{relative_path}.task_id")
    condition_id = _require_text(started.get("condition_id"), f"{relative_path}.condition_id")
    if task_id != trajectory.task_id or condition_id != trajectory.condition_id:
        raise BagenSwebenchAuditError(f"{relative_path} canonical identity fields disagree")
    if started.get("source_file_sha256") != raw["sha256"]:
        raise BagenSwebenchAuditError(f"{relative_path} source SHA-256 does not close")
    if started.get("benchmark_id") != "swe-bench":
        raise BagenSwebenchAuditError(f"{relative_path} has an unsupported benchmark_id")
    if started.get("canonical_time_source") != "synthetic_order_only":
        raise BagenSwebenchAuditError(f"{relative_path} has an unsupported time source")

    model_family = _require_text(started.get("model_family"), f"{relative_path}.model_family")
    provider = _require_text(started.get("provider"), f"{relative_path}.provider")
    configured_model = _require_text(started.get("model_id"), f"{relative_path}.model_id")
    agent_id = _require_text(started.get("agent_id"), f"{relative_path}.agent_id")
    agent_type = _require_text(started.get("agent_type"), f"{relative_path}.agent_type")
    agent_version = _require_text(started.get("agent_version"), f"{relative_path}.agent_version")
    mini_version = _require_text(started.get("mini_version"), f"{relative_path}.mini_version")
    if agent_version != mini_version:
        raise BagenSwebenchAuditError(f"{relative_path} agent and mini versions disagree")

    requests: dict[str, CanonicalEvent] = {}
    attempt_starts: dict[tuple[str, str], CanonicalEvent] = {}
    attempt_terminals: dict[tuple[str, str], CanonicalEvent] = {}
    tool_events: list[CanonicalEvent] = []
    for event in events:
        context = f"{relative_path} event {event.event_seq}"
        if event.event_type == EventType.REQUEST_BUILT:
            call_id = _event_call_id(event, context)
            if call_id in requests:
                raise BagenSwebenchAuditError(f"{context} duplicates a request call id")
            requests[call_id] = event
        elif event.event_type == EventType.API_ATTEMPT_STARTED:
            key = _event_attempt_key(event, context)
            if key in attempt_starts:
                raise BagenSwebenchAuditError(f"{context} duplicates an attempt start")
            attempt_starts[key] = event
        elif event.event_type in {EventType.API_COMPLETED, EventType.API_FAILED}:
            key = _event_attempt_key(event, context)
            if key in attempt_terminals:
                raise BagenSwebenchAuditError(f"{context} duplicates an attempt terminal")
            attempt_terminals[key] = event
        elif event.event_type in {EventType.TOOL_COMPLETED, EventType.TOOL_FAILED}:
            tool_events.append(event)

    if set(attempt_starts) != set(attempt_terminals):
        raise BagenSwebenchAuditError(f"{relative_path} attempt terminal closure failed")
    attempts_per_call: Counter[str] = Counter(call_id for call_id, _ in attempt_starts)
    if set(requests) != set(attempts_per_call):
        raise BagenSwebenchAuditError(f"{relative_path} request/attempt call closure failed")
    if any(count != 1 for count in attempts_per_call.values()):
        raise BagenSwebenchAuditError(
            f"{relative_path} has unsupported multi-attempt logical calls"
        )

    local_token_sums: Counter[str] = Counter()
    local_token_coverage: Counter[str] = Counter()
    local_attempt_status: Counter[str] = Counter()
    local_total_source: Counter[str] = Counter()
    local_error_type: Counter[str] = Counter()
    local_error_status_code: Counter[str] = Counter()
    local_error_retryable: Counter[str] = Counter()
    attempt_usages: list[_UsageEvidence] = []
    resolved_model_attempts: Counter[str] = Counter()
    format_error_count = 0
    complete_usage_count = 0
    missing_usage_count = 0
    for key in sorted(attempt_terminals):
        terminal = attempt_terminals[key]
        context = f"{relative_path} attempt {key[1]} usage"
        usage = _usage_evidence(terminal.payload.get("usage"), context)
        attempt_usages.append(usage)
        local_total_source[usage.total_source] += 1
        if usage.complete:
            complete_usage_count += 1
            state.attempt_total_closure["closed"] += 1
        else:
            missing_usage_count += 1
            state.attempt_total_closure["unavailable"] += 1
        for field, value in usage.values.items():
            if value is not None:
                local_token_sums[field] += value
                local_token_coverage[field] += 1

        if terminal.event_type == EventType.API_COMPLETED:
            local_attempt_status["completed"] += 1
            resolved_model = _require_text(
                terminal.payload.get("model"), f"{relative_path} completed attempt model"
            )
            resolved_model_attempts[resolved_model] += 1
        else:
            local_attempt_status["failed"] += 1
            error_type = _require_text(
                terminal.payload.get("error_type"),
                f"{relative_path} failed attempt error_type",
            )
            if error_type != "FormatError":
                raise BagenSwebenchAuditError(
                    f"{relative_path} has an unsupported API failure type"
                )
            retryable = terminal.payload.get("retryable")
            if not isinstance(retryable, bool):
                raise BagenSwebenchAuditError(
                    f"{relative_path} failed attempt retryable must be boolean"
                )
            if not retryable:
                raise BagenSwebenchAuditError(
                    f"{relative_path} FormatError must be marked retryable"
                )
            status_code = terminal.payload.get("status_code")
            status_code_key = (
                "missing"
                if status_code is None
                else str(
                    _require_non_negative_int(
                        status_code,
                        f"{relative_path} failed attempt status_code",
                    )
                )
            )
            local_error_type[error_type] += 1
            local_error_status_code[status_code_key] += 1
            local_error_retryable[str(retryable).lower()] += 1
            format_error_count += 1
            failed_model = terminal.payload.get("model")
            if failed_model is not None:
                resolved_model_attempts[
                    _require_text(failed_model, f"{relative_path} failed attempt model")
                ] += 1

    tool_failure_count = 0
    tool_terminal_intercept_count = 0
    regular_tool_message_count = 0
    local_tool_names: Counter[str] = Counter()
    for tool_event in tool_events:
        context = f"{relative_path} tool event {tool_event.event_seq}"
        call_id = _event_call_id(tool_event, context)
        if call_id not in requests:
            raise BagenSwebenchAuditError(f"{context} has no matching request")
        tool_name = _require_text(tool_event.payload.get("tool_name"), f"{context}.tool_name")
        local_tool_names[tool_name] += 1
        if tool_event.event_type == EventType.TOOL_FAILED:
            tool_failure_count += 1
        if "terminal_intercept" in tool_event.payload:
            if tool_event.payload.get("terminal_intercept") is not True:
                raise BagenSwebenchAuditError(
                    f"{context}.terminal_intercept must be true when present"
                )
            tool_terminal_intercept_count += 1
        else:
            regular_tool_message_count += 1

    exit_status = _require_text(finished.get("exit_status"), f"{relative_path}.exit_status")
    expected_terminal = {
        "Submitted": (EventType.TASK_FINISHED, "submitted", "agent_finished"),
        "LimitsExceeded": (EventType.TASK_ABORTED, "limits_exceeded", "max_turns"),
    }.get(exit_status)
    if expected_terminal is None:
        raise BagenSwebenchAuditError(f"{relative_path} has an unsupported exit status")
    terminal_type, expected_outcome, expected_reason = expected_terminal
    if events[-1].event_type != terminal_type:
        raise BagenSwebenchAuditError(
            f"{relative_path} exit status and canonical terminal event disagree"
        )
    if finished.get("outcome") != expected_outcome or finished.get("reason") != expected_reason:
        raise BagenSwebenchAuditError(f"{relative_path} exit status and terminal outcome disagree")
    if (
        _require_non_negative_int(
            finished.get("known_usage_attempts"), f"{relative_path}.known_usage_attempts"
        )
        != complete_usage_count
    ):
        raise BagenSwebenchAuditError(f"{relative_path} known usage count does not close")
    if (
        _require_non_negative_int(
            finished.get("missing_usage_attempts"), f"{relative_path}.missing_usage_attempts"
        )
        != missing_usage_count
    ):
        raise BagenSwebenchAuditError(f"{relative_path} missing usage count does not close")
    if (
        _require_non_negative_int(
            finished.get("format_error_recovery_calls"),
            f"{relative_path}.format_error_recovery_calls",
        )
        != format_error_count
    ):
        raise BagenSwebenchAuditError(f"{relative_path} FormatError count does not close")
    if _require_non_negative_int(
        finished.get("tool_terminal_count"), f"{relative_path}.tool_terminal_count"
    ) != len(tool_events):
        raise BagenSwebenchAuditError(f"{relative_path} tool event count does not close")
    if (
        _require_non_negative_int(
            finished.get("tool_failure_count"), f"{relative_path}.tool_failure_count"
        )
        != tool_failure_count
    ):
        raise BagenSwebenchAuditError(f"{relative_path} tool failure count does not close")
    if _require_non_negative_int(
        finished.get("reported_api_calls"), f"{relative_path}.reported_api_calls"
    ) != len(requests):
        raise BagenSwebenchAuditError(f"{relative_path} reported API calls do not close")

    resolved_models = _require_string_list(
        finished.get("resolved_models"), f"{relative_path}.resolved_models"
    )
    if resolved_models != sorted(resolved_model_attempts):
        raise BagenSwebenchAuditError(f"{relative_path} resolved model set does not close")
    task_closure = _verify_task_usage(
        finished.get("usage"), attempt_usages, f"{relative_path}.task_usage"
    )

    assistant_message_count = local_attempt_status["completed"]
    message_role_counts = {
        "assistant": assistant_message_count,
        "exit": 1,
        "system": 1,
        "tool": regular_tool_message_count,
        "user": 1 + format_error_count,
    }
    message_count = sum(message_role_counts.values())
    dataset_row_count, dataset_counts, dataset_invalid_reasons = _dataset_counts(trajectory)
    canonical_hash = _canonical_trajectory_hash(trajectory)

    state.task_ids.add(task_id)
    state.trajectory_ids.add(trajectory.trajectory_id)
    state.condition_tasks[condition_id].add(task_id)
    state.condition_trajectories[condition_id] += 1
    state.trajectories_per_task[task_id] += 1
    state.call_count += len(requests)
    state.attempt_count += len(attempt_starts)
    state.complete_usage_attempts += complete_usage_count
    state.missing_usage_attempts += missing_usage_count
    state.retry_count += format_error_count
    state.within_call_retry_count += sum(max(0, count - 1) for count in attempts_per_call.values())
    state.format_error_recovery_calls += format_error_count
    state.tool_event_count += len(tool_events)
    state.tool_failure_count += tool_failure_count
    state.tool_terminal_intercept_count += tool_terminal_intercept_count
    state.token_sums.update(local_token_sums)
    state.token_sum_coverage.update(local_token_coverage)
    state.task_total_closure[task_closure] += 1
    state.attempt_status.update(local_attempt_status)
    state.attempt_usage_total_source.update(local_total_source)
    state.api_error_type.update(local_error_type)
    state.api_error_status_code.update(local_error_status_code)
    state.api_error_retryable.update(local_error_retryable)
    state.messages_per_trajectory[message_count] += 1
    state.message_roles.update(message_role_counts)
    state.exit_status[exit_status] += 1
    state.task_terminal_event[terminal_type.value] += 1
    state.model_family[model_family] += 1
    state.provider[provider] += 1
    state.configured_model[configured_model] += 1
    for resolved_model in resolved_models:
        state.resolved_model_trajectory_presence[resolved_model] += 1
    state.resolved_model_attempts.update(resolved_model_attempts)
    state.agent_id[agent_id] += 1
    state.agent_type[agent_type] += 1
    state.agent_version[agent_version] += 1
    state.mini_version[mini_version] += 1
    state.format_errors_per_trajectory[format_error_count] += 1
    state.tool_events_per_trajectory[len(tool_events)] += 1
    state.tool_failures_per_trajectory[tool_failure_count] += 1
    state.tool_name.update(local_tool_names)
    state.dataset_row_count += dataset_row_count
    state.dataset_counts.update(dataset_counts)
    state.dataset_invalid_reasons.update(dataset_invalid_reasons)
    state.canonical_hashes[relative_path] = canonical_hash

    state.raw_files.append(
        {
            **raw,
            "task_id": task_id,
            "trajectory_id": trajectory.trajectory_id,
            "condition_id": condition_id,
            "canonical_content_sha256": canonical_hash,
            "canonical_rerun_consistent": True,
            "message_count": message_count,
            "message_role_counts": message_role_counts,
            "call_count": len(requests),
            "attempt_count": len(attempt_starts),
            "complete_usage_attempts": complete_usage_count,
            "missing_usage_attempts": missing_usage_count,
            "retry_count": format_error_count,
            "within_call_retry_count": sum(
                max(0, count - 1) for count in attempts_per_call.values()
            ),
            "format_error_recovery_calls": format_error_count,
            "error_type_counts": _sorted_counts(local_error_type),
            "error_status_code_counts": _sorted_counts(local_error_status_code),
            "error_retryable_counts": _sorted_counts(local_error_retryable),
            "tool_event_count": len(tool_events),
            "tool_failure_count": tool_failure_count,
            "token_sums": _token_counter(local_token_sums),
            "token_sum_coverage_attempts": _token_counter(local_token_coverage),
            "attempt_total_closure": {
                "closed": complete_usage_count,
                "unavailable": missing_usage_count,
                "all_known_totals_match": True,
            },
            "task_total_closure": task_closure,
            "dataset_row_count": dataset_row_count,
            "exit_status": exit_status,
            "task_terminal_event": terminal_type.value,
            "model_family": model_family,
            "provider": provider,
            "configured_model": configured_model,
            "resolved_models": resolved_models,
            "agent_id": agent_id,
            "agent_type": agent_type,
            "agent_version": agent_version,
            "mini_version": mini_version,
        }
    )


def _sorted_counts(counter: Mapping[Any, int]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter, key=lambda item: str(item))}


def _token_counter(counter: Mapping[str, int]) -> dict[str, int]:
    return {field: int(counter.get(field, 0)) for field in _TOKEN_FIELDS}


def _status_counts(
    counts: Mapping[tuple[str, str, str], int],
    *,
    position: PredictionPosition | None = None,
    target: PredictionTarget | None = None,
) -> dict[str, int]:
    result: dict[str, int] = {}
    for status in LabelStatus:
        total = 0
        for (row_position, row_target, row_status), count in counts.items():
            if position is not None and row_position != position.value:
                continue
            if target is not None and row_target != target.value:
                continue
            if row_status == status.value:
                total += count
        result[status.value] = total
    return result


def _dataset_summary(state: _AuditState) -> dict[str, Any]:
    by_position: dict[str, Any] = {}
    for position in PredictionPosition:
        status_counts = _status_counts(state.dataset_counts, position=position)
        by_position[position.value] = {
            "row_count": sum(status_counts.values()),
            "status_counts": status_counts,
        }

    by_target: dict[str, Any] = {}
    for target in PredictionTarget:
        status_counts = _status_counts(state.dataset_counts, target=target)
        by_target[target.value] = {
            "row_count": sum(status_counts.values()),
            "status_counts": status_counts,
        }

    by_position_target: list[dict[str, Any]] = []
    for position in PredictionPosition:
        for target in PredictionTarget:
            status_counts = _status_counts(
                state.dataset_counts,
                position=position,
                target=target,
            )
            by_position_target.append(
                {
                    "position": position.value,
                    "target": target.value,
                    "row_count": sum(status_counts.values()),
                    "status_counts": status_counts,
                }
            )

    actual_positions = sorted({position for position, _, _ in state.dataset_counts})
    actual_targets = sorted({target for _, target, _ in state.dataset_counts})
    return {
        "row_count": state.dataset_row_count,
        "supported_positions": actual_positions,
        "supported_targets": actual_targets,
        "status_counts": _status_counts(state.dataset_counts),
        "by_position": by_position,
        "by_target": by_target,
        "by_position_target": by_position_target,
        "invalid_reason_counts": _sorted_counts(state.dataset_invalid_reasons),
    }


def _rerun_canonical_hashes(root: Path, paths: Sequence[Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path, trajectory in _paired_trajectories(root, paths):
        relative_path = path.relative_to(root).as_posix()
        hashes[relative_path] = _canonical_trajectory_hash(trajectory)
    return hashes


def build_audit(root: Path) -> dict[str, Any]:
    resolved_root = root.resolve()
    if not resolved_root.is_dir():
        raise BagenSwebenchAuditError("trajectory root must be a directory")
    paths = _trajectory_paths(resolved_root)
    state = _AuditState()
    trajectories: list[Trajectory] = []
    for path, trajectory in _paired_trajectories(resolved_root, paths):
        _audit_trajectory(resolved_root, path, trajectory, state)
        trajectories.append(trajectory)

    if len(state.model_family) != 1:
        raise BagenSwebenchAuditError("family audit requires exactly one canonical model_family")
    rerun_hashes = _rerun_canonical_hashes(resolved_root, paths)
    if rerun_hashes != state.canonical_hashes:
        raise BagenSwebenchAuditError("canonical content hashes changed on reader rerun")

    first_family_hash = _canonical_family_hash(state.canonical_hashes)
    rerun_family_hash = _canonical_family_hash(rerun_hashes)
    if first_family_hash != rerun_family_hash:
        raise BagenSwebenchAuditError("canonical family hash changed on reader rerun")

    condition_task_counts = {
        condition_id: {
            "task_count": len(state.condition_tasks[condition_id]),
            "trajectory_count": state.condition_trajectories[condition_id],
        }
        for condition_id in sorted(state.condition_tasks)
    }
    raw_files = sorted(state.raw_files, key=lambda item: str(item["path"]))
    dataset_summary = _dataset_summary(state)
    dataset_summary["dataset_id"] = build_supervised_dataset(trajectories).dataset_id
    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "source_id": BagenSwebenchReader.source_id,
        "reader_version": BagenSwebenchReader.source_id,
        "family_root": resolved_root.name,
        "family": next(iter(state.model_family)),
        "raw_file_count": len(raw_files),
        "raw_bytes": sum(int(item["bytes"]) for item in raw_files),
        "source_hashes": {str(item["path"]): str(item["sha256"]) for item in raw_files},
        "raw_files": raw_files,
        "task_count": len(state.task_ids),
        "trajectory_count": len(state.trajectory_ids),
        "condition_count": len(state.condition_tasks),
        "message_count": sum(state.message_roles.values()),
        "call_count": state.call_count,
        "attempt_count": state.attempt_count,
        "complete_usage_attempts": state.complete_usage_attempts,
        "missing_usage_attempts": state.missing_usage_attempts,
        "retry_count": state.retry_count,
        "retry_count_definition": "retryable FormatError recovery attempts",
        "within_call_retry_count": state.within_call_retry_count,
        "format_error_recovery_calls": state.format_error_recovery_calls,
        "tool_event_count": state.tool_event_count,
        "tool_failure_count": state.tool_failure_count,
        "tool_terminal_intercept_count": state.tool_terminal_intercept_count,
        "token_sums": _token_counter(state.token_sums),
        "token_sum_coverage_attempts": _token_counter(state.token_sum_coverage),
        "token_accounting": {
            "accounted_total_definition": "input_tokens + output_tokens",
            "cache_and_reasoning_added_to_total": False,
            "attempt_total_closure": {
                "started_attempts": state.attempt_count,
                "terminal_attempts": state.attempt_count,
                "closed": state.attempt_total_closure["closed"],
                "unavailable": state.attempt_total_closure["unavailable"],
                "mismatched": 0,
                "all_attempts_have_one_terminal": True,
                "all_known_totals_match": True,
            },
            "task_total_closure": _sorted_counts(state.task_total_closure),
        },
        "condition_task_counts": condition_task_counts,
        "task_trajectory_counts": _sorted_counts(state.trajectories_per_task),
        "distributions": {
            "messages_per_trajectory": _sorted_counts(state.messages_per_trajectory),
            "message_roles": _sorted_counts(state.message_roles),
            "exit_status": _sorted_counts(state.exit_status),
            "task_terminal_event": _sorted_counts(state.task_terminal_event),
            "model_family": _sorted_counts(state.model_family),
            "provider": _sorted_counts(state.provider),
            "configured_model": _sorted_counts(state.configured_model),
            "resolved_model_trajectory_presence": _sorted_counts(
                state.resolved_model_trajectory_presence
            ),
            "resolved_model_attempts": _sorted_counts(state.resolved_model_attempts),
            "agent_id": _sorted_counts(state.agent_id),
            "agent_type": _sorted_counts(state.agent_type),
            "agent_version": _sorted_counts(state.agent_version),
            "mini_version": _sorted_counts(state.mini_version),
            "attempt_status": _sorted_counts(state.attempt_status),
            "attempt_usage_total_source": _sorted_counts(state.attempt_usage_total_source),
            "api_error_type": _sorted_counts(state.api_error_type),
            "api_error_status_code": _sorted_counts(state.api_error_status_code),
            "api_error_retryable": _sorted_counts(state.api_error_retryable),
            "format_errors_per_trajectory": _sorted_counts(state.format_errors_per_trajectory),
            "tool_events_per_trajectory": _sorted_counts(state.tool_events_per_trajectory),
            "tool_failures_per_trajectory": _sorted_counts(state.tool_failures_per_trajectory),
            "tool_name": _sorted_counts(state.tool_name),
        },
        "dataset": dataset_summary,
        "supported_observables": sorted(
            observable.value for observable in BagenSwebenchReader.capabilities.observables
        ),
        "canonical_content_sha256": first_family_hash,
        "canonical_rerun_content_sha256": rerun_family_hash,
        "canonical_rerun_consistent": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream-audit one BAGEN SWE-bench model-family directory."
    )
    parser.add_argument("root", type=Path, help="BAGEN family root containing *.traj.json")
    parser.add_argument("output", type=Path, help="deterministic JSON audit output")
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output.resolve()
    if root.is_dir() and output.is_relative_to(root) and output.name.endswith(".traj.json"):
        parser.error("output inside the family root must not match *.traj.json")
    try:
        audit = build_audit(root)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (BagenSwebenchAuditError, BagenSwebenchSchemaError, OSError) as exc:
        parser.exit(2, f"audit failed: {exc}\n")


if __name__ == "__main__":
    main()
