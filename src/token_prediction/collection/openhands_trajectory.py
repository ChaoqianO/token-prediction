from __future__ import annotations

import hashlib
import json
import math
import re
import tarfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence

from token_prediction.contracts import (
    CanonicalEvent,
    EventType,
    Observable,
    SourceCapabilities,
)
from token_prediction.trajectory import Trajectory


class OpenHandsArchiveError(ValueError):
    """Base error for an OpenHands archive that cannot be normalized safely."""


class OpenHandsArchiveSchemaError(OpenHandsArchiveError):
    """Raised when preserved telemetry does not match the verified schema."""


@dataclass(frozen=True)
class OpenHandsArchiveMetadata:
    """Declared facts that are not safely recoverable from completion payloads.

    ``archive_identity`` should normally be the frozen archive SHA256 from an
    inventory.  A deterministic name-and-size identity is used when it is not
    supplied; that fallback is intentionally labelled in task metadata.
    """

    benchmark_id: str = "swe-bench"
    archive_identity: str | None = None
    openhands_version: str = "0.62.0"
    max_iterations: int = 500
    hint_mode: str = "no-hint"
    max_json_bytes: int = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _RunDescriptor:
    directory: str
    run_id: str
    model: str
    openhands_version: str
    max_iterations: int
    hint_mode: str


@dataclass(frozen=True, slots=True)
class _ContentSummary:
    content_hash: str
    chars: int
    bytes: int


@dataclass(frozen=True, slots=True)
class _ToolCallSnapshot:
    source_id: str
    public_id: str
    name: str
    arguments_hash: str
    arguments_chars: int


@dataclass(frozen=True, slots=True)
class _MessageSnapshot:
    fingerprint: str
    role: str
    content: _ContentSummary
    tool_calls: tuple[_ToolCallSnapshot, ...] = ()
    tool_call_id: str | None = None
    public_tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass(frozen=True, slots=True)
class _Usage:
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cached_input_tokens: int | None
    cache_write_input_tokens: int | None
    reasoning_output_tokens: int | None
    image_output_tokens: int | None
    reasoning_subset_valid: bool | None

    @property
    def complete(self) -> bool:
        return self.input_tokens is not None and self.output_tokens is not None

    @classmethod
    def missing(cls) -> "_Usage":
        return cls(None, None, None, None, None, None, None, None)

    def to_payload(
        self,
        *,
        total_source: str = "reported_current_response",
        usage_scope: str = "current_response_only",
    ) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_input_tokens": self.cache_write_input_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
            "image_output_tokens": self.image_output_tokens,
            "reasoning_subset_valid": self.reasoning_subset_valid,
            "total_source": total_source if self.complete else "missing",
            "usage_scope": usage_scope,
        }


@dataclass(frozen=True, slots=True)
class _ResponseSnapshot:
    response_id: str
    response_id_hash: str
    created: int
    model: str
    provider: str
    finish_reason: str
    content: _ContentSummary
    reasoning: _ContentSummary | None
    tool_calls: tuple[_ToolCallSnapshot, ...]
    usage: _Usage
    provider_error_envelope: bool


@dataclass(frozen=True, slots=True)
class _CompletionSnapshot:
    member_name: str
    raw_sha256: str
    filename_timestamp: Decimal
    top_timestamp: Decimal
    response_created: int
    filename_model: str
    filename_provider: str
    tool_config_hash: str
    messages: tuple[_MessageSnapshot, ...]
    response: _ResponseSnapshot

    @property
    def sort_key(self) -> tuple[Decimal, Decimal, int, str]:
        return (
            self.filename_timestamp,
            self.top_timestamp,
            self.response_created,
            self.response.response_id,
        )


@dataclass(frozen=True, slots=True)
class _ReportSnapshot:
    member_name: str
    patch_is_none: bool
    patch_exists: bool
    patch_successfully_applied: bool
    evaluator_resolved: bool
    test_counts: tuple[tuple[str, int, int], ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "patch_is_none": self.patch_is_none,
            "patch_exists": self.patch_exists,
            "patch_successfully_applied": self.patch_successfully_applied,
            "resolved": self.evaluator_resolved,
            "tests_status_counts": {
                category: {"success": success, "failure": failure}
                for category, success, failure in self.test_counts
            },
        }


@dataclass(frozen=True, slots=True)
class _CallSnapshot:
    member_name: str
    filename_timestamp: Decimal
    top_timestamp: Decimal
    configured_provider: str
    request_message_count: int
    request_role_counts: tuple[tuple[str, int], ...]
    request_content_chars: int
    request_content_hash: str
    response: _ResponseSnapshot


@dataclass(frozen=True, slots=True)
class _FallbackToolResult:
    message_index: int
    tool_call_id: str
    public_tool_call_id: str
    tool_name: str
    content: _ContentSummary


@dataclass(frozen=True, slots=True)
class _PreparedTask:
    run: _RunDescriptor
    task_id: str
    tool_config_hash: str
    configured_provider: str
    resolved_model: str | None
    calls: tuple[_CallSnapshot, ...]
    fallback_tools: tuple[tuple[_FallbackToolResult, ...], ...]
    message_prefix_reset_count: int
    repeated_request_snapshot_count: int
    response_not_materialized_count: int


@dataclass(frozen=True, slots=True)
class _HistoryTool:
    source_action_id: int
    source_observation_id: int
    source_action_timestamp: str
    source_observation_timestamp: str
    tool_call_id: str
    public_tool_call_id: str
    tool_name: str
    output: _ContentSummary
    failed: bool
    failure_evidence: str | None
    action_history_index: int
    observation_history_index: int


@dataclass(frozen=True, slots=True)
class _TaskLogSummary:
    task_id: str
    member_name: str
    line_number: int
    history_present: bool
    finished: bool
    error_hash: str | None
    error_chars: int | None
    terminal_source_timestamp: str | None
    task_usage: _Usage
    token_usages: tuple[tuple[str, _Usage], ...]
    history_llm_metrics_count: int
    tools: tuple[_HistoryTool, ...]


@dataclass(frozen=True, slots=True)
class _AggregateReport:
    submitted_ids: frozenset[str]


class OpenHandsArchiveReader:
    """Normalize preserved OpenHands ``llm_completions`` in one tar stream.

    The archive is opened only in ``r|gz`` mode.  Members are never extracted
    and the member list is never materialized.  A task buffers only redacted
    message fingerprints, lengths, role/tool structure, and current-response
    usage; raw prompts, responses, and tool arguments are discarded as soon as
    their snapshot is constructed.

    Every completion proves one request, one API attempt start, and one API
    completion.  ``output.jsonl`` supplies independently recorded task
    lifecycle, task usage, and action/observation pairs: an explicit final
    ``finish`` action becomes ``TASK_FINISHED``; a top-level task error becomes
    ``TASK_ABORTED``; missing lifecycle evidence remains censored as
    ``logging_incomplete``.  Provider error envelopes never become API failure
    or retry events, and no generation checkpoint is synthesized.
    """

    source_id = "openhands_archive_trajectory_v3"
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
        metadata: OpenHandsArchiveMetadata | None = None,
        *,
        max_trajectories: int | None = None,
    ) -> tuple[Trajectory, ...]:
        """Read the archive into a deterministic tuple of trajectories."""

        return tuple(
            self.iter_archive(
                location,
                metadata,
                max_trajectories=max_trajectories,
            )
        )

    def read_all(
        self,
        location: str | Path,
        metadata: OpenHandsArchiveMetadata | None = None,
        *,
        max_trajectories: int | None = None,
    ) -> tuple[Trajectory, ...]:
        """Alias for :meth:`read`, matching other multi-trajectory readers."""

        return self.read(
            location,
            metadata,
            max_trajectories=max_trajectories,
        )

    def iter_archive(
        self,
        location: str | Path,
        metadata: OpenHandsArchiveMetadata | None = None,
        *,
        max_trajectories: int | None = None,
    ) -> Iterator[Trajectory]:
        """Yield run/task trajectories from one forward-only gzip tar pass."""

        source = Path(location).resolve()
        if not source.is_file() or not source.name.lower().endswith((".tar.gz", ".tgz")):
            raise OpenHandsArchiveError("source must be one .tar.gz or .tgz file")
        resolved = metadata or OpenHandsArchiveMetadata()
        _validate_metadata(resolved)
        if max_trajectories is not None and max_trajectories <= 0:
            raise OpenHandsArchiveError("max_trajectories must be positive or None")

        stat = source.stat()
        if resolved.archive_identity is None:
            archive_identity = _semantic_hash(
                {"archive_name": source.name, "archive_bytes": stat.st_size}
            )
            archive_identity_source = "archive_name_and_size_fallback"
        else:
            archive_identity = resolved.archive_identity.strip()
            if not archive_identity:
                raise OpenHandsArchiveError("archive_identity must be non-empty")
            archive_identity_source = "declared"
        archive_identity_hash = _text_hash(archive_identity)

        active_run: _RunDescriptor | None = None
        reports: dict[str, _ReportSnapshot] = {}
        prepared_tasks: dict[str, _PreparedTask] = {}
        task_logs: dict[str, _TaskLogSummary] = {}
        aggregate_report: _AggregateReport | None = None
        current_task: str | None = None
        snapshots: list[_CompletionSnapshot] = []
        run_directories: dict[str, str] = {}
        yielded = 0
        saw_task_evidence = False

        def close_completion_group() -> None:
            nonlocal current_task, snapshots
            if current_task is None:
                if snapshots:
                    raise OpenHandsArchiveSchemaError(
                        "completion grouping state is inconsistent"
                    )
                return
            if active_run is None:
                raise OpenHandsArchiveSchemaError("completion has no active run")
            if current_task in prepared_tasks:
                raise OpenHandsArchiveSchemaError(
                    "completion members for one run/task are not contiguous"
                )
            prepared_tasks[current_task] = _prepare_task_snapshots(
                run=active_run,
                task_id=current_task,
                snapshots=snapshots,
            )
            current_task = None
            snapshots = []

        def reset_run_state(run: _RunDescriptor | None) -> None:
            nonlocal active_run, reports, prepared_tasks, task_logs
            nonlocal aggregate_report, current_task, snapshots
            active_run = run
            reports = {}
            prepared_tasks = {}
            task_logs = {}
            aggregate_report = None
            current_task = None
            snapshots = []

        try:
            with tarfile.open(source, mode="r|gz") as archive:
                for member in archive:
                    member_name = _canonical_member_name(member.name)
                    located = _locate_run(member_name)
                    if located is None:
                        continue
                    run_directory, relative_parts = located
                    relevant = _relevant_member_kind(relative_parts, member.isfile())
                    if relevant is None:
                        continue
                    run = _parse_run_descriptor(run_directory, resolved)
                    prior_directory = run_directories.setdefault(run.run_id, run.directory)
                    if prior_directory != run.directory:
                        raise OpenHandsArchiveSchemaError(
                            "one run_id is represented by multiple run directories"
                        )

                    if active_run is None:
                        reset_run_state(run)
                    elif run.run_id != active_run.run_id:
                        close_completion_group()
                        for trajectory in _finalize_run(
                            source_name=source.name,
                            run=active_run,
                            prepared_tasks=prepared_tasks,
                            task_logs=task_logs,
                            reports=reports,
                            aggregate_report=aggregate_report,
                            metadata=resolved,
                            archive_identity=archive_identity,
                            archive_identity_hash=archive_identity_hash,
                            archive_identity_source=archive_identity_source,
                        ):
                            yielded += 1
                            yield trajectory
                            if (
                                max_trajectories is not None
                                and yielded >= max_trajectories
                            ):
                                return
                        reset_run_state(run)
                    elif run != active_run:
                        raise OpenHandsArchiveSchemaError(
                            "one run_id has conflicting run metadata"
                        )

                    if relevant == "report":
                        if not member.isfile():
                            raise OpenHandsArchiveSchemaError(
                                "report member must be a regular file"
                            )
                        report_payload = _read_json_member(
                            archive,
                            member,
                            resolved.max_json_bytes,
                            "report",
                        )
                        task_hint = (
                            relative_parts[1]
                            if len(relative_parts) == 3
                            and relative_parts[0] == "eval_outputs"
                            else None
                        )
                        for task_id, report in _parse_report(
                            report_payload,
                            member_name,
                            task_hint,
                        ).items():
                            if task_id in reports:
                                raise OpenHandsArchiveSchemaError(
                                    "a run/task has duplicate evaluator reports"
                                )
                            reports[task_id] = report
                            saw_task_evidence = True
                        continue

                    if relevant == "aggregate_report":
                        close_completion_group()
                        if aggregate_report is not None:
                            raise OpenHandsArchiveSchemaError(
                                "run contains multiple aggregate reports"
                            )
                        payload = _read_json_member(
                            archive,
                            member,
                            resolved.max_json_bytes,
                            "aggregate report",
                        )
                        aggregate_report = _parse_aggregate_report(payload)
                        continue

                    if relevant == "task_log":
                        close_completion_group()
                        if task_logs:
                            raise OpenHandsArchiveSchemaError(
                                "run contains multiple output.jsonl task logs"
                            )
                        task_logs = _read_task_log_member(
                            archive,
                            member,
                            resolved.max_json_bytes,
                            run,
                        )
                        saw_task_evidence = saw_task_evidence or bool(task_logs)
                        continue

                    task_id = relative_parts[1]
                    if current_task is not None and task_id != current_task:
                        close_completion_group()
                    if task_id in prepared_tasks:
                        raise OpenHandsArchiveSchemaError(
                            "completion members for one run/task are not contiguous"
                        )
                    if current_task is None:
                        current_task = task_id
                    if not member.isfile():
                        raise OpenHandsArchiveSchemaError(
                            "completion member must be a regular file"
                        )
                    payload, raw_sha256 = _read_json_member_with_hash(
                        archive,
                        member,
                        resolved.max_json_bytes,
                        "completion",
                    )
                    snapshots.append(
                        _parse_completion(
                            payload,
                            raw_sha256=raw_sha256,
                            member_name=member_name,
                            run=run,
                        )
                    )
                    saw_task_evidence = True

                close_completion_group()
                if active_run is not None:
                    for trajectory in _finalize_run(
                        source_name=source.name,
                        run=active_run,
                        prepared_tasks=prepared_tasks,
                        task_logs=task_logs,
                        reports=reports,
                        aggregate_report=aggregate_report,
                        metadata=resolved,
                        archive_identity=archive_identity,
                        archive_identity_hash=archive_identity_hash,
                        archive_identity_source=archive_identity_source,
                    ):
                        yielded += 1
                        yield trajectory
                        if (
                            max_trajectories is not None
                            and yielded >= max_trajectories
                        ):
                            return
        except OpenHandsArchiveError:
            raise
        except (OSError, EOFError, tarfile.TarError, UnicodeError) as exc:
            raise OpenHandsArchiveError(f"cannot stream OpenHands archive: {exc}") from exc

        if not saw_task_evidence or yielded == 0:
            raise OpenHandsArchiveSchemaError(
                "archive contains no supported task-run evidence"
            )


def _validate_metadata(metadata: OpenHandsArchiveMetadata) -> None:
    if not metadata.benchmark_id.strip():
        raise OpenHandsArchiveError("benchmark_id must be non-empty")
    if not re.fullmatch(r"\d+(?:\.\d+){2}", metadata.openhands_version):
        raise OpenHandsArchiveError("openhands_version must be a semantic version")
    if metadata.max_iterations <= 0:
        raise OpenHandsArchiveError("max_iterations must be positive")
    if metadata.hint_mode not in {"hint", "no-hint"}:
        raise OpenHandsArchiveError("hint_mode must be 'hint' or 'no-hint'")
    if metadata.max_json_bytes <= 0:
        raise OpenHandsArchiveError("max_json_bytes must be positive")


def _canonical_member_name(value: str) -> str:
    if not value or "\\" in value or value.startswith("/"):
        raise OpenHandsArchiveSchemaError("archive contains a non-canonical member path")
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise OpenHandsArchiveSchemaError("archive contains an unsafe member path")
    rendered = path.as_posix()
    if rendered != value.rstrip("/"):
        raise OpenHandsArchiveSchemaError("archive member path is not canonical")
    return rendered


def _locate_run(member_name: str) -> tuple[str, tuple[str, ...]] | None:
    parts = PurePosixPath(member_name).parts
    for index, part in enumerate(parts):
        if re.search(r"-run_\d+$", part):
            return part, tuple(parts[index + 1 :])
    return None


def _relevant_member_kind(
    relative_parts: tuple[str, ...],
    is_file: bool,
) -> str | None:
    if not relative_parts:
        return None
    if relative_parts[0] == "llm_completions":
        if not is_file:
            return None
        if len(relative_parts) != 3 or not relative_parts[2].endswith(".json"):
            raise OpenHandsArchiveSchemaError(
                "unsupported member below llm_completions"
            )
        if not relative_parts[1]:
            raise OpenHandsArchiveSchemaError("completion task directory is empty")
        return "completion"
    if relative_parts == ("report.json",):
        return "aggregate_report"
    if relative_parts == ("output.jsonl",):
        return "task_log"
    if (
        len(relative_parts) == 3
        and relative_parts[0] == "eval_outputs"
        and relative_parts[2] == "report.json"
    ):
        return "report"
    return None


_RUN_RE = re.compile(
    r"^(?P<model>.+)_maxiter_(?P<max_iterations>\d+)_N_"
    r"v(?P<version>\d+(?:\.\d+){2})-"
    r"(?P<hint_mode>no-hint|hint)-run_(?P<run>\d+)$"
)


def _parse_run_descriptor(
    directory: str,
    metadata: OpenHandsArchiveMetadata,
) -> _RunDescriptor:
    match = _RUN_RE.fullmatch(directory)
    if match is None:
        raise OpenHandsArchiveSchemaError(
            "run directory does not match the verified OpenHands naming schema"
        )
    version = match.group("version")
    max_iterations = int(match.group("max_iterations"))
    hint_mode = match.group("hint_mode")
    if version != metadata.openhands_version:
        raise OpenHandsArchiveSchemaError(
            "run directory OpenHands version disagrees with metadata"
        )
    if max_iterations != metadata.max_iterations:
        raise OpenHandsArchiveSchemaError(
            "run directory max iterations disagrees with metadata"
        )
    if hint_mode != metadata.hint_mode:
        raise OpenHandsArchiveSchemaError(
            "run directory hint mode disagrees with metadata"
        )
    return _RunDescriptor(
        directory=directory,
        run_id=f"run_{int(match.group('run'))}",
        model=match.group("model"),
        openhands_version=version,
        max_iterations=max_iterations,
        hint_mode=hint_mode,
    )


def _read_json_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    max_bytes: int,
    context: str,
) -> Mapping[str, Any]:
    value, _ = _read_json_member_with_hash(archive, member, max_bytes, context)
    return value


def _read_json_member_with_hash(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    max_bytes: int,
    context: str,
) -> tuple[Mapping[str, Any], str]:
    if member.size < 0 or member.size > max_bytes:
        raise OpenHandsArchiveSchemaError(
            f"{context} JSON exceeds the configured byte limit"
        )
    handle = archive.extractfile(member)
    if handle is None:
        raise OpenHandsArchiveSchemaError(f"{context} JSON member is unreadable")
    raw = handle.read(max_bytes + 1)
    if len(raw) != member.size or len(raw) > max_bytes:
        raise OpenHandsArchiveSchemaError(
            f"{context} JSON size disagrees with its archive header"
        )
    digest = hashlib.sha256(raw).hexdigest()

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise OpenHandsArchiveSchemaError(
                    f"{context} JSON contains a duplicate object key"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise OpenHandsArchiveSchemaError(
            f"{context} JSON contains a non-finite numeric constant"
        )

    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except OpenHandsArchiveError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise OpenHandsArchiveSchemaError(f"invalid {context} JSON") from exc
    if not isinstance(decoded, Mapping):
        raise OpenHandsArchiveSchemaError(f"{context} JSON root must be an object")
    return decoded, digest


_COMPLETION_FILE_RE = re.compile(
    r"^(?P<provider>[^/]+?)__(?P<model>.+)-"
    r"(?P<timestamp>\d+(?:\.\d+)?)\.json$"
)


def _parse_completion(
    value: Mapping[str, Any],
    *,
    raw_sha256: str,
    member_name: str,
    run: _RunDescriptor,
) -> _CompletionSnapshot:
    _validate_exact_keys(
        value,
        {"messages", "response", "args", "kwargs", "timestamp", "cost"},
        context="completion root",
    )
    filename = PurePosixPath(member_name).name
    file_match = _COMPLETION_FILE_RE.fullmatch(filename)
    if file_match is None:
        raise OpenHandsArchiveSchemaError(
            "completion filename does not contain the verified numeric suffix"
        )
    filename_model = file_match.group("model")
    filename_provider = file_match.group("provider")
    if filename_model != run.model:
        raise OpenHandsArchiveSchemaError(
            "completion filename model disagrees with its run directory"
        )

    args = value.get("args")
    if not isinstance(args, list) or args:
        raise OpenHandsArchiveSchemaError("completion args must be an empty array")
    kwargs = _require_mapping(value.get("kwargs"), "completion kwargs")
    _validate_exact_keys(kwargs, {"tools"}, context="completion kwargs")
    tools = kwargs.get("tools")
    _validate_tools(tools)
    tool_config_hash = _semantic_hash(tools)

    raw_messages = value.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise OpenHandsArchiveSchemaError(
            "completion messages must be a non-empty array"
        )
    messages = tuple(
        _parse_request_message(message, index)
        for index, message in enumerate(raw_messages)
    )

    response = _parse_response(value.get("response"), filename_provider)
    top_timestamp = _decimal_number(value.get("timestamp"), "completion timestamp")
    filename_timestamp = _decimal_text(
        file_match.group("timestamp"), "completion filename timestamp"
    )
    _finite_number(value.get("cost"), "completion cost")
    return _CompletionSnapshot(
        member_name=member_name,
        raw_sha256=raw_sha256,
        filename_timestamp=filename_timestamp,
        top_timestamp=top_timestamp,
        response_created=response.created,
        filename_model=filename_model,
        filename_provider=filename_provider,
        tool_config_hash=tool_config_hash,
        messages=messages,
        response=response,
    )


def _parse_request_message(value: Any, index: int) -> _MessageSnapshot:
    message = _require_mapping(value, f"request message {index}")
    role = _required_text(message, "role", f"request message {index}")
    if role in {"system", "user"}:
        _validate_exact_keys(
            message,
            {"content", "role"},
            context=f"request message {index}",
        )
        tool_calls: tuple[_ToolCallSnapshot, ...] = ()
        tool_call_id = None
        public_tool_call_id = None
        tool_name = None
    elif role == "assistant":
        _validate_exact_keys(
            message,
            {"content", "role"},
            optional={"tool_calls"},
            context=f"request message {index}",
        )
        tool_calls = _parse_tool_calls(
            message.get("tool_calls", []),
            f"request message {index}",
            response_shape=False,
        )
        tool_call_id = None
        public_tool_call_id = None
        tool_name = None
    elif role == "tool":
        _validate_exact_keys(
            message,
            {"content", "role", "tool_call_id", "name"},
            context=f"request message {index}",
        )
        tool_calls = ()
        tool_call_id = _required_text(
            message, "tool_call_id", f"request message {index}"
        )
        public_tool_call_id = _public_tool_call_id(tool_call_id)
        tool_name = _required_text(message, "name", f"request message {index}")
    else:
        raise OpenHandsArchiveSchemaError(
            f"request message {index} has an unsupported role"
        )
    return _MessageSnapshot(
        fingerprint=_semantic_hash(message),
        role=role,
        content=_request_content_summary(
            message.get("content"), f"request message {index}.content"
        ),
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
        public_tool_call_id=public_tool_call_id,
        tool_name=tool_name,
    )


def _request_content_summary(value: Any, context: str) -> _ContentSummary:
    if not isinstance(value, list) or len(value) > 1:
        raise OpenHandsArchiveSchemaError(
            f"{context} must be an array containing at most one text part"
        )
    texts: list[str] = []
    for part_index, raw_part in enumerate(value):
        part = _require_mapping(raw_part, f"{context}[{part_index}]")
        _validate_exact_keys(
            part,
            {"type", "text"},
            context=f"{context}[{part_index}]",
        )
        if part.get("type") != "text" or not isinstance(part.get("text"), str):
            raise OpenHandsArchiveSchemaError(
                f"{context}[{part_index}] must be one text content part"
            )
        texts.append(part["text"])
    return _plain_text_summary("".join(texts))


def _parse_response(value: Any, filename_provider: str) -> _ResponseSnapshot:
    response = _require_mapping(value, "completion response")
    _validate_exact_keys(
        response,
        {
            "id",
            "created",
            "model",
            "object",
            "system_fingerprint",
            "choices",
            "provider",
        },
        optional={"usage"},
        context="completion response",
    )
    response_id = _required_text(response, "id", "completion response")
    created = _non_negative_int(response.get("created"), "response.created")
    model = _required_text(response, "model", "completion response")
    provider = _required_text(response, "provider", "completion response")
    provider_matches = provider.casefold() == filename_provider.casefold()
    known_openai_azure_route = (
        filename_provider.casefold() == "openai"
        and provider.casefold() == "azure"
    )
    if not provider_matches and not known_openai_azure_route:
        raise OpenHandsArchiveSchemaError(
            "response provider disagrees with completion filename"
        )
    _required_text(response, "object", "completion response")
    fingerprint = response.get("system_fingerprint")
    if fingerprint is not None and not isinstance(fingerprint, str):
        raise OpenHandsArchiveSchemaError(
            "response.system_fingerprint must be a string or null"
        )
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise OpenHandsArchiveSchemaError("response must contain exactly one choice")
    choice = _require_mapping(choices[0], "response choice")
    _validate_exact_keys(
        choice,
        {"finish_reason", "index", "message", "provider_specific_fields"},
        context="response choice",
    )
    finish_reason = _required_text(choice, "finish_reason", "response choice")
    if _non_negative_int(choice.get("index"), "response choice.index") != 0:
        raise OpenHandsArchiveSchemaError("response choice index must be zero")
    provider_fields = _require_mapping(
        choice.get("provider_specific_fields"), "provider_specific_fields"
    )
    _validate_exact_keys(
        provider_fields,
        {"native_finish_reason"},
        optional={"error"},
        context="provider_specific_fields",
    )
    native_finish_reason = provider_fields.get("native_finish_reason")
    if native_finish_reason is not None and not isinstance(native_finish_reason, str):
        raise OpenHandsArchiveSchemaError(
            "provider native_finish_reason must be a string or null"
        )
    if "error" in provider_fields:
        _validate_opaque_provider_error(provider_fields["error"])
    message = _require_mapping(choice.get("message"), "response choice message")
    _validate_exact_keys(
        message,
        {"content", "role", "tool_calls", "function_call"},
        optional={"reasoning_content"},
        context="response choice message",
    )
    if message.get("role") != "assistant":
        raise OpenHandsArchiveSchemaError("response choice role must be assistant")
    content = message.get("content")
    if not isinstance(content, str):
        raise OpenHandsArchiveSchemaError("response choice content must be a string")
    if message.get("function_call") is not None:
        raise OpenHandsArchiveSchemaError(
            "legacy response function_call is unsupported"
        )
    reasoning_value = message.get("reasoning_content")
    if reasoning_value is not None and not isinstance(reasoning_value, str):
        raise OpenHandsArchiveSchemaError(
            "response reasoning_content must be a string when present"
        )
    tool_calls = _parse_tool_calls(
        message.get("tool_calls"),
        "response choice message",
        response_shape=True,
    )
    usage = _parse_usage(response.get("usage"))
    return _ResponseSnapshot(
        response_id=response_id,
        response_id_hash=_text_hash(response_id),
        created=created,
        model=model,
        provider=provider,
        finish_reason=finish_reason,
        content=_plain_text_summary(content),
        reasoning=(
            _plain_text_summary(reasoning_value)
            if isinstance(reasoning_value, str)
            else None
        ),
        tool_calls=tool_calls,
        usage=usage,
        provider_error_envelope="error" in provider_fields,
    )


def _validate_opaque_provider_error(value: Any) -> None:
    """Validate a known provider envelope without treating it as attempt failure."""

    error = _require_mapping(value, "provider_specific_fields.error")
    _validate_exact_keys(
        error,
        {"message", "code", "metadata"},
        context="provider_specific_fields.error",
    )
    if not isinstance(error.get("message"), str):
        raise OpenHandsArchiveSchemaError("provider error message must be a string")
    _non_negative_int(error.get("code"), "provider error code")
    metadata = _require_mapping(error.get("metadata"), "provider error metadata")
    _validate_exact_keys(
        metadata,
        {"provider_name"},
        optional={"raw"},
        context="provider error metadata",
    )
    if not isinstance(metadata.get("provider_name"), str):
        raise OpenHandsArchiveSchemaError("provider error name must be a string")
    if "raw" in metadata:
        raw = _require_mapping(metadata.get("raw"), "provider error raw envelope")
        _validate_exact_keys(
            raw,
            {"code", "message"},
            context="provider error raw envelope",
        )
        if not isinstance(raw.get("code"), str) or not isinstance(
            raw.get("message"), str
        ):
            raise OpenHandsArchiveSchemaError(
                "provider error raw code/message must be strings"
            )


def _parse_tool_calls(
    value: Any,
    context: str,
    *,
    response_shape: bool,
) -> tuple[_ToolCallSnapshot, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise OpenHandsArchiveSchemaError(f"{context}.tool_calls must be an array")
    result: list[_ToolCallSnapshot] = []
    seen: set[str] = set()
    for index, raw_call in enumerate(value):
        call = _require_mapping(raw_call, f"{context}.tool_calls[{index}]")
        required = {"id", "type", "function"}
        if response_shape:
            required.add("index")
        _validate_exact_keys(
            call,
            required,
            context=f"{context}.tool_calls[{index}]",
        )
        if call.get("type") != "function":
            raise OpenHandsArchiveSchemaError("only function tool calls are supported")
        if response_shape and _non_negative_int(
            call.get("index"), f"{context}.tool_calls[{index}].index"
        ) != index:
            raise OpenHandsArchiveSchemaError(
                "response tool call indexes must be contiguous"
            )
        source_id = _required_text(
            call, "id", f"{context}.tool_calls[{index}]"
        )
        if source_id in seen:
            raise OpenHandsArchiveSchemaError("one message repeats a tool_call_id")
        seen.add(source_id)
        function = _require_mapping(
            call.get("function"), f"{context}.tool_calls[{index}].function"
        )
        _validate_exact_keys(
            function,
            {"name", "arguments"},
            context=f"{context}.tool_calls[{index}].function",
        )
        name = _required_text(
            function, "name", f"{context}.tool_calls[{index}].function"
        )
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            raise OpenHandsArchiveSchemaError("tool arguments must be a string")
        result.append(
            _ToolCallSnapshot(
                source_id=source_id,
                public_id=_public_tool_call_id(source_id),
                name=name,
                arguments_hash=_text_hash(arguments),
                arguments_chars=len(arguments),
            )
        )
    return tuple(result)


def _validate_tools(value: Any) -> None:
    if not isinstance(value, list):
        raise OpenHandsArchiveSchemaError("completion kwargs.tools must be an array")
    names: set[str] = set()
    for index, raw_tool in enumerate(value):
        tool = _require_mapping(raw_tool, f"kwargs.tools[{index}]")
        _validate_exact_keys(
            tool,
            {"type", "function"},
            context=f"kwargs.tools[{index}]",
        )
        if tool.get("type") != "function":
            raise OpenHandsArchiveSchemaError("only function tool definitions are supported")
        function = _require_mapping(
            tool.get("function"), f"kwargs.tools[{index}].function"
        )
        _validate_exact_keys(
            function,
            {"name", "description", "parameters"},
            context=f"kwargs.tools[{index}].function",
        )
        name = _required_text(function, "name", f"kwargs.tools[{index}].function")
        if name in names:
            raise OpenHandsArchiveSchemaError("tool configuration repeats a name")
        names.add(name)
        if not isinstance(function.get("description"), str):
            raise OpenHandsArchiveSchemaError("tool description must be a string")
        parameters = _require_mapping(
            function.get("parameters"), f"kwargs.tools[{index}].parameters"
        )
        _validate_exact_keys(
            parameters,
            {"type", "properties", "required"},
            optional={"additionalProperties"},
            context=f"kwargs.tools[{index}].parameters",
        )
        if parameters.get("type") != "object":
            raise OpenHandsArchiveSchemaError("tool parameters must describe an object")
        _require_mapping(
            parameters.get("properties"), f"kwargs.tools[{index}].properties"
        )
        required = parameters.get("required")
        if not isinstance(required, list) or not all(
            isinstance(item, str) for item in required
        ):
            raise OpenHandsArchiveSchemaError(
                "tool parameters.required must be an array of strings"
            )
        additional = parameters.get("additionalProperties")
        if additional is not None and not isinstance(additional, bool):
            raise OpenHandsArchiveSchemaError(
                "tool additionalProperties must be boolean when present"
            )


def _parse_usage(value: Any) -> _Usage:
    if value is None:
        return _Usage.missing()
    usage = _require_mapping(value, "response usage")
    _validate_exact_keys(
        usage,
        {
            "completion_tokens",
            "prompt_tokens",
            "total_tokens",
            "completion_tokens_details",
            "prompt_tokens_details",
            "cost",
            "is_byok",
            "cost_details",
        },
        context="response usage",
    )
    input_tokens = _non_negative_int(usage.get("prompt_tokens"), "usage.prompt_tokens")
    output_tokens = _non_negative_int(
        usage.get("completion_tokens"), "usage.completion_tokens"
    )
    total_tokens = _non_negative_int(usage.get("total_tokens"), "usage.total_tokens")
    if input_tokens + output_tokens != total_tokens:
        raise OpenHandsArchiveSchemaError(
            "response usage total does not equal prompt plus completion tokens"
        )
    prompt_details = _require_mapping(
        usage.get("prompt_tokens_details"), "usage.prompt_tokens_details"
    )
    _validate_exact_keys(
        prompt_details,
        {"audio_tokens", "cached_tokens", "text_tokens", "image_tokens"},
        optional={"video_tokens"},
        context="usage.prompt_tokens_details",
    )
    completion_details = _require_mapping(
        usage.get("completion_tokens_details"), "usage.completion_tokens_details"
    )
    _validate_exact_keys(
        completion_details,
        {
            "accepted_prediction_tokens",
            "audio_tokens",
            "reasoning_tokens",
            "rejected_prediction_tokens",
            "text_tokens",
            "image_tokens",
        },
        context="usage.completion_tokens_details",
    )
    cached = _non_negative_int(
        prompt_details.get("cached_tokens"), "usage.cached_tokens"
    )
    reasoning = _non_negative_int(
        completion_details.get("reasoning_tokens"), "usage.reasoning_tokens"
    )
    image_output = _non_negative_int(
        completion_details.get("image_tokens"), "usage.image_tokens"
    )
    if cached > input_tokens:
        raise OpenHandsArchiveSchemaError(
            "cached prompt tokens exceed prompt tokens"
        )
    reasoning_subset_valid = reasoning <= output_tokens
    for key in ("audio_tokens", "text_tokens", "image_tokens", "video_tokens"):
        _optional_non_negative_int(
            prompt_details.get(key), f"usage.prompt_tokens_details.{key}"
        )
    for key in (
        "accepted_prediction_tokens",
        "audio_tokens",
        "rejected_prediction_tokens",
        "text_tokens",
    ):
        _optional_non_negative_int(
            completion_details.get(key), f"usage.completion_tokens_details.{key}"
        )
    _finite_number(usage.get("cost"), "usage.cost")
    if not isinstance(usage.get("is_byok"), bool):
        raise OpenHandsArchiveSchemaError("usage.is_byok must be boolean")
    cost_details = _require_mapping(usage.get("cost_details"), "usage.cost_details")
    _validate_exact_keys(
        cost_details,
        {
            "upstream_inference_cost",
            "upstream_inference_prompt_cost",
            "upstream_inference_completions_cost",
        },
        context="usage.cost_details",
    )
    for key in cost_details:
        if cost_details[key] is not None:
            _finite_number(cost_details[key], f"usage.cost_details.{key}")
    return _Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached,
        cache_write_input_tokens=None,
        reasoning_output_tokens=reasoning,
        image_output_tokens=image_output,
        reasoning_subset_valid=reasoning_subset_valid,
    )


_TEST_CATEGORIES = (
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "FAIL_TO_FAIL",
    "PASS_TO_FAIL",
)


def _parse_report(
    value: Mapping[str, Any],
    member_name: str,
    task_hint: str | None,
) -> dict[str, _ReportSnapshot]:
    if not value:
        raise OpenHandsArchiveSchemaError("report must contain at least one task")
    if task_hint is not None and set(value) != {task_hint}:
        raise OpenHandsArchiveSchemaError(
            "per-task report key disagrees with its task directory"
        )
    result: dict[str, _ReportSnapshot] = {}
    for raw_task_id, raw_entry in value.items():
        if not isinstance(raw_task_id, str) or not raw_task_id:
            raise OpenHandsArchiveSchemaError("report task id must be a non-empty string")
        entry = _require_mapping(raw_entry, "report task entry")
        _validate_exact_keys(
            entry,
            {
                "patch_is_None",
                "patch_exists",
                "patch_successfully_applied",
                "resolved",
                "tests_status",
            },
            context="report task entry",
        )
        booleans: dict[str, bool] = {}
        for key in (
            "patch_is_None",
            "patch_exists",
            "patch_successfully_applied",
            "resolved",
        ):
            raw = entry.get(key)
            if not isinstance(raw, bool):
                raise OpenHandsArchiveSchemaError(f"report {key} must be boolean")
            booleans[key] = raw
        tests_status = _require_mapping(entry.get("tests_status"), "report tests_status")
        _validate_exact_keys(
            tests_status,
            set(_TEST_CATEGORIES),
            context="report tests_status",
        )
        counts: list[tuple[str, int, int]] = []
        for category in _TEST_CATEGORIES:
            bucket = _require_mapping(
                tests_status.get(category), f"report tests_status.{category}"
            )
            _validate_exact_keys(
                bucket,
                {"success", "failure"},
                context=f"report tests_status.{category}",
            )
            success = bucket.get("success")
            failure = bucket.get("failure")
            if not isinstance(success, list) or not isinstance(failure, list):
                raise OpenHandsArchiveSchemaError(
                    "report test success/failure values must be arrays"
                )
            counts.append((category, len(success), len(failure)))
        result[raw_task_id] = _ReportSnapshot(
            member_name=member_name,
            patch_is_none=booleans["patch_is_None"],
            patch_exists=booleans["patch_exists"],
            patch_successfully_applied=booleans["patch_successfully_applied"],
            evaluator_resolved=booleans["resolved"],
            test_counts=tuple(counts),
        )
    return result


_AGGREGATE_LIST_FIELDS = {
    "completed_ids",
    "empty_patch_ids",
    "error_ids",
    "incomplete_ids",
    "resolved_ids",
    "submitted_ids",
    "unresolved_ids",
}
_AGGREGATE_COUNT_FIELDS = {
    "completed_instances",
    "empty_patch_instances",
    "error_instances",
    "resolved_instances",
    "submitted_instances",
    "total_instances",
    "unresolved_instances",
}


def _parse_aggregate_report(value: Mapping[str, Any]) -> _AggregateReport:
    _validate_exact_keys(
        value,
        _AGGREGATE_LIST_FIELDS | _AGGREGATE_COUNT_FIELDS | {"schema_version"},
        context="aggregate report",
    )
    lists: dict[str, list[str]] = {}
    for key in _AGGREGATE_LIST_FIELDS:
        raw = value.get(key)
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            raise OpenHandsArchiveSchemaError(
                "aggregate report id fields must be arrays of strings"
            )
        if len(raw) != len(set(raw)):
            raise OpenHandsArchiveSchemaError(
                "aggregate report id fields must not contain duplicates"
            )
        lists[key] = raw
    counts = {
        key: _non_negative_int(value.get(key), f"aggregate report.{key}")
        for key in _AGGREGATE_COUNT_FIELDS | {"schema_version"}
    }
    for stem in (
        "completed",
        "empty_patch",
        "error",
        "resolved",
        "submitted",
        "unresolved",
    ):
        if counts[f"{stem}_instances"] != len(lists[f"{stem}_ids"]):
            raise OpenHandsArchiveSchemaError(
                "aggregate report count disagrees with its id list"
            )
    if counts["total_instances"] != len(lists["submitted_ids"]):
        raise OpenHandsArchiveSchemaError(
            "aggregate report total disagrees with submitted ids"
        )
    return _AggregateReport(frozenset(lists["submitted_ids"]))


def _read_task_log_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    max_line_bytes: int,
    run: _RunDescriptor,
) -> dict[str, _TaskLogSummary]:
    handle = archive.extractfile(member)
    if handle is None:
        raise OpenHandsArchiveSchemaError("output.jsonl member is unreadable")
    result: dict[str, _TaskLogSummary] = {}
    consumed = 0
    for line_number, raw in enumerate(handle, start=1):
        consumed += len(raw)
        if len(raw) > max_line_bytes:
            raise OpenHandsArchiveSchemaError(
                "output.jsonl line exceeds the configured JSON byte limit"
            )
        value = _decode_json_object(raw, f"output.jsonl line {line_number}")
        summary = _parse_task_log_line(
            value,
            member_name=member.name.replace("\\", "/"),
            line_number=line_number,
            run=run,
        )
        if summary.task_id in result:
            raise OpenHandsArchiveSchemaError(
                "output.jsonl repeats one task identity"
            )
        result[summary.task_id] = summary
    if consumed != member.size:
        raise OpenHandsArchiveSchemaError(
            "output.jsonl byte count disagrees with its archive header"
        )
    if not result:
        raise OpenHandsArchiveSchemaError("output.jsonl must contain at least one task")
    return result


def _decode_json_object(raw: bytes, context: str) -> Mapping[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise OpenHandsArchiveSchemaError(
                    f"{context} contains a duplicate object key"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise OpenHandsArchiveSchemaError(
            f"{context} contains a non-finite numeric constant"
        )

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except OpenHandsArchiveError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise OpenHandsArchiveSchemaError(f"invalid {context}") from exc
    if not isinstance(value, Mapping):
        raise OpenHandsArchiveSchemaError(f"{context} root must be an object")
    return value


def _parse_task_log_line(
    value: Mapping[str, Any],
    *,
    member_name: str,
    line_number: int,
    run: _RunDescriptor,
) -> _TaskLogSummary:
    _validate_exact_keys(
        value,
        {
            "instance_id",
            "test_result",
            "instruction",
            "metadata",
            "history",
            "metrics",
            "error",
            "instance",
        },
        context="output.jsonl task",
    )
    task_id = _required_text(value, "instance_id", "output.jsonl task")
    if not isinstance(value.get("test_result"), Mapping):
        raise OpenHandsArchiveSchemaError("output.jsonl test_result must be an object")
    raw_error = value.get("error")
    if raw_error is not None and not isinstance(raw_error, str):
        raise OpenHandsArchiveSchemaError(
            "output.jsonl error must be a string or null"
        )
    error_hash = _text_hash(raw_error) if isinstance(raw_error, str) else None
    error_chars = len(raw_error) if isinstance(raw_error, str) else None
    history = value.get("history")
    if history is None:
        if raw_error is None:
            raise OpenHandsArchiveSchemaError(
                "task with no history must carry an explicit top-level error"
            )
        for key in ("instruction", "metadata", "metrics", "instance"):
            if value.get(key) is not None:
                raise OpenHandsArchiveSchemaError(
                    "task with no history has inconsistent non-null task fields"
                )
        return _TaskLogSummary(
            task_id=task_id,
            member_name=member_name,
            line_number=line_number,
            history_present=False,
            finished=False,
            error_hash=error_hash,
            error_chars=error_chars,
            terminal_source_timestamp=None,
            task_usage=_Usage.missing(),
            token_usages=(),
            history_llm_metrics_count=0,
            tools=(),
        )
    if not isinstance(history, list) or not history:
        raise OpenHandsArchiveSchemaError(
            "output.jsonl history must be a non-empty array or null"
        )
    if not isinstance(value.get("instruction"), str):
        raise OpenHandsArchiveSchemaError("task instruction must be a string")
    metadata = _require_mapping(value.get("metadata"), "task metadata")
    if _non_negative_int(metadata.get("max_iterations"), "metadata.max_iterations") != (
        run.max_iterations
    ):
        raise OpenHandsArchiveSchemaError(
            "task metadata max_iterations disagrees with its run directory"
        )
    instance = _require_mapping(value.get("instance"), "task instance")
    if instance.get("instance_id") != task_id:
        raise OpenHandsArchiveSchemaError(
            "task instance identity disagrees with output.jsonl root"
        )
    task_usage, token_usages = _parse_task_metrics(value.get("metrics"))
    tools, finished, terminal_timestamp, llm_metrics_count = _parse_history(history)
    if finished and raw_error is not None:
        raise OpenHandsArchiveSchemaError(
            "task cannot contain both a finish action and a top-level error"
        )
    if not finished and raw_error is None:
        # The log is still usable, but its lifecycle remains explicitly unknown.
        terminal_timestamp = _history_timestamp(history[-1], "last history event")
    return _TaskLogSummary(
        task_id=task_id,
        member_name=member_name,
        line_number=line_number,
        history_present=True,
        finished=finished,
        error_hash=error_hash,
        error_chars=error_chars,
        terminal_source_timestamp=terminal_timestamp,
        task_usage=task_usage,
        token_usages=token_usages,
        history_llm_metrics_count=llm_metrics_count,
        tools=tools,
    )


def _parse_task_metrics(value: Any) -> tuple[_Usage, tuple[tuple[str, _Usage], ...]]:
    metrics = _require_mapping(value, "task metrics")
    _validate_exact_keys(
        metrics,
        {
            "accumulated_cost",
            "max_budget_per_task",
            "accumulated_token_usage",
            "costs",
            "response_latencies",
            "token_usages",
            "condenser",
        },
        context="task metrics",
    )
    _finite_number(metrics.get("accumulated_cost"), "metrics.accumulated_cost")
    budget = metrics.get("max_budget_per_task")
    if budget is not None:
        _finite_number(budget, "metrics.max_budget_per_task")
    task_usage, _ = _parse_metric_usage(
        metrics.get("accumulated_token_usage"),
        "metrics.accumulated_token_usage",
        allow_empty_response_id=True,
    )
    raw_token_usages = metrics.get("token_usages")
    if not isinstance(raw_token_usages, list):
        raise OpenHandsArchiveSchemaError("metrics.token_usages must be an array")
    token_usages: list[tuple[str, _Usage]] = []
    seen_response_ids: set[str] = set()
    for index, raw_usage in enumerate(raw_token_usages):
        usage, response_id = _parse_metric_usage(
            raw_usage,
            f"metrics.token_usages[{index}]",
            allow_empty_response_id=False,
        )
        if response_id in seen_response_ids:
            raise OpenHandsArchiveSchemaError(
                "metrics.token_usages repeats a response_id"
            )
        seen_response_ids.add(response_id)
        token_usages.append((response_id, usage))
    costs = metrics.get("costs")
    latencies = metrics.get("response_latencies")
    condenser = metrics.get("condenser")
    if not isinstance(costs, list) or not isinstance(latencies, list):
        raise OpenHandsArchiveSchemaError(
            "metrics costs and response_latencies must be arrays"
        )
    if not isinstance(condenser, list):
        raise OpenHandsArchiveSchemaError("metrics.condenser must be an array")
    if len(costs) != len(token_usages) or len(latencies) != len(token_usages):
        raise OpenHandsArchiveSchemaError(
            "task metric ledgers have inconsistent lengths"
        )
    for index, raw_cost in enumerate(costs):
        item = _require_mapping(raw_cost, f"metrics.costs[{index}]")
        _validate_exact_keys(
            item,
            {"model", "cost", "timestamp"},
            context=f"metrics.costs[{index}]",
        )
        _required_text(item, "model", f"metrics.costs[{index}]")
        _finite_number(item.get("cost"), f"metrics.costs[{index}].cost")
        _finite_number(item.get("timestamp"), f"metrics.costs[{index}].timestamp")
    for index, raw_latency in enumerate(latencies):
        item = _require_mapping(raw_latency, f"metrics.response_latencies[{index}]")
        _validate_exact_keys(
            item,
            {"model", "latency", "response_id"},
            context=f"metrics.response_latencies[{index}]",
        )
        _required_text(item, "model", f"metrics.response_latencies[{index}]")
        _required_text(item, "response_id", f"metrics.response_latencies[{index}]")
        _finite_number(
            item.get("latency"), f"metrics.response_latencies[{index}].latency"
        )
    ledger_input = sum(int(item.input_tokens or 0) for _, item in token_usages)
    ledger_output = sum(int(item.output_tokens or 0) for _, item in token_usages)
    ledger_cached = sum(int(item.cached_input_tokens or 0) for _, item in token_usages)
    ledger_cache_write = sum(
        int(item.cache_write_input_tokens or 0) for _, item in token_usages
    )
    if (
        ledger_input != task_usage.input_tokens
        or ledger_output != task_usage.output_tokens
        or ledger_cached != task_usage.cached_input_tokens
        or ledger_cache_write != task_usage.cache_write_input_tokens
    ):
        raise OpenHandsArchiveSchemaError(
            "metrics token ledger sum disagrees with accumulated usage"
        )
    return task_usage, tuple(token_usages)


def _parse_metric_usage(
    value: Any,
    context: str,
    *,
    allow_empty_response_id: bool,
) -> tuple[_Usage, str]:
    usage = _require_mapping(value, context)
    _validate_exact_keys(
        usage,
        {
            "model",
            "prompt_tokens",
            "completion_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "context_window",
            "per_turn_token",
            "response_id",
        },
        context=context,
    )
    _required_text(usage, "model", context)
    response_id = usage.get("response_id")
    if not isinstance(response_id, str) or (not allow_empty_response_id and not response_id):
        raise OpenHandsArchiveSchemaError(f"{context}.response_id has invalid shape")
    input_tokens = _non_negative_int(usage.get("prompt_tokens"), f"{context}.prompt")
    output_tokens = _non_negative_int(
        usage.get("completion_tokens"), f"{context}.completion"
    )
    cached = _non_negative_int(
        usage.get("cache_read_tokens"), f"{context}.cache_read"
    )
    cache_write = _non_negative_int(
        usage.get("cache_write_tokens"), f"{context}.cache_write"
    )
    _non_negative_int(usage.get("context_window"), f"{context}.context_window")
    _non_negative_int(usage.get("per_turn_token"), f"{context}.per_turn_token")
    if cached > input_tokens:
        raise OpenHandsArchiveSchemaError(f"{context} cached tokens exceed prompt tokens")
    return (
        _Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cached_input_tokens=cached,
            cache_write_input_tokens=cache_write,
            reasoning_output_tokens=None,
            image_output_tokens=None,
            reasoning_subset_valid=None,
        ),
        response_id,
    )


@dataclass(frozen=True, slots=True)
class _HistoryAction:
    event_id: int
    history_index: int
    timestamp: str
    action: str
    tool_call_id: str | None
    public_tool_call_id: str | None
    tool_name: str | None


@dataclass(frozen=True, slots=True)
class _HistoryObservation:
    event_id: int
    history_index: int
    timestamp: str
    observation: str
    cause: int | None
    tool_call_id: str | None
    output: _ContentSummary
    success: bool | None


def _parse_history(
    history: list[Any],
) -> tuple[tuple[_HistoryTool, ...], bool, str, int]:
    allowed_fields = {
        "id",
        "timestamp",
        "source",
        "message",
        "action",
        "args",
        "cause",
        "observation",
        "content",
        "extras",
        "tool_call_metadata",
        "llm_metrics",
        "timeout",
        "success",
    }
    allowed_actions = {
        "system",
        "message",
        "recall",
        "task_tracking",
        "run",
        "read",
        "edit",
        "finish",
        "think",
    }
    allowed_observations = {
        "recall",
        "task_tracking",
        "run",
        "read",
        "edit",
        "think",
        "error",
    }
    actions: dict[int, _HistoryAction] = {}
    observations: list[_HistoryObservation] = []
    event_ids: list[int] = []
    llm_metrics_count = 0
    finish_indexes: list[int] = []
    tool_ids: set[str] = set()
    for index, raw_event in enumerate(history):
        event = _require_mapping(raw_event, f"history[{index}]")
        unexpected = set(event) - allowed_fields
        if unexpected:
            raise OpenHandsArchiveSchemaError(
                f"history[{index}] has unsupported fields: {sorted(unexpected)!r}"
            )
        event_id = _non_negative_int(event.get("id"), f"history[{index}].id")
        event_ids.append(event_id)
        timestamp = _history_timestamp(event, f"history[{index}]")
        if not isinstance(event.get("source"), str) or not isinstance(
            event.get("message"), str
        ):
            raise OpenHandsArchiveSchemaError(
                "history source/message fields must be strings"
            )
        is_action = "action" in event
        is_observation = "observation" in event
        if is_action == is_observation:
            raise OpenHandsArchiveSchemaError(
                "history event must be exactly one action or observation"
            )
        if "llm_metrics" in event:
            _require_mapping(event.get("llm_metrics"), f"history[{index}].llm_metrics")
            llm_metrics_count += 1
        if is_action:
            action = event.get("action")
            if action not in allowed_actions:
                raise OpenHandsArchiveSchemaError("history contains an unknown action class")
            _require_mapping(event.get("args"), f"history[{index}].args")
            tool_call_id = public_id = tool_name = None
            if "tool_call_metadata" in event:
                tool_call_id, public_id, tool_name = _parse_tool_call_metadata(
                    event.get("tool_call_metadata"), f"history[{index}]"
                )
                if tool_call_id in tool_ids:
                    raise OpenHandsArchiveSchemaError(
                        "history repeats a tool_call_id across actions"
                    )
                tool_ids.add(tool_call_id)
            if "timeout" in event:
                if action != "run":
                    raise OpenHandsArchiveSchemaError(
                        "history timeout is only supported on run actions"
                    )
                _finite_number(event.get("timeout"), f"history[{index}].timeout")
            if "success" in event:
                raise OpenHandsArchiveSchemaError("action cannot contain success")
            if action == "finish":
                finish_indexes.append(index)
            actions[event_id] = _HistoryAction(
                event_id=event_id,
                history_index=index,
                timestamp=timestamp,
                action=str(action),
                tool_call_id=tool_call_id,
                public_tool_call_id=public_id,
                tool_name=tool_name,
            )
        else:
            observation = event.get("observation")
            if observation not in allowed_observations:
                raise OpenHandsArchiveSchemaError(
                    "history contains an unknown observation class"
                )
            content = event.get("content")
            if not isinstance(content, str):
                raise OpenHandsArchiveSchemaError(
                    "history observation content must be a string"
                )
            _require_mapping(event.get("extras"), f"history[{index}].extras")
            cause = event.get("cause")
            if cause is not None:
                cause = _non_negative_int(cause, f"history[{index}].cause")
            elif observation != "error":
                raise OpenHandsArchiveSchemaError(
                    "non-error observation must identify its cause action"
                )
            tool_call_id = None
            if "tool_call_metadata" in event:
                tool_call_id, _, _ = _parse_tool_call_metadata(
                    event.get("tool_call_metadata"), f"history[{index}]"
                )
            success = event.get("success")
            if success is not None and not isinstance(success, bool):
                raise OpenHandsArchiveSchemaError(
                    "history observation success must be boolean"
                )
            if success is not None and observation != "run":
                raise OpenHandsArchiveSchemaError(
                    "explicit success is only supported on run observations"
                )
            observations.append(
                _HistoryObservation(
                    event_id=event_id,
                    history_index=index,
                    timestamp=timestamp,
                    observation=str(observation),
                    cause=cause,
                    tool_call_id=tool_call_id,
                    output=_plain_text_summary(content),
                    success=success,
                )
            )
    if len(set(event_ids)) != len(event_ids) or any(
        left >= right for left, right in zip(event_ids, event_ids[1:])
    ):
        raise OpenHandsArchiveSchemaError(
            "history event ids must be unique and strictly increasing"
        )
    if len(finish_indexes) > 1:
        raise OpenHandsArchiveSchemaError("history contains multiple finish actions")
    finished = bool(finish_indexes)
    if finished and finish_indexes[0] != len(history) - 1:
        raise OpenHandsArchiveSchemaError("finish action must be the final history event")

    observations_by_cause: dict[int, _HistoryObservation] = {}
    for observation in observations:
        if observation.cause is None:
            continue
        if observation.cause not in actions:
            raise OpenHandsArchiveSchemaError(
                "history observation cause does not reference an action"
            )
        if observation.cause in observations_by_cause:
            raise OpenHandsArchiveSchemaError(
                "one history action has multiple terminal observations"
            )
        observations_by_cause[observation.cause] = observation

    tools: list[_HistoryTool] = []
    for action in actions.values():
        if action.tool_call_id is None or action.action == "finish":
            continue
        observation = observations_by_cause.get(action.event_id)
        if observation is None:
            raise OpenHandsArchiveSchemaError(
                "tool action lacks a terminal observation"
            )
        if observation.observation not in {action.action, "error"}:
            raise OpenHandsArchiveSchemaError(
                "tool action and observation classes disagree"
            )
        if (
            observation.tool_call_id is not None
            and observation.tool_call_id != action.tool_call_id
        ):
            raise OpenHandsArchiveSchemaError(
                "tool action/observation identifiers disagree"
            )
        failed = observation.observation == "error" or observation.success is False
        evidence = (
            "error_observation"
            if observation.observation == "error"
            else "success_false"
            if observation.success is False
            else None
        )
        tools.append(
            _HistoryTool(
                source_action_id=action.event_id,
                source_observation_id=observation.event_id,
                source_action_timestamp=action.timestamp,
                source_observation_timestamp=observation.timestamp,
                tool_call_id=action.tool_call_id,
                public_tool_call_id=str(action.public_tool_call_id),
                tool_name=str(action.tool_name),
                output=observation.output,
                failed=failed,
                failure_evidence=evidence,
                action_history_index=action.history_index,
                observation_history_index=observation.history_index,
            )
        )
    terminal_timestamp = _history_timestamp(history[-1], "last history event")
    return tuple(tools), finished, terminal_timestamp, llm_metrics_count


def _parse_tool_call_metadata(value: Any, context: str) -> tuple[str, str, str]:
    metadata = _require_mapping(value, f"{context}.tool_call_metadata")
    _validate_exact_keys(
        metadata,
        {"function_name", "tool_call_id", "model_response", "total_calls_in_response"},
        context=f"{context}.tool_call_metadata",
    )
    function_name = _required_text(
        metadata, "function_name", f"{context}.tool_call_metadata"
    )
    tool_call_id = _required_text(
        metadata, "tool_call_id", f"{context}.tool_call_metadata"
    )
    _non_negative_int(
        metadata.get("total_calls_in_response"),
        f"{context}.tool_call_metadata.total_calls_in_response",
    )
    model_response = _require_mapping(
        metadata.get("model_response"), f"{context}.tool_call_metadata.model_response"
    )
    _required_text(
        model_response, "id", f"{context}.tool_call_metadata.model_response"
    )
    return tool_call_id, _public_tool_call_id(tool_call_id), function_name


def _history_timestamp(value: Mapping[str, Any], context: str) -> str:
    raw = value.get("timestamp")
    if not isinstance(raw, str):
        raise OpenHandsArchiveSchemaError(f"{context}.timestamp must be a string")
    try:
        datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise OpenHandsArchiveSchemaError(
            f"{context}.timestamp must be ISO-8601"
        ) from exc
    return raw


def _prepare_task_snapshots(
    *,
    run: _RunDescriptor,
    task_id: str,
    snapshots: Sequence[_CompletionSnapshot],
) -> _PreparedTask:
    if not snapshots:
        raise OpenHandsArchiveSchemaError("run/task completion group is empty")
    ordered = _validate_and_sort_snapshots(snapshots)
    (
        transitions,
        reset_count,
        repeated_count,
        not_materialized_count,
    ) = _validate_message_transitions(ordered)
    tool_config_hashes = {snapshot.tool_config_hash for snapshot in ordered}
    filename_models = {snapshot.filename_model for snapshot in ordered}
    filename_providers = {snapshot.filename_provider for snapshot in ordered}
    resolved_models = {snapshot.response.model for snapshot in ordered}
    if len(tool_config_hashes) != 1:
        raise OpenHandsArchiveSchemaError(
            "tool configuration changes within one task trajectory"
        )
    if (
        filename_models != {run.model}
        or len(filename_providers) != 1
        or len(resolved_models) != 1
    ):
        raise OpenHandsArchiveSchemaError(
            "configured filename identity or resolved response model changes "
            "within one task trajectory"
        )
    calls: list[_CallSnapshot] = []
    for snapshot in ordered:
        role_counts = Counter(message.role for message in snapshot.messages)
        calls.append(
            _CallSnapshot(
                member_name=snapshot.member_name,
                filename_timestamp=snapshot.filename_timestamp,
                top_timestamp=snapshot.top_timestamp,
                configured_provider=snapshot.filename_provider,
                request_message_count=len(snapshot.messages),
                request_role_counts=tuple(sorted(role_counts.items())),
                request_content_chars=sum(
                    message.content.chars for message in snapshot.messages
                ),
                request_content_hash=_semantic_hash(
                    [message.fingerprint for message in snapshot.messages]
                ),
                response=snapshot.response,
            )
        )
    fallback_tools = tuple(
        tuple(
            _FallbackToolResult(
                message_index=message_index,
                tool_call_id=str(message.tool_call_id),
                public_tool_call_id=str(message.public_tool_call_id),
                tool_name=str(message.tool_name),
                content=message.content,
            )
            for message_index, message in transition
        )
        for transition in transitions
    )
    return _PreparedTask(
        run=run,
        task_id=task_id,
        tool_config_hash=next(iter(tool_config_hashes)),
        configured_provider=next(iter(filename_providers)),
        resolved_model=next(iter(resolved_models)),
        calls=tuple(calls),
        fallback_tools=fallback_tools,
        message_prefix_reset_count=reset_count,
        repeated_request_snapshot_count=repeated_count,
        response_not_materialized_count=not_materialized_count,
    )


def _finalize_run(
    *,
    source_name: str,
    run: _RunDescriptor,
    prepared_tasks: Mapping[str, _PreparedTask],
    task_logs: Mapping[str, _TaskLogSummary],
    reports: Mapping[str, _ReportSnapshot],
    aggregate_report: _AggregateReport | None,
    metadata: OpenHandsArchiveMetadata,
    archive_identity: str,
    archive_identity_hash: str,
    archive_identity_source: str,
) -> Iterator[Trajectory]:
    completion_ids = set(prepared_tasks)
    log_ids = set(task_logs)
    report_ids = set(reports)
    if task_logs:
        if completion_ids - log_ids:
            raise OpenHandsArchiveSchemaError(
                "completion task is missing from output.jsonl"
            )
        if report_ids - log_ids:
            raise OpenHandsArchiveSchemaError(
                "task report is missing from output.jsonl"
            )
        task_ids = log_ids
    else:
        if report_ids - completion_ids:
            raise OpenHandsArchiveSchemaError(
                "archive contains report-only task-runs without task logs"
            )
        task_ids = completion_ids
    if aggregate_report is not None:
        evidenced = log_ids if task_logs else completion_ids
        if aggregate_report.submitted_ids != evidenced:
            raise OpenHandsArchiveSchemaError(
                "aggregate submitted task set disagrees with task evidence"
            )
    if not task_ids:
        raise OpenHandsArchiveSchemaError("run contains no normalizable task-runs")

    tool_hashes = {item.tool_config_hash for item in prepared_tasks.values()}
    configured_providers = {
        item.configured_provider for item in prepared_tasks.values()
    }
    resolved_models = {
        item.resolved_model
        for item in prepared_tasks.values()
        if item.calls and item.resolved_model is not None
    }
    if not tool_hashes or len(tool_hashes) != 1:
        raise OpenHandsArchiveSchemaError(
            "run does not expose one stable completion tool configuration"
        )
    if len(configured_providers) != 1:
        raise OpenHandsArchiveSchemaError(
            "run does not expose one stable configured filename provider"
        )
    if len(resolved_models) != 1:
        raise OpenHandsArchiveSchemaError(
            "run does not expose one stable resolved response model"
        )
    default_tool_hash = next(iter(tool_hashes))
    default_configured_provider = next(iter(configured_providers))
    for task_id in sorted(task_ids):
        prepared = prepared_tasks.get(task_id)
        if prepared is None:
            prepared = _PreparedTask(
                run=run,
                task_id=task_id,
                tool_config_hash=default_tool_hash,
                configured_provider=default_configured_provider,
                resolved_model=None,
                calls=(),
                fallback_tools=(),
                message_prefix_reset_count=0,
                repeated_request_snapshot_count=0,
                response_not_materialized_count=0,
            )
        yield _normalize_prepared_task(
            source_name=source_name,
            prepared=prepared,
            task_log=task_logs.get(task_id),
            report=reports.get(task_id),
            metadata=metadata,
            archive_identity=archive_identity,
            archive_identity_hash=archive_identity_hash,
            archive_identity_source=archive_identity_source,
        )


def _normalize_task(
    *,
    source_name: str,
    run: _RunDescriptor,
    task_id: str,
    snapshots: Sequence[_CompletionSnapshot],
    report: _ReportSnapshot | None,
    metadata: OpenHandsArchiveMetadata,
    archive_identity: str,
    archive_identity_hash: str,
    archive_identity_source: str,
) -> Trajectory:
    """Completion-only compatibility path used when output.jsonl is absent."""

    prepared = _prepare_task_snapshots(run=run, task_id=task_id, snapshots=snapshots)
    return _normalize_prepared_task(
        source_name=source_name,
        prepared=prepared,
        task_log=None,
        report=report,
        metadata=metadata,
        archive_identity=archive_identity,
        archive_identity_hash=archive_identity_hash,
        archive_identity_source=archive_identity_source,
    )


def _normalize_prepared_task(
    *,
    source_name: str,
    prepared: _PreparedTask,
    task_log: _TaskLogSummary | None,
    report: _ReportSnapshot | None,
    metadata: OpenHandsArchiveMetadata,
    archive_identity: str,
    archive_identity_hash: str,
    archive_identity_source: str,
) -> Trajectory:
    if bool(prepared.calls) != (prepared.resolved_model is not None):
        raise OpenHandsArchiveSchemaError(
            "realized response model must exist exactly when completion calls exist"
        )
    run = prepared.run
    task_id = prepared.task_id
    tool_config_hash = prepared.tool_config_hash
    # Task-pre identity may use only configured facts.  Realized response route
    # and model are completion-time observations and therefore stay out of this
    # semantic and the TASK_STARTED payload below.
    condition_semantic = {
        "model": run.model,
        "openhands_version": run.openhands_version,
        "max_iterations": run.max_iterations,
        "hint_mode": run.hint_mode,
        "tool_config_hash": tool_config_hash,
    }
    condition_id = f"condition:{_semantic_hash(condition_semantic)[:20]}"
    trajectory_semantic = {
        "source_id": OpenHandsArchiveReader.source_id,
        "model": run.model,
        "run_id": run.run_id,
        "task_id": task_id,
        "archive_identity": archive_identity,
    }
    trajectory_id = f"openhands:trajectory:{_semantic_hash(trajectory_semantic)[:32]}"
    alignment = _validate_task_log_alignment(prepared, task_log)
    history_tools = _history_tools_by_call(prepared, task_log)
    events: list[CanonicalEvent] = []
    base_time = datetime(2000, 1, 1, tzinfo=UTC)

    def append(
        event_type: EventType,
        *,
        payload: Mapping[str, Any],
        raw_ref: str,
        logical_call_id: str | None = None,
        attempt_id: str | None = None,
    ) -> None:
        seq = len(events)
        events.append(
            CanonicalEvent.create(
                schema_version=1,
                event_id=f"{trajectory_id}:event:{seq}",
                trajectory_id=trajectory_id,
                event_seq=seq,
                event_type=event_type,
                occurred_at=(base_time + timedelta(microseconds=seq)).isoformat(),
                payload=payload,
                logical_call_id=logical_call_id,
                attempt_id=attempt_id,
                raw_ref=raw_ref,
            )
        )

    if prepared.calls:
        first_ref = f"{source_name}!/{prepared.calls[0].member_name}"
    elif task_log is not None:
        first_ref = (
            f"{source_name}!/{task_log.member_name}#/line/{task_log.line_number}"
        )
    else:
        raise OpenHandsArchiveSchemaError("task has neither completion nor task-log evidence")
    append(
        EventType.TASK_STARTED,
        payload={
            "task_id": task_id,
            "run_id": run.run_id,
            "trajectory_id": trajectory_id,
            "prediction_context_id": f"{task_id}:initial",
            "condition_id": condition_id,
            "benchmark_id": metadata.benchmark_id,
            "model_id": run.model,
            "agent_id": "OpenHands",
            "agent_version": run.openhands_version,
            "max_steps": run.max_iterations,
            "hint_mode": run.hint_mode,
            "tool_config_hash": tool_config_hash,
            "archive_identity_hash": archive_identity_hash,
            "archive_identity_source": archive_identity_source,
            "canonical_time_source": "synthetic_order_only",
            "task_tokens": None,
        },
        raw_ref=first_ref,
    )

    for index, snapshot in enumerate(prepared.calls):
        call_id = f"{trajectory_id}:call:{index}"
        attempt_id = f"{call_id}:attempt:0"
        raw_ref = f"{source_name}!/{snapshot.member_name}"
        append(
            EventType.REQUEST_BUILT,
            logical_call_id=call_id,
            raw_ref=raw_ref,
            payload={
                "request_tokens_local": None,
                "request_token_count_source": "missing",
                "prediction_context_id": f"{task_id}:call:{index}",
                "request_message_count": snapshot.request_message_count,
                "request_role_counts": dict(snapshot.request_role_counts),
                "request_content_chars": snapshot.request_content_chars,
                "request_content_hash": snapshot.request_content_hash,
                "configured_provider": snapshot.configured_provider,
                "configured_model": run.model,
                "tool_config_hash": tool_config_hash,
                "source_filename_timestamp": str(snapshot.filename_timestamp),
                "source_log_timestamp": str(snapshot.top_timestamp),
            },
        )
        append(
            EventType.API_ATTEMPT_STARTED,
            logical_call_id=call_id,
            attempt_id=attempt_id,
            raw_ref=raw_ref,
            payload={
                "configured_provider": snapshot.configured_provider,
                "configured_model": run.model,
                "tool_config_hash": tool_config_hash,
            },
        )
        append(
            EventType.API_COMPLETED,
            logical_call_id=call_id,
            attempt_id=attempt_id,
            raw_ref=raw_ref,
            payload={
                "usage": snapshot.response.usage.to_payload(),
                "provider": snapshot.response.provider,
                "resolved_model": snapshot.response.model,
                "finish_reason": snapshot.response.finish_reason,
                "response_id_hash": snapshot.response.response_id_hash,
                "source_response_created": snapshot.response.created,
                "retry_observable": False,
                "attempt_ordinal_source": "one_completion_one_observed_attempt",
                "output_content_hash": snapshot.response.content.content_hash,
                "output_content_chars": snapshot.response.content.chars,
                "output_content_bytes": snapshot.response.content.bytes,
                "reasoning_content_hash": (
                    snapshot.response.reasoning.content_hash
                    if snapshot.response.reasoning is not None
                    else None
                ),
                "reasoning_content_chars": (
                    snapshot.response.reasoning.chars
                    if snapshot.response.reasoning is not None
                    else None
                ),
                "tool_call_count": len(snapshot.response.tool_calls),
                "usage_scope": "current_response_only",
                "error_observable": False,
                "provider_error_envelope_present": (
                    snapshot.response.provider_error_envelope
                ),
                "generation_checkpoint_observable": False,
            },
        )

        if task_log is not None:
            for tool in history_tools.get(index, ()):
                base_ref = (
                    f"{source_name}!/{task_log.member_name}#/line/"
                    f"{task_log.line_number}/history"
                )
                append(
                    EventType.TOOL_STARTED,
                    logical_call_id=call_id,
                    raw_ref=f"{base_ref}/{tool.action_history_index}",
                    payload={
                        "tool_call_id": tool.public_tool_call_id,
                        "tool_call_id_source": "sha256_redacted",
                        "tool_name": tool.tool_name,
                        "source_event_id": tool.source_action_id,
                        "source_timestamp": tool.source_action_timestamp,
                    },
                )
                append(
                    EventType.TOOL_FAILED if tool.failed else EventType.TOOL_COMPLETED,
                    logical_call_id=call_id,
                    raw_ref=f"{base_ref}/{tool.observation_history_index}",
                    payload={
                        "tool_call_id": tool.public_tool_call_id,
                        "tool_call_id_source": "sha256_redacted",
                        "tool_name": tool.tool_name,
                        "source_event_id": tool.source_observation_id,
                        "source_cause_event_id": tool.source_action_id,
                        "source_timestamp": tool.source_observation_timestamp,
                        "output_content_hash": tool.output.content_hash,
                        "output_content_chars": tool.output.chars,
                        "output_content_bytes": tool.output.bytes,
                        "failure_observable": True,
                        "failure_evidence": tool.failure_evidence,
                    },
                )
        elif index < len(prepared.fallback_tools):
            following = prepared.calls[index + 1]
            for tool in prepared.fallback_tools[index]:
                append(
                    EventType.TOOL_COMPLETED,
                    logical_call_id=call_id,
                    raw_ref=(
                        f"{source_name}!/{following.member_name}"
                        f"#/messages/{tool.message_index}"
                    ),
                    payload={
                        "tool_call_id": tool.public_tool_call_id,
                        "tool_call_id_source": "sha256_redacted",
                        "tool_name": tool.tool_name,
                        "output_content_hash": tool.content.content_hash,
                        "output_content_chars": tool.content.chars,
                        "output_content_bytes": tool.content.bytes,
                        "failure_observable": False,
                    },
                )

    terminal_usage, terminal_usage_source, terminal_usage_scope = _task_terminal_usage(
        prepared,
        task_log,
    )
    terminal_ref = (
        f"{source_name}!/{task_log.member_name}#/line/{task_log.line_number}"
        if task_log is not None
        else f"{source_name}!/{prepared.calls[-1].member_name}"
    )
    if task_log is not None and task_log.finished:
        terminal_type = EventType.TASK_FINISHED
        outcome = "finished"
        reason = "agent_finished"
    elif task_log is not None and task_log.error_hash is not None:
        terminal_type = EventType.TASK_ABORTED
        outcome = "error"
        reason = "task_error"
    else:
        terminal_type = EventType.TASK_ABORTED
        outcome = "unknown"
        reason = "logging_incomplete"
    append(
        terminal_type,
        raw_ref=terminal_ref,
        payload={
            "outcome": outcome,
            "reason": reason,
            "usage": terminal_usage.to_payload(
                total_source=terminal_usage_source,
                usage_scope=terminal_usage_scope,
            ),
            "usage_scope": terminal_usage_scope,
            "known_usage_attempts": sum(
                snapshot.response.usage.complete for snapshot in prepared.calls
            ),
            "missing_usage_attempts": sum(
                not snapshot.response.usage.complete for snapshot in prepared.calls
            ),
            "completion_snapshot_count": len(prepared.calls),
            "reasoning_subset_anomaly_count": sum(
                call.response.usage.reasoning_subset_valid is False
                for call in prepared.calls
            ),
            "provider_error_envelope_count": sum(
                call.response.provider_error_envelope for call in prepared.calls
            ),
            "message_prefix_reset_count": prepared.message_prefix_reset_count,
            "repeated_request_snapshot_count": prepared.repeated_request_snapshot_count,
            "response_not_materialized_in_next_request_count": (
                prepared.response_not_materialized_count
            ),
            "completion_logging_complete": alignment["completion_logging_complete"],
            "task_usage_reconciled": alignment["task_usage_reconciled"],
            "metrics_completion_extra_count": alignment[
                "metrics_completion_extra_count"
            ],
            "metrics_missing_completion_count": alignment[
                "metrics_missing_completion_count"
            ],
            "metrics_scope": (
                "missing"
                if task_log is None
                else "current_output_session_without_completion_boundaries"
                if alignment["metrics_missing_completion_count"]
                else "current_output_session_only"
                if alignment["metrics_completion_extra_count"]
                else "all_preserved_sessions"
            ),
            "history_llm_metrics_count_matches_ledger": alignment[
                "history_llm_metrics_count_matches_ledger"
            ],
            "history_present": task_log.history_present if task_log is not None else False,
            "source_terminal_timestamp": (
                task_log.terminal_source_timestamp if task_log is not None else None
            ),
            "task_error_hash": task_log.error_hash if task_log is not None else None,
            "task_error_chars": task_log.error_chars if task_log is not None else None,
            "evaluator_report": report.to_payload() if report is not None else None,
            "evaluator_report_present": report is not None,
            "lifecycle_source": "output.jsonl" if task_log is not None else "missing",
        },
    )
    return Trajectory.from_events(events)


def _task_terminal_usage(
    prepared: _PreparedTask,
    task_log: _TaskLogSummary | None,
) -> tuple[_Usage, str, str]:
    completion_aggregate = _aggregate_usage(
        [call.response.usage for call in prepared.calls]
    )
    if task_log is None or not task_log.task_usage.complete:
        if not prepared.calls:
            return (
                _Usage.missing(),
                "missing",
                "missing_no_completion_or_task_metrics",
            )
        return (
            completion_aggregate,
            "derived_complete_attempt_sum_all_preserved_sessions",
            (
                "all_preserved_sessions"
                if completion_aggregate.complete
                else "missing_due_to_incomplete_attempt_usage"
            ),
        )

    ledger_ids = {response_id for response_id, _ in task_log.token_usages}
    extras = [
        call.response.usage
        for call in prepared.calls
        if call.response.response_id not in ledger_ids
    ]
    if not extras:
        return (
            task_log.task_usage,
            "output_metrics_accumulated_token_usage",
            (
                "explicit_zero_call_task"
                if not prepared.calls and not ledger_ids
                else "current_output_session_without_completion_boundaries"
                if not prepared.calls and ledger_ids
                else "all_preserved_sessions"
            ),
        )
    if any(not usage.complete for usage in extras):
        return (
            _Usage.missing(),
            "missing",
            "missing_due_to_incomplete_extra_session_usage",
        )
    combined = _combine_usage(task_log.task_usage, extras)
    return (
        combined,
        "output_metrics_plus_complete_completion_extras",
        "all_preserved_sessions",
    )


def _combine_usage(base: _Usage, extras: Sequence[_Usage]) -> _Usage:
    if not base.complete or any(not usage.complete for usage in extras):
        return _Usage.missing()

    def optional_sum(name: str) -> int | None:
        values = [getattr(base, name), *(getattr(item, name) for item in extras)]
        if any(value is None for value in values):
            return None
        return sum(int(value or 0) for value in values)

    input_tokens = int(base.input_tokens or 0) + sum(
        int(item.input_tokens or 0) for item in extras
    )
    output_tokens = int(base.output_tokens or 0) + sum(
        int(item.output_tokens or 0) for item in extras
    )
    return _Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cached_input_tokens=optional_sum("cached_input_tokens"),
        cache_write_input_tokens=optional_sum("cache_write_input_tokens"),
        reasoning_output_tokens=optional_sum("reasoning_output_tokens"),
        image_output_tokens=optional_sum("image_output_tokens"),
        reasoning_subset_valid=(
            all(
                item.reasoning_subset_valid is not False
                for item in (base, *extras)
            )
            if any(
                item.reasoning_subset_valid is not None
                for item in (base, *extras)
            )
            else None
        ),
    )


def _validate_task_log_alignment(
    prepared: _PreparedTask,
    task_log: _TaskLogSummary | None,
) -> dict[str, bool | int]:
    if task_log is None:
        return {
            "completion_logging_complete": bool(prepared.calls),
            "task_usage_reconciled": False,
            "history_llm_metrics_count_matches_ledger": False,
            "metrics_completion_extra_count": 0,
            "metrics_missing_completion_count": 0,
        }
    ledger = dict(task_log.token_usages)
    history_metrics_match = task_log.history_llm_metrics_count == len(ledger)
    if not prepared.calls:
        return {
            "completion_logging_complete": not ledger,
            "task_usage_reconciled": not ledger,
            "history_llm_metrics_count_matches_ledger": history_metrics_match,
            "metrics_completion_extra_count": 0,
            "metrics_missing_completion_count": len(ledger),
        }
    call_ids = {call.response.response_id for call in prepared.calls}
    ledger_ids = set(ledger)
    if ledger_ids - call_ids:
        raise OpenHandsArchiveSchemaError(
            "metrics.token_usages contains an id absent from completions"
        )
    for call in prepared.calls:
        if call.response.response_id not in ledger:
            continue
        response_usage = call.response.usage
        metric_usage = ledger[call.response.response_id]
        if response_usage.complete and (
            response_usage.input_tokens != metric_usage.input_tokens
            or response_usage.output_tokens != metric_usage.output_tokens
            or response_usage.cached_input_tokens != metric_usage.cached_input_tokens
        ):
            raise OpenHandsArchiveSchemaError(
                "current-response usage disagrees with output.jsonl token ledger"
            )
    exact_session = call_ids == ledger_ids
    extras = [
        call.response.usage
        for call in prepared.calls
        if call.response.response_id not in ledger_ids
    ]
    reconciled = exact_session or all(usage.complete for usage in extras)
    return {
        "completion_logging_complete": True,
        "task_usage_reconciled": reconciled,
        "history_llm_metrics_count_matches_ledger": history_metrics_match,
        "metrics_completion_extra_count": len(call_ids - ledger_ids),
        "metrics_missing_completion_count": 0,
    }


def _history_tools_by_call(
    prepared: _PreparedTask,
    task_log: _TaskLogSummary | None,
) -> dict[int, tuple[_HistoryTool, ...]]:
    if task_log is None:
        return {}
    if not prepared.calls:
        if task_log.tools:
            raise OpenHandsArchiveSchemaError(
                "task has history tool events but no completion call boundaries"
            )
        return {}
    by_tool_id: dict[str, int] = {}
    for call_index, call in enumerate(prepared.calls):
        for tool_call in call.response.tool_calls:
            if tool_call.source_id in by_tool_id:
                raise OpenHandsArchiveSchemaError(
                    "completion responses reuse a tool_call_id"
                )
            by_tool_id[tool_call.source_id] = call_index
    grouped: dict[int, list[_HistoryTool]] = {}
    for tool in task_log.tools:
        call_index = by_tool_id.get(tool.tool_call_id)
        if call_index is None:
            raise OpenHandsArchiveSchemaError(
                "history tool event does not map to a completion response"
            )
        grouped.setdefault(call_index, []).append(tool)
    return {key: tuple(value) for key, value in grouped.items()}


def _validate_and_sort_snapshots(
    snapshots: Sequence[_CompletionSnapshot],
) -> tuple[_CompletionSnapshot, ...]:
    if len({snapshot.raw_sha256 for snapshot in snapshots}) != len(snapshots):
        raise OpenHandsArchiveSchemaError(
            "one run/task contains duplicate raw completion snapshots"
        )
    if len({snapshot.response.response_id for snapshot in snapshots}) != len(snapshots):
        raise OpenHandsArchiveSchemaError(
            "one run/task contains a duplicate response.id"
        )
    fields = (
        ("filename timestamp", lambda item: item.filename_timestamp),
        ("top-level timestamp", lambda item: item.top_timestamp),
        ("response.created", lambda item: item.response_created),
    )
    rankings: list[tuple[str, tuple[int, ...]]] = []
    for label, getter in fields:
        values = [getter(snapshot) for snapshot in snapshots]
        if len(set(values)) != len(values):
            raise OpenHandsArchiveSchemaError(
                f"{label} is not unique within one run/task"
            )
        ranking = tuple(sorted(range(len(snapshots)), key=lambda index: values[index]))
        rankings.append((label, ranking))
    if len({ranking for _, ranking in rankings}) != 1:
        raise OpenHandsArchiveSchemaError(
            "completion timestamp sources disagree on pairwise ordering"
        )
    return tuple(sorted(snapshots, key=lambda snapshot: snapshot.sort_key))


def _validate_message_transitions(
    ordered: Sequence[_CompletionSnapshot],
) -> tuple[
    tuple[tuple[tuple[int, _MessageSnapshot], ...], ...],
    int,
    int,
    int,
]:
    transitions: list[tuple[tuple[int, _MessageSnapshot], ...]] = []
    reset_count = 0
    repeated_count = 0
    not_materialized_count = 0
    for index, (current, following) in enumerate(zip(ordered, ordered[1:])):
        current_fingerprints = tuple(message.fingerprint for message in current.messages)
        following_fingerprints = tuple(
            message.fingerprint for message in following.messages
        )
        if (
            following_fingerprints == current_fingerprints
            and (
                current.response.provider_error_envelope
                or (
                    bool(current.response.tool_calls)
                    and bool(following.response.tool_calls)
                )
            )
        ):
            repeated_count += 1
            not_materialized_count += 1
            transitions.append(())
            continue
        if (
            len(following.messages) == 2
            and tuple(message.role for message in following.messages)
            == ("system", "user")
            and following_fingerprints[: len(current_fingerprints)]
            != current_fingerprints
        ):
            reset_count += 1
            not_materialized_count += 1
            transitions.append(())
            continue
        if (
            len(following_fingerprints) <= len(current_fingerprints)
            or following_fingerprints[: len(current_fingerprints)]
            != current_fingerprints
        ):
            raise OpenHandsArchiveSchemaError(
                "messages are not an exact monotonic prefix extension"
            )
        delta = following.messages[len(current.messages) :]
        if len(delta) == 1 and delta[0].role == "user":
            not_materialized_count += 1
            transitions.append(())
            continue
        if not delta or delta[0].role != "assistant":
            raise OpenHandsArchiveSchemaError(
                "adjacent completion delta does not begin with the prior response"
            )
        _validate_historical_response(delta[0], current.response, index)
        trailing_messages = delta[1:]
        if any(message.role not in {"tool", "user"} for message in trailing_messages):
            raise OpenHandsArchiveSchemaError(
                "adjacent completion delta contains an unsupported role"
            )
        tool_messages = tuple(
            message for message in trailing_messages if message.role == "tool"
        )
        expected = {call.source_id: call for call in current.response.tool_calls}
        if len(tool_messages) != len(expected):
            raise OpenHandsArchiveSchemaError(
                "prior response tool calls do not close in the next request snapshot"
            )
        seen: set[str] = set()
        mapped: list[tuple[int, _MessageSnapshot]] = []
        for offset, message in enumerate(tool_messages, start=len(current.messages) + 1):
            source_id = message.tool_call_id
            if source_id is None or source_id not in expected or source_id in seen:
                raise OpenHandsArchiveSchemaError(
                    "tool result cannot be mapped uniquely to the prior response"
                )
            seen.add(source_id)
            call = expected[source_id]
            if message.tool_name != call.name:
                raise OpenHandsArchiveSchemaError(
                    "tool result name disagrees with the prior tool call"
                )
            mapped.append((offset, message))
        transitions.append(tuple(mapped))
    return tuple(transitions), reset_count, repeated_count, not_materialized_count


def _validate_historical_response(
    historical: _MessageSnapshot,
    response: _ResponseSnapshot,
    call_index: int,
) -> None:
    if historical.content != response.content:
        raise OpenHandsArchiveSchemaError(
            f"historical assistant content disagrees at call {call_index}"
        )
    historical_tools = tuple(
        (
            call.source_id,
            call.name,
            call.arguments_hash,
            call.arguments_chars,
        )
        for call in historical.tool_calls
    )
    response_tools = tuple(
        (
            call.source_id,
            call.name,
            call.arguments_hash,
            call.arguments_chars,
        )
        for call in response.tool_calls
    )
    if historical_tools != response_tools:
        raise OpenHandsArchiveSchemaError(
            f"historical assistant tool calls disagree at call {call_index}"
        )


def _aggregate_usage(usages: Sequence[_Usage]) -> _Usage:
    if not usages or any(not item.complete for item in usages):
        return _Usage.missing()
    return _Usage(
        input_tokens=sum(int(item.input_tokens or 0) for item in usages),
        output_tokens=sum(int(item.output_tokens or 0) for item in usages),
        total_tokens=sum(int(item.total_tokens or 0) for item in usages),
        cached_input_tokens=sum(int(item.cached_input_tokens or 0) for item in usages),
        cache_write_input_tokens=(
            sum(int(item.cache_write_input_tokens or 0) for item in usages)
            if all(item.cache_write_input_tokens is not None for item in usages)
            else None
        ),
        reasoning_output_tokens=sum(
            int(item.reasoning_output_tokens or 0) for item in usages
        ),
        image_output_tokens=sum(int(item.image_output_tokens or 0) for item in usages),
        reasoning_subset_valid=(
            all(item.reasoning_subset_valid is not False for item in usages)
            if any(item.reasoning_subset_valid is not None for item in usages)
            else None
        ),
    )


def _validate_exact_keys(
    value: Mapping[str, Any],
    required: set[str],
    *,
    optional: set[str] | None = None,
    context: str,
) -> None:
    optional = optional or set()
    keys = set(value)
    missing = required - keys
    unexpected = keys - required - optional
    if missing or unexpected:
        raise OpenHandsArchiveSchemaError(
            f"{context} has unsupported fields "
            f"(missing={sorted(missing)!r}, unexpected={sorted(unexpected)!r})"
        )


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OpenHandsArchiveSchemaError(f"{context} must be an object")
    return value


def _required_text(value: Mapping[str, Any], key: str, context: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw:
        raise OpenHandsArchiveSchemaError(f"{context}.{key} must be a non-empty string")
    return raw


def _non_negative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpenHandsArchiveSchemaError(f"{context} must be a non-negative integer")
    return value


def _optional_non_negative_int(value: Any, context: str) -> int | None:
    if value is None:
        return None
    return _non_negative_int(value, context)


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpenHandsArchiveSchemaError(f"{context} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise OpenHandsArchiveSchemaError(f"{context} must be finite")
    return parsed


def _decimal_number(value: Any, context: str) -> Decimal:
    _finite_number(value, context)
    return _decimal_text(str(value), context)


def _decimal_text(value: str, context: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise OpenHandsArchiveSchemaError(f"{context} must be decimal") from exc
    if not parsed.is_finite() or parsed < 0:
        raise OpenHandsArchiveSchemaError(f"{context} must be finite and non-negative")
    return parsed


def _plain_text_summary(value: str) -> _ContentSummary:
    encoded = value.encode("utf-8")
    return _ContentSummary(
        content_hash=hashlib.sha256(encoded).hexdigest(),
        chars=len(value),
        bytes=len(encoded),
    )


def _public_tool_call_id(value: str) -> str:
    return f"tool:{_text_hash(value)[:20]}"


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _semantic_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
