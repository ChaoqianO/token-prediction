from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from token_prediction.collection.openhands_trajectory import (
    OpenHandsArchiveMetadata,
    OpenHandsArchiveReader,
)
from token_prediction.contracts import EventType, TokenUsage
from token_prediction.dataset import (
    DATASET_SCHEMA_VERSION,
    DatasetRow,
    LabelStatus,
    PredictionPosition,
    PredictionTarget,
    build_supervised_dataset,
)
from token_prediction.features import FEATURE_SCHEMA_VERSION
from token_prediction.trajectory import Trajectory


AUDIT_SCHEMA_VERSION = 1
HUB_REPO = "loong0814/openhands_trajectories"
RESOLVED_REVISION = "fa9cbb063f770df596da95af24f7af3b8f595778"
EXPECTED_ARCHIVE_BYTES = 2_908_192_516
EXPECTED_ARCHIVE_SHA256 = (
    "993abcb55aae423f9067d5e6c8e1aeaccf83b9ce31474a215982686527934214"
)
EXPECTED_RUN_IDS = ("run_1", "run_2", "run_3", "run_4")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = (
    REPOSITORY_ROOT
    / "workspace"
    / "external"
    / "spend_your_money"
    / "gpt_5.2_4runs.tar.gz"
)
DEFAULT_INVENTORY = (
    REPOSITORY_ROOT
    / "workspace"
    / "external"
    / "spend_your_money"
    / "gpt_5.2_inventory.json"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "workspace"
    / "external"
    / "spend_your_money"
    / "gpt_5.2_trajectory_audit.json"
)
DEFAULT_SQLITE_PARENT = (
    REPOSITORY_ROOT / "workspace" / "tmp" / "openhands_trajectory_audit"
)
CODE_SOURCE_PATHS = (
    "src/token_prediction/collection/openhands_trajectory.py",
    "src/token_prediction/dataset/builder.py",
    "src/token_prediction/dataset/labels.py",
    "scripts/audit_openhands_trajectory.py",
)
CHUNK_BYTES = 8 * 1024 * 1024

_STATUS_VALUES = tuple(status.value for status in LabelStatus)
_BUILDER_CELLS = frozenset(
    {
        (
            PredictionPosition.TASK_LAUNCH.value,
            PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS.value,
        ),
        (
            PredictionPosition.TASK_PRE.value,
            PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS.value,
        ),
        (
            PredictionPosition.TASK_UPDATE.value,
            PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS.value,
        ),
        (
            PredictionPosition.CALL_PRE.value,
            PredictionTarget.CALL_UNKNOWN_BILLABLE_TOKENS.value,
        ),
        (
            PredictionPosition.CALL_PRE.value,
            PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS.value,
        ),
        (
            PredictionPosition.CALL_PRE.value,
            PredictionTarget.CALL_FINAL_RESPONSE_OUTPUT_TOKENS.value,
        ),
        (
            PredictionPosition.CALL_UPDATE.value,
            PredictionTarget.CALL_REMAINING_OUTPUT_TOKENS.value,
        ),
    }
)
_METRIC_KEYS = (
    "trajectory_count",
    "event_count",
    "logical_call_count",
    "attempt_count",
    "attempt_terminal_count",
    "api_completed_count",
    "api_failed_count",
    "attempt_usage_complete_count",
    "attempt_usage_missing_count",
    "attempt_usage_invalid_count",
    "call_usage_complete_count",
    "call_usage_missing_count",
    "call_usage_invalid_count",
    "task_usage_complete_count",
    "task_usage_missing_count",
    "task_usage_invalid_count",
    "request_count",
    "request_tokens_local_observed_count",
    "request_tokens_local_missing_count",
    "generation_checkpoint_count",
    "tool_started_count",
    "tool_completed_count",
    "tool_failed_count",
    "tool_terminal_failure_observable_count",
    "tool_terminal_failure_unobservable_count",
    "task_finished_count",
    "task_aborted_count",
    "task_error_count",
    "task_lifecycle_observed_count",
    "task_lifecycle_censored_count",
    "task_log_observed_count",
    "task_log_missing_count",
    "evaluator_report_observed_count",
    "evaluator_report_missing_count",
    "completion_logging_complete_count",
    "completion_logging_incomplete_count",
    "task_usage_reconciled_count",
    "task_usage_unreconciled_count",
    "metrics_completion_extra_task_count",
    "metrics_completion_extra_count",
    "metrics_missing_completion_task_count",
    "metrics_missing_completion_count",
    "history_llm_metrics_ledger_match_count",
    "history_llm_metrics_ledger_mismatch_count",
    "message_prefix_reset_count",
    "repeated_request_snapshot_count",
    "response_not_materialized_in_next_request_count",
    "reasoning_subset_anomaly_count",
    "task_usage_all_preserved_sessions_count",
    "task_usage_explicit_zero_call_count",
    "task_usage_current_session_only_count",
    "task_usage_missing_extra_session_count",
    "task_usage_missing_no_evidence_count",
    "provider_error_envelope_count",
)


class OpenHandsTrajectoryAuditError(RuntimeError):
    """Raised when a trajectory freeze cannot be reproduced safely."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _semantic_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.name


def _code_source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in CODE_SOURCE_PATHS:
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            raise OpenHandsTrajectoryAuditError(
                f"required code source is missing: {relative}"
            )
        hashes[relative] = _sha256_file(path)
    return hashes


def _load_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise OpenHandsTrajectoryAuditError(
                    "inventory JSON contains a duplicate object key"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise OpenHandsTrajectoryAuditError(
            f"inventory JSON contains non-finite constant {value!r}"
        )

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=reject_duplicates,
                parse_constant=reject_constant,
            )
    except OpenHandsTrajectoryAuditError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OpenHandsTrajectoryAuditError("inventory is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise OpenHandsTrajectoryAuditError("inventory JSON root must be an object")
    return value


def _required_inventory_text(inventory: Mapping[str, Any], key: str) -> str:
    value = inventory.get(key)
    if not isinstance(value, str) or not value:
        raise OpenHandsTrajectoryAuditError(f"inventory {key!r} must be non-empty text")
    return value


def _required_inventory_int(inventory: Mapping[str, Any], key: str) -> int:
    value = inventory.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpenHandsTrajectoryAuditError(
            f"inventory {key!r} must be a non-negative integer"
        )
    return value


def _empty_metrics() -> dict[str, int]:
    return {key: 0 for key in _METRIC_KEYS}


def _empty_event_counts() -> dict[str, int]:
    return {event_type.value: 0 for event_type in EventType}


def _merge_counts(target: dict[str, int], source: Mapping[str, int]) -> None:
    for key, value in source.items():
        target[key] += value


def _usage_state(event: Any) -> str:
    payload = event.payload
    usage = TokenUsage.from_mapping(
        payload.get("usage") if isinstance(payload.get("usage"), dict) else None
    )
    if not usage.is_complete:
        return "missing"
    if usage.reported_total_matches is False:
        return "invalid"
    return "complete"


def _combined_usage_state(events: Sequence[Any]) -> str:
    if not events:
        return "missing"
    states = {_usage_state(event) for event in events}
    if "invalid" in states:
        return "invalid"
    if "missing" in states:
        return "missing"
    return "complete"


def _trajectory_metrics(
    trajectory: Trajectory,
) -> tuple[dict[str, int], dict[str, int]]:
    metrics = _empty_metrics()
    event_counts = _empty_event_counts()
    metrics["trajectory_count"] = 1
    metrics["event_count"] = len(trajectory.events)
    for event in trajectory.events:
        event_counts[event.event_type.value] += 1

    requests = [
        event
        for event in trajectory.events
        if event.event_type == EventType.REQUEST_BUILT
    ]
    metrics["logical_call_count"] = len(requests)
    metrics["request_count"] = len(requests)
    metrics["request_tokens_local_observed_count"] = sum(
        event.payload.get("request_tokens_local") is not None for event in requests
    )
    metrics["request_tokens_local_missing_count"] = (
        len(requests) - metrics["request_tokens_local_observed_count"]
    )

    started = {
        (str(event.logical_call_id), str(event.attempt_id))
        for event in trajectory.events
        if event.event_type == EventType.API_ATTEMPT_STARTED
    }
    attempt_terminals = {
        (str(event.logical_call_id), str(event.attempt_id)): event
        for event in trajectory.events
        if event.event_type in {EventType.API_COMPLETED, EventType.API_FAILED}
    }
    metrics["attempt_count"] = len(started)
    metrics["attempt_terminal_count"] = len(attempt_terminals)
    metrics["api_completed_count"] = event_counts[EventType.API_COMPLETED.value]
    metrics["api_failed_count"] = event_counts[EventType.API_FAILED.value]
    metrics["provider_error_envelope_count"] = sum(
        event.payload.get("provider_error_envelope_present") is True
        for event in attempt_terminals.values()
    )
    for attempt in sorted(started):
        terminal = attempt_terminals.get(attempt)
        state = _usage_state(terminal) if terminal is not None else "missing"
        metrics[f"attempt_usage_{state}_count"] += 1

    terminals_by_call: dict[str, list[Any]] = {}
    for (call_id, _), event in attempt_terminals.items():
        terminals_by_call.setdefault(call_id, []).append(event)
    for request in requests:
        state = _combined_usage_state(
            terminals_by_call.get(str(request.logical_call_id), [])
        )
        metrics[f"call_usage_{state}_count"] += 1

    terminal = trajectory.events[-1]
    task_usage_state = _usage_state(terminal)
    metrics[f"task_usage_{task_usage_state}_count"] = 1
    metrics["generation_checkpoint_count"] = event_counts[
        EventType.GENERATION_CHECKPOINT.value
    ]
    metrics["tool_started_count"] = event_counts[EventType.TOOL_STARTED.value]
    metrics["tool_completed_count"] = event_counts[EventType.TOOL_COMPLETED.value]
    metrics["tool_failed_count"] = event_counts[EventType.TOOL_FAILED.value]
    terminal_tool_events = [
        event
        for event in trajectory.events
        if event.event_type in {EventType.TOOL_COMPLETED, EventType.TOOL_FAILED}
    ]
    metrics["tool_terminal_failure_observable_count"] = sum(
        event.payload.get("failure_observable") is True
        for event in terminal_tool_events
    )
    metrics["tool_terminal_failure_unobservable_count"] = (
        len(terminal_tool_events)
        - metrics["tool_terminal_failure_observable_count"]
    )
    metrics["task_finished_count"] = event_counts[EventType.TASK_FINISHED.value]
    metrics["task_aborted_count"] = event_counts[EventType.TASK_ABORTED.value]
    metrics["task_error_count"] = int(
        terminal.payload.get("outcome") == "error"
        or terminal.payload.get("reason") == "task_error"
    )
    lifecycle_observed = terminal.payload.get("reason") in {
        "agent_finished",
        "task_error",
    }
    metrics["task_lifecycle_observed_count"] = int(lifecycle_observed)
    metrics["task_lifecycle_censored_count"] = int(not lifecycle_observed)
    task_log_observed = terminal.payload.get("lifecycle_source") == "output.jsonl"
    metrics["task_log_observed_count"] = int(task_log_observed)
    metrics["task_log_missing_count"] = int(not task_log_observed)
    report_present = terminal.payload.get("evaluator_report_present") is True
    metrics["evaluator_report_observed_count"] = int(report_present)
    metrics["evaluator_report_missing_count"] = int(not report_present)
    completion_logging_complete = (
        terminal.payload.get("completion_logging_complete") is True
    )
    metrics["completion_logging_complete_count"] = int(completion_logging_complete)
    metrics["completion_logging_incomplete_count"] = int(
        not completion_logging_complete
    )
    task_usage_reconciled = terminal.payload.get("task_usage_reconciled") is True
    metrics["task_usage_reconciled_count"] = int(task_usage_reconciled)
    metrics["task_usage_unreconciled_count"] = int(not task_usage_reconciled)
    completion_extra = int(terminal.payload.get("metrics_completion_extra_count") or 0)
    missing_completion = int(
        terminal.payload.get("metrics_missing_completion_count") or 0
    )
    if completion_extra < 0 or missing_completion < 0:
        raise OpenHandsTrajectoryAuditError(
            "reader emitted a negative metrics/completion reconciliation count"
        )
    metrics["metrics_completion_extra_task_count"] = int(completion_extra > 0)
    metrics["metrics_completion_extra_count"] = completion_extra
    metrics["metrics_missing_completion_task_count"] = int(missing_completion > 0)
    metrics["metrics_missing_completion_count"] = missing_completion
    history_metrics_match = (
        terminal.payload.get("history_llm_metrics_count_matches_ledger") is True
    )
    metrics["history_llm_metrics_ledger_match_count"] = int(history_metrics_match)
    metrics["history_llm_metrics_ledger_mismatch_count"] = int(
        not history_metrics_match
    )
    for payload_key, metric_key in (
        ("message_prefix_reset_count", "message_prefix_reset_count"),
        ("repeated_request_snapshot_count", "repeated_request_snapshot_count"),
        (
            "response_not_materialized_in_next_request_count",
            "response_not_materialized_in_next_request_count",
        ),
        ("reasoning_subset_anomaly_count", "reasoning_subset_anomaly_count"),
    ):
        value = int(terminal.payload.get(payload_key) or 0)
        if value < 0:
            raise OpenHandsTrajectoryAuditError(
                f"reader emitted a negative {payload_key}"
            )
        metrics[metric_key] = value
    usage_scope = str(terminal.payload.get("usage_scope") or "")
    completion_snapshot_count = terminal.payload.get("completion_snapshot_count")
    if (
        isinstance(completion_snapshot_count, bool)
        or not isinstance(completion_snapshot_count, int)
        or completion_snapshot_count < 0
    ):
        raise OpenHandsTrajectoryAuditError(
            "reader emitted an invalid completion_snapshot_count"
        )
    if completion_snapshot_count != len(requests):
        raise OpenHandsTrajectoryAuditError(
            "reader completion_snapshot_count disagrees with canonical requests"
        )
    explicit_zero_call = usage_scope == "explicit_zero_call_task"
    if explicit_zero_call:
        usage_payload = terminal.payload.get("usage")
        if not isinstance(usage_payload, dict):
            raise OpenHandsTrajectoryAuditError(
                "explicit zero-call task is missing its source-reported usage"
            )
        usage = TokenUsage.from_mapping(usage_payload)
        if (
            task_usage_state != "complete"
            or usage.accounted_total_tokens != 0
            or usage.reported_total_tokens != 0
            or completion_snapshot_count != 0
            or usage_payload.get("total_source")
            != "output_metrics_accumulated_token_usage"
        ):
            raise OpenHandsTrajectoryAuditError(
                "explicit zero-call task must have source-reported zero usage and "
                "zero completion snapshots"
            )
    metrics["task_usage_explicit_zero_call_count"] = int(explicit_zero_call)
    metrics["task_usage_all_preserved_sessions_count"] = int(
        usage_scope in {"all_preserved_sessions", "explicit_zero_call_task"}
    )
    metrics["task_usage_current_session_only_count"] = int(
        usage_scope == "current_output_session_without_completion_boundaries"
    )
    metrics["task_usage_missing_extra_session_count"] = int(
        usage_scope == "missing_due_to_incomplete_extra_session_usage"
    )
    metrics["task_usage_missing_no_evidence_count"] = int(
        usage_scope == "missing_no_completion_or_task_metrics"
    )
    return metrics, event_counts


def _row_semantic(row: DatasetRow) -> dict[str, Any]:
    """Mirror ``build_supervised_dataset`` row semantics exactly."""

    point = row.point
    return {
        "point": {
            "point_id": point.point_id,
            "source_event_id": point.source_event_id,
            "task_id": point.task_id,
            "trajectory_id": point.trajectory_id,
            "run_id": point.run_id,
            "prediction_context_id": point.prediction_context_id,
            "condition_id": point.condition_id,
            "logical_call_id": point.logical_call_id,
            "attempt_id": point.attempt_id,
            "cutoff_event_seq": point.cutoff_event_seq,
            "position": point.position.value,
            "target": point.target.value,
            "features": dict(point.features),
            "known_offset_tokens": point.known_offset_tokens,
        },
        "label": row.label,
        "status": row.status.value,
        "invalid_reason": row.invalid_reason,
    }


def _open_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode = OFF")
    connection.execute("PRAGMA synchronous = OFF")
    connection.execute("PRAGMA temp_store = FILE")
    connection.execute("PRAGMA cache_size = -32768")
    connection.executescript(
        """
        CREATE TABLE rows (
            point_id TEXT PRIMARY KEY COLLATE BINARY,
            run_id TEXT NOT NULL,
            position TEXT NOT NULL,
            target TEXT NOT NULL,
            status TEXT NOT NULL,
            reason TEXT NOT NULL,
            row_json TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE INDEX rows_run_point ON rows(run_id, point_id COLLATE BINARY);
        CREATE INDEX rows_matrix ON rows(run_id, position, target, status, reason);

        CREATE TABLE trajectories (
            trajectory_id TEXT PRIMARY KEY COLLATE BINARY,
            task_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            condition_id TEXT NOT NULL,
            canonical_sha256 TEXT NOT NULL,
            event_count INTEGER NOT NULL,
            call_count INTEGER NOT NULL,
            row_count INTEGER NOT NULL,
            aggregate_record_json TEXT NOT NULL,
            UNIQUE(task_id, run_id)
        ) WITHOUT ROWID;
        CREATE INDEX trajectories_task_run
            ON trajectories(task_id COLLATE BINARY, run_id COLLATE BINARY);
        CREATE INDEX trajectories_run
            ON trajectories(run_id, trajectory_id COLLATE BINARY);
        """
    )
    return connection


@contextmanager
def _workspace_temp_environment(path: Path) -> Iterable[None]:
    """Keep SQLite spill files under the ignored workspace on every platform."""

    parent = Path(path).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    previous = {name: os.environ.get(name) for name in ("TEMP", "TMP")}
    os.environ["TEMP"] = str(parent)
    os.environ["TMP"] = str(parent)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _insert_trajectory(
    connection: sqlite3.Connection,
    trajectory: Trajectory,
) -> tuple[int, dict[str, int], dict[str, int]]:
    event_semantic = [event.to_dict() for event in trajectory.events]
    canonical_sha256 = _semantic_sha256(event_semantic)
    dataset = build_supervised_dataset([trajectory])
    try:
        for row in dataset.rows:
            connection.execute(
                """
                INSERT INTO rows
                    (point_id, run_id, position, target, status, reason, row_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.point.point_id,
                    trajectory.run_id,
                    row.point.position.value,
                    row.point.target.value,
                    row.status.value,
                    row.invalid_reason,
                    _canonical_json(_row_semantic(row)),
                ),
            )
        metrics, event_counts = _trajectory_metrics(trajectory)
        record = {
            "task_id": trajectory.task_id,
            "run_id": trajectory.run_id,
            "trajectory_id": trajectory.trajectory_id,
            "condition_id": trajectory.condition_id,
            "canonical_sha256": canonical_sha256,
        }
        connection.execute(
            """
            INSERT INTO trajectories
                (trajectory_id, task_id, run_id, condition_id, canonical_sha256,
                 event_count, call_count, row_count, aggregate_record_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trajectory.trajectory_id,
                trajectory.task_id,
                trajectory.run_id,
                trajectory.condition_id,
                canonical_sha256,
                len(trajectory.events),
                metrics["logical_call_count"],
                len(dataset.rows),
                _canonical_json(record),
            ),
        )
    except sqlite3.IntegrityError as exc:
        raise OpenHandsTrajectoryAuditError(
            "reader emitted duplicate trajectory, task/run, or dataset point identity"
        ) from exc
    return len(dataset.rows), metrics, event_counts


def _stream_json_array_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    digest.update(b"[")
    first = True
    for value in values:
        if not first:
            digest.update(b",")
        digest.update(value.encode("utf-8"))
        first = False
    digest.update(b"]")
    return digest.hexdigest()


def _canonical_aggregate_sha256(
    connection: sqlite3.Connection,
    run_id: str | None = None,
) -> str:
    if run_id is None:
        cursor = connection.execute(
            """
            SELECT aggregate_record_json
            FROM trajectories
            ORDER BY trajectory_id COLLATE BINARY
            """
        )
    else:
        cursor = connection.execute(
            """
            SELECT aggregate_record_json
            FROM trajectories
            WHERE run_id = ?
            ORDER BY trajectory_id COLLATE BINARY
            """,
            (run_id,),
        )
    return _stream_json_array_sha256(row[0] for row in cursor)


def _external_dataset_digest(
    connection: sqlite3.Connection,
    run_id: str | None = None,
) -> tuple[str, int]:
    """Hash SQLite-external-sorted rows with builder-identical JSON bytes."""

    if run_id is None:
        cursor = connection.execute(
            "SELECT row_json FROM rows ORDER BY point_id COLLATE BINARY"
        )
    else:
        cursor = connection.execute(
            """
            SELECT row_json FROM rows
            WHERE run_id = ?
            ORDER BY point_id COLLATE BINARY
            """,
            (run_id,),
        )
    digest = hashlib.sha256()
    digest.update(
        (
            '{"feature_schema_version":'
            f"{FEATURE_SCHEMA_VERSION}"
            ',"rows":['
        ).encode("utf-8")
    )
    count = 0
    for (row_json,) in cursor:
        if count:
            digest.update(b",")
        digest.update(row_json.encode("utf-8"))
        count += 1
    digest.update(
        (
            '],"schema_version":'
            f"{DATASET_SCHEMA_VERSION}"
            "}"
        ).encode("utf-8")
    )
    return digest.hexdigest(), count


def _matrix(
    connection: sqlite3.Connection,
    run_id: str | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    for position in PredictionPosition:
        targets: dict[str, dict[str, Any]] = {}
        for target in PredictionTarget:
            targets[target.value] = {
                "structurally_emitted_by_builder": (
                    position.value,
                    target.value,
                )
                in _BUILDER_CELLS,
                "row_count": 0,
                "eligible_row_count": 0,
                "eligible_for_supervised_training": False,
                "status_counts": {status: 0 for status in _STATUS_VALUES},
                "reason_counts": {},
            }
        matrix[position.value] = targets

    where = "" if run_id is None else "WHERE run_id = ?"
    parameters: tuple[str, ...] = () if run_id is None else (run_id,)
    cursor = connection.execute(
        f"""
        SELECT position, target, status, reason, COUNT(*)
        FROM rows
        {where}
        GROUP BY position, target, status, reason
        ORDER BY position, target, status, reason
        """,  # noqa: S608 - fixed internal clause, no user SQL
        parameters,
    )
    for position, target, status, reason, count in cursor:
        if position not in matrix or target not in matrix[position]:
            raise OpenHandsTrajectoryAuditError(
                "builder emitted an unknown prediction position or target"
            )
        if status not in _STATUS_VALUES:
            raise OpenHandsTrajectoryAuditError("builder emitted an unknown label status")
        cell = matrix[position][target]
        cell["row_count"] += count
        cell["status_counts"][status] += count
        if status != LabelStatus.OBSERVED.value:
            reason_key = reason or "unspecified"
            cell["reason_counts"][reason_key] = (
                cell["reason_counts"].get(reason_key, 0) + count
            )
    for targets in matrix.values():
        for cell in targets.values():
            observed = cell["status_counts"][LabelStatus.OBSERVED.value]
            cell["eligible_row_count"] = observed
            cell["eligible_for_supervised_training"] = observed > 0
            cell["reason_counts"] = dict(sorted(cell["reason_counts"].items()))
    return matrix


def _distribution(
    connection: sqlite3.Connection,
    column: str,
    run_id: str | None,
) -> dict[str, Any]:
    if column not in {"event_count", "call_count", "row_count"}:
        raise ValueError("unsupported trajectory distribution column")
    where = "" if run_id is None else "WHERE run_id = ?"
    parameters: tuple[str, ...] = () if run_id is None else (run_id,)
    count, minimum, maximum, total = connection.execute(
        f"""
        SELECT COUNT(*), MIN({column}), MAX({column}), SUM({column})
        FROM trajectories {where}
        """,  # noqa: S608 - column/where are fixed internal allowlists
        parameters,
    ).fetchone()
    histogram = {
        str(value): frequency
        for value, frequency in connection.execute(
            f"""
            SELECT {column}, COUNT(*)
            FROM trajectories {where}
            GROUP BY {column}
            ORDER BY {column}
            """,  # noqa: S608 - column/where are fixed internal allowlists
            parameters,
        )
    }
    return {
        "count": count,
        "min": minimum,
        "max": maximum,
        "sum": total,
        "mean": (total / count) if count else None,
        "histogram": histogram,
    }


def _condition_counts(
    connection: sqlite3.Connection,
    run_id: str | None = None,
) -> dict[str, int]:
    if run_id is None:
        cursor = connection.execute(
            """
            SELECT condition_id, COUNT(*) FROM trajectories
            GROUP BY condition_id ORDER BY condition_id COLLATE BINARY
            """
        )
    else:
        cursor = connection.execute(
            """
            SELECT condition_id, COUNT(*) FROM trajectories
            WHERE run_id = ?
            GROUP BY condition_id ORDER BY condition_id COLLATE BINARY
            """,
            (run_id,),
        )
    return {condition_id: count for condition_id, count in cursor}


def _task_run_mapping(
    connection: sqlite3.Connection,
    expected_run_ids: Sequence[str] | None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    active_task: str | None = None
    runs: list[dict[str, str]] = []

    def finish() -> None:
        if active_task is None:
            return
        actual = tuple(item["run_id"] for item in runs)
        if expected_run_ids is not None and actual != tuple(expected_run_ids):
            raise OpenHandsTrajectoryAuditError(
                f"task {active_task!r} does not map exactly to the expected runs"
            )
        result.append({"task_id": active_task, "runs": list(runs)})

    cursor = connection.execute(
        """
        SELECT task_id, run_id, trajectory_id, condition_id
        FROM trajectories
        ORDER BY task_id COLLATE BINARY, run_id COLLATE BINARY
        """
    )
    for task_id, run_id, trajectory_id, condition_id in cursor:
        if active_task is not None and task_id != active_task:
            finish()
            runs = []
        active_task = task_id
        runs.append(
            {
                "run_id": run_id,
                "trajectory_id": trajectory_id,
                "condition_id": condition_id,
            }
        )
    finish()
    return result


def _canonical_trajectories(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    return [
        {
            "task_id": task_id,
            "run_id": run_id,
            "trajectory_id": trajectory_id,
            "condition_id": condition_id,
            "canonical_sha256": canonical_sha256,
        }
        for task_id, run_id, trajectory_id, condition_id, canonical_sha256 in (
            connection.execute(
                """
                SELECT task_id, run_id, trajectory_id, condition_id, canonical_sha256
                FROM trajectories
                ORDER BY trajectory_id COLLATE BINARY
                """
            )
        )
    ]


def _reconcile_inventory(
    inventory: Mapping[str, Any],
    *,
    task_count: int,
    trajectory_count: int,
    call_count: int,
    report_count: int,
    run_ids: Sequence[str],
) -> None:
    comparisons = {
        "task_count": task_count,
        "task_run_count": trajectory_count,
        "llm_completions_count": call_count,
        # Inventory v2 separates per-task evaluator reports from the four
        # run-level aggregate reports.  Only the former are attached to
        # canonical task trajectories.
        "task_report_count": report_count,
        "output_jsonl_record_count": trajectory_count,
    }
    for key, observed in comparisons.items():
        if _required_inventory_int(inventory, key) != observed:
            raise OpenHandsTrajectoryAuditError(
                f"reader {key} disagrees with the frozen inventory"
            )
    if _required_inventory_int(inventory, "aggregate_report_count") != len(run_ids):
        raise OpenHandsTrajectoryAuditError(
            "inventory must contain exactly one aggregate report per run"
        )
    raw_runs = inventory.get("runs")
    if isinstance(raw_runs, list):
        inventory_runs = tuple(
            f"run_{_required_inventory_int(item, 'run_id')}"
            for item in raw_runs
            if isinstance(item, Mapping)
        )
        if len(inventory_runs) != len(raw_runs) or tuple(run_ids) != inventory_runs:
            raise OpenHandsTrajectoryAuditError(
                "reader run identities disagree with the frozen inventory"
            )


def _source_capability(
    reader: OpenHandsArchiveReader,
    metrics: Mapping[str, int],
) -> dict[str, Any]:
    request_observed = metrics["request_tokens_local_observed_count"]
    checkpoints = metrics["generation_checkpoint_count"]
    return {
        "source_id": reader.source_id,
        "declared_observables": sorted(
            observable.value for observable in reader.capabilities.observables
        ),
        "request_tokens_local": {
            "available": request_observed > 0,
            "observed_count": request_observed,
            "missing_count": metrics["request_tokens_local_missing_count"],
            "reason": (
                "no_local_tokenizer_count_in_archive"
                if request_observed == 0
                else "partially_observed"
            ),
            "gates_targets": [
                PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS.value,
                PredictionTarget.CALL_UNKNOWN_BILLABLE_TOKENS.value,
            ],
        },
        "attempt_usage": {
            "available": metrics["attempt_usage_complete_count"] > 0,
            "complete_count": metrics["attempt_usage_complete_count"],
            "missing_count": metrics["attempt_usage_missing_count"],
            "invalid_count": metrics["attempt_usage_invalid_count"],
            "scope": "current_response_only",
        },
        "task_usage": {
            "available": metrics["task_usage_complete_count"] > 0,
            "complete_count": metrics["task_usage_complete_count"],
            "missing_count": metrics["task_usage_missing_count"],
            "invalid_count": metrics["task_usage_invalid_count"],
            "scope": (
                "output.metrics current-session aggregate plus complete preserved "
                "completion extras, plus source-reported explicit zero-call usage; "
                "never backfilled into attempt events"
            ),
            "explicit_zero_call_count": metrics[
                "task_usage_explicit_zero_call_count"
            ],
            "explicit_zero_call_source": (
                "output.metrics.accumulated_token_usage"
            ),
            "explicit_zero_call_never_imputed": True,
            "explicit_zero_call_criteria": (
                "usage_scope=explicit_zero_call_task, accounted and reported total "
                "tokens both zero, completion_snapshot_count=0"
            ),
            "all_preserved_sessions_count": metrics[
                "task_usage_all_preserved_sessions_count"
            ],
            "current_session_without_completion_boundaries_count": metrics[
                "task_usage_current_session_only_count"
            ],
            "missing_incomplete_extra_session_count": metrics[
                "task_usage_missing_extra_session_count"
            ],
            "missing_no_completion_or_task_metrics_count": metrics[
                "task_usage_missing_no_evidence_count"
            ],
        },
        "retry": {
            "supported": False,
            "retry_count": None,
            "reason": "provider_transport_retry_ledger_not_preserved",
        },
        "tool_events": {
            "available": (
                metrics["tool_started_count"]
                + metrics["tool_completed_count"]
                + metrics["tool_failed_count"]
            )
            > 0,
            "started_count": metrics["tool_started_count"],
            "completed_count": metrics["tool_completed_count"],
            "failed_count": metrics["tool_failed_count"],
            "failure_observable_count": metrics[
                "tool_terminal_failure_observable_count"
            ],
            "failure_unobservable_count": metrics[
                "tool_terminal_failure_unobservable_count"
            ],
            "failure_status_scope": "explicit_output_jsonl_only",
        },
        "errors": {
            "task_error_available": metrics["task_error_count"] > 0,
            "task_error_count": metrics["task_error_count"],
            "attempt_error_available": metrics["api_failed_count"] > 0,
            "attempt_error_count": metrics["api_failed_count"],
            "provider_error_envelope_available": (
                metrics["provider_error_envelope_count"] > 0
            ),
            "provider_error_envelope_count": metrics[
                "provider_error_envelope_count"
            ],
            "provider_error_envelope_semantics": (
                "preserved on a completed response; not classified as API_FAILED "
                "or a transport retry"
            ),
            "reason": "task_errors_attempt_failures_and_provider_envelopes_are_distinct",
        },
        "task_termination": {
            "available": metrics["task_lifecycle_observed_count"] > 0,
            "finished_count": metrics["task_finished_count"],
            "aborted_count": metrics["task_aborted_count"],
            "observed_lifecycle_count": metrics[
                "task_lifecycle_observed_count"
            ],
            "censored_lifecycle_count": metrics[
                "task_lifecycle_censored_count"
            ],
            "task_log_observed_count": metrics["task_log_observed_count"],
            "task_log_missing_count": metrics["task_log_missing_count"],
            "source": "output.jsonl_when_present_else_censored_logging_incomplete",
        },
        "generation_checkpoint": {
            "available": checkpoints > 0,
            "observed_count": checkpoints,
            "reason": (
                "no_streaming_generation_deltas_or_checkpoints_in_archive"
                if checkpoints == 0
                else "observed"
            ),
            "gates_position": PredictionPosition.CALL_UPDATE.value,
        },
        "session_reconciliation": {
            "completion_logging_complete_count": metrics[
                "completion_logging_complete_count"
            ],
            "completion_logging_incomplete_count": metrics[
                "completion_logging_incomplete_count"
            ],
            "task_usage_reconciled_count": metrics["task_usage_reconciled_count"],
            "task_usage_unreconciled_count": metrics[
                "task_usage_unreconciled_count"
            ],
            "metrics_completion_extra_task_count": metrics[
                "metrics_completion_extra_task_count"
            ],
            "metrics_completion_extra_count": metrics[
                "metrics_completion_extra_count"
            ],
            "metrics_missing_completion_task_count": metrics[
                "metrics_missing_completion_task_count"
            ],
            "metrics_missing_completion_count": metrics[
                "metrics_missing_completion_count"
            ],
            "history_llm_metrics_ledger_match_count": metrics[
                "history_llm_metrics_ledger_match_count"
            ],
            "history_llm_metrics_ledger_mismatch_count": metrics[
                "history_llm_metrics_ledger_mismatch_count"
            ],
            "message_prefix_reset_count": metrics["message_prefix_reset_count"],
            "repeated_request_snapshot_count": metrics[
                "repeated_request_snapshot_count"
            ],
            "response_not_materialized_in_next_request_count": metrics[
                "response_not_materialized_in_next_request_count"
            ],
            "reasoning_subset_anomaly_count": metrics[
                "reasoning_subset_anomaly_count"
            ],
        },
    }


def build_trajectory_audit(
    archive_path: Path,
    inventory_path: Path,
    *,
    expected_archive_bytes: int | None = EXPECTED_ARCHIVE_BYTES,
    expected_archive_sha256: str | None = EXPECTED_ARCHIVE_SHA256,
    expected_hub_repo: str = HUB_REPO,
    expected_revision: str = RESOLVED_REVISION,
    expected_run_ids: Sequence[str] | None = EXPECTED_RUN_IDS,
    sqlite_parent: Path | None = None,
) -> dict[str, Any]:
    """Freeze the full archive into a deterministic, content-free machine audit."""

    archive = Path(archive_path)
    inventory_source = Path(inventory_path)
    if not archive.is_file():
        raise OpenHandsTrajectoryAuditError("OpenHands archive is missing")
    if not inventory_source.is_file():
        raise OpenHandsTrajectoryAuditError("OpenHands inventory is missing")
    archive_bytes = archive.stat().st_size
    if expected_archive_bytes is not None and archive_bytes != expected_archive_bytes:
        raise OpenHandsTrajectoryAuditError("archive byte size disagrees with the pin")
    archive_sha256 = _sha256_file(archive)
    if (
        expected_archive_sha256 is not None
        and archive_sha256 != expected_archive_sha256
    ):
        raise OpenHandsTrajectoryAuditError("archive SHA256 disagrees with the pin")
    inventory_sha256 = _sha256_file(inventory_source)
    inventory = _load_json_object(inventory_source)
    if _required_inventory_int(inventory, "inventory_schema_version") != 2:
        raise OpenHandsTrajectoryAuditError(
            "trajectory audit requires the full-JSONL Spend inventory schema v2"
        )
    if _required_inventory_int(inventory, "archive_bytes") != archive_bytes:
        raise OpenHandsTrajectoryAuditError("inventory archive size does not match input")
    if _required_inventory_text(inventory, "archive_sha256") != archive_sha256:
        raise OpenHandsTrajectoryAuditError("inventory archive SHA256 does not match input")
    if _required_inventory_text(inventory, "hub_repo") != expected_hub_repo:
        raise OpenHandsTrajectoryAuditError("inventory Hub repository disagrees with pin")
    if _required_inventory_text(inventory, "resolved_revision") != expected_revision:
        raise OpenHandsTrajectoryAuditError("inventory revision disagrees with pin")

    parent = (
        Path(sqlite_parent)
        if sqlite_parent is not None
        else DEFAULT_SQLITE_PARENT
    )
    parent.mkdir(parents=True, exist_ok=True)
    with _workspace_temp_environment(parent), tempfile.TemporaryDirectory(
        prefix="openhands-trajectory-audit-", dir=parent
    ) as temporary:
        database_path = Path(temporary) / "audit.sqlite3"
        connection = _open_database(database_path)
        global_metrics = _empty_metrics()
        global_event_counts = _empty_event_counts()
        run_metrics: dict[str, dict[str, int]] = {}
        run_event_counts: dict[str, dict[str, int]] = {}
        try:
            reader = OpenHandsArchiveReader()
            metadata = OpenHandsArchiveMetadata(archive_identity=archive_sha256)
            for trajectory in reader.iter_archive(archive, metadata):
                _, metrics, event_counts = _insert_trajectory(connection, trajectory)
                _merge_counts(global_metrics, metrics)
                _merge_counts(global_event_counts, event_counts)
                metrics_for_run = run_metrics.setdefault(
                    trajectory.run_id, _empty_metrics()
                )
                events_for_run = run_event_counts.setdefault(
                    trajectory.run_id, _empty_event_counts()
                )
                _merge_counts(metrics_for_run, metrics)
                _merge_counts(events_for_run, event_counts)
            connection.commit()

            run_ids = tuple(
                row[0]
                for row in connection.execute(
                    """
                    SELECT DISTINCT run_id FROM trajectories
                    ORDER BY run_id COLLATE BINARY
                    """
                )
            )
            if expected_run_ids is not None and run_ids != tuple(expected_run_ids):
                raise OpenHandsTrajectoryAuditError(
                    "reader did not emit exactly the expected run identities"
                )
            task_count = connection.execute(
                "SELECT COUNT(DISTINCT task_id) FROM trajectories"
            ).fetchone()[0]
            trajectory_count = connection.execute(
                "SELECT COUNT(*) FROM trajectories"
            ).fetchone()[0]
            condition_count = connection.execute(
                "SELECT COUNT(DISTINCT condition_id) FROM trajectories"
            ).fetchone()[0]
            report_count = global_metrics["evaluator_report_observed_count"]
            _reconcile_inventory(
                inventory,
                task_count=task_count,
                trajectory_count=trajectory_count,
                call_count=global_metrics["logical_call_count"],
                report_count=report_count,
                run_ids=run_ids,
            )

            dataset_id, row_count = _external_dataset_digest(connection)
            task_mapping = _task_run_mapping(connection, expected_run_ids)
            per_run: dict[str, Any] = {}
            for run_id in run_ids:
                run_dataset_id, run_row_count = _external_dataset_digest(
                    connection, run_id
                )
                per_run[run_id] = {
                    "task_count": connection.execute(
                        """
                        SELECT COUNT(DISTINCT task_id) FROM trajectories
                        WHERE run_id = ?
                        """,
                        (run_id,),
                    ).fetchone()[0],
                    "trajectory_count": run_metrics[run_id]["trajectory_count"],
                    "condition_counts": _condition_counts(connection, run_id),
                    "canonical_aggregate_sha256": _canonical_aggregate_sha256(
                        connection, run_id
                    ),
                    "dataset": {
                        "dataset_id": run_dataset_id,
                        "row_count": run_row_count,
                    },
                    "metrics": {
                        **run_metrics[run_id],
                        "event_type_counts": run_event_counts[run_id],
                    },
                    "label_matrix": _matrix(connection, run_id),
                    "distributions": {
                        "events_per_trajectory": _distribution(
                            connection, "event_count", run_id
                        ),
                        "calls_per_trajectory": _distribution(
                            connection, "call_count", run_id
                        ),
                        "dataset_rows_per_trajectory": _distribution(
                            connection, "row_count", run_id
                        ),
                    },
                }

            audit: dict[str, Any] = {
                "trajectory_audit_schema_version": AUDIT_SCHEMA_VERSION,
                "archive": {
                    "local_relative_path": _display_path(archive),
                    "bytes": archive_bytes,
                    "sha256": archive_sha256,
                    "hub_repo": expected_hub_repo,
                    "resolved_revision": expected_revision,
                },
                "inventory": {
                    "local_relative_path": _display_path(inventory_source),
                    "sha256": inventory_sha256,
                    "inventory_schema_version": inventory.get(
                        "inventory_schema_version"
                    ),
                    "archive_identity_match": True,
                },
                "code_source_hashes": _code_source_hashes(),
                "counts": {
                    "task_id_count": task_count,
                    "run_id_count": len(run_ids),
                    "trajectory_id_count": trajectory_count,
                    "condition_id_count": condition_count,
                    "dataset_row_count": row_count,
                },
                "run_ids": list(run_ids),
                "condition_counts": _condition_counts(connection),
                "task_run_mapping": task_mapping,
                "canonical_trajectories": _canonical_trajectories(connection),
                "canonical_source_aggregate_sha256": _canonical_aggregate_sha256(
                    connection
                ),
                "dataset": {
                    "dataset_id": dataset_id,
                    "row_count": row_count,
                    "schema_version": DATASET_SCHEMA_VERSION,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "construction": (
                        "build_supervised_dataset([trajectory]) then SQLite BINARY "
                        "external sort by point_id and builder-identical canonical JSON"
                    ),
                },
                "metrics": {
                    **global_metrics,
                    "event_type_counts": global_event_counts,
                },
                "label_matrix": _matrix(connection),
                "source_capability": _source_capability(reader, global_metrics),
                "per_run": per_run,
                "distributions": {
                    "events_per_trajectory": _distribution(
                        connection, "event_count", None
                    ),
                    "calls_per_trajectory": _distribution(
                        connection, "call_count", None
                    ),
                    "dataset_rows_per_trajectory": _distribution(
                        connection, "row_count", None
                    ),
                },
                "hash_definitions": {
                    "trajectory_canonical_sha256": (
                        "SHA256(canonical JSON array of CanonicalEvent.to_dict(), "
                        "sort_keys=true, separators=(',', ':'))"
                    ),
                    "canonical_source_aggregate_sha256": (
                        "SHA256(canonical JSON array of trajectory identity/hash records "
                        "sorted by trajectory_id)"
                    ),
                    "dataset_id": (
                        "byte-identical build_supervised_dataset semantic JSON with rows "
                        "externally sorted by point_id"
                    ),
                    "audit_payload_sha256": (
                        "SHA256(canonical JSON of this audit before adding "
                        "audit_payload_sha256)"
                    ),
                },
            }
            audit["audit_payload_sha256"] = _semantic_sha256(audit)
            return audit
        finally:
            connection.close()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the pinned Spend GPT-5.2 OpenHands trajectories."
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    audit = build_trajectory_audit(args.archive, args.inventory)
    atomic_write_json(args.output, audit)
    print(
        json.dumps(
            {
                "output_path": _display_path(args.output),
                "archive_sha256": audit["archive"]["sha256"],
                "inventory_sha256": audit["inventory"]["sha256"],
                "task_id_count": audit["counts"]["task_id_count"],
                "trajectory_id_count": audit["counts"]["trajectory_id_count"],
                "dataset_id": audit["dataset"]["dataset_id"],
                "audit_payload_sha256": audit["audit_payload_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
