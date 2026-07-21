from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import tarfile
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO


DATASET_ID = "loong0814/openhands_trajectories"
RESOLVED_REVISION = "fa9cbb063f770df596da95af24f7af3b8f595778"
ARCHIVE_NAME = "gpt_5.2_4runs.tar.gz"
ARCHIVE_WRAPPER = "gpt_5.2_4runs"
EXPECTED_BYTES = 2_908_192_516
EXPECTED_SHA256 = "993abcb55aae423f9067d5e6c8e1aeaccf83b9ce31474a215982686527934214"
EXPECTED_RUN_IDS = frozenset({1, 2, 3, 4})
EXPECTED_TASKS = 500
EXPECTED_TASK_RUNS = 2_000
SOURCE_ID = "spend_your_money/openhands_trajectories:gpt_5.2_4runs"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = (
    PROJECT_ROOT / "workspace" / "external" / "spend_your_money" / ARCHIVE_NAME
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "workspace"
    / "external"
    / "spend_your_money"
    / "gpt_5.2_inventory.json"
)
DEFAULT_TEMP_PARENT = PROJECT_ROOT / "workspace" / "tmp" / "openhands_archive_audit"

CHUNK_BYTES = 8 * 1024 * 1024
DEFAULT_SCHEMA_SAMPLE_COUNT = 5
DEFAULT_MAX_SCHEMA_JSON_BYTES = 64 * 1024 * 1024
MAX_JSONL_LINE_BYTES = 64 * 1024 * 1024
MAX_MEMBER_BYTES = 4 * 1024 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 128 * 1024 * 1024 * 1024
MAX_MEMBER_COUNT = 10_000_000
MAX_DUPLICATE_GROUPS = 100
MAX_LARGEST_ITEMS = 25
MAX_SCHEMA_DEPTH = 7
MAX_SCHEMA_FIELDS = 64
MAX_SCHEMA_ARRAY_ITEMS = 16

RUN_RE = re.compile(r"(?:^|-)run_(?P<run_id>[1-9][0-9]*)$")
SAFE_EXTENSION_RE = re.compile(r"^\.[a-z0-9][a-z0-9._+-]{0,15}$")

AGGREGATE_REPORT_LIST_FIELDS = frozenset(
    {
        "completed_ids",
        "empty_patch_ids",
        "error_ids",
        "incomplete_ids",
        "resolved_ids",
        "submitted_ids",
        "unresolved_ids",
    }
)
AGGREGATE_REPORT_COUNT_FIELDS = frozenset(
    {
        "completed_instances",
        "empty_patch_instances",
        "error_instances",
        "resolved_instances",
        "submitted_instances",
        "total_instances",
        "unresolved_instances",
    }
)
AGGREGATE_REPORT_FIELDS = (
    AGGREGATE_REPORT_LIST_FIELDS
    | AGGREGATE_REPORT_COUNT_FIELDS
    | {"schema_version"}
)

# Only these known field names are ever emitted verbatim by the schema sampler.
# Unknown dictionary keys are represented by a one-way hash, because report roots,
# provider extensions, and tool payloads can contain task or request identifiers.
SAFE_SCHEMA_FIELDS = frozenset(
    {
        "arguments",
        "args",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "cached_tokens",
        "choices",
        "completion_tokens",
        "completion_tokens_details",
        "content",
        "cost",
        "cost_details",
        "created",
        "error",
        "finish_reason",
        "function",
        "id",
        "index",
        "is_byok",
        "logprobs",
        "message",
        "messages",
        "model",
        "name",
        "object",
        "patch_exists",
        "patch_is_None",
        "patch_successfully_applied",
        "prompt_tokens",
        "prompt_tokens_details",
        "provider",
        "reasoning_tokens",
        "resolved",
        "response",
        "role",
        "service_tier",
        "system_fingerprint",
        "tests_status",
        "timestamp",
        "tool_call_id",
        "tool_calls",
        "total_tokens",
        "type",
        "usage",
        "kwargs",
        "FAIL_TO_FAIL",
        "FAIL_TO_PASS",
        "PASS_TO_FAIL",
        "PASS_TO_PASS",
    }
)

CATEGORY_CODES = {
    "run_file": 1,
    "llm_completion": 2,
    "report": 3,
    "eval_artifact": 4,
    "infer_log": 5,
    "other": 6,
    "run_report": 7,
    "output_jsonl": 8,
    "output_swebench_jsonl": 9,
    "output_backup": 10,
}
CATEGORY_NAMES = {value: key for key, value in CATEGORY_CODES.items()}

JSONL_KIND_CODES = {"output_jsonl": 1, "output_swebench_jsonl": 2}
JSONL_KIND_NAMES = {value: key for key, value in JSONL_KIND_CODES.items()}
STATE_CODES = {"observed": 1, "missing": 2, "censored": 3, "invalid": 4}
STATE_NAMES = {value: key for key, value in STATE_CODES.items()}

OUTPUT_RECORD_FIELDS = frozenset(
    {
        "error",
        "history",
        "instance",
        "instance_id",
        "instruction",
        "metadata",
        "metrics",
        "test_result",
    }
)
SWEBENCH_RECORD_FIELDS = frozenset(
    {"instance_id", "model_name_or_path", "model_patch", "report"}
)
SWEBENCH_REPORT_FIELDS = frozenset(
    {"empty_generation", "error_eval", "failed_apply_patch", "resolved", "test_timeout"}
)
TASK_REPORT_FIELDS = frozenset(
    {
        "patch_exists",
        "patch_is_None",
        "patch_successfully_applied",
        "resolved",
        "tests_status",
    }
)
TASK_REPORT_TEST_BUCKETS = frozenset(
    {"FAIL_TO_FAIL", "FAIL_TO_PASS", "PASS_TO_FAIL", "PASS_TO_PASS"}
)
OUTCOME_CODES = {
    "resolved": 1,
    "unresolved": 2,
    "empty_patch": 3,
    "error": 4,
    "incomplete": 5,
}


class OpenHandsArchiveAuditError(RuntimeError):
    """Raised when the pinned archive cannot be audited without guessing."""


@contextmanager
def _workspace_temp_environment(path: Path):
    """Keep SQLite spill files inside the ignored workspace."""

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


class _HashingReader:
    """Hash compressed bytes while tarfile consumes the archive once."""

    def __init__(self, raw: BinaryIO) -> None:
        self._raw = raw
        self._digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        data = self._raw.read(size)
        if data:
            self._digest.update(data)
            self.bytes_read += len(data)
        return data

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _relative_archive_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        # Do not expose arbitrary absolute paths in a portable inventory.
        return path.name


def _safe_parts(member_name: str) -> tuple[str, ...]:
    if not member_name or "\x00" in member_name or "\\" in member_name:
        raise OpenHandsArchiveAuditError("archive contains an unsafe member path")
    if member_name.startswith(("/", "//")) or re.match(r"^[A-Za-z]:", member_name):
        raise OpenHandsArchiveAuditError("archive contains an absolute member path")
    path = PurePosixPath(member_name)
    parts = path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise OpenHandsArchiveAuditError("archive contains a non-canonical member path")
    if str(path) != member_name.rstrip("/"):
        raise OpenHandsArchiveAuditError("archive contains a non-canonical member path")
    return parts


def _parse_run_id(run_basename: str) -> int:
    match = RUN_RE.search(run_basename)
    if match is None:
        raise OpenHandsArchiveAuditError(
            "run directory basename does not end with the required run_N identity"
        )
    return int(match.group("run_id"))


def _safe_extension(name: str) -> str:
    suffix = PurePosixPath(name).suffix.lower()
    return suffix if SAFE_EXTENSION_RE.fullmatch(suffix) else "<none-or-redacted>"


def _task_identity(parts: tuple[str, ...], *, is_file: bool) -> tuple[str | None, str]:
    """Return a transient raw task id plus a safe artifact category."""
    relative = parts[2:]
    if not relative:
        return None, "other"

    root = relative[0]
    if root == "llm_completions":
        if len(relative) == 1:
            return None, "other"
        task_id = relative[1]
        if not task_id:
            raise OpenHandsArchiveAuditError("empty task identity under llm_completions")
        if is_file:
            if len(relative) != 3 or not relative[2].lower().endswith(".json"):
                raise OpenHandsArchiveAuditError(
                    "unknown llm_completions path shape; refusing to guess task identity"
                )
            return task_id, "llm_completion"
        if len(relative) > 2:
            raise OpenHandsArchiveAuditError(
                "nested directory under llm_completions has an unknown schema"
            )
        return task_id, "other"

    if root == "eval_outputs":
        if len(relative) == 1:
            return None, "other"
        task_id = relative[1]
        if not task_id:
            raise OpenHandsArchiveAuditError("empty task identity under eval_outputs")
        if is_file and len(relative) == 3 and relative[2] == "report.json":
            return task_id, "report"
        return task_id, "eval_artifact" if is_file else "other"

    if root == "infer_logs" and is_file:
        if len(relative) != 2:
            raise OpenHandsArchiveAuditError("unknown infer_logs path shape")
        basename = relative[1]
        if not basename.startswith("instance_") or not basename.endswith(".log"):
            raise OpenHandsArchiveAuditError("unknown infer_logs filename shape")
        task_id = basename[len("instance_") : -len(".log")]
        if not task_id:
            raise OpenHandsArchiveAuditError("empty task identity in infer log filename")
        return task_id, "infer_log"

    if len(relative) == 1 and is_file:
        basename = relative[0]
        if basename == "report.json":
            return None, "run_report"
        if basename == "output.jsonl":
            return None, "output_jsonl"
        if basename == "output.swebench.jsonl":
            return None, "output_swebench_jsonl"
        if basename.endswith(".bak"):
            return None, "output_backup"
        return None, "run_file"
    return None, "other"


def _path_template(category: str, *, is_file: bool) -> str:
    if not is_file:
        return "<wrapper>/<run>/<directory>"
    templates = {
        "llm_completion": "<wrapper>/<run>/llm_completions/<task_id>/<completion_file>",
        "report": "<wrapper>/<run>/eval_outputs/<task_id>/report.json",
        "eval_artifact": "<wrapper>/<run>/eval_outputs/<task_id>/<artifact>",
        "infer_log": "<wrapper>/<run>/infer_logs/<task_log>",
        "run_file": "<wrapper>/<run>/<run_file>",
        "run_report": "<wrapper>/<run>/report.json",
        "output_jsonl": "<wrapper>/<run>/output.jsonl",
        "output_swebench_jsonl": "<wrapper>/<run>/output.swebench.jsonl",
        "output_backup": "<wrapper>/<run>/<backup_file>.bak",
        "other": "<wrapper>/<run>/<other>",
    }
    return templates[category]


def _open_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = FILE;
        CREATE TABLE files (
            path_hash TEXT PRIMARY KEY,
            size_bytes INTEGER NOT NULL,
            run_id INTEGER,
            task_hash TEXT,
            category_code INTEGER NOT NULL
        );
        CREATE TABLE task_runs (
            task_hash TEXT NOT NULL,
            run_id INTEGER NOT NULL,
            size_bytes INTEGER NOT NULL DEFAULT 0,
            file_count INTEGER NOT NULL DEFAULT 0,
            completion_count INTEGER NOT NULL DEFAULT 0,
            completion_bytes INTEGER NOT NULL DEFAULT 0,
            report_count INTEGER NOT NULL DEFAULT 0,
            report_bytes INTEGER NOT NULL DEFAULT 0,
            report_resolved INTEGER,
            report_schema_hash TEXT,
            PRIMARY KEY (task_hash, run_id)
        );
        CREATE TABLE completions (
            content_hash TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            task_hash TEXT NOT NULL,
            run_id INTEGER NOT NULL
        );
        CREATE TABLE aggregate_submitted_tasks (
            task_hash TEXT NOT NULL,
            run_id INTEGER NOT NULL,
            PRIMARY KEY (task_hash, run_id)
        );
        CREATE TABLE aggregate_task_outcomes (
            task_hash TEXT NOT NULL,
            run_id INTEGER NOT NULL,
            outcome_code INTEGER NOT NULL,
            PRIMARY KEY (task_hash, run_id)
        );
        CREATE TABLE jsonl_files (
            path_hash TEXT PRIMARY KEY,
            file_hash TEXT NOT NULL,
            run_id INTEGER NOT NULL,
            kind_code INTEGER NOT NULL,
            size_bytes INTEGER NOT NULL,
            record_count INTEGER NOT NULL,
            UNIQUE (run_id, kind_code)
        );
        CREATE TABLE jsonl_records (
            task_hash TEXT NOT NULL,
            run_id INTEGER NOT NULL,
            kind_code INTEGER NOT NULL,
            record_hash TEXT NOT NULL,
            schema_hash TEXT NOT NULL,
            history_schema_hash TEXT,
            metrics_schema_hash TEXT,
            usage_schema_hash TEXT,
            history_state INTEGER NOT NULL,
            metrics_state INTEGER NOT NULL,
            usage_state INTEGER NOT NULL,
            error_nonempty INTEGER NOT NULL,
            history_length INTEGER NOT NULL,
            derived_empty_generation INTEGER,
            derived_error_eval INTEGER,
            derived_failed_apply_patch INTEGER,
            derived_resolved INTEGER,
            derived_test_timeout INTEGER,
            PRIMARY KEY (task_hash, run_id, kind_code)
        );
        CREATE INDEX completions_hash_idx ON completions(content_hash);
        CREATE INDEX task_runs_task_idx ON task_runs(task_hash);
        CREATE INDEX jsonl_records_kind_idx ON jsonl_records(kind_code, run_id);
        """
    )
    return connection


def _ensure_task_run(connection: sqlite3.Connection, task_hash: str, run_id: int) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO task_runs(task_hash, run_id) VALUES (?, ?)",
        (task_hash, run_id),
    )


def _record_file(
    connection: sqlite3.Connection,
    *,
    path_hash: str,
    size_bytes: int,
    run_id: int | None,
    task_hash: str | None,
    category: str,
) -> None:
    try:
        connection.execute(
            "INSERT INTO files VALUES (?, ?, ?, ?, ?)",
            (path_hash, size_bytes, run_id, task_hash, CATEGORY_CODES[category]),
        )
    except sqlite3.IntegrityError as exc:
        raise OpenHandsArchiveAuditError(
            "archive contains a duplicate member path (reported only by hash)"
        ) from exc
    if task_hash is None or run_id is None:
        return
    _ensure_task_run(connection, task_hash, run_id)
    completion = int(category == "llm_completion")
    report = int(category == "report")
    connection.execute(
        """
        UPDATE task_runs
        SET size_bytes = size_bytes + ?,
            file_count = file_count + 1,
            completion_count = completion_count + ?,
            completion_bytes = completion_bytes + ?,
            report_count = report_count + ?,
            report_bytes = report_bytes + ?
        WHERE task_hash = ? AND run_id = ?
        """,
        (
            size_bytes,
            completion,
            size_bytes if completion else 0,
            report,
            size_bytes if report else 0,
            task_hash,
            run_id,
        ),
    )


def _read_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    retain: bool,
    max_retained_bytes: int,
) -> tuple[str, bytes | None]:
    if retain and member.size > max_retained_bytes:
        raise OpenHandsArchiveAuditError(
            "selected schema JSON exceeds the explicit in-memory audit limit"
        )
    handle = archive.extractfile(member)
    if handle is None:
        raise OpenHandsArchiveAuditError("regular archive member could not be opened")
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if retain else None
    size = 0
    with handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
            if chunks is not None:
                chunks.append(chunk)
    if size != member.size:
        raise OpenHandsArchiveAuditError("archive member size does not match tar metadata")
    return digest.hexdigest(), b"".join(chunks) if chunks is not None else None


def _strict_json_value(payload: bytes, *, context: str) -> Any:
    try:
        text = payload.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_key,
            parse_constant=lambda value: (_ for _ in ()).throw(
                OpenHandsArchiveAuditError(f"non-finite JSON number: {value}")
            ),
        )
    except OpenHandsArchiveAuditError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenHandsArchiveAuditError(f"{context} is not strict UTF-8 JSON") from exc


def _schema_hash(value: Any) -> str:
    schema = _json_schema(value)
    encoded = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _missing_state(*, error_nonempty: bool) -> int:
    return STATE_CODES["censored" if error_nonempty else "missing"]


def _validate_output_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != OUTPUT_RECORD_FIELDS:
        raise OpenHandsArchiveAuditError("output.jsonl record has an unknown top-level schema")
    task_id = value["instance_id"]
    instance = value["instance"]
    error = value["error"]
    error_nonempty = error not in (None, "", [], {})
    if not isinstance(task_id, str) or not task_id:
        raise OpenHandsArchiveAuditError(
            "output.jsonl task identity is missing"
        )
    if isinstance(instance, dict):
        if instance.get("instance_id") != task_id:
            raise OpenHandsArchiveAuditError(
                "output.jsonl nested task identity is inconsistent"
            )
    elif instance is not None or not error_nonempty:
        raise OpenHandsArchiveAuditError(
            "output.jsonl instance may be null only for a non-empty task error"
        )
    instruction = value["instruction"]
    if not isinstance(instruction, str) and not (
        instruction is None and error_nonempty
    ):
        raise OpenHandsArchiveAuditError(
            "output.jsonl instruction may be null only for a non-empty task error"
        )
    metadata = value["metadata"]
    if not isinstance(metadata, dict) and not (metadata is None and error_nonempty):
        raise OpenHandsArchiveAuditError(
            "output.jsonl metadata may be null only for a non-empty task error"
        )
    if value["test_result"] is not None and not isinstance(value["test_result"], dict):
        raise OpenHandsArchiveAuditError(
            "output.jsonl test_result must be an object or null"
        )

    history = value["history"]
    history_schema_hash: str | None = None
    history_length = 0
    if isinstance(history, list):
        if any(not isinstance(item, dict) for item in history):
            raise OpenHandsArchiveAuditError(
                "output.jsonl history must contain only event objects"
            )
        history_length = len(history)
        history_state = (
            STATE_CODES["observed"]
            if history
            else _missing_state(error_nonempty=error_nonempty)
        )
        history_schema_hash = _schema_hash(history)
    elif history is None:
        history_state = _missing_state(error_nonempty=error_nonempty)
    else:
        history_state = STATE_CODES["invalid"]

    metrics = value["metrics"]
    metrics_schema_hash: str | None = None
    usage_schema_hash: str | None = None
    if isinstance(metrics, dict) and metrics:
        metrics_state = STATE_CODES["observed"]
        metrics_schema_hash = _schema_hash(metrics)
        if "accumulated_token_usage" not in metrics:
            usage_state = _missing_state(error_nonempty=error_nonempty)
        else:
            usage = metrics["accumulated_token_usage"]
            if isinstance(usage, dict) and usage:
                usage_state = STATE_CODES["observed"]
                usage_schema_hash = _schema_hash(usage)
            elif usage is None or usage == {}:
                usage_state = _missing_state(error_nonempty=error_nonempty)
            else:
                usage_state = STATE_CODES["invalid"]
    elif metrics is None or metrics == {}:
        metrics_state = _missing_state(error_nonempty=error_nonempty)
        usage_state = _missing_state(error_nonempty=error_nonempty)
    else:
        metrics_state = STATE_CODES["invalid"]
        usage_state = STATE_CODES["invalid"]

    return {
        "task_id": task_id,
        "schema_hash": _schema_hash(value),
        "history_schema_hash": history_schema_hash,
        "metrics_schema_hash": metrics_schema_hash,
        "usage_schema_hash": usage_schema_hash,
        "history_state": history_state,
        "metrics_state": metrics_state,
        "usage_state": usage_state,
        "error_nonempty": int(error_nonempty),
        "history_length": history_length,
        "derived_empty_generation": None,
        "derived_error_eval": None,
        "derived_failed_apply_patch": None,
        "derived_resolved": None,
        "derived_test_timeout": None,
    }


def _validate_swebench_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != SWEBENCH_RECORD_FIELDS:
        raise OpenHandsArchiveAuditError(
            "output.swebench.jsonl record has an unknown top-level schema"
        )
    task_id = value["instance_id"]
    if not isinstance(task_id, str) or not task_id:
        raise OpenHandsArchiveAuditError(
            "output.swebench.jsonl task identity must be a non-empty string"
        )
    if not isinstance(value["model_name_or_path"], str) or not isinstance(
        value["model_patch"], str
    ):
        raise OpenHandsArchiveAuditError(
            "output.swebench.jsonl model fields must be strings"
        )
    report = value["report"]
    if not isinstance(report, dict) or set(report) != SWEBENCH_REPORT_FIELDS:
        raise OpenHandsArchiveAuditError(
            "output.swebench.jsonl report has an unknown schema"
        )
    if any(not isinstance(report[field], bool) for field in SWEBENCH_REPORT_FIELDS):
        raise OpenHandsArchiveAuditError(
            "output.swebench.jsonl report flags must be booleans"
        )
    return {
        "task_id": task_id,
        "schema_hash": _schema_hash(value),
        "history_schema_hash": None,
        "metrics_schema_hash": None,
        "usage_schema_hash": None,
        "history_state": STATE_CODES["missing"],
        "metrics_state": STATE_CODES["missing"],
        "usage_state": STATE_CODES["missing"],
        "error_nonempty": 0,
        "history_length": 0,
        "derived_empty_generation": int(report["empty_generation"]),
        "derived_error_eval": int(report["error_eval"]),
        "derived_failed_apply_patch": int(report["failed_apply_patch"]),
        "derived_resolved": int(report["resolved"]),
        "derived_test_timeout": int(report["test_timeout"]),
    }


def _audit_jsonl_member(
    connection: sqlite3.Connection,
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    path_hash: str,
    run_id: int,
    category: str,
    max_line_bytes: int,
) -> None:
    if max_line_bytes <= 0:
        raise ValueError("max_line_bytes must be positive")
    handle = archive.extractfile(member)
    if handle is None:
        raise OpenHandsArchiveAuditError("JSONL archive member could not be opened")
    kind_code = JSONL_KIND_CODES[category]
    file_digest = hashlib.sha256()
    bytes_read = 0
    record_count = 0
    with handle:
        while line := handle.readline(max_line_bytes + 1):
            bytes_read += len(line)
            file_digest.update(line)
            if len(line) > max_line_bytes:
                raise OpenHandsArchiveAuditError(
                    "JSONL record exceeds the explicit 64 MiB line limit"
                )
            if not line.strip():
                raise OpenHandsArchiveAuditError("JSONL contains an empty record line")
            record_count += 1
            value = _strict_json_value(
                line,
                context=f"{category} record {record_count}",
            )
            parsed = (
                _validate_output_record(value)
                if category == "output_jsonl"
                else _validate_swebench_record(value)
            )
            task_hash = _hash_text(parsed.pop("task_id"))
            record_hash = hashlib.sha256(line).hexdigest()
            try:
                connection.execute(
                    """
                    INSERT INTO jsonl_records VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        task_hash,
                        run_id,
                        kind_code,
                        record_hash,
                        parsed["schema_hash"],
                        parsed["history_schema_hash"],
                        parsed["metrics_schema_hash"],
                        parsed["usage_schema_hash"],
                        parsed["history_state"],
                        parsed["metrics_state"],
                        parsed["usage_state"],
                        parsed["error_nonempty"],
                        parsed["history_length"],
                        parsed["derived_empty_generation"],
                        parsed["derived_error_eval"],
                        parsed["derived_failed_apply_patch"],
                        parsed["derived_resolved"],
                        parsed["derived_test_timeout"],
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise OpenHandsArchiveAuditError(
                    "JSONL repeats a task identity within one run"
                ) from exc
    if bytes_read != member.size:
        raise OpenHandsArchiveAuditError("JSONL byte count does not match tar metadata")
    try:
        connection.execute(
            "INSERT INTO jsonl_files VALUES (?, ?, ?, ?, ?, ?)",
            (
                path_hash,
                file_digest.hexdigest(),
                run_id,
                kind_code,
                member.size,
                record_count,
            ),
        )
    except sqlite3.IntegrityError as exc:
        raise OpenHandsArchiveAuditError(
            "run contains more than one formal JSONL file of the same kind"
        ) from exc


def _reject_duplicate_json_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OpenHandsArchiveAuditError("sampled JSON contains a duplicate object key")
        result[key] = value
    return result


def _parse_json(payload: bytes, *, kind: str, task_id: str) -> Any:
    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_key,
            parse_constant=lambda value: (_ for _ in ()).throw(
                OpenHandsArchiveAuditError(f"non-finite JSON number: {value}")
            ),
        )
    except OpenHandsArchiveAuditError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenHandsArchiveAuditError(f"sampled {kind} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise OpenHandsArchiveAuditError(f"sampled {kind} JSON root is not an object")
    if kind == "completion":
        if not isinstance(value.get("messages"), list) or not isinstance(
            value.get("response"), dict
        ):
            raise OpenHandsArchiveAuditError(
                "sampled completion schema requires messages:list and response:object"
            )
    if kind == "report":
        if len(value) != 1 or task_id not in value or not isinstance(value[task_id], dict):
            raise OpenHandsArchiveAuditError(
                "sampled task report does not map its path task identity to one object"
            )
        report = value[task_id]
        if set(report) != TASK_REPORT_FIELDS:
            raise OpenHandsArchiveAuditError("task report has an unknown evaluator schema")
        if any(
            not isinstance(report[field], bool)
            for field in (
                "patch_exists",
                "patch_is_None",
                "patch_successfully_applied",
                "resolved",
            )
        ):
            raise OpenHandsArchiveAuditError("task report evaluator flags must be booleans")
        tests_status = report["tests_status"]
        if (
            not isinstance(tests_status, dict)
            or set(tests_status) != TASK_REPORT_TEST_BUCKETS
            or any(not isinstance(item, dict) for item in tests_status.values())
        ):
            raise OpenHandsArchiveAuditError("task report tests_status has an unknown schema")
    if kind == "aggregate_report":
        if set(value) != AGGREGATE_REPORT_FIELDS:
            raise OpenHandsArchiveAuditError(
                "run-level report has an unknown aggregate schema"
            )
        for field in AGGREGATE_REPORT_LIST_FIELDS:
            items = value[field]
            if (
                not isinstance(items, list)
                or any(not isinstance(item, str) for item in items)
                or len(items) != len(set(items))
            ):
                raise OpenHandsArchiveAuditError(
                    "run-level report id fields must be duplicate-free string lists"
                )
        for field in AGGREGATE_REPORT_COUNT_FIELDS | {"schema_version"}:
            item = value[field]
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise OpenHandsArchiveAuditError(
                    "run-level report count fields must be non-negative integers"
                )
        for stem in (
            "completed",
            "empty_patch",
            "error",
            "resolved",
            "submitted",
            "unresolved",
        ):
            if value[f"{stem}_instances"] != len(value[f"{stem}_ids"]):
                raise OpenHandsArchiveAuditError(
                    "run-level report count does not match its id-list length"
                )
        if value["total_instances"] != len(value["submitted_ids"]):
            raise OpenHandsArchiveAuditError(
                "run-level report total does not match submitted ids"
            )
        resolved = set(value["resolved_ids"])
        unresolved = set(value["unresolved_ids"])
        completed = set(value["completed_ids"])
        empty_patch = set(value["empty_patch_ids"])
        errors = set(value["error_ids"])
        incomplete = set(value["incomplete_ids"])
        submitted = set(value["submitted_ids"])
        if resolved & unresolved or completed != resolved | unresolved:
            raise OpenHandsArchiveAuditError(
                "run-level report resolved/unresolved partition is inconsistent"
            )
        terminal_groups = (completed, empty_patch, errors, incomplete)
        if any(
            left & right
            for index, left in enumerate(terminal_groups)
            for right in terminal_groups[index + 1 :]
        ) or submitted != set().union(*terminal_groups):
            raise OpenHandsArchiveAuditError(
                "run-level report submitted-task partition is inconsistent"
            )
    return value


def _schema_field_name(name: str) -> dict[str, str]:
    if name in SAFE_SCHEMA_FIELDS:
        return {"name": name}
    return {"name": "<redacted>", "name_sha256": _hash_text(name)}


def _json_schema(value: Any, *, depth: int = 0) -> dict[str, Any]:
    if depth >= MAX_SCHEMA_DEPTH:
        return {"type": type(value).__name__, "truncated": True}
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        selected: list[Any]
        if len(value) <= MAX_SCHEMA_ARRAY_ITEMS:
            selected = value
        else:
            half = MAX_SCHEMA_ARRAY_ITEMS // 2
            selected = [*value[:half], *value[-half:]]
        item_schemas: dict[str, dict[str, Any]] = {}
        for item in selected:
            schema = _json_schema(item, depth=depth + 1)
            fingerprint = hashlib.sha256(
                json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            item_schemas.setdefault(fingerprint, schema)
        return {
            "type": "array",
            "empty": not value,
            "item_schemas": [item_schemas[key] for key in sorted(item_schemas)],
            "items_sampled": len(selected),
            "items_truncated": len(value) > len(selected),
        }
    if isinstance(value, dict):
        fields: list[dict[str, Any]] = []
        keys = sorted(value)
        for key in keys[:MAX_SCHEMA_FIELDS]:
            fields.append(
                {
                    **_schema_field_name(key),
                    "schema": _json_schema(value[key], depth=depth + 1),
                }
            )
        return {
            "type": "object",
            "field_count": len(keys),
            "fields": fields,
            "fields_truncated": len(keys) > len(fields),
        }
    raise OpenHandsArchiveAuditError("sampled JSON contains an unsupported Python value")


def _record_schema(
    samples: dict[str, dict[str, Any]],
    *,
    task_hash: str,
    kind: str,
    value: Any,
) -> None:
    schema = _json_schema(value)
    encoded = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    fingerprint = hashlib.sha256(encoded).hexdigest()
    bucket = samples[task_hash][f"{kind}_schemas"]
    if fingerprint not in bucket:
        bucket[fingerprint] = {"count": 0, "schema": schema}
    bucket[fingerprint]["count"] += 1


def _aggregate_report_stats(value: dict[str, Any]) -> dict[str, int]:
    """Retain only aggregate integers; raw task ids never enter the inventory."""
    return {
        "schema_version": value["schema_version"],
        "total_instances": value["total_instances"],
        "submitted_instances": value["submitted_instances"],
        "completed_instances": value["completed_instances"],
        "resolved_instances": value["resolved_instances"],
        "unresolved_instances": value["unresolved_instances"],
        "empty_patch_instances": value["empty_patch_instances"],
        "error_instances": value["error_instances"],
        "incomplete_instances": len(value["incomplete_ids"]),
    }


def _consider_sample(
    samples: dict[str, dict[str, Any]], task_hash: str, sample_count: int
) -> bool:
    if task_hash in samples:
        return True
    if len(samples) < sample_count:
        samples[task_hash] = {"completion_schemas": {}, "report_schemas": {}}
        return True
    largest = max(samples)
    if task_hash >= largest:
        return False
    del samples[largest]
    samples[task_hash] = {"completion_schemas": {}, "report_schemas": {}}
    return True


def _sql_distribution(
    connection: sqlite3.Connection,
    source_sql: str,
    parameters: tuple[Any, ...] = (),
) -> dict[str, int | float | str | None]:
    count, total, minimum, maximum, mean = connection.execute(
        f"""
        SELECT COUNT(*), COALESCE(SUM(size_bytes), 0), MIN(size_bytes),
               MAX(size_bytes), AVG(size_bytes)
        FROM ({source_sql})
        """,
        parameters,
    ).fetchone()

    def quantile(proportion: float) -> int | None:
        if count == 0:
            return None
        offset = max(0, math.ceil(proportion * count) - 1)
        return connection.execute(
            f"SELECT size_bytes FROM ({source_sql}) ORDER BY size_bytes LIMIT 1 OFFSET ?",
            (*parameters, offset),
        ).fetchone()[0]

    return {
        "method": "nearest_rank",
        "count": count,
        "sum_bytes": total,
        "sum": total,
        "min_bytes": minimum,
        "p25_bytes": quantile(0.25),
        "p50_bytes": quantile(0.50),
        "p75_bytes": quantile(0.75),
        "p90_bytes": quantile(0.90),
        "p95_bytes": quantile(0.95),
        "p99_bytes": quantile(0.99),
        "max_bytes": maximum,
        "mean_bytes": round(mean, 3) if mean is not None else None,
    }


def _duplicate_summary(connection: sqlite3.Connection) -> dict[str, Any]:
    summary = connection.execute(
        """
        WITH grouped AS (
            SELECT content_hash, size_bytes, COUNT(*) AS copies,
                   COUNT(DISTINCT run_id) AS run_count,
                   COUNT(DISTINCT task_hash) AS task_count
            FROM completions
            GROUP BY content_hash, size_bytes
            HAVING COUNT(*) > 1
        )
        SELECT COUNT(*), COALESCE(SUM(copies), 0),
               COALESCE(SUM(copies - 1), 0),
               COALESCE(SUM(CASE WHEN run_count = 1 AND task_count = 1 THEN 1 ELSE 0 END), 0),
               COALESCE(SUM(CASE WHEN run_count > 1 THEN 1 ELSE 0 END), 0),
               COALESCE(SUM(CASE WHEN task_count > 1 THEN 1 ELSE 0 END), 0)
        FROM grouped
        """
    ).fetchone()
    rows = connection.execute(
        """
        WITH grouped AS (
            SELECT content_hash, size_bytes, COUNT(*) AS copies,
                   COUNT(DISTINCT run_id) AS run_count,
                   COUNT(DISTINCT task_hash) AS task_count
            FROM completions
            GROUP BY content_hash, size_bytes
            HAVING COUNT(*) > 1
        )
        SELECT content_hash, size_bytes, copies, run_count, task_count
        FROM grouped
        ORDER BY copies DESC, size_bytes DESC, content_hash
        LIMIT ?
        """,
        (MAX_DUPLICATE_GROUPS,),
    ).fetchall()
    groups = [
        {
            "sha256": content_hash,
            "size_bytes": size_bytes,
            "count": copies,
            "run_count": run_count,
            "task_count": task_count,
            "cross_run": run_count > 1,
            "cross_task": task_count > 1,
        }
        for content_hash, size_bytes, copies, run_count, task_count in rows
    ]
    group_count, file_count, extra_copies, within_one, cross_run, cross_task = summary
    return {
        "definition": "byte-identical complete llm_completions member payloads",
        "duplicate_hash_count": group_count,
        "duplicate_file_count": file_count,
        "duplicate_extra_copies": extra_copies,
        "within_one_task_run_group_count": within_one,
        "cross_run_group_count": cross_run,
        "cross_task_group_count": cross_task,
        "groups": groups,
        "groups_returned": len(groups),
        "groups_truncated": group_count > len(groups),
    }


def _schema_samples(
    connection: sqlite3.Connection, samples: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for task_hash in sorted(samples):
        row = connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(completion_count), 0),
                   COALESCE(SUM(report_count), 0)
            FROM task_runs WHERE task_hash = ?
            """,
            (task_hash,),
        ).fetchone()
        run_ids = [
            item[0]
            for item in connection.execute(
                "SELECT run_id FROM task_runs WHERE task_hash = ? ORDER BY run_id",
                (task_hash,),
            )
        ]
        sample = samples[task_hash]

        def schemas(kind: str) -> list[dict[str, Any]]:
            values = sample[f"{kind}_schemas"]
            return [
                {
                    "schema_sha256": fingerprint,
                    "observed_file_count": values[fingerprint]["count"],
                    "schema": values[fingerprint]["schema"],
                }
                for fingerprint in sorted(values)
            ]

        output.append(
            {
                "task_id_sha256": task_hash,
                "run_ids": run_ids,
                "task_run_count": row[0],
                "completion_file_count": row[1],
                "report_file_count": row[2],
                "completion_schemas": schemas("completion"),
                "report_schemas": schemas("report"),
            }
        )
    return output


def _four_state_counts(
    connection: sqlite3.Connection,
    *,
    column: str,
    kind_code: int,
    run_id: int | None = None,
) -> dict[str, int]:
    if column not in {"history_state", "metrics_state", "usage_state"}:
        raise ValueError("unsupported state column")
    where = "kind_code = ?"
    parameters: list[int] = [kind_code]
    if run_id is not None:
        where += " AND run_id = ?"
        parameters.append(run_id)
    observed = dict(
        connection.execute(
            f"""
            SELECT {column}, COUNT(*) FROM jsonl_records
            WHERE {where} GROUP BY {column}
            """,
            tuple(parameters),
        )
    )
    return {name: observed.get(code, 0) for name, code in STATE_CODES.items()}


def _schema_count_summary(
    connection: sqlite3.Connection,
    *,
    column: str,
    kind_code: int,
    run_id: int | None = None,
) -> dict[str, Any]:
    if column not in {
        "schema_hash",
        "history_schema_hash",
        "metrics_schema_hash",
        "usage_schema_hash",
    }:
        raise ValueError("unsupported schema column")
    where = f"kind_code = ? AND {column} IS NOT NULL"
    parameters: list[int] = [kind_code]
    if run_id is not None:
        where += " AND run_id = ?"
        parameters.append(run_id)
    unique_count = connection.execute(
        f"SELECT COUNT(DISTINCT {column}) FROM jsonl_records WHERE {where}",
        tuple(parameters),
    ).fetchone()[0]
    rows = connection.execute(
        f"""
        SELECT {column}, COUNT(*) FROM jsonl_records WHERE {where}
        GROUP BY {column} ORDER BY COUNT(*) DESC, {column} LIMIT 100
        """,
        tuple(parameters),
    ).fetchall()
    return {
        "unique_schema_count": unique_count,
        "schemas": [
            {"schema_sha256": schema_hash, "record_count": count}
            for schema_hash, count in rows
        ],
        "schemas_returned": len(rows),
        "schemas_truncated": unique_count > len(rows),
    }


def _set_source(
    *, kind_code: int | None = None, table: str | None = None, run_id: int | None = None
) -> tuple[str, tuple[int, ...]]:
    if kind_code is not None:
        sql = "SELECT task_hash, run_id FROM jsonl_records WHERE kind_code = ?"
        parameters: tuple[int, ...] = (kind_code,)
    elif table == "aggregate":
        sql = "SELECT task_hash, run_id FROM aggregate_submitted_tasks"
        parameters = ()
    elif table == "completion":
        sql = "SELECT task_hash, run_id FROM task_runs WHERE completion_count > 0"
        parameters = ()
    elif table == "task_artifact":
        sql = "SELECT task_hash, run_id FROM task_runs"
        parameters = ()
    else:
        raise ValueError("unknown task-set source")
    if run_id is not None:
        sql += " AND run_id = ?" if " WHERE " in sql else " WHERE run_id = ?"
        parameters = (*parameters, run_id)
    return sql, parameters


def _compare_task_sets(
    connection: sqlite3.Connection,
    left: tuple[str, tuple[int, ...]],
    right: tuple[str, tuple[int, ...]],
) -> dict[str, int | bool]:
    left_sql, left_parameters = left
    right_sql, right_parameters = right
    left_count = connection.execute(
        f"SELECT COUNT(*) FROM ({left_sql})", left_parameters
    ).fetchone()[0]
    right_count = connection.execute(
        f"SELECT COUNT(*) FROM ({right_sql})", right_parameters
    ).fetchone()[0]
    left_only = connection.execute(
        f"""
        SELECT COUNT(*) FROM ({left_sql}) AS left_set
        LEFT JOIN ({right_sql}) AS right_set
          ON left_set.task_hash = right_set.task_hash
         AND left_set.run_id = right_set.run_id
        WHERE right_set.task_hash IS NULL
        """,
        (*left_parameters, *right_parameters),
    ).fetchone()[0]
    right_only = connection.execute(
        f"""
        SELECT COUNT(*) FROM ({right_sql}) AS right_set
        LEFT JOIN ({left_sql}) AS left_set
          ON left_set.task_hash = right_set.task_hash
         AND left_set.run_id = right_set.run_id
        WHERE left_set.task_hash IS NULL
        """,
        (*right_parameters, *left_parameters),
    ).fetchone()[0]
    return {
        "left_count": left_count,
        "right_count": right_count,
        "left_only_count": left_only,
        "right_only_count": right_only,
        "matches": left_only == 0 and right_only == 0,
    }


def _jsonl_summary(
    connection: sqlite3.Connection,
    *,
    category: str,
    run_id: int | None = None,
) -> dict[str, Any]:
    kind_code = JSONL_KIND_CODES[category]
    where = "kind_code = ?"
    parameters: tuple[int, ...] = (kind_code,)
    if run_id is not None:
        where += " AND run_id = ?"
        parameters = (kind_code, run_id)
    file_count, size_bytes, record_count = connection.execute(
        f"""
        SELECT COUNT(*), COALESCE(SUM(size_bytes), 0),
               COALESCE(SUM(record_count), 0)
        FROM jsonl_files WHERE {where}
        """,
        parameters,
    ).fetchone()
    files = [
        {
            "run_id": file_run_id,
            "member_path_sha256": path_hash,
            "sha256": file_hash,
            "bytes": file_bytes,
            "record_count": records,
        }
        for path_hash, file_hash, file_run_id, file_bytes, records in connection.execute(
            f"""
            SELECT path_hash, file_hash, run_id, size_bytes, record_count
            FROM jsonl_files WHERE {where} ORDER BY run_id
            """,
            parameters,
        )
    ]
    record_where = "kind_code = ?"
    record_parameters: tuple[int, ...] = (kind_code,)
    if run_id is not None:
        record_where += " AND run_id = ?"
        record_parameters = (kind_code, run_id)
    unique_task_count, error_nonempty_count = connection.execute(
        f"""
        SELECT COUNT(DISTINCT task_hash), COALESCE(SUM(error_nonempty), 0)
        FROM jsonl_records WHERE {record_where}
        """,
        record_parameters,
    ).fetchone()
    base_source = _set_source(kind_code=kind_code, run_id=run_id)
    other_category = (
        "output_swebench_jsonl" if category == "output_jsonl" else "output_jsonl"
    )
    task_sets = {
        "aggregate_submitted": _compare_task_sets(
            connection,
            base_source,
            _set_source(table="aggregate", run_id=run_id),
        ),
        "other_formal_jsonl": _compare_task_sets(
            connection,
            base_source,
            _set_source(
                kind_code=JSONL_KIND_CODES[other_category], run_id=run_id
            ),
        ),
        "nonempty_llm_completions": _compare_task_sets(
            connection,
            base_source,
            _set_source(table="completion", run_id=run_id),
        ),
        "all_task_artifacts": _compare_task_sets(
            connection,
            base_source,
            _set_source(table="task_artifact", run_id=run_id),
        ),
    }
    result: dict[str, Any] = {
        "filename": (
            "output.jsonl"
            if category == "output_jsonl"
            else "output.swebench.jsonl"
        ),
        "file_count": file_count,
        "bytes": size_bytes,
        "record_count": record_count,
        "unique_task_count": unique_task_count,
        "files": files,
        "task_sets": task_sets,
        "record_schemas": _schema_count_summary(
            connection, column="schema_hash", kind_code=kind_code, run_id=run_id
        ),
    }
    if category == "output_jsonl":
        result.update(
            {
                "history_status_counts": _four_state_counts(
                    connection,
                    column="history_state",
                    kind_code=kind_code,
                    run_id=run_id,
                ),
                "metrics_status_counts": _four_state_counts(
                    connection,
                    column="metrics_state",
                    kind_code=kind_code,
                    run_id=run_id,
                ),
                "usage_status_counts": _four_state_counts(
                    connection,
                    column="usage_state",
                    kind_code=kind_code,
                    run_id=run_id,
                ),
                "task_error_nonempty_count": error_nonempty_count,
                "history_schemas": _schema_count_summary(
                    connection,
                    column="history_schema_hash",
                    kind_code=kind_code,
                    run_id=run_id,
                ),
                "metrics_schemas": _schema_count_summary(
                    connection,
                    column="metrics_schema_hash",
                    kind_code=kind_code,
                    run_id=run_id,
                ),
                "usage_schemas": _schema_count_summary(
                    connection,
                    column="usage_schema_hash",
                    kind_code=kind_code,
                    run_id=run_id,
                ),
                "history_length_distribution": _sql_distribution(
                    connection,
                    f"""
                    SELECT history_length AS size_bytes FROM jsonl_records
                    WHERE {record_where}
                    """,
                    record_parameters,
                ),
            }
        )
    else:
        flags = {}
        for column in (
            "derived_empty_generation",
            "derived_error_eval",
            "derived_failed_apply_patch",
            "derived_resolved",
            "derived_test_timeout",
        ):
            false_count, true_count = connection.execute(
                f"""
                SELECT COALESCE(SUM(CASE WHEN {column} = 0 THEN 1 ELSE 0 END), 0),
                       COALESCE(SUM(CASE WHEN {column} = 1 THEN 1 ELSE 0 END), 0)
                FROM jsonl_records WHERE {record_where}
                """,
                record_parameters,
            ).fetchone()
            flags[column.removeprefix("derived_")] = {
                "false": false_count,
                "true": true_count,
            }
        result["derived_report_flag_counts"] = flags
        result["resolved_semantics"] = (
            "derived transport field; false is not treated as observed evaluator accuracy"
        )
    return result


def _evaluator_status_counts(connection: sqlite3.Connection) -> dict[str, int]:
    observed, missing, censored, invalid = connection.execute(
        """
        SELECT
          COALESCE(SUM(CASE
            WHEN outcome.outcome_code IN (1, 2)
             AND task.report_count = 1
             AND task.report_resolved = CASE WHEN outcome.outcome_code = 1 THEN 1 ELSE 0 END
            THEN 1 ELSE 0 END), 0),
          COALESCE(SUM(CASE
            WHEN outcome.outcome_code = 3
              OR (outcome.outcome_code IN (1, 2) AND task.report_count = 0)
            THEN 1 ELSE 0 END), 0),
          COALESCE(SUM(CASE WHEN outcome.outcome_code = 5 THEN 1 ELSE 0 END), 0),
          COALESCE(SUM(CASE
            WHEN outcome.outcome_code = 4
              OR (outcome.outcome_code IN (1, 2) AND (
                   task.report_count <> 1
                   OR task.report_resolved IS NULL
                   OR task.report_resolved <> CASE WHEN outcome.outcome_code = 1 THEN 1 ELSE 0 END
                 ) AND task.report_count <> 0)
            THEN 1 ELSE 0 END), 0)
        FROM aggregate_task_outcomes AS outcome
        LEFT JOIN task_runs AS task
          ON task.task_hash = outcome.task_hash AND task.run_id = outcome.run_id
        """
    ).fetchone()
    return {
        "observed": observed,
        "missing": missing,
        "censored": censored,
        "invalid": invalid,
    }


def _finalize_inventory(
    *,
    connection: sqlite3.Connection,
    archive_path: Path,
    archive_bytes: int,
    archive_sha256: str,
    member_count: int,
    directory_count: int,
    regular_file_count: int,
    regular_file_bytes: int,
    runs_by_id: dict[int, str],
    run_member_counts: Counter[int],
    run_file_counts: Counter[int],
    run_bytes: Counter[int],
    extension_counts: Counter[str],
    extension_bytes: Counter[str],
    template_counts: Counter[str],
    template_bytes: Counter[str],
    depth_counts: Counter[int],
    category_counts: Counter[str],
    category_bytes: Counter[str],
    samples: dict[str, dict[str, Any]],
    aggregate_report_stats: dict[int, dict[str, int]],
    expected_wrapper: str,
    schema_sample_count: int,
    expected_task_count: int,
    max_jsonl_line_bytes: int,
) -> dict[str, Any]:
    task_count = connection.execute(
        "SELECT COUNT(DISTINCT task_hash) FROM task_runs"
    ).fetchone()[0]
    task_run_count = connection.execute("SELECT COUNT(*) FROM task_runs").fetchone()[0]
    completion_count, completion_bytes = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM completions"
    ).fetchone()
    task_report_count, task_report_bytes = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM files WHERE category_code = ?",
        (CATEGORY_CODES["report"],),
    ).fetchone()
    aggregate_report_count, aggregate_report_bytes = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM files WHERE category_code = ?",
        (CATEGORY_CODES["run_report"],),
    ).fetchone()
    report_count = task_report_count + aggregate_report_count
    report_bytes = task_report_bytes + aggregate_report_bytes

    coverage_rows = connection.execute(
        "SELECT task_hash, COUNT(*) FROM task_runs GROUP BY task_hash ORDER BY task_hash"
    ).fetchall()
    run_count_distribution = Counter(count for _, count in coverage_rows)
    exact_four = sum(count == 4 for _, count in coverage_rows)
    missing_completion_task_runs = connection.execute(
        "SELECT COUNT(*) FROM task_runs WHERE completion_count = 0"
    ).fetchone()[0]
    missing_report_task_runs = connection.execute(
        "SELECT COUNT(*) FROM task_runs WHERE report_count = 0"
    ).fetchone()[0]
    multiple_report_task_runs = connection.execute(
        "SELECT COUNT(*) FROM task_runs WHERE report_count > 1"
    ).fetchone()[0]

    largest_files = [
        {
            "member_path_sha256": path_hash,
            "size_bytes": size_bytes,
            "run_id": run_id,
            "task_id_sha256": task_hash,
            "category": CATEGORY_NAMES[category_code],
        }
        for path_hash, size_bytes, run_id, task_hash, category_code in connection.execute(
            """
            SELECT path_hash, size_bytes, run_id, task_hash, category_code
            FROM files ORDER BY size_bytes DESC, path_hash LIMIT ?
            """,
            (MAX_LARGEST_ITEMS,),
        )
    ]
    largest_tasks = [
        {
            "task_id_sha256": task_hash,
            "size_bytes": size_bytes,
            "task_run_count": run_count,
            "file_count": file_count,
            "llm_completions_count": completion_files,
            "report_count": reports,
        }
        for task_hash, size_bytes, run_count, file_count, completion_files, reports in connection.execute(
            """
            SELECT task_hash, SUM(size_bytes), COUNT(*), SUM(file_count),
                   SUM(completion_count), SUM(report_count)
            FROM task_runs GROUP BY task_hash
            ORDER BY SUM(size_bytes) DESC, task_hash LIMIT ?
            """,
            (MAX_LARGEST_ITEMS,),
        )
    ]

    runs: list[dict[str, Any]] = []
    for run_id in sorted(runs_by_id):
        run_task_count, run_completion_count, run_completion_bytes = (
            connection.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(completion_count), 0),
                       COALESCE(SUM(completion_bytes), 0)
                FROM task_runs WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        )
        run_task_report_count, run_task_report_bytes = connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM files
            WHERE run_id = ? AND category_code = ?
            """,
            (run_id, CATEGORY_CODES["report"]),
        ).fetchone()
        run_aggregate_report_count, run_aggregate_report_bytes = connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM files
            WHERE run_id = ? AND category_code = ?
            """,
            (run_id, CATEGORY_CODES["run_report"]),
        ).fetchone()
        missing_run_completions, missing_run_reports, multiple_run_reports = connection.execute(
            """
            SELECT SUM(CASE WHEN completion_count = 0 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN report_count = 0 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN report_count > 1 THEN 1 ELSE 0 END)
            FROM task_runs WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        missing_run_completions = missing_run_completions or 0
        missing_run_reports = missing_run_reports or 0
        multiple_run_reports = multiple_run_reports or 0
        submitted_task_count = connection.execute(
            "SELECT COUNT(*) FROM aggregate_submitted_tasks WHERE run_id = ?",
            (run_id,),
        ).fetchone()[0]
        submitted_without_task_artifact = connection.execute(
            """
            SELECT COUNT(*) FROM aggregate_submitted_tasks AS submitted
            LEFT JOIN task_runs AS task
              ON task.task_hash = submitted.task_hash AND task.run_id = submitted.run_id
            WHERE submitted.run_id = ? AND task.task_hash IS NULL
            """,
            (run_id,),
        ).fetchone()[0]
        task_artifact_without_submission = connection.execute(
            """
            SELECT COUNT(*) FROM task_runs AS task
            LEFT JOIN aggregate_submitted_tasks AS submitted
              ON task.task_hash = submitted.task_hash AND task.run_id = submitted.run_id
            WHERE task.run_id = ? AND submitted.task_hash IS NULL
            """,
            (run_id,),
        ).fetchone()[0]
        aggregate = aggregate_report_stats.get(run_id)
        run_output = _jsonl_summary(
            connection, category="output_jsonl", run_id=run_id
        )
        run_swebench_output = _jsonl_summary(
            connection, category="output_swebench_jsonl", run_id=run_id
        )
        run_backup_count, run_backup_bytes = connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM files
            WHERE run_id = ? AND category_code = ?
            """,
            (run_id, CATEGORY_CODES["output_backup"]),
        ).fetchone()
        run_task_report_resolved_count = connection.execute(
            """
            SELECT COALESCE(SUM(report_resolved), 0) FROM task_runs
            WHERE run_id = ? AND report_count = 1
            """,
            (run_id,),
        ).fetchone()[0]
        runs.append(
            {
                "run_id": run_id,
                "run_basename": runs_by_id[run_id],
                "member_count": run_member_counts[run_id],
                "regular_file_count": run_file_counts[run_id],
                "regular_file_bytes": run_bytes[run_id],
                "task_count": run_task_count,
                "task_run_count": run_task_count,
                "llm_completions_count": run_completion_count,
                "llm_completions_bytes": run_completion_bytes,
                "missing_completion_task_run_count": missing_run_completions,
                "task_report_count": run_task_report_count,
                "task_report_bytes": run_task_report_bytes,
                "missing_task_report_task_run_count": missing_run_reports,
                "task_report_resolved_count": run_task_report_resolved_count,
                "aggregate_report_count": run_aggregate_report_count,
                "aggregate_report_bytes": run_aggregate_report_bytes,
                "report_count": run_task_report_count + run_aggregate_report_count,
                "report_bytes": run_task_report_bytes + run_aggregate_report_bytes,
                "report_present": run_task_report_count > 0,
                "task_report_present": run_task_report_count > 0,
                "aggregate_report_present": run_aggregate_report_count > 0,
                "reports_complete": (
                    run_task_count > 0
                    and missing_run_reports == 0
                    and multiple_run_reports == 0
                ),
                # Inventorying presence is not an accuracy audit. In particular, a
                # missing report must never be silently converted to resolved=0.
                "resolved_count": (
                    aggregate["resolved_instances"] if aggregate is not None else None
                ),
                "resolved_count_status": (
                    "observed in validated aggregate report"
                    if aggregate is not None
                    else "missing; never impute unresolved/zero"
                ),
                "aggregate_report": aggregate,
                "aggregate_submitted_task_count": submitted_task_count,
                "aggregate_task_set_matches_task_artifacts": (
                    aggregate is not None
                    and submitted_without_task_artifact == 0
                    and task_artifact_without_submission == 0
                ),
                "aggregate_submitted_without_task_artifact_count": (
                    submitted_without_task_artifact
                ),
                "task_artifact_without_aggregate_submission_count": (
                    task_artifact_without_submission
                ),
                "output_jsonl": run_output,
                "output_swebench_jsonl": run_swebench_output,
                "output_backup": {
                    "file_count": run_backup_count,
                    "bytes": run_backup_bytes,
                    "consumed": False,
                    "inventory_only": True,
                },
            }
        )

    duplicate_snapshots = _duplicate_summary(connection)
    observed_run_ids = frozenset(runs_by_id)
    expected_task_runs = expected_task_count * len(EXPECTED_RUN_IDS)
    output_jsonl = _jsonl_summary(connection, category="output_jsonl")
    output_swebench_jsonl = _jsonl_summary(
        connection, category="output_swebench_jsonl"
    )
    backup_count, backup_bytes = connection.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM files
        WHERE category_code = ?
        """,
        (CATEGORY_CODES["output_backup"],),
    ).fetchone()
    evaluator_status_counts = _evaluator_status_counts(connection)
    task_report_resolved_count = connection.execute(
        """
        SELECT COALESCE(SUM(report_resolved), 0) FROM task_runs
        WHERE report_count = 1
        """
    ).fetchone()[0]
    outcome_counts_raw = dict(
        connection.execute(
            """
            SELECT outcome_code, COUNT(*) FROM aggregate_task_outcomes
            GROUP BY outcome_code
            """
        )
    )
    outcome_counts = {
        name: outcome_counts_raw.get(code, 0) for name, code in OUTCOME_CODES.items()
    }
    task_report_schema_rows = connection.execute(
        """
        SELECT report_schema_hash, COUNT(*) FROM task_runs
        WHERE report_schema_hash IS NOT NULL
        GROUP BY report_schema_hash ORDER BY COUNT(*) DESC, report_schema_hash
        LIMIT 100
        """
    ).fetchall()
    task_report_unique_schema_count = connection.execute(
        """
        SELECT COUNT(DISTINCT report_schema_hash) FROM task_runs
        WHERE report_schema_hash IS NOT NULL
        """
    ).fetchone()[0]
    coverage_complete = exact_four == task_count and task_count > 0
    expected_runs_present = observed_run_ids == EXPECTED_RUN_IDS
    anomalies: list[dict[str, Any]] = []

    def anomaly(code: str, count: int, detail: Any | None = None) -> None:
        if not count:
            return
        item: dict[str, Any] = {"code": code, "count": count}
        if detail is not None:
            item["detail"] = detail
        anomalies.append(item)

    anomaly(
        "unexpected_run_id_set",
        int(not expected_runs_present),
        {"expected": sorted(EXPECTED_RUN_IDS), "observed": sorted(observed_run_ids)},
    )
    anomaly("unexpected_task_count", abs(task_count - expected_task_count))
    anomaly("unexpected_task_run_count", abs(task_run_count - expected_task_runs))
    anomaly("tasks_without_exactly_four_runs", task_count - exact_four)
    anomaly("task_runs_without_llm_completions", missing_completion_task_runs)
    anomaly(
        "task_runs_without_report",
        missing_report_task_runs,
        {"label_policy": "missing; never impute unresolved/zero"},
    )
    anomaly("task_runs_with_multiple_reports", multiple_report_task_runs)
    anomaly(
        "byte_identical_completion_snapshot_groups",
        duplicate_snapshots["duplicate_hash_count"],
    )

    schema_output = _schema_samples(connection, samples)
    schema_sample_complete = bool(schema_output) and all(
        item["completion_schemas"] and item["report_schemas"] for item in schema_output
    )
    aggregate_reports_complete = set(aggregate_report_stats) == set(runs_by_id)
    aggregate_task_sets_match = aggregate_reports_complete and all(
        item["aggregate_task_set_matches_task_artifacts"] for item in runs
    )
    formal_jsonl_files_complete = (
        output_jsonl["file_count"] == len(EXPECTED_RUN_IDS)
        and output_swebench_jsonl["file_count"] == len(EXPECTED_RUN_IDS)
    )
    formal_jsonl_records_complete = (
        output_jsonl["record_count"] == expected_task_runs
        and output_swebench_jsonl["record_count"] == expected_task_runs
    )
    formal_jsonl_task_sets_match = (
        output_jsonl["task_sets"]["aggregate_submitted"]["matches"]
        and output_jsonl["task_sets"]["other_formal_jsonl"]["matches"]
        and output_jsonl["task_sets"]["all_task_artifacts"]["matches"]
        and output_swebench_jsonl["task_sets"]["aggregate_submitted"]["matches"]
    )
    formal_jsonl_ready = (
        formal_jsonl_files_complete
        and formal_jsonl_records_complete
        and formal_jsonl_task_sets_match
    )
    trajectory_ready = (
        expected_runs_present
        and task_count == expected_task_count
        and task_run_count == expected_task_runs
        and coverage_complete
        and formal_jsonl_ready
        and missing_completion_task_runs == 0
    )
    report_ready = missing_report_task_runs == 0 and multiple_report_task_runs == 0
    evaluator_full_coverage = all(
        evaluator_status_counts[state] == 0
        for state in ("missing", "censored", "invalid")
    )

    anomaly(
        "runs_without_validated_aggregate_report",
        len(runs_by_id) - len(aggregate_report_stats),
    )
    anomaly(
        "runs_with_aggregate_task_set_mismatch",
        sum(
            item["aggregate_report"] is not None
            and not item["aggregate_task_set_matches_task_artifacts"]
            for item in runs
        ),
    )
    anomaly(
        "unexpected_output_jsonl_record_count",
        abs(output_jsonl["record_count"] - expected_task_runs),
    )
    anomaly(
        "unexpected_output_swebench_jsonl_record_count",
        abs(output_swebench_jsonl["record_count"] - expected_task_runs),
    )
    anomaly(
        "formal_jsonl_task_set_mismatch",
        int(not formal_jsonl_task_sets_match),
    )
    anomaly(
        "output_records_without_nonempty_completion_set_match",
        output_jsonl["task_sets"]["nonempty_llm_completions"]["left_only_count"],
    )
    anomaly(
        "evaluator_accuracy_missing",
        evaluator_status_counts["missing"],
        {"reason": "empty_patch or absent completed-task report; never impute false"},
    )
    anomaly(
        "evaluator_accuracy_censored",
        evaluator_status_counts["censored"],
    )
    anomaly(
        "evaluator_accuracy_invalid",
        evaluator_status_counts["invalid"],
        {"reason": "evaluator error or contradictory task report"},
    )

    return {
        "inventory_schema_version": 2,
        "source_id": SOURCE_ID,
        "archive_path": _relative_archive_path(archive_path),
        "hub_repo": DATASET_ID,
        "resolved_revision": RESOLVED_REVISION,
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha256,
        "source_hashes": {"archive_sha256": archive_sha256},
        "member_count": member_count,
        "directory_count": directory_count,
        "regular_file_count": regular_file_count,
        "regular_file_bytes": regular_file_bytes,
        "task_count": task_count,
        "trajectory_count": task_run_count,
        "task_run_count": task_run_count,
        "run_count": len(runs),
        "runs": runs,
        "completions_count": completion_count,
        "llm_completions_count": completion_count,
        "llm_completions_bytes": completion_bytes,
        "missing_completion_task_run_count": missing_completion_task_runs,
        "report_count": report_count,
        "report_bytes": report_bytes,
        "task_report_count": task_report_count,
        "task_report_bytes": task_report_bytes,
        "missing_task_report_task_run_count": missing_report_task_runs,
        "task_report_resolved_count": task_report_resolved_count,
        "task_report_schemas": {
            "unique_schema_count": task_report_unique_schema_count,
            "schemas": [
                {"schema_sha256": schema_hash, "report_count": count}
                for schema_hash, count in task_report_schema_rows
            ],
            "schemas_returned": len(task_report_schema_rows),
            "schemas_truncated": (
                task_report_unique_schema_count > len(task_report_schema_rows)
            ),
        },
        "aggregate_report_count": aggregate_report_count,
        "aggregate_report_bytes": aggregate_report_bytes,
        "aggregate_outcome_counts": outcome_counts,
        "output_jsonl_record_count": output_jsonl["record_count"],
        "output_swebench_jsonl_record_count": output_swebench_jsonl["record_count"],
        "jsonl_audit": {
            "line_limit_bytes": max_jsonl_line_bytes,
            "output_jsonl": output_jsonl,
            "output_swebench_jsonl": output_swebench_jsonl,
            "backup_files": {
                "file_count": backup_count,
                "bytes": backup_bytes,
                "consumed": False,
                "inventory_only": True,
            },
        },
        "telemetry_status_counts": {
            "history": output_jsonl["history_status_counts"],
            "metrics": output_jsonl["metrics_status_counts"],
            "usage": output_jsonl["usage_status_counts"],
            "task_error_nonempty": output_jsonl["task_error_nonempty_count"],
        },
        "label_status_counts": {
            "evaluator_accuracy": evaluator_status_counts,
            "task_termination": {
                "observed": 0,
                "missing": task_run_count,
                "censored": 0,
                "invalid": 0,
            },
        },
        "task_termination_evidence": {
            "source": "output.jsonl history",
            "uniform_explicit_terminal_field": None,
            "complete_termination_claim_supported": False,
            "reason": (
                "task reports are evaluator outcomes, not agent termination; "
                "history has no validated uniform explicit terminal field"
            ),
        },
        "task_run_coverage": {
            str(key): value for key, value in sorted(run_count_distribution.items())
        },
        "task_run_coverage_details": {
            "task_count": task_count,
            "task_run_count": task_run_count,
            "run_count_per_task_distribution": {
                str(key): value for key, value in sorted(run_count_distribution.items())
            },
            "exactly_four_runs_task_count": exact_four,
            "not_exactly_four_runs_task_count": task_count - exact_four,
        },
        "exactly_four_runs": coverage_complete,
        "size_distributions": {
            "files": _sql_distribution(connection, "SELECT size_bytes FROM files"),
            "task_runs": _sql_distribution(
                connection, "SELECT size_bytes FROM task_runs"
            ),
            "tasks": _sql_distribution(
                connection,
                """
                SELECT SUM(size_bytes) AS size_bytes
                FROM task_runs GROUP BY task_hash
                """,
            ),
        },
        "largest_files": largest_files,
        "largest_tasks": largest_tasks,
        "path_structure": {
            "wrapper": expected_wrapper,
            "run_parser": "generic basename suffix run_N",
            "templates": {
                key: {"member_count": template_counts[key], "bytes": template_bytes[key]}
                for key in sorted(template_counts)
            },
            "depth_member_counts": {
                str(key): value for key, value in sorted(depth_counts.items())
            },
            "categories": {
                key: {"file_count": category_counts[key], "bytes": category_bytes[key]}
                for key in sorted(category_counts)
            },
            "extensions": {
                key: {"file_count": extension_counts[key], "bytes": extension_bytes[key]}
                for key in sorted(extension_counts)
            },
        },
        "duplicate_snapshots": duplicate_snapshots,
        "schema_sample_method": {
            "selection": "lexicographically smallest SHA256(task_id) values",
            "requested_count": schema_sample_count,
            "redaction": "values omitted; unknown field names represented only by SHA256",
            "scope": "fixed task sample only; not full completion-schema validation",
        },
        "schema_samples": schema_output,
        "readiness": {
            "inventory_complete": True,
            "archive_identity_verified": True,
            "expected_four_runs_present": expected_runs_present,
            "expected_500_tasks_present": task_count == expected_task_count,
            "all_tasks_have_exactly_four_runs": coverage_complete,
            "all_task_runs_have_completions": missing_completion_task_runs == 0,
            "all_task_runs_have_exactly_one_report": report_ready,
            "reports_complete": report_ready,
            "aggregate_reports_complete": aggregate_reports_complete,
            "aggregate_report_task_sets_match": aggregate_task_sets_match,
            "schema_sample_complete": schema_sample_complete,
            "schema_sample_scope": "sample_only",
            "full_completion_schema_validated": False,
            "full_archive_schema_validated": False,
            "formal_jsonl_files_complete": formal_jsonl_files_complete,
            "formal_jsonl_records_complete": formal_jsonl_records_complete,
            "formal_jsonl_task_sets_match": formal_jsonl_task_sets_match,
            "formal_jsonl_full_schema_validated": formal_jsonl_ready,
            "partial_trajectory_ingestion_ready": formal_jsonl_ready,
            "trajectory_ingestion_ready": trajectory_ready,
            "termination_labels_ready": False,
            "accuracy_observed_subset_ready": (
                evaluator_status_counts["observed"] > 0
            ),
            "accuracy_labels_ready": evaluator_full_coverage,
            "accuracy_full_coverage_ready": evaluator_full_coverage,
            "missing_reports_are_failures": False,
            "missing_report_policy": "missing; never impute unresolved/zero",
            "overall_ready": False,
            "overall_ready_reason": (
                "full completion schema and uniform explicit task termination are not "
                "validated; sample completeness is not a full-valid signal"
            ),
        },
        "anomalies": anomalies,
    }


def build_inventory(
    archive_path: Path,
    *,
    expected_bytes: int | None = EXPECTED_BYTES,
    expected_sha256: str | None = EXPECTED_SHA256,
    expected_wrapper: str = ARCHIVE_WRAPPER,
    expected_task_count: int = EXPECTED_TASKS,
    schema_sample_count: int = DEFAULT_SCHEMA_SAMPLE_COUNT,
    max_schema_json_bytes: int = DEFAULT_MAX_SCHEMA_JSON_BYTES,
    max_jsonl_line_bytes: int = MAX_JSONL_LINE_BYTES,
) -> dict[str, Any]:
    """Audit a gzip tar archive without extracting or materializing its member list."""
    archive_path = Path(archive_path)
    if schema_sample_count <= 0:
        raise ValueError("schema_sample_count must be positive")
    if max_schema_json_bytes <= 0:
        raise ValueError("max_schema_json_bytes must be positive")
    if expected_task_count <= 0:
        raise ValueError("expected_task_count must be positive")
    if max_jsonl_line_bytes <= 0:
        raise ValueError("max_jsonl_line_bytes must be positive")
    if not archive_path.is_file():
        raise OpenHandsArchiveAuditError("archive is missing")
    archive_bytes = archive_path.stat().st_size
    if expected_bytes is not None and archive_bytes != expected_bytes:
        raise OpenHandsArchiveAuditError(
            f"archive size mismatch: expected {expected_bytes}, got {archive_bytes}"
        )

    member_count = 0
    directory_count = 0
    regular_file_count = 0
    regular_file_bytes = 0
    runs_by_id: dict[int, str] = {}
    run_ids_by_name: dict[str, int] = {}
    run_member_counts: Counter[int] = Counter()
    run_file_counts: Counter[int] = Counter()
    run_bytes: Counter[int] = Counter()
    extension_counts: Counter[str] = Counter()
    extension_bytes: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()
    template_bytes: Counter[str] = Counter()
    depth_counts: Counter[int] = Counter()
    category_counts: Counter[str] = Counter()
    category_bytes: Counter[str] = Counter()
    samples: dict[str, dict[str, Any]] = {}
    aggregate_reports: dict[int, dict[str, int]] = {}

    with _workspace_temp_environment(DEFAULT_TEMP_PARENT), tempfile.TemporaryDirectory(
        prefix="openhands-archive-audit-", dir=DEFAULT_TEMP_PARENT
    ) as temporary:
        database_path = Path(temporary) / "inventory.sqlite3"
        connection = _open_database(database_path)
        hashing_reader: _HashingReader | None = None
        try:
            with archive_path.open("rb") as raw:
                hashing_reader = _HashingReader(raw)
                try:
                    with tarfile.open(fileobj=hashing_reader, mode="r|gz") as archive:
                        for member in archive:
                            member_count += 1
                            if member_count > MAX_MEMBER_COUNT:
                                raise OpenHandsArchiveAuditError(
                                    "archive exceeds the explicit member-count safety limit"
                                )
                            parts = _safe_parts(member.name)
                            depth_counts[len(parts)] += 1
                            if parts[0] != expected_wrapper:
                                raise OpenHandsArchiveAuditError(
                                    "archive member is outside the expected wrapper directory"
                                )
                            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                                raise OpenHandsArchiveAuditError(
                                    "archive contains a link, device, or FIFO member"
                                )
                            if not (member.isdir() or member.isfile()):
                                raise OpenHandsArchiveAuditError(
                                    "archive contains an unsupported tar member type"
                                )
                            if member.size < 0 or member.size > MAX_MEMBER_BYTES:
                                raise OpenHandsArchiveAuditError(
                                    "archive member exceeds the explicit size safety limit"
                                )
                            if member.isfile() and member.sparse is not None:
                                raise OpenHandsArchiveAuditError(
                                    "sparse archive members are not supported"
                                )

                            if len(parts) == 1:
                                if not member.isdir():
                                    raise OpenHandsArchiveAuditError(
                                        "archive wrapper must be a directory"
                                    )
                                directory_count += 1
                                continue
                            run_basename = parts[1]
                            run_id = _parse_run_id(run_basename)
                            existing_name = runs_by_id.get(run_id)
                            if existing_name is not None and existing_name != run_basename:
                                raise OpenHandsArchiveAuditError(
                                    "one run id maps to multiple run directory basenames"
                                )
                            existing_id = run_ids_by_name.get(run_basename)
                            if existing_id is not None and existing_id != run_id:
                                raise OpenHandsArchiveAuditError(
                                    "one run directory basename maps to multiple run ids"
                                )
                            runs_by_id[run_id] = run_basename
                            run_ids_by_name[run_basename] = run_id
                            run_member_counts[run_id] += 1

                            if len(parts) == 2:
                                if not member.isdir():
                                    raise OpenHandsArchiveAuditError(
                                        "run root must be represented by a directory member"
                                    )
                                directory_count += 1
                                continue

                            if member.isdir():
                                directory_count += 1
                                task_id, _ = _task_identity(parts, is_file=False)
                                if task_id is not None:
                                    task_hash = _hash_text(task_id)
                                    _ensure_task_run(connection, task_hash, run_id)
                                    _consider_sample(samples, task_hash, schema_sample_count)
                                continue

                            regular_file_count += 1
                            regular_file_bytes += member.size
                            if regular_file_bytes > MAX_UNCOMPRESSED_BYTES:
                                raise OpenHandsArchiveAuditError(
                                    "archive exceeds the explicit uncompressed-byte safety limit"
                                )
                            run_file_counts[run_id] += 1
                            run_bytes[run_id] += member.size
                            task_id, category = _task_identity(parts, is_file=True)
                            task_hash = _hash_text(task_id) if task_id is not None else None
                            selected_sample = task_hash is not None and _consider_sample(
                                samples, task_hash, schema_sample_count
                            )
                            path_hash = _hash_text(member.name)
                            _record_file(
                                connection,
                                path_hash=path_hash,
                                size_bytes=member.size,
                                run_id=run_id,
                                task_hash=task_hash,
                                category=category,
                            )
                            extension = _safe_extension(parts[-1])
                            extension_counts[extension] += 1
                            extension_bytes[extension] += member.size
                            template = _path_template(category, is_file=True)
                            template_counts[template] += 1
                            template_bytes[template] += member.size
                            category_counts[category] += 1
                            category_bytes[category] += member.size

                            if category == "llm_completion":
                                content_hash, payload = _read_member(
                                    archive,
                                    member,
                                    retain=selected_sample,
                                    max_retained_bytes=max_schema_json_bytes,
                                )
                                connection.execute(
                                    "INSERT INTO completions VALUES (?, ?, ?, ?)",
                                    (content_hash, member.size, task_hash, run_id),
                                )
                                if selected_sample:
                                    if payload is None or task_id is None or task_hash is None:
                                        raise AssertionError("sample state is internally inconsistent")
                                    value = _parse_json(
                                        payload, kind="completion", task_id=task_id
                                    )
                                    _record_schema(
                                        samples,
                                        task_hash=task_hash,
                                        kind="completion",
                                        value=value,
                                    )
                            elif category == "report" and task_id is not None:
                                _, payload = _read_member(
                                    archive,
                                    member,
                                    retain=True,
                                    max_retained_bytes=max_schema_json_bytes,
                                )
                                if payload is None or task_hash is None:
                                    raise AssertionError("sample state is internally inconsistent")
                                value = _parse_json(payload, kind="report", task_id=task_id)
                                report = value[task_id]
                                connection.execute(
                                    """
                                    UPDATE task_runs
                                    SET report_resolved = ?, report_schema_hash = ?
                                    WHERE task_hash = ? AND run_id = ?
                                    """,
                                    (
                                        int(report["resolved"]),
                                        _schema_hash(report),
                                        task_hash,
                                        run_id,
                                    ),
                                )
                                if selected_sample:
                                    _record_schema(
                                        samples,
                                        task_hash=task_hash,
                                        kind="report",
                                        value=report,
                                    )
                            elif category == "run_report":
                                _, payload = _read_member(
                                    archive,
                                    member,
                                    retain=True,
                                    max_retained_bytes=max_schema_json_bytes,
                                )
                                if payload is None:
                                    raise AssertionError("report payload was not retained")
                                value = _parse_json(
                                    payload, kind="aggregate_report", task_id=""
                                )
                                if run_id in aggregate_reports:
                                    raise OpenHandsArchiveAuditError(
                                        "run contains multiple aggregate report files"
                                    )
                                aggregate_reports[run_id] = _aggregate_report_stats(value)
                                for submitted_task_id in value["submitted_ids"]:
                                    connection.execute(
                                        """
                                        INSERT INTO aggregate_submitted_tasks VALUES (?, ?)
                                        """,
                                        (_hash_text(submitted_task_id), run_id),
                                    )
                                for outcome_name, field in (
                                    ("resolved", "resolved_ids"),
                                    ("unresolved", "unresolved_ids"),
                                    ("empty_patch", "empty_patch_ids"),
                                    ("error", "error_ids"),
                                    ("incomplete", "incomplete_ids"),
                                ):
                                    for outcome_task_id in value[field]:
                                        connection.execute(
                                            """
                                            INSERT INTO aggregate_task_outcomes VALUES (?, ?, ?)
                                            """,
                                            (
                                                _hash_text(outcome_task_id),
                                                run_id,
                                                OUTCOME_CODES[outcome_name],
                                            ),
                                        )
                            elif category in JSONL_KIND_CODES:
                                _audit_jsonl_member(
                                    connection,
                                    archive,
                                    member,
                                    path_hash=path_hash,
                                    run_id=run_id,
                                    category=category,
                                    max_line_bytes=max_jsonl_line_bytes,
                                )

                    # Drain any compressed trailing bytes not consumed by tarfile. This keeps
                    # archive hashing and tar iteration to one physical sequential pass.
                    while hashing_reader.read(CHUNK_BYTES):
                        pass
                except (tarfile.TarError, EOFError, OSError) as exc:
                    raise OpenHandsArchiveAuditError(
                        "archive is not a complete readable gzip tar stream"
                    ) from exc

            if hashing_reader is None:
                raise AssertionError("hashing reader was not initialized")
            if hashing_reader.bytes_read != archive_bytes:
                raise OpenHandsArchiveAuditError(
                    "archive byte count changed or was not consumed completely"
                )
            archive_sha256 = hashing_reader.hexdigest()
            if expected_sha256 is not None and archive_sha256 != expected_sha256:
                raise OpenHandsArchiveAuditError(
                    "archive SHA-256 mismatch: refusing to emit an inventory"
                )
            if not runs_by_id:
                raise OpenHandsArchiveAuditError("archive contains no recognized runs")
            if frozenset(runs_by_id) != EXPECTED_RUN_IDS:
                raise OpenHandsArchiveAuditError(
                    "archive must contain exactly the four run identities 1, 2, 3, and 4"
                )
            connection.commit()
            return _finalize_inventory(
                connection=connection,
                archive_path=archive_path,
                archive_bytes=archive_bytes,
                archive_sha256=archive_sha256,
                member_count=member_count,
                directory_count=directory_count,
                regular_file_count=regular_file_count,
                regular_file_bytes=regular_file_bytes,
                runs_by_id=runs_by_id,
                run_member_counts=run_member_counts,
                run_file_counts=run_file_counts,
                run_bytes=run_bytes,
                extension_counts=extension_counts,
                extension_bytes=extension_bytes,
                template_counts=template_counts,
                template_bytes=template_bytes,
                depth_counts=depth_counts,
                category_counts=category_counts,
                category_bytes=category_bytes,
                samples=samples,
                aggregate_report_stats=aggregate_reports,
                expected_wrapper=expected_wrapper,
                schema_sample_count=schema_sample_count,
                expected_task_count=expected_task_count,
                max_jsonl_line_bytes=max_jsonl_line_bytes,
            )
        finally:
            connection.close()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream-audit the pinned Spend GPT-5.2 OpenHands archive."
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--schema-sample-count",
        type=int,
        choices=range(1, 6),
        default=DEFAULT_SCHEMA_SAMPLE_COUNT,
    )
    args = parser.parse_args()
    inventory = build_inventory(
        args.archive,
        schema_sample_count=args.schema_sample_count,
    )
    atomic_write_json(args.output, inventory)
    print(
        json.dumps(
            {
                "archive_path": inventory["archive_path"],
                "archive_sha256": inventory["archive_sha256"],
                "output_path": _relative_archive_path(args.output),
                "member_count": inventory["member_count"],
                "task_count": inventory["task_count"],
                "task_run_count": inventory["task_run_count"],
                "llm_completions_count": inventory["llm_completions_count"],
                "report_count": inventory["report_count"],
                "overall_ready": inventory["readiness"]["overall_ready"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
