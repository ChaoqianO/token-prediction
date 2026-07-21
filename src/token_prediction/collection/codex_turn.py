from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from token_prediction.contracts import (
    CanonicalEvent,
    EventType,
    Observable,
    SourceCapabilities,
)
from token_prediction.trajectory import Trajectory


@dataclass(frozen=True)
class CodexTurnMetadata:
    task_id: str
    started_at: str
    finished_at: str
    task_tokens: int | None = None
    task_hash: str | None = None
    run_id: str | None = None
    condition_id: str | None = None
    model_id: str | None = None
    resolved_model_id: str | None = None
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id is required")
        if self.task_tokens is not None and self.task_tokens < 0:
            raise ValueError("task_tokens must be non-negative or missing")


class CodexTurnReader:
    """Normalize the verified turn-level subset of ``codex exec --json``.

    It emits no request, Call, attempt, tool, or generation event. The reader is
    intentionally unable to create those boundaries from item/error events.
    """

    source_id = "codex_exec_jsonl_turn_v1"
    capabilities = SourceCapabilities(
        source_id=source_id,
        observables=frozenset({Observable.TASK_USAGE}),
        source="observed",
    )

    def read(self, location: Path, metadata: CodexTurnMetadata) -> Trajectory:
        path = Path(location).resolve()
        raw = path.read_bytes()
        raw_hash = hashlib.sha256(raw).hexdigest()
        events: list[dict[str, object]] = []
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid Codex JSONL line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Codex JSONL line {line_number} is not an object")
            events.append(value)
        types = Counter(str(event.get("type") or "") for event in events)
        completed = [event for event in events if event.get("type") == "turn.completed"]
        if len(completed) > 1:
            raise ValueError("one raw Codex file must contain at most one completed turn")
        stable = hashlib.sha256(
            f"{metadata.task_id}\0{raw_hash}".encode("utf-8")
        ).hexdigest()[:24]
        trajectory_id = f"codex-turn-{stable}"
        start_payload = {
            "task_id": metadata.task_id,
            "run_id": metadata.run_id or trajectory_id,
            "prediction_context_id": f"task:{metadata.task_id}:initial",
            "condition_id": metadata.condition_id,
            "task_tokens": metadata.task_tokens,
            "task_hash": metadata.task_hash,
            "agent_id": "codex_cli",
            "model_id": metadata.model_id,
            "resolved_model_id": metadata.resolved_model_id,
            "reasoning_effort": metadata.reasoning_effort,
            "raw_format": self.source_id,
            "raw_sha256": raw_hash,
        }
        canonical = [
            CanonicalEvent.create(
                event_id=f"{trajectory_id}-start",
                trajectory_id=trajectory_id,
                event_seq=0,
                event_type=EventType.TASK_STARTED,
                occurred_at=metadata.started_at,
                raw_ref=str(path),
                payload=start_payload,
            )
        ]
        if completed:
            raw_usage = completed[0].get("usage")
            canonical.append(
                CanonicalEvent.create(
                    event_id=f"{trajectory_id}-finish",
                    trajectory_id=trajectory_id,
                    event_seq=1,
                    event_type=EventType.TASK_FINISHED,
                    occurred_at=metadata.finished_at,
                    raw_ref=str(path),
                    payload={
                        "outcome": "completed",
                        "reason": "codex_turn_completed",
                        "usage": raw_usage if isinstance(raw_usage, dict) else None,
                        "raw_event_type_counts": dict(sorted(types.items())),
                    },
                )
            )
        else:
            canonical.append(
                CanonicalEvent.create(
                    event_id=f"{trajectory_id}-abort",
                    trajectory_id=trajectory_id,
                    event_seq=1,
                    event_type=EventType.TASK_ABORTED,
                    occurred_at=metadata.finished_at,
                    raw_ref=str(path),
                    payload={
                        "outcome": "unknown",
                        "reason": "logging_incomplete",
                        "raw_event_type_counts": dict(sorted(types.items())),
                    },
                )
            )
        return Trajectory.from_events(canonical)
