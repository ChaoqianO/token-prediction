from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from token_prediction.collection.source import (
    CollectedTrajectory,
    CollectionTask,
    EventSink,
)
from token_prediction.contracts import CanonicalEvent, Observable, SourceCapabilities


class ReplayTrajectorySource:
    """Deterministically emits an already-normalized canonical trajectory."""

    source_id = "canonical_jsonl"
    capabilities = SourceCapabilities(
        source_id=source_id,
        observables=frozenset(
            {
                Observable.CALL_USAGE,
                Observable.ATTEMPT_USAGE,
                Observable.REQUEST_LOCAL_COUNT,
                Observable.TOOL_EVENTS,
            }
        ),
        source="fixture",
    )

    def __init__(self, events: Iterable[CanonicalEvent]) -> None:
        self._events = tuple(sorted(events, key=lambda event: event.event_seq))

    def collect(self, task: CollectionTask, sink: EventSink) -> CollectedTrajectory:
        del task
        if not self._events:
            raise ValueError("replay requires at least one event")
        trajectory_ids = {event.trajectory_id for event in self._events}
        if len(trajectory_ids) != 1:
            raise ValueError("one replay source must contain one trajectory")
        for event in self._events:
            sink.append(event)
        final = self._events[-1]
        payload = final.payload
        return CollectedTrajectory(
            trajectory_id=final.trajectory_id,
            outcome=str(payload.get("outcome") or "unknown"),
            termination_reason=str(payload.get("reason") or final.event_type.value),
            event_count=len(self._events),
        )


def write_canonical_jsonl(path: str | Path, events: Iterable[CanonicalEvent]) -> Path:
    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite canonical trajectory: {destination}")
    materialized = tuple(events)
    if not materialized:
        raise ValueError("cannot write an empty canonical trajectory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(
            json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True)
            for event in materialized
        )
        + "\n",
        encoding="utf-8",
    )
    return destination
