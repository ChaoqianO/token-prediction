from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from token_prediction.contracts import CanonicalEvent, SourceCapabilities

if TYPE_CHECKING:
    from token_prediction.trajectory import Trajectory


class EventSink(Protocol):
    def append(self, event: CanonicalEvent) -> None: ...


@dataclass(frozen=True)
class CollectionTask:
    task_id: str
    prompt: str
    workspace: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CollectedTrajectory:
    trajectory_id: str
    outcome: str
    termination_reason: str
    event_count: int


class TrajectorySource(Protocol):
    """A source of canonical trajectory events.

    Live collectors and deterministic readers may both implement this protocol,
    but a source must declare the facts it really observes.  It must never
    synthesize a missing request or call boundary merely to satisfy a dataset.
    """

    source_id: str
    capabilities: SourceCapabilities

    def collect(self, task: CollectionTask, sink: EventSink) -> CollectedTrajectory: ...


class TrajectoryReader(Protocol):
    """Deterministically normalizes a preserved raw run."""

    source_id: str
    capabilities: SourceCapabilities

    def read(self, location: Path, metadata: Any) -> "Trajectory": ...
