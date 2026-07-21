"""Trajectory acquisition and deterministic ingestion boundaries."""

from .codex_cli import CodexAuthState, CodexAuthStatus, CodexCLI, CodexCLIError
from .codex_turn import CodexTurnMetadata, CodexTurnReader
from .bagen_sokoban import (
    BagenSokobanMetadata,
    BagenSokobanReader,
    BagenSokobanSchemaError,
)
from .bagen_swebench import (
    BagenSwebenchMetadata,
    BagenSwebenchReader,
    BagenSwebenchSchemaError,
)
from .openhands_trajectory import (
    OpenHandsArchiveError,
    OpenHandsArchiveMetadata,
    OpenHandsArchiveReader,
    OpenHandsArchiveSchemaError,
)
from .replay import ReplayTrajectorySource, write_canonical_jsonl
from .source import (
    CollectedTrajectory,
    CollectionTask,
    EventSink,
    TrajectoryReader,
    TrajectorySource,
)

__all__ = [
    "CodexAuthState",
    "CodexAuthStatus",
    "CodexCLI",
    "CodexCLIError",
    "CodexTurnMetadata",
    "CodexTurnReader",
    "BagenSokobanMetadata",
    "BagenSokobanReader",
    "BagenSokobanSchemaError",
    "BagenSwebenchMetadata",
    "BagenSwebenchReader",
    "BagenSwebenchSchemaError",
    "OpenHandsArchiveError",
    "OpenHandsArchiveMetadata",
    "OpenHandsArchiveReader",
    "OpenHandsArchiveSchemaError",
    "CollectedTrajectory",
    "CollectionTask",
    "EventSink",
    "ReplayTrajectorySource",
    "TrajectorySource",
    "TrajectoryReader",
    "write_canonical_jsonl",
]
