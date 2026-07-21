from __future__ import annotations

import unittest
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from token_prediction.collection import CodexTurnMetadata, CodexTurnReader
from token_prediction.contracts import EventType, Observable
from token_prediction.dataset import (
    PredictionPosition,
    PredictionTarget,
    build_supervised_dataset,
)
from token_prediction.cli import main
from token_prediction.pipeline import load_trajectory


FIXTURE = Path(__file__).parent / "fixtures" / "codex_turn_events.jsonl"


class CodexTurnReaderTests(unittest.TestCase):
    def test_turn_reader_declares_only_observed_task_usage(self) -> None:
        reader = CodexTurnReader()
        self.assertEqual(reader.capabilities.observables, frozenset({Observable.TASK_USAGE}))

    def test_codex_turn_builds_task_total_without_fabricating_calls(self) -> None:
        trajectory = CodexTurnReader().read(
            FIXTURE,
            CodexTurnMetadata(
                task_id="codex-task",
                task_tokens=7,
                model_id="gpt-fixture",
                reasoning_effort="medium",
                started_at="2026-07-21T00:00:00+00:00",
                finished_at="2026-07-21T00:00:01+00:00",
            ),
        )
        self.assertEqual(
            [event.event_type for event in trajectory.events],
            [EventType.TASK_STARTED, EventType.TASK_FINISHED],
        )
        dataset = build_supervised_dataset((trajectory,))
        task = dataset.select(
            PredictionPosition.TASK_LAUNCH,
            PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
        )
        calls = dataset.select(
            PredictionPosition.CALL_PRE,
            PredictionTarget.CALL_UNKNOWN_BILLABLE_TOKENS,
        )
        self.assertEqual(len(task.rows), 1)
        self.assertEqual(task.rows[0].label, 12411)
        self.assertEqual(len(calls.rows), 0)
        finish_usage = trajectory.events[-1].payload["usage"]
        self.assertEqual(finish_usage["cached_input_tokens"], 9984)
        self.assertEqual(finish_usage["reasoning_output_tokens"], 0)

    def test_cli_ingest_writes_a_canonical_task_only_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "canonical.jsonl"
            with redirect_stdout(StringIO()):
                code = main(
                    [
                        "ingest",
                        "codex-turn",
                        "--raw",
                        str(FIXTURE),
                        "--output",
                        str(output),
                        "--task-id",
                        "cli-task",
                        "--started-at",
                        "2026-07-21T00:00:00+00:00",
                        "--finished-at",
                        "2026-07-21T00:00:01+00:00",
                    ]
                )
            self.assertEqual(code, 0)
            trajectory = load_trajectory(output)
            self.assertEqual(len(trajectory.events), 2)


if __name__ == "__main__":
    unittest.main()
