from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from token_prediction.contracts import CanonicalEvent, EventType
from token_prediction.recording import EventStoreError, SQLiteEventStore


class RecordingTests(unittest.TestCase):
    def test_store_is_append_only_and_redacts_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.sqlite"
            event = CanonicalEvent.create(
                trajectory_id="t1",
                event_seq=0,
                event_type=EventType.TASK_STARTED,
                payload={
                    "input_tokens": 4,
                    "access_token": "secret-value",
                    "header": "Authorization: Bearer abc.def",
                },
            )
            with SQLiteEventStore(path) as store:
                store.append(event)
                stored = list(store.iter_events("t1"))[0]
                self.assertEqual(stored.payload["input_tokens"], 4)
                self.assertEqual(stored.payload["access_token"], "[REDACTED]")
                self.assertNotIn("abc.def", stored.payload["header"])
                with self.assertRaises(EventStoreError):
                    store.append(event)

    def test_non_monotonic_sequence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with SQLiteEventStore(Path(temporary) / "events.sqlite") as store:
                store.append(
                    CanonicalEvent.create(
                        trajectory_id="t1",
                        event_seq=2,
                        event_type=EventType.TASK_STARTED,
                    )
                )
                with self.assertRaises(EventStoreError):
                    store.append(
                        CanonicalEvent.create(
                            trajectory_id="t1",
                            event_seq=1,
                            event_type=EventType.TASK_FINISHED,
                        )
                    )


if __name__ == "__main__":
    unittest.main()
