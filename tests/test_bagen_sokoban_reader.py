from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from token_prediction.collection import BagenSokobanReader
from token_prediction.contracts import EventType, Observable
from token_prediction.dataset import (
    LabelStatus,
    PredictionPosition,
    PredictionTarget,
    build_supervised_dataset,
)
from token_prediction.features import replay_feature_snapshots


def _rollout(absolute_env_id: int, output_delta: int = 0) -> dict[str, object]:
    turns: list[dict[str, object]] = []
    for index, (input_tokens, output_tokens) in enumerate(
        ((100, 10 + output_delta), (100, 30 + output_delta)), start=1
    ):
        turns.append(
            {
                "turn_idx": index,
                "api_input_tokens": input_tokens,
                "api_output_tokens": output_tokens,
                "api_total_tokens": input_tokens + output_tokens,
                "api_interactions": [
                    {
                        "attempt": 1,
                        "success": True,
                        "provider": "openai",
                        "model": "gpt-test",
                        "request_id": f"request-{absolute_env_id}-{index}",
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    }
                ],
                "actions": [1],
                "action_names": ["Up"],
                "toolcalls_used": 1,
                "success": index == 2,
            }
        )
    total_input = sum(int(turn["api_input_tokens"]) for turn in turns)
    total_output = sum(int(turn["api_output_tokens"]) for turn in turns)
    return {
        "env_id": 7,
        "absolute_env_id": absolute_env_id,
        "tag": "CoordSokoban",
        "initial_state": "same puzzle board",
        "final_state": "solved board",
        "api_input_tokens": total_input,
        "api_output_tokens": total_output,
        "api_total_tokens": total_input + total_output,
        "turns": turns,
    }


class BagenSokobanReaderTests(unittest.TestCase):
    def test_repeated_rollouts_share_task_but_not_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dialogues.json"
            path.write_text(
                json.dumps([_rollout(0), _rollout(1, output_delta=5)]),
                encoding="utf-8",
            )
            trajectories = BagenSokobanReader().read_all(path)

        self.assertEqual(len(trajectories), 2)
        self.assertEqual(trajectories[0].task_id, trajectories[1].task_id)
        self.assertNotEqual(
            trajectories[0].trajectory_id, trajectories[1].trajectory_id
        )
        dataset = build_supervised_dataset(trajectories)
        updates = tuple(
            row
            for row in dataset.rows
            if row.point.position == PredictionPosition.TASK_UPDATE
            and row.point.target
            == PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS
        )
        self.assertEqual(len(updates), 2)
        first = next(
            row
            for row in updates
            if row.point.trajectory_id == trajectories[0].trajectory_id
        )
        self.assertIsNone(first.label)
        self.assertEqual(first.status, LabelStatus.MISSING)
        self.assertEqual(first.invalid_reason, "missing_request_tokens_local")
        self.assertEqual(first.point.features["completed_call_count"], 1)
        self.assertEqual(
            first.point.features["cumulative_provider_input_tokens"], 100
        )
        self.assertEqual(first.point.features["cumulative_provider_output_tokens"], 10)
        self.assertIsNone(first.point.features["current_request_tokens_local"])
        self.assertIsNone(first.point.known_offset_tokens)

        requests = [
            event
            for event in trajectories[0].events
            if event.event_type == EventType.REQUEST_BUILT
        ]
        self.assertEqual(
            [event.payload["request_tokens_local"] for event in requests],
            [None, None],
        )
        terminals = [
            event
            for event in trajectories[0].events
            if event.event_type == EventType.API_COMPLETED
        ]
        self.assertEqual(
            [
                event.payload["provider_input_tokens_post_response_audit"]
                for event in terminals
            ],
            [100, 100],
        )

    def test_capabilities_advertise_boundaries_and_termination_not_local_count(
        self,
    ) -> None:
        observables = BagenSokobanReader.capabilities.observables
        self.assertIn(Observable.REQUEST_BOUNDARIES, observables)
        self.assertIn(Observable.TASK_TERMINATION, observables)
        self.assertNotIn(Observable.REQUEST_LOCAL_COUNT, observables)

    def test_current_response_usage_changes_only_post_response_audit_and_labels(
        self,
    ) -> None:
        base_rollout = _rollout(0)
        changed_rollout = copy.deepcopy(base_rollout)
        changed_turn = changed_rollout["turns"][0]  # type: ignore[index]
        changed_interaction = changed_turn["api_interactions"][0]  # type: ignore[index]
        changed_interaction["input_tokens"] = 900  # type: ignore[index]
        changed_interaction["total_tokens"] = 910  # type: ignore[index]
        changed_turn["api_input_tokens"] = 900  # type: ignore[index]
        changed_turn["api_total_tokens"] = 910  # type: ignore[index]
        changed_rollout["api_input_tokens"] = 1_000
        changed_rollout["api_total_tokens"] = 1_040

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path = root / "base.json"
            changed_path = root / "changed.json"
            base_path.write_text(json.dumps([base_rollout]), encoding="utf-8")
            changed_path.write_text(json.dumps([changed_rollout]), encoding="utf-8")
            reader = BagenSokobanReader()
            base = reader.read_all(base_path)[0]
            changed = reader.read_all(changed_path)[0]

        base_snapshots = replay_feature_snapshots(
            base.events, include_task_started=True
        )
        changed_snapshots = replay_feature_snapshots(
            changed.events, include_task_started=True
        )
        self.assertEqual(
            [snapshot.feature_hash for snapshot in base_snapshots[:2]],
            [snapshot.feature_hash for snapshot in changed_snapshots[:2]],
        )
        self.assertIsNone(base_snapshots[0].values["task_tokens"])
        self.assertIsNone(base_snapshots[1].values["current_request_tokens_local"])

        base_request = next(
            event for event in base.events if event.event_type == EventType.REQUEST_BUILT
        )
        changed_request = next(
            event
            for event in changed.events
            if event.event_type == EventType.REQUEST_BUILT
        )
        self.assertEqual(base_request.payload, changed_request.payload)
        base_terminal = next(
            event for event in base.events if event.event_type == EventType.API_COMPLETED
        )
        changed_terminal = next(
            event
            for event in changed.events
            if event.event_type == EventType.API_COMPLETED
        )
        self.assertEqual(
            base_terminal.payload["provider_input_tokens_post_response_audit"], 100
        )
        self.assertEqual(
            changed_terminal.payload["provider_input_tokens_post_response_audit"],
            900,
        )

        base_total = build_supervised_dataset((base,)).select(
            PredictionPosition.TASK_LAUNCH,
            PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
        ).rows[0]
        changed_total = build_supervised_dataset((changed,)).select(
            PredictionPosition.TASK_LAUNCH,
            PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
        ).rows[0]
        self.assertEqual(base_total.label, 240)
        self.assertEqual(changed_total.label, 1_040)


if __name__ == "__main__":
    unittest.main()
