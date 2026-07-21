from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.audit_openhands_trajectory import (
    OpenHandsTrajectoryAuditError,
    _source_capability,
    _trajectory_metrics,
    atomic_write_json,
    build_trajectory_audit,
)
from token_prediction.collection.openhands_trajectory import (
    OpenHandsArchiveMetadata,
    OpenHandsArchiveReader,
)
from token_prediction.contracts import CanonicalEvent, EventType
from token_prediction.dataset import (
    LabelStatus,
    PredictionPosition,
    PredictionTarget,
    build_supervised_dataset,
)
from token_prediction.trajectory import Trajectory
from tests.test_openhands_trajectory_reader import (
    TASK_ID,
    _completion,
    _completion_name,
    _jsonl_bytes,
    _member,
    _report,
    _report_member,
    _task_log_line,
    _task_log_member,
    _write_archive,
)


TEST_HUB_REPO = "fixture/openhands-trajectories"
TEST_REVISION = "1" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_archive(path: Path) -> Path:
    members: list[tuple[str, object]] = []
    for run in range(1, 5):
        response_id = f"response-run-{run}"
        task_log = _task_log_line(
            TASK_ID,
            response_usages=((response_id, 10, 3),),
            finished=run in {1, 2},
            error="RAW_TASK_ERROR_SENTINEL" if run == 3 else None,
        )
        members.append((_task_log_member(run), _jsonl_bytes(task_log)))
        if run == 1:
            members.append((_report_member(run, TASK_ID), _report(TASK_ID)))
        completion = _completion(
            timestamp=1.0,
            created=1_700_000_000 + run,
            response_id=response_id,
            input_tokens=10,
            output_tokens=3,
        )
        if run == 2:
            completion["response"]["usage"] = None  # type: ignore[index]
        if run == 4:
            completion["response"]["choices"][0]["provider_specific_fields"][  # type: ignore[index]
                "error"
            ] = {
                "message": "RAW_PROVIDER_ERROR_SENTINEL",
                "code": 500,
                "metadata": {
                    "provider_name": "openai",
                    "raw": {"code": "upstream", "message": "RAW_ERROR_SENTINEL"},
                },
            }
        members.append(
            (_member(run, TASK_ID, _completion_name(1.0)), completion)
        )
    return _write_archive(path, members)


def _write_inventory(path: Path, archive: Path) -> Path:
    value = {
        "inventory_schema_version": 2,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": _sha256(archive),
        "hub_repo": TEST_HUB_REPO,
        "resolved_revision": TEST_REVISION,
        "task_count": 1,
        "task_run_count": 4,
        "llm_completions_count": 4,
        "report_count": 1,
        "task_report_count": 1,
        "aggregate_report_count": 4,
        "output_jsonl_record_count": 4,
        "runs": [{"run_id": run} for run in range(1, 5)],
    }
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _build(archive: Path, inventory: Path) -> dict[str, object]:
    return build_trajectory_audit(
        archive,
        inventory,
        expected_archive_bytes=archive.stat().st_size,
        expected_archive_sha256=_sha256(archive),
        expected_hub_repo=TEST_HUB_REPO,
        expected_revision=TEST_REVISION,
        expected_run_ids=("run_1", "run_2", "run_3", "run_4"),
        sqlite_parent=archive.parent / "sqlite-tmp",
    )


def _zero_call_trajectory(
    *,
    usage: dict[str, object] | None = None,
    usage_scope: str = "explicit_zero_call_task",
    completion_snapshot_count: int = 0,
) -> Trajectory:
    terminal_usage = usage or {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_source": "output_metrics_accumulated_token_usage",
        "usage_scope": usage_scope,
    }
    events = (
        CanonicalEvent.create(
            event_id="zero-call-started",
            trajectory_id="zero-call-trajectory",
            event_seq=0,
            event_type=EventType.TASK_STARTED,
            occurred_at="2026-07-21T00:00:00+00:00",
            payload={
                "task_id": "zero-call-task",
                "run_id": "run_2",
                "condition_id": "condition:fixture",
            },
        ),
        CanonicalEvent.create(
            event_id="zero-call-terminal",
            trajectory_id="zero-call-trajectory",
            event_seq=1,
            event_type=EventType.TASK_ABORTED,
            occurred_at="2026-07-21T00:00:01+00:00",
            payload={
                "outcome": "error",
                "reason": "task_error",
                "usage": terminal_usage,
                "usage_scope": usage_scope,
                "completion_snapshot_count": completion_snapshot_count,
            },
        ),
    )
    return Trajectory.from_events(events)


class OpenHandsTrajectoryAuditTests(unittest.TestCase):
    def test_explicit_zero_call_usage_is_source_reported_and_never_imputed(
        self,
    ) -> None:
        metrics, _ = _trajectory_metrics(_zero_call_trajectory())

        self.assertEqual(metrics["task_usage_complete_count"], 1)
        self.assertEqual(metrics["task_usage_explicit_zero_call_count"], 1)
        capability = _source_capability(OpenHandsArchiveReader(), metrics)["task_usage"]
        self.assertEqual(capability["explicit_zero_call_count"], 1)
        self.assertEqual(
            capability["explicit_zero_call_source"],
            "output.metrics.accumulated_token_usage",
        )
        self.assertTrue(capability["explicit_zero_call_never_imputed"])

        missing_metrics, _ = _trajectory_metrics(
            _zero_call_trajectory(
                usage={
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "total_source": "missing",
                    "usage_scope": "missing_no_completion_or_task_metrics",
                },
                usage_scope="missing_no_completion_or_task_metrics",
            )
        )
        self.assertEqual(missing_metrics["task_usage_missing_count"], 1)
        self.assertEqual(missing_metrics["task_usage_explicit_zero_call_count"], 0)

    def test_explicit_zero_call_usage_fails_closed_on_inconsistent_evidence(
        self,
    ) -> None:
        invalid_cases = (
            {
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 1,
                    "total_tokens": 1,
                    "total_source": "output_metrics_accumulated_token_usage",
                    "usage_scope": "explicit_zero_call_task",
                }
            },
            {"completion_snapshot_count": 1},
            {
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "total_source": "derived_complete_attempt_sum_all_preserved_sessions",
                    "usage_scope": "explicit_zero_call_task",
                }
            },
        )
        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(OpenHandsTrajectoryAuditError):
                    _trajectory_metrics(_zero_call_trajectory(**case))

    def test_external_dataset_digest_matches_builder_for_all_and_each_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = _fixture_archive(root / "fixture.tar.gz")
            inventory = _write_inventory(root / "inventory.json", archive)
            archive_hash = _sha256(archive)
            reader = OpenHandsArchiveReader()
            trajectories = reader.read(
                archive,
                OpenHandsArchiveMetadata(archive_identity=archive_hash),
            )
            expected_all = build_supervised_dataset(trajectories)

            with (
                patch.object(
                    OpenHandsArchiveReader,
                    "read",
                    side_effect=AssertionError("audit must stream with iter_archive"),
                ),
                patch(
                    "scripts.audit_openhands_trajectory.build_supervised_dataset",
                    wraps=build_supervised_dataset,
                ) as per_trajectory_builder,
            ):
                audit = _build(archive, inventory)

        self.assertEqual(audit["dataset"]["dataset_id"], expected_all.dataset_id)
        self.assertEqual(audit["dataset"]["row_count"], len(expected_all.rows))
        self.assertEqual(per_trajectory_builder.call_count, 4)
        self.assertTrue(
            all(len(call.args[0]) == 1 for call in per_trajectory_builder.call_args_list)
        )
        for run in range(1, 5):
            run_id = f"run_{run}"
            expected_run = build_supervised_dataset(
                trajectory
                for trajectory in trajectories
                if trajectory.run_id == run_id
            )
            self.assertEqual(
                audit["per_run"][run_id]["dataset"]["dataset_id"],
                expected_run.dataset_id,
            )
            self.assertEqual(
                audit["per_run"][run_id]["dataset"]["row_count"],
                len(expected_run.rows),
            )

    def test_mapping_metrics_matrix_and_missing_usage_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = _fixture_archive(root / "fixture.tar.gz")
            inventory = _write_inventory(root / "inventory.json", archive)
            audit = _build(archive, inventory)

        self.assertEqual(
            audit["counts"],
            {
                "task_id_count": 1,
                "run_id_count": 4,
                "trajectory_id_count": 4,
                "condition_id_count": 1,
                "dataset_row_count": 20,
            },
        )
        self.assertEqual(len(audit["task_run_mapping"]), 1)
        mapping = audit["task_run_mapping"][0]
        self.assertEqual(mapping["task_id"], TASK_ID)
        self.assertEqual(
            [item["run_id"] for item in mapping["runs"]],
            ["run_1", "run_2", "run_3", "run_4"],
        )
        self.assertEqual(len({item["trajectory_id"] for item in mapping["runs"]}), 4)
        self.assertEqual(len({item["condition_id"] for item in mapping["runs"]}), 1)

        metrics = audit["metrics"]
        self.assertEqual(metrics["trajectory_count"], 4)
        self.assertEqual(metrics["logical_call_count"], 4)
        self.assertEqual(metrics["attempt_count"], 4)
        self.assertEqual(metrics["api_completed_count"], 4)
        self.assertEqual(metrics["attempt_usage_complete_count"], 3)
        self.assertEqual(metrics["attempt_usage_missing_count"], 1)
        self.assertEqual(metrics["attempt_usage_invalid_count"], 0)
        self.assertEqual(metrics["task_usage_complete_count"], 4)
        self.assertEqual(metrics["task_finished_count"], 2)
        self.assertEqual(metrics["task_aborted_count"], 2)
        self.assertEqual(metrics["task_error_count"], 1)
        self.assertEqual(metrics["task_lifecycle_observed_count"], 3)
        self.assertEqual(metrics["task_lifecycle_censored_count"], 1)
        self.assertEqual(metrics["task_log_observed_count"], 4)
        self.assertEqual(metrics["task_log_missing_count"], 0)
        self.assertEqual(metrics["evaluator_report_observed_count"], 1)
        self.assertEqual(metrics["evaluator_report_missing_count"], 3)
        self.assertEqual(metrics["request_tokens_local_observed_count"], 0)
        self.assertEqual(metrics["request_tokens_local_missing_count"], 4)
        self.assertEqual(metrics["generation_checkpoint_count"], 0)
        self.assertEqual(metrics["provider_error_envelope_count"], 1)

        matrix = audit["label_matrix"]
        self.assertEqual(set(matrix), {item.value for item in PredictionPosition})
        for targets in matrix.values():
            self.assertEqual(set(targets), {item.value for item in PredictionTarget})
            for cell in targets.values():
                self.assertEqual(
                    set(cell["status_counts"]),
                    {item.value for item in LabelStatus},
                )
        task_total = matrix[PredictionPosition.TASK_LAUNCH.value][
            PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS.value
        ]
        self.assertEqual(task_total["status_counts"][LabelStatus.OBSERVED.value], 2)
        self.assertEqual(task_total["status_counts"][LabelStatus.CENSORED.value], 2)
        call_output = matrix[PredictionPosition.CALL_PRE.value][
            PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS.value
        ]
        self.assertEqual(call_output["status_counts"][LabelStatus.OBSERVED.value], 3)
        self.assertEqual(call_output["status_counts"][LabelStatus.MISSING.value], 1)
        self.assertEqual(call_output["reason_counts"], {"missing_usage": 1})
        call_update = matrix[PredictionPosition.CALL_UPDATE.value][
            PredictionTarget.CALL_REMAINING_OUTPUT_TOKENS.value
        ]
        self.assertEqual(call_update["row_count"], 0)
        self.assertEqual(
            call_update["status_counts"],
            {status.value: 0 for status in LabelStatus},
        )

        capability = audit["source_capability"]
        self.assertFalse(capability["request_tokens_local"]["available"])
        self.assertFalse(capability["generation_checkpoint"]["available"])
        self.assertIsNone(capability["retry"]["retry_count"])
        self.assertFalse(capability["retry"]["supported"])
        self.assertEqual(capability["errors"]["provider_error_envelope_count"], 1)

    def test_hashes_and_serialized_output_are_deterministic_and_content_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = _fixture_archive(root / "fixture.tar.gz")
            inventory = _write_inventory(root / "inventory.json", archive)
            first = _build(archive, inventory)
            second = _build(archive, inventory)
            first_output = root / "first.json"
            second_output = root / "second.json"
            atomic_write_json(first_output, first)
            atomic_write_json(second_output, second)
            first_bytes = first_output.read_bytes()
            second_bytes = second_output.read_bytes()

        self.assertEqual(first, second)
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(len(first["canonical_trajectories"]), 4)
        self.assertEqual(
            len({item["canonical_sha256"] for item in first["canonical_trajectories"]}),
            4,
        )
        for item in first["canonical_trajectories"]:
            self.assertRegex(item["canonical_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(first["canonical_source_aggregate_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(first["audit_payload_sha256"], r"^[0-9a-f]{64}$")
        without_audit_hash = copy.deepcopy(first)
        del without_audit_hash["audit_payload_sha256"]
        expected_audit_hash = hashlib.sha256(
            json.dumps(
                without_audit_hash,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(first["audit_payload_sha256"], expected_audit_hash)

        rendered = first_bytes.decode("utf-8")
        for forbidden in (
            str(root),
            "RAW_TASK_ERROR_SENTINEL",
            "RAW_TASK_INSTRUCTION_SENTINEL",
            "RAW_SYSTEM_PROMPT_SENTINEL",
            "RAW_USER_PROMPT_SENTINEL",
            "RAW_ASSISTANT_ANSWER_SENTINEL",
            "occurred_at",
            "raw_ref",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_inventory_identity_mismatch_fails_before_reader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = _fixture_archive(root / "fixture.tar.gz")
            inventory = _write_inventory(root / "inventory.json", archive)
            value = json.loads(inventory.read_text(encoding="utf-8"))
            value["archive_sha256"] = "0" * 64
            inventory.write_text(json.dumps(value), encoding="utf-8")
            with (
                patch.object(
                    OpenHandsArchiveReader,
                    "iter_archive",
                    side_effect=AssertionError("reader must not run after identity mismatch"),
                ),
                self.assertRaises(OpenHandsTrajectoryAuditError),
            ):
                _build(archive, inventory)


if __name__ == "__main__":
    unittest.main()
