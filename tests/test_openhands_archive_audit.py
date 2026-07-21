from __future__ import annotations

import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from scripts.audit_openhands_archive import (
    OpenHandsArchiveAuditError,
    build_inventory,
)


WRAPPER = "fixture-openhands-archive"
RAW_TASK_A = "raw-private-task-id-a"
RAW_TASK_B = "raw-private-task-id-b"
SECRET = "SECRET_DO_NOT_LEAK"
RAW_RESPONSE_ID = "provider-response-raw-id"


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _completion_bytes(tag: str, *, include_secret: bool = False) -> bytes:
    content = f"assistant response {tag}"
    if include_secret:
        content += f" {SECRET}"
    return _json_bytes(
        {
            "args": [],
            "cost": 0.01,
            "kwargs": {"tools": []},
            "messages": [
                {
                    "content": [{"text": f"prompt {tag}", "type": "text"}],
                    "role": "user",
                },
            ],
            "response": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {
                            "content": content,
                            "function_call": None,
                            "role": "assistant",
                            "tool_calls": [],
                        },
                        "provider_specific_fields": {},
                    }
                ],
                "created": 1_700_000_000,
                "id": RAW_RESPONSE_ID if include_secret else f"response-{tag}",
                "model": "gpt-5.2",
                "object": "chat.completion",
                "provider": "openai",
                "system_fingerprint": None,
                "usage": {
                    "completion_tokens": 7,
                    "completion_tokens_details": {
                        "accepted_prediction_tokens": None,
                        "audio_tokens": None,
                        "image_tokens": 0,
                        "reasoning_tokens": 2,
                        "rejected_prediction_tokens": None,
                        "text_tokens": None,
                    },
                    "cost": 0.01,
                    "cost_details": {
                        "upstream_inference_completions_cost": 0.003,
                        "upstream_inference_cost": 0.01,
                        "upstream_inference_prompt_cost": 0.007,
                    },
                    "is_byok": False,
                    "prompt_tokens": 31,
                    "prompt_tokens_details": {
                        "audio_tokens": None,
                        "cached_tokens": 5,
                        "image_tokens": None,
                        "text_tokens": None,
                    },
                    "total_tokens": 38,
                },
            },
            "timestamp": 1_700_000_000.25,
        }
    )


def _report_task(*, resolved: bool) -> dict[str, object]:
    test_bucket = {"failure": [], "success": []}
    return {
        "patch_exists": True,
        "patch_is_None": False,
        "patch_successfully_applied": True,
        "resolved": resolved,
        "tests_status": {
            "FAIL_TO_FAIL": test_bucket,
            "FAIL_TO_PASS": test_bucket,
            "PASS_TO_FAIL": test_bucket,
            "PASS_TO_PASS": test_bucket,
        },
    }


def _run_directory(run_id: int) -> str:
    return f"model-with-an-arbitrary-basename-run_{run_id}"


def _completion_path(run_id: int, task_id: str, filename: str) -> str:
    return (
        f"{WRAPPER}/{_run_directory(run_id)}/llm_completions/"
        f"{task_id}/{filename}.json"
    )


def _report_path(run_id: int, task_id: str) -> str:
    return f"{WRAPPER}/{_run_directory(run_id)}/eval_outputs/{task_id}/report.json"


def _run_file_path(run_id: int, filename: str) -> str:
    return f"{WRAPPER}/{_run_directory(run_id)}/{filename}"


def _jsonl_bytes(records: list[dict[str, object]]) -> bytes:
    return b"\n".join(_json_bytes(record) for record in records) + b"\n"


def _aggregate_report(statuses: dict[str, str]) -> dict[str, object]:
    def ids(status: str) -> list[str]:
        return sorted(task_id for task_id, value in statuses.items() if value == status)

    resolved = ids("resolved")
    unresolved = ids("unresolved")
    empty_patch = ids("empty_patch")
    error = ids("error")
    incomplete = ids("incomplete")
    completed = sorted([*resolved, *unresolved])
    submitted = sorted(statuses)
    return {
        "completed_ids": completed,
        "completed_instances": len(completed),
        "empty_patch_ids": empty_patch,
        "empty_patch_instances": len(empty_patch),
        "error_ids": error,
        "error_instances": len(error),
        "incomplete_ids": incomplete,
        "resolved_ids": resolved,
        "resolved_instances": len(resolved),
        "schema_version": 2,
        "submitted_ids": submitted,
        "submitted_instances": len(submitted),
        "total_instances": len(submitted),
        "unresolved_ids": unresolved,
        "unresolved_instances": len(unresolved),
    }


def _output_record(task_id: str, *, status: str) -> dict[str, object]:
    error = f"task error {SECRET}" if status == "error" else None
    observed = status != "error"
    return {
        "error": error,
        "history": (
            [{"action": "finish", "message": SECRET, "source": "agent"}]
            if observed
            else None
        ),
        "instance": {"instance_id": task_id} if observed else None,
        "instance_id": task_id,
        "instruction": f"private instruction {SECRET}" if observed else None,
        "metadata": {} if observed else None,
        "metrics": (
            {
                "accumulated_token_usage": {
                    "completion_tokens": 2,
                    "prompt_tokens": 3,
                    "response_id": RAW_RESPONSE_ID,
                }
            }
            if observed
            else None
        ),
        "test_result": {},
    }


def _swebench_record(task_id: str, *, status: str) -> dict[str, object]:
    return {
        "instance_id": task_id,
        "model_name_or_path": "gpt-5.2",
        "model_patch": SECRET,
        "report": {
            "empty_generation": status == "empty_patch",
            "error_eval": status == "error",
            "failed_apply_patch": False,
            "resolved": status == "resolved",
            "test_timeout": False,
        },
    }


def _fixture_members(
    *,
    run_ids: tuple[int, ...] = (1, 2, 3, 4),
    report_run_ids: tuple[int, ...] | None = None,
    outcome_overrides: dict[tuple[int, str], str] | None = None,
) -> dict[str, bytes]:
    """Build two tasks with three deliberately different duplicate classes."""
    report_ids = set(run_ids if report_run_ids is None else report_run_ids)
    within_task_run = _completion_bytes("within-one-task-run", include_secret=True)
    same_task_cross_run = _completion_bytes("same-task-cross-run")
    cross_task_same_run = _completion_bytes("cross-task-same-run")
    members: dict[str, bytes] = {}
    overrides = outcome_overrides or {}

    for run_id in run_ids:
        if run_id == 1:
            members[_completion_path(run_id, RAW_TASK_A, "completion-001")] = (
                within_task_run
            )
            members[_completion_path(run_id, RAW_TASK_A, "completion-002")] = (
                within_task_run
            )
        elif run_id in {2, 3}:
            members[_completion_path(run_id, RAW_TASK_A, "completion-001")] = (
                same_task_cross_run
            )
        else:
            members[_completion_path(run_id, RAW_TASK_A, "completion-001")] = (
                cross_task_same_run
            )

        task_b_payload = (
            cross_task_same_run
            if run_id == 4
            else _completion_bytes(f"unique-task-b-run-{run_id}")
        )
        members[_completion_path(run_id, RAW_TASK_B, "completion-001")] = task_b_payload

        statuses = {
            RAW_TASK_A: overrides.get(
                (run_id, RAW_TASK_A),
                "resolved" if run_id % 2 else "unresolved",
            ),
            RAW_TASK_B: overrides.get((run_id, RAW_TASK_B), "unresolved"),
        }
        if run_id in report_ids:
            for task_id, status in statuses.items():
                if status not in {"resolved", "unresolved"}:
                    continue
                members[_report_path(run_id, task_id)] = _json_bytes(
                    {
                        task_id: _report_task(
                            resolved=status == "resolved"
                        )
                    }
                )
        members[_run_file_path(run_id, "report.json")] = _json_bytes(
            _aggregate_report(statuses)
        )
        members[_run_file_path(run_id, "output.jsonl")] = _jsonl_bytes(
            [
                _output_record(task_id, status=status)
                for task_id, status in sorted(statuses.items())
            ]
        )
        members[_run_file_path(run_id, "output.swebench.jsonl")] = _jsonl_bytes(
            [
                _swebench_record(task_id, status=status)
                for task_id, status in sorted(statuses.items())
            ]
        )
        members[_run_file_path(run_id, "output.jsonl.bak")] = b"not-json " + SECRET.encode()
    return members


def _write_tar_gz(
    path: Path,
    members: dict[str, bytes],
    *,
    extra_members: tuple[tuple[tarfile.TarInfo, bytes | None], ...] = (),
) -> None:
    with tarfile.open(path, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.mtime = 0
            info.mode = 0o644
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        for info, payload in extra_members:
            archive.addfile(info, None if payload is None else io.BytesIO(payload))


def _build(path: Path, **overrides: Any) -> dict[str, Any]:
    options: dict[str, Any] = {
        "expected_bytes": None,
        "expected_sha256": None,
        "expected_wrapper": WRAPPER,
        "expected_task_count": 2,
        "schema_sample_count": 5,
        "max_schema_json_bytes": 1024 * 1024,
    }
    options.update(overrides)
    return build_inventory(path, **options)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_record(inventory: dict[str, Any], run_id: int) -> dict[str, Any]:
    runs = inventory["runs"]
    if isinstance(runs, dict):
        value = runs.get(str(run_id), runs.get(run_id))
        if not isinstance(value, dict):
            raise AssertionError(f"missing run {run_id} in {runs!r}")
        return value
    if isinstance(runs, list):
        for value in runs:
            if isinstance(value, dict) and value.get("run_id") == run_id:
                return value
    raise AssertionError(f"unsupported runs inventory: {runs!r}")


class OpenHandsArchiveAuditTests(unittest.TestCase):
    def test_streams_without_listing_or_extracting_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.tar.gz"
            _write_tar_gz(path, _fixture_members())
            real_open = tarfile.open

            with (
                patch.object(
                    tarfile.TarFile,
                    "getmembers",
                    side_effect=AssertionError("getmembers materializes the full archive"),
                ),
                patch.object(
                    tarfile.TarFile,
                    "extract",
                    side_effect=AssertionError("archive members must not be extracted"),
                ),
                patch.object(
                    tarfile.TarFile,
                    "extractall",
                    side_effect=AssertionError("archive members must not be extracted"),
                ),
                patch.object(tarfile, "open", wraps=real_open) as mocked_open,
            ):
                inventory = _build(path)

        self.assertEqual(inventory["run_count"], 4)
        modes = [
            call.kwargs.get("mode", call.args[1] if len(call.args) > 1 else None)
            for call in mocked_open.call_args_list
        ]
        self.assertIn("r|gz", modes)

    def test_counts_sizes_four_run_coverage_and_duplicate_classes(self) -> None:
        members = _fixture_members()
        completion_members = {
            name: payload for name, payload in members.items() if "/llm_completions/" in name
        }
        report_members = {
            name: payload for name, payload in members.items() if name.endswith("/report.json")
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.tar.gz"
            _write_tar_gz(path, members)
            inventory = _build(path)

            self.assertEqual(inventory["archive_bytes"], path.stat().st_size)
            self.assertEqual(inventory["archive_sha256"], _sha256(path))

        self.assertEqual(inventory["member_count"], len(members))
        self.assertEqual(inventory["regular_file_count"], len(members))
        self.assertEqual(inventory["task_count"], 2)
        self.assertEqual(inventory["trajectory_count"], 8)
        self.assertEqual(inventory["task_run_count"], 8)
        self.assertEqual(inventory["run_count"], 4)
        self.assertTrue(inventory["exactly_four_runs"])
        self.assertEqual(inventory["llm_completions_count"], len(completion_members))
        self.assertEqual(inventory["completions_count"], len(completion_members))
        self.assertEqual(
            inventory["llm_completions_bytes"], sum(map(len, completion_members.values()))
        )
        self.assertEqual(inventory["report_count"], len(report_members))
        self.assertEqual(inventory["report_bytes"], sum(map(len, report_members.values())))
        self.assertEqual(inventory["task_report_count"], 8)
        self.assertEqual(inventory["aggregate_report_count"], 4)
        self.assertEqual(inventory["output_jsonl_record_count"], 8)
        self.assertEqual(inventory["output_swebench_jsonl_record_count"], 8)
        self.assertEqual(inventory["jsonl_audit"]["backup_files"]["file_count"], 4)
        self.assertFalse(inventory["jsonl_audit"]["backup_files"]["consumed"])
        self.assertTrue(
            inventory["jsonl_audit"]["output_jsonl"]["task_sets"]
            ["aggregate_submitted"]["matches"]
        )
        self.assertTrue(inventory["readiness"]["formal_jsonl_full_schema_validated"])
        self.assertFalse(inventory["readiness"]["termination_labels_ready"])
        self.assertFalse(inventory["readiness"]["overall_ready"])
        self.assertEqual(
            inventory["label_status_counts"]["evaluator_accuracy"],
            {"observed": 8, "missing": 0, "censored": 0, "invalid": 0},
        )

        files = inventory["size_distributions"]["files"]
        self.assertEqual(files["count"], len(members))
        self.assertEqual(files["sum"], sum(map(len, members.values())))
        self.assertEqual({int(key) for key in inventory["task_run_coverage"]}, {4})
        self.assertEqual(inventory["task_run_coverage"]["4"], 2)

        duplicates = inventory["duplicate_snapshots"]
        self.assertEqual(duplicates["duplicate_hash_count"], 3)
        self.assertEqual(duplicates["duplicate_file_count"], 6)
        self.assertEqual(duplicates["duplicate_extra_copies"], 3)
        groups = duplicates["groups"]
        by_cardinality = {
            (group["run_count"], group["task_count"]): group for group in groups
        }
        self.assertEqual(set(by_cardinality), {(1, 1), (2, 1), (1, 2)})
        self.assertFalse(by_cardinality[(1, 1)]["cross_run"])
        self.assertFalse(by_cardinality[(1, 1)]["cross_task"])
        self.assertTrue(by_cardinality[(2, 1)]["cross_run"])
        self.assertFalse(by_cardinality[(2, 1)]["cross_task"])
        self.assertFalse(by_cardinality[(1, 2)]["cross_run"])
        self.assertTrue(by_cardinality[(1, 2)]["cross_task"])

    def test_unsafe_member_paths_fail_closed(self) -> None:
        for unsafe_name in (
            "../escape.json",
            "/absolute/path.json",
            f"{WRAPPER}/{_run_directory(1)}/../../escape.json",
            f"{WRAPPER}\\{_run_directory(1)}\\report.json",
        ):
            with self.subTest(unsafe_name=unsafe_name), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "unsafe.tar.gz"
                info = tarfile.TarInfo(unsafe_name)
                payload = b"{}"
                info.size = len(payload)
                _write_tar_gz(
                    path,
                    _fixture_members(),
                    extra_members=((info, payload),),
                )
                with self.assertRaisesRegex(
                    OpenHandsArchiveAuditError, "(?i)(unsafe|canonical|member|path)"
                ):
                    _build(path)

    def test_missing_run_fails_and_missing_report_is_not_imputed_as_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing_run = root / "missing-run.tar.gz"
            _write_tar_gz(missing_run, _fixture_members(run_ids=(1, 2, 3)))
            with self.assertRaisesRegex(
                OpenHandsArchiveAuditError, "(?i)(four|run|coverage)"
            ):
                _build(missing_run)

            missing_report = root / "missing-report.tar.gz"
            _write_tar_gz(
                missing_report,
                _fixture_members(report_run_ids=(1, 2, 3)),
            )
            inventory = _build(missing_report)

        self.assertEqual(inventory["report_count"], 10)
        self.assertEqual(inventory["task_report_count"], 6)
        self.assertEqual(inventory["aggregate_report_count"], 4)
        self.assertFalse(inventory["readiness"]["reports_complete"])
        run_four = _run_record(inventory, 4)
        self.assertFalse(run_four["report_present"])
        self.assertEqual(run_four["resolved_count"], 0)
        self.assertEqual(run_four["missing_task_report_task_run_count"], 2)
        self.assertEqual(
            inventory["label_status_counts"]["evaluator_accuracy"],
            {"observed": 6, "missing": 2, "censored": 0, "invalid": 0},
        )
        self.assertTrue(
            any("report" in str(anomaly).lower() for anomaly in inventory["anomalies"])
        )

    def test_inventory_is_deterministic_and_does_not_expose_raw_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.tar.gz"
            _write_tar_gz(path, _fixture_members())
            first = _build(path)
            repeated = _build(path)

        self.assertEqual(first, repeated)
        rendered = json.dumps(first, ensure_ascii=False, sort_keys=True)
        self.assertNotIn(SECRET, rendered)
        self.assertNotIn(RAW_RESPONSE_ID, rendered)
        self.assertNotIn(RAW_TASK_A, rendered)
        self.assertNotIn(RAW_TASK_B, rendered)
        self.assertEqual(first["source_hashes"]["archive_sha256"], first["archive_sha256"])

    def test_unknown_completion_schema_fails_closed(self) -> None:
        members = _fixture_members()
        first_completion = next(
            name for name in members if "/llm_completions/" in name
        )
        members[first_completion] = _json_bytes({"unknown_future_schema": True})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unknown-schema.tar.gz"
            _write_tar_gz(path, members)
            with self.assertRaisesRegex(
                OpenHandsArchiveAuditError, "(?i)(schema|completion|response)"
            ):
                _build(path)

    def test_oversize_completion_fails_before_loading_json(self) -> None:
        members = _fixture_members()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "oversize.tar.gz"
            _write_tar_gz(path, members)
            with self.assertRaisesRegex(
                OpenHandsArchiveAuditError, "(?i)(large|limit|size|bytes)"
            ):
                _build(path, max_schema_json_bytes=32)

    def test_jsonl_telemetry_and_evaluator_four_states_are_not_imputed(self) -> None:
        members = _fixture_members(
            outcome_overrides={
                (1, RAW_TASK_A): "empty_patch",
                (1, RAW_TASK_B): "error",
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "four-states.tar.gz"
            _write_tar_gz(path, members)
            inventory = _build(path)

        self.assertEqual(
            inventory["label_status_counts"]["evaluator_accuracy"],
            {"observed": 6, "missing": 1, "censored": 0, "invalid": 1},
        )
        self.assertEqual(
            inventory["label_status_counts"]["task_termination"],
            {"observed": 0, "missing": 8, "censored": 0, "invalid": 0},
        )
        telemetry = inventory["telemetry_status_counts"]
        self.assertEqual(telemetry["task_error_nonempty"], 1)
        self.assertEqual(telemetry["history"]["censored"], 1)
        self.assertEqual(telemetry["metrics"]["censored"], 1)
        self.assertEqual(telemetry["usage"]["censored"], 1)
        self.assertFalse(inventory["readiness"]["accuracy_full_coverage_ready"])
        self.assertTrue(inventory["readiness"]["accuracy_observed_subset_ready"])
        self.assertEqual(inventory["aggregate_outcome_counts"]["empty_patch"], 1)
        self.assertEqual(inventory["aggregate_outcome_counts"]["error"], 1)

    def test_only_exact_task_report_path_is_counted(self) -> None:
        members = _fixture_members()
        nested = (
            f"{WRAPPER}/{_run_directory(1)}/eval_outputs/{RAW_TASK_A}/"
            "nested/report.json"
        )
        members[nested] = _json_bytes({RAW_TASK_A: _report_task(resolved=True)})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested-report.tar.gz"
            _write_tar_gz(path, members)
            inventory = _build(path)

        self.assertEqual(inventory["task_report_count"], 8)
        self.assertEqual(
            inventory["path_structure"]["categories"]["eval_artifact"]["file_count"],
            1,
        )

    def test_jsonl_malformed_unknown_and_oversize_fail_closed(self) -> None:
        cases = (
            (b'{"broken"\n', {}, "(?i)(json|record)"),
            (
                _jsonl_bytes([{"unknown_future_schema": True}]),
                {},
                "(?i)(schema|output)",
            ),
            (None, {"max_jsonl_line_bytes": 32}, "(?i)(line|limit|record)"),
        )
        for payload, overrides, pattern in cases:
            with self.subTest(pattern=pattern), tempfile.TemporaryDirectory() as temporary:
                members = _fixture_members()
                if payload is not None:
                    members[_run_file_path(1, "output.jsonl")] = payload
                path = Path(temporary) / "bad-jsonl.tar.gz"
                _write_tar_gz(path, members)
                with self.assertRaisesRegex(OpenHandsArchiveAuditError, pattern):
                    _build(path, **overrides)


if __name__ == "__main__":
    unittest.main()
