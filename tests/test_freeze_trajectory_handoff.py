from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.freeze_trajectory_handoff import (
    BAGEN_FAMILY_AUDITS,
    BAGEN_REPO,
    BAGEN_REVISION,
    BUILDER_CELLS,
    POSITIONS,
    REPO_ROOT,
    SPEND_READER_SOURCE_ID,
    TARGETS,
    TrajectoryHandoffError,
    _assert_no_absolute_paths,
    _contains_absolute_local_path,
    atomic_write_json,
    build_handoff,
)


FIXTURE_SPEND_REPO = "fixture/openhands"
FIXTURE_SPEND_REVISION = "2" * 40
FIXTURE_XET_ETAG = "3" * 64


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _semantic_sha(value: object) -> str:
    return _sha_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _git_blob_sha(path: Path) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {path.stat().st_size}\0".encode())
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _bagen_family_sha(hashes: dict[str, str]) -> str:
    digest = hashlib.sha256(b"bagen-swebench-canonical-family-v1\0")
    for relative_path in sorted(hashes):
        for value in (relative_path.encode("utf-8"), bytes.fromhex(hashes[relative_path])):
            digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
            digest.update(value)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _rehash_payload_file(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    value.pop("audit_payload_sha256", None)
    value["audit_payload_sha256"] = _semantic_sha(value)
    _write_json(path, value)
    return value


def _status_counts(
    *, observed: int = 0, missing: int = 0, censored: int = 0, invalid: int = 0
) -> dict[str, int]:
    return {
        "observed": observed,
        "missing": missing,
        "censored": censored,
        "invalid": invalid,
    }


def _bagen_matrix() -> list[dict[str, object]]:
    output = []
    for position in POSITIONS:
        for target in TARGETS:
            observed = int(
                position == "call_pre" and target == "call_billable_output_tokens"
            )
            output.append(
                {
                    "position": position,
                    "target": target,
                    "row_count": observed,
                    "status_counts": _status_counts(observed=observed),
                }
            )
    return output


def _spend_matrix(
    selected_status: str | None = None,
) -> dict[str, dict[str, dict[str, object]]]:
    output: dict[str, dict[str, dict[str, object]]] = {}
    for position in POSITIONS:
        output[position] = {}
        for target in TARGETS:
            selected = position == "call_pre" and target == "call_billable_output_tokens"
            if selected and selected_status is None:
                counts = _status_counts(observed=1, missing=1, censored=1, invalid=1)
            elif selected and selected_status is not None:
                counts = _status_counts(**{selected_status: 1})
            else:
                counts = _status_counts()
            output[position][target] = {
                "structurally_emitted_by_builder": (position, target) in BUILDER_CELLS,
                "row_count": sum(counts.values()),
                "eligible_row_count": counts["observed"],
                "eligible_for_supervised_training": counts["observed"] > 0,
                "status_counts": counts,
                "reason_counts": (
                    {
                        key: value
                        for key, value in (
                            ("missing_usage", counts["missing"]),
                            ("logging_incomplete", counts["censored"]),
                            ("usage_mismatch", counts["invalid"]),
                        )
                        if value
                    }
                ),
            }
    return output


class _Fixture:
    def __init__(self, root: Path) -> None:
        self.workspace = root / "workspace"
        self.bagen_root = self.workspace / "external" / "bagen"
        self.spend_root = self.workspace / "external" / "spend_your_money"
        self.manifest_summary, self.bagen_combined = self._write_bagen()
        self.spend_inventory, self.spend_audit, self.archive = self._write_spend()

    def _write_bagen(self) -> tuple[Path, Path]:
        self.bagen_root.mkdir(parents=True, exist_ok=True)
        audit_dir = self.bagen_root / "audits"
        manifest_entries: list[dict[str, object]] = []
        combined_families: list[dict[str, object]] = []
        canonical_trajectory_index: list[dict[str, str]] = []
        combined_task_trajectories: list[dict[str, str]] = []
        for family_index, (family, filename) in enumerate(sorted(BAGEN_FAMILY_AUDITS.items())):
            family_root = f"swebench-origin-{family}"
            relative_path = "model/task-1/task-1.traj.json"
            raw_path = self.bagen_root / "origin" / family_root / relative_path
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(f"fixture-{family}".encode())
            trajectory_id = f"trajectory-{family}"
            condition_id = f"condition-{family_index}"
            raw_record = {
                "path": relative_path,
                "bytes": raw_path.stat().st_size,
                "sha256": _sha_file(raw_path),
                "canonical_content_sha256": _sha_bytes(f"canonical-{family}".encode()),
                "canonical_rerun_consistent": True,
                "task_id": "task-1",
                "trajectory_id": trajectory_id,
                "condition_id": condition_id,
            }
            hub_path = f"origin/{family_root}/{relative_path}"
            manifest_entries.append({"path": hub_path, "size_bytes": raw_path.stat().st_size})
            canonical_trajectory_index.append(
                {
                    "family": family,
                    "path": hub_path,
                    "canonical_content_sha256": raw_record["canonical_content_sha256"],
                }
            )
            combined_task_trajectories.append(
                {
                    "family": family,
                    "run_id": f"{family_root}/{relative_path}",
                    "trajectory_id": trajectory_id,
                    "condition_id": condition_id,
                }
            )
            canonical = _bagen_family_sha(
                {relative_path: str(raw_record["canonical_content_sha256"])}
            )
            audit = {
                "audit_schema_version": 1,
                "source_id": "bagen_swebench_traj_v1",
                "reader_version": "bagen_swebench_traj_v1",
                "family_root": family_root,
                "family": family,
                "raw_file_count": 1,
                "raw_bytes": raw_path.stat().st_size,
                "raw_files": [raw_record],
                "source_hashes": {relative_path: raw_record["sha256"]},
                "task_count": 1,
                "trajectory_count": 1,
                "condition_count": 1,
                "call_count": 1,
                "attempt_count": 1,
                "complete_usage_attempts": 1,
                "missing_usage_attempts": 0,
                "retry_count": 0,
                "within_call_retry_count": 0,
                "tool_event_count": 1,
                "tool_failure_count": 0,
                "distributions": {
                    "task_terminal_event": {"task_finished": 1},
                    "exit_status": {"Submitted": 1},
                },
                "dataset": {
                    "dataset_id": _sha_bytes(f"dataset-{family}".encode()),
                    "row_count": 1,
                    "status_counts": _status_counts(observed=1),
                    "by_position_target": _bagen_matrix(),
                },
                "canonical_content_sha256": canonical,
                "canonical_rerun_content_sha256": canonical,
                "canonical_rerun_consistent": True,
                "raw_response": "RAW_RESPONSE_SENTINEL_MUST_NOT_LEAK",
            }
            audit_path = _write_json(audit_dir / filename, audit)
            combined_families.append(
                {
                    "family": family,
                    "family_root": family_root,
                    "local_relative_root": f"workspace/external/bagen/origin/{family_root}",
                    "audit_path": f"workspace/external/bagen/audits/{filename}",
                    "audit_bytes": audit_path.stat().st_size,
                    "audit_sha256": _sha_file(audit_path),
                    "raw_file_count": 1,
                    "raw_bytes": raw_path.stat().st_size,
                    "task_count": 1,
                    "run_count": 1,
                    "trajectory_count": 1,
                    "condition_count": 1,
                    "dataset": {
                        "dataset_id": audit["dataset"]["dataset_id"],
                        "row_count": 1,
                        "schema_version": 1,
                        "feature_schema_version": 2,
                    },
                    "canonical_content_sha256": canonical,
                }
            )

        gpt_root = (
            self.bagen_root / "origin" / "swebench-origin-gpt5.2instant"
        )
        for index in range(5):
            aux = gpt_root / f"aux-{index}.json"
            aux.write_bytes(f"aux-{index}".encode())
            manifest_entries.append(
                {
                    "path": (
                        "origin/swebench-origin-gpt5.2instant/"
                        f"aux-{index}.json"
                    ),
                    "size_bytes": aux.stat().st_size,
                }
            )

        manifest = self.bagen_root / "manifest.jsonl"
        manifest.write_text(
            "".join(
                json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                for item in sorted(manifest_entries, key=lambda item: str(item["path"]))
            ),
            encoding="utf-8",
        )
        trajectory_entries = [
            item for item in manifest_entries if str(item["path"]).endswith(".traj.json")
        ]
        summary = {
            "manifest_bytes": manifest.stat().st_size,
            "manifest_etag": _git_blob_sha(manifest),
            "manifest_file": manifest.name,
            "manifest_sha256": _sha_file(manifest),
            "resolved_revision": BAGEN_REVISION,
            "source_url": f"https://huggingface.co/datasets/{BAGEN_REPO}",
            "file_count": len(manifest_entries),
            "total_bytes": sum(int(item["size_bytes"]) for item in manifest_entries),
            "traj_json_count": len(trajectory_entries),
            "traj_json_bytes": sum(int(item["size_bytes"]) for item in trajectory_entries),
        }
        summary_path = _write_json(self.bagen_root / "manifest_summary.json", summary)

        source_paths = {
            "reader": "src/token_prediction/collection/bagen_swebench.py",
            "builder": "src/token_prediction/dataset/builder.py",
            "labels": "src/token_prediction/dataset/labels.py",
            "audit": "scripts/audit_bagen_combined.py",
            "family_audit": "scripts/audit_bagen_swebench.py",
        }
        source_files = {
            role: {
                "path": path,
                "bytes": (REPO_ROOT / path).stat().st_size,
                "sha256": _sha_file(REPO_ROOT / path),
            }
            for role, path in source_paths.items()
        }
        canonical_family_index = sorted(
            [
                {
                    "family": str(item["family"]),
                    "canonical_content_sha256": str(item["canonical_content_sha256"]),
                }
                for item in combined_families
            ],
            key=lambda item: item["family"],
        )
        canonical_trajectory_index.sort(key=lambda item: (item["family"], item["path"]))
        family_audit_index = {
            str(item["family"]): {
                "path": item["audit_path"],
                "bytes": item["audit_bytes"],
                "sha256": item["audit_sha256"],
            }
            for item in combined_families
        }
        family_dataset_id_index = {
            str(item["family"]): str(item["dataset"]["dataset_id"])
            for item in combined_families
        }
        combined_dataset_id = _sha_bytes(b"combined-bagen-dataset")
        combined = {
            "combined_audit_schema_version": 1,
            "source_id": "bagen_swebench_combined_audit_v1",
            "hub": {"repo": BAGEN_REPO, "resolved_revision": BAGEN_REVISION},
            "manifest": {
                "summary": {
                    "path": "workspace/external/bagen/manifest_summary.json",
                    "bytes": summary_path.stat().st_size,
                    "sha256": _sha_file(summary_path),
                },
                "raw": {
                    "path": "workspace/external/bagen/manifest.jsonl",
                    "bytes": manifest.stat().st_size,
                    "sha256": _sha_file(manifest),
                    "git_blob_etag": _git_blob_sha(manifest),
                    "file_count": len(manifest_entries),
                    "total_bytes": sum(
                        int(item["size_bytes"]) for item in manifest_entries
                    ),
                    "traj_json_count": len(trajectory_entries),
                    "traj_json_bytes": sum(
                        int(item["size_bytes"]) for item in trajectory_entries
                    ),
                },
            },
            "families": combined_families,
            "family_audit_index": family_audit_index,
            "family_dataset_id_index": family_dataset_id_index,
            "counts": {
                "task_id_count": 1,
                "run_id_count": 5,
                "trajectory_id_count": 5,
                "condition_id_count": 5,
                "dataset_row_count": 5,
                "raw_file_count": 5,
                "raw_bytes": sum(int(item["raw_bytes"]) for item in combined_families),
            },
            "combined_dataset": {
                "dataset_id": combined_dataset_id,
                "row_count": 5,
                "schema_version": 1,
                "feature_schema_version": 2,
            },
            "task_cross_family_distribution": {"5": 1},
            "task_family_mapping": [
                {
                    "task_id": "task-1",
                    "family_count": 5,
                    "families": sorted(BAGEN_FAMILY_AUDITS),
                    "trajectories": sorted(
                        combined_task_trajectories,
                        key=lambda item: (item["family"], item["run_id"]),
                    ),
                }
            ],
            "condition_trajectory_counts": {
                f"condition-{index}": 1 for index in range(5)
            },
            "canonical_family_index": canonical_family_index,
            "canonical_family_index_sha256": _semantic_sha(canonical_family_index),
            "canonical_trajectory_index_sha256": _semantic_sha(canonical_trajectory_index),
            "source_files": source_files,
            "source_hashes": {
                item["path"]: item["sha256"] for item in source_files.values()
            },
            "construction": {
                "command": "$env:PYTHONPATH='src'; python scripts/audit_bagen_combined.py",
                "reader": "BagenSwebenchReader",
                "dataset_builder": "build_supervised_dataset",
                "family_order": sorted(BAGEN_FAMILY_AUDITS),
                "output": "workspace/external/bagen/combined_swebench_audit.json",
            },
        }
        combined["audit_payload_sha256"] = _semantic_sha(combined)
        combined_path = _write_json(
            self.bagen_root / "combined_swebench_audit.json", combined
        )
        return summary_path, combined_path

    def _write_spend(self) -> tuple[Path, Path, Path]:
        self.spend_root.mkdir(parents=True, exist_ok=True)
        archive = self.spend_root / "gpt_5.2_4runs.tar.gz"
        archive.write_bytes(b"fixture archive bytes")
        runs = []
        for run_id in range(1, 5):
            run_basename = f"fixture-run_{run_id}"
            output_path = f"gpt_5.2_4runs/{run_basename}/output.jsonl"
            swebench_path = f"gpt_5.2_4runs/{run_basename}/output.swebench.jsonl"
            runs.append(
                {
                    "run_id": run_id,
                    "run_basename": run_basename,
                    "task_count": 1,
                    "task_run_count": 1,
                    "llm_completions_count": 1,
                    "task_report_count": 1,
                    "aggregate_report_count": 1,
                    "report_count": 2,
                    "task_report_bytes": 2,
                    "aggregate_report_bytes": 3,
                    "report_bytes": 5,
                    "output_jsonl": {
                        "filename": "output.jsonl",
                        "files": [
                            {
                                "bytes": 10,
                                "sha256": _sha_bytes(f"output-{run_id}".encode()),
                                "record_count": 1,
                                "member_path_sha256": _sha_bytes(output_path.encode()),
                            }
                        ],
                    },
                    "output_swebench_jsonl": {
                        "filename": "output.swebench.jsonl",
                        "files": [
                            {
                                "bytes": 11,
                                "sha256": _sha_bytes(f"swebench-{run_id}".encode()),
                                "record_count": 1,
                                "member_path_sha256": _sha_bytes(swebench_path.encode()),
                            }
                        ],
                    },
                }
            )
        inventory = {
            "inventory_schema_version": 2,
            "source_id": "spend_your_money/openhands_trajectories:gpt_5.2_4runs",
            "archive_path": "workspace/external/spend_your_money/gpt_5.2_4runs.tar.gz",
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": _sha_file(archive),
            "hub_repo": FIXTURE_SPEND_REPO,
            "resolved_revision": FIXTURE_SPEND_REVISION,
            "task_count": 1,
            "trajectory_count": 4,
            "run_count": 4,
            "task_report_count": 4,
            "aggregate_report_count": 4,
            "report_count": 8,
            "task_report_bytes": 8,
            "aggregate_report_bytes": 12,
            "report_bytes": 20,
            "runs": runs,
            "telemetry_status_counts": {
                "history": _status_counts(observed=4),
                "metrics": _status_counts(observed=4),
                "usage": _status_counts(observed=3, censored=1),
                "task_error_nonempty": 1,
            },
            "label_status_counts": {
                "evaluator_accuracy": _status_counts(observed=4),
                "task_termination": _status_counts(missing=4),
            },
        }
        inventory_path = _write_json(self.spend_root / "gpt_5.2_inventory.json", inventory)
        task_runs = []
        canonical = []
        per_run = {}
        for run_id in range(1, 5):
            run_key = f"run_{run_id}"
            trajectory_id = f"spend-trajectory-{run_id}"
            condition_id = f"spend-condition-{run_id}"
            task_runs.append(
                {
                    "run_id": run_key,
                    "trajectory_id": trajectory_id,
                    "condition_id": condition_id,
                }
            )
            canonical_record = {
                "task_id": "task-1",
                "run_id": run_key,
                "trajectory_id": trajectory_id,
                "condition_id": condition_id,
                "canonical_sha256": _sha_bytes(f"canonical-{run_id}".encode()),
            }
            canonical.append(canonical_record)
            selected_status = ("observed", "missing", "censored", "invalid")[run_id - 1]
            per_run[run_key] = {
                "task_count": 1,
                "trajectory_count": 1,
                "condition_counts": {condition_id: 1},
                "canonical_aggregate_sha256": _semantic_sha([canonical_record]),
                "dataset": {
                    "dataset_id": _sha_bytes(f"run-dataset-{run_id}".encode()),
                    "row_count": 1,
                },
                "metrics": {
                    "evaluator_report_observed_count": 1,
                    "evaluator_report_missing_count": 0,
                    "logical_call_count": 1,
                },
                "label_matrix": _spend_matrix(selected_status),
            }
        source_capability = {
            "source_id": SPEND_READER_SOURCE_ID,
            "declared_observables": [
                "attempt_usage",
                "call_usage",
                "request_messages",
                "task_usage",
                "tool_events",
            ],
            "request_tokens_local": {
                "available": False,
                "observed_count": 0,
                "missing_count": 4,
                "reason": "no_local_tokenizer_count_in_archive",
                "gates_targets": [
                    "task_unknown_remaining_tokens",
                    "call_unknown_billable_tokens",
                ],
            },
            "attempt_usage": {
                "available": True,
                "complete_count": 3,
                "missing_count": 1,
                "invalid_count": 0,
                "scope": "current_response_only",
            },
            "task_usage": {
                "available": True,
                "complete_count": 3,
                "missing_count": 1,
                "invalid_count": 0,
                "scope": (
                    "output.metrics current-session aggregate plus complete preserved "
                    "completion extras, plus source-reported explicit zero-call usage; "
                    "never backfilled into attempt events"
                ),
                "explicit_zero_call_count": 0,
                "explicit_zero_call_source": "output.metrics.accumulated_token_usage",
                "explicit_zero_call_never_imputed": True,
                "explicit_zero_call_criteria": (
                    "usage_scope=explicit_zero_call_task, accounted and reported total "
                    "tokens both zero, completion_snapshot_count=0"
                ),
                "all_preserved_sessions_count": 3,
                "current_session_without_completion_boundaries_count": 0,
                "missing_incomplete_extra_session_count": 1,
                "missing_no_completion_or_task_metrics_count": 0,
            },
            "retry": {
                "supported": False,
                "retry_count": None,
                "reason": "provider_transport_retry_ledger_not_preserved",
            },
            "tool_events": {
                "available": True,
                "started_count": 4,
                "completed_count": 4,
                "failed_count": 0,
                "failure_observable_count": 4,
                "failure_unobservable_count": 0,
                "failure_status_scope": "explicit_output_jsonl_only",
            },
            "errors": {
                "task_error_available": True,
                "task_error_count": 1,
                "attempt_error_available": False,
                "attempt_error_count": 0,
                "provider_error_envelope_available": False,
                "provider_error_envelope_count": 0,
                "provider_error_envelope_semantics": (
                    "preserved on a completed response; not classified as API_FAILED "
                    "or a transport retry"
                ),
                "reason": (
                    "task_errors_attempt_failures_and_provider_envelopes_are_distinct"
                ),
            },
            "task_termination": {
                "available": False,
                "finished_count": 0,
                "aborted_count": 0,
                "observed_lifecycle_count": 0,
                "censored_lifecycle_count": 4,
                "task_log_observed_count": 0,
                "task_log_missing_count": 4,
                "source": "output.jsonl_when_present_else_censored_logging_incomplete",
            },
            "generation_checkpoint": {
                "available": False,
                "observed_count": 0,
                "reason": "no_streaming_generation_deltas_or_checkpoints_in_archive",
                "gates_position": "call_update",
            },
            "session_reconciliation": {
                "completion_logging_complete_count": 4,
                "completion_logging_incomplete_count": 0,
                "task_usage_reconciled_count": 3,
                "task_usage_unreconciled_count": 1,
                "metrics_completion_extra_task_count": 0,
                "metrics_completion_extra_count": 0,
                "metrics_missing_completion_task_count": 0,
                "metrics_missing_completion_count": 0,
                "history_llm_metrics_ledger_match_count": 3,
                "history_llm_metrics_ledger_mismatch_count": 1,
                "message_prefix_reset_count": 0,
                "repeated_request_snapshot_count": 0,
                "response_not_materialized_in_next_request_count": 0,
                "reasoning_subset_anomaly_count": 0,
            },
        }
        metrics = {
            "trajectory_count": 4,
            "attempt_count": 4,
            "request_count": 4,
            "request_tokens_local_observed_count": 0,
            "request_tokens_local_missing_count": 4,
            "attempt_usage_complete_count": 3,
            "attempt_usage_missing_count": 1,
            "attempt_usage_invalid_count": 0,
            "task_usage_complete_count": 3,
            "task_usage_missing_count": 1,
            "task_usage_invalid_count": 0,
            "task_usage_explicit_zero_call_count": 0,
            "task_usage_all_preserved_sessions_count": 3,
            "task_usage_current_session_only_count": 0,
            "task_usage_missing_extra_session_count": 1,
            "task_usage_missing_no_evidence_count": 0,
            "tool_started_count": 4,
            "tool_completed_count": 4,
            "tool_failed_count": 0,
            "tool_terminal_failure_observable_count": 4,
            "tool_terminal_failure_unobservable_count": 0,
            "task_error_count": 1,
            "api_failed_count": 0,
            "provider_error_envelope_count": 0,
            "task_finished_count": 0,
            "task_aborted_count": 0,
            "task_lifecycle_observed_count": 0,
            "task_lifecycle_censored_count": 4,
            "task_log_observed_count": 0,
            "task_log_missing_count": 4,
            "generation_checkpoint_count": 0,
            "completion_logging_complete_count": 4,
            "completion_logging_incomplete_count": 0,
            "task_usage_reconciled_count": 3,
            "task_usage_unreconciled_count": 1,
            "metrics_completion_extra_task_count": 0,
            "metrics_completion_extra_count": 0,
            "metrics_missing_completion_task_count": 0,
            "metrics_missing_completion_count": 0,
            "history_llm_metrics_ledger_match_count": 3,
            "history_llm_metrics_ledger_mismatch_count": 1,
            "message_prefix_reset_count": 0,
            "repeated_request_snapshot_count": 0,
            "response_not_materialized_in_next_request_count": 0,
            "reasoning_subset_anomaly_count": 0,
            "evaluator_report_observed_count": 4,
            "evaluator_report_missing_count": 0,
        }
        audit = {
            "trajectory_audit_schema_version": 1,
            "archive": {
                "local_relative_path": inventory["archive_path"],
                "bytes": archive.stat().st_size,
                "sha256": _sha_file(archive),
                "hub_repo": FIXTURE_SPEND_REPO,
                "resolved_revision": FIXTURE_SPEND_REVISION,
            },
            "inventory": {
                "local_relative_path": (
                    "workspace/external/spend_your_money/gpt_5.2_inventory.json"
                ),
                "sha256": _sha_file(inventory_path),
                "inventory_schema_version": 2,
                "archive_identity_match": True,
            },
            "counts": {
                "task_id_count": 1,
                "run_id_count": 4,
                "trajectory_id_count": 4,
                "condition_id_count": 4,
                "dataset_row_count": 4,
            },
            "run_ids": ["run_1", "run_2", "run_3", "run_4"],
            "condition_counts": {f"spend-condition-{index}": 1 for index in range(1, 5)},
            "task_run_mapping": [{"task_id": "task-1", "runs": task_runs}],
            "canonical_trajectories": canonical,
            "canonical_source_aggregate_sha256": _semantic_sha(
                sorted(canonical, key=lambda item: item["trajectory_id"])
            ),
            "dataset": {
                "dataset_id": _sha_bytes(b"spend-dataset"),
                "row_count": 4,
                "schema_version": 1,
                "feature_schema_version": 2,
                "construction": "fixture external-sort construction",
            },
            "label_matrix": _spend_matrix(),
            "source_capability": source_capability,
            "per_run": per_run,
            "metrics": metrics,
            "raw_messages": "RAW_MESSAGE_SENTINEL_MUST_NOT_LEAK",
        }
        audit["audit_payload_sha256"] = _semantic_sha(audit)
        audit_path = _write_json(
            self.spend_root / "gpt_5.2_trajectory_audit.json", audit
        )
        return inventory_path, audit_path, archive

    @property
    def bagen_audits(self) -> dict[str, Path]:
        return {
            family: self.bagen_root / "audits" / filename
            for family, filename in BAGEN_FAMILY_AUDITS.items()
        }

    def build(self, **overrides: object) -> dict[str, object]:
        arguments = {
            "workspace_root": self.workspace,
            "changed_files": ["scripts/freeze_trajectory_handoff.py", "tests/test_freeze.py"],
            "validation_results": [
                {
                    "name": "ruff",
                    "command": "python -m ruff check src tests scripts",
                    "status": "passed",
                    "result": "All checks passed!",
                },
                {
                    "name": "pytest",
                    "command": "python -m pytest -q",
                    "status": "passed",
                    "result": "2 passed",
                },
            ],
            "repo_root": REPO_ROOT,
            "bagen_expected_combined_dataset_id": _sha_bytes(
                b"combined-bagen-dataset"
            ),
            "bagen_expected_combined_counts": {
                "task_id_count": 1,
                "run_id_count": 5,
                "trajectory_id_count": 5,
                "condition_id_count": 5,
                "dataset_row_count": 5,
                "raw_file_count": 5,
                "raw_bytes": sum(
                    path.stat().st_size
                    for path in (self.bagen_root / "origin").rglob("*.traj.json")
                ),
            },
            "bagen_expected_task_cross_family_distribution": {"5": 1},
            "spend_code_artifact_pins": {
                role: {"path": path, "sha256": _sha_file(REPO_ROOT / path)}
                for role, path in {
                    "reader": "src/token_prediction/collection/openhands_trajectory.py",
                    "builder": "src/token_prediction/dataset/builder.py",
                    "labels": "src/token_prediction/dataset/labels.py",
                    "audit": "scripts/audit_openhands_trajectory.py",
                }.items()
            },
            "spend_expected_repo": FIXTURE_SPEND_REPO,
            "spend_expected_revision": FIXTURE_SPEND_REVISION,
            "spend_expected_archive_bytes": self.archive.stat().st_size,
            "spend_expected_archive_sha256": _sha_file(self.archive),
            "spend_archive_xet_etag": FIXTURE_XET_ETAG,
        }
        arguments.update(overrides)
        return build_handoff(
            self.manifest_summary,
            self.bagen_combined,
            self.bagen_audits,
            self.spend_inventory,
            self.spend_audit,
            **arguments,  # type: ignore[arg-type]
        )


class FreezeTrajectoryHandoffTests(unittest.TestCase):
    def test_build_is_deterministic_complete_and_content_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            first = fixture.build()
            second = fixture.build()

            self.assertEqual(first, second)
            payload_hash = first["handoff_payload_sha256"]
            unhashed = dict(first)
            del unhashed["handoff_payload_sha256"]
            self.assertEqual(payload_hash, _semantic_sha(unhashed))

            encoded = json.dumps(first, ensure_ascii=False, sort_keys=True)
            self.assertNotIn("RAW_RESPONSE_SENTINEL", encoded)
            self.assertNotIn("RAW_MESSAGE_SENTINEL", encoded)
            self.assertNotIn(str(Path(temporary).resolve()), encoded)
            self.assertNotIn("generated_at", encoded.lower())
            self.assertNotIn("created_at", encoded.lower())

            bagen = first["sources"]["bagen"]
            self.assertEqual(bagen["identity"]["task_id_count"], 1)
            self.assertEqual(bagen["identity"]["run_id_count"], 5)
            self.assertEqual(bagen["identity"]["trajectory_id_count"], 5)
            self.assertEqual(
                bagen["identity"]["task_cross_family_mapping"][0]["family_count"], 5
            )
            gpt = next(
                item for item in bagen["families"] if item["family"] == "gpt5.2instant"
            )
            self.assertEqual(len(gpt["auxiliary_files"]), 5)
            self.assertTrue(all(item["sha256"] for item in gpt["auxiliary_files"]))
            self.assertEqual(
                bagen["combined_audit"]["schema_version"], 1
            )
            self.assertEqual(
                bagen["combined_audit"]["dataset_id"],
                _sha_bytes(b"combined-bagen-dataset"),
            )

            spend = first["sources"]["spend_your_money"]
            self.assertEqual(spend["identity"]["run_id_count"], 4)
            self.assertEqual(spend["identity"]["trajectory_id_count"], 4)
            self.assertEqual(
                spend["runs"][0]["output_jsonl"]["archive_internal_path"],
                "gpt_5.2_4runs/fixture-run_1/output.jsonl",
            )
            status = spend["position_target_matrix"]["call_pre"][
                "call_billable_output_tokens"
            ]["status_counts"]
            self.assertEqual(
                status, {"observed": 1, "missing": 1, "censored": 1, "invalid": 1}
            )
            self.assertEqual(
                spend["schema_pins"],
                {
                    "trajectory_audit_schema_version": 1,
                    "inventory_schema_version": 2,
                    "dataset_schema_version": 1,
                    "feature_schema_version": 2,
                },
            )
            self.assertEqual(
                set(spend["semantic_code_artifacts"]),
                {"reader", "builder", "labels", "audit"},
            )
            self.assertEqual(
                spend["telemetry_capability"]["session_reconciliation"][
                    "completion_logging_complete_count"
                ],
                4,
            )
            self.assertEqual(
                spend["inventory"]["report_evidence"][
                    "evaluator_report_missing_count"
                ],
                0,
            )
            self.assertFalse(first["policy"]["redistribution_allowed"])
            self.assertEqual(
                first["policy"]["missing_value_policy"],
                "never impute zero; preserve observed/missing/censored/invalid",
            )
            recommendations = first["recommended_experiments"]
            spend_task_total = next(
                item
                for item in recommendations["immediate"]
                if item["id"] == "spend_task_total_observed_subset"
            )
            self.assertEqual(
                spend_task_total["eligibility_evidence"],
                {
                    "observed_rows": 1_896,
                    "excluded_censored_rows": 104,
                    "excluded_censored_reason": "task_error",
                },
            )
            self.assertIn("group splits by task_id", spend_task_total["guard"])
            gated = {item["id"]: item for item in recommendations["gated"]}
            self.assertNotIn("spend_task_termination_prediction", gated)
            self.assertIn("post-launch outcome metadata", gated["task_lifecycle_feature_leakage"]["gate"])
            tool_gate = gated["cross_source_tool_failure_prediction"]["gate"]
            self.assertIn("instrumentation and failure semantics", tool_gate)
            self.assertIn("scope normalization", tool_gate)
            self.assertNotIn("not uniformly observable", tool_gate)
            self.assertTrue(
                first["implementation_validation"]["tests"]["all_passed"]
            )

    def test_failed_validation_gate_refuses_to_build_frozen_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            validation = [
                {
                    "name": "ruff",
                    "command": "python -m ruff check src tests scripts",
                    "status": "passed",
                    "result": "All checks passed!",
                },
                {
                    "name": "pytest",
                    "command": "python -m pytest -q",
                    "status": "failed",
                    "result": "1 failed",
                },
            ]
            with self.assertRaisesRegex(
                TrajectoryHandoffError,
                "freeze validation gates failed: pytest",
            ):
                fixture.build(validation_results=validation)

    def test_atomic_write_replaces_destination_without_leftover(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            output = fixture.workspace / "handoffs" / "summary.json"
            handoff = fixture.build()
            atomic_write_json(output, {"old": True})
            atomic_write_json(output, handoff)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), handoff)
            self.assertEqual(list(output.parent.glob(".summary.json.*.tmp")), [])

    def test_missing_trajectory_audit_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            fixture.spend_audit.unlink()
            with self.assertRaisesRegex(TrajectoryHandoffError, "Spend trajectory audit"):
                fixture.build()

    def test_missing_combined_bagen_audit_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            fixture.bagen_combined.unlink()
            with self.assertRaisesRegex(
                TrajectoryHandoffError, "BAGEN combined SWE-bench audit"
            ):
                fixture.build()

    def test_tampered_trajectory_audit_payload_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            audit = json.loads(fixture.spend_audit.read_text(encoding="utf-8"))
            audit["counts"]["trajectory_id_count"] = 99
            _write_json(fixture.spend_audit, audit)
            with self.assertRaisesRegex(TrajectoryHandoffError, "payload SHA256"):
                fixture.build()

    def test_rehashed_combined_bagen_count_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            audit = json.loads(fixture.bagen_combined.read_text(encoding="utf-8"))
            audit["counts"]["trajectory_id_count"] = 99
            _write_json(fixture.bagen_combined, audit)
            _rehash_payload_file(fixture.bagen_combined)
            with self.assertRaisesRegex(TrajectoryHandoffError, "counts do not close"):
                fixture.build()

    def test_rehashed_spend_canonical_mapping_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            audit = json.loads(fixture.spend_audit.read_text(encoding="utf-8"))
            audit["canonical_trajectories"][0]["task_id"] = "different-task"
            _write_json(fixture.spend_audit, audit)
            _rehash_payload_file(fixture.spend_audit)
            with self.assertRaisesRegex(TrajectoryHandoffError, "quadruples disagree"):
                fixture.build()

    def test_rehashed_spend_matrix_eligibility_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            audit = json.loads(fixture.spend_audit.read_text(encoding="utf-8"))
            cell = audit["label_matrix"]["call_pre"]["call_billable_output_tokens"]
            cell["eligible_row_count"] = 0
            _write_json(fixture.spend_audit, audit)
            _rehash_payload_file(fixture.spend_audit)
            with self.assertRaisesRegex(
                TrajectoryHandoffError, "eligible rows/boolean disagree"
            ):
                fixture.build()

    def test_rehashed_spend_matrix_row_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            audit = json.loads(fixture.spend_audit.read_text(encoding="utf-8"))
            cell = audit["label_matrix"]["call_pre"]["call_billable_output_tokens"]
            cell["row_count"] = 5
            _write_json(fixture.spend_audit, audit)
            _rehash_payload_file(fixture.spend_audit)
            with self.assertRaisesRegex(TrajectoryHandoffError, "row count does not close"):
                fixture.build()

    def test_missing_status_key_is_never_implicitly_zero_filled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            audit = json.loads(fixture.spend_audit.read_text(encoding="utf-8"))
            cell = audit["label_matrix"]["call_pre"]["call_billable_output_tokens"]
            del cell["status_counts"]["invalid"]
            _write_json(fixture.spend_audit, audit)
            _rehash_payload_file(fixture.spend_audit)
            with self.assertRaisesRegex(
                TrajectoryHandoffError,
                "explicitly contain observed/missing/censored/invalid",
            ):
                fixture.build()

    def test_rehashed_spend_unavailable_telemetry_cannot_be_observed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            audit = json.loads(fixture.spend_audit.read_text(encoding="utf-8"))
            cell = audit["per_run"]["run_1"]["label_matrix"]["call_pre"][
                "call_unknown_billable_tokens"
            ]
            cell.update(
                {
                    "row_count": 1,
                    "eligible_row_count": 1,
                    "eligible_for_supervised_training": True,
                    "status_counts": _status_counts(observed=1),
                    "reason_counts": {},
                }
            )
            _write_json(fixture.spend_audit, audit)
            _rehash_payload_file(fixture.spend_audit)
            with self.assertRaisesRegex(
                TrajectoryHandoffError, "observed rows for unavailable telemetry"
            ):
                fixture.build()

    def test_rehashed_spend_capability_count_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            audit = json.loads(fixture.spend_audit.read_text(encoding="utf-8"))
            audit["source_capability"]["attempt_usage"]["complete_count"] = 99
            _write_json(fixture.spend_audit, audit)
            _rehash_payload_file(fixture.spend_audit)
            with self.assertRaisesRegex(TrajectoryHandoffError, "disagrees with metrics"):
                fixture.build()

    def test_rehashed_spend_capability_unknown_field_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            audit = json.loads(fixture.spend_audit.read_text(encoding="utf-8"))
            audit["source_capability"]["session_reconciliation"]["raw_message"] = (
                "must not enter handoff"
            )
            _write_json(fixture.spend_audit, audit)
            _rehash_payload_file(fixture.spend_audit)
            with self.assertRaisesRegex(TrajectoryHandoffError, "fields disagree"):
                fixture.build()

    def test_rehashed_explicit_zero_count_cannot_exceed_complete_usage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            audit = json.loads(fixture.spend_audit.read_text(encoding="utf-8"))
            audit["source_capability"]["task_usage"]["explicit_zero_call_count"] = 4
            audit["metrics"]["task_usage_explicit_zero_call_count"] = 4
            _write_json(fixture.spend_audit, audit)
            _rehash_payload_file(fixture.spend_audit)
            with self.assertRaisesRegex(TrajectoryHandoffError, "task_usage.*does not close"):
                fixture.build()

    def test_rehashed_inventory_report_reconciliation_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            inventory = json.loads(fixture.spend_inventory.read_text(encoding="utf-8"))
            inventory["runs"][0]["task_report_count"] = 2
            inventory["runs"][0]["report_count"] = 3
            _write_json(fixture.spend_inventory, inventory)
            audit = json.loads(fixture.spend_audit.read_text(encoding="utf-8"))
            audit["inventory"]["sha256"] = _sha_file(fixture.spend_inventory)
            _write_json(fixture.spend_audit, audit)
            _rehash_payload_file(fixture.spend_audit)
            with self.assertRaisesRegex(
                TrajectoryHandoffError, "report counts do not close"
            ):
                fixture.build()

    def test_rehashed_inventory_aggregate_report_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            inventory = json.loads(fixture.spend_inventory.read_text(encoding="utf-8"))
            inventory["runs"][0]["aggregate_report_count"] = 2
            inventory["runs"][0]["report_count"] = 3
            inventory["aggregate_report_count"] = 5
            inventory["report_count"] = 9
            _write_json(fixture.spend_inventory, inventory)
            audit = json.loads(fixture.spend_audit.read_text(encoding="utf-8"))
            audit["inventory"]["sha256"] = _sha_file(fixture.spend_inventory)
            _write_json(fixture.spend_audit, audit)
            _rehash_payload_file(fixture.spend_audit)
            with self.assertRaisesRegex(TrajectoryHandoffError, "report counts do not close"):
                fixture.build()

    def test_spend_archive_and_audit_paths_are_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            inventory = json.loads(fixture.spend_inventory.read_text(encoding="utf-8"))
            inventory["archive_path"] = "workspace/raw/gpt_5.2_4runs.tar.gz"
            _write_json(fixture.spend_inventory, inventory)
            audit = json.loads(fixture.spend_audit.read_text(encoding="utf-8"))
            audit["inventory"]["sha256"] = _sha_file(fixture.spend_inventory)
            audit["archive"]["local_relative_path"] = inventory["archive_path"]
            _write_json(fixture.spend_audit, audit)
            _rehash_payload_file(fixture.spend_audit)
            with self.assertRaisesRegex(TrajectoryHandoffError, "ignored-workspace path"):
                fixture.build()

    def test_spend_archive_member_segments_are_canonical_and_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            inventory = json.loads(fixture.spend_inventory.read_text(encoding="utf-8"))
            inventory["runs"][0]["run_basename"] = ".."
            _write_json(fixture.spend_inventory, inventory)
            audit = json.loads(fixture.spend_audit.read_text(encoding="utf-8"))
            audit["inventory"]["sha256"] = _sha_file(fixture.spend_inventory)
            _write_json(fixture.spend_audit, audit)
            _rehash_payload_file(fixture.spend_audit)
            with self.assertRaisesRegex(TrajectoryHandoffError, "canonical path segment"):
                fixture.build()

    def test_spend_inventory_source_status_and_member_counts_are_strict(self) -> None:
        mutations = (
            (
                "source",
                lambda inventory: inventory.update(source_id="different/source"),
            ),
            (
                "status",
                lambda inventory: inventory["telemetry_status_counts"].update(
                    raw_content="must not enter handoff"
                ),
            ),
            (
                "record count",
                lambda inventory: inventory["runs"][0]["output_jsonl"]["files"][
                    0
                ].update(record_count=2),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = _Fixture(Path(temporary))
                inventory = json.loads(
                    fixture.spend_inventory.read_text(encoding="utf-8")
                )
                mutate(inventory)
                _write_json(fixture.spend_inventory, inventory)
                audit = json.loads(fixture.spend_audit.read_text(encoding="utf-8"))
                audit["inventory"]["sha256"] = _sha_file(fixture.spend_inventory)
                _write_json(fixture.spend_audit, audit)
                _rehash_payload_file(fixture.spend_audit)
                with self.assertRaises(TrajectoryHandoffError):
                    fixture.build()

    def test_bagen_declared_identity_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            path = fixture.bagen_audits["qwen3-235b"]
            audit = json.loads(path.read_text(encoding="utf-8"))
            audit["task_count"] = 2
            _write_json(path, audit)
            with self.assertRaisesRegex(TrajectoryHandoffError, "identity totals disagree"):
                fixture.build()

    def test_bagen_family_matrix_row_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            path = fixture.bagen_audits["qwen3-235b"]
            audit = json.loads(path.read_text(encoding="utf-8"))
            audit["dataset"]["by_position_target"][0]["row_count"] = 1
            _write_json(path, audit)
            with self.assertRaisesRegex(TrajectoryHandoffError, "row count does not close"):
                fixture.build()

    def test_bagen_family_canonical_aggregate_is_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            path = fixture.bagen_audits["qwen3-235b"]
            audit = json.loads(path.read_text(encoding="utf-8"))
            audit["canonical_content_sha256"] = "f" * 64
            audit["canonical_rerun_content_sha256"] = "f" * 64
            _write_json(path, audit)
            with self.assertRaisesRegex(TrajectoryHandoffError, "canonical family/rerun"):
                fixture.build()

    def test_bagen_family_schema_and_telemetry_are_strict(self) -> None:
        mutations = (
            ("schema", lambda audit: audit.update(audit_schema_version=99)),
            ("telemetry", lambda audit: audit.update(attempt_count=2)),
        )
        for name, mutate in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = _Fixture(Path(temporary))
                path = fixture.bagen_audits["qwen3-235b"]
                audit = json.loads(path.read_text(encoding="utf-8"))
                mutate(audit)
                _write_json(path, audit)
                with self.assertRaises(TrajectoryHandoffError):
                    fixture.build()

    def test_rehashed_bagen_combined_redundant_evidence_tamper_fails_closed(self) -> None:
        mutations = (
            (
                "task mapping",
                lambda audit: audit["task_family_mapping"][0].update(family_count=4),
            ),
            (
                "source role",
                lambda audit: audit["source_files"].update(
                    reader=dict(audit["source_files"]["builder"])
                ),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = _Fixture(Path(temporary))
                audit = json.loads(fixture.bagen_combined.read_text(encoding="utf-8"))
                mutate(audit)
                _write_json(fixture.bagen_combined, audit)
                _rehash_payload_file(fixture.bagen_combined)
                with self.assertRaises(TrajectoryHandoffError):
                    fixture.build()

    def test_duplicate_manifest_path_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            manifest = fixture.bagen_root / "manifest.jsonl"
            first_line = manifest.read_text(encoding="utf-8").splitlines()[0]
            manifest.write_text(first_line + "\n" + first_line + "\n", encoding="utf-8")
            with self.assertRaisesRegex(TrajectoryHandoffError, "repeats path"):
                from scripts.freeze_trajectory_handoff import _load_bagen_manifest_index

                _load_bagen_manifest_index(manifest)

    def test_manifest_size_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            manifest = fixture.bagen_root / "manifest.jsonl"
            entries = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
            ]
            trajectory = next(
                item for item in entries if str(item["path"]).endswith(".traj.json")
            )
            trajectory["size_bytes"] += 1
            manifest.write_text(
                "".join(
                    json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                    for item in entries
                ),
                encoding="utf-8",
            )
            summary = json.loads(fixture.manifest_summary.read_text(encoding="utf-8"))
            summary.update(
                {
                    "manifest_bytes": manifest.stat().st_size,
                    "manifest_etag": _git_blob_sha(manifest),
                    "manifest_sha256": _sha_file(manifest),
                    "total_bytes": sum(int(item["size_bytes"]) for item in entries),
                    "traj_json_bytes": sum(
                        int(item["size_bytes"])
                        for item in entries
                        if str(item["path"]).endswith(".traj.json")
                    ),
                }
            )
            _write_json(fixture.manifest_summary, summary)
            combined = json.loads(fixture.bagen_combined.read_text(encoding="utf-8"))
            combined["manifest"]["summary"].update(
                {
                    "bytes": fixture.manifest_summary.stat().st_size,
                    "sha256": _sha_file(fixture.manifest_summary),
                }
            )
            combined["manifest"]["raw"].update(
                {
                    "bytes": manifest.stat().st_size,
                    "sha256": _sha_file(manifest),
                    "git_blob_etag": _git_blob_sha(manifest),
                }
            )
            _write_json(fixture.bagen_combined, combined)
            _rehash_payload_file(fixture.bagen_combined)
            with self.assertRaisesRegex(TrajectoryHandoffError, "manifest path/size mismatch"):
                fixture.build()

    def test_duplicate_and_non_finite_json_fail_closed(self) -> None:
        invalid_documents = {
            "duplicate": '{"file_count": 1, "file_count": 1}',
            "nan": '{"file_count": NaN}',
            "infinity": '{"file_count": 1e999}',
        }
        for name, document in invalid_documents.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = _Fixture(Path(temporary))
                fixture.manifest_summary.write_text(document, encoding="utf-8")
                with self.assertRaises(TrajectoryHandoffError):
                    fixture.build()

    def test_absolute_changed_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            for path in (
                "C:/secret/file.txt",
                "C:\\secret\\file.txt",
                "\\\\server\\share\\file.txt",
                "\\secret\\file.txt",
                "/var/tmp/file.txt",
            ):
                with self.subTest(path=path), self.assertRaisesRegex(
                    TrajectoryHandoffError, "canonical POSIX relative path"
                ):
                    fixture.build(changed_files=[path])

    def test_absolute_path_in_nested_validation_result_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            for path in (
                "D:/private/result.txt",
                "\\\\server\\share\\result.txt",
                "\\private\\result.txt",
                "/opt/private/result.txt",
            ):
                validation = [
                    {
                        "name": "pytest",
                        "command": "python -m pytest -q",
                        "status": "failed",
                        "result": f"failure evidence: {path}",
                    }
                ]
                with self.subTest(path=path), self.assertRaisesRegex(
                    TrajectoryHandoffError, "absolute path|absolute local path"
                ):
                    fixture.build(validation_results=validation)

    def test_absolute_paths_in_mapping_keys_and_prefixed_text_are_rejected(self) -> None:
        for value in (
            {"/home/user/raw.json": "value"},
            {"C:\\Users\\user\\raw.json": "value"},
            {"\\\\server\\share\\raw.json": "value"},
        ):
            with self.subTest(value=value), self.assertRaises(TrajectoryHandoffError):
                _assert_no_absolute_paths(value)
        for value in (
            "prefix:/home/user/raw.json",
            "prefix:C:\\Users\\user\\raw.json",
            "prefix:\\\\server\\share\\raw.json",
        ):
            with self.subTest(value=value):
                self.assertTrue(_contains_absolute_local_path(value))


if __name__ == "__main__":
    unittest.main()
