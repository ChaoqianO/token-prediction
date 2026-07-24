"""Verify the frozen Stage 4 selection and one-time final-holdout release.

The full verifier deliberately stops at immutable artifacts and selected model
bundles.  It never rebuilds a dataset, reads the final source cohort, predicts,
or scores labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from token_prediction.estimators.lightgbm_bundle import load_lightgbm_bundle
from token_prediction.evaluation import FittedExpansionCalibrator
from token_prediction.final_ensemble import EmpiricalFoldState, semantic_sha256
from token_prediction.lifecycle_bundle import load_lifecycle_bundle
from token_prediction.lineage import sha256_file, verify_artifact

if __package__:
    from scripts.prepare_stage4_selection import (
        SELECTION_ENSEMBLE_POLICY_ID,
        SELECTION_POLICY_ID,
        SOURCE_ARTIFACTS,
        Stage4SelectionError,
        verify_selection_document,
    )
    from scripts.run_data_foundation_baseline import (
        _is_link_or_reparse,
        _repo_path,
        _safe_relative,
    )
    from scripts.run_stage4_final import (
        DEFAULT_SELECTION_LOCK,
        FINAL_RUN_POLICY_ID,
        FINAL_STAGE_NAME,
        SELECTION_TAG,
        Stage4FinalError,
        _validate_selection_lock_document,
        verify_final_results_document,
    )
    from scripts.run_stage4_experiments import (
        Stage4ExperimentError,
        verify_stage4_results_document,
    )
else:  # pragma: no cover - production CLI invocation
    from prepare_stage4_selection import (
        SELECTION_ENSEMBLE_POLICY_ID,
        SELECTION_POLICY_ID,
        SOURCE_ARTIFACTS,
        Stage4SelectionError,
        verify_selection_document,
    )
    from run_data_foundation_baseline import (
        _is_link_or_reparse,
        _repo_path,
        _safe_relative,
    )
    from run_stage4_final import (
        DEFAULT_SELECTION_LOCK,
        FINAL_RUN_POLICY_ID,
        FINAL_STAGE_NAME,
        SELECTION_TAG,
        Stage4FinalError,
        _validate_selection_lock_document,
        verify_final_results_document,
    )
    from run_stage4_experiments import (
        Stage4ExperimentError,
        verify_stage4_results_document,
    )


DEFAULT_RELEASE_LOCK = "configs/stage4_release.json"
DEFAULT_REPORT = "docs/stage-4-final-report.md"
RELEASE_SCHEMA_VERSION = 1
RELEASE_POLICY_ID = "stage4_commit_bound_single_final_holdout_release_v1"
EVALUATION_CODE_POLICY_ID = "stage4_final_evaluation_code_tree_v1"
HISTORICAL_CLOSURE_POLICY_ID = "stage4_historical_evaluation_closure_amendment_v1"
HISTORICAL_AMENDED_CODE_POLICY_ID = (
    "stage4_final_executed_code_tree_release_amendment_v1"
)
HISTORICAL_OBSERVATION_POLICY_ID = "stage4_contemporaneous_execution_observation_v1"
SELECTION_CODE_POLICY_ID = "stage4_selection_code_tree_v1"
VERIFICATION_MODE_ID = "artifact_only_no_source_replay_v1"
EXPECTED_CELL_COUNT = 29
EXPECTED_MEMBER_COUNT = 435
EXPECTED_MEMBERS_PER_CELL = 15
EXPECTED_PREDICTION_COUNT = 86_335
EXPECTED_SOURCE_COUNT = 4
EXPECTED_SELECTION_MANIFEST_FILES = 16
EXPECTED_FINAL_MANIFEST_FILES = 2

MAX_RELEASE_JSON_BYTES = 2 * 1024 * 1024
MAX_SELECTION_LOCK_BYTES = 2 * 1024 * 1024
MAX_SELECTION_JSON_BYTES = 16 * 1024 * 1024
MAX_FINAL_RESULTS_BYTES = 32 * 1024 * 1024
MAX_REPORT_BYTES = 4 * 1024 * 1024
MAX_LEDGER_BYTES = 2 * 1024 * 1024
MAX_MEMBER_JSON_BYTES = 4 * 1024 * 1024
MAX_BUNDLE_FILE_BYTES = 1024 * 1024 * 1024
MAX_BUNDLE_TOTAL_BYTES = 4 * 1024 * 1024 * 1024
MAX_BUNDLE_FILE_COUNT = 10_000
MAX_SOURCE_RESULTS_BYTES = 32 * 1024 * 1024

ORIGINAL_EVALUATION_EXPLICIT_PATHS = frozenset(
    {
        "scripts/run_stage4_final.py",
        "scripts/prepare_stage4_selection.py",
        DEFAULT_SELECTION_LOCK,
    }
)
HISTORICAL_EXECUTED_EXPLICIT_PATHS = frozenset(
    {
        *ORIGINAL_EVALUATION_EXPLICIT_PATHS,
        "scripts/run_data_foundation_baseline.py",
        "scripts/run_stage2_experiments.py",
        "scripts/run_stage3_experiments.py",
        "scripts/run_stage4_experiments.py",
        "configs/data_foundation_prediction_baseline.json",
        "configs/data_foundation_v2_baseline.json",
        "configs/stage2_auxiliary_sources.json",
        "configs/source_descriptors/bagen_swebench.json",
        "configs/source_descriptors/spend_openhands.json",
    }
)
HISTORICAL_ADDED_EXPLICIT_PATHS = tuple(
    sorted(HISTORICAL_EXECUTED_EXPLICIT_PATHS - ORIGINAL_EVALUATION_EXPLICIT_PATHS)
)
PROTECTED_RELEASE_TAGS = (
    "stage2-artifact-source-v1",
    "stage3-artifact-source-v1",
    "stage4-artifact-source-v1",
    "stage4-final-selection-v1",
    "stage4-final-release-v1",
)

METADATA_AMENDMENTS = [
    {
        "policy_id": "stage4_final_dataset_task_count_amendment_v1",
        "source_artifact_key": "stage4_bagen_swebench",
        "artifact_field": "datasets[source_name=bagen_swebench].task_count",
        "artifact_value": 13,
        "authoritative_field": (
            "development_protocol.permanent_holdout."
            "assignments[cohort=final_holdout]"
        ),
        "authoritative_value": 14,
        "semantic_correction": (
            "source_holdout_task_count_not_first_cell_scored_task_count"
        ),
        "impact": "metadata_only_no_prediction_selection_or_score_change",
    }
]

_SELECTION_ARTIFACT_KEYS = {
    "path",
    "artifact_id",
    "run_id",
    "selection_id",
    "selection_payload_sha256",
    "manifest_file_count",
}
_FINAL_ARTIFACT_KEYS = {
    "path",
    "artifact_id",
    "run_id",
    "selection_id",
    "results_payload_sha256",
    "manifest_file_count",
}
_POINT_MEMBER_KEYS = {
    "origin",
    "bundle_kind",
    "split_seed",
    "split_plan_id",
    "fold",
    "bundle_path",
    "bundle_tree_sha256",
    "bundle_file_count",
    "calibrator_path",
    "calibrator_sha256",
    "provenance_path",
    "provenance_sha256",
    "member_sha256",
}
_EMPIRICAL_MEMBER_KEYS = {
    "origin",
    "bundle_kind",
    "split_seed",
    "split_plan_id",
    "fold",
    "state_path",
    "state_sha256",
    "member_sha256",
}


class Stage4ReleaseError(RuntimeError):
    """The Stage 4 release evidence does not close safely."""


@dataclass(frozen=True)
class Stage4ReleaseVerification:
    lock_path: str
    report_path: str
    selection_commit: str
    evaluation_code_tree_sha256: str
    locked_cell_count: int
    locked_member_count: int
    locked_prediction_count: int
    verified_artifact_count: int
    verified_member_count: int
    independently_loaded_bundle_count: int
    verified_empirical_state_count: int
    metadata_amendment_count: int
    final_holdout_evaluation_count: int
    source_data_replayed: bool


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Stage4ReleaseError("Stage 4 release JSON contains duplicate keys")
        result[key] = value
    return result


def _constant(value: str) -> Any:
    raise Stage4ReleaseError(f"Stage 4 release JSON contains non-finite value {value}")


def _reject_non_finite(value: object, *, description: str) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Stage4ReleaseError(f"{description} contains a non-finite number")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _reject_non_finite(item, description=f"{description}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_finite(item, description=f"{description}[{index}]")


def _regular_file(path: Path, *, maximum_bytes: int, description: str) -> bytes:
    if _is_link_or_reparse(path):
        raise Stage4ReleaseError(f"{description} must not be a link or reparse point")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow:
        flags |= nofollow
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Stage4ReleaseError(f"{description} is missing or unreadable") from exc
    try:
        try:
            before = os.fstat(descriptor)
            if _is_link_or_reparse(path):
                raise Stage4ReleaseError(
                    f"{description} must not be a link or reparse point"
                )
            if not stat.S_ISREG(before.st_mode):
                raise Stage4ReleaseError(f"{description} must be a regular file")
            if before.st_size <= 0 or before.st_size > maximum_bytes:
                raise Stage4ReleaseError(f"{description} has an invalid size")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            after = os.fstat(descriptor)
            path_after = path.lstat()
        except OSError as exc:
            raise Stage4ReleaseError(f"{description} is unreadable") from exc
    finally:
        os.close(descriptor)
    before_snapshot = (
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        getattr(before, "st_ino", 0),
        getattr(before, "st_dev", 0),
    )
    after_snapshot = (
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        getattr(after, "st_ino", 0),
        getattr(after, "st_dev", 0),
    )
    handle_identity = (
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        getattr(after, "st_ino", 0),
        getattr(after, "st_dev", 0),
    )
    path_identity = (
        path_after.st_mode,
        path_after.st_size,
        path_after.st_mtime_ns,
        getattr(path_after, "st_ino", 0),
        getattr(path_after, "st_dev", 0),
    )
    if (
        len(payload) != before.st_size
        or before_snapshot != after_snapshot
        or handle_identity != path_identity
        or _is_link_or_reparse(path)
    ):
        raise Stage4ReleaseError(f"{description} changed while being read")
    return payload


def _parse_json_payload(payload: bytes, *, description: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage4ReleaseError(f"{description} is not strict UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise Stage4ReleaseError(f"{description} must contain a JSON object")
    _reject_non_finite(value, description=description)
    return value


def _load_json(
    path: Path,
    *,
    maximum_bytes: int,
    description: str,
) -> Mapping[str, Any]:
    payload = _regular_file(path, maximum_bytes=maximum_bytes, description=description)
    return _parse_json_payload(payload, description=description)


def _exact(
    value: object,
    keys: set[str],
    *,
    description: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise Stage4ReleaseError(f"{description} keys do not match")
    return value


def _text(value: object, *, description: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise Stage4ReleaseError(f"{description} must be normalized non-empty text")
    return value


def _integer(value: object, *, description: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Stage4ReleaseError(f"{description} must be an integer >= {minimum}")
    return value


def _sha256(value: object, *, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Stage4ReleaseError(f"{description} must be a lowercase SHA-256")
    return value


def _commit(value: object, *, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Stage4ReleaseError(f"{description} must be a lowercase Git commit")
    return value


def _relative(value: object, *, description: str) -> str:
    try:
        return _safe_relative(value, label=description)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Stage4ReleaseError(f"{description} is not a safe relative path") from exc


def _path(root: Path, value: object, *, description: str) -> Path:
    relative = _relative(value, description=description)
    try:
        return _repo_path(root, relative, label=description)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Stage4ReleaseError(f"{description} is unsafe") from exc


def _git(root: Path, *arguments: str, maximum_bytes: int = 32 * 1024 * 1024) -> bytes:
    completed = subprocess.run(
        ["git", "-c", "core.quotepath=false", *arguments],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise Stage4ReleaseError(f"Git command failed: {message}")
    if len(completed.stdout) > maximum_bytes:
        raise Stage4ReleaseError("Git command output exceeds its safe size limit")
    return completed.stdout


def _validate_release_document(value: Mapping[str, Any]) -> None:
    _exact(
        value,
        {
            "release_schema_version",
            "stage_name",
            "policy_id",
            "selection",
            "final_artifact",
            "evaluation_code_binding",
            "historical_code_closure_amendments",
            "remote_controls",
            "source_artifacts",
            "metadata_amendments",
            "protocol",
            "ledger",
            "report",
        },
        description="Stage 4 release",
    )
    if (
        value["release_schema_version"] != RELEASE_SCHEMA_VERSION
        or value["stage_name"] != FINAL_STAGE_NAME
        or value["policy_id"] != RELEASE_POLICY_ID
    ):
        raise Stage4ReleaseError("Stage 4 release policy identity is invalid")

    selection = _exact(
        value["selection"],
        {"lock_path", "lock_sha256", "tag", "commit", "artifact"},
        description="Stage 4 selection release binding",
    )
    if _relative(
        selection["lock_path"], description="Stage 4 selection lock path"
    ) != DEFAULT_SELECTION_LOCK:
        raise Stage4ReleaseError("Stage 4 selection lock path is invalid")
    _sha256(selection["lock_sha256"], description="Stage 4 selection lock")
    if selection["tag"] != SELECTION_TAG:
        raise Stage4ReleaseError("Stage 4 selection tag is invalid")
    _commit(selection["commit"], description="Stage 4 selection commit")
    selection_artifact = _exact(
        selection["artifact"],
        _SELECTION_ARTIFACT_KEYS,
        description="Stage 4 selection artifact binding",
    )
    selection_path = _relative(
        selection_artifact["path"],
        description="Stage 4 selection artifact path",
    )
    if not selection_path.startswith("workspace/stage4/selection/"):
        raise Stage4ReleaseError("Stage 4 selection artifact leaves its workspace")
    for name in ("artifact_id", "selection_id", "selection_payload_sha256"):
        _sha256(
            selection_artifact[name],
            description=f"Stage 4 selection artifact {name}",
        )
    _text(selection_artifact["run_id"], description="Stage 4 selection run id")
    if (
        _integer(
            selection_artifact["manifest_file_count"],
            description="Stage 4 selection manifest file count",
            minimum=1,
        )
        != EXPECTED_SELECTION_MANIFEST_FILES
    ):
        raise Stage4ReleaseError("Stage 4 selection manifest cardinality is invalid")

    final_artifact = _exact(
        value["final_artifact"],
        _FINAL_ARTIFACT_KEYS,
        description="Stage 4 final artifact binding",
    )
    final_path = _relative(
        final_artifact["path"],
        description="Stage 4 final artifact path",
    )
    if not final_path.startswith("workspace/stage4/final/"):
        raise Stage4ReleaseError("Stage 4 final artifact leaves its workspace")
    for name in ("artifact_id", "selection_id", "results_payload_sha256"):
        _sha256(final_artifact[name], description=f"Stage 4 final artifact {name}")
    _text(final_artifact["run_id"], description="Stage 4 final run id")
    if (
        _integer(
            final_artifact["manifest_file_count"],
            description="Stage 4 final manifest file count",
            minimum=1,
        )
        != EXPECTED_FINAL_MANIFEST_FILES
    ):
        raise Stage4ReleaseError("Stage 4 final manifest cardinality is invalid")
    if final_artifact["selection_id"] != selection_artifact["selection_id"]:
        raise Stage4ReleaseError("Stage 4 selection and final artifacts disagree")

    code = _exact(
        value["evaluation_code_binding"],
        {"policy_id", "git_commit", "code_tree_sha256"},
        description="Stage 4 evaluation code binding",
    )
    if (
        code["policy_id"] != EVALUATION_CODE_POLICY_ID
        or _commit(code["git_commit"], description="evaluation code commit")
        != selection["commit"]
    ):
        raise Stage4ReleaseError("Stage 4 evaluation code identity is invalid")
    _sha256(code["code_tree_sha256"], description="evaluation code tree")

    amendments = value["historical_code_closure_amendments"]
    if not isinstance(amendments, list) or len(amendments) != 1:
        raise Stage4ReleaseError(
            "Stage 4 release must contain one historical code closure amendment"
        )
    amendment = _exact(
        amendments[0],
        {
            "policy_id",
            "selection_commit",
            "selection_code_commit",
            "original_binding",
            "amended_binding",
            "added_paths",
            "execution_observation",
            "impact",
            "residual_limitation",
        },
        description="historical code closure amendment",
    )
    if (
        amendment["policy_id"] != HISTORICAL_CLOSURE_POLICY_ID
        or amendment["selection_commit"] != selection["commit"]
    ):
        raise Stage4ReleaseError("historical code closure amendment identity is invalid")
    _commit(
        amendment["selection_code_commit"],
        description="historical selection code commit",
    )
    original = _exact(
        amendment["original_binding"],
        {"policy_id", "git_commit", "code_tree_sha256", "path_count"},
        description="historical original code binding",
    )
    if (
        original["policy_id"] != code["policy_id"]
        or original["git_commit"] != code["git_commit"]
        or original["code_tree_sha256"] != code["code_tree_sha256"]
    ):
        raise Stage4ReleaseError("historical original code binding differs from artifact")
    _integer(
        original["path_count"],
        description="historical original path count",
        minimum=1,
    )
    amended = _exact(
        amendment["amended_binding"],
        {
            "policy_id",
            "git_commit",
            "code_tree_sha256",
            "path_count",
            "path_projection_sha256",
        },
        description="historical amended code binding",
    )
    if (
        amended["policy_id"] != HISTORICAL_AMENDED_CODE_POLICY_ID
        or amended["git_commit"] != selection["commit"]
    ):
        raise Stage4ReleaseError("historical amended code identity is invalid")
    _sha256(amended["code_tree_sha256"], description="historical amended code tree")
    _sha256(
        amended["path_projection_sha256"],
        description="historical amended path projection",
    )
    _integer(
        amended["path_count"],
        description="historical amended path count",
        minimum=1,
    )
    added_paths = amendment["added_paths"]
    if not isinstance(added_paths, list) or len(added_paths) != len(
        HISTORICAL_ADDED_EXPLICIT_PATHS
    ):
        raise Stage4ReleaseError("historical added path records are invalid")
    for record in added_paths:
        bound = _exact(
            record,
            {"path", "git_blob_oid", "sha256"},
            description="historical added path record",
        )
        _relative(bound["path"], description="historical added path")
        _text(bound["git_blob_oid"], description="historical Git blob object")
        _sha256(bound["sha256"], description="historical added path SHA-256")
    observation = _exact(
        amendment["execution_observation"],
        {
            "policy_id",
            "tracked_worktree_clean",
            "observed_process_count",
            "recorded_pid",
            "canonical_output_root",
            "canonical_checkpoint_root",
            "canonical_final_artifact_count",
            "canonical_checkpoint_run_count",
            "checkpoint_cell_count",
        },
        description="historical final execution observation",
    )
    if observation != {
        "policy_id": HISTORICAL_OBSERVATION_POLICY_ID,
        "tracked_worktree_clean": True,
        "observed_process_count": 1,
        "recorded_pid": 4984,
        "canonical_output_root": "workspace/stage4/final",
        "canonical_checkpoint_root": "workspace/stage4/final-checkpoints",
        "canonical_final_artifact_count": 1,
        "canonical_checkpoint_run_count": 1,
        "checkpoint_cell_count": EXPECTED_CELL_COUNT,
    }:
        raise Stage4ReleaseError("historical final execution observation is invalid")
    if (
        amendment["impact"]
        != "code_provenance_only_no_prediction_selection_calibration_or_score_change"
        or amendment["residual_limitation"]
        != "historical_runner_lacked_intrinsic_exclusive_open_and_complete_import_origin_binding"
    ):
        raise Stage4ReleaseError("historical code closure disclosure is invalid")

    remote = _exact(
        value["remote_controls"],
        {
            "provider",
            "repository",
            "tag_ruleset_id",
            "ruleset_name",
            "final_release_tag",
            "target",
            "enforcement",
            "bypass_actor_count",
            "rules",
            "protected_tags",
        },
        description="Stage 4 remote release controls",
    )
    if remote != {
        "provider": "github",
        "repository": "ChaoqianO/token-prediction",
        "tag_ruleset_id": 19652329,
        "ruleset_name": "Protect immutable experiment tags",
        "final_release_tag": "stage4-final-release-v1",
        "target": "tag",
        "enforcement": "active",
        "bypass_actor_count": 0,
        "rules": ["update", "deletion"],
        "protected_tags": list(PROTECTED_RELEASE_TAGS),
    }:
        raise Stage4ReleaseError("Stage 4 remote release controls are invalid")

    expected_sources = [asdict(item) for item in SOURCE_ARTIFACTS]
    if value["source_artifacts"] != expected_sources:
        raise Stage4ReleaseError("Stage 4 source artifacts differ from frozen inventory")
    if value["metadata_amendments"] != METADATA_AMENDMENTS:
        raise Stage4ReleaseError("Stage 4 metadata amendments are invalid")

    protocol = value["protocol"]
    if protocol != {
        "run_policy_id": FINAL_RUN_POLICY_ID,
        "selection_policy_id": SELECTION_POLICY_ID,
        "ensemble_policy_id": SELECTION_ENSEMBLE_POLICY_ID,
        "final_holdout_evaluation_count": 1,
        "final_holdout_prediction_count": EXPECTED_PREDICTION_COUNT,
        "selected_cell_count": EXPECTED_CELL_COUNT,
        "ensemble_member_count": EXPECTED_MEMBER_COUNT,
        "member_count_per_cell": EXPECTED_MEMBERS_PER_CELL,
        "refit_selected_learned_models": False,
        "calibration_application_count": 1,
        "verification_mode": VERIFICATION_MODE_ID,
    }:
        raise Stage4ReleaseError("Stage 4 final protocol is invalid")

    ledger = _exact(
        value["ledger"],
        {
            "path",
            "schema_version",
            "status",
            "completed_cell_count",
            "final_artifact_id",
        },
        description="Stage 4 final ledger binding",
    )
    expected_ledger_path = (
        f"workspace/stage4/final-checkpoints/{final_artifact['run_id']}/ledger.json"
    )
    if (
        _relative(ledger["path"], description="Stage 4 final ledger path")
        != expected_ledger_path
        or ledger["schema_version"] != 1
        or ledger["status"] != "published"
        or ledger["completed_cell_count"] != EXPECTED_CELL_COUNT
        or ledger["final_artifact_id"] != final_artifact["artifact_id"]
    ):
        raise Stage4ReleaseError("Stage 4 final ledger binding is invalid")

    report = _exact(
        value["report"],
        {"path", "sha256"},
        description="Stage 4 final report binding",
    )
    if (
        _relative(report["path"], description="Stage 4 final report path")
        != DEFAULT_REPORT
    ):
        raise Stage4ReleaseError("Stage 4 final report path is invalid")
    _sha256(report["sha256"], description="Stage 4 final report")


def _evaluation_code_paths_at_commit(
    root: Path,
    commit: str,
    *,
    explicit_paths: frozenset[str] = ORIGINAL_EVALUATION_EXPLICIT_PATHS,
) -> tuple[str, ...]:
    raw = _git(
        root,
        "ls-tree",
        "-r",
        "--name-only",
        "-z",
        commit,
        "--",
        "src/token_prediction",
        *sorted(explicit_paths),
    )
    paths: list[str] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        try:
            relative = item.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise Stage4ReleaseError("Git returned a non-UTF-8 evaluation path") from exc
        paths.append(_relative(relative, description="historical evaluation code path"))
    result = tuple(sorted(set(paths)))
    if not explicit_paths <= set(result) or not any(
        path.startswith("src/token_prediction/") for path in result
    ):
        raise Stage4ReleaseError("historical evaluation code closure is incomplete")
    return result


def _evaluation_code_binding_at_commit(
    root: Path,
    commit: str,
) -> Mapping[str, object]:
    resolved_commit = _commit(commit, description="evaluation source commit")
    paths = _evaluation_code_paths_at_commit(root, resolved_commit)
    # This byte namespace is part of the frozen runner's historical hash
    # contract; it intentionally uses hyphens while the public policy ID uses
    # underscores.
    digest = hashlib.sha256(b"stage4-final-evaluation-code-tree-v1\0")
    for relative in paths:
        payload = _git(
            root,
            "show",
            f"{resolved_commit}:{relative}",
            maximum_bytes=MAX_BUNDLE_FILE_BYTES,
        )
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return {
        "policy_id": EVALUATION_CODE_POLICY_ID,
        "git_commit": resolved_commit,
        "code_tree_sha256": digest.hexdigest(),
        "paths": list(paths),
    }


def _historical_amended_code_binding_at_commit(
    root: Path,
    commit: str,
) -> Mapping[str, object]:
    resolved_commit = _commit(commit, description="historical amended code commit")
    paths = _evaluation_code_paths_at_commit(
        root,
        resolved_commit,
        explicit_paths=HISTORICAL_EXECUTED_EXPLICIT_PATHS,
    )
    digest = hashlib.sha256(b"stage4-final-evaluation-code-tree-v1\0")
    path_digest = hashlib.sha256(b"stage4-final-executed-paths-v1\0")
    for relative in paths:
        payload = _git(
            root,
            "show",
            f"{resolved_commit}:{relative}",
            maximum_bytes=MAX_BUNDLE_FILE_BYTES,
        )
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        path_digest.update(len(encoded).to_bytes(8, "big"))
        path_digest.update(encoded)
    return {
        "policy_id": HISTORICAL_AMENDED_CODE_POLICY_ID,
        "git_commit": resolved_commit,
        "code_tree_sha256": digest.hexdigest(),
        "path_count": len(paths),
        "path_projection_sha256": path_digest.hexdigest(),
        "paths": list(paths),
    }


def _historical_added_path_records(
    root: Path,
    commit: str,
) -> list[Mapping[str, str]]:
    resolved_commit = _commit(commit, description="historical added path commit")
    records: list[Mapping[str, str]] = []
    for relative in HISTORICAL_ADDED_EXPLICIT_PATHS:
        payload = _git(
            root,
            "show",
            f"{resolved_commit}:{relative}",
            maximum_bytes=MAX_BUNDLE_FILE_BYTES,
        )
        oid = (
            _git(root, "rev-parse", "--verify", f"{resolved_commit}:{relative}")
            .decode("ascii")
            .strip()
        )
        if (
            len(oid) not in {40, 64}
            or oid != oid.lower()
            or any(character not in "0123456789abcdef" for character in oid)
        ):
            raise Stage4ReleaseError("historical added path Git blob is invalid")
        records.append(
            {
                "path": relative,
                "git_blob_oid": oid,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return records


def _verify_historical_code_closure(
    root: Path,
    release: Mapping[str, Any],
    selection_lock: Mapping[str, Any],
    original_code: Mapping[str, object],
) -> Mapping[str, object]:
    amendment = release["historical_code_closure_amendments"][0]
    selection_commit = str(release["selection"]["commit"])
    selection_code_commit = str(
        selection_lock["selection_artifact"]["selection_code_commit"]
    )
    original_expected = {
        "policy_id": original_code["policy_id"],
        "git_commit": original_code["git_commit"],
        "code_tree_sha256": original_code["code_tree_sha256"],
        "path_count": len(original_code["paths"]),
    }
    amended = _historical_amended_code_binding_at_commit(root, selection_commit)
    amended_expected = {
        key: amended[key]
        for key in (
            "policy_id",
            "git_commit",
            "code_tree_sha256",
            "path_count",
            "path_projection_sha256",
        )
    }
    if (
        amendment["selection_code_commit"] != selection_code_commit
        or amendment["original_binding"] != original_expected
        or amendment["amended_binding"] != amended_expected
        or amendment["added_paths"]
        != _historical_added_path_records(root, selection_commit)
    ):
        raise Stage4ReleaseError(
            "historical final code closure amendment differs from Git blobs"
        )
    stable_paths = [
        path for path in amended["paths"] if path != DEFAULT_SELECTION_LOCK
    ]
    changed = _git(
        root,
        "diff",
        "--name-only",
        "-z",
        selection_code_commit,
        selection_commit,
        "--",
        *stable_paths,
    )
    if changed:
        raise Stage4ReleaseError(
            "runtime code changed between selection construction and final tag"
        )
    return amended


def _verify_tracked_bindings(
    root: Path,
    release: Mapping[str, Any],
    *,
    release_relative: str,
    require_git_clean: bool,
) -> tuple[Mapping[str, Any], Mapping[str, object]]:
    selection_binding = release["selection"]
    selection_relative = str(selection_binding["lock_path"])
    report_relative = str(release["report"]["path"])
    controls = (release_relative, selection_relative, report_relative)
    tracked = {
        item.decode("utf-8", errors="strict")
        for item in _git(root, "ls-files", "-z", "--", *controls).split(b"\0")
        if item
    }
    if tracked != set(controls):
        raise Stage4ReleaseError("Stage 4 release controls must be tracked")
    if require_git_clean and _git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--",
        *controls,
    ):
        raise Stage4ReleaseError("Stage 4 release controls must be clean at HEAD")
    final_release_tag = str(release["remote_controls"]["final_release_tag"])
    final_release_commit = (
        _git(
            root,
            "rev-parse",
            "--verify",
            f"refs/tags/{final_release_tag}^{{commit}}",
        )
        .decode("ascii")
        .strip()
    )
    _commit(final_release_commit, description="Stage 4 final release tag commit")
    for relative in controls:
        tagged_payload = _git(
            root,
            "show",
            f"{final_release_commit}:{relative}",
            maximum_bytes=MAX_BUNDLE_FILE_BYTES,
        )
        current_payload = _regular_file(
            _path(root, relative, description=f"Stage 4 release control {relative}"),
            maximum_bytes=MAX_BUNDLE_FILE_BYTES,
            description=f"Stage 4 release control {relative}",
        )
        if tagged_payload != current_payload:
            raise Stage4ReleaseError(
                "Stage 4 release controls differ from the immutable release tag"
            )

    selection_path = _path(
        root,
        selection_relative,
        description="Stage 4 selection lock",
    )
    selection_payload = _regular_file(
        selection_path,
        maximum_bytes=MAX_SELECTION_LOCK_BYTES,
        description="Stage 4 selection lock",
    )
    if hashlib.sha256(selection_payload).hexdigest() != selection_binding["lock_sha256"]:
        raise Stage4ReleaseError("Stage 4 selection lock SHA-256 differs")
    selection_lock = _load_json(
        selection_path,
        maximum_bytes=MAX_SELECTION_LOCK_BYTES,
        description="Stage 4 selection lock",
    )
    try:
        _validate_selection_lock_document(selection_lock)
    except (Stage4FinalError, TypeError, ValueError, RuntimeError) as exc:
        raise Stage4ReleaseError("Stage 4 selection lock schema is invalid") from exc

    selection_commit = _git(
        root,
        "rev-parse",
        "--verify",
        f"refs/tags/{SELECTION_TAG}^{{commit}}",
    ).decode("ascii").strip()
    if selection_commit != selection_binding["commit"]:
        raise Stage4ReleaseError("Stage 4 selection tag points to another commit")
    tagged_lock = _git(
        root,
        "show",
        f"{selection_commit}:{selection_relative}",
        maximum_bytes=MAX_SELECTION_LOCK_BYTES,
    )
    if tagged_lock != selection_payload:
        raise Stage4ReleaseError("current selection lock differs from its frozen tag bytes")

    if selection_lock["selection_tag"] != selection_binding["tag"]:
        raise Stage4ReleaseError("selection lock and release tag identities differ")
    lock_artifact = selection_lock["selection_artifact"]
    release_artifact = selection_binding["artifact"]
    for name in (
        "path",
        "artifact_id",
        "run_id",
        "selection_id",
        "selection_payload_sha256",
    ):
        if lock_artifact[name] != release_artifact[name]:
            raise Stage4ReleaseError("selection artifact release binding differs from lock")
    if selection_lock["source_artifacts"] != release["source_artifacts"]:
        raise Stage4ReleaseError("selection and release source inventories differ")

    for spec in SOURCE_ARTIFACTS:
        tagged = _git(
            root,
            "rev-parse",
            "--verify",
            f"refs/tags/{spec.source_tag}^{{commit}}",
        ).decode("ascii").strip()
        if tagged != spec.source_commit:
            raise Stage4ReleaseError(
                f"{spec.source_tag} does not point to {spec.key}'s source commit"
            )

    actual_code = _evaluation_code_binding_at_commit(root, selection_commit)
    compact_code = {
        key: actual_code[key]
        for key in ("policy_id", "git_commit", "code_tree_sha256")
    }
    if compact_code != release["evaluation_code_binding"]:
        raise Stage4ReleaseError("frozen evaluation code tree differs from release")
    _verify_historical_code_closure(
        root,
        release,
        selection_lock,
        actual_code,
    )

    report_path = _path(root, report_relative, description="Stage 4 final report")
    report_payload = _regular_file(
        report_path,
        maximum_bytes=MAX_REPORT_BYTES,
        description="Stage 4 final report",
    )
    if hashlib.sha256(report_payload).hexdigest() != release["report"]["sha256"]:
        raise Stage4ReleaseError("Stage 4 final report SHA-256 differs")
    return selection_lock, actual_code


def _bundle_tree_projection(directory: Path) -> tuple[str, int]:
    if _is_link_or_reparse(directory):
        raise Stage4ReleaseError("selected bundle root is linked or reparse-backed")
    try:
        root_status = directory.lstat()
    except OSError as exc:
        raise Stage4ReleaseError("selected bundle is missing") from exc
    if not stat.S_ISDIR(root_status.st_mode):
        raise Stage4ReleaseError("selected bundle root is not a directory")

    files: list[dict[str, str]] = []
    total_bytes = 0
    pending = [directory]
    while pending:
        current = pending.pop()
        try:
            with os.scandir(current) as entries:
                children = sorted(entries, key=lambda item: item.name)
        except OSError as exc:
            raise Stage4ReleaseError("selected bundle tree is unreadable") from exc
        for entry in children:
            path = Path(entry.path)
            if _is_link_or_reparse(path):
                raise Stage4ReleaseError("selected bundle contains a link or reparse point")
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise Stage4ReleaseError("selected bundle member is unreadable") from exc
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(path)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise Stage4ReleaseError("selected bundle contains a special file")
            if metadata.st_size <= 0 or metadata.st_size > MAX_BUNDLE_FILE_BYTES:
                raise Stage4ReleaseError("selected bundle file has an invalid size")
            total_bytes += metadata.st_size
            if (
                total_bytes > MAX_BUNDLE_TOTAL_BYTES
                or len(files) >= MAX_BUNDLE_FILE_COUNT
            ):
                raise Stage4ReleaseError("selected bundle exceeds its safe bounds")
            relative = path.relative_to(directory).as_posix()
            if (
                not relative
                or "\\" in relative
                or any(part in {"", ".", ".."} for part in PurePosixPath(relative).parts)
            ):
                raise Stage4ReleaseError("selected bundle contains an unsafe path")
            files.append({"path": relative, "sha256": sha256_file(path)})
    if not files:
        raise Stage4ReleaseError("selected bundle is empty")
    files.sort(key=lambda item: item["path"])
    return semantic_sha256(files), len(files)


def _verify_json_digest(
    root: Path,
    raw_path: object,
    expected_sha256: object,
    *,
    description: str,
) -> Mapping[str, Any]:
    path = _path(root, raw_path, description=description)
    payload = _regular_file(
        path,
        maximum_bytes=MAX_MEMBER_JSON_BYTES,
        description=description,
    )
    if hashlib.sha256(payload).hexdigest() != _sha256(
        expected_sha256,
        description=description,
    ):
        raise Stage4ReleaseError(f"{description} SHA-256 differs")
    return _parse_json_payload(payload, description=description)


def _verify_member_shape(
    member: object,
    *,
    cell: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str]:
    if not isinstance(member, Mapping):
        raise Stage4ReleaseError("selected member must be an object")
    kind = member.get("bundle_kind")
    keys = _EMPIRICAL_MEMBER_KEYS if kind == "empirical_json" else _POINT_MEMBER_KEYS
    _exact(member, keys, description="selected ensemble member")
    if kind not in {"empirical_json", "lightgbm", "lifecycle"}:
        raise Stage4ReleaseError("selected member bundle kind is invalid")
    digest = _sha256(member["member_sha256"], description="selected member")
    payload = dict(member)
    payload.pop("member_sha256")
    if semantic_sha256(payload) != digest:
        raise Stage4ReleaseError("selected member checksum differs")
    if (
        member["origin"]
        != (
            "selection_artifact"
            if kind == "empirical_json"
            else cell["selected_artifact_key"]
        )
        or member["split_plan_id"] is None
    ):
        raise Stage4ReleaseError("selected member origin or split identity is invalid")
    _sha256(member["split_plan_id"], description="selected member split plan")
    _integer(member["split_seed"], description="selected member split seed")
    fold = _integer(member["fold"], description="selected member fold")
    if fold >= 5:
        raise Stage4ReleaseError("selected member fold is outside the frozen plan")
    return member, str(kind)


def _verify_selected_members(
    root: Path,
    selection_root: Path,
    selection: Mapping[str, Any],
    release: Mapping[str, Any],
) -> tuple[int, int, int]:
    source_prefixes = {
        item["key"]: f"{item['path']}/" for item in release["source_artifacts"]
    }
    member_hashes: set[str] = set()
    primary_paths: set[str] = set()
    loaded_bundle_count = 0
    empirical_count = 0
    for cell in selection["cells"]:
        members = cell["members"]
        if len(members) != EXPECTED_MEMBERS_PER_CELL:
            raise Stage4ReleaseError("selected cell does not contain 15 members")
        for raw_member in members:
            member, kind = _verify_member_shape(raw_member, cell=cell)
            member_hash = str(member["member_sha256"])
            if member_hash in member_hashes:
                raise Stage4ReleaseError("selected member checksum is reused")
            member_hashes.add(member_hash)
            if kind == "empirical_json":
                relative = _relative(
                    member["state_path"],
                    description="selected empirical state path",
                )
                if relative in primary_paths:
                    raise Stage4ReleaseError("selected empirical state path is reused")
                primary_paths.add(relative)
                state_path = _path(
                    selection_root,
                    relative,
                    description="selected empirical state",
                )
                state_payload = _regular_file(
                    state_path,
                    maximum_bytes=MAX_MEMBER_JSON_BYTES,
                    description="selected empirical state",
                )
                if hashlib.sha256(state_payload).hexdigest() != _sha256(
                    member["state_sha256"],
                    description="selected empirical state",
                ):
                    raise Stage4ReleaseError("selected empirical state SHA-256 differs")
                state_document = _parse_json_payload(
                    state_payload,
                    description="selected empirical state",
                )
                try:
                    state = EmpiricalFoldState.from_dict(state_document)
                except (TypeError, ValueError, OSError) as exc:
                    raise Stage4ReleaseError(
                        "selected empirical state cannot be loaded safely"
                    ) from exc
                if (
                    state.split_seed != member["split_seed"]
                    or state.fold != member["fold"]
                    or state.split_plan_id != member["split_plan_id"]
                    or state.target.value != cell["target"]
                ):
                    raise Stage4ReleaseError(
                        "selected empirical state scope differs from selection"
                    )
                empirical_count += 1
                continue

            prefix = source_prefixes.get(str(member["origin"]))
            if prefix is None:
                raise Stage4ReleaseError("selected bundle origin is not frozen")
            bundle_relative = _relative(
                member["bundle_path"],
                description="selected bundle path",
            )
            if (
                not bundle_relative.startswith(prefix)
                or bundle_relative in primary_paths
            ):
                raise Stage4ReleaseError("selected bundle path is reused or outside origin")
            primary_paths.add(bundle_relative)
            bundle_root = _path(
                root,
                bundle_relative,
                description="selected bundle",
            )
            tree_hash, file_count = _bundle_tree_projection(bundle_root)
            if (
                tree_hash
                != _sha256(
                    member["bundle_tree_sha256"],
                    description="selected bundle tree",
                )
                or file_count
                != _integer(
                    member["bundle_file_count"],
                    description="selected bundle file count",
                    minimum=1,
                )
            ):
                raise Stage4ReleaseError("selected bundle tree differs from selection")

            for role in ("calibrator", "provenance"):
                auxiliary = _relative(
                    member[f"{role}_path"],
                    description=f"selected {role} path",
                )
                if not auxiliary.startswith(prefix):
                    raise Stage4ReleaseError(
                        f"selected {role} path is outside its frozen origin"
                    )
            calibrator = _verify_json_digest(
                root,
                member["calibrator_path"],
                member["calibrator_sha256"],
                description="selected calibrator",
            )
            try:
                loaded_calibrator = FittedExpansionCalibrator.from_dict(calibrator)
            except (TypeError, ValueError) as exc:
                raise Stage4ReleaseError("selected calibrator cannot be loaded") from exc
            if (
                loaded_calibrator.calibrator_id != cell["calibrator_id"]
                or loaded_calibrator.interval_alpha != float(cell["alpha"])
            ):
                raise Stage4ReleaseError("selected calibrator scope differs from cell")
            provenance = _verify_json_digest(
                root,
                member["provenance_path"],
                member["provenance_sha256"],
                description="selected provenance",
            )
            expected_provenance = {
                "candidate_id": cell["candidate_id"],
                "candidate_hash": cell["candidate_hash"],
                "condition_id": cell["condition_id"],
                "position": cell["position"],
                "target": cell["target"],
                "split_plan_id": member["split_plan_id"],
                "calibrator_id": cell["calibrator_id"],
            }
            if any(
                provenance.get(key) != value
                for key, value in expected_provenance.items()
            ):
                raise Stage4ReleaseError("selected provenance differs from cell")
            provenance_fold = provenance.get(
                "fold",
                provenance.get("outer_fold"),
            )
            if provenance_fold != member["fold"]:
                raise Stage4ReleaseError("selected provenance fold differs")

            try:
                if kind == "lightgbm":
                    loaded = load_lightgbm_bundle(bundle_root)
                    if (
                        loaded.target.value != cell["target"]
                        or loaded.position.value != cell["position"]
                        or loaded.allowed_condition_ids != (cell["condition_id"],)
                    ):
                        raise Stage4ReleaseError(
                            "loaded LightGBM bundle scope differs from selection"
                        )
                else:
                    lifecycle = load_lifecycle_bundle(bundle_root)
                    manifest = lifecycle.manifest
                    if (
                        manifest["candidate_id"] != cell["candidate_id"]
                        or manifest["candidate_hash"] != cell["candidate_hash"]
                        or manifest["target"] != cell["target"]
                        or manifest["condition_id"] != cell["condition_id"]
                        or manifest["outer_fold"] != member["fold"]
                        or manifest["split_plan_id"] != member["split_plan_id"]
                    ):
                        raise Stage4ReleaseError(
                            "loaded lifecycle bundle scope differs from selection"
                        )
            except Stage4ReleaseError:
                raise
            except (OSError, TypeError, ValueError, RuntimeError) as exc:
                raise Stage4ReleaseError(
                    f"selected {kind} bundle cannot be loaded safely"
                ) from exc
            if _bundle_tree_projection(bundle_root) != (tree_hash, file_count):
                raise Stage4ReleaseError("selected bundle changed while being loaded")
            loaded_bundle_count += 1
    if (
        len(member_hashes) != EXPECTED_MEMBER_COUNT
        or loaded_bundle_count + empirical_count != EXPECTED_MEMBER_COUNT
    ):
        raise Stage4ReleaseError("selected member cardinalities do not close")
    return len(member_hashes), loaded_bundle_count, empirical_count


def _verify_selection_code_binding_from_git(
    root: Path,
    code: Mapping[str, Any],
) -> None:
    _exact(
        code,
        {"policy_id", "git_commit", "code_tree_sha256", "paths"},
        description="selection code binding",
    )
    commit = _commit(code["git_commit"], description="selection code commit")
    required = frozenset(
        {
            "scripts/prepare_stage4_selection.py",
            "scripts/run_stage2_experiments.py",
            "scripts/run_stage3_experiments.py",
            "scripts/run_stage4_experiments.py",
            "configs/data_foundation_prediction_baseline.json",
        }
    )
    raw = _git(
        root,
        "ls-tree",
        "-r",
        "--name-only",
        "-z",
        commit,
        "--",
        "src/token_prediction",
        *sorted(required),
    )
    expected_paths = tuple(
        sorted(
            _relative(
                item.decode("utf-8", errors="strict"),
                description="selection historical code path",
            )
            for item in raw.split(b"\0")
            if item
        )
    )
    if (
        code["policy_id"] != SELECTION_CODE_POLICY_ID
        or code["paths"] != list(expected_paths)
        or not required <= set(expected_paths)
        or not any(path.startswith("src/token_prediction/") for path in expected_paths)
    ):
        raise Stage4ReleaseError("selection code path closure differs from Git")
    digest = hashlib.sha256(f"{SELECTION_CODE_POLICY_ID}\0".encode("ascii"))
    for relative in expected_paths:
        payload = _git(
            root,
            "show",
            f"{commit}:{relative}",
            maximum_bytes=MAX_BUNDLE_FILE_BYTES,
        )
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    if digest.hexdigest() != code["code_tree_sha256"]:
        raise Stage4ReleaseError("selection code tree differs from committed Git blobs")


def _verify_selection_artifact(
    root: Path,
    release: Mapping[str, Any],
    selection_lock: Mapping[str, Any],
) -> tuple[Path, Mapping[str, Any], int, int, int]:
    binding = release["selection"]["artifact"]
    selection_root = _path(
        root,
        binding["path"],
        description="Stage 4 selection artifact",
    )
    try:
        manifest = verify_artifact(selection_root)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise Stage4ReleaseError("Stage 4 selection artifact is invalid") from exc
    if (
        manifest.stage_name != "stage4_frozen_selection"
        or manifest.schema_version != 1
        or manifest.artifact_id != binding["artifact_id"]
        or len(manifest.files) != binding["manifest_file_count"]
        or manifest.metadata.get("run_id") != binding["run_id"]
        or manifest.metadata.get("selection_id") != binding["selection_id"]
        or manifest.metadata.get("selection_payload_sha256")
        != binding["selection_payload_sha256"]
        or manifest.metadata.get("final_holdout_evaluated") is not False
    ):
        raise Stage4ReleaseError("Stage 4 selection manifest differs from release")
    selection = _load_json(
        selection_root / "selection.json",
        maximum_bytes=MAX_SELECTION_JSON_BYTES,
        description="Stage 4 selection document",
    )
    try:
        payload_hash = verify_selection_document(selection)
    except (Stage4SelectionError, TypeError, ValueError, RuntimeError) as exc:
        raise Stage4ReleaseError("Stage 4 selection document is invalid") from exc
    if (
        payload_hash != binding["selection_payload_sha256"]
        or selection["selection_id"] != binding["selection_id"]
        or selection["summary"].get("cell_count") != EXPECTED_CELL_COUNT
        or selection["summary"].get("ensemble_member_count") != EXPECTED_MEMBER_COUNT
    ):
        raise Stage4ReleaseError("Stage 4 selection document differs from release")
    code = selection.get("code_binding")
    locked = selection_lock["selection_artifact"]
    if (
        not isinstance(code, Mapping)
        or code.get("git_commit") != locked["selection_code_commit"]
        or code.get("code_tree_sha256") != locked["selection_code_tree_sha256"]
    ):
        raise Stage4ReleaseError("selection artifact code binding differs from lock")
    _verify_selection_code_binding_from_git(root, code)
    specs = {item.key: asdict(item) for item in SOURCE_ARTIFACTS}
    selected_sources = selection.get("source_artifacts")
    if not isinstance(selected_sources, list) or len(selected_sources) != len(specs):
        raise Stage4ReleaseError("selection artifact source inventory is invalid")
    for item in selected_sources:
        if not isinstance(item, Mapping) or item.get("key") not in specs:
            raise Stage4ReleaseError("selection artifact source entry is invalid")
        spec = specs[str(item["key"])]
        if any(item.get(key) != value for key, value in spec.items()):
            raise Stage4ReleaseError("selection artifact source binding differs")
    verified, loaded, empirical = _verify_selected_members(
        root,
        selection_root,
        selection,
        release,
    )
    return selection_root, selection, verified, loaded, empirical


def _verify_final_cell_bindings(
    selection: Mapping[str, Any],
    final_results: Mapping[str, Any],
) -> None:
    selected_cells = {str(item["cell_id"]): item for item in selection["cells"]}
    final_cells = {str(item["cell_id"]): item for item in final_results["cells"]}
    if set(selected_cells) != set(final_cells):
        raise Stage4ReleaseError("final cells differ from frozen selection")
    prediction_count = 0
    for cell_id, selected in selected_cells.items():
        final = final_cells[cell_id]
        for name in (
            "source_name",
            "source_id",
            "condition_id",
            "position",
            "target",
            "candidate_id",
            "candidate_hash",
            "calibrator_id",
            "alpha",
        ):
            if final.get(name) != selected.get(name):
                raise Stage4ReleaseError("final cell differs from selected cell")
        execution = _exact(
            final.get("model_execution"),
            {
                "ensemble_policy_id",
                "member_count",
                "member_projection_sha256",
                "execution_mode",
                "refit",
                "calibration_application_count",
            },
            description="final cell model execution",
        )
        expected_projection = semantic_sha256(
            [member["member_sha256"] for member in selected["members"]]
        )
        expected_mode = (
            "strict_loaded_calibrated_full_trajectory_only"
            if selected["selected_artifact_key"] == "stage3_spend_openhands"
            else "strict_loaded_bundle_only"
        )
        if execution != {
            "ensemble_policy_id": SELECTION_ENSEMBLE_POLICY_ID,
            "member_count": EXPECTED_MEMBERS_PER_CELL,
            "member_projection_sha256": expected_projection,
            "execution_mode": expected_mode,
            "refit": False,
            "calibration_application_count": 1,
        }:
            raise Stage4ReleaseError("final cell model execution protocol is invalid")
        prediction_count += _integer(
            final.get("prediction_count"),
            description="final cell prediction count",
            minimum=1,
        )
    if prediction_count != EXPECTED_PREDICTION_COUNT:
        raise Stage4ReleaseError("final cell prediction cardinality is invalid")


def _verify_final_artifact(
    root: Path,
    release: Mapping[str, Any],
    selection_lock: Mapping[str, Any],
    selection: Mapping[str, Any],
    evaluation_code: Mapping[str, object],
) -> tuple[Mapping[str, Any], object]:
    binding = release["final_artifact"]
    final_root = _path(
        root,
        binding["path"],
        description="Stage 4 final artifact",
    )
    try:
        manifest = verify_artifact(final_root)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise Stage4ReleaseError("Stage 4 final artifact is invalid") from exc
    if (
        manifest.stage_name != FINAL_STAGE_NAME
        or manifest.schema_version != 1
        or manifest.artifact_id != binding["artifact_id"]
        or len(manifest.files) != binding["manifest_file_count"]
        or set(manifest.files) != {"results.json", "selection-lock.json"}
        or manifest.metadata.get("run_id") != binding["run_id"]
        or manifest.metadata.get("selection_id") != binding["selection_id"]
        or manifest.metadata.get("results_payload_sha256")
        != binding["results_payload_sha256"]
        or manifest.metadata.get("final_holdout_evaluated") is not True
        or manifest.metadata.get("evaluation_count") != 1
    ):
        raise Stage4ReleaseError("Stage 4 final manifest differs from release")
    elapsed = manifest.metadata.get("elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) <= 0
    ):
        raise Stage4ReleaseError("Stage 4 final elapsed time is invalid")

    results = _load_json(
        final_root / "results.json",
        maximum_bytes=MAX_FINAL_RESULTS_BYTES,
        description="Stage 4 final results",
    )
    try:
        payload_hash = verify_final_results_document(results)
    except (Stage4FinalError, TypeError, ValueError, RuntimeError) as exc:
        raise Stage4ReleaseError("Stage 4 final results are invalid") from exc
    if (
        payload_hash != binding["results_payload_sha256"]
        or results["run_id"] != binding["run_id"]
        or results["summary"]
        != {
            "source_count": EXPECTED_SOURCE_COUNT,
            "cell_count": EXPECTED_CELL_COUNT,
            "ensemble_member_count": EXPECTED_MEMBER_COUNT,
            "prediction_count": EXPECTED_PREDICTION_COUNT,
        }
        or results["final_holdout"]
        != {
            "evaluated": True,
            "evaluation_count": 1,
            "prediction_count": EXPECTED_PREDICTION_COUNT,
            "target_values_used_for_fit": False,
            "target_values_used_for_calibration": False,
            "target_values_used_for_scoring": True,
            "model_selection_after_open": False,
        }
    ):
        raise Stage4ReleaseError("Stage 4 final results differ from release protocol")

    expected_selection = {
        "selection_id": binding["selection_id"],
        "selection_artifact_id": release["selection"]["artifact"]["artifact_id"],
        "selection_payload_sha256": release["selection"]["artifact"][
            "selection_payload_sha256"
        ],
        "selection_lock_path": release["selection"]["lock_path"],
        "selection_lock_sha256": release["selection"]["lock_sha256"],
        "selection_tag": release["selection"]["tag"],
        "selection_commit": release["selection"]["commit"],
    }
    if results["selection"] != expected_selection:
        raise Stage4ReleaseError("final results selection binding is invalid")
    if results["evaluation_code_binding"] != evaluation_code:
        raise Stage4ReleaseError("final results evaluation code binding differs")
    _verify_final_cell_bindings(selection, results)

    copied_lock = _load_json(
        final_root / "selection-lock.json",
        maximum_bytes=MAX_SELECTION_LOCK_BYTES,
        description="final artifact selection lock",
    )
    if copied_lock != selection_lock:
        raise Stage4ReleaseError("final artifact embeds another selection lock")

    run_semantic = {
        "run_policy_id": FINAL_RUN_POLICY_ID,
        "selection_id": binding["selection_id"],
        "selection_artifact_id": release["selection"]["artifact"]["artifact_id"],
        "selection_lock_sha256": release["selection"]["lock_sha256"],
        "selection_commit": release["selection"]["commit"],
        "evaluation_code_binding": evaluation_code,
    }
    if manifest.metadata.get("run_semantic") != run_semantic:
        raise Stage4ReleaseError("final artifact run semantics differ from release")
    return results, manifest


def _verify_metadata_amendment(
    root: Path,
    release: Mapping[str, Any],
    final_results: Mapping[str, Any],
) -> None:
    amendment = release["metadata_amendments"][0]
    matching_specs = [
        spec for spec in SOURCE_ARTIFACTS if spec.key == amendment["source_artifact_key"]
    ]
    if len(matching_specs) != 1:
        raise Stage4ReleaseError("metadata amendment source binding is ambiguous")
    spec = matching_specs[0]
    source_root = _path(
        root,
        spec.path,
        description="metadata amendment source artifact",
    )
    try:
        manifest = verify_artifact(source_root)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise Stage4ReleaseError(
            "metadata amendment source artifact is invalid"
        ) from exc
    if (
        manifest.stage_name != "stage4_development_source"
        or manifest.schema_version != 1
        or manifest.artifact_id != spec.artifact_id
        or manifest.metadata.get("run_id") != spec.run_id
        or manifest.metadata.get("results_payload_sha256")
        != spec.results_payload_sha256
    ):
        raise Stage4ReleaseError(
            "metadata amendment source manifest differs from frozen inventory"
        )
    source_results = _load_json(
        source_root / "results.json",
        maximum_bytes=MAX_SOURCE_RESULTS_BYTES,
        description="metadata amendment source results",
    )
    try:
        payload_hash = verify_stage4_results_document(source_results)
    except (Stage4ExperimentError, TypeError, ValueError, RuntimeError) as exc:
        raise Stage4ReleaseError(
            "metadata amendment source results are invalid"
        ) from exc
    if (
        payload_hash != spec.results_payload_sha256
        or source_results.get("run_id") != spec.run_id
    ):
        raise Stage4ReleaseError(
            "metadata amendment source results differ from frozen inventory"
        )
    _verify_metadata_amendment_documents(
        source_results,
        final_results,
        amendment=amendment,
    )


def _verify_metadata_amendment_documents(
    source_results: Mapping[str, Any],
    final_results: Mapping[str, Any],
    *,
    amendment: Mapping[str, Any],
) -> None:
    development_protocol = source_results.get("development_protocol")
    if not isinstance(development_protocol, Mapping):
        raise Stage4ReleaseError(
            "metadata amendment lacks authoritative development protocol"
        )
    permanent_holdout = development_protocol.get("permanent_holdout")
    if not isinstance(permanent_holdout, Mapping):
        raise Stage4ReleaseError(
            "metadata amendment lacks authoritative permanent holdout"
        )
    assignments = permanent_holdout.get("assignments")
    if (
        not isinstance(assignments, list)
        or not assignments
        or not all(isinstance(item, Mapping) for item in assignments)
    ):
        raise Stage4ReleaseError(
            "metadata amendment authoritative assignments are invalid"
        )
    pseudonyms = [item.get("task_pseudonym") for item in assignments]
    if (
        any(
            not isinstance(pseudonym, str)
            or len(pseudonym) != 64
            or any(character not in "0123456789abcdef" for character in pseudonym)
            for pseudonym in pseudonyms
        )
        or len(set(pseudonyms)) != len(pseudonyms)
        or any(
            item.get("cohort") not in {"development", "final_holdout"}
            for item in assignments
        )
    ):
        raise Stage4ReleaseError(
            "metadata amendment authoritative assignments are malformed"
        )
    authoritative_count = sum(
        item["cohort"] == "final_holdout" for item in assignments
    )
    if authoritative_count != amendment["authoritative_value"]:
        raise Stage4ReleaseError(
            "metadata amendment authoritative task count differs"
        )

    dataset_matches = [
        item
        for item in final_results["datasets"]
        if item.get("source_name") == "bagen_swebench"
    ]
    if (
        len(dataset_matches) != 1
        or dataset_matches[0].get("task_count") != amendment["artifact_value"]
    ):
        raise Stage4ReleaseError("metadata amendment artifact field differs")
    source_cells = [
        cell
        for cell in final_results["cells"]
        if cell.get("source_name") == "bagen_swebench"
    ]
    task_counts: list[object] = []
    for cell in source_cells:
        final_dataset = cell.get("final_dataset")
        metrics = cell.get("metrics")
        if not isinstance(final_dataset, Mapping) or not isinstance(metrics, Mapping):
            raise Stage4ReleaseError(
                "metadata amendment source cell lacks scored task metadata"
            )
        scored_tasks = metrics.get("n_tasks")
        if final_dataset.get("task_count") != scored_tasks:
            raise Stage4ReleaseError(
                "metadata amendment source cell task counts disagree"
            )
        task_counts.append(scored_tasks)
    if (
        len(task_counts) != len(source_cells)
        or not task_counts
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for count in task_counts
        )
        or max(task_counts) != amendment["authoritative_value"]
        or amendment["authoritative_value"] not in task_counts
    ):
        raise Stage4ReleaseError(
            "metadata amendment cell coverage does not reach authoritative task count"
        )


def _verify_ledger(
    root: Path,
    release: Mapping[str, Any],
    final_results: Mapping[str, Any],
) -> None:
    ledger_binding = release["ledger"]
    ledger = _load_json(
        _path(root, ledger_binding["path"], description="Stage 4 final ledger"),
        maximum_bytes=MAX_LEDGER_BYTES,
        description="Stage 4 final ledger",
    )
    _exact(
        ledger,
        {
            "ledger_schema_version",
            "run_policy_id",
            "run_id",
            "selection_id",
            "status",
            "completed_cell_ids",
            "final_artifact_id",
        },
        description="Stage 4 final ledger",
    )
    expected_cells = sorted(str(cell["cell_id"]) for cell in final_results["cells"])
    if (
        ledger["ledger_schema_version"] != ledger_binding["schema_version"]
        or ledger["run_policy_id"] != FINAL_RUN_POLICY_ID
        or ledger["run_id"] != release["final_artifact"]["run_id"]
        or ledger["selection_id"] != release["final_artifact"]["selection_id"]
        or ledger["status"] != "published"
        or ledger["completed_cell_ids"] != expected_cells
        or len(set(expected_cells)) != EXPECTED_CELL_COUNT
        or ledger["final_artifact_id"] != release["final_artifact"]["artifact_id"]
    ):
        raise Stage4ReleaseError("Stage 4 final ledger is not a complete publication")


def _verify_canonical_final_state(
    root: Path,
    release: Mapping[str, Any],
) -> None:
    final_binding = release["final_artifact"]
    final_root = _path(
        root,
        "workspace/stage4/final",
        description="canonical final root",
    )
    checkpoint_root = _path(
        root,
        "workspace/stage4/final-checkpoints",
        description="canonical final checkpoint root",
    )
    expected_final_name = PurePosixPath(str(final_binding["path"])).name
    expected_run_name = str(final_binding["run_id"])
    for directory, expected, description in (
        (final_root, {expected_final_name}, "canonical final root"),
        (checkpoint_root, {expected_run_name}, "canonical final checkpoint root"),
    ):
        if _is_link_or_reparse(directory) or not directory.is_dir():
            raise Stage4ReleaseError(f"{description} is unsafe or missing")
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise Stage4ReleaseError(f"{description} cannot be enumerated") from exc
        names = {entry.name for entry in entries}
        if names != expected or any(
            entry.is_symlink() or not entry.is_dir(follow_symlinks=False)
            for entry in entries
        ):
            raise Stage4ReleaseError(
                f"{description} contains another final evaluation identity"
            )
    run_root = checkpoint_root / expected_run_name
    cells_root = run_root / "cells"
    if (
        _is_link_or_reparse(run_root)
        or _is_link_or_reparse(cells_root)
        or not cells_root.is_dir()
    ):
        raise Stage4ReleaseError("canonical final checkpoint tree is unsafe")
    run_entries = {entry.name for entry in os.scandir(run_root)}
    if run_entries != {"cells", "ledger.json"}:
        raise Stage4ReleaseError("canonical final checkpoint run has extra members")
    cell_entries = list(os.scandir(cells_root))
    if (
        len(cell_entries) != EXPECTED_CELL_COUNT
        or any(
            entry.is_symlink()
            or not entry.is_file(follow_symlinks=False)
            or not entry.name.endswith(".json")
            for entry in cell_entries
        )
    ):
        raise Stage4ReleaseError("canonical final checkpoint cell set is incomplete")


def verify_stage4_release(
    repository_root: str | Path,
    *,
    release_lock: str = DEFAULT_RELEASE_LOCK,
    tracked_only: bool = False,
    require_git_clean: bool = True,
) -> Stage4ReleaseVerification:
    supplied_root = Path(repository_root)
    if _is_link_or_reparse(supplied_root):
        raise Stage4ReleaseError("repository root must not be linked or reparse-backed")
    try:
        root = supplied_root.resolve(strict=True)
    except OSError as exc:
        raise Stage4ReleaseError("repository root is missing") from exc
    if not root.is_dir():
        raise Stage4ReleaseError("repository root is not a directory")
    release_relative = _relative(release_lock, description="Stage 4 release lock path")
    release = _load_json(
        _path(root, release_relative, description="Stage 4 release lock"),
        maximum_bytes=MAX_RELEASE_JSON_BYTES,
        description="Stage 4 release lock",
    )
    _validate_release_document(release)
    selection_lock, evaluation_code = _verify_tracked_bindings(
        root,
        release,
        release_relative=release_relative,
        require_git_clean=require_git_clean,
    )
    if tracked_only:
        return Stage4ReleaseVerification(
            lock_path=release_relative,
            report_path=str(release["report"]["path"]),
            selection_commit=str(release["selection"]["commit"]),
            evaluation_code_tree_sha256=str(
                release["evaluation_code_binding"]["code_tree_sha256"]
            ),
            locked_cell_count=EXPECTED_CELL_COUNT,
            locked_member_count=EXPECTED_MEMBER_COUNT,
            locked_prediction_count=EXPECTED_PREDICTION_COUNT,
            verified_artifact_count=0,
            verified_member_count=0,
            independently_loaded_bundle_count=0,
            verified_empirical_state_count=0,
            metadata_amendment_count=1,
            final_holdout_evaluation_count=1,
            source_data_replayed=False,
        )

    _selection_root, selection, member_count, bundle_count, empirical_count = (
        _verify_selection_artifact(root, release, selection_lock)
    )
    final_results, _manifest = _verify_final_artifact(
        root,
        release,
        selection_lock,
        selection,
        evaluation_code,
    )
    _verify_metadata_amendment(root, release, final_results)
    _verify_ledger(root, release, final_results)
    _verify_canonical_final_state(root, release)
    return Stage4ReleaseVerification(
        lock_path=release_relative,
        report_path=str(release["report"]["path"]),
        selection_commit=str(release["selection"]["commit"]),
        evaluation_code_tree_sha256=str(
            release["evaluation_code_binding"]["code_tree_sha256"]
        ),
        locked_cell_count=EXPECTED_CELL_COUNT,
        locked_member_count=EXPECTED_MEMBER_COUNT,
        locked_prediction_count=EXPECTED_PREDICTION_COUNT,
        verified_artifact_count=3,
        verified_member_count=member_count,
        independently_loaded_bundle_count=bundle_count,
        verified_empirical_state_count=empirical_count,
        metadata_amendment_count=1,
        final_holdout_evaluation_count=1,
        source_data_replayed=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the frozen Stage 4 selection and final release."
    )
    parser.add_argument(
        "--repository-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument("--release-lock", default=DEFAULT_RELEASE_LOCK)
    parser.add_argument(
        "--tracked-only",
        action="store_true",
        help="Verify tracked locks, report, tags, and historical code only.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow tracked release control files to differ from HEAD.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = verify_stage4_release(
            args.repository_root,
            release_lock=args.release_lock,
            tracked_only=args.tracked_only,
            require_git_clean=not args.allow_dirty,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Stage 4 release verification failed: {exc}") from exc
    print(json.dumps(asdict(summary), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
