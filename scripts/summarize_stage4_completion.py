"""Summarize Stage 4 completion experiments without opening the final holdout.

The command accepts exactly four Stage 4 development artifact directories by
default.  The sole alternate input is the formal
``configs/stage4_completion_release.json`` lock, whose exact v2 schema binds
safe repository-relative development artifact paths.

Only ``manifest.json``, ``results.json``, and ``_SUCCESS`` are read from each
development artifact.  Fold artifacts, source data, final artifacts, and final
labels are deliberately outside this tool's input surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

from token_prediction.lineage import verify_artifact


ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_RUNS_ROOT = ROOT / "workspace" / "stage4" / "runs"
DIAGNOSTICS_ROOT = ROOT / "workspace" / "stage4" / "completion_diagnostics"
DEVELOPMENT_STAGE_NAME = "stage4_development_source"
RESULTS_STAGE_NAME = DEVELOPMENT_STAGE_NAME
SUMMARY_SCHEMA_VERSION = 4
SUMMARY_POLICY_ID = "stage4_completion_development_only_summary_v3"
FINAL_HOLDOUT_POLICY_ID = (
    "summary_reader_never_opens_final_holdout_or_source_labels_v2"
)
REPLACEMENT_RULE_ID = (
    "seven_condition_all_seed_task_cluster_bootstrap_interval_score_v3"
)
SEED_POLICY_BOOTSTRAP_ITERATIONS = 10_000
SEED_POLICY_BOOTSTRAP_SEED_DERIVATION_ID = (
    "stage4-seed-policy-interval-and-task-coverage-bootstrap-v3"
)
EXPECTED_SEED_POLICY_CONDITION_COUNT = 7
SEED_POLICY_COVERAGE_TOLERANCE = 0.02
SEED_POLICY_UPPER_TAIL_TOLERANCE = 0.02
SEED_POLICY_POSITION = "task_update"
SEED_POLICY_TARGET = "task_provider_accounted_remaining_tokens"
SEED_POLICY_CALIBRATOR_ID = "task_max_conformal"
SEED_POLICY_ALPHA = 0.1
SEED_POLICY_EXPERIMENT_AXIS = None
SEED_POLICY_CANDIDATE_AXIS = "seed_policy"
COHORT_PROJECTION_ID = "stage4_prediction_cohort_projection_v1"
TASK_METRIC_POLICY_ID = "stage4_task_pseudonym_v1"
DEVELOPMENT_TASK_PSEUDONYM_POLICY_ID = "sha256_development_task_pseudonym_v1"
EXPECTED_SEED_POLICY_CELLS = frozenset(
    {
        (
            "bagen_sokoban",
            "bagen_sokoban_dialogues_v1",
            "condition:effa60eb1d4380d124bf",
            SEED_POLICY_EXPERIMENT_AXIS,
            SEED_POLICY_POSITION,
            SEED_POLICY_TARGET,
            SEED_POLICY_CALIBRATOR_ID,
            SEED_POLICY_ALPHA,
        ),
        *(
            (
                "bagen_swebench",
                "bagen_swebench_traj_v2",
                condition_id,
                SEED_POLICY_EXPERIMENT_AXIS,
                SEED_POLICY_POSITION,
                SEED_POLICY_TARGET,
                SEED_POLICY_CALIBRATOR_ID,
                SEED_POLICY_ALPHA,
            )
            for condition_id in (
                "condition:54cb50fce273f0aa2d74",
                "condition:949ac3b7a342718cd505",
                "condition:d94078c05d91b0d58aee",
                "condition:dce86ced00dc11c77205",
                "condition:f95ae2a5e11682f6b7fc",
            )
        ),
        (
            "spend_openhands",
            "openhands_archive_trajectory_v3",
            "condition:b407e0d1ec34f386ebc4",
            SEED_POLICY_EXPERIMENT_AXIS,
            SEED_POLICY_POSITION,
            SEED_POLICY_TARGET,
            SEED_POLICY_CALIBRATOR_ID,
            SEED_POLICY_ALPHA,
        ),
    }
)
RAW_SEED_CANDIDATE_ID = "cross_position_deduct_raw_repaired_oof_seed"
POINT_ONLY_SEED_CANDIDATE_ID = "cross_position_deduct_point_only_oof_seed"
MLP_CANDIDATE_ID = "mlp_history"
LIGHTGBM_CANDIDATE_ID = "lightgbm_history"
RAW_SEED_POLICY_ID = "inner_oof_uncalibrated_repaired_quantile_mean_v1"
POINT_ONLY_SEED_POLICY_ID = (
    "inner_oof_uncalibrated_repaired_point_only_mean_v1"
)
FROZEN_SPLIT_SEEDS = (20260719, 20260720, 20260721)
MAX_RESULTS_BYTES = 128 * 1024 * 1024
MAX_METADATA_BYTES = 8 * 1024 * 1024
COMPLETION_DIAGNOSTICS_ARTIFACT_SCHEMA_VERSION = 2
DIAGNOSTICS_REQUIRED_ROOT_FILES = frozenset(
    {"results.json", "manifest.json", "_SUCCESS"}
)
DIAGNOSTICS_ALLOWED_ROOT_FILES = DIAGNOSTICS_REQUIRED_ROOT_FILES | {
    "manifest.sha256"
}
CANONICAL_RELEASE_LOCK = PurePosixPath("configs/stage4_completion_release.json")
RELEASE_SCHEMA_VERSION = 2
RELEASE_STAGE_NAME = "stage4_development_completion_supplement"
RELEASE_POLICY_ID = "stage4_development_only_completion_release_v1"
RELEASE_TOP_LEVEL_KEYS = {
    "release_schema_version",
    "stage_name",
    "policy_id",
    "release_control",
    "source_binding",
    "parent_final_release",
    "artifacts",
    "diagnostics_artifact",
    "protocol",
    "report",
}
RELEASE_ARTIFACT_KEYS = {
    "source_name",
    "source_id",
    "path",
    "artifact_id",
    "run_id",
    "results_payload_sha256",
    "manifest_sha256",
    "matrix_id",
    "experiment_count",
    "candidate_seed_run_count",
    "manifest_file_count",
}
RELEASE_PROTOCOL_FINAL_SAFE_SENTINEL = {
    "development_only": True,
    "final_holdout_evaluated": False,
    "final_holdout_prediction_count": 0,
    "final_holdout_target_values_used_for_fit_calibration_scoring": False,
    "final_holdout_selection_claim": "none",
}
INTERVAL_RESERVE_FIELDS = (
    "interval_diagnostics_id",
    "interval_below_truth_rate",
    "interval_above_truth_rate",
    "target_exceeds_upper_rate",
    "mean_extra_reserved_tokens",
    "raw_interval_below_truth_rate",
    "raw_interval_above_truth_rate",
    "raw_target_exceeds_upper_rate",
    "raw_mean_extra_reserved_tokens",
)
RUN_DISPERSION_FIELDS = (
    "run_dispersion_extension_id",
    "mean_within_task_run_mae_iqr",
    "median_within_task_run_mae_iqr",
    "max_within_task_run_mae_iqr",
    "mean_within_task_run_mae_max_minus_min",
    "median_within_task_run_mae_max_minus_min",
    "max_within_task_run_mae_max_minus_min",
)
DIAGNOSTICS_LIFECYCLE_UNAVAILABLE_REASON = (
    "no_presealed_development_lifecycle_projection_v1"
)
DIAGNOSTICS_UNAVAILABLE_LIFECYCLE_METRICS = [
    "progress",
    "run_variance_iqr_max_minus_min",
    "termination",
]
EXPECTED_FINAL_HOLDOUT = {
    "evaluated": False,
    "prediction_count": 0,
    "target_values_used_for_fit_calibration_scoring": False,
    "selection_claim": "none",
}
EXPECTED_RESULTS_KEYS = {
    "results_schema_version",
    "stage_name",
    "run_policy_id",
    "artifact_layout_id",
    "checkpoint_policy_id",
    "run_id",
    "source",
    "data_foundation",
    "code_binding",
    "runtime_versions",
    "dataset",
    "development_protocol",
    "matrix",
    "experiments",
    "matched_coverage_calibration",
    "paired_same_task_across_conditions",
    "summary",
    "final_holdout",
    "results_payload_sha256",
}


class CompletionSummaryError(RuntimeError):
    """Completion artifacts cannot be summarized without violating the contract."""


@dataclass(frozen=True)
class ArtifactReference:
    path: Path
    expected_artifact_id: str | None = None
    expected_results_payload_sha256: str | None = None


@dataclass(frozen=True)
class LoadedArtifact:
    path: Path
    artifact_id: str
    results_payload_sha256: str
    document: Mapping[str, object]


@dataclass(frozen=True)
class LoadedDiagnosticsArtifact:
    path: Path
    artifact_id: str
    results_payload_sha256: str
    document: Mapping[str, object]


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise CompletionSummaryError("metadata is not finite canonical JSON") from exc


def _semantic_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _required_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CompletionSummaryError(f"{name} must be a non-empty string")
    return value


def _required_sha256(value: object, *, name: str) -> str:
    text = _required_string(value, name=name)
    if (
        len(text) != 64
        or text != text.lower()
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise CompletionSummaryError(f"{name} must be a lowercase SHA-256")
    return text


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise CompletionSummaryError(f"{name} must be a JSON object")
    return value


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise CompletionSummaryError(f"{name} must be a JSON array")
    return value


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompletionSummaryError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise CompletionSummaryError(f"{name} must be finite")
    return number


def _read_regular_bytes(path: Path, *, maximum_bytes: int, name: str) -> bytes:
    path = _plain_lexical_path(path, name=name)
    try:
        before = path.lstat()
    except OSError as exc:
        raise CompletionSummaryError(f"{name} cannot be inspected") from exc
    if (
        _is_link_or_reparse(path)
        or not stat.S_ISREG(before.st_mode)
        or int(getattr(before, "st_nlink", 1)) != 1
    ):
        raise CompletionSummaryError(f"{name} must be a regular file")
    if before.st_size > maximum_bytes:
        raise CompletionSummaryError(f"{name} exceeds the size limit")
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(
        getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CompletionSummaryError(f"{name} cannot be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        before_identity = (
            int(getattr(before, "st_dev", 0)),
            int(getattr(before, "st_ino", 0)),
        )
        opened_identity = (
            int(getattr(opened, "st_dev", 0)),
            int(getattr(opened, "st_ino", 0)),
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or int(getattr(opened, "st_nlink", 1)) != 1
            or before.st_size != opened.st_size
            or (
                0 not in before_identity
                and 0 not in opened_identity
                and before_identity != opened_identity
            )
        ):
            raise CompletionSummaryError(f"{name} changed identity before reading")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            payload = handle.read(maximum_bytes + 1)
            opened_after = os.fstat(handle.fileno())
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    try:
        after = path.lstat()
    except OSError as exc:
        raise CompletionSummaryError(f"{name} cannot be re-inspected") from exc
    if (
        len(payload) > maximum_bytes
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
        or int(getattr(before, "st_nlink", 1))
        != int(getattr(after, "st_nlink", 1))
        or int(getattr(before, "st_file_attributes", 0))
        != int(getattr(after, "st_file_attributes", 0))
        or len(payload) != after.st_size
        or opened_after.st_size != before.st_size
        or opened_after.st_mtime_ns != opened.st_mtime_ns
        or opened_after.st_ctime_ns != opened.st_ctime_ns
        or _is_link_or_reparse(path)
    ):
        raise CompletionSummaryError(f"{name} changed while being read")
    return payload


def _json_object(payload: bytes, *, name: str) -> Mapping[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompletionSummaryError(f"{name} is not valid UTF-8 JSON") from exc
    return _mapping(value, name=name)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _is_link_or_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except (FileNotFoundError, NotADirectoryError):
        return False
    if stat.S_ISLNK(metadata.st_mode):
        return True
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    if int(getattr(metadata, "st_file_attributes", 0)) & reparse_flag:
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _plain_lexical_path(path: Path, *, name: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    for current in reversed((lexical, *lexical.parents)):
        if _is_link_or_reparse(current):
            raise CompletionSummaryError(
                f"{name} traverses a symlink, junction, or reparse point"
            )
    if not lexical.exists():
        raise CompletionSummaryError(f"{name} does not exist")
    return lexical


def _verify_diagnostics_physical_topology(path: Path) -> None:
    if _is_link_or_reparse(path):
        raise CompletionSummaryError(
            "completion diagnostics artifact root is linked or reparse-backed"
        )
    try:
        root_metadata = path.lstat()
        with os.scandir(path) as entries:
            children = list(entries)
    except OSError as exc:
        raise CompletionSummaryError(
            "completion diagnostics artifact topology cannot be inspected"
        ) from exc
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise CompletionSummaryError(
            "completion diagnostics artifact must be a regular directory"
        )
    names: set[str] = set()
    for entry in children:
        child = path / entry.name
        try:
            metadata = child.lstat()
        except OSError as exc:
            raise CompletionSummaryError(
                "completion diagnostics artifact member cannot be inspected"
            ) from exc
        if (
            entry.name in names
            or _is_link_or_reparse(child)
            or not stat.S_ISREG(metadata.st_mode)
            or int(getattr(metadata, "st_nlink", 1)) != 1
        ):
            raise CompletionSummaryError(
                "completion diagnostics artifact physical topology differs"
            )
        names.add(entry.name)
    if (
        not DIAGNOSTICS_REQUIRED_ROOT_FILES <= names
        or not names <= DIAGNOSTICS_ALLOWED_ROOT_FILES
    ):
        raise CompletionSummaryError(
            "completion diagnostics artifact physical topology differs"
        )


def _development_artifact_directory(
    path: Path,
    *,
    development_runs_root: Path,
) -> Path:
    candidate = path.parent if path.name == "results.json" else path
    try:
        resolved_root = development_runs_root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise CompletionSummaryError("development artifact path does not exist") from exc
    if resolved == resolved_root or not _is_within(resolved, resolved_root):
        raise CompletionSummaryError(
            "artifact must be below the Stage 4 development runs root; "
            "final artifacts are never opened"
        )
    if not resolved.is_dir():
        raise CompletionSummaryError("development artifact path must be a directory")
    return resolved


def _safe_lock_artifact_path(value: object, *, repo_root: Path) -> Path:
    text = _required_string(value, name="release-lock artifact path")
    if "\\" in text:
        raise CompletionSummaryError(
            "release-lock artifact paths must use forward slashes"
        )
    pure = PurePosixPath(text)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or pure.as_posix() != text
        or len(pure.parts) != 4
    ):
        raise CompletionSummaryError(
            "release-lock artifact paths must be safe repository-relative paths"
        )
    expected_prefix = PurePosixPath("workspace/stage4/runs")
    if pure == expected_prefix or not pure.is_relative_to(expected_prefix):
        raise CompletionSummaryError(
            "release-lock artifacts must be below workspace/stage4/runs"
        )
    return repo_root.joinpath(*pure.parts)


def _release_lock_references(
    path: Path,
    *,
    repo_root: Path,
) -> list[ArtifactReference]:
    supplied_text = os.fspath(path)
    supplied = Path(path)
    supplied_parts = supplied_text.replace("\\", "/").split("/")
    if (
        not supplied_text
        or supplied_text != supplied_text.strip()
        or "\x00" in supplied_text
        or any(part in {".", ".."} for part in supplied_parts)
    ):
        raise CompletionSummaryError("completion release lock path is not canonical")
    root = _plain_lexical_path(repo_root, name="repository root")
    if not root.is_dir():
        raise CompletionSummaryError("repository root must be a directory")
    expected = root.joinpath(*CANONICAL_RELEASE_LOCK.parts)
    if supplied.is_absolute():
        candidate = Path(os.path.abspath(os.fspath(supplied.expanduser())))
    else:
        if (
            supplied_text != CANONICAL_RELEASE_LOCK.as_posix()
            or "\\" in supplied_text
        ):
            raise CompletionSummaryError(
                "only configs/stage4_completion_release.json may be used "
                "as a completion release lock"
            )
        candidate = root.joinpath(*CANONICAL_RELEASE_LOCK.parts)
    expected_lexical = Path(os.path.abspath(os.fspath(expected)))
    if candidate != expected_lexical:
        raise CompletionSummaryError(
            "only configs/stage4_completion_release.json may be used "
            "as a completion release lock"
        )
    candidate = _plain_lexical_path(
        candidate,
        name="completion release lock",
    )
    if candidate != expected_lexical:
        raise CompletionSummaryError("completion release lock path is not canonical")
    document = _json_object(
        _read_regular_bytes(
            candidate,
            maximum_bytes=MAX_METADATA_BYTES,
            name="completion release lock",
        ),
        name="completion release lock",
    )
    if (
        set(document) != RELEASE_TOP_LEVEL_KEYS
        or document.get("release_schema_version") != RELEASE_SCHEMA_VERSION
        or document.get("stage_name") != RELEASE_STAGE_NAME
        or document.get("policy_id") != RELEASE_POLICY_ID
    ):
        raise CompletionSummaryError(
            "completion release lock schema or identity differs"
        )
    for key in (
        "release_control",
        "source_binding",
        "parent_final_release",
        "diagnostics_artifact",
        "report",
    ):
        _mapping(document.get(key), name=f"completion release lock.{key}")
    protocol = _mapping(
        document.get("protocol"),
        name="completion release lock.protocol",
    )
    if any(
        protocol.get(key) != expected
        or type(protocol.get(key)) is not type(expected)
        for key, expected in RELEASE_PROTOCOL_FINAL_SAFE_SENTINEL.items()
    ):
        raise CompletionSummaryError(
            "completion release lock protocol is not final-safe"
        )
    entries = _list(document.get("artifacts"), name="artifacts")
    if not entries:
        raise CompletionSummaryError("completion release lock has no development artifacts")
    references: list[ArtifactReference] = []
    for index, entry in enumerate(entries):
        item = _mapping(entry, name=f"artifacts[{index}]")
        if set(item) != RELEASE_ARTIFACT_KEYS:
            raise CompletionSummaryError(
                f"artifacts[{index}] schema differs from the formal release"
            )
        _required_string(item.get("source_name"), name=f"artifacts[{index}].source_name")
        _required_string(item.get("source_id"), name=f"artifacts[{index}].source_id")
        run_id = _required_string(item.get("run_id"), name=f"artifacts[{index}].run_id")
        if len(run_id) < 20 or any(
            character not in "0123456789abcdef" for character in run_id
        ):
            raise CompletionSummaryError(
                f"artifacts[{index}].run_id must be hexadecimal"
            )
        for key in ("experiment_count", "candidate_seed_run_count", "manifest_file_count"):
            _positive_integer(item.get(key), name=f"artifacts[{index}].{key}")
        expected_artifact_id = _required_sha256(
            item.get("artifact_id"),
            name=f"artifacts[{index}].artifact_id",
        )
        expected_results = _required_sha256(
            item.get("results_payload_sha256"),
            name=f"artifacts[{index}].results_payload_sha256",
        )
        for key in ("manifest_sha256", "matrix_id"):
            _required_sha256(item.get(key), name=f"artifacts[{index}].{key}")
        artifact_path = _safe_lock_artifact_path(item.get("path"), repo_root=root)
        if artifact_path.name != f"s4-{run_id[:20]}":
            raise CompletionSummaryError(
                f"artifacts[{index}] path and run_id differ"
            )
        references.append(
            ArtifactReference(
                path=artifact_path,
                expected_artifact_id=expected_artifact_id,
                expected_results_payload_sha256=expected_results,
            )
        )
    return references


def resolve_artifact_references(
    inputs: Sequence[str | os.PathLike[str]],
    *,
    repo_root: Path = ROOT,
    development_runs_root: Path | None = None,
) -> tuple[ArtifactReference, ...]:
    """Resolve direct run directories or an explicit completion release lock."""

    if not inputs:
        raise CompletionSummaryError("at least one input is required")
    runs_root = development_runs_root or repo_root / "workspace" / "stage4" / "runs"
    references: list[ArtifactReference] = []
    for raw in inputs:
        path = Path(raw)
        if not path.is_absolute():
            path = repo_root / path
        if path.is_dir() or path.name == "results.json":
            references.append(
                ArtifactReference(
                    _development_artifact_directory(
                        path,
                        development_runs_root=runs_root,
                    )
                )
            )
            continue
        if path.suffix.casefold() != ".json":
            raise CompletionSummaryError(
                "inputs must be development run directories, results.json files, "
                "or configs/stage4_completion_release.json"
            )
        references.extend(_release_lock_references(path, repo_root=repo_root))

    resolved: list[ArtifactReference] = []
    seen: set[Path] = set()
    for reference in references:
        directory = _development_artifact_directory(
            reference.path,
            development_runs_root=runs_root,
        )
        if directory in seen:
            raise CompletionSummaryError(
                f"duplicate development artifact input: {directory}"
            )
        seen.add(directory)
        resolved.append(
            ArtifactReference(
                directory,
                reference.expected_artifact_id,
                reference.expected_results_payload_sha256,
            )
        )
    return tuple(resolved)


def load_development_artifact(reference: ArtifactReference) -> LoadedArtifact:
    """Load only aggregate-safe metadata from one development artifact."""

    manifest_payload = _read_regular_bytes(
        reference.path / "manifest.json",
        maximum_bytes=MAX_METADATA_BYTES,
        name="development artifact manifest",
    )
    manifest = _json_object(manifest_payload, name="development artifact manifest")
    if set(manifest) != {
        "artifact_id",
        "stage_name",
        "schema_version",
        "files",
        "metadata",
    }:
        raise CompletionSummaryError("development artifact manifest keys differ")
    if manifest["stage_name"] != DEVELOPMENT_STAGE_NAME:
        raise CompletionSummaryError(
            "only Stage 4 development source artifacts may be summarized"
        )
    artifact_id = _required_sha256(
        manifest["artifact_id"], name="development artifact id"
    )
    files = _mapping(manifest["files"], name="development artifact files")
    results_file_hash = _required_sha256(
        files.get("results.json"), name="manifest results.json SHA-256"
    )
    metadata = _mapping(manifest["metadata"], name="development artifact metadata")
    manifest_semantic = {
        "stage_name": manifest["stage_name"],
        "schema_version": manifest["schema_version"],
        "files": dict(files),
        "metadata": dict(metadata),
    }
    if _semantic_sha256(manifest_semantic) != artifact_id:
        raise CompletionSummaryError("development artifact semantic id does not close")

    success = _read_regular_bytes(
        reference.path / "_SUCCESS",
        maximum_bytes=256,
        name="development artifact success marker",
    )
    try:
        success_id = success.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise CompletionSummaryError("success marker is not ASCII") from exc
    if success_id != artifact_id:
        raise CompletionSummaryError("success marker does not match artifact id")

    results_payload = _read_regular_bytes(
        reference.path / "results.json",
        maximum_bytes=MAX_RESULTS_BYTES,
        name="Stage 4 development results",
    )
    if hashlib.sha256(results_payload).hexdigest() != results_file_hash:
        raise CompletionSummaryError("results.json does not match the artifact manifest")
    results = _json_object(results_payload, name="Stage 4 development results")
    if set(results) != EXPECTED_RESULTS_KEYS:
        raise CompletionSummaryError("Stage 4 development results keys differ")
    if results["stage_name"] != RESULTS_STAGE_NAME:
        raise CompletionSummaryError("results are not a Stage 4 development document")
    if results["final_holdout"] != EXPECTED_FINAL_HOLDOUT:
        raise CompletionSummaryError(
            "development results claim final-holdout access; refusing to summarize"
        )
    results_digest = _required_sha256(
        results["results_payload_sha256"],
        name="Stage 4 results payload SHA-256",
    )
    digest_document = dict(results)
    digest_document.pop("results_payload_sha256")
    if _semantic_sha256(digest_document) != results_digest:
        raise CompletionSummaryError("Stage 4 results payload SHA-256 does not close")
    if metadata.get("results_payload_sha256") != results_digest:
        raise CompletionSummaryError(
            "artifact metadata and results payload SHA-256 differ"
        )
    if metadata.get("run_id") != results.get("run_id"):
        raise CompletionSummaryError("artifact metadata and results run id differ")
    if (
        reference.expected_artifact_id is not None
        and reference.expected_artifact_id != artifact_id
    ):
        raise CompletionSummaryError("release lock artifact id differs")
    if (
        reference.expected_results_payload_sha256 is not None
        and reference.expected_results_payload_sha256 != results_digest
    ):
        raise CompletionSummaryError("release lock results payload SHA-256 differs")
    return LoadedArtifact(reference.path, artifact_id, results_digest, results)


def load_completion_diagnostics_artifact(
    path: str | os.PathLike[str],
    *,
    repo_root: Path = ROOT,
    diagnostics_root: Path | None = None,
) -> LoadedDiagnosticsArtifact:
    """Load and fully verify the optional immutable diagnostics supplement."""

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    root = diagnostics_root or repo_root / "workspace" / "stage4" / (
        "completion_diagnostics"
    )
    lexical_root = _plain_lexical_path(root, name="completion diagnostics root")
    lexical_candidate = _plain_lexical_path(
        candidate,
        name="completion diagnostics artifact",
    )
    if lexical_candidate == lexical_root or not _is_within(
        lexical_candidate, lexical_root
    ):
        raise CompletionSummaryError(
            "diagnostics supplement must be below the completion diagnostics root"
        )
    try:
        resolved_root = lexical_root.resolve(strict=True)
        resolved = lexical_candidate.resolve(strict=True)
    except OSError as exc:
        raise CompletionSummaryError(
            "completion diagnostics artifact path does not exist"
        ) from exc
    if resolved == resolved_root or not _is_within(resolved, resolved_root):
        raise CompletionSummaryError(
            "diagnostics supplement must be below the completion diagnostics root"
        )
    _verify_diagnostics_physical_topology(resolved)
    try:
        manifest = verify_artifact(resolved)
    except Exception as exc:
        raise CompletionSummaryError(
            "completion diagnostics artifact verification failed"
        ) from exc
    if (
        manifest.stage_name != "stage4_completion_diagnostics"
        or manifest.schema_version
        != COMPLETION_DIAGNOSTICS_ARTIFACT_SCHEMA_VERSION
    ):
        raise CompletionSummaryError(
            "input is not a Stage 4 completion diagnostics artifact"
        )
    if (
        "results.json" not in manifest.files
        or set(manifest.files) - {"results.json", "manifest.sha256"}
    ):
        raise CompletionSummaryError(
            "completion diagnostics artifact manifest topology differs"
        )
    _verify_diagnostics_physical_topology(resolved)
    payload = _read_regular_bytes(
        resolved / "results.json",
        maximum_bytes=MAX_RESULTS_BYTES,
        name="completion diagnostics results",
    )
    document = _json_object(payload, name="completion diagnostics results")
    try:
        if __package__:
            from scripts.run_stage4_completion_diagnostics import (
                verify_diagnostics_results_document,
            )
        else:  # pragma: no cover - direct production CLI invocation
            from run_stage4_completion_diagnostics import (
                verify_diagnostics_results_document,
            )

        results_digest = verify_diagnostics_results_document(document)
    except Exception as exc:
        raise CompletionSummaryError(
            "completion diagnostics results failed verification"
        ) from exc
    if (
        manifest.files.get("results.json")
        != hashlib.sha256(payload).hexdigest()
        or manifest.metadata.get("results_payload_sha256") != results_digest
    ):
        raise CompletionSummaryError(
            "completion diagnostics manifest/results binding differs"
        )
    return LoadedDiagnosticsArtifact(
        resolved,
        manifest.artifact_id,
        results_digest,
        document,
    )


def _seed_results(candidate: Mapping[str, object], *, name: str) -> list[Mapping[str, object]]:
    values = _list(candidate.get("seed_results"), name=f"{name}.seed_results")
    resolved = [
        _mapping(value, name=f"{name}.seed_results[{index}]")
        for index, value in enumerate(values)
    ]
    seeds = tuple(
        _positive_integer(
            value.get("split_seed"),
            name=f"{name}.seed_results[{index}].split_seed",
        )
        for index, value in enumerate(resolved)
    )
    if seeds != FROZEN_SPLIT_SEEDS:
        raise CompletionSummaryError(
            f"{name} must contain the three frozen split seeds in order"
        )
    return resolved


def _candidate_map(
    experiment: Mapping[str, object],
    *,
    name: str,
) -> dict[str, Mapping[str, object]]:
    candidates = _list(experiment.get("candidates"), name=f"{name}.candidates")
    result: dict[str, Mapping[str, object]] = {}
    for index, raw in enumerate(candidates):
        candidate = _mapping(raw, name=f"{name}.candidates[{index}]")
        candidate_id = _required_string(
            candidate.get("candidate_id"),
            name=f"{name}.candidates[{index}].candidate_id",
        )
        if candidate_id in result:
            raise CompletionSummaryError(f"{name} has duplicate candidate ids")
        result[candidate_id] = candidate
    return result


def _comparison_seed_row(
    *,
    candidate_seed: Mapping[str, object],
    reference_seed: Mapping[str, object],
    candidate_id: str,
    reference_id: str,
    name: str,
) -> dict[str, object]:
    if candidate_seed.get("split_seed") != reference_seed.get("split_seed"):
        raise CompletionSummaryError(f"{name} seed pairing differs")
    paired = _mapping(
        candidate_seed.get("paired_vs_reference"),
        name=f"{name}.paired_vs_reference",
    )
    if (
        paired.get("candidate_id") != candidate_id
        or paired.get("reference_id") != reference_id
    ):
        raise CompletionSummaryError(f"{name} paired candidate identities differ")
    candidate_mae = _finite_number(
        paired.get("candidate_mae"), name=f"{name}.candidate_mae"
    )
    reference_mae = _finite_number(
        paired.get("reference_mae"), name=f"{name}.reference_mae"
    )
    delta = _finite_number(paired.get("mae_delta"), name=f"{name}.mae_delta")
    lower = _finite_number(
        paired.get("mae_delta_ci_lower"), name=f"{name}.mae_delta_ci_lower"
    )
    upper = _finite_number(
        paired.get("mae_delta_ci_upper"), name=f"{name}.mae_delta_ci_upper"
    )
    win_probability = _finite_number(
        paired.get("candidate_win_probability"),
        name=f"{name}.candidate_win_probability",
    )
    if not 0 <= win_probability <= 1 or lower > upper:
        raise CompletionSummaryError(f"{name} paired bootstrap values are invalid")
    if not math.isclose(
        delta,
        candidate_mae - reference_mae,
        rel_tol=1e-10,
        abs_tol=1e-8,
    ):
        raise CompletionSummaryError(f"{name} paired MAE delta does not close")
    candidate_metrics = _mapping(
        candidate_seed.get("metrics"), name=f"{name}.candidate metrics"
    )
    reference_metrics = _mapping(
        reference_seed.get("metrics"), name=f"{name}.reference metrics"
    )
    if not math.isclose(
        _finite_number(candidate_metrics.get("mae"), name=f"{name}.candidate metrics.mae"),
        candidate_mae,
        rel_tol=1e-10,
        abs_tol=1e-8,
    ) or not math.isclose(
        _finite_number(reference_metrics.get("mae"), name=f"{name}.reference metrics.mae"),
        reference_mae,
        rel_tol=1e-10,
        abs_tol=1e-8,
    ):
        raise CompletionSummaryError(f"{name} paired MAE differs from candidate metrics")
    if upper < 0:
        bootstrap_outcome = "candidate_supported"
    elif lower > 0:
        bootstrap_outcome = "reference_supported"
    else:
        bootstrap_outcome = "inconclusive"
    return {
        "split_seed": candidate_seed["split_seed"],
        "candidate_mae": candidate_mae,
        "reference_mae": reference_mae,
        "mae_delta": delta,
        "mae_winner": (
            "candidate" if delta < 0 else "reference" if delta > 0 else "tie"
        ),
        "mae_delta_ci_lower": lower,
        "mae_delta_ci_upper": upper,
        "candidate_win_probability": win_probability,
        "bootstrap_outcome": bootstrap_outcome,
    }


def _ablation(
    candidate: Mapping[str, object],
    *,
    reference_id: str,
    axis: str,
    allowed_paths: set[str] | None,
    name: str,
) -> None:
    ablation = _mapping(candidate.get("ablation"), name=f"{name}.ablation")
    if (
        ablation.get("reference_candidate_id") != reference_id
        or ablation.get("axis") != axis
    ):
        raise CompletionSummaryError(f"{name} ablation identity differs")
    if allowed_paths is not None:
        paths = _list(
            ablation.get("allowed_config_paths"),
            name=f"{name}.ablation.allowed_config_paths",
        )
        if set(paths) != allowed_paths:
            raise CompletionSummaryError(f"{name} ablation paths differ")


def _graph_seed_policy(candidate: Mapping[str, object], *, name: str) -> str:
    graph = _mapping(candidate.get("candidate_graph"), name=f"{name}.candidate_graph")
    return _required_string(
        graph.get("seed_policy_id"), name=f"{name}.candidate_graph.seed_policy_id"
    )


def _comparison_document(
    *,
    artifact: LoadedArtifact,
    experiment: Mapping[str, object],
    candidate: Mapping[str, object],
    reference: Mapping[str, object],
    candidate_id: str,
    reference_id: str,
    mae_seed_outcomes_role: str,
    name: str,
) -> dict[str, object]:
    candidate_seeds = _seed_results(candidate, name=f"{name}.{candidate_id}")
    reference_seeds = _seed_results(reference, name=f"{name}.{reference_id}")
    seed_rows = [
        _comparison_seed_row(
            candidate_seed=candidate_seed,
            reference_seed=reference_seed,
            candidate_id=candidate_id,
            reference_id=reference_id,
            name=f"{name}.seed[{candidate_seed['split_seed']}]",
        )
        for candidate_seed, reference_seed in zip(
            candidate_seeds, reference_seeds, strict=True
        )
    ]
    source = _mapping(artifact.document.get("source"), name="artifact source")
    return {
        "artifact_id": artifact.artifact_id,
        "run_id": artifact.document["run_id"],
        "source_name": _required_string(
            source.get("source_name"), name="source.source_name"
        ),
        "experiment_id": _required_string(
            experiment.get("experiment_id"), name=f"{name}.experiment_id"
        ),
        "position": _required_string(
            experiment.get("position"), name=f"{name}.position"
        ),
        "target": _required_string(experiment.get("target"), name=f"{name}.target"),
        "condition_id": _required_string(
            experiment.get("condition_id"), name=f"{name}.condition_id"
        ),
        "candidate_id": candidate_id,
        "reference_candidate_id": reference_id,
        "seed_results": seed_rows,
        "mae_seed_outcomes": {
            "candidate_supported": sum(
                row["bootstrap_outcome"] == "candidate_supported" for row in seed_rows
            ),
            "reference_supported": sum(
                row["bootstrap_outcome"] == "reference_supported" for row in seed_rows
            ),
            "inconclusive": sum(
                row["bootstrap_outcome"] == "inconclusive" for row in seed_rows
            ),
        },
        "mae_seed_outcomes_role": _required_string(
            mae_seed_outcomes_role, name=f"{name}.mae_seed_outcomes_role"
        ),
    }


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CompletionSummaryError(f"{name} must be a positive integer")
    return value


def _task_metric_map(
    seed_result: Mapping[str, object],
    *,
    name: str,
) -> dict[str, dict[str, float | int | str]]:
    rows = _list(seed_result.get("task_metrics"), name=f"{name}.task_metrics")
    if not rows:
        raise CompletionSummaryError(f"{name}.task_metrics cannot be empty")
    result: dict[str, dict[str, float | int | str]] = {}
    for index, raw in enumerate(rows):
        row_name = f"{name}.task_metrics[{index}]"
        row = _mapping(raw, name=row_name)
        pseudonym = _required_string(
            row.get("task_pseudonym"), name=f"{row_name}.task_pseudonym"
        )
        if pseudonym in result:
            raise CompletionSummaryError(f"{name} has duplicate task pseudonyms")
        n_points = _positive_integer(
            row.get("n_points"), name=f"{row_name}.n_points"
        )
        n_trajectories = _positive_integer(
            row.get("n_trajectories"), name=f"{row_name}.n_trajectories"
        )
        weight_sum = _finite_number(
            row.get("weight_sum"), name=f"{row_name}.weight_sum"
        )
        coverage = _finite_number(
            row.get("weighted_coverage"), name=f"{row_name}.weighted_coverage"
        )
        interval_score = _finite_number(
            row.get("weighted_interval_score"),
            name=f"{row_name}.weighted_interval_score",
        )
        mae = _finite_number(
            row.get("weighted_mae"), name=f"{row_name}.weighted_mae"
        )
        if weight_sum <= 0:
            raise CompletionSummaryError(f"{row_name}.weight_sum must be positive")
        if not 0 <= coverage <= 1:
            raise CompletionSummaryError(
                f"{row_name}.weighted_coverage must be within [0, 1]"
            )
        if interval_score < 0 or mae < 0:
            raise CompletionSummaryError(
                f"{row_name} interval score and MAE must be non-negative"
            )
        result[pseudonym] = {
            "task_pseudonym": pseudonym,
            "n_points": n_points,
            "n_trajectories": n_trajectories,
            "weight_sum": weight_sum,
            "weighted_coverage": coverage,
            "weighted_interval_score": interval_score,
            "weighted_mae": mae,
        }
    return result


def _seed_comparability_contract(
    seed_result: Mapping[str, object],
    *,
    condition_id: str,
    split_seed: int,
    name: str,
) -> dict[str, object]:
    comparability_key = _list(
        seed_result.get("comparability_key"), name=f"{name}.comparability_key"
    )
    if len(comparability_key) != 9 or any(
        not isinstance(value, str) or not value for value in comparability_key
    ):
        raise CompletionSummaryError(
            f"{name}.comparability_key must contain nine non-empty strings"
        )
    for index in (0, 1, 2):
        _required_sha256(
            comparability_key[index],
            name=f"{name}.comparability_key[{index}]",
        )
    split_plan_id = _required_sha256(
        seed_result.get("split_plan_id"), name=f"{name}.split_plan_id"
    )
    if comparability_key[1] != split_plan_id:
        raise CompletionSummaryError(
            f"{name}.comparability_key does not bind split_plan_id"
        )
    expected_suffix = [
        SEED_POLICY_POSITION,
        SEED_POLICY_TARGET,
        condition_id,
        SEED_POLICY_CALIBRATOR_ID,
        format(SEED_POLICY_ALPHA, ".1f"),
        "token_prediction_metrics_v2",
    ]
    if comparability_key[3:] != expected_suffix:
        raise CompletionSummaryError(
            f"{name}.comparability_key semantic suffix differs"
        )
    cohort_projection_id = _required_string(
        seed_result.get("cohort_projection_id"),
        name=f"{name}.cohort_projection_id",
    )
    if cohort_projection_id != COHORT_PROJECTION_ID:
        raise CompletionSummaryError(f"{name}.cohort_projection_id differs")
    cohort_projection_sha256 = _required_sha256(
        seed_result.get("cohort_projection_sha256"),
        name=f"{name}.cohort_projection_sha256",
    )
    prediction_count = _positive_integer(
        seed_result.get("prediction_count"), name=f"{name}.prediction_count"
    )
    task_metric_policy_id = _required_string(
        seed_result.get("task_metric_policy_id"),
        name=f"{name}.task_metric_policy_id",
    )
    if task_metric_policy_id != TASK_METRIC_POLICY_ID:
        raise CompletionSummaryError(f"{name}.task_metric_policy_id differs")
    if (
        _positive_integer(seed_result.get("split_seed"), name=f"{name}.split_seed")
        != split_seed
    ):
        raise CompletionSummaryError(f"{name}.split_seed identity differs")
    return {
        "comparability_key": list(comparability_key),
        "split_plan_id": split_plan_id,
        "cohort_projection_id": cohort_projection_id,
        "cohort_projection_sha256": cohort_projection_sha256,
        "prediction_count": prediction_count,
        "task_metric_policy_id": task_metric_policy_id,
    }


def _paired_task_cohort_sha256(
    rows: Sequence[Mapping[str, float | int | str]],
    *,
    comparability_key: Sequence[object],
    condition_id: str,
) -> str:
    task_shapes = sorted(
        [
            {
                "n_points": int(row["n_points"]),
                "n_trajectories": int(row["n_trajectories"]),
                "weight_sum": float(row["weight_sum"]),
            }
            for row in rows
        ],
        key=lambda item: (
            int(item["n_points"]),
            int(item["n_trajectories"]),
            float(item["weight_sum"]),
        ),
    )
    return _semantic_sha256(
        {
            "cohort_digest_policy_id": (
                "paired_task_cohort_without_split_bound_pseudonym_or_fold_v2"
            ),
            "task_metric_policy_id": TASK_METRIC_POLICY_ID,
            "dataset_id": comparability_key[0],
            "cohort_contract_sha256": comparability_key[2],
            "condition_id": condition_id,
            "task_count": len(task_shapes),
            "prediction_count": sum(
                int(item["n_points"]) for item in task_shapes
            ),
            "task_shapes": task_shapes,
        }
    )


def _weighted_task_aggregate(
    rows: Sequence[Mapping[str, float | int | str]],
    *,
    metric: str,
) -> float:
    denominator = sum(float(row["weight_sum"]) for row in rows)
    if denominator <= 0:
        raise CompletionSummaryError("paired task-metric weight sum must be positive")
    return sum(
        float(row["weight_sum"]) * float(row[metric]) for row in rows
    ) / denominator


def _task_simultaneous_coverage(
    rows: Sequence[Mapping[str, float | int | str]],
) -> float:
    if not rows:
        raise CompletionSummaryError(
            "task simultaneous coverage requires at least one task"
        )
    return sum(float(row["weighted_coverage"]) == 1.0 for row in rows) / len(rows)


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise CompletionSummaryError("bootstrap distribution cannot be empty")
    if not 0 <= probability <= 1:
        raise CompletionSummaryError("bootstrap percentile is outside [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return (
        ordered[lower_index] * (1.0 - fraction)
        + ordered[upper_index] * fraction
    )


def _paired_task_cluster_bootstrap(
    candidate_rows: Sequence[Mapping[str, float | int | str]],
    reference_rows: Sequence[Mapping[str, float | int | str]],
    *,
    split_seed: int,
    experiment_id: str,
    candidate_id: str,
    reference_id: str,
    source_name: str,
    source_id: str,
    condition_id: str,
    position: str,
    target: str,
    calibrator_id: str,
    alpha: float,
    paired_task_cohort_sha256: str,
) -> dict[str, float | int | str]:
    """Bootstrap paired task clusters using the frozen deterministic policy."""

    if len(candidate_rows) != len(reference_rows) or not candidate_rows:
        raise CompletionSummaryError("paired task bootstrap rows differ")
    if split_seed not in FROZEN_SPLIT_SEEDS:
        raise CompletionSummaryError("paired task bootstrap split seed differs")
    experiment_id = _required_string(
        experiment_id, name="paired task bootstrap experiment_id"
    )
    candidate_id = _required_string(
        candidate_id, name="paired task bootstrap candidate_id"
    )
    reference_id = _required_string(
        reference_id, name="paired task bootstrap reference_id"
    )
    source_name = _required_string(
        source_name, name="paired task bootstrap source_name"
    )
    source_id = _required_string(source_id, name="paired task bootstrap source_id")
    condition_id = _required_string(
        condition_id, name="paired task bootstrap condition_id"
    )
    position = _required_string(position, name="paired task bootstrap position")
    target = _required_string(target, name="paired task bootstrap target")
    calibrator_id = _required_string(
        calibrator_id, name="paired task bootstrap calibrator_id"
    )
    if position != SEED_POLICY_POSITION or target != SEED_POLICY_TARGET:
        raise CompletionSummaryError("paired task bootstrap lifecycle identity differs")
    if (
        calibrator_id != SEED_POLICY_CALIBRATOR_ID
        or not math.isclose(alpha, SEED_POLICY_ALPHA, rel_tol=0.0, abs_tol=0.0)
    ):
        raise CompletionSummaryError(
            "paired task bootstrap calibration identity differs"
        )
    paired_task_cohort_sha256 = _required_sha256(
        paired_task_cohort_sha256,
        name="paired task bootstrap cohort SHA-256",
    )
    seed_material = {
        "policy_id": SEED_POLICY_BOOTSTRAP_SEED_DERIVATION_ID,
        "split_seed": split_seed,
        "experiment_id": experiment_id,
        "candidate_id": candidate_id,
        "reference_id": reference_id,
        "source_name": source_name,
        "source_id": source_id,
        "condition_id": condition_id,
        "position": position,
        "target": target,
        "calibrator_id": calibrator_id,
        "alpha": alpha,
        "paired_task_cohort_sha256": paired_task_cohort_sha256,
    }
    seed_material_sha256 = hashlib.sha256(
        _canonical_json_bytes(seed_material)
    ).hexdigest()
    bootstrap_seed = int(
        seed_material_sha256[:16], 16
    )
    weights = [float(row["weight_sum"]) for row in candidate_rows]
    interval_differences = [
        weight
        * (
            float(candidate["weighted_interval_score"])
            - float(reference["weighted_interval_score"])
        )
        for weight, candidate, reference in zip(
            weights, candidate_rows, reference_rows, strict=True
        )
    ]
    task_coverage_differences = [
        float(float(candidate["weighted_coverage"]) == 1.0)
        - float(float(reference["weighted_coverage"]) == 1.0)
        for candidate, reference in zip(
            candidate_rows, reference_rows, strict=True
        )
    ]
    generator = random.Random(bootstrap_seed)
    interval_distribution: list[float] = []
    task_coverage_distribution: list[float] = []
    task_count = len(candidate_rows)
    for _ in range(SEED_POLICY_BOOTSTRAP_ITERATIONS):
        denominator = 0.0
        interval_numerator = 0.0
        coverage_numerator = 0.0
        for _ in range(task_count):
            task_index = generator.randrange(task_count)
            denominator += weights[task_index]
            interval_numerator += interval_differences[task_index]
            coverage_numerator += task_coverage_differences[task_index]
        interval_distribution.append(interval_numerator / denominator)
        task_coverage_distribution.append(coverage_numerator / task_count)
    return {
        "bootstrap_policy_id": (
            "paired_task_cluster_interval_and_task_coverage_percentile_bootstrap_v2"
        ),
        "bootstrap_iterations": SEED_POLICY_BOOTSTRAP_ITERATIONS,
        "bootstrap_seed_derivation_policy_id": (
            SEED_POLICY_BOOTSTRAP_SEED_DERIVATION_ID
        ),
        "bootstrap_seed_material_sha256": seed_material_sha256,
        "bootstrap_random_seed": bootstrap_seed,
        "interval_score_delta_ci_lower": _percentile(
            interval_distribution, 0.025
        ),
        "interval_score_delta_ci_upper": _percentile(
            interval_distribution, 0.975
        ),
        "task_simultaneous_coverage_delta_ci_lower": _percentile(
            task_coverage_distribution, 0.025
        ),
        "task_simultaneous_coverage_delta_ci_upper": _percentile(
            task_coverage_distribution, 0.975
        ),
        "interval_score_win_probability": sum(
            value < 0 for value in interval_distribution
        )
        / SEED_POLICY_BOOTSTRAP_ITERATIONS,
    }


def _close_task_aggregate(
    *,
    aggregate: float,
    metrics: Mapping[str, object],
    metrics_key: str,
    name: str,
) -> None:
    reported = _finite_number(metrics.get(metrics_key), name=f"{name}.{metrics_key}")
    if not math.isclose(aggregate, reported, rel_tol=1e-10, abs_tol=1e-8):
        raise CompletionSummaryError(
            f"{name}.{metrics_key} does not close over task_metrics"
        )


def _seed_policy_seed_row(
    *,
    base_row: Mapping[str, object],
    candidate_seed: Mapping[str, object],
    reference_seed: Mapping[str, object],
    alpha: float,
    experiment_id: str,
    candidate_id: str,
    reference_id: str,
    source_name: str,
    source_id: str,
    condition_id: str,
    position: str,
    target: str,
    calibrator_id: str,
    name: str,
) -> dict[str, object]:
    split_seed = candidate_seed.get("split_seed")
    if (
        split_seed != reference_seed.get("split_seed")
        or split_seed != base_row.get("split_seed")
        or split_seed not in FROZEN_SPLIT_SEEDS
    ):
        raise CompletionSummaryError(f"{name} bootstrap identity seed differs")
    if candidate_seed.get("candidate_id") != candidate_id:
        raise CompletionSummaryError(f"{name} bootstrap candidate identity differs")
    if reference_seed.get("candidate_id") != reference_id:
        raise CompletionSummaryError(f"{name} bootstrap reference identity differs")
    candidate_contract = _seed_comparability_contract(
        candidate_seed,
        condition_id=condition_id,
        split_seed=split_seed,
        name=f"{name}.candidate",
    )
    reference_contract = _seed_comparability_contract(
        reference_seed,
        condition_id=condition_id,
        split_seed=split_seed,
        name=f"{name}.reference",
    )
    if candidate_contract != reference_contract:
        raise CompletionSummaryError(f"{name} paired seed comparability fields differ")
    candidate_tasks = _task_metric_map(candidate_seed, name=f"{name}.candidate")
    reference_tasks = _task_metric_map(reference_seed, name=f"{name}.reference")
    if set(candidate_tasks) != set(reference_tasks):
        raise CompletionSummaryError(f"{name} paired task pseudonym sets differ")
    task_ids = sorted(candidate_tasks)
    for task_id in task_ids:
        candidate_task = candidate_tasks[task_id]
        reference_task = reference_tasks[task_id]
        if candidate_task["n_points"] != reference_task["n_points"]:
            raise CompletionSummaryError(
                f"{name} paired task n_points differ for {task_id}"
            )
        if candidate_task["n_trajectories"] != reference_task["n_trajectories"]:
            raise CompletionSummaryError(
                f"{name} paired task n_trajectories differ for {task_id}"
            )
        if not math.isclose(
            float(candidate_task["weight_sum"]),
            float(reference_task["weight_sum"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise CompletionSummaryError(
                f"{name} paired task weight_sum differs for {task_id}"
            )
    candidate_rows = [candidate_tasks[task_id] for task_id in task_ids]
    reference_rows = [reference_tasks[task_id] for task_id in task_ids]
    prediction_count = sum(int(row["n_points"]) for row in candidate_rows)
    if prediction_count != candidate_contract["prediction_count"]:
        raise CompletionSummaryError(
            f"{name} prediction_count does not close over task_metrics"
        )
    paired_task_cohort_sha256 = _paired_task_cohort_sha256(
        candidate_rows,
        comparability_key=candidate_contract["comparability_key"],
        condition_id=condition_id,
    )
    candidate_metrics = _mapping(
        candidate_seed.get("metrics"), name=f"{name}.candidate.metrics"
    )
    reference_metrics = _mapping(
        reference_seed.get("metrics"), name=f"{name}.reference.metrics"
    )
    candidate_point_coverage = _weighted_task_aggregate(
        candidate_rows, metric="weighted_coverage"
    )
    reference_point_coverage = _weighted_task_aggregate(
        reference_rows, metric="weighted_coverage"
    )
    candidate_task_coverage = _task_simultaneous_coverage(candidate_rows)
    reference_task_coverage = _task_simultaneous_coverage(reference_rows)
    candidate_interval_score = _weighted_task_aggregate(
        candidate_rows, metric="weighted_interval_score"
    )
    reference_interval_score = _weighted_task_aggregate(
        reference_rows, metric="weighted_interval_score"
    )
    candidate_mae = _weighted_task_aggregate(
        candidate_rows, metric="weighted_mae"
    )
    reference_mae = _weighted_task_aggregate(
        reference_rows, metric="weighted_mae"
    )
    for aggregate, metrics, key, side in (
        (candidate_point_coverage, candidate_metrics, "coverage", "candidate"),
        (reference_point_coverage, reference_metrics, "coverage", "reference"),
        (
            candidate_task_coverage,
            candidate_metrics,
            "task_simultaneous_coverage",
            "candidate",
        ),
        (
            reference_task_coverage,
            reference_metrics,
            "task_simultaneous_coverage",
            "reference",
        ),
        (
            candidate_interval_score,
            candidate_metrics,
            "interval_score",
            "candidate",
        ),
        (
            reference_interval_score,
            reference_metrics,
            "interval_score",
            "reference",
        ),
        (candidate_mae, candidate_metrics, "mae", "candidate"),
        (reference_mae, reference_metrics, "mae", "reference"),
    ):
        _close_task_aggregate(
            aggregate=aggregate,
            metrics=metrics,
            metrics_key=key,
            name=f"{name}.{side}.metrics",
        )
    candidate_upper_tail = _finite_number(
        candidate_metrics.get("target_exceeds_upper_rate"),
        name=f"{name}.candidate.metrics.target_exceeds_upper_rate",
    )
    reference_upper_tail = _finite_number(
        reference_metrics.get("target_exceeds_upper_rate"),
        name=f"{name}.reference.metrics.target_exceeds_upper_rate",
    )
    if not 0 <= candidate_upper_tail <= 1 or not 0 <= reference_upper_tail <= 1:
        raise CompletionSummaryError(f"{name} upper-tail rates must be within [0, 1]")
    bootstrap = _paired_task_cluster_bootstrap(
        candidate_rows,
        reference_rows,
        split_seed=split_seed,
        experiment_id=experiment_id,
        candidate_id=candidate_id,
        reference_id=reference_id,
        source_name=source_name,
        source_id=source_id,
        condition_id=condition_id,
        position=position,
        target=target,
        calibrator_id=calibrator_id,
        alpha=alpha,
        paired_task_cohort_sha256=paired_task_cohort_sha256,
    )
    interval_delta = candidate_interval_score - reference_interval_score
    point_coverage_delta = candidate_point_coverage - reference_point_coverage
    task_coverage_delta = candidate_task_coverage - reference_task_coverage
    upper_tail_delta = candidate_upper_tail - reference_upper_tail
    minimum_task_coverage = 1.0 - alpha - SEED_POLICY_COVERAGE_TOLERANCE
    guards = {
        "candidate_task_simultaneous_coverage_at_least_nominal_minus_tolerance": (
            candidate_task_coverage >= minimum_task_coverage
        ),
        "aggregate_task_simultaneous_coverage_delta_within_tolerance": (
            abs(task_coverage_delta) <= SEED_POLICY_COVERAGE_TOLERANCE
        ),
        "task_simultaneous_coverage_delta_ci_lower_at_least_negative_tolerance": (
            float(bootstrap["task_simultaneous_coverage_delta_ci_lower"])
            >= -SEED_POLICY_COVERAGE_TOLERANCE
        ),
        "interval_score_delta_ci_upper_below_zero": (
            float(bootstrap["interval_score_delta_ci_upper"]) < 0
        ),
        "upper_tail_rate_not_worse_beyond_tolerance": (
            upper_tail_delta <= SEED_POLICY_UPPER_TAIL_TOLERANCE
        ),
        "candidate_upper_tail_at_most_alpha_plus_tolerance": (
            candidate_upper_tail <= alpha + SEED_POLICY_UPPER_TAIL_TOLERANCE
        ),
    }
    return {
        **dict(base_row),
        "paired_task_count": len(task_ids),
        "paired_task_cohort_sha256": paired_task_cohort_sha256,
        "cohort_digest_policy_id": (
            "paired_task_cohort_without_split_bound_pseudonym_or_fold_v2"
        ),
        "candidate_weighted_interval_score": candidate_interval_score,
        "reference_weighted_interval_score": reference_interval_score,
        "weighted_interval_score_delta": interval_delta,
        "candidate_weighted_point_coverage": candidate_point_coverage,
        "reference_weighted_point_coverage": reference_point_coverage,
        "weighted_point_coverage_delta": point_coverage_delta,
        "candidate_task_simultaneous_coverage": candidate_task_coverage,
        "reference_task_simultaneous_coverage": reference_task_coverage,
        "task_simultaneous_coverage_delta": task_coverage_delta,
        "candidate_target_exceeds_upper_rate": candidate_upper_tail,
        "reference_target_exceeds_upper_rate": reference_upper_tail,
        "target_exceeds_upper_rate_delta": upper_tail_delta,
        "candidate_minimum_task_simultaneous_coverage": minimum_task_coverage,
        **bootstrap,
        "selection_guards": guards,
        "seed_pass": all(guards.values()),
        "mae_parity_role": "reported_only_not_used_for_selection",
        "raw_forecast_interval_metric_role": (
            "diagnostic_only_not_used_for_selection"
        ),
    }


def _call_pre_comparisons(artifacts: Sequence[LoadedArtifact]) -> list[dict[str, object]]:
    documents: list[dict[str, object]] = []
    seen_cells: set[tuple[str, str, str, str]] = set()
    for artifact in artifacts:
        experiments = _list(
            artifact.document.get("experiments"), name="artifact experiments"
        )
        for index, raw in enumerate(experiments):
            experiment = _mapping(raw, name=f"experiments[{index}]")
            if experiment.get("position") != "call_pre":
                continue
            candidates = _candidate_map(experiment, name=f"experiments[{index}]")
            if MLP_CANDIDATE_ID not in candidates:
                continue
            if LIGHTGBM_CANDIDATE_ID not in candidates:
                raise CompletionSummaryError(
                    "Call-pre MLP experiment lacks the LightGBM reference"
                )
            candidate = candidates[MLP_CANDIDATE_ID]
            _ablation(
                candidate,
                reference_id=LIGHTGBM_CANDIDATE_ID,
                axis="method",
                allowed_paths=None,
                name=f"experiments[{index}].{MLP_CANDIDATE_ID}",
            )
            document = _comparison_document(
                artifact=artifact,
                experiment=experiment,
                candidate=candidate,
                reference=candidates[LIGHTGBM_CANDIDATE_ID],
                candidate_id=MLP_CANDIDATE_ID,
                reference_id=LIGHTGBM_CANDIDATE_ID,
                mae_seed_outcomes_role="primary_comparison_evidence",
                name=f"experiments[{index}]",
            )
            cell = (
                str(document["source_name"]),
                str(document["condition_id"]),
                str(document["position"]),
                str(document["target"]),
            )
            if cell in seen_cells:
                raise CompletionSummaryError("duplicate Call-pre comparison cell")
            seen_cells.add(cell)
            documents.append(document)
    return sorted(
        documents,
        key=lambda item: (
            str(item["source_name"]),
            str(item["condition_id"]),
            str(item["target"]),
        ),
    )


def _seed_policy_plan_index(
    artifact: LoadedArtifact,
) -> dict[str, Mapping[str, object]]:
    matrix = _mapping(artifact.document.get("matrix"), name="artifact matrix")
    plans = _list(matrix.get("plans"), name="artifact matrix.plans")
    result: dict[str, Mapping[str, object]] = {}
    for index, raw in enumerate(plans):
        plan = _mapping(raw, name=f"artifact matrix.plans[{index}]")
        spec = _mapping(
            plan.get("spec"), name=f"artifact matrix.plans[{index}].spec"
        )
        candidates = _list(
            spec.get("candidates"),
            name=f"artifact matrix.plans[{index}].spec.candidates",
        )
        candidate_ids = []
        for candidate_index, candidate_raw in enumerate(candidates):
            candidate = _mapping(
                candidate_raw,
                name=(
                    f"artifact matrix.plans[{index}].spec.candidates"
                    f"[{candidate_index}]"
                ),
            )
            candidate_ids.append(
                _required_string(
                    candidate.get("candidate_id"),
                    name=(
                        f"artifact matrix.plans[{index}].spec.candidates"
                        f"[{candidate_index}].candidate_id"
                    ),
                )
            )
        if POINT_ONLY_SEED_CANDIDATE_ID not in candidate_ids:
            continue
        if set(candidate_ids) != {
            RAW_SEED_CANDIDATE_ID,
            POINT_ONLY_SEED_CANDIDATE_ID,
        } or len(candidate_ids) != 2:
            raise CompletionSummaryError(
                "seed-policy matrix plan candidate set differs"
            )
        experiment_id = _required_string(
            spec.get("experiment_id"),
            name=f"artifact matrix.plans[{index}].spec.experiment_id",
        )
        if experiment_id in result:
            raise CompletionSummaryError(
                "duplicate seed-policy matrix plan experiment id"
            )
        result[experiment_id] = plan
    return result


def _validate_seed_policy_matrix_closure(
    *,
    plan: Mapping[str, object],
    experiment: Mapping[str, object],
    candidates: Mapping[str, Mapping[str, object]],
    name: str,
) -> None:
    spec = _mapping(plan.get("spec"), name=f"{name}.plan.spec")
    for plan_key, experiment_key in (
        ("role", "plan_role"),
        ("axis", "axis"),
        ("reference_experiment_id", "reference_experiment_id"),
        ("allowed_config_paths", "allowed_config_paths"),
    ):
        if plan.get(plan_key) != experiment.get(experiment_key):
            raise CompletionSummaryError(
                f"{name} matrix plan {plan_key} does not close"
            )
    for key in (
        "alpha",
        "calibrator_id",
        "condition_id",
        "experiment_id",
        "position",
        "target",
    ):
        if spec.get(key) != experiment.get(key):
            raise CompletionSummaryError(
                f"{name} matrix plan spec.{key} does not close"
            )
    required_features = _list(
        spec.get("required_features"), name=f"{name}.plan.spec.required_features"
    )
    if required_features:
        raise CompletionSummaryError(
            f"{name} seed-policy plan required_features must be empty"
        )
    plan_candidates = {
        _required_string(
            candidate.get("candidate_id"),
            name=f"{name}.plan candidate id",
        ): candidate
        for candidate in (
            _mapping(raw, name=f"{name}.plan candidate")
            for raw in _list(
                spec.get("candidates"), name=f"{name}.plan.spec.candidates"
            )
        )
    }
    if set(plan_candidates) != set(candidates):
        raise CompletionSummaryError(f"{name} matrix candidate ids do not close")
    for candidate_id, candidate in candidates.items():
        plan_candidate = plan_candidates[candidate_id]
        for plan_key, candidate_key in (
            ("candidate_hash", "candidate_hash"),
            ("estimator_id", "estimator_id"),
            ("feature_set_hash", "feature_set_hash"),
            ("feature_set_id", "feature_set_id"),
            ("role", "role"),
            ("ablation", "ablation"),
        ):
            if plan_candidate.get(plan_key) != candidate.get(candidate_key):
                raise CompletionSummaryError(
                    f"{name} matrix candidate {candidate_id}.{plan_key} "
                    "does not close"
                )
        if plan_candidate.get("graph") != candidate.get("candidate_graph"):
            raise CompletionSummaryError(
                f"{name} matrix candidate {candidate_id}.graph does not close"
            )


def _seed_policy_semantic_cell(
    *,
    artifact: LoadedArtifact,
    experiment: Mapping[str, object],
    name: str,
) -> tuple[str, str, str, None, str, str, str, float]:
    source = _mapping(artifact.document.get("source"), name=f"{name}.source")
    source_name = _required_string(
        source.get("source_name"), name=f"{name}.source.source_name"
    )
    source_id = _required_string(
        source.get("source_id"), name=f"{name}.source.source_id"
    )
    condition_id = _required_string(
        experiment.get("condition_id"), name=f"{name}.condition_id"
    )
    if experiment.get("axis") is not SEED_POLICY_EXPERIMENT_AXIS:
        raise CompletionSummaryError(f"{name}.axis differs from the frozen cell")
    position = _required_string(
        experiment.get("position"), name=f"{name}.position"
    )
    target = _required_string(experiment.get("target"), name=f"{name}.target")
    calibrator_id = _required_string(
        experiment.get("calibrator_id"), name=f"{name}.calibrator_id"
    )
    alpha = _finite_number(experiment.get("alpha"), name=f"{name}.alpha")
    cell = (
        source_name,
        source_id,
        condition_id,
        SEED_POLICY_EXPERIMENT_AXIS,
        position,
        target,
        calibrator_id,
        alpha,
    )
    if cell not in EXPECTED_SEED_POLICY_CELLS:
        raise CompletionSummaryError(f"{name} is not a frozen seed-policy cell")
    if (
        experiment.get("plan_role") != "primary"
        or experiment.get("reference_experiment_id") is not None
        or experiment.get("allowed_config_paths") != []
    ):
        raise CompletionSummaryError(f"{name} primary experiment contract differs")
    return cell


def _semantic_cell_document(
    cell: tuple[str, str, str, None, str, str, str, float],
) -> dict[str, object]:
    return {
        "source_name": cell[0],
        "source_id": cell[1],
        "condition_id": cell[2],
        "axis": cell[3],
        "position": cell[4],
        "target": cell[5],
        "calibrator_id": cell[6],
        "alpha": cell[7],
    }


def _development_task_identity_proof(
    diagnostics_by_key: Mapping[
        tuple[str, str, str, int],
        Mapping[str, object],
    ],
    *,
    source_name: str,
    condition_id: str,
    experiment_id: str,
) -> dict[str, object]:
    candidate_ids = (POINT_ONLY_SEED_CANDIDATE_ID, RAW_SEED_CANDIDATE_ID)
    expected_keys = [
        (source_name, experiment_id, candidate_id, split_seed)
        for candidate_id in candidate_ids
        for split_seed in FROZEN_SPLIT_SEEDS
    ]
    if not diagnostics_by_key:
        return {
            "status": "unavailable",
            "verified": False,
            "reason_code": "completion_diagnostics_supplement_not_supplied",
            "policy_id": DEVELOPMENT_TASK_PSEUDONYM_POLICY_ID,
            "required_candidate_ids": sorted(candidate_ids),
            "required_split_seeds": list(FROZEN_SPLIT_SEEDS),
            "required_candidate_seed_record_count": len(expected_keys),
        }
    missing = [key for key in expected_keys if key not in diagnostics_by_key]
    if missing:
        raise CompletionSummaryError(
            "completion diagnostics lack the seed-policy development task "
            "identity proof"
        )
    task_counts: set[int] = set()
    task_projections: set[str] = set()
    for key in expected_keys:
        diagnostic = diagnostics_by_key[key]
        if diagnostic.get("condition_id") != condition_id:
            raise CompletionSummaryError(
                "completion diagnostic development task identity condition differs"
            )
        parity = _mapping(
            diagnostic.get("checkpoint_parity"),
            name="completion diagnostic checkpoint parity",
        )
        task_count = parity.get("development_task_count")
        if (
            parity.get("status") != "exact"
            or parity.get("development_cohort_status") != "development_only"
            or isinstance(task_count, bool)
            or not isinstance(task_count, int)
            or task_count <= 0
        ):
            raise CompletionSummaryError(
                "completion diagnostic development task identity parity differs"
            )
        task_counts.add(task_count)
        task_projections.add(
            _required_sha256(
                parity.get("development_task_projection_sha256"),
                name="development task identity projection",
            )
        )
    if len(task_counts) != 1 or len(task_projections) != 1:
        raise CompletionSummaryError(
            "split-independent development task identity projection differs "
            "within a semantic cell"
        )
    return {
        "status": "verified",
        "verified": True,
        "reason_code": None,
        "policy_id": DEVELOPMENT_TASK_PSEUDONYM_POLICY_ID,
        "required_candidate_ids": sorted(candidate_ids),
        "required_split_seeds": list(FROZEN_SPLIT_SEEDS),
        "required_candidate_seed_record_count": len(expected_keys),
        "development_task_count": next(iter(task_counts)),
        "development_task_projection_sha256": next(iter(task_projections)),
    }


def _seed_policy_comparisons(
    artifacts: Sequence[LoadedArtifact],
    *,
    diagnostics_by_key: Mapping[
        tuple[str, str, str, int],
        Mapping[str, object],
    ]
    | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    diagnostic_index = diagnostics_by_key or {}
    documents: list[dict[str, object]] = []
    seen_cells: set[tuple[str, str, str, None, str, str, str, float]] = set()
    for artifact in artifacts:
        plan_index = _seed_policy_plan_index(artifact)
        used_plan_ids: set[str] = set()
        experiments = _list(
            artifact.document.get("experiments"), name="artifact experiments"
        )
        for index, raw in enumerate(experiments):
            experiment = _mapping(raw, name=f"experiments[{index}]")
            candidates = _candidate_map(experiment, name=f"experiments[{index}]")
            if POINT_ONLY_SEED_CANDIDATE_ID not in candidates:
                continue
            if RAW_SEED_CANDIDATE_ID not in candidates:
                raise CompletionSummaryError(
                    "point-only seed experiment lacks the raw repaired reference"
                )
            experiment_id = _required_string(
                experiment.get("experiment_id"),
                name=f"experiments[{index}].experiment_id",
            )
            if experiment_id not in plan_index:
                raise CompletionSummaryError(
                    "seed-policy experiment lacks a 1:1 matrix plan"
                )
            if experiment_id in used_plan_ids:
                raise CompletionSummaryError(
                    "duplicate seed-policy experiment for matrix plan"
                )
            used_plan_ids.add(experiment_id)
            _validate_seed_policy_matrix_closure(
                plan=plan_index[experiment_id],
                experiment=experiment,
                candidates=candidates,
                name=f"experiments[{index}]",
            )
            semantic_cell = _seed_policy_semantic_cell(
                artifact=artifact,
                experiment=experiment,
                name=f"experiments[{index}]",
            )
            if semantic_cell in seen_cells:
                raise CompletionSummaryError("duplicate frozen seed-policy cell")
            seen_cells.add(semantic_cell)
            source_name, source_id, condition_id = semantic_cell[:3]
            candidate = candidates[POINT_ONLY_SEED_CANDIDATE_ID]
            reference = candidates[RAW_SEED_CANDIDATE_ID]
            _ablation(
                candidate,
                reference_id=RAW_SEED_CANDIDATE_ID,
                axis=SEED_POLICY_CANDIDATE_AXIS,
                allowed_paths={"graph.seed_policy_id"},
                name=f"experiments[{index}].{POINT_ONLY_SEED_CANDIDATE_ID}",
            )
            if (
                _graph_seed_policy(candidate, name=POINT_ONLY_SEED_CANDIDATE_ID)
                != POINT_ONLY_SEED_POLICY_ID
                or _graph_seed_policy(reference, name=RAW_SEED_CANDIDATE_ID)
                != RAW_SEED_POLICY_ID
            ):
                raise CompletionSummaryError("seed-policy candidate graph differs")
            document = _comparison_document(
                artifact=artifact,
                experiment=experiment,
                candidate=candidate,
                reference=reference,
                candidate_id=POINT_ONLY_SEED_CANDIDATE_ID,
                reference_id=RAW_SEED_CANDIDATE_ID,
                mae_seed_outcomes_role="parity_report_only_not_selection",
                name=f"experiments[{index}]",
            )
            alpha = _finite_number(
                experiment.get("alpha"), name=f"experiments[{index}].alpha"
            )
            if not 0 < alpha < 1:
                raise CompletionSummaryError(
                    f"experiments[{index}].alpha must be within (0, 1)"
                )
            candidate_seeds = _seed_results(
                candidate, name=f"experiments[{index}].{POINT_ONLY_SEED_CANDIDATE_ID}"
            )
            reference_seeds = _seed_results(
                reference, name=f"experiments[{index}].{RAW_SEED_CANDIDATE_ID}"
            )
            base_seed_rows = _list(
                document["seed_results"],
                name=f"experiments[{index}].comparison.seed_results",
            )
            strict_seed_rows = [
                _seed_policy_seed_row(
                    base_row=_mapping(
                        base_row,
                        name=f"experiments[{index}].comparison.seed_results[{seed_index}]",
                    ),
                    candidate_seed=candidate_seed,
                    reference_seed=reference_seed,
                    alpha=alpha,
                    experiment_id=experiment_id,
                    candidate_id=POINT_ONLY_SEED_CANDIDATE_ID,
                    reference_id=RAW_SEED_CANDIDATE_ID,
                    source_name=source_name,
                    source_id=source_id,
                    condition_id=condition_id,
                    position=semantic_cell[4],
                    target=semantic_cell[5],
                    calibrator_id=semantic_cell[6],
                    name=f"experiments[{index}].seed[{candidate_seed['split_seed']}]",
                )
                for seed_index, (base_row, candidate_seed, reference_seed) in enumerate(
                    zip(
                        base_seed_rows,
                        candidate_seeds,
                        reference_seeds,
                        strict=True,
                    )
                )
            ]
            cohort_digests = {
                str(row["paired_task_cohort_sha256"]) for row in strict_seed_rows
            }
            if len(cohort_digests) != 1:
                raise CompletionSummaryError(
                    "seed-policy paired task cohort differs across split seeds"
                )
            document["seed_results"] = strict_seed_rows
            document["semantic_cell"] = _semantic_cell_document(semantic_cell)
            document["paired_task_cohort_sha256"] = next(iter(cohort_digests))
            task_identity = _development_task_identity_proof(
                diagnostic_index,
                source_name=source_name,
                condition_id=condition_id,
                experiment_id=experiment_id,
            )
            document["development_task_identity"] = task_identity
            all_seeds_pass = all(bool(row["seed_pass"]) for row in strict_seed_rows)
            task_identity_verified = bool(task_identity["verified"])
            document["replacement_rule"] = {
                "rule_id": REPLACEMENT_RULE_ID,
                "required_seed_count": len(FROZEN_SPLIT_SEEDS),
                "bootstrap_iterations_per_seed": SEED_POLICY_BOOTSTRAP_ITERATIONS,
                "primary_effect": (
                    "calibrated_weighted_interval_score_candidate_minus_reference"
                ),
                "coverage_tolerance": SEED_POLICY_COVERAGE_TOLERANCE,
                "coverage_metric": "task_simultaneous_coverage",
                "coverage_aggregation": "equal_weight_per_task_all_points_covered",
                "upper_tail_tolerance": SEED_POLICY_UPPER_TAIL_TOLERANCE,
                "mae_role": "parity_report_only_not_selection",
                "raw_forecast_interval_metric_role": (
                    "diagnostic_only_not_selection"
                ),
                "all_three_seeds_pass": all_seeds_pass,
                "split_independent_development_task_projection_verified": (
                    task_identity_verified
                ),
                "replace_reference": all_seeds_pass and task_identity_verified,
                "application_scope": "prospective_only",
                "parent_final_reselected": False,
            }
            documents.append(document)
        if used_plan_ids != set(plan_index):
            raise CompletionSummaryError(
                "seed-policy matrix plan lacks a 1:1 experiment result"
            )
    documents.sort(
        key=lambda item: (str(item["source_name"]), str(item["condition_id"]))
    )
    passing = sum(
        bool(item["replacement_rule"]["replace_reference"])  # type: ignore[index]
        for item in documents
    )
    missing_cells = EXPECTED_SEED_POLICY_CELLS - seen_cells
    condition_set_complete = seen_cells == EXPECTED_SEED_POLICY_CELLS
    overall = {
        "rule_id": REPLACEMENT_RULE_ID,
        "coverage_metric": "task_simultaneous_coverage",
        "coverage_aggregation": "equal_weight_per_task_all_points_covered",
        "coverage_tolerance": SEED_POLICY_COVERAGE_TOLERANCE,
        "expected_condition_count": EXPECTED_SEED_POLICY_CONDITION_COUNT,
        "condition_count": len(documents),
        "passing_condition_count": passing,
        "condition_set_complete": condition_set_complete,
        "observed_semantic_cells": [
            _semantic_cell_document(cell)
            for cell in sorted(seen_cells, key=lambda item: (item[0], item[2]))
        ],
        "missing_semantic_cells": [
            _semantic_cell_document(cell)
            for cell in sorted(missing_cells, key=lambda item: (item[0], item[2]))
        ],
        "all_conditions_pass": (
            condition_set_complete
            and passing == EXPECTED_SEED_POLICY_CONDITION_COUNT
        ),
        "decision": (
            "prospectively_replace_raw_repaired_reference"
            if condition_set_complete
            and passing == EXPECTED_SEED_POLICY_CONDITION_COUNT
            else "retain_raw_repaired_reference_for_prospective_runs"
        ),
        "application_scope": "prospective_only",
        "parent_final_reselected": False,
        "parent_final_selection_unchanged": True,
        "mae_role": "parity_report_only_not_selection",
        "raw_forecast_interval_metric_role": "diagnostic_only_not_selection",
    }
    return documents, overall


def _find_run_dispersion(
    value: object,
    *,
    path: str,
) -> list[tuple[str, Mapping[str, object]]]:
    found: list[tuple[str, Mapping[str, object]]] = []
    if isinstance(value, Mapping):
        if "run_dispersion_extension_id" in value:
            found.append((path, _mapping(value, name=path)))
        for key, item in value.items():
            if isinstance(key, str):
                found.extend(
                    _find_run_dispersion(item, path=f"{path}.{key}")
                )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(
                _find_run_dispersion(item, path=f"{path}[{index}]")
            )
    return found


def _is_lifecycle_candidate(candidate: Mapping[str, object]) -> bool:
    graph = _mapping(candidate.get("candidate_graph"), name="candidate graph")
    return graph.get("initializer_estimator_id") != "none"


def _diagnostics_index(
    artifacts: Sequence[LoadedArtifact],
    diagnostics: LoadedDiagnosticsArtifact | None,
) -> dict[tuple[str, str, str, int], Mapping[str, object]]:
    if diagnostics is None:
        return {}
    expected_artifacts: dict[str, tuple[str, str]] = {}
    for artifact in artifacts:
        source = _mapping(artifact.document.get("source"), name="artifact source")
        source_name = _required_string(
            source.get("source_name"), name="source.source_name"
        )
        expected_artifacts[source_name] = (
            artifact.artifact_id,
            artifact.results_payload_sha256,
        )
    declared: dict[str, tuple[str, str]] = {}
    for index, raw in enumerate(
        _list(
            diagnostics.document.get("source_artifacts"),
            name="diagnostics source_artifacts",
        )
    ):
        item = _mapping(raw, name=f"diagnostics source_artifacts[{index}]")
        declared[
            _required_string(item.get("source_name"), name="diagnostics source name")
        ] = (
            _required_sha256(
                item.get("artifact_id"), name="diagnostics source artifact id"
            ),
            _required_sha256(
                item.get("results_payload_sha256"),
                name="diagnostics source results hash",
            ),
        )
    if declared != expected_artifacts:
        raise CompletionSummaryError(
            "diagnostics supplement is bound to another source artifact set"
        )
    index: dict[tuple[str, str, str, int], Mapping[str, object]] = {}
    for item_index, raw in enumerate(
        _list(
            diagnostics.document.get("diagnostics"),
            name="completion diagnostics",
        )
    ):
        item = _mapping(raw, name=f"completion diagnostics[{item_index}]")
        seed = item.get("split_seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise CompletionSummaryError("completion diagnostic split seed is invalid")
        key = (
            _required_string(item.get("source_name"), name="diagnostic source"),
            _required_string(item.get("experiment_id"), name="diagnostic experiment"),
            _required_string(item.get("candidate_id"), name="diagnostic candidate"),
            seed,
        )
        if key in index:
            raise CompletionSummaryError("completion diagnostics repeat an identity")
        index[key] = item
    return index


def _metric_coverage(
    artifacts: Sequence[LoadedArtifact],
    diagnostics: LoadedDiagnosticsArtifact | None = None,
) -> dict[str, object]:
    diagnostics_by_key = _diagnostics_index(artifacts, diagnostics)
    consumed_diagnostics: set[tuple[str, str, str, int]] = set()
    artifact_rows: list[dict[str, object]] = []
    total_seed_results = 0
    interval_complete = 0
    total_lifecycle_seed_results = 0
    run_dispersion_complete = 0
    run_dispersion_declared_unavailable = 0
    for artifact in artifacts:
        source = _mapping(artifact.document.get("source"), name="artifact source")
        source_name = _required_string(
            source.get("source_name"), name="source.source_name"
        )
        interval_missing: list[dict[str, object]] = []
        run_missing: list[dict[str, object]] = []
        artifact_seed_results = 0
        artifact_interval_complete = 0
        artifact_lifecycle_seed_results = 0
        artifact_run_complete = 0
        artifact_run_unavailable = 0
        experiments = _list(
            artifact.document.get("experiments"), name="artifact experiments"
        )
        for experiment_index, raw_experiment in enumerate(experiments):
            experiment = _mapping(
                raw_experiment, name=f"experiments[{experiment_index}]"
            )
            experiment_id = _required_string(
                experiment.get("experiment_id"),
                name=f"experiments[{experiment_index}].experiment_id",
            )
            candidates = _candidate_map(
                experiment, name=f"experiments[{experiment_index}]"
            )
            for candidate_id, candidate in candidates.items():
                lifecycle = _is_lifecycle_candidate(candidate)
                for seed in _seed_results(
                    candidate, name=f"{experiment_id}.{candidate_id}"
                ):
                    artifact_seed_results += 1
                    metrics = _mapping(
                        seed.get("metrics"),
                        name=f"{experiment_id}.{candidate_id}.metrics",
                    )
                    missing_interval = [
                        field for field in INTERVAL_RESERVE_FIELDS if field not in metrics
                    ]
                    if missing_interval:
                        interval_missing.append(
                            {
                                "experiment_id": experiment_id,
                                "candidate_id": candidate_id,
                                "split_seed": seed["split_seed"],
                                "missing_fields": missing_interval,
                            }
                        )
                    else:
                        artifact_interval_complete += 1
                    if lifecycle:
                        artifact_lifecycle_seed_results += 1
                        found = _find_run_dispersion(seed, path="seed_result")
                        diagnostic_key = (
                            source_name,
                            experiment_id,
                            candidate_id,
                            int(seed["split_seed"]),
                        )
                        supplement = diagnostics_by_key.get(diagnostic_key)
                        if supplement is not None:
                            if "lifecycle_metrics" in supplement:
                                lifecycle_metrics = _mapping(
                                    supplement.get("lifecycle_metrics"),
                                    name="completion diagnostic lifecycle metrics",
                                )
                                if (
                                    lifecycle_metrics.get("status") != "unavailable"
                                    or lifecycle_metrics.get("reason_code")
                                    != DIAGNOSTICS_LIFECYCLE_UNAVAILABLE_REASON
                                    or lifecycle_metrics.get("labels_present") is not False
                                    or lifecycle_metrics.get(
                                        "lifecycle_sequences_present"
                                    )
                                    is not False
                                    or lifecycle_metrics.get("unavailable_metrics")
                                    != DIAGNOSTICS_UNAVAILABLE_LIFECYCLE_METRICS
                                    or lifecycle_metrics.get(
                                        "historical_stage3_reference"
                                    )
                                    is not None
                                ):
                                    raise CompletionSummaryError(
                                        "completion diagnostic unavailable "
                                        "lifecycle declaration differs"
                                    )
                                if not found:
                                    artifact_run_unavailable += 1
                            else:
                                supplement_variance = _mapping(
                                    supplement.get("run_variance"),
                                    name="completion diagnostic run variance",
                                )
                                if found and found[0][1] != supplement_variance:
                                    raise CompletionSummaryError(
                                        "source and supplement run dispersion differ"
                                    )
                                if not found:
                                    found = [
                                        (
                                            "completion_diagnostics.run_variance",
                                            supplement_variance,
                                        )
                                    ]
                            consumed_diagnostics.add(diagnostic_key)
                        if len(found) > 1:
                            raise CompletionSummaryError(
                                "lifecycle seed contains multiple run-dispersion documents"
                            )
                        missing_run = (
                            list(RUN_DISPERSION_FIELDS)
                            if not found
                            else [
                                field
                                for field in RUN_DISPERSION_FIELDS
                                if field not in found[0][1]
                            ]
                        )
                        if missing_run:
                            run_missing.append(
                                {
                                    "experiment_id": experiment_id,
                                    "candidate_id": candidate_id,
                                    "split_seed": seed["split_seed"],
                                    "document_path": found[0][0] if found else None,
                                    "missing_fields": missing_run,
                                    "availability_status": (
                                        "declared_unavailable"
                                        if supplement is not None
                                        and "lifecycle_metrics" in supplement
                                        else "missing"
                                    ),
                                }
                            )
                        else:
                            artifact_run_complete += 1
        total_seed_results += artifact_seed_results
        interval_complete += artifact_interval_complete
        total_lifecycle_seed_results += artifact_lifecycle_seed_results
        run_dispersion_complete += artifact_run_complete
        run_dispersion_declared_unavailable += artifact_run_unavailable
        artifact_rows.append(
            {
                "artifact_id": artifact.artifact_id,
                "run_id": artifact.document["run_id"],
                "source_name": source_name,
                "candidate_seed_result_count": artifact_seed_results,
                "interval_reserve_complete_count": artifact_interval_complete,
                "interval_reserve_missing": interval_missing,
                "lifecycle_candidate_seed_result_count": (
                    artifact_lifecycle_seed_results
                ),
                "run_dispersion_complete_count": artifact_run_complete,
                "run_dispersion_declared_unavailable_count": (
                    artifact_run_unavailable
                ),
                "run_dispersion_missing": run_missing,
            }
        )
    if consumed_diagnostics != set(diagnostics_by_key):
        extra = sorted(set(diagnostics_by_key) - consumed_diagnostics)
        raise CompletionSummaryError(
            f"completion diagnostics do not match lifecycle seed results: {extra[:3]}"
        )
    return {
        "interval_reserve": {
            "required_fields": list(INTERVAL_RESERVE_FIELDS),
            "complete_count": interval_complete,
            "expected_count": total_seed_results,
            "complete": interval_complete == total_seed_results,
        },
        "repeated_run_dispersion": {
            "required_fields": list(RUN_DISPERSION_FIELDS),
            "complete_count": run_dispersion_complete,
            "expected_lifecycle_count": total_lifecycle_seed_results,
            "complete": run_dispersion_complete == total_lifecycle_seed_results,
            "declared_unavailable_count": run_dispersion_declared_unavailable,
            "status": (
                "complete"
                if run_dispersion_complete == total_lifecycle_seed_results
                else "declared_unavailable"
                if (
                    total_lifecycle_seed_results > 0
                    and run_dispersion_declared_unavailable
                    == total_lifecycle_seed_results
                )
                else "incomplete"
            ),
            "supplement_artifact_id": (
                diagnostics.artifact_id if diagnostics is not None else None
            ),
        },
        "artifacts": artifact_rows,
    }


def build_completion_summary(
    artifacts: Sequence[LoadedArtifact],
    *,
    diagnostics: LoadedDiagnosticsArtifact | None = None,
) -> dict[str, object]:
    """Build the machine-readable completion comparison summary."""

    if not artifacts:
        raise CompletionSummaryError("cannot summarize an empty artifact set")
    artifact_ids = [artifact.artifact_id for artifact in artifacts]
    if len(set(artifact_ids)) != len(artifact_ids):
        raise CompletionSummaryError("development artifact ids must be unique")
    call_pre = _call_pre_comparisons(artifacts)
    diagnostics_by_key = _diagnostics_index(artifacts, diagnostics)
    seed_policy, replacement = _seed_policy_comparisons(
        artifacts,
        diagnostics_by_key=diagnostics_by_key,
    )
    coverage = _metric_coverage(artifacts, diagnostics)
    sources = []
    for artifact in artifacts:
        source = _mapping(artifact.document.get("source"), name="artifact source")
        sources.append(
            {
                "source_name": _required_string(
                    source.get("source_name"), name="source.source_name"
                ),
                "run_id": artifact.document["run_id"],
                "artifact_id": artifact.artifact_id,
                "results_payload_sha256": artifact.results_payload_sha256,
            }
        )
    return {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "policy_id": SUMMARY_POLICY_ID,
        "final_holdout_access": {
            "policy_id": FINAL_HOLDOUT_POLICY_ID,
            "attestation_scope": "this_summary_reader_process_only",
            "allowed": False,
            "accessed": False,
            "summary_reader_opened_final_artifacts": False,
            "summary_reader_opened_source_data": False,
            "summary_reader_opened_final_labels": False,
            "artifact_count_checked": len(artifacts),
            "all_development_documents_closed": True,
            "development_result_attestation": {
                "final_target_values_used_for_fit_calibration_scoring": False,
            },
            "upstream_loader_audit_disclosure": {
                "scope": "development_artifact_production_before_this_summary",
                "mixed_raw_source_loader_behavior": (
                    "source payloads were parsed before the development/final "
                    "task filter was applied"
                ),
                "raw_final_task_records_may_have_been_parsed": True,
                "claim_about_upstream_final_source_access": "none",
            },
            "statement": (
                "This summary reader opened only aggregate-safe Stage 4 "
                "development results. Its inputs attest that final target "
                "values were not used for fit, calibration, or scoring. "
                "Upstream artifact production used a mixed raw loader that "
                "parsed source payloads before task filtering."
            ),
        },
        "sources": sorted(sources, key=lambda item: str(item["source_name"])),
        "call_pre_mlp_vs_lightgbm": call_pre,
        "seed_policy_point_only_vs_raw_repaired": seed_policy,
        "seed_policy_frozen_replacement_rule": replacement,
        "metric_coverage": coverage,
        "diagnostics_supplement": (
            {
                "artifact_id": diagnostics.artifact_id,
                "results_payload_sha256": diagnostics.results_payload_sha256,
            }
            if diagnostics is not None
            else None
        ),
        "completion_status": {
            "has_call_pre_comparisons": bool(call_pre),
            "has_seed_policy_comparisons": bool(seed_policy),
            "metric_coverage_complete": bool(
                coverage["interval_reserve"]["complete"]  # type: ignore[index]
                and coverage["repeated_run_dispersion"]["complete"]  # type: ignore[index]
            ),
        },
    }


def _compact_seed_rows(rows: Iterable[Mapping[str, object]]) -> str:
    parts = []
    for row in rows:
        parts.append(
            (
                f"{row['split_seed']}: delta={float(row['mae_delta']):.3f}, "
                f"CI=[{float(row['mae_delta_ci_lower']):.3f},"
                f"{float(row['mae_delta_ci_upper']):.3f}], "
                f"{row['bootstrap_outcome']}"
            )
        )
    return "<br>".join(parts)


def _compact_seed_policy_rows(rows: Iterable[Mapping[str, object]]) -> str:
    parts = []
    for row in rows:
        parts.append(
            (
                f"{row['split_seed']}: WIS delta="
                f"{float(row['weighted_interval_score_delta']):.3f}, "
                f"CI=[{float(row['interval_score_delta_ci_lower']):.3f},"
                f"{float(row['interval_score_delta_ci_upper']):.3f}], "
                "task simultaneous coverage delta="
                f"{float(row['task_simultaneous_coverage_delta']):.3f}, "
                f"{'pass' if row['seed_pass'] else 'fail'}; "
                f"MAE delta={float(row['mae_delta']):.3f} (parity only)"
            )
        )
    return "<br>".join(parts)


def render_markdown(summary: Mapping[str, object]) -> str:
    """Render a compact human-readable view of the machine summary."""

    sources = _list(summary.get("sources"), name="summary.sources")
    call_pre = _list(
        summary.get("call_pre_mlp_vs_lightgbm"),
        name="summary.call_pre_mlp_vs_lightgbm",
    )
    seed_policy = _list(
        summary.get("seed_policy_point_only_vs_raw_repaired"),
        name="summary.seed_policy_point_only_vs_raw_repaired",
    )
    replacement = _mapping(
        summary.get("seed_policy_frozen_replacement_rule"),
        name="summary.seed_policy_frozen_replacement_rule",
    )
    coverage = _mapping(
        summary.get("metric_coverage"), name="summary.metric_coverage"
    )
    interval = _mapping(
        coverage.get("interval_reserve"), name="metric_coverage.interval_reserve"
    )
    dispersion = _mapping(
        coverage.get("repeated_run_dispersion"),
        name="metric_coverage.repeated_run_dispersion",
    )
    lines = [
        "# Stage 4 completion development summary",
        "",
        f"- Development artifacts: {len(sources)}",
        (
            "- Final holdout (summary-reader scope): final artifacts, source "
            "data, and final labels were not opened by this reader; development "
            "results attest final target values were not used for fit, "
            "calibration, or scoring."
        ),
        (
            "- Upstream loader audit: development artifact production parsed "
            "mixed raw source payloads before applying the task filter."
        ),
        "",
        "## Call-pre MLP vs LightGBM",
        "",
    ]
    if call_pre:
        lines.extend(
            [
                "| Source | Condition | Target | Per-seed paired MAE result |",
                "|---|---|---|---|",
            ]
        )
        for raw in call_pre:
            row = _mapping(raw, name="Call-pre comparison")
            lines.append(
                "| {source} | `{condition}` | `{target}` | {seeds} |".format(
                    source=row["source_name"],
                    condition=row["condition_id"],
                    target=row["target"],
                    seeds=_compact_seed_rows(
                        _mapping(item, name="Call-pre seed")
                        for item in _list(
                            row.get("seed_results"),
                            name="Call-pre comparison.seed_results",
                        )
                    ),
                )
            )
    else:
        lines.append("No Call-pre MLP comparison was present.")
    lines.extend(
        [
            "",
            "## Seed policy",
            "",
            (
                "Coverage guard: equal-weight `task_simultaneous_coverage`; "
                "a task is covered only when all of its positive-weight points "
                "are covered."
            ),
            "",
        ]
    )
    if seed_policy:
        lines.extend(
            [
                "| Source | Condition | Per-seed WIS + task-coverage rule | Prospective rule |",
                "|---|---|---|---|",
            ]
        )
        for raw in seed_policy:
            row = _mapping(raw, name="seed-policy comparison")
            rule = _mapping(
                row.get("replacement_rule"),
                name="seed-policy comparison.replacement_rule",
            )
            lines.append(
                "| {source} | `{condition}` | {seeds} | {decision} |".format(
                    source=row["source_name"],
                    condition=row["condition_id"],
                    seeds=_compact_seed_policy_rows(
                        _mapping(item, name="seed-policy seed")
                        for item in _list(
                            row.get("seed_results"),
                            name="seed-policy comparison.seed_results",
                        )
                    ),
                    decision=(
                        "replace" if rule["replace_reference"] else "retain reference"
                    ),
                )
            )
    else:
        lines.append("No seed-policy comparison was present.")
    lines.extend(
        [
            "",
            (
                f"Prospective replacement rule: `{replacement['decision']}` "
                f"({replacement['passing_condition_count']}/"
                f"{replacement['expected_condition_count']} required conditions "
                "passed); the parent final selection is unchanged."
            ),
            "",
            "## Metric-field coverage",
            "",
            "| Group | Complete / expected | Status |",
            "|---|---:|---|",
            (
                f"| Interval/reserve | {interval['complete_count']} / "
                f"{interval['expected_count']} | "
                f"{'complete' if interval['complete'] else 'incomplete'} |"
            ),
            (
                f"| Repeated-run dispersion (lifecycle only) | "
                f"{dispersion['complete_count']} / "
                f"{dispersion['expected_lifecycle_count']} | "
                f"{str(dispersion['status']).replace('_', ' ')} |"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize four Stage 4 development artifacts without opening "
            "the final holdout."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "development artifact directories/results.json files, or one "
            "canonical configs/stage4_completion_release.json lock"
        ),
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="stdout format (the command never writes artifact or report files)",
    )
    parser.add_argument(
        "--diagnostics-artifact",
        help=(
            "optional immutable completion diagnostics artifact used to close "
            "lifecycle repeated-run coverage"
        ),
    )
    parser.add_argument(
        "--expected-artifacts",
        type=int,
        default=4,
        help="required distinct development artifact count; use 0 to disable",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        references = resolve_artifact_references(args.inputs)
        if args.expected_artifacts < 0:
            raise CompletionSummaryError("--expected-artifacts cannot be negative")
        if args.expected_artifacts and len(references) != args.expected_artifacts:
            raise CompletionSummaryError(
                f"expected {args.expected_artifacts} development artifacts, "
                f"found {len(references)}"
            )
        artifacts = tuple(load_development_artifact(item) for item in references)
        diagnostics = (
            load_completion_diagnostics_artifact(args.diagnostics_artifact)
            if args.diagnostics_artifact
            else None
        )
        summary = build_completion_summary(
            artifacts,
            diagnostics=diagnostics,
        )
    except CompletionSummaryError as exc:
        print(f"completion summary failed: {exc}", file=sys.stderr)
        return 2
    if args.format == "markdown":
        print(render_markdown(summary))
    else:
        print(
            json.dumps(
                summary,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
