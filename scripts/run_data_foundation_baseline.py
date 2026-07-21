from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Iterable, Mapping, Sequence

from token_prediction.collection import (
    BagenSwebenchReader,
    OpenHandsArchiveMetadata,
    OpenHandsArchiveReader,
)
from token_prediction.contracts import SourceCapabilities, SourceDescriptor
from token_prediction.dataset import (
    CAPABILITY_DATASET_SCHEMA_VERSION,
    DatasetRow,
    DatasetSlice,
    PredictionPosition,
    PredictionTarget,
    SplitPlan,
    SupervisedDataset,
    build_capability_supervised_dataset,
)
from token_prediction.estimators import (
    EmpiricalQuantileEstimator,
    FitContext,
    RunContext,
    TokenForecast,
    TrainingExample,
    TrainingView,
)
from token_prediction.evaluation import (
    METRIC_SUITE_ID,
    CalibrationExample,
    ScoredForecast,
    TaskMaxConformalCalibrator,
    evaluate_forecasts,
)


BASELINE_ARTIFACT_SCHEMA_VERSION = 1
EMPIRICAL_BUNDLE_SCHEMA_VERSION = 1
BASELINE_ID = "data_foundation_empirical_development_v2"
FOLDS = 5
SPLIT_SEEDS = (20260719, 20260720, 20260721)
ALPHA = 0.10
CALIBRATOR_ID = "task_max_conformal"
CANDIDATE_ID = "empirical_quantile"
SPLIT_ASSIGNMENT_POLICY_ID = "pseudonymous_task_sha256_rank_round_robin_v1"
FINAL_HOLDOUT_POLICY_ID = "stable_task_sha256_bucket_v1"
FINAL_HOLDOUT_SALT = "token-prediction/final-holdout/2026-07-21/v1"
FINAL_HOLDOUT_BUCKET_COUNT = 10_000
FINAL_HOLDOUT_BUCKET_THRESHOLD = 2_000
MIN_DEVELOPMENT_TASKS_PER_CONDITION = 10
FROZEN_BAGEN_ESTIMABLE_CONDITIONS = frozenset(
    {
        "condition:54cb50fce273f0aa2d74",
        "condition:949ac3b7a342718cd505",
        "condition:d94078c05d91b0d58aee",
        "condition:dce86ced00dc11c77205",
        "condition:f95ae2a5e11682f6b7fc",
    }
)
FROZEN_BAGEN_NOT_ESTIMABLE_CONDITIONS = frozenset(
    {
        "condition:20f615a22697984db6cc",
        "condition:562b4f6934238e459db9",
        "condition:686d78e7865f5e646e0b",
        "condition:8fe0be8b5f924006a166",
    }
)
FROZEN_SPEND_CONDITIONS = frozenset({"condition:b407e0d1ec34f386ebc4"})
RUNNER_RELATIVE = "scripts/run_data_foundation_baseline.py"
DEFAULT_BASELINE_LOCK = "configs/data_foundation_v2_baseline.json"
DEFAULT_OUTPUT = (
    "workspace/data_foundation/baselines/empirical-development-v2"
)
OUTPUT_PREFIX = "workspace/data_foundation/baselines/"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class DataFoundationBaselineError(RuntimeError):
    """The prediction baseline cannot be built or verified safely."""


@dataclass(frozen=True)
class CodeBinding:
    git_commit: str
    source_tree_sha256: str
    paths: tuple[str, ...]


@dataclass(frozen=True)
class FileSnapshot:
    path: str
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class SourceLock:
    name: str
    descriptor_path: str
    descriptor_file_sha256: str
    descriptor: SourceDescriptor
    manifest_path: str
    manifest_sha256: str
    dataset_id: str
    dataset_row_count: int
    raw_artifact_path: str
    raw_artifact_sha256: str
    raw_artifact_sha256_kind: str
    raw_artifact_bytes: int
    raw_file_index: tuple[tuple[str, int, str], ...] = ()


@dataclass(frozen=True)
class LockContext:
    baseline_lock_path: str
    baseline_lock_file_sha256: str
    audit_path: str
    audit_file_sha256: str
    audit_payload_sha256: str
    audit_git_commit: str
    audit_source_tree_sha256: str
    sources: Mapping[str, SourceLock]


@dataclass(frozen=True)
class HoldoutPlan:
    source_dataset_id: str
    development_dataset_id: str
    assignment_digest: str
    development_tasks: frozenset[str]
    final_holdout_tasks: frozenset[str]

    def to_evidence(self) -> dict[str, Any]:
        return {
            "policy_id": FINAL_HOLDOUT_POLICY_ID,
            "salt": FINAL_HOLDOUT_SALT,
            "bucket_count": FINAL_HOLDOUT_BUCKET_COUNT,
            "final_holdout_bucket_threshold_exclusive": (
                FINAL_HOLDOUT_BUCKET_THRESHOLD
            ),
            "source_dataset_id": self.source_dataset_id,
            "development_dataset_id": self.development_dataset_id,
            "assignment_digest": self.assignment_digest,
            "development_task_count": len(self.development_tasks),
            "final_holdout_task_count": len(self.final_holdout_tasks),
            "development_task_set_sha256": _task_hash(self.development_tasks),
            "final_holdout_task_set_sha256": _task_hash(self.final_holdout_tasks),
        }


@dataclass(frozen=True)
class ReloadedEmpiricalBundle:
    target: PredictionTarget
    lower: float
    point: float
    upper: float
    expansion: float

    def predict(self, point_id: str) -> TokenForecast:
        return TokenForecast(
            point_id=point_id,
            target=self.target,
            lower=max(0.0, self.lower - self.expansion),
            point=self.point,
            upper=max(self.point, self.upper + self.expansion),
            raw_lower=self.lower,
            raw_point=self.point,
            raw_upper=self.upper,
        )


def _canonical_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DataFoundationBaselineError(f"value is not canonical JSON: {exc}") from exc
    return (rendered + "\n").encode("utf-8")


def _semantic_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DataFoundationBaselineError(f"cannot hash {path.name!r}") from exc
    return digest.hexdigest()


def _reject_json_constant(value: str) -> Any:
    raise DataFoundationBaselineError(f"non-finite JSON constant is forbidden: {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DataFoundationBaselineError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _reject_non_finite(value: Any, *, label: str) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DataFoundationBaselineError(f"{label} contains a non-finite number")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _reject_non_finite(item, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_finite(item, label=f"{label}[{index}]")


def _strict_json_loads(payload: str, *, label: str) -> Any:
    try:
        result = json.loads(
            payload,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except DataFoundationBaselineError:
        raise
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise DataFoundationBaselineError(f"{label} is not strict JSON") from exc
    _reject_non_finite(result, label=label)
    return result


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or _is_link_or_reparse(path):
        raise DataFoundationBaselineError(f"{label} must be one regular file")
    try:
        value = _strict_json_loads(path.read_text(encoding="utf-8"), label=label)
    except (OSError, UnicodeError) as exc:
        raise DataFoundationBaselineError(f"cannot read {label}") from exc
    if not isinstance(value, dict):
        raise DataFoundationBaselineError(f"{label} must contain one JSON object")
    return value


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataFoundationBaselineError(f"{label} must be an object")
    return value


def _require_list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise DataFoundationBaselineError(f"{label} must be a list")
    return value


def _require_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataFoundationBaselineError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    result = _require_text(value, label=label)
    if not SHA256_RE.fullmatch(result):
        raise DataFoundationBaselineError(f"{label} must be a lowercase SHA-256")
    return result


def _require_commit(value: Any, *, label: str) -> str:
    result = _require_text(value, label=label)
    if not COMMIT_RE.fullmatch(result):
        raise DataFoundationBaselineError(f"{label} must be a full lowercase Git commit")
    return result


def _require_non_negative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DataFoundationBaselineError(f"{label} must be a non-negative integer")
    return value


def _require_finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataFoundationBaselineError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise DataFoundationBaselineError(f"{label} must be a finite number")
    return result


def _safe_relative(value: Any, *, label: str) -> str:
    raw = _require_text(value, label=label)
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if (
        "\\" in raw
        or "\x00" in raw
        or raw != raw.strip()
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or posix.as_posix() != raw
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise DataFoundationBaselineError(f"{label} must be a canonical relative POSIX path")
    return raw


def _is_link_or_reparse(path: Path) -> bool:
    try:
        status = path.lstat()
    except (FileNotFoundError, NotADirectoryError):
        return False
    if stat.S_ISLNK(status.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if reparse_flag and getattr(status, "st_file_attributes", 0) & reparse_flag:
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _assert_tree_no_links(root: Path, *, label: str) -> None:
    if _is_link_or_reparse(root):
        raise DataFoundationBaselineError(f"{label} root must not be linked or reparse-backed")
    for directory, names, files in os.walk(root, topdown=True, followlinks=False):
        parent = Path(directory)
        for name in [*names, *files]:
            if _is_link_or_reparse(parent / name):
                raise DataFoundationBaselineError(
                    f"{label} must not contain symlinks, junctions, or reparse points"
                )


def _repo_path(root: Path, relative: Any, *, label: str) -> Path:
    canonical = _safe_relative(relative, label=label)
    resolved_root = root.resolve()
    path = resolved_root.joinpath(*PurePosixPath(canonical).parts)
    current = resolved_root
    if _is_link_or_reparse(current):
        raise DataFoundationBaselineError("repository root must not be linked or reparse-backed")
    for part in PurePosixPath(canonical).parts:
        current /= part
        if _is_link_or_reparse(current):
            raise DataFoundationBaselineError(
                f"{label} must not traverse a symlink, junction, or reparse point"
            )
    try:
        path.resolve(strict=False).relative_to(resolved_root)
    except ValueError as exc:
        raise DataFoundationBaselineError(f"{label} escapes the repository") from exc
    return path


def _git_executable() -> str:
    discovered = shutil.which("git")
    if discovered:
        return discovered
    if os.name == "nt":
        candidate = (
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "Git"
            / "cmd"
            / "git.exe"
        )
        if candidate.is_file():
            return str(candidate)
    raise DataFoundationBaselineError("Git is required for source binding")


def _git(root: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            [_git_executable(), "-C", str(root), *arguments],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DataFoundationBaselineError("Git source-binding command failed") from exc
    return result.stdout


def _code_paths_at_head(root: Path) -> tuple[str, ...]:
    raw = _git(
        root,
        "ls-tree",
        "-r",
        "--name-only",
        "-z",
        "HEAD",
        "--",
        "src/token_prediction",
        RUNNER_RELATIVE,
    )
    paths: list[str] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        try:
            relative = item.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise DataFoundationBaselineError("Git returned a non-UTF-8 path") from exc
        relative = _safe_relative(relative, label="tracked source path")
        if relative == RUNNER_RELATIVE or (
            relative.startswith("src/token_prediction/") and relative.endswith(".py")
        ):
            paths.append(relative)
    paths = sorted(set(paths))
    if RUNNER_RELATIVE not in paths or not any(
        path.startswith("src/token_prediction/") for path in paths
    ):
        raise DataFoundationBaselineError(
            "HEAD must track the runner and token_prediction Python source tree"
        )
    return tuple(paths)


def _framed_code_hash(items: Iterable[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256(b"data-foundation-prediction-code-tree-v1\0")
    for relative, payload in items:
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def capture_code_binding(root: Path) -> CodeBinding:
    resolved_root = root.resolve()
    commit = _require_commit(
        _git(resolved_root, "rev-parse", "--verify", "HEAD^{commit}")
        .decode("ascii", errors="strict")
        .strip(),
        label="HEAD",
    )
    paths = _code_paths_at_head(resolved_root)
    status = _git(
        resolved_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--",
        "src/token_prediction",
        RUNNER_RELATIVE,
    )
    if status:
        raise DataFoundationBaselineError(
            "runner and token_prediction source must be clean at HEAD"
        )
    workspace_items: list[tuple[str, bytes]] = []
    commit_items: list[tuple[str, bytes]] = []
    for relative in paths:
        workspace_path = _repo_path(resolved_root, relative, label="source path")
        if not workspace_path.is_file() or _is_link_or_reparse(workspace_path):
            raise DataFoundationBaselineError("source binding contains a non-regular file")
        workspace_items.append((relative, workspace_path.read_bytes()))
        commit_items.append((relative, _git(resolved_root, "show", f"HEAD:{relative}")))
    workspace_hash = _framed_code_hash(workspace_items)
    if workspace_hash != _framed_code_hash(commit_items):
        raise DataFoundationBaselineError("workspace source bytes do not match HEAD blobs")
    return CodeBinding(commit, workspace_hash, paths)


def verify_execution_origin(root: Path) -> None:
    resolved_root = root.resolve()
    expected_runner = _repo_path(
        resolved_root, RUNNER_RELATIVE, label="executing runner path"
    )
    actual_runner = Path(__file__)
    if _is_link_or_reparse(actual_runner) or actual_runner.resolve() != expected_runner.resolve():
        raise DataFoundationBaselineError(
            "executing baseline runner does not originate from repository_root"
        )
    package_root = _repo_path(
        resolved_root, "src/token_prediction", label="imported package root"
    ).resolve()
    required_modules = {
        "token_prediction.contracts",
        "token_prediction.collection",
        "token_prediction.dataset",
        "token_prediction.estimators",
        "token_prediction.evaluation",
    }
    if not required_modules <= set(sys.modules):
        raise DataFoundationBaselineError("required token_prediction modules are not imported")
    for name, module in sorted(sys.modules.items()):
        if name != "token_prediction" and not name.startswith("token_prediction."):
            continue
        origin = getattr(module, "__file__", None)
        if origin is None:
            continue
        origin_path = Path(origin)
        if _is_link_or_reparse(origin_path):
            raise DataFoundationBaselineError("imported token_prediction module is linked")
        try:
            origin_path.resolve().relative_to(package_root)
        except ValueError as exc:
            raise DataFoundationBaselineError(
                "imported token_prediction module does not originate from repository_root"
            ) from exc


def tracked_control_tree_sha256(
    root: Path,
    relatives: Iterable[str],
    *,
    git_commit: str,
) -> str:
    resolved_root = root.resolve()
    paths = tuple(sorted({_safe_relative(value, label="tracked control path") for value in relatives}))
    if not paths:
        raise DataFoundationBaselineError("tracked control path set is empty")
    status = _git(
        resolved_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--",
        *paths,
    )
    if status:
        raise DataFoundationBaselineError("baseline lock/descriptors must be tracked-clean at HEAD")
    digest = hashlib.sha256(b"data-foundation-baseline-controls-v1\0")
    for relative in paths:
        tracked = _git(
            resolved_root,
            "ls-files",
            "--error-unmatch",
            "--",
            relative,
        )
        try:
            tracked_path = tracked.decode("utf-8", errors="strict").strip()
        except UnicodeDecodeError as exc:
            raise DataFoundationBaselineError("Git returned a non-UTF-8 control path") from exc
        if tracked_path != relative:
            raise DataFoundationBaselineError("tracked control path identity differs")
        path = _repo_path(resolved_root, relative, label="tracked control path")
        workspace = path.read_bytes()
        committed = _git(resolved_root, "show", f"{git_commit}:{relative}")
        if workspace != committed:
            raise DataFoundationBaselineError("tracked control bytes differ from HEAD")
        for framed in (relative.encode("utf-8"), workspace):
            digest.update(len(framed).to_bytes(8, byteorder="big", signed=False))
            digest.update(framed)
    return digest.hexdigest()


def audit_compatible_source_tree_sha256(root: Path) -> str:
    """Recompute the exact source-tree identity used by the frozen v2 audit."""

    resolved = root.resolve()
    paths = sorted(
        [
            path.relative_to(resolved).as_posix()
            for path in (resolved / "src" / "token_prediction").rglob("*.py")
            if path.is_file()
        ]
        + ["scripts/audit_data_foundation_v2.py"]
    )
    if "scripts/audit_data_foundation_v2.py" not in paths or len(paths) < 2:
        raise DataFoundationBaselineError("audit-compatible source tree is incomplete")
    digest = hashlib.sha256(b"data-foundation-source-tree-v1\0")
    for relative in paths:
        path = _repo_path(resolved, relative, label="audit-compatible source path")
        if not path.is_file() or _is_link_or_reparse(path):
            raise DataFoundationBaselineError("audit-compatible source path is unsafe")
        for framed in (
            relative.encode("utf-8"),
            bytes.fromhex(_sha256_file(path)),
        ):
            digest.update(len(framed).to_bytes(8, byteorder="big", signed=False))
            digest.update(framed)
    return digest.hexdigest()


def _verify_embedded_payload(value: Mapping[str, Any], *, field: str, label: str) -> str:
    declared = _require_sha256(value.get(field), label=f"{label}.{field}")
    payload = dict(value)
    payload.pop(field)
    if _semantic_sha256(payload) != declared:
        raise DataFoundationBaselineError(f"{label} embedded payload hash is invalid")
    return declared


def _load_descriptor(
    root: Path,
    relative: str,
    *,
    expected_sha256: str,
    expected_source_id: str,
    expected_capabilities: SourceCapabilities,
) -> SourceDescriptor:
    path = _repo_path(root, relative, label="descriptor path")
    if _sha256_file(path) != expected_sha256:
        raise DataFoundationBaselineError("descriptor file SHA-256 does not match its lock")
    payload = _load_json(path, label="source descriptor")
    try:
        descriptor = SourceDescriptor.from_dict(payload)
    except (TypeError, ValueError) as exc:
        raise DataFoundationBaselineError("source descriptor is invalid") from exc
    if descriptor.to_dict() != payload:
        raise DataFoundationBaselineError("source descriptor is not in canonical schema form")
    if descriptor.source_id != expected_source_id:
        raise DataFoundationBaselineError("descriptor source_id disagrees with active reader")
    if descriptor.capabilities != expected_capabilities:
        raise DataFoundationBaselineError(
            "descriptor capabilities disagree with active reader"
        )
    return descriptor


def _framed_file_index_sha256(
    entries: Sequence[tuple[str, int, str]],
) -> tuple[str, int]:
    if not entries:
        raise DataFoundationBaselineError("raw file index must not be empty")
    digest = hashlib.sha256(b"data-foundation-file-index-v1\0")
    total_bytes = 0
    seen: set[str] = set()
    for relative, size, sha256 in sorted(entries):
        relative = _safe_relative(relative, label="raw file index path")
        if relative in seen:
            raise DataFoundationBaselineError("raw file index repeats a path")
        seen.add(relative)
        size = _require_non_negative_int(size, label="raw file index bytes")
        sha256 = _require_sha256(sha256, label="raw file index SHA-256")
        total_bytes += size
        for framed in (
            relative.encode("utf-8"),
            str(size).encode("ascii"),
            bytes.fromhex(sha256),
        ):
            digest.update(len(framed).to_bytes(8, byteorder="big", signed=False))
            digest.update(framed)
    return digest.hexdigest(), total_bytes


def _load_bagen_raw_file_index(
    root: Path,
    artifacts: Mapping[str, Any],
    raw_artifact: Mapping[str, Any],
) -> tuple[tuple[str, int, str], ...]:
    combined_artifact = _require_mapping(
        artifacts.get("combined_audit"), label="BAGEN combined audit artifact"
    )
    combined_path = _repo_path(
        root,
        _safe_relative(
            combined_artifact.get("path"), label="BAGEN combined audit path"
        ),
        label="BAGEN combined audit path",
    )
    combined_sha = _require_sha256(
        combined_artifact.get("sha256"), label="BAGEN combined audit SHA-256"
    )
    combined_bytes = _require_non_negative_int(
        combined_artifact.get("bytes"), label="BAGEN combined audit bytes"
    )
    if (
        _sha256_file(combined_path) != combined_sha
        or combined_path.stat().st_size != combined_bytes
    ):
        raise DataFoundationBaselineError("BAGEN combined audit identity does not close")
    combined = _load_json(combined_path, label="BAGEN combined audit")
    _verify_embedded_payload(
        combined, field="audit_payload_sha256", label="BAGEN combined audit"
    )
    families = _require_list(combined.get("families"), label="BAGEN combined families")
    if len(families) != 5:
        raise DataFoundationBaselineError("BAGEN combined audit must contain five families")
    entries: list[tuple[str, int, str]] = []
    for index, family_value in enumerate(families):
        family = _require_mapping(family_value, label=f"BAGEN family {index}")
        family_root = _safe_relative(
            family.get("local_relative_root"), label="BAGEN family root"
        )
        audit_relative = _safe_relative(
            family.get("audit_path"), label="BAGEN family audit path"
        )
        audit_path = _repo_path(root, audit_relative, label="BAGEN family audit path")
        audit_sha = _require_sha256(
            family.get("audit_sha256"), label="BAGEN family audit SHA-256"
        )
        audit_bytes = _require_non_negative_int(
            family.get("audit_bytes"), label="BAGEN family audit bytes"
        )
        if _sha256_file(audit_path) != audit_sha or audit_path.stat().st_size != audit_bytes:
            raise DataFoundationBaselineError("BAGEN family audit identity does not close")
        audit = _load_json(audit_path, label="BAGEN family audit")
        raw_files = _require_list(audit.get("raw_files"), label="BAGEN family raw files")
        source_hashes = _require_mapping(
            audit.get("source_hashes"), label="BAGEN family source hashes"
        )
        family_seen: set[str] = set()
        for raw_index, raw_value in enumerate(raw_files):
            raw = _require_mapping(raw_value, label=f"BAGEN raw file {raw_index}")
            relative = _safe_relative(raw.get("path"), label="BAGEN family raw path")
            if relative in family_seen:
                raise DataFoundationBaselineError("BAGEN family audit repeats a raw path")
            family_seen.add(relative)
            raw_sha = _require_sha256(raw.get("sha256"), label="BAGEN raw SHA-256")
            if source_hashes.get(relative) != raw_sha:
                raise DataFoundationBaselineError("BAGEN family source hash does not close")
            raw_bytes = _require_non_negative_int(
                raw.get("bytes"), label="BAGEN raw bytes"
            )
            entries.append((f"{family_root}/{relative}", raw_bytes, raw_sha))
        if set(source_hashes) != family_seen:
            raise DataFoundationBaselineError("BAGEN family source hash paths do not close")
    framed_sha, total_bytes = _framed_file_index_sha256(entries)
    expected_sha = _require_sha256(
        raw_artifact.get("sha256"), label="BAGEN raw aggregate SHA-256"
    )
    expected_bytes = _require_non_negative_int(
        raw_artifact.get("bytes"), label="BAGEN raw aggregate bytes"
    )
    expected_count = _require_non_negative_int(
        raw_artifact.get("file_count"), label="BAGEN raw aggregate file count"
    )
    if framed_sha != expected_sha or total_bytes != expected_bytes or len(entries) != expected_count:
        raise DataFoundationBaselineError("BAGEN raw framed file index does not close")
    return tuple(sorted(entries))


def _source_lock(
    root: Path,
    *,
    name: str,
    lock_source: Mapping[str, Any],
    audit_source: Mapping[str, Any],
    expected_source_id: str,
    expected_capabilities: SourceCapabilities,
    raw_artifact_name: str,
) -> SourceLock:
    artifacts = _require_mapping(audit_source.get("artifacts"), label=f"audit.{name}.artifacts")
    descriptor_artifact = _require_mapping(
        artifacts.get("descriptor"), label=f"audit.{name}.descriptor artifact"
    )
    manifest_artifact_name = "manifest" if name == "bagen_swebench" else "inventory"
    manifest_artifact = _require_mapping(
        artifacts.get(manifest_artifact_name),
        label=f"audit.{name}.{manifest_artifact_name} artifact",
    )
    raw_artifact = _require_mapping(
        artifacts.get(raw_artifact_name), label=f"audit.{name}.{raw_artifact_name} artifact"
    )
    descriptor_path = _safe_relative(
        descriptor_artifact.get("path"), label=f"{name} descriptor path"
    )
    descriptor_sha = _require_sha256(
        descriptor_artifact.get("sha256"), label=f"{name} descriptor SHA-256"
    )
    if lock_source.get("descriptor_file_sha256") != descriptor_sha:
        raise DataFoundationBaselineError(f"{name} descriptor lock does not close")
    descriptor = _load_descriptor(
        root,
        descriptor_path,
        expected_sha256=descriptor_sha,
        expected_source_id=expected_source_id,
        expected_capabilities=expected_capabilities,
    )
    audit_descriptor = _require_mapping(
        audit_source.get("source_descriptor"), label=f"audit.{name}.source_descriptor"
    )
    if audit_descriptor != descriptor.to_dict():
        raise DataFoundationBaselineError(f"{name} audit descriptor does not close")
    if lock_source.get("source_descriptor_hash") != descriptor.descriptor_hash:
        raise DataFoundationBaselineError(f"{name} descriptor semantic hash does not close")
    if lock_source.get("capability_contract_hash") != descriptor.capabilities.contract_hash:
        raise DataFoundationBaselineError(f"{name} capability contract does not close")

    manifest_path = _safe_relative(
        manifest_artifact.get("path"), label=f"{name} manifest path"
    )
    manifest_sha = _require_sha256(
        manifest_artifact.get("sha256"), label=f"{name} manifest SHA-256"
    )
    if descriptor.manifest_path != manifest_path or descriptor.manifest_sha256 != manifest_sha:
        raise DataFoundationBaselineError(f"{name} descriptor manifest identity does not close")
    if lock_source.get("manifest_sha256") != manifest_sha:
        raise DataFoundationBaselineError(f"{name} baseline manifest lock does not close")
    manifest_file = _repo_path(root, manifest_path, label=f"{name} manifest path")
    if _sha256_file(manifest_file) != manifest_sha:
        raise DataFoundationBaselineError(f"{name} manifest file SHA-256 does not close")

    audit_dataset = _require_mapping(
        audit_source.get("dataset"), label=f"audit.{name}.dataset"
    )
    dataset_id = _require_sha256(audit_dataset.get("dataset_id"), label=f"{name} dataset id")
    dataset_rows = _require_non_negative_int(
        audit_dataset.get("row_count"), label=f"{name} dataset row count"
    )
    if lock_source.get("dataset_id") != dataset_id or lock_source.get("row_count") != dataset_rows:
        raise DataFoundationBaselineError(f"{name} dataset baseline lock does not close")

    raw_sha_kind = _require_text(
        raw_artifact.get("sha256_kind"), label=f"{name} raw artifact SHA-256 kind"
    )
    expected_raw_sha_kind = (
        "framed_file_index_v1" if name == "bagen_swebench" else "file_bytes"
    )
    if raw_sha_kind != expected_raw_sha_kind:
        raise DataFoundationBaselineError(f"{name} raw artifact SHA-256 kind is wrong")
    raw_file_index = (
        _load_bagen_raw_file_index(root, artifacts, raw_artifact)
        if name == "bagen_swebench"
        else ()
    )
    return SourceLock(
        name=name,
        descriptor_path=descriptor_path,
        descriptor_file_sha256=descriptor_sha,
        descriptor=descriptor,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        dataset_id=dataset_id,
        dataset_row_count=dataset_rows,
        raw_artifact_path=_safe_relative(
            raw_artifact.get("path"), label=f"{name} raw artifact path"
        ),
        raw_artifact_sha256=_require_sha256(
            raw_artifact.get("sha256"), label=f"{name} raw artifact SHA-256"
        ),
        raw_artifact_sha256_kind=raw_sha_kind,
        raw_artifact_bytes=_require_non_negative_int(
            raw_artifact.get("bytes"), label=f"{name} raw artifact bytes"
        ),
        raw_file_index=raw_file_index,
    )


def load_lock_context(root: Path, baseline_lock_relative: str) -> LockContext:
    baseline_lock_path = _safe_relative(
        baseline_lock_relative, label="baseline lock path"
    )
    lock_file = _repo_path(root, baseline_lock_path, label="baseline lock path")
    lock_sha = _sha256_file(lock_file)
    lock = _load_json(lock_file, label="Data Foundation baseline lock")
    if lock.get("baseline_schema_version") != 1 or lock.get("baseline_type") != "data_foundation_v2":
        raise DataFoundationBaselineError("Data Foundation baseline lock schema is unsupported")
    production = _require_mapping(
        lock.get("production_audit"), label="baseline lock.production_audit"
    )
    audit_path = _safe_relative(
        production.get("relative_path"), label="production audit path"
    )
    audit_file_sha = _require_sha256(
        production.get("file_sha256"), label="production audit file SHA-256"
    )
    audit_file = _repo_path(root, audit_path, label="production audit path")
    if _sha256_file(audit_file) != audit_file_sha:
        raise DataFoundationBaselineError("production audit file SHA-256 does not close")
    audit = _load_json(audit_file, label="Data Foundation production audit")
    audit_payload_sha = _verify_embedded_payload(
        audit,
        field="audit_payload_sha256",
        label="Data Foundation production audit",
    )
    if production.get("audit_payload_sha256") != audit_payload_sha:
        raise DataFoundationBaselineError("production audit payload lock does not close")
    if audit.get("dataset_schema_version") != CAPABILITY_DATASET_SCHEMA_VERSION:
        raise DataFoundationBaselineError("production audit is not dataset schema v2")

    lock_implementation = _require_mapping(
        lock.get("implementation"), label="baseline lock.implementation"
    )
    audit_implementation = _require_mapping(
        audit.get("implementation"), label="production audit.implementation"
    )
    audit_commit = _require_commit(
        audit_implementation.get("git_commit"), label="production audit Git commit"
    )
    audit_tree = _require_sha256(
        audit_implementation.get("source_tree_sha256"),
        label="production audit source tree SHA-256",
    )
    if (
        lock_implementation.get("git_commit") != audit_commit
        or lock_implementation.get("source_tree_sha256") != audit_tree
    ):
        raise DataFoundationBaselineError("baseline lock and production audit source binding differ")

    lock_sources = _require_mapping(lock.get("sources"), label="baseline lock.sources")
    audit_sources = _require_mapping(audit.get("sources"), label="production audit.sources")
    expected_names = {"bagen_swebench", "spend_openhands"}
    if set(lock_sources) != expected_names or set(audit_sources) != expected_names:
        raise DataFoundationBaselineError("baseline lock must contain exactly the two frozen sources")
    bagen = _source_lock(
        root,
        name="bagen_swebench",
        lock_source=_require_mapping(lock_sources["bagen_swebench"], label="BAGEN lock"),
        audit_source=_require_mapping(audit_sources["bagen_swebench"], label="BAGEN audit"),
        expected_source_id=BagenSwebenchReader.source_id,
        expected_capabilities=BagenSwebenchReader.capabilities,
        raw_artifact_name="raw_trajectories",
    )
    spend = _source_lock(
        root,
        name="spend_openhands",
        lock_source=_require_mapping(lock_sources["spend_openhands"], label="Spend lock"),
        audit_source=_require_mapping(audit_sources["spend_openhands"], label="Spend audit"),
        expected_source_id=OpenHandsArchiveReader.source_id,
        expected_capabilities=OpenHandsArchiveReader.capabilities,
        raw_artifact_name="archive",
    )
    return LockContext(
        baseline_lock_path=baseline_lock_path,
        baseline_lock_file_sha256=lock_sha,
        audit_path=audit_path,
        audit_file_sha256=audit_file_sha,
        audit_payload_sha256=audit_payload_sha,
        audit_git_commit=audit_commit,
        audit_source_tree_sha256=audit_tree,
        sources={"bagen_swebench": bagen, "spend_openhands": spend},
    )


def _load_bagen_manifest(root: Path, source: SourceLock) -> tuple[Path, ...]:
    manifest = _repo_path(root, source.manifest_path, label="BAGEN manifest path")
    parent = PurePosixPath(source.manifest_path).parent
    paths: list[Path] = []
    seen: set[str] = set()
    expected_raw = {
        relative: (size, sha256)
        for relative, size, sha256 in source.raw_file_index
    }
    if not expected_raw:
        raise DataFoundationBaselineError("BAGEN audit lock has no raw file index")
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise DataFoundationBaselineError("cannot read BAGEN manifest") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise DataFoundationBaselineError("BAGEN manifest contains a blank line")
        value = _strict_json_loads(line, label=f"BAGEN manifest line {line_number}")
        entry = _require_mapping(value, label=f"BAGEN manifest line {line_number}")
        relative = _safe_relative(
            entry.get("path"), label=f"BAGEN manifest line {line_number}.path"
        )
        if not relative.endswith(".traj.json"):
            continue
        combined = (parent / PurePosixPath(relative)).as_posix()
        if combined in seen:
            raise DataFoundationBaselineError("BAGEN manifest repeats a trajectory path")
        seen.add(combined)
        path = _repo_path(root, combined, label="BAGEN trajectory path")
        expected_bytes = _require_non_negative_int(
            entry.get("size_bytes"), label="BAGEN trajectory bytes"
        )
        try:
            audited_bytes, audited_sha = expected_raw[combined]
        except KeyError as exc:
            raise DataFoundationBaselineError(
                "BAGEN manifest trajectory is absent from frozen family audits"
            ) from exc
        if expected_bytes != audited_bytes:
            raise DataFoundationBaselineError("BAGEN manifest/family byte size differs")
        if (
            not path.is_file()
            or _is_link_or_reparse(path)
            or path.stat().st_size != expected_bytes
        ):
            raise DataFoundationBaselineError("BAGEN trajectory identity does not match manifest")
        if _sha256_file(path) != audited_sha:
            raise DataFoundationBaselineError("BAGEN trajectory SHA-256 does not match audit")
        paths.append(path)
    if not paths or seen != set(expected_raw):
        raise DataFoundationBaselineError("BAGEN manifest/family raw path sets do not close")
    return tuple(sorted(paths, key=lambda item: item.as_posix()))


def load_bagen_dataset(
    root: Path,
    source: SourceLock,
    *,
    verified_paths: tuple[Path, ...] | None = None,
) -> tuple[SupervisedDataset, tuple[Path, ...]]:
    paths = verified_paths if verified_paths is not None else _load_bagen_manifest(root, source)
    if not paths:
        raise DataFoundationBaselineError("verified BAGEN trajectory set is empty")
    reader = BagenSwebenchReader()
    dataset = build_capability_supervised_dataset(
        (reader.read(path) for path in paths), source.descriptor
    )
    _verify_realized_dataset(dataset, source)
    return dataset, paths


def _verify_spend_archive(root: Path, source: SourceLock) -> Path:
    inventory_path = _repo_path(root, source.manifest_path, label="Spend inventory path")
    inventory = _load_json(inventory_path, label="Spend inventory")
    archive_relative = _safe_relative(inventory.get("archive_path"), label="Spend archive path")
    archive_sha = _require_sha256(
        inventory.get("archive_sha256"), label="Spend archive SHA-256"
    )
    archive_bytes = _require_non_negative_int(
        inventory.get("archive_bytes"), label="Spend archive bytes"
    )
    if (
        archive_relative != source.raw_artifact_path
        or archive_sha != source.raw_artifact_sha256
        or archive_bytes != source.raw_artifact_bytes
    ):
        raise DataFoundationBaselineError("Spend archive identity disagrees with audit lock")
    archive = _repo_path(root, archive_relative, label="Spend archive path")
    if (
        not archive.is_file()
        or _is_link_or_reparse(archive)
        or archive.stat().st_size != archive_bytes
    ):
        raise DataFoundationBaselineError("Spend archive is missing or has the wrong byte size")
    if _sha256_file(archive) != archive_sha:
        raise DataFoundationBaselineError("Spend archive SHA-256 does not match audit lock")
    return archive


def load_spend_dataset(
    root: Path,
    source: SourceLock,
    *,
    verified_archive: Path | None = None,
) -> tuple[SupervisedDataset, tuple[Path, ...]]:
    archive = (
        verified_archive
        if verified_archive is not None
        else _verify_spend_archive(root, source)
    )
    if not archive.is_file() or _is_link_or_reparse(archive):
        raise DataFoundationBaselineError("verified Spend archive is missing or unsafe")
    reader = OpenHandsArchiveReader()
    dataset = build_capability_supervised_dataset(
        reader.iter_archive(
            archive,
            OpenHandsArchiveMetadata(archive_identity=source.raw_artifact_sha256),
        ),
        source.descriptor,
    )
    _verify_realized_dataset(dataset, source)
    return dataset, (archive,)


def _verify_realized_dataset(dataset: SupervisedDataset, source: SourceLock) -> None:
    if dataset.schema_version != CAPABILITY_DATASET_SCHEMA_VERSION:
        raise DataFoundationBaselineError("realized dataset is not schema v2")
    if dataset.dataset_id != source.dataset_id or len(dataset.rows) != source.dataset_row_count:
        raise DataFoundationBaselineError("realized dataset disagrees with frozen audit")
    if dataset.source_descriptor_hash != source.descriptor.descriptor_hash:
        raise DataFoundationBaselineError("realized dataset descriptor hash does not close")
    if dataset.capability_contract_hash != source.descriptor.capabilities.contract_hash:
        raise DataFoundationBaselineError("realized dataset capability hash does not close")


def _training_view(
    dataset_slice: DatasetSlice,
    rows: Sequence[DatasetRow],
    weights: Mapping[str, float],
) -> TrainingView:
    return TrainingView(
        dataset_id=dataset_slice.dataset_id,
        position=dataset_slice.position,
        target=dataset_slice.target,
        examples=tuple(
            TrainingExample(
                point=row.point.with_features({}),
                target_value=float(row.label),
                sample_weight=weights[row.point.point_id],
            )
            for row in rows
            if row.label is not None
        ),
    )


def _predict_static(fitted: Any, rows: Sequence[DatasetRow]) -> dict[str, TokenForecast]:
    forecasts: dict[str, TokenForecast] = {}
    for row in sorted(rows, key=lambda item: item.point.point_id):
        point = row.point.with_features({})
        session = fitted.start(RunContext(point.task_id, point.trajectory_id, point.run_id))
        forecast = session.predict(point)
        if forecast.point_id != point.point_id or forecast.target != point.target:
            raise DataFoundationBaselineError("empirical estimator returned a wrong identity")
        forecasts[point.point_id] = forecast
    if len(forecasts) != len(rows):
        raise DataFoundationBaselineError("empirical estimator returned duplicate point ids")
    return forecasts


def _task_hash(tasks: Iterable[str]) -> str:
    return _semantic_sha256(
        sorted({_holdout_task_digest(task_id) for task_id in tasks})
    )


def _public_identity(kind: str, value: str) -> str:
    return hashlib.sha256(
        f"data-foundation-baseline-public-id-v1\0{kind}\0{value}".encode("utf-8")
    ).hexdigest()


def _holdout_task_digest(task_id: str) -> str:
    return hashlib.sha256(
        f"{FINAL_HOLDOUT_SALT}\0{task_id}".encode("utf-8")
    ).hexdigest()


def _development_dataset_id(
    dataset: SupervisedDataset,
    development_tasks: frozenset[str],
    assignment_digest: str,
) -> str:
    """Bind only the development cohort while retaining source provenance."""

    rows = [
        {
            "point": {
                "point_id": row.point.point_id,
                "source_event_id": row.point.source_event_id,
                "task_id": row.point.task_id,
                "trajectory_id": row.point.trajectory_id,
                "run_id": row.point.run_id,
                "prediction_context_id": row.point.prediction_context_id,
                "condition_id": row.point.condition_id,
                "logical_call_id": row.point.logical_call_id,
                "attempt_id": row.point.attempt_id,
                "cutoff_event_seq": row.point.cutoff_event_seq,
                "position": row.point.position.value,
                "target": row.point.target.value,
                "features": dict(row.point.features),
                "known_offset_tokens": row.point.known_offset_tokens,
            },
            "label": row.label,
            "status": row.status.value,
            "invalid_reason": row.invalid_reason,
        }
        for row in sorted(dataset.rows, key=lambda item: item.point.point_id)
        if row.point.task_id in development_tasks
    ]
    if not rows:
        raise DataFoundationBaselineError("development identity has no rows")
    return _semantic_sha256(
        {
            "development_dataset_identity_schema_version": 1,
            "source_schema_version": dataset.schema_version,
            "source_descriptor_hash": dataset.source_descriptor_hash,
            "capability_contract_hash": dataset.capability_contract_hash,
            "holdout_policy_id": FINAL_HOLDOUT_POLICY_ID,
            "holdout_assignment_digest": assignment_digest,
            "cohort": "development",
            "rows": rows,
        }
    )


def make_holdout_plan(dataset: SupervisedDataset) -> HoldoutPlan:
    tasks = sorted(dataset.task_ids)
    if len(tasks) < MIN_DEVELOPMENT_TASKS_PER_CONDITION + 1:
        raise DataFoundationBaselineError("source has too few tasks for a permanent holdout")
    development: set[str] = set()
    holdout: set[str] = set()
    assignments: list[dict[str, Any]] = []
    for task_id in tasks:
        task_digest = _holdout_task_digest(task_id)
        bucket = int(task_digest, 16) % FINAL_HOLDOUT_BUCKET_COUNT
        cohort = (
            "final_holdout"
            if bucket < FINAL_HOLDOUT_BUCKET_THRESHOLD
            else "development"
        )
        (holdout if cohort == "final_holdout" else development).add(task_id)
        assignments.append(
            {
                "task_id_sha256": task_digest,
                "bucket": bucket,
                "cohort": cohort,
            }
        )
    if not holdout or len(development) < MIN_DEVELOPMENT_TASKS_PER_CONDITION:
        raise DataFoundationBaselineError("stable task-hash holdout produced an empty/small cohort")
    assignment_digest = _semantic_sha256(assignments)
    development_tasks = frozenset(development)
    development_dataset_id = _development_dataset_id(
        dataset,
        development_tasks,
        assignment_digest,
    )
    return HoldoutPlan(
        source_dataset_id=dataset.dataset_id,
        development_dataset_id=development_dataset_id,
        assignment_digest=assignment_digest,
        development_tasks=development_tasks,
        final_holdout_tasks=frozenset(holdout),
    )


def development_dataset(
    dataset: SupervisedDataset, plan: HoldoutPlan
) -> SupervisedDataset:
    if dataset.dataset_id != plan.source_dataset_id:
        raise DataFoundationBaselineError("holdout plan belongs to another source dataset")
    rows = tuple(
        row for row in dataset.rows if row.point.task_id in plan.development_tasks
    )
    if not rows or any(row.point.task_id in plan.final_holdout_tasks for row in rows):
        raise DataFoundationBaselineError("development cohort isolation failed")
    return SupervisedDataset(
        dataset_id=plan.development_dataset_id,
        rows=rows,
        schema_version=dataset.schema_version,
        source_descriptor_hash=dataset.source_descriptor_hash,
        capability_contract_hash=dataset.capability_contract_hash,
    )


def _bundle_document(
    *,
    fitted: Any,
    expansion: float,
    dataset_slice: DatasetSlice,
    split_plan_id: str,
    split_seed: int,
    fold: int,
    partition: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "empirical_bundle_schema_version": EMPIRICAL_BUNDLE_SCHEMA_VERSION,
        "identity": {
            "candidate_id": CANDIDATE_ID,
            "dataset_id": dataset_slice.dataset_id,
            "split_plan_id": split_plan_id,
            "split_seed": split_seed,
            "fold": fold,
            "position": dataset_slice.position.value,
            "target": dataset_slice.target.value,
            "condition_id": dataset_slice.condition_id,
            "train_tasks_sha256": _task_hash(partition.train_tasks),
            "validation_tasks_sha256": _task_hash(partition.validation_tasks),
            "calibration_tasks_sha256": _task_hash(partition.calibration_tasks),
            "development_score_tasks_sha256": _task_hash(partition.test_tasks),
        },
        "estimator": {
            "estimator_id": CANDIDATE_ID,
            "lower": float(fitted.lower),
            "point": float(fitted.point),
            "upper": float(fitted.upper),
        },
        "calibrator": {
            "calibrator_id": CALIBRATOR_ID,
            "alpha": ALPHA,
            "expansion": float(expansion),
        },
    }
    payload["bundle_payload_sha256"] = _semantic_sha256(payload)
    return payload


def load_empirical_bundle_bytes(payload: bytes) -> tuple[dict[str, Any], ReloadedEmpiricalBundle]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise DataFoundationBaselineError("empirical bundle is not UTF-8") from exc
    value = _strict_json_loads(text, label="empirical bundle")
    bundle = _require_mapping(value, label="empirical bundle")
    if payload != _canonical_bytes(bundle):
        raise DataFoundationBaselineError("empirical bundle JSON is not canonical")
    if set(bundle) != {
        "empirical_bundle_schema_version",
        "identity",
        "estimator",
        "calibrator",
        "bundle_payload_sha256",
    }:
        raise DataFoundationBaselineError("empirical bundle has missing or extra fields")
    if bundle.get("empirical_bundle_schema_version") != EMPIRICAL_BUNDLE_SCHEMA_VERSION:
        raise DataFoundationBaselineError("empirical bundle schema is unsupported")
    _verify_embedded_payload(bundle, field="bundle_payload_sha256", label="empirical bundle")
    identity = _require_mapping(bundle.get("identity"), label="empirical bundle.identity")
    if set(identity) != {
        "candidate_id",
        "dataset_id",
        "split_plan_id",
        "split_seed",
        "fold",
        "position",
        "target",
        "condition_id",
        "train_tasks_sha256",
        "validation_tasks_sha256",
        "calibration_tasks_sha256",
        "development_score_tasks_sha256",
    }:
        raise DataFoundationBaselineError("empirical bundle identity is not exact")
    if identity.get("candidate_id") != CANDIDATE_ID:
        raise DataFoundationBaselineError("empirical bundle candidate is wrong")
    for key in (
        "dataset_id",
        "split_plan_id",
        "train_tasks_sha256",
        "validation_tasks_sha256",
        "calibration_tasks_sha256",
        "development_score_tasks_sha256",
    ):
        _require_sha256(identity.get(key), label=f"empirical bundle.identity.{key}")
    _require_non_negative_int(identity.get("split_seed"), label="empirical bundle split seed")
    fold = _require_non_negative_int(identity.get("fold"), label="empirical bundle fold")
    if fold >= FOLDS:
        raise DataFoundationBaselineError("empirical bundle fold is out of range")
    try:
        PredictionPosition(_require_text(identity.get("position"), label="bundle position"))
        target = PredictionTarget(_require_text(identity.get("target"), label="bundle target"))
    except ValueError as exc:
        raise DataFoundationBaselineError("empirical bundle position or target is unknown") from exc
    _require_text(identity.get("condition_id"), label="bundle condition")

    estimator = _require_mapping(bundle.get("estimator"), label="empirical bundle.estimator")
    if set(estimator) != {"estimator_id", "lower", "point", "upper"}:
        raise DataFoundationBaselineError("empirical estimator payload is not exact")
    if estimator.get("estimator_id") != CANDIDATE_ID:
        raise DataFoundationBaselineError("empirical estimator id is wrong")
    lower = _require_finite_number(estimator.get("lower"), label="empirical lower")
    point = _require_finite_number(estimator.get("point"), label="empirical point")
    upper = _require_finite_number(estimator.get("upper"), label="empirical upper")
    if not 0 <= lower <= point <= upper:
        raise DataFoundationBaselineError("empirical quantiles are negative or unordered")

    calibrator = _require_mapping(bundle.get("calibrator"), label="empirical bundle.calibrator")
    if set(calibrator) != {"calibrator_id", "alpha", "expansion"}:
        raise DataFoundationBaselineError("empirical calibrator payload is not exact")
    if calibrator.get("calibrator_id") != CALIBRATOR_ID:
        raise DataFoundationBaselineError("empirical calibrator id is wrong")
    alpha = _require_finite_number(calibrator.get("alpha"), label="empirical alpha")
    if alpha != ALPHA:
        raise DataFoundationBaselineError("empirical bundle alpha is not frozen")
    expansion = _require_finite_number(
        calibrator.get("expansion"), label="empirical calibrator expansion"
    )
    if expansion < 0:
        raise DataFoundationBaselineError("empirical calibrator expansion is negative")
    return dict(bundle), ReloadedEmpiricalBundle(target, lower, point, upper, expansion)


def _forecast_dict(forecast: TokenForecast) -> dict[str, float]:
    if forecast.raw_lower is None or forecast.raw_point is None or forecast.raw_upper is None:
        raise DataFoundationBaselineError("baseline forecast lacks raw quantile diagnostics")
    return {
        "lower": forecast.lower,
        "point": forecast.point,
        "upper": forecast.upper,
        "raw_lower": forecast.raw_lower,
        "raw_point": forecast.raw_point,
        "raw_upper": forecast.raw_upper,
    }


def _partition_evidence(split_plan: Any) -> list[dict[str, Any]]:
    def hashed(tasks: Iterable[str]) -> list[str]:
        return sorted(_holdout_task_digest(task) for task in tasks)

    evidence: list[dict[str, Any]] = []
    for fold in range(split_plan.folds):
        partition = split_plan.partition(fold)
        evidence.append(
            {
                "fold": fold,
                "train_task_id_sha256": hashed(partition.train_tasks),
                "validation_task_id_sha256": hashed(partition.validation_tasks),
                "calibration_task_id_sha256": hashed(partition.calibration_tasks),
                "development_score_task_id_sha256": hashed(partition.test_tasks),
            }
        )
    return evidence


def _aggregate_metrics(seed_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not seed_results:
        raise DataFoundationBaselineError("cannot aggregate empty seed results")
    metrics = [_require_mapping(result.get("metrics"), label="seed metrics") for result in seed_results]
    keys = set(metrics[0])
    if any(set(item) != keys for item in metrics[1:]):
        raise DataFoundationBaselineError("metric suites differ across split seeds")
    aggregate: dict[str, Any] = {
        "metric_suite_id": METRIC_SUITE_ID,
        "split_seed_count": len(seed_results),
    }
    for key in sorted(keys):
        values = [item[key] for item in metrics]
        if key == "metric_suite_id":
            if set(values) != {METRIC_SUITE_ID}:
                raise DataFoundationBaselineError("metric suite id changed across seeds")
            continue
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            numeric = [float(value) for value in values]
            aggregate[key] = {
                "mean": sum(numeric) / len(numeric),
                "min": min(numeric),
                "max": max(numeric),
            }
    return aggregate


def _split_identity_from_hashed_assignments(
    assignments: Mapping[str, int],
    *,
    dataset_id: str,
    seed: int,
) -> tuple[str, str]:
    records = [
        {"task_id_sha256": task_hash, "fold": assignments[task_hash]}
        for task_hash in sorted(assignments)
    ]
    assignment_id = _semantic_sha256(
        {
            "assignment_policy_id": SPLIT_ASSIGNMENT_POLICY_ID,
            "folds": FOLDS,
            "seed": seed,
            "assignments": records,
        }
    )
    split_plan_id = _semantic_sha256(
        {"assignment_id": assignment_id, "dataset_id": dataset_id}
    )
    return assignment_id, split_plan_id


def make_baseline_split_plan(
    task_ids: Iterable[str],
    *,
    dataset_id: str,
    seed: int,
) -> SplitPlan:
    tasks = sorted(set(task_ids))
    if len(tasks) < FOLDS:
        raise DataFoundationBaselineError("development cohort has fewer tasks than folds")
    by_hash = {_holdout_task_digest(task_id): task_id for task_id in tasks}
    if len(by_hash) != len(tasks):
        raise DataFoundationBaselineError("pseudonymous task identity collision")
    ranked_hashes = sorted(
        by_hash,
        key=lambda task_hash: hashlib.sha256(
            f"{seed}\0{task_hash}".encode("utf-8")
        ).hexdigest(),
    )
    hashed_assignments = {
        task_hash: index % FOLDS for index, task_hash in enumerate(ranked_hashes)
    }
    assignment_id, split_plan_id = _split_identity_from_hashed_assignments(
        hashed_assignments, dataset_id=dataset_id, seed=seed
    )
    raw_assignments = tuple(
        sorted((by_hash[task_hash], fold) for task_hash, fold in hashed_assignments.items())
    )
    return SplitPlan(
        split_plan_id=split_plan_id,
        assignment_id=assignment_id,
        dataset_id=dataset_id,
        folds=FOLDS,
        seed=seed,
        assignments=raw_assignments,
    )


def run_development_cell(
    dataset: SupervisedDataset,
    *,
    source_name: str,
    position: PredictionPosition,
    target: PredictionTarget,
    condition_id: str,
    split_seed: int,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    dataset_slice = dataset.select(position, target, condition_id=condition_id)
    if not dataset_slice.rows:
        raise DataFoundationBaselineError("development baseline cell has no eligible rows")
    split_plan = make_baseline_split_plan(
        dataset.task_ids,
        dataset_id=dataset.dataset_id,
        seed=split_seed,
    )
    split_plan.validate_tasks(
        (row.point.task_id for row in dataset_slice.rows), require_exact=False
    )
    weights = {
        weighted.row.point.point_id: weighted.sample_weight
        for weighted in dataset_slice.weighted_rows()
    }
    all_scored: list[ScoredForecast] = []
    predictions: list[dict[str, Any]] = []
    fold_metrics: dict[str, Mapping[str, Any]] = {}
    bundles: dict[str, bytes] = {}
    parity_count = 0

    condition_hash = hashlib.sha256(condition_id.encode("utf-8")).hexdigest()[:20]
    for fold in range(FOLDS):
        partition = split_plan.partition(fold)
        train = [row for row in dataset_slice.rows if row.point.task_id in partition.train_tasks]
        validation = [
            row for row in dataset_slice.rows if row.point.task_id in partition.validation_tasks
        ]
        calibration = [
            row for row in dataset_slice.rows if row.point.task_id in partition.calibration_tasks
        ]
        score = [row for row in dataset_slice.rows if row.point.task_id in partition.test_tasks]
        if not train or not validation or not calibration or not score:
            raise DataFoundationBaselineError(
                f"development fold {fold} contains an empty partition"
            )
        fitted = EmpiricalQuantileEstimator(alpha=ALPHA).fit(
            _training_view(dataset_slice, train, weights),
            _training_view(dataset_slice, validation, weights),
            FitContext(seed=split_seed, fold=fold, interval_alpha=ALPHA),
        )
        calibration_raw = _predict_static(fitted, calibration)
        fitted_calibrator = TaskMaxConformalCalibrator(alpha=ALPHA).fit(
            [
                CalibrationExample(
                    task_id=row.point.task_id,
                    forecast=calibration_raw[row.point.point_id],
                    target_value=float(row.label),
                )
                for row in calibration
                if row.label is not None
            ]
        )
        expansion = float(fitted_calibrator.expansion)
        bundle = _bundle_document(
            fitted=fitted,
            expansion=expansion,
            dataset_slice=dataset_slice,
            split_plan_id=split_plan.split_plan_id,
            split_seed=split_seed,
            fold=fold,
            partition=partition,
        )
        bundle_bytes = _canonical_bytes(bundle)
        loaded_document, loaded = load_empirical_bundle_bytes(bundle_bytes)
        if loaded_document != bundle:
            raise DataFoundationBaselineError("empirical bundle JSON reload changed semantics")
        bundle_path = (
            f"bundles/{source_name}/{condition_hash}/seed-{split_seed}/fold-{fold}.json"
        )
        _safe_relative(bundle_path, label="bundle path")
        bundles[bundle_path] = bundle_bytes
        raw_score = _predict_static(fitted, score)
        fold_scored: list[ScoredForecast] = []
        for row in sorted(score, key=lambda item: item.point.point_id):
            if row.label is None:
                raise DataFoundationBaselineError("eligible development row has no label")
            forecast = fitted_calibrator.transform(raw_score[row.point.point_id])
            reloaded = loaded.predict(row.point.point_id)
            if _forecast_dict(forecast) != _forecast_dict(reloaded):
                raise DataFoundationBaselineError("empirical bundle prediction parity failed")
            parity_count += 1
            scored = ScoredForecast(
                task_id=row.point.task_id,
                trajectory_id=row.point.trajectory_id,
                forecast=forecast,
                target_value=float(row.label),
                sample_weight=weights[row.point.point_id],
            )
            fold_scored.append(scored)
            all_scored.append(scored)
            predictions.append(
                {
                    "point_id_sha256": _public_identity("point", row.point.point_id),
                    "task_id_sha256": _holdout_task_digest(row.point.task_id),
                    "trajectory_id_sha256": _public_identity(
                        "trajectory", row.point.trajectory_id
                    ),
                    "run_id_sha256": _public_identity("run", row.point.run_id),
                    "condition_id": row.point.condition_id,
                    "fold": fold,
                    "target_value": row.label,
                    "sample_weight": weights[row.point.point_id],
                    "bundle_path": bundle_path,
                    "bundle_payload_sha256": bundle["bundle_payload_sha256"],
                    "forecast": _forecast_dict(forecast),
                }
            )
        fold_metrics[str(fold)] = evaluate_forecasts(fold_scored, alpha=ALPHA)

    expected = {row.point.point_id for row in dataset_slice.rows}
    actual = [record["point_id_sha256"] for record in predictions]
    expected_hashed = {_public_identity("point", point_id) for point_id in expected}
    if len(actual) != len(set(actual)) or set(actual) != expected_hashed:
        raise DataFoundationBaselineError(
            "development CV must score every eligible point exactly once"
        )
    result = {
        "candidate_id": CANDIDATE_ID,
        "source_name": source_name,
        "dataset_id": dataset.dataset_id,
        "position": position.value,
        "target": target.value,
        "condition_id": condition_id,
        "alpha": ALPHA,
        "calibrator_id": CALIBRATOR_ID,
        "split_seed": split_seed,
        "split_assignment_policy_id": SPLIT_ASSIGNMENT_POLICY_ID,
        "split_plan_id": split_plan.split_plan_id,
        "assignment_id": split_plan.assignment_id,
        "eligibility_hash": dataset_slice.eligibility_hash,
        "weighting_id": dataset_slice.weighting_id,
        "eligible_point_count": len(dataset_slice.rows),
        "eligible_task_count": len({row.point.task_id for row in dataset_slice.rows}),
        "split_assignments": [
            {
                "task_id_sha256": _holdout_task_digest(task_id),
                "fold": fold_number,
            }
            for task_id, fold_number in split_plan.assignments
        ],
        "partitions": _partition_evidence(split_plan),
        "fold_metrics": fold_metrics,
        "metrics": evaluate_forecasts(all_scored, alpha=ALPHA),
        "prediction_count": len(predictions),
        "bundle_reload_parity": {
            "status": "exact",
            "record_count": parity_count,
            "mismatch_count": 0,
        },
        "predictions": sorted(
            predictions, key=lambda item: item["point_id_sha256"]
        ),
    }
    return result, bundles


def _condition_ids(
    dataset: SupervisedDataset,
    *,
    position: PredictionPosition,
    target: PredictionTarget,
) -> tuple[str, ...]:
    conditions = sorted(
        {
            row.point.condition_id
            for row in dataset.rows
            if row.eligible and row.point.position == position and row.point.target == target
        }
    )
    if not conditions:
        raise DataFoundationBaselineError("required baseline target has no eligible conditions")
    return tuple(conditions)


def build_results(
    *,
    bagen_dataset: SupervisedDataset,
    spend_dataset: SupervisedDataset,
    lock_context: LockContext,
    code_binding: CodeBinding,
    audit_compatible_source_tree_hash: str,
    tracked_control_tree_hash: str,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    if audit_compatible_source_tree_hash != lock_context.audit_source_tree_sha256:
        raise DataFoundationBaselineError(
            "current dataset/audit source bytes differ from the frozen production audit"
        )
    specifications = (
        (
            "bagen_swebench",
            bagen_dataset,
            PredictionPosition.TASK_UPDATE,
            PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
        ),
        (
            "spend_openhands",
            spend_dataset,
            PredictionPosition.TASK_LAUNCH,
            PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
        ),
    )
    cells: list[dict[str, Any]] = []
    not_estimable_conditions: list[dict[str, Any]] = []
    bundles: dict[str, bytes] = {}
    holdout_evidence: dict[str, dict[str, Any]] = {}
    for source_name, source_dataset, position, target in specifications:
        source_lock = lock_context.sources[source_name]
        if (
            source_dataset.dataset_id != source_lock.dataset_id
            or len(source_dataset.rows) != source_lock.dataset_row_count
        ):
            raise DataFoundationBaselineError(
                "baseline input dataset differs from its production audit lock"
            )
        plan = make_holdout_plan(source_dataset)
        dataset = development_dataset(source_dataset, plan)
        holdout_evidence[source_name] = plan.to_evidence()
        conditions = _condition_ids(source_dataset, position=position, target=target)
        frozen_estimable = (
            FROZEN_BAGEN_ESTIMABLE_CONDITIONS
            if source_name == "bagen_swebench"
            else FROZEN_SPEND_CONDITIONS
        )
        frozen_gated = (
            FROZEN_BAGEN_NOT_ESTIMABLE_CONDITIONS
            if source_name == "bagen_swebench"
            else frozenset()
        )
        if set(conditions) != frozen_estimable | frozen_gated:
            raise DataFoundationBaselineError(
                f"{source_name} frozen condition identity set changed"
            )
        for condition_id in conditions:
            eligible_rows = [
                row
                for row in source_dataset.rows
                if row.eligible
                and row.point.position == position
                and row.point.target == target
                and row.point.condition_id == condition_id
            ]
            eligible_tasks = {row.point.task_id for row in eligible_rows}
            development_tasks = {
                row.point.task_id
                for row in dataset.rows
                if row.eligible
                and row.point.position == position
                and row.point.target == target
                and row.point.condition_id == condition_id
            }
            holdout_tasks = {
                row.point.task_id
                for row in source_dataset.rows
                if row.eligible
                and row.point.position == position
                and row.point.target == target
                and row.point.condition_id == condition_id
                and row.point.task_id in plan.final_holdout_tasks
            }
            is_estimable = (
                len(development_tasks) < MIN_DEVELOPMENT_TASKS_PER_CONDITION
                or not holdout_tasks
            ) is False
            if condition_id in frozen_gated:
                if is_estimable:
                    raise DataFoundationBaselineError(
                        "a frozen not-estimable condition unexpectedly became estimable"
                    )
                not_estimable_conditions.append(
                    {
                        "source_name": source_name,
                        "position": position.value,
                        "target": target.value,
                        "condition_id": condition_id,
                        "status": "not_estimable",
                        "reason": (
                            "insufficient_development_or_final_holdout_tasks_for_five_fold_cv"
                        ),
                        "eligible_point_count": len(eligible_rows),
                        "eligible_task_count": len(eligible_tasks),
                        "development_task_count": len(development_tasks),
                        "final_holdout_task_count": len(holdout_tasks),
                        "eligible_task_set_sha256": _task_hash(eligible_tasks),
                        "development_task_set_sha256": _task_hash(development_tasks),
                        "final_holdout_task_set_sha256": _task_hash(holdout_tasks),
                        "required_fold_count": FOLDS,
                        "required_development_task_count": (
                            MIN_DEVELOPMENT_TASKS_PER_CONDITION
                        ),
                        "required_final_holdout_task_count": 1,
                        "holdout_policy_id": FINAL_HOLDOUT_POLICY_ID,
                        "target_values_used_for_fit_calibration_scoring": False,
                        "prediction_count": 0,
                        "bundle_count": 0,
                    }
                )
                continue
            if condition_id not in frozen_estimable or not is_estimable:
                raise DataFoundationBaselineError(
                    "a frozen estimable condition has an empty/small cohort"
                )
            seed_results: list[dict[str, Any]] = []
            for split_seed in SPLIT_SEEDS:
                seed_result, seed_bundles = run_development_cell(
                    dataset,
                    source_name=source_name,
                    position=position,
                    target=target,
                    condition_id=condition_id,
                    split_seed=split_seed,
                )
                overlap = set(bundles) & set(seed_bundles)
                if overlap:
                    raise DataFoundationBaselineError("bundle paths collided")
                bundles.update(seed_bundles)
                seed_results.append(seed_result)
            holdout_task_hashes = {
                _holdout_task_digest(task_id) for task_id in plan.final_holdout_tasks
            }
            for seed_result in seed_results:
                assigned = {
                    record["task_id_sha256"]
                    for record in seed_result["split_assignments"]
                }
                predicted = {
                    record["task_id_sha256"]
                    for record in seed_result["predictions"]
                }
                if assigned & holdout_task_hashes or predicted & holdout_task_hashes:
                    raise DataFoundationBaselineError(
                        "final holdout task entered development assignment or predictions"
                    )
            first_assignments = {
                result["split_seed"]: result["split_assignments"] for result in seed_results
            }
            if len(first_assignments) != len(SPLIT_SEEDS):
                raise DataFoundationBaselineError("split seeds did not remain distinct")
            cells.append(
                {
                    "source_name": source_name,
                    "position": position.value,
                    "target": target.value,
                    "condition_id": condition_id,
                    "candidate_id": CANDIDATE_ID,
                    "development_task_count": len(development_tasks),
                    "final_holdout_task_count": len(holdout_tasks),
                    "development_task_set_sha256": _task_hash(development_tasks),
                    "final_holdout_task_set_sha256": _task_hash(holdout_tasks),
                    "cohort_disjointness_verified": True,
                    "aggregate_metrics": _aggregate_metrics(seed_results),
                    "seed_results": seed_results,
                }
            )
    source_records = {
        name: {
            "source_id": source.descriptor.source_id,
            "revision": source.descriptor.revision,
            "descriptor_path": source.descriptor_path,
            "descriptor_file_sha256": source.descriptor_file_sha256,
            "source_descriptor_hash": source.descriptor.descriptor_hash,
            "capability_contract_hash": source.descriptor.capabilities.contract_hash,
            "manifest_path": source.manifest_path,
            "manifest_sha256": source.manifest_sha256,
            "raw_artifact_path": source.raw_artifact_path,
            "raw_artifact_sha256": source.raw_artifact_sha256,
            "raw_artifact_sha256_kind": source.raw_artifact_sha256_kind,
            "raw_artifact_bytes": source.raw_artifact_bytes,
            "dataset_id": source.dataset_id,
            "dataset_row_count": source.dataset_row_count,
        }
        for name, source in sorted(lock_context.sources.items())
    }
    results: dict[str, Any] = {
        "baseline_results_schema_version": BASELINE_ARTIFACT_SCHEMA_VERSION,
        "baseline_id": BASELINE_ID,
        "evaluation_scope": "development_cross_validation_only",
        "final_holdout_evaluated": False,
        "final_holdout_prediction_count": 0,
        "final_holdout_target_values_used_for_fit_calibration_scoring": False,
        "final_model_selection_claim": "none",
        "candidate_id": CANDIDATE_ID,
        "fold_count": FOLDS,
        "split_seeds": list(SPLIT_SEEDS),
        "alpha": ALPHA,
        "calibrator_id": CALIBRATOR_ID,
        "metric_suite_id": METRIC_SUITE_ID,
        "condition_gate_policy": {
            "policy_id": "frozen_condition_minimum_cohort_gate_v1",
            "required_fold_count": FOLDS,
            "required_development_task_count": MIN_DEVELOPMENT_TASKS_PER_CONDITION,
            "required_final_holdout_task_count": 1,
            "bagen_estimable_condition_count": len(
                FROZEN_BAGEN_ESTIMABLE_CONDITIONS
            ),
            "bagen_not_estimable_condition_count": len(
                FROZEN_BAGEN_NOT_ESTIMABLE_CONDITIONS
            ),
            "spend_estimable_condition_count": len(FROZEN_SPEND_CONDITIONS),
        },
        "not_estimable_conditions": sorted(
            not_estimable_conditions,
            key=lambda item: (item["source_name"], item["condition_id"]),
        ),
        "permanent_final_holdout_policy": {
            "policy_id": FINAL_HOLDOUT_POLICY_ID,
            "salt": FINAL_HOLDOUT_SALT,
            "bucket_count": FINAL_HOLDOUT_BUCKET_COUNT,
            "final_holdout_bucket_threshold_exclusive": (
                FINAL_HOLDOUT_BUCKET_THRESHOLD
            ),
            "assignment_inputs": "task_id_only",
            "independent_of_split_seed_labels_and_suffixes": True,
            "source_plans": holdout_evidence,
        },
        "source_binding": {
            "git_commit": code_binding.git_commit,
            "runner_and_src_code_tree_sha256": code_binding.source_tree_sha256,
            "tracked_control_tree_sha256": _require_sha256(
                tracked_control_tree_hash, label="tracked control tree SHA-256"
            ),
            "code_paths": list(code_binding.paths),
            "baseline_lock_path": lock_context.baseline_lock_path,
            "baseline_lock_file_sha256": lock_context.baseline_lock_file_sha256,
            "data_foundation_audit_path": lock_context.audit_path,
            "data_foundation_audit_file_sha256": lock_context.audit_file_sha256,
            "data_foundation_audit_payload_sha256": lock_context.audit_payload_sha256,
            "data_foundation_audit_git_commit": lock_context.audit_git_commit,
            "data_foundation_audit_source_tree_sha256": (
                lock_context.audit_source_tree_sha256
            ),
            "current_audit_compatible_source_tree_sha256": (
                audit_compatible_source_tree_hash
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "sources": source_records,
        "cells": sorted(cells, key=lambda item: (item["source_name"], item["condition_id"])),
        "bundle_count": len(bundles),
    }
    results["results_payload_sha256"] = _semantic_sha256(results)
    return results, bundles


def _snapshot_files(root: Path, relatives: Iterable[str]) -> tuple[FileSnapshot, ...]:
    snapshots: list[FileSnapshot] = []
    for relative in sorted(set(relatives)):
        path = _repo_path(root, relative, label="input snapshot path")
        if not path.is_file() or _is_link_or_reparse(path):
            raise DataFoundationBaselineError("input snapshot contains a non-regular file")
        stat = path.stat()
        snapshots.append(FileSnapshot(relative, stat.st_size, stat.st_mtime_ns))
    return tuple(snapshots)


def _relative_paths(root: Path, paths: Iterable[Path]) -> tuple[str, ...]:
    resolved = root.resolve()
    values: list[str] = []
    for path in paths:
        try:
            values.append(path.resolve().relative_to(resolved).as_posix())
        except ValueError as exc:
            raise DataFoundationBaselineError("source input escapes repository") from exc
    return tuple(values)


def _safe_output(root: Path, relative: str) -> tuple[str, Path]:
    canonical = _safe_relative(relative, label="output path")
    if not canonical.startswith(OUTPUT_PREFIX) or canonical == OUTPUT_PREFIX.rstrip("/"):
        raise DataFoundationBaselineError(
            f"output must be below the ignored {OUTPUT_PREFIX!r} directory"
        )
    return canonical, _repo_path(root, canonical, label="output path")


def _artifact_manifest(
    *,
    results: Mapping[str, Any],
    files: Mapping[str, bytes],
) -> dict[str, Any]:
    file_records = [
        {
            "path": relative,
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
        }
        for relative, payload in sorted(files.items())
    ]
    identity = {
        "artifact_schema_version": BASELINE_ARTIFACT_SCHEMA_VERSION,
        "baseline_id": results["baseline_id"],
        "git_commit": results["source_binding"]["git_commit"],
        "runner_and_src_code_tree_sha256": results["source_binding"][
            "runner_and_src_code_tree_sha256"
        ],
        "tracked_control_tree_sha256": results["source_binding"][
            "tracked_control_tree_sha256"
        ],
        "results_payload_sha256": results["results_payload_sha256"],
        "files": file_records,
    }
    return {**identity, "artifact_id": _semantic_sha256(identity)}


def publish_artifact(
    root: Path,
    output_relative: str,
    *,
    results: Mapping[str, Any],
    bundles: Mapping[str, bytes],
    pre_publish_check: Callable[[], None],
) -> Path:
    _, output = _safe_output(root, output_relative)
    if output.exists() or _is_link_or_reparse(output):
        raise DataFoundationBaselineError("baseline artifact already exists; overwrite is forbidden")
    output.parent.mkdir(parents=True, exist_ok=True)
    if _is_link_or_reparse(output.parent):
        raise DataFoundationBaselineError(
            "baseline artifact parent must not be linked or reparse-backed"
        )
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        files = dict(bundles)
        if "results.json" in files:
            raise DataFoundationBaselineError("bundle path collides with results.json")
        results_bytes = _canonical_bytes(results)
        files["results.json"] = results_bytes
        for relative, payload in sorted(files.items()):
            safe = _safe_relative(relative, label="artifact member path")
            destination = temporary.joinpath(*PurePosixPath(safe).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        manifest = _artifact_manifest(results=results, files=files)
        manifest_bytes = _canonical_bytes(manifest)
        (temporary / "manifest.json").write_bytes(manifest_bytes)
        (temporary / "manifest.sha256").write_text(
            f"{_sha256_bytes(manifest_bytes)}\n", encoding="ascii", newline="\n"
        )
        verify_artifact(temporary)
        pre_publish_check()
        if output.exists() or _is_link_or_reparse(output):
            raise DataFoundationBaselineError("baseline artifact appeared before publish")
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    verify_artifact(output)
    return output


def verify_artifact(path: Path) -> dict[str, Any]:
    if not path.is_dir() or _is_link_or_reparse(path):
        raise DataFoundationBaselineError("baseline artifact must be one regular directory")
    _assert_tree_no_links(path, label="baseline artifact")
    manifest_path = path / "manifest.json"
    sidecar_path = path / "manifest.sha256"
    manifest = _load_json(manifest_path, label="baseline artifact manifest")
    if manifest_path.read_bytes() != _canonical_bytes(manifest):
        raise DataFoundationBaselineError("artifact manifest JSON is not canonical")
    try:
        sidecar = sidecar_path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise DataFoundationBaselineError("cannot read artifact manifest SHA-256") from exc
    if not re.fullmatch(r"[0-9a-f]{64}\n", sidecar):
        raise DataFoundationBaselineError("artifact manifest SHA-256 sidecar is malformed")
    if sidecar.strip() != _sha256_file(manifest_path):
        raise DataFoundationBaselineError("artifact manifest SHA-256 does not match")
    expected_manifest_keys = {
        "artifact_schema_version",
        "baseline_id",
        "git_commit",
        "runner_and_src_code_tree_sha256",
        "tracked_control_tree_sha256",
        "results_payload_sha256",
        "files",
        "artifact_id",
    }
    if set(manifest) != expected_manifest_keys:
        raise DataFoundationBaselineError("artifact manifest has missing or extra fields")
    if (
        manifest.get("artifact_schema_version") != BASELINE_ARTIFACT_SCHEMA_VERSION
        or manifest.get("baseline_id") != BASELINE_ID
    ):
        raise DataFoundationBaselineError("artifact manifest identity is unsupported")
    manifest_commit = _require_commit(manifest.get("git_commit"), label="artifact Git commit")
    manifest_code_hash = _require_sha256(
        manifest.get("runner_and_src_code_tree_sha256"),
        label="artifact runner/source tree SHA-256",
    )
    manifest_control_hash = _require_sha256(
        manifest.get("tracked_control_tree_sha256"),
        label="artifact tracked control tree SHA-256",
    )
    manifest_results_hash = _require_sha256(
        manifest.get("results_payload_sha256"),
        label="artifact results payload SHA-256",
    )
    identity = dict(manifest)
    artifact_id = _require_sha256(identity.pop("artifact_id"), label="artifact id")
    if _semantic_sha256(identity) != artifact_id:
        raise DataFoundationBaselineError("artifact id does not close")
    records = _require_list(manifest.get("files"), label="artifact files")
    expected_files = {"manifest.json", "manifest.sha256"}
    file_payloads: dict[str, bytes] = {}
    for index, record_value in enumerate(records):
        record = _require_mapping(record_value, label=f"artifact files[{index}]")
        if set(record) != {"path", "bytes", "sha256"}:
            raise DataFoundationBaselineError("artifact file record is not exact")
        relative = _safe_relative(record.get("path"), label="artifact member path")
        if relative in expected_files:
            raise DataFoundationBaselineError("artifact manifest repeats a member")
        expected_files.add(relative)
        member = path.joinpath(*PurePosixPath(relative).parts)
        if not member.is_file() or _is_link_or_reparse(member):
            raise DataFoundationBaselineError("artifact member is missing or unsafe")
        payload = member.read_bytes()
        if len(payload) != _require_non_negative_int(
            record.get("bytes"), label="artifact member bytes"
        ) or _sha256_bytes(payload) != _require_sha256(
            record.get("sha256"), label="artifact member SHA-256"
        ):
            raise DataFoundationBaselineError("artifact member hash or size mismatch")
        file_payloads[relative] = payload
    actual_files = {
        item.relative_to(path).as_posix() for item in path.rglob("*") if item.is_file()
    }
    actual_dirs = {item for item in path.rglob("*") if item.is_dir()}
    if actual_files != expected_files or any(not any(directory.iterdir()) for directory in actual_dirs):
        raise DataFoundationBaselineError("artifact contains missing, extra, or empty members")
    try:
        results_value = _strict_json_loads(
            file_payloads["results.json"].decode("utf-8", errors="strict"),
            label="baseline results",
        )
    except KeyError as exc:
        raise DataFoundationBaselineError("artifact has no results.json") from exc
    if not isinstance(results_value, dict):
        raise DataFoundationBaselineError("baseline results must be an object")
    if file_payloads["results.json"] != _canonical_bytes(results_value):
        raise DataFoundationBaselineError("baseline results JSON is not canonical")
    expected_results_keys = {
        "baseline_results_schema_version",
        "baseline_id",
        "evaluation_scope",
        "final_holdout_evaluated",
        "final_holdout_prediction_count",
        "final_holdout_target_values_used_for_fit_calibration_scoring",
        "final_model_selection_claim",
        "candidate_id",
        "fold_count",
        "split_seeds",
        "alpha",
        "calibrator_id",
        "metric_suite_id",
        "condition_gate_policy",
        "not_estimable_conditions",
        "permanent_final_holdout_policy",
        "source_binding",
        "runtime",
        "sources",
        "cells",
        "bundle_count",
        "results_payload_sha256",
    }
    if set(results_value) != expected_results_keys or results_value.get(
        "baseline_results_schema_version"
    ) != BASELINE_ARTIFACT_SCHEMA_VERSION:
        raise DataFoundationBaselineError("baseline results schema is not exact")
    results_hash = _verify_embedded_payload(
        results_value, field="results_payload_sha256", label="baseline results"
    )
    if results_hash != manifest_results_hash:
        raise DataFoundationBaselineError("results hash disagrees with manifest")
    source_binding = _require_mapping(
        results_value.get("source_binding"), label="results source binding"
    )
    expected_binding_keys = {
        "git_commit",
        "runner_and_src_code_tree_sha256",
        "tracked_control_tree_sha256",
        "code_paths",
        "baseline_lock_path",
        "baseline_lock_file_sha256",
        "data_foundation_audit_path",
        "data_foundation_audit_file_sha256",
        "data_foundation_audit_payload_sha256",
        "data_foundation_audit_git_commit",
        "data_foundation_audit_source_tree_sha256",
        "current_audit_compatible_source_tree_sha256",
    }
    if set(source_binding) != expected_binding_keys:
        raise DataFoundationBaselineError("results source binding is not exact")
    _require_commit(source_binding.get("data_foundation_audit_git_commit"), label="audit commit")
    audit_tree = _require_sha256(
        source_binding.get("data_foundation_audit_source_tree_sha256"),
        label="audit source tree SHA-256",
    )
    if _require_sha256(
        source_binding.get("current_audit_compatible_source_tree_sha256"),
        label="current audit-compatible tree SHA-256",
    ) != audit_tree:
        raise DataFoundationBaselineError("current source bytes do not close to the audit tree")
    for key in (
        "tracked_control_tree_sha256",
        "baseline_lock_file_sha256",
        "data_foundation_audit_file_sha256",
        "data_foundation_audit_payload_sha256",
    ):
        _require_sha256(source_binding.get(key), label=f"source binding {key}")
    for key in ("baseline_lock_path", "data_foundation_audit_path"):
        _safe_relative(source_binding.get(key), label=f"source binding {key}")
    code_paths = _require_list(source_binding.get("code_paths"), label="bound code paths")
    canonical_code_paths = [
        _safe_relative(value, label="bound code path") for value in code_paths
    ]
    if (
        len(set(canonical_code_paths)) != len(canonical_code_paths)
        or RUNNER_RELATIVE not in canonical_code_paths
        or not any(path.startswith("src/token_prediction/") for path in canonical_code_paths)
    ):
        raise DataFoundationBaselineError("bound code path set is incomplete or duplicate")
    if (
        source_binding.get("git_commit") != manifest_commit
        or source_binding.get("runner_and_src_code_tree_sha256") != manifest_code_hash
        or source_binding.get("tracked_control_tree_sha256") != manifest_control_hash
        or results_value.get("baseline_id") != manifest.get("baseline_id")
    ):
        raise DataFoundationBaselineError("results and manifest source identities differ")
    if (
        results_value.get("candidate_id") != CANDIDATE_ID
        or results_value.get("fold_count") != FOLDS
        or results_value.get("split_seeds") != list(SPLIT_SEEDS)
        or results_value.get("alpha") != ALPHA
        or results_value.get("calibrator_id") != CALIBRATOR_ID
        or results_value.get("metric_suite_id") != METRIC_SUITE_ID
        or results_value.get("final_model_selection_claim") != "none"
    ):
        raise DataFoundationBaselineError("baseline protocol constants are not frozen")
    runtime = _require_mapping(results_value.get("runtime"), label="baseline runtime")
    if set(runtime) != {"python", "python_implementation", "platform"}:
        raise DataFoundationBaselineError("baseline runtime record is not exact")
    for key, value in runtime.items():
        _require_text(value, label=f"baseline runtime.{key}")
    sources = _require_mapping(results_value.get("sources"), label="baseline sources")
    if set(sources) != {"bagen_swebench", "spend_openhands"}:
        raise DataFoundationBaselineError("baseline source records are incomplete")
    expected_source_ids = {
        "bagen_swebench": BagenSwebenchReader.source_id,
        "spend_openhands": OpenHandsArchiveReader.source_id,
    }
    for name, source_value in sources.items():
        source = _require_mapping(source_value, label=f"baseline source {name}")
        if set(source) != {
            "source_id",
            "revision",
            "descriptor_path",
            "descriptor_file_sha256",
            "source_descriptor_hash",
            "capability_contract_hash",
            "manifest_path",
            "manifest_sha256",
            "raw_artifact_path",
            "raw_artifact_sha256",
            "raw_artifact_sha256_kind",
            "raw_artifact_bytes",
            "dataset_id",
            "dataset_row_count",
        } or source.get("source_id") != expected_source_ids[name]:
            raise DataFoundationBaselineError("baseline source identity is not exact")
        _require_text(source.get("revision"), label=f"{name} revision")
        expected_kind = "framed_file_index_v1" if name == "bagen_swebench" else "file_bytes"
        if source.get("raw_artifact_sha256_kind") != expected_kind:
            raise DataFoundationBaselineError(f"{name} raw artifact hash kind is invalid")
        for key in ("descriptor_path", "manifest_path", "raw_artifact_path"):
            _safe_relative(source.get(key), label=f"{name} {key}")
        for key in (
            "descriptor_file_sha256",
            "source_descriptor_hash",
            "capability_contract_hash",
            "manifest_sha256",
            "raw_artifact_sha256",
            "dataset_id",
        ):
            _require_sha256(source.get(key), label=f"{name} {key}")
        _require_non_negative_int(source.get("raw_artifact_bytes"), label=f"{name} raw bytes")
        _require_non_negative_int(source.get("dataset_row_count"), label=f"{name} dataset rows")
    if results_value.get("evaluation_scope") != "development_cross_validation_only":
        raise DataFoundationBaselineError("artifact is not development-only")
    if (
        results_value.get("final_holdout_evaluated") is not False
        or results_value.get("final_holdout_prediction_count") != 0
        or results_value.get(
            "final_holdout_target_values_used_for_fit_calibration_scoring"
        )
        is not False
    ):
        raise DataFoundationBaselineError(
            "artifact used the permanent final holdout for evaluation or fitting"
        )

    bundle_members = {
        relative for relative in file_payloads if relative.startswith("bundles/")
    }
    if set(file_payloads) != bundle_members | {"results.json"} or results_value.get(
        "bundle_count"
    ) != len(bundle_members):
        raise DataFoundationBaselineError("artifact bundle file count or location differs")
    holdout_policy = _require_mapping(
        results_value.get("permanent_final_holdout_policy"),
        label="permanent final holdout policy",
    )
    if set(holdout_policy) != {
        "policy_id",
        "salt",
        "bucket_count",
        "final_holdout_bucket_threshold_exclusive",
        "assignment_inputs",
        "independent_of_split_seed_labels_and_suffixes",
        "source_plans",
    }:
        raise DataFoundationBaselineError("permanent final holdout policy schema is not exact")
    if (
        holdout_policy.get("policy_id") != FINAL_HOLDOUT_POLICY_ID
        or holdout_policy.get("salt") != FINAL_HOLDOUT_SALT
        or holdout_policy.get("bucket_count") != FINAL_HOLDOUT_BUCKET_COUNT
        or holdout_policy.get("final_holdout_bucket_threshold_exclusive")
        != FINAL_HOLDOUT_BUCKET_THRESHOLD
        or holdout_policy.get("assignment_inputs") != "task_id_only"
        or holdout_policy.get("independent_of_split_seed_labels_and_suffixes")
        is not True
    ):
        raise DataFoundationBaselineError("permanent final holdout policy changed")
    source_plans = _require_mapping(
        holdout_policy.get("source_plans"), label="holdout source plans"
    )
    if set(source_plans) != {"bagen_swebench", "spend_openhands"}:
        raise DataFoundationBaselineError("holdout plans do not cover both frozen sources")
    expected_plan_keys = {
        "policy_id",
        "salt",
        "bucket_count",
        "final_holdout_bucket_threshold_exclusive",
        "source_dataset_id",
        "development_dataset_id",
        "assignment_digest",
        "development_task_count",
        "final_holdout_task_count",
        "development_task_set_sha256",
        "final_holdout_task_set_sha256",
    }
    for name, plan_value in source_plans.items():
        plan = _require_mapping(plan_value, label=f"holdout plan {name}")
        if (
            set(plan) != expected_plan_keys
            or plan.get("policy_id") != FINAL_HOLDOUT_POLICY_ID
            or plan.get("salt") != FINAL_HOLDOUT_SALT
            or plan.get("bucket_count") != FINAL_HOLDOUT_BUCKET_COUNT
            or plan.get("final_holdout_bucket_threshold_exclusive")
            != FINAL_HOLDOUT_BUCKET_THRESHOLD
            or plan.get("source_dataset_id") != sources[name].get("dataset_id")
        ):
            raise DataFoundationBaselineError("source holdout plan identity is invalid")
        for key in (
            "source_dataset_id",
            "development_dataset_id",
            "assignment_digest",
            "development_task_set_sha256",
            "final_holdout_task_set_sha256",
        ):
            _require_sha256(plan.get(key), label=f"holdout plan {name}.{key}")
        development_count = _require_non_negative_int(
            plan.get("development_task_count"), label=f"holdout plan {name} development count"
        )
        holdout_count = _require_non_negative_int(
            plan.get("final_holdout_task_count"), label=f"holdout plan {name} holdout count"
        )
        if development_count < MIN_DEVELOPMENT_TASKS_PER_CONDITION or holdout_count < 1:
            raise DataFoundationBaselineError("source holdout plan contains an empty/small cohort")

    gate_policy = _require_mapping(
        results_value.get("condition_gate_policy"), label="condition gate policy"
    )
    if set(gate_policy) != {
        "policy_id",
        "required_fold_count",
        "required_development_task_count",
        "required_final_holdout_task_count",
        "bagen_estimable_condition_count",
        "bagen_not_estimable_condition_count",
        "spend_estimable_condition_count",
    } or dict(gate_policy) != {
        "policy_id": "frozen_condition_minimum_cohort_gate_v1",
        "required_fold_count": FOLDS,
        "required_development_task_count": MIN_DEVELOPMENT_TASKS_PER_CONDITION,
        "required_final_holdout_task_count": 1,
        "bagen_estimable_condition_count": len(FROZEN_BAGEN_ESTIMABLE_CONDITIONS),
        "bagen_not_estimable_condition_count": len(
            FROZEN_BAGEN_NOT_ESTIMABLE_CONDITIONS
        ),
        "spend_estimable_condition_count": len(FROZEN_SPEND_CONDITIONS),
    }:
        raise DataFoundationBaselineError("condition gate policy changed")
    gate_keys = {
        "source_name",
        "position",
        "target",
        "condition_id",
        "status",
        "reason",
        "eligible_point_count",
        "eligible_task_count",
        "development_task_count",
        "final_holdout_task_count",
        "eligible_task_set_sha256",
        "development_task_set_sha256",
        "final_holdout_task_set_sha256",
        "required_fold_count",
        "required_development_task_count",
        "required_final_holdout_task_count",
        "holdout_policy_id",
        "target_values_used_for_fit_calibration_scoring",
        "prediction_count",
        "bundle_count",
    }
    gated_conditions: set[str] = set()
    gates = _require_list(
        results_value.get("not_estimable_conditions"),
        label="not-estimable conditions",
    )
    for gate_value in gates:
        gate = _require_mapping(gate_value, label="not-estimable condition")
        condition_id = _require_text(
            gate.get("condition_id"), label="gated condition id"
        )
        if (
            set(gate) != gate_keys
            or gate.get("source_name") != "bagen_swebench"
            or gate.get("position") != PredictionPosition.TASK_UPDATE.value
            or gate.get("target")
            != PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS.value
            or condition_id not in FROZEN_BAGEN_NOT_ESTIMABLE_CONDITIONS
            or condition_id in gated_conditions
            or gate.get("status") != "not_estimable"
            or gate.get("reason")
            != "insufficient_development_or_final_holdout_tasks_for_five_fold_cv"
            or gate.get("required_fold_count") != FOLDS
            or gate.get("required_development_task_count")
            != MIN_DEVELOPMENT_TASKS_PER_CONDITION
            or gate.get("required_final_holdout_task_count") != 1
            or gate.get("holdout_policy_id") != FINAL_HOLDOUT_POLICY_ID
            or gate.get("target_values_used_for_fit_calibration_scoring") is not False
            or gate.get("prediction_count") != 0
            or gate.get("bundle_count") != 0
        ):
            raise DataFoundationBaselineError("not-estimable condition gate is invalid")
        gated_conditions.add(condition_id)
        eligible_count = _require_non_negative_int(
            gate.get("eligible_task_count"), label="gated eligible task count"
        )
        development_count = _require_non_negative_int(
            gate.get("development_task_count"), label="gated development task count"
        )
        holdout_count = _require_non_negative_int(
            gate.get("final_holdout_task_count"), label="gated holdout task count"
        )
        _require_non_negative_int(
            gate.get("eligible_point_count"), label="gated eligible point count"
        )
        if (
            eligible_count != development_count + holdout_count
            or (
                development_count >= MIN_DEVELOPMENT_TASKS_PER_CONDITION
                and holdout_count >= 1
            )
        ):
            raise DataFoundationBaselineError("gated condition cohort counts are inconsistent")
        for key in (
            "eligible_task_set_sha256",
            "development_task_set_sha256",
            "final_holdout_task_set_sha256",
        ):
            _require_sha256(gate.get(key), label=f"gated condition {key}")
    if gated_conditions != FROZEN_BAGEN_NOT_ESTIMABLE_CONDITIONS:
        raise DataFoundationBaselineError("frozen not-estimable condition set changed")

    parity = 0
    referenced_bundles: set[str] = set()
    seen_cells: set[tuple[str, str]] = set()
    for cell_value in _require_list(results_value.get("cells"), label="results cells"):
        cell = _require_mapping(cell_value, label="results cell")
        if set(cell) != {
            "source_name",
            "position",
            "target",
            "condition_id",
            "candidate_id",
            "development_task_count",
            "final_holdout_task_count",
            "development_task_set_sha256",
            "final_holdout_task_set_sha256",
            "cohort_disjointness_verified",
            "aggregate_metrics",
            "seed_results",
        }:
            raise DataFoundationBaselineError("baseline cell schema is not exact")
        source_name = _require_text(cell.get("source_name"), label="cell source name")
        if source_name not in source_plans or cell.get("candidate_id") != CANDIDATE_ID:
            raise DataFoundationBaselineError("cell source or candidate identity is invalid")
        try:
            cell_position = PredictionPosition(
                _require_text(cell.get("position"), label="cell position")
            )
            cell_target = PredictionTarget(
                _require_text(cell.get("target"), label="cell target")
            )
        except ValueError as exc:
            raise DataFoundationBaselineError("cell position or target is unknown") from exc
        condition_id = _require_text(cell.get("condition_id"), label="cell condition")
        cell_key = (source_name, condition_id)
        if cell_key in seen_cells:
            raise DataFoundationBaselineError("baseline repeats a source/condition cell")
        seen_cells.add(cell_key)
        cell_development_count = _require_non_negative_int(
            cell.get("development_task_count"), label="cell development task count"
        )
        cell_holdout_count = _require_non_negative_int(
            cell.get("final_holdout_task_count"), label="cell holdout task count"
        )
        if (
            cell_development_count < MIN_DEVELOPMENT_TASKS_PER_CONDITION
            or cell_holdout_count < 1
            or cell.get("cohort_disjointness_verified") is not True
        ):
            raise DataFoundationBaselineError("cell contains an empty/small or unverified cohort")
        cell_development_hash = _require_sha256(
            cell.get("development_task_set_sha256"),
            label="cell development task set SHA-256",
        )
        _require_sha256(
            cell.get("final_holdout_task_set_sha256"),
            label="cell final holdout task set SHA-256",
        )
        if (source_name, cell_position, cell_target) not in {
            (
                "bagen_swebench",
                PredictionPosition.TASK_UPDATE,
                PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
            ),
            (
                "spend_openhands",
                PredictionPosition.TASK_LAUNCH,
                PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
            ),
        }:
            raise DataFoundationBaselineError("cell is outside the approved baseline matrix")
        seed_values = _require_list(cell.get("seed_results"), label="cell seed results")
        seed_mappings = [
            _require_mapping(value, label="cell seed result") for value in seed_values
        ]
        if [value.get("split_seed") for value in seed_mappings] != list(SPLIT_SEEDS):
            raise DataFoundationBaselineError("cell split seeds are missing or reordered")
        for seed in seed_mappings:
            if set(seed) != {
                "candidate_id",
                "source_name",
                "dataset_id",
                "position",
                "target",
                "condition_id",
                "alpha",
                "calibrator_id",
                "split_seed",
                "split_assignment_policy_id",
                "split_plan_id",
                "assignment_id",
                "eligibility_hash",
                "weighting_id",
                "eligible_point_count",
                "eligible_task_count",
                "split_assignments",
                "partitions",
                "fold_metrics",
                "metrics",
                "prediction_count",
                "bundle_reload_parity",
                "predictions",
            }:
                raise DataFoundationBaselineError("seed result schema is not exact")
            split_seed = _require_non_negative_int(
                seed.get("split_seed"), label="seed split seed"
            )
            if (
                seed.get("split_assignment_policy_id") != SPLIT_ASSIGNMENT_POLICY_ID
                or seed.get("weighting_id") != "task_run_point_equal_v1"
            ):
                raise DataFoundationBaselineError("split or weighting policy changed")
            split_plan_id = _require_sha256(
                seed.get("split_plan_id"), label="seed split plan id"
            )
            assignment_id = _require_sha256(
                seed.get("assignment_id"), label="seed assignment id"
            )
            if (
                seed.get("source_name") != source_name
                or seed.get("dataset_id")
                != source_plans[source_name].get("development_dataset_id")
                or seed.get("position") != cell_position.value
                or seed.get("target") != cell_target.value
                or seed.get("condition_id") != condition_id
                or seed.get("candidate_id") != CANDIDATE_ID
                or seed.get("alpha") != ALPHA
                or seed.get("calibrator_id") != CALIBRATOR_ID
            ):
                raise DataFoundationBaselineError("seed result identity differs from its cell")
            assignments = _require_list(
                seed.get("split_assignments"), label="seed split assignments"
            )
            assigned_hashes: set[str] = set()
            assignment_by_hash: dict[str, int] = {}
            for assignment_value in assignments:
                assignment = _require_mapping(
                    assignment_value, label="seed split assignment"
                )
                if set(assignment) != {"task_id_sha256", "fold"}:
                    raise DataFoundationBaselineError("split assignment is not exact")
                task_hash = _require_sha256(
                    assignment.get("task_id_sha256"), label="split task id SHA-256"
                )
                fold = _require_non_negative_int(
                    assignment.get("fold"), label="split assignment fold"
                )
                if fold >= FOLDS or task_hash in assigned_hashes:
                    raise DataFoundationBaselineError("split assignment is duplicate/out of range")
                assigned_hashes.add(task_hash)
                assignment_by_hash[task_hash] = fold
            ranked_hashes = sorted(
                assigned_hashes,
                key=lambda task_hash: hashlib.sha256(
                    f"{split_seed}\0{task_hash}".encode("utf-8")
                ).hexdigest(),
            )
            expected_assignment = {
                task_hash: index % FOLDS
                for index, task_hash in enumerate(ranked_hashes)
            }
            expected_assignment_id, expected_split_plan_id = (
                _split_identity_from_hashed_assignments(
                    expected_assignment,
                    dataset_id=_require_sha256(
                        seed.get("dataset_id"), label="seed development dataset id"
                    ),
                    seed=split_seed,
                )
            )
            if (
                assignment_by_hash != expected_assignment
                or assignment_id != expected_assignment_id
                or split_plan_id != expected_split_plan_id
            ):
                raise DataFoundationBaselineError("split assignment provenance does not close")
            source_plan = _require_mapping(
                source_plans[source_name], label="source holdout plan"
            )
            if _semantic_sha256(sorted(assigned_hashes)) != source_plan.get(
                "development_task_set_sha256"
            ):
                raise DataFoundationBaselineError("split assignment includes a non-development task")

            partitions = _require_list(seed.get("partitions"), label="seed partitions")
            if len(partitions) != FOLDS:
                raise DataFoundationBaselineError("seed partition evidence is incomplete")
            partition_groups_by_fold: dict[int, tuple[set[str], ...]] = {}
            for expected_fold, partition_value in enumerate(partitions):
                partition = _require_mapping(partition_value, label="seed partition")
                if set(partition) != {
                    "fold",
                    "train_task_id_sha256",
                    "validation_task_id_sha256",
                    "calibration_task_id_sha256",
                    "development_score_task_id_sha256",
                }:
                    raise DataFoundationBaselineError("seed partition schema is not exact")
                if partition.get("fold") != expected_fold:
                    raise DataFoundationBaselineError("seed partitions are reordered")
                groups: list[set[str]] = []
                for key in (
                    "train_task_id_sha256",
                    "validation_task_id_sha256",
                    "calibration_task_id_sha256",
                    "development_score_task_id_sha256",
                ):
                    values = _require_list(partition.get(key), label=f"partition {key}")
                    hashes = {
                        _require_sha256(value, label=f"partition {key} value")
                        for value in values
                    }
                    if len(hashes) != len(values):
                        raise DataFoundationBaselineError("partition task hashes repeat")
                    groups.append(hashes)
                if any(
                    left & right
                    for index, left in enumerate(groups)
                    for right in groups[index + 1 :]
                ) or set().union(*groups) != assigned_hashes:
                    raise DataFoundationBaselineError("partition task groups overlap or omit tasks")
                calibration_fold = (expected_fold + 1) % FOLDS
                validation_fold = (expected_fold + 2) % FOLDS
                expected_score = {
                    task_hash
                    for task_hash, fold in expected_assignment.items()
                    if fold == expected_fold
                }
                expected_calibration = {
                    task_hash
                    for task_hash, fold in expected_assignment.items()
                    if fold == calibration_fold
                }
                expected_validation = {
                    task_hash
                    for task_hash, fold in expected_assignment.items()
                    if fold == validation_fold
                }
                expected_train = assigned_hashes - (
                    expected_score | expected_calibration | expected_validation
                )
                expected_groups = (
                    expected_train,
                    expected_validation,
                    expected_calibration,
                    expected_score,
                )
                if tuple(groups) != expected_groups:
                    raise DataFoundationBaselineError("partition rotation provenance is invalid")
                partition_groups_by_fold[expected_fold] = expected_groups

            seed_scored: list[ScoredForecast] = []
            fold_scored: dict[int, list[ScoredForecast]] = {
                fold: [] for fold in range(FOLDS)
            }
            predictions = _require_list(seed.get("predictions"), label="predictions")
            predicted_points: set[str] = set()
            predicted_tasks: set[str] = set()
            weight_structure: dict[str, dict[str, list[float]]] = {}
            for record_value in predictions:
                record = _require_mapping(record_value, label="prediction record")
                expected_record_keys = {
                    "point_id_sha256",
                    "task_id_sha256",
                    "trajectory_id_sha256",
                    "run_id_sha256",
                    "condition_id",
                    "fold",
                    "target_value",
                    "sample_weight",
                    "bundle_path",
                    "bundle_payload_sha256",
                    "forecast",
                }
                if set(record) != expected_record_keys:
                    raise DataFoundationBaselineError("prediction record is not exact")
                point_hash = _require_sha256(
                    record.get("point_id_sha256"), label="prediction point id SHA-256"
                )
                task_hash = _require_sha256(
                    record.get("task_id_sha256"), label="prediction task id SHA-256"
                )
                trajectory_hash = _require_sha256(
                    record.get("trajectory_id_sha256"),
                    label="prediction trajectory id SHA-256",
                )
                _require_sha256(
                    record.get("run_id_sha256"), label="prediction run id SHA-256"
                )
                record_fold = _require_non_negative_int(
                    record.get("fold"), label="prediction fold"
                )
                if (
                    record_fold >= FOLDS
                    or point_hash in predicted_points
                    or task_hash not in assigned_hashes
                    or assignment_by_hash.get(task_hash) != record_fold
                    or record.get("condition_id") != condition_id
                ):
                    raise DataFoundationBaselineError("prediction identity is inconsistent")
                predicted_points.add(point_hash)
                predicted_tasks.add(task_hash)
                bundle_path = _safe_relative(
                    record.get("bundle_path"), label="prediction bundle path"
                )
                if bundle_path not in bundle_members:
                    raise DataFoundationBaselineError("prediction references a missing bundle")
                referenced_bundles.add(bundle_path)
                bundle_document, loaded = load_empirical_bundle_bytes(
                    file_payloads[bundle_path]
                )
                if record.get("bundle_payload_sha256") != bundle_document.get(
                    "bundle_payload_sha256"
                ):
                    raise DataFoundationBaselineError("prediction bundle identity differs")
                bundle_identity = _require_mapping(
                    bundle_document.get("identity"), label="bundle identity"
                )
                if (
                    bundle_identity.get("dataset_id") != seed.get("dataset_id")
                    or bundle_identity.get("split_plan_id") != split_plan_id
                    or bundle_identity.get("split_seed") != split_seed
                    or bundle_identity.get("fold") != record_fold
                    or bundle_identity.get("position") != cell_position.value
                    or bundle_identity.get("target") != cell_target.value
                    or bundle_identity.get("condition_id") != condition_id
                ):
                    raise DataFoundationBaselineError("bundle and prediction identities differ")
                expected_groups = partition_groups_by_fold[record_fold]
                for key, group in zip(
                    (
                        "train_tasks_sha256",
                        "validation_tasks_sha256",
                        "calibration_tasks_sha256",
                        "development_score_tasks_sha256",
                    ),
                    expected_groups,
                ):
                    if bundle_identity.get(key) != _semantic_sha256(sorted(group)):
                        raise DataFoundationBaselineError(
                            "bundle partition provenance does not close"
                        )
                forecast = loaded.predict(point_hash)
                expected = _require_mapping(record.get("forecast"), label="prediction forecast")
                if _forecast_dict(forecast) != expected:
                    raise DataFoundationBaselineError("reloaded artifact prediction parity failed")
                target_value = _require_non_negative_int(
                    record.get("target_value"), label="prediction target value"
                )
                sample_weight = _require_finite_number(
                    record.get("sample_weight"), label="prediction sample weight"
                )
                if sample_weight <= 0:
                    raise DataFoundationBaselineError("prediction sample weight is not positive")
                weight_structure.setdefault(task_hash, {}).setdefault(
                    trajectory_hash, []
                ).append(sample_weight)
                scored = ScoredForecast(
                    task_id=task_hash,
                    trajectory_id=trajectory_hash,
                    forecast=forecast,
                    target_value=float(target_value),
                    sample_weight=sample_weight,
                )
                seed_scored.append(scored)
                fold_scored[record_fold].append(scored)
                parity += 1
            for trajectories in weight_structure.values():
                run_count = len(trajectories)
                if run_count <= 0:
                    raise DataFoundationBaselineError("weighting task has no trajectories")
                for point_weights in trajectories.values():
                    expected_weight = 1.0 / (run_count * len(point_weights))
                    if any(weight != expected_weight for weight in point_weights):
                        raise DataFoundationBaselineError(
                            "task_run_point_equal_v1 weights do not replay"
                        )
            if (
                seed.get("prediction_count") != len(predictions)
                or seed.get("eligible_point_count") != len(predictions)
                or _semantic_sha256(sorted(predicted_tasks))
                != cell_development_hash
                or len(predicted_tasks) != cell_development_count
                or seed.get("eligible_task_count") != cell_development_count
            ):
                raise DataFoundationBaselineError("prediction cohort evidence does not close")
            parity_evidence = _require_mapping(
                seed.get("bundle_reload_parity"), label="bundle reload parity"
            )
            if (
                set(parity_evidence)
                != {"status", "record_count", "mismatch_count"}
                or
                parity_evidence.get("status") != "exact"
                or parity_evidence.get("record_count") != len(predictions)
                or parity_evidence.get("mismatch_count") != 0
            ):
                raise DataFoundationBaselineError("bundle reload parity evidence is invalid")
            if evaluate_forecasts(seed_scored, alpha=ALPHA) != seed.get("metrics"):
                raise DataFoundationBaselineError("seed aggregate metrics do not replay")
            stored_fold_metrics = _require_mapping(
                seed.get("fold_metrics"), label="seed fold metrics"
            )
            if set(stored_fold_metrics) != {str(fold) for fold in range(FOLDS)}:
                raise DataFoundationBaselineError("fold metric keys are incomplete")
            for fold in range(FOLDS):
                if evaluate_forecasts(fold_scored[fold], alpha=ALPHA) != stored_fold_metrics.get(
                    str(fold)
                ):
                    raise DataFoundationBaselineError("fold metrics do not replay")
        if _aggregate_metrics(seed_mappings) != cell.get("aggregate_metrics"):
            raise DataFoundationBaselineError("cross-seed aggregate metrics do not replay")
    if parity <= 0:
        raise DataFoundationBaselineError("artifact contains no prediction parity evidence")
    bagen_cells = {
        condition for name, condition in seen_cells if name == "bagen_swebench"
    }
    spend_cells = {
        condition for name, condition in seen_cells if name == "spend_openhands"
    }
    if (
        bagen_cells != FROZEN_BAGEN_ESTIMABLE_CONDITIONS
        or spend_cells != FROZEN_SPEND_CONDITIONS
        or bagen_cells & gated_conditions
    ):
        raise DataFoundationBaselineError("baseline estimable condition set changed")
    if referenced_bundles != bundle_members:
        raise DataFoundationBaselineError("artifact contains an unreferenced empirical bundle")
    return manifest


def run_baseline(
    *,
    repository_root: str | Path,
    baseline_lock: str = DEFAULT_BASELINE_LOCK,
    output: str = DEFAULT_OUTPUT,
) -> Path:
    supplied_root = Path(repository_root)
    if _is_link_or_reparse(supplied_root):
        raise DataFoundationBaselineError("repository root must not be linked or reparse-backed")
    root = supplied_root.resolve()
    if not root.is_dir():
        raise DataFoundationBaselineError("repository root is not a directory")
    verify_execution_origin(root)
    _safe_output(root, output)
    before_code = capture_code_binding(root)
    lock_context = load_lock_context(root, baseline_lock)
    tracked_control_paths = (
        lock_context.baseline_lock_path,
        *(source.descriptor_path for source in lock_context.sources.values()),
    )
    control_tree_hash = tracked_control_tree_sha256(
        root,
        tracked_control_paths,
        git_commit=before_code.git_commit,
    )
    audit_compatible_hash = audit_compatible_source_tree_sha256(root)
    if audit_compatible_hash != lock_context.audit_source_tree_sha256:
        raise DataFoundationBaselineError(
            "current dataset/audit source bytes differ from the frozen production audit"
        )

    bagen_paths = _load_bagen_manifest(
        root, lock_context.sources["bagen_swebench"]
    )
    spend_archive = _verify_spend_archive(
        root, lock_context.sources["spend_openhands"]
    )
    spend_paths = (spend_archive,)
    control_paths = (
        lock_context.baseline_lock_path,
        lock_context.audit_path,
        *(source.descriptor_path for source in lock_context.sources.values()),
        *(source.manifest_path for source in lock_context.sources.values()),
    )
    raw_paths = _relative_paths(root, (*bagen_paths, *spend_paths))
    before_inputs = _snapshot_files(root, (*control_paths, *raw_paths))
    bagen_dataset, realized_bagen_paths = load_bagen_dataset(
        root,
        lock_context.sources["bagen_swebench"],
        verified_paths=bagen_paths,
    )
    spend_dataset, realized_spend_paths = load_spend_dataset(
        root,
        lock_context.sources["spend_openhands"],
        verified_archive=spend_archive,
    )
    if realized_bagen_paths != bagen_paths or realized_spend_paths != spend_paths:
        raise DataFoundationBaselineError("verified source paths changed before reader use")
    results, bundles = build_results(
        bagen_dataset=bagen_dataset,
        spend_dataset=spend_dataset,
        lock_context=lock_context,
        code_binding=before_code,
        audit_compatible_source_tree_hash=audit_compatible_hash,
        tracked_control_tree_hash=control_tree_hash,
    )

    def pre_publish_check() -> None:
        after_code = capture_code_binding(root)
        if after_code != before_code:
            raise DataFoundationBaselineError("HEAD or relevant source changed during baseline run")
        if audit_compatible_source_tree_sha256(root) != audit_compatible_hash:
            raise DataFoundationBaselineError("audit-compatible source changed during baseline run")
        if tracked_control_tree_sha256(
            root,
            tracked_control_paths,
            git_commit=before_code.git_commit,
        ) != control_tree_hash:
            raise DataFoundationBaselineError("tracked controls changed during baseline run")
        if _snapshot_files(root, (*control_paths, *raw_paths)) != before_inputs:
            raise DataFoundationBaselineError("a frozen input changed during baseline run")
        if _load_bagen_manifest(
            root, lock_context.sources["bagen_swebench"]
        ) != bagen_paths:
            raise DataFoundationBaselineError("BAGEN raw input set changed during run")
        spend_archive = spend_paths[0]
        if (
            _sha256_file(spend_archive)
            != lock_context.sources["spend_openhands"].raw_artifact_sha256
        ):
            raise DataFoundationBaselineError("Spend archive changed during baseline run")
        after_lock = load_lock_context(root, baseline_lock)
        if after_lock != lock_context:
            raise DataFoundationBaselineError("Data Foundation lock changed during baseline run")

    return publish_artifact(
        root,
        output,
        results=results,
        bundles=bundles,
        pre_publish_check=pre_publish_check,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the commit-bound empirical Data Foundation development-CV baseline."
        )
    )
    parser.add_argument("--repository-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--baseline-lock", default=DEFAULT_BASELINE_LOCK)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        artifact = run_baseline(
            repository_root=arguments.repository_root,
            baseline_lock=arguments.baseline_lock,
            output=arguments.output,
        )
        manifest = verify_artifact(artifact)
    except DataFoundationBaselineError as exc:
        print(f"Data Foundation prediction baseline failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "artifact_path": artifact.relative_to(Path(arguments.repository_root).resolve()).as_posix(),
                "artifact_id": manifest["artifact_id"],
                "manifest_sha256": _sha256_file(artifact / "manifest.json"),
                "evaluation_scope": "development_cross_validation_only",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
