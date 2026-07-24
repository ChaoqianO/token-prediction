"""Publish immutable, final-safe Stage 4 completion diagnostics.

The final holdout has already been opened exactly once.  This runner therefore
has no raw-source code path: it verifies immutable Stage 4 development
artifacts, their candidate checkpoints, and every seed-policy lifecycle bundle.
The checkpoints contain forecasts and aggregate development metrics, but not
labels or lifecycle sequences.  Forecast/cohort/aggregate parity is recomputed
exactly; label-dependent lifecycle extensions are recorded as unavailable
instead of reopening a mixed raw payload or fabricating values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from token_prediction.dataset import PredictionPosition, PredictionTarget
from token_prediction.development import (
    STAGE_SPLIT_SEEDS,
    TASK_PSEUDONYM_POLICY_ID as DEVELOPMENT_TASK_PSEUDONYM_POLICY_ID,
)
from token_prediction.evaluation import METRIC_SUITE_ID
from token_prediction.estimators.base import TokenForecast
from token_prediction.experiment import CandidateResult, PredictionRecord
from token_prediction.lifecycle_bundle import load_lifecycle_bundle
from token_prediction.lineage import publish_artifact, verify_artifact

if __package__:
    from scripts.run_data_foundation_baseline import (
        DEFAULT_BASELINE_LOCK,
        _is_link_or_reparse,
        _repo_path,
        _safe_relative,
    )
    from scripts.run_stage4_experiments import (
        STAGE4_TASK_PSEUDONYM_POLICY_ID,
        _artifact_key,
        _assert_aggregate_safe,
        _framed_code_hash,
        _git,
        _stage4_code_paths,
        cohort_projection_sha256,
        prediction_projection_sha256,
    )
    from scripts.summarize_stage4_completion import (
        EXPECTED_FINAL_HOLDOUT,
        POINT_ONLY_SEED_CANDIDATE_ID,
        RAW_SEED_CANDIDATE_ID,
        ArtifactReference,
        LoadedArtifact,
        _list,
        _mapping,
        _required_sha256,
        _required_string,
        _semantic_sha256,
        load_development_artifact,
    )
else:  # pragma: no cover - direct production CLI invocation
    from run_data_foundation_baseline import (
        DEFAULT_BASELINE_LOCK,
        _is_link_or_reparse,
        _repo_path,
        _safe_relative,
    )
    from run_stage4_experiments import (
        STAGE4_TASK_PSEUDONYM_POLICY_ID,
        _artifact_key,
        _assert_aggregate_safe,
        _framed_code_hash,
        _git,
        _stage4_code_paths,
        cohort_projection_sha256,
        prediction_projection_sha256,
    )
    from summarize_stage4_completion import (
        EXPECTED_FINAL_HOLDOUT,
        POINT_ONLY_SEED_CANDIDATE_ID,
        RAW_SEED_CANDIDATE_ID,
        ArtifactReference,
        LoadedArtifact,
        _list,
        _mapping,
        _required_sha256,
        _required_string,
        _semantic_sha256,
        load_development_artifact,
    )


RESULTS_SCHEMA_VERSION = 2
ARTIFACT_SCHEMA_VERSION = 2
STAGE_NAME = "stage4_completion_diagnostics"
POLICY_ID = "stage4_completion_artifact_checkpoint_only_v2"
DEFAULT_OUTPUT_ROOT = "workspace/stage4/completion_diagnostics"
ALLOWED_OUTPUT_ROOT = PurePosixPath(DEFAULT_OUTPUT_ROOT)
OUTPUT_KEY_HEX_LENGTH = 20
EXPECTED_SOURCE_COMMIT = "c1ac2484f44ed65705cdd00eba7b70a739a3ac0b"
EXPECTED_SOURCE_CODE_TREE_SHA256 = (
    "6418545afa08a39df1797486e4c845063c2de13b29f20c81500933fad2201757"
)
SOURCE_STAGE_NAME = "stage4_development_source"
SOURCE_ARTIFACT_SCHEMA_VERSION = 1
SOURCE_RUN_SEMANTIC_KEYS = {
    "results_schema_version",
    "run_policy_id",
    "checkpoint_policy_id",
    "source_name",
    "source_id",
    "revision",
    "raw_artifact_sha256",
    "data_foundation_baseline_lock_sha256",
    "base_dataset_id",
    "derived_dataset_id",
    "development_protocol_id",
    "matrix_id",
    "git_commit",
    "code_tree_sha256",
    "runtime_versions",
}
DIAGNOSTICS_SOURCE_TAG = "stage4-completion-diagnostics-source-v1"
DIAGNOSTICS_DIRECT_CODE_PATHS = frozenset(
    {
        "scripts/run_stage4_completion_diagnostics.py",
        "scripts/summarize_stage4_completion.py",
    }
)
EXPECTED_BOUND_SOURCE_ARTIFACT_COUNT = 4
EXPECTED_LIFECYCLE_SOURCE_COUNT = 3
EXPECTED_LIFECYCLE_CONDITION_COUNT = 7
EXPECTED_LIFECYCLE_CANDIDATE_COUNT = 2
EXPECTED_LIFECYCLE_CANDIDATE_CELL_COUNT = 14
EXPECTED_LIFECYCLE_CANDIDATE_SEED_COUNT = 42
EXPECTED_LIFECYCLE_BUNDLE_COUNT = 210
EXPECTED_LIFECYCLE_CANDIDATES = frozenset(
    {RAW_SEED_CANDIDATE_ID, POINT_ONLY_SEED_CANDIDATE_ID}
)
EXPECTED_LIFECYCLE_SOURCES = frozenset(
    {"bagen_sokoban", "bagen_swebench", "spend_openhands"}
)
EXPECTED_POINT_ONLY_SOURCES = frozenset({"spend_aggregate"})
EXPECTED_SOURCE_NAMES = EXPECTED_LIFECYCLE_SOURCES | EXPECTED_POINT_ONLY_SOURCES
EXPECTED_SOURCE_ARTIFACT_IDENTITIES = {
    "bagen_sokoban": {
        "run_id": "5eb71975b60aad8d844cdb99",
        "artifact_id": (
            "0cc652b52b32c7d636320023152e8d685dab8c7a5c39f677667aecf8e39d4747"
        ),
        "results_payload_sha256": (
            "1ef254633fce24b57a4f4c20a1e3d06c195a7bfb3ab3a9a415171bebe21ca245"
        ),
    },
    "bagen_swebench": {
        "run_id": "321ab0f381a0030e7026bb91",
        "artifact_id": (
            "ebfd502aee73272f393bfa5582d826dd81ec67771ca60f03e643b28ab61648e9"
        ),
        "results_payload_sha256": (
            "f7dec1f6415d8be0fbdd6ae509789bdfd6fa918f01b6f0facb953ab4a16627a4"
        ),
    },
    "spend_aggregate": {
        "run_id": "f9120ade982bc74f8e86755f",
        "artifact_id": (
            "0b2224c79923b99fadf6c5a95e65ff4514123008ea1e72ba97d07b1a59e734e5"
        ),
        "results_payload_sha256": (
            "29b5a3873aee063f1cc7def1183f93395770c35d85c72fa261f260e30511b506"
        ),
    },
    "spend_openhands": {
        "run_id": "00732b02ba8b15727b8d113d",
        "artifact_id": (
            "0d68ea3f1a5f30f24eefae900ed02cfe058077d6fa08e582a9a49b837b9d0eed"
        ),
        "results_payload_sha256": (
            "1fc60f15c676665800c2f2ace1da68159892ef9665051414306755c646532cde"
        ),
    },
}
SOURCE_ARTIFACT_KEYS = {
    "source_name",
    "source_id",
    "run_id",
    "artifact_id",
    "results_payload_sha256",
    "matrix_id",
    "development_protocol_id",
    "lifecycle_status",
}
BUNDLE_INVENTORY_KEYS = {
    "source_name",
    "condition_id",
    "experiment_id",
    "candidate_id",
    "candidate_hash",
    "split_seed",
    "split_plan_id",
    "fold",
    "bundle_relative_path",
    "bundle_manifest_sha256",
    "bundle_file_count",
    "load_status",
}
DIAGNOSTIC_KEYS = {
    "source_name",
    "condition_id",
    "experiment_id",
    "candidate_id",
    "candidate_hash",
    "split_seed",
    "split_plan_id",
    "bundle_folds",
    "bundle_projection_sha256",
    "checkpoint_parity",
    "lifecycle_metrics",
}
CHECKPOINT_PARITY_KEYS = {
    "status",
    "checkpoint_artifact_id",
    "checkpoint_result_sha256",
    "prediction_count",
    "expected_prediction_count",
    "prediction_projection_sha256",
    "expected_prediction_projection_sha256",
    "cohort_projection_sha256",
    "expected_cohort_projection_sha256",
    "aggregate_metrics_projection_sha256",
    "expected_aggregate_metrics_projection_sha256",
    "development_cohort_status",
    "development_task_count",
    "development_task_projection_sha256",
}
CHECKPOINT_RESULT_KEYS = {
    "candidate_id",
    "candidate_hash",
    "dataset_id",
    "split_plan_id",
    "eligibility_hash",
    "position",
    "target",
    "condition_id",
    "calibrator_id",
    "alpha",
    "metric_suite_id",
    "predictions",
    "metrics",
    "fold_metrics",
    "task_metrics",
    "fold_artifacts",
}
CHECKPOINT_FOLD_ARTIFACT_KEYS = {
    "fold",
    "encoder",
    "fit_report",
    "feature_importance",
    "model_strings",
    "bundle_files",
    "calibrator",
    "provenance",
}
LIFECYCLE_METRICS_KEYS = {
    "status",
    "reason_code",
    "labels_present",
    "lifecycle_sequences_present",
    "unavailable_metrics",
    "historical_stage3_reference",
}
COVERAGE_KEYS = {
    "bound_source_artifact_count",
    "lifecycle_source_count",
    "lifecycle_condition_count",
    "lifecycle_candidate_count",
    "lifecycle_candidate_cell_count",
    "lifecycle_candidate_seed_count",
    "lifecycle_bundle_count",
    "checkpoint_verified_candidate_seed_count",
    "lifecycle_replayed_candidate_seed_count",
    "lifecycle_metrics_unavailable_candidate_seed_count",
}
LIFECYCLE_UNAVAILABLE_REASON = (
    "no_presealed_development_lifecycle_projection_v1"
)
UNAVAILABLE_LIFECYCLE_METRICS = [
    "progress",
    "run_variance_iqr_max_minus_min",
    "termination",
]
RESULTS_KEYS = {
    "results_schema_version",
    "stage_name",
    "policy_id",
    "source_binding",
    "diagnostics_code_binding",
    "source_artifacts",
    "coverage",
    "bundle_inventory",
    "diagnostics",
    "final_holdout",
    "results_payload_sha256",
}
class Stage4CompletionDiagnosticsError(RuntimeError):
    """The diagnostics supplement cannot be produced safely."""


@dataclass(frozen=True)
class CompletionDiagnosticsSummary:
    run_id: str
    output_dir: Path
    artifact_id: str
    results_payload_sha256: str
    diagnostic_count: int
    bundle_count: int


@dataclass(frozen=True)
class VerifiedSourceArtifact:
    loaded: LoadedArtifact
    artifact_manifest: Any


@dataclass(frozen=True)
class AuditedDiagnostic:
    document: Mapping[str, object]
    inventory: tuple[Mapping[str, object], ...]


def _runner_sha256() -> str:
    path = Path(__file__)
    if _is_link_or_reparse(path) or not path.is_file():
        raise Stage4CompletionDiagnosticsError("diagnostics runner origin is unsafe")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_runner_origin(root: Path) -> None:
    expected = _repo_path(
        root,
        "scripts/run_stage4_completion_diagnostics.py",
        label="completion diagnostics runner",
    )
    actual = Path(__file__)
    if _is_link_or_reparse(actual) or actual.resolve() != expected.resolve():
        raise Stage4CompletionDiagnosticsError(
            "executing diagnostics runner is outside repository_root"
        )


def _safe_output_parent(root: Path, value: str) -> Path:
    try:
        relative = _safe_relative(value, label="completion diagnostics output root")
    except Exception as exc:
        raise Stage4CompletionDiagnosticsError(
            "completion diagnostics output root is unsafe"
        ) from exc
    if PurePosixPath(relative) != ALLOWED_OUTPUT_ROOT:
        raise Stage4CompletionDiagnosticsError(
            f"completion diagnostics output root must be {DEFAULT_OUTPUT_ROOT!r}"
        )
    parent = _repo_path(root, relative, label="completion diagnostics output root")
    if parent.exists() and _is_link_or_reparse(parent):
        raise Stage4CompletionDiagnosticsError(
            "completion diagnostics output root is linked"
        )
    return parent


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _assert_plain_lexical_ancestry(
    path: Path,
    *,
    label: str,
    require_exists: bool = True,
) -> Path:
    """Reject a link/reparse point in the lexical path before resolving it."""

    lexical = _absolute_lexical(path)
    chain = [lexical, *lexical.parents]
    for current in reversed(chain):
        if current.exists() or _is_link_or_reparse(current):
            if _is_link_or_reparse(current):
                raise Stage4CompletionDiagnosticsError(
                    f"{label} traverses a symlink, junction, or reparse point"
                )
    if require_exists and not lexical.exists():
        raise Stage4CompletionDiagnosticsError(f"{label} does not exist")
    return lexical


def _direct_artifact_references(
    root: Path,
    artifact_inputs: Sequence[str | os.PathLike[str]],
) -> tuple[ArtifactReference, ...]:
    """Authorize canonical development artifacts before any payload read."""

    if not artifact_inputs:
        raise Stage4CompletionDiagnosticsError(
            "at least one Stage 4 development artifact is required"
        )
    try:
        runs_root = _repo_path(
            root,
            "workspace/stage4/runs",
            label="Stage 4 development runs root",
        )
    except Exception as exc:
        raise Stage4CompletionDiagnosticsError(
            "Stage 4 development runs root traverses a symlink, junction, "
            "or reparse point"
        ) from exc
    runs_root = _assert_plain_lexical_ancestry(
        runs_root,
        label="Stage 4 development runs root",
    )
    references: list[ArtifactReference] = []
    seen: set[Path] = set()
    for raw in artifact_inputs:
        raw_text = os.fspath(raw)
        raw_parts = raw_text.replace("\\", "/").split("/")
        supplied = Path(raw_text)
        if (
            not raw_text
            or raw_text != raw_text.strip()
            or "\x00" in raw_text
            or any(part in {".", ".."} for part in raw_parts)
            or (not supplied.is_absolute() and "\\" in raw_text)
        ):
            raise Stage4CompletionDiagnosticsError(
                "Stage 4 artifact input path is not canonical"
            )
        if not supplied.is_absolute():
            supplied = root / supplied
        supplied = _absolute_lexical(supplied)
        if supplied.name == "results.json":
            artifact_path = supplied.parent
            results_path = supplied
        else:
            if supplied.suffix.casefold() == ".json":
                raise Stage4CompletionDiagnosticsError(
                    "release-lock and arbitrary JSON artifact inputs are forbidden"
                )
            artifact_path = supplied
            results_path = artifact_path / "results.json"
        artifact_path = _assert_plain_lexical_ancestry(
            artifact_path,
            label="Stage 4 development artifact",
        )
        if artifact_path.parent != runs_root:
            raise Stage4CompletionDiagnosticsError(
                "artifact must be one canonical directory directly below "
                "workspace/stage4/runs"
            )
        key = artifact_path.name
        if (
            len(key) != 23
            or not key.startswith("s4-")
            or any(character not in "0123456789abcdef" for character in key[3:])
        ):
            raise Stage4CompletionDiagnosticsError(
                "Stage 4 development artifact directory name is not canonical"
            )
        if not artifact_path.is_dir():
            raise Stage4CompletionDiagnosticsError(
                "Stage 4 development artifact must be a directory"
            )
        _assert_plain_lexical_ancestry(
            results_path,
            label="Stage 4 development results",
        )
        if not results_path.is_file():
            raise Stage4CompletionDiagnosticsError(
                "Stage 4 development results.json is missing"
            )
        if artifact_path in seen:
            raise Stage4CompletionDiagnosticsError(
                "duplicate Stage 4 development artifact input"
            )
        seen.add(artifact_path)
        references.append(ArtifactReference(artifact_path))
    return tuple(references)


def _output_key(run_id: str) -> str:
    if (
        len(run_id) < OUTPUT_KEY_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in run_id)
    ):
        raise Stage4CompletionDiagnosticsError("diagnostics run id is invalid")
    return f"s4diag-{run_id[:OUTPUT_KEY_HEX_LENGTH]}"


def _source_binding(
    root: Path,
    artifacts: Sequence[VerifiedSourceArtifact],
) -> tuple[dict[str, str], tuple[str, ...]]:
    bindings = []
    for artifact in artifacts:
        bindings.append(
            _mapping(
                artifact.loaded.document.get("code_binding"),
                name="Stage 4 source code binding",
            )
        )
    commits = {binding.get("git_commit") for binding in bindings}
    hashes = {binding.get("code_tree_sha256") for binding in bindings}
    path_sets = {
        tuple(
            _required_string(item, name="Stage 4 code path")
            for item in _list(binding.get("code_paths"), name="Stage 4 code paths")
        )
        for binding in bindings
    }
    if commits != {EXPECTED_SOURCE_COMMIT} or hashes != {
        EXPECTED_SOURCE_CODE_TREE_SHA256
    }:
        raise Stage4CompletionDiagnosticsError(
            "source artifacts are not bound to the c1ac248 completion source"
        )
    if len(path_sets) != 1:
        raise Stage4CompletionDiagnosticsError(
            "source artifacts disagree on the Stage 4 code path set"
        )
    paths = next(iter(path_sets))
    if paths != tuple(sorted(set(paths))) or paths != _stage4_code_paths(root):
        raise Stage4CompletionDiagnosticsError(
            "source artifact code paths differ from the frozen Stage 4 path set"
        )
    workspace_items: list[tuple[str, bytes]] = []
    source_commit_items: list[tuple[str, bytes]] = []
    for relative in paths:
        path = _repo_path(root, relative, label="frozen Stage 4 source path")
        if _is_link_or_reparse(path) or not path.is_file():
            raise Stage4CompletionDiagnosticsError(
                "frozen Stage 4 source path is unsafe"
            )
        workspace = path.read_bytes()
        try:
            committed = _git(root, "show", f"{EXPECTED_SOURCE_COMMIT}:{relative}")
        except Exception as exc:
            raise Stage4CompletionDiagnosticsError(
                "cannot read the frozen Stage 4 source commit"
            ) from exc
        if workspace != committed:
            raise Stage4CompletionDiagnosticsError(
                f"workspace source differs from c1ac248: {relative}"
            )
        workspace_items.append((relative, workspace))
        source_commit_items.append((relative, committed))
    workspace_hash = _framed_code_hash(workspace_items)
    if (
        workspace_hash != EXPECTED_SOURCE_CODE_TREE_SHA256
        or _framed_code_hash(source_commit_items) != EXPECTED_SOURCE_CODE_TREE_SHA256
    ):
        raise Stage4CompletionDiagnosticsError(
            "frozen Stage 4 source tree hash does not close"
        )
    return (
        {
            "git_commit": EXPECTED_SOURCE_COMMIT,
            "code_tree_sha256": EXPECTED_SOURCE_CODE_TREE_SHA256,
        },
        paths,
    )


def capture_diagnostics_code_binding(root: Path) -> dict[str, object]:
    """Bind the clean committed replay implementation, distinct from training."""

    paths = tuple(
        sorted(set(_stage4_code_paths(root)) | DIAGNOSTICS_DIRECT_CODE_PATHS)
    )
    tracked = {
        item.decode("utf-8", errors="strict")
        for item in _git(
            root,
            "ls-files",
            "-z",
            "--",
            *sorted(DIAGNOSTICS_DIRECT_CODE_PATHS),
        ).split(b"\0")
        if item
    }
    if tracked != DIAGNOSTICS_DIRECT_CODE_PATHS:
        raise Stage4CompletionDiagnosticsError(
            "diagnostics runner dependencies must be committed before replay"
        )
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--",
        *paths,
    )
    if status:
        raise Stage4CompletionDiagnosticsError(
            "diagnostics code paths must be clean at HEAD"
        )
    commit = _git(root, "rev-parse", "--verify", "HEAD^{commit}").decode(
        "ascii"
    ).strip()
    if (
        len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise Stage4CompletionDiagnosticsError(
            "diagnostics HEAD is not a full Git commit"
        )
    try:
        tagged = _git(
            root,
            "rev-parse",
            "--verify",
            f"refs/tags/{DIAGNOSTICS_SOURCE_TAG}^{{commit}}",
        ).decode("ascii").strip()
    except Exception as exc:
        raise Stage4CompletionDiagnosticsError(
            "diagnostics source tag is missing"
        ) from exc
    if tagged != commit:
        raise Stage4CompletionDiagnosticsError(
            "diagnostics source tag does not point to HEAD"
        )
    items: list[tuple[str, bytes]] = []
    for relative in paths:
        path = _repo_path(root, relative, label="diagnostics code path")
        if _is_link_or_reparse(path) or not path.is_file():
            raise Stage4CompletionDiagnosticsError(
                "diagnostics code path is unsafe"
            )
        workspace = path.read_bytes()
        committed = _git(root, "show", f"{commit}:{relative}")
        if workspace != committed:
            raise Stage4CompletionDiagnosticsError(
                f"diagnostics workspace differs from HEAD: {relative}"
            )
        items.append((relative, workspace))
    return {
        "git_commit": commit,
        "code_tree_sha256": _framed_code_hash(items),
        "code_paths": list(paths),
    }


def _source_run_identity(
    artifact: VerifiedSourceArtifact,
) -> tuple[str, str, Mapping[str, object]]:
    manifest = artifact.artifact_manifest
    metadata = _mapping(
        manifest.metadata,
        name="Stage 4 source artifact metadata",
    )
    if set(metadata) != {
        "run_id",
        "run_semantic",
        "results_payload_sha256",
    }:
        raise Stage4CompletionDiagnosticsError(
            "Stage 4 source artifact metadata scope differs"
        )
    run_semantic = _mapping(
        metadata.get("run_semantic"),
        name="Stage 4 source run semantic",
    )
    run_semantic_sha256 = _semantic_sha256(run_semantic)
    run_id = _required_string(
        metadata.get("run_id"),
        name="Stage 4 source run id",
    )
    if (
        len(run_id) != 24
        or any(character not in "0123456789abcdef" for character in run_id)
        or run_id != run_semantic_sha256[:24]
        or artifact.loaded.document.get("run_id") != run_id
        or metadata.get("results_payload_sha256")
        != artifact.loaded.results_payload_sha256
    ):
        raise Stage4CompletionDiagnosticsError(
            "Stage 4 source run semantic identity does not close"
        )
    return run_id, run_semantic_sha256, run_semantic


def _verify_source_manifest_scope(artifact: VerifiedSourceArtifact) -> None:
    manifest = artifact.artifact_manifest
    if (
        manifest.stage_name != SOURCE_STAGE_NAME
        or manifest.schema_version != SOURCE_ARTIFACT_SCHEMA_VERSION
        or "results.json" not in manifest.files
        or any(
            relative != "results.json"
            and not relative.startswith("fold_artifacts/")
            for relative in manifest.files
        )
    ):
        raise Stage4CompletionDiagnosticsError(
            "Stage 4 source artifact manifest scope differs"
        )
    _run_id, _run_semantic_sha256, run_semantic = _source_run_identity(artifact)
    if set(run_semantic) != SOURCE_RUN_SEMANTIC_KEYS:
        raise Stage4CompletionDiagnosticsError(
            "Stage 4 source run semantic schema differs"
        )
    results = artifact.loaded.document
    source = _mapping(results.get("source"), name="Stage 4 source")
    dataset = _mapping(results.get("dataset"), name="Stage 4 dataset")
    matrix = _mapping(results.get("matrix"), name="Stage 4 matrix")
    protocol = _mapping(
        results.get("development_protocol"),
        name="Stage 4 development protocol",
    )
    code_binding = _mapping(
        results.get("code_binding"),
        name="Stage 4 code binding",
    )
    expected = {
        "results_schema_version": results.get("results_schema_version"),
        "source_name": source.get("source_name"),
        "source_id": source.get("source_id"),
        "revision": source.get("revision"),
        "raw_artifact_sha256": source.get("raw_artifact_sha256"),
        "base_dataset_id": dataset.get("base_dataset_id"),
        "derived_dataset_id": dataset.get("derived_dataset_id"),
        "development_protocol_id": protocol.get("protocol_id"),
        "matrix_id": matrix.get("matrix_id"),
        "git_commit": code_binding.get("git_commit"),
        "code_tree_sha256": code_binding.get("code_tree_sha256"),
        "runtime_versions": results.get("runtime_versions"),
    }
    if any(run_semantic.get(key) != value for key, value in expected.items()):
        raise Stage4CompletionDiagnosticsError(
            "Stage 4 source run semantic differs from its results"
        )
    for key in (
        "raw_artifact_sha256",
        "data_foundation_baseline_lock_sha256",
        "development_protocol_id",
        "matrix_id",
        "code_tree_sha256",
    ):
        _required_sha256(
            run_semantic.get(key),
            name=f"Stage 4 source run semantic {key}",
        )
    for key in ("run_policy_id", "checkpoint_policy_id"):
        _required_string(
            run_semantic.get(key),
            name=f"Stage 4 source run semantic {key}",
        )
    for key in ("base_dataset_id", "derived_dataset_id"):
        _required_string(
            run_semantic.get(key),
            name=f"Stage 4 source run semantic {key}",
        )
    if (
        run_semantic.get("git_commit") != EXPECTED_SOURCE_COMMIT
        or run_semantic.get("code_tree_sha256")
        != EXPECTED_SOURCE_CODE_TREE_SHA256
    ):
        raise Stage4CompletionDiagnosticsError(
            "Stage 4 source run semantic code binding differs"
        )


def _verify_pinned_source_identity(
    artifact: VerifiedSourceArtifact,
    *,
    source_name: str,
) -> None:
    expected = EXPECTED_SOURCE_ARTIFACT_IDENTITIES.get(source_name)
    if expected is None or {
        "run_id": artifact.loaded.document.get("run_id"),
        "artifact_id": artifact.loaded.artifact_id,
        "results_payload_sha256": artifact.loaded.results_payload_sha256,
    } != expected or artifact.loaded.path.name != (
        "s4-" + str(expected["run_id"])[:20]
    ):
        raise Stage4CompletionDiagnosticsError(
            "Stage 4 source artifact differs from the frozen completion identity"
        )


def _verify_source_artifacts(
    references: Sequence[ArtifactReference],
) -> tuple[VerifiedSourceArtifact, ...]:
    if len(references) != EXPECTED_BOUND_SOURCE_ARTIFACT_COUNT:
        raise Stage4CompletionDiagnosticsError(
            "diagnostics require exactly four Stage 4 development artifacts"
        )
    verified: list[VerifiedSourceArtifact] = []
    for reference in references:
        try:
            artifact_manifest = verify_artifact(reference.path)
            loaded = load_development_artifact(reference)
        except Exception as exc:
            raise Stage4CompletionDiagnosticsError(
                f"Stage 4 development artifact verification failed: {reference.path}"
            ) from exc
        if artifact_manifest.artifact_id != loaded.artifact_id:
            raise Stage4CompletionDiagnosticsError(
                "artifact manifest and aggregate loader identities differ"
            )
        artifact = VerifiedSourceArtifact(loaded, artifact_manifest)
        _verify_source_manifest_scope(artifact)
        verified.append(artifact)
    by_source: dict[str, VerifiedSourceArtifact] = {}
    for artifact in verified:
        source = _mapping(
            artifact.loaded.document.get("source"), name="Stage 4 source"
        )
        source_name = _required_string(
            source.get("source_name"), name="Stage 4 source name"
        )
        if source_name in by_source:
            raise Stage4CompletionDiagnosticsError(
                "duplicate Stage 4 source artifact"
            )
        _verify_pinned_source_identity(
            artifact,
            source_name=source_name,
        )
        by_source[source_name] = artifact
    if set(by_source) != EXPECTED_SOURCE_NAMES:
        raise Stage4CompletionDiagnosticsError(
            "Stage 4 artifacts do not cover the four frozen sources"
        )
    return tuple(by_source[name] for name in sorted(by_source))


def _candidate_documents(
    experiment: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    documents: dict[str, Mapping[str, object]] = {}
    for index, item in enumerate(
        _list(experiment.get("candidates"), name="experiment candidates")
    ):
        candidate = _mapping(item, name=f"experiment candidates[{index}]")
        candidate_id = _required_string(
            candidate.get("candidate_id"), name="candidate id"
        )
        if candidate_id in documents:
            raise Stage4CompletionDiagnosticsError(
                "experiment repeats a candidate id"
            )
        documents[candidate_id] = candidate
    return documents


def _seed_results(
    candidate: Mapping[str, object],
) -> dict[int, Mapping[str, object]]:
    resolved: dict[int, Mapping[str, object]] = {}
    for index, item in enumerate(
        _list(candidate.get("seed_results"), name="candidate seed results")
    ):
        seed_result = _mapping(item, name=f"candidate seed results[{index}]")
        seed = seed_result.get("split_seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise Stage4CompletionDiagnosticsError("candidate split seed is invalid")
        if seed in resolved:
            raise Stage4CompletionDiagnosticsError(
                "candidate repeats a split seed"
            )
        resolved[seed] = seed_result
    if tuple(resolved) != STAGE_SPLIT_SEEDS:
        raise Stage4CompletionDiagnosticsError(
            "candidate does not contain the frozen split seeds in order"
        )
    return resolved


def _safe_compact_key(value: object, *, prefix: str, name: str) -> str:
    text = _required_string(value, name=name)
    if (
        len(text) != 18
        or not text.startswith(prefix + "_")
        or any(character not in "0123456789abcdef" for character in text[2:])
    ):
        raise Stage4CompletionDiagnosticsError(f"{name} is not canonical")
    return text


def _bundle_files(path: Path) -> tuple[Path, ...]:
    files = tuple(sorted(item for item in path.rglob("*") if item.is_file()))
    if not files:
        raise Stage4CompletionDiagnosticsError("lifecycle bundle is empty")
    for item in files:
        if _is_link_or_reparse(item):
            raise Stage4CompletionDiagnosticsError(
                "lifecycle bundle contains a linked file"
            )
        try:
            item.resolve().relative_to(path.resolve())
        except ValueError as exc:
            raise Stage4CompletionDiagnosticsError(
                "lifecycle bundle file escaped its root"
            ) from exc
    return files


def _bundle_projection(inventory: Sequence[Mapping[str, object]]) -> str:
    projection = [
        {key: item[key] for key in sorted(BUNDLE_INVENTORY_KEYS)}
        for item in sorted(inventory, key=lambda value: int(value["fold"]))
    ]
    return _semantic_sha256(projection)


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Stage4CompletionDiagnosticsError(f"{name} must be numeric")
    resolved = float(value)
    if not (-float("inf") < resolved < float("inf")):
        raise Stage4CompletionDiagnosticsError(f"{name} must be finite")
    return resolved


def _strict_json_object(payload: str, *, name: str) -> Mapping[str, object]:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise Stage4CompletionDiagnosticsError(
                    f"{name} contains a duplicate JSON key"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            payload,
            object_pairs_hook=unique,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                Stage4CompletionDiagnosticsError(
                    f"{name} contains non-finite JSON constant {constant}"
                )
            ),
        )
    except json.JSONDecodeError as exc:
        raise Stage4CompletionDiagnosticsError(
            f"{name} is not valid JSON"
        ) from exc
    return _mapping(value, name=name)


def _checkpoint_prediction_record(
    value: object,
    *,
    candidate_id: str,
) -> PredictionRecord:
    item = _mapping(value, name="candidate checkpoint prediction")
    if set(item) != {
        "candidate_id",
        "point_id",
        "task_id",
        "trajectory_id",
        "condition_id",
        "fold",
        "target",
        "forecast",
        "sample_weight",
    }:
        raise Stage4CompletionDiagnosticsError(
            "candidate checkpoint prediction schema differs"
        )
    if item.get("candidate_id") != candidate_id:
        raise Stage4CompletionDiagnosticsError(
            "candidate checkpoint prediction identity differs"
        )
    forecast_document = _mapping(
        item.get("forecast"), name="candidate checkpoint forecast"
    )
    if set(forecast_document) != {
        "point_id",
        "target",
        "lower",
        "point",
        "upper",
        "latency_ms",
        "overhead_input_tokens",
        "overhead_output_tokens",
        "raw_lower",
        "raw_point",
        "raw_upper",
    }:
        raise Stage4CompletionDiagnosticsError(
            "candidate checkpoint forecast schema differs"
        )
    point_id = _required_string(item.get("point_id"), name="checkpoint point id")
    target_text = _required_string(item.get("target"), name="checkpoint target")
    if (
        forecast_document.get("point_id") != point_id
        or forecast_document.get("target") != target_text
    ):
        raise Stage4CompletionDiagnosticsError(
            "candidate checkpoint forecast scope differs"
        )
    raw_values: list[float | None] = []
    for name in ("raw_lower", "raw_point", "raw_upper"):
        raw = forecast_document.get(name)
        raw_values.append(
            None
            if raw is None
            else _finite_number(raw, name=f"checkpoint forecast {name}")
        )
    if any(value is None for value in raw_values) != all(
        value is None for value in raw_values
    ):
        raise Stage4CompletionDiagnosticsError(
            "candidate checkpoint raw forecast is partial"
        )
    fold = item.get("fold")
    for name in ("overhead_input_tokens", "overhead_output_tokens"):
        overhead = forecast_document.get(name)
        if (
            isinstance(overhead, bool)
            or not isinstance(overhead, int)
            or overhead < 0
        ):
            raise Stage4CompletionDiagnosticsError(
                f"candidate checkpoint {name} is invalid"
            )
    if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(5):
        raise Stage4CompletionDiagnosticsError(
            "candidate checkpoint prediction fold is invalid"
        )
    try:
        target = PredictionTarget(target_text)
        forecast = TokenForecast(
            point_id=point_id,
            target=target,
            lower=_finite_number(
                forecast_document.get("lower"), name="checkpoint forecast lower"
            ),
            point=_finite_number(
                forecast_document.get("point"), name="checkpoint forecast point"
            ),
            upper=_finite_number(
                forecast_document.get("upper"), name="checkpoint forecast upper"
            ),
            latency_ms=_finite_number(
                forecast_document.get("latency_ms"),
                name="checkpoint forecast latency",
            ),
            overhead_input_tokens=int(forecast_document["overhead_input_tokens"]),
            overhead_output_tokens=int(
                forecast_document["overhead_output_tokens"]
            ),
            raw_lower=raw_values[0],
            raw_point=raw_values[1],
            raw_upper=raw_values[2],
        )
    except (TypeError, ValueError) as exc:
        raise Stage4CompletionDiagnosticsError(
            "candidate checkpoint forecast is invalid"
        ) from exc
    return PredictionRecord(
        candidate_id=candidate_id,
        point_id=point_id,
        task_id=_required_string(item.get("task_id"), name="checkpoint task id"),
        trajectory_id=_required_string(
            item.get("trajectory_id"), name="checkpoint trajectory id"
        ),
        condition_id=_required_string(
            item.get("condition_id"), name="checkpoint condition"
        ),
        fold=fold,
        target=target,
        forecast=forecast,
        sample_weight=_finite_number(
            item.get("sample_weight"), name="checkpoint sample weight"
        ),
    )


def _task_metric_projection(
    result: CandidateResult,
) -> list[dict[str, object]]:
    return sorted(
        (
            {
                "task_pseudonym": hashlib.sha256(
                    (
                        f"{STAGE4_TASK_PSEUDONYM_POLICY_ID}\0"
                        f"{result.split_plan_id}\0{task_id}"
                    ).encode()
                ).hexdigest(),
                **dict(metrics),
            }
            for task_id, metrics in result.task_metrics.items()
        ),
        key=lambda item: str(item["task_pseudonym"]),
    )


def _aggregate_metrics_projection(
    *,
    metrics: Mapping[str, object],
    fold_metrics: Mapping[str, object],
    task_metrics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "metrics": dict(metrics),
        "fold_metrics": dict(fold_metrics),
        "task_metric_policy_id": STAGE4_TASK_PSEUDONYM_POLICY_ID,
        "task_metrics": [dict(item) for item in task_metrics],
    }


def _checkpoint_parity(
    *,
    root: Path,
    artifact: VerifiedSourceArtifact,
    experiment: Mapping[str, object],
    candidate: Mapping[str, object],
    split_seed: int,
    seed_result: Mapping[str, object],
    source_provenance_hash: str,
) -> dict[str, object]:
    candidate_id = _required_string(
        candidate.get("candidate_id"), name="checkpoint candidate id"
    )
    candidate_hash = _required_sha256(
        candidate.get("candidate_hash"), name="checkpoint candidate hash"
    )
    comparability = _list(
        seed_result.get("comparability_key"), name="checkpoint comparability key"
    )
    if len(comparability) != 9:
        raise Stage4CompletionDiagnosticsError(
            "checkpoint comparability key length differs"
        )
    execution_key = {
        "experiment_id": _required_string(
            experiment.get("experiment_id"), name="checkpoint experiment id"
        ),
        "candidate_id": candidate_id,
        "candidate_hash": candidate_hash,
        "dataset_id": _required_string(
            comparability[0], name="checkpoint dataset id"
        ),
        "split_plan_id": _required_sha256(
            seed_result.get("split_plan_id"), name="checkpoint split plan id"
        ),
        "split_seed": split_seed,
        "eligibility_hash": _required_sha256(
            comparability[2], name="checkpoint eligibility hash"
        ),
        "position": _required_string(
            experiment.get("position"), name="checkpoint position"
        ),
        "target": _required_string(
            experiment.get("target"), name="checkpoint target"
        ),
        "condition_id": _required_string(
            experiment.get("condition_id"), name="checkpoint condition"
        ),
        "calibrator_id": _required_string(
            experiment.get("calibrator_id"), name="checkpoint calibrator"
        ),
        "alpha": _finite_number(
            experiment.get("alpha"), name="checkpoint alpha"
        ),
        "source_provenance_hash": _required_sha256(
            source_provenance_hash, name="checkpoint source provenance hash"
        ),
    }
    expected_comparability = [
        execution_key["dataset_id"],
        execution_key["split_plan_id"],
        execution_key["eligibility_hash"],
        execution_key["position"],
        execution_key["target"],
        execution_key["condition_id"],
        execution_key["calibrator_id"],
        str(execution_key["alpha"]),
        METRIC_SUITE_ID,
    ]
    if comparability != expected_comparability:
        raise Stage4CompletionDiagnosticsError(
            "checkpoint comparability key differs from the execution identity"
        )
    execution_hash = _semantic_sha256(execution_key)
    run_id, run_semantic_sha256, _run_semantic = _source_run_identity(artifact)
    checkpoint = _repo_path(
        root,
        (
            f"workspace/stage4/checkpoints/{run_id}/candidates/"
            f"{execution_hash}"
        ),
        label="candidate checkpoint",
    )
    if _is_link_or_reparse(checkpoint) or not checkpoint.is_dir():
        raise Stage4CompletionDiagnosticsError(
            "candidate checkpoint is missing or unsafe"
        )
    try:
        manifest = verify_artifact(checkpoint)
    except Exception as exc:
        raise Stage4CompletionDiagnosticsError(
            "candidate checkpoint artifact failed verification"
        ) from exc
    if (
        manifest.stage_name != "development_candidate_checkpoint"
        or manifest.schema_version != 1
        or set(manifest.files) != {"candidate_result.json"}
        or manifest.metadata
        != {
            "candidate_execution_hash": execution_hash,
            "run_id": run_id,
            "run_semantic_sha256": run_semantic_sha256,
        }
    ):
        raise Stage4CompletionDiagnosticsError(
            "candidate checkpoint artifact identity differs"
        )
    try:
        document = _strict_json_object(
            (checkpoint / "candidate_result.json").read_text(encoding="utf-8"),
            name="candidate checkpoint document",
        )
    except (OSError, UnicodeError, Stage4CompletionDiagnosticsError) as exc:
        raise Stage4CompletionDiagnosticsError(
            "candidate checkpoint document is invalid"
        ) from exc
    wrapper = _mapping(document, name="candidate checkpoint document")
    if set(wrapper) != {
        "checkpoint_schema_version",
        "execution_key",
        "result",
        "result_sha256",
    } or wrapper.get("checkpoint_schema_version") != 1:
        raise Stage4CompletionDiagnosticsError(
            "candidate checkpoint document schema differs"
        )
    if _mapping(wrapper.get("execution_key"), name="checkpoint execution key") != (
        execution_key
    ):
        raise Stage4CompletionDiagnosticsError(
            "candidate checkpoint execution key differs"
        )
    result_document = _mapping(
        wrapper.get("result"), name="candidate checkpoint result"
    )
    if set(result_document) != CHECKPOINT_RESULT_KEYS:
        raise Stage4CompletionDiagnosticsError(
            "candidate checkpoint result schema differs"
        )
    fold_artifacts = [
        _mapping(item, name="candidate checkpoint fold artifact")
        for item in _list(
            result_document.get("fold_artifacts"),
            name="candidate checkpoint fold artifacts",
        )
    ]
    if (
        len(fold_artifacts) != 5
        or any(set(item) != CHECKPOINT_FOLD_ARTIFACT_KEYS for item in fold_artifacts)
        or [item.get("fold") for item in fold_artifacts] != list(range(5))
    ):
        raise Stage4CompletionDiagnosticsError(
            "candidate checkpoint fold artifact schema differs"
        )
    result_sha256 = _required_sha256(
        wrapper.get("result_sha256"), name="candidate checkpoint result SHA-256"
    )
    if _semantic_sha256(result_document) != result_sha256:
        raise Stage4CompletionDiagnosticsError(
            "candidate checkpoint result checksum does not close"
        )
    for key, expected in (
        ("candidate_id", candidate_id),
        ("candidate_hash", candidate_hash),
        ("dataset_id", execution_key["dataset_id"]),
        ("split_plan_id", execution_key["split_plan_id"]),
        ("eligibility_hash", execution_key["eligibility_hash"]),
        ("position", execution_key["position"]),
        ("target", execution_key["target"]),
        ("condition_id", execution_key["condition_id"]),
        ("calibrator_id", execution_key["calibrator_id"]),
        ("alpha", execution_key["alpha"]),
        ("metric_suite_id", METRIC_SUITE_ID),
    ):
        if result_document.get(key) != expected:
            raise Stage4CompletionDiagnosticsError(
                f"candidate checkpoint result {key} differs"
            )
    records = tuple(
        _checkpoint_prediction_record(item, candidate_id=candidate_id)
        for item in _list(
            result_document.get("predictions"),
            name="candidate checkpoint predictions",
        )
    )
    fold_metrics_document = _mapping(
        result_document.get("fold_metrics"),
        name="candidate checkpoint fold metrics",
    )
    task_metrics_document = _mapping(
        result_document.get("task_metrics"),
        name="candidate checkpoint task metrics",
    )
    result = CandidateResult(
        candidate_id=candidate_id,
        candidate_hash=candidate_hash,
        dataset_id=str(execution_key["dataset_id"]),
        split_plan_id=str(execution_key["split_plan_id"]),
        eligibility_hash=str(execution_key["eligibility_hash"]),
        position=PredictionPosition(str(execution_key["position"])),
        target=PredictionTarget(str(execution_key["target"])),
        condition_id=str(execution_key["condition_id"]),
        calibrator_id=str(execution_key["calibrator_id"]),
        alpha=float(execution_key["alpha"]),
        metric_suite_id=METRIC_SUITE_ID,
        predictions=records,
        metrics=dict(
            _mapping(
                result_document.get("metrics"),
                name="candidate checkpoint metrics",
            )
        ),
        fold_metrics={
            int(key): dict(_mapping(value, name="checkpoint fold metric"))
            for key, value in fold_metrics_document.items()
            if isinstance(key, str) and key.isdecimal()
        },
        task_metrics={
            str(key): dict(_mapping(value, name="checkpoint task metric"))
            for key, value in task_metrics_document.items()
        },
    )
    observed_aggregate = _aggregate_metrics_projection(
        metrics=result.metrics,
        fold_metrics={
            str(fold): dict(metrics)
            for fold, metrics in result.fold_metrics.items()
        },
        task_metrics=_task_metric_projection(result),
    )
    expected_aggregate = _aggregate_metrics_projection(
        metrics=_mapping(
            seed_result.get("metrics"), name="finalized seed metrics"
        ),
        fold_metrics=_mapping(
            seed_result.get("fold_metrics"), name="finalized seed fold metrics"
        ),
        task_metrics=[
            _mapping(item, name="finalized seed task metric")
            for item in _list(
                seed_result.get("task_metrics"),
                name="finalized seed task metrics",
            )
        ],
    )
    observed_prediction = prediction_projection_sha256(result)
    observed_cohort = cohort_projection_sha256(result)
    expected_prediction = _required_sha256(
        seed_result.get("prediction_projection_sha256"),
        name="finalized prediction projection SHA-256",
    )
    expected_cohort = _required_sha256(
        seed_result.get("cohort_projection_sha256"),
        name="finalized cohort projection SHA-256",
    )
    expected_count = seed_result.get("prediction_count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
        or len(records) != expected_count
        or observed_prediction != expected_prediction
        or observed_cohort != expected_cohort
        or observed_aggregate != expected_aggregate
    ):
        raise Stage4CompletionDiagnosticsError(
            "candidate checkpoint differs from its finalized seed result"
        )
    protocol = _mapping(
        artifact.loaded.document.get("development_protocol"),
        name="checkpoint development protocol",
    )
    holdout = _mapping(
        protocol.get("permanent_holdout"),
        name="checkpoint permanent holdout",
    )
    assignment_rows = [
        _mapping(item, name="checkpoint holdout assignment")
        for item in _list(
            holdout.get("assignments"),
            name="checkpoint holdout assignments",
        )
    ]
    assignments: dict[str, str] = {}
    for item in assignment_rows:
        if set(item) != {"task_pseudonym", "cohort"}:
            raise Stage4CompletionDiagnosticsError(
                "checkpoint holdout assignment schema differs"
            )
        pseudonym = _required_sha256(
            item.get("task_pseudonym"),
            name="checkpoint holdout task pseudonym",
        )
        cohort = _required_string(
            item.get("cohort"), name="checkpoint holdout cohort"
        )
        if pseudonym in assignments or cohort not in {
            "development",
            "final_holdout",
        }:
            raise Stage4CompletionDiagnosticsError(
                "checkpoint holdout assignments are invalid"
            )
        assignments[pseudonym] = cohort
    task_pseudonyms = sorted(
        {
            hashlib.sha256(
                (
                    f"{DEVELOPMENT_TASK_PSEUDONYM_POLICY_ID}\0"
                    f"{record.task_id}"
                ).encode()
            ).hexdigest()
            for record in records
        }
    )
    if not task_pseudonyms or any(
        assignments.get(pseudonym) != "development"
        for pseudonym in task_pseudonyms
    ):
        raise Stage4CompletionDiagnosticsError(
            "candidate checkpoint contains a final or unassigned task"
        )
    aggregate_sha256 = _semantic_sha256(observed_aggregate)
    return {
        "status": "exact",
        "checkpoint_artifact_id": manifest.artifact_id,
        "checkpoint_result_sha256": result_sha256,
        "prediction_count": len(records),
        "expected_prediction_count": expected_count,
        "prediction_projection_sha256": observed_prediction,
        "expected_prediction_projection_sha256": expected_prediction,
        "cohort_projection_sha256": observed_cohort,
        "expected_cohort_projection_sha256": expected_cohort,
        "aggregate_metrics_projection_sha256": aggregate_sha256,
        "expected_aggregate_metrics_projection_sha256": _semantic_sha256(
            expected_aggregate
        ),
        "development_cohort_status": "development_only",
        "development_task_count": len(task_pseudonyms),
        "development_task_projection_sha256": _semantic_sha256(
            [
                {
                    "task_pseudonym": pseudonym,
                    "cohort": assignments[pseudonym],
                }
                for pseudonym in task_pseudonyms
            ]
        ),
    }


def _audit_candidate_seed_without_raw(
    *,
    root: Path,
    artifact: VerifiedSourceArtifact,
    experiment: Mapping[str, object],
    candidate: Mapping[str, object],
    split_seed: int,
    seed_result: Mapping[str, object],
) -> AuditedDiagnostic:
    """Verify frozen forecasts/bundles without reconstructing source rows."""

    source_document = _mapping(
        artifact.loaded.document.get("source"), name="Stage 4 source"
    )
    dataset_document = _mapping(
        artifact.loaded.document.get("dataset"), name="Stage 4 dataset"
    )
    runtime_versions = dict(
        _mapping(
            artifact.loaded.document.get("runtime_versions"),
            name="Stage 4 runtime versions",
        )
    )
    source_name = _required_string(
        source_document.get("source_name"), name="Stage 4 source name"
    )
    experiment_id = _required_string(
        experiment.get("experiment_id"), name="lifecycle experiment id"
    )
    condition_id = _required_string(
        experiment.get("condition_id"), name="lifecycle condition"
    )
    candidate_id = _required_string(
        candidate.get("candidate_id"), name="lifecycle candidate id"
    )
    candidate_hash = _required_sha256(
        candidate.get("candidate_hash"), name="lifecycle candidate hash"
    )
    split_plan_id = _required_sha256(
        seed_result.get("split_plan_id"), name="candidate split plan id"
    )
    if (
        seed_result.get("candidate_id") != candidate_id
        or seed_result.get("candidate_hash") != candidate_hash
    ):
        raise Stage4CompletionDiagnosticsError(
            "candidate seed result identity differs"
        )
    experiment_key = _safe_compact_key(
        experiment.get("artifact_key"),
        prefix="e",
        name="experiment artifact key",
    )
    candidate_key = _safe_compact_key(
        candidate.get("artifact_key"),
        prefix="c",
        name="candidate artifact key",
    )
    if (
        experiment_key != _artifact_key("e", experiment_id)
        or candidate_key != _artifact_key("c", candidate_hash)
    ):
        raise Stage4CompletionDiagnosticsError(
            "compact fold artifact keys do not close"
        )
    inventory: list[Mapping[str, object]] = []
    provenance_hashes: set[str] = set()
    for fold in range(5):
        relative = (
            PurePosixPath("fold_artifacts")
            / experiment_key
            / candidate_key
            / f"seed_{split_seed}"
            / f"fold_{fold}"
            / "bundle"
        )
        bundle_path = artifact.loaded.path.joinpath(*relative.parts)
        if not bundle_path.is_dir() or _is_link_or_reparse(bundle_path):
            raise Stage4CompletionDiagnosticsError(
                "lifecycle bundle path is missing or unsafe"
            )
        bundle_files = _bundle_files(bundle_path)
        manifest_path = bundle_path / "manifest.json"
        bundle_manifest_sha256 = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
        try:
            bundle = load_lifecycle_bundle(bundle_path)
        except Exception as exc:
            raise Stage4CompletionDiagnosticsError(
                "lifecycle bundle failed safe loading"
            ) from exc
        manifest = bundle.manifest
        if (
            manifest.get("candidate_id") != candidate_id
            or manifest.get("candidate_hash") != candidate_hash
            or manifest.get("outer_fold") != fold
            or manifest.get("split_plan_id") != split_plan_id
            or manifest.get("dataset_id")
            != dataset_document.get("development_dataset_id")
            or manifest.get("condition_id") != condition_id
            or manifest.get("position") != experiment.get("position")
            or manifest.get("target") != experiment.get("target")
            or manifest.get("code_hash") != EXPECTED_SOURCE_CODE_TREE_SHA256
            or manifest.get("source_descriptor_hash")
            != source_document.get("source_descriptor_hash")
            or manifest.get("capability_contract_hash")
            != source_document.get("capability_contract_hash")
            or manifest.get("runtime_versions") != runtime_versions
        ):
            raise Stage4CompletionDiagnosticsError(
                "loaded lifecycle bundle scope differs"
            )
        source_provenance = {
            "source_descriptor": dict(
                _mapping(
                    manifest.get("source_descriptor"),
                    name="bundle source descriptor",
                )
            ),
            "source_descriptor_hash": _required_sha256(
                manifest.get("source_descriptor_hash"),
                name="bundle source descriptor hash",
            ),
            "code_hash": _required_sha256(
                manifest.get("code_hash"), name="bundle code hash"
            ),
            "runtime_versions": runtime_versions,
        }
        provenance_hashes.add(_semantic_sha256(source_provenance))
        inventory.append(
            {
                "source_name": source_name,
                "condition_id": condition_id,
                "experiment_id": experiment_id,
                "candidate_id": candidate_id,
                "candidate_hash": candidate_hash,
                "split_seed": split_seed,
                "split_plan_id": split_plan_id,
                "fold": fold,
                "bundle_relative_path": relative.as_posix(),
                "bundle_manifest_sha256": bundle_manifest_sha256,
                "bundle_file_count": len(bundle_files),
                "load_status": "safe_loaded",
            }
        )
    if len(provenance_hashes) != 1:
        raise Stage4CompletionDiagnosticsError(
            "candidate lifecycle bundles disagree on source provenance"
        )
    parity = _checkpoint_parity(
        root=root,
        artifact=artifact,
        experiment=experiment,
        candidate=candidate,
        split_seed=split_seed,
        seed_result=seed_result,
        source_provenance_hash=next(iter(provenance_hashes)),
    )
    document = {
        "source_name": source_name,
        "condition_id": condition_id,
        "experiment_id": experiment_id,
        "candidate_id": candidate_id,
        "candidate_hash": candidate_hash,
        "split_seed": split_seed,
        "split_plan_id": split_plan_id,
        "bundle_folds": list(range(5)),
        "bundle_projection_sha256": _bundle_projection(inventory),
        "checkpoint_parity": parity,
        "lifecycle_metrics": {
            "status": "unavailable",
            "reason_code": LIFECYCLE_UNAVAILABLE_REASON,
            "labels_present": False,
            "lifecycle_sequences_present": False,
            "unavailable_metrics": list(UNAVAILABLE_LIFECYCLE_METRICS),
            "historical_stage3_reference": None,
        },
    }
    return AuditedDiagnostic(
        document=document,
        inventory=tuple(inventory),
    )


def _artifact_experiments(
    artifact: VerifiedSourceArtifact,
) -> list[Mapping[str, object]]:
    return [
        _mapping(item, name=f"artifact experiments[{index}]")
        for index, item in enumerate(
            _list(
                artifact.loaded.document.get("experiments"),
                name="artifact experiments",
            )
        )
    ]


def _source_artifact_document(
    artifact: VerifiedSourceArtifact,
    *,
    lifecycle_status: str,
) -> dict[str, object]:
    results = artifact.loaded.document
    source = _mapping(results.get("source"), name="Stage 4 source")
    matrix = _mapping(results.get("matrix"), name="Stage 4 matrix")
    protocol = _mapping(
        results.get("development_protocol"), name="Stage 4 development protocol"
    )
    return {
        "source_name": _required_string(
            source.get("source_name"), name="Stage 4 source name"
        ),
        "source_id": _required_string(
            source.get("source_id"), name="Stage 4 source id"
        ),
        "run_id": _required_string(results.get("run_id"), name="Stage 4 run id"),
        "artifact_id": artifact.loaded.artifact_id,
        "results_payload_sha256": artifact.loaded.results_payload_sha256,
        "matrix_id": _required_sha256(
            matrix.get("matrix_id"), name="Stage 4 matrix id"
        ),
        "development_protocol_id": _required_sha256(
            protocol.get("protocol_id"), name="development protocol id"
        ),
        "lifecycle_status": lifecycle_status,
    }


def _source_diagnostics(
    *,
    root: Path,
    artifact: VerifiedSourceArtifact,
) -> tuple[
    dict[str, object],
    list[Mapping[str, object]],
    list[Mapping[str, object]],
]:
    source = _mapping(
        artifact.loaded.document.get("source"), name="Stage 4 source"
    )
    source_name = _required_string(
        source.get("source_name"), name="Stage 4 source name"
    )
    artifact_experiments = _artifact_experiments(artifact)
    lifecycle_experiments = []
    for experiment in artifact_experiments:
        candidates = _candidate_documents(experiment)
        if EXPECTED_LIFECYCLE_CANDIDATES <= set(candidates):
            if set(candidates) != EXPECTED_LIFECYCLE_CANDIDATES:
                raise Stage4CompletionDiagnosticsError(
                    "seed-policy experiment has extra candidates"
                )
            lifecycle_experiments.append((experiment, candidates))
    if source_name in EXPECTED_POINT_ONLY_SOURCES:
        if lifecycle_experiments:
            raise Stage4CompletionDiagnosticsError(
                "aggregate source unexpectedly exposes lifecycle candidates"
            )
        return (
            _source_artifact_document(
                artifact,
                lifecycle_status="not_applicable_no_lifecycle",
            ),
            [],
            [],
        )
    if source_name not in EXPECTED_LIFECYCLE_SOURCES or not lifecycle_experiments:
        raise Stage4CompletionDiagnosticsError(
            "lifecycle source lacks seed-policy experiments"
        )
    diagnostics: list[Mapping[str, object]] = []
    inventory: list[Mapping[str, object]] = []
    for experiment, candidates in lifecycle_experiments:
        for candidate_id in sorted(EXPECTED_LIFECYCLE_CANDIDATES):
            candidate = candidates[candidate_id]
            for split_seed, seed_result in _seed_results(candidate).items():
                audited = _audit_candidate_seed_without_raw(
                    root=root,
                    artifact=artifact,
                    experiment=experiment,
                    candidate=candidate,
                    split_seed=split_seed,
                    seed_result=seed_result,
                )
                diagnostics.append(audited.document)
                inventory.extend(audited.inventory)
    return (
        _source_artifact_document(
            artifact,
            lifecycle_status="unavailable_no_presealed_replay_projection",
        ),
        diagnostics,
        inventory,
    )


def _finite_json(value: object, *, path: str = "results") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise Stage4CompletionDiagnosticsError(
                    f"{path} contains a non-string key"
                )
            _finite_json(item, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_json(item, path=f"{path}[{index}]")
        return
    if isinstance(value, float):
        if not (-float("inf") < value < float("inf")):
            raise Stage4CompletionDiagnosticsError(
                f"{path} contains a non-finite number"
            )
        return
    if value is None or isinstance(value, (str, int, bool)):
        return
    raise Stage4CompletionDiagnosticsError(
        f"{path} contains unsupported JSON data"
    )


def _diagnostic_identity(
    value: Mapping[str, object],
) -> tuple[str, str, str, str, str, str, int]:
    seed = value.get("split_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise Stage4CompletionDiagnosticsError(
            "diagnostic split seed is invalid"
        )
    return (
        _required_string(value.get("source_name"), name="diagnostic source"),
        _required_string(value.get("condition_id"), name="diagnostic condition"),
        _required_string(
            value.get("experiment_id"),
            name="diagnostic experiment",
        ),
        _required_string(value.get("candidate_id"), name="diagnostic candidate"),
        _required_sha256(
            value.get("candidate_hash"),
            name="diagnostic candidate hash",
        ),
        _required_sha256(
            value.get("split_plan_id"),
            name="diagnostic split plan id",
        ),
        seed,
    )


def _inventory_identity(
    value: Mapping[str, object],
) -> tuple[str, str, str, str, str, str, int, int]:
    base = _diagnostic_identity(value)
    fold = value.get("fold")
    if isinstance(fold, bool) or not isinstance(fold, int):
        raise Stage4CompletionDiagnosticsError("inventory fold is invalid")
    return (*base, fold)


def verify_diagnostics_results_document(
    value: Mapping[str, object],
) -> str:
    """Verify the complete aggregate diagnostics schema and semantic digest."""

    if set(value) != RESULTS_KEYS:
        raise Stage4CompletionDiagnosticsError(
            "completion diagnostics results keys differ"
        )
    if (
        value.get("results_schema_version") != RESULTS_SCHEMA_VERSION
        or value.get("stage_name") != STAGE_NAME
        or value.get("policy_id") != POLICY_ID
    ):
        raise Stage4CompletionDiagnosticsError(
            "completion diagnostics policy identity differs"
        )
    source_binding = _mapping(
        value.get("source_binding"), name="diagnostics source binding"
    )
    if source_binding != {
        "git_commit": EXPECTED_SOURCE_COMMIT,
        "code_tree_sha256": EXPECTED_SOURCE_CODE_TREE_SHA256,
    }:
        raise Stage4CompletionDiagnosticsError(
            "diagnostics source binding differs"
        )
    diagnostics_binding = _mapping(
        value.get("diagnostics_code_binding"),
        name="diagnostics code binding",
    )
    if set(diagnostics_binding) != {
        "git_commit",
        "code_tree_sha256",
        "code_paths",
    }:
        raise Stage4CompletionDiagnosticsError(
            "diagnostics execution code binding keys differ"
        )
    _required_sha256(
        diagnostics_binding.get("code_tree_sha256"),
        name="diagnostics execution code tree SHA-256",
    )
    diagnostics_commit = _required_string(
        diagnostics_binding.get("git_commit"),
        name="diagnostics execution Git commit",
    )
    if (
        len(diagnostics_commit) != 40
        or any(
            character not in "0123456789abcdef"
            for character in diagnostics_commit
        )
    ):
        raise Stage4CompletionDiagnosticsError(
            "diagnostics execution Git commit is invalid"
        )
    diagnostics_paths = _list(
        diagnostics_binding.get("code_paths"),
        name="diagnostics execution code paths",
    )
    if (
        diagnostics_paths != sorted(set(diagnostics_paths))
        or not DIAGNOSTICS_DIRECT_CODE_PATHS <= set(diagnostics_paths)
    ):
        raise Stage4CompletionDiagnosticsError(
            "diagnostics execution code paths are not canonical"
        )
    if value.get("final_holdout") != EXPECTED_FINAL_HOLDOUT:
        raise Stage4CompletionDiagnosticsError(
            "completion diagnostics opened the final holdout"
        )
    source_artifacts = [
        _mapping(item, name=f"source artifacts[{index}]")
        for index, item in enumerate(
            _list(value.get("source_artifacts"), name="source artifacts")
        )
    ]
    if (
        len(source_artifacts) != EXPECTED_BOUND_SOURCE_ARTIFACT_COUNT
        or any(set(item) != SOURCE_ARTIFACT_KEYS for item in source_artifacts)
        or [item["source_name"] for item in source_artifacts]
        != sorted(EXPECTED_SOURCE_NAMES)
        or {
            item["source_name"]
            for item in source_artifacts
            if item["lifecycle_status"]
            == "unavailable_no_presealed_replay_projection"
        }
        != EXPECTED_LIFECYCLE_SOURCES
        or {
            item["source_name"]
            for item in source_artifacts
            if item["lifecycle_status"] == "not_applicable_no_lifecycle"
        }
        != EXPECTED_POINT_ONLY_SOURCES
    ):
        raise Stage4CompletionDiagnosticsError(
            "diagnostics source artifact coverage differs"
        )
    if len({item["artifact_id"] for item in source_artifacts}) != len(
        source_artifacts
    ):
        raise Stage4CompletionDiagnosticsError(
            "diagnostics source artifact ids repeat"
        )
    for item in source_artifacts:
        for key in (
            "artifact_id",
            "results_payload_sha256",
            "matrix_id",
            "development_protocol_id",
        ):
            _required_sha256(item[key], name=f"source artifact {key}")

    inventory = [
        _mapping(item, name=f"bundle inventory[{index}]")
        for index, item in enumerate(
            _list(value.get("bundle_inventory"), name="bundle inventory")
        )
    ]
    if (
        len(inventory) != EXPECTED_LIFECYCLE_BUNDLE_COUNT
        or any(set(item) != BUNDLE_INVENTORY_KEYS for item in inventory)
    ):
        raise Stage4CompletionDiagnosticsError(
            "diagnostics bundle inventory coverage differs"
        )
    inventory_identities = [_inventory_identity(item) for item in inventory]
    if inventory_identities != sorted(inventory_identities) or len(
        set(inventory_identities)
    ) != len(inventory_identities):
        raise Stage4CompletionDiagnosticsError(
            "diagnostics bundle inventory is not canonical and unique"
        )
    for item in inventory:
        relative = _safe_relative(
            item["bundle_relative_path"], label="bundle inventory path"
        )
        expected_relative = (
            PurePosixPath("fold_artifacts")
            / _artifact_key("e", item["experiment_id"])
            / _artifact_key("c", item["candidate_hash"])
            / f"seed_{item['split_seed']}"
            / f"fold_{item['fold']}"
            / "bundle"
        ).as_posix()
        if (
            relative != expected_relative
            or item["load_status"] != "safe_loaded"
            or item["fold"] not in range(5)
            or isinstance(item["bundle_file_count"], bool)
            or not isinstance(item["bundle_file_count"], int)
            or item["bundle_file_count"] <= 0
        ):
            raise Stage4CompletionDiagnosticsError(
                "diagnostics bundle inventory entry is invalid"
            )
        _required_sha256(
            item["bundle_manifest_sha256"], name="bundle manifest SHA-256"
        )

    diagnostics = [
        _mapping(item, name=f"diagnostics[{index}]")
        for index, item in enumerate(
            _list(value.get("diagnostics"), name="diagnostics")
        )
    ]
    if (
        len(diagnostics) != EXPECTED_LIFECYCLE_CANDIDATE_SEED_COUNT
        or any(set(item) != DIAGNOSTIC_KEYS for item in diagnostics)
    ):
        raise Stage4CompletionDiagnosticsError(
            "diagnostic candidate-seed coverage differs"
        )
    diagnostic_identities = [_diagnostic_identity(item) for item in diagnostics]
    if diagnostic_identities != sorted(diagnostic_identities) or len(
        set(diagnostic_identities)
    ) != len(diagnostic_identities):
        raise Stage4CompletionDiagnosticsError(
            "diagnostics are not canonical and unique"
        )
    if (
        {item["source_name"] for item in diagnostics}
        != EXPECTED_LIFECYCLE_SOURCES
        or len({(item["source_name"], item["condition_id"]) for item in diagnostics})
        != EXPECTED_LIFECYCLE_CONDITION_COUNT
        or {item["candidate_id"] for item in diagnostics}
        != EXPECTED_LIFECYCLE_CANDIDATES
        or {item["split_seed"] for item in diagnostics} != set(STAGE_SPLIT_SEEDS)
    ):
        raise Stage4CompletionDiagnosticsError(
            "diagnostic source/condition/candidate/seed coverage differs"
        )
    condition_cells = {
        (str(item["source_name"]), str(item["condition_id"]))
        for item in diagnostics
    }
    for source_name, condition_id in condition_cells:
        cell = [
            item
            for item in diagnostics
            if item["source_name"] == source_name
            and item["condition_id"] == condition_id
        ]
        if (
            len(cell)
            != len(EXPECTED_LIFECYCLE_CANDIDATES) * len(STAGE_SPLIT_SEEDS)
            or {item["candidate_id"] for item in cell}
            != EXPECTED_LIFECYCLE_CANDIDATES
            or {item["split_seed"] for item in cell} != set(STAGE_SPLIT_SEEDS)
            or len({item["experiment_id"] for item in cell}) != 1
        ):
            raise Stage4CompletionDiagnosticsError(
                "diagnostic condition-cell topology differs"
            )
        for candidate_id in EXPECTED_LIFECYCLE_CANDIDATES:
            candidate_rows = [
                item for item in cell if item["candidate_id"] == candidate_id
            ]
            if (
                len(candidate_rows) != len(STAGE_SPLIT_SEEDS)
                or {item["split_seed"] for item in candidate_rows}
                != set(STAGE_SPLIT_SEEDS)
                or len({item["candidate_hash"] for item in candidate_rows}) != 1
            ):
                raise Stage4CompletionDiagnosticsError(
                    "diagnostic candidate topology differs"
                )
        for split_seed in STAGE_SPLIT_SEEDS:
            seed_rows = [
                item for item in cell if item["split_seed"] == split_seed
            ]
            if (
                len(seed_rows) != len(EXPECTED_LIFECYCLE_CANDIDATES)
                or len({item["split_plan_id"] for item in seed_rows}) != 1
            ):
                raise Stage4CompletionDiagnosticsError(
                    "diagnostic split-plan topology differs"
                )
    for item in diagnostics:
        if item["bundle_folds"] != list(range(5)):
            raise Stage4CompletionDiagnosticsError(
                "diagnostic fold coverage differs"
            )
        matching_inventory = [
            record
            for record in inventory
            if _diagnostic_identity(record) == _diagnostic_identity(item)
        ]
        if len(matching_inventory) != 5 or item[
            "bundle_projection_sha256"
        ] != _bundle_projection(matching_inventory):
            raise Stage4CompletionDiagnosticsError(
                "diagnostic bundle projection differs"
            )
        checkpoint_parity = _mapping(
            item.get("checkpoint_parity"), name="diagnostic checkpoint parity"
        )
        if (
            set(checkpoint_parity) != CHECKPOINT_PARITY_KEYS
            or checkpoint_parity.get("status") != "exact"
            or checkpoint_parity.get("prediction_count")
            != checkpoint_parity.get("expected_prediction_count")
            or not isinstance(checkpoint_parity.get("prediction_count"), int)
            or isinstance(checkpoint_parity.get("prediction_count"), bool)
            or checkpoint_parity["prediction_count"] <= 0
            or checkpoint_parity.get("prediction_projection_sha256")
            != checkpoint_parity.get("expected_prediction_projection_sha256")
            or checkpoint_parity.get("cohort_projection_sha256")
            != checkpoint_parity.get("expected_cohort_projection_sha256")
            or checkpoint_parity.get("aggregate_metrics_projection_sha256")
            != checkpoint_parity.get(
                "expected_aggregate_metrics_projection_sha256"
            )
            or checkpoint_parity.get("development_cohort_status")
            != "development_only"
            or isinstance(checkpoint_parity.get("development_task_count"), bool)
            or not isinstance(
                checkpoint_parity.get("development_task_count"), int
            )
            or checkpoint_parity["development_task_count"] <= 0
        ):
            raise Stage4CompletionDiagnosticsError(
                "diagnostic checkpoint parity differs"
            )
        for key in (
            "checkpoint_artifact_id",
            "checkpoint_result_sha256",
            "prediction_projection_sha256",
            "cohort_projection_sha256",
            "aggregate_metrics_projection_sha256",
            "development_task_projection_sha256",
        ):
            _required_sha256(
                checkpoint_parity[key],
                name=f"diagnostic checkpoint {key}",
            )
        lifecycle_metrics = _mapping(
            item.get("lifecycle_metrics"),
            name="diagnostic lifecycle metrics",
        )
        if (
            set(lifecycle_metrics) != LIFECYCLE_METRICS_KEYS
            or lifecycle_metrics.get("status") != "unavailable"
            or lifecycle_metrics.get("reason_code")
            != LIFECYCLE_UNAVAILABLE_REASON
            or lifecycle_metrics.get("labels_present") is not False
            or lifecycle_metrics.get("lifecycle_sequences_present") is not False
            or lifecycle_metrics.get("unavailable_metrics")
            != UNAVAILABLE_LIFECYCLE_METRICS
            or lifecycle_metrics.get("historical_stage3_reference") is not None
        ):
            raise Stage4CompletionDiagnosticsError(
                "diagnostic unavailable lifecycle declaration differs"
            )

    for source_name, condition_id in condition_cells:
        cell = [
            item
            for item in diagnostics
            if item["source_name"] == source_name
            and item["condition_id"] == condition_id
        ]
        task_counts = {
            _mapping(
                item["checkpoint_parity"],
                name="diagnostic checkpoint parity",
            )["development_task_count"]
            for item in cell
        }
        task_projections = {
            _mapping(
                item["checkpoint_parity"],
                name="diagnostic checkpoint parity",
            )["development_task_projection_sha256"]
            for item in cell
        }
        if len(task_counts) != 1 or len(task_projections) != 1:
            raise Stage4CompletionDiagnosticsError(
                "diagnostic split-independent development task identity "
                "projection differs within a condition cell"
            )

    coverage = _mapping(value.get("coverage"), name="diagnostics coverage")
    expected_coverage = {
        "bound_source_artifact_count": EXPECTED_BOUND_SOURCE_ARTIFACT_COUNT,
        "lifecycle_source_count": EXPECTED_LIFECYCLE_SOURCE_COUNT,
        "lifecycle_condition_count": EXPECTED_LIFECYCLE_CONDITION_COUNT,
        "lifecycle_candidate_count": EXPECTED_LIFECYCLE_CANDIDATE_COUNT,
        "lifecycle_candidate_cell_count": (
            EXPECTED_LIFECYCLE_CANDIDATE_CELL_COUNT
        ),
        "lifecycle_candidate_seed_count": (
            EXPECTED_LIFECYCLE_CANDIDATE_SEED_COUNT
        ),
        "lifecycle_bundle_count": EXPECTED_LIFECYCLE_BUNDLE_COUNT,
        "checkpoint_verified_candidate_seed_count": (
            EXPECTED_LIFECYCLE_CANDIDATE_SEED_COUNT
        ),
        "lifecycle_replayed_candidate_seed_count": 0,
        "lifecycle_metrics_unavailable_candidate_seed_count": (
            EXPECTED_LIFECYCLE_CANDIDATE_SEED_COUNT
        ),
    }
    if set(coverage) != COVERAGE_KEYS or coverage != expected_coverage:
        raise Stage4CompletionDiagnosticsError(
            "diagnostics coverage counters do not close"
        )
    _finite_json(value)
    try:
        _assert_aggregate_safe(value)
    except Exception as exc:
        raise Stage4CompletionDiagnosticsError(
            "diagnostics results are not aggregate-safe"
        ) from exc
    declared = _required_sha256(
        value.get("results_payload_sha256"),
        name="completion diagnostics results payload SHA-256",
    )
    semantic = dict(value)
    semantic.pop("results_payload_sha256")
    if _semantic_sha256(semantic) != declared:
        raise Stage4CompletionDiagnosticsError(
            "completion diagnostics results payload SHA-256 does not close"
        )
    return declared


def _build_results(
    *,
    source_binding: Mapping[str, str],
    diagnostics_code_binding: Mapping[str, object],
    source_artifacts: Sequence[Mapping[str, object]],
    diagnostics: Sequence[Mapping[str, object]],
    inventory: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    results: dict[str, object] = {
        "results_schema_version": RESULTS_SCHEMA_VERSION,
        "stage_name": STAGE_NAME,
        "policy_id": POLICY_ID,
        "source_binding": dict(source_binding),
        "diagnostics_code_binding": dict(diagnostics_code_binding),
        "source_artifacts": list(source_artifacts),
        "coverage": {
            "bound_source_artifact_count": len(source_artifacts),
            "lifecycle_source_count": sum(
                item["lifecycle_status"]
                == "unavailable_no_presealed_replay_projection"
                for item in source_artifacts
            ),
            "lifecycle_condition_count": len(
                {
                    (item["source_name"], item["condition_id"])
                    for item in diagnostics
                }
            ),
            "lifecycle_candidate_count": len(
                {item["candidate_id"] for item in diagnostics}
            ),
            "lifecycle_candidate_cell_count": len(
                {
                    (
                        item["source_name"],
                        item["condition_id"],
                        item["candidate_id"],
                    )
                    for item in diagnostics
                }
            ),
            "lifecycle_candidate_seed_count": len(diagnostics),
            "lifecycle_bundle_count": len(inventory),
            "checkpoint_verified_candidate_seed_count": len(diagnostics),
            "lifecycle_replayed_candidate_seed_count": 0,
            "lifecycle_metrics_unavailable_candidate_seed_count": len(
                diagnostics
            ),
        },
        "bundle_inventory": list(inventory),
        "diagnostics": list(diagnostics),
        "final_holdout": dict(EXPECTED_FINAL_HOLDOUT),
    }
    results["results_payload_sha256"] = _semantic_sha256(results)
    verify_diagnostics_results_document(results)
    return results


def _run_semantic(
    *,
    source_binding: Mapping[str, str],
    diagnostics_code_binding: Mapping[str, object],
    artifacts: Sequence[VerifiedSourceArtifact],
    runner_sha256: str,
) -> dict[str, object]:
    return {
        "results_schema_version": RESULTS_SCHEMA_VERSION,
        "policy_id": POLICY_ID,
        "source_binding": dict(source_binding),
        "diagnostics_code_binding": dict(diagnostics_code_binding),
        "source_artifacts": [
            {
                "source_name": _mapping(
                    artifact.loaded.document.get("source"), name="Stage 4 source"
                )["source_name"],
                "artifact_id": artifact.loaded.artifact_id,
                "results_payload_sha256": (
                    artifact.loaded.results_payload_sha256
                ),
            }
            for artifact in artifacts
        ],
        "diagnostics_runner_sha256": runner_sha256,
        "final_holdout": dict(EXPECTED_FINAL_HOLDOUT),
    }


def _load_existing(
    output: Path,
    *,
    run_id: str,
    run_semantic: Mapping[str, object],
    expected_results: Mapping[str, object],
) -> CompletionDiagnosticsSummary:
    try:
        manifest = verify_artifact(output)
    except Exception as exc:
        raise Stage4CompletionDiagnosticsError(
            "existing diagnostics artifact failed verification"
        ) from exc
    if manifest.stage_name != STAGE_NAME or manifest.schema_version != (
        ARTIFACT_SCHEMA_VERSION
    ):
        raise Stage4CompletionDiagnosticsError(
            "existing diagnostics artifact identity differs"
        )
    metadata = manifest.metadata
    if (
        metadata.get("run_id") != run_id
        or metadata.get("run_semantic") != run_semantic
    ):
        raise Stage4CompletionDiagnosticsError(
            "existing diagnostics artifact run semantic differs"
        )
    results_path = output / "results.json"
    try:
        results = _strict_json_object(
            results_path.read_text(encoding="utf-8"),
            name="existing diagnostics results",
        )
    except (OSError, UnicodeError, Stage4CompletionDiagnosticsError) as exc:
        raise Stage4CompletionDiagnosticsError(
            "existing diagnostics results are invalid"
        ) from exc
    document = _mapping(results, name="existing diagnostics results")
    payload_hash = verify_diagnostics_results_document(document)
    if (
        document != expected_results
        or _semantic_sha256(document) != _semantic_sha256(expected_results)
    ):
        raise Stage4CompletionDiagnosticsError(
            "existing diagnostics results differ from the complete recomputation"
        )
    expected_metadata = {
        "run_id": run_id,
        "run_semantic": dict(run_semantic),
        "results_payload_sha256": payload_hash,
        "source_git_commit": EXPECTED_SOURCE_COMMIT,
        "source_code_tree_sha256": EXPECTED_SOURCE_CODE_TREE_SHA256,
        "diagnostics_code_binding": expected_results[
            "diagnostics_code_binding"
        ],
        "source_artifact_ids": [
            item["artifact_id"]
            for item in _list(
                expected_results["source_artifacts"],
                name="recomputed source artifacts",
            )
        ],
        "coverage": expected_results["coverage"],
        "diagnostics_runner_sha256": run_semantic[
            "diagnostics_runner_sha256"
        ],
    }
    if metadata != expected_metadata:
        raise Stage4CompletionDiagnosticsError(
            "existing diagnostics artifact metadata differs from recomputation"
        )
    coverage = _mapping(document["coverage"], name="existing diagnostics coverage")
    return CompletionDiagnosticsSummary(
        run_id=run_id,
        output_dir=output,
        artifact_id=manifest.artifact_id,
        results_payload_sha256=payload_hash,
        diagnostic_count=int(coverage["lifecycle_candidate_seed_count"]),
        bundle_count=int(coverage["lifecycle_bundle_count"]),
    )


def run_completion_diagnostics(
    *,
    repository_root: str | Path,
    artifact_inputs: Sequence[str | os.PathLike[str]],
    baseline_lock: str = DEFAULT_BASELINE_LOCK,
    output_root: str = DEFAULT_OUTPUT_ROOT,
) -> CompletionDiagnosticsSummary:
    """Audit frozen checkpoints/bundles and publish one final-safe supplement."""

    supplied_root = Path(repository_root)
    supplied_root = _assert_plain_lexical_ancestry(
        supplied_root,
        label="repository root",
    )
    root = supplied_root.resolve(strict=True)
    if not root.is_dir():
        raise Stage4CompletionDiagnosticsError(
            "repository root is not a directory"
        )
    _verify_runner_origin(root)
    output_parent = _safe_output_parent(root, output_root)
    references = _direct_artifact_references(root, artifact_inputs)
    artifacts = _verify_source_artifacts(references)
    source_binding, _source_paths = _source_binding(root, artifacts)
    diagnostics_code_binding = capture_diagnostics_code_binding(root)
    runner_sha256 = _runner_sha256()
    run_semantic = _run_semantic(
        source_binding=source_binding,
        diagnostics_code_binding=diagnostics_code_binding,
        artifacts=artifacts,
        runner_sha256=runner_sha256,
    )
    run_id = _semantic_sha256(run_semantic)[:24]
    output = output_parent / _output_key(run_id)
    # Kept as a CLI compatibility parameter only.  Final-safe diagnostics do
    # not load the data-foundation lock or any source/inventory/raw payload.
    del baseline_lock
    source_documents: list[Mapping[str, object]] = []
    diagnostics: list[Mapping[str, object]] = []
    inventory: list[Mapping[str, object]] = []
    for artifact in artifacts:
        source_document, source_diagnostics, source_inventory = _source_diagnostics(
            root=root,
            artifact=artifact,
        )
        source_documents.append(source_document)
        diagnostics.extend(source_diagnostics)
        inventory.extend(source_inventory)
    source_documents.sort(key=lambda item: str(item["source_name"]))
    diagnostics.sort(key=_diagnostic_identity)
    inventory.sort(key=_inventory_identity)
    results = _build_results(
        source_binding=source_binding,
        diagnostics_code_binding=diagnostics_code_binding,
        source_artifacts=source_documents,
        diagnostics=diagnostics,
        inventory=inventory,
    )
    results_payload_sha256 = verify_diagnostics_results_document(results)
    if (
        _runner_sha256() != runner_sha256
        or capture_diagnostics_code_binding(root) != diagnostics_code_binding
    ):
        raise Stage4CompletionDiagnosticsError(
            "diagnostics runner changed during audit"
        )
    for artifact in artifacts:
        if verify_artifact(artifact.loaded.path).artifact_id != (
            artifact.loaded.artifact_id
        ):
            raise Stage4CompletionDiagnosticsError(
                "source artifact changed during diagnostics audit"
            )
    if output.exists():
        _assert_plain_lexical_ancestry(
            output,
            label="existing completion diagnostics artifact",
        )
        return _load_existing(
            output,
            run_id=run_id,
            run_semantic=run_semantic,
            expected_results=results,
        )
    output_parent.mkdir(parents=True, exist_ok=True)
    if _is_link_or_reparse(output_parent):
        raise Stage4CompletionDiagnosticsError(
            "completion diagnostics output parent is unsafe"
        )
    temporary = Path(
        tempfile.mkdtemp(prefix=".s4diag-", dir=output_parent)
    )
    try:
        (temporary / "results.json").write_bytes(
            json.dumps(
                results,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ).encode()
            + b"\n"
        )
        manifest = publish_artifact(
            temporary,
            stage_name=STAGE_NAME,
            schema_version=ARTIFACT_SCHEMA_VERSION,
            metadata={
                "run_id": run_id,
                "run_semantic": run_semantic,
                "results_payload_sha256": results_payload_sha256,
                "source_git_commit": EXPECTED_SOURCE_COMMIT,
                "source_code_tree_sha256": EXPECTED_SOURCE_CODE_TREE_SHA256,
                "diagnostics_code_binding": diagnostics_code_binding,
                "source_artifact_ids": [
                    item["artifact_id"] for item in source_documents
                ],
                "coverage": results["coverage"],
                "diagnostics_runner_sha256": runner_sha256,
            },
        )
        if (
            _runner_sha256() != runner_sha256
            or capture_diagnostics_code_binding(root)
            != diagnostics_code_binding
        ):
            raise Stage4CompletionDiagnosticsError(
                "diagnostics runner changed during publication"
            )
        for artifact in artifacts:
            if verify_artifact(artifact.loaded.path).artifact_id != (
                artifact.loaded.artifact_id
            ):
                raise Stage4CompletionDiagnosticsError(
                    "source artifact changed during publication"
                )
        if output.exists():
            raise FileExistsError(
                f"completion diagnostics destination appeared: {output}"
            )
        os.replace(temporary, output)
        if verify_artifact(output) != manifest:
            raise Stage4CompletionDiagnosticsError(
                "published diagnostics artifact failed verification"
            )
    finally:
        if temporary.exists():
            try:
                temporary.resolve().relative_to(output_parent.resolve())
            except ValueError as exc:
                raise Stage4CompletionDiagnosticsError(
                    "temporary diagnostics artifact escaped its root"
                ) from exc
            shutil.rmtree(temporary)
    return CompletionDiagnosticsSummary(
        run_id=run_id,
        output_dir=output,
        artifact_id=manifest.artifact_id,
        results_payload_sha256=results_payload_sha256,
        diagnostic_count=len(diagnostics),
        bundle_count=len(inventory),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Safely verify frozen Stage 4 development checkpoints and lifecycle "
            "bundles without reopening source or final payloads."
        )
    )
    parser.add_argument(
        "--repository-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        help=(
            "Stage 4 development artifact directory or its results.json; "
            "pass exactly four times"
        ),
    )
    parser.add_argument("--baseline-lock", default=DEFAULT_BASELINE_LOCK)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = run_completion_diagnostics(
            repository_root=args.repository_root,
            artifact_inputs=args.artifact,
            baseline_lock=args.baseline_lock,
            output_root=args.output_root,
        )
    except Exception as exc:
        print(f"Stage 4 completion diagnostics failed: {exc}")
        return 2
    print(
        json.dumps(
            {
                "run_id": summary.run_id,
                "output_dir": str(summary.output_dir),
                "artifact_id": summary.artifact_id,
                "results_payload_sha256": summary.results_payload_sha256,
                "diagnostic_count": summary.diagnostic_count,
                "bundle_count": summary.bundle_count,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
