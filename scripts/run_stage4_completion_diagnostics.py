"""Publish immutable development-only lifecycle diagnostics for Stage 4.

This runner never trains or mutates a source artifact.  It verifies four
immutable Stage 4 development artifacts, reconstructs their frozen development
protocols, safely loads every seed-policy lifecycle bundle, replays each outer
test fold, proves exact scored-prediction parity, and publishes aggregate-only
diagnostics as a separate immutable artifact.
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

from token_prediction.dataset import (
    PredictionPosition,
    PredictionTarget,
    build_lifecycle_slice,
)
from token_prediction.development import (
    STAGE_SPLIT_SEEDS,
    DevelopmentProtocol,
    build_development_protocol,
)
from token_prediction.evaluation import (
    METRIC_SUITE_ID,
    evaluate_progress_checkpoints,
    evaluate_same_task_run_variance,
    evaluate_termination_strata,
)
from token_prediction.experiment import CandidateResult, PredictionRecord
from token_prediction.lifecycle_bundle import load_lifecycle_bundle
from token_prediction.lineage import publish_artifact, verify_artifact
from token_prediction.stage4_matrix import Stage4Matrix, build_stage4_matrix

if __package__:
    from scripts.run_data_foundation_baseline import (
        DEFAULT_BASELINE_LOCK,
        _is_link_or_reparse,
        _repo_path,
        _safe_relative,
        load_lock_context,
    )
    from scripts.run_stage2_experiments import (
        SOURCE_NAMES,
        Stage2LoadedSource,
        _verify_source_inputs,
        load_stage2_source,
    )
    from scripts.run_stage4_experiments import (
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
        resolve_artifact_references,
    )
else:  # pragma: no cover - direct production CLI invocation
    from run_data_foundation_baseline import (
        DEFAULT_BASELINE_LOCK,
        _is_link_or_reparse,
        _repo_path,
        _safe_relative,
        load_lock_context,
    )
    from run_stage2_experiments import (
        SOURCE_NAMES,
        Stage2LoadedSource,
        _verify_source_inputs,
        load_stage2_source,
    )
    from run_stage4_experiments import (
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
        resolve_artifact_references,
    )


RESULTS_SCHEMA_VERSION = 1
ARTIFACT_SCHEMA_VERSION = 1
STAGE_NAME = "stage4_completion_diagnostics"
POLICY_ID = "stage4_completion_development_lifecycle_replay_v1"
DEFAULT_OUTPUT_ROOT = "workspace/stage4/completion_diagnostics"
ALLOWED_OUTPUT_ROOT = PurePosixPath(DEFAULT_OUTPUT_ROOT)
OUTPUT_KEY_HEX_LENGTH = 20
EXPECTED_SOURCE_COMMIT = "c1ac2484f44ed65705cdd00eba7b70a739a3ac0b"
EXPECTED_SOURCE_CODE_TREE_SHA256 = (
    "6418545afa08a39df1797486e4c845063c2de13b29f20c81500933fad2201757"
)
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
    "reload_parity",
    "progress",
    "termination",
    "run_variance",
}
RELOAD_PARITY_KEYS = {
    "status",
    "scored_prediction_count",
    "expected_prediction_count",
    "prediction_projection_sha256",
    "expected_prediction_projection_sha256",
}
COVERAGE_KEYS = {
    "bound_source_artifact_count",
    "lifecycle_source_count",
    "lifecycle_condition_count",
    "lifecycle_candidate_count",
    "lifecycle_candidate_cell_count",
    "lifecycle_candidate_seed_count",
    "lifecycle_bundle_count",
    "replayed_run_count",
    "scored_run_count",
    "scored_boundary_count",
    "unscored_context_boundary_count",
}
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
RUN_VARIANCE_ID = "same_task_run_mae_variance_v1"
RUN_DISPERSION_EXTENSION_ID = "same_task_run_mae_iqr_max_minus_min_v1"
PROGRESS_ID = "lifecycle_progress_checkpoints_v1"
TERMINATION_ID = "lifecycle_termination_strata_v1"


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
class ReplayedDiagnostic:
    document: Mapping[str, object]
    inventory: tuple[Mapping[str, object], ...]
    run_count: int
    scored_run_count: int
    scored_boundary_count: int
    unscored_context_boundary_count: int


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
        verified.append(VerifiedSourceArtifact(loaded, artifact_manifest))
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
        by_source[source_name] = artifact
    if set(by_source) != set(SOURCE_NAMES):
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
        {
            "fold": item["fold"],
            "bundle_manifest_sha256": item["bundle_manifest_sha256"],
            "bundle_file_count": item["bundle_file_count"],
        }
        for item in sorted(inventory, key=lambda value: int(value["fold"]))
    ]
    return _semantic_sha256(projection)


def _prediction_result(
    *,
    candidate_id: str,
    candidate_hash: str,
    seed_result: Mapping[str, object],
    experiment: Mapping[str, object],
    records: Sequence[PredictionRecord],
) -> CandidateResult:
    comparability = _list(
        seed_result.get("comparability_key"), name="candidate comparability key"
    )
    if len(comparability) != 9:
        raise Stage4CompletionDiagnosticsError(
            "candidate comparability key length differs"
        )
    try:
        position = PredictionPosition(
            _required_string(experiment.get("position"), name="experiment position")
        )
        target = PredictionTarget(
            _required_string(experiment.get("target"), name="experiment target")
        )
    except ValueError as exc:
        raise Stage4CompletionDiagnosticsError(
            "experiment position or target is invalid"
        ) from exc
    split_plan_id = _required_sha256(
        seed_result.get("split_plan_id"), name="candidate split plan id"
    )
    if (
        comparability[1] != split_plan_id
        or comparability[3] != position.value
        or comparability[4] != target.value
        or comparability[5] != experiment.get("condition_id")
        or comparability[6] != experiment.get("calibrator_id")
        or comparability[7] != str(experiment.get("alpha"))
        or comparability[8] != METRIC_SUITE_ID
    ):
        raise Stage4CompletionDiagnosticsError(
            "candidate comparability key differs from the experiment"
        )
    return CandidateResult(
        candidate_id=candidate_id,
        candidate_hash=candidate_hash,
        dataset_id=_required_string(comparability[0], name="candidate dataset id"),
        split_plan_id=split_plan_id,
        eligibility_hash=_required_sha256(
            comparability[2], name="candidate eligibility hash"
        ),
        position=position,
        target=target,
        condition_id=_required_string(
            experiment.get("condition_id"), name="experiment condition"
        ),
        calibrator_id=_required_string(
            experiment.get("calibrator_id"), name="experiment calibrator"
        ),
        alpha=float(experiment["alpha"]),
        metric_suite_id=METRIC_SUITE_ID,
        predictions=tuple(records),
        metrics={},
    )


def _replay_candidate_seed(
    *,
    root: Path,
    artifact: VerifiedSourceArtifact,
    loaded_source: Stage2LoadedSource,
    protocol: DevelopmentProtocol,
    experiment: Mapping[str, object],
    candidate: Mapping[str, object],
    split_seed: int,
    seed_result: Mapping[str, object],
) -> ReplayedDiagnostic:
    source_document = _mapping(
        artifact.loaded.document.get("source"), name="Stage 4 source"
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
    if seed_result.get("candidate_id") != candidate_id or seed_result.get(
        "candidate_hash"
    ) != candidate_hash:
        raise Stage4CompletionDiagnosticsError(
            "candidate seed result identity differs"
        )
    split_plan = next(
        (plan for plan in protocol.outer_plans if plan.seed == split_seed),
        None,
    )
    if split_plan is None:
        raise Stage4CompletionDiagnosticsError(
            "diagnostic split plan is missing"
        )
    split_plan_id = _required_sha256(
        seed_result.get("split_plan_id"), name="candidate split plan id"
    )
    if split_plan.split_plan_id != split_plan_id:
        raise Stage4CompletionDiagnosticsError(
            "reconstructed split plan differs from the artifact"
        )
    try:
        target = PredictionTarget(
            _required_string(experiment.get("target"), name="lifecycle target")
        )
    except ValueError as exc:
        raise Stage4CompletionDiagnosticsError("lifecycle target is invalid") from exc
    lifecycle_slice = build_lifecycle_slice(
        protocol.development_dataset,
        target=target,
        condition_id=condition_id,
    )
    if (
        {sequence.task_id for sequence in lifecycle_slice.sequences}
        & protocol.final_holdout_tasks
    ):
        raise Stage4CompletionDiagnosticsError(
            "lifecycle slice includes final-holdout tasks"
        )
    experiment_key = _safe_compact_key(
        experiment.get("artifact_key"), prefix="e", name="experiment artifact key"
    )
    candidate_key = _safe_compact_key(
        candidate.get("artifact_key"), prefix="c", name="candidate artifact key"
    )
    if experiment_key != _artifact_key("e", experiment_id) or candidate_key != (
        _artifact_key("c", candidate_hash)
    ):
        raise Stage4CompletionDiagnosticsError(
            "compact fold artifact keys do not close"
        )
    source_provenance = {
        "source_descriptor": loaded_source.source_lock.descriptor.to_dict(),
        "source_descriptor_hash": (
            loaded_source.source_lock.descriptor.descriptor_hash
        ),
        "code_hash": EXPECTED_SOURCE_CODE_TREE_SHA256,
        "runtime_versions": dict(
            _mapping(
                artifact.loaded.document.get("runtime_versions"),
                name="Stage 4 runtime versions",
            )
        ),
    }
    inventory: list[Mapping[str, object]] = []
    records: list[PredictionRecord] = []
    runs = []
    seen_sequences: set[str] = set()
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
            bundle = load_lifecycle_bundle(
                bundle_path,
                expected_source_provenance=source_provenance,
            )
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
            != protocol.development_dataset.dataset_id
            or manifest.get("condition_id") != condition_id
            or manifest.get("target") != target.value
            or manifest.get("code_hash") != EXPECTED_SOURCE_CODE_TREE_SHA256
        ):
            raise Stage4CompletionDiagnosticsError(
                "loaded lifecycle bundle scope differs"
            )
        test_tasks = split_plan.partition(fold).test_tasks
        sequences = tuple(
            sequence
            for sequence in lifecycle_slice.sequences
            if sequence.task_id in test_tasks
        )
        if not sequences or {sequence.task_id for sequence in sequences} & (
            protocol.final_holdout_tasks
        ):
            raise Stage4CompletionDiagnosticsError(
                "diagnostic fold sequence cohort is invalid"
            )
        fold_runs = bundle.run_calibrated(sequences)
        if len(fold_runs) != len(sequences):
            raise Stage4CompletionDiagnosticsError(
                "lifecycle replay run count differs"
            )
        for run in fold_runs:
            if run.sequence.trajectory_id in seen_sequences:
                raise Stage4CompletionDiagnosticsError(
                    "lifecycle replay repeated a trajectory"
                )
            seen_sequences.add(run.sequence.trajectory_id)
            for prediction in run.scored_predictions:
                point = prediction.step.point
                records.append(
                    PredictionRecord(
                        candidate_id=candidate_id,
                        point_id=point.point_id,
                        task_id=point.task_id,
                        trajectory_id=point.trajectory_id,
                        condition_id=point.condition_id,
                        fold=fold,
                        target=target,
                        forecast=prediction.forecast,
                        sample_weight=prediction.step.sample_weight,
                    )
                )
        runs.extend(fold_runs)
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
    if len(seen_sequences) != len(lifecycle_slice.sequences):
        raise Stage4CompletionDiagnosticsError(
            "five diagnostic folds do not cover the lifecycle slice"
        )
    if len({record.point_id for record in records}) != len(records):
        raise Stage4CompletionDiagnosticsError(
            "lifecycle replay repeated a scored prediction"
        )
    result = _prediction_result(
        candidate_id=candidate_id,
        candidate_hash=candidate_hash,
        seed_result=seed_result,
        experiment=experiment,
        records=records,
    )
    observed_projection = prediction_projection_sha256(result)
    expected_projection = _required_sha256(
        seed_result.get("prediction_projection_sha256"),
        name="expected prediction projection SHA-256",
    )
    expected_cohort = _required_sha256(
        seed_result.get("cohort_projection_sha256"),
        name="expected cohort projection SHA-256",
    )
    expected_count = seed_result.get("prediction_count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
        or len(records) != expected_count
        or observed_projection != expected_projection
        or cohort_projection_sha256(result) != expected_cohort
    ):
        raise Stage4CompletionDiagnosticsError(
            "reloaded lifecycle scored trajectory differs from the source artifact"
        )
    run_variance = evaluate_same_task_run_variance(tuple(runs))
    if (
        run_variance.get("run_variance_id") != RUN_VARIANCE_ID
        or run_variance.get("run_dispersion_extension_id")
        != RUN_DISPERSION_EXTENSION_ID
    ):
        raise Stage4CompletionDiagnosticsError(
            "repeated-run dispersion evaluator identity differs"
        )
    alpha = float(experiment["alpha"])
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
        "reload_parity": {
            "status": "exact",
            "scored_prediction_count": len(records),
            "expected_prediction_count": expected_count,
            "prediction_projection_sha256": observed_projection,
            "expected_prediction_projection_sha256": expected_projection,
        },
        "progress": evaluate_progress_checkpoints(tuple(runs), alpha=alpha),
        "termination": evaluate_termination_strata(tuple(runs), alpha=alpha),
        "run_variance": run_variance,
    }
    return ReplayedDiagnostic(
        document=document,
        inventory=tuple(inventory),
        run_count=len(runs),
        scored_run_count=sum(bool(run.scored_predictions) for run in runs),
        scored_boundary_count=len(records),
        unscored_context_boundary_count=sum(
            len(run.predictions) - len(run.scored_predictions) for run in runs
        ),
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


def _validate_reconstructed_source(
    artifact: VerifiedSourceArtifact,
    loaded_source: Stage2LoadedSource,
    protocol: DevelopmentProtocol,
    matrix: Stage4Matrix,
) -> None:
    results = artifact.loaded.document
    source = _mapping(results.get("source"), name="Stage 4 source")
    dataset = _mapping(results.get("dataset"), name="Stage 4 dataset")
    matrix_document = _mapping(results.get("matrix"), name="Stage 4 matrix")
    development_document = _mapping(
        results.get("development_protocol"), name="Stage 4 development protocol"
    )
    if (
        source.get("source_id")
        != loaded_source.source_lock.descriptor.source_id
        or source.get("source_descriptor_hash")
        != loaded_source.source_lock.descriptor.descriptor_hash
        or dataset.get("base_dataset_id") != loaded_source.base_dataset_id
        or dataset.get("derived_dataset_id")
        != loaded_source.derived_dataset.dataset_id
        or dataset.get("development_dataset_id")
        != protocol.development_dataset.dataset_id
        or development_document.get("protocol_id") != protocol.protocol_id
        or matrix_document.get("matrix_id") != matrix.matrix_id
        or matrix_document.get("development_protocol_id") != protocol.protocol_id
    ):
        raise Stage4CompletionDiagnosticsError(
            "reconstructed development source differs from its artifact"
        )
    if protocol.development_dataset.task_ids & protocol.final_holdout_tasks:
        raise Stage4CompletionDiagnosticsError(
            "reconstructed development dataset intersects final holdout"
        )


def _source_diagnostics(
    *,
    root: Path,
    lock_context: Any,
    artifact: VerifiedSourceArtifact,
) -> tuple[
    dict[str, object],
    list[Mapping[str, object]],
    list[Mapping[str, object]],
    tuple[int, int, int, int],
]:
    source = _mapping(
        artifact.loaded.document.get("source"), name="Stage 4 source"
    )
    source_name = _required_string(
        source.get("source_name"), name="Stage 4 source name"
    )
    loaded_source = load_stage2_source(
        root,
        lock_context,
        source_name=source_name,
    )
    protocol = build_development_protocol(loaded_source.derived_dataset)
    matrix = build_stage4_matrix(
        protocol,
        source_id=loaded_source.source_lock.descriptor.source_id,
        capabilities=loaded_source.source_lock.descriptor.capabilities,
    )
    _validate_reconstructed_source(artifact, loaded_source, protocol, matrix)
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
        _verify_source_inputs(root, lock_context, loaded_source)
        return (
            _source_artifact_document(
                artifact,
                lifecycle_status="not_applicable_no_lifecycle",
            ),
            [],
            [],
            (0, 0, 0, 0),
        )
    if source_name not in EXPECTED_LIFECYCLE_SOURCES or not lifecycle_experiments:
        raise Stage4CompletionDiagnosticsError(
            "lifecycle source lacks seed-policy experiments"
        )
    matrix_specs = {spec.experiment_id: spec for spec in matrix.experiments}
    diagnostics: list[Mapping[str, object]] = []
    inventory: list[Mapping[str, object]] = []
    run_count = 0
    scored_run_count = 0
    scored_boundary_count = 0
    unscored_count = 0
    for experiment, candidates in lifecycle_experiments:
        experiment_id = _required_string(
            experiment.get("experiment_id"), name="lifecycle experiment id"
        )
        spec = matrix_specs.get(experiment_id)
        if spec is None:
            raise Stage4CompletionDiagnosticsError(
                "reconstructed matrix lacks a lifecycle experiment"
            )
        spec_candidates = {candidate.candidate_id: candidate for candidate in spec.candidates}
        if set(spec_candidates) != EXPECTED_LIFECYCLE_CANDIDATES:
            raise Stage4CompletionDiagnosticsError(
                "reconstructed lifecycle candidate set differs"
            )
        for candidate_id in sorted(EXPECTED_LIFECYCLE_CANDIDATES):
            candidate = candidates[candidate_id]
            if (
                candidate.get("candidate_hash")
                != spec_candidates[candidate_id].content_hash
            ):
                raise Stage4CompletionDiagnosticsError(
                    "reconstructed lifecycle candidate hash differs"
                )
            for split_seed, seed_result in _seed_results(candidate).items():
                replayed = _replay_candidate_seed(
                    root=root,
                    artifact=artifact,
                    loaded_source=loaded_source,
                    protocol=protocol,
                    experiment=experiment,
                    candidate=candidate,
                    split_seed=split_seed,
                    seed_result=seed_result,
                )
                diagnostics.append(replayed.document)
                inventory.extend(replayed.inventory)
                run_count += replayed.run_count
                scored_run_count += replayed.scored_run_count
                scored_boundary_count += replayed.scored_boundary_count
                unscored_count += replayed.unscored_context_boundary_count
    _verify_source_inputs(root, lock_context, loaded_source)
    return (
        _source_artifact_document(artifact, lifecycle_status="complete"),
        diagnostics,
        inventory,
        (run_count, scored_run_count, scored_boundary_count, unscored_count),
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
) -> tuple[str, str, str, int]:
    seed = value.get("split_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise Stage4CompletionDiagnosticsError(
            "diagnostic split seed is invalid"
        )
    return (
        _required_string(value.get("source_name"), name="diagnostic source"),
        _required_string(value.get("condition_id"), name="diagnostic condition"),
        _required_string(value.get("candidate_id"), name="diagnostic candidate"),
        seed,
    )


def _inventory_identity(
    value: Mapping[str, object],
) -> tuple[str, str, str, int, int]:
    base = _diagnostic_identity(value)
    fold = value.get("fold")
    if isinstance(fold, bool) or not isinstance(fold, int):
        raise Stage4CompletionDiagnosticsError("inventory fold is invalid")
    return (*base, fold)


def _count_termination(document: Mapping[str, object]) -> tuple[int, int]:
    strata = _mapping(document.get("strata"), name="termination strata")
    scored = 0
    unscored = 0
    for value in strata.values():
        stratum = _mapping(value, name="termination stratum")
        raw_scored = stratum.get("n_scored")
        raw_unscored = stratum.get("n_context_only")
        if (
            isinstance(raw_scored, bool)
            or not isinstance(raw_scored, int)
            or raw_scored < 0
            or isinstance(raw_unscored, bool)
            or not isinstance(raw_unscored, int)
            or raw_unscored < 0
        ):
            raise Stage4CompletionDiagnosticsError(
                "termination counts are invalid"
            )
        scored += raw_scored
        unscored += raw_unscored
    return scored, unscored


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
        != sorted(SOURCE_NAMES)
        or {
            item["source_name"]
            for item in source_artifacts
            if item["lifecycle_status"] == "complete"
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
        if (
            not relative.startswith("fold_artifacts/")
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
    derived_replayed_runs = 0
    derived_scored_runs = 0
    derived_scored_boundaries = 0
    derived_unscored_boundaries = 0
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
        reload_parity = _mapping(
            item.get("reload_parity"), name="diagnostic reload parity"
        )
        if (
            set(reload_parity) != RELOAD_PARITY_KEYS
            or reload_parity.get("status") != "exact"
            or reload_parity.get("scored_prediction_count")
            != reload_parity.get("expected_prediction_count")
            or not isinstance(reload_parity.get("scored_prediction_count"), int)
            or isinstance(reload_parity.get("scored_prediction_count"), bool)
            or reload_parity["scored_prediction_count"] <= 0
            or reload_parity.get("prediction_projection_sha256")
            != reload_parity.get("expected_prediction_projection_sha256")
        ):
            raise Stage4CompletionDiagnosticsError(
                "diagnostic reload parity differs"
            )
        _required_sha256(
            reload_parity["prediction_projection_sha256"],
            name="diagnostic prediction projection SHA-256",
        )
        progress = _mapping(item.get("progress"), name="diagnostic progress")
        termination = _mapping(
            item.get("termination"), name="diagnostic termination"
        )
        run_variance = _mapping(
            item.get("run_variance"), name="diagnostic run variance"
        )
        if (
            progress.get("stratification_id") != PROGRESS_ID
            or termination.get("stratification_id") != TERMINATION_ID
            or run_variance.get("run_variance_id") != RUN_VARIANCE_ID
            or run_variance.get("run_dispersion_extension_id")
            != RUN_DISPERSION_EXTENSION_ID
        ):
            raise Stage4CompletionDiagnosticsError(
                "diagnostic evaluator identity differs"
            )
        progress_strata = _mapping(
            progress.get("strata"), name="diagnostic progress strata"
        )
        if set(progress_strata) != {"p25", "p50", "p75"}:
            raise Stage4CompletionDiagnosticsError(
                "diagnostic progress strata differ"
            )
        first_progress = _mapping(
            progress_strata["p25"], name="diagnostic p25 progress"
        )
        run_count = first_progress.get("n_sequences")
        scored_runs = run_variance.get("n_scored_runs")
        if (
            isinstance(run_count, bool)
            or not isinstance(run_count, int)
            or run_count <= 0
            or isinstance(scored_runs, bool)
            or not isinstance(scored_runs, int)
            or scored_runs <= 0
        ):
            raise Stage4CompletionDiagnosticsError(
                "diagnostic replayed/scored run counts are invalid"
            )
        derived_replayed_runs += run_count
        derived_scored_runs += scored_runs
        scored_boundaries, unscored_boundaries = _count_termination(termination)
        if scored_boundaries != reload_parity["scored_prediction_count"]:
            raise Stage4CompletionDiagnosticsError(
                "termination and prediction counts differ"
            )
        derived_scored_boundaries += scored_boundaries
        derived_unscored_boundaries += unscored_boundaries

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
        "replayed_run_count": derived_replayed_runs,
        "scored_run_count": derived_scored_runs,
        "scored_boundary_count": derived_scored_boundaries,
        "unscored_context_boundary_count": derived_unscored_boundaries,
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
    replay_totals: tuple[int, int, int, int],
) -> dict[str, object]:
    run_count, scored_runs, scored_boundaries, unscored_boundaries = replay_totals
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
                item["lifecycle_status"] == "complete"
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
            "replayed_run_count": run_count,
            "scored_run_count": scored_runs,
            "scored_boundary_count": scored_boundaries,
            "unscored_context_boundary_count": unscored_boundaries,
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
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Stage4CompletionDiagnosticsError(
            "existing diagnostics results are invalid"
        ) from exc
    document = _mapping(results, name="existing diagnostics results")
    payload_hash = verify_diagnostics_results_document(document)
    if metadata.get("results_payload_sha256") != payload_hash:
        raise Stage4CompletionDiagnosticsError(
            "existing artifact/result payload hashes differ"
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
    """Replay all frozen lifecycle candidates and publish one supplement."""

    supplied_root = Path(repository_root)
    if _is_link_or_reparse(supplied_root):
        raise Stage4CompletionDiagnosticsError("repository root must not be linked")
    root = supplied_root.resolve()
    if not root.is_dir():
        raise Stage4CompletionDiagnosticsError(
            "repository root is not a directory"
        )
    _verify_runner_origin(root)
    output_parent = _safe_output_parent(root, output_root)
    try:
        references = resolve_artifact_references(
            artifact_inputs,
            repo_root=root,
            development_runs_root=root / "workspace" / "stage4" / "runs",
        )
    except Exception as exc:
        raise Stage4CompletionDiagnosticsError(
            "Stage 4 artifact inputs are unsafe"
        ) from exc
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
    if output.exists():
        return _load_existing(
            output,
            run_id=run_id,
            run_semantic=run_semantic,
        )
    lock_context = load_lock_context(root, baseline_lock)
    source_documents: list[Mapping[str, object]] = []
    diagnostics: list[Mapping[str, object]] = []
    inventory: list[Mapping[str, object]] = []
    totals = [0, 0, 0, 0]
    for artifact in artifacts:
        source_document, source_diagnostics, source_inventory, source_totals = (
            _source_diagnostics(
                root=root,
                lock_context=lock_context,
                artifact=artifact,
            )
        )
        source_documents.append(source_document)
        diagnostics.extend(source_diagnostics)
        inventory.extend(source_inventory)
        for index, count in enumerate(source_totals):
            totals[index] += count
    source_documents.sort(key=lambda item: str(item["source_name"]))
    diagnostics.sort(key=_diagnostic_identity)
    inventory.sort(key=_inventory_identity)
    results = _build_results(
        source_binding=source_binding,
        diagnostics_code_binding=diagnostics_code_binding,
        source_artifacts=source_documents,
        diagnostics=diagnostics,
        inventory=inventory,
        replay_totals=tuple(totals),  # type: ignore[arg-type]
    )
    results_payload_sha256 = verify_diagnostics_results_document(results)
    if (
        _runner_sha256() != runner_sha256
        or capture_diagnostics_code_binding(root) != diagnostics_code_binding
    ):
        raise Stage4CompletionDiagnosticsError(
            "diagnostics runner changed during replay"
        )
    for artifact in artifacts:
        if verify_artifact(artifact.loaded.path).artifact_id != (
            artifact.loaded.artifact_id
        ):
            raise Stage4CompletionDiagnosticsError(
                "source artifact changed during diagnostics replay"
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
            "Safely replay frozen Stage 4 development lifecycle bundles and "
            "publish immutable aggregate diagnostics."
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
            "Stage 4 development artifact directory; pass exactly four times "
            "(or pass one completion release lock)"
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
