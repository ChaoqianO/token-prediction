"""Verify the frozen Stage 2 release lock, report, and optional local artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from token_prediction.lineage import verify_artifact

if __package__:
    from scripts.audit_stage2_sokoban import verify_sokoban_audit_results
    from scripts.run_stage2_experiments import (
        DATA_FOUNDATION_BASELINE_RELATIVE,
        STAGE1_VERIFIER_RELATIVE,
        STAGE2_AUXILIARY_MANIFEST_RELATIVE,
        STAGE2_METADATA_EXTRACTOR_RELATIVE,
        STAGE2_RUNNER_RELATIVE,
        STAGE2_SOKOBAN_AUDITOR_RELATIVE,
        Stage2ExperimentError,
        _framed_code_hash,
        _git,
        _is_link_or_reparse,
        _repo_path,
        _required_sha256,
        _safe_relative,
        _sha256_file,
        verify_stage2_results_document,
    )
else:  # pragma: no cover - exercised by the production CLI invocation
    from audit_stage2_sokoban import verify_sokoban_audit_results
    from run_stage2_experiments import (
        DATA_FOUNDATION_BASELINE_RELATIVE,
        STAGE1_VERIFIER_RELATIVE,
        STAGE2_AUXILIARY_MANIFEST_RELATIVE,
        STAGE2_METADATA_EXTRACTOR_RELATIVE,
        STAGE2_RUNNER_RELATIVE,
        STAGE2_SOKOBAN_AUDITOR_RELATIVE,
        Stage2ExperimentError,
        _framed_code_hash,
        _git,
        _is_link_or_reparse,
        _repo_path,
        _required_sha256,
        _safe_relative,
        _sha256_file,
        verify_stage2_results_document,
    )


DEFAULT_RELEASE_LOCK = "configs/stage2_release.json"
RELEASE_SCHEMA_VERSION = 1
RELEASE_STAGE_NAME = "stage2_development"
RELEASE_POLICY_ID = "stage2_commit_bound_four_source_release_v1"
EXPERIMENT_NAMES = frozenset(
    {"spend_aggregate", "bagen_sokoban", "bagen_swebench", "spend_openhands"}
)
AUDIT_NAME = "bagen_sokoban_compatibility"
MAX_RELEASE_JSON_BYTES = 1024 * 1024
MAX_RESULTS_JSON_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class Stage2ReleaseVerification:
    lock_path: str
    report_path: str
    code_tree_sha256: str
    artifact_commit_status: str
    locked_artifact_count: int
    verified_artifact_count: int
    manifest_file_count: int
    candidate_seed_run_count: int
    reloadable_candidate_seed_run_count: int
    exact_reload_fold_count: int
    final_holdout_evaluated: bool


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Stage2ExperimentError("Stage 2 release JSON contains duplicate keys")
        result[key] = value
    return result


def _constant(value: str) -> Any:
    raise Stage2ExperimentError(f"Stage 2 release JSON contains {value}")


def _load_json(path: Path, *, maximum_bytes: int, description: str) -> Mapping[str, Any]:
    if _is_link_or_reparse(path) or not path.is_file():
        raise Stage2ExperimentError(f"{description} must be a regular non-link file")
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise Stage2ExperimentError(f"{description} has an invalid size")
    try:
        value = json.loads(
            path.read_bytes().decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage2ExperimentError(f"{description} is not strict UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise Stage2ExperimentError(f"{description} must contain a JSON object")
    return value


def _exact(value: object, keys: set[str], *, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise Stage2ExperimentError(f"{description} keys do not match")
    return value


def _integer(value: object, *, description: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Stage2ExperimentError(f"{description} must be an integer >= {minimum}")
    return value


def _text(value: object, *, description: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise Stage2ExperimentError(f"{description} must be non-empty normalized text")
    return value


def _validate_release_document(value: Mapping[str, Any]) -> None:
    _exact(
        value,
        {
            "release_schema_version",
            "stage_name",
            "policy_id",
            "code_binding",
            "protocol",
            "artifacts",
            "totals",
            "report",
        },
        description="Stage 2 release lock",
    )
    if (
        value["release_schema_version"] != RELEASE_SCHEMA_VERSION
        or value["stage_name"] != RELEASE_STAGE_NAME
        or value["policy_id"] != RELEASE_POLICY_ID
    ):
        raise Stage2ExperimentError("Stage 2 release lock identity is invalid")
    code = _exact(
        value["code_binding"],
        {"artifact_git_commit", "code_tree_sha256"},
        description="Stage 2 release code binding",
    )
    commit = _text(code["artifact_git_commit"], description="artifact Git commit")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise Stage2ExperimentError("artifact Git commit must be a full lowercase SHA")
    _required_sha256(code["code_tree_sha256"], name="Stage 2 code tree")

    protocol = _exact(
        value["protocol"],
        {
            "outer_folds",
            "inner_folds",
            "split_seeds",
            "calibrator_id",
            "alpha",
            "final_holdout_evaluated",
            "final_holdout_prediction_count",
        },
        description="Stage 2 release protocol",
    )
    if (
        protocol["outer_folds"] != 5
        or protocol["inner_folds"] != 5
        or protocol["split_seeds"] != [20260719, 20260720, 20260721]
        or protocol["calibrator_id"] != "task_max_conformal"
        or protocol["alpha"] != 0.1
        or protocol["final_holdout_evaluated"] is not False
        or protocol["final_holdout_prediction_count"] != 0
    ):
        raise Stage2ExperimentError("Stage 2 release protocol is not frozen")

    artifacts = value["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != EXPERIMENT_NAMES | {AUDIT_NAME}:
        raise Stage2ExperimentError("Stage 2 release artifact set is incomplete")
    experiment_keys = {
        "kind",
        "path",
        "source_id",
        "source_descriptor_hash",
        "run_id",
        "artifact_id",
        "results_payload_sha256",
        "base_dataset_id",
        "derived_dataset_id",
        "development_dataset_id",
        "development_protocol_id",
        "matrix_id",
        "experiment_count",
        "candidate_seed_run_count",
        "manifest_file_count",
    }
    audit_keys = {
        "kind",
        "path",
        "source_id",
        "source_descriptor_hash",
        "run_id",
        "artifact_id",
        "results_payload_sha256",
        "stage1_artifact_id",
        "stage1_bundle_count",
        "stage1_parity_record_count",
        "stage1_parity_mismatch_count",
        "manifest_file_count",
    }
    for name, raw in artifacts.items():
        entry = _exact(
            raw,
            audit_keys if name == AUDIT_NAME else experiment_keys,
            description=f"Stage 2 artifact {name}",
        )
        expected_kind = "audit" if name == AUDIT_NAME else "experiment"
        if entry["kind"] != expected_kind:
            raise Stage2ExperimentError(f"Stage 2 artifact {name} has another kind")
        relative = _safe_relative(entry["path"], label=f"Stage 2 artifact {name} path")
        if not relative.startswith("workspace/stage2/experiments/s2-"):
            raise Stage2ExperimentError(f"Stage 2 artifact {name} path is outside release root")
        _text(entry["source_id"], description=f"Stage 2 artifact {name} source id")
        _required_sha256(
            entry["source_descriptor_hash"],
            name=f"Stage 2 artifact {name} source descriptor",
        )
        run_id = _text(entry["run_id"], description=f"Stage 2 artifact {name} run id")
        if len(run_id) != 24 or any(character not in "0123456789abcdef" for character in run_id):
            raise Stage2ExperimentError(f"Stage 2 artifact {name} run id is invalid")
        for field in ("artifact_id", "results_payload_sha256"):
            _required_sha256(entry[field], name=f"Stage 2 artifact {name} {field}")
        _integer(
            entry["manifest_file_count"],
            description=f"Stage 2 artifact {name} manifest file count",
            minimum=1,
        )
        if name == AUDIT_NAME:
            _required_sha256(entry["stage1_artifact_id"], name="Stage 1 artifact id")
            if (
                entry["stage1_bundle_count"] != 20
                or entry["stage1_parity_record_count"] != 992
                or entry["stage1_parity_mismatch_count"] != 0
            ):
                raise Stage2ExperimentError("Stage 1 parity lock is invalid")
        else:
            for field in (
                "base_dataset_id",
                "derived_dataset_id",
                "development_dataset_id",
                "development_protocol_id",
                "matrix_id",
            ):
                identifier = _text(
                    entry[field],
                    description=f"Stage 2 artifact {name} {field}",
                )
                digest = identifier.removeprefix("spend-your-money:")
                _required_sha256(digest, name=f"Stage 2 artifact {name} {field}")
            _integer(
                entry["experiment_count"],
                description=f"Stage 2 artifact {name} experiment count",
                minimum=1,
            )
            _integer(
                entry["candidate_seed_run_count"],
                description=f"Stage 2 artifact {name} candidate run count",
                minimum=1,
            )

    totals = _exact(
        value["totals"],
        {
            "experiment_artifact_count",
            "audit_artifact_count",
            "experiment_count",
            "candidate_seed_run_count",
            "reloadable_candidate_seed_run_count",
            "exact_reload_fold_count",
            "manifest_file_count",
        },
        description="Stage 2 release totals",
    )
    if totals != {
        "experiment_artifact_count": 4,
        "audit_artifact_count": 1,
        "experiment_count": 15,
        "candidate_seed_run_count": 282,
        "reloadable_candidate_seed_run_count": 195,
        "exact_reload_fold_count": 975,
        "manifest_file_count": 12665,
    }:
        raise Stage2ExperimentError("Stage 2 release totals are invalid")

    report = _exact(
        value["report"],
        {"path", "sha256"},
        description="Stage 2 release report",
    )
    if _safe_relative(report["path"], label="Stage 2 report path") != "docs/stage-2-report.md":
        raise Stage2ExperimentError("Stage 2 report path is invalid")
    _required_sha256(report["sha256"], name="Stage 2 report")


_EXPLICIT_CODE_PATHS = frozenset(
    {
        DATA_FOUNDATION_BASELINE_RELATIVE,
        STAGE1_VERIFIER_RELATIVE,
        STAGE2_AUXILIARY_MANIFEST_RELATIVE,
        STAGE2_METADATA_EXTRACTOR_RELATIVE,
        STAGE2_RUNNER_RELATIVE,
        STAGE2_SOKOBAN_AUDITOR_RELATIVE,
    }
)


def _code_paths_at_commit(root: Path, commit: str) -> tuple[str, ...]:
    raw = _git(
        root,
        "ls-tree",
        "-r",
        "--name-only",
        "-z",
        commit,
        "--",
        "src/token_prediction",
        *sorted(_EXPLICIT_CODE_PATHS),
    )
    paths: list[str] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        try:
            relative = item.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise Stage2ExperimentError("Git returned a non-UTF-8 Stage 2 path") from exc
        relative = _safe_relative(relative, label="Stage 2 historical code path")
        if relative in _EXPLICIT_CODE_PATHS or (
            relative.startswith("src/token_prediction/") and relative.endswith(".py")
        ):
            paths.append(relative)
    resolved = tuple(sorted(set(paths)))
    if not _EXPLICIT_CODE_PATHS <= set(resolved) or not any(
        path.startswith("src/token_prediction/") for path in resolved
    ):
        raise Stage2ExperimentError("historical commit lacks the complete Stage 2 source set")
    return resolved


def _code_hash_at_commit(root: Path, commit: str) -> tuple[str, tuple[str, ...]]:
    paths = _code_paths_at_commit(root, commit)
    items = [(relative, _git(root, "show", f"{commit}:{relative}")) for relative in paths]
    return _framed_code_hash(items), paths


def _resolve_artifact_source(
    root: Path,
    commit: str,
    expected_code_hash: str,
) -> tuple[str, tuple[str, ...]]:
    try:
        _git(root, "cat-file", "-e", f"{commit}^{{commit}}")
    except Stage2ExperimentError:
        pass
    else:
        actual, paths = _code_hash_at_commit(root, commit)
        if actual != expected_code_hash:
            raise Stage2ExperimentError("artifact Git commit does not reproduce the code tree")
        return "artifact_commit_and_source_tree_verified", paths

    raw = _git(root, "rev-list", "--all")
    try:
        revisions = raw.decode("ascii", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise Stage2ExperimentError("Git returned non-ASCII revision identities") from exc
    for revision in revisions:
        try:
            actual, paths = _code_hash_at_commit(root, revision)
        except Stage2ExperimentError:
            continue
        if actual == expected_code_hash:
            return f"source_tree_reproduced_at:{revision}", paths
    raise Stage2ExperimentError("no reachable Git commit reproduces the Stage 2 code tree")


def _require_tracked_clean(root: Path, paths: Sequence[str]) -> None:
    tracked = {
        item.decode("utf-8", errors="strict")
        for item in _git(root, "ls-files", "-z", "--", *paths).split(b"\0")
        if item
    }
    if tracked != set(paths):
        raise Stage2ExperimentError("Stage 2 release controls must be tracked")
    if _git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--",
        *paths,
    ):
        raise Stage2ExperimentError("Stage 2 release controls must be clean at HEAD")


def _verify_experiment_entry(
    root: Path,
    name: str,
    entry: Mapping[str, Any],
    *,
    expected_code_hash: str,
    expected_commit: str,
    expected_code_paths: tuple[str, ...],
) -> tuple[int, int, int, int]:
    artifact_path = _repo_path(root, entry["path"], label=f"Stage 2 artifact {name}")
    manifest = verify_artifact(artifact_path)
    if (
        manifest.artifact_id != entry["artifact_id"]
        or len(manifest.files) != entry["manifest_file_count"]
        or manifest.metadata.get("run_id") != entry["run_id"]
        or manifest.metadata.get("results_payload_sha256")
        != entry["results_payload_sha256"]
    ):
        raise Stage2ExperimentError(f"Stage 2 artifact {name} manifest differs from lock")
    results = _load_json(
        artifact_path / "results.json",
        maximum_bytes=MAX_RESULTS_JSON_BYTES,
        description=f"Stage 2 artifact {name} results",
    )
    if verify_stage2_results_document(results) != entry["results_payload_sha256"]:
        raise Stage2ExperimentError(f"Stage 2 artifact {name} results hash differs")
    code = results["code_binding"]
    source = results["source"]
    dataset = results["dataset"]
    protocol = results["development_protocol"]
    matrix = results["matrix"]
    summary = results["summary"]
    if (
        results["run_id"] != entry["run_id"]
        or code["git_commit"] != expected_commit
        or code["code_tree_sha256"] != expected_code_hash
        or tuple(code["code_paths"]) != expected_code_paths
        or source["source_id"] != entry["source_id"]
        or source["source_descriptor_hash"] != entry["source_descriptor_hash"]
        or dataset["base_dataset_id"] != entry["base_dataset_id"]
        or dataset["derived_dataset_id"] != entry["derived_dataset_id"]
        or dataset["development_dataset_id"] != entry["development_dataset_id"]
        or protocol["protocol_id"] != entry["development_protocol_id"]
        or matrix["matrix_id"] != entry["matrix_id"]
        or summary["experiment_count"] != entry["experiment_count"]
        or summary["candidate_seed_run_count"] != entry["candidate_seed_run_count"]
    ):
        raise Stage2ExperimentError(f"Stage 2 artifact {name} identity differs from lock")
    if results["final_holdout"] != {
        "evaluated": False,
        "prediction_count": 0,
        "selection_claim": "none",
        "target_values_used_for_fit_calibration_scoring": False,
    }:
        raise Stage2ExperimentError(f"Stage 2 artifact {name} final holdout is not sealed")

    run_count = 0
    reloadable_count = 0
    reload_fold_count = 0
    for experiment in results["experiments"]:
        if experiment["alpha"] != 0.1 or experiment["calibrator_id"] != "task_max_conformal":
            raise Stage2ExperimentError(f"Stage 2 artifact {name} calibration differs")
        for candidate in experiment["candidates"]:
            seed_results = candidate["seed_results"]
            if [item["split_seed"] for item in seed_results] != [
                20260719,
                20260720,
                20260721,
            ]:
                raise Stage2ExperimentError(f"Stage 2 artifact {name} seeds differ")
            for seed_result in seed_results:
                run_count += 1
                parity = seed_result["bundle_reload_parity"]
                status = parity["status"]
                folds = _integer(
                    parity["fold_count"],
                    description=f"Stage 2 artifact {name} reload fold count",
                )
                if status == "exact_during_execution":
                    if folds != 5:
                        raise Stage2ExperimentError(
                            f"Stage 2 artifact {name} exact reload omitted folds"
                        )
                    reloadable_count += 1
                    reload_fold_count += folds
                elif status == "not_applicable_stateless_or_mechanical":
                    if folds != 0:
                        raise Stage2ExperimentError(
                            f"Stage 2 artifact {name} mechanical reload has folds"
                        )
                else:
                    raise Stage2ExperimentError(
                        f"Stage 2 artifact {name} has unsupported reload status"
                    )
    if run_count != entry["candidate_seed_run_count"]:
        raise Stage2ExperimentError(f"Stage 2 artifact {name} run count does not close")
    return len(manifest.files), run_count, reloadable_count, reload_fold_count


def _verify_audit_entry(root: Path, entry: Mapping[str, Any]) -> int:
    artifact_path = _repo_path(root, entry["path"], label="Stage 2 Sokoban audit")
    manifest = verify_artifact(artifact_path)
    if (
        manifest.artifact_id != entry["artifact_id"]
        or len(manifest.files) != entry["manifest_file_count"]
        or manifest.metadata.get("run_id") != entry["run_id"]
        or manifest.metadata.get("results_payload_sha256")
        != entry["results_payload_sha256"]
    ):
        raise Stage2ExperimentError("Stage 2 Sokoban audit manifest differs from lock")
    results = _load_json(
        artifact_path / "results.json",
        maximum_bytes=MAX_RESULTS_JSON_BYTES,
        description="Stage 2 Sokoban audit results",
    )
    if verify_sokoban_audit_results(results) != entry["results_payload_sha256"]:
        raise Stage2ExperimentError("Stage 2 Sokoban audit payload differs from lock")
    source = results["source"]
    stage1 = results["stage1_regression"]
    if (
        source["source_id"] != entry["source_id"]
        or source["source_descriptor_hash"] != entry["source_descriptor_hash"]
        or stage1["artifact_id"] != entry["stage1_artifact_id"]
        or stage1["bundle_count"] != entry["stage1_bundle_count"]
        or stage1["parity_record_count"] != entry["stage1_parity_record_count"]
        or stage1["parity_mismatch_count"] != entry["stage1_parity_mismatch_count"]
    ):
        raise Stage2ExperimentError("Stage 2 Sokoban audit evidence differs from lock")
    return len(manifest.files)


def verify_stage2_release(
    repository_root: str | Path,
    *,
    lock_path: str = DEFAULT_RELEASE_LOCK,
    tracked_only: bool = False,
    require_git_clean: bool = True,
) -> Stage2ReleaseVerification:
    root = Path(repository_root).resolve()
    relative_lock = _safe_relative(lock_path, label="Stage 2 release lock path")
    lock_file = _repo_path(root, relative_lock, label="Stage 2 release lock")
    release = _load_json(
        lock_file,
        maximum_bytes=MAX_RELEASE_JSON_BYTES,
        description="Stage 2 release lock",
    )
    _validate_release_document(release)
    report = release["report"]
    report_path = _repo_path(root, report["path"], label="Stage 2 report")
    if _is_link_or_reparse(report_path) or _sha256_file(report_path) != report["sha256"]:
        raise Stage2ExperimentError("Stage 2 report differs from release lock")

    controls = (relative_lock, str(report["path"]), "scripts/verify_stage2_release.py")
    if require_git_clean:
        _require_tracked_clean(root, controls)
    expected_code = str(release["code_binding"]["code_tree_sha256"])
    expected_commit = str(release["code_binding"]["artifact_git_commit"])
    commit_status, expected_code_paths = _resolve_artifact_source(
        root,
        expected_commit,
        expected_code,
    )

    verified_artifacts = 0
    file_count = 0
    run_count = 0
    reloadable_count = 0
    reload_fold_count = 0
    if not tracked_only:
        for name in sorted(EXPERIMENT_NAMES):
            counts = _verify_experiment_entry(
                root,
                name,
                release["artifacts"][name],
                expected_code_hash=expected_code,
                expected_commit=expected_commit,
                expected_code_paths=expected_code_paths,
            )
            files, runs, reloadable, reload_folds = counts
            file_count += files
            run_count += runs
            reloadable_count += reloadable
            reload_fold_count += reload_folds
            verified_artifacts += 1
        file_count += _verify_audit_entry(root, release["artifacts"][AUDIT_NAME])
        verified_artifacts += 1
        totals = release["totals"]
        if (
            file_count != totals["manifest_file_count"]
            or run_count != totals["candidate_seed_run_count"]
            or reloadable_count != totals["reloadable_candidate_seed_run_count"]
            or reload_fold_count != totals["exact_reload_fold_count"]
        ):
            raise Stage2ExperimentError("Stage 2 verified totals differ from release lock")

    return Stage2ReleaseVerification(
        lock_path=relative_lock,
        report_path=str(report["path"]),
        code_tree_sha256=expected_code,
        artifact_commit_status=commit_status,
        locked_artifact_count=len(release["artifacts"]),
        verified_artifact_count=verified_artifacts,
        manifest_file_count=file_count,
        candidate_seed_run_count=run_count,
        reloadable_candidate_seed_run_count=reloadable_count,
        exact_reload_fold_count=reload_fold_count,
        final_holdout_evaluated=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the frozen Stage 2 release")
    parser.add_argument(
        "--repository-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument("--lock", default=DEFAULT_RELEASE_LOCK)
    parser.add_argument(
        "--tracked-only",
        action="store_true",
        help="verify tracked controls and code binding without ignored local artifacts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = verify_stage2_release(
            args.repository_root,
            lock_path=args.lock,
            tracked_only=args.tracked_only,
        )
    except (OSError, Stage2ExperimentError, ValueError) as exc:
        raise SystemExit(f"Stage 2 release verification failed: {exc}") from exc
    print(json.dumps(asdict(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
