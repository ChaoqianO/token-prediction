from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from token_prediction.collection import BagenSokobanMetadata, BagenSokobanReader
from token_prediction.contracts import EventType
from token_prediction.dataset import (
    PredictionPosition,
    PredictionTarget,
    build_supervised_dataset,
)
from token_prediction.estimators import RunContext, load_lightgbm_bundle
from token_prediction.lineage import ArtifactVerificationError, sha256_file, verify_artifact
from token_prediction.trajectory import Trajectory


BASELINE_SCHEMA_VERSION = 1
STAGE_NAME = "preliminary_lightgbm_experiment"
BAGEN_EXPERIMENT_ID = "bagen_codex_task_update"
BAGEN_CANDIDATES = frozenset(
    {"lightgbm_history_only", "lightgbm_history_request_proxy"}
)
PARITY_PROJECTION = "legacy_proxy_projection_v1"
STAGE1_SCRIPT = "scripts/run_lightgbm_preliminary.py"
BUNDLE_MANIFEST = re.compile(
    r"^models/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/fold_([0-9]+)/bundle/manifest\.json$"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class BaselineVerificationError(RuntimeError):
    """The Stage 1 artifact, parity result, or source binding did not close."""


@dataclass(frozen=True)
class VerificationSummary:
    artifact_id: str
    artifact_manifest_sha256: str
    bagen_source_sha256: str
    bundle_count: int
    parity_candidates: tuple[str, ...]
    parity_projection: str
    parity_record_count: int
    parity_mismatch_count: int
    parity_sha256: str
    protocol_code_sha256: str
    source_binding_status: str
    source_commit: str | None


def _reject_json_constant(value: str) -> Any:
    raise BaselineVerificationError(f"non-finite JSON constant is forbidden: {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BaselineVerificationError("duplicate JSON field is forbidden")
        result[key] = value
    return result


def _reject_non_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise BaselineVerificationError("non-finite JSON number is forbidden")
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_non_finite(item)
    elif isinstance(value, list):
        for item in value:
            _reject_non_finite(item)


def _strict_json_loads(payload: str, *, label: str) -> Any:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise BaselineVerificationError(f"{label} is not valid JSON") from exc
    _reject_non_finite(value)
    return value


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise BaselineVerificationError(f"{label} must be a regular file")
    try:
        value = _strict_json_loads(path.read_text(encoding="utf-8"), label=label)
    except (OSError, UnicodeDecodeError) as exc:
        raise BaselineVerificationError(f"cannot read {label}") from exc
    if not isinstance(value, dict):
        raise BaselineVerificationError(f"{label} must be a JSON object")
    return value


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _require_text(value: Mapping[str, Any], key: str, *, label: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise BaselineVerificationError(f"{label}.{key} must be a non-empty string")
    return item


def _require_sha(value: Mapping[str, Any], key: str, *, label: str) -> str:
    item = _require_text(value, key, label=label)
    if not SHA256.fullmatch(item):
        raise BaselineVerificationError(f"{label}.{key} must be a lowercase SHA256")
    return item


def _require_int(value: Mapping[str, Any], key: str, *, label: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int) or item < 0:
        raise BaselineVerificationError(f"{label}.{key} must be a non-negative integer")
    return item


def _safe_relative(value: str, *, label: str) -> str:
    if not value or "\\" in value or "\x00" in value:
        raise BaselineVerificationError(f"{label} is not a safe relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise BaselineVerificationError(f"{label} is not a safe relative path")
    if ":" in path.parts[0]:
        raise BaselineVerificationError(f"{label} is not a safe relative path")
    return path.as_posix()


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _git_executable() -> str:
    discovered = shutil.which("git")
    if discovered:
        return discovered
    if os.name == "nt":
        candidate = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git/cmd/git.exe"
        if candidate.is_file():
            return str(candidate)
    raise BaselineVerificationError("git is required for source binding")


def _git(root: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            [_git_executable(), "-C", str(root), *arguments],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BaselineVerificationError("git source-binding command failed") from exc
    return result.stdout


def _resolve_commit(root: Path, revision: str) -> str:
    if not revision or revision.startswith("-"):
        raise BaselineVerificationError("source revision is unsafe")
    raw = _git(root, "rev-parse", "--verify", f"{revision}^{{commit}}")
    try:
        commit = raw.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise BaselineVerificationError("git returned a non-ASCII commit") from exc
    if not COMMIT_SHA.fullmatch(commit):
        raise BaselineVerificationError("git did not resolve a full commit SHA")
    return commit


def _code_hash_at_commit(root: Path, revision: str) -> tuple[str, str]:
    commit = _resolve_commit(root, revision)
    raw_paths = _git(
        root,
        "ls-tree",
        "-r",
        "--name-only",
        "-z",
        commit,
        "--",
        "src",
        STAGE1_SCRIPT,
    )
    paths: list[str] = []
    for raw in raw_paths.split(b"\0"):
        if not raw:
            continue
        try:
            path = _safe_relative(raw.decode("utf-8", errors="strict"), label="git path")
        except UnicodeDecodeError as exc:
            raise BaselineVerificationError("git returned a non-UTF-8 source path") from exc
        if (path.startswith("src/") and path.endswith(".py")) or path == STAGE1_SCRIPT:
            paths.append(path)
    path_set = set(paths)
    if STAGE1_SCRIPT not in path_set or not any(
        path.startswith("src/") for path in path_set
    ):
        raise BaselineVerificationError("commit does not contain the complete Stage 1 source set")
    paths = sorted(path for path in path_set if path != STAGE1_SCRIPT)
    paths.append(STAGE1_SCRIPT)
    digest = hashlib.sha256()
    for path in paths:
        payload = _git(root, "show", f"{commit}:{path}")
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return commit, digest.hexdigest()


def discover_source_commit(root: Path, code_hash: str) -> str | None:
    raw = _git(root, "rev-list", "--all")
    try:
        revisions = raw.decode("ascii", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise BaselineVerificationError("git returned a non-ASCII revision list") from exc
    for revision in revisions:
        try:
            commit, candidate_hash = _code_hash_at_commit(root, revision)
        except BaselineVerificationError:
            continue
        if candidate_hash == code_hash:
            return commit
    return None


def _artifact_manifest(artifact: Path, expected_artifact_id: str | None) -> Any:
    if artifact.is_symlink() or not artifact.is_dir():
        raise BaselineVerificationError("artifact root must be a regular directory")
    if any(path.is_symlink() for path in artifact.rglob("*")):
        raise BaselineVerificationError("artifact must not contain symbolic links")
    try:
        manifest = verify_artifact(artifact)
    except ArtifactVerificationError as exc:
        raise BaselineVerificationError("artifact integrity verification failed") from exc
    if manifest.stage_name != STAGE_NAME or manifest.schema_version != 1:
        raise BaselineVerificationError("artifact stage or schema is unsupported")
    if expected_artifact_id is not None and manifest.artifact_id != expected_artifact_id:
        raise BaselineVerificationError("artifact ID does not match the expected baseline")
    return manifest


def _load_bundles(artifact: Path, files: Mapping[str, str]) -> dict[tuple[str, str, int], Any]:
    keys: dict[tuple[str, str, int], Path] = {}
    for raw_path in files:
        relative = _safe_relative(raw_path, label="artifact manifest path")
        match = BUNDLE_MANIFEST.fullmatch(relative)
        if match is None:
            continue
        key = (match.group(1), match.group(2), int(match.group(3)))
        bundle = artifact / PurePosixPath(relative).parent
        if key in keys:
            raise BaselineVerificationError("artifact contains a duplicate bundle scope")
        keys[key] = bundle
    if not keys:
        raise BaselineVerificationError("artifact contains no deployable bundles")
    loaded: dict[tuple[str, str, int], Any] = {}
    for key, bundle in sorted(keys.items()):
        try:
            loaded[key] = load_lightgbm_bundle(bundle)
        except Exception as exc:
            raise BaselineVerificationError("a Stage 1 bundle failed strict loading") from exc
    return loaded


def _legacy_proxy_projection(trajectory: Trajectory) -> Trajectory:
    """Reconstruct only the explicitly non-causal feature projection used by Stage 1.

    This never mutates the source reader or schema-v2 datasets. The historical
    pilot copied each call's first observed provider input audit into its
    request feature and used the first logical call's value as its task-token
    proxy. If that first call had no usage, the task proxy remained missing.
    """

    provider_input_by_call: dict[str, int] = {}
    for event in trajectory.events:
        if event.event_type not in {EventType.API_COMPLETED, EventType.API_FAILED}:
            continue
        call_id = event.logical_call_id
        value = event.payload.get("provider_input_tokens_post_response_audit")
        if call_id is None or value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BaselineVerificationError(
                "BAGEN provider-input audit is invalid for Stage 1 compatibility"
            )
        # The retired reader used the first provider input value it could
        # observe for a logical call, rather than the sum across retry attempts.
        # Preserve that quirk only in this artifact-compatibility projection.
        provider_input_by_call.setdefault(call_id, value)

    first_request_call = next(
        (
            event.logical_call_id
            for event in trajectory.events
            if event.event_type == EventType.REQUEST_BUILT
        ),
        None,
    )
    compatibility_events = []
    for event in trajectory.events:
        payload = event.payload
        if event.event_type == EventType.TASK_STARTED:
            payload["task_tokens"] = provider_input_by_call.get(first_request_call or "")
            payload["task_tokens_source"] = "historical_provider_input_proxy"
            compatibility_events.append(event.with_payload(payload))
            continue
        if event.event_type != EventType.REQUEST_BUILT:
            compatibility_events.append(event)
            continue
        call_id = event.logical_call_id
        if call_id is None or call_id not in provider_input_by_call:
            compatibility_events.append(event)
            continue
        payload["request_tokens_local"] = provider_input_by_call[call_id]
        payload["request_token_count_source"] = "historical_provider_input_proxy"
        compatibility_events.append(event.with_payload(payload))
    return Trajectory.from_events(compatibility_events)


def _build_bagen_points(path: Path) -> dict[str, Any]:
    current = BagenSokobanReader().read_all(
        path,
        BagenSokobanMetadata(reasoning_effort="low"),
    )
    trajectories = [_legacy_proxy_projection(trajectory) for trajectory in current]
    dataset = build_supervised_dataset(trajectories)
    return {
        row.point.point_id: row.point
        for row in dataset.rows
        if row.point.position == PredictionPosition.TASK_UPDATE
        and row.point.target == PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS
    }


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineVerificationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise BaselineVerificationError(f"{label} must be finite")
    return result


def _prediction_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise BaselineVerificationError("artifact predictions must be a regular file")
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise BaselineVerificationError("predictions contain a blank line")
                value = _strict_json_loads(line, label=f"prediction line {line_number}")
                if not isinstance(value, dict):
                    raise BaselineVerificationError("prediction record must be an object")
                records.append(value)
    except (OSError, UnicodeDecodeError) as exc:
        raise BaselineVerificationError("cannot read artifact predictions") from exc
    if not records:
        raise BaselineVerificationError("artifact predictions are empty")
    return records


def _verify_parity(
    artifact: Path,
    bagen_json: Path,
    bundles: Mapping[tuple[str, str, int], Any],
) -> tuple[int, int, str]:
    points = _build_bagen_points(bagen_json)
    records = _prediction_records(artifact / "predictions.jsonl")
    selected = [
        record
        for record in records
        if record.get("experiment_id") == BAGEN_EXPERIMENT_ID
        and record.get("candidate_id") in BAGEN_CANDIDATES
    ]
    if not selected:
        raise BaselineVerificationError("artifact has no Stage 1 BAGEN learned predictions")
    candidate_counts = {
        candidate: sum(record.get("candidate_id") == candidate for record in selected)
        for candidate in BAGEN_CANDIDATES
    }
    if any(count == 0 for count in candidate_counts.values()) or len(
        set(candidate_counts.values())
    ) != 1:
        raise BaselineVerificationError(
            "artifact parity must cover both Stage 1 BAGEN learned candidates equally"
        )

    seen: set[tuple[str, str]] = set()
    mismatch_count = 0
    digest = hashlib.sha256(b"stage1-bagen-raw-parity-v1\0")
    for index, record in enumerate(selected):
        candidate = _require_text(record, "candidate_id", label="prediction")
        point_id = _require_text(record, "point_id", label="prediction")
        fold = _require_int(record, "fold", label="prediction")
        record_key = (candidate, point_id)
        if record_key in seen:
            raise BaselineVerificationError("artifact contains a duplicate parity prediction")
        seen.add(record_key)
        bundle_key = (BAGEN_EXPERIMENT_ID, candidate, fold)
        if bundle_key not in bundles:
            raise BaselineVerificationError("prediction has no matching deployable bundle")
        point = points.get(point_id)
        if point is None:
            raise BaselineVerificationError("prediction point is absent from the pinned BAGEN input")

        expected = (
            _number(record.get("raw_lower"), label="raw_lower"),
            _number(record.get("raw_prediction"), label="raw_prediction"),
            _number(record.get("raw_upper"), label="raw_upper"),
        )
        fitted = bundles[bundle_key]
        forecast = fitted.start(
            RunContext(point.task_id, point.trajectory_id, point.run_id)
        ).predict(point)
        actual = (forecast.raw_lower, forecast.raw_point, forecast.raw_upper)
        if actual != expected:
            mismatch_count += 1
        digest.update(
            _canonical_bytes(
                {
                    "candidate": candidate,
                    "fold": fold,
                    "index": index,
                    "point_id_sha256": hashlib.sha256(point_id.encode("utf-8")).hexdigest(),
                    "raw": list(actual),
                }
            )
        )
    return len(selected), mismatch_count, digest.hexdigest()


def _protocol(artifact: Path) -> tuple[str, str]:
    protocol = _load_mapping(artifact / "protocol.json", label="protocol")
    code_hash = _require_sha(protocol, "code_hash", label="protocol")
    source_hashes = protocol.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise BaselineVerificationError("protocol.source_hashes must be an object")
    bagen_hash = _require_sha(source_hashes, "bagen_json", label="protocol.source_hashes")
    return code_hash, bagen_hash


def _baseline_document(summary: VerificationSummary) -> dict[str, Any]:
    if summary.source_binding_status != "bound" or summary.source_commit is None:
        raise BaselineVerificationError("a baseline file requires a recoverable source commit")
    return {
        "baseline_schema_version": BASELINE_SCHEMA_VERSION,
        "artifact": {
            "artifact_id": summary.artifact_id,
            "manifest_sha256": summary.artifact_manifest_sha256,
        },
        "bagen_source_sha256": summary.bagen_source_sha256,
        "bundle_count": summary.bundle_count,
        "parity": {
            "candidates": list(summary.parity_candidates),
            "mismatch_count": summary.parity_mismatch_count,
            "projection": summary.parity_projection,
            "record_count": summary.parity_record_count,
            "sha256": summary.parity_sha256,
        },
        "source": {
            "code_sha256": summary.protocol_code_sha256,
            "commit_sha": summary.source_commit,
        },
    }


def _baseline_source_commit(document: Mapping[str, Any]) -> str:
    if set(document) != {
        "artifact",
        "bagen_source_sha256",
        "baseline_schema_version",
        "bundle_count",
        "parity",
        "source",
    }:
        raise BaselineVerificationError("baseline file has an unsupported field set")
    if document.get("baseline_schema_version") != BASELINE_SCHEMA_VERSION:
        raise BaselineVerificationError("baseline file schema version is unsupported")
    source = document.get("source")
    if not isinstance(source, dict) or set(source) != {"code_sha256", "commit_sha"}:
        raise BaselineVerificationError("baseline source binding is malformed")
    _require_sha(source, "code_sha256", label="baseline.source")
    commit = _require_text(source, "commit_sha", label="baseline.source")
    if not COMMIT_SHA.fullmatch(commit):
        raise BaselineVerificationError("baseline source commit must be a full lowercase SHA")
    return commit


def verify_stage1(
    artifact: str | Path,
    bagen_json: str | Path,
    *,
    repository_root: str | Path,
    expected_artifact_id: str | None = None,
    expected_bundles: int | None = None,
    expected_parity: int | None = None,
    source_commit: str | None = None,
    discover_commit: bool = False,
) -> VerificationSummary:
    root = Path(repository_root).resolve()
    artifact_path = Path(artifact)
    bagen_path = Path(bagen_json)
    manifest = _artifact_manifest(artifact_path, expected_artifact_id)
    code_hash, expected_bagen_hash = _protocol(artifact_path)
    actual_bagen_hash = _sha256(bagen_path)
    if actual_bagen_hash != expected_bagen_hash:
        raise BaselineVerificationError("BAGEN source hash does not match the artifact protocol")

    bundles = _load_bundles(artifact_path, manifest.files)
    if expected_bundles is not None and len(bundles) != expected_bundles:
        raise BaselineVerificationError("bundle count does not match the expected baseline")
    parity_count, mismatches, parity_hash = _verify_parity(
        artifact_path, bagen_path, bundles
    )
    if expected_parity is not None and parity_count != expected_parity:
        raise BaselineVerificationError("parity record count does not match the expected baseline")
    if mismatches:
        raise BaselineVerificationError("raw bundle prediction parity failed")

    resolved_commit: str | None = None
    source_status = "unassessed"
    if source_commit is not None:
        resolved_commit, commit_hash = _code_hash_at_commit(root, source_commit)
        if commit_hash != code_hash:
            raise BaselineVerificationError(
                "artifact code hash does not match the requested source commit"
            )
        source_status = "bound"
    elif discover_commit:
        resolved_commit = discover_source_commit(root, code_hash)
        source_status = "bound" if resolved_commit is not None else "unrecoverable"

    return VerificationSummary(
        artifact_id=manifest.artifact_id,
        artifact_manifest_sha256=_sha256(artifact_path / "manifest.json"),
        bagen_source_sha256=actual_bagen_hash,
        bundle_count=len(bundles),
        parity_candidates=tuple(sorted(BAGEN_CANDIDATES)),
        parity_projection=PARITY_PROJECTION,
        parity_record_count=parity_count,
        parity_mismatch_count=mismatches,
        parity_sha256=parity_hash,
        protocol_code_sha256=code_hash,
        source_binding_status=source_status,
        source_commit=resolved_commit,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a Stage 1 artifact, reload every bundle, reproduce BAGEN raw "
            "predictions, and optionally bind the recorded source hash to a Git commit."
        )
    )
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--bagen-json", required=True, type=Path)
    parser.add_argument("--expected-artifact-id")
    parser.add_argument("--expected-bundles", type=int)
    parser.add_argument("--expected-parity", type=int)
    parser.add_argument("--source-commit")
    parser.add_argument("--discover-source-commit", action="store_true")
    baseline = parser.add_mutually_exclusive_group()
    baseline.add_argument("--baseline", type=Path, help="verify an existing bound baseline file")
    baseline.add_argument(
        "--write-baseline", type=Path, help="write a new bound baseline file after verification"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    baseline_document: dict[str, Any] | None = None
    source_commit = args.source_commit
    try:
        if args.expected_artifact_id is not None and not SHA256.fullmatch(
            args.expected_artifact_id
        ):
            raise BaselineVerificationError("expected artifact ID must be a lowercase SHA256")
        if args.expected_bundles is not None and args.expected_bundles <= 0:
            raise BaselineVerificationError("expected bundle count must be positive")
        if args.expected_parity is not None and args.expected_parity <= 0:
            raise BaselineVerificationError("expected parity count must be positive")
        if args.baseline is not None:
            baseline_document = _load_mapping(args.baseline, label="baseline")
            bound_commit = _baseline_source_commit(baseline_document)
            if source_commit is not None and _resolve_commit(
                repository_root, source_commit
            ) != bound_commit:
                raise BaselineVerificationError("CLI source commit disagrees with baseline file")
            source_commit = bound_commit

        summary = verify_stage1(
            args.artifact,
            args.bagen_json,
            repository_root=repository_root,
            expected_artifact_id=args.expected_artifact_id,
            expected_bundles=args.expected_bundles,
            expected_parity=args.expected_parity,
            source_commit=source_commit,
            discover_commit=args.discover_source_commit,
        )
        if baseline_document is not None:
            if baseline_document != _baseline_document(summary):
                raise BaselineVerificationError("baseline file does not match verification results")
        if args.write_baseline is not None:
            if args.write_baseline.exists():
                raise BaselineVerificationError("refusing to overwrite an existing baseline file")
            document = _baseline_document(summary)
            args.write_baseline.parent.mkdir(parents=True, exist_ok=True)
            args.write_baseline.write_bytes(_canonical_bytes(document))
    except (BaselineVerificationError, OSError, UnicodeError, ValueError) as exc:
        print(f"Stage 1 baseline verification failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(asdict(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
