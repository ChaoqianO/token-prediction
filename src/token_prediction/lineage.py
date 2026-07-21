from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from token_prediction.recording.redaction import redact_secrets


class ArtifactVerificationError(RuntimeError):
    pass


_ROOT_CONTROL_FILES = frozenset({"manifest.json", "_SUCCESS"})


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ArtifactManifest:
    artifact_id: str
    stage_name: str
    schema_version: int
    files: dict[str, str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "stage_name": self.stage_name,
            "schema_version": self.schema_version,
            "files": dict(sorted(self.files.items())),
            "metadata": self.metadata,
        }


def publish_artifact(
    root: str | Path,
    *,
    stage_name: str,
    schema_version: int = 1,
    metadata: Mapping[str, Any] | None = None,
) -> ArtifactManifest:
    directory = Path(root).resolve()
    success = directory / "_SUCCESS"
    if success.exists():
        raise FileExistsError(f"published artifact is immutable: {directory}")
    files: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        if relative in _ROOT_CONTROL_FILES:
            continue
        files[relative] = sha256_file(path)
    safe_metadata = redact_secrets(dict(metadata or {}))
    semantic = {
        "stage_name": stage_name,
        "schema_version": schema_version,
        "files": files,
        "metadata": safe_metadata,
    }
    artifact_id = hashlib.sha256(_canonical_json(semantic).encode("utf-8")).hexdigest()
    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        stage_name=stage_name,
        schema_version=schema_version,
        files=files,
        metadata=safe_metadata,
    )
    manifest_path = directory / "manifest.json"
    temporary = directory / "manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    success.write_text(artifact_id + "\n", encoding="utf-8")
    return manifest


def verify_artifact(root: str | Path) -> ArtifactManifest:
    directory = Path(root).resolve()
    manifest_path = directory / "manifest.json"
    success_path = directory / "_SUCCESS"
    if not manifest_path.exists() or not success_path.exists():
        raise ArtifactVerificationError("artifact is incomplete")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = ArtifactManifest(
        artifact_id=str(payload.get("artifact_id") or ""),
        stage_name=str(payload.get("stage_name") or ""),
        schema_version=int(payload.get("schema_version") or 0),
        files=dict(payload.get("files") or {}),
        metadata=dict(payload.get("metadata") or {}),
    )
    if success_path.read_text(encoding="utf-8").strip() != manifest.artifact_id:
        raise ArtifactVerificationError("_SUCCESS does not match manifest")
    actual_files = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
        and path.relative_to(directory).as_posix() not in _ROOT_CONTROL_FILES
    }
    if actual_files != set(manifest.files):
        missing = sorted(set(manifest.files) - actual_files)
        extra = sorted(actual_files - set(manifest.files))
        raise ArtifactVerificationError(
            f"artifact file set mismatch; missing={missing}, extra={extra}"
        )
    for relative, expected in manifest.files.items():
        path = directory / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ArtifactVerificationError(f"artifact file checksum mismatch: {relative}")
    semantic = {
        "stage_name": manifest.stage_name,
        "schema_version": manifest.schema_version,
        "files": manifest.files,
        "metadata": manifest.metadata,
    }
    expected_id = hashlib.sha256(_canonical_json(semantic).encode("utf-8")).hexdigest()
    if expected_id != manifest.artifact_id:
        raise ArtifactVerificationError("artifact semantic hash mismatch")
    return manifest
