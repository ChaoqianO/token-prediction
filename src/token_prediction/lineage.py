from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

from token_prediction.recording.redaction import redact_secrets


class ArtifactVerificationError(RuntimeError):
    pass


_ROOT_CONTROL_FILES = frozenset({"manifest.json", "_SUCCESS"})
_MANIFEST_TEMP = "manifest.json.tmp"
_MAX_ARTIFACT_ENTRIES = 100_000
_MAX_ARTIFACT_DEPTH = 64
_MAX_ARTIFACT_FILE_BYTES = 8 * 1024**3
_MAX_ARTIFACT_TOTAL_BYTES = 32 * 1024**3
_MAX_MANIFEST_BYTES = 16 * 1024**2
_MAX_SUCCESS_BYTES = 65
_READ_CHUNK_BYTES = 1024 * 1024


def _is_reparse(metadata: os.stat_result) -> bool:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return bool(attributes & flag)


def _reliable_identity(metadata: os.stat_result) -> tuple[int, int] | None:
    """Return a filesystem identity only when both fields are usable.

    Some Windows filesystems expose zero device/inode values.  Those sentinel
    values are never treated as a wildcard: callers fall back to a complete
    metadata snapshot and, for verified artifacts, a second byte-for-byte hash
    pass plus a final membership scan.
    """

    device = int(getattr(metadata, "st_dev", 0))
    inode = int(getattr(metadata, "st_ino", 0))
    if device == 0 or inode == 0:
        return None
    return device, inode


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    if stat.S_IFMT(left.st_mode) != stat.S_IFMT(right.st_mode):
        return False
    left_identity = _reliable_identity(left)
    right_identity = _reliable_identity(right)
    if (left_identity is None) != (right_identity is None):
        return False
    if left_identity is not None:
        return left_identity == right_identity
    return _fallback_snapshot(left) == _fallback_snapshot(right)


def _fallback_snapshot(
    metadata: os.stat_result,
) -> tuple[int, int | None, int, int, int]:
    """Capture all portable replacement signals when identity is unavailable."""

    return (
        stat.S_IFMT(metadata.st_mode),
        int(metadata.st_size) if stat.S_ISREG(metadata.st_mode) else None,
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
        int(_is_reparse(metadata)),
    )


def _snapshot(
    metadata: os.stat_result,
) -> tuple[tuple[int, int] | None, tuple[int, int | None, int, int, int]]:
    return (
        _reliable_identity(metadata),
        _fallback_snapshot(metadata),
    )


def _same_snapshot(left: os.stat_result, right: os.stat_result) -> bool:
    left_identity, left_fallback = _snapshot(left)
    right_identity, right_fallback = _snapshot(right)
    if left_fallback != right_fallback:
        return False
    if (left_identity is None) != (right_identity is None):
        return False
    return left_identity is None or left_identity == right_identity


def _lstat(path: Path, *, description: str) -> os.stat_result:
    try:
        return path.lstat()
    except OSError as exc:
        raise ArtifactVerificationError(f"cannot inspect {description}: {path}") from exc


def _require_directory(path: Path, *, description: str) -> os.stat_result:
    metadata = _lstat(path, description=description)
    if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
        raise ArtifactVerificationError(
            f"{description} must not be a symlink or reparse point: {path}"
        )
    if not stat.S_ISDIR(metadata.st_mode):
        raise ArtifactVerificationError(f"{description} is not a directory: {path}")
    return metadata


def _require_regular_file(path: Path, *, description: str) -> os.stat_result:
    metadata = _lstat(path, description=description)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise ArtifactVerificationError(
            f"{description} must be a regular non-link file: {path}"
        )
    return metadata


def _open_regular(
    path: Path,
    *,
    description: str,
    expected_metadata: os.stat_result | None = None,
) -> tuple[int, os.stat_result, os.stat_result]:
    before = _require_regular_file(path, description=description)
    if expected_metadata is not None and not _same_snapshot(before, expected_metadata):
        raise ArtifactVerificationError(
            f"{description} changed after the artifact tree was scanned: {path}"
        )
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(
        getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ArtifactVerificationError(f"cannot open {description}: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(opened.st_mode)
            or _is_reparse(opened)
            or not stat.S_ISREG(opened.st_mode)
            or not _same_identity(opened, before)
        ):
            raise ArtifactVerificationError(
                f"{description} changed identity or resolved through a link: {path}"
            )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, before, opened


def _read_regular(
    path: Path,
    *,
    maximum_bytes: int,
    description: str,
    collect: bool,
    expected_metadata: os.stat_result | None = None,
) -> tuple[str, int, bytes | None]:
    descriptor, before, opened = _open_regular(
        path,
        description=description,
        expected_metadata=expected_metadata,
    )
    if before.st_size < 0 or before.st_size > maximum_bytes:
        os.close(descriptor)
        raise ArtifactVerificationError(f"{description} exceeds its safe size limit: {path}")
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if collect else None
    total = 0
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            while True:
                chunk = handle.read(_READ_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum_bytes:
                    raise ArtifactVerificationError(
                        f"{description} exceeds its safe size limit while reading: {path}"
                    )
                digest.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
            opened_after = os.fstat(handle.fileno())
    except BaseException:
        raise
    after = _require_regular_file(path, description=description)
    if (
        total != opened.st_size
        or opened.st_size != before.st_size
        or not _same_snapshot(opened_after, opened)
        or not _same_snapshot(after, before)
        or (
            expected_metadata is not None
            and not _same_snapshot(after, expected_metadata)
        )
    ):
        raise ArtifactVerificationError(f"{description} changed while being read: {path}")
    payload = b"".join(chunks) if chunks is not None else None
    return digest.hexdigest(), total, payload


def sha256_file(path: str | Path) -> str:
    digest, _size, _payload = _read_regular(
        Path(path),
        maximum_bytes=_MAX_ARTIFACT_FILE_BYTES,
        description="artifact file",
        collect=False,
    )
    return digest


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


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


@dataclass(frozen=True)
class _ArtifactEntry:
    path: Path
    metadata: os.stat_result


@dataclass(frozen=True)
class _ArtifactTree:
    root: _ArtifactEntry
    files: Mapping[str, _ArtifactEntry]
    directories: Mapping[str, _ArtifactEntry]
    total_bytes: int


def _artifact_root(root: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(root).expanduser())))


def _scan_artifact(directory: Path) -> _ArtifactTree:
    root_metadata = _require_directory(directory, description="artifact root")
    files: dict[str, _ArtifactEntry] = {}
    directories: dict[str, _ArtifactEntry] = {}
    entry_count = 0
    total_bytes = 0

    def visit(
        current: Path,
        relative: PurePosixPath,
        expected_metadata: os.stat_result,
    ) -> None:
        nonlocal entry_count, total_bytes
        before = _require_directory(current, description="artifact directory")
        if not _same_snapshot(before, expected_metadata):
            raise ArtifactVerificationError(
                f"artifact directory changed identity before enumeration: {current}"
            )
        try:
            entries = os.scandir(current)
        except OSError as exc:
            raise ArtifactVerificationError(
                f"cannot enumerate artifact directory: {current}"
            ) from exc
        with entries:
            for entry in entries:
                entry_count += 1
                if entry_count > _MAX_ARTIFACT_ENTRIES:
                    raise ArtifactVerificationError(
                        "artifact exceeds its safe entry-count limit"
                    )
                if not entry.name or entry.name in {".", ".."} or "\\" in entry.name:
                    raise ArtifactVerificationError(
                        f"artifact contains an unsafe entry name: {entry.name!r}"
                    )
                child_relative = relative / entry.name
                if len(child_relative.parts) > _MAX_ARTIFACT_DEPTH:
                    raise ArtifactVerificationError(
                        "artifact exceeds its safe directory-depth limit"
                    )
                child = child_relative.as_posix()
                path = current / entry.name
                metadata = _lstat(path, description=f"artifact entry {child}")
                if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
                    raise ArtifactVerificationError(
                        f"artifact entry is a symlink or reparse point: {child}"
                    )
                if stat.S_ISDIR(metadata.st_mode):
                    directories[child] = _ArtifactEntry(path, metadata)
                    visit(path, child_relative, metadata)
                elif stat.S_ISREG(metadata.st_mode):
                    if metadata.st_size < 0 or metadata.st_size > _MAX_ARTIFACT_FILE_BYTES:
                        raise ArtifactVerificationError(
                            f"artifact file exceeds its safe size limit: {child}"
                        )
                    total_bytes += int(metadata.st_size)
                    if total_bytes > _MAX_ARTIFACT_TOTAL_BYTES:
                        raise ArtifactVerificationError(
                            "artifact exceeds its safe total-byte limit"
                        )
                    files[child] = _ArtifactEntry(path, metadata)
                else:
                    raise ArtifactVerificationError(
                        f"artifact entry is not a regular file or directory: {child}"
                    )
        after = _require_directory(current, description="artifact directory")
        if not _same_snapshot(after, before):
            raise ArtifactVerificationError(
                f"artifact directory changed while being enumerated: {current}"
            )

    visit(directory, PurePosixPath(), root_metadata)
    root_after = _require_directory(directory, description="artifact root")
    if not _same_snapshot(root_after, root_metadata):
        raise ArtifactVerificationError("artifact root changed identity during enumeration")
    return _ArtifactTree(
        root=_ArtifactEntry(directory, root_metadata),
        files=dict(files),
        directories=dict(directories),
        total_bytes=total_bytes,
    )


def _compare_artifact_trees(expected: _ArtifactTree, actual: _ArtifactTree) -> None:
    """Reject membership, metadata, link, or root changes between scans."""

    if expected.root.path != actual.root.path or not _same_snapshot(
        expected.root.metadata,
        actual.root.metadata,
    ):
        raise ArtifactVerificationError("artifact root changed during verification")
    if set(expected.files) != set(actual.files):
        missing = sorted(set(expected.files) - set(actual.files))
        extra = sorted(set(actual.files) - set(expected.files))
        raise ArtifactVerificationError(
            f"artifact file set changed during verification; missing={missing}, extra={extra}"
        )
    if set(expected.directories) != set(actual.directories):
        missing = sorted(set(expected.directories) - set(actual.directories))
        extra = sorted(set(actual.directories) - set(expected.directories))
        raise ArtifactVerificationError(
            "artifact directory set changed during verification; "
            f"missing={missing}, extra={extra}"
        )
    if expected.total_bytes != actual.total_bytes:
        raise ArtifactVerificationError("artifact byte total changed during verification")
    for relative, expected_entry in expected.directories.items():
        actual_entry = actual.directories[relative]
        if (
            expected_entry.path != actual_entry.path
            or not _same_snapshot(expected_entry.metadata, actual_entry.metadata)
        ):
            raise ArtifactVerificationError(
                f"artifact directory changed during verification: {relative}"
            )
    for relative, expected_entry in expected.files.items():
        actual_entry = actual.files[relative]
        if (
            expected_entry.path != actual_entry.path
            or not _same_snapshot(expected_entry.metadata, actual_entry.metadata)
        ):
            raise ArtifactVerificationError(
                f"artifact file changed during verification: {relative}"
            )


def _require_bound_root(directory: Path, expected: os.stat_result) -> os.stat_result:
    """Require the artifact path to keep naming the scanned root object.

    Publication itself changes directory timestamps.  A reliable filesystem
    identity is therefore used after control-file writes.  If the platform
    supplies no identity, the caller must perform a complete byte and
    membership verification before returning.
    """

    actual = _require_directory(directory, description="artifact root")
    expected_identity = _reliable_identity(expected)
    actual_identity = _reliable_identity(actual)
    if (expected_identity is None) != (actual_identity is None):
        raise ArtifactVerificationError("artifact root identity availability changed")
    if expected_identity is not None and expected_identity != actual_identity:
        raise ArtifactVerificationError("artifact root changed identity")
    return actual


def _required_directories(files: Mapping[str, object] | set[str]) -> frozenset[str]:
    paths = files if isinstance(files, set) else set(files)
    result: set[str] = set()
    for raw in paths:
        path = PurePosixPath(raw)
        for length in range(1, len(path.parts)):
            result.add(PurePosixPath(*path.parts[:length]).as_posix())
    return frozenset(result)


def _validate_relative_file_name(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or "\\" in value:
        raise ArtifactVerificationError("manifest file names must be safe relative POSIX paths")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or posix.as_posix() != value
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ArtifactVerificationError("manifest file names must be safe relative POSIX paths")
    if value in _ROOT_CONTROL_FILES or value == _MANIFEST_TEMP:
        raise ArtifactVerificationError("manifest must not list artifact control files")
    return value


def _validate_sha256(value: Any, *, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArtifactVerificationError(f"{description} must be a lowercase SHA-256")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactVerificationError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _parse_manifest(payload: bytes) -> ArtifactManifest:
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ArtifactVerificationError(f"non-finite JSON constant is forbidden: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactVerificationError("artifact manifest is not valid UTF-8 JSON") from exc
    if not isinstance(document, dict) or set(document) != {
        "artifact_id",
        "stage_name",
        "schema_version",
        "files",
        "metadata",
    }:
        raise ArtifactVerificationError("artifact manifest keys do not match its schema")
    artifact_id = _validate_sha256(document["artifact_id"], description="artifact id")
    stage_name = document["stage_name"]
    if not isinstance(stage_name, str) or not stage_name.strip():
        raise ArtifactVerificationError("artifact stage_name must be a non-empty string")
    schema_version = document["schema_version"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version <= 0:
        raise ArtifactVerificationError("artifact schema_version must be a positive integer")
    raw_files = document["files"]
    if not isinstance(raw_files, dict):
        raise ArtifactVerificationError("artifact manifest files must be an object")
    files: dict[str, str] = {}
    for raw_name, raw_digest in raw_files.items():
        name = _validate_relative_file_name(raw_name)
        files[name] = _validate_sha256(raw_digest, description=f"checksum for {name}")
    metadata = document["metadata"]
    if not isinstance(metadata, dict):
        raise ArtifactVerificationError("artifact manifest metadata must be an object")
    return ArtifactManifest(
        artifact_id=artifact_id,
        stage_name=stage_name,
        schema_version=schema_version,
        files=files,
        metadata=metadata,
    )


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _write_new_regular(path: Path, payload: bytes, *, description: str) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | int(getattr(os, "O_BINARY", 0))
        | int(getattr(os, "O_NOFOLLOW", 0))
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ArtifactVerificationError(f"cannot create {description}: {path}") from exc
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ArtifactVerificationError(f"short write while creating {description}")
            view = view[written:]
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or _is_reparse(metadata)
            or metadata.st_size != len(payload)
        ):
            raise ArtifactVerificationError(f"created {description} is not a regular file")
    finally:
        os.close(descriptor)
    _require_regular_file(path, description=description)


def publish_artifact(
    root: str | Path,
    *,
    stage_name: str,
    schema_version: int = 1,
    metadata: Mapping[str, Any] | None = None,
) -> ArtifactManifest:
    directory = _artifact_root(root)
    initial_root_metadata = _require_directory(directory, description="artifact root")
    if not isinstance(stage_name, str) or not stage_name.strip():
        raise ValueError("stage_name must be a non-empty string")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version <= 0:
        raise ValueError("schema_version must be a positive integer")
    success = directory / "_SUCCESS"
    if _lexists(success):
        _require_regular_file(success, description="artifact success marker")
        raise FileExistsError(f"published artifact is immutable: {directory}")
    temporary = directory / _MANIFEST_TEMP
    if _lexists(temporary):
        raise ArtifactVerificationError(f"stale artifact manifest temporary exists: {temporary}")

    tree = _scan_artifact(directory)
    if not _same_snapshot(tree.root.metadata, initial_root_metadata):
        raise ArtifactVerificationError(
            "artifact root changed before the initial publication scan"
        )
    if "_SUCCESS" in tree.files or "_SUCCESS" in tree.directories:
        raise FileExistsError(f"published artifact is immutable: {directory}")
    if _MANIFEST_TEMP in tree.files or _MANIFEST_TEMP in tree.directories:
        raise ArtifactVerificationError(
            f"stale artifact manifest temporary exists: {temporary}"
        )
    payload_entries = {
        relative: entry
        for relative, entry in tree.files.items()
        if relative not in _ROOT_CONTROL_FILES
    }
    required_directories = _required_directories(payload_entries)
    if set(tree.directories) != required_directories:
        extra = sorted(set(tree.directories) - required_directories)
        raise ArtifactVerificationError(
            f"artifact contains unbound or empty directories: {extra}"
        )
    files: dict[str, str] = {}
    for relative in sorted(payload_entries):
        entry = payload_entries[relative]
        digest, _size, _payload = _read_regular(
            entry.path,
            maximum_bytes=_MAX_ARTIFACT_FILE_BYTES,
            description=f"artifact file {relative}",
            collect=False,
            expected_metadata=entry.metadata,
        )
        files[relative] = digest

    # Close the scan/read race before publishing control files.  A second byte
    # pass is intentional: it is the strong fallback on filesystems that do not
    # expose reliable device/inode identities.
    prepublish_tree = _scan_artifact(directory)
    _compare_artifact_trees(tree, prepublish_tree)
    for relative, expected_digest in files.items():
        entry = prepublish_tree.files[relative]
        digest, _size, _payload = _read_regular(
            entry.path,
            maximum_bytes=_MAX_ARTIFACT_FILE_BYTES,
            description=f"artifact file {relative}",
            collect=False,
            expected_metadata=entry.metadata,
        )
        if digest != expected_digest:
            raise ArtifactVerificationError(
                f"artifact file changed during publication: {relative}"
            )
    final_payload_tree = _scan_artifact(directory)
    _compare_artifact_trees(prepublish_tree, final_payload_tree)
    safe_metadata = redact_secrets(dict(metadata or {}))
    semantic = {
        "stage_name": stage_name,
        "schema_version": schema_version,
        "files": files,
        "metadata": safe_metadata,
    }
    try:
        artifact_id = hashlib.sha256(_canonical_json(semantic).encode("utf-8")).hexdigest()
    except (TypeError, ValueError) as exc:
        raise ArtifactVerificationError("artifact metadata is not finite JSON data") from exc
    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        stage_name=stage_name,
        schema_version=schema_version,
        files=files,
        metadata=safe_metadata,
    )
    try:
        manifest_payload = (
            json.dumps(
                manifest.to_dict(),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArtifactVerificationError("artifact manifest is not finite JSON data") from exc
    if len(manifest_payload) > _MAX_MANIFEST_BYTES:
        raise ArtifactVerificationError("artifact manifest exceeds its safe size limit")
    _require_bound_root(directory, tree.root.metadata)
    _write_new_regular(temporary, manifest_payload, description="artifact manifest temporary")
    _require_bound_root(directory, tree.root.metadata)
    manifest_path = directory / "manifest.json"
    try:
        os.replace(temporary, manifest_path)
    except OSError as exc:
        raise ArtifactVerificationError("cannot publish artifact manifest atomically") from exc
    _require_regular_file(manifest_path, description="artifact manifest")
    _require_bound_root(directory, tree.root.metadata)
    _write_new_regular(
        success,
        (artifact_id + "\n").encode("ascii"),
        description="artifact success marker",
    )
    _require_bound_root(directory, tree.root.metadata)
    # In the zero-identity fallback, this complete verifier is what binds the
    # newly published bytes and final membership rather than silently trusting
    # timestamps alone.
    if verify_artifact(directory) != manifest:
        raise ArtifactVerificationError("published artifact failed final verification")
    _require_bound_root(directory, tree.root.metadata)
    return manifest


@dataclass(frozen=True)
class _VerifiedArtifactTree:
    manifest: ArtifactManifest
    file_digests: Mapping[str, str]


def _verify_scanned_artifact(
    tree: _ArtifactTree,
    *,
    allow_legacy_crlf_success: bool,
) -> _VerifiedArtifactTree:
    manifest_entry = tree.files.get("manifest.json")
    success_entry = tree.files.get("_SUCCESS")
    if manifest_entry is None or success_entry is None:
        raise ArtifactVerificationError("artifact is incomplete")
    manifest_digest, _manifest_size, manifest_payload = _read_regular(
        manifest_entry.path,
        maximum_bytes=_MAX_MANIFEST_BYTES,
        description="artifact manifest",
        collect=True,
        expected_metadata=manifest_entry.metadata,
    )
    assert manifest_payload is not None
    manifest = _parse_manifest(manifest_payload)
    success_digest, _success_size, success_payload = _read_regular(
        success_entry.path,
        maximum_bytes=(
            _MAX_SUCCESS_BYTES + 1
            if allow_legacy_crlf_success
            else _MAX_SUCCESS_BYTES
        ),
        description="artifact success marker",
        collect=True,
        expected_metadata=success_entry.metadata,
    )
    assert success_payload is not None
    expected_success = (manifest.artifact_id + "\n").encode("ascii")
    legacy_success = (manifest.artifact_id + "\r\n").encode("ascii")
    if success_payload != expected_success and not (
        allow_legacy_crlf_success and success_payload == legacy_success
    ):
        raise ArtifactVerificationError("_SUCCESS does not match manifest")

    actual_files = set(tree.files) - _ROOT_CONTROL_FILES
    if actual_files != set(manifest.files):
        missing = sorted(set(manifest.files) - actual_files)
        extra = sorted(actual_files - set(manifest.files))
        raise ArtifactVerificationError(
            f"artifact file set mismatch; missing={missing}, extra={extra}"
        )
    required_directories = _required_directories(set(manifest.files))
    actual_directories = set(tree.directories)
    if actual_directories != required_directories:
        missing = sorted(required_directories - actual_directories)
        extra = sorted(actual_directories - required_directories)
        raise ArtifactVerificationError(
            f"artifact directory set mismatch; missing={missing}, extra={extra}"
        )
    observed_digests = {
        "manifest.json": manifest_digest,
        "_SUCCESS": success_digest,
    }
    for relative, expected in manifest.files.items():
        entry = tree.files[relative]
        digest, _size, _payload = _read_regular(
            entry.path,
            maximum_bytes=_MAX_ARTIFACT_FILE_BYTES,
            description=f"artifact file {relative}",
            collect=False,
            expected_metadata=entry.metadata,
        )
        observed_digests[relative] = digest
        if digest != expected:
            raise ArtifactVerificationError(f"artifact file checksum mismatch: {relative}")
    semantic = {
        "stage_name": manifest.stage_name,
        "schema_version": manifest.schema_version,
        "files": manifest.files,
        "metadata": manifest.metadata,
    }
    try:
        expected_id = hashlib.sha256(_canonical_json(semantic).encode("utf-8")).hexdigest()
    except (TypeError, ValueError) as exc:
        raise ArtifactVerificationError("artifact metadata is not finite JSON data") from exc
    if expected_id != manifest.artifact_id:
        raise ArtifactVerificationError("artifact semantic hash mismatch")
    return _VerifiedArtifactTree(manifest, dict(observed_digests))


def verify_artifact(
    root: str | Path,
    *,
    allow_legacy_crlf_success: bool = False,
) -> ArtifactManifest:
    """Verify an immutable artifact, optionally accepting an exact legacy CRLF marker."""

    directory = _artifact_root(root)
    initial_tree = _scan_artifact(directory)
    initial = _verify_scanned_artifact(
        initial_tree,
        allow_legacy_crlf_success=allow_legacy_crlf_success,
    )

    # Re-scan and re-hash every byte.  This closes both the stale-path window
    # after the initial scan and the Windows zero-identity fallback without
    # accepting metadata-only equivalence.
    verification_tree = _scan_artifact(directory)
    _compare_artifact_trees(initial_tree, verification_tree)
    verification = _verify_scanned_artifact(
        verification_tree,
        allow_legacy_crlf_success=allow_legacy_crlf_success,
    )
    if (
        verification.manifest != initial.manifest
        or dict(verification.file_digests) != dict(initial.file_digests)
    ):
        raise ArtifactVerificationError("artifact bytes changed during verification")

    # The second read pass can itself race with an insertion or root swap.  A
    # final complete membership/metadata scan is therefore required immediately
    # before returning the verified manifest.
    final_tree = _scan_artifact(directory)
    _compare_artifact_trees(verification_tree, final_tree)
    return initial.manifest
