from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

if __package__:
    from .audit_bagen_manifest import (
        B1_PREFIX,
        DATASET_ID,
        PINNED_MANIFEST_ETAG,
        RESOLVED_REVISION,
        read_manifest,
    )
else:
    from audit_bagen_manifest import (  # type: ignore[no-redef]
        B1_PREFIX,
        DATASET_ID,
        PINNED_MANIFEST_ETAG,
        RESOLVED_REVISION,
        read_manifest,
    )


MANIFEST_NAME = "manifest.jsonl"
MANIFEST_SELECTION = "manifest"
MANIFEST_BYTES = 228_164
WORKSPACE = Path(__file__).resolve().parents[1] / "workspace"
BAGEN_ROOT = WORKSPACE / "external" / "bagen"
CHUNK_BYTES = 1024 * 1024
PUBLIC_USER_AGENT = "token-prediction-bagen-public-downloader/1"
B3_PREFIXES = (
    "origin/swebench-origin-claude-opus4.7/",
    "origin/swebench-origin-claude-sonnet4.6/",
    "origin/swebench-origin-qwen3-235b/",
    "origin/swebench-origin-gemini3.1/",
)
ALLOWED_SELECTIONS = (MANIFEST_SELECTION, B1_PREFIX, *B3_PREFIXES)


class DownloadVerificationError(RuntimeError):
    """Raised when a public BAGEN download does not match its pinned manifest."""


def configure_public_hf_environment(workspace: Path = WORKSPACE) -> Path:
    """Disable implicit Hub credentials without inspecting their existing values."""
    hf_home = (workspace / ".hf-public").resolve()
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_TOKEN_PATH"] = str(hf_home / "disabled-token")
    os.environ["HF_TOKEN"] = ""
    os.environ["HUGGING_FACE_HUB_TOKEN"] = ""
    return hf_home


def pinned_url(path: str) -> str:
    return (
        f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
        f"{RESOLVED_REVISION}/{quote(path, safe='/')}"
    )


def _git_blob_etag(path: Path) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_pinned_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DownloadVerificationError(f"pinned manifest is missing: {path}")
    actual_size = path.stat().st_size
    if actual_size != MANIFEST_BYTES:
        raise DownloadVerificationError(
            f"manifest size mismatch: expected {MANIFEST_BYTES}, got {actual_size}"
        )
    actual_etag = _git_blob_etag(path)
    if actual_etag != PINNED_MANIFEST_ETAG:
        raise DownloadVerificationError(
            f"manifest ETag mismatch: expected {PINNED_MANIFEST_ETAG}, got {actual_etag}"
        )
    return read_manifest(path)


def _safe_destination(root: Path, manifest_path: str) -> Path:
    pure_path = PurePosixPath(manifest_path)
    if (
        not manifest_path
        or pure_path.is_absolute()
        or pure_path.as_posix() != manifest_path
        or any(part in {"", ".", ".."} for part in pure_path.parts)
    ):
        raise DownloadVerificationError(f"manifest path is not canonical: {manifest_path!r}")
    parts = pure_path.parts
    destination = root.joinpath(*parts)
    resolved_root = root.resolve()
    try:
        destination.resolve(strict=False).relative_to(resolved_root)
    except ValueError as exc:
        raise DownloadVerificationError(
            f"manifest path escapes the BAGEN root: {manifest_path!r}"
        ) from exc
    return destination


def _download_atomic(url: str, destination: Path, expected_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        url,
        headers={"Accept-Encoding": "identity", "User-Agent": PUBLIC_USER_AGENT},
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".part",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            with urlopen(request, timeout=60) as response:  # noqa: S310 - fixed HTTPS host
                total = 0
                while chunk := response.read(CHUNK_BYTES):
                    output.write(chunk)
                    total += len(chunk)
        if total != expected_size:
            raise DownloadVerificationError(
                f"download size mismatch for {destination}: expected {expected_size}, got {total}"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _download_manifest(destination: Path) -> tuple[int, int]:
    reused = 0
    if destination.is_file():
        try:
            _verify_pinned_manifest(destination)
            reused = 1
        except DownloadVerificationError:
            pass
    if not reused:
        _download_atomic(pinned_url(MANIFEST_NAME), destination, MANIFEST_BYTES)
    _verify_pinned_manifest(destination)
    return 1 - reused, reused


def _b1_items(manifest_path: Path) -> list[dict[str, Any]]:
    items = _verify_pinned_manifest(manifest_path)
    selected = sorted(
        (item for item in items if item["path"].startswith(B1_PREFIX)),
        key=lambda item: item["path"],
    )
    if not selected:
        raise DownloadVerificationError(
            f"pinned manifest does not contain the B1 prefix: {B1_PREFIX}"
        )
    return selected


def _selection_items(manifest_path: Path, selection: str) -> list[dict[str, Any]]:
    if selection == B1_PREFIX:
        return _b1_items(manifest_path)
    if selection not in B3_PREFIXES:
        raise DownloadVerificationError(f"unsupported BAGEN selection: {selection!r}")
    items = _verify_pinned_manifest(manifest_path)
    selected = sorted(
        (
            item
            for item in items
            if item["path"].startswith(selection)
            and item["path"].endswith(".traj.json")
        ),
        key=lambda item: item["path"],
    )
    if not selected:
        raise DownloadVerificationError(
            f"pinned manifest contains no individual trajectories under {selection!r}"
        )
    return selected


def _download_b1(root: Path, items: list[dict[str, Any]]) -> tuple[int, int]:
    downloaded = 0
    reused = 0
    for item in items:
        destination = _safe_destination(root, item["path"])
        if destination.is_file() and destination.stat().st_size == item["size_bytes"]:
            reused += 1
            continue
        _download_atomic(pinned_url(item["path"]), destination, item["size_bytes"])
        downloaded += 1
    verify_b1(root, items)
    return downloaded, reused


def _download_selection(
    root: Path,
    selection: str,
    items: list[dict[str, Any]],
) -> tuple[int, int]:
    downloaded = 0
    reused = 0
    for item in items:
        destination = _safe_destination(root, item["path"])
        if destination.is_file() and destination.stat().st_size == item["size_bytes"]:
            reused += 1
            continue
        _download_atomic(pinned_url(item["path"]), destination, item["size_bytes"])
        downloaded += 1
    verify_selection(root, selection, items)
    return downloaded, reused


def _visible_files(root: Path) -> dict[str, int]:
    if not root.exists():
        return {}
    files: dict[str, int] = {}
    for directory, directory_names, file_names in os.walk(root):
        directory_names[:] = sorted(name for name in directory_names if name != ".cache")
        for file_name in sorted(file_names):
            path = Path(directory, file_name)
            relative = path.relative_to(root)
            if ".cache" not in relative.parts:
                files[relative.as_posix()] = path.stat().st_size
    return files


def verify_b1(root: Path, items: list[dict[str, Any]]) -> None:
    expected = {item["path"]: item["size_bytes"] for item in items}
    b1_root = _safe_destination(root, B1_PREFIX.rstrip("/"))
    actual_local = _visible_files(b1_root)
    actual = {f"{B1_PREFIX}{path}": size for path, size in actual_local.items()}

    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    mismatched = sorted(
        path for path in expected.keys() & actual.keys() if expected[path] != actual[path]
    )
    if missing or unexpected or mismatched:
        details = {
            "missing": missing,
            "size_mismatches": [
                {
                    "actual": actual[path],
                    "expected": expected[path],
                    "path": path,
                }
                for path in mismatched
            ],
            "unexpected": unexpected,
        }
        raise DownloadVerificationError(
            "B1 path-set/size verification failed: "
            + json.dumps(details, sort_keys=True, separators=(",", ":"))
        )


def verify_selection(
    root: Path,
    selection: str,
    items: list[dict[str, Any]],
) -> None:
    expected = {item["path"]: item["size_bytes"] for item in items}
    selection_root = _safe_destination(root, selection.rstrip("/"))
    actual_local = _visible_files(selection_root)
    actual = {
        f"{selection}{path}": size
        for path, size in actual_local.items()
        if path.endswith(".traj.json")
    }
    missing = sorted(expected.keys() - actual.keys())
    mismatched = sorted(
        path for path in expected.keys() & actual.keys() if expected[path] != actual[path]
    )
    if missing or mismatched:
        details = {
            "missing": missing,
            "size_mismatches": [
                {"actual": actual[path], "expected": expected[path], "path": path}
                for path in mismatched
            ],
        }
        raise DownloadVerificationError(
            "trajectory selection verification failed: "
            + json.dumps(details, sort_keys=True, separators=(",", ":"))
        )


def _plan(selection: str, items: list[dict[str, Any]] | None) -> dict[str, Any]:
    if selection == MANIFEST_SELECTION:
        file_count = 1
        total_bytes = MANIFEST_BYTES
        destination = BAGEN_ROOT / MANIFEST_NAME
    else:
        assert items is not None
        file_count = len(items)
        total_bytes = sum(item["size_bytes"] for item in items)
        destination = _safe_destination(BAGEN_ROOT, selection.rstrip("/"))
    return {
        "destination": str(destination),
        "file_count": file_count,
        "hf_home": str((WORKSPACE / ".hf-public").resolve()),
        "resolved_revision": RESOLVED_REVISION,
        "selection": selection,
        "total_bytes": total_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download an allowlisted slice from the pinned public BAGEN manifest.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "selection",
        nargs="?",
        choices=ALLOWED_SELECTIONS,
        default=MANIFEST_SELECTION,
        help="allowed values: " + ", ".join(repr(value) for value in ALLOWED_SELECTIONS),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform downloads; without this flag the command only prints a plan",
    )
    args = parser.parse_args()

    configure_public_hf_environment()
    manifest_path = BAGEN_ROOT / MANIFEST_NAME
    items = (
        _selection_items(manifest_path, args.selection)
        if args.selection != MANIFEST_SELECTION
        else None
    )
    result = {"apply": args.apply, **_plan(args.selection, items)}

    if args.apply:
        if args.selection == MANIFEST_SELECTION:
            downloaded, reused = _download_manifest(manifest_path)
        else:
            assert items is not None
            if args.selection == B1_PREFIX:
                downloaded, reused = _download_b1(BAGEN_ROOT, items)
            else:
                downloaded, reused = _download_selection(
                    BAGEN_ROOT, args.selection, items
                )
        result.update(
            {
                "downloaded_file_count": downloaded,
                "reused_file_count": reused,
                "verified": True,
            }
        )

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
