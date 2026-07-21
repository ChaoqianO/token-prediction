from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.parse import quote


DATASET_ID = "MLL-Lab/BAGEN"
RESOLVED_REVISION = "58189576e54b675fdd0e1d6c1c9f189c2992732f"
PUBLISHED_REVISION = "main"
PINNED_MANIFEST_ETAG = "8a7f701692d90bf17b719220431c4b02ba14e780"
B1_PREFIX = "origin/swebench-origin-gpt5.2instant/"

_MANIFEST_FIELDS = {
    "download_url",
    "extension",
    "path",
    "relative_path",
    "size_bytes",
    "top_level",
}
_SAFE_PATH = re.compile(r"[A-Za-z0-9._/-]+\Z")
_PUBLISHED_DOWNLOAD_ROOT = (
    f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{PUBLISHED_REVISION}/"
)


class ManifestSchemaError(ValueError):
    """Raised when the published BAGEN manifest does not match its contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_etag(path: Path) -> str:
    """Return the Git blob object ID used as the public manifest's ETag."""
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_posix_path(value: Any, *, field: str, line_number: int) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestSchemaError(f"line {line_number} has a non-string or empty {field}")
    if not _SAFE_PATH.fullmatch(value):
        raise ManifestSchemaError(f"line {line_number} has an unsafe {field}: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ManifestSchemaError(f"line {line_number} has a non-canonical {field}: {value!r}")
    return value


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ManifestSchemaError(f"duplicate JSON field: {key!r}")
        value[key] = item
    return value


def _benchmark(path: str) -> str:
    directory = path.split("/", 2)[1].lower()
    if directory.startswith("newwarehouse-"):
        return "warehouse"
    if directory.startswith("searchr1-"):
        return "search-r1"
    if directory.startswith("sokoban-"):
        return "sokoban"
    if directory.startswith(("swebench-", "swe-banch-", "swebanch-")):
        return "swe-bench"
    raise ManifestSchemaError(f"unknown benchmark directory: {directory!r}")


def _family(path: str) -> str:
    lowered = path.lower()
    families = (
        (("claude-opus4.7", "claude-opus-4.7", "newwarehouse-opus4.7"), "claude-opus4.7"),
        (("claude-opus-4-6", "opus4.6"), "claude-opus4.6"),
        (
            ("claude-sonnet4.6", "claude-sonnet-4.6", "newwarehouse-sonnet4.6"),
            "claude-sonnet4.6",
        ),
        (("gpt5.2instant", "gpt5.2-instant"), "gpt5.2instant"),
        (("openai-5-2-codex",), "openai-5.2-codex"),
        (("gemini3.1", "gemini-3-1", "gemini-3.1"), "gemini3.1"),
        (("qwen3-235b", "qwen-qwen3-235b"), "qwen3-235b"),
        (("sera7b", "sera8b"), "sera"),
    )
    for aliases, family in families:
        if any(alias in lowered for alias in aliases):
            return family
    if "newwarehouse-gpt5.2" in lowered:
        return "gpt5.2"
    if "newwarehouse-qwen" in lowered:
        return "qwen3-235b"
    raise ManifestSchemaError(f"unknown model family in path: {path!r}")


def _breakdown(
    items: list[dict[str, Any]], classifier: Callable[[dict[str, Any]], str]
) -> dict[str, dict[str, int]]:
    result: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"file_count": 0, "total_bytes": 0, "traj_json_count": 0}
    )
    for item in items:
        bucket = result[classifier(item)]
        bucket["file_count"] += 1
        bucket["total_bytes"] += item["size_bytes"]
        bucket["traj_json_count"] += int(item["path"].endswith(".traj.json"))
    return dict(sorted(result.items()))


def read_manifest(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            try:
                value = json.loads(raw_line, object_pairs_hook=_object_without_duplicate_keys)
            except json.JSONDecodeError as exc:
                raise ManifestSchemaError(f"line {line_number} is not valid JSON: {exc}") from exc
            except ManifestSchemaError as exc:
                raise ManifestSchemaError(f"line {line_number} has {exc}") from exc
            if not isinstance(value, dict):
                raise ManifestSchemaError(f"line {line_number} is not an object")
            fields = set(value)
            if fields != _MANIFEST_FIELDS:
                missing = _MANIFEST_FIELDS - fields
                unexpected = fields - _MANIFEST_FIELDS
                details: list[str] = []
                if missing:
                    details.append(f"missing {', '.join(sorted(missing))}")
                if unexpected:
                    details.append(f"unexpected {', '.join(sorted(unexpected))}")
                raise ManifestSchemaError(
                    f"line {line_number} has invalid fields: {'; '.join(details)}"
                )
            item_path = _canonical_posix_path(value["path"], field="path", line_number=line_number)
            if item_path in seen_paths:
                raise ManifestSchemaError(f"duplicate manifest path: {item_path}")
            top_level = value["top_level"]
            if not isinstance(top_level, str) or top_level not in {"origin", "estimation"}:
                raise ManifestSchemaError(f"line {line_number} has an invalid top_level")
            path_prefix = f"{top_level}/"
            if not item_path.startswith(path_prefix):
                raise ManifestSchemaError(f"line {line_number} has inconsistent top_level")

            expected_relative_path = item_path[len(path_prefix) :]
            relative_path = _canonical_posix_path(
                value["relative_path"], field="relative_path", line_number=line_number
            )
            if relative_path != expected_relative_path:
                raise ManifestSchemaError(f"line {line_number} has inconsistent relative_path")

            extension = value["extension"]
            expected_extension = PurePosixPath(item_path).suffix
            if not isinstance(extension, str) or extension != expected_extension:
                raise ManifestSchemaError(f"line {line_number} has inconsistent extension")

            size_bytes = value["size_bytes"]
            if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
                raise ManifestSchemaError(f"line {line_number} has an invalid size")

            download_url = value["download_url"]
            expected_url = _PUBLISHED_DOWNLOAD_ROOT + quote(item_path, safe="/")
            if not isinstance(download_url, str) or download_url != expected_url:
                raise ManifestSchemaError(f"line {line_number} has an unexpected download URL")

            seen_paths.add(item_path)
            items.append(value)
    if not items:
        raise ManifestSchemaError("manifest is empty")
    return items


def build_summary(path: Path) -> dict[str, Any]:
    items = read_manifest(path)
    largest = max(items, key=lambda item: (item["size_bytes"], item["path"]))
    origin_items = [item for item in items if item["top_level"] == "origin"]
    if not origin_items:
        raise ManifestSchemaError("manifest does not contain origin files")
    largest_origin = max(origin_items, key=lambda item: (item["size_bytes"], item["path"]))
    b1_items = [item for item in items if item["path"].startswith(B1_PREFIX)]
    if not b1_items:
        raise ManifestSchemaError(f"manifest does not contain the B1 prefix: {B1_PREFIX}")

    def file_summary(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "path": item["path"],
            "size_bytes": item["size_bytes"],
            "download_url": item["download_url"],
        }

    return {
        "source_id": "MLL-Lab/BAGEN@main",
        "source_url": "https://huggingface.co/datasets/MLL-Lab/BAGEN",
        "manifest_url": (
            "https://huggingface.co/datasets/MLL-Lab/BAGEN/resolve/main/manifest.jsonl"
        ),
        "manifest_file": path.name,
        "manifest_bytes": path.stat().st_size,
        "manifest_sha256": _sha256(path),
        "manifest_etag": manifest_etag(path),
        "resolved_revision": RESOLVED_REVISION,
        "file_count": len(items),
        "total_bytes": sum(item["size_bytes"] for item in items),
        "top_level": _breakdown(items, lambda item: str(item["top_level"])),
        "benchmark": _breakdown(items, lambda item: _benchmark(str(item["path"]))),
        "family": _breakdown(items, lambda item: _family(str(item["path"]))),
        "extension": _breakdown(items, lambda item: str(item["extension"])),
        "traj_json_count": sum(item["path"].endswith(".traj.json") for item in items),
        "traj_json_bytes": sum(
            item["size_bytes"] for item in items if item["path"].endswith(".traj.json")
        ),
        "largest_file": file_summary(largest),
        "largest_origin_file": file_summary(largest_origin),
        "b1_slice": {
            "prefix": B1_PREFIX,
            "file_count": len(b1_items),
            "total_bytes": sum(item["size_bytes"] for item in b1_items),
            "traj_json_count": sum(item["path"].endswith(".traj.json") for item in b1_items),
            "traj_json_bytes": sum(
                item["size_bytes"] for item in b1_items if item["path"].endswith(".traj.json")
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the published BAGEN manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    summary = build_summary(args.manifest.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
