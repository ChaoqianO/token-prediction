from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


DATASET_ID = "loong0814/openhands_trajectories"
RESOLVED_REVISION = "fa9cbb063f770df596da95af24f7af3b8f595778"
ARCHIVE_NAME = "gpt_5.2_4runs.tar.gz"
EXPECTED_BYTES = 2_908_192_516
EXPECTED_SHA256 = "993abcb55aae423f9067d5e6c8e1aeaccf83b9ce31474a215982686527934214"
XET_ETAG = "5824153171526bdfb245b74fca532407cf68add02079b4fa0f7c1cf47ea1c1c8"
WORKSPACE = Path(__file__).resolve().parents[1] / "workspace"
DESTINATION = WORKSPACE / "external" / "spend_your_money" / ARCHIVE_NAME
CHUNK_BYTES = 8 * 1024 * 1024
PROGRESS_BYTES = 256 * 1024 * 1024
PUBLIC_USER_AGENT = "token-prediction-spend-public-downloader/1"


class DownloadVerificationError(RuntimeError):
    """Raised when the pinned public Spend archive does not match its LFS object."""


def configure_public_hf_environment() -> None:
    """Disable implicit Hub credentials without inspecting their values."""
    hf_home = (WORKSPACE / ".hf-public").resolve()
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_TOKEN_PATH"] = str(hf_home / "disabled-token")
    os.environ["HF_TOKEN"] = ""
    os.environ["HUGGING_FACE_HUB_TOKEN"] = ""


def pinned_url() -> str:
    filename = quote(ARCHIVE_NAME, safe="")
    return (
        f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
        f"{RESOLVED_REVISION}/{filename}"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path) -> None:
    if not path.is_file():
        raise DownloadVerificationError(f"archive is missing: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != EXPECTED_BYTES:
        raise DownloadVerificationError(
            f"archive size mismatch: expected {EXPECTED_BYTES}, got {actual_bytes}"
        )
    actual_sha256 = sha256(path)
    if actual_sha256 != EXPECTED_SHA256:
        raise DownloadVerificationError(
            f"archive SHA-256 mismatch: expected {EXPECTED_SHA256}, got {actual_sha256}"
        )


def download(path: Path) -> str:
    if path.is_file() and path.stat().st_size == EXPECTED_BYTES:
        verify(path)
        return "reused"
    if path.exists():
        raise DownloadVerificationError("existing archive has an unexpected size")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(f".{path.name}.part")
    offset = partial_path.stat().st_size if partial_path.is_file() else 0
    if offset > EXPECTED_BYTES:
        raise DownloadVerificationError("partial archive is larger than the pinned object")

    digest = hashlib.sha256()
    if offset:
        with partial_path.open("rb") as existing:
            for chunk in iter(lambda: existing.read(CHUNK_BYTES), b""):
                digest.update(chunk)
    if offset == EXPECTED_BYTES:
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != EXPECTED_SHA256:
            raise DownloadVerificationError(
                f"partial SHA-256 mismatch: expected {EXPECTED_SHA256}, got {actual_sha256}"
            )
        os.replace(partial_path, path)
        return "resumed"
    headers = {"Accept-Encoding": "identity", "User-Agent": PUBLIC_USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(pinned_url(), headers=headers)
    total = offset
    next_progress = ((total // PROGRESS_BYTES) + 1) * PROGRESS_BYTES
    mode = "ab" if offset else "wb"
    with urlopen(request, timeout=120) as response:  # noqa: S310 - fixed HTTPS host
        status = getattr(response, "status", None)
        if offset and status != 206:
            raise DownloadVerificationError(
                f"server did not honor resume range at byte {offset}: HTTP {status}"
            )
        with partial_path.open(mode) as output:
            while chunk := response.read(CHUNK_BYTES):
                output.write(chunk)
                digest.update(chunk)
                total += len(chunk)
                if total >= next_progress:
                    print(
                        json.dumps(
                            {
                                "downloaded_bytes": total,
                                "expected_bytes": EXPECTED_BYTES,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    next_progress += PROGRESS_BYTES
    if total != EXPECTED_BYTES:
        raise DownloadVerificationError(
            f"download size mismatch: expected {EXPECTED_BYTES}, got {total}"
        )
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != EXPECTED_SHA256:
        raise DownloadVerificationError(
            f"download SHA-256 mismatch: expected {EXPECTED_SHA256}, got {actual_sha256}"
        )
    os.replace(partial_path, path)
    return "resumed" if offset else "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the one pinned Spend GPT-5.2 full archive."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the download; without this flag only print the pinned plan",
    )
    args = parser.parse_args()
    configure_public_hf_environment()
    result: dict[str, object] = {
        "apply": args.apply,
        "dataset_id": DATASET_ID,
        "resolved_revision": RESOLVED_REVISION,
        "filename": ARCHIVE_NAME,
        "destination": str(DESTINATION.resolve()),
        "expected_bytes": EXPECTED_BYTES,
        "expected_sha256": EXPECTED_SHA256,
        "xet_etag": XET_ETAG,
    }
    if args.apply:
        result["result"] = download(DESTINATION)
        result["verified"] = True
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
