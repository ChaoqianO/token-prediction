from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


MAX_TEXT_BYTES = 16 * 1024 * 1024
FORBIDDEN_BASENAMES = frozenset(
    {
        ".env",
        "auth.json",
        "auth-profiles.json",
        "config.toml",
        "credentials.json",
        "secrets.json",
    }
)
FORBIDDEN_SUFFIXES = (".key", ".pem", ".p12", ".pfx")
PLACEHOLDER_MARKERS = (
    b"CHANGEME",
    b"DO_NOT_LEAK",
    b"DUMMY",
    b"EXAMPLE",
    b"FAKE",
    b"PLACEHOLDER",
    b"REDACTED",
    b"TEST",
)
SECRET_PATTERNS = (
    (
        "openai_api_key",
        re.compile(rb"\bsk-(?:(?:proj|svcacct)-)?[A-Za-z0-9_-]{20,}\b"),
    ),
    ("anthropic_api_key", re.compile(rb"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("huggingface_token", re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b")),
    (
        "github_token",
        re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    ),
    ("slack_token", re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
    ("aws_access_key", re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("google_api_key", re.compile(rb"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("stripe_live_key", re.compile(rb"\b[rs]k_live_[0-9A-Za-z]{20,}\b")),
    (
        "private_key",
        re.compile(rb"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    ),
    (
        "credentialed_url",
        re.compile(rb"\bhttps?://[^\s/:@]{1,64}:[^\s/@]{8,128}@[^\s/]+"),
    ),
    (
        "jwt",
        re.compile(rb"\beyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\b"),
    ),
)
GENERIC_ASSIGNMENT = re.compile(
    rb"(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password|secret)"
    rb"\s*[:=]\s*[\"']([^\"'\r\n]{12,})[\"']"
)


class SecretScanError(RuntimeError):
    """The repository file inventory could not be scanned safely."""


@dataclass(frozen=True, order=True)
class Finding:
    path: str
    line: int
    rule: str


def _git_executable() -> str:
    discovered = shutil.which("git")
    if discovered:
        return discovered
    if os.name == "nt":
        candidate = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git/cmd/git.exe"
        if candidate.is_file():
            return str(candidate)
    raise SecretScanError("git is required when no explicit paths are supplied")


def repository_files(root: str | Path) -> tuple[Path, ...]:
    repository = Path(root).resolve()
    command = [
        _git_executable(),
        "-C",
        str(repository),
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SecretScanError("git could not enumerate repository files") from exc
    paths: list[Path] = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        try:
            relative = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise SecretScanError("git returned a non-UTF-8 path") from exc
        candidate = repository / relative
        resolved = candidate.resolve()
        try:
            resolved.relative_to(repository)
        except ValueError as exc:
            raise SecretScanError("git returned a path outside the repository") from exc
        paths.append(candidate)
    return tuple(sorted(set(paths), key=lambda path: path.as_posix()))


def _expand_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    expanded: set[Path] = set()
    for path in paths:
        if path.is_dir() and not path.is_symlink():
            expanded.update(candidate for candidate in path.rglob("*") if candidate.is_file())
        else:
            expanded.add(path)
    return tuple(sorted(expanded, key=lambda path: path.as_posix()))


def _display_path(path: Path, root: Path | None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.name


def _is_placeholder(value: bytes) -> bool:
    upper = value.upper()
    return any(marker in upper for marker in PLACEHOLDER_MARKERS)


def _scan_payload(payload: bytes, *, display_path: str) -> list[Finding]:
    findings: list[Finding] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        for rule, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append(Finding(display_path, line_number, rule))
        for match in GENERIC_ASSIGNMENT.finditer(line):
            if not _is_placeholder(match.group(1)):
                findings.append(Finding(display_path, line_number, "credential_assignment"))
    return findings


def scan_paths(
    paths: Iterable[str | Path],
    *,
    root: str | Path | None = None,
) -> tuple[Finding, ...]:
    repository = Path(root).resolve() if root is not None else None
    findings: set[Finding] = set()
    expanded = _expand_paths(Path(path) for path in paths)
    for path in expanded:
        display = _display_path(path, repository)
        basename = path.name.casefold()
        if basename in FORBIDDEN_BASENAMES and basename != ".env.example":
            findings.add(Finding(display, 0, "forbidden_secret_filename"))
        if basename.endswith(FORBIDDEN_SUFFIXES):
            findings.add(Finding(display, 0, "forbidden_secret_filename"))
        if path.is_symlink():
            findings.add(Finding(display, 0, "unscanned_symlink"))
            continue
        if not path.is_file():
            findings.add(Finding(display, 0, "unreadable_path"))
            continue
        try:
            size = path.stat().st_size
            with path.open("rb") as handle:
                sample = handle.read(min(size, 8192))
                if b"\0" in sample:
                    continue
                if size > MAX_TEXT_BYTES:
                    findings.add(Finding(display, 0, "unscanned_large_text"))
                    continue
                payload = sample + handle.read()
        except OSError:
            findings.add(Finding(display, 0, "unreadable_path"))
            continue
        findings.update(_scan_payload(payload, display_path=display))
    return tuple(sorted(findings))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan repository-intended files for credential signatures without printing values."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="explicit files/directories; defaults to tracked and non-ignored repository files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        paths = tuple(args.paths) if args.paths else repository_files(root)
        findings = scan_paths(paths, root=root)
    except SecretScanError as exc:
        print(f"secret scan failed: {exc}", file=sys.stderr)
        return 2
    report = {
        "files_scanned": len(_expand_paths(paths)),
        "findings": [asdict(finding) for finding in findings],
    }
    print(json.dumps(report, sort_keys=True))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
