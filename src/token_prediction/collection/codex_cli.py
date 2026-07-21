from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable, Sequence

from token_prediction.recording.redaction import redact_text


class CodexAuthState(StrEnum):
    AUTHENTICATED = "authenticated"
    UNAUTHENTICATED = "unauthenticated"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


@dataclass(frozen=True)
class CodexAuthStatus:
    state: CodexAuthState
    executable: str | None
    message: str = ""


class CodexCLIError(RuntimeError):
    pass


RunCommand = Callable[..., subprocess.CompletedProcess[str]]


class CodexCLI:
    """Small boundary around the official Codex executable.

    It deliberately exposes no token-bearing credential API and never opens
    ``auth.json``.  The official CLI owns sign-in, persistence, and refresh.
    Live raw collection will be added here only after its JSONL format is
    normalized by a versioned reader.
    """

    def __init__(
        self,
        executable: str | Path | None = None,
        *,
        run_command: RunCommand = subprocess.run,
    ) -> None:
        self._explicit_executable = str(executable) if executable else None
        self._run_command = run_command

    def resolve_executable(self) -> str | None:
        if self._explicit_executable:
            path = Path(self._explicit_executable)
            if path.exists():
                return str(path)
            return shutil.which(self._explicit_executable)
        return shutil.which("codex") or shutil.which("codex.exe")

    def _args(self, *parts: str) -> list[str]:
        executable = self.resolve_executable()
        if not executable:
            raise CodexCLIError("Codex CLI is not available on PATH")
        return [executable, *parts]

    def _captured(self, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return self._run_command(
            list(args),
            shell=False,
            check=False,
            capture_output=True,
            text=True,
        )

    def auth_status(self) -> CodexAuthStatus:
        executable = self.resolve_executable()
        if not executable:
            return CodexAuthStatus(CodexAuthState.UNAVAILABLE, None, "Codex CLI not found")
        try:
            result = self._captured([executable, "login", "status"])
        except OSError as exc:
            return CodexAuthStatus(
                CodexAuthState.ERROR,
                executable,
                redact_text(f"{type(exc).__name__}: {exc}"),
            )
        message = redact_text(
            "\n".join(part for part in (result.stdout, result.stderr) if part)
        ).strip()
        if result.returncode == 0:
            return CodexAuthStatus(CodexAuthState.AUTHENTICATED, executable, message[:500])
        lowered = message.lower()
        state = (
            CodexAuthState.UNAUTHENTICATED
            if any(
                token in lowered
                for token in ("not logged", "login required", "unauthenticated")
            )
            else CodexAuthState.ERROR
        )
        return CodexAuthStatus(state, executable, message[:500])

    def login(self) -> CodexAuthStatus:
        result = self._run_command(self._args("login"), shell=False, check=False)
        if result.returncode != 0:
            raise CodexCLIError(f"codex login exited with status {result.returncode}")
        return self.auth_status()

    def logout(self) -> CodexAuthStatus:
        result = self._run_command(self._args("logout"), shell=False, check=False)
        if result.returncode != 0:
            raise CodexCLIError(f"codex logout exited with status {result.returncode}")
        return self.auth_status()
