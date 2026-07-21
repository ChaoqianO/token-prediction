from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from token_prediction.collection import CodexAuthState, CodexCLI


class CodexAuthTests(unittest.TestCase):
    def test_login_delegates_to_cli_without_shell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "codex.exe"
            executable.write_bytes(b"")
            runner = Mock(
                side_effect=[
                    subprocess.CompletedProcess([], 0, "", ""),
                    subprocess.CompletedProcess([], 0, "Logged in", ""),
                ]
            )
            client = CodexCLI(executable, run_command=runner)
            status = client.login()
            self.assertEqual(status.state, CodexAuthState.AUTHENTICATED)
            login_call = runner.call_args_list[0]
            self.assertEqual(login_call.args[0], [str(executable), "login"])
            self.assertFalse(login_call.kwargs["shell"])

    def test_status_does_not_read_auth_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "codex.exe"
            executable.write_bytes(b"")
            runner = Mock(return_value=subprocess.CompletedProcess([], 0, "Logged in", ""))
            client = CodexCLI(executable, run_command=runner)
            with patch.object(Path, "read_text", side_effect=AssertionError("credential read")):
                status = client.auth_status()
            self.assertEqual(status.state, CodexAuthState.AUTHENTICATED)


if __name__ == "__main__":
    unittest.main()
