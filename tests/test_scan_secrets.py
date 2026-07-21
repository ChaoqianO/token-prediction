from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts.scan_secrets import main, scan_paths


class SecretScanTests(unittest.TestCase):
    def test_clean_files_and_explicit_placeholders_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean = root / "clean.py"
            clean.write_text(
                'secret = "SECRET_DO_NOT_LEAK"\nmessage = "ordinary text"\n',
                encoding="utf-8",
            )

            self.assertEqual(scan_paths((clean,), root=root), ())

    def test_detects_provider_key_without_echoing_credential(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "leak.txt"
            credential = "sk-" + "A1b2" * 8
            path.write_text(f"value={credential}\n", encoding="utf-8")
            output = io.StringIO()

            with redirect_stdout(output):
                status = main((str(path),))

            report = json.loads(output.getvalue())
            self.assertEqual(status, 1)
            self.assertEqual(report["findings"][0]["rule"], "openai_api_key")
            self.assertNotIn(credential, output.getvalue())

    def test_forbidden_filename_and_private_key_marker_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "auth.json"
            marker = "-----BEGIN " + "PRIVATE KEY-----"
            path.write_text(marker + "\n", encoding="utf-8")

            findings = scan_paths((path,), root=root)

            self.assertEqual(
                {finding.rule for finding in findings},
                {"forbidden_secret_filename", "private_key"},
            )
            self.assertTrue(all(finding.path == "auth.json" for finding in findings))

    def test_binary_files_are_not_interpreted_as_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.bin"
            path.write_bytes(b"\x00sk-" + b"A" * 40)

            self.assertEqual(scan_paths((path,)), ())


if __name__ == "__main__":
    unittest.main()
