from __future__ import annotations

import base64
import csv
import hashlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.check_distribution import DistributionCheckError, inspect_wheel


DIST_INFO = "token_prediction-0.1.0.dist-info"


def _record_hash(payload: bytes) -> str:
    digest = hashlib.sha256(payload).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _wheel_members() -> dict[str, bytes]:
    return {
        "token_prediction/__init__.py": b'__version__ = "0.1.0"\n',
        "token_prediction/cli.py": b"def main():\n    return 0\n",
        f"{DIST_INFO}/METADATA": (
            b"Metadata-Version: 2.3\n"
            b"Name: token-prediction\n"
            b"Version: 0.1.0\n"
            b"Requires-Python: >=3.11\n\n"
        ),
        f"{DIST_INFO}/WHEEL": (
            b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
        f"{DIST_INFO}/entry_points.txt": (
            b"[console_scripts]\ntp = token_prediction.cli:main\n"
        ),
    }


def _write_wheel(
    path: Path,
    *,
    extra: dict[str, bytes] | None = None,
    corrupt_record: bool = False,
) -> None:
    members = _wheel_members()
    members.update(extra or {})
    record_name = f"{DIST_INFO}/RECORD"
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    for name, payload in sorted(members.items()):
        digest = "sha256=invalid" if corrupt_record and name.endswith("cli.py") else _record_hash(payload)
        writer.writerow((name, digest, len(payload)))
    writer.writerow((record_name, "", ""))
    members[record_name] = stream.getvalue().encode("utf-8")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


class DistributionCheckTests(unittest.TestCase):
    def test_valid_wheel_has_closed_record_and_expected_import_surface(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            wheel = Path(temporary) / "token_prediction-0.1.0-py3-none-any.whl"
            _write_wheel(wheel)

            report = inspect_wheel(wheel)

            self.assertEqual(report.distribution, "token-prediction")
            self.assertEqual(report.version, "0.1.0")
            self.assertEqual(report.requires_python, ">=3.11")
            self.assertEqual(report.file_count, report.record_entries)
            self.assertEqual(report.package_file_count, 2)

    def test_record_checksum_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            wheel = Path(temporary) / "token_prediction-0.1.0-py3-none-any.whl"
            _write_wheel(wheel, corrupt_record=True)

            with self.assertRaisesRegex(DistributionCheckError, "checksum"):
                inspect_wheel(wheel)

    def test_repository_only_and_sensitive_files_are_rejected(self) -> None:
        cases = {
            "repository material": {"docs/report.md": b"not for the wheel\n"},
            "sensitive material": {"token_prediction/auth.json": b"{}\n"},
            "tests": {"token_prediction/tests/test_cli.py": b"pass\n"},
        }
        for label, extra in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                wheel = Path(temporary) / "token_prediction-0.1.0-py3-none-any.whl"
                _write_wheel(wheel, extra=extra)

                with self.assertRaises(DistributionCheckError):
                    inspect_wheel(wheel)


if __name__ == "__main__":
    unittest.main()
