from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import token_prediction.lineage as lineage
from token_prediction.lineage import (
    ArtifactVerificationError,
    publish_artifact,
    verify_artifact,
)


class ArtifactContractTests(unittest.TestCase):
    def test_published_artifact_detects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            root.mkdir()
            data = root / "data.json"
            data.write_text('{"value": 1}\n', encoding="utf-8")
            manifest = publish_artifact(root, stage_name="fixture")
            self.assertEqual(verify_artifact(root).artifact_id, manifest.artifact_id)
            data.write_text('{"value": 2}\n', encoding="utf-8")
            with self.assertRaises(ArtifactVerificationError):
                verify_artifact(root)

    def test_success_marker_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ArtifactVerificationError):
                verify_artifact(root)

    def test_legacy_crlf_success_marker_is_explicitly_gated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data.json").write_text("{}", encoding="utf-8")
            manifest = publish_artifact(root, stage_name="fixture")
            (root / "_SUCCESS").write_bytes(
                (manifest.artifact_id + "\r\n").encode("ascii")
            )
            with self.assertRaises(ArtifactVerificationError):
                verify_artifact(root)
            self.assertEqual(
                verify_artifact(root, allow_legacy_crlf_success=True),
                manifest,
            )
            (root / "_SUCCESS").write_bytes(
                (manifest.artifact_id + "\r\nX").encode("ascii")
            )
            with self.assertRaises(ArtifactVerificationError):
                verify_artifact(root, allow_legacy_crlf_success=True)

    def test_published_artifact_rejects_unlisted_extra_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data.json").write_text("{}", encoding="utf-8")
            publish_artifact(root, stage_name="fixture")
            (root / "late-file.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactVerificationError, "file set"):
                verify_artifact(root)

    def test_nested_manifest_is_hashed_and_verified_as_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "fold" / "bundle"
            nested.mkdir(parents=True)
            inner_manifest = nested / "manifest.json"
            inner_manifest.write_text('{"bundle": 1}\n', encoding="utf-8")

            manifest = publish_artifact(root, stage_name="fixture")
            relative = "fold/bundle/manifest.json"
            self.assertIn(relative, manifest.files)
            self.assertEqual(verify_artifact(root).artifact_id, manifest.artifact_id)

            inner_manifest.write_text('{"bundle": 2}\n', encoding="utf-8")
            with self.assertRaisesRegex(ArtifactVerificationError, "checksum"):
                verify_artifact(root)

    def test_manifest_redacts_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data.json").write_text("{}", encoding="utf-8")
            manifest = publish_artifact(
                root,
                stage_name="fixture",
                metadata={"access_token": "do-not-persist", "input_tokens": 3},
            )
            self.assertEqual(manifest.metadata["access_token"], "[REDACTED]")
            self.assertEqual(manifest.metadata["input_tokens"], 3)
            self.assertNotIn(
                "do-not-persist",
                (root / "manifest.json").read_text(encoding="utf-8"),
            )

    def test_verifier_rejects_external_symlink_with_identical_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            root = workspace / "artifact"
            root.mkdir()
            data = root / "data.json"
            data.write_text('{"value": 1}\n', encoding="utf-8")
            publish_artifact(root, stage_name="fixture")

            external = workspace / "external.json"
            external.write_bytes(data.read_bytes())
            data.unlink()
            try:
                os.symlink(external, data)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlinks are unavailable on this platform: {exc}")

            with self.assertRaisesRegex(
                ArtifactVerificationError,
                "symlink|reparse point|regular non-link",
            ):
                verify_artifact(root)

    def test_verifier_rejects_atomic_root_swap_after_initial_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            root = workspace / "artifact"
            root.mkdir()
            (root / "data.json").write_text('{"value": 1}\n', encoding="utf-8")
            publish_artifact(root, stage_name="fixture")

            replacement = workspace / "replacement"
            shutil.copytree(root, replacement)
            displaced = workspace / "displaced"
            original_scan = lineage._scan_artifact
            swapped = False

            def scan_then_swap(directory: Path) -> object:
                nonlocal swapped
                tree = original_scan(directory)
                if not swapped:
                    os.replace(root, displaced)
                    os.replace(replacement, root)
                    swapped = True
                return tree

            with (
                mock.patch.object(lineage, "_scan_artifact", side_effect=scan_then_swap),
                self.assertRaisesRegex(
                    ArtifactVerificationError,
                    "root changed|changed after the artifact tree was scanned|changed identity",
                ),
            ):
                verify_artifact(root)
            self.assertTrue(swapped)

    def test_publisher_rejects_atomic_root_swap_before_initial_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            root = workspace / "artifact"
            root.mkdir()
            (root / "original.json").write_text('{"root": "a"}\n', encoding="utf-8")

            replacement = workspace / "replacement"
            replacement.mkdir()
            (replacement / "replacement.json").write_text(
                '{"root": "b"}\n',
                encoding="utf-8",
            )
            displaced = workspace / "displaced"
            original_scan = lineage._scan_artifact
            swapped = False

            def swap_then_scan(directory: Path) -> object:
                nonlocal swapped
                if not swapped:
                    os.replace(root, displaced)
                    os.replace(replacement, root)
                    swapped = True
                return original_scan(directory)

            with (
                mock.patch.object(lineage, "_scan_artifact", side_effect=swap_then_scan),
                self.assertRaisesRegex(
                    ArtifactVerificationError,
                    "root changed before the initial publication scan",
                ),
            ):
                publish_artifact(root, stage_name="fixture")
            self.assertTrue(swapped)
            self.assertFalse((root / "manifest.json").exists())
            self.assertFalse((root / "_SUCCESS").exists())

    def test_publish_rejects_entry_count_above_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "one.json").write_text("{}", encoding="utf-8")
            (root / "two.json").write_text("{}", encoding="utf-8")
            with (
                mock.patch.object(lineage, "_MAX_ARTIFACT_ENTRIES", 1),
                self.assertRaisesRegex(ArtifactVerificationError, "entry-count"),
            ):
                publish_artifact(root, stage_name="fixture")

    def test_publish_rejects_file_above_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data.bin"
            data.write_bytes(b"1234")
            with (
                mock.patch.object(lineage, "_MAX_ARTIFACT_FILE_BYTES", 3),
                self.assertRaisesRegex(ArtifactVerificationError, "size limit"),
            ):
                lineage.sha256_file(data)
            with (
                mock.patch.object(lineage, "_MAX_ARTIFACT_FILE_BYTES", 3),
                self.assertRaisesRegex(ArtifactVerificationError, "size limit"),
            ):
                publish_artifact(root, stage_name="fixture")


if __name__ == "__main__":
    unittest.main()
