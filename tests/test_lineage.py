from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
