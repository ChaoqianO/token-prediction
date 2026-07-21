from __future__ import annotations

import json
import unittest
from pathlib import Path

from token_prediction.collection import BagenSwebenchReader, OpenHandsArchiveReader
from token_prediction.contracts import SourceDescriptor


ROOT = Path(__file__).resolve().parents[1]


class FrozenSourceDescriptorTests(unittest.TestCase):
    def test_tracked_descriptors_match_reader_capability_contracts(self) -> None:
        cases = {
            "bagen_swebench.json": (
                "bagen_swebench_traj_v2",
                BagenSwebenchReader.capabilities,
            ),
            "spend_openhands.json": (
                "openhands_archive_trajectory_v3",
                OpenHandsArchiveReader.capabilities,
            ),
        }
        for filename, (expected_source_id, reader_capabilities) in cases.items():
            with self.subTest(filename=filename):
                path = ROOT / "configs" / "source_descriptors" / filename
                payload = json.loads(path.read_text(encoding="utf-8"))
                descriptor = SourceDescriptor.from_dict(payload)
                self.assertEqual(descriptor.source_id, expected_source_id)
                self.assertEqual(descriptor.capabilities, reader_capabilities)
                self.assertEqual(
                    payload["capability_contract_hash"],
                    descriptor.capabilities.contract_hash,
                )
                self.assertEqual(len(descriptor.descriptor_hash), 64)


if __name__ == "__main__":
    unittest.main()
