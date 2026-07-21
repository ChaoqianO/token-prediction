from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_bagen_combined import (
    BAGEN_REPO,
    BAGEN_REVISION,
    FAMILY_SPECS,
    BagenCombinedAuditError,
    atomic_write_json,
    build_combined_audit,
)
from scripts.audit_bagen_swebench import build_audit as build_family_audit
from tests.test_bagen_swebench_reader import (
    _assistant,
    _base_messages,
    _payload,
    _write_trajectory,
)


TASK_ID = "fixture__project-1"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob_etag(path: Path) -> str:
    value = path.read_bytes()
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(value)}\0".encode("ascii"))
    digest.update(value)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


class _Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.bagen_root = root / "workspace" / "external" / "bagen"
        self.family_roots: dict[str, Path] = {}
        self.family_audits: dict[str, Path] = {}
        self.manifest = self.bagen_root / "manifest.jsonl"
        self.manifest_summary = self.bagen_root / "manifest_summary.json"
        self.expected_counts: dict[str, int] = {}
        self._build()

    def _build(self) -> None:
        manifest_items: list[dict[str, object]] = []
        raw_bytes = 0
        row_count = 0
        conditions: set[str] = set()
        for family, (family_root_name, audit_filename) in FAMILY_SPECS.items():
            family_root = self.bagen_root / "origin" / family_root_name
            self.family_roots[family] = family_root
            payload = _payload(
                _base_messages(
                    _assistant(
                        f"response-{family}",
                        100,
                        10,
                        content="RAW_FIXTURE_CONTENT_MUST_NOT_LEAK",
                    )
                ),
                instance_id=TASK_ID,
            )
            raw_path = _write_trajectory(family_root, payload, instance_id=TASK_ID)
            relative_path = raw_path.relative_to(family_root).as_posix()
            hub_path = f"origin/{family_root_name}/{relative_path}"
            manifest_items.append(
                {
                    "download_url": (
                        f"https://huggingface.co/datasets/{BAGEN_REPO}/resolve/main/"
                        f"{hub_path}"
                    ),
                    "extension": ".json",
                    "path": hub_path,
                    "relative_path": hub_path[len("origin/") :],
                    "size_bytes": raw_path.stat().st_size,
                    "top_level": "origin",
                }
            )
            raw_bytes += raw_path.stat().st_size

            audit = build_family_audit(family_root)
            audit_path = self.bagen_root / "audits" / audit_filename
            self.family_audits[family] = _write_json(audit_path, audit)
            row_count += int(audit["dataset"]["row_count"])
            conditions.update(str(item["condition_id"]) for item in audit["raw_files"])

        self.bagen_root.mkdir(parents=True, exist_ok=True)
        self.manifest.write_text(
            "".join(
                json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
                for item in sorted(manifest_items, key=lambda item: str(item["path"]))
            ),
            encoding="utf-8",
        )
        self._write_manifest_summary(manifest_items)
        self.expected_counts = {
            "task_id_count": 1,
            "run_id_count": len(FAMILY_SPECS),
            "trajectory_id_count": len(FAMILY_SPECS),
            "condition_id_count": len(conditions),
            "dataset_row_count": row_count,
            "raw_file_count": len(FAMILY_SPECS),
            "raw_bytes": raw_bytes,
        }

    def _write_manifest_summary(self, items: list[dict[str, object]]) -> None:
        traj_items = [item for item in items if str(item["path"]).endswith(".traj.json")]
        summary = {
            "source_id": f"{BAGEN_REPO}@main",
            "source_url": f"https://huggingface.co/datasets/{BAGEN_REPO}",
            "resolved_revision": BAGEN_REVISION,
            "manifest_file": self.manifest.name,
            "manifest_bytes": self.manifest.stat().st_size,
            "manifest_sha256": _file_sha256(self.manifest),
            "manifest_etag": _git_blob_etag(self.manifest),
            "file_count": len(items),
            "total_bytes": sum(int(item["size_bytes"]) for item in items),
            "traj_json_count": len(traj_items),
            "traj_json_bytes": sum(int(item["size_bytes"]) for item in traj_items),
        }
        _write_json(self.manifest_summary, summary)

    def build(self) -> dict[str, object]:
        return build_combined_audit(
            self.manifest_summary,
            self.manifest,
            self.family_roots,
            self.family_audits,
            expected_counts=self.expected_counts,
            expected_task_cross_family_distribution={str(len(FAMILY_SPECS)): 1},
            expected_dataset_id=None,
        )


class BagenCombinedAuditTests(unittest.TestCase):
    def test_build_verifies_all_families_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            first = fixture.build()
            second = fixture.build()

        self.assertEqual(first, second)
        payload_sha256 = first["audit_payload_sha256"]
        unhashed = dict(first)
        del unhashed["audit_payload_sha256"]
        self.assertEqual(payload_sha256, _canonical_sha256(unhashed))

        self.assertEqual(first["combined_audit_schema_version"], 1)
        self.assertEqual(first["hub"]["repo"], BAGEN_REPO)
        self.assertEqual(first["hub"]["resolved_revision"], BAGEN_REVISION)
        self.assertEqual(first["counts"], fixture.expected_counts)
        self.assertEqual(first["task_cross_family_distribution"], {"5": 1})
        self.assertEqual(len(first["families"]), 5)
        self.assertEqual(len(first["family_audit_index"]), 5)
        self.assertEqual(len(first["family_dataset_id_index"]), 5)
        self.assertEqual(len(first["task_family_mapping"]), 1)
        self.assertEqual(
            len(first["task_family_mapping"][0]["trajectories"]), len(FAMILY_SPECS)
        )
        self.assertEqual(
            first["canonical_family_index_sha256"],
            _canonical_sha256(first["canonical_family_index"]),
        )
        self.assertEqual(
            set(first["source_files"]),
            {"reader", "builder", "labels", "audit", "family_audit"},
        )
        self.assertEqual(
            first["construction"]["command"],
            "$env:PYTHONPATH='src'; python scripts/audit_bagen_combined.py",
        )

        encoded = json.dumps(first, ensure_ascii=False, sort_keys=True)
        self.assertNotIn("RAW_FIXTURE_CONTENT_MUST_NOT_LEAK", encoded)
        self.assertNotIn(str(Path(temporary).resolve()), encoded)
        self.assertNotIn("generated_at", encoded.lower())
        self.assertNotIn("created_at", encoded.lower())

    def test_tampered_family_raw_sha_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            audit_path = fixture.family_audits["qwen3-235b"]
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["raw_files"][0]["sha256"] = "0" * 64
            _write_json(audit_path, audit)
            with self.assertRaisesRegex(BagenCombinedAuditError, "sha256"):
                fixture.build()

    def test_tampered_family_dataset_id_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            audit_path = fixture.family_audits["gemini3.1"]
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["dataset"]["dataset_id"] = "0" * 64
            _write_json(audit_path, audit)
            with self.assertRaisesRegex(BagenCombinedAuditError, "dataset.dataset_id"):
                fixture.build()

    def test_manifest_size_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            lines = fixture.manifest.read_text(encoding="utf-8").splitlines()
            first = json.loads(lines[0])
            first["size_bytes"] += 1
            lines[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
            fixture.manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(BagenCombinedAuditError, "manifest summary"):
                fixture.build()

    def test_duplicate_and_non_finite_input_fields_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            fixture.manifest_summary.write_text(
                '{"source_id":"first","source_id":"second"}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(BagenCombinedAuditError, "duplicate JSON field"):
                fixture.build()

        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            fixture.manifest_summary.write_text('{"value":NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(BagenCombinedAuditError, "non-finite"):
                fixture.build()

    def test_atomic_write_replaces_destination_and_cleans_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            output = fixture.bagen_root / "combined_swebench_audit.json"
            audit = fixture.build()
            atomic_write_json(output, {"old": True})
            atomic_write_json(output, audit)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), audit)
            self.assertEqual(
                list(output.parent.glob(".combined_swebench_audit.json.*.tmp")), []
            )


if __name__ == "__main__":
    unittest.main()
