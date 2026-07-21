from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from token_prediction.contracts import (
    CanonicalEvent,
    CapabilityError,
    EventType,
    Observable,
    SourceCapabilities,
    SourceDescriptor,
    SourceRequirements,
    TokenUsage,
    resolve_canonical_input_file,
)


class EventContractTests(unittest.TestCase):
    def test_roundtrip_preserves_identity(self) -> None:
        source_payload = {"request_tokens_local": 10}
        event = CanonicalEvent.create(
            trajectory_id="t1",
            event_seq=0,
            event_type=EventType.REQUEST_BUILT,
            logical_call_id="c1",
            payload=source_payload,
        )
        source_payload["request_tokens_local"] = 999
        restored = CanonicalEvent.from_dict(event.to_dict())
        self.assertEqual(restored.to_dict(), event.to_dict())
        self.assertEqual(restored.payload["request_tokens_local"], 10)

    def test_api_event_requires_attempt_identity(self) -> None:
        with self.assertRaises(ValueError):
            CanonicalEvent.create(
                trajectory_id="t1",
                event_seq=0,
                event_type=EventType.API_COMPLETED,
                logical_call_id="c1",
            )

    def test_missing_usage_stays_unknown(self) -> None:
        usage = TokenUsage.from_mapping({})
        self.assertIsNone(usage.input_tokens)
        self.assertIsNone(usage.output_tokens)
        self.assertIsNone(usage.accounted_total_tokens)

    def test_cache_and_reasoning_fields_do_not_double_count(self) -> None:
        usage = TokenUsage.from_mapping(
            {
                "input_tokens": 100,
                "cached_input_tokens": 80,
                "output_tokens": 20,
                "reasoning_output_tokens": 10,
                "total_tokens": 120,
            }
        )
        self.assertEqual(usage.accounted_total_tokens, 120)
        self.assertTrue(usage.reported_total_matches)

    def test_missing_event_sequence_is_rejected(self) -> None:
        event = CanonicalEvent.create(
            trajectory_id="t1",
            event_seq=0,
            event_type=EventType.TASK_STARTED,
        ).to_dict()
        del event["event_seq"]
        with self.assertRaises(ValueError):
            CanonicalEvent.from_dict(event)

    def test_canonical_event_values_are_type_strict(self) -> None:
        base = CanonicalEvent.create(
            trajectory_id="t1",
            event_seq=0,
            event_type=EventType.TASK_STARTED,
        ).to_dict()
        cases = (
            ("schema_version", True, "integer"),
            ("event_seq", 1.5, "integer"),
            ("event_id", 7, "string"),
            ("payload", [], "object"),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                malformed = dict(base)
                malformed[field] = value
                with self.assertRaisesRegex(ValueError, message):
                    CanonicalEvent.from_dict(malformed)
        unknown = {**base, "future_field": True}
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            CanonicalEvent.from_dict(unknown)

    def test_token_usage_rejects_coerced_counts(self) -> None:
        for value in (1.9, "3", True, -1):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "non-negative integer"
            ):
                TokenUsage.from_mapping(
                    {"input_tokens": value, "output_tokens": 1}
                )


class CapabilityTests(unittest.TestCase):
    def test_missing_capabilities_fail_closed(self) -> None:
        capabilities = SourceCapabilities(
            source_id="codex",
            observables=frozenset({Observable.TASK_USAGE}),
        )
        requirements = SourceRequirements(
            observables=frozenset(
                {Observable.CALL_USAGE, Observable.REQUEST_LOCAL_COUNT}
            ),
        )
        with self.assertRaises(CapabilityError) as caught:
            capabilities.require(requirements)
        self.assertEqual(
            caught.exception.missing,
            ("call_usage", "request_local_count"),
        )

    def test_capability_contract_serialization_and_hash_are_stable(self) -> None:
        first = SourceCapabilities(
            source_id="fixture",
            observables=frozenset(
                {
                    Observable.TASK_TERMINATION,
                    Observable.ATTEMPT_USAGE,
                    Observable.REQUEST_BOUNDARIES,
                }
            ),
        )
        second = SourceCapabilities.from_dict(
            {
                "source": "declared",
                "source_id": "fixture",
                "observables": [
                    "request_boundaries",
                    "attempt_usage",
                    "task_termination",
                ],
            }
        )
        self.assertEqual(first, second)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.contract_hash, second.contract_hash)
        self.assertEqual(first.capability_hash, first.contract_hash)

    def test_source_descriptor_binds_manifest_revision_and_capabilities(self) -> None:
        capabilities = SourceCapabilities(
            source_id="fixture",
            observables=frozenset({Observable.TASK_USAGE}),
        )
        descriptor = SourceDescriptor(
            source_id="fixture",
            revision="revision-1",
            manifest_path="workspace/manifests/fixture.json",
            manifest_sha256="a" * 64,
            capabilities=capabilities,
        )
        restored = SourceDescriptor.from_dict(descriptor.to_dict())
        self.assertEqual(restored, descriptor)
        self.assertEqual(restored.descriptor_hash, descriptor.descriptor_hash)
        changed = SourceDescriptor(
            source_id="fixture",
            revision="revision-2",
            manifest_path=descriptor.manifest_path,
            manifest_sha256=descriptor.manifest_sha256,
            capabilities=capabilities,
        )
        self.assertNotEqual(changed.descriptor_hash, descriptor.descriptor_hash)

        tampered = descriptor.to_dict()
        tampered["capability_contract_hash"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "capability contract hash"):
            SourceDescriptor.from_dict(tampered)

        for mutation, message in (
            ({**descriptor.to_dict(), "unknown": True}, "missing or unknown"),
            (
                {**descriptor.to_dict(), "descriptor_schema_version": 0},
                "schema version",
            ),
            (
                {
                    key: value
                    for key, value in descriptor.to_dict().items()
                    if key != "capability_contract_hash"
                },
                "missing or unknown",
            ),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                SourceDescriptor.from_dict(mutation)

        duplicate_capability = descriptor.to_dict()
        duplicate_capability["capabilities"]["observables"] = [
            "task_usage",
            "task_usage",
        ]
        with self.assertRaisesRegex(ValueError, "must be unique"):
            SourceDescriptor.from_dict(duplicate_capability)

    def test_source_descriptor_rejects_unsafe_manifest_paths(self) -> None:
        capabilities = SourceCapabilities(source_id="fixture")
        unsafe = (
            "/absolute/manifest.json",
            "C:/secret/manifest.json",
            "C:\\secret\\manifest.json",
            "../manifest.json",
            "nested/../manifest.json",
            "nested\\manifest.json",
            "nested//manifest.json",
            "./manifest.json",
            " manifest.json",
            "manifest.json ",
        )
        for path in unsafe:
            with self.subTest(path=path), self.assertRaisesRegex(
                ValueError, "canonical relative POSIX"
            ):
                SourceDescriptor(
                    source_id="fixture",
                    revision="revision",
                    manifest_path=path,
                    manifest_sha256="a" * 64,
                    capabilities=capabilities,
                )

    def test_safe_input_resolver_rejects_in_root_file_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.json"
            target.write_text("{}\n", encoding="utf-8")
            link = root / "linked.json"
            try:
                link.symlink_to(target)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"file symlink creation is unavailable: {exc}")
            with self.assertRaisesRegex(ValueError, "symlinks.*reparse points"):
                resolve_canonical_input_file(
                    root,
                    "linked.json",
                    context="test input",
                )

    def test_safe_input_resolver_rejects_symlinked_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real_parent = root / "real-parent"
            real_parent.mkdir()
            (real_parent / "input.json").write_text("{}\n", encoding="utf-8")
            linked_parent = root / "linked-parent"
            try:
                linked_parent.symlink_to(real_parent, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"directory symlink creation is unavailable: {exc}")
            with self.assertRaisesRegex(ValueError, "symlinks.*reparse points"):
                resolve_canonical_input_file(
                    root,
                    "linked-parent/input.json",
                    context="test input",
                )


if __name__ == "__main__":
    unittest.main()
