from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import verify_stage1_baseline as baseline
from token_prediction.collection import BagenSokobanReader
from token_prediction.contracts import EventType
from tests.test_bagen_sokoban_reader import _rollout


class _FakeBundle:
    def __init__(self, raw: tuple[float, float, float]) -> None:
        self.raw = raw

    def start(self, context: object) -> _FakeBundle:
        del context
        return self

    def predict(self, point: object) -> SimpleNamespace:
        del point
        return SimpleNamespace(
            raw_lower=self.raw[0],
            raw_point=self.raw[1],
            raw_upper=self.raw[2],
        )


class Stage1BaselineVerifierTests(unittest.TestCase):
    def test_legacy_projection_is_explicit_and_does_not_mutate_reader_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bagen.json"
            path.write_text(json.dumps([_rollout(0)]), encoding="utf-8")
            original = BagenSokobanReader().read_all(path)[0]

        projected = baseline._legacy_proxy_projection(original)
        original_start = original.events[0]
        projected_start = projected.events[0]
        original_requests = [
            event for event in original.events if event.event_type == EventType.REQUEST_BUILT
        ]
        projected_requests = [
            event for event in projected.events if event.event_type == EventType.REQUEST_BUILT
        ]

        self.assertIsNone(original_start.payload["task_tokens"])
        self.assertEqual(projected_start.payload["task_tokens"], 100)
        self.assertEqual(
            [event.payload["request_tokens_local"] for event in original_requests],
            [None, None],
        )
        self.assertEqual(
            [event.payload["request_tokens_local"] for event in projected_requests],
            [100, 100],
        )
        self.assertTrue(
            all(
                event.payload["request_token_count_source"]
                == "historical_provider_input_proxy"
                for event in projected_requests
            )
        )

    def test_legacy_task_proxy_stays_missing_when_first_call_usage_is_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bagen.json"
            path.write_text(json.dumps([_rollout(0)]), encoding="utf-8")
            original = BagenSokobanReader().read_all(path)[0]

        first_call = next(
            event.logical_call_id
            for event in original.events
            if event.event_type == EventType.REQUEST_BUILT
        )
        events = []
        for event in original.events:
            if (
                event.logical_call_id == first_call
                and event.event_type in {EventType.API_COMPLETED, EventType.API_FAILED}
            ):
                payload = event.payload
                payload["provider_input_tokens_post_response_audit"] = None
                events.append(event.with_payload(payload))
            else:
                events.append(event)

        projected = baseline._legacy_proxy_projection(
            baseline.Trajectory.from_events(events)
        )
        projected_requests = [
            event for event in projected.events if event.event_type == EventType.REQUEST_BUILT
        ]

        self.assertIsNone(projected.events[0].payload["task_tokens"])
        self.assertEqual(
            [event.payload["request_tokens_local"] for event in projected_requests],
            [None, 100],
        )

    def test_raw_prediction_parity_is_exact_and_identity_safe(self) -> None:
        point = SimpleNamespace(task_id="task", trajectory_id="trajectory", run_id="run")
        records = [
            {
                "experiment_id": baseline.BAGEN_EXPERIMENT_ID,
                "candidate_id": "lightgbm_history_only",
                "point_id": "point-1",
                "fold": 0,
                "raw_lower": 1.25,
                "raw_prediction": 2.5,
                "raw_upper": 4.75,
            },
            {
                "experiment_id": baseline.BAGEN_EXPERIMENT_ID,
                "candidate_id": "lightgbm_history_request_proxy",
                "point_id": "point-1",
                "fold": 0,
                "raw_lower": 1.25,
                "raw_prediction": 2.5,
                "raw_upper": 4.75,
            },
            {
                "experiment_id": "unrelated",
                "candidate_id": "lightgbm_history_only",
                "point_id": "not-selected",
                "fold": 0,
                "raw_lower": 0.0,
                "raw_prediction": 0.0,
                "raw_upper": 0.0,
            },
        ]
        bundles = {
            (
                baseline.BAGEN_EXPERIMENT_ID,
                "lightgbm_history_only",
                0,
            ): _FakeBundle((1.25, 2.5, 4.75)),
            (
                baseline.BAGEN_EXPERIMENT_ID,
                "lightgbm_history_request_proxy",
                0,
            ): _FakeBundle((1.25, 2.5, 4.75)),
        }
        with (
            patch.object(baseline, "_build_bagen_points", return_value={"point-1": point}),
            patch.object(baseline, "_prediction_records", return_value=records),
        ):
            first = baseline._verify_parity(Path("artifact"), Path("bagen.json"), bundles)
            second = baseline._verify_parity(Path("artifact"), Path("bagen.json"), bundles)

        self.assertEqual(first, second)
        self.assertEqual(first[:2], (2, 0))
        self.assertRegex(first[2], r"^[0-9a-f]{64}$")

    def test_raw_prediction_mismatch_fails_the_full_verifier(self) -> None:
        manifest = SimpleNamespace(
            artifact_id="a" * 64,
            files={"bundle": "hash"},
        )
        with (
            patch.object(baseline, "_artifact_manifest", return_value=manifest),
            patch.object(baseline, "_protocol", return_value=("b" * 64, "c" * 64)),
            patch.object(baseline, "_sha256", side_effect=("c" * 64, "d" * 64)),
            patch.object(baseline, "_load_bundles", return_value={"bundle": object()}),
            patch.object(baseline, "_verify_parity", return_value=(7, 1, "e" * 64)),
        ):
            with self.assertRaisesRegex(baseline.BaselineVerificationError, "parity"):
                baseline.verify_stage1(
                    "artifact",
                    "bagen.json",
                    repository_root=".",
                )

    def test_commit_hash_reconstructs_the_original_stage1_code_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "src" / "token_prediction" / "__init__.py"
            runner = root / baseline.STAGE1_SCRIPT
            source.parent.mkdir(parents=True)
            runner.parent.mkdir(parents=True)
            source.write_bytes(b"VALUE = 1\n")
            runner.write_bytes(b"print('stage1')\n")
            git = baseline._git_executable()
            for command in (
                ("init", "-q"),
                ("config", "user.email", "ci@example.invalid"),
                ("config", "user.name", "CI"),
                ("add", "src", baseline.STAGE1_SCRIPT),
                ("commit", "-q", "-m", "fixture"),
            ):
                subprocess.run((git, "-C", str(root), *command), check=True)

            commit, actual = baseline._code_hash_at_commit(root, "HEAD")
            expected = hashlib.sha256()
            for relative, payload in (
                ("src/token_prediction/__init__.py", b"VALUE = 1\n"),
                (baseline.STAGE1_SCRIPT, b"print('stage1')\n"),
            ):
                expected.update(relative.encode("utf-8"))
                expected.update(b"\0")
                expected.update(payload)
                expected.update(b"\0")

            self.assertRegex(commit, r"^[0-9a-f]{40}$")
            self.assertEqual(actual, expected.hexdigest())
            self.assertEqual(baseline.discover_source_commit(root, actual), commit)

    def test_bound_baseline_document_is_closed_and_unbound_output_is_rejected(self) -> None:
        summary = baseline.VerificationSummary(
            artifact_id="a" * 64,
            artifact_manifest_sha256="b" * 64,
            bagen_source_sha256="c" * 64,
            bundle_count=20,
            parity_candidates=tuple(sorted(baseline.BAGEN_CANDIDATES)),
            parity_projection=baseline.PARITY_PROJECTION,
            parity_record_count=992,
            parity_mismatch_count=0,
            parity_sha256="d" * 64,
            protocol_code_sha256="e" * 64,
            source_binding_status="bound",
            source_commit="f" * 40,
        )
        document = baseline._baseline_document(summary)

        self.assertEqual(baseline._baseline_source_commit(document), "f" * 40)
        with self.assertRaisesRegex(baseline.BaselineVerificationError, "field set"):
            baseline._baseline_source_commit({**document, "extra": True})
        with self.assertRaisesRegex(baseline.BaselineVerificationError, "recoverable"):
            baseline._baseline_document(
                baseline.VerificationSummary(
                    **{
                        **summary.__dict__,
                        "source_binding_status": "unrecoverable",
                        "source_commit": None,
                    }
                )
            )

    def test_strict_json_rejects_duplicate_fields_and_non_finite_numbers(self) -> None:
        for payload in ('{"a": 1, "a": 2}', '{"a": 1e999}'):
            with self.subTest(payload=payload):
                with self.assertRaises(baseline.BaselineVerificationError):
                    baseline._strict_json_loads(payload, label="fixture")


if __name__ == "__main__":
    unittest.main()
