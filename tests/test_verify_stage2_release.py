from __future__ import annotations

import copy
import unittest
from pathlib import Path

from scripts.run_stage2_experiments import Stage2ExperimentError
from scripts.verify_stage2_release import (
    DEFAULT_RELEASE_LOCK,
    MAX_RELEASE_JSON_BYTES,
    _code_hash_at_commit,
    _load_json,
    _resolve_artifact_source,
    _validate_release_document,
    verify_stage2_release,
)


ROOT = Path(__file__).resolve().parents[1]


class Stage2ReleaseVerifierTests(unittest.TestCase):
    def test_repository_release_controls_and_source_tree_close(self) -> None:
        result = verify_stage2_release(
            ROOT,
            tracked_only=True,
            require_git_clean=False,
        )
        self.assertEqual(result.locked_artifact_count, 5)
        self.assertEqual(result.verified_artifact_count, 0)
        self.assertFalse(result.final_holdout_evaluated)
        self.assertEqual(
            result.code_tree_sha256,
            "4e15c9f9b1c1eeeec14b1f22f8db74613591d3b4ecd14255018e9c035cf2c650",
        )

        release = _load_json(
            ROOT / DEFAULT_RELEASE_LOCK,
            maximum_bytes=MAX_RELEASE_JSON_BYTES,
            description="Stage 2 release test lock",
        )
        code = release["code_binding"]
        code_hash, paths = _code_hash_at_commit(ROOT, code["artifact_git_commit"])
        self.assertEqual(code_hash, code["code_tree_sha256"])
        self.assertGreater(len(paths), 50)

        status, reproduced_paths = _resolve_artifact_source(
            ROOT,
            "0" * 40,
            code["code_tree_sha256"],
        )
        self.assertTrue(status.startswith("source_tree_reproduced_at:"))
        self.assertEqual(reproduced_paths, paths)

    def test_unknown_fields_and_total_drift_fail_closed(self) -> None:
        release = _load_json(
            ROOT / DEFAULT_RELEASE_LOCK,
            maximum_bytes=MAX_RELEASE_JSON_BYTES,
            description="Stage 2 release test lock",
        )
        extra = copy.deepcopy(dict(release))
        extra["unexpected"] = True
        with self.assertRaisesRegex(Stage2ExperimentError, "keys"):
            _validate_release_document(extra)

        changed = copy.deepcopy(dict(release))
        changed["totals"]["exact_reload_fold_count"] = 974
        with self.assertRaisesRegex(Stage2ExperimentError, "totals"):
            _validate_release_document(changed)

    def test_final_holdout_claim_fails_closed(self) -> None:
        release = _load_json(
            ROOT / DEFAULT_RELEASE_LOCK,
            maximum_bytes=MAX_RELEASE_JSON_BYTES,
            description="Stage 2 release test lock",
        )
        changed = copy.deepcopy(dict(release))
        changed["protocol"]["final_holdout_evaluated"] = True
        with self.assertRaisesRegex(Stage2ExperimentError, "protocol"):
            _validate_release_document(changed)


if __name__ == "__main__":
    unittest.main()
