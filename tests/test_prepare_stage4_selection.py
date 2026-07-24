from __future__ import annotations

import copy
import unittest

from scripts.prepare_stage4_selection import (
    EXPECTED_FOLDS,
    SELECTION_ENSEMBLE_POLICY_ID,
    SELECTION_HOLDOUT_PROTOCOL_ID,
    SELECTION_POLICY_ID,
    SELECTION_REPLACEMENT_POLICY_ID,
    SELECTION_SCHEMA_VERSION,
    SOURCE_ARTIFACTS,
    Stage4SelectionError,
    _replacement_guard,
    verify_selection_document,
)
from token_prediction.development import STAGE_SPLIT_SEEDS
from token_prediction.final_ensemble import semantic_sha256


def _candidate(
    candidate_id: str,
    *,
    mean_mae: float,
    ci_uppers: tuple[float, float, float] | None,
) -> dict[str, object]:
    seed_results = []
    for split_seed, upper in zip(STAGE_SPLIT_SEEDS, ci_uppers or (0.0, 0.0, 0.0)):
        seed_results.append(
            {
                "split_seed": split_seed,
                "paired_vs_reference": (
                    None
                    if ci_uppers is None
                    else {
                        "mae_delta": upper - 1.0,
                        "mae_delta_ci_lower": upper - 2.0,
                        "mae_delta_ci_upper": upper,
                    }
                ),
            }
        )
    return {
        "candidate_id": candidate_id,
        "cross_seed_metrics": {"mae": {"mean": mean_mae}},
        "seed_results": seed_results,
    }


def _selection_document() -> dict[str, object]:
    cells = []
    for index in range(29):
        coordinates = {
            "source_name": f"source-{index}",
            "source_id": f"source-id-{index}",
            "condition_id": f"condition-{index}",
            "position": "call_pre",
            "target": "call_billable_total_tokens",
        }
        members = [
            {
                "split_seed": seed,
                "fold": fold,
                "member_sha256": semantic_sha256([index, seed, fold]),
            }
            for seed in STAGE_SPLIT_SEEDS
            for fold in range(EXPECTED_FOLDS)
        ]
        cells.append(
            {
                "cell_id": semantic_sha256(coordinates),
                **coordinates,
                "ensemble_member_count": 15,
                "members": members,
            }
        )
    base: dict[str, object] = {
        "selection_schema_version": SELECTION_SCHEMA_VERSION,
        "policy_id": SELECTION_POLICY_ID,
        "replacement_policy_id": SELECTION_REPLACEMENT_POLICY_ID,
        "ensemble_policy_id": SELECTION_ENSEMBLE_POLICY_ID,
        "code_binding": {},
        "source_artifacts": [{} for _value in SOURCE_ARTIFACTS],
        "cells": cells,
        "summary": {
            "source_artifact_count": len(SOURCE_ARTIFACTS),
            "cell_count": 29,
            "ensemble_member_count": 435,
            "split_seeds": list(STAGE_SPLIT_SEEDS),
            "outer_folds": EXPECTED_FOLDS,
        },
        "final_holdout": {
            "protocol_id": SELECTION_HOLDOUT_PROTOCOL_ID,
            "evaluated": False,
            "prediction_count": 0,
            "target_values_used_for_fit_calibration_scoring": False,
        },
    }
    base["selection_id"] = semantic_sha256(base)
    base["selection_payload_sha256"] = semantic_sha256(base)
    return base


class Stage4SelectionTests(unittest.TestCase):
    def test_frozen_source_inventory_has_unique_keys_and_paths(self) -> None:
        self.assertEqual(len({value.key for value in SOURCE_ARTIFACTS}), len(SOURCE_ARTIFACTS))
        self.assertEqual(len({value.path for value in SOURCE_ARTIFACTS}), len(SOURCE_ARTIFACTS))

    def test_selection_document_closes_over_29_by_15_members(self) -> None:
        document = _selection_document()
        self.assertEqual(
            verify_selection_document(document),
            document["selection_payload_sha256"],
        )

    def test_selection_document_rejects_member_loss_even_if_checksums_are_recomputed(
        self,
    ) -> None:
        document = _selection_document()
        document["cells"][0]["members"].pop()
        without_hashes = dict(document)
        without_hashes.pop("selection_id")
        without_hashes.pop("selection_payload_sha256")
        document["selection_id"] = semantic_sha256(without_hashes)
        payload = dict(document)
        payload.pop("selection_payload_sha256")
        document["selection_payload_sha256"] = semantic_sha256(payload)
        with self.assertRaisesRegex(Stage4SelectionError, "ensemble"):
            verify_selection_document(document)

    def test_replacement_guard_requires_all_three_ci_upper_values_below_zero(self) -> None:
        experiment = {
            "candidates": [
                _candidate("empirical", mean_mae=20.0, ci_uppers=None),
                _candidate("lightgbm_history", mean_mae=15.0, ci_uppers=None),
                _candidate("unstable_ablation", mean_mae=14.0, ci_uppers=(-1.0, 0.5, -0.2)),
            ]
        }
        guard = _replacement_guard(experiment)
        self.assertFalse(guard["candidates"][0]["qualified_replacement"])

    def test_replacement_guard_fails_when_an_ablation_is_stably_better(self) -> None:
        experiment = {
            "candidates": [
                _candidate("empirical", mean_mae=20.0, ci_uppers=None),
                _candidate("lightgbm_history", mean_mae=15.0, ci_uppers=None),
                _candidate("stable_ablation", mean_mae=14.0, ci_uppers=(-1.0, -0.5, -0.2)),
            ]
        }
        with self.assertRaisesRegex(Stage4SelectionError, "qualifies"):
            _replacement_guard(experiment)

    def test_payload_tampering_is_detected(self) -> None:
        document = _selection_document()
        tampered = copy.deepcopy(document)
        tampered["summary"]["cell_count"] = 30
        with self.assertRaises(Stage4SelectionError):
            verify_selection_document(tampered)


if __name__ == "__main__":
    unittest.main()
