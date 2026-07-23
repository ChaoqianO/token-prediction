from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from token_prediction.checkpoint import CandidateCheckpointStore, CheckpointError
from token_prediction.dataset import PredictionPosition, PredictionTarget
from token_prediction.estimators import TokenForecast
from token_prediction.experiment import (
    CandidateExecutionKey,
    CandidateResult,
    FoldArtifact,
    PredictionRecord,
)
from token_prediction.lineage import ArtifactVerificationError


TARGET = PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS


def _key() -> CandidateExecutionKey:
    return CandidateExecutionKey(
        experiment_id="checkpoint-roundtrip",
        candidate_id="candidate",
        candidate_hash="a" * 64,
        dataset_id="dataset",
        split_plan_id="b" * 64,
        split_seed=20260719,
        eligibility_hash="c" * 64,
        position=PredictionPosition.TASK_PRE,
        target=TARGET,
        condition_id="condition",
        calibrator_id="none",
        alpha=0.1,
        source_provenance_hash="d" * 64,
    )


def _result() -> CandidateResult:
    forecast = TokenForecast(
        point_id="point-1",
        target=TARGET,
        lower=10.0,
        point=20.0,
        upper=30.0,
        raw_lower=9.0,
        raw_point=20.0,
        raw_upper=31.0,
    )
    prediction = PredictionRecord(
        candidate_id="candidate",
        point_id="point-1",
        task_id="task-1",
        trajectory_id="trajectory-1",
        condition_id="condition",
        fold=0,
        target=TARGET,
        forecast=forecast,
        sample_weight=1.0,
    )
    metrics = {
        "n_points": 1,
        "n_tasks": 1,
        "weight_sum": 1.0,
        "mae": 0.0,
    }
    return CandidateResult(
        candidate_id="candidate",
        candidate_hash="a" * 64,
        dataset_id="dataset",
        split_plan_id="b" * 64,
        eligibility_hash="c" * 64,
        position=PredictionPosition.TASK_PRE,
        target=TARGET,
        condition_id="condition",
        calibrator_id="none",
        alpha=0.1,
        metric_suite_id="metric-suite",
        predictions=(prediction,),
        metrics=metrics,
        fold_metrics={0: metrics},
        task_metrics={"task-1": {"n_points": 1, "mae": 0.0}},
        fold_artifacts=(
            FoldArtifact(
                fold=0,
                encoder={"schema_version": 1, "names": ("feature",)},
                fit_report={"device": "cpu", "best_epoch": 2},
                feature_importance=({"feature": "feature", "gain": 1.0},),
                model_strings={"model-0.txt": "model"},
                bundle_files={"manifest.json": b"{}\n"},
                calibrator={"calibrator_id": "none"},
                provenance={"source_hash": "e" * 64},
            ),
        ),
    )


class CandidateCheckpointStoreTests(unittest.TestCase):
    def test_candidate_result_roundtrip_is_exact_and_pickle_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CandidateCheckpointStore(
                temporary,
                run_id="roundtrip",
                run_semantic={"source": "fixture", "schema_version": 1},
            )
            key = _key()
            result = _result()
            store.save(key, result)
            loaded = store.load(key)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.candidate_id, result.candidate_id)
            self.assertEqual(loaded.predictions, result.predictions)
            self.assertEqual(loaded.fold_artifacts[0].encoder["names"], ["feature"])
            store.save(key, result)
            files = {
                path.name
                for path in (
                    Path(temporary) / "roundtrip" / "candidates" / key.content_hash
                ).iterdir()
            }
            self.assertEqual(
                files,
                {"candidate_result.json", "manifest.json", "_SUCCESS"},
            )
            self.assertFalse(
                any(
                    path.suffix in {".pkl", ".pickle", ".pt"} for path in Path(temporary).rglob("*")
                )
            )

    def test_candidate_tamper_and_extra_file_are_rejected(self) -> None:
        for mutation in ("tamper", "extra"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                store = CandidateCheckpointStore(
                    temporary,
                    run_id="tamper",
                    run_semantic={"source": "fixture"},
                )
                key = _key()
                store.save(key, _result())
                artifact = Path(temporary) / "tamper" / "candidates" / key.content_hash
                if mutation == "tamper":
                    with (artifact / "candidate_result.json").open("ab") as stream:
                        stream.write(b" ")
                else:
                    (artifact / "unexpected.txt").write_text("extra", encoding="utf-8")
                with self.assertRaises((CheckpointError, ArtifactVerificationError)):
                    store.load(key)

    def test_fit_checkpoint_keeps_latest_complete_epoch_and_binds_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CandidateCheckpointStore(
                temporary,
                run_id="fit-state",
                run_semantic={"source": "fixture"},
            )
            checkpoint = store.fit_checkpoint(_key(), 3)
            identity = {"fit": "neural", "fold": 3}
            checkpoint.save(identity, epoch=1, files={"state.safetensors": b"one"})
            checkpoint.save(identity, epoch=2, files={"state.safetensors": b"two"})
            self.assertEqual(checkpoint.load(identity), {"state.safetensors": b"two"})
            generations = tuple(
                path
                for path in (
                    Path(temporary) / "fit-state" / "fits" / _key().content_hash / "fold-03"
                ).iterdir()
                if path.name.startswith("epoch-")
            )
            self.assertEqual(len(generations), 1)
            with self.assertRaisesRegex(CheckpointError, "identity"):
                checkpoint.load({"fit": "different", "fold": 3})


if __name__ == "__main__":
    unittest.main()
