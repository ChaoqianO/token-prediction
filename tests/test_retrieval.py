from __future__ import annotations

import hashlib
import json
import unittest
from functools import lru_cache

from token_prediction.contracts import Observable, SourceCapabilities
from token_prediction.dataset import (
    DatasetRow,
    LabelStatus,
    PredictionPoint,
    PredictionPosition,
    PredictionTarget,
    SupervisedDataset,
)
from token_prediction.development import (
    STAGE_SPLIT_SEEDS,
    DevelopmentProtocol,
    build_development_protocol,
)
from token_prediction.retrieval import (
    FoldFittedTfidfRetriever,
    RetrievalExample,
    fit_fold_tfidf_retriever,
)


INPUT_CONTRACT_HASH = "e" * 64
OUTER_FOLD = 0


def _capabilities(*observables: Observable) -> SourceCapabilities:
    return SourceCapabilities("source", frozenset(observables))


@lru_cache(maxsize=1)
def _protocol() -> tuple[DevelopmentProtocol, SourceCapabilities]:
    capabilities = _capabilities(Observable.TASK_TEXT)
    rows = tuple(
        DatasetRow(
            point=PredictionPoint(
                point_id=f"point-{index:03d}",
                source_event_id=f"event-{index:03d}",
                task_id=f"task-{index:03d}",
                trajectory_id=f"trajectory-{index:03d}",
                run_id=f"run-{index:03d}",
                prediction_context_id=f"context-{index:03d}",
                condition_id="condition",
                logical_call_id=None,
                attempt_id=None,
                cutoff_event_seq=0,
                position=PredictionPosition.TASK_LAUNCH,
                target=PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
                features={"task_char_count": index + 1},
                known_offset_tokens=0,
            ),
            label=100 + index,
            status=LabelStatus.OBSERVED,
        )
        for index in range(50)
    )
    dataset = SupervisedDataset(
        dataset_id="a" * 64,
        rows=rows,
        schema_version=2,
        source_descriptor_hash="b" * 64,
        capability_contract_hash=capabilities.contract_hash,
        input_contract_hash=INPUT_CONTRACT_HASH,
    )
    return build_development_protocol(dataset), capabilities


def _fit_examples(
    protocol: DevelopmentProtocol,
    *,
    split_seed: int = STAGE_SPLIT_SEEDS[0],
    outer_fold: int = OUTER_FOLD,
) -> tuple[RetrievalExample, ...]:
    plan = next(plan for plan in protocol.outer_plans if plan.seed == split_seed)
    train_tasks = sorted(plan.partition(outer_fold).train_tasks)
    result: list[RetrievalExample] = []
    for index, task_id in enumerate(train_tasks):
        if index == 0:
            text = "fix parser error handling"
        elif index == 1:
            text = "fix parser timeout handling"
        else:
            text = f"improve image renderer cache module {index}"
        result.append(
            RetrievalExample(
                task_id,
                text,
                100 + 200 * index,
                4 + 4 * index,
            )
        )
    return tuple(result)


def _fit(
    examples: tuple[RetrievalExample, ...] | None = None,
    *,
    protocol: DevelopmentProtocol | None = None,
    capabilities: SourceCapabilities | None = None,
    split_seed: int = STAGE_SPLIT_SEEDS[0],
    outer_fold: int = OUTER_FOLD,
    k: int = 5,
) -> FoldFittedTfidfRetriever:
    frozen_protocol, frozen_capabilities = _protocol()
    resolved_protocol = protocol or frozen_protocol
    return fit_fold_tfidf_retriever(
        examples or _fit_examples(
            resolved_protocol,
            split_seed=split_seed,
            outer_fold=outer_fold,
        ),
        capabilities=capabilities or frozen_capabilities,
        input_contract_hash=INPUT_CONTRACT_HASH,
        development_protocol=resolved_protocol,
        split_seed=split_seed,
        outer_fold=outer_fold,
        k=k,
    )


def _context() -> dict[str, object]:
    protocol, capabilities = _protocol()
    return {
        "development_protocol": protocol,
        "split_seed": STAGE_SPLIT_SEEDS[0],
        "outer_fold": OUTER_FOLD,
        "capabilities": capabilities,
        "input_contract_hash": INPUT_CONTRACT_HASH,
    }


def _reclose(value: dict[str, object]) -> dict[str, object]:
    payload = {key: item for key, item in value.items() if key != "content_hash"}
    value["content_hash"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return value


class FoldFittedRetrievalTests(unittest.TestCase):
    def test_missing_real_task_text_capability_fails_closed(self) -> None:
        protocol, _capabilities_with_text = _protocol()
        with self.assertRaisesRegex(ValueError, "missing_observables:task_text"):
            fit_fold_tfidf_retriever(
                _fit_examples(protocol),
                capabilities=_capabilities(Observable.REQUEST_MESSAGES),
                input_contract_hash=INPUT_CONTRACT_HASH,
                development_protocol=protocol,
                split_seed=STAGE_SPLIT_SEEDS[0],
                outer_fold=OUTER_FOLD,
            )

    def test_fit_scope_comes_from_frozen_outer_partition_and_is_deterministic(
        self,
    ) -> None:
        examples = _fit_examples(_protocol()[0])
        first = _fit(examples, k=2)
        second = _fit(tuple(reversed(examples)), k=2)
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertEqual(
            first.provenance.dataset_id,
            _protocol()[0].development_dataset.dataset_id,
        )
        self.assertEqual(
            first.provenance.development_protocol_id,
            _protocol()[0].protocol_id,
        )
        self.assertEqual(
            first.provenance.split_plan_id,
            _protocol()[0].outer_plans[0].split_plan_id,
        )
        self.assertEqual(first.provenance.split_seed, STAGE_SPLIT_SEEDS[0])
        self.assertEqual(first.provenance.outer_fold, OUTER_FOLD)
        self.assertEqual(
            first.provenance.capability_contract_hash,
            _protocol()[1].contract_hash,
        )
        self.assertEqual(
            first.provenance.input_contract_hash,
            INPUT_CONTRACT_HASH,
        )

        result = first.transform(
            task_id=examples[0].task_id,
            task_text="fix parser handling",
            **_context(),
        )
        self.assertEqual(result.neighbor_count, 1)
        self.assertEqual(result.total_tokens_median, examples[1].total_tokens)
        self.assertEqual(result.call_count_median, examples[1].call_count)
        self.assertGreater(float(result.mean_similarity or 0), 0)

    def test_outer_non_train_and_final_holdout_examples_are_rejected(self) -> None:
        protocol, capabilities = _protocol()
        split_plan = protocol.outer_plans[0]
        partition = split_plan.partition(OUTER_FOLD)
        base = _fit_examples(protocol)
        forbidden = {
            "validation": next(iter(partition.validation_tasks)),
            "calibration": next(iter(partition.calibration_tasks)),
            "outer_test": next(iter(partition.test_tasks)),
            "final_holdout": next(iter(protocol.final_holdout_tasks)),
        }
        for cohort, task_id in forbidden.items():
            leaked = base + (
                RetrievalExample(task_id, "leaked task text", 999, 99),
            )
            with self.subTest(cohort=cohort):
                with self.assertRaisesRegex(ValueError, "non_train"):
                    fit_fold_tfidf_retriever(
                        leaked,
                        capabilities=capabilities,
                        input_contract_hash=INPUT_CONTRACT_HASH,
                        development_protocol=protocol,
                        split_seed=split_plan.seed,
                        outer_fold=OUTER_FOLD,
                    )

    def test_bundle_is_strict_json_pickle_free_and_tamper_evident(self) -> None:
        fitted = _fit()
        payload = fitted.to_json_bytes()
        self.assertNotIn(b"task-000", payload)
        self.assertNotIn(b"pickle", payload.lower())
        reloaded = FoldFittedTfidfRetriever.from_json_bytes(payload, **_context())
        self.assertEqual(reloaded, fitted)
        tampered = json.loads(payload)
        tampered["idf"][0] += 1
        with self.assertRaisesRegex(ValueError, "content hash mismatch"):
            FoldFittedTfidfRetriever.from_json_bytes(
                json.dumps(tampered, sort_keys=True).encode(),
                **_context(),
            )
        duplicate = payload.replace(
            b'{"content_hash":',
            b'{"content_hash":"0","content_hash":',
            1,
        )
        with self.assertRaisesRegex(ValueError, "duplicate retrieval JSON key"):
            FoldFittedTfidfRetriever.from_json_bytes(duplicate, **_context())
        for field, replacement in (("k", "5"), ("max_features", True)):
            malformed = json.loads(payload)
            malformed[field] = replacement
            with self.subTest(field=field):
                with self.assertRaisesRegex(TypeError, "must be integers"):
                    FoldFittedTfidfRetriever.from_dict(
                        _reclose(malformed),
                        **_context(),
                    )
        malformed_total = json.loads(payload)
        malformed_total["documents"][0]["total_tokens"] = "100"
        with self.assertRaisesRegex(TypeError, "totals must be numeric"):
            FoldFittedTfidfRetriever.from_dict(
                _reclose(malformed_total),
                **_context(),
            )

    def test_load_and_transform_reject_mismatched_frozen_provenance(self) -> None:
        protocol, capabilities = _protocol()
        fitted = _fit()
        payload = fitted.to_json_bytes()
        with self.assertRaisesRegex(ValueError, "artifact provenance"):
            FoldFittedTfidfRetriever.from_json_bytes(
                payload,
                development_protocol=protocol,
                split_seed=STAGE_SPLIT_SEEDS[0],
                outer_fold=1,
                capabilities=capabilities,
                input_contract_hash=INPUT_CONTRACT_HASH,
            )
        with self.assertRaisesRegex(ValueError, "artifact provenance"):
            fitted.transform(
                task_id=next(iter(protocol.outer_plans[0].partition(0).test_tasks)),
                task_text="fix parser handling",
                development_protocol=protocol,
                split_seed=STAGE_SPLIT_SEEDS[1],
                outer_fold=OUTER_FOLD,
                capabilities=capabilities,
                input_contract_hash=INPUT_CONTRACT_HASH,
            )
        with self.assertRaisesRegex(ValueError, "input contract"):
            FoldFittedTfidfRetriever.from_json_bytes(
                payload,
                development_protocol=protocol,
                split_seed=STAGE_SPLIT_SEEDS[0],
                outer_fold=OUTER_FOLD,
                capabilities=capabilities,
                input_contract_hash="f" * 64,
            )

    def test_point_augmentation_changes_only_features_and_missing_similarity_stays_missing(
        self,
    ) -> None:
        fitted = _fit()
        point = PredictionPoint(
            point_id="point",
            source_event_id="event",
            task_id="new-task",
            trajectory_id="trajectory",
            run_id="run",
            prediction_context_id="context",
            condition_id="condition:a",
            logical_call_id=None,
            attempt_id=None,
            cutoff_event_seq=0,
            position=PredictionPosition.TASK_LAUNCH,
            target=PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
            features={"task_char_count": 10},
            known_offset_tokens=0,
        )
        augmented = fitted.transform_point(
            point,
            task_text="completely unrelated zyzzyva vocabulary",
            **_context(),
        )
        self.assertEqual(augmented.point_id, point.point_id)
        self.assertEqual(augmented.features["task_char_count"], 10)
        self.assertIsNone(
            augmented.features["similar_task_total_tokens_median"]
        )


if __name__ == "__main__":
    unittest.main()
