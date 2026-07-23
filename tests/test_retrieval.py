from __future__ import annotations

import json
import hashlib
import unittest

from token_prediction.contracts import Observable, SourceCapabilities
from token_prediction.dataset import PredictionPoint, PredictionPosition, PredictionTarget
from token_prediction.retrieval import (
    FoldFittedTfidfRetriever,
    RetrievalExample,
    fit_fold_tfidf_retriever,
)


def _capabilities(*observables: Observable) -> SourceCapabilities:
    return SourceCapabilities("source", frozenset(observables))


def _examples() -> tuple[RetrievalExample, ...]:
    return (
        RetrievalExample("task-a", "fix parser error handling", 100, 4),
        RetrievalExample("task-b", "fix parser timeout handling", 300, 8),
        RetrievalExample("task-c", "improve image renderer cache", 900, 12),
    )


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
        with self.assertRaisesRegex(ValueError, "missing_observables:task_text"):
            fit_fold_tfidf_retriever(
                _examples(),
                capabilities=_capabilities(Observable.REQUEST_MESSAGES),
                fit_task_ids=frozenset({"task-a", "task-b", "task-c"}),
            )

    def test_fit_scope_is_exact_deterministic_and_excludes_the_query_task(self) -> None:
        examples = _examples()
        capabilities = _capabilities(Observable.TASK_TEXT)
        fit_tasks = frozenset(example.task_id for example in examples)
        first = fit_fold_tfidf_retriever(
            examples,
            capabilities=capabilities,
            fit_task_ids=fit_tasks,
            k=2,
        )
        second = fit_fold_tfidf_retriever(
            reversed(examples),
            capabilities=capabilities,
            fit_task_ids=fit_tasks,
            k=2,
        )
        self.assertEqual(first.content_hash, second.content_hash)
        result = first.transform(
            task_id="task-a",
            task_text="fix parser handling",
        )
        self.assertEqual(result.neighbor_count, 1)
        self.assertEqual(result.total_tokens_median, 300)
        self.assertEqual(result.call_count_median, 8)
        self.assertGreater(float(result.mean_similarity or 0), 0)

        with self.assertRaisesRegex(ValueError, "exactly match"):
            fit_fold_tfidf_retriever(
                examples,
                capabilities=capabilities,
                fit_task_ids=frozenset({"task-a", "task-b", "task-c", "holdout"}),
            )

    def test_bundle_is_strict_json_pickle_free_and_tamper_evident(self) -> None:
        examples = _examples()
        fitted = fit_fold_tfidf_retriever(
            examples,
            capabilities=_capabilities(Observable.TASK_TEXT),
            fit_task_ids=frozenset(example.task_id for example in examples),
        )
        payload = fitted.to_json_bytes()
        self.assertNotIn(b"task-a", payload)
        self.assertNotIn(b"pickle", payload.lower())
        reloaded = FoldFittedTfidfRetriever.from_json_bytes(payload)
        self.assertEqual(reloaded, fitted)
        tampered = json.loads(payload)
        tampered["idf"][0] += 1
        with self.assertRaisesRegex(ValueError, "content hash mismatch"):
            FoldFittedTfidfRetriever.from_json_bytes(
                json.dumps(tampered, sort_keys=True).encode()
            )
        duplicate = payload.replace(
            b'{"content_hash":',
            b'{"content_hash":"0","content_hash":',
            1,
        )
        with self.assertRaisesRegex(ValueError, "duplicate retrieval JSON key"):
            FoldFittedTfidfRetriever.from_json_bytes(duplicate)
        for field, replacement in (("k", "5"), ("max_features", True)):
            malformed = json.loads(payload)
            malformed[field] = replacement
            with self.subTest(field=field):
                with self.assertRaisesRegex(TypeError, "must be integers"):
                    FoldFittedTfidfRetriever.from_dict(_reclose(malformed))
        malformed_total = json.loads(payload)
        malformed_total["documents"][0]["total_tokens"] = "100"
        with self.assertRaisesRegex(TypeError, "totals must be numeric"):
            FoldFittedTfidfRetriever.from_dict(_reclose(malformed_total))

    def test_point_augmentation_changes_only_features_and_missing_similarity_stays_missing(
        self,
    ) -> None:
        examples = _examples()
        fitted = fit_fold_tfidf_retriever(
            examples,
            capabilities=_capabilities(Observable.TASK_TEXT),
            fit_task_ids=frozenset(example.task_id for example in examples),
        )
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
        )
        self.assertEqual(augmented.point_id, point.point_id)
        self.assertEqual(augmented.features["task_char_count"], 10)
        self.assertIsNone(
            augmented.features["similar_task_total_tokens_median"]
        )


if __name__ == "__main__":
    unittest.main()
