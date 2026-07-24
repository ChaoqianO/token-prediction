from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from token_prediction.contracts import Observable, SourceCapabilities
from token_prediction.dataset import PredictionPoint
from token_prediction.development import DevelopmentProtocol
from token_prediction.features.reducer import FeatureValue


RETRIEVAL_SCHEMA_VERSION = 2
RETRIEVAL_POLICY_ID = "fold_fitted_word_unigram_bigram_tfidf_v2"
RETRIEVAL_FEATURE_NAMES = frozenset(
    {
        "similar_task_total_tokens_median",
        "similar_task_total_tokens_iqr",
        "similar_task_call_count_median",
        "similar_task_mean_similarity",
    }
)
_TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _semantic_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _task_hash(task_id: str) -> str:
    return hashlib.sha256(f"{RETRIEVAL_POLICY_ID}\0{task_id}".encode()).hexdigest()


def _required_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _terms(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    words = _TOKEN_PATTERN.findall(normalized)
    unigrams = [f"u:{word}" for word in words]
    bigrams = [f"b:{left}\u241f{right}" for left, right in zip(words, words[1:])]
    return tuple((*unigrams, *bigrams))


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile values are empty")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _sparse_vector(
    text: str,
    vocabulary_index: Mapping[str, int],
    idf: Sequence[float],
) -> tuple[tuple[int, float], ...]:
    counts = Counter(
        index
        for term in _terms(text)
        if (index := vocabulary_index.get(term)) is not None
    )
    weighted = {
        index: (1.0 + math.log(count)) * float(idf[index])
        for index, count in counts.items()
    }
    norm = math.sqrt(sum(value * value for value in weighted.values()))
    if norm == 0:
        return ()
    return tuple(
        (index, weighted[index] / norm)
        for index in sorted(weighted)
    )


def _cosine(
    left: Sequence[tuple[int, float]],
    right: Sequence[tuple[int, float]],
) -> float:
    left_index = 0
    right_index = 0
    total = 0.0
    while left_index < len(left) and right_index < len(right):
        left_key, left_value = left[left_index]
        right_key, right_value = right[right_index]
        if left_key == right_key:
            total += left_value * right_value
            left_index += 1
            right_index += 1
        elif left_key < right_key:
            left_index += 1
        else:
            right_index += 1
    return total


@dataclass(frozen=True)
class RetrievalExample:
    task_id: str
    task_text: str
    total_tokens: float
    call_count: float

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("retrieval task_id is required")
        if not isinstance(self.task_text, str) or not self.task_text.strip():
            raise ValueError("retrieval requires real non-empty task text")
        for name in ("total_tokens", "call_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class RetrievalFeatures:
    total_tokens_median: float | None
    total_tokens_iqr: float | None
    call_count_median: float | None
    mean_similarity: float | None
    neighbor_count: int

    def __post_init__(self) -> None:
        if self.neighbor_count < 0:
            raise ValueError("neighbor_count must be non-negative")
        values = (
            self.total_tokens_median,
            self.total_tokens_iqr,
            self.call_count_median,
            self.mean_similarity,
        )
        if self.neighbor_count == 0 and any(value is not None for value in values):
            raise ValueError("empty retrieval results must remain explicitly missing")
        if self.neighbor_count > 0 and any(value is None for value in values):
            raise ValueError("non-empty retrieval results require every feature")
        if any(
            value is not None
            and (not math.isfinite(float(value)) or float(value) < 0)
            for value in values
        ):
            raise ValueError("retrieval features must be finite and non-negative")

    def as_point_features(self) -> dict[str, FeatureValue]:
        return {
            "similar_task_total_tokens_median": self.total_tokens_median,
            "similar_task_total_tokens_iqr": self.total_tokens_iqr,
            "similar_task_call_count_median": self.call_count_median,
            "similar_task_mean_similarity": self.mean_similarity,
        }


@dataclass(frozen=True)
class _RetrievalDocument:
    task_hash: str
    vector: tuple[tuple[int, float], ...]
    total_tokens: float
    call_count: float

    def __post_init__(self) -> None:
        _required_sha256(self.task_hash, name="retrieval task hash")
        for name in ("total_tokens", "call_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"retrieval document {name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(
                    f"retrieval document {name} must be finite and non-negative"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "task_hash": self.task_hash,
            "vector": [[index, value] for index, value in self.vector],
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
        }


@dataclass(frozen=True)
class RetrievalProvenance:
    dataset_id: str
    development_protocol_id: str
    split_plan_id: str
    split_seed: int
    outer_fold: int
    capability_contract_hash: str
    input_contract_hash: str
    fit_task_set_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "dataset_id",
            "development_protocol_id",
            "split_plan_id",
            "capability_contract_hash",
            "input_contract_hash",
            "fit_task_set_sha256",
        ):
            _required_sha256(
                getattr(self, name),
                name=f"retrieval provenance {name}",
            )
        if (
            isinstance(self.split_seed, bool)
            or not isinstance(self.split_seed, int)
            or self.split_seed < 0
        ):
            raise ValueError("retrieval provenance split_seed must be non-negative")
        if (
            isinstance(self.outer_fold, bool)
            or not isinstance(self.outer_fold, int)
            or self.outer_fold < 0
        ):
            raise ValueError("retrieval provenance outer_fold must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "development_protocol_id": self.development_protocol_id,
            "split_plan_id": self.split_plan_id,
            "split_seed": self.split_seed,
            "outer_fold": self.outer_fold,
            "capability_contract_hash": self.capability_contract_hash,
            "input_contract_hash": self.input_contract_hash,
            "fit_task_set_sha256": self.fit_task_set_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RetrievalProvenance:
        expected = {
            "dataset_id",
            "development_protocol_id",
            "split_plan_id",
            "split_seed",
            "outer_fold",
            "capability_contract_hash",
            "input_contract_hash",
            "fit_task_set_sha256",
        }
        if set(value) != expected:
            raise ValueError("retrieval provenance has missing or extra fields")
        for name in expected - {"split_seed", "outer_fold"}:
            if not isinstance(value[name], str):
                raise TypeError(f"retrieval provenance {name} must be a string")
        for name in ("split_seed", "outer_fold"):
            if isinstance(value[name], bool) or not isinstance(value[name], int):
                raise TypeError(f"retrieval provenance {name} must be an integer")
        return cls(
            dataset_id=value["dataset_id"],
            development_protocol_id=value["development_protocol_id"],
            split_plan_id=value["split_plan_id"],
            split_seed=value["split_seed"],
            outer_fold=value["outer_fold"],
            capability_contract_hash=value["capability_contract_hash"],
            input_contract_hash=value["input_contract_hash"],
            fit_task_set_sha256=value["fit_task_set_sha256"],
        )


def _frozen_outer_provenance(
    *,
    development_protocol: DevelopmentProtocol,
    split_seed: int,
    outer_fold: int,
    capabilities: SourceCapabilities,
    input_contract_hash: str,
) -> tuple[RetrievalProvenance, frozenset[str]]:
    if Observable.TASK_TEXT not in capabilities.observables:
        raise ValueError("fold-fitted retrieval is gated: missing_observables:task_text")
    _required_sha256(input_contract_hash, name="retrieval input_contract_hash")
    if (
        development_protocol.parent_capability_contract_hash
        != capabilities.contract_hash
        or development_protocol.development_dataset.capability_contract_hash
        != capabilities.contract_hash
    ):
        raise ValueError(
            "retrieval capabilities do not match the frozen development protocol"
        )
    if (
        development_protocol.parent_input_contract_hash != input_contract_hash
        or development_protocol.development_dataset.input_contract_hash
        != input_contract_hash
    ):
        raise ValueError(
            "retrieval input contract does not match the frozen development protocol"
        )
    matching_plans = tuple(
        plan for plan in development_protocol.outer_plans if plan.seed == split_seed
    )
    if len(matching_plans) != 1:
        raise ValueError(
            "retrieval split seed does not identify one frozen outer split plan"
        )
    split_plan = matching_plans[0]
    if split_plan.dataset_id != development_protocol.development_dataset.dataset_id:
        raise ValueError(
            "retrieval outer split is not bound to the development dataset"
        )
    partition = split_plan.partition(outer_fold)
    fit_task_hashes = sorted(_task_hash(task_id) for task_id in partition.train_tasks)
    return (
        RetrievalProvenance(
            dataset_id=development_protocol.development_dataset.dataset_id,
            development_protocol_id=development_protocol.protocol_id,
            split_plan_id=split_plan.split_plan_id,
            split_seed=split_plan.seed,
            outer_fold=outer_fold,
            capability_contract_hash=capabilities.contract_hash,
            input_contract_hash=input_contract_hash,
            fit_task_set_sha256=_semantic_sha256(fit_task_hashes),
        ),
        partition.train_tasks,
    )


@dataclass(frozen=True)
class FoldFittedTfidfRetriever:
    vocabulary: tuple[str, ...]
    idf: tuple[float, ...]
    documents: tuple[_RetrievalDocument, ...]
    provenance: RetrievalProvenance
    k: int = 5
    max_features: int = 4096
    policy_id: str = RETRIEVAL_POLICY_ID
    schema_version: int = RETRIEVAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(
            self.schema_version, int
        ):
            raise TypeError("retrieval schema version must be an integer")
        if self.schema_version != RETRIEVAL_SCHEMA_VERSION:
            raise ValueError("unsupported retrieval schema version")
        if self.policy_id != RETRIEVAL_POLICY_ID:
            raise ValueError("unsupported retrieval policy")
        if (
            isinstance(self.k, bool)
            or not isinstance(self.k, int)
            or isinstance(self.max_features, bool)
            or not isinstance(self.max_features, int)
        ):
            raise TypeError("retrieval k and max_features must be integers")
        if self.k <= 0 or self.max_features <= 0:
            raise ValueError("retrieval k and max_features must be positive")
        if not self.vocabulary or len(self.vocabulary) != len(self.idf):
            raise ValueError("retrieval vocabulary and idf must be non-empty and aligned")
        if len(self.vocabulary) > self.max_features:
            raise ValueError("retrieval vocabulary exceeds max_features")
        if self.vocabulary != tuple(sorted(self.vocabulary)):
            raise ValueError("retrieval vocabulary must use canonical order")
        if len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError("retrieval vocabulary terms must be unique")
        if any(not math.isfinite(value) or value <= 0 for value in self.idf):
            raise ValueError("retrieval idf values must be finite and positive")
        task_hashes = [document.task_hash for document in self.documents]
        if not task_hashes or task_hashes != sorted(task_hashes):
            raise ValueError("retrieval documents must use canonical task-hash order")
        if len(task_hashes) != len(set(task_hashes)):
            raise ValueError("retrieval task hashes must be unique")
        for document in self.documents:
            indices = [index for index, _value in document.vector]
            if indices != sorted(indices) or len(indices) != len(set(indices)):
                raise ValueError("retrieval sparse vectors must use unique canonical indices")
            if any(index < 0 or index >= len(self.vocabulary) for index in indices):
                raise ValueError("retrieval sparse vector index is out of range")
            if any(not math.isfinite(value) or value <= 0 for _index, value in document.vector):
                raise ValueError("retrieval sparse vector values must be finite and positive")
        expected_set_hash = _semantic_sha256(task_hashes)
        if self.provenance.fit_task_set_sha256 != expected_set_hash:
            raise ValueError("retrieval fit task set hash does not match documents")

    @property
    def fit_task_set_sha256(self) -> str:
        return self.provenance.fit_task_set_sha256

    @property
    def content_hash(self) -> str:
        return _semantic_sha256(self.to_dict(include_content_hash=False))

    def to_dict(self, *, include_content_hash: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "k": self.k,
            "max_features": self.max_features,
            "provenance": self.provenance.to_dict(),
            "vocabulary": list(self.vocabulary),
            "idf": list(self.idf),
            "documents": [document.to_dict() for document in self.documents],
        }
        if include_content_hash:
            value["content_hash"] = _semantic_sha256(value)
        return value

    def to_json_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_dict())

    def validate_provenance(
        self,
        *,
        development_protocol: DevelopmentProtocol,
        split_seed: int,
        outer_fold: int,
        capabilities: SourceCapabilities,
        input_contract_hash: str,
    ) -> None:
        expected, _fit_tasks = _frozen_outer_provenance(
            development_protocol=development_protocol,
            split_seed=split_seed,
            outer_fold=outer_fold,
            capabilities=capabilities,
            input_contract_hash=input_contract_hash,
        )
        if self.provenance != expected:
            raise ValueError(
                "retrieval artifact provenance does not match the frozen outer partition"
            )

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        development_protocol: DevelopmentProtocol,
        split_seed: int,
        outer_fold: int,
        capabilities: SourceCapabilities,
        input_contract_hash: str,
    ) -> FoldFittedTfidfRetriever:
        expected = {
            "schema_version",
            "policy_id",
            "k",
            "max_features",
            "provenance",
            "vocabulary",
            "idf",
            "documents",
            "content_hash",
        }
        if set(value) != expected:
            raise ValueError("retrieval bundle has missing or extra fields")
        payload = dict(value)
        content_hash = payload.pop("content_hash")
        _required_sha256(content_hash, name="retrieval content hash")
        if content_hash != _semantic_sha256(payload):
            raise ValueError("retrieval bundle content hash mismatch")
        if (
            isinstance(payload["schema_version"], bool)
            or not isinstance(payload["schema_version"], int)
            or isinstance(payload["k"], bool)
            or not isinstance(payload["k"], int)
            or isinstance(payload["max_features"], bool)
            or not isinstance(payload["max_features"], int)
        ):
            raise TypeError(
                "retrieval schema_version, k, and max_features must be integers"
            )
        if not isinstance(payload["policy_id"], str):
            raise TypeError("retrieval policy_id must be a string")
        raw_provenance = payload["provenance"]
        if not isinstance(raw_provenance, Mapping):
            raise TypeError("retrieval provenance must be an object")
        provenance = RetrievalProvenance.from_dict(raw_provenance)
        raw_documents = payload["documents"]
        if not isinstance(raw_documents, list):
            raise TypeError("retrieval documents must be an array")
        documents: list[_RetrievalDocument] = []
        for raw in raw_documents:
            if not isinstance(raw, Mapping) or set(raw) != {
                "task_hash",
                "vector",
                "total_tokens",
                "call_count",
            }:
                raise ValueError("retrieval document has missing or extra fields")
            raw_vector = raw["vector"]
            if not isinstance(raw_vector, list):
                raise TypeError("retrieval vector must be an array")
            if not isinstance(raw["task_hash"], str):
                raise TypeError("retrieval document task_hash must be a string")
            _required_sha256(raw["task_hash"], name="retrieval document task hash")
            if any(
                isinstance(raw[name], bool)
                or not isinstance(raw[name], (int, float))
                for name in ("total_tokens", "call_count")
            ):
                raise TypeError("retrieval document totals must be numeric")
            vector: list[tuple[int, float]] = []
            for item in raw_vector:
                if (
                    not isinstance(item, list)
                    or len(item) != 2
                    or isinstance(item[0], bool)
                    or not isinstance(item[0], int)
                    or isinstance(item[1], bool)
                    or not isinstance(item[1], (int, float))
                ):
                    raise TypeError("retrieval vector entries must be [integer, number]")
                vector.append((item[0], float(item[1])))
            documents.append(
                _RetrievalDocument(
                    task_hash=raw["task_hash"],
                    vector=tuple(vector),
                    total_tokens=float(raw["total_tokens"]),
                    call_count=float(raw["call_count"]),
                )
            )
        raw_vocabulary = payload["vocabulary"]
        raw_idf = payload["idf"]
        if not isinstance(raw_vocabulary, list) or not all(
            isinstance(item, str) for item in raw_vocabulary
        ):
            raise TypeError("retrieval vocabulary must be an array of strings")
        if not isinstance(raw_idf, list) or not all(
            not isinstance(item, bool) and isinstance(item, (int, float))
            for item in raw_idf
        ):
            raise TypeError("retrieval idf must be an array of numbers")
        fitted = cls(
            vocabulary=tuple(raw_vocabulary),
            idf=tuple(float(item) for item in raw_idf),
            documents=tuple(documents),
            provenance=provenance,
            k=payload["k"],
            max_features=payload["max_features"],
            policy_id=payload["policy_id"],
            schema_version=payload["schema_version"],
        )
        fitted.validate_provenance(
            development_protocol=development_protocol,
            split_seed=split_seed,
            outer_fold=outer_fold,
            capabilities=capabilities,
            input_contract_hash=input_contract_hash,
        )
        return fitted

    @classmethod
    def from_json_bytes(
        cls,
        payload: bytes,
        *,
        development_protocol: DevelopmentProtocol,
        split_seed: int,
        outer_fold: int,
        capabilities: SourceCapabilities,
        input_contract_hash: str,
    ) -> FoldFittedTfidfRetriever:
        def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError(f"duplicate retrieval JSON key: {key!r}")
                result[key] = item
            return result

        try:
            value = json.loads(
                payload.decode(),
                object_pairs_hook=strict_object,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant: {token}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("retrieval bundle is not strict UTF-8 JSON") from exc
        if not isinstance(value, Mapping):
            raise TypeError("retrieval bundle root must be an object")
        return cls.from_dict(
            value,
            development_protocol=development_protocol,
            split_seed=split_seed,
            outer_fold=outer_fold,
            capabilities=capabilities,
            input_contract_hash=input_contract_hash,
        )

    def transform(
        self,
        *,
        task_id: str,
        task_text: str,
        development_protocol: DevelopmentProtocol,
        split_seed: int,
        outer_fold: int,
        capabilities: SourceCapabilities,
        input_contract_hash: str,
    ) -> RetrievalFeatures:
        self.validate_provenance(
            development_protocol=development_protocol,
            split_seed=split_seed,
            outer_fold=outer_fold,
            capabilities=capabilities,
            input_contract_hash=input_contract_hash,
        )
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("retrieval query task_id is required")
        if not isinstance(task_text, str) or not task_text.strip():
            raise ValueError("retrieval query requires real non-empty task text")
        vocabulary_index = {term: index for index, term in enumerate(self.vocabulary)}
        query = _sparse_vector(task_text, vocabulary_index, self.idf)
        if not query:
            return RetrievalFeatures(None, None, None, None, 0)
        query_hash = _task_hash(task_id)
        scored = [
            (_cosine(query, document.vector), document)
            for document in self.documents
            if document.task_hash != query_hash
        ]
        neighbors = [
            (similarity, document)
            for similarity, document in sorted(
                scored,
                key=lambda item: (-item[0], item[1].task_hash),
            )[: self.k]
            if similarity > 0
        ]
        if not neighbors:
            return RetrievalFeatures(None, None, None, None, 0)
        totals = [document.total_tokens for _similarity, document in neighbors]
        calls = [document.call_count for _similarity, document in neighbors]
        similarities = [similarity for similarity, _document in neighbors]
        return RetrievalFeatures(
            total_tokens_median=float(median(totals)),
            total_tokens_iqr=_quantile(totals, 0.75) - _quantile(totals, 0.25),
            call_count_median=float(median(calls)),
            mean_similarity=sum(similarities) / len(similarities),
            neighbor_count=len(neighbors),
        )

    def transform_point(
        self,
        point: PredictionPoint,
        *,
        task_text: str,
        development_protocol: DevelopmentProtocol,
        split_seed: int,
        outer_fold: int,
        capabilities: SourceCapabilities,
        input_contract_hash: str,
    ) -> PredictionPoint:
        self.validate_provenance(
            development_protocol=development_protocol,
            split_seed=split_seed,
            outer_fold=outer_fold,
            capabilities=capabilities,
            input_contract_hash=input_contract_hash,
        )
        overlap = RETRIEVAL_FEATURE_NAMES & set(point.features)
        if overlap:
            raise ValueError(
                f"point already contains retrieval features: {', '.join(sorted(overlap))}"
            )
        features = self.transform(
            task_id=point.task_id,
            task_text=task_text,
            development_protocol=development_protocol,
            split_seed=split_seed,
            outer_fold=outer_fold,
            capabilities=capabilities,
            input_contract_hash=input_contract_hash,
        )
        return point.with_features({**point.features, **features.as_point_features()})


def fit_fold_tfidf_retriever(
    examples: Iterable[RetrievalExample],
    *,
    capabilities: SourceCapabilities,
    input_contract_hash: str,
    development_protocol: DevelopmentProtocol,
    split_seed: int,
    outer_fold: int,
    k: int = 5,
    max_features: int = 4096,
) -> FoldFittedTfidfRetriever:
    provenance, fit_task_ids = _frozen_outer_provenance(
        development_protocol=development_protocol,
        split_seed=split_seed,
        outer_fold=outer_fold,
        capabilities=capabilities,
        input_contract_hash=input_contract_hash,
    )
    if k <= 0 or max_features <= 0:
        raise ValueError("retrieval k and max_features must be positive")
    resolved = tuple(examples)
    if len(resolved) < 2:
        raise ValueError("retrieval fit requires at least two tasks")
    task_ids = [example.task_id for example in resolved]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("retrieval fit task ids must be unique")
    if frozenset(task_ids) != fit_task_ids:
        non_train_count = len(frozenset(task_ids) - fit_task_ids)
        missing_train_count = len(fit_task_ids - frozenset(task_ids))
        raise ValueError(
            "retrieval examples must exactly match the frozen outer-train partition; "
            f"non_train_count={non_train_count}, "
            f"missing_train_count={missing_train_count}"
        )

    term_sets = [set(_terms(example.task_text)) for example in resolved]
    document_frequency = Counter(
        term
        for terms in term_sets
        for term in terms
    )
    ranked = sorted(document_frequency, key=lambda term: (-document_frequency[term], term))
    vocabulary = tuple(sorted(ranked[:max_features]))
    if not vocabulary:
        raise ValueError("retrieval task text produced an empty vocabulary")
    idf = tuple(
        math.log((1 + len(resolved)) / (1 + document_frequency[term])) + 1.0
        for term in vocabulary
    )
    vocabulary_index = {term: index for index, term in enumerate(vocabulary)}
    documents = tuple(
        sorted(
            (
                _RetrievalDocument(
                    task_hash=_task_hash(example.task_id),
                    vector=_sparse_vector(example.task_text, vocabulary_index, idf),
                    total_tokens=float(example.total_tokens),
                    call_count=float(example.call_count),
                )
                for example in resolved
            ),
            key=lambda document: document.task_hash,
        )
    )
    return FoldFittedTfidfRetriever(
        vocabulary=vocabulary,
        idf=idf,
        documents=documents,
        provenance=provenance,
        k=k,
        max_features=max_features,
    )


__all__ = [
    "RETRIEVAL_FEATURE_NAMES",
    "RETRIEVAL_POLICY_ID",
    "RETRIEVAL_SCHEMA_VERSION",
    "FoldFittedTfidfRetriever",
    "RetrievalExample",
    "RetrievalFeatures",
    "RetrievalProvenance",
    "fit_fold_tfidf_retriever",
]
