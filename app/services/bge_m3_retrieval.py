"""Optional local BGE-M3 hybrid passage discovery for shadow ablations.

The production retrieval path does not import FlagEmbedding.  This module
accepts an already constructed model so the heavyweight optional dependency
and model lifecycle remain explicit.  It encodes a source once, scores exact
candidate/facet queries through dense, learned sparse and ColBERT-style
multi-vector signals, and exposes only passage IDs and scores.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class HybridPassage:
    passage_id: str
    text: str


@dataclass(frozen=True)
class HybridRankedPassage:
    passage_id: str
    rank: int
    hybrid_score: float
    dense_score: float
    sparse_score: float
    colbert_score: float


class BgeM3HybridIndex:
    """One reusable source-local BGE-M3 hybrid index."""

    def __init__(
        self,
        *,
        model: Any,
        passages: Sequence[HybridPassage],
        batch_size: int = 2,
        passage_max_length: int = 512,
    ) -> None:
        if not passages:
            raise ValueError("A BGE-M3 source index requires at least one passage")
        passage_ids = [item.passage_id for item in passages]
        if len(passage_ids) != len(set(passage_ids)):
            raise ValueError("BGE-M3 source-index passage IDs must be unique")
        self.model = model
        self.passages = list(passages)
        started = time.monotonic()
        encoded = model.encode(
            [item.text for item in passages],
            batch_size=batch_size,
            max_length=passage_max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )
        self.dense = np.asarray(encoded["dense_vecs"])
        self.sparse = list(encoded["lexical_weights"])
        self.colbert = list(encoded["colbert_vecs"])
        self.build_time_ms = (time.monotonic() - started) * 1000
        self.index_bytes = _embedding_bytes(self.dense, self.sparse, self.colbert)

    def rank(
        self,
        query: str,
        *,
        limit: int = 10,
        query_max_length: int = 256,
    ) -> list[HybridRankedPassage]:
        if not query.strip():
            raise ValueError("A BGE-M3 retrieval query cannot be blank")
        encoded = self.model.encode(
            [query],
            batch_size=1,
            max_length=query_max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )
        query_dense = np.asarray(encoded["dense_vecs"])[0]
        query_sparse = list(encoded["lexical_weights"])[0]
        query_colbert = list(encoded["colbert_vecs"])[0]

        dense_scores = self.dense @ query_dense
        sparse_scores = np.asarray(
            [
                _lexical_score(query_sparse, passage_sparse)
                for passage_sparse in self.sparse
            ],
            dtype=float,
        )
        colbert_scores = np.asarray(
            [
                float(self.model.colbert_score(query_colbert, passage).item())
                for passage in self.colbert
            ],
            dtype=float,
        )
        hybrid_scores = (dense_scores + sparse_scores + colbert_scores) / 3.0
        order = sorted(
            range(len(self.passages)),
            key=lambda index: (-float(hybrid_scores[index]), self.passages[index].passage_id),
        )[: max(1, min(limit, len(self.passages)))]
        return [
            HybridRankedPassage(
                passage_id=self.passages[index].passage_id,
                rank=rank,
                hybrid_score=float(hybrid_scores[index]),
                dense_score=float(dense_scores[index]),
                sparse_score=float(sparse_scores[index]),
                colbert_score=float(colbert_scores[index]),
            )
            for rank, index in enumerate(order, start=1)
        ]


def reciprocal_rank_fusion(
    rankings: Iterable[Sequence[HybridRankedPassage]],
    *,
    constant: int = 60,
) -> list[tuple[str, float]]:
    """Fuse independent exact/facet query rankings without source text."""
    if constant < 1:
        raise ValueError("RRF constant must be positive")
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for item in ranking:
            if item.passage_id in seen:
                raise ValueError("One BGE-M3 query ranking repeated a passage ID")
            seen.add(item.passage_id)
            scores[item.passage_id] = scores.get(item.passage_id, 0.0) + (
                1.0 / (constant + item.rank)
            )
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _lexical_score(left: dict[str, float], right: dict[str, float]) -> float:
    return float(sum(weight * right.get(token, 0.0) for token, weight in left.items()))


def _embedding_bytes(
    dense: np.ndarray,
    sparse: Sequence[dict[str, float]],
    colbert: Sequence[np.ndarray],
) -> int:
    # Sparse dictionaries need a measured in-memory estimate because the model
    # returns Python token/weight mappings rather than a packed matrix.
    sparse_bytes = sum(
        sys.getsizeof(item)
        + sum(sys.getsizeof(token) + sys.getsizeof(weight) for token, weight in item.items())
        for item in sparse
    )
    return int(dense.nbytes + sparse_bytes + sum(item.nbytes for item in colbert))
