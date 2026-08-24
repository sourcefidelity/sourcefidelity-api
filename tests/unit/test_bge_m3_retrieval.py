"""Model-free regressions for optional BGE-M3 retrieval orchestration."""

import numpy as np

from app.services.bge_m3_retrieval import (
    BgeM3HybridIndex,
    HybridPassage,
    HybridRankedPassage,
    reciprocal_rank_fusion,
)


class _FakeModel:
    def encode(self, texts, **kwargs):
        dense = []
        sparse = []
        colbert = []
        for text in texts:
            value = 1.0 if "target" in text else 0.1
            dense.append([value, 1.0 - value])
            sparse.append({"target": value})
            colbert.append(np.asarray([[value, 1.0 - value]], dtype=np.float32))
        return {
            "dense_vecs": np.asarray(dense, dtype=np.float32),
            "lexical_weights": sparse,
            "colbert_vecs": colbert,
        }

    def colbert_score(self, query, passage):
        import torch

        return torch.tensor(float(query[0] @ passage[0]))


def test_hybrid_index_ranks_matching_passage_first_and_measures_index():
    index = BgeM3HybridIndex(
        model=_FakeModel(),
        passages=[
            HybridPassage(passage_id="other", text="unrelated"),
            HybridPassage(passage_id="match", text="target evidence"),
        ],
    )

    ranking = index.rank("target query")

    assert [item.passage_id for item in ranking] == ["match", "other"]
    assert ranking[0].hybrid_score > ranking[1].hybrid_score
    assert index.build_time_ms >= 0
    assert index.index_bytes > 0


def test_rrf_rewards_passages_recovered_by_multiple_facet_queries():
    first = [
        HybridRankedPassage("shared", 1, 1, 1, 1, 1),
        HybridRankedPassage("first-only", 2, 1, 1, 1, 1),
    ]
    second = [
        HybridRankedPassage("second-only", 1, 1, 1, 1, 1),
        HybridRankedPassage("shared", 2, 1, 1, 1, 1),
    ]

    fused = reciprocal_rank_fusion([first, second])

    assert fused[0][0] == "shared"
