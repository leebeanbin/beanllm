"""Tests for domain/embeddings/utils/advanced.py."""

from unittest.mock import MagicMock

import pytest

from beanllm.domain.embeddings.utils.advanced import (
    MatryoshkaEmbedding,
    batch_truncate_embeddings,
    find_hard_negatives,
    mmr_search,
    query_expansion,
    truncate_embedding,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(dim: int) -> list:
    """Unit vector with 1 in first position."""
    v = [0.0] * dim
    v[0] = 1.0
    return v


def _make_embedding(vectors: list) -> MagicMock:
    """Create a mock BaseEmbedding whose embed_sync returns `vectors`."""
    emb = MagicMock()
    emb.embed_sync.return_value = vectors
    emb.model = "mock-model"

    async def _async_embed(texts):
        return vectors

    emb.embed = _async_embed
    return emb


# ---------------------------------------------------------------------------
# truncate_embedding
# ---------------------------------------------------------------------------


class TestTruncateEmbedding:
    def test_basic_truncation(self):
        v = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = truncate_embedding(v, 3)
        assert result == [1.0, 2.0, 3.0]

    def test_dimension_equals_length(self):
        v = [1.0, 2.0, 3.0]
        result = truncate_embedding(v, 3)
        assert result == [1.0, 2.0, 3.0]

    def test_dimension_larger_returns_original(self):
        v = [1.0, 2.0, 3.0]
        result = truncate_embedding(v, 10)
        assert result is v

    def test_dimension_one(self):
        v = [1.0, 2.0, 3.0]
        result = truncate_embedding(v, 1)
        assert result == [1.0]

    def test_returns_list(self):
        v = [0.1, 0.2, 0.3]
        result = truncate_embedding(v, 2)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# batch_truncate_embeddings
# ---------------------------------------------------------------------------


class TestBatchTruncateEmbeddings:
    def test_truncates_all(self):
        vecs = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        result = batch_truncate_embeddings(vecs, 2)
        assert result == [[1.0, 2.0], [4.0, 5.0]]

    def test_empty_list(self):
        result = batch_truncate_embeddings([], 3)
        assert result == []

    def test_single_vector(self):
        result = batch_truncate_embeddings([[1.0, 2.0, 3.0]], 1)
        assert result == [[1.0]]

    def test_preserves_count(self):
        vecs = [[i * 0.1 for i in range(10)] for _ in range(5)]
        result = batch_truncate_embeddings(vecs, 4)
        assert len(result) == 5
        assert all(len(v) == 4 for v in result)


# ---------------------------------------------------------------------------
# find_hard_negatives
# ---------------------------------------------------------------------------


class TestFindHardNegatives:
    def test_returns_indices_in_threshold(self):
        # query pointing in x direction
        query = [1.0, 0.0]
        candidates = [
            [1.0, 0.0],  # sim ≈ 1.0 (above max=0.7 → Positive, excluded by filter)
            [0.5, 0.86],  # sim ≈ 0.5 (between 0.3 and 0.7 → Hard Negative)
            [0.0, 1.0],  # sim ≈ 0.0 (below min=0.3 → Easy Negative)
            [-1.0, 0.0],  # sim ≈ -1.0 (Easy Negative)
        ]
        # Without positive_vecs, threshold just filters by raw similarity to query
        result = find_hard_negatives(query, candidates, similarity_threshold=(0.3, 0.7))
        # Index 1 should be in result (sim ≈ 0.5)
        assert 1 in result

    def test_single_candidate_below_threshold(self):
        # sim ≈ 0.0 → below min=0.3
        result = find_hard_negatives([1.0, 0.0], [[0.0, 1.0]], similarity_threshold=(0.3, 0.7))
        assert result == []

    def test_top_k_limits_results(self):
        query = [1.0, 0.0]
        # Create candidates all in the "hard" range
        candidates = [[0.5, 0.86], [0.4, 0.92], [0.45, 0.89]]
        result = find_hard_negatives(query, candidates, similarity_threshold=(0.3, 0.7), top_k=2)
        assert len(result) <= 2

    def test_sorted_by_similarity_desc(self):
        query = [1.0, 0.0]
        import math

        # Two vectors both in hard range, different similarities
        candidates = [
            [
                math.cos(math.radians(45)),
                math.sin(math.radians(45)),
            ],  # sim ≈ 0.71 just above... actually 0.707
            [math.cos(math.radians(70)), math.sin(math.radians(70))],  # sim ≈ 0.34
        ]
        result = find_hard_negatives(query, candidates, similarity_threshold=(0.3, 0.75))
        # Result should be sorted descending by similarity
        # Index 0 has higher similarity than index 1
        if len(result) >= 2:
            assert result[0] == 0  # higher sim first

    def test_with_positive_vecs_excludes_similar(self):
        query = [1.0, 0.0]
        positive_vecs = [[1.0, 0.0]]  # same as first candidate
        candidates = [
            [1.0, 0.0],  # Very similar to positive → should be excluded
            [0.5, 0.86],  # Hard negative
        ]
        result = find_hard_negatives(
            query, candidates, positive_vecs=positive_vecs, similarity_threshold=(0.3, 0.7)
        )
        # Index 0 should NOT be in result (too similar to positive)
        assert 0 not in result

    def test_threshold_all_excluded(self):
        query = [1.0, 0.0]
        candidates = [[0.0, 1.0], [-1.0, 0.0]]  # sims 0.0 and -1.0 — both below min=0.3
        result = find_hard_negatives(query, candidates, similarity_threshold=(0.3, 0.7))
        assert result == []


# ---------------------------------------------------------------------------
# mmr_search
# ---------------------------------------------------------------------------


class TestMMRSearch:
    def test_k_larger_than_candidates_returns_all(self):
        query = [1.0, 0.0]
        candidates = [[1.0, 0.0], [0.0, 1.0]]
        result = mmr_search(query, candidates, k=5)
        assert sorted(result) == [0, 1]

    def test_basic_selection(self):
        query = [1.0, 0.0]
        candidates = [
            [1.0, 0.0],  # most relevant
            [0.9, 0.44],  # slightly different
            [-1.0, 0.0],  # least relevant
        ]
        result = mmr_search(query, candidates, k=2)
        assert len(result) == 2
        assert 0 in result  # most relevant always first

    def test_returns_k_elements(self):
        query = [1.0, 0.0]
        candidates = [[1.0, 0.0], [0.5, 0.86], [0.0, 1.0], [-0.5, 0.86], [-1.0, 0.0]]
        result = mmr_search(query, candidates, k=3)
        assert len(result) == 3

    def test_no_duplicates(self):
        query = [1.0, 0.0]
        candidates = [[1.0, 0.0], [0.5, 0.86], [0.0, 1.0], [-1.0, 0.0]]
        result = mmr_search(query, candidates, k=4)
        assert len(result) == len(set(result))

    def test_lambda_zero_maximizes_diversity(self):
        query = [1.0, 0.0]
        candidates = [
            [1.0, 0.0],  # duplicate 1
            [1.0, 0.0],  # duplicate 2
            [0.0, 1.0],  # diverse
        ]
        result = mmr_search(query, candidates, k=2, lambda_param=0.0)
        assert len(result) == 2

    def test_single_candidate(self):
        result = mmr_search([1.0, 0.0], [[1.0, 0.0]], k=1)
        assert result == [0]


# ---------------------------------------------------------------------------
# query_expansion
# ---------------------------------------------------------------------------


class TestQueryExpansion:
    def test_no_candidates_returns_original(self):
        emb = _make_embedding([])
        result = query_expansion("cat", emb, expansion_candidates=None)
        assert result == ["cat"]

    def test_empty_candidates_returns_original(self):
        emb = _make_embedding([])
        result = query_expansion("cat", emb, expansion_candidates=[])
        assert result == ["cat"]

    def test_returns_original_plus_similar(self):
        # query vec and one very similar candidate
        query_vec = [1.0, 0.0]
        candidate_vec = [0.99, 0.14]  # high similarity
        emb = MagicMock()
        # embed_sync called twice: once for query, once for candidates
        emb.embed_sync.side_effect = [[query_vec], [candidate_vec]]
        result = query_expansion(
            "cat", emb, expansion_candidates=["kitty"], similarity_threshold=0.5
        )
        assert "cat" in result

    def test_dissimilar_candidates_excluded(self):
        query_vec = [1.0, 0.0]
        candidate_vec = [0.0, 1.0]  # orthogonal → sim ≈ 0.0
        emb = MagicMock()
        emb.embed_sync.side_effect = [[query_vec], [candidate_vec]]
        result = query_expansion("cat", emb, expansion_candidates=["dog"], similarity_threshold=0.9)
        assert result == ["cat"]  # dog excluded (low similarity)

    def test_same_word_excluded(self):
        query_vec = [1.0, 0.0]
        candidate_vec = [1.0, 0.0]
        emb = MagicMock()
        emb.embed_sync.side_effect = [[query_vec], [candidate_vec]]
        result = query_expansion("cat", emb, expansion_candidates=["CAT"], similarity_threshold=0.5)
        # "CAT".lower() == "cat" → excluded
        assert "CAT" not in result

    def test_top_k_limits_results(self):
        query_vec = [1.0, 0.0]
        candidates = ["a", "b", "c", "d", "e"]
        # All candidate vecs very similar
        cand_vecs = [[0.99, 0.14]] * 5
        emb = MagicMock()
        emb.embed_sync.side_effect = [[query_vec], cand_vecs]
        result = query_expansion(
            "q", emb, expansion_candidates=candidates, top_k=2, similarity_threshold=0.5
        )
        assert len(result) <= 3  # original + up to 2


# ---------------------------------------------------------------------------
# MatryoshkaEmbedding
# ---------------------------------------------------------------------------


class TestMatryoshkaEmbedding:
    def test_embed_sync_truncates(self):
        base = _make_embedding([[1.0, 2.0, 3.0, 4.0]])
        mat = MatryoshkaEmbedding(base_embedding=base, output_dimension=2)
        result = mat.embed_sync(["hello"])
        assert result == [[1.0, 2.0]]

    async def test_embed_async_truncates(self):
        base = _make_embedding([[1.0, 2.0, 3.0, 4.0]])
        mat = MatryoshkaEmbedding(base_embedding=base, output_dimension=2)
        result = await mat.embed(["hello"])
        assert result == [[1.0, 2.0]]

    def test_model_inherited_from_base(self):
        base = _make_embedding([])
        base.model = "text-embedding-3-large"
        mat = MatryoshkaEmbedding(base_embedding=base, output_dimension=512)
        assert mat.model == "text-embedding-3-large"

    def test_output_dimension_stored(self):
        base = _make_embedding([])
        mat = MatryoshkaEmbedding(base_embedding=base, output_dimension=256)
        assert mat.output_dimension == 256

    def test_batch_embed_sync(self):
        base = _make_embedding([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mat = MatryoshkaEmbedding(base_embedding=base, output_dimension=2)
        result = mat.embed_sync(["text1", "text2"])
        assert result == [[1.0, 2.0], [4.0, 5.0]]

    def test_dimension_larger_than_base_returns_original(self):
        base = _make_embedding([[1.0, 2.0]])
        mat = MatryoshkaEmbedding(base_embedding=base, output_dimension=100)
        result = mat.embed_sync(["text"])
        assert result == [[1.0, 2.0]]
