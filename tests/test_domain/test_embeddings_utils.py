"""Tests for domain/embeddings: base classes and math utilities."""

import math
import os
from unittest.mock import MagicMock, patch

import pytest

from beanllm.domain.embeddings.base import BaseAPIEmbedding, BaseEmbedding, BaseLocalEmbedding
from beanllm.domain.embeddings.utils.utils import (
    batch_cosine_similarity,
    cosine_similarity,
    euclidean_distance,
    normalize_vector,
)

# ---------------------------------------------------------------------------
# Concrete subclasses for testing abstract bases
# ---------------------------------------------------------------------------


class _ConcreteAPIEmb(BaseAPIEmbedding):
    def embed_sync(self, texts):
        return [[0.1, 0.2] for _ in texts]


class _ConcreteLocalEmb(BaseLocalEmbedding):
    def _load_model(self):
        self._model = MagicMock()

    def embed_sync(self, texts):
        return [[0.3, 0.4] for _ in texts]

    async def embed(self, texts):
        return [[0.3, 0.4] for _ in texts]


# ---------------------------------------------------------------------------
# BaseEmbedding._get_api_key
# ---------------------------------------------------------------------------


class TestGetApiKey:
    def setup_method(self):
        self.emb = _ConcreteAPIEmb(model="test-model")

    def test_direct_key_used(self):
        key = self.emb._get_api_key("sk-direct", ["OPENAI_API_KEY"], "OpenAI")
        assert key == "sk-direct"

    def test_env_var_fallback(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env"}):
            key = self.emb._get_api_key(None, ["OPENAI_API_KEY"], "OpenAI")
        assert key == "sk-env"

    def test_second_env_var_fallback(self):
        env = {k: v for k, v in os.environ.items() if k not in ("KEY_A", "KEY_B")}
        env["KEY_B"] = "sk-b"
        with patch.dict(os.environ, env, clear=True):
            key = self.emb._get_api_key(None, ["KEY_A", "KEY_B"], "Provider")
        assert key == "sk-b"

    def test_no_key_raises(self):
        env = {k: v for k, v in os.environ.items() if k not in ("MY_KEY",)}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="MY_KEY"):
                self.emb._get_api_key(None, ["MY_KEY"], "MyProvider")


# ---------------------------------------------------------------------------
# BaseEmbedding._validate_import
# ---------------------------------------------------------------------------


class TestValidateImport:
    def setup_method(self):
        self.emb = _ConcreteAPIEmb(model="test-model")

    def test_existing_module_ok(self):
        # json is always available
        self.emb._validate_import("json", "json")

    def test_missing_module_raises_import_error(self):
        with pytest.raises(ImportError, match="nonexistent_module_xyz"):
            self.emb._validate_import("nonexistent_module_xyz", "nonexistent-pkg")

    def test_missing_with_extra_shows_bracket(self):
        with pytest.raises(ImportError, match=r"\[myextra\]"):
            self.emb._validate_import("nonexistent_xyz", "beanllm", install_extra="myextra")

    def test_missing_without_extra_shows_plain_install(self):
        with pytest.raises(ImportError, match="pip install somepackage"):
            self.emb._validate_import("nonexistent_xyz2", "somepackage")


# ---------------------------------------------------------------------------
# BaseAPIEmbedding.embed (delegates to embed_sync)
# ---------------------------------------------------------------------------


class TestBaseAPIEmbeddingEmbed:
    async def test_embed_delegates_to_embed_sync(self):
        emb = _ConcreteAPIEmb(model="test-model")
        result = await emb.embed(["hello", "world"])
        assert result == [[0.1, 0.2], [0.1, 0.2]]


# ---------------------------------------------------------------------------
# BaseLocalEmbedding
# ---------------------------------------------------------------------------


class TestBaseLocalEmbedding:
    def test_init_defaults(self):
        emb = _ConcreteLocalEmb(model="local-model")
        assert emb.use_gpu is True
        assert emb._model is None
        assert emb._device is None

    def test_init_no_gpu(self):
        emb = _ConcreteLocalEmb(model="local-model", use_gpu=False)
        assert emb.use_gpu is False

    async def test_embed(self):
        emb = _ConcreteLocalEmb(model="local-model")
        result = await emb.embed(["a", "b"])
        assert result == [[0.3, 0.4], [0.3, 0.4]]

    def test_embed_sync(self):
        emb = _ConcreteLocalEmb(model="local-model")
        result = emb.embed_sync(["x"])
        assert result == [[0.3, 0.4]]


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        sim = cosine_similarity(v, v)
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        sim = cosine_similarity(v1, v2)
        assert sim == pytest.approx(0.0, abs=1e-5)

    def test_opposite_vectors(self):
        v1 = [1.0, 0.0]
        v2 = [-1.0, 0.0]
        sim = cosine_similarity(v1, v2)
        assert sim == pytest.approx(-1.0, abs=1e-5)

    def test_similar_vectors(self):
        v1 = [1.0, 1.0]
        v2 = [1.0, 0.9]
        sim = cosine_similarity(v1, v2)
        assert 0.99 < sim <= 1.0

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="차원"):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_zero_vector_returns_zero(self):
        sim = cosine_similarity([0.0, 0.0], [1.0, 0.0])
        assert sim == 0.0

    def test_result_bounded(self):
        v1 = [0.5, 0.3, 0.7]
        v2 = [0.1, 0.9, 0.2]
        sim = cosine_similarity(v1, v2)
        assert -1.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# euclidean_distance
# ---------------------------------------------------------------------------


class TestEuclideanDistance:
    def test_same_vector_zero_distance(self):
        v = [1.0, 2.0, 3.0]
        dist = euclidean_distance(v, v)
        assert dist == pytest.approx(0.0, abs=1e-5)

    def test_known_distance(self):
        v1 = [0.0, 0.0]
        v2 = [3.0, 4.0]
        dist = euclidean_distance(v1, v2)
        assert dist == pytest.approx(5.0, abs=1e-4)

    def test_unit_vectors(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        dist = euclidean_distance(v1, v2)
        assert dist == pytest.approx(math.sqrt(2), abs=1e-4)

    def test_dimension_mismatch_raises(self):
        with pytest.raises((ValueError, Exception)):
            euclidean_distance([1.0, 2.0], [1.0])

    def test_non_negative(self):
        v1 = [1.0, -2.0, 3.0]
        v2 = [-1.0, 2.0, -3.0]
        dist = euclidean_distance(v1, v2)
        assert dist >= 0


# ---------------------------------------------------------------------------
# normalize_vector
# ---------------------------------------------------------------------------


class TestNormalizeVector:
    def test_unit_vector_unchanged(self):
        v = [1.0, 0.0, 0.0]
        result = normalize_vector(v)
        norm = math.sqrt(sum(x**2 for x in result))
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_arbitrary_vector(self):
        v = [3.0, 4.0]
        result = normalize_vector(v)
        norm = math.sqrt(sum(x**2 for x in result))
        assert norm == pytest.approx(1.0, abs=1e-5)
        assert result[0] == pytest.approx(0.6, abs=1e-5)
        assert result[1] == pytest.approx(0.8, abs=1e-5)

    def test_zero_vector_returns_original(self):
        v = [0.0, 0.0, 0.0]
        result = normalize_vector(v)
        assert result == v

    def test_negative_components(self):
        v = [-1.0, 0.0]
        result = normalize_vector(v)
        norm = math.sqrt(sum(x**2 for x in result))
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_returns_list(self):
        result = normalize_vector([1.0, 2.0, 3.0])
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# batch_cosine_similarity
# ---------------------------------------------------------------------------


class TestBatchCosineSimilarity:
    def test_returns_correct_count(self):
        query = [1.0, 0.0]
        candidates = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        scores = batch_cosine_similarity(query, candidates)
        assert len(scores) == 3

    def test_identical_scores_highest(self):
        query = [1.0, 0.0]
        candidates = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        scores = batch_cosine_similarity(query, candidates)
        assert scores[0] > scores[1] > scores[2]

    def test_single_candidate(self):
        query = [1.0, 0.0]
        scores = batch_cosine_similarity(query, [[1.0, 0.0]])
        assert len(scores) == 1
        assert scores[0] == pytest.approx(1.0, abs=1e-4)

    def test_scores_bounded(self):
        import random

        query = [random.uniform(-1, 1) for _ in range(8)]
        candidates = [[random.uniform(-1, 1) for _ in range(8)] for _ in range(10)]
        scores = batch_cosine_similarity(query, candidates)
        assert all(-1.0 <= s <= 1.0 for s in scores)
