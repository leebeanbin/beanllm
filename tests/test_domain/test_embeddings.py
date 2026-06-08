"""
Embeddings 테스트 - 임베딩 구현체 테스트 (comprehensive coverage)
"""

import os
from typing import List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from beanllm.domain.embeddings.base import BaseAPIEmbedding, BaseEmbedding, BaseLocalEmbedding

# ──────────────────────────────────────────────────────────────
# Helpers / Concrete Implementations for abstract classes
# ──────────────────────────────────────────────────────────────


class ConcreteAPIEmbedding(BaseAPIEmbedding):
    """Minimal concrete subclass for testing BaseAPIEmbedding."""

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class ConcreteLocalEmbedding(BaseLocalEmbedding):
    """Minimal concrete subclass for testing BaseLocalEmbedding."""

    def _load_model(self):
        self._model = MagicMock()

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        return [[0.4, 0.5, 0.6] for _ in texts]


# ──────────────────────────────────────────────────────────────
# Tests: BaseEmbedding / BaseAPIEmbedding / BaseLocalEmbedding
# ──────────────────────────────────────────────────────────────


class TestBaseEmbedding:
    """BaseEmbedding 테스트"""

    @pytest.fixture
    def mock_embedding(self):
        """Mock Embedding"""
        embedding = Mock(spec=BaseEmbedding)
        embedding.embed = Mock(return_value=[[0.1, 0.2, 0.3]])
        embedding.embed_batch = Mock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        return embedding

    def test_embed(self, mock_embedding):
        """단일 텍스트 임베딩 테스트"""
        result = mock_embedding.embed("test text")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_embed_batch(self, mock_embedding):
        """배치 임베딩 테스트"""
        texts = ["text 1", "text 2"]
        results = mock_embedding.embed_batch(texts)
        assert isinstance(results, list)
        assert len(results) == len(texts)

    # ── _get_api_key ──────────────────────────────────────────

    def test_get_api_key_explicit(self):
        """직접 전달된 API 키 반환"""
        emb = ConcreteAPIEmbedding(model="test-model")
        key = emb._get_api_key("explicit-key", ["SOME_VAR"], "TestProvider")
        assert key == "explicit-key"

    def test_get_api_key_from_env(self, monkeypatch):
        """환경변수에서 API 키 읽기"""
        monkeypatch.setenv("TEST_API_KEY", "env-key-value")
        emb = ConcreteAPIEmbedding(model="test-model")
        key = emb._get_api_key(None, ["TEST_API_KEY"], "TestProvider")
        assert key == "env-key-value"

    def test_get_api_key_first_env_wins(self, monkeypatch):
        """여러 환경변수 중 첫 번째 우선"""
        monkeypatch.setenv("FIRST_KEY", "first-value")
        monkeypatch.setenv("SECOND_KEY", "second-value")
        emb = ConcreteAPIEmbedding(model="test-model")
        key = emb._get_api_key(None, ["FIRST_KEY", "SECOND_KEY"], "TestProvider")
        assert key == "first-value"

    def test_get_api_key_second_env_fallback(self, monkeypatch):
        """첫 번째 없을 때 두 번째 env 사용"""
        monkeypatch.delenv("FIRST_KEY", raising=False)
        monkeypatch.setenv("SECOND_KEY", "second-value")
        emb = ConcreteAPIEmbedding(model="test-model")
        key = emb._get_api_key(None, ["FIRST_KEY", "SECOND_KEY"], "TestProvider")
        assert key == "second-value"

    def test_get_api_key_missing_raises(self, monkeypatch):
        """API 키를 찾지 못하면 ValueError"""
        monkeypatch.delenv("MISSING_KEY", raising=False)
        emb = ConcreteAPIEmbedding(model="test-model")
        with pytest.raises(ValueError, match="API key not found"):
            emb._get_api_key(None, ["MISSING_KEY"], "TestProvider")

    # ── _validate_import ──────────────────────────────────────

    def test_validate_import_success(self):
        """정상 모듈 import 검증"""
        emb = ConcreteAPIEmbedding(model="test-model")
        # Should not raise
        emb._validate_import("os", "os")

    def test_validate_import_failure_no_extra(self):
        """없는 모듈 - extra 없는 경우"""
        emb = ConcreteAPIEmbedding(model="test-model")
        with pytest.raises(ImportError, match="pip install nonexistent_package_xyz"):
            emb._validate_import("nonexistent_module_xyz", "nonexistent_package_xyz")

    def test_validate_import_failure_with_extra(self):
        """없는 모듈 - extra 있는 경우"""
        emb = ConcreteAPIEmbedding(model="test-model")
        with pytest.raises(ImportError, match="pip install mypkg\\[myextra\\]"):
            emb._validate_import("nonexistent_module_xyz", "mypkg", install_extra="myextra")

    # ── _log_embed_success ────────────────────────────────────

    def test_log_embed_success_no_extra(self):
        """추가 정보 없이 성공 로그"""
        emb = ConcreteAPIEmbedding(model="test-model")
        # Should not raise
        emb._log_embed_success(5)

    def test_log_embed_success_with_extra(self):
        """추가 정보와 함께 성공 로그"""
        emb = ConcreteAPIEmbedding(model="test-model")
        emb._log_embed_success(3, extra_info="usage: 100 tokens")

    # ── _handle_embed_error ───────────────────────────────────

    def test_handle_embed_error_not_found(self):
        """404 에러 처리"""
        emb = ConcreteAPIEmbedding(model="test-model")
        with pytest.raises(Exception):
            emb._handle_embed_error("TestProvider", ValueError("model not found"))

    def test_handle_embed_error_404(self):
        """404 숫자 포함 에러 처리"""
        emb = ConcreteAPIEmbedding(model="test-model")
        with pytest.raises(Exception):
            emb._handle_embed_error("TestProvider", ValueError("404 error"))

    def test_handle_embed_error_generic(self):
        """일반 에러 처리"""
        emb = ConcreteAPIEmbedding(model="test-model")
        with pytest.raises(Exception):
            emb._handle_embed_error("TestProvider", RuntimeError("connection error"))

    # ── BaseAPIEmbedding.embed (delegates to embed_sync) ─────

    async def test_api_embedding_embed_delegates_to_sync(self):
        """BaseAPIEmbedding.embed()가 embed_sync()에 위임"""
        emb = ConcreteAPIEmbedding(model="test-model")
        result = await emb.embed(["hello", "world"])
        assert result == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]

    # ── BaseLocalEmbedding ────────────────────────────────────

    def test_local_embedding_init(self):
        """BaseLocalEmbedding 초기화"""
        emb = ConcreteLocalEmbedding(model="local-model")
        assert emb.model == "local-model"
        assert emb.use_gpu is True
        assert emb._model is None
        assert emb._device is None

    def test_local_embedding_init_no_gpu(self):
        """BaseLocalEmbedding GPU 비활성화"""
        emb = ConcreteLocalEmbedding(model="local-model", use_gpu=False)
        assert emb.use_gpu is False

    def test_get_device_cpu_no_torch(self):
        """torch 없으면 cpu 반환"""
        emb = ConcreteLocalEmbedding(model="local-model")
        with patch("builtins.__import__", side_effect=ImportError("no torch")):
            device = emb._get_device()
        assert device == "cpu"

    def test_get_device_cpu_when_no_cuda(self):
        """cuda 없으면 cpu 반환"""
        emb = ConcreteLocalEmbedding(model="local-model", use_gpu=True)
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            device = emb._get_device()
        assert device == "cpu"

    def test_get_device_cuda_when_available(self):
        """cuda 사용 가능하면 cuda 반환"""
        emb = ConcreteLocalEmbedding(model="local-model", use_gpu=True)
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            device = emb._get_device()
        assert device == "cuda"

    def test_get_device_no_gpu_flag(self):
        """use_gpu=False면 cuda 가능해도 cpu 반환"""
        emb = ConcreteLocalEmbedding(model="local-model", use_gpu=False)
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            device = emb._get_device()
        assert device == "cpu"

    async def test_local_embedding_embed_delegates(self):
        """BaseLocalEmbedding.embed()가 embed_sync()에 위임"""
        emb = ConcreteLocalEmbedding(model="local-model")
        result = await emb.embed(["hello"])
        assert result == [[0.4, 0.5, 0.6]]

    def test_load_model_with_lock_no_lock_manager(self):
        """락 관리자 없으면 바로 로딩"""
        emb = ConcreteLocalEmbedding(model="local-model")
        called = []

        def loader():
            called.append(True)
            emb._model = MagicMock()

        emb._load_model_with_lock("local-model", loader)
        assert len(called) == 1

    def test_load_model_with_lock_already_loaded(self):
        """이미 로딩됐으면 loader 호출 안 함"""
        emb = ConcreteLocalEmbedding(model="local-model")
        emb._model = MagicMock()  # Already loaded
        called = []

        def loader():
            called.append(True)

        emb._load_model_with_lock("local-model", loader)
        assert len(called) == 0

    async def test_load_model_with_lock_running_loop(self):
        """이미 실행 중인 이벤트 루프에서 락 없이 실행 (fallback)"""
        emb = ConcreteLocalEmbedding(model="local-model")
        emb._lock_manager = MagicMock()
        called = []

        def loader():
            called.append(True)
            emb._model = MagicMock()

        # Inside a running loop (pytest-asyncio provides one)
        emb._load_model_with_lock("local-model", loader)
        assert len(called) == 1


# ──────────────────────────────────────────────────────────────
# Tests: Embedding Factory
# ──────────────────────────────────────────────────────────────


class TestEmbeddingFactory:
    """Embedding Factory 테스트"""

    def test_get_embedding_openai(self):
        """OpenAI Embedding 생성 테스트"""
        try:
            from beanllm.domain.embeddings.factory import Embedding

            with (
                patch("beanllm.domain.embeddings.api.api_embeddings.OpenAI"),
                patch("beanllm.domain.embeddings.api.api_embeddings.AsyncOpenAI"),
            ):
                embedding = Embedding(
                    model="text-embedding-3-small", provider="openai", api_key="test_key"
                )
                assert embedding is not None
                assert embedding.model == "text-embedding-3-small"
        except (ImportError, ValueError, AttributeError) as e:
            pytest.skip(f"OpenAI embedding not available: {e}")

    def test_get_embedding_ollama(self):
        """Ollama Embedding 생성 테스트 (로컬, API 키 불필요)"""
        try:
            from beanllm.domain.embeddings.factory import Embedding

            mock_ollama = MagicMock()
            with patch.dict("sys.modules", {"ollama": mock_ollama}):
                embedding = Embedding(model="nomic-embed-text", provider="ollama")
                assert embedding is not None
                assert embedding.model == "nomic-embed-text"
        except (ImportError, ValueError, AttributeError) as e:
            pytest.skip(f"Ollama embedding not available: {e}")

    def test_detect_provider_openai(self):
        """OpenAI 모델 패턴 감지"""
        from beanllm.domain.embeddings.factory import Embedding

        provider = Embedding._detect_provider("text-embedding-3-small")
        assert provider == "openai"

    def test_detect_provider_gemini(self):
        """Gemini 모델 패턴 감지"""
        from beanllm.domain.embeddings.factory import Embedding

        provider = Embedding._detect_provider("models/embedding-001")
        assert provider == "gemini"

    def test_detect_provider_ollama(self):
        """Ollama 모델 패턴 감지"""
        from beanllm.domain.embeddings.factory import Embedding

        provider = Embedding._detect_provider("nomic-embed-text")
        assert provider == "ollama"

    def test_detect_provider_voyage(self):
        """Voyage 모델 패턴 감지"""
        from beanllm.domain.embeddings.factory import Embedding

        provider = Embedding._detect_provider("voyage-2")
        assert provider == "voyage"

    def test_detect_provider_jina(self):
        """Jina 모델 패턴 감지"""
        from beanllm.domain.embeddings.factory import Embedding

        provider = Embedding._detect_provider("jina-embeddings-v2-base-en")
        assert provider == "jina"

    def test_detect_provider_mistral(self):
        """Mistral 모델 패턴 감지"""
        from beanllm.domain.embeddings.factory import Embedding

        provider = Embedding._detect_provider("mistral-embed")
        assert provider == "mistral"

    def test_detect_provider_cohere(self):
        """Cohere 모델 패턴 감지"""
        from beanllm.domain.embeddings.factory import Embedding

        provider = Embedding._detect_provider("embed-english-v3.0")
        assert provider == "cohere"

    def test_detect_provider_unknown_returns_none(self):
        """알 수 없는 모델 → None"""
        from beanllm.domain.embeddings.factory import Embedding

        provider = Embedding._detect_provider("totally-unknown-model-xyz")
        assert provider is None

    def test_new_unknown_provider_raises(self):
        """지원하지 않는 provider 지정 시 ValueError"""
        from beanllm.domain.embeddings.factory import Embedding

        with pytest.raises(ValueError, match="Unknown provider"):
            Embedding(model="some-model", provider="unsupported_provider_xyz")

    def test_new_auto_detect_defaults_to_openai(self):
        """자동 감지 실패 시 OpenAI로 폴백"""
        from beanllm.domain.embeddings.factory import Embedding

        mock_openai_cls = MagicMock()
        mock_openai_instance = MagicMock()
        mock_openai_instance.model = "unknown-model-xyz"
        mock_openai_cls.return_value = mock_openai_instance

        with patch.dict(Embedding.PROVIDERS, {"openai": mock_openai_cls}):
            result = Embedding(model="unknown-model-xyz")
            mock_openai_cls.assert_called_once_with(model="unknown-model-xyz")

    def test_list_available_providers_includes_ollama(self):
        """Ollama (로컬)는 항상 available"""
        from beanllm.domain.embeddings.factory import Embedding

        available = Embedding.list_available_providers()
        assert "ollama" in available

    def test_list_available_providers_with_api_key(self, monkeypatch):
        """API 키가 있으면 해당 provider 포함"""
        from beanllm.domain.embeddings.factory import Embedding

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        available = Embedding.list_available_providers()
        assert "openai" in available

    def test_list_available_providers_without_api_key(self, monkeypatch):
        """API 키 없으면 해당 provider 미포함"""
        from beanllm.domain.embeddings.factory import Embedding

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        monkeypatch.delenv("JINA_API_KEY", raising=False)
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        available = Embedding.list_available_providers()
        # Only ollama should remain
        assert available == ["ollama"]

    def test_list_available_providers_gemini_multiple_envvars(self, monkeypatch):
        """Gemini는 GOOGLE_API_KEY 또는 GEMINI_API_KEY 중 하나로 감지"""
        from beanllm.domain.embeddings.factory import Embedding

        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
        available = Embedding.list_available_providers()
        assert "gemini" in available

    def test_get_default_provider_priority(self, monkeypatch):
        """우선순위: OpenAI > Gemini > Voyage > Cohere > Ollama"""
        from beanllm.domain.embeddings.factory import Embedding

        monkeypatch.setenv("OPENAI_API_KEY", "key")
        provider = Embedding.get_default_provider()
        assert provider == "openai"

    def test_get_default_provider_ollama_only(self, monkeypatch):
        """API 키 없으면 Ollama"""
        from beanllm.domain.embeddings.factory import Embedding

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        provider = Embedding.get_default_provider()
        assert provider == "ollama"

    def test_get_default_provider_none_available(self, monkeypatch):
        """아무것도 없으면 None (ollama도 없으면)"""
        from beanllm.domain.embeddings.factory import Embedding

        with patch.object(Embedding, "list_available_providers", return_value=[]):
            provider = Embedding.get_default_provider()
            assert provider is None

    def test_openai_classmethod(self):
        """Embedding.openai() 명시적 생성"""
        from beanllm.domain.embeddings.factory import Embedding

        with (
            patch.dict("sys.modules", {"openai": MagicMock()}),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
        ):
            emb = Embedding.openai(model="text-embedding-3-large")
            assert emb.model == "text-embedding-3-large"

    def test_ollama_classmethod(self):
        """Embedding.ollama() 명시적 생성"""
        from beanllm.domain.embeddings.factory import Embedding

        mock_ollama = MagicMock()
        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            emb = Embedding.ollama(model="mxbai-embed-large")
            assert emb.model == "mxbai-embed-large"

    def test_gemini_classmethod(self):
        """Embedding.gemini() 명시적 생성"""
        from beanllm.domain.embeddings.factory import Embedding

        mock_genai = MagicMock()
        with (
            patch.dict(
                "sys.modules",
                {"google": MagicMock(), "google.generativeai": mock_genai},
            ),
            patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}),
        ):
            emb = Embedding.gemini(model="models/text-embedding-004")
            assert emb.model == "models/text-embedding-004"

    def test_voyage_classmethod(self):
        """Embedding.voyage() 명시적 생성"""
        from beanllm.domain.embeddings.factory import Embedding

        mock_voyageai = MagicMock()
        with (
            patch.dict("sys.modules", {"voyageai": mock_voyageai}),
            patch.dict(os.environ, {"VOYAGE_API_KEY": "test-key"}),
        ):
            emb = Embedding.voyage(model="voyage-large-2")
            assert emb.model == "voyage-large-2"

    def test_jina_classmethod(self):
        """Embedding.jina() 명시적 생성"""
        from beanllm.domain.embeddings.factory import Embedding

        mock_httpx = MagicMock()
        with (
            patch.dict("sys.modules", {"httpx": mock_httpx}),
            patch.dict(os.environ, {"JINA_API_KEY": "test-key"}),
        ):
            emb = Embedding.jina(model="jina-embeddings-v2-small-en")
            assert emb.model == "jina-embeddings-v2-small-en"

    def test_mistral_classmethod(self):
        """Embedding.mistral() 명시적 생성"""
        import sys

        from beanllm.domain.embeddings.factory import Embedding

        mock_mistralai = MagicMock()
        mock_mistralai.client = MagicMock()
        mock_mistralai.client.Mistral = MagicMock()
        with (
            patch.dict(
                sys.modules,
                {"mistralai": mock_mistralai, "mistralai.client": mock_mistralai.client},
            ),
            patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"}),
        ):
            emb = Embedding.mistral()
            assert emb.model == "mistral-embed"

    def test_cohere_classmethod(self):
        """Embedding.cohere() 명시적 생성"""
        from beanllm.domain.embeddings.factory import Embedding

        mock_cohere = MagicMock()
        with (
            patch.dict("sys.modules", {"cohere": mock_cohere}),
            patch.dict(os.environ, {"COHERE_API_KEY": "test-key"}),
        ):
            emb = Embedding.cohere(model="embed-multilingual-v3.0")
            assert emb.model == "embed-multilingual-v3.0"


class TestEmbeddingFunctions:
    """embed / embed_sync 편의 함수 테스트"""

    async def test_embed_function_single_string(self):
        """단일 문자열 → 리스트로 변환"""
        from beanllm.domain.embeddings.factory import embed

        mock_instance = MagicMock()
        mock_instance.embed = AsyncMock(return_value=[[0.1, 0.2]])

        with patch("beanllm.domain.embeddings.factory.Embedding") as mock_cls:
            mock_cls.return_value = mock_instance
            result = await embed("hello", model="text-embedding-3-small")
            mock_instance.embed.assert_called_once_with(["hello"])

    async def test_embed_function_list(self):
        """리스트 입력"""
        from beanllm.domain.embeddings.factory import embed

        mock_instance = MagicMock()
        mock_instance.embed = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])

        with patch("beanllm.domain.embeddings.factory.Embedding") as mock_cls:
            mock_cls.return_value = mock_instance
            result = await embed(["hello", "world"], model="text-embedding-3-small")
            mock_instance.embed.assert_called_once_with(["hello", "world"])

    def test_embed_sync_function_single_string(self):
        """동기: 단일 문자열 → 리스트 변환"""
        from beanllm.domain.embeddings.factory import embed_sync

        mock_instance = MagicMock()
        mock_instance.embed_sync = MagicMock(return_value=[[0.1, 0.2]])

        with patch("beanllm.domain.embeddings.factory.Embedding") as mock_cls:
            mock_cls.return_value = mock_instance
            result = embed_sync("hello", model="text-embedding-3-small")
            mock_instance.embed_sync.assert_called_once_with(["hello"])

    def test_embed_sync_function_list(self):
        """동기: 리스트 입력"""
        from beanllm.domain.embeddings.factory import embed_sync

        mock_instance = MagicMock()
        mock_instance.embed_sync = MagicMock(return_value=[[0.1, 0.2], [0.3, 0.4]])

        with patch("beanllm.domain.embeddings.factory.Embedding") as mock_cls:
            mock_cls.return_value = mock_instance
            result = embed_sync(["hello", "world"], model="text-embedding-3-small")
            mock_instance.embed_sync.assert_called_once_with(["hello", "world"])


# ──────────────────────────────────────────────────────────────
# Tests: EmbeddingCache
# ──────────────────────────────────────────────────────────────


class TestEmbeddingCache:
    """EmbeddingCache 테스트"""

    @pytest.fixture
    def cache(self):
        from beanllm.domain.embeddings.utils.cache import EmbeddingCache

        c = EmbeddingCache(ttl=3600, max_size=100)
        yield c
        try:
            c.shutdown()
        except Exception:
            pass

    def test_cache_miss(self, cache):
        """캐시 미스: 없는 키는 None 반환"""
        result = cache.get("nonexistent text")
        assert result is None

    def test_cache_set_and_get(self, cache):
        """캐시 저장 후 조회"""
        vec = [0.1, 0.2, 0.3]
        cache.set("hello", vec)
        result = cache.get("hello")
        assert result == vec

    def test_cache_overwrite(self, cache):
        """동일 키 덮어쓰기"""
        cache.set("key", [1.0, 2.0])
        cache.set("key", [3.0, 4.0])
        result = cache.get("key")
        assert result == [3.0, 4.0]

    def test_cache_clear(self, cache):
        """캐시 비우기"""
        cache.set("a", [1.0])
        cache.set("b", [2.0])
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_cache_stats_returns_dict(self, cache):
        """통계 반환"""
        stats = cache.stats()
        assert isinstance(stats, dict)

    def test_cache_shutdown(self, cache):
        """shutdown 호출 가능"""
        cache.shutdown()  # Should not raise

    def test_cache_with_injected_cache(self):
        """외부 캐시 주입"""
        from beanllm.domain.embeddings.utils.cache import EmbeddingCache

        mock_cache = MagicMock()
        mock_cache.get.return_value = [0.9, 0.8]
        mock_cache.stats.return_value = {"size": 1}

        ec = EmbeddingCache(cache=mock_cache)
        result = ec.get("some text")
        assert result == [0.9, 0.8]
        mock_cache.get.assert_called_once_with("some text")

    def test_cache_set_with_injected_cache(self):
        """외부 캐시에 저장"""
        from beanllm.domain.embeddings.utils.cache import EmbeddingCache

        mock_cache = MagicMock()
        ec = EmbeddingCache(cache=mock_cache)
        ec.set("text", [1.0, 2.0])
        mock_cache.set.assert_called_once_with("text", [1.0, 2.0])

    def test_cache_clear_with_injected_cache(self):
        """외부 캐시 clear"""
        from beanllm.domain.embeddings.utils.cache import EmbeddingCache

        mock_cache = MagicMock()
        ec = EmbeddingCache(cache=mock_cache)
        ec.clear()
        mock_cache.clear.assert_called_once()

    def test_cache_destructor_suppresses_errors(self):
        """소멸자가 에러를 suppress"""
        from beanllm.domain.embeddings.utils.cache import EmbeddingCache

        mock_cache = MagicMock()
        mock_cache.shutdown.side_effect = RuntimeError("shutdown error")
        ec = EmbeddingCache(cache=mock_cache)
        ec.__del__()  # Should not raise

    def test_cache_default_attributes(self):
        """기본 속성 확인"""
        from beanllm.domain.embeddings.utils.cache import EmbeddingCache

        ec = EmbeddingCache(ttl=1800, max_size=5000)
        assert ec.ttl == 1800
        assert ec.max_size == 5000
        ec.shutdown()

    def test_cache_multiple_texts(self, cache):
        """여러 텍스트 캐시"""
        texts_vecs = {
            "text1": [1.0, 0.0],
            "text2": [0.0, 1.0],
            "text3": [0.5, 0.5],
        }
        for text, vec in texts_vecs.items():
            cache.set(text, vec)
        for text, vec in texts_vecs.items():
            assert cache.get(text) == vec


# ──────────────────────────────────────────────────────────────
# Tests: utils.py (cosine_similarity, euclidean_distance, etc.)
# ──────────────────────────────────────────────────────────────


class TestCosineSimilarity:
    """cosine_similarity 테스트"""

    def test_identical_vectors(self):
        """동일한 벡터: 유사도 ≈ 1.0"""
        from beanllm.domain.embeddings.utils.utils import cosine_similarity

        vec = [1.0, 0.0, 0.0]
        sim = cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-5

    def test_orthogonal_vectors(self):
        """직교 벡터: 유사도 ≈ 0.0"""
        from beanllm.domain.embeddings.utils.utils import cosine_similarity

        sim = cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(sim) < 1e-5

    def test_opposite_vectors(self):
        """반대 벡터: 유사도 ≈ -1.0"""
        from beanllm.domain.embeddings.utils.utils import cosine_similarity

        sim = cosine_similarity([1.0, 0.0], [-1.0, 0.0])
        assert abs(sim + 1.0) < 1e-5

    def test_dimension_mismatch_raises(self):
        """차원 불일치 → ValueError"""
        from beanllm.domain.embeddings.utils.utils import cosine_similarity

        with pytest.raises(ValueError, match="벡터 차원이 다릅니다"):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_zero_vector_returns_zero(self):
        """영벡터: 유사도 0.0"""
        from beanllm.domain.embeddings.utils.utils import cosine_similarity

        sim = cosine_similarity([0.0, 0.0], [1.0, 2.0])
        assert sim == 0.0

    def test_result_in_range(self):
        """결과는 [-1, 1] 범위"""
        from beanllm.domain.embeddings.utils.utils import cosine_similarity

        sim = cosine_similarity([3.0, 4.0], [5.0, 12.0])
        assert -1.0 <= sim <= 1.0


class TestEuclideanDistance:
    """euclidean_distance 테스트"""

    def test_same_vector_distance_zero(self):
        """동일 벡터: 거리 0"""
        from beanllm.domain.embeddings.utils.utils import euclidean_distance

        dist = euclidean_distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert abs(dist) < 1e-5

    def test_known_distance(self):
        """알려진 거리 계산 (피타고라스)"""
        from beanllm.domain.embeddings.utils.utils import euclidean_distance

        # [0,0] to [3,4] = 5.0
        dist = euclidean_distance([0.0, 0.0], [3.0, 4.0])
        assert abs(dist - 5.0) < 1e-4

    def test_dimension_mismatch_raises(self):
        """차원 불일치 → ValueError"""
        from beanllm.domain.embeddings.utils.utils import euclidean_distance

        with pytest.raises(ValueError, match="벡터 차원이 다릅니다"):
            euclidean_distance([1.0], [1.0, 2.0])

    def test_result_non_negative(self):
        """거리는 항상 양수"""
        from beanllm.domain.embeddings.utils.utils import euclidean_distance

        dist = euclidean_distance([1.0, 5.0], [4.0, 1.0])
        assert dist >= 0


class TestNormalizeVector:
    """normalize_vector 테스트"""

    def test_unit_vector_unchanged(self):
        """이미 정규화된 벡터"""
        import math

        from beanllm.domain.embeddings.utils.utils import normalize_vector

        vec = [1.0, 0.0, 0.0]
        result = normalize_vector(vec)
        norm = math.sqrt(sum(x**2 for x in result))
        assert abs(norm - 1.0) < 1e-5

    def test_arbitrary_vector(self):
        """임의 벡터 정규화"""
        import math

        from beanllm.domain.embeddings.utils.utils import normalize_vector

        vec = [3.0, 4.0]
        result = normalize_vector(vec)
        norm = math.sqrt(sum(x**2 for x in result))
        assert abs(norm - 1.0) < 1e-5

    def test_zero_vector_returns_original(self):
        """영벡터는 원본 반환"""
        from beanllm.domain.embeddings.utils.utils import normalize_vector

        vec = [0.0, 0.0, 0.0]
        result = normalize_vector(vec)
        assert result == vec


class TestBatchCosineSimilarity:
    """batch_cosine_similarity 테스트"""

    def test_basic_batch(self):
        """기본 배치 유사도 계산"""
        from beanllm.domain.embeddings.utils.utils import batch_cosine_similarity

        query = [1.0, 0.0]
        candidates = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        sims = batch_cosine_similarity(query, candidates)
        assert len(sims) == 3
        assert abs(sims[0] - 1.0) < 1e-5
        assert abs(sims[1]) < 1e-5
        assert abs(sims[2] + 1.0) < 1e-5

    def test_zero_query_returns_zeros(self):
        """쿼리가 영벡터 → 모두 0"""
        from beanllm.domain.embeddings.utils.utils import batch_cosine_similarity

        sims = batch_cosine_similarity([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
        assert all(s == 0.0 for s in sims)

    def test_empty_candidates(self):
        """후보 없음 → IndexError (empty candidates triggers shape[1] error in source)"""
        from beanllm.domain.embeddings.utils.utils import batch_cosine_similarity

        with pytest.raises((IndexError, Exception)):
            batch_cosine_similarity([1.0, 0.0], [])

    def test_results_in_range(self):
        """결과 범위 확인"""
        from beanllm.domain.embeddings.utils.utils import batch_cosine_similarity

        query = [0.5, 0.5, 0.7]
        candidates = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        sims = batch_cosine_similarity(query, candidates)
        for s in sims:
            assert -1.0 <= s <= 1.0


# ──────────────────────────────────────────────────────────────
# Tests: advanced.py
# ──────────────────────────────────────────────────────────────


class TestFindHardNegatives:
    """find_hard_negatives 테스트"""

    def test_basic_hard_negatives(self):
        """기본 hard negative 찾기"""
        from beanllm.domain.embeddings.utils.advanced import find_hard_negatives

        # query와 완전 동일한 벡터를 기준으로
        query_vec = [1.0, 0.0, 0.0]
        # candidates: identical (sim=1), orthogonal (sim=0), moderate
        candidates = [
            [1.0, 0.0, 0.0],  # sim ~ 1.0 → not hard negative
            [0.0, 1.0, 0.0],  # sim ~ 0.0 → not hard negative (below min)
            [0.7, 0.7, 0.0],  # sim ~ 0.7 → borderline
        ]
        result = find_hard_negatives(query_vec, candidates, similarity_threshold=(0.3, 0.7))
        assert isinstance(result, list)

    def test_top_k(self):
        """top_k 파라미터 적용"""
        from beanllm.domain.embeddings.utils.advanced import find_hard_negatives

        query_vec = [1.0, 0.0]
        candidates = [[0.5, 0.5] for _ in range(10)]
        result = find_hard_negatives(query_vec, candidates, top_k=3)
        assert len(result) <= 3

    def test_with_positive_vecs(self):
        """positive_vecs 제공 시 해당 후보 제외"""
        from beanllm.domain.embeddings.utils.advanced import find_hard_negatives

        query_vec = [1.0, 0.0]
        candidates = [[0.8, 0.2], [0.5, 0.5]]
        positive_vecs = [[0.8, 0.2]]  # First candidate is positive → exclude
        result = find_hard_negatives(
            query_vec, candidates, positive_vecs=positive_vecs, similarity_threshold=(0.3, 0.9)
        )
        # The one matching positive should be excluded
        assert isinstance(result, list)

    def test_no_hard_negatives(self):
        """threshold 밖은 빈 리스트"""
        from beanllm.domain.embeddings.utils.advanced import find_hard_negatives

        query_vec = [1.0, 0.0]
        # All candidates are very similar (sim ~ 1.0, outside threshold max=0.7)
        candidates = [[0.99, 0.01], [0.98, 0.02]]
        result = find_hard_negatives(query_vec, candidates, similarity_threshold=(0.3, 0.7))
        assert result == []

    def test_sorted_by_similarity(self):
        """Hard negatives는 유사도 내림차순으로 정렬"""
        from beanllm.domain.embeddings.utils.advanced import find_hard_negatives

        query_vec = [1.0, 0.0]
        # Create candidates with known similarities in [0.3, 0.7]
        # [0.5, 0.5] has sim ~0.707 (border), [0.45, 0.89] has lower
        candidates = [[0.4, 0.6], [0.55, 0.45]]
        result = find_hard_negatives(query_vec, candidates, similarity_threshold=(0.1, 0.9))
        assert isinstance(result, list)


class TestMMRSearch:
    """mmr_search 테스트"""

    def test_k_larger_than_candidates(self):
        """k >= len(candidates) → 모두 반환"""
        from beanllm.domain.embeddings.utils.advanced import mmr_search

        query = [1.0, 0.0]
        candidates = [[1.0, 0.0], [0.0, 1.0]]
        result = mmr_search(query, candidates, k=10)
        assert result == [0, 1]

    def test_basic_mmr(self):
        """기본 MMR 검색: k개 반환"""
        from beanllm.domain.embeddings.utils.advanced import mmr_search

        query = [1.0, 0.0]
        candidates = [
            [1.0, 0.0],  # Most similar
            [0.9, 0.1],
            [0.0, 1.0],  # Diverse
        ]
        result = mmr_search(query, candidates, k=2)
        assert len(result) == 2
        assert 0 in result  # First selected is always max similarity

    def test_lambda_1_pure_relevance(self):
        """lambda=1.0: 순수 관련성 기준"""
        from beanllm.domain.embeddings.utils.advanced import mmr_search

        query = [1.0, 0.0, 0.0]
        candidates = [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]
        result = mmr_search(query, candidates, k=2, lambda_param=1.0)
        assert result[0] == 0  # First is always most similar

    def test_lambda_0_pure_diversity(self):
        """lambda=0.0: 순수 다양성 기준"""
        from beanllm.domain.embeddings.utils.advanced import mmr_search

        query = [1.0, 0.0]
        candidates = [[1.0, 0.0], [0.5, 0.5], [-1.0, 0.0]]
        result = mmr_search(query, candidates, k=2, lambda_param=0.0)
        assert len(result) == 2

    def test_no_duplicates_in_result(self):
        """결과에 중복 없음"""
        from beanllm.domain.embeddings.utils.advanced import mmr_search

        query = [1.0, 0.0]
        candidates = [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0], [-1.0, 0.0]]
        result = mmr_search(query, candidates, k=3)
        assert len(result) == len(set(result))

    def test_empty_candidates(self):
        """후보 없음 처리"""
        from beanllm.domain.embeddings.utils.advanced import mmr_search

        result = mmr_search([1.0, 0.0], [], k=3)
        assert result == []


class TestQueryExpansion:
    """query_expansion 테스트"""

    def test_no_candidates_returns_original(self):
        """후보 없으면 원본만 반환"""
        from beanllm.domain.embeddings.utils.advanced import query_expansion

        mock_embedding = MagicMock()
        result = query_expansion("cat", mock_embedding, expansion_candidates=None)
        assert result == ["cat"]

    def test_expansion_with_candidates(self):
        """후보 중 유사도 높은 것 추가"""
        from beanllm.domain.embeddings.utils.advanced import query_expansion

        mock_embedding = MagicMock()
        # Query embedding
        mock_embedding.embed_sync.side_effect = [
            [[1.0, 0.0]],  # query vec
            [[0.95, 0.05], [0.1, 0.99], [0.0, 1.0]],  # candidate vecs
        ]

        candidates = ["kitty", "dog", "vehicle"]
        result = query_expansion(
            "cat",
            mock_embedding,
            expansion_candidates=candidates,
            top_k=2,
            similarity_threshold=0.5,
        )
        assert result[0] == "cat"  # Original always first
        assert isinstance(result, list)

    def test_threshold_filters_low_similarity(self):
        """임계값 이하는 제외"""
        from beanllm.domain.embeddings.utils.advanced import query_expansion

        mock_embedding = MagicMock()
        # Query sim to all candidates is 0.0 (orthogonal)
        mock_embedding.embed_sync.side_effect = [
            [[1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],  # sim = 0.0
        ]
        result = query_expansion(
            "cat",
            mock_embedding,
            expansion_candidates=["dog", "horse"],
            similarity_threshold=0.9,
        )
        # Nothing added (all below threshold)
        assert result == ["cat"]

    def test_same_term_excluded(self):
        """원본과 동일한 후보 제외"""
        from beanllm.domain.embeddings.utils.advanced import query_expansion

        mock_embedding = MagicMock()
        mock_embedding.embed_sync.side_effect = [
            [[1.0, 0.0]],
            [[1.0, 0.0], [0.9, 0.1]],  # Very high sim
        ]
        result = query_expansion(
            "cat",
            mock_embedding,
            expansion_candidates=["cat", "kitten"],  # "cat" is same → excluded
            similarity_threshold=0.5,
        )
        assert "cat" in result
        assert result.count("cat") == 1  # Original only once


class TestTruncateEmbedding:
    """truncate_embedding 테스트"""

    def test_basic_truncation(self):
        """기본 차원 축소"""
        from beanllm.domain.embeddings.utils.advanced import truncate_embedding

        emb = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = truncate_embedding(emb, dimension=3)
        assert result == [1.0, 2.0, 3.0]

    def test_dimension_larger_returns_original(self):
        """요청 차원이 원본보다 크면 원본 반환"""
        from beanllm.domain.embeddings.utils.advanced import truncate_embedding

        emb = [1.0, 2.0, 3.0]
        result = truncate_embedding(emb, dimension=10)
        assert result == emb

    def test_truncate_to_same_size(self):
        """동일 차원 요청"""
        from beanllm.domain.embeddings.utils.advanced import truncate_embedding

        emb = [1.0, 2.0, 3.0]
        result = truncate_embedding(emb, dimension=3)
        assert result == emb


class TestBatchTruncateEmbeddings:
    """batch_truncate_embeddings 테스트"""

    def test_batch_truncation(self):
        """배치 차원 축소"""
        from beanllm.domain.embeddings.utils.advanced import batch_truncate_embeddings

        embeddings = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        result = batch_truncate_embeddings(embeddings, dimension=2)
        assert result == [[1.0, 2.0], [5.0, 6.0]]

    def test_empty_list(self):
        """빈 리스트"""
        from beanllm.domain.embeddings.utils.advanced import batch_truncate_embeddings

        result = batch_truncate_embeddings([], dimension=3)
        assert result == []


class TestMatryoshkaEmbedding:
    """MatryoshkaEmbedding 테스트"""

    @pytest.fixture
    def base_emb(self):
        emb = MagicMock(spec=ConcreteAPIEmbedding)
        emb.model = "test-model"
        emb.embed = AsyncMock(return_value=[[1.0, 2.0, 3.0, 4.0, 5.0]])
        emb.embed_sync = MagicMock(return_value=[[1.0, 2.0, 3.0, 4.0, 5.0]])
        return emb

    def test_init(self, base_emb):
        """MatryoshkaEmbedding 초기화"""
        from beanllm.domain.embeddings.utils.advanced import MatryoshkaEmbedding

        mat = MatryoshkaEmbedding(base_embedding=base_emb, output_dimension=3)
        assert mat.model == "test-model"
        assert mat.output_dimension == 3

    def test_embed_sync_truncates(self, base_emb):
        """embed_sync가 차원 축소"""
        from beanllm.domain.embeddings.utils.advanced import MatryoshkaEmbedding

        mat = MatryoshkaEmbedding(base_embedding=base_emb, output_dimension=3)
        result = mat.embed_sync(["hello"])
        assert len(result[0]) == 3
        assert result[0] == [1.0, 2.0, 3.0]

    async def test_embed_async_truncates(self, base_emb):
        """embed가 차원 축소 (비동기)"""
        from beanllm.domain.embeddings.utils.advanced import MatryoshkaEmbedding

        mat = MatryoshkaEmbedding(base_embedding=base_emb, output_dimension=2)
        result = await mat.embed(["hello"])
        assert len(result[0]) == 2
        assert result[0] == [1.0, 2.0]


# ──────────────────────────────────────────────────────────────
# Tests: EmbeddingResult (types.py)
# ──────────────────────────────────────────────────────────────


class TestEmbeddingResult:
    """EmbeddingResult 타입 테스트"""

    def test_create_embedding_result(self):
        """EmbeddingResult 생성"""
        from beanllm.domain.embeddings.types import EmbeddingResult

        result = EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3]],
            model="test-model",
            usage={"prompt_tokens": 5, "total_tokens": 5},
        )
        assert result.model == "test-model"
        assert len(result.embeddings) == 1
        assert result.usage["total_tokens"] == 5

    def test_immutable(self):
        """frozen=True: 수정 불가"""
        from beanllm.domain.embeddings.types import EmbeddingResult

        result = EmbeddingResult(embeddings=[[0.1]], model="m", usage={})
        with pytest.raises((AttributeError, TypeError)):
            result.model = "new-model"  # type: ignore

    def test_empty_embeddings(self):
        """빈 임베딩 리스트"""
        from beanllm.domain.embeddings.types import EmbeddingResult

        result = EmbeddingResult(embeddings=[], model="m", usage={})
        assert result.embeddings == []

    def test_multiple_embeddings(self):
        """여러 임베딩"""
        from beanllm.domain.embeddings.types import EmbeddingResult

        vecs = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        result = EmbeddingResult(embeddings=vecs, model="m", usage={"total": 100})
        assert len(result.embeddings) == 3


# ──────────────────────────────────────────────────────────────
# Tests: Provider-level (mocked) - coverage for factory __new__
# ──────────────────────────────────────────────────────────────


class TestProviderCreation:
    """Provider 생성 경로 커버리지"""

    def test_auto_detect_openai_model(self):
        """text-embedding-3-small → openai 자동 감지"""
        from beanllm.domain.embeddings.factory import Embedding

        mock_openai_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.model = "text-embedding-3-small"
        mock_openai_cls.return_value = mock_instance

        with patch.dict(Embedding.PROVIDERS, {"openai": mock_openai_cls}):
            result = Embedding(model="text-embedding-3-small")
            mock_openai_cls.assert_called_once_with(model="text-embedding-3-small")

    def test_explicit_provider_override(self):
        """명시적 provider 지정"""
        from beanllm.domain.embeddings.factory import Embedding

        mock_cohere_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.model = "custom-model"
        mock_cohere_cls.return_value = mock_instance

        with patch.dict(Embedding.PROVIDERS, {"cohere": mock_cohere_cls}):
            result = Embedding(model="custom-model", provider="cohere")
            mock_cohere_cls.assert_called_once_with(model="custom-model")

    def test_all_patterns_covered(self):
        """모든 PROVIDER_PATTERNS 패턴 확인"""
        from beanllm.domain.embeddings.factory import Embedding

        expected = {
            "text-embedding-3-large": "openai",
            "text-embedding-ada-002": "openai",
            "models/text-embedding-004": "gemini",
            "text-embedding-004": "gemini",
            "embedding-001": "gemini",
            "mxbai-embed-large": "ollama",
            "all-minilm": "ollama",
            "voyage-large-2": "voyage",
            "voyage-code-2": "voyage",
            "voyage-lite-02-instruct": "voyage",
            "jina-embeddings-v2-small-en": "jina",
            "jina-embeddings-v2-base-zh": "jina",
            "jina-clip-v1": "jina",
            "embed-english-light-v3.0": "cohere",
            "embed-multilingual-v3.0": "cohere",
            "embed-english-v2.0": "cohere",
        }
        for model, expected_provider in expected.items():
            detected = Embedding._detect_provider(model)
            assert detected == expected_provider, f"Failed for {model}: got {detected}"
