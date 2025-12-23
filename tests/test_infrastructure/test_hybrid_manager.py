"""
HybridModelManager 테스트 - 하이브리드 모델 관리자 테스트
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from llmkit.infrastructure.hybrid import HybridModelInfo, HybridModelManager, create_hybrid_manager


class TestHybridModelManager:
    """HybridModelManager 테스트"""

    @pytest.fixture
    def manager(self):
        """HybridModelManager 인스턴스"""
        return HybridModelManager()

    @pytest.fixture
    def mock_model_info(self):
        """Mock HybridModelInfo"""
        return HybridModelInfo(
            model_id="test-model",
            provider="openai",
            display_name="Test Model",
            source="local",
            inference_confidence=1.0,
        )

    @pytest.mark.asyncio
    async def test_load_without_scan(self, manager):
        """API 스캔 없이 로드 테스트"""
        await manager.load(scan_api=False)

        assert manager._loaded is True

    @pytest.mark.asyncio
    async def test_load_with_scan(self, manager):
        """API 스캔 포함 로드 테스트"""
        # scanner를 Mock하여 실제 API 호출 방지
        manager.scanner.scan_provider = AsyncMock(return_value=[])

        await manager.load(scan_api=True)

        assert manager._loaded is True

    def test_get_model_info_with_provider(self, manager, mock_model_info):
        """Provider 지정 모델 정보 조회 테스트"""
        # 로드 없이 테스트하려면 직접 모델 추가
        manager.models["openai"]["test-model"] = mock_model_info
        manager._loaded = True

        result = manager.get_model_info("test-model", provider="openai")

        assert result is not None
        assert result.model_id == "test-model"

    def test_get_model_info_without_provider(self, manager, mock_model_info):
        """Provider 미지정 모델 정보 조회 테스트"""
        manager.models["openai"]["test-model"] = mock_model_info
        manager._loaded = True

        result = manager.get_model_info("test-model")

        assert result is not None
        assert result.model_id == "test-model"

    def test_get_model_info_not_found(self, manager):
        """모델 정보 없음 테스트"""
        manager._loaded = True

        result = manager.get_model_info("non-existent-model")

        assert result is None

    def test_get_model_info_not_loaded(self, manager):
        """로드되지 않은 상태에서 조회 테스트"""
        manager._loaded = False

        with pytest.raises(RuntimeError, match="not loaded"):
            manager.get_model_info("test-model")

    def test_get_models_by_provider(self, manager, mock_model_info):
        """Provider별 모델 목록 조회 테스트"""
        manager.models["openai"]["test-model"] = mock_model_info
        manager._loaded = True

        models = manager.get_models_by_provider("openai")

        assert isinstance(models, list)
        assert len(models) > 0

    def test_get_models_by_provider_not_loaded(self, manager):
        """로드되지 않은 상태에서 Provider별 조회 테스트"""
        manager._loaded = False

        with pytest.raises(RuntimeError, match="not loaded"):
            manager.get_models_by_provider("openai")

    def test_get_all_models(self, manager, mock_model_info):
        """모든 모델 목록 조회 테스트"""
        manager.models["openai"]["test-model"] = mock_model_info
        manager._loaded = True

        models = manager.get_all_models()

        assert isinstance(models, list)
        assert len(models) > 0

    def test_get_new_models(self, manager):
        """신규 모델 목록 조회 테스트"""
        new_model = HybridModelInfo(
            model_id="new-model",
            provider="openai",
            display_name="New Model",
            source="inferred",
            inference_confidence=0.8,
        )
        manager.models["openai"]["new-model"] = new_model
        manager._loaded = True

        new_models = manager.get_new_models()

        assert isinstance(new_models, list)
        # inferred 소스 모델이 있는지 확인
        assert any(m.source == "inferred" for m in new_models)

    def test_get_local_models(self, manager, mock_model_info):
        """로컬 모델 목록 조회 테스트"""
        manager.models["openai"]["test-model"] = mock_model_info
        manager._loaded = True

        local_models = manager.get_local_models()

        assert isinstance(local_models, list)
        # local 소스 모델이 있는지 확인
        assert any(m.source == "local" for m in local_models)

    def test_get_total_count(self, manager, mock_model_info):
        """전체 모델 수 조회 테스트"""
        manager.models["openai"]["test-model"] = mock_model_info
        manager.models["anthropic"]["test-model-2"] = HybridModelInfo(
            model_id="test-model-2",
            provider="anthropic",
            display_name="Test Model 2",
            source="local",
            inference_confidence=1.0,
        )
        manager._loaded = True  # 로드 상태 설정

        count = manager.get_total_count()

        assert isinstance(count, int)
        assert count >= 2

    @pytest.mark.asyncio
    async def test_create_hybrid_manager(self):
        """create_hybrid_manager 팩토리 함수 테스트"""
        # create_hybrid_manager는 async 함수일 수 있음
        result = create_hybrid_manager()

        # coroutine인지 확인
        if hasattr(result, "__await__"):
            manager = await result
        else:
            manager = result

        assert isinstance(manager, HybridModelManager)


