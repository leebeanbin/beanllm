"""
Callbacks 테스트 - 콜백 시스템 테스트
"""

import pytest
from unittest.mock import Mock, AsyncMock

from beanllm.utils.callbacks import (
    BaseCallback,
    CallbackManager,
    LoggingCallback,
    CostTrackingCallback,
    TimingCallback,
    StreamingCallback,
    FunctionCallback,
    create_callback_manager,
    CallbackEvent,
)


class TestBaseCallback:
    """BaseCallback 테스트"""

    def test_base_callback_instantiation(self):
        """BaseCallback 인스턴스화 테스트"""
        # BaseCallback은 추상 클래스가 아니므로 인스턴스화 가능
        callback = BaseCallback()
        assert callback is not None


class TestLoggingCallback:
    """LoggingCallback 테스트"""

    @pytest.fixture
    def logging_callback(self):
        """LoggingCallback 인스턴스"""
        return LoggingCallback()

    def test_logging_callback_on_llm_start(self, logging_callback):
        """LoggingCallback LLM 시작 이벤트 처리 테스트"""
        # 에러 없이 실행되어야 함
        logging_callback.on_llm_start(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "test"}]
        )


class TestCostTrackingCallback:
    """CostTrackingCallback 테스트"""

    @pytest.fixture
    def cost_callback(self):
        """CostTrackingCallback 인스턴스"""
        return CostTrackingCallback()

    def test_cost_callback_on_llm_end(self, cost_callback):
        """CostTrackingCallback LLM 종료 이벤트 처리 테스트"""
        cost_callback.on_llm_end(
            model="gpt-4o-mini",
            response="Test response",
            input_tokens=100,
            output_tokens=50,
        )

        # 총 비용 확인
        total_cost = cost_callback.get_total_cost()
        assert isinstance(total_cost, float)
        assert total_cost >= 0

    def test_cost_callback_get_stats(self, cost_callback):
        """CostTrackingCallback 통계 조회 테스트"""
        cost_callback.on_llm_end(
            model="gpt-4o-mini",
            response="Test response",
            input_tokens=100,
            output_tokens=50,
        )

        stats = cost_callback.get_stats()

        assert isinstance(stats, dict)
        assert "total_cost" in stats


class TestTimingCallback:
    """TimingCallback 테스트"""

    @pytest.fixture
    def timing_callback(self):
        """TimingCallback 인스턴스"""
        return TimingCallback()

    def test_timing_callback_on_llm_start_end(self, timing_callback):
        """TimingCallback LLM 시작/종료 이벤트 처리 테스트"""
        timing_callback.on_llm_start(model="gpt-4o-mini", messages=[])

        timing_callback.on_llm_end(model="gpt-4o-mini", response="Test response")

        # 통계 확인
        stats = timing_callback.get_stats()
        assert isinstance(stats, dict)


class TestStreamingCallback:
    """StreamingCallback 테스트"""

    @pytest.fixture
    def streaming_callback(self):
        """StreamingCallback 인스턴스"""
        return StreamingCallback()

    def test_streaming_callback_on_llm_token(self, streaming_callback):
        """StreamingCallback 토큰 이벤트 처리 테스트"""
        # StreamingCallback은 on_llm_token을 통해 토큰을 수집
        streaming_callback.on_llm_token(token="test")

        # StreamingCallback은 버퍼를 사용하므로 정상 작동 확인
        assert streaming_callback is not None


class TestFunctionCallback:
    """FunctionCallback 테스트"""

    def test_function_callback_on_llm_start(self):
        """FunctionCallback LLM 시작 이벤트 처리 테스트"""
        call_count = 0

        def test_func(model, messages, **kwargs):
            nonlocal call_count
            call_count += 1

        callback = FunctionCallback(test_func)

        callback.on_llm_start(model="gpt-4o-mini", messages=[{"role": "user", "content": "test"}])

        assert call_count == 1


class TestCallbackManager:
    """CallbackManager 테스트"""

    @pytest.fixture
    def callback_manager(self):
        """CallbackManager 인스턴스"""
        return CallbackManager()

    def test_callback_manager_add_callback(self, callback_manager):
        """콜백 추가 테스트"""
        callback = LoggingCallback()
        callback_manager.add_callback(callback)

        assert len(callback_manager.callbacks) == 1

    def test_callback_manager_trigger(self, callback_manager):
        """이벤트 트리거 테스트"""
        callback = Mock(spec=BaseCallback)
        callback_manager.add_callback(callback)

        callback_manager.trigger("on_llm_start", model="gpt-4o-mini", messages=[])

        callback.on_llm_start.assert_called_once_with(model="gpt-4o-mini", messages=[])

    def test_callback_manager_remove_callback(self, callback_manager):
        """콜백 제거 테스트"""
        callback = LoggingCallback()
        callback_manager.add_callback(callback)
        callback_manager.remove_callback(callback)

        assert len(callback_manager.callbacks) == 0


class TestCreateCallbackManager:
    """create_callback_manager 테스트"""

    def test_create_callback_manager(self):
        """CallbackManager 생성 테스트"""
        manager = create_callback_manager()

        assert isinstance(manager, CallbackManager)


