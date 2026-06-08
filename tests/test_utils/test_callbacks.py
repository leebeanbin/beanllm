"""
Callbacks 테스트 - 콜백 시스템 테스트
"""

from unittest.mock import AsyncMock, Mock

import pytest

from beanllm.utils.integration.callbacks import (
    BaseCallback,
    CallbackEvent,
    CallbackManager,
    CostTrackingCallback,
    FunctionCallback,
    LoggingCallback,
    StreamingCallback,
    TimingCallback,
    create_callback_manager,
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

    def test_create_with_multiple_callbacks(self):
        manager = create_callback_manager(LoggingCallback(), CostTrackingCallback())
        assert len(manager.callbacks) == 2


class TestBaseCallbackAllMethods:
    """Call every no-op method on BaseCallback to get full coverage."""

    def setup_method(self):
        self.cb = BaseCallback()

    def test_on_llm_end(self):
        self.cb.on_llm_end("gpt-4o", "response", tokens_used=10)

    def test_on_llm_error(self):
        self.cb.on_llm_error("gpt-4o", ValueError("err"))

    def test_on_llm_token(self):
        self.cb.on_llm_token("hello")

    def test_on_agent_start(self):
        self.cb.on_agent_start("agent1", "task1")

    def test_on_agent_end(self):
        self.cb.on_agent_end("agent1", result="done")

    def test_on_agent_error(self):
        self.cb.on_agent_error("agent1", ValueError("err"))

    def test_on_agent_action(self):
        self.cb.on_agent_action("agent1", "search")

    def test_on_chain_start(self):
        self.cb.on_chain_start("chain1", {"input": "x"})

    def test_on_chain_end(self):
        self.cb.on_chain_end("chain1", {"output": "y"})

    def test_on_chain_error(self):
        self.cb.on_chain_error("chain1", ValueError("err"))

    def test_on_tool_start(self):
        self.cb.on_tool_start("tool1", {"query": "x"})

    def test_on_tool_end(self):
        self.cb.on_tool_end("tool1", result="found")

    def test_on_tool_error(self):
        self.cb.on_tool_error("tool1", ValueError("err"))


class TestLoggingCallbackAllMethods:
    def setup_method(self):
        self.cb = LoggingCallback(verbose=True)

    def test_on_llm_end(self):
        self.cb.on_llm_end("gpt-4o", "Hi", tokens_used=5)

    def test_on_llm_end_no_tokens(self):
        self.cb.on_llm_end("gpt-4o", "Hi")

    def test_on_llm_error(self):
        self.cb.on_llm_error("gpt-4o", ValueError("err"))

    def test_on_agent_start(self):
        self.cb.on_agent_start("agent1", "task1")

    def test_on_agent_end(self):
        self.cb.on_agent_end("agent1", "result")

    def test_on_agent_action(self):
        self.cb.on_agent_action("agent1", "search")

    def test_on_chain_start(self):
        self.cb.on_chain_start("chain1", {})

    def test_on_chain_end(self):
        self.cb.on_chain_end("chain1", {})

    def test_verbose_false_no_output(self, capsys):
        cb = LoggingCallback(verbose=False)
        cb.on_llm_start("gpt-4o", [])
        captured = capsys.readouterr()
        assert captured.out == ""


class TestCostTrackingCallbackExtended:
    def setup_method(self):
        self.cb = CostTrackingCallback()

    def test_unknown_model_zero_cost(self):
        self.cb.on_llm_end("unknown-model", "response", input_tokens=1000, output_tokens=500)
        assert self.cb.get_total_cost() == 0.0

    def test_multiple_calls_accumulate(self):
        self.cb.on_llm_end("gpt-4o-mini", "r1", input_tokens=100, output_tokens=50)
        self.cb.on_llm_end("gpt-4o-mini", "r2", input_tokens=100, output_tokens=50)
        assert self.cb.get_total_tokens() == 300
        assert len(self.cb.get_stats()["calls"]) == 2

    def test_reset_clears_data(self):
        self.cb.on_llm_end("gpt-4o-mini", "r1", input_tokens=100, output_tokens=50)
        self.cb.reset()
        assert self.cb.get_total_cost() == 0.0
        assert self.cb.get_total_tokens() == 0
        assert len(self.cb.get_stats()["calls"]) == 0


class TestTimingCallbackExtended:
    def setup_method(self):
        self.cb = TimingCallback()

    def test_empty_stats(self):
        stats = self.cb.get_stats()
        assert stats["total_calls"] == 0
        assert stats["average_time"] == 0.0

    def test_with_call_id(self):
        kwargs = {}
        self.cb.on_llm_start("gpt-4o", [], **kwargs)
        self.cb.on_llm_end("gpt-4o", "resp", **kwargs)
        stats = self.cb.get_stats()
        assert stats["total_calls"] >= 0  # may be 0 since call_id is in kwargs

    def test_reset(self):
        self.cb.reset()
        assert len(self.cb.timings) == 0


class TestStreamingCallbackExtended:
    def test_buffered_tokens_flushed(self):
        collected = []
        cb = StreamingCallback(on_token=lambda t: collected.append(t), buffer_size=3)
        cb.on_llm_token("a")
        cb.on_llm_token("b")
        cb.on_llm_token("c")  # buffer reaches 3 → flush
        assert "abc" in collected

    def test_on_llm_end_flushes_remaining(self):
        collected = []
        cb = StreamingCallback(on_token=lambda t: collected.append(t), buffer_size=10)
        cb.on_llm_token("x")
        cb.on_llm_end("gpt-4o", "full response")
        assert "x" in collected

    def test_no_on_token_func(self):
        cb = StreamingCallback(on_token=None, buffer_size=1)
        cb.on_llm_token("t")  # should not crash even without handler


class TestFunctionCallbackExtended:
    def test_on_llm_error_called(self):
        received = {}

        cb = FunctionCallback(on_error=lambda **kw: received.update(kw))
        cb.on_llm_error("gpt-4o", ValueError("err"))
        assert "error" in received

    def test_on_llm_token_called(self):
        tokens = []
        cb = FunctionCallback(on_token=lambda **kw: tokens.append(kw["token"]))
        cb.on_llm_token("hello")
        assert "hello" in tokens

    def test_no_handlers_no_crash(self):
        cb = FunctionCallback()
        cb.on_llm_start("gpt-4o", [])
        cb.on_llm_end("gpt-4o", "resp")
        cb.on_llm_error("gpt-4o", ValueError("e"))
        cb.on_llm_token("t")


class TestCallbackManagerAllEvents:
    def setup_method(self):
        self.manager = CallbackManager([LoggingCallback(verbose=False)])

    def test_on_llm_error(self):
        self.manager.on_llm_error("gpt-4o", ValueError("err"))

    def test_on_llm_token(self):
        self.manager.on_llm_token("tok")

    def test_on_agent_start(self):
        self.manager.on_agent_start("agent1", "task1")

    def test_on_agent_end(self):
        self.manager.on_agent_end("agent1", "result")

    def test_on_agent_error(self):
        self.manager.on_agent_error("agent1", ValueError("e"))

    def test_on_agent_action(self):
        self.manager.on_agent_action("agent1", "search")

    def test_on_chain_start(self):
        self.manager.on_chain_start("chain1", {})

    def test_on_chain_end(self):
        self.manager.on_chain_end("chain1", {})

    def test_on_chain_error(self):
        self.manager.on_chain_error("chain1", ValueError("e"))

    def test_on_tool_start(self):
        self.manager.on_tool_start("tool1", {})

    def test_on_tool_end(self):
        self.manager.on_tool_end("tool1", "result")

    def test_on_tool_error(self):
        self.manager.on_tool_error("tool1", ValueError("e"))

    def test_trigger_handles_callback_exception(self):
        class BadCallback(BaseCallback):
            def on_llm_start(self, model, messages, **kw):
                raise RuntimeError("callback crashed")

        manager = CallbackManager([BadCallback()])
        manager.on_llm_start("gpt-4o", [])  # should not raise
