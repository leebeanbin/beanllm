"""
Callbacks - 이벤트 핸들링 시스템
LLM, Agent, Chain 실행 중 이벤트 처리
"""

import time
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


@dataclass
class CallbackEvent:
    """콜백 이벤트"""

    event_type: str  # start, end, error, token, etc.
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)


class BaseCallback(ABC):
    """
    콜백 베이스 클래스

    모든 콜백은 이 클래스를 상속받아 구현
    """

    def on_llm_start(self, model: str, messages: List[Dict[str, Any]], **kwargs):
        """LLM 호출 시작"""
        pass

    def on_llm_end(self, model: str, response: str, tokens_used: Optional[int] = None, **kwargs):
        """LLM 호출 종료"""
        pass

    def on_llm_error(self, model: str, error: Exception, **kwargs):
        """LLM 호출 에러"""
        pass

    def on_llm_token(self, token: str, **kwargs):
        """LLM 토큰 생성 (스트리밍)"""
        pass

    def on_agent_start(self, agent_name: str, task: str, **kwargs):
        """Agent 실행 시작"""
        pass

    def on_agent_end(self, agent_name: str, result: Any, **kwargs):
        """Agent 실행 종료"""
        pass

    def on_agent_error(self, agent_name: str, error: Exception, **kwargs):
        """Agent 실행 에러"""
        pass

    def on_agent_action(self, agent_name: str, action: str, **kwargs):
        """Agent 액션 (도구 사용 등)"""
        pass

    def on_chain_start(self, chain_name: str, inputs: Dict[str, Any], **kwargs):
        """Chain 실행 시작"""
        pass

    def on_chain_end(self, chain_name: str, outputs: Dict[str, Any], **kwargs):
        """Chain 실행 종료"""
        pass

    def on_chain_error(self, chain_name: str, error: Exception, **kwargs):
        """Chain 실행 에러"""
        pass

    def on_tool_start(self, tool_name: str, inputs: Dict[str, Any], **kwargs):
        """도구 실행 시작"""
        pass

    def on_tool_end(self, tool_name: str, result: Any, **kwargs):
        """도구 실행 종료"""
        pass

    def on_tool_error(self, tool_name: str, error: Exception, **kwargs):
        """도구 실행 에러"""
        pass


class LoggingCallback(BaseCallback):
    """
    로깅 콜백

    모든 이벤트를 로그로 출력

    Example:
        callback = LoggingCallback(verbose=True)
        client = Client(callbacks=[callback])
    """

    def __init__(self, verbose: bool = True):
        """
        Args:
            verbose: 상세 로그 출력
        """
        self.verbose = verbose

    def _log(self, message: str):
        """로그 출력"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")

    def on_llm_start(self, model: str, messages: List[Dict[str, Any]], **kwargs):
        self._log(f"🚀 LLM Start: {model}")

    def on_llm_end(self, model: str, response: str, tokens_used: Optional[int] = None, **kwargs):
        token_info = f" ({tokens_used} tokens)" if tokens_used else ""
        self._log(f"✅ LLM End: {model}{token_info}")

    def on_llm_error(self, model: str, error: Exception, **kwargs):
        self._log(f"❌ LLM Error: {model} - {error}")

    def on_agent_start(self, agent_name: str, task: str, **kwargs):
        self._log(f"🤖 Agent Start: {agent_name}")

    def on_agent_end(self, agent_name: str, result: Any, **kwargs):
        self._log(f"✅ Agent End: {agent_name}")

    def on_agent_action(self, agent_name: str, action: str, **kwargs):
        self._log(f"⚡ Agent Action: {action}")

    def on_chain_start(self, chain_name: str, inputs: Dict[str, Any], **kwargs):
        self._log(f"🔗 Chain Start: {chain_name}")

    def on_chain_end(self, chain_name: str, outputs: Dict[str, Any], **kwargs):
        self._log(f"✅ Chain End: {chain_name}")


class CostTrackingCallback(BaseCallback):
    """
    비용 추적 콜백

    LLM 사용 비용 계산 및 추적

    Example:
        callback = CostTrackingCallback()
        client = Client(callbacks=[callback])

        # 사용 후
        print(f"Total cost: ${callback.get_total_cost():.4f}")
    """

    # 모델별 가격 (per 1M tokens)
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.150, "output": 0.600},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "claude-3-opus": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
    }

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

    def on_llm_end(
        self,
        model: str,
        response: str,
        tokens_used: Optional[int] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        **kwargs,
    ):
        """LLM 호출 종료 시 비용 계산"""
        # 토큰 수
        input_tok = input_tokens or 0
        output_tok = output_tokens or 0

        # 비용 계산
        cost = 0.0
        if model in self.PRICING:
            pricing = self.PRICING[model]
            cost = (input_tok / 1_000_000) * pricing["input"] + (output_tok / 1_000_000) * pricing[
                "output"
            ]

        # 기록
        self.calls.append(
            {
                "model": model,
                "input_tokens": input_tok,
                "output_tokens": output_tok,
                "cost": cost,
                "timestamp": datetime.now(),
            }
        )

        self.total_input_tokens += input_tok
        self.total_output_tokens += output_tok
        self.total_cost += cost

    def get_total_cost(self) -> float:
        """총 비용"""
        return self.total_cost

    def get_total_tokens(self) -> int:
        """총 토큰 수"""
        return self.total_input_tokens + self.total_output_tokens

    def get_stats(self) -> Dict[str, Any]:
        """통계"""
        return {
            "total_calls": len(self.calls),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.get_total_tokens(),
            "total_cost": self.total_cost,
            "calls": self.calls,
        }

    def reset(self):
        """통계 초기화"""
        self.calls.clear()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0


class TimingCallback(BaseCallback):
    """
    타이밍 추적 콜백

    각 호출의 실행 시간 측정

    Example:
        callback = TimingCallback()
        client = Client(callbacks=[callback])

        # 사용 후
        stats = callback.get_stats()
        print(f"Average time: {stats['average_time']:.2f}s")
    """

    def __init__(self):
        self.start_times: Dict[str, float] = {}
        self.timings: List[Dict[str, Any]] = []

    def on_llm_start(self, model: str, messages: List[Dict[str, Any]], **kwargs):
        """시작 시간 기록"""
        call_id = f"llm_{model}_{time.time()}"
        self.start_times[call_id] = time.time()
        kwargs["_call_id"] = call_id

    def on_llm_end(self, model: str, response: str, **kwargs):
        """종료 시간 및 duration 계산"""
        call_id = kwargs.get("_call_id")
        if call_id and call_id in self.start_times:
            duration = time.time() - self.start_times[call_id]

            self.timings.append(
                {"type": "llm", "model": model, "duration": duration, "timestamp": datetime.now()}
            )

            del self.start_times[call_id]

    def get_stats(self) -> Dict[str, Any]:
        """통계"""
        if not self.timings:
            return {
                "total_calls": 0,
                "total_time": 0.0,
                "average_time": 0.0,
                "min_time": 0.0,
                "max_time": 0.0,
            }

        durations = [t["duration"] for t in self.timings]

        return {
            "total_calls": len(self.timings),
            "total_time": sum(durations),
            "average_time": sum(durations) / len(durations),
            "min_time": min(durations),
            "max_time": max(durations),
            "timings": self.timings,
        }

    def reset(self):
        """통계 초기화"""
        self.start_times.clear()
        self.timings.clear()


class StreamingCallback(BaseCallback):
    """
    스트리밍 콜백

    토큰을 실시간으로 처리

    Example:
        def print_token(token: str):
            print(token, end="", flush=True)

        callback = StreamingCallback(on_token=print_token)
        client = Client(callbacks=[callback])
    """

    def __init__(self, on_token: Optional[Callable[[str], None]] = None, buffer_size: int = 1):
        """
        Args:
            on_token: 토큰 처리 함수
            buffer_size: 버퍼 크기 (여러 토큰을 모아서 처리)
        """
        self.on_token_func = on_token
        self.buffer_size = buffer_size
        self.buffer: List[str] = []

    def on_llm_token(self, token: str, **kwargs):
        """토큰 처리"""
        self.buffer.append(token)

        # 버퍼가 차면 처리
        if len(self.buffer) >= self.buffer_size:
            self._flush_buffer()

    def _flush_buffer(self):
        """버퍼 비우기"""
        if self.buffer and self.on_token_func:
            text = "".join(self.buffer)
            self.on_token_func(text)
            self.buffer.clear()

    def on_llm_end(self, model: str, response: str, **kwargs):
        """종료 시 남은 버퍼 비우기"""
        self._flush_buffer()


class FunctionCallback(BaseCallback):
    """
    함수 기반 콜백

    커스텀 함수를 쉽게 콜백으로 사용

    Example:
        callback = FunctionCallback(
            on_start=lambda model, **kw: print(f"Start: {model}"),
            on_end=lambda model, response, **kw: print(f"End: {model}")
        )
    """

    def __init__(
        self,
        on_start: Optional[Callable] = None,
        on_end: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        on_token: Optional[Callable] = None,
        **custom_handlers,
    ):
        """
        Args:
            on_start: 시작 핸들러
            on_end: 종료 핸들러
            on_error: 에러 핸들러
            on_token: 토큰 핸들러
            **custom_handlers: 커스텀 핸들러
        """
        self.handlers = {
            "start": on_start,
            "end": on_end,
            "error": on_error,
            "token": on_token,
            **custom_handlers,
        }

    def on_llm_start(self, model: str, messages: List[Dict[str, Any]], **kwargs):
        if self.handlers.get("start"):
            self.handlers["start"](model=model, messages=messages, **kwargs)

    def on_llm_end(self, model: str, response: str, **kwargs):
        if self.handlers.get("end"):
            self.handlers["end"](model=model, response=response, **kwargs)

    def on_llm_error(self, model: str, error: Exception, **kwargs):
        if self.handlers.get("error"):
            self.handlers["error"](model=model, error=error, **kwargs)

    def on_llm_token(self, token: str, **kwargs):
        if self.handlers.get("token"):
            self.handlers["token"](token=token, **kwargs)


class CallbackManager:
    """
    콜백 관리자

    여러 콜백을 한 번에 관리

    Example:
        manager = CallbackManager([
            LoggingCallback(),
            CostTrackingCallback(),
            TimingCallback()
        ])

        client = Client(callback_manager=manager)
    """

    def __init__(self, callbacks: Optional[List[BaseCallback]] = None):
        """
        Args:
            callbacks: 콜백 리스트
        """
        self.callbacks = callbacks or []

    def add_callback(self, callback: BaseCallback):
        """콜백 추가"""
        self.callbacks.append(callback)

    def remove_callback(self, callback: BaseCallback):
        """콜백 제거"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def trigger(self, event: str, **kwargs):
        """
        이벤트 트리거

        Args:
            event: 이벤트 이름 (e.g., "on_llm_start")
            **kwargs: 이벤트 파라미터
        """
        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if method and callable(method):
                try:
                    method(**kwargs)
                except Exception as e:
                    # 콜백 에러가 전체 실행을 막지 않도록
                    print(f"Callback error in {event}: {e}")

    # Convenience methods
    def on_llm_start(self, model: str, messages: List[Dict[str, Any]], **kwargs):
        self.trigger("on_llm_start", model=model, messages=messages, **kwargs)

    def on_llm_end(self, model: str, response: str, **kwargs):
        self.trigger("on_llm_end", model=model, response=response, **kwargs)

    def on_llm_error(self, model: str, error: Exception, **kwargs):
        self.trigger("on_llm_error", model=model, error=error, **kwargs)

    def on_llm_token(self, token: str, **kwargs):
        self.trigger("on_llm_token", token=token, **kwargs)

    def on_agent_start(self, agent_name: str, task: str, **kwargs):
        self.trigger("on_agent_start", agent_name=agent_name, task=task, **kwargs)

    def on_agent_end(self, agent_name: str, result: Any, **kwargs):
        self.trigger("on_agent_end", agent_name=agent_name, result=result, **kwargs)

    def on_agent_error(self, agent_name: str, error: Exception, **kwargs):
        self.trigger("on_agent_error", agent_name=agent_name, error=error, **kwargs)

    def on_agent_action(self, agent_name: str, action: str, **kwargs):
        self.trigger("on_agent_action", agent_name=agent_name, action=action, **kwargs)

    def on_chain_start(self, chain_name: str, inputs: Dict[str, Any], **kwargs):
        self.trigger("on_chain_start", chain_name=chain_name, inputs=inputs, **kwargs)

    def on_chain_end(self, chain_name: str, outputs: Dict[str, Any], **kwargs):
        self.trigger("on_chain_end", chain_name=chain_name, outputs=outputs, **kwargs)

    def on_chain_error(self, chain_name: str, error: Exception, **kwargs):
        self.trigger("on_chain_error", chain_name=chain_name, error=error, **kwargs)

    def on_tool_start(self, tool_name: str, inputs: Dict[str, Any], **kwargs):
        self.trigger("on_tool_start", tool_name=tool_name, inputs=inputs, **kwargs)

    def on_tool_end(self, tool_name: str, result: Any, **kwargs):
        self.trigger("on_tool_end", tool_name=tool_name, result=result, **kwargs)

    def on_tool_error(self, tool_name: str, error: Exception, **kwargs):
        self.trigger("on_tool_error", tool_name=tool_name, error=error, **kwargs)


# 편의 함수
def create_callback_manager(*callbacks: BaseCallback) -> CallbackManager:
    """
    CallbackManager 생성 (간편 함수)

    Example:
        manager = create_callback_manager(
            LoggingCallback(),
            CostTrackingCallback()
        )
    """
    return CallbackManager(list(callbacks))
