"""
Callbacks - 이벤트 핸들링 시스템
로깅, 비용 추적, 타이밍, 스트리밍 등
"""
import time
from beanllm import (
    BaseCallback,
    LoggingCallback,
    CostTrackingCallback,
    TimingCallback,
    StreamingCallback,
    FunctionCallback,
    CallbackManager,
    create_callback_manager
)


def demo_logging_callback():
    """로깅 콜백"""
    print("\n" + "="*60)
    print("1️⃣  LoggingCallback - 로깅")
    print("="*60)

    callback = LoggingCallback(verbose=True)

    # 시뮬레이션
    print("\n[LLM 호출 시뮬레이션]")

    callback.on_llm_start("gpt-4o-mini", [{"role": "user", "content": "Hello"}])
    time.sleep(0.5)  # 시뮬레이션
    callback.on_llm_end("gpt-4o-mini", "Hi there!", tokens_used=15)

    print("\n[Agent 실행 시뮬레이션]")
    callback.on_agent_start("researcher", "Find information about AI")
    callback.on_agent_action("researcher", "Using search tool")
    time.sleep(0.3)
    callback.on_agent_end("researcher", "Found 5 results")

    print("\n💡 모든 이벤트를 로그로 출력!")


def demo_cost_tracking():
    """비용 추적 콜백"""
    print("\n" + "="*60)
    print("2️⃣  CostTrackingCallback - 비용 추적")
    print("="*60)

    callback = CostTrackingCallback()

    # 여러 LLM 호출 시뮬레이션
    print("\n[LLM 호출 시뮬레이션]")

    # GPT-4o-mini (저렴)
    callback.on_llm_end(
        "gpt-4o-mini",
        "Response 1",
        input_tokens=100,
        output_tokens=50
    )
    print(f"  Call 1: gpt-4o-mini (100 in + 50 out)")

    # GPT-4o (비쌈)
    callback.on_llm_end(
        "gpt-4o",
        "Response 2",
        input_tokens=200,
        output_tokens=100
    )
    print(f"  Call 2: gpt-4o (200 in + 100 out)")

    # GPT-3.5-turbo
    callback.on_llm_end(
        "gpt-3.5-turbo",
        "Response 3",
        input_tokens=150,
        output_tokens=75
    )
    print(f"  Call 3: gpt-3.5-turbo (150 in + 75 out)")

    # 통계
    print("\n[비용 통계]")
    stats = callback.get_stats()
    print(f"  총 호출:        {stats['total_calls']}번")
    print(f"  총 입력 토큰:   {stats['total_input_tokens']:,}")
    print(f"  총 출력 토큰:   {stats['total_output_tokens']:,}")
    print(f"  총 토큰:        {stats['total_tokens']:,}")
    print(f"  총 비용:        ${stats['total_cost']:.6f}")

    print("\n[호출별 비용]")
    for i, call in enumerate(stats['calls'], 1):
        print(f"  {i}. {call['model']}: ${call['cost']:.6f}")

    print("\n💡 LLM 사용 비용을 자동으로 추적!")


def demo_timing_callback():
    """타이밍 추적 콜백"""
    print("\n" + "="*60)
    print("3️⃣  TimingCallback - 실행 시간 추적")
    print("="*60)

    callback = TimingCallback()

    # 여러 호출 시뮬레이션
    print("\n[LLM 호출 시뮬레이션]")

    for i in range(3):
        callback.on_llm_start("gpt-4o-mini", [{"role": "user", "content": f"Query {i+1}"}])
        time.sleep(0.1 * (i + 1))  # 각기 다른 시간
        callback.on_llm_end("gpt-4o-mini", f"Response {i+1}")
        print(f"  Call {i+1} completed")

    # 통계
    print("\n[타이밍 통계]")
    stats = callback.get_stats()
    print(f"  총 호출:      {stats['total_calls']}번")
    print(f"  총 시간:      {stats['total_time']:.3f}초")
    print(f"  평균 시간:    {stats['average_time']:.3f}초")
    print(f"  최소 시간:    {stats['min_time']:.3f}초")
    print(f"  최대 시간:    {stats['max_time']:.3f}초")

    print("\n[호출별 시간]")
    for i, timing in enumerate(stats['timings'], 1):
        print(f"  {i}. {timing['model']}: {timing['duration']:.3f}초")

    print("\n💡 각 호출의 실행 시간을 측정!")


def demo_streaming_callback():
    """스트리밍 콜백"""
    print("\n" + "="*60)
    print("4️⃣  StreamingCallback - 스트리밍")
    print("="*60)

    # 간단한 토큰 출력
    print("\n[방법 1: 즉시 출력]")
    print("  Output: ", end="")

    callback = StreamingCallback(
        on_token=lambda token: print(token, end="", flush=True)
    )

    # 토큰 시뮬레이션
    tokens = ["Hello", " ", "World", "!", " ", "How", " ", "are", " ", "you", "?"]
    for token in tokens:
        callback.on_llm_token(token)
        time.sleep(0.05)

    callback.on_llm_end("gpt-4o-mini", "")
    print()

    # 버퍼링
    print("\n[방법 2: 버퍼링 (3토큰씩)]")
    print("  Output: ", end="")

    buffered_tokens = []

    def buffer_token(text: str):
        buffered_tokens.append(text)
        print(f"[{text}]", end=" ", flush=True)

    callback = StreamingCallback(
        on_token=buffer_token,
        buffer_size=3
    )

    for token in tokens:
        callback.on_llm_token(token)
        time.sleep(0.05)

    callback.on_llm_end("gpt-4o-mini", "")
    print()

    print(f"\n  버퍼링된 청크: {len(buffered_tokens)}개")

    print("\n💡 토큰을 실시간으로 처리!")


def demo_function_callback():
    """함수 기반 콜백"""
    print("\n" + "="*60)
    print("5️⃣  FunctionCallback - 커스텀 함수")
    print("="*60)

    # 간단한 핸들러
    print("\n[커스텀 핸들러]")

    callback = FunctionCallback(
        on_start=lambda model, **kw: print(f"  🚀 Starting: {model}"),
        on_end=lambda model, response, **kw: print(f"  ✅ Finished: {model}"),
        on_error=lambda model, error, **kw: print(f"  ❌ Error in {model}: {error}")
    )

    # 시뮬레이션
    callback.on_llm_start("gpt-4o-mini", [])
    time.sleep(0.2)
    callback.on_llm_end("gpt-4o-mini", "Response")

    # 에러
    try:
        raise ValueError("Test error")
    except Exception as e:
        callback.on_llm_error("gpt-4o", e)

    print("\n💡 간단한 함수로 커스텀 콜백!")


def demo_custom_callback():
    """커스텀 콜백 클래스"""
    print("\n" + "="*60)
    print("6️⃣  커스텀 Callback 클래스")
    print("="*60)

    class MyCustomCallback(BaseCallback):
        """커스텀 콜백 예제"""

        def __init__(self):
            self.events = []

        def on_llm_start(self, model: str, messages, **kwargs):
            event = f"LLM Start: {model}"
            self.events.append(event)
            print(f"  📝 {event}")

        def on_llm_end(self, model: str, response: str, **kwargs):
            event = f"LLM End: {model}"
            self.events.append(event)
            print(f"  📝 {event}")

        def on_agent_action(self, agent_name: str, action: str, **kwargs):
            event = f"Agent Action: {agent_name} - {action}"
            self.events.append(event)
            print(f"  📝 {event}")

        def get_event_count(self) -> int:
            return len(self.events)

    # 사용
    print("\n[커스텀 콜백 사용]")
    callback = MyCustomCallback()

    callback.on_llm_start("gpt-4o-mini", [])
    callback.on_agent_action("researcher", "searching")
    callback.on_llm_end("gpt-4o-mini", "Done")

    print(f"\n  총 이벤트: {callback.get_event_count()}개")

    print("\n💡 BaseCallback을 상속하여 완전 커스텀!")


def demo_callback_manager():
    """콜백 매니저 - 여러 콜백 한 번에"""
    print("\n" + "="*60)
    print("7️⃣  CallbackManager - 여러 콜백 관리")
    print("="*60)

    # 여러 콜백 생성
    logging_cb = LoggingCallback(verbose=True)
    cost_cb = CostTrackingCallback()
    timing_cb = TimingCallback()

    # 매니저로 관리
    manager = create_callback_manager(
        logging_cb,
        cost_cb,
        timing_cb
    )

    # 한 번에 트리거
    print("\n[LLM 호출 시뮬레이션]")

    manager.on_llm_start("gpt-4o-mini", [{"role": "user", "content": "Test"}])
    time.sleep(0.3)
    manager.on_llm_end(
        "gpt-4o-mini",
        "Response",
        input_tokens=100,
        output_tokens=50
    )

    # 각 콜백 통계
    print("\n[비용]")
    print(f"  총 비용: ${cost_cb.get_total_cost():.6f}")

    print("\n[타이밍]")
    timing_stats = timing_cb.get_stats()
    print(f"  평균 시간: {timing_stats['average_time']:.3f}초")

    print("\n💡 여러 콜백을 한 번에 관리하고 트리거!")


def demo_practical_usage():
    """실전 사용 예제"""
    print("\n" + "="*60)
    print("8️⃣  실전 사용 - Client와 통합")
    print("="*60)

    print("\n[방법 1: Client에 직접 전달]")
    print("""
    from beanllm import Client, LoggingCallback, CostTrackingCallback

    callbacks = [
        LoggingCallback(),
        CostTrackingCallback()
    ]

    client = Client(model="gpt-4o-mini", callbacks=callbacks)

    # 사용 - 자동으로 콜백 호출됨
    response = client.chat("Hello")
    """)

    print("\n[방법 2: CallbackManager 사용]")
    print("""
    from beanllm import Client, create_callback_manager
    from beanllm import LoggingCallback, CostTrackingCallback

    manager = create_callback_manager(
        LoggingCallback(),
        CostTrackingCallback()
    )

    client = Client(model="gpt-4o-mini", callback_manager=manager)

    response = client.chat("Hello")
    """)

    print("\n[방법 3: 실행 중 콜백 추가]")
    print("""
    client = Client(model="gpt-4o-mini")

    # 실행 중 콜백 추가
    client.add_callback(LoggingCallback())
    client.add_callback(CostTrackingCallback())

    response = client.chat("Hello")
    """)

    print("\n💡 Client, Agent, Chain 등 모든 곳에서 사용 가능!")


def main():
    """모든 데모 실행"""
    print("="*60)
    print("🚀 Callbacks - 이벤트 핸들링 시스템")
    print("="*60)
    print("\n8가지 콜백 타입:")
    print("  1. LoggingCallback - 로깅")
    print("  2. CostTrackingCallback - 비용 추적")
    print("  3. TimingCallback - 실행 시간")
    print("  4. StreamingCallback - 스트리밍")
    print("  5. FunctionCallback - 커스텀 함수")
    print("  6. 커스텀 Callback 클래스")
    print("  7. CallbackManager - 여러 콜백 관리")
    print("  8. 실전 사용법")

    demo_logging_callback()
    demo_cost_tracking()
    demo_timing_callback()
    demo_streaming_callback()
    demo_function_callback()
    demo_custom_callback()
    demo_callback_manager()
    demo_practical_usage()

    print("\n" + "="*60)
    print("🎉 Callbacks 데모 완료!")
    print("="*60)
    print("\n✨ 핵심 기능:")
    print("  내장 콜백:")
    print("    • LoggingCallback - 이벤트 로깅")
    print("    • CostTrackingCallback - 비용 계산")
    print("    • TimingCallback - 실행 시간")
    print("    • StreamingCallback - 실시간 처리")
    print("\n  커스텀:")
    print("    • FunctionCallback - 간단한 함수")
    print("    • BaseCallback 상속 - 완전 제어")
    print("\n  관리:")
    print("    • CallbackManager - 여러 콜백 한 번에")
    print("\n  이벤트:")
    print("    • on_llm_start/end/error/token")
    print("    • on_agent_start/end/error/action")
    print("    • on_chain_start/end/error")
    print("    • on_tool_start/end/error")
    print("\n💡 모든 실행 이벤트를 쉽게 추적하고 처리!")


if __name__ == "__main__":
    main()
