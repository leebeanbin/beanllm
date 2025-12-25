"""
Phase 5 통합 데모: Tools, Agent, Memory, Chain
LangChain 스타일의 고급 기능 시연
"""
import asyncio
from beanllm import (
    Client,
    # Tools
    Tool,
    ToolRegistry,
    register_tool,
    # Agent
    Agent,
    create_agent,
    # Memory
    BufferMemory,
    WindowMemory,
    TokenMemory,
    ConversationMemory,
    create_memory,
    # Chain
    Chain,
    PromptChain,
    SequentialChain,
    ParallelChain,
    ChainBuilder,
    create_chain,
)


# ============================================================================
# 1. Tool System 데모
# ============================================================================

def demo_tools():
    """Tool 시스템 기본 사용"""
    print("="*60)
    print("1. Tool System Demo")
    print("="*60)

    # 도구 정의
    @register_tool
    def get_weather(city: str) -> str:
        """도시의 날씨 정보 가져오기"""
        # 실제로는 API 호출
        return f"{city}의 날씨는 맑음, 기온 20도입니다"

    @register_tool
    def translate(text: str, target_lang: str) -> str:
        """텍스트 번역"""
        return f"[{target_lang}로 번역됨] {text}"

    # 도구 실행
    from beanllm.tools import get_all_tools
    tools = get_all_tools()
    print(f"\n등록된 도구: {len(tools)}개")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    # 직접 실행
    result = get_weather(city="서울")
    print(f"\n실행 결과: {result}")


# ============================================================================
# 2. Agent 데모
# ============================================================================

async def demo_agent():
    """ReAct 에이전트 사용"""
    print("\n" + "="*60)
    print("2. Agent Demo (ReAct Pattern)")
    print("="*60)

    # 커스텀 도구 정의
    def calculator(operation: str, a: float, b: float) -> float:
        """계산기"""
        ops = {
            "add": lambda x, y: x + y,
            "multiply": lambda x, y: x * y,
        }
        return ops.get(operation, lambda x, y: 0)(a, b)

    def search(query: str) -> str:
        """웹 검색 시뮬레이션"""
        if "서울 인구" in query:
            return "서울의 인구는 약 950만명입니다"
        return f"'{query}'에 대한 검색 결과"

    # 에이전트 생성
    agent = Agent(
        model="gpt-4o-mini",
        tools=[
            Tool.from_function(calculator),
            Tool.from_function(search)
        ],
        max_iterations=5,
        verbose=True
    )

    # 복잡한 작업 수행
    print("\n작업: 서울 인구를 찾고, 2를 곱해줘")
    result = await agent.run("서울 인구를 찾고, 2를 곱해줘")

    print(f"\n최종 답변: {result.answer}")
    print(f"실행 단계: {result.total_steps}단계")
    print(f"성공 여부: {result.success}")


# ============================================================================
# 3. Memory 데모
# ============================================================================

async def demo_memory():
    """메모리 시스템 사용"""
    print("\n" + "="*60)
    print("3. Memory System Demo")
    print("="*60)

    # (1) BufferMemory
    print("\n[BufferMemory] 모든 메시지 저장")
    memory = BufferMemory(max_messages=10)
    memory.add_message("user", "안녕하세요")
    memory.add_message("assistant", "안녕하세요! 무엇을 도와드릴까요?")
    memory.add_message("user", "날씨 알려줘")
    print(f"  저장된 메시지: {len(memory)}개")

    # (2) WindowMemory
    print("\n[WindowMemory] 최근 N개만 유지")
    window = WindowMemory(window_size=3)
    for i in range(10):
        window.add_message("user", f"메시지 {i}")
    print(f"  저장된 메시지: {len(window)}개 (최근 3개만)")

    # (3) TokenMemory
    print("\n[TokenMemory] 토큰 수 제한")
    token_memory = TokenMemory(max_tokens=100)
    token_memory.add_message("user", "짧은 메시지")
    token_memory.add_message("assistant", "응답 메시지")
    print(f"  저장된 메시지: {len(token_memory)}개")

    # (4) ConversationMemory
    print("\n[ConversationMemory] 대화 쌍 관리")
    conv = ConversationMemory(max_pairs=5)
    conv.add_user_message("첫 번째 질문")
    conv.add_ai_message("첫 번째 답변")
    conv.add_user_message("두 번째 질문")
    conv.add_ai_message("두 번째 답변")
    pairs = conv.get_conversation_pairs()
    print(f"  대화 쌍: {len(pairs)}개")


# ============================================================================
# 4. Chain 데모
# ============================================================================

async def demo_chain():
    """체인 시스템 사용"""
    print("\n" + "="*60)
    print("4. Chain System Demo")
    print("="*60)

    client = Client(model="gpt-4o-mini")

    # (1) 기본 Chain
    print("\n[Basic Chain]")
    chain = Chain(client, memory=BufferMemory())
    result = await chain.run("Python이란?")
    print(f"  응답: {result.output[:100]}...")

    # (2) PromptChain
    print("\n[Prompt Chain] 템플릿 사용")
    template = """
    You are a {role}.
    Answer briefly: {question}
    """
    prompt_chain = PromptChain(client, template)
    result = await prompt_chain.run(
        role="Python expert",
        question="What is async/await?"
    )
    print(f"  응답: {result.output[:100]}...")

    # (3) ChainBuilder (Fluent API)
    print("\n[Chain Builder] Fluent API")
    result = await (
        ChainBuilder(client)
        .with_memory("window", window_size=5)
        .with_template("Explain {topic} in one sentence")
        .run(topic="Docker")
    )
    print(f"  응답: {result.output}")

    # (4) SequentialChain
    print("\n[Sequential Chain] 순차 실행")
    chain1 = PromptChain(client, "Generate 3 topics about {subject}")
    chain2 = PromptChain(client, "Choose the best one from: {input}")

    seq = SequentialChain([chain1, chain2])
    result = await seq.run(subject="AI")
    print(f"  최종 결과: {result.output[:100]}...")

    # (5) ParallelChain
    print("\n[Parallel Chain] 병렬 실행")
    chains = [
        PromptChain(client, "Analyze {topic} from technical view (one sentence)"),
        PromptChain(client, "Analyze {topic} from business view (one sentence)"),
        PromptChain(client, "Analyze {topic} from user view (one sentence)"),
    ]

    parallel = ParallelChain(chains)
    result = await parallel.run(topic="ChatGPT")
    print(f"  병렬 실행 결과 {len(result.metadata.get('outputs', []))}개:")
    for i, output in enumerate(result.metadata.get("outputs", [])):
        print(f"    {i+1}. {output[:80]}...")


# ============================================================================
# 5. 통합 시나리오
# ============================================================================

async def demo_integrated():
    """모든 기능을 통합한 실전 시나리오"""
    print("\n" + "="*60)
    print("5. Integrated Scenario")
    print("="*60)
    print("시나리오: 메모리를 가진 에이전트 + 체인")

    # 도구 정의
    def search_docs(query: str) -> str:
        """문서 검색"""
        docs = {
            "python": "Python은 간결하고 읽기 쉬운 프로그래밍 언어입니다",
            "docker": "Docker는 컨테이너 기반 가상화 플랫폼입니다",
        }
        for key, value in docs.items():
            if key in query.lower():
                return value
        return "관련 문서를 찾을 수 없습니다"

    # 메모리가 있는 에이전트
    agent = Agent(
        model="gpt-4o-mini",
        tools=[Tool.from_function(search_docs)],
        max_iterations=3,
        verbose=False
    )

    # 체인과 결합
    print("\n질문 1: Python에 대해 알려줘")
    result1 = await agent.run("Python에 대해 알려줘")
    print(f"답변 1: {result1.answer}")

    print("\n질문 2: Docker는 뭐야?")
    result2 = await agent.run("Docker는 뭐야?")
    print(f"답변 2: {result2.answer}")


# ============================================================================
# Main
# ============================================================================

async def main():
    """전체 데모 실행"""
    print("\n" + "🚀" * 30)
    print("Phase 5 통합 데모: Tools, Agent, Memory, Chain")
    print("🚀" * 30)

    # 1. Tools
    demo_tools()

    # 2. Agent
    # await demo_agent()  # API 키 필요

    # 3. Memory
    await demo_memory()

    # 4. Chain
    # await demo_chain()  # API 키 필요

    # 5. Integrated
    # await demo_integrated()  # API 키 필요

    print("\n" + "✅" * 30)
    print("데모 완료!")
    print("✅" * 30)


if __name__ == "__main__":
    asyncio.run(main())
