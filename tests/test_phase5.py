"""
Phase 5 통합 테스트
Tools, Agent, Memory, Chain 테스트
"""
from llmkit import (
    Tool,
    ToolRegistry,
    register_tool,
    Agent,
    AgentStep,
    AgentResult,
    BufferMemory,
    WindowMemory,
    TokenMemory,
    ConversationMemory,
    create_memory,
    Chain,
    PromptChain,
    ChainBuilder,
)


# ============================================================================
# Tool Tests
# ============================================================================

def test_tool_from_function():
    """Tool.from_function 테스트"""
    def add(a: int, b: int) -> int:
        """두 수를 더함"""
        return a + b

    tool = Tool.from_function(add)

    assert tool.name == "add"
    assert tool.description == "두 수를 더함"
    assert len(tool.parameters) == 2

    result = tool.execute({"a": 5, "b": 3})
    assert result == 8


def test_tool_registry():
    """ToolRegistry 테스트"""
    registry = ToolRegistry()

    @registry.register
    def multiply(x: float, y: float) -> float:
        """곱하기"""
        return x * y

    assert "multiply" in [t.name for t in registry.get_all()]

    result = registry.execute("multiply", {"x": 4, "y": 5})
    assert result == 20


def test_tool_openai_format():
    """OpenAI 형식 변환 테스트"""
    def search(query: str) -> str:
        """검색"""
        return f"Results for {query}"

    tool = Tool.from_function(search)
    openai_format = tool.to_openai_format()

    assert openai_format["type"] == "function"
    assert openai_format["function"]["name"] == "search"
    assert "query" in openai_format["function"]["parameters"]["properties"]


def test_tool_anthropic_format():
    """Anthropic 형식 변환 테스트"""
    def calculator(a: float, b: float) -> float:
        """계산"""
        return a + b

    tool = Tool.from_function(calculator)
    anthropic_format = tool.to_anthropic_format()

    assert anthropic_format["name"] == "calculator"
    assert "a" in anthropic_format["input_schema"]["properties"]
    assert "b" in anthropic_format["input_schema"]["properties"]


# ============================================================================
# Memory Tests
# ============================================================================

def test_buffer_memory():
    """BufferMemory 테스트"""
    memory = BufferMemory(max_messages=5)

    memory.add_message("user", "안녕")
    memory.add_message("assistant", "반가워요")
    memory.add_message("user", "날씨는?")

    assert len(memory) == 3
    messages = memory.get_messages()
    assert messages[0].role == "user"
    assert messages[0].content == "안녕"


def test_buffer_memory_max_limit():
    """BufferMemory 최대 제한 테스트"""
    memory = BufferMemory(max_messages=3)

    for i in range(10):
        memory.add_message("user", f"Message {i}")

    assert len(memory) == 3
    messages = memory.get_messages()
    assert messages[0].content == "Message 7"


def test_window_memory():
    """WindowMemory 테스트"""
    memory = WindowMemory(window_size=5)

    for i in range(10):
        memory.add_message("user", f"Message {i}")

    assert len(memory) == 5
    messages = memory.get_messages()
    assert messages[0].content == "Message 5"
    assert messages[-1].content == "Message 9"


def test_token_memory():
    """TokenMemory 테스트"""
    memory = TokenMemory(max_tokens=100)

    memory.add_message("user", "짧은 메시지")
    memory.add_message("assistant", "응답")

    assert len(memory) >= 2


def test_conversation_memory():
    """ConversationMemory 테스트"""
    memory = ConversationMemory(max_pairs=3)

    memory.add_user_message("질문 1")
    memory.add_ai_message("답변 1")
    memory.add_user_message("질문 2")
    memory.add_ai_message("답변 2")

    pairs = memory.get_conversation_pairs()
    assert len(pairs) == 2
    assert pairs[0][0].content == "질문 1"
    assert pairs[0][1].content == "답변 1"


def test_create_memory_factory():
    """create_memory 팩토리 테스트"""
    buffer = create_memory("buffer", max_messages=10)
    assert isinstance(buffer, BufferMemory)

    window = create_memory("window", window_size=5)
    assert isinstance(window, WindowMemory)

    token = create_memory("token", max_tokens=1000)
    assert isinstance(token, TokenMemory)


def test_memory_clear():
    """메모리 초기화 테스트"""
    memory = BufferMemory()
    memory.add_message("user", "test")
    assert len(memory) == 1

    memory.clear()
    assert len(memory) == 0


def test_memory_dict_messages():
    """get_dict_messages 테스트"""
    memory = BufferMemory()
    memory.add_message("user", "안녕")
    memory.add_message("assistant", "반가워요")

    dict_msgs = memory.get_dict_messages()
    assert len(dict_msgs) == 2
    assert dict_msgs[0]["role"] == "user"
    assert dict_msgs[0]["content"] == "안녕"


# ============================================================================
# Integration Tests
# ============================================================================

def test_memory_with_messages():
    """메모리 메시지 통합 테스트"""
    memory = ConversationMemory(max_pairs=10)

    # 대화 추가
    for i in range(5):
        memory.add_user_message(f"Question {i}")
        memory.add_ai_message(f"Answer {i}")

    # 검증
    messages = memory.get_messages()
    assert len(messages) == 10

    pairs = memory.get_conversation_pairs()
    assert len(pairs) == 5


def test_tool_parameter_types():
    """Tool 파라미터 타입 테스트"""
    def typed_func(
        text: str,
        count: int,
        ratio: float,
        enabled: bool,
        items: list,
        config: dict
    ) -> str:
        """타입이 있는 함수"""
        return "result"

    tool = Tool.from_function(typed_func)

    params = {p.name: p.type for p in tool.parameters}
    assert params["text"] == "string"
    assert params["count"] == "number"
    assert params["ratio"] == "number"
    assert params["enabled"] == "boolean"
    assert params["items"] == "array"
    assert params["config"] == "object"


def test_multiple_tools():
    """여러 도구 관리 테스트"""
    registry = ToolRegistry()

    @registry.register
    def tool1(x: int) -> int:
        return x * 2

    @registry.register
    def tool2(x: int) -> int:
        return x + 10

    tools = registry.get_all()
    assert len(tools) >= 2

    assert registry.execute("tool1", {"x": 5}) == 10
    assert registry.execute("tool2", {"x": 5}) == 15


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    tests = [
        ('Tool from_function', test_tool_from_function),
        ('Tool registry', test_tool_registry),
        ('Tool OpenAI format', test_tool_openai_format),
        ('Tool Anthropic format', test_tool_anthropic_format),
        ('BufferMemory', test_buffer_memory),
        ('BufferMemory max limit', test_buffer_memory_max_limit),
        ('WindowMemory', test_window_memory),
        ('TokenMemory', test_token_memory),
        ('ConversationMemory', test_conversation_memory),
        ('create_memory factory', test_create_memory_factory),
        ('Memory clear', test_memory_clear),
        ('Memory dict messages', test_memory_dict_messages),
        ('Memory with messages', test_memory_with_messages),
        ('Tool parameter types', test_tool_parameter_types),
        ('Multiple tools', test_multiple_tools),
    ]

    print('Running Phase 5 Tests...')
    print('=' * 60)

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            print(f'✅ {name}')
            passed += 1
        except Exception as e:
            print(f'❌ {name}: {e}')
            failed += 1

    print('=' * 60)
    print(f'\nResults: {passed} passed, {failed} failed')

    if failed == 0:
        print('🎉 All tests passed!')
    else:
        print(f'⚠️  {failed} test(s) failed')
