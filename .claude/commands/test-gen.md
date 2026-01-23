# /test-gen - Test Generation

**트리거**: `/test-gen`
**모델**: sonnet
**설명**: TDD 기반 테스트 자동 생성

## Command Description

코드에 대한 단위 테스트, 통합 테스트, E2E 테스트를 자동으로 생성합니다. 80% 커버리지 목표를 달성하기 위한 테스트 케이스를 제안합니다.

## Usage

```
/test-gen
/test-gen --path src/beanllm/facade/core/client_facade.py
/test-gen --type unit
/test-gen --coverage-goal 90
```

## Options

- `--path`: 테스트할 파일 경로 (기본: 현재 파일)
- `--type`: 테스트 타입 (`unit`, `integration`, `e2e`, `all`) (기본: `unit`)
- `--coverage-goal`: 목표 커버리지 (기본: 80%)
- `--fixtures`: pytest fixtures 자동 생성

## Execution Steps

### 1. 코드 분석

```python
import ast
from pathlib import Path

def analyze_code(file_path: str):
    """코드에서 테스트할 함수/메서드 추출"""
    with open(file_path) as f:
        tree = ast.parse(f.read())

    functions = []
    classes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append({
                "name": node.name,
                "args": [arg.arg for arg in node.args.args],
                "returns": ast.unparse(node.returns) if node.returns else None,
                "is_async": False,
            })
        elif isinstance(node, ast.AsyncFunctionDef):
            functions.append({
                "name": node.name,
                "args": [arg.arg for arg in node.args.args],
                "returns": ast.unparse(node.returns) if node.returns else None,
                "is_async": True,
            })
        elif isinstance(node, ast.ClassDef):
            classes.append({
                "name": node.name,
                "methods": [m.name for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))],
            })

    return {"functions": functions, "classes": classes}

# 실행
result = analyze_code("src/beanllm/facade/core/client_facade.py")
print(f"Found {len(result['functions'])} functions, {len(result['classes'])} classes")
```

### 2. 테스트 케이스 생성

```python
# 예: Client.chat() 메서드에 대한 테스트 생성

# Before 분석
class Client:
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o",
        temperature: float = 0.7
    ) -> ChatResponse:
        """LLM과 대화합니다."""
        if not messages:
            raise ValueError("messages는 비어있을 수 없습니다")

        return await self._handler.handle_chat(...)

# After 테스트 생성
import pytest
from unittest.mock import AsyncMock, Mock
from beanllm import Client
from beanllm.dto.response.core.chat_response import ChatResponse

class TestClient:
    """Client 클래스 테스트"""

    @pytest.fixture
    async def client(self):
        """Client 인스턴스 생성"""
        client = Client(model="gpt-4o")
        yield client
        await client.close()

    @pytest.fixture
    def chat_messages(self):
        """테스트용 메시지"""
        return [{"role": "user", "content": "Hello"}]

    # 1. 정상 케이스 테스트
    @pytest.mark.asyncio
    async def test_chat_returns_valid_response_when_given_valid_messages(
        self, client, chat_messages
    ):
        """유효한 메시지가 주어졌을 때 응답을 반환합니다"""
        # Arrange
        # (fixtures로 이미 준비됨)

        # Act
        response = await client.chat(messages=chat_messages)

        # Assert
        assert isinstance(response, ChatResponse)
        assert response.content is not None
        assert len(response.content) > 0
        assert response.model == "gpt-4o"
        assert response.usage is not None

    # 2. 엣지 케이스 테스트
    @pytest.mark.asyncio
    async def test_chat_raises_value_error_when_messages_empty(self, client):
        """빈 메시지 목록이 주어졌을 때 ValueError를 발생시킵니다"""
        with pytest.raises(ValueError, match="messages는 비어있을 수 없습니다"):
            await client.chat(messages=[])

    # 3. 파라미터 변형 테스트
    @pytest.mark.asyncio
    @pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0, 2.0])
    async def test_chat_accepts_various_temperatures(
        self, client, chat_messages, temperature
    ):
        """다양한 temperature 값을 허용합니다"""
        response = await client.chat(
            messages=chat_messages,
            temperature=temperature
        )
        assert isinstance(response, ChatResponse)

    # 4. 모델 변형 테스트
    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", [
        "gpt-4o",
        "claude-sonnet-4-20250514",
        "gemini-2.5-pro",
    ])
    async def test_chat_works_with_different_models(
        self, chat_messages, model
    ):
        """다양한 모델을 지원합니다"""
        client = Client(model=model)
        response = await client.chat(messages=chat_messages)
        assert response.model == model
        await client.close()

    # 5. 에러 처리 테스트
    @pytest.mark.asyncio
    async def test_chat_handles_api_error_gracefully(self, client, chat_messages):
        """API 에러가 발생했을 때 적절히 처리합니다"""
        # Mock handler to raise API error
        client._handler.handle_chat = AsyncMock(
            side_effect=APIError("Rate limit exceeded")
        )

        with pytest.raises(APIError, match="Rate limit exceeded"):
            await client.chat(messages=chat_messages)

    # 6. 재시도 테스트
    @pytest.mark.asyncio
    async def test_chat_retries_on_rate_limit_error(self, client, chat_messages):
        """Rate limit 에러 시 재시도합니다"""
        # First call: rate limit error
        # Second call: success
        client._handler.handle_chat = AsyncMock(
            side_effect=[
                RateLimitError("Rate limit exceeded"),
                ChatResponse(content="Hello!", model="gpt-4o")
            ]
        )

        response = await client.chat(messages=chat_messages)
        assert response.content == "Hello!"
        assert client._handler.handle_chat.call_count == 2
```

### 3. 통합 테스트 생성

```python
# Integration Tests
@pytest.mark.integration
class TestClientIntegration:
    """Client 통합 테스트 (실제 Provider 사용)"""

    @pytest.mark.asyncio
    async def test_chat_with_ollama(self):
        """Ollama Provider와 통합 테스트"""
        # Ollama가 실행 중이어야 함
        client = Client(model="qwen2.5:0.5b")
        response = await client.chat([
            {"role": "user", "content": "Hello"}
        ])

        assert response.content is not None
        assert len(response.content) > 0
        await client.close()

    @pytest.mark.asyncio
    @pytest.mark.requires_api_key
    async def test_chat_with_openai(self):
        """OpenAI Provider와 통합 테스트"""
        client = Client(model="gpt-4o-mini")
        response = await client.chat([
            {"role": "user", "content": "Say 'test' only"}
        ])

        assert "test" in response.content.lower()
        await client.close()
```

### 4. E2E 테스트 생성

```python
# End-to-End Tests
@pytest.mark.e2e
class TestClientE2E:
    """Client E2E 테스트 (전체 플로우)"""

    @pytest.mark.asyncio
    async def test_full_conversation_flow(self):
        """전체 대화 플로우 테스트"""
        client = Client(model="qwen2.5:0.5b")

        # 1. 첫 번째 메시지
        response1 = await client.chat([
            {"role": "user", "content": "My name is Alice"}
        ])
        assert response1.content is not None

        # 2. 대화 이어가기
        response2 = await client.chat([
            {"role": "user", "content": "My name is Alice"},
            {"role": "assistant", "content": response1.content},
            {"role": "user", "content": "What is my name?"}
        ])
        assert "alice" in response2.content.lower()

        await client.close()
```

## Output Format

```
=================================================
🧪 Test Generation Report
=================================================

📋 Summary:
  Target file: src/beanllm/facade/core/client_facade.py
  Classes found: 1 (Client)
  Methods found: 8 (chat, stream_chat, ...)
  Tests generated: 42
  Coverage estimate: 87%

=================================================
📁 Generated Test Files
=================================================

1. tests/facade/core/test_client_facade.py
   - Unit tests: 24
   - Integration tests: 12
   - E2E tests: 6
   - Fixtures: 5

2. tests/conftest.py (updated)
   - Added fixtures: 3

=================================================
✅ Test Cases Generated
=================================================

Unit Tests (24):
  ✅ test_chat_returns_valid_response_when_given_valid_messages
  ✅ test_chat_raises_value_error_when_messages_empty
  ✅ test_chat_accepts_various_temperatures
  ✅ test_chat_works_with_different_models
  ✅ test_chat_handles_api_error_gracefully
  ✅ test_chat_retries_on_rate_limit_error
  ...

Integration Tests (12):
  ✅ test_chat_with_ollama
  ✅ test_chat_with_openai
  ✅ test_chat_with_anthropic
  ...

E2E Tests (6):
  ✅ test_full_conversation_flow
  ✅ test_multimodal_chat_with_images
  ...

=================================================
📊 Coverage Analysis
=================================================

Current coverage: 61%
After adding tests: 87% (estimated)
Goal: 80%

Status: ✅ GOAL ACHIEVED

Uncovered lines:
  - src/beanllm/facade/core/client_facade.py:156-162 (error handling)
  - src/beanllm/facade/core/client_facade.py:203-208 (cleanup)

Recommendation:
  Add tests for error handling and cleanup logic to reach 90%

=================================================
🚀 Next Steps
=================================================

1. Review generated tests: tests/facade/core/test_client_facade.py
2. Run tests: pytest tests/facade/core/test_client_facade.py -v
3. Check coverage: pytest --cov=src/beanllm/facade/core/client_facade.py
4. Add missing tests for uncovered lines
5. Update documentation if needed
```

## Pytest Commands

```bash
# 생성된 테스트 실행
pytest tests/facade/core/test_client_facade.py -v

# 커버리지 확인
pytest tests/facade/core/test_client_facade.py \
  --cov=src/beanllm/facade/core/client_facade.py \
  --cov-report=html

# 특정 마커만 실행
pytest -m unit  # Unit tests only
pytest -m "integration and not requires_api_key"
```

## Related Commands

- `/tdd` - TDD 워크플로우 시작
- `/test-run` - 테스트 실행 및 커버리지 확인

## Related Documents

- `.claude/rules/testing.md` - 테스트 규칙
- `pyproject.toml` - pytest 설정
- `CLAUDE.md` - 프로젝트 컨텍스트
