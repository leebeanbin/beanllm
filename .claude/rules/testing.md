# 테스트 규칙

**우선순위**: HIGH
**적용 범위**: 모든 코드 변경

## 테스트 커버리지

### 목표

- **전체 커버리지**: 80% 이상 (현재 61%)
- **Public API**: 100% 커버리지 필수
- **Critical Path**: 100% 커버리지 필수

### 측정

```bash
# 전체 커버리지 확인
pytest --cov=src/beanllm --cov-report=html --cov-report=term

# HTML 리포트 확인
open htmlcov/index.html
```

## 테스트 우선 작성 (TDD)

### 워크플로우

1. **테스트 작성** (Red)
2. **최소 구현** (Green)
3. **리팩토링** (Refactor)

```python
# 1. Red: 테스트 작성
def test_chat_returns_response():
    client = Client(model="gpt-4o")
    response = await client.chat([{"role": "user", "content": "Hello"}])

    assert isinstance(response, ChatResponse)
    assert response.content is not None
    assert response.model == "gpt-4o"

# 2. Green: 최소 구현
class Client:
    async def chat(self, messages):
        return ChatResponse(
            content="Hello!",
            model=self.model,
            usage={"total_tokens": 10}
        )

# 3. Refactor: 실제 구현으로 리팩토링
class Client:
    async def chat(self, messages):
        provider = self._get_provider(self.model)
        response = await provider.chat(messages)
        return ChatResponse(
            content=response.content,
            model=self.model,
            usage=response.usage
        )
```

## 테스트 구조

### 파일 구조

```
tests/
├── facade/
│   └── test_client_facade.py
├── handler/
│   └── test_chat_handler.py
├── service/
│   └── test_chat_service.py
├── domain/
│   └── test_loaders.py
├── infrastructure/
│   └── test_providers.py
└── integration/
    └── test_end_to_end.py
```

### 명명 규칙

```python
# ✅ Good: 명확한 테스트 이름
def test_chat_returns_valid_response_when_given_valid_messages():
    pass

def test_chat_raises_value_error_when_messages_empty():
    pass

def test_chat_retries_on_rate_limit_error():
    pass

# ❌ Bad: 모호한 테스트 이름
def test_chat():
    pass

def test_error():
    pass
```

## 테스트 종류

### 1. Unit Tests (단위 테스트)

**목적**: 개별 함수/메서드 테스트

```python
# ✅ Good: Mock을 사용한 단위 테스트
from unittest.mock import Mock, AsyncMock

async def test_chat_handler_calls_service_with_correct_params():
    # Arrange
    mock_service = AsyncMock()
    mock_service.chat.return_value = ChatResponse(content="Hello!")
    handler = ChatHandler(chat_service=mock_service)
    request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])

    # Act
    await handler.handle_chat(request)

    # Assert
    mock_service.chat.assert_called_once_with(request)
```

### 2. Integration Tests (통합 테스트)

**목적**: 여러 컴포넌트 간 상호작용 테스트

```python
# ✅ Good: 실제 Provider를 사용한 통합 테스트
@pytest.mark.integration
async def test_client_chat_with_real_provider():
    # Ollama가 실행 중이어야 함
    client = Client(model="qwen2.5:0.5b")
    response = await client.chat([{"role": "user", "content": "Hello"}])

    assert response.content is not None
    assert len(response.content) > 0
```

### 3. End-to-End Tests (E2E 테스트)

**목적**: 전체 플로우 테스트

```python
# ✅ Good: RAG 전체 플로우 테스트
@pytest.mark.e2e
async def test_rag_full_workflow():
    # 1. 문서 로드
    loader = DirectoryLoader("docs/")
    documents = await loader.load()

    # 2. RAG 구축
    rag = RAGChain.from_documents(documents)

    # 3. 쿼리
    result = await rag.query("What is beanllm?")

    assert result.answer is not None
    assert len(result.sources) > 0
```

## 엣지 케이스 테스트

```python
# ✅ Good: 엣지 케이스 테스트
def test_chat_with_empty_messages():
    with pytest.raises(ValueError, match="messages는 비어있을 수 없습니다"):
        await client.chat(messages=[])

def test_chat_with_invalid_role():
    with pytest.raises(ValueError, match="유효하지 않은 role"):
        await client.chat(messages=[{"role": "invalid", "content": "Hi"}])

def test_chat_with_very_long_message():
    long_message = "A" * 100_000
    with pytest.raises(ValueError, match="메시지 길이가 최대"):
        await client.chat(messages=[{"role": "user", "content": long_message}])
```

## 에러 처리 테스트

```python
# ✅ Good: 에러 처리 테스트
async def test_chat_retries_on_rate_limit_error():
    # Rate limit 에러 발생 시 재시도하는지 테스트
    mock_provider = AsyncMock()
    mock_provider.chat.side_effect = [
        RateLimitError("Rate limit exceeded"),
        ChatResponse(content="Hello!")
    ]

    client = Client(model="gpt-4o", provider=mock_provider)
    response = await client.chat([{"role": "user", "content": "Hi"}])

    assert mock_provider.chat.call_count == 2
    assert response.content == "Hello!"

async def test_chat_raises_api_error_after_max_retries():
    mock_provider = AsyncMock()
    mock_provider.chat.side_effect = RateLimitError("Rate limit exceeded")

    client = Client(model="gpt-4o", provider=mock_provider, max_retries=3)

    with pytest.raises(APIError):
        await client.chat([{"role": "user", "content": "Hi"}])

    assert mock_provider.chat.call_count == 3
```

## Fixtures

```python
# ✅ Good: 재사용 가능한 fixtures
@pytest.fixture
def chat_request():
    return ChatRequest(
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-4o",
        temperature=0.7
    )

@pytest.fixture
async def client():
    client = Client(model="gpt-4o")
    yield client
    # Cleanup
    await client.close()

@pytest.fixture
async def rag_chain(tmp_path):
    # 임시 문서 생성
    doc_path = tmp_path / "test.txt"
    doc_path.write_text("This is a test document.")

    # RAG 구축
    rag = RAGChain.from_documents(str(tmp_path))
    yield rag

    # Cleanup
    await rag.cleanup()
```

## Parametrize

```python
# ✅ Good: 여러 케이스 테스트
@pytest.mark.parametrize("model,expected_provider", [
    ("gpt-4o", "openai"),
    ("claude-sonnet-4-20250514", "anthropic"),
    ("gemini-2.5-pro", "google"),
    ("qwen2.5:0.5b", "ollama"),
])
async def test_client_detects_correct_provider(model, expected_provider):
    client = Client(model=model)
    assert client.provider_name == expected_provider
```

## 테스트 마커

```python
# pytest.ini or pyproject.toml
[tool.pytest.ini_options]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "e2e: End-to-end tests",
    "slow: Slow tests",
    "requires_api_key: Tests requiring API keys",
]

# 테스트에 마커 적용
@pytest.mark.unit
def test_something():
    pass

@pytest.mark.integration
@pytest.mark.requires_api_key
async def test_real_api():
    pass

# 특정 마커만 실행
# pytest -m unit
# pytest -m "integration and not requires_api_key"
```

## 테스트 실행

```bash
# 전체 테스트
pytest

# 특정 디렉토리
pytest tests/facade/

# 특정 파일
pytest tests/facade/test_client_facade.py

# 특정 테스트
pytest tests/facade/test_client_facade.py::test_chat

# 커버리지와 함께
pytest --cov=src/beanllm --cov-report=html

# 병렬 실행 (빠름)
pytest -n auto

# 실패한 테스트만 재실행
pytest --lf

# 느린 테스트 제외
pytest -m "not slow"
```

## CI/CD 통합

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install -e ".[dev,all]"

      - name: Run tests
        run: |
          pytest --cov=src/beanllm --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

## 참고 문서

- pytest 공식 문서: https://docs.pytest.org/
- `pyproject.toml` - pytest 설정
- `CLAUDE.md` - 프로젝트 컨텍스트
