# Testing

beanllm은 6,340개 테스트와 80% 커버리지를 목표로 합니다.
Clean Architecture 덕분에 HTTP 모킹 없이 대부분의 비즈니스 로직을 단위 테스트할 수 있습니다.

---

## 레이어별 테스트 전략

### Domain Layer — 순수 단위 테스트

Domain 레이어는 외부 의존성이 없으므로 가장 단순하고 빠릅니다.

```python
# tests/test_domain.py
from beanllm.providers.base_provider import LLMResponse

def test_llm_response():
    resp = LLMResponse(content="Hello", model="gpt-4o", usage={"total_tokens": 10})
    assert resp.content == "Hello"
    assert resp.usage["total_tokens"] == 10

def test_agent_step_immutable():
    from beanllm.facade.core.agent_facade import AgentStep
    step = AgentStep(step_number=1, thought="Thinking...", is_final=False)
    assert step.step_number == 1
    # frozen=True이므로 수정 불가
```

**파일:** `tests/test_domain.py`

### Service Layer — MockProvider 주입

Service는 Provider 인터페이스에 의존합니다. MockProvider를 주입해 HTTP 없이 테스트합니다.

```python
# tests/conftest.py
import pytest
from unittest.mock import AsyncMock
from beanllm.providers.base_provider import LLMResponse

@pytest.fixture
def mock_provider():
    provider = AsyncMock()
    provider.chat.return_value = LLMResponse(
        content="Mocked response",
        model="mock-model",
        usage={"total_tokens": 5},
    )
    return provider

# tests/test_integration.py
@pytest.mark.asyncio
async def test_chat_service_with_mock(mock_provider):
    from beanllm.service.chat_service import ChatService
    service = ChatService(provider=mock_provider)
    result = await service.chat(
        messages=[{"role": "user", "content": "Hi"}],
        model="mock-model",
    )
    assert "Mocked response" in result.content
    mock_provider.chat.assert_called_once()
```

**파일:** `tests/test_integration.py`, `tests/conftest.py`

### Provider Layer — httpx mock

실제 HTTP 레이어를 테스트할 때는 `httpx`의 mock transport를 사용합니다.

```python
import httpx
import pytest
from unittest.mock import patch

@pytest.mark.asyncio
async def test_openai_provider_chat():
    mock_response = {
        "choices": [{"message": {"content": "test response"}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 10},
        "model": "gpt-4o-mini",
    }

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = httpx.Response(200, json=mock_response)

        from beanllm.providers.openai_provider import OpenAIProvider
        provider = OpenAIProvider({"api_key": "test-key"})
        result = await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4o-mini",
        )
        assert result.content == "test response"
```

**파일:** `tests/test_infrastructure_legacy.py`

### Facade Layer — 통합 테스트

Facade는 전체 스택을 통합 테스트합니다. Mock 또는 실제 환경변수를 사용합니다.

```python
# tests/test_facade.py
@pytest.mark.asyncio
async def test_client_facade_chat(mock_provider, monkeypatch):
    from beanllm import Client
    # 환경 격리
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    client = Client(model="gpt-4o-mini")
    # 내부 provider를 mock으로 교체
    client._chat_handler._service._provider = mock_provider

    response = await client.chat([{"role": "user", "content": "Hello"}])
    assert response.content == "Mocked response"
```

**파일:** `tests/test_facade.py`

### E2E 테스트

실제 API 키가 있을 때만 실행하는 e2e 테스트:

```python
# tests/test_e2e.py
import pytest
import os

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No API key")
@pytest.mark.asyncio
async def test_e2e_openai_chat():
    from beanllm import Client
    client = Client(model="gpt-4o-mini")
    response = await client.chat([{"role": "user", "content": "Say 'ok'"}])
    assert len(response.content) > 0
```

**파일:** `tests/test_e2e.py`

---

## 테스트 파일 구조

```
tests/
  conftest.py                  공통 fixture (mock_provider, mock_messages, ...)
  test_domain.py               Domain 엔티티 순수 단위 테스트
  test_decorators_validation.py 데코레이터·검증 테스트
  test_registry.py             ModelRegistry 테스트
  test_facade.py               Facade 통합 테스트
  test_integration.py          Service·Handler 레이어 통합
  test_infrastructure_legacy.py Provider HTTP 레이어 테스트
  test_utils_and_nodes.py      유틸리티·StateGraph 노드 테스트
  test_text_splitters.py       청킹 전략 테스트
  test_config.py               EnvConfig·설정 테스트
  test_cli.py                  CLI 명령어 테스트
  test_e2e.py                  실제 API 통합 (skip if no key)
  test_import.py               import 가능성 확인
  test_utils_legacy.py         레거시 유틸 테스트
```

---

## pytest 실행

```bash
# 전체 테스트 (커버리지 포함)
pytest

# 특정 파일
pytest tests/test_domain.py

# 특정 테스트 함수
pytest tests/test_facade.py::test_client_facade_chat

# e2e 포함 (API 키 필요)
OPENAI_API_KEY=sk-... pytest tests/test_e2e.py

# 빠른 실행 (커버리지 제외)
pytest tests/ --no-cov -x

# 병렬 실행 (pytest-xdist 설치 필요)
pytest -n auto
```

---

## 커버리지 목표

| 레이어 | 목표 커버리지 | 비고 |
|--------|-------------|------|
| Domain | 95%+ | 외부 의존성 없음 |
| Service | 85%+ | MockProvider 주입 |
| Handler | 80%+ | 통합 테스트 |
| Provider | 70%+ | HTTP mock 필요 |
| Facade | 75%+ | 통합 테스트 |
| 전체 | 80% | 현재 달성 |

커버리지 리포트는 `htmlcov/index.html`에서 확인할 수 있습니다.

```bash
# 커버리지 리포트 열기
open htmlcov/index.html
```

---

## CI 설정

`Makefile`에 주요 테스트 명령이 정의되어 있습니다:

```bash
make test        # pytest 전체 실행
make lint        # ruff + mypy
make format      # black
make check       # lint + test (CI용)
```

---

## 관련 문서

- [Contributing](contributing.md) — 테스트 작성 가이드 포함
- [Architecture](architecture.md) — 레이어 격리가 테스트 전략에 미치는 영향
