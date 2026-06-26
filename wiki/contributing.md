# Contributing

beanllm에 기여하는 방법을 설명합니다. 새 LLM 프로바이더를 추가하는 것이 가장 일반적인 기여 형태입니다.

---

## 개발 환경 설정

```bash
git clone https://github.com/leebeanbin/beanllm.git
cd beanllm

# Poetry로 의존성 설치
poetry install --with dev

# 또는 pip
pip install -e ".[dev]"

# pre-commit 훅 설치
pre-commit install
```

---

## 새 프로바이더 추가

### 1. `BaseLLMProvider` 상속 구현

`src/beanllm/providers/{name}_provider.py` 파일을 생성합니다.

```python
# src/beanllm/providers/newprovider_provider.py
from typing import Any, AsyncGenerator, Dict, List, Optional
import httpx

from beanllm.providers.base_provider import BaseLLMProvider, LLMResponse
from beanllm.utils.constants import DEFAULT_TEMPERATURE
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class NewProvider(BaseLLMProvider):
    """NewProvider LLM integration."""

    BASE_URL = "https://api.newprovider.com/v1"

    def __init__(self, config: Dict):
        super().__init__(config)  # CircuitBreaker + RateLimit 자동 초기화
        self.api_key = config.get("api_key") or os.environ.get("NEWPROVIDER_API_KEY", "")

    def is_available(self) -> bool:
        """API 키가 있으면 사용 가능."""
        return bool(self.api_key)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        system: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """비스트리밍 채팅."""
        payload = self._build_payload(messages, model, system, temperature, max_tokens)

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()

        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data.get("model", model),
            usage=data.get("usage"),
        )

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        system: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """스트리밍 채팅."""
        payload = {**self._build_payload(messages, model, system, temperature, max_tokens), "stream": True}

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=60.0,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        import json
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0].get("delta", {})
                        if content := delta.get("content"):
                            yield content

    async def embed(self, texts: List[str], model: str = "newprovider-embed", **kwargs) -> List[List[float]]:
        """임베딩 생성."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.BASE_URL}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"input": texts, "model": model},
            )
            resp.raise_for_status()
            data = resp.json()
        return [item["embedding"] for item in data["data"]]

    def _build_payload(self, messages, model, system, temperature, max_tokens):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)
        payload = {"model": model, "messages": msgs, "temperature": temperature}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        return payload
```

### 2. `provider_factory.py`에 등록

`src/beanllm/providers/provider_factory.py`를 수정합니다.

```python
# 상단 import 블록에 추가
try:
    from .newprovider_provider import NewProvider
except Exception as e:
    logger.warning(f"Failed to import NewProvider: {e}")
    NewProvider = None

# _get_provider_priority() 메서드에 추가
if NewProvider is not None:
    priority.append(("newprovider", NewProvider, "NEWPROVIDER_API_KEY"))

# provider_map 딕셔너리에 추가 (get_provider 메서드 내부)
"newprovider": (NewProvider, "NEWPROVIDER_API_KEY"),
```

### 3. `provider_registry.py`에 모델 패턴 등록

`src/beanllm/providers/provider_registry.py`에 모델 감지 패턴을 추가합니다.

```python
# PROVIDER_MODEL_PATTERNS 딕셔너리에 추가
"newprovider": [
    r"^newprovider-",
    r"^np-",
],

# PROVIDER_FACTORY_NAME_MAP에 추가 (필요한 경우)
"newprovider": "newprovider",
```

### 4. `pyproject.toml`에 optional extra 추가

```toml
[project.optional-dependencies]
# ... 기존 항목들 ...

# New Provider
newprovider = [
    "newprovider-sdk>=1.0.0,<2.0.0",
]

# all에도 추가
all = [
    # ... 기존 ...
    "newprovider-sdk>=1.0.0,<2.0.0",
]
```

### 5. 테스트 작성

`tests/test_infrastructure_legacy.py` 또는 별도 파일에 추가합니다.

```python
# tests/test_newprovider.py
import pytest
from unittest.mock import patch, AsyncMock
import httpx


@pytest.mark.asyncio
async def test_newprovider_chat_success():
    mock_response = {
        "choices": [{"message": {"content": "Hello from NewProvider"}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 15},
        "model": "newprovider-turbo",
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = httpx.Response(200, json=mock_response)

        from beanllm.providers.newprovider_provider import NewProvider
        provider = NewProvider({"api_key": "test-key"})
        result = await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="newprovider-turbo",
        )

    assert result.content == "Hello from NewProvider"
    assert result.model == "newprovider-turbo"


@pytest.mark.asyncio
async def test_newprovider_is_available():
    from beanllm.providers.newprovider_provider import NewProvider
    provider = NewProvider({"api_key": "test-key"})
    assert provider.is_available() is True

    provider_no_key = NewProvider({})
    assert provider_no_key.is_available() is False


@pytest.mark.asyncio
async def test_newprovider_circuit_breaker():
    """Circuit breaker activates after 5 consecutive failures."""
    from beanllm.providers.newprovider_provider import NewProvider
    from beanllm.utils.exceptions import CircuitBreakerError

    provider = NewProvider({"api_key": "test-key"})

    with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("Connection failed")):
        for _ in range(5):
            with pytest.raises(Exception):
                await provider.chat([{"role": "user", "content": "Hi"}], model="np-test")

    # 6번째 호출은 CircuitBreakerError
    with pytest.raises(CircuitBreakerError):
        await provider.chat([{"role": "user", "content": "Hi"}], model="np-test")
```

### 6. README 업데이트

`README.md`의 프로바이더 표와 설치 섹션에 새 프로바이더를 추가합니다.

---

## 체크리스트

새 프로바이더 추가 시 확인할 사항:

- [ ] `BaseLLMProvider` 상속, `chat()`, `stream_chat()`, `is_available()` 구현
- [ ] `provider_factory.py`에 import 및 우선순위 등록
- [ ] `provider_registry.py`에 모델 패턴 등록
- [ ] `pyproject.toml`에 optional extra 추가 (`newprovider`, `all` 그룹)
- [ ] 단위 테스트 작성 (httpx mock 사용, circuit breaker 포함)
- [ ] `README.md` 프로바이더 표 업데이트
- [ ] `.env.example`에 새 환경변수 추가

---

## 코드 스타일

```bash
# 포맷팅
black src/ tests/

# 린트
ruff check src/ tests/

# 타입 체크
mypy src/beanllm/

# 전체 검사
make check
```

커밋 전 pre-commit이 자동으로 black, ruff, mypy를 실행합니다.

---

## 커밋 규칙

- 논리 단위로 커밋을 분리합니다.
- `Co-Authored-By` 줄을 커밋에 포함하지 않습니다.
- 예시 형식:

```
feat(providers): add NewProvider integration
test(providers): add NewProvider unit tests with httpx mock
docs(providers): update provider table for NewProvider
```

---

## 관련 문서

- [Architecture](architecture.md) — 레이어 규칙 (의존성 방향)
- [Testing](testing.md) — 레이어별 테스트 전략
- [Providers](providers.md) — 기존 프로바이더 구현 참조
- [Playbook: Add New Provider](../docs/playbooks/03-add-new-provider.md) — 운영 체크리스트
