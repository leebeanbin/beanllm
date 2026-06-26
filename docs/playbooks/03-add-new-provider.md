# Playbook 03 — 새 프로바이더 추가

새 LLM 프로바이더를 beanllm에 통합하는 절차입니다.

---

## 체크리스트

- [ ] Step 1: `BaseLLMProvider` 상속 구현
- [ ] Step 2: `is_available()` 구현 (API 키 확인)
- [ ] Step 3: `chat()`, `stream_chat()`, `embed()` 구현
- [ ] Step 4: `provider_factory.py`에 등록
- [ ] Step 5: `provider_registry.py`에 모델 패턴 등록
- [ ] Step 6: `pyproject.toml` optional extra 추가
- [ ] Step 7: 테스트 작성 (httpx Mock, CircuitBreaker 포함)
- [ ] Step 8: README 프로바이더 표 업데이트

---

## Step 1: 프로바이더 파일 생성

`src/beanllm/providers/{name}_provider.py`를 생성합니다.

기존 구현을 참고하세요:
- 최소 구현: `deepseek_provider.py` (OpenAI 호환 API)
- 스트리밍: `claude_provider.py` (SSE 방식)
- 임베딩 포함: `openai_provider.py`

필수 구현 메서드:

```python
class NewProvider(BaseLLMProvider):
    def is_available(self) -> bool: ...
    async def chat(self, messages, model, system, temperature, max_tokens, **kwargs) -> LLMResponse: ...
    async def stream_chat(self, messages, model, ...) -> AsyncGenerator[str, None]: ...
    async def embed(self, texts, model, **kwargs) -> List[List[float]]: ...  # 임베딩 지원 시
```

자세한 구현 방법은 [wiki/contributing.md](../../wiki/contributing.md)를 참고하세요.

---

## Step 2: `is_available()` 구현

환경변수에 API 키가 있을 때만 `True`를 반환합니다.

```python
def is_available(self) -> bool:
    return bool(
        self.api_key or os.environ.get("NEWPROVIDER_API_KEY")
    )
```

Ollama처럼 API 키가 없는 경우:
```python
def is_available(self) -> bool:
    # 서버에 연결 가능한지 확인
    try:
        import httpx
        httpx.get(self.base_url, timeout=2.0)
        return True
    except Exception:
        return False
```

---

## Step 3: `chat()` 및 `stream_chat()` 구현

`BaseLLMProvider.__init__(config)`를 반드시 호출해야 CircuitBreaker가 초기화됩니다.

오류는 `ProviderError`로 래핑합니다:

```python
from beanllm.utils.exceptions import ProviderError

async def chat(self, messages, model, ...):
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(...)
            resp.raise_for_status()
            ...
    except httpx.HTTPStatusError as e:
        raise ProviderError(f"HTTP {e.response.status_code}: {e.response.text}") from e
    except Exception as e:
        raise ProviderError(f"Provider error: {e}") from e
```

---

## Step 4: `provider_factory.py` 등록

`src/beanllm/providers/provider_factory.py`에서:

1. 상단 import 블록에 추가 (try/except 패턴 유지)
2. `_get_provider_priority()` 리스트에 추가
3. `get_provider()` 내 `provider_map`에 추가

```python
# 1. import
try:
    from .newprovider_provider import NewProvider
except Exception as e:
    logger.warning(f"Failed to import NewProvider: {e}")
    NewProvider = None

# 2. priority list (기존 항목 뒤에 추가)
if NewProvider is not None:
    priority.append(("newprovider", NewProvider, "NEWPROVIDER_API_KEY"))

# 3. provider_map
"newprovider": (NewProvider, "NEWPROVIDER_API_KEY"),
```

---

## Step 5: `provider_registry.py` 등록

`src/beanllm/providers/provider_registry.py`에서 모델 ID 패턴을 추가합니다.

```python
# PROVIDER_MODEL_PATTERNS 딕셔너리에 추가
"newprovider": [
    r"^newprovider-",
    r"^np-[a-z]",
],
```

---

## Step 6: `pyproject.toml` 수정

```toml
[project.optional-dependencies]
# 추가
newprovider = [
    "newprovider-python>=1.0.0,<2.0.0",
]

# all 그룹에도 추가
all = [
    # ... 기존 항목 ...
    "newprovider-python>=1.0.0,<2.0.0",
]
```

---

## Step 7: 테스트 작성

`tests/` 아래에 테스트 파일을 추가합니다.

필수 테스트 케이스:
1. `is_available()` — 키 있을 때 True, 없을 때 False
2. `chat()` 성공 — 정상 응답 파싱
3. `chat()` HTTP 오류 → `ProviderError` 변환
4. `stream_chat()` 청크 스트리밍
5. CircuitBreaker 트리거 (5회 연속 실패)
6. `ProviderFactory.get_provider(provider_name="newprovider")` 통합

```bash
# 테스트 실행
pytest tests/test_newprovider.py -v

# 기존 테스트 회귀 확인
pytest tests/ --no-cov -x -q
```

---

## Step 8: README 업데이트

`README.md`에서:

1. 프로바이더 지원 표에 행 추가
2. 설치 가이드에 extra 추가
3. 예시 코드에 새 프로바이더 언급 (선택)

---

## 검증

```python
# 통합 검증
from beanllm.providers.provider_factory import ProviderFactory

# 환경변수 설정 후
import os
os.environ["NEWPROVIDER_API_KEY"] = "test-key"

available = ProviderFactory.get_available_providers()
assert "newprovider" in available, "Provider not registered"

# 실제 API 호출 (e2e)
from beanllm import Client
client = Client(model="newprovider-turbo", provider="newprovider")
resp = await client.chat([{"role": "user", "content": "hello"}])
assert resp.content
```

---

## 관련 문서

- [wiki/contributing.md](../../wiki/contributing.md) — 개발 환경 설정 및 전체 구현 가이드
- [wiki/architecture.md](../../wiki/architecture.md) — 레이어 의존성 규칙
- [wiki/providers.md](../../wiki/providers.md) — 기존 프로바이더 레퍼런스
