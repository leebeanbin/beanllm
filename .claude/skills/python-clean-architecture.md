# Python Clean Architecture Patterns

**자동 활성화**: Python 프로젝트에서 레이어 구조 작업 시
**모델**: sonnet

## Skill Description

Clean Architecture 패턴을 적용하여 Python 코드를 작성하고 리팩토링합니다. 의존성 방향을 검증하고, 레이어 간 경계를 명확히 합니다.

## When to Use

이 스킬은 다음 키워드 감지 시 자동 활성화됩니다:
- "facade", "handler", "service", "domain", "infrastructure"
- "레이어", "layer", "의존성", "dependency"
- "리팩토링", "refactor", "아키텍처", "architecture"

## Pattern Recognition

### 레이어 식별

```python
# Facade Layer: facade/
- 사용자 친화적 API
- Handler 호출만

# Handler Layer: handler/
- 입력 검증
- Service 인터페이스 호출만

# Service Layer: service/
- 인터페이스: service/*.py
- 구현체: service/impl/

# Domain Layer: domain/
- 순수 비즈니스 로직
- 외부 의존성 없음

# Infrastructure Layer: infrastructure/, providers/
- Domain 인터페이스 구현
- 외부 시스템 연동
```

### 의존성 방향 검증

```python
# ✅ Good: 올바른 의존성 방향
# Facade → Handler
class Client:  # facade/
    def __init__(self):
        self._handler = ChatHandler(chat_service)  # ✅

# Handler → Service (인터페이스)
class ChatHandler:  # handler/
    def __init__(self, chat_service: IChatService):  # ✅ 인터페이스
        self._service = chat_service

# Service → Domain
class ChatServiceImpl:  # service/impl/
    def __init__(self):
        self._loader = DocumentLoader()  # ✅ Domain 사용 OK

# ❌ Bad: 역방향 의존
# Handler → Service (구현체) - 금지!
class ChatHandler:
    def __init__(self):
        self._service = ChatServiceImpl(...)  # ❌ 구현체 직접 사용

# Domain → Service - 금지!
class Document:  # domain/
    def process(self):
        service = ChatServiceImpl(...)  # ❌ 역방향 의존
```

## Refactoring Patterns

### 1. Handler가 Service 구현체를 사용하는 경우

```python
# ❌ Before
class ChatHandler:
    def __init__(self):
        self._service = ChatServiceImpl(
            provider_factory=ProviderFactory(),
            adapter=ParameterAdapter()
        )

# ✅ After: 인터페이스 + DI
# handler/core/chat_handler.py
class ChatHandler:
    def __init__(self, chat_service: IChatService):
        self._service = chat_service

# service/factory.py
class ServiceFactory:
    @staticmethod
    def create_chat_service() -> IChatService:
        return ChatServiceImpl(
            provider_factory=ProviderFactory(),
            adapter=ParameterAdapter()
        )

# facade/core/client_facade.py
class Client:
    def __init__(self):
        service = ServiceFactory.create_chat_service()
        self._handler = ChatHandler(chat_service=service)
```

### 2. Domain이 Service를 사용하는 경우

```python
# ❌ Before: Domain에서 Service 호출
class Document:  # domain/
    def __init__(self, content: str):
        self.content = content

    def process(self):
        # ❌ Domain이 Service 의존
        service = ChatServiceImpl()
        return service.process(self.content)

# ✅ After: Service에서 Domain 호출
# domain/loaders.py
class Document:
    def __init__(self, content: str):
        self.content = content

    def get_chunks(self, chunk_size: int) -> List[str]:
        # 순수 비즈니스 로직만
        return [self.content[i:i+chunk_size]
                for i in range(0, len(self.content), chunk_size)]

# service/impl/rag_service_impl.py
class RAGServiceImpl:
    def process_document(self, document: Document):
        # Service가 Domain 사용
        chunks = document.get_chunks(chunk_size=1000)
        # ...
```

### 3. 순환 의존 해결

```python
# ❌ Before: A ↔ B 순환 의존
# service_a.py
from .service_b import ServiceB

class ServiceA:
    def __init__(self):
        self._service_b = ServiceB()  # ❌

# service_b.py
from .service_a import ServiceA

class ServiceB:
    def __init__(self):
        self._service_a = ServiceA()  # ❌ 순환 의존!

# ✅ After: 인터페이스로 분리
# service/types.py
from typing import Protocol

class IServiceA(Protocol):
    def method_a(self): ...

class IServiceB(Protocol):
    def method_b(self): ...

# service/impl/service_a_impl.py
class ServiceAImpl:
    def __init__(self, service_b: IServiceB):
        self._service_b = service_b

# service/impl/service_b_impl.py
class ServiceBImpl:
    def __init__(self, service_a: IServiceA):
        self._service_a = service_a

# service/factory.py (순환 의존 해결)
class ServiceFactory:
    @staticmethod
    def create_services() -> Tuple[IServiceA, IServiceB]:
        service_a = ServiceAImpl(service_b=None)
        service_b = ServiceBImpl(service_a=service_a)
        service_a._service_b = service_b
        return service_a, service_b
```

## Checklist

코드 작성/리팩토링 시 다음 체크리스트를 확인합니다:

- [ ] **Facade**: Handler만 호출, Service 직접 사용 금지
- [ ] **Handler**: Service 인터페이스만 의존, 구현체 직접 사용 금지
- [ ] **Service**: Domain/Infrastructure 인터페이스만 의존
- [ ] **Domain**: 외부 의존성 없음 (순수)
- [ ] **Infrastructure**: Domain 인터페이스 구현
- [ ] **순환 의존 없음**: A → B, B → A 방지
- [ ] **역방향 의존 없음**: Domain → Service 방지
- [ ] **Import 절대 경로**: `from beanllm.domain...` 사용

## Verification Commands

```bash
# Handler가 Service 구현체를 import하는지 확인
grep -r "from.*service\.impl" src/beanllm/handler/

# Domain이 Service를 import하는지 확인
grep -r "from.*service\." src/beanllm/domain/

# Infrastructure가 Handler를 import하는지 확인
grep -r "from.*handler\." src/beanllm/infrastructure/

# 순환 import 확인
python -m py_compile src/beanllm/**/*.py
```

## Output

리팩토링 완료 후 다음 정보를 제공합니다:

1. **변경 사항 요약**
   - 수정된 파일 목록
   - 의존성 방향 변경 내역

2. **의존성 다이어그램**
   ```
   Client (Facade)
     ↓
   ChatHandler (Handler)
     ↓
   IChatService (Service Interface)
     ↑
   ChatServiceImpl (Service Impl)
     ↓
   DocumentLoader (Domain)
     ↑
   OpenAIProvider (Infrastructure)
   ```

3. **검증 결과**
   - [ ] 의존성 방향 올바름
   - [ ] 순환 의존 없음
   - [ ] 테스트 통과

## Related Documents

- `DEPENDENCY_RULES.md` - 전체 의존성 규칙
- `ARCHITECTURE.md` - 아키텍처 상세 설명
- `.claude/rules/clean-architecture.md` - Clean Architecture 규칙
