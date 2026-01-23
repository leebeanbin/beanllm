# Clean Architecture 의존성 규칙

**우선순위**: CRITICAL
**적용 범위**: 모든 코드 변경

## 핵심 원칙

### 의존성 방향 (반드시 준수)

```
Facade → Handler → Service (인터페이스) → Domain (인터페이스) ← Infrastructure
```

## 레이어별 허용/금지 사항

### Facade Layer (`facade/`)

✅ **허용**:
- Handler 호출
- DTO 사용
- Utils 사용
- Domain 직접 사용 (기존 API 유지)
- Infrastructure 직접 사용 (기존 API 유지)

❌ **금지**:
- Service 직접 호출 (Handler를 통해서만)
- Service 구현체 직접 생성

### Handler Layer (`handler/`)

✅ **허용**:
- Service **인터페이스**만 의존
- DTO 사용
- Utils 사용
- Decorators 사용

❌ **금지**:
- Service **구현체** 직접 사용 → 인터페이스로 대체
- Domain 직접 사용 → Service를 통해 접근
- Infrastructure 직접 사용 → Service를 통해 접근

### Service Layer (`service/`)

#### Service 인터페이스 (`service/*.py`)

✅ **허용**:
- DTO만 의존
- Domain 인터페이스/프로토콜만 의존

❌ **금지**:
- Infrastructure 구현체 의존
- Handler, Facade 의존

#### Service 구현체 (`service/impl/`)

✅ **허용**:
- Service 인터페이스 구현
- Domain 인터페이스 의존
- Infrastructure 인터페이스 의존
- Domain 직접 사용 (비즈니스 로직)
- DTO 사용

❌ **금지**:
- Handler, Facade 의존
- 다른 Service 구현체 직접 의존

### Domain Layer (`domain/`)

✅ **허용**:
- Domain 내부 모듈만
- 표준 라이브러리만

❌ **금지**:
- Service, Handler, Facade 의존
- Infrastructure 구현체 의존
- 외부 라이브러리 의존 (최소화)

### Infrastructure Layer (`infrastructure/`, `providers/`)

✅ **허용**:
- Domain 인터페이스 구현
- Utils 사용

❌ **금지**:
- Service, Handler, Facade 의존

## 위반 시 조치

### 1. Handler가 Service 구현체를 사용하는 경우

```python
# ❌ Before
class ChatHandler:
    def __init__(self):
        self._service = ChatServiceImpl(...)  # 구현체 직접 사용

# ✅ After
class ChatHandler:
    def __init__(self, chat_service: IChatService):  # 인터페이스
        self._service = chat_service
```

### 2. Domain이 Service를 사용하는 경우

```python
# ❌ Before - Domain에서 Service 사용
class Document:
    def process(self):
        service = ChatServiceImpl(...)  # 역방향 의존

# ✅ After - Service에서 Domain 사용
class ChatServiceImpl:
    def process_document(self, doc: Document):
        # Domain 객체를 Service에서 처리
        pass
```

### 3. 순환 의존이 발생하는 경우

```python
# ❌ Before - A ↔ B 순환 의존

# ✅ After - 인터페이스로 분리
# interface.py
class IService(Protocol):
    def method(self): pass

# service_a.py
class ServiceA:
    def __init__(self, service_b: IService):
        self._service_b = service_b

# service_b.py
class ServiceB:
    def __init__(self, service_a: IService):
        self._service_a = service_a
```

## 검증 방법

### Import 검사

```bash
# Handler가 Service 구현체를 import하는지 확인
grep -r "from.*service\.impl" src/beanllm/handler/

# Domain이 Service를 import하는지 확인
grep -r "from.*service\." src/beanllm/domain/

# Infrastructure가 Handler를 import하는지 확인
grep -r "from.*handler\." src/beanllm/infrastructure/
```

### Python 컴파일 테스트

```bash
# 순환 import 확인
python -m py_compile src/beanllm/**/*.py
```

## 예외 사항

### Factory 패턴

Factory는 모든 레이어를 알 수 있음:

```python
class ServiceFactory:
    def create_chat_service(self) -> IChatService:
        from beanllm.service.impl.core.chat_service_impl import ChatServiceImpl
        return ChatServiceImpl(...)
```

### DI Container

DI Container는 의존성 주입을 위해 모든 레이어 접근 가능:

```python
class DIContainer:
    def __init__(self):
        from beanllm.facade.core.client_facade import Client
        from beanllm.handler.core.chat_handler import ChatHandler
        # ...
```

### `__init__.py` (Public API)

Public API export는 모든 레이어 접근 가능:

```python
from beanllm.facade.core.client_facade import Client
from beanllm.service.chat_service import IChatService
```

## 참고 문서

- `DEPENDENCY_RULES.md` - 전체 의존성 규칙
- `CLAUDE.md` - 프로젝트 컨텍스트
- `ARCHITECTURE.md` - 아키텍처 상세 설명
