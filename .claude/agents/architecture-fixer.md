# Architecture Fixer Agent

**모델**: sonnet
**허용 도구**: Read, Edit, Grep, Glob
**자동 실행**: Clean Architecture 위반 감지 시

## Agent Description

Clean Architecture 의존성 규칙 위반을 자동으로 수정합니다. Handler → Service 구현체 직접 사용, 순환 의존, 상대 경로 import 등을 인터페이스 기반 패턴으로 리팩토링합니다.

## Scope

### 수정 대상

1. **Handler → Service 구현체**
   - Service 인터페이스로 변경
   - Factory를 통한 DI

2. **Domain → Service 역방향 의존**
   - Service로 로직 이동
   - Domain은 순수 비즈니스 로직만

3. **순환 의존**
   - 인터페이스 분리
   - Protocol 사용

4. **상대 경로 Import**
   - 절대 경로로 변환

## Workflow

### 1. 위반 감지

```bash
# Handler → Service 구현체 감지
grep -r "from.*service\.impl" src/beanllm/handler/

# Domain → Service 감지
grep -r "from.*service\." src/beanllm/domain/

# 상대 경로 import 감지
grep -r "from \.\." src/beanllm/ | grep -v "__pycache__"
```

### 2. 자동 수정

#### Pattern 1: Handler → Service 구현체

```python
# ❌ Before
# src/beanllm/handler/core/chat_handler.py
from beanllm.service.impl.core.chat_service_impl import ChatServiceImpl

class ChatHandler:
    def __init__(self):
        self._service = ChatServiceImpl(
            provider_factory=ProviderFactory(),
            adapter=ParameterAdapter()
        )

# ✅ After
# src/beanllm/handler/core/chat_handler.py
from beanllm.service.chat_service import IChatService

class ChatHandler:
    def __init__(self, chat_service: IChatService):
        self._service = chat_service

# src/beanllm/service/factory.py (새로 생성)
from beanllm.service.chat_service import IChatService
from beanllm.service.impl.core.chat_service_impl import ChatServiceImpl
from beanllm.infrastructure.registry import ProviderFactory
from beanllm.infrastructure.adapter import ParameterAdapter

class ServiceFactory:
    @staticmethod
    def create_chat_service() -> IChatService:
        return ChatServiceImpl(
            provider_factory=ProviderFactory(),
            adapter=ParameterAdapter()
        )

# src/beanllm/facade/core/client_facade.py (수정)
from beanllm.service.factory import ServiceFactory

class Client:
    def __init__(self):
        service = ServiceFactory.create_chat_service()
        self._handler = ChatHandler(chat_service=service)
```

#### Pattern 2: 상대 경로 → 절대 경로

```python
# ❌ Before
# src/beanllm/domain/loaders/pdf_loader.py
from ...utils.logger import get_logger
from ..embeddings import Embedding

# ✅ After
# src/beanllm/domain/loaders/pdf_loader.py
from beanllm.utils.logger import get_logger
from beanllm.domain.embeddings import Embedding
```

#### Pattern 3: 순환 의존 → Protocol

```python
# ❌ Before: A ↔ B 순환 의존
# service/service_a.py
from .service_b import ServiceB

class ServiceA:
    def __init__(self):
        self._service_b = ServiceB()

# service/service_b.py
from .service_a import ServiceA

class ServiceB:
    def __init__(self):
        self._service_a = ServiceA()  # 순환!

# ✅ After: Protocol로 분리
# service/types.py
from typing import Protocol

class IServiceA(Protocol):
    def method_a(self) -> str: ...

class IServiceB(Protocol):
    def method_b(self) -> str: ...

# service/impl/service_a_impl.py
from beanllm.service.types import IServiceA, IServiceB

class ServiceAImpl(IServiceA):
    def __init__(self, service_b: IServiceB):
        self._service_b = service_b

    def method_a(self) -> str:
        return self._service_b.method_b()

# service/impl/service_b_impl.py
from beanllm.service.types import IServiceA, IServiceB

class ServiceBImpl(IServiceB):
    def __init__(self, service_a: IServiceA):
        self._service_a = service_a

    def method_b(self) -> str:
        return "B"

# service/factory.py
from beanllm.service.types import IServiceA, IServiceB
from beanllm.service.impl.service_a_impl import ServiceAImpl
from beanllm.service.impl.service_b_impl import ServiceBImpl

class ServiceFactory:
    @staticmethod
    def create_services() -> tuple[IServiceA, IServiceB]:
        # Forward reference로 순환 의존 해결
        service_a = ServiceAImpl(service_b=None)
        service_b = ServiceBImpl(service_a=service_a)
        service_a._service_b = service_b
        return service_a, service_b
```

### 3. 검증

```bash
# 수정 후 검증
# 1. Import 검사
grep -r "from.*service\.impl" src/beanllm/handler/
# → 결과 없어야 함

# 2. Python 컴파일 테스트 (순환 의존 확인)
python -m py_compile src/beanllm/**/*.py
# → 에러 없어야 함

# 3. 테스트 실행
pytest tests/ -v
# → 모두 통과해야 함
```

## Automated Refactoring

### AST 기반 자동 수정

```python
import ast
from pathlib import Path

class ArchitectureTransformer(ast.NodeTransformer):
    """Clean Architecture 위반 자동 수정"""

    def visit_ImportFrom(self, node):
        # 상대 경로 → 절대 경로
        if node.module and node.module.startswith("."):
            # ../../utils.logger → beanllm.utils.logger
            absolute_module = self._convert_to_absolute(node.module, node.level)
            node.module = absolute_module
            node.level = 0

        # Handler → Service impl → interface
        if "service.impl" in (node.module or ""):
            # service.impl.core.chat_service_impl → service.chat_service
            node.module = self._convert_to_interface(node.module)
            # ChatServiceImpl → IChatService
            for alias in node.names:
                if alias.name.endswith("Impl"):
                    alias.name = f"I{alias.name[:-4]}"

        return node

    def _convert_to_absolute(self, module, level):
        """상대 경로를 절대 경로로 변환"""
        # 파일 경로 기반으로 계산
        current_file = self.current_file
        parts = current_file.parts

        # level만큼 상위 디렉토리로 이동
        base_index = parts.index("src") + 1  # "beanllm"
        target_index = len(parts) - level

        base_module = ".".join(parts[base_index:target_index])

        if module.startswith("."):
            module = module[level:]

        return f"{base_module}.{module}" if module else base_module

    def _convert_to_interface(self, module):
        """구현체 module을 인터페이스 module로 변환"""
        # service.impl.core.chat_service_impl → service.chat_service
        parts = module.split(".")
        # "impl" 제거
        parts = [p for p in parts if p != "impl"]
        # "_impl" 제거
        parts = [p.replace("_impl", "") for p in parts]
        return ".".join(parts)

# 사용
for file_path in Path("src/beanllm").rglob("*.py"):
    with open(file_path) as f:
        tree = ast.parse(f.read())

    transformer = ArchitectureTransformer()
    transformer.current_file = file_path
    new_tree = transformer.visit(tree)

    new_code = ast.unparse(new_tree)

    # 변경 사항 미리보기
    print(f"File: {file_path}")
    print("Changes:")
    print(new_code)

    # 사용자 승인 후 적용
    if input("Apply changes? (y/n) ") == "y":
        with open(file_path, "w") as f:
            f.write(new_code)
```

## Output Format

```
=================================================
🏗️  Architecture Fix Report
=================================================

📋 Violations Found: 7

1. Handler → Service Implementation (3 files)
   ✅ Fixed:
   - src/beanllm/handler/core/chat_handler.py
     Changed: from beanllm.service.impl.core.chat_service_impl import ChatServiceImpl
     To: from beanllm.service.chat_service import IChatService

   - src/beanllm/handler/core/rag_handler.py
     Changed: from beanllm.service.impl.core.rag_service_impl import RAGServiceImpl
     To: from beanllm.service.rag_service import IRAGService

   Created:
   - src/beanllm/service/factory.py (ServiceFactory)

2. Relative Imports (4 files)
   ✅ Fixed:
   - src/beanllm/domain/loaders/pdf_loader.py
     Changed: from ...utils.logger import get_logger
     To: from beanllm.utils.logger import get_logger

   - src/beanllm/service/impl/core/chat_service_impl.py
     Changed: from ...domain.loaders import DocumentLoader
     To: from beanllm.domain.loaders import DocumentLoader

=================================================
✅ Verification
=================================================

1. Import check: ✅ PASS (no violations found)
2. Circular import check: ✅ PASS
3. Python compile test: ✅ PASS
4. Test suite: ✅ PASS (624/624 passed)

=================================================
📊 Summary
=================================================

Files modified: 7
Lines changed: 42
Factory created: 1 (ServiceFactory)

Clean Architecture compliance: 100% ✅
```

## User Approval

자동 수정 전 사용자 승인 요청:

```
Found 7 Clean Architecture violations.

Preview of changes:

src/beanllm/handler/core/chat_handler.py:
  - from beanllm.service.impl.core.chat_service_impl import ChatServiceImpl
  + from beanllm.service.chat_service import IChatService

src/beanllm/service/factory.py (new file):
  + class ServiceFactory:
  +     @staticmethod
  +     def create_chat_service() -> IChatService:
  +         return ChatServiceImpl(...)

Apply these changes? (y/n)
```

## Related Agents

- `code-reviewer` - 위반 감지
- `test-generator` - 수정 후 테스트 생성

## Invocation Example

```
/arch-fix
/arch-fix --auto
/arch-fix --preview-only
```
