# /arch-check - Architecture Verification

**트리거**: `/arch-check`
**모델**: sonnet
**설명**: Clean Architecture 의존성 규칙 검증

## Command Description

프로젝트의 Clean Architecture 규칙 준수 여부를 검증합니다. 의존성 방향, 순환 의존, 역방향 의존을 자동으로 검사합니다.

## Usage

```
/arch-check
/arch-check --layer handler
/arch-check --verbose
```

## Execution Steps

### 1. Import 검사

```bash
# Handler가 Service 구현체를 import하는지 확인
echo "🔍 Checking Handler → Service implementation..."
grep -r "from.*service\.impl" src/beanllm/handler/
if [ $? -eq 0 ]; then
    echo "❌ VIOLATION: Handler는 Service 인터페이스만 의존해야 합니다"
else
    echo "✅ OK: Handler가 Service 인터페이스만 의존합니다"
fi

# Domain이 Service를 import하는지 확인
echo "\n🔍 Checking Domain → Service..."
grep -r "from.*service\." src/beanllm/domain/
if [ $? -eq 0 ]; then
    echo "❌ VIOLATION: Domain은 Service에 의존할 수 없습니다"
else
    echo "✅ OK: Domain이 외부 의존성이 없습니다"
fi

# Infrastructure가 Handler를 import하는지 확인
echo "\n🔍 Checking Infrastructure → Handler..."
grep -r "from.*handler\." src/beanllm/infrastructure/
if [ $? -eq 0 ]; then
    echo "❌ VIOLATION: Infrastructure는 Handler에 의존할 수 없습니다"
else
    echo "✅ OK: Infrastructure가 올바른 의존성 방향을 따릅니다"
fi

# Domain이 Infrastructure를 import하는지 확인
echo "\n🔍 Checking Domain → Infrastructure..."
grep -r "from.*infrastructure\." src/beanllm/domain/
if [ $? -eq 0 ]; then
    echo "❌ VIOLATION: Domain은 Infrastructure에 의존할 수 없습니다"
else
    echo "✅ OK: Domain이 외부 의존성이 없습니다"
fi
```

### 2. 순환 Import 검사

```bash
echo "\n🔍 Checking circular imports..."
python -m py_compile src/beanllm/**/*.py 2>&1 | grep -i "circular"
if [ $? -eq 0 ]; then
    echo "❌ VIOLATION: 순환 import가 감지되었습니다"
else
    echo "✅ OK: 순환 import가 없습니다"
fi
```

### 3. 상대 경로 Import 검사

```bash
echo "\n🔍 Checking relative imports..."
grep -r "from \.\." src/beanllm/ | grep -v "__pycache__" | grep -v ".pyc"
if [ $? -eq 0 ]; then
    echo "❌ VIOLATION: 상대 경로 import가 발견되었습니다 (절대 경로 사용 필수)"
else
    echo "✅ OK: 모든 import가 절대 경로를 사용합니다"
fi
```

### 4. 레이어별 검증

```python
# Python 스크립트로 상세 검증
import ast
import os
from pathlib import Path

def check_layer_dependencies(layer: str):
    """레이어별 의존성 검증"""
    violations = []
    layer_path = Path(f"src/beanllm/{layer}")

    for py_file in layer_path.rglob("*.py"):
        with open(py_file) as f:
            try:
                tree = ast.parse(f.read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        # 레이어별 규칙 검증
                        if layer == "handler" and "service.impl" in module:
                            violations.append(f"{py_file}: {module}")
                        elif layer == "domain" and "service" in module:
                            violations.append(f"{py_file}: {module}")
                        # ...
            except SyntaxError:
                pass

    return violations

# 실행
for layer in ["facade", "handler", "service", "domain", "infrastructure"]:
    print(f"\n🔍 Checking {layer} layer...")
    violations = check_layer_dependencies(layer)
    if violations:
        print(f"❌ {len(violations)} violations found:")
        for v in violations:
            print(f"  - {v}")
    else:
        print(f"✅ OK: {layer} layer follows dependency rules")
```

## Output Format

```
=================================================
🏗️  Architecture Verification Report
=================================================

📋 Summary:
  ✅ Handler → Service (Interface): OK
  ✅ Domain → No External Deps: OK
  ✅ Infrastructure → Domain (Interface): OK
  ❌ Handler → Service (Implementation): 2 violations
  ✅ No Circular Imports: OK
  ❌ Relative Imports: 5 violations

=================================================
❌ Violations Found (7 total)
=================================================

Handler → Service Implementation (2):
  - src/beanllm/handler/core/chat_handler.py:10
    from beanllm.service.impl.core.chat_service_impl import ChatServiceImpl

  - src/beanllm/handler/core/rag_handler.py:8
    from beanllm.service.impl.core.rag_service_impl import RAGServiceImpl

Relative Imports (5):
  - src/beanllm/domain/loaders/pdf_loader.py:3
    from ...utils.logger import get_logger

  - src/beanllm/service/impl/core/chat_service_impl.py:5
    from ...domain.loaders import DocumentLoader

=================================================
💡 Recommendations
=================================================

1. Handler → Service Implementation:
   ✅ 해결: Service 인터페이스로 변경
   - from beanllm.service.chat_service import IChatService

2. Relative Imports:
   ✅ 해결: 절대 경로로 변경
   - from beanllm.utils.logger import get_logger
   - from beanllm.domain.loaders import DocumentLoader

=================================================
📚 Related Documents
=================================================
  - DEPENDENCY_RULES.md
  - .claude/rules/clean-architecture.md
  - ARCHITECTURE.md
```

## Auto-Fix Option

```
/arch-check --fix
```

`--fix` 옵션 사용 시:
1. 상대 경로 → 절대 경로 자동 변환
2. Handler → Service 구현체 → 인터페이스로 제안
3. 변경 사항 미리보기 제공
4. 사용자 승인 후 적용

## Related Commands

- `/refactor` - 의존성 위반 자동 리팩토링
- `/plan` - 아키텍처 개선 계획 수립

## Related Documents

- `DEPENDENCY_RULES.md` - 전체 의존성 규칙
- `.claude/rules/clean-architecture.md` - Clean Architecture 규칙
- `ARCHITECTURE.md` - 아키텍처 상세 설명
