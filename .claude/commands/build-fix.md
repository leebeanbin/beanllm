# /build-fix - Build Error Fixer

**트리거**: `/build-fix`
**모델**: sonnet
**설명**: 빌드/린트/타입 에러 자동 수정

## Command Description

빌드, 린트, 타입 체크 에러를 자동으로 분석하고 수정합니다. Black, Ruff, MyPy 에러를 한 번에 해결합니다.

## Usage

```
/build-fix
/build-fix --check-only
/build-fix --type python
/build-fix --type typescript
```

## Options

- `--check-only`: 수정 없이 에러만 확인
- `--type`: 언어 지정 (`python`, `typescript`)
- `--auto-fix`: 사용자 승인 없이 자동 수정 (주의)

## Execution Steps

### 1. Python 빌드 체크

```bash
echo "🔍 Checking Python build..."

# Black 포매팅 체크
black --check src/beanllm/

# Ruff 린트 체크
ruff check src/beanllm/

# MyPy 타입 체크
mypy src/beanllm/

# pytest 실행 (빌드 검증)
pytest tests/ --tb=short -x
```

### 2. TypeScript 빌드 체크

```bash
echo "🔍 Checking TypeScript build..."

cd playground/frontend

# TypeScript 컴파일
pnpm tsc --noEmit

# ESLint
pnpm eslint src/

# Next.js 빌드
pnpm build
```

### 3. 에러 분석 및 수정

```python
# Black 에러 자동 수정
black src/beanllm/

# Ruff 에러 자동 수정
ruff check --fix src/beanllm/

# MyPy 에러 분석
# 1. Missing imports → import 추가
# 2. Type annotation missing → 타입 힌트 추가
# 3. Type mismatch → 타입 수정

# 예시: Missing import 수정
# ❌ Error: Cannot find implementation or library stub for module named 'httpx'
# ✅ Fix: pip install httpx (또는 requirements.txt 확인)

# 예시: Type annotation 수정
# ❌ Error: Function is missing a return type annotation
def get_embedding(text):  # ❌
    return [0.1, 0.2, 0.3]

# ✅ Fix:
def get_embedding(text: str) -> List[float]:  # ✅
    return [0.1, 0.2, 0.3]
```

## Common Error Patterns

### Python

#### 1. Import Errors

```python
# ❌ Error: Module 'beanllm.domain.loaders' has no attribute 'DocumentLoader'
# 원인: __init__.py에서 export하지 않음

# ✅ Fix: __init__.py에 추가
from beanllm.domain.loaders.loaders import DocumentLoader

__all__ = ["DocumentLoader"]
```

#### 2. Type Errors

```python
# ❌ Error: Incompatible return value type (got "None", expected "str")
def get_model_name(model: str) -> str:
    if model in MODELS:
        return MODELS[model]
    # ❌ Implicit None return

# ✅ Fix: 명시적 에러 처리
def get_model_name(model: str) -> str:
    if model in MODELS:
        return MODELS[model]
    raise ValueError(f"Unknown model: {model}")  # ✅
```

#### 3. Missing Type Annotations

```python
# ❌ Error: Function is missing a return type annotation
def calculate_similarity(a, b):
    return sum(x * y for x, y in zip(a, b))

# ✅ Fix: 타입 힌트 추가
from typing import List

def calculate_similarity(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
```

### TypeScript

#### 1. Type Errors

```typescript
// ❌ Error: Property 'content' does not exist on type 'Message'
interface Message {
  role: string;
}

function displayMessage(msg: Message) {
  return msg.content;  // ❌
}

// ✅ Fix: 타입 정의 수정
interface Message {
  role: string;
  content: string;  // ✅ 추가
}

function displayMessage(msg: Message) {
  return msg.content;  // ✅
}
```

#### 2. Null/Undefined Errors

```typescript
// ❌ Error: Object is possibly 'undefined'
function getFirstMessage(messages?: Message[]) {
  return messages[0].content;  // ❌
}

// ✅ Fix: Optional chaining
function getFirstMessage(messages?: Message[]) {
  return messages?.[0]?.content;  // ✅
}
```

## Output Format

```
=================================================
🔧 Build Fix Report
=================================================

📊 Summary:
  Python errors: 12
  TypeScript errors: 5
  Total: 17

=================================================
🐍 Python Errors (12)
=================================================

Black Formatting (3):
  ✅ Auto-fixed:
  - src/beanllm/domain/loaders/pdf_loader.py
  - src/beanllm/service/impl/core/rag_service_impl.py
  - src/beanllm/facade/core/client_facade.py

Ruff Lint (4):
  ✅ Auto-fixed:
  - F401: Unused import in src/beanllm/utils/logger.py
  - E501: Line too long in src/beanllm/domain/retrieval/hyde.py

  ⚠️  Manual fix needed:
  - F841: Local variable 'result' is assigned but never used
    File: src/beanllm/service/impl/core/chat_service_impl.py:45
    Fix: Remove unused variable or use it

MyPy Type Errors (5):
  ✅ Fixed:
  - Missing return type annotation (3 files)
    Added: -> List[float], -> ChatResponse, -> str

  ⚠️  Manual fix needed:
  - Incompatible return value type
    File: src/beanllm/utils/token_counter.py:25
    Expected: int
    Got: None

    Fix needed:
    ```python
    def count_tokens(text: str) -> int:
        if not text:
            return 0  # ✅ Add explicit return
        return len(text.split())
    ```

=================================================
📘 TypeScript Errors (5)
=================================================

Type Errors (3):
  ✅ Fixed:
  - Property 'content' does not exist on type 'Message'
    File: src/components/ChatMessage.tsx:12
    Fixed: Added 'content' to Message interface

  ⚠️  Manual fix needed:
  - Object is possibly 'undefined'
    File: src/hooks/useChatStream.ts:45
    Fix: Use optional chaining (messages?.[0])

ESLint (2):
  ✅ Auto-fixed:
  - no-unused-vars in src/lib/api.ts
  - prefer-const in src/components/ChatInput.tsx

=================================================
✅ Auto-fixed: 10/17 (59%)
⚠️  Manual fixes needed: 7/17 (41%)
=================================================

Next steps:
1. Review auto-fixed changes
2. Apply manual fixes above
3. Re-run build checks
4. Commit changes

Run tests now? (y/n)
```

## Auto-fix Script

```bash
#!/bin/bash

echo "🔧 Auto-fixing build errors..."

# Python
echo "📝 Formatting Python code..."
black src/beanllm/

echo "🔍 Fixing Ruff errors..."
ruff check --fix src/beanllm/

echo "🔍 Checking MyPy..."
mypy src/beanllm/ --show-error-codes

# TypeScript
if [ -d "playground/frontend" ]; then
    echo "📝 Formatting TypeScript code..."
    cd playground/frontend
    pnpm prettier --write src/

    echo "🔍 Fixing ESLint errors..."
    pnpm eslint --fix src/

    cd ../..
fi

echo "✅ Auto-fix complete!"
echo "Run 'pytest' to verify all tests pass"
```

## Related Commands

- `/test-gen` - 테스트 생성
- `/arch-check` - 아키텍처 검증

## Invocation Example

```
User: /build-fix

Claude: [Runs build checks]

🔍 Checking Python build...
  Black: 3 files need formatting
  Ruff: 7 issues found
  MyPy: 5 type errors

Auto-fixing...
  ✅ Black: 3 files formatted
  ✅ Ruff: 4 issues fixed
  ⚠️  MyPy: 2 issues need manual fix

[Shows detailed report above]

Apply auto-fixes? (y/n)
```
