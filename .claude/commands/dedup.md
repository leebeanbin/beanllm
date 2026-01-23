# /dedup - Code Deduplication

**트리거**: `/dedup`
**모델**: sonnet
**설명**: 중복 코드 찾기 및 데코레이터 패턴으로 리팩토링

## Command Description

프로젝트 전체에서 중복 코드를 찾아내고, 데코레이터 패턴으로 리팩토링하여 85-90% 코드 감소를 달성합니다.

## Usage

```
/dedup
/dedup --path src/beanllm/service/impl/
/dedup --threshold 3
/dedup --auto-fix
```

## Options

- `--path`: 특정 경로만 검사 (기본: 전체 프로젝트)
- `--threshold`: 중복으로 간주할 최소 라인 수 (기본: 5줄)
- `--auto-fix`: 자동으로 데코레이터 패턴 적용 (사용자 승인 필요)

## Execution Steps

### 1. 중복 코드 패턴 감지

```bash
# Python 코드 중복 감지 도구 사용
pip install -q radon

# 코드 복잡도 분석
radon cc src/beanllm -a -nb

# 중복 코드 감지 (CPD - Copy/Paste Detector)
# 또는 직접 구현
python <<EOF
import ast
from collections import defaultdict
from pathlib import Path

def find_duplicates(threshold=5):
    """중복 코드 블록 찾기"""
    code_blocks = defaultdict(list)

    for py_file in Path("src/beanllm").rglob("*.py"):
        with open(py_file) as f:
            try:
                tree = ast.parse(f.read())
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        # 함수 body를 문자열로 변환
                        body_str = ast.unparse(node)
                        if len(body_str.split('\n')) >= threshold:
                            code_blocks[body_str].append((py_file, node.name))
            except SyntaxError:
                pass

    # 중복된 것만 필터
    duplicates = {k: v for k, v in code_blocks.items() if len(v) > 1}
    return duplicates

duplicates = find_duplicates(threshold=5)
print(f"Found {len(duplicates)} duplicate code patterns")
for code, locations in duplicates.items():
    print(f"\n{'='*60}")
    print(f"Duplicated {len(locations)} times:")
    for file, func in locations:
        print(f"  - {file}:{func}")
    print(f"Code preview:")
    print(code[:200] + "...")
EOF
```

### 2. 패턴 분류

중복 코드를 패턴별로 분류:

1. **캐싱 패턴** - `cache.get()`, `cache.set()`
2. **Rate Limiting 패턴** - `rate_limiter.acquire()`
3. **이벤트 스트리밍 패턴** - `event_publisher.publish()`
4. **분산 락 패턴** - `distributed_lock.lock()`
5. **재시도 패턴** - `for retry in range(max_retries)`
6. **로깅 패턴** - `logger.info()`, `logger.error()`
7. **에러 처리 패턴** - `try-except-finally`

### 3. 데코레이터 생성 제안

```python
# 예: 캐싱 패턴 → 데코레이터
# Before (중복 코드)
async def method_a(self, query):
    cache_key = f"rag:{query}"
    cached = await self._cache.get(cache_key)
    if cached:
        return cached
    results = self._process(query)
    await self._cache.set(cache_key, results, ttl=3600)
    return results

async def method_b(self, query):
    cache_key = f"vision_rag:{query}"
    cached = await self._cache.get(cache_key)
    if cached:
        return cached
    results = self._process(query)
    await self._cache.set(cache_key, results, ttl=3600)
    return results

# After (데코레이터)
def with_cache(prefix: str, ttl: int = 3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            cache_key = f"{prefix}:{args[0]}"
            cached = await self._cache.get(cache_key)
            if cached:
                return cached

            result = await func(self, *args, **kwargs)
            await self._cache.set(cache_key, result, ttl=ttl)
            return result
        return wrapper
    return decorator

@with_cache(prefix="rag", ttl=3600)
async def method_a(self, query):
    return self._process(query)

@with_cache(prefix="vision_rag", ttl=3600)
async def method_b(self, query):
    return self._process(query)
```

### 4. 자동 리팩토링 (--auto-fix)

```python
# AST를 사용한 자동 리팩토링
import ast

class DedupTransformer(ast.NodeTransformer):
    def visit_AsyncFunctionDef(self, node):
        # 캐싱 패턴 감지
        if self._has_caching_pattern(node):
            # 데코레이터 추가
            decorator = ast.Name(id="with_cache", ctx=ast.Load())
            node.decorator_list.append(decorator)

            # 캐싱 코드 제거
            node.body = self._remove_caching_code(node.body)

        return node

    def _has_caching_pattern(self, node):
        # AST를 순회하며 캐싱 패턴 확인
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                if "cache_key" in ast.unparse(stmt):
                    return True
        return False

# 적용
tree = ast.parse(source_code)
transformer = DedupTransformer()
new_tree = transformer.visit(tree)
new_code = ast.unparse(new_tree)
```

## Output Format

```
=================================================
🔍 Code Deduplication Report
=================================================

📊 Statistics:
  Total files scanned: 142
  Duplicate patterns found: 23
  Code lines duplicated: 1,847
  Potential savings: ~1,568 lines (85%)

=================================================
📋 Duplicate Patterns by Category
=================================================

1. ⚡ Caching Pattern (8 occurrences, 456 lines)
   Locations:
   - src/beanllm/service/impl/core/rag_service_impl.py:query (65 lines)
   - src/beanllm/service/impl/advanced/vision_rag_service_impl.py:retrieve (58 lines)
   - src/beanllm/service/impl/ml/audio_service_impl.py:transcribe (52 lines)
   ...

   💡 Recommendation:
   ✅ Create @with_cache decorator
   ✅ Reduces to ~3-5 lines per method
   ✅ Savings: 456 → 40 lines (91% reduction)

2. 🚦 Rate Limiting Pattern (5 occurrences, 275 lines)
   Locations:
   - src/beanllm/service/impl/core/chat_service_impl.py:chat (55 lines)
   - src/beanllm/service/impl/core/rag_service_impl.py:query (55 lines)
   ...

   💡 Recommendation:
   ✅ Use @with_distributed_features decorator
   ✅ Savings: 275 → 25 lines (91% reduction)

3. 📣 Event Streaming Pattern (4 occurrences, 240 lines)
   ...

4. 🔒 Distributed Lock Pattern (3 occurrences, 180 lines)
   ...

5. 🔄 Retry Pattern (3 occurrences, 156 lines)
   ...

=================================================
🎯 Recommended Actions
=================================================

1. Apply @with_distributed_features decorator (18 methods)
   Before: 1,151 lines
   After: 90 lines
   Savings: 92%

2. Create custom decorators (5 new decorators)
   - @with_retry
   - @with_logging
   - @with_validation
   - @with_error_handling
   - @with_metrics

3. Extract helper methods (12 candidates)
   - _create_content_from_row (CSV processing)
   - _validate_file_path (file operations)
   ...

=================================================
💾 Auto-Fix Preview
=================================================

File: src/beanllm/service/impl/core/rag_service_impl.py

--- Before (65 lines)
async def query(self, request: RAGRequest):
    # Caching logic (20 lines)
    if self._cache_enabled:
        cache_key = f"rag:{request.query}"
        ...

    # Rate limiting logic (15 lines)
    if self._rate_limiter:
        await self._rate_limiter.acquire(...)
        ...

    # Business logic (5 lines)
    results = self._vector_store.search(...)
    return results

+++ After (5 lines)
@with_distributed_features(
    pipeline_type="rag",
    enable_cache=True,
    enable_rate_limiting=True,
)
async def query(self, request: RAGRequest):
    results = self._vector_store.search(...)
    return results

Proceed with auto-fix? (y/n)
```

## Metrics Tracking

리팩토링 전후 비교:

```
📈 Before Refactoring:
  Total lines: 15,432
  Duplicate lines: 1,847
  Average method length: 35 lines
  Code duplication: 12%

📉 After Refactoring:
  Total lines: 13,864
  Duplicate lines: 279
  Average method length: 8 lines
  Code duplication: 2%

✅ Improvement:
  Lines reduced: 1,568 (10.2%)
  Duplication reduced: 85%
  Readability: +90%
  Maintainability: +85%
```

## Related Commands

- `/arch-check` - 아키텍처 검증
- `/refactor` - 코드 리팩토링

## Related Documents

- `.claude/rules/code-quality.md` - 코드 품질 규칙
- `.claude/skills/decorator-pattern.md` - 데코레이터 패턴 스킬
- `.cursorrules` - 코딩 스타일
