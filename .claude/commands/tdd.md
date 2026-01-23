# /tdd - Test-Driven Development Guide

**트리거**: `/tdd`
**모델**: sonnet
**설명**: TDD (Red-Green-Refactor) 워크플로우 가이드

## Command Description

Test-Driven Development 방법론에 따라 테스트를 먼저 작성하고, 구현하고, 리팩토링하는 전체 사이클을 안내합니다.

## Usage

```bash
/tdd
/tdd --feature "HyDE query expansion"
/tdd --class "RAGService"
```

## Workflow: Red-Green-Refactor

### Phase 1: RED (실패하는 테스트 작성)

**목표**: 요구사항을 테스트 코드로 명확히 정의

```python
# Step 1: 테스트 파일 생성
# tests/domain/retrieval/test_hyde.py

import pytest
from beanllm.domain.retrieval.hyde import HyDEQueryExpander

def test_hyde_generates_hypothetical_document():
    """HyDE가 쿼리에 대한 가상 문서를 생성해야 함"""
    # Arrange
    expander = HyDEQueryExpander(model="gpt-4o")
    query = "What is RAG?"

    # Act
    hypothetical = await expander.generate_hypothetical(query)

    # Assert
    assert hypothetical is not None
    assert len(hypothetical) > 0
    assert "retrieval" in hypothetical.lower() or "rag" in hypothetical.lower()

def test_hyde_expands_query_with_embedding():
    """HyDE가 가상 문서를 임베딩하여 쿼리를 확장해야 함"""
    # Arrange
    expander = HyDEQueryExpander(model="gpt-4o")
    query = "Explain transformers"

    # Act
    expanded_embedding = await expander.expand_query(query)

    # Assert
    assert expanded_embedding is not None
    assert len(expanded_embedding) == 1536  # OpenAI embedding dimension
    assert all(isinstance(x, float) for x in expanded_embedding)

def test_hyde_handles_empty_query():
    """HyDE가 빈 쿼리를 적절히 처리해야 함"""
    # Arrange
    expander = HyDEQueryExpander(model="gpt-4o")

    # Act & Assert
    with pytest.raises(ValueError, match="Query cannot be empty"):
        await expander.expand_query("")
```

**실행**:
```bash
pytest tests/domain/retrieval/test_hyde.py -v
```

**예상 결과**: 🔴 **모든 테스트 실패** (아직 구현 안 됨)

### Phase 2: GREEN (최소 구현으로 테스트 통과)

**목표**: 테스트를 통과하는 최소한의 코드 작성

```python
# src/beanllm/domain/retrieval/hyde.py

from typing import List
from openai import AsyncOpenAI

class HyDEQueryExpander:
    """Hypothetical Document Embeddings for query expansion."""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self._client = AsyncOpenAI()

    async def generate_hypothetical(self, query: str) -> str:
        """Generate a hypothetical document that would answer the query."""
        if not query:
            raise ValueError("Query cannot be empty")

        # 최소 구현: 간단한 프롬프트로 가상 문서 생성
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Generate a concise document that would answer this question."
                },
                {"role": "user", "content": query}
            ],
            max_tokens=200
        )
        return response.choices[0].message.content

    async def expand_query(self, query: str) -> List[float]:
        """Expand query by generating and embedding hypothetical document."""
        if not query:
            raise ValueError("Query cannot be empty")

        # 가상 문서 생성
        hypothetical = await self.generate_hypothetical(query)

        # 가상 문서 임베딩 (쿼리 직접 임베딩 대신)
        embedding_response = await self._client.embeddings.create(
            model="text-embedding-3-small",
            input=hypothetical
        )
        return embedding_response.data[0].embedding
```

**실행**:
```bash
pytest tests/domain/retrieval/test_hyde.py -v
```

**예상 결과**: 🟢 **모든 테스트 통과**

### Phase 3: REFACTOR (코드 품질 개선)

**목표**: 중복 제거, 가독성 향상, 성능 최적화 (테스트는 계속 통과)

#### 3.1 설정 추출

```python
# src/beanllm/domain/retrieval/hyde.py

from dataclasses import dataclass

@dataclass
class HyDEConfig:
    """HyDE configuration."""
    llm_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    max_tokens: int = 200
    system_prompt: str = "Generate a concise document that would answer this question."

class HyDEQueryExpander:
    def __init__(self, config: HyDEConfig = None):
        self.config = config or HyDEConfig()
        self._client = AsyncOpenAI()
```

#### 3.2 에러 처리 강화

```python
from beanllm.utils.exceptions import APIError

async def generate_hypothetical(self, query: str) -> str:
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    try:
        response = await self._client.chat.completions.create(
            model=self.config.llm_model,
            messages=[
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": query}
            ],
            max_tokens=self.config.max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        raise APIError(f"Failed to generate hypothetical document: {e}")
```

#### 3.3 캐싱 추가 (성능 최적화)

```python
from functools import lru_cache

class HyDEQueryExpander:
    def __init__(self, config: HyDEConfig = None):
        self.config = config or HyDEConfig()
        self._client = AsyncOpenAI()
        self._cache = {}  # 간단한 인메모리 캐시

    async def expand_query(self, query: str) -> List[float]:
        # 캐시 확인
        if query in self._cache:
            return self._cache[query]

        # 생성 및 캐싱
        embedding = await self._generate_and_embed(query)
        self._cache[query] = embedding
        return embedding
```

**실행**:
```bash
pytest tests/domain/retrieval/test_hyde.py -v
```

**예상 결과**: 🟢 **테스트 여전히 통과** (리팩토링 성공)

## TDD Best Practices for beanllm

### 1. 테스트 파일 구조

```
tests/
├── domain/
│   ├── retrieval/
│   │   ├── test_hyde.py           # HyDE 단위 테스트
│   │   ├── test_reranker.py       # Reranker 단위 테스트
│   │   └── test_query_expansion.py
│   └── loaders/
│       └── test_directory_loader.py
├── service/
│   └── test_rag_service.py        # RAG 서비스 통합 테스트
└── integration/
    └── test_rag_end_to_end.py     # E2E 테스트
```

### 2. 테스트 커버리지 목표

- **Domain Layer**: 100% (핵심 비즈니스 로직)
- **Service Layer**: 90%+
- **Handler/Facade**: 80%+
- **Infrastructure**: 70%+ (외부 의존성 많음)

### 3. 테스트 명명 규칙

```python
def test_[method]_[scenario]_[expected_result]():
    """[What should happen in this scenario]"""
    pass

# ✅ Good
def test_expand_query_with_valid_input_returns_embedding():
    """expand_query should return 1536-dim embedding for valid input"""
    pass

# ❌ Bad
def test_expand_query():
    pass
```

### 4. AAA 패턴 (Arrange-Act-Assert)

```python
def test_generate_hypothetical_with_empty_query_raises_error():
    # Arrange
    expander = HyDEQueryExpander()
    empty_query = ""

    # Act & Assert
    with pytest.raises(ValueError, match="Query cannot be empty"):
        await expander.generate_hypothetical(empty_query)
```

### 5. Fixtures 활용

```python
# conftest.py
@pytest.fixture
async def hyde_expander():
    """HyDE expander fixture with test configuration."""
    config = HyDEConfig(
        llm_model="gpt-4o-mini",  # 테스트용 저렴한 모델
        max_tokens=100
    )
    expander = HyDEQueryExpander(config)
    yield expander
    # Cleanup if needed

# test_hyde.py
def test_generate_hypothetical_with_fixture(hyde_expander):
    result = await hyde_expander.generate_hypothetical("What is AI?")
    assert result is not None
```

## TDD Workflow Checklist

### 🔴 RED Phase
- [ ] 요구사항을 명확히 이해했는가?
- [ ] 테스트가 실패하는지 확인했는가? (테스트 실행 후 RED 확인)
- [ ] 엣지 케이스를 고려했는가? (빈 입력, None, 큰 값 등)
- [ ] 에러 처리 테스트를 작성했는가?

### 🟢 GREEN Phase
- [ ] 최소한의 코드로 테스트를 통과시켰는가?
- [ ] 모든 테스트가 통과하는가?
- [ ] 타입 힌트를 추가했는가?
- [ ] Docstring을 작성했는가?

### 🔵 REFACTOR Phase
- [ ] 중복 코드를 제거했는가?
- [ ] 변수/함수명이 명확한가?
- [ ] 성능 최적화가 필요한가? (프로파일링)
- [ ] Clean Architecture 규칙을 준수하는가? (`/arch-check`)
- [ ] 테스트가 여전히 통과하는가?

## Integration with Other Commands

```bash
# 1. TDD 사이클 시작
/tdd --feature "HyDE query expansion"

# 2. RED: 테스트 작성
# [테스트 코드 작성]

# 3. GREEN: 구현
# [구현 코드 작성]

# 4. REFACTOR: 중복 제거
/dedup

# 5. Architecture 검증
/arch-check

# 6. 커버리지 확인
pytest --cov=src/beanllm --cov-report=term

# 7. 코드 리뷰
/code-review
```

## Example: Full TDD Cycle for RAG Feature

### 1. Start TDD
```bash
/tdd --feature "Add HyDE to RAG pipeline"
```

### 2. RED - Write failing test
```python
# tests/service/test_rag_service.py
def test_rag_service_uses_hyde_for_query_expansion():
    service = RAGServiceImpl(use_hyde=True)
    result = await service.query("What is RAG?", k=5)

    # HyDE should improve retrieval accuracy
    assert result.metadata["used_hyde"] is True
    assert len(result.sources) == 5
```

### 3. GREEN - Minimal implementation
```python
# src/beanllm/service/impl/core/rag_service_impl.py
from beanllm.domain.retrieval.hyde import HyDEQueryExpander

class RAGServiceImpl:
    def __init__(self, use_hyde: bool = False):
        self._use_hyde = use_hyde
        if use_hyde:
            self._hyde = HyDEQueryExpander()

    async def query(self, query: str, k: int = 5):
        if self._use_hyde:
            expanded_embedding = await self._hyde.expand_query(query)
            results = self._vector_store.similarity_search_by_vector(
                expanded_embedding, k=k
            )
        else:
            results = self._vector_store.similarity_search(query, k=k)

        return RAGResponse(
            sources=results,
            metadata={"used_hyde": self._use_hyde}
        )
```

### 4. REFACTOR - Improve quality
```bash
/dedup           # Find duplicate code
/arch-check      # Verify architecture
```

### 5. Verify
```bash
pytest --cov=src/beanllm/service --cov-report=term
```

## Quick Reference

| Phase | Command | Purpose |
|-------|---------|---------|
| Start | `/tdd` | Begin TDD cycle |
| RED | Write test | Define requirements |
| GREEN | Write code | Pass tests minimally |
| REFACTOR | `/dedup` | Remove duplication |
| VERIFY | `/arch-check` | Check architecture |
| REVIEW | `/code-review` | Final quality check |

## Related Documents

- `.claude/rules/testing.md` - Testing standards
- `.claude/skills/tdd-workflow/README.md` - TDD methodology
- `CLAUDE.md` - TDD workflow section

---

**💡 Remember**:
1. ⛔ **RED**: Write a failing test first
2. ✅ **GREEN**: Make it pass with minimal code
3. ♻️ **REFACTOR**: Improve without breaking tests

**🎯 Goal**: 80% test coverage with high-quality, maintainable code
