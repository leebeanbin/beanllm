# RAG 실무 가이드: 프로덕션 RAG 시스템 구축

**실무 적용 문서**  
**대상**: AI 엔지니어, 백엔드 개발자, 풀스택 개발자

---

## 목차

1. [RAG 시스템 아키텍처](#1-rag-시스템-아키텍처)
2. [문서 처리 파이프라인](#2-문서-처리-파이프라인)
3. [검색 전략 선택](#3-검색-전략-선택)
4. [프롬프트 엔지니어링](#4-프롬프트-엔지니어링)
5. [성능 최적화](#5-성능-최적화)
6. [프로덕션 배포](#6-프로덕션-배포)
7. [모니터링과 평가](#7-모니터링과-평가)

---

## 1. RAG 시스템 아키텍처

### 1.1 기본 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                  RAG 시스템 아키텍처                     │
└─────────────────────────────────────────────────────────┘

사용자 쿼리
    │
    ▼
┌──────────────┐
│  Query      │
│  Processing │  ← 쿼리 전처리, 확장
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Retrieval   │  ← 벡터 검색, 하이브리드
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Reranking   │  ← Cross-encoder 재순위화
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Context     │  ← 컨텍스트 구성
│  Building    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Generation  │  ← LLM 답변 생성
└──────┬───────┘
       │
       ▼
답변 + 소스
```

### 1.2 llmkit RAGChain 사용

```python
from llmkit import RAGChain

# 간단한 사용
rag = RAGChain.from_documents("documents.pdf")

# 질문하기
answer = rag.query("이 문서의 주요 내용은?")
```

---

## 2. 문서 처리 파이프라인

### 2.1 문서 로딩

```python
from llmkit import DocumentLoader

# 다양한 형식 지원
docs = DocumentLoader.load("file.pdf")
docs = DocumentLoader.load("file.docx")
docs = DocumentLoader.load("file.txt")
docs = DocumentLoader.load("https://example.com/page")
```

### 2.2 텍스트 분할 (Chunking)

**청킹 전략:**

```python
from llmkit import TextSplitter

# 기본 청킹
chunks = TextSplitter.split(
    documents=docs,
    chunk_size=500,      # 토큰 수
    chunk_overlap=50    # 겹침
)

# 의미 단위 청킹 (권장)
chunks = TextSplitter.split(
    documents=docs,
    chunk_size=500,
    chunk_overlap=50,
    separator="\n\n"    # 문단 단위
)
```

**청킹 가이드라인:**
- **chunk_size**: 200-1000 토큰 (권장: 500)
- **chunk_overlap**: chunk_size의 10-20% (권장: 50)
- **separator**: 의미 단위 (문단, 섹션)

### 2.3 임베딩 생성

```python
from llmkit import Embedding

emb = Embedding(model="text-embedding-3-small")

# 배치 처리로 효율적 생성
embeddings = emb.embed_sync([chunk.content for chunk in chunks])
```

### 2.4 벡터 저장소 구축

```python
from llmkit import VectorStore

# 자동 구축
store = VectorStore.from_documents(
    chunks,
    embedding_model="text-embedding-3-small"
)

# 또는 수동 구축
store = VectorStore()
store.add_documents(chunks, embeddings=embeddings)
```

---

## 3. 검색 전략 선택

### 3.1 기본 검색

```python
# 단순 유사도 검색
results = rag.retrieve("질문", k=5)
```

**사용 시기:**
- 빠른 응답 필요
- 단순한 검색 요구사항

### 3.2 MMR 검색

```python
# 다양성 고려 검색
results = rag.retrieve("질문", k=5, mmr=True, lambda_param=0.6)
```

**사용 시기:**
- 중복 제거 필요
- 다양한 관점 필요
- 추천 시스템

### 3.3 하이브리드 검색

```python
# 벡터 + 키워드 검색
results = rag.retrieve("질문", k=5, hybrid=True, alpha=0.7)
```

**사용 시기:**
- 정확한 키워드 매칭 중요
- 도메인 특화 용어
- 다국어 검색

### 3.4 재순위화 (Reranking)

```python
# Cross-encoder로 재순위화
results = rag.retrieve("질문", k=5, rerank=True)
```

**사용 시기:**
- 높은 정확도 필요
- 검색 품질이 중요
- 비용 여유

### 3.5 전략 조합

```python
# 모든 전략 조합
results = rag.retrieve(
    "질문",
    k=5,
    hybrid=True,      # 하이브리드 검색
    rerank=True,     # 재순위화
    mmr=True         # 다양성 고려
)
```

---

## 4. 프롬프트 엔지니어링

### 4.1 기본 프롬프트

```python
DEFAULT_PROMPT = """Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
```

### 4.2 개선된 프롬프트

```python
IMPROVED_PROMPT = """You are a helpful assistant. Answer the question based on the provided context.

Context:
{context}

Instructions:
- Answer based only on the context provided
- If the answer is not in the context, say "I don't know"
- Cite sources using [1], [2], etc.

Question: {question}

Answer:"""
```

### 4.3 도메인 특화 프롬프트

```python
MEDICAL_PROMPT = """You are a medical assistant. Provide accurate information based on the medical documents.

Medical Context:
{context}

Patient Question: {question}

Guidelines:
- Provide evidence-based answers
- Include relevant medical terms
- Note any limitations or uncertainties

Answer:"""
```

### 4.4 프롬프트 템플릿 사용

```python
rag = RAGChain(
    vector_store=store,
    llm=llm,
    prompt_template=IMPROVED_PROMPT
)
```

---

## 5. 성능 최적화

### 5.1 검색 최적화

**1. 적절한 k 값 선택**
```python
# 너무 작으면: 정보 부족
results = rag.retrieve(query, k=2)  # ❌

# 너무 크면: 노이즈 증가, 비용 증가
results = rag.retrieve(query, k=20)  # ❌

# 적절한 값: 4-10
results = rag.retrieve(query, k=5)  # ✅
```

**2. 컨텍스트 길이 제한**
```python
# 토큰 제한 고려
max_tokens = 4000
context = build_context(results, max_tokens=max_tokens)
```

### 5.2 LLM 최적화

**1. 모델 선택**
```python
# 빠른 응답: gpt-4o-mini
rag = RAGChain(llm=Client(model="gpt-4o-mini"))

# 높은 품질: gpt-4o
rag = RAGChain(llm=Client(model="gpt-4o"))
```

**2. Temperature 조절**
```python
# 사실 기반: 낮은 temperature
answer = rag.query(query, temperature=0.1)

# 창의적: 높은 temperature
answer = rag.query(query, temperature=0.7)
```

### 5.3 캐싱

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_query(query: str):
    return rag.query(query)
```

---

## 6. 프로덕션 배포

### 6.1 API 서버 구축

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
rag = RAGChain.from_documents("documents/")

class QueryRequest(BaseModel):
    question: str
    k: int = 5
    include_sources: bool = False

@app.post("/query")
async def query(request: QueryRequest):
    result = rag.query(
        request.question,
        k=request.k,
        include_sources=request.include_sources
    )
    return result
```

### 6.2 비동기 처리

```python
import asyncio
from llmkit import RAGChain

async def async_query(query: str):
    rag = RAGChain.from_documents("documents/")
    return await rag.aquery(query)  # 비동기 메서드
```

### 6.3 에러 처리

```python
def safe_query(query: str):
    try:
        return rag.query(query)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다."
```

---

## 7. 모니터링과 평가

### 7.1 주요 메트릭

```python
def evaluate_rag(query: str, expected_answer: str):
    answer, sources = rag.query(query, include_sources=True)
    
    metrics = {
        "answer_length": len(answer),
        "source_count": len(sources),
        "avg_source_score": sum(s.score for s in sources) / len(sources),
        "has_answer": expected_answer.lower() in answer.lower()
    }
    return metrics
```

### 7.2 로깅

```python
import logging

logger = logging.getLogger(__name__)

def query_with_logging(query: str):
    logger.info(f"Query: {query}")
    start_time = time.time()
    
    answer, sources = rag.query(query, include_sources=True)
    
    elapsed = time.time() - start_time
    logger.info(f"Answer generated in {elapsed:.2f}s")
    logger.debug(f"Sources: {len(sources)}")
    
    return answer, sources
```

---

## 베스트 프랙티스 체크리스트

- [ ] 문서 전처리 (정규화, 청킹)
- [ ] 적절한 임베딩 모델 선택
- [ ] 검색 전략 선택 (기본/MMR/하이브리드)
- [ ] 프롬프트 최적화
- [ ] 성능 모니터링
- [ ] 에러 처리
- [ ] 캐싱 구현
- [ ] 로깅 설정
- [ ] 비용 추적

---

## 참고 자료

- [이론: RAG 확률 모델](./01_rag_probabilistic_model.md)
- [임베딩 실무 가이드](../embeddings/practice_01_embeddings_usage.md)

---

**작성일**: 2025-01-XX  
**버전**: 1.0

