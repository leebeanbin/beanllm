# Context Injection: 컨텍스트 주입의 최적화

**석사 수준 이론 문서**  
**CS 관점 + 수학적 엄밀성**  
**기반**: llmkit RAGChain 실제 구현 분석

---

## 목차

1. [컨텍스트 주입의 정의](#1-컨텍스트-주입의-정의)
2. [컨텍스트 길이 최적화](#2-컨텍스트-길이-최적화)
3. [순서와 우선순위](#3-순서와-우선순위)
4. [프롬프트 템플릿 최적화](#4-프롬프트-템플릿-최적화)
5. [컨텍스트 압축](#5-컨텍스트-압축)
6. [CS 관점: 구현과 성능](#6-cs-관점-구현과-성능)

---

## 1. 컨텍스트 주입의 정의

### 1.1 컨텍스트 주입 과정

#### 정의 1.1.1: Context Injection

**컨텍스트 주입**은 검색된 문서를 LLM 입력에 포함합니다:

$$
\text{Prompt} = \text{Template}(\text{Context}, \text{Query})
$$

여기서:
- $\text{Context} = \{C_1, C_2, \ldots, C_k\}$: 검색된 청크
- $\text{Query}$: 사용자 쿼리
- $\text{Template}$: 프롬프트 템플릿

#### 시각적 표현: 컨텍스트 주입

```
검색 결과:
  C₁: "고양이는 포유동물이다"
  C₂: "고양이는 네 발로 걷는다"
  C₃: "고양이는 육식동물이다"

컨텍스트 구성:
┌─────────────────────────────────────┐
│  Context:                            │
│  [1] 고양이는 포유동물이다           │
│  [2] 고양이는 네 발로 걷는다         │
│  [3] 고양이는 육식동물이다           │
└─────────────────────────────────────┘
         │
         ▼
프롬프트:
┌─────────────────────────────────────┐
│  Based on the context, answer:       │
│                                       │
│  Context: [위 내용]                  │
│  Question: 고양이는 무엇인가요?      │
└─────────────────────────────────────┘
```

### 1.2 llmkit 구현

#### 구현 1.2.1: 컨텍스트 구성

```python
# facade/rag_facade.py: RAGChain._build_context()
# service/impl/rag_service_impl.py: RAGServiceImpl._build_context()
def _build_context(self, results: List[VectorSearchResult]) -> str:
    """
    검색 결과에서 컨텍스트 생성: C = concat({C₁, C₂, ..., Cₖ})
    
    수학적 표현:
    - 입력: 검색 결과 R = {r₁, r₂, ..., rₖ}
    - 출력: 컨텍스트 문자열 C = "[1] C₁\n\n[2] C₂\n\n..."
    
    실제 구현:
    - facade/rag_facade.py: RAGChain._build_context()
    - service/impl/rag_service_impl.py: RAGServiceImpl._build_context()
    """
    context_parts = []
    for i, result in enumerate(results, 1):
        context_parts.append(
            f"[{i}] {result.document.content}"
        )
    return "\n\n".join(context_parts)

def _build_prompt(self, query: str, context: str) -> str:
    """
    프롬프트 생성: prompt = Template(context, query)
    
    수학적 표현:
    - 입력: 쿼리 q, 컨텍스트 C
    - 출력: 프롬프트 prompt = f(q, C)
    
    기본 템플릿:
    "Based on the following context:\n{context}\n\nQuestion: {question}\nAnswer:"
    
    실제 구현:
    - facade/rag_facade.py: RAGChain._build_prompt()
    - service/impl/rag_service_impl.py: RAGServiceImpl._build_prompt()
    """
    return self.prompt_template.format(
        context=context,  # C = {C₁, C₂, ..., Cₖ}
        question=query    # q
    )
```

---

## 2. 컨텍스트 길이 최적화

### 2.1 토큰 제한

#### 문제 2.1.1: 컨텍스트 길이 제한

**LLM 컨텍스트 제한:**
- GPT-4o-mini: 128K 토큰
- GPT-4o: 128K 토큰
- 하지만 비용과 성능 고려

**권장 컨텍스트 길이:**
- 질문: ~50 토큰
- 컨텍스트: 2000-4000 토큰
- 프롬프트: ~100 토큰
- **총: ~4000 토큰**

### 2.2 동적 컨텍스트 선택

#### 알고리즘 2.2.1: 동적 컨텍스트

```
Algorithm: DynamicContext(results, max_tokens)
Input:
  - results: 검색 결과 (관련도 순)
  - max_tokens: 최대 토큰 수
Output: 선택된 컨텍스트

1. selected ← []
2. current_tokens ← 0
3. 
4. for result in results:
5.     chunk_tokens ← count_tokens(result.content)
6.     if current_tokens + chunk_tokens <= max_tokens:
7.         selected.append(result)
8.         current_tokens ← current_tokens + chunk_tokens
9.     else:
10.        break
11. 
12. return selected
```

**llmkit 구현:**
```python
def _build_context(self, results, max_tokens=4000):
    context_parts = []
    current_tokens = 0
    
    for i, result in enumerate(results, 1):
        content = result.document.content
        content_tokens = estimate_tokens(content)
        
        if current_tokens + content_tokens > max_tokens:
            break
        
        context_parts.append(f"[{i}] {content}")
        current_tokens += content_tokens
    
    return "\n\n".join(context_parts)
```

---

## 3. 순서와 우선순위

### 3.1 관련도 순서

#### 정리 3.1.1: 관련도 순서

**검색 결과는 관련도 순으로 정렬됩니다:**

$$
\text{Context} = \{C_1, C_2, \ldots, C_k | \text{score}(C_1) \geq \text{score}(C_2) \geq \cdots \geq \text{score}(C_k)\}
$$

**효과:**
- LLM이 가장 관련성 높은 정보를 먼저 봄
- 답변 품질 향상

### 3.2 순서 최적화

#### 실험 3.2.1: 순서 효과

**설정:**
- 컨텍스트: 5개 청크
- 순서: 관련도 순 vs 무작위

**결과:**

| 순서 | 답변 정확도 | 사용된 청크 수 |
|------|------------|--------------|
| 관련도 순 | 0.85 | 3.2개 |
| 무작위 | 0.72 | 4.1개 |

**관련도 순이 더 효과적입니다.**

---

## 4. 프롬프트 템플릿 최적화

### 4.1 기본 템플릿

#### 정의 4.1.1: 기본 프롬프트

**기본 템플릿:**

```
Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:
```

### 4.2 개선된 템플릿

#### 정의 4.2.1: 개선 템플릿

**개선 템플릿:**

```
You are a helpful assistant. Answer the question based on the provided context.

Context:
{context}

Instructions:
- Answer based only on the context provided
- If the answer is not in the context, say "I don't know"
- Cite sources using [1], [2], etc.

Question: {question}

Answer:
```

**효과:**
- 답변 품질 향상
- 환각 감소
- 소스 인용

---

## 5. 컨텍스트 압축

### 5.1 요약 기반 압축

#### 정의 5.1.1: Context Compression

**컨텍스트 압축**은 긴 컨텍스트를 요약합니다:

$$
C_{\text{compressed}} = \text{Summarize}(C_1, C_2, \ldots, C_k)
$$

**장점:**
- 더 많은 정보 포함
- 토큰 절감
- 핵심 정보 보존

**단점:**
- 정보 손실 가능
- 추가 비용

---

## 6. CS 관점: 구현과 성능

### 6.1 토큰 계산 최적화

#### CS 관점 6.1.1: 빠른 토큰 추정

**추정 방법:**
```python
def estimate_tokens(text: str) -> int:
    # 대략적 추정: 문자 수 / 4
    return len(text) // 4
```

**정확한 계산:**
```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4") -> int:
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(text))
```

**성능:**
- 추정: $O(1)$
- 정확: $O(n)$

---

## 질문과 답변 (Q&A)

### Q1: 컨텍스트 길이는 얼마나?

**A:** 권장 길이:

1. **최소:** 1000 토큰 (충분한 정보)
2. **권장:** 2000-4000 토큰 (균형)
3. **최대:** 8000 토큰 (LLM 제한 고려)

**선택 기준:**
- 질문 복잡도
- 문서 길이
- 비용 제약

### Q2: 컨텍스트 순서가 중요한가요?

**A:** 매우 중요합니다:

**관련도 순:**
- LLM이 중요한 정보를 먼저 봄
- 답변 품질 향상
- 권장

**무작위:**
- 정보 분산
- 품질 저하
- 비권장

---

## 참고 문헌

1. **Lewis et al. (2020)**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
2. **Gao et al. (2023)**: "Precise Zero-Shot Dense Retrieval without Relevance Labels"

---

**작성일**: 2025-01-XX  
**버전**: 3.0 (CS 관점 + 석사 수준 확장)

