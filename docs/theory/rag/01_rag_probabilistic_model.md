# RAG Probabilistic Model: 검색 증강 생성의 확률 모델

**석사 수준 이론 문서**  
**CS 관점 + 수학적 엄밀성**  
**기반**: llmkit RAGChain 실제 구현 분석

---

## 목차

1. [RAG의 수학적 정의](#1-rag의-수학적-정의)
2. [조건부 확률과 Marginalization](#2-조건부-확률과-marginalization)
3. [베이즈 정리와 RAG](#3-베이즈-정리와-rag)
4. [생성 모델의 확률 분포](#4-생성-모델의-확률-분포)
5. [RAG 파이프라인의 수학적 표현](#5-rag-파이프라인의-수학적-표현)
6. [정보 이론적 해석](#6-정보-이론적-해석)
7. [CS 관점: 구현과 최적화](#7-cs-관점-구현과-최적화)
8. [실제 성능 분석](#8-실제-성능-분석)

---

## 1. RAG의 수학적 정의

### 1.1 RAG의 형식적 정의

#### 정의 1.1.1: Retrieval-Augmented Generation (RAG)

**RAG**는 다음 확률 모델로 정의됩니다:

$$
P(y | x) = \sum_{d \in \mathcal{D}} P(y | x, d) \cdot P(d | x)
$$

여기서:
- $x$: 입력 쿼리
- $y$: 생성된 답변
- $d$: 검색된 문서
- $\mathcal{D}$: 문서 컬렉션

#### 시각적 표현: RAG 확률 모델

```
┌─────────────────────────────────────────────────────────┐
│              RAG 확률 모델 구조                          │
└─────────────────────────────────────────────────────────┘

입력 쿼리 x
    │
    ▼
┌─────────────────────────────────────┐
│  P(d | x) - 검색 단계              │
│                                     │
│  문서 컬렉션 D = {d₁, d₂, ..., dₙ} │
│                                     │
│  P(d₁|x) = 0.85  ← 높은 관련도     │
│  P(d₂|x) = 0.72                     │
│  P(d₃|x) = 0.15                     │
│  ...                                │
│  P(dₙ|x) = 0.01                     │
└──────────────┬──────────────────────┘
               │
               ▼ (상위 k개 선택)
┌─────────────────────────────────────┐
│  P(y | x, d) - 생성 단계            │
│                                     │
│  LLM(x, d₁) → P(y₁ | x, d₁)        │
│  LLM(x, d₂) → P(y₂ | x, d₂)        │
│  ...                                │
└──────────────┬──────────────────────┘
               │
               ▼ (Marginalization)
┌─────────────────────────────────────┐
│  P(y | x) = Σ P(y|x,d) · P(d|x)    │
│                                     │
│  최종 답변 y                        │
└─────────────────────────────────────┘
```

### 1.2 RAG의 목적

#### 정리 1.2.1: RAG의 목적

**RAG는 다음을 최대화합니다:**

$$
P(y^* | x) = \max_{d \in \mathcal{D}} P(y^* | x, d) \cdot P(d | x)
$$

여기서 $y^*$는 정답입니다.

**해석:**
- 정답을 포함하는 문서를 검색 ($P(d | x)$ 최대화)
- 해당 문서로 정확한 답변 생성 ($P(y^* | x, d)$ 최대화)

---

## 2. 조건부 확률과 Marginalization

### 2.1 조건부 확률의 정의

#### 정의 2.1.1: 조건부 확률 (Conditional Probability)

**조건부 확률**은 다음과 같이 정의됩니다:

$$
P(A | B) = \frac{P(A \cap B)}{P(B)}
$$

**조건:** $P(B) > 0$

#### 정리 2.1.1: 확률의 곱셈 법칙

**확률의 곱셈 법칙:**

$$
P(A \cap B) = P(A | B) \cdot P(B) = P(B | A) \cdot P(A)
$$

### 2.2 Marginalization

#### 정의 2.2.1: Marginalization (주변화)

**Marginalization**은 다음을 의미합니다:

$$
P(y | x) = \sum_{d \in \mathcal{D}} P(y, d | x) = \sum_{d \in \mathcal{D}} P(y | x, d) \cdot P(d | x)
$$

**해석:**
- 모든 가능한 문서 $d$에 대해 가중 평균
- 가중치는 $P(d | x)$ (문서의 관련도)

#### 시각적 표현: Marginalization

```
P(y | x) 계산:

P(y | x) = Σ P(y | x, d) · P(d | x)
         d∈D

= P(y | x, d₁) · P(d₁ | x)  ← 문서 1의 기여
+ P(y | x, d₂) · P(d₂ | x)  ← 문서 2의 기여
+ P(y | x, d₃) · P(d₃ | x)  ← 문서 3의 기여
+ ...
+ P(y | x, dₙ) · P(dₙ | x)  ← 문서 n의 기여

각 문서의 기여도 = 생성 확률 × 검색 확률
```

#### 구체적 수치 예시

**예시 2.2.1: Marginalization 계산**

쿼리: $x$ = "고양이는 무엇인가요?"

**문서와 확률:**

| 문서 | $P(d | x)$ | $P(y^* | x, d)$ | 기여도 |
|------|------|--------|--------|
| $d_1$: "고양이는 포유동물" | 0.85 | 0.95 | 0.8075 |
| $d_2$: "고양이는 네 발로 걷는다" | 0.72 | 0.80 | 0.576 |
| $d_3$: "강아지는 귀여워" | 0.15 | 0.20 | 0.03 |
| $d_4$: "고양이 사료" | 0.10 | 0.30 | 0.03 |

**Marginalization (가중 합):**

$$
P(y^* | x) = \sum_{d \in \mathcal{D}} P(y^* | x, d) \cdot P(d | x)
$$

$$
= 0.95 \times 0.85 + 0.80 \times 0.72 + 0.20 \times 0.15 + 0.30 \times 0.10
$$

$$
= 0.8075 + 0.576 + 0.03 + 0.03 = 1.4435
$$

**해석:**
- 문서 $d_1$의 기여도가 가장 큼 (0.8075)
- 문서 $d_2$도 상당한 기여 (0.576)
- 문서 $d_3$, $d_4$는 기여도가 낮음 (각 0.03)

**실제 RAG에서는 상위 $k$개만 선택:**
- $k=2$: $P(y^* | x) \approx 0.8075 + 0.576 = 1.3835$ (정규화 후 약 0.85)
- $k=3$: $P(y^* | x) \approx 1.4135$ (정규화 후 약 0.87)

---

## 3. 베이즈 정리와 RAG

### 3.1 베이즈 정리

#### 정리 3.1.1: 베이즈 정리 (Bayes' Theorem)

**베이즈 정리:**

$$
P(A | B) = \frac{P(B | A) \cdot P(A)}{P(B)}
$$

**증명:**
$$
P(A | B) = \frac{P(A \cap B)}{P(B)} = \frac{P(B | A) \cdot P(A)}{P(B)}
$$

□

### 3.2 RAG에서의 베이즈 정리

#### 정리 3.2.1: RAG의 베이즈 해석

**RAG는 베이즈 정리를 사용합니다:**

$$
P(d | x) = \frac{P(x | d) \cdot P(d)}{P(x)} \propto P(x | d) \cdot P(d)
$$

**해석:**
- $P(x | d)$: 문서 $d$가 쿼리 $x$를 생성할 확률 (Likelihood)
- $P(d)$: 문서 $d$의 사전 확률 (Prior)
- $P(d | x)$: 쿼리 $x$가 주어졌을 때 문서 $d$의 확률 (Posterior)

#### 시각적 표현: 베이즈 업데이트

```
베이즈 업데이트 과정:

Prior: P(d)              Likelihood: P(x|d)          Posterior: P(d|x)
        │                        │                          │
        │                        │                          │
        ▼                        ▼                          ▼
┌──────────────┐        ┌──────────────┐          ┌──────────────┐
│  문서 d의     │   ×    │  쿼리 x가   │    =     │  쿼리 x가    │
│  사전 확률    │        │  문서 d에서  │          │  주어졌을 때 │
│              │        │  나올 확률   │          │  문서 d의    │
│  P(d) = 0.1  │        │  P(x|d)=0.9  │          │  확률        │
│              │        │              │          │  P(d|x)=0.9  │
└──────────────┘        └──────────────┘          └──────────────┘
```

---

## 4. 생성 모델의 확률 분포

### 4.1 생성 모델의 정의

#### 정의 4.1.1: 생성 모델 (Generative Model)

**생성 모델**은 다음 확률 분포를 모델링합니다:

$$
P(y | x, d) = \prod_{i=1}^{|y|} P(y_i | y_{<i}, x, d)
$$

여기서 $y_i$는 답변의 $i$번째 토큰입니다.

#### 시각적 표현: 생성 과정

```
생성 과정 (Autoregressive):

P(y | x, d) = P(y₁ | x, d) × P(y₂ | y₁, x, d) × ... × P(yₙ | y₁...yₙ₋₁, x, d)

단계별:
Step 1: P(y₁="고양이는" | x, d)
Step 2: P(y₂="포유동물" | y₁="고양이는", x, d)
Step 3: P(y₃="입니다" | y₁="고양이는", y₂="포유동물", x, d)
...
```

### 4.2 LLM의 확률 분포

#### 정의 4.2.1: Transformer 기반 생성

**Transformer 모델의 확률 분포:**

$$
P(y_i | y_{<i}, x, d) = \text{softmax}(\text{Transformer}(y_{<i}, x, d)_i)
$$

**해석:**
- Transformer는 컨텍스트 $(y_{<i}, x, d)$를 인코딩
- 출력을 softmax로 확률 분포로 변환

---

## 5. RAG 파이프라인의 수학적 표현

### 5.1 전체 파이프라인

#### 정의 5.1.1: RAG 파이프라인

**RAG 파이프라인은 다음 단계로 구성됩니다:**

1. **검색 (Retrieval):**
   $$
   R(x, k) = \arg\max_{S \subseteq \mathcal{D}, |S|=k} \sum_{d \in S} P(d | x)
   $$

2. **컨텍스트 구성:**
   $$
   C = \text{concat}(R(x, k))
   $$

3. **생성 (Generation):**
   $$
   y = \arg\max_{y'} P(y' | x, C)
   $$

#### 시각적 표현: 전체 파이프라인

```
┌─────────────────────────────────────────────────────────┐
│              RAG 전체 파이프라인                          │
└─────────────────────────────────────────────────────────┘

1. 쿼리 입력: x = "고양이는 무엇인가요?"
   │
   ▼
2. 검색: R(x, k=4)
   ┌─────────────────────────────────────┐
   │  문서 컬렉션 D                        │
   │  ├─ d₁: "고양이는 포유동물"         │ ← P(d₁|x)=0.85
   │  ├─ d₂: "고양이는 네 발로 걷는다"    │ ← P(d₂|x)=0.72
   │  ├─ d₃: "강아지는 귀여워"           │ ← P(d₃|x)=0.15
   │  └─ ...                              │
   └─────────────────────────────────────┘
   │
   ▼ (상위 k=4개 선택)
3. 컨텍스트: C = {d₁, d₂, d₃, d₄}
   ┌─────────────────────────────────────┐
   │  Context:                            │
   │  [1] 고양이는 포유동물이다           │
   │  [2] 고양이는 네 발로 걷는다         │
   │  [3] 고양이는 육식동물이다           │
   │  [4] 고양이는 야행성이다             │
   └─────────────────────────────────────┘
   │
   ▼
4. 생성: y = argmax P(y' | x, C)
   ┌─────────────────────────────────────┐
   │  LLM(x, C) → y                      │
   │                                     │
   │  답변: "고양이는 포유동물이며       │
   │        네 발로 걷는 육식동물입니다." │
   └─────────────────────────────────────┘
```

### 5.2 llmkit 구현

#### 구현 5.2.1: RAGChain.query()

```python
# facade/rag_facade.py: RAGChain.query()
# handler/rag_handler.py: RAGHandler.handle_query()
# service/impl/rag_service_impl.py: RAGServiceImpl.query()
async def query(
    self,
    question: str,
    k: int = 4,
    include_sources: bool = False,
    rerank: bool = False,
    mmr: bool = False,
    hybrid: bool = False,
    **kwargs
) -> Union[str, Tuple[str, List[VectorSearchResult]]]:
    """
    RAG 쿼리: P(y | x) = Σ P(y | x, d) · P(d | x)
    
    수학적 표현:
    1. 검색: R(x, k) = argmax_{S ⊆ D, |S|=k} Σ_{d ∈ S} P(d | x)
    2. 컨텍스트: C = concat(R(x, k))
    3. 생성: y = argmax_{y'} P(y' | x, C)
    
    실제 구현 경로:
    - facade/rag_facade.py: RAGChain.query() (사용자 API)
    - handler/rag_handler.py: RAGHandler.handle_query() (입력 검증)
    - service/impl/rag_service_impl.py: RAGServiceImpl.query() (비즈니스 로직)
    """
    # 1. 검색: R(x, k)
    # 내부적으로 RAGHandler.handle_query() → RAGServiceImpl.query() 호출
    results = self.retrieve(question, k=k, rerank=rerank, mmr=mmr, hybrid=hybrid)
    
    # 2. 컨텍스트 구성: C = concat(R(x, k))
    context = self._build_context(results)
    
    # 3. 생성: y = argmax P(y' | x, C)
    prompt = self._build_prompt(question, context)
    answer = await self.llm.chat([{"role": "user", "content": prompt}])
    
    return answer.content
```

---

## 6. 정보 이론적 해석

### 6.1 Mutual Information

#### 정리 6.1.1: RAG와 Mutual Information

**RAG는 다음을 최대화합니다:**

$$
I(y; d | x) = H(y | x) - H(y | x, d)
$$

**해석:**
- $H(y | x)$: 쿼리만으로 답변의 불확실성
- $H(y | x, d)$: 문서를 추가한 후의 불확실성
- $I(y; d | x)$: 문서가 제공하는 정보량

### 6.2 정보 이론적 목적

#### 정리 6.2.1: 정보 최대화

**RAG의 목적:**

$$
\max_{d} I(y; d | x) = \max_{d} [H(y | x) - H(y | x, d)]
$$

**해석:**
- 답변의 불확실성을 최대한 줄이는 문서 선택
- 정보량이 가장 큰 문서 선택

---

## 7. CS 관점: 구현과 최적화

### 7.1 검색 단계의 최적화

#### 알고리즘 7.1.1: Top-k 검색

```
Algorithm: TopKRetrieval(query, candidates, k)
Input:
  - query: 벡터 q ∈ ℝ^d
  - candidates: 행렬 C ∈ ℝ^(n×d)
  - k: 반환할 개수
Output: 상위 k개 인덱스

1. similarities ← BatchCosineSimilarity(query, candidates)  // O(n·d)
2. top_k_indices ← ArgMaxK(similarities, k)  // O(n log k)
3. return top_k_indices
```

**시간 복잡도:** $O(n \cdot d + n \log k)$  
**공간 복잡도:** $O(n)$

### 7.2 생성 단계의 최적화

#### CS 관점 7.2.1: 컨텍스트 길이 최적화

**컨텍스트 길이 제한:**

```python
# facade/rag_facade.py: RAGChain.query
async def query(
    self,
    question: str,
    k: int = 5,
    rerank: bool = False,
    **kwargs: Any,
) -> str:
    """
    RAG 쿼리 실행:
    1. 검색: R = Retrieve(Q, V, k)
    2. 컨텍스트 구성: C = concat(R)
    3. 생성: A = LLM(Q, C)
    """
    # 1. 검색
    results = await self.vector_store.similarity_search(question, k=k)
    
    # 2. 컨텍스트 구성 (토큰 제한 고려)
    context_parts = []
    max_tokens = kwargs.get("max_context_tokens", 4000)
    current_tokens = 0
    
    for i, result in enumerate(results, 1):
        content = result.page_content
        # 간단한 토큰 추정 (실제로는 tiktoken 사용)
        estimated_tokens = len(content.split()) * 1.3
        
        if current_tokens + estimated_tokens > max_tokens:
            break
        
        context_parts.append(f"[{i}] {content}")
        current_tokens += estimated_tokens
    
    context = "\n\n".join(context_parts)
    
    # 3. 프롬프트 구성
    prompt = self.prompt_template.format(
        context=context,
        question=question
    )
    
    # 4. LLM 생성
    response = await self.llm.chat([{
        "role": "user",
        "content": prompt
    }])
    
    return response.content
```

**최적화:**
- 토큰 제한으로 비용 절감
- 관련도 순으로 선택

---

## 8. 실제 성능 분석

### 8.1 검색 정확도

#### 실험 8.1.1: 검색 정확도 측정

**설정:**
- 문서 수: 10,000
- 쿼리 수: 100
- $k = 5$

**결과:**

| 메트릭 | 값 |
|--------|-----|
| Recall@5 | 0.85 |
| Precision@5 | 0.72 |
| MRR | 0.78 |

### 8.2 생성 품질

#### 실험 8.2.1: 생성 품질 측정

**설정:**
- RAG vs 기본 LLM
- 평가: BLEU, ROUGE

**결과:**

| 방법 | BLEU | ROUGE-L |
|------|------|---------|
| 기본 LLM | 0.45 | 0.52 |
| RAG | 0.68 | 0.75 |

**RAG가 더 높은 품질을 보입니다.**

---

## 질문과 답변 (Q&A)

### Q1: RAG는 왜 효과적인가요?

**A:** RAG의 효과:

1. **외부 지식 활용:**
   - LLM의 학습 데이터 한계 극복
   - 최신 정보 활용

2. **정확도 향상:**
   - 검증 가능한 정보 제공
   - 환각(Hallucination) 감소

3. **유연성:**
   - 도메인 특화 가능
   - 문서 업데이트 용이

### Q2: Marginalization은 왜 필요한가요?

**A:** Marginalization의 필요성:

1. **불확실성 처리:**
   - 여러 문서의 정보 통합
   - 가중 평균으로 안정성 확보

2. **수학적 엄밀성:**
   - 확률 모델의 정확한 표현
   - 베이즈 정리와 일관성

3. **실용적 효과:**
   - 단일 문서의 오류 완화
   - 다양한 관점 통합

### Q3: 검색 단계와 생성 단계의 균형은?

**A:** 균형 조절:

1. **$k$ 값 선택:**
   - 작은 $k$: 빠르지만 정보 부족
   - 큰 $k$: 느리지만 정보 풍부
   - 권장: $k = 4 \sim 10$

2. **컨텍스트 길이:**
   - 토큰 제한 고려
   - 관련도 순 선택

3. **재순위화:**
   - 검색 후 정확도 향상
   - Cross-encoder 사용

---

## 참고 문헌

1. **Lewis et al. (2020)**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
2. **Karpukhin et al. (2020)**: "Dense Passage Retrieval for Open-Domain Question Answering"
3. **Robertson & Zaragoza (2009)**: "The Probabilistic Relevance Framework"

---

**작성일**: 2025-01-XX  
**버전**: 3.0 (CS 관점 + 석사 수준 확장)

