# RAG Theory: 검색 증강 생성의 수학적 모델과 알고리즘

**석사 수준 이론 문서**  
**기반**: llmkit RAGChain, VectorStore 실제 구현 분석

---

## 목차

### Part I: RAG의 확률 모델
1. [RAG의 수학적 정의](#part-i-rag의-확률-모델)
2. [조건부 확률과 베이즈 정리](#12-조건부-확률과-베이즈-정리)
3. [생성 모델의 수학적 표현](#13-생성-모델의-수학적-표현)

### Part II: 검색 알고리즘
4. [벡터 검색의 수학적 기초](#part-ii-검색-알고리즘)
5. [Approximate Nearest Neighbor (ANN)](#42-approximate-nearest-neighbor-ann)
6. [Hybrid Search: 정보 융합 이론](#43-hybrid-search-정보-융합-이론)

### Part III: 재순위화와 최적화
7. [Re-ranking: Cross-encoder 모델](#part-iii-재순위화와-최적화)
8. [MMR의 최적화 이론](#72-mmr의-최적화-이론)
9. [Query Expansion의 정보 이론](#73-query-expansion의-정보-이론)

### Part IV: RAG 파이프라인 분석
10. [전체 파이프라인의 수학적 모델](#part-iv-rag-파이프라인-분석)
11. [청킹 전략의 수학적 최적화](#102-청킹-전략의-수학적-최적화)
12. [컨텍스트 주입의 최적 길이](#103-컨텍스트-주입의-최적-길이)

---

## Part I: RAG의 확률 모델

### 1.1 RAG의 수학적 정의

#### 정의 1.1.1: Retrieval-Augmented Generation

**RAG**는 다음 확률 모델로 정의됩니다:

$$
P(y | x) = \sum_{d \in \mathcal{D}} P(y | x, d) \cdot P(d | x)
$$

여기서:
- $x$: 입력 쿼리
- $y$: 생성된 답변
- $d$: 검색된 문서
- $\mathcal{D}$: 문서 컬렉션

#### 시각적 표현: RAG 파이프라인

```
┌─────────────────────────────────────────────────────────┐
│                    RAG 파이프라인                        │
└─────────────────────────────────────────────────────────┘

1. 쿼리 입력: x = "고양이는 무엇인가요?"
   │
   ▼
2. 검색 단계: P(d | x)
   ┌─────────────────────────────────────┐
   │  문서 컬렉션 D = {d₁, d₂, ..., dₙ} │
   │                                     │
   │  d₁: "고양이는 포유동물이다"       │ ← P(d₁|x) = 0.85
   │  d₂: "고양이는 네 발로 걷는다"      │ ← P(d₂|x) = 0.72
   │  d₃: "강아지는 귀여워"             │ ← P(d₃|x) = 0.15
   └─────────────────────────────────────┘
   │
   ▼ (상위 k개 선택, k=2)
3. 컨텍스트 구성: C = {d₁, d₂}
   ┌─────────────────────────────────────┐
   │  Context:                           │
   │  [1] 고양이는 포유동물이다          │
   │  [2] 고양이는 네 발로 걷는다         │
   └─────────────────────────────────────┘
   │
   ▼
4. 생성 단계: P(y | x, d)
   ┌─────────────────────────────────────┐
   │  LLM(x, C) → y                      │
   │                                     │
   │  답변: "고양이는 포유동물이며       │
   │        네 발로 걷는 동물입니다."     │
   └─────────────────────────────────────┘
```

#### 해석 1.1.1: 확률 모델의 의미

1. **$P(d | x)$**: 쿼리 $x$에 대한 문서 $d$의 관련도 (검색 단계)
2. **$P(y | x, d)$**: 문서 $d$를 컨텍스트로 사용한 답변 생성 (생성 단계)

#### 구체적 수치 예시

**예시 1.1.1: RAG 확률 계산**

쿼리: $x$ = "고양이는 무엇인가요?"

**1단계: 검색 (P(d | x))**

문서 컬렉션:
- $d_1$: "고양이는 포유동물이다" → $P(d_1 | x) = 0.85$
- $d_2$: "고양이는 네 발로 걷는다" → $P(d_2 | x) = 0.72$
- $d_3$: "강아지는 귀여워" → $P(d_3 | x) = 0.15$

**2단계: 생성 (P(y | x, d))**

컨텍스트: $C = \{d_1, d_2\}$ (상위 2개)

$$
P(y | x, d_1, d_2) = \text{LLM}(x, C)
$$

**최종 답변 확률:**

$$
P(y | x) = P(y | x, d_1) \cdot P(d_1 | x) + P(y | x, d_2) \cdot P(d_2 | x)
$$

$$
= 0.95 \times 0.85 + 0.88 \times 0.72 = 0.8075 + 0.6336 = 1.4411
$$

(정규화 후 실제 확률)

**llmkit 구현:**
```python
# facade/rag_facade.py: RAGChain
# service/impl/rag_service_impl.py: RAGServiceImpl
# handler/rag_handler.py: RAGHandler
class RAGChain:
    """
    RAG 파이프라인: RAG(x) = LLM(x, Retrieve(x, D))
    
    수학적 표현:
    - P(d | x): 문서 검색 확률
    - P(y | x, d): LLM 생성 확률
    
    실제 구현:
    - facade/rag_facade.py: RAGChain (사용자 API)
    - service/impl/rag_service_impl.py: RAGServiceImpl (비즈니스 로직)
    - handler/rag_handler.py: RAGHandler (입력 검증)
    """
    def retrieve(self, query: str, k: int = 4) -> List[VectorSearchResult]:
        """
        P(d | x) 계산: 벡터 검색으로 관련 문서 찾기
        
        실제 구현:
        - facade/rag_facade.py: RAGChain.retrieve()
        - service/impl/rag_service_impl.py: RAGServiceImpl.retrieve()
        """
        results = self.vector_store.similarity_search(query, k=k)
        return results  # 상위 k개 문서

    def query(self, question: str, k: int = 4) -> str:
        """
        전체 RAG 파이프라인:
        1. P(d | x) 계산 (retrieve)
        2. P(y | x, d) 계산 (LLM 생성)
        
        실제 구현:
        - facade/rag_facade.py: RAGChain.query()
        - service/impl/rag_service_impl.py: RAGServiceImpl.query()
        """
        results = self.retrieve(question, k=k)
        context = self._build_context(results)
        prompt = self._build_prompt(question, context)
        answer = await self.llm.chat([{"role": "user", "content": prompt}])
        return answer.content
```

---

### 1.2 조건부 확률과 베이즈 정리

#### 정리 1.2.1: 베이즈 정리 적용

**베이즈 정리**에 의해:

$$
P(d | x) = \frac{P(x | d) \cdot P(d)}{P(x)} \propto P(x | d) \cdot P(d)
$$

**해석:**
- $P(x | d)$: 문서 $d$가 쿼리 $x$를 생성할 확률 (유사도)
- $P(d)$: 문서의 사전 확률 (일반적으로 균등 분포)

#### 정의 1.2.1: 유사도 기반 확률

**벡터 유사도를 확률로 변환:**

$$
P(d | x) = \frac{\exp(\text{sim}(E(x), E(d)) / \tau)}{\sum_{d' \in \mathcal{D}} \exp(\text{sim}(E(x), E(d')) / \tau)}
$$

여기서 $\tau$는 temperature parameter입니다.

**llmkit 구현:**
```python
# domain/embeddings/utils.py: cosine_similarity()
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    코사인 유사도 계산: cosine(u, v) = (u·v) / (||u|| ||v||)
    이후 softmax로 확률 변환 가능
    
    실제 구현:
    - domain/embeddings/utils.py: cosine_similarity()
    - NumPy 벡터화 연산 사용
    """
    a = np.array(vec1, dtype=np.float32)
    b = np.array(vec2, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

---

### 1.3 생성 모델의 수학적 표현

#### 정의 1.3.1: 조건부 생성 모델

**LLM의 조건부 생성:**

$$
P(y | x, d) = \prod_{i=1}^{|y|} P(y_i | y_{<i}, x, d)
$$

여기서 $y_i$는 $i$번째 토큰입니다.

#### 정리 1.3.1: 컨텍스트 주입

**컨텍스트가 포함된 프롬프트:**

$$
\text{prompt} = f(x, d_1, d_2, \ldots, d_k)
$$

**llmkit 구현:**
```python
# facade/rag_facade.py: RAGChain._build_context()
# service/impl/rag_service_impl.py: RAGServiceImpl._build_context()
def _build_context(self, results: List[VectorSearchResult]) -> str:
    """
    검색 결과에서 컨텍스트 생성
    
    수학적 표현: C = concat({d₁, d₂, ..., dₖ})
    
    실제 구현:
    - facade/rag_facade.py: RAGChain._build_context()
    - service/impl/rag_service_impl.py: RAGServiceImpl._build_context()
    """
    context_parts = []
    for i, result in enumerate(results, 1):
        context_parts.append(f"[{i}] {result.document.content}")
    return "\n\n".join(context_parts)

def _build_prompt(self, query: str, context: str) -> str:
    """
    프롬프트 생성: f(x, d₁, d₂, ..., dₖ)
    
    수학적 표현: prompt = f(x, C) where C = {d₁, d₂, ..., dₖ}
    
    실제 구현:
    - facade/rag_facade.py: RAGChain._build_prompt()
    - 기본 템플릿: "Based on the following context:\n{context}\n\nQuestion: {question}\nAnswer:"
    """
    return self.prompt_template.format(
        context=context,  # d₁, d₂, ..., dₖ
        question=query    # x
    )
```

---

## Part II: 검색 알고리즘

### 2.1 벡터 검색의 수학적 기초

#### 정의 2.1.1: k-Nearest Neighbor (k-NN)

**k-NN 문제:**

$$
\text{top-k}(q) = \arg\max_{S \subseteq \mathcal{D}, |S| = k} \sum_{d \in S} \text{sim}(q, d)
$$

**시간 복잡도:**
- Naive: $O(n \cdot d)$ (n: 문서 수, d: 차원)
- 최적화: $O(\log n \cdot d)$ (인덱싱 사용)

**llmkit 구현:**
```python
# service/types.py: VectorStoreProtocol
# domain/vector_stores/base.py: BaseVectorStore
# infrastructure/vector_stores/chroma.py: ChromaVectorStore
async def similarity_search(
    self,
    query: str,
    k: int = 4,
    **kwargs
) -> List[VectorSearchResult]:
    """
    k-NN 검색 구현: top-k(q) = argmax_{S ⊆ D, |S|=k} Σ_{d ∈ S} sim(q, d)
    
    수학적 표현:
    - 입력: 쿼리 q, 문서 컬렉션 D, k
    - 출력: 상위 k개 문서 S ⊆ D
    
    시간 복잡도:
    - Naive: O(n·d) (n: 문서 수, d: 차원)
    - 최적화 (HNSW): O(log n·d)
    
    실제 구현:
    - service/types.py: VectorStoreProtocol (인터페이스)
    - domain/vector_stores/base.py: BaseVectorStore (추상 클래스)
    - infrastructure/vector_stores/chroma.py: ChromaVectorStore (Chroma 구현)
    - infrastructure/vector_stores/faiss.py: FAISSVectorStore (FAISS 구현)
    - 각 provider가 최적화된 인덱스 사용 (HNSW, IVF, 등)
    """
    # 1. 쿼리 임베딩 생성
    query_vec = await self.embedding_function([query])
    query_vec = query_vec[0] if isinstance(query_vec, list) else query_vec
    
    # 2. Provider별 최적화된 검색 알고리즘 사용
    # - Chroma: 자체 최적화 인덱스
    # - FAISS: HNSW 또는 IVF 인덱스
    # - Pinecone: 관리형 서비스
    results = await self._search_vectors(query_vec, k=k, **kwargs)
    
    return results
```

---

### 2.2 Approximate Nearest Neighbor (ANN)

#### 정의 2.2.1: ANN 문제

**정확한 k-NN 대신 근사 해를 찾습니다:**

$$
\text{ANN}(q, k, \epsilon) = \{d | \text{sim}(q, d) \geq (1-\epsilon) \cdot \text{sim}(q, d^*)\}
$$

여기서 $d^*$는 정확한 최근접 이웃입니다.

#### 알고리즘 2.2.1: HNSW (Hierarchical Navigable Small World)

**HNSW**는 그래프 기반 ANN 알고리즘입니다:

1. **다층 그래프 구조**: $L_0, L_1, \ldots, L_m$
2. **탐색**: 상위 레이어에서 시작하여 하위 레이어로 이동
3. **시간 복잡도**: $O(\log n)$

**llmkit에서의 사용:**
- FAISS: HNSW 인덱스 지원
- Chroma: 자체 최적화 인덱스

---

### 2.3 Hybrid Search: 정보 융합 이론

#### 정의 2.3.1: Hybrid Search

**Hybrid Search**는 벡터 검색과 키워드 검색을 결합합니다:

$$
\text{score}(q, d) = \alpha \cdot s_{\text{vector}}(q, d) + (1-\alpha) \cdot s_{\text{keyword}}(q, d)
$$

#### 시각적 표현: Hybrid Search 구조

```
┌─────────────────────────────────────────────────────────┐
│                  Hybrid Search 구조                      │
└─────────────────────────────────────────────────────────┘

쿼리: "machine learning"
│
├─→ 벡터 검색 (Dense)          ├─→ 키워드 검색 (Sparse)
│                              │
│  E(q) = [0.12, 0.45, ...]    │  q = {"machine", "learning"}
│         │                     │         │
│         ▼                     │         ▼
│  유사도 계산                  │  BM25 점수 계산
│         │                     │         │
│         ▼                     │         ▼
│  결과:                        │  결과:
│  1. d₁ (sim=0.92)            │  1. d₃ (score=8.5)
│  2. d₂ (sim=0.85)            │  2. d₁ (score=7.2)
│  3. d₃ (sim=0.78)            │  3. d₄ (score=6.8)
│                              │
└──────────┬───────────────────┴──────────┬──────────────
           │                               │
           ▼                               ▼
    ┌─────────────────────────────────────────────┐
    │      RRF 점수 결합 (α = 0.7)                │
    │                                             │
    │  d₁: 0.7×0.92 + 0.3×7.2 = 0.644 + 2.16     │
    │      = 2.804 ← 1위                         │
    │                                             │
    │  d₂: 0.7×0.85 + 0.3×0 = 0.595 ← 3위       │
    │                                             │
    │  d₃: 0.7×0.78 + 0.3×8.5 = 0.546 + 2.55     │
    │      = 3.096 ← 2위                         │
    │                                             │
    │  d₄: 0.7×0 + 0.3×6.8 = 2.04 ← 4위          │
    └─────────────────────────────────────────────┘
           │
           ▼
    최종 결과: [d₁, d₃, d₂, d₄]
```

#### 구체적 수치 예시

**예시 2.3.1: Hybrid Search 계산**

쿼리: "machine learning"
$\alpha = 0.7$ (벡터 검색 가중치)

**벡터 검색 결과:**
- $d_1$: sim = 0.92
- $d_2$: sim = 0.85
- $d_3$: sim = 0.78

**키워드 검색 결과 (BM25):**
- $d_3$: score = 8.5
- $d_1$: score = 7.2
- $d_4$: score = 6.8

**RRF 점수 계산 ($k = 60$):**

1. **$d_1$:**
   - 벡터 순위: 1 → RRF = $0.7 \times \frac{1}{60+1} = 0.7 \times 0.0164 = 0.0115$
   - 키워드 순위: 2 → RRF = $0.3 \times \frac{1}{60+2} = 0.3 \times 0.0161 = 0.0048$
   - **총점**: $0.0115 + 0.0048 = 0.0163$

2. **$d_3$:**
   - 벡터 순위: 3 → RRF = $0.7 \times \frac{1}{60+3} = 0.7 \times 0.0159 = 0.0111$
   - 키워드 순위: 1 → RRF = $0.3 \times \frac{1}{60+1} = 0.3 \times 0.0164 = 0.0049$
   - **총점**: $0.0111 + 0.0049 = 0.0160$

**최종 순위:** $d_1$ (0.0163) > $d_3$ (0.0160) > $d_2$ > $d_4$

#### 알고리즘 2.3.1: Reciprocal Rank Fusion (RRF)

**RRF**는 순위를 점수로 변환합니다:

$$
\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}
$$

여기서 $R$은 검색 방법들의 순위 리스트, $k$는 상수입니다.

#### RRF 점수 변환 테이블

```
순위 → RRF 점수 (k=60)

순위  │ RRF 점수    │ 해석
──────┼────────────┼──────────────────
  1   │ 0.0164     │ 최고 점수
  2   │ 0.0161     │
  3   │ 0.0159     │
  5   │ 0.0154     │
 10   │ 0.0143     │
 20   │ 0.0125     │
 50   │ 0.0091     │
100   │ 0.0063     │ 낮은 점수
```

**llmkit 구현:**
```python
# domain/vector_stores/search.py: SearchAlgorithms._combine_results()
def _combine_results(
    vector_results: List[VectorSearchResult],
    keyword_results: List[VectorSearchResult],
    alpha: float = 0.5
) -> List[VectorSearchResult]:
    """
    RRF로 결과 결합: RRF(d) = α · (1/(k + r_vec)) + (1-α) · (1/(k + r_key))
    
    수학적 표현:
    score = α · (1/(k + r_vec)) + (1-α) · (1/(k + r_key))
    
    실제 구현:
    - domain/vector_stores/search.py: SearchAlgorithms._combine_results()
    - vector_stores/search.py: SearchAlgorithms._combine_results() (레거시)
    """
    k_constant = 60  # RRF constant
    vec_score = alpha / (k_constant + vec_rank) if vec_rank else 0
    key_score = (1 - alpha) / (k_constant + key_rank) if key_rank else 0
    total_score = vec_score + key_score
```

#### 정리 2.3.1: RRF의 성질

**RRF는 다음을 만족합니다:**

1. **단조성**: 순위가 높을수록 점수 증가
2. **경계**: 점수는 $[0, 1/k]$ 범위
3. **결합성**: 여러 검색 방법을 자연스럽게 결합

---

## Part III: 재순위화와 최적화

### 3.1 Re-ranking: Cross-encoder 모델

#### 정의 3.1.1: Bi-encoder vs Cross-encoder

**Bi-encoder:**
$$
\text{sim}(q, d) = \cos(E_{\text{query}}(q), E_{\text{doc}}(d))
$$

**Cross-encoder:**
$$
\text{score}(q, d) = f_{\text{cross}}([E(q); E(d)])
$$

여기서 $[E(q); E(d)]$는 concatenation입니다.

#### 정리 3.1.1: Cross-encoder의 정확도

**Cross-encoder는 Bi-encoder보다 정확하지만 느립니다:**

- **Bi-encoder**: $O(d)$ 시간 (벡터만 계산)
- **Cross-encoder**: $O(d^2)$ 시간 (쿼리-문서 쌍 계산)

**llmkit 구현:**
```python
# domain/vector_stores/search.py: SearchAlgorithms.rerank()
# facade/rag_facade.py: RAGChain.rerank()
def rerank(
    query: str,
    results: List[VectorSearchResult],
    top_k: int = 5
) -> List[VectorSearchResult]:
    """
    Cross-encoder로 재순위화: Rerank(R, q) = arg sort_{d ∈ R} Score_cross(q, d)
    
    Note: sentence-transformers의 CrossEncoder 사용
    
    실제 구현:
    - domain/vector_stores/search.py: SearchAlgorithms.rerank()
    - facade/rag_facade.py: RAGChain.rerank() (사용자 API)
    """
    # 1차: Bi-encoder로 후보 선정 (빠름)
    candidates = results  # 이미 검색됨
    
    # 2차: Cross-encoder로 재순위화 (느리지만 정확)
    # CrossEncoder([query; doc]) 계산
```

---

### 3.2 MMR의 최적화 이론

#### 정의 3.2.1: MMR 최적화 문제

**MMR은 다음 최적화 문제를 해결합니다:**

$$
\max_{S, |S|=k} \left[ \lambda \sum_{d \in S} \text{sim}(q, d) - (1-\lambda) \sum_{d_i, d_j \in S, i \neq j} \text{sim}(d_i, d_j) \right]
$$

#### 시각적 표현: MMR 알고리즘

```
┌─────────────────────────────────────────────────────────┐
│              MMR (Maximal Marginal Relevance)            │
└─────────────────────────────────────────────────────────┘

쿼리: "고양이"
λ = 0.6 (관련성 가중치)

후보 문서들:
d₁: "고양이 사료"        (sim(q,d₁) = 0.92)
d₂: "고양이 사료 추천"   (sim(q,d₂) = 0.90)
d₃: "고양이 사료 종류"   (sim(q,d₃) = 0.88)
d₄: "고양이 건강"        (sim(q,d₄) = 0.75)
d₅: "고양이 행동"        (sim(q,d₅) = 0.70)

───────────────────────────────────────────────────────────

Step 1: 첫 번째 선택
  S = {d₁}  (가장 관련성 높음, sim=0.92)

Step 2: 두 번째 선택
  후보: d₂, d₃, d₄, d₅
  
  d₂: relevance = 0.6 × 0.90 = 0.54
      diversity = 0.4 × sim(d₂,d₁) = 0.4 × 0.95 = 0.38
      MMR = 0.54 - 0.38 = 0.16
  
  d₃: relevance = 0.6 × 0.88 = 0.528
      diversity = 0.4 × sim(d₃,d₁) = 0.4 × 0.93 = 0.372
      MMR = 0.528 - 0.372 = 0.156
  
  d₄: relevance = 0.6 × 0.75 = 0.45
      diversity = 0.4 × sim(d₄,d₁) = 0.4 × 0.65 = 0.26
      MMR = 0.45 - 0.26 = 0.19  ← 최대!
  
  d₅: relevance = 0.6 × 0.70 = 0.42
      diversity = 0.4 × sim(d₅,d₁) = 0.4 × 0.60 = 0.24
      MMR = 0.42 - 0.24 = 0.18
  
  → S = {d₁, d₄}

Step 3: 세 번째 선택
  후보: d₂, d₃, d₅
  
  d₂: relevance = 0.54
      diversity = 0.4 × max(sim(d₂,d₁), sim(d₂,d₄))
                = 0.4 × max(0.95, 0.68) = 0.4 × 0.95 = 0.38
      MMR = 0.54 - 0.38 = 0.16
  
  d₅: relevance = 0.42
      diversity = 0.4 × max(sim(d₅,d₁), sim(d₅,d₄))
                = 0.4 × max(0.60, 0.55) = 0.4 × 0.60 = 0.24
      MMR = 0.42 - 0.24 = 0.18  ← 최대!
  
  → S = {d₁, d₄, d₅}

최종 결과: [d₁ (사료), d₄ (건강), d₅ (행동)]
→ 다양성 확보! (모두 "사료" 관련이 아님)
```

#### 구체적 수치 예시

**예시 3.2.1: MMR 계산**

쿼리: "고양이"
$\lambda = 0.6$, $k = 3$

**후보 문서:**
- $d_1$: "고양이 사료" (sim(q,d₁) = 0.92)
- $d_2$: "고양이 사료 추천" (sim(q,d₂) = 0.90, sim(d₂,d₁) = 0.95)
- $d_3$: "고양이 건강" (sim(q,d₃) = 0.75, sim(d₃,d₁) = 0.65)

**Step 1:** $S = \{d_1\}$ (가장 관련성 높음)

**Step 2:** 두 번째 선택

$d_2$의 MMR:
$$
\text{MMR}(d_2) = 0.6 \times 0.90 - 0.4 \times 0.95 = 0.54 - 0.38 = 0.16
$$

$d_3$의 MMR:
$$
\text{MMR}(d_3) = 0.6 \times 0.75 - 0.4 \times 0.65 = 0.45 - 0.26 = 0.19
$$

→ $d_3$ 선택 (MMR이 더 높음)

**Step 3:** 세 번째 선택
$S = \{d_1, d_3\}$

$d_2$의 MMR:
$$
\text{MMR}(d_2) = 0.6 \times 0.90 - 0.4 \times \max(0.95, 0.68) = 0.54 - 0.38 = 0.16
$$

→ $d_2$ 선택

**최종 결과:** $\{d_1, d_3, d_2\}$ (다양성 확보)

#### 알고리즘 3.2.1: Greedy MMR

**Greedy 알고리즘:**

```
1. S = {argmax_d sim(q, d)}  // 가장 관련성 높은 것
2. while |S| < k:
     d* = argmax_{d ∉ S} [λ·sim(q,d) - (1-λ)·max_{d'∈S} sim(d,d')]
     S = S ∪ {d*}
3. return S
```

**근사 비율:** Greedy는 최적해의 $(1-1/e) \approx 0.632$ 배를 보장합니다.

#### MMR vs 일반 검색 비교

```
일반 검색 (관련성만):
┌─────────────────────────────────────┐
│ 결과: [d₁, d₂, d₃]                 │
│ 모두 "사료" 관련 (다양성 없음)      │
└─────────────────────────────────────┘

MMR 검색 (관련성 + 다양성):
┌─────────────────────────────────────┐
│ 결과: [d₁, d₄, d₅]                 │
│ - d₁: 사료                          │
│ - d₄: 건강 (다양함!)                │
│ - d₅: 행동 (다양함!)                │
└─────────────────────────────────────┘
```

**llmkit 구현:**
```python
# domain/vector_stores/search.py: SearchAlgorithms.mmr_search()
def mmr_search(
    query_vec: List[float],
    candidate_vecs: List[List[float]],
    k: int = 5,
    lambda_param: float = 0.6,
) -> List[int]:
    """
    Greedy MMR 알고리즘: argmax_i [λ·sim(q, c_i) - (1-λ)·max_j∈S sim(c_i, c_j)]
    
    수학적 표현:
    mmr_score = λ × relevance - (1-λ) × diversity
    
    실제 구현:
    - domain/vector_stores/search.py: SearchAlgorithms.mmr_search()
    - vector_stores/search.py: SearchAlgorithms.mmr_search() (레거시)
    """
    # 첫 번째: 가장 관련성 높은 것
    selected = [query_similarities.index(max(query_similarities))]
    
    # 나머지 k-1개: Greedy 선택
    for _ in range(k - 1):
        best_score = float("-inf")
        for idx in remaining:
            relevance = query_similarities[idx]
            diversity = max(candidate_sims) if candidate_sims else 0.0
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
```

---

### 3.3 Query Expansion의 정보 이론

#### 정의 3.3.1: 정보 이론적 관점

**Query Expansion은 Mutual Information을 최대화합니다:**

$$
I(Q_{\text{exp}}; D) = H(D) - H(D | Q_{\text{exp}})
$$

**목표:** 확장된 쿼리 $Q_{\text{exp}}$가 문서 $D$에 대한 불확실성을 최소화

#### 정리 3.3.1: 확장 쿼리의 효과

**확장 쿼리는 리콜을 향상시킵니다:**

$$
\text{Recall}(Q_{\text{exp}}) \geq \text{Recall}(Q)
$$

**증명 스케치:**
확장된 쿼리는 더 많은 관련 문서를 포함하므로 리콜이 증가합니다.

**llmkit 구현:**
```python
# domain/embeddings/utils.py (또는 직접 구현)
def query_expansion(
    query: str,
    embedding: BaseEmbedding,
    expansion_candidates: Optional[List[str]] = None,
    similarity_threshold: float = 0.7,
) -> List[str]:
    """
    Query Expansion: Q_exp = {w | I(Q; w) > τ}
    
    정보 이론적 관점:
    Q_exp = {w | I(Q; w) > τ}
    
    실제 구현:
    - domain/embeddings/utils.py (또는 직접 구현)
    - batch_cosine_similarity() 사용
    """
    query_vec = embedding.embed_sync([query])[0]
    candidate_vecs = embedding.embed_sync(expansion_candidates)
    similarities = batch_cosine_similarity(query_vec, candidate_vecs)
    
    # Mutual Information이 임계값 이상인 단어만 추가
    for candidate, sim in candidate_with_sim:
        if sim >= similarity_threshold:  # I(Q; w) > τ
            expanded.append(candidate)
```

---

## Part IV: RAG 파이프라인 분석

### 4.1 전체 파이프라인의 수학적 모델

#### 정의 4.1.1: RAG 파이프라인

**전체 파이프라인:**

$$
\text{RAG}(x) = \text{LLM}(x, \text{Retrieve}(x, \mathcal{D}))
$$

**단계별 분해:**

1. **문서 수집**: $\mathcal{D} = \{d_1, d_2, \ldots, d_n\}$
2. **청킹**: $C = \text{Chunk}(\mathcal{D}) = \{c_1, c_2, \ldots, c_m\}$
3. **임베딩**: $E = \text{Embed}(C) = \{E(c_1), E(c_2), \ldots, E(c_m)\}$
4. **저장**: $V = \text{Store}(E)$
5. **검색**: $R = \text{Retrieve}(x, V, k) = \{r_1, r_2, \ldots, r_k\}$
6. **생성**: $y = \text{LLM}(x, R)$

#### 시각적 표현: 전체 파이프라인

```
┌─────────────────────────────────────────────────────────────┐
│                  RAG 파이프라인 전체 흐름                     │
└─────────────────────────────────────────────────────────────┘

1. 문서 수집
   ┌─────────────────────────────────────┐
   │  원본 문서들                        │
   │  - doc1.pdf (10,000자)             │
   │  - doc2.txt (5,000자)              │
   │  - doc3.md (8,000자)               │
   └─────────────────────────────────────┘
   │
   ▼
2. 청킹 (Chunking)
   ┌─────────────────────────────────────┐
   │  chunk_size=500, overlap=50        │
   │                                     │
   │  c₁: "AI는 인공지능..." (500자)    │
   │  c₂: "...지능이다. 머신러닝..." (500자)│
   │  c₃: "...러닝은 데이터..." (500자)  │
   │  ...                                │
   │  cₘ: "...최종 청크" (300자)        │
   │                                     │
   │  총 m = 47개 청크                  │
   └─────────────────────────────────────┘
   │
   ▼
3. 임베딩 (Embedding)
   ┌─────────────────────────────────────┐
   │  각 청크를 벡터로 변환               │
   │                                     │
   │  E(c₁) = [0.12, 0.45, ..., 0.78]   │ (1536차원)
   │  E(c₂) = [0.23, 0.56, ..., 0.89]   │
   │  ...                                │
   │  E(cₘ) = [0.34, 0.67, ..., 0.90]   │
   └─────────────────────────────────────┘
   │
   ▼
4. 벡터 저장 (Vector Store)
   ┌─────────────────────────────────────┐
   │  Vector Database                    │
   │                                     │
   │  ID │ Content      │ Embedding      │
   │  ───┼──────────────┼─────────────── │
   │  1  │ c₁ (500자)   │ E(c₁) [1536]   │
   │  2  │ c₂ (500자)   │ E(c₂) [1536]   │
   │  ...│ ...          │ ...            │
   │  47 │ cₘ (300자)   │ E(cₘ) [1536]   │
   └─────────────────────────────────────┘
   │
   ▼
5. 검색 (Retrieval) - 쿼리: "AI란 무엇인가?"
   ┌─────────────────────────────────────┐
   │  Query Embedding: E(x)              │
   │  [0.15, 0.48, ..., 0.82]           │
   │                                     │
   │  유사도 계산:                       │
   │  sim(E(x), E(c₁)) = 0.92  ← 1위    │
   │  sim(E(x), E(c₂)) = 0.85  ← 2위    │
   │  sim(E(x), E(c₃)) = 0.78  ← 3위    │
   │  ...                                │
   │                                     │
   │  상위 k=3개 선택                    │
   └─────────────────────────────────────┘
   │
   ▼
6. 생성 (Generation)
   ┌─────────────────────────────────────┐
   │  LLM 입력:                          │
   │                                     │
   │  Context:                           │
   │  [1] c₁: "AI는 인공지능..."        │
   │  [2] c₂: "...지능이다. 머신..."    │
   │  [3] c₃: "...머신러닝은..."        │
   │                                     │
   │  Question: "AI란 무엇인가?"         │
   │                                     │
   │  ────────────────────────────────  │
   │                                     │
   │  LLM 출력:                          │
   │  "AI는 인공지능으로, 머신러닝을     │
   │   통해 데이터로부터 학습하는        │
   │   시스템입니다."                    │
   └─────────────────────────────────────┘
```

#### 구체적 수치 예시

**예시 4.1.1: 실제 파이프라인 실행**

**입력:**
- 원본 문서: 10,000자 PDF 파일
- 쿼리: "머신러닝이란 무엇인가?"

**1단계: 청킹**
```
chunk_size = 500
chunk_overlap = 50

원본: 10,000자
→ 청크 수: m = ⌈(10000 - 50) / (500 - 50)⌉ = ⌈9950 / 450⌉ = 23개
```

**2단계: 임베딩**
```
각 청크 → 1536차원 벡터
총 벡터 수: 23개
총 차원: 23 × 1536 = 35,328개 실수
```

**3단계: 검색**
```
쿼리 임베딩: E(x) [1536차원]

유사도 계산 (23개 청크):
- c₁: 0.92
- c₅: 0.88
- c₁₂: 0.85
- ... (나머지 < 0.80)

상위 k=3 선택: {c₁, c₅, c₁₂}
```

**4단계: 생성**
```
컨텍스트 길이: 500 + 500 + 500 = 1,500자
LLM 토큰: ~400 tokens
생성 시간: ~2초
```

**llmkit 구현:**
```python
# facade/rag_facade.py: RAGBuilder.from_documents()
# service/impl/rag_service_impl.py: RAGServiceImpl.build_chain()
@classmethod
def from_documents(
    cls,
    source: Union[str, Path, List[Document]],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedding_model: str = "text-embedding-3-small",
    **kwargs
) -> 'RAGChain':
    """
    전체 파이프라인 구현:
    1. 문서 로딩: D = load(source)
    2. 청킹: C = split(D, chunk_size, overlap)
    3. 임베딩: E = embed(C, model)
    4. 저장: V = store(E)
    5. RAGChain 생성
    
    실제 구현:
    - facade/rag_facade.py: RAGBuilder.from_documents()
    - service/impl/rag_service_impl.py: RAGServiceImpl.build_chain()
    """
    from ...domain.loaders import DocumentLoader
    from ...domain.splitters import TextSplitter
    from ...domain.embeddings import Embedding
    
    documents = DocumentLoader.load(source)  # 1
    chunks = TextSplitter.split(documents, chunk_size, chunk_overlap)  # 2
    embedding = Embedding(model=embedding_model)  # 3
    embed_func = embedding.embed_sync
    vector_store = from_documents(chunks, embed_func)  # 4
    return cls(vector_store=vector_store, **kwargs)  # 5
```

---

### 4.2 청킹 전략의 수학적 최적화

#### 정의 4.2.1: 최적 청크 크기

**청크 크기 최적화 문제:**

$$
\min_{s, o} \left[ \text{Loss}(s, o) + \lambda \cdot \text{Cost}(s, o) \right]
$$

여기서:
- $s$: 청크 크기 (chunk_size)
- $o$: 겹침 크기 (overlap)
- $\text{Loss}$: 정보 손실
- $\text{Cost}$: 계산 비용

#### 정리 4.2.1: 정보 손실 모델

**정보 손실은 청크 경계에서 발생:**

$$
\text{Loss}(s, o) = \sum_{i=1}^{n/s} I(c_i; c_{i+1}) \cdot (1 - \text{overlap\_ratio})
$$

**llmkit 구현:**
```python
# domain/splitters/base.py: BaseTextSplitter
class BaseTextSplitter:
    """
    텍스트 분할 베이스 클래스
    
    실제 구현:
    - domain/splitters/base.py: BaseTextSplitter (추상 클래스)
    - domain/splitters/splitters.py: CharacterTextSplitter, RecursiveCharacterTextSplitter
    """
    def __init__(
        self,
        chunk_size: int = 1000,      # s
        chunk_overlap: int = 200,    # o
        length_function: Callable[[str], int] = len,
    ):
        """
        최적 파라미터:
        - chunk_size: 모델 컨텍스트 길이의 50-75%
        - overlap: chunk_size의 10-20%
        
        실제 구현:
        - domain/splitters/base.py: BaseTextSplitter.__init__()
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
```

---

### 4.3 컨텍스트 주입의 최적 길이

#### 정의 4.3.1: 컨텍스트 길이 제한

**LLM의 컨텍스트 윈도우 제약:**

$$
|C| \leq L_{\max}
$$

여기서 $L_{\max}$는 최대 컨텍스트 길이입니다.

#### 정리 4.3.1: 최적 컨텍스트 크기

**최적 컨텍스트 크기는 다음을 최대화합니다:**

$$
\max_k \left[ \text{Relevance}(k) - \lambda \cdot \text{Noise}(k) \right]
$$

**실험적 결과:**
- $k = 3-5$: 최적 성능
- $k > 10$: 노이즈 증가로 성능 저하

**llmkit 구현:**
```python
# facade/rag_facade.py: RAGChain.query()
# service/impl/rag_service_impl.py: RAGServiceImpl.query()
def query(
    self,
    question: str,
    k: int = 4,  # 기본값: 최적 성능 범위
    **kwargs
) -> str:
    """
    RAG 쿼리: RAG(x) = LLM(x, Retrieve(x, D))
    
    k=4가 실험적으로 최적 성능
    
    실제 구현:
    - facade/rag_facade.py: RAGChain.query()
    - service/impl/rag_service_impl.py: RAGServiceImpl.query()
    """
    results = self.retrieve(question, k=k)
```

---

## 참고 문헌

1. **Lewis et al. (2020)**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
2. **Khattab & Zaharia (2020)**: "ColBERT: Efficient and Effective Passage Search"
3. **Cormack et al. (2009)**: "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"

---

**작성일**: 2025-01-XX  
**버전**: 2.0 (석사 수준 확장)
