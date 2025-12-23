# Vector Search and ANN: 벡터 검색과 근사 최근접 이웃

**석사 수준 이론 문서**  
**CS 관점 + 수학적 엄밀성**  
**기반**: llmkit VectorStore 실제 구현 분석

---

## 목차

1. [벡터 검색의 수학적 기초](#1-벡터-검색의-수학적-기초)
2. [Exact Nearest Neighbor Search](#2-exact-nearest-neighbor-search)
3. [Approximate Nearest Neighbor (ANN)](#3-approximate-nearest-neighbor-ann)
4. [ANN 알고리즘 비교](#4-ann-알고리즘-비교)
5. [FAISS와 HNSW](#5-faiss와-hnsw)
6. [CS 관점: 인덱싱과 검색 복잡도](#6-cs-관점-인덱싱과-검색-복잡도)
7. [실제 구현과 성능](#7-실제-구현과-성능)

---

## 1. 벡터 검색의 수학적 기초

### 1.1 벡터 검색 문제 정의

#### 정의 1.1.1: Nearest Neighbor Search (NNS)

**Nearest Neighbor Search**는 다음을 해결합니다:

$$
\text{NN}(q, \mathcal{D}) = \arg\min_{d \in \mathcal{D}} d(q, d)
$$

여기서:
- $q$: 쿼리 벡터
- $\mathcal{D}$: 문서 벡터 집합
- $d(q, d)$: 거리 함수 (유클리드, 코사인 등)

#### 정의 1.1.2: k-Nearest Neighbors (k-NN)

**k-Nearest Neighbors**는 상위 $k$개를 반환합니다:

$$
\text{kNN}(q, \mathcal{D}, k) = \{d_1, d_2, \ldots, d_k | d(q, d_1) \leq d(q, d_2) \leq \cdots \leq d(q, d_k) \leq d(q, d_j) \forall j > k\}
$$

#### 시각적 표현: 벡터 검색

```
벡터 공간 ℝ^d:

        q (쿼리) ★
         │
         │ d(q, d₁)
         │
         ▼
        d₁ ★
         │
         │ d(q, d₂)
         │
         ▼
        d₂ ★
         │
         │ d(q, d₃)
         │
         ▼
        d₃ ★

k=3 검색: {d₁, d₂, d₃} (가장 가까운 3개)
```

---

## 2. Exact Nearest Neighbor Search

### 2.1 Linear Search

#### 알고리즘 2.1.1: Linear Search

```
Algorithm: LinearSearch(query, candidates, k)
Input:
  - query: 벡터 q ∈ ℝ^d
  - candidates: 행렬 C ∈ ℝ^(n×d)
  - k: 반환할 개수
Output: 상위 k개 인덱스

1. distances ← []
2. for i = 1 to n:
3.     dist ← distance(query, candidates[i])
4.     distances.append((i, dist))
5. Sort distances by dist (ascending)
6. return [idx for (idx, _) in distances[:k]]
```

**시간 복잡도:** $O(n \cdot d + n \log n)$  
**공간 복잡도:** $O(n)$

**llmkit 구현:**
```python
# domain/vector_stores/base.py: BaseVectorStore
# infrastructure/vector_stores/chroma.py: ChromaVectorStore
# infrastructure/vector_stores/faiss.py: FAISSVectorStore
from abc import ABC, abstractmethod

class BaseVectorStore(ABC):
    """
    벡터 스토어 베이스 클래스
    
    수학적 정의:
    - k-NN: top-k(q) = argmax_{S ⊆ D, |S|=k} Σ_{d ∈ S} sim(q, d)
    - 시간 복잡도: O(n·d) (naive), O(log n·d) (HNSW)
    
    실제 구현:
    - domain/vector_stores/base.py: BaseVectorStore (추상 클래스)
    - infrastructure/vector_stores/chroma.py: ChromaVectorStore (Chroma 사용)
    - infrastructure/vector_stores/faiss.py: FAISSVectorStore (FAISS 사용)
    """
    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[VectorSearchResult]:
        """
        k-NN 검색: top-k(q) = argmax_{S ⊆ D, |S|=k} Σ_{d ∈ S} sim(q, d)
        
        Process:
        1. 쿼리 임베딩 생성: q_vec = embed(query)
        2. 거리 계산: distances = [sim(q_vec, d_vec) for d_vec in D]
        3. 정렬 및 상위 k개 선택
        
        시간 복잡도:
        - Naive: O(n·d + n log n)
        - HNSW (FAISS): O(log n·d)
        - Chroma: 자체 최적화 인덱스
        
        실제 구현:
        - domain/vector_stores/base.py: BaseVectorStore.similarity_search() (추상)
        - infrastructure/vector_stores/chroma.py: ChromaVectorStore.similarity_search()
        - infrastructure/vector_stores/faiss.py: FAISSVectorStore.similarity_search()
        """
        # 1. 쿼리 임베딩
        query_vec = await self.embedding_function([query])
        query_vec = query_vec[0] if isinstance(query_vec, list) else query_vec
        
        # 2. Provider별 최적화된 검색
        # - Chroma: 자체 인덱스 (HNSW 유사)
        # - FAISS: HNSW 또는 IVF 인덱스
        # - Pinecone: 관리형 서비스
        results = await self._search_vectors(query_vec, k=k, **kwargs)
        
        return results
```

**HNSW 구현 (FAISS):**
```python
# infrastructure/vector_stores/faiss.py: FAISSVectorStore
import faiss

class FAISSVectorStore(BaseVectorStore):
    """
    FAISS 벡터 스토어: HNSW 인덱스 사용
    
    HNSW 알고리즘:
    - 다층 그래프: L₀, L₁, ..., Lₘ
    - 탐색: 상위 레이어 → 하위 레이어
    - 시간 복잡도: O(log n·d)
    
    실제 구현:
    - infrastructure/vector_stores/faiss.py: FAISSVectorStore
    - FAISS HNSW 인덱스 사용
    """
    def __init__(self, dimension: int, index_type: str = "HNSW", **kwargs):
        """
        Args:
            dimension: 벡터 차원 d
            index_type: 인덱스 타입 ('HNSW', 'IVF', 'Flat')
        """
        self.dimension = dimension
        
        if index_type == "HNSW":
            # HNSW 인덱스 생성
            # M: 각 레이어의 최대 연결 수 (기본값: 32)
            # ef_construction: 인덱싱 시 탐색 범위 (기본값: 200)
            self.index = faiss.IndexHNSWFlat(dimension, M=32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 50  # 검색 시 탐색 범위
        elif index_type == "IVF":
            # IVF 인덱스 (Inverted File Index)
            nlist = kwargs.get("nlist", 100)  # 클러스터 수
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        else:
            # Flat 인덱스 (Linear Search)
            self.index = faiss.IndexFlatL2(dimension)
    
    def add_vectors(self, vectors: np.ndarray):
        """
        벡터 추가 및 인덱싱
        
        시간 복잡도:
        - HNSW: O(n log n·d)
        - IVF: O(n·k·d) where k = 클러스터 수
        - Flat: O(1) (인덱싱 없음)
        """
        vectors = np.array(vectors, dtype=np.float32)
        
        if isinstance(self.index, faiss.IndexIVFFlat):
            # IVF는 학습 필요
            if not self.index.is_trained:
                self.index.train(vectors)
        
        self.index.add(vectors)
    
    def search(self, query_vec: np.ndarray, k: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        HNSW 검색: O(log n·d)
        
        Returns:
            (distances, indices): 거리와 인덱스
        """
        query_vec = np.array([query_vec], dtype=np.float32)
        distances, indices = self.index.search(query_vec, k)
        return distances[0], indices[0]
```

### 2.2 공간 분할 기법

#### 정의 2.2.1: KD-Tree

**KD-Tree**는 $k$차원 공간을 재귀적으로 분할합니다:

**구성:**
- 각 노드는 한 차원에서 분할
- 왼쪽 자식: 해당 차원 값 < 분할값
- 오른쪽 자식: 해당 차원 값 ≥ 분할값

**시간 복잡도:**
- 인덱싱: $O(n \log n)$
- 검색: $O(\log n)$ (평균), $O(n)$ (최악)

**한계:**
- 고차원에서 비효율적 (차원의 저주)
- $d > 20$에서 Linear Search보다 느림

---

## 3. Approximate Nearest Neighbor (ANN)

### 3.1 ANN의 필요성

#### 문제 3.1.1: 차원의 저주

**고차원 공간에서:**
- 모든 점이 거의 같은 거리
- 공간 분할 비효율적
- Exact search가 느림

**해결책:**
- 근사 검색 (Approximate)
- 속도와 정확도 트레이드오프

#### 정의 3.1.1: ANN

**ANN**은 다음을 해결합니다:

$$
\text{ANN}(q, \mathcal{D}, k, \epsilon) = \{d_1, \ldots, d_k | d(q, d_i) \leq (1+\epsilon) \cdot d(q, \text{NN}(q, \mathcal{D}))\}
$$

여기서 $\epsilon$는 허용 오차입니다.

**해석:**
- 정확한 최근접 이웃의 $(1+\epsilon)$ 배 이내
- $\epsilon = 0$: Exact search
- $\epsilon > 0$: Approximate (더 빠름)

---

## 4. ANN 알고리즘 비교

### 4.1 LSH (Locality-Sensitive Hashing)

#### 정의 4.1.1: LSH

**LSH**는 유사한 벡터를 같은 해시 버킷에 매핑합니다:

$$
h: \mathbb{R}^d \rightarrow \{0, 1\}^b
$$

**성질:**
- 유사한 벡터 → 같은 해시 (높은 확률)
- 다른 벡터 → 다른 해시 (높은 확률)

#### 정리 4.1.1: LSH의 확률 보장

**LSH 함수 $h$는 다음을 만족합니다:**

$$
P(h(\mathbf{u}) = h(\mathbf{v})) = f(\text{sim}(\mathbf{u}, \mathbf{v}))
$$

여기서 $f$는 단조 증가 함수입니다.

**증명:** LSH 함수의 정의로부터

**시간 복잡도:**
- 인덱싱: $O(n \cdot d)$
- 검색: $O(n^\rho \cdot d)$ ($\rho < 1$)

### 4.2 HNSW (Hierarchical Navigable Small World)

#### 정의 4.2.1: HNSW

**HNSW**는 계층적 그래프 구조입니다:

**구조:**
- 여러 레벨의 그래프
- 상위 레벨: 긴 거리 연결 (빠른 탐색)
- 하위 레벨: 짧은 거리 연결 (정확한 검색)

#### 시각적 표현: HNSW 구조

```
HNSW 계층 구조:

Level 2 (상위):     ★───────────★
                    │           │
                    │           │
Level 1 (중간):   ★───★───★───★
                  │   │   │   │
                  │   │   │   │
Level 0 (하위):  ★─★─★─★─★─★─★
                │ │ │ │ │ │ │ │
                │ │ │ │ │ │ │ │
                ★ ★ ★ ★ ★ ★ ★ ★

검색 과정:
1. 상위 레벨에서 시작 (빠른 탐색)
2. 하위 레벨로 내려가며 정확도 향상
3. 최종 결과 반환
```

#### 정리 4.2.1: HNSW의 복잡도

**HNSW의 시간 복잡도:**

- 인덱싱: $O(n \log n \cdot d)$
- 검색: $O(\log n \cdot d)$ (평균)

**공간 복잡도:** $O(n \cdot d)$

### 4.3 IVF (Inverted File Index)

#### 정의 4.3.1: IVF

**IVF**는 벡터를 클러스터로 그룹화합니다:

**과정:**
1. **클러스터링:** K-means로 클러스터 생성
2. **인덱싱:** 각 벡터를 가장 가까운 클러스터에 할당
3. **검색:** 쿼리와 가까운 클러스터만 검색

#### 시각적 표현: IVF 구조

```
IVF 인덱스:

클러스터 C₁:  {d₁, d₂, d₃}
클러스터 C₂:  {d₄, d₅, d₆}
클러스터 C₃:  {d₇, d₈, d₉}

쿼리 q:
1. 가장 가까운 클러스터 찾기: C₁
2. C₁ 내에서만 검색: {d₁, d₂, d₃}
3. 상위 k개 반환
```

**시간 복잡도:**
- 인덱싱: $O(n \cdot k \cdot d)$ (K-means)
- 검색: $O(k \cdot d + n/k \cdot d)$

---

## 5. FAISS와 HNSW

### 5.1 FAISS (Facebook AI Similarity Search)

#### 정의 5.1.1: FAISS

**FAISS**는 효율적인 벡터 검색 라이브러리입니다.

**주요 인덱스 타입:**

1. **Flat (L2):**
   - Exact search
   - 시간: $O(n \cdot d)$

2. **IVF-Flat:**
   - Inverted File Index
   - 시간: $O(n/k \cdot d)$

3. **HNSW:**
   - Hierarchical Navigable Small World
   - 시간: $O(\log n \cdot d)$

4. **IVF-PQ:**
   - Product Quantization
   - 압축 + 빠른 검색

#### 구체적 수치 예시

**예시 5.1.1: FAISS 성능 비교**

**설정:**
- 벡터 수: $n = 1,000,000$
- 차원: $d = 1536$
- $k = 10$

**결과:**

| 인덱스 타입 | 인덱싱 시간 | 검색 시간 | 메모리 |
|------------|------------|----------|--------|
| Flat | - | 1.5초 | 6GB |
| IVF-Flat | 30초 | 0.1초 | 6GB |
| HNSW | 60초 | 0.05초 | 12GB |
| IVF-PQ | 45초 | 0.08초 | 1.5GB |

### 5.2 HNSW 상세 분석

#### 알고리즘 5.2.1: HNSW 검색

```
Algorithm: HNSWSearch(query, graph, k)
Input:
  - query: 벡터 q ∈ ℝ^d
  - graph: HNSW 그래프
  - k: 반환할 개수
Output: 상위 k개 벡터

1. entry_point ← top_level_entry
2. current ← entry_point
3. 
4. // 상위 레벨 탐색
5. for level = max_level to 1:
6.     current ← GreedySearch(current, query, level)
7. 
8. // 하위 레벨 정확 검색
9. candidates ← GreedySearch(current, query, 0)
10. return TopK(candidates, k)
```

**GreedySearch:**
```
Algorithm: GreedySearch(start, query, level)
1. current ← start
2. improved ← True
3. while improved:
4.     improved ← False
5.     neighbors ← graph.get_neighbors(current, level)
6.     for neighbor in neighbors:
7.         if distance(query, neighbor) < distance(query, current):
8.             current ← neighbor
9.             improved ← True
10. return current
```

---

## 6. CS 관점: 인덱싱과 검색 복잡도

### 6.1 시간 복잡도 비교

#### 정리 6.1.1: 알고리즘별 복잡도

| 알고리즘 | 인덱싱 | 검색 | 메모리 |
|---------|--------|------|--------|
| Linear Search | $O(1)$ | $O(n \cdot d)$ | $O(n \cdot d)$ |
| KD-Tree | $O(n \log n)$ | $O(\log n)$ (평균) | $O(n \cdot d)$ |
| LSH | $O(n \cdot d)$ | $O(n^\rho \cdot d)$ | $O(n \cdot d)$ |
| HNSW | $O(n \log n \cdot d)$ | $O(\log n \cdot d)$ | $O(n \cdot d)$ |
| IVF | $O(n \cdot k \cdot d)$ | $O(n/k \cdot d)$ | $O(n \cdot d)$ |

### 6.2 공간 복잡도

#### CS 관점 6.2.1: 메모리 사용량

**벡터 저장:**
- 각 벡터: $d \times 4$ bytes (float32)
- $n$개 벡터: $n \cdot d \cdot 4$ bytes

**인덱스 오버헤드:**
- HNSW: 그래프 구조 → 약 2배
- IVF: 클러스터 정보 → 약 1.1배
- LSH: 해시 테이블 → 약 1.2배

**예시:**
- $n = 1,000,000$, $d = 1536$
- 벡터만: $1M \times 1536 \times 4 = 6.14$ GB
- HNSW 인덱스: 약 12 GB

---

## 7. 실제 구현과 성능

### 7.1 llmkit의 벡터 검색

#### 구현 7.1.1: 기본 검색

```python
# vector_stores/base.py
def similarity_search(
    self,
    query: str,
    k: int = 4,
    **kwargs
) -> List[VectorSearchResult]:
    """
    벡터 검색: kNN(q, D, k)
    
    시간 복잡도: O(n·d + n log n)
    """
    # 1. 쿼리 임베딩
    query_vec = self.embedding_model.embed_sync([query])[0]
    
    # 2. 모든 후보와 유사도 계산
    similarities = []
    for doc_id, doc_vec in self.vectors.items():
        sim = cosine_similarity(query_vec, doc_vec)
        similarities.append((doc_id, sim))
    
    # 3. 정렬 및 상위 k개
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 4. 결과 반환
    results = []
    for doc_id, score in similarities[:k]:
        results.append(VectorSearchResult(
            document=self.documents[doc_id],
            score=score
        ))
    
    return results
```

### 7.2 최적화: 배치 처리

#### 구현 7.2.1: 배치 유사도 계산

```python
# NumPy 벡터화로 최적화
import numpy as np

def batch_similarity_search(query_vec, candidate_vecs, k):
    """
    배치 벡터화 검색
    
    시간 복잡도: O(n·d) (SIMD 활용)
    """
    query = np.array(query_vec, dtype=np.float32)
    candidates = np.array(candidate_vecs, dtype=np.float32)
    
    # 벡터화된 코사인 유사도
    similarities = np.dot(candidates, query) / (
        np.linalg.norm(candidates, axis=1) * np.linalg.norm(query)
    )
    
    # Top-k
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    return top_k_indices
```

**성능:**
- 순수 Python: $O(n \cdot d)$ (느림)
- NumPy: $O(n \cdot d)$ (빠름, SIMD)

---

## 질문과 답변 (Q&A)

### Q1: 언제 Exact Search를 사용하나요?

**A:** Exact Search 사용 시기:

1. **소규모 데이터 ($n < 10,000$):**
   - ANN 오버헤드가 더 큼
   - Linear Search가 충분히 빠름

2. **정확도가 중요:**
   - 의료, 법률 등
   - 근사 오차 허용 불가

3. **단순성:**
   - 구현이 간단
   - 디버깅 용이

### Q2: ANN 알고리즘은 어떻게 선택하나요?

**A:** 선택 기준:

1. **HNSW:**
   - 빠른 검색 필요
   - 메모리 여유
   - 권장: 대부분의 경우

2. **IVF:**
   - 메모리 제한
   - 대규모 데이터
   - 클러스터링 가능

3. **LSH:**
   - 매우 대규모
   - 정확도 약간 포기 가능

**권장:** HNSW (FAISS)

### Q3: 고차원에서도 ANN이 효과적인가요?

**A:** 네, 하지만 제한적입니다:

1. **차원의 저주:**
   - $d > 1000$: 모든 점이 거의 같은 거리
   - ANN 이점 감소

2. **해결책:**
   - 차원 축소 (PCA)
   - Product Quantization
   - 더 많은 후보 검색

---

## 참고 문헌

1. **Malkov & Yashunin (2018)**: "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
2. **Johnson et al. (2019)**: "Billion-scale similarity search with GPUs"
3. **Indyk & Motwani (1998)**: "Approximate nearest neighbors: towards removing the curse of dimensionality"

---

**작성일**: 2025-01-XX  
**버전**: 3.0 (CS 관점 + 석사 수준 확장)

