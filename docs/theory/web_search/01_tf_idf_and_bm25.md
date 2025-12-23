# TF-IDF and BM25: 검색 랭킹 함수의 수학적 기초

**석사 수준 이론 문서**  
**CS 관점 + 수학적 엄밀성**  
**기반**: llmkit WebSearch 실제 구현 분석

---

## 목차

1. [TF-IDF의 수학적 모델](#1-tf-idf의-수학적-모델)
2. [BM25 랭킹 함수](#2-bm25-랭킹-함수)
3. [TF-IDF vs BM25](#3-tf-idf-vs-bm25)
4. [구체적 계산 예시](#4-구체적-계산-예시)
5. [CS 관점: 구현과 최적화](#5-cs-관점-구현과-최적화)

---

## 1. TF-IDF의 수학적 모델

### 1.1 TF-IDF 정의

#### 정의 1.1.1: TF-IDF

**TF-IDF**는 단어의 중요도를 측정합니다:

$$
\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)
$$

**Term Frequency:**

$$
\text{TF}(t, d) = \frac{f_{t,d}}{\max\{f_{t',d} : t' \in d\}}
$$

**Inverse Document Frequency:**

$$
\text{IDF}(t, D) = \log \frac{N}{|\{d \in D : t \in d\}|}
$$

### 1.2 구체적 수치 예시

**예시 1.2.1: TF-IDF 계산**

문서 컬렉션:
- $d_1$: "고양이는 포유동물이다"
- $d_2$: "고양이는 네 발로 걷는다"
- $d_3$: "강아지는 귀여워"

단어 "고양이"에 대한 TF-IDF:

**1. TF 계산 (문서 $d_1$):**
$$
f_{\text{고양이}, d_1} = 1, \quad \max\{f_{t', d_1}\} = 1
$$

$$
\text{TF}(\text{고양이}, d_1) = \frac{1}{1} = 1.0
$$

**2. IDF 계산:**
$$
N = 3, \quad |\{d : \text{고양이} \in d\}| = 2
$$

$$
\text{IDF}(\text{고양이}, D) = \log \frac{3}{2} = \log(1.5) \approx 0.405
$$

**3. TF-IDF:**
$$
\text{TF-IDF}(\text{고양이}, d_1, D) = 1.0 \times 0.405 = 0.405
$$

---

## 2. BM25 랭킹 함수

### 2.1 BM25 정의

#### 정의 2.1.1: BM25

**BM25**는 TF-IDF의 개선된 버전입니다:

$$
\text{score}(D, Q) = \sum_{i=1}^n \text{IDF}(q_i) \times \frac{f(q_i, D) \times (k_1 + 1)}{f(q_i, D) + k_1 \times (1 - b + b \times |D| / \text{avgdl})}
$$

**파라미터:**
- $k_1 = 1.2$: TF 정규화
- $b = 0.75$: 길이 정규화

### 2.2 구체적 수치 예시

**예시 2.2.1: BM25 계산**

쿼리: $Q$ = "고양이 포유동물"
문서: $D$ = "고양이는 포유동물이다"

**단어별 계산:**

**1. "고양이":**
- $f(\text{고양이}, D) = 1$
- $|D| = 5$ (단어 수)
- $\text{avgdl} = 4$ (평균 문서 길이)
- $\text{IDF}(\text{고양이}) = 0.405$

$$
\text{score}_1 = 0.405 \times \frac{1 \times (1.2 + 1)}{1 + 1.2 \times (1 - 0.75 + 0.75 \times 5/4)}
$$

$$
= 0.405 \times \frac{2.2}{1 + 1.2 \times 1.1875} = 0.405 \times \frac{2.2}{2.425} \approx 0.367
$$

**2. "포유동물":**
- $f(\text{포유동물}, D) = 1$
- $\text{IDF}(\text{포유동물}) = 0.693$ (더 희귀)

$$
\text{score}_2 = 0.693 \times \frac{2.2}{2.425} \approx 0.628
$$

**총 점수:**
$$
\text{score}(D, Q) = 0.367 + 0.628 = 0.995
$$

---

## 3. TF-IDF vs BM25

### 3.1 비교

#### 정리 3.1.1: TF-IDF vs BM25

| 측면 | TF-IDF | BM25 |
|------|--------|------|
| TF 정규화 | 선형 | 포화 함수 |
| 길이 정규화 | 없음 | 있음 |
| 성능 | 기본 | 더 좋음 |
| 계산 | 간단 | 복잡 |

**BM25가 일반적으로 더 좋은 성능**

---

## 4. 구체적 계산 예시

### 4.1 전체 계산 과정

**예시 4.1.1: BM25 전체 계산**

문서 컬렉션:
- $d_1$: "고양이는 포유동물이다" (5단어)
- $d_2$: "강아지는 포유동물이다" (5단어)
- $d_3$: "고양이는 귀여워" (3단어)

쿼리: "고양이 포유동물"

**평균 문서 길이:**
$$
\text{avgdl} = \frac{5 + 5 + 3}{3} = \frac{13}{3} \approx 4.33
$$

**문서 $d_1$의 BM25 점수:**

**단어 "고양이":**
- $f = 1$, $|D| = 5$, $\text{IDF} = \log(3/1) = 1.099$

$$
\text{score}_1 = 1.099 \times \frac{1 \times 2.2}{1 + 1.2 \times (1 - 0.75 + 0.75 \times 5/4.33)} = 1.099 \times 0.905 = 0.995
$$

**단어 "포유동물":**
- $f = 1$, $\text{IDF} = \log(3/2) = 0.405$

$$
\text{score}_2 = 0.405 \times 0.905 = 0.367
$$

**총 점수:**
$$
\text{score}(d_1, Q) = 0.995 + 0.367 = 1.362
$$

---

## 5. CS 관점: 구현과 최적화

### 5.1 효율적인 구현

#### 알고리즘 5.1.1: BM25 계산

```
Algorithm: BM25Score(document, query, k1, b, avgdl)
Input:
  - document: 문서 D
  - query: 쿼리 Q = {q₁, ..., qₙ}
  - k1, b: 파라미터
  - avgdl: 평균 문서 길이
Output: BM25 점수

1. score ← 0
2. doc_length ← |D|
3. 
4. for term in query:
5.     term_freq ← count(term, document)
6.     idf ← log(N / df(term))
7.     
8.     numerator ← term_freq × (k1 + 1)
9.     denominator ← term_freq + k1 × (1 - b + b × doc_length / avgdl)
10.    term_score ← idf × (numerator / denominator)
11.    score ← score + term_score
12. 
13. return score
```

**시간 복잡도:** $O(|Q| \cdot |D|)$

**llmkit 구현:**
```python
# domain/web_search/engines.py: BaseSearchEngine
# facade/web_search_facade.py: WebSearch
# service/impl/web_search_service_impl.py: WebSearchServiceImpl
from abc import ABC, abstractmethod
import math
from collections import Counter

class BaseSearchEngine(ABC):
    """
    검색 엔진 베이스 클래스
    
    실제 구현:
    - domain/web_search/engines.py: BaseSearchEngine (추상 클래스)
    - facade/web_search_facade.py: WebSearch (사용자 API)
    - service/impl/web_search_service_impl.py: WebSearchServiceImpl (비즈니스 로직)
    """
    @abstractmethod
    def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """검색 실행"""
        pass

def compute_tf_idf(term: str, document: str, document_collection: List[str]) -> float:
    """
    TF-IDF 계산: TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)
    
    수학적 표현:
    - TF(t, d) = f_{t,d} / max{f_{t',d} : t' ∈ d}
    - IDF(t, D) = log(N / |{d ∈ D : t ∈ d}|)
    - TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)
    
    시간 복잡도: O(|d| + |D|)
    
    실제 구현:
    - domain/web_search/engines.py: BaseSearchEngine (기본 검색 엔진)
    - facade/web_search_facade.py: WebSearch (사용자 API)
    - service/impl/web_search_service_impl.py: WebSearchServiceImpl (비즈니스 로직)
    """
    import math
    from collections import Counter
    
    # 1. TF 계산
    doc_tokens = document.lower().split()
    term_freq = Counter(doc_tokens)
    max_freq = max(term_freq.values()) if term_freq else 1
    tf = term_freq.get(term.lower(), 0) / max_freq
    
    # 2. IDF 계산
    N = len(document_collection)
    docs_with_term = sum(1 for doc in document_collection if term.lower() in doc.lower())
    idf = math.log(N / docs_with_term) if docs_with_term > 0 else 0.0
    
    # 3. TF-IDF
    tf_idf = tf * idf
    
    return tf_idf

def compute_bm25(
    document: str,
    query: str,
    document_collection: List[str],
    k1: float = 1.2,
    b: float = 0.75
) -> float:
    """
    BM25 점수 계산: score(D, Q) = Σ IDF(q_i) × (f(q_i, D) × (k_1 + 1)) / (f(q_i, D) + k_1 × (1 - b + b × |D| / avgdl))
    
    수학적 표현:
    - score(D, Q) = Σ_{i=1}^n IDF(q_i) × (f(q_i, D) × (k_1 + 1)) / (f(q_i, D) + k_1 × (1 - b + b × |D| / avgdl))
    - k_1 = 1.2 (TF 정규화)
    - b = 0.75 (길이 정규화)
    
    시간 복잡도: O(|Q| · |D|)
    
    실제 구현:
    - domain/web_search/engines.py: BaseSearchEngine (기본 검색 엔진)
    - facade/web_search_facade.py: WebSearch (사용자 API)
    - service/impl/web_search_service_impl.py: WebSearchServiceImpl (비즈니스 로직)
    """
    import math
    from collections import Counter
    
    # 평균 문서 길이 계산
    doc_lengths = [len(doc.split()) for doc in document_collection]
    avgdl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
    
    # 문서 길이 및 단어 빈도
    doc_tokens = document.lower().split()
    doc_length = len(doc_tokens)
    term_freq = Counter(doc_tokens)
    
    # 쿼리 단어별 점수 계산
    query_terms = query.lower().split()
    score = 0.0
    
    for term in query_terms:
        # 단어 빈도
        f = term_freq.get(term, 0)
        
        # IDF 계산
        N = len(document_collection)
        docs_with_term = sum(1 for doc in document_collection if term in doc.lower())
        idf = math.log((N - docs_with_term + 0.5) / (docs_with_term + 0.5)) if docs_with_term > 0 else 0.0
        
        # BM25 점수
        numerator = f * (k1 + 1)
        denominator = f + k1 * (1 - b + b * doc_length / avgdl)
        term_score = idf * (numerator / denominator) if denominator > 0 else 0.0
        
        score += term_score
    
    return score
```

---

## 질문과 답변 (Q&A)

### Q1: TF-IDF와 BM25 중 어떤 것을 사용하나요?

**A:** 선택 기준:

**TF-IDF:**
- 간단한 구현
- 빠른 계산
- 기본 검색

**BM25:**
- 더 나은 성능
- 실무 표준
- 권장

**권장:** BM25 (대부분의 경우)

### Q2: BM25 파라미터는 어떻게 튜닝하나요?

**A:** 파라미터 튜닝:

**$k_1$ (TF 정규화):**
- 기본값: 1.2
- 범위: 1.0-2.0
- 높을수록: TF에 더 가중치

**$b$ (길이 정규화):**
- 기본값: 0.75
- 범위: 0.0-1.0
- 높을수록: 길이에 더 페널티

**권장:** 기본값으로 시작, 데이터에 맞게 조정

---

## 참고 문헌

1. **Salton & McGill (1983)**: "Introduction to Modern Information Retrieval" - TF-IDF
2. **Robertson & Zaragoza (2009)**: "The Probabilistic Relevance Framework: BM25 and Beyond"

---

**작성일**: 2025-01-XX  
**버전**: 3.0 (CS 관점 + 석사 수준 확장)

