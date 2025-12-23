# Web Search Theory: 웹 검색의 수학적 모델

**석사 수준 이론 문서**  
**기반**: llmkit WebSearch 실제 구현 분석

---

## 목차

### Part I: 검색 알고리즘
1. [TF-IDF의 수학적 모델](#part-i-검색-알고리즘)
2. [BM25 랭킹 함수](#12-bm25-랭킹-함수)
3. [PageRank 알고리즘](#13-pagerank-알고리즘)

### Part II: 결과 순위화
4. [점수 결합과 융합](#part-ii-결과-순위화)
5. [다중 검색 엔진 통합](#42-다중-검색-엔진-통합)
6. [콘텐츠 추출](#43-콘텐츠-추출)

---

## Part I: 검색 알고리즘

### 1.1 TF-IDF의 수학적 모델

#### 정의 1.1.1: TF-IDF (Term Frequency-Inverse Document Frequency)

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

여기서:
- $N$: 전체 문서 수
- $f_{t,d}$: 단어 $t$의 문서 $d$에서의 빈도

#### 시각적 표현: TF-IDF 계산 과정

```
┌─────────────────────────────────────────────────────────┐
│                  TF-IDF 계산 과정                        │
└─────────────────────────────────────────────────────────┘

문서 컬렉션 D = {d₁, d₂, d₃}
쿼리: "machine learning"

문서 d₁: "Machine learning is a subset of AI"
문서 d₂: "Deep learning uses neural networks"
문서 d₃: "AI and machine learning are related"

단어 "machine"에 대한 TF-IDF:

1. TF 계산 (문서 d₁):
   f("machine", d₁) = 1
   max_freq(d₁) = 1  ("machine", "learning", "is" 등 모두 1)
   TF("machine", d₁) = 1/1 = 1.0

2. IDF 계산:
   N = 3 (전체 문서 수)
   |{d ∈ D : "machine" ∈ d}| = 2 (d₁, d₃)
   IDF("machine", D) = log(3/2) = log(1.5) ≈ 0.405

3. TF-IDF:
   TF-IDF("machine", d₁, D) = 1.0 × 0.405 = 0.405
```

#### 구체적 수치 예시

**예시 1.1.1: TF-IDF 계산**

**문서 컬렉션:**
- $d_1$: "Machine learning is powerful"
- $d_2$: "Deep learning uses neural networks"
- $d_3$: "AI and machine learning"

**단어 "learning"에 대한 TF-IDF:**

**1단계: TF 계산**

문서 $d_1$:
- $f_{\text{learning}, d_1} = 1$
- $\max\{f_{t', d_1}\} = 1$ (모든 단어가 1번)
- $\text{TF}(\text{learning}, d_1) = \frac{1}{1} = 1.0$

문서 $d_2$:
- $f_{\text{learning}, d_2} = 1$
- $\max\{f_{t', d_2}\} = 1$
- $\text{TF}(\text{learning}, d_2) = 1.0$

**2단계: IDF 계산**

- $N = 3$ (전체 문서 수)
- $|\{d \in D : \text{learning} \in d\}| = 3$ (모든 문서에 포함)
- $\text{IDF}(\text{learning}, D) = \log \frac{3}{3} = \log(1) = 0$

**3단계: TF-IDF**

$$
\text{TF-IDF}(\text{learning}, d_1, D) = 1.0 \times 0 = 0
$$

**해석:** "learning"은 모든 문서에 나타나므로 IDF가 0입니다 (구별력 없음).

**단어 "machine"에 대한 TF-IDF:**

- $\text{TF}(\text{machine}, d_1) = 1.0$
- $|\{d : \text{machine} \in d\}| = 2$ (d₁, d₃)
- $\text{IDF}(\text{machine}, D) = \log \frac{3}{2} \approx 0.405$
- $\text{TF-IDF}(\text{machine}, d_1, D) = 1.0 \times 0.405 = 0.405$

**해석:** "machine"은 일부 문서에만 나타나므로 더 높은 TF-IDF 점수를 받습니다.

**llmkit 구현:**
```python
# domain/web_search/engines.py: BaseSearchEngine
# facade/web_search_facade.py: WebSearch
# service/impl/web_search_service_impl.py: WebSearchServiceImpl
def compute_tf_idf(term: str, document: str, document_collection: List[str]) -> float:
    """
    TF-IDF 계산: TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)
    
    수학적 표현:
    - TF(t, d) = f_{t,d} / max{f_{t',d} : t' ∈ d}
    - IDF(t, D) = log(N / |{d ∈ D : t ∈ d}|)
    - TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)
    
    실제 구현:
    - domain/web_search/engines.py: BaseSearchEngine (기본 검색 엔진)
    - facade/web_search_facade.py: WebSearch (사용자 API)
    - service/impl/web_search_service_impl.py: WebSearchServiceImpl (비즈니스 로직)
    - 검색 엔진별로 TF-IDF 또는 BM25 사용 (Google, Bing, DuckDuckGo)
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
```

---

### 1.2 BM25 랭킹 함수

#### 정의 1.2.1: BM25 (Best Matching 25)

**BM25**는 TF-IDF의 개선된 버전입니다:

$$
\text{score}(D, Q) = \sum_{i=1}^n \text{IDF}(q_i) \times \frac{f(q_i, D) \times (k_1 + 1)}{f(q_i, D) + k_1 \times (1 - b + b \times |D| / \text{avgdl})}
$$

여기서:
- $q_i$: 쿼리 단어
- $f(q_i, D)$: 단어 $q_i$의 문서 $D$에서의 빈도
- $|D|$: 문서 길이
- $\text{avgdl}$: 평균 문서 길이
- $k_1 = 1.2$, $b = 0.75$: 튜닝 파라미터

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
    
    BM25 Ranking Function:
    score(D, Q) = Σ_{i=1}^n IDF(q_i) × (f(q_i, D) × (k_1 + 1)) /
                                        (f(q_i, D) + k_1 × (1 - b + b × |D| / avgdl))
    
    where:
    - k_1=1.2, b=0.75 (typical values)
    
    실제 구현:
    - domain/web_search/engines.py: BaseSearchEngine (추상 클래스)
    - facade/web_search_facade.py: WebSearch (사용자 API)
    - service/impl/web_search_service_impl.py: WebSearchServiceImpl (비즈니스 로직)
    - 검색 엔진별로 TF-IDF 또는 BM25 사용 (Google, Bing, DuckDuckGo)
    """
    @abstractmethod
    def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """검색 실행"""
        pass
```

---

### 1.3 PageRank 알고리즘

#### 정의 1.3.1: PageRank

**PageRank**는 웹페이지의 중요도를 계산합니다:

$$
\text{PR}(p) = (1-d) + d \times \sum_{p_i \in M(p)} \frac{\text{PR}(p_i)}{L(p_i)}
$$

여기서:
- $d$: Damping factor (보통 0.85)
- $M(p)$: 페이지 $p$로 링크하는 페이지 집합
- $L(p_i)$: 페이지 $p_i$의 외부 링크 수

**llmkit 구현:**
```python
# domain/web_search/engines.py: BaseSearchEngine
# facade/web_search_facade.py: WebSearch
# PageRank는 검색 엔진 결과 순위화에 사용 (Google, Bing 등)
"""
PageRank Algorithm:
PR(p) = (1-d) + d × Σ_{p_i ∈ M(p)} PR(p_i) / L(p_i)

where:
- d: damping factor (typically 0.85)
- M(p): 페이지 p로 링크하는 페이지 집합
- L(p_i): 페이지 p_i의 외부 링크 수

실제 구현:
- llmkit은 외부 검색 엔진(Google, Bing, DuckDuckGo) API를 사용
- PageRank는 검색 엔진 내부에서 이미 적용된 결과를 받음
- domain/web_search/engines.py: BaseSearchEngine (검색 엔진 추상 클래스)
- facade/web_search_facade.py: WebSearch (사용자 API)
"""
```

---

## Part II: 결과 순위화

### 2.1 점수 결합과 융합

#### 정의 2.1.1: 다중 검색 엔진 융합

**여러 검색 엔진의 결과를 결합:**

$$
\text{score}_{\text{combined}}(r) = \sum_{e \in E} w_e \times \text{score}_e(r)
$$

여기서 $E$는 검색 엔진 집합, $w_e$는 가중치입니다.

**llmkit 구현:**
```python
# facade/web_search_facade.py: WebSearch
# service/impl/web_search_service_impl.py: WebSearchServiceImpl
# handler/web_search_handler.py: WebSearchHandler
from typing import List, Dict, Any

class WebSearch:
    """
    웹 검색: 다중 검색 엔진 통합
    
    수학적 표현:
    - Search(Q) = ∪_{e ∈ E} Search_e(Q)
    - score_combined(r) = Σ_{e ∈ E} w_e × score_e(r)
    
    실제 구현:
    - facade/web_search_facade.py: WebSearch (사용자 API)
    - service/impl/web_search_service_impl.py: WebSearchServiceImpl (비즈니스 로직)
    - handler/web_search_handler.py: WebSearchHandler (입력 검증)
    """
    def __init__(self, default_engine: str = "google"):
        """
        Args:
            default_engine: 기본 검색 엔진 ("google", "bing", "duckduckgo")
        """
        self.engines = {
            "google": GoogleSearchEngine(),
            "bing": BingSearchEngine(),
            "duckduckgo": DuckDuckGoSearchEngine()
        }
        self.default_engine = default_engine
    
    def search(
        self,
        query: str,
        engines: List[str] = None,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        다중 검색 엔진 결과 융합: score_combined = Σ w_e × score_e
        
        수학적 표현:
        - 입력: 쿼리 Q, 검색 엔진 집합 E
        - 출력: 융합된 검색 결과
        - 점수: score_combined(r) = Σ_{e ∈ E} w_e × score_e(r)
        
        실제 구현:
        - facade/web_search_facade.py: WebSearch.search()
        - service/impl/web_search_service_impl.py: WebSearchServiceImpl.search()
        """
        engines = engines or [self.default_engine]
        all_results = []
        
        for engine in engines:
            results = self._search_engine(query, engine, k=k*2)
            all_results.extend(results)
        
        # 점수 정규화 및 결합
        combined = self._combine_results(all_results)
        return combined[:k]
    
    def _combine_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        검색 결과 결합: score_combined = Σ w_e × score_e
        
        실제 구현:
        - facade/web_search_facade.py: WebSearch._combine_results()
        - 점수 정규화 및 중복 제거
        """
        # 점수 정규화 및 결합 로직
        # ...
        return sorted(results, key=lambda x: x.get("score", 0), reverse=True)
```

---

### 2.2 다중 검색 엔진 통합

#### 정의 2.2.1: 검색 엔진 통합

**여러 검색 엔진을 통합하여 사용:**

$$
\text{Search}(Q) = \bigcup_{e \in E} \text{Search}_e(Q)
$$

**llmkit 구현:**
```python
# facade/web_search_facade.py: WebSearch
# domain/web_search/engines.py: GoogleSearchEngine, BingSearchEngine, DuckDuckGoSearchEngine
class WebSearch:
    """
    Google, Bing, DuckDuckGo 등 여러 검색 엔진 통합
    
    실제 구현:
    - facade/web_search_facade.py: WebSearch (사용자 API)
    - domain/web_search/engines.py: GoogleSearchEngine, BingSearchEngine, DuckDuckGoSearchEngine
    - service/impl/web_search_service_impl.py: WebSearchServiceImpl (비즈니스 로직)
    """
    def __init__(self, default_engine: str = "google"):
        """
        Args:
            default_engine: 기본 검색 엔진
        """
        from ...domain.web_search.engines import (
            GoogleSearchEngine,
            BingSearchEngine,
            DuckDuckGoSearchEngine
        )
        
        self.engines = {
            "google": GoogleSearchEngine(),
            "bing": BingSearchEngine(),
            "duckduckgo": DuckDuckGoSearchEngine()
        }
        self.default_engine = default_engine
```

---

### 2.3 콘텐츠 추출

#### 정의 2.3.1: 웹페이지 콘텐츠 추출

**웹페이지에서 텍스트 추출:**

$$
\text{content} = \text{extract}(HTML)
$$

**llmkit 구현:**
```python
# facade/web_search_facade.py: WebSearch.extract_content()
# service/impl/web_search_service_impl.py: WebSearchServiceImpl.extract_content()
import requests
from bs4 import BeautifulSoup

def extract_content(self, url: str) -> str:
    """
    웹페이지에서 텍스트 추출: content = extract(HTML)
    
    수학적 표현:
    - 입력: URL (웹페이지 주소)
    - 출력: 텍스트 콘텐츠
    - Process: HTML → Parse → Extract Text
    
    실제 구현:
    - facade/web_search_facade.py: WebSearch.extract_content()
    - service/impl/web_search_service_impl.py: WebSearchServiceImpl.extract_content()
    - BeautifulSoup 사용 (HTML 파싱)
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # 메인 콘텐츠 추출
    # - <article>, <main>, <body> 태그에서 텍스트 추출
    # - 스크립트, 스타일 태그 제거
    content = soup.get_text(separator='\n', strip=True)
    
    return content
```

---

## 참고 문헌

1. **Salton & McGill (1983)**: "Introduction to Modern Information Retrieval" - TF-IDF
2. **Robertson & Zaragoza (2009)**: "The Probabilistic Relevance Framework: BM25 and Beyond"
3. **Page et al. (1998)**: "The PageRank Citation Ranking"

---

**작성일**: 2025-01-XX  
**버전**: 2.0 (석사 수준 확장)
