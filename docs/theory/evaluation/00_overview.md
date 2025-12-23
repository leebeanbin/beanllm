# Evaluation Theory: LLM 평가의 수학적 모델과 메트릭

**석사 수준 이론 문서**  
**기반**: llmkit Evaluation, Metrics 실제 구현 분석

---

## 목차

### Part I: 평가 메트릭의 수학적 기초
1. [평가 메트릭의 형식적 정의](#part-i-평가-메트릭의-수학적-기초)
2. [정확도 기반 메트릭](#12-정확도-기반-메트릭)
3. [유사도 기반 메트릭](#13-유사도-기반-메트릭)

### Part II: RAG 평가 메트릭
4. [Context Recall: 검색 완전성 평가](#part-ii-rag-평가-메트릭)
5. [Context Precision: 검색 정확도 평가](#42-context-precision-검색-정확도-평가)
6. [Faithfulness: 환각 검출](#43-faithfulness-환각-검출)
7. [Answer Relevance: 답변 관련성](#44-answer-relevance-답변-관련성)

### Part III: LLM-as-Judge
8. [LLM-as-Judge의 확률 모델](#part-iii-llm-as-judge)
9. [프롬프트 엔지니어링과 평가 일관성](#42-프롬프트-엔지니어링과-평가-일관성)

### Part IV: Human-in-the-Loop 평가
10. [인간 피드백의 통계적 모델](#part-iv-human-in-the-loop-평가)
11. [하이브리드 평가의 가중 평균](#102-하이브리드-평가의-가중-평균)
12. [비교 평가와 Bradley-Terry 모델](#103-비교-평가와-bradley-terry-모델)

### Part V: 지속적 평가와 드리프트 감지
13. [Continuous Evaluation의 시간 시리즈 모델](#part-v-지속적-평가와-드리프트-감지)
14. [Drift Detection의 통계적 검정](#132-drift-detection의-통계적-검정)
15. [트렌드 분석과 상관관계](#133-트렌드-분석과-상관관계)

### Part VI: 구조화된 평가
16. [Rubric-Driven Grading의 가중 합](#part-vi-구조화된-평가)
17. [CheckEval의 Boolean 평가 모델](#162-checkeval의-boolean-평가-모델)

---

## Part I: 평가 메트릭의 수학적 기초

### 1.1 평가 메트릭의 형식적 정의

#### 정의 1.1.1: 평가 메트릭 (Evaluation Metric)

**평가 메트릭**은 다음 함수로 정의됩니다:

$$
M: \mathcal{P} \times \mathcal{R} \rightarrow [0, 1]
$$

여기서:
- $\mathcal{P}$: 예측 공간 (predictions)
- $\mathcal{R}$: 참조 공간 (references)
- $[0, 1]$: 정규화된 점수 범위

#### 성질 1.1.1: 메트릭의 기본 성질

1. **정규화 (Normalization)**
   $$
   \forall p, r: 0 \leq M(p, r) \leq 1
   $$

2. **대칭성 (일부 메트릭)**
   $$
   M(p, r) = M(r, p) \text{ (일부 메트릭만)}
   $$

3. **항등성 (Identity)**
   $$
   M(p, p) = 1 \text{ (완벽한 일치)}
   $$

**llmkit 구현:**
```python
# domain/evaluation/base_metric.py: BaseMetric
# domain/evaluation/results.py: EvaluationResult
class BaseMetric:
    """
    평가 메트릭: M: P × R → [0, 1]
    
    수학적 정의:
    - P: 예측 공간 (predictions)
    - R: 참조 공간 (references)
    - [0, 1]: 정규화된 점수 범위
    
    실제 구현 경로:
    - domain/evaluation/base_metric.py: BaseMetric (추상 클래스)
    - domain/evaluation/metrics.py: 구체적 메트릭 구현체들
    - domain/evaluation/results.py: EvaluationResult (결과 데이터 구조)
    """
    def compute(
        self, 
        prediction: str, 
        reference: str, 
        **kwargs
    ) -> EvaluationResult:
        """
        메트릭 계산: score = M(prediction, reference)
        
        Args:
            prediction: 예측 텍스트 (p ∈ P)
            reference: 참조 텍스트 (r ∈ R)
            **kwargs: 추가 파라미터 (contexts, ground_truth_contexts 등)
        
        Returns:
            EvaluationResult: score ∈ [0, 1]
        """
        ...
```

---

### 1.2 정확도 기반 메트릭

#### 정의 1.2.1: Exact Match (EM)

**Exact Match**는 완전 일치 여부를 평가합니다:

$$
\text{EM}(p, r) = \begin{cases}
1 & \text{if } p = r \\
0 & \text{otherwise}
\end{cases}
$$

**llmkit 구현:**
```python
# domain/evaluation/metrics.py: ExactMatchMetric
class ExactMatchMetric(BaseMetric):
    """
    Exact Match: EM(p, r) = 1 if p == r else 0
    """
    def __init__(self, case_sensitive: bool = True, normalize_whitespace: bool = True):
        super().__init__("exact_match", MetricType.SIMILARITY)
        self.case_sensitive = case_sensitive
        self.normalize_whitespace = normalize_whitespace
    
    def compute(self, prediction: str, reference: str, **kwargs) -> EvaluationResult:
        """
        Exact Match 계산: EM(p, r) = 1 if p == r else 0
        
        실제 구현:
        - domain/evaluation/metrics.py: ExactMatchMetric
        - 정규화 옵션 지원 (대소문자, 공백)
        """
        pred = prediction
        ref = reference
        
        # 정규화
        if self.normalize_whitespace:
            pred = " ".join(pred.split())
            ref = " ".join(ref.split())
        
        if not self.case_sensitive:
            pred = pred.lower()
            ref = ref.lower()
        
        score = 1.0 if pred == ref else 0.0
        
        return EvaluationResult(
            metric_name="exact_match",
            score=score,
            metadata={"prediction": prediction, "reference": reference}
        )
```

#### 정의 1.2.2: F1 Score

**F1 Score**는 Precision과 Recall의 조화 평균입니다:

$$
\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

여기서:
- **Precision**: $P = \frac{|\text{common tokens}|}{|\text{prediction tokens}|}$
- **Recall**: $R = \frac{|\text{common tokens}|}{|\text{reference tokens}|}$

**구체적 수치 예시:**

**예시 1.2.1: F1 Score 계산**

- **예측**: $p$ = "고양이는 포유동물이다"
- **참조**: $r$ = "고양이는 포유동물"

**토큰화:**
- $p_{\text{tokens}} = \{$"고양이는", "포유동물이다"$\}$
- $r_{\text{tokens}} = \{$"고양이는", "포유동물"$\}$

**공통 토큰:**
- $\text{common} = \{$"고양이는"$\}$
- $|\text{common}| = 1$

**계산:**
- $\text{Precision} = \frac{1}{2} = 0.5$
- $\text{Recall} = \frac{1}{2} = 0.5$
- $\text{F1} = \frac{2 \times 0.5 \times 0.5}{0.5 + 0.5} = 0.5$

**llmkit 구현:**
```python
# domain/evaluation/metrics.py: F1ScoreMetric
from collections import Counter

class F1ScoreMetric(BaseMetric):
    """
    F1 Score: F1 = 2·P·R / (P + R)
    
    where:
    - P = |common tokens| / |prediction tokens|
    - R = |common tokens| / |reference tokens|
    """
    def __init__(self):
        super().__init__("f1_score", MetricType.SIMILARITY)
    
    def _tokenize(self, text: str) -> List[str]:
        """간단한 토큰화 (공백 기준)"""
        return text.lower().split()
    
    def compute(self, prediction: str, reference: str, **kwargs) -> EvaluationResult:
        """
        F1 Score 계산
        
        실제 구현:
        - domain/evaluation/metrics.py: F1ScoreMetric
        - 토큰 기반 오버랩 계산
        """
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)
        
        # 공통 토큰 계산 (집합 교집합)
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common.values())
        
        # Precision & Recall
        precision = num_common / len(pred_tokens) if pred_tokens else 0.0
        recall = num_common / len(ref_tokens) if ref_tokens else 0.0
        
        # F1 Score
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return EvaluationResult(
            metric_name="f1_score",
            score=f1,
            metadata={
                "precision": precision,
                "recall": recall,
                "common_tokens": num_common
            }
        )
```

---

### 1.3 유사도 기반 메트릭

#### 정의 1.3.1: BLEU Score

**BLEU (Bilingual Evaluation Understudy)**는 n-gram 기반 정밀도를 측정합니다:

$$
\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$

여기서:
- $p_n$: n-gram 정밀도
- $w_n$: 가중치 (일반적으로 $w_n = 1/N$)
- $\text{BP}$: Brevity Penalty

**Brevity Penalty:**

$$
\text{BP} = \begin{cases}
1 & \text{if } |p| > |r| \\
e^{1 - |r|/|p|} & \text{if } |p| \leq |r|
\end{cases}
$$

**n-gram 정밀도:**

$$
p_n = \frac{\sum_{\text{ngram} \in p} \text{Count}_{\text{clip}}(\text{ngram})}{\sum_{\text{ngram} \in p} \text{Count}(\text{ngram})}
$$

**구체적 수치 예시:**

**예시 1.3.1: BLEU-4 계산**

- **예측**: $p$ = "the cat is on the mat"
- **참조**: $r$ = "the cat is sitting on the mat"

**1-gram 정밀도:**
- $p$의 1-gram: $\{$"the"(2), "cat"(1), "is"(1), "on"(1), "mat"(1)$\}$
- $r$의 1-gram: $\{$"the"(2), "cat"(1), "is"(1), "sitting"(1), "on"(1), "mat"(1)$\}$
- $\text{Count}_{\text{clip}} = \min(2, 2) + \min(1, 1) + \min(1, 1) + \min(1, 0) + \min(1, 1) + \min(1, 1) = 6$
- $p_1 = \frac{6}{6} = 1.0$

**2-gram 정밀도:**
- $p$의 2-gram: $\{$"the cat", "cat is", "is on", "on the", "the mat"$\}$
- $r$의 2-gram: $\{$"the cat", "cat is", "is sitting", "sitting on", "on the", "the mat"$\}$
- $\text{Count}_{\text{clip}} = 1 + 1 + 0 + 1 + 1 + 1 = 5$
- $p_2 = \frac{5}{5} = 1.0$

**BLEU-4 (균등 가중치):**
- $p_1 = 1.0, p_2 = 1.0, p_3 = 0.75, p_4 = 0.5$ (가정)
- $\text{BP} = 1$ ($|p| = 6 \leq |r| = 7$이지만 짧아서 패널티 없음)
- $\text{BLEU-4} = 1 \times \exp\left(\frac{1}{4}(\log 1.0 + \log 1.0 + \log 0.75 + \log 0.5)\right) \approx 0.84$

**llmkit 구현:**
```python
# domain/evaluation/metrics.py: BLEUMetric
import math
from collections import Counter

class BLEUMetric(BaseMetric):
    """
    BLEU Score: BLEU = BP · exp(Σ w_n log p_n)
    
    where:
    - p_n: n-gram precision
    - w_n: weight (typically 1/N)
    - BP: Brevity Penalty
    """
    def __init__(self, n: int = 4):
        super().__init__("bleu", MetricType.SIMILARITY)
        self.n = n
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """n-gram 추출"""
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    def _precision_n(self, pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
        """n-gram precision 계산"""
        pred_ngrams = self._get_ngrams(pred_tokens, n)
        ref_ngrams = self._get_ngrams(ref_tokens, n)
        
        # Clipped count
        clipped_count = sum(
            min(pred_ngrams[ngram], ref_ngrams[ngram])
            for ngram in pred_ngrams
        )
        total_count = sum(pred_ngrams.values())
        
        return clipped_count / total_count if total_count > 0 else 0.0
    
    def _brevity_penalty(self, pred_len: int, ref_len: int) -> float:
        """Brevity Penalty: BP = 1 if pred_len > ref_len else exp(1 - ref_len/pred_len)"""
        if pred_len > ref_len:
            return 1.0
        return math.exp(1 - ref_len / pred_len) if pred_len > 0 else 0.0
    
    def compute(self, prediction: str, reference: str, **kwargs) -> EvaluationResult:
        """
        BLEU Score 계산
        
        실제 구현:
        - domain/evaluation/metrics.py: BLEUMetric
        - BLEU-1 to BLEU-4 계산 (기본값: n=4)
        """
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)
        
        # n-gram 정밀도 계산
        precisions = []
        for n in range(1, self.n + 1):
            p_n = self._precision_n(pred_tokens, ref_tokens, n)
            precisions.append(p_n)
        
        # Brevity Penalty
        bp = self._brevity_penalty(len(pred_tokens), len(ref_tokens))
        
        # BLEU 계산
        if any(p == 0 for p in precisions):
            bleu = 0.0
        else:
            bleu = bp * math.exp(sum(math.log(p) for p in precisions) / self.n)
        
        return EvaluationResult(
            metric_name="bleu",
            score=bleu,
            metadata={
                "precisions": precisions,
                "brevity_penalty": bp,
                "n": self.n
            }
        )
```

#### 정의 1.3.2: ROUGE Score

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**는 Recall 중심 평가입니다:

**ROUGE-N:**

$$
\text{ROUGE-N} = \frac{\sum_{s \in r} \text{Count}_{\text{match}}(\text{ngram}_n, s)}{\sum_{s \in r} \text{Count}(\text{ngram}_n, s)}
$$

**ROUGE-L (Longest Common Subsequence):**

$$
\text{ROUGE-L} = \frac{\text{LCS}(p, r)}{|r|}
$$

**구체적 수치 예시:**

**예시 1.3.2: ROUGE-L 계산**

- **예측**: $p$ = "the cat is on the mat"
- **참조**: $r$ = "the cat is sitting on the mat"

**LCS 계산 (동적 프로그래밍):**
```
        t  h  e     c  a  t     i  s     s  i  t  t  i  n  g     o  n     t  h  e     m  a  t
t    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
h    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
e    [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
...
```

**LCS 길이**: 6 ("the cat is on the mat")
- $\text{ROUGE-L Precision} = \frac{6}{6} = 1.0$
- $\text{ROUGE-L Recall} = \frac{6}{7} \approx 0.857$
- $\text{ROUGE-L F1} = \frac{2 \times 1.0 \times 0.857}{1.0 + 0.857} \approx 0.923$

**llmkit 구현:**
```python
# evaluation/metrics.py: ROUGEMetric
class ROUGEMetric(BaseMetric):
    def _lcs_length(self, x: List[str], y: List[str]) -> int:
        """Longest Common Subsequence 길이"""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    def compute(self, prediction: str, reference: str, **kwargs):
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        
        lcs = self._lcs_length(pred_tokens, ref_tokens)
        precision = lcs / len(pred_tokens) if pred_tokens else 0.0
        recall = lcs / len(ref_tokens) if ref_tokens else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return EvaluationResult(metric_name="rouge-l", score=f1)
```

---

## Part II: RAG 평가 메트릭

### 2.1 Context Recall: 검색 완전성 평가

#### 정의 2.1.1: Context Recall

**Context Recall**은 모든 관련 문서가 검색되었는지 평가합니다:

$$
\text{Context Recall} = \frac{|\{d \in \mathcal{D}_{\text{gt}} : \exists d' \in \mathcal{D}_{\text{ret}} \text{ s.t. } \text{sim}(d, d') \geq \theta\}|}{|\mathcal{D}_{\text{gt}}|}
$$

여기서:
- $\mathcal{D}_{\text{gt}}$: Ground truth 관련 문서 집합
- $\mathcal{D}_{\text{ret}}$: 검색된 문서 집합
- $\text{sim}(d, d')$: 문서 유사도 (임베딩 또는 토큰 기반)
- $\theta$: 유사도 임계값

#### 시각적 표현: Context Recall 계산

```
┌─────────────────────────────────────────────────────────┐
│              Context Recall 평가                        │
└─────────────────────────────────────────────────────────┘

Ground Truth 문서: D_gt = {d₁, d₂, d₃, d₄}
검색된 문서: D_ret = {d₁, d₅, d₃}

매칭:
- d₁ ∈ D_gt ∩ D_ret → 매칭 ✓
- d₂ ∈ D_gt, d₂ ∉ D_ret → 누락 ✗
- d₃ ∈ D_gt ∩ D_ret → 매칭 ✓
- d₄ ∈ D_gt, d₄ ∉ D_ret → 누락 ✗

Context Recall = 2/4 = 0.5 (50%)
```

#### 구체적 수치 예시

**예시 2.1.1: Context Recall 계산**

**Ground Truth 문서:**
- $d_1$: "고양이는 포유동물이다"
- $d_2$: "고양이는 네 발로 걷는다"
- $d_3$: "고양이는 야행성 동물이다"
- $d_4$: "고양이는 육식동물이다"

**검색된 문서:**
- $d_1$: "고양이는 포유동물이다"
- $d_5$: "강아지는 귀여워" (관련 없음)
- $d_3$: "고양이는 야행성 동물이다"

**임베딩 기반 매칭 (코사인 유사도 > 0.8):**
- $\text{sim}(d_1, d_1) = 1.0 \geq 0.8$ → 매칭 ✓
- $\text{sim}(d_2, d_1) = 0.65 < 0.8$ → 불일치
- $\text{sim}(d_2, d_5) = 0.12 < 0.8$ → 불일치
- $\text{sim}(d_3, d_3) = 1.0 \geq 0.8$ → 매칭 ✓
- $\text{sim}(d_4, d_1) = 0.58 < 0.8$ → 불일치
- $\text{sim}(d_4, d_3) = 0.42 < 0.8$ → 불일치

**결과:**
- 매칭된 문서: $\{d_1, d_3\}$ (2개)
- $\text{Context Recall} = \frac{2}{4} = 0.5$

**llmkit 구현:**
```python
# domain/evaluation/metrics.py: ContextRecallMetric
class ContextRecallMetric(BaseMetric):
    """
    Context Recall: CR = |{d ∈ D_gt : ∃d' ∈ D_ret, sim(d, d') ≥ θ}| / |D_gt|
    
    where:
    - D_gt: Ground truth 관련 문서 집합
    - D_ret: 검색된 문서 집합
    - sim(d, d'): 문서 유사도 (임베딩 또는 토큰 기반)
    - θ: 유사도 임계값 (기본값: 0.7-0.8)
    """
    def __init__(self, embedding_function: Optional[Callable] = None):
        super().__init__("context_recall", MetricType.RAG)
        self.embedding_function = embedding_function
    
    def compute(
        self,
        prediction: str,
        reference: str,
        contexts: Optional[List[str]] = None,
        ground_truth_contexts: Optional[List[str]] = None,
        **kwargs,
    ) -> EvaluationResult:
        """
        Context Recall 계산
        
        실제 구현:
        - domain/evaluation/metrics.py: ContextRecallMetric
        - 임베딩 기반 매칭 (embedding_function 제공 시)
        - 토큰 기반 매칭 (폴백)
        """
        if not ground_truth_contexts:
            return EvaluationResult(
                metric_name="context_recall", 
                score=0.0,
                metadata={"error": "No ground truth contexts provided"}
            )
        
        if not contexts:
            return EvaluationResult(
                metric_name="context_recall",
                score=0.0,
                metadata={"error": "No retrieved contexts provided"}
            )
        
        # 임베딩 기반 매칭 (더 정확)
        if self.embedding_function:
            recall = self._compute_recall_with_embeddings(
                contexts, 
                ground_truth_contexts,
                threshold=0.7
            )
        else:
            # 토큰 기반 매칭 (간단한 방법)
            recall = self._compute_recall_with_tokens(
                contexts,
                ground_truth_contexts,
                threshold=0.3  # 30% 토큰 오버랩
            )
        
        return EvaluationResult(
            metric_name="context_recall",
            score=recall,
            metadata={
                "retrieved_count": len(contexts),
                "ground_truth_count": len(ground_truth_contexts),
                "method": "embedding" if self.embedding_function else "token"
            }
        )
    
    def _compute_recall_with_embeddings(
        self, contexts: List[str], ground_truth_contexts: List[str]
    ) -> float:
        """임베딩 기반 재현율 계산 (코사인 유사도 > 0.7)"""
        # domain/evaluation/metrics.py: Line 632-660 참조
        ...
    
    def _compute_recall_with_tokens(
        self, contexts: List[str], ground_truth_contexts: List[str]
    ) -> float:
        """토큰 기반 재현율 계산 (30% 이상 오버랩)"""
        # domain/evaluation/metrics.py: Line 662-687 참조
        ...
```

---

### 2.2 Context Precision: 검색 정확도 평가

#### 정의 2.2.1: Context Precision

**Context Precision**은 검색된 문서가 질문에 대한 답변과 얼마나 관련있는지 평가합니다:

$$
\text{Context Precision} = \frac{|\{d \in \mathcal{D}_{\text{ret}} : \text{relevant}(d, q, a)\}|}{|\mathcal{D}_{\text{ret}}|}
$$

여기서 $\text{relevant}(d, q, a)$는 문서 $d$가 질문 $q$와 답변 $a$에 관련있다는 것을 의미합니다.

**llmkit 구현:**
```python
# evaluation/metrics.py: ContextPrecisionMetric
class ContextPrecisionMetric(BaseMetric):
    def compute(
        self,
        prediction: str,
        reference: str,
        contexts: Optional[List[str]] = None,
        **kwargs,
    ) -> EvaluationResult:
        """
        Context Precision 계산:
        CP = |{d ∈ D_ret : relevant(d, q, a)}| / |D_ret|
        """
        if not contexts:
            return EvaluationResult(
                metric_name="context_precision",
                score=0.0,
                metadata={"error": "No contexts provided"}
            )
        
        answer_tokens = set(prediction.lower().split())
        relevant_count = 0
        
        for ctx in contexts:
            ctx_tokens = set(ctx.lower().split())
            overlap = len(answer_tokens & ctx_tokens)
            
            # 충분한 오버랩이 있으면 관련있다고 판단
            if overlap >= min(3, len(ctx_tokens) * 0.3):
                relevant_count += 1
        
        precision = relevant_count / len(contexts)
        
        return EvaluationResult(
            metric_name="context_precision",
            score=precision,
            metadata={
                "total_contexts": len(contexts),
                "relevant_contexts": relevant_count
            }
        )
```

---

### 2.3 Faithfulness: 환각 검출

#### 정의 2.3.1: Faithfulness

**Faithfulness**는 생성된 답변이 제공된 컨텍스트에 충실한지 평가합니다:

$$
\text{Faithfulness} = P(\text{all claims in } a \text{ are supported by } \mathcal{D}_{\text{ret}})
$$

**환각 검출:**

$$
\text{Hallucination Rate} = 1 - \text{Faithfulness}
$$

**llmkit 구현:**
```python
# evaluation/metrics.py: FaithfulnessMetric
class FaithfulnessMetric(BaseMetric):
    def compute(
        self,
        prediction: str,
        reference: str,
        contexts: Optional[List[str]] = None,
        **kwargs,
    ) -> EvaluationResult:
        """
        Faithfulness 평가:
        F = P(all claims in answer are supported by contexts)
        """
        if not contexts:
            return EvaluationResult(
                metric_name="faithfulness",
                score=0.0,
                metadata={"error": "No contexts provided"}
            )
        
        client = self._get_client()
        context_text = "\n\n".join(contexts)
        
        prompt = (
            f"Given the following context:\n{context_text}\n\n"
            f"Evaluate if the following statement is faithful to the context "
            f"(i.e., all information is supported by the context):\n{prediction}\n\n"
            f"Respond with a score from 0 to 1, where 1 means fully faithful.\n"
            f"Format: SCORE: <number>"
        )
        
        response = client.chat([{"role": "user", "content": prompt}])
        output = response.content
        
        score_match = re.search(r"SCORE:\s*([\d.]+)", output)
        score = float(score_match.group(1)) if score_match else 0.5
        
        return EvaluationResult(
            metric_name="faithfulness",
            score=score,
            metadata={"contexts_count": len(contexts)}
        )
```

---

### 2.4 Answer Relevance: 답변 관련성

#### 정의 2.4.1: Answer Relevance

**Answer Relevance**는 생성된 답변이 질문과 얼마나 관련있는지 평가합니다:

$$
\text{Answer Relevance} = \text{LLM-Judge}(\text{relevance}(a, q))
$$

**llmkit 구현:**
```python
# evaluation/metrics.py: AnswerRelevanceMetric
class AnswerRelevanceMetric(BaseMetric):
    def compute(self, prediction: str, reference: str, **kwargs) -> EvaluationResult:
        """
        Answer Relevance 평가:
        AR = LLM-Judge(relevance(answer, question))
        """
        question = reference
        answer = prediction
        
        judge = LLMJudgeMetric(
            client=self.client,
            criterion="relevance",
            use_reference=True
        )
        
        result = judge.compute(answer, question)
        result.metric_name = self.name
        
        return result
```

---

## Part III: LLM-as-Judge

### 3.1 LLM-as-Judge의 확률 모델

#### 정의 3.1.1: LLM-as-Judge

**LLM-as-Judge**는 LLM을 평가자로 사용합니다:

$$
\text{Score} = f_{\text{LLM}}(\text{prompt}(p, r, \text{criterion}))
$$

여기서 $f_{\text{LLM}}$은 LLM의 출력을 점수로 변환하는 함수입니다.

#### 시각적 표현: LLM-as-Judge 프로세스

```
┌─────────────────────────────────────────────────────────┐
│              LLM-as-Judge 평가 프로세스                  │
└─────────────────────────────────────────────────────────┘

입력:
- 예측: p = "고양이는 포유동물이다"
- 참조: r = "고양이는 포유동물"
- 기준: criterion = "accuracy"

    │
    ▼
┌─────────────────────────────────────┐
│  Judge 프롬프트 생성                │
│                                     │
│  "Evaluate the following response:  │
│   Prediction: {p}                   │
│   Reference: {r}                    │
│   Criterion: {criterion}             │
│                                     │
│   Score from 0 to 1:                │
│   SCORE: <number>"                  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  LLM 실행                           │
│  GPT-4 / Claude / Gemini            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  출력 파싱                          │
│  "SCORE: 0.95"                      │
│  "EXPLANATION: ..."                 │
└──────────────┬──────────────────────┘
               │
               ▼
    EvaluationResult(score=0.95)
```

**llmkit 구현:**
```python
# evaluation/metrics.py: LLMJudgeMetric
class LLMJudgeMetric(BaseMetric):
    def compute(self, prediction: str, reference: str, **kwargs) -> EvaluationResult:
        client = self._get_client()
        
        prompt = self._create_judge_prompt(
            prediction,
            reference if self.use_reference else None,
            self.criterion
        )
        
        response = client.chat([{"role": "user", "content": prompt}])
        judge_output = response.content
        
        # 점수 추출
        score_match = re.search(r"SCORE:\s*([\d.]+)", judge_output)
        if score_match:
            score = float(score_match.group(1))
        else:
            score = 0.5  # 기본값
        
        # 설명 추출
        explanation_match = re.search(
            r"EXPLANATION:\s*(.+)", 
            judge_output, 
            re.DOTALL
        )
        explanation = (
            explanation_match.group(1).strip() 
            if explanation_match 
            else judge_output
        )
        
        return EvaluationResult(
            metric_name=self.name,
            score=score,
            metadata={"criterion": self.criterion},
            explanation=explanation,
        )
```

---

## Part IV: Human-in-the-Loop 평가

### 4.1 인간 피드백의 통계적 모델

#### 정의 4.1.1: Human Feedback

**인간 피드백**은 다음 튜플로 정의됩니다:

$$
\text{HF} = (id, type, output, rating, comment, timestamp)
$$

**피드백 타입:**
- **Rating**: $r \in [0, 1]$ (평점)
- **Comparison**: $(a, b, \text{winner})$ (비교 평가)
- **Correction**: $\text{corrected\_output}$ (수정 제안)
- **Comment**: $\text{text}$ (자유 텍스트)

**llmkit 구현:**
```python
# evaluation/human_feedback.py: HumanFeedback
@dataclass
class HumanFeedback:
    """
    인간 피드백: HF = (id, type, output, rating, comment, timestamp)
    """
    feedback_id: str
    feedback_type: FeedbackType
    output: str
    rating: Optional[float] = None  # r ∈ [0, 1]
    comment: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
```

---

### 4.2 하이브리드 평가의 가중 평균

#### 정의 4.2.1: Hybrid Evaluator

**하이브리드 평가**는 LLM 평가와 인간 피드백을 결합합니다:

$$
\text{Hybrid Score} = w_h \cdot S_h + w_l \cdot S_l
$$

여기서:
- $S_h$: 인간 피드백 점수
- $S_l$: LLM 평가 점수
- $w_h + w_l = 1$ (가중치 합)

#### 구체적 수치 예시

**예시 4.2.1: 하이브리드 평가 계산**

- **LLM 평가**: $S_l = 0.85$
- **인간 피드백**: $S_h = 0.90$
- **가중치**: $w_l = 0.3$, $w_h = 0.7$

**하이브리드 점수:**
$$
\text{Hybrid} = 0.7 \times 0.90 + 0.3 \times 0.85 = 0.63 + 0.255 = 0.885
$$

**llmkit 구현:**
```python
# domain/evaluation/hybrid_evaluator.py: HybridEvaluator
# domain/evaluation/human_feedback.py: HumanFeedback, HumanFeedbackCollector
class HybridEvaluator:
    """
    하이브리드 평가기: Score = w_h · S_h + w_l · S_l
    
    where:
    - S_h: 인간 피드백 점수
    - S_l: LLM 평가 점수
    - w_h + w_l = 1 (가중치 합)
    """
    def __init__(
        self,
        llm_grader: LLMJudgeMetric,
        feedback_collector: Optional[HumanFeedbackCollector] = None,
        human_weight: float = 0.7,
        llm_weight: float = 0.3,
    ):
        """
        Args:
            llm_grader: LLM 평가 메트릭 (domain/evaluation/metrics.py: LLMJudgeMetric)
            feedback_collector: 피드백 수집기 (domain/evaluation/human_feedback.py)
            human_weight: 인간 피드백 가중치 (기본값: 0.7)
            llm_weight: LLM 평가 가중치 (기본값: 0.3)
        """
        if abs(human_weight + llm_weight - 1.0) > 0.01:
            raise ValueError(
                f"human_weight ({human_weight}) + llm_weight ({llm_weight}) must equal 1.0"
            )
        
        self.llm_grader = llm_grader
        self.feedback_collector = feedback_collector or HumanFeedbackCollector()
        self.human_weight = human_weight
        self.llm_weight = llm_weight
    
    async def evaluate_hybrid(
        self,
        output: str,
        reference: Optional[str] = None,
        human_feedback: Optional[HumanFeedback] = None,
        criteria: Optional[str] = None,
        **kwargs,
    ) -> EvaluationResult:
        """
        하이브리드 평가 실행
        
        Process:
        1. LLM으로 1차 평가: S_l = LLM-Judge(output, reference)
        2. 인간 피드백이 있으면 가중 평균: Score = w_h·S_h + w_l·S_l
        3. 인간 피드백이 없으면 LLM 평가만 사용: Score = S_l
        
        실제 구현:
        - domain/evaluation/hybrid_evaluator.py: HybridEvaluator
        - domain/evaluation/human_feedback.py: HumanFeedback, HumanFeedbackCollector
        """
        # 1. LLM 평가
        llm_result = self.llm_grader.compute(
            prediction=output,
            reference=reference or "",
            criteria=criteria,
            **kwargs,
        )
        
        # 2. 인간 피드백이 없으면 LLM 평가만 반환
        if human_feedback is None:
            return EvaluationResult(
                metric_name="hybrid_evaluation",
                score=llm_result.score,
                metadata={
                    "llm_score": llm_result.score,
                    "human_score": None,
                    "has_human_feedback": False,
                }
            )
        
        # 3. 인간 피드백에서 점수 추출
        human_score = self._extract_score_from_feedback(human_feedback)
        
        # 4. 가중 평균 계산
        hybrid_score = (
            self.human_weight * human_score + 
            self.llm_weight * llm_result.score
        )
        
        return EvaluationResult(
            metric_name="hybrid_evaluation",
            score=hybrid_score,
            metadata={
                "llm_score": llm_result.score,
                "human_score": human_score,
                "human_weight": self.human_weight,
                "llm_weight": self.llm_weight,
                "has_human_feedback": True,
            }
        )
```

---

### 4.3 비교 평가와 Bradley-Terry 모델

#### 정의 4.3.1: Bradley-Terry 모델

**Bradley-Terry 모델**은 비교 평가에서 항목의 강도를 추정합니다:

$$
P(A > B) = \frac{e^{\beta_A}}{e^{\beta_A} + e^{\beta_B}}
$$

여기서 $\beta_A$, $\beta_B$는 각 항목의 강도 파라미터입니다.

**llmkit 구현:**
```python
# evaluation/human_feedback.py: ComparisonFeedback
@dataclass
class ComparisonFeedback(HumanFeedback):
    """
    비교 평가: (output_a, output_b, winner)
    """
    output_a: str
    output_b: str
    winner: ComparisonWinner  # A, B, or TIE
```

---

## Part V: 지속적 평가와 드리프트 감지

### 5.1 Continuous Evaluation의 시간 시리즈 모델

#### 정의 5.1.1: Evaluation Time Series

**평가 시계열**은 다음과 같이 정의됩니다:

$$
S(t) = \{s_1, s_2, \ldots, s_t\}
$$

여기서 $s_i$는 시간 $i$에서의 평가 점수입니다.

#### 정의 5.1.2: Trend Analysis

**트렌드 분석**은 선형 회귀를 사용합니다:

$$
s(t) = \alpha + \beta t + \epsilon
$$

여기서:
- $\alpha$: 절편
- $\beta$: 기울기 (트렌드)
- $\epsilon$: 오차

**트렌드 분류:**
- $\beta > 0$: 개선 (improving)
- $\beta < 0$: 악화 (declining)
- $\beta \approx 0$: 안정 (stable)

**llmkit 구현:**
```python
# domain/evaluation/continuous.py: ContinuousEvaluator, EvaluationTask, EvaluationRun
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

class ContinuousEvaluator:
    """
    지속적 평가 시스템
    
    정기적으로 평가를 실행하고 결과를 추적
    
    시간 시리즈 모델: S(t) = {s_1, s_2, ..., s_t}
    where s_i는 시간 i에서의 평가 점수
    """
    def __init__(self, storage_path: Optional[str] = None):
        """
        Args:
            storage_path: 결과 저장 경로 (선택적)
        """
        self.storage_path = storage_path
        self._tasks: Dict[str, EvaluationTask] = {}
        self._runs: List[EvaluationRun] = []
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._run_counter = 0
    
    def add_task(
        self,
        task_id: str,
        name: str,
        evaluator: Evaluator,
        test_cases: List[Dict[str, Any]],
        schedule: Optional[str] = None,  # Cron 표현식
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluationTask:
        """
        평가 작업 추가
        
        Args:
            task_id: 작업 ID
            name: 작업 이름
            evaluator: 평가기 (domain/evaluation/evaluator.py: Evaluator)
            test_cases: 테스트 케이스 리스트
            schedule: Cron 표현식 (예: "0 9 * * *" = 매일 9시)
            metadata: 추가 메타데이터
        
        Cron 표현식 예시:
        - "0 9 * * *": 매일 9시
        - "0 */6 * * *": 6시간마다
        - "0 0 * * 1": 매주 월요일 자정
        """
        task = EvaluationTask(
            task_id=task_id,
            name=name,
            evaluator=evaluator,
            test_cases=test_cases,
            schedule=schedule,
            metadata=metadata or {},
        )
        self._tasks[task_id] = task
        
        if schedule:
            self._schedule_task(task)
        
        return task
    
    async def run_task(self, task_id: str) -> EvaluationRun:
        """
        평가 작업 실행
        
        Process:
        1. 각 테스트 케이스에 대해 평가 실행
        2. 평균 점수 계산: μ = (1/n) Σ s_i
        3. 결과 저장 및 추적
        
        Returns:
            EvaluationRun: 평가 실행 결과
        
        실제 구현:
        - domain/evaluation/continuous.py: ContinuousEvaluator
        - apscheduler를 사용한 스케줄링 (선택적 의존성)
        """
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        if not task.enabled:
            raise ValueError(f"Task {task_id} is disabled")
        
        results = []
        for test_case in task.test_cases:
            prediction = test_case.get("prediction", "")
            reference = test_case.get("reference", "")
            kwargs = {k: v for k, v in test_case.items() if k not in ["prediction", "reference"]}
            
            result = task.evaluator.evaluate(prediction, reference, **kwargs)
            results.append(result)
        
        # 평균 점수 계산
        if results:
            all_scores = []
            for result in results:
                all_scores.append(result.average_score)
            average_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        else:
            average_score = 0.0
        
        # 실행 결과 생성
        run_id = f"run_{self._run_counter}"
        self._run_counter += 1
        
        run = EvaluationRun(
            run_id=run_id,
            task_id=task_id,
            timestamp=datetime.now(),
            results=results,
            average_score=average_score,
            metadata={
                "task_name": task.name,
                "test_cases_count": len(task.test_cases),
                "results_count": len(results),
            },
        )
        
        self._runs.append(run)
        self._save_if_needed()
        
        return run
```

---

### 5.2 Drift Detection의 통계적 검정

#### 정의 5.2.1: Performance Drift

**성능 드리프트**는 평가 점수의 통계적으로 유의미한 변화입니다:

$$
\text{Drift} = \begin{cases}
\text{True} & \text{if } |\mu_{\text{current}} - \mu_{\text{baseline}}| \geq \theta_{\text{std}} \cdot \sigma_{\text{baseline}} \\
\text{False} & \text{otherwise}
\end{cases}
$$

여기서:
- $\mu_{\text{baseline}}$: 기준선 평균
- $\mu_{\text{current}}$: 현재 평균
- $\sigma_{\text{baseline}}$: 기준선 표준편차
- $\theta_{\text{std}}$: 표준편차 임계값 (기본값: 2.0 = 2σ)

#### 정의 5.2.2: Z-Score Test

**Z-Score 검정:**

$$
z = \frac{|\mu_{\text{current}} - \mu_{\text{baseline}}|}{\sigma_{\text{baseline}}}
$$

**드리프트 판정:**
- $z \geq 2.0$: 드리프트 감지 (95% 신뢰도)
- $z \geq 3.0$: 강한 드리프트 (99.7% 신뢰도)

#### 구체적 수치 예시

**예시 5.2.1: Drift Detection 계산**

**기준선 (7일간):**
- 점수: $[0.85, 0.87, 0.86, 0.88, 0.85, 0.86, 0.87]$
- $\mu_{\text{baseline}} = 0.863$
- $\sigma_{\text{baseline}} = 0.011$

**현재 점수:**
- $\mu_{\text{current}} = 0.75$

**Z-Score 계산:**
$$
z = \frac{|0.75 - 0.863|}{0.011} = \frac{0.113}{0.011} \approx 10.27
$$

**결과:**
- $z = 10.27 \geq 2.0$ → **드리프트 감지** ✓
- 심각도: **Critical** (10σ 이상)

**llmkit 구현:**
```python
# domain/evaluation/drift_detection.py: DriftDetector, DriftAlert
import statistics
from datetime import datetime, timedelta

class DriftDetector:
    """
    모델 드리프트 감지기
    
    Z-Score 검정: z = |μ_current - μ_baseline| / σ_baseline
    
    where:
    - μ_baseline: 기준선 평균 (baseline_window_days 기간)
    - μ_current: 현재 평균
    - σ_baseline: 기준선 표준편차
    - threshold_std: 표준편차 임계값 (기본값: 2.0 = 2σ, 95% 신뢰도)
    """
    def __init__(
        self,
        baseline_window_days: int = 7,
        detection_window_days: int = 1,
        threshold_std: float = 2.0,  # 2σ (95% 신뢰도)
        threshold_percent: float = 0.2,  # 20% 변화
    ):
        """
        Args:
            baseline_window_days: 기준선 계산 기간 (일)
            detection_window_days: 감지 기간 (일)
            threshold_std: 표준편차 임계값 (기본값: 2.0 = 2σ)
            threshold_percent: 백분율 변화 임계값 (기본값: 0.2 = 20%)
        """
        self.baseline_window_days = baseline_window_days
        self.detection_window_days = detection_window_days
        self.threshold_std = threshold_std
        self.threshold_percent = threshold_percent
        self._history: List[Dict[str, Any]] = []
        self._alert_counter = 0
    
    def record_score(
        self,
        metric_name: str,
        score: float,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """점수 기록"""
        self._history.append({
            "timestamp": timestamp or datetime.now(),
            "metric_name": metric_name,
            "score": score,
            "metadata": metadata or {},
        })
    
    def detect_drift(
        self,
        metric_name: Optional[str] = None,
        current_score: Optional[float] = None,
    ) -> List[DriftAlert]:
        """
        드리프트 감지
        
        Z-Score 검정: z = |μ_current - μ_baseline| / σ_baseline
        
        실제 구현:
        - domain/evaluation/drift_detection.py: DriftDetector
        - 통계적 검정 (Z-Score, 분포 변화 감지)
        """
        alerts = []
        metrics_to_check = [metric_name] if metric_name else self._get_all_metrics()
        
        for metric in metrics_to_check:
            metric_alerts = self._detect_drift_for_metric(metric, current_score)
            alerts.extend(metric_alerts)
        
        return alerts
    
    def _detect_drift_for_metric(
        self,
        metric_name: str,
        current_score: Optional[float] = None,
    ) -> List[DriftAlert]:
        """특정 메트릭에 대한 드리프트 감지"""
        metric_history = [
            h for h in self._history 
            if h["metric_name"] == metric_name
        ]
        
        if len(metric_history) < 2:
            return []
        
        # 현재 점수 결정
        if current_score is None:
            current_score = metric_history[-1]["score"]
        
        # 기준선 계산
        cutoff_date = datetime.now() - timedelta(days=self.baseline_window_days)
        baseline_scores = [
            h["score"] for h in metric_history 
            if h["timestamp"] >= cutoff_date
        ]
        
        if len(baseline_scores) < 2:
            return []
        
        # 기준선 통계
        baseline_mean = statistics.mean(baseline_scores)
        baseline_std = statistics.stdev(baseline_scores) if len(baseline_scores) > 1 else 0.0
        
        alerts = []
        
        # 1. 성능 저하 감지 (Z-Score 검정)
        score_diff = current_score - baseline_mean
        percent_change = abs(score_diff / baseline_mean) if baseline_mean != 0 else 0.0
        
        if score_diff < 0 and percent_change >= self.threshold_percent:
            if baseline_std > 0:
                z_score = abs(score_diff) / baseline_std
                
                if z_score >= self.threshold_std:
                    severity = self._calculate_severity(percent_change, z_score)
                    alerts.append(
                        DriftAlert(
                            alert_id=f"drift_{self._alert_counter}",
                            metric_name=metric_name,
                            timestamp=datetime.now(),
                            current_score=current_score,
                            baseline_score=baseline_mean,
                            drift_magnitude=abs(score_diff),
                            drift_type="performance_degradation",
                            severity=severity,
                            metadata={
                                "percent_change": percent_change,
                                "z_score": z_score,
                                "baseline_std": baseline_std,
                            },
                        )
                    )
                    self._alert_counter += 1
        
        # 2. 분포 변화 감지 (변동성 증가)
        if len(baseline_scores) >= 5:
            recent_scores = [h["score"] for h in metric_history[-5:]]
            recent_std = statistics.stdev(recent_scores) if len(recent_scores) > 1 else 0.0
            
            if baseline_std > 0 and recent_std > baseline_std * 1.5:
                alerts.append(
                    DriftAlert(
                        alert_id=f"drift_{self._alert_counter}",
                        metric_name=metric_name,
                        timestamp=datetime.now(),
                        current_score=current_score,
                        baseline_score=baseline_mean,
                        drift_magnitude=recent_std - baseline_std,
                        drift_type="distribution_shift",
                        severity="medium",
                        metadata={
                            "baseline_std": baseline_std,
                            "recent_std": recent_std,
                        },
                    )
                )
                self._alert_counter += 1
        
        return alerts
```

---

### 5.3 트렌드 분석과 상관관계

#### 정의 5.3.1: Pearson Correlation

**피어슨 상관계수**는 두 메트릭 간의 선형 관계를 측정합니다:

$$
r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}
$$

**해석:**
- $r \in [-1, 1]$
- $|r| > 0.7$: 강한 상관관계
- $|r| \in [0.3, 0.7]$: 중간 상관관계
- $|r| < 0.3$: 약한 상관관계

**llmkit 구현:**
```python
# domain/evaluation/analytics.py: EvaluationAnalyticsEngine, CorrelationAnalysis
import statistics
from datetime import datetime, timedelta

class EvaluationAnalyticsEngine:
    """
    평가 분석 엔진
    
    평가 결과를 분석하여 트렌드, 상관관계, 인사이트 제공
    """
    def __init__(self):
        self._history: List[Dict[str, Any]] = []
    
    def add_evaluation_result(
        self,
        result: BatchEvaluationResult,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """평가 결과 추가"""
        self._history.append({
            "timestamp": timestamp or datetime.now(),
            "result": result,
            "metadata": metadata or {},
        })
    
    def analyze_correlations(
        self,
        metric_a: str,
        metric_b: str,
        window_days: int = 30,
    ) -> CorrelationAnalysis:
        """
        상관관계 분석
        
        Pearson Correlation: 
        r = Σ(x_i - x̄)(y_i - ȳ) / (σ_x · σ_y)
        
        where:
        - x_i, y_i: 메트릭 A, B의 점수
        - x̄, ȳ: 평균
        - σ_x, σ_y: 표준편차
        
        해석:
        - |r| > 0.7: 강한 상관관계
        - |r| ∈ [0.3, 0.7]: 중간 상관관계
        - |r| < 0.3: 약한 상관관계
        
        실제 구현:
        - domain/evaluation/analytics.py: EvaluationAnalyticsEngine
        """
        cutoff_date = datetime.now() - timedelta(days=window_days)
        recent_history = [
            h for h in self._history 
            if h["timestamp"] >= cutoff_date
        ]
        
        scores_a = []
        scores_b = []
        
        for entry in recent_history:
            result = entry["result"]
            score_a = None
            score_b = None
            
            for r in result.results:
                if r.metric_name == metric_a:
                    score_a = r.score
                if r.metric_name == metric_b:
                    score_b = r.score
            
            if score_a is not None and score_b is not None:
                scores_a.append(score_a)
                scores_b.append(score_b)
        
        if len(scores_a) < 2:
            return CorrelationAnalysis(
                metric_a=metric_a,
                metric_b=metric_b,
                correlation=0.0,
                significance="none"
            )
        
        # Pearson Correlation 계산
        # r = Σ(x_i - x̄)(y_i - ȳ) / (σ_x · σ_y)
        mean_a = statistics.mean(scores_a)
        mean_b = statistics.mean(scores_b)
        
        numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(scores_a, scores_b))
        std_a = statistics.stdev(scores_a) if len(scores_a) > 1 else 1.0
        std_b = statistics.stdev(scores_b) if len(scores_b) > 1 else 1.0
        
        correlation = numerator / (len(scores_a) * std_a * std_b) if std_a > 0 and std_b > 0 else 0.0
        
        # 유의도 판정
        if abs(correlation) > 0.7:
            significance = "strong"
        elif abs(correlation) > 0.3:
            significance = "moderate"
        elif abs(correlation) > 0.1:
            significance = "weak"
        else:
            significance = "none"
        
        return CorrelationAnalysis(
            metric_a=metric_a,
            metric_b=metric_b,
            correlation=correlation,
            significance=significance
        )
```

---

## Part VI: 구조화된 평가

### 6.1 Rubric-Driven Grading의 가중 합

#### 정의 6.1.1: Rubric

**루브릭**은 구조화된 평가 기준입니다:

$$
\text{Rubric} = \{C_1, C_2, \ldots, C_n\}
$$

여기서 각 기준 $C_i$는 다음 튜플입니다:

$$
C_i = (\text{name}, \text{description}, w_i, L_i)
$$

- $w_i$: 가중치
- $L_i$: 레벨 집합 (예: $\{$"excellent": 1.0, "good": 0.8, ...$\}$)

#### 정의 6.1.2: Rubric Score

**루브릭 점수**는 가중 합으로 계산됩니다:

$$
\text{Rubric Score} = \sum_{i=1}^{n} w_i \cdot s_i
$$

여기서 $s_i$는 기준 $i$에 대한 점수입니다.

**가중치 정규화:**

$$
\sum_{i=1}^{n} w_i = 1
$$

#### 구체적 수치 예시

**예시 6.1.1: Rubric-Driven Grading 계산**

**루브릭:**
- $C_1$: "정확성" (가중치: 0.4)
- $C_2$: "완전성" (가중치: 0.3)
- $C_3$: "명확성" (가중치: 0.3)

**평가 결과:**
- $s_1 = 0.9$ (정확성: "good")
- $s_2 = 1.0$ (완전성: "excellent")
- $s_3 = 0.8$ (명확성: "good")

**최종 점수:**
$$
\text{Rubric Score} = 0.4 \times 0.9 + 0.3 \times 1.0 + 0.3 \times 0.8 = 0.36 + 0.30 + 0.24 = 0.90
$$

**llmkit 구현:**
```python
# domain/evaluation/rubric.py: RubricGrader, Rubric, RubricCriterion
class RubricGrader(BaseMetric):
    """
    루브릭 기반 평가기
    
    Rubric Score = Σ_{i=1}^{n} w_i · s_i
    
    where:
    - w_i: 기준 i의 가중치 (정규화: Σ w_i = 1)
    - s_i: 기준 i에 대한 점수 (0.0 ~ 1.0)
    """
    def __init__(
        self,
        rubric: Rubric,
        client=None,
        use_llm: bool = True,
    ):
        """
        Args:
            rubric: 평가 루브릭 (domain/evaluation/rubric.py: Rubric)
            client: LLM 클라이언트 (use_llm=True일 때 필요)
            use_llm: LLM을 사용하여 평가할지 여부
        """
        super().__init__(f"rubric_{rubric.name}", MetricType.QUALITY)
        self.rubric = rubric
        self.client = client
        self.use_llm = use_llm
    
    def compute(
        self,
        prediction: str,
        reference: Optional[str] = None,
        **kwargs,
    ) -> EvaluationResult:
        """
        루브릭 평가 실행
        
        Process:
        1. 각 기준에 대해 점수 계산 (LLM 또는 수동)
        2. 가중 합 계산: Score = Σ w_i · s_i
        
        실제 구현:
        - domain/evaluation/rubric.py: RubricGrader
        - LLM 기반 평가 또는 수동 점수 입력 지원
        """
        if self.use_llm:
            scores = self._llm_grade(prediction, reference)
        else:
            # 수동 평가 (점수는 kwargs에서 제공)
            scores = kwargs.get("manual_scores", {})
        
        # 가중 합 계산
        total_score = 0.0
        criterion_scores = {}
        
        for criterion in self.rubric.criteria:
            score = scores.get(criterion.name, 0.0)
            weighted_score = criterion.weight * score
            total_score += weighted_score
            criterion_scores[criterion.name] = {
                "raw_score": score,
                "weight": criterion.weight,
                "weighted_score": weighted_score,
            }
        
        return EvaluationResult(
            metric_name=self.name,
            score=total_score,
            metadata={
                "criterion_scores": criterion_scores,
                "rubric_name": self.rubric.name,
                "criteria_count": len(self.rubric.criteria),
            }
        )
```

---

### 6.2 CheckEval의 Boolean 평가 모델

#### 정의 6.2.1: Checklist

**체크리스트**는 Boolean 질문 집합입니다:

$$
\text{Checklist} = \{Q_1, Q_2, \ldots, Q_n\}
$$

각 질문 $Q_i$는 다음 튜플입니다:

$$
Q_i = (\text{question}, w_i, \text{required})
$$

#### 정의 6.2.2: Checklist Score

**체크리스트 점수**는 가중 Boolean 합입니다:

$$
\text{Checklist Score} = \frac{\sum_{i=1}^{n} w_i \cdot \mathbf{1}(Q_i)}{\sum_{i=1}^{n} w_i}
$$

여기서 $\mathbf{1}(Q_i)$는 질문 $Q_i$에 대한 답변이 "예"이면 1, 아니면 0입니다.

**필수 항목 검증:**

$$
\text{Pass} = \begin{cases}
\text{True} & \text{if } \forall Q_i \in \text{required}: \mathbf{1}(Q_i) = 1 \\
\text{False} & \text{otherwise}
\end{cases}
$$

#### 구체적 수치 예시

**예시 6.2.1: CheckEval 계산**

**체크리스트:**
- $Q_1$: "답변이 질문에 직접적으로 답하는가?" (가중치: 0.3, 필수: ✓)
- $Q_2$: "답변이 컨텍스트를 참조하는가?" (가중치: 0.2, 필수: ✗)
- $Q_3$: "답변이 사실적으로 정확한가?" (가중치: 0.3, 필수: ✓)
- $Q_4$: "답변이 완전한가?" (가중치: 0.2, 필수: ✗)

**평가 결과:**
- $Q_1$: 예 (1) ✓
- $Q_2$: 아니오 (0) ✗
- $Q_3$: 예 (1) ✓
- $Q_4$: 예 (1) ✓

**점수 계산:**
$$
\text{Score} = \frac{0.3 \times 1 + 0.2 \times 0 + 0.3 \times 1 + 0.2 \times 1}{0.3 + 0.2 + 0.3 + 0.2} = \frac{0.8}{1.0} = 0.8
$$

**필수 항목 검증:**
- $Q_1$: ✓, $Q_3$: ✓ → **Pass** ✓

**llmkit 구현:**
```python
# domain/evaluation/checklist.py: ChecklistGrader, Checklist, ChecklistItem
class ChecklistGrader(BaseMetric):
    """
    체크리스트 기반 평가기
    
    Checklist Score = Σ w_i · 1(Q_i) / Σ w_i
    
    where:
    - Q_i: 체크리스트 항목 i
    - 1(Q_i): Boolean 함수 (예=1, 아니오=0)
    - w_i: 항목 i의 가중치
    """
    def __init__(
        self,
        checklist: Checklist,
        client=None,
        use_llm: bool = True,
    ):
        """
        Args:
            checklist: 평가 체크리스트 (domain/evaluation/checklist.py: Checklist)
            client: LLM 클라이언트 (use_llm=True일 때 필요)
            use_llm: LLM을 사용하여 평가할지 여부
        """
        super().__init__(f"checklist_{checklist.name}", MetricType.QUALITY)
        self.checklist = checklist
        self.client = client
        self.use_llm = use_llm
    
    def compute(
        self,
        prediction: str,
        reference: Optional[str] = None,
        **kwargs,
    ) -> EvaluationResult:
        """
        체크리스트 평가 실행
        
        Process:
        1. 각 항목에 대해 Boolean 평가 (LLM 또는 수동)
        2. 필수 항목 검증: Pass = ∀ Q_i ∈ required: 1(Q_i) = 1
        3. 가중 합 계산: Score = Σ w_i · 1(Q_i) / Σ w_i
        
        실제 구현:
        - domain/evaluation/checklist.py: ChecklistGrader
        - Boolean 질문 기반 평가로 명확하고 신뢰성 높은 평가 제공
        """
        if self.use_llm:
            answers = self._llm_check(prediction, reference)
        else:
            # 수동 평가
            answers = kwargs.get("manual_answers", {})
        
        # 필수 항목 검증
        required_items = [
            item for item in self.checklist.items 
            if item.required
        ]
        
        all_required_passed = all(
            answers.get(item.question, False) 
            for item in required_items
        )
        
        # 점수 계산 (가중 Boolean 합)
        total_weight = 0.0
        weighted_sum = 0.0
        
        for item in self.checklist.items:
            answer = answers.get(item.question, False)
            weighted_sum += item.weight * (1.0 if answer else 0.0)
            total_weight += item.weight
        
        score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return EvaluationResult(
            metric_name=self.name,
            score=score,
            metadata={
                "all_required_passed": all_required_passed,
                "answers": answers,
                "checklist_name": self.checklist.name,
                "total_items": len(self.checklist.items),
                "required_items_count": len(required_items),
            }
        )
```

---

## 참고 문헌

1. **Papineni et al. (2002)**: "BLEU: a method for automatic evaluation of machine translation" - BLEU Score
2. **Lin (2004)**: "ROUGE: A Package for Automatic Evaluation of Summaries" - ROUGE Score
3. **Zheng et al. (2023)**: "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" - LLM-as-Judge
4. **Bradley & Terry (1952)**: "Rank analysis of incomplete block designs" - Bradley-Terry 모델
5. **Gustafson et al. (2023)**: "Measuring and Improving Faithfulness in RAG" - Faithfulness Metric

---

**작성일**: 2025-01-XX  
**버전**: 1.0 (석사 수준 이론 문서)

