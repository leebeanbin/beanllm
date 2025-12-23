# Euclidean Distance and Norms: 유클리드 거리와 Norm의 완전한 분석

**석사 수준 이론 문서**  
**CS 관점 + 수학적 엄밀성**  
**기반**: llmkit 실제 구현 코드 분석

---

## 목차

1. [Norm의 정의와 종류](#1-norm의-정의와-종류)
2. [L2 Norm (유클리드 Norm)](#2-l2-norm-유클리드-norm)
3. [유클리드 거리](#3-유클리드-거리)
4. [거리 공간 (Metric Space)](#4-거리-공간-metric-space)
5. [Norm과 거리의 관계](#5-norm과-거리의-관계)
6. [고차원 공간에서의 거리](#6-고차원-공간에서의-거리)
7. [CS 관점: 거리 계산 알고리즘](#7-cs-관점-거리-계산-알고리즘)
8. [수치 안정성과 오차 분석](#8-수치-안정성과-오차-분석)
9. [코사인 유사도와의 관계](#9-코사인-유사도와의-관계)

---

## 1. Norm의 정의와 종류

### 1.1 Norm의 수학적 정의

#### 정의 1.1.1: Norm (노름)

**Norm**은 벡터 공간 $V$에서 다음을 만족하는 함수입니다:

$$
\|\cdot\|: V \rightarrow \mathbb{R}_{\geq 0}
$$

**Norm 공리:**

1. **양의 정부호 (Positive Definiteness):**
   $$
   \|\mathbf{v}\| \geq 0 \quad \text{and} \quad \|\mathbf{v}\| = 0 \iff \mathbf{v} = \mathbf{0}
   $$

2. **동질성 (Homogeneity):**
   $$
   \|c\mathbf{v}\| = |c| \|\mathbf{v}\| \quad \forall c \in \mathbb{F}, \mathbf{v} \in V
   $$

3. **삼각 부등식 (Triangle Inequality):**
   $$
   \|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\| \quad \forall \mathbf{u}, \mathbf{v} \in V
   $$

#### 정리 1.1.1: Norm의 기본 성질

**다음이 성립합니다:**

1. $\|-\mathbf{v}\| = \|\mathbf{v}\|$
2. $|\|\mathbf{u}\| - \|\mathbf{v}\|| \leq \|\mathbf{u} - \mathbf{v}\|$ (역삼각 부등식)

**증명 1:**
$$
\|-\mathbf{v}\| = \|(-1)\mathbf{v}\| = |-1| \|\mathbf{v}\| = \|\mathbf{v}\|
$$

**증명 2:**
$$
\|\mathbf{u}\| = \|(\mathbf{u} - \mathbf{v}) + \mathbf{v}\| \leq \|\mathbf{u} - \mathbf{v}\| + \|\mathbf{v}\|
$$

따라서:
$$
\|\mathbf{u}\| - \|\mathbf{v}\| \leq \|\mathbf{u} - \mathbf{v}\|
$$

비슷하게:
$$
\|\mathbf{v}\| - \|\mathbf{u}\| \leq \|\mathbf{v} - \mathbf{u}\| = \|\mathbf{u} - \mathbf{v}\|
$$

따라서:
$$
|\|\mathbf{u}\| - \|\mathbf{v}\|| \leq \|\mathbf{u} - \mathbf{v}\|
$$

□

### 1.2 다양한 Norm

#### 정의 1.2.1: Lp Norm

**Lp Norm**은 다음과 같이 정의됩니다:

$$
\|\mathbf{v}\|_p = \left(\sum_{i=1}^d |v_i|^p\right)^{1/p}
$$

**특수한 경우:**

1. **L1 Norm (Manhattan Distance):**
   $$
   \|\mathbf{v}\|_1 = \sum_{i=1}^d |v_i|
   $$

2. **L2 Norm (Euclidean Norm):**
   $$
   \|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^d v_i^2}
   $$

3. **L∞ Norm (Chebyshev Distance):**
   $$
   \|\mathbf{v}\|_\infty = \max_{i=1,\ldots,d} |v_i|
   $$

#### 시각적 표현: 다양한 Norm의 단위 원

```
L1 Norm (다이아몬드):        L2 Norm (원):            L∞ Norm (사각형):
        y                          y                        y
        ↑                          ↑                        ↑
        |     ╱╲                  |     ╱─╲                |     ───
        |    ╱  ╲                 |    ╱   ╲               |    │   │
        |   ╱    ╲                |   ╱     ╲              |   │     │
        |  ╱      ╲               |  ╱       ╲             |  │       │
        | ╱        ╲              | ╱         ╲            | │         │
        |╱__________╲→ x          |╱___________╲→ x        |│_________│→ x
```

---

## 2. L2 Norm (유클리드 Norm)

### 2.1 L2 Norm의 정의

#### 정의 2.1.1: L2 Norm (유클리드 Norm)

**L2 Norm**은 다음과 같이 정의됩니다:

$$
\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^d v_i^2} = \sqrt{\mathbf{v} \cdot \mathbf{v}}
$$

**시각적 표현: 2D 공간**

```
        y
        ↑
        |     v (v₁, v₂)
        |    /
        |   /|
        |  / |
        | /  | v₂
        |/___|__→ x
       0  v₁
        
||v||₂ = √(v₁² + v₂²)
      = √(v₁² + v₂²)  (피타고라스 정리)
```

#### 구체적 수치 예시

**예시 2.1.1: L2 Norm 계산**

$\mathbf{v} = [3, 4, 0, 12]$:

**단계별 계산:**

1. **각 성분 제곱:**
   $$
   3^2 = 9, \quad 4^2 = 16, \quad 0^2 = 0, \quad 12^2 = 144
   $$

2. **합계:**
   $$
   9 + 16 + 0 + 144 = 169
   $$

3. **제곱근:**
   $$
   \|\mathbf{v}\|_2 = \sqrt{169} = 13
   $$

**시각적 표현: 4D 공간에서의 길이**

```
4D 공간 (투영):
v = [3, 4, 0, 12]

||v||₂ = √(3² + 4² + 0² + 12²)
      = √(9 + 16 + 0 + 144)
      = √169
      = 13
```

### 2.2 L2 Norm의 성질

#### 정리 2.2.1: L2 Norm과 내적의 관계

**L2 Norm은 내적으로 표현됩니다:**

$$
\|\mathbf{v}\|_2^2 = \mathbf{v} \cdot \mathbf{v}
$$

**증명:**
$$
\|\mathbf{v}\|_2^2 = \left(\sqrt{\sum_{i=1}^d v_i^2}\right)^2 = \sum_{i=1}^d v_i^2 = \mathbf{v} \cdot \mathbf{v}
$$

□

#### 정리 2.2.2: 평행사변형 법칙 (Parallelogram Law)

**다음이 성립합니다:**

$$
\|\mathbf{u} + \mathbf{v}\|_2^2 + \|\mathbf{u} - \mathbf{v}\|_2^2 = 2(\|\mathbf{u}\|_2^2 + \|\mathbf{v}\|_2^2)
$$

**증명:**
$$
\|\mathbf{u} + \mathbf{v}\|_2^2 = (\mathbf{u} + \mathbf{v}) \cdot (\mathbf{u} + \mathbf{v}) = \|\mathbf{u}\|_2^2 + 2\mathbf{u} \cdot \mathbf{v} + \|\mathbf{v}\|_2^2
$$

$$
\|\mathbf{u} - \mathbf{v}\|_2^2 = (\mathbf{u} - \mathbf{v}) \cdot (\mathbf{u} - \mathbf{v}) = \|\mathbf{u}\|_2^2 - 2\mathbf{u} \cdot \mathbf{v} + \|\mathbf{v}\|_2^2
$$

따라서:
$$
\|\mathbf{u} + \mathbf{v}\|_2^2 + \|\mathbf{u} - \mathbf{v}\|_2^2 = 2(\|\mathbf{u}\|_2^2 + \|\mathbf{v}\|_2^2)
$$

□

---

## 3. 유클리드 거리

### 3.1 유클리드 거리의 정의

#### 정의 3.1.1: 유클리드 거리 (Euclidean Distance)

**유클리드 거리**는 L2 Norm의 차이입니다:

$$
d_{\text{euc}}(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{\sum_{i=1}^d (u_i - v_i)^2}
$$

#### 시각적 표현: 2D 공간

```
        y
        ↑
        |     v (v₁, v₂)
        |    /
        |   /|
        |  / | d
        | /  |
        |/___|__→ x
       u (u₁, u₂)
       
d = ||u - v||₂
  = √((u₁-v₁)² + (u₂-v₂)²)
```

#### 구체적 수치 예시

**예시 3.1.1: 유클리드 거리 계산**

$\mathbf{u} = [1, 2, 3]$, $\mathbf{v} = [4, 6, 8]$:

**단계별 계산:**

1. **차이 벡터:**
   $$
   \mathbf{u} - \mathbf{v} = [1-4, 2-6, 3-8] = [-3, -4, -5]
   $$

2. **제곱:**
   $$
   (-3)^2 = 9, \quad (-4)^2 = 16, \quad (-5)^2 = 25
   $$

3. **합계:**
   $$
   9 + 16 + 25 = 50
   $$

4. **제곱근:**
   $$
   d_{\text{euc}}(\mathbf{u}, \mathbf{v}) = \sqrt{50} \approx 7.071
   $$

### 3.2 유클리드 거리의 성질

#### 정리 3.2.1: 거리 함수의 공리

**유클리드 거리는 거리 함수의 공리를 만족합니다:**

1. **비음성:**
   $$
   d_{\text{euc}}(\mathbf{u}, \mathbf{v}) \geq 0
   $$

2. **구별성:**
   $$
   d_{\text{euc}}(\mathbf{u}, \mathbf{v}) = 0 \iff \mathbf{u} = \mathbf{v}
   $$

3. **대칭성:**
   $$
   d_{\text{euc}}(\mathbf{u}, \mathbf{v}) = d_{\text{euc}}(\mathbf{v}, \mathbf{u})
   $$

4. **삼각 부등식:**
   $$
   d_{\text{euc}}(\mathbf{u}, \mathbf{w}) \leq d_{\text{euc}}(\mathbf{u}, \mathbf{v}) + d_{\text{euc}}(\mathbf{v}, \mathbf{w})
   $$

**증명 4 (삼각 부등식):**
$$
d_{\text{euc}}(\mathbf{u}, \mathbf{w}) = \|\mathbf{u} - \mathbf{w}\|_2 = \|(\mathbf{u} - \mathbf{v}) + (\mathbf{v} - \mathbf{w})\|_2
$$

$$
\leq \|\mathbf{u} - \mathbf{v}\|_2 + \|\mathbf{v} - \mathbf{w}\|_2 = d_{\text{euc}}(\mathbf{u}, \mathbf{v}) + d_{\text{euc}}(\mathbf{v}, \mathbf{w})
$$

□

---

## 4. 거리 공간 (Metric Space)

### 4.1 거리 공간의 정의

#### 정의 4.1.1: 거리 공간 (Metric Space)

**거리 공간** $(M, d)$는 다음을 만족합니다:

1. **비음성:**
   $$
   d(x, y) \geq 0
   $$

2. **구별성:**
   $$
   d(x, y) = 0 \iff x = y
   $$

3. **대칭성:**
   $$
   d(x, y) = d(y, x)
   $$

4. **삼각 부등식:**
   $$
   d(x, z) \leq d(x, y) + d(y, z)
   $$

#### 정리 4.1.1: Norm으로부터 거리 생성

**Norm이 주어지면 거리를 정의할 수 있습니다:**

$$
d(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|
$$

**증명:**
Norm의 공리로부터 거리 공리의 모든 조건이 만족됩니다. □

---

## 5. Norm과 거리의 관계

### 5.1 Norm으로부터 거리 생성

#### 정리 5.1.1: Norm으로부터 거리

**Norm $\|\cdot\|$이 주어지면:**

$$
d(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|
$$

는 거리 함수입니다.

**증명:**
1. **비음성:** $\|\mathbf{u} - \mathbf{v}\| \geq 0$ (Norm 공리)
2. **구별성:** $\|\mathbf{u} - \mathbf{v}\| = 0 \iff \mathbf{u} - \mathbf{v} = \mathbf{0} \iff \mathbf{u} = \mathbf{v}$
3. **대칭성:** $\|\mathbf{u} - \mathbf{v}\| = \|(-1)(\mathbf{v} - \mathbf{u})\| = |(-1)|\|\mathbf{v} - \mathbf{u}\| = \|\mathbf{v} - \mathbf{u}\|$
4. **삼각 부등식:** Norm의 삼각 부등식으로부터

□

### 5.2 거리로부터 Norm 생성

#### 정리 5.2.1: 거리로부터 Norm

**거리 $d$가 주어지고 $\mathbf{0}$이 존재하면:**

$$
\|\mathbf{v}\| = d(\mathbf{0}, \mathbf{v})
$$

는 Norm입니다 (일반적으로는 아님, 추가 조건 필요).

---

## 6. 고차원 공간에서의 거리

### 6.1 고차원 공간의 기하학

#### 정리 6.1.1: 고차원에서의 거리 분포

**고차원 공간 $\mathbb{R}^d$에서 두 랜덤 벡터의 거리는:**

$$
E[d_{\text{euc}}(\mathbf{u}, \mathbf{v})] \propto \sqrt{d}
$$

**해석:** 차원이 높을수록 평균 거리가 증가합니다.

#### 시각적 표현: 차원에 따른 거리 분포

```
차원 d에 따른 평균 거리:

d=2:   평균 거리 ≈ 1.4
d=10:  평균 거리 ≈ 3.2
d=100: 평균 거리 ≈ 10.0
d→∞:  평균 거리 → ∞

→ 고차원에서는 모든 점이 멀리 떨어져 있음
```

### 6.2 차원의 저주 (Curse of Dimensionality)

#### 정의 6.2.1: 차원의 저주

**차원이 높을수록:**

1. **거리 분산 감소:**
   - 모든 거리가 비슷해짐
   - 구별이 어려워짐

2. **데이터 희소성:**
   - 공간이 거대해짐
   - 데이터 포인트가 희소해짐

3. **계산 복잡도 증가:**
   - $O(d)$ 시간 복잡도
   - 메모리 사용량 증가

---

## 7. CS 관점: 거리 계산 알고리즘

### 7.1 유클리드 거리 계산 알고리즘

#### 알고리즘 7.1.1: Naive 유클리드 거리

```
Algorithm: EuclideanDistance(u, v)
Input: 벡터 u, v ∈ ℝ^d
Output: 유클리드 거리 d

1. sum_squared_diff ← 0
2. for i = 1 to d:
3.     diff ← u[i] - v[i]
4.     sum_squared_diff ← sum_squared_diff + diff²
5. return sqrt(sum_squared_diff)
```

**시간 복잡도:** $O(d)$  
**공간 복잡도:** $O(1)$

#### 알고리즘 7.1.2: 최적화된 유클리드 거리

```
Algorithm: EuclideanDistanceOptimized(u, v)
Input: 벡터 u, v ∈ ℝ^d
Output: 유클리드 거리 d

1. diff_vector ← u - v  // 벡터화 연산
2. return ||diff_vector||₂  // L2 norm 계산
```

**llmkit 구현:**
```python
# domain/embeddings/utils.py: euclidean_distance()
import numpy as np
from typing import List

def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    """
    유클리드 거리 계산: d(u, v) = ||u - v||₂
    
    수학적 표현:
    - 입력: 벡터 u, v ∈ ℝ^d
    - 출력: 거리 d = √(Σ(u_i - v_i)²)
    
    시간 복잡도: O(d)
    
    실제 구현:
    - domain/embeddings/utils.py: euclidean_distance()
    - NumPy 벡터화 연산 사용
    """
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    
    # 유클리드 거리 = L2 norm of difference
    distance = np.linalg.norm(v1 - v2)
    
    return float(distance)
```

### 7.2 배치 거리 계산

#### 알고리즘 7.2.1: 배치 유클리드 거리

```
Algorithm: BatchEuclideanDistance(query, candidates)
Input:
  - query: 벡터 q ∈ ℝ^d
  - candidates: 행렬 C ∈ ℝ^(n×d)
Output: 거리 리스트 [d₁, d₂, ..., dₙ]

1. diff_matrix ← C - q  // 브로드캐스팅
2. distances ← [||diff_matrix[i]||₂ for i = 1 to n]
3. return distances
```

**시간 복잡도:** $O(n \cdot d)$  
**공간 복잡도:** $O(n \cdot d)$

**NumPy 구현:**
```python
# 벡터화된 배치 계산
query = np.array(query_vec)
candidates = np.array(candidate_vecs)
distances = np.linalg.norm(candidates - query, axis=1)
```

---

## 8. 수치 안정성과 오차 분석

### 8.1 제곱근 계산의 오차

#### 문제 8.1.1: 제곱근 계산 오차

**제곱근 계산 시 오차:**

$$
\text{fl}(\sqrt{x}) = \sqrt{x}(1 + \epsilon_{\text{sqrt}})
$$

여기서 $\epsilon_{\text{sqrt}} \approx 10^{-7}$ (float32)

#### 정리 8.1.1: 거리 계산의 상대 오차

**상대 오차:**

$$
\frac{|\hat{d} - d|}{d} \leq \epsilon_{\text{machine}} \times \text{condition\_number}
$$

### 8.2 수치 안정성 개선

#### 기법 8.2.1: 제곱 거리 사용

**제곱 거리 (Squared Distance) 사용:**

$$
d_{\text{euc}}^2(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2^2 = \sum_{i=1}^d (u_i - v_i)^2
$$

**장점:**
- 제곱근 계산 불필요
- 더 빠름
- 순서 비교에 충분

**llmkit 구현:**
```python
# 제곱 거리로 비교 (더 빠름)
squared_distances = np.sum((candidates - query)**2, axis=1)
# 순서는 동일하므로 제곱근 불필요
```

---

## 9. 코사인 유사도와의 관계

### 9.1 정규화된 벡터의 거리

#### 정리 9.1.1: 정규화된 벡터의 거리와 코사인 유사도

**정규화된 벡터 ($\|\mathbf{u}\| = \|\mathbf{v}\| = 1$)의 경우:**

$$
d_{\text{euc}}^2(\mathbf{u}, \mathbf{v}) = 2(1 - \text{cosine}(\mathbf{u}, \mathbf{v}))
$$

**증명:**
$$
d_{\text{euc}}^2(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2^2 = (\mathbf{u} - \mathbf{v}) \cdot (\mathbf{u} - \mathbf{v})
$$

$$
= \|\mathbf{u}\|_2^2 - 2\mathbf{u} \cdot \mathbf{v} + \|\mathbf{v}\|_2^2
$$

$$
= 1 - 2\text{cosine}(\mathbf{u}, \mathbf{v}) + 1 = 2(1 - \text{cosine}(\mathbf{u}, \mathbf{v}))
$$

□

#### 역변환

**코사인 유사도:**

$$
\text{cosine}(\mathbf{u}, \mathbf{v}) = 1 - \frac{d_{\text{euc}}^2(\mathbf{u}, \mathbf{v})}{2}
$$

**시각적 관계:**

```
코사인 유사도 vs 유클리드 거리 (정규화된 벡터):

cos = 1.0  →  d = 0.0   (같은 벡터)
cos = 0.5  →  d = 1.0   (중간)
cos = 0.0  →  d = √2    (직교)
cos = -1.0 →  d = 2.0   (반대)
```

---

## 질문과 답변 (Q&A)

### Q1: 유클리드 거리와 코사인 유사도 중 어떤 것을 사용해야 하나요?

**A:** 사용 목적에 따라 다릅니다:

**유클리드 거리 사용:**
- 절대적인 차이 측정
- 크기가 중요한 경우
- 클러스터링 (K-means 등)

**코사인 유사도 사용:**
- 의미적 유사도 측정
- 크기와 무관한 경우
- 텍스트 임베딩 비교

**llmkit 권장:**
- 텍스트 임베딩 → 코사인 유사도
- 특징 벡터 → 유클리드 거리

### Q2: 고차원에서 거리가 의미가 있나요?

**A:** 네, 의미가 있습니다:

1. **임베딩 모델:**
   - 의미를 보존하도록 학습
   - 랜덤 벡터가 아님

2. **실제 검증:**
   - 의미 있는 거리 측정 가능
   - 검색 성능 검증됨

3. **차원 축소:**
   - 필요시 PCA, t-SNE 등 사용
   - 시각화 목적

### Q3: 제곱 거리를 사용해도 되나요?

**A:** 네, 순서 비교에는 충분합니다:

**제곱 거리:**
- 더 빠름 (제곱근 불필요)
- 순서는 동일
- 정규화 불필요

**사용 시기:**
- Top-k 검색
- 순위 비교
- 성능이 중요한 경우

---

## 참고 문헌

1. **Rudin (1976)**: "Principles of Mathematical Analysis" - 거리 공간
2. **Boyd & Vandenberghe (2004)**: "Convex Optimization" - Norm과 거리
3. **Bellman (1957)**: "Dynamic Programming" - 차원의 저주

---

**작성일**: 2025-01-XX  
**버전**: 3.0 (CS 관점 + 석사 수준 확장)

