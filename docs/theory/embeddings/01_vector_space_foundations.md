# Vector Space Foundations: 벡터 공간의 수학적 기초

**석사 수준 이론 문서**  
**CS 관점 + 수학적 엄밀성**  
**기반**: llmkit 실제 구현 코드 분석

---

## 목차

1. [벡터 공간의 정의와 공리](#1-벡터-공간의-정의와-공리)
2. [벡터 공간의 성질과 정리](#2-벡터-공간의-성질과-정리)
3. [유클리드 공간과 내적 공간](#3-유클리드-공간과-내적-공간)
4. [차원과 기저](#4-차원과-기저)
5. [선형 변환과 행렬](#5-선형-변환과-행렬)
6. [임베딩 공간으로의 확장](#6-임베딩-공간으로의-확장)
7. [CS 관점: 데이터 구조와 알고리즘](#7-cs-관점-데이터-구조와-알고리즘)
8. [실제 구현과 최적화](#8-실제-구현과-최적화)

---

## 1. 벡터 공간의 정의와 공리

### 1.1 벡터 공간의 형식적 정의

#### 정의 1.1.1: 벡터 공간 (Vector Space over Field $\mathbb{F}$)

**벡터 공간** $V$는 체(field) $\mathbb{F}$ 위에서 정의되며, 다음 조건을 만족합니다:

**1. 덧셈 연산 (Addition)**
$$
+: V \times V \rightarrow V
$$
$$
(\mathbf{v}, \mathbf{w}) \mapsto \mathbf{v} + \mathbf{w}
$$

**2. 스칼라 곱 (Scalar Multiplication)**
$$
\cdot: \mathbb{F} \times V \rightarrow V
$$
$$
(c, \mathbf{v}) \mapsto c\mathbf{v}
$$

**3. 벡터 공간 공리 (Vector Space Axioms)**

**덧셈 공리:**
- **교환법칙 (Commutativity):**
  $$
  \forall \mathbf{v}, \mathbf{w} \in V: \mathbf{v} + \mathbf{w} = \mathbf{w} + \mathbf{v}
  $$

- **결합법칙 (Associativity):**
  $$
  \forall \mathbf{u}, \mathbf{v}, \mathbf{w} \in V: (\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})
  $$

- **항등원 (Identity Element):**
  $$
  \exists \mathbf{0} \in V: \forall \mathbf{v} \in V: \mathbf{v} + \mathbf{0} = \mathbf{v}
  $$

- **역원 (Inverse Element):**
  $$
  \forall \mathbf{v} \in V, \exists -\mathbf{v} \in V: \mathbf{v} + (-\mathbf{v}) = \mathbf{0}
  $$

**스칼라 곱 공리:**
- **분배법칙 1:**
  $$
  \forall c \in \mathbb{F}, \mathbf{v}, \mathbf{w} \in V: c(\mathbf{v} + \mathbf{w}) = c\mathbf{v} + c\mathbf{w}
  $$

- **분배법칙 2:**
  $$
  \forall c, d \in \mathbb{F}, \mathbf{v} \in V: (c + d)\mathbf{v} = c\mathbf{v} + d\mathbf{v}
  $$

- **결합법칙:**
  $$
  \forall c, d \in \mathbb{F}, \mathbf{v} \in V: (cd)\mathbf{v} = c(d\mathbf{v})
  $$

- **항등원:**
  $$
  \forall \mathbf{v} \in V: 1 \cdot \mathbf{v} = \mathbf{v}
  $$

### 1.2 증명: 벡터 공간의 기본 성질

#### 정리 1.2.1: 영벡터의 유일성

**영벡터는 유일합니다.**

**증명:**
$\mathbf{0}_1$과 $\mathbf{0}_2$가 모두 영벡터라고 가정하면:
$$
\mathbf{0}_1 = \mathbf{0}_1 + \mathbf{0}_2 = \mathbf{0}_2 + \mathbf{0}_1 = \mathbf{0}_2
$$

따라서 $\mathbf{0}_1 = \mathbf{0}_2$입니다. □

#### 정리 1.2.2: 스칼라 곱의 성질

**다음이 성립합니다:**

1. $0 \cdot \mathbf{v} = \mathbf{0}$
2. $c \cdot \mathbf{0} = \mathbf{0}$
3. $(-1) \cdot \mathbf{v} = -\mathbf{v}$

**증명 1:**
$$
0 \cdot \mathbf{v} = (0 + 0) \cdot \mathbf{v} = 0 \cdot \mathbf{v} + 0 \cdot \mathbf{v}
$$

양변에 $-(0 \cdot \mathbf{v})$를 더하면:
$$
\mathbf{0} = 0 \cdot \mathbf{v}
$$

**증명 2:**
$$
c \cdot \mathbf{0} = c \cdot (\mathbf{0} + \mathbf{0}) = c \cdot \mathbf{0} + c \cdot \mathbf{0}
$$

양변에 $-(c \cdot \mathbf{0})$를 더하면:
$$
\mathbf{0} = c \cdot \mathbf{0}
$$

**증명 3:**
$$
\mathbf{v} + (-1) \cdot \mathbf{v} = 1 \cdot \mathbf{v} + (-1) \cdot \mathbf{v} = (1 + (-1)) \cdot \mathbf{v} = 0 \cdot \mathbf{v} = \mathbf{0}
$$

따라서 $(-1) \cdot \mathbf{v} = -\mathbf{v}$입니다. □

---

## 2. 벡터 공간의 성질과 정리

### 2.1 부분 공간 (Subspace)

#### 정의 2.1.1: 부분 공간

$V$의 부분집합 $W$가 다음을 만족하면 **부분 공간**입니다:

1. $\mathbf{0} \in W$
2. $\forall \mathbf{v}, \mathbf{w} \in W: \mathbf{v} + \mathbf{w} \in W$
3. $\forall c \in \mathbb{F}, \mathbf{v} \in W: c\mathbf{v} \in W$

#### 정리 2.1.1: 부분 공간의 교집합

**부분 공간들의 교집합도 부분 공간입니다.**

**증명:**
$W_1, W_2$가 부분 공간이고 $W = W_1 \cap W_2$라고 하면:

1. $\mathbf{0} \in W_1, W_2$이므로 $\mathbf{0} \in W$
2. $\mathbf{v}, \mathbf{w} \in W$이면 $\mathbf{v}, \mathbf{w} \in W_1, W_2$이므로 $\mathbf{v} + \mathbf{w} \in W_1, W_2$ → $\mathbf{v} + \mathbf{w} \in W$
3. $c \in \mathbb{F}, \mathbf{v} \in W$이면 $c\mathbf{v} \in W_1, W_2$ → $c\mathbf{v} \in W$

따라서 $W$는 부분 공간입니다. □

### 2.2 선형 결합과 생성 (Linear Combination and Span)

#### 정의 2.2.1: 선형 결합

벡터 $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n$의 **선형 결합**은:

$$
c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_n\mathbf{v}_n
$$

여기서 $c_i \in \mathbb{F}$입니다.

#### 정의 2.2.2: 생성 (Span)

**Span**은 다음과 같이 정의됩니다:

$$
\text{span}(\{\mathbf{v}_1, \ldots, \mathbf{v}_n\}) = \{c_1\mathbf{v}_1 + \cdots + c_n\mathbf{v}_n | c_i \in \mathbb{F}\}
$$

#### 정리 2.2.1: Span은 부분 공간

**$\text{span}(S)$는 부분 공간입니다.**

**증명:**
1. $\mathbf{0} = 0 \cdot \mathbf{v}_1 + \cdots + 0 \cdot \mathbf{v}_n \in \text{span}(S)$
2. $\mathbf{v}, \mathbf{w} \in \text{span}(S)$이면:
   $$
   \mathbf{v} = \sum_{i=1}^n a_i\mathbf{v}_i, \quad \mathbf{w} = \sum_{i=1}^n b_i\mathbf{v}_i
   $$
   $$
   \mathbf{v} + \mathbf{w} = \sum_{i=1}^n (a_i + b_i)\mathbf{v}_i \in \text{span}(S)
   $$
3. $c \in \mathbb{F}, \mathbf{v} \in \text{span}(S)$이면:
   $$
   c\mathbf{v} = \sum_{i=1}^n (ca_i)\mathbf{v}_i \in \text{span}(S)
   $$

따라서 $\text{span}(S)$는 부분 공간입니다. □

---

## 3. 유클리드 공간과 내적 공간

### 3.1 유클리드 공간

#### 정의 3.1.1: $n$-차원 유클리드 공간

**$n$-차원 유클리드 공간** $\mathbb{R}^n$은 다음과 같이 정의됩니다:

$$
\mathbb{R}^n = \{(x_1, x_2, \ldots, x_n) | x_i \in \mathbb{R}, i = 1, 2, \ldots, n\}
$$

**벡터 연산:**

**덧셈:**
$$
(x_1, \ldots, x_n) + (y_1, \ldots, y_n) = (x_1 + y_1, \ldots, x_n + y_n)
$$

**스칼라 곱:**
$$
c(x_1, \ldots, x_n) = (cx_1, \ldots, cx_n)
$$

#### 예시 3.1.1: 3차원 공간

**3차원 공간** $\mathbb{R}^3$:

$$
\mathbb{R}^3 = \{(x, y, z) | x, y, z \in \mathbb{R}\}
$$

**시각적 표현:**

```
        z
        ↑
        |     v (3, 4, 5)
        |    /
        |   /
        |  /
        | /
        |/________→ y
       /
      /
     / x
```

### 3.2 내적 공간 (Inner Product Space)

#### 정의 3.2.1: 내적 (Inner Product)

**내적**은 다음을 만족하는 함수입니다:

$$
\langle \cdot, \cdot \rangle: V \times V \rightarrow \mathbb{R}
$$

**내적 공리:**

1. **양의 정부호 (Positive Definiteness):**
   $$
   \forall \mathbf{v} \in V: \langle \mathbf{v}, \mathbf{v} \rangle \geq 0
   $$
   $$
   \langle \mathbf{v}, \mathbf{v} \rangle = 0 \iff \mathbf{v} = \mathbf{0}
   $$

2. **대칭성 (Symmetry):**
   $$
   \forall \mathbf{v}, \mathbf{w} \in V: \langle \mathbf{v}, \mathbf{w} \rangle = \langle \mathbf{w}, \mathbf{v} \rangle
   $$

3. **선형성 (Linearity):**
   $$
   \forall c \in \mathbb{R}, \mathbf{u}, \mathbf{v}, \mathbf{w} \in V:
   $$
   $$
   \langle c\mathbf{u} + \mathbf{v}, \mathbf{w} \rangle = c\langle \mathbf{u}, \mathbf{w} \rangle + \langle \mathbf{v}, \mathbf{w} \rangle
   $$

#### 정의 3.2.2: 표준 내적 (Dot Product)

**$\mathbb{R}^n$의 표준 내적:**

$$
\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i
$$

**행렬 표기:**

$$
\mathbf{u} \cdot \mathbf{v} = \mathbf{u}^T \mathbf{v}
$$

#### 정리 3.2.1: 코시-슈바르츠 부등식 (Cauchy-Schwarz Inequality)

**다음이 성립합니다:**

$$
|\langle \mathbf{u}, \mathbf{v} \rangle| \leq \|\mathbf{u}\| \|\mathbf{v}\|
$$

**등호는 $\mathbf{u}$와 $\mathbf{v}$가 선형 종속일 때 성립합니다.**

**증명:**
$\mathbf{v} = \mathbf{0}$이면 자명합니다. $\mathbf{v} \neq \mathbf{0}$라고 가정하면:

$$
0 \leq \left\|\mathbf{u} - \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{v}\|^2}\mathbf{v}\right\|^2
$$

$$
= \|\mathbf{u}\|^2 - \frac{|\langle \mathbf{u}, \mathbf{v} \rangle|^2}{\|\mathbf{v}\|^2}
$$

따라서:
$$
|\langle \mathbf{u}, \mathbf{v} \rangle|^2 \leq \|\mathbf{u}\|^2 \|\mathbf{v}\|^2
$$

양변에 제곱근을 취하면:
$$
|\langle \mathbf{u}, \mathbf{v} \rangle| \leq \|\mathbf{u}\| \|\mathbf{v}\|
$$

□

---

## 4. 차원과 기저

### 4.1 선형 독립과 기저

#### 정의 4.1.1: 선형 독립 (Linear Independence)

벡터 $\mathbf{v}_1, \ldots, \mathbf{v}_n$이 **선형 독립**인 것은:

$$
c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_n\mathbf{v}_n = \mathbf{0} \implies c_1 = c_2 = \cdots = c_n = 0
$$

#### 정의 4.1.2: 기저 (Basis)

벡터 집합 $B = \{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$이 **기저**인 것은:

1. $B$는 선형 독립
2. $\text{span}(B) = V$

#### 정리 4.1.1: 기저의 유일성

**모든 벡터는 기저에 대해 유일한 표현을 가집니다.**

**증명:**
$\mathbf{v} = \sum_{i=1}^n a_i\mathbf{v}_i = \sum_{i=1}^n b_i\mathbf{v}_i$라고 하면:

$$
\sum_{i=1}^n (a_i - b_i)\mathbf{v}_i = \mathbf{0}
$$

선형 독립성에 의해 $a_i - b_i = 0$이므로 $a_i = b_i$입니다. □

### 4.2 차원 (Dimension)

#### 정의 4.2.1: 차원

**차원**은 기저의 크기입니다:

$$
\dim(V) = |B|
$$

여기서 $B$는 $V$의 기저입니다.

#### 정리 4.2.1: 차원의 일관성

**모든 기저는 같은 크기를 가집니다.**

**증명:**
$B_1 = \{\mathbf{u}_1, \ldots, \mathbf{u}_m\}$과 $B_2 = \{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$이 모두 기저라고 하면:

$B_1$이 선형 독립이고 $B_2$가 생성하므로 $m \leq n$입니다.
$B_2$가 선형 독립이고 $B_1$이 생성하므로 $n \leq m$입니다.

따라서 $m = n$입니다. □

---

## 5. 선형 변환과 행렬

### 5.1 선형 변환

#### 정의 5.1.1: 선형 변환 (Linear Transformation)

함수 $T: V \rightarrow W$가 **선형 변환**인 것은:

1. **가산성 (Additivity):**
   $$
   T(\mathbf{v} + \mathbf{w}) = T(\mathbf{v}) + T(\mathbf{w})
   $$

2. **동질성 (Homogeneity):**
   $$
   T(c\mathbf{v}) = cT(\mathbf{v})
   $$

#### 정리 5.1.1: 선형 변환의 성질

**선형 변환은 다음을 보존합니다:**

1. $T(\mathbf{0}) = \mathbf{0}$
2. $T(-\mathbf{v}) = -T(\mathbf{v})$
3. $T(c_1\mathbf{v}_1 + \cdots + c_n\mathbf{v}_n) = c_1T(\mathbf{v}_1) + \cdots + c_nT(\mathbf{v}_n)$

**증명 1:**
$$
T(\mathbf{0}) = T(0 \cdot \mathbf{v}) = 0 \cdot T(\mathbf{v}) = \mathbf{0}
$$

**증명 2:**
$$
T(-\mathbf{v}) = T((-1) \cdot \mathbf{v}) = (-1) \cdot T(\mathbf{v}) = -T(\mathbf{v})
$$

**증명 3:**
수학적 귀납법으로 증명 가능합니다. □

### 5.2 행렬 표현

#### 정리 5.2.1: 선형 변환의 행렬 표현

**$T: \mathbb{R}^n \rightarrow \mathbb{R}^m$ 선형 변환은 $m \times n$ 행렬로 표현됩니다:**

$$
T(\mathbf{v}) = A\mathbf{v}
$$

여기서 $A$의 $j$번째 열은 $T(\mathbf{e}_j)$입니다 ($\mathbf{e}_j$는 표준 기저 벡터).

**증명:**
$\mathbf{v} = \sum_{j=1}^n v_j\mathbf{e}_j$라고 하면:

$$
T(\mathbf{v}) = T\left(\sum_{j=1}^n v_j\mathbf{e}_j\right) = \sum_{j=1}^n v_j T(\mathbf{e}_j) = A\mathbf{v}
$$

여기서 $A = [T(\mathbf{e}_1) | T(\mathbf{e}_2) | \cdots | T(\mathbf{e}_n)]$입니다. □

---

## 6. 임베딩 공간으로의 확장

### 6.1 임베딩 함수의 수학적 정의

#### 정의 6.1.1: 임베딩 함수

**임베딩 함수**는 다음과 같이 정의됩니다:

$$
f: X \rightarrow \mathbb{R}^d
$$

여기서:
- $X$: 원본 공간 (단어, 문장, 문서 등)
- $\mathbb{R}^d$: $d$-차원 실수 벡터 공간
- $d$: 임베딩 차원

#### 정리 6.1.1: Johnson-Lindenstrauss Lemma

**$n$개의 점을 $d = O(\log n / \epsilon^2)$ 차원으로 임베딩할 수 있으며, 거리는 $(1 \pm \epsilon)$ 배 이내로 보존됩니다.**

**llmkit에서의 적용:**
- 텍스트 임베딩 차원: 1536 (text-embedding-3-small)
- 대규모 문서 컬렉션에서도 의미 보존

### 6.2 거리 보존 (Distance Preservation)

#### 정의 6.2.1: 거리 보존 임베딩

**좋은 임베딩 함수는 원본 공간의 거리를 보존합니다:**

$$
d_X(x_1, x_2) \approx d_{\mathbb{R}^d}(f(x_1), f(x_2))
$$

**시각적 표현:**

```
원본 공간 X              임베딩 공간 ℝ^d
    x₁                        f(x₁) ★
     │                            │
     │ d_X(x₁, x₂)                │ d(f(x₁), f(x₂))
     │                            │
     ▼                            ▼
    x₂                        f(x₂) ★
    
거리 비율: d(f(x₁), f(x₂)) / d_X(x₁, x₂) ≈ 1
```

---

## 7. CS 관점: 데이터 구조와 알고리즘

### 7.1 벡터 표현의 데이터 구조

#### CS 관점 7.1.1: 벡터의 메모리 표현

**벡터 $\mathbf{v} \in \mathbb{R}^d$의 메모리 표현:**

```
메모리 레이아웃 (Row-major):

주소:  [0]    [4]    [8]    [12]   ...
값:    v[0]   v[1]   v[2]   v[3]   ...
      (float32, 4 bytes each)

총 메모리: d × 4 bytes (float32)
예: d=1536 → 6,144 bytes = 6 KB
```

**llmkit 구현:**
```python
# domain/embeddings/utils.py: cosine_similarity()
# NumPy float32 사용 (메모리 효율적)
v1 = np.array(vec1, dtype=np.float32)  # 4 bytes per element
v2 = np.array(vec2, dtype=np.float32)  # 메모리 효율적

# 실제 구현:
# - domain/embeddings/utils.py: cosine_similarity() (Line 76-77)
# - float32: 4 bytes, 정밀도 7자리 (임베딩에 충분)
# - SIMD 명령어 활용 가능 (벡터화 연산 가속)
```

### 7.2 벡터 연산의 시간 복잡도

#### CS 관점 7.2.1: 벡터 연산 복잡도

**기본 연산의 시간 복잡도:**

| 연산 | 시간 복잡도 | 공간 복잡도 |
|------|------------|------------|
| 덧셈 $\mathbf{u} + \mathbf{v}$ | $O(d)$ | $O(d)$ |
| 스칼라 곱 $c\mathbf{v}$ | $O(d)$ | $O(d)$ |
| 내적 $\mathbf{u} \cdot \mathbf{v}$ | $O(d)$ | $O(1)$ |
| Norm $\|\mathbf{v}\|$ | $O(d)$ | $O(1)$ |

**llmkit 구현:**
```python
# domain/embeddings/utils.py: cosine_similarity()
# 순수 Python 구현 (numpy 없을 때 폴백)
dot_product = sum(a * b for a, b in zip(vec1, vec2))  # O(d)
norm1 = sum(a * a for a in vec1) ** 0.5  # O(d)
norm2 = sum(b * b for b in vec2) ** 0.5  # O(d)
similarity = dot_product / (norm1 * norm2)  # O(1)

# NumPy 구현: 벡터화로 더 빠름
v1 = np.array(vec1, dtype=np.float32)
v2 = np.array(vec2, dtype=np.float32)
norm1 = np.linalg.norm(v1)  # O(d) but SIMD 가속
norm2 = np.linalg.norm(v2)
similarity = np.dot(v1, v2) / (norm1 * norm2)  # C 레벨 최적화

# 실제 구현:
# - domain/embeddings/utils.py: cosine_similarity() (Line 64-98)
# - NumPy SIMD 명령어 활용 (AVX, SSE 등)
```

### 7.3 행렬-벡터 곱의 최적화

#### CS 관점 7.3.1: 행렬-벡터 곱

**행렬-벡터 곱 $A\mathbf{v}$:**

$$
(A\mathbf{v})_i = \sum_{j=1}^n A_{ij} v_j
$$

**시간 복잡도:** $O(mn)$ (행렬 $m \times n$, 벡터 $n$)

**최적화 기법:**

1. **벡터화 (Vectorization):**
   - NumPy, BLAS 라이브러리 활용
   - SIMD (Single Instruction Multiple Data) 명령어

2. **캐시 최적화:**
   - 메모리 접근 패턴 최적화
   - 블록 행렬 곱

**llmkit 구현:**
```python
# NumPy는 BLAS 사용
result = np.dot(matrix, vector)  # 최적화된 구현
```

---

## 8. 실제 구현과 최적화

### 8.1 llmkit의 벡터 표현

#### 구현 8.1.1: Python 리스트 vs NumPy 배열

**Python 리스트:**
```python
vec = [0.1, 0.2, 0.3, ...]  # List[float]
# 메모리: 각 요소가 Python 객체 (24 bytes)
# 시간: 순수 Python 루프 (느림)
```

**NumPy 배열:**
```python
vec = np.array([0.1, 0.2, 0.3, ...], dtype=np.float32)
# 메모리: 연속 메모리 (4 bytes per element)
# 시간: C 레벨 최적화 (빠름)
```

**성능 비교:**

| 연산 | Python List | NumPy Array | 속도 향상 |
|------|------------|-------------|----------|
| 내적 | $O(d)$ (느림) | $O(d)$ (빠름) | ~100배 |
| Norm | $O(d)$ (느림) | $O(d)$ (빠름) | ~50배 |

### 8.2 배치 처리 최적화

#### 구현 8.2.1: 배치 벡터 연산

**단일 쿼리:**
```python
# O(d) 시간
similarity = cosine_similarity(query_vec, candidate_vec)
```

**배치 처리:**
```python
# O(n·d) 시간, 하지만 벡터화로 더 빠름
similarities = batch_cosine_similarity(query_vec, candidate_vecs)
```

**NumPy 벡터화:**
```python
# domain/embeddings/utils.py (배치 처리)
# 배치 코사인 유사도 계산
query = np.array(query_vec, dtype=np.float32)
candidates = np.array(candidate_vecs, dtype=np.float32)  # [n, d] 행렬

# 실제 구현:
# - domain/embeddings/utils.py: batch_cosine_similarity() (또는 직접 구현)
# - NumPy 행렬 연산 사용 (SIMD 가속)
```

# 벡터화된 연산: O(n·d) 하지만 SIMD로 가속
similarities = np.dot(candidates, query) / (norms * query_norm)
```

**성능:**
- 순수 Python: $O(n \cdot d)$ (느림)
- NumPy: $O(n \cdot d)$ (빠름, SIMD 활용)

---

## 질문과 답변 (Q&A)

### Q1: 왜 벡터 공간을 사용하나요?

**A:** 벡터 공간은 다음 이유로 유용합니다:

1. **선형 연산의 편의성:**
   - 덧셈, 스칼라 곱이 자연스럽게 정의됨
   - 선형 대수 도구 활용 가능

2. **기하학적 직관:**
   - 거리, 각도, 유사도 계산 가능
   - 시각화 용이

3. **계산 효율성:**
   - 행렬 연산으로 최적화 가능
   - 병렬 처리 용이

### Q2: 임베딩 차원은 어떻게 선택하나요?

**A:** 차원 선택은 다음을 고려합니다:

1. **Johnson-Lindenstrauss Lemma:**
   $$
   d = O\left(\frac{\log n}{\epsilon^2}\right)
   $$
   - $n$: 데이터 포인트 수
   - $\epsilon$: 허용 오차

2. **실험적 결과:**
   - 128-512: 작은 데이터셋
   - 512-1024: 중간 데이터셋
   - 1024-1536: 대규모 데이터셋

3. **메모리 제약:**
   - 차원이 높을수록 메모리 사용량 증가
   - $d \times 4$ bytes per vector (float32)

### Q3: 왜 코사인 유사도를 사용하나요?

**A:** 코사인 유사도의 장점:

1. **크기 불변성:**
   - 벡터의 크기와 무관, 방향만 측정
   - 텍스트 길이에 영향받지 않음

2. **범위 제한:**
   - $[-1, 1]$ 범위로 해석 용이
   - 확률로 변환 가능

3. **계산 효율성:**
   - 정규화된 벡터는 내적만으로 계산
   - $O(d)$ 시간 복잡도

---

## 참고 문헌

1. **Axler (2015)**: "Linear Algebra Done Right" - 벡터 공간 이론
2. **Johnson & Lindenstrauss (1984)**: "Extensions of Lipschitz mappings into a Hilbert space"
3. **Strang (2016)**: "Introduction to Linear Algebra" - 행렬과 선형 변환

---

**작성일**: 2025-01-XX  
**버전**: 3.0 (CS 관점 + 석사 수준 확장)

