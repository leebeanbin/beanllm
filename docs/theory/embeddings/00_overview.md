# Embeddings Theory: 수학적 기초와 구현 원리

**석사 수준 이론 문서**  
**기반**: llmkit 실제 구현 코드 분석

---

## 목차

### Part I: 수학적 기초
1. [벡터 공간 이론](#part-i-수학적-기초)
2. [임베딩 함수의 수학적 정의](#12-임베딩-함수의-수학적-정의)
3. [거리와 유사도 측정](#13-거리와-유사도-측정)

### Part II: 임베딩 모델 아키텍처
4. [Transformer 기반 임베딩](#part-ii-임베딩-모델-아키텍처)
5. [문맥 임베딩의 수학적 모델](#42-문맥-임베딩의-수학적-모델)
6. [다국어 임베딩과 Cross-lingual Alignment](#43-다국어-임베딩과-cross-lingual-alignment)

### Part III: 유사도 계산의 수학적 원리
7. [코사인 유사도: 기하학적 해석](#part-iii-유사도-계산의-수학적-원리)
8. [유클리드 거리와 L2 Norm](#72-유클리드-거리와-l2-norm)
9. [벡터 정규화의 수학적 의미](#73-벡터-정규화의-수학적-의미)

### Part IV: 고급 기법의 수학적 분석
10. [Hard Negative Mining: Contrastive Learning 이론](#part-iv-고급-기법의-수학적-분석)
11. [MMR: 정보 이론적 관점](#102-mmr-정보-이론적-관점)
12. [Query Expansion: 확률 모델](#103-query-expansion-확률-모델)

### Part V: 구현과 최적화
13. [배치 처리의 계산 복잡도](#part-v-구현과-최적화)
14. [캐싱 전략과 시간 복잡도](#132-캐싱-전략과-시간-복잡도)

---

## Part I: 수학적 기초

### 1.1 벡터 공간 이론

#### 정의 1.1.1: 벡터 공간 (Vector Space)

**벡터 공간** $V$는 다음 조건을 만족하는 집합입니다:

1. **덧셈 닫힘 (Closure under Addition)**
   $$
   \forall \mathbf{v}, \mathbf{w} \in V: \mathbf{v} + \mathbf{w} \in V
   $$

2. **스칼라 곱 닫힘 (Closure under Scalar Multiplication)**
   $$
   \forall \mathbf{v} \in V, c \in \mathbb{R}: c\mathbf{v} \in V
   $$

3. **벡터 공간 공리 (Vector Space Axioms)**
   - 교환법칙: $\mathbf{v} + \mathbf{w} = \mathbf{w} + \mathbf{v}$
   - 결합법칙: $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$
   - 항등원: $\exists \mathbf{0} \in V: \mathbf{v} + \mathbf{0} = \mathbf{v}$
   - 역원: $\forall \mathbf{v} \in V, \exists -\mathbf{v}: \mathbf{v} + (-\mathbf{v}) = \mathbf{0}$

#### 예시 1.1.1: 유클리드 공간

**$n$-차원 유클리드 공간** $\mathbb{R}^n$:
$$
\mathbb{R}^n = \{(x_1, x_2, \ldots, x_n) | x_i \in \mathbb{R}, i = 1, 2, \ldots, n\}
$$

**llmkit 구현:**
```python
# domain/embeddings/utils.py: cosine_similarity()
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    코사인 유사도 계산: cosine(u, v) = (u·v) / (||u|| ||v||)
    
    Args:
        vec1: 첫 번째 임베딩 벡터 (예: text-embedding-3-small → 1536차원)
        vec2: 두 번째 임베딩 벡터 (같은 차원이어야 함)
    
    Returns:
        코사인 유사도 값 (-1 ~ 1, 1에 가까울수록 유사)
    
    실제 구현:
        - domain/embeddings/utils.py: cosine_similarity()
        - numpy 기반 효율적 계산 (없으면 순수 Python 폴백)
        - 수치 안정성을 위해 -1과 1 사이로 클리핑
    """
    v1 = np.array(vec1, dtype=np.float32)  # ℝ^d (예: d=1536)
    v2 = np.array(vec2, dtype=np.float32)  # ℝ^d
    
    # L2 Norm 계산
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # 코사인 유사도 = (A · B) / (||A|| * ||B||)
    similarity = np.dot(v1, v2) / (norm1 * norm2)
    
    # 수치 안정성을 위해 -1과 1 사이로 클리핑
    return float(np.clip(similarity, -1.0, 1.0))
```

---

### 1.2 임베딩 함수의 수학적 정의

#### 정의 1.2.1: 임베딩 함수 (Embedding Function)

**임베딩 함수**는 다음과 같이 정의됩니다:

$$
f: X \rightarrow \mathbb{R}^d
$$

여기서:
- $X$: 원본 공간 (단어, 문장, 문서 등)
- $\mathbb{R}^d$: $d$-차원 실수 벡터 공간
- $d$: 임베딩 차원 (embedding dimension)

#### 성질 1.2.1: 거리 보존 (Distance Preservation)

좋은 임베딩 함수는 원본 공간의 거리를 보존합니다:

$$
d_X(x_1, x_2) \approx d_{\mathbb{R}^d}(f(x_1), f(x_2))
$$

여기서 $d_X$는 원본 공간의 거리 함수, $d_{\mathbb{R}^d}$는 벡터 공간의 거리 함수입니다.

#### 정리 1.2.1: Johnson-Lindenstrauss Lemma

**Johnson-Lindenstrauss Lemma** (1984): $n$개의 점을 $d = O(\log n / \epsilon^2)$ 차원으로 임베딩할 수 있으며, 거리는 $(1 \pm \epsilon)$ 배 이내로 보존됩니다.

**llmkit에서의 적용:**
- 텍스트 임베딩 차원: 1536 (text-embedding-3-small)
- 대규모 문서 컬렉션에서도 의미 보존

---

### 1.3 거리와 유사도 측정

#### 정의 1.3.1: 거리 함수 (Distance Function)

**거리 함수** $d: V \times V \rightarrow \mathbb{R}$는 다음을 만족합니다:

1. **비음성 (Non-negativity)**: $d(\mathbf{v}, \mathbf{w}) \geq 0$
2. **구별성 (Identity)**: $d(\mathbf{v}, \mathbf{w}) = 0 \iff \mathbf{v} = \mathbf{w}$
3. **대칭성 (Symmetry)**: $d(\mathbf{v}, \mathbf{w}) = d(\mathbf{w}, \mathbf{v})$
4. **삼각 부등식 (Triangle Inequality)**: $d(\mathbf{u}, \mathbf{w}) \leq d(\mathbf{u}, \mathbf{v}) + d(\mathbf{v}, \mathbf{w})$

#### 정의 1.3.2: 유사도 함수 (Similarity Function)

**유사도 함수** $s: V \times V \rightarrow [-1, 1]$는 거리 함수의 역변환입니다:

$$
s(\mathbf{v}, \mathbf{w}) = 1 - \frac{d(\mathbf{v}, \mathbf{w})}{\max d}
$$

---

## Part II: 임베딩 모델 아키텍처

### 2.1 Transformer 기반 임베딩

#### 아키텍처 2.1.1: Encoder-Only 모델

**BERT 스타일 임베딩:**

$$
\mathbf{h}_i = \text{Transformer-Encoder}(\mathbf{x}_1, \ldots, \mathbf{x}_n)_i
$$

$$
\mathbf{s} = \text{mean-pooling}(\mathbf{h}_1, \ldots, \mathbf{h}_n) = \frac{1}{n}\sum_{i=1}^n \mathbf{h}_i
$$

**llmkit 구현:**
```python
# infrastructure/providers/openai_provider.py: OpenAIProvider
# domain/embeddings/base.py: BaseEmbedding
# 각 Provider는 Transformer 기반 모델 사용
class OpenAIEmbedding(BaseEmbedding):
    """
    OpenAI 임베딩: Transformer 기반 모델
    
    실제 구현:
    - infrastructure/providers/openai_provider.py: OpenAIProvider
    - domain/embeddings/base.py: BaseEmbedding (추상 클래스)
    """
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        OpenAI API는 내부적으로 Transformer 사용
        
        실제 구현:
        - infrastructure/providers/openai_provider.py: OpenAIProvider.embed()
        """
        response = await self.async_client.embeddings.create(
            input=texts, model=self.model
        )
        return [item.embedding for item in response.data]
```

---

### 2.2 문맥 임베딩의 수학적 모델

#### 정의 2.2.1: 문맥 임베딩 (Contextual Embedding)

**문맥 임베딩**은 단어의 의미가 문맥에 따라 달라지는 것을 모델링합니다:

$$
E(w, C) = f_{\text{transformer}}(w, C)
$$

여기서 $C$는 문맥 (context), $w$는 단어입니다.

#### 예시 2.2.1: 동음이의어 처리

**한국어 예시:**
- "은행" (금융기관): $E(\text{은행}, C_{\text{금융}}) = \mathbf{v}_1$
- "은행" (강가): $E(\text{은행}, C_{\text{강가}}) = \mathbf{v}_2$

$$
\mathbf{v}_1 \neq \mathbf{v}_2 \text{ (다른 벡터)}
$$

**수학적 표현:**
$$
\text{sim}(\mathbf{v}_1, \mathbf{v}_2) < \text{threshold}
$$

---

### 2.3 다국어 임베딩과 Cross-lingual Alignment

#### 정의 2.3.1: Cross-lingual Embedding Space

**다국어 임베딩 공간**은 여러 언어를 같은 벡터 공간에 매핑합니다:

$$
f_{\text{ko}}: \text{한국어} \rightarrow \mathbb{R}^d
$$
$$
f_{\text{en}}: \text{영어} \rightarrow \mathbb{R}^d
$$

**정렬 조건:**
$$
\text{sim}(f_{\text{ko}}(\text{고양이}), f_{\text{en}}(\text{cat})) \approx 1
$$

**llmkit 구현:**
```python
# domain/embeddings/base.py: BaseEmbedding
# facade/embeddings_facade.py: Embedding
# 다국어 모델 자동 선택
emb = Embedding(model="embed-multilingual-v3.0")
# 한국어와 영어를 같은 공간에 매핑

# 실제 구현:
# - domain/embeddings/base.py: BaseEmbedding (추상 클래스)
# - facade/embeddings_facade.py: Embedding (사용자 API)
# - infrastructure/providers/: 각 Provider별 구현
```

---

## Part III: 유사도 계산의 수학적 원리

### 3.1 코사인 유사도: 기하학적 해석

#### 정의 3.1.1: 코사인 유사도 (Cosine Similarity)

**코사인 유사도**는 두 벡터 사이의 각도를 측정합니다:

$$
\text{cosine}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \cos(\theta)
$$

여기서 $\theta$는 두 벡터 사이의 각도입니다.

#### 시각적 표현: 2D 벡터 공간

```
        y
        ↑
        |     v (3, 4)
        |    /
        |   /  θ
        |  / 
        | /
        |/________→ x
       u (1, 2)
       
각도 θ가 작을수록 → 유사도 높음 (cos(θ) ≈ 1)
각도 θ가 클수록 → 유사도 낮음 (cos(θ) ≈ 0)
```

#### 구체적 수치 예시

**예시 3.1.1: 실제 벡터 계산**

두 텍스트의 임베딩 벡터:
- $\mathbf{u} = [0.5, 0.3, 0.8, 0.2]$ (텍스트: "고양이는 귀여워")
- $\mathbf{v} = [0.6, 0.2, 0.7, 0.3]$ (텍스트: "강아지는 귀여워")

**단계별 계산:**

1. **내적 (Dot Product):**
   $$
   \mathbf{u} \cdot \mathbf{v} = 0.5 \times 0.6 + 0.3 \times 0.2 + 0.8 \times 0.7 + 0.2 \times 0.3
   $$
   $$
   = 0.30 + 0.06 + 0.56 + 0.06 = 0.98
   $$

2. **L2 Norm 계산:**
   $$
   \|\mathbf{u}\| = \sqrt{0.5^2 + 0.3^2 + 0.8^2 + 0.2^2} = \sqrt{0.25 + 0.09 + 0.64 + 0.04} = \sqrt{1.02} \approx 1.010
   $$
   $$
   \|\mathbf{v}\| = \sqrt{0.6^2 + 0.2^2 + 0.7^2 + 0.3^2} = \sqrt{0.36 + 0.04 + 0.49 + 0.09} = \sqrt{0.98} \approx 0.990
   $$

3. **코사인 유사도:**
   $$
   \text{cosine}(\mathbf{u}, \mathbf{v}) = \frac{0.98}{1.010 \times 0.990} = \frac{0.98}{1.000} \approx 0.980
   $$

**해석:** 유사도 0.980은 매우 높은 유사도를 의미합니다 (거의 같은 방향).

#### 정리 3.1.1: 코사인 유사도의 성질

1. **범위**: $\text{cosine}(\mathbf{u}, \mathbf{v}) \in [-1, 1]$
   - $1$: 완전히 같은 방향 (동일한 의미)
   - $0$: 직교 (독립적)
   - $-1$: 반대 방향 (반대 의미)

2. **방향성**: 벡터의 크기와 무관, 방향만 측정
   ```
   예시:
   u = [1, 2]     → ||u|| = √5 ≈ 2.236
   u' = [2, 4]    → ||u'|| = √20 ≈ 4.472
   
   cosine(u, u') = (1×2 + 2×4) / (√5 × √20)
                 = 10 / 10 = 1.0
   
   → 크기가 달라도 방향이 같으면 유사도 = 1
   ```

3. **정규화**: $\|\mathbf{u}\| = \|\mathbf{v}\| = 1$이면 $\text{cosine}(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v}$

#### 증명 3.1.1: 코사인 법칙

**코사인 법칙**에 의해:
$$
\|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2\|\mathbf{u}\|\|\mathbf{v}\|\cos(\theta)
$$

정리하면:
$$
\cos(\theta) = \frac{\|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - \|\mathbf{u} - \mathbf{v}\|^2}{2\|\mathbf{u}\|\|\mathbf{v}\|}
$$

**시각적 증명:**

```
        v
        ↑
        |\
        | \
        |  \ u-v
        |   \
        |    \
        |_____\→ u
        θ
       
삼각형의 코사인 법칙 적용
```

#### 실제 llmkit 사용 예시

**llmkit 구현:**
```python
# domain/embeddings/utils.py: cosine_similarity()
# 코사인 유사도 = (A · B) / (||A|| * ||B||)
similarity = np.dot(v1, v2) / (norm1 * norm2)

# 실제 구현:
# - domain/embeddings/utils.py: cosine_similarity() (Line 95)
# - NumPy 벡터화 연산 사용
```

**실제 사용 예시:**
```python
from llmkit.embeddings import embed_sync, cosine_similarity

# 1. 텍스트 임베딩
text1 = "고양이는 귀여워"
text2 = "강아지는 귀여워"
text3 = "자동차는 빠르다"

vec1 = embed_sync(text1)[0]  # [0.5, 0.3, 0.8, ...] (1536차원)
vec2 = embed_sync(text2)[0]  # [0.6, 0.2, 0.7, ...]
vec3 = embed_sync(text3)[0]  # [0.1, 0.9, 0.2, ...]

# 2. 유사도 계산
sim_12 = cosine_similarity(vec1, vec2)  # ≈ 0.85 (높은 유사도)
sim_13 = cosine_similarity(vec1, vec3)  # ≈ 0.15 (낮은 유사도)

print(f"'{text1}' vs '{text2}': {sim_12:.3f}")
print(f"'{text1}' vs '{text3}': {sim_13:.3f}")

# 출력:
# '고양이는 귀여워' vs '강아지는 귀여워': 0.850
# '고양이는 귀여워' vs '자동차는 빠르다': 0.150
```

#### 유사도 분포 시각화

```
유사도 분포 예시:

1.0 |                    ★ (같은 텍스트)
    |
0.8 |         ★ (유사한 의미)
    |    ★
0.5 | ★
    |★
0.0 |________________________
    -1.0  -0.5  0.0  0.5  1.0
    
★ = 텍스트 쌍의 유사도
```

---

### 3.2 유클리드 거리와 L2 Norm

#### 정의 3.2.1: L2 Norm (유클리드 Norm)

**L2 Norm**은 벡터의 길이를 측정합니다:

$$
\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^d v_i^2} = \sqrt{\mathbf{v} \cdot \mathbf{v}}
$$

#### 시각적 표현: 2D 공간에서의 벡터 길이

```
        y
        ↑
        |     v (3, 4)
        |    /
        |   /|
        |  / |
        | /  | 4
        |/___|__→ x
       0  3
        
||v|| = √(3² + 4²) = √(9 + 16) = √25 = 5
```

#### 구체적 수치 예시

**예시 3.2.1: L2 Norm 계산**

벡터 $\mathbf{v} = [3, 4, 0, 12]$:

$$
\|\mathbf{v}\|_2 = \sqrt{3^2 + 4^2 + 0^2 + 12^2} = \sqrt{9 + 16 + 0 + 144} = \sqrt{169} = 13
$$

**단계별 계산:**
1. 각 성분 제곱: $3^2=9, 4^2=16, 0^2=0, 12^2=144$
2. 합계: $9 + 16 + 0 + 144 = 169$
3. 제곱근: $\sqrt{169} = 13$

#### 정의 3.2.2: 유클리드 거리 (Euclidean Distance)

**유클리드 거리**는 L2 Norm의 차이입니다:

$$
d_{\text{euc}}(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{\sum_{i=1}^d (u_i - v_i)^2}
$$

#### 시각적 표현: 2D 공간에서의 거리

```
        y
        ↑
        |     v (4, 5)
        |    /
        |   /|
        |  / | d
        | /  |
        |/___|__→ x
       u (1, 2)
       
d = √((4-1)² + (5-2)²) = √(9 + 9) = √18 ≈ 4.24
```

#### 구체적 수치 예시

**예시 3.2.2: 유클리드 거리 계산**

두 벡터:
- $\mathbf{u} = [1, 2, 3]$
- $\mathbf{v} = [4, 6, 8]$

**단계별 계산:**

1. **차이 벡터:**
   $$
   \mathbf{u} - \mathbf{v} = [1-4, 2-6, 3-8] = [-3, -4, -5]
   $$

2. **제곱:**
   $$
   (-3)^2 = 9, (-4)^2 = 16, (-5)^2 = 25
   $$

3. **합계:**
   $$
   9 + 16 + 25 = 50
   $$

4. **제곱근:**
   $$
   d_{\text{euc}}(\mathbf{u}, \mathbf{v}) = \sqrt{50} \approx 7.071
   $$

#### 정리 3.2.1: 거리와 유사도의 관계

코사인 유사도와 유클리드 거리의 관계:

$$
\text{cosine}(\mathbf{u}, \mathbf{v}) = 1 - \frac{d_{\text{euc}}(\mathbf{u}', \mathbf{v}')^2}{2}
$$

여기서 $\mathbf{u}'$, $\mathbf{v}'$는 정규화된 벡터입니다.

**시각적 비교:**

```
코사인 유사도 vs 유클리드 거리:

유사도 높음 (cos ≈ 1.0)  →  거리 작음 (d ≈ 0)
유사도 중간 (cos ≈ 0.5)  →  거리 중간 (d ≈ 1.0)
유사도 낮음 (cos ≈ 0.0)  →  거리 큼 (d ≈ 1.4)

정규화된 벡터의 경우:
cos(θ) = 1 - d²/2
```

#### 실제 llmkit 사용 예시

**llmkit 구현:**
```python
# domain/embeddings/utils.py: euclidean_distance()
# 유클리드 거리 = sqrt(sum((a_i - b_i)^2))
distance = np.linalg.norm(v1 - v2)

# 실제 구현:
# - domain/embeddings/utils.py: euclidean_distance() (Line 105-106)
# - NumPy 벡터화 연산 사용
```

**실제 사용 예시:**
```python
from llmkit.embeddings import embed_sync, euclidean_distance, cosine_similarity

# 텍스트 임베딩
text1 = "고양이는 귀여워"
text2 = "강아지는 귀여워"

vec1 = embed_sync(text1)[0]
vec2 = embed_sync(text2)[0]

# 유클리드 거리
dist = euclidean_distance(vec1, vec2)
print(f"유클리드 거리: {dist:.3f}")  # 예: 0.523

# 코사인 유사도
sim = cosine_similarity(vec1, vec2)
print(f"코사인 유사도: {sim:.3f}")  # 예: 0.850

# 관계 확인 (정규화된 벡터의 경우)
# sim ≈ 1 - dist²/2
```

---

### 3.3 벡터 정규화의 수학적 의미

#### 정의 3.3.1: L2 정규화 (L2 Normalization)

**L2 정규화**는 벡터를 단위 벡터로 변환합니다:

$$
\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2} = \frac{\mathbf{v}}{\sqrt{\sum_{i=1}^d v_i^2}}
$$

#### 정리 3.3.1: 정규화 후 Norm

정규화된 벡터의 L2 Norm은 항상 1입니다:

$$
\|\mathbf{v}_{\text{norm}}\|_2 = \left\|\frac{\mathbf{v}}{\|\mathbf{v}\|_2}\right\|_2 = \frac{\|\mathbf{v}\|_2}{\|\mathbf{v}\|_2} = 1
$$

#### 정리 3.3.2: 정규화 후 코사인 유사도

정규화된 벡터의 코사인 유사도는 내적과 같습니다:

$$
\text{cosine}(\mathbf{u}_{\text{norm}}, \mathbf{v}_{\text{norm}}) = \mathbf{u}_{\text{norm}} \cdot \mathbf{v}_{\text{norm}}
$$

**증명:**
$$
\text{cosine}(\mathbf{u}_{\text{norm}}, \mathbf{v}_{\text{norm}}) = \frac{\mathbf{u}_{\text{norm}} \cdot \mathbf{v}_{\text{norm}}}{\|\mathbf{u}_{\text{norm}}\| \|\mathbf{v}_{\text{norm}}\|} = \frac{\mathbf{u}_{\text{norm}} \cdot \mathbf{v}_{\text{norm}}}{1 \cdot 1} = \mathbf{u}_{\text{norm}} \cdot \mathbf{v}_{\text{norm}}
$$

**llmkit 구현:**
```python
# domain/embeddings/utils.py (또는 직접 구현)
def normalize_vector(vec: List[float]) -> List[float]:
    """
    L2 정규화: v_norm = v / ||v||
    
    실제 구현:
    - domain/embeddings/utils.py (또는 직접 구현)
    - NumPy 사용
    """
    v = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(v)  # L2 norm
    if norm == 0:
        return vec
    normalized = v / norm  # 정규화
    return normalized.tolist()
```

---

## Part IV: 고급 기법의 수학적 분석

### 4.1 Hard Negative Mining: Contrastive Learning 이론

#### 정의 4.1.1: Contrastive Learning

**Contrastive Learning**은 유사한 샘플은 가깝게, 다른 샘플은 멀게 배치하는 학습 방법입니다.

**목적 함수:**
$$
\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(q, p^+) / \tau)}{\sum_{i=1}^N \exp(\text{sim}(q, p_i) / \tau)}
$$

여기서:
- $q$: 쿼리 벡터
- $p^+$: Positive 샘플
- $p_i$: Negative 샘플들
- $\tau$: Temperature parameter

#### 시각적 표현: Contrastive Learning 공간

```
임베딩 공간:

        p+ (Positive)
         ★
        /
       /
      q (Query)
       \
        \
         ★ n1 (Hard Negative, sim=0.5)
          \
           \
            ★ n2 (Easy Negative, sim=0.1)
            
목표: q와 p+는 가깝게, q와 n들은 멀게
```

#### 구체적 수치 예시

**예시 4.1.1: Contrastive Loss 계산**

쿼리: "고양이 사료"
- Positive: "고양이 먹이" (sim = 0.85)
- Negative 1: "강아지 사료" (sim = 0.55) ← Hard Negative
- Negative 2: "자동차" (sim = 0.10) ← Easy Negative
- Negative 3: "컴퓨터" (sim = 0.05) ← Easy Negative

$\tau = 0.1$ (Temperature)

**단계별 계산:**

1. **분자 (Positive):**
   $$
   \exp(0.85 / 0.1) = \exp(8.5) \approx 4914.77
   $$

2. **분모 (모든 샘플):**
   $$
   \sum = \exp(8.5) + \exp(5.5) + \exp(1.0) + \exp(0.5)
   $$
   $$
   = 4914.77 + 244.69 + 2.72 + 1.65 = 5163.83
   $$

3. **Loss:**
   $$
   \mathcal{L} = -\log \frac{4914.77}{5163.83} = -\log(0.952) \approx 0.049
   $$

**해석:** Hard Negative가 있으면 Loss가 증가하여 학습이 더 효과적입니다.

#### 정의 4.1.2: Hard Negative

**Hard Negative**는 다음 조건을 만족하는 샘플입니다:

$$
\tau_{\min} < \text{sim}(q, n) < \tau_{\max}
$$

여기서:
- $\tau_{\min}$: 최소 유사도 임계값 (예: 0.3)
- $\tau_{\max}$: 최대 유사도 임계값 (예: 0.7)

#### 시각적 분류

```
유사도 분포:

1.0 |                    ★ (Positive, sim > 0.7)
    |
0.7 |━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    |         ★★★ (Hard Negative, 0.3 < sim < 0.7)
0.3 |━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    | ★★★★★★★★★ (Easy Negative, sim < 0.3)
0.0 |________________________________________

학습 효과:
- Easy Negative: 너무 달라서 학습에 도움 안 됨
- Hard Negative: 비슷하지만 다름 → 학습에 중요!
- Positive: 같음 → 제외
```

#### 구체적 예시

**예시 4.1.2: Hard Negative 찾기**

쿼리: "고양이 사료"
후보들:
1. "강아지 사료" → sim = 0.55 → **Hard Negative** ✓
2. "고양이 장난감" → sim = 0.45 → **Hard Negative** ✓
3. "고양이 먹이" → sim = 0.82 → Positive (제외)
4. "자동차" → sim = 0.12 → Easy Negative
5. "고양이 건강" → sim = 0.38 → **Hard Negative** ✓

**llmkit 구현:**
```python
# domain/embeddings/utils.py (또는 직접 구현)
def find_hard_negatives(
    query_vec: List[float],
    candidate_vecs: List[List[float]],
    similarity_threshold: tuple = (0.3, 0.7),  # (τ_min, τ_max)
    top_k: Optional[int] = None,
) -> List[int]:
    """
    Hard Negative Mining: N_hard = {n_i | τ_min < sim(q, n_i) < τ_max}
    
    실제 구현:
    - domain/embeddings/utils.py (또는 직접 구현)
    - batch_cosine_similarity() 사용
    """
    similarities = batch_cosine_similarity(query_vec, candidate_vecs)
    min_sim, max_sim = similarity_threshold
    # Hard Negative: τ_min < sim < τ_max
    hard_neg_indices = [
        i for i, sim in enumerate(similarities)
        if min_sim < sim < max_sim
    ]
    return hard_neg_indices
```

**실제 사용 예시:**
```python
from llmkit.embeddings import embed_sync, find_hard_negatives

# 쿼리
query = "고양이 사료"
query_vec = embed_sync([query])[0]

# 후보들
candidates = [
    "강아지 사료",      # Hard Negative 예상
    "고양이 장난감",    # Hard Negative 예상
    "고양이 먹이",      # Positive (제외)
    "자동차",           # Easy Negative
    "고양이 건강"       # Hard Negative 예상
]
candidate_vecs = embed_sync(candidates)

# Hard Negative 찾기
hard_neg_indices = find_hard_negatives(
    query_vec,
    candidate_vecs,
    similarity_threshold=(0.3, 0.7),
    top_k=3
)

print("Hard Negatives:")
for idx in hard_neg_indices:
    print(f"  - {candidates[idx]}")

# 출력:
# Hard Negatives:
#   - 강아지 사료
#   - 고양이 장난감
#   - 고양이 건강
```

#### 정리 4.1.1: Hard Negative의 학습 효과

Hard Negative를 사용하면 모델이 더 세밀한 구분을 학습합니다:

$$
\nabla_\theta \mathcal{L}_{\text{hard}} > \nabla_\theta \mathcal{L}_{\text{easy}}
$$

**증명 스케치:**
Hard Negative는 gradient가 더 크므로 학습에 더 효과적입니다.

---

### 4.2 MMR: 정보 이론적 관점

#### 정의 4.2.1: Maximal Marginal Relevance (MMR)

**MMR**은 관련성과 다양성을 균형있게 고려합니다:

$$
\text{MMR} = \arg\max_{d \in \mathcal{D} \setminus S} \left[ \lambda \cdot \text{sim}(q, d) - (1-\lambda) \cdot \max_{d' \in S} \text{sim}(d, d') \right]
$$

여기서:
- $q$: 쿼리
- $d$: 후보 문서
- $S$: 이미 선택된 문서 집합
- $\lambda$: 관련성 가중치 (0-1)

#### 정보 이론적 해석

**Mutual Information 관점:**

$$
I(q; d) = H(q) - H(q|d)
$$

MMR은 다음을 최대화합니다:

$$
\lambda \cdot I(q; d) - (1-\lambda) \cdot I(d; S)
$$

**llmkit 구현:**
```python
# domain/vector_stores/search.py: SearchAlgorithms.mmr_search()
def mmr_search(
    query_vec: List[float],
    candidate_vecs: List[List[float]],
    k: int = 5,
    lambda_param: float = 0.6,  # λ
) -> List[int]:
    """
    MMR 검색: argmax_i [λ·sim(q, c_i) - (1-λ)·max_j∈S sim(c_i, c_j)]
    
    실제 구현:
    - domain/vector_stores/search.py: SearchAlgorithms.mmr_search()
    - vector_stores/search.py: SearchAlgorithms.mmr_search() (레거시)
    """
    # 관련성 점수
    relevance = query_similarities[idx]
    
    # 다양성 점수 (이미 선택된 것과의 최대 유사도)
    diversity = max(candidate_sims) if candidate_sims else 0.0
    
    # MMR 점수
    mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
```

#### 정리 4.2.1: MMR의 최적성

MMR은 다음 최적화 문제를 해결합니다:

$$
\max_{S, |S|=k} \left[ \lambda \sum_{d \in S} \text{sim}(q, d) - (1-\lambda) \sum_{d_i, d_j \in S, i \neq j} \text{sim}(d_i, d_j) \right]
$$

**증명:** Greedy 알고리즘으로 근사 최적해를 찾습니다.

---

### 4.3 Query Expansion: 확률 모델

#### 정의 4.3.1: Query Expansion

**Query Expansion**은 쿼리를 유사어로 확장합니다:

$$
Q_{\text{expanded}} = Q \cup \{w | \text{sim}(E(Q), E(w)) > \tau\}
$$

#### 확률 모델

**Language Model 관점:**

$$
P(w | Q) = \frac{\exp(\text{sim}(E(Q), E(w)) / \tau)}{\sum_{w' \in V} \exp(\text{sim}(E(Q), E(w')) / \tau)}
$$

**llmkit 구현:**
```python
# domain/embeddings/utils.py (또는 직접 구현)
def query_expansion(
    query: str,
    embedding: BaseEmbedding,
    expansion_candidates: Optional[List[str]] = None,
    similarity_threshold: float = 0.7,  # τ
) -> List[str]:
    """
    Query Expansion: Q_exp = {w | sim(E(Q), E(w)) > τ}
    
    실제 구현:
    - domain/embeddings/utils.py (또는 직접 구현)
    - batch_cosine_similarity() 사용
    """
    query_vec = embedding.embed_sync([query])[0]
    candidate_vecs = embedding.embed_sync(expansion_candidates)
    similarities = batch_cosine_similarity(query_vec, candidate_vecs)
    
    # 임계값 이상만 추가
    for candidate, sim in candidate_with_sim:
        if sim >= similarity_threshold:
            expanded.append(candidate)
```

---

## Part V: 구현과 최적화

### 5.1 배치 처리의 계산 복잡도

#### 정리 5.1.1: 배치 코사인 유사도의 복잡도

**시간 복잡도:**
- 단일 쿼리: $O(d)$ (차원 수)
- 배치 처리: $O(n \cdot d)$ (n: 후보 수, d: 차원)

**공간 복잡도:**
- $O(n \cdot d)$ (모든 벡터 저장)

**llmkit 구현:**
```python
# domain/embeddings/utils.py (또는 직접 구현)
def batch_cosine_similarity(
    query_vec: List[float],
    candidate_vecs: List[List[float]]
) -> List[float]:
    """
    배치 코사인 유사도 계산: O(n·d)
    
    실제 구현:
    - domain/embeddings/utils.py (또는 직접 구현)
    - NumPy 벡터화 연산으로 효율적 계산
    """
    # NumPy 벡터화 연산으로 효율적 계산
    query = np.array(query_vec, dtype=np.float32)
    candidates = np.array(candidate_vecs, dtype=np.float32)
    
    # O(n·d) 시간 복잡도
    similarities = np.dot(candidates, query) / (candidate_norms.flatten() * query_norm)
```

#### 최적화 기법

**1. 벡터화 (Vectorization)**
- NumPy의 행렬 연산 활용
- Python 루프 대신 C 레벨 연산

**2. 메모리 효율성**
- `dtype=np.float32` 사용 (메모리 50% 절감)

---

### 5.2 캐싱 전략과 시간 복잡도

#### 정의 5.2.1: LRU 캐시 (Least Recently Used)

**LRU 캐시**는 가장 오래 사용되지 않은 항목을 제거합니다.

**시간 복잡도:**
- 조회: $O(1)$ (해시 테이블)
- 삽입: $O(1)$ (OrderedDict)

**llmkit 구현:**
```python
# domain/embeddings/cache.py: EmbeddingCache
class EmbeddingCache:
    """
    LRU + TTL 캐시: 가장 오래 사용되지 않은 항목 제거
    
    실제 구현:
    - domain/embeddings/cache.py: EmbeddingCache
    - OrderedDict 사용 (LRU 구현)
    """
    def __init__(self, ttl: int = 3600, max_size: int = 10000):
        self.cache: OrderedDict[str, tuple[List[float], float]] = OrderedDict()
        # LRU: OrderedDict 사용
    
    def get(self, text: str) -> Optional[List[float]]:
        """
        캐시 조회: O(1)
        
        실제 구현:
        - domain/embeddings/cache.py: EmbeddingCache.get()
        """
        # O(1) 조회
        if text in self.cache:
            self.cache.move_to_end(text)  # LRU 업데이트
            return vector
    
    def set(self, text: str, vector: List[float]):
        """
        캐시 저장: O(1)
        
        실제 구현:
        - domain/embeddings/cache.py: EmbeddingCache.set()
        """
        # O(1) 삽입
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # 가장 오래된 항목 제거
        self.cache[text] = (vector, timestamp)
```

#### 정리 5.2.1: 캐시 히트율

**캐시 히트율 (Hit Rate):**

$$
\text{Hit Rate} = \frac{\text{Cache Hits}}{\text{Total Requests}}
$$

**비용 절감:**

$$
\text{Cost Savings} = \text{Hit Rate} \times \text{API Cost per Request}
$$

---

## 참고 문헌

1. **Mikolov et al. (2013)**: "Efficient Estimation of Word Representations in Vector Space"
2. **Devlin et al. (2018)**: "BERT: Pre-training of Deep Bidirectional Transformers"
3. **Reimers & Gurevych (2019)**: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
4. **Johnson & Lindenstrauss (1984)**: "Extensions of Lipschitz mappings into a Hilbert space"

---

**작성일**: 2025-01-XX  
**버전**: 2.0 (석사 수준 확장)
