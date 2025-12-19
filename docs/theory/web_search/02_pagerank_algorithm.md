# PageRank Algorithm: 페이지랭크 알고리즘

**석사 수준 이론 문서**  
**CS 관점 + 수학적 엄밀성**  
**기반**: 웹 검색 이론

---

## 목차

1. [PageRank의 수학적 정의](#1-pagerank의-수학적-정의)
2. [반복 알고리즘](#2-반복-알고리즘)
3. [수렴 분석](#3-수렴-분석)
4. [CS 관점: 구현과 최적화](#4-cs-관점-구현과-최적화)

---

## 1. PageRank의 수학적 정의

### 1.1 PageRank 공식

#### 정의 1.1.1: PageRank

**PageRank**는 웹페이지의 중요도를 계산합니다:

$$
\text{PR}(p) = (1-d) + d \times \sum_{p_i \in M(p)} \frac{\text{PR}(p_i)}{L(p_i)}
$$

여기서:
- $d$: Damping factor (보통 0.85)
- $M(p)$: 페이지 $p$로 링크하는 페이지 집합
- $L(p_i)$: 페이지 $p_i$의 외부 링크 수

### 1.2 행렬 표현

#### 정의 1.2.1: PageRank 행렬

**PageRank는 행렬로 표현됩니다:**

$$
\mathbf{PR} = (1-d)\mathbf{1} + d \times M \times \mathbf{PR}
$$

여기서 $M$은 전이 행렬입니다.

---

## 2. 반복 알고리즘

### 2.1 Power Iteration

#### 알고리즘 2.1.1: PageRank 계산

```
Algorithm: PageRank(graph, d, epsilon)
Input:
  - graph: 링크 그래프
  - d: damping factor
  - epsilon: 수렴 임계값
Output: PageRank 벡터

1. PR ← [1/N, 1/N, ..., 1/N]  // 초기화
2. 
3. while not converged:
4.     PR_new ← (1-d)/N + d × M × PR
5.     if ||PR_new - PR|| < epsilon:
6.         break
7.     PR ← PR_new
8. 
9. return PR
```

**시간 복잡도:** $O(k \cdot (V + E))$ ($k$ = 반복 횟수)

---

## 3. 수렴 분석

### 3.1 수렴 조건

#### 정리 3.1.1: PageRank 수렴

**PageRank는 항상 수렴합니다:**

$$
\lim_{k \to \infty} \mathbf{PR}^{(k)} = \mathbf{PR}^*
$$

**증명:** Perron-Frobenius 정리

---

## 4. CS 관점: 구현과 최적화

### 4.1 희소 행렬 최적화

#### CS 관점 4.1.1: 희소 행렬

**웹 그래프는 희소합니다:**

- 대부분의 페이지는 소수의 링크만 가짐
- 희소 행렬 표현 사용
- 메모리 효율적

---

## 질문과 답변 (Q&A)

### Q1: Damping factor는 왜 0.85인가요?

**A:** 0.85 선택 이유:

1. **실험적 검증:**
   - 다양한 웹 그래프에서 테스트
   - 0.85가 최적 성능

2. **수학적 해석:**
   - 너무 작으면: 랜덤 서핑 비중 높음
   - 너무 크면: 링크 구조 무시
   - 0.85: 균형

---

## 참고 문헌

1. **Page et al. (1998)**: "The PageRank Citation Ranking"

---

**작성일**: 2025-01-XX  
**버전**: 3.0 (CS 관점 + 석사 수준 확장)

