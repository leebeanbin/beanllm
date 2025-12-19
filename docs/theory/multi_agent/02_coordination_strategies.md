# Coordination Strategies: 조정 전략

**석사 수준 이론 문서**  
**CS 관점 + 수학적 엄밀성**  
**기반**: llmkit MultiAgentSystem 실제 구현 분석

---

## 목차

1. [순차 실행의 함수 합성](#1-순차-실행의-함수-합성)
2. [병렬 실행과 속도 향상](#2-병렬-실행과-속도-향상)
3. [계층적 구조와 트리 이론](#3-계층적-구조와-트리-이론)
4. [속도 향상 분석](#4-속도-향상-분석)
5. [CS 관점: 구현과 성능](#5-cs-관점-구현과-성능)

---

## 1. 순차 실행의 함수 합성

### 1.1 순차 실행

#### 정의 1.1.1: 순차 실행

**순차 실행**은 함수 합성으로 표현됩니다:

$$
\text{result} = f_n \circ f_{n-1} \circ \cdots \circ f_2 \circ f_1(\text{task})
$$

**시간 복잡도:**

$$
T_{\text{sequential}} = \sum_{i=1}^n T_i
$$

#### 시각적 표현: 순차 실행

```
순차 실행:

시간 →
Agent1: ████████ (8초)
Agent2:         ████████████ (12초)
Agent3:                 ████████ (8초)
───────────────────────────────────────────────
총 시간: 8 + 12 + 8 = 28초
```

---

## 2. 병렬 실행과 속도 향상

### 2.1 병렬 실행

#### 정의 2.1.1: 병렬 실행

**병렬 실행:**

$$
\text{results} = \{f_1(\text{task}), f_2(\text{task}), \ldots, f_n(\text{task})\} \text{ (동시 실행)}
$$

**시간 복잡도:**

$$
T_{\text{parallel}} = \max(T_1, T_2, \ldots, T_n)
$$

#### 시각적 표현: 병렬 실행

```
병렬 실행:

시간 →
Agent1: ████████ (8초)
Agent2: ████████████ (12초)
Agent3: ████████ (8초)
───────────────────────────────────────────────
총 시간: max(8, 12, 8) = 12초

속도 향상: S = 28 / 12 = 2.33배
```

### 2.2 속도 향상

#### 정리 2.2.1: 속도 향상 (Speedup)

**속도 향상:**

$$
S = \frac{T_{\text{sequential}}}{T_{\text{parallel}}} = \frac{\sum_{i=1}^n T_i}{\max(T_1, T_2, \ldots, T_n)}
$$

**이상적 경우:** $S = n$ (모든 에이전트가 같은 시간)

#### 구체적 수치 예시

**예시 2.2.1: 속도 향상 계산**

에이전트 3개:
- Agent1: $T_1 = 8$초
- Agent2: $T_2 = 12$초
- Agent3: $T_3 = 8$초

**순차:**
$$
T_{\text{seq}} = 8 + 12 + 8 = 28 \text{초}
$$

**병렬:**
$$
T_{\text{par}} = \max(8, 12, 8) = 12 \text{초}
$$

**속도 향상:**
$$
S = \frac{28}{12} = 2.33 \text{배}
$$

---

## 3. 계층적 구조와 트리 이론

### 3.1 계층적 구조

#### 정의 3.1.1: 계층적 구조

**계층적 구조**는 트리로 표현됩니다:

$$
T = (V, E, \text{root})
$$

**레벨:**
- Level 0: Manager
- Level 1: Workers
- Level 2: Sub-workers

#### 시각적 표현: 계층적 구조

```
계층적 구조:

        Manager (Level 0)
         │
    ┌────┴────┐
    │         │
Worker1    Worker2 (Level 1)
 │           │
 │      ┌────┴────┐
 │      │         │
Sub1   Sub2     Sub3 (Level 2)
```

---

## 4. 속도 향상 분석

### 4.1 효율성

#### 정의 4.1.1: 효율성 (Efficiency)

**효율성:**

$$
E = \frac{S}{n} = \frac{T_{\text{seq}}}{n \cdot T_{\text{par}}}
$$

**범위:** $[0, 1]$
- $E = 1$: 이상적 (완전 병렬)
- $E < 1$: 오버헤드 존재

---

## 5. CS 관점: 구현과 성능

### 5.1 llmkit 구현

#### 구현 5.1.1: 병렬 실행

```python
# multi_agent.py: Line 233-270
class ParallelStrategy(CoordinationStrategy):
    async def execute(self, agents, task, **kwargs):
        """
        병렬 실행: T_par = max(T₁, T₂, ..., Tₙ)
        """
        tasks = [agent.run(task) for agent in agents]
        results = await asyncio.gather(*tasks)
        return self._aggregate(results)
```

**시간 복잡도:** $O(\max(T_1, T_2, \ldots, T_n))$

---

## 질문과 답변 (Q&A)

### Q1: 언제 순차, 언제 병렬?

**A:** 선택 기준:

**순차 실행:**
- 이전 결과가 다음 입력
- 의존성 있음
- 예: 연구 → 작성 → 검토

**병렬 실행:**
- 독립적인 작업
- 의존성 없음
- 예: 여러 소스 검색

---

## 참고 문헌

1. **Wooldridge (2009)**: "An Introduction to MultiAgent Systems"

---

**작성일**: 2025-01-XX  
**버전**: 3.0 (CS 관점 + 석사 수준 확장)

