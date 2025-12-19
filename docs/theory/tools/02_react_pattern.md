# ReAct Pattern: 추론과 행동의 결합

**석사 수준 이론 문서**  
**CS 관점 + 수학적 엄밀성**  
**기반**: llmkit Agent 실제 구현 분석

---

## 목차

1. [ReAct 패턴의 정의](#1-react-패턴의-정의)
2. [Thought-Action-Observation 사이클](#2-thought-action-observation-사이클)
3. [도구 선택의 확률 모델](#3-도구-선택의-확률-모델)
4. [수렴 조건](#4-수렴-조건)
5. [CS 관점: 구현과 최적화](#5-cs-관점-구현과-최적화)

---

## 1. ReAct 패턴의 정의

### 1.1 ReAct의 구성

#### 정의 1.1.1: ReAct (Reasoning + Acting)

**ReAct 패턴**은 추론과 행동을 결합합니다:

$$
\text{ReAct} = \text{Reasoning} + \text{Acting}
$$

**단계:**
1. **Thought**: 추론 (어떤 도구 사용할지)
2. **Action**: 행동 (도구 실행)
3. **Observation**: 관찰 (결과 확인)

#### 시각적 표현: ReAct 사이클

```
ReAct 사이클:

Task: "서울의 날씨는?"
    │
    ▼
┌─────────────────────────────────────┐
│ Step 1: Thought                     │
│ "날씨 정보가 필요하므로             │
│  웹 검색 도구를 사용해야 한다"      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Step 2: Action                      │
│ Action: search_weather              │
│ Action Input: {"city": "Seoul"}     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Step 3: Observation                 │
│ "서울: 맑음, 15°C"                  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Step 4: Thought                     │
│ "정보를 얻었으므로 최종 답변 가능"   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Final Answer:                       │
│ "서울의 현재 날씨는 맑고 15°C입니다"│
└─────────────────────────────────────┘
```

---

## 2. Thought-Action-Observation 사이클

### 2.1 사이클의 수학적 모델

#### 정의 2.1.1: T-A-O 사이클

**T-A-O 사이클:**

$$
\text{State}_{t+1} = f(\text{State}_t, \text{Thought}_t, \text{Action}_t, \text{Observation}_t)
$$

**Thought:**
$$
\text{Thought}_t = \text{LLM}(\text{State}_t, \text{Task})
$$

**Action:**
$$
\text{Action}_t = \text{SelectTool}(\text{Thought}_t)
$$

**Observation:**
$$
\text{Observation}_t = \text{ExecuteTool}(\text{Action}_t)
$$

### 2.2 llmkit 구현

#### 구현 2.2.1: ReAct 실행

```python
# agent.py: Line 131-214
async def run(self, task: str) -> AgentResult:
    """
    ReAct 패턴 실행
    """
    state = {"task": task, "history": []}
    
    for step in range(self.max_iterations):
        # Thought
        thought = await self._think(state)
        
        # Action
        action = self._parse_action(thought)
        
        if action:
            # Observation
            observation = await self._execute_tool(action)
            state["history"].append((thought, action, observation))
        else:
            # Final Answer
            return AgentResult(answer=thought)
```

---

## 3. 도구 선택의 확률 모델

### 3.1 도구 선택 확률

#### 정의 3.1.1: 도구 선택

**LLM이 도구를 선택할 확률:**

$$
P(\text{tool}_i | \text{query}) = \text{softmax}([\text{score}_1, \text{score}_2, \ldots, \text{score}_n])_i
$$

**구체적 수치 예시:**

**예시 3.1.1: 도구 선택 확률**

쿼리: "서울의 날씨는?"

가능한 도구:
- `search_weather`: score = 8.5
- `search_web`: score = 6.2
- `calculator`: score = 0.1

**Softmax 계산:**

$$
P(\text{search\_weather}) = \frac{\exp(8.5)}{\exp(8.5) + \exp(6.2) + \exp(0.1)} = 0.909
$$

$$
P(\text{search\_web}) = 0.091
$$

$$
P(\text{calculator}) = 0.0002
$$

**결과:** `search_weather` 선택 (90.9%)

---

## 4. 수렴 조건

### 4.1 종료 조건

#### 정의 4.1.1: 종료 조건

**종료 조건:**

1. **최대 반복:**
   $$
   \text{step} \geq \text{max\_iterations}
   $$

2. **최종 답변:**
   $$
   \text{Action} = \text{None} \implies \text{종료}
   $$

3. **에러:**
   $$
   \text{Error} \implies \text{종료}
   $$

---

## 5. CS 관점: 구현과 성능

### 5.1 반복 제한

#### CS 관점 5.1.1: 무한 루프 방지

**최대 반복 설정:**

```python
max_iterations = 10  # 무한 루프 방지

for step in range(max_iterations):
    # ReAct 사이클
    if should_stop(state):
        break
```

**효과:**
- 무한 루프 방지
- 리소스 보호

---

## 질문과 답변 (Q&A)

### Q1: ReAct는 언제 사용하나요?

**A:** 사용 시기:

1. **복잡한 작업:**
   - 여러 단계 필요
   - 도구 체이닝

2. **동적 결정:**
   - 상황에 따라 다른 도구
   - 조건부 실행

3. **에러 처리:**
   - 실패 시 재시도
   - 대안 경로

---

## 참고 문헌

1. **Yao et al. (2022)**: "ReAct: Synergizing Reasoning and Acting in Language Models"

---

**작성일**: 2025-01-XX  
**버전**: 3.0 (CS 관점 + 석사 수준 확장)

