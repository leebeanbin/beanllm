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

**llmkit 구현:**
```python
# domain/multi_agent/strategies.py: SequentialStrategy, ParallelStrategy, HierarchicalStrategy
# service/impl/multi_agent_service_impl.py: MultiAgentServiceImpl
# facade/multi_agent_facade.py: MultiAgentCoordinator
from abc import ABC, abstractmethod
import asyncio

class CoordinationStrategy(ABC):
    """
    조정 전략 베이스 클래스
    
    실제 구현:
    - domain/multi_agent/strategies.py: CoordinationStrategy (추상 클래스)
    - domain/multi_agent/strategies.py: SequentialStrategy, ParallelStrategy, HierarchicalStrategy
    - service/impl/multi_agent_service_impl.py: MultiAgentServiceImpl (전략 실행)
    """
    @abstractmethod
    async def execute(self, agents: List[Any], task: str, **kwargs) -> Dict[str, Any]:
        """전략 실행"""
        pass

class SequentialStrategy(CoordinationStrategy):
    """
    순차 실행 전략: result = fₙ ∘ fₙ₋₁ ∘ ... ∘ f₁(task)
    
    수학적 표현:
    - 함수 합성: result = fₙ ∘ fₙ₋₁ ∘ ... ∘ f₂ ∘ f₁(task)
    - 시간 복잡도: T_sequential = Σ T_i
    
    실제 구현:
    - domain/multi_agent/strategies.py: SequentialStrategy
    - service/impl/multi_agent_service_impl.py: MultiAgentServiceImpl.execute_sequential()
    """
    async def execute(self, agents: List[Any], task: str, **kwargs) -> Dict[str, Any]:
        """
        순차 실행
        
        Process:
        1. Agent 1 실행: result₁ = f₁(task)
        2. Agent 2 실행: result₂ = f₂(result₁)
        3. Agent 3 실행: result₃ = f₃(result₂)
        ...
        n. Agent n 실행: resultₙ = fₙ(resultₙ₋₁)
        
        시간: T = T₁ + T₂ + ... + Tₙ
        """
        results = []
        current_input = task
        
        for i, agent in enumerate(agents):
            result = await agent.run(current_input)
            results.append(result)
            
            # 다음 agent의 입력은 이전 agent의 출력
            current_input = result.answer
        
        return {
            "final_result": results[-1].answer if results else None,
            "intermediate_results": [r.answer for r in results],
            "all_steps": results,
            "strategy": "sequential",
        }

class ParallelStrategy(CoordinationStrategy):
    """
    병렬 실행 전략: results = {f₁(task), f₂(task), ..., fₙ(task)} (동시 실행)
    
    수학적 표현:
    - 동시 실행: results = {f₁(task), f₂(task), ..., fₙ(task)}
    - 시간 복잡도: T_parallel = max(T₁, T₂, ..., Tₙ)
    - 속도 향상: S = T_sequential / T_parallel = Σ T_i / max(T_i)
    
    실제 구현:
    - domain/multi_agent/strategies.py: ParallelStrategy
    - service/impl/multi_agent_service_impl.py: MultiAgentServiceImpl.execute_parallel()
    - asyncio.gather() 사용
    """
    def __init__(self, aggregation: str = "concatenate"):
        """
        Args:
            aggregation: 결과 집계 방법 ("concatenate", "vote", "average")
        """
        self.aggregation = aggregation
    
    async def execute(self, agents: List[Any], task: str, **kwargs) -> Dict[str, Any]:
        """
        병렬 실행
        
        Process:
        1. 모든 agent를 동시에 실행: asyncio.gather()
        2. 결과 집계: aggregation(results)
        
        시간: T = max(T₁, T₂, ..., Tₙ)
        속도 향상: S = (T₁ + T₂ + ... + Tₙ) / max(T₁, T₂, ..., Tₙ)
        """
        # 모든 agent를 동시에 실행
        tasks = [agent.run(task) for agent in agents]
        results = await asyncio.gather(*tasks)
        
        # 결과 집계
        if self.aggregation == "concatenate":
            final_result = "\n\n".join([r.answer for r in results])
        elif self.aggregation == "vote":
            # 투표 기반 집계
            final_result = max(set([r.answer for r in results]), key=[r.answer for r in results].count)
        else:
            final_result = results[0].answer if results else None
        
        return {
            "final_result": final_result,
            "all_results": [r.answer for r in results],
            "strategy": "parallel",
            "aggregation": self.aggregation,
        }

class HierarchicalStrategy(CoordinationStrategy):
    """
    계층적 실행 전략: Manager → Workers
    
    수학적 표현:
    - 트리 구조: manager (root) → {worker₁, worker₂, ..., workerₙ} (leaves)
    - 시간 복잡도: T_hierarchical = T_manager + max(T_worker_i)
    
    실제 구현:
    - domain/multi_agent/strategies.py: HierarchicalStrategy
    - service/impl/multi_agent_service_impl.py: MultiAgentServiceImpl.execute_hierarchical()
    """
    def __init__(self, manager_agent: Any):
        """
        Args:
            manager_agent: 매니저 역할 agent
        """
        self.manager = manager_agent
    
    async def execute(self, agents: List[Any], task: str, **kwargs) -> Dict[str, Any]:
        """
        계층적 실행
        
        Process:
        1. Manager가 작업 분해: subtasks = Manager.decompose(task)
        2. Workers 병렬 실행: results = {Worker₁(subtask₁), ..., Workerₙ(subtaskₙ)}
        3. Manager가 결과 종합: final = Manager.synthesize(results)
        
        시간: T = T_decompose + max(T_worker_i) + T_synthesize
        """
        # 1. Manager가 작업 분해
        delegation_prompt = f"""Break down this task into {len(agents)} subtasks.
Task: {task}
Return JSON: {{"subtasks": ["subtask1", "subtask2", ...]}}"""
        
        delegation_result = await self.manager.run(delegation_prompt)
        
        # JSON 파싱
        import json
        import re
        json_match = re.search(r"\{.*\}", delegation_result.answer, re.DOTALL)
        if json_match:
            subtasks_data = json.loads(json_match.group())
            subtasks = subtasks_data.get("subtasks", [])
        else:
            subtasks = [task] * len(agents)
        
        # 2. Workers 병렬 실행
        worker_tasks = [agent.run(subtask) for agent, subtask in zip(agents, subtasks)]
        worker_results = await asyncio.gather(*worker_tasks)
        
        # 3. Manager가 결과 종합
        synthesis_prompt = f"""Synthesize these results into a final answer.
Results: {[r.answer for r in worker_results]}
Original task: {task}"""
        
        final_result = await self.manager.run(synthesis_prompt)
        
        return {
            "final_result": final_result.answer,
            "subtasks": subtasks,
            "worker_results": [r.answer for r in worker_results],
            "strategy": "hierarchical",
        }
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

