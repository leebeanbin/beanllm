# Conditional Routing and Cycles: 조건부 라우팅과 순환

**석사 수준 이론 문서**  
**CS 관점 + 수학적 엄밀성**  
**기반**: llmkit Graph 실제 구현 분석

---

## 목차

1. [조건부 엣지의 수학적 모델](#1-조건부-엣지의-수학적-모델)
2. [조건 함수의 정의](#2-조건-함수의-정의)
3. [순환 감지 알고리즘](#3-순환-감지-알고리즘)
4. [고정점 이론](#4-고정점-이론)
5. [동적 분기](#5-동적-분기)
6. [CS 관점: 구현과 최적화](#6-cs-관점-구현과-최적화)

---

## 1. 조건부 엣지의 수학적 모델

### 1.1 조건부 전이

#### 정의 1.1.1: 조건부 엣지

**조건부 엣지**는 상태에 따라 다음 노드를 결정합니다:

$$
\text{next}(v, \text{state}) = \begin{cases}
v_A & \text{if } \text{condition}_A(\text{state}) \\
v_B & \text{if } \text{condition}_B(\text{state}) \\
\vdots \\
v_n & \text{otherwise}
\end{cases}
$$

#### 시각적 표현: 조건부 라우팅

```
조건부 라우팅:

    node1
      │
      ▼
┌─────────────┐
│  condition  │  ← 상태 평가
└──────┬──────┘
       │
   ┌───┴───┐
   │       │
   ▼       ▼
node2   node3
(true)  (false)
```

### 1.2 llmkit 구현

#### 구현 1.2.1: Conditional Edge

**llmkit 구현:**
```python
# domain/graph/nodes.py: ConditionalNode
# facade/state_graph_facade.py: StateGraph.add_conditional_edge()
# service/impl/state_graph_service_impl.py: StateGraphServiceImpl._get_next_node()
from abc import ABC

class ConditionalNode(BaseNode):
    """
    조건부 노드: 조건에 따라 다른 노드로 라우팅
    
    수학적 표현:
    - next(v, state) = v_A if condition_A(state) else v_B
    - condition: State → {True, False}
    
    실제 구현:
    - domain/graph/nodes.py: ConditionalNode
    - facade/state_graph_facade.py: StateGraph.add_conditional_edge()
    - service/impl/state_graph_service_impl.py: StateGraphServiceImpl._get_next_node()
    """
    def __init__(
        self,
        name: str,
        condition: Callable[[GraphState], bool],
        true_node: Optional[BaseNode] = None,
        false_node: Optional[BaseNode] = None
    ):
        """
        Args:
            name: 노드 이름
            condition: 조건 함수 c: State → {True, False}
            true_node: 조건이 True일 때 이동할 노드
            false_node: 조건이 False일 때 이동할 노드
        """
        super().__init__(name)
        self.condition = condition
        self.true_node = true_node
        self.false_node = false_node
    
    async def execute(self, state: GraphState) -> Dict[str, Any]:
        """
        조건 평가 및 노드 실행
        
        Process:
        1. 조건 평가: result = condition(state)
        2. 노드 선택: selected = true_node if result else false_node
        3. 선택된 노드 실행
        
        수학적 표현:
        - result = c(state)
        - selected = result ? v_true : v_false
        - output = selected.execute(state)
        """
        condition_result = self.condition(state)
        selected_node = self.true_node if condition_result else self.false_node
        
        if selected_node:
            result = await selected_node.execute(state)
            return result
        return {}
```

**조건부 엣지 추가:**
```python
# facade/state_graph_facade.py: StateGraph.add_conditional_edge()
class StateGraph:
    def add_conditional_edge(
        self,
        from_node: str,
        condition: Callable[[Dict[str, Any]], str],
        edge_mapping: Optional[Dict[str, str]] = None,
    ):
        """
        조건부 엣지 추가: next(v, state) = condition(state)
        
        수학적 표현:
        - 조건 함수: c: State → NodeName
        - 전이: next = c(state)
        
        실제 구현:
        - facade/state_graph_facade.py: StateGraph.add_conditional_edge()
        - service/impl/state_graph_service_impl.py: StateGraphServiceImpl._get_next_node()
        """
        self.conditional_edges[from_node] = (condition, edge_mapping)
```

**순환 감지:**
```python
# service/impl/graph_service_impl.py: GraphServiceImpl.run_graph()
# service/impl/state_graph_service_impl.py: StateGraphServiceImpl.invoke()
class StateGraphServiceImpl:
    async def invoke(self, request: StateGraphRequest) -> StateGraphResponse:
        """
        그래프 실행 (순환 감지 포함)
        
        순환 감지 알고리즘:
        - visited: 방문한 노드 집합
        - 순환: visited에 이미 있는 노드 재방문 시 감지
        
        시간 복잡도: O(V + E) (DFS)
        
        실제 구현:
        - service/impl/state_graph_service_impl.py: StateGraphServiceImpl.invoke()
        - service/impl/graph_service_impl.py: GraphServiceImpl.run_graph()
        """
        visited = set()
        current_node = request.entry_point
        state = request.initial_state
        iteration = 0
        
        while current_node and iteration < request.max_iterations:
            # 순환 감지
            if current_node in visited:
                logger.warning(f"Cycle detected: {current_node} already visited")
                break
            
            visited.add(current_node)
            
            # 노드 실행
            node_func = request.nodes[current_node]
            state = node_func(state)
            
            # 다음 노드 결정 (조건부 엣지 우선)
            current_node = self._get_next_node(
                current_node,
                state,
                request.edges or {},
                request.conditional_edges or {},
                request.nodes or {},
            )
            
            iteration += 1
        
        return StateGraphResponse(final_state=state)
```

---

## 2. 조건 함수의 정의

### 2.1 불린 조건

#### 정의 2.1.1: 불린 조건 함수

**불린 조건 함수:**

$$
c: \text{State} \rightarrow \{\text{True}, \text{False}\}
$$

**예시:**
```python
def should_refine(state):
    grade = state.get("grade", 0)
    return grade < 0.7
```

### 2.2 다중 분기

#### 정의 2.2.1: 다중 분기

**다중 분기 조건:**

$$
\text{next}(v, \text{state}) = \begin{cases}
v_1 & \text{if } c_1(\text{state}) \\
v_2 & \text{if } c_2(\text{state}) \\
\vdots \\
v_n & \text{otherwise}
\end{cases}
$$

---

## 3. 순환 감지 알고리즘

### 3.1 DFS 기반 감지

#### 알고리즘 3.1.1: Cycle Detection

```
Algorithm: HasCycle(graph)
Input: 그래프 G = (V, E)
Output: 사이클 존재 여부

1. visited ← {}
2. rec_stack ← {}  // 재귀 스택
3. 
4. function DFS(node):
5.     visited.add(node)
6.     rec_stack.add(node)
7.     
8.     for neighbor in graph[node]:
9.         if neighbor not in visited:
10.            if DFS(neighbor):
11.                return True
12.        elif neighbor in rec_stack:
13.            return True  // 백 엣지 발견 → 사이클
14.    
15.    rec_stack.remove(node)
16.    return False
17. 
18. for node in V:
19.     if node not in visited:
20.         if DFS(node):
21.             return True
22. 
23. return False
```

**시간 복잡도:** $O(V + E)$

### 3.2 llmkit의 순환 처리

#### 구현 3.2.1: 무한 루프 방지

**llmkit 구현:**
```python
# service/impl/state_graph_service_impl.py: StateGraphServiceImpl.invoke()
# service/impl/graph_service_impl.py: GraphServiceImpl.run_graph()
async def invoke(self, request: StateGraphRequest) -> StateGraphResponse:
    """
    그래프 실행 (순환 감지 포함)
    
    순환 감지 알고리즘:
    - visited: 방문한 노드 집합
    - 순환: visited에 이미 있는 노드 재방문 시 감지
    
    시간 복잡도: O(V + E) (DFS)
    
    실제 구현:
    - service/impl/state_graph_service_impl.py: StateGraphServiceImpl.invoke()
    - service/impl/graph_service_impl.py: GraphServiceImpl.run_graph()
    """
    max_iterations = request.max_iterations or 100  # 무한 루프 방지
    visited = set()
    
    for iteration in range(max_iterations):
        if current_node in visited:
            logger.warning(f"Node {current_node} already visited")
            break
        
        visited.add(current_node)
        # 노드 실행
        # ...
```

---

## 4. 고정점 이론

### 4.1 고정점의 정의

#### 정의 4.1.1: 고정점 (Fixed Point)

**고정점**은 다음을 만족하는 상태입니다:

$$
f(\text{state}) = \text{state}
$$

**해석:**
- 상태 전이 후 상태가 변하지 않음
- 수렴 상태
- 종료 조건

### 4.2 고정점 수렴

#### 정리 4.2.1: 고정점 수렴

**조건부 전이가 고정점으로 수렴하는 조건:**

$$
\lim_{n \to \infty} f^n(\text{state}_0) = \text{state}^*
$$

여기서 $f(\text{state}^*) = \text{state}^*$입니다.

---

## 5. 동적 분기

### 5.1 확률적 라우팅

#### 정의 5.1.1: 확률적 분기

**확률적 분기:**

$$
P(\text{next} = v_i | \text{state}) = p_i(\text{state})
$$

여기서 $\sum_i p_i = 1$입니다.

---

## 6. CS 관점: 구현과 최적화

### 6.1 조건 평가 최적화

#### CS 관점 6.1.1: 조건 캐싱

**조건 결과 캐싱:**

```python
@lru_cache(maxsize=1000)
def cached_condition(state_hash):
    return condition(state)
```

**효과:**
- 동일 상태 재평가 방지
- 성능 향상

---

## 질문과 답변 (Q&A)

### Q1: 조건부 라우팅은 언제 사용하나요?

**A:** 사용 시기:

1. **품질 검사:**
   - 결과 품질 평가
   - 재처리 필요 여부

2. **에러 처리:**
   - 성공/실패 분기
   - 재시도 로직

3. **동적 워크플로우:**
   - 입력에 따른 분기
   - 조건부 처리

---

## 참고 문헌

1. **Cormen et al. (2009)**: "Introduction to Algorithms" - 그래프 알고리즘
2. **Tarjan (1972)**: "Depth-first search and linear graph algorithms"

---

**작성일**: 2025-01-XX  
**버전**: 3.0 (CS 관점 + 석사 수준 확장)

