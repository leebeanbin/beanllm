# Directed Graphs and State Transitions: 유향 그래프와 상태 전이

**석사 수준 이론 문서**  
**CS 관점 + 수학적 엄밀성**  
**기반**: llmkit StateGraph, Graph 실제 구현 분석

---

## 목차

1. [유향 그래프의 수학적 정의](#1-유향-그래프의-수학적-정의)
2. [경로와 사이클](#2-경로와-사이클)
3. [상태 전이 시스템](#3-상태-전이-시스템)
4. [노드와 엣지의 형식적 정의](#4-노드와-엣지의-형식적-정의)
5. [그래프 실행 모델](#5-그래프-실행-모델)
6. [CS 관점: 그래프 표현과 알고리즘](#6-cs-관점-그래프-표현과-알고리즘)

---

## 1. 유향 그래프의 수학적 정의

### 1.1 기본 정의

#### 정의 1.1.1: 유향 그래프 (Directed Graph)

**유향 그래프** $G$는 다음과 같이 정의됩니다:

$$
G = (V, E, s, t)
$$

여기서:
- $V$: 노드 집합 (vertices)
- $E$: 엣지 집합 (edges)
- $s: E \rightarrow V$: 소스 함수 (source)
- $t: E \rightarrow V$: 타겟 함수 (target)

#### 시각적 표현: 유향 그래프

```
┌─────────────────────────────────────────────────────────┐
│                  유향 그래프 예시                        │
└─────────────────────────────────────────────────────────┘

        start
          │
          ▼
    ┌─────────┐
    │  node1  │
    └────┬────┘
         │ e₁
         ▼
    ┌─────────┐
    │  node2  │
    └────┬────┘
         │ e₂
         ├─────────┐
         │         │ e₃
         ▼         ▼
    ┌─────────┐ ┌─────────┐
    │  node3  │ │  node4  │
    └────┬────┘ └────┬────┘
         │ e₄        │ e₅
         └─────┬─────┘
               │
               ▼
          ┌─────────┐
          │   end   │
          └─────────┘

노드: V = {start, node1, node2, node3, node4, end}
엣지: E = {e₁, e₂, e₃, e₄, e₅}
소스: s(e₁) = node1, t(e₁) = node2
```

### 1.2 인접 행렬 표현

#### 정의 1.2.1: 인접 행렬 (Adjacency Matrix)

**인접 행렬** $A$는 다음과 같이 정의됩니다:

$$
A_{ij} = \begin{cases}
1 & \text{if } (v_i, v_j) \in E \\
0 & \text{otherwise}
\end{cases}
$$

#### 구체적 수치 예시

**예시 1.2.1: 인접 행렬**

위 그래프의 인접 행렬:

$$
A = \begin{bmatrix}
0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}
$$

**해석:**
- $A_{01} = 1$: start → node1
- $A_{23} = 1$: node2 → node3
- $A_{24} = 1$: node2 → node4

---

## 2. 경로와 사이클

### 2.1 경로의 정의

#### 정의 2.1.1: 경로 (Path)

**경로**는 다음과 같이 정의됩니다:

$$
P = (v_0, e_1, v_1, e_2, \ldots, e_n, v_n)
$$

여기서 $t(e_i) = s(e_{i+1})$입니다.

#### 정리 2.1.1: 경로의 존재성

**경로 $P$가 존재하는 것은:**

$$
\exists e_1, e_2, \ldots, e_n: t(e_i) = s(e_{i+1}) \quad \forall i
$$

### 2.2 사이클

#### 정의 2.2.1: 사이클 (Cycle)

**사이클**은 시작과 끝이 같은 경로입니다:

$$
P = (v_0, e_1, v_1, \ldots, e_n, v_n) \quad \text{where } v_0 = v_n
$$

#### 정리 2.2.1: 사이클 감지

**사이클 존재 여부는 DFS로 $O(V + E)$ 시간에 판단 가능합니다.**

**증명:** DFS 중 백 엣지(back edge) 발견 시 사이클 존재

---

## 3. 상태 전이 시스템

### 3.1 상태 공간

#### 정의 3.1.1: 상태 (State)

**상태**는 그래프 실행 중의 데이터입니다:

$$
\text{State} = \{key_1: value_1, key_2: value_2, \ldots\}
$$

#### 정의 3.1.2: 상태 전이 함수

**상태 전이 함수**는 다음과 같습니다:

$$
f: \text{State} \times \text{Node} \rightarrow \text{State}
$$

$$
\text{State}_{\text{new}} = f(\text{State}_{\text{old}}, \text{node})
$$

### 3.2 llmkit 구현

#### 구현 3.2.1: StateGraph

```python
**llmkit 구현:**
```python
# facade/state_graph_facade.py: StateGraph
# service/impl/state_graph_service_impl.py: StateGraphServiceImpl
class StateGraph:
    """
    상태 그래프: G = (V, E, s, t)
    
    수학적 정의:
    - V: 노드 집합 (nodes)
    - E: 엣지 집합 (edges)
    - s: 소스 함수 (source)
    - t: 타겟 함수 (target)
    
    상태 전이: State_new = f_node(State_old)
    """
    def __init__(
        self, 
        state_schema: Optional[type] = None, 
        config: Optional[GraphConfig] = None
    ):
        """
        Args:
            state_schema: State TypedDict 클래스 (옵션)
            config: 그래프 설정 (체크포인팅, 재시도 등)
        """
        self.state_schema = state_schema
        self.config = config or GraphConfig()
        self.nodes: Dict[str, Callable] = {}  # V
        self.edges: Dict[str, Union[str, type[END]]] = {}  # E
        self.conditional_edges: Dict[str, tuple] = {}  # 조건부 엣지
        self.entry_point: Optional[str] = None  # 시작 노드
        # 내부적으로 StateGraphHandler와 StateGraphService 사용
    
    def add_node(self, name: str, func: Callable[[StateType], StateType]):
        """
        노드 추가: f_node: State → State
        
        Args:
            name: 노드 이름 (v ∈ V)
            func: 노드 함수 (State를 변환하는 함수)
        """
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists")
        self.nodes[name] = func
    
    def add_edge(self, from_node: str, to_node: Union[str, type[END]]):
        """
        엣지 추가: (from_node, to_node) ∈ E
        
        Args:
            from_node: 소스 노드
            to_node: 타겟 노드 (END로 종료 가능)
        """
        if from_node not in self.nodes:
            raise ValueError(f"Node '{from_node}' not found")
        if to_node != END and to_node not in self.nodes:
            raise ValueError(f"Node '{to_node}' not found")
        self.edges[from_node] = to_node
    
    def invoke(self, initial_state: StateType) -> StateType:
        """
        그래프 실행:
        State_0 → f_1 → State_1 → f_2 → ... → State_n
        
        수학적 표현:
        State_{i+1} = f_{node_i}(State_i)
        
        내부적으로 StateGraphHandler.handle_invoke() 사용
        """
        # 내부 구현은 service/impl/state_graph_service_impl.py 참조
        pass
```

---

## 4. 노드와 엣지의 형식적 정의

### 4.1 노드 함수

#### 정의 4.1.1: 노드 함수

**노드 함수**는 상태를 변환합니다:

$$
f_{\text{node}}: \text{State} \rightarrow \text{State}
$$

**특성:**
- 입력: 현재 상태
- 출력: 업데이트된 상태
- 부작용 가능 (로깅, API 호출 등)

### 4.2 엣지 조건

#### 정의 4.2.1: 조건부 엣지

**조건부 엣지**는 상태에 따라 다음 노드를 결정합니다:

$$
\text{next\_node} = \begin{cases}
\text{node}_A & \text{if } \text{condition}(state) \\
\text{node}_B & \text{otherwise}
\end{cases}
$$

---

## 5. 그래프 실행 모델

### 5.1 순차 실행

#### 정의 5.1.1: 순차 실행

**순차 실행**은 노드를 하나씩 실행합니다:

$$
\text{State}_0 \xrightarrow{f_1} \text{State}_1 \xrightarrow{f_2} \text{State}_2 \xrightarrow{f_3} \cdots
$$

**시간 복잡도:** $O(n)$ ($n$ = 노드 수)

### 5.2 병렬 실행

#### 정의 5.2.1: 병렬 실행

**병렬 실행**은 독립적인 노드를 동시에 실행합니다:

$$
\text{State}_0 \xrightarrow{\{f_1, f_2\}} \text{State}_1
$$

**시간 복잡도:** $O(\max(t_1, t_2))$ (병렬)

---

## 6. CS 관점: 그래프 표현과 알고리즘

### 6.1 그래프 표현 방법

#### CS 관점 6.1.1: 인접 리스트

**인접 리스트 표현:**

```python
graph = {
    "node1": ["node2"],
    "node2": ["node3", "node4"],
    "node3": ["end"],
    "node4": ["end"]
}
```

**공간 복잡도:** $O(V + E)$

### 6.2 그래프 탐색 알고리즘

#### 알고리즘 6.2.1: BFS (Breadth-First Search)

```
Algorithm: BFS(graph, start)
1. queue ← [start]
2. visited ← {start}
3. 
4. while queue is not empty:
5.     node ← queue.dequeue()
6.     process(node)
7.     for neighbor in graph[node]:
8.         if neighbor not in visited:
9.             visited.add(neighbor)
10.            queue.enqueue(neighbor)
```

**시간 복잡도:** $O(V + E)$

---

## 질문과 답변 (Q&A)

### Q1: 유향 그래프와 무향 그래프의 차이는?

**A:** 차이점:

1. **유향 그래프:**
   - 엣지에 방향 있음
   - $A \rightarrow B \neq B \rightarrow A$
   - 워크플로우에 적합

2. **무향 그래프:**
   - 엣지에 방향 없음
   - $A - B = B - A$
   - 관계 표현에 적합

**워크플로우는 유향 그래프 사용**

### Q2: 사이클이 있으면 안 되나요?

**A:** 상황에 따라 다릅니다:

**사이클 허용:**
- 반복 처리 필요
- 루프 노드 사용
- 종료 조건 필요

**사이클 방지:**
- 단순 워크플로우
- 무한 루프 방지
- max_iterations 설정

---

## 참고 문헌

1. **Cormen et al. (2009)**: "Introduction to Algorithms" - 그래프 이론
2. **Diestel (2017)**: "Graph Theory" - 수학적 기초

---

**작성일**: 2025-01-XX  
**버전**: 3.0 (CS 관점 + 석사 수준 확장)

