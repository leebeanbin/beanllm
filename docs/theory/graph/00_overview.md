# Graph Workflows Theory: 상태 기반 워크플로우의 수학적 모델

**석사 수준 이론 문서**  
**기반**: llmkit StateGraph, Graph 실제 구현 분석

---

## 목차

### Part I: 그래프 이론 기초
1. [유향 그래프의 수학적 정의](#part-i-그래프-이론-기초)
2. [상태 전이 시스템](#12-상태-전이-시스템)
3. [노드와 엣지의 형식적 정의](#13-노드와-엣지의-형식적-정의)

### Part II: 상태 관리 이론
4. [상태 공간과 전이 함수](#part-ii-상태-관리-이론)
5. [캐싱과 메모이제이션](#42-캐싱과-메모이제이션)
6. [체크포인트와 복구](#43-체크포인트와-복구)

### Part III: 조건부 라우팅
7. [조건부 엣지의 수학적 모델](#part-iii-조건부-라우팅)
8. [순환 감지와 고정점](#72-순환-감지와-고정점)
9. [동적 분기와 확률적 라우팅](#73-동적-분기와-확률적-라우팅)

### Part IV: 실행 모델
10. [비동기 실행의 수학적 모델](#part-iv-실행-모델)
11. [병렬 실행과 동시성](#102-병렬-실행과-동시성)
12. [오류 처리와 재시도](#103-오류-처리와-재시도)

---

## Part I: 그래프 이론 기초

### 1.1 유향 그래프의 수학적 정의

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
```

#### 정의 1.1.2: 경로 (Path)

**경로**는 다음과 같이 정의됩니다:

$$
P = (v_0, e_1, v_1, e_2, \ldots, e_n, v_n)
$$

여기서 $t(e_i) = s(e_{i+1})$입니다.

#### 구체적 예시

**예시 1.1.1: 경로 예시**

위 그래프에서 가능한 경로:

1. **경로 1:** $P_1 = (\text{start}, e_1, \text{node1}, e_2, \text{node2}, e_4, \text{node3}, e_4, \text{end})$
2. **경로 2:** $P_2 = (\text{start}, e_1, \text{node1}, e_2, \text{node2}, e_3, \text{node4}, e_5, \text{end})$

**시각적 표현:**

```
경로 1: start → node1 → node2 → node3 → end
        ─────   ─────   ─────   ─────   ───
         e₁      e₂      e₄      e₄

경로 2: start → node1 → node2 → node4 → end
        ─────   ─────   ─────   ─────   ───
         e₁      e₂      e₃      e₅
```

**llmkit 구현:**
```python
# domain/graph/graph_state.py: GraphState
@dataclass
class GraphState:
    """
    그래프 상태: V의 각 노드가 가질 수 있는 상태
    
    수학적 표현: S = {s_v | v ∈ V}
    
    실제 구현:
    - domain/graph/graph_state.py: GraphState
    """
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update(self, updates: Dict[str, Any]):
        """
        상태 업데이트: s_v ← f(s_v, updates)
        
        실제 구현:
        - domain/graph/graph_state.py: GraphState.update()
        """
        self.data.update(updates)
```

---

### 1.2 상태 전이 시스템

#### 정의 1.2.1: 상태 전이 시스템 (State Transition System)

**상태 전이 시스템**은 다음과 같이 정의됩니다:

$$
\mathcal{S} = (S, \Sigma, \delta, s_0, F)
$$

여기서:
- $S$: 상태 집합
- $\Sigma$: 입력 알파벳 (이벤트)
- $\delta: S \times \Sigma \rightarrow S$: 전이 함수
- $s_0$: 초기 상태
- $F \subseteq S$: 최종 상태 집합

#### 정의 1.2.2: 전이 함수

**노드의 전이 함수:**

$$
\delta_v: S \rightarrow S'
$$

**llmkit 구현:**
```python
# domain/graph/node.py: BaseNode
# facade/state_graph_facade.py: StateGraph
# service/impl/state_graph_service_impl.py: StateGraphServiceImpl
class BaseNode(ABC):
    """
    그래프 노드: v = (f_v, I_v, O_v)
    
    where:
    - f_v: S → S' (전이 함수)
    - I_v ⊆ V (입력 노드 집합)
    - O_v ⊆ V (출력 노드 집합)
    """
    @abstractmethod
    async def execute(self, state: GraphState) -> GraphState:
        """
        전이 함수 구현: δ_v(s) = s'
        
        수학적 표현:
        - 입력: 상태 s ∈ S
        - 출력: 새 상태 s' ∈ S
        
        실제 구현:
        - domain/graph/node.py: BaseNode (추상 클래스)
        - facade/state_graph_facade.py: StateGraph.add_node() (노드 추가)
        - service/impl/state_graph_service_impl.py: StateGraphServiceImpl.invoke() (실행)
        """
        pass
```

---

### 1.3 노드와 엣지의 형식적 정의

#### 정의 1.3.1: 노드 (Node)

**노드**는 다음 튜플로 정의됩니다:

$$
v = (f_v, I_v, O_v)
$$

여기서:
- $f_v: S \rightarrow S'$: 노드 함수
- $I_v \subseteq V$: 입력 노드 집합
- $O_v \subseteq V$: 출력 노드 집합

#### 정의 1.3.2: 엣지 (Edge)

**엣지**는 다음 튜플로 정의됩니다:

$$
e = (v_s, v_t, c_e)
$$

여기서:
- $v_s$: 소스 노드
- $v_t$: 타겟 노드
- $c_e: S \rightarrow \{True, False\}$: 조건 함수

**llmkit 구현:**
```python
# facade/state_graph_facade.py: StateGraph
# service/impl/state_graph_service_impl.py: StateGraphServiceImpl
class StateGraph:
    """
    상태 그래프: G = (V, E, s, t)
    
    where:
    - V: 노드 집합
    - E: 엣지 집합
    - s: E → V (소스 함수)
    - t: E → V (타겟 함수)
    """
    def add_edge(
        self,
        from_node: str,
        to_node: str,
        condition: Optional[Callable] = None,
    ):
        """
        엣지 추가: e = (v_s, v_t, c_e)
        
        Args:
            from_node: 소스 노드 v_s
            to_node: 타겟 노드 v_t
            condition: 조건 함수 c_e: S → {True, False} (선택적)
        
        실제 구현:
        - facade/state_graph_facade.py: StateGraph.add_edge()
        - service/impl/state_graph_service_impl.py: StateGraphServiceImpl._get_next_node()
        """
        if condition:
            # 조건부 엣지: c_e(s) = True일 때만 전이
            self.conditional_edges[from_node] = (condition, to_node)
        else:
            # 무조건 엣지: 항상 전이
            self.edges[from_node] = to_node
    
    def connect(
        self,
        from_node: str,
        to_node: str,
        condition: Optional[Callable] = None
    ):
        """
        엣지 추가: e = (from_node, to_node, condition)
        
        condition이 None이면 항상 True (무조건 전이)
        """
        edge = (from_node, to_node, condition)
        self.edges.append(edge)
```

---

## Part II: 상태 관리 이론

### 2.1 상태 공간과 전이 함수

#### 정의 2.1.1: 상태 공간 (State Space)

**상태 공간**은 모든 가능한 상태의 집합입니다:

$$
\mathcal{S} = \prod_{v \in V} S_v
$$

여기서 $S_v$는 노드 $v$의 상태 공간입니다.

#### 정리 2.1.1: 상태 전이의 합성

**여러 노드를 순차 실행하면:**

$$
\delta_{v_n} \circ \delta_{v_{n-1}} \circ \cdots \circ \delta_{v_1}(s_0) = s_n
$$

**llmkit 구현:**
```python
# service/impl/state_graph_service_impl.py: StateGraphServiceImpl.invoke()
# facade/state_graph_facade.py: StateGraph.invoke()
async def invoke(self, request: StateGraphRequest) -> StateGraphResponse:
    """
    상태 전이 합성: s_final = δ_vn ∘ δ_vn-1 ∘ ... ∘ δ_v1(s_0)
    
    실제 구현:
    - service/impl/state_graph_service_impl.py: StateGraphServiceImpl.invoke()
    - facade/state_graph_facade.py: StateGraph.invoke()
    """
    current_state = request.initial_state
    for node_name in execution_order:
        node = request.nodes[node_name]
        current_state = await node.execute(current_state)
    return StateGraphResponse(final_state=current_state)
```

---

### 2.2 캐싱과 메모이제이션

#### 정의 2.2.1: 노드 캐시

**노드 캐시**는 이전 계산 결과를 저장합니다:

$$
\text{Cache}: (v, s) \mapsto s'
$$

**캐시 키 생성:**

$$
\text{key}(v, s) = \text{hash}(v, \text{serialize}(s))
$$

**llmkit 구현:**
```python
# domain/graph/node_cache.py: NodeCache
class NodeCache:
    """
    노드 캐시: Cache(node, state) → result
    
    실제 구현:
    - domain/graph/node_cache.py: NodeCache
    """
    def get_key(self, node_name: str, state: GraphState) -> str:
        """
        캐시 키 생성: hash(node_name, serialize(state))
        
        실제 구현:
        - domain/graph/node_cache.py: NodeCache.get_key()
        """
        state_json = json.dumps(state.data, sort_keys=True)
        hash_value = hashlib.md5(state_json.encode()).hexdigest()
        return f"{node_name}:{hash_value}"
    
    def get(self, node_name: str, state: GraphState) -> Optional[Any]:
        """
        캐시 조회: O(1) 시간 복잡도
        
        실제 구현:
        - domain/graph/node_cache.py: NodeCache.get()
        """
        key = self.get_key(node_name, state)
        if key in self.cache:
            return self.cache[key]
        return None
```

#### 정리 2.2.1: 캐시의 시간 복잡도 개선

**캐시 사용 시:**

- **캐시 히트**: $O(1)$ (캐시 조회)
- **캐시 미스**: $O(T)$ (노드 실행 시간)

**전체 시간 복잡도:**

$$
T_{\text{total}} = (1 - h) \cdot T + h \cdot O(1)
$$

여기서 $h$는 캐시 히트율입니다.

---

### 2.3 체크포인트와 복구

#### 정의 2.3.1: 체크포인트 (Checkpoint)

**체크포인트**는 특정 시점의 상태 스냅샷입니다:

$$
\text{Checkpoint}_t = (s_t, \text{metadata}_t, \text{timestamp}_t)
$$

**llmkit 구현:**
```python
# domain/state_graph/checkpoint.py: Checkpoint
@dataclass
class Checkpoint:
    """
    체크포인트: 특정 시점의 상태 저장
    
    실제 구현:
    - domain/state_graph/checkpoint.py: Checkpoint
    """
    state: GraphState
    step: int
    timestamp: datetime
    metadata: Dict[str, Any]
```

---

## Part III: 조건부 라우팅

### 3.1 조건부 엣지의 수학적 모델

#### 정의 3.1.1: 조건부 엣지

**조건부 엣지**는 조건 함수에 따라 전이합니다:

$$
e_{\text{cond}} = (v_s, v_t, c: S \rightarrow \{True, False\})
$$

**전이 규칙:**

$$
\text{transition}(s) = \begin{cases}
v_t & \text{if } c(s) = True \\
\text{skip} & \text{otherwise}
\end{cases}
$$

**llmkit 구현:**
```python
# facade/state_graph_facade.py: StateGraph.add_conditional_edge()
def add_conditional_edge(
    self,
    from_node: str,
    condition: Callable[[Dict[str, Any]], str],
    edge_mapping: Optional[Dict[str, str]] = None,
):
    """
    조건부 엣지: c(s) ? then_node : else_node
    
    실제 구현:
    - facade/state_graph_facade.py: StateGraph.add_conditional_edge()
    - service/impl/state_graph_service_impl.py: StateGraphServiceImpl._get_next_node()
    """
    self.conditional_edges[from_node] = (condition, edge_mapping)
```

---

### 3.2 순환 감지와 고정점

#### 정의 3.2.1: 순환 (Cycle)

**순환**은 다음을 만족하는 경로입니다:

$$
P = (v_0, e_1, v_1, \ldots, e_n, v_n)
$$

여기서 $v_0 = v_n$입니다.

#### 정리 3.2.1: 고정점 (Fixed Point)

**고정점**은 다음을 만족하는 상태입니다:

$$
\exists s^*: \delta(s^*) = s^*
$$

**수렴 조건:**

$$
\lim_{t \to \infty} s_t = s^*
$$

**llmkit 구현:**
```python
# service/impl/state_graph_service_impl.py: StateGraphServiceImpl.invoke()
# service/impl/graph_service_impl.py: GraphServiceImpl.run_graph()
async def invoke(self, request: StateGraphRequest) -> StateGraphResponse:
    """
    그래프 실행 (순환 감지 및 고정점 찾기)
    
    실제 구현:
    - service/impl/state_graph_service_impl.py: StateGraphServiceImpl.invoke()
    - service/impl/graph_service_impl.py: GraphServiceImpl.run_graph()
    """
    current_state = request.initial_state
    visited = set()
    
    for iteration in range(request.max_iterations):
        if current_node in visited:
            # 순환 감지
            break
        visited.add(current_node)
        
        # 노드 실행
        current_state = await self._execute_node(current_node, current_state)
        
        # 다음 노드 결정
        current_node = self._get_next_node(...)
```

---

### 3.3 동적 분기와 확률적 라우팅

#### 정의 3.3.1: 확률적 라우팅

**확률적 라우팅**은 확률 분포에 따라 분기합니다:

$$
P(v_t | v_s, s) = \text{softmax}([score_1, score_2, \ldots, score_n])
$$

**llmkit 구현:**
```python
# domain/graph/nodes.py (향후 구현)
def probabilistic_routing(
    from_node: str,
    scores: Dict[str, float]
) -> str:
    """
    확률 분포에 따라 노드 선택: P(v_i) = exp(score_i) / Σ exp(score_j)
    
    실제 구현:
    - domain/graph/nodes.py (향후 구현)
    - softmax 기반 확률적 라우팅
    """
    import numpy as np
    nodes = list(scores.keys())
    probs = np.softmax([scores[n] for n in nodes])
    return np.random.choice(nodes, p=probs)
```

---

## Part IV: 실행 모델

### 4.1 비동기 실행의 수학적 모델

#### 정의 4.1.1: 비동기 실행

**비동기 실행**은 여러 작업을 동시에 처리합니다:

$$
\text{Async}(T_1, T_2, \ldots, T_n) = \text{await} [T_1 || T_2 || \cdots || T_n]
$$

**시간 복잡도:**

$$
T_{\text{async}} = \max(T_1, T_2, \ldots, T_n)
$$

**llmkit 구현:**
```python
# service/impl/state_graph_service_impl.py: StateGraphServiceImpl.invoke() (병렬 노드 지원)
async def run_parallel(
    nodes: List[str],
    state: GraphState
) -> GraphState:
    """
    병렬 실행: T_parallel = max(T_i)
    
    실제 구현:
    - service/impl/state_graph_service_impl.py: StateGraphServiceImpl.invoke() (병렬 노드 지원)
    - asyncio.gather() 사용
    """
    tasks = [self.nodes[n].execute(state) for n in nodes]
    results = await asyncio.gather(*tasks)
    # 결과 병합
    return merge_results(results)
```

---

### 4.2 병렬 실행과 동시성

#### 정의 4.2.1: 병렬 실행 그래프

**병렬 실행 가능한 노드 집합:**

$$
V_{\text{parallel}} = \{v | \text{indegree}(v) = 0 \text{ or } \text{parents}(v) \text{ completed}\}
$$

#### 정리 4.2.1: 최소 실행 시간

**최소 실행 시간은 Critical Path에 의해 결정됩니다:**

$$
T_{\min} = \max_{P \in \text{paths}} \sum_{v \in P} T(v)
$$

**llmkit 구현:**
```python
# domain/graph/utils.py (또는 직접 구현)
def find_critical_path(graph: StateGraph) -> List[str]:
    """
    Critical Path 찾기: 최장 경로
    
    수학적 표현: T_min = max_{P ∈ paths} Σ_{v ∈ P} T(v)
    
    실제 구현:
    - domain/graph/utils.py (또는 직접 구현)
    - Topological sort + longest path (Dijkstra 알고리즘 변형)
    """
    # Topological sort + longest path
    # Dijkstra 알고리즘 변형
    pass
```

---

### 4.3 오류 처리와 재시도

#### 정의 4.3.1: 재시도 전략

**지수 백오프 (Exponential Backoff):**

$$
\text{delay}_n = \min(2^n \cdot \text{base}, \text{max\_delay})
$$

**llmkit 구현:**
```python
# service/impl/state_graph_service_impl.py: StateGraphServiceImpl.invoke() (재시도 지원)
# utils/error_handling.py: RetryHandler
async def run_with_retry(
    self,
    node_name: str,
    state: GraphState,
    max_retries: int = 3
) -> GraphState:
    """
    지수 백오프 재시도
    """
    for attempt in range(max_retries):
        try:
            return await self.nodes[node_name].execute(state)
        except Exception as e:
            if attempt < max_retries - 1:
                delay = min(2 ** attempt * 0.1, 1.0)
                await asyncio.sleep(delay)
            else:
                raise
```

---

## 참고 문헌

1. **Cormen et al. (2009)**: "Introduction to Algorithms" - 그래프 이론
2. **Milner (1980)**: "A Calculus of Communicating Systems" - 상태 전이 시스템
3. **Lamport (1978)**: "Time, Clocks, and the Ordering of Events" - 분산 시스템

---

**작성일**: 2025-01-XX  
**버전**: 2.0 (석사 수준 확장)
