# Graph Workflows 실무 가이드: 노드 기반 워크플로우 구축

**실무 적용 문서**  
**대상**: AI 엔지니어, 백엔드 개발자

---

## 목차

1. [그래프 워크플로우 기본](#1-그래프-워크플로우-기본)
2. [노드 타입과 사용법](#2-노드-타입과-사용법)
3. [조건부 라우팅](#3-조건부-라우팅)
4. [병렬 실행](#4-병렬-실행)
5. [캐싱과 최적화](#5-캐싱과-최적화)
6. [에러 처리](#6-에러-처리)
7. [프로덕션 배포](#7-프로덕션-배포)

---

## 1. 그래프 워크플로우 기본

### 1.1 기본 사용법

```python
from llmkit.graph import Graph
from llmkit import Client

# 그래프 생성
graph = Graph()

# 노드 추가
def process_data(state):
    return {"processed": state["input"].upper()}

graph.add_function_node("process", process_data)

# 엣지 추가
graph.add_edge("process", END)

# 실행
result = await graph.run({"input": "hello"})
print(result["processed"])  # "HELLO"
```

### 1.2 StateGraph 사용

```python
from llmkit.state_graph import StateGraph
from typing import TypedDict

# State 정의
class MyState(TypedDict):
    input: str
    output: str
    count: int

# 그래프 생성
graph = StateGraph(MyState)

# 노드 추가
def process(state: MyState) -> MyState:
    state["output"] = state["input"].upper()
    state["count"] += 1
    return state

graph.add_node("process", process)
graph.add_edge("process", END)
graph.set_entry_point("process")

# 실행
result = graph.invoke({"input": "hello", "output": "", "count": 0})
```

---

## 2. 노드 타입과 사용법

### 2.1 Function Node

```python
def extract_keywords(state):
    text = state["text"]
    keywords = text.split()[:5]
    return {"keywords": keywords}

graph.add_function_node("extract", extract_keywords)
```

### 2.2 LLM Node

```python
from llmkit import Client

client = Client(model="gpt-4o-mini")

graph.add_llm_node(
    "summarizer",
    client,
    template="Summarize: {text}",
    input_keys=["text"],
    output_key="summary"
)
```

### 2.3 Agent Node

```python
from llmkit import Agent

agent = Agent(model="gpt-4o-mini", tools=[...])

graph.add_agent_node(
    "researcher",
    agent,
    input_key="query",
    output_key="result"
)
```

### 2.4 Grader Node

```python
graph.add_grader_node(
    "quality_check",
    client,
    criteria="Is this summary good?",
    input_key="summary",
    output_key="grade"
)
```

---

## 3. 조건부 라우팅

### 3.1 기본 조건부 라우팅

```python
def should_refine(state):
    grade = state.get("grade", 0)
    return grade < 0.7

graph.add_conditional_node(
    "quality_check",
    condition=should_refine,
    true_node=refine_node,
    false_node=END
)
```

### 3.2 복잡한 조건

```python
def route_decision(state):
    if state["type"] == "urgent":
        return "urgent_handler"
    elif state["priority"] > 5:
        return "high_priority_handler"
    else:
        return "normal_handler"

graph.add_conditional_edge("router", route_decision)
```

---

## 4. 병렬 실행

### 4.1 Parallel Node

```python
from llmkit.graph import ParallelNode

parallel = ParallelNode(
    "parallel_processing",
    nodes=[node1, node2, node3],
    aggregate_strategy="merge"
)

graph.add_node(parallel)
```

### 4.2 병렬 실행 전략

**Merge (병합):**
```python
# 모든 결과를 병합
aggregate_strategy="merge"
```

**First (첫 번째):**
```python
# 첫 번째 완료된 결과만 사용
aggregate_strategy="first"
```

**Vote (투표):**
```python
# 다수결로 결정
aggregate_strategy="vote"
```

---

## 5. 캐싱과 최적화

### 5.1 노드 캐싱

```python
# 노드별 캐싱 활성화
graph.add_function_node(
    "expensive_operation",
    expensive_func,
    cache=True
)
```

### 5.2 전역 캐싱

```python
# 그래프 생성 시 캐싱 활성화
graph = Graph(enable_cache=True)
```

### 5.3 캐시 무효화

```python
# 특정 노드 캐시 클리어
graph.cache.clear_node("node_name")
```

---

## 6. 에러 처리

### 6.1 Try-Except

```python
def safe_node(state):
    try:
        result = risky_operation(state)
        return {"result": result}
    except Exception as e:
        return {"error": str(e), "result": None}
```

### 6.2 재시도 로직

```python
from tenacity import retry, stop_after_attempt

@retry(stop=stop_after_attempt(3))
def retry_node(state):
    return external_api_call(state)
```

---

## 7. 프로덕션 배포

### 7.1 체크포인트

```python
from llmkit.state_graph import StateGraph, GraphConfig

config = GraphConfig(
    enable_checkpointing=True,
    checkpoint_dir="./checkpoints"
)

graph = StateGraph(MyState, config=config)

# 실행 중단 후 재개
result = graph.invoke(
    initial_state,
    resume_from="checkpoint_id"
)
```

### 7.2 모니터링

```python
# 실행 기록 확인
executions = graph.executions
for exec in executions:
    print(f"Duration: {exec.duration}")
    print(f"Nodes: {len(exec.nodes_executed)}")
```

---

## 베스트 프랙티스

- [ ] 명확한 노드 이름 사용
- [ ] 상태 타입 정의 (TypedDict)
- [ ] 조건부 라우팅 명확히
- [ ] 캐싱 전략 수립
- [ ] 에러 처리 구현
- [ ] 체크포인트 활용
- [ ] 모니터링 설정

---

**작성일**: 2025-01-XX  
**버전**: 1.0

