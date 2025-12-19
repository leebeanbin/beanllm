# Node Caching and Checkpointing: 노드 캐싱과 체크포인트

**석사 수준 이론 문서**  
**CS 관점 + 수학적 엄밀성**  
**기반**: llmkit Graph 실제 구현 분석

---

## 목차

1. [노드 캐싱의 수학적 모델](#1-노드-캐싱의-수학적-모델)
2. [캐시 키 생성](#2-캐시-키-생성)
3. [체크포인트 이론](#3-체크포인트-이론)
4. [복구 알고리즘](#4-복구-알고리즘)
5. [CS 관점: 구현과 성능](#5-cs-관점-구현과-성능)

---

## 1. 노드 캐싱의 수학적 모델

### 1.1 캐싱 함수

#### 정의 1.1.1: 노드 캐싱

**노드 캐싱**은 입력 상태에 대해 결과를 저장합니다:

$$
\text{Cache}: (\text{node}, \text{state}) \rightarrow \text{result}
$$

**캐시 히트:**
$$
\text{Cache}(\text{node}, \text{state}) \neq \text{None} \implies \text{재사용}
$$

**캐시 미스:**
$$
\text{Cache}(\text{node}, \text{state}) = \text{None} \implies \text{계산 후 저장}
$$

### 1.2 llmkit 구현

#### 구현 1.1.1: NodeCache

```python
# graph.py: Line 200-280
class NodeCache:
    def __init__(self):
        self.cache: Dict[Tuple[str, str], Any] = {}
    
    def get(self, node_name: str, state: GraphState) -> Optional[Any]:
        """캐시 조회"""
        cache_key = self._make_key(node_name, state)
        return self.cache.get(cache_key)
    
    def set(self, node_name: str, state: GraphState, result: Any):
        """캐시 저장"""
        cache_key = self._make_key(node_name, state)
        self.cache[cache_key] = result
```

---

## 2. 캐시 키 생성

### 2.1 상태 해싱

#### 정의 2.1.1: 캐시 키

**캐시 키**는 노드와 상태의 해시입니다:

$$
\text{key} = \text{hash}(\text{node\_name}, \text{state})
$$

**구현:**
```python
def _make_key(self, node_name: str, state: GraphState) -> str:
    state_hash = hash(frozenset(state.items()))
    return f"{node_name}:{state_hash}"
```

---

## 3. 체크포인트 이론

### 3.1 체크포인트의 정의

#### 정의 3.1.1: Checkpoint

**체크포인트**는 실행 상태를 저장합니다:

$$
\text{Checkpoint} = (\text{state}, \text{current\_node}, \text{timestamp})
$$

**용도:**
- 실행 중단 후 재개
- 디버깅
- 롤백

### 3.2 llmkit 구현

#### 구현 3.2.1: Checkpoint

```python
# state_graph.py: Line 160-163
if self.config.enable_checkpointing:
    self.checkpoint = Checkpoint(self.config.checkpoint_dir)

# 실행 중 체크포인트 저장
if self.checkpoint:
    self.checkpoint.save(execution_id, state, current_node)
```

---

## 4. 복구 알고리즘

### 4.1 체크포인트에서 재개

#### 알고리즘 4.1.1: Resume from Checkpoint

```
Algorithm: Resume(checkpoint_id)
1. checkpoint ← Load(checkpoint_id)
2. state ← checkpoint.state
3. current_node ← checkpoint.current_node
4. 
5. // 체크포인트 이후부터 실행
6. while current_node != END:
7.     state ← ExecuteNode(current_node, state)
8.     current_node ← GetNextNode(current_node, state)
9. 
10. return state
```

---

## 5. CS 관점: 구현과 성능

### 5.1 캐시 성능

#### CS 관점 5.1.1: 캐시 히트율

**캐시 히트율:**

$$
H = \frac{\text{Hits}}{\text{Hits} + \text{Misses}}
$$

**효과:**
- 히트율 80% → 실행 시간 80% 단축
- 비용 80% 절감

---

## 질문과 답변 (Q&A)

### Q1: 캐싱은 항상 유리한가요?

**A:** 상황에 따라 다릅니다:

**유리한 경우:**
- 노드 실행 비용 높음
- 동일 입력 반복
- 메모리 여유

**불리한 경우:**
- 노드 실행 빠름
- 입력 항상 다름
- 메모리 제한

---

## 참고 문헌

1. **Hennessy & Patterson (2019)**: "Computer Architecture" - 캐싱

---

**작성일**: 2025-01-XX  
**버전**: 3.0 (CS 관점 + 석사 수준 확장)

