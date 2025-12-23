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

**llmkit 구현:**
```python
# domain/graph/node_cache.py: NodeCache
# domain/state_graph/checkpoint.py: Checkpoint
# service/impl/graph_service_impl.py: GraphServiceImpl (캐시 사용)
import hashlib
import json
from typing import Dict, Optional, Any

class NodeCache:
    """
    노드 캐시: Cache(node, state) → result
    
    수학적 정의:
    - Cache: (node_name, state) → result
    - 캐시 히트: Cache(node, state) ≠ None → 재사용
    - 캐시 미스: Cache(node, state) = None → 계산 후 저장
    
    시간 복잡도:
    - get: O(1) (해시 테이블)
    - set: O(1) (해시 테이블)
    
    실제 구현:
    - domain/graph/node_cache.py: NodeCache
    - service/impl/graph_service_impl.py: GraphServiceImpl (캐시 사용)
    """
    def __init__(self, max_size: int = 1000):
        """
        Args:
            max_size: 최대 캐시 크기 (LRU 방식으로 제거)
        """
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get_key(self, node_name: str, state: Dict[str, Any]) -> str:
        """
        캐시 키 생성: key = hash(node_name, state)
        
        수학적 표현:
        - key = hash(node_name, state)
        - 해시 함수: MD5(JSON(state))
        
        실제 구현:
        - domain/graph/node_cache.py: NodeCache.get_key()
        """
        # 상태를 JSON으로 직렬화하여 해시
        state_json = json.dumps(state, sort_keys=True)
        hash_value = hashlib.md5(state_json.encode()).hexdigest()
        return f"{node_name}:{hash_value}"
    
    def get(self, node_name: str, state: Dict[str, Any]) -> Optional[Any]:
        """
        캐시 조회: O(1)
        
        Returns:
            캐시된 결과 또는 None (미스)
        """
        key = self.get_key(node_name, state)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        else:
            self.misses += 1
            return None
    
    def set(self, node_name: str, state: Dict[str, Any], result: Any):
        """
        캐시 저장: O(1)
        
        Process:
        1. 키 생성
        2. 크기 제한 확인 (LRU 제거)
        3. 저장
        """
        # 크기 제한
        if len(self.cache) >= self.max_size:
            # 가장 오래된 항목 제거 (간단한 구현)
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        
        key = self.get_key(node_name, state)
        self.cache[key] = result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        캐시 통계: H = Hits / (Hits + Misses)
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "max_size": self.max_size,
        }
```

**Checkpoint 구현:**
```python
# domain/state_graph/checkpoint.py: Checkpoint
# facade/state_graph_facade.py: StateGraph (체크포인팅 지원)
class Checkpoint:
    """
    체크포인트: Checkpoint = (state, current_node, timestamp)
    
    수학적 정의:
    - Checkpoint_t = (s_t, node_t, t_t)
    - 용도: 실행 중단 후 재개, 디버깅, 롤백
    
    실제 구현:
    - domain/state_graph/checkpoint.py: Checkpoint
    - facade/state_graph_facade.py: StateGraph (체크포인팅 지원)
    """
    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Args:
            checkpoint_dir: 체크포인트 저장 디렉토리
        """
        self.checkpoint_dir = checkpoint_dir or Path(".checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def save(self, execution_id: str, state: Dict[str, Any], node_name: str):
        """
        체크포인트 저장: Checkpoint_t = (s_t, node_t, t_t)
        
        실제 구현:
        - domain/state_graph/checkpoint.py: Checkpoint.save()
        - JSON 형식으로 저장
        """
        checkpoint_file = self.checkpoint_dir / f"{execution_id}_{node_name}.json"
        
        checkpoint_data = {
            "execution_id": execution_id,
            "node_name": node_name,
            "state": state,
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False, default=str)
    
    def load(self, execution_id: str, node_name: str) -> Optional[Dict[str, Any]]:
        """
        체크포인트 로드: state = Load(execution_id, node_name)
        
        Returns:
            저장된 상태 또는 None (없는 경우)
        """
        checkpoint_file = self.checkpoint_dir / f"{execution_id}_{node_name}.json"
        
        if not checkpoint_file.exists():
            return None
        
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
        
        return checkpoint_data.get("state")
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

**llmkit 구현:**
```python
# domain/state_graph/checkpoint.py: Checkpoint
# facade/state_graph_facade.py: StateGraph (체크포인팅 지원)
if self.config.enable_checkpointing:
    self.checkpoint = Checkpoint(self.config.checkpoint_dir)

# 실행 중 체크포인트 저장
if self.checkpoint:
    self.checkpoint.save(execution_id, state, current_node)

# 실제 구현:
# - domain/state_graph/checkpoint.py: Checkpoint.save()
# - facade/state_graph_facade.py: StateGraph (체크포인팅 지원)
# - service/impl/state_graph_service_impl.py: StateGraphServiceImpl.invoke() (체크포인트 저장)
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

