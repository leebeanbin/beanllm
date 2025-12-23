# Multi-Agent Theory: 다중 에이전트 시스템의 수학적 모델

**석사 수준 이론 문서**  
**기반**: llmkit MultiAgentSystem, CommunicationBus 실제 구현 분석

---

## 목차

### Part I: 메시지 전달 이론
1. [메시지 전달 시스템의 수학적 모델](#part-i-메시지-전달-이론)
2. [통신 프로토콜과 채널 용량](#12-통신-프로토콜과-채널-용량)
3. [Publish-Subscribe 패턴](#13-publish-subscribe-패턴)

### Part II: 조정 전략
4. [순차 실행의 함수 합성](#part-ii-조정-전략)
5. [병렬 실행과 속도 향상](#42-병렬-실행과-속도-향상)
6. [계층적 구조와 트리 이론](#43-계층적-구조와-트리-이론)

### Part III: 합의 알고리즘
7. [투표 기반 합의](#part-iii-합의-알고리즘)
8. [Consensus 알고리즘](#72-consensus-알고리즘)
9. [게임 이론적 관점](#73-게임-이론적-관점)

---

## Part I: 메시지 전달 이론

### 1.1 메시지 전달 시스템의 수학적 모델

#### 정의 1.1.1: 메시지 (Message)

**메시지**는 다음 튜플로 정의됩니다:

$$
m = (id, sender, receiver, type, content, timestamp)
$$

**llmkit 구현:**
```python
# domain/multi_agent/communication.py: AgentMessage
# domain/multi_agent/communication.py: CommunicationBus
@dataclass
class AgentMessage:
    """
    메시지: m = (id, sender, receiver, type, content, timestamp)
    
    수학적 정의:
    - id: 고유 식별자
    - sender: 송신자 a_s ∈ A
    - receiver: 수신자 a_r ∈ A ∪ {None} (None = broadcast)
    - type: 메시지 타입 (INFORM, REQUEST, RESPONSE 등)
    - content: 메시지 내용
    - timestamp: 전송 시간
    
    실제 구현:
    - domain/multi_agent/communication.py: AgentMessage
    - domain/multi_agent/communication.py: CommunicationBus.publish()
    - facade/multi_agent_facade.py: MultiAgentCoordinator.send_message()
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""                          # 송신자 a_s
    receiver: Optional[str] = None            # 수신자 a_r (None = broadcast)
    message_type: MessageType = MessageType.INFORM
    content: Any = None                       # 내용
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (직렬화)"""
        return {
            "id": self.id,
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }
```

#### 정의 1.1.2: 메시지 전달 함수

**메시지 전달 함수:**

$$
\text{deliver}: M \times A \rightarrow A'
$$

여기서 $M$은 메시지 집합, $A$는 에이전트 집합입니다.

---

### 1.2 통신 프로토콜과 채널 용량

#### 정의 1.2.1: 채널 용량 (Channel Capacity)

**채널 용량**은 Shannon의 정보 이론에 의해 정의됩니다:

$$
C = \max_{P(X)} I(X; Y)
$$

여기서:
- $X$: 입력 메시지
- $Y$: 출력 메시지
- $I(X; Y)$: Mutual Information

#### 정리 1.2.1: 전송 보장 수준

**전송 보장 수준:**

1. **At-most-once**: 최대 1번 전송 (빠름, 손실 가능)
2. **At-least-once**: 최소 1번 전송 (중복 가능)
3. **Exactly-once**: 정확히 1번 전송 (느리지만 보장)

**llmkit 구현:**
```python
# domain/multi_agent/communication.py: CommunicationBus
class CommunicationBus:
    """
    통신 버스: 메시지 전달 시스템
    
    수학적 모델:
    - deliver: M × A → A'
    - publish: M → void (broadcast)
    - subscribe: A × Callback → void
    
    시간 복잡도:
    - publish: O(n) where n = number of subscribers
    - subscribe: O(1)
    - get_history: O(m) where m = message count
    """
    def __init__(self, delivery_guarantee: str = "at-most-once"):
        """
        전송 보장 수준:
        - at-most-once: O(1) 시간, 손실 가능 (기본값)
        - at-least-once: O(n) 시간, 중복 가능
        - exactly-once: O(n) 시간 + 중복 체크 (O(1) lookup)
        
        실제 구현:
        - domain/multi_agent/communication.py: CommunicationBus
        - facade/multi_agent_facade.py: MultiAgentCoordinator (CommunicationBus 사용)
        """
        self.delivery_guarantee = delivery_guarantee
        self.subscribers: Dict[str, List[Callable]] = {}  # agent_id → callbacks
        self.messages: List[AgentMessage] = []  # 메시지 히스토리
        self.delivered_messages: Set[str] = set()  # exactly-once용
        self.delivery_guarantee = delivery_guarantee
        self.delivered_messages: set = set()  # Exactly-once용
    
    async def publish(self, message: AgentMessage):
        """
        메시지 발행
        
        시간 복잡도:
        - Unicast: O(1) (수신자 1명)
        - Broadcast: O(n) (n = 구독자 수)
        """
        if self.delivery_guarantee == "exactly-once":
            if message.id in self.delivered_messages:
                return  # 중복 방지
            self.delivered_messages.add(message.id)
```

---

### 1.3 Publish-Subscribe 패턴

#### 정의 1.3.1: Publish-Subscribe 모델

**Publish-Subscribe 패턴:**

$$
\text{Publisher}: P \rightarrow \{e_1, e_2, \ldots, e_n\}
$$

$$
\text{Subscriber}: S \leftarrow \{e \in E | \text{filter}(e)\}
$$

**llmkit 구현:**
```python
# domain/multi_agent/communication.py: CommunicationBus
# facade/multi_agent_facade.py: MultiAgentCoordinator
class CommunicationBus:
    """
    메시지 버스: Publisher-Subscriber 패턴
    
    수학적 표현:
    - Subscribe: S ← {e | filter(e)}
    - Publish: P → {e₁, e₂, ..., eₙ}
    
    실제 구현:
    - domain/multi_agent/communication.py: CommunicationBus
    - facade/multi_agent_facade.py: MultiAgentCoordinator
    """
    def subscribe(self, agent_id: str, callback: Callable[[AgentMessage], None]):
        """
        구독: S ← {e | filter(e)}
        
        실제 구현:
        - domain/multi_agent/communication.py: CommunicationBus.subscribe()
        """
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        self.subscribers[agent_id].append(callback)

    async def publish(self, message: AgentMessage):
        """
        발행: P → {e₁, e₂, ..., eₙ}
        
        실제 구현:
        - domain/multi_agent/communication.py: CommunicationBus.publish()
        """
        if message.receiver:
            # Unicast: 1:1
            if message.receiver in self.subscribers:
                for callback in self.subscribers[message.receiver]:
                    await callback(message)
        else:
            # Broadcast: 1:N
            for agent_id, callbacks in self.subscribers.items():
                if agent_id != message.sender:
                    for callback in callbacks:
                        await callback(message)
```

---

## Part II: 조정 전략

### 2.1 순차 실행의 함수 합성

#### 정의 2.1.1: 순차 실행

**순차 실행**은 함수 합성으로 표현됩니다:

$$
\text{result} = f_n \circ f_{n-1} \circ \cdots \circ f_2 \circ f_1(\text{task})
$$

**시간 복잡도:**

$$
T_{\text{sequential}} = \sum_{i=1}^n T_i
$$

**llmkit 구현:**
```python
# domain/multi_agent/strategies.py: SequentialStrategy
# service/impl/multi_agent_service_impl.py: MultiAgentServiceImpl.execute_sequential()
class SequentialStrategy(CoordinationStrategy):
    """
    순차 실행: fₙ ∘ fₙ₋₁ ∘ ... ∘ f₁(task)
    
    시간 복잡도: O(Σ Tᵢ)
    
    실제 구현:
    - domain/multi_agent/strategies.py: SequentialStrategy
    - service/impl/multi_agent_service_impl.py: MultiAgentServiceImpl.execute_sequential()
    """
    async def execute(
        self,
        agents: List[Agent],
        task: str,
        **kwargs
    ) -> Dict[str, Any]:
        current_input = task
        
        for agent in agents:
            # fᵢ(current_input) 실행
            result = await agent.run(current_input)
            current_input = result.answer  # 다음 입력
        
        return {"final_result": current_input}
```

---

### 2.2 병렬 실행과 속도 향상

#### 정의 2.2.1: 병렬 실행

**병렬 실행:**

$$
\text{results} = \{f_1(\text{task}), f_2(\text{task}), \ldots, f_n(\text{task})\} \text{ (동시 실행)}
$$

**시간 복잡도:**

$$
T_{\text{parallel}} = \max(T_1, T_2, \ldots, T_n)
$$

#### 시각적 표현: 순차 vs 병렬

```
┌─────────────────────────────────────────────────────────┐
│              순차 실행 vs 병렬 실행                      │
└─────────────────────────────────────────────────────────┘

순차 실행 (Sequential):
───────────────────────
시간 →

Agent1: ████████ (8초)
Agent2:         ████████████ (12초)
Agent3:                 ████████ (8초)
───────────────────────────────────────────────
총 시간: 8 + 12 + 8 = 28초

병렬 실행 (Parallel):
───────────────────────
시간 →

Agent1: ████████ (8초)
Agent2: ████████████ (12초)
Agent3: ████████ (8초)
───────────────────────────────────────────────
총 시간: max(8, 12, 8) = 12초

속도 향상: S = 28 / 12 = 2.33배
```

#### 구체적 수치 예시

**예시 2.2.1: 병렬 실행 계산**

작업: "고양이에 대한 정보 수집"
에이전트 3개:
- Agent1 (웹 검색): $T_1 = 8$초
- Agent2 (데이터베이스): $T_2 = 12$초
- Agent3 (API 호출): $T_3 = 8$초

**순차 실행:**
$$
T_{\text{sequential}} = 8 + 12 + 8 = 28 \text{초}
$$

**병렬 실행:**
$$
T_{\text{parallel}} = \max(8, 12, 8) = 12 \text{초}
$$

**속도 향상:**
$$
S = \frac{28}{12} = 2.33 \text{배}
$$

#### 정리 2.2.1: 속도 향상 (Speedup)

**속도 향상:**

$$
S = \frac{T_{\text{sequential}}}{T_{\text{parallel}}} = \frac{\sum_{i=1}^n T_i}{\max(T_1, T_2, \ldots, T_n)}
$$

**이상적 경우:** $S = n$ (모든 에이전트가 같은 시간 소요)

#### 속도 향상 그래프

```
속도 향상 (S) vs 에이전트 수 (n)

S
│
n │     ★ (이상적: S = n)
  │    /
  │   /
  │  /  ★ (실제: S < n, 작업 시간이 다름)
  │ /
  │/★
  └──────────────────→ n
  1  2  3  4  5

예시:
- n=3, T=[8,12,8]: S = 28/12 = 2.33
- n=3, T=[10,10,10]: S = 30/10 = 3.0 (이상적)
```

**llmkit 구현:**
```python
# domain/multi_agent/strategies.py: ParallelStrategy
# service/impl/multi_agent_service_impl.py: MultiAgentServiceImpl.execute_parallel()
class ParallelStrategy(CoordinationStrategy):
    """
    병렬 실행: {f₁(task), f₂(task), ..., fₙ(task)} 동시 실행
    
    시간 복잡도: O(max(T₁, T₂, ..., Tₙ))
    속도 향상: S = T_sequential / T_parallel
    
    실제 구현:
    - domain/multi_agent/strategies.py: ParallelStrategy
    - service/impl/multi_agent_service_impl.py: MultiAgentServiceImpl.execute_parallel()
    """
    async def execute(
        self,
        agents: List[Agent],
        task: str,
        **kwargs
    ) -> Dict[str, Any]:
        # 모든 agent를 병렬 실행
        tasks = [agent.run(task) for agent in agents]
        results = await asyncio.gather(*tasks)  # 동시 실행
        
        # 결과 집계
        if self.aggregation == "vote":
            # 투표: 다수결
            from collections import Counter
            vote_counts = Counter([r.answer for r in results])
            final_answer = vote_counts.most_common(1)[0][0]
            return {"final_result": final_answer}
```

---

### 2.3 계층적 구조와 트리 이론

#### 정의 2.3.1: 계층적 구조

**계층적 구조**는 트리로 표현됩니다:

$$
T = (V, E, r)
$$

여기서:
- $V$: 노드 (에이전트)
- $E$: 엣지 (관계)
- $r$: 루트 (매니저 에이전트)

#### 정리 2.3.1: 계층적 실행 시간

**계층적 실행 시간:**

$$
T_{\text{hierarchical}} = d \times T_{\max}
$$

여기서 $d$는 트리 깊이, $T_{\max}$는 최대 에이전트 실행 시간입니다.

**llmkit 구현:**
```python
# domain/multi_agent/strategies.py: HierarchicalStrategy
# service/impl/multi_agent_service_impl.py: MultiAgentServiceImpl.execute_hierarchical()
class HierarchicalStrategy(CoordinationStrategy):
    """
    계층적 구조:
    manager ─┬─ worker₁
             ├─ worker₂
             └─ worker₃
    
    시간 복잡도: O(d × T_max)
    
    실제 구현:
    - domain/multi_agent/strategies.py: HierarchicalStrategy
    - service/impl/multi_agent_service_impl.py: MultiAgentServiceImpl.execute_hierarchical()
    """
    def __init__(self, manager_agent: Agent):
        self.manager = manager_agent
    
    async def execute(
        self,
        agents: List[Agent],  # Worker agents
        task: str,
        **kwargs
    ) -> Dict[str, Any]:
        # 1. Manager가 작업 분배
        subtasks = await self.manager.run(f"Divide task: {task}")
        
        # 2. Workers 병렬 실행
        worker_tasks = [agent.run(subtask) for agent, subtask in zip(agents, subtasks)]
        worker_results = await asyncio.gather(*worker_tasks)
        
        # 3. Manager가 결과 집계
        final_result = await self.manager.run(f"Combine: {worker_results}")
        return {"final_result": final_result}
```

---

## Part III: 합의 알고리즘

### 3.1 투표 기반 합의

#### 정의 3.1.1: 다수결 투표 (Majority Vote)

**다수결 투표:**

$$
\text{consensus} = \arg\max_{v \in V} \sum_{a \in A} \mathbf{1}[\text{vote}(a) = v]
$$

여기서:
- $V$: 가능한 답변 집합
- $A$: 에이전트 집합
- $\mathbf{1}[\cdot]$: 지시 함수

**llmkit 구현:**
```python
# domain/multi_agent/strategies.py: ParallelStrategy (aggregation="vote")
if self.aggregation == "vote":
    """
    다수결 투표:
    consensus = argmax_v Σ 1[vote(a) = v]
    
    실제 구현:
    - domain/multi_agent/strategies.py: ParallelStrategy (aggregation="vote")
    """
    from collections import Counter
    vote_counts = Counter([r.answer for r in results])
    final_answer = vote_counts.most_common(1)[0][0]
    
    agreement_rate = vote_counts[final_answer] / len(results)
    return {
        "final_result": final_answer,
        "agreement_rate": agreement_rate
    }
```

#### 정리 3.1.1: 투표의 정확도

**에이전트가 독립적이고 각각 정확도 $p$를 가지면:**

$$
P(\text{correct}) = \sum_{k = \lceil n/2 \rceil}^n \binom{n}{k} p^k (1-p)^{n-k}
$$

**증명:** Binomial Distribution의 누적 분포 함수

---

### 3.2 Consensus 알고리즘

#### 정의 3.2.1: 합의 (Consensus)

**합의 조건:**

$$
\forall a_i, a_j \in A: \text{decision}(a_i) = \text{decision}(a_j)
$$

**llmkit 구현:**
```python
# domain/multi_agent/strategies.py: ParallelStrategy (aggregation="consensus")
elif self.aggregation == "consensus":
    """
    합의: 모든 에이전트가 같은 답변
    ∀ aᵢ, aⱼ: decision(aᵢ) = decision(aⱼ)
    
    실제 구현:
    - domain/multi_agent/strategies.py: ParallelStrategy (aggregation="consensus")
    """
    answers = [r.answer for r in results]
    if len(set(answers)) == 1:
        return {
            "final_result": answers[0],
            "consensus": True
        }
    else:
        return {
            "final_result": None,
            "consensus": False
        }
```

---

### 3.3 게임 이론적 관점

#### 정의 3.3.1: 협력 게임 (Cooperative Game)

**협력 게임**은 다음으로 정의됩니다:

$$
G = (N, v)
$$

여기서:
- $N$: 플레이어 집합 (에이전트)
- $v: 2^N \rightarrow \mathbb{R}$: 특성 함수 (value function)

#### 정의 3.3.2: Shapley Value

**Shapley Value**는 공정한 기여도를 계산합니다:

$$
\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [v(S \cup \{i\}) - v(S)]
$$

**llmkit에서의 적용:**
- 각 에이전트의 기여도 측정
- 보상 분배의 공정성 보장

---

## 참고 문헌

1. **Shannon (1948)**: "A Mathematical Theory of Communication" - 채널 용량
2. **Lamport et al. (1982)**: "The Byzantine Generals Problem" - 합의 알고리즘
3. **Shapley (1953)**: "A Value for n-person Games" - Shapley Value

---

**작성일**: 2025-01-XX  
**버전**: 2.0 (석사 수준 확장)
