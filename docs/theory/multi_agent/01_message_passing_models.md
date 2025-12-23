# Message Passing Models: 메시지 전달 모델

**석사 수준 이론 문서**  
**CS 관점 + 수학적 엄밀성**  
**기반**: llmkit MultiAgentSystem 실제 구현 분석

---

## 목차

1. [메시지 전달 시스템의 수학적 모델](#1-메시지-전달-시스템의-수학적-모델)
2. [통신 프로토콜](#2-통신-프로토콜)
3. [Publish-Subscribe 패턴](#3-publish-subscribe-패턴)
4. [메시지 순서 보장](#4-메시지-순서-보장)
5. [CS 관점: 구현과 성능](#5-cs-관점-구현과-성능)

---

## 1. 메시지 전달 시스템의 수학적 모델

### 1.1 메시지의 정의

#### 정의 1.1.1: 메시지 (Message)

**메시지**는 다음 튜플로 정의됩니다:

$$
m = (id, sender, receiver, type, content, timestamp)
$$

**llmkit 구현:**
```python
# domain/multi_agent/communication.py: AgentMessage
@dataclass
class AgentMessage:
    """
    메시지: m = (id, sender, receiver, type, content, timestamp)
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    receiver: Optional[str] = None  # None = broadcast
    message_type: MessageType = MessageType.INFORM
    content: Any = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "id": self.id,
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }
```

### 1.2 메시지 전달 함수

#### 정의 1.2.1: 메시지 전달

**메시지 전달 함수:**

$$
\text{send}: \text{Agent} \times \text{Message} \rightarrow \text{void}
$$

$$
\text{receive}: \text{Agent} \rightarrow \text{Message}
$$

---

## 2. 통신 프로토콜

### 2.1 전달 보장

#### 정의 2.1.1: 전달 보장 수준

**1. At-most-once:**
- 메시지가 최대 1번 전달
- 손실 가능

**2. At-least-once:**
- 메시지가 최소 1번 전달
- 중복 가능

**3. Exactly-once:**
- 메시지가 정확히 1번 전달
- 중복 방지 필요

### 2.2 llmkit 구현

#### 구현 2.2.1: Exactly-once 보장

```python
# domain/multi_agent/communication.py: CommunicationBus
class CommunicationBus:
    """
    통신 버스: 메시지 전달 시스템
    
    전달 보장 수준:
    - at-most-once: O(1) 시간, 손실 가능
    - at-least-once: O(n) 시간, 중복 가능
    - exactly-once: O(n) 시간 + 중복 체크
    """
    def __init__(self, delivery_guarantee: str = "at-most-once"):
        self.delivery_guarantee = delivery_guarantee
        self._messages: List[AgentMessage] = []
        self._delivered_messages: Set[str] = set()
        self._subscribers: Dict[str, List[Callable]] = {}
    
    def send(self, message: AgentMessage):
        """
        메시지 전송: send(Agent, Message) → void
        """
        if self.delivery_guarantee == "exactly-once":
            if message.id in self._delivered_messages:
                return  # 중복 방지
            self._delivered_messages.add(message.id)
        
        self._messages.append(message)
        
        # 구독자에게 전달
        if message.receiver is None:
            # Broadcast
            for callbacks in self._subscribers.values():
                for callback in callbacks:
                    callback(message)
        elif message.receiver in self._subscribers:
            # 특정 수신자
            for callback in self._subscribers[message.receiver]:
                callback(message)
```

---

## 3. Publish-Subscribe 패턴

### 3.1 Pub-Sub 모델

#### 정의 3.1.1: Publish-Subscribe

**Publisher:**
$$
P: \text{Message} \rightarrow \{e_1, e_2, \ldots, e_n\}
$$

**Subscriber:**
$$
S \leftarrow \{e \in E | \text{filter}(e)\}
$$

---

## 4. 메시지 순서 보장

### 4.1 순서 보장

#### 정의 4.1.1: 순서 보장

**FIFO (First-In-First-Out):**
- 발신자 순서 보장
- 수신자 순서 보장

**Causal Order:**
- 인과 관계 순서 보장

---

## 5. CS 관점: 구현과 성능

### 5.1 메시지 큐

#### CS 관점 5.1.1: 큐 구현

**메시지 큐:**

```python
from collections import deque

class MessageQueue:
    def __init__(self):
        self.queue = deque()
    
    def enqueue(self, message):
        self.queue.append(message)
    
    def dequeue(self):
        return self.queue.popleft()
```

**시간 복잡도:**
- Enqueue: $O(1)$
- Dequeue: $O(1)$

---

## 질문과 답변 (Q&A)

### Q1: 메시지 전달 보장은 왜 중요한가요?

**A:** 중요성:

1. **데이터 일관성:**
   - 메시지 손실 방지
   - 중복 처리 방지

2. **신뢰성:**
   - Exactly-once 보장
   - 시스템 안정성

---

## 참고 문헌

1. **Lamport (1978)**: "Time, Clocks, and the Ordering of Events in a Distributed System"

---

**작성일**: 2025-01-XX  
**버전**: 3.0 (CS 관점 + 석사 수준 확장)

