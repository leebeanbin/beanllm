"""
분산 아키텍처 인터페이스 정의

모든 분산 컴포넌트의 추상 인터페이스를 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncContextManager, AsyncIterator, Dict, Generic, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class RateLimiterInterface(ABC):
    """Rate Limiter 추상 인터페이스"""

    @abstractmethod
    async def acquire(self, key: str, cost: float = 1.0) -> bool:
        """
        토큰 획득 시도 (대기하지 않음)

        Args:
            key: Rate Limit 키 (예: "llm:gpt-4o", "ocr:paddleocr")
            cost: 필요한 토큰 수

        Returns:
            True: 토큰 획득 성공, False: 토큰 부족
        """
        pass

    @abstractmethod
    async def wait(self, key: str, cost: float = 1.0):
        """
        토큰이 충분할 때까지 대기

        Args:
            key: Rate Limit 키
            cost: 필요한 토큰 수
        """
        pass

    @abstractmethod
    def get_status(self, key: str) -> Dict[str, Any]:
        """
        현재 상태 조회

        Args:
            key: Rate Limit 키

        Returns:
            상태 정보 딕셔너리
        """
        pass


class CacheInterface(ABC, Generic[K, V]):
    """Cache 추상 인터페이스 (비동기 지원)"""

    @abstractmethod
    async def get(self, key: K) -> Optional[V]:
        """
        값 조회

        Args:
            key: 캐시 키

        Returns:
            캐시된 값 또는 None
        """
        pass

    @abstractmethod
    async def set(self, key: K, value: V, ttl: Optional[int] = None):
        """
        값 저장

        Args:
            key: 캐시 키
            value: 저장할 값
            ttl: Time-to-Live (초), None이면 기본값 사용
        """
        pass

    @abstractmethod
    async def delete(self, key: K):
        """
        값 삭제

        Args:
            key: 캐시 키
        """
        pass

    @abstractmethod
    async def clear(self):
        """모든 캐시 삭제"""
        pass


class TaskQueueInterface(ABC):
    """작업 큐 추상 인터페이스"""

    @abstractmethod
    async def enqueue(self, task_type: str, data: Dict[str, Any], priority: int = 0) -> str:
        """
        작업 큐에 추가

        Args:
            task_type: 작업 타입 (예: "ocr.recognize", "embedding.process")
            data: 작업 데이터
            priority: 우선순위 (높을수록 우선 처리)

        Returns:
            task_id: 작업 ID
        """
        pass

    @abstractmethod
    async def dequeue(
        self, task_type: str, timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        작업 큐에서 가져오기

        Args:
            task_type: 작업 타입
            timeout: 대기 시간 (초), None이면 무한 대기

        Returns:
            작업 데이터 또는 None (타임아웃)
        """
        pass

    @abstractmethod
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        작업 상태 조회

        Args:
            task_id: 작업 ID

        Returns:
            작업 상태 또는 None
        """
        pass


class EventProducerInterface(ABC):
    """이벤트 발행자 추상 인터페이스"""

    @abstractmethod
    async def publish(self, topic: str, event: Dict[str, Any]):
        """
        이벤트 발행

        Args:
            topic: 이벤트 토픽 (예: "llm.events", "ocr.events")
            event: 이벤트 데이터
        """
        pass


class EventConsumerInterface(ABC):
    """이벤트 구독자 추상 인터페이스"""

    @abstractmethod
    async def subscribe(self, topic: str, handler: Any) -> AsyncIterator[Dict[str, Any]]:
        """
        이벤트 구독

        Args:
            topic: 이벤트 토픽
            handler: 이벤트 핸들러 함수

        Yields:
            이벤트 데이터
        """
        pass


class DistributedLockInterface(ABC):
    """분산 락 추상 인터페이스"""

    @abstractmethod
    def acquire(self, key: str, timeout: float = 30.0) -> AsyncContextManager[None]:
        """
        락 획득 (context manager)

        Args:
            key: 락 키 (예: "vector_store:update:123")
            timeout: 락 타임아웃 (초)

        Yields:
            락 컨텍스트

        Example:
            ```python
            async with lock.acquire("resource:123"):
                # 락 보호 영역
                await update_resource()
            ```
        """
        ...
