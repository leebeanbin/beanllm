"""
Base LLM Provider
LLM 제공자 추상화 인터페이스
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List, Optional


@dataclass
class LLMResponse:
    """LLM 응답 모델"""

    content: str
    model: str
    usage: Optional[Dict] = None


class BaseLLMProvider(ABC):
    """LLM 제공자 기본 인터페이스"""

    def __init__(self, config: Dict):
        self.config = config
        self.name = self.__class__.__name__

    @abstractmethod
    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """
        스트리밍 채팅

        Args:
            messages: 대화 메시지 리스트
            model: 사용할 모델
            system: 시스템 메시지
            temperature: 온도
            max_tokens: 최대 토큰 수

        Yields:
            응답 청크 (str)
        """
        pass

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        일반 채팅 (비스트리밍)

        Args:
            messages: 대화 메시지 리스트
            model: 사용할 모델
            system: 시스템 메시지
            temperature: 온도
            max_tokens: 최대 토큰 수

        Returns:
            LLMResponse
        """
        pass

    @abstractmethod
    async def list_models(self) -> List[str]:
        """사용 가능한 모델 목록 조회"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """제공자 사용 가능 여부"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """건강 상태 확인"""
        pass
