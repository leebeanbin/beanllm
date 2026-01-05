"""
DeepSeek Provider
DeepSeek API 통합 (OpenAI 호환 API 사용)

DeepSeek-V3:
- 671B 전체 파라미터, 37B 활성화 (MoE)
- 오픈소스 모델 중 최고 성능
- OpenAI 호환 API 제공
- 모델: deepseek-chat (일반), deepseek-reasoner (사고 모드)
"""

import sys
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

# 선택적 의존성
try:
    from openai import APIError, APITimeoutError, AsyncOpenAI
except ImportError:
    APIError = Exception  # type: ignore
    APITimeoutError = Exception  # type: ignore
    AsyncOpenAI = None  # type: ignore

sys.path.insert(0, str(Path(__file__).parent.parent))

from beanllm.utils.config import EnvConfig
from beanllm.utils.exceptions import ProviderError
from beanllm.utils.logger import get_logger
from beanllm.utils.retry import retry

from .base_provider import BaseLLMProvider, LLMResponse

logger = get_logger(__name__)


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek 제공자 (OpenAI 호환 API)"""

    def __init__(self, config: Dict = None):
        super().__init__(config or {})

        if AsyncOpenAI is None:
            raise ImportError(
                "openai package is required for DeepSeekProvider. "
                "Install it with: pip install openai or poetry add openai"
            )

        # API 키 확인
        api_key = EnvConfig.DEEPSEEK_API_KEY
        if not api_key:
            raise ValueError("DeepSeek is not available. Please set DEEPSEEK_API_KEY")

        # AsyncOpenAI 클라이언트 생성 (DeepSeek base URL 사용)
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=300.0,  # 5분 타임아웃
        )
        self.default_model = "deepseek-chat"

        # 모델 목록
        self._available_models = [
            "deepseek-chat",  # 일반 대화
            "deepseek-reasoner",  # 사고 모드 (복잡한 추론)
        ]

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """스트리밍 채팅 (OpenAI 호환 API)"""
        try:
            openai_messages = messages.copy()
            if system:
                openai_messages.insert(0, {"role": "system", "content": system})

            request_params = {
                "model": model or self.default_model,
                "messages": openai_messages,
                "stream": True,
                "temperature": temperature,
            }

            if max_tokens is not None:
                request_params["max_tokens"] = max_tokens

            response = await self.client.chat.completions.create(**request_params)

            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except (APIError, APITimeoutError) as e:
            logger.error(f"DeepSeek API error: {str(e)}")
            raise ProviderError(f"DeepSeek API error: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error in DeepSeek stream_chat: {str(e)}")
            raise ProviderError(f"Unexpected error: {str(e)}") from e

    @retry(max_attempts=3, delay=1.0, backoff=2.0)
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """일반 채팅 (비스트리밍)"""
        try:
            openai_messages = messages.copy()
            if system:
                openai_messages.insert(0, {"role": "system", "content": system})

            request_params = {
                "model": model or self.default_model,
                "messages": openai_messages,
                "stream": False,
                "temperature": temperature,
            }

            if max_tokens is not None:
                request_params["max_tokens"] = max_tokens

            response = await self.client.chat.completions.create(**request_params)

            # 사용량 정보 추출
            usage_info = None
            if hasattr(response, "usage") and response.usage:
                usage_info = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage=usage_info,
            )

        except (APIError, APITimeoutError) as e:
            logger.error(f"DeepSeek API error: {str(e)}")
            raise ProviderError(f"DeepSeek API error: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error in DeepSeek chat: {str(e)}")
            raise ProviderError(f"Unexpected error: {str(e)}") from e

    async def list_models(self) -> List[str]:
        """사용 가능한 모델 목록 조회"""
        return self._available_models

    def is_available(self) -> bool:
        """제공자 사용 가능 여부"""
        try:
            return bool(EnvConfig.DEEPSEEK_API_KEY)
        except Exception:
            return False

    async def health_check(self) -> bool:
        """건강 상태 확인"""
        try:
            # 간단한 채팅으로 건강 상태 확인
            response = await self.chat(
                messages=[{"role": "user", "content": "Hi"}],
                model=self.default_model,
                max_tokens=10,
            )
            return bool(response.content)
        except Exception as e:
            logger.error(f"DeepSeek health check failed: {str(e)}")
            return False

    def __repr__(self) -> str:
        return f"DeepSeekProvider(model={self.default_model})"
