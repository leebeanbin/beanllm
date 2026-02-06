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

from beanllm.decorators.provider_error_handler import provider_error_handler
from beanllm.utils.config import EnvConfig
from beanllm.utils.logging import get_logger
from beanllm.utils.resilience.retry import retry

from .base_provider import BaseLLMProvider, LLMResponse

logger = get_logger(__name__)


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek 제공자 (OpenAI 호환 API)"""

    def __init__(self, config: Optional[Dict] = None):
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
            # Core Models
            "deepseek-chat",  # 일반 대화
            "deepseek-reasoner",  # 사고 모드 (복잡한 추론)
            # V3 Series (2025)
            "deepseek-v3-0324",  # V3 (2025.03.24)
            "deepseek-v3-1",  # V3.1
            "deepseek-v3-2",  # V3.2
            # R1 Series (2025) - Reasoning models
            "deepseek-r1",  # R1 Reasoning
            "deepseek-r1-0528",  # R1 (2025.05.28)
        ]

    @provider_error_handler(
        operation="stream_chat",
        api_error_types=(APIError, APITimeoutError),
        custom_error_message="DeepSeek API error",
    )
    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """스트리밍 채팅 (OpenAI 호환 API)"""
        # Rate Limiting (분산 또는 인메모리)
        await self._acquire_rate_limit(f"deepseek:{model or self.default_model}", cost=1.0)

        try:
            openai_messages = messages.copy()
            if system:
                openai_messages.insert(0, {"role": "system", "content": system})

            # kwargs에서 파라미터 추출 (우선순위: kwargs > 직접 전달)
            temperature_param = kwargs.get("temperature", temperature)
            max_tokens_param = kwargs.get("max_tokens", max_tokens)

            request_params = {
                "model": model or self.default_model,
                "messages": openai_messages,
                "stream": True,
                "temperature": temperature_param,
            }

            if max_tokens_param is not None:
                request_params["max_tokens"] = max_tokens_param

            response = await self.client.chat.completions.create(**request_params)

            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"DeepSeek stream_chat failed: {e}")
            raise

    @retry(max_retries=3, initial_delay=1.0)
    @provider_error_handler(
        operation="chat",
        api_error_types=(APIError, APITimeoutError),
        custom_error_message="DeepSeek API error",
    )
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """일반 채팅 (비스트리밍)"""
        # Rate Limiting (분산 또는 인메모리)
        await self._acquire_rate_limit(f"deepseek:{model or self.default_model}", cost=1.0)

        openai_messages = messages.copy()
        if system:
            openai_messages.insert(0, {"role": "system", "content": system})

        # kwargs에서 파라미터 추출 (우선순위: kwargs > 직접 전달)
        temperature_param = kwargs.get("temperature", temperature)
        max_tokens_param = kwargs.get("max_tokens", max_tokens)

        request_params = {
            "model": model or self.default_model,
            "messages": openai_messages,
            "stream": False,
            "temperature": temperature_param,
        }

        if max_tokens_param is not None:
            request_params["max_tokens"] = max_tokens_param

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
