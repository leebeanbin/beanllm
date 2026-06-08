"""
Grok Provider
xAI Grok API 통합 (OpenAI 호환 API 사용)

Grok 4.3:
- xAI의 최신 reasoning model
- OpenAI 호환 API 제공 (base_url: https://api.x.ai/v1)
- 실시간 웹 검색 + 추론 능력 결합
"""

import sys
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

try:
    from openai import APIError, APITimeoutError, AsyncOpenAI
except ImportError:
    APIError = Exception  # type: ignore
    APITimeoutError = Exception  # type: ignore
    AsyncOpenAI = None  # type: ignore

sys.path.insert(0, str(Path(__file__).parent.parent))

from beanllm.decorators.provider_error_handler import provider_error_handler
from beanllm.utils.config import EnvConfig
from beanllm.utils.constants import DEFAULT_MAX_RETRIES
from beanllm.utils.logging import get_logger
from beanllm.utils.resilience.retry import retry

from .base_provider import BaseLLMProvider, LLMResponse

logger = get_logger(__name__)


class GrokProvider(BaseLLMProvider):
    """Grok/xAI 제공자 (OpenAI 호환 API)"""

    BASE_URL = "https://api.x.ai/v1"
    DEFAULT_MODEL = "grok-4"

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config or {})

        if AsyncOpenAI is None:
            raise ImportError(
                "openai package is required for GrokProvider. "
                "Install it with: pip install openai"
            )

        api_key = EnvConfig.XAI_API_KEY
        if not api_key:
            raise ValueError("Grok/xAI is not available. Please set XAI_API_KEY")

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.BASE_URL,
            timeout=300.0,
        )
        self.default_model = self.DEFAULT_MODEL

        self._available_models = [
            "grok-4",
            "grok-4-0709",
            "grok-3",
            "grok-3-mini",
            "grok-3-fast",
            "grok-2-vision-1212",
        ]

    @provider_error_handler(
        operation="stream_chat",
        api_error_types=(APIError, APITimeoutError),
        custom_error_message="Grok API error",
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
        await self._acquire_rate_limit(f"grok:{model or self.default_model}", cost=1.0)

        try:
            openai_messages = messages.copy()
            if system:
                openai_messages.insert(0, {"role": "system", "content": system})

            temperature_param = kwargs.get("temperature", temperature)
            max_tokens_param = kwargs.get("max_tokens", max_tokens)

            request_params: Dict = {
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
            logger.error(f"Grok stream_chat failed: {e}")
            raise

    @retry(max_retries=DEFAULT_MAX_RETRIES, initial_delay=1.0)
    @provider_error_handler(
        operation="chat",
        api_error_types=(APIError, APITimeoutError),
        custom_error_message="Grok API error",
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
        await self._acquire_rate_limit(f"grok:{model or self.default_model}", cost=1.0)

        openai_messages = messages.copy()
        if system:
            openai_messages.insert(0, {"role": "system", "content": system})

        temperature_param = kwargs.get("temperature", temperature)
        max_tokens_param = kwargs.get("max_tokens", max_tokens)

        request_params: Dict = {
            "model": model or self.default_model,
            "messages": openai_messages,
            "stream": False,
            "temperature": temperature_param,
        }

        if max_tokens_param is not None:
            request_params["max_tokens"] = max_tokens_param

        response = await self.client.chat.completions.create(**request_params)

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
        return self._available_models

    def is_available(self) -> bool:
        try:
            return bool(EnvConfig.XAI_API_KEY)
        except Exception:
            return False

    async def health_check(self) -> bool:
        try:
            response = await self.chat(
                messages=[{"role": "user", "content": "Hi"}],
                model=self.default_model,
                max_tokens=10,
            )
            return bool(response.content)
        except Exception as e:
            logger.error(f"Grok health check failed: {str(e)}")
            return False

    def __repr__(self) -> str:
        return f"GrokProvider(model={self.default_model})"
