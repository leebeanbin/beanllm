"""
Gemini Provider
Google Gemini API 통합 (최신 SDK: google-genai 사용)
"""

from typing import AsyncGenerator, Dict, List, Optional

# 선택적 의존성
try:
    from google import genai
except ImportError:
    genai = None  # type: ignore

from beanllm.decorators.provider_error_handler import provider_error_handler
from beanllm.utils.config import EnvConfig
from beanllm.utils.logging import get_logger
from beanllm.utils.resilience.retry import retry

from .base_provider import BaseLLMProvider, LLMResponse

logger = get_logger(__name__)


class GeminiProvider(BaseLLMProvider):
    """Gemini 제공자 (최신 SDK: google-genai 패키지 사용)"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config or {})

        if genai is None:
            raise ImportError(
                "google-generativeai package is required for GeminiProvider. "
                "Install it with: pip install google-generativeai or poetry add google-generativeai"
            )

        api_key = EnvConfig.GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini provider")

        self.client = genai.Client(api_key=api_key)
        self.default_model = "gemini-2.0-flash-exp"

    @retry(max_retries=3, retry_on=(Exception,))
    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        스트리밍 채팅 (최신 SDK: aio.models.generate_content_stream 사용, 재시도 로직 포함)
        """
        # Rate Limiting (분산 또는 인메모리)
        await self._acquire_rate_limit(f"gemini:{model or self.default_model}", cost=1.0)

        try:
            # 메시지를 contents 형식으로 변환
            contents = []
            if system:
                contents.append(system)

            for msg in messages:
                if msg["role"] == "user":
                    contents.append(msg["content"])
                elif msg["role"] == "assistant":
                    contents.append(f"Assistant: {msg['content']}")

            # ParameterAdapter가 max_tokens를 max_output_tokens로 변환함
            # kwargs에서 변환된 파라미터 추출 (우선순위: kwargs > 직접 전달)
            max_output_tokens = kwargs.get("max_output_tokens", max_tokens)
            temperature_param = kwargs.get("temperature", temperature)

            # 최신 SDK: aio.models.generate_content_stream 사용
            async for chunk in await self.client.aio.models.generate_content_stream(
                model=model or self.default_model,
                contents=contents,
                max_output_tokens=max_output_tokens,
                temperature=temperature_param,
            ):
                if hasattr(chunk, "text") and chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Gemini stream_chat failed: {e}")
            raise

    @retry(max_retries=3, retry_on=(Exception,))
    @provider_error_handler(operation="chat", custom_error_message="Gemini chat failed")
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """일반 채팅 (비스트리밍, 재시도 로직 포함)"""
        # Rate Limiting (분산 또는 인메모리)
        await self._acquire_rate_limit(f"gemini:{model or self.default_model}", cost=1.0)

        contents = []
        if system:
            contents.append(system)

        for msg in messages:
            if msg["role"] == "user":
                contents.append(msg["content"])
            elif msg["role"] == "assistant":
                contents.append(f"Assistant: {msg['content']}")

        # ParameterAdapter가 max_tokens를 max_output_tokens로 변환함
        # kwargs에서 변환된 파라미터 추출 (우선순위: kwargs > 직접 전달)
        max_output_tokens = kwargs.get("max_output_tokens", max_tokens)
        temperature_param = kwargs.get("temperature", temperature)

        response = await self.client.aio.models.generate_content(
            model=model or self.default_model,
            contents=contents,
            max_output_tokens=max_output_tokens,
            temperature=temperature_param,
        )

        return LLMResponse(
            content=response.text if hasattr(response, "text") else str(response),
            model=model or self.default_model,
        )

    async def list_models(self) -> List[str]:
        """사용 가능한 모델 목록"""
        return [
            # Gemini 1.5 Series
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            # Gemini 2.0 Series (2025)
            "gemini-2.0-flash-exp",  # Experimental
            "gemini-2.0-flash",
            "gemini-2.0-pro",
            "gemini-2.0-flash-lite",
            # Gemini 2.5 Series (2025)
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            # Gemini 3.0 Series (2025-2026)
            "gemini-3.0-pro",
            "gemini-3.0-deep-think",  # Reasoning model
        ]

    def is_available(self) -> bool:
        """사용 가능 여부"""
        return EnvConfig.GEMINI_API_KEY is not None

    async def health_check(self) -> bool:
        """건강 상태 확인"""
        try:
            response = await self.client.aio.models.generate_content(
                model=self.default_model,
                contents=["test"],
            )
            return hasattr(response, "text") and response.text is not None
        except Exception as e:
            logger.error(f"Gemini health check failed: {e}")
            return False
