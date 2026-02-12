"""
Claude Provider
Anthropic Claude API 통합 (최신 SDK 사용)
"""

# 독립적인 utils 사용
import sys
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

# 선택적 의존성 - 지연 import
try:
    from anthropic import APIError, APITimeoutError, AsyncAnthropic
except ImportError:
    # anthropic가 설치되지 않은 경우
    APIError = Exception  # type: ignore
    APITimeoutError = Exception  # type: ignore
    AsyncAnthropic = None  # type: ignore

sys.path.insert(0, str(Path(__file__).parent.parent))

from beanllm.decorators.provider_error_handler import provider_error_handler
from beanllm.utils.config import EnvConfig
from beanllm.utils.constants import (
    CLAUDE_DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TEMPERATURE,
    HEALTH_CHECK_MAX_TOKENS,
)
from beanllm.utils.logging import get_logger
from beanllm.utils.resilience.retry import retry

from .base_provider import BaseLLMProvider, LLMResponse

logger = get_logger(__name__)


class ClaudeProvider(BaseLLMProvider):
    """Claude 제공자 (최신 SDK: messages.stream() 사용)"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config or {})
        if AsyncAnthropic is None:
            raise ImportError(
                "anthropic package is required. Install it with: pip install anthropic"
            )

        api_key = EnvConfig.ANTHROPIC_API_KEY
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for Claude provider")

        # 타임아웃 설정 (5분)
        self.client = AsyncAnthropic(
            api_key=api_key,
            timeout=300.0,  # 5분 타임아웃
        )
        self.default_model = "claude-3-5-sonnet-20241022"

    @retry(max_retries=DEFAULT_MAX_RETRIES, retry_on=(Exception,))
    @provider_error_handler(
        operation="stream_chat",
        api_error_types=(APITimeoutError, APIError),
        custom_error_message="Claude stream_chat API error",
    )
    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        system: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """
        스트리밍 채팅 (최신 SDK: messages.stream() 사용, 재시도 로직 포함)
        """
        # Rate Limiting (분산 또는 인메모리)
        await self._acquire_rate_limit(f"claude:{model or self.default_model}", cost=1.0)

        # Claude 메시지 형식 변환
        claude_messages = []
        for msg in messages:
            if msg["role"] == "user":
                claude_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                claude_messages.append({"role": "assistant", "content": msg["content"]})

        # 최신 SDK: messages.stream() 사용
        async with self.client.messages.stream(
            model=model or self.default_model,
            max_tokens=max_tokens or CLAUDE_DEFAULT_MAX_TOKENS,
            system=system,
            temperature=temperature,
            messages=claude_messages,
        ) as stream:
            # text_stream을 통해 텍스트만 추출
            async for text in stream.text_stream:
                if text:
                    yield text

    @retry(max_retries=DEFAULT_MAX_RETRIES, retry_on=(Exception,))
    @provider_error_handler(
        operation="chat",
        api_error_types=(APITimeoutError, APIError),
        custom_error_message="Claude API error",
    )
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        system: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """일반 채팅 (비스트리밍, 재시도 로직 포함)"""
        # Rate Limiting (분산 또는 인메모리)
        await self._acquire_rate_limit(f"claude:{model or self.default_model}", cost=1.0)

        claude_messages = []
        for msg in messages:
            if msg["role"] in ["user", "assistant"]:
                claude_messages.append({"role": msg["role"], "content": msg["content"]})

        response = await self.client.messages.create(
            model=model or self.default_model,
            max_tokens=max_tokens or CLAUDE_DEFAULT_MAX_TOKENS,
            system=system,
            temperature=temperature,
            messages=claude_messages,
        )

        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    async def list_models(self) -> List[str]:
        """사용 가능한 모델 목록"""
        return [
            # Claude 3 Series
            "claude-3-5-sonnet-20241022",  # Latest Sonnet 3.5
            "claude-3-5-haiku-20241022",  # Latest Haiku 3.5
            "claude-3-opus-20240229",  # Opus 3
            "claude-3-sonnet-20240229",  # Sonnet 3.0
            "claude-3-haiku-20240307",  # Haiku 3.0
            # Claude 4 Series (2025)
            "claude-opus-4",  # Opus 4
            "claude-sonnet-4",  # Sonnet 4
            "claude-haiku-4",  # Haiku 4
            # Claude 4.1 Series (2025)
            "claude-opus-4-1",  # Opus 4.1
            "claude-sonnet-4-1",  # Sonnet 4.1
            # Claude 4.5 Series (2025)
            "claude-opus-4-5",  # Opus 4.5 (with thinking/effort)
            "claude-sonnet-4-5",  # Sonnet 4.5 (with thinking)
            "claude-haiku-4-5",  # Haiku 4.5
        ]

    def is_available(self) -> bool:
        """사용 가능 여부"""
        return EnvConfig.ANTHROPIC_API_KEY is not None

    async def health_check(self) -> bool:
        """건강 상태 확인"""
        try:
            response = await self.client.messages.create(
                model=self.default_model,
                max_tokens=HEALTH_CHECK_MAX_TOKENS,
                messages=[{"role": "user", "content": "test"}],
            )
            return response.content[0].text is not None
        except Exception as e:
            logger.error(f"Claude health check failed: {e}")
            return False
