"""
Perplexity Provider
Perplexity AI API 통합 (실시간 웹 검색 + LLM)

Perplexity Sonar:
- Llama 3.3 70B 기반
- 실시간 웹 검색 + LLM 통합
- 1200 토큰/초 속도
- Search Arena 평가 1위 (GPT-4o Search, Gemini 2.0 Flash 능가)
- 모델: sonar, sonar-pro, sonar-reasoning-pro
- 상세한 인용 제공 (2025년부터 인용 토큰 무료)
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


class PerplexityProvider(BaseLLMProvider):
    """Perplexity 제공자 (실시간 웹 검색 + LLM)"""

    def __init__(self, config: Dict = None):
        super().__init__(config or {})

        if AsyncOpenAI is None:
            raise ImportError(
                "openai package is required for PerplexityProvider. "
                "Install it with: pip install openai or poetry add openai"
            )

        # API 키 확인
        api_key = EnvConfig.PERPLEXITY_API_KEY
        if not api_key:
            raise ValueError("Perplexity is not available. Please set PERPLEXITY_API_KEY")

        # AsyncOpenAI 클라이언트 생성 (Perplexity base URL 사용)
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai",
            timeout=300.0,  # 5분 타임아웃
        )
        self.default_model = "sonar"

        # 모델 목록
        self._available_models = [
            "sonar",  # Llama 3.3 70B 기반, 실시간 웹 검색
            "sonar-pro",  # 심층 검색 및 후속 질문
            "sonar-reasoning-pro",  # 복잡한 분석 작업용 프리미엄
        ]

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """스트리밍 채팅 (실시간 웹 검색 포함)"""
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
            logger.error(f"Perplexity API error: {str(e)}")
            raise ProviderError(f"Perplexity API error: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error in Perplexity stream_chat: {str(e)}")
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
        """일반 채팅 (비스트리밍, 실시간 웹 검색 포함)"""
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

            # Perplexity는 citations (인용) 제공
            content = response.choices[0].message.content

            # citations가 있으면 메타데이터에 포함
            if hasattr(response, "citations"):
                if usage_info is None:
                    usage_info = {}
                usage_info["citations"] = response.citations

            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage_info,
            )

        except (APIError, APITimeoutError) as e:
            logger.error(f"Perplexity API error: {str(e)}")
            raise ProviderError(f"Perplexity API error: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error in Perplexity chat: {str(e)}")
            raise ProviderError(f"Unexpected error: {str(e)}") from e

    async def list_models(self) -> List[str]:
        """사용 가능한 모델 목록 조회"""
        return self._available_models

    def is_available(self) -> bool:
        """제공자 사용 가능 여부"""
        try:
            return bool(EnvConfig.PERPLEXITY_API_KEY)
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
            logger.error(f"Perplexity health check failed: {str(e)}")
            return False

    def __repr__(self) -> str:
        return f"PerplexityProvider(model={self.default_model})"
