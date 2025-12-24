"""
Ollama Provider
Ollama API 통합 (최신 SDK: ollama 패키지의 AsyncClient 사용)
"""

# 독립적인 utils 사용
import sys
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

# 선택적 의존성
try:
    from ollama import AsyncClient
except ImportError:
    AsyncClient = None  # type: ignore

sys.path.insert(0, str(Path(__file__).parent.parent))

from ...utils.config import EnvConfig
from ...utils.exceptions import ProviderError
from ...utils.logger import get_logger
from ...utils.retry import retry
from .base_provider import BaseLLMProvider, LLMResponse

logger = get_logger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Ollama 제공자"""

    def __init__(self, config: Dict = None):
        if AsyncClient is None:
            raise ImportError(
                "ollama package is required for OllamaProvider. Install it with: pip install ollama"
            )
        super().__init__(config or {})
        config_dict = config or {}
        host = config_dict.get("host") if config_dict else EnvConfig.OLLAMA_HOST
        # 최신 SDK: AsyncClient 사용 (타임아웃 설정)
        self.client = AsyncClient(
            host=host,
            timeout=300.0,  # 5분 타임아웃
        )
        self.default_model = "qwen2.5:7b"

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """
        스트리밍 채팅 (최신 SDK: AsyncClient.chat() 사용)
        """
        try:
            # 최신 SDK: client.chat() 사용
            stream = await self.client.chat(
                model=model or self.default_model,
                messages=messages,
                system=system,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
                stream=True,
            )

            # 스트리밍 응답 처리
            async for part in stream:
                if "message" in part and "content" in part["message"]:
                    content = part["message"]["content"]
                    if content:
                        yield content
        except Exception as e:
            logger.error(f"Ollama stream_chat error: {e}")
            yield f"[Error: {str(e)}]"

    @retry(max_attempts=3, exceptions=(Exception,))
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """일반 채팅 (비스트리밍, 재시도 로직 포함)"""
        try:
            response = await self.client.chat(
                model=model or self.default_model,
                messages=messages,
                system=system,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
                stream=False,
            )

            return LLMResponse(
                content=response["message"]["content"],
                model=response.get("model", model or self.default_model),
            )
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            raise ProviderError(f"Ollama chat failed: {str(e)}") from e

    async def list_models(self) -> List[str]:
        """사용 가능한 모델 목록"""
        try:
            # 최신 SDK: list() 메서드 사용
            models = await self.client.list()
            return [m["name"] for m in models.get("models", [])]
        except Exception as e:
            logger.error(f"Ollama list_models error: {e}")
            return []

    def is_available(self) -> bool:
        """사용 가능 여부"""
        # Ollama는 API 키가 필요 없으므로 연결 확인만
        return True

    async def health_check(self) -> bool:
        """건강 상태 확인"""
        try:
            # list() 호출로 연결 확인
            await self.client.list()
            return True
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    async def close(self):
        """리소스 정리"""
        # AsyncClient는 자동으로 정리됨
        pass
