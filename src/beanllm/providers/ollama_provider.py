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

from beanllm.decorators.provider_error_handler import provider_error_handler
from beanllm.utils.config import EnvConfig
from beanllm.utils.exceptions import ProviderError
from beanllm.utils.logging import get_logger
from beanllm.utils.resilience.retry import retry

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
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        스트리밍 채팅 (최신 SDK: AsyncClient.chat() 사용)
        """
        # Rate Limiting (분산 또는 인메모리)
        await self._acquire_rate_limit(f"ollama:{model or self.default_model}", cost=1.0)
        
        try:
            # ParameterAdapter가 max_tokens를 num_predict로 변환함
            # kwargs에서 변환된 파라미터 추출 (우선순위: kwargs > 직접 전달)
            num_predict = kwargs.get("num_predict", max_tokens)
            temperature_param = kwargs.get("temperature", temperature)

            # System message를 messages 배열에 추가 (Ollama는 system 파라미터를 직접 지원하지 않음)
            chat_messages = messages.copy()
            if system:
                chat_messages = [{"role": "system", "content": system}] + chat_messages

            # 최신 SDK: client.chat() 사용
            stream = await self.client.chat(
                model=model or self.default_model,
                messages=chat_messages,
                options={
                    "temperature": temperature_param,
                    "num_predict": num_predict,
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
            logger.error(f"Ollama stream_chat failed: {e}")
            raise

    @retry(max_retries=3, retry_on=(Exception,))
    @provider_error_handler(operation="chat", custom_error_message="Ollama chat failed")
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
        await self._acquire_rate_limit(f"ollama:{model or self.default_model}", cost=1.0)

        try:
            # ParameterAdapter가 max_tokens를 num_predict로 변환함
            # kwargs에서 변환된 파라미터 추출 (우선순위: kwargs > 직접 전달)
            num_predict = kwargs.get("num_predict", max_tokens)
            temperature_param = kwargs.get("temperature", temperature)

            # System message를 messages 배열에 추가 (Ollama는 system 파라미터를 직접 지원하지 않음)
            chat_messages = messages.copy()
            if system:
                chat_messages = [{"role": "system", "content": system}] + chat_messages

            response = await self.client.chat(
                model=model or self.default_model,
                messages=chat_messages,
                options={
                    "temperature": temperature_param,
                    "num_predict": num_predict,
                },
                stream=False,
            )

            return LLMResponse(
                content=response["message"]["content"],
                model=response.get("model", model or self.default_model),
            )
        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
            raise

    async def list_models(self) -> List[str]:
        """사용 가능한 모델 목록"""
        try:
            logger.debug(f"[OllamaProvider] Calling client.list() with host: {self.client.host if hasattr(self.client, 'host') else 'default'}")
            
            # 최신 SDK: list() 메서드 사용
            models_response = await self.client.list()
            
            logger.debug(f"[OllamaProvider] client.list() returned: {models_response} (type: {type(models_response)})")
            
            # models_response 처리: dict, list, ListResponse 객체 모두 처리
            models_list = []
            
            # 1. ListResponse 객체인 경우 (ollama._types.ListResponse)
            if hasattr(models_response, 'models'):
                models_list = models_response.models
                logger.debug(f"[OllamaProvider] Extracted models_list from ListResponse.models: {len(models_list)} models")
            # 2. dict인 경우
            elif isinstance(models_response, dict):
                models_list = models_response.get("models", [])
                logger.debug(f"[OllamaProvider] Extracted models_list from dict: {models_list} (count: {len(models_list)})")
            # 3. list인 경우
            elif isinstance(models_response, list):
                models_list = models_response
                logger.debug(f"[OllamaProvider] models_response is list: {models_list} (count: {len(models_list)})")
            else:
                logger.warning(f"[OllamaProvider] Unexpected response type: {type(models_response)}, value: {models_response}")
                models_list = []
            
            # 안전하게 모델 이름 추출
            model_names = []
            for i, m in enumerate(models_list):
                if m is None:
                    logger.debug(f"[OllamaProvider] Model item {i} is None, skipping")
                    continue
                
                # Model 객체인 경우 (ollama._types.Model)
                if hasattr(m, 'model'):
                    # Model 객체의 model 속성 사용 (예: Model.model = 'phi3.5:latest')
                    name = m.model
                    if name:
                        model_names.append(str(name))
                        logger.debug(f"[OllamaProvider] Extracted model name from Model object: {name}")
                # dict인 경우
                elif isinstance(m, dict):
                    # "name" 키가 있으면 사용, 없으면 다른 키 시도
                    name = m.get("name") or m.get("model") or m.get("id")
                    if name:
                        model_names.append(str(name))
                        logger.debug(f"[OllamaProvider] Extracted model name from dict: {name}")
                    else:
                        logger.debug(f"[OllamaProvider] Model dict {i} has no name/model/id keys: {m}")
                # str인 경우
                elif isinstance(m, str):
                    model_names.append(m)
                    logger.debug(f"[OllamaProvider] Model item {i} is string: {m}")
                else:
                    # 다른 타입이면 문자열로 변환 시도
                    try:
                        # 객체에 model 속성이 있는지 확인
                        if hasattr(m, 'model'):
                            name = m.model
                            model_names.append(str(name))
                            logger.debug(f"[OllamaProvider] Extracted model name from object.model attribute: {name}")
                        else:
                            model_names.append(str(m))
                            logger.debug(f"[OllamaProvider] Converted model item {i} to string: {str(m)}")
                    except Exception as conv_error:
                        logger.debug(f"[OllamaProvider] Failed to convert model item {i} to string: {conv_error}")
                        pass
            
            logger.info(f"[OllamaProvider] list_models() returning {len(model_names)} models: {model_names}")
            return model_names
        except Exception as e:
            logger.error(f"Ollama list_models error: {e}", exc_info=True)
            import traceback
            logger.debug(f"Ollama list_models traceback: {traceback.format_exc()}")
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
