"""
OpenAI Provider
OpenAI API 통합 (최신 SDK: AsyncOpenAI 사용)
"""

# 독립적인 utils 사용
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

from ...utils.config import EnvConfig
from ...utils.exceptions import ProviderError
from ...utils.logger import get_logger
from ...utils.retry import retry
from .base_provider import BaseLLMProvider, LLMResponse

logger = get_logger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI 제공자"""

    def __init__(self, config: Dict = None):
        super().__init__(config or {})

        if AsyncOpenAI is None:
            raise ImportError(
                "openai package is required for OpenAIProvider. "
                "Install it with: pip install openai or poetry add openai"
            )

        # API 키 확인
        api_key = EnvConfig.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OpenAI is not available. Please set OPENAI_API_KEY")

        # AsyncOpenAI 클라이언트 직접 생성
        self.client = AsyncOpenAI(api_key=api_key, timeout=300.0)  # 5분 타임아웃
        self.default_model = "gpt-4o-mini"

        # 모델 목록 캐싱 (성능 최적화)
        self._models_cache = None
        self._models_cache_time = None
        self._models_cache_ttl = 3600  # 1시간 캐싱

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """
        스트리밍 채팅 (최신 SDK: AsyncOpenAI 사용)
        temperature 기본값: 0.0 (사용자 요청)
        """
        try:
            openai_messages = messages.copy()
            if system:
                openai_messages.insert(0, {"role": "system", "content": system})

            # 최신 SDK: AsyncOpenAI.chat.completions.create 사용
            # 모델별 파라미터 지원 여부 확인
            request_params = {
                "model": model or self.default_model,
                "messages": openai_messages,
                "stream": True,
            }

            # 모델별 파라미터 설정 가져오기
            model_name = model or self.default_model
            param_config = self._get_model_parameter_config(model_name)

            logger.debug(
                f"Model {model_name} param config: temp={param_config['supports_temperature']}, "
                f"max_tokens={param_config['supports_max_tokens']}, "
                f"max_completion={param_config['uses_max_completion_tokens']}"
            )

            # temperature: 모델이 지원하는 경우에만 전달 (기본값 0.0)
            if param_config["supports_temperature"]:
                request_params["temperature"] = temperature
            else:
                logger.debug(f"Model {model_name} does not support temperature, skipping")

            # max_tokens/max_completion_tokens: 모델에 맞게 처리
            if max_tokens is not None:
                if param_config["uses_max_completion_tokens"]:
                    # gpt-5, gpt-4.1 시리즈는 max_completion_tokens 사용
                    request_params["max_completion_tokens"] = max_tokens
                elif param_config["supports_max_tokens"]:
                    # 일반 모델은 max_tokens 사용
                    request_params["max_tokens"] = max_tokens

            stream = await self.client.chat.completions.create(**request_params)

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI stream_chat error: {e}")
            yield f"[Error: {str(e)}]"

    def _get_model_parameter_config(self, model: str) -> Dict[str, bool]:
        """
        모델의 파라미터 지원 정보를 가져옴
        ModelConfig에서 먼저 확인하고, 없으면 패턴 기반으로 추론

        날짜가 포함된 모델 이름 (예: gpt-5-nano-2025-08-07)도 처리

        Args:
            model: 모델 이름

        Returns:
            파라미터 지원 정보 딕셔너리
        """
        import re

        # ModelConfig에서 먼저 확인 (정확한 이름) - 선택적 의존성
        try:
            from .._source_models.model_config import ModelConfigManager

            config = ModelConfigManager.get_model_config(model)
            if config:
                return {
                    "supports_temperature": config.supports_temperature,
                    "supports_max_tokens": config.supports_max_tokens,
                    "uses_max_completion_tokens": config.uses_max_completion_tokens,
                }
        except ImportError:
            # ModelConfigManager가 없으면 패턴 기반으로 진행
            logger.debug("ModelConfigManager not available, using pattern-based inference")
            pass

        # 날짜가 포함된 모델 이름에서 기본 모델 이름 추출 (예: gpt-5-nano-2025-08-07 -> gpt-5-nano)
        # 패턴: 모델명-날짜 형식
        # 여러 패턴 시도: -2025-08-07, -2025-01-31, -2024-07-18 등
        base_model = model
        # YYYY-MM-DD 형식 제거
        base_model = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", base_model)
        # YYYY 형식 제거
        base_model = re.sub(r"-\d{4}$", "", base_model)

        # 날짜가 제거되었고 원본과 다르면 다시 확인
        if base_model != model:
            logger.debug(f"Extracted base model from {model}: {base_model}")
            try:
                from .._source_models.model_config import ModelConfigManager

                config = ModelConfigManager.get_model_config(base_model)
                if config:
                    logger.debug(
                        f"Found config for {base_model}: temp={config.supports_temperature}, "
                        f"max_tokens={config.supports_max_tokens}, "
                        f"max_completion={config.uses_max_completion_tokens}"
                    )
                    return {
                        "supports_temperature": config.supports_temperature,
                        "supports_max_tokens": config.supports_max_tokens,
                        "uses_max_completion_tokens": config.uses_max_completion_tokens,
                    }
            except Exception:
                pass  # ModelConfig 없음, 패턴 기반으로 진행

        # ModelConfig에 없으면 패턴 기반으로 추론 (동적으로 발견된 모델용)
        # 날짜가 제거된 base_model을 사용 (없으면 원본 model 사용)
        model_for_pattern = base_model if base_model != model else model
        model_lower = model_for_pattern.lower()

        logger.debug(f"Using pattern-based inference for {model} (base: {model_for_pattern})")

        # gpt-5, gpt-4.1 시리즈는 max_completion_tokens 사용
        uses_max_completion_tokens = "gpt-5" in model_lower or "gpt-4.1" in model_lower

        # nano, mini, o3, o4는 temperature 미지원 (기본값 1만 지원)
        supports_temperature = not any(x in model_lower for x in ["nano", "mini", "o3", "o4"])

        # max_tokens 지원 여부 (nano, gpt-5, gpt-4.1는 max_tokens 미지원)
        supports_max_tokens = not any(x in model_lower for x in ["nano", "gpt-5", "gpt-4.1"])

        logger.debug(
            f"Pattern-based config for {model}: temp={supports_temperature}, "
            f"max_tokens={supports_max_tokens}, max_completion={uses_max_completion_tokens}"
        )

        return {
            "supports_temperature": supports_temperature,
            "supports_max_tokens": supports_max_tokens,
            "uses_max_completion_tokens": uses_max_completion_tokens,
        }

    @retry(max_attempts=3, exceptions=(APITimeoutError, APIError, Exception))
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        일반 채팅 (비스트리밍, 최신 SDK 사용, 재시도 로직 포함)
        temperature 기본값: 0.0 (사용자 요청)
        """
        try:
            openai_messages = messages.copy()
            if system:
                openai_messages.insert(0, {"role": "system", "content": system})

            # 최신 SDK: AsyncOpenAI.chat.completions.create 사용
            # 모델별 파라미터 지원 여부 확인
            request_params = {
                "model": model or self.default_model,
                "messages": openai_messages,
            }

            # 모델별 파라미터 설정 가져오기
            model_name = model or self.default_model
            param_config = self._get_model_parameter_config(model_name)

            logger.debug(
                f"Model {model_name} param config: temp={param_config['supports_temperature']}, "
                f"max_tokens={param_config['supports_max_tokens']}, "
                f"max_completion={param_config['uses_max_completion_tokens']}"
            )

            # temperature: 모델이 지원하는 경우에만 전달 (기본값 0.0)
            if param_config["supports_temperature"]:
                request_params["temperature"] = temperature
            else:
                logger.debug(f"Model {model_name} does not support temperature, skipping")

            # max_tokens/max_completion_tokens: 모델에 맞게 처리
            if max_tokens is not None:
                if param_config["uses_max_completion_tokens"]:
                    # gpt-5, gpt-4.1 시리즈는 max_completion_tokens 사용
                    request_params["max_completion_tokens"] = max_tokens
                elif param_config["supports_max_tokens"]:
                    # 일반 모델은 max_tokens 사용
                    request_params["max_tokens"] = max_tokens

            response = await self.client.chat.completions.create(**request_params)

            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            )
        except (APITimeoutError, APIError) as e:
            logger.error(f"OpenAI chat error: {e}")
            raise ProviderError(f"OpenAI API error: {str(e)}") from e
        except Exception as e:
            logger.error(f"OpenAI chat error: {e}")
            raise ProviderError(f"OpenAI chat failed: {str(e)}") from e

    async def list_models(self) -> List[str]:
        """
        OpenAI API에서 실제 사용 가능한 모델 목록을 가져옴 (캐싱 적용)
        """
        import time

        # 캐시 확인
        current_time = time.time()
        if (
            self._models_cache is not None
            and self._models_cache_time is not None
            and (current_time - self._models_cache_time) < self._models_cache_ttl
        ):
            logger.debug(f"Using cached OpenAI models: {len(self._models_cache)} models")
            return self._models_cache

        try:
            # OpenAI API에서 모델 목록 가져오기
            models_response = await self.client.models.list()
            model_ids = [model.id for model in models_response.data]

            # 캐시 저장
            self._models_cache = model_ids
            self._models_cache_time = current_time

            logger.debug(f"OpenAI API models: {len(model_ids)} models found (cached)")
            return model_ids
        except Exception as e:
            logger.warning(f"Failed to fetch OpenAI models from API: {e}, using default list")
            # API 호출 실패 시 기본 목록 반환
            default_models = [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
            ]
            # 기본 목록도 캐싱 (짧은 시간)
            self._models_cache = default_models
            self._models_cache_time = current_time
            return default_models

    def find_lightweight_model(self, available_models: List[str]) -> Optional[str]:
        """
        사용 가능한 모델 목록에서 경량 모델을 찾음
        2025년 12월 15일 기준 실제 API 모델 목록 기반:
        - nano (가장 작음): gpt-5-nano-2025-08-07, gpt-5-nano, gpt-4.1-nano-2025-04-14, gpt-4.1-nano
        - mini: gpt-5-mini-2025-08-07, gpt-5-mini,
          gpt-4.1-mini-2025-04-14, gpt-4.1-mini,
          gpt-4o-mini-2024-07-18, gpt-4o-mini
        - o3-mini-2025-01-31, o3-mini
        - o4-mini-2025-04-16, o4-mini

        우선순위: nano (최신) > mini (최신) > o3-mini > o4-mini

        Args:
            available_models: 사용 가능한 모델 ID 리스트

        Returns:
            경량 모델 이름 또는 None
        """
        if not available_models:
            return None

        # 채팅용 모델만 필터링
        # (text-embedding, tts, dall-e, whisper, codex, audio, realtime, search, image 등 제외)
        excluded_keywords = [
            "embedding",
            "tts",
            "dall-e",
            "whisper",
            "codex",
            "transcribe",
            "audio",
            "realtime",
            "search",
            "image",
            "moderation",
            "diarize",
        ]
        chat_models = [
            m
            for m in available_models
            if (
                (m.startswith("gpt-") or m.startswith("o"))
                and not any(x in m.lower() for x in excluded_keywords)
                and not m.endswith("-tts")
                and not m.endswith("-transcribe")
            )
        ]

        if not chat_models:
            return None

        # 경량 모델 우선순위 (작은 것부터, 최신 버전 우선)
        # 1순위: nano (가장 작음) - gpt-5-nano > gpt-4.1-nano
        nano_models = [m for m in chat_models if "nano" in m.lower()]
        if nano_models:
            # gpt-5-nano 우선, 그 다음 날짜가 있는 버전
            gpt5_nano = [m for m in nano_models if "gpt-5" in m]
            if gpt5_nano:
                # 날짜가 있는 버전 우선
                dated = [m for m in gpt5_nano if any(c.isdigit() for c in m[-10:])]
                if dated:
                    dated.sort(reverse=True)
                    selected = dated[0]
                    logger.info(f"Found lightweight model (gpt-5-nano): {selected}")
                    return selected
                else:
                    selected = gpt5_nano[0]
                    logger.info(f"Found lightweight model (gpt-5-nano): {selected}")
                    return selected

            # gpt-4.1-nano
            gpt41_nano = [m for m in nano_models if "gpt-4.1" in m]
            if gpt41_nano:
                dated = [m for m in gpt41_nano if any(c.isdigit() for c in m[-10:])]
                if dated:
                    dated.sort(reverse=True)
                    selected = dated[0]
                    logger.info(f"Found lightweight model (gpt-4.1-nano): {selected}")
                    return selected
                else:
                    selected = gpt41_nano[0]
                    logger.info(f"Found lightweight model (gpt-4.1-nano): {selected}")
                    return selected

        # 2순위: mini - gpt-5-mini > gpt-4.1-mini > gpt-4o-mini
        # 채팅용 mini 모델만 (audio, realtime, search, codex 등 제외)
        mini_models = [
            m
            for m in chat_models
            if "mini" in m.lower()
            and "nano" not in m.lower()
            and not any(
                x in m.lower()
                for x in ["audio", "realtime", "search", "codex", "transcribe", "tts"]
            )
        ]
        if mini_models:
            # gpt-5-mini 우선
            gpt5_mini = [m for m in mini_models if "gpt-5" in m]
            if gpt5_mini:
                dated = [m for m in gpt5_mini if any(c.isdigit() for c in m[-10:])]
                if dated:
                    dated.sort(reverse=True)
                    selected = dated[0]
                    logger.info(f"Found lightweight model (gpt-5-mini): {selected}")
                    return selected
                else:
                    selected = gpt5_mini[0]
                    logger.info(f"Found lightweight model (gpt-5-mini): {selected}")
                    return selected

            # gpt-4.1-mini
            gpt41_mini = [m for m in mini_models if "gpt-4.1" in m]
            if gpt41_mini:
                dated = [m for m in gpt41_mini if any(c.isdigit() for c in m[-10:])]
                if dated:
                    dated.sort(reverse=True)
                    selected = dated[0]
                    logger.info(f"Found lightweight model (gpt-4.1-mini): {selected}")
                    return selected
                else:
                    selected = gpt41_mini[0]
                    logger.info(f"Found lightweight model (gpt-4.1-mini): {selected}")
                    return selected

            # gpt-4o-mini
            gpt4o_mini = [m for m in mini_models if "gpt-4o" in m]
            if gpt4o_mini:
                dated = [m for m in gpt4o_mini if any(c.isdigit() for c in m[-10:])]
                if dated:
                    dated.sort(reverse=True)
                    selected = dated[0]
                    logger.info(f"Found lightweight model (gpt-4o-mini): {selected}")
                    return selected
                else:
                    selected = gpt4o_mini[0]
                    logger.info(f"Found lightweight model (gpt-4o-mini): {selected}")
                    return selected

        # 3순위: o3-mini, o4-mini
        o3_mini = [m for m in chat_models if "o3-mini" in m.lower()]
        if o3_mini:
            dated = [m for m in o3_mini if any(c.isdigit() for c in m[-10:])]
            if dated:
                dated.sort(reverse=True)
                selected = dated[0]
                logger.info(f"Found lightweight model (o3-mini): {selected}")
                return selected
            else:
                selected = o3_mini[0]
                logger.info(f"Found lightweight model (o3-mini): {selected}")
                return selected

        o4_mini = [m for m in chat_models if "o4-mini" in m.lower()]
        if o4_mini:
            dated = [m for m in o4_mini if any(c.isdigit() for c in m[-10:])]
            if dated:
                dated.sort(reverse=True)
                selected = dated[0]
                logger.info(f"Found lightweight model (o4-mini): {selected}")
                return selected
            else:
                selected = o4_mini[0]
                logger.info(f"Found lightweight model (o4-mini): {selected}")
                return selected

        return None

    def is_available(self) -> bool:
        """사용 가능 여부"""
        return EnvConfig.is_provider_available("openai")

    async def health_check(self) -> bool:
        """건강 상태 확인"""
        try:
            # 간단한 테스트 요청
            response = await self.client.chat.completions.create(
                model=self.default_model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
            )
            return response.choices[0].message.content is not None
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False
