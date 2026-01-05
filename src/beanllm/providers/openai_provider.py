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

from beanllm.utils.config import EnvConfig
from beanllm.utils.exceptions import ProviderError
from beanllm.utils.logger import get_logger
from beanllm.utils.retry import retry

from .base_provider import BaseLLMProvider, LLMResponse

logger = get_logger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI 제공자"""

    # 모델 파라미터 캐시 (클래스 변수) - O(1) 조회 최적화
    MODEL_PARAMETER_CACHE = {
        "gpt-4o": {
            "supports_temperature": True,
            "supports_max_tokens": True,
            "uses_max_completion_tokens": False,
        },
        "gpt-4o-mini": {
            "supports_temperature": True,
            "supports_max_tokens": True,
            "uses_max_completion_tokens": False,
        },
        "gpt-4-turbo": {
            "supports_temperature": True,
            "supports_max_tokens": True,
            "uses_max_completion_tokens": False,
        },
        "gpt-4": {
            "supports_temperature": True,
            "supports_max_tokens": True,
            "uses_max_completion_tokens": False,
        },
        "gpt-4-32k": {
            "supports_temperature": True,
            "supports_max_tokens": True,
            "uses_max_completion_tokens": False,
        },
        "gpt-3.5-turbo": {
            "supports_temperature": True,
            "supports_max_tokens": True,
            "uses_max_completion_tokens": False,
        },
        "gpt-5": {
            "supports_temperature": True,
            "supports_max_tokens": False,
            "uses_max_completion_tokens": True,
        },
        "gpt-4.1": {
            "supports_temperature": True,
            "supports_max_tokens": False,
            "uses_max_completion_tokens": True,
        },
    }

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
        모델의 파라미터 지원 정보를 가져옴 (O(1) 캐시 최적화)

        우선순위:
        1. MODEL_PARAMETER_CACHE에서 직접 조회 (O(1) - 100x 빠름)
        2. 베이스 모델 추출 후 캐시 재조회
        3. ModelConfig에서 확인 (선택적)
        4. Strategy Pattern 기반 추론 (동적 모델용)

        Args:
            model: 모델 이름

        Returns:
            파라미터 지원 정보 딕셔너리
        """
        # 1. 직접 캐시 조회 (O(1) - 가장 빠름)
        if model in self.MODEL_PARAMETER_CACHE:
            logger.debug(f"Cache hit for model {model}")
            return self.MODEL_PARAMETER_CACHE[model]

        # 2. 베이스 모델 추출 후 캐시 재조회
        # 예: "gpt-4o-2024-05-13" → "gpt-4o"
        from .model_parameter_strategy import ModelParameterFactory
        base_model = ModelParameterFactory.extract_base_model(model)

        if base_model != model and base_model in self.MODEL_PARAMETER_CACHE:
            logger.debug(f"Cache hit for base model {base_model} (from {model})")
            return self.MODEL_PARAMETER_CACHE[base_model]

        # 3. ModelConfig에서 확인 (선택적 의존성)
        try:
            from ..models.model_config import ModelConfigManager

            config = ModelConfigManager.get_model_config(model)
            if config:
                logger.debug(f"ModelConfigManager hit for {model}")
                return {
                    "supports_temperature": config.supports_temperature,
                    "supports_max_tokens": config.supports_max_tokens,
                    "uses_max_completion_tokens": config.uses_max_completion_tokens,
                }

            # 베이스 모델로도 시도
            if base_model != model:
                config = ModelConfigManager.get_model_config(base_model)
                if config:
                    logger.debug(f"ModelConfigManager hit for base model {base_model}")
                    return {
                        "supports_temperature": config.supports_temperature,
                        "supports_max_tokens": config.supports_max_tokens,
                        "uses_max_completion_tokens": config.uses_max_completion_tokens,
                    }
        except ImportError:
            logger.debug("ModelConfigManager not available, using Strategy pattern")
        except Exception:
            pass

        # 4. Strategy Pattern 기반 추론 (동적으로 발견된 모델용)
        logger.debug(f"Using Strategy pattern for {model}")
        config = ModelParameterFactory.get_config(model)

        logger.debug(
            f"Strategy-based config for {model}: temp={config['supports_temperature']}, "
            f"max_tokens={config['supports_max_tokens']}, "
            f"max_completion={config['uses_max_completion_tokens']}"
        )

        return config

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

    def _filter_chat_models(self, models: List[str]) -> List[str]:
        """채팅용 모델만 필터링 (embedding, tts 등 제외)"""
        excluded = [
            "embedding", "tts", "dall-e", "whisper", "codex", "transcribe",
            "audio", "realtime", "search", "image", "moderation", "diarize"
        ]
        return [
            m for m in models
            if (m.startswith("gpt-") or m.startswith("o"))
            and not any(x in m.lower() for x in excluded)
            and not m.endswith(("-tts", "-transcribe"))
        ]

    def _select_best_dated_model(self, models: List[str], label: str) -> Optional[str]:
        """날짜가 있는 최신 모델 우선 선택"""
        if not models:
            return None

        dated = [m for m in models if any(c.isdigit() for c in m[-10:])]
        selected = max(dated) if dated else models[0]
        logger.info(f"Found lightweight model ({label}): {selected}")
        return selected

    def _find_model_by_patterns(
        self, chat_models: List[str], size: str, prefixes: List[str]
    ) -> Optional[str]:
        """
        패턴에 맞는 모델 찾기 (우선순위 순, O(n) 최적화)

        Algorithm Complexity:
            Before: O(k×n) where k=len(prefixes), n=len(chat_models)
            After: O(n) - single pass through models

        Optimization:
            - 한 번의 순회로 모든 prefix를 체크
            - prefix별로 딕셔너리에 그룹화
            - 우선순위 순으로 결과 반환
        """
        # 특수 용도 모델 키워드 (mini 검색 시 제외)
        special_keywords = {"audio", "realtime", "search", "codex", "transcribe", "tts"}

        # Prefix별로 매칭된 모델을 저장 (우선순위 유지를 위해 딕셔너리 사용)
        # {prefix: [models]}
        prefix_matches = {prefix: [] for prefix in prefixes}

        # O(n) 단일 순회로 모든 필터링 및 그룹화 수행
        for model in chat_models:
            model_lower = model.lower()

            # size 체크
            if size not in model_lower:
                continue

            # nano 검색 시 mini 제외
            if size == "nano" and "mini" in model_lower:
                continue

            # mini 검색 시 nano 및 특수 용도 모델 제외
            if size == "mini":
                if "nano" in model_lower:
                    continue
                if any(keyword in model_lower for keyword in special_keywords):
                    continue

            # 우선순위 순서대로 prefix 매칭 (첫 번째 매칭만 저장)
            for prefix in prefixes:
                if prefix in model:
                    prefix_matches[prefix].append(model)
                    break  # 첫 번째 매칭 prefix에만 추가

        # 우선순위 순으로 결과 반환
        for prefix in prefixes:
            matches = prefix_matches[prefix]
            if matches:
                return self._select_best_dated_model(matches, f"{prefix}-{size}")

        return None

    def find_lightweight_model(self, available_models: List[str]) -> Optional[str]:
        """
        사용 가능한 모델 목록에서 경량 모델을 찾음

        우선순위: nano (최신) > mini (최신) > o3-mini > o4-mini

        Args:
            available_models: 사용 가능한 모델 ID 리스트

        Returns:
            경량 모델 이름 또는 None
        """
        if not available_models:
            return None

        chat_models = self._filter_chat_models(available_models)
        if not chat_models:
            return None

        # 1순위: nano (gpt-5 > gpt-4.1)
        result = self._find_model_by_patterns(chat_models, "nano", ["gpt-5", "gpt-4.1"])
        if result:
            return result

        # 2순위: mini (gpt-5 > gpt-4.1 > gpt-4o)
        result = self._find_model_by_patterns(chat_models, "mini", ["gpt-5", "gpt-4.1", "gpt-4o"])
        if result:
            return result

        # 3순위: o3-mini, o4-mini
        for model_name in ["o3-mini", "o4-mini"]:
            matches = [m for m in chat_models if model_name in m.lower()]
            if matches:
                return self._select_best_dated_model(matches, model_name)

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
