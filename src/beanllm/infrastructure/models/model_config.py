"""
Model Configuration
모델 설정 및 관리 (모든 제공자 지원)

데이터는 model_registry_data.py에서 관리.
이 파일은 조회/필터링 로직만 담당 (SRP).
"""

from dataclasses import dataclass
from typing import Dict, Optional

from .llm_provider import LLMProvider


@dataclass
class ModelConfig:
    """모델 설정"""

    name: str
    display_name: str
    provider: LLMProvider  # 제공자
    type: str  # 'slm' or 'llm'
    max_tokens: int
    temperature: float
    description: str
    use_case: str
    # 파라미터 지원 정보 (2025년 12월 15일 기준)
    supports_temperature: bool = True  # temperature 파라미터 지원 여부
    supports_max_tokens: bool = True  # max_tokens 파라미터 지원 여부
    uses_max_completion_tokens: bool = (
        False  # max_completion_tokens 사용 여부 (gpt-5, gpt-4.1 시리즈)
    )


class ModelConfigManager:
    """
    모델 설정 관리자

    책임: 모델 조회 및 필터링만 (SRP)
    데이터: model_registry_data.py에서 로드
    """

    # 역호환: 기존 코드에서 ModelConfigManager.MODELS 접근 지원
    # 지연 로딩으로 순환 import 방지
    MODELS: Dict[str, "ModelConfig"] = {}

    @classmethod
    def _ensure_loaded(cls) -> None:
        """모델 데이터 지연 로딩"""
        if not cls.MODELS:
            from beanllm.infrastructure.models.model_registry_data import (
                MODELS as _MODELS,
            )

            cls.MODELS = _MODELS

    @classmethod
    def get_model_config(cls, model_name: str) -> Optional[ModelConfig]:
        """모델 설정 조회"""
        cls._ensure_loaded()
        return cls.MODELS.get(model_name)

    @classmethod
    def get_models_by_provider(cls, provider: LLMProvider) -> Dict[str, ModelConfig]:
        """제공자별 모델 조회"""
        cls._ensure_loaded()
        return {name: config for name, config in cls.MODELS.items() if config.provider == provider}

    @classmethod
    def get_models_by_type(cls, model_type: str) -> Dict[str, ModelConfig]:
        """타입별 모델 조회"""
        cls._ensure_loaded()
        return {name: config for name, config in cls.MODELS.items() if config.type == model_type}

    @classmethod
    def get_slm_models(cls) -> Dict[str, ModelConfig]:
        """SLM 모델 목록"""
        return cls.get_models_by_type("slm")

    @classmethod
    def get_llm_models(cls) -> Dict[str, ModelConfig]:
        """LLM 모델 목록"""
        return cls.get_models_by_type("llm")

    @classmethod
    def get_default_model(
        cls, provider: Optional[LLMProvider] = None, model_type: str = "llm"
    ) -> Optional[str]:
        """기본 모델 조회"""
        if provider:
            models = cls.get_models_by_provider(provider)
            for name, config in models.items():
                if config.type == model_type:
                    return name
        else:
            if model_type == "slm":
                return "phi3.5"
            elif model_type == "llm":
                # 사용 가능한 제공자에 따라 기본 모델 선택 (EnvConfig 사용)
                from beanllm.utils.config import EnvConfig

                if EnvConfig.ANTHROPIC_API_KEY:
                    return "claude-3-5-sonnet-20241022"
                elif EnvConfig.OPENAI_API_KEY:
                    return "gpt-4o-mini"
                elif EnvConfig.GEMINI_API_KEY:
                    return "gemini-1.5-flash"
                elif EnvConfig.DEEPSEEK_API_KEY:
                    return "deepseek-chat"
                elif EnvConfig.PERPLEXITY_API_KEY:
                    return "sonar"
                else:
                    return "qwen2.5:7b"
        return None
