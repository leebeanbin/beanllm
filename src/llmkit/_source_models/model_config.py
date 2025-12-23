"""
Model Configuration
모델 설정 및 관리 (모든 제공자 지원)
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
    """모델 설정 관리자"""

    # 모델 설정 정의 (모든 제공자 포함)
    MODELS: Dict[str, ModelConfig] = {
        # Ollama 모델
        "phi3.5": ModelConfig(
            name="phi3.5",
            display_name="Phi-3.5 (SLM)",
            provider=LLMProvider.OLLAMA,
            type="slm",
            max_tokens=2048,
            temperature=0.0,
            description="빠른 응답을 위한 Small Language Model",
            use_case="간단한 질문, 검색 제안, 자동완성",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "qwen2.5:7b": ModelConfig(
            name="qwen2.5:7b",
            display_name="Qwen2.5 7B (LLM)",
            provider=LLMProvider.OLLAMA,
            type="llm",
            max_tokens=4096,
            temperature=0.0,
            description="균형잡힌 성능의 Large Language Model",
            use_case="일반 대화, 설명, 분석",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "llama3.1:70b": ModelConfig(
            name="llama3.1:70b",
            display_name="Llama 3.1 70B (Large LLM)",
            provider=LLMProvider.OLLAMA,
            type="llm",
            max_tokens=8192,
            temperature=0.0,
            description="고성능 추론을 위한 Large Language Model",
            use_case="복잡한 분석, 전략 수립, 심층 추론",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "ax:3.1-lite": ModelConfig(
            name="ax:3.1-lite",
            display_name="A.X 3.1 Lite (Korean)",
            provider=LLMProvider.OLLAMA,
            type="llm",
            max_tokens=4096,
            temperature=0.0,
            description="한국어 특화 모델",
            use_case="한국어 금융 질문, 한국 시장 분석",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        # OpenAI 모델
        "gpt-4o-mini": ModelConfig(
            name="gpt-4o-mini",
            display_name="GPT-4o Mini",
            provider=LLMProvider.OPENAI,
            type="llm",
            max_tokens=16384,
            temperature=0.0,
            description="OpenAI의 빠르고 저렴한 모델",
            use_case="일반 대화, 빠른 응답",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "gpt-4o": ModelConfig(
            name="gpt-4o",
            display_name="GPT-4o",
            provider=LLMProvider.OPENAI,
            type="llm",
            max_tokens=128000,
            temperature=0.0,
            description="OpenAI의 최신 고성능 모델",
            use_case="복잡한 분석, 정확한 답변",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "gpt-4-turbo": ModelConfig(
            name="gpt-4-turbo",
            display_name="GPT-4 Turbo",
            provider=LLMProvider.OPENAI,
            type="llm",
            max_tokens=128000,
            temperature=0.0,
            description="OpenAI의 고성능 모델",
            use_case="복잡한 작업, 긴 컨텍스트",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        # GPT-5 시리즈 (max_completion_tokens 사용)
        "gpt-5-mini": ModelConfig(
            name="gpt-5-mini",
            display_name="GPT-5 Mini",
            provider=LLMProvider.OPENAI,
            type="llm",
            max_tokens=16384,
            temperature=0.0,
            description="OpenAI의 최신 경량 모델",
            use_case="일반 대화, 빠른 응답",
            supports_temperature=False,  # mini는 temperature 미지원 (기본값 1만 지원)
            supports_max_tokens=False,
            uses_max_completion_tokens=True,
        ),
        "gpt-5-nano": ModelConfig(
            name="gpt-5-nano",
            display_name="GPT-5 Nano",
            provider=LLMProvider.OPENAI,
            type="llm",
            max_tokens=16384,
            temperature=0.0,
            description="OpenAI의 최신 초경량 모델",
            use_case="초고속 응답, 간단한 작업",
            supports_temperature=False,  # nano는 temperature 미지원
            supports_max_tokens=False,
            uses_max_completion_tokens=True,
        ),
        "gpt-5": ModelConfig(
            name="gpt-5",
            display_name="GPT-5",
            provider=LLMProvider.OPENAI,
            type="llm",
            max_tokens=128000,
            temperature=0.0,
            description="OpenAI의 최신 고성능 모델",
            use_case="복잡한 분석, 정확한 답변",
            supports_temperature=True,
            supports_max_tokens=False,
            uses_max_completion_tokens=True,
        ),
        # GPT-4.1 시리즈 (max_completion_tokens 사용)
        "gpt-4.1-mini": ModelConfig(
            name="gpt-4.1-mini",
            display_name="GPT-4.1 Mini",
            provider=LLMProvider.OPENAI,
            type="llm",
            max_tokens=16384,
            temperature=0.0,
            description="OpenAI의 경량 모델",
            use_case="일반 대화, 빠른 응답",
            supports_temperature=False,  # mini는 temperature 미지원 (기본값 1만 지원)
            supports_max_tokens=False,
            uses_max_completion_tokens=True,
        ),
        "gpt-4.1-nano": ModelConfig(
            name="gpt-4.1-nano",
            display_name="GPT-4.1 Nano",
            provider=LLMProvider.OPENAI,
            type="llm",
            max_tokens=16384,
            temperature=0.0,
            description="OpenAI의 초경량 모델",
            use_case="초고속 응답, 간단한 작업",
            supports_temperature=False,  # nano는 temperature 미지원
            supports_max_tokens=False,
            uses_max_completion_tokens=True,
        ),
        "gpt-4.1": ModelConfig(
            name="gpt-4.1",
            display_name="GPT-4.1",
            provider=LLMProvider.OPENAI,
            type="llm",
            max_tokens=128000,
            temperature=0.0,
            description="OpenAI의 고성능 모델",
            use_case="복잡한 분석, 정확한 답변",
            supports_temperature=True,
            supports_max_tokens=False,
            uses_max_completion_tokens=True,
        ),
        # O3, O4 시리즈 (temperature 미지원)
        "o3-mini": ModelConfig(
            name="o3-mini",
            display_name="O3 Mini",
            provider=LLMProvider.OPENAI,
            type="llm",
            max_tokens=16384,
            temperature=0.0,
            description="OpenAI의 추론 모델 경량 버전",
            use_case="추론 작업, 수학, 과학",
            supports_temperature=False,  # o3는 temperature 미지원
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "o3": ModelConfig(
            name="o3",
            display_name="O3",
            provider=LLMProvider.OPENAI,
            type="llm",
            max_tokens=16384,
            temperature=0.0,
            description="OpenAI의 추론 모델",
            use_case="고급 추론 작업, 수학, 과학",
            supports_temperature=False,  # o3는 temperature 미지원
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "o4-mini": ModelConfig(
            name="o4-mini",
            display_name="O4 Mini",
            provider=LLMProvider.OPENAI,
            type="llm",
            max_tokens=16384,
            temperature=0.0,
            description="OpenAI의 최신 추론 모델 경량 버전",
            use_case="추론 작업, 수학, 과학",
            supports_temperature=False,  # o4는 temperature 미지원
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        # Anthropic (Claude) 모델
        "claude-3-5-sonnet-20241022": ModelConfig(
            name="claude-3-5-sonnet-20241022",
            display_name="Claude 3.5 Sonnet",
            provider=LLMProvider.ANTHROPIC,
            type="llm",
            max_tokens=8192,
            temperature=0.0,
            description="Anthropic의 최신 고성능 모델",
            use_case="복잡한 추론, 정확한 분석",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "claude-3-opus-20240229": ModelConfig(
            name="claude-3-opus-20240229",
            display_name="Claude 3 Opus",
            provider=LLMProvider.ANTHROPIC,
            type="llm",
            max_tokens=4096,
            temperature=0.0,
            description="Anthropic의 최고 성능 모델",
            use_case="최고 수준의 추론, 복잡한 작업",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "claude-3-haiku-20240307": ModelConfig(
            name="claude-3-haiku-20240307",
            display_name="Claude 3 Haiku",
            provider=LLMProvider.ANTHROPIC,
            type="llm",
            max_tokens=4096,
            temperature=0.0,
            description="Anthropic의 빠른 모델",
            use_case="빠른 응답, 간단한 작업",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        # Google (Gemini) 모델
        "gemini-1.5-pro": ModelConfig(
            name="gemini-1.5-pro",
            display_name="Gemini 1.5 Pro",
            provider=LLMProvider.GOOGLE,
            type="llm",
            max_tokens=8192,
            temperature=0.0,
            description="Google의 고성능 모델",
            use_case="복잡한 분석, 멀티모달",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "gemini-1.5-flash": ModelConfig(
            name="gemini-1.5-flash",
            display_name="Gemini 1.5 Flash",
            provider=LLMProvider.GOOGLE,
            type="llm",
            max_tokens=8192,
            temperature=0.0,
            description="Google의 빠른 모델",
            use_case="빠른 응답, 일반 작업",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
    }

    @classmethod
    def get_model_config(cls, model_name: str) -> Optional[ModelConfig]:
        """모델 설정 조회"""
        return cls.MODELS.get(model_name)

    @classmethod
    def get_models_by_provider(cls, provider: LLMProvider) -> Dict[str, ModelConfig]:
        """제공자별 모델 조회"""
        return {name: config for name, config in cls.MODELS.items() if config.provider == provider}

    @classmethod
    def get_models_by_type(cls, model_type: str) -> Dict[str, ModelConfig]:
        """타입별 모델 조회"""
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
                from ...utils.config import EnvConfig

                if EnvConfig.ANTHROPIC_API_KEY:
                    return "claude-3-5-sonnet-20241022"
                elif EnvConfig.OPENAI_API_KEY:
                    return "gpt-4o-mini"
                elif EnvConfig.GEMINI_API_KEY:
                    return "gemini-1.5-flash"
                else:
                    return "qwen2.5:7b"
        return None
