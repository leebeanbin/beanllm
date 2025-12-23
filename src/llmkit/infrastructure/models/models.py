"""
Model Definitions
실제 insightstock-ai-service의 ModelConfigManager.MODELS 기반
"""

from typing import Dict, Optional

MODELS = {
    "phi3.5": {
        "name": "phi3.5",
        "display_name": "Phi-3.5 (SLM)",
        "provider": "ollama",
        "type": "slm",
        "max_tokens": 2048,
        "temperature": 0.0,
        "description": "빠른 응답을 위한 Small Language Model",
        "use_case": "간단한 질문, 검색 제안, 자동완성",
        "supports_temperature": True,
        "supports_max_tokens": True,
        "uses_max_completion_tokens": False,
    },
    "qwen2.5:7b": {
        "name": "qwen2.5:7b",
        "display_name": "Qwen2.5 7B (LLM)",
        "provider": "ollama",
        "type": "llm",
        "max_tokens": 4096,
        "temperature": 0.0,
        "description": "균형잡힌 성능의 Large Language Model",
        "use_case": "일반 대화, 설명, 분석",
        "supports_temperature": True,
        "supports_max_tokens": True,
        "uses_max_completion_tokens": False,
    },
    "llama3.1:70b": {
        "name": "llama3.1:70b",
        "display_name": "Llama 3.1 70B (Large LLM)",
        "provider": "ollama",
        "type": "llm",
        "max_tokens": 8192,
        "temperature": 0.0,
        "description": "고성능 추론을 위한 Large Language Model",
        "use_case": "복잡한 분석, 전략 수립, 심층 추론",
        "supports_temperature": True,
        "supports_max_tokens": True,
        "uses_max_completion_tokens": False,
    },
    "ax:3.1-lite": {
        "name": "ax:3.1-lite",
        "display_name": "A.X 3.1 Lite (Korean)",
        "provider": "ollama",
        "type": "llm",
        "max_tokens": 4096,
        "temperature": 0.0,
        "description": "한국어 특화 모델",
        "use_case": "한국어 금융 질문, 한국 시장 분석",
        "supports_temperature": True,
        "supports_max_tokens": True,
        "uses_max_completion_tokens": False,
    },
    "gpt-4o-mini": {
        "name": "gpt-4o-mini",
        "display_name": "GPT-4o Mini",
        "provider": "openai",
        "type": "llm",
        "max_tokens": 16384,
        "temperature": 0.0,
        "description": "OpenAI의 빠르고 저렴한 모델",
        "use_case": "일반 대화, 빠른 응답",
        "supports_temperature": True,
        "supports_max_tokens": True,
        "uses_max_completion_tokens": False,
    },
    "gpt-4o": {
        "name": "gpt-4o",
        "display_name": "GPT-4o",
        "provider": "openai",
        "type": "llm",
        "max_tokens": 128000,
        "temperature": 0.0,
        "description": "OpenAI의 최신 고성능 모델",
        "use_case": "복잡한 분석, 정확한 답변",
        "supports_temperature": True,
        "supports_max_tokens": True,
        "uses_max_completion_tokens": False,
    },
    "gpt-4-turbo": {
        "name": "gpt-4-turbo",
        "display_name": "GPT-4 Turbo",
        "provider": "openai",
        "type": "llm",
        "max_tokens": 128000,
        "temperature": 0.0,
        "description": "OpenAI의 고성능 모델",
        "use_case": "복잡한 작업, 긴 컨텍스트",
        "supports_temperature": True,
        "supports_max_tokens": True,
        "uses_max_completion_tokens": False,
    },
    "gpt-5-mini": {
        "name": "gpt-5-mini",
        "display_name": "GPT-5 Mini",
        "provider": "openai",
        "type": "llm",
        "max_tokens": 16384,
        "temperature": 0.0,
        "description": "OpenAI의 최신 경량 모델",
        "use_case": "일반 대화, 빠른 응답",
        "supports_temperature": False,
        "supports_max_tokens": False,
        "uses_max_completion_tokens": True,
    },
    "gpt-5-nano": {
        "name": "gpt-5-nano",
        "display_name": "GPT-5 Nano",
        "provider": "openai",
        "type": "llm",
        "max_tokens": 16384,
        "temperature": 0.0,
        "description": "OpenAI의 최신 초경량 모델",
        "use_case": "초고속 응답, 간단한 작업",
        "supports_temperature": False,
        "supports_max_tokens": False,
        "uses_max_completion_tokens": True,
    },
    "gpt-5": {
        "name": "gpt-5",
        "display_name": "GPT-5",
        "provider": "openai",
        "type": "llm",
        "max_tokens": 128000,
        "temperature": 0.0,
        "description": "OpenAI의 최신 고성능 모델",
        "use_case": "복잡한 분석, 정확한 답변",
        "supports_temperature": True,
        "supports_max_tokens": False,
        "uses_max_completion_tokens": True,
    },
    "gpt-4.1-mini": {
        "name": "gpt-4.1-mini",
        "display_name": "GPT-4.1 Mini",
        "provider": "openai",
        "type": "llm",
        "max_tokens": 16384,
        "temperature": 0.0,
        "description": "OpenAI의 경량 모델",
        "use_case": "일반 대화, 빠른 응답",
        "supports_temperature": False,
        "supports_max_tokens": False,
        "uses_max_completion_tokens": True,
    },
    "gpt-4.1-nano": {
        "name": "gpt-4.1-nano",
        "display_name": "GPT-4.1 Nano",
        "provider": "openai",
        "type": "llm",
        "max_tokens": 16384,
        "temperature": 0.0,
        "description": "OpenAI의 초경량 모델",
        "use_case": "초고속 응답, 간단한 작업",
        "supports_temperature": False,
        "supports_max_tokens": False,
        "uses_max_completion_tokens": True,
    },
    "gpt-4.1": {
        "name": "gpt-4.1",
        "display_name": "GPT-4.1",
        "provider": "openai",
        "type": "llm",
        "max_tokens": 128000,
        "temperature": 0.0,
        "description": "OpenAI의 고성능 모델",
        "use_case": "복잡한 분석, 정확한 답변",
        "supports_temperature": True,
        "supports_max_tokens": False,
        "uses_max_completion_tokens": True,
    },
    "o3-mini": {
        "name": "o3-mini",
        "display_name": "O3 Mini",
        "provider": "openai",
        "type": "llm",
        "max_tokens": 16384,
        "temperature": 0.0,
        "description": "OpenAI의 추론 모델 경량 버전",
        "use_case": "추론 작업, 수학, 과학",
        "supports_temperature": False,
        "supports_max_tokens": True,
        "uses_max_completion_tokens": False,
    },
    "o3": {
        "name": "o3",
        "display_name": "O3",
        "provider": "openai",
        "type": "llm",
        "max_tokens": 16384,
        "temperature": 0.0,
        "description": "OpenAI의 추론 모델",
        "use_case": "고급 추론 작업, 수학, 과학",
        "supports_temperature": False,
        "supports_max_tokens": True,
        "uses_max_completion_tokens": False,
    },
    "o4-mini": {
        "name": "o4-mini",
        "display_name": "O4 Mini",
        "provider": "openai",
        "type": "llm",
        "max_tokens": 16384,
        "temperature": 0.0,
        "description": "OpenAI의 최신 추론 모델 경량 버전",
        "use_case": "추론 작업, 수학, 과학",
        "supports_temperature": False,
        "supports_max_tokens": True,
        "uses_max_completion_tokens": False,
    },
    "claude-3-5-sonnet-20241022": {
        "name": "claude-3-5-sonnet-20241022",
        "display_name": "Claude 3.5 Sonnet",
        "provider": "anthropic",
        "type": "llm",
        "max_tokens": 8192,
        "temperature": 0.0,
        "description": "Anthropic의 최신 고성능 모델",
        "use_case": "복잡한 추론, 정확한 분석",
        "supports_temperature": True,
        "supports_max_tokens": True,
        "uses_max_completion_tokens": False,
    },
    "claude-3-opus-20240229": {
        "name": "claude-3-opus-20240229",
        "display_name": "Claude 3 Opus",
        "provider": "anthropic",
        "type": "llm",
        "max_tokens": 4096,
        "temperature": 0.0,
        "description": "Anthropic의 최고 성능 모델",
        "use_case": "최고 수준의 추론, 복잡한 작업",
        "supports_temperature": True,
        "supports_max_tokens": True,
        "uses_max_completion_tokens": False,
    },
    "claude-3-haiku-20240307": {
        "name": "claude-3-haiku-20240307",
        "display_name": "Claude 3 Haiku",
        "provider": "anthropic",
        "type": "llm",
        "max_tokens": 4096,
        "temperature": 0.0,
        "description": "Anthropic의 빠른 모델",
        "use_case": "빠른 응답, 간단한 작업",
        "supports_temperature": True,
        "supports_max_tokens": True,
        "uses_max_completion_tokens": False,
    },
    "gemini-1.5-pro": {
        "name": "gemini-1.5-pro",
        "display_name": "Gemini 1.5 Pro",
        "provider": "google",
        "type": "llm",
        "max_tokens": 8192,
        "temperature": 0.0,
        "description": "Google의 고성능 모델",
        "use_case": "복잡한 분석, 멀티모달",
        "supports_temperature": True,
        "supports_max_tokens": True,
        "uses_max_completion_tokens": False,
    },
    "gemini-1.5-flash": {
        "name": "gemini-1.5-flash",
        "display_name": "Gemini 1.5 Flash",
        "provider": "google",
        "type": "llm",
        "max_tokens": 8192,
        "temperature": 0.0,
        "description": "Google의 빠른 모델",
        "use_case": "빠른 응답, 일반 작업",
        "supports_temperature": True,
        "supports_max_tokens": True,
        "uses_max_completion_tokens": False,
    },
}


def get_all_models() -> Dict[str, Dict]:
    """모든 모델 정보 조회"""
    return MODELS.copy()


def get_models_by_provider(provider: str) -> Dict[str, Dict]:
    """제공자별 모델 조회"""
    provider_map = {
        "openai": "openai",
        "anthropic": "anthropic",
        "claude": "anthropic",
        "google": "google",
        "gemini": "google",
        "ollama": "ollama",
    }
    normalized = provider_map.get(provider.lower(), provider.lower())
    return {k: v for k, v in MODELS.items() if v["provider"] == normalized}


def get_models_by_type(model_type: str) -> Dict[str, Dict]:
    """타입별 모델 조회"""
    return {k: v for k, v in MODELS.items() if v["type"] == model_type}


def get_default_model(provider: Optional[str] = None, model_type: str = "llm") -> Optional[str]:
    """기본 모델 조회"""
    from ...utils.config import Config

    if provider:
        models = get_models_by_provider(provider)
        for name, config in models.items():
            if config["type"] == model_type:
                return name
    else:
        if model_type == "slm":
            return "phi3.5"
        elif model_type == "llm":
            if Config.ANTHROPIC_API_KEY:
                return "claude-3-5-sonnet-20241022"
            elif Config.OPENAI_API_KEY:
                return "gpt-4o-mini"
            elif Config.GEMINI_API_KEY:
                return "gemini-1.5-flash"
            else:
                return "qwen2.5:7b"
    return None
