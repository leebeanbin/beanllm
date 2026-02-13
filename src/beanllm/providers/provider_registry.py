"""
Provider Registry - 중앙화된 Provider 감지 및 매핑

모든 Provider 관련 if-else 체인을 데이터 기반 Registry로 교체하여:
- 새 Provider 추가 시 한 곳만 수정
- 모델명 → Provider 감지 로직 통일
- Provider 이름 정규화 로직 통일
- 환경변수 → 가용성 확인 통일
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from beanllm.utils.config import EnvConfig

# ============================================================
# 1. Provider 모델 패턴 매핑 (모델명 → provider)
# ============================================================

# (패턴 키워드 리스트, provider 이름)
# 순서가 중요: 먼저 매칭되는 것이 우선
MODEL_PATTERN_REGISTRY: List[Tuple[Sequence[str], str]] = [
    (["gpt", "o1", "o3", "o4"], "openai"),
    (["claude"], "anthropic"),
    (["gemini"], "google"),
    (["deepseek"], "deepseek"),
    (["perplexity", "sonar"], "perplexity"),
]

# Ollama 전용 패턴 (콜론 포함 또는 로컬 모델명)
OLLAMA_PATTERNS: List[str] = ["qwen", "llama", "phi", "ax:"]


def detect_provider_from_model(model: str) -> str:
    """
    모델 이름으로부터 Provider를 감지합니다.

    Args:
        model: 모델 이름 (e.g., "gpt-4o", "claude-sonnet-4-20250514", "qwen2.5:0.5b")

    Returns:
        Provider 이름 (e.g., "openai", "anthropic", "ollama")
    """
    model_lower = model.lower()

    # Ollama 모델 패턴 확인 (콜론 포함 또는 등록된 패턴)
    if ":" in model or any(p in model_lower for p in OLLAMA_PATTERNS):
        return "ollama"

    # API Provider 패턴 매칭
    for patterns, provider_name in MODEL_PATTERN_REGISTRY:
        if any(p in model_lower for p in patterns):
            return provider_name

    # 기본값: Ollama (로컬 모델)
    return "ollama"


# ============================================================
# 2. Provider 이름 정규화 매핑
# ============================================================

# 다양한 Provider 이름을 표준 이름으로 정규화
# key: 이름에 포함될 수 있는 문자열, value: 정규화된 이름
PROVIDER_NAME_NORMALIZATION: Dict[str, str] = {
    "deepseek": "deepseek",
    "perplexity": "perplexity",
    "openai": "openai",
    "gemini": "google",
    "google": "google",
    "claude": "anthropic",
    "anthropic": "anthropic",
    "ollama": "ollama",
}

# ProviderFactory용 정규화 (ProviderFactory는 "claude", "gemini" 사용)
PROVIDER_FACTORY_NAME_MAP: Dict[str, str] = {
    "openai": "openai",
    "claude": "claude",
    "anthropic": "claude",
    "gemini": "gemini",
    "google": "gemini",
    "deepseek": "deepseek",
    "perplexity": "perplexity",
    "ollama": "ollama",
}


def normalize_provider_name(provider: str) -> str:
    """
    Provider 이름을 표준 이름으로 정규화합니다.

    Args:
        provider: Provider 이름 또는 클래스 이름 (e.g., "DeepSeekProvider", "openai")

    Returns:
        정규화된 Provider 이름 (e.g., "deepseek", "openai")
    """
    provider_lower = provider.lower()

    for keyword, normalized in PROVIDER_NAME_NORMALIZATION.items():
        if keyword in provider_lower:
            return normalized

    return provider_lower


# ============================================================
# 3. Provider 환경변수 매핑
# ============================================================

# (provider_name, env_key)
PROVIDER_ENV_MAP: Dict[str, Optional[str]] = {
    "openai": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
    "ollama": None,  # Ollama는 API 키가 필요 없음
}


def is_provider_env_available(provider_name: str) -> bool:
    """
    Provider의 환경변수가 설정되어 있는지 확인합니다.

    Args:
        provider_name: Provider 이름

    Returns:
        True if available, False otherwise
    """
    env_key = PROVIDER_ENV_MAP.get(provider_name)

    if env_key is None:
        # Ollama처럼 API 키가 필요 없는 Provider
        return True

    env_value = getattr(EnvConfig, env_key, None)
    return bool(env_value)
