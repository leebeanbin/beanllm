"""
LLM Provider Enum
지원하는 LLM 제공자 열거형
"""

from enum import Enum


class LLMProvider(str, Enum):
    """지원하는 LLM 제공자"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"  # Claude
    GOOGLE = "google"  # Gemini
    OLLAMA = "ollama"
    AUTO = "auto"  # 자동 선택
