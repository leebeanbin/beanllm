"""
Configuration
환경변수 기반 설정 관리
"""

import os
from pathlib import Path
from typing import Optional

# dotenv 선택적 로드
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    # dotenv가 없어도 작동하도록
    pass


class Config:
    """환경변수 설정"""

    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    @classmethod
    def get_active_providers(cls) -> list[str]:
        """활성화된 제공자 목록"""
        providers = []
        if cls.OPENAI_API_KEY:
            providers.append("openai")
        if cls.ANTHROPIC_API_KEY:
            providers.append("anthropic")
        if cls.GEMINI_API_KEY:
            providers.append("google")
        providers.append("ollama")  # 항상 가능
        return providers
