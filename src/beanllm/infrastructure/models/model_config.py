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
        # Ollama 최신 모델 (2025년)
        "llama3.3:70b": ModelConfig(
            name="llama3.3:70b",
            display_name="Llama 3.3 70B",
            provider=LLMProvider.OLLAMA,
            type="llm",
            max_tokens=8192,
            temperature=0.0,
            description="Meta의 Llama 3.3 70B 모델 (2025)",
            use_case="고성능 추론, 복잡한 분석",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "qwen3": ModelConfig(
            name="qwen3",
            display_name="Qwen 3",
            provider=LLMProvider.OLLAMA,
            type="llm",
            max_tokens=8192,
            temperature=0.0,
            description="Alibaba의 Qwen 3 모델 (2025)",
            use_case="다국어 지원, 코드 생성",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "phi4": ModelConfig(
            name="phi4",
            display_name="Phi-4",
            provider=LLMProvider.OLLAMA,
            type="slm",
            max_tokens=4096,
            temperature=0.0,
            description="Microsoft의 Phi-4 Small Language Model (2025)",
            use_case="빠른 응답, 경량 작업",
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
        # O1 시리즈 (2024-2025 출시)
        "o1": ModelConfig(
            name="o1",
            display_name="O1",
            provider=LLMProvider.OPENAI,
            type="llm",
            max_tokens=100000,
            temperature=1.0,
            description="OpenAI의 추론 모델 (2024)",
            use_case="복잡한 추론, 수학, 과학, 코딩",
            supports_temperature=False,  # o1은 temperature 고정
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "o1-mini": ModelConfig(
            name="o1-mini",
            display_name="O1 Mini",
            provider=LLMProvider.OPENAI,
            type="llm",
            max_tokens=65536,
            temperature=1.0,
            description="OpenAI의 빠른 추론 모델",
            use_case="코딩, 수학, 과학 (비용 효율적)",
            supports_temperature=False,  # o1-mini는 temperature 고정
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "o1-preview": ModelConfig(
            name="o1-preview",
            display_name="O1 Preview",
            provider=LLMProvider.OPENAI,
            type="llm",
            max_tokens=128000,
            temperature=1.0,
            description="OpenAI의 o1 프리뷰 모델",
            use_case="고급 추론, 복잡한 문제 해결",
            supports_temperature=False,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        # O3-mini (2025년 1월 출시)
        "o3-mini": ModelConfig(
            name="o3-mini",
            display_name="O3 Mini",
            provider=LLMProvider.OPENAI,
            type="llm",
            max_tokens=100000,
            temperature=1.0,
            description="OpenAI의 최신 추론 모델 (2025)",
            use_case="고급 추론, 수학, 과학, 코딩",
            supports_temperature=False,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        # O4-mini (2025년 4월 출시)
        "o4-mini": ModelConfig(
            name="o4-mini",
            display_name="O4 Mini",
            provider=LLMProvider.OPENAI,
            type="llm",
            max_tokens=100000,
            temperature=1.0,
            description="OpenAI의 o4 추론 모델 (2025.04)",
            use_case="고급 추론, 수학, 과학, 코딩",
            supports_temperature=False,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        # GPT-4.1 시리즈 (2025년 4월 출시)
        "gpt-4.1": ModelConfig(
            name="gpt-4.1",
            display_name="GPT-4.1",
            provider=LLMProvider.OPENAI,
            type="llm",
            max_tokens=1000000,  # 1M context
            temperature=0.0,
            description="OpenAI의 GPT-4.1 모델 (2025.04, 1M context)",
            use_case="복잡한 분석, 긴 컨텍스트 처리",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "gpt-4.1-mini": ModelConfig(
            name="gpt-4.1-mini",
            display_name="GPT-4.1 Mini",
            provider=LLMProvider.OPENAI,
            type="llm",
            max_tokens=500000,
            temperature=0.0,
            description="OpenAI의 GPT-4.1 Mini 모델 (2025.04)",
            use_case="빠른 응답, 비용 효율적",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "gpt-4.1-nano": ModelConfig(
            name="gpt-4.1-nano",
            display_name="GPT-4.1 Nano",
            provider=LLMProvider.OPENAI,
            type="llm",
            max_tokens=200000,
            temperature=0.0,
            description="OpenAI의 GPT-4.1 Nano 모델 (2025.04)",
            use_case="초고속 응답, 매우 경제적",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        # GPT-4.5 시리즈 (2025년 출시)
        "gpt-4.5": ModelConfig(
            name="gpt-4.5",
            display_name="GPT-4.5",
            provider=LLMProvider.OPENAI,
            type="llm",
            max_tokens=1000000,
            temperature=0.0,
            description="OpenAI의 GPT-4.5 모델 (2025)",
            use_case="최고 성능 분석, 복잡한 작업",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "gpt-4.5-mini": ModelConfig(
            name="gpt-4.5-mini",
            display_name="GPT-4.5 Mini",
            provider=LLMProvider.OPENAI,
            type="llm",
            max_tokens=500000,
            temperature=0.0,
            description="OpenAI의 GPT-4.5 Mini 모델 (2025)",
            use_case="빠른 응답, 균형잡힌 성능",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        # GPT-5 시리즈 (2025년 8월 출시)
        "gpt-5": ModelConfig(
            name="gpt-5",
            display_name="GPT-5",
            provider=LLMProvider.OPENAI,
            type="llm",
            max_tokens=200000,
            temperature=1.0,
            description="OpenAI의 GPT-5 추론 모델 (2025.08)",
            use_case="최고급 추론, 복잡한 문제 해결, 코딩",
            supports_temperature=False,  # Reasoning model
            supports_max_tokens=False,
            uses_max_completion_tokens=True,
        ),
        "gpt-5.1": ModelConfig(
            name="gpt-5.1",
            display_name="GPT-5.1",
            provider=LLMProvider.OPENAI,
            type="llm",
            max_tokens=200000,
            temperature=1.0,
            description="OpenAI의 GPT-5.1 추론 모델 (2025)",
            use_case="고급 추론, 수학, 과학, 전략 수립",
            supports_temperature=False,
            supports_max_tokens=False,
            uses_max_completion_tokens=True,
        ),
        "gpt-5.2": ModelConfig(
            name="gpt-5.2",
            display_name="GPT-5.2",
            provider=LLMProvider.OPENAI,
            type="llm",
            max_tokens=200000,
            temperature=1.0,
            description="OpenAI의 GPT-5.2 추론 모델 (2025)",
            use_case="최첨단 추론, 복잡한 분석",
            supports_temperature=False,
            supports_max_tokens=False,
            uses_max_completion_tokens=True,
        ),
        # Anthropic (Claude) 모델
        "claude-3-5-sonnet-20241022": ModelConfig(
            name="claude-3-5-sonnet-20241022",
            display_name="Claude 3.5 Sonnet",
            provider=LLMProvider.ANTHROPIC,
            type="llm",
            max_tokens=8192,
            temperature=0.0,
            description="Anthropic의 최신 고성능 모델 (2024)",
            use_case="복잡한 추론, 정확한 분석",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "claude-3-5-haiku-20241022": ModelConfig(
            name="claude-3-5-haiku-20241022",
            display_name="Claude 3.5 Haiku",
            provider=LLMProvider.ANTHROPIC,
            type="llm",
            max_tokens=8192,
            temperature=0.0,
            description="Anthropic의 최신 빠른 모델 (2024)",
            use_case="빠른 응답, 비용 효율적",
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
        "claude-3-sonnet-20240229": ModelConfig(
            name="claude-3-sonnet-20240229",
            display_name="Claude 3 Sonnet",
            provider=LLMProvider.ANTHROPIC,
            type="llm",
            max_tokens=4096,
            temperature=0.0,
            description="Anthropic의 균형잡힌 모델",
            use_case="일반 대화, 분석",
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
        # Claude 4 Series (2025년 출시)
        "claude-opus-4": ModelConfig(
            name="claude-opus-4",
            display_name="Claude Opus 4",
            provider=LLMProvider.ANTHROPIC,
            type="llm",
            max_tokens=16384,
            temperature=0.0,
            description="Anthropic의 Claude 4 Opus 모델 (2025)",
            use_case="최고 수준 추론, 복잡한 분석",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "claude-sonnet-4": ModelConfig(
            name="claude-sonnet-4",
            display_name="Claude Sonnet 4",
            provider=LLMProvider.ANTHROPIC,
            type="llm",
            max_tokens=16384,
            temperature=0.0,
            description="Anthropic의 Claude 4 Sonnet 모델 (2025)",
            use_case="균형잡힌 성능, 일반 작업",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "claude-haiku-4": ModelConfig(
            name="claude-haiku-4",
            display_name="Claude Haiku 4",
            provider=LLMProvider.ANTHROPIC,
            type="llm",
            max_tokens=16384,
            temperature=0.0,
            description="Anthropic의 Claude 4 Haiku 모델 (2025)",
            use_case="빠른 응답, 비용 효율적",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        # Claude 4.1 Series (2025년 출시)
        "claude-opus-4-1": ModelConfig(
            name="claude-opus-4-1",
            display_name="Claude Opus 4.1",
            provider=LLMProvider.ANTHROPIC,
            type="llm",
            max_tokens=32768,
            temperature=0.0,
            description="Anthropic의 Claude 4.1 Opus 모델 (2025)",
            use_case="최고급 추론, 장문 생성",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "claude-sonnet-4-1": ModelConfig(
            name="claude-sonnet-4-1",
            display_name="Claude Sonnet 4.1",
            provider=LLMProvider.ANTHROPIC,
            type="llm",
            max_tokens=32768,
            temperature=0.0,
            description="Anthropic의 Claude 4.1 Sonnet 모델 (2025)",
            use_case="고성능 분석, 복잡한 작업",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        # Claude 4.5 Series (2025년 출시)
        "claude-opus-4-5": ModelConfig(
            name="claude-opus-4-5",
            display_name="Claude Opus 4.5",
            provider=LLMProvider.ANTHROPIC,
            type="llm",
            max_tokens=64000,  # 64k output
            temperature=0.0,
            description="Anthropic의 Claude 4.5 Opus 모델 (2025, 200k context)",
            use_case="최첨단 추론, extended thinking, effort control",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "claude-sonnet-4-5": ModelConfig(
            name="claude-sonnet-4-5",
            display_name="Claude Sonnet 4.5",
            provider=LLMProvider.ANTHROPIC,
            type="llm",
            max_tokens=64000,
            temperature=0.0,
            description="Anthropic의 Claude 4.5 Sonnet 모델 (2025)",
            use_case="고성능 추론, extended reasoning",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "claude-haiku-4-5": ModelConfig(
            name="claude-haiku-4-5",
            display_name="Claude Haiku 4.5",
            provider=LLMProvider.ANTHROPIC,
            type="llm",
            max_tokens=64000,
            temperature=0.0,
            description="Anthropic의 Claude 4.5 Haiku 모델 (2025)",
            use_case="빠른 추론, 경제적",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        # Google (Gemini) 모델
        "gemini-2.0-flash-exp": ModelConfig(
            name="gemini-2.0-flash-exp",
            display_name="Gemini 2.0 Flash (Exp)",
            provider=LLMProvider.GOOGLE,
            type="llm",
            max_tokens=8192,
            temperature=0.0,
            description="Google의 최신 실험 모델 (2025)",
            use_case="최신 기능 테스트, 멀티모달",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
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
        # Gemini 2.0 Series (2025년 출시)
        "gemini-2.0-flash": ModelConfig(
            name="gemini-2.0-flash",
            display_name="Gemini 2.0 Flash",
            provider=LLMProvider.GOOGLE,
            type="llm",
            max_tokens=8192,
            temperature=0.0,
            description="Google의 Gemini 2.0 Flash 모델 (2025)",
            use_case="빠른 응답, 멀티모달",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "gemini-2.0-pro": ModelConfig(
            name="gemini-2.0-pro",
            display_name="Gemini 2.0 Pro",
            provider=LLMProvider.GOOGLE,
            type="llm",
            max_tokens=8192,
            temperature=0.0,
            description="Google의 Gemini 2.0 Pro 모델 (2025)",
            use_case="고성능 분석, 복잡한 작업",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "gemini-2.0-flash-lite": ModelConfig(
            name="gemini-2.0-flash-lite",
            display_name="Gemini 2.0 Flash Lite",
            provider=LLMProvider.GOOGLE,
            type="llm",
            max_tokens=8192,
            temperature=0.0,
            description="Google의 Gemini 2.0 Flash Lite 모델 (2025)",
            use_case="초고속 응답, 경제적",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        # Gemini 2.5 Series (2025년 출시)
        "gemini-2.5-pro": ModelConfig(
            name="gemini-2.5-pro",
            display_name="Gemini 2.5 Pro",
            provider=LLMProvider.GOOGLE,
            type="llm",
            max_tokens=65536,  # 65k max
            temperature=0.0,
            description="Google의 Gemini 2.5 Pro 모델 (2025)",
            use_case="최고 성능, 복잡한 추론",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "gemini-2.5-flash": ModelConfig(
            name="gemini-2.5-flash",
            display_name="Gemini 2.5 Flash",
            provider=LLMProvider.GOOGLE,
            type="llm",
            max_tokens=65536,
            temperature=0.0,
            description="Google의 Gemini 2.5 Flash 모델 (2025)",
            use_case="빠른 성능, 균형잡힌",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "gemini-2.5-flash-lite": ModelConfig(
            name="gemini-2.5-flash-lite",
            display_name="Gemini 2.5 Flash Lite",
            provider=LLMProvider.GOOGLE,
            type="llm",
            max_tokens=65536,
            temperature=0.0,
            description="Google의 Gemini 2.5 Flash Lite 모델 (2025)",
            use_case="초고속, 비용 효율적",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        # Gemini 3.0 Series (2025-2026년 출시)
        "gemini-3.0-pro": ModelConfig(
            name="gemini-3.0-pro",
            display_name="Gemini 3.0 Pro",
            provider=LLMProvider.GOOGLE,
            type="llm",
            max_tokens=65536,
            temperature=0.0,
            description="Google의 Gemini 3.0 Pro 모델 (2025)",
            use_case="최첨단 성능, 복잡한 추론",
            supports_temperature=True,  # 경고 있음
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "gemini-3.0-deep-think": ModelConfig(
            name="gemini-3.0-deep-think",
            display_name="Gemini 3.0 Deep Think",
            provider=LLMProvider.GOOGLE,
            type="llm",
            max_tokens=65536,
            temperature=0.0,
            description="Google의 Gemini 3.0 Deep Think 추론 모델 (2025)",
            use_case="심층 추론, 단계별 사고",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        # DeepSeek 모델
        "deepseek-chat": ModelConfig(
            name="deepseek-chat",
            display_name="DeepSeek Chat",
            provider=LLMProvider.DEEPSEEK,
            type="llm",
            max_tokens=8192,
            temperature=0.0,
            description="DeepSeek의 일반 대화 모델 (671B 파라미터, 37B 활성화 MoE)",
            use_case="일반 대화, 코드 생성, 분석",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "deepseek-reasoner": ModelConfig(
            name="deepseek-reasoner",
            display_name="DeepSeek Reasoner",
            provider=LLMProvider.DEEPSEEK,
            type="llm",
            max_tokens=8192,
            temperature=0.0,
            description="DeepSeek의 사고 모델 (단계별 추론)",
            use_case="복잡한 추론, 수학, 과학",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        # DeepSeek V3 Series (2025년 출시)
        "deepseek-v3-0324": ModelConfig(
            name="deepseek-v3-0324",
            display_name="DeepSeek V3 (2025.03.24)",
            provider=LLMProvider.DEEPSEEK,
            type="llm",
            max_tokens=8192,
            temperature=0.0,
            description="DeepSeek V3 모델 (2025.03.24)",
            use_case="일반 대화, 코드 생성",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "deepseek-v3-1": ModelConfig(
            name="deepseek-v3-1",
            display_name="DeepSeek V3.1",
            provider=LLMProvider.DEEPSEEK,
            type="llm",
            max_tokens=8192,
            temperature=0.0,
            description="DeepSeek V3.1 모델 (2025)",
            use_case="향상된 성능, 코드 생성, 분석",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "deepseek-v3-2": ModelConfig(
            name="deepseek-v3-2",
            display_name="DeepSeek V3.2",
            provider=LLMProvider.DEEPSEEK,
            type="llm",
            max_tokens=8192,
            temperature=0.0,
            description="DeepSeek V3.2 모델 (2025)",
            use_case="최신 성능, 복잡한 작업",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        # DeepSeek R1 Series (2025년 출시)
        "deepseek-r1": ModelConfig(
            name="deepseek-r1",
            display_name="DeepSeek R1",
            provider=LLMProvider.DEEPSEEK,
            type="llm",
            max_tokens=8192,
            temperature=0.0,
            description="DeepSeek R1 추론 모델 (2025)",
            use_case="복잡한 추론, 수학, 과학 (thinking mode)",
            supports_temperature=False,  # No temperature in thinking mode
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "deepseek-r1-0528": ModelConfig(
            name="deepseek-r1-0528",
            display_name="DeepSeek R1 (2025.05.28)",
            provider=LLMProvider.DEEPSEEK,
            type="llm",
            max_tokens=8192,
            temperature=0.0,
            description="DeepSeek R1 추론 모델 (2025.05.28)",
            use_case="최신 추론, 복잡한 문제 해결",
            supports_temperature=False,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        # Perplexity 모델
        "sonar": ModelConfig(
            name="sonar",
            display_name="Perplexity Sonar",
            provider=LLMProvider.PERPLEXITY,
            type="llm",
            max_tokens=4096,
            temperature=0.0,
            description="Perplexity의 실시간 웹 검색 + LLM 통합 모델 (Llama 3.3 70B 기반)",
            use_case="실시간 정보 검색, 최신 정보 질문",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "sonar-pro": ModelConfig(
            name="sonar-pro",
            display_name="Perplexity Sonar Pro",
            provider=LLMProvider.PERPLEXITY,
            type="llm",
            max_tokens=4096,
            temperature=0.0,
            description="Perplexity의 고성능 모델 (더 정확한 검색 및 답변)",
            use_case="고품질 검색, 상세한 인용이 필요한 작업",
            supports_temperature=True,
            supports_max_tokens=True,
            uses_max_completion_tokens=False,
        ),
        "sonar-reasoning-pro": ModelConfig(
            name="sonar-reasoning-pro",
            display_name="Perplexity Sonar Reasoning Pro",
            provider=LLMProvider.PERPLEXITY,
            type="llm",
            max_tokens=4096,
            temperature=0.0,
            description="Perplexity의 추론 모델 (사고 과정 포함)",
            use_case="복잡한 추론이 필요한 검색, 단계별 분석",
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
