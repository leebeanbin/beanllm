"""
Token Pricing & Cost Estimation

모델별 가격 데이터, 컨텍스트 윈도우, 비용 추정 유틸리티.

- ModelPricing: per-1M-token 가격 데이터
- ModelContextWindow: 모델별 컨텍스트 윈도우 크기
- CostEstimate / CostEstimator: 비용 추정
- estimate_cost, get_cheapest_model, get_context_window: 편의 함수

References:
- OpenAI Tokenizer: https://github.com/openai/tiktoken
- Token Pricing: https://openai.com/pricing
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

from beanllm.utils.constants import CLAUDE_DEFAULT_MAX_TOKENS

# ============================================================================
# Part 1: Token Pricing Database
# ============================================================================


class ModelPricing:
    """
    모델별 가격 정보 (per 1M tokens)

    Prices as of December 2024
    Update regularly from provider websites
    """

    # OpenAI Pricing (per 1M tokens)
    OPENAI = {
        # GPT-4o series
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.150, "output": 0.600},
        "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
        "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00},
        "gpt-4o-2024-05-13": {"input": 5.00, "output": 15.00},
        "gpt-4o-mini-2024-07-18": {"input": 0.150, "output": 0.600},
        # O-series (Reasoning models)
        "o1": {"input": 15.00, "output": 60.00},
        "o1-mini": {"input": 3.00, "output": 12.00},
        "o1-preview": {"input": 15.00, "output": 60.00},
        "o1-preview-2024-09-12": {"input": 15.00, "output": 60.00},
        "o1-mini-2024-09-12": {"input": 3.00, "output": 12.00},
        # GPT-4 Turbo
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4-turbo-2024-04-09": {"input": 10.00, "output": 30.00},
        "gpt-4-turbo-preview": {"input": 10.00, "output": 30.00},
        "gpt-4-0125-preview": {"input": 10.00, "output": 30.00},
        "gpt-4-1106-preview": {"input": 10.00, "output": 30.00},
        # GPT-4
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-4-0613": {"input": 30.00, "output": 60.00},
        "gpt-4-32k": {"input": 60.00, "output": 120.00},
        "gpt-4-32k-0613": {"input": 60.00, "output": 120.00},
        # GPT-3.5 Turbo
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "gpt-3.5-turbo-0125": {"input": 0.50, "output": 1.50},
        "gpt-3.5-turbo-1106": {"input": 1.00, "output": 2.00},
        "gpt-3.5-turbo-16k": {"input": 3.00, "output": 4.00},
        # Embeddings
        "text-embedding-3-large": {"input": 0.13, "output": 0.0},
        "text-embedding-3-small": {"input": 0.02, "output": 0.0},
        "text-embedding-ada-002": {"input": 0.10, "output": 0.0},
    }

    # Anthropic Claude Pricing
    ANTHROPIC = {
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        "claude-2.1": {"input": 8.00, "output": 24.00},
        "claude-2.0": {"input": 8.00, "output": 24.00},
        "claude-instant-1.2": {"input": 0.80, "output": 2.40},
    }

    # Google Gemini Pricing
    GOOGLE = {
        "gemini-2.0-flash-exp": {"input": 0.0, "output": 0.0},  # Free preview
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-1.5-pro-002": {"input": 1.25, "output": 5.00},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-1.5-flash-002": {"input": 0.075, "output": 0.30},
        "gemini-1.5-flash-8b": {"input": 0.0375, "output": 0.15},
        "gemini-1.0-pro": {"input": 0.50, "output": 1.50},
    }

    # Ollama (Local - Free)
    OLLAMA = {
        "llama3.2": {"input": 0.0, "output": 0.0},
        "llama3.1": {"input": 0.0, "output": 0.0},
        "llama3": {"input": 0.0, "output": 0.0},
        "phi4": {"input": 0.0, "output": 0.0},
        "qwen2.5": {"input": 0.0, "output": 0.0},
        "mistral": {"input": 0.0, "output": 0.0},
        "mixtral": {"input": 0.0, "output": 0.0},
    }

    # 통합
    ALL_MODELS = {**OPENAI, **ANTHROPIC, **GOOGLE, **OLLAMA}

    @classmethod
    def get_pricing(cls, model: str) -> Optional[Dict[str, float]]:
        """모델의 가격 정보 조회"""
        # 정확한 매치
        if model in cls.ALL_MODELS:
            return cls.ALL_MODELS[model]

        # 부분 매치 (예: "gpt-4o-mini-2024-07-18" → "gpt-4o-mini")
        for model_key in cls.ALL_MODELS:
            if model.startswith(model_key):
                return cls.ALL_MODELS[model_key]

        return None


# ============================================================================
# Part 2: Model Context Windows
# ============================================================================


class ModelContextWindow:
    """모델별 컨텍스트 윈도우 크기"""

    CONTEXT_WINDOWS = {
        # OpenAI
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "o1": 200000,
        "o1-mini": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-3.5-turbo": 16385,
        "gpt-3.5-turbo-16k": 16385,
        # Anthropic
        "claude-3-5-sonnet-20241022": 200000,
        "claude-3-5-haiku-20241022": 200000,
        "claude-3-opus-20240229": 200000,
        "claude-3-sonnet-20240229": 200000,
        "claude-3-haiku-20240307": 200000,
        "claude-2.1": 200000,
        "claude-2.0": 100000,
        "claude-instant-1.2": 100000,
        # Google
        "gemini-2.0-flash-exp": 1000000,
        "gemini-1.5-pro": 2000000,
        "gemini-1.5-flash": 1000000,
        "gemini-1.5-flash-8b": 1000000,
        "gemini-1.0-pro": 32768,
        # Ollama (depends on hardware, typical values)
        "llama3.2": 128000,
        "llama3.1": 128000,
        "llama3": 8192,
        "phi4": 16384,
        "qwen2.5": 32768,
        "mistral": 32768,
        "mixtral": 32768,
    }

    @classmethod
    def get_context_window(cls, model: str) -> int:
        """모델의 컨텍스트 윈도우 크기 조회"""
        # 정확한 매치
        if model in cls.CONTEXT_WINDOWS:
            return cls.CONTEXT_WINDOWS[model]

        # 부분 매치
        for model_key, window in cls.CONTEXT_WINDOWS.items():
            if model.startswith(model_key):
                return window

        # 기본값 (안전하게 작게)
        return CLAUDE_DEFAULT_MAX_TOKENS


# ============================================================================
# Part 3: Cost Estimate & Estimator
# ============================================================================


@dataclass
class CostEstimate:
    """비용 추정 결과"""

    input_tokens: int
    output_tokens: int
    input_cost: float  # USD
    output_cost: float  # USD
    total_cost: float  # USD
    model: str
    currency: str = "USD"

    def __str__(self) -> str:
        return (
            f"Cost Estimate for {self.model}:\n"
            f"  Input: {self.input_tokens:,} tokens → ${self.input_cost:.6f}\n"
            f"  Output: {self.output_tokens:,} tokens → ${self.output_cost:.6f}\n"
            f"  Total: ${self.total_cost:.6f}"
        )


class CostEstimator:
    """비용 추정기"""

    def __init__(self, model: str = "gpt-4o") -> None:
        """
        Args:
            model: 모델 이름
        """
        from beanllm.utils.token_counter import TokenCounter

        self.model = model
        self.counter = TokenCounter(model)

    def estimate_cost(
        self,
        input_text: Optional[str] = None,
        output_text: Optional[str] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> CostEstimate:
        """
        비용 추정

        Args:
            input_text: 입력 텍스트
            output_text: 출력 텍스트
            input_tokens: 입력 토큰 수 (직접 제공)
            output_tokens: 출력 토큰 수 (직접 제공)
            messages: 메시지 리스트 (채팅)

        Returns:
            CostEstimate
        """
        # 토큰 수 계산
        if input_tokens is None:
            if messages is not None:
                input_tokens = self.counter.count_tokens_from_messages(messages)
            elif input_text is not None:
                input_tokens = self.counter.count_tokens(input_text)
            else:
                input_tokens = 0

        if output_tokens is None:
            if output_text is not None:
                output_tokens = self.counter.count_tokens(output_text)
            else:
                output_tokens = 0

        # 가격 정보 조회
        pricing = ModelPricing.get_pricing(self.model)

        if pricing is None:
            warnings.warn(f"Pricing not found for model: {self.model}. Using default.")
            pricing = {"input": 0.0, "output": 0.0}

        # 비용 계산 (per 1M tokens)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        return CostEstimate(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            model=self.model,
        )

    def compare_models(
        self, models: List[str], input_text: str, output_tokens: int = 1000
    ) -> List[CostEstimate]:
        """
        여러 모델의 비용 비교

        Args:
            models: 모델 리스트
            input_text: 입력 텍스트
            output_tokens: 예상 출력 토큰 수

        Returns:
            모델별 비용 추정 리스트
        """
        estimates = []

        for model in models:
            estimator = CostEstimator(model)
            estimate = estimator.estimate_cost(input_text=input_text, output_tokens=output_tokens)
            estimates.append(estimate)

        # 비용 순으로 정렬
        estimates.sort(key=lambda x: x.total_cost)

        return estimates


# ============================================================================
# Convenience Functions (pricing/cost only)
# ============================================================================


def estimate_cost(input_text: str, output_text: str = "", model: str = "gpt-4o") -> CostEstimate:
    """
    간편한 비용 추정 함수

    Args:
        input_text: 입력 텍스트
        output_text: 출력 텍스트
        model: 모델 이름

    Returns:
        CostEstimate

    Example:
        >>> cost = estimate_cost("Hello", "Hi there!", model="gpt-4o")
        >>> print(f"Total cost: ${cost.total_cost:.6f}")
    """
    estimator = CostEstimator(model)
    return estimator.estimate_cost(input_text=input_text, output_text=output_text)


def get_cheapest_model(
    input_text: str, output_tokens: int = 1000, models: Optional[List[str]] = None
) -> str:
    """
    가장 저렴한 모델 찾기

    Args:
        input_text: 입력 텍스트
        output_tokens: 예상 출력 토큰 수
        models: 비교할 모델 리스트 (None이면 주요 모델)

    Returns:
        가장 저렴한 모델 이름

    Example:
        >>> cheapest = get_cheapest_model("Long text...", output_tokens=1000)
        >>> print(f"Cheapest model: {cheapest}")
    """
    if models is None:
        models = ["gpt-4o-mini", "gpt-3.5-turbo", "claude-3-5-haiku-20241022", "gemini-1.5-flash"]

    estimator = CostEstimator(models[0])
    estimates = estimator.compare_models(models, input_text, output_tokens)

    return estimates[0].model if estimates else models[0]


def get_context_window(model: str) -> int:
    """
    모델의 컨텍스트 윈도우 크기 조회

    Args:
        model: 모델 이름

    Returns:
        컨텍스트 윈도우 크기 (토큰)

    Example:
        >>> window = get_context_window("gpt-4o")
        >>> print(f"Context window: {window:,} tokens")
    """
    return ModelContextWindow.get_context_window(model)
