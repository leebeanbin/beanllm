"""
Token Counting & Cost Estimation

tiktoken 기반 정확한 토큰 계산 및 비용 추정

Mathematical Foundations:
=======================

1. Token Counting:
   tokens(text) = |tokenizer.encode(text)|

   where tokenizer is BPE (Byte-Pair Encoding)

2. Cost Estimation:
   cost = (input_tokens × input_price + output_tokens × output_price) / 1M

3. Context Window Management:
   available_tokens = model_limit - (system_tokens + user_tokens + reserved_tokens)

References:
----------
- OpenAI Tokenizer: https://github.com/openai/tiktoken
- Token Pricing: https://openai.com/pricing

Author: LLMKit Team
"""

import warnings
from typing import Dict, List, Optional

from beanllm.utils.token_pricing import (
    CostEstimate,
    ModelContextWindow,
    ModelPricing,
    get_context_window,
)

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None


# ============================================================================
# Token Counter
# ============================================================================


class TokenCounter:
    """
    Token 계산기

    tiktoken 기반 정확한 토큰 계산
    """

    # 모델별 인코딩
    MODEL_ENCODINGS = {
        # GPT-4o, GPT-4, GPT-3.5 Turbo
        "gpt-4o": "o200k_base",
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-embedding-3-large": "cl100k_base",
        "text-embedding-3-small": "cl100k_base",
        "text-embedding-ada-002": "cl100k_base",
        # Claude (approximation using cl100k_base)
        "claude": "cl100k_base",
        # Gemini (approximation)
        "gemini": "cl100k_base",
    }

    def __init__(self, model: str = "gpt-4o"):
        """
        Args:
            model: 모델 이름
        """
        self.model = model
        self._encoding = None

        if not TIKTOKEN_AVAILABLE:
            warnings.warn(
                "tiktoken not installed. Token counts will be approximate. "
                "Install with: pip install tiktoken"
            )

    def _get_encoding(self):
        """인코딩 가져오기 (lazy loading)"""
        if self._encoding is not None:
            return self._encoding

        if not TIKTOKEN_AVAILABLE:
            return None

        # 모델별 인코딩 결정
        encoding_name = None

        for model_prefix, enc_name in self.MODEL_ENCODINGS.items():
            if self.model.startswith(model_prefix):
                encoding_name = enc_name
                break

        # 기본값
        if encoding_name is None:
            if "gpt-4" in self.model or "gpt-3.5" in self.model:
                encoding_name = "cl100k_base"
            else:
                encoding_name = "cl100k_base"  # Safe default

        try:
            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            # Fallback to model-specific encoding
            try:
                self._encoding = tiktoken.encoding_for_model(self.model)
            except Exception:
                self._encoding = tiktoken.get_encoding("cl100k_base")

        return self._encoding

    def count_tokens(self, text: str) -> int:
        """
        텍스트의 토큰 수 계산

        Args:
            text: 입력 텍스트

        Returns:
            토큰 수
        """
        encoding = self._get_encoding()

        if encoding is None:
            # Approximation: ~4 characters per token
            return len(text) // 4

        return len(encoding.encode(text))

    def count_tokens_from_messages(self, messages: List[Dict[str, str]]) -> int:
        """
        채팅 메시지의 토큰 수 계산

        Args:
            messages: 메시지 리스트 [{"role": "user", "content": "..."}]

        Returns:
            총 토큰 수
        """
        encoding = self._get_encoding()

        if encoding is None:
            # Approximation
            total = 0
            for message in messages:
                total += len(message.get("content", "")) // 4
                total += 4  # role, name, etc overhead
            return total

        # GPT-4o / GPT-4 / GPT-3.5 토큰 계산
        # 참조: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

        tokens_per_message = 3  # Every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = 1  # If there's a name, the role is omitted

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(str(value)))
                if key == "name":
                    num_tokens += tokens_per_name

        num_tokens += 3  # Every reply is primed with <|start|>assistant<|message|>

        return num_tokens

    def estimate_tokens(self, text: str) -> int:
        """
        토큰 수 추정 (빠른 근사치)

        Args:
            text: 입력 텍스트

        Returns:
            추정 토큰 수
        """
        # 간단한 휴리스틱: 4 characters ≈ 1 token
        return len(text) // 4

    def get_available_tokens(self, messages: List[Dict[str, str]], reserved: int = 0) -> int:
        """
        사용 가능한 토큰 수 계산

        Args:
            messages: 현재 메시지
            reserved: 응답을 위해 예약할 토큰 수

        Returns:
            사용 가능한 토큰 수
        """
        context_window = ModelContextWindow.get_context_window(self.model)
        used_tokens = self.count_tokens_from_messages(messages)

        available = context_window - used_tokens - reserved

        return max(0, available)


# ============================================================================
# Cost Estimator
# ============================================================================


class CostEstimator:
    """비용 추정기"""

    def __init__(self, model: str = "gpt-4o"):
        """
        Args:
            model: 모델 이름
        """
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
# Convenience Functions
# ============================================================================


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    간편한 토큰 계산 함수

    Args:
        text: 입력 텍스트
        model: 모델 이름

    Returns:
        토큰 수

    Example:
        >>> tokens = count_tokens("Hello, world!", model="gpt-4o")
        >>> print(tokens)
        4
    """
    counter = TokenCounter(model)
    return counter.count_tokens(text)


def count_message_tokens(messages: List[Dict[str, str]], model: str = "gpt-4o") -> int:
    """
    메시지의 토큰 수 계산

    Args:
        messages: 메시지 리스트
        model: 모델 이름

    Returns:
        총 토큰 수

    Example:
        >>> messages = [
        ...     {"role": "user", "content": "Hello!"},
        ...     {"role": "assistant", "content": "Hi there!"}
        ... ]
        >>> tokens = count_message_tokens(messages, model="gpt-4o")
    """
    counter = TokenCounter(model)
    return counter.count_tokens_from_messages(messages)


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


# ============================================================================
# Backward Compatibility: Re-export from token_pricing
# ============================================================================

__all__ = [
    "CostEstimate",
    "CostEstimator",
    "ModelContextWindow",
    "ModelPricing",
    "TokenCounter",
    "count_message_tokens",
    "count_tokens",
    "estimate_cost",
    "get_cheapest_model",
    "get_context_window",
]
