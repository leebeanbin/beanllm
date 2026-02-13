"""
Token Counting

tiktoken 기반 정확한 토큰 계산.

Mathematical Foundations:
========================
tokens(text) = |tokenizer.encode(text)|
where tokenizer is BPE (Byte-Pair Encoding)

References:
----------
- OpenAI Tokenizer: https://github.com/openai/tiktoken

Pricing/cost symbols (ModelPricing, CostEstimate, etc.) are re-exported from
beanllm.utils.token_pricing for backward compatibility.
"""

from __future__ import annotations

import warnings
from types import ModuleType
from typing import Dict, List, Optional

tiktoken: Optional[ModuleType] = None
try:
    import tiktoken as _tiktoken

    tiktoken = _tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Re-export pricing symbols for backward compatibility (e.g. from beanllm.utils.token_counter import ModelPricing)
from beanllm.utils.token_pricing import (
    CostEstimate,
    CostEstimator,
    ModelContextWindow,
    ModelPricing,
    estimate_cost,
    get_cheapest_model,
    get_context_window,
)

__all__ = [
    "TokenCounter",
    "count_tokens",
    "count_message_tokens",
    # Re-exports for backward compatibility
    "ModelPricing",
    "ModelContextWindow",
    "CostEstimate",
    "CostEstimator",
    "estimate_cost",
    "get_cheapest_model",
    "get_context_window",
]


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

    def __init__(self, model: str = "gpt-4o") -> None:
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
# Convenience Functions (token counting only)
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
