"""
Shared type definitions for DTOs.

Replaces Dict[str, Any] with concrete types for type safety.
"""

from __future__ import annotations

from typing import TypedDict


class TokenUsage(TypedDict, total=False):
    """Token usage statistics from LLM providers."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class StepInfo(TypedDict, total=False):
    """Step information for chain/agent execution."""

    step_number: int
    name: str
    input: str
    output: str
    duration: float
    metadata: dict[str, object]


class StatisticalResult(TypedDict, total=False):
    """Statistical test results for A/B testing."""

    p_value: float
    significant: bool
    test_type: str
    effect_size: float


# Type aliases for common patterns
Metadata = dict[str, object]
ExtraParams = dict[str, object]
