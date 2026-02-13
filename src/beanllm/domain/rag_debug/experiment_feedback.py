"""
Chunking Experimenter - Feedback/analysis logic.

Extracted from chunking_experimenter.py for single responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

try:
    from beanllm.utils.logging import get_logger
except ImportError:
    import logging

    def get_logger(name: str) -> logging.Logger:  # type: ignore[no-redef]
        return logging.getLogger(name)


logger = get_logger(__name__)

__all__ = ["ChunkFeedback", "add_feedback", "get_feedback_summary", "improve_from_feedback"]


@dataclass
class ChunkFeedback:
    """Single chunk feedback record."""

    query: str
    chunk_id: str
    rating: float
    feedback_type: str
    comment: Optional[str] = None
    strategy_name: Optional[str] = None


def add_feedback(
    feedbacks: List[ChunkFeedback],
    current_chunks: Dict[str, List[str]],
    query: str,
    chunk_id: str,
    rating: float,
    feedback_type: str = "relevance",
    comment: Optional[str] = None,
) -> None:
    """
    Add chunk feedback and append to list.

    Args:
        feedbacks: List to append to (mutated).
        current_chunks: strategy_name -> chunks (for optional validation).
        query: Query text.
        chunk_id: Chunk identifier.
        rating: Rating value.
        feedback_type: Type of feedback.
        comment: Optional comment.
    """
    strategy_name = None
    for name, chunks in current_chunks.items():
        if chunk_id in chunks or any(chunk_id in c for c in chunks):
            strategy_name = name
            break
    feedbacks.append(
        ChunkFeedback(
            query=query,
            chunk_id=chunk_id,
            rating=rating,
            feedback_type=feedback_type,
            comment=comment,
            strategy_name=strategy_name,
        )
    )
    logger.debug(f"Feedback added: query={query[:30]}..., chunk_id={chunk_id}, rating={rating}")


def get_feedback_summary(feedbacks: List[ChunkFeedback]) -> Dict[str, Any]:
    """
    Summarize feedback list.

    Returns:
        Dict with total, avg_rating, by_type, etc.
    """
    if not feedbacks:
        return {"total": 0, "avg_rating": 0.0, "by_type": {}}
    total = len(feedbacks)
    avg_rating = sum(f.rating for f in feedbacks) / total
    by_type: Dict[str, List[float]] = {}
    for f in feedbacks:
        by_type.setdefault(f.feedback_type, []).append(f.rating)
    return {
        "total": total,
        "avg_rating": avg_rating,
        "by_type": {k: sum(v) / len(v) for k, v in by_type.items()},
    }


def improve_from_feedback(
    feedbacks: List[ChunkFeedback],
    find_best_strategy_fn: Callable[[], Optional[Dict[str, Any]]],
    min_rating_threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    Derive improvement suggestions from feedback.

    Args:
        feedbacks: List of ChunkFeedback.
        find_best_strategy_fn: Callable that returns best strategy dict or None.
        min_rating_threshold: Minimum rating to consider positive.

    Returns:
        Dict with recommended_configs, suggestions, low_rated_chunks, etc.
    """
    low_rated = [f for f in feedbacks if f.rating < min_rating_threshold]
    suggestions: List[str] = []
    if low_rated:
        suggestions.append(
            f"Consider increasing chunk overlap or splitting differently: "
            f"{len(low_rated)} low-rated chunks (rating < {min_rating_threshold})."
        )
    best = find_best_strategy_fn()
    recommended_configs = [best["config"]] if best and best.get("config") else []
    return {
        "recommended_configs": recommended_configs,
        "suggestions": suggestions,
        "low_rated_count": len(low_rated),
        "total_feedbacks": len(feedbacks),
    }
