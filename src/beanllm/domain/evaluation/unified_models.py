"""
Unified Evaluator - 데이터 모델 (EvalRecord, ImprovementSuggestion)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List


@dataclass
class EvalRecord:
    """평가 기록"""

    record_id: str
    query: str
    response: str
    contexts: List[str]

    # 자동 평가 결과
    auto_scores: Dict[str, float] = field(default_factory=dict)
    auto_avg_score: float = 0.0

    # Human 피드백
    human_ratings: List[float] = field(default_factory=list)
    human_avg_rating: float = 0.0
    human_feedback_count: int = 0
    human_comments: List[str] = field(default_factory=list)

    # 통합 점수
    unified_score: float = 0.0

    # 메타데이터
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImprovementSuggestion:
    """개선 제안"""

    category: str  # "chunking", "retrieval", "generation", "prompt"
    priority: str  # "high", "medium", "low"
    issue: str
    suggestion: str
    affected_queries: List[str] = field(default_factory=list)
    expected_improvement: float = 0.0
