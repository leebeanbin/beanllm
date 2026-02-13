"""
Human Feedback - Human 피드백 관리 모듈
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from beanllm.domain.evaluation.unified_models import EvalRecord
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class HumanFeedbackManager:
    """Human 피드백 관리자"""

    def __init__(self, human_weight: float = 0.5):
        """
        Args:
            human_weight: Human 피드백 가중치
        """
        self.human_weight = human_weight

    def collect_feedback(
        self,
        record: EvalRecord,
        rating: float,
        feedback_type: str = "overall",
        comment: Optional[str] = None,
    ) -> None:
        """
        Human 피드백 수집

        Args:
            record: 평가 레코드
            rating: 평점 (0.0-1.0)
            feedback_type: 피드백 유형 ("overall", "relevance", "faithfulness", ...)
            comment: 추가 코멘트
        """
        # 피드백 추가
        record.human_ratings.append(rating)
        record.human_feedback_count += 1
        record.human_avg_rating = sum(record.human_ratings) / len(record.human_ratings)

        if comment:
            record.human_comments.append(comment)

        logger.info(
            f"Human feedback collected: query='{record.query[:30]}...', "
            f"rating={rating:.2f}, type={feedback_type}"
        )

    def collect_comparison_feedback(
        self,
        record: EvalRecord,
        response_a: str,
        response_b: str,
        winner: str,  # "A", "B", "TIE"
    ) -> None:
        """
        비교 피드백 수집 (A/B)

        Args:
            record: 평가 레코드
            response_a: 응답 A
            response_b: 응답 B
            winner: 선택된 응답 ("A", "B", "TIE")
        """
        # Winner를 rating으로 변환
        if winner == "A":
            self.collect_feedback(record, 1.0, "comparison", "Winner: A")
        elif winner == "B":
            self.collect_feedback(record, 0.0, "comparison", "Winner: B")
        else:
            self.collect_feedback(record, 0.5, "comparison", "TIE")
