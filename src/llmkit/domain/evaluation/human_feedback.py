"""
Human Feedback - 인간 피드백 수집 및 관리
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class FeedbackType(str, Enum):
    """피드백 타입"""

    RATING = "rating"  # 평점 (0.0 ~ 1.0)
    COMPARISON = "comparison"  # 비교 평가 (A vs B)
    CORRECTION = "correction"  # 수정 제안
    COMMENT = "comment"  # 자유 텍스트 코멘트


class ComparisonWinner(str, Enum):
    """비교 평가 승자"""

    A = "A"
    B = "B"
    TIE = "TIE"


@dataclass
class HumanFeedback:
    """인간 피드백 데이터"""

    feedback_id: str
    feedback_type: FeedbackType
    output: str  # 평가 대상 출력
    criteria: Optional[str] = None  # 평가 기준
    rating: Optional[float] = None  # 평점 (0.0 ~ 1.0)
    comment: Optional[str] = None  # 코멘트
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """검증"""
        if self.rating is not None:
            if not 0.0 <= self.rating <= 1.0:
                raise ValueError(f"Rating must be between 0.0 and 1.0, got {self.rating}")

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "feedback_id": self.feedback_id,
            "feedback_type": self.feedback_type.value,
            "output": self.output,
            "criteria": self.criteria,
            "rating": self.rating,
            "comment": self.comment,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class ComparisonFeedback(HumanFeedback):
    """
    비교 평가 피드백

    Note: dataclass 데코레이터 제거 (부모 클래스의 기본값 필드와 충돌 방지)
    """

    def __init__(
        self,
        feedback_id: str,
        output_a: str,
        output_b: str,
        winner: ComparisonWinner,
        criteria: Optional[str] = None,
        comment: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        # 비교 평가는 두 출력을 결합한 형태로 저장
        combined_output = f"Output A: {output_a}\n\nOutput B: {output_b}"
        super().__init__(
            feedback_id=feedback_id,
            feedback_type=FeedbackType.COMPARISON,
            output=combined_output,
            criteria=criteria,
            comment=comment,
            timestamp=timestamp or datetime.now(),
            metadata=metadata or {},
        )
        self.output_a = output_a
        self.output_b = output_b
        self.winner = winner

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "output_a": self.output_a,
                "output_b": self.output_b,
                "winner": self.winner.value,
            }
        )
        return base_dict


class HumanFeedbackCollector:
    """
    인간 피드백 수집기

    다양한 형태의 인간 피드백을 수집하고 관리
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Args:
            storage_path: 피드백 저장 경로 (선택적, 파일 기반 저장)
        """
        self.storage_path = storage_path
        self._feedbacks: List[HumanFeedback] = []
        self._feedback_counter = 0

    def collect_rating(
        self,
        output: str,
        rating: float,
        criteria: Optional[str] = None,
        comment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HumanFeedback:
        """
        평점 수집

        Args:
            output: 평가 대상 출력
            rating: 평점 (0.0 ~ 1.0)
            criteria: 평가 기준 (선택적)
            comment: 코멘트 (선택적)
            metadata: 추가 메타데이터 (선택적)

        Returns:
            HumanFeedback: 수집된 피드백
        """
        feedback_id = f"rating_{self._feedback_counter}"
        self._feedback_counter += 1

        feedback = HumanFeedback(
            feedback_id=feedback_id,
            feedback_type=FeedbackType.RATING,
            output=output,
            criteria=criteria,
            rating=rating,
            comment=comment,
            metadata=metadata or {},
        )

        self._feedbacks.append(feedback)
        self._save_if_needed()

        return feedback

    def collect_comparison(
        self,
        output_a: str,
        output_b: str,
        winner: ComparisonWinner,
        criteria: Optional[str] = None,
        comment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ComparisonFeedback:
        """
        비교 평가 수집

        Args:
            output_a: 첫 번째 출력
            output_b: 두 번째 출력
            winner: 승자 (A, B, 또는 TIE)
            criteria: 평가 기준 (선택적)
            comment: 코멘트 (선택적)
            metadata: 추가 메타데이터 (선택적)

        Returns:
            ComparisonFeedback: 수집된 비교 피드백
        """
        feedback_id = f"comparison_{self._feedback_counter}"
        self._feedback_counter += 1

        feedback = ComparisonFeedback(
            feedback_id=feedback_id,
            output_a=output_a,
            output_b=output_b,
            winner=winner,
            criteria=criteria,
            comment=comment,
            metadata=metadata or {},
        )

        self._feedbacks.append(feedback)
        self._save_if_needed()

        return feedback

    def collect_correction(
        self,
        output: str,
        corrected_output: str,
        comment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HumanFeedback:
        """
        수정 제안 수집

        Args:
            output: 원본 출력
            corrected_output: 수정된 출력
            comment: 수정 이유 (선택적)
            metadata: 추가 메타데이터 (선택적)

        Returns:
            HumanFeedback: 수집된 수정 피드백
        """
        feedback_id = f"correction_{self._feedback_counter}"
        self._feedback_counter += 1

        # 수정 제안은 코멘트에 포함
        full_comment = f"Original: {output}\nCorrected: {corrected_output}"
        if comment:
            full_comment += f"\nReason: {comment}"

        feedback = HumanFeedback(
            feedback_id=feedback_id,
            feedback_type=FeedbackType.CORRECTION,
            output=output,
            comment=full_comment,
            metadata=metadata or {},
        )

        self._feedbacks.append(feedback)
        self._save_if_needed()

        return feedback

    def collect_comment(
        self,
        output: str,
        comment: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HumanFeedback:
        """
        자유 텍스트 코멘트 수집

        Args:
            output: 평가 대상 출력
            comment: 코멘트
            metadata: 추가 메타데이터 (선택적)

        Returns:
            HumanFeedback: 수집된 코멘트 피드백
        """
        feedback_id = f"comment_{self._feedback_counter}"
        self._feedback_counter += 1

        feedback = HumanFeedback(
            feedback_id=feedback_id,
            feedback_type=FeedbackType.COMMENT,
            output=output,
            comment=comment,
            metadata=metadata or {},
        )

        self._feedbacks.append(feedback)
        self._save_if_needed()

        return feedback

    def get_feedback(self, feedback_id: str) -> Optional[HumanFeedback]:
        """피드백 조회"""
        for feedback in self._feedbacks:
            if feedback.feedback_id == feedback_id:
                return feedback
        return None

    def get_all_feedbacks(self) -> List[HumanFeedback]:
        """모든 피드백 조회"""
        return self._feedbacks.copy()

    def get_feedbacks_by_type(self, feedback_type: FeedbackType) -> List[HumanFeedback]:
        """타입별 피드백 조회"""
        return [f for f in self._feedbacks if f.feedback_type == feedback_type]

    def clear(self):
        """모든 피드백 삭제"""
        self._feedbacks.clear()
        self._feedback_counter = 0

    def _save_if_needed(self):
        """필요시 저장 (파일 기반 저장 구현 예정)"""
        # TODO: 파일 기반 저장 구현
        pass
