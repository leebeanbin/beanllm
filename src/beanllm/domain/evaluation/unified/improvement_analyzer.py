"""
Improvement Analyzer - 개선 제안 분석 모듈
"""

from __future__ import annotations

from typing import List

from beanllm.domain.evaluation.unified_models import EvalRecord, ImprovementSuggestion


class ImprovementAnalyzer:
    """개선 제안 분석기"""

    @staticmethod
    def get_improvement_suggestions(
        records: List[EvalRecord],
    ) -> List[ImprovementSuggestion]:
        """
        피드백 기반 개선 제안 생성

        Args:
            records: 평가 레코드 리스트

        Returns:
            개선 제안 리스트
        """
        suggestions: List[ImprovementSuggestion] = []

        if not records:
            return suggestions

        # 1. 낮은 Faithfulness 감지
        low_faithfulness = [r for r in records if r.auto_scores.get("faithfulness", 1.0) < 0.5]
        if low_faithfulness:
            suggestions.append(
                ImprovementSuggestion(
                    category="retrieval",
                    priority="high",
                    issue=f"{len(low_faithfulness)}개 쿼리에서 환각 감지됨",
                    suggestion=(
                        "검색된 컨텍스트가 불충분합니다. "
                        "청크 크기를 늘리거나, top_k를 증가시키거나, "
                        "Reranker를 추가해보세요."
                    ),
                    affected_queries=[r.query for r in low_faithfulness[:5]],
                    expected_improvement=0.2,
                )
            )

        # 2. 낮은 Relevance 감지
        low_relevance = [r for r in records if r.auto_scores.get("relevance", 1.0) < 0.5]
        if low_relevance:
            suggestions.append(
                ImprovementSuggestion(
                    category="generation",
                    priority="high",
                    issue=f"{len(low_relevance)}개 쿼리에서 관련성 부족",
                    suggestion=(
                        "프롬프트를 개선하여 질문에 더 집중하도록 하세요. "
                        "또는 쿼리 확장(HyDE, MultiQuery)을 시도해보세요."
                    ),
                    affected_queries=[r.query for r in low_relevance[:5]],
                    expected_improvement=0.15,
                )
            )

        # 3. 낮은 Human 피드백
        low_human = [r for r in records if r.human_avg_rating < 0.4 and r.human_feedback_count > 0]
        if low_human:
            # 코멘트 분석
            all_comments = []
            for r in low_human:
                all_comments.extend(r.human_comments)

            suggestions.append(
                ImprovementSuggestion(
                    category="overall",
                    priority="high",
                    issue=f"{len(low_human)}개 쿼리에서 사용자 불만족",
                    suggestion=(
                        "사용자 피드백이 낮습니다. "
                        f"주요 코멘트: {all_comments[:3] if all_comments else '없음'}. "
                        "청킹 전략 변경과 프롬프트 개선을 고려하세요."
                    ),
                    affected_queries=[r.query for r in low_human[:5]],
                    expected_improvement=0.25,
                )
            )

        # 4. Context Precision 낮음
        low_ctx_precision = [
            r for r in records if r.auto_scores.get("context_precision", 1.0) < 0.5
        ]
        if low_ctx_precision:
            suggestions.append(
                ImprovementSuggestion(
                    category="chunking",
                    priority="medium",
                    issue=f"{len(low_ctx_precision)}개 쿼리에서 컨텍스트 정확도 낮음",
                    suggestion=(
                        "검색된 청크가 질문과 관련이 적습니다. "
                        "시맨틱 청킹을 사용하거나, 청크 크기를 줄여보세요. "
                        "하이브리드 검색(BM25 + Dense)도 효과적입니다."
                    ),
                    affected_queries=[r.query for r in low_ctx_precision[:5]],
                    expected_improvement=0.2,
                )
            )

        # 우선순위순 정렬
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda s: priority_order.get(s.priority, 2))

        return suggestions
