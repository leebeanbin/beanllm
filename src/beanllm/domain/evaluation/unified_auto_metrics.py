"""
Unified Evaluator - 자동 평가 메트릭 계산 (AutoMetricsMixin)
"""

from __future__ import annotations

import math
from typing import Any, List


class AutoMetricsMixin:
    """
    자동 평가 메트릭 계산 Mixin.

    _compute_metric 및 개별 _compute_* 메서드 제공.
    사용처에서 llm_judge, embedding_function 속성이 필요합니다.
    """

    llm_judge: Any  # Callable[[str, str, List[str]], float] | None
    embedding_function: Any  # Callable[[str], List[float]] | None

    def _compute_metric(
        self,
        metric: str,
        query: str,
        response: str,
        contexts: List[str],
    ) -> float:
        """개별 메트릭 계산"""
        context_text = " ".join(contexts)

        if metric == "faithfulness":
            return self._compute_faithfulness(response, context_text)

        elif metric == "relevance":
            return self._compute_relevance(query, response)

        elif metric == "context_precision":
            return self._compute_context_precision(query, contexts)

        elif metric == "context_recall":
            return self._compute_context_recall(query, response, contexts)

        elif metric == "coherence":
            return self._compute_coherence(response)

        elif metric == "completeness":
            return self._compute_completeness(query, response)

        return 0.5  # 기본값

    def _compute_faithfulness(self, response: str, context: str) -> float:
        """Faithfulness 계산 (환각 감지)"""
        if self.llm_judge:
            prompt = f"""
            Context: {context[:2000]}
            Response: {response}

            Is the response faithful to the context? (0.0-1.0)
            Only return a number.
            """
            try:
                return float(self.llm_judge(prompt, "", []))
            except (ValueError, TypeError):
                pass

        # 간단한 단어 겹침 기반
        response_words = set(response.lower().split())
        context_words = set(context.lower().split())
        overlap = len(response_words & context_words)
        return min(1.0, overlap / len(response_words)) if response_words else 0.0

    def _compute_relevance(self, query: str, response: str) -> float:
        """Relevance 계산"""
        if self.embedding_function:
            query_emb = self.embedding_function(query)
            response_emb = self.embedding_function(response)

            dot = sum(a * b for a, b in zip(query_emb, response_emb))
            mag1 = math.sqrt(sum(a * a for a in query_emb))
            mag2 = math.sqrt(sum(b * b for b in response_emb))

            if mag1 > 0 and mag2 > 0:
                return float((dot / (mag1 * mag2) + 1) / 2)  # 0-1 범위로 정규화

        # 단어 겹침 기반
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words & response_words)
        return overlap / len(query_words) if query_words else 0.0

    def _compute_context_precision(self, query: str, contexts: List[str]) -> float:
        """Context Precision 계산"""
        if not contexts:
            return 0.0

        query_words = set(query.lower().split())
        relevant_count = 0

        for ctx in contexts:
            ctx_words = set(ctx.lower().split())
            overlap = len(query_words & ctx_words)
            if overlap / len(query_words) > 0.2:  # 20% 이상 겹치면 관련
                relevant_count += 1

        return relevant_count / len(contexts)

    def _compute_context_recall(self, query: str, response: str, contexts: List[str]) -> float:
        """Context Recall 계산"""
        if not contexts:
            return 0.0

        # 응답에서 사용된 컨텍스트 비율
        response_words = set(response.lower().split())
        used_contexts = 0

        for ctx in contexts:
            ctx_words = set(ctx.lower().split())
            overlap = len(response_words & ctx_words)
            if overlap > 5:  # 5개 이상 단어 겹치면 사용된 것으로 간주
                used_contexts += 1

        return used_contexts / len(contexts)

    def _compute_coherence(self, response: str) -> float:
        """Coherence 계산 (문장 연결성)"""
        sentences = response.split(".")
        if len(sentences) < 2:
            return 1.0  # 단일 문장은 일관성 있음

        # 간단한 휴리스틱: 문장 길이 일관성
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if not lengths:
            return 0.5

        avg_len = sum(lengths) / len(lengths)
        variance = sum((length - avg_len) ** 2 for length in lengths) / len(lengths)

        # 분산이 작을수록 일관성 높음
        return max(0.0, 1.0 - variance / 100)

    def _compute_completeness(self, query: str, response: str) -> float:
        """Completeness 계산"""
        # 질문 유형 감지 및 응답 완전성 평가
        query_lower = query.lower()

        # WH-질문 감지
        wh_words = ["what", "why", "how", "when", "where", "who", "which"]
        is_wh_question = any(w in query_lower for w in wh_words)

        # 응답 길이 기반 완전성
        response_words = len(response.split())

        if is_wh_question:
            # WH-질문은 최소 10단어 이상 답변 기대
            return min(1.0, response_words / 20)
        else:
            # 일반 질문은 최소 5단어
            return min(1.0, response_words / 10)
