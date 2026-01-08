"""
SimilarityTester - 유사도 검색 테스트 및 비교
SOLID 원칙:
- SRP: 유사도 검색 테스트만 담당
- OCP: 새로운 검색 전략 추가 가능
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from beanllm.utils.logger import get_logger

if TYPE_CHECKING:
    from beanllm.domain.vector_stores import BaseVectorStore

logger = get_logger(__name__)


class SimilarityTester:
    """
    유사도 검색 테스터

    책임:
    - 쿼리 시뮬레이션
    - 검색 전략 비교 (Similarity vs MMR vs Hybrid)
    - 거리 메트릭 분석
    - 검색 품질 평가
    """

    def __init__(self, vector_store: "BaseVectorStore") -> None:
        """
        Args:
            vector_store: 테스트할 VectorStore
        """
        self.vector_store = vector_store

    def test_query(
        self, query: str, k: int = 4, strategies: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        단일 쿼리 테스트 (여러 검색 전략)

        Args:
            query: 테스트 쿼리
            k: 반환할 결과 수
            strategies: 테스트할 전략 목록 (None이면 모두 테스트)

        Returns:
            Dict: 전략별 검색 결과
                - similarity: 기본 유사도 검색
                - mmr: Maximal Marginal Relevance
                - hybrid: 하이브리드 검색 (if available)
        """
        strategies = strategies or ["similarity", "mmr", "hybrid"]
        logger.info(f"Testing query: '{query}' with strategies {strategies}")

        results = {}

        # 1. Similarity search
        if "similarity" in strategies:
            try:
                similarity_results = self.vector_store.similarity_search(query, k=k)
                results["similarity"] = {
                    "results": [
                        {
                            "content": r.document.page_content[:100]
                            if hasattr(r.document, "page_content")
                            else str(r.document)[:100],
                            "score": r.score,
                            "metadata": r.metadata,
                        }
                        for r in similarity_results
                    ],
                    "num_results": len(similarity_results),
                }
            except Exception as e:
                logger.error(f"Similarity search failed: {e}")
                results["similarity"] = {"error": str(e)}

        # 2. MMR search
        if "mmr" in strategies:
            try:
                # Check if vector_store supports MMR
                if hasattr(self.vector_store, "max_marginal_relevance_search"):
                    mmr_results = self.vector_store.max_marginal_relevance_search(
                        query, k=k
                    )
                    results["mmr"] = {
                        "results": [
                            {
                                "content": r.document.page_content[:100]
                                if hasattr(r.document, "page_content")
                                else str(r.document)[:100],
                                "score": r.score,
                                "metadata": r.metadata,
                            }
                            for r in mmr_results
                        ],
                        "num_results": len(mmr_results),
                    }
                else:
                    results["mmr"] = {"error": "MMR not supported by this VectorStore"}
            except Exception as e:
                logger.error(f"MMR search failed: {e}")
                results["mmr"] = {"error": str(e)}

        # 3. Hybrid search
        if "hybrid" in strategies:
            try:
                # Check if vector_store supports hybrid search
                if hasattr(self.vector_store, "hybrid_search"):
                    hybrid_results = self.vector_store.hybrid_search(query, k=k)
                    results["hybrid"] = {
                        "results": [
                            {
                                "content": r.document.page_content[:100]
                                if hasattr(r.document, "page_content")
                                else str(r.document)[:100],
                                "score": r.score,
                                "metadata": r.metadata,
                            }
                            for r in hybrid_results
                        ],
                        "num_results": len(hybrid_results),
                    }
                else:
                    results["hybrid"] = {
                        "error": "Hybrid search not supported by this VectorStore"
                    }
            except Exception as e:
                logger.error(f"Hybrid search failed: {e}")
                results["hybrid"] = {"error": str(e)}

        logger.info(f"Query test completed: {len(results)} strategies tested")
        return results

    def batch_test(
        self, queries: List[str], k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        배치 쿼리 테스트

        Args:
            queries: 테스트 쿼리 목록
            k: 반환할 결과 수

        Returns:
            List[Dict]: 쿼리별 결과
        """
        logger.info(f"Running batch test for {len(queries)} queries")

        results = []
        for query in queries:
            result = self.test_query(query, k=k)
            results.append({"query": query, "results": result})

        return results

    def compare_strategies(
        self, query: str, k: int = 4
    ) -> Dict[str, Any]:
        """
        검색 전략 비교 분석

        Args:
            query: 테스트 쿼리
            k: 반환할 결과 수

        Returns:
            Dict: 비교 분석 결과
                - strategy_results: 전략별 결과
                - overlap_analysis: 전략 간 결과 중복 분석
                - recommendations: 권장사항
        """
        logger.info(f"Comparing search strategies for query: '{query}'")

        # Get results from all strategies
        strategy_results = self.test_query(query, k=k)

        # Analyze overlap between strategies
        overlap_analysis = self._analyze_overlap(strategy_results)

        # Generate recommendations
        recommendations = self._generate_strategy_recommendations(
            strategy_results, overlap_analysis
        )

        return {
            "query": query,
            "strategy_results": strategy_results,
            "overlap_analysis": overlap_analysis,
            "recommendations": recommendations,
        }

    def _analyze_overlap(
        self, strategy_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        전략 간 결과 중복 분석

        Args:
            strategy_results: 전략별 결과

        Returns:
            Dict: 중복 분석 결과
        """
        # Extract document IDs or content hashes
        strategy_docs = {}
        for strategy, result in strategy_results.items():
            if "error" not in result:
                docs = result.get("results", [])
                # Use first 100 chars as identifier
                strategy_docs[strategy] = set(d["content"] for d in docs)

        # Compute pairwise overlaps
        overlaps = {}
        strategies = list(strategy_docs.keys())

        for i in range(len(strategies)):
            for j in range(i + 1, len(strategies)):
                strategy_a = strategies[i]
                strategy_b = strategies[j]

                docs_a = strategy_docs[strategy_a]
                docs_b = strategy_docs[strategy_b]

                intersection = len(docs_a & docs_b)
                union = len(docs_a | docs_b)

                jaccard = intersection / union if union > 0 else 0.0

                overlaps[f"{strategy_a}_vs_{strategy_b}"] = {
                    "intersection": intersection,
                    "jaccard_similarity": jaccard,
                }

        return overlaps

    def _generate_strategy_recommendations(
        self, strategy_results: Dict[str, Any], overlap_analysis: Dict[str, Any]
    ) -> List[str]:
        """
        검색 전략 권장사항 생성

        Args:
            strategy_results: 전략별 결과
            overlap_analysis: 중복 분석

        Returns:
            List[str]: 권장사항
        """
        recommendations = []

        # Check which strategies are available
        available_strategies = [
            s for s, r in strategy_results.items() if "error" not in r
        ]

        if len(available_strategies) == 0:
            recommendations.append("⚠️  No search strategies available")
            return recommendations

        # Analyze overlaps
        for overlap_key, overlap_data in overlap_analysis.items():
            jaccard = overlap_data["jaccard_similarity"]

            if jaccard < 0.3:
                recommendations.append(
                    f"💡 {overlap_key}: Low overlap ({jaccard:.2%}). "
                    "Strategies return very different results - consider using hybrid."
                )
            elif jaccard > 0.8:
                recommendations.append(
                    f"ℹ️  {overlap_key}: High overlap ({jaccard:.2%}). "
                    "Strategies return similar results."
                )

        # MMR recommendation
        if "mmr" in available_strategies:
            recommendations.append(
                "💡 MMR (Maximal Marginal Relevance) reduces redundancy. "
                "Use when diversity is important."
            )

        # Hybrid recommendation
        if "hybrid" in available_strategies:
            recommendations.append(
                "💡 Hybrid search combines vector + keyword search. "
                "Best for queries with specific terms."
            )

        return recommendations
