"""
ParameterTuner - RAG 파라미터 실시간 튜닝
SOLID 원칙:
- SRP: 파라미터 튜닝만 담당
- OCP: 새로운 파라미터 추가 가능
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from beanllm.utils.logging import get_logger

if TYPE_CHECKING:
    from beanllm.domain.vector_stores import BaseVectorStore

logger = get_logger(__name__)


class ParameterTuner:
    """
    RAG 파라미터 튜너

    책임:
    - 파라미터 실시간 조정 및 테스트
    - 파라미터별 성능 비교
    - 최적 파라미터 추천

    Tunable Parameters:
    - top_k: 검색 결과 수
    - score_threshold: 최소 유사도 점수
    - mmr_lambda: MMR diversity 파라미터 (0~1)
    - chunk_size: 청크 크기 (문서 분할 시)
    - chunk_overlap: 청크 간 overlap
    """

    def __init__(
        self, vector_store: "BaseVectorStore", baseline_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Args:
            vector_store: 튜닝할 VectorStore
            baseline_params: 기준 파라미터 (비교 기준)
        """
        self.vector_store = vector_store
        self.baseline_params = baseline_params or {
            "top_k": 4,
            "score_threshold": 0.0,
            "mmr_lambda": 0.5,
        }

    def tune_top_k(
        self, query: str, k_values: List[int]
    ) -> Dict[str, Any]:
        """
        top_k 파라미터 튜닝

        Args:
            query: 테스트 쿼리
            k_values: 테스트할 k 값 목록

        Returns:
            Dict: k별 결과
        """
        logger.info(f"Tuning top_k for query: '{query}' with values {k_values}")

        results = {}

        for k in k_values:
            try:
                search_results = self.vector_store.similarity_search(query, k=k)
                results[f"k={k}"] = {
                    "num_results": len(search_results),
                    "avg_score": sum(r.score for r in search_results) / len(search_results)
                    if search_results
                    else 0.0,
                    "min_score": min(r.score for r in search_results)
                    if search_results
                    else 0.0,
                    "max_score": max(r.score for r in search_results)
                    if search_results
                    else 0.0,
                }
            except Exception as e:
                logger.error(f"Error tuning top_k={k}: {e}")
                results[f"k={k}"] = {"error": str(e)}

        return results

    def tune_threshold(
        self, query: str, thresholds: List[float], k: int = 10
    ) -> Dict[str, Any]:
        """
        score_threshold 파라미터 튜닝

        Args:
            query: 테스트 쿼리
            thresholds: 테스트할 threshold 값 목록
            k: 초기 검색 결과 수

        Returns:
            Dict: threshold별 결과
        """
        logger.info(
            f"Tuning score_threshold for query: '{query}' with values {thresholds}"
        )

        # Get initial search results
        try:
            all_results = self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Error in initial search: {e}")
            return {"error": str(e)}

        results = {}

        for threshold in thresholds:
            # Filter results by threshold
            filtered = [r for r in all_results if r.score >= threshold]

            results[f"threshold={threshold}"] = {
                "num_results": len(filtered),
                "avg_score": sum(r.score for r in filtered) / len(filtered)
                if filtered
                else 0.0,
                "filtered_out": len(all_results) - len(filtered),
            }

        return results

    def tune_mmr_lambda(
        self, query: str, lambda_values: List[float], k: int = 4
    ) -> Dict[str, Any]:
        """
        MMR lambda 파라미터 튜닝

        Args:
            query: 테스트 쿼리
            lambda_values: 테스트할 lambda 값 목록 (0~1)
            k: 검색 결과 수

        Returns:
            Dict: lambda별 결과

        Note:
            lambda = 0: 완전한 diversity (유사도 무시)
            lambda = 1: 완전한 relevance (diversity 무시)
            lambda = 0.5: 균형 (기본값)
        """
        logger.info(f"Tuning MMR lambda for query: '{query}' with values {lambda_values}")

        # Check if MMR is supported
        if not hasattr(self.vector_store, "max_marginal_relevance_search"):
            return {"error": "MMR not supported by this VectorStore"}

        results = {}

        for lambda_val in lambda_values:
            try:
                # Note: Actual MMR implementation may vary
                # This is a simplified version
                mmr_results = self.vector_store.max_marginal_relevance_search(
                    query, k=k, fetch_k=k * 2, lambda_mult=lambda_val
                )

                results[f"lambda={lambda_val}"] = {
                    "num_results": len(mmr_results),
                    "avg_score": sum(r.score for r in mmr_results) / len(mmr_results)
                    if mmr_results
                    else 0.0,
                }
            except Exception as e:
                logger.error(f"Error tuning lambda={lambda_val}: {e}")
                results[f"lambda={lambda_val}"] = {"error": str(e)}

        return results

    def compare_with_baseline(
        self, query: str, new_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        새 파라미터를 baseline과 비교

        Args:
            query: 테스트 쿼리
            new_params: 새 파라미터

        Returns:
            Dict: 비교 결과
        """
        logger.info(f"Comparing params: baseline={self.baseline_params}, new={new_params}")

        # Baseline results
        try:
            baseline_results = self.vector_store.similarity_search(
                query, k=self.baseline_params.get("top_k", 4)
            )
            baseline_score = (
                sum(r.score for r in baseline_results) / len(baseline_results)
                if baseline_results
                else 0.0
            )
        except Exception as e:
            logger.error(f"Error in baseline search: {e}")
            baseline_score = 0.0

        # New params results
        try:
            new_results = self.vector_store.similarity_search(
                query, k=new_params.get("top_k", 4)
            )
            new_score = (
                sum(r.score for r in new_results) / len(new_results)
                if new_results
                else 0.0
            )
        except Exception as e:
            logger.error(f"Error in new params search: {e}")
            new_score = 0.0

        # Compare
        improvement = ((new_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0.0

        return {
            "baseline": {
                "params": self.baseline_params,
                "avg_score": baseline_score,
            },
            "new": {
                "params": new_params,
                "avg_score": new_score,
            },
            "improvement_pct": improvement,
            "recommendation": "Use new params" if improvement > 5 else "Keep baseline",
        }

    def auto_tune(
        self, test_queries: List[str], param_ranges: Optional[Dict[str, List[Any]]] = None
    ) -> Dict[str, Any]:
        """
        자동 파라미터 튜닝 (Grid search)

        Args:
            test_queries: 테스트 쿼리 목록
            param_ranges: 파라미터 범위
                예: {"top_k": [4, 6, 8], "score_threshold": [0.0, 0.3, 0.5]}

        Returns:
            Dict: 최적 파라미터 및 결과
        """
        param_ranges = param_ranges or {
            "top_k": [4, 6, 8, 10],
            "score_threshold": [0.0, 0.3, 0.5],
        }

        logger.info(
            f"Auto-tuning with {len(test_queries)} queries and ranges {param_ranges}"
        )

        # TODO: Implement full grid search
        # For now, simplified version

        best_params = self.baseline_params.copy()
        best_score = 0.0

        # Test each parameter independently (not full grid)
        for param_name, param_values in param_ranges.items():
            for param_value in param_values:
                test_params = self.baseline_params.copy()
                test_params[param_name] = param_value

                # Test on all queries
                scores = []
                for query in test_queries:
                    try:
                        results = self.vector_store.similarity_search(
                            query, k=test_params.get("top_k", 4)
                        )
                        avg_score = (
                            sum(r.score for r in results) / len(results)
                            if results
                            else 0.0
                        )
                        scores.append(avg_score)
                    except Exception:
                        continue

                mean_score = sum(scores) / len(scores) if scores else 0.0

                if mean_score > best_score:
                    best_score = mean_score
                    best_params[param_name] = param_value

        return {
            "best_params": best_params,
            "best_score": best_score,
            "baseline_params": self.baseline_params,
            "improvement_pct": (
                (best_score - baseline_score) / baseline_score * 100
                if baseline_score > 0
                else 0.0
            ),
        }
