"""
Recommender - 최적화 권장사항 생성기
SOLID 원칙:
- SRP: 권장사항 생성만 담당
- OCP: 새로운 규칙 추가 가능
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class RecommendationCategory(Enum):
    """권장사항 카테고리"""

    PERFORMANCE = "performance"  # 성능 개선
    COST = "cost"  # 비용 절감
    QUALITY = "quality"  # 품질 향상
    RELIABILITY = "reliability"  # 안정성
    BEST_PRACTICE = "best_practice"  # 모범 사례


class Priority(Enum):
    """우선순위"""

    CRITICAL = "critical"  # 즉시 조치 필요
    HIGH = "high"  # 높음
    MEDIUM = "medium"  # 중간
    LOW = "low"  # 낮음


@dataclass
class Recommendation:
    """
    최적화 권장사항

    Attributes:
        category: 카테고리
        priority: 우선순위
        title: 제목
        description: 설명
        rationale: 근거
        action: 조치 방법
        expected_impact: 예상 효과
        metadata: 추가 메타데이터
    """

    category: RecommendationCategory
    priority: Priority
    title: str
    description: str
    rationale: str = ""
    action: str = ""
    expected_impact: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class Recommender:
    """
    최적화 권장사항 생성기

    책임:
    - 프로파일링 결과 분석
    - 벤치마크 결과 분석
    - 파라미터 최적화 결과 분석
    - 실행 가능한 권장사항 생성

    Example:
        ```python
        recommender = Recommender()

        # Analyze profile result
        profile_recommendations = recommender.analyze_profile(profile_result)

        # Analyze benchmark result
        benchmark_recommendations = recommender.analyze_benchmark(benchmark_result)

        # Analyze parameters
        param_recommendations = recommender.analyze_parameters(current_params)

        # Get all recommendations
        all_recommendations = (
            profile_recommendations +
            benchmark_recommendations +
            param_recommendations
        )

        # Sort by priority
        critical = [r for r in all_recommendations if r.priority == Priority.CRITICAL]
        high = [r for r in all_recommendations if r.priority == Priority.HIGH]

        for rec in critical + high:
            print(f"[{rec.priority.value}] {rec.title}")
            print(f"  {rec.description}")
            print(f"  Action: {rec.action}")
        ```
    """

    def __init__(self) -> None:
        """Initialize recommender"""
        pass

    def analyze_profile(self, profile_result: Any) -> List[Recommendation]:
        """
        프로파일링 결과 분석 후 권장사항 생성

        Args:
            profile_result: ProfileResult

        Returns:
            List[Recommendation]: 권장사항 리스트
        """
        recommendations = []

        # Check total duration
        if profile_result.total_duration_ms > 5000:  # 5 seconds
            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.PERFORMANCE,
                    priority=Priority.CRITICAL,
                    title="High Latency Detected",
                    description=f"Total latency is {profile_result.total_duration_ms:.0f}ms, "
                    "which exceeds the 5-second threshold.",
                    rationale="High latency degrades user experience and may cause timeouts.",
                    action="Profile individual components to identify bottlenecks. "
                    "Consider parallelizing independent operations.",
                    expected_impact="Reduce latency by 30-50%",
                )
            )

        # Check cost
        if profile_result.total_cost > 0.10:  # $0.10 per query
            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.COST,
                    priority=Priority.HIGH,
                    title="High Cost Per Query",
                    description=f"Cost is ${profile_result.total_cost:.4f} per query.",
                    rationale="High per-query cost may not be sustainable at scale.",
                    action="Consider using a cheaper model, reducing token usage, "
                    "or implementing caching.",
                    expected_impact="Reduce cost by 40-60%",
                )
            )

        # Check bottlenecks
        breakdown = profile_result.get_breakdown()

        for component_name, percentage in breakdown.items():
            if percentage > 40:
                metrics = profile_result.components[component_name]

                if "embedding" in component_name.lower():
                    recommendations.append(
                        Recommendation(
                            category=RecommendationCategory.PERFORMANCE,
                            priority=Priority.HIGH,
                            title=f"Embedding Bottleneck ({percentage:.1f}%)",
                            description=f"Embedding takes {percentage:.1f}% of total time.",
                            rationale="Embedding is the slowest component.",
                            action="Consider caching embeddings, using a faster model "
                            "(e.g., all-MiniLM-L6-v2), or batching requests.",
                            expected_impact="Reduce embedding time by 50-70%",
                        )
                    )

                elif "retrieval" in component_name.lower():
                    recommendations.append(
                        Recommendation(
                            category=RecommendationCategory.PERFORMANCE,
                            priority=Priority.HIGH,
                            title=f"Retrieval Bottleneck ({percentage:.1f}%)",
                            description=f"Retrieval takes {percentage:.1f}% of total time.",
                            rationale="Retrieval is the slowest component.",
                            action="Optimize vector index (use HNSW), reduce top_k, "
                            "or use approximate search.",
                            expected_impact="Reduce retrieval time by 30-50%",
                        )
                    )

                elif "generation" in component_name.lower():
                    recommendations.append(
                        Recommendation(
                            category=RecommendationCategory.PERFORMANCE,
                            priority=Priority.MEDIUM,
                            title=f"Generation Bottleneck ({percentage:.1f}%)",
                            description=f"Generation takes {percentage:.1f}% of total time.",
                            rationale="LLM generation is the slowest component.",
                            action="Use a faster model, reduce max_tokens, "
                            "or implement streaming.",
                            expected_impact="Reduce generation time by 20-40%",
                        )
                    )

        return recommendations

    def analyze_benchmark(self, benchmark_result: Any) -> List[Recommendation]:
        """
        벤치마크 결과 분석 후 권장사항 생성

        Args:
            benchmark_result: BenchmarkResult

        Returns:
            List[Recommendation]: 권장사항 리스트
        """
        recommendations = []

        # Check average score
        if benchmark_result.avg_score < 0.7:  # Below 70%
            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.QUALITY,
                    priority=Priority.CRITICAL,
                    title="Low Quality Score",
                    description=f"Average quality score is {benchmark_result.avg_score:.2f}, "
                    "which is below the 0.7 threshold.",
                    rationale="Low quality scores indicate poor system performance.",
                    action="Review retrieval strategy, improve prompt engineering, "
                    "or use a more capable model.",
                    expected_impact="Improve quality score to >0.8",
                )
            )

        # Check p95 latency
        if benchmark_result.p95_latency > 3.0:  # 3 seconds
            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.RELIABILITY,
                    priority=Priority.HIGH,
                    title="High P95 Latency",
                    description=f"P95 latency is {benchmark_result.p95_latency:.2f}s.",
                    rationale="High tail latency affects user experience for 5% of requests.",
                    action="Investigate outliers, implement timeouts, or add caching.",
                    expected_impact="Reduce P95 latency by 30-50%",
                )
            )

        # Check throughput
        if benchmark_result.throughput < 1.0:  # < 1 query/sec
            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.PERFORMANCE,
                    priority=Priority.MEDIUM,
                    title="Low Throughput",
                    description=f"Throughput is {benchmark_result.throughput:.2f} queries/sec.",
                    rationale="Low throughput may not meet production requirements.",
                    action="Parallelize operations, use batching, or scale horizontally.",
                    expected_impact="Increase throughput to >5 queries/sec",
                )
            )

        return recommendations

    def analyze_parameters(
        self,
        current_params: Dict[str, Any],
        param_ranges: Optional[Dict[str, tuple]] = None,
    ) -> List[Recommendation]:
        """
        파라미터 분석 후 권장사항 생성

        Args:
            current_params: 현재 파라미터
            param_ranges: 파라미터 범위 {param_name: (min, max)}

        Returns:
            List[Recommendation]: 권장사항 리스트
        """
        recommendations = []

        # Check top_k
        if "top_k" in current_params:
            top_k = current_params["top_k"]

            if top_k > 20:
                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.PERFORMANCE,
                        priority=Priority.MEDIUM,
                        title="High top_k Value",
                        description=f"top_k is set to {top_k}, which may be too high.",
                        rationale="High top_k increases retrieval time and may introduce noise.",
                        action="Consider reducing top_k to 10-15 and using reranking.",
                        expected_impact="Reduce retrieval time by 20-30%",
                    )
                )

        # Check score_threshold
        if "score_threshold" in current_params:
            threshold = current_params["score_threshold"]

            if threshold < 0.5:
                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.QUALITY,
                        priority=Priority.LOW,
                        title="Low Score Threshold",
                        description=f"score_threshold is {threshold}, which may be too low.",
                        rationale="Low threshold may allow irrelevant documents.",
                        action="Consider increasing threshold to 0.6-0.7.",
                        expected_impact="Improve precision by filtering low-quality results",
                    )
                )

        # Check temperature
        if "temperature" in current_params:
            temp = current_params["temperature"]

            if temp > 1.0:
                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.QUALITY,
                        priority=Priority.LOW,
                        title="High Temperature",
                        description=f"temperature is {temp}, which may be too high.",
                        rationale="High temperature increases randomness and may reduce quality.",
                        action="Consider reducing temperature to 0.3-0.7 for factual tasks.",
                        expected_impact="Improve consistency and accuracy",
                    )
                )

        # Check max_tokens
        if "max_tokens" in current_params:
            max_tokens = current_params["max_tokens"]

            if max_tokens > 2000:
                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.COST,
                        priority=Priority.MEDIUM,
                        title="High max_tokens",
                        description=f"max_tokens is {max_tokens}, which may be excessive.",
                        rationale="High max_tokens increases cost and latency.",
                        action="Consider reducing max_tokens to 500-1000 unless long responses are needed.",
                        expected_impact="Reduce cost by 30-50%",
                    )
                )

        return recommendations

    def analyze_best_practices(
        self,
        system_config: Dict[str, Any],
    ) -> List[Recommendation]:
        """
        모범 사례 체크

        Args:
            system_config: 시스템 설정

        Returns:
            List[Recommendation]: 권장사항 리스트
        """
        recommendations = []

        # Check if using caching
        if not system_config.get("caching_enabled", False):
            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.BEST_PRACTICE,
                    priority=Priority.MEDIUM,
                    title="Caching Not Enabled",
                    description="Caching is not enabled.",
                    rationale="Caching can significantly reduce latency and cost for repeated queries.",
                    action="Enable caching for embeddings and LLM responses.",
                    expected_impact="Reduce latency and cost by 40-60% for repeated queries",
                )
            )

        # Check if using monitoring
        if not system_config.get("monitoring_enabled", False):
            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.BEST_PRACTICE,
                    priority=Priority.HIGH,
                    title="Monitoring Not Enabled",
                    description="Monitoring is not enabled.",
                    rationale="Monitoring is essential for production systems.",
                    action="Enable monitoring with metrics, logging, and tracing.",
                    expected_impact="Detect and fix issues faster",
                )
            )

        # Check if using evaluation
        if not system_config.get("evaluation_enabled", False):
            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.BEST_PRACTICE,
                    priority=Priority.MEDIUM,
                    title="Evaluation Not Enabled",
                    description="Automated evaluation is not enabled.",
                    rationale="Continuous evaluation ensures quality over time.",
                    action="Set up automated evaluation with TruLens, RAGAS, or DeepEval.",
                    expected_impact="Maintain quality as system evolves",
                )
            )

        return recommendations

    def generate_optimization_plan(
        self,
        all_recommendations: List[Recommendation],
    ) -> Dict[str, List[Recommendation]]:
        """
        최적화 계획 생성 (우선순위별로 정렬)

        Args:
            all_recommendations: 모든 권장사항

        Returns:
            Dict[str, List[Recommendation]]: 우선순위별 권장사항
        """
        plan: Dict[str, List[Recommendation]] = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
        }

        for rec in all_recommendations:
            plan[rec.priority.value].append(rec)

        return plan


def print_recommendations(
    recommendations: List[Recommendation],
    max_items: int = 10,
) -> None:
    """
    권장사항 출력 (편의 함수)

    Args:
        recommendations: 권장사항 리스트
        max_items: 최대 출력 개수
    """
    # Sort by priority
    priority_order = {
        Priority.CRITICAL: 0,
        Priority.HIGH: 1,
        Priority.MEDIUM: 2,
        Priority.LOW: 3,
    }

    sorted_recommendations = sorted(recommendations, key=lambda r: priority_order[r.priority])

    for i, rec in enumerate(sorted_recommendations[:max_items], 1):
        print(f"\n{i}. [{rec.priority.value.upper()}] {rec.title}")
        print(f"   Category: {rec.category.value}")
        print(f"   {rec.description}")

        if rec.rationale:
            print(f"   Rationale: {rec.rationale}")

        if rec.action:
            print(f"   Action: {rec.action}")

        if rec.expected_impact:
            print(f"   Expected Impact: {rec.expected_impact}")
