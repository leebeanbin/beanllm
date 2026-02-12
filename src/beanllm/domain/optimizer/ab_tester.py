"""
ABTester - A/B 테스팅 프레임워크
SOLID 원칙:
- SRP: A/B 테스팅만 담당
- OCP: 새로운 통계 테스트 추가 가능
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ABTestResult:
    """
    A/B 테스트 결과

    Attributes:
        variant_a_name: Variant A 이름
        variant_b_name: Variant B 이름
        variant_a_mean: Variant A 평균
        variant_b_mean: Variant B 평균
        variant_a_std: Variant A 표준편차
        variant_b_std: Variant B 표준편차
        p_value: p-value (통계적 유의성)
        is_significant: 통계적으로 유의한지 (p < 0.05)
        confidence_level: 신뢰 수준
        winner: 승자 ("A", "B", "tie")
        lift: 향상률 (B가 A보다 얼마나 나은지, %)
        sample_size_a: Variant A 샘플 수
        sample_size_b: Variant B 샘플 수
    """

    variant_a_name: str
    variant_b_name: str
    variant_a_mean: float
    variant_b_mean: float
    variant_a_std: float = 0.0
    variant_b_std: float = 0.0
    p_value: float = 1.0
    is_significant: bool = False
    confidence_level: float = 0.95
    winner: str = "tie"
    lift: float = 0.0
    sample_size_a: int = 0
    sample_size_b: int = 0

    def __post_init__(self):
        """자동 계산"""
        # Calculate lift
        if self.variant_a_mean > 0:
            self.lift = (self.variant_b_mean - self.variant_a_mean) / self.variant_a_mean * 100

        # Determine winner
        if self.is_significant:
            if self.variant_b_mean > self.variant_a_mean:
                self.winner = "B"
            else:
                self.winner = "A"
        else:
            self.winner = "tie"


class ABTester:
    """
    A/B 테스터

    책임:
    - A/B 테스트 실행
    - 통계적 유의성 검증
    - 성능 비교

    Example:
        ```python
        tester = ABTester()

        # Define variants
        variant_a = lambda query: system_v1.query(query)
        variant_b = lambda query: system_v2.query(query)

        # Define evaluation function
        def evaluate(result):
            # Return quality score 0.0-1.0
            return calculate_quality(result)

        # Run A/B test
        result = tester.run_test(
            variant_a=variant_a,
            variant_b=variant_b,
            evaluation_fn=evaluate,
            queries=test_queries,
            variant_a_name="Baseline",
            variant_b_name="Optimized"
        )

        print(f"Winner: {result.winner}")
        print(f"Lift: {result.lift:.1f}%")
        print(f"P-value: {result.p_value:.4f}")
        print(f"Significant: {result.is_significant}")
        ```
    """

    def __init__(self) -> None:
        """Initialize A/B tester"""
        pass

    def run_test(
        self,
        variant_a: Callable[[Any], Any],
        variant_b: Callable[[Any], Any],
        evaluation_fn: Callable[[Any], float],
        queries: List[Any],
        variant_a_name: str = "A",
        variant_b_name: str = "B",
        confidence_level: float = 0.95,
    ) -> ABTestResult:
        """
        A/B 테스트 실행

        Args:
            variant_a: Variant A 함수 (query -> result)
            variant_b: Variant B 함수 (query -> result)
            evaluation_fn: 평가 함수 (result -> score)
            queries: 테스트 쿼리 리스트
            variant_a_name: Variant A 이름
            variant_b_name: Variant B 이름
            confidence_level: 신뢰 수준 (default: 0.95)

        Returns:
            ABTestResult: A/B 테스트 결과
        """
        logger.info(
            f"Running A/B test: {variant_a_name} vs {variant_b_name}, {len(queries)} queries"
        )

        scores_a = []
        scores_b = []

        for i, query in enumerate(queries):
            # Evaluate variant A
            try:
                result_a = variant_a(query)
                score_a = evaluation_fn(result_a)
                scores_a.append(score_a)
            except Exception as e:
                logger.error(f"Error in variant A on query {i}: {e}")
                scores_a.append(0.0)

            # Evaluate variant B
            try:
                result_b = variant_b(query)
                score_b = evaluation_fn(result_b)
                scores_b.append(score_b)
            except Exception as e:
                logger.error(f"Error in variant B on query {i}: {e}")
                scores_b.append(0.0)

            if (i + 1) % 10 == 0:
                logger.debug(f"Evaluated {i + 1}/{len(queries)} queries")

        # Calculate statistics
        mean_a = statistics.mean(scores_a)
        mean_b = statistics.mean(scores_b)

        std_a = statistics.stdev(scores_a) if len(scores_a) > 1 else 0.0
        std_b = statistics.stdev(scores_b) if len(scores_b) > 1 else 0.0

        # Perform t-test
        p_value = self._t_test(scores_a, scores_b)

        # Determine significance
        alpha = 1.0 - confidence_level
        is_significant = p_value < alpha

        result = ABTestResult(
            variant_a_name=variant_a_name,
            variant_b_name=variant_b_name,
            variant_a_mean=mean_a,
            variant_b_mean=mean_b,
            variant_a_std=std_a,
            variant_b_std=std_b,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=confidence_level,
            sample_size_a=len(scores_a),
            sample_size_b=len(scores_b),
        )

        logger.info(
            f"A/B test completed: winner={result.winner}, "
            f"lift={result.lift:.1f}%, p-value={result.p_value:.4f}"
        )

        return result

    def _t_test(self, scores_a: List[float], scores_b: List[float]) -> float:
        """
        Independent two-sample t-test

        Returns:
            p-value
        """
        if len(scores_a) < 2 or len(scores_b) < 2:
            return 1.0

        mean_a = statistics.mean(scores_a)
        mean_b = statistics.mean(scores_b)

        var_a = statistics.variance(scores_a)
        var_b = statistics.variance(scores_b)

        n_a = len(scores_a)
        n_b = len(scores_b)

        # Pooled standard error
        pooled_se = ((var_a / n_a) + (var_b / n_b)) ** 0.5

        if pooled_se == 0:
            return 1.0

        # T-statistic
        t_stat = (mean_b - mean_a) / pooled_se

        # Degrees of freedom (Welch's approximation)
        df = (var_a / n_a + var_b / n_b) ** 2 / (
            (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        )

        # Calculate p-value (two-tailed)
        p_value = self._t_distribution_p_value(abs(t_stat), df)

        return p_value * 2  # two-tailed

    def _t_distribution_p_value(self, t_stat: float, df: float) -> float:
        """
        T-분포 p-value 계산 (근사)

        Uses normal approximation for large df (>30)
        """
        if df > 30:
            # Use normal approximation
            return self._normal_distribution_p_value(t_stat)

        # For small df, use lookup table (simplified)
        # Critical values for df=10, two-tailed, alpha=0.05: t=2.228
        critical_values = {
            5: 2.571,
            10: 2.228,
            20: 2.086,
            30: 2.042,
        }

        # Find closest df
        closest_df = min(critical_values.keys(), key=lambda x: abs(x - df))
        critical_value = critical_values[closest_df]

        if abs(t_stat) > critical_value:
            return 0.01  # p < 0.05
        else:
            return 0.10  # p > 0.05

    def _normal_distribution_p_value(self, z_score: float) -> float:
        """
        정규분포 p-value 계산 (근사)

        Uses error function approximation
        """
        import math

        # Standard normal CDF approximation
        x = z_score / math.sqrt(2.0)

        # Error function approximation
        a = 0.3275911
        t = 1.0 / (1.0 + a * abs(x))

        erf = 1.0 - (
            (
                (((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
                + 0.254829592
            )
            * t
            * math.exp(-x * x)
        )

        if x < 0:
            erf = -erf

        # CDF
        cdf = 0.5 * (1.0 + erf)

        # p-value (one-tailed)
        return 1.0 - cdf

    def calculate_required_sample_size(
        self,
        baseline_mean: float,
        baseline_std: float,
        minimum_detectable_effect: float,
        power: float = 0.8,
        alpha: float = 0.05,
    ) -> int:
        """
        필요한 샘플 크기 계산

        Args:
            baseline_mean: 베이스라인 평균
            baseline_std: 베이스라인 표준편차
            minimum_detectable_effect: 탐지하려는 최소 효과 (% lift)
            power: 통계적 검정력 (default: 0.8)
            alpha: 유의 수준 (default: 0.05)

        Returns:
            int: 필요한 샘플 크기 (각 variant당)

        Example:
            ```python
            n = tester.calculate_required_sample_size(
                baseline_mean=0.75,
                baseline_std=0.15,
                minimum_detectable_effect=5.0,  # 5% lift
                power=0.8,
                alpha=0.05
            )
            print(f"Need {n} samples per variant")
            ```
        """
        import math

        # Convert effect to absolute difference
        effect_size = baseline_mean * (minimum_detectable_effect / 100)

        # Cohen's d
        d = effect_size / baseline_std

        # Z-scores for alpha and power
        z_alpha = 1.96  # for alpha=0.05 (two-tailed)
        z_power = 0.84  # for power=0.8

        # Sample size formula
        n = 2 * ((z_alpha + z_power) / d) ** 2

        return int(math.ceil(n))


def compare_multiple_variants(
    variants: Dict[str, Callable[[Any], Any]],
    evaluation_fn: Callable[[Any], float],
    queries: List[Any],
) -> Dict[str, ABTestResult]:
    """
    여러 variant 비교 (편의 함수)

    Args:
        variants: {variant_name: variant_fn}
        evaluation_fn: 평가 함수
        queries: 테스트 쿼리

    Returns:
        Dict[str, ABTestResult]: 쌍별 비교 결과

    Example:
        ```python
        variants = {
            "baseline": system_v1.query,
            "optimized_v1": system_v2.query,
            "optimized_v2": system_v3.query,
        }

        results = compare_multiple_variants(variants, evaluate, queries)

        for comparison_name, result in results.items():
            print(f"{comparison_name}: winner={result.winner}, lift={result.lift:.1f}%")
        ```
    """
    tester = ABTester()

    variant_names = list(variants.keys())
    results = {}

    for i, name_a in enumerate(variant_names):
        for name_b in variant_names[i + 1 :]:
            comparison_name = f"{name_a}_vs_{name_b}"

            result = tester.run_test(
                variant_a=variants[name_a],
                variant_b=variants[name_b],
                evaluation_fn=evaluation_fn,
                queries=queries,
                variant_a_name=name_a,
                variant_b_name=name_b,
            )

            results[comparison_name] = result

    return results
