"""
Evaluation Analytics - 평가 분석 및 인사이트
"""

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .results import BatchEvaluationResult


@dataclass
class MetricTrend:
    """메트릭 추이"""

    metric_name: str
    timestamps: List[str]
    scores: List[float]
    trend: str  # "improving", "declining", "stable"
    average_score: float
    min_score: float
    max_score: float
    std_dev: float


@dataclass
class CorrelationAnalysis:
    """상관관계 분석"""

    metric_a: str
    metric_b: str
    correlation: float
    significance: str  # "strong", "moderate", "weak", "none"


@dataclass
class EvaluationAnalytics:
    """평가 분석 결과"""

    metric_trends: List[MetricTrend]
    correlations: List[CorrelationAnalysis]
    summary_stats: Dict[str, Any]
    insights: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvaluationAnalyticsEngine:
    """
    평가 분석 엔진

    평가 결과를 분석하여 트렌드, 상관관계, 인사이트 제공
    """

    def __init__(self):
        self._history: List[Dict[str, Any]] = []

    def add_evaluation_result(
        self,
        result: BatchEvaluationResult,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        평가 결과 추가

        Args:
            result: 배치 평가 결과
            timestamp: 평가 시간 (선택적)
            metadata: 추가 메타데이터 (선택적)
        """
        self._history.append(
            {
                "timestamp": timestamp or datetime.now(),
                "result": result,
                "metadata": metadata or {},
            }
        )

    def analyze_trends(
        self,
        metric_name: Optional[str] = None,
        window_days: int = 30,
    ) -> List[MetricTrend]:
        """
        메트릭 추이 분석

        Args:
            metric_name: 메트릭 이름 (선택적, 없으면 모든 메트릭)
            window_days: 분석 기간 (일)

        Returns:
            List[MetricTrend]: 메트릭별 추이
        """
        cutoff_date = datetime.now() - timedelta(days=window_days)
        recent_history = [h for h in self._history if h["timestamp"] >= cutoff_date]

        if not recent_history:
            return []

        # 메트릭별로 그룹화
        metrics_data: Dict[str, List[Dict[str, Any]]] = {}

        for record in recent_history:
            result = record["result"]
            for eval_result in result.results:
                metric = eval_result.metric_name
                if metric_name and metric != metric_name:
                    continue

                if metric not in metrics_data:
                    metrics_data[metric] = []

                metrics_data[metric].append(
                    {
                        "timestamp": record["timestamp"],
                        "score": eval_result.score,
                    }
                )

        # 추이 계산
        trends = []
        for metric, data in metrics_data.items():
            # 시간순 정렬
            data.sort(key=lambda x: x["timestamp"])

            timestamps = [d["timestamp"].isoformat() for d in data]
            scores = [d["score"] for d in data]

            if len(scores) < 2:
                continue

            # 통계 계산
            avg_score = statistics.mean(scores)
            min_score = min(scores)
            max_score = max(scores)
            std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0

            # 추이 계산 (선형 회귀 기울기)
            if len(scores) > 1:
                # 간단한 추이: 최근 점수와 초기 점수 비교
                mid_point = len(scores) // 2
                recent_avg = statistics.mean(scores[:mid_point]) if mid_point > 0 else avg_score
                early_avg = (
                    statistics.mean(scores[mid_point:]) if mid_point < len(scores) else avg_score
                )

                if recent_avg > early_avg * 1.05:
                    trend = "improving"
                elif recent_avg < early_avg * 0.95:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "stable"

            trends.append(
                MetricTrend(
                    metric_name=metric,
                    timestamps=timestamps,
                    scores=scores,
                    trend=trend,
                    average_score=avg_score,
                    min_score=min_score,
                    max_score=max_score,
                    std_dev=std_dev,
                )
            )

        return trends

    def analyze_correlations(
        self,
        window_days: int = 30,
        min_samples: int = 10,
    ) -> List[CorrelationAnalysis]:
        """
        메트릭 간 상관관계 분석

        Args:
            window_days: 분석 기간 (일)
            min_samples: 최소 샘플 수

        Returns:
            List[CorrelationAnalysis]: 상관관계 분석 결과
        """
        cutoff_date = datetime.now() - timedelta(days=window_days)
        recent_history = [h for h in self._history if h["timestamp"] >= cutoff_date]

        if len(recent_history) < min_samples:
            return []

        # 메트릭별 점수 수집
        metrics_scores: Dict[str, List[float]] = {}

        for record in recent_history:
            result = record["result"]
            for eval_result in result.results:
                metric = eval_result.metric_name
                if metric not in metrics_scores:
                    metrics_scores[metric] = []
                metrics_scores[metric].append(eval_result.score)

        # 최소 샘플 수 확인
        valid_metrics = {
            m: scores for m, scores in metrics_scores.items() if len(scores) >= min_samples
        }

        if len(valid_metrics) < 2:
            return []

        # 모든 메트릭 쌍에 대해 상관관계 계산
        correlations = []
        metric_names = list(valid_metrics.keys())

        for i in range(len(metric_names)):
            for j in range(i + 1, len(metric_names)):
                metric_a = metric_names[i]
                metric_b = metric_names[j]

                scores_a = valid_metrics[metric_a]
                scores_b = valid_metrics[metric_b]

                # 길이 맞추기 (최소 길이로)
                min_len = min(len(scores_a), len(scores_b))
                scores_a = scores_a[:min_len]
                scores_b = scores_b[:min_len]

                if len(scores_a) < min_samples:
                    continue

                # 피어슨 상관계수 계산
                correlation = self._calculate_correlation(scores_a, scores_b)

                # 유의성 판단
                abs_corr = abs(correlation)
                if abs_corr >= 0.7:
                    significance = "strong"
                elif abs_corr >= 0.4:
                    significance = "moderate"
                elif abs_corr >= 0.2:
                    significance = "weak"
                else:
                    significance = "none"

                correlations.append(
                    CorrelationAnalysis(
                        metric_a=metric_a,
                        metric_b=metric_b,
                        correlation=correlation,
                        significance=significance,
                    )
                )

        return correlations

    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """피어슨 상관계수 계산"""
        if len(x) != len(y):
            raise ValueError("Lists must have same length")

        n = len(x)
        if n < 2:
            return 0.0

        # 평균 계산
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)

        # 분산 및 공분산 계산
        sum_xy = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        sum_x2 = sum((x[i] - mean_x) ** 2 for i in range(n))
        sum_y2 = sum((y[i] - mean_y) ** 2 for i in range(n))

        # 상관계수 계산
        denominator = (sum_x2 * sum_y2) ** 0.5
        if denominator == 0:
            return 0.0

        correlation = sum_xy / denominator
        return correlation

    def generate_analytics(
        self,
        window_days: int = 30,
        include_insights: bool = True,
    ) -> EvaluationAnalytics:
        """
        종합 분석 생성

        Args:
            window_days: 분석 기간 (일)
            include_insights: 인사이트 포함 여부

        Returns:
            EvaluationAnalytics: 분석 결과
        """
        # 추이 분석
        trends = self.analyze_trends(window_days=window_days)

        # 상관관계 분석
        correlations = self.analyze_correlations(window_days=window_days)

        # 요약 통계
        summary_stats = self._calculate_summary_stats(window_days=window_days)

        # 인사이트 생성
        insights = []
        if include_insights:
            insights = self._generate_insights(trends, correlations, summary_stats)

        return EvaluationAnalytics(
            metric_trends=trends,
            correlations=correlations,
            summary_stats=summary_stats,
            insights=insights,
            metadata={
                "window_days": window_days,
                "total_evaluations": len(self._history),
                "generated_at": datetime.now().isoformat(),
            },
        )

    def _calculate_summary_stats(self, window_days: int = 30) -> Dict[str, Any]:
        """요약 통계 계산"""
        cutoff_date = datetime.now() - timedelta(days=window_days)
        recent_history = [h for h in self._history if h["timestamp"] >= cutoff_date]

        if not recent_history:
            return {
                "total_evaluations": 0,
                "average_scores": {},
                "metric_counts": {},
            }

        # 메트릭별 통계
        metric_stats: Dict[str, List[float]] = {}

        for record in recent_history:
            result = record["result"]
            for eval_result in result.results:
                metric = eval_result.metric_name
                if metric not in metric_stats:
                    metric_stats[metric] = []
                metric_stats[metric].append(eval_result.score)

        # 평균 점수 계산
        average_scores = {
            metric: statistics.mean(scores) for metric, scores in metric_stats.items()
        }

        # 메트릭별 개수
        metric_counts = {metric: len(scores) for metric, scores in metric_stats.items()}

        return {
            "total_evaluations": len(recent_history),
            "average_scores": average_scores,
            "metric_counts": metric_counts,
            "metrics": list(metric_stats.keys()),
        }

    def _generate_insights(
        self,
        trends: List[MetricTrend],
        correlations: List[CorrelationAnalysis],
        summary_stats: Dict[str, Any],
    ) -> List[str]:
        """인사이트 생성"""
        insights = []

        # 추이 기반 인사이트
        improving_metrics = [t for t in trends if t.trend == "improving"]
        declining_metrics = [t for t in trends if t.trend == "declining"]

        if improving_metrics:
            insights.append(
                f"✅ {len(improving_metrics)}개 메트릭이 개선 중: "
                f"{', '.join(m.metric_name for m in improving_metrics)}"
            )

        if declining_metrics:
            insights.append(
                f"⚠️ {len(declining_metrics)}개 메트릭이 하락 중: "
                f"{', '.join(m.metric_name for m in declining_metrics)}"
            )

        # 상관관계 기반 인사이트
        strong_correlations = [c for c in correlations if c.significance == "strong"]
        if strong_correlations:
            insights.append(
                f"🔗 {len(strong_correlations)}개의 강한 상관관계 발견: "
                f"{', '.join(f'{c.metric_a}-{c.metric_b}' for c in strong_correlations[:3])}"
            )

        # 요약 통계 기반 인사이트
        if summary_stats.get("average_scores"):
            avg_scores = summary_stats["average_scores"]
            best_metric = max(avg_scores.items(), key=lambda x: x[1])
            worst_metric = min(avg_scores.items(), key=lambda x: x[1])

            insights.append(f"📊 최고 성능 메트릭: {best_metric[0]} ({best_metric[1]:.3f})")
            insights.append(f"📉 개선 필요 메트릭: {worst_metric[0]} ({worst_metric[1]:.3f})")

        return insights

    def clear_history(self, days: Optional[int] = None):
        """
        기록 삭제

        Args:
            days: 유지할 기간 (일), None이면 모두 삭제
        """
        if days is None:
            self._history.clear()
        else:
            cutoff_date = datetime.now() - timedelta(days=days)
            self._history = [h for h in self._history if h["timestamp"] >= cutoff_date]
