"""
Drift Detection - 모델 드리프트 감지
"""

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


@dataclass
class DriftAlert:
    """드리프트 알림"""

    alert_id: str
    metric_name: str
    timestamp: datetime
    current_score: float
    baseline_score: float
    drift_magnitude: float  # 변화량
    drift_type: str  # "performance_degradation", "distribution_shift", etc.
    severity: str  # "low", "medium", "high", "critical"
    metadata: Dict[str, Any] = field(default_factory=dict)


class DriftDetector:
    """
    모델 드리프트 감지기

    평가 점수의 변화를 모니터링하고 드리프트를 감지
    """

    def __init__(
        self,
        baseline_window_days: int = 7,
        detection_window_days: int = 1,
        threshold_std: float = 2.0,  # 표준편차 기준
        threshold_percent: float = 0.2,  # 20% 변화
    ):
        """
        Args:
            baseline_window_days: 기준선 계산 기간 (일)
            detection_window_days: 감지 기간 (일)
            threshold_std: 표준편차 임계값 (기본값: 2.0 = 2σ)
            threshold_percent: 백분율 변화 임계값 (기본값: 0.2 = 20%)
        """
        self.baseline_window_days = baseline_window_days
        self.detection_window_days = detection_window_days
        self.threshold_std = threshold_std
        self.threshold_percent = threshold_percent
        self._history: List[
            Dict[str, Any]
        ] = []  # [{"timestamp": ..., "metric": ..., "score": ...}]
        self._alert_counter = 0

    def record_score(
        self,
        metric_name: str,
        score: float,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        점수 기록

        Args:
            metric_name: 메트릭 이름
            score: 점수
            timestamp: 시간 (선택적, 없으면 현재 시간)
            metadata: 추가 메타데이터
        """
        self._history.append(
            {
                "timestamp": timestamp or datetime.now(),
                "metric_name": metric_name,
                "score": score,
                "metadata": metadata or {},
            }
        )

    def detect_drift(
        self,
        metric_name: Optional[str] = None,
        current_score: Optional[float] = None,
    ) -> List[DriftAlert]:
        """
        드리프트 감지

        Args:
            metric_name: 메트릭 이름 (선택적, 없으면 모든 메트릭)
            current_score: 현재 점수 (선택적, 없으면 최근 기록 사용)

        Returns:
            List[DriftAlert]: 감지된 드리프트 알림 리스트
        """
        alerts = []

        # 메트릭별로 처리
        metrics_to_check = [metric_name] if metric_name else self._get_all_metrics()

        for metric in metrics_to_check:
            metric_alerts = self._detect_drift_for_metric(metric, current_score)
            alerts.extend(metric_alerts)

        return alerts

    def _get_all_metrics(self) -> List[str]:
        """모든 메트릭 이름 조회"""
        return list(set(h["metric_name"] for h in self._history))

    def _detect_drift_for_metric(
        self,
        metric_name: str,
        current_score: Optional[float] = None,
    ) -> List[DriftAlert]:
        """특정 메트릭에 대한 드리프트 감지"""
        # 메트릭별 기록 필터링
        metric_history = [h for h in self._history if h["metric_name"] == metric_name]

        if len(metric_history) < 2:
            return []  # 데이터 부족

        # 현재 점수 결정
        if current_score is None:
            current_score = metric_history[-1]["score"]

        # 기준선 계산 (baseline_window_days 기간)
        cutoff_date = datetime.now() - timedelta(days=self.baseline_window_days)
        baseline_scores = [h["score"] for h in metric_history if h["timestamp"] >= cutoff_date]

        if len(baseline_scores) < 2:
            return []  # 기준선 데이터 부족

        # 기준선 통계
        baseline_mean = statistics.mean(baseline_scores)
        baseline_std = statistics.stdev(baseline_scores) if len(baseline_scores) > 1 else 0.0

        # 드리프트 감지
        alerts = []

        # 1. 성능 저하 감지 (점수 하락)
        score_diff = current_score - baseline_mean
        percent_change = abs(score_diff / baseline_mean) if baseline_mean != 0 else 0.0

        if score_diff < 0 and percent_change >= self.threshold_percent:
            # 표준편차 기준 확인
            if baseline_std > 0:
                z_score = abs(score_diff) / baseline_std
                if z_score >= self.threshold_std:
                    severity = self._calculate_severity(percent_change, z_score)
                    alerts.append(
                        DriftAlert(
                            alert_id=f"drift_{self._alert_counter}",
                            metric_name=metric_name,
                            timestamp=datetime.now(),
                            current_score=current_score,
                            baseline_score=baseline_mean,
                            drift_magnitude=abs(score_diff),
                            drift_type="performance_degradation",
                            severity=severity,
                            metadata={
                                "percent_change": percent_change,
                                "z_score": z_score,
                                "baseline_std": baseline_std,
                            },
                        )
                    )
                    self._alert_counter += 1

        # 2. 분포 변화 감지 (변동성 증가)
        if len(baseline_scores) >= 5:
            recent_scores = [h["score"] for h in metric_history[-5:]]
            recent_std = statistics.stdev(recent_scores) if len(recent_scores) > 1 else 0.0

            if baseline_std > 0 and recent_std > baseline_std * 1.5:
                alerts.append(
                    DriftAlert(
                        alert_id=f"drift_{self._alert_counter}",
                        metric_name=metric_name,
                        timestamp=datetime.now(),
                        current_score=current_score,
                        baseline_score=baseline_mean,
                        drift_magnitude=recent_std - baseline_std,
                        drift_type="distribution_shift",
                        severity="medium",
                        metadata={
                            "baseline_std": baseline_std,
                            "recent_std": recent_std,
                        },
                    )
                )
                self._alert_counter += 1

        return alerts

    def _calculate_severity(self, percent_change: float, z_score: float) -> str:
        """심각도 계산"""
        if percent_change >= 0.5 or z_score >= 3.0:
            return "critical"
        elif percent_change >= 0.3 or z_score >= 2.5:
            return "high"
        elif percent_change >= 0.2 or z_score >= 2.0:
            return "medium"
        else:
            return "low"

    def get_baseline_stats(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """기준선 통계 조회"""
        cutoff_date = datetime.now() - timedelta(days=self.baseline_window_days)
        baseline_scores = [
            h["score"]
            for h in self._history
            if h["metric_name"] == metric_name and h["timestamp"] >= cutoff_date
        ]

        if not baseline_scores:
            return None

        return {
            "metric_name": metric_name,
            "mean": statistics.mean(baseline_scores),
            "median": statistics.median(baseline_scores),
            "std": statistics.stdev(baseline_scores) if len(baseline_scores) > 1 else 0.0,
            "min": min(baseline_scores),
            "max": max(baseline_scores),
            "count": len(baseline_scores),
        }

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
