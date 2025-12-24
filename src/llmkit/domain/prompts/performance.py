"""
Prompt Performance Tracking - 프롬프트 성능 추적
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .versioning import PromptVersionManager


@dataclass
class PerformanceRecord:
    """성능 기록"""

    timestamp: datetime
    metric_name: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PromptPerformanceTracker:
    """프롬프트 성능 추적기"""

    def __init__(self, version_manager: PromptVersionManager):
        self.version_manager = version_manager
        self.performance_history: Dict[str, List[PerformanceRecord]] = defaultdict(list)

    def track_performance(
        self,
        prompt_name: str,
        version: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        성능 메트릭 기록

        Args:
            prompt_name: 프롬프트 이름
            version: 버전
            metrics: 메트릭 딕셔너리 {"accuracy": 0.95, "latency": 0.5}
            metadata: 추가 메타데이터
        """
        # 버전 객체에 메트릭 저장
        try:
            prompt_version = self.version_manager.get_version(prompt_name, version)

            for metric_name, value in metrics.items():
                # 버전의 평균 메트릭 업데이트 (이동 평균)
                if metric_name in prompt_version.performance_metrics:
                    # 이동 평균 (가중치 0.9)
                    current_avg = prompt_version.performance_metrics[metric_name]
                    prompt_version.performance_metrics[metric_name] = (
                        current_avg * 0.9 + value * 0.1
                    )
                else:
                    prompt_version.performance_metrics[metric_name] = value

                # 히스토리 기록
                record = PerformanceRecord(
                    timestamp=datetime.now(),
                    metric_name=metric_name,
                    value=value,
                    metadata=metadata or {},
                )
                key = f"{prompt_name}:{version}"
                self.performance_history[key].append(record)

                # 최근 1000개만 유지
                if len(self.performance_history[key]) > 1000:
                    self.performance_history[key] = self.performance_history[key][-1000:]

            # 저장소에 저장 (파일 기반인 경우)
            if self.version_manager.storage_path:
                self.version_manager._save_to_storage()

        except ValueError:
            # 버전이 없으면 무시 (또는 경고)
            pass

    def get_best_version(
        self,
        prompt_name: str,
        metric: str = "accuracy",
        min_samples: int = 10,
    ) -> Optional[str]:
        """
        최고 성능 버전 조회

        Args:
            prompt_name: 프롬프트 이름
            metric: 평가 메트릭
            min_samples: 최소 샘플 수

        Returns:
            최고 성능 버전 번호 (None if insufficient data)
        """
        if prompt_name not in self.version_manager.versions:
            return None

        best_version = None
        best_value = float("-inf")

        for version in self.version_manager.versions[prompt_name]:
            if metric in version.performance_metrics:
                value = version.performance_metrics[metric]
                if value > best_value and version.usage_count >= min_samples:
                    best_value = value
                    best_version = version.version

        return best_version

    def get_performance_history(
        self,
        prompt_name: str,
        version: str,
        metric: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        성능 히스토리 조회

        Args:
            prompt_name: 프롬프트 이름
            version: 버전
            metric: 특정 메트릭만 (None이면 모든 메트릭)

        Returns:
            성능 기록 리스트
        """
        key = f"{prompt_name}:{version}"
        records = self.performance_history.get(key, [])

        if metric:
            records = [r for r in records if r.metric_name == metric]

        return [
            {
                "timestamp": r.timestamp.isoformat(),
                "metric": r.metric_name,
                "value": r.value,
                "metadata": r.metadata,
            }
            for r in records
        ]

    def get_performance_trend(
        self,
        prompt_name: str,
        version: str,
        metric: str,
        window_size: int = 10,
    ) -> Dict[str, Any]:
        """
        성능 추세 분석 (이동 평균)

        Args:
            prompt_name: 프롬프트 이름
            version: 버전
            metric: 메트릭 이름
            window_size: 윈도우 크기

        Returns:
            {
                "trend": "increasing" | "decreasing" | "stable",
                "average": float,
                "recent_average": float,
                "change_percent": float
            }
        """
        history = self.get_performance_history(prompt_name, version, metric)
        values = [r["value"] for r in history]

        if len(values) < window_size:
            return {
                "trend": "insufficient_data",
                "average": sum(values) / len(values) if values else 0.0,
                "recent_average": sum(values) / len(values) if values else 0.0,
                "change_percent": 0.0,
            }

        # 전체 평균
        overall_avg = sum(values) / len(values)

        # 최근 평균
        recent_values = values[-window_size:]
        recent_avg = sum(recent_values) / len(recent_values)

        # 이전 평균
        previous_values = (
            values[-window_size * 2 : -window_size]
            if len(values) >= window_size * 2
            else values[:window_size]
        )
        previous_avg = (
            sum(previous_values) / len(previous_values) if previous_values else overall_avg
        )

        # 추세 판단
        change_percent = (
            ((recent_avg - previous_avg) / previous_avg * 100) if previous_avg > 0 else 0.0
        )

        if change_percent > 5:
            trend = "increasing"
        elif change_percent < -5:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "average": overall_avg,
            "recent_average": recent_avg,
            "change_percent": change_percent,
        }
