"""
Drift Detector - 성능 저하 감지 모듈
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from beanllm.domain.evaluation.unified_models import EvalRecord


class DriftDetector:
    """Drift 감지기"""

    def __init__(self, drift_threshold: float = 0.2):
        """
        Args:
            drift_threshold: Drift 감지 임계값
        """
        self.drift_threshold = drift_threshold

    def detect_drift(
        self,
        records: List[EvalRecord],
    ) -> Optional[Dict[str, Any]]:
        """
        성능 저하 감지

        Args:
            records: 평가 레코드 리스트

        Returns:
            Drift 감지 결과 또는 None
        """
        if len(records) < 10:
            return None

        sorted_records = sorted(records, key=lambda r: r.timestamp)

        # 최근 vs 이전 비교
        midpoint = len(sorted_records) // 2
        old_records = sorted_records[:midpoint]
        new_records = sorted_records[midpoint:]

        old_avg = sum(r.unified_score for r in old_records) / len(old_records)
        new_avg = sum(r.unified_score for r in new_records) / len(new_records)

        drift = old_avg - new_avg

        if drift > self.drift_threshold:
            return {
                "detected": True,
                "old_score": old_avg,
                "new_score": new_avg,
                "drift_magnitude": drift,
                "severity": "high" if drift > 0.3 else "medium",
                "message": (f"성능 저하 감지: {old_avg:.2f} → {new_avg:.2f} (하락폭: {drift:.2f})"),
            }

        return {"detected": False, "old_score": old_avg, "new_score": new_avg}
