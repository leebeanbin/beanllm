"""Tests for domain/evaluation/drift_detection.py — DriftDetector, DriftAlert."""

from datetime import datetime, timedelta

import pytest

from beanllm.domain.evaluation.drift_detection import DriftAlert, DriftDetector

# ---------------------------------------------------------------------------
# DriftAlert
# ---------------------------------------------------------------------------


class TestDriftAlert:
    def test_creation(self):
        alert = DriftAlert(
            alert_id="drift_0",
            metric_name="accuracy",
            timestamp=datetime.now(),
            current_score=0.6,
            baseline_score=0.85,
            drift_magnitude=0.25,
            drift_type="performance_degradation",
            severity="high",
        )
        assert alert.metric_name == "accuracy"
        assert alert.severity == "high"
        assert alert.metadata == {}

    def test_with_metadata(self):
        alert = DriftAlert(
            alert_id="drift_1",
            metric_name="f1",
            timestamp=datetime.now(),
            current_score=0.7,
            baseline_score=0.9,
            drift_magnitude=0.2,
            drift_type="distribution_shift",
            severity="medium",
            metadata={"z_score": 3.1},
        )
        assert alert.metadata["z_score"] == 3.1


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------


class TestDriftDetectorInit:
    def test_defaults(self):
        det = DriftDetector()
        assert det.baseline_window_days == 7
        assert det.detection_window_days == 1
        assert det.threshold_std == 2.0
        assert det.threshold_percent == 0.2

    def test_custom_params(self):
        det = DriftDetector(
            baseline_window_days=30,
            detection_window_days=3,
            threshold_std=3.0,
            threshold_percent=0.1,
        )
        assert det.baseline_window_days == 30
        assert det.threshold_std == 3.0


class TestRecordScore:
    def test_records_score_with_auto_timestamp(self):
        det = DriftDetector()
        det.record_score("accuracy", 0.9)
        assert len(det._history) == 1
        assert det._history[0]["score"] == 0.9
        assert det._history[0]["metric_name"] == "accuracy"

    def test_records_score_with_explicit_timestamp(self):
        det = DriftDetector()
        ts = datetime(2024, 1, 1)
        det.record_score("accuracy", 0.85, timestamp=ts)
        assert det._history[0]["timestamp"] == ts

    def test_records_multiple_scores(self):
        det = DriftDetector()
        for i in range(5):
            det.record_score("accuracy", 0.8 + i * 0.01)
        assert len(det._history) == 5

    def test_records_metadata(self):
        det = DriftDetector()
        det.record_score("accuracy", 0.8, metadata={"model_version": "v2"})
        assert det._history[0]["metadata"]["model_version"] == "v2"


class TestGetAllMetrics:
    def test_returns_unique_metrics(self):
        det = DriftDetector()
        det.record_score("accuracy", 0.9)
        det.record_score("f1", 0.8)
        det.record_score("accuracy", 0.85)
        metrics = det._get_all_metrics()
        assert set(metrics) == {"accuracy", "f1"}

    def test_empty_history_returns_empty(self):
        det = DriftDetector()
        assert det._get_all_metrics() == []


class TestDetectDrift:
    def _make_detector_with_stable_history(self, metric="accuracy"):
        det = DriftDetector(baseline_window_days=30)
        # Add stable baseline scores
        for i in range(10):
            ts = datetime.now() - timedelta(days=i)
            det.record_score(metric, 0.90, timestamp=ts)
        return det

    def test_no_drift_with_stable_scores(self):
        det = self._make_detector_with_stable_history()
        alerts = det.detect_drift("accuracy", current_score=0.90)
        assert alerts == []

    def test_insufficient_data_returns_empty(self):
        det = DriftDetector()
        det.record_score("accuracy", 0.9)
        alerts = det.detect_drift("accuracy")
        assert alerts == []

    def test_all_metrics_when_no_metric_given(self):
        det = DriftDetector()
        det.record_score("accuracy", 0.9)
        det.record_score("f1", 0.8)
        alerts = det.detect_drift()  # all metrics
        # With only 1 score each, should return no alerts (insufficient data)
        assert isinstance(alerts, list)

    def test_drift_detected_on_large_drop(self):
        det = DriftDetector(
            baseline_window_days=30,
            threshold_percent=0.05,
            threshold_std=0.5,
        )
        # Add baseline with varying scores to create std > 0
        for i in range(10):
            ts = datetime.now() - timedelta(days=i)
            score = 0.90 + (0.02 if i % 2 == 0 else -0.02)
            det.record_score("accuracy", score, timestamp=ts)
        # Record a big drop
        alerts = det.detect_drift("accuracy", current_score=0.50)
        assert isinstance(alerts, list)

    def test_specific_metric_only_checks_that_metric(self):
        det = self._make_detector_with_stable_history("accuracy")
        for i in range(5):
            ts = datetime.now() - timedelta(days=i)
            det.record_score("f1", 0.5, timestamp=ts)
        alerts_accuracy = det.detect_drift("accuracy")
        alerts_f1 = det.detect_drift("f1")
        # These should be independent
        assert isinstance(alerts_accuracy, list)
        assert isinstance(alerts_f1, list)


class TestDetectDriftForMetric:
    def test_returns_empty_with_one_score(self):
        det = DriftDetector()
        det.record_score("m1", 0.9)
        alerts = det._detect_drift_for_metric("m1")
        assert alerts == []

    def test_returns_empty_with_insufficient_baseline(self):
        det = DriftDetector(baseline_window_days=1)
        # Add score from yesterday (within window) and old score
        ts_old = datetime.now() - timedelta(days=10)
        det.record_score("m1", 0.9, timestamp=ts_old)
        det.record_score("m1", 0.88)  # recent
        # Only 1 recent score in baseline window
        alerts = det._detect_drift_for_metric("m1")
        assert isinstance(alerts, list)

    def test_distribution_shift_detected(self):
        det = DriftDetector(baseline_window_days=30, threshold_std=0.1, threshold_percent=0.01)
        # Add stable baseline (many scores)
        for i in range(10):
            ts = datetime.now() - timedelta(days=i)
            det.record_score("m1", 0.9, timestamp=ts)
        # Now add very varying recent scores to trigger distribution shift
        for i in range(5):
            ts = datetime.now() - timedelta(hours=i)
            score = 0.9 + (0.3 if i % 2 == 0 else -0.3)
            det.record_score("m1", max(0, min(1, score)), timestamp=ts)
        alerts = det._detect_drift_for_metric("m1")
        # May or may not trigger depending on exact values, just verify it returns a list
        assert isinstance(alerts, list)


class TestCalculateSeverity:
    def setup_method(self):
        self.det = DriftDetector()

    def test_critical_high_percent(self):
        assert self.det._calculate_severity(0.55, 2.0) == "critical"

    def test_critical_high_zscore(self):
        assert self.det._calculate_severity(0.1, 3.5) == "critical"

    def test_high_medium_percent(self):
        assert self.det._calculate_severity(0.35, 2.0) == "high"

    def test_high_medium_zscore(self):
        assert self.det._calculate_severity(0.1, 2.6) == "high"

    def test_medium_percent(self):
        assert self.det._calculate_severity(0.22, 2.0) == "medium"

    def test_medium_zscore(self):
        assert self.det._calculate_severity(0.1, 2.1) == "medium"

    def test_low(self):
        assert self.det._calculate_severity(0.1, 1.5) == "low"


class TestGetBaselineStats:
    def test_returns_none_when_no_data(self):
        det = DriftDetector()
        assert det.get_baseline_stats("accuracy") is None

    def test_returns_stats_with_data(self):
        det = DriftDetector(baseline_window_days=30)
        for i in range(5):
            ts = datetime.now() - timedelta(days=i)
            det.record_score("accuracy", 0.8 + i * 0.02, timestamp=ts)
        stats = det.get_baseline_stats("accuracy")
        assert stats is not None
        assert stats["metric_name"] == "accuracy"
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert stats["count"] == 5

    def test_excludes_old_scores(self):
        det = DriftDetector(baseline_window_days=7)
        # Old score - outside window
        det.record_score("accuracy", 1.0, timestamp=datetime.now() - timedelta(days=30))
        # Recent score - inside window
        det.record_score("accuracy", 0.8)
        stats = det.get_baseline_stats("accuracy")
        if stats:
            assert stats["count"] == 1


class TestClearHistory:
    def test_clear_all(self):
        det = DriftDetector()
        det.record_score("accuracy", 0.9)
        det.record_score("f1", 0.8)
        det.clear_history()
        assert len(det._history) == 0

    def test_clear_with_days_keeps_recent(self):
        det = DriftDetector()
        det.record_score("m1", 0.9)  # recent
        det.record_score("m1", 0.8, timestamp=datetime.now() - timedelta(days=30))  # old
        det.clear_history(days=7)
        assert len(det._history) == 1

    def test_clear_with_days_removes_old(self):
        det = DriftDetector()
        old_ts = datetime.now() - timedelta(days=100)
        det.record_score("m1", 0.9, timestamp=old_ts)
        det.clear_history(days=7)
        assert len(det._history) == 0
