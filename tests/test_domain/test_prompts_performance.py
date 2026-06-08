"""Tests for domain/prompts/performance.py."""

import pytest

from beanllm.domain.prompts.performance import PerformanceRecord, PromptPerformanceTracker
from beanllm.domain.prompts.versioning import PromptVersionManager


def make_tracker() -> tuple[PromptPerformanceTracker, PromptVersionManager]:
    """Create a tracker with an in-memory version manager."""
    vm = PromptVersionManager(storage_path=None)
    tracker = PromptPerformanceTracker(version_manager=vm)
    return tracker, vm


class TestTrackPerformance:
    def test_track_new_metric_stores_value(self):
        tracker, vm = make_tracker()
        vm.create_version("chat", "Hello {name}", version="v1")

        tracker.track_performance("chat", "v1", {"accuracy": 0.9})

        v = vm.get_version("chat", "v1")
        assert "accuracy" in v.performance_metrics
        assert v.performance_metrics["accuracy"] == 0.9

    def test_track_existing_metric_applies_moving_average(self):
        tracker, vm = make_tracker()
        vm.create_version("chat", "Hello {name}", version="v1")

        tracker.track_performance("chat", "v1", {"accuracy": 0.9})
        tracker.track_performance("chat", "v1", {"accuracy": 0.1})

        v = vm.get_version("chat", "v1")
        # Moving average: 0.9 * 0.9 + 0.1 * 0.1 = 0.81 + 0.01 = 0.82
        assert abs(v.performance_metrics["accuracy"] - 0.82) < 0.001

    def test_track_stores_record_in_history(self):
        tracker, vm = make_tracker()
        vm.create_version("q&a", "Answer: {text}", version="v1")

        tracker.track_performance("q&a", "v1", {"latency": 0.3})

        key = "q&a:v1"
        assert key in tracker.performance_history
        records = tracker.performance_history[key]
        assert len(records) == 1
        assert records[0].metric_name == "latency"
        assert records[0].value == 0.3

    def test_track_multiple_metrics_at_once(self):
        tracker, vm = make_tracker()
        vm.create_version("multi", "text", version="v1")

        tracker.track_performance("multi", "v1", {"accuracy": 0.8, "latency": 0.5})

        v = vm.get_version("multi", "v1")
        assert "accuracy" in v.performance_metrics
        assert "latency" in v.performance_metrics

    def test_track_with_metadata(self):
        tracker, vm = make_tracker()
        vm.create_version("meta", "text", version="v1")

        tracker.track_performance("meta", "v1", {"score": 0.7}, metadata={"env": "prod"})

        records = tracker.performance_history["meta:v1"]
        assert records[0].metadata == {"env": "prod"}

    def test_track_nonexistent_version_silently_ignored(self):
        tracker, vm = make_tracker()
        # Should not raise
        tracker.track_performance("nonexistent", "v99", {"metric": 0.5})

    def test_track_prunes_history_over_1000(self):
        tracker, vm = make_tracker()
        vm.create_version("prune", "text", version="v1")

        for _ in range(1001):
            tracker.track_performance("prune", "v1", {"x": 0.5})

        assert len(tracker.performance_history["prune:v1"]) == 1000

    def test_track_with_storage_path_calls_save(self, tmp_path):
        import json

        storage_file = tmp_path / "prompts.json"
        storage_file.write_text("{}")

        vm = PromptVersionManager(storage_path=str(storage_file))
        tracker = PromptPerformanceTracker(version_manager=vm)
        vm.create_version("saved", "template", version="v1")

        tracker.track_performance("saved", "v1", {"accuracy": 0.95})
        # Should not raise (calls _save_to_storage)


class TestGetBestVersion:
    def test_no_prompt_returns_none(self):
        tracker, vm = make_tracker()
        result = tracker.get_best_version("nonexistent", "accuracy")
        assert result is None

    def test_returns_version_with_highest_metric(self):
        tracker, vm = make_tracker()
        vm.create_version("pick", "text", version="v1")
        vm.create_version("pick", "text v2", version="v2")

        v1 = vm.get_version("pick", "v1")
        v2 = vm.get_version("pick", "v2")
        v1.performance_metrics["accuracy"] = 0.7
        v1.usage_count = 10
        v2.performance_metrics["accuracy"] = 0.9
        v2.usage_count = 15

        best = tracker.get_best_version("pick", "accuracy", min_samples=10)
        assert best == "v2"

    def test_returns_none_when_no_metric(self):
        tracker, vm = make_tracker()
        vm.create_version("nometic", "text", version="v1")

        result = tracker.get_best_version("nometic", "accuracy")
        assert result is None

    def test_ignores_versions_below_min_samples(self):
        tracker, vm = make_tracker()
        vm.create_version("minsamples", "text", version="v1")
        vm.create_version("minsamples", "text2", version="v2")

        v1 = vm.get_version("minsamples", "v1")
        v2 = vm.get_version("minsamples", "v2")
        v1.performance_metrics["accuracy"] = 0.9
        v1.usage_count = 5  # Below min_samples=10
        v2.performance_metrics["accuracy"] = 0.7
        v2.usage_count = 15

        best = tracker.get_best_version("minsamples", "accuracy", min_samples=10)
        assert best == "v2"


class TestGetPerformanceHistory:
    def test_returns_all_records_for_version(self):
        tracker, vm = make_tracker()
        vm.create_version("hist", "text", version="v1")

        tracker.track_performance("hist", "v1", {"accuracy": 0.8})
        tracker.track_performance("hist", "v1", {"latency": 0.3})

        history = tracker.get_performance_history("hist", "v1")
        assert len(history) == 2

    def test_history_record_has_correct_fields(self):
        tracker, vm = make_tracker()
        vm.create_version("fields", "text", version="v1")
        tracker.track_performance("fields", "v1", {"score": 0.77})

        history = tracker.get_performance_history("fields", "v1")
        record = history[0]
        assert "timestamp" in record
        assert record["metric"] == "score"
        assert record["value"] == 0.77
        assert "metadata" in record

    def test_filter_by_metric(self):
        tracker, vm = make_tracker()
        vm.create_version("filter", "text", version="v1")

        tracker.track_performance("filter", "v1", {"accuracy": 0.8, "latency": 0.5})
        tracker.track_performance("filter", "v1", {"accuracy": 0.9})

        history = tracker.get_performance_history("filter", "v1", metric="accuracy")
        assert len(history) == 2
        assert all(r["metric"] == "accuracy" for r in history)

    def test_empty_history_returns_empty_list(self):
        tracker, vm = make_tracker()
        history = tracker.get_performance_history("none", "v1")
        assert history == []

    def test_no_metric_filter_returns_all(self):
        tracker, vm = make_tracker()
        vm.create_version("nofilter", "text", version="v1")
        tracker.track_performance("nofilter", "v1", {"a": 1.0, "b": 2.0})

        history = tracker.get_performance_history("nofilter", "v1", metric=None)
        assert len(history) == 2


class TestGetPerformanceTrend:
    def _fill_history(self, tracker, vm, name, version, values):
        """Helper to manually fill history records."""
        from datetime import datetime

        key = f"{name}:{version}"
        for v in values:
            tracker.performance_history[key].append(
                PerformanceRecord(timestamp=datetime.now(), metric_name="accuracy", value=v)
            )

    def test_insufficient_data_returns_insufficient_data(self):
        tracker, vm = make_tracker()
        vm.create_version("insuf", "text", version="v1")
        tracker.track_performance("insuf", "v1", {"accuracy": 0.5})

        result = tracker.get_performance_trend("insuf", "v1", "accuracy", window_size=10)
        assert result["trend"] == "insufficient_data"

    def test_insufficient_data_with_no_values(self):
        tracker, vm = make_tracker()
        result = tracker.get_performance_trend("empty", "v1", "accuracy", window_size=5)
        assert result["trend"] == "insufficient_data"
        assert result["average"] == 0.0

    def test_increasing_trend(self):
        tracker, vm = make_tracker()
        vm.create_version("inc", "text", version="v1")
        # Old values: low, recent values: high
        self._fill_history(tracker, vm, "inc", "v1", [0.2] * 10 + [0.9] * 10)

        result = tracker.get_performance_trend("inc", "v1", "accuracy", window_size=10)
        assert result["trend"] == "increasing"
        assert result["change_percent"] > 5

    def test_decreasing_trend(self):
        tracker, vm = make_tracker()
        vm.create_version("dec", "text", version="v1")
        # Old values: high, recent values: low
        self._fill_history(tracker, vm, "dec", "v1", [0.9] * 10 + [0.1] * 10)

        result = tracker.get_performance_trend("dec", "v1", "accuracy", window_size=10)
        assert result["trend"] == "decreasing"

    def test_stable_trend(self):
        tracker, vm = make_tracker()
        vm.create_version("stable", "text", version="v1")
        self._fill_history(tracker, vm, "stable", "v1", [0.7] * 20)

        result = tracker.get_performance_trend("stable", "v1", "accuracy", window_size=10)
        assert result["trend"] == "stable"
        assert result["change_percent"] == pytest.approx(0.0, abs=0.01)

    def test_trend_result_has_all_keys(self):
        tracker, vm = make_tracker()
        vm.create_version("keys", "text", version="v1")
        self._fill_history(tracker, vm, "keys", "v1", [0.5] * 20)

        result = tracker.get_performance_trend("keys", "v1", "accuracy", window_size=10)
        assert "trend" in result
        assert "average" in result
        assert "recent_average" in result
        assert "change_percent" in result

    def test_trend_with_not_enough_for_both_windows(self):
        tracker, vm = make_tracker()
        vm.create_version("half", "text", version="v1")
        # Fill 12 values — enough for window_size=10 but not for 2×window
        self._fill_history(tracker, vm, "half", "v1", [0.5] * 12)

        result = tracker.get_performance_trend("half", "v1", "accuracy", window_size=10)
        assert "trend" in result
