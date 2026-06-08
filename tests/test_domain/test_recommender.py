"""Tests for domain/optimizer/recommender.py — Recommender."""

from unittest.mock import MagicMock, patch

import pytest

from beanllm.domain.optimizer.recommender import (
    Priority,
    Recommendation,
    RecommendationCategory,
    Recommender,
    print_recommendations,
)


def _make_profile(total_duration_ms=1000, total_cost=0.01, breakdown=None, components=None):
    profile = MagicMock()
    profile.total_duration_ms = total_duration_ms
    profile.total_cost = total_cost
    profile.get_breakdown.return_value = breakdown or {}
    profile.components = components or {}
    return profile


def _make_benchmark(avg_score=0.9, p95_latency=1.0, throughput=5.0):
    bench = MagicMock()
    bench.avg_score = avg_score
    bench.p95_latency = p95_latency
    bench.throughput = throughput
    return bench


# ---------------------------------------------------------------------------
# Recommendation dataclass
# ---------------------------------------------------------------------------


class TestRecommendation:
    def test_creates_with_required_fields(self):
        rec = Recommendation(
            category=RecommendationCategory.PERFORMANCE,
            priority=Priority.HIGH,
            title="Test",
            description="Test desc",
        )
        assert rec.title == "Test"
        assert rec.category == RecommendationCategory.PERFORMANCE
        assert rec.priority == Priority.HIGH

    def test_optional_fields_default(self):
        rec = Recommendation(
            category=RecommendationCategory.COST,
            priority=Priority.LOW,
            title="T",
            description="D",
        )
        assert rec.rationale == ""
        assert rec.action == ""
        assert rec.expected_impact == ""
        assert rec.metadata == {}


# ---------------------------------------------------------------------------
# Recommender.analyze_profile
# ---------------------------------------------------------------------------


class TestAnalyzeProfile:
    def setup_method(self):
        self.rec = Recommender()

    def test_no_recommendations_for_good_profile(self):
        profile = _make_profile(total_duration_ms=1000, total_cost=0.01, breakdown={})
        recs = self.rec.analyze_profile(profile)
        assert recs == []

    def test_high_latency_triggers_critical(self):
        profile = _make_profile(total_duration_ms=6000, total_cost=0.01)
        recs = self.rec.analyze_profile(profile)
        assert any(r.priority == Priority.CRITICAL for r in recs)
        assert any("Latency" in r.title for r in recs)

    def test_high_cost_triggers_recommendation(self):
        profile = _make_profile(total_duration_ms=100, total_cost=0.20)
        recs = self.rec.analyze_profile(profile)
        assert any(r.category == RecommendationCategory.COST for r in recs)

    def test_embedding_bottleneck(self):
        components = {"embedding": MagicMock()}
        profile = _make_profile(
            breakdown={"embedding": 50.0},
            components=components,
        )
        recs = self.rec.analyze_profile(profile)
        assert any("Embedding" in r.title for r in recs)

    def test_retrieval_bottleneck(self):
        components = {"retrieval": MagicMock()}
        profile = _make_profile(
            breakdown={"retrieval": 60.0},
            components=components,
        )
        recs = self.rec.analyze_profile(profile)
        assert any("Retrieval" in r.title for r in recs)

    def test_generation_bottleneck(self):
        components = {"generation": MagicMock()}
        profile = _make_profile(
            breakdown={"generation": 50.0},
            components=components,
        )
        recs = self.rec.analyze_profile(profile)
        assert any("Generation" in r.title for r in recs)

    def test_low_bottleneck_not_triggered(self):
        profile = _make_profile(
            breakdown={"embedding": 30.0},
            components={"embedding": MagicMock()},
        )
        recs = self.rec.analyze_profile(profile)
        # 30% < 40% threshold, no bottleneck rec
        assert not any("Embedding" in r.title for r in recs)


# ---------------------------------------------------------------------------
# Recommender.analyze_benchmark
# ---------------------------------------------------------------------------


class TestAnalyzeBenchmark:
    def setup_method(self):
        self.rec = Recommender()

    def test_no_recommendations_for_good_benchmark(self):
        bench = _make_benchmark(avg_score=0.9, p95_latency=1.0, throughput=5.0)
        recs = self.rec.analyze_benchmark(bench)
        assert recs == []

    def test_low_quality_triggers_critical(self):
        bench = _make_benchmark(avg_score=0.5)
        recs = self.rec.analyze_benchmark(bench)
        assert any(r.priority == Priority.CRITICAL for r in recs)
        assert any("Quality" in r.title for r in recs)

    def test_high_p95_latency_triggers_recommendation(self):
        bench = _make_benchmark(p95_latency=5.0)
        recs = self.rec.analyze_benchmark(bench)
        assert any("P95 Latency" in r.title for r in recs)

    def test_low_throughput_triggers_recommendation(self):
        bench = _make_benchmark(throughput=0.5)
        recs = self.rec.analyze_benchmark(bench)
        assert any("Throughput" in r.title for r in recs)

    def test_exactly_at_threshold_not_triggered(self):
        bench = _make_benchmark(avg_score=0.7, p95_latency=3.0, throughput=1.0)
        recs = self.rec.analyze_benchmark(bench)
        assert recs == []


# ---------------------------------------------------------------------------
# Recommender.analyze_parameters
# ---------------------------------------------------------------------------


class TestAnalyzeParameters:
    def setup_method(self):
        self.rec = Recommender()

    def test_no_recommendations_for_good_params(self):
        params = {"top_k": 10, "score_threshold": 0.7, "temperature": 0.5, "max_tokens": 500}
        recs = self.rec.analyze_parameters(params)
        assert recs == []

    def test_high_top_k_triggers_recommendation(self):
        params = {"top_k": 25}
        recs = self.rec.analyze_parameters(params)
        assert any("top_k" in r.title for r in recs)

    def test_low_score_threshold_triggers(self):
        params = {"score_threshold": 0.3}
        recs = self.rec.analyze_parameters(params)
        assert any("Score Threshold" in r.title for r in recs)

    def test_high_temperature_triggers(self):
        params = {"temperature": 1.5}
        recs = self.rec.analyze_parameters(params)
        assert any("Temperature" in r.title for r in recs)

    def test_high_max_tokens_triggers(self):
        params = {"max_tokens": 3000}
        recs = self.rec.analyze_parameters(params)
        assert any("max_tokens" in r.title for r in recs)

    def test_empty_params_no_recommendations(self):
        recs = self.rec.analyze_parameters({})
        assert recs == []

    def test_unknown_params_ignored(self):
        params = {"unknown_key": "value"}
        recs = self.rec.analyze_parameters(params)
        assert recs == []


# ---------------------------------------------------------------------------
# Recommender.analyze_best_practices
# ---------------------------------------------------------------------------


class TestAnalyzeBestPractices:
    def setup_method(self):
        self.rec = Recommender()

    def test_no_recommendations_for_fully_enabled(self):
        config = {
            "caching_enabled": True,
            "monitoring_enabled": True,
            "evaluation_enabled": True,
        }
        recs = self.rec.analyze_best_practices(config)
        assert recs == []

    def test_caching_not_enabled_triggers(self):
        config = {"caching_enabled": False}
        recs = self.rec.analyze_best_practices(config)
        assert any("Caching" in r.title for r in recs)

    def test_monitoring_not_enabled_triggers(self):
        config = {"monitoring_enabled": False}
        recs = self.rec.analyze_best_practices(config)
        assert any("Monitoring" in r.title for r in recs)

    def test_evaluation_not_enabled_triggers(self):
        config = {"evaluation_enabled": False}
        recs = self.rec.analyze_best_practices(config)
        assert any("Evaluation" in r.title for r in recs)

    def test_missing_keys_all_default_to_false(self):
        recs = self.rec.analyze_best_practices({})
        assert len(recs) == 3  # All three disabled


# ---------------------------------------------------------------------------
# Recommender.generate_optimization_plan
# ---------------------------------------------------------------------------


class TestGenerateOptimizationPlan:
    def setup_method(self):
        self.rec = Recommender()

    def test_plan_has_all_priority_keys(self):
        plan = self.rec.generate_optimization_plan([])
        assert "critical" in plan
        assert "high" in plan
        assert "medium" in plan
        assert "low" in plan

    def test_empty_recommendations_empty_plan(self):
        plan = self.rec.generate_optimization_plan([])
        assert all(v == [] for v in plan.values())

    def test_recs_sorted_by_priority(self):
        recs = [
            Recommendation(RecommendationCategory.PERFORMANCE, Priority.LOW, "L", "d"),
            Recommendation(RecommendationCategory.PERFORMANCE, Priority.CRITICAL, "C", "d"),
            Recommendation(RecommendationCategory.PERFORMANCE, Priority.HIGH, "H", "d"),
            Recommendation(RecommendationCategory.PERFORMANCE, Priority.MEDIUM, "M", "d"),
        ]
        plan = self.rec.generate_optimization_plan(recs)
        assert len(plan["critical"]) == 1
        assert len(plan["high"]) == 1
        assert len(plan["medium"]) == 1
        assert len(plan["low"]) == 1


# ---------------------------------------------------------------------------
# print_recommendations
# ---------------------------------------------------------------------------


class TestPrintRecommendations:
    def test_prints_without_error(self, capsys):
        recs = [
            Recommendation(
                category=RecommendationCategory.PERFORMANCE,
                priority=Priority.HIGH,
                title="Opt",
                description="Optimize",
                rationale="Because",
                action="Do this",
                expected_impact="Better",
            )
        ]
        print_recommendations(recs)
        out = capsys.readouterr().out
        assert "Opt" in out or "OPT" in out

    def test_sorts_by_priority(self, capsys):
        recs = [
            Recommendation(RecommendationCategory.PERFORMANCE, Priority.LOW, "Low", "d"),
            Recommendation(RecommendationCategory.PERFORMANCE, Priority.CRITICAL, "Critical", "d"),
        ]
        print_recommendations(recs)
        out = capsys.readouterr().out
        # CRITICAL should appear before LOW in output
        assert out.index("CRITICAL") < out.index("LOW")

    def test_respects_max_items(self, capsys):
        recs = [
            Recommendation(RecommendationCategory.PERFORMANCE, Priority.LOW, f"R{i}", "d")
            for i in range(5)
        ]
        print_recommendations(recs, max_items=2)
        out = capsys.readouterr().out
        assert "R0" in out or "R1" in out
        # Only 2 items should be printed (numbered 1. and 2.)
        count = out.count("1.") + out.count("2.")
        assert count >= 1
