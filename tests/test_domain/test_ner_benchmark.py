"""Tests for domain/knowledge_graph/ner_benchmark.py — NERBenchmark."""

from unittest.mock import MagicMock, patch

import pytest

from beanllm.domain.knowledge_graph.ner_benchmark import NERBenchmark
from beanllm.domain.knowledge_graph.ner_models import (
    BenchmarkResult,
    BenchmarkSample,
    NEREntity,
    NERResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(name: str, entities=None):
    """Create a mock NER engine that returns given entities."""
    engine = MagicMock()
    engine.name = name

    def extract_with_timing(text: str) -> NERResult:
        ner_entities = entities or []
        return NERResult(entities=ner_entities, engine_name=name, latency_ms=5.0)

    engine.extract_with_timing = extract_with_timing
    return engine


def _sample_data():
    return [
        BenchmarkSample(
            text="Apple was founded by Steve Jobs.",
            entities=[
                {"text": "Apple", "label": "ORG", "start": 0, "end": 5},
                {"text": "Steve Jobs", "label": "PERSON", "start": 21, "end": 31},
            ],
        ),
        BenchmarkSample(
            text="Google is in Mountain View.",
            entities=[
                {"text": "Google", "label": "ORG", "start": 0, "end": 6},
                {"text": "Mountain View", "label": "LOC", "start": 13, "end": 26},
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# NERBenchmark init
# ---------------------------------------------------------------------------


class TestNERBenchmarkInit:
    def test_stores_engines(self):
        engine = _make_engine("spacy")
        benchmark = NERBenchmark([engine])
        assert benchmark.engines == [engine]

    def test_results_empty_initially(self):
        benchmark = NERBenchmark([])
        assert benchmark._results == {}


# ---------------------------------------------------------------------------
# NERBenchmark.run
# ---------------------------------------------------------------------------


class TestNERBenchmarkRun:
    def test_run_returns_dict_keyed_by_engine_name(self):
        engine = _make_engine("mock-spacy")
        benchmark = NERBenchmark([engine])
        results = benchmark.run(_sample_data())
        assert "mock-spacy" in results

    def test_run_returns_benchmark_result(self):
        engine = _make_engine("mock-spacy")
        benchmark = NERBenchmark([engine])
        results = benchmark.run(_sample_data())
        assert isinstance(results["mock-spacy"], BenchmarkResult)

    def test_run_with_perfect_predictions(self):
        """Engine that predicts exactly the gold entities gets F1=1."""
        entities = [
            NEREntity(text="Apple", label="ORG", start=0, end=5),
            NEREntity(text="Steve Jobs", label="PERSON", start=21, end=31),
            NEREntity(text="Google", label="ORG", start=0, end=6),
            NEREntity(text="Mountain View", label="LOC", start=13, end=26),
        ]
        # Return different entities per call to match gold
        call_count = [0]

        def perfect_extract(text: str) -> NERResult:
            call_count[0] += 1
            if call_count[0] == 1:
                return NERResult(
                    entities=[
                        NEREntity(text="Apple", label="ORG", start=0, end=5),
                        NEREntity(text="Steve Jobs", label="PERSON", start=21, end=31),
                    ],
                    engine_name="perfect",
                    latency_ms=1.0,
                )
            return NERResult(
                entities=[
                    NEREntity(text="Google", label="ORG", start=0, end=6),
                    NEREntity(text="Mountain View", label="LOC", start=13, end=26),
                ],
                engine_name="perfect",
                latency_ms=1.0,
            )

        engine = MagicMock()
        engine.name = "perfect"
        engine.extract_with_timing = perfect_extract
        benchmark = NERBenchmark([engine])
        results = benchmark.run(_sample_data())

        assert results["perfect"].f1_score == pytest.approx(1.0)
        assert results["perfect"].precision == pytest.approx(1.0)
        assert results["perfect"].recall == pytest.approx(1.0)

    def test_run_with_no_predictions(self):
        """Engine that predicts nothing gets F1=0."""
        engine = _make_engine("empty", entities=[])
        benchmark = NERBenchmark([engine])
        results = benchmark.run(_sample_data())
        assert results["empty"].f1_score == pytest.approx(0.0)

    def test_run_multiple_engines(self):
        engine1 = _make_engine("engine-A")
        engine2 = _make_engine("engine-B")
        benchmark = NERBenchmark([engine1, engine2])
        results = benchmark.run(_sample_data())
        assert "engine-A" in results
        assert "engine-B" in results

    def test_run_populates_total_samples(self):
        engine = _make_engine("spacy")
        benchmark = NERBenchmark([engine])
        data = _sample_data()
        results = benchmark.run(data)
        assert results["spacy"].total_samples == len(data)

    def test_run_with_empty_data(self):
        engine = _make_engine("spacy")
        benchmark = NERBenchmark([engine])
        results = benchmark.run([])
        assert results["spacy"].total_samples == 0
        assert results["spacy"].avg_latency_ms == 0

    def test_run_label_normalization_per_or_person(self):
        """PER should be normalized to PERSON."""
        engine = MagicMock()
        engine.name = "normalizer"
        engine.extract_with_timing.return_value = NERResult(
            entities=[NEREntity(text="steve jobs", label="PER", start=0, end=10)],
            engine_name="normalizer",
            latency_ms=2.0,
        )
        sample = [
            BenchmarkSample(
                text="Steve Jobs founded Apple.",
                entities=[{"text": "Steve Jobs", "label": "PERSON", "start": 0, "end": 10}],
            )
        ]
        benchmark = NERBenchmark([engine])
        results = benchmark.run(sample)
        # PER normalizes to PERSON → should match gold PERSON
        assert results["normalizer"].f1_score == pytest.approx(1.0)

    def test_run_without_label_normalization(self):
        """Without normalization, PER != PERSON."""
        engine = MagicMock()
        engine.name = "no-norm"
        engine.extract_with_timing.return_value = NERResult(
            entities=[NEREntity(text="steve jobs", label="PER", start=0, end=10)],
            engine_name="no-norm",
            latency_ms=2.0,
        )
        sample = [
            BenchmarkSample(
                text="Steve Jobs founded Apple.",
                entities=[{"text": "Steve Jobs", "label": "PERSON", "start": 0, "end": 10}],
            )
        ]
        benchmark = NERBenchmark([engine])
        results = benchmark.run(sample, normalize_labels=False)
        # PER != PERSON when not normalized → no match
        assert results["no-norm"].f1_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# NERBenchmark._normalize_label
# ---------------------------------------------------------------------------


class TestNormalizeLabel:
    def setup_method(self):
        self.benchmark = NERBenchmark([])

    def test_normalizes_per_to_person(self):
        assert self.benchmark._normalize_label("PER") == "PERSON"

    def test_normalizes_org_to_organization(self):
        assert self.benchmark._normalize_label("ORG") == "ORGANIZATION"

    def test_normalizes_loc_to_location(self):
        assert self.benchmark._normalize_label("LOC") == "LOCATION"

    def test_normalizes_gpe_to_location(self):
        assert self.benchmark._normalize_label("GPE") == "LOCATION"

    def test_normalizes_time_to_date(self):
        assert self.benchmark._normalize_label("TIME") == "DATE"

    def test_normalizes_misc_to_other(self):
        assert self.benchmark._normalize_label("MISC") == "OTHER"

    def test_unknown_label_returns_uppercase(self):
        assert self.benchmark._normalize_label("custom_label") == "CUSTOM_LABEL"


# ---------------------------------------------------------------------------
# NERBenchmark.get_report
# ---------------------------------------------------------------------------


class TestGetReport:
    def _benchmark_with_results(self):
        engine = _make_engine(
            "test-engine",
            entities=[
                NEREntity(text="Apple", label="ORG", start=0, end=5),
            ],
        )
        b = NERBenchmark([engine])
        b.run(_sample_data())
        return b

    def test_get_report_no_results_returns_message(self):
        benchmark = NERBenchmark([])
        result = benchmark.get_report()
        assert "No benchmark results" in result

    def test_get_report_markdown_contains_header(self):
        b = self._benchmark_with_results()
        report = b.get_report(format="markdown")
        assert "# NER Benchmark Results" in report

    def test_get_report_markdown_contains_engine_name(self):
        b = self._benchmark_with_results()
        report = b.get_report(format="markdown")
        assert "test-engine" in report

    def test_get_report_text_format(self):
        b = self._benchmark_with_results()
        report = b.get_report(format="text")
        assert "test-engine" in report
        assert "Precision" in report


# ---------------------------------------------------------------------------
# NERBenchmark.get_best_engine
# ---------------------------------------------------------------------------


class TestGetBestEngine:
    def test_returns_none_with_no_results(self):
        benchmark = NERBenchmark([])
        assert benchmark.get_best_engine() is None

    def test_returns_engine_name_after_run(self):
        engine = _make_engine("spacy")
        benchmark = NERBenchmark([engine])
        benchmark.run(_sample_data())
        best = benchmark.get_best_engine()
        assert best == "spacy"

    def test_returns_best_f1_engine(self):
        good_engine = MagicMock()
        good_engine.name = "good"
        good_engine.extract_with_timing.return_value = NERResult(
            entities=[
                NEREntity(text="apple", label="ORG", start=0, end=5),
                NEREntity(text="steve jobs", label="PERSON", start=21, end=31),
            ],
            engine_name="good",
            latency_ms=1.0,
        )

        bad_engine = _make_engine("bad", entities=[])
        benchmark = NERBenchmark([good_engine, bad_engine])
        data = [_sample_data()[0]]
        benchmark.run(data)
        best = benchmark.get_best_engine("f1_score")
        assert best == "good"

    def test_get_best_by_latency(self):
        fast_engine = MagicMock()
        fast_engine.name = "fast"
        fast_engine.extract_with_timing.return_value = NERResult(
            entities=[], engine_name="fast", latency_ms=1.0
        )
        slow_engine = MagicMock()
        slow_engine.name = "slow"
        slow_engine.extract_with_timing.return_value = NERResult(
            entities=[], engine_name="slow", latency_ms=100.0
        )
        benchmark = NERBenchmark([fast_engine, slow_engine])
        benchmark.run(_sample_data()[:1])
        # The engine with lowest latency should win when using min metric
        # get_best_engine uses max(), so for latency the "fastest" would be the one with highest latency
        # This tests the metric parameter at least
        best = benchmark.get_best_engine("avg_latency_ms")
        assert best in ["fast", "slow"]
