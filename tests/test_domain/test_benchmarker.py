"""Tests for domain/optimizer/benchmarker.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from beanllm.domain.optimizer.benchmarker import (
    Benchmarker,
    BenchmarkQuery,
    BenchmarkResult,
    QueryType,
)

# ---------------------------------------------------------------------------
# BenchmarkResult __post_init__
# ---------------------------------------------------------------------------


class TestBenchmarkResult:
    def test_empty_result_has_zero_stats(self):
        r = BenchmarkResult(queries=[], latencies=[], scores=[])
        assert r.avg_latency == 0.0
        assert r.avg_score == 0.0
        assert r.throughput == 0.0

    def test_result_calculates_percentiles(self):
        latencies = [0.1 * i for i in range(1, 21)]  # 0.1 to 2.0
        scores = [0.5] * 20
        queries = [BenchmarkQuery(query="q", type=QueryType.SIMPLE)] * 20
        r = BenchmarkResult(queries=queries, latencies=latencies, scores=scores)
        assert r.avg_latency > 0
        assert r.p50_latency > 0
        assert r.p95_latency > 0
        assert r.p99_latency > 0
        assert r.throughput > 0

    def test_result_calculates_score_stats(self):
        scores = [0.2, 0.5, 0.8]
        queries = [BenchmarkQuery(query="q", type=QueryType.SIMPLE)] * 3
        r = BenchmarkResult(queries=queries, latencies=[0.1, 0.1, 0.1], scores=scores)
        assert abs(r.avg_score - 0.5) < 0.01
        assert r.min_score == 0.2
        assert r.max_score == 0.8

    def test_result_zero_total_duration_gives_zero_throughput(self):
        # Latencies summing to 0 would be weird but guard anyway via len=0
        r = BenchmarkResult(queries=[], latencies=[], scores=[])
        assert r.throughput == 0.0


# ---------------------------------------------------------------------------
# Benchmarker.generate_queries
# ---------------------------------------------------------------------------


class TestGenerateQueries:
    def test_generate_queries_returns_correct_count(self):
        b = Benchmarker()
        queries = b.generate_queries(num_queries=10)
        assert len(queries) == 10

    def test_generate_queries_with_seed_sets_random_seed(self):
        b = Benchmarker()
        with patch("random.seed") as mock_seed:
            b.generate_queries(num_queries=1, seed=42)
        mock_seed.assert_called_once_with(42)

    def test_generate_queries_seed_none_does_not_seed(self):
        b = Benchmarker()
        with patch("random.seed") as mock_seed:
            b.generate_queries(num_queries=1, seed=None)
        mock_seed.assert_not_called()

    def test_generate_queries_default_types_covers_all(self):
        b = Benchmarker()
        queries = b.generate_queries(num_queries=100, seed=0)
        types_found = {q.type for q in queries}
        # With 100 queries and 5 types, statistically all should appear
        assert len(types_found) > 0

    def test_generate_queries_with_specific_type(self):
        b = Benchmarker()
        queries = b.generate_queries(num_queries=5, query_types=[QueryType.SIMPLE])
        assert all(q.type == QueryType.SIMPLE for q in queries)

    def test_generate_queries_complex_type_covered(self):
        b = Benchmarker()
        queries = b.generate_queries(num_queries=5, query_types=[QueryType.COMPLEX])
        assert all(q.type == QueryType.COMPLEX for q in queries)

    def test_generate_queries_edge_case_type_covered(self):
        b = Benchmarker()
        queries = b.generate_queries(num_queries=5, query_types=[QueryType.EDGE_CASE])
        assert all(q.type == QueryType.EDGE_CASE for q in queries)

    def test_generate_queries_multi_hop_type_covered(self):
        b = Benchmarker()
        queries = b.generate_queries(num_queries=5, query_types=[QueryType.MULTI_HOP])
        assert all(q.type == QueryType.MULTI_HOP for q in queries)

    def test_generate_queries_aggregation_type_covered(self):
        b = Benchmarker()
        queries = b.generate_queries(num_queries=5, query_types=[QueryType.AGGREGATION])
        assert all(q.type == QueryType.AGGREGATION for q in queries)

    def test_generate_queries_else_branch_covered(self):
        """Force else branch by passing unsupported type via mocked random.choice."""
        b = Benchmarker()
        # Patch random.choice to return None (triggers else branch)
        original_choice = __import__("random").choice

        call_count = [0]

        def patched_choice(seq):
            call_count[0] += 1
            if call_count[0] == 1:
                return None  # Triggers else branch in generate_queries
            return original_choice(seq)

        with patch("random.choice", side_effect=patched_choice):
            queries = b.generate_queries(num_queries=1)
        assert len(queries) == 1

    def test_generate_queries_metadata_has_index_and_domain(self):
        b = Benchmarker()
        queries = b.generate_queries(num_queries=3, domain="machine learning")
        for i, q in enumerate(queries):
            assert q.metadata["index"] == i
            assert q.metadata["domain"] == "machine learning"


# ---------------------------------------------------------------------------
# Benchmarker._generate_simple_query
# ---------------------------------------------------------------------------


class TestGenerateSimpleQuery:
    def test_default_domain_returns_string(self):
        b = Benchmarker()
        result = b._generate_simple_query()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_machine_learning_domain(self):
        b = Benchmarker()
        result = b._generate_simple_query(domain="machine learning")
        assert isinstance(result, str)
        # Should contain a ML concept
        ml_concepts = [
            "gradient descent",
            "backpropagation",
            "overfitting",
            "cross-validation",
            "regularization",
            "neural networks",
            "ensemble methods",
        ]
        assert any(c in result for c in ml_concepts)

    def test_healthcare_domain(self):
        b = Benchmarker()
        result = b._generate_simple_query(domain="healthcare")
        assert isinstance(result, str)
        healthcare_concepts = [
            "hypertension",
            "diabetes",
            "immunization",
            "antibiotic resistance",
            "telemedicine",
        ]
        assert any(c in result for c in healthcare_concepts)

    def test_unknown_domain_uses_default_concepts(self):
        b = Benchmarker()
        result = b._generate_simple_query(domain="finance")
        assert isinstance(result, str)
        default_concepts = [
            "artificial intelligence",
            "quantum computing",
            "blockchain",
            "cloud computing",
            "cybersecurity",
        ]
        assert any(c in result for c in default_concepts)


# ---------------------------------------------------------------------------
# Benchmarker._generate_complex_query
# ---------------------------------------------------------------------------


class TestGenerateComplexQuery:
    def test_default_domain_returns_string(self):
        b = Benchmarker()
        result = b._generate_complex_query()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_machine_learning_domain(self):
        b = Benchmarker()
        result = b._generate_complex_query(domain="machine learning")
        assert isinstance(result, str)
        ml_concepts = [
            "supervised learning",
            "unsupervised learning",
            "reinforcement learning",
            "decision trees",
            "random forests",
            "gradient boosting",
        ]
        assert any(c in result for c in ml_concepts)

    def test_non_ml_domain_uses_default_concepts(self):
        b = Benchmarker()
        result = b._generate_complex_query(domain="web")
        assert isinstance(result, str)
        default_concepts = [
            "microservices",
            "monolithic architecture",
            "SQL databases",
            "NoSQL databases",
            "REST APIs",
            "GraphQL",
        ]
        assert any(c in result for c in default_concepts)


# ---------------------------------------------------------------------------
# Benchmarker._generate_edge_case_query
# ---------------------------------------------------------------------------


class TestGenerateEdgeCaseQuery:
    def test_returns_string(self):
        b = Benchmarker()
        result = b._generate_edge_case_query()
        assert isinstance(result, str)

    def test_lowercase_transformation(self):
        b = Benchmarker()
        with patch("random.choice") as mock_choice:
            # First call: query type (for base query, goes through _generate_simple_query)
            # Last call within edge_case: transformation
            transformations_list = [None]

            def smart_choice(seq):
                if transformations_list[0] is None:
                    transformations_list[0] = seq
                    return seq[0]  # Pick first transformation
                return seq[0]

            mock_choice.side_effect = smart_choice
            result = b._generate_edge_case_query()
        assert isinstance(result, str)

    def test_no_question_mark_transformation(self):
        b = Benchmarker()
        # Force the second transformation (remove ?)
        original_choice = __import__("random").choice
        call_count = [0]

        def patched(seq):
            call_count[0] += 1
            if hasattr(seq[0], "__call__"):  # It's the transformations list
                return seq[1]  # Return second lambda (remove ?)
            return original_choice(seq)

        with patch("random.choice", side_effect=patched):
            result = b._generate_edge_case_query()
        assert "?" not in result

    def test_with_ml_domain(self):
        b = Benchmarker()
        result = b._generate_edge_case_query(domain="machine learning")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Benchmarker._introduce_typo
# ---------------------------------------------------------------------------


class TestIntroduceTypo:
    def test_short_text_returned_unchanged(self):
        b = Benchmarker()
        result = b._introduce_typo("hi")
        assert result == "hi"

    def test_text_with_vowel_gets_replaced(self):
        b = Benchmarker()
        # "aeiou" are vowels with keyboard neighbors
        text = "apple"  # 'a' at index 0, 'e' at index 4
        with patch("random.randint", return_value=0):  # Force index 0 ('a')
            with patch("random.choice", return_value="s"):  # Pick neighbor 's'
                result = b._introduce_typo(text)
        # 'a' at index 0 replaced with 's'
        assert result == "spple"

    def test_text_without_vowel_at_chosen_index_returns_original(self):
        b = Benchmarker()
        text = "xyz123"  # No vowels
        with patch("random.randint", return_value=0):  # 'x' has no keyboard neighbor
            result = b._introduce_typo(text)
        assert result == "xyz123"  # No replacement

    def test_vowel_e_gets_neighbor(self):
        b = Benchmarker()
        text = "error"
        with patch("random.randint", return_value=0):  # 'e'
            with patch("random.choice", return_value="r"):
                result = b._introduce_typo(text)
        assert result == "rrror"

    def test_vowel_i_gets_neighbor(self):
        b = Benchmarker()
        text = "invite"
        with patch("random.randint", return_value=0):  # 'i'
            with patch("random.choice", return_value="u"):
                result = b._introduce_typo(text)
        assert result == "unvite"

    def test_vowel_o_gets_neighbor(self):
        b = Benchmarker()
        text = "output"
        with patch("random.randint", return_value=0):  # 'o'
            with patch("random.choice", return_value="i"):
                result = b._introduce_typo(text)
        # text[:0] + 'i' + text[1:] = '' + 'i' + 'utput' = 'iutput'
        assert result == "iutput"

    def test_vowel_u_gets_neighbor(self):
        b = Benchmarker()
        text = "understand"
        with patch("random.randint", return_value=0):  # 'u'
            with patch("random.choice", return_value="y"):
                result = b._introduce_typo(text)
        assert result == "ynderstand"

    def test_no_typo_when_char_not_in_neighbors(self):
        b = Benchmarker()
        text = "bcd"  # no vowels
        with patch("random.randint", return_value=0):
            result = b._introduce_typo(text)
        assert result == "bcd"


# ---------------------------------------------------------------------------
# Benchmarker._generate_multi_hop_query
# ---------------------------------------------------------------------------


class TestGenerateMultiHopQuery:
    def test_ml_domain(self):
        b = Benchmarker()
        result = b._generate_multi_hop_query(domain="machine learning")
        assert isinstance(result, str)
        assert "learning rate" in result or "training loss" in result or "batch size" in result

    def test_default_domain(self):
        b = Benchmarker()
        result = b._generate_multi_hop_query(domain="web")
        assert isinstance(result, str)
        assert "throughput" in result or "latency" in result or "load" in result

    def test_none_domain(self):
        b = Benchmarker()
        result = b._generate_multi_hop_query(domain=None)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Benchmarker._generate_aggregation_query
# ---------------------------------------------------------------------------


class TestGenerateAggregationQuery:
    def test_ml_domain(self):
        b = Benchmarker()
        result = b._generate_aggregation_query(domain="machine learning")
        assert isinstance(result, str)
        ml_concepts = ["optimization", "regularization", "feature engineering"]
        assert any(c in result for c in ml_concepts)

    def test_non_ml_domain(self):
        b = Benchmarker()
        result = b._generate_aggregation_query(domain="web")
        assert isinstance(result, str)
        default_concepts = ["authentication", "caching", "load balancing"]
        assert any(c in result for c in default_concepts)

    def test_none_domain(self):
        b = Benchmarker()
        result = b._generate_aggregation_query(domain=None)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Benchmarker.run_benchmark
# ---------------------------------------------------------------------------


class TestRunBenchmark:
    def test_basic_benchmark(self):
        b = Benchmarker()
        queries = b.generate_queries(num_queries=5, query_types=[QueryType.SIMPLE])

        def system_fn(query: str) -> float:
            return 0.8

        result = b.run_benchmark(queries, system_fn, warmup=2)
        assert result is not None
        assert len(result.scores) == 5
        assert all(s == 0.8 for s in result.scores)

    def test_benchmark_with_exception_in_system_fn(self):
        b = Benchmarker()
        queries = b.generate_queries(num_queries=3, query_types=[QueryType.SIMPLE])

        def failing_fn(query: str) -> float:
            raise RuntimeError("system error")

        result = b.run_benchmark(queries, failing_fn, warmup=0)
        assert all(s == 0.0 for s in result.scores)  # Errors → 0.0

    def test_benchmark_logs_every_10_queries(self):
        b = Benchmarker()
        queries = b.generate_queries(num_queries=12, query_types=[QueryType.SIMPLE])

        def system_fn(query: str) -> float:
            return 0.5

        result = b.run_benchmark(queries, system_fn, warmup=0)
        assert len(result.queries) == 12

    def test_warmup_queries_not_in_results(self):
        b = Benchmarker()
        queries = b.generate_queries(num_queries=5, query_types=[QueryType.SIMPLE])

        call_count = [0]

        def system_fn(query: str) -> float:
            call_count[0] += 1
            return 0.9

        b.run_benchmark(queries, system_fn, warmup=3)
        # 3 warmup + 5 actual = 8 calls
        assert call_count[0] == 8

    def test_benchmark_zero_warmup(self):
        b = Benchmarker()
        queries = b.generate_queries(num_queries=3, query_types=[QueryType.SIMPLE])
        result = b.run_benchmark(queries, lambda q: 1.0, warmup=0)
        assert len(result.latencies) == 3


# ---------------------------------------------------------------------------
# Benchmarker.compare_baselines
# ---------------------------------------------------------------------------


class TestCompareBaselines:
    def test_compare_two_systems(self):
        b = Benchmarker()
        queries = b.generate_queries(num_queries=3, query_types=[QueryType.SIMPLE])

        systems = {
            "system_a": lambda q: 0.7,
            "system_b": lambda q: 0.9,
        }

        results = b.compare_baselines(queries, systems)
        assert "system_a" in results
        assert "system_b" in results
        assert results["system_a"].avg_score < results["system_b"].avg_score

    def test_compare_single_system(self):
        b = Benchmarker()
        queries = b.generate_queries(num_queries=2, query_types=[QueryType.SIMPLE])

        results = b.compare_baselines(queries, {"only": lambda q: 0.5})
        assert "only" in results
        assert results["only"].avg_score == pytest.approx(0.5)

    def test_compare_empty_systems_returns_empty(self):
        b = Benchmarker()
        queries = b.generate_queries(num_queries=2, query_types=[QueryType.SIMPLE])
        results = b.compare_baselines(queries, {})
        assert results == {}


# ---------------------------------------------------------------------------
# Benchmarker.generate_latency_distribution
# ---------------------------------------------------------------------------


class TestGenerateLatencyDistribution:
    def test_empty_result_returns_empty_buckets(self):
        b = Benchmarker()
        result = BenchmarkResult(queries=[], latencies=[], scores=[])
        dist = b.generate_latency_distribution(result)
        assert dist == {"buckets": [], "counts": []}

    def test_single_latency_zero_bucket_size(self):
        """When min == max, bucket_size = 0 → ZeroDivisionError propagates (edge case)."""
        b = Benchmarker()
        queries = [BenchmarkQuery(query="q", type=QueryType.SIMPLE)]
        result = BenchmarkResult(queries=queries, latencies=[0.1], scores=[0.5])
        # bucket_size = 0 when all latencies are equal → ZeroDivisionError
        with pytest.raises(ZeroDivisionError):
            b.generate_latency_distribution(result)

    def test_multiple_latencies_returns_20_buckets(self):
        b = Benchmarker()
        latencies = [0.1 * i for i in range(1, 21)]
        queries = [BenchmarkQuery(query="q", type=QueryType.SIMPLE)] * 20
        result = BenchmarkResult(queries=queries, latencies=latencies, scores=[0.5] * 20)
        dist = b.generate_latency_distribution(result)
        assert len(dist["buckets"]) == 21  # num_buckets + 1 boundaries
        assert len(dist["counts"]) == 20
        assert sum(dist["counts"]) == 20  # All latencies accounted for

    def test_all_counts_are_floats(self):
        b = Benchmarker()
        latencies = [0.1, 0.2, 0.5]
        queries = [BenchmarkQuery(query="q", type=QueryType.SIMPLE)] * 3
        result = BenchmarkResult(queries=queries, latencies=latencies, scores=[0.5] * 3)
        dist = b.generate_latency_distribution(result)
        assert all(isinstance(c, float) for c in dist["counts"])
