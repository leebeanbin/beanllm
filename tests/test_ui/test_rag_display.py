"""
Tests for beanllm.ui.repl.rag_display
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

try:
    from beanllm.ui.repl.rag_display import (
        display_chunk_validation,
        display_embedding_analysis,
        display_full_analysis_summary,
        display_quality_assessment,
        display_session_info,
        display_tuning_results,
    )

    AVAILABLE = True
except ImportError:
    AVAILABLE = False


def _make_console():
    return MagicMock()


@pytest.mark.skipif(not AVAILABLE, reason="beanllm.ui.repl.rag_display not available")
class TestDisplaySessionInfo:
    """Tests for display_session_info."""

    def test_basic(self):
        console = _make_console()
        response = MagicMock()
        response.session_name = "Test Session"
        response.session_id = "abc123456789xyz"
        response.num_documents = 100
        response.num_embeddings = 100
        response.embedding_dim = 1536
        response.created_at = "2024-01-01 10:00:00"
        display_session_info(console, response)
        assert console.print.call_count >= 3

    def test_none_session_name(self):
        """session_name=None triggers 'Unnamed' in title."""
        console = _make_console()
        response = MagicMock()
        response.session_name = None
        response.session_id = "x" * 20
        response.num_documents = 0
        response.num_embeddings = 0
        response.embedding_dim = 0
        response.created_at = "N/A"
        display_session_info(console, response)
        assert console.print.call_count >= 3

    def test_large_numbers(self):
        console = _make_console()
        response = MagicMock()
        response.session_name = "Big Session"
        response.session_id = "1234567890abcdef"
        response.num_documents = 1_000_000
        response.num_embeddings = 1_000_000
        response.embedding_dim = 3072
        response.created_at = "2025-12-31 23:59:59"
        display_session_info(console, response)
        assert console.print.call_count >= 3


@pytest.mark.skipif(not AVAILABLE, reason="beanllm.ui.repl.rag_display not available")
class TestDisplayEmbeddingAnalysis:
    """Tests for display_embedding_analysis."""

    def test_with_silhouette_score(self):
        console = _make_console()
        response = MagicMock()
        response.method = "umap"
        response.num_clusters = 3
        response.outliers = [1, 2, 3]
        response.silhouette_score = 0.75
        response.cluster_sizes = {0: 100, 1: 80, 2: 50}
        display_embedding_analysis(console, response)
        assert console.print.call_count >= 1

    def test_without_silhouette_score(self):
        """silhouette_score=None or 0 skips quality assessment."""
        console = _make_console()
        response = MagicMock()
        response.method = "pca"
        response.num_clusters = 2
        response.outliers = []
        response.silhouette_score = None
        response.cluster_sizes = {0: 50, 1: 50}
        display_embedding_analysis(console, response)
        assert console.print.call_count >= 1

    def test_zero_silhouette_score(self):
        """silhouette_score=0.0 is falsy → skips quality branch."""
        console = _make_console()
        response = MagicMock()
        response.method = "tsne"
        response.num_clusters = 1
        response.outliers = []
        response.silhouette_score = 0.0
        response.cluster_sizes = {0: 200}
        display_embedding_analysis(console, response)
        assert console.print.call_count >= 1

    def test_empty_cluster_sizes(self):
        console = _make_console()
        response = MagicMock()
        response.method = "umap"
        response.num_clusters = 0
        response.outliers = []
        response.silhouette_score = None
        response.cluster_sizes = {}
        display_embedding_analysis(console, response)
        assert console.print.call_count >= 1


@pytest.mark.skipif(not AVAILABLE, reason="beanllm.ui.repl.rag_display not available")
class TestDisplayQualityAssessment:
    """Tests for display_quality_assessment (standalone)."""

    def test_excellent(self):
        console = _make_console()
        display_quality_assessment(console, 0.85)
        assert console.print.call_count >= 1

    def test_good(self):
        console = _make_console()
        display_quality_assessment(console, 0.60)
        assert console.print.call_count >= 1

    def test_fair(self):
        console = _make_console()
        display_quality_assessment(console, 0.40)
        assert console.print.call_count >= 1

    def test_poor(self):
        console = _make_console()
        display_quality_assessment(console, 0.10)
        assert console.print.call_count >= 1

    def test_boundary_excellent(self):
        console = _make_console()
        display_quality_assessment(console, 0.71)  # just above 0.7
        assert console.print.call_count >= 1

    def test_boundary_good(self):
        console = _make_console()
        display_quality_assessment(console, 0.51)  # just above 0.5
        assert console.print.call_count >= 1

    def test_boundary_fair(self):
        console = _make_console()
        display_quality_assessment(console, 0.26)  # just above 0.25
        assert console.print.call_count >= 1


@pytest.mark.skipif(not AVAILABLE, reason="beanllm.ui.repl.rag_display not available")
class TestDisplayChunkValidation:
    """Tests for display_chunk_validation."""

    def test_no_issues_no_recommendations(self):
        console = _make_console()
        response = MagicMock()
        response.total_chunks = 500
        response.valid_chunks = 500
        response.issues = []
        response.duplicate_chunks = []
        response.recommendations = []
        display_chunk_validation(console, response)
        assert console.print.call_count >= 1

    def test_with_few_issues(self):
        console = _make_console()
        response = MagicMock()
        response.total_chunks = 100
        response.valid_chunks = 95
        response.issues = ["Issue A", "Issue B", "Issue C"]
        response.duplicate_chunks = ["chunk_1"]
        response.recommendations = ["Rec 1"]
        display_chunk_validation(console, response)
        assert console.print.call_count >= 1

    def test_with_many_issues_truncated(self):
        """More than 10 issues triggers the '+N more' message."""
        console = _make_console()
        response = MagicMock()
        response.total_chunks = 200
        response.valid_chunks = 180
        response.issues = [f"Issue {i}" for i in range(15)]
        response.duplicate_chunks = []
        response.recommendations = []
        display_chunk_validation(console, response)
        assert console.print.call_count >= 1

    def test_with_recommendations(self):
        console = _make_console()
        response = MagicMock()
        response.total_chunks = 100
        response.valid_chunks = 90
        response.issues = ["Problem"]
        response.duplicate_chunks = []
        response.recommendations = ["Fix A", "Fix B"]
        display_chunk_validation(console, response)
        assert console.print.call_count >= 1


@pytest.mark.skipif(not AVAILABLE, reason="beanllm.ui.repl.rag_display not available")
class TestDisplayTuningResults:
    """Tests for display_tuning_results."""

    def test_basic_no_comparison(self):
        console = _make_console()
        response = MagicMock()
        response.parameters = {"k": 5, "chunk_size": 512}
        response.avg_score = 0.8234
        response.comparison_with_baseline = None
        response.recommendations = []
        display_tuning_results(console, response)
        assert console.print.call_count >= 1

    def test_with_positive_improvement(self):
        """Improvement > 5% → green branch."""
        console = _make_console()
        response = MagicMock()
        response.parameters = {"k": 10}
        response.avg_score = 0.9000
        response.comparison_with_baseline = {"improvement_pct": 10.5}
        response.recommendations = ["Use bigger k"]
        display_tuning_results(console, response)
        assert console.print.call_count >= 1

    def test_with_negative_improvement(self):
        """Improvement < -5% → red branch."""
        console = _make_console()
        response = MagicMock()
        response.parameters = {"k": 3}
        response.avg_score = 0.5000
        response.comparison_with_baseline = {"improvement_pct": -8.0}
        response.recommendations = []
        display_tuning_results(console, response)
        assert console.print.call_count >= 1

    def test_with_neutral_improvement(self):
        """Improvement between -5% and 5% → yellow branch."""
        console = _make_console()
        response = MagicMock()
        response.parameters = {"k": 7}
        response.avg_score = 0.7500
        response.comparison_with_baseline = {"improvement_pct": 2.0}
        response.recommendations = []
        display_tuning_results(console, response)
        assert console.print.call_count >= 1

    def test_with_multiple_recommendations(self):
        console = _make_console()
        response = MagicMock()
        response.parameters = {}
        response.avg_score = 0.6
        response.comparison_with_baseline = None
        response.recommendations = ["Rec 1", "Rec 2", "Rec 3"]
        display_tuning_results(console, response)
        assert console.print.call_count >= 1


@pytest.mark.skipif(not AVAILABLE, reason="beanllm.ui.repl.rag_display not available")
class TestDisplayFullAnalysisSummary:
    """Tests for display_full_analysis_summary."""

    def test_all_keys(self):
        console = _make_console()
        results = {
            "embedding_analysis": {"status": "done"},
            "chunk_validation": {"status": "done"},
            "parameter_tuning": {"status": "done"},
        }
        display_full_analysis_summary(console, results)
        assert console.print.call_count >= 1

    def test_only_embedding(self):
        console = _make_console()
        results = {"embedding_analysis": {"status": "done"}}
        display_full_analysis_summary(console, results)
        assert console.print.call_count >= 1

    def test_empty_results(self):
        console = _make_console()
        display_full_analysis_summary(console, {})
        assert console.print.call_count >= 1

    def test_partial_keys(self):
        console = _make_console()
        results = {"chunk_validation": {}, "parameter_tuning": {}}
        display_full_analysis_summary(console, results)
        assert console.print.call_count >= 1
