"""
Tests for beanllm.ui.visualizers.embedding_viz
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

try:
    from beanllm.ui.visualizers.embedding_viz import EmbeddingVisualizer

    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@pytest.mark.skipif(not AVAILABLE, reason="beanllm.ui.visualizers.embedding_viz not available")
class TestEmbeddingVisualizer:
    """Tests for EmbeddingVisualizer class."""

    def setup_method(self):
        self.console = MagicMock()
        self.viz = EmbeddingVisualizer(console=self.console)

    # ------------------------------------------------------------------ #
    # plot_scatter
    # ------------------------------------------------------------------ #

    def test_plot_scatter_basic(self):
        reduced_embeddings = [[0.1, 0.2], [0.5, 0.6], [0.9, 0.1]]
        labels = [0, 0, 1]
        self.viz.plot_scatter(reduced_embeddings, labels)
        assert self.console.print.call_count >= 1

    def test_plot_scatter_empty(self):
        self.viz.plot_scatter([], [])
        self.console.print.assert_called_once()

    def test_plot_scatter_with_outliers(self):
        reduced_embeddings = [[0.1, 0.2], [0.5, 0.6], [0.9, 0.9], [0.3, 0.4]]
        labels = [0, 0, 1, -1]
        outliers = [2]
        self.viz.plot_scatter(reduced_embeddings, labels, outliers=outliers)
        assert self.console.print.call_count >= 1

    def test_plot_scatter_with_noise_label(self):
        """Label -1 represents noise."""
        reduced_embeddings = [[0.1, 0.2], [0.5, 0.6]]
        labels = [-1, -1]
        self.viz.plot_scatter(reduced_embeddings, labels)
        assert self.console.print.call_count >= 1

    def test_plot_scatter_many_clusters(self):
        """More than 10 clusters cycles through markers."""
        reduced_embeddings = [[float(i) * 0.05, float(i) * 0.05] for i in range(15)]
        labels = list(range(15))
        self.viz.plot_scatter(reduced_embeddings, labels)
        assert self.console.print.call_count >= 1

    def test_plot_scatter_1d_embeddings(self):
        """Single-dimension embeddings should not raise."""
        reduced_embeddings = [[0.1], [0.5], [0.9]]
        labels = [0, 1, 2]
        self.viz.plot_scatter(reduced_embeddings, labels)
        assert self.console.print.call_count >= 1

    def test_plot_scatter_identical_points(self):
        """All same coords: x_range == 0 fallback."""
        reduced_embeddings = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
        labels = [0, 0, 0]
        self.viz.plot_scatter(reduced_embeddings, labels)
        assert self.console.print.call_count >= 1

    def test_plot_scatter_custom_size(self):
        reduced_embeddings = [[0.1, 0.2], [0.5, 0.6]]
        labels = [0, 1]
        self.viz.plot_scatter(reduced_embeddings, labels, width=40, height=15, title="Custom Plot")
        assert self.console.print.call_count >= 1

    # ------------------------------------------------------------------ #
    # _show_legend
    # ------------------------------------------------------------------ #

    def test_show_legend_no_outliers(self):
        self.viz._show_legend([0, 0, 1, 1])
        assert self.console.print.call_count >= 1

    def test_show_legend_with_noise_and_outliers(self):
        self.viz._show_legend([0, -1, 1], outliers=[1])
        assert self.console.print.call_count >= 1

    # ------------------------------------------------------------------ #
    # show_cluster_summary
    # ------------------------------------------------------------------ #

    def test_show_cluster_summary_basic(self):
        cluster_sizes = {0: 50, 1: 30, 2: 20}
        self.viz.show_cluster_summary(cluster_sizes)
        assert self.console.print.call_count >= 1

    def test_show_cluster_summary_with_noise(self):
        cluster_sizes = {-1: 5, 0: 50, 1: 30}
        self.viz.show_cluster_summary(cluster_sizes, silhouette_score=0.65)
        assert self.console.print.call_count >= 1

    def test_show_cluster_summary_excellent_silhouette(self):
        cluster_sizes = {0: 100}
        self.viz.show_cluster_summary(cluster_sizes, silhouette_score=0.85)
        assert self.console.print.call_count >= 1

    def test_show_cluster_summary_good_silhouette(self):
        cluster_sizes = {0: 100}
        self.viz.show_cluster_summary(cluster_sizes, silhouette_score=0.6)
        assert self.console.print.call_count >= 1

    def test_show_cluster_summary_fair_silhouette(self):
        cluster_sizes = {0: 100}
        self.viz.show_cluster_summary(cluster_sizes, silhouette_score=0.35)
        assert self.console.print.call_count >= 1

    def test_show_cluster_summary_poor_silhouette(self):
        cluster_sizes = {0: 100}
        self.viz.show_cluster_summary(cluster_sizes, silhouette_score=0.10)
        assert self.console.print.call_count >= 1

    def test_show_cluster_summary_no_silhouette(self):
        cluster_sizes = {0: 100}
        self.viz.show_cluster_summary(cluster_sizes, silhouette_score=None)
        assert self.console.print.call_count >= 1

    def test_show_cluster_summary_empty(self):
        """Zero total points edge case."""
        cluster_sizes = {0: 0}
        self.viz.show_cluster_summary(cluster_sizes)
        assert self.console.print.call_count >= 1

    def test_show_cluster_summary_custom_method(self):
        cluster_sizes = {0: 100}
        self.viz.show_cluster_summary(cluster_sizes, method="PCA")
        assert self.console.print.call_count >= 1

    # ------------------------------------------------------------------ #
    # _show_quality_score
    # ------------------------------------------------------------------ #

    def test_show_quality_score_excellent(self):
        self.viz._show_quality_score(0.80)
        assert self.console.print.call_count >= 1

    def test_show_quality_score_good(self):
        self.viz._show_quality_score(0.60)
        assert self.console.print.call_count >= 1

    def test_show_quality_score_fair(self):
        self.viz._show_quality_score(0.40)
        assert self.console.print.call_count >= 1

    def test_show_quality_score_poor(self):
        self.viz._show_quality_score(0.10)
        assert self.console.print.call_count >= 1

    # ------------------------------------------------------------------ #
    # show_outlier_details
    # ------------------------------------------------------------------ #

    def test_show_outlier_details_normal_ratio(self):
        self.viz.show_outlier_details(outliers=[5, 10], total_points=100)
        assert self.console.print.call_count >= 1

    def test_show_outlier_details_high_ratio(self):
        """More than threshold triggers recommendation block."""
        self.viz.show_outlier_details(outliers=list(range(20)), total_points=100, threshold=0.05)
        assert self.console.print.call_count >= 1

    def test_show_outlier_details_empty_outliers(self):
        self.viz.show_outlier_details(outliers=[], total_points=100)
        assert self.console.print.call_count >= 1

    def test_show_outlier_details_zero_total_points(self):
        """total_points=0 should not raise ZeroDivisionError."""
        self.viz.show_outlier_details(outliers=[], total_points=0)
        assert self.console.print.call_count >= 1

    def test_show_outlier_details_custom_threshold(self):
        self.viz.show_outlier_details(outliers=[1, 2, 3], total_points=10, threshold=0.20)
        assert self.console.print.call_count >= 1

    # ------------------------------------------------------------------ #
    # show_3d_projection
    # ------------------------------------------------------------------ #

    def test_show_3d_projection_basic(self):
        embeddings_3d = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        labels = [0, 0, 1]
        self.viz.show_3d_projection(embeddings_3d, labels)
        assert self.console.print.call_count >= 1

    def test_show_3d_projection_empty(self):
        self.viz.show_3d_projection([], [])
        self.console.print.assert_called_once()

    def test_show_3d_projection_1d_embeddings(self):
        embeddings = [[0.5], [0.6]]
        labels = [0, 1]
        self.viz.show_3d_projection(embeddings, labels)
        assert self.console.print.call_count >= 1

    def test_show_3d_projection_custom_params(self):
        embeddings_3d = [[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]]
        labels = [0, 1]
        self.viz.show_3d_projection(embeddings_3d, labels, width=50, height=15, title="My 3D Plot")
        assert self.console.print.call_count >= 1

    # ------------------------------------------------------------------ #
    # show_distribution_histogram
    # ------------------------------------------------------------------ #

    def test_show_distribution_histogram_basic(self):
        cluster_sizes = {0: 100, 1: 50, 2: 75}
        self.viz.show_distribution_histogram(cluster_sizes)
        assert self.console.print.call_count >= 1

    def test_show_distribution_histogram_empty(self):
        self.viz.show_distribution_histogram({})
        self.console.print.assert_called_once()

    def test_show_distribution_histogram_with_noise(self):
        cluster_sizes = {-1: 10, 0: 90}
        self.viz.show_distribution_histogram(cluster_sizes)
        assert self.console.print.call_count >= 1

    def test_show_distribution_histogram_single_cluster(self):
        self.viz.show_distribution_histogram({0: 100})
        assert self.console.print.call_count >= 1

    def test_show_distribution_histogram_custom_max_width(self):
        cluster_sizes = {0: 100, 1: 50}
        self.viz.show_distribution_histogram(cluster_sizes, max_width=30)
        assert self.console.print.call_count >= 1

    def test_show_distribution_histogram_zero_max_size(self):
        """When max_size == 0, bar_length should be 0 (no ZeroDivisionError)."""
        cluster_sizes = {0: 0}
        self.viz.show_distribution_histogram(cluster_sizes)
        assert self.console.print.call_count >= 1


@pytest.mark.skipif(not AVAILABLE, reason="beanllm.ui.visualizers.embedding_viz not available")
class TestEmbeddingVisualizerDefaultConsole:
    """Tests that EmbeddingVisualizer works without passing a console."""

    def test_default_console_used(self):
        with patch("beanllm.ui.visualizers.embedding_viz.get_console") as mock_gc:
            mock_console = MagicMock()
            mock_gc.return_value = mock_console
            viz = EmbeddingVisualizer()
            assert viz.console is mock_console
