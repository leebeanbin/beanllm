"""
Tests for beanllm.utils.core.evaluation_dashboard
Goal: maximize line coverage for EvaluationDashboard

Strategy:
  - plotly and matplotlib are likely not installed in test env
  - We patch sys.modules to inject mocks, then reload the module
  - We also test init branches and error paths with AVAILABLE flags patched
"""

from __future__ import annotations

import importlib
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

try:
    from beanllm.utils.core.evaluation_dashboard import EvaluationDashboard

    AVAILABLE = True
except ImportError:
    AVAILABLE = False

pytestmark = pytest.mark.skipif(not AVAILABLE, reason="evaluation_dashboard not available")


# ---------------------------------------------------------------------------
# Helpers / sample data
# ---------------------------------------------------------------------------

SAMPLE_RESULTS: List[Dict[str, Any]] = [
    {"metric": "precision", "score": 0.80},
    {"metric": "precision", "score": 0.90},
    {"metric": "recall", "score": 0.70},
    {"metric": "f1", "score": 0.75},
]

SAMPLE_TIME_SERIES: List[Dict[str, Any]] = [
    {"metric": "precision", "timestamp": "2024-01-01", "score": 0.80},
    {"metric": "precision", "timestamp": "2024-01-02", "score": 0.85},
    {"metric": "recall", "timestamp": "2024-01-01", "score": 0.70},
    {"metric": "recall", "timestamp": "2024-01-02", "score": 0.75},
]

SAMPLE_MATRIX: Dict[str, Dict[str, float]] = {
    "precision": {"case1": 0.80, "case2": 0.90},
    "recall": {"case1": 0.70, "case2": 0.65},
}


def _make_fig_mock() -> MagicMock:
    fig = MagicMock()
    fig.update_layout = MagicMock()
    fig.write_html = MagicMock()
    fig.add_trace = MagicMock()
    return fig


def _make_go_mock(fig_mock=None):
    if fig_mock is None:
        fig_mock = _make_fig_mock()
    go = MagicMock()
    go.Figure.return_value = fig_mock
    go.Bar.return_value = MagicMock()
    go.Scatter.return_value = MagicMock()
    go.Heatmap.return_value = MagicMock()
    go.Scatter3d = MagicMock()
    return go


def _make_matplotlib_mocks():
    ax = MagicMock()
    fig = MagicMock()

    bar_item = MagicMock()
    bar_item.get_height.return_value = 0.5
    bar_item.get_x.return_value = 0.0
    bar_item.get_width.return_value = 1.0
    ax.bar.return_value = [bar_item]
    ax.text = MagicMock()
    ax.imshow = MagicMock(return_value=MagicMock())

    plt = MagicMock()
    plt.subplots.return_value = (fig, ax)
    plt.figure.return_value = fig
    plt.xticks = MagicMock()
    plt.tight_layout = MagicMock()
    plt.show = MagicMock()
    plt.colorbar = MagicMock()

    return plt, fig, ax


# ===========================================================================
# Constructor / __init__ tests
# ===========================================================================


class TestEvaluationDashboardInit:
    def test_init_plotly_preferred_when_available(self):
        import beanllm.utils.core.evaluation_dashboard as mod

        with (
            patch.object(mod, "PLOTLY_AVAILABLE", True),
            patch.object(mod, "MATPLOTLIB_AVAILABLE", True),
        ):
            d = EvaluationDashboard(use_plotly=True)
            assert d.use_plotly is True

    def test_init_matplotlib_when_plotly_not_wanted(self):
        import beanllm.utils.core.evaluation_dashboard as mod

        with (
            patch.object(mod, "PLOTLY_AVAILABLE", True),
            patch.object(mod, "MATPLOTLIB_AVAILABLE", True),
        ):
            d = EvaluationDashboard(use_plotly=False)
            assert d.use_plotly is False

    def test_init_falls_back_to_matplotlib_when_plotly_unavailable(self):
        import beanllm.utils.core.evaluation_dashboard as mod

        with (
            patch.object(mod, "PLOTLY_AVAILABLE", False),
            patch.object(mod, "MATPLOTLIB_AVAILABLE", True),
        ):
            d = EvaluationDashboard(use_plotly=True)
            assert d.use_plotly is False

    def test_init_raises_when_neither_backend_available(self):
        import beanllm.utils.core.evaluation_dashboard as mod

        with (
            patch.object(mod, "PLOTLY_AVAILABLE", False),
            patch.object(mod, "MATPLOTLIB_AVAILABLE", False),
        ):
            with pytest.raises(ImportError, match="Either plotly or matplotlib"):
                EvaluationDashboard(use_plotly=True)

    def test_init_raises_when_plotly_false_and_matplotlib_unavailable(self):
        import beanllm.utils.core.evaluation_dashboard as mod

        with (
            patch.object(mod, "PLOTLY_AVAILABLE", True),
            patch.object(mod, "MATPLOTLIB_AVAILABLE", False),
        ):
            with pytest.raises(ImportError):
                EvaluationDashboard(use_plotly=False)


# ===========================================================================
# Error-path tests for private methods (AVAILABLE=False branches)
# ===========================================================================


class TestPrivateMethodErrorPaths:
    """Test ImportError branches in private methods."""

    def _make_plotly_dash(self) -> EvaluationDashboard:
        d = EvaluationDashboard.__new__(EvaluationDashboard)
        d.use_plotly = True
        return d

    def _make_mpl_dash(self) -> EvaluationDashboard:
        d = EvaluationDashboard.__new__(EvaluationDashboard)
        d.use_plotly = False
        return d

    def test_plotly_comparison_raises_when_plotly_unavailable(self):
        import beanllm.utils.core.evaluation_dashboard as mod

        with patch.object(mod, "PLOTLY_AVAILABLE", False):
            d = self._make_plotly_dash()
            with pytest.raises(ImportError, match="plotly is required"):
                d._create_plotly_comparison(SAMPLE_RESULTS)

    def test_matplotlib_comparison_raises_when_mpl_unavailable(self):
        import beanllm.utils.core.evaluation_dashboard as mod

        with patch.object(mod, "MATPLOTLIB_AVAILABLE", False):
            d = self._make_mpl_dash()
            with pytest.raises(ImportError, match="matplotlib is required"):
                d._create_matplotlib_comparison(SAMPLE_RESULTS)

    def test_plotly_trend_raises_when_plotly_unavailable(self):
        import beanllm.utils.core.evaluation_dashboard as mod

        with patch.object(mod, "PLOTLY_AVAILABLE", False):
            d = self._make_plotly_dash()
            with pytest.raises(ImportError, match="plotly is required"):
                d._create_plotly_trend(SAMPLE_TIME_SERIES)

    def test_matplotlib_trend_raises_when_mpl_unavailable(self):
        import beanllm.utils.core.evaluation_dashboard as mod

        with patch.object(mod, "MATPLOTLIB_AVAILABLE", False):
            d = self._make_mpl_dash()
            with pytest.raises(ImportError, match="matplotlib is required"):
                d._create_matplotlib_trend(SAMPLE_TIME_SERIES)

    def test_plotly_heatmap_raises_when_plotly_unavailable(self):
        import beanllm.utils.core.evaluation_dashboard as mod

        with patch.object(mod, "PLOTLY_AVAILABLE", False):
            d = self._make_plotly_dash()
            with pytest.raises(ImportError, match="plotly is required"):
                d._create_plotly_heatmap(SAMPLE_MATRIX)

    def test_matplotlib_heatmap_raises_when_mpl_unavailable(self):
        import beanllm.utils.core.evaluation_dashboard as mod

        with patch.object(mod, "MATPLOTLIB_AVAILABLE", False):
            d = self._make_mpl_dash()
            with pytest.raises(ImportError, match="matplotlib is required"):
                d._create_matplotlib_heatmap(SAMPLE_MATRIX)


# ===========================================================================
# Plotly path tests (mock plotly via sys.modules)
# ===========================================================================


class TestPlotlyPaths:
    """Test the plotly code paths by injecting mock plotly into sys.modules."""

    def setup_method(self):
        """Inject a fake plotly into sys.modules and reload the dashboard module."""
        self.fig_mock = _make_fig_mock()
        self.go_mock = _make_go_mock(self.fig_mock)

        plotly_mock = MagicMock()
        plotly_go_mock = self.go_mock

        self._orig_plotly = sys.modules.get("plotly")
        self._orig_plotly_go = sys.modules.get("plotly.graph_objects")

        sys.modules["plotly"] = plotly_mock
        sys.modules["plotly.graph_objects"] = plotly_go_mock

    def teardown_method(self):
        if self._orig_plotly is None:
            sys.modules.pop("plotly", None)
        else:
            sys.modules["plotly"] = self._orig_plotly

        if self._orig_plotly_go is None:
            sys.modules.pop("plotly.graph_objects", None)
        else:
            sys.modules["plotly.graph_objects"] = self._orig_plotly_go

    def _reload_and_make_dash(self) -> EvaluationDashboard:
        """Reload the module with mocked plotly and return a plotly-enabled dashboard."""
        import beanllm.utils.core.evaluation_dashboard as mod

        # Patch PLOTLY_AVAILABLE=True and inject the go mock
        mod.PLOTLY_AVAILABLE = True
        # Inject go directly into the module namespace
        mod.go = self.go_mock

        d = EvaluationDashboard.__new__(EvaluationDashboard)
        d.use_plotly = True
        return d

    def test_create_metrics_comparison_no_save(self):
        import beanllm.utils.core.evaluation_dashboard as mod

        mod.PLOTLY_AVAILABLE = True
        mod.go = self.go_mock

        d = EvaluationDashboard.__new__(EvaluationDashboard)
        d.use_plotly = True
        result = d.create_metrics_comparison(SAMPLE_RESULTS)

        assert result is self.fig_mock
        self.fig_mock.write_html.assert_not_called()

    def test_create_metrics_comparison_with_save(self, tmp_path):
        import beanllm.utils.core.evaluation_dashboard as mod

        mod.PLOTLY_AVAILABLE = True
        mod.go = self.go_mock

        save_file = str(tmp_path / "comp.html")
        d = EvaluationDashboard.__new__(EvaluationDashboard)
        d.use_plotly = True
        d.create_metrics_comparison(SAMPLE_RESULTS, save_path=save_file)

        self.fig_mock.write_html.assert_called_once_with(save_file)

    def test_create_metrics_comparison_empty_results(self):
        import beanllm.utils.core.evaluation_dashboard as mod

        mod.PLOTLY_AVAILABLE = True
        mod.go = self.go_mock

        d = EvaluationDashboard.__new__(EvaluationDashboard)
        d.use_plotly = True
        result = d.create_metrics_comparison([])

        assert result is self.fig_mock

    def test_create_metrics_comparison_missing_fields(self):
        """Results without proper keys should use defaults."""
        import beanllm.utils.core.evaluation_dashboard as mod

        mod.PLOTLY_AVAILABLE = True
        mod.go = self.go_mock

        d = EvaluationDashboard.__new__(EvaluationDashboard)
        d.use_plotly = True
        # One result without 'metric', one without 'score'
        result = d.create_metrics_comparison([{"metric": "acc"}, {"score": 0.9}])

        assert result is self.fig_mock

    def test_create_trend_chart_no_filter(self):
        import beanllm.utils.core.evaluation_dashboard as mod

        mod.PLOTLY_AVAILABLE = True
        mod.go = self.go_mock

        d = EvaluationDashboard.__new__(EvaluationDashboard)
        d.use_plotly = True
        result = d.create_trend_chart(SAMPLE_TIME_SERIES)

        assert result is self.fig_mock

    def test_create_trend_chart_with_metric_filter(self):
        import beanllm.utils.core.evaluation_dashboard as mod

        mod.PLOTLY_AVAILABLE = True
        mod.go = self.go_mock

        d = EvaluationDashboard.__new__(EvaluationDashboard)
        d.use_plotly = True
        result = d.create_trend_chart(SAMPLE_TIME_SERIES, metric_name="precision")

        assert result is self.fig_mock
        # Only precision traces should be added
        self.fig_mock.add_trace.assert_called()

    def test_create_trend_chart_with_save(self, tmp_path):
        import beanllm.utils.core.evaluation_dashboard as mod

        mod.PLOTLY_AVAILABLE = True
        mod.go = self.go_mock

        save_file = str(tmp_path / "trend.html")
        d = EvaluationDashboard.__new__(EvaluationDashboard)
        d.use_plotly = True
        d.create_trend_chart(SAMPLE_TIME_SERIES, save_path=save_file)

        self.fig_mock.write_html.assert_called_once_with(save_file)

    def test_create_trend_chart_empty(self):
        import beanllm.utils.core.evaluation_dashboard as mod

        mod.PLOTLY_AVAILABLE = True
        mod.go = self.go_mock

        d = EvaluationDashboard.__new__(EvaluationDashboard)
        d.use_plotly = True
        result = d.create_trend_chart([])

        assert result is self.fig_mock

    def test_create_heatmap_no_save(self):
        import beanllm.utils.core.evaluation_dashboard as mod

        mod.PLOTLY_AVAILABLE = True
        mod.go = self.go_mock

        d = EvaluationDashboard.__new__(EvaluationDashboard)
        d.use_plotly = True
        result = d.create_heatmap(SAMPLE_MATRIX)

        assert result is self.fig_mock

    def test_create_heatmap_with_save(self, tmp_path):
        import beanllm.utils.core.evaluation_dashboard as mod

        mod.PLOTLY_AVAILABLE = True
        mod.go = self.go_mock

        save_file = str(tmp_path / "heatmap.html")
        d = EvaluationDashboard.__new__(EvaluationDashboard)
        d.use_plotly = True
        d.create_heatmap(SAMPLE_MATRIX, save_path=save_file)

        self.fig_mock.write_html.assert_called_once_with(save_file)

    def test_create_heatmap_missing_case_defaults_to_zero(self):
        """Matrix with partial data should still work."""
        import beanllm.utils.core.evaluation_dashboard as mod

        mod.PLOTLY_AVAILABLE = True
        mod.go = self.go_mock

        partial = {
            "precision": {"case1": 0.9, "case2": 0.8},
            "recall": {"case1": 0.7},
        }
        d = EvaluationDashboard.__new__(EvaluationDashboard)
        d.use_plotly = True
        result = d.create_heatmap(partial)

        assert result is self.fig_mock

    def test_create_heatmap_single_entry(self):
        import beanllm.utils.core.evaluation_dashboard as mod

        mod.PLOTLY_AVAILABLE = True
        mod.go = self.go_mock

        d = EvaluationDashboard.__new__(EvaluationDashboard)
        d.use_plotly = True
        result = d.create_heatmap({"accuracy": {"case1": 0.95}})

        assert result is self.fig_mock


# ===========================================================================
# Matplotlib path tests
# ===========================================================================


class TestMatplotlibPaths:
    """Test the matplotlib code paths by injecting mock matplotlib + numpy."""

    def setup_method(self):
        import numpy as np  # real numpy should be available or we skip

        self.np = np

        self.plt_mock, self.fig_mock, self.ax_mock = _make_matplotlib_mocks()
        self._orig_plt = sys.modules.get("matplotlib.pyplot")
        self._orig_mpl = sys.modules.get("matplotlib")
        self._orig_sns = sys.modules.get("seaborn")

        self.mpl_mock = MagicMock()
        sys.modules["matplotlib"] = self.mpl_mock
        sys.modules["matplotlib.pyplot"] = self.plt_mock

    def teardown_method(self):
        for key, val in [
            ("matplotlib", self._orig_mpl),
            ("matplotlib.pyplot", self._orig_plt),
            ("seaborn", self._orig_sns),
        ]:
            if val is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = val

    def _make_mpl_dash(self) -> EvaluationDashboard:
        import beanllm.utils.core.evaluation_dashboard as mod

        mod.MATPLOTLIB_AVAILABLE = True
        mod.PLOTLY_AVAILABLE = False
        mod.plt = self.plt_mock
        mod.np = self.np

        d = EvaluationDashboard.__new__(EvaluationDashboard)
        d.use_plotly = False
        return d

    def test_create_metrics_comparison_no_save(self):
        d = self._make_mpl_dash()
        result = d.create_metrics_comparison(SAMPLE_RESULTS)
        assert result is self.fig_mock
        self.fig_mock.savefig.assert_not_called()

    def test_create_metrics_comparison_with_save(self, tmp_path):
        d = self._make_mpl_dash()
        save_file = str(tmp_path / "cmp.png")
        d.create_metrics_comparison(SAMPLE_RESULTS, save_path=save_file)
        self.fig_mock.savefig.assert_called_once()

    def test_create_metrics_comparison_empty(self):
        d = self._make_mpl_dash()
        result = d.create_metrics_comparison([])
        assert result is self.fig_mock

    def test_create_trend_chart_no_filter(self):
        d = self._make_mpl_dash()
        result = d.create_trend_chart(SAMPLE_TIME_SERIES)
        assert result is self.fig_mock

    def test_create_trend_chart_with_filter(self):
        d = self._make_mpl_dash()
        result = d.create_trend_chart(SAMPLE_TIME_SERIES, metric_name="recall")
        assert result is self.fig_mock

    def test_create_trend_chart_with_save(self, tmp_path):
        d = self._make_mpl_dash()
        save_file = str(tmp_path / "trend.png")
        d.create_trend_chart(SAMPLE_TIME_SERIES, save_path=save_file)
        self.fig_mock.savefig.assert_called_once()

    def test_create_heatmap_with_seaborn(self):
        """When seaborn is available, sns.heatmap should be called."""
        sns_mock = MagicMock()
        sys.modules["seaborn"] = sns_mock

        d = self._make_mpl_dash()
        result = d.create_heatmap(SAMPLE_MATRIX)
        assert result is self.fig_mock

    def test_create_heatmap_without_seaborn_uses_imshow(self):
        """When seaborn is not importable, ax.imshow should be used."""
        # Remove seaborn from sys.modules
        sys.modules.pop("seaborn", None)
        # Simulate ImportError by setting to None
        sys.modules["seaborn"] = None

        d = self._make_mpl_dash()
        result = d.create_heatmap(SAMPLE_MATRIX)
        assert result is self.fig_mock

    def test_create_heatmap_with_save(self, tmp_path):
        sns_mock = MagicMock()
        sys.modules["seaborn"] = sns_mock

        d = self._make_mpl_dash()
        save_file = str(tmp_path / "hm.png")
        d.create_heatmap(SAMPLE_MATRIX, save_path=save_file)
        self.fig_mock.savefig.assert_called_once()

    def test_create_heatmap_partial_matrix(self):
        sns_mock = MagicMock()
        sys.modules["seaborn"] = sns_mock

        partial = {"precision": {"c1": 0.9}, "recall": {"c1": 0.7, "c2": 0.6}}
        d = self._make_mpl_dash()
        result = d.create_heatmap(partial)
        assert result is self.fig_mock


# ===========================================================================
# Public method routing: create_* dispatches to plotly vs matplotlib
# ===========================================================================


class TestPublicMethodRouting:
    def test_create_metrics_comparison_routes_to_plotly(self):
        import beanllm.utils.core.evaluation_dashboard as mod

        mod.PLOTLY_AVAILABLE = True
        fig_mock = _make_fig_mock()
        go_mock = _make_go_mock(fig_mock)
        mod.go = go_mock

        d = EvaluationDashboard.__new__(EvaluationDashboard)
        d.use_plotly = True
        result = d.create_metrics_comparison(SAMPLE_RESULTS)
        assert result is fig_mock

    def test_create_trend_chart_routes_to_plotly(self):
        import beanllm.utils.core.evaluation_dashboard as mod

        mod.PLOTLY_AVAILABLE = True
        fig_mock = _make_fig_mock()
        go_mock = _make_go_mock(fig_mock)
        mod.go = go_mock

        d = EvaluationDashboard.__new__(EvaluationDashboard)
        d.use_plotly = True
        result = d.create_trend_chart(SAMPLE_TIME_SERIES)
        assert result is fig_mock

    def test_create_heatmap_routes_to_plotly(self):
        import beanllm.utils.core.evaluation_dashboard as mod

        mod.PLOTLY_AVAILABLE = True
        fig_mock = _make_fig_mock()
        go_mock = _make_go_mock(fig_mock)
        mod.go = go_mock

        d = EvaluationDashboard.__new__(EvaluationDashboard)
        d.use_plotly = True
        result = d.create_heatmap(SAMPLE_MATRIX)
        assert result is fig_mock
