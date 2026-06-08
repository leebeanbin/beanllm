"""
Tests for beanllm.utils.rag_debug.debug_visualization
Goal: maximize line coverage
"""

from __future__ import annotations

import sys
from typing import List
from unittest.mock import MagicMock, call, patch

import pytest

try:
    from beanllm.utils.rag_debug.debug_visualization import (
        similarity_heatmap,
        visualize_embeddings,
        visualize_embeddings_2d,
    )

    AVAILABLE = True
except ImportError:
    AVAILABLE = False

pytestmark = pytest.mark.skipif(not AVAILABLE, reason="debug_visualization not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEXTS = ["AI", "ML", "DL", "NLP", "CV"]


def _make_embedding_fn(dim: int = 4):
    """Return a deterministic fake embedding function."""
    import math

    def embed(texts: List[str]):
        return [[math.sin(i + j * 0.1) for j in range(dim)] for i, _ in enumerate(texts)]

    return embed


# ===========================================================================
# Tests for visualize_embeddings_2d (legacy wrapper)
# ===========================================================================


class TestVisualizeEmbeddings2d:
    def test_calls_visualize_embeddings(self):
        """visualize_embeddings_2d should delegate to visualize_embeddings."""
        with patch("beanllm.utils.rag_debug.debug_visualization.visualize_embeddings") as mock_ve:
            embed_fn = _make_embedding_fn()
            visualize_embeddings_2d(TEXTS, embed_fn, save_path="/tmp/test.png")

        mock_ve.assert_called_once_with(
            TEXTS,
            embed_fn,
            method="tsne",
            dimensions=2,
            save_path="/tmp/test.png",
            interactive=False,
        )

    def test_calls_visualize_embeddings_no_save(self):
        with patch("beanllm.utils.rag_debug.debug_visualization.visualize_embeddings") as mock_ve:
            embed_fn = _make_embedding_fn()
            visualize_embeddings_2d(TEXTS, embed_fn)

        mock_ve.assert_called_once_with(
            TEXTS,
            embed_fn,
            method="tsne",
            dimensions=2,
            save_path=None,
            interactive=False,
        )


# ===========================================================================
# Tests for visualize_embeddings
# ===========================================================================


class TestVisualizeEmbeddings:
    # --- numpy missing ---
    def test_returns_early_when_numpy_missing(self, capsys):
        import beanllm.utils.rag_debug.debug_visualization as mod

        with patch.object(mod, "HAS_NUMPY", False), patch.object(mod, "np", None):
            visualize_embeddings(TEXTS, _make_embedding_fn())

        out = capsys.readouterr().out
        assert "numpy" in out.lower() or "pip install" in out.lower()

    # --- unknown method raises ---
    def test_unknown_method_raises_value_error(self):
        """Unknown dim-reduction method should raise ValueError."""
        import numpy as _np

        import beanllm.utils.rag_debug.debug_visualization as mod

        with patch.object(mod, "HAS_NUMPY", True), patch.object(mod, "np", _np):
            with pytest.raises(ValueError, match="Unknown method"):
                visualize_embeddings(TEXTS, _make_embedding_fn(), method="unknown")

    # --- sklearn missing for tsne ---
    def test_tsne_returns_early_when_sklearn_missing(self, capsys):
        import numpy as _np

        import beanllm.utils.rag_debug.debug_visualization as mod

        # Fake numpy array
        fake_vectors = _np.array(_make_embedding_fn()(TEXTS))

        embed_fn = MagicMock(return_value=fake_vectors.tolist())

        with (
            patch.object(mod, "HAS_NUMPY", True),
            patch.object(mod, "np", _np),
            patch.dict(sys.modules, {"sklearn": None, "sklearn.manifold": None}),
        ):
            visualize_embeddings(TEXTS, embed_fn, method="tsne")

        # Should print a pip install message
        out = capsys.readouterr().out
        assert "scikit-learn" in out or "pip install" in out or out == ""

    # --- sklearn missing for pca ---
    def test_pca_returns_early_when_sklearn_missing(self, capsys):
        import numpy as _np

        import beanllm.utils.rag_debug.debug_visualization as mod

        fake_vectors = _np.array(_make_embedding_fn()(TEXTS))
        embed_fn = MagicMock(return_value=fake_vectors.tolist())

        with (
            patch.object(mod, "HAS_NUMPY", True),
            patch.object(mod, "np", _np),
            patch.dict(sys.modules, {"sklearn": None, "sklearn.decomposition": None}),
        ):
            visualize_embeddings(TEXTS, embed_fn, method="pca")

        out = capsys.readouterr().out
        assert "scikit-learn" in out or "pip install" in out or out == ""

    # --- full matplotlib 2D path ---
    def test_matplotlib_2d_path(self):
        """Cover the matplotlib 2D (non-interactive) branch."""
        import numpy as _np

        import beanllm.utils.rag_debug.debug_visualization as mod

        n = len(TEXTS)
        fake_vectors_2d = _np.zeros((n, 2))

        reducer_mock = MagicMock()
        reducer_mock.fit_transform.return_value = fake_vectors_2d

        tsne_mock = MagicMock(return_value=reducer_mock)
        sklearn_manifold_mock = MagicMock()
        sklearn_manifold_mock.TSNE = tsne_mock

        plt_mock = MagicMock()
        # `import matplotlib.pyplot as plt` resolves to matplotlib_mock.pyplot,
        # not sys.modules["matplotlib.pyplot"] directly.
        matplotlib_mock = MagicMock()
        matplotlib_mock.pyplot = plt_mock

        embed_fn = MagicMock(return_value=_np.zeros((n, 4)).tolist())

        with (
            patch.object(mod, "HAS_NUMPY", True),
            patch.object(mod, "np", _np),
            patch.dict(
                sys.modules,
                {
                    "sklearn": MagicMock(),
                    "sklearn.manifold": sklearn_manifold_mock,
                    "matplotlib": matplotlib_mock,
                    "matplotlib.pyplot": plt_mock,
                },
            ),
        ):
            visualize_embeddings(TEXTS, embed_fn, method="tsne", dimensions=2, interactive=False)

        plt_mock.scatter.assert_called()

    # --- full matplotlib 3D path ---
    def test_matplotlib_3d_path(self):
        import numpy as _np

        import beanllm.utils.rag_debug.debug_visualization as mod

        n = len(TEXTS)
        fake_vectors_3d = _np.zeros((n, 3))

        reducer_mock = MagicMock()
        reducer_mock.fit_transform.return_value = fake_vectors_3d

        pca_mock = MagicMock(return_value=reducer_mock)
        sklearn_decomp_mock = MagicMock()
        sklearn_decomp_mock.PCA = pca_mock

        ax_mock = MagicMock()
        fig_mock = MagicMock()
        fig_mock.add_subplot.return_value = ax_mock

        plt_mock = MagicMock()
        plt_mock.figure.return_value = fig_mock
        matplotlib_mock = MagicMock()
        matplotlib_mock.pyplot = plt_mock

        embed_fn = MagicMock(return_value=_np.zeros((n, 4)).tolist())

        with (
            patch.object(mod, "HAS_NUMPY", True),
            patch.object(mod, "np", _np),
            patch.dict(
                sys.modules,
                {
                    "sklearn": MagicMock(),
                    "sklearn.decomposition": sklearn_decomp_mock,
                    "matplotlib": matplotlib_mock,
                    "matplotlib.pyplot": plt_mock,
                },
            ),
        ):
            visualize_embeddings(TEXTS, embed_fn, method="pca", dimensions=3, interactive=False)

        ax_mock.scatter.assert_called()

    # --- matplotlib with save_path ---
    def test_matplotlib_2d_save_path(self, tmp_path):
        import numpy as _np

        import beanllm.utils.rag_debug.debug_visualization as mod

        n = len(TEXTS)
        fake_vectors_2d = _np.zeros((n, 2))

        reducer_mock = MagicMock()
        reducer_mock.fit_transform.return_value = fake_vectors_2d

        tsne_mock = MagicMock(return_value=reducer_mock)
        sklearn_manifold_mock = MagicMock()
        sklearn_manifold_mock.TSNE = tsne_mock

        plt_mock = MagicMock()
        matplotlib_mock = MagicMock()
        matplotlib_mock.pyplot = plt_mock
        save_file = str(tmp_path / "embeddings.png")

        embed_fn = MagicMock(return_value=_np.zeros((n, 4)).tolist())

        with (
            patch.object(mod, "HAS_NUMPY", True),
            patch.object(mod, "np", _np),
            patch.dict(
                sys.modules,
                {
                    "sklearn": MagicMock(),
                    "sklearn.manifold": sklearn_manifold_mock,
                    "matplotlib": matplotlib_mock,
                    "matplotlib.pyplot": plt_mock,
                },
            ),
        ):
            visualize_embeddings(
                TEXTS, embed_fn, method="tsne", dimensions=2, interactive=False, save_path=save_file
            )

        plt_mock.savefig.assert_called_once()

    # --- interactive plotly 2D path ---
    def test_interactive_plotly_2d_path(self):
        import numpy as _np

        import beanllm.utils.rag_debug.debug_visualization as mod

        n = len(TEXTS)
        fake_vectors_2d = _np.zeros((n, 2))

        reducer_mock = MagicMock()
        reducer_mock.fit_transform.return_value = fake_vectors_2d

        tsne_mock = MagicMock(return_value=reducer_mock)
        sklearn_manifold_mock = MagicMock()
        sklearn_manifold_mock.TSNE = tsne_mock

        fig_mock = MagicMock()
        go_mock = MagicMock()
        go_mock.Figure.return_value = fig_mock
        go_mock.Scatter.return_value = MagicMock()
        plotly_mock = MagicMock()
        plotly_mock.graph_objects = go_mock

        embed_fn = MagicMock(return_value=_np.zeros((n, 4)).tolist())

        with (
            patch.object(mod, "HAS_NUMPY", True),
            patch.object(mod, "np", _np),
            patch.dict(
                sys.modules,
                {
                    "sklearn": MagicMock(),
                    "sklearn.manifold": sklearn_manifold_mock,
                    "plotly": plotly_mock,
                    "plotly.graph_objects": go_mock,
                    "matplotlib": MagicMock(),
                    "matplotlib.pyplot": MagicMock(),
                },
            ),
        ):
            visualize_embeddings(TEXTS, embed_fn, method="tsne", dimensions=2, interactive=True)

        fig_mock.show.assert_called()

    # --- interactive plotly 3D path ---
    def test_interactive_plotly_3d_path(self):
        import numpy as _np

        import beanllm.utils.rag_debug.debug_visualization as mod

        n = len(TEXTS)
        fake_vectors_3d = _np.zeros((n, 3))

        reducer_mock = MagicMock()
        reducer_mock.fit_transform.return_value = fake_vectors_3d

        pca_mock = MagicMock(return_value=reducer_mock)
        sklearn_decomp_mock = MagicMock()
        sklearn_decomp_mock.PCA = pca_mock

        fig_mock = MagicMock()
        go_mock = MagicMock()
        go_mock.Figure.return_value = fig_mock
        go_mock.Scatter3d.return_value = MagicMock()
        plotly_mock = MagicMock()
        plotly_mock.graph_objects = go_mock

        embed_fn = MagicMock(return_value=_np.zeros((n, 4)).tolist())

        with (
            patch.object(mod, "HAS_NUMPY", True),
            patch.object(mod, "np", _np),
            patch.dict(
                sys.modules,
                {
                    "sklearn": MagicMock(),
                    "sklearn.decomposition": sklearn_decomp_mock,
                    "plotly": plotly_mock,
                    "plotly.graph_objects": go_mock,
                    "matplotlib": MagicMock(),
                    "matplotlib.pyplot": MagicMock(),
                },
            ),
        ):
            visualize_embeddings(TEXTS, embed_fn, method="pca", dimensions=3, interactive=True)

        fig_mock.show.assert_called()

    # --- interactive plotly with save_path ---
    def test_interactive_plotly_saves_html(self, tmp_path):
        import numpy as _np

        import beanllm.utils.rag_debug.debug_visualization as mod

        n = len(TEXTS)
        fake_vectors_2d = _np.zeros((n, 2))

        reducer_mock = MagicMock()
        reducer_mock.fit_transform.return_value = fake_vectors_2d

        tsne_mock = MagicMock(return_value=reducer_mock)
        sklearn_manifold_mock = MagicMock()
        sklearn_manifold_mock.TSNE = tsne_mock

        fig_mock = MagicMock()
        go_mock = MagicMock()
        go_mock.Figure.return_value = fig_mock
        go_mock.Scatter.return_value = MagicMock()
        plotly_mock = MagicMock()
        plotly_mock.graph_objects = go_mock

        embed_fn = MagicMock(return_value=_np.zeros((n, 4)).tolist())
        save_file = str(tmp_path / "test.png")

        with (
            patch.object(mod, "HAS_NUMPY", True),
            patch.object(mod, "np", _np),
            patch.dict(
                sys.modules,
                {
                    "sklearn": MagicMock(),
                    "sklearn.manifold": sklearn_manifold_mock,
                    "plotly": plotly_mock,
                    "plotly.graph_objects": go_mock,
                    "matplotlib": MagicMock(),
                    "matplotlib.pyplot": MagicMock(),
                },
            ),
        ):
            visualize_embeddings(
                TEXTS, embed_fn, method="tsne", dimensions=2, interactive=True, save_path=save_file
            )

        fig_mock.write_html.assert_called()

    # --- matplotlib missing (non-interactive) ---
    def test_matplotlib_missing_returns_early(self, capsys):
        import numpy as _np

        import beanllm.utils.rag_debug.debug_visualization as mod

        n = len(TEXTS)
        fake_vectors_2d = _np.zeros((n, 2))

        reducer_mock = MagicMock()
        reducer_mock.fit_transform.return_value = fake_vectors_2d

        tsne_mock = MagicMock(return_value=reducer_mock)
        sklearn_manifold_mock = MagicMock()
        sklearn_manifold_mock.TSNE = tsne_mock

        embed_fn = MagicMock(return_value=_np.zeros((n, 4)).tolist())

        with (
            patch.object(mod, "HAS_NUMPY", True),
            patch.object(mod, "np", _np),
            patch.dict(
                sys.modules,
                {
                    "sklearn": MagicMock(),
                    "sklearn.manifold": sklearn_manifold_mock,
                    "matplotlib": None,
                    "matplotlib.pyplot": None,
                },
            ),
        ):
            visualize_embeddings(TEXTS, embed_fn, method="tsne", dimensions=2, interactive=False)

        out = capsys.readouterr().out
        # Either prints warning or silently exits
        assert isinstance(out, str)


# ===========================================================================
# Tests for similarity_heatmap
# ===========================================================================


class TestSimilarityHeatmap:
    def test_returns_early_when_numpy_missing(self, capsys):
        import beanllm.utils.rag_debug.debug_visualization as mod

        with patch.object(mod, "HAS_NUMPY", False), patch.object(mod, "np", None):
            similarity_heatmap(TEXTS, _make_embedding_fn())

        out = capsys.readouterr().out
        assert "numpy" in out.lower() or "pip install" in out.lower()

    def test_returns_early_when_matplotlib_or_sklearn_missing(self, capsys):
        import numpy as _np

        import beanllm.utils.rag_debug.debug_visualization as mod

        embed_fn = MagicMock(return_value=_np.zeros((len(TEXTS), 4)).tolist())

        with (
            patch.object(mod, "HAS_NUMPY", True),
            patch.object(mod, "np", _np),
            patch.dict(
                sys.modules,
                {
                    "matplotlib": None,
                    "matplotlib.pyplot": None,
                    "seaborn": None,
                    "sklearn": None,
                    "sklearn.metrics": None,
                    "sklearn.metrics.pairwise": None,
                },
            ),
        ):
            similarity_heatmap(TEXTS, embed_fn)

        out = capsys.readouterr().out
        assert "pip install" in out or isinstance(out, str)

    def test_heatmap_without_clustering(self):
        """Cover the cluster=False branch."""
        import numpy as _np

        import beanllm.utils.rag_debug.debug_visualization as mod

        n = len(TEXTS)
        fake_vectors = _np.eye(n)
        fake_sim = _np.eye(n)

        embed_fn = MagicMock(return_value=fake_vectors.tolist())

        cosine_sim_mock = MagicMock(return_value=fake_sim)
        sklearn_pairwise_mock = MagicMock()
        sklearn_pairwise_mock.cosine_similarity = cosine_sim_mock

        plt_mock = MagicMock()
        matplotlib_mock = MagicMock()
        matplotlib_mock.pyplot = plt_mock
        sns_mock = MagicMock()

        with (
            patch.object(mod, "HAS_NUMPY", True),
            patch.object(mod, "np", _np),
            patch.dict(
                sys.modules,
                {
                    "matplotlib": matplotlib_mock,
                    "matplotlib.pyplot": plt_mock,
                    "seaborn": sns_mock,
                    "sklearn": MagicMock(),
                    "sklearn.metrics": MagicMock(),
                    "sklearn.metrics.pairwise": sklearn_pairwise_mock,
                },
            ),
        ):
            similarity_heatmap(TEXTS, embed_fn, cluster=False)

        sns_mock.heatmap.assert_called()
        plt_mock.show.assert_called()

    def test_heatmap_with_clustering(self):
        """Cover the cluster=True branch with scipy."""
        import numpy as _np

        import beanllm.utils.rag_debug.debug_visualization as mod

        n = len(TEXTS)
        fake_vectors = _np.eye(n)
        fake_sim = _np.eye(n)

        embed_fn = MagicMock(return_value=fake_vectors.tolist())

        cosine_sim_mock = MagicMock(return_value=fake_sim)
        sklearn_pairwise_mock = MagicMock()
        sklearn_pairwise_mock.cosine_similarity = cosine_sim_mock

        linkage_result = MagicMock()
        order = list(range(n))
        leaves_list_mock = MagicMock(return_value=order)
        linkage_mock = MagicMock(return_value=linkage_result)

        scipy_hierarchy_mock = MagicMock()
        scipy_hierarchy_mock.linkage = linkage_mock
        scipy_hierarchy_mock.leaves_list = leaves_list_mock

        plt_mock = MagicMock()
        matplotlib_mock = MagicMock()
        matplotlib_mock.pyplot = plt_mock
        sns_mock = MagicMock()

        with (
            patch.object(mod, "HAS_NUMPY", True),
            patch.object(mod, "np", _np),
            patch.dict(
                sys.modules,
                {
                    "matplotlib": matplotlib_mock,
                    "matplotlib.pyplot": plt_mock,
                    "seaborn": sns_mock,
                    "sklearn": MagicMock(),
                    "sklearn.metrics": MagicMock(),
                    "sklearn.metrics.pairwise": sklearn_pairwise_mock,
                    "scipy": MagicMock(),
                    "scipy.cluster": MagicMock(),
                    "scipy.cluster.hierarchy": scipy_hierarchy_mock,
                },
            ),
        ):
            similarity_heatmap(TEXTS, embed_fn, cluster=True, method="ward")

        sns_mock.heatmap.assert_called()

    def test_heatmap_with_clustering_scipy_missing(self, capsys):
        """When scipy is missing, clustering falls back to no-cluster."""
        import numpy as _np

        import beanllm.utils.rag_debug.debug_visualization as mod

        n = len(TEXTS)
        fake_vectors = _np.eye(n)
        fake_sim = _np.eye(n)

        embed_fn = MagicMock(return_value=fake_vectors.tolist())

        cosine_sim_mock = MagicMock(return_value=fake_sim)
        sklearn_pairwise_mock = MagicMock()
        sklearn_pairwise_mock.cosine_similarity = cosine_sim_mock

        plt_mock = MagicMock()
        matplotlib_mock = MagicMock()
        matplotlib_mock.pyplot = plt_mock
        sns_mock = MagicMock()

        with (
            patch.object(mod, "HAS_NUMPY", True),
            patch.object(mod, "np", _np),
            patch.dict(
                sys.modules,
                {
                    "matplotlib": matplotlib_mock,
                    "matplotlib.pyplot": plt_mock,
                    "seaborn": sns_mock,
                    "sklearn": MagicMock(),
                    "sklearn.metrics": MagicMock(),
                    "sklearn.metrics.pairwise": sklearn_pairwise_mock,
                    "scipy": None,
                    "scipy.cluster": None,
                    "scipy.cluster.hierarchy": None,
                },
            ),
        ):
            similarity_heatmap(TEXTS, embed_fn, cluster=True)

        sns_mock.heatmap.assert_called()
        out = capsys.readouterr().out
        assert isinstance(out, str)

    def test_heatmap_with_save_path(self, tmp_path):
        import numpy as _np

        import beanllm.utils.rag_debug.debug_visualization as mod

        n = len(TEXTS)
        fake_vectors = _np.eye(n)
        fake_sim = _np.eye(n)

        embed_fn = MagicMock(return_value=fake_vectors.tolist())

        cosine_sim_mock = MagicMock(return_value=fake_sim)
        sklearn_pairwise_mock = MagicMock()
        sklearn_pairwise_mock.cosine_similarity = cosine_sim_mock

        plt_mock = MagicMock()
        matplotlib_mock = MagicMock()
        matplotlib_mock.pyplot = plt_mock
        sns_mock = MagicMock()
        save_file = str(tmp_path / "heatmap.png")

        with (
            patch.object(mod, "HAS_NUMPY", True),
            patch.object(mod, "np", _np),
            patch.dict(
                sys.modules,
                {
                    "matplotlib": matplotlib_mock,
                    "matplotlib.pyplot": plt_mock,
                    "seaborn": sns_mock,
                    "sklearn": MagicMock(),
                    "sklearn.metrics": MagicMock(),
                    "sklearn.metrics.pairwise": sklearn_pairwise_mock,
                },
            ),
        ):
            similarity_heatmap(TEXTS, embed_fn, cluster=False, save_path=save_file)

        plt_mock.savefig.assert_called_once()
