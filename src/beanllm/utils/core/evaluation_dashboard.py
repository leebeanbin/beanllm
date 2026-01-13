"""
Evaluation Dashboard - 평가 결과 시각화 대시보드
"""

from typing import Any, Dict, List, Optional

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import numpy as np

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class EvaluationDashboard:
    """
    평가 결과 시각화 대시보드

    Plotly 기반 인터랙티브 대시보드 생성
    """

    def __init__(self, use_plotly: bool = True):
        """
        Args:
            use_plotly: Plotly 사용 여부 (False면 matplotlib 사용)
        """
        self.use_plotly = use_plotly and PLOTLY_AVAILABLE

        if not self.use_plotly and not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "Either plotly or matplotlib is required. "
                "Install with: pip install plotly or pip install matplotlib"
            )

    def create_metrics_comparison(
        self,
        results: List[Dict[str, Any]],
        save_path: Optional[str] = None,
    ) -> Any:
        """
        메트릭 비교 차트 생성

        Args:
            results: 평가 결과 리스트 [{"metric": "...", "score": 0.8, ...}, ...]
            save_path: 저장 경로 (선택적)

        Returns:
            Figure 객체 (Plotly 또는 matplotlib)
        """
        if self.use_plotly:
            return self._create_plotly_comparison(results, save_path)
        else:
            return self._create_matplotlib_comparison(results, save_path)

    def _create_plotly_comparison(
        self,
        results: List[Dict[str, Any]],
        save_path: Optional[str] = None,
    ) -> "go.Figure":
        """Plotly 비교 차트"""
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly is required. Install with: pip install plotly")

        # 메트릭별로 그룹화
        metrics = {}
        for result in results:
            metric_name = result.get("metric", "unknown")
            score = result.get("score", 0.0)
            if metric_name not in metrics:
                metrics[metric_name] = []
            metrics[metric_name].append(score)

        # 평균 계산
        metric_names = list(metrics.keys())
        metric_scores = [sum(scores) / len(scores) for scores in metrics.values()]

        # Bar 차트 생성
        fig = go.Figure(
            data=[
                go.Bar(
                    x=metric_names,
                    y=metric_scores,
                    text=[f"{s:.3f}" for s in metric_scores],
                    textposition="auto",
                    marker_color="steelblue",
                )
            ]
        )

        fig.update_layout(
            title="Evaluation Metrics Comparison",
            xaxis_title="Metric",
            yaxis_title="Score",
            yaxis_range=[0, 1],
            height=500,
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def _create_matplotlib_comparison(
        self,
        results: List[Dict[str, Any]],
        save_path: Optional[str] = None,
    ) -> "plt.Figure":
        """Matplotlib 비교 차트"""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required. Install with: pip install matplotlib")

        # 메트릭별로 그룹화
        metrics = {}
        for result in results:
            metric_name = result.get("metric", "unknown")
            score = result.get("score", 0.0)
            if metric_name not in metrics:
                metrics[metric_name] = []
            metrics[metric_name].append(score)

        # 평균 계산
        metric_names = list(metrics.keys())
        metric_scores = [sum(scores) / len(scores) for scores in metrics.values()]

        # Bar 차트 생성
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(metric_names, metric_scores, color="steelblue")

        # 값 표시
        for bar, score in zip(bars, metric_scores):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{score:.3f}",
                ha="center",
                va="bottom",
            )

        ax.set_title("Evaluation Metrics Comparison")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_trend_chart(
        self,
        time_series: List[Dict[str, Any]],
        metric_name: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Any:
        """
        추이 차트 생성

        Args:
            time_series: 시계열 데이터 [{"timestamp": "...", "score": 0.8, "metric": "..."}, ...]
            metric_name: 메트릭 이름 필터 (선택적)
            save_path: 저장 경로 (선택적)

        Returns:
            Figure 객체
        """
        if self.use_plotly:
            return self._create_plotly_trend(time_series, metric_name, save_path)
        else:
            return self._create_matplotlib_trend(time_series, metric_name, save_path)

    def _create_plotly_trend(
        self,
        time_series: List[Dict[str, Any]],
        metric_name: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> "go.Figure":
        """Plotly 추이 차트"""
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly is required. Install with: pip install plotly")

        # 필터링
        filtered = time_series
        if metric_name:
            filtered = [d for d in filtered if d.get("metric") == metric_name]

        # 데이터 정렬
        filtered.sort(key=lambda x: x.get("timestamp", ""))

        # 메트릭별로 그룹화
        metrics = {}
        for data in filtered:
            metric = data.get("metric", "unknown")
            timestamp = data.get("timestamp", "")
            score = data.get("score", 0.0)

            if metric not in metrics:
                metrics[metric] = {"timestamps": [], "scores": []}
            metrics[metric]["timestamps"].append(timestamp)
            metrics[metric]["scores"].append(score)

        # Line 차트 생성
        fig = go.Figure()

        for metric, data in metrics.items():
            fig.add_trace(
                go.Scatter(
                    x=data["timestamps"],
                    y=data["scores"],
                    mode="lines+markers",
                    name=metric,
                    line=dict(width=2),
                )
            )

        fig.update_layout(
            title="Evaluation Score Trend",
            xaxis_title="Time",
            yaxis_title="Score",
            yaxis_range=[0, 1],
            height=500,
            hovermode="x unified",
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def _create_matplotlib_trend(
        self,
        time_series: List[Dict[str, Any]],
        metric_name: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> "plt.Figure":
        """Matplotlib 추이 차트"""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required. Install with: pip install matplotlib")

        # 필터링
        filtered = time_series
        if metric_name:
            filtered = [d for d in filtered if d.get("metric") == metric_name]

        # 데이터 정렬
        filtered.sort(key=lambda x: x.get("timestamp", ""))

        # 메트릭별로 그룹화
        metrics = {}
        for data in filtered:
            metric = data.get("metric", "unknown")
            timestamp = data.get("timestamp", "")
            score = data.get("score", 0.0)

            if metric not in metrics:
                metrics[metric] = {"timestamps": [], "scores": []}
            metrics[metric]["timestamps"].append(timestamp)
            metrics[metric]["scores"].append(score)

        # Line 차트 생성
        fig, ax = plt.subplots(figsize=(12, 6))

        for metric, data in metrics.items():
            ax.plot(
                data["timestamps"],
                data["scores"],
                marker="o",
                label=metric,
                linewidth=2,
            )

        ax.set_title("Evaluation Score Trend")
        ax.set_xlabel("Time")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_heatmap(
        self,
        matrix_data: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
    ) -> Any:
        """
        히트맵 생성

        Args:
            matrix_data: 행렬 데이터 {"metric1": {"case1": 0.8, "case2": 0.9, ...}, ...}
            save_path: 저장 경로 (선택적)

        Returns:
            Figure 객체
        """
        if self.use_plotly:
            return self._create_plotly_heatmap(matrix_data, save_path)
        else:
            return self._create_matplotlib_heatmap(matrix_data, save_path)

    def _create_plotly_heatmap(
        self,
        matrix_data: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
    ) -> "go.Figure":
        """Plotly 히트맵"""
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly is required. Install with: pip install plotly")

        # 데이터 변환
        metrics = list(matrix_data.keys())
        cases = set()
        for metric_data in matrix_data.values():
            cases.update(metric_data.keys())
        cases = sorted(list(cases))

        # 행렬 생성
        z = []
        for metric in metrics:
            row = [matrix_data[metric].get(case, 0.0) for case in cases]
            z.append(row)

        # 히트맵 생성
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=cases,
                y=metrics,
                colorscale="Viridis",
                text=[[f"{val:.2f}" for val in row] for row in z],
                texttemplate="%{text}",
                textfont={"size": 10},
            )
        )

        fig.update_layout(
            title="Evaluation Heatmap",
            xaxis_title="Test Case",
            yaxis_title="Metric",
            height=400 + len(metrics) * 30,
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def _create_matplotlib_heatmap(
        self,
        matrix_data: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
    ) -> "plt.Figure":
        """Matplotlib 히트맵"""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required. Install with: pip install matplotlib")

        try:
            import seaborn as sns

            SEABORN_AVAILABLE = True
        except ImportError:
            SEABORN_AVAILABLE = False

        # 데이터 변환
        metrics = list(matrix_data.keys())
        cases = set()
        for metric_data in matrix_data.values():
            cases.update(metric_data.keys())
        cases = sorted(list(cases))

        # 행렬 생성
        z = []
        for metric in metrics:
            row = [matrix_data[metric].get(case, 0.0) for case in cases]
            z.append(row)

        z_array = np.array(z)

        # 히트맵 생성
        fig, ax = plt.subplots(figsize=(max(8, len(cases) * 0.8), max(6, len(metrics) * 0.6)))

        if SEABORN_AVAILABLE:
            sns.heatmap(
                z_array,
                xticklabels=cases,
                yticklabels=metrics,
                annot=True,
                fmt=".2f",
                cmap="viridis",
                vmin=0,
                vmax=1,
                ax=ax,
            )
        else:
            im = ax.imshow(z_array, cmap="viridis", aspect="auto", vmin=0, vmax=1)
            ax.set_xticks(range(len(cases)))
            ax.set_xticklabels(cases, rotation=45, ha="right")
            ax.set_yticks(range(len(metrics)))
            ax.set_yticklabels(metrics)

            # 값 표시
            for i in range(len(metrics)):
                for j in range(len(cases)):
                    ax.text(j, i, f"{z_array[i, j]:.2f}", ha="center", va="center", color="white")

            plt.colorbar(im, ax=ax)

        ax.set_title("Evaluation Heatmap")
        ax.set_xlabel("Test Case")
        ax.set_ylabel("Metric")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig
