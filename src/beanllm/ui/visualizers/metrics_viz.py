"""
Metrics Visualizer - 성능 메트릭 시각화
SOLID 원칙:
- SRP: 메트릭 시각화만 담당
- OCP: 새로운 메트릭 타입 추가 가능 (Mixin으로 확장)
"""

from __future__ import annotations

from typing import Optional

from rich.console import Console

from beanllm.ui.console import get_console
from beanllm.ui.visualizers.metrics_charts import MetricsChartsMixin
from beanllm.ui.visualizers.metrics_summary import MetricsSummaryMixin
from beanllm.ui.visualizers.metrics_tables import MetricsTablesMixin
from beanllm.ui.visualizers.metrics_viz_graph import GraphMetricsMixin
from beanllm.ui.visualizers.metrics_viz_optimizer import OptimizerMetricsMixin


class MetricsVisualizer(
    OptimizerMetricsMixin,
    GraphMetricsMixin,
    MetricsChartsMixin,
    MetricsTablesMixin,
    MetricsSummaryMixin,
):
    """
    성능 메트릭 시각화

    책임:
    - 검색 성능 메트릭 표시
    - 파라미터 비교 대시보드
    - 청크 통계 시각화
    - Optimizer 메트릭 (OptimizerMetricsMixin)
    - Knowledge Graph 메트릭 (GraphMetricsMixin)

    Example:
        ```python
        viz = MetricsVisualizer()

        # Search performance dashboard
        viz.show_search_dashboard(
            metrics={
                "avg_score": 0.85,
                "avg_latency_ms": 120,
                "total_queries": 100
            }
        )

        # Parameter comparison
        viz.compare_parameters(
            baseline={"top_k": 4, "score": 0.75},
            new={"top_k": 10, "score": 0.82}
        )
        ```
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        """
        Args:
            console: Rich Console (optional)
        """
        self.console = console or get_console()
