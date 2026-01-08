"""
Embedding Visualizer - Embedding 분석 시각화
SOLID 원칙:
- SRP: Embedding 시각화만 담당
- OCP: 새로운 시각화 방법 추가 가능
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from beanllm.ui.components import Badge, StatusIcon
from beanllm.ui.console import get_console


class EmbeddingVisualizer:
    """
    Embedding 시각화

    책임:
    - 2D/3D 좌표를 ASCII 산점도로 시각화
    - 클러스터 요약 표시
    - 이상치 하이라이트

    Example:
        ```python
        viz = EmbeddingVisualizer()

        # Scatter plot
        viz.plot_scatter(
            reduced_embeddings=[[0.1, 0.2], [0.3, 0.4], ...],
            labels=[0, 0, 1, 1, ...],
            outliers=[5, 10]
        )

        # Cluster summary
        viz.show_cluster_summary(
            cluster_sizes={0: 100, 1: 80, -1: 5},
            silhouette_score=0.75
        )
        ```
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        """
        Args:
            console: Rich Console (optional)
        """
        self.console = console or get_console()

    def plot_scatter(
        self,
        reduced_embeddings: List[List[float]],
        labels: List[int],
        outliers: Optional[List[int]] = None,
        width: int = 80,
        height: int = 30,
        title: str = "Embedding Scatter Plot",
    ) -> None:
        """
        2D 산점도 ASCII 시각화

        Args:
            reduced_embeddings: 2D 좌표 [[x1, y1], [x2, y2], ...]
            labels: 클러스터 레이블 [0, 0, 1, 1, -1, ...]
            outliers: 이상치 인덱스 목록
            width: 차트 너비
            height: 차트 높이
            title: 차트 제목
        """
        if not reduced_embeddings:
            self.console.print("[red]No embeddings to visualize[/red]")
            return

        # Ensure 2D coordinates
        coords_2d = [
            (emb[0], emb[1]) if len(emb) >= 2 else (emb[0], 0.0)
            for emb in reduced_embeddings
        ]

        # Normalize coordinates to fit in ASCII grid
        x_coords = [c[0] for c in coords_2d]
        y_coords = [c[1] for c in coords_2d]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Prevent division by zero
        x_range = x_max - x_min if x_max != x_min else 1.0
        y_range = y_max - y_min if y_max != y_min else 1.0

        # Create ASCII grid
        grid = [[" " for _ in range(width)] for _ in range(height)]

        # Map coordinates to grid
        for idx, (x, y) in enumerate(coords_2d):
            grid_x = int((x - x_min) / x_range * (width - 1))
            grid_y = int((y - y_min) / y_range * (height - 1))

            # Clamp to grid bounds
            grid_x = max(0, min(width - 1, grid_x))
            grid_y = max(0, min(height - 1, grid_y))

            # Invert y for display (0 at top)
            grid_y = height - 1 - grid_y

            # Determine marker
            label = labels[idx] if idx < len(labels) else -1
            is_outlier = outliers and idx in outliers

            if is_outlier:
                marker = "X"  # Outlier
            elif label == -1:
                marker = "·"  # Noise
            else:
                # Use different markers for clusters (up to 10)
                markers = ["●", "○", "■", "□", "▲", "△", "◆", "◇", "★", "☆"]
                marker = markers[label % len(markers)]

            grid[grid_y][grid_x] = marker

        # Render grid with Rich
        self.console.print()
        self.console.print(f"[bold cyan]{title}[/bold cyan]")
        self.console.print("─" * width)

        for row in grid:
            line = "".join(row)
            self.console.print(line)

        self.console.print("─" * width)

        # Legend
        self._show_legend(labels, outliers)

    def _show_legend(
        self, labels: List[int], outliers: Optional[List[int]] = None
    ) -> None:
        """범례 표시"""
        unique_labels = sorted(set(labels))
        markers = ["●", "○", "■", "□", "▲", "△", "◆", "◇", "★", "☆"]

        self.console.print()
        self.console.print("[bold]Legend:[/bold]")

        for label in unique_labels:
            if label == -1:
                self.console.print("  [dim]· Noise points[/dim]")
            else:
                marker = markers[label % len(markers)]
                count = labels.count(label)
                self.console.print(
                    f"  {marker} Cluster {label} ({count} points)"
                )

        if outliers:
            self.console.print(f"  [red]X Outliers ({len(outliers)} points)[/red]")

        self.console.print()

    def show_cluster_summary(
        self,
        cluster_sizes: Dict[int, int],
        silhouette_score: Optional[float] = None,
        method: str = "UMAP",
    ) -> None:
        """
        클러스터 요약 표시

        Args:
            cluster_sizes: {cluster_id: size} 딕셔너리
            silhouette_score: Silhouette 점수
            method: 차원 축소 방법
        """
        # Create summary table
        table = Table(
            title=f"📊 Cluster Summary ({method})",
            title_style="bold cyan",
            box=box.ROUNDED,
        )

        table.add_column("Cluster", style="bold cyan", justify="center")
        table.add_column("Size", style="white", justify="right")
        table.add_column("Percentage", style="green", justify="right")

        total_points = sum(cluster_sizes.values())

        for cluster_id in sorted(cluster_sizes.keys()):
            size = cluster_sizes[cluster_id]
            percentage = (size / total_points * 100) if total_points > 0 else 0.0

            # Label
            if cluster_id == -1:
                label = "Noise"
                style = "dim"
            else:
                label = f"C{cluster_id}"
                style = "white"

            # Add row
            table.add_row(
                f"[{style}]{label}[/{style}]",
                f"{size:,}",
                f"{percentage:.1f}%",
            )

        self.console.print()
        self.console.print(table)

        # Quality score
        if silhouette_score is not None:
            self._show_quality_score(silhouette_score)

    def _show_quality_score(self, silhouette_score: float) -> None:
        """클러스터링 품질 점수 표시"""
        self.console.print()
        self.console.print("[bold]Clustering Quality:[/bold]")

        # Quality bar
        bar_length = 40
        filled = int(silhouette_score * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)

        # Color based on score
        if silhouette_score > 0.7:
            color = "green"
            assessment = "Excellent"
            icon = StatusIcon.success()
        elif silhouette_score > 0.5:
            color = "cyan"
            assessment = "Good"
            icon = StatusIcon.success()
        elif silhouette_score > 0.25:
            color = "yellow"
            assessment = "Fair"
            icon = StatusIcon.warning()
        else:
            color = "red"
            assessment = "Poor"
            icon = StatusIcon.error()

        self.console.print(
            f"  Silhouette Score: [{color}]{bar}[/{color}] {silhouette_score:.4f}"
        )
        self.console.print(f"  Assessment: {icon} [{color}]{assessment}[/{color}]")
        self.console.print()

    def show_outlier_details(
        self,
        outliers: List[int],
        total_points: int,
        threshold: float = 0.05,
    ) -> None:
        """
        이상치 상세 정보 표시

        Args:
            outliers: 이상치 인덱스 목록
            total_points: 전체 포인트 수
            threshold: 이상치 비율 임계값
        """
        outlier_ratio = len(outliers) / total_points if total_points > 0 else 0.0

        self.console.print()
        self.console.print("[bold]Outlier Analysis:[/bold]")

        # Status
        if outlier_ratio > threshold:
            status = f"{StatusIcon.warning()} [yellow]High outlier ratio[/yellow]"
        else:
            status = f"{StatusIcon.success()} [green]Normal outlier ratio[/green]"

        self.console.print(f"  {status}")
        self.console.print(f"  Outliers: {len(outliers):,} / {total_points:,}")
        self.console.print(f"  Ratio: {outlier_ratio:.2%}")

        # Recommendations
        if outlier_ratio > threshold:
            self.console.print()
            self.console.print(f"{StatusIcon.info()} [cyan]Recommendations:[/cyan]")
            self.console.print("  • Check for data quality issues")
            self.console.print("  • Consider different chunking strategy")
            self.console.print("  • Review outlier embeddings manually")

        self.console.print()

    def show_3d_projection(
        self,
        reduced_embeddings: List[List[float]],
        labels: List[int],
        width: int = 60,
        height: int = 20,
        title: str = "3D Projection (Top View)",
    ) -> None:
        """
        3D 좌표의 2D 투영 시각화 (Top-down view)

        Args:
            reduced_embeddings: 3D 좌표 [[x, y, z], ...]
            labels: 클러스터 레이블
            width: 차트 너비
            height: 차트 높이
            title: 차트 제목
        """
        if not reduced_embeddings:
            self.console.print("[red]No embeddings to visualize[/red]")
            return

        # Project 3D -> 2D (use x, y coordinates, ignore z)
        coords_2d = [
            (emb[0], emb[1]) if len(emb) >= 2 else (emb[0] if len(emb) >= 1 else 0.0, 0.0)
            for emb in reduced_embeddings
        ]

        # Use existing 2D scatter plot
        self.plot_scatter(
            reduced_embeddings=coords_2d,
            labels=labels,
            width=width,
            height=height,
            title=title,
        )

    def show_distribution_histogram(
        self,
        cluster_sizes: Dict[int, int],
        max_width: int = 50,
    ) -> None:
        """
        클러스터 크기 분포 히스토그램

        Args:
            cluster_sizes: {cluster_id: size} 딕셔너리
            max_width: 막대 최대 너비
        """
        if not cluster_sizes:
            self.console.print("[red]No cluster data[/red]")
            return

        max_size = max(cluster_sizes.values())

        self.console.print()
        self.console.print("[bold]Cluster Size Distribution:[/bold]")
        self.console.print()

        for cluster_id in sorted(cluster_sizes.keys()):
            size = cluster_sizes[cluster_id]
            bar_length = int((size / max_size) * max_width) if max_size > 0 else 0

            # Label
            if cluster_id == -1:
                label = f"[dim]Noise[/dim]"
                bar_color = "dim"
            else:
                label = f"C{cluster_id}"
                bar_color = "cyan"

            bar = "█" * bar_length

            self.console.print(
                f"  {label:>6} [{bar_color}]{bar}[/{bar_color}] {size:,}"
            )

        self.console.print()
