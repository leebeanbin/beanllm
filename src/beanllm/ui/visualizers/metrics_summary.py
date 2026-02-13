"""
Metrics Visualizer - Summary/Report Mixin

추천사항, 진행 요약, 전략 비교, 에러 요약 등의 표시를 담당합니다.
"""

from __future__ import annotations

from typing import Any, Dict, List

from rich import box
from rich.table import Table

from beanllm.ui.components import StatusIcon


class MetricsSummaryMixin:
    """
    메트릭 요약/보고서 표시를 담당하는 Mixin.

    - 추천사항 표시
    - 진행 요약
    - 전략 비교 그리드
    - 에러 요약
    """

    console: Any  # Rich Console (MetricsVisualizer에서 초기화)

    def show_recommendations(
        self,
        recommendations: List[str],
        title: str = "Recommendations",
    ) -> None:
        """
        추천사항 표시

        Args:
            recommendations: 추천사항 목록
            title: 제목
        """
        if not recommendations:
            return

        self.console.print()
        self.console.print(f"{StatusIcon.info()} [cyan bold]{title}:[/cyan bold]")
        self.console.print()

        for rec in recommendations:
            # Parse emoji/icon from recommendation
            if rec.startswith("\u2705") or rec.startswith("\u2713"):
                style = "green"
            elif rec.startswith("\u26a0\ufe0f") or rec.startswith("\u26a0"):
                style = "yellow"
            elif rec.startswith("\U0001f4a1"):
                style = "cyan"
            else:
                style = "white"

            self.console.print(f"  [{style}]{rec}[/{style}]")

        self.console.print()

    def show_progress_summary(
        self,
        completed_steps: List[str],
        total_steps: int,
    ) -> None:
        """
        진행 요약 표시

        Args:
            completed_steps: 완료된 단계 목록
            total_steps: 전체 단계 수
        """
        completed = len(completed_steps)
        percentage = (completed / total_steps * 100) if total_steps > 0 else 0.0

        self.console.print()
        self.console.print("[bold]Analysis Progress:[/bold]")
        self.console.print()

        # Progress bar
        bar_length = 40
        filled = int(percentage / 100 * bar_length)
        bar = "\u2588" * filled + "\u2591" * (bar_length - filled)

        self.console.print(f"  [cyan]{bar}[/cyan] {percentage:.0f}%")
        self.console.print()

        # Completed steps
        for step in completed_steps:
            self.console.print(f"  {StatusIcon.success()} [green]{step}[/green]")

        self.console.print()

    def show_comparison_grid(
        self,
        strategies: List[str],
        results: Dict[str, List[float]],
    ) -> None:
        """
        전략 비교 그리드

        Args:
            strategies: 전략 이름 목록 ["similarity", "mmr", "hybrid"]
            results: {query_id: [score1, score2, score3]} 딕셔너리
        """
        table = Table(
            title="\U0001f4cb Strategy Comparison",
            title_style="bold purple",
            box=box.ROUNDED,
        )

        table.add_column("Query", style="cyan", width=15)

        for strategy in strategies:
            table.add_column(strategy.capitalize(), style="white", justify="right")

        table.add_column("Best", style="green bold", justify="center")

        for query_id, scores in results.items():
            best_idx = max(range(len(scores)), key=scores.__getitem__) if scores else 0
            best_strategy = strategies[best_idx] if best_idx < len(strategies) else "N/A"

            row = [f"Q{query_id}"]
            for score in scores:
                row.append(f"{score:.3f}")
            row.append(best_strategy.upper())

            table.add_row(*row)

        self.console.print()
        self.console.print(table)
        self.console.print()

    def show_error_summary(
        self,
        errors: List[Dict[str, Any]],
        max_display: int = 10,
    ) -> None:
        """
        에러 요약 표시

        Args:
            errors: 에러 목록
            max_display: 최대 표시 개수
        """
        if not errors:
            self.console.print()
            self.console.print(f"{StatusIcon.success()} [green]No errors found![/green]")
            self.console.print()
            return

        self.console.print()
        self.console.print(f"{StatusIcon.error()} [red bold]Errors Found: {len(errors)}[/red bold]")
        self.console.print()

        for idx, error in enumerate(errors[:max_display], 1):
            error_type = error.get("type", "Unknown")
            message = error.get("message", "")
            count = error.get("count", 1)

            self.console.print(f"  {idx}. [{error_type}] {message}")
            if count > 1:
                self.console.print(f"     [dim](occurred {count} times)[/dim]")

        if len(errors) > max_display:
            self.console.print(f"\n  [dim]... and {len(errors) - max_display} more errors[/dim]")

        self.console.print()
