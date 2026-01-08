"""
Metrics Visualizer - 성능 메트릭 시각화
SOLID 원칙:
- SRP: 메트릭 시각화만 담당
- OCP: 새로운 메트릭 타입 추가 가능
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from beanllm.ui.components import Badge, Divider, StatusIcon
from beanllm.ui.console import get_console


class MetricsVisualizer:
    """
    성능 메트릭 시각화

    책임:
    - 검색 성능 메트릭 표시
    - 파라미터 비교 대시보드
    - 청크 통계 시각화

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

    def show_search_dashboard(
        self,
        metrics: Dict[str, Any],
        title: str = "Search Performance Dashboard",
    ) -> None:
        """
        검색 성능 대시보드

        Args:
            metrics: 메트릭 딕셔너리
                예: {
                    "avg_score": 0.85,
                    "avg_latency_ms": 120,
                    "total_queries": 100,
                    "top_k": 4
                }
            title: 대시보드 제목
        """
        # Create metrics table
        table = Table(
            title=title,
            title_style="bold green",
            box=box.ROUNDED,
            show_header=False,
        )

        table.add_column("Metric", style="bold cyan", width=30)
        table.add_column("Value", style="white")
        table.add_column("Status", style="white", justify="center")

        # Average score
        if "avg_score" in metrics:
            score = metrics["avg_score"]
            status = self._get_score_status(score)
            table.add_row(
                "Average Relevance Score",
                f"{score:.4f}",
                status,
            )

        # Latency
        if "avg_latency_ms" in metrics:
            latency = metrics["avg_latency_ms"]
            status = self._get_latency_status(latency)
            table.add_row(
                "Average Latency",
                f"{latency:.2f} ms",
                status,
            )

        # Total queries
        if "total_queries" in metrics:
            table.add_row(
                "Total Queries",
                f"{metrics['total_queries']:,}",
                "",
            )

        # Top K
        if "top_k" in metrics:
            table.add_row(
                "Top K",
                str(metrics["top_k"]),
                "",
            )

        # Score threshold
        if "score_threshold" in metrics:
            table.add_row(
                "Score Threshold",
                f"{metrics['score_threshold']:.2f}",
                "",
            )

        self.console.print()
        self.console.print(table)
        self.console.print()

    def _get_score_status(self, score: float) -> str:
        """점수 상태 평가"""
        if score >= 0.8:
            return f"{StatusIcon.success()} [green]Excellent[/green]"
        elif score >= 0.6:
            return f"{StatusIcon.success()} [cyan]Good[/cyan]"
        elif score >= 0.4:
            return f"{StatusIcon.warning()} [yellow]Fair[/yellow]"
        else:
            return f"{StatusIcon.error()} [red]Poor[/red]"

    def _get_latency_status(self, latency_ms: float) -> str:
        """지연 시간 상태 평가"""
        if latency_ms < 100:
            return f"{StatusIcon.success()} [green]Fast[/green]"
        elif latency_ms < 300:
            return f"{StatusIcon.success()} [cyan]Normal[/cyan]"
        elif latency_ms < 1000:
            return f"{StatusIcon.warning()} [yellow]Slow[/yellow]"
        else:
            return f"{StatusIcon.error()} [red]Very Slow[/red]"

    def compare_parameters(
        self,
        baseline: Dict[str, Any],
        new: Dict[str, Any],
        metrics: Optional[List[str]] = None,
    ) -> None:
        """
        파라미터 비교

        Args:
            baseline: 기준 파라미터 및 메트릭
            new: 새로운 파라미터 및 메트릭
            metrics: 비교할 메트릭 키 목록 (None이면 모두)
        """
        # Determine metrics to compare
        if metrics is None:
            metrics = list(set(baseline.keys()) | set(new.keys()))

        # Create comparison table
        table = Table(
            title="⚖️  Parameter Comparison",
            title_style="bold magenta",
            box=box.ROUNDED,
        )

        table.add_column("Metric", style="bold cyan")
        table.add_column("Baseline", style="white", justify="right")
        table.add_column("New", style="white", justify="right")
        table.add_column("Change", style="white", justify="center")

        for metric in metrics:
            baseline_val = baseline.get(metric, "N/A")
            new_val = new.get(metric, "N/A")

            # Calculate change
            change_str = ""
            if isinstance(baseline_val, (int, float)) and isinstance(new_val, (int, float)):
                change = new_val - baseline_val
                change_pct = (change / baseline_val * 100) if baseline_val != 0 else 0.0

                if change > 0:
                    change_str = f"[green]+{change_pct:+.1f}%[/green]"
                elif change < 0:
                    change_str = f"[red]{change_pct:.1f}%[/red]"
                else:
                    change_str = "[dim]0%[/dim]"

            # Format values
            baseline_str = f"{baseline_val:.4f}" if isinstance(baseline_val, float) else str(baseline_val)
            new_str = f"{new_val:.4f}" if isinstance(new_val, float) else str(new_val)

            table.add_row(metric, baseline_str, new_str, change_str)

        self.console.print()
        self.console.print(table)
        self.console.print()

    def show_chunk_statistics(
        self,
        stats: Dict[str, Any],
        title: str = "Chunk Statistics",
    ) -> None:
        """
        청크 통계 표시

        Args:
            stats: 통계 딕셔너리
                예: {
                    "total_chunks": 500,
                    "avg_size": 1200,
                    "min_size": 100,
                    "max_size": 2000,
                    "duplicates": 10
                }
            title: 제목
        """
        # Create stats table
        table = Table(
            title=title,
            title_style="bold blue",
            box=box.ROUNDED,
            show_header=False,
        )

        table.add_column("Metric", style="bold cyan", width=25)
        table.add_column("Value", style="white")

        # Total chunks
        if "total_chunks" in stats:
            table.add_row("Total Chunks", f"{stats['total_chunks']:,}")

        # Size statistics
        if "avg_size" in stats:
            table.add_row("Average Size", f"{stats['avg_size']:.0f} chars")
        if "min_size" in stats:
            table.add_row("Min Size", f"{stats['min_size']:,} chars")
        if "max_size" in stats:
            table.add_row("Max Size", f"{stats['max_size']:,} chars")

        # Duplicates
        if "duplicates" in stats:
            dup_count = stats["duplicates"]
            if dup_count > 0:
                table.add_row(
                    "Duplicates",
                    f"{StatusIcon.warning()} [yellow]{dup_count}[/yellow]",
                )
            else:
                table.add_row(
                    "Duplicates",
                    f"{StatusIcon.success()} [green]None[/green]",
                )

        # Overlap ratio
        if "avg_overlap_ratio" in stats:
            overlap = stats["avg_overlap_ratio"]
            table.add_row("Avg Overlap Ratio", f"{overlap:.2%}")

        self.console.print()
        self.console.print(table)
        self.console.print()

    def show_size_distribution(
        self,
        size_distribution: Dict[str, int],
        max_width: int = 50,
    ) -> None:
        """
        청크 크기 분포 히스토그램

        Args:
            size_distribution: {"0-500": 10, "500-1000": 50, ...}
            max_width: 막대 최대 너비
        """
        if not size_distribution:
            self.console.print("[red]No distribution data[/red]")
            return

        max_count = max(size_distribution.values())

        self.console.print()
        self.console.print("[bold]Chunk Size Distribution:[/bold]")
        self.console.print()

        for size_range in sorted(size_distribution.keys()):
            count = size_distribution[size_range]
            bar_length = int((count / max_count) * max_width) if max_count > 0 else 0

            bar = "█" * bar_length

            self.console.print(
                f"  {size_range:>12} [cyan]{bar}[/cyan] {count:,}"
            )

        self.console.print()

    def show_test_results(
        self,
        test_results: List[Dict[str, Any]],
        show_queries: bool = False,
    ) -> None:
        """
        테스트 결과 표시

        Args:
            test_results: 테스트 결과 리스트
                예: [
                    {
                        "query": "...",
                        "baseline": {"avg_score": 0.7, ...},
                        "new": {"avg_score": 0.8, ...}
                    }
                ]
            show_queries: 쿼리 텍스트 표시 여부
        """
        if not test_results:
            self.console.print("[yellow]No test results[/yellow]")
            return

        # Create results table
        table = Table(
            title="🧪 Test Results",
            title_style="bold yellow",
            box=box.ROUNDED,
        )

        if show_queries:
            table.add_column("#", style="dim", justify="right", width=4)
            table.add_column("Query", style="cyan", width=40)
        else:
            table.add_column("Test #", style="dim", justify="right", width=8)

        table.add_column("Baseline", style="white", justify="right")
        table.add_column("New", style="white", justify="right")
        table.add_column("Improvement", style="white", justify="center")

        for idx, result in enumerate(test_results, 1):
            baseline_score = result.get("baseline", {}).get("avg_score", 0.0)
            new_score = result.get("new", {}).get("avg_score", 0.0)
            improvement = new_score - baseline_score

            # Improvement indicator
            if improvement > 0.05:
                improvement_str = f"{StatusIcon.success()} [green]+{improvement:.3f}[/green]"
            elif improvement < -0.05:
                improvement_str = f"{StatusIcon.error()} [red]{improvement:.3f}[/red]"
            else:
                improvement_str = f"[dim]{improvement:+.3f}[/dim]"

            if show_queries:
                query = result.get("query", "")[:37] + "..." if len(result.get("query", "")) > 40 else result.get("query", "")
                table.add_row(
                    str(idx),
                    query,
                    f"{baseline_score:.3f}",
                    f"{new_score:.3f}",
                    improvement_str,
                )
            else:
                table.add_row(
                    f"Test {idx}",
                    f"{baseline_score:.3f}",
                    f"{new_score:.3f}",
                    improvement_str,
                )

        self.console.print()
        self.console.print(table)
        self.console.print()

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
            if rec.startswith("✅") or rec.startswith("✓"):
                style = "green"
            elif rec.startswith("⚠️") or rec.startswith("⚠"):
                style = "yellow"
            elif rec.startswith("💡"):
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
        bar = "█" * filled + "░" * (bar_length - filled)

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
        # Create comparison table
        table = Table(
            title="📋 Strategy Comparison",
            title_style="bold purple",
            box=box.ROUNDED,
        )

        table.add_column("Query", style="cyan", width=15)

        for strategy in strategies:
            table.add_column(strategy.capitalize(), style="white", justify="right")

        table.add_column("Best", style="green bold", justify="center")

        for query_id, scores in results.items():
            # Find best strategy
            best_idx = scores.index(max(scores)) if scores else 0
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
                예: [{"type": "ValueError", "message": "...", "count": 5}]
            max_display: 최대 표시 개수
        """
        if not errors:
            self.console.print()
            self.console.print(
                f"{StatusIcon.success()} [green]No errors found![/green]"
            )
            self.console.print()
            return

        self.console.print()
        self.console.print(
            f"{StatusIcon.error()} [red bold]Errors Found: {len(errors)}[/red bold]"
        )
        self.console.print()

        for idx, error in enumerate(errors[:max_display], 1):
            error_type = error.get("type", "Unknown")
            message = error.get("message", "")
            count = error.get("count", 1)

            self.console.print(f"  {idx}. [{error_type}] {message}")
            if count > 1:
                self.console.print(f"     [dim](occurred {count} times)[/dim]")

        if len(errors) > max_display:
            self.console.print(
                f"\n  [dim]... and {len(errors) - max_display} more errors[/dim]"
            )

        self.console.print()

    # ===== Optimizer-specific Methods =====

    def show_latency_distribution(
        self,
        avg: float,
        p50: float,
        p95: float,
        p99: float,
        max_width: int = 50,
    ) -> None:
        """
        Show latency distribution with percentiles

        Args:
            avg: Average latency (seconds)
            p50: P50 latency (seconds)
            p95: P95 latency (seconds)
            p99: P99 latency (seconds)
            max_width: Max bar width (default: 50)
        """
        max_latency = max(avg, p50, p95, p99)

        # Create bars
        avg_bar = self._create_bar(avg, max_latency, max_width, "green")
        p50_bar = self._create_bar(p50, max_latency, max_width, "cyan")
        p95_bar = self._create_bar(p95, max_latency, max_width, "yellow")
        p99_bar = self._create_bar(p99, max_latency, max_width, "red")

        # Table
        table = Table(title="⏱️  Latency Distribution", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value (s)", style="yellow", justify="right")
        table.add_column("Distribution", style="white")

        table.add_row("Average", f"{avg:.3f}", avg_bar)
        table.add_row("P50 (Median)", f"{p50:.3f}", p50_bar)
        table.add_row("P95", f"{p95:.3f}", p95_bar)
        table.add_row("P99", f"{p99:.3f}", p99_bar)

        self.console.print()
        self.console.print(table)
        self.console.print()

    def show_component_breakdown(
        self,
        breakdown: Dict[str, float],
        max_width: int = 40,
    ) -> None:
        """
        Show component breakdown with bars

        Args:
            breakdown: {component_name: percentage}
            max_width: Max bar width (default: 40)
        """
        if not breakdown:
            self.console.print("[dim]No component data[/dim]")
            return

        # Sort by percentage (descending)
        sorted_breakdown = sorted(
            breakdown.items(), key=lambda x: x[1], reverse=True
        )

        # Table
        table = Table(title="🔍 Component Breakdown", box=box.ROUNDED)
        table.add_column("Component", style="cyan")
        table.add_column("% of Total", style="yellow", justify="right")
        table.add_column("Distribution", style="white")

        for component, pct in sorted_breakdown:
            bar = self._create_percentage_bar(pct, max_width)
            table.add_row(component, f"{pct:.1f}%", bar)

        self.console.print()
        self.console.print(table)
        self.console.print()

    def show_convergence(
        self,
        history: List[Dict[str, Any]],
        max_points: int = 20,
    ) -> None:
        """
        Show optimization convergence (ASCII sparkline)

        Args:
            history: Optimization history [{trial, score, params}, ...]
            max_points: Max points to display (default: 20)
        """
        if not history:
            self.console.print("[dim]No convergence data[/dim]")
            return

        # Sample if too many points
        if len(history) > max_points:
            step = len(history) // max_points
            history = history[::step]

        scores = [h.get("score", 0) for h in history]

        # ASCII sparkline
        sparkline = self._create_sparkline(scores)

        # Stats
        initial_score = scores[0]
        final_score = scores[-1]
        best_score = max(scores)
        improvement = ((final_score - initial_score) / initial_score * 100) if initial_score > 0 else 0

        # Display
        self.console.print()
        self.console.print(Panel(
            f"[bold]Convergence Progress:[/bold]\n\n"
            f"{sparkline}\n\n"
            f"[cyan]Initial:[/cyan] {initial_score:.4f}\n"
            f"[cyan]Final:[/cyan] {final_score:.4f}\n"
            f"[cyan]Best:[/cyan] {best_score:.4f}\n"
            f"[cyan]Improvement:[/cyan] {improvement:+.1f}%",
            title="📈 Optimization Convergence",
            border_style="green",
        ))
        self.console.print()

    def show_pareto_frontier(
        self,
        pareto_solutions: List[Dict[str, Any]],
        objectives: List[str],
        max_items: int = 10,
    ) -> None:
        """
        Show Pareto optimal solutions

        Args:
            pareto_solutions: List of Pareto optimal solutions
            objectives: Objective names
            max_items: Max solutions to display (default: 10)
        """
        if not pareto_solutions:
            self.console.print("[dim]No Pareto solutions[/dim]")
            return

        # Limit
        pareto_solutions = pareto_solutions[:max_items]

        # Table
        table = Table(
            title=f"🎯 Pareto Frontier ({len(pareto_solutions)} solutions)",
            box=box.ROUNDED
        )
        table.add_column("#", style="dim", justify="right")

        for obj in objectives:
            table.add_column(obj.capitalize(), style="yellow", justify="right")

        for i, solution in enumerate(pareto_solutions, 1):
            scores = solution.get("scores", {})
            row = [str(i)]

            for obj in objectives:
                score = scores.get(obj, 0)
                row.append(f"{score:.4f}")

            table.add_row(*row)

        self.console.print()
        self.console.print(table)
        self.console.print()

    def show_ab_comparison(
        self,
        variant_a_name: str,
        variant_b_name: str,
        variant_a_mean: float,
        variant_b_mean: float,
        lift: float,
        is_significant: bool,
        max_width: int = 40,
    ) -> None:
        """
        Show A/B test comparison

        Args:
            variant_a_name: Variant A name
            variant_b_name: Variant B name
            variant_a_mean: Variant A mean score
            variant_b_mean: Variant B mean score
            lift: Lift percentage
            is_significant: Is statistically significant
            max_width: Max bar width (default: 40)
        """
        max_val = max(variant_a_mean, variant_b_mean)

        # Create bars
        bar_a = self._create_bar(variant_a_mean, max_val, max_width, "yellow")
        bar_b = self._create_bar(variant_b_mean, max_val, max_width, "green")

        # Table
        table = Table(title="🧪 A/B Comparison", box=box.ROUNDED)
        table.add_column("Variant", style="cyan")
        table.add_column("Mean Score", style="yellow", justify="right")
        table.add_column("Distribution", style="white")

        table.add_row(variant_a_name, f"{variant_a_mean:.4f}", bar_a)
        table.add_row(variant_b_name, f"{variant_b_mean:.4f}", bar_b)

        self.console.print()
        self.console.print(table)

        # Lift indicator
        lift_color = "green" if lift > 0 else "red"
        lift_emoji = "📈" if lift > 0 else "📉"
        sig_emoji = "✅" if is_significant else "⚠️ "

        self.console.print(
            f"\n{lift_emoji} [bold {lift_color}]Lift: {lift:+.1f}%[/bold {lift_color}] "
            f"{sig_emoji} {'Significant' if is_significant else 'Not significant'}"
        )
        self.console.print()

    def show_priority_distribution(
        self,
        summary: Dict[str, int],
    ) -> None:
        """
        Show recommendation priority distribution

        Args:
            summary: {"critical": 2, "high": 5, "medium": 3, "low": 1}
        """
        total = sum(summary.values())

        if total == 0:
            self.console.print("[dim]No recommendations[/dim]")
            return

        # Table
        table = Table(
            title=f"💡 Recommendation Priorities (Total: {total})",
            box=box.ROUNDED
        )
        table.add_column("Priority", style="cyan")
        table.add_column("Count", style="yellow", justify="right")
        table.add_column("% of Total", style="green", justify="right")
        table.add_column("Distribution", style="white")

        priorities = [
            ("critical", "🔴", "red"),
            ("high", "🟡", "yellow"),
            ("medium", "🔵", "cyan"),
            ("low", "⚪", "white"),
        ]

        for priority, emoji, color in priorities:
            count = summary.get(priority, 0)
            pct = (count / total * 100) if total > 0 else 0
            bar = self._create_percentage_bar(pct, 30, color)
            table.add_row(f"{emoji} {priority.capitalize()}", str(count), f"{pct:.1f}%", bar)

        self.console.print()
        self.console.print(table)
        self.console.print()

    # ===== Helper Methods for Optimizer =====

    def _create_bar(
        self,
        value: float,
        max_value: float,
        max_width: int,
        color: str = "green",
    ) -> str:
        """Create a horizontal bar"""
        if max_value == 0:
            return "[dim]░[/dim]" * max_width

        filled = int((value / max_value) * max_width)
        bar = f"[{color}]" + "█" * filled + f"[/{color}]"
        bar += "[dim]░[/dim]" * (max_width - filled)

        return bar

    def _create_percentage_bar(
        self,
        percentage: float,
        max_width: int,
        color: str = "green",
    ) -> str:
        """Create a percentage bar"""
        filled = int((percentage / 100) * max_width)
        bar = f"[{color}]" + "█" * filled + f"[/{color}]"
        bar += "[dim]░[/dim]" * (max_width - filled)

        return bar

    def _create_sparkline(self, values: List[float]) -> str:
        """Create ASCII sparkline"""
        if not values:
            return ""

        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val

        if range_val == 0:
            return "▄" * len(values)

        # Sparkline characters (8 levels)
        chars = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

        sparkline = ""
        for value in values:
            normalized = (value - min_val) / range_val
            index = int(normalized * (len(chars) - 1))
            sparkline += chars[index]

        return f"[cyan]{sparkline}[/cyan]"
