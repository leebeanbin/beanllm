"""
Optimizer Metrics Visualization Mixin

Optimizer 관련 메트릭 시각화 메서드를 제공하는 Mixin 클래스.
MetricsVisualizer에 mix-in 되어 사용됩니다.

SOLID 원칙:
- SRP: Optimizer 시각화만 담당
- OCP: 새로운 Optimizer 메트릭 추가 가능
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from rich import box
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from rich.console import Console


class OptimizerMetricsMixin:
    """Optimizer 관련 메트릭 시각화 Mixin

    MetricsVisualizer에 mix-in 되어, self.console 및
    self._create_bar / self._create_percentage_bar / self._create_sparkline 을
    통해 시각화를 수행합니다.
    """

    console: Console

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
        avg_bar = self._create_bar(avg, max_latency, max_width, "green")  # type: ignore[attr-defined]
        p50_bar = self._create_bar(p50, max_latency, max_width, "cyan")  # type: ignore[attr-defined]
        p95_bar = self._create_bar(p95, max_latency, max_width, "yellow")  # type: ignore[attr-defined]
        p99_bar = self._create_bar(p99, max_latency, max_width, "red")  # type: ignore[attr-defined]

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
        sorted_breakdown = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)

        # Table
        table = Table(title="🔍 Component Breakdown", box=box.ROUNDED)
        table.add_column("Component", style="cyan")
        table.add_column("% of Total", style="yellow", justify="right")
        table.add_column("Distribution", style="white")

        for component, pct in sorted_breakdown:
            bar = self._create_percentage_bar(pct, max_width)  # type: ignore[attr-defined]
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
        sparkline = self._create_sparkline(scores)  # type: ignore[attr-defined]

        # Stats
        initial_score = scores[0]
        final_score = scores[-1]
        best_score = max(scores)
        improvement = (
            ((final_score - initial_score) / initial_score * 100) if initial_score > 0 else 0
        )

        # Display
        self.console.print()
        self.console.print(
            Panel(
                f"[bold]Convergence Progress:[/bold]\n\n"
                f"{sparkline}\n\n"
                f"[cyan]Initial:[/cyan] {initial_score:.4f}\n"
                f"[cyan]Final:[/cyan] {final_score:.4f}\n"
                f"[cyan]Best:[/cyan] {best_score:.4f}\n"
                f"[cyan]Improvement:[/cyan] {improvement:+.1f}%",
                title="📈 Optimization Convergence",
                border_style="green",
            )
        )
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
            title=f"🎯 Pareto Frontier ({len(pareto_solutions)} solutions)", box=box.ROUNDED
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
        bar_a = self._create_bar(variant_a_mean, max_val, max_width, "yellow")  # type: ignore[attr-defined]
        bar_b = self._create_bar(variant_b_mean, max_val, max_width, "green")  # type: ignore[attr-defined]

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
        table = Table(title=f"💡 Recommendation Priorities (Total: {total})", box=box.ROUNDED)
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
            bar = self._create_percentage_bar(pct, 30, color)  # type: ignore[attr-defined]
            table.add_row(f"{emoji} {priority.capitalize()}", str(count), f"{pct:.1f}%", bar)

        self.console.print()
        self.console.print(table)
        self.console.print()
