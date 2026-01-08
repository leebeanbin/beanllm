"""
OptimizerCommands - Rich CLI interface for Auto-Optimizer
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.tree import Tree

from beanllm.facade.optimizer_facade import Optimizer
from beanllm.ui.visualizers.metrics_viz import MetricsVisualizer
from beanllm.utils.logger import get_logger

logger = get_logger(__name__)


class OptimizerCommands:
    """
    Rich CLI Commands for Auto-Optimizer

    Provides interactive commands for:
    - Benchmarking
    - Parameter optimization
    - System profiling
    - A/B testing
    - Recommendations
    - Configuration comparison

    Example:
        ```python
        from beanllm.ui.repl.optimizer_commands import OptimizerCommands

        commands = OptimizerCommands()

        # Run benchmark
        await commands.cmd_benchmark(
            num_queries=50,
            query_types=["simple", "complex"],
            domain="machine learning"
        )

        # Optimize parameters
        await commands.cmd_optimize(
            parameters=[
                {"name": "top_k", "type": "integer", "low": 1, "high": 20},
            ],
            method="bayesian",
            n_trials=30
        )
        ```
    """

    def __init__(
        self,
        optimizer: Optional[Optimizer] = None,
        console: Optional[Console] = None,
    ) -> None:
        """
        Initialize optimizer commands

        Args:
            optimizer: Optional Optimizer facade (for DI)
            console: Optional Rich console
        """
        self._optimizer = optimizer or Optimizer()
        self.console = console or Console()
        self._visualizer = MetricsVisualizer(console=self.console)

        logger.info("OptimizerCommands initialized")

    # ===== CLI Commands =====

    async def cmd_benchmark(
        self,
        num_queries: Optional[int] = None,
        queries: Optional[List[str]] = None,
        query_types: Optional[List[str]] = None,
        domain: Optional[str] = None,
        show_queries: bool = False,
    ) -> None:
        """
        Run benchmark and display results

        Args:
            num_queries: Number of synthetic queries (default: 50)
            queries: Optional custom queries
            query_types: Query types to generate
            domain: Domain for synthetic queries
            show_queries: Show generated queries (default: False)

        Example:
            ```python
            # Synthetic queries
            await commands.cmd_benchmark(
                num_queries=50,
                query_types=["simple", "complex"],
                domain="healthcare"
            )

            # Custom queries
            await commands.cmd_benchmark(
                queries=["What is RAG?", "How does it work?"]
            )
            ```
        """
        self.console.print("\n[bold cyan]🔍 Running Benchmark...[/bold cyan]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                f"Benchmarking ({num_queries or len(queries or [])} queries)...",
                total=None,
            )

            try:
                result = await self._optimizer.benchmark(
                    num_queries=num_queries,
                    queries=queries,
                    query_types=query_types,
                    domain=domain,
                )

                progress.update(task, completed=True)

                # Display results
                self.console.print("\n[bold green]✅ Benchmark Complete![/bold green]\n")

                # Summary table
                table = Table(title=f"📊 Benchmark Results (ID: {result.benchmark_id})")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="yellow")

                table.add_row("Total Queries", str(result.num_queries))
                table.add_row("Avg Latency", f"{result.avg_latency:.3f}s")
                table.add_row("P50 Latency", f"{result.p50_latency:.3f}s")
                table.add_row("P95 Latency", f"{result.p95_latency:.3f}s")
                table.add_row("P99 Latency", f"{result.p99_latency:.3f}s")
                table.add_row("Throughput", f"{result.throughput:.2f} q/s")
                table.add_row("Avg Score", f"{result.avg_score:.3f}")
                table.add_row("Min Score", f"{result.min_score:.3f}")
                table.add_row("Max Score", f"{result.max_score:.3f}")
                table.add_row("Total Duration", f"{result.total_duration:.2f}s")

                self.console.print(table)

                # Latency distribution
                self._visualizer.show_latency_distribution(
                    avg=result.avg_latency,
                    p50=result.p50_latency,
                    p95=result.p95_latency,
                    p99=result.p99_latency,
                )

                # Show queries if requested
                if show_queries and result.queries:
                    self.console.print("\n[bold]Generated Queries:[/bold]")
                    for i, query in enumerate(result.queries[:10], 1):
                        self.console.print(f"  {i}. {query}")
                    if len(result.queries) > 10:
                        self.console.print(
                            f"  ... and {len(result.queries) - 10} more"
                        )

            except Exception as e:
                progress.update(task, completed=True)
                self.console.print(f"\n[bold red]❌ Benchmark failed: {e}[/bold red]\n")
                logger.error(f"Benchmark error: {e}")

    async def cmd_optimize(
        self,
        parameters: List[Dict[str, Any]],
        method: str = "bayesian",
        n_trials: int = 30,
        multi_objective: bool = False,
        objectives: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Optimize parameters and display results

        Args:
            parameters: Parameter definitions
            method: Optimization method (default: "bayesian")
            n_trials: Number of trials (default: 30)
            multi_objective: Enable multi-objective (default: False)
            objectives: Objectives for multi-objective

        Example:
            ```python
            # Single-objective
            await commands.cmd_optimize(
                parameters=[
                    {"name": "top_k", "type": "integer", "low": 1, "high": 20},
                    {"name": "threshold", "type": "float", "low": 0.0, "high": 1.0},
                ],
                method="bayesian",
                n_trials=30
            )

            # Multi-objective
            await commands.cmd_optimize(
                parameters=[...],
                multi_objective=True,
                objectives=[
                    {"name": "quality", "maximize": True, "weight": 0.6},
                    {"name": "latency", "maximize": False, "weight": 0.3},
                ],
                n_trials=50
            )
            ```
        """
        self.console.print("\n[bold cyan]🎯 Running Optimization...[/bold cyan]\n")

        # Parameter summary
        param_table = Table(title="Parameters to Optimize")
        param_table.add_column("Name", style="cyan")
        param_table.add_column("Type", style="yellow")
        param_table.add_column("Range/Categories", style="green")

        for param in parameters:
            name = param["name"]
            param_type = param["type"]

            if param_type in ["integer", "float"]:
                range_str = f"[{param['low']}, {param['high']}]"
            elif param_type == "categorical":
                range_str = str(param["categories"])
            else:
                range_str = "boolean"

            param_table.add_row(name, param_type, range_str)

        self.console.print(param_table)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                f"Optimizing with {method} ({n_trials} trials)...",
                total=None,
            )

            try:
                result = await self._optimizer.optimize(
                    parameters=parameters,
                    method=method,
                    n_trials=n_trials,
                    multi_objective=multi_objective,
                    objectives=objectives,
                )

                progress.update(task, completed=True)

                # Display results
                self.console.print("\n[bold green]✅ Optimization Complete![/bold green]\n")

                # Best parameters
                self.console.print(Panel(
                    f"[bold yellow]Optimization ID:[/bold yellow] {result.optimization_id}\n"
                    f"[bold yellow]Best Score:[/bold yellow] {result.best_score:.4f}\n"
                    f"[bold yellow]Trials:[/bold yellow] {result.n_trials}",
                    title="📈 Results",
                    border_style="green",
                ))

                # Best parameters table
                best_table = Table(title="🏆 Best Parameters")
                best_table.add_column("Parameter", style="cyan")
                best_table.add_column("Value", style="yellow")

                for param_name, param_value in result.best_params.items():
                    if isinstance(param_value, float):
                        value_str = f"{param_value:.4f}"
                    else:
                        value_str = str(param_value)
                    best_table.add_row(param_name, value_str)

                self.console.print(best_table)

                # Convergence data
                if result.convergence_data:
                    self.console.print("\n[bold]Convergence Info:[/bold]")
                    for key, value in result.convergence_data.items():
                        self.console.print(f"  • {key}: {value}")

            except Exception as e:
                progress.update(task, completed=True)
                self.console.print(f"\n[bold red]❌ Optimization failed: {e}[/bold red]\n")
                logger.error(f"Optimization error: {e}")

    async def cmd_profile(
        self,
        components: Optional[List[str]] = None,
        show_recommendations: bool = True,
    ) -> None:
        """
        Profile system and display results

        Args:
            components: Components to profile (default: all)
            show_recommendations: Show recommendations (default: True)

        Example:
            ```python
            await commands.cmd_profile(
                components=["embedding", "retrieval", "generation"],
                show_recommendations=True
            )
            ```
        """
        self.console.print("\n[bold cyan]📊 Profiling System...[/bold cyan]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("Profiling components...", total=None)

            try:
                result = await self._optimizer.profile(components=components)

                progress.update(task, completed=True)

                # Display results
                self.console.print("\n[bold green]✅ Profiling Complete![/bold green]\n")

                # Summary
                self.console.print(Panel(
                    f"[bold yellow]Profile ID:[/bold yellow] {result.profile_id}\n"
                    f"[bold yellow]Total Duration:[/bold yellow] {result.total_duration_ms:.1f}ms\n"
                    f"[bold yellow]Total Tokens:[/bold yellow] {result.total_tokens}\n"
                    f"[bold yellow]Total Cost:[/bold yellow] ${result.total_cost:.4f}\n"
                    f"[bold yellow]Bottleneck:[/bold yellow] {result.bottleneck}",
                    title="⚡ Profile Summary",
                    border_style="cyan",
                ))

                # Component breakdown
                if result.components:
                    comp_table = Table(title="🔍 Component Breakdown")
                    comp_table.add_column("Component", style="cyan")
                    comp_table.add_column("Duration (ms)", style="yellow")
                    comp_table.add_column("Tokens", style="green")
                    comp_table.add_column("Cost ($)", style="magenta")
                    comp_table.add_column("% of Total", style="blue")

                    for name, metrics in result.components.items():
                        duration = metrics["duration_ms"]
                        tokens = metrics["tokens"]
                        cost = metrics["cost"]
                        pct = result.breakdown.get(name, 0)

                        comp_table.add_row(
                            name,
                            f"{duration:.1f}",
                            str(tokens),
                            f"{cost:.4f}",
                            f"{pct:.1f}%",
                        )

                    self.console.print(comp_table)

                # Breakdown visualization
                self._visualizer.show_component_breakdown(result.breakdown)

                # Recommendations
                if show_recommendations and result.recommendations:
                    self._show_recommendations_panel(result.recommendations)

            except Exception as e:
                progress.update(task, completed=True)
                self.console.print(f"\n[bold red]❌ Profiling failed: {e}[/bold red]\n")
                logger.error(f"Profiling error: {e}")

    async def cmd_ab_test(
        self,
        variant_a_name: str,
        variant_b_name: str,
        num_queries: int = 50,
        confidence_level: float = 0.95,
    ) -> None:
        """
        Run A/B test and display results

        Args:
            variant_a_name: Name of variant A
            variant_b_name: Name of variant B
            num_queries: Number of test queries (default: 50)
            confidence_level: Confidence level (default: 0.95)

        Example:
            ```python
            await commands.cmd_ab_test(
                variant_a_name="Baseline",
                variant_b_name="Optimized",
                num_queries=100,
                confidence_level=0.95
            )
            ```
        """
        self.console.print("\n[bold cyan]🧪 Running A/B Test...[/bold cyan]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                f"Testing {variant_a_name} vs {variant_b_name} ({num_queries} queries)...",
                total=None,
            )

            try:
                result = await self._optimizer.ab_test(
                    variant_a_name=variant_a_name,
                    variant_b_name=variant_b_name,
                    num_queries=num_queries,
                    confidence_level=confidence_level,
                )

                progress.update(task, completed=True)

                # Display results
                self.console.print("\n[bold green]✅ A/B Test Complete![/bold green]\n")

                # Winner announcement
                winner_emoji = "🏆" if result.winner != "tie" else "🤝"
                winner_text = (
                    f"[bold green]{result.winner}[/bold green]"
                    if result.winner != "tie"
                    else "[bold yellow]Tie (no significant difference)[/bold yellow]"
                )

                self.console.print(Panel(
                    f"{winner_emoji} [bold]Winner:[/bold] {winner_text}\n\n"
                    f"[bold yellow]Lift:[/bold yellow] {result.lift:+.1f}%\n"
                    f"[bold yellow]P-value:[/bold yellow] {result.p_value:.4f}\n"
                    f"[bold yellow]Significant:[/bold yellow] {'Yes ✅' if result.is_significant else 'No ❌'}\n"
                    f"[bold yellow]Confidence:[/bold yellow] {result.confidence_level * 100:.0f}%",
                    title="🧪 A/B Test Results",
                    border_style="green" if result.is_significant else "yellow",
                ))

                # Comparison table
                comp_table = Table(title="📊 Variant Comparison")
                comp_table.add_column("Metric", style="cyan")
                comp_table.add_column(f"Variant A ({variant_a_name})", style="yellow")
                comp_table.add_column(f"Variant B ({variant_b_name})", style="green")

                comp_table.add_row("Mean Score", f"{result.variant_a_mean:.4f}", f"{result.variant_b_mean:.4f}")
                comp_table.add_row("Winner", "✅" if result.winner == "A" else "", "✅" if result.winner == "B" else "")

                self.console.print(comp_table)

                # Interpretation
                if result.is_significant:
                    if result.lift > 0:
                        self.console.print(
                            f"\n[bold green]📈 {variant_b_name} is {abs(result.lift):.1f}% better than {variant_a_name}[/bold green]"
                        )
                    else:
                        self.console.print(
                            f"\n[bold red]📉 {variant_b_name} is {abs(result.lift):.1f}% worse than {variant_a_name}[/bold red]"
                        )
                else:
                    self.console.print(
                        "\n[bold yellow]⚠️  No statistically significant difference detected[/bold yellow]"
                    )

            except Exception as e:
                progress.update(task, completed=True)
                self.console.print(f"\n[bold red]❌ A/B test failed: {e}[/bold red]\n")
                logger.error(f"A/B test error: {e}")

    async def cmd_recommendations(
        self,
        profile_id: str,
        priority: Optional[str] = None,
        category: Optional[str] = None,
        max_items: int = 10,
    ) -> None:
        """
        Get and display recommendations

        Args:
            profile_id: Profile ID
            priority: Filter by priority (critical, high, medium, low)
            category: Filter by category (performance, cost, quality, reliability, best_practice)
            max_items: Max items to display (default: 10)

        Example:
            ```python
            await commands.cmd_recommendations(
                profile_id="abc-123",
                priority="critical",
                max_items=5
            )
            ```
        """
        self.console.print("\n[bold cyan]💡 Getting Recommendations...[/bold cyan]\n")

        try:
            result = await self._optimizer.get_recommendations(profile_id)

            # Filter
            recommendations = result.recommendations

            if priority:
                recommendations = [
                    r for r in recommendations if r["priority"] == priority.lower()
                ]

            if category:
                recommendations = [
                    r for r in recommendations if r["category"] == category.lower()
                ]

            recommendations = recommendations[:max_items]

            # Display
            self.console.print(f"[bold green]✅ Found {len(result.recommendations)} recommendations[/bold green]\n")

            # Summary
            self.console.print(Panel(
                f"[bold yellow]Profile ID:[/bold yellow] {profile_id}\n"
                f"[bold red]Critical:[/bold red] {result.summary['critical']}\n"
                f"[bold yellow]High:[/bold yellow] {result.summary['high']}\n"
                f"[bold cyan]Medium:[/bold cyan] {result.summary['medium']}\n"
                f"[bold]Low:[/bold] {result.summary['low']}",
                title="📊 Recommendation Summary",
                border_style="cyan",
            ))

            # Recommendations
            self._show_recommendations_panel(recommendations)

        except Exception as e:
            self.console.print(f"\n[bold red]❌ Failed to get recommendations: {e}[/bold red]\n")
            logger.error(f"Recommendations error: {e}")

    async def cmd_compare(
        self,
        config_ids: List[str],
    ) -> None:
        """
        Compare multiple configurations

        Args:
            config_ids: List of config IDs

        Example:
            ```python
            await commands.cmd_compare([
                "opt-abc-123",
                "opt-def-456",
                "profile-xyz-789"
            ])
            ```
        """
        self.console.print("\n[bold cyan]⚖️  Comparing Configurations...[/bold cyan]\n")

        try:
            result = await self._optimizer.compare_configs(config_ids)

            # Display
            self.console.print(f"[bold green]✅ Compared {len(config_ids)} configs[/bold green]\n")

            # Summary
            self.console.print(Panel(
                f"[bold yellow]Total Configs:[/bold yellow] {result['summary']['total_configs']}\n"
                f"[bold yellow]Found:[/bold yellow] {result['summary']['found']}",
                title="📊 Comparison Summary",
                border_style="cyan",
            ))

            # Comparison table
            comp_table = Table(title="⚖️  Configuration Comparison")
            comp_table.add_column("Config ID", style="cyan")
            comp_table.add_column("Type", style="yellow")
            comp_table.add_column("Key Metrics", style="green")

            for config_id, config_data in result["configs"].items():
                config_type = config_data["type"]

                if config_type == "optimization":
                    metrics = (
                        f"Score: {config_data['best_score']:.4f}\n"
                        f"Trials: {config_data['n_trials']}"
                    )
                elif config_type == "profile":
                    metrics = (
                        f"Duration: {config_data['total_duration_ms']:.1f}ms\n"
                        f"Cost: ${config_data['total_cost']:.4f}\n"
                        f"Bottleneck: {config_data['bottleneck']}"
                    )
                elif config_type == "ab_test":
                    metrics = (
                        f"Winner: {config_data['winner']}\n"
                        f"Lift: {config_data['lift']:.1f}%\n"
                        f"Significant: {config_data['is_significant']}"
                    )
                else:
                    metrics = config_data.get("error", "Unknown")

                comp_table.add_row(config_id, config_type, metrics)

            self.console.print(comp_table)

        except Exception as e:
            self.console.print(f"\n[bold red]❌ Comparison failed: {e}[/bold red]\n")
            logger.error(f"Comparison error: {e}")

    # ===== Helper Methods =====

    def _show_recommendations_panel(self, recommendations: List[Dict[str, Any]]) -> None:
        """Show recommendations in a tree panel"""
        if not recommendations:
            self.console.print("[dim]No recommendations[/dim]")
            return

        tree = Tree("💡 [bold]Recommendations[/bold]")

        for i, rec in enumerate(recommendations, 1):
            priority = rec["priority"]
            title = rec["title"]
            description = rec["description"]
            action = rec.get("action", "")
            impact = rec.get("expected_impact", "")

            # Priority emoji
            priority_emoji = {
                "critical": "🔴",
                "high": "🟡",
                "medium": "🔵",
                "low": "⚪",
            }.get(priority, "⚪")

            # Create branch
            branch = tree.add(
                f"{priority_emoji} [{i}] [bold]{title}[/bold] ([{priority.upper()}])"
            )
            branch.add(f"[dim]{description}[/dim]")
            if action:
                branch.add(f"[cyan]Action:[/cyan] {action}")
            if impact:
                branch.add(f"[green]Impact:[/green] {impact}")

        self.console.print(tree)
