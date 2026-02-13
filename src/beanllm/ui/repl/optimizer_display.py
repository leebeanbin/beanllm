"""
Optimizer display helpers - Rich CLI display for Auto-Optimizer results.
Extracted from optimizer_commands for single responsibility.
"""

from __future__ import annotations

from typing import Any, Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree


def show_string_recommendations(console: Console, recommendations: List[str]) -> None:
    """Show string recommendations as a simple tree list."""
    if not recommendations:
        console.print("[dim]No recommendations[/dim]")
        return
    tree = Tree("💡 [bold]Recommendations[/bold]")
    for i, rec in enumerate(recommendations, 1):
        tree.add(f"[{i}] {rec}")
    console.print(tree)


def show_recommendations_panel(console: Console, recommendations: List[Dict[str, Any]]) -> None:
    """Show recommendations (dict with priority, title, description, etc.) in a tree panel."""
    if not recommendations:
        console.print("[dim]No recommendations[/dim]")
        return
    tree = Tree("💡 [bold]Recommendations[/bold]")
    priority_emoji = {
        "critical": "🔴",
        "high": "🟡",
        "medium": "🔵",
        "low": "⚪",
    }
    for i, rec in enumerate(recommendations, 1):
        priority = rec["priority"]
        title = rec["title"]
        description = rec["description"]
        action = rec.get("action", "")
        impact = rec.get("expected_impact", "")
        emoji = priority_emoji.get(priority, "⚪")
        branch = tree.add(f"{emoji} [{i}] [bold]{title}[/bold] ([{priority.upper()}])")
        branch.add(f"[dim]{description}[/dim]")
        if action:
            branch.add(f"[cyan]Action:[/cyan] {action}")
        if impact:
            branch.add(f"[green]Impact:[/green] {impact}")
    console.print(tree)


def show_benchmark_results(
    console: Console,
    result: Any,
    visualizer: Any,
    show_queries: bool = False,
) -> None:
    """Display benchmark result: summary table, latency distribution, optional queries."""
    console.print("\n[bold green]✅ Benchmark Complete![/bold green]\n")
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
    console.print(table)
    visualizer.show_latency_distribution(
        avg=result.avg_latency,
        p50=result.p50_latency,
        p95=result.p95_latency,
        p99=result.p99_latency,
    )
    if show_queries and result.queries:
        console.print("\n[bold]Generated Queries:[/bold]")
        for i, query in enumerate(result.queries[:10], 1):
            console.print(f"  {i}. {query}")
        if len(result.queries) > 10:
            console.print(f"  ... and {len(result.queries) - 10} more")


def show_optimize_parameter_table(console: Console, parameters: List[Dict[str, Any]]) -> None:
    """Display parameter summary table before optimization."""
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
    console.print(param_table)


def show_optimize_results(console: Console, result: Any) -> None:
    """Display optimization result: panel, best params table, convergence info."""
    console.print("\n[bold green]✅ Optimization Complete![/bold green]\n")
    console.print(
        Panel(
            f"[bold yellow]Optimization ID:[/bold yellow] {result.optimization_id}\n"
            f"[bold yellow]Best Score:[/bold yellow] {result.best_score:.4f}\n"
            f"[bold yellow]Trials:[/bold yellow] {result.n_trials}",
            title="📈 Results",
            border_style="green",
        )
    )
    best_table = Table(title="🏆 Best Parameters")
    best_table.add_column("Parameter", style="cyan")
    best_table.add_column("Value", style="yellow")
    best_params = result.best_params or {}
    for param_name, param_value in best_params.items():
        value_str = f"{param_value:.4f}" if isinstance(param_value, float) else str(param_value)
        best_table.add_row(param_name, value_str)
    console.print(best_table)
    if result.convergence_data:
        console.print("\n[bold]Convergence Info:[/bold]")
        console.print(f"  • Total trials: {len(result.convergence_data)}")
        if result.convergence_data:
            last_trial = result.convergence_data[-1]
            for key, value in last_trial.items():
                console.print(f"  • {key}: {value}")


def show_profile_results(
    console: Console,
    result: Any,
    visualizer: Any,
    show_recommendations: bool = True,
) -> None:
    """Display profile result: summary panel, component table,
    breakdown, optional recommendations."""
    console.print("\n[bold green]✅ Profiling Complete![/bold green]\n")
    console.print(
        Panel(
            f"[bold yellow]Profile ID:[/bold yellow] {result.profile_id}\n"
            f"[bold yellow]Total Duration:[/bold yellow] {result.total_duration_ms:.1f}ms\n"
            f"[bold yellow]Total Tokens:[/bold yellow] {result.total_tokens}\n"
            f"[bold yellow]Total Cost:[/bold yellow] ${result.total_cost:.4f}\n"
            f"[bold yellow]Bottleneck:[/bold yellow] {result.bottleneck}",
            title="⚡ Profile Summary",
            border_style="cyan",
        )
    )
    if result.components:
        comp_table = Table(title="🔍 Component Breakdown")
        comp_table.add_column("Component", style="cyan")
        comp_table.add_column("Duration (ms)", style="yellow")
        comp_table.add_column("Tokens", style="green")
        comp_table.add_column("Cost ($)", style="magenta")
        comp_table.add_column("% of Total", style="blue")
        breakdown = result.breakdown or {}
        for component in result.components:
            name = component.get("name", "unknown")
            duration = component.get("duration_ms", 0)
            tokens = component.get("tokens", 0)
            cost = component.get("cost", 0)
            pct = breakdown.get(name, 0)
            comp_table.add_row(name, f"{duration:.1f}", str(tokens), f"{cost:.4f}", f"{pct:.1f}%")
        console.print(comp_table)
    if result.breakdown:
        visualizer.show_component_breakdown(result.breakdown)
    if show_recommendations and result.recommendations:
        show_string_recommendations(console, result.recommendations)


def show_ab_test_results(
    console: Console,
    result: Any,
    variant_a_name: str,
    variant_b_name: str,
) -> None:
    """Display A/B test result: winner panel, comparison table, interpretation."""
    console.print("\n[bold green]✅ A/B Test Complete![/bold green]\n")
    winner_emoji = "🏆" if result.winner != "tie" else "🤝"
    winner_text = (
        f"[bold green]{result.winner}[/bold green]"
        if result.winner != "tie"
        else "[bold yellow]Tie (no significant difference)[/bold yellow]"
    )
    console.print(
        Panel(
            f"{winner_emoji} [bold]Winner:[/bold] {winner_text}\n\n"
            f"[bold yellow]Lift:[/bold yellow] {result.lift:+.1f}%\n"
            f"[bold yellow]P-value:[/bold yellow] {result.p_value:.4f}\n"
            f"[bold yellow]Significant:[/bold yellow] "
            f"{'Yes ✅' if result.is_significant else 'No ❌'}\n"
            f"[bold yellow]Confidence:[/bold yellow] {result.confidence_level * 100:.0f}%",
            title="🧪 A/B Test Results",
            border_style="green" if result.is_significant else "yellow",
        )
    )
    comp_table = Table(title="📊 Variant Comparison")
    comp_table.add_column("Metric", style="cyan")
    comp_table.add_column(f"Variant A ({variant_a_name})", style="yellow")
    comp_table.add_column(f"Variant B ({variant_b_name})", style="green")
    comp_table.add_row("Mean Score", f"{result.variant_a_mean:.4f}", f"{result.variant_b_mean:.4f}")
    comp_table.add_row(
        "Winner",
        "✅" if result.winner == "A" else "",
        "✅" if result.winner == "B" else "",
    )
    console.print(comp_table)
    if result.is_significant:
        if result.lift > 0:
            console.print(
                f"\n[bold green]📈 {variant_b_name} is "
                f"{abs(result.lift):.1f}% better than "
                f"{variant_a_name}[/bold green]"
            )
        else:
            console.print(
                f"\n[bold red]📉 {variant_b_name} is "
                f"{abs(result.lift):.1f}% worse than "
                f"{variant_a_name}[/bold red]"
            )
    else:
        console.print(
            "\n[bold yellow]⚠️  No statistically significant difference detected[/bold yellow]"
        )


def show_recommendations_result(
    console: Console,
    profile_id: str,
    result: Any,
    recommendations: List[Dict[str, Any]],
) -> None:
    """Display get_recommendations result: summary panel and recommendations tree."""
    console.print(
        f"[bold green]✅ Found {len(result.recommendations)} recommendations[/bold green]\n"
    )
    summary = result.summary or {"critical": 0, "high": 0, "medium": 0, "low": 0}
    console.print(
        Panel(
            f"[bold yellow]Profile ID:[/bold yellow] {profile_id}\n"
            f"[bold red]Critical:[/bold red] {summary.get('critical', 0)}\n"
            f"[bold yellow]High:[/bold yellow] {summary.get('high', 0)}\n"
            f"[bold cyan]Medium:[/bold cyan] {summary.get('medium', 0)}\n"
            f"[bold]Low:[/bold] {summary.get('low', 0)}",
            title="📊 Recommendation Summary",
            border_style="cyan",
        )
    )
    show_recommendations_panel(console, recommendations)


def show_compare_results(console: Console, result: Dict[str, Any], config_ids: List[str]) -> None:
    """Display compare_configs result: summary panel and comparison table."""
    console.print(f"[bold green]✅ Compared {len(config_ids)} configs[/bold green]\n")
    console.print(
        Panel(
            f"[bold yellow]Total Configs:[/bold yellow] {result['summary']['total_configs']}\n"
            f"[bold yellow]Found:[/bold yellow] {result['summary']['found']}",
            title="📊 Comparison Summary",
            border_style="cyan",
        )
    )
    comp_table = Table(title="⚖️  Configuration Comparison")
    comp_table.add_column("Config ID", style="cyan")
    comp_table.add_column("Type", style="yellow")
    comp_table.add_column("Key Metrics", style="green")
    for config_id, config_data in result["configs"].items():
        config_type = config_data["type"]
        if config_type == "optimization":
            metrics = f"Score: {config_data['best_score']:.4f}\nTrials: {config_data['n_trials']}"
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
    console.print(comp_table)
