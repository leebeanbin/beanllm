"""
Knowledge Graph stats display - statistics tables and distributions.
"""

from __future__ import annotations

from typing import Any, Dict

from rich.console import Console
from rich.table import Table


def render_stats_tables(
    console: Console,
    stats: Dict[str, Any],
    visualizer: Any,
    show_distributions: bool = True,
) -> None:
    """Render basic statistics and optional entity/relation distribution tables."""
    stats_table = Table(title="Basic Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="yellow")
    stats_table.add_row("Graph ID", stats["graph_id"])
    stats_table.add_row("Nodes", f"{stats['num_nodes']:,}")
    stats_table.add_row("Edges", f"{stats['num_edges']:,}")
    stats_table.add_row("Density", f"{stats['density']:.4f}")
    stats_table.add_row("Average Degree", f"{stats['average_degree']:.2f}")
    stats_table.add_row("Connected Components", str(stats["num_connected_components"]))
    console.print(stats_table)
    console.print()

    if show_distributions and "entity_type_counts" in stats:
        entity_table = Table(title="Entity Type Distribution")
        entity_table.add_column("Type", style="cyan")
        entity_table.add_column("Count", style="yellow")
        entity_table.add_column("Distribution", style="green")
        total_entities = sum(stats["entity_type_counts"].values())
        for entity_type, count in sorted(
            stats["entity_type_counts"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            percentage = (count / total_entities) * 100
            bar = visualizer._create_bar(count, total_entities, 30, "cyan")
            entity_table.add_row(entity_type, f"{count:,}", f"{bar} {percentage:.1f}%")
        console.print(entity_table)
        console.print()

    if show_distributions and "relation_type_counts" in stats:
        relation_table = Table(title="Relation Type Distribution")
        relation_table.add_column("Type", style="cyan")
        relation_table.add_column("Count", style="yellow")
        relation_table.add_column("Distribution", style="green")
        total_relations = sum(stats["relation_type_counts"].values())
        for relation_type, count in sorted(
            stats["relation_type_counts"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            percentage = (count / total_relations) * 100
            bar = visualizer._create_bar(count, total_relations, 30, "green")
            relation_table.add_row(relation_type, f"{count:,}", f"{bar} {percentage:.1f}%")
        console.print(relation_table)
        console.print()
