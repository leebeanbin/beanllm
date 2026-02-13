"""
Graph Metrics Visualization Mixin

Knowledge Graph 관련 메트릭 시각화 메서드를 제공하는 Mixin 클래스.
MetricsVisualizer에 mix-in 되어 사용됩니다.

SOLID 원칙:
- SRP: Graph 시각화만 담당
- OCP: 새로운 Graph 메트릭 추가 가능
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

if TYPE_CHECKING:
    from rich.console import Console


class GraphMetricsMixin:
    """Knowledge Graph 관련 메트릭 시각화 Mixin

    MetricsVisualizer에 mix-in 되어, self.console 및
    self._create_bar / self._create_percentage_bar 을
    통해 시각화를 수행합니다.
    """

    console: Console

    def show_graph_network(
        self,
        num_nodes: int,
        num_edges: int,
        density: float,
        num_components: int,
        avg_degree: float,
    ) -> None:
        """
        Show graph network structure overview

        Args:
            num_nodes: Number of nodes
            num_edges: Number of edges
            density: Graph density
            num_components: Number of connected components
            avg_degree: Average node degree
        """
        table = Table(
            title="📊 Graph Network Structure",
            title_style="bold cyan",
            box=box.ROUNDED,
            show_header=False,
        )

        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Value", style="yellow")
        table.add_column("Visualization", style="white")

        # Nodes
        table.add_row(
            "Nodes",
            f"{num_nodes:,}",
            self._create_bar(num_nodes, max(num_nodes, 100), 20, "cyan"),  # type: ignore[attr-defined]
        )

        # Edges
        table.add_row(
            "Edges",
            f"{num_edges:,}",
            self._create_bar(num_edges, max(num_edges, 200), 20, "green"),  # type: ignore[attr-defined]
        )

        # Density
        density_pct = density * 100
        density_status = "🟢 High" if density > 0.5 else "🟡 Medium" if density > 0.2 else "🔴 Low"
        table.add_row(
            "Density",
            f"{density:.4f}",
            f"{self._create_percentage_bar(density_pct, 20, 'magenta')} {density_status}",  # type: ignore[attr-defined]
        )

        # Average degree
        table.add_row(
            "Average Degree",
            f"{avg_degree:.2f}",
            self._create_bar(avg_degree, max(avg_degree, 10), 20, "yellow"),  # type: ignore[attr-defined]
        )

        # Connected components
        component_status = (
            "✅ Connected" if num_components == 1 else f"⚠️  {num_components} components"
        )
        table.add_row("Connected Components", str(num_components), component_status)

        self.console.print()
        self.console.print(table)
        self.console.print()

    def show_entity_distribution(
        self,
        entity_counts: Dict[str, int],
        title: str = "Entity Type Distribution",
        max_display: int = 15,
    ) -> None:
        """
        Show entity type distribution

        Args:
            entity_counts: Entity type -> count mapping
            title: Table title
            max_display: Maximum entity types to display
        """
        if not entity_counts:
            self.console.print("[yellow]No entities found[/yellow]")
            return

        total = sum(entity_counts.values())

        table = Table(
            title=f"📊 {title} (Total: {total:,})",
            title_style="bold cyan",
            box=box.ROUNDED,
        )

        table.add_column("Rank", style="dim", width=5)
        table.add_column("Entity Type", style="cyan", width=20)
        table.add_column("Count", style="yellow", justify="right", width=10)
        table.add_column("Percentage", style="white", justify="right", width=8)
        table.add_column("Distribution", style="white", width=30)

        # Sort by count descending
        sorted_entities = sorted(
            entity_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for i, (entity_type, count) in enumerate(sorted_entities[:max_display], 1):
            percentage = (count / total) * 100
            bar = self._create_bar(count, total, 25, "cyan")  # type: ignore[attr-defined]

            # Emoji for rank
            rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}"

            table.add_row(rank_emoji, entity_type.upper(), f"{count:,}", f"{percentage:.1f}%", bar)

        if len(sorted_entities) > max_display:
            remaining = len(sorted_entities) - max_display
            remaining_count = sum(count for _, count in sorted_entities[max_display:])
            table.add_row(
                "...",
                f"Others ({remaining})",
                f"{remaining_count:,}",
                f"{(remaining_count / total) * 100:.1f}%",
                "[dim]...[/dim]",
            )

        self.console.print()
        self.console.print(table)
        self.console.print()

    def show_relation_distribution(
        self,
        relation_counts: Dict[str, int],
        title: str = "Relation Type Distribution",
        max_display: int = 15,
    ) -> None:
        """
        Show relation type distribution

        Args:
            relation_counts: Relation type -> count mapping
            title: Table title
            max_display: Maximum relation types to display
        """
        if not relation_counts:
            self.console.print("[yellow]No relations found[/yellow]")
            return

        total = sum(relation_counts.values())

        table = Table(
            title=f"📊 {title} (Total: {total:,})",
            title_style="bold green",
            box=box.ROUNDED,
        )

        table.add_column("Rank", style="dim", width=5)
        table.add_column("Relation Type", style="green", width=20)
        table.add_column("Count", style="yellow", justify="right", width=10)
        table.add_column("Percentage", style="white", justify="right", width=8)
        table.add_column("Distribution", style="white", width=30)

        # Sort by count descending
        sorted_relations = sorted(
            relation_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for i, (relation_type, count) in enumerate(sorted_relations[:max_display], 1):
            percentage = (count / total) * 100
            bar = self._create_bar(count, total, 25, "green")  # type: ignore[attr-defined]

            # Emoji for rank
            rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}"

            table.add_row(
                rank_emoji, relation_type.upper(), f"{count:,}", f"{percentage:.1f}%", bar
            )

        if len(sorted_relations) > max_display:
            remaining = len(sorted_relations) - max_display
            remaining_count = sum(count for _, count in sorted_relations[max_display:])
            table.add_row(
                "...",
                f"Others ({remaining})",
                f"{remaining_count:,}",
                f"{(remaining_count / total) * 100:.1f}%",
                "[dim]...[/dim]",
            )

        self.console.print()
        self.console.print(table)
        self.console.print()

    def show_path_visualization(
        self,
        path: List[str],
        entity_names: Optional[Dict[str, str]] = None,
        relation_types: Optional[List[str]] = None,
    ) -> None:
        """
        Show path visualization

        Args:
            path: List of entity IDs in path
            entity_names: Entity ID -> name mapping (optional)
            relation_types: List of relation types along path (optional)
        """
        if not path:
            self.console.print("[yellow]No path found[/yellow]")
            return

        self.console.print()
        self.console.print("[bold cyan]🛤️  Path Visualization[/bold cyan]")
        self.console.print(f"[dim]Length: {len(path)} nodes[/dim]\n")

        # Create path tree
        tree = Tree("🏁 Start")

        for i, entity_id in enumerate(path):
            # Get entity name
            name = entity_names.get(entity_id, entity_id) if entity_names else entity_id

            # Get relation type
            if relation_types and i < len(relation_types):
                relation = relation_types[i]
                label = f"--[{relation}]--> {name}"
            else:
                label = name

            # Add to tree
            if i == 0:
                tree.label = f"🏁 [bold]{name}[/bold]"
            elif i == len(path) - 1:
                tree.add(f"🎯 [bold green]{label}[/bold green]")
            else:
                tree.add(f"📍 {label}")

        self.console.print(tree)
        self.console.print()

        # Summary
        path_str = " → ".join(entity_names.get(eid, eid) if entity_names else eid for eid in path)
        summary = Panel(
            f"[cyan]Path: {path_str}[/cyan]",
            title="Summary",
            border_style="cyan",
        )
        self.console.print(summary)
        self.console.print()
