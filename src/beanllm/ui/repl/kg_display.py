"""
Knowledge Graph display helpers - query results and quick commands.
"""
from __future__ import annotations

from typing import Any, Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def show_query_results(
    console: Console,
    results: List[Dict[str, Any]],
    query_type: str,
) -> None:
    """Show query results in formatted table."""
    if query_type in ["find_entities_by_type", "find_entities_by_name"]:
        table = Table(title="Entities")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="yellow")
        table.add_column("Type", style="magenta")
        for result in results[:20]:
            table.add_row(
                result.get("id", ""), result.get("name", ""), result.get("type", "")
            )
        console.print(table)
        if len(results) > 20:
            console.print(f"\n[dim]... and {len(results) - 20} more[/dim]")

    elif query_type == "find_related_entities":
        table = Table(title="Related Entities")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="yellow")
        table.add_column("Type", style="magenta")
        table.add_column("Relation", style="green")
        for result in results[:20]:
            table.add_row(
                result.get("id", ""),
                result.get("name", ""),
                result.get("type", ""),
                result.get("relation_type", ""),
            )
        console.print(table)
        if len(results) > 20:
            console.print(f"\n[dim]... and {len(results) - 20} more[/dim]")

    elif query_type == "find_shortest_path":
        if results:
            path = results[0].get("path", [])
            console.print(f"[cyan]Path length: {len(path)}[/cyan]\n")
            console.print(" → ".join(path))

    elif query_type == "get_entity_details":
        if results:
            details = results[0]
            details_table = Table(title="Entity Details")
            details_table.add_column("Field", style="cyan")
            details_table.add_column("Value", style="yellow")
            details_table.add_row("ID", details.get("id", ""))
            details_table.add_row("Name", details.get("name", ""))
            details_table.add_row("Type", details.get("type", ""))
            details_table.add_row(
                "Outgoing Relations",
                str(len(details.get("outgoing_relations", []))),
            )
            details_table.add_row(
                "Incoming Relations",
                str(len(details.get("incoming_relations", []))),
            )
            console.print(details_table)

    else:
        console.print(results)


def show_quick_commands(console: Console, graph_id: str) -> None:
    """Show quick follow-up commands."""
    panel = Panel(
        f"""[cyan]Quick Commands:[/cyan]

1. Query entities:
   await commands.cmd_query(
       graph_id="{graph_id}",
       query_type="find_entities_by_type",
       entity_type="PERSON"
   )

2. Visualize graph:
   await commands.cmd_visualize(graph_id="{graph_id}")

3. Show statistics:
   await commands.cmd_stats(graph_id="{graph_id}")

4. Graph RAG:
   await commands.cmd_graph_rag(
       query="Your question here",
       graph_id="{graph_id}"
   )
""",
        title="[bold]Next Steps[/bold]",
        border_style="green",
    )
    console.print(panel)
