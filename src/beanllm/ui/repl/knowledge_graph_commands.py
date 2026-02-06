"""
KnowledgeGraphCommands - Rich CLI interface for Knowledge Graph Builder
"""

from __future__ import annotations

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

from beanllm.facade.advanced.knowledge_graph_facade import KnowledgeGraph
from beanllm.ui.visualizers.metrics_viz import MetricsVisualizer
from beanllm.utils.core.logger import get_logger

logger = get_logger(__name__)


class KnowledgeGraphCommands:
    """
    Rich CLI Commands for Knowledge Graph Builder

    Provides interactive commands for:
    - Building knowledge graphs
    - Querying graphs
    - Visualizing graphs
    - Graph statistics
    - Graph-based RAG
    - Managing graphs

    Example:
        ```python
        from beanllm.ui.repl.knowledge_graph_commands import KnowledgeGraphCommands

        commands = KnowledgeGraphCommands()

        # Build graph
        await commands.cmd_build_graph(
            documents=["Apple was founded by Steve Jobs."],
            graph_id="tech_companies"
        )

        # Query graph
        await commands.cmd_query(
            graph_id="tech_companies",
            query_type="find_entities_by_type",
            entity_type="PERSON"
        )

        # Graph RAG
        await commands.cmd_graph_rag(
            query="Who founded Apple?",
            graph_id="tech_companies"
        )
        ```
    """

    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        console: Optional[Console] = None,
    ) -> None:
        """
        Initialize knowledge graph commands

        Args:
            knowledge_graph: Optional KnowledgeGraph facade (for DI)
            console: Optional Rich console
        """
        self._kg = knowledge_graph or KnowledgeGraph()
        self.console = console or Console()
        self._visualizer = MetricsVisualizer(console=self.console)

        logger.info("KnowledgeGraphCommands initialized")

    # ===== CLI Commands =====

    async def cmd_build_graph(
        self,
        documents: Optional[List[str]] = None,
        file_paths: Optional[List[str]] = None,
        graph_id: Optional[str] = None,
        entity_types: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
        persist_to_neo4j: bool = False,
    ) -> None:
        """
        Build knowledge graph from documents

        Args:
            documents: Document texts
            file_paths: Paths to document files
            graph_id: Graph ID (optional, auto-generated)
            entity_types: Entity types to extract
            relation_types: Relation types to extract
            persist_to_neo4j: Save to Neo4j (default: False)

        Example:
            ```python
            await commands.cmd_build_graph(
                documents=["Apple was founded by Steve Jobs in 1976."],
                graph_id="tech_companies",
                entity_types=["person", "organization"],
                persist_to_neo4j=True
            )
            ```
        """
        self.console.print("\n[bold cyan]📊 Building Knowledge Graph[/bold cyan]\n")

        # Load documents from files
        if file_paths:
            # TODO: Load documents from files
            pass

        if not documents:
            self.console.print("[red]❌ No documents provided[/red]")
            return

        # Show configuration
        config_table = Table(title="Configuration")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="yellow")

        config_table.add_row("Documents", str(len(documents)))
        config_table.add_row("Graph ID", graph_id or "(auto-generated)")
        config_table.add_row("Entity Types", ", ".join(entity_types) if entity_types else "All")
        config_table.add_row(
            "Relation Types", ", ".join(relation_types) if relation_types else "All"
        )
        config_table.add_row("Persist to Neo4j", "✅ Yes" if persist_to_neo4j else "❌ No")

        self.console.print(config_table)
        self.console.print()

        # Build graph with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("Building knowledge graph...", total=len(documents))

            try:
                response = await self._kg.build_graph(
                    documents=documents,
                    graph_id=graph_id,
                    entity_types=entity_types,
                    relation_types=relation_types,
                    persist_to_neo4j=persist_to_neo4j,
                )

                progress.update(task, completed=len(documents))

                # Display results
                self.console.print("\n[bold green]✅ Knowledge Graph Built![/bold green]\n")

                result_table = Table(title=f"Graph: {response.graph_id}")
                result_table.add_column("Metric", style="cyan")
                result_table.add_column("Value", style="yellow")

                result_table.add_row("Graph ID", response.graph_id)
                result_table.add_row("Nodes", f"{response.num_nodes:,}")
                result_table.add_row("Edges", f"{response.num_edges:,}")
                result_table.add_row("Density", f"{response.density:.4f}")
                result_table.add_row("Connected Components", str(response.num_connected_components))

                self.console.print(result_table)
                self.console.print()

                # Show quick commands
                self._show_quick_commands(response.graph_id)

            except Exception as e:
                progress.stop()
                self.console.print(f"\n[red]❌ Failed to build graph: {e}[/red]")
                logger.error(f"Failed to build graph: {e}", exc_info=True)

    async def cmd_query(
        self,
        graph_id: str,
        query_type: str = "find_entities_by_type",
        entity_type: Optional[str] = None,
        name: Optional[str] = None,
        entity_id: Optional[str] = None,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        relation_type: Optional[str] = None,
        max_hops: int = 1,
        fuzzy: bool = False,
        cypher_query: Optional[str] = None,
    ) -> None:
        """
        Query knowledge graph

        Args:
            graph_id: Graph ID
            query_type: Query type
                - "find_entities_by_type": Find by entity type
                - "find_entities_by_name": Find by name
                - "find_related_entities": Find related entities
                - "find_shortest_path": Find shortest path
                - "get_entity_details": Get entity details
                - "cypher": Cypher query (Neo4j)
            entity_type: Entity type (for find_entities_by_type)
            name: Entity name (for find_entities_by_name)
            entity_id: Entity ID (for find_related_entities, get_entity_details)
            source_id: Source entity ID (for find_shortest_path)
            target_id: Target entity ID (for find_shortest_path)
            relation_type: Relation type filter (optional)
            max_hops: Maximum hops (default: 1)
            fuzzy: Fuzzy name matching (default: False)
            cypher_query: Cypher query string (for cypher)

        Example:
            ```python
            # Find all persons
            await commands.cmd_query(
                graph_id="tech_companies",
                query_type="find_entities_by_type",
                entity_type="PERSON"
            )

            # Find shortest path
            await commands.cmd_query(
                graph_id="tech_companies",
                query_type="find_shortest_path",
                source_id="steve_jobs",
                target_id="apple"
            )
            ```
        """
        self.console.print(f"\n[bold cyan]🔍 Querying Graph: {graph_id}[/bold cyan]\n")

        # Build params
        params: Dict[str, Any] = {}

        if query_type == "find_entities_by_type":
            if not entity_type:
                self.console.print("[red]❌ entity_type is required[/red]")
                return
            params["entity_type"] = entity_type

        elif query_type == "find_entities_by_name":
            if not name:
                self.console.print("[red]❌ name is required[/red]")
                return
            params["name"] = name
            params["fuzzy"] = fuzzy

        elif query_type == "find_related_entities":
            if not entity_id:
                self.console.print("[red]❌ entity_id is required[/red]")
                return
            params["entity_id"] = entity_id
            params["relation_type"] = relation_type
            params["max_hops"] = max_hops

        elif query_type == "find_shortest_path":
            if not source_id or not target_id:
                self.console.print("[red]❌ source_id and target_id are required[/red]")
                return
            params["source_id"] = source_id
            params["target_id"] = target_id

        elif query_type == "get_entity_details":
            if not entity_id:
                self.console.print("[red]❌ entity_id is required[/red]")
                return
            params["entity_id"] = entity_id

        elif query_type == "cypher":
            if not cypher_query:
                self.console.print("[red]❌ cypher_query is required[/red]")
                return

        # Execute query
        try:
            with self.console.status("[bold green]Executing query..."):
                if query_type == "cypher":
                    response = await self._kg.query_graph(
                        graph_id=graph_id,
                        query_type=query_type,
                        query=cypher_query,
                        params=params,
                    )
                else:
                    response = await self._kg.query_graph(
                        graph_id=graph_id,
                        query_type=query_type,
                        params=params,
                    )

            # Display results
            self.console.print("\n[bold green]✅ Query Completed![/bold green]")
            self.console.print(f"[cyan]Results: {response.num_results}[/cyan]\n")

            if response.results:
                self._show_query_results(response.results, query_type)
            else:
                self.console.print("[yellow]No results found[/yellow]")

        except Exception as e:
            self.console.print(f"\n[red]❌ Query failed: {e}[/red]")
            logger.error(f"Query failed: {e}", exc_info=True)

    async def cmd_visualize(
        self,
        graph_id: str,
        format: str = "ascii",
    ) -> None:
        """
        Visualize knowledge graph

        Args:
            graph_id: Graph ID
            format: Visualization format (default: "ascii")

        Example:
            ```python
            await commands.cmd_visualize(graph_id="tech_companies")
            ```
        """
        self.console.print(f"\n[bold cyan]📊 Visualizing Graph: {graph_id}[/bold cyan]\n")

        try:
            with self.console.status("[bold green]Generating visualization..."):
                diagram = await self._kg.visualize_graph(graph_id)

            # Display ASCII diagram
            panel = Panel(
                diagram,
                title=f"[bold]Graph: {graph_id}[/bold]",
                border_style="cyan",
            )
            self.console.print(panel)

        except Exception as e:
            self.console.print(f"\n[red]❌ Visualization failed: {e}[/red]")
            logger.error(f"Visualization failed: {e}", exc_info=True)

    async def cmd_stats(
        self,
        graph_id: str,
        show_distributions: bool = True,
    ) -> None:
        """
        Show graph statistics

        Args:
            graph_id: Graph ID
            show_distributions: Show entity/relation distributions (default: True)

        Example:
            ```python
            await commands.cmd_stats(graph_id="tech_companies")
            ```
        """
        self.console.print(f"\n[bold cyan]📈 Graph Statistics: {graph_id}[/bold cyan]\n")

        try:
            with self.console.status("[bold green]Calculating statistics..."):
                stats = await self._kg.get_graph_stats(graph_id)

            # Basic statistics
            stats_table = Table(title="Basic Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="yellow")

            stats_table.add_row("Graph ID", stats["graph_id"])
            stats_table.add_row("Nodes", f"{stats['num_nodes']:,}")
            stats_table.add_row("Edges", f"{stats['num_edges']:,}")
            stats_table.add_row("Density", f"{stats['density']:.4f}")
            stats_table.add_row("Average Degree", f"{stats['average_degree']:.2f}")
            stats_table.add_row("Connected Components", str(stats["num_connected_components"]))

            self.console.print(stats_table)
            self.console.print()

            # Entity type distribution
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
                    bar = self._visualizer._create_bar(count, total_entities, 30, "cyan")
                    entity_table.add_row(entity_type, f"{count:,}", f"{bar} {percentage:.1f}%")

                self.console.print(entity_table)
                self.console.print()

            # Relation type distribution
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
                    bar = self._visualizer._create_bar(count, total_relations, 30, "green")
                    relation_table.add_row(relation_type, f"{count:,}", f"{bar} {percentage:.1f}%")

                self.console.print(relation_table)
                self.console.print()

        except Exception as e:
            self.console.print(f"\n[red]❌ Failed to get statistics: {e}[/red]")
            logger.error(f"Failed to get statistics: {e}", exc_info=True)

    async def cmd_graph_rag(
        self,
        query: str,
        graph_id: str,
        show_details: bool = True,
    ) -> None:
        """
        Execute Graph RAG query

        Args:
            query: User query
            graph_id: Graph ID
            show_details: Show detailed results (default: True)

        Example:
            ```python
            await commands.cmd_graph_rag(
                query="Who founded Apple?",
                graph_id="tech_companies"
            )
            ```
        """
        self.console.print("\n[bold cyan]🤖 Graph RAG Query[/bold cyan]")
        self.console.print(f"[dim]Query: {query}[/dim]")
        self.console.print(f"[dim]Graph: {graph_id}[/dim]\n")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Processing query...", total=None)

                response = await self._kg.graph_rag(
                    query=query,
                    graph_id=graph_id,
                )

                progress.update(task, completed=True)

            # Display answer
            answer = await self._kg.ask(query=query, graph_id=graph_id)

            answer_panel = Panel(
                answer,
                title="[bold]Answer[/bold]",
                border_style="green",
            )
            self.console.print(answer_panel)
            self.console.print()

            # Show details
            if show_details:
                entity_count = len(response.entity_results) if response.entity_results else 0
                path_count = len(response.path_results) if response.path_results else 0
                hybrid_count = len(response.hybrid_results) if response.hybrid_results else 0
                self.console.print(f"[cyan]Entity Results: {entity_count}[/cyan]")
                self.console.print(f"[cyan]Path Results: {path_count}[/cyan]")
                self.console.print(f"[cyan]Hybrid Results: {hybrid_count}[/cyan]\n")

                # Show top hybrid results
                if response.hybrid_results:
                    results_table = Table(title="Top Results")
                    results_table.add_column("Rank", style="cyan")
                    results_table.add_column("Entity", style="yellow")
                    results_table.add_column("Type", style="magenta")
                    results_table.add_column("Score", style="green")

                    for i, result in enumerate(response.hybrid_results[:10], 1):
                        entity = result.get("entity", {})
                        name = entity.get("name", "Unknown")
                        entity_type = entity.get("type", "UNKNOWN")
                        score = result.get("score", 0.0)

                        results_table.add_row(str(i), name, entity_type, f"{score:.3f}")

                    self.console.print(results_table)
                    self.console.print()

        except Exception as e:
            self.console.print(f"\n[red]❌ Graph RAG failed: {e}[/red]")
            logger.error(f"Graph RAG failed: {e}", exc_info=True)

    async def cmd_list_graphs(self) -> None:
        """
        List all knowledge graphs

        Example:
            ```python
            await commands.cmd_list_graphs()
            ```
        """
        self.console.print("\n[bold cyan]📚 Knowledge Graphs[/bold cyan]\n")

        try:
            # Get service to access list_graphs

            # Access internal service (temporary workaround)
            import asyncio

            service = self._kg._handler._service
            graphs = asyncio.run(service.list_graphs())

            if not graphs:
                self.console.print("[yellow]No graphs found[/yellow]")
                return

            graphs_table = Table(title=f"Total Graphs: {len(graphs)}")
            graphs_table.add_column("Index", style="cyan")
            graphs_table.add_column("Graph ID", style="yellow")
            graphs_table.add_column("Nodes", style="green")
            graphs_table.add_column("Edges", style="blue")

            for i, graph_info in enumerate(graphs, 1):
                graphs_table.add_row(
                    str(i),
                    str(graph_info.get("id", "")),
                    str(graph_info.get("num_nodes", 0)),
                    str(graph_info.get("num_edges", 0)),
                )

            self.console.print(graphs_table)
            self.console.print()

        except Exception as e:
            self.console.print(f"\n[red]❌ Failed to list graphs: {e}[/red]")
            logger.error(f"Failed to list graphs: {e}", exc_info=True)

    # ===== Helper Methods =====

    def _show_query_results(
        self,
        results: List[Dict[str, Any]],
        query_type: str,
    ) -> None:
        """Show query results in formatted table"""

        if query_type in ["find_entities_by_type", "find_entities_by_name"]:
            # Entity results
            table = Table(title="Entities")
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="yellow")
            table.add_column("Type", style="magenta")

            for result in results[:20]:  # Show first 20
                table.add_row(result.get("id", ""), result.get("name", ""), result.get("type", ""))

            self.console.print(table)

            if len(results) > 20:
                self.console.print(f"\n[dim]... and {len(results) - 20} more[/dim]")

        elif query_type == "find_related_entities":
            # Related entities
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

            self.console.print(table)

            if len(results) > 20:
                self.console.print(f"\n[dim]... and {len(results) - 20} more[/dim]")

        elif query_type == "find_shortest_path":
            # Path results
            if results:
                path = results[0].get("path", [])
                self.console.print(f"[cyan]Path length: {len(path)}[/cyan]\n")
                self.console.print(" → ".join(path))

        elif query_type == "get_entity_details":
            # Entity details
            if results:
                details = results[0]

                details_table = Table(title="Entity Details")
                details_table.add_column("Field", style="cyan")
                details_table.add_column("Value", style="yellow")

                details_table.add_row("ID", details.get("id", ""))
                details_table.add_row("Name", details.get("name", ""))
                details_table.add_row("Type", details.get("type", ""))
                details_table.add_row(
                    "Outgoing Relations", str(len(details.get("outgoing_relations", [])))
                )
                details_table.add_row(
                    "Incoming Relations", str(len(details.get("incoming_relations", [])))
                )

                self.console.print(details_table)

        else:
            # Generic results
            self.console.print(results)

    def _show_quick_commands(self, graph_id: str) -> None:
        """Show quick follow-up commands"""

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

        self.console.print(panel)
