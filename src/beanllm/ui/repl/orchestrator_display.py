"""
Orchestrator display helpers - workflow/execution/analytics formatting and tables.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from rich import box
from rich.console import Console
from rich.table import Table


def status_badge(status: str) -> str:
    """상태 뱃지 생성."""
    badges = {
        "completed": "[green]✓ COMPLETED[/green]",
        "failed": "[red]✗ FAILED[/red]",
        "running": "[yellow]⟳ RUNNING[/yellow]",
        "pending": "[dim]○ PENDING[/dim]",
    }
    return badges.get(status.lower(), f"[dim]{status}[/dim]")


def format_workflow_info(workflow: Any) -> str:
    """워크플로우 정보 포맷팅."""
    lines = [
        f"[bold]Workflow ID:[/bold] {workflow.workflow_id}",
        f"[bold]Name:[/bold] {workflow.workflow_name}",
        f"[bold]Strategy:[/bold] {workflow.strategy}",
        f"[bold]Nodes:[/bold] {workflow.num_nodes}",
        f"[bold]Edges:[/bold] {workflow.num_edges}",
        f"[bold]Created:[/bold] {workflow.created_at}",
    ]
    if workflow.metadata:
        lines.append(f"[bold]Metadata:[/bold] {json.dumps(workflow.metadata, indent=2)}")
    return "\n".join(lines)


def format_execution_result(result: Any) -> str:
    """실행 결과 포맷팅."""
    lines = [
        f"[bold]Execution ID:[/bold] {result.execution_id}",
        f"[bold]Workflow ID:[/bold] {result.workflow_id}",
        f"[bold]Status:[/bold] {status_badge(result.status)}",
        f"[bold]Execution Time:[/bold] {result.execution_time:.2f}s",
    ]
    if result.result:
        lines.append(f"\n[bold]Result:[/bold]\n{json.dumps(result.result, indent=2)}")
    if result.error:
        lines.append(f"\n[bold red]Error:[/bold red] {result.error}")
    return "\n".join(lines)


def format_analytics(analytics: Any) -> str:
    """분석 결과 포맷팅."""
    lines = [
        f"[bold]Total Executions:[/bold] {analytics.total_executions}",
        f"[bold]Avg Execution Time:[/bold] {analytics.avg_execution_time:.2f}s",
        f"[bold]Success Rate:[/bold] {analytics.success_rate * 100:.1f}%",
        f"[bold]Bottlenecks:[/bold] {len(analytics.bottlenecks)}",
    ]
    if analytics.agent_utilization:
        lines.append(
            f"\n[bold]Agent Utilization:[/bold]\n"
            f"{json.dumps(analytics.agent_utilization, indent=2)}"
        )
    return "\n".join(lines)


def display_node_results(console: Console, node_results: List[Dict]) -> None:
    """노드 실행 결과 테이블 출력."""
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    table.add_column("Node", style="yellow")
    table.add_column("Status", style="white")
    table.add_column("Duration", style="cyan")
    table.add_column("Output", style="dim", max_width=50)
    for node in node_results:
        table.add_row(
            node.get("node_id", ""),
            status_badge(node.get("status", "unknown")),
            f"{node.get('duration_ms', 0):.0f}ms",
            str(node.get("output", ""))[:50],
        )
    console.print(table)


def display_bottlenecks(console: Console, bottlenecks: List[Dict]) -> None:
    """병목 테이블 출력."""
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold yellow")
    table.add_column("Node ID", style="yellow")
    table.add_column("Duration", style="red")
    table.add_column("% of Total", style="cyan")
    table.add_column("Recommendation", style="dim", max_width=40)
    for bn in bottlenecks:
        table.add_row(
            bn["node_id"],
            f"{bn['duration_ms']:.0f}ms",
            f"{bn['percentage']:.1f}%",
            bn.get("recommendation", ""),
        )
    console.print(table)
