"""
Orchestrator monitor - real-time monitoring display logic.
"""

from __future__ import annotations

from typing import Any, Optional

from rich import box
from rich.panel import Panel


def create_monitor_display(status: Optional[Any]) -> Panel:
    """모니터링 디스플레이 생성."""
    if not status:
        return Panel(
            "[dim]Connecting to monitor...[/dim]",
            title="📊 Workflow Monitor",
            border_style="cyan",
        )
    progress_pct = status.progress * 100
    bar_width = 40
    filled = int(bar_width * status.progress)
    bar = "█" * filled + "░" * (bar_width - filled)
    lines = [
        f"[bold]Execution ID:[/bold] {status.execution_id}",
        f"[bold]Current Node:[/bold] {status.current_node or 'N/A'}",
        f"\n[bold]Progress:[/bold] {progress_pct:.1f}%",
        f"[cyan]{bar}[/cyan]",
        f"\n[bold]Nodes Completed:[/bold] {len(status.nodes_completed)}",
        f"[bold]Nodes Pending:[/bold] {len(status.nodes_pending)}",
        f"[bold]Elapsed Time:[/bold] {status.elapsed_time:.1f}s",
    ]
    return Panel(
        "\n".join(lines),
        title="📊 Workflow Monitor",
        border_style="cyan",
        box=box.ROUNDED,
    )
