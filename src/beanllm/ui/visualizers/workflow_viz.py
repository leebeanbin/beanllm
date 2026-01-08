"""
Workflow Visualizer - 워크플로우 실행 시각화
SOLID 원칙:
- SRP: 워크플로우 시각화만 담당
- OCP: 새로운 시각화 방법 추가 가능
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskID, TextColumn
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from beanllm.ui.components import Badge, StatusIcon
from beanllm.ui.console import get_console


class WorkflowVisualizer:
    """
    워크플로우 시각화

    책임:
    - 워크플로우 구조 다이어그램
    - 실행 진행 상황 시각화
    - 노드 상태 트리

    Example:
        ```python
        viz = WorkflowVisualizer()

        # Workflow diagram
        viz.show_diagram(diagram_ascii)

        # Execution progress
        viz.show_progress(
            workflow_id="wf-123",
            nodes_completed=["node1", "node2"],
            nodes_running=["node3"],
            nodes_pending=["node4", "node5"]
        )

        # Node states tree
        viz.show_node_states(node_states)
        ```
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        """
        Args:
            console: Rich Console (optional)
        """
        self.console = console or get_console()

    def show_diagram(
        self,
        diagram: str,
        title: str = "Workflow Diagram",
        border_style: str = "cyan",
    ) -> None:
        """
        워크플로우 다이어그램 출력

        Args:
            diagram: ASCII 다이어그램
            title: 제목
            border_style: 테두리 스타일
        """
        panel = Panel(
            diagram,
            title=f"🎨 {title}",
            border_style=border_style,
            box=box.ROUNDED,
        )
        self.console.print(panel)

    def show_progress(
        self,
        workflow_id: str,
        total_nodes: int,
        nodes_completed: List[str],
        nodes_running: List[str],
        nodes_pending: List[str],
        nodes_failed: Optional[List[str]] = None,
        elapsed_time: float = 0.0,
    ) -> None:
        """
        실행 진행 상황 테이블 출력

        Args:
            workflow_id: 워크플로우 ID
            total_nodes: 총 노드 수
            nodes_completed: 완료된 노드 목록
            nodes_running: 실행 중인 노드 목록
            nodes_pending: 대기 중인 노드 목록
            nodes_failed: 실패한 노드 목록
            elapsed_time: 경과 시간 (초)
        """
        nodes_failed = nodes_failed or []

        # Calculate progress
        completed_count = len(nodes_completed)
        failed_count = len(nodes_failed)
        total_finished = completed_count + failed_count
        progress_pct = (total_finished / total_nodes * 100) if total_nodes > 0 else 0

        # Create progress table
        table = Table(
            title=f"📊 Workflow Progress: {workflow_id}",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Metric", style="bold white", width=20)
        table.add_column("Value", style="cyan", width=40)

        # Progress bar
        bar_width = 30
        filled = int(bar_width * (total_finished / total_nodes)) if total_nodes > 0 else 0
        bar = "[green]" + "█" * filled + "[/green]" + "[dim]░[/dim]" * (bar_width - filled)

        table.add_row("Progress", f"{bar} {progress_pct:.1f}%")
        table.add_row("Total Nodes", str(total_nodes))
        table.add_row(
            "Completed",
            f"[green]{completed_count}[/green] ({', '.join(nodes_completed[:3])}" +
            (f", +{len(nodes_completed) - 3} more" if len(nodes_completed) > 3 else "") + ")"
            if nodes_completed else "[dim]None[/dim]",
        )
        table.add_row(
            "Running",
            f"[yellow]{len(nodes_running)}[/yellow] ({', '.join(nodes_running)})"
            if nodes_running
            else "[dim]None[/dim]",
        )
        table.add_row(
            "Pending",
            f"[dim]{len(nodes_pending)}[/dim] ({', '.join(nodes_pending[:3])}" +
            (f", +{len(nodes_pending) - 3} more" if len(nodes_pending) > 3 else "") + ")"
            if nodes_pending else "[dim]None[/dim]",
        )

        if nodes_failed:
            table.add_row(
                "Failed",
                f"[red]{failed_count}[/red] ({', '.join(nodes_failed)})",
            )

        table.add_row("Elapsed Time", f"{elapsed_time:.1f}s")

        self.console.print(table)

    def show_node_states(
        self,
        node_states: Dict[str, Any],
        title: str = "Node States",
    ) -> None:
        """
        노드 상태 트리 출력

        Args:
            node_states: {node_id: state_dict}
            title: 제목
        """
        tree = Tree(f"🌲 {title}", guide_style="dim")

        for node_id, state in node_states.items():
            # Node status
            status = state.get("status", "unknown")
            status_icon = self._get_status_icon(status)
            status_text = f"{status_icon} {node_id}"

            # Add node branch
            node_branch = tree.add(status_text)

            # Add details
            if state.get("start_time"):
                node_branch.add(f"[dim]Started: {state['start_time']}[/dim]")

            if state.get("end_time"):
                node_branch.add(f"[dim]Ended: {state['end_time']}[/dim]")

            if state.get("duration_ms"):
                node_branch.add(f"[cyan]Duration: {state['duration_ms']:.0f}ms[/cyan]")

            if state.get("error"):
                node_branch.add(f"[red]Error: {state['error']}[/red]")

            if state.get("output"):
                output_str = str(state["output"])[:50]
                node_branch.add(f"[green]Output: {output_str}...[/green]")

        panel = Panel(
            tree,
            border_style="cyan",
            box=box.ROUNDED,
        )
        self.console.print(panel)

    def show_execution_timeline(
        self,
        events: List[Dict[str, Any]],
        title: str = "Execution Timeline",
        max_events: int = 20,
    ) -> None:
        """
        실행 타임라인 테이블 출력

        Args:
            events: 이벤트 목록 [{timestamp, event_type, node_id, data}, ...]
            title: 제목
            max_events: 최대 이벤트 수
        """
        if not events:
            self.console.print("[dim]No events to display[/dim]")
            return

        # Create timeline table
        table = Table(
            title=f"⏱️  {title}",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Timestamp", style="dim", width=20)
        table.add_column("Event", style="yellow", width=20)
        table.add_column("Node", style="white", width=15)
        table.add_column("Details", style="dim", max_width=40)

        # Show recent events
        recent_events = events[-max_events:]

        for event in recent_events:
            timestamp = event.get("timestamp", "")
            event_type = event.get("event_type", "")
            node_id = event.get("node_id", "N/A")
            data = event.get("data", {})

            # Format event type
            event_icon = self._get_event_icon(event_type)
            event_text = f"{event_icon} {event_type}"

            # Format details
            details = ", ".join([f"{k}={v}" for k, v in data.items()])

            table.add_row(timestamp, event_text, node_id, details[:40])

        self.console.print(table)

    def show_bottlenecks(
        self,
        bottlenecks: List[Dict[str, Any]],
        title: str = "Performance Bottlenecks",
    ) -> None:
        """
        병목 분석 테이블 출력

        Args:
            bottlenecks: 병목 목록 [{node_id, duration_ms, percentage, recommendation}, ...]
            title: 제목
        """
        if not bottlenecks:
            self.console.print("[green]✓ No bottlenecks detected[/green]")
            return

        table = Table(
            title=f"⚠️  {title}",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold yellow",
        )

        table.add_column("Rank", style="dim", width=6)
        table.add_column("Node ID", style="yellow", width=20)
        table.add_column("Duration", style="red", width=12)
        table.add_column("% of Total", style="cyan", width=12)
        table.add_column("Recommendation", style="dim", max_width=40)

        for i, bn in enumerate(bottlenecks, 1):
            table.add_row(
                f"#{i}",
                bn.get("node_id", ""),
                f"{bn.get('duration_ms', 0):.0f}ms",
                f"{bn.get('percentage', 0):.1f}%",
                bn.get("recommendation", ""),
            )

        self.console.print(table)

    def show_agent_utilization(
        self,
        agent_utilization: Dict[str, float],
        title: str = "Agent Utilization",
    ) -> None:
        """
        에이전트 활용도 테이블 출력

        Args:
            agent_utilization: {agent_id: success_rate}
            title: 제목
        """
        if not agent_utilization:
            self.console.print("[dim]No utilization data available[/dim]")
            return

        table = Table(
            title=f"📈 {title}",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Agent ID", style="white", width=25)
        table.add_column("Success Rate", style="green", width=15)
        table.add_column("Utilization Bar", style="cyan", width=40)

        for agent_id, success_rate in sorted(
            agent_utilization.items(), key=lambda x: x[1], reverse=True
        ):
            # Success rate bar
            bar_width = 30
            filled = int(bar_width * success_rate)
            bar = "[green]█[/green]" * filled + "[dim]░[/dim]" * (bar_width - filled)

            # Success rate badge
            rate_pct = success_rate * 100
            if rate_pct >= 90:
                rate_badge = f"[green]{rate_pct:.1f}%[/green]"
            elif rate_pct >= 70:
                rate_badge = f"[yellow]{rate_pct:.1f}%[/yellow]"
            else:
                rate_badge = f"[red]{rate_pct:.1f}%[/red]"

            table.add_row(agent_id, rate_badge, bar)

        self.console.print(table)

    def show_cost_breakdown(
        self,
        cost_breakdown: Dict[str, float],
        title: str = "Cost Breakdown",
    ) -> None:
        """
        비용 분석 테이블 출력

        Args:
            cost_breakdown: {node_id: estimated_cost}
            title: 제목
        """
        if not cost_breakdown:
            self.console.print("[dim]No cost data available[/dim]")
            return

        total_cost = sum(cost_breakdown.values())

        table = Table(
            title=f"💰 {title}",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Node ID", style="white", width=25)
        table.add_column("Cost", style="green", width=15)
        table.add_column("% of Total", style="cyan", width=15)

        for node_id, cost in sorted(
            cost_breakdown.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = (cost / total_cost * 100) if total_cost > 0 else 0

            table.add_row(
                node_id,
                f"${cost:.4f}",
                f"{percentage:.1f}%",
            )

        # Add total row
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]${total_cost:.4f}[/bold]",
            "[bold]100.0%[/bold]",
        )

        self.console.print(table)

    def show_workflow_summary(
        self,
        workflow_id: str,
        workflow_name: str,
        num_nodes: int,
        num_edges: int,
        strategy: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        워크플로우 요약 정보 출력

        Args:
            workflow_id: 워크플로우 ID
            workflow_name: 워크플로우 이름
            num_nodes: 노드 수
            num_edges: 엣지 수
            strategy: 전략
            metadata: 추가 메타데이터
        """
        table = Table(
            title=f"📋 Workflow Summary: {workflow_name}",
            box=box.ROUNDED,
            show_header=False,
        )

        table.add_column("Property", style="bold white", width=20)
        table.add_column("Value", style="cyan", width=50)

        table.add_row("Workflow ID", workflow_id)
        table.add_row("Name", workflow_name)
        table.add_row("Strategy", strategy)
        table.add_row("Nodes", str(num_nodes))
        table.add_row("Edges", str(num_edges))

        if metadata:
            table.add_row(
                "Start Nodes",
                str(metadata.get("start_nodes", "N/A")),
            )
            table.add_row(
                "End Nodes",
                str(metadata.get("end_nodes", "N/A")),
            )

        self.console.print(table)

    # ========================================
    # Helper Methods
    # ========================================

    def _get_status_icon(self, status: str) -> str:
        """상태 아이콘 반환"""
        icons = {
            "completed": "[green]✓[/green]",
            "failed": "[red]✗[/red]",
            "running": "[yellow]⟳[/yellow]",
            "pending": "[dim]○[/dim]",
            "skipped": "[dim]⊘[/dim]",
        }
        return icons.get(status.lower(), "[dim]?[/dim]")

    def _get_event_icon(self, event_type: str) -> str:
        """이벤트 아이콘 반환"""
        icons = {
            "workflow_start": "▶️",
            "workflow_end": "⏹️",
            "node_start": "▶",
            "node_end": "✓",
            "node_error": "✗",
            "edge_traversed": "→",
            "state_changed": "🔄",
        }
        return icons.get(event_type.lower(), "•")


# ========================================
# Convenience Functions
# ========================================


def show_workflow_diagram(
    diagram: str,
    title: str = "Workflow Diagram",
    console: Optional[Console] = None,
) -> None:
    """
    워크플로우 다이어그램 빠르게 출력

    Args:
        diagram: ASCII 다이어그램
        title: 제목
        console: Rich Console (optional)
    """
    viz = WorkflowVisualizer(console=console)
    viz.show_diagram(diagram, title=title)


def show_execution_progress(
    workflow_id: str,
    total_nodes: int,
    nodes_completed: List[str],
    nodes_running: List[str],
    nodes_pending: List[str],
    elapsed_time: float = 0.0,
    console: Optional[Console] = None,
) -> None:
    """
    실행 진행 상황 빠르게 출력

    Args:
        workflow_id: 워크플로우 ID
        total_nodes: 총 노드 수
        nodes_completed: 완료된 노드 목록
        nodes_running: 실행 중인 노드 목록
        nodes_pending: 대기 중인 노드 목록
        elapsed_time: 경과 시간
        console: Rich Console (optional)
    """
    viz = WorkflowVisualizer(console=console)
    viz.show_progress(
        workflow_id=workflow_id,
        total_nodes=total_nodes,
        nodes_completed=nodes_completed,
        nodes_running=nodes_running,
        nodes_pending=nodes_pending,
        elapsed_time=elapsed_time,
    )


def show_workflow_analytics(
    bottlenecks: List[Dict[str, Any]],
    agent_utilization: Dict[str, float],
    cost_breakdown: Dict[str, float],
    console: Optional[Console] = None,
) -> None:
    """
    워크플로우 분석 결과 빠르게 출력

    Args:
        bottlenecks: 병목 목록
        agent_utilization: 에이전트 활용도
        cost_breakdown: 비용 분석
        console: Rich Console (optional)
    """
    viz = WorkflowVisualizer(console=console)

    viz.show_bottlenecks(bottlenecks)
    viz.console.print()
    viz.show_agent_utilization(agent_utilization)
    viz.console.print()
    viz.show_cost_breakdown(cost_breakdown)
