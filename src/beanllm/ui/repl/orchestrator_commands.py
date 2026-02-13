"""
Orchestrator Commands - Rich CLI 인터페이스
SOLID 원칙:
- SRP: CLI 명령어 처리만 담당
- DIP: Facade 인터페이스에 의존
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from beanllm.facade.advanced.orchestrator_facade import Orchestrator
from beanllm.ui.components import OutputBlock, StatusIcon
from beanllm.ui.console import get_console
from beanllm.ui.repl.orchestrator_display import (
    display_bottlenecks,
    display_node_results,
    format_analytics,
    format_execution_result,
    format_workflow_info,
)
from beanllm.ui.repl.orchestrator_monitor import create_monitor_display
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class OrchestratorCommands:
    """
    Multi-Agent 워크플로우 오케스트레이터 CLI 명령어 모음

    책임:
    - CLI 명령어 파싱 및 실행
    - Rich 포맷팅된 출력
    - 실시간 진행상황 표시

    Example:
        ```python
        # In REPL
        commands = OrchestratorCommands()

        # List templates
        await commands.cmd_templates()

        # Create workflow
        await commands.cmd_create(
            name="Research Pipeline",
            strategy="research_write",
            config={"researcher_id": "r1", "writer_id": "w1"}
        )

        # Execute workflow
        await commands.cmd_execute(
            workflow_id="wf-123",
            agents=agents_dict,
            task="Research AI trends"
        )

        # Monitor execution
        await commands.cmd_monitor(
            workflow_id="wf-123",
            execution_id="exec-456"
        )

        # Analyze performance
        await commands.cmd_analyze(workflow_id="wf-123")

        # Visualize workflow
        await commands.cmd_visualize(workflow_id="wf-123")
        ```
    """

    def __init__(
        self,
        orchestrator: Optional[Orchestrator] = None,
        console: Optional[Console] = None,
    ) -> None:
        """
        Args:
            orchestrator: Orchestrator 인스턴스 (optional, 자동 생성됨)
            console: Rich Console (optional)
        """
        self.console = console or get_console()
        self._orchestrator = orchestrator or Orchestrator()
        self._workflows: Dict[str, Any] = {}  # workflow_id -> workflow_info
        self._executions: Dict[str, Any] = {}  # execution_id -> execution_info

    # ========================================
    # Command: List Templates
    # ========================================

    async def cmd_templates(self) -> None:
        """
        사전 정의된 워크플로우 템플릿 목록 출력

        Example:
            ```
            await cmd_templates()
            ```
        """
        self.console.print(f"\n{StatusIcon.info()} [cyan]워크플로우 템플릿 조회 중...[/cyan]")

        try:
            templates = await self._orchestrator.get_templates()

            # Create table
            table = Table(
                title="📋 Workflow Templates",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Strategy", style="bold yellow", width=20)
            table.add_column("Name", style="bold white", width=25)
            table.add_column("Description", style="dim", width=50)
            table.add_column("Parameters", style="green")

            for strategy, info in templates.items():
                table.add_row(
                    strategy,
                    info["name"],
                    info["description"],
                    "\n".join([f"• {p}" for p in info["params"]]),
                )

            self.console.print(table)
            self.console.print(
                f"\n{StatusIcon.success()} [green]{len(templates)} templates available[/green]\n"
            )

        except Exception as e:
            self.console.print(f"{StatusIcon.error()} [red]Error: {e}[/red]")
            logger.error(f"Failed to get templates: {e}")

    # ========================================
    # Command: Create Workflow
    # ========================================

    async def cmd_create(
        self,
        name: str,
        strategy: str = "custom",
        config: Optional[Dict[str, Any]] = None,
        nodes: Optional[List[Dict[str, Any]]] = None,
        edges: Optional[List[Dict[str, Any]]] = None,
        show_diagram: bool = True,
    ) -> Optional[str]:
        """
        워크플로우 생성

        Args:
            name: 워크플로우 이름
            strategy: 전략 ("research_write", "parallel", "hierarchical", "debate", "pipeline", "custom")
            config: 전략별 설정
            nodes: 노드 정의 (strategy="custom"일 때)
            edges: 엣지 정의 (strategy="custom"일 때)
            show_diagram: 다이어그램 출력 여부

        Returns:
            str: workflow_id (성공 시)

        Example:
            ```
            # Template 사용
            await cmd_create(
                name="Research Pipeline",
                strategy="research_write",
                config={"researcher_id": "r1", "writer_id": "w1"}
            )

            # Custom workflow
            await cmd_create(
                name="Custom Flow",
                strategy="custom",
                nodes=[{"type": "agent", "name": "agent1"}],
                edges=[{"from": "agent1", "to": "agent2"}]
            )
            ```
        """
        self.console.print(f"\n{StatusIcon.info()} [cyan]Creating workflow: {name}...[/cyan]")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Creating workflow...", total=None)

                workflow = await self._orchestrator.create_workflow(
                    name=name,
                    strategy=strategy,
                    config=config or {},
                    nodes=nodes or [],
                    edges=edges or [],
                )

                progress.update(task, completed=True)

            # Store workflow info
            self._workflows[workflow.workflow_id] = workflow

            # Display result
            panel = Panel(
                format_workflow_info(workflow),
                title="✅ Workflow Created",
                border_style="green",
                box=box.ROUNDED,
            )
            self.console.print(panel)

            # Show diagram
            if show_diagram and workflow.visualization:
                self.console.print("\n[cyan]Workflow Diagram:[/cyan]")
                self.console.print(OutputBlock(workflow.visualization))

            self.console.print(
                f"\n{StatusIcon.success()} [green]Workflow ID: {workflow.workflow_id}[/green]\n"
            )

            return workflow.workflow_id

        except Exception as e:
            self.console.print(f"{StatusIcon.error()} [red]Error: {e}[/red]")
            logger.error(f"Failed to create workflow: {e}")
            return None

    # ========================================
    # Command: Execute Workflow
    # ========================================

    async def cmd_execute(
        self,
        workflow_id: str,
        agents: Dict[str, Any],
        task: str,
        tools: Optional[Dict[str, Any]] = None,
        monitor: bool = True,
    ) -> Optional[str]:
        """
        워크플로우 실행

        Args:
            workflow_id: 워크플로우 ID
            agents: Agent 인스턴스 딕셔너리
            task: 실행할 태스크
            tools: 사용 가능한 도구
            monitor: 실시간 모니터링 여부

        Returns:
            str: execution_id (성공 시)

        Example:
            ```
            await cmd_execute(
                workflow_id="wf-123",
                agents={"researcher": r_agent, "writer": w_agent},
                task="Research quantum computing"
            )
            ```
        """
        self.console.print(f"\n{StatusIcon.info()} [cyan]Executing workflow...[/cyan]")

        try:
            # Execute
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=self.console,
            ) as progress:
                progress_task = progress.add_task("Executing workflow...", total=None)

                result = await self._orchestrator.execute(
                    workflow_id=workflow_id,
                    agents=agents,
                    task=task,
                    tools=tools,
                )

                progress.update(progress_task, completed=True)

            # Store execution info
            self._executions[result.execution_id] = result

            # Display result
            if result.status == "completed":
                panel = Panel(
                    format_execution_result(result),
                    title="✅ Execution Completed",
                    border_style="green",
                    box=box.ROUNDED,
                )
            else:
                panel = Panel(
                    format_execution_result(result),
                    title="❌ Execution Failed",
                    border_style="red",
                    box=box.ROUNDED,
                )

            self.console.print(panel)

            # Show node results
            if result.node_results:
                self.console.print("\n[cyan]Node Results:[/cyan]")
                display_node_results(self.console, result.node_results)

            self.console.print(
                f"\n{StatusIcon.success()} [green]Execution ID: {result.execution_id}[/green]\n"
            )

            return result.execution_id

        except Exception as e:
            self.console.print(f"{StatusIcon.error()} [red]Error: {e}[/red]")
            logger.error(f"Failed to execute workflow: {e}")
            return None

    # ========================================
    # Command: Monitor Workflow
    # ========================================

    async def cmd_monitor(
        self,
        workflow_id: str,
        execution_id: str,
        refresh_interval: float = 1.0,
        duration: Optional[float] = None,
    ) -> None:
        """
        워크플로우 실시간 모니터링

        Args:
            workflow_id: 워크플로우 ID
            execution_id: 실행 ID
            refresh_interval: 갱신 간격 (초)
            duration: 모니터링 지속 시간 (None이면 수동 종료)

        Example:
            ```
            # 5초 동안 모니터링
            await cmd_monitor(
                workflow_id="wf-123",
                execution_id="exec-456",
                duration=5.0
            )
            ```
        """
        self.console.print(f"\n{StatusIcon.info()} [cyan]Monitoring workflow execution...[/cyan]")
        self.console.print("[dim]Press Ctrl+C to stop[/dim]\n")

        start_time = asyncio.get_event_loop().time()

        try:
            with Live(
                create_monitor_display(None),
                console=self.console,
                refresh_per_second=1,
            ) as live:
                while True:
                    # Fetch status
                    status = await self._orchestrator.monitor(
                        workflow_id=workflow_id,
                        execution_id=execution_id,
                    )

                    # Update display
                    live.update(create_monitor_display(status))

                    # Check duration
                    if duration:
                        elapsed = asyncio.get_event_loop().time() - start_time
                        if elapsed >= duration:
                            break

                    # Check if completed
                    if status.progress >= 1.0:
                        break

                    await asyncio.sleep(refresh_interval)

            self.console.print(f"\n{StatusIcon.success()} [green]Monitoring completed[/green]\n")

        except KeyboardInterrupt:
            self.console.print(
                f"\n{StatusIcon.warning()} [yellow]Monitoring stopped by user[/yellow]\n"
            )
        except Exception as e:
            self.console.print(f"{StatusIcon.error()} [red]Error: {e}[/red]")
            logger.error(f"Failed to monitor workflow: {e}")

    # ========================================
    # Command: Analyze Workflow
    # ========================================

    async def cmd_analyze(self, workflow_id: str) -> None:
        """
        워크플로우 성능 분석

        Args:
            workflow_id: 워크플로우 ID

        Example:
            ```
            await cmd_analyze(workflow_id="wf-123")
            ```
        """
        self.console.print(f"\n{StatusIcon.info()} [cyan]Analyzing workflow...[/cyan]")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Analyzing performance...", total=None)

                analytics = await self._orchestrator.analyze(workflow_id)

                progress.update(task, completed=True)

            # Display analytics
            self.console.print(
                Panel(
                    format_analytics(analytics),
                    title="📊 Workflow Analytics",
                    border_style="cyan",
                    box=box.ROUNDED,
                )
            )

            # Show bottlenecks
            if analytics.bottlenecks:
                self.console.print("\n[yellow]⚠️  Bottlenecks Detected:[/yellow]")
                display_bottlenecks(self.console, analytics.bottlenecks)

            # Show recommendations
            if analytics.recommendations:
                self.console.print("\n[green]💡 Optimization Recommendations:[/green]")
                for i, rec in enumerate(analytics.recommendations, 1):
                    self.console.print(f"  {i}. {rec}")

            self.console.print()

        except Exception as e:
            self.console.print(f"{StatusIcon.error()} [red]Error: {e}[/red]")
            logger.error(f"Failed to analyze workflow: {e}")

    # ========================================
    # Command: Visualize Workflow
    # ========================================

    async def cmd_visualize(
        self,
        workflow_id: str,
        style: str = "box",
    ) -> None:
        """
        워크플로우 시각화 (ASCII 다이어그램)

        Args:
            workflow_id: 워크플로우 ID
            style: 다이어그램 스타일 ("box", "simple", "compact")

        Example:
            ```
            await cmd_visualize(workflow_id="wf-123")
            ```
        """
        self.console.print(f"\n{StatusIcon.info()} [cyan]Visualizing workflow...[/cyan]\n")

        try:
            diagram = await self._orchestrator.visualize(workflow_id, style=style)

            self.console.print(
                Panel(
                    diagram,
                    title=f"🎨 Workflow Diagram (style: {style})",
                    border_style="cyan",
                    box=box.ROUNDED,
                )
            )
            self.console.print()

        except Exception as e:
            self.console.print(f"{StatusIcon.error()} [red]Error: {e}[/red]")
            logger.error(f"Failed to visualize workflow: {e}")

