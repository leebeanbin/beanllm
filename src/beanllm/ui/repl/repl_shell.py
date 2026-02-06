"""
REPL Shell - beanllm Interactive CLI

간단하고 빠른 REPL 구현
"""

import asyncio
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from .common_commands import create_common_commands

console = Console()


class REPLShell:
    """
    beanllm REPL Shell

    기존 command 모듈들을 통합하는 간단한 REPL

    Example:
        ```python
        shell = REPLShell()
        shell.run()
        ```
    """

    def __init__(self):
        """초기화"""
        self.running = False
        self.common_commands = create_common_commands()

        # Additional command modules
        self.command_modules: Dict[str, Any] = {}

        # Client for commands that need it
        self.client: Optional[Any] = None

    def register_module(self, name: str, module: Any, category: str):
        """
        명령어 모듈 등록

        Args:
            name: 모듈 이름
            module: 명령어 모듈 인스턴스
            category: 카테고리
        """
        self.command_modules[name] = {
            "instance": module,
            "category": category,
        }

        # Register all cmd_* methods
        for attr_name in dir(module):
            if attr_name.startswith("cmd_"):
                cmd_name = attr_name[4:]  # Remove 'cmd_' prefix
                handler = getattr(module, attr_name)

                # Get description from docstring
                description = handler.__doc__ or f"{cmd_name} command"
                if description:
                    description = description.strip().split("\n")[0]

                self.common_commands.register_command(
                    name=cmd_name,
                    handler=handler,
                    description=description,
                    category=category,
                )

    def parse_input(self, user_input: str) -> tuple:
        """
        사용자 입력 파싱

        Args:
            user_input: 사용자 입력

        Returns:
            (command, args) tuple
        """
        parts = user_input.strip().split()
        if not parts:
            return None, []

        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        return command, args

    async def execute_command(self, command: str, args: List[str]) -> bool:
        """
        명령어 실행

        Args:
            command: 명령어
            args: 인자

        Returns:
            True if should exit, False otherwise
        """
        if command not in self.common_commands.command_registry:
            console.print(f"[red]Unknown command: {command}[/red]")
            console.print("[dim]Type 'help' to see available commands.[/dim]\n")
            return False

        try:
            handler = self.common_commands.command_registry[command]["handler"]

            # Check if handler is async
            if asyncio.iscoroutinefunction(handler):
                result = await handler(args)
            else:
                result = handler(args)

            # If result is True, exit
            if result is True:
                return True

        except KeyboardInterrupt:
            console.print("\n[yellow]Command interrupted[/yellow]\n")
        except Exception as e:
            console.print(f"[red]Error executing command: {e}[/red]\n")
            if "--debug" in args or "-d" in args:
                import traceback

                console.print("[dim]" + traceback.format_exc() + "[/dim]")

        return False

    def show_welcome(self):
        """환영 메시지 표시"""
        welcome_panel = Panel(
            "[bold cyan]beanllm REPL[/bold cyan]\n\n"
            "Unified LLM Framework with Clean Architecture\n\n"
            "📚 Type [bold]help[/bold] to see available commands\n"
            "🚀 Type [bold]status[/bold] to check system status\n"
            "👋 Type [bold]exit[/bold] to quit",
            border_style="cyan",
            title="[bold]Welcome[/bold]",
        )
        console.print("\n", welcome_panel, "\n")

    def setup_default_modules(self):
        """기본 모듈 설정 (optional)"""
        # Try to load command modules
        try:
            from .knowledge_graph_commands import KnowledgeGraphCommands

            kg_commands = KnowledgeGraphCommands(client=self.client)
            self.register_module("kg", kg_commands, "Knowledge Graph")
            console.print("[dim]✓ Knowledge Graph commands loaded[/dim]")
        except Exception as e:
            console.print(f"[dim]⚠ Knowledge Graph commands unavailable: {e}[/dim]")

        try:
            from .rag_commands import RAGDebugCommands

            rag_commands = RAGDebugCommands(client=self.client)
            self.register_module("rag", rag_commands, "RAG Debug")
            console.print("[dim]✓ RAG Debug commands loaded[/dim]")
        except Exception as e:
            console.print(f"[dim]⚠ RAG Debug commands unavailable: {e}[/dim]")

        try:
            from .optimizer_commands import OptimizerCommands

            opt_commands = OptimizerCommands(client=self.client)
            self.register_module("optimizer", opt_commands, "Optimizer")
            console.print("[dim]✓ Optimizer commands loaded[/dim]")
        except Exception as e:
            console.print(f"[dim]⚠ Optimizer commands unavailable: {e}[/dim]")

        try:
            from .orchestrator_commands import OrchestratorCommands

            orch_commands = OrchestratorCommands(client=self.client)
            self.register_module("orchestrator", orch_commands, "Orchestrator")
            console.print("[dim]✓ Orchestrator commands loaded[/dim]")
        except Exception as e:
            console.print(f"[dim]⚠ Orchestrator commands unavailable: {e}[/dim]")

        console.print()

    async def run_async(self):
        """비동기 REPL 루프"""
        self.running = True
        self.show_welcome()

        # Setup modules
        console.print("[cyan]Loading command modules...[/cyan]")
        self.setup_default_modules()

        console.print("[green]Ready![/green]\n")

        while self.running:
            try:
                # Get user input
                user_input = Prompt.ask("[bold cyan]beanllm[/bold cyan]")

                if not user_input.strip():
                    continue

                # Parse and execute
                command, args = self.parse_input(user_input)

                if command:
                    should_exit = await self.execute_command(command, args)
                    if should_exit:
                        self.running = False
                        break

            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' or 'quit' to leave the REPL[/yellow]\n")
            except EOFError:
                console.print("\n[cyan]Goodbye! 👋[/cyan]\n")
                break
            except Exception as e:
                console.print(f"[red]Unexpected error: {e}[/red]\n")

    def run(self):
        """REPL 실행 (동기 래퍼)"""
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            console.print("\n[cyan]Goodbye! 👋[/cyan]\n")


def main():
    """CLI entry point"""
    shell = REPLShell()
    shell.run()


if __name__ == "__main__":
    main()
