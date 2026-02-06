"""
Common REPL Commands

공통 명령어들 (help, exit, clear 등)
"""

from typing import Any, Callable, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class CommonCommands:
    """공통 REPL 명령어"""

    def __init__(self):
        self.command_registry: Dict[str, Dict[str, Any]] = {}

    def register_command(
        self,
        name: str,
        handler: Callable,
        description: str,
        category: str = "General",
        usage: str = "",
    ):
        """
        명령어 등록

        Args:
            name: 명령어 이름
            handler: 핸들러 함수
            description: 설명
            category: 카테고리
            usage: 사용법
        """
        self.command_registry[name] = {
            "handler": handler,
            "description": description,
            "category": category,
            "usage": usage,
        }

    def cmd_help(self, args: Optional[List[str]] = None) -> None:
        """
        도움말 표시

        Usage:
            help [command]
        """
        if args and len(args) > 0:
            # Specific command help
            cmd_name = args[0]
            if cmd_name in self.command_registry:
                cmd_info = self.command_registry[cmd_name]
                console.print(f"\n[bold cyan]{cmd_name}[/bold cyan]")
                console.print(f"Category: {cmd_info['category']}")
                console.print(f"Description: {cmd_info['description']}")
                if cmd_info["usage"]:
                    console.print(f"\nUsage:\n  {cmd_info['usage']}")
            else:
                console.print(f"[red]Unknown command: {cmd_name}[/red]")
            return

        # Show all commands grouped by category
        categories: Dict[str, List[tuple]] = {}
        for cmd_name, cmd_info in self.command_registry.items():
            category = cmd_info["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append((cmd_name, cmd_info["description"]))

        console.print("\n[bold cyan]Available Commands[/bold cyan]\n")

        for category in sorted(categories.keys()):
            table = Table(title=f"[bold]{category}[/bold]", show_header=True)
            table.add_column("Command", style="cyan", no_wrap=True)
            table.add_column("Description", style="white")

            for cmd_name, description in sorted(categories[category]):
                table.add_row(cmd_name, description)

            console.print(table)
            console.print()

        console.print("[dim]Type 'help <command>' for detailed information.[/dim]\n")

    def cmd_exit(self, args: Optional[List[str]] = None) -> bool:
        """
        REPL 종료

        Returns:
            True to exit
        """
        console.print("\n[cyan]Goodbye! 👋[/cyan]\n")
        return True

    def cmd_quit(self, args: Optional[List[str]] = None) -> bool:
        """Alias for exit"""
        return self.cmd_exit(args)

    def cmd_clear(self, args: Optional[List[str]] = None) -> None:
        """화면 지우기"""
        import os

        os.system("cls" if os.name == "nt" else "clear")
        console.print("[bold cyan]beanllm REPL[/bold cyan] - Type 'help' for commands\n")

    def cmd_version(self, args: Optional[List[str]] = None) -> None:
        """beanllm 버전 정보 표시"""
        try:
            import beanllm

            version = getattr(beanllm, "__version__", "unknown")
        except:
            version = "unknown"

        info_panel = Panel(
            f"[bold]beanllm[/bold] v{version}\n"
            f"Unified LLM Framework with Clean Architecture\n\n"
            f"📚 Documentation: https://github.com/leebeanbin/beanllm\n"
            f"🐛 Issues: https://github.com/leebeanbin/beanllm/issues",
            title="Version Info",
            border_style="cyan",
        )
        console.print("\n", info_panel, "\n")

    def cmd_status(self, args: Optional[List[str]] = None) -> None:
        """현재 REPL 상태 표시"""
        table = Table(title="[bold]REPL Status[/bold]")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")

        table.add_row("REPL Shell", "✅ Running")
        table.add_row("Commands Loaded", f"✅ {len(self.command_registry)}")

        # Check distributed features
        try:
            from beanllm.infrastructure.distributed import check_kafka_health, check_redis_health

            redis_status = "✅ Connected" if check_redis_health() else "❌ Disconnected"
            kafka_status = "✅ Connected" if check_kafka_health() else "❌ Disconnected"
        except:
            redis_status = "⚠️  Not configured"
            kafka_status = "⚠️  Not configured"

        table.add_row("Redis", redis_status)
        table.add_row("Kafka", kafka_status)

        console.print("\n", table, "\n")

    def cmd_config(self, args: Optional[List[str]] = None) -> None:
        """환경 설정 표시"""
        import os

        config_table = Table(title="[bold]Configuration[/bold]")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="yellow")

        # Environment variables
        env_vars = [
            "USE_DISTRIBUTED",
            "REDIS_HOST",
            "REDIS_PORT",
            "KAFKA_BOOTSTRAP_SERVERS",
        ]

        for var in env_vars:
            value = os.getenv(var, "[dim]not set[/dim]")
            config_table.add_row(var, value)

        console.print("\n", config_table, "\n")


def create_common_commands() -> CommonCommands:
    """
    공통 명령어 인스턴스 생성 및 등록

    Returns:
        CommonCommands 인스턴스
    """
    common = CommonCommands()

    # Register common commands
    common.register_command(
        name="help",
        handler=common.cmd_help,
        description="Show help for commands",
        category="General",
        usage="help [command]",
    )

    common.register_command(
        name="exit",
        handler=common.cmd_exit,
        description="Exit the REPL",
        category="General",
        usage="exit",
    )

    common.register_command(
        name="quit",
        handler=common.cmd_quit,
        description="Exit the REPL (alias for exit)",
        category="General",
        usage="quit",
    )

    common.register_command(
        name="clear",
        handler=common.cmd_clear,
        description="Clear the screen",
        category="General",
        usage="clear",
    )

    common.register_command(
        name="version",
        handler=common.cmd_version,
        description="Show beanllm version info",
        category="General",
        usage="version",
    )

    common.register_command(
        name="status",
        handler=common.cmd_status,
        description="Show REPL status",
        category="General",
        usage="status",
    )

    common.register_command(
        name="config",
        handler=common.cmd_config,
        description="Show configuration",
        category="General",
        usage="config",
    )

    return common
