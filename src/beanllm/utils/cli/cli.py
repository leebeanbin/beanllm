"""
CLI Tool - Beautiful Terminal UI
터미널 디자인 시스템 적용
"""

import asyncio
import json
import sys

try:
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.tree import Tree

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Panel = None
    Progress = None
    SpinnerColumn = None
    TextColumn = None
    Syntax = None
    Table = None
    Tree = None

try:
    from ...infrastructure.hybrid import create_hybrid_manager
    from ...infrastructure.registry import get_model_registry
    from ...ui import ErrorPattern, get_console, print_logo
except ImportError:
    # Fallback
    def get_console():
        class Console:
            def print(self, *args, **kwargs):
                print(*args, **kwargs)

            def rule(self, *args, **kwargs):
                pass

        return Console()

    def print_logo(*args, **kwargs):
        pass

    class ErrorPattern:
        @staticmethod
        def render(*args, **kwargs):
            print(*args, **kwargs)

    def create_hybrid_manager(*args, **kwargs):
        raise ImportError("hybrid_manager not available")

    def get_model_registry():
        raise ImportError("model_registry not available")


console = get_console()


def main():
    if len(sys.argv) < 2:
        print_help()
        return

    command = sys.argv[1]

    # Async 명령어
    if command in ["scan", "analyze"]:
        asyncio.run(async_main(command))
        return

    # Sync 명령어
    registry = get_model_registry()
    if command == "list":
        list_models(registry)
    elif command == "show":
        if len(sys.argv) < 3:
            ErrorPattern.render(
                "Usage: beanllm show <model_name>",
                error_type="MissingArgument",
                suggestion="Provide a model name to show details",
            )
            return
        show_model(registry, sys.argv[2])
    elif command == "providers":
        list_providers(registry)
    elif command == "export":
        export_models(registry)
    elif command == "summary":
        show_summary(registry)
    else:
        print_help()


async def async_main(command: str):
    """Async 명령어 처리"""
    if command == "scan":
        await scan_models()
    elif command == "analyze":
        if len(sys.argv) < 3:
            ErrorPattern.render(
                "Usage: beanllm analyze <model_name>",
                error_type="MissingArgument",
                suggestion="Provide a model name to analyze",
            )
            return
        await analyze_model(sys.argv[2])


def print_help():
    """Help 메시지 (디자인 시스템 적용)"""
    # 로고 출력 (도움 패키지로서 커맨드 표시)
    print_logo(style="ascii", color="magenta", show_motto=True, show_commands=True)

    if not RICH_AVAILABLE:
        print("Commands: list, show, providers, export, summary, scan, analyze")
        return

    help_panel = Panel(
        """[bold cyan]Commands:[/bold cyan]

[yellow]Basic:[/yellow]
  [green]list[/green]              List all available models
  [green]show[/green] <model>      Show detailed model information
  [green]providers[/green]         List all LLM providers
  [green]summary[/green]           Show summary statistics
  [green]export[/green]            Export all models as JSON

[yellow]Advanced:[/yellow]
  [green]scan[/green]              Scan APIs for new models 🔍
  [green]analyze[/green] <model>   Analyze model with pattern inference 🧠

[dim]Examples:[/dim]
  beanllm list
  beanllm show gpt-4o-mini
  beanllm scan
  beanllm analyze gpt-5-nano
""",
        title="[bold magenta]beanllm[/bold magenta] - Unified LLM Model Manager",
        border_style="cyan",
        expand=False,
    )
    console.print(help_panel)


def list_models(registry):
    """모델 목록 출력"""
    models = registry.get_available_models()
    active_providers = registry.get_active_providers()
    active_names = [p.name for p in active_providers]

    console.print(f"\n[bold]Active Providers:[/bold] {', '.join(active_names)}")
    console.print(f"[bold]Total Models:[/bold] {len(models)}\n")

    if not RICH_AVAILABLE:
        for model in models:
            print(f"{model.model_name} ({model.provider})")
        return

    table = Table(show_header=True, header_style="bold cyan", border_style="dim")
    table.add_column("Status", justify="center", width=6)
    table.add_column("Model", style="green")
    table.add_column("Provider", style="blue")
    table.add_column("Stream", justify="center")
    table.add_column("Temp", justify="center")
    table.add_column("Max Tokens", justify="right")

    for model in models:
        status = "✅" if model.provider in active_names else "❌"
        stream = "✅" if model.supports_streaming else "❌"
        temp = "✅" if model.supports_temperature else "❌"
        max_tokens = str(model.max_tokens) if model.max_tokens else "N/A"

        table.add_row(status, model.model_name, model.provider, stream, temp, max_tokens)

    console.print(table)


def show_model(registry, model_name: str):
    """모델 상세 정보"""
    model = registry.get_model_info(model_name)
    if not model:
        console.print(f"[red]❌ Model not found:[/red] {model_name}")
        return

    if not RICH_AVAILABLE:
        print(f"Model: {model.model_name}")
        print(f"Provider: {model.provider}")
        print(f"Description: {model.description}")
        return

    # 메인 패널
    info_text = f"""[bold cyan]Provider:[/bold cyan] {model.provider}
[bold cyan]Description:[/bold cyan] {model.description or "N/A"}

[bold yellow]Capabilities:[/bold yellow]
  • Streaming: {"✅ Yes" if model.supports_streaming else "❌ No"}
  • Temperature: {"✅ Yes" if model.supports_temperature else "❌ No"}
  • Max Tokens: {"✅ Yes" if model.supports_max_tokens else "❌ No"}"""

    if model.uses_max_completion_tokens:
        info_text += "\n  • Uses max_completion_tokens: ✅ Yes"

    console.print(
        Panel(
            info_text, title=f"[bold magenta]{model.model_name}[/bold magenta]", border_style="cyan"
        )
    )

    # 파라미터 테이블
    if model.parameters:
        console.print("\n[bold]Parameters:[/bold]\n")
        param_table = Table(show_header=True, header_style="bold cyan", border_style="dim")
        param_table.add_column("Status", justify="center", width=6)
        param_table.add_column("Parameter")
        param_table.add_column("Type")
        param_table.add_column("Default")
        param_table.add_column("Required", justify="center")

        for param in model.parameters:
            status = "✅" if param.supported else "❌"
            required = "Yes" if param.required else "No"
            param_table.add_row(status, param.name, param.type, str(param.default), required)

        console.print(param_table)

    if model.example_usage:
        console.print("\n[bold]Example Usage:[/bold]\n")
        syntax = Syntax(model.example_usage, "python", theme="monokai", line_numbers=True)
        console.print(syntax)


def list_providers(registry):
    """Provider 목록"""
    providers = registry.get_all_providers()

    console.print("\n[bold]LLM Providers:[/bold]\n")

    for name, provider in providers.items():
        status_icon = "✅" if provider.status.value == "active" else "❌"
        env_status = "✅ Set" if provider.env_value_set else "❌ Not set"

        if not RICH_AVAILABLE:
            print(f"{status_icon} {name}: {provider.status.value}")
            continue

        info = f"""[bold cyan]Status:[/bold cyan] {provider.status.value}
[bold cyan]Env Key:[/bold cyan] {provider.env_key} [{env_status}]
[bold cyan]Available Models:[/bold cyan] {len(provider.available_models)}"""

        if provider.default_model:
            info += f"\n[bold cyan]Default Model:[/bold cyan] {provider.default_model}"

        console.print(
            Panel(
                info,
                title=f"{status_icon} [bold]{name}[/bold]",
                border_style="green" if provider.status.value == "active" else "red",
                expand=False,
            )
        )


def export_models(registry):
    """JSON export"""
    models = registry.get_available_models()
    data = {"models": [model.to_dict() for model in models], "summary": registry.get_summary()}
    print(json.dumps(data, indent=2, ensure_ascii=False))


def show_summary(registry):
    """요약 정보"""
    summary = registry.get_summary()

    if not RICH_AVAILABLE:
        print(f"Total Providers: {summary['total_providers']}")
        print(f"Total Models: {summary['total_models']}")
        return

    summary_text = f"""[bold cyan]Total Providers:[/bold cyan] {summary["total_providers"]}
[bold cyan]Active Providers:[/bold cyan] {summary["active_providers"]}
[bold cyan]Total Models:[/bold cyan] {summary["total_models"]}

[bold yellow]Active Providers:[/bold yellow] {", ".join(summary["active_provider_names"])}"""

    console.print(
        Panel(summary_text, title="[bold magenta]Summary[/bold magenta]", border_style="cyan")
    )

    # Provider별 상세
    console.print("\n[bold]Provider Details:[/bold]\n")
    detail_table = Table(show_header=True, header_style="bold cyan", border_style="dim")
    detail_table.add_column("Provider")
    detail_table.add_column("Status")
    detail_table.add_column("Models", justify="right")
    detail_table.add_column("Default Model")

    for name, info in summary["providers"].items():
        detail_table.add_row(
            name,
            info["status"],
            str(info["available_models_count"]),
            info["default_model"] or "N/A",
        )

    console.print(detail_table)


async def scan_models():
    """API 스캔 및 신규 모델 감지"""
    if RICH_AVAILABLE:
        console.rule("[bold cyan]🔍 Scanning APIs for Models[/bold cyan]")

    try:
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(), TextColumn("[bold blue]{task.description}"), console=console
            ) as progress:
                task = progress.add_task("Loading models and scanning APIs...", total=None)

                # HybridModelManager 생성 (API 스캔 포함)
                manager = await create_hybrid_manager(scan_api=True)

                progress.update(task, completed=True)
        else:
            print("Loading models and scanning APIs...")
            manager = await create_hybrid_manager(scan_api=True)

        # 요약
        summary = manager.get_summary()

        if RICH_AVAILABLE:
            console.print()
            summary_panel = Panel(
                f"""[bold cyan]Total Models:[/bold cyan] {summary["total"]}
[bold cyan]Local Models:[/bold cyan] {summary["by_source"]["local"]}
[bold cyan]New Models:[/bold cyan] {summary["by_source"]["inferred"]}
[bold cyan]Average Confidence:[/bold cyan] {summary["avg_confidence"]:.2%}""",
                title="[bold magenta]📊 Scan Results[/bold magenta]",
                border_style="cyan",
            )
            console.print(summary_panel)

            # Provider별
            console.print("\n[bold]📦 Models by Provider:[/bold]\n")
            provider_table = Table(show_header=True, header_style="bold cyan", border_style="dim")
            provider_table.add_column("Provider", style="blue")
            provider_table.add_column("Count", justify="right", style="green")

            for provider, count in summary["by_provider"].items():
                if count > 0:
                    provider_table.add_row(provider, str(count))

            console.print(provider_table)

            # 신규 모델
            new_models = manager.get_new_models()
            if new_models:
                console.print()
                console.rule(
                    f"[bold yellow]✨ New Models Discovered: {len(new_models)}[/bold yellow]"
                )
                console.print()

                for model in new_models:
                    confidence_color = (
                        "green"
                        if model.inference_confidence >= 0.8
                        else "yellow"
                        if model.inference_confidence >= 0.6
                        else "red"
                    )

                    model_info = f"""[bold cyan]Provider:[/bold cyan] {model.provider}
[bold cyan]Display Name:[/bold cyan] {model.display_name}
[bold cyan]Confidence:[/bold cyan] [{confidence_color}]{model.inference_confidence:.2f} ({int(model.inference_confidence * 100)}%)[/{confidence_color}]
[bold cyan]Matched Patterns:[/bold cyan] {", ".join(model.matched_patterns)}

[bold yellow]Parameters:[/bold yellow]
  • Temperature: {"✅ Yes" if model.supports_temperature else "❌ No"}
  • Max Tokens: {model.max_tokens or "N/A"}
  • Max Completion Tokens: {"✅ Yes" if model.uses_max_completion_tokens else "❌ No"}"""

                    console.print(
                        Panel(
                            model_info,
                            title=f"[bold magenta]• {model.model_id}[/bold magenta]",
                            border_style=confidence_color,
                            expand=False,
                        )
                    )
            else:
                console.print()
                console.print(
                    Panel(
                        "[green]✅ No new models discovered. All models are up to date![/green]",
                        border_style="green",
                    )
                )
        else:
            print(f"Total Models: {summary['total']}")
            print(f"New Models: {summary['by_source']['inferred']}")

    except Exception as e:
        console.print(f"\n[red]❌ Error scanning APIs:[/red] {e}")
        sys.exit(1)


async def analyze_model(model_id: str):
    """특정 모델 분석 (패턴 기반 추론)"""
    if RICH_AVAILABLE:
        console.rule(f"[bold cyan]🔍 Analyzing Model: {model_id}[/bold cyan]")

    try:
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(), TextColumn("[bold blue]{task.description}"), console=console
            ) as progress:
                task = progress.add_task("Loading and analyzing model...", total=None)

                # HybridModelManager 생성 (API 스캔 포함)
                manager = await create_hybrid_manager(scan_api=True)

                progress.update(task, completed=True)
        else:
            print("Loading and analyzing model...")
            manager = await create_hybrid_manager(scan_api=True)

        # 모델 검색
        model = manager.get_model_info(model_id)

        if not model:
            console.print(f"\n[red]❌ Model not found:[/red] {model_id}")
            console.print("\n[dim]Try running 'beanllm scan' first to discover new models.[/dim]")
            sys.exit(1)

        if not RICH_AVAILABLE:
            print(f"Model: {model.model_id}")
            print(f"Provider: {model.provider}")
            print(f"Confidence: {model.inference_confidence:.2f}")
            return

        # 소스 색상
        source_color = "green" if model.source == "local" else "yellow"
        confidence_color = (
            "green"
            if model.inference_confidence >= 0.8
            else "yellow"
            if model.inference_confidence >= 0.6
            else "red"
        )

        # 모델 정보
        console.print()
        basic_info = f"""[bold cyan]Provider:[/bold cyan] {model.provider}
[bold cyan]Display Name:[/bold cyan] {model.display_name}
[bold cyan]Source:[/bold cyan] [{source_color}]{model.source}[/{source_color}]"""

        console.print(
            Panel(
                basic_info,
                title=f"[bold magenta]📋 {model.model_id}[/bold magenta]",
                border_style="cyan",
            )
        )

        # 파라미터
        console.print()
        param_tree = Tree("[bold yellow]🔧 Parameters[/bold yellow]")
        param_tree.add(f"Streaming: {'✅ Yes' if model.supports_streaming else '❌ No'}")
        param_tree.add(f"Temperature: {'✅ Yes' if model.supports_temperature else '❌ No'}")
        param_tree.add(f"Max Tokens: {'✅ Yes' if model.supports_max_tokens else '❌ No'}")
        param_tree.add(
            f"Max Completion Tokens: {'✅ Yes' if model.uses_max_completion_tokens else '❌ No'}"
        )

        if model.max_tokens:
            param_tree.add(f"Max Tokens Value: {model.max_tokens}")
        if model.tier:
            param_tree.add(f"Tier: {model.tier}")
        if model.speed:
            param_tree.add(f"Speed: {model.speed}")

        console.print(param_tree)

        # 추론 정보
        console.print()
        inference_info = f"""[bold cyan]Confidence:[/bold cyan] [{confidence_color}]{model.inference_confidence:.2f} ({int(model.inference_confidence * 100)}%)[/{confidence_color}]"""

        if model.matched_patterns:
            inference_info += (
                f"\n[bold cyan]Matched Patterns:[/bold cyan] {', '.join(model.matched_patterns)}"
            )
        if model.discovered_at:
            inference_info += f"\n[bold cyan]Discovered At:[/bold cyan] {model.discovered_at}"
        if model.last_seen:
            inference_info += f"\n[bold cyan]Last Seen:[/bold cyan] {model.last_seen}"

        console.print(
            Panel(
                inference_info,
                title="[bold yellow]📊 Inference Information[/bold yellow]",
                border_style=confidence_color,
            )
        )

    except Exception as e:
        console.print(f"\n[red]❌ Error analyzing model:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
