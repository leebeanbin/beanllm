"""
CLI command definitions - list, show, providers, export, summary, help.
"""
from __future__ import annotations

import json
from typing import Any

from beanllm.utils.cli.cli_utils import (
    ErrorPattern,
    Panel,
    RICH_AVAILABLE,
    Syntax,
    Table,
    console,
    get_model_registry,
    print_logo,
)


def print_help() -> None:
    """Help 메시지 (디자인 시스템 적용)."""
    print_logo(style="ascii", color="magenta", show_motto=True, show_commands=True)

    if not RICH_AVAILABLE:
        print("Commands: list, show, providers, export, summary, scan, analyze, admin")
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

[yellow]Admin:[/yellow]
  [green]admin[/green] <command>   Admin commands for Google Workspace monitoring 👑
    [dim]analyze[/dim]         Analyze usage patterns with Gemini
    [dim]stats[/dim]           Show Google service statistics
    [dim]optimize[/dim]        Get cost optimization recommendations
    [dim]security[/dim]        Check security events
    [dim]dashboard[/dim]       Launch Streamlit dashboard

[dim]Examples:[/dim]
  beanllm list
  beanllm show gpt-4o-mini
  beanllm scan
  beanllm analyze gpt-5-nano
  beanllm admin analyze --hours=24
  beanllm admin dashboard
""",
        title="[bold magenta]beanllm[/bold magenta] - Unified LLM Model Manager",
        border_style="cyan",
        expand=False,
    )
    console.print(help_panel)


def list_models(registry: Any) -> None:
    """모델 목록 출력."""
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


def show_model(registry: Any, model_name: str) -> None:
    """모델 상세 정보."""
    model = registry.get_model_info(model_name)
    if not model:
        console.print(f"[red]❌ Model not found:[/red] {model_name}")
        return

    if not RICH_AVAILABLE:
        print(f"Model: {model.model_name}")
        print(f"Provider: {model.provider}")
        print(f"Description: {model.description}")
        return

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
            info_text,
            title=f"[bold magenta]{model.model_name}[/bold magenta]",
            border_style="cyan",
        )
    )

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


def list_providers(registry: Any) -> None:
    """Provider 목록."""
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


def export_models(registry: Any) -> None:
    """JSON export."""
    models = registry.get_available_models()
    data = {"models": [model.to_dict() for model in models], "summary": registry.get_summary()}
    print(json.dumps(data, indent=2, ensure_ascii=False))


def show_summary(registry: Any) -> None:
    """요약 정보."""
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
