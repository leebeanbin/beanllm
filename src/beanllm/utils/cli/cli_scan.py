"""
CLI scan - API scan and discovery logic.
"""

from __future__ import annotations

import sys

from beanllm.utils.cli.cli_utils import (
    Panel,
    Progress,
    RICH_AVAILABLE,
    SpinnerColumn,
    Table,
    TextColumn,
    console,
    create_hybrid_manager,
)


async def scan_models() -> None:
    """API 스캔 및 신규 모델 감지."""
    if RICH_AVAILABLE:
        console.rule("[bold cyan]🔍 Scanning APIs for Models[/bold cyan]")

    try:
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(), TextColumn("[bold blue]{task.description}"), console=console
            ) as progress:
                task = progress.add_task("Loading models and scanning APIs...", total=None)
                manager = await create_hybrid_manager(scan_api=True)
                progress.update(task, completed=True)
        else:
            print("Loading models and scanning APIs...")
            manager = await create_hybrid_manager(scan_api=True)

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
            console.print("\n[bold]📦 Models by Provider:[/bold]\n")
            provider_table = Table(show_header=True, header_style="bold cyan", border_style="dim")
            provider_table.add_column("Provider", style="blue")
            provider_table.add_column("Count", justify="right", style="green")
            for provider, count in summary["by_provider"].items():
                if count > 0:
                    provider_table.add_row(provider, str(count))
            console.print(provider_table)

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
