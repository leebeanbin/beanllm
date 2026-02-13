"""
CLI analyze - Model analysis with pattern inference.
"""

from __future__ import annotations

import sys
from typing import Any

from beanllm.utils.cli.cli_utils import (
    RICH_AVAILABLE,
    Panel,
    Progress,
    SpinnerColumn,
    TextColumn,
    Tree,
    console,
    create_hybrid_manager,
)


async def analyze_model(model_id: str) -> None:
    """특정 모델 분석 (패턴 기반 추론)."""
    if RICH_AVAILABLE:
        console.rule(f"[bold cyan]🔍 Analyzing Model: {model_id}[/bold cyan]")

    try:
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(), TextColumn("[bold blue]{task.description}"), console=console
            ) as progress:
                task = progress.add_task("Loading and analyzing model...", total=None)
                manager = await create_hybrid_manager(scan_api=True)
                progress.update(task, completed=True)
        else:
            print("Loading and analyzing model...")
            manager = await create_hybrid_manager(scan_api=True)

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

        source_color = "green" if model.source == "local" else "yellow"
        confidence_color = (
            "green"
            if model.inference_confidence >= 0.8
            else "yellow"
            if model.inference_confidence >= 0.6
            else "red"
        )
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
