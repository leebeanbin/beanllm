"""
RAG Debug export - report export and export results display.
"""

from __future__ import annotations

from typing import Any, Dict

from rich import box
from rich.console import Console
from rich.table import Table

from beanllm.ui.components import StatusIcon


def display_export_results(console: Console, results: Dict[str, str]) -> None:
    """리포트 내보내기 결과 표시."""
    console.print()
    console.print(
        f"{StatusIcon.success()} [green bold]리포트가 성공적으로 내보내졌습니다![/green bold]"
    )
    console.print()
    table = Table(
        title="📁 Exported Files",
        title_style="bold green",
        box=box.ROUNDED,
    )
    table.add_column("Format", style="bold cyan")
    table.add_column("File Path", style="white")
    for fmt, path in results.items():
        table.add_row(fmt.upper(), path)
    console.print(table)
    console.print()
