"""
Metrics Charts - Chart/plot rendering functions

Chart rendering utilities for metrics visualization:
- Size distribution histograms
- Sparklines
- Progress bars
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from rich.console import Console


class MetricsChartsMixin:
    """Chart rendering mixin for MetricsVisualizer"""

    console: "Console"

    def show_size_distribution(
        self,
        size_distribution: Dict[str, int],
        max_width: int = 50,
    ) -> None:
        """
        청크 크기 분포 히스토그램

        Args:
            size_distribution: {"0-500": 10, "500-1000": 50, ...}
            max_width: 막대 최대 너비
        """
        if not size_distribution:
            self.console.print("[red]No distribution data[/red]")
            return

        max_count = max(size_distribution.values())

        self.console.print()
        self.console.print("[bold]Chunk Size Distribution:[/bold]")
        self.console.print()

        for size_range in sorted(size_distribution.keys()):
            count = size_distribution[size_range]
            bar_length = int((count / max_count) * max_width) if max_count > 0 else 0

            bar = "█" * bar_length

            self.console.print(f"  {size_range:>12} [cyan]{bar}[/cyan] {count:,}")

        self.console.print()

    def _create_bar(
        self,
        value: float,
        max_value: float,
        max_width: int,
        color: str = "green",
    ) -> str:
        """Create a horizontal bar"""
        if max_value == 0:
            return "[dim]░[/dim]" * max_width

        filled = int((value / max_value) * max_width)
        bar = f"[{color}]" + "█" * filled + f"[/{color}]"
        bar += "[dim]░[/dim]" * (max_width - filled)

        return bar

    def _create_percentage_bar(
        self,
        percentage: float,
        max_width: int,
        color: str = "green",
    ) -> str:
        """Create a percentage bar"""
        filled = int((percentage / 100) * max_width)
        bar = f"[{color}]" + "█" * filled + f"[/{color}]"
        bar += "[dim]░[/dim]" * (max_width - filled)

        return bar

    def _create_sparkline(self, values: List[float]) -> str:
        """Create ASCII sparkline"""
        if not values:
            return ""

        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val

        if range_val == 0:
            return "▄" * len(values)

        # Sparkline characters (8 levels)
        chars = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

        sparkline = ""
        for value in values:
            normalized = (value - min_val) / range_val
            index = int(normalized * (len(chars) - 1))
            sparkline += chars[index]

        return f"[cyan]{sparkline}[/cyan]"
