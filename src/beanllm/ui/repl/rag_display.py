"""
RAG Debug display helpers - session, embedding, validation, tuning, summary.
"""

from __future__ import annotations

from typing import Any, Dict

from rich import box
from rich.console import Console
from rich.table import Table

from beanllm.ui.components import Badge, Divider, StatusIcon


def display_session_info(console: Console, response: Any) -> None:
    """세션 정보 표시."""
    table = Table(
        title=f"🔍 RAG Debug Session: {response.session_name or 'Unnamed'}",
        title_style="bold cyan",
        box=box.ROUNDED,
        show_header=False,
    )
    table.add_column("Key", style="bold cyan", width=20)
    table.add_column("Value", style="white")
    table.add_row("Session ID", response.session_id[:12] + "...")
    table.add_row("Status", f"{Badge.success('ACTIVE')}")
    table.add_row("Documents", f"{response.num_documents:,}")
    table.add_row("Embeddings", f"{response.num_embeddings:,}")
    table.add_row("Embedding Dim", str(response.embedding_dim))
    table.add_row("Created At", response.created_at)
    console.print()
    console.print(table)
    console.print()


def display_embedding_analysis(console: Console, response: Any) -> None:
    """Embedding 분석 결과 표시."""
    table = Table(
        title=f"📊 Embedding Analysis ({response.method.upper()})",
        title_style="bold green",
        box=box.ROUNDED,
        show_header=False,
    )
    table.add_column("Metric", style="bold cyan", width=25)
    table.add_column("Value", style="white")
    table.add_row("Clusters Found", str(response.num_clusters))
    table.add_row("Outliers Detected", str(len(response.outliers)))
    table.add_row(
        "Silhouette Score",
        f"{response.silhouette_score:.4f}" if response.silhouette_score else "N/A",
    )
    cluster_sizes_str = ", ".join(f"C{k}: {v}" for k, v in sorted(response.cluster_sizes.items()))
    table.add_row("Cluster Sizes", cluster_sizes_str)
    console.print()
    console.print(table)
    if response.silhouette_score:
        display_quality_assessment(console, response.silhouette_score)
    console.print()


def display_quality_assessment(console: Console, silhouette_score: float) -> None:
    """클러스터링 품질 평가 표시."""
    console.print()
    console.print("[bold]Clustering Quality:[/bold]")
    if silhouette_score > 0.7:
        assessment = f"{StatusIcon.success()} Excellent (강력한 클러스터 구조)"
        color = "green"
    elif silhouette_score > 0.5:
        assessment = f"{StatusIcon.success()} Good (명확한 클러스터)"
        color = "cyan"
    elif silhouette_score > 0.25:
        assessment = f"{StatusIcon.warning()} Fair (약한 클러스터 구조)"
        color = "yellow"
    else:
        assessment = f"{StatusIcon.error()} Poor (클러스터가 불명확)"
        color = "red"
    console.print(f"  [{color}]{assessment}[/{color}]")


def display_chunk_validation(console: Console, response: Any) -> None:
    """청크 검증 결과 표시."""
    table = Table(
        title="📝 Chunk Validation Results",
        title_style="bold blue",
        box=box.ROUNDED,
        show_header=False,
    )
    table.add_column("Metric", style="bold cyan", width=25)
    table.add_column("Value", style="white")
    table.add_row("Total Chunks", f"{response.total_chunks:,}")
    table.add_row("Valid Chunks", f"{response.valid_chunks:,}")
    table.add_row("Issues Found", str(len(response.issues)))
    table.add_row("Duplicate Chunks", str(len(response.duplicate_chunks)))
    console.print()
    console.print(table)
    if response.issues:
        console.print()
        console.print(f"{StatusIcon.warning()} [yellow bold]Issues Found:[/yellow bold]")
        for issue in response.issues[:10]:
            console.print(f"  • [yellow]{issue}[/yellow]")
        if len(response.issues) > 10:
            console.print(f"  [dim]... and {len(response.issues) - 10} more[/dim]")
    if response.recommendations:
        console.print()
        console.print(f"{StatusIcon.info()} [cyan bold]Recommendations:[/cyan bold]")
        for rec in response.recommendations:
            console.print(f"  💡 [cyan]{rec}[/cyan]")
    console.print()


def display_tuning_results(console: Console, response: Any) -> None:
    """파라미터 튜닝 결과 표시."""
    table = Table(
        title="⚙️  Parameter Tuning Results",
        title_style="bold magenta",
        box=box.ROUNDED,
        show_header=False,
    )
    table.add_column("Metric", style="bold cyan", width=25)
    table.add_column("Value", style="white")
    table.add_row("New Parameters", str(response.parameters))
    table.add_row("Average Score", f"{response.avg_score:.4f}")
    if response.comparison_with_baseline:
        comparison = response.comparison_with_baseline
        improvement = comparison.get("improvement_pct", 0.0)
        improvement_str = f"{improvement:+.2f}%"
        if improvement > 5:
            improvement_str = f"[green]{improvement_str} {StatusIcon.SUCCESS}[/green]"
        elif improvement < -5:
            improvement_str = f"[red]{improvement_str} {StatusIcon.ERROR}[/red]"
        else:
            improvement_str = f"[yellow]{improvement_str}[/yellow]"
        table.add_row("vs Baseline", improvement_str)
    console.print()
    console.print(table)
    if response.recommendations:
        console.print()
        console.print(f"{StatusIcon.info()} [cyan bold]Recommendations:[/cyan bold]")
        for rec in response.recommendations:
            console.print(f"  {rec}")
    console.print()


def display_full_analysis_summary(console: Console, results: Dict[str, Any]) -> None:
    """전체 분석 요약 표시."""
    console.print()
    console.print(Divider.thick())
    console.print("[bold green]✅ 전체 분석 완료![/bold green]")
    console.print(Divider.thick())
    console.print()
    completed = []
    if "embedding_analysis" in results:
        completed.append("📊 Embedding Analysis")
    if "chunk_validation" in results:
        completed.append("📝 Chunk Validation")
    if "parameter_tuning" in results:
        completed.append("⚙️  Parameter Tuning")
    for item in completed:
        console.print(f"{StatusIcon.success()} {item}")
    console.print()
