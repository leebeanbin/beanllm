"""
Chunking Experimenter - Report generation.

Extracted from chunking_experimenter.py for single responsibility.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List


def get_comparison_report(
    results: List[Any],
    find_best_strategy_fn: Callable[[], Any],
    get_feedback_summary_fn: Callable[[], Dict[str, Any]],
) -> str:
    """Build chunking strategy comparison report (markdown)."""
    if not results:
        return "No experiment results."

    lines = ["# Chunking Strategy Comparison Report", ""]
    lines.append("## Results (sorted by score)")
    lines.append("")
    lines.append("| Strategy | Score | Chunks | Avg Size | Latency |")
    lines.append("|----------|-------|--------|----------|---------|")

    for r in sorted(results, key=lambda x: x.avg_retrieval_score, reverse=True):
        lines.append(
            f"| {r.strategy_name} | {r.avg_retrieval_score:.4f} | "
            f"{r.chunk_count} | {r.avg_chunk_size:.0f} | {r.latency_ms:.1f}ms |"
        )

    best = find_best_strategy_fn()
    if best:
        lines.append("")
        lines.append("## Best Strategy")
        lines.append(f"- **Name**: {best['strategy']}")
        lines.append(f"- **Score**: {best['score']:.4f}")
        lines.append(f"- **Config**: {best['config']}")

    fb_summary = get_feedback_summary_fn()
    if fb_summary.get("total", 0) > 0:
        lines.append("")
        lines.append("## Feedback Summary")
        lines.append(f"- Total feedbacks: {fb_summary['total']}")
        lines.append(f"- Average rating: {fb_summary['avg_rating']:.2f}")

    return "\n".join(lines)
