"""
RAG Improvement Loop - Report generation.

Extracted from improvement_loop.py for single responsibility.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable, List


def export_full_report(
    format: str,
    cycles: List[Any],
    baseline_score: float,
    get_current_score_fn: Callable[[], float],
    chunking_report_fn: Callable[[], str],
    evaluator_report_fn: Any,
    get_improvement_plan_fn: Callable[[], List[Any]],
    current_best_config: Any,
) -> str:
    """
    Export full improvement report as markdown or JSON.

    Args:
        format: "markdown" or "json"
        cycles: List of ImprovementCycle
        baseline_score: Baseline score
        get_current_score_fn: Callable that returns current score
        chunking_report_fn: Callable that returns chunking comparison report
        evaluator_report_fn: Callable or None that returns evaluator report
        get_improvement_plan_fn: Callable that returns list of ImprovementPlan
        current_best_config: Best config dict or None

    Returns:
        Report string
    """
    if format == "markdown":
        return _export_markdown(
            cycles=cycles,
            baseline_score=baseline_score,
            get_current_score_fn=get_current_score_fn,
            chunking_report_fn=chunking_report_fn,
            evaluator_report_fn=evaluator_report_fn,
            get_improvement_plan_fn=get_improvement_plan_fn,
            current_best_config=current_best_config,
        )
    return _export_json(
        baseline_score=baseline_score,
        get_current_score_fn=get_current_score_fn,
        cycles_count=len(cycles),
        current_best_config=current_best_config,
    )


def _export_markdown(
    cycles: List[Any],
    baseline_score: float,
    get_current_score_fn: Callable[[], float],
    chunking_report_fn: Callable[[], str],
    evaluator_report_fn: Any,
    get_improvement_plan_fn: Callable[[], List[Any]],
    current_best_config: Any,
) -> str:
    """Build markdown report."""
    lines = [
        "# RAG Improvement Report",
        "",
        f"**Generated**: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Summary",
        f"- Total improvement cycles: {len(cycles)}",
        f"- Baseline score: {baseline_score:.4f}",
        f"- Current score: {get_current_score_fn():.4f}",
        "",
    ]

    lines.append("## Chunking Experiments")
    lines.append(chunking_report_fn())
    lines.append("")

    if evaluator_report_fn:
        lines.append("## Evaluation Results")
        lines.append(evaluator_report_fn())
        lines.append("")

    if cycles:
        lines.append("## Improvement History")
        lines.append("")
        lines.append("| Cycle | Before | After | Improvement | Strategy |")
        lines.append("|-------|--------|-------|-------------|----------|")
        for c in cycles:
            lines.append(
                f"| {c.cycle_number} | {c.eval_score_before:.4f} | "
                f"{c.eval_score_after:.4f} | {c.improvement:+.4f} | "
                f"{c.strategy_used[:30]}... |"
            )

    lines.append("")
    lines.append("## Final Recommendations")
    if current_best_config:
        lines.append(f"**Best Chunking Config**: {current_best_config}")

    plans = get_improvement_plan_fn()
    if plans:
        lines.append("")
        lines.append("**Remaining Improvements**:")
        for p in plans[:3]:
            lines.append(f"- [{p.priority}] {p.area}: {p.action[:50]}...")

    return "\n".join(lines)


def _export_json(
    baseline_score: float,
    get_current_score_fn: Callable[[], float],
    cycles_count: int,
    current_best_config: Any,
) -> str:
    """Build JSON report."""
    return json.dumps(
        {
            "baseline_score": baseline_score,
            "current_score": get_current_score_fn(),
            "cycles": cycles_count,
            "best_config": current_best_config,
        },
        indent=2,
    )
