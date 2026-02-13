"""
RAG Improvement Loop - Main improvement cycle logic.

Extracted from improvement_loop.py for single responsibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def get_logger(name: str) -> logging.Logger:
    try:
        from beanllm.utils.logging import get_logger as _get_logger

        return _get_logger(name)
    except ImportError:
        return logging.getLogger(name)


logger = get_logger(__name__)


@dataclass
class ImprovementCycle:
    """Single improvement cycle record."""

    cycle_number: int
    timestamp: datetime
    chunking_result: Optional[Any]
    eval_score_before: float
    eval_score_after: float
    improvement: float
    strategy_used: str
    changes_made: List[str]


@dataclass
class ImprovementPlan:
    """Improvement plan item."""

    priority: str
    area: str
    issue: str
    action: str
    expected_improvement: float
    config_changes: Dict[str, Any] = field(default_factory=dict)


def get_current_score(chunking_experimenter: Any, evaluator: Optional[Any]) -> float:
    """Compute current combined score."""
    best = chunking_experimenter.find_best_strategy()
    chunking_score = best["score"] if best else 0.0
    eval_score = 0.0
    if evaluator:
        summary = evaluator.get_evaluation_summary()
        eval_score = summary.get("unified_score", {}).get("avg", 0.0)
    if chunking_score > 0 and eval_score > 0:
        return (chunking_score + eval_score) / 2
    if chunking_score > 0:
        return chunking_score
    if eval_score > 0:
        return eval_score
    return 0.0


def run_improvement_cycle_step(
    loop: Any, improvement_plan: Optional[ImprovementPlan] = None
) -> ImprovementCycle:
    """Execute one improvement cycle and append to loop._cycles."""
    cycle_number = len(loop._cycles) + 1
    score_before = get_current_score(loop.chunking_experimenter, loop.evaluator)
    if improvement_plan is None:
        plans = loop.get_improvement_plan()
        if not plans:
            cycle = ImprovementCycle(
                cycle_number=cycle_number,
                timestamp=datetime.now(timezone.utc),
                chunking_result=None,
                eval_score_before=score_before,
                eval_score_after=score_before,
                improvement=0.0,
                strategy_used="none",
                changes_made=[],
            )
            loop._cycles.append(cycle)
            return cycle
        improvement_plan = plans[0]
    changes_made: List[str] = []
    chunking_result = None
    if improvement_plan.area == "chunking" and improvement_plan.config_changes:
        new_config = improvement_plan.config_changes
        chunking_result = loop.chunking_experimenter.run_experiment(
            config=new_config, strategy_name=f"improved_cycle_{cycle_number}"
        )
        changes_made.append(f"Chunking config: {new_config}")
        loop._current_best_config = new_config
    score_after = get_current_score(loop.chunking_experimenter, loop.evaluator)
    cycle = ImprovementCycle(
        cycle_number=cycle_number,
        timestamp=datetime.now(timezone.utc),
        chunking_result=chunking_result,
        eval_score_before=score_before,
        eval_score_after=score_after,
        improvement=score_after - score_before,
        strategy_used=improvement_plan.action[:50],
        changes_made=changes_made,
    )
    loop._cycles.append(cycle)
    logger.info(
        f"Improvement cycle {cycle_number}: {score_before:.4f} -> {score_after:.4f} (+{cycle.improvement:.4f})"
    )
    return cycle


def run_full_cycle(
    loop: Any, max_iterations: int = 3, target_improvement: float = 0.2
) -> Dict[str, Any]:
    """Run improvement cycles until target or max iterations."""
    initial_score = get_current_score(loop.chunking_experimenter, loop.evaluator)
    total_improvement = 0.0
    for _ in range(max_iterations):
        cycle = run_improvement_cycle_step(loop)
        total_improvement += cycle.improvement
        if total_improvement >= target_improvement or cycle.improvement < 0.01:
            break
    final_score = get_current_score(loop.chunking_experimenter, loop.evaluator)
    return {
        "initial_score": initial_score,
        "final_score": final_score,
        "total_improvement": total_improvement,
        "cycles_run": len(loop._cycles),
        "best_config": loop._current_best_config,
        "cycles": loop._cycles,
    }
