"""
RAG Improvement Loop - Individual improvement phases/steps.

Extracted from improvement_loop.py for single responsibility.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

try:
    from beanllm.utils.logging import get_logger
except ImportError:

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger(__name__)


def run_initial_experiments(
    chunking_experimenter: Any,
    configs: Optional[List[Dict[str, Any]]] = None,
    use_grid_search: bool = False,
) -> tuple[List[Any], Optional[Dict[str, Any]], float]:
    """
    Run initial chunking strategy experiments.

    Returns:
        (results, best_config, baseline_score)
    """
    if configs is None:
        configs = [
            {"type": "recursive", "chunk_size": 500, "chunk_overlap": 50, "name": "recursive_500"},
            {"type": "recursive", "chunk_size": 1000, "chunk_overlap": 100, "name": "recursive_1000"},
            {"type": "recursive", "chunk_size": 1500, "chunk_overlap": 150, "name": "recursive_1500"},
        ]
    if use_grid_search:
        results = chunking_experimenter.grid_search(
            splitter_type="recursive",
            chunk_sizes=[256, 512, 1000],
            chunk_overlaps=[0, 50, 100],
        )
    else:
        results = chunking_experimenter.compare_strategies(configs)
    best = chunking_experimenter.find_best_strategy()
    best_config = best["config"] if best else None
    baseline_score = best["score"] if best else 0.0
    logger.info(f"Initial experiments complete: {len(results)} strategies, baseline: {baseline_score:.4f}")
    return results, best_config, baseline_score


def evaluate_pipeline(
    evaluator: Optional[Any],
    query: str,
    response: str,
    contexts: List[str],
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate RAG pipeline for a single query/response."""
    if evaluator is None:
        logger.warning("Evaluator not available")
        return {"auto_scores": {}, "unified_score": 0.0}
    auto_scores = evaluator.evaluate_auto(query=query, response=response, contexts=contexts, metrics=metrics)
    unified_score = evaluator.get_unified_score(query)
    avg = sum(auto_scores.values()) / len(auto_scores) if auto_scores else 0.0
    return {"query": query, "auto_scores": auto_scores, "unified_score": unified_score or avg}


def batch_evaluate(
    evaluate_fn: Callable[..., Dict[str, Any]],
    qa_pairs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Batch evaluate multiple QA pairs."""
    results = []
    for qa in qa_pairs:
        result = evaluate_fn(query=qa["query"], response=qa["response"], contexts=qa.get("contexts", []))
        results.append(result)
    avg_score = sum(r["unified_score"] for r in results) / len(results) if results else 0.0
    return {"total": len(results), "results": results, "avg_unified_score": avg_score}


def add_human_feedback(
    evaluator: Optional[Any],
    chunking_experimenter: Any,
    query: str,
    rating: float,
    feedback_type: str = "overall",
    comment: Optional[str] = None,
    chunk_id: Optional[str] = None,
) -> None:
    """Add human feedback."""
    if evaluator:
        evaluator.collect_human_feedback(query=query, rating=rating, feedback_type=feedback_type, comment=comment)
    if chunk_id:
        chunking_experimenter.add_feedback(query=query, chunk_id=chunk_id, rating=rating, feedback_type=feedback_type, comment=comment)
    logger.info(f"Human feedback added: query='{query[:30]}...', rating={rating:.2f}")


def add_comparison_feedback(
    evaluator: Optional[Any],
    query: str,
    response_a: str,
    response_b: str,
    winner: str,
) -> None:
    """Add A/B comparison feedback."""
    if evaluator:
        evaluator.collect_comparison_feedback(query=query, response_a=response_a, response_b=response_b, winner=winner)


def get_improvement_plan(
    evaluator: Optional[Any],
    chunking_experimenter: Any,
    improvement_plan_class: type,
) -> List[Any]:
    """Build improvement plan list (priority-sorted)."""
    plans: List[Any] = []
    if evaluator:
        for s in evaluator.get_improvement_suggestions():
            plans.append(improvement_plan_class(priority=s.priority, area=s.category, issue=s.issue, action=s.suggestion, expected_improvement=s.expected_improvement))
    chunking_suggestions = chunking_experimenter.improve_from_feedback()
    rec = chunking_suggestions.get("recommended_configs", [{}])
    config_changes = rec[0] if rec else {}
    for suggestion in chunking_suggestions.get("suggestions", []):
        plans.append(improvement_plan_class(priority="medium", area="chunking", issue="청킹 개선 필요", action=suggestion, expected_improvement=0.15, config_changes=config_changes))
    priority_order = {"high": 0, "medium": 1, "low": 2}
    plans.sort(key=lambda p: priority_order.get(p.priority, 2))
    return plans
