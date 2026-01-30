"""
Evaluation Router

LLM evaluation endpoints.
Uses Python best practices: duck typing, comprehensions.
"""

import logging
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/evaluation", tags=["Evaluation"])


# ============================================================================
# Request/Response Models
# ============================================================================

class EvaluationRequest(BaseModel):
    """Request for evaluation"""
    task_type: str = Field(..., description="Evaluation type: rag, agent, chain")
    queries: List[str] = Field(..., description="Queries or predictions to evaluate")
    ground_truth: Optional[List[str]] = Field(None, description="Reference answers")
    config: Optional[Dict[str, Any]] = Field(None, description="Additional config")
    model: Optional[str] = Field(None)


class EvaluationResultItem(BaseModel):
    """Single evaluation result"""
    prediction: str
    reference: str
    metrics: Dict[str, float] = Field(default_factory=dict)


class EvaluationResponse(BaseModel):
    """Response from evaluation"""
    task_type: str
    num_queries: int
    metrics: Dict[str, float] = Field(default_factory=dict)
    results: List[EvaluationResultItem] = Field(default_factory=list)
    summary: Dict[str, float] = Field(default_factory=dict)


# ============================================================================
# Helper Functions
# ============================================================================

def _extract_metrics(result: Any) -> Dict[str, float]:
    """Extract metrics using duck typing"""
    if hasattr(result, "metrics"):
        return dict(result.metrics)
    elif isinstance(result, dict):
        return result.get("metrics", {})
    return {}


def _aggregate_metrics(results: List[Any]) -> Dict[str, List[float]]:
    """Aggregate metrics from multiple results"""
    all_metrics: Dict[str, List[float]] = {}

    for result in results:
        metrics = _extract_metrics(result)
        for key, value in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = []
            all_metrics[key].append(value)

    return all_metrics


def _calculate_summary(all_metrics: Dict[str, List[float]]) -> Dict[str, float]:
    """Calculate summary statistics (averages)"""
    return {
        key: sum(values) / len(values)
        for key, values in all_metrics.items()
        if values
    }


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluation_evaluate(request: EvaluationRequest) -> EvaluationResponse:
    """
    Run evaluation on predictions against ground truth.

    Supports batch evaluation with multiple queries.
    """
    try:
        from beanllm.facade.ml.eval_facade import EvaluatorFacade

        evaluator = EvaluatorFacade()

        # Batch evaluation
        if request.ground_truth and len(request.ground_truth) == len(request.queries):
            results = await evaluator.batch_evaluate_async(
                predictions=request.queries,
                references=request.ground_truth,
            )

            all_metrics = _aggregate_metrics(results)
            summary = _calculate_summary(all_metrics)

            result_items = [
                EvaluationResultItem(
                    prediction=request.queries[i],
                    reference=request.ground_truth[i],
                    metrics=_extract_metrics(result),
                )
                for i, result in enumerate(results)
            ]

            return EvaluationResponse(
                task_type=request.task_type,
                num_queries=len(request.queries),
                metrics=summary,
                results=result_items,
                summary=summary,
            )

        # Single evaluation
        prediction = request.queries[0] if request.queries else ""
        reference = request.ground_truth[0] if request.ground_truth else ""

        result = await evaluator.evaluate_async(
            prediction=prediction,
            reference=reference,
        )

        metrics = _extract_metrics(result)

        return EvaluationResponse(
            task_type=request.task_type,
            num_queries=1,
            metrics=metrics,
            results=[
                EvaluationResultItem(
                    prediction=prediction,
                    reference=reference,
                    metrics=metrics,
                )
            ],
            summary=metrics,
        )

    except Exception as e:
        logger.error(f"Evaluation error: {e}", exc_info=True)
        raise HTTPException(500, f"Evaluation error: {str(e)}")
