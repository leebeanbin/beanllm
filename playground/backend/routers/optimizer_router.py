"""
Optimizer Router

Hyperparameter optimization endpoints.
Uses Python best practices: duck typing, tuple unpacking.
"""

import logging
from typing import Dict, Tuple, Any, Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from common import get_optimizer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/optimizer", tags=["Optimizer"])


# ============================================================================
# Request/Response Models
# ============================================================================

class OptimizeRequest(BaseModel):
    """Request for hyperparameter optimization"""
    task_type: str = Field(default="rag", description="Task type: rag, agent, chain")
    config: Optional[Dict[str, Any]] = Field(None, description="Base configuration")
    top_k_range: Optional[Tuple[int, int]] = Field(None, description="Range for top_k (min, max)")
    threshold_range: Optional[Tuple[float, float]] = Field(None, description="Range for threshold")
    method: str = Field(default="bayesian", description="Optimization method")
    n_trials: int = Field(default=30, ge=1, le=200, description="Number of trials")
    test_queries: Optional[List[str]] = Field(None, description="Test queries for evaluation")
    model: Optional[str] = Field(None)


class OptimizeResponse(BaseModel):
    """Response from optimization"""
    task_type: str
    optimized_config: Dict[str, Any] = Field(default_factory=dict)
    improvements: Dict[str, str] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    best_params: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Helper Functions
# ============================================================================

def _safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely get attribute using duck typing"""
    if hasattr(obj, attr):
        return getattr(obj, attr)
    elif isinstance(obj, dict):
        return obj.get(attr, default)
    return default


def _parse_range(
    value: Optional[Tuple[Any, Any]],
    default: Tuple[Any, Any],
) -> Tuple[Any, Any]:
    """Parse range with default fallback"""
    if value is None:
        return default
    return tuple(value)


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/optimize", response_model=OptimizeResponse)
async def optimize(request: OptimizeRequest) -> OptimizeResponse:
    """
    Run hyperparameter optimization.

    Methods:
    - bayesian: Bayesian optimization (default, efficient)
    - grid: Grid search (exhaustive)
    - random: Random search (fast)
    - genetic: Genetic algorithm (complex landscapes)
    """
    try:
        optimizer = get_optimizer()

        # Parse ranges with defaults using tuple unpacking
        top_k_range = _parse_range(request.top_k_range, (1, 20))
        threshold_range = _parse_range(request.threshold_range, (0.0, 1.0))

        response = await optimizer.quick_optimize(
            top_k_range=top_k_range,
            threshold_range=threshold_range,
            method=request.method,
            n_trials=request.n_trials,
        )

        # Extract results using duck typing
        best_params = _safe_get(response, "best_params", {})
        improvement = _safe_get(response, "improvement_percentage", 0)
        metrics = _safe_get(response, "metrics", {})

        return OptimizeResponse(
            task_type=request.task_type,
            optimized_config=best_params,
            improvements={
                "latency": f"{improvement:.1f}%",
                "quality": "improved" if improvement > 0 else "unchanged",
            },
            metrics=metrics,
            best_params=best_params,
        )

    except Exception as e:
        logger.error(f"Optimizer error: {e}", exc_info=True)
        raise HTTPException(500, f"Optimizer error: {str(e)}")


@router.get("/methods")
async def list_methods() -> Dict[str, Any]:
    """List available optimization methods"""
    return {
        "methods": [
            {
                "id": "bayesian",
                "name": "Bayesian Optimization",
                "description": "Efficient, uses probabilistic model",
                "recommended": True,
            },
            {
                "id": "grid",
                "name": "Grid Search",
                "description": "Exhaustive search, good for small spaces",
                "recommended": False,
            },
            {
                "id": "random",
                "name": "Random Search",
                "description": "Fast, good baseline",
                "recommended": False,
            },
            {
                "id": "genetic",
                "name": "Genetic Algorithm",
                "description": "Good for complex parameter landscapes",
                "recommended": False,
            },
        ]
    }
