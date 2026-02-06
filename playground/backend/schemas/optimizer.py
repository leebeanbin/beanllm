"""
Optimizer Request Schemas
"""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel


class OptimizeRequest(BaseModel):
    """Request to optimize a pipeline"""

    task_type: str = "rag"  # rag, agent, chain
    config: Optional[Dict[str, Any]] = None
    top_k_range: Optional[Tuple[int, int]] = None
    threshold_range: Optional[Tuple[float, float]] = None
    method: str = "bayesian"  # bayesian, grid, random, genetic
    n_trials: int = 30
    test_queries: Optional[List[str]] = None
    model: Optional[str] = None
