"""
Evaluation Request Schemas
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class EvaluationRequest(BaseModel):
    """Request to evaluate a pipeline"""

    task_type: str  # rag, agent, chain
    queries: List[str]
    ground_truth: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
