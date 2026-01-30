"""
Fine-tuning Request Schemas
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class FineTuningCreateRequest(BaseModel):
    """Request to create a fine-tuning job"""
    base_model: str
    training_data: List[Dict[str, Any]]
    job_name: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    model: Optional[str] = None


class FineTuningStatusRequest(BaseModel):
    """Request to check fine-tuning status"""
    job_id: str
