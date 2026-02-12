"""
Fine-tuning Router

LLM fine-tuning endpoints.
Uses Python best practices: context managers, duck typing.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from utils.file_upload import temp_directory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/finetuning", tags=["Fine-tuning"])


# ============================================================================
# Request/Response Models
# ============================================================================


class FineTuningCreateRequest(BaseModel):
    """Request to create fine-tuning job"""

    base_model: str = Field(..., description="Base model to fine-tune")
    training_data: List[Dict[str, Any]] = Field(
        ..., description="Training examples in JSONL format"
    )
    validation_data: Optional[List[Dict[str, Any]]] = Field(None, description="Validation examples")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Training hyperparameters")
    suffix: Optional[str] = Field(None, description="Model suffix for naming")


class FineTuningJobResponse(BaseModel):
    """Response from fine-tuning job creation"""

    job_id: str
    status: str
    base_model: str
    created_at: Optional[str] = None


class FineTuningStatusResponse(BaseModel):
    """Response from fine-tuning status check"""

    job_id: str
    status: str
    progress: float = 0.0
    fine_tuned_model: Optional[str] = None
    training_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    error: Optional[str] = None


# ============================================================================
# Helper Functions
# ============================================================================


def _get_finetuning_manager():
    """Get or create fine-tuning manager"""
    from beanllm.facade.ml.finetuning_facade import (
        FineTuningManagerFacade,
        create_finetuning_provider,
    )

    provider = create_finetuning_provider(provider="openai")
    return FineTuningManagerFacade(provider=provider)


def _safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely get attribute using duck typing"""
    if hasattr(obj, attr):
        return getattr(obj, attr)
    elif isinstance(obj, dict):
        return obj.get(attr, default)
    return default


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/create", response_model=FineTuningJobResponse)
async def finetuning_create(request: FineTuningCreateRequest) -> FineTuningJobResponse:
    """
    Create a new fine-tuning job.

    Training data should be in OpenAI JSONL format:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    try:
        finetuning = _get_finetuning_manager()

        # Create temp file for training data inside a managed temp directory
        async with temp_directory() as temp_dir:
            temp_path = Path(temp_dir) / "training_data.jsonl"
            with open(temp_path, "w") as temp_file:
                for example in request.training_data:
                    json.dump(example, temp_file)
                    temp_file.write("\n")

            job = finetuning.start_training(
                model=request.base_model,
                training_file=str(temp_path),
                **(request.hyperparameters or {}),
            )

            return FineTuningJobResponse(
                job_id=_safe_get(job, "job_id", "job_001"),
                status=_safe_get(job, "status", "created"),
                base_model=request.base_model,
                created_at=_safe_get(job, "created_at"),
            )

    except Exception as e:
        logger.error(f"Fine-tuning create error: {e}", exc_info=True)
        raise HTTPException(500, f"Fine-tuning create error: {str(e)}")


@router.get("/status/{job_id}", response_model=FineTuningStatusResponse)
async def finetuning_status(job_id: str) -> FineTuningStatusResponse:
    """
    Get fine-tuning job status.

    Returns progress, losses, and fine-tuned model name when complete.
    """
    try:
        finetuning = _get_finetuning_manager()
        progress = finetuning.get_training_progress(job_id)

        return FineTuningStatusResponse(
            job_id=job_id,
            status=_safe_get(progress, "status", "unknown"),
            progress=_safe_get(progress, "progress", 0.0),
            fine_tuned_model=_safe_get(progress, "fine_tuned_model"),
            training_loss=_safe_get(progress, "training_loss"),
            validation_loss=_safe_get(progress, "validation_loss"),
            error=_safe_get(progress, "error"),
        )

    except Exception as e:
        logger.error(f"Fine-tuning status error: {e}", exc_info=True)
        raise HTTPException(500, f"Fine-tuning status error: {str(e)}")


@router.post("/cancel/{job_id}")
async def finetuning_cancel(job_id: str) -> Dict[str, str]:
    """Cancel a fine-tuning job"""
    try:
        finetuning = _get_finetuning_manager()
        finetuning.cancel_training(job_id)

        return {"job_id": job_id, "status": "cancelled"}

    except Exception as e:
        logger.error(f"Fine-tuning cancel error: {e}", exc_info=True)
        raise HTTPException(500, f"Fine-tuning cancel error: {str(e)}")


@router.get("/list")
async def finetuning_list() -> Dict[str, Any]:
    """List all fine-tuning jobs"""
    try:
        finetuning = _get_finetuning_manager()
        jobs = finetuning.list_jobs() if hasattr(finetuning, "list_jobs") else []

        return {
            "jobs": [
                {
                    "job_id": _safe_get(job, "job_id", ""),
                    "status": _safe_get(job, "status", ""),
                    "base_model": _safe_get(job, "model", ""),
                    "created_at": _safe_get(job, "created_at"),
                }
                for job in jobs
            ],
            "total": len(jobs),
        }

    except Exception as e:
        logger.error(f"Fine-tuning list error: {e}", exc_info=True)
        raise HTTPException(500, f"Fine-tuning list error: {str(e)}")
