"""
ML Features Router

Audio, OCR, Vision RAG, Evaluation, Fine-tuning, Web Search
"""

import logging
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from common import get_vision_rag, get_audio_rag, get_web_search, get_evaluator, get_finetuning

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ml", tags=["ML"])


# TODO: Add endpoints here
