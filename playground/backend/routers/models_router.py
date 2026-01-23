"""
Models Router

Model management endpoints (list, pull, analyze)
"""

import logging
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from common import get_ollama_model_name_for_chat, track_downloaded_model, get_downloaded_models

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["Models"])


# TODO: Add endpoints here
