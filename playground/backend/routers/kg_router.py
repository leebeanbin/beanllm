"""
Knowledge Graph Router

Knowledge Graph endpoints
"""

import logging
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from common import get_kg

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/kg", tags=["Knowledge Graph"])


# TODO: Add endpoints here
