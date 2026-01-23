"""
RAG Router

RAG (Retrieval-Augmented Generation) endpoints
"""

import logging
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from common import get_rag_chain, set_rag_chain

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag", tags=["RAG"])


# TODO: Add endpoints here
