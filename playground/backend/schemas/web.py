"""
Web Search Request Schemas
"""

from typing import Optional
from pydantic import BaseModel


class WebSearchRequest(BaseModel):
    """Request for web search"""
    query: str
    num_results: int = 5
    engine: str = "duckduckgo"
    model: Optional[str] = None  # LLM model for summarization
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    summarize: bool = False  # Summarize results with LLM
