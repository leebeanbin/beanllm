"""
Web Search Router

Web search endpoints with optional LLM summarization.
Uses Python best practices: dict comprehensions, optional chaining.
"""

import logging
from typing import Any, Dict, List, Optional

from common import get_web_search
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/web", tags=["Web Search"])


# ============================================================================
# Request/Response Models
# ============================================================================


class WebSearchRequest(BaseModel):
    """Request for web search"""

    query: str = Field(..., description="Search query")
    num_results: int = Field(default=5, ge=1, le=20, description="Number of results")
    engine: str = Field(default="duckduckgo", description="Search engine")
    model: Optional[str] = Field(None, description="LLM model for summarization")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=32000)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    summarize: bool = Field(default=False, description="Summarize results with LLM")


class SearchResult(BaseModel):
    """Single search result"""

    title: str
    url: str
    snippet: str


class WebSearchResponse(BaseModel):
    """Response from web search"""

    query: str
    results: List[SearchResult]
    num_results: int
    summary: Optional[str] = None


# ============================================================================
# Helper Functions
# ============================================================================


def _build_chat_kwargs(request: WebSearchRequest) -> Dict[str, Any]:
    """Build chat kwargs from request, filtering None values"""
    param_mapping = {
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "top_p": request.top_p,
        "frequency_penalty": request.frequency_penalty,
        "presence_penalty": request.presence_penalty,
    }
    # Filter out None values using dict comprehension
    return {k: v for k, v in param_mapping.items() if v is not None}


def _format_results_for_summary(results: List[Dict[str, str]], query: str) -> str:
    """Format search results into a prompt for LLM summarization"""
    context = "\n\n".join(
        f"{i+1}. {r['title']}\n   {r['snippet']}\n   URL: {r['url']}"
        for i, r in enumerate(results[:5])
    )

    return f"""다음은 '{query}'에 대한 웹 검색 결과입니다.

검색 결과:
{context}

위 검색 결과를 바탕으로 다음을 수행해주세요:
1. 핵심 내용을 3-5개의 주요 포인트로 요약
2. 각 포인트에 대한 간단한 설명 추가
3. 가장 중요한 정보를 강조

요약:"""


async def _generate_summary(
    results: List[Dict[str, str]],
    request: WebSearchRequest,
) -> Optional[str]:
    """Generate summary using LLM"""
    if not request.summarize or not request.model:
        return None

    try:
        from beanllm.facade.core.client_facade import Client

        client = Client(model=request.model)
        prompt = _format_results_for_summary(results, request.query)
        chat_kwargs = _build_chat_kwargs(request)

        response = await client.chat(
            messages=[{"role": "user", "content": prompt}],
            **chat_kwargs,
        )
        return response.content

    except Exception as e:
        logger.warning(f"Failed to generate summary: {e}")
        return None


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/search", response_model=WebSearchResponse)
async def web_search(request: WebSearchRequest) -> WebSearchResponse:
    """
    Perform web search with optional LLM summarization.

    Supported engines: duckduckgo, google, bing
    """
    try:
        from beanllm.domain.web_search.web_search import SearchEngine

        web = get_web_search()

        response = await web.search_async(
            query=request.query,
            engine=SearchEngine(request.engine),
            max_results=request.num_results,
        )

        # Extract results using list comprehension
        results = [
            {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("snippet", "")[:200],
            }
            for result in response.results
        ]

        # Generate summary if requested
        summary = await _generate_summary(results, request)

        # Convert to response models
        search_results = [SearchResult(**r) for r in results]

        return WebSearchResponse(
            query=request.query,
            results=search_results,
            num_results=len(results),
            summary=summary,
        )

    except Exception as e:
        logger.error(f"Web search error: {e}", exc_info=True)
        raise HTTPException(500, f"Web search error: {str(e)}")


@router.get("/engines")
async def list_engines() -> Dict[str, Any]:
    """List available search engines"""
    return {
        "engines": [
            {"id": "duckduckgo", "name": "DuckDuckGo", "description": "Privacy-focused search"},
            {"id": "google", "name": "Google", "description": "Requires API key"},
            {"id": "bing", "name": "Bing", "description": "Microsoft search"},
        ]
    }
