"""
Config & Health Router

System health checks and configuration endpoints.
"""

import logging
from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Dict, Any

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Config & Health"])


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "ok",
        "service": "beanllm-playground-api",
        "version": "1.0.0",
        "features": [
            "chat",
            "rag",
            "multi-agent",
            "knowledge-graph",
            "audio",
            "ocr",
            "google-workspace",
            "web-search",
            "evaluation",
            "finetuning",
        ],
    }


@router.get("/health")
async def health():
    """Health check endpoint"""
    try:
        # Import here to avoid circular dependency
        from common import (
            _client,
            _kg,
            _rag_chains,
            _web_search,
            _rag_debugger,
            _optimizer,
            _multi_agent,
            _orchestrator,
        )
        from database import ping_mongodb

        mongodb_status = await ping_mongodb()

        return {
            "status": "healthy",
            "services": {
                "mongodb": mongodb_status,
                "client": _client is not None,
                "kg": _kg is not None,
                "rag_chains": len(_rag_chains),
                "web_search": _web_search is not None,
                "rag_debugger": _rag_debugger is not None,
                "optimizer": _optimizer is not None,
                "multi_agent": _multi_agent is not None,
                "orchestrator": _orchestrator is not None,
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
        }


@router.get("/api/config/providers")
async def get_active_providers():
    """
    Get active providers based on available API keys.

    Returns:
        {
            "providers": ["openai", "anthropic", "ollama"],
            "config": {
                "openai_api_key": "***MASKED***",
                "anthropic_api_key": None,
                ...
            }
        }
    """
    try:
        from beanllm.utils.config import EnvConfig

        return {
            "providers": EnvConfig.get_active_providers(),
            "config": EnvConfig.get_safe_config_dict(),
        }
    except Exception as e:
        logger.error(f"Failed to get active providers: {e}")
        # Fallback: Ollama is always available
        return {
            "providers": ["ollama"],
            "config": {},
        }


@router.get("/api/config/models")
async def get_available_models():
    """
    Get available models per active provider.

    Returns:
        {
            "openai": ["gpt-4o", "gpt-4o-mini", ...],
            "anthropic": ["claude-sonnet-4", ...],
            "ollama": ["qwen2.5:0.5b", ...]
        }
    """
    try:
        from beanllm.utils.config import EnvConfig

        available_models = {}

        # OpenAI models
        if EnvConfig.is_provider_available("openai"):
            available_models["openai"] = [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
            ]

        # Anthropic models
        if EnvConfig.is_provider_available("anthropic"):
            available_models["anthropic"] = [
                "claude-sonnet-4-20250514",
                "claude-opus-4-20250514",
                "claude-haiku-4-20250514",
            ]

        # Google models
        if EnvConfig.is_provider_available("google"):
            available_models["google"] = [
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-1.5-pro",
            ]

        # DeepSeek models
        if EnvConfig.is_provider_available("deepseek"):
            available_models["deepseek"] = [
                "deepseek-chat",
                "deepseek-coder",
            ]

        # Perplexity models
        if EnvConfig.is_provider_available("perplexity"):
            available_models["perplexity"] = [
                "llama-3.1-sonar-large-128k-online",
                "llama-3.1-sonar-small-128k-online",
            ]

        # Ollama models (always available)
        try:
            from beanllm.providers.ollama import OllamaProvider

            ollama = OllamaProvider()
            ollama_models = ollama.list_models()
            available_models["ollama"] = [m["name"] for m in ollama_models]
        except Exception as e:
            logger.warning(f"Failed to list Ollama models: {e}")
            available_models["ollama"] = ["qwen2.5:0.5b"]  # Fallback

        return available_models

    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        return {"ollama": ["qwen2.5:0.5b"]}
