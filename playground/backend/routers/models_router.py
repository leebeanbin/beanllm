"""
Models Router

Model management endpoints (list, pull, analyze, scan)
"""

import logging
import json
from urllib.parse import unquote
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any

from common import get_ollama_model_name_for_chat, track_downloaded_model, get_downloaded_models

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["Models"])


@router.get("")
async def get_models() -> Dict[str, Any]:
    """Get all available models grouped by provider"""
    try:
        from beanllm.infrastructure.models.models import get_all_models
        from beanllm.providers.ollama_provider import OllamaProvider

        models = get_all_models()
        downloaded_models = get_downloaded_models()

        # Group by provider
        grouped = {}
        for model_name, model_info in models.items():
            provider = model_info["provider"]
            if provider not in grouped:
                grouped[provider] = []
            grouped[provider].append({
                "name": model_name,
                "display_name": model_info["display_name"],
                "description": model_info["description"],
                "use_case": model_info["use_case"],
                "max_tokens": model_info["max_tokens"],
                "type": model_info["type"],
                "provider": provider,
                "installed": None,
            })

        # Check Ollama installed models
        try:
            ollama_provider = OllamaProvider()
            installed_models = await ollama_provider.list_models()

            # Add downloaded models to installed list
            for display_name, ollama_name in downloaded_models.items():
                if ollama_name not in installed_models:
                    installed_models.append(ollama_name)

            installed_set = set(m.lower() for m in installed_models)

            # Update installed status for Ollama models
            for model in grouped.get("ollama", []):
                model_name = model.get("name")
                mapped_name = get_ollama_model_name_for_chat(model_name)

                if (model_name.lower() in installed_set or
                    mapped_name.lower() in installed_set or
                    model_name in downloaded_models):
                    model["installed"] = True
                else:
                    model["installed"] = False

            # Add installed models not in our list
            existing_names = {m["name"].lower() for m in grouped.get("ollama", [])}
            for installed_model in installed_models:
                if installed_model.lower() not in existing_names:
                    if "ollama" not in grouped:
                        grouped["ollama"] = []
                    grouped["ollama"].append({
                        "name": installed_model,
                        "display_name": installed_model,
                        "description": f"Installed Ollama model: {installed_model}",
                        "use_case": "chat",
                        "max_tokens": 4096,
                        "type": "llm",
                        "installed": True,
                        "provider": "ollama",
                    })

        except Exception as e:
            logger.debug(f"Failed to scan Ollama models: {e}")
            for model in grouped.get("ollama", []):
                if model.get("installed") is None:
                    model["installed"] = False

        return grouped
    except Exception as e:
        raise HTTPException(500, f"Failed to get models: {str(e)}")


@router.get("/{provider}")
async def get_models_by_provider(provider: str) -> Dict[str, Any]:
    """Get models for a specific provider"""
    try:
        from beanllm.infrastructure.models.models import get_models_by_provider as get_provider_models

        models = get_provider_models(provider)

        return {
            "provider": provider,
            "models": [
                {
                    "name": model_name,
                    "display_name": model_info["display_name"],
                    "description": model_info["description"],
                    "use_case": model_info["use_case"],
                    "max_tokens": model_info["max_tokens"],
                    "type": model_info["type"],
                }
                for model_name, model_info in models.items()
            ],
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to get models for provider {provider}: {str(e)}")


@router.get("/{model_name}/parameters")
async def get_model_parameters(model_name: str) -> Dict[str, Any]:
    """Get parameter support information for a specific model"""
    try:
        from beanllm import get_registry

        model_name = unquote(model_name)
        registry = get_registry()
        model_info = registry.get_model_info(model_name)

        if not model_info:
            # Infer provider from model name
            provider = _infer_provider(model_name)
            defaults = _get_provider_defaults(provider)

            return {
                "model": model_name,
                "provider": provider,
                "supports": defaults["supports"],
                "max_tokens": defaults["max_tokens"],
                "default_temperature": defaults["default_temperature"],
                "uses_max_completion_tokens": False,
            }

        provider = model_info.provider.lower()
        supports = {
            "temperature": model_info.supports_temperature,
            "max_tokens": model_info.supports_max_tokens,
            "top_p": True,
            "frequency_penalty": provider in ["openai", "deepseek", "perplexity"],
            "presence_penalty": provider in ["openai", "deepseek", "perplexity"],
        }

        return {
            "model": model_name,
            "provider": model_info.provider,
            "supports": supports,
            "max_tokens": model_info.max_tokens,
            "default_temperature": model_info.default_temperature,
            "uses_max_completion_tokens": model_info.uses_max_completion_tokens,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model parameters for {model_name}: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to get model parameters: {str(e)}")


@router.post("/scan")
async def scan_models() -> Dict[str, Any]:
    """Scan APIs for new models"""
    try:
        from beanllm.infrastructure.hybrid import create_hybrid_manager

        manager = create_hybrid_manager()
        results = await manager.scan_all_providers()

        return {"status": "success", "results": results, "message": "Model scan completed"}
    except Exception as e:
        raise HTTPException(500, f"Failed to scan models: {str(e)}")


@router.post("/{model_name}/pull")
async def pull_model(model_name: str):
    """Download an Ollama model"""
    try:
        from beanllm.providers.ollama_provider import OllamaProvider

        model_name = unquote(model_name)
        ollama_model_name = model_name
        ollama_provider = OllamaProvider()

        # Fail fast if Ollama is not reachable (avoid 200 + stream then error)
        if not await ollama_provider.health_check():
            raise HTTPException(
                503,
                "Cannot connect to Ollama. Ensure Ollama is running (e.g. run `ollama serve` or start the Ollama app).",
            )

        logger.info(f"Pulling Ollama model: {ollama_model_name}")

        async def generate():
            try:
                pull_stream = await ollama_provider.client.pull(
                    model=ollama_model_name, stream=True
                )

                async for chunk in pull_stream:
                    status = getattr(chunk, 'status', None) if not isinstance(chunk, dict) else chunk.get('status')
                    completed = getattr(chunk, 'completed', 0) if not isinstance(chunk, dict) else chunk.get('completed', 0)
                    total = getattr(chunk, 'total', 0) if not isinstance(chunk, dict) else chunk.get('total', 0)

                    if status is None:
                        continue

                    progress = 0
                    if completed and total and total > 0:
                        progress = completed / total * 100
                    elif status == "success":
                        progress = 100

                    progress_data = {
                        "status": status,
                        "completed": completed or 0,
                        "total": total or 0,
                        "progress": round(progress, 2),
                    }
                    yield f"data: {json.dumps(progress_data)}\n\n"

                    if status == "success":
                        mapped_name = get_ollama_model_name_for_chat(model_name)
                        track_downloaded_model(model_name, mapped_name)
                        yield f"data: {json.dumps({'status': 'completed', 'model': model_name, 'ollama_model': mapped_name})}\n\n"
                        break

            except Exception as e:
                logger.error(f"Failed to pull model {model_name}: {e}")
                yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    except Exception as e:
        logger.error(f"Pull model error: {e}")
        raise HTTPException(500, f"Failed to pull model: {str(e)}")


@router.post("/{model_name}/analyze")
async def analyze_model(model_name: str) -> Dict[str, Any]:
    """Analyze model with pattern inference"""
    try:
        from beanllm import get_registry

        registry = get_registry()
        model_info = registry.get_model_info(model_name)

        if not model_info:
            raise HTTPException(404, f"Model {model_name} not found")

        analysis = {
            "model": model_name,
            "provider": model_info.provider,
            "type": model_info.model_type,
            "capabilities": {
                "streaming": model_info.supports_streaming,
                "temperature": model_info.supports_temperature,
                "max_tokens": model_info.supports_max_tokens,
                "uses_max_completion_tokens": model_info.uses_max_completion_tokens,
            },
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "supported": p.supported,
                    "description": p.description,
                    "default": p.default,
                }
                for p in model_info.parameters
            ],
            "max_tokens": model_info.max_tokens,
            "default_temperature": model_info.default_temperature,
        }

        return {"status": "success", "analysis": analysis}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to analyze model: {str(e)}")


def _infer_provider(model_name: str) -> str:
    """Infer provider from model name"""
    if ":" in model_name or model_name.startswith(("qwen", "phi", "llama", "mistral", "gemma")):
        return "ollama"
    elif model_name.startswith(("gpt", "o1", "o3")):
        return "openai"
    elif model_name.startswith("claude"):
        return "anthropic"
    elif model_name.startswith("gemini"):
        return "google"
    elif model_name.startswith("deepseek"):
        return "deepseek"
    elif model_name.startswith("sonar"):
        return "perplexity"
    return "unknown"


def _get_provider_defaults(provider: str) -> Dict[str, Any]:
    """Get default parameter support for a provider"""
    defaults = {
        "ollama": {
            "supports": {
                "temperature": True,
                "max_tokens": True,
                "top_p": True,
                "frequency_penalty": False,
                "presence_penalty": False,
            },
            "max_tokens": 4096,
            "default_temperature": 0.7,
        },
        "openai": {
            "supports": {
                "temperature": True,
                "max_tokens": True,
                "top_p": True,
                "frequency_penalty": True,
                "presence_penalty": True,
            },
            "max_tokens": 4096,
            "default_temperature": 0.7,
        },
        "anthropic": {
            "supports": {
                "temperature": True,
                "max_tokens": True,
                "top_p": True,
                "frequency_penalty": False,
                "presence_penalty": False,
            },
            "max_tokens": 4096,
            "default_temperature": 0.7,
        },
        "google": {
            "supports": {
                "temperature": True,
                "max_tokens": True,
                "top_p": True,
                "frequency_penalty": False,
                "presence_penalty": False,
            },
            "max_tokens": 8192,
            "default_temperature": 0.7,
        },
    }
    return defaults.get(provider, defaults["ollama"])
