"""
API Routers

Modular API endpoints organized by feature.
"""

from .config_router import router as config_router
from .chat_router import router as chat_router
from .rag_router import router as rag_router
from .kg_router import router as kg_router
from .models_router import router as models_router
from .agent_router import router as agent_router
from .audio_router import router as audio_router
from .chain_router import router as chain_router
from .evaluation_router import router as evaluation_router
from .finetuning_router import router as finetuning_router
from .google_auth_router import router as google_auth_router
from .monitoring_router import router as monitoring_router
from .ocr_router import router as ocr_router
from .optimizer_router import router as optimizer_router
from .vision_router import router as vision_router
from .web_router import router as web_router

__all__ = [
    "config_router",
    "chat_router",
    "rag_router",
    "kg_router",
    "models_router",
    "agent_router",
    "audio_router",
    "chain_router",
    "evaluation_router",
    "finetuning_router",
    "google_auth_router",
    "monitoring_router",
    "ocr_router",
    "optimizer_router",
    "vision_router",
    "web_router",
]
