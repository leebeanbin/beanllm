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
from .ml_router import router as ml_router

__all__ = [
    "config_router",
    "chat_router",
    "rag_router",
    "kg_router",
    "models_router",
    "agent_router",
    "ml_router",
]
