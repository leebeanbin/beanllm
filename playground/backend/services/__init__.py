# beanllm Playground Backend Services
"""
Services for the playground backend.

- encryption_service: API key encryption/decryption
- key_validator: API key validation
- config_service: Runtime configuration management
- intent_classifier: Intent classification for agentic routing
- tool_registry: Tool management and requirement checking
- orchestrator: Agentic tool execution and SSE streaming
- google_oauth_service: Google OAuth 2.0 flow management
- message_vector_store: Vector DB message storage
- session_search_service: Hybrid session search (MongoDB + Vector DB)
- session_cache: Redis-based session caching
- mcp_client_service: MCP tools direct invocation
- context_manager: Session context management with beanllm memory
"""

from services.config_service import ConfigService, config_service, init_config_on_startup
from services.context_manager import ContextManager, context_manager
from services.encryption_service import EncryptionService, encryption_service
from services.google_oauth_service import GOOGLE_SCOPES, GoogleOAuthService, google_oauth_service
from services.intent_classifier import IntentClassifier, IntentResult, IntentType, intent_classifier
from services.key_validator import KeyValidator, key_validator
from services.mcp_client_service import MCPClientService, mcp_client
from services.message_vector_store import (
    MessageVectorStore,
    message_vector_store,
)
from services.orchestrator import (
    AgenticEvent,
    AgenticOrchestrator,
    EventType,
    OrchestratorContext,
    orchestrator,
)
from services.prompt_builder import (
    PromptBuilder,
    PromptComposer,
    PromptOptimizer,
    PromptTemplate,
    prompt_builder,
)
from services.query_refiner import (
    QueryRefiner,
    RefinedQuery,
    query_refiner,
)
from services.session_cache import (
    SessionCacheService,
    session_cache,
)
from services.session_rag_service import (
    SessionRAGInfo,
    SessionRAGService,
    session_rag_service,
)
from services.session_search_service import (
    SessionSearchService,
    session_search,
)
from services.tool_registry import Tool, ToolCheckResult, ToolRegistry, ToolStatus, tool_registry

__all__ = [
    # Config & Security
    "EncryptionService",
    "encryption_service",
    "KeyValidator",
    "key_validator",
    "ConfigService",
    "config_service",
    "init_config_on_startup",
    # Agentic - Intent
    "IntentType",
    "IntentResult",
    "IntentClassifier",
    "intent_classifier",
    # Agentic - Tools
    "Tool",
    "ToolRegistry",
    "ToolCheckResult",
    "ToolStatus",
    "tool_registry",
    # Agentic - Orchestrator
    "AgenticOrchestrator",
    "AgenticEvent",
    "OrchestratorContext",
    "EventType",
    "orchestrator",
    # Google OAuth
    "GoogleOAuthService",
    "google_oauth_service",
    "GOOGLE_SCOPES",
    # MCP Client
    "MCPClientService",
    "mcp_client",
    # Context Manager
    "ContextManager",
    "context_manager",
    # Prompt Builder
    "PromptBuilder",
    "PromptTemplate",
    "PromptComposer",
    "PromptOptimizer",
    "prompt_builder",
    # Query Refiner
    "QueryRefiner",
    "RefinedQuery",
    "query_refiner",
    # Session RAG Service
    "SessionRAGService",
    "SessionRAGInfo",
    "session_rag_service",
    # Session Cache
    "SessionCacheService",
    "session_cache",
    # Session Search
    "SessionSearchService",
    "session_search",
    # Message Vector Store
    "MessageVectorStore",
    "message_vector_store",
]
