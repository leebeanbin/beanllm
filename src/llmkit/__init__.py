"""
llmkit - Unified toolkit for managing and using multiple LLM providers
환경변수 기반 LLM 모델 활성화 및 관리 패키지
"""

from .adapter import ParameterAdapter, adapt_parameters
from .agent import Agent, AgentResult, AgentStep, create_agent
from .audio_speech import (
    AudioRAG,
    AudioSegment,
    TextToSpeech,
    TranscriptionResult,
    TranscriptionSegment,
    TTSProvider,
    WhisperModel,
    WhisperSTT,
    text_to_speech,
    transcribe_audio,
)
from .callbacks import (
    BaseCallback,
    CallbackEvent,
    CallbackManager,
    CostTrackingCallback,
    FunctionCallback,
    LoggingCallback,
    StreamingCallback,
    TimingCallback,
    create_callback_manager,
)
from .chain import (
    Chain,
    ChainBuilder,
    ChainResult,
    ParallelChain,
    PromptChain,
    SequentialChain,
    create_chain,
)
from .client import ChatResponse, Client, create_client
from .document_loaders import (
    BaseDocumentLoader,
    CSVLoader,
    DirectoryLoader,
    Document,
    DocumentLoader,
    PDFLoader,
    TextLoader,
    load_documents,
)
from .embeddings import (
    BaseEmbedding,
    CohereEmbedding,
    Embedding,
    EmbeddingCache,
    EmbeddingResult,
    GeminiEmbedding,
    JinaEmbedding,
    MistralEmbedding,
    OllamaEmbedding,
    OpenAIEmbedding,
    VoyageEmbedding,
    embed,
    embed_sync,
    # Advanced features
    find_hard_negatives,
    mmr_search,
    query_expansion,
)
from .error_handling import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
    ErrorHandler,
    ErrorHandlerConfig,
    ErrorRecord,
    ErrorTracker,
    FallbackHandler,
    LLMKitError,
    MaxRetriesExceededError,
    ProviderError,
    RateLimitConfig,
    RateLimiter,
    RateLimitError,
    RetryConfig,
    RetryHandler,
    RetryStrategy,
    TimeoutError,
    ValidationError,
    circuit_breaker,
    fallback,
    get_error_tracker,
    rate_limit,
    retry,
    timeout,
    with_error_handling,
)
from .evaluation import (
    AnswerRelevanceMetric,
    BaseMetric,
    BatchEvaluationResult,
    BLEUMetric,
    ContextPrecisionMetric,
    CustomMetric,
    EvaluationResult,
    Evaluator,
    ExactMatchMetric,
    F1ScoreMetric,
    FaithfulnessMetric,
    LLMJudgeMetric,
    MetricType,
    ROUGEMetric,
    SemanticSimilarityMetric,
    create_evaluator,
    evaluate_rag,
    evaluate_text,
)
from .finetuning import (
    BaseFineTuningProvider,
    DatasetBuilder,
    DataValidator,
    FineTuningConfig,
    FineTuningCostEstimator,
    FineTuningJob,
    FineTuningManager,
    FineTuningMetrics,
    FineTuningStatus,
    ModelProvider,
    OpenAIFineTuningProvider,
    TrainingExample,
    create_finetuning_provider,
    quick_finetune,
)
from .graph import (
    AgentNode,
    BaseNode,
    ConditionalNode,
    FunctionNode,
    GraderNode,
    Graph,
    GraphState,
    LLMNode,
    LoopNode,
    NodeCache,
    ParallelNode,
    create_simple_graph,
)
from .hybrid_manager import HybridModelManager, create_hybrid_manager
from .inferrer import MetadataInferrer
from .memory import (
    BaseMemory,
    BufferMemory,
    ConversationMemory,
    Message,
    SummaryMemory,
    TokenMemory,
    WindowMemory,
    create_memory,
)
from .ml_models import (
    BaseMLModel,
    MLModelFactory,
    PyTorchModel,
    SklearnModel,
    TensorFlowModel,
    load_ml_model,
)
from .model_info import ModelCapabilityInfo, ProviderInfo
from .multi_agent import (
    AgentMessage,
    CommunicationBus,
    CoordinationStrategy,
    DebateStrategy,
    HierarchicalStrategy,
    MessageType,
    MultiAgentCoordinator,
    ParallelStrategy,
    SequentialStrategy,
    create_coordinator,
    quick_debate,
)
from .output_parsers import (
    BaseOutputParser,
    BooleanOutputParser,
    CommaSeparatedListOutputParser,
    DatetimeOutputParser,
    EnumOutputParser,
    JSONOutputParser,
    NumberedListOutputParser,
    OutputParserException,
    PydanticOutputParser,
    RetryOutputParser,
    parse_bool,
    parse_json,
    parse_list,
)
from .prompts import (
    BasePromptTemplate,
    ChatMessage,
    ChatPromptTemplate,
    ExampleSelector,
    FewShotPromptTemplate,
    PredefinedTemplates,
    PromptCache,
    PromptComposer,
    PromptExample,
    PromptOptimizer,
    PromptTemplate,
    PromptVersioning,
    SystemMessageTemplate,
    TemplateFormat,
    clear_cache,
    create_chat_template,
    create_few_shot_template,
    create_prompt_template,
    get_cache_stats,
    get_cached_prompt,
)
from .provider_factory import ProviderFactory
from .rag_chain import RAG, RAGBuilder, RAGChain, RAGResponse, create_rag
from .rag_debug import (
    EmbeddingInfo,
    RAGDebugger,
    SimilarityInfo,
    compare_texts,
    inspect_embedding,
    similarity_heatmap,
    validate_pipeline,
    visualize_embeddings_2d,
)
from .registry import ModelRegistry
from .registry import get_model_registry as get_registry
from .scanner import ModelScanner
from .state_graph import (
    END,
    Checkpoint,
    GraphConfig,
    GraphExecution,
    NodeExecution,
    StateGraph,
    create_state_graph,
)
from .streaming import (
    StreamBuffer,
    StreamResponse,
    StreamStats,
    pretty_stream,
    stream_collect,
    stream_print,
    stream_response,
)
from .text_splitters import (
    BaseTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
    TokenTextSplitter,
    split_documents,
)
from .token_counter import (
    CostEstimate,
    CostEstimator,
    ModelContextWindow,
    ModelPricing,
    TokenCounter,
    count_message_tokens,
    count_tokens,
    estimate_cost,
    get_cheapest_model,
    get_context_window,
)
from .tools import Tool, ToolParameter, ToolRegistry, get_all_tools, get_tool, register_tool
from .tools_advanced import (
    APIConfig,
    APIProtocol,
    ExternalAPITool,
    SchemaGenerator,
    ToolChain,
    ToolValidator,
    default_registry,
    tool,
)
from .tools_advanced import ToolRegistry as AdvancedToolRegistry
from .tracer import Trace, Tracer, TraceSpan, enable_tracing, get_tracer
from .vector_stores import (
    BaseVectorStore,
    ChromaVectorStore,
    FAISSVectorStore,
    PineconeVectorStore,
    QdrantVectorStore,
    VectorSearchResult,
    VectorStore,
    VectorStoreBuilder,
    WeaviateVectorStore,
    create_vector_store,
    from_documents,
)
from .vision_embeddings import CLIPEmbedding, MultimodalEmbedding, create_vision_embedding
from .vision_loaders import (
    ImageDocument,
    ImageLoader,
    PDFWithImagesLoader,
    load_images,
    load_pdf_with_images,
)
from .vision_rag import MultimodalRAG, VisionRAG, create_vision_rag
from .web_search import (
    BaseSearchEngine,
    BingSearch,
    DuckDuckGoSearch,
    GoogleSearch,
    SearchEngine,
    SearchResponse,
    SearchResult,
    WebScraper,
    WebSearch,
    search_web,
)

__version__ = "0.1.0"
__all__ = [
    "ModelRegistry",
    "get_registry",
    "get_model_registry",  # 하위 호환성
    "ProviderFactory",
    "ModelCapabilityInfo",
    "ProviderInfo",
    "HybridModelManager",
    "create_hybrid_manager",
    "MetadataInferrer",
    "ModelScanner",
    "Client",
    "create_client",
    "ChatResponse",
    "ParameterAdapter",
    "adapt_parameters",
    # Streaming
    "stream_response",
    "stream_print",
    "stream_collect",
    "pretty_stream",
    "StreamResponse",
    "StreamStats",
    "StreamBuffer",
    # Tracer
    "Tracer",
    "get_tracer",
    "enable_tracing",
    "Trace",
    "TraceSpan",
    # Tools (NEW!)
    "Tool",
    "ToolParameter",
    "ToolRegistry",
    "register_tool",
    "get_tool",
    "get_all_tools",
    # Agent (NEW!)
    "Agent",
    "AgentStep",
    "AgentResult",
    "create_agent",
    # Memory (NEW!)
    "BaseMemory",
    "BufferMemory",
    "WindowMemory",
    "TokenMemory",
    "SummaryMemory",
    "ConversationMemory",
    "Message",
    "create_memory",
    # Chain (NEW!)
    "Chain",
    "PromptChain",
    "SequentialChain",
    "ParallelChain",
    "ChainBuilder",
    "ChainResult",
    "create_chain",
    # Output Parsers (NEW!)
    "BaseOutputParser",
    "PydanticOutputParser",
    "JSONOutputParser",
    "CommaSeparatedListOutputParser",
    "NumberedListOutputParser",
    "DatetimeOutputParser",
    "EnumOutputParser",
    "BooleanOutputParser",
    "RetryOutputParser",
    "OutputParserException",
    "parse_json",
    "parse_list",
    "parse_bool",
    # Document Loaders (NEW!)
    "Document",
    "BaseDocumentLoader",
    "TextLoader",
    "PDFLoader",
    "CSVLoader",
    "DirectoryLoader",
    "DocumentLoader",
    "load_documents",
    # Text Splitters (NEW!)
    "BaseTextSplitter",
    "CharacterTextSplitter",
    "RecursiveCharacterTextSplitter",
    "TokenTextSplitter",
    "MarkdownHeaderTextSplitter",
    "TextSplitter",
    "split_documents",
    # Embeddings (NEW!)
    "BaseEmbedding",
    "OpenAIEmbedding",
    "GeminiEmbedding",
    "OllamaEmbedding",
    "VoyageEmbedding",
    "JinaEmbedding",
    "MistralEmbedding",
    "CohereEmbedding",
    "Embedding",
    "EmbeddingResult",
    "embed",
    "embed_sync",
    "find_hard_negatives",
    "mmr_search",
    "query_expansion",
    "EmbeddingCache",
    # Vector Stores (NEW!)
    "BaseVectorStore",
    "ChromaVectorStore",
    "PineconeVectorStore",
    "FAISSVectorStore",
    "QdrantVectorStore",
    "WeaviateVectorStore",
    "VectorStore",
    "VectorStoreBuilder",
    "VectorSearchResult",
    "create_vector_store",
    "from_documents",
    # RAG Debug Utils (NEW!)
    "RAGDebugger",
    "EmbeddingInfo",
    "SimilarityInfo",
    "inspect_embedding",
    "compare_texts",
    "validate_pipeline",
    "visualize_embeddings_2d",
    "similarity_heatmap",
    # RAG Chain (NEW!)
    "RAGChain",
    "RAGBuilder",
    "RAGResponse",
    "create_rag",
    "RAG",
    # StateGraph (NEW!)
    "StateGraph",
    "END",
    "Checkpoint",
    "GraphConfig",
    "GraphExecution",
    "NodeExecution",
    "create_state_graph",
    # Callbacks (NEW!)
    "BaseCallback",
    "LoggingCallback",
    "CostTrackingCallback",
    "TimingCallback",
    "StreamingCallback",
    "FunctionCallback",
    "CallbackManager",
    "CallbackEvent",
    "create_callback_manager",
    # Vision (NEW!)
    "ImageDocument",
    "ImageLoader",
    "PDFWithImagesLoader",
    "load_images",
    "load_pdf_with_images",
    "CLIPEmbedding",
    "MultimodalEmbedding",
    "create_vision_embedding",
    "VisionRAG",
    "MultimodalRAG",
    "create_vision_rag",
    # ML Models (NEW!)
    "BaseMLModel",
    "TensorFlowModel",
    "PyTorchModel",
    "SklearnModel",
    "MLModelFactory",
    "load_ml_model",
    # Graph & Nodes (NEW!)
    "GraphState",
    "NodeCache",
    "BaseNode",
    "FunctionNode",
    "AgentNode",
    "LLMNode",
    "GraderNode",
    "ConditionalNode",
    "LoopNode",
    "ParallelNode",
    "Graph",
    "create_simple_graph",
    # Multi-Agent (NEW!)
    "MessageType",
    "AgentMessage",
    "CommunicationBus",
    "CoordinationStrategy",
    "SequentialStrategy",
    "ParallelStrategy",
    "HierarchicalStrategy",
    "DebateStrategy",
    "MultiAgentCoordinator",
    "create_coordinator",
    "quick_debate",
    # Advanced Tools (NEW!)
    "SchemaGenerator",
    "ToolValidator",
    "APIProtocol",
    "APIConfig",
    "ExternalAPITool",
    "ToolChain",
    "tool",
    "AdvancedToolRegistry",
    "default_registry",
    # Web Search (NEW!)
    "SearchResult",
    "SearchResponse",
    "SearchEngine",
    "BaseSearchEngine",
    "GoogleSearch",
    "BingSearch",
    "DuckDuckGoSearch",
    "WebScraper",
    "WebSearch",
    "search_web",
    # Audio & Speech (NEW!)
    "AudioSegment",
    "TranscriptionSegment",
    "TranscriptionResult",
    "WhisperModel",
    "WhisperSTT",
    "TTSProvider",
    "TextToSpeech",
    "AudioRAG",
    "transcribe_audio",
    "text_to_speech",
    # Token Counting & Cost (NEW!)
    "TokenCounter",
    "CostEstimator",
    "CostEstimate",
    "ModelPricing",
    "ModelContextWindow",
    "count_tokens",
    "count_message_tokens",
    "estimate_cost",
    "get_cheapest_model",
    "get_context_window",
    # Prompt Templates (NEW!)
    "TemplateFormat",
    "PromptExample",
    "BasePromptTemplate",
    "PromptTemplate",
    "FewShotPromptTemplate",
    "ChatMessage",
    "ChatPromptTemplate",
    "SystemMessageTemplate",
    "PromptComposer",
    "PromptOptimizer",
    "PredefinedTemplates",
    "ExampleSelector",
    "PromptVersioning",
    "PromptCache",
    "create_prompt_template",
    "create_chat_template",
    "create_few_shot_template",
    "get_cached_prompt",
    "get_cache_stats",
    "clear_cache",
    # Evaluation Metrics (NEW!)
    "MetricType",
    "EvaluationResult",
    "BatchEvaluationResult",
    "BaseMetric",
    "ExactMatchMetric",
    "F1ScoreMetric",
    "BLEUMetric",
    "ROUGEMetric",
    "SemanticSimilarityMetric",
    "LLMJudgeMetric",
    "AnswerRelevanceMetric",
    "ContextPrecisionMetric",
    "FaithfulnessMetric",
    "CustomMetric",
    "Evaluator",
    "evaluate_text",
    "evaluate_rag",
    "create_evaluator",
    # Fine-tuning (NEW!)
    "FineTuningStatus",
    "ModelProvider",
    "TrainingExample",
    "FineTuningConfig",
    "FineTuningJob",
    "FineTuningMetrics",
    "BaseFineTuningProvider",
    "OpenAIFineTuningProvider",
    "DatasetBuilder",
    "DataValidator",
    "FineTuningManager",
    "FineTuningCostEstimator",
    "create_finetuning_provider",
    "quick_finetune",
    # Error Handling (NEW!)
    "LLMKitError",
    "ProviderError",
    "RateLimitError",
    "TimeoutError",
    "ValidationError",
    "CircuitBreakerError",
    "MaxRetriesExceededError",
    "RetryStrategy",
    "RetryConfig",
    "RetryHandler",
    "retry",
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "circuit_breaker",
    "RateLimitConfig",
    "RateLimiter",
    "rate_limit",
    "FallbackHandler",
    "fallback",
    "ErrorRecord",
    "ErrorTracker",
    "get_error_tracker",
    "ErrorHandlerConfig",
    "ErrorHandler",
    "with_error_handling",
    "timeout",
]

# 하위 호환성을 위한 별칭
get_model_registry = get_registry


# 설치 안내 메시지 (선택적 의존성)
def _check_optional_dependencies():
    """선택적 의존성 확인 및 안내 (디자인 시스템 적용)"""
    import sys

    # UI 모듈 import (에러 발생해도 import는 성공)
    try:
        from .ui import InfoPattern

        use_ui = True
    except ImportError:
        use_ui = False

    missing = []

    # 선택적 의존성 체크 (import 없이)
    from importlib.util import find_spec

    if find_spec("google.generativeai") is None:
        missing.append("gemini")

    if find_spec("ollama") is None:
        missing.append("ollama")

    if missing and not hasattr(sys, "_llmkit_install_warned"):
        sys._llmkit_install_warned = True

        if use_ui:
            # 디자인 시스템 사용
            install_commands = []
            for pkg in missing:
                if pkg == "gemini":
                    install_commands.append("pip install llmkit[gemini]")
                elif pkg == "ollama":
                    install_commands.append("pip install llmkit[ollama]")

            InfoPattern.render(
                "Some provider SDKs are not installed",
                details=[f"Install: {cmd}" for cmd in install_commands]
                + ["Or install all: pip install llmkit[all]"],
            )
        else:
            # 기본 출력 (UI 없을 때)
            print("\n" + "=" * 60)
            print("📦 llmkit - Optional Provider SDKs")
            print("=" * 60)
            print("\nℹ️  Some provider SDKs are not installed:")
            for pkg in missing:
                if pkg == "gemini":
                    print("  • Gemini: pip install llmkit[gemini]")
                elif pkg == "ollama":
                    print("  • Ollama: pip install llmkit[ollama]")
            print("\nOr install all providers:")
            print("  pip install llmkit[all]")
            print("\n" + "=" * 60 + "\n")


def _print_welcome_banner():
    """환영 배너 출력 (선택적, 환경변수로 제어) - 디자인 시스템 적용"""
    import os

    # 환경변수로 제어 (기본값: False)
    if not os.getenv("LLMKIT_SHOW_BANNER", "false").lower() == "true":
        return

    try:
        from .ui import OnboardingPattern, print_logo
    except ImportError:
        return  # UI 없으면 출력 안 함

    # 로고 출력
    print_logo(style="minimal", color="magenta")

    # 온보딩 패턴
    OnboardingPattern.render(
        "Welcome to llmkit!",
        steps=[
            {
                "title": "Set environment variables",
                "description": "export OPENAI_API_KEY='your-key'",
            },
            {
                "title": "Try it out",
                "description": "from llmkit import get_registry; r = get_registry()",
            },
            {"title": "Use CLI", "description": "llmkit list"},
        ],
    )


# 패키지 import 시 확인 (경고만, 에러 아님)
try:
    _check_optional_dependencies()
    _print_welcome_banner()
except Exception:
    pass  # 에러 발생해도 import는 성공
