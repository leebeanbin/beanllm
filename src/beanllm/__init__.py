"""
beanllm - Unified toolkit for managing and using multiple LLM providers
환경변수 기반 LLM 모델 활성화 및 관리 패키지
"""

# Infrastructure - infrastructure/__init__.py에서 통합 export
# Domain
from .domain import (
    END,
    AdvancedToolRegistry,
    # Multi-Agent
    AgentMessage,
    # Graph
    AgentNode,
    # Evaluation
    AnswerRelevanceMetric,
    # Advanced Tools
    APIConfig,
    APIProtocol,
    # Audio
    AudioSegment,
    # Document Loaders
    BaseDocumentLoader,
    # Embeddings
    BaseEmbedding,
    # Fine-tuning
    BaseFineTuningProvider,
    # Memory
    BaseMemory,
    BaseMetric,
    BaseNode,
    # Output Parsers
    BaseOutputParser,
    # Prompts
    BasePromptTemplate,
    # Web Search
    BaseSearchEngine,
    # Text Splitters
    BaseTextSplitter,
    # Vector Stores
    BaseVectorStore,
    BatchEvaluationResult,
    BingSearch,
    BLEUMetric,
    BooleanOutputParser,
    BufferMemory,
    CharacterTextSplitter,
    ChatMessage,
    ChatPromptTemplate,
    # State Graph
    Checkpoint,
    ChromaVectorStore,
    # Vision
    CLIPEmbedding,
    CohereEmbedding,
    CommaSeparatedListOutputParser,
    CommunicationBus,
    ConditionalNode,
    ContextPrecisionMetric,
    ConversationMemory,
    CoordinationStrategy,
    CSVLoader,
    CustomMetric,
    DatasetBuilder,
    DataValidator,
    DatetimeOutputParser,
    DebateStrategy,
    DirectoryLoader,
    Document,
    DocumentLoader,
    DuckDuckGoSearch,
    Embedding,
    EmbeddingCache,
    EmbeddingResult,
    EnumOutputParser,
    EvaluationResult,
    ExactMatchMetric,
    ExampleSelector,
    ExternalAPITool,
    F1ScoreMetric,
    FAISSVectorStore,
    FaithfulnessMetric,
    FewShotPromptTemplate,
    FineTuningConfig,
    FineTuningCostEstimator,
    FineTuningJob,
    FineTuningMetrics,
    FineTuningStatus,
    FunctionNode,
    GeminiEmbedding,
    GoogleSearch,
    GraderNode,
    GraphConfig,
    GraphExecution,
    GraphState,
    HierarchicalStrategy,
    ImageDocument,
    ImageLoader,
    JinaEmbedding,
    JSONOutputParser,
    LLMJudgeMetric,
    LLMNode,
    LoopNode,
    MarkdownHeaderTextSplitter,
    Message,
    MessageType,
    MetricType,
    MistralEmbedding,
    ModelProvider,
    MultimodalEmbedding,
    NodeCache,
    NodeExecution,
    NumberedListOutputParser,
    OllamaEmbedding,
    OpenAIEmbedding,
    OpenAIFineTuningProvider,
    OutputParserException,
    ParallelNode,
    ParallelStrategy,
    PDFLoader,
    PDFWithImagesLoader,
    PineconeVectorStore,
    PredefinedTemplates,
    PromptCache,
    PromptComposer,
    PromptExample,
    PromptOptimizer,
    PromptTemplate,
    PromptVersioning,
    PydanticOutputParser,
    QdrantVectorStore,
    RecursiveCharacterTextSplitter,
    RetryOutputParser,
    ROUGEMetric,
    SchemaGenerator,
    SearchEngine,
    SearchResponse,
    SearchResult,
    SemanticSimilarityMetric,
    SequentialStrategy,
    SummaryMemory,
    SystemMessageTemplate,
    TemplateFormat,
    TextLoader,
    TextSplitter,
    TokenMemory,
    TokenTextSplitter,
    # Tools
    Tool,
    ToolChain,
    ToolParameter,
    ToolRegistry,
    ToolValidator,
    TrainingExample,
    TranscriptionResult,
    TranscriptionSegment,
    TTSProvider,
    VectorSearchResult,
    VectorStore,
    VectorStoreBuilder,
    VoyageEmbedding,
    WeaviateVectorStore,
    WebScraper,
    WhisperModel,
    WindowMemory,
    batch_cosine_similarity,
    calculator,
    clear_cache,
    cosine_similarity,
    create_chat_template,
    create_few_shot_template,
    create_memory,
    create_prompt_template,
    create_vector_store,
    create_vision_embedding,
    default_registry,
    echo,
    embed,
    embed_sync,
    euclidean_distance,
    find_hard_negatives,
    from_documents,
    get_all_tools,
    get_cache_stats,
    get_cached_prompt,
    get_current_time,
    get_tool,
    load_documents,
    load_images,
    load_pdf_with_images,
    mmr_search,
    normalize_vector,
    parse_bool,
    parse_json,
    parse_list,
    query_expansion,
    register_tool,
    # search_web는 domain.tools에서 이미 import됨
    split_documents,
    tool,
)

# DTO
from .dto.response.rag_response import RAGResponse

# Facade
from .facade import Agent, Client, RAGChain
from .facade.agent_facade import AgentResult, AgentStep, create_agent

# Audio Facade
from .facade.audio_facade import (
    AudioRAG,
    TextToSpeech,
    WhisperSTT,
    text_to_speech,
    transcribe_audio,
)
from .facade.chain_facade import (
    Chain,
    ChainBuilder,
    ChainResult,
    ParallelChain,
    PromptChain,
    SequentialChain,
    create_chain,
)
from .facade.client_facade import ChatResponse, create_client
from .facade.evaluation_facade import (
    Evaluator,
    create_evaluator,
    evaluate_rag,
    evaluate_text,
)
from .facade.finetuning_facade import (
    FineTuningManagerFacade,
    create_finetuning_provider,
    quick_finetune,
)
from .facade.graph_facade import Graph, create_simple_graph
from .facade.multi_agent_facade import (
    MultiAgentCoordinator,
    create_coordinator,
    quick_debate,
)
from .facade.rag_facade import RAG, RAGBuilder, create_rag
from .facade.state_graph_facade import StateGraph, create_state_graph
from .facade.vision_rag_facade import MultimodalRAG, VisionRAG, create_vision_rag
from .facade.web_search_facade import WebSearch

# search_web는 domain.tools에서 이미 import됨
from .infrastructure import (
    MODELS,
    AdaptedParameters,
    # ML Models
    BaseMLModel,
    HybridModelInfo,
    HybridModelManager,
    MetadataInferrer,
    MLModelFactory,
    ModelCapabilityInfo,
    ModelRegistry,
    ModelScanner,
    ModelStatus,
    ParameterAdapter,
    ParameterInfo,
    ProviderFactory,
    ProviderInfo,
    PyTorchModel,
    ScannedModel,
    SklearnModel,
    TensorFlowModel,
    adapt_parameters,
    create_hybrid_manager,
    get_all_models,
    get_default_model,
    get_model_registry,
    get_models_by_provider,
    get_models_by_type,
    load_ml_model,
    validate_parameters,
)

# Utils
from .utils import (
    # Callbacks
    BaseCallback,
    CallbackEvent,
    CallbackManager,
    # Error Handling (retry_decorator는 retry와 동일)
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
    # Config
    Config,
    # Token Counter
    CostEstimate,
    CostEstimator,
    CostTrackingCallback,
    # RAG Debug
    EmbeddingInfo,
    EnvConfig,
    ErrorHandler,
    ErrorHandlerConfig,
    ErrorRecord,
    ErrorTracker,
    FallbackHandler,
    FunctionCallback,
    LLMKitError,
    LoggingCallback,
    MaxRetriesExceededError,
    ModelContextWindow,
    ModelPricing,
    RAGDebugger,
    RateLimitConfig,
    RateLimiter,
    RateLimitError,
    RetryConfig,
    RetryHandler,
    RetryStrategy,
    SimilarityInfo,
    # Streaming
    StreamBuffer,
    StreamingCallback,
    StreamResponse,
    StreamStats,
    TimeoutError,
    TimingCallback,
    TokenCounter,
    # Tracer
    Trace,
    Tracer,
    TraceSpan,
    ValidationError,
    circuit_breaker,
    compare_texts,
    count_message_tokens,
    count_tokens,
    create_callback_manager,
    enable_tracing,
    estimate_cost,
    fallback,
    get_cheapest_model,
    get_context_window,
    get_error_tracker,
    # Exceptions (utils에서 이미 import됨, 중복 제거)
    # ModelNotFoundError,
    # ProviderError as UtilsProviderError,
    # RateLimitError as UtilsRateLimitError,
    # Logger
    get_logger,
    get_tracer,
    inspect_embedding,
    # CLI
    main,
    pretty_stream,
    rate_limit,
    # Retry
    retry,
    # retry_decorator는 retry와 동일하므로 제거
    similarity_heatmap,
    stream_collect,
    stream_print,
    stream_response,
    timeout,
    validate_pipeline,
    visualize_embeddings_2d,
    with_error_handling,
)

# 하위 호환성
FineTuningManager = FineTuningManagerFacade
get_registry = get_model_registry

__version__ = "0.1.0"
__all__ = [
    # Infrastructure
    "ParameterAdapter",
    "adapt_parameters",
    "validate_parameters",
    "AdaptedParameters",
    "ModelRegistry",
    "get_model_registry",
    "get_registry",
    "ProviderFactory",
    "MODELS",
    "get_all_models",
    "get_models_by_provider",
    "get_models_by_type",
    "get_default_model",
    "ModelStatus",
    "ParameterInfo",
    "ProviderInfo",
    "ModelCapabilityInfo",
    "HybridModelManager",
    "create_hybrid_manager",
    "HybridModelInfo",
    "MetadataInferrer",
    "ModelScanner",
    "ScannedModel",
    "BaseMLModel",
    "TensorFlowModel",
    "PyTorchModel",
    "SklearnModel",
    "MLModelFactory",
    "load_ml_model",
    # Facade
    "Client",
    "create_client",
    "ChatResponse",
    "Agent",
    "AgentStep",
    "AgentResult",
    "create_agent",
    "RAGChain",
    "RAG",
    "RAGBuilder",
    "create_rag",
    "RAGResponse",
    "Chain",
    "PromptChain",
    "SequentialChain",
    "ParallelChain",
    "ChainBuilder",
    "ChainResult",
    "create_chain",
    "Graph",
    "create_simple_graph",
    "StateGraph",
    "create_state_graph",
    "MultiAgentCoordinator",
    "create_coordinator",
    "quick_debate",
    "VisionRAG",
    "MultimodalRAG",
    "create_vision_rag",
    "WebSearch",
    # "search_web",  # domain.tools에서 이미 export됨
    "AudioRAG",
    "TextToSpeech",
    "WhisperSTT",
    "text_to_speech",
    "transcribe_audio",
    # Domain - Document Loaders
    "Document",
    "BaseDocumentLoader",
    "TextLoader",
    "PDFLoader",
    "CSVLoader",
    "DirectoryLoader",
    "DocumentLoader",
    "load_documents",
    # Domain - Embeddings
    "EmbeddingResult",
    "BaseEmbedding",
    "OpenAIEmbedding",
    "GeminiEmbedding",
    "OllamaEmbedding",
    "VoyageEmbedding",
    "JinaEmbedding",
    "MistralEmbedding",
    "CohereEmbedding",
    "Embedding",
    "EmbeddingCache",
    "embed",
    "embed_sync",
    "cosine_similarity",
    "euclidean_distance",
    "normalize_vector",
    "batch_cosine_similarity",
    "find_hard_negatives",
    "mmr_search",
    "query_expansion",
    # Domain - Text Splitters
    "BaseTextSplitter",
    "CharacterTextSplitter",
    "RecursiveCharacterTextSplitter",
    "TokenTextSplitter",
    "MarkdownHeaderTextSplitter",
    "TextSplitter",
    "split_documents",
    # Domain - Output Parsers
    "OutputParserException",
    "BaseOutputParser",
    "PydanticOutputParser",
    "JSONOutputParser",
    "CommaSeparatedListOutputParser",
    "NumberedListOutputParser",
    "DatetimeOutputParser",
    "EnumOutputParser",
    "BooleanOutputParser",
    "RetryOutputParser",
    "parse_json",
    "parse_list",
    "parse_bool",
    # Domain - Prompts
    "TemplateFormat",
    "PromptExample",
    "ChatMessage",
    "BasePromptTemplate",
    "PromptTemplate",
    "ChatPromptTemplate",
    "FewShotPromptTemplate",
    "SystemMessageTemplate",
    "PromptComposer",
    "PromptOptimizer",
    "PromptCache",
    "PromptVersioning",
    "ExampleSelector",
    "PredefinedTemplates",
    "create_prompt_template",
    "create_chat_template",
    "create_few_shot_template",
    "get_cached_prompt",
    "get_cache_stats",
    "clear_cache",
    # Domain - Memory
    "BaseMemory",
    "Message",
    "BufferMemory",
    "WindowMemory",
    "TokenMemory",
    "SummaryMemory",
    "ConversationMemory",
    "create_memory",
    # Domain - Tools
    "Tool",
    "ToolParameter",
    "ToolRegistry",
    "register_tool",
    "get_tool",
    "get_all_tools",
    "echo",
    "calculator",
    "get_current_time",
    # "search_web",  # domain.tools에서 이미 export됨
    # Domain - Advanced Tools
    "SchemaGenerator",
    "ToolValidator",
    "APIProtocol",
    "APIConfig",
    "ExternalAPITool",
    "ToolChain",
    "tool",
    "AdvancedToolRegistry",
    "default_registry",
    # Domain - Graph
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
    # Domain - Multi-Agent
    "MessageType",
    "AgentMessage",
    "CommunicationBus",
    "CoordinationStrategy",
    "SequentialStrategy",
    "ParallelStrategy",
    "HierarchicalStrategy",
    "DebateStrategy",
    # Domain - State Graph
    "GraphConfig",
    "NodeExecution",
    "GraphExecution",
    "Checkpoint",
    "END",
    # Domain - Vector Stores
    "BaseVectorStore",
    "VectorSearchResult",
    "ChromaVectorStore",
    "PineconeVectorStore",
    "FAISSVectorStore",
    "QdrantVectorStore",
    "WeaviateVectorStore",
    "VectorStore",
    "VectorStoreBuilder",
    "create_vector_store",
    "from_documents",
    # Domain - Vision
    "CLIPEmbedding",
    "MultimodalEmbedding",
    "create_vision_embedding",
    "ImageDocument",
    "ImageLoader",
    "PDFWithImagesLoader",
    "load_images",
    "load_pdf_with_images",
    # Domain - Web Search
    "SearchResult",
    "SearchResponse",
    "SearchEngine",
    "BaseSearchEngine",
    "GoogleSearch",
    "BingSearch",
    "DuckDuckGoSearch",
    "WebScraper",
    # Domain - Evaluation
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
    # Domain - Fine-tuning
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
    # Domain - Audio
    "AudioSegment",
    "TranscriptionSegment",
    "TranscriptionResult",
    "WhisperModel",
    "TTSProvider",
    # Utils - Config
    "Config",
    "EnvConfig",
    # Utils - Error Handling
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
    # Utils - Streaming
    "StreamStats",
    "StreamResponse",
    "StreamBuffer",
    "stream_response",
    "stream_print",
    "stream_collect",
    "pretty_stream",
    # Utils - Token Counter
    "ModelPricing",
    "ModelContextWindow",
    "TokenCounter",
    "CostEstimate",
    "CostEstimator",
    "count_tokens",
    "count_message_tokens",
    "estimate_cost",
    "get_cheapest_model",
    "get_context_window",
    # Utils - Tracer
    "Trace",
    "TraceSpan",
    "Tracer",
    "get_tracer",
    "enable_tracing",
    # Utils - Callbacks
    "CallbackEvent",
    "BaseCallback",
    "LoggingCallback",
    "CostTrackingCallback",
    "TimingCallback",
    "StreamingCallback",
    "FunctionCallback",
    "CallbackManager",
    "create_callback_manager",
    # Utils - RAG Debug
    "EmbeddingInfo",
    "SimilarityInfo",
    "RAGDebugger",
    "inspect_embedding",
    "compare_texts",
    "validate_pipeline",
    "visualize_embeddings_2d",
    "similarity_heatmap",
    # Utils - Others
    "get_logger",
    "main",
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

    if missing and not hasattr(sys, "_beanllm_install_warned"):
        sys._beanllm_install_warned = True

        if use_ui:
            # 디자인 시스템 사용
            install_commands = []
            for pkg in missing:
                if pkg == "gemini":
                    install_commands.append("pip install beanllm[gemini]")
                elif pkg == "ollama":
                    install_commands.append("pip install beanllm[ollama]")

            InfoPattern.render(
                "Some provider SDKs are not installed",
                details=[f"Install: {cmd}" for cmd in install_commands]
                + ["Or install all: pip install beanllm[all]"],
            )
        else:
            # 기본 출력 (UI 없을 때)
            print("\n" + "=" * 60)
            print("📦 beanllm - Optional Provider SDKs")
            print("=" * 60)
            print("\nℹ️  Some provider SDKs are not installed:")
            for pkg in missing:
                if pkg == "gemini":
                    print("  • Gemini: pip install beanllm[gemini]")
                elif pkg == "ollama":
                    print("  • Ollama: pip install beanllm[ollama]")
            print("\nOr install all providers:")
            print("  pip install beanllm[all]")
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
        "Welcome to beanllm!",
        steps=[
            {
                "title": "Set environment variables",
                "description": "export OPENAI_API_KEY='your-key'",
            },
            {
                "title": "Try it out",
                "description": "from beanllm import get_registry; r = get_registry()",
            },
            {"title": "Use CLI", "description": "beanllm list"},
        ],
    )


# 패키지 import 시 확인 (경고만, 에러 아님)
try:
    _check_optional_dependencies()
    _print_welcome_banner()
except Exception:
    pass  # 에러 발생해도 import는 성공
