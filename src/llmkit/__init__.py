"""
llmkit - Unified toolkit for managing and using multiple LLM providers
환경변수 기반 LLM 모델 활성화 및 관리 패키지
"""

from .registry import ModelRegistry, get_model_registry as get_registry
from .provider_factory import ProviderFactory
from .model_info import ModelCapabilityInfo, ProviderInfo
from .hybrid_manager import HybridModelManager, create_hybrid_manager
from .inferrer import MetadataInferrer
from .scanner import ModelScanner
from .client import Client, create_client, ChatResponse
from .adapter import ParameterAdapter, adapt_parameters
from .streaming import (
    stream_response,
    stream_print,
    stream_collect,
    pretty_stream,
    StreamResponse,
    StreamStats,
    StreamBuffer
)
from .tracer import (
    Tracer,
    get_tracer,
    enable_tracing,
    Trace,
    TraceSpan
)
from .tools import (
    Tool,
    ToolParameter,
    ToolRegistry,
    register_tool,
    get_tool,
    get_all_tools
)
from .agent import (
    Agent,
    AgentStep,
    AgentResult,
    create_agent
)
from .memory import (
    BaseMemory,
    BufferMemory,
    WindowMemory,
    TokenMemory,
    SummaryMemory,
    ConversationMemory,
    Message,
    create_memory
)
from .chain import (
    Chain,
    PromptChain,
    SequentialChain,
    ParallelChain,
    ChainBuilder,
    ChainResult,
    create_chain
)
from .output_parsers import (
    BaseOutputParser,
    PydanticOutputParser,
    JSONOutputParser,
    CommaSeparatedListOutputParser,
    NumberedListOutputParser,
    DatetimeOutputParser,
    EnumOutputParser,
    BooleanOutputParser,
    RetryOutputParser,
    OutputParserException,
    parse_json,
    parse_list,
    parse_bool,
)
from .document_loaders import (
    Document,
    BaseDocumentLoader,
    TextLoader,
    PDFLoader,
    CSVLoader,
    DirectoryLoader,
    DocumentLoader,
    load_documents
)
from .text_splitters import (
    BaseTextSplitter,
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    TextSplitter,
    split_documents
)
from .embeddings import (
    BaseEmbedding,
    OpenAIEmbedding,
    GeminiEmbedding,
    OllamaEmbedding,
    VoyageEmbedding,
    JinaEmbedding,
    MistralEmbedding,
    CohereEmbedding,
    Embedding,
    EmbeddingResult,
    embed,
    embed_sync,
    # Advanced features
    find_hard_negatives,
    mmr_search,
    query_expansion,
    EmbeddingCache
)
from .vector_stores import (
    BaseVectorStore,
    ChromaVectorStore,
    PineconeVectorStore,
    FAISSVectorStore,
    QdrantVectorStore,
    WeaviateVectorStore,
    VectorStore,
    VectorStoreBuilder,
    VectorSearchResult,
    create_vector_store,
    from_documents
)
from .rag_debug import (
    RAGDebugger,
    EmbeddingInfo,
    SimilarityInfo,
    inspect_embedding,
    compare_texts,
    validate_pipeline,
    visualize_embeddings_2d,
    similarity_heatmap
)
from .rag_chain import (
    RAGChain,
    RAGBuilder,
    RAGResponse,
    create_rag,
    RAG
)
from .state_graph import (
    StateGraph,
    END,
    Checkpoint,
    GraphConfig,
    GraphExecution,
    NodeExecution,
    create_state_graph
)
from .callbacks import (
    BaseCallback,
    LoggingCallback,
    CostTrackingCallback,
    TimingCallback,
    StreamingCallback,
    FunctionCallback,
    CallbackManager,
    CallbackEvent,
    create_callback_manager
)
from .vision_loaders import (
    ImageDocument,
    ImageLoader,
    PDFWithImagesLoader,
    load_images,
    load_pdf_with_images
)
from .vision_embeddings import (
    CLIPEmbedding,
    MultimodalEmbedding,
    create_vision_embedding
)
from .vision_rag import (
    VisionRAG,
    MultimodalRAG,
    create_vision_rag
)
from .ml_models import (
    BaseMLModel,
    TensorFlowModel,
    PyTorchModel,
    SklearnModel,
    MLModelFactory,
    load_ml_model
)
from .graph import (
    GraphState,
    NodeCache,
    BaseNode,
    FunctionNode,
    AgentNode,
    LLMNode,
    GraderNode,
    ConditionalNode,
    LoopNode,
    ParallelNode,
    Graph,
    create_simple_graph
)
from .multi_agent import (
    MessageType,
    AgentMessage,
    CommunicationBus,
    CoordinationStrategy,
    SequentialStrategy,
    ParallelStrategy,
    HierarchicalStrategy,
    DebateStrategy,
    MultiAgentCoordinator,
    create_coordinator,
    quick_debate
)
from .tools_advanced import (
    SchemaGenerator,
    ToolValidator,
    APIProtocol,
    APIConfig,
    ExternalAPITool,
    ToolChain,
    tool,
    ToolRegistry as AdvancedToolRegistry,
    default_registry
)
from .web_search import (
    SearchResult,
    SearchResponse,
    SearchEngine,
    BaseSearchEngine,
    GoogleSearch,
    BingSearch,
    DuckDuckGoSearch,
    WebScraper,
    WebSearch,
    search_web
)
from .audio_speech import (
    AudioSegment,
    TranscriptionSegment,
    TranscriptionResult,
    WhisperModel,
    WhisperSTT,
    TTSProvider,
    TextToSpeech,
    AudioRAG,
    transcribe_audio,
    text_to_speech
)
from .token_counter import (
    TokenCounter,
    CostEstimator,
    CostEstimate,
    ModelPricing,
    ModelContextWindow,
    count_tokens,
    count_message_tokens,
    estimate_cost,
    get_cheapest_model,
    get_context_window
)
from .prompts import (
    TemplateFormat,
    PromptExample,
    BasePromptTemplate,
    PromptTemplate,
    FewShotPromptTemplate,
    ChatMessage,
    ChatPromptTemplate,
    SystemMessageTemplate,
    PromptComposer,
    PromptOptimizer,
    PredefinedTemplates,
    ExampleSelector,
    PromptVersioning,
    PromptCache,
    create_prompt_template,
    create_chat_template,
    create_few_shot_template,
    get_cached_prompt,
    get_cache_stats,
    clear_cache
)
from .evaluation import (
    MetricType,
    EvaluationResult,
    BatchEvaluationResult,
    BaseMetric,
    ExactMatchMetric,
    F1ScoreMetric,
    BLEUMetric,
    ROUGEMetric,
    SemanticSimilarityMetric,
    LLMJudgeMetric,
    AnswerRelevanceMetric,
    ContextPrecisionMetric,
    FaithfulnessMetric,
    CustomMetric,
    Evaluator,
    evaluate_text,
    evaluate_rag,
    create_evaluator
)
from .finetuning import (
    FineTuningStatus,
    ModelProvider,
    TrainingExample,
    FineTuningConfig,
    FineTuningJob,
    FineTuningMetrics,
    BaseFineTuningProvider,
    OpenAIFineTuningProvider,
    DatasetBuilder,
    DataValidator,
    FineTuningManager,
    FineTuningCostEstimator,
    create_finetuning_provider,
    quick_finetune
)
from .error_handling import (
    LLMKitError,
    ProviderError,
    RateLimitError,
    TimeoutError,
    ValidationError,
    CircuitBreakerError,
    MaxRetriesExceededError,
    RetryStrategy,
    RetryConfig,
    RetryHandler,
    retry,
    CircuitState,
    CircuitBreakerConfig,
    CircuitBreaker,
    circuit_breaker,
    RateLimitConfig,
    RateLimiter,
    rate_limit,
    FallbackHandler,
    fallback,
    ErrorRecord,
    ErrorTracker,
    get_error_tracker,
    ErrorHandlerConfig,
    ErrorHandler,
    with_error_handling,
    timeout
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
        from .ui import InfoPattern, CommandBlock, get_console
        use_ui = True
    except ImportError:
        use_ui = False

    missing = []

    try:
        import google.generativeai
    except ImportError:
        missing.append("gemini")

    try:
        import ollama
    except ImportError:
        missing.append("ollama")

    if missing and not hasattr(sys, '_llmkit_install_warned'):
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
                details=[
                    f"Install: {cmd}" for cmd in install_commands
                ] + ["Or install all: pip install llmkit[all]"]
            )
        else:
            # 기본 출력 (UI 없을 때)
            print("\n" + "="*60)
            print("📦 llmkit - Optional Provider SDKs")
            print("="*60)
            print("\nℹ️  Some provider SDKs are not installed:")
            for pkg in missing:
                if pkg == "gemini":
                    print(f"  • Gemini: pip install llmkit[gemini]")
                elif pkg == "ollama":
                    print(f"  • Ollama: pip install llmkit[ollama]")
            print("\nOr install all providers:")
            print("  pip install llmkit[all]")
            print("\n" + "="*60 + "\n")


def _print_welcome_banner():
    """환영 배너 출력 (선택적, 환경변수로 제어) - 디자인 시스템 적용"""
    import os
    import sys
    
    # 환경변수로 제어 (기본값: False)
    if not os.getenv("LLMKIT_SHOW_BANNER", "false").lower() == "true":
        return
    
    try:
        from .ui import print_logo, OnboardingPattern
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
                "description": "export OPENAI_API_KEY='your-key'"
            },
            {
                "title": "Try it out",
                "description": "from llmkit import get_registry; r = get_registry()"
            },
            {
                "title": "Use CLI",
                "description": "llmkit list"
            }
        ]
    )

# 패키지 import 시 확인 (경고만, 에러 아님)
try:
    _check_optional_dependencies()
    _print_welcome_banner()
except Exception:
    pass  # 에러 발생해도 import는 성공
