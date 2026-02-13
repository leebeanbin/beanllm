"""
beanllm - Unified toolkit for managing and using multiple LLM providers
환경변수 기반 LLM 모델 활성화 및 관리 패키지
"""

import importlib

from beanllm._lazy_imports import _LAZY_IMPORT_MAP, _OPTIONAL_LAZY_IMPORT_MAP

__version__ = "0.3.0"

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


def __getattr__(name: str):
    if name in _LAZY_IMPORT_MAP:
        mod_path, attr = _LAZY_IMPORT_MAP[name]
        mod = importlib.import_module(mod_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    if name in _OPTIONAL_LAZY_IMPORT_MAP:
        mod_path, attr = _OPTIONAL_LAZY_IMPORT_MAP[name]
        try:
            mod = importlib.import_module(mod_path)
            val = getattr(mod, attr)
        except ImportError:
            val = None
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)
