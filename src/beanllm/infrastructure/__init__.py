"""
Infrastructure Layer - 외부 시스템과의 인터페이스, 어댑터, 레지스트리 등
"""

# Adapter
from .adapter import (
    AdaptedParameters,
    ParameterAdapter,
    adapt_parameters,
    validate_parameters,
)

# Hybrid Manager
from .hybrid import (
    HybridModelInfo,
    HybridModelManager,
    create_hybrid_manager,
)

# Inferrer
from .inferrer import MetadataInferrer

# ML Models (미사용 - 필요시 주석 해제)
# from .ml import (
#     BaseMLModel,
#     MLModelFactory,
#     PyTorchModel,
#     SklearnModel,
#     TensorFlowModel,
#     load_ml_model,
# )

# Models
from .models import (
    LLMProvider,
    MODELS,
    ModelCapabilityInfo,
    ModelConfig,
    ModelConfigManager,
    ModelStatus,
    ParameterInfo,
    ProviderInfo,
    get_all_models,
    get_default_model,
    get_models_by_provider,
    get_models_by_type,
)

# Registry
from .registry import ModelRegistry, get_model_registry

# Scanner
from .scanner import ModelScanner, ScannedModel

# Security (선택적)
try:
    from .security import SecureConfig
    SECURITY_AVAILABLE = True
except ImportError:
    SecureConfig = None  # type: ignore
    SECURITY_AVAILABLE = False

# Integrations (선택적 - 외부 프레임워크 통합)
try:
    from .integrations import (
        LangGraphBridge,
        LangGraphWorkflow,
        LlamaIndexBridge,
        LlamaIndexQueryEngine,
        WorkflowBuilder,
        create_llamaindex_query_engine,
        create_workflow,
    )
    INTEGRATIONS_AVAILABLE = True
except ImportError:
    LangGraphBridge = None  # type: ignore
    LangGraphWorkflow = None  # type: ignore
    LlamaIndexBridge = None  # type: ignore
    LlamaIndexQueryEngine = None  # type: ignore
    WorkflowBuilder = None  # type: ignore
    create_llamaindex_query_engine = None  # type: ignore
    create_workflow = None  # type: ignore
    INTEGRATIONS_AVAILABLE = False

__all__ = [
    # Adapter
    "AdaptedParameters",
    "ParameterAdapter",
    "adapt_parameters",
    "validate_parameters",
    # Registry
    "ModelRegistry",
    "get_model_registry",
    # Models (최신)
    "ModelConfig",
    "ModelConfigManager",
    "LLMProvider",
    # Models (backward compatibility)
    "MODELS",
    "ModelStatus",
    "ParameterInfo",
    "ProviderInfo",
    "ModelCapabilityInfo",
    "get_all_models",
    "get_models_by_provider",
    "get_models_by_type",
    "get_default_model",
    # Hybrid Manager
    "HybridModelInfo",
    "HybridModelManager",
    "create_hybrid_manager",
    # Inferrer
    "MetadataInferrer",
    # Scanner
    "ScannedModel",
    "ModelScanner",
    # Security
    "SecureConfig",
    # Integrations (선택적)
    "LangGraphBridge",
    "LangGraphWorkflow",
    "LlamaIndexBridge",
    "LlamaIndexQueryEngine",
    "WorkflowBuilder",
    "create_llamaindex_query_engine",
    "create_workflow",
    "INTEGRATIONS_AVAILABLE",
    # ML Models (미사용 - 필요시 주석 해제)
    # "BaseMLModel",
    # "TensorFlowModel",
    # "PyTorchModel",
    # "SklearnModel",
    # "MLModelFactory",
    # "load_ml_model",
]
