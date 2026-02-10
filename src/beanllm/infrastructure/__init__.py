"""
Infrastructure Layer - 외부 시스템과의 인터페이스, 어댑터, 레지스트리 등
"""

import importlib

_LAZY_IMPORT_MAP: dict[str, tuple[str, str]] = {
    # Adapter
    "AdaptedParameters": ("beanllm.infrastructure.adapter", "AdaptedParameters"),
    "ParameterAdapter": ("beanllm.infrastructure.adapter", "ParameterAdapter"),
    "adapt_parameters": ("beanllm.infrastructure.adapter", "adapt_parameters"),
    "validate_parameters": ("beanllm.infrastructure.adapter", "validate_parameters"),
    # Hybrid Manager
    "HybridModelInfo": ("beanllm.infrastructure.hybrid", "HybridModelInfo"),
    "HybridModelManager": ("beanllm.infrastructure.hybrid", "HybridModelManager"),
    "create_hybrid_manager": ("beanllm.infrastructure.hybrid", "create_hybrid_manager"),
    # Inferrer
    "MetadataInferrer": ("beanllm.infrastructure.inferrer", "MetadataInferrer"),
    # Models
    "MODELS": ("beanllm.infrastructure.models", "MODELS"),
    "LLMProvider": ("beanllm.infrastructure.models", "LLMProvider"),
    "ModelCapabilityInfo": ("beanllm.infrastructure.models", "ModelCapabilityInfo"),
    "ModelConfig": ("beanllm.infrastructure.models", "ModelConfig"),
    "ModelConfigManager": ("beanllm.infrastructure.models", "ModelConfigManager"),
    "ModelStatus": ("beanllm.infrastructure.models", "ModelStatus"),
    "ParameterInfo": ("beanllm.infrastructure.models", "ParameterInfo"),
    "ProviderInfo": ("beanllm.infrastructure.models", "ProviderInfo"),
    "get_all_models": ("beanllm.infrastructure.models", "get_all_models"),
    "get_default_model": ("beanllm.infrastructure.models", "get_default_model"),
    "get_models_by_provider": ("beanllm.infrastructure.models", "get_models_by_provider"),
    "get_models_by_type": ("beanllm.infrastructure.models", "get_models_by_type"),
    # Registry
    "ModelRegistry": ("beanllm.infrastructure.registry", "ModelRegistry"),
    "get_model_registry": ("beanllm.infrastructure.registry", "get_model_registry"),
    # Scanner
    "ModelScanner": ("beanllm.infrastructure.scanner", "ModelScanner"),
    "ScannedModel": ("beanllm.infrastructure.scanner", "ScannedModel"),
}

_OPTIONAL_LAZY_IMPORT_MAP: dict[str, tuple[str, str]] = {
    # Security (선택적)
    "SecureConfig": ("beanllm.infrastructure.security", "SecureConfig"),
    # Integrations (선택적 - 외부 프레임워크 통합)
    "LangGraphBridge": ("beanllm.infrastructure.integrations", "LangGraphBridge"),
    "LangGraphWorkflow": ("beanllm.infrastructure.integrations", "LangGraphWorkflow"),
    "LlamaIndexBridge": ("beanllm.infrastructure.integrations", "LlamaIndexBridge"),
    "LlamaIndexQueryEngine": ("beanllm.infrastructure.integrations", "LlamaIndexQueryEngine"),
    "WorkflowBuilder": ("beanllm.infrastructure.integrations", "WorkflowBuilder"),
    "create_llamaindex_query_engine": (
        "beanllm.infrastructure.integrations",
        "create_llamaindex_query_engine",
    ),
    "create_workflow": ("beanllm.infrastructure.integrations", "create_workflow"),
}

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
    "SECURITY_AVAILABLE",
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
    if name == "SECURITY_AVAILABLE":
        try:
            importlib.import_module("beanllm.infrastructure.security")
            val = True
        except ImportError:
            val = False
        globals()[name] = val
        return val
    if name == "INTEGRATIONS_AVAILABLE":
        try:
            importlib.import_module("beanllm.infrastructure.integrations")
            val = True
        except ImportError:
            val = False
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)
