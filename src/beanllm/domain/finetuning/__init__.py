"""
Finetuning Domain - 파인튜닝 도메인
"""

from .enums import FineTuningStatus, ModelProvider
from .providers import BaseFineTuningProvider, OpenAIFineTuningProvider
from .types import (
    FineTuningConfig,
    FineTuningJob,
    FineTuningMetrics,
    TrainingExample,
)
from .utils import (
    DatasetBuilder,
    DataValidator,
    FineTuningCostEstimator,
    FineTuningManager,
)

# 로컬 Fine-tuning Providers (선택적 의존성)
try:
    from .local_providers import AxolotlProvider, UnslothProvider
except ImportError:
    AxolotlProvider = None  # type: ignore
    UnslothProvider = None  # type: ignore

__all__ = [
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
    # Local Providers (2024-2025)
    "AxolotlProvider",
    "UnslothProvider",
]
