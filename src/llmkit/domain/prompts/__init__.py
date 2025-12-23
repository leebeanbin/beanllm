"""
Prompts Domain - 프롬프트 템플릿 도메인
"""

from .base import BasePromptTemplate
from .cache import PromptCache, clear_cache, get_cache_stats, get_cached_prompt
from .composer import PromptComposer
from .enums import TemplateFormat
from .factory import create_chat_template, create_few_shot_template, create_prompt_template
from .optimizer import PromptOptimizer
from .predefined import PredefinedTemplates
from .selectors import ExampleSelector
from .templates import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    PromptTemplate,
    SystemMessageTemplate,
)
from .types import ChatMessage, PromptExample
from .versioning import PromptVersion, PromptVersioning, PromptVersionManager

# A/B Testing (optional dependency)
try:
    from .ab_testing import ABTestConfig, ABTestResult, ABTestRunner

    AB_TESTING_AVAILABLE = True
except ImportError:
    AB_TESTING_AVAILABLE = False
    ABTestConfig = None
    ABTestResult = None
    ABTestRunner = None

# Performance Tracking
try:
    from .performance import PerformanceRecord, PromptPerformanceTracker

    PERFORMANCE_TRACKING_AVAILABLE = True
except ImportError:
    PERFORMANCE_TRACKING_AVAILABLE = False
    PerformanceRecord = None
    PromptPerformanceTracker = None

__all__ = [
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
    "PromptVersion",
    "PromptVersionManager",
    "ExampleSelector",
    "PredefinedTemplates",
    "create_prompt_template",
    "create_chat_template",
    "create_few_shot_template",
    "get_cached_prompt",
    "get_cache_stats",
    "clear_cache",
]

# A/B Testing exports (if available)
if AB_TESTING_AVAILABLE:
    __all__.extend(["ABTestConfig", "ABTestResult", "ABTestRunner"])

# Performance Tracking exports (if available)
if PERFORMANCE_TRACKING_AVAILABLE:
    __all__.extend(["PerformanceRecord", "PromptPerformanceTracker"])
