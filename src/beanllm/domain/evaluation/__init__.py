"""
Evaluation Domain - 평가 메트릭 도메인
"""

from .base_framework import BaseEvaluationFramework
from .base_metric import BaseMetric
from .checklist import Checklist, ChecklistGrader, ChecklistItem

# Continuous Evaluation은 선택적 의존성 (apscheduler 필요)
try:
    from .continuous import ContinuousEvaluator, EvaluationRun, EvaluationTask
except ImportError:
    ContinuousEvaluator = None  # type: ignore
    EvaluationRun = None  # type: ignore
    EvaluationTask = None  # type: ignore

# 외부 평가 프레임워크 (선택적 의존성)
try:
    from .deepeval_wrapper import DeepEvalWrapper
except ImportError:
    DeepEvalWrapper = None  # type: ignore

try:
    from .lm_eval_harness_wrapper import LMEvalHarnessWrapper
except ImportError:
    LMEvalHarnessWrapper = None  # type: ignore

try:
    from .ragas_wrapper import RAGASWrapper
except ImportError:
    RAGASWrapper = None  # type: ignore

try:
    from .trulens_wrapper import TruLensWrapper
except ImportError:
    TruLensWrapper = None  # type: ignore

from .drift_detection import DriftAlert, DriftDetector
from .enums import MetricType
from .evaluator import Evaluator
from .factory import create_evaluation_framework, list_available_frameworks
from .human_feedback import (
    ComparisonFeedback,
    ComparisonWinner,
    FeedbackType,
    HumanFeedback,
    HumanFeedbackCollector,
)
from .hybrid_evaluator import HybridEvaluator
from .metrics import (
    AnswerRelevanceMetric,
    BLEUMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    CustomMetric,
    ExactMatchMetric,
    F1ScoreMetric,
    FaithfulnessMetric,
    LLMJudgeMetric,
    ROUGEMetric,
    SemanticSimilarityMetric,
)
from .results import BatchEvaluationResult, EvaluationResult
from .rubric import Rubric, RubricCriterion, RubricGrader

__all__ = [
    "MetricType",
    "EvaluationResult",
    "BatchEvaluationResult",
    "BaseMetric",
    "BaseEvaluationFramework",
    "ExactMatchMetric",
    "F1ScoreMetric",
    "BLEUMetric",
    "ROUGEMetric",
    "SemanticSimilarityMetric",
    "LLMJudgeMetric",
    "AnswerRelevanceMetric",
    "ContextPrecisionMetric",
    "ContextRecallMetric",
    "FaithfulnessMetric",
    "CustomMetric",
    "Evaluator",
    # Human Feedback
    "HumanFeedback",
    "ComparisonFeedback",
    "HumanFeedbackCollector",
    "FeedbackType",
    "ComparisonWinner",
    # Hybrid Evaluator
    "HybridEvaluator",
    # Continuous Evaluation
    "ContinuousEvaluator",
    "EvaluationTask",
    "EvaluationRun",
    # Drift Detection
    "DriftDetector",
    "DriftAlert",
    # Rubric-Driven Grading
    "Rubric",
    "RubricCriterion",
    "RubricGrader",
    # CheckEval
    "Checklist",
    "ChecklistItem",
    "ChecklistGrader",
    # Evaluation Analytics
    "EvaluationAnalyticsEngine",
    "EvaluationAnalytics",
    "MetricTrend",
    "CorrelationAnalysis",
    # External Frameworks (2024-2025)
    "DeepEvalWrapper",
    "LMEvalHarnessWrapper",
    "RAGASWrapper",
    "TruLensWrapper",
    "create_evaluation_framework",
    "list_available_frameworks",
]
