"""
Evaluation Domain - 평가 메트릭 도메인
"""

from .base_metric import BaseMetric
from .checklist import Checklist, ChecklistGrader, ChecklistItem

# Continuous Evaluation은 선택적 의존성 (apscheduler 필요)
try:
    from .continuous import ContinuousEvaluator, EvaluationRun, EvaluationTask
except ImportError:
    ContinuousEvaluator = None  # type: ignore
    EvaluationRun = None  # type: ignore
    EvaluationTask = None  # type: ignore

from .drift_detection import DriftAlert, DriftDetector
from .enums import MetricType
from .evaluator import Evaluator
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
]
