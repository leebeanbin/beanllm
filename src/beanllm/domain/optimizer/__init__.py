"""
Optimizer Domain - 자동 성능 최적화

Phase 4: Auto-Optimizer
- OptimizerEngine: 다양한 최적화 알고리즘 (Bayesian, Grid, Random, Genetic)
- Benchmarker: 합성 쿼리 생성 및 벤치마킹
- Profiler: 컴포넌트별 성능 프로파일링
- ParameterSearch: 다목적 최적화 및 Pareto frontier
- ABTester: A/B 테스팅 프레임워크
- Recommender: 최적화 권장사항 생성
"""

from .ab_tester import ABTester, ABTestResult, compare_multiple_variants
from .benchmarker import (
    Benchmarker,
    BenchmarkQuery,
    BenchmarkResult,
    QueryType,
)
from .optimizer_engine import (
    OptimizationMethod,
    OptimizationResult,
    OptimizerEngine,
    ParameterSpace,
    ParameterType,
)
from .parameter_search import MetadataInferrer
from .profiler import (
    ComponentMetrics,
    ComponentType,
    ProfileContext,
    Profiler,
    ProfileResult,
    profile_rag_pipeline,
)
from .recommender import (
    Priority,
    Recommendation,
    RecommendationCategory,
    Recommender,
    print_recommendations,
)

__all__ = [
    # Optimizer Engine
    "OptimizerEngine",
    "OptimizationMethod",
    "OptimizationResult",
    "ParameterSpace",
    "ParameterType",
    # Benchmarker
    "Benchmarker",
    "BenchmarkQuery",
    "BenchmarkResult",
    "QueryType",
    # Profiler
    "Profiler",
    "ProfileContext",
    "ProfileResult",
    "ComponentMetrics",
    "ComponentType",
    "profile_rag_pipeline",
    # Parameter Search / Metadata
    "MetadataInferrer",
    # A/B Tester
    "ABTester",
    "ABTestResult",
    "compare_multiple_variants",
    # Recommender
    "Recommender",
    "Recommendation",
    "RecommendationCategory",
    "Priority",
    "print_recommendations",
]
