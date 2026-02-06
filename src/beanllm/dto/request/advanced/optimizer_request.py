"""
Optimizer Request DTOs - 최적화 요청 데이터 전송 객체
책임: 요청 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkRequest:
    """
    벤치마크 요청 DTO

    책임:
    - 데이터 구조 정의만
    - 검증 없음 (Handler에서 처리)
    """

    system_id: str  # RAG system or Agent system ID
    system_type: str = "rag"  # "rag" or "agent"
    num_queries: int = 100
    synthetic: bool = True  # Generate synthetic queries
    test_queries: Optional[List[str]] = None
    queries: Optional[List[str]] = None  # Alias for test_queries
    query_types: Optional[List[str]] = None  # ["simple", "complex", "multi_hop"]
    domain: Optional[str] = None  # Domain for synthetic query generation
    metrics: Optional[List[str]] = None  # ["latency", "quality", "cost"]

    def __post_init__(self):
        if self.test_queries is None:
            self.test_queries = []
        if self.queries is None:
            self.queries = self.test_queries
        if self.metrics is None:
            self.metrics = ["latency", "quality", "cost"]


@dataclass
class OptimizeRequest:
    """
    파라미터 최적화 요청 DTO
    """

    system_id: str
    parameter_space: Dict[str, Any]  # {"top_k": [5, 10, 20], "chunk_size": [200, 500, 1000]}
    parameters: Optional[List[Dict[str, Any]]] = None  # Alternative format for parameter_space
    optimization_method: str = "bayesian"  # "bayesian", "grid", "genetic"
    method: Optional[str] = None  # Alias for optimization_method
    max_trials: int = 30
    n_trials: Optional[int] = None  # Alias for max_trials
    objective: str = "quality"  # "quality", "latency", "cost", or "multi"
    objectives: Optional[List[str]] = None  # Alias for multi_objectives
    multi_objectives: Optional[List[str]] = None  # For multi-objective optimization
    multi_objective: bool = False  # Whether to use multi-objective optimization

    def __post_init__(self):
        if self.method is None:
            self.method = self.optimization_method
        if self.n_trials is None:
            self.n_trials = self.max_trials
        if self.parameters is None:
            self.parameters = []
        if self.objectives is None and self.multi_objectives is not None:
            self.objectives = self.multi_objectives
        if self.multi_objectives is None and self.objective == "multi":
            self.multi_objectives = ["quality", "latency"]
            self.objectives = self.multi_objectives


@dataclass
class ProfileRequest:
    """
    프로파일링 요청 DTO
    """

    system_id: str
    duration: int = 60  # seconds
    sample_queries: Optional[List[str]] = None
    profile_components: bool = True  # Component-level profiling
    components: Optional[List[str]] = None  # Specific components to profile

    def __post_init__(self):
        if self.sample_queries is None:
            self.sample_queries = []
        if self.components is None:
            self.components = []


@dataclass
class ABTestRequest:
    """
    A/B 테스트 요청 DTO
    """

    config_a_id: str
    config_b_id: str
    test_queries: List[str]
    variant_a_name: str = "variant_a"
    variant_b_name: str = "variant_b"
    num_queries: int = 100
    confidence_level: float = 0.95
    metrics: Optional[List[str]] = None
    statistical_test: str = "ttest"  # "ttest", "mannwhitney", "wilcoxon"

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["quality", "latency"]
