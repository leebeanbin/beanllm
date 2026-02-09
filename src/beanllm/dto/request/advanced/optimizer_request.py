"""
Optimizer Request DTOs - 최적화 요청 데이터 전송 객체
책임: 요청 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True, kw_only=True)
class BenchmarkRequest:
    """
    벤치마크 요청 DTO

    책임:
    - 데이터 구조 정의만
    - 검증 없음 (Handler에서 처리)
    """

    system_id: str
    system_type: str = "rag"
    num_queries: int = 100
    synthetic: bool = True
    test_queries: List[str] = field(default_factory=list)
    queries: Optional[List[str]] = None  # Alias for test_queries
    query_types: Optional[List[str]] = None
    domain: Optional[str] = None
    metrics: List[str] = field(default_factory=lambda: ["latency", "quality", "cost"])

    def __post_init__(self) -> None:
        if self.queries is None:
            self.queries = self.test_queries


@dataclass(slots=True, kw_only=True)
class OptimizeRequest:
    """
    파라미터 최적화 요청 DTO
    """

    system_id: str
    parameter_space: Dict[str, Any]
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    optimization_method: str = "bayesian"
    method: Optional[str] = None  # Alias for optimization_method
    max_trials: int = 30
    n_trials: Optional[int] = None  # Alias for max_trials
    objective: str = "quality"
    objectives: Optional[List[str]] = None
    multi_objectives: Optional[List[str]] = None
    multi_objective: bool = False

    def __post_init__(self) -> None:
        if self.method is None:
            self.method = self.optimization_method
        if self.n_trials is None:
            self.n_trials = self.max_trials
        if self.objectives is None and self.multi_objectives is not None:
            self.objectives = self.multi_objectives
        if self.multi_objectives is None and self.objective == "multi":
            self.multi_objectives = ["quality", "latency"]
            self.objectives = self.multi_objectives


@dataclass(slots=True, kw_only=True)
class ProfileRequest:
    """
    프로파일링 요청 DTO
    """

    system_id: str
    duration: int = 60
    sample_queries: List[str] = field(default_factory=list)
    profile_components: bool = True
    components: List[str] = field(default_factory=list)


@dataclass(slots=True, kw_only=True)
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
    metrics: List[str] = field(default_factory=lambda: ["quality", "latency"])
    statistical_test: str = "ttest"
