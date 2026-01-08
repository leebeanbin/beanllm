"""
Optimizer Request DTOs - 최적화 요청 데이터 전송 객체
책임: 요청 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    metrics: Optional[List[str]] = None  # ["latency", "quality", "cost"]

    def __post_init__(self):
        if self.test_queries is None:
            self.test_queries = []
        if self.metrics is None:
            self.metrics = ["latency", "quality", "cost"]


@dataclass
class OptimizeRequest:
    """
    파라미터 최적화 요청 DTO
    """

    system_id: str
    optimization_method: str = "bayesian"  # "bayesian", "grid", "genetic"
    parameter_space: Dict[str, Any]  # {"top_k": [5, 10, 20], "chunk_size": [200, 500, 1000]}
    max_trials: int = 30
    objective: str = "quality"  # "quality", "latency", "cost", or "multi"
    multi_objectives: Optional[List[str]] = None  # For multi-objective optimization

    def __post_init__(self):
        if self.multi_objectives is None and self.objective == "multi":
            self.multi_objectives = ["quality", "latency"]


@dataclass
class ProfileRequest:
    """
    프로파일링 요청 DTO
    """

    system_id: str
    duration: int = 60  # seconds
    sample_queries: Optional[List[str]] = None
    profile_components: bool = True  # Component-level profiling

    def __post_init__(self):
        if self.sample_queries is None:
            self.sample_queries = []


@dataclass
class ABTestRequest:
    """
    A/B 테스트 요청 DTO
    """

    config_a_id: str
    config_b_id: str
    test_queries: List[str]
    metrics: Optional[List[str]] = None
    statistical_test: str = "ttest"  # "ttest", "mannwhitney", "wilcoxon"

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["quality", "latency"]
