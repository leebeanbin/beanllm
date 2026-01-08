"""
Optimizer Response DTOs - 최적화 응답 데이터 전송 객체
책임: 응답 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkResponse:
    """
    벤치마크 응답 DTO

    책임:
    - 응답 데이터 구조 정의만
    - 변환 로직 없음 (Service에서 처리)
    """

    benchmark_id: str
    system_id: str
    system_type: str
    num_queries: int
    baseline_metrics: Dict[str, float]  # {"latency": 1.5, "quality": 0.85, "cost": 0.001}
    detailed_results: List[Dict[str, Any]]
    bottlenecks: List[str]
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class OptimizeResponse:
    """
    파라미터 최적화 응답 DTO
    """

    optimization_id: str
    system_id: str
    optimal_parameters: Dict[str, Any]
    improvement_metrics: Dict[str, float]  # {"latency": -20%, "quality": +5%}
    num_trials: int
    convergence_curve: Optional[List[float]] = None
    best_score: float = 0.0
    baseline_score: float = 0.0
    recommendations: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ProfileResponse:
    """
    프로파일링 응답 DTO
    """

    profile_id: str
    system_id: str
    duration: float
    component_breakdown: Dict[str, Dict[str, float]]  # {"embedding": {"time": 0.5, "cost": 0.0001}}
    total_latency: float
    total_cost: float
    bottlenecks: List[Dict[str, Any]]
    cost_breakdown: Dict[str, float]
    recommendations: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ABTestResponse:
    """
    A/B 테스트 응답 DTO
    """

    test_id: str
    config_a_id: str
    config_b_id: str
    num_queries: int
    results_a: Dict[str, float]  # {"quality": 0.85, "latency": 1.5}
    results_b: Dict[str, float]
    statistical_significance: Dict[str, Any]  # {"p_value": 0.03, "significant": True}
    winner: Optional[str] = None  # "config_a", "config_b", or None (no significant diff)
    effect_size: Optional[Dict[str, float]] = None
    recommendations: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RecommendationResponse:
    """
    최적화 권장사항 응답 DTO
    """

    profile_id: str
    recommendations: List[Dict[str, Any]]  # [{"type": "reduce_chunk_size", "priority": "high", ...}]
    estimated_improvements: Dict[str, float]
    implementation_difficulty: Dict[str, str]  # {"reduce_chunk_size": "easy"}
    priority_order: List[str]
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
