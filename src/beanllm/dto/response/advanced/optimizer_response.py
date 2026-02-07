"""
Optimizer Response DTOs - 최적화 응답 데이터 전송 객체
책임: 응답 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass
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
    num_queries: int
    # Optional fields for backward compatibility
    system_id: Optional[str] = None
    system_type: Optional[str] = None
    queries: Optional[List[str]] = None
    baseline_metrics: Optional[Dict[str, float]] = None
    detailed_results: Optional[List[Dict[str, Any]]] = None
    bottlenecks: Optional[List[str]] = None
    timestamp: Optional[str] = None
    # Latency metrics
    avg_latency: float = 0.0
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    # Score metrics
    avg_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    # Throughput metrics
    throughput: float = 0.0
    total_duration: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.baseline_metrics is None:
            self.baseline_metrics = {}
        if self.detailed_results is None:
            self.detailed_results = []
        if self.bottlenecks is None:
            self.bottlenecks = []
        if self.queries is None:
            self.queries = []


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
    best_params: Optional[Dict[str, Any]] = None  # Alias for optimal_parameters
    n_trials: int = 0  # Alias for num_trials
    convergence_data: Optional[List[Dict[str, Any]]] = None  # Detailed convergence data
    recommendations: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []
        if self.metadata is None:
            self.metadata = {}
        if self.best_params is None:
            self.best_params = self.optimal_parameters
        if self.n_trials == 0:
            self.n_trials = self.num_trials


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
    # Additional fields for service compatibility
    total_duration_ms: float = 0.0
    total_tokens: int = 0
    components: Optional[List[Dict[str, Any]]] = None
    bottleneck: Optional[str] = None  # Single bottleneck string (alias)
    breakdown: Optional[Dict[str, float]] = None  # Alias for cost_breakdown
    recommendations: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []
        if self.metadata is None:
            self.metadata = {}
        if self.components is None:
            self.components = []
        if self.breakdown is None:
            self.breakdown = self.cost_breakdown
        if self.total_duration_ms == 0.0:
            self.total_duration_ms = self.total_latency * 1000


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
    variant_a_name: str = "variant_a"
    variant_b_name: str = "variant_b"
    effect_size: Optional[Dict[str, float]] = None
    recommendations: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    # Convenience properties computed from existing fields
    lift: float = 0.0
    p_value: float = 1.0
    is_significant: bool = False
    confidence_level: float = 0.95
    variant_a_mean: float = 0.0
    variant_b_mean: float = 0.0

    def __post_init__(self) -> None:
        if self.recommendations is None:
            self.recommendations = []
        if self.metadata is None:
            self.metadata = {}
        # Compute convenience properties from existing fields
        self._compute_convenience_properties()

    def _compute_convenience_properties(self) -> None:
        """Compute convenience properties from existing DTO fields"""
        # Extract from statistical_significance
        if self.statistical_significance:
            self.p_value = float(self.statistical_significance.get("p_value", 1.0))
            self.is_significant = bool(self.statistical_significance.get("significant", False))
        # Extract means from results
        if self.results_a:
            self.variant_a_mean = float(self.results_a.get("quality", 0.0))
        if self.results_b:
            self.variant_b_mean = float(self.results_b.get("quality", 0.0))
        # Compute lift
        if self.variant_a_mean > 0:
            self.lift = ((self.variant_b_mean - self.variant_a_mean) / self.variant_a_mean) * 100


@dataclass
class RecommendationResponse:
    """
    최적화 권장사항 응답 DTO
    """

    profile_id: str
    recommendations: List[
        Dict[str, Any]
    ]  # [{"type": "reduce_chunk_size", "priority": "high", ...}]
    estimated_improvements: Dict[str, float]
    implementation_difficulty: Dict[str, str]  # {"reduce_chunk_size": "easy"}
    priority_order: List[str]
    metadata: Optional[Dict[str, Any]] = None
    # Computed summary of recommendation counts by priority
    summary: Optional[Dict[str, int]] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}
        if self.summary is None:
            self._compute_summary()

    def _compute_summary(self) -> None:
        """Compute summary counts by priority level"""
        self.summary = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for rec in self.recommendations:
            priority = rec.get("priority", "low").lower()
            if priority in self.summary:
                self.summary[priority] += 1
