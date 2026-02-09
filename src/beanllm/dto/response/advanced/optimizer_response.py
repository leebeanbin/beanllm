"""
Optimizer Response DTOs - 최적화 응답 데이터 전송 객체
책임: 응답 데이터만 전달
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, model_validator

from beanllm.dto.response.base_response import BaseResponse


class BenchmarkResponse(BaseResponse):
    """
    벤치마크 응답 DTO

    책임:
    - 응답 데이터 구조 정의만
    - 변환 로직 없음 (Service에서 처리)
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_id: str
    num_queries: int
    # Optional fields for backward compatibility
    system_id: Optional[str] = None
    system_type: Optional[str] = None
    queries: List[str] = []
    baseline_metrics: Dict[str, float] = {}
    detailed_results: List[Dict[str, Any]] = []
    bottlenecks: List[str] = []
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
    metadata: Dict[str, Any] = {}


class OptimizeResponse(BaseResponse):
    """
    파라미터 최적화 응답 DTO
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

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
    convergence_data: Optional[List[Dict[str, Any]]] = None
    recommendations: List[str] = []
    metadata: Dict[str, Any] = {}

    @model_validator(mode="before")
    @classmethod
    def compute_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get("best_params") is None:
                data["best_params"] = data.get("optimal_parameters")
            if data.get("n_trials", 0) == 0:
                data["n_trials"] = data.get("num_trials", 0)
        return data


class ProfileResponse(BaseResponse):
    """
    프로파일링 응답 DTO
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str
    system_id: str
    duration: float
    component_breakdown: Dict[str, Dict[str, float]]
    total_latency: float
    total_cost: float
    bottlenecks: List[Dict[str, Any]]
    cost_breakdown: Dict[str, float]
    # Additional fields for service compatibility
    total_duration_ms: float = 0.0
    total_tokens: int = 0
    components: List[Dict[str, Any]] = []
    bottleneck: Optional[str] = None
    breakdown: Optional[Dict[str, float]] = None
    recommendations: List[str] = []
    metadata: Dict[str, Any] = {}

    @model_validator(mode="before")
    @classmethod
    def compute_derived(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get("breakdown") is None:
                data["breakdown"] = data.get("cost_breakdown")
            if data.get("total_duration_ms", 0.0) == 0.0:
                total_latency = data.get("total_latency", 0.0)
                data["total_duration_ms"] = total_latency * 1000
        return data


class ABTestResponse(BaseResponse):
    """
    A/B 테스트 응답 DTO
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    test_id: str
    config_a_id: str
    config_b_id: str
    num_queries: int
    results_a: Dict[str, float]
    results_b: Dict[str, float]
    statistical_significance: Dict[str, Any]
    winner: Optional[str] = None
    variant_a_name: str = "variant_a"
    variant_b_name: str = "variant_b"
    effect_size: Optional[Dict[str, float]] = None
    recommendations: List[str] = []
    metadata: Dict[str, Any] = {}
    # Convenience properties computed from existing fields
    lift: float = 0.0
    p_value: float = 1.0
    is_significant: bool = False
    confidence_level: float = 0.95
    variant_a_mean: float = 0.0
    variant_b_mean: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def compute_convenience_properties(cls, data: Any) -> Any:
        if isinstance(data, dict):
            stat_sig = data.get("statistical_significance", {})
            if stat_sig:
                data.setdefault("p_value", float(stat_sig.get("p_value", 1.0)))
                data.setdefault("is_significant", bool(stat_sig.get("significant", False)))

            results_a = data.get("results_a", {})
            results_b = data.get("results_b", {})
            if results_a:
                data.setdefault("variant_a_mean", float(results_a.get("quality", 0.0)))
            if results_b:
                data.setdefault("variant_b_mean", float(results_b.get("quality", 0.0)))

            va_mean = data.get("variant_a_mean", 0.0)
            vb_mean = data.get("variant_b_mean", 0.0)
            if va_mean > 0:
                data.setdefault("lift", ((vb_mean - va_mean) / va_mean) * 100)
        return data


class RecommendationResponse(BaseResponse):
    """
    최적화 권장사항 응답 DTO
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str
    recommendations: List[Dict[str, Any]]
    estimated_improvements: Dict[str, float]
    implementation_difficulty: Dict[str, str]
    priority_order: List[str]
    metadata: Dict[str, Any] = {}
    summary: Optional[Dict[str, int]] = None

    @model_validator(mode="before")
    @classmethod
    def compute_summary(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("summary") is None:
            recs = data.get("recommendations", [])
            summary: Dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            for rec in recs:
                priority = rec.get("priority", "low").lower()
                if priority in summary:
                    summary[priority] += 1
            data["summary"] = summary
        return data
