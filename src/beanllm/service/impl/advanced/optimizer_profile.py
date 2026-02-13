"""Optimizer service - Profile and recommendation methods (mixin)."""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from beanllm.domain.optimizer import Priority, ProfileResult, Recommender
from beanllm.dto.request.advanced.optimizer_request import ProfileRequest
from beanllm.dto.response.advanced.optimizer_response import (
    ProfileResponse,
    RecommendationResponse,
)
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class OptimizerProfileMixin:
    """Mixin providing profile() and get_recommendations() for OptimizerServiceImpl."""

    async def profile(self, request: ProfileRequest) -> ProfileResponse:
        """Profile system components."""
        logger.info("Starting profiling: components=%s", request.components or [])
        profile_id = str(uuid.uuid4())
        try:
            result = ProfileResult(components={})
            self._profiles[profile_id] = result
            recommendations = self._recommender.analyze_profile(result)
            logger.info("Profiling completed: %s, %s recommendations", profile_id, len(recommendations))
            component_breakdown = {
                name: {"duration_ms": m.duration_ms, "token_count": float(m.token_count), "estimated_cost": m.estimated_cost}
                for name, m in result.components.items()
            }
            cost_breakdown = {name: m.estimated_cost for name, m in result.components.items()}
            bottleneck_str = result.bottleneck.value if result.bottleneck else None
            recommendation_strs = [f"[{r.priority.value}] {r.title}: {r.action}" for r in recommendations]
            return ProfileResponse(
                profile_id=profile_id,
                system_id=request.system_id or "default",
                duration=result.total_duration_ms / 1000.0,
                component_breakdown=component_breakdown,
                total_latency=result.total_duration_ms / 1000.0,
                total_cost=result.total_cost,
                bottlenecks=[{"component": bottleneck_str}] if bottleneck_str else [],
                cost_breakdown=cost_breakdown,
                total_duration_ms=result.total_duration_ms,
                bottleneck=bottleneck_str,
                recommendations=recommendation_strs,
                metadata={"components_profiled": request.components or []},
            )
        except Exception as e:
            logger.error("Profiling failed: %s", e)
            raise RuntimeError(f"Failed to profile system: {e}") from e

    async def get_recommendations(self, profile_id: str) -> RecommendationResponse:
        """Get optimization recommendations for a profile."""
        logger.info("Getting recommendations for profile: %s", profile_id)
        if profile_id not in self._profiles:
            raise ValueError(f"Profile not found: {profile_id}")
        profile_result = self._profiles[profile_id]
        recommendations = self._recommender.analyze_profile(profile_result)
        priority_order = {Priority.CRITICAL: 0, Priority.HIGH: 1, Priority.MEDIUM: 2, Priority.LOW: 3}
        recommendations = sorted(recommendations, key=lambda r: priority_order[r.priority])
        rec_list = [
            {"category": r.category.value, "priority": r.priority.value, "title": r.title, "description": r.description, "rationale": r.rationale, "action": r.action, "expected_impact": r.expected_impact if isinstance(r.expected_impact, dict) else {}}
            for r in recommendations
        ]
        rec_priority_order = [r.title for r in recommendations]
        implementation_difficulty = {r.title: r.category.value for r in recommendations}
        estimated_improvements = {}
        for r in recommendations:
            estimated_improvements[r.title] = float(r.expected_impact.get("latency_reduction", 0.0)) if isinstance(r.expected_impact, dict) else 0.0
        return RecommendationResponse(
            profile_id=profile_id,
            recommendations=rec_list,
            estimated_improvements=estimated_improvements,
            implementation_difficulty=implementation_difficulty,
            priority_order=rec_priority_order,
            metadata={"summary": {"critical": sum(1 for r in recommendations if r.priority == Priority.CRITICAL), "high": sum(1 for r in recommendations if r.priority == Priority.HIGH), "medium": sum(1 for r in recommendations if r.priority == Priority.MEDIUM), "low": sum(1 for r in recommendations if r.priority == Priority.LOW)}},
        )
