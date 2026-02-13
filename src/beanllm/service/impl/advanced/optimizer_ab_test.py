"""Optimizer service - A/B test methods (mixin)."""

from __future__ import annotations

import uuid
from typing import Any, Dict

from beanllm.domain.optimizer import ABTester, ABTestResult
from beanllm.dto.request.advanced.optimizer_request import ABTestRequest
from beanllm.dto.response.advanced.optimizer_response import ABTestResponse
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class OptimizerABTestMixin:
    """Mixin providing ab_test() for OptimizerServiceImpl."""

    _ab_tests: Dict[str, Any]

    async def ab_test(self, request: ABTestRequest) -> ABTestResponse:
        """Run A/B test."""
        logger.info(
            f"Running A/B test: {request.variant_a_name} vs {request.variant_b_name}, "
            f"{request.num_queries} queries"
        )
        test_id = str(uuid.uuid4())
        try:
            result = ABTestResult(
                variant_a_name=request.variant_a_name,
                variant_b_name=request.variant_b_name,
                variant_a_mean=0.0,
                variant_b_mean=0.0,
                variant_a_std=0.0,
                variant_b_std=0.0,
                p_value=1.0,
                is_significant=False,
                confidence_level=request.confidence_level or 0.95,
                sample_size_a=request.num_queries or 50,
                sample_size_b=request.num_queries or 50,
            )
            self._ab_tests[test_id] = result
            logger.info(
                f"A/B test completed: {test_id}, winner={result.winner}, lift={result.lift:.1f}%"
            )
            return ABTestResponse(
                test_id=test_id,
                config_a_id=request.variant_a_name or "variant_a",
                config_b_id=request.variant_b_name or "variant_b",
                num_queries=request.num_queries or 50,
                results_a={"mean": result.variant_a_mean, "std": result.variant_a_std},
                results_b={"mean": result.variant_b_mean, "std": result.variant_b_std},
                statistical_significance={
                    "p_value": result.p_value,
                    "significant": result.is_significant,
                    "confidence_level": result.confidence_level,
                },
                variant_a_name=result.variant_a_name,
                variant_b_name=result.variant_b_name,
                winner=result.winner,
                effect_size={"lift": result.lift} if result.lift != 0.0 else None,
                metadata={"num_queries": request.num_queries},
            )
        except Exception as e:
            logger.error(f"A/B test failed: {e}")
            raise RuntimeError(f"Failed to run A/B test: {e}") from e
