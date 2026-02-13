"""
Optimizer service - Optimize methods (mixin).
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, cast

from beanllm.domain.optimizer import (
    OptimizationMethod,
    OptimizationResult,
    ParameterSpace,
    ParameterType,
)
from beanllm.dto.request.advanced.optimizer_request import OptimizeRequest
from beanllm.dto.response.advanced.optimizer_response import OptimizeResponse
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class OptimizerOptimizeMixin:
    """Mixin providing optimize() for OptimizerServiceImpl."""

    _optimizer_engine: Any
    _optimizations: Dict[str, Any]

    async def optimize(self, request: OptimizeRequest) -> OptimizeResponse:
        """
        Optimize parameters using selected algorithm.

        Args:
            request: OptimizeRequest.

        Returns:
            OptimizeResponse: Optimization results.

        Raises:
            RuntimeError: If optimization fails.
        """
        logger.info(
            "Starting optimization: method=%s, n_trials=%s",
            request.method or request.optimization_method,
            request.n_trials or request.max_trials,
        )
        optimization_id = str(uuid.uuid4())
        try:
            param_spaces: List[ParameterSpace] = []
            parameters = request.parameters or []
            for param in parameters:
                ptype_str = (cast(str, param.get("type") or "float")).lower()
                try:
                    param_type = ParameterType[ptype_str.upper()]
                except KeyError:
                    param_type = ParameterType.FLOAT

                if param_type == ParameterType.INTEGER:
                    space = ParameterSpace(
                        name=cast(str, param["name"]),
                        type=param_type,
                        low=cast(int, param.get("low", 0)),
                        high=cast(int, param.get("high", 100)),
                    )
                elif param_type == ParameterType.FLOAT:
                    space = ParameterSpace(
                        name=cast(str, param["name"]),
                        type=param_type,
                        low=cast(float, param.get("low", 0.0)),
                        high=cast(float, param.get("high", 1.0)),
                    )
                elif param_type == ParameterType.CATEGORICAL:
                    space = ParameterSpace(
                        name=cast(str, param["name"]),
                        type=param_type,
                        categories=cast(List[str], param.get("categories", [])),
                    )
                elif param_type == ParameterType.BOOLEAN:
                    space = ParameterSpace(name=cast(str, param["name"]), type=param_type)
                else:
                    continue
                param_spaces.append(space)

            n_trials = request.n_trials or request.max_trials or 50
            method_str = request.method or request.optimization_method or "random"
            try:
                method = OptimizationMethod[method_str.upper()]
            except KeyError:
                method = OptimizationMethod.RANDOM

            if not param_spaces:
                optimization_result = OptimizationResult(
                    best_params={},
                    best_score=0.0,
                    total_trials=0,
                    history=[],
                    method=method.value,
                )
            else:
                optimization_result = self._optimizer_engine.optimize(
                    param_spaces=param_spaces,
                    objective_fn=lambda **kwargs: 0.0,
                    n_trials=n_trials,
                    method=method,
                )

            self._optimizations[optimization_id] = optimization_result

            logger.info(
                "Optimization completed: %s, best_score=%.4f",
                optimization_id,
                optimization_result.best_score,
            )

            return OptimizeResponse(
                optimization_id=optimization_id,
                system_id=request.system_id or "default",
                optimal_parameters=optimization_result.best_params,
                improvement_metrics={},
                num_trials=optimization_result.total_trials,
                best_params=optimization_result.best_params,
                best_score=optimization_result.best_score,
                n_trials=optimization_result.total_trials,
                metadata={
                    "method": request.method or "random",
                    "multi_objective": request.multi_objective,
                },
            )
        except Exception as e:
            logger.error("Optimization failed: %s", e)
            raise RuntimeError(f"Failed to optimize parameters: {e}") from e
