"""
Standalone optimizer functions - quick_optimizer, quick_profile.

Module-level one-liners that create an Optimizer instance and run a single operation.
"""

from __future__ import annotations

from typing import Any, Dict, List

from beanllm.dto.response.advanced.optimizer_response import (
    OptimizeResponse,
    ProfileResponse,
)


async def quick_optimizer(
    parameters: List[Dict[str, Any]],
    method: str = "bayesian",
    n_trials: int = 30,
) -> OptimizeResponse:
    """
    One-liner for quick optimization.

    Args:
        parameters: Parameter definitions
        method: Optimization method (default: "bayesian")
        n_trials: Number of trials (default: 30)

    Returns:
        OptimizeResponse: Optimization results

    Example:
        ```python
        from beanllm.facade.advanced.optimizer_facade import quick_optimizer

        result = await quick_optimizer(
            parameters=[
                {"name": "top_k", "type": "integer", "low": 1, "high": 20},
            ],
            method="bayesian",
            n_trials=20
        )
        print(f"Best top_k: {result.best_params['top_k']}")
        ```
    """
    from beanllm.facade.advanced.optimizer_facade import Optimizer

    optimizer = Optimizer()
    return await optimizer.optimize(
        parameters=parameters,
        method=method,
        n_trials=n_trials,
    )


async def quick_profile() -> ProfileResponse:
    """
    One-liner for quick profiling.

    Returns:
        ProfileResponse: Profile results

    Example:
        ```python
        from beanllm.facade.advanced.optimizer_facade import quick_profile

        result = await quick_profile()
        print(f"Bottleneck: {result.bottleneck}")
        print(f"Total duration: {result.total_duration_ms}ms")
        ```
    """
    from beanllm.facade.advanced.optimizer_facade import Optimizer

    optimizer = Optimizer()
    return await optimizer.profile()
