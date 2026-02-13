"""
Optimizer service - Benchmark methods (mixin).
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List

from beanllm.domain.optimizer import (
    Benchmarker,
    BenchmarkQuery,
    BenchmarkResult,
    QueryType,
)
from beanllm.dto.request.advanced.optimizer_request import BenchmarkRequest
from beanllm.dto.response.advanced.optimizer_response import BenchmarkResponse
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class OptimizerBenchmarkMixin:
    """Mixin providing benchmark() for OptimizerServiceImpl. Uses self._benchmarker and self._benchmarks from the concrete service."""

    _benchmarker: "Benchmarker"
    _benchmarks: Dict[str, BenchmarkResult]

    async def benchmark(self, request: BenchmarkRequest) -> BenchmarkResponse:
        """
        Run system benchmarking.

        Args:
            request: BenchmarkRequest.

        Returns:
            BenchmarkResponse: Benchmark results.

        Raises:
            RuntimeError: If benchmarking fails.
        """
        logger.info(f"Starting benchmark: system_id={request.system_id}, num_queries={request.num_queries}")
        benchmark_id = str(uuid.uuid4())
        try:
            benchmarker: Benchmarker = self._benchmarker
            queries: List[BenchmarkQuery] = []
            if request.test_queries or request.queries:
                qlist = request.test_queries or request.queries or []
                for q in qlist[: request.num_queries]:
                    queries.append(
                        BenchmarkQuery(query=q, type=QueryType.SIMPLE, metadata={})
                    )
            if not queries and request.synthetic:
                queries = benchmarker.generate_queries(
                    num_queries=request.num_queries,
                    domain=request.domain,
                )

            def _placeholder_system_fn(query: str) -> float:
                """Placeholder: caller should provide real system under test."""
                return 0.0

            result: BenchmarkResult = benchmarker.run_benchmark(
                queries=queries,
                system_fn=_placeholder_system_fn,
            )
            self._benchmarks[benchmark_id] = result

            logger.info(
                f"Benchmark completed: {benchmark_id}, "
                f"avg_latency={result.avg_latency:.4f}s, avg_score={result.avg_score:.4f}"
            )

            return BenchmarkResponse(
                benchmark_id=benchmark_id,
                num_queries=len(queries),
                system_id=request.system_id,
                system_type=request.system_type,
                queries=[q.query for q in result.queries],
                baseline_metrics={
                    "avg_latency": result.avg_latency,
                    "avg_score": result.avg_score,
                    "p50_latency": result.p50_latency,
                    "p95_latency": result.p95_latency,
                    "p99_latency": result.p99_latency,
                    "throughput": result.throughput,
                    "total_duration": result.total_duration,
                },
                detailed_results=[],
                bottlenecks=[],
                avg_latency=result.avg_latency,
                p50_latency=result.p50_latency,
                p95_latency=result.p95_latency,
                p99_latency=result.p99_latency,
                avg_score=result.avg_score,
                min_score=result.min_score,
                max_score=result.max_score,
                throughput=result.throughput,
                total_duration=result.total_duration,
                metadata={"benchmark_id": benchmark_id},
            )
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise RuntimeError(f"Failed to run benchmark: {e}") from e
