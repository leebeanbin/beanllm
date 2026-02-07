"""
Benchmarker - 합성 쿼리 생성 및 벤치마킹
SOLID 원칙:
- SRP: 벤치마킹만 담당
- OCP: 새로운 쿼리 타입 추가 가능
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class QueryType(Enum):
    """쿼리 타입"""

    SIMPLE = "simple"  # 간단한 팩트 쿼리
    COMPLEX = "complex"  # 복잡한 추론 쿼리
    EDGE_CASE = "edge_case"  # 엣지 케이스 (애매한 쿼리, 오타 등)
    MULTI_HOP = "multi_hop"  # 다단계 추론 필요
    AGGREGATION = "aggregation"  # 집계 필요


@dataclass
class BenchmarkQuery:
    """
    벤치마크 쿼리

    Attributes:
        query: 쿼리 텍스트
        type: 쿼리 타입
        expected_answer: 기대 답변 (optional, evaluation용)
        metadata: 추가 메타데이터
    """

    query: str
    type: QueryType
    expected_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """
    벤치마크 결과

    Attributes:
        queries: 사용된 쿼리 목록
        latencies: 각 쿼리별 지연시간 (초)
        scores: 각 쿼리별 품질 점수 (0.0-1.0)
        avg_latency: 평균 지연시간
        avg_score: 평균 품질 점수
        min_score: 최소 점수
        max_score: 최대 점수
        p50_latency: 50th percentile 지연시간
        p95_latency: 95th percentile 지연시간
        p99_latency: 99th percentile 지연시간
        throughput: 처리량 (queries/sec)
        total_duration: 총 실행 시간 (초)
        metadata: 추가 메타데이터
    """

    queries: List[BenchmarkQuery]
    latencies: List[float]
    scores: List[float]
    avg_latency: float = 0.0
    avg_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    throughput: float = 0.0
    total_duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """통계 계산"""
        if self.latencies:
            self.avg_latency = sum(self.latencies) / len(self.latencies)

            sorted_latencies = sorted(self.latencies)
            n = len(sorted_latencies)

            self.p50_latency = sorted_latencies[int(n * 0.5)]
            self.p95_latency = sorted_latencies[int(n * 0.95)]
            self.p99_latency = sorted_latencies[int(n * 0.99)]

            self.total_duration = sum(self.latencies)
            self.throughput = (
                len(self.latencies) / self.total_duration if self.total_duration > 0 else 0.0
            )

        if self.scores:
            self.avg_score = sum(self.scores) / len(self.scores)
            self.min_score = min(self.scores)
            self.max_score = max(self.scores)


class Benchmarker:
    """
    벤치마커

    책임:
    - 합성 쿼리 생성
    - 시스템 성능 벤치마킹
    - 베이스라인 측정

    Example:
        ```python
        benchmarker = Benchmarker()

        # Generate synthetic queries
        queries = benchmarker.generate_queries(
            num_queries=50,
            query_types=[QueryType.SIMPLE, QueryType.COMPLEX],
            domain="machine learning"
        )

        # Run benchmark
        def system_under_test(query):
            result = rag_system.query(query)
            score = evaluate(result)  # 0.0-1.0
            return score

        result = benchmarker.run_benchmark(
            queries=queries,
            system_fn=system_under_test
        )

        print(f"Avg latency: {result.avg_latency:.3f}s")
        print(f"Avg score: {result.avg_score:.3f}")
        print(f"P95 latency: {result.p95_latency:.3f}s")
        print(f"Throughput: {result.throughput:.1f} queries/sec")
        ```
    """

    def __init__(self) -> None:
        """Initialize benchmarker"""
        pass

    def generate_queries(
        self,
        num_queries: int = 50,
        query_types: Optional[List[QueryType]] = None,
        domain: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> List[BenchmarkQuery]:
        """
        합성 쿼리 생성

        Args:
            num_queries: 생성할 쿼리 수
            query_types: 쿼리 타입 리스트 (None이면 모든 타입)
            domain: 도메인 (e.g., "machine learning", "healthcare")
            seed: 랜덤 시드

        Returns:
            List[BenchmarkQuery]: 생성된 쿼리 리스트
        """
        import random

        if seed is not None:
            random.seed(seed)

        if query_types is None:
            query_types = list(QueryType)

        queries = []

        for i in range(num_queries):
            query_type = random.choice(query_types)

            # Generate query based on type
            if query_type == QueryType.SIMPLE:
                query_text = self._generate_simple_query(domain)
            elif query_type == QueryType.COMPLEX:
                query_text = self._generate_complex_query(domain)
            elif query_type == QueryType.EDGE_CASE:
                query_text = self._generate_edge_case_query(domain)
            elif query_type == QueryType.MULTI_HOP:
                query_text = self._generate_multi_hop_query(domain)
            elif query_type == QueryType.AGGREGATION:
                query_text = self._generate_aggregation_query(domain)
            else:
                query_text = self._generate_simple_query(domain)

            queries.append(
                BenchmarkQuery(
                    query=query_text,
                    type=query_type,
                    metadata={"index": i, "domain": domain},
                )
            )

        logger.info(f"Generated {num_queries} synthetic queries")

        return queries

    def _generate_simple_query(self, domain: Optional[str] = None) -> str:
        """간단한 팩트 쿼리 생성"""
        import random

        templates = [
            "What is {concept}?",
            "Define {concept}",
            "Explain {concept} in simple terms",
            "What does {concept} mean?",
            "How does {concept} work?",
        ]

        # Domain-specific concepts
        if domain == "machine learning":
            concepts = [
                "gradient descent",
                "backpropagation",
                "overfitting",
                "cross-validation",
                "regularization",
                "neural networks",
                "ensemble methods",
            ]
        elif domain == "healthcare":
            concepts = [
                "hypertension",
                "diabetes",
                "immunization",
                "antibiotic resistance",
                "telemedicine",
            ]
        else:
            concepts = [
                "artificial intelligence",
                "quantum computing",
                "blockchain",
                "cloud computing",
                "cybersecurity",
            ]

        template = random.choice(templates)
        concept = random.choice(concepts)

        return template.format(concept=concept)

    def _generate_complex_query(self, domain: Optional[str] = None) -> str:
        """복잡한 추론 쿼리 생성"""
        import random

        templates = [
            "Compare and contrast {concept1} and {concept2}",
            "What are the advantages and disadvantages of {concept}?",
            "How does {concept1} relate to {concept2}?",
            "Explain the difference between {concept1} and {concept2}",
            "What are the trade-offs between {concept1} and {concept2}?",
        ]

        if domain == "machine learning":
            concepts = [
                "supervised learning",
                "unsupervised learning",
                "reinforcement learning",
                "decision trees",
                "random forests",
                "gradient boosting",
            ]
        else:
            concepts = [
                "microservices",
                "monolithic architecture",
                "SQL databases",
                "NoSQL databases",
                "REST APIs",
                "GraphQL",
            ]

        template = random.choice(templates)

        if "{concept1}" in template:
            concept1, concept2 = random.sample(concepts, 2)
            return template.format(concept1=concept1, concept2=concept2)
        else:
            concept = random.choice(concepts)
            return template.format(concept=concept)

    def _generate_edge_case_query(self, domain: Optional[str] = None) -> str:
        """엣지 케이스 쿼리 생성 (오타, 애매한 표현 등)"""
        import random

        # Generate a base query
        base_query = self._generate_simple_query(domain)

        # Apply edge case transformation
        transformations = [
            lambda q: q.lower(),  # 소문자
            lambda q: q.replace("?", ""),  # 물음표 제거
            lambda q: q + "...",  # 말줄임표 추가
            lambda q: self._introduce_typo(q),  # 오타 추가
        ]

        transformation = random.choice(transformations)
        return str(transformation(base_query))

    def _introduce_typo(self, text: str) -> str:
        """무작위로 오타 추가"""
        import random

        if len(text) < 5:
            return text

        # 한 글자를 무작위로 변경
        idx = random.randint(0, len(text) - 1)
        char = text[idx]

        # 인접한 키로 교체 (간단한 시뮬레이션)
        keyboard_neighbors = {
            "a": ["s", "q", "w"],
            "e": ["r", "w", "d"],
            "i": ["u", "o", "k"],
            "o": ["i", "p", "l"],
            "u": ["y", "i", "j"],
        }

        if char.lower() in keyboard_neighbors:
            replacement = random.choice(keyboard_neighbors[char.lower()])
            return text[:idx] + replacement + text[idx + 1 :]

        return text

    def _generate_multi_hop_query(self, domain: Optional[str] = None) -> str:
        """다단계 추론 쿼리 생성"""
        import random

        templates = [
            "If {condition}, then what would be the impact on {concept}?",
            "Assuming {condition}, how would {concept} change?",
            "What happens to {concept1} when {concept2} increases?",
        ]

        template = random.choice(templates)

        if domain == "machine learning":
            return template.format(
                condition="we increase the learning rate",
                concept="model convergence",
                concept1="training loss",
                concept2="batch size",
            )
        else:
            return template.format(
                condition="we scale horizontally",
                concept="system throughput",
                concept1="latency",
                concept2="load",
            )

    def _generate_aggregation_query(self, domain: Optional[str] = None) -> str:
        """집계 쿼리 생성"""
        import random

        templates = [
            "List all {concept} techniques",
            "What are the main types of {concept}?",
            "Summarize the key points about {concept}",
            "What are common {concept} approaches?",
        ]

        template = random.choice(templates)

        if domain == "machine learning":
            concepts = ["optimization", "regularization", "feature engineering"]
        else:
            concepts = ["authentication", "caching", "load balancing"]

        concept = random.choice(concepts)
        return template.format(concept=concept)

    def run_benchmark(
        self,
        queries: List[BenchmarkQuery],
        system_fn: Callable[[str], float],
        warmup: int = 5,
    ) -> BenchmarkResult:
        """
        벤치마크 실행

        Args:
            queries: 벤치마크 쿼리 리스트
            system_fn: 시스템 함수 (query_text -> quality_score)
            warmup: 워밍업 쿼리 수

        Returns:
            BenchmarkResult: 벤치마크 결과
        """
        logger.info(f"Running benchmark with {len(queries)} queries (warmup={warmup})")

        latencies = []
        scores = []

        # Warmup
        for i in range(min(warmup, len(queries))):
            _ = system_fn(queries[i].query)

        # Actual benchmark
        for i, query in enumerate(queries):
            start_time = time.time()

            try:
                score = system_fn(query.query)
            except Exception as e:
                logger.error(f"Error on query {i}: {e}")
                score = 0.0

            latency = time.time() - start_time

            latencies.append(latency)
            scores.append(score)

            if (i + 1) % 10 == 0:
                logger.debug(f"Processed {i + 1}/{len(queries)} queries")

        result = BenchmarkResult(
            queries=queries,
            latencies=latencies,
            scores=scores,
        )

        logger.info(
            f"Benchmark completed: avg_latency={result.avg_latency:.3f}s, "
            f"avg_score={result.avg_score:.3f}, "
            f"throughput={result.throughput:.1f} q/s"
        )

        return result

    def compare_baselines(
        self,
        queries: List[BenchmarkQuery],
        systems: Dict[str, Callable[[str], float]],
    ) -> Dict[str, BenchmarkResult]:
        """
        여러 시스템 비교

        Args:
            queries: 벤치마크 쿼리 리스트
            systems: {system_name: system_fn}

        Returns:
            Dict[str, BenchmarkResult]: 시스템별 벤치마크 결과
        """
        logger.info(f"Comparing {len(systems)} systems")

        results = {}

        for system_name, system_fn in systems.items():
            logger.info(f"Benchmarking {system_name}...")
            results[system_name] = self.run_benchmark(queries, system_fn)

        return results

    def generate_latency_distribution(self, result: BenchmarkResult) -> Dict[str, List[float]]:
        """
        지연시간 분포 데이터 생성

        Returns:
            {
                "buckets": [0.0, 0.1, 0.2, ...],  # 버킷 경계
                "counts": [10, 25, 40, ...]  # 각 버킷의 쿼리 수
            }
        """

        latencies = result.latencies

        if not latencies:
            return {"buckets": [], "counts": []}

        # Determine bucket size
        min_latency = min(latencies)
        max_latency = max(latencies)
        num_buckets = 20

        bucket_size = (max_latency - min_latency) / num_buckets

        # Create buckets
        buckets = [min_latency + i * bucket_size for i in range(num_buckets + 1)]
        counts = [0] * num_buckets

        # Count latencies in each bucket
        for latency in latencies:
            bucket_idx = int((latency - min_latency) / bucket_size)
            bucket_idx = min(bucket_idx, num_buckets - 1)
            counts[bucket_idx] += 1

        return {"buckets": buckets, "counts": counts}
