"""
Profiler - 컴포넌트별 성능 프로파일링
SOLID 원칙:
- SRP: 프로파일링만 담당
- OCP: 새로운 메트릭 추가 가능
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class ComponentType(Enum):
    """컴포넌트 타입"""

    EMBEDDING = "embedding"  # Embedding 생성
    RETRIEVAL = "retrieval"  # 문서 검색
    RERANKING = "reranking"  # 재순위화
    GENERATION = "generation"  # LLM 생성
    PREPROCESSING = "preprocessing"  # 전처리
    POSTPROCESSING = "postprocessing"  # 후처리
    TOTAL = "total"  # 전체


@dataclass
class ComponentMetrics:
    """
    컴포넌트 메트릭

    Attributes:
        component_type: 컴포넌트 타입
        duration_ms: 실행 시간 (밀리초)
        memory_mb: 메모리 사용량 (MB)
        token_count: 토큰 수 (LLM 호출 시)
        estimated_cost: 추정 비용 ($)
        metadata: 추가 메타데이터
    """

    component_type: ComponentType
    duration_ms: float = 0.0
    memory_mb: float = 0.0
    token_count: int = 0
    estimated_cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileResult:
    """
    프로파일링 결과

    Attributes:
        components: 컴포넌트별 메트릭
        total_duration_ms: 총 실행 시간
        total_cost: 총 비용
        bottleneck: 병목 컴포넌트
        recommendations: 최적화 권장사항
    """

    components: Dict[str, ComponentMetrics] = field(default_factory=dict)
    total_duration_ms: float = 0.0
    total_cost: float = 0.0
    bottleneck: Optional[ComponentType] = None
    recommendations: List[str] = field(default_factory=list)

    def __post_init__(self):
        """자동 계산"""
        if self.components:
            # Calculate totals
            self.total_duration_ms = sum(m.duration_ms for m in self.components.values())
            self.total_cost = sum(m.estimated_cost for m in self.components.values())

            # Find bottleneck
            max_duration_component = max(
                self.components.items(),
                key=lambda x: x[1].duration_ms,
            )
            self.bottleneck = max_duration_component[1].component_type

    def get_breakdown(self) -> Dict[str, float]:
        """
        컴포넌트별 시간 비율 반환

        Returns:
            {component_name: percentage}
        """
        if self.total_duration_ms == 0:
            return {}

        return {
            name: (metrics.duration_ms / self.total_duration_ms * 100)
            for name, metrics in self.components.items()
        }


class Profiler:
    """
    성능 프로파일러

    책임:
    - 컴포넌트별 실행 시간 측정
    - 메모리 사용량 추적
    - 비용 추정
    - 병목 지점 식별

    Example:
        ```python
        profiler = Profiler()

        # Start profiling
        profiler.start("total")

        # Profile embedding
        with profiler.profile("embedding"):
            embeddings = embedding_model.embed(documents)

        # Profile retrieval
        with profiler.profile("retrieval"):
            results = vector_store.search(query_embedding, top_k=10)

        # Profile generation
        with profiler.profile("generation") as p:
            response = llm.generate(prompt)
            p.set_tokens(response.token_count)  # Track tokens

        profiler.end("total")

        # Get results
        result = profiler.get_result()
        print(f"Total time: {result.total_duration_ms}ms")
        print(f"Bottleneck: {result.bottleneck}")
        print(f"Breakdown: {result.get_breakdown()}")
        ```
    """

    def __init__(self) -> None:
        """Initialize profiler"""
        self._metrics: Dict[str, ComponentMetrics] = {}
        self._start_times: Dict[str, float] = {}
        self._active_profiles: List[str] = []

    def start(self, component_name: str) -> None:
        """
        컴포넌트 프로파일링 시작

        Args:
            component_name: 컴포넌트 이름
        """
        self._start_times[component_name] = time.time()
        self._active_profiles.append(component_name)

        logger.debug(f"Started profiling: {component_name}")

    def end(self, component_name: str) -> ComponentMetrics:
        """
        컴포넌트 프로파일링 종료

        Args:
            component_name: 컴포넌트 이름

        Returns:
            ComponentMetrics: 측정된 메트릭
        """
        if component_name not in self._start_times:
            logger.warning(f"Component {component_name} was not started")
            return ComponentMetrics(component_type=ComponentType.TOTAL)

        duration = time.time() - self._start_times[component_name]
        duration_ms = duration * 1000

        # Determine component type
        component_type = self._infer_component_type(component_name)

        metrics = ComponentMetrics(
            component_type=component_type,
            duration_ms=duration_ms,
        )

        self._metrics[component_name] = metrics

        if component_name in self._active_profiles:
            self._active_profiles.remove(component_name)

        logger.debug(f"Ended profiling: {component_name}, duration={duration_ms:.2f}ms")

        return metrics

    def profile(self, component_name: str) -> "ProfileContext":
        """
        Context manager로 프로파일링

        Args:
            component_name: 컴포넌트 이름

        Returns:
            ProfileContext: Context manager

        Example:
            ```python
            with profiler.profile("embedding"):
                embeddings = model.embed(texts)
            ```
        """
        return ProfileContext(self, component_name)

    def set_tokens(self, component_name: str, token_count: int) -> None:
        """
        토큰 수 설정 (LLM 호출 시)

        Args:
            component_name: 컴포넌트 이름
            token_count: 토큰 수
        """
        if component_name in self._metrics:
            self._metrics[component_name].token_count = token_count

            # Estimate cost (예: GPT-4: $0.03/1K tokens)
            cost_per_1k_tokens = 0.03
            self._metrics[component_name].estimated_cost = token_count / 1000 * cost_per_1k_tokens

    def set_memory(self, component_name: str, memory_mb: float) -> None:
        """
        메모리 사용량 설정

        Args:
            component_name: 컴포넌트 이름
            memory_mb: 메모리 사용량 (MB)
        """
        if component_name in self._metrics:
            self._metrics[component_name].memory_mb = memory_mb

    def get_result(self) -> ProfileResult:
        """
        프로파일링 결과 반환

        Returns:
            ProfileResult: 프로파일링 결과
        """
        result = ProfileResult(components=self._metrics.copy())

        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)

        return result

    def reset(self) -> None:
        """프로파일러 리셋"""
        self._metrics.clear()
        self._start_times.clear()
        self._active_profiles.clear()

        logger.debug("Profiler reset")

    def _infer_component_type(self, component_name: str) -> ComponentType:
        """컴포넌트 타입 추론"""
        name_lower = component_name.lower()

        if "embed" in name_lower:
            return ComponentType.EMBEDDING
        elif "retriev" in name_lower or "search" in name_lower:
            return ComponentType.RETRIEVAL
        elif "rerank" in name_lower:
            return ComponentType.RERANKING
        elif "generat" in name_lower or "llm" in name_lower:
            return ComponentType.GENERATION
        elif "preprocess" in name_lower:
            return ComponentType.PREPROCESSING
        elif "postprocess" in name_lower:
            return ComponentType.POSTPROCESSING
        elif "total" in name_lower:
            return ComponentType.TOTAL
        else:
            return ComponentType.TOTAL

    def _generate_recommendations(self, result: ProfileResult) -> List[str]:
        """최적화 권장사항 생성"""
        recommendations = []

        breakdown = result.get_breakdown()

        # Check for slow components (>40% of total time)
        for component_name, percentage in breakdown.items():
            if percentage > 40:
                metrics = result.components[component_name]

                if metrics.component_type == ComponentType.EMBEDDING:
                    recommendations.append(
                        f"Embedding takes {percentage:.1f}% of time. "
                        "Consider caching embeddings or using a faster embedding model."
                    )
                elif metrics.component_type == ComponentType.RETRIEVAL:
                    recommendations.append(
                        f"Retrieval takes {percentage:.1f}% of time. "
                        "Consider optimizing index (e.g., HNSW) or reducing top_k."
                    )
                elif metrics.component_type == ComponentType.GENERATION:
                    recommendations.append(
                        f"Generation takes {percentage:.1f}% of time. "
                        "Consider using a faster model or reducing max_tokens."
                    )
                elif metrics.component_type == ComponentType.RERANKING:
                    recommendations.append(
                        f"Reranking takes {percentage:.1f}% of time. "
                        "Consider reducing rerank candidates or using a lighter reranker."
                    )

        # Check for high cost
        if result.total_cost > 0.10:  # $0.10 per query
            recommendations.append(
                f"High cost detected (${result.total_cost:.4f} per query). "
                "Consider using a cheaper model or reducing token usage."
            )

        # Check for slow total time
        if result.total_duration_ms > 5000:  # 5 seconds
            recommendations.append(
                f"Total latency is high ({result.total_duration_ms:.0f}ms). "
                "Consider parallelizing components or using async processing."
            )

        return recommendations


class ProfileContext:
    """
    프로파일링 Context Manager

    Example:
        ```python
        with profiler.profile("my_component") as p:
            result = expensive_operation()
            p.set_tokens(result.token_count)
        ```
    """

    def __init__(self, profiler: Profiler, component_name: str) -> None:
        """
        Args:
            profiler: Profiler 인스턴스
            component_name: 컴포넌트 이름
        """
        self.profiler = profiler
        self.component_name = component_name
        self.metrics: Optional[ComponentMetrics] = None

    def __enter__(self) -> "ProfileContext":
        """Enter context"""
        self.profiler.start(self.component_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context"""
        self.metrics = self.profiler.end(self.component_name)

    def set_tokens(self, token_count: int) -> None:
        """토큰 수 설정"""
        self.profiler.set_tokens(self.component_name, token_count)

    def set_memory(self, memory_mb: float) -> None:
        """메모리 사용량 설정"""
        self.profiler.set_memory(self.component_name, memory_mb)


# ========================================
# Convenience Functions
# ========================================


def profile_rag_pipeline(
    rag_fn: Callable[[str], Any],
    query: str,
    component_names: Optional[Dict[str, str]] = None,
) -> ProfileResult:
    """
    RAG 파이프라인 프로파일링 (편의 함수)

    Args:
        rag_fn: RAG 함수 (query -> result)
        query: 쿼리
        component_names: 컴포넌트 이름 매핑

    Returns:
        ProfileResult: 프로파일링 결과

    Example:
        ```python
        def my_rag(query):
            # ... RAG logic ...
            return result

        profile_result = profile_rag_pipeline(my_rag, "What is AI?")
        print(profile_result.get_breakdown())
        ```
    """
    profiler = Profiler()

    profiler.start("total")
    result = rag_fn(query)
    profiler.end("total")

    return profiler.get_result()
