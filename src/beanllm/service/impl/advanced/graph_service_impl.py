"""
GraphServiceImpl - Graph 서비스 구현체
SOLID 원칙:
- SRP: Graph 비즈니스 로직만 담당
- DIP: 인터페이스에 의존 (의존성 주입)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Set, cast

from beanllm.domain.graph import GraphState, NodeCache
from beanllm.dto.request.graph.graph_request import GraphRequest
from beanllm.dto.response.graph.graph_response import GraphResponse
from beanllm.infrastructure.distributed.pipeline_decorators import with_distributed_features
from beanllm.service.graph_service import IGraphService
from beanllm.utils.logging import get_logger

if TYPE_CHECKING:
    from beanllm.domain.graph import BaseNode

logger = get_logger(__name__)


class GraphServiceImpl(IGraphService):
    """
    Graph 서비스 구현체

    책임:
    - Graph 비즈니스 로직만
    - 검증 없음 (Handler에서 처리)
    - 에러 처리 없음 (Handler에서 처리)

    SOLID:
    - SRP: Graph 비즈니스 로직만
    - DIP: 인터페이스에 의존 (의존성 주입)
    """

    def __init__(self) -> None:
        """의존성 주입을 통한 생성자"""
        pass

    @with_distributed_features(
        pipeline_type="graph",
        enable_rate_limiting=True,
        enable_event_streaming=True,
        enable_distributed_lock=True,
        rate_limit_key=lambda self, args, kwargs: (
            f"graph:node:{(args[0] if args else kwargs.get('request')).entry_point or 'default'}"
        ),
        lock_key=lambda self, args, kwargs: (
            f"graph:node:{hash(str((args[0] if args else kwargs.get('request')).nodes)) if hasattr(args[0] if args else kwargs.get('request'), 'nodes') else 'default'}"
        ),
        event_type="graph.run",
    )
    async def run_graph(self, request: GraphRequest) -> GraphResponse:
        """
        Graph 실행 (기존 graph.py의 Graph.run() 정확히 마이그레이션)

        Args:
            request: Graph 요청 DTO

        Returns:
            GraphResponse: Graph 응답 DTO
        """
        # State 생성 (기존과 동일)
        if isinstance(request.initial_state, dict):
            state = GraphState(data=request.initial_state)
        else:
            state = request.initial_state

        # 노드 딕셔너리 생성 (기존: self.nodes)
        nodes: Dict[str, "BaseNode"] = {}
        for node in request.nodes or []:
            base_node = cast("BaseNode", node)
            nodes[base_node.name] = base_node

        # 캐시 생성 (기존과 동일)
        cache = NodeCache() if request.enable_cache else None

        # 시작 노드 결정 (기존과 동일)
        if request.entry_point:
            current_node = request.entry_point
        else:
            if not nodes:
                raise ValueError("No nodes in graph")
            current_node = next(iter(nodes))

        visited: Set[str] = set()
        max_iterations = request.max_iterations

        # 기존 graph.py의 Graph.run() 로직 정확히 마이그레이션
        for iteration in range(max_iterations):
            if current_node in visited:
                logger.warning(f"Node {current_node} already visited, stopping")
                break

            if current_node not in nodes:
                logger.error(f"Node not found: {current_node}")
                break

            visited.add(current_node)

            if request.verbose:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Executing node: {current_node}")
                logger.info(f"{'=' * 60}")

            # 노드 실행
            node = nodes[current_node]

            # 캐시 체크 (기존과 동일)
            if cache and node.cache_enabled:
                cached_result = cache.get(current_node, state)
                if cached_result is not None:
                    update = cached_result
                    if request.verbose:
                        logger.info("Using cached result")
                else:
                    update = await node.execute(state)
                    cache.set(current_node, state, update)
            else:
                update = await node.execute(state)

            # 상태 업데이트 (기존과 동일)
            state.update(update)

            if request.verbose:
                logger.info(f"State updated: {list(update.keys())}")

            # 다음 노드 결정 (기존과 동일)
            next_node = None

            if current_node in (request.conditional_edges or {}):
                condition_func = request.conditional_edges[current_node]
                next_node = condition_func(state)
                if request.verbose:
                    logger.info(f"Conditional edge -> {next_node}")
            elif current_node in (request.edges or {}):
                edges = request.edges[current_node]
                if edges:
                    next_node = edges[0]
                    if request.verbose:
                        logger.info(f"Edge -> {next_node}")

            if not next_node:
                if request.verbose:
                    logger.info("No next node, finishing")
                break

            current_node = cast(str, next_node)

        # 캐시 통계 (기존과 동일)
        cache_stats = None
        if cache and request.verbose:
            cache_stats = cache.get_stats()
            logger.info(f"\nCache stats: {cache_stats}")

        # 결과 반환
        return GraphResponse(
            final_state=state.data,
            metadata=state.metadata,
            cache_stats=cache_stats,
            visited_nodes=list(visited),
            iterations=iteration + 1,
        )
