"""
GraphServiceImpl - Graph 서비스 구현체
SOLID 원칙:
- SRP: Graph 비즈니스 로직만 담당
- DIP: 인터페이스에 의존 (의존성 주입)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Set

from ...domain.graph import GraphState, NodeCache
from ...dto.request.graph_request import GraphRequest
from ...dto.response.graph_response import GraphResponse
from ...utils.logger import get_logger
from ..graph_service import IGraphService

if TYPE_CHECKING:
    from ...domain.graph import BaseNode

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
            nodes[node.name] = node

        # 캐시 생성 (기존과 동일)
        cache = NodeCache() if request.enable_cache else None

        # 시작 노드 결정 (기존과 동일)
        if request.entry_point:
            current_node = request.entry_point
        else:
            # 첫 번째 노드
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

            # 조건부 엣지 확인 (기존과 동일)
            if current_node in (request.conditional_edges or {}):
                condition_func = request.conditional_edges[current_node]
                next_node = condition_func(state)
                if request.verbose:
                    logger.info(f"Conditional edge -> {next_node}")

            # 일반 엣지 확인 (기존과 동일)
            elif current_node in (request.edges or {}):
                edges = request.edges[current_node]
                if edges:
                    next_node = edges[0]  # 첫 번째 엣지
                    if request.verbose:
                        logger.info(f"Edge -> {next_node}")

            # 다음 노드 없으면 종료 (기존과 동일)
            if not next_node:
                if request.verbose:
                    logger.info("No next node, finishing")
                break

            current_node = next_node

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
