"""
GraphHandler - Graph 요청 처리 (Controller 역할)
책임 분리:
- 모든 if-else/try-catch 처리
- 입력 검증
- DTO 변환
- 결과 출력
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from ..decorators.error_handler import handle_errors
from ..decorators.logger import log_handler_call
from ..decorators.validation import validate_input
from ..dto.request.graph_request import GraphRequest
from ..dto.response.graph_response import GraphResponse
from ..service.graph_service import IGraphService


class GraphHandler:
    """
    Graph 요청 처리 Handler

    책임:
    - 입력 검증 (if-else)
    - 에러 처리 (try-catch)
    - DTO 변환
    - Service 호출
    - 비즈니스 로직 없음
    """

    def __init__(self, graph_service: IGraphService) -> None:
        """
        의존성 주입

        Args:
            graph_service: Graph 서비스 (인터페이스에 의존 - DIP)
        """
        self._graph_service = graph_service

    @log_handler_call
    @handle_errors(error_message="Graph execution failed")
    @validate_input(
        required_params=["initial_state"],
        param_types={"initial_state": dict, "enable_cache": bool, "verbose": bool},
    )
    async def handle_run(
        self,
        initial_state: Dict[str, Any],
        nodes: Optional[List[Any]] = None,
        edges: Optional[Dict[str, List[str]]] = None,
        conditional_edges: Optional[Dict[str, Callable]] = None,
        entry_point: Optional[str] = None,
        enable_cache: bool = True,
        verbose: bool = False,
        max_iterations: int = 100,
        **kwargs: Any,
    ) -> GraphResponse:
        """
        Graph 실행 요청 처리 (모든 검증 및 에러 처리 포함)

        Args:
            initial_state: 초기 상태
            nodes: 노드 리스트
            edges: 엣지 딕셔너리
            conditional_edges: 조건부 엣지 딕셔너리
            entry_point: 시작 노드
            enable_cache: 캐싱 활성화
            verbose: 상세 로그
            max_iterations: 최대 반복 횟수
            **kwargs: 추가 파라미터

        Returns:
            GraphResponse: Graph 응답

        책임:
            - 입력 검증 (decorator로 처리)
            - 에러 처리 (decorator로 처리)
            - DTO 변환
            - Service 호출
        """
        # DTO 생성
        request = GraphRequest(
            initial_state=initial_state,
            nodes=nodes or [],
            edges=edges or {},
            conditional_edges=conditional_edges or {},
            entry_point=entry_point,
            enable_cache=enable_cache,
            verbose=verbose,
            max_iterations=max_iterations,
            extra_params=kwargs,
        )

        # Service 호출 (에러 처리는 decorator가 담당)
        return await self._graph_service.run_graph(request)
