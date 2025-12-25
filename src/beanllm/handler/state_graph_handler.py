"""
StateGraphHandler - StateGraph 요청 처리 (Controller 역할)
책임 분리:
- 모든 if-else/try-catch 처리
- 입력 검증
- DTO 변환
- 결과 출력
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, Optional, Type, Union

from ..decorators.error_handler import handle_errors
from ..decorators.logger import log_handler_call
from ..decorators.validation import validate_input
from ..domain.state_graph import END
from ..dto.request.state_graph_request import StateGraphRequest
from ..dto.response.state_graph_response import StateGraphResponse
from ..service.state_graph_service import IStateGraphService


class StateGraphHandler:
    """
    StateGraph 요청 처리 Handler

    책임:
    - 입력 검증 (if-else)
    - 에러 처리 (try-catch)
    - DTO 변환
    - Service 호출
    - 비즈니스 로직 없음
    """

    def __init__(self, state_graph_service: IStateGraphService) -> None:
        """
        의존성 주입

        Args:
            state_graph_service: StateGraph 서비스 (인터페이스에 의존 - DIP)
        """
        self._state_graph_service = state_graph_service

    @log_handler_call
    @handle_errors(error_message="StateGraph execution failed")
    @validate_input(
        required_params=["initial_state"],
        param_types={"initial_state": dict, "entry_point": str},
    )
    async def handle_invoke(
        self,
        initial_state: Dict[str, Any],
        state_schema: Optional[Type] = None,
        nodes: Optional[Dict[str, Callable]] = None,
        edges: Optional[Dict[str, Union[str, Type[END]]]] = None,
        conditional_edges: Optional[Dict[str, tuple]] = None,
        entry_point: Optional[str] = None,
        execution_id: Optional[str] = None,
        resume_from: Optional[str] = None,
        max_iterations: int = 100,
        enable_checkpointing: bool = False,
        checkpoint_dir: Optional[Any] = None,  # Path
        debug: bool = False,
        **kwargs: Any,
    ) -> StateGraphResponse:
        """
        StateGraph 실행 요청 처리 (모든 검증 및 에러 처리 포함)

        Args:
            initial_state: 초기 상태
            state_schema: State TypedDict 클래스
            nodes: 노드 딕셔너리
            edges: 엣지 딕셔너리
            conditional_edges: 조건부 엣지 딕셔너리
            entry_point: 시작 노드
            execution_id: 실행 ID
            resume_from: 재개할 노드
            max_iterations: 최대 반복 횟수
            enable_checkpointing: 체크포인팅 활성화
            checkpoint_dir: 체크포인트 디렉토리
            debug: 디버그 모드
            **kwargs: 추가 파라미터

        Returns:
            StateGraphResponse: StateGraph 응답

        책임:
            - 입력 검증 (decorator로 처리)
            - 에러 처리 (decorator로 처리)
            - DTO 변환
            - Service 호출
        """
        # DTO 생성
        request = StateGraphRequest(
            initial_state=initial_state,
            state_schema=state_schema,
            nodes=nodes or {},
            edges=edges or {},
            conditional_edges=conditional_edges or {},
            entry_point=entry_point,
            execution_id=execution_id,
            resume_from=resume_from,
            max_iterations=max_iterations,
            enable_checkpointing=enable_checkpointing,
            checkpoint_dir=checkpoint_dir,
            debug=debug,
            extra_params=kwargs,
        )

        # Service 호출 (에러 처리는 decorator가 담당)
        return await self._state_graph_service.invoke(request)

    @log_handler_call
    @handle_errors(error_message="StateGraph streaming failed")
    @validate_input(
        required_params=["initial_state", "entry_point"],
        param_types={"initial_state": dict, "entry_point": str},
    )
    def handle_stream(
        self,
        initial_state: Dict[str, Any],
        state_schema: Optional[Type] = None,
        nodes: Optional[Dict[str, Callable]] = None,
        edges: Optional[Dict[str, Union[str, Type[END]]]] = None,
        conditional_edges: Optional[Dict[str, tuple]] = None,
        entry_point: Optional[str] = None,
        execution_id: Optional[str] = None,
        max_iterations: int = 100,
        enable_checkpointing: bool = False,
        checkpoint_dir: Optional[Any] = None,  # Path
        debug: bool = False,
        **kwargs: Any,
    ) -> Iterator[tuple[str, Dict[str, Any]]]:
        """
        StateGraph 스트리밍 실행 요청 처리

        Args:
            initial_state: 초기 상태
            state_schema: State TypedDict 클래스
            nodes: 노드 딕셔너리
            edges: 엣지 딕셔너리
            conditional_edges: 조건부 엣지 딕셔너리
            entry_point: 시작 노드
            execution_id: 실행 ID
            max_iterations: 최대 반복 횟수
            enable_checkpointing: 체크포인팅 활성화
            checkpoint_dir: 체크포인트 디렉토리
            debug: 디버그 모드
            **kwargs: 추가 파라미터

        Yields:
            (node_name, state) 튜플
        """
        # DTO 생성
        request = StateGraphRequest(
            initial_state=initial_state,
            state_schema=state_schema,
            nodes=nodes or {},
            edges=edges or {},
            conditional_edges=conditional_edges or {},
            entry_point=entry_point,
            execution_id=execution_id,
            max_iterations=max_iterations,
            enable_checkpointing=enable_checkpointing,
            checkpoint_dir=checkpoint_dir,
            debug=debug,
            extra_params=kwargs,
        )

        # Service 호출 (에러 처리는 decorator가 담당)
        return self._state_graph_service.stream(request)
