"""
StateGraphServiceImpl - StateGraph 서비스 구현체
SOLID 원칙:
- SRP: StateGraph 비즈니스 로직만 담당
- DIP: 인터페이스에 의존 (의존성 주입)
"""

from __future__ import annotations

import copy
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    Optional,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from ...domain.graph.graph_state import GraphState
from ...domain.state_graph import END, Checkpoint, GraphExecution, NodeExecution
from ...dto.request.state_graph_request import StateGraphRequest
from ...dto.response.state_graph_response import StateGraphResponse
from ...utils.logger import get_logger
from ..state_graph_service import IStateGraphService

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

StateType = Dict[str, Any]


class StateGraphServiceImpl(IStateGraphService):
    """
    StateGraph 서비스 구현체

    책임:
    - StateGraph 비즈니스 로직만
    - 검증 없음 (Handler에서 처리)
    - 에러 처리 없음 (Handler에서 처리)

    SOLID:
    - SRP: StateGraph 비즈니스 로직만
    - DIP: 인터페이스에 의존 (의존성 주입)
    """

    def __init__(self) -> None:
        """의존성 주입을 통한 생성자"""
        pass

    async def invoke(self, request: StateGraphRequest) -> StateGraphResponse:
        """
        StateGraph 실행 (기존 state_graph.py의 StateGraph.invoke() 정확히 마이그레이션)

        Args:
            request: StateGraph 요청 DTO

        Returns:
            StateGraphResponse: StateGraph 응답 DTO
        """
        if not request.entry_point:
            raise ValueError("Entry point not set. Call set_entry_point() first.")

        # State 검증 (기존과 동일)
        self._validate_state(request.initial_state, request.state_schema, request.debug)

        # Execution ID (기존과 동일)
        if not request.execution_id:
            execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            execution_id = request.execution_id

        # 실행 기록 시작
        execution = GraphExecution(execution_id=execution_id, start_time=datetime.now())

        # 상태 복사 (원본 보존) - 최적화: GraphState.copy() 사용
        if isinstance(request.initial_state, GraphState):
            state = request.initial_state.copy()  # 얕은 복사 (GraphState 메서드 사용)
        elif isinstance(request.initial_state, dict):
            state = dict(request.initial_state)  # Dict는 얕은 복사
        else:
            state = copy.deepcopy(request.initial_state)  # 기타 타입은 깊은 복사

        # Checkpoint 생성 (기존과 동일)
        checkpoint: Optional[Checkpoint] = None
        if request.enable_checkpointing:
            checkpoint = Checkpoint(request.checkpoint_dir)

        # 체크포인트에서 복원 (기존과 동일)
        if request.resume_from and checkpoint:
            restored_state = checkpoint.load(execution_id, request.resume_from)
            if restored_state:
                state = restored_state
                current_node = request.resume_from
            else:
                current_node = request.entry_point
        else:
            current_node = request.entry_point

        # 그래프 실행 (기존 state_graph.py의 StateGraph.invoke() 로직 정확히 마이그레이션)
        iteration = 0
        try:
            while current_node != END and iteration < request.max_iterations:
                if request.debug:
                    logger.debug(f"[{iteration}] Executing node: {current_node}")

                # 노드 실행
                node_func = request.nodes[current_node]
                node_start = datetime.now()

                try:
                    # 노드 함수 실행 - 최적화: 실행 기록용으로만 복사
                    if isinstance(state, GraphState):
                        input_state = state.copy()  # GraphState.copy() 사용
                    elif isinstance(state, dict):
                        input_state = dict(state)  # Dict는 얕은 복사
                    else:
                        input_state = copy.deepcopy(state)  # 기타 타입은 깊은 복사
                    state = node_func(state)

                    # 노드 실행 기록 (기존과 동일)
                    node_execution = NodeExecution(
                        node_name=current_node,
                        input_state=input_state,
                        output_state=state,
                        timestamp=node_start,
                    )
                    execution.nodes_executed.append(node_execution)

                    # 체크포인트 저장 (기존과 동일)
                    if checkpoint:
                        checkpoint.save(execution_id, state, current_node)

                except Exception as e:
                    # 노드 실행 에러 (기존과 동일)
                    node_execution = NodeExecution(
                        node_name=current_node,
                        input_state=state,
                        output_state={},
                        timestamp=node_start,
                        error=e,
                    )
                    execution.nodes_executed.append(node_execution)
                    raise

                # 다음 노드 결정 (기존과 동일)
                current_node = self._get_next_node(
                    current_node,
                    state,
                    request.edges or {},
                    request.conditional_edges or {},
                    request.nodes or {},
                )
                iteration += 1

            # 무한 루프 체크 (기존과 동일)
            if iteration >= request.max_iterations:
                raise RuntimeError(
                    f"Max iterations ({request.max_iterations}) reached. Possible infinite loop."
                )

            # 실행 완료 (기존과 동일)
            execution.end_time = datetime.now()
            execution.final_state = state

            # 결과 반환
            return StateGraphResponse(
                final_state=state,
                execution_id=execution_id,
                nodes_executed=[ne.node_name for ne in execution.nodes_executed],
                iterations=iteration,
                metadata={"execution": execution},
            )

        except Exception as e:
            execution.end_time = datetime.now()
            execution.error = e
            raise

    def stream(self, request: StateGraphRequest) -> Iterator[tuple[str, Dict[str, Any]]]:
        """
        StateGraph 스트리밍 실행 (기존 state_graph.py의 StateGraph.stream() 정확히 마이그레이션)

        Args:
            request: StateGraph 요청 DTO

        Yields:
            (node_name, state) 튜플
        """
        if not request.entry_point:
            raise ValueError("Entry point not set")

        self._validate_state(request.initial_state, request.state_schema, request.debug)

        if not request.execution_id:
            execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            execution_id = request.execution_id

        # 상태 복사 - 최적화: GraphState.copy() 사용
        if isinstance(request.initial_state, GraphState):
            state = request.initial_state.copy()  # 얕은 복사 (GraphState 메서드 사용)
        elif isinstance(request.initial_state, dict):
            state = dict(request.initial_state)  # Dict는 얕은 복사
        else:
            state = copy.deepcopy(request.initial_state)  # 기타 타입은 깊은 복사

        current_node = request.entry_point

        checkpoint: Optional[Checkpoint] = None
        if request.enable_checkpointing:
            checkpoint = Checkpoint(request.checkpoint_dir)

        iteration = 0
        while current_node != END and iteration < request.max_iterations:
            # 노드 실행
            node_func = request.nodes[current_node]
            state = node_func(state)

            # 상태 반환 - 최적화: GraphState.copy() 사용
            if isinstance(state, GraphState):
                state_copy = state.copy()  # GraphState.copy() 사용
            elif isinstance(state, dict):
                state_copy = dict(state)  # Dict는 얕은 복사
            else:
                state_copy = copy.deepcopy(state)  # 기타 타입은 깊은 복사

            yield (current_node, state_copy)

            # 체크포인트 (기존과 동일)
            if checkpoint:
                checkpoint.save(execution_id, state, current_node)

            # 다음 노드 (기존과 동일)
            current_node = self._get_next_node(
                current_node,
                state,
                request.edges or {},
                request.conditional_edges or {},
                request.nodes or {},
            )
            iteration += 1

        if iteration >= request.max_iterations:
            raise RuntimeError("Max iterations reached")

    def _validate_state(
        self, state: Dict[str, Any], state_schema: Optional[Type], debug: bool
    ) -> bool:
        """State 스키마 검증 (TypedDict) - 기존 state_graph.py의 _validate_state() 정확히 마이그레이션"""
        if not state_schema:
            return True

        # TypedDict 타입 힌트 가져오기 (기존과 동일)
        try:
            type_hints = get_type_hints(state_schema)

            # 필수 필드 체크 (기존과 동일)
            for key, type_hint in type_hints.items():
                if key not in state:
                    # Optional 체크 (기존과 동일)
                    origin = get_origin(type_hint)
                    if origin is Union:
                        args = get_args(type_hint)
                        if type(None) not in args:
                            raise ValueError(f"Required field '{key}' missing in state")
                    else:
                        raise ValueError(f"Required field '{key}' missing in state")

            return True

        except Exception as e:
            if debug:
                logger.debug(f"State validation warning: {e}")
            return True

    def _get_next_node(
        self,
        current_node: str,
        state: Dict[str, Any],
        edges: Dict[str, Union[str, Type[END]]],
        conditional_edges: Dict[str, tuple],
        nodes: Dict[str, Callable],
    ) -> Optional[Union[str, Type[END]]]:
        """다음 노드 결정 - 기존 state_graph.py의 _get_next_node() 정확히 마이그레이션"""
        # 조건부 엣지 우선 (기존과 동일)
        if current_node in conditional_edges:
            condition_func, edge_mapping = conditional_edges[current_node]
            result = condition_func(state)

            if edge_mapping:
                return edge_mapping.get(result, END)
            else:
                # 직접 노드 이름 반환 (기존과 동일)
                return result if result in nodes else END

        # 고정 엣지 (기존과 동일)
        if current_node in edges:
            return edges[current_node]

        # 엣지 없으면 종료 (기존과 동일)
        return END
