"""
StateGraph Facade - 기존 StateGraph API를 위한 Facade
책임: 하위 호환성 유지, 내부적으로는 Handler/Service 사용
SOLID 원칙:
- Facade 패턴: 복잡한 내부 구조를 단순한 인터페이스로
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
)

from ..domain.state_graph import END, Checkpoint, GraphConfig, GraphExecution
from ..utils.logger import get_logger

logger = get_logger(__name__)

StateType = TypeVar("StateType", bound=Dict[str, Any])


class StateGraph:
    """
    상태 기반 워크플로우 그래프 (Facade 패턴)

    기존 API를 유지하면서 내부적으로는 Handler/Service 사용

    Example:
        # State 정의
        class MyState(TypedDict):
            input: str
            output: str
            count: int

        # 그래프 생성
        graph = StateGraph(MyState)

        # 노드 추가
        def process(state: MyState) -> MyState:
            state["output"] = state["input"].upper()
            return state

        graph.add_node("process", process)
        graph.add_edge("process", END)
        graph.set_entry_point("process")

        # 실행
        result = graph.invoke({"input": "hello", "count": 0})
    """

    def __init__(self, state_schema: Optional[type] = None, config: Optional[GraphConfig] = None):
        """
        Args:
            state_schema: State TypedDict 클래스 (옵션)
            config: 그래프 설정
        """
        self.state_schema = state_schema
        self.config = config or GraphConfig()

        self.nodes: Dict[str, Callable] = {}
        self.edges: Dict[str, Union[str, type[END]]] = {}
        self.conditional_edges: Dict[str, tuple] = {}
        self.entry_point: Optional[str] = None

        # Checkpointing
        self.checkpoint: Optional[Checkpoint] = None
        if self.config.enable_checkpointing:
            self.checkpoint = Checkpoint(self.config.checkpoint_dir)

        # 실행 기록
        self.executions: List[GraphExecution] = []

        # Handler/Service 초기화 (의존성 주입)
        self._init_services()

    def _init_services(self) -> None:
        """Service 및 Handler 초기화 (의존성 주입) - DI Container 사용"""
        from ..utils.di_container import get_container

        container = get_container()
        handler_factory = container.handler_factory
        self._state_graph_handler = handler_factory.create_state_graph_handler()

    def add_node(self, name: str, func: Callable[[StateType], StateType]):
        """
        노드 추가

        Args:
            name: 노드 이름
            func: 노드 함수 (state -> state)
        """
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists")

        self.nodes[name] = func

    def add_edge(self, from_node: str, to_node: Union[str, type[END]]):
        """
        엣지 추가 (고정 연결)

        Args:
            from_node: 시작 노드
            to_node: 종료 노드 또는 END
        """
        if from_node not in self.nodes:
            raise ValueError(f"Node '{from_node}' not found")

        if to_node != END and to_node not in self.nodes:
            raise ValueError(f"Node '{to_node}' not found")

        self.edges[from_node] = to_node

    def add_conditional_edge(
        self,
        from_node: str,
        condition_func: Callable[[StateType], str],
        edge_mapping: Optional[Dict[str, Union[str, type[END]]]] = None,
    ):
        """
        조건부 엣지 추가 (동적 라우팅)

        Args:
            from_node: 시작 노드
            condition_func: 조건 함수 (state -> next_node_name)
            edge_mapping: 조건 결과 -> 노드 매핑 (옵션)
        """
        if from_node not in self.nodes:
            raise ValueError(f"Node '{from_node}' not found")

        self.conditional_edges[from_node] = (condition_func, edge_mapping or {})

    def set_entry_point(self, node_name: str):
        """
        진입점 설정

        Args:
            node_name: 시작 노드
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found")

        self.entry_point = node_name

    async def invoke(
        self,
        initial_state: StateType,
        execution_id: Optional[str] = None,
        resume_from: Optional[str] = None,
    ) -> StateType:
        """
        그래프 실행

        내부적으로 Handler를 사용하여 처리

        Args:
            initial_state: 초기 상태
            execution_id: 실행 ID (체크포인팅용)
            resume_from: 재개할 노드 (체크포인트에서 복원)

        Returns:
            최종 상태
        """
        # Handler를 통한 처리
        response = await self._state_graph_handler.handle_invoke(
            initial_state=initial_state,
            state_schema=self.state_schema,
            nodes=self.nodes,
            edges=self.edges,
            conditional_edges=self.conditional_edges,
            entry_point=self.entry_point,
            execution_id=execution_id,
            resume_from=resume_from,
            max_iterations=self.config.max_iterations,
            enable_checkpointing=self.config.enable_checkpointing,
            checkpoint_dir=self.config.checkpoint_dir,
            debug=self.config.debug,
        )

        # GraphResponse를 StateType으로 변환 (기존 API 유지)
        return response.final_state

    def stream(self, initial_state: StateType, execution_id: Optional[str] = None):
        """
        스트리밍 실행 (각 노드 실행 후 상태 반환)

        내부적으로 Handler를 사용하여 처리

        Args:
            initial_state: 초기 상태
            execution_id: 실행 ID

        Yields:
            (node_name, state) 튜플
        """
        # Handler를 통한 처리
        for node_name, state in self._state_graph_handler.handle_stream(
            initial_state=initial_state,
            state_schema=self.state_schema,
            nodes=self.nodes,
            edges=self.edges,
            conditional_edges=self.conditional_edges,
            entry_point=self.entry_point,
            execution_id=execution_id,
            max_iterations=self.config.max_iterations,
            enable_checkpointing=self.config.enable_checkpointing,
            checkpoint_dir=self.config.checkpoint_dir,
            debug=self.config.debug,
        ):
            yield (node_name, state)

    def get_execution_history(self, execution_id: Optional[str] = None) -> List[GraphExecution]:
        """실행 기록 조회"""
        if execution_id:
            return [e for e in self.executions if e.execution_id == execution_id]
        return self.executions

    def visualize(self) -> str:
        """
        그래프 구조 시각화 (텍스트)

        Returns:
            그래프 구조 문자열
        """
        lines = ["Graph Structure:", "=" * 50]

        lines.append(f"\nEntry Point: {self.entry_point}")

        lines.append("\nNodes:")
        for name in self.nodes:
            lines.append(f"  • {name}")

        lines.append("\nEdges:")
        for from_node, to_node in self.edges.items():
            to_str = "END" if to_node == END else to_node
            lines.append(f"  {from_node} → {to_str}")

        lines.append("\nConditional Edges:")
        for from_node, (func, mapping) in self.conditional_edges.items():
            lines.append(f"  {from_node} → (conditional)")
            if mapping:
                for condition, to_node in mapping.items():
                    to_str = "END" if to_node == END else to_node
                    lines.append(f"    - {condition}: {to_str}")

        return "\n".join(lines)


# 편의 함수
def create_state_graph(
    state_schema: Optional[type] = None, enable_checkpointing: bool = False, debug: bool = False
) -> StateGraph:
    """
    StateGraph 생성 (간편 함수)

    Args:
        state_schema: State TypedDict
        enable_checkpointing: 체크포인팅 활성화
        debug: 디버그 모드

    Returns:
        StateGraph

    Example:
        class MyState(TypedDict):
            value: int

        graph = create_state_graph(MyState, debug=True)
    """
    config = GraphConfig(enable_checkpointing=enable_checkpointing, debug=debug)
    return StateGraph(state_schema=state_schema, config=config)
