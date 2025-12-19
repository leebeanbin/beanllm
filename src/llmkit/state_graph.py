"""
StateGraph - LangGraph-style TypedDict State + Checkpointing
타입 안전 상태 관리 및 체크포인팅 지원
"""

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

# Type variables
StateType = TypeVar("StateType", bound=Dict[str, Any])


@dataclass
class GraphConfig:
    """그래프 설정"""

    max_iterations: int = 100  # 무한 루프 방지
    enable_checkpointing: bool = False
    checkpoint_dir: Optional[Path] = None
    debug: bool = False


@dataclass
class NodeExecution:
    """노드 실행 기록"""

    node_name: str
    input_state: Dict[str, Any]
    output_state: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[Exception] = None


@dataclass
class GraphExecution:
    """그래프 실행 기록"""

    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    nodes_executed: List[NodeExecution] = field(default_factory=list)
    final_state: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None


class Checkpoint:
    """상태 체크포인트"""

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        self.checkpoint_dir = checkpoint_dir or Path(".checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)

    def save(self, execution_id: str, state: Dict[str, Any], node_name: str):
        """체크포인트 저장"""
        checkpoint_file = self.checkpoint_dir / f"{execution_id}_{node_name}.json"

        checkpoint_data = {
            "execution_id": execution_id,
            "node_name": node_name,
            "state": state,
            "timestamp": datetime.now().isoformat(),
        }

        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False, default=str)

    def load(self, execution_id: str, node_name: str) -> Optional[Dict[str, Any]]:
        """체크포인트 로드"""
        checkpoint_file = self.checkpoint_dir / f"{execution_id}_{node_name}.json"

        if not checkpoint_file.exists():
            return None

        with open(checkpoint_file, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)

        return checkpoint_data.get("state")

    def list_checkpoints(self, execution_id: str) -> List[str]:
        """체크포인트 목록"""
        pattern = f"{execution_id}_*.json"
        return [p.stem for p in self.checkpoint_dir.glob(pattern)]

    def clear(self, execution_id: Optional[str] = None):
        """체크포인트 삭제"""
        if execution_id:
            pattern = f"{execution_id}_*.json"
        else:
            pattern = "*.json"

        for p in self.checkpoint_dir.glob(pattern):
            p.unlink()


class END:
    """종료 노드 마커"""

    pass


class StateGraph:
    """
    상태 기반 워크플로우 그래프 (LangGraph 스타일)

    TypedDict 기반 타입 안전 상태 + Checkpointing 지원

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

        Example:
            def route(state):
                if state["count"] > 10:
                    return "end"
                return "continue"

            graph.add_conditional_edge(
                "check",
                route,
                {"end": END, "continue": "process"}
            )
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

    def _validate_state(self, state: Dict[str, Any]) -> bool:
        """State 스키마 검증 (TypedDict)"""
        if not self.state_schema:
            return True

        # TypedDict 타입 힌트 가져오기
        try:
            type_hints = get_type_hints(self.state_schema)

            # 필수 필드 체크
            for key, type_hint in type_hints.items():
                if key not in state:
                    # Optional 체크
                    origin = get_origin(type_hint)
                    if origin is Union:
                        args = get_args(type_hint)
                        if type(None) not in args:
                            raise ValueError(f"Required field '{key}' missing in state")
                    else:
                        raise ValueError(f"Required field '{key}' missing in state")

            return True

        except Exception as e:
            if self.config.debug:
                print(f"State validation warning: {e}")
            return True

    def _get_next_node(
        self, current_node: str, state: StateType
    ) -> Optional[Union[str, type[END]]]:
        """다음 노드 결정"""
        # 조건부 엣지 우선
        if current_node in self.conditional_edges:
            condition_func, edge_mapping = self.conditional_edges[current_node]
            result = condition_func(state)

            if edge_mapping:
                return edge_mapping.get(result, END)
            else:
                # 직접 노드 이름 반환
                return result if result in self.nodes else END

        # 고정 엣지
        if current_node in self.edges:
            return self.edges[current_node]

        # 엣지 없으면 종료
        return END

    def invoke(
        self,
        initial_state: StateType,
        execution_id: Optional[str] = None,
        resume_from: Optional[str] = None,
    ) -> StateType:
        """
        그래프 실행

        Args:
            initial_state: 초기 상태
            execution_id: 실행 ID (체크포인팅용)
            resume_from: 재개할 노드 (체크포인트에서 복원)

        Returns:
            최종 상태
        """
        if not self.entry_point:
            raise ValueError("Entry point not set. Call set_entry_point() first.")

        # State 검증
        self._validate_state(initial_state)

        # Execution ID
        if not execution_id:
            execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 실행 기록 시작
        execution = GraphExecution(execution_id=execution_id, start_time=datetime.now())

        # 상태 복사 (원본 보존)
        state = copy.deepcopy(initial_state)

        # 체크포인트에서 복원
        if resume_from and self.checkpoint:
            restored_state = self.checkpoint.load(execution_id, resume_from)
            if restored_state:
                state = restored_state
                current_node = resume_from
            else:
                current_node = self.entry_point
        else:
            current_node = self.entry_point

        # 그래프 실행
        iteration = 0
        try:
            while current_node != END and iteration < self.config.max_iterations:
                if self.config.debug:
                    print(f"[{iteration}] Executing node: {current_node}")

                # 노드 실행
                node_func = self.nodes[current_node]
                node_start = datetime.now()

                try:
                    # 노드 함수 실행
                    input_state = copy.deepcopy(state)
                    state = node_func(state)

                    # 노드 실행 기록
                    node_execution = NodeExecution(
                        node_name=current_node,
                        input_state=input_state,
                        output_state=state,
                        timestamp=node_start,
                    )
                    execution.nodes_executed.append(node_execution)

                    # 체크포인트 저장
                    if self.checkpoint:
                        self.checkpoint.save(execution_id, state, current_node)

                except Exception as e:
                    # 노드 실행 에러
                    node_execution = NodeExecution(
                        node_name=current_node,
                        input_state=state,
                        output_state={},
                        timestamp=node_start,
                        error=e,
                    )
                    execution.nodes_executed.append(node_execution)
                    raise

                # 다음 노드 결정
                current_node = self._get_next_node(current_node, state)
                iteration += 1

            # 무한 루프 체크
            if iteration >= self.config.max_iterations:
                raise RuntimeError(
                    f"Max iterations ({self.config.max_iterations}) reached. "
                    "Possible infinite loop."
                )

            # 실행 완료
            execution.end_time = datetime.now()
            execution.final_state = state
            self.executions.append(execution)

            return state

        except Exception as e:
            execution.end_time = datetime.now()
            execution.error = e
            self.executions.append(execution)
            raise

    def stream(self, initial_state: StateType, execution_id: Optional[str] = None):
        """
        스트리밍 실행 (각 노드 실행 후 상태 반환)

        Yields:
            (node_name, state) 튜플
        """
        if not self.entry_point:
            raise ValueError("Entry point not set")

        self._validate_state(initial_state)

        if not execution_id:
            execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        state = copy.deepcopy(initial_state)
        current_node = self.entry_point

        iteration = 0
        while current_node != END and iteration < self.config.max_iterations:
            # 노드 실행
            node_func = self.nodes[current_node]
            state = node_func(state)

            # 상태 반환
            yield (current_node, copy.deepcopy(state))

            # 체크포인트
            if self.checkpoint:
                self.checkpoint.save(execution_id, state, current_node)

            # 다음 노드
            current_node = self._get_next_node(current_node, state)
            iteration += 1

        if iteration >= self.config.max_iterations:
            raise RuntimeError("Max iterations reached")

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
