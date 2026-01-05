"""
LangGraph Workflow - beanLLM 스타일 Workflow Builder

LangGraph의 StateGraph를 beanLLM 인터페이스로 제공합니다.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from .bridge import LangGraphBridge

try:
    from beanllm.utils.logger import get_logger
except ImportError:

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class LangGraphWorkflow:
    """
    LangGraph Workflow (beanLLM 인터페이스)

    LangGraph의 StateGraph를 beanLLM 스타일로 제공합니다.

    Features:
    - State Machine 기반 워크플로우
    - Conditional Edges (조건부 분기)
    - Human-in-the-loop
    - Streaming

    Example:
        ```python
        from beanllm.integrations.langgraph import WorkflowBuilder
        from beanllm.domain.state_graph import GraphState

        # State 정의
        class AgentState(GraphState):
            query: str
            documents: list
            answer: str

        # Workflow 생성
        workflow = (
            WorkflowBuilder(AgentState)
            .add_node("retrieve", retrieve_fn)
            .add_node("generate", generate_fn)
            .add_edge("retrieve", "generate")
            .set_entry_point("retrieve")
            .set_finish_point("generate")
            .build()
        )

        # 실행
        result = workflow.run({"query": "What is AI?"})
        print(result["answer"])
        ```
    """

    def __init__(
        self,
        langgraph_app: Any,
        **kwargs,
    ):
        """
        Args:
            langgraph_app: LangGraph CompiledGraph
            **kwargs: 추가 파라미터
        """
        self.app = langgraph_app
        self.kwargs = kwargs

    def run(
        self,
        initial_state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        워크플로우 실행

        Args:
            initial_state: 초기 상태
            config: 실행 설정 (checkpointer, etc.)

        Returns:
            최종 상태
        """
        # LangGraph 실행
        result = self.app.invoke(initial_state, config=config)

        logger.info(f"LangGraph workflow executed: initial_keys={list(initial_state.keys())}")

        return result

    def stream(
        self,
        initial_state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        워크플로우 스트리밍 실행

        Args:
            initial_state: 초기 상태
            config: 실행 설정

        Yields:
            중간 상태
        """
        # LangGraph 스트리밍
        for event in self.app.stream(initial_state, config=config):
            yield event

    def __repr__(self) -> str:
        return f"LangGraphWorkflow(app={type(self.app).__name__})"


class WorkflowBuilder:
    """
    Workflow Builder (Fluent Interface)

    LangGraph StateGraph를 Fluent Interface로 구축합니다.

    Example:
        ```python
        from beanllm.integrations.langgraph import WorkflowBuilder

        workflow = (
            WorkflowBuilder(StateClass)
            .add_node("node1", fn1)
            .add_node("node2", fn2)
            .add_edge("node1", "node2")
            .add_conditional_edges(
                "node2",
                condition_fn,
                {"continue": "node1", "end": END}
            )
            .set_entry_point("node1")
            .build()
        )
        ```
    """

    def __init__(
        self,
        state_class: type,
        **kwargs,
    ):
        """
        Args:
            state_class: State 클래스 (beanLLM GraphState 또는 TypedDict)
            **kwargs: 추가 파라미터
        """
        try:
            from langgraph.graph import END, StateGraph
        except ImportError:
            raise ImportError(
                "langgraph is required for WorkflowBuilder. "
                "Install it with: pip install langgraph"
            )

        self.state_class = state_class
        self.kwargs = kwargs
        self.END = END

        # LangGraph StateGraph 생성
        self.graph = StateGraph(state_class)

        # 진입점 및 종료점
        self.entry_point = None
        self.finish_point = None

        logger.info(f"WorkflowBuilder initialized with state: {state_class.__name__}")

    def add_node(
        self,
        name: str,
        function: Callable[[Dict], Dict],
    ) -> "WorkflowBuilder":
        """
        노드 추가

        Args:
            name: 노드 이름
            function: 노드 함수 (state -> state)

        Returns:
            self (Fluent Interface)
        """
        # beanLLM 노드 함수 래핑
        bridge = LangGraphBridge()
        wrapped_fn = bridge.wrap_node_function(function)

        # LangGraph 노드 추가
        self.graph.add_node(name, wrapped_fn)

        logger.info(f"Added node: {name}")

        return self

    def add_edge(
        self,
        from_node: str,
        to_node: str,
    ) -> "WorkflowBuilder":
        """
        간선 추가

        Args:
            from_node: 시작 노드
            to_node: 종료 노드

        Returns:
            self (Fluent Interface)
        """
        self.graph.add_edge(from_node, to_node)

        logger.info(f"Added edge: {from_node} -> {to_node}")

        return self

    def add_conditional_edges(
        self,
        from_node: str,
        condition: Callable[[Dict], str],
        edge_map: Dict[str, str],
    ) -> "WorkflowBuilder":
        """
        조건부 간선 추가

        Args:
            from_node: 시작 노드
            condition: 조건 함수 (state -> next_node_name)
            edge_map: 조건 값 -> 다음 노드 매핑

        Returns:
            self (Fluent Interface)

        Example:
            ```python
            .add_conditional_edges(
                "decide",
                lambda state: "continue" if state["score"] > 0.5 else "end",
                {"continue": "process", "end": END}
            )
            ```
        """
        # beanLLM 조건 함수 래핑
        bridge = LangGraphBridge()
        wrapped_condition = bridge.wrap_conditional_edge(condition)

        # LangGraph 조건부 간선 추가
        self.graph.add_conditional_edges(from_node, wrapped_condition, edge_map)

        logger.info(f"Added conditional edges from: {from_node}")

        return self

    def set_entry_point(self, node_name: str) -> "WorkflowBuilder":
        """
        진입점 설정

        Args:
            node_name: 진입점 노드 이름

        Returns:
            self (Fluent Interface)
        """
        self.entry_point = node_name
        self.graph.set_entry_point(node_name)

        logger.info(f"Set entry point: {node_name}")

        return self

    def set_finish_point(self, node_name: str) -> "WorkflowBuilder":
        """
        종료점 설정 (편의 함수)

        Args:
            node_name: 종료 전 마지막 노드

        Returns:
            self (Fluent Interface)
        """
        self.finish_point = node_name
        self.graph.add_edge(node_name, self.END)

        logger.info(f"Set finish point: {node_name} -> END")

        return self

    def build(self) -> LangGraphWorkflow:
        """
        Workflow 빌드

        Returns:
            LangGraphWorkflow 인스턴스
        """
        # LangGraph 컴파일
        app = self.graph.compile()

        logger.info("LangGraph workflow built and compiled")

        return LangGraphWorkflow(app, **self.kwargs)


def create_workflow(
    state_class: type,
    nodes: Dict[str, Callable],
    edges: List[tuple],
    entry_point: str,
    **kwargs,
) -> LangGraphWorkflow:
    """
    Workflow 생성 (편의 함수)

    Args:
        state_class: State 클래스
        nodes: {노드 이름: 노드 함수} 딕셔너리
        edges: [(from_node, to_node), ...] 간선 리스트
        entry_point: 진입점 노드
        **kwargs: 추가 파라미터

    Returns:
        LangGraphWorkflow 인스턴스

    Example:
        ```python
        from beanllm.integrations.langgraph import create_workflow
        from langgraph.graph import END

        workflow = create_workflow(
            state_class=MyState,
            nodes={
                "retrieve": retrieve_fn,
                "generate": generate_fn,
            },
            edges=[
                ("retrieve", "generate"),
                ("generate", END),
            ],
            entry_point="retrieve"
        )

        result = workflow.run({"query": "..."})
        ```
    """
    builder = WorkflowBuilder(state_class, **kwargs)

    # 노드 추가
    for name, fn in nodes.items():
        builder.add_node(name, fn)

    # 간선 추가
    for from_node, to_node in edges:
        builder.add_edge(from_node, to_node)

    # 진입점 설정
    builder.set_entry_point(entry_point)

    # 빌드
    return builder.build()
