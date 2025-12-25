"""
Graph Facade - 기존 Graph API를 위한 Facade
책임: 하위 호환성 유지, 내부적으로는 Handler/Service 사용
SOLID 원칙:
- Facade 패턴: 복잡한 내부 구조를 단순한 인터페이스로
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

from ..domain.graph import BaseNode, GraphState, NodeCache
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Graph:
    """
    노드 기반 워크플로우 그래프 (Facade 패턴)

    기존 API를 유지하면서 내부적으로는 Handler/Service 사용

    Example:
        ```python
        from beanllm.graph import Graph
        from beanllm import Client, Agent, Tool

        # 그래프 생성
        graph = Graph()

        # 노드 추가
        graph.add_llm_node(
            "summarizer",
            client,
            template="Summarize: {text}",
            input_keys=["text"],
            output_key="summary"
        )

        graph.add_grader_node(
            "quality_check",
            client,
            criteria="Is this summary good?",
            input_key="summary"
        )

        # 엣지
        graph.add_edge("summarizer", "quality_check")

        # 실행
        result = await graph.run({"text": "Long text..."})
        print(result["summary"])
        print(result["grade"])
        ```
    """

    def __init__(self, enable_cache: bool = True):
        """
        Args:
            enable_cache: 전역 캐싱 활성화
        """
        self.nodes: Dict[str, BaseNode] = {}
        self.edges: Dict[str, List[str]] = {}  # node_name -> [next_nodes]
        self.conditional_edges: Dict[str, Callable] = {}  # node_name -> condition_func
        self.cache = NodeCache() if enable_cache else None
        self.entry_point: Optional[str] = None

        # Handler/Service 초기화 (의존성 주입)
        self._init_services()

    def _init_services(self) -> None:
        """Service 및 Handler 초기화 (의존성 주입) - DI Container 사용"""
        from ..utils.di_container import get_container

        container = get_container()
        handler_factory = container.handler_factory
        self._graph_handler = handler_factory.create_graph_handler()

    def add_node(self, node: BaseNode):
        """노드 추가"""
        self.nodes[node.name] = node
        logger.info(f"Added node: {node.name}")

    def add_function_node(self, name: str, func: Callable, cache: bool = False, **kwargs):
        """함수 노드 추가"""
        from ..domain.graph import FunctionNode

        node = FunctionNode(name, func, cache=cache, **kwargs)
        self.add_node(node)

    def add_agent_node(
        self,
        name: str,
        agent: Any,  # Agent
        input_key: str = "input",
        output_key: str = "output",
        cache: bool = False,
        **kwargs,
    ):
        """Agent 노드 추가"""
        from ..domain.graph import AgentNode

        node = AgentNode(name, agent, input_key, output_key, cache=cache, **kwargs)
        self.add_node(node)

    def add_llm_node(
        self,
        name: str,
        client: Any,  # Client
        template: str,
        input_keys: List[str],
        output_key: str = "output",
        cache: bool = False,
        parser: Optional[Any] = None,  # BaseOutputParser
        **kwargs,
    ):
        """LLM 노드 추가"""
        from ..domain.graph import LLMNode

        node = LLMNode(
            name, client, template, input_keys, output_key, cache=cache, parser=parser, **kwargs
        )
        self.add_node(node)

    def add_grader_node(
        self,
        name: str,
        client: Any,  # Client
        criteria: str,
        input_key: str,
        output_key: str = "grade",
        scale: int = 10,
        cache: bool = False,
        **kwargs,
    ):
        """Grader 노드 추가"""
        from ..domain.graph import GraderNode

        node = GraderNode(
            name, client, criteria, input_key, output_key, scale, cache=cache, **kwargs
        )
        self.add_node(node)

    def add_edge(self, from_node: str, to_node: str):
        """무조건 엣지 추가"""
        if from_node not in self.edges:
            self.edges[from_node] = []
        self.edges[from_node].append(to_node)
        logger.debug(f"Added edge: {from_node} -> {to_node}")

    def add_conditional_edge(self, from_node: str, condition: Callable[[GraphState], str]):
        """
        조건부 엣지 추가

        Args:
            from_node: 시작 노드
            condition: state를 받아서 다음 노드 이름을 반환하는 함수
        """
        self.conditional_edges[from_node] = condition
        logger.debug(f"Added conditional edge from: {from_node}")

    def set_entry_point(self, node_name: str):
        """시작 노드 설정"""
        self.entry_point = node_name

    async def run(
        self, initial_state: Union[Dict[str, Any], GraphState], verbose: bool = False
    ) -> GraphState:
        """
        그래프 실행

        내부적으로 Handler를 사용하여 처리

        Args:
            initial_state: 초기 상태
            verbose: 상세 로그

        Returns:
            최종 상태
        """
        # initial_state를 dict로 변환
        if isinstance(initial_state, GraphState):
            initial_state_dict = initial_state.data
        else:
            initial_state_dict = initial_state

        # Handler를 통한 처리
        response = await self._graph_handler.handle_run(
            initial_state=initial_state_dict,
            nodes=list(self.nodes.values()),
            edges=self.edges,
            conditional_edges=self.conditional_edges,
            entry_point=self.entry_point,
            enable_cache=self.cache is not None,
            verbose=verbose,
        )

        # GraphResponse를 GraphState로 변환 (기존 API 유지)
        final_state = GraphState(data=response.final_state, metadata=response.metadata)
        return final_state

    def visualize(self) -> str:
        """그래프 시각화 (텍스트)"""
        lines = ["Graph Structure:", ""]

        for node_name, node in self.nodes.items():
            desc = f" - {node.description}" if node.description else ""
            cache_mark = " [cached]" if node.cache_enabled else ""
            lines.append(f"  [{node.__class__.__name__}] {node_name}{cache_mark}{desc}")

            # 엣지
            if node_name in self.edges:
                for next_node in self.edges[node_name]:
                    lines.append(f"    └─> {next_node}")

            if node_name in self.conditional_edges:
                lines.append("    └─> [conditional]")

        return "\n".join(lines)


# 편의 함수
def create_simple_graph(nodes: List[tuple], edges: List[tuple], enable_cache: bool = True) -> Graph:
    """
    간단한 그래프 생성

    Args:
        nodes: [(node_name, node_instance), ...]
        edges: [(from, to), ...]
        enable_cache: 캐싱 활성화

    Returns:
        Graph
    """
    graph = Graph(enable_cache=enable_cache)

    # 노드 추가
    for node_name, node in nodes:
        graph.add_node(node)

    # 엣지 추가
    for from_node, to_node in edges:
        graph.add_edge(from_node, to_node)

    return graph
