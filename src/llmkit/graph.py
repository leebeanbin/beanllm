"""
Graph System - LangGraph-style Workflow
노드 기반 워크플로우 with 자동 캐싱, 평가, 조건부 분기
"""

import asyncio
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union

from .agent import Agent
from .client import Client
from .output_parsers import BaseOutputParser
from .utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class GraphState:
    """
    그래프 상태

    노드 간 데이터 전달용
    """

    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """값 가져오기"""
        return self.data.get(key, default)

    def set(self, key: str, value: Any):
        """값 설정"""
        self.data[key] = value

    def update(self, updates: Dict[str, Any]):
        """여러 값 업데이트"""
        self.data.update(updates)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any):
        self.data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.data


class NodeCache:
    """
    노드 캐시

    같은 입력에 대해 이전 결과 재사용
    """

    def __init__(self, max_size: int = 1000):
        """
        Args:
            max_size: 최대 캐시 크기
        """
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get_key(self, node_name: str, state: GraphState) -> str:
        """캐시 키 생성"""
        # state를 JSON으로 직렬화하여 해시
        state_json = json.dumps(state.data, sort_keys=True)
        hash_value = hashlib.md5(state_json.encode()).hexdigest()
        return f"{node_name}:{hash_value}"

    def get(self, node_name: str, state: GraphState) -> Optional[Any]:
        """캐시에서 가져오기"""
        key = self.get_key(node_name, state)
        if key in self.cache:
            self.hits += 1
            logger.debug(f"Cache hit for {node_name}")
            return self.cache[key]
        else:
            self.misses += 1
            return None

    def set(self, node_name: str, state: GraphState, result: Any):
        """캐시에 저장"""
        # 캐시 크기 제한
        if len(self.cache) >= self.max_size:
            # 가장 오래된 항목 제거 (간단하게 첫 번째 삭제)
            first_key = next(iter(self.cache))
            del self.cache[first_key]

        key = self.get_key(node_name, state)
        self.cache[key] = result
        logger.debug(f"Cached result for {node_name}")

    def clear(self):
        """캐시 초기화"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
        }


class BaseNode(ABC):
    """
    노드 베이스 클래스
    """

    def __init__(self, name: str, cache: bool = False, description: Optional[str] = None):
        """
        Args:
            name: 노드 이름
            cache: 캐싱 사용 여부
            description: 설명
        """
        self.name = name
        self.cache_enabled = cache
        self.description = description or ""

    @abstractmethod
    async def execute(self, state: GraphState) -> Dict[str, Any]:
        """
        노드 실행

        Args:
            state: 현재 상태

        Returns:
            상태 업데이트 딕셔너리
        """
        pass


class FunctionNode(BaseNode):
    """
    함수 기반 노드

    Example:
        ```python
        async def my_node(state: GraphState) -> Dict[str, Any]:
            result = process(state["input"])
            return {"output": result}

        node = FunctionNode("process", my_node)
        ```
    """

    def __init__(
        self,
        name: str,
        func: Callable[[GraphState], Union[Dict[str, Any], Any]],
        cache: bool = False,
        description: Optional[str] = None,
    ):
        """
        Args:
            name: 노드 이름
            func: 실행 함수 (state -> update_dict)
            cache: 캐싱 여부
            description: 설명
        """
        super().__init__(name, cache, description)
        self.func = func

    async def execute(self, state: GraphState) -> Dict[str, Any]:
        """함수 실행"""
        # 동기/비동기 함수 모두 지원
        if asyncio.iscoroutinefunction(self.func):
            result = await self.func(state)
        else:
            result = self.func(state)

        # Dict가 아니면 {"result": value}로 래핑
        if not isinstance(result, dict):
            result = {"result": result}

        return result


class AgentNode(BaseNode):
    """
    Agent 기반 노드

    Example:
        ```python
        from llmkit import Agent, Tool

        agent = Agent(model="gpt-4o-mini", tools=[...])
        node = AgentNode("researcher", agent, input_key="query", output_key="answer")
        ```
    """

    def __init__(
        self,
        name: str,
        agent: Agent,
        input_key: str = "input",
        output_key: str = "output",
        cache: bool = False,
        description: Optional[str] = None,
    ):
        """
        Args:
            name: 노드 이름
            agent: Agent 인스턴스
            input_key: state에서 가져올 입력 키
            output_key: state에 저장할 출력 키
            cache: 캐싱 여부
            description: 설명
        """
        super().__init__(name, cache, description)
        self.agent = agent
        self.input_key = input_key
        self.output_key = output_key

    async def execute(self, state: GraphState) -> Dict[str, Any]:
        """Agent 실행"""
        input_value = state.get(self.input_key, "")

        # Agent 실행
        result = await self.agent.run(input_value)

        return {
            self.output_key: result.answer,
            f"{self.output_key}_steps": result.total_steps,
            f"{self.output_key}_success": result.success,
        }


class LLMNode(BaseNode):
    """
    LLM 기반 노드

    Example:
        ```python
        from llmkit import Client

        client = Client(model="gpt-4o-mini")
        node = LLMNode(
            "summarizer",
            client,
            template="Summarize: {text}",
            input_keys=["text"],
            output_key="summary"
        )
        ```
    """

    def __init__(
        self,
        name: str,
        client: Client,
        template: str,
        input_keys: List[str],
        output_key: str = "output",
        cache: bool = False,
        parser: Optional[BaseOutputParser] = None,
        description: Optional[str] = None,
    ):
        """
        Args:
            name: 노드 이름
            client: LLM Client
            template: 프롬프트 템플릿
            input_keys: state에서 가져올 입력 키들
            output_key: state에 저장할 출력 키
            cache: 캐싱 여부
            parser: Output Parser (선택)
            description: 설명
        """
        super().__init__(name, cache, description)
        self.client = client
        self.template = template
        self.input_keys = input_keys
        self.output_key = output_key
        self.parser = parser

    async def execute(self, state: GraphState) -> Dict[str, Any]:
        """LLM 실행"""
        # 템플릿 변수 추출
        template_vars = {key: state.get(key, "") for key in self.input_keys}

        # 프롬프트 생성
        prompt = self.template.format(**template_vars)

        # LLM 호출
        response = await self.client.chat([{"role": "user", "content": prompt}])

        # 파싱
        output = response.content
        if self.parser:
            output = self.parser.parse(output)

        return {self.output_key: output}


class GraderNode(BaseNode):
    """
    평가/검증 노드

    출력을 평가하고 점수 부여

    Example:
        ```python
        node = GraderNode(
            "quality_checker",
            client,
            criteria="Is this answer accurate and complete?",
            input_key="answer",
            output_key="grade"
        )
        ```
    """

    def __init__(
        self,
        name: str,
        client: Client,
        criteria: str,
        input_key: str,
        output_key: str = "grade",
        scale: int = 10,
        cache: bool = False,
        description: Optional[str] = None,
    ):
        """
        Args:
            name: 노드 이름
            client: LLM Client
            criteria: 평가 기준
            input_key: 평가할 값의 키
            output_key: 점수 저장 키
            scale: 평가 척도 (1-scale)
            cache: 캐싱 여부
            description: 설명
        """
        super().__init__(name, cache, description)
        self.client = client
        self.criteria = criteria
        self.input_key = input_key
        self.output_key = output_key
        self.scale = scale

    async def execute(self, state: GraphState) -> Dict[str, Any]:
        """평가 실행"""
        value_to_grade = state.get(self.input_key, "")

        # 평가 프롬프트
        prompt = f"""Evaluate the following based on this criteria:
{self.criteria}

Content to evaluate:
{value_to_grade}

Provide a score from 1 to {self.scale}, where 1 is lowest and {self.scale} is highest.
Also provide a brief explanation.

Return in format:
Score: [number]
Explanation: [text]"""

        response = await self.client.chat([{"role": "user", "content": prompt}])

        # 점수 추출
        content = response.content
        score_match = re.search(r"Score:\s*(\d+)", content)
        score = int(score_match.group(1)) if score_match else 0

        # 설명 추출
        explanation_match = re.search(r"Explanation:\s*(.+)", content, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else ""

        return {
            self.output_key: score,
            f"{self.output_key}_explanation": explanation,
            f"{self.output_key}_max": self.scale,
        }


class ConditionalNode(BaseNode):
    """
    조건부 실행 노드

    조건에 따라 다른 노드를 실행합니다.

    Mathematical Foundation:
        조건부 계산 그래프에서 분기 로직 구현
        f(x) = { g₁(x) if condition₁(x)
                 g₂(x) if condition₂(x)
                 ...
                 gₙ(x) otherwise }

    Example:
        ```python
        def is_high_quality(state):
            return state.get("grade", 0) > 7

        node = ConditionalNode(
            "quality_router",
            condition=is_high_quality,
            true_node=approve_node,
            false_node=reject_node
        )
        ```
    """

    def __init__(
        self,
        name: str,
        condition: Callable[[GraphState], bool],
        true_node: Optional[BaseNode] = None,
        false_node: Optional[BaseNode] = None,
        cache: bool = False,
        description: Optional[str] = None,
    ):
        """
        Args:
            name: 노드 이름
            condition: 조건 함수 (state -> bool)
            true_node: 조건이 True일 때 실행할 노드
            false_node: 조건이 False일 때 실행할 노드
            cache: 캐싱 여부
            description: 설명
        """
        super().__init__(name, cache, description)
        self.condition = condition
        self.true_node = true_node
        self.false_node = false_node

    async def execute(self, state: GraphState) -> Dict[str, Any]:
        """조건 평가 및 노드 실행"""
        # 조건 평가
        condition_result = self.condition(state)

        logger.debug(f"Condition result: {condition_result}")

        # 노드 선택
        selected_node = self.true_node if condition_result else self.false_node

        if selected_node is None:
            return {f"{self.name}_condition": condition_result, f"{self.name}_executed": None}

        # 선택된 노드 실행
        result = await selected_node.execute(state)

        # 메타데이터 추가
        result[f"{self.name}_condition"] = condition_result
        result[f"{self.name}_executed"] = selected_node.name

        return result


class LoopNode(BaseNode):
    """
    반복 실행 노드

    종료 조건이 충족될 때까지 자식 노드를 반복 실행합니다.

    Mathematical Foundation:
        재귀적 계산 구조
        x₀ = initial_state
        xₙ₊₁ = f(xₙ)  while not termination_condition(xₙ)

        종료 조건 (Termination Condition):
        - 반드시 유한 시간 내에 True가 되어야 함
        - 정지 문제(Halting Problem)와 관련

    Example:
        ```python
        def should_continue(state):
            return state.get("iterations", 0) < 5

        node = LoopNode(
            "refiner",
            body_node=refine_node,
            termination_condition=lambda s: not should_continue(s),
            max_iterations=10
        )
        ```
    """

    def __init__(
        self,
        name: str,
        body_node: BaseNode,
        termination_condition: Callable[[GraphState], bool],
        max_iterations: int = 10,
        cache: bool = False,
        description: Optional[str] = None,
    ):
        """
        Args:
            name: 노드 이름
            body_node: 반복 실행할 노드
            termination_condition: 종료 조건 (state -> bool, True면 종료)
            max_iterations: 최대 반복 횟수 (무한 루프 방지)
            cache: 캐싱 여부
            description: 설명
        """
        super().__init__(name, cache, description)
        self.body_node = body_node
        self.termination_condition = termination_condition
        self.max_iterations = max_iterations

    async def execute(self, state: GraphState) -> Dict[str, Any]:
        """반복 실행"""
        iterations = 0
        loop_results = []

        # 초기 종료 조건 체크
        while not self.termination_condition(state) and iterations < self.max_iterations:
            logger.debug(f"Loop iteration {iterations + 1}/{self.max_iterations}")

            # Body 노드 실행
            result = await self.body_node.execute(state)
            loop_results.append(result)

            # 상태 업데이트
            state.update(result)

            iterations += 1

        logger.info(f"Loop completed after {iterations} iterations")

        # 최종 결과
        return {
            f"{self.name}_iterations": iterations,
            f"{self.name}_terminated": self.termination_condition(state),
            f"{self.name}_results": loop_results,
        }


class ParallelNode(BaseNode):
    """
    병렬 실행 노드

    여러 노드를 병렬로 실행하고 결과를 합칩니다.

    Mathematical Foundation:
        병렬 계산 모델
        f(x) = (g₁(x), g₂(x), ..., gₙ(x))  executed in parallel

        결과 합성:
        result = aggregate([r₁, r₂, ..., rₙ])

        시간 복잡도:
        T_parallel = max(T₁, T₂, ..., Tₙ)  (이상적인 경우)
        vs T_sequential = T₁ + T₂ + ... + Tₙ

    Example:
        ```python
        node = ParallelNode(
            "multi_analyzer",
            child_nodes=[
                sentiment_node,
                entity_extraction_node,
                summarization_node
            ],
            aggregate_strategy="merge"
        )
        ```
    """

    def __init__(
        self,
        name: str,
        child_nodes: List[BaseNode],
        aggregate_strategy: str = "merge",
        cache: bool = False,
        description: Optional[str] = None,
    ):
        """
        Args:
            name: 노드 이름
            child_nodes: 병렬 실행할 노드들
            aggregate_strategy: 결과 집계 전략
                - "merge": 모든 결과를 하나의 dict로 병합
                - "list": 결과를 리스트로 반환
                - "first": 첫 번째 완료된 결과만 사용
            cache: 캐싱 여부
            description: 설명
        """
        super().__init__(name, cache, description)
        self.child_nodes = child_nodes
        self.aggregate_strategy = aggregate_strategy

    async def execute(self, state: GraphState) -> Dict[str, Any]:
        """병렬 실행"""
        logger.debug(f"Executing {len(self.child_nodes)} nodes in parallel")

        # 모든 노드를 병렬 실행
        tasks = [node.execute(state) for node in self.child_nodes]

        if self.aggregate_strategy == "first":
            # 첫 번째 완료된 것만 사용
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            # 나머지 취소
            for task in pending:
                task.cancel()

            result = list(done)[0].result()
            return {
                **result,
                f"{self.name}_completed": 1,
                f"{self.name}_total": len(self.child_nodes),
            }

        else:
            # 모든 노드 완료 대기
            results = await asyncio.gather(*tasks)

            if self.aggregate_strategy == "list":
                # 리스트로 반환
                return {f"{self.name}_results": results, f"{self.name}_count": len(results)}

            elif self.aggregate_strategy == "merge":
                # 모든 결과를 하나의 dict로 병합
                merged = {}
                for i, result in enumerate(results):
                    # 충돌 방지: 노드 이름을 prefix로 추가
                    node_name = self.child_nodes[i].name
                    for key, value in result.items():
                        merged[f"{node_name}_{key}"] = value

                merged[f"{self.name}_count"] = len(results)
                return merged

            else:
                raise ValueError(f"Unknown aggregate strategy: {self.aggregate_strategy}")


class Graph:
    """
    노드 기반 워크플로우 그래프

    LangGraph 스타일의 간단한 그래프 시스템

    Example:
        ```python
        from llmkit.graph import Graph
        from llmkit import Client, Agent, Tool

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

    def add_node(self, node: BaseNode):
        """노드 추가"""
        self.nodes[node.name] = node
        logger.info(f"Added node: {node.name}")

    def add_function_node(self, name: str, func: Callable, cache: bool = False, **kwargs):
        """함수 노드 추가"""
        node = FunctionNode(name, func, cache=cache, **kwargs)
        self.add_node(node)

    def add_agent_node(
        self,
        name: str,
        agent: Agent,
        input_key: str = "input",
        output_key: str = "output",
        cache: bool = False,
        **kwargs,
    ):
        """Agent 노드 추가"""
        node = AgentNode(name, agent, input_key, output_key, cache=cache, **kwargs)
        self.add_node(node)

    def add_llm_node(
        self,
        name: str,
        client: Client,
        template: str,
        input_keys: List[str],
        output_key: str = "output",
        cache: bool = False,
        parser: Optional[BaseOutputParser] = None,
        **kwargs,
    ):
        """LLM 노드 추가"""
        node = LLMNode(
            name, client, template, input_keys, output_key, cache=cache, parser=parser, **kwargs
        )
        self.add_node(node)

    def add_grader_node(
        self,
        name: str,
        client: Client,
        criteria: str,
        input_key: str,
        output_key: str = "grade",
        scale: int = 10,
        cache: bool = False,
        **kwargs,
    ):
        """Grader 노드 추가"""
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

        Args:
            initial_state: 초기 상태
            verbose: 상세 로그

        Returns:
            최종 상태
        """
        # State 생성
        if isinstance(initial_state, dict):
            state = GraphState(data=initial_state)
        else:
            state = initial_state

        # 시작 노드 결정
        if self.entry_point:
            current_node = self.entry_point
        else:
            # 첫 번째 노드
            current_node = next(iter(self.nodes))

        visited: Set[str] = set()
        max_iterations = 100  # 무한 루프 방지

        for iteration in range(max_iterations):
            if current_node in visited:
                logger.warning(f"Node {current_node} already visited, stopping")
                break

            if current_node not in self.nodes:
                logger.error(f"Node not found: {current_node}")
                break

            visited.add(current_node)

            if verbose:
                logger.info(f"\n{'='*60}")
                logger.info(f"Executing node: {current_node}")
                logger.info(f"{'='*60}")

            # 노드 실행
            node = self.nodes[current_node]

            # 캐시 체크
            if self.cache and node.cache_enabled:
                cached_result = self.cache.get(current_node, state)
                if cached_result is not None:
                    update = cached_result
                    if verbose:
                        logger.info("Using cached result")
                else:
                    update = await node.execute(state)
                    self.cache.set(current_node, state, update)
            else:
                update = await node.execute(state)

            # 상태 업데이트
            state.update(update)

            if verbose:
                logger.info(f"State updated: {list(update.keys())}")

            # 다음 노드 결정
            next_node = None

            # 조건부 엣지 확인
            if current_node in self.conditional_edges:
                condition_func = self.conditional_edges[current_node]
                next_node = condition_func(state)
                if verbose:
                    logger.info(f"Conditional edge -> {next_node}")

            # 일반 엣지 확인
            elif current_node in self.edges:
                edges = self.edges[current_node]
                if edges:
                    next_node = edges[0]  # 첫 번째 엣지
                    if verbose:
                        logger.info(f"Edge -> {next_node}")

            # 다음 노드 없으면 종료
            if not next_node:
                if verbose:
                    logger.info("No next node, finishing")
                break

            current_node = next_node

        # 캐시 통계
        if self.cache and verbose:
            stats = self.cache.get_stats()
            logger.info(f"\nCache stats: {stats}")

        return state

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


# import 누락 추가
import re
