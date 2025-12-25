"""
Graph Nodes - 노드 구현체들
"""

import asyncio
import re
from typing import Any, Callable, Dict, List, Optional, Union

from ...utils.logger import get_logger
from .base_node import BaseNode
from .graph_state import GraphState

logger = get_logger(__name__)


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
        from beanllm import Agent, Tool

        agent = Agent(model="gpt-4o-mini", tools=[...])
        node = AgentNode("researcher", agent, input_key="query", output_key="answer")
        ```
    """

    def __init__(
        self,
        name: str,
        agent: Any,  # Agent (순환 참조 방지)
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
        from beanllm import Client

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
        client: Any,  # Client (순환 참조 방지)
        template: str,
        input_keys: List[str],
        output_key: str = "output",
        cache: bool = False,
        parser: Optional[Any] = None,  # BaseOutputParser
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
        client: Any,  # Client
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
