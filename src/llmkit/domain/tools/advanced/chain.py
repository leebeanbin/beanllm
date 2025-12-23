"""
Tool Composition and Chaining - 도구 체이닝 및 조합
"""

import asyncio
from typing import Any, Callable, List, Optional, Union


class ToolChain:
    """
    도구 체이닝 및 조합

    Mathematical Foundation:
        Function Composition in Category Theory:

        Given tools f: A → B and g: B → C:
        (g ∘ f): A → C
        (g ∘ f)(x) = g(f(x))

        Properties:
        1. Associativity: h ∘ (g ∘ f) = (h ∘ g) ∘ f
        2. Identity: id_B ∘ f = f ∘ id_A = f

        Sequential Execution:
        result = fₙ(fₙ₋₁(...f₂(f₁(input))))

        Parallel Execution:
        results = (f₁(input), f₂(input), ..., fₙ(input))
    """

    def __init__(self, tools: List[Callable]):
        """
        Args:
            tools: 체이닝할 도구 함수 리스트
        """
        self.tools = tools

    def execute(self, initial_input: Any) -> Any:
        """
        순차적 도구 실행 (Composition)

        Args:
            initial_input: 첫 번째 도구의 입력

        Returns:
            마지막 도구의 출력

        Example:
            >>> chain = ToolChain([str.lower, str.strip, str.title])
            >>> chain.execute("  HELLO WORLD  ")
            'Hello World'
        """
        result = initial_input
        for tool in self.tools:
            result = tool(result)
        return result

    async def execute_async(self, initial_input: Any) -> Any:
        """비동기 순차 실행"""
        result = initial_input
        for tool in self.tools:
            if asyncio.iscoroutinefunction(tool):
                result = await tool(result)
            else:
                result = tool(result)
        return result

    @staticmethod
    async def execute_parallel(
        tools: List[Callable], inputs: Union[Any, List[Any]], aggregate: Optional[Callable] = None
    ) -> Union[List[Any], Any]:
        """
        병렬 도구 실행

        Args:
            tools: 실행할 도구 리스트
            inputs: 각 도구의 입력 (단일 값이면 모든 도구에 동일하게 적용)
            aggregate: 결과 집계 함수 (선택)

        Returns:
            각 도구의 결과 리스트 (aggregate가 있으면 집계된 결과)

        Example:
            >>> async def f1(x): return x + 1
            >>> async def f2(x): return x * 2
            >>> results = await ToolChain.execute_parallel([f1, f2], 5)
            >>> results
            [6, 10]
        """
        # Prepare inputs
        if not isinstance(inputs, list):
            inputs = [inputs] * len(tools)

        # Execute in parallel
        tasks = []
        for tool, input_val in zip(tools, inputs):
            if asyncio.iscoroutinefunction(tool):
                tasks.append(tool(input_val))
            else:
                tasks.append(asyncio.to_thread(tool, input_val))

        results = await asyncio.gather(*tasks)

        # Aggregate if needed
        if aggregate:
            return aggregate(results)

        return list(results)
