"""
Chain Builder - Fluent API for LLM Workflows

참고: LangChain의 체인 개념에서 영감을 받았습니다.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .client import Client
from .memory import BaseMemory, BufferMemory
from .tools import Tool
from .utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ChainResult:
    """체인 실행 결과"""

    output: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


class Chain:
    """
    기본 체인

    Example:
        ```python
        from llmkit import Client, Chain

        client = Client(model="gpt-4o-mini")

        # 간단한 체인
        chain = Chain(client)
        result = await chain.run("파이썬이란?")
        print(result.output)
        ```
    """

    def __init__(self, client: Client, memory: Optional[BaseMemory] = None, verbose: bool = False):
        """
        Args:
            client: LLM Client
            memory: 메모리 (없으면 BufferMemory 사용)
            verbose: 상세 로그
        """
        self.client = client
        self.memory = memory or BufferMemory()
        self.verbose = verbose

    async def run(self, user_input: str, **kwargs) -> ChainResult:
        """
        체인 실행

        Args:
            user_input: 사용자 입력
            **kwargs: 추가 파라미터

        Returns:
            ChainResult: 실행 결과
        """
        try:
            # 메모리에 사용자 메시지 추가
            self.memory.add_message("user", user_input)

            # LLM 호출
            messages = self.memory.get_dict_messages()
            response = await self.client.chat(messages, **kwargs)

            # 메모리에 응답 추가
            self.memory.add_message("assistant", response.content)

            return ChainResult(
                output=response.content,
                steps=[{"type": "llm", "input": user_input, "output": response.content}],
                success=True,
            )

        except Exception as e:
            logger.error(f"Chain error: {e}")
            return ChainResult(output="", success=False, error=str(e))


class PromptChain:
    """
    프롬프트 템플릿 체인

    Example:
        ```python
        from llmkit import Client
        from llmkit.chain import PromptChain

        client = Client(model="gpt-4o-mini")

        # 템플릿 정의
        template = \"\"\"
        You are a {role}.
        Answer the following question: {question}
        \"\"\"

        chain = PromptChain(client, template)
        result = await chain.run(role="Python expert", question="What is async/await?")
        print(result.output)
        ```
    """

    def __init__(self, client: Client, template: str, memory: Optional[BaseMemory] = None):
        """
        Args:
            client: LLM Client
            template: 프롬프트 템플릿
            memory: 메모리
        """
        self.client = client
        self.template = template
        self.memory = memory

    async def run(self, **kwargs) -> ChainResult:
        """
        체인 실행

        Args:
            **kwargs: 템플릿 변수

        Returns:
            ChainResult: 실행 결과
        """
        try:
            # 템플릿 렌더링
            prompt = self.template.format(**kwargs)

            # 메모리 사용
            messages = []
            if self.memory:
                messages = self.memory.get_dict_messages()

            messages.append({"role": "user", "content": prompt})

            # LLM 호출
            response = await self.client.chat(messages)

            # 메모리 업데이트
            if self.memory:
                self.memory.add_message("user", prompt)
                self.memory.add_message("assistant", response.content)

            return ChainResult(
                output=response.content,
                steps=[{"type": "prompt", "template": self.template, "vars": kwargs}],
                success=True,
            )

        except Exception as e:
            logger.error(f"PromptChain error: {e}")
            return ChainResult(output="", success=False, error=str(e))


class SequentialChain:
    """
    순차 실행 체인

    여러 체인을 순차적으로 실행

    Example:
        ```python
        from llmkit import Client
        from llmkit.chain import SequentialChain, PromptChain

        client = Client(model="gpt-4o-mini")

        # 체인 1: 주제 생성
        chain1 = PromptChain(client, "Generate 3 blog post topics about {topic}")

        # 체인 2: 선택 및 작성
        chain2 = PromptChain(client, "Choose the best topic and write an outline")

        # 순차 실행
        seq_chain = SequentialChain([chain1, chain2])
        result = await seq_chain.run(topic="AI")
        ```
    """

    def __init__(self, chains: List[Union[Chain, PromptChain]]):
        """
        Args:
            chains: 체인 목록
        """
        self.chains = chains

    async def run(self, **kwargs) -> ChainResult:
        """
        순차 실행

        Args:
            **kwargs: 초기 입력

        Returns:
            ChainResult: 최종 결과
        """
        steps = []
        current_output = None

        try:
            for i, chain in enumerate(self.chains):
                logger.debug(f"Executing chain {i + 1}/{len(self.chains)}")

                # 첫 번째 체인은 kwargs 사용, 이후는 이전 출력 사용
                if i == 0:
                    result = await chain.run(**kwargs)
                else:
                    # 이전 출력을 다음 체인의 입력으로
                    if isinstance(chain, PromptChain):
                        result = await chain.run(input=current_output)
                    else:
                        result = await chain.run(current_output)

                if not result.success:
                    return result

                current_output = result.output
                steps.extend(result.steps)

            return ChainResult(output=current_output, steps=steps, success=True)

        except Exception as e:
            logger.error(f"SequentialChain error: {e}")
            return ChainResult(output="", steps=steps, success=False, error=str(e))


class ParallelChain:
    """
    병렬 실행 체인

    여러 체인을 동시에 실행

    Example:
        ```python
        from llmkit import Client
        from llmkit.chain import ParallelChain, PromptChain

        client = Client(model="gpt-4o-mini")

        # 여러 관점에서 동시 분석
        chains = [
            PromptChain(client, "Analyze {topic} from technical perspective"),
            PromptChain(client, "Analyze {topic} from business perspective"),
            PromptChain(client, "Analyze {topic} from user perspective"),
        ]

        parallel = ParallelChain(chains)
        result = await parallel.run(topic="AI chatbots")

        # 모든 결과가 리스트로 반환
        for i, output in enumerate(result.outputs):
            print(f"Chain {i + 1}: {output}")
        ```
    """

    def __init__(self, chains: List[Union[Chain, PromptChain]]):
        """
        Args:
            chains: 체인 목록
        """
        self.chains = chains

    async def run(self, **kwargs) -> ChainResult:
        """
        병렬 실행

        Args:
            **kwargs: 입력

        Returns:
            ChainResult: 결합된 결과
        """
        try:
            # 모든 체인을 동시에 실행
            tasks = [chain.run(**kwargs) for chain in self.chains]
            results = await asyncio.gather(*tasks)

            # 결과 결합
            outputs = [r.output for r in results]
            all_steps = []
            for r in results:
                all_steps.extend(r.steps)

            # 성공 여부 확인
            success = all(r.success for r in results)
            errors = [r.error for r in results if r.error]

            return ChainResult(
                output="\n\n---\n\n".join(outputs),
                steps=all_steps,
                metadata={"outputs": outputs, "count": len(outputs)},
                success=success,
                error="; ".join(errors) if errors else None,
            )

        except Exception as e:
            logger.error(f"ParallelChain error: {e}")
            return ChainResult(output="", success=False, error=str(e))


class ChainBuilder:
    """
    체인 빌더 (Fluent API)

    Example:
        ```python
        from llmkit import Client
        from llmkit.chain import ChainBuilder

        client = Client(model="gpt-4o-mini")

        # Fluent API로 체인 구성
        result = await (
            ChainBuilder(client)
            .with_memory("window", window_size=5)
            .with_template("Translate to {language}: {text}")
            .run(language="Korean", text="Hello, World!")
        )

        print(result.output)
        ```
    """

    def __init__(self, client: Client):
        """
        Args:
            client: LLM Client
        """
        self.client = client
        self._memory: Optional[BaseMemory] = None
        self._template: Optional[str] = None
        self._tools: List[Tool] = []
        self._verbose: bool = False

    def with_memory(self, memory_type: str = "buffer", **kwargs) -> "ChainBuilder":
        """
        메모리 설정

        Args:
            memory_type: 메모리 타입
            **kwargs: 메모리 파라미터

        Returns:
            ChainBuilder: self (체이닝)
        """
        from .memory import create_memory

        self._memory = create_memory(memory_type, **kwargs)
        return self

    def with_template(self, template: str) -> "ChainBuilder":
        """
        프롬프트 템플릿 설정

        Args:
            template: 템플릿 문자열

        Returns:
            ChainBuilder: self
        """
        self._template = template
        return self

    def with_tools(self, tools: List[Tool]) -> "ChainBuilder":
        """
        도구 추가

        Args:
            tools: 도구 목록

        Returns:
            ChainBuilder: self
        """
        self._tools = tools
        return self

    def verbose(self, enabled: bool = True) -> "ChainBuilder":
        """
        상세 로그 활성화

        Args:
            enabled: 활성화 여부

        Returns:
            ChainBuilder: self
        """
        self._verbose = enabled
        return self

    async def run(self, **kwargs) -> ChainResult:
        """
        체인 실행

        Args:
            **kwargs: 입력 파라미터

        Returns:
            ChainResult: 실행 결과
        """
        # 적절한 체인 타입 선택
        if self._template:
            chain = PromptChain(self.client, self._template, memory=self._memory)
            return await chain.run(**kwargs)
        else:
            chain = Chain(self.client, memory=self._memory, verbose=self._verbose)
            # kwargs에서 user_input 추출
            user_input = kwargs.pop("input", None) or kwargs.pop("question", "")
            return await chain.run(user_input, **kwargs)

    def build(self) -> Chain:
        """
        체인 빌드

        Returns:
            Chain: 구성된 체인
        """
        if self._template:
            return PromptChain(self.client, self._template, memory=self._memory)
        else:
            return Chain(self.client, memory=self._memory, verbose=self._verbose)


# 편의 함수
def create_chain(client: Client, chain_type: str = "basic", **kwargs) -> Union[Chain, PromptChain]:
    """
    체인 생성 팩토리

    Args:
        client: LLM Client
        chain_type: 체인 타입 (basic, prompt)
        **kwargs: 체인 파라미터

    Returns:
        Chain: 생성된 체인

    Example:
        ```python
        from llmkit import Client, create_chain

        client = Client(model="gpt-4o-mini")

        # 기본 체인
        chain = create_chain(client, "basic")

        # 프롬프트 체인
        chain = create_chain(
            client,
            "prompt",
            template="Explain {topic} in simple terms"
        )
        ```
    """
    if chain_type == "basic":
        return Chain(client, **kwargs)
    elif chain_type == "prompt":
        template = kwargs.pop("template", "")
        return PromptChain(client, template, **kwargs)
    else:
        raise ValueError(f"Unknown chain type: {chain_type}")
