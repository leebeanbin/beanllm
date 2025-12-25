"""
Chain Facade - 기존 Chain API를 위한 Facade
책임: 하위 호환성 유지, 내부적으로는 Handler/Service 사용
SOLID 원칙:
- Facade 패턴: 복잡한 내부 구조를 단순한 인터페이스로
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..domain.memory import BaseMemory, BufferMemory, create_memory
from ..domain.tools import Tool
from ..utils.logger import get_logger
from .client_facade import Client

logger = get_logger(__name__)


@dataclass
class ChainResult:
    """체인 실행 결과 (기존 API 유지)"""

    output: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


class Chain:
    """
    기본 체인 (Facade 패턴)

    기존 API를 유지하면서 내부적으로는 Handler/Service 사용

    Example:
        ```python
        from beanllm import Client, Chain

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

        # Handler/Service 초기화 (의존성 주입)
        self._init_services()

    def _init_services(self) -> None:
        """Service 및 Handler 초기화 (의존성 주입) - DI Container 사용"""
        from ..utils.di_container import get_container

        container = get_container()
        handler_factory = container.handler_factory

        # ChainHandler 생성
        self._chain_handler = handler_factory.create_chain_handler()

    async def run(self, user_input: str, **kwargs) -> ChainResult:
        """
        체인 실행

        내부적으로 Handler를 사용하여 처리

        Args:
            user_input: 사용자 입력
            **kwargs: 추가 파라미터

        Returns:
            ChainResult: 실행 결과
        """
        # Handler를 통한 처리
        response = await self._chain_handler.handle_run(
            chain_type="basic",
            user_input=user_input,
            model=self.client.model,
            memory_type="buffer" if isinstance(self.memory, BufferMemory) else None,
            verbose=self.verbose,
            **kwargs,
        )

        # ChainResponse를 ChainResult로 변환 (기존 API 유지)
        return ChainResult(
            output=response.output,
            steps=response.steps,
            metadata=response.metadata,
            success=response.success,
            error=response.error,
        )


class PromptChain:
    """
    프롬프트 템플릿 체인 (Facade 패턴)

    기존 API를 유지하면서 내부적으로는 Handler/Service 사용
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

        # Handler/Service 초기화
        self._init_services()

    def _init_services(self) -> None:
        """Service 및 Handler 초기화"""
        from ..utils.di_container import get_container

        container = get_container()
        handler_factory = container.handler_factory
        self._chain_handler = handler_factory.create_chain_handler()

    async def run(self, **kwargs) -> ChainResult:
        """
        체인 실행

        내부적으로 Handler를 사용하여 처리

        Args:
            **kwargs: 템플릿 변수

        Returns:
            ChainResult: 실행 결과
        """
        # Handler를 통한 처리
        response = await self._chain_handler.handle_run(
            chain_type="prompt",
            template=self.template,
            template_vars=kwargs,
            model=self.client.model,
            memory_type="buffer" if self.memory and isinstance(self.memory, BufferMemory) else None,
        )

        # ChainResponse를 ChainResult로 변환
        return ChainResult(
            output=response.output,
            steps=response.steps,
            metadata=response.metadata,
            success=response.success,
            error=response.error,
        )


class SequentialChain:
    """
    순차 실행 체인 (Facade 패턴)

    기존 API를 유지하면서 내부적으로는 Handler/Service 사용
    """

    def __init__(self, chains: List[Union[Chain, PromptChain]]):
        """
        Args:
            chains: 체인 목록
        """
        self.chains = chains

        # Handler/Service 초기화
        self._init_services()

    def _init_services(self) -> None:
        """Service 및 Handler 초기화 - DI Container 사용"""
        from ..utils.di_container import get_container

        container = get_container()
        handler_factory = container.handler_factory
        self._chain_handler = handler_factory.create_chain_handler()

    async def run(self, **kwargs) -> ChainResult:
        """
        순차 실행

        내부적으로 각 Chain을 직접 실행 (기존 chain.py의 SequentialChain.run() 정확히 마이그레이션)

        Args:
            **kwargs: 초기 입력

        Returns:
            ChainResult: 최종 결과
        """
        steps: List[Dict[str, Any]] = []
        current_output: Optional[str] = None

        # 기존 chain.py의 SequentialChain.run() 로직 정확히 마이그레이션
        try:
            for i, chain in enumerate(self.chains):
                logger.debug(f"Executing chain {i + 1}/{len(self.chains)}")

                # 첫 번째 체인은 kwargs 사용, 이후는 이전 출력 사용 (기존과 동일)
                if i == 0:
                    result = await chain.run(**kwargs)
                else:
                    # 이전 출력을 다음 체인의 입력으로 (기존과 동일)
                    if isinstance(chain, PromptChain):
                        result = await chain.run(input=current_output)
                    else:
                        result = await chain.run(current_output)

                if not result.success:
                    return result

                current_output = result.output
                steps.extend(result.steps)

            return ChainResult(output=current_output or "", steps=steps, success=True)

        except Exception as e:
            logger.error(f"SequentialChain error: {e}")
            return ChainResult(output="", steps=steps, success=False, error=str(e))


class ParallelChain:
    """
    병렬 실행 체인 (Facade 패턴)

    기존 API를 유지하면서 내부적으로는 Handler/Service 사용
    """

    def __init__(self, chains: List[Union[Chain, PromptChain]]):
        """
        Args:
            chains: 체인 목록
        """
        self.chains = chains

        # Handler/Service 초기화
        self._init_services()

    def _init_services(self) -> None:
        """Service 및 Handler 초기화"""
        from ..utils.di_container import get_container

        container = get_container()
        handler_factory = container.handler_factory
        self._chain_handler = handler_factory.create_chain_handler()

    async def run(self, **kwargs) -> ChainResult:
        """
        병렬 실행

        내부적으로 각 Chain을 직접 실행 (기존 chain.py의 ParallelChain.run() 정확히 마이그레이션)

        Args:
            **kwargs: 입력

        Returns:
            ChainResult: 결합된 결과
        """
        # 기존 chain.py의 ParallelChain.run() 로직 정확히 마이그레이션
        try:
            # 모든 체인을 동시에 실행 (기존과 동일)
            tasks = [chain.run(**kwargs) for chain in self.chains]
            results = await asyncio.gather(*tasks)

            # 결과 결합 (기존과 동일)
            outputs = [r.output for r in results]
            all_steps: List[Dict[str, Any]] = []
            for r in results:
                all_steps.extend(r.steps)

            # 성공 여부 확인 (기존과 동일)
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
    체인 빌더 (Fluent API) - Facade 패턴

    기존 API를 유지하면서 내부적으로는 Handler/Service 사용
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

        내부적으로 Handler를 사용하여 처리

        Args:
            **kwargs: 입력 파라미터

        Returns:
            ChainResult: 실행 결과
        """
        # Handler/Service 초기화
        from ..utils.di_container import get_container

        container = get_container()
        handler_factory = container.handler_factory
        chain_handler = handler_factory.create_chain_handler()

        # 적절한 체인 타입 선택
        if self._template:
            response = await chain_handler.handle_run(
                chain_type="prompt",
                template=self._template,
                template_vars=kwargs,
                model=self.client.model,
                memory_type="buffer" if isinstance(self._memory, BufferMemory) else None,
            )
        else:
            user_input = kwargs.pop("input", None) or kwargs.pop("question", "")
            response = await chain_handler.handle_run(
                chain_type="basic",
                user_input=user_input,
                model=self.client.model,
                memory_type="buffer" if isinstance(self._memory, BufferMemory) else None,
                verbose=self._verbose,
                **kwargs,
            )

        # ChainResponse를 ChainResult로 변환
        return ChainResult(
            output=response.output,
            steps=response.steps,
            metadata=response.metadata,
            success=response.success,
            error=response.error,
        )

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
        from beanllm import Client, create_chain

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
