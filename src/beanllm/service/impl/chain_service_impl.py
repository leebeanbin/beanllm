"""
ChainServiceImpl - Chain 서비스 구현체
SOLID 원칙:
- SRP: Chain 비즈니스 로직만 담당
- DIP: 인터페이스에 의존 (의존성 주입)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ...dto.request.chain_request import ChainRequest
from ...dto.response.chain_response import ChainResponse
from ...utils.logger import get_logger
from ..chain_service import IChainService

if TYPE_CHECKING:
    from ...service.chat_service import IChatService

logger = get_logger(__name__)


class ChainServiceImpl(IChainService):
    """
    Chain 서비스 구현체

    책임:
    - Chain 비즈니스 로직만
    - 검증 없음 (Handler에서 처리)
    - 에러 처리 없음 (Handler에서 처리)

    SOLID:
    - SRP: Chain 비즈니스 로직만
    - DIP: 인터페이스에 의존 (의존성 주입)
    """

    def __init__(
        self,
        chat_service: "IChatService",
    ) -> None:
        """
        의존성 주입을 통한 생성자

        Args:
            chat_service: 채팅 서비스
        """
        self._chat_service = chat_service

    async def run_chain(self, request: ChainRequest) -> ChainResponse:
        """
        기본 Chain 실행 (기존 chain.py의 Chain.run() 정확히 마이그레이션)

        Args:
            request: Chain 요청 DTO

        Returns:
            ChainResponse: Chain 응답 DTO
        """
        from ...domain.memory import BufferMemory, create_memory
        from ...dto.request.chat_request import ChatRequest

        # 메모리 생성 (기존: memory or BufferMemory())
        if request.memory_type:
            memory = create_memory(request.memory_type, **request.memory_config)
        else:
            memory = BufferMemory()

        # 메모리에 사용자 메시지 추가 (기존과 동일)
        if request.user_input:
            memory.add_message("user", request.user_input)

        # LLM 호출 (기존: await self.client.chat(messages, **kwargs))
        messages = memory.get_dict_messages()
        chat_request = ChatRequest(
            messages=messages,
            model=request.model,
            **request.extra_params,
        )
        response = await self._chat_service.chat(chat_request)

        # 메모리에 응답 추가 (기존과 동일)
        memory.add_message("assistant", response.content)

        # 결과 반환 (기존과 동일)
        return ChainResponse(
            output=response.content,
            steps=[{"type": "llm", "input": request.user_input or "", "output": response.content}],
            success=True,
        )

    async def run_prompt_chain(self, request: ChainRequest) -> ChainResponse:
        """
        Prompt Chain 실행 (기존 chain.py의 PromptChain.run() 정확히 마이그레이션)

        Args:
            request: Chain 요청 DTO

        Returns:
            ChainResponse: Chain 응답 DTO
        """
        from ...domain.memory import create_memory
        from ...dto.request.chat_request import ChatRequest

        if not request.template:
            raise ValueError("Template is required for PromptChain")

        # 템플릿 렌더링 (기존과 동일)
        prompt = request.template.format(**request.template_vars)

        # 메모리 사용 (기존과 동일)
        messages = []
        memory = None
        if request.memory_type:
            memory = create_memory(request.memory_type, **request.memory_config)
            messages = memory.get_dict_messages()

        messages.append({"role": "user", "content": prompt})

        # LLM 호출 (기존: await self.client.chat(messages))
        chat_request = ChatRequest(
            messages=messages,
            model=request.model,
            **request.extra_params,
        )
        response = await self._chat_service.chat(chat_request)

        # 메모리 업데이트 (기존과 동일)
        if memory:
            memory.add_message("user", prompt)
            memory.add_message("assistant", response.content)

        # 결과 반환 (기존과 동일)
        return ChainResponse(
            output=response.content,
            steps=[{"type": "prompt", "template": request.template, "vars": request.template_vars}],
            success=True,
        )

    async def run_sequential_chain(self, request: ChainRequest) -> ChainResponse:
        """
        Sequential Chain 실행 (기존 chain.py의 SequentialChain.run() 정확히 마이그레이션)

        Args:
            request: Chain 요청 DTO

        Returns:
            ChainResponse: Chain 응답 DTO

        Note: 기존 코드는 Chain/PromptChain 객체 리스트를 받지만,
        새로운 구조에서는 ChainRequest 리스트를 받아서 처리합니다.
        """
        steps: List[Dict[str, Any]] = []
        current_output: Optional[str] = None

        # 기존 로직 정확히 마이그레이션
        # 기존: for i, chain in enumerate(self.chains)
        # 새로운 구조: request.chains는 ChainRequest 리스트
        chains = request.chains or []
        template_vars = request.template_vars or {}

        for i, chain_request in enumerate(chains):
            logger.debug(f"Executing chain {i + 1}/{len(chains)}")

            # 첫 번째 체인은 kwargs 사용, 이후는 이전 출력 사용 (기존과 동일)
            if i == 0:
                # 첫 번째 체인: template_vars 사용
                if chain_request.template:
                    chain_request.template_vars = template_vars
                    result = await self.run_prompt_chain(chain_request)
                else:
                    result = await self.run_chain(chain_request)
            else:
                # 이전 출력을 다음 체인의 입력으로 (기존과 동일)
                if chain_request.template:
                    # PromptChain인 경우 input 파라미터로 전달
                    chain_request.template_vars = {"input": current_output}
                    result = await self.run_prompt_chain(chain_request)
                else:
                    # Chain인 경우 user_input으로 전달
                    chain_request.user_input = current_output or ""
                    result = await self.run_chain(chain_request)

            if not result.success:
                return result

            current_output = result.output
            steps.extend(result.steps)

        return ChainResponse(output=current_output or "", steps=steps, success=True)

    async def run_parallel_chain(self, request: ChainRequest) -> ChainResponse:
        """
        Parallel Chain 실행 (기존 chain.py의 ParallelChain.run() 정확히 마이그레이션)

        Args:
            request: Chain 요청 DTO

        Returns:
            ChainResponse: Chain 응답 DTO
        """
        # 모든 체인을 동시에 실행 (기존: await asyncio.gather(*tasks))
        # 기존: tasks = [chain.run(**kwargs) for chain in self.chains]
        chains = request.chains or []
        template_vars = request.template_vars or {}

        async def run_chain_request(chain_req: ChainRequest) -> ChainResponse:
            """체인 요청 실행 (타입에 따라 분기)"""
            if chain_req.template:
                chain_req.template_vars = template_vars
                return await self.run_prompt_chain(chain_req)
            else:
                return await self.run_chain(chain_req)

        tasks = [run_chain_request(chain_request) for chain_request in chains]
        results = await asyncio.gather(*tasks)

        # 결과 결합 (기존과 동일)
        outputs = [r.output for r in results]
        all_steps: List[Dict[str, Any]] = []
        for r in results:
            all_steps.extend(r.steps)

        # 성공 여부 확인 (기존과 동일)
        success = all(r.success for r in results)
        errors = [r.error for r in results if r.error]

        return ChainResponse(
            output="\n\n---\n\n".join(outputs),
            steps=all_steps,
            metadata={"outputs": outputs, "count": len(outputs)},
            success=success,
            error="; ".join(errors) if errors else None,
        )
