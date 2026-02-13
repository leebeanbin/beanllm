"""
ChainServiceImpl - Chain 서비스 구현체
SOLID 원칙:
- SRP: Chain 비즈니스 로직만 담당
- DIP: 인터페이스에 의존 (의존성 주입)

Architecture:
- 메서드 호출로 흐름 표현 (if문 최소화)
- 헬퍼 메서드로 공통 로직 추출
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

from beanllm.dto.request.core.chain_request import ChainRequest
from beanllm.dto.response.core.chain_response import ChainResponse
from beanllm.infrastructure.distributed.pipeline_decorators import (
    with_distributed_features,
)
from beanllm.service.chain_service import IChainService
from beanllm.utils.logging import get_logger

# 환경변수로 분산 모드 활성화 여부 확인
USE_DISTRIBUTED = os.getenv("USE_DISTRIBUTED", "false").lower() == "true"

if TYPE_CHECKING:
    from beanllm.domain.memory.base import BaseMemory
    from beanllm.dto.response.core.chat_response import ChatResponse
    from beanllm.service.chat_service import IChatService

logger = get_logger(__name__)


class ChainServiceImpl(IChainService):
    """
    Chain 서비스 구현체

    책임:
    - Chain 비즈니스 로직만
    - 검증 없음 (Handler에서 처리)
    - 에러 처리 없음 (Handler에서 처리)

    Architecture:
    - 각 public 메서드는 헬퍼 메서드 호출로 흐름 표현
    - Optional 처리는 헬퍼 메서드에서 일괄 처리
    """

    def __init__(self, chat_service: "IChatService") -> None:
        self._chat_service = chat_service

    # ===== Public Methods (흐름만 표현) =====

    @with_distributed_features(
        pipeline_type="chain",
        enable_cache=True,
        enable_rate_limiting=True,
        enable_event_streaming=True,
        cache_key_prefix="chain:basic",
        rate_limit_key=lambda self, args, kwargs: f"llm:{self._extract_model(args, kwargs)}",
        event_type="chain.basic",
    )
    async def run_chain(self, request: ChainRequest) -> ChainResponse:
        """기본 Chain 실행 - 메모리 기반 대화"""
        memory = self._create_memory(request)
        self._add_user_message(memory, request.user_input)
        response = await self._call_llm(memory.get_dict_messages(), request)
        self._add_assistant_message(memory, response.content)
        return self._build_chain_response(request.user_input or "", response.content, "llm")

    async def run_prompt_chain(self, request: ChainRequest) -> ChainResponse:
        """Prompt Chain 실행 - 템플릿 기반 대화"""
        self._validate_template(request)
        prompt = self._render_template(request)
        messages = self._build_messages_with_memory(request, prompt)
        response = await self._call_llm(messages, request)
        self._update_memory_if_exists(request, prompt, response.content)
        return self._build_prompt_chain_response(request, response.content)

    @with_distributed_features(
        pipeline_type="chain",
        enable_event_streaming=True,
        enable_distributed_lock=True,
        lock_key=lambda self, args, kwargs: f"chain:sequential:{self._hash_chains(args, kwargs)}",
        event_type="chain.sequential",
    )
    async def run_sequential_chain(self, request: ChainRequest) -> ChainResponse:
        """Sequential Chain 실행 - 순차적 체인 연결"""
        chains = request.chains or []
        template_vars = self._get_template_vars(request)

        steps: List[Dict[str, Any]] = []
        current_output: Optional[str] = None

        for i, chain_request in enumerate(chains):
            logger.debug(f"Executing chain {i + 1}/{len(chains)}")
            result = await self._execute_sequential_step(
                cast(ChainRequest, chain_request),
                template_vars,
                current_output,
                i,
            )
            if not result.success:
                return result
            current_output = result.output
            steps.extend(result.steps)

        return ChainResponse(output=current_output or "", steps=steps, success=True)

    @with_distributed_features(
        pipeline_type="chain",
        enable_rate_limiting=True,
        enable_event_streaming=True,
        rate_limit_key=lambda self, args, kwargs: f"llm:{self._extract_model(args, kwargs)}",
        event_type="chain.parallel",
    )
    async def run_parallel_chain(self, request: ChainRequest) -> ChainResponse:
        """Parallel Chain 실행 - 병렬 체인 처리"""
        chains = request.chains or []
        template_vars = self._get_template_vars(request)

        chain_results = await self._execute_parallel_chains(
            cast(List[ChainRequest], chains), template_vars
        )
        return self._combine_parallel_results(chain_results)

    # ===== Helper Methods: Request Parameter Extraction =====

    def _get_template_vars(self, request: ChainRequest) -> Dict[str, Any]:
        """템플릿 변수 추출 (None-safe)"""
        return request.template_vars or {}

    def _get_extra_params(self, request: ChainRequest) -> Dict[str, Any]:
        """추가 파라미터 추출 (None-safe)"""
        return request.extra_params or {}

    def _get_memory_config(self, request: ChainRequest) -> Dict[str, Any]:
        """메모리 설정 추출 (None-safe)"""
        return request.memory_config or {}

    # ===== Helper Methods: Memory Management =====

    def _create_memory(self, request: ChainRequest) -> "BaseMemory":
        """메모리 생성 (타입에 따라 분기)"""
        from beanllm.domain.memory import BufferMemory, create_memory

        if request.memory_type:
            return create_memory(request.memory_type, **self._get_memory_config(request))
        return BufferMemory()

    def _add_user_message(self, memory: "BaseMemory", user_input: Optional[str]) -> None:
        """사용자 메시지 추가"""
        if user_input:
            memory.add_message("user", user_input)

    def _add_assistant_message(self, memory: "BaseMemory", content: str) -> None:
        """어시스턴트 메시지 추가"""
        memory.add_message("assistant", content)

    def _update_memory_if_exists(
        self, request: ChainRequest, prompt: str, response_content: str
    ) -> None:
        """메모리가 있으면 업데이트"""
        if request.memory_type:
            from beanllm.domain.memory import create_memory

            memory = create_memory(request.memory_type, **self._get_memory_config(request))
            memory.add_message("user", prompt)
            memory.add_message("assistant", response_content)

    # ===== Helper Methods: LLM Interaction =====

    async def _call_llm(
        self, messages: List[Dict[str, str]], request: ChainRequest
    ) -> "ChatResponse":
        """LLM 호출"""
        from beanllm.dto.request.core.chat_request import ChatRequest as ChatReq

        chat_request = ChatReq(
            messages=messages,
            model=request.model,
            **self._get_extra_params(request),
        )
        return await self._chat_service.chat(chat_request)

    # ===== Helper Methods: Template Processing =====

    def _validate_template(self, request: ChainRequest) -> None:
        """템플릿 필수 확인"""
        if not request.template:
            raise ValueError("Template is required for PromptChain")

    def _render_template(self, request: ChainRequest) -> str:
        """템플릿 렌더링"""
        template = request.template
        assert template is not None  # _validate_template에서 확인됨
        return template.format(**self._get_template_vars(request))

    def _build_messages_with_memory(
        self, request: ChainRequest, prompt: str
    ) -> List[Dict[str, str]]:
        """메모리 메시지 + 현재 프롬프트 결합"""
        from beanllm.domain.memory import create_memory

        messages: List[Dict[str, str]] = []
        if request.memory_type:
            memory = create_memory(request.memory_type, **self._get_memory_config(request))
            messages = memory.get_dict_messages()
        messages.append({"role": "user", "content": prompt})
        return messages

    # ===== Helper Methods: Response Building =====

    def _build_chain_response(
        self, input_text: str, output_text: str, chain_type: str
    ) -> ChainResponse:
        """Chain 응답 생성"""
        return ChainResponse(
            output=output_text,
            steps=[{"type": chain_type, "input": input_text, "output": output_text}],
            success=True,
        )

    def _build_prompt_chain_response(
        self, request: ChainRequest, output_text: str
    ) -> ChainResponse:
        """Prompt Chain 응답 생성"""
        return ChainResponse(
            output=output_text,
            steps=[
                {
                    "type": "prompt",
                    "template": request.template,
                    "vars": request.template_vars,
                }
            ],
            success=True,
        )

    # ===== Helper Methods: Sequential Chain =====

    async def _execute_sequential_step(
        self,
        chain_request: ChainRequest,
        template_vars: Dict[str, Any],
        previous_output: Optional[str],
        step_index: int,
    ) -> ChainResponse:
        """Sequential Chain의 단일 스텝 실행"""
        if step_index == 0:
            return await self._execute_first_step(chain_request, template_vars)
        return await self._execute_subsequent_step(chain_request, previous_output)

    async def _execute_first_step(
        self, chain_request: ChainRequest, template_vars: Dict[str, Any]
    ) -> ChainResponse:
        """첫 번째 스텝 실행 (초기 변수 사용)"""
        if chain_request.template:
            chain_request.template_vars = template_vars
            return cast(ChainResponse, await self.run_prompt_chain(chain_request))
        return cast(ChainResponse, await self.run_chain(chain_request))

    async def _execute_subsequent_step(
        self, chain_request: ChainRequest, previous_output: Optional[str]
    ) -> ChainResponse:
        """이후 스텝 실행 (이전 출력 사용)"""
        if chain_request.template:
            chain_request.template_vars = {"input": previous_output}
            return cast(ChainResponse, await self.run_prompt_chain(chain_request))
        chain_request.user_input = previous_output or ""
        return cast(ChainResponse, await self.run_chain(chain_request))

    # ===== Helper Methods: Parallel Chain =====

    async def _execute_parallel_chains(
        self, chains: List[ChainRequest], template_vars: Dict[str, Any]
    ) -> List[ChainResponse]:
        """병렬 체인 실행 (분산/로컬 자동 선택)"""
        if USE_DISTRIBUTED and len(chains) > 3:
            return await self._execute_distributed_parallel(chains, template_vars)
        return await self._execute_local_parallel(chains, template_vars)

    async def _execute_local_parallel(
        self, chains: List[ChainRequest], template_vars: Dict[str, Any]
    ) -> List[ChainResponse]:
        """로컬 병렬 실행"""
        tasks = [self._execute_single_chain(chain, template_vars) for chain in chains]
        results = await asyncio.gather(*tasks)
        return list(results)

    async def _execute_distributed_parallel(
        self, chains: List[ChainRequest], template_vars: Dict[str, Any]
    ) -> List[ChainResponse]:
        """분산 병렬 실행 (실패 시 로컬로 fallback)"""
        try:
            return await self._try_distributed_execution(chains, template_vars)
        except Exception as e:
            logger.warning(f"Distributed parallel chain failed: {e}, falling back to local")
            return await self._execute_local_parallel(chains, template_vars)

    async def _try_distributed_execution(
        self, chains: List[ChainRequest], template_vars: Dict[str, Any]
    ) -> List[ChainResponse]:
        """분산 실행 시도"""
        from beanllm.infrastructure.distributed import BatchProcessor

        batch_processor = BatchProcessor(task_type="chain.tasks", max_concurrent=10)
        results = await batch_processor.process_items(
            items=chains,
            handler=lambda chain_req: self._execute_single_chain(chain_req, template_vars),
        )
        return [
            r if isinstance(r, ChainResponse) else ChainResponse(output=str(r), success=False)
            for r in results
        ]

    async def _execute_single_chain(
        self, chain_request: ChainRequest, template_vars: Dict[str, Any]
    ) -> ChainResponse:
        """단일 체인 실행"""
        if chain_request.template:
            chain_request.template_vars = template_vars
            return cast(ChainResponse, await self.run_prompt_chain(chain_request))
        return cast(ChainResponse, await self.run_chain(chain_request))

    def _combine_parallel_results(self, results: List[ChainResponse]) -> ChainResponse:
        """병렬 실행 결과 결합"""
        outputs = [r.output for r in results]
        all_steps: List[Dict[str, Any]] = []
        for r in results:
            all_steps.extend(r.steps)

        success = all(r.success for r in results)
        errors = [r.error for r in results if r.error]

        return ChainResponse(
            output="\n\n---\n\n".join(outputs),
            steps=all_steps,
            metadata={"outputs": outputs, "count": len(outputs)},
            success=success,
            error="; ".join(errors) if errors else None,
        )

    # ===== Utility Methods for Decorators =====

    @staticmethod
    def _extract_model(args: tuple, kwargs: Dict[str, Any]) -> str:
        """데코레이터용 모델 추출"""
        request = args[0] if args else kwargs.get("request")
        if request and hasattr(request, "model") and request.model:
            return str(request.model)
        return "default"

    @staticmethod
    def _hash_chains(args: tuple, kwargs: Dict[str, Any]) -> str:
        """데코레이터용 체인 해시"""
        request = args[0] if args else kwargs.get("request")
        if request and hasattr(request, "chains") and request.chains:
            return str(hash(str(request.chains)))
        return "default"
