"""
ChainHandler - Chain 요청 처리 (Controller 역할)
책임 분리:
- 모든 if-else/try-catch 처리
- 입력 검증
- DTO 변환
- 결과 출력
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..decorators.error_handler import handle_errors
from ..decorators.logger import log_handler_call
from ..decorators.validation import validate_input
from ..dto.request.chain_request import ChainRequest
from ..dto.response.chain_response import ChainResponse
from ..service.chain_service import IChainService


class ChainHandler:
    """
    Chain 요청 처리 Handler

    책임:
    - 입력 검증 (if-else)
    - 에러 처리 (try-catch)
    - DTO 변환
    - Service 호출
    - 비즈니스 로직 없음
    """

    def __init__(self, chain_service: IChainService) -> None:
        """
        의존성 주입

        Args:
            chain_service: Chain 서비스 (인터페이스에 의존 - DIP)
        """
        self._chain_service = chain_service

    @log_handler_call
    @handle_errors(error_message="Chain execution failed")
    @validate_input(
        required_params=["chain_type"],
        param_types={"chain_type": str, "user_input": str, "template": str},
    )
    async def handle_run(
        self,
        chain_type: str = "basic",
        user_input: Optional[str] = None,
        template: Optional[str] = None,
        template_vars: Optional[Dict[str, Any]] = None,
        chains: Optional[List[Any]] = None,
        model: str = "gpt-4o-mini",
        memory_type: Optional[str] = None,
        memory_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Any]] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> ChainResponse:
        """
        Chain 실행 요청 처리 (모든 검증 및 에러 처리 포함)

        Args:
            chain_type: 체인 타입 (basic, prompt, sequential, parallel)
            user_input: 사용자 입력 (basic Chain용)
            template: 프롬프트 템플릿 (prompt Chain용)
            template_vars: 템플릿 변수 (prompt Chain용)
            chains: 체인 리스트 (sequential, parallel Chain용)
            model: 모델 이름
            memory_type: 메모리 타입
            memory_config: 메모리 설정
            tools: 도구 리스트
            verbose: 상세 로그
            **kwargs: 추가 파라미터

        Returns:
            ChainResponse: Chain 응답

        책임:
            - 입력 검증 (decorator로 처리)
            - 에러 처리 (decorator로 처리)
            - DTO 변환
            - Service 호출
        """
        # DTO 생성
        request = ChainRequest(
            chain_type=chain_type,
            user_input=user_input,
            template=template,
            template_vars=template_vars or {},
            chains=chains or [],
            model=model,
            memory_type=memory_type,
            memory_config=memory_config or {},
            tools=tools or [],
            verbose=verbose,
            extra_params=kwargs,
        )

        # Service 호출 (Strategy 패턴 적용 - 통합 execute 메서드 사용)
        return await self._chain_service.execute(request)
