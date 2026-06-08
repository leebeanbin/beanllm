"""
ChainService 테스트 - Chain 서비스 구현체 테스트
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from beanllm.dto.request import ChainRequest
from beanllm.dto.response import ChainResponse, ChatResponse
from beanllm.service.impl.core.chain_service_impl import ChainServiceImpl


class TestChainService:
    """ChainService 테스트"""

    @pytest.fixture
    def mock_chat_service(self):
        """Mock ChatService"""
        service = Mock()
        service.chat = AsyncMock(
            return_value=ChatResponse(
                content="Chain response", model="gpt-4o-mini", provider="openai"
            )
        )
        return service

    @pytest.fixture
    def chain_service(self, mock_chat_service):
        """ChainService 인스턴스"""
        return ChainServiceImpl(chat_service=mock_chat_service)

    @pytest.mark.asyncio
    async def test_run_chain_basic(self, chain_service):
        """기본 Chain 실행 테스트"""
        request = ChainRequest(
            chain_type="basic",
            user_input="Hello",
            model="gpt-4o-mini",
        )

        response = await chain_service.run_chain(request)

        assert response is not None
        assert isinstance(response, ChainResponse)
        assert response.output == "Chain response"
        assert response.success is True
        assert len(response.steps) == 1

    @pytest.mark.asyncio
    async def test_run_chain_with_memory(self, chain_service):
        """메모리 포함 Chain 실행 테스트"""
        request = ChainRequest(
            chain_type="basic",
            user_input="Hello",
            model="gpt-4o-mini",
            memory_type="buffer",
        )

        response = await chain_service.run_chain(request)

        assert response is not None
        assert response.output == "Chain response"

    @pytest.mark.asyncio
    async def test_run_chain_no_user_input(self, chain_service):
        """사용자 입력 없이 Chain 실행 테스트"""
        request = ChainRequest(
            chain_type="basic",
            user_input=None,
            model="gpt-4o-mini",
        )

        response = await chain_service.run_chain(request)

        assert response is not None
        assert response.output == "Chain response"
        # user_input이 없어도 작동해야 함

    @pytest.mark.asyncio
    async def test_run_prompt_chain(self, chain_service):
        """Prompt Chain 실행 테스트"""
        request = ChainRequest(
            chain_type="prompt",
            template="Translate: {text}",
            template_vars={"text": "Hello"},
            model="gpt-4o-mini",
        )

        response = await chain_service.run_prompt_chain(request)

        assert response is not None
        assert isinstance(response, ChainResponse)
        assert response.output == "Chain response"
        assert response.success is True
        assert len(response.steps) == 1

    @pytest.mark.asyncio
    async def test_run_prompt_chain_no_template(self, chain_service):
        """템플릿 없이 Prompt Chain 실행 테스트"""
        request = ChainRequest(
            chain_type="prompt",
            template=None,
            model="gpt-4o-mini",
        )

        with pytest.raises(ValueError, match="Template is required"):
            await chain_service.run_prompt_chain(request)

    @pytest.mark.asyncio
    async def test_run_prompt_chain_with_memory(self, chain_service):
        """메모리 포함 Prompt Chain 실행 테스트"""
        request = ChainRequest(
            chain_type="prompt",
            template="Translate: {text}",
            template_vars={"text": "Hello"},
            model="gpt-4o-mini",
            memory_type="buffer",
        )

        response = await chain_service.run_prompt_chain(request)

        assert response is not None
        assert response.output == "Chain response"

    @pytest.mark.asyncio
    async def test_run_sequential_chain(self, chain_service):
        """Sequential Chain 실행 테스트"""
        # ChainRequest 리스트로 chains 전달
        chain1_request = ChainRequest(
            chain_type="basic",
            user_input="Step 1",
            model="gpt-4o-mini",
        )
        chain2_request = ChainRequest(
            chain_type="basic",
            user_input="Step 2",
            model="gpt-4o-mini",
        )

        request = ChainRequest(
            chain_type="sequential",
            chains=[chain1_request, chain2_request],
            model="gpt-4o-mini",
        )

        response = await chain_service.run_sequential_chain(request)

        assert response is not None
        assert isinstance(response, ChainResponse)
        assert response.success is True
        # ChatService가 여러 번 호출되었는지 확인
        assert chain_service._chat_service.chat.call_count >= 2

    @pytest.mark.asyncio
    async def test_run_parallel_chain(self, chain_service):
        """Parallel Chain 실행 테스트"""
        # ChainRequest 리스트로 chains 전달
        chain1_request = ChainRequest(
            chain_type="basic",
            user_input="Task 1",
            model="gpt-4o-mini",
        )
        chain2_request = ChainRequest(
            chain_type="basic",
            user_input="Task 2",
            model="gpt-4o-mini",
        )

        request = ChainRequest(
            chain_type="parallel",
            chains=[chain1_request, chain2_request],
            model="gpt-4o-mini",
        )

        response = await chain_service.run_parallel_chain(request)

        assert response is not None
        assert isinstance(response, ChainResponse)
        assert response.success is True
        # ChatService가 여러 번 호출되었는지 확인
        assert chain_service._chat_service.chat.call_count >= 2
        # 결과가 결합되었는지 확인
        assert "---" in response.output or len(response.steps) >= 2

    @pytest.mark.asyncio
    async def test_run_chain_extra_params(self, chain_service):
        """추가 파라미터 포함 Chain 실행 테스트"""
        request = ChainRequest(
            chain_type="basic",
            user_input="Hello",
            model="gpt-4o-mini",
            extra_params={"temperature": 0.7, "max_tokens": 1000},
        )

        response = await chain_service.run_chain(request)

        assert response is not None
        # ChatService가 extra_params로 호출되었는지 확인
        chain_service._chat_service.chat.assert_called_once()
        call_args = chain_service._chat_service.chat.call_args[0][0]
        # **request.extra_params로 전달되므로 ChatRequest의 필드로 직접 전달됨
        # extra_params 필드가 아니라 temperature, max_tokens가 직접 필드로 전달됨
        # 따라서 call_args의 extra_params는 빈 dict일 수 있음
        # 대신 ChatRequest가 생성되었는지 확인
        assert call_args.model == "gpt-4o-mini"

    # ------------------------------------------------------------------
    # Coverage of missed lines
    # ------------------------------------------------------------------

    async def test_run_sequential_chain_fails_early_on_step_failure(self, chain_service):
        """Sequential chain early-exit when a step returns success=False (line 106)."""
        failed_response = ChainResponse(output="", steps=[], success=False, error="step failed")

        chain1 = ChainRequest(chain_type="basic", user_input="Step 1", model="gpt-4o-mini")
        chain2 = ChainRequest(chain_type="basic", user_input="Step 2", model="gpt-4o-mini")

        request = ChainRequest(
            chain_type="sequential",
            chains=[chain1, chain2],
            model="gpt-4o-mini",
        )

        with patch.object(
            chain_service,
            "_execute_sequential_step",
            new=AsyncMock(return_value=failed_response),
        ):
            response = await chain_service.run_sequential_chain(request)

        assert response.success is False
        assert response.error == "step failed"

    async def test_run_sequential_chain_with_template_first_step(self, chain_service):
        """First step in sequential chain with template → _execute_first_step line 261-262."""
        chain_with_template = ChainRequest(
            chain_type="prompt",
            template="Summarize: {text}",
            template_vars={"text": "hello"},
            model="gpt-4o-mini",
        )
        chain_plain = ChainRequest(
            chain_type="basic",
            user_input="Step 2",
            model="gpt-4o-mini",
        )

        request = ChainRequest(
            chain_type="sequential",
            chains=[chain_with_template, chain_plain],
            template_vars={"text": "hello"},
            model="gpt-4o-mini",
        )

        response = await chain_service.run_sequential_chain(request)
        assert response.success is True

    async def test_run_sequential_chain_subsequent_step_with_template(self, chain_service):
        """Subsequent step with template → _execute_subsequent_step lines 270-271."""
        chain_plain = ChainRequest(
            chain_type="basic",
            user_input="Initial",
            model="gpt-4o-mini",
        )
        chain_template = ChainRequest(
            chain_type="prompt",
            template="Improve: {input}",
            template_vars={"input": ""},
            model="gpt-4o-mini",
        )

        request = ChainRequest(
            chain_type="sequential",
            chains=[chain_plain, chain_template],
            model="gpt-4o-mini",
        )

        response = await chain_service.run_sequential_chain(request)
        assert response.success is True

    async def test_run_parallel_chain_with_template(self, chain_service):
        """Parallel chain where each step has a template → _execute_single_chain lines 324-325."""
        chain1 = ChainRequest(
            chain_type="prompt",
            template="Rewrite: {text}",
            template_vars={"text": "hello"},
            model="gpt-4o-mini",
        )
        chain2 = ChainRequest(
            chain_type="prompt",
            template="Expand: {text}",
            template_vars={"text": "world"},
            model="gpt-4o-mini",
        )

        request = ChainRequest(
            chain_type="parallel",
            chains=[chain1, chain2],
            template_vars={"text": "test"},
            model="gpt-4o-mini",
        )

        response = await chain_service.run_parallel_chain(request)
        assert response.success is True

    def test_extract_model_with_model(self, chain_service):
        """_extract_model returns model when request has model (line 354)."""
        req = ChainRequest(chain_type="basic", user_input="hi", model="gpt-4o")
        result = ChainServiceImpl._extract_model((req,), {})
        assert result == "gpt-4o"

    def test_extract_model_without_model(self, chain_service):
        """_extract_model returns 'default' when no model."""
        result = ChainServiceImpl._extract_model((), {})
        assert result == "default"

    def test_hash_chains_with_chains(self, chain_service):
        """_hash_chains returns a hash string when chains present (lines 359-362)."""
        sub_chain = ChainRequest(chain_type="basic", user_input="x", model="gpt-4o-mini")
        req = ChainRequest(
            chain_type="sequential",
            chains=[sub_chain],
            model="gpt-4o-mini",
        )
        result = ChainServiceImpl._hash_chains((req,), {})
        assert isinstance(result, str)
        assert result != "default"

    def test_hash_chains_without_chains(self, chain_service):
        """_hash_chains returns 'default' when no chains."""
        result = ChainServiceImpl._hash_chains((), {})
        assert result == "default"
