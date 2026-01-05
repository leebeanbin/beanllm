"""
Query Expansion - 쿼리 확장 기법들 (2024-2025)

검색 품질 향상을 위한 쿼리 확장 전략들을 제공합니다.

Query Expansion 기법:
- HyDE (Hypothetical Document Embeddings): 가상 문서 생성 후 임베딩
- Multi-Query: 여러 관점의 쿼리 생성
- Step-back Prompting: 넓은 맥락에서 쿼리 재구성

HyDE 특징:
- 쿼리와 문서 간의 의미적 갭 해소
- LLM으로 가상 답변 생성 → 이를 임베딩하여 검색
- 30-40% 검색 품질 향상 (특히 전문 도메인)

References:
    - "Precise Zero-Shot Dense Retrieval without Relevance Labels" (HyDE)
    - https://arxiv.org/abs/2212.10496
"""

import logging
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union

try:
    from beanllm.utils.logger import get_logger
except ImportError:

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class BaseQueryExpander(ABC):
    """
    쿼리 확장 베이스 클래스

    검색 쿼리를 확장하여 검색 품질을 향상시킵니다.
    """

    @abstractmethod
    def expand(self, query: str) -> Union[str, List[str]]:
        """
        쿼리 확장

        Args:
            query: 원본 쿼리

        Returns:
            확장된 쿼리 (단일 또는 리스트)
        """
        pass


class HyDEExpander(BaseQueryExpander):
    """
    HyDE (Hypothetical Document Embeddings) 쿼리 확장

    LLM으로 가상 답변(hypothetical document)을 생성하고,
    이를 사용하여 검색하는 기법입니다.

    HyDE 작동 방식:
    1. 사용자 쿼리 입력
    2. LLM으로 가상 답변 생성
    3. 가상 답변을 임베딩
    4. 임베딩으로 문서 검색

    장점:
    - 쿼리-문서 간 의미적 갭 해소
    - Zero-shot 검색 품질 향상
    - 전문 도메인에서 특히 효과적

    Example:
        ```python
        from beanllm.domain.retrieval import HyDEExpander

        # HyDE 생성 (LLM 함수 제공)
        def llm_generate(prompt: str) -> str:
            # OpenAI, Claude 등 LLM API 사용
            return llm.chat(prompt)

        expander = HyDEExpander(
            llm_function=llm_generate,
            prompt_template="Please answer: {query}"
        )

        # 쿼리 확장
        query = "What is machine learning?"
        hypothetical_doc = expander.expand(query)

        # 확장된 쿼리로 검색
        embedding = embed_model.embed(hypothetical_doc)
        results = vector_store.search(embedding, top_k=5)
        ```

    References:
        - Paper: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
        - https://arxiv.org/abs/2212.10496
    """

    def __init__(
        self,
        llm_function: Callable[[str], str],
        prompt_template: Optional[str] = None,
        num_documents: int = 1,
        max_tokens: Optional[int] = 512,
        temperature: float = 0.7,
        **kwargs,
    ):
        """
        Args:
            llm_function: LLM 생성 함수 (prompt -> response)
            prompt_template: 프롬프트 템플릿 ("{query}"를 쿼리로 치환)
            num_documents: 생성할 가상 문서 개수 (기본: 1)
            max_tokens: 최대 토큰 수
            temperature: LLM 온도 (0.7 추천)
            **kwargs: 추가 파라미터
        """
        self.llm_function = llm_function
        self.num_documents = num_documents
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.kwargs = kwargs

        # 프롬프트 템플릿 설정
        if prompt_template is None:
            self.prompt_template = self._default_prompt_template()
        else:
            self.prompt_template = prompt_template

        logger.info(
            f"HyDEExpander initialized: num_docs={num_documents}, "
            f"max_tokens={max_tokens}, temperature={temperature}"
        )

    def _default_prompt_template(self) -> str:
        """
        기본 HyDE 프롬프트 템플릿

        Returns:
            프롬프트 템플릿
        """
        return """Please write a comprehensive answer to the following question.
Write as if you are answering the question directly.

Question: {query}

Answer:"""

    def expand(self, query: str) -> Union[str, List[str]]:
        """
        HyDE 쿼리 확장

        Args:
            query: 원본 쿼리

        Returns:
            가상 문서 (단일 또는 리스트)
        """
        # 프롬프트 생성
        prompt = self.prompt_template.format(query=query)

        logger.info(f"Generating hypothetical document for query: {query[:50]}...")

        # 단일 문서 생성
        if self.num_documents == 1:
            hypothetical_doc = self.llm_function(prompt)

            logger.info(
                f"HyDE document generated: length={len(hypothetical_doc)} chars"
            )

            return hypothetical_doc

        # 여러 문서 생성
        hypothetical_docs = []
        for i in range(self.num_documents):
            doc = self.llm_function(prompt)
            hypothetical_docs.append(doc)

            logger.info(
                f"HyDE document {i+1}/{self.num_documents} generated: "
                f"length={len(doc)} chars"
            )

        return hypothetical_docs

    def __repr__(self) -> str:
        return (
            f"HyDEExpander(num_docs={self.num_documents}, "
            f"temperature={self.temperature})"
        )


class MultiQueryExpander(BaseQueryExpander):
    """
    Multi-Query 확장

    하나의 쿼리를 여러 관점에서 재구성하여 검색 범위를 확장합니다.

    작동 방식:
    1. 원본 쿼리 입력
    2. LLM으로 여러 관점의 쿼리 생성
    3. 각 쿼리로 검색 수행
    4. 결과 결합

    Example:
        ```python
        from beanllm.domain.retrieval import MultiQueryExpander

        expander = MultiQueryExpander(
            llm_function=llm_generate,
            num_queries=3
        )

        # 쿼리 확장
        query = "How does AI work?"
        expanded_queries = expander.expand(query)
        # → [
        #   "What are the principles of artificial intelligence?",
        #   "Explain the mechanisms behind AI systems",
        #   "How do machine learning algorithms function?"
        # ]
        ```
    """

    def __init__(
        self,
        llm_function: Callable[[str], str],
        prompt_template: Optional[str] = None,
        num_queries: int = 3,
        **kwargs,
    ):
        """
        Args:
            llm_function: LLM 생성 함수
            prompt_template: 프롬프트 템플릿
            num_queries: 생성할 쿼리 개수
            **kwargs: 추가 파라미터
        """
        self.llm_function = llm_function
        self.num_queries = num_queries
        self.kwargs = kwargs

        # 프롬프트 템플릿 설정
        if prompt_template is None:
            self.prompt_template = self._default_prompt_template()
        else:
            self.prompt_template = prompt_template

        logger.info(f"MultiQueryExpander initialized: num_queries={num_queries}")

    def _default_prompt_template(self) -> str:
        """기본 Multi-Query 프롬프트 템플릿"""
        return """Generate {num_queries} different versions of the following question.
Each version should ask the same thing but from a different perspective.

Original question: {query}

Please provide {num_queries} alternative versions:"""

    def expand(self, query: str) -> List[str]:
        """
        Multi-Query 확장

        Args:
            query: 원본 쿼리

        Returns:
            확장된 쿼리 리스트
        """
        # 프롬프트 생성
        prompt = self.prompt_template.format(
            query=query, num_queries=self.num_queries
        )

        logger.info(f"Generating {self.num_queries} alternative queries...")

        # LLM으로 확장 쿼리 생성
        response = self.llm_function(prompt)

        # 응답 파싱 (줄바꿈 기준 분리)
        lines = [line.strip() for line in response.split("\n") if line.strip()]

        # 번호 제거 (1., 2., -, * 등)
        import re

        queries = []
        for line in lines:
            # 번호, 불릿 제거
            cleaned = re.sub(r"^[\d\.\-\*\)\]]+\s*", "", line)
            if cleaned and len(cleaned) > 10:  # 최소 길이 체크
                queries.append(cleaned)

        # num_queries 개수만큼 선택
        queries = queries[: self.num_queries]

        logger.info(f"Multi-Query expansion completed: {len(queries)} queries generated")

        return queries

    def __repr__(self) -> str:
        return f"MultiQueryExpander(num_queries={self.num_queries})"


class StepBackExpander(BaseQueryExpander):
    """
    Step-back Prompting 확장

    구체적인 쿼리를 더 넓은 맥락의 쿼리로 재구성합니다.

    작동 방식:
    1. 구체적인 쿼리 입력
    2. LLM으로 더 일반적인 배경 지식 쿼리 생성
    3. 배경 지식 검색 → 원본 쿼리 검색
    4. 결합하여 더 나은 답변 생성

    Example:
        ```python
        from beanllm.domain.retrieval import StepBackExpander

        expander = StepBackExpander(llm_function=llm_generate)

        # 쿼리 확장
        query = "What was the impact of COVID-19 on the tech industry in 2020?"
        step_back_query = expander.expand(query)
        # → "What is the general relationship between pandemics and technology sectors?"
        ```

    References:
        - "Take a Step Back: Evoking Reasoning via Abstraction in LLMs"
    """

    def __init__(
        self,
        llm_function: Callable[[str], str],
        prompt_template: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            llm_function: LLM 생성 함수
            prompt_template: 프롬프트 템플릿
            **kwargs: 추가 파라미터
        """
        self.llm_function = llm_function
        self.kwargs = kwargs

        # 프롬프트 템플릿 설정
        if prompt_template is None:
            self.prompt_template = self._default_prompt_template()
        else:
            self.prompt_template = prompt_template

        logger.info("StepBackExpander initialized")

    def _default_prompt_template(self) -> str:
        """기본 Step-back 프롬프트 템플릿"""
        return """Given the following specific question, generate a more general question
that would help provide background knowledge to answer the original question.

Specific question: {query}

General question:"""

    def expand(self, query: str) -> str:
        """
        Step-back 확장

        Args:
            query: 원본 쿼리

        Returns:
            Step-back 쿼리
        """
        # 프롬프트 생성
        prompt = self.prompt_template.format(query=query)

        logger.info(f"Generating step-back query for: {query[:50]}...")

        # LLM으로 step-back 쿼리 생성
        step_back_query = self.llm_function(prompt)

        logger.info(f"Step-back query generated: {step_back_query[:50]}...")

        return step_back_query.strip()

    def __repr__(self) -> str:
        return "StepBackExpander()"
