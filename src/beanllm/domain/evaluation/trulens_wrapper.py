"""
TruLens Wrapper - TruLens 통합 (2024-2025)

TruLens는 RAG 시스템 평가 및 모니터링을 위한 프레임워크입니다.

TruLens 특징:
- RAG Triad 메트릭 (Context Relevance, Groundedness, Answer Relevance)
- LLM 추적 및 디버깅
- Snowflake 지원 (엔터프라이즈급 신뢰성)
- LangChain, LlamaIndex 통합
- 시각화 대시보드

RAG Triad 메트릭:
1. Context Relevance: 검색된 컨텍스트가 질문과 관련있는지
2. Groundedness: 답변이 컨텍스트에 근거하는지 (Hallucination 방지)
3. Answer Relevance: 답변이 질문에 적절한지

TruLens vs RAGAS:
- TruLens: RAG Triad, 시각화, 트레이싱, Snowflake 지원
- RAGAS: 더 많은 메트릭, reference-free, 오픈소스 커뮤니티

Requirements:
    pip install trulens-eval

References:
    - https://www.trulens.org/
    - https://github.com/truera/trulens
    - RAG Triad: https://www.trulens.org/trulens_eval/core_concepts_rag_triad/
"""

import logging
from typing import Any, Dict, List, Optional, Union

from .base_framework import BaseEvaluationFramework

try:
    from beanllm.utils.logger import get_logger
except ImportError:

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class TruLensWrapper(BaseEvaluationFramework):
    """
    TruLens 통합 래퍼

    TruLens의 RAG Triad 메트릭을 beanLLM 스타일로 사용할 수 있게 합니다.

    지원 메트릭 (RAG Triad):
    1. Context Relevance: 검색된 컨텍스트가 질문과 관련있는지
       - 쿼리와 관련 없는 컨텍스트 필터링
       - 0-1 점수 (1 = 완전히 관련)

    2. Groundedness: 답변이 컨텍스트에 근거하는지
       - Hallucination 방지
       - 0-1 점수 (1 = 완전히 근거함)

    3. Answer Relevance: 답변이 질문에 적절한지
       - 질문에 대한 직접적 답변 여부
       - 0-1 점수 (1 = 완전히 관련)

    Example:
        ```python
        from beanllm.domain.evaluation import TruLensWrapper

        # 기본 사용
        evaluator = TruLensWrapper(
            provider="openai",
            model="gpt-4o-mini"
        )

        # RAG Triad 평가
        result = evaluator.evaluate_rag_triad(
            question="What is the capital of France?",
            answer="Paris is the capital of France.",
            contexts=["Paris is the capital and largest city of France."]
        )
        print(result)
        # {
        #   "context_relevance": 1.0,
        #   "groundedness": 1.0,
        #   "answer_relevance": 1.0
        # }

        # 개별 메트릭 평가
        context_relevance = evaluator.evaluate_context_relevance(
            question="What is Paris?",
            contexts=["Paris is the capital of France.", "London is in the UK."]
        )
        print(context_relevance)  # {"context_relevance": 0.75}

        # Groundedness 평가 (Hallucination 체크)
        groundedness = evaluator.evaluate_groundedness(
            answer="Paris is the capital of France and has a population of 2 million.",
            contexts=["Paris is the capital and largest city of France."]
        )
        print(groundedness)  # {"groundedness": 0.8}
        ```

    Advanced Usage:
        ```python
        # LangChain 앱 추적
        from langchain.chains import RetrievalQA
        from trulens_eval import TruChain

        # TruLens 래퍼로 추적
        evaluator = TruLensWrapper()
        rag_chain = RetrievalQA(...)

        # 자동 추적 및 평가
        with evaluator.track_app(rag_chain) as recorder:
            result = rag_chain.run("What is AI?")

        # 추적 결과 확인
        evaluator.show_dashboard()
        ```
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        enable_dashboard: bool = False,
        **kwargs,
    ):
        """
        Args:
            provider: LLM 제공자 (openai, anthropic, azure 등)
            model: 모델 이름 (gpt-4o-mini, claude-3-5-sonnet-20241022 등)
            api_key: API 키 (None이면 환경변수)
            enable_dashboard: 대시보드 활성화 여부
            **kwargs: 추가 파라미터
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.enable_dashboard = enable_dashboard
        self.kwargs = kwargs

        # Lazy loading
        self._trulens = None
        self._llm = None
        self._feedback_functions = None

        logger.info(
            f"TruLensWrapper initialized: provider={provider}, "
            f"model={model}, dashboard={enable_dashboard}"
        )

    def _check_dependencies(self):
        """의존성 확인"""
        try:
            import trulens_eval
        except ImportError:
            raise ImportError(
                "trulens-eval is required for TruLensWrapper. "
                "Install it with: pip install trulens-eval"
            )

        self._trulens = trulens_eval

        logger.info("TruLens dependencies loaded")

    def _get_llm(self):
        """LLM 모델 가져오기 (lazy loading)"""
        if self._llm is not None:
            return self._llm

        self._check_dependencies()

        try:
            from trulens_eval import LLMProvider

            # LLM Provider 생성
            if self.provider.lower() == "openai":
                self._llm = LLMProvider.create(
                    provider_name="openai", model_name=self.model, api_key=self.api_key
                )
            elif self.provider.lower() == "anthropic":
                self._llm = LLMProvider.create(
                    provider_name="anthropic", model_name=self.model, api_key=self.api_key
                )
            elif self.provider.lower() == "azure":
                self._llm = LLMProvider.create(
                    provider_name="azure_openai",
                    model_name=self.model,
                    api_key=self.api_key,
                )
            else:
                raise ValueError(
                    f"Unsupported provider: {self.provider}. "
                    f"Available: openai, anthropic, azure"
                )

            logger.info(f"LLM provider initialized: {self.provider}/{self.model}")

        except Exception as e:
            logger.warning(f"Failed to create LLM provider: {e}. Using default.")
            self._llm = None

        return self._llm

    def _get_feedback_functions(self):
        """Feedback Functions 가져오기 (lazy loading)"""
        if self._feedback_functions is not None:
            return self._feedback_functions

        self._check_dependencies()

        try:
            from trulens_eval.feedback import Feedback, GroundTruthAgreement

            # LLM Provider
            llm = self._get_llm()

            # Feedback Functions 생성
            self._feedback_functions = {
                "context_relevance": Feedback(
                    provider=llm,
                    name="Context Relevance",
                ).on_input_output(),
                "groundedness": Feedback(
                    provider=llm,
                    name="Groundedness",
                ).on_input_output(),
                "answer_relevance": Feedback(
                    provider=llm,
                    name="Answer Relevance",
                ).on_input_output(),
            }

            logger.info("Feedback functions initialized")

        except Exception as e:
            logger.error(f"Failed to create feedback functions: {e}")
            self._feedback_functions = {}

        return self._feedback_functions

    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """
        평가 실행 (BaseEvaluationFramework 인터페이스)

        Args:
            **kwargs: 평가 파라미터
                - question: 질문
                - answer: 답변
                - contexts: 컨텍스트 리스트
                - metric: 메트릭 이름 (context_relevance, groundedness, answer_relevance, triad)

        Returns:
            평가 결과
        """
        metric = kwargs.get("metric", "triad")

        if metric == "triad":
            return self.evaluate_rag_triad(
                question=kwargs["question"],
                answer=kwargs["answer"],
                contexts=kwargs["contexts"],
            )
        elif metric == "context_relevance":
            return self.evaluate_context_relevance(
                question=kwargs["question"], contexts=kwargs["contexts"]
            )
        elif metric == "groundedness":
            return self.evaluate_groundedness(
                answer=kwargs["answer"], contexts=kwargs["contexts"]
            )
        elif metric == "answer_relevance":
            return self.evaluate_answer_relevance(
                question=kwargs["question"], answer=kwargs["answer"]
            )
        else:
            raise ValueError(
                f"Unknown metric: {metric}. "
                f"Available: context_relevance, groundedness, answer_relevance, triad"
            )

    def evaluate_rag_triad(
        self, question: str, answer: str, contexts: List[str]
    ) -> Dict[str, float]:
        """
        RAG Triad 평가 (3가지 메트릭 한번에)

        Args:
            question: 질문
            answer: 답변
            contexts: 검색된 컨텍스트 리스트

        Returns:
            {
                "context_relevance": float,
                "groundedness": float,
                "answer_relevance": float
            }
        """
        self._check_dependencies()

        logger.info("Evaluating RAG Triad...")

        # 개별 메트릭 평가
        context_relevance = self.evaluate_context_relevance(question, contexts)[
            "context_relevance"
        ]
        groundedness = self.evaluate_groundedness(answer, contexts)["groundedness"]
        answer_relevance = self.evaluate_answer_relevance(question, answer)[
            "answer_relevance"
        ]

        result = {
            "context_relevance": context_relevance,
            "groundedness": groundedness,
            "answer_relevance": answer_relevance,
        }

        logger.info(
            f"RAG Triad: CR={context_relevance:.3f}, "
            f"G={groundedness:.3f}, AR={answer_relevance:.3f}"
        )

        return result

    def evaluate_context_relevance(
        self, question: str, contexts: List[str]
    ) -> Dict[str, float]:
        """
        Context Relevance 평가

        검색된 컨텍스트가 질문과 관련있는지 평가합니다.

        Args:
            question: 질문
            contexts: 검색된 컨텍스트 리스트

        Returns:
            {"context_relevance": float}  # 0-1 점수
        """
        self._check_dependencies()

        try:
            from trulens_eval.feedback.provider.openai import OpenAI as TruLensOpenAI

            # TruLens OpenAI Provider
            provider = TruLensOpenAI(model_engine=self.model, api_key=self.api_key)

            # Context Relevance 평가
            scores = []
            for context in contexts:
                score = provider.context_relevance(question=question, statement=context)
                scores.append(score)

            # 평균 점수
            avg_score = sum(scores) / len(scores) if scores else 0.0

            logger.info(f"Context Relevance: {avg_score:.3f}")

            return {"context_relevance": avg_score}

        except Exception as e:
            logger.error(f"Failed to evaluate context relevance: {e}")
            return {"context_relevance": 0.0}

    def evaluate_groundedness(
        self, answer: str, contexts: List[str]
    ) -> Dict[str, float]:
        """
        Groundedness 평가 (Hallucination 체크)

        답변이 컨텍스트에 근거하는지 평가합니다.

        Args:
            answer: 답변
            contexts: 컨텍스트 리스트

        Returns:
            {"groundedness": float}  # 0-1 점수
        """
        self._check_dependencies()

        try:
            from trulens_eval.feedback.provider.openai import OpenAI as TruLensOpenAI

            # TruLens OpenAI Provider
            provider = TruLensOpenAI(model_engine=self.model, api_key=self.api_key)

            # Groundedness 평가
            source = "\n\n".join(contexts)
            score = provider.groundedness_measure_with_cot_reasons(
                source=source, statement=answer
            )

            # score는 (점수, 이유) 튜플일 수 있음
            if isinstance(score, tuple):
                score = score[0]

            logger.info(f"Groundedness: {score:.3f}")

            return {"groundedness": float(score)}

        except Exception as e:
            logger.error(f"Failed to evaluate groundedness: {e}")
            return {"groundedness": 0.0}

    def evaluate_answer_relevance(
        self, question: str, answer: str
    ) -> Dict[str, float]:
        """
        Answer Relevance 평가

        답변이 질문에 적절한지 평가합니다.

        Args:
            question: 질문
            answer: 답변

        Returns:
            {"answer_relevance": float}  # 0-1 점수
        """
        self._check_dependencies()

        try:
            from trulens_eval.feedback.provider.openai import OpenAI as TruLensOpenAI

            # TruLens OpenAI Provider
            provider = TruLensOpenAI(model_engine=self.model, api_key=self.api_key)

            # Answer Relevance 평가
            score = provider.relevance(prompt=question, response=answer)

            logger.info(f"Answer Relevance: {score:.3f}")

            return {"answer_relevance": float(score)}

        except Exception as e:
            logger.error(f"Failed to evaluate answer relevance: {e}")
            return {"answer_relevance": 0.0}

    def list_tasks(self) -> Dict[str, str]:
        """
        사용 가능한 메트릭 목록

        Returns:
            {메트릭 이름: 설명}
        """
        return {
            "context_relevance": "검색된 컨텍스트가 질문과 관련있는지 평가",
            "groundedness": "답변이 컨텍스트에 근거하는지 평가 (Hallucination 방지)",
            "answer_relevance": "답변이 질문에 적절한지 평가",
            "triad": "RAG Triad 3가지 메트릭을 한번에 평가",
        }

    def track_app(self, app: Any):
        """
        LangChain/LlamaIndex 앱 추적

        Args:
            app: LangChain Chain 또는 LlamaIndex QueryEngine

        Returns:
            TruChain 또는 TruLlama 래퍼
        """
        self._check_dependencies()

        try:
            from trulens_eval import TruChain

            # Feedback Functions
            feedbacks = self._get_feedback_functions()

            # TruChain 래퍼
            tru_app = TruChain(app, app_id="beanllm_app", feedbacks=list(feedbacks.values()))

            logger.info("App tracking enabled")

            return tru_app

        except Exception as e:
            logger.error(f"Failed to track app: {e}")
            return app

    def show_dashboard(self):
        """
        TruLens 대시보드 실행

        브라우저에서 http://localhost:8501 에 대시보드가 열립니다.
        """
        if not self.enable_dashboard:
            logger.warning(
                "Dashboard is disabled. Set enable_dashboard=True to enable."
            )
            return

        self._check_dependencies()

        try:
            from trulens_eval import Tru

            tru = Tru()
            tru.run_dashboard()

            logger.info("Dashboard started at http://localhost:8501")

        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")

    def __repr__(self) -> str:
        return (
            f"TruLensWrapper(provider={self.provider}, "
            f"model={self.model}, dashboard={self.enable_dashboard})"
        )
