"""
Rerankers - 재순위화 모델 구현체들
"""

import os
from typing import List, Optional

from .base import BaseReranker
from .types import RerankResult

try:
    from beanllm.utils.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class BGEReranker(BaseReranker):
    """
    BGE Reranker v2 (BAAI, 2024-2025)

    BAAI의 최신 재순위화 모델로 BEIR, MIRACL 등 벤치마크에서 대폭 개선되었습니다.

    모델 라인업 (추천 순):
    - BAAI/bge-reranker-v2-m3: 다국어 최강 (100+ 언어)
    - BAAI/bge-reranker-v2-gemma: LLM 백본 (높은 성능)
    - BAAI/bge-reranker-v2-minicpm-layerwise: 중국어/영어 특화
    - BAAI/bge-reranker-base: 경량 (빠른 속도)
    - BAAI/bge-reranker-large: 고성능

    Features:
    - Cross-encoder 아키텍처 (bi-encoder보다 깊은 이해)
    - 다국어 지원 (m3 모델)
    - 최대 입력 크기 확장
    - BEIR, C-MTEB 벤치마크 SOTA

    Example:
        ```python
        from beanllm.domain.retrieval import BGEReranker

        # 다국어 모델 (추천)
        reranker = BGEReranker(model="BAAI/bge-reranker-v2-m3")
        results = reranker.rerank(
            query="What is machine learning?",
            documents=[
                "ML is a subset of AI...",
                "Python is a programming language...",
                "Deep learning uses neural networks..."
            ],
            top_k=2
        )

        for result in results:
            print(f"Score: {result.score:.4f}, Text: {result.text[:50]}")
        # Score: 0.9823, Text: ML is a subset of AI...
        # Score: 0.7654, Text: Deep learning uses neural networks...
        ```
    """

    def __init__(
        self,
        model: str = "BAAI/bge-reranker-v2-m3",
        use_gpu: bool = True,
        batch_size: int = 32,
        max_length: int = 512,
        **kwargs,
    ):
        """
        Args:
            model: BGE Reranker 모델
                - BAAI/bge-reranker-v2-m3: 다국어 (기본값, 추천)
                - BAAI/bge-reranker-v2-gemma: LLM 백본
                - BAAI/bge-reranker-v2-minicpm-layerwise: 중국어/영어
                - BAAI/bge-reranker-base: 경량
                - BAAI/bge-reranker-large: 고성능
            use_gpu: GPU 사용 여부 (기본: True)
            batch_size: 배치 크기 (기본: 32)
            max_length: 최대 토큰 길이 (기본: 512)
            **kwargs: 추가 파라미터
        """
        self.model_name = model
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.max_length = max_length
        self.kwargs = kwargs

        # Lazy loading
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load_model(self):
        """모델 로딩 (lazy loading)"""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers and torch required for BGEReranker. "
                "Install: pip install transformers torch"
            )

        # Device 설정
        if self.use_gpu and torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        logger.info(f"Loading BGE Reranker: {self.model_name} on {self._device}")

        # 모델 및 토크나이저 로드
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self._device)
        self._model.eval()

        logger.info(f"BGE Reranker loaded: {self.model_name}")

    def rerank(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        문서를 재순위화

        Args:
            query: 검색 쿼리
            documents: 재순위화할 문서 리스트
            top_k: 반환할 상위 k개 (None이면 전체)

        Returns:
            재순위화된 결과 (점수 내림차순)
        """
        # 모델 로드
        self._load_model()

        try:
            import torch

            # 쿼리-문서 페어 생성
            pairs = [[query, doc] for doc in documents]

            # 배치 처리
            all_scores = []

            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i : i + self.batch_size]

                # Tokenization
                inputs = self._tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self._device)

                # Forward pass
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    scores = outputs.logits.squeeze(-1)

                    # Sigmoid (확률로 변환)
                    if scores.dim() == 0:
                        scores = scores.unsqueeze(0)

                    # CPU로 이동
                    scores = scores.cpu().tolist()
                    if isinstance(scores, float):
                        scores = [scores]

                    all_scores.extend(scores)

            # RerankResult 생성
            results = [
                RerankResult(text=doc, score=float(score), index=idx)
                for idx, (doc, score) in enumerate(zip(documents, all_scores))
            ]

            # 점수로 정렬
            results.sort(key=lambda x: x.score, reverse=True)

            # Top-k 선택
            if top_k is not None:
                results = results[:top_k]

            logger.info(
                f"Reranked {len(documents)} documents, top score: {results[0].score:.4f}"
            )

            return results

        except Exception as e:
            logger.error(f"BGE Reranker failed: {e}")
            raise


class CohereReranker(BaseReranker):
    """
    Cohere Rerank (2024-2025)

    Cohere의 최신 재순위화 모델로 100개 이상의 언어를 지원합니다.

    모델:
    - rerank-3-nimble: 프로덕션용 고속 (기본값)
    - rerank-4: 32K context (2024년 12월 최신)
        - 3.5 대비 4배 context window
        - 최초의 self-learning reranker
        - 추가 라벨링 없이 사용 사례 맞춤화 가능

    Features:
    - 100+ 언어 지원
    - 32K context window (rerank-4)
    - Self-learning (rerank-4)
    - 프로덕션급 속도 (nimble)

    Example:
        ```python
        from beanllm.domain.retrieval import CohereReranker

        # Rerank 3 Nimble (고속)
        reranker = CohereReranker(model="rerank-3-nimble", api_key="...")
        results = reranker.rerank(
            query="What is AI?",
            documents=["AI is...", "Python is...", "ML is..."],
            top_k=2
        )

        # Rerank 4 (최신, 32K context)
        reranker = CohereReranker(model="rerank-4", api_key="...")
        results = reranker.rerank(query="...", documents=long_docs, top_k=5)
        ```
    """

    def __init__(
        self,
        model: str = "rerank-3-nimble",
        api_key: Optional[str] = None,
        max_chunks_per_doc: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            model: Cohere rerank 모델
                - rerank-3-nimble: 고속 (기본값)
                - rerank-4: 32K context, self-learning (최신)
                - rerank-english-v3.0: 영어 특화
                - rerank-multilingual-v3.0: 다국어
            api_key: Cohere API 키 (None이면 환경변수)
            max_chunks_per_doc: 문서당 최대 청크 수 (긴 문서용)
            **kwargs: 추가 파라미터
        """
        self.model = model
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.max_chunks_per_doc = max_chunks_per_doc
        self.kwargs = kwargs

        if not self.api_key:
            raise ValueError("COHERE_API_KEY not found in environment variables")

        # Cohere 클라이언트
        self._client = None

    def _get_client(self):
        """Cohere 클라이언트 가져오기 (lazy)"""
        if self._client is not None:
            return self._client

        try:
            import cohere
        except ImportError:
            raise ImportError("cohere required for CohereReranker. Install: pip install cohere")

        self._client = cohere.Client(api_key=self.api_key)
        return self._client

    def rerank(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        문서를 재순위화

        Args:
            query: 검색 쿼리
            documents: 재순위화할 문서 리스트
            top_k: 반환할 상위 k개 (None이면 전체)

        Returns:
            재순위화된 결과 (점수 내림차순)
        """
        client = self._get_client()

        try:
            # Cohere rerank API 호출
            response = client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_k if top_k else len(documents),
                max_chunks_per_doc=self.max_chunks_per_doc,
                **self.kwargs,
            )

            # RerankResult 생성
            results = [
                RerankResult(
                    text=documents[result.index],
                    score=result.relevance_score,
                    index=result.index,
                )
                for result in response.results
            ]

            logger.info(
                f"Reranked {len(documents)} documents with Cohere {self.model}, "
                f"top score: {results[0].score:.4f}"
            )

            return results

        except Exception as e:
            logger.error(f"Cohere Reranker failed: {e}")
            raise


class CrossEncoderReranker(BaseReranker):
    """
    범용 Cross-Encoder Reranker

    HuggingFace의 모든 cross-encoder 모델을 지원합니다.

    추천 모델:
    - cross-encoder/ms-marco-MiniLM-L-6-v2: 경량 (빠름)
    - cross-encoder/ms-marco-MiniLM-L-12-v2: 균형
    - cross-encoder/ms-marco-electra-base: 고성능

    Example:
        ```python
        from beanllm.domain.retrieval import CrossEncoderReranker

        reranker = CrossEncoderReranker(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        results = reranker.rerank(
            query="What is Python?",
            documents=["Python is a language...", "Java is..."],
            top_k=1
        )
        ```
    """

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_gpu: bool = True,
        batch_size: int = 32,
        **kwargs,
    ):
        """
        Args:
            model: HuggingFace cross-encoder 모델
            use_gpu: GPU 사용 여부
            batch_size: 배치 크기
            **kwargs: 추가 파라미터
        """
        self.model_name = model
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.kwargs = kwargs

        # Lazy loading
        self._model = None

    def _load_model(self):
        """모델 로딩"""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers required. Install: pip install sentence-transformers"
            )

        device = "cuda" if self.use_gpu else "cpu"
        self._model = CrossEncoder(self.model_name, device=device)

        logger.info(f"CrossEncoder loaded: {self.model_name} on {device}")

    def rerank(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """문서를 재순위화"""
        self._load_model()

        try:
            # 쿼리-문서 페어
            pairs = [[query, doc] for doc in documents]

            # 점수 계산
            scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

            # RerankResult 생성
            results = [
                RerankResult(text=doc, score=float(score), index=idx)
                for idx, (doc, score) in enumerate(zip(documents, scores))
            ]

            # 정렬
            results.sort(key=lambda x: x.score, reverse=True)

            # Top-k
            if top_k is not None:
                results = results[:top_k]

            logger.info(
                f"Reranked {len(documents)} documents with CrossEncoder, "
                f"top score: {results[0].score:.4f}"
            )

            return results

        except Exception as e:
            logger.error(f"CrossEncoder Reranker failed: {e}")
            raise


class PositionEngineeringReranker(BaseReranker):
    """
    Position Engineering Reranker (2024-2025)

    LLM의 "Lost in the Middle" 문제를 해결하기 위한 문서 재배치 전략입니다.

    연구 배경 (Liu et al., 2023):
    - LLM은 긴 컨텍스트의 중간 부분에 있는 정보를 잘 활용하지 못함
    - 중요한 정보를 앞(head) 또는 뒤(tail)에 배치하면 성능 향상
    - 추가 비용 없이 10-30% 성능 개선 가능

    Position Strategies:
    - head: 중요한 문서를 앞에 배치 (기본값, 가장 일반적)
    - tail: 중요한 문서를 뒤에 배치
    - head_tail: 가장 중요한 문서를 앞에, 두 번째로 중요한 것을 뒤에
        - 예: [1st, 3rd, 5th, ..., 6th, 4th, 2nd]
    - side: 중요한 문서를 양쪽 끝에 번갈아 배치
        - 예: [1st, 4th, 6th, ..., 7th, 5th, 3rd, 2nd]

    Features:
    - 다른 reranker와 함께 사용 가능 (wrapper pattern)
    - 무료 성능 향상 (추가 비용 없음)
    - 다양한 배치 전략 지원

    Example:
        ```python
        from beanllm.domain.retrieval import (
            BGEReranker,
            PositionEngineeringReranker
        )

        # BGE Reranker로 점수 계산 후 Position Engineering 적용
        base_reranker = BGEReranker(model="BAAI/bge-reranker-v2-m3")
        reranker = PositionEngineeringReranker(
            base_reranker=base_reranker,
            strategy="head_tail"
        )

        results = reranker.rerank(
            query="What is machine learning?",
            documents=[...],
            top_k=5
        )
        # [1st, 3rd, 5th, 4th, 2nd] 순서로 재배치됨

        # 단독 사용 (점수 기반으로만 재배치)
        reranker = PositionEngineeringReranker(strategy="head")
        results = reranker.rerank(
            query="...",
            documents=[...],
            scores=[0.9, 0.7, 0.5, 0.3]  # 미리 계산된 점수
        )
        ```

    References:
        - Liu et al. (2023): "Lost in the Middle: How Language Models Use Long Contexts"
        - https://arxiv.org/abs/2307.03172
    """

    def __init__(
        self,
        base_reranker: Optional[BaseReranker] = None,
        strategy: str = "head",
        **kwargs,
    ):
        """
        Args:
            base_reranker: 기본 reranker (점수 계산용)
                None이면 입력된 scores를 사용하거나 원본 순서 유지
            strategy: 배치 전략
                - "head": 중요한 문서를 앞에 (기본값)
                - "tail": 중요한 문서를 뒤에
                - "head_tail": 중요한 것을 앞뒤에
                - "side": 양쪽 끝에 번갈아 배치
            **kwargs: 추가 파라미터
        """
        self.base_reranker = base_reranker
        self.strategy = strategy.lower()
        self.kwargs = kwargs

        # 유효한 전략인지 확인
        valid_strategies = ["head", "tail", "head_tail", "side"]
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"Invalid strategy: {self.strategy}. "
                f"Available: {valid_strategies}"
            )

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        scores: Optional[List[float]] = None,
    ) -> List[RerankResult]:
        """
        문서를 재순위화 및 재배치

        Args:
            query: 검색 쿼리
            documents: 재순위화할 문서 리스트
            top_k: 반환할 상위 k개 (None이면 전체)
            scores: 미리 계산된 점수 (base_reranker가 없을 때 사용)

        Returns:
            Position Engineering이 적용된 재순위화 결과
        """
        # 1. 기본 reranker로 점수 계산 (있는 경우)
        if self.base_reranker is not None:
            # Base reranker로 점수 계산
            ranked_results = self.base_reranker.rerank(
                query=query,
                documents=documents,
                top_k=top_k,
            )
        elif scores is not None:
            # 미리 계산된 점수 사용
            if len(scores) != len(documents):
                raise ValueError("scores와 documents의 길이가 일치하지 않습니다.")

            # RerankResult 생성
            ranked_results = [
                RerankResult(text=doc, score=float(score), index=idx)
                for idx, (doc, score) in enumerate(zip(documents, scores))
            ]

            # 점수로 정렬
            ranked_results.sort(key=lambda x: x.score, reverse=True)

            # Top-k 선택
            if top_k is not None:
                ranked_results = ranked_results[:top_k]
        else:
            # 점수 없이 원본 순서 유지
            ranked_results = [
                RerankResult(text=doc, score=1.0 / (idx + 1), index=idx)
                for idx, doc in enumerate(documents)
            ]

            if top_k is not None:
                ranked_results = ranked_results[:top_k]

        # 2. Position Engineering 적용
        reordered = self._apply_position_engineering(ranked_results)

        logger.info(
            f"Position Engineering applied: strategy={self.strategy}, "
            f"count={len(reordered)}"
        )

        return reordered

    def _apply_position_engineering(
        self, results: List[RerankResult]
    ) -> List[RerankResult]:
        """
        Position Engineering 전략 적용

        Args:
            results: 점수로 정렬된 결과 (내림차순)

        Returns:
            재배치된 결과
        """
        n = len(results)

        if n == 0:
            return results

        if self.strategy == "head":
            # 가장 간단: 점수 순서대로 (이미 정렬됨)
            return results

        elif self.strategy == "tail":
            # 역순 (중요한 것을 뒤에)
            return results[::-1]

        elif self.strategy == "head_tail":
            # 중요한 것을 앞뒤에 번갈아 배치
            # 예: [1st, 3rd, 5th, ..., 6th, 4th, 2nd]
            reordered = []
            left = []
            right = []

            for i, result in enumerate(results):
                if i % 2 == 0:
                    # 짝수 인덱스 (1st, 3rd, 5th, ...) -> 앞에
                    left.append(result)
                else:
                    # 홀수 인덱스 (2nd, 4th, 6th, ...) -> 뒤에
                    right.append(result)

            # 왼쪽 + 오른쪽 역순
            reordered = left + right[::-1]
            return reordered

        elif self.strategy == "side":
            # 양쪽 끝에 번갈아 배치
            # 예: [1st, 4th, 6th, ..., 7th, 5th, 3rd, 2nd]
            reordered = [None] * n

            # 앞에서부터 채우기
            front_idx = 0
            # 뒤에서부터 채우기
            back_idx = n - 1

            for i, result in enumerate(results):
                if i % 2 == 0:
                    # 앞에 배치
                    reordered[front_idx] = result
                    front_idx += 1
                else:
                    # 뒤에 배치
                    reordered[back_idx] = result
                    back_idx -= 1

            return reordered

        else:
            # 폴백 (안전장치)
            return results

    def __repr__(self) -> str:
        base_name = (
            self.base_reranker.__class__.__name__
            if self.base_reranker
            else "None"
        )
        return (
            f"PositionEngineeringReranker("
            f"base={base_name}, strategy={self.strategy})"
        )
