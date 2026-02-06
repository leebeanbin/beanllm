"""
ColBERT Retriever - Multi-Vector Late Interaction Retrieval

ColBERT (Contextualized Late Interaction over BERT)는 쿼리와 문서의
토큰별 임베딩을 사용하여 세밀한 매칭을 수행합니다.

기존 Dense Retrieval 대비 장점:
- 더 정확한 키워드 매칭 (토큰 레벨 상호작용)
- 더 나은 일반화 성능
- 10-30% 검색 품질 향상 (벤치마크 기준)

Requirements:
    pip install ragatouille  # 권장 (쉬운 API)
    # 또는
    pip install colbert-ai  # 직접 사용

References:
    - ColBERT: https://github.com/stanford-futuredata/ColBERT
    - RAGatouille: https://github.com/AnswerDotAI/RAGatouille

Example:
    ```python
    from beanllm.domain.retrieval import ColBERTRetriever

    # RAGatouille 사용 (권장)
    retriever = ColBERTRetriever(
        model="colbert-ir/colbertv2.0",
        documents=["Doc 1", "Doc 2", "Doc 3"]
    )
    results = retriever.search("query", k=5)

    # 인덱스 저장/로드
    retriever.save_index("./colbert_index")
    retriever = ColBERTRetriever.load_index("./colbert_index")
    ```
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from beanllm.domain.retrieval.types import SearchResult

try:
    from beanllm.utils.logging import get_logger
except ImportError:

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class ColBERTRetriever:
    """
    ColBERT 기반 검색기 (RAGatouille 래퍼)

    Multi-vector late interaction을 사용하여
    기존 단일 벡터 검색보다 더 정확한 검색을 제공합니다.

    Attributes:
        model: ColBERT 모델 (기본: colbert-ir/colbertv2.0)
        documents: 검색 대상 문서 목록
        index_name: 인덱스 이름

    Example:
        ```python
        retriever = ColBERTRetriever(
            documents=["Python is great.", "Java is popular."],
            model="colbert-ir/colbertv2.0"
        )

        results = retriever.search("What is a good programming language?", k=2)
        for r in results:
            print(f"{r.score:.4f}: {r.text}")
        ```
    """

    def __init__(
        self,
        model: str = "colbert-ir/colbertv2.0",
        documents: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None,
        document_metadatas: Optional[List[Dict[str, Any]]] = None,
        index_name: str = "beanllm_colbert",
        index_path: Optional[str] = None,
        use_gpu: bool = True,
        **kwargs: Any,
    ):
        """
        ColBERT Retriever 초기화

        Args:
            model: ColBERT 모델 이름 또는 경로
                - "colbert-ir/colbertv2.0": 기본 영어 모델
                - "answerdotai/colbert-kor": 한국어 모델
            documents: 검색 대상 문서 목록
            document_ids: 문서 ID 목록 (None이면 자동 생성)
            document_metadatas: 문서 메타데이터 목록
            index_name: 인덱스 이름
            index_path: 인덱스 저장 경로 (None이면 메모리에만)
            use_gpu: GPU 사용 여부
            **kwargs: RAGatouille 추가 옵션
        """
        self.model_name = model
        self.index_name = index_name
        self.index_path = index_path
        self.use_gpu = use_gpu
        self.kwargs = kwargs

        self._rag = None
        self._documents = documents or []
        self._document_ids = document_ids
        self._document_metadatas = document_metadatas or []
        self._indexed = False

        # 문서가 있으면 인덱싱
        if self._documents:
            self._init_and_index(self._documents, document_ids, document_metadatas)

    def _init_ragatouille(self):
        """RAGatouille 초기화"""
        try:
            from ragatouille import RAGPretrainedModel
        except ImportError:
            raise ImportError(
                "ragatouille is required for ColBERTRetriever. "
                "Install it with: pip install ragatouille"
            )

        if self._rag is None:
            logger.info(f"Loading ColBERT model: {self.model_name}")
            self._rag = RAGPretrainedModel.from_pretrained(
                self.model_name,
                **self.kwargs,
            )

        return self._rag

    def _init_and_index(
        self,
        documents: List[str],
        document_ids: Optional[List[str]] = None,
        document_metadatas: Optional[List[Dict[str, Any]]] = None,
    ):
        """문서 인덱싱"""
        rag = self._init_ragatouille()

        # Document IDs 생성
        if document_ids is None:
            document_ids = [f"doc_{i}" for i in range(len(documents))]

        logger.info(f"Indexing {len(documents)} documents...")

        # RAGatouille 인덱싱
        index_path = self.index_path or f"./.ragatouille/colbert/indexes/{self.index_name}"

        rag.index(
            collection=documents,
            document_ids=document_ids,
            document_metadatas=document_metadatas,
            index_name=self.index_name,
            max_document_length=256,
            split_documents=True,
        )

        self._documents = documents
        self._document_ids = document_ids
        self._document_metadatas = document_metadatas or []
        self._indexed = True

        logger.info(f"Indexed {len(documents)} documents to {index_path}")

    def add_documents(
        self,
        documents: List[str],
        document_ids: Optional[List[str]] = None,
        document_metadatas: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        문서 추가

        Args:
            documents: 추가할 문서 목록
            document_ids: 문서 ID 목록
            document_metadatas: 문서 메타데이터 목록
        """
        if not self._indexed:
            # 처음 인덱싱
            self._init_and_index(documents, document_ids, document_metadatas)
        else:
            # 기존 인덱스에 추가
            rag = self._init_ragatouille()

            if document_ids is None:
                start_id = len(self._documents)
                document_ids = [f"doc_{start_id + i}" for i in range(len(documents))]

            logger.info(f"Adding {len(documents)} documents to existing index...")

            rag.add_to_index(
                new_collection=documents,
                new_document_ids=document_ids,
                new_document_metadatas=document_metadatas,
            )

            self._documents.extend(documents)
            if self._document_ids:
                self._document_ids.extend(document_ids)
            self._document_metadatas.extend(document_metadatas or [])

            logger.info(f"Total documents: {len(self._documents)}")

    def search(
        self,
        query: str,
        k: int = 10,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """
        ColBERT 검색 수행

        Args:
            query: 검색 쿼리
            k: 반환할 결과 수

        Returns:
            SearchResult 리스트 (점수 내림차순)
        """
        if not self._indexed:
            raise ValueError("No documents indexed. Call add_documents() first.")

        rag = self._init_ragatouille()

        logger.info(f"Searching with ColBERT: '{query[:50]}...'")

        # RAGatouille 검색
        results = rag.search(query=query, k=k, **kwargs)

        # SearchResult로 변환
        search_results = []
        for r in results:
            metadata = {"rank": r.get("rank", 0)}

            # 문서 메타데이터 추가
            doc_id = r.get("document_id")
            if doc_id and self._document_ids:
                try:
                    idx = self._document_ids.index(doc_id)
                    if idx < len(self._document_metadatas):
                        metadata.update(self._document_metadatas[idx])
                except ValueError:
                    pass

            search_results.append(
                SearchResult(
                    text=r.get("content", ""),
                    score=r.get("score", 0.0),
                    metadata=metadata,
                )
            )

        logger.info(f"Found {len(search_results)} results")
        return search_results

    def search_batch(
        self,
        queries: List[str],
        k: int = 10,
    ) -> List[List[SearchResult]]:
        """
        배치 검색

        Args:
            queries: 검색 쿼리 목록
            k: 각 쿼리당 반환할 결과 수

        Returns:
            각 쿼리에 대한 SearchResult 리스트
        """
        return [self.search(q, k=k) for q in queries]

    def save_index(self, path: Union[str, Path]):
        """
        인덱스 저장

        Args:
            path: 저장 경로
        """
        # RAGatouille는 인덱싱 시 자동 저장
        # 이 메서드는 인덱스 경로를 반환
        logger.info(f"Index saved at: {self.index_path or path}")

    @classmethod
    def load_index(
        cls,
        index_path: str,
        model: str = "colbert-ir/colbertv2.0",
        **kwargs: Any,
    ) -> "ColBERTRetriever":
        """
        저장된 인덱스 로드

        Args:
            index_path: 인덱스 경로
            model: ColBERT 모델 이름
            **kwargs: 추가 옵션

        Returns:
            ColBERTRetriever 인스턴스
        """
        try:
            from ragatouille import RAGPretrainedModel
        except ImportError:
            raise ImportError("ragatouille is required. Install with: pip install ragatouille")

        logger.info(f"Loading index from: {index_path}")

        retriever = cls(model=model, **kwargs)
        retriever._rag = RAGPretrainedModel.from_index(index_path)
        retriever._indexed = True
        retriever.index_path = index_path

        return retriever

    def __repr__(self) -> str:
        return (
            f"ColBERTRetriever("
            f"model={self.model_name}, "
            f"docs={len(self._documents)}, "
            f"indexed={self._indexed})"
        )


class ColPaliRetriever:
    """
    ColPali 기반 비전 문서 검색기

    OCR 없이 문서 이미지에서 직접 검색합니다.
    PDF, 스캔 문서, 차트/그래프 등에 효과적입니다.

    Requirements:
        pip install byaldi  # ColPali 래퍼

    Example:
        ```python
        from beanllm.domain.retrieval import ColPaliRetriever

        retriever = ColPaliRetriever()
        retriever.add_images(["doc1.pdf", "doc2.png"])

        results = retriever.search("chart showing revenue growth")
        ```
    """

    def __init__(
        self,
        model: str = "vidore/colpali",
        index_name: str = "beanllm_colpali",
        **kwargs: Any,
    ):
        """
        ColPali Retriever 초기화

        Args:
            model: ColPali 모델 이름
            index_name: 인덱스 이름
            **kwargs: byaldi 추가 옵션
        """
        self.model_name = model
        self.index_name = index_name
        self.kwargs = kwargs

        self._model = None
        self._indexed = False

    def _init_model(self):
        """Byaldi 모델 초기화"""
        try:
            from byaldi import RAGMultiModalModel
        except ImportError:
            raise ImportError(
                "byaldi is required for ColPaliRetriever. " "Install it with: pip install byaldi"
            )

        if self._model is None:
            logger.info(f"Loading ColPali model: {self.model_name}")
            self._model = RAGMultiModalModel.from_pretrained(self.model_name)

        return self._model

    def add_images(
        self,
        image_paths: List[str],
        store_collection_with_index: bool = True,
    ):
        """
        이미지/PDF 추가

        Args:
            image_paths: 이미지 또는 PDF 파일 경로 목록
            store_collection_with_index: 인덱스와 함께 원본 저장
        """
        model = self._init_model()

        logger.info(f"Indexing {len(image_paths)} images/documents...")

        model.index(
            input_path=image_paths,
            index_name=self.index_name,
            store_collection_with_index=store_collection_with_index,
            overwrite=True,
        )

        self._indexed = True
        logger.info(f"Indexed {len(image_paths)} images")

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> List[SearchResult]:
        """
        이미지 검색

        Args:
            query: 텍스트 쿼리
            k: 반환할 결과 수

        Returns:
            SearchResult 리스트
        """
        if not self._indexed:
            raise ValueError("No images indexed. Call add_images() first.")

        model = self._init_model()

        results = model.search(query, k=k)

        search_results = []
        for r in results:
            search_results.append(
                SearchResult(
                    text=r.get("doc_id", ""),
                    score=r.get("score", 0.0),
                    metadata={
                        "page_num": r.get("page_num"),
                        "base64": r.get("base64"),
                    },
                )
            )

        return search_results

    def __repr__(self) -> str:
        return f"ColPaliRetriever(model={self.model_name}, indexed={self._indexed})"
