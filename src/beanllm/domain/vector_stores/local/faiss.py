"""
FAISS Vector Store Implementation

Facebook AI Similarity Search
"""

import uuid
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from beanllm.domain.loaders import Document
else:
    try:
        from beanllm.domain.loaders import Document
    except ImportError:
        Document = Any  # type: ignore

from beanllm.domain.vector_stores.base import BaseVectorStore, VectorSearchResult
from beanllm.domain.vector_stores.search import AdvancedSearchMixin
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


_SUPPORTED_INDEX_TYPES = frozenset(
    {
        "IndexFlatL2",
        "IndexFlatIP",
        "IndexHNSWFlat",
        "IndexIVFFlat",
        "auto",
    }
)

# 자동 인덱스 선택 임계값
_ANN_THRESHOLD = 10_000
_HNSW_DEFAULT_M = 32
_IVF_DEFAULT_NLIST = 100


class FAISSVectorStore(BaseVectorStore, AdvancedSearchMixin):
    """
    FAISS vector store - 로컬, 매우 빠름

    index_type 옵션:
        - "IndexFlatL2": 브루트포스 L2 거리 (기본, 정확도 100%)
        - "IndexFlatIP": 브루트포스 내적 (코사인 유사도)
        - "IndexHNSWFlat": HNSW 근사 검색 (빠름, n > 10K 추천)
        - "IndexIVFFlat": IVF 근사 검색 (메모리 효율, n > 100K 추천)
        - "auto": 문서 수에 따라 자동 선택
    """

    def __init__(
        self,
        embedding_function=None,
        dimension: int = 1536,
        index_type: str = "IndexFlatL2",
        **kwargs,
    ):
        super().__init__(embedding_function)

        try:
            import faiss
            import numpy as np
        except ImportError:
            raise ImportError("FAISS not installed. pip install faiss-cpu  # or faiss-gpu")

        if index_type not in _SUPPORTED_INDEX_TYPES:
            raise ValueError(
                f"Unknown index type: {index_type}. "
                f"Supported: {', '.join(sorted(_SUPPORTED_INDEX_TYPES))}"
            )

        self.faiss = faiss
        self.np = np
        self.dimension = dimension
        self.index_type = index_type
        self._is_trained = False

        # FAISS 인덱스 생성
        self.index = self._create_index(index_type, dimension)

    def _create_index(self, index_type: str, dimension: int) -> Any:
        """
        인덱스 타입에 따라 FAISS 인덱스를 생성합니다.

        Args:
            index_type: 인덱스 타입 문자열
            dimension: 벡터 차원

        Returns:
            FAISS 인덱스 객체
        """
        if index_type == "IndexFlatL2":
            return self.faiss.IndexFlatL2(dimension)
        elif index_type == "IndexFlatIP":
            return self.faiss.IndexFlatIP(dimension)
        elif index_type == "IndexHNSWFlat":
            # HNSW: 그래프 기반 ANN (높은 recall, 빠른 검색)
            return self.faiss.IndexHNSWFlat(dimension, _HNSW_DEFAULT_M)
        elif index_type == "IndexIVFFlat":
            # IVF: 역인덱스 기반 ANN (training 필요)
            quantizer = self.faiss.IndexFlatL2(dimension)
            index = self.faiss.IndexIVFFlat(quantizer, dimension, _IVF_DEFAULT_NLIST)
            self._is_trained = False
            return index
        else:
            # auto 또는 기타: 기본 Flat
            return self.faiss.IndexFlatL2(dimension)

        self.documents = []  # 문서 저장
        self.ids_to_index = {}  # ID -> index 매핑

    def add_documents(self, documents: List[Any], **kwargs) -> List[str]:
        """문서 추가"""
        texts = [doc.content for doc in documents]

        # 임베딩 생성
        if not self.embedding_function:
            raise ValueError("Embedding function required for FAISS")

        embeddings = self.embedding_function(texts)

        # numpy array로 변환
        embeddings_array = self.np.array(embeddings).astype("float32")

        # auto 모드: 문서 수에 따라 인덱스 업그레이드
        total_docs = len(self.documents) + len(documents)
        if (
            self.index_type == "auto"
            and total_docs >= _ANN_THRESHOLD
            and not isinstance(self.index, self.faiss.IndexHNSWFlat)
        ):
            logger.info(f"Auto-upgrading FAISS index to HNSW (total docs: {total_docs})")
            self._upgrade_to_hnsw(embeddings_array)

        # IVF 인덱스인 경우 training 필요
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            if embeddings_array.shape[0] >= _IVF_DEFAULT_NLIST:
                self.index.train(embeddings_array)
                self._is_trained = True
            else:
                logger.warning(
                    f"Not enough data to train IVF index "
                    f"({embeddings_array.shape[0]} < {_IVF_DEFAULT_NLIST}). "
                    f"Falling back to FlatL2."
                )
                self.index = self.faiss.IndexFlatL2(self.dimension)

        # ID 생성
        ids = [str(uuid.uuid4()) for _ in texts]

        # 인덱스에 추가
        start_idx = len(self.documents)
        self.index.add(embeddings_array)

        # 문서 및 매핑 저장
        for i, (doc, id_) in enumerate(zip(documents, ids)):
            self.documents.append(doc)
            self.ids_to_index[id_] = start_idx + i

        return ids

    def _upgrade_to_hnsw(self, new_embeddings: Any) -> None:
        """기존 Flat 인덱스를 HNSW로 업그레이드합니다."""
        new_index = self.faiss.IndexHNSWFlat(self.dimension, _HNSW_DEFAULT_M)

        # 기존 벡터를 새 인덱스로 이동
        if self.index.ntotal > 0:
            existing = self.faiss.rev_swig_ptr(
                self.index.get_xb(), self.index.ntotal * self.dimension
            )
            existing = self.np.array(existing).reshape(self.index.ntotal, self.dimension)
            new_index.add(existing.astype("float32"))

        self.index = new_index
        self.index_type = "IndexHNSWFlat"

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[VectorSearchResult]:
        """유사도 검색"""
        if not self.embedding_function:
            raise ValueError("Embedding function required for FAISS")

        # 쿼리 임베딩
        query_embedding = self.embedding_function([query])[0]
        query_array = self.np.array([query_embedding]).astype("float32")

        # 검색
        distances, indices = self.index.search(query_array, k)

        # 결과 변환
        search_results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                # L2 distance -> similarity score
                score = 1 / (1 + distances[0][i])
                search_results.append(
                    VectorSearchResult(document=doc, score=score, metadata=doc.metadata)
                )

        return search_results

    def _get_all_vectors_and_docs(self) -> tuple[List[List[float]], List[Any]]:
        """FAISS에서 모든 벡터 가져오기"""
        if not self.documents:
            return [], []

        # FAISS 인덱스에서 모든 벡터 가져오기
        try:
            # FAISS는 직접 벡터를 가져올 수 없으므로 문서에서 재임베딩
            # 또는 인덱스를 재구축해야 함
            # 여기서는 간단히 빈 리스트 반환 (배치 검색은 비효율적)
            # 실제로는 인덱스에 벡터를 저장해야 함
            return [], []
        except Exception:
            return [], []

    def reset(self):
        """
        인덱스 초기화 (메모리 누수 방지)

        기존 인덱스와 문서를 모두 삭제하고 새로운 인덱스를 생성합니다.
        """
        # 기존 인덱스 명시적 삭제
        if hasattr(self, "index"):
            del self.index

        # 새 인덱스 생성
        index_type = getattr(self, "index_type", "IndexFlatL2")
        self.index = self._create_index(index_type, self.dimension)
        self._is_trained = False

        # 문서 및 매핑 초기화
        self.documents = []
        self.ids_to_index = {}

    def close(self):
        """
        리소스 정리 (메모리 누수 방지)

        FAISS 인덱스와 관련 데이터 구조를 명시적으로 정리합니다.
        """
        # 인덱스 삭제
        if hasattr(self, "index"):
            del self.index

        # 문서 리스트 정리
        if hasattr(self, "documents"):
            self.documents.clear()

        # ID 매핑 정리
        if hasattr(self, "ids_to_index"):
            self.ids_to_index.clear()

        # GC 강제 실행 (선택적)
        import gc

        gc.collect()

    def __del__(self):
        """소멸자 - 리소스 자동 정리"""
        try:
            self.close()
        except Exception as e:
            logger.debug(f"Close in destructor failed (safe to ignore): {e}")

    async def asimilarity_search_by_vector(
        self, query_vec: List[float], k: int = 4, **kwargs
    ) -> List[VectorSearchResult]:
        """벡터로 직접 검색"""
        query_array = self.np.array([query_vec]).astype("float32")
        distances, indices = self.index.search(query_array, k)

        search_results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                score = 1 / (1 + distances[0][i])
                search_results.append(
                    VectorSearchResult(document=doc, score=score, metadata=doc.metadata)
                )
        return search_results

    def delete(self, ids: List[str], **kwargs) -> bool:
        """문서 삭제 (FAISS는 삭제 미지원, 재구축 필요)"""
        # FAISS는 직접 삭제를 지원하지 않음
        # 실제로는 삭제할 문서를 제외하고 인덱스 재구축
        raise NotImplementedError(
            "FAISS does not support direct deletion. "
            "Rebuild index without deleted documents instead."
        )

    def save(self, path: str):
        """인덱스 저장"""
        import json

        # FAISS 인덱스 저장
        self.faiss.write_index(self.index, f"{path}.index")

        # 문서 및 매핑 저장 (JSON으로 안전하게 직렬화)
        serialized_docs = []
        for doc in self.documents:
            serialized_docs.append({"content": doc.content, "metadata": doc.metadata})

        with open(f"{path}.json", "w", encoding="utf-8") as f:
            json.dump(
                {"documents": serialized_docs, "ids_to_index": self.ids_to_index},
                f,
                ensure_ascii=False,
                indent=2,
            )

    def load(self, path: str):
        """인덱스 로드"""
        import json

        from beanllm.domain.loaders import Document

        # FAISS 인덱스 로드
        self.index = self.faiss.read_index(f"{path}.index")

        # 문서 및 매핑 로드 (JSON에서 안전하게 역직렬화)
        with open(f"{path}.json", "r", encoding="utf-8") as f:
            data = json.load(f)

            # Document 객체로 재구성
            self.documents = []
            for doc_data in data["documents"]:
                doc = Document(content=doc_data["content"], metadata=doc_data.get("metadata", {}))
                self.documents.append(doc)

            self.ids_to_index = data["ids_to_index"]
