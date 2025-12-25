"""
타입 정의 - Service 레이어용 타입 힌트
명확한 타입 힌트를 위한 Protocol 및 TypeVar 정의
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
)

if TYPE_CHECKING:
    from .._source_providers.base_provider import BaseLLMProvider


# TypeVar 정의
T = TypeVar("T")
ProviderT = TypeVar("ProviderT", bound="BaseLLMProvider")


# Provider Factory Protocol
class ProviderFactoryProtocol(Protocol):
    """Provider Factory 인터페이스"""

    def create(self, model: str, provider_name: Optional[str] = None) -> "BaseLLMProvider":
        """Provider 생성"""
        ...


# Vector Store Protocol
class VectorStoreProtocol(Protocol):
    """Vector Store 인터페이스"""

    def similarity_search(self, query: str, k: int, **kwargs: Any) -> List[Any]:
        """유사도 검색"""
        ...

    def hybrid_search(self, query: str, k: int, **kwargs: Any) -> List[Any]:
        """하이브리드 검색"""
        ...

    def mmr_search(self, query: str, k: int, **kwargs: Any) -> List[Any]:
        """MMR 검색"""
        ...

    def rerank(self, query: str, results: List[Any], top_k: int) -> List[Any]:
        """재순위화"""
        ...


# Embedding Service Protocol
class EmbeddingServiceProtocol(Protocol):
    """Embedding Service 인터페이스"""

    def embed(self, texts: List[str]) -> List[List[float]]:
        """임베딩 생성"""
        ...


# Document Loader Protocol
class DocumentLoaderProtocol(Protocol):
    """Document Loader 인터페이스"""

    def load(self, source: Union[str, Any]) -> List[Any]:
        """문서 로드"""
        ...


# Text Splitter Protocol
class TextSplitterProtocol(Protocol):
    """Text Splitter 인터페이스"""

    def split(self, documents: List[Any], chunk_size: int, chunk_overlap: int) -> List[Any]:
        """텍스트 분할"""
        ...


# Tool Registry Protocol
class ToolRegistryProtocol(Protocol):
    """Tool Registry 인터페이스 (기존 ToolRegistry와 호환)"""

    def add_tool(self, tool: Any) -> None:
        """도구 추가"""
        ...

    def get_all(self) -> List[Any]:
        """모든 도구 가져오기 (기존 ToolRegistry.get_all()와 동일)"""
        ...

    def get_all_tools(self) -> Dict[str, Any]:
        """모든 도구 가져오기 (Dict 형태)"""
        ...

    def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        """도구 실행"""
        ...

    def get_tool(self, name: str) -> Optional[Any]:
        """도구 가져오기"""
        ...


# 타입 별칭
MessageDict = Dict[str, str]
MessageList = List[MessageDict]
ExtraParams = Dict[str, Any]
MetadataDict = Dict[str, Any]
