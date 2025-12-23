"""
Service Types 테스트 - Protocol 및 타입 정의 테스트
"""

import pytest
from typing import List, Dict, Any, Optional

from llmkit.service.types import (
    ProviderFactoryProtocol,
    VectorStoreProtocol,
    EmbeddingServiceProtocol,
    DocumentLoaderProtocol,
    TextSplitterProtocol,
    ToolRegistryProtocol,
    MessageDict,
    MessageList,
    ExtraParams,
    MetadataDict,
    T,
    ProviderT,
)


class TestServiceTypes:
    """Service Types 테스트"""

    def test_type_aliases(self):
        """타입 별칭 테스트"""
        # 타입 별칭들이 올바르게 정의되었는지 확인
        assert MessageDict == Dict[str, str]
        assert MessageList == List[MessageDict]
        assert ExtraParams == Dict[str, Any]
        assert MetadataDict == Dict[str, Any]

    def test_provider_factory_protocol(self):
        """ProviderFactoryProtocol 인터페이스 테스트"""

        # Protocol은 타입 체크용이므로 실제 구현체가 필요
        class MockProviderFactory:
            def create(self, model: str, provider_name: Optional[str] = None):
                return None

        factory = MockProviderFactory()
        # Protocol을 구현한 클래스는 타입 체크를 통과해야 함
        assert hasattr(factory, "create")
        assert callable(factory.create)

    def test_vector_store_protocol(self):
        """VectorStoreProtocol 인터페이스 테스트"""

        class MockVectorStore:
            def similarity_search(self, query: str, k: int, **kwargs: Any) -> List[Any]:
                return []

            def hybrid_search(self, query: str, k: int, **kwargs: Any) -> List[Any]:
                return []

            def mmr_search(self, query: str, k: int, **kwargs: Any) -> List[Any]:
                return []

            def rerank(self, query: str, results: List[Any], top_k: int) -> List[Any]:
                return []

        store = MockVectorStore()
        assert hasattr(store, "similarity_search")
        assert hasattr(store, "hybrid_search")
        assert hasattr(store, "mmr_search")
        assert hasattr(store, "rerank")

    def test_embedding_service_protocol(self):
        """EmbeddingServiceProtocol 인터페이스 테스트"""

        class MockEmbeddingService:
            def embed(self, texts: List[str]) -> List[List[float]]:
                return [[0.1, 0.2, 0.3] for _ in texts]

        service = MockEmbeddingService()
        assert hasattr(service, "embed")
        result = service.embed(["test"])
        assert isinstance(result, list)

    def test_document_loader_protocol(self):
        """DocumentLoaderProtocol 인터페이스 테스트"""

        class MockDocumentLoader:
            def load(self, source: Any) -> List[Any]:
                return []

        loader = MockDocumentLoader()
        assert hasattr(loader, "load")
        result = loader.load("test")
        assert isinstance(result, list)

    def test_text_splitter_protocol(self):
        """TextSplitterProtocol 인터페이스 테스트"""

        class MockTextSplitter:
            def split(self, documents: List[Any], chunk_size: int, chunk_overlap: int) -> List[Any]:
                return []

        splitter = MockTextSplitter()
        assert hasattr(splitter, "split")
        result = splitter.split([], 100, 20)
        assert isinstance(result, list)

    def test_tool_registry_protocol(self):
        """ToolRegistryProtocol 인터페이스 테스트"""

        class MockToolRegistry:
            def add_tool(self, tool: Any) -> None:
                pass

            def get_all(self) -> List[Any]:
                return []

            def get_all_tools(self) -> Dict[str, Any]:
                return {}

            def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
                return None

            def get_tool(self, name: str) -> Optional[Any]:
                return None

        registry = MockToolRegistry()
        assert hasattr(registry, "add_tool")
        assert hasattr(registry, "get_all")
        assert hasattr(registry, "get_all_tools")
        assert hasattr(registry, "execute")
        assert hasattr(registry, "get_tool")


"""
Service Types 테스트 - Protocol 및 타입 정의 테스트
"""

import pytest
from typing import List, Dict, Any, Optional

from llmkit.service.types import (
    ProviderFactoryProtocol,
    VectorStoreProtocol,
    EmbeddingServiceProtocol,
    DocumentLoaderProtocol,
    TextSplitterProtocol,
    ToolRegistryProtocol,
    MessageDict,
    MessageList,
    ExtraParams,
    MetadataDict,
    T,
    ProviderT,
)


class TestServiceTypes:
    """Service Types 테스트"""

    def test_type_aliases(self):
        """타입 별칭 테스트"""
        # 타입 별칭들이 올바르게 정의되었는지 확인
        assert MessageDict == Dict[str, str]
        assert MessageList == List[MessageDict]
        assert ExtraParams == Dict[str, Any]
        assert MetadataDict == Dict[str, Any]

    def test_provider_factory_protocol(self):
        """ProviderFactoryProtocol 인터페이스 테스트"""

        # Protocol은 타입 체크용이므로 실제 구현체가 필요
        class MockProviderFactory:
            def create(self, model: str, provider_name: Optional[str] = None):
                return None

        factory = MockProviderFactory()
        # Protocol을 구현한 클래스는 타입 체크를 통과해야 함
        assert hasattr(factory, "create")
        assert callable(factory.create)

    def test_vector_store_protocol(self):
        """VectorStoreProtocol 인터페이스 테스트"""

        class MockVectorStore:
            def similarity_search(self, query: str, k: int, **kwargs: Any) -> List[Any]:
                return []

            def hybrid_search(self, query: str, k: int, **kwargs: Any) -> List[Any]:
                return []

            def mmr_search(self, query: str, k: int, **kwargs: Any) -> List[Any]:
                return []

            def rerank(self, query: str, results: List[Any], top_k: int) -> List[Any]:
                return []

        store = MockVectorStore()
        assert hasattr(store, "similarity_search")
        assert hasattr(store, "hybrid_search")
        assert hasattr(store, "mmr_search")
        assert hasattr(store, "rerank")

    def test_embedding_service_protocol(self):
        """EmbeddingServiceProtocol 인터페이스 테스트"""

        class MockEmbeddingService:
            def embed(self, texts: List[str]) -> List[List[float]]:
                return [[0.1, 0.2, 0.3] for _ in texts]

        service = MockEmbeddingService()
        assert hasattr(service, "embed")
        result = service.embed(["test"])
        assert isinstance(result, list)

    def test_document_loader_protocol(self):
        """DocumentLoaderProtocol 인터페이스 테스트"""

        class MockDocumentLoader:
            def load(self, source: Any) -> List[Any]:
                return []

        loader = MockDocumentLoader()
        assert hasattr(loader, "load")
        result = loader.load("test")
        assert isinstance(result, list)

    def test_text_splitter_protocol(self):
        """TextSplitterProtocol 인터페이스 테스트"""

        class MockTextSplitter:
            def split(self, documents: List[Any], chunk_size: int, chunk_overlap: int) -> List[Any]:
                return []

        splitter = MockTextSplitter()
        assert hasattr(splitter, "split")
        result = splitter.split([], 100, 20)
        assert isinstance(result, list)

    def test_tool_registry_protocol(self):
        """ToolRegistryProtocol 인터페이스 테스트"""

        class MockToolRegistry:
            def add_tool(self, tool: Any) -> None:
                pass

            def get_all(self) -> List[Any]:
                return []

            def get_all_tools(self) -> Dict[str, Any]:
                return {}

            def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
                return None

            def get_tool(self, name: str) -> Optional[Any]:
                return None

        registry = MockToolRegistry()
        assert hasattr(registry, "add_tool")
        assert hasattr(registry, "get_all")
        assert hasattr(registry, "get_all_tools")
        assert hasattr(registry, "execute")
        assert hasattr(registry, "get_tool")


"""
Service Types 테스트 - Protocol 및 타입 정의 테스트
"""

import pytest
from typing import List, Dict, Any, Optional

from llmkit.service.types import (
    ProviderFactoryProtocol,
    VectorStoreProtocol,
    EmbeddingServiceProtocol,
    DocumentLoaderProtocol,
    TextSplitterProtocol,
    ToolRegistryProtocol,
    MessageDict,
    MessageList,
    ExtraParams,
    MetadataDict,
    T,
    ProviderT,
)


class TestServiceTypes:
    """Service Types 테스트"""

    def test_type_aliases(self):
        """타입 별칭 테스트"""
        # 타입 별칭들이 올바르게 정의되었는지 확인
        assert MessageDict == Dict[str, str]
        assert MessageList == List[MessageDict]
        assert ExtraParams == Dict[str, Any]
        assert MetadataDict == Dict[str, Any]

    def test_provider_factory_protocol(self):
        """ProviderFactoryProtocol 인터페이스 테스트"""

        # Protocol은 타입 체크용이므로 실제 구현체가 필요
        class MockProviderFactory:
            def create(self, model: str, provider_name: Optional[str] = None):
                return None

        factory = MockProviderFactory()
        # Protocol을 구현한 클래스는 타입 체크를 통과해야 함
        assert hasattr(factory, "create")
        assert callable(factory.create)

    def test_vector_store_protocol(self):
        """VectorStoreProtocol 인터페이스 테스트"""

        class MockVectorStore:
            def similarity_search(self, query: str, k: int, **kwargs: Any) -> List[Any]:
                return []

            def hybrid_search(self, query: str, k: int, **kwargs: Any) -> List[Any]:
                return []

            def mmr_search(self, query: str, k: int, **kwargs: Any) -> List[Any]:
                return []

            def rerank(self, query: str, results: List[Any], top_k: int) -> List[Any]:
                return []

        store = MockVectorStore()
        assert hasattr(store, "similarity_search")
        assert hasattr(store, "hybrid_search")
        assert hasattr(store, "mmr_search")
        assert hasattr(store, "rerank")

    def test_embedding_service_protocol(self):
        """EmbeddingServiceProtocol 인터페이스 테스트"""

        class MockEmbeddingService:
            def embed(self, texts: List[str]) -> List[List[float]]:
                return [[0.1, 0.2, 0.3] for _ in texts]

        service = MockEmbeddingService()
        assert hasattr(service, "embed")
        result = service.embed(["test"])
        assert isinstance(result, list)

    def test_document_loader_protocol(self):
        """DocumentLoaderProtocol 인터페이스 테스트"""

        class MockDocumentLoader:
            def load(self, source: Any) -> List[Any]:
                return []

        loader = MockDocumentLoader()
        assert hasattr(loader, "load")
        result = loader.load("test")
        assert isinstance(result, list)

    def test_text_splitter_protocol(self):
        """TextSplitterProtocol 인터페이스 테스트"""

        class MockTextSplitter:
            def split(self, documents: List[Any], chunk_size: int, chunk_overlap: int) -> List[Any]:
                return []

        splitter = MockTextSplitter()
        assert hasattr(splitter, "split")
        result = splitter.split([], 100, 20)
        assert isinstance(result, list)

    def test_tool_registry_protocol(self):
        """ToolRegistryProtocol 인터페이스 테스트"""

        class MockToolRegistry:
            def add_tool(self, tool: Any) -> None:
                pass

            def get_all(self) -> List[Any]:
                return []

            def get_all_tools(self) -> Dict[str, Any]:
                return {}

            def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
                return None

            def get_tool(self, name: str) -> Optional[Any]:
                return None

        registry = MockToolRegistry()
        assert hasattr(registry, "add_tool")
        assert hasattr(registry, "get_all")
        assert hasattr(registry, "get_all_tools")
        assert hasattr(registry, "execute")
        assert hasattr(registry, "get_tool")



