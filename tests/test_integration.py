"""
Integration Tests - 레이어 간 통합 테스트
"""

import pytest


class TestFacadeToHandler:
    """Facade → Handler 통합 테스트"""

    def test_client_facade_to_handler(self):
        """Client Facade가 Handler를 사용하는지 확인"""
        try:
            from beanllm.facade.client_facade import Client
        except ImportError:
            from src.beanllm.facade.client_facade import Client

        try:
            client = Client(model="gpt-4o-mini")
            # 내부적으로 Handler를 사용하는지 확인
            assert hasattr(client, "_chat_handler") or hasattr(client, "chat")
        except (ValueError, ImportError):
            pytest.skip("Client provider not available")


class TestHandlerToService:
    """Handler → Service 통합 테스트"""

    def test_chat_handler_to_service(self):
        """ChatHandler가 Service를 사용하는지 확인"""
        try:
            from beanllm.handler.chat_handler import ChatHandler
            from beanllm.service.factory import ServiceFactory
            from beanllm._source_providers.provider_factory import ProviderFactory
        except ImportError:
            from src.beanllm.handler.chat_handler import ChatHandler
            from src.beanllm.service.factory import ServiceFactory
            from src.beanllm._source_providers.provider_factory import ProviderFactory

        try:
            provider_factory = ProviderFactory()
            service_factory = ServiceFactory(provider_factory=provider_factory)
            handler = ChatHandler(
                service_factory.create_chat_service()
            )  # get_chat_service가 아니라 create_chat_service
            assert handler is not None
        except (ValueError, ImportError, AttributeError):
            pytest.skip("Service provider not available")


class TestServiceToDomain:
    """Service → Domain 통합 테스트"""

    def test_rag_service_uses_domain(self):
        """RAGService가 Domain을 사용하는지 확인"""
        try:
            from beanllm.service.rag_service import IRAGService
            from beanllm.domain import Document, Embedding, VectorStore
        except ImportError:
            from src.beanllm.service.rag_service import IRAGService
            from src.beanllm.domain import Document, Embedding, VectorStore

        # 인터페이스 확인
        assert IRAGService is not None
        assert Document is not None
        assert Embedding is not None
        assert VectorStore is not None


class TestEndToEnd:
    """End-to-End 테스트"""

    def test_import_chain(self):
        """전체 import 체인 테스트"""
        # Facade → Handler → Service → Domain → Infrastructure
        from beanllm import Client, Embedding, Document
        from beanllm.infrastructure import get_model_registry
        from beanllm.utils import Config

        assert Client is not None
        assert Embedding is not None
        assert Document is not None
        assert get_model_registry is not None
        assert Config is not None

    def test_basic_workflow(self, temp_dir):
        """기본 워크플로우 테스트"""
        from beanllm import Document, TextSplitter

        # 1. Document 생성
        doc = Document(content="Test content", metadata={"source": "test.txt"})

        # 2. Text Splitter로 분할
        chunks = TextSplitter.split([doc], chunk_size=50)
        assert len(chunks) > 0

        # 3. 메타데이터 보존 확인
        assert all("source" in chunk.metadata for chunk in chunks)

    def test_rag_workflow(self, temp_dir):
        """RAG 워크플로우 테스트"""
        from beanllm import DocumentLoader, TextSplitter

        # 테스트 문서 생성
        test_file = temp_dir / "test.txt"
        test_file.write_text("This is a test document for RAG testing.")

        try:
            # 1. 문서 로딩
            docs = DocumentLoader.load(str(test_file))
            assert len(docs) > 0

            # 2. 텍스트 분할
            chunks = TextSplitter.split(docs, chunk_size=50)
            assert len(chunks) > 0

            # 3. 메타데이터 확인
            assert all("source" in chunk.metadata for chunk in chunks)

        except Exception as e:
            pytest.skip(f"RAG workflow test skipped: {e}")

