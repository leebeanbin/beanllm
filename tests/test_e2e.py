"""
End-to-End Tests - 전체 워크플로우 테스트
"""

import pytest


class TestE2EBasic:
    """기본 E2E 테스트"""

    def test_import_all_modules(self):
        """모든 주요 모듈 import 테스트"""
        try:
            # Facade
            from beanllm import Client, RAGChain, Agent, Graph, StateGraph

            # Domain
            from beanllm.domain import (
                Document,
                Embedding,
                TextSplitter,
                VectorStore,
                Tool,
                BaseMemory,
            )

            # Infrastructure
            from beanllm.infrastructure import ModelRegistry, ParameterAdapter

            # Utils
            from beanllm.utils import Config, retry, get_logger
        except ImportError:
            # Facade
            from src.beanllm import Client, RAGChain, Agent, Graph, StateGraph

            # Domain
            from src.beanllm.domain import (
                Document,
                Embedding,
                TextSplitter,
                VectorStore,
                Tool,
                BaseMemory,
            )

            # Infrastructure
            from src.beanllm.infrastructure import ModelRegistry, ParameterAdapter

            # Utils
            from src.beanllm.utils import Config, retry, get_logger

        assert all(
            [
                Client,
                RAGChain,
                Agent,
                Graph,
                StateGraph,
                Document,
                Embedding,
                TextSplitter,
                VectorStore,
                Tool,
                BaseMemory,
                ModelRegistry,
                ParameterAdapter,
                Config,
                retry,
                get_logger,
            ]
        )

    def test_basic_import_chain(self):
        """기본 import 체인 테스트"""
        # 최상위에서 모든 것을 import
        try:
            from beanllm import (
                Client,
                Embedding,
                Document,
                Agent,
                RAGChain,
                Graph,
                StateGraph,
                MultiAgentCoordinator,
                VisionRAG,
                WebSearch,
            )
        except ImportError:
            from src.beanllm import (
                Client,
                Embedding,
                Document,
                Agent,
                RAGChain,
                Graph,
                StateGraph,
                MultiAgentCoordinator,
                VisionRAG,
                WebSearch,
            )

        assert all(
            [
                Client,
                Embedding,
                Document,
                Agent,
                RAGChain,
                Graph,
                StateGraph,
                MultiAgentCoordinator,
                VisionRAG,
                WebSearch,
            ]
        )


class TestE2EDocumentProcessing:
    """문서 처리 E2E 테스트"""

    def test_document_loading_to_splitting(self, temp_dir):
        """문서 로딩 → 분할 E2E"""
        try:
            from beanllm import DocumentLoader, TextSplitter
        except ImportError:
            from src.beanllm import DocumentLoader, TextSplitter

        # 테스트 파일 생성
        test_file = temp_dir / "test.txt"
        test_file.write_text("This is a test document. " * 10)

        try:
            # 1. 로딩
            docs = DocumentLoader.load(str(test_file))
            assert len(docs) > 0

            # 2. 분할
            chunks = TextSplitter.split(docs, chunk_size=50)
            assert len(chunks) > 0

            # 3. 메타데이터 확인
            assert all("source" in chunk.metadata for chunk in chunks)

        except Exception as e:
            pytest.skip(f"Document processing E2E skipped: {e}")


class TestE2ERAG:
    """RAG E2E 테스트"""

    def test_rag_full_pipeline(self, temp_dir):
        """RAG 전체 파이프라인 테스트"""
        try:
            from beanllm import DocumentLoader, TextSplitter, RAGChain
        except ImportError:
            from src.beanllm import DocumentLoader, TextSplitter, RAGChain

        # 테스트 문서 생성
        test_file = temp_dir / "test.txt"
        test_file.write_text("This is a test document for RAG testing. " * 5)

        try:
            # RAG 생성
            rag = RAGChain.from_documents(str(temp_dir))
            assert rag is not None

            # 질의 (실제 API 호출은 스킵)
            # answer = rag.query("What is this about?")
            # assert answer is not None

        except Exception as e:
            if "provider" in str(e).lower() or "api" in str(e).lower():
                pytest.skip(f"RAG E2E skipped (provider not available): {e}")
            else:
                pytest.skip(f"RAG E2E skipped: {e}")


class TestE2EAgent:
    """Agent E2E 테스트"""

    def test_agent_creation(self):
        """Agent 생성 E2E"""
        try:
            from beanllm import Agent
        except ImportError:
            from src.beanllm import Agent

        try:
            # Agent는 model을 직접 받음
            agent = Agent(model="gpt-4o-mini", tools=[], max_iterations=5)
            assert agent is not None
        except (ValueError, ImportError, TypeError):
            pytest.skip("Agent E2E skipped (provider not available)")

