"""
Pytest 설정 및 공통 Fixtures
"""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest

# 테스트 환경 변수 설정
os.environ.setdefault("PYTEST", "true")


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """임시 디렉토리 생성"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_text() -> str:
    """샘플 텍스트"""
    return """
# Introduction

This is a test document for testing text splitting functionality.
It contains multiple paragraphs and sections.

## Section 1

First section content here.
Multiple lines of text for testing.

## Section 2

Second section with more content.
This helps test the splitting algorithms.

# Conclusion

Final thoughts and summary.
    """.strip()


@pytest.fixture
def sample_documents():
    """샘플 문서 리스트"""
    from beanllm import Document

    return [
        Document(
            content="First document content here.", metadata={"source": "doc1.txt", "page": 1}
        ),
        Document(
            content="Second document with different content.",
            metadata={"source": "doc2.txt", "page": 1},
        ),
        Document(
            content="Third document for testing purposes.",
            metadata={"source": "doc3.txt", "page": 1},
        ),
    ]


@pytest.fixture
def mock_env(monkeypatch):
    """Mock 환경 변수"""
    # API 키는 설정하지 않음 (선택적 의존성 테스트)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")


@pytest.fixture
def skip_if_no_provider():
    """Provider가 없으면 테스트 스킵"""
    import pytest
    from beanllm._source_providers import OpenAIProvider

    try:
        # OpenAI Provider가 사용 가능한지 확인
        if OpenAIProvider is None:
            pytest.skip("OpenAI provider not available")
    except (ImportError, AttributeError):
        pytest.skip("Provider not available")


@pytest.fixture
def mock_client():
    """Mock Client for testing"""
    from unittest.mock import MagicMock
    from beanllm.facade.client_facade import Client

    mock = MagicMock(spec=Client)
    mock.model = "gpt-4o-mini"
    return mock


@pytest.fixture
def sample_text_long():
    """긴 샘플 텍스트"""
    return """
    Artificial intelligence (AI) is transforming the world in unprecedented ways.
    Machine learning, a subset of AI, enables computers to learn from data without explicit programming.
    Deep learning uses neural networks with multiple layers to process complex patterns.
    Natural language processing allows machines to understand and generate human language.
    Computer vision enables machines to interpret and understand visual information.
    These technologies are being applied across industries, from healthcare to finance to transportation.
    The future of AI holds great promise but also raises important ethical questions.
    As AI systems become more powerful, we must ensure they are developed and deployed responsibly.
    """.strip()
