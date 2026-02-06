"""
Splitters Domain - 텍스트 분할 도메인

기본 분할기:
- CharacterTextSplitter: 단순 문자 기반 분할
- RecursiveCharacterTextSplitter: 계층적 구분자 분할 (권장)
- TokenTextSplitter: 토큰 기반 분할 (LLM 최적화)
- MarkdownHeaderTextSplitter: 마크다운 헤더 기반 분할

시맨틱 분할기 (의미 기반):
- SemanticTextSplitter: 임베딩 유사도 기반 분할 (20-40% 검색 품질 향상)
- CoherenceTextSplitter: 주제 일관성 기반 클러스터링 분할

Example:
    ```python
    from beanllm.domain.splitters import SemanticTextSplitter

    # 의미 기반 분할 (가장 효과적)
    splitter = SemanticTextSplitter(
        model="all-MiniLM-L6-v2",
        threshold=0.5,
        max_chunk_size=1000
    )
    chunks = splitter.split_text(text)

    # semchunk 사용 (더 빠름)
    splitter = SemanticTextSplitter(use_semchunk=True, chunk_size=512)
    ```
"""

from beanllm.domain.splitters.base import BaseTextSplitter
from beanllm.domain.splitters.factory import TextSplitter, split_documents
from beanllm.domain.splitters.splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

# Semantic Splitters (의존성이 없을 수 있음)
try:
    from beanllm.domain.splitters.semantic import (
        CoherenceTextSplitter,
        SemanticTextSplitter,
    )
except ImportError:
    SemanticTextSplitter = None  # type: ignore
    CoherenceTextSplitter = None  # type: ignore

__all__ = [
    # Base
    "BaseTextSplitter",
    # Basic Splitters
    "CharacterTextSplitter",
    "RecursiveCharacterTextSplitter",
    "TokenTextSplitter",
    "MarkdownHeaderTextSplitter",
    # Factory
    "TextSplitter",
    "split_documents",
]

# Semantic Splitters (있는 경우만 추가)
if SemanticTextSplitter is not None:
    __all__.extend(["SemanticTextSplitter", "CoherenceTextSplitter"])
