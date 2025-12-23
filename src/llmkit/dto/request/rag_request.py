"""
RAGRequest - RAG 요청 DTO
책임: RAG 요청 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from ...service.types import VectorStoreProtocol


@dataclass
class RAGRequest:
    """
    RAG 요청 DTO

    책임:
    - 데이터 구조 정의만
    - 검증 없음 (Handler에서 처리)
    - 비즈니스 로직 없음 (Service에서 처리)
    """

    query: str
    source: Optional[Union[str, Path, List[Any]]] = None
    vector_store: Optional["VectorStoreProtocol"] = None
    k: int = 4
    rerank: bool = False
    mmr: bool = False
    hybrid: bool = False
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    prompt_template: Optional[str] = None
    retriever_config: Optional[Dict[str, Any]] = None
    extra_params: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """기본값 설정"""
        if self.retriever_config is None:
            self.retriever_config = {}
        if self.extra_params is None:
            self.extra_params = {}
