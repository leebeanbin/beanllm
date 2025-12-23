"""
VisionRAGRequest - Vision RAG 요청 DTO
책임: Vision RAG 요청 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from ...facade.client_facade import Client
    from ...service.types import VectorStoreProtocol
    from ...vision_embeddings import CLIPEmbedding, MultimodalEmbedding


@dataclass
class VisionRAGRequest:
    """
    Vision RAG 요청 DTO

    책임:
    - 데이터 구조 정의만
    - 검증 없음 (Handler에서 처리)
    - 비즈니스 로직 없음 (Service에서 처리)
    """

    # retrieve 메서드용
    query: Optional[str] = None
    k: int = 4

    # query 메서드용
    question: Optional[str] = None
    include_sources: bool = False
    include_images: bool = True

    # batch_query 메서드용
    questions: Optional[List[str]] = None

    # from_images/from_sources 메서드용
    source: Optional[Union[str, Path, List[Union[str, Path]]]] = None
    sources: Optional[List[Union[str, Path]]] = None
    generate_captions: bool = True
    llm_model: str = "gpt-4o"

    # __init__ 메서드용
    vector_store: Optional["VectorStoreProtocol"] = None
    vision_embedding: Optional[Union["CLIPEmbedding", "MultimodalEmbedding"]] = None
    llm: Optional["Client"] = None
    prompt_template: Optional[str] = None

    # 추가 파라미터
    extra_params: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """기본값 설정"""
        if self.extra_params is None:
            self.extra_params = {}
