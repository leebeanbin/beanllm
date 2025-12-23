"""
Loaders Base - 문서 로더 베이스 클래스
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .types import Document
else:
    # 런타임에만 import
    try:
        from .types import Document
    except ImportError:
        from typing import Any

        Document = Any  # type: ignore


class BaseDocumentLoader(ABC):
    """Document Loader 베이스 클래스"""

    @abstractmethod
    def load(self) -> List["Document"]:
        """문서 로딩"""
        pass

    @abstractmethod
    def lazy_load(self):
        """지연 로딩 (제너레이터)"""
        pass
