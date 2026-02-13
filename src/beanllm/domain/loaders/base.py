"""
Loaders Base - 문서 로더 베이스 클래스
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from beanllm.domain.loaders.types import Document
else:
    try:
        from beanllm.domain.loaders.types import Document
    except ImportError:
        Document = Any  # noqa: F811


class BaseDocumentLoader(ABC):
    """Document Loader 베이스 클래스"""

    @abstractmethod
    def load(self, *args: Any, **kwargs: Any) -> List["Document"]:
        """문서 로딩"""
        pass

    @abstractmethod
    def lazy_load(self, *args: Any, **kwargs: Any) -> Any:
        """지연 로딩 (제너레이터)"""
        pass
