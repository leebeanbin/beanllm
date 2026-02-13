"""
Facade Base - 공통 DI Container 초기화 로직
책임: Facade 클래스들의 중복된 _init_services() 패턴 추출
SOLID 원칙:
- DRY: 중복 코드 제거
- Template Method: _init_handlers() 훅으로 서브클래스별 초기화
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from beanllm.handler.factory import HandlerFactory
    from beanllm.service.factory import ServiceFactory
    from beanllm.utils.core.di_container import DIContainer


class FacadeBase:
    """
    Base class for all facades - provides common DI container initialization.

    Subclasses override _init_handlers() to create their specific handlers.
    For facades that need custom service_factory (e.g. vector_store), override
    _prepare_container_dependencies() instead of _init_services().
    """

    def __init__(self) -> None:
        """Initialize facade - calls _init_services()."""
        self._init_services()

    def _init_services(self) -> None:
        """Service & Handler initialization via DI Container."""
        from beanllm.utils.core.di_container import get_container

        container = get_container()
        self._prepare_container_dependencies(container)
        self._init_handlers()

    def _prepare_container_dependencies(self, container: "DIContainer") -> None:
        """
        Prepare handler_factory and service_factory from container.

        Override in subclasses that need custom service_factory
        (e.g. RAGChain with vector_store).

        Args:
            container: DI container instance.
        """
        self._handler_factory: "HandlerFactory" = container.handler_factory
        self._service_factory: "ServiceFactory" = container.get_service_factory()

    def _init_handlers(self) -> None:
        """Override in subclasses to create specific handlers."""
        pass
