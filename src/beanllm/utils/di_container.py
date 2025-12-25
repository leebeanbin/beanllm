"""
Dependency Injection Container - 의존성 주입 컨테이너
책임: Factory 객체 재사용 및 중복 제거 (DRY 원칙)
SOLID 원칙:
- SRP: 의존성 관리만 담당
- DIP: 인터페이스에 의존
- 싱글톤 패턴으로 객체 재사용
"""

import threading
from typing import Any, Dict, Optional

from .._source_providers.provider_factory import ProviderFactory as SourceProviderFactory
from ..facade.client_facade import SourceProviderFactoryAdapter
from ..handler.factory import HandlerFactory
from ..service.factory import ServiceFactory


class DIContainer:
    """
    의존성 주입 컨테이너 (싱글톤)

    책임:
    - Factory 객체 재사용
    - 중복 코드 제거
    - 의존성 관리

    SOLID:
    - SRP: 의존성 관리만
    - 싱글톤: 객체 재사용
    """

    _instance: Optional["DIContainer"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "DIContainer":
        """싱글톤 패턴"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """초기화 (한 번만 실행)"""
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._provider_factory: Optional[Any] = None
        self._service_factory: Optional[ServiceFactory] = None
        self._handler_factory: Optional[HandlerFactory] = None
        self._service_factories: Dict[str, ServiceFactory] = {}  # 커스텀 ServiceFactory 캐시
        self._initialized = True

    @property
    def provider_factory(self) -> Any:
        """
        Provider Factory (싱글톤)

        Returns:
            SourceProviderFactoryAdapter 인스턴스
        """
        if self._provider_factory is None:
            self._provider_factory = SourceProviderFactoryAdapter(SourceProviderFactory)
        return self._provider_factory

    def get_service_factory(self, **kwargs) -> ServiceFactory:
        """
        Service Factory 가져오기 (캐싱 지원)

        Args:
            **kwargs: ServiceFactory 생성 인자
                - provider_factory: ProviderFactory (기본: 싱글톤)
                - vector_store: VectorStore (선택적)
                - parameter_adapter: ParameterAdapter (선택적)
                - 기타 ServiceFactory 생성 인자

        Returns:
            ServiceFactory 인스턴스
        """
        # 캐시 키 생성 (kwargs 기반)
        cache_key = self._get_cache_key(**kwargs)

        if cache_key not in self._service_factories:
            # ProviderFactory는 기본적으로 싱글톤 사용
            provider_factory = kwargs.pop("provider_factory", self.provider_factory)

            # ServiceFactory 생성
            service_factory = ServiceFactory(provider_factory=provider_factory, **kwargs)
            self._service_factories[cache_key] = service_factory

        return self._service_factories[cache_key]

    @property
    def service_factory(self) -> ServiceFactory:
        """
        기본 Service Factory (싱글톤)

        Returns:
            ServiceFactory 인스턴스 (기본 설정)
        """
        if self._service_factory is None:
            self._service_factory = self.get_service_factory()
        return self._service_factory

    @property
    def handler_factory(self) -> HandlerFactory:
        """
        Handler Factory (싱글톤)

        Returns:
            HandlerFactory 인스턴스
        """
        if self._handler_factory is None:
            self._handler_factory = HandlerFactory(self.service_factory)
        return self._handler_factory

    def get_handler_factory(
        self, service_factory: Optional[ServiceFactory] = None
    ) -> HandlerFactory:
        """
        Handler Factory 가져오기 (커스텀 ServiceFactory 지원)

        Args:
            service_factory: ServiceFactory (None이면 기본 사용)

        Returns:
            HandlerFactory 인스턴스
        """
        if service_factory is None:
            return self.handler_factory

        # 커스텀 ServiceFactory를 사용하는 경우 새 HandlerFactory 생성
        return HandlerFactory(service_factory)

    def _get_cache_key(self, **kwargs) -> str:
        """
        캐시 키 생성

        Args:
            **kwargs: ServiceFactory 생성 인자

        Returns:
            캐시 키 문자열
        """
        # 중요한 인자만 키로 사용 (vector_store 등)
        key_parts = []

        if "vector_store" in kwargs:
            # vector_store는 객체 ID로 구분
            key_parts.append(f"vector_store:{id(kwargs['vector_store'])}")

        if "parameter_adapter" in kwargs:
            key_parts.append(f"adapter:{id(kwargs['parameter_adapter'])}")

        # 기본 키
        if not key_parts:
            return "default"

        return "|".join(key_parts)

    def reset(self):
        """
        컨테이너 리셋 (테스트용)
        """
        self._provider_factory = None
        self._service_factory = None
        self._handler_factory = None
        self._service_factories.clear()


# 전역 싱글톤 인스턴스
_container = DIContainer()


def get_container() -> DIContainer:
    """
    DI Container 인스턴스 가져오기

    Returns:
        DIContainer 싱글톤 인스턴스
    """
    return _container


__all__ = [
    "DIContainer",
    "get_container",
]
