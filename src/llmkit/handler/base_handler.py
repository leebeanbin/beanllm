"""
BaseHandler - Handler 기본 클래스
책임: 중복 코드 제거 (DRY 원칙)
SOLID 원칙:
- DRY: 공통 패턴 추출
- SRP: 공통 로직만 담당
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Type, TypeVar

T = TypeVar("T")


class BaseHandler(ABC):
    """
    Handler 기본 클래스

    책임:
    - 공통 패턴 제공 (DTO 생성, Service 호출 등)
    - 중복 코드 제거

    SOLID:
    - DRY: 공통 패턴 재사용
    - SRP: 공통 로직만 담당
    """

    def __init__(self, service: Any) -> None:
        """
        의존성 주입

        Args:
            service: Service 인스턴스 (인터페이스에 의존 - DIP)
        """
        self._service = service

    def _create_request(self, request_class: Type[T], **kwargs: Any) -> T:
        """
        DTO 생성 헬퍼

        Args:
            request_class: Request DTO 클래스
            **kwargs: DTO 생성 인자

        Returns:
            Request DTO 인스턴스
        """
        return request_class(**kwargs)

    async def _call_service(self, method_name: str, request: Any) -> Any:
        """
        Service 메서드 호출 헬퍼

        Args:
            method_name: Service 메서드 이름
            request: Request DTO

        Returns:
            Service 메서드 반환값
        """
        method = getattr(self._service, method_name)
        if not callable(method):
            raise AttributeError(f"Method '{method_name}' not found in service")

        # 비동기 메서드인지 확인
        import asyncio

        if asyncio.iscoroutinefunction(method):
            return await method(request)
        else:
            return method(request)
