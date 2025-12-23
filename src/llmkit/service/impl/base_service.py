"""
BaseService - Service 구현체의 공통 로직
책임: 중복 코드 제거 (DRY 원칙)
SOLID 원칙:
- DRY: 공통 패턴 추출
- SRP: 공통 로직만 담당
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, Optional

from ...infrastructure.adapter import ParameterAdapter, adapt_parameters

if TYPE_CHECKING:
    from ...service.types import ProviderFactoryProtocol


class BaseService(ABC):
    """
    Service 구현체의 기본 클래스

    책임:
    - 공통 로직 제공 (Provider 생성, 파라미터 변환 등)
    - 중복 코드 제거

    SOLID:
    - DRY: 공통 패턴 재사용
    - SRP: 공통 로직만 담당
    """

    def __init__(
        self,
        provider_factory: Optional["ProviderFactoryProtocol"] = None,
        parameter_adapter: Optional[ParameterAdapter] = None,
    ) -> None:
        """
        공통 의존성 주입

        Args:
            provider_factory: Provider 생성 팩토리 (선택적)
            parameter_adapter: 파라미터 어댑터 (선택적)
        """
        self._provider_factory = provider_factory
        self._parameter_adapter = parameter_adapter

    def _create_provider(
        self, model: str, provider_name: Optional[str] = None
    ) -> Any:  # BaseLLMProvider
        """
        Provider 생성 (공통 로직)

        Args:
            model: 모델 이름
            provider_name: Provider 이름 (선택적)

        Returns:
            Provider 인스턴스

        책임:
            - Provider 생성만 (비즈니스 로직)
        """
        if not self._provider_factory:
            raise ValueError("Provider factory is required")
        return self._provider_factory.create(model, provider_name)

    def _adapt_parameters(
        self, provider_name: str, model: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        파라미터 변환 (공통 로직)

        Args:
            provider_name: Provider 이름
            model: 모델 이름
            params: 원본 파라미터

        Returns:
            변환된 파라미터

        책임:
            - 파라미터 변환만 (비즈니스 로직)
        """
        # None 값 제거
        clean_params = {k: v for k, v in params.items() if v is not None}

        # ParameterAdapter 사용 (의존성 주입)
        if self._parameter_adapter:
            adapted = self._parameter_adapter.adapt(provider_name, model, clean_params)
            return adapted.params

        # 기본 변환 (adapter 없을 때)
        adapted = adapt_parameters(provider_name, model, clean_params)
        return adapted.params
