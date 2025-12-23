"""
BaseResponse - 응답 DTO의 공통 로직
책임: DTO 변환 패턴 재사용 (DRY 원칙)
"""

from abc import ABC
from typing import Any, Dict


class BaseResponse(ABC):
    """
    응답 DTO의 기본 클래스

    책임:
    - 공통 변환 로직 제공
    - 중복 코드 제거

    SOLID:
    - DRY: 공통 패턴 재사용
    """

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs) -> "BaseResponse":
        """
        딕셔너리에서 응답 생성 (공통 로직)

        Args:
            data: 딕셔너리 데이터
            **kwargs: 추가 파라미터

        Returns:
            응답 인스턴스
        """
        # 하위 클래스에서 구현
        raise NotImplementedError

    @classmethod
    def from_provider_response(cls, provider_response: Dict[str, Any], **kwargs) -> "BaseResponse":
        """
        Provider 응답에서 생성 (공통 로직)

        Args:
            provider_response: Provider 응답 딕셔너리
            **kwargs: 추가 파라미터

        Returns:
            응답 인스턴스
        """
        # 하위 클래스에서 구현
        raise NotImplementedError
