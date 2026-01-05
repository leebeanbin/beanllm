"""
Model Parameter Strategy Pattern

Strategy Pattern을 사용하여 모델별 파라미터 지원 정보 관리
Open/Closed Principle 준수: 새로운 모델 추가 시 기존 코드 수정 불필요
"""

import re
from abc import ABC, abstractmethod
from typing import Dict


class ModelParameterStrategy(ABC):
    """
    모델 파라미터 설정 Strategy 베이스 클래스

    각 모델 카테고리별로 파라미터 지원 여부를 정의합니다.
    """

    @abstractmethod
    def get_config(self) -> Dict[str, bool]:
        """
        모델의 파라미터 지원 정보 반환

        Returns:
            Dictionary with:
                - supports_temperature: temperature 파라미터 지원 여부
                - supports_max_tokens: max_tokens 파라미터 지원 여부
                - uses_max_completion_tokens: max_completion_tokens 사용 여부
        """
        pass


class GPT5Strategy(ModelParameterStrategy):
    """GPT-5 시리즈 모델 Strategy"""

    def get_config(self) -> Dict[str, bool]:
        return {
            "supports_temperature": True,
            "supports_max_tokens": False,  # max_tokens 미지원
            "uses_max_completion_tokens": True,  # max_completion_tokens 사용
        }


class GPT41Strategy(ModelParameterStrategy):
    """GPT-4.1 시리즈 모델 Strategy"""

    def get_config(self) -> Dict[str, bool]:
        return {
            "supports_temperature": True,
            "supports_max_tokens": False,  # max_tokens 미지원
            "uses_max_completion_tokens": True,  # max_completion_tokens 사용
        }


class NanoModelStrategy(ModelParameterStrategy):
    """Nano 모델 Strategy (GPT-5-nano 등)"""

    def get_config(self) -> Dict[str, bool]:
        return {
            "supports_temperature": False,  # temperature 미지원 (기본값 1만 지원)
            "supports_max_tokens": False,  # max_tokens 미지원
            "uses_max_completion_tokens": False,
        }


class MiniModelStrategy(ModelParameterStrategy):
    """Mini 모델 Strategy (gpt-4o-mini 등)"""

    def get_config(self) -> Dict[str, bool]:
        return {
            "supports_temperature": False,  # temperature 미지원
            "supports_max_tokens": True,
            "uses_max_completion_tokens": False,
        }


class O3ModelStrategy(ModelParameterStrategy):
    """O3 모델 Strategy"""

    def get_config(self) -> Dict[str, bool]:
        return {
            "supports_temperature": False,  # temperature 미지원
            "supports_max_tokens": True,
            "uses_max_completion_tokens": False,
        }


class O4ModelStrategy(ModelParameterStrategy):
    """O4 모델 Strategy"""

    def get_config(self) -> Dict[str, bool]:
        return {
            "supports_temperature": False,  # temperature 미지원
            "supports_max_tokens": True,
            "uses_max_completion_tokens": False,
        }


class DefaultModelStrategy(ModelParameterStrategy):
    """기본 모델 Strategy (GPT-4, GPT-3.5 등)"""

    def get_config(self) -> Dict[str, bool]:
        return {
            "supports_temperature": True,
            "supports_max_tokens": True,
            "uses_max_completion_tokens": False,
        }


class ModelParameterFactory:
    """
    Model Parameter Strategy Factory

    모델 이름을 기반으로 적절한 Strategy를 반환합니다.
    우선순위 기반 매칭: 더 구체적인 패턴이 먼저 매칭됩니다.
    """

    # 우선순위 순서로 정렬된 Strategy 매핑
    # 더 구체적인 패턴이 먼저 매칭되어야 함
    STRATEGIES = [
        # 가장 구체적인 패턴부터 (nano는 gpt-5-nano처럼 복합적으로 나타날 수 있음)
        ("gpt-5-nano", NanoModelStrategy),
        ("gpt-4.1-nano", NanoModelStrategy),
        # GPT-5, GPT-4.1 시리즈
        ("gpt-5", GPT5Strategy),
        ("gpt-4.1", GPT41Strategy),
        # 특수 모델들
        ("nano", NanoModelStrategy),
        ("mini", MiniModelStrategy),
        ("o3", O3ModelStrategy),
        ("o4", O4ModelStrategy),
    ]

    @classmethod
    def extract_base_model(cls, model: str) -> str:
        """
        날짜가 포함된 모델 이름에서 기본 모델 이름 추출

        Args:
            model: 모델 이름 (예: gpt-5-nano-2025-08-07)

        Returns:
            기본 모델 이름 (예: gpt-5-nano)

        Examples:
            >>> ModelParameterFactory.extract_base_model("gpt-5-nano-2025-08-07")
            'gpt-5-nano'
            >>> ModelParameterFactory.extract_base_model("gpt-4o-2024-05-13")
            'gpt-4o'
        """
        base_model = model

        # YYYY-MM-DD 형식 제거 (예: -2025-08-07)
        base_model = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", base_model)

        # YYYY 형식 제거 (예: -2025)
        base_model = re.sub(r"-\d{4}$", "", base_model)

        return base_model

    @classmethod
    def get_strategy(cls, model: str) -> ModelParameterStrategy:
        """
        모델 이름에 따라 적절한 Strategy 반환

        Args:
            model: 모델 이름

        Returns:
            ModelParameterStrategy 인스턴스

        Examples:
            >>> factory = ModelParameterFactory()
            >>> strategy = factory.get_strategy("gpt-5-nano")
            >>> config = strategy.get_config()
            >>> config['supports_temperature']
            False
        """
        # 날짜 제거
        base_model = cls.extract_base_model(model)
        model_lower = base_model.lower()

        # 우선순위 순서로 매칭 (더 구체적인 패턴이 먼저)
        for pattern, strategy_class in cls.STRATEGIES:
            if pattern in model_lower:
                return strategy_class()

        # 기본 Strategy 반환
        return DefaultModelStrategy()

    @classmethod
    def get_config(cls, model: str) -> Dict[str, bool]:
        """
        모델 이름에 따라 파라미터 설정 반환 (간편 메서드)

        Args:
            model: 모델 이름

        Returns:
            파라미터 지원 정보 딕셔너리
        """
        strategy = cls.get_strategy(model)
        return strategy.get_config()
