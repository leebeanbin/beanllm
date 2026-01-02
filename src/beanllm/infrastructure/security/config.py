"""
Secure configuration management with API key masking

Prevents accidental exposure of sensitive credentials in logs and exceptions.
"""

import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SecureConfig:
    """
    안전한 설정 관리 클래스 (API 키 마스킹)

    민감한 정보를 안전하게 저장하고, 로그나 예외에 노출되지 않도록 마스킹합니다.

    Security Features:
        - API 키 자동 마스킹
        - __repr__, __str__ 오버라이드
        - dict() 변환 시 마스킹
        - 민감 정보 패턴 자동 감지

    Example:
        ```python
        from beanllm.infrastructure.security import SecureConfig

        # 민감한 정보를 안전하게 저장
        config = SecureConfig(
            api_key="sk-1234567890abcdef",
            api_secret="secret_abc123",
            model="gpt-4"
        )

        # 로그에 출력해도 안전 (마스킹됨)
        print(config)  # SecureConfig(api_key=***MASKED***, api_secret=***MASKED***, model=gpt-4)

        # 실제 값은 안전하게 접근 가능
        actual_key = config.get_secret("api_key")  # "sk-1234567890abcdef"

        # dict 변환 시에도 마스킹
        config_dict = config.to_dict()  # {'api_key': '***MASKED***', ...}

        # 민감하지 않은 값만 가져오기
        safe_dict = config.to_dict(mask_secrets=False, include_only_safe=True)
        ```
    """

    # 민감한 정보로 간주할 키 패턴
    SENSITIVE_PATTERNS = [
        r".*key.*",
        r".*secret.*",
        r".*password.*",
        r".*token.*",
        r".*credential.*",
        r".*auth.*",
        r".*bearer.*",
    ]

    # 안전한 키 패턴 (민감하지 않은 것으로 명시적으로 허용)
    SAFE_PATTERNS = [
        r".*_key_id$",  # key_id는 안전 (actual key가 아님)
        r".*_public_key$",  # public key는 안전
        r"^model$",
        r"^region$",
        r"^endpoint$",
        r"^timeout$",
        r"^max_.*",
        r"^temperature$",
    ]

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: 설정 값들 (민감한 정보 포함 가능)
        """
        self._config: Dict[str, Any] = {}
        self._sensitive_keys: set = set()

        for key, value in kwargs.items():
            self._config[key] = value

            # 민감한 키 자동 감지
            if self._is_sensitive_key(key):
                self._sensitive_keys.add(key)

    def _is_sensitive_key(self, key: str) -> bool:
        """
        키가 민감한 정보인지 판단

        Args:
            key: 검사할 키 이름

        Returns:
            민감한 키면 True
        """
        key_lower = key.lower()

        # 안전한 패턴에 먼저 매칭 (우선순위)
        for pattern in self.SAFE_PATTERNS:
            if re.match(pattern, key_lower):
                return False

        # 민감한 패턴에 매칭
        for pattern in self.SENSITIVE_PATTERNS:
            if re.match(pattern, key_lower):
                return True

        return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        안전한 값 가져오기 (마스킹된 값 반환)

        Args:
            key: 키 이름
            default: 기본값

        Returns:
            값 (민감한 경우 마스킹)
        """
        value = self._config.get(key, default)

        if key in self._sensitive_keys:
            return "***MASKED***"

        return value

    def get_secret(self, key: str, default: Any = None) -> Any:
        """
        실제 비밀 값 가져오기 (마스킹 없음)

        주의: 이 메서드는 실제 민감한 값을 반환합니다.
        로그나 예외 메시지에 직접 사용하지 마세요.

        Args:
            key: 키 이름
            default: 기본값

        Returns:
            실제 값
        """
        return self._config.get(key, default)

    def set(self, key: str, value: Any, sensitive: Optional[bool] = None):
        """
        값 설정

        Args:
            key: 키 이름
            value: 값
            sensitive: 민감한 정보 여부 (None이면 자동 감지)
        """
        self._config[key] = value

        if sensitive is True:
            self._sensitive_keys.add(key)
        elif sensitive is False:
            self._sensitive_keys.discard(key)
        else:
            # 자동 감지
            if self._is_sensitive_key(key):
                self._sensitive_keys.add(key)

    def mark_sensitive(self, *keys: str):
        """
        특정 키를 민감한 정보로 표시

        Args:
            *keys: 민감한 정보로 표시할 키들
        """
        for key in keys:
            if key in self._config:
                self._sensitive_keys.add(key)

    def to_dict(
        self, mask_secrets: bool = True, include_only_safe: bool = False
    ) -> Dict[str, Any]:
        """
        딕셔너리로 변환

        Args:
            mask_secrets: 민감한 정보 마스킹 여부
            include_only_safe: 안전한 정보만 포함 (민감한 정보 제외)

        Returns:
            설정 딕셔너리
        """
        result = {}

        for key, value in self._config.items():
            if include_only_safe and key in self._sensitive_keys:
                continue  # 민감한 정보 제외

            if mask_secrets and key in self._sensitive_keys:
                result[key] = "***MASKED***"
            else:
                result[key] = value

        return result

    def __getitem__(self, key: str) -> Any:
        """dict처럼 접근 가능 (마스킹된 값)"""
        return self.get(key)

    def __setitem__(self, key: str, value: Any):
        """dict처럼 설정 가능"""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """in 연산자 지원"""
        return key in self._config

    def __repr__(self) -> str:
        """repr 오버라이드 (민감한 정보 마스킹)"""
        masked_config = self.to_dict(mask_secrets=True)
        items = [f"{k}={repr(v)}" for k, v in masked_config.items()]
        return f"SecureConfig({', '.join(items)})"

    def __str__(self) -> str:
        """str 오버라이드 (민감한 정보 마스킹)"""
        return self.__repr__()

    def keys(self):
        """dict.keys() 호환"""
        return self._config.keys()

    def values(self):
        """dict.values() 호환 (마스킹됨)"""
        return [self.get(k) for k in self._config.keys()]

    def items(self):
        """dict.items() 호환 (마스킹됨)"""
        return [(k, self.get(k)) for k in self._config.keys()]


# 편의 함수: 환경 변수에서 안전하게 로드
def load_from_env(prefix: str = "BEANLLM_") -> SecureConfig:
    """
    환경 변수에서 설정 로드

    Args:
        prefix: 환경 변수 접두사

    Returns:
        SecureConfig 인스턴스

    Example:
        ```python
        # 환경 변수:
        # BEANLLM_API_KEY=sk-123
        # BEANLLM_MODEL=gpt-4

        config = load_from_env("BEANLLM_")
        # SecureConfig(api_key=***MASKED***, model=gpt-4)
        ```
    """
    import os

    config_dict = {}

    for key, value in os.environ.items():
        if key.startswith(prefix):
            # 접두사 제거하고 소문자로 변환
            config_key = key[len(prefix) :].lower()
            config_dict[config_key] = value

    return SecureConfig(**config_dict)
