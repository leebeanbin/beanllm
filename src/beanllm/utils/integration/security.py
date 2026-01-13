"""
Security Utilities - 보안 관련 유틸리티
책임: 에러 메시지 마스킹 등 보안 기능 제공
"""

import re
from typing import Optional


def sanitize_error_message(error: Exception) -> str:
    """
    에러 메시지에서 API 키 마스킹 (범용 Helper 함수)

    Args:
        error: 원본 예외

    Returns:
        마스킹된 에러 메시지

    Example:
        ```python
        try:
            # API 호출
        except Exception as e:
            safe_message = sanitize_error_message(e)
            logger.error(safe_message)
        ```
    """
    error_str = str(error)
    try:
        from ..resilience.error_tracker import ProductionErrorSanitizer
        return ProductionErrorSanitizer.sanitize_message(error_str, production=True)
    except ImportError:
        # ProductionErrorSanitizer가 없으면 기본 마스킹
        return _basic_sanitize(error_str)


def _basic_sanitize(message: str) -> str:
    """
    기본 마스킹 (ProductionErrorSanitizer가 없을 때)

    Args:
        message: 원본 메시지

    Returns:
        마스킹된 메시지
    """
    patterns = [
        # API 키 패턴
        (
            r"(api[_-]?key|token|secret|password|passwd|pwd)['\"\s:=]+([a-zA-Z0-9_\-./]{10,})",
            r"\1=***MASKED***",
            re.IGNORECASE,
        ),
        # 환경변수 패턴
        (
            r"([A-Z_]+_(?:API_KEY|TOKEN|SECRET|PASSWORD))['\"\s:=]+([a-zA-Z0-9_\-./]{10,})",
            r"\1=***MASKED***",
            0,
        ),
        # Bearer 토큰
        (
            r"Bearer\s+([a-zA-Z0-9_\-./]{10,})",
            "Bearer ***MASKED***",
            re.IGNORECASE,
        ),
    ]

    sanitized = message
    for pattern, replacement, flags in patterns:
        sanitized = re.sub(pattern, replacement, sanitized, flags=flags)

    return sanitized

