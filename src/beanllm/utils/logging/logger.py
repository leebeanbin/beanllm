"""
Logger
독립적인 로거 (loguru 대체)
API 키 마스킹 지원
"""

import logging
import re
import sys


class SecureFormatter(logging.Formatter):
    """
    API 키 마스킹을 지원하는 로그 포맷터
    """

    # API 키 패턴
    API_KEY_PATTERNS = [
        re.compile(
            r"(api[_-]?key|token|secret|password|passwd|pwd)['\"\s:=]+([a-zA-Z0-9_\-./]{10,})",
            re.IGNORECASE,
        ),
        re.compile(r"([A-Z_]+_(?:API_KEY|TOKEN|SECRET|PASSWORD))['\"\s:=]+([a-zA-Z0-9_\-./]{10,})"),
        re.compile(r"Bearer\s+([a-zA-Z0-9_\-./]{10,})", re.IGNORECASE),
    ]

    MASK_STR = "***MASKED***"

    def format(self, record: logging.LogRecord) -> str:
        """로그 메시지 포맷 (API 키 마스킹)"""
        # 원본 메시지 가져오기
        original_msg = record.getMessage()

        # API 키 마스킹
        masked_msg = original_msg
        for pattern in self.API_KEY_PATTERNS:
            if pattern == self.API_KEY_PATTERNS[0]:  # api_key 패턴
                masked_msg = pattern.sub(rf"\1={self.MASK_STR}", masked_msg)
            elif pattern == self.API_KEY_PATTERNS[1]:  # env_var 패턴
                masked_msg = pattern.sub(rf"\1={self.MASK_STR}", masked_msg)
            else:  # Bearer 토큰
                masked_msg = pattern.sub(f"Bearer {self.MASK_STR}", masked_msg)

        # 마스킹된 메시지로 레코드 업데이트
        record.msg = masked_msg
        record.args = ()

        # 기본 포맷터로 포맷
        return super().format(record)


def get_logger(name: str, level: str = "INFO", secure: bool = True) -> logging.Logger:
    """
    로거 생성

    Args:
        name: 로거 이름
        level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        secure: API 키 마스킹 활성화 (기본: True)

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # 이미 핸들러가 있으면 중복 추가 방지
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))

    # 콘솔 핸들러
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    # 포맷터 (보안 활성화 시 SecureFormatter 사용)
    if secure:
        formatter = SecureFormatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


# 패키지 레벨 로거 (보안 활성화)
logger = get_logger("llm_model_manager", secure=True)
