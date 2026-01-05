"""
beanllm.utils.resilience.error_tracker - Error Tracking and Monitoring
에러 추적 및 모니터링

이 모듈은 에러 추적, 분석, 보안 정제 기능을 제공합니다:
- 에러 발생 기록 및 통계
- 프로덕션 환경용 민감 정보 제거
- 스택 트레이스 정제
- 안전한 에러 응답 생성
"""

import re
import threading
import time
import traceback as tb_module
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern


@dataclass
class ErrorRecord:
    """에러 기록"""

    timestamp: float
    error_type: str
    error_message: str
    traceback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ErrorTracker:
    """
    에러 추적기

    에러 발생을 기록하고 분석
    """

    def __init__(self, max_records: int = 1000):
        self.max_records = max_records
        self.errors = deque(maxlen=max_records)
        self._lock = threading.Lock()

    def record(self, exception: Exception, metadata: Optional[Dict[str, Any]] = None):
        """에러 기록"""
        import traceback as tb

        with self._lock:
            record = ErrorRecord(
                timestamp=time.time(),
                error_type=type(exception).__name__,
                error_message=str(exception),
                traceback=tb.format_exc(),
                metadata=metadata or {},
            )
            self.errors.append(record)

    def get_recent_errors(self, n: int = 10) -> List[ErrorRecord]:
        """최근 에러 조회"""
        with self._lock:
            return list(self.errors)[-n:]

    def get_error_summary(self) -> Dict[str, Any]:
        """에러 요약 통계"""
        with self._lock:
            if not self.errors:
                return {"total_errors": 0, "error_types": {}, "error_rate": 0.0}

            # 에러 타입별 카운트
            type_counts = {}
            for error in self.errors:
                error_type = error.error_type
                type_counts[error_type] = type_counts.get(error_type, 0) + 1

            # 에러율 계산 (최근 1시간)
            now = time.time()
            recent_errors = sum(1 for e in self.errors if now - e.timestamp <= 3600)

            return {
                "total_errors": len(self.errors),
                "error_types": type_counts,
                "recent_errors_1h": recent_errors,
                "most_common_error": (
                    max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None
                ),
            }

    def clear(self):
        """에러 기록 초기화"""
        with self._lock:
            self.errors.clear()


# 전역 에러 트래커
_global_error_tracker = ErrorTracker()


def get_error_tracker() -> ErrorTracker:
    """전역 에러 트래커 가져오기"""
    return _global_error_tracker


class FallbackHandler:
    """
    Fallback 핸들러

    에러 발생 시 대체 전략 실행
    """

    def __init__(
        self,
        fallback_func: Optional[callable] = None,
        fallback_value: Optional[Any] = None,
        raise_on_fallback: bool = False,
    ):
        self.fallback_func = fallback_func
        self.fallback_value = fallback_value
        self.raise_on_fallback = raise_on_fallback

    def call(self, func: callable, *args, **kwargs) -> Any:
        """
        Fallback이 적용된 함수 호출

        Args:
            func: 실행할 함수
            *args, **kwargs: 함수 인자

        Returns:
            함수 실행 결과 또는 fallback 값
        """
        try:
            return func(*args, **kwargs)

        except Exception as e:
            if self.raise_on_fallback:
                raise

            # Fallback 전략 실행
            if self.fallback_func:
                return self.fallback_func(e, *args, **kwargs)
            else:
                return self.fallback_value


class ProductionErrorSanitizer:
    """
    프로덕션 환경용 에러 메시지 정제기

    민감한 정보를 제거하여 안전한 에러 메시지를 생성합니다:
    - API 키, 비밀번호 패턴 마스킹
    - 파일 경로 제거/축약
    - 스택 트레이스 간소화
    - 데이터베이스 스키마 정보 제거
    - IP 주소, 포트 번호 마스킹

    Security Benefits:
        - API 키 노출 방지
        - 내부 파일 구조 숨김
        - 데이터베이스 스키마 보호
        - 네트워크 토폴로지 보호
    """

    # 민감 정보 패턴
    PATTERNS: Dict[str, Pattern] = {
        # API 키 패턴 (예: sk-..., api_key_..., token_...)
        "api_key": re.compile(
            r"(api[_-]?key|token|secret|password|passwd|pwd)['\"\s:=]+([a-zA-Z0-9_\-./]{10,})",
            re.IGNORECASE,
        ),
        # 환경변수 패턴 (예: OPENAI_API_KEY=sk-...)
        "env_var": re.compile(
            r"([A-Z_]+_(?:API_KEY|TOKEN|SECRET|PASSWORD))['\"\s:=]+([a-zA-Z0-9_\-./]{10,})"
        ),
        # Bearer 토큰
        "bearer": re.compile(r"Bearer\s+([a-zA-Z0-9_\-./]{10,})", re.IGNORECASE),
        # 절대 파일 경로 (Unix/Windows)
        "abs_path": re.compile(r"(/[a-zA-Z0-9_./\-]+/[a-zA-Z0-9_./\-]+|[C-Z]:\\[^\s]+)"),
        # IP 주소
        "ipv4": re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
        # 포트 번호 포함 주소
        "host_port": re.compile(r"(localhost|127\.0\.0\.1|0\.0\.0\.0):(\d{2,5})"),
        # 데이터베이스 연결 문자열
        "db_conn": re.compile(
            r"(postgresql|mysql|mongodb)://([^:]+):([^@]+)@([^:/]+)(:\d+)?(/[^\s]+)?",
            re.IGNORECASE,
        ),
        # SQL 테이블/컬럼명
        "sql_schema": re.compile(r"\b(table|column|schema)\s+['\"]?([a-zA-Z0-9_]+)['\"]?", re.IGNORECASE),
    }

    # 마스킹 문자열
    MASK_STR = "***"
    MASK_PATH = "[PATH]"
    MASK_IP = "[IP]"
    MASK_PORT = "[PORT]"
    MASK_DB = "[DB_CONN]"

    @classmethod
    def sanitize_message(cls, message: str, production: bool = True) -> str:
        """
        에러 메시지 정제

        Args:
            message: 원본 에러 메시지
            production: 프로덕션 모드 (기본: True)

        Returns:
            정제된 에러 메시지

        Example:
            >>> ProductionErrorSanitizer.sanitize_message(
            ...     "API key sk-1234567890 failed at /home/user/app/config.py:42"
            ... )
            'API key *** failed at [PATH]'
        """
        if not production:
            return message

        sanitized = message

        # API 키/토큰 마스킹
        sanitized = cls.PATTERNS["api_key"].sub(rf"\1={cls.MASK_STR}", sanitized)
        sanitized = cls.PATTERNS["env_var"].sub(rf"\1={cls.MASK_STR}", sanitized)
        sanitized = cls.PATTERNS["bearer"].sub(f"Bearer {cls.MASK_STR}", sanitized)

        # 데이터베이스 연결 문자열 마스킹
        sanitized = cls.PATTERNS["db_conn"].sub(cls.MASK_DB, sanitized)

        # 파일 경로 마스킹
        sanitized = cls.PATTERNS["abs_path"].sub(cls.MASK_PATH, sanitized)

        # IP 주소 마스킹 (localhost 제외)
        sanitized = cls.PATTERNS["ipv4"].sub(
            lambda m: m.group(0) if m.group(0).startswith("127.") else cls.MASK_IP, sanitized
        )

        # 포트 번호 마스킹
        sanitized = cls.PATTERNS["host_port"].sub(rf"\1:{cls.MASK_PORT}", sanitized)

        # SQL 스키마 정보 마스킹
        sanitized = cls.PATTERNS["sql_schema"].sub(rf"\1 {cls.MASK_STR}", sanitized)

        return sanitized

    @classmethod
    def sanitize_traceback(cls, traceback_str: str, production: bool = True, max_frames: int = 3) -> str:
        """
        스택 트레이스 정제

        Args:
            traceback_str: 원본 트레이스백 문자열
            production: 프로덕션 모드 (기본: True)
            max_frames: 표시할 최대 프레임 수 (프로덕션 모드)

        Returns:
            정제된 트레이스백

        Example:
            >>> ProductionErrorSanitizer.sanitize_traceback(
            ...     "File '/home/user/app.py', line 42..."
            ... )
            'File [PATH], line 42...'
        """
        if not production:
            return traceback_str

        # 파일 경로 마스킹
        sanitized = cls.PATTERNS["abs_path"].sub(cls.MASK_PATH, traceback_str)

        # 프로덕션 모드: 스택 프레임 수 제한
        lines = sanitized.split("\n")
        if len(lines) > max_frames * 2:  # 각 프레임은 보통 2줄
            # 처음 몇 프레임만 유지
            sanitized = "\n".join(lines[: max_frames * 2] + ["  ... (truncated for security)"])

        return sanitized

    @classmethod
    def create_safe_error(cls, exception: Exception, production: bool = True) -> Dict[str, Any]:
        """
        안전한 에러 응답 생성

        Args:
            exception: 원본 예외
            production: 프로덕션 모드 (기본: True)

        Returns:
            안전한 에러 정보 딕셔너리

        Example:
            >>> try:
            ...     raise ValueError("API key sk-123 is invalid")
            ... except Exception as e:
            ...     safe_error = ProductionErrorSanitizer.create_safe_error(e)
            ...     print(safe_error["message"])
            'API key *** is invalid'
        """
        error_type = type(exception).__name__
        error_message = str(exception)

        # 메시지 정제
        safe_message = cls.sanitize_message(error_message, production)

        result = {
            "error_type": error_type,
            "message": safe_message,
            "production": production,
        }

        # 스택 트레이스 (프로덕션에서는 제한적)
        if production:
            # 프로덕션: 간소화된 트레이스
            traceback_str = tb_module.format_exc()
            result["traceback"] = cls.sanitize_traceback(traceback_str, production, max_frames=2)
        else:
            # 개발: 전체 트레이스
            result["traceback"] = tb_module.format_exc()

        return result


def sanitize_error_message(message: str, production: bool = True) -> str:
    """
    에러 메시지 정제 (헬퍼 함수)

    Args:
        message: 원본 에러 메시지
        production: 프로덕션 모드

    Returns:
        정제된 에러 메시지
    """
    return ProductionErrorSanitizer.sanitize_message(message, production)


def create_safe_error_response(exception: Exception, production: bool = True) -> Dict[str, Any]:
    """
    안전한 에러 응답 생성 (헬퍼 함수)

    Args:
        exception: 원본 예외
        production: 프로덕션 모드

    Returns:
        안전한 에러 정보
    """
    return ProductionErrorSanitizer.create_safe_error(exception, production)
