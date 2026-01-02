"""
Security utilities for loaders

Path Traversal 방지를 위한 경로 검증 유틸리티
파일 크기 제한 검증
"""

import logging
import os
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


# 보안: Path Traversal 방지를 위한 허용 디렉토리 설정
# 환경변수로 설정 가능 (쉼표로 구분)
_ALLOWED_DIRS_ENV = os.getenv("BEANLLM_ALLOWED_DIRS", "")
if _ALLOWED_DIRS_ENV:
    ALLOWED_DIRECTORIES = [Path(d.strip()) for d in _ALLOWED_DIRS_ENV.split(",")]
else:
    # 기본값: 현재 작업 디렉토리 및 하위
    ALLOWED_DIRECTORIES = [Path.cwd()]

# 파일 크기 제한 (기본: 100MB, 환경변수로 설정 가능)
# DoS 공격 방지: 과도하게 큰 파일 로드 차단
MAX_FILE_SIZE_BYTES = int(os.getenv("BEANLLM_MAX_FILE_SIZE", str(100 * 1024 * 1024)))


def validate_file_size(file_path: Union[str, Path], max_size: int = MAX_FILE_SIZE_BYTES) -> None:
    """
    파일 크기 검증 (DoS 방지)

    Args:
        file_path: 검증할 파일 경로
        max_size: 최대 파일 크기 (바이트)

    Raises:
        ValueError: 파일이 너무 큰 경우

    Example:
        ```python
        from beanllm.domain.loaders.security import validate_file_size

        # 파일 크기 검증
        validate_file_size("./data/large_file.pdf")
        ```
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size = path.stat().st_size

    if file_size > max_size:
        max_mb = max_size / (1024 * 1024)
        actual_mb = file_size / (1024 * 1024)
        raise ValueError(
            f"File too large: {file_path} ({actual_mb:.2f} MB) exceeds maximum size ({max_mb:.2f} MB). "
            f"Set BEANLLM_MAX_FILE_SIZE environment variable to increase limit."
        )


def validate_file_path(
    file_path: Union[str, Path],
    allow_parent_access: bool = False,
    check_size: bool = True,
    max_size: int = MAX_FILE_SIZE_BYTES,
) -> Path:
    """
    파일 경로 검증 (Path Traversal 방지, 파일 크기 제한)

    Args:
        file_path: 검증할 파일 경로
        allow_parent_access: 상위 디렉토리 접근 허용 여부
        check_size: 파일 크기 검증 여부 (기본: True)
        max_size: 최대 파일 크기 (바이트)

    Returns:
        검증된 절대 경로

    Raises:
        ValueError: 허용되지 않은 경로 또는 파일이 너무 큰 경우
        FileNotFoundError: 파일이 존재하지 않는 경우 (check_size=True일 때)

    Security:
        - 절대 경로로 정규화
        - 심볼릭 링크 해결
        - 허용된 디렉토리 외부 접근 차단
        - 파일 크기 제한 (DoS 방지)

    Example:
        ```python
        from beanllm.domain.loaders.security import validate_file_path

        # 안전한 경로 검증
        safe_path = validate_file_path("./data/file.txt")

        # 허용되지 않은 경로는 에러 발생
        # validate_file_path("../../../etc/passwd")  # ValueError
        ```
    """
    try:
        # 절대 경로로 변환 (심볼릭 링크 해결)
        path = Path(file_path).resolve(strict=False)

        # 상위 디렉토리 접근 차단
        if not allow_parent_access and ".." in str(file_path):
            raise ValueError(
                f"Path traversal detected: {file_path} contains '..' (parent directory reference)"
            )

        # 허용된 디렉토리 확인
        allowed = any(
            str(path).startswith(str(allowed_dir.resolve()))
            for allowed_dir in ALLOWED_DIRECTORIES
        )

        if not allowed:
            raise ValueError(
                f"Access denied: {file_path} is not in allowed directories.\n"
                f"Allowed directories: {[str(d) for d in ALLOWED_DIRECTORIES]}\n"
                f"Set BEANLLM_ALLOWED_DIRS environment variable to customize."
            )

        # 파일 크기 검증 (선택적)
        if check_size:
            validate_file_size(path, max_size)

        return path

    except Exception as e:
        logger.error(f"Path validation failed for {file_path}: {e}")
        raise
