"""
Text Loader

텍스트 파일 로더 (mmap 최적화)
"""

import logging
import mmap
import os
import re
from pathlib import Path
from typing import Iterator, List, Optional, Union

from .base import BaseDocumentLoader
from .security import validate_file_path
from .types import Document

try:
    from beanllm.utils.logger import get_logger
except ImportError:
    def get_logger(name: str):
        return logging.getLogger(name)

logger = get_logger(__name__)

class TextLoader(BaseDocumentLoader):
    """
    텍스트 파일 로더 (메모리 매핑 I/O 최적화)

    Features:
        - mmap 기반 대용량 파일 처리 (메모리 효율적)
        - 자동 임계값 기반 mmap 활성화 (10MB+)
        - 스트리밍 로드 (청크 단위 처리)
        - 인코딩 자동 감지

    Performance:
        - 일반 read: 전체 파일을 메모리에 로드
        - mmap: OS 수준에서 파일을 메모리에 매핑 (lazy loading)

    Example:
        ```python
        from beanllm.domain.loaders import TextLoader

        # 기본 로딩 (자동 mmap for 10MB+)
        loader = TextLoader("file.txt", encoding="utf-8")
        docs = loader.load()

        # 강제 mmap 사용
        loader = TextLoader("large.txt", use_mmap=True)
        docs = loader.load()

        # 스트리밍 로딩 (청크 단위)
        loader = TextLoader("huge.txt", chunk_size=1024*1024)
        for doc in loader.load_streaming():
            process(doc)
        ```
    """

    # 10MB 이상 파일은 자동으로 mmap 사용
    MMAP_THRESHOLD_BYTES = 10 * 1024 * 1024

    def __init__(
        self,
        file_path: Union[str, Path],
        encoding: str = "utf-8",
        autodetect_encoding: bool = True,
        validate_path: bool = True,
        use_mmap: Optional[bool] = None,
        chunk_size: int = 1024 * 1024,  # 1MB chunks for streaming
    ):
        """
        Args:
            file_path: 파일 경로
            encoding: 인코딩
            autodetect_encoding: 인코딩 자동 감지
            validate_path: 경로 검증 여부 (기본: True, Path Traversal 방지)
            use_mmap: mmap 사용 여부 (None이면 파일 크기 기반 자동)
            chunk_size: 스트리밍 청크 크기 (바이트)
        """
        # 경로 검증 (Path Traversal 방지)
        if validate_path:
            self.file_path = validate_file_path(file_path)
        else:
            self.file_path = Path(file_path)

        self.encoding = encoding
        self.autodetect_encoding = autodetect_encoding
        self.use_mmap = use_mmap
        self.chunk_size = chunk_size

    def load(self) -> List[Document]:
        """파일 로딩"""
        try:
            content = self._read_file()
            return [
                Document(
                    content=content,
                    metadata={"source": str(self.file_path), "encoding": self.encoding},
                )
            ]
        except Exception as e:
            logger.error(f"Failed to load {self.file_path}: {e}")
            raise

    def lazy_load(self):
        """지연 로딩"""
        yield from self.load()

    def load_streaming(self):
        """
        스트리밍 로딩 (청크 단위, 메모리 효율적)

        대용량 파일을 청크 단위로 읽어 yield합니다.
        각 청크가 별도의 Document로 반환됩니다.

        Yields:
            Document: 청크별 Document

        Example:
            ```python
            loader = TextLoader("huge.log", chunk_size=1024*1024)
            for doc in loader.load_streaming():
                # 각 1MB 청크 처리
                process_chunk(doc.content)
            ```
        """
        try:
            # mmap 사용 여부 결정
            should_use_mmap = self._should_use_mmap()

            if should_use_mmap:
                # mmap 기반 스트리밍
                yield from self._stream_with_mmap()
            else:
                # 일반 파일 읽기 기반 스트리밍
                yield from self._stream_normal()

        except Exception as e:
            logger.error(f"Failed to stream {self.file_path}: {e}")
            raise

    def _read_file(self) -> str:
        """파일 읽기 (mmap 또는 일반 읽기)"""
        # mmap 사용 여부 결정
        should_use_mmap = self._should_use_mmap()

        if should_use_mmap:
            return self._read_with_mmap()
        else:
            return self._read_normal()

    def _should_use_mmap(self) -> bool:
        """mmap 사용 여부 결정"""
        # 명시적 설정이 있으면 사용
        if self.use_mmap is not None:
            return self.use_mmap

        # 파일 크기 확인
        try:
            file_size = os.path.getsize(self.file_path)
            return file_size >= self.MMAP_THRESHOLD_BYTES
        except Exception:
            return False

    def _read_with_mmap(self) -> str:
        """mmap을 사용한 파일 읽기 (메모리 효율적)"""
        try:
            with open(self.file_path, "r+b") as f:
                # mmap 생성
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                    # 인코딩 자동 감지
                    if self.autodetect_encoding:
                        content = self._decode_with_encoding_detection(mmapped_file[:])
                    else:
                        content = mmapped_file[:].decode(self.encoding)

                    logger.debug(f"Read {len(content)} chars with mmap from {self.file_path}")
                    return content

        except Exception as e:
            logger.warning(f"mmap read failed, falling back to normal read: {e}")
            return self._read_normal()

    def _read_normal(self) -> str:
        """일반 파일 읽기"""
        # 인코딩 자동 감지
        if self.autodetect_encoding:
            try:
                with open(self.file_path, "r", encoding=self.encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                # UTF-8 실패 시 다른 인코딩 시도
                for encoding in ["cp949", "euc-kr", "latin-1"]:
                    try:
                        with open(self.file_path, "r", encoding=encoding) as f:
                            content = f.read()
                            self.encoding = encoding
                            logger.info(f"Auto-detected encoding: {encoding}")
                            return content
                    except UnicodeDecodeError:
                        continue
                raise
        else:
            with open(self.file_path, "r", encoding=self.encoding) as f:
                return f.read()

    def _decode_with_encoding_detection(self, data: bytes) -> str:
        """인코딩 자동 감지하여 디코딩"""
        # 먼저 설정된 인코딩 시도
        try:
            return data.decode(self.encoding)
        except UnicodeDecodeError:
            # 다른 인코딩 시도
            for encoding in ["cp949", "euc-kr", "latin-1"]:
                try:
                    content = data.decode(encoding)
                    self.encoding = encoding
                    logger.info(f"Auto-detected encoding: {encoding}")
                    return content
                except UnicodeDecodeError:
                    continue
            # 모두 실패하면 에러와 함께 원래 인코딩 시도
            raise

    def _stream_with_mmap(self):
        """mmap 기반 스트리밍"""
        try:
            with open(self.file_path, "r+b") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                    file_size = len(mmapped_file)
                    chunk_index = 0

                    for offset in range(0, file_size, self.chunk_size):
                        # 청크 읽기
                        chunk_end = min(offset + self.chunk_size, file_size)
                        chunk_bytes = mmapped_file[offset:chunk_end]

                        # 디코딩
                        if self.autodetect_encoding and chunk_index == 0:
                            # 첫 청크에서만 인코딩 감지
                            chunk_text = self._decode_with_encoding_detection(chunk_bytes)
                        else:
                            chunk_text = chunk_bytes.decode(self.encoding)

                        # Document 생성
                        metadata = {
                            "source": str(self.file_path),
                            "encoding": self.encoding,
                            "chunk_index": chunk_index,
                            "chunk_size": len(chunk_bytes),
                            "file_size": file_size,
                            "offset": offset,
                        }

                        yield Document(content=chunk_text, metadata=metadata)
                        chunk_index += 1

            logger.debug(f"Streamed {chunk_index} chunks with mmap from {self.file_path}")

        except Exception as e:
            logger.warning(f"mmap streaming failed, falling back to normal: {e}")
            yield from self._stream_normal()

    def _stream_normal(self):
        """일반 파일 읽기 기반 스트리밍"""
        chunk_index = 0

        # 인코딩 감지 (첫 청크에서)
        if self.autodetect_encoding:
            with open(self.file_path, "rb") as f:
                first_chunk_bytes = f.read(self.chunk_size)
                self._decode_with_encoding_detection(first_chunk_bytes)

        # 스트리밍
        with open(self.file_path, "r", encoding=self.encoding) as f:
            while True:
                chunk_text = f.read(self.chunk_size)
                if not chunk_text:
                    break

                metadata = {
                    "source": str(self.file_path),
                    "encoding": self.encoding,
                    "chunk_index": chunk_index,
                    "chunk_size": len(chunk_text),
                }

                yield Document(content=chunk_text, metadata=metadata)
                chunk_index += 1

        logger.debug(f"Streamed {chunk_index} chunks (normal) from {self.file_path}")


