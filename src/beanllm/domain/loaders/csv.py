"""
CSV Loader

CSV 파일 로더
"""

import csv
import logging
import mmap
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

class CSVLoader(BaseDocumentLoader):
    """
    CSV 로더 (중복 코드 제거 최적화)

    Example:
        ```python
        from beanllm.domain.loaders import CSVLoader

        # 행별로 문서 생성
        loader = CSVLoader("data.csv")
        docs = loader.load()

        # 특정 컬럼만 content로
        loader = CSVLoader("data.csv", content_columns=["text", "description"])
        ```
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
        encoding: str = "utf-8",
    ):
        """
        Args:
            file_path: CSV 경로
            content_columns: content로 사용할 컬럼들 (None이면 전체)
            metadata_columns: metadata로 저장할 컬럼들
            encoding: 인코딩
        """
        self.file_path = Path(file_path)
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns
        self.encoding = encoding

    def _create_content_from_row(self, row: dict) -> str:
        """
        CSV 행에서 content 생성 (헬퍼 메서드 - 중복 제거)

        Args:
            row: CSV 행 딕셔너리

        Returns:
            생성된 content 문자열
        """
        if self.content_columns:
            content_parts = [
                f"{col}: {row.get(col, '')}"
                for col in self.content_columns
                if col in row
            ]
            return "\n".join(content_parts)
        else:
            # 모든 컬럼 사용
            return "\n".join([f"{k}: {v}" for k, v in row.items()])

    def _create_metadata_from_row(self, row: dict, row_index: int) -> dict:
        """
        CSV 행에서 metadata 생성 (헬퍼 메서드 - 중복 제거)

        Args:
            row: CSV 행 딕셔너리
            row_index: 행 번호

        Returns:
            생성된 metadata 딕셔너리
        """
        metadata = {"source": str(self.file_path), "row": row_index}

        if self.metadata_columns:
            for col in self.metadata_columns:
                if col in row:
                    metadata[col] = row[col]

        return metadata

    def load(self) -> List[Document]:
        """CSV 로딩 (행별 문서)"""
        documents = []

        try:
            with open(self.file_path, "r", encoding=self.encoding) as f:
                reader = csv.DictReader(f)

                for i, row in enumerate(reader):
                    # 헬퍼 메서드 사용 (중복 제거)
                    content = self._create_content_from_row(row)
                    metadata = self._create_metadata_from_row(row, i)

                    documents.append(Document(content=content, metadata=metadata))

            logger.info(f"Loaded {len(documents)} rows from {self.file_path}")
            return documents

        except Exception as e:
            logger.error(f"Failed to load CSV {self.file_path}: {e}")
            raise

    def lazy_load(self):
        """지연 로딩"""
        with open(self.file_path, "r", encoding=self.encoding) as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                # 헬퍼 메서드 사용 (중복 제거)
                content = self._create_content_from_row(row)
                metadata = self._create_metadata_from_row(row, i)

                yield Document(content=content, metadata=metadata)


