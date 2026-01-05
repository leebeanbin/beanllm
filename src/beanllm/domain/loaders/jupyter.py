"""
Jupyter Loader

Jupyter Notebook 로더
"""

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

class JupyterLoader(BaseDocumentLoader):
    """
    Jupyter Notebook 로더 (.ipynb, 2024-2025)

    Jupyter notebook 파일을 로드하여 코드 셀, 마크다운 셀, 출력을 추출합니다.

    Features:
    - 코드 셀 추출 (실행 순서 보존)
    - 마크다운 셀 추출
    - 셀 출력 포함/제외 옵션
    - 메타데이터 보존 (셀 타입, 실행 횟수)

    Example:
        ```python
        from beanllm.domain.loaders import JupyterLoader

        # 기본 (출력 포함)
        loader = JupyterLoader("analysis.ipynb", include_outputs=True)
        docs = loader.load()

        # 코드만 (출력 제외)
        loader = JupyterLoader("notebook.ipynb", include_outputs=False)
        docs = loader.load()

        # 셀 타입 필터링
        loader = JupyterLoader(
            "notebook.ipynb",
            filter_cell_types=["code"]  # 코드 셀만
        )
        docs = loader.load()
        ```
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        include_outputs: bool = True,
        filter_cell_types: Optional[List[str]] = None,
        concatenate_cells: bool = True,
        **kwargs,
    ):
        """
        Args:
            file_path: .ipynb 파일 경로
            include_outputs: 셀 출력 포함 여부 (기본: True)
            filter_cell_types: 포함할 셀 타입 (기본: None = 모두)
                - ["code"]: 코드 셀만
                - ["markdown"]: 마크다운 셀만
                - ["code", "markdown"]: 둘 다
            concatenate_cells: 모든 셀을 하나의 Document로 결합 (기본: True)
            **kwargs: 추가 파라미터
        """
        self.file_path = Path(file_path)
        self.include_outputs = include_outputs
        self.filter_cell_types = filter_cell_types
        self.concatenate_cells = concatenate_cells

    def load(self) -> List[Document]:
        """Jupyter Notebook 로딩"""
        try:
            import nbformat
        except ImportError:
            raise ImportError(
                "nbformat is required for JupyterLoader. Install: pip install nbformat"
            )

        try:
            # Notebook 로드
            with open(self.file_path, "r", encoding="utf-8") as f:
                notebook = nbformat.read(f, as_version=4)

            # 메타데이터 추출
            nb_metadata = {
                "source": str(self.file_path),
                "kernel": notebook.metadata.get("kernelspec", {}).get("name", "unknown"),
                "language": notebook.metadata.get("kernelspec", {}).get("language", "unknown"),
            }

            # 셀 처리
            if self.concatenate_cells:
                # 모든 셀을 하나의 Document로
                content_parts = []

                for idx, cell in enumerate(notebook.cells):
                    # 셀 타입 필터링
                    if self.filter_cell_types and cell.cell_type not in self.filter_cell_types:
                        continue

                    cell_content = self._format_cell(cell, idx)
                    if cell_content:
                        content_parts.append(cell_content)

                combined_content = ("\n\n" + "="*80 + "\n\n").join(content_parts)

                return [Document(content=combined_content, metadata=nb_metadata)]

            else:
                # 각 셀을 별도 Document로
                documents = []

                for idx, cell in enumerate(notebook.cells):
                    if self.filter_cell_types and cell.cell_type not in self.filter_cell_types:
                        continue

                    cell_content = self._format_cell(cell, idx)
                    if cell_content:
                        cell_metadata = nb_metadata.copy()
                        cell_metadata.update({
                            "cell_index": idx,
                            "cell_type": cell.cell_type,
                            "execution_count": cell.get("execution_count"),
                        })

                        documents.append(Document(content=cell_content, metadata=cell_metadata))

                return documents

        except Exception as e:
            logger.error(f"Failed to load Jupyter notebook {self.file_path}: {e}")
            raise

    def lazy_load(self):
        """지연 로딩"""
        yield from self.load()

    def _format_cell(self, cell, idx: int) -> str:
        """셀 포맷팅"""
        parts = []

        # 셀 헤더
        cell_type = cell.cell_type.upper()
        exec_count = cell.get("execution_count", "")
        if exec_count:
            header = f"[{idx}] {cell_type} (execution {exec_count})"
        else:
            header = f"[{idx}] {cell_type}"

        parts.append(header)
        parts.append("-" * 80)

        # 셀 소스 코드/마크다운
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)

        if source.strip():
            parts.append(source)

        # 출력 (코드 셀만, include_outputs=True일 때)
        if self.include_outputs and cell.cell_type == "code":
            outputs = cell.get("outputs", [])
            if outputs:
                parts.append("\n--- OUTPUT ---")
                for output in outputs:
                    output_text = self._format_output(output)
                    if output_text:
                        parts.append(output_text)

        return "\n".join(parts)

    def _format_output(self, output) -> str:
        """셀 출력 포맷팅"""
        output_type = output.get("output_type", "")

        if output_type == "stream":
            # 표준 출력/에러
            text = output.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            return text

        elif output_type == "execute_result" or output_type == "display_data":
            # 실행 결과/디스플레이 데이터
            data = output.get("data", {})

            # 텍스트 표현 우선
            if "text/plain" in data:
                text = data["text/plain"]
                if isinstance(text, list):
                    text = "".join(text)
                return text

            # HTML (간단히 표시)
            elif "text/html" in data:
                return "[HTML OUTPUT]"

            # 이미지 (경로 표시)
            elif any(k.startswith("image/") for k in data.keys()):
                image_formats = [k for k in data.keys() if k.startswith("image/")]
                return f"[IMAGE: {', '.join(image_formats)}]"

        elif output_type == "error":
            # 에러
            ename = output.get("ename", "Error")
            evalue = output.get("evalue", "")
            traceback = output.get("traceback", [])

            error_parts = [f"{ename}: {evalue}"]
            if traceback:
                error_parts.append("\n".join(traceback))

            return "\n".join(error_parts)

        return ""


