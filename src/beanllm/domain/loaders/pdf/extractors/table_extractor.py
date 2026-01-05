"""
테이블 메타데이터 추출 및 관리

Document 리스트에서 테이블 메타데이터를 추출하여 구조화된 형태로 제공합니다.
"""

from pathlib import Path
from typing import List, Optional


class TableExtractor:
    """
    테이블 메타데이터 추출기

    Document 리스트에서 테이블 정보를 추출하여 DataFrame으로 변환하거나
    구조화된 형태로 조회할 수 있게 합니다.

    Example:
        ```python
        from beanllm.domain.loaders import beanPDFLoader
        from beanllm.domain.loaders.pdf.extractors import TableExtractor

        # PDF 로딩
        loader = beanPDFLoader("report.pdf", extract_tables=True)
        docs = loader.load()

        # 테이블 추출
        extractor = TableExtractor(docs)

        # 모든 테이블을 DataFrame 리스트로
        tables = extractor.get_all_tables()
        for table in tables:
            print(table['dataframe'])
            print(f"Page: {table['page']}, Confidence: {table['confidence']}")

        # 특정 페이지의 테이블만
        page_tables = extractor.get_tables_by_page(0)

        # 고품질 테이블만 (confidence >= 0.8)
        high_quality = extractor.get_high_quality_tables(min_confidence=0.8)
        ```
    """

    def __init__(self, documents: List):
        """
        Args:
            documents: beanPDFLoader.load() 결과 (Document 리스트)
        """
        self.documents = documents
        self._tables_cache = None

    def get_all_tables(self) -> List[dict]:
        """
        모든 테이블 메타데이터 추출

        Returns:
            테이블 정보 리스트, 각 항목은:
                - page: 페이지 번호 (0-based)
                - table_index: 페이지 내 테이블 인덱스
                - rows: 행 수
                - cols: 열 수
                - confidence: 신뢰도 (0.0 ~ 1.0)
                - has_dataframe: DataFrame 사용 가능 여부
                - has_markdown: Markdown 사용 가능 여부
                - has_csv: CSV 사용 가능 여부
                - source: 소스 파일 경로
                - dataframe: pandas DataFrame (있는 경우)
                - markdown: Markdown 문자열 (있는 경우)
                - csv: CSV 문자열 (있는 경우)
        """
        if self._tables_cache is not None:
            return self._tables_cache

        tables = []

        for doc in self.documents:
            if "tables" not in doc.metadata:
                continue

            page = doc.metadata.get("page", 0)
            source = doc.metadata.get("source", "")

            for table_meta in doc.metadata["tables"]:
                table_info = {
                    "page": page,
                    "table_index": table_meta.get("table_index", 0),
                    "rows": table_meta.get("rows", 0),
                    "cols": table_meta.get("cols", 0),
                    "confidence": table_meta.get("confidence", 0.0),
                    "has_dataframe": table_meta.get("has_dataframe", False),
                    "has_markdown": table_meta.get("has_markdown", False),
                    "has_csv": table_meta.get("has_csv", False),
                    "source": source,
                }

                # 실제 데이터는 원본 Document에서 가져와야 함
                # (메타데이터에는 요약 정보만 있음)
                # 여기서는 메타데이터만 제공

                tables.append(table_info)

        self._tables_cache = tables
        return tables

    def get_tables_by_page(self, page: int) -> List[dict]:
        """
        특정 페이지의 테이블만 추출

        Args:
            page: 페이지 번호 (0-based)

        Returns:
            해당 페이지의 테이블 리스트
        """
        all_tables = self.get_all_tables()
        return [t for t in all_tables if t["page"] == page]

    def get_high_quality_tables(self, min_confidence: float = 0.8) -> List[dict]:
        """
        고품질 테이블만 추출 (신뢰도 기준)

        Args:
            min_confidence: 최소 신뢰도 (기본: 0.8)

        Returns:
            신뢰도가 min_confidence 이상인 테이블 리스트
        """
        all_tables = self.get_all_tables()
        return [t for t in all_tables if t["confidence"] >= min_confidence]

    def get_tables_by_size(
        self,
        min_rows: Optional[int] = None,
        max_rows: Optional[int] = None,
        min_cols: Optional[int] = None,
        max_cols: Optional[int] = None
    ) -> List[dict]:
        """
        크기 기준으로 테이블 필터링

        Args:
            min_rows: 최소 행 수
            max_rows: 최대 행 수
            min_cols: 최소 열 수
            max_cols: 최대 열 수

        Returns:
            조건을 만족하는 테이블 리스트
        """
        all_tables = self.get_all_tables()
        filtered = all_tables

        if min_rows is not None:
            filtered = [t for t in filtered if t["rows"] >= min_rows]
        if max_rows is not None:
            filtered = [t for t in filtered if t["rows"] <= max_rows]
        if min_cols is not None:
            filtered = [t for t in filtered if t["cols"] >= min_cols]
        if max_cols is not None:
            filtered = [t for t in filtered if t["cols"] <= max_cols]

        return filtered

    def get_summary(self) -> dict:
        """
        테이블 추출 요약 정보

        Returns:
            요약 정보:
                - total_tables: 전체 테이블 수
                - pages_with_tables: 테이블이 있는 페이지 수
                - avg_confidence: 평균 신뢰도
                - tables_by_page: 페이지별 테이블 수
                - high_quality_count: 고품질 테이블 수 (confidence >= 0.8)
        """
        all_tables = self.get_all_tables()

        if not all_tables:
            return {
                "total_tables": 0,
                "pages_with_tables": 0,
                "avg_confidence": 0.0,
                "tables_by_page": {},
                "high_quality_count": 0,
            }

        pages_with_tables = set(t["page"] for t in all_tables)
        avg_confidence = sum(t["confidence"] for t in all_tables) / len(all_tables)
        high_quality = len([t for t in all_tables if t["confidence"] >= 0.8])

        tables_by_page = {}
        for t in all_tables:
            page = t["page"]
            tables_by_page[page] = tables_by_page.get(page, 0) + 1

        return {
            "total_tables": len(all_tables),
            "pages_with_tables": len(pages_with_tables),
            "avg_confidence": avg_confidence,
            "tables_by_page": tables_by_page,
            "high_quality_count": high_quality,
        }

    def export_to_markdown(self, output_path: Optional[str] = None) -> str:
        """
        모든 테이블을 Markdown 형식으로 내보내기

        Args:
            output_path: 출력 파일 경로 (None이면 문자열만 반환)

        Returns:
            Markdown 문자열
        """
        all_tables = self.get_all_tables()

        md_lines = ["# Extracted Tables\n"]

        for t in all_tables:
            md_lines.append(f"\n## Page {t['page'] + 1}, Table {t['table_index'] + 1}")
            md_lines.append(f"- Rows: {t['rows']}, Cols: {t['cols']}")
            md_lines.append(f"- Confidence: {t['confidence']:.2f}")
            md_lines.append(f"- Source: {t['source']}\n")

            # 실제 테이블 데이터는 메타데이터에 없으므로 요약만
            if t["has_markdown"]:
                md_lines.append("*(Markdown data available)*\n")
            else:
                md_lines.append("*(No Markdown data)*\n")

        markdown_text = "\n".join(md_lines)

        if output_path:
            Path(output_path).write_text(markdown_text, encoding="utf-8")

        return markdown_text
