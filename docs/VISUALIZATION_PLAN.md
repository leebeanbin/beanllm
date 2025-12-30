# 문서 시각화 기능 구현 계획

**작성일**: 2025-12-30
**상태**: 계획 단계
**예상 기간**: 1-2주

---

## 🎯 목표

문서 처리 결과를 쉽게 시각화하여 디버깅 및 품질 확인 지원

**핵심 가치**:
- Zero Configuration - 설정 없이 바로 사용
- One-liner - 한 줄로 시각화
- Progressive Disclosure - 간단 → 고급
- 기존 RAG 도구 확장

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────┐
│      Document Visualizer (Facade)       │
│  - PDF 페이지 미리보기                   │
│  - 테이블 시각화                         │
│  - 이미지 표시                           │
│  - 레이아웃 분석 결과                    │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│     Existing RAG Debugging Tools         │
│  - RAGDebugger (확장)                    │
│  - RAGPipelineVisualizer (확장)          │
│  - RAGEvaluationDashboard (확장)         │
└──────────────────────────────────────────┘
```

---

## 📦 Phase 1: Zero Configuration API (Week 1)

### TODO-VIZ-101: 기본 Document Visualizer

**예상 시간**: 6시간

```python
# src/beanllm/utils/visualization/document_visualizer.py
class DocumentVisualizer:
    """
    문서 시각화 (Zero Configuration)

    Example:
        ```python
        from beanllm.domain.loaders import beanPDFLoader
        from beanllm.utils.visualization import DocumentVisualizer

        # PDF 로딩
        loader = beanPDFLoader("document.pdf", extract_tables=True)
        docs = loader.load()

        # 시각화 (자동 표시)
        viz = DocumentVisualizer(docs)
        viz.show()  # Jupyter에서 자동 렌더링

        # 특정 페이지만
        viz.show_page(0)

        # 테이블만
        viz.show_tables()
        ```
    """

    def __init__(self, documents: List[Document]):
        self.documents = documents
        self._check_environment()

    def _check_environment(self):
        """실행 환경 감지 (Jupyter, CLI, etc.)"""
        try:
            from IPython import get_ipython
            self.is_jupyter = get_ipython() is not None
        except:
            self.is_jupyter = False

    def show(self, max_pages: int = 5):
        """전체 문서 시각화"""
        if self.is_jupyter:
            self._show_in_jupyter(max_pages)
        else:
            self._show_in_terminal(max_pages)

    def _show_in_jupyter(self, max_pages):
        """Jupyter Notebook에서 렌더링"""
        from IPython.display import display, HTML

        for i, doc in enumerate(self.documents[:max_pages]):
            # 페이지 제목
            html = f"<h3>Page {doc.metadata.get('page', i) + 1}</h3>"

            # 텍스트 미리보기
            preview = doc.content[:500] + "..." if len(doc.content) > 500 else doc.content
            html += f"<pre style='background:#f5f5f5;padding:10px;'>{preview}</pre>"

            # 메타데이터
            html += "<h4>Metadata</h4>"
            html += "<ul>"
            for key, value in doc.metadata.items():
                if key not in ["content"]:
                    html += f"<li><strong>{key}</strong>: {value}</li>"
            html += "</ul>"

            # 테이블 (있으면)
            if "tables" in doc.metadata:
                html += self._render_tables_html(doc.metadata["tables"])

            display(HTML(html))

    def _show_in_terminal(self, max_pages):
        """터미널에서 출력"""
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()

        for i, doc in enumerate(self.documents[:max_pages]):
            # 페이지 패널
            page_num = doc.metadata.get('page', i) + 1
            console.print(Panel(
                f"[bold]Page {page_num}[/bold]",
                style="blue"
            ))

            # 텍스트 미리보기
            preview = doc.content[:300] + "..." if len(doc.content) > 300 else doc.content
            console.print(preview)
            console.print()

            # 메타데이터 테이블
            if doc.metadata:
                meta_table = Table(title="Metadata")
                meta_table.add_column("Key", style="cyan")
                meta_table.add_column("Value", style="green")

                for key, value in doc.metadata.items():
                    if key not in ["content", "tables", "images"]:
                        meta_table.add_row(key, str(value))

                console.print(meta_table)
                console.print()

    def show_page(self, page_num: int):
        """특정 페이지만 표시"""
        page_docs = [d for d in self.documents if d.metadata.get("page") == page_num]
        if page_docs:
            temp_viz = DocumentVisualizer(page_docs)
            temp_viz.show()
        else:
            print(f"Page {page_num} not found")

    def show_tables(self):
        """모든 테이블 시각화"""
        from .extractors import TableExtractor

        extractor = TableExtractor(self.documents)
        tables = extractor.get_all_tables()

        if self.is_jupyter:
            self._show_tables_jupyter(tables)
        else:
            self._show_tables_terminal(tables)

    def _show_tables_jupyter(self, tables):
        """Jupyter에서 테이블 렌더링"""
        from IPython.display import display, HTML
        import pandas as pd

        for table in tables:
            html = f"<h4>Page {table['page'] + 1}, Table {table['table_index'] + 1}</h4>"
            html += f"<p>Rows: {table['rows']}, Cols: {table['cols']}, Confidence: {table['confidence']:.2f}</p>"

            # DataFrame이 있으면 표시
            if table.get("has_dataframe"):
                # 실제 DataFrame은 원본 Document에서 가져와야 함
                html += "<p><em>(DataFrame available)</em></p>"

            display(HTML(html))
```

---

### TODO-VIZ-102: One-liner Helper Functions

**예상 시간**: 4시간

```python
# src/beanllm/utils/visualization/helpers.py
"""
One-liner 시각화 함수들

매우 간단한 사용을 위한 helper functions
"""

def quick_preview(pdf_path: str, page: int = 0):
    """
    PDF 빠른 미리보기 (One-liner)

    Example:
        >>> from beanllm.utils.visualization import quick_preview
        >>> quick_preview("document.pdf", page=0)
    """
    from ...domain.loaders import beanPDFLoader
    from .document_visualizer import DocumentVisualizer

    loader = beanPDFLoader(pdf_path)
    docs = loader.load()

    viz = DocumentVisualizer(docs)
    viz.show_page(page)


def preview_tables(pdf_path: str):
    """
    PDF 테이블 빠른 미리보기

    Example:
        >>> from beanllm.utils.visualization import preview_tables
        >>> preview_tables("report.pdf")
    """
    from ...domain.loaders import beanPDFLoader
    from .document_visualizer import DocumentVisualizer

    loader = beanPDFLoader(pdf_path, extract_tables=True)
    docs = loader.load()

    viz = DocumentVisualizer(docs)
    viz.show_tables()


def preview_images(pdf_path: str):
    """
    PDF 이미지 빠른 미리보기

    Example:
        >>> from beanllm.utils.visualization import preview_images
        >>> preview_images("images.pdf")
    """
    from ...domain.loaders import beanPDFLoader
    from .extractors import ImageExtractor

    loader = beanPDFLoader(pdf_path, extract_images=True, strategy="fast")
    docs = loader.load()

    extractor = ImageExtractor(docs)
    images = extractor.get_all_images()

    # 이미지 요약 표시
    summary = extractor.get_summary()
    print(f"Total images: {summary['total_images']}")
    print(f"Formats: {summary['formats']}")
    print(f"Average size: {summary['avg_width']}x{summary['avg_height']}px")


def compare_strategies(pdf_path: str, page: int = 0):
    """
    Fast vs Accurate 전략 비교

    Example:
        >>> from beanllm.utils.visualization import compare_strategies
        >>> compare_strategies("document.pdf", page=0)
    """
    from ...domain.loaders import beanPDFLoader
    import time

    # Fast Layer
    start = time.time()
    loader_fast = beanPDFLoader(pdf_path, strategy="fast")
    docs_fast = loader_fast.load()
    time_fast = time.time() - start

    # Accurate Layer
    start = time.time()
    loader_accurate = beanPDFLoader(pdf_path, strategy="accurate")
    docs_accurate = loader_accurate.load()
    time_accurate = time.time() - start

    # 비교 출력
    print("=== Strategy Comparison ===")
    print(f"\nFast Layer (PyMuPDF):")
    print(f"  Time: {time_fast:.2f}s")
    print(f"  Text length: {len(docs_fast[page].content)} chars")

    print(f"\nAccurate Layer (pdfplumber):")
    print(f"  Time: {time_accurate:.2f}s")
    print(f"  Text length: {len(docs_accurate[page].content)} chars")
    print(f"  Speed ratio: {time_accurate / time_fast:.1f}x slower")
```

---

## 🎨 Phase 2: PDF 페이지 렌더링 (Week 1)

### TODO-VIZ-201: PDF 페이지 이미지 렌더링

**예상 시간**: 6시간

```python
# src/beanllm/utils/visualization/pdf_renderer.py
class PDFPageRenderer:
    """
    PDF 페이지를 이미지로 렌더링

    Example:
        ```python
        renderer = PDFPageRenderer("document.pdf")

        # Jupyter에서 표시
        renderer.show_page(0)

        # 파일로 저장
        renderer.save_page(0, "page_0.png")

        # 여러 페이지 그리드
        renderer.show_grid([0, 1, 2, 3], cols=2)
        ```
    """

    def __init__(self, pdf_path: str, dpi: int = 150):
        self.pdf_path = Path(pdf_path)
        self.dpi = dpi
        self._check_dependencies()

    def _check_dependencies(self):
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF is required for rendering")

    def render_page(self, page_num: int) -> "PIL.Image":
        """페이지를 PIL Image로 렌더링"""
        import fitz
        from PIL import Image

        doc = fitz.open(self.pdf_path)
        page = doc[page_num]

        # 고해상도 렌더링
        mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
        pix = page.get_pixmap(matrix=mat)

        # PIL Image 변환
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()

        return img

    def show_page(self, page_num: int):
        """Jupyter에서 페이지 표시"""
        img = self.render_page(page_num)

        try:
            from IPython.display import display
            display(img)
        except:
            # Jupyter가 아니면 파일로 저장 후 안내
            temp_path = f"/tmp/page_{page_num}.png"
            img.save(temp_path)
            print(f"Saved to: {temp_path}")

    def save_page(self, page_num: int, output_path: str):
        """페이지를 파일로 저장"""
        img = self.render_page(page_num)
        img.save(output_path)

    def show_grid(self, page_nums: List[int], cols: int = 3):
        """여러 페이지를 그리드로 표시"""
        from PIL import Image
        import math

        images = [self.render_page(p) for p in page_nums]

        # 그리드 크기 계산
        rows = math.ceil(len(images) / cols)

        # 각 이미지 크기 조정 (균일하게)
        target_width = 300
        resized = []
        for img in images:
            ratio = target_width / img.width
            new_height = int(img.height * ratio)
            resized.append(img.resize((target_width, new_height)))

        # 그리드 이미지 생성
        grid_width = target_width * cols
        grid_height = max(img.height for img in resized) * rows

        grid = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))

        for i, img in enumerate(resized):
            row = i // cols
            col = i % cols
            x = col * target_width
            y = row * max(img.height for img in resized)
            grid.paste(img, (x, y))

        # 표시
        try:
            from IPython.display import display
            display(grid)
        except:
            grid.save("/tmp/grid.png")
            print("Saved grid to: /tmp/grid.png")
```

---

## 📊 Phase 3: Interactive Dashboard (Week 2)

### TODO-VIZ-301: Streamlit Dashboard

**예상 시간**: 8시간

```python
# src/beanllm/utils/visualization/streamlit_dashboard.py
"""
Streamlit 기반 문서 분석 대시보드

실행:
    streamlit run streamlit_dashboard.py
"""

import streamlit as st
from beanllm.domain.loaders import beanPDFLoader
from beanllm.domain.loaders.pdf.extractors import TableExtractor, ImageExtractor


def main():
    st.set_page_config(page_title="PDF Analysis Dashboard", layout="wide")

    st.title("📄 PDF Analysis Dashboard")

    # 파일 업로드
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        # 옵션
        col1, col2, col3 = st.columns(3)
        with col1:
            strategy = st.selectbox("Strategy", ["auto", "fast", "accurate"])
        with col2:
            extract_tables = st.checkbox("Extract Tables", value=True)
        with col3:
            extract_images = st.checkbox("Extract Images", value=False)

        # PDF 로딩
        if st.button("Analyze PDF"):
            with st.spinner("Analyzing..."):
                # 임시 파일 저장
                temp_path = f"/tmp/{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # beanPDFLoader 실행
                loader = beanPDFLoader(
                    temp_path,
                    strategy=strategy,
                    extract_tables=extract_tables,
                    extract_images=extract_images,
                )
                docs = loader.load()

                # 결과 표시
                st.success(f"✅ Loaded {len(docs)} pages")

                # 탭으로 분리
                tabs = st.tabs(["📄 Pages", "📊 Tables", "🖼️ Images", "📈 Stats"])

                with tabs[0]:
                    # 페이지 표시
                    page_num = st.selectbox("Select Page", range(len(docs)))
                    st.subheader(f"Page {page_num + 1}")
                    st.text_area("Content", docs[page_num].content, height=400)
                    st.json(docs[page_num].metadata)

                with tabs[1]:
                    # 테이블 표시
                    if extract_tables:
                        extractor = TableExtractor(docs)
                        tables = extractor.get_all_tables()
                        summary = extractor.get_summary()

                        st.metric("Total Tables", summary["total_tables"])
                        st.metric("Avg Confidence", f"{summary['avg_confidence']:.2f}")

                        for table in tables:
                            st.write(f"**Page {table['page'] + 1}, Table {table['table_index'] + 1}**")
                            st.write(f"Size: {table['rows']}x{table['cols']}, Confidence: {table['confidence']:.2f}")

                with tabs[2]:
                    # 이미지 표시
                    if extract_images:
                        extractor = ImageExtractor(docs)
                        images = extractor.get_all_images()
                        summary = extractor.get_summary()

                        st.metric("Total Images", summary["total_images"])
                        st.json(summary["formats"])

                        for img in images:
                            st.write(f"**Page {img['page'] + 1}, Image {img['image_index'] + 1}**")
                            st.write(f"Format: {img['format']}, Size: {img['width']}x{img['height']}px")

                with tabs[3]:
                    # 통계
                    st.subheader("Document Statistics")
                    st.metric("Total Pages", len(docs))
                    st.metric("Total Characters", sum(len(doc.content) for doc in docs))
                    st.metric("Engine", docs[0].metadata.get("engine", "unknown"))
                    st.metric("Strategy", docs[0].metadata.get("strategy", "unknown"))


if __name__ == "__main__":
    main()
```

---

## 🔧 Phase 4: RAG Debugging Tools 확장 (Week 2)

### TODO-VIZ-401: RAGDebugger 확장

**예상 시간**: 4시간

```python
# src/beanllm/utils/rag_debug/debugger.py 확장
class RAGDebugger:
    # ... 기존 코드 ...

    def visualize_document_chunks(self, documents: List[Document]):
        """
        문서 청크 시각화 (신규)

        Example:
            >>> debugger = RAGDebugger()
            >>> debugger.visualize_document_chunks(chunks)
        """
        from rich.console import Console
        from rich.table import Table

        console = Console()

        table = Table(title="Document Chunks")
        table.add_column("Index", style="cyan")
        table.add_column("Source", style="green")
        table.add_column("Page", style="yellow")
        table.add_column("Length", style="magenta")
        table.add_column("Preview", style="white")

        for i, doc in enumerate(documents[:20]):  # 최대 20개
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", -1)
            length = len(doc.content)
            preview = doc.content[:50] + "..." if len(doc.content) > 50 else doc.content

            table.add_row(
                str(i),
                source,
                str(page),
                str(length),
                preview
            )

        console.print(table)

    def compare_extraction_methods(self, pdf_path: str):
        """
        추출 방법 비교 (신규)

        PDFLoader vs beanPDFLoader 비교
        """
        from ..loaders import PDFLoader
        from ..loaders.pdf import beanPDFLoader
        import time

        # 기존 PDFLoader
        start = time.time()
        old_loader = PDFLoader(pdf_path)
        old_docs = old_loader.load()
        old_time = time.time() - start

        # beanPDFLoader
        start = time.time()
        new_loader = beanPDFLoader(pdf_path, extract_tables=True)
        new_docs = new_loader.load()
        new_time = time.time() - start

        # 비교 출력
        print("=== Extraction Method Comparison ===")
        print(f"\nPDFLoader (Basic):")
        print(f"  Time: {old_time:.2f}s")
        print(f"  Pages: {len(old_docs)}")
        print(f"  Total chars: {sum(len(d.content) for d in old_docs)}")

        print(f"\nbeanPDFLoader (Advanced):")
        print(f"  Time: {new_time:.2f}s")
        print(f"  Pages: {len(new_docs)}")
        print(f"  Total chars: {sum(len(d.content) for d in new_docs)}")
        print(f"  Tables extracted: {sum(1 for d in new_docs if 'tables' in d.metadata)}")
```

---

## 📦 의존성

```toml
# pyproject.toml
[project.optional-dependencies]
visualization = [
    "pillow>=10.0.0",
    "matplotlib>=3.7.0",
    "rich>=13.0.0",  # 이미 있음
]

dashboard = [
    "streamlit>=1.28.0",
    "plotly>=5.17.0",
]
```

---

## 🗓️ 구현 일정

| Week | Task | Hours |
|------|------|-------|
| Week 1 | Phase 1-2 (Zero Config + 렌더링) | 16h |
| Week 2 | Phase 3-4 (Dashboard + RAG 확장) | 12h |

**Total**: ~28 hours (1-2주)

---

## 🎯 성능 목표

- Zero Configuration: 3줄 이내 코드로 시각화
- 렌더링 속도: <1초/페이지
- Jupyter 통합: 자동 렌더링
- Dashboard 로딩: <5초
