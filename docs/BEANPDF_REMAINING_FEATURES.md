# beanPDFLoader 미구현 기능 구현 계획

**작성일**: 2025-12-30
**상태**: Phase 1 완료 (Fast/Accurate Layer), Phase 2-4 계획

---

## 📋 Phase 1 완료 현황

### ✅ 완료된 기능 (2025-12-30)

1. **3-Layer Architecture 기반 구조**
   - BasePDFEngine 추상 클래스
   - PyMuPDFEngine (Fast Layer) - 335 lines
   - PDFPlumberEngine (Accurate Layer) - 421 lines
   - beanPDFLoader 메인 로더 - 374 lines

2. **데이터 모델**
   - PageData, TableData, ImageData
   - PDFLoadConfig, PDFLoadResult
   - 5개 모델 완성

3. **핵심 기능**
   - 자동 전략 선택 (테이블/이미지/페이지수 기반)
   - 테이블 추출 (DataFrame/Markdown/CSV 변환)
   - 이미지 추출 (bbox 자동 추출)
   - 신뢰도 계산
   - Factory 자동 감지 통합

4. **메타데이터 구조화**
   - TableExtractor - 테이블 메타데이터 조회
   - ImageExtractor - 이미지 메타데이터 조회
   - 필터링, 요약, 내보내기 기능

5. **테스트**
   - 70개 단위 테스트 (100% 통과)
   - 테스트 픽스처 (3개 PDF 파일)

---

## 🎯 Phase 2: Markdown 변환 & Layout Analysis

### TODO-201: Markdown 변환 기능 구현

**우선순위**: P0 (높음)
**예상 시간**: 4시간
**의존성**: Phase 1 완료

**구현 내용**:

```python
# src/beanllm/domain/loaders/pdf/utils/markdown_converter.py
class MarkdownConverter:
    """
    PDF 추출 결과를 Markdown으로 변환

    Features:
    - 텍스트 → Markdown 변환
    - 제목 레벨 자동 감지 (폰트 크기 기반)
    - 테이블 → Markdown 테이블
    - 이미지 → ![image](path) 링크
    - 페이지 구분자 삽입
    """

    def convert_to_markdown(self, result: PDFLoadResult) -> str:
        """PDF 결과를 Markdown으로 변환"""
        pass

    def _detect_headings(self, page: PageData) -> List[dict]:
        """폰트 크기 기반 제목 감지"""
        pass

    def _convert_table_to_markdown(self, table: TableData) -> str:
        """테이블 → Markdown 테이블"""
        pass
```

**사용 예제**:
```python
from beanllm.domain.loaders import beanPDFLoader

loader = beanPDFLoader("document.pdf", to_markdown=True, extract_tables=True)
docs = loader.load()

# docs[0].content가 Markdown 형식
print(docs[0].content)
# # Document Title
#
# ## Section 1
# Content here...
#
# | Header 1 | Header 2 |
# |----------|----------|
# | Data 1   | Data 2   |
```

**테스트 계획**:
- 제목 감지 정확도 테스트
- 테이블 Markdown 변환 테스트
- 복잡한 문서 변환 테스트

---

### TODO-202: Layout Analysis 완전 구현

**우선순위**: P1 (중-높)
**예상 시간**: 6시간
**의존성**: TODO-201

**구현 내용**:

```python
# src/beanllm/domain/loaders/pdf/utils/layout_analyzer.py
class LayoutAnalyzer:
    """
    PDF 레이아웃 분석

    Features:
    - 블록 감지 (제목, 본문, 표, 이미지)
    - Reading order 복원
    - 다단 레이아웃 처리
    - 헤더/푸터 제거
    """

    def analyze_layout(self, page: PageData) -> dict:
        """레이아웃 분석 및 구조 추출"""
        pass

    def detect_blocks(self, page: PageData) -> List[dict]:
        """블록 감지 (제목, 본문, 표, 이미지)"""
        pass

    def restore_reading_order(self, blocks: List[dict]) -> List[dict]:
        """읽기 순서 복원 (왼쪽→오른쪽, 위→아래)"""
        pass

    def detect_multi_column(self, page: PageData) -> bool:
        """다단 레이아웃 감지"""
        pass

    def remove_header_footer(self, blocks: List[dict]) -> List[dict]:
        """헤더/푸터 제거"""
        pass
```

**통합**:
```python
# PyMuPDFEngine 및 PDFPlumberEngine에 통합
if config.get("layout_analysis", False):
    analyzer = LayoutAnalyzer()
    layout_info = analyzer.analyze_layout(page_data)
    page_data["layout"] = layout_info
```

**테스트 계획**:
- 단일 컬럼 문서 테스트
- 다단 레이아웃 문서 테스트
- 헤더/푸터 제거 테스트

---

## 🤖 Phase 3: ML Layer (marker-pdf)

### TODO-301: MarkerEngine 기본 구현

**우선순위**: P2 (중)
**예상 시간**: 8시간
**의존성**: marker-pdf 라이브러리

**구현 내용**:

```python
# src/beanllm/domain/loaders/pdf/engines/marker_engine.py
class MarkerEngine(BasePDFEngine):
    """
    marker-pdf 기반 ML Layer

    Features:
    - 구조 보존 Markdown 변환
    - 98% 정확도
    - ~10초/100 pages (GPU)
    - 복잡한 레이아웃 처리
    """

    def __init__(self, use_gpu: bool = True):
        super().__init__(name="Marker")
        self.use_gpu = use_gpu
        self._check_dependencies()

    def _check_dependencies(self):
        """marker-pdf 라이브러리 확인"""
        try:
            import marker
        except ImportError:
            raise ImportError(
                "marker-pdf is required for MarkerEngine. "
                "Install it with: pip install marker-pdf"
            )

    def extract(self, pdf_path, config) -> dict:
        """marker-pdf로 구조 보존 추출"""
        import marker

        # marker-pdf 실행
        result = marker.convert_pdf(
            pdf_path,
            use_gpu=self.use_gpu,
            # ...
        )

        # PDFLoadResult 형식으로 변환
        return self._convert_marker_result(result)
```

**의존성 추가**:
```toml
# pyproject.toml
[project.optional-dependencies]
ml = [
    "marker-pdf>=0.2.0",  # ML Layer
    "torch>=2.0.0",       # marker-pdf 의존성
]
```

**전략 선택 업데이트**:
```python
# beanPDFLoader._select_strategy()
if self.config.to_markdown and "ml" in self._engines:
    return "ml"  # Markdown 변환 시 ML Layer 우선
```

**테스트 계획**:
- 기본 Markdown 변환 테스트
- 복잡한 레이아웃 문서 테스트
- GPU vs CPU 성능 비교

---

### TODO-302: marker-pdf 통합 및 최적화

**우선순위**: P2 (중)
**예상 시간**: 4시간
**의존성**: TODO-301

**최적화 내용**:
1. 배치 처리 지원
2. GPU 메모리 관리
3. 캐싱 메커니즘
4. 대용량 PDF 처리

---

## 📸 Phase 4: OCR 통합

### TODO-401: OCR 모듈 기본 구조

**우선순위**: P1 (중-높)
**예상 시간**: 10시간
**의존성**: 별도 OCR 모듈 구현 (다음 문서 참조)

**구현 내용**:

```python
# src/beanllm/domain/loaders/pdf/utils/ocr_processor.py
class OCRProcessor:
    """
    PDF용 OCR 처리기

    beanOCR 모듈을 래핑하여 PDF 처리에 최적화
    """

    def __init__(self, engine: str = "paddleocr"):
        from ....ocr import beanOCR  # 별도 OCR 모듈
        self.ocr = beanOCR(engine=engine)

    def process_page(self, page_image, config: dict) -> dict:
        """페이지 이미지 OCR 처리"""
        pass

    def detect_scanned_page(self, page: PageData) -> bool:
        """스캔된 페이지 감지"""
        # 텍스트가 거의 없으면 스캔 문서로 판단
        pass
```

**beanPDFLoader 통합**:
```python
# PyMuPDFEngine/PDFPlumberEngine 수정
if config.get("enable_ocr", False):
    # 텍스트가 거의 없으면 OCR 실행
    if len(text.strip()) < 50:
        ocr_processor = OCRProcessor()
        ocr_result = ocr_processor.process_page(page_image, config)
        text = ocr_result["text"]
        page_data["ocr_applied"] = True
```

**사용 예제**:
```python
# 스캔된 PDF 처리
loader = beanPDFLoader("scanned.pdf", enable_ocr=True)
docs = loader.load()

# OCR이 적용된 페이지 확인
for doc in docs:
    if doc.metadata.get("ocr_applied"):
        print(f"Page {doc.metadata['page']}: OCR applied")
```

---

## 📊 전체 구현 로드맵

### Week 1-2: Phase 1 ✅ DONE
- beanPDFLoader 핵심 구현
- Fast/Accurate Layer
- 메타데이터 구조화

### Week 3: Phase 2
- TODO-201: Markdown 변환 (2일)
- TODO-202: Layout Analysis (3일)

### Week 4: Phase 3
- TODO-301: MarkerEngine 기본 (3일)
- TODO-302: marker-pdf 통합 (2일)

### Week 5: Phase 4 (OCR 모듈 완료 후)
- TODO-401: OCR 통합 (5일)

---

## 🎯 우선순위 요약

**P0 (즉시 구현)**:
- TODO-201: Markdown 변환

**P1 (다음 주)**:
- TODO-202: Layout Analysis
- TODO-401: OCR 통합

**P2 (2주 후)**:
- TODO-301: MarkerEngine
- TODO-302: marker-pdf 최적화

---

## 📝 다음 문서

이 문서 완료 후 다음 계획:
1. **OCR_MODULE_PLAN.md** - OCR 모듈 상세 계획
2. **VISUALIZATION_PLAN.md** - 시각화 기능 계획
3. **OFFICE_INTEGRATION_PLAN.md** - Office 문서 처리 계획
