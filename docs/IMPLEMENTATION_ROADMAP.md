# beanllm 고급 기능 구현 로드맵

**작성일**: 2025-12-30
**상태**: Phase 1 완료, Phase 2-4 계획 중
**전체 예상 기간**: 6-8주

---

## 📋 전체 구조

```
beanllm 고급 기능
├── Phase 1: beanPDFLoader ✅ DONE (Week 1-2)
├── Phase 2: Markdown & Layout ⏳ In Progress (Week 3)
├── Phase 3: ML Layer (Week 4)
├── Phase 4: OCR Module (Week 5-6)
└── Phase 5: Visualization (Week 7-8)
```

---

## ✅ Phase 1: beanPDFLoader 핵심 (완료)

**기간**: Week 1-2 (2025-12-23 ~ 2025-12-30)
**상태**: ✅ 100% 완료

### 완료된 기능

1. **3-Layer Architecture**
   - ✅ BasePDFEngine 추상 클래스
   - ✅ PyMuPDFEngine (Fast Layer) - 335 lines
   - ✅ PDFPlumberEngine (Accurate Layer) - 421 lines
   - ✅ beanPDFLoader 메인 로더 - 374 lines

2. **데이터 모델**
   - ✅ PageData, TableData, ImageData
   - ✅ PDFLoadConfig, PDFLoadResult
   - ✅ 5개 모델 완성

3. **핵심 기능**
   - ✅ 자동 전략 선택 (테이블/이미지/페이지수 기반)
   - ✅ 테이블 추출 (DataFrame/Markdown/CSV 변환)
   - ✅ 이미지 추출 (bbox 자동 추출)
   - ✅ 신뢰도 계산
   - ✅ Factory 자동 감지 통합

4. **메타데이터 구조화**
   - ✅ TableExtractor - 테이블 메타데이터 조회
   - ✅ ImageExtractor - 이미지 메타데이터 조회
   - ✅ 필터링, 요약, 내보내기 기능

5. **테스트**
   - ✅ 70개 단위 테스트 (100% 통과)
   - ✅ 테스트 픽스처 (3개 PDF 파일)

### 성과
- **코드**: ~2,600 lines
- **테스트**: 70 tests, 100% pass
- **문서**: README 업데이트, 사용 예제 추가

---

## 🔄 Phase 2: Markdown & Layout Analysis

**기간**: Week 3 (2025-12-31 ~ 2026-01-06)
**예상 시간**: 10시간
**문서**: `docs/BEANPDF_REMAINING_FEATURES.md`

### TODO 목록

#### TODO-201: Markdown 변환 기능 (P0)
- [ ] MarkdownConverter 클래스 구현
- [ ] 제목 레벨 자동 감지 (폰트 크기 기반)
- [ ] 테이블 → Markdown 테이블 변환
- [ ] 이미지 → ![image](path) 링크
- [ ] 페이지 구분자 삽입
- [ ] beanPDFLoader 통합 (`to_markdown=True`)
- [ ] 단위 테스트 (10개)

**예상 시간**: 4시간

#### TODO-202: Layout Analysis 완전 구현 (P1)
- [ ] LayoutAnalyzer 클래스 구현
- [ ] 블록 감지 (제목, 본문, 표, 이미지)
- [ ] Reading order 복원
- [ ] 다단 레이아웃 처리
- [ ] 헤더/푸터 제거
- [ ] PyMuPDFEngine/PDFPlumberEngine 통합
- [ ] 단위 테스트 (12개)

**예상 시간**: 6시간

### 완료 기준
- ✅ `to_markdown=True` 옵션 작동
- ✅ 복잡한 레이아웃 문서 정확히 파싱
- ✅ 22개 테스트 통과

---

## 🤖 Phase 3: ML Layer (marker-pdf)

**기간**: Week 4 (2026-01-07 ~ 2026-01-13)
**예상 시간**: 12시간
**문서**: `docs/BEANPDF_REMAINING_FEATURES.md`

### TODO 목록

#### TODO-301: MarkerEngine 기본 구현 (P2)
- [ ] MarkerEngine 클래스 구현
- [ ] marker-pdf 라이브러리 통합
- [ ] GPU/CPU 모드 지원
- [ ] PDFLoadResult 형식 변환
- [ ] 의존성 추가 (`pip install marker-pdf`)
- [ ] 단위 테스트 (8개)

**예상 시간**: 8시간

#### TODO-302: marker-pdf 통합 및 최적화 (P2)
- [ ] 배치 처리 지원
- [ ] GPU 메모리 관리
- [ ] 캐싱 메커니즘
- [ ] 대용량 PDF 처리
- [ ] 성능 벤치마크

**예상 시간**: 4시간

### 완료 기준
- ✅ ML Layer 전략 작동
- ✅ 98% 정확도 달성
- ✅ GPU 모드 10초/100페이지

---

## 📸 Phase 4: OCR Module

**기간**: Week 5-6 (2026-01-14 ~ 2026-01-27)
**예상 시간**: 60시간
**문서**: `docs/OCR_MODULE_PLAN.md`

### Week 5: 핵심 구조 & PaddleOCR

#### TODO-OCR-101: 기본 인터페이스 및 모델 (4h)
- [ ] OCRResult, OCRConfig 모델
- [ ] beanOCR 메인 클래스
- [ ] 컴포넌트 초기화

#### TODO-OCR-102: beanOCR 메인 클래스 (6h)
- [ ] recognize() 메서드
- [ ] recognize_pdf_page() 메서드
- [ ] batch_recognize() 메서드

#### TODO-OCR-201: PaddleOCR 엔진 (8h)
- [ ] PaddleOCREngine 클래스
- [ ] 다국어 모델 초기화
- [ ] 결과 변환 로직
- [ ] 다국어 최적화 (한글, 중국어, 일본어)
- [ ] 단위 테스트 (15개)

**Week 5 Total**: 20시간

### Week 6: 대체 엔진 & 전후처리

#### TODO-OCR-202: 대체 엔진 구현 (10h)
- [ ] EasyOCR 엔진 (2h)
- [ ] TrOCR 엔진 - 손글씨 (3h)
- [ ] Nougat 엔진 - 학술 논문 (3h)
- [ ] Tesseract 엔진 - Fallback (2h)

#### TODO-OCR-301: 이미지 전처리 파이프라인 (6h)
- [ ] ImagePreprocessor 클래스
- [ ] 노이즈 제거
- [ ] 대비 조정 (CLAHE)
- [ ] 회전 보정
- [ ] 이진화

#### TODO-OCR-302: LLM 후처리 (8h)
- [ ] LLMPostprocessor 클래스
- [ ] 오타 수정
- [ ] 문맥 기반 보정
- [ ] 맞춤법 검사

#### TODO-OCR-401: Hybrid OCR 전략 (4h)
- [ ] Local + Cloud Hybrid 구현
- [ ] 신뢰도 기반 자동 선택
- [ ] 비용 최적화 (95% 절감)

#### TODO-OCR-402: beanPDFLoader OCR 통합 (6h)
- [ ] OCRProcessor 구현
- [ ] 스캔 페이지 자동 감지
- [ ] PyMuPDFEngine/PDFPlumberEngine 통합
- [ ] enable_ocr=True 옵션

**Week 6 Total**: 34시간

### 완료 기준
- ✅ 7개 OCR 엔진 작동
- ✅ 90-96% 정확도 (일반 문서)
- ✅ 98%+ 정확도 (LLM 후처리)
- ✅ 한글 95%+ 정확도
- ✅ 80개 테스트 통과

---

## 🎨 Phase 5: Visualization

**기간**: Week 7-8 (2026-01-28 ~ 2026-02-10)
**예상 시간**: 28시간
**문서**: `docs/VISUALIZATION_PLAN.md`

### Week 7: Zero Configuration & 렌더링

#### TODO-VIZ-101: Document Visualizer (6h)
- [ ] DocumentVisualizer 클래스
- [ ] Jupyter 렌더링
- [ ] 터미널 출력 (Rich)
- [ ] show(), show_page(), show_tables()

#### TODO-VIZ-102: One-liner Helpers (4h)
- [ ] quick_preview()
- [ ] preview_tables()
- [ ] preview_images()
- [ ] compare_strategies()

#### TODO-VIZ-201: PDF 페이지 렌더링 (6h)
- [ ] PDFPageRenderer 클래스
- [ ] 고해상도 렌더링 (150 DPI)
- [ ] 그리드 표시
- [ ] 파일 저장

**Week 7 Total**: 16시간

### Week 8: Dashboard & RAG 확장

#### TODO-VIZ-301: Streamlit Dashboard (8h)
- [ ] 파일 업로드 UI
- [ ] 옵션 선택 (strategy, extract_tables, etc.)
- [ ] 탭 기반 결과 표시 (Pages, Tables, Images, Stats)
- [ ] 실시간 분석

#### TODO-VIZ-401: RAGDebugger 확장 (4h)
- [ ] visualize_document_chunks()
- [ ] compare_extraction_methods()
- [ ] PDF 특화 디버깅 기능

**Week 8 Total**: 12시간

### 완료 기준
- ✅ 3줄 이내 코드로 시각화
- ✅ Jupyter 자동 렌더링
- ✅ Dashboard 5초 내 로딩
- ✅ RAGDebugger PDF 지원

---

## 📊 전체 통계 요약

### 개발 규모
| Phase | Lines of Code | Tests | Hours |
|-------|---------------|-------|-------|
| Phase 1 ✅ | 2,600 | 70 | 40h |
| Phase 2 | 800 | 22 | 10h |
| Phase 3 | 600 | 12 | 12h |
| Phase 4 | 3,000 | 80 | 60h |
| Phase 5 | 1,500 | 30 | 28h |
| **Total** | **8,500** | **214** | **150h** |

### 일정 요약
- **Week 1-2**: Phase 1 (beanPDFLoader 핵심) ✅ DONE
- **Week 3**: Phase 2 (Markdown & Layout)
- **Week 4**: Phase 3 (ML Layer)
- **Week 5-6**: Phase 4 (OCR Module)
- **Week 7-8**: Phase 5 (Visualization)

**Total**: 8주 (2개월)

---

## 🎯 성능 목표

### beanPDFLoader
- ✅ Fast Layer: ~2초/100페이지
- ✅ Accurate Layer: ~15초/100페이지
- 🔄 ML Layer: ~10초/100페이지 (GPU)
- ✅ 테이블 추출: 95% 정확도
- ✅ 이미지 추출: bbox 자동 추출

### OCR Module
- 🎯 정확도 (일반): 90-96%
- 🎯 정확도 (LLM 후처리): 98%+
- 🎯 한글 정확도: 95%+
- 🎯 처리 속도: ~1초/페이지 (GPU)
- 🎯 비용 절감: 95% (Hybrid)

### Visualization
- 🎯 렌더링 속도: <1초/페이지
- 🎯 Dashboard 로딩: <5초
- 🎯 사용성: 3줄 이내 코드

---

## 📦 의존성 요약

```toml
# pyproject.toml
[project.dependencies]
# 기존 의존성...
"PyMuPDF>=1.23.0",
"pdfplumber>=0.10.0",
"pandas>=2.0.0",

[project.optional-dependencies]
# ML Layer
ml = [
    "marker-pdf>=0.2.0",
    "torch>=2.0.0",
]

# OCR
ocr = [
    "paddleocr>=2.7.0",
    "easyocr>=1.7.0",
    "opencv-python>=4.8.0",
    "pillow>=10.0.0",
]

ocr-full = [
    "paddleocr>=2.7.0",
    "easyocr>=1.7.0",
    "transformers>=4.35.0",
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "opencv-python>=4.8.0",
    "pillow>=10.0.0",
    "pytesseract>=0.3.10",
    "surya-ocr>=0.4.0",
]

# Visualization
visualization = [
    "pillow>=10.0.0",
    "matplotlib>=3.7.0",
    "rich>=13.0.0",
]

dashboard = [
    "streamlit>=1.28.0",
    "plotly>=5.17.0",
]

# All
all-advanced = [
    "marker-pdf>=0.2.0",
    "paddleocr>=2.7.0",
    "streamlit>=1.28.0",
    # ...
]
```

---

## 🚀 다음 단계

**즉시 시작 (Week 3)**:
1. TODO-201: Markdown 변환 구현
2. TODO-202: Layout Analysis 구현

**준비 사항**:
- marker-pdf 라이브러리 조사
- PaddleOCR 모델 다운로드
- Streamlit 프로토타입 테스트

---

## 📚 관련 문서

1. **`BEANPDF_REMAINING_FEATURES.md`** - beanPDFLoader 미구현 기능
2. **`OCR_MODULE_PLAN.md`** - OCR 모듈 상세 계획
3. **`VISUALIZATION_PLAN.md`** - 시각화 기능 계획

---

**마지막 업데이트**: 2025-12-30
**작성자**: AI Assistant
**상태**: Phase 1 완료, Phase 2-5 계획 완료
