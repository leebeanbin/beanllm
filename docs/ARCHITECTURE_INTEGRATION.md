# beanPDFLoader 아키텍처 통합 완료 체크리스트

## ✅ 완료된 통합 사항

### 1. BaseDocumentLoader 상속 ✅
- [x] `beanPDFLoader`는 `BaseDocumentLoader` 상속
- [x] `load() -> List[Document]` 구현
- [x] `lazy_load()` 제너레이터 구현

### 2. Document 타입 사용 ✅
- [x] 최종 결과는 `Document` 타입으로 변환
- [x] `content: str` 및 `metadata: Dict[str, Any]` 구조 준수

### 3. 로거 패턴 준수 ✅
- [x] `try/except`로 `get_logger` import
- [x] 실패 시 `logging.getLogger` 사용

### 4. 에러 처리 패턴 준수 ✅
- [x] ImportError 시 명확한 메시지
- [x] Exception 발생 시 로깅 후 raise

### 5. Factory 패턴 통합 ✅
- [x] `DocumentLoader`에 beanPDFLoader 추가
- [x] `loader_type="beanpdf"` 또는 `"bean-pdf"`로 사용 가능
- [x] 선택적 통합 (의존성 없어도 기존 PDFLoader 사용 가능)

### 6. __init__.py 업데이트 ✅
- [x] `src/beanllm/domain/loaders/pdf/__init__.py` 업데이트
- [x] `src/beanllm/domain/loaders/__init__.py` 업데이트
- [x] 선택적 import 처리

## 📋 사용 방법

### 방법 1: 직접 사용 (권장)
```python
from beanllm.domain.loaders.pdf import beanPDFLoader

loader = beanPDFLoader("document.pdf", extract_tables=True)
docs = loader.load()
```

### 방법 2: Factory 패턴 사용
```python
from beanllm.domain.loaders import DocumentLoader

# 고급 PDF 로더 사용
docs = DocumentLoader.load("document.pdf", loader_type="beanpdf", extract_tables=True)

# 기본 PDF 로더 사용 (기존 방식)
docs = DocumentLoader.load("document.pdf")  # PDFLoader 사용
```

### 방법 3: 편의 함수 사용
```python
from beanllm.domain.loaders import load_documents

# 고급 PDF 로더
docs = load_documents("document.pdf", loader_type="beanpdf", extract_tables=True)
```

## 🔄 기존 코드와의 호환성

### 기존 PDFLoader 유지
- 기존 `PDFLoader`는 그대로 유지
- 기본 동작은 변경 없음
- `DocumentLoader.load("file.pdf")`는 여전히 `PDFLoader` 사용

### beanPDFLoader는 선택적
- 의존성 없어도 기존 코드 동작
- 명시적으로 `loader_type="beanpdf"` 지정 시에만 사용

## ⚠️ 주의사항

1. **의존성**: beanPDFLoader 사용 시 `PyMuPDF` 또는 `pdfplumber` 필요
2. **CLI 통합**: 현재 CLI에는 로더 기능이 없음 (필요 시 추가 가능)
3. **기본 동작**: 기본 PDF 로딩은 여전히 `PDFLoader` 사용

## 🚀 향후 개선 사항

- [ ] CLI에 PDF 로딩 명령어 추가 (선택적)
- [ ] 환경 변수로 기본 PDF 로더 선택 가능
- [ ] 자동 Fallback (beanPDFLoader 실패 시 PDFLoader로)



