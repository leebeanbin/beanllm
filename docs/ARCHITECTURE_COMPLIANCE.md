# beanPDFLoader 아키텍처 준수 가이드

## 📋 기존 아키텍처 패턴

### 1. BaseDocumentLoader 상속 필수

```python
from ..base import BaseDocumentLoader
from ..types import Document

class beanPDFLoader(BaseDocumentLoader):
    def load(self) -> List[Document]:
        """List[Document] 반환 필수"""
        pass
    
    def lazy_load(self):
        """제너레이터 반환"""
        yield from self.load()
```

### 2. Document 타입 사용

```python
Document(
    content: str,  # 텍스트 내용
    metadata: Dict[str, Any]  # 메타데이터
)
```

### 3. 로거 패턴

```python
try:
    from ...utils.logger import get_logger
except ImportError:
    import logging
    def get_logger(name: str):
        return logging.getLogger(name)

logger = get_logger(__name__)
```

### 4. 에러 처리

```python
try:
    import library
except ImportError:
    raise ImportError("library is required. Install it with: pip install library")
```

## 🔄 beanPDFLoader 설계 방향

### 구조

```
beanPDFLoader (BaseDocumentLoader 상속)
    ├── load() -> List[Document]  # 기존 패턴 준수
    ├── lazy_load() -> Generator[Document]  # 기존 패턴 준수
    └── 내부 구현
        ├── BasePDFEngine (내부 엔진 추상 클래스)
        ├── PyMuPDFEngine (Fast Layer)
        ├── PDFPlumberEngine (Accurate Layer)
        └── 내부 모델 (PageData, TableData 등)
            └── 최종적으로 Document로 변환
```

### 변환 로직

```python
# 내부 엔진 결과 (PageData)
page_data = PageData(page=0, text="...", ...)

# Document로 변환
document = Document(
    content=page_data.text,
    metadata={
        "source": str(pdf_path),
        "page": page_data.page,
        "width": page_data.width,
        "height": page_data.height,
        **page_data.metadata
    }
)
```

### 테이블 처리

```python
# 테이블은 metadata에 포함
document = Document(
    content=page_data.text,
    metadata={
        "source": str(pdf_path),
        "page": page_data.page,
        "tables": [table.to_dict() for table in tables],  # 테이블 정보
    }
)
```

### 이미지 처리

```python
# 이미지는 metadata에 경로/정보만 포함 (실제 이미지 데이터는 별도 저장)
document = Document(
    content=page_data.text,
    metadata={
        "source": str(pdf_path),
        "page": page_data.page,
        "images": [image.to_dict() for image in images],  # 이미지 메타데이터
    }
)
```

## ✅ 체크리스트

- [x] BaseDocumentLoader 상속
- [x] load() -> List[Document] 반환
- [x] lazy_load() 제너레이터 구현
- [x] Document 타입 사용
- [x] 로거 패턴 준수
- [x] 에러 처리 패턴 준수
- [x] 기존 PDFLoader와 호환 (같은 인터페이스)

